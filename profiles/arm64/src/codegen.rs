//! UIR → AArch64 asm walker.
//!
//! Per-op emitters land here as Tasks 3-5 progress.

use crate::types::{ParamKind, ParamSlot};
use crate::{Asm, FnSig, LowerError};
use compiler::{NodeId, NodeKind, StdOp, Uir, UirModel};

/// True iff the Linear op's attribute list includes `bias=true`.
fn linear_has_bias(attrs: &[compiler::OpAttr]) -> bool {
    attrs.iter().any(|a| {
        a.name == "bias" && matches!(&a.value, compiler::AttrValue::Symbol(s) if s == "true")
    })
}

/// Walk the entire UIR, returning the combined asm source + per-model FnSigs.
pub fn walk_uir(uir: &Uir) -> Result<Asm, LowerError> {
    // First pass: detect duplicate model names.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for model in &uir.models {
        if !seen.insert(model.name.as_str()) {
            return Err(LowerError::DuplicateModelName {
                name: model.name.clone(),
                span: model.source_span,
            });
        }
    }

    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for model in &uir.models {
        let (model_asm, sig) = walk_model(model)?;
        source.push_str(&model_asm);
        source.push('\n');
        functions.push(sig);
    }

    Ok(Asm { source, functions })
}

fn walk_model(model: &UirModel) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue, LeafKind};
    use crate::buffer::{assign_buffers, compute_callee_saved, compute_is_leaf};

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 2. Compute layout, ABI sizes (kept from Task 1).
    let input_id = *model.inputs.first().ok_or(LowerError::ShapeNotConcrete {
        span: model.source_span,
    })?;
    let input_floats: usize = model.nodes[input_id].ty.shape.0.iter().product::<u64>() as usize;
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op {
            op: StdOp::Linear,
            operands,
            attrs,
        } = &node.kind
        {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete {
                    span: node.source_span,
                });
            }
            let k = in_shape.0[1] as usize;
            let n = out_shape.0[1] as usize;
            params_layout.push(ParamSlot {
                kind: ParamKind::LinearWeight,
                origin_node: node_idx,
                offset: params_floats,
                size: k * n,
            });
            params_floats += k * n;
            if linear_has_bias(attrs) {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        output_floats,
        params_floats,
        params_layout,
    };

    // 3. Buffer assignment + leaf analysis.
    let assignment = assign_buffers(model);
    let leaf = if compute_is_leaf(model) {
        LeafKind::Leaf
    } else {
        LeafKind::NonLeaf
    };
    let regs = compute_callee_saved(model);

    // 4. Emit prologue + body + epilogue.
    let mut body = String::new();
    body.push_str(&format_function_prologue(
        &sig,
        leaf,
        regs,
        assignment.stack_bytes,
    ));

    // Per-op emission (Tasks 5-8 refactor this dispatch into ops/*).
    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op, operands, .. } = &node.kind {
            match op {
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];

                    // Resolve buffer pointers via assignment.
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    // Linear never gets Alias, but resolve defensively.
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    let weight_offset = sig
                        .params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    body.push_str(&emit_matmul_with_locs(
                        b,
                        k,
                        n,
                        linear_idx,
                        src_loc,
                        dst_loc,
                        weight_offset,
                    ));
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    // Relu may be Alias(operand) when intermediate; resolve the chain.
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&emit_relu_with_locs(total, relu_idx, src_loc, dst_loc));
                    relu_idx += 1;
                }
                _ => unreachable!("classify_op should have caught this"),
            }
        }
    }

    body.push_str(&format_function_epilogue(
        leaf,
        regs,
        assignment.stack_bytes,
    ));
    Ok((body, sig))
}

/// Resolve `Alias` chains to a concrete BufferLoc.
fn resolve_loc(locs: &[crate::buffer::BufferLoc], id: NodeId) -> crate::buffer::BufferLoc {
    use crate::buffer::BufferLoc;
    let mut cur = id;
    loop {
        match locs[cur] {
            BufferLoc::Alias(next) => cur = next,
            other => return other,
        }
    }
}

fn emit_matmul_with_locs(
    b: u64,
    k: u64,
    n: u64,
    linear_idx: usize,
    src_loc: crate::buffer::BufferLoc,
    dst_loc: crate::buffer::BufferLoc,
    weight_offset: usize,
) -> String {
    let lid = linear_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]\n"
    ));

    // Materialise src and dst pointers into x_src, x_dst registers.
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    // Weight pointer = x1 (params) + weight_offset*4
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&format!("    mov     x9, #{}\n", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}

fn emit_relu_with_locs(
    total_floats: u64,
    relu_idx: usize,
    src_loc: crate::buffer::BufferLoc,
    dst_loc: crate::buffer::BufferLoc,
) -> String {
    let rid = relu_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; relu: copy-clamp from src to dst ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}

/// Materialise a `BufferLoc` into a GPR (e.g. x11, x12).
fn materialise_ptr(reg: &str, loc: crate::buffer::BufferLoc) -> String {
    use crate::buffer::BufferLoc;
    match loc {
        BufferLoc::InputReg => format!("    mov     {}, x0\n", reg),
        BufferLoc::OutputReg => format!("    mov     {}, x2\n", reg),
        BufferLoc::StackOffset(off) => {
            if off == 0 {
                format!("    mov     {}, sp\n", reg)
            } else if off <= 4095 {
                format!("    add     {}, sp, #{}\n", reg, off)
            } else {
                let lo = (off & 0xFFFF) as u16;
                let hi = ((off >> 16) & 0xFFFF) as u16;
                let mut s = String::new();
                s.push_str(&format!("    movz    w10, #0x{:04x}\n", lo));
                if hi != 0 {
                    s.push_str(&format!("    movk    w10, #0x{:04x}, lsl #16\n", hi));
                }
                s.push_str(&format!("    add     {}, sp, x10\n", reg));
                s
            }
        }
        BufferLoc::Alias(_) => unreachable!("alias must be resolved by caller"),
    }
}

/// Validate that an op is supported in M4a; return error otherwise.
/// Linear with `bias=true` rejected; UnsupportedOp for softmax, dropout.
fn classify_op(
    op: StdOp,
    attrs: &[compiler::OpAttr],
    span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => {
            if linear_has_bias(attrs) {
                Err(LowerError::LinearWithBias { span })
            } else {
                Ok(())
            }
        }
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Err(LowerError::UnsupportedOp {
            op: "dropout".into(),
            span,
        }),
        StdOp::Softmax => Err(LowerError::UnsupportedOp {
            op: "softmax".into(),
            span,
        }),
    }
}
