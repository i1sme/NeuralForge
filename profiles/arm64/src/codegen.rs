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

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    let weight_offset = sig
                        .params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    body.push_str(&crate::ops::emit_linear(
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
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_relu(total, relu_idx, src_loc, dst_loc));
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
            BufferLoc::Alias(next) => {
                debug_assert!(next < cur, "alias must point backward (cycle defense)");
                cur = next;
            }
            other => return other,
        }
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
