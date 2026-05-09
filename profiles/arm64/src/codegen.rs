// SPDX-License-Identifier: Apache-2.0

//! UIR → AArch64 asm walker.
//!
//! Per-op emitters land here as Tasks 3-5 progress.

use crate::abi::{AbiContext, INPUT_REGS};
use compiler::{NodeId, NodeKind, StdOp, Uir, UirModel};
use profile_api::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

/// Walk the entire UIR, returning the combined asm source + per-model FnSigs.
pub fn walk_uir(uir: &Uir, sym_prefix: &'static str) -> Result<Asm, LowerError> {
    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for (model_idx, model) in uir.models.iter().enumerate() {
        let (model_asm, sig) = walk_model(model_idx, model, sym_prefix)?;
        source.push_str(&model_asm);
        source.push('\n');
        functions.push(sig);
    }

    Ok(Asm { source, functions })
}

fn walk_model(
    model_idx: usize,
    model: &UirModel,
    sym_prefix: &'static str,
) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue, LeafKind};
    use crate::buffer::{assign_buffers, compute_callee_saved, compute_is_leaf};

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 1b. Arity check (M12 spec §5.3): N + 2 ≤ INPUT_REGS.len(). The
    // current cap is N=4 (INPUT_REGS = x0..x5). Larger N would require
    // stack-spill on the input side, deferred to a future milestone.
    let n_inputs = model.inputs.len();
    if n_inputs + 2 > INPUT_REGS.len() {
        return Err(LowerError::TooManyInputs {
            n: n_inputs,
            max: INPUT_REGS.len() - 2,
            span: model.source_span,
        });
    }
    let abi = AbiContext { n_inputs };

    // 2. Compute layout, ABI sizes. inputs_floats is now a per-input
    // vec (M12 multi-input ABI); for N=1 this is just `vec![input_0]`.
    if model.inputs.is_empty() {
        return Err(LowerError::ShapeNotConcrete {
            span: model.source_span,
        });
    }
    let inputs_floats: Vec<usize> = model
        .inputs
        .iter()
        .map(|&id| model.nodes[id].ty.shape.0.iter().product::<u64>() as usize)
        .collect();
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op {
            op: StdOp::Linear,
            operands,
            attrs,
            ..
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
            if compiler::ir::linear_has_bias(attrs) {
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
        inputs_floats,
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
        sym_prefix,
    ));

    // Per-op emission (Tasks 5-8 refactor this dispatch into ops/*).
    let mut add_idx = 0usize;
    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    let mut softmax_idx = 0usize;
    let mut dropout_idx = 0usize;
    let mut matmul_idx = 0usize;
    let mut mulscalar_idx = 0usize;
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
                    let bias_offset = sig
                        .params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearBias && s.origin_node == node_idx)
                        .map(|s| s.offset);

                    // M5a: read fused_post_ops from the node and pass through.
                    let NodeKind::Op { fused_post_ops, .. } = &node.kind else {
                        unreachable!("walk_model already matched NodeKind::Op")
                    };

                    body.push_str(&crate::ops::emit_linear(
                        &abi,
                        b,
                        k,
                        n,
                        model_idx,
                        linear_idx,
                        src_loc,
                        dst_loc,
                        weight_offset,
                        bias_offset,
                        node.source_span,
                        fused_post_ops,
                        sym_prefix,
                    )?);
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_relu(
                        &abi, total, model_idx, relu_idx, src_loc, dst_loc,
                    ));
                    relu_idx += 1;
                }
                StdOp::Dropout => {
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    if matches!(dst_loc, crate::buffer::BufferLoc::OutputReg) {
                        // Bug-fix path (M8): dropout-as-output requires explicit copy
                        // because BufferLoc::Alias redirection doesn't apply when this
                        // node IS the output. See `ops/dropout.rs` module doc.
                        let total: u64 = node.ty.shape.0.iter().product();
                        body.push_str(&crate::ops::emit_dropout_copy(
                            &abi,
                            total,
                            model_idx,
                            dropout_idx,
                            src_loc,
                            dst_loc,
                        ));
                        dropout_idx += 1;
                    }
                    // else BufferLoc::Alias: no asm — downstream reads operand directly.
                }
                StdOp::Softmax => {
                    // Last-axis softmax. b = product(shape[..rank-1]) (total
                    // rows), k = shape[rank-1] (row width). For 2D
                    // [batch, dim] this collapses to b=batch, k=dim
                    // (identical to pre-M10 behaviour). For 4D
                    // [B, H, M, K] this gives b = B*H*M, k = K.
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let last = in_shape.0.len() - 1;
                    let k = in_shape.0[last];
                    let b: u64 = in_shape.0[..last].iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_softmax(
                        &abi,
                        b,
                        k,
                        model_idx,
                        softmax_idx,
                        src_loc,
                        dst_loc,
                        sym_prefix,
                    ));
                    softmax_idx += 1;
                }
                StdOp::Matmul => {
                    // Operands: input (operands[0]) is A (the LHS, which
                    // came from the pipeline). The Tensor-resolved B
                    // operand is operands[1] — pushed by build_op from
                    // `tensor_operands`.
                    let a_id = operands[0];
                    let b_id = operands[1];
                    let a_shape = &model.nodes[a_id].ty.shape;
                    let b_shape = &model.nodes[b_id].ty.shape;
                    let r = a_shape.0.len();
                    debug_assert!(r >= 2, "matmul shape inference enforces rank >= 2");

                    let leading_count: u64 = a_shape.0[..(r - 2)].iter().product();
                    let m = a_shape.0[r - 2];
                    let k = a_shape.0[r - 1];
                    let transpose_b = compiler::ir::stdlib::matmul_transpose_b(match &node.kind {
                        NodeKind::Op { attrs, .. } => attrs,
                        _ => unreachable!("matched NodeKind::Op above"),
                    });
                    let n = if transpose_b {
                        b_shape.0[r - 2]
                    } else {
                        b_shape.0[r - 1]
                    };

                    let a_loc = resolve_loc(&assignment.locs, a_id);
                    let b_loc = resolve_loc(&assignment.locs, b_id);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_matmul(
                        &abi,
                        leading_count,
                        m,
                        k,
                        n,
                        transpose_b,
                        model_idx,
                        matmul_idx,
                        a_loc,
                        b_loc,
                        dst_loc,
                        node.source_span,
                    )?);
                    matmul_idx += 1;
                }
                StdOp::MulScalar => {
                    let total: u64 = node.ty.shape.0.iter().product();
                    let attrs = match &node.kind {
                        NodeKind::Op { attrs, .. } => attrs,
                        _ => unreachable!(),
                    };
                    // f64 stored in attrs; truncate to f32 bits at the
                    // codegen boundary per spec §6.5.
                    let scalar_f64 = attrs
                        .iter()
                        .find(|a| a.name == "value")
                        .and_then(|a| match a.value {
                            compiler::AttrValue::Float(v) => Some(v),
                            _ => None,
                        })
                        .expect("MulScalar.value attr must be Float (signature enforces)");
                    let scalar_bits = (scalar_f64 as f32).to_bits();

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_mulscalar(
                        &abi,
                        total,
                        scalar_bits,
                        model_idx,
                        mulscalar_idx,
                        src_loc,
                        dst_loc,
                    ));
                    mulscalar_idx += 1;
                }
                StdOp::Add => {
                    let total_elements: u64 = node.ty.shape.0.iter().product();
                    let a_loc = resolve_loc(&assignment.locs, operands[0]);
                    let other_loc = resolve_loc(&assignment.locs, operands[1]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_add(
                        &abi,
                        total_elements,
                        model_idx,
                        add_idx,
                        a_loc,
                        other_loc,
                        dst_loc,
                    ));
                    add_idx += 1;
                }
                // M5c: #[non_exhaustive] on StdOp requires a wildcard
                // arm. Future ops (e.g. Tanh, Gelu, Embedding) will
                // route here until codegen learns them. Returning
                // LowerError::UnsupportedOp keeps the failure mode
                // graceful (M4b-era variant, made live by this arm).
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(LowerError::UnsupportedOp {
                        op: format!("{op}"),
                        span: node.source_span,
                    });
                }
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

/// Validate that an op is supported in M4b; return error otherwise.
/// All M4b ops are supported; kept as extension point for M5+ ops.
fn classify_op(
    op: StdOp,
    _attrs: &[compiler::OpAttr],
    _span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => Ok(()),
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Ok(()),
        StdOp::Softmax => Ok(()),
        StdOp::Matmul => Ok(()),
        StdOp::MulScalar => Ok(()),
        StdOp::Add => Ok(()),
        // M5c: #[non_exhaustive] on StdOp requires a wildcard arm.
        // Future ops are rejected here until codegen learns them.
        #[allow(unreachable_patterns)]
        _ => Err(LowerError::UnsupportedOp {
            op: format!("{op}"),
            span: _span,
        }),
    }
}
