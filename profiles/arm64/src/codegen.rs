//! UIR → AArch64 asm walker.
//!
//! Per-op emitters land here as Tasks 3-5 progress.

use crate::asm;
use crate::{Asm, FnSig, LowerError};
use compiler::{NodeKind, StdOp, Uir, UirModel};

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
    // Validate: every Op node must be a supported op. Walk first to surface
    // errors before emitting any asm.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // Compute ABI sizes from input + output shapes.
    let input_id = *model.inputs.first().ok_or(LowerError::ShapeNotConcrete {
        span: model.source_span,
    })?;
    let input_shape = &model.nodes[input_id].ty.shape;
    let output_shape = &model.nodes[model.output].ty.shape;
    let input_floats: usize = input_shape.0.iter().product::<u64>() as usize;
    let output_floats: usize = output_shape.0.iter().product::<u64>() as usize;

    // Sum weight sizes for all Linear ops in topological (UIR-node) order.
    let mut weight_floats: usize = 0;
    for node in &model.nodes {
        if let NodeKind::Op { op: StdOp::Linear, operands, .. } = &node.kind {
            // Input shape of this linear is the operand's shape; output rank-2 col is N.
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            let k = in_shape.0[in_shape.0.len() - 1] as usize;
            let n = out_shape.0[out_shape.0.len() - 1] as usize;
            weight_floats += k * n;
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        weight_floats,
        output_floats,
    };

    let mut body = String::new();
    body.push_str(&asm::format_function_header(&sig));

    // Emit per-op asm, in topological (UIR-node) order.
    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    for node in &model.nodes {
        if let NodeKind::Op { op, operands, .. } = &node.kind {
            match op {
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    // shape is [batch, k] for input and [batch, n] for output
                    if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                        return Err(LowerError::ShapeNotConcrete { span: node.source_span });
                    }
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];
                    body.push_str(&emit_matmul(b, k, n, linear_idx));
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    // Operates in-place on the producer's output buffer.
                    // For M4a (terminal-relu only) this is the model output (x2).
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    body.push_str(&emit_relu(total, relu_idx));
                    relu_idx += 1;
                }
                _ => unreachable!("classify_op should have caught this"),
            }
        }
    }

    body.push_str(&asm::format_function_footer());

    Ok((body, sig))
}

/// Emit the AArch64 matmul body for one Linear op.
///
/// Per spec §7: input is row-major [B, K], weights row-major [K, N],
/// output row-major [B, N]. ABI registers: x0=input, x1=weights, x2=output.
/// `linear_idx` is a unique-per-Linear suffix used in label names so
/// multiple Linear ops in one model don't collide on labels.
fn emit_matmul(b: u64, k: u64, n: u64, linear_idx: usize) -> String {
    let mut s = String::new();
    let lid = linear_idx;

    s.push_str(&format!("    ; matmul: input [{b},{k}] × weights [{k},{n}] → output [{b},{n}]\n"));

    // Hoist loop-invariant stride constants out of the inner loops.
    // x10 = K (input row stride and weight row count)
    // x11 = N (weight row stride and output row stride)
    s.push_str(&format!("    mov     x10, #{k}\n"));
    s.push_str(&format!("    mov     x11, #{n}\n"));

    // Outer i loop
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    // j loop
    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    // sum = 0
    s.push_str("    fmov    s0, wzr\n");

    // k loop
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    // input[i*K + k]
    s.push_str("    mul     x6, x3, x10\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x0, x6, lsl #2]\n");

    // weights[k*N + j]
    s.push_str("    mul     x7, x5, x11\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x1, x7, lsl #2]\n");

    // sum += s1 * s2
    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // store output[i*N + j]
    s.push_str("    mul     x6, x3, x11\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x2, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}

/// Emit AArch64 elementwise relu over a buffer of `total_floats` f32 elements.
///
/// Operates in-place on the buffer pointed to by `x2` (the model output buffer).
/// In M4a this is always the producer's terminal output. M4b will generalise
/// to intermediate buffers when multi-stage Linear is added.
///
/// Uses `s4` for the persistent zero; `s3` for the per-element load/store.
/// `s4` is chosen because `s0`–`s3` are already used by matmul (sum, ldr × 2).
/// `relu_idx` is a unique-per-Relu suffix for label naming.
fn emit_relu(total_floats: u64, relu_idx: usize) -> String {
    let mut s = String::new();
    let rid = relu_idx;

    s.push_str(&format!("    ; relu: in-place clamp on output buffer ({total_floats} elements)\n"));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x2, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x2, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));

    s
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
            // bias=true is not yet supported.
            for a in attrs {
                if a.name == "bias" {
                    if let compiler::AttrValue::Symbol(s) = &a.value {
                        if s == "true" {
                            return Err(LowerError::LinearWithBias { span });
                        }
                    }
                }
            }
            Ok(())
        }
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Err(LowerError::UnsupportedOp { op: "dropout".into(), span }),
        StdOp::Softmax => Err(LowerError::UnsupportedOp { op: "softmax".into(), span }),
    }
}
