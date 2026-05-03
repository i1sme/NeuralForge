//! UIR → AArch64 asm walker.
//!
//! Per-op emitters land here as Tasks 3-5 progress.

use crate::asm;
use crate::{Asm, FnSig, LowerError};
use compiler::{NodeKind, StdOp, Uir, UirModel};

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
    for (i, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op: StdOp::Linear, operands, .. } = &node.kind {
            // Input shape of this linear is the operand's shape; output rank-2 col is N.
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            let k = in_shape.0[in_shape.0.len() - 1] as usize;
            let n = out_shape.0[out_shape.0.len() - 1] as usize;
            weight_floats += k * n;
            let _ = i; // index reserved for future weight-layout metadata
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
    // Body emission (matmul, relu) lands in Tasks 4 and 5.
    body.push_str(&asm::format_function_footer());

    Ok((body, sig))
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
