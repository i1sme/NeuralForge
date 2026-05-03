//! NeuralForge arm64 scalar codegen profile (M4a).
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via [`lower`].

mod types;

pub use types::{Asm, FnSig, LowerError};

use compiler::Uir;

/// Lower a [`Uir`] to AArch64 assembly.
///
/// Returns [`LowerError`] on any unsupported op or structural problem.
/// See the M4a spec for op coverage details.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    // Stub: real impl arrives in Tasks 3-6.
    if let Some(model) = uir.models.first() {
        // Find the first op in the model and report it as unsupported,
        // so the stub at least returns a meaningful error per UIR.
        for node in &model.nodes {
            if let compiler::NodeKind::Op { op, .. } = &node.kind {
                return Err(LowerError::UnsupportedOp {
                    op: format!("{}", op),
                    span: node.source_span,
                });
            }
        }
    }
    Ok(Asm { source: String::new(), functions: Vec::new() })
}

#[cfg(test)]
mod tests;
