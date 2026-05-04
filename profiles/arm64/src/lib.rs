//! NeuralForge arm64 scalar codegen profile (M4a).
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via [`lower`].

mod asm;
// Task 3 will wire buffer into codegen; allow dead_code until then.
#[allow(dead_code)]
mod buffer;
mod codegen;
mod types;

pub use types::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

use compiler::Uir;

/// Lower a [`Uir`] to AArch64 assembly.
///
/// Returns [`LowerError`] on any unsupported op or structural problem.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    codegen::walk_uir(uir)
}

#[cfg(test)]
mod tests;
