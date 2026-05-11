// SPDX-License-Identifier: Apache-2.0

//! NeuralForge arm64 scalar codegen profile.
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via
//! [`Arm64Profile`]. The free [`lower`] shim is preserved for direct
//! callers (arm64 integration tests) that pre-date the trait.

mod abi;
mod asm;
mod buffer;
mod codegen;
mod ops;

pub use profile_api::{Asm, FnSig, Inspection, LowerError, ParamKind, ParamSlot};

use compiler::Uir;
use profile_api::Profile;

/// arm64 profile implementation. Lowers UIR to AArch64 Mach-O assembly.
pub struct Arm64Profile;

impl Profile for Arm64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        "_"
    }

    fn inspect(&self, uir: &Uir) -> Result<profile_api::Inspection, LowerError> {
        codegen::inspect_uir(uir)
    }
}

/// Free-function shim retained for direct callers (arm64 integration
/// tests). Equivalent to `Arm64Profile.lower(uir)`.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    Arm64Profile.lower(uir)
}

#[cfg(test)]
mod tests;
