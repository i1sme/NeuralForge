// SPDX-License-Identifier: Apache-2.0

//! NeuralForge x86_64 scalar codegen profile.
//!
//! Lowers a [`compiler::Uir`] to x86_64 Linux ELF assembly text via
//! [`X86_64Profile`]. Scalar SSE2 only — no SIMD/AVX. AT&T syntax.

mod asm;
mod buffer;
mod codegen;
mod ops;

pub use profile_api::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

use compiler::Uir;
use profile_api::Profile;

/// x86_64 profile implementation. Lowers UIR to x86_64 Linux ELF assembly.
pub struct X86_64Profile;

impl Profile for X86_64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        ""
    }
}

/// Free-function shim, mirror of arm64's. Equivalent to
/// `X86_64Profile.lower(uir)`.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    X86_64Profile.lower(uir)
}

#[cfg(test)]
mod tests;
