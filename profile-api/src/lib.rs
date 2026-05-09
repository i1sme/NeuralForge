// SPDX-License-Identifier: Apache-2.0

//! Public profile contract.
//!
//! Architecture profiles (`profiles/arm64`, `profiles/x86_64`) implement
//! the [`Profile`] trait. The compiler core (`compiler/`) does not depend
//! on any specific profile — UIR is profile-agnostic.

use compiler::ast::Span;
use compiler::ir::types::Uir;
use compiler::NodeId;

/// Generated assembly source plus per-function metadata.
#[derive(Debug, Clone)]
pub struct Asm {
    /// Full assembly source. UTF-8.
    pub source: String,
    /// One entry per UirModel in the input UIR, in declaration order.
    pub functions: Vec<FnSig>,
}

/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// External symbol name without leading underscore. e.g. "nfl_forward_TinyMLP".
    /// Mach-O linkers prepend the underscore; ELF linkers do not. `dlsym`
    /// callers pass this name verbatim.
    pub name: String,
    /// Original UIR model name.
    pub model: String,
    /// Number of f32 elements per input buffer, in declaration order.
    /// Length = arity (number of inputs); for single-input models length = 1.
    pub inputs_floats: Vec<usize>,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
    /// Total number of f32 elements in the packed params buffer.
    pub params_floats: usize,
    /// Layout of the packed params buffer, one entry per parameter slot in
    /// UIR-node order.
    pub params_layout: Vec<ParamSlot>,
}

/// One slot within the packed `params` buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSlot {
    pub kind: ParamKind,
    pub origin_node: NodeId,
    pub offset: usize,
    pub size: usize,
}

/// Type tag for a `ParamSlot`. `#[non_exhaustive]` keeps the door open
/// for future kinds (e.g. `LayerNormScale`) without breaking match arms
/// in downstream crates.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    LinearWeight,
    LinearBias,
}

/// Errors that can occur during lowering.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// Defensive: op encountered that the codegen doesn't know how to lower.
    UnsupportedOp { op: String, span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    ShapeNotConcrete { span: Span },
    /// Defensive: post-op variant not supported by this profile.
    UnsupportedPostOp { op: String, span: Span },
    /// Model declared more inputs than the profile's ABI register window
    /// can hold without stack-spilling. M12 caps both profiles at N=4
    /// (max=4 in the variant).
    TooManyInputs { n: usize, max: usize, span: Span },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } => {
                write!(f, "operation '{}' is not supported by this profile", op)
            }
            LowerError::ShapeNotConcrete { .. } => write!(
                f,
                "internal: UIR shape was not fully resolved before lowering"
            ),
            LowerError::UnsupportedPostOp { op, .. } => {
                write!(f, "post-op '{}' is not supported by this profile", op)
            }
            LowerError::TooManyInputs { n, max, .. } => write!(
                f,
                "model declares {} inputs but this profile supports a maximum of {}",
                n, max
            ),
        }
    }
}

impl std::error::Error for LowerError {}

impl LowerError {
    /// Returns the source span associated with the error.
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::UnsupportedPostOp { span, .. } => *span,
            LowerError::TooManyInputs { span, .. } => *span,
        }
    }
}

/// The profile contract.
///
/// Each backend profile (arm64 Mach-O, x86_64 Linux ELF, ...) provides
/// one `impl Profile` for its profile struct. The compiler core never
/// references a concrete profile by type — only through this trait.
///
/// **Trait grows by request, not by anticipation** (per M9 brainstorm
/// hard rule). Adding a method requires a real consumer in the codebase
/// that needs it.
pub trait Profile {
    /// Lower a [`Uir`] to the profile's target assembly.
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;

    /// Platform-specific external-symbol prefix.
    /// `"_"` on Mach-O (linker prepends underscore for C linkage),
    /// `""` on ELF (linker uses raw symbol name).
    fn sym_prefix(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span::new(1, 1)
    }

    #[test]
    fn asm_round_trip_through_debug() {
        let a = Asm {
            source: "x".into(),
            functions: vec![],
        };
        let dbg = format!("{:?}", a);
        assert!(dbg.contains("source"));
    }

    #[test]
    fn fn_sig_round_trip_through_debug() {
        let s = FnSig {
            name: "f".into(),
            model: "M".into(),
            inputs_floats: vec![1],
            output_floats: 1,
            params_floats: 0,
            params_layout: vec![],
        };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("FnSig"));
        assert!(dbg.contains("inputs_floats: [1]"));
    }

    #[test]
    fn param_slot_round_trip_through_debug() {
        let p = ParamSlot {
            kind: ParamKind::LinearWeight,
            origin_node: 0,
            offset: 0,
            size: 4,
        };
        let dbg = format!("{:?}", p);
        assert!(dbg.contains("LinearWeight"));
    }

    #[test]
    fn lower_error_display_message_is_profile_neutral() {
        let e = LowerError::UnsupportedOp {
            op: "foo".into(),
            span: dummy_span(),
        };
        let msg = format!("{}", e);
        assert!(
            msg.contains("not supported by this profile"),
            "Display message must be profile-neutral; got: {}",
            msg
        );
        assert!(
            !msg.contains("arm64"),
            "Display must not mention arm64; got: {}",
            msg
        );
        assert!(
            !msg.contains("x86_64"),
            "Display must not mention x86_64; got: {}",
            msg
        );
    }

    #[test]
    fn lower_error_span_round_trip() {
        let s = Span::new(3, 7);
        let e = LowerError::ShapeNotConcrete { span: s };
        assert_eq!(e.span().line, 3);
        assert_eq!(e.span().col, 7);
    }

    #[test]
    fn lower_error_too_many_inputs_display() {
        let e = LowerError::TooManyInputs {
            n: 5,
            max: 4,
            span: Span::new(1, 1),
        };
        let msg = format!("{}", e);
        assert!(msg.contains("5"), "got: {msg}");
        assert!(msg.contains("4"), "got: {msg}");
        assert!(
            msg.contains("not supported") || msg.contains("maximum"),
            "msg should explain the limit; got: {msg}"
        );
    }
}
