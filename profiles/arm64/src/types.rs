//! Public types for the arm64 codegen profile.

use compiler::ast::Span;

/// Generated assembly source plus per-function metadata.
#[derive(Debug, Clone)]
pub struct Asm {
    /// Full AArch64 Mach-O assembly source. UTF-8.
    pub source: String,
    /// One entry per UirModel in the input UIR, in declaration order.
    pub functions: Vec<FnSig>,
}

/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// External symbol name without leading underscore. e.g. "nfl_forward_TinyMLP".
    /// Mach-O linkers prepend the underscore; `dlsym` users pass this name verbatim.
    pub name: String,
    /// Original UIR model name.
    pub model: String,
    /// Number of f32 elements in the input buffer.
    pub input_floats: usize,
    /// Total number of f32 elements across all weight matrices, packed in
    /// UIR-node order. M4a always single-Linear so this equals the one
    /// matrix's element count.
    pub weight_floats: usize,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
}

/// Errors that can occur during lowering.
///
/// `#[non_exhaustive]` — variants representing deferred features
/// (`UnsupportedOp`, `LinearWithBias`) become unreachable as M4b/c add
/// coverage and may be removed at that point.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// Op is not supported in the current M4 slice.
    /// `op` is the lowercase token name (e.g. "softmax", "dropout").
    UnsupportedOp { op: String, span: Span },
    /// `linear[N, bias=true]` is not yet implemented (M4b).
    LinearWithBias { span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    /// Should be unreachable if the IR builder did its job.
    ShapeNotConcrete { span: Span },
    /// Two `UirModel`s share the same `name` — would emit duplicate
    /// `nfl_forward_<name>` symbols. M4b moves this check into `ir::build`.
    DuplicateModelName { name: String, span: Span },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } =>
                write!(f, "operation '{}' is not supported by the arm64 profile in M4a", op),
            LowerError::LinearWithBias { .. } =>
                write!(f, "linear[..., bias=true] is not yet implemented (M4b)"),
            LowerError::ShapeNotConcrete { .. } =>
                write!(f, "internal: UIR shape was not fully resolved before lowering"),
            LowerError::DuplicateModelName { name, .. } =>
                write!(f, "duplicate model name '{}': would emit conflicting symbols", name),
        }
    }
}

impl LowerError {
    /// Returns the source span associated with the error.
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::LinearWithBias { span } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::DuplicateModelName { span, .. } => *span,
        }
    }
}
