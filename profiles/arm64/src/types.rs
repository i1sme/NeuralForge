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
    pub origin_node: compiler::NodeId,
    pub offset: usize,
    pub size: usize,
}

/// Type tag for a `ParamSlot`. `#[non_exhaustive]` per spec §5.2.
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
    /// All M4b ops (linear/relu/dropout/softmax with or without bias) are
    /// supported; this variant exists as a guard for M5+ ops landing before
    /// codegen catches up.
    #[allow(dead_code)]
    UnsupportedOp { op: String, span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    ShapeNotConcrete { span: Span },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } => write!(
                f,
                "operation '{}' is not supported by the arm64 profile in M4a",
                op
            ),
            LowerError::ShapeNotConcrete { .. } => write!(
                f,
                "internal: UIR shape was not fully resolved before lowering"
            ),
        }
    }
}

impl LowerError {
    /// Returns the source span associated with the error.
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
        }
    }
}
