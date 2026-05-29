// SPDX-License-Identifier: Apache-2.0

//! Public profile contract.
//!
//! Architecture profiles (`profiles/arm64`, `profiles/x86_64`) implement
//! the [`Profile`] trait. The compiler core (`compiler/`) does not depend
//! on any specific profile — UIR is profile-agnostic.

use compiler::ast::Span;
use compiler::ir::types::Uir;
use compiler::NodeId;

/// Where an Op-node's output buffer lives at run time.
///
/// `InputReg(idx)` carries the input's position in `model.inputs`
/// (M12+). The codegen profile maps `idx` → ABI register via
/// `AbiContext::input_reg`. For N=1 this is always `0` (= `x0` on
/// arm64, `%rdi` on x86_64), preserving M3-M11 single-input behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    /// Input pointer at `model.inputs[idx]`. Mapped to a profile arg
    /// register by `AbiContext::input_reg(idx)`.
    InputReg(usize),
    /// Output pointer (the FFI register at `INPUT_REGS[n_inputs + 1]`).
    OutputReg,
    /// Stack slot at `[sp + offset]` (arm64) or `[%rsp + offset]` (x86_64).
    StackOffset(usize),
    /// This buffer is an alias for another node's buffer. Resolved by
    /// `codegen::resolve_loc` before any emit.
    Alias(NodeId),
}

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
/// for future kinds without breaking match arms in downstream crates.
/// (M14 added LayerNormScale + LayerNormBias.)
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    LinearWeight,
    LinearBias,
    /// M14: LayerNorm γ scale parameter, shape `[input.last_dim]`.
    /// MUST be pushed before LayerNormBias in `params_layout` — order
    /// is contract; Pass 3 emitter reads γ then β by `find` on
    /// `(kind, origin_node)`. Mirror of LinearWeight / LinearBias
    /// ordering.
    LayerNormScale,
    /// M14: LayerNorm β bias parameter, shape `[input.last_dim]`.
    /// MUST be pushed AFTER LayerNormScale (see LayerNormScale doc).
    LayerNormBias,
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
    /// Model declared more inputs than this profile's ABI register window
    /// can hold without stack-spilling. The `max` field carries the
    /// profile-specific cap.
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

// ----------------------------------------------------------------------------
// M16 (A3): Profile-aware inspection schema.
//
// Returned by `Profile::inspect()`. Mirror of `Asm` in role: where Asm
// is "what lowering produces (text)", Inspection is "what lowering
// would compute (structured analysis)". Both consume the same
// per-profile analyze() preamble (M16 Task 1) — drift between them
// is impossible by construction.
// ----------------------------------------------------------------------------

/// Profile-aware annotation of one Uir, returned by `Profile::inspect`.
/// One entry per UirModel in the input UIR, in declaration order.
#[derive(Debug, Clone)]
pub struct Inspection {
    pub functions: Vec<FnAnnotations>,
}

/// Annotation for one UirModel under one profile.
///
/// `nodes.len() == post_pass_model.nodes.len()` — strictly index-aligned
/// with the **post-pass** UirModel that gets lowered. Pre-pass alignment
/// would produce a report whose node IDs don't match what `lower()`
/// actually compiles, defeating the point of A3.
#[derive(Debug, Clone)]
pub struct FnAnnotations {
    pub fn_sig: FnSig,
    pub stack_bytes: usize,
    /// Textual rendering of the profile's RegSet — lossy by design.
    /// arm64: e.g. `["d8-d9", "x19-x23"]`. x86_64: e.g. `["%rbx", "%r12-%r15"]`.
    /// Empty Vec if no callee-saved registers are touched by this function.
    pub callee_saved: Vec<String>,
    /// True iff the model contains no softmax (`!UirModel::has_softmax()`).
    /// Conservative: softmax models stay non-leaf through M17's exp-inline;
    /// precise reclassification is deferred to M18. (Today a softmax model
    /// also emits `bl _expf` / `call expf@PLT`, but the predicate is
    /// `has_softmax`, not "calls extern math".)
    pub leaf: bool,
    /// Real NodeId of each input in the post-pass UirModel, in
    /// declaration order. Renderer uses these to produce `n<id>` refs;
    /// without this field, positional indices would not match actual
    /// NodeIds in models where inputs are not the first N nodes.
    pub input_nodes: Vec<compiler::NodeId>,
    /// Real NodeId of the model output in the post-pass UirModel.
    pub output_node: compiler::NodeId,
    pub nodes: Vec<NodeAnnotation>,
}

/// Per-node annotation. Index in `FnAnnotations.nodes` corresponds to
/// `NodeId` in the post-pass `UirModel`.
///
/// **Growth rule:** new fields land here only when meaningful for both
/// profiles. Profile-specific information goes into `extra_notes` rather
/// than as a top-level field, to keep the schema honest cross-profile.
#[derive(Debug, Clone)]
pub struct NodeAnnotation {
    /// Pre-rendered description of the node — op kind, shape, operands,
    /// attrs, fused post-ops. Format mirrors `Display for compiler::Node`
    /// (the `--uir-verbose` style); produced once at inspect time so the
    /// renderer doesn't need access to the source UirModel.
    /// Examples:
    /// - `input "x"        :: Tensor[8, 4]`
    /// - `linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]    fused=[softmax_row]`
    pub label: String,
    pub buffer_loc: BufferLoc,
    /// `element_count * 4` (BYTES_PER_ELEMENT). For aliased nodes this
    /// is still the *logical* output size — the node "produces" this
    /// many bytes; physical placement is captured by `buffer_loc`.
    pub output_bytes: usize,
    /// `Some(N)` for ops that consume packed `params` slots:
    /// `Linear` (weights ± bias) and `LayerNorm[affine=true]`
    /// (γ + β). `None` for all other ops.
    pub params_floats: Option<usize>,
    /// Profile-specific freeform annotations. Empty for now; reserved
    /// for the growth-rule escape hatch.
    pub extra_notes: Vec<String>,
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

    /// M16 (A3): inspect the UIR under this profile, returning per-model
    /// and per-node annotations matching what `lower()` would produce.
    /// Both methods share an internal `analyze()` preamble — drift
    /// between inspection output and lowered asm is structurally
    /// impossible.
    fn inspect(&self, uir: &Uir) -> Result<Inspection, LowerError>;
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
        assert!(msg.contains("maximum of"), "got: {msg}");
    }
}
