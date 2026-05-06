// SPDX-License-Identifier: Apache-2.0

//! UIR-level optimisation passes.
//!
//! Passes are functional transformations on a `Uir`: they take an
//! immutable `&Uir` and return a fresh `Uir` with the transformation
//! applied. NodeIds in the new graph are freshly numbered 0..N;
//! references (operands, model.inputs, model.output) are remapped during
//! reconstruction. This guarantees no stale-NodeId hazards for downstream
//! consumers (codegen, viewer, future passes).
//!
//! # Adding a new pass
//!
//! 1. Create `passes/<name>.rs` exposing a unit struct that implements
//!    `UirPass`. The `name()` method returns a stable snake_case
//!    identifier (used by CLI flags like `--passes=...`); never change
//!    once shipped.
//! 2. Add the pass to `default_pipeline()` if it should run by default.
//! 3. Add inline `#[cfg(test)] mod tests` covering pattern detection,
//!    NodeId remapping, edge cases, and the `pass.name()` contract.
//!
//! # Why functional?
//!
//! In-place mutation requires every consumer of a `&Uir` to know about
//! tombstones / removed nodes / "this NodeId may be invalid". Functional
//! passes hand back a clean, dense graph: NodeIds 0..N, all valid.
//! Tests can compare pre- and post-pass UIRs side-by-side.
//!
//! # Pipeline
//!
//! `default_pipeline()` returns a `Vec<Box<dyn UirPass>>` of passes to
//! run by default, in order. `run_pipeline(uir, &passes)` threads the
//! UIR through each pass; on the first error the pipeline halts.
//!
//! Currently registered: `EliminateDropout`, `FuseLinearRelu`, `FuseLinearSoftmax`.
//! See `default_pipeline()` for the canonical order and the
//! reasoning behind it.

use crate::ast::Span;
use crate::Uir;

pub(crate) mod rewriter;

pub mod eliminate_dropout;
pub mod fuse_linear_relu;
pub mod fuse_linear_softmax;

#[cfg(test)]
mod tests;

/// A UIR-level optimisation pass.
pub trait UirPass {
    /// Stable identifier for CLI flags (`--passes=...`), error messages,
    /// log lines. Snake_case. Once shipped, never change.
    fn name(&self) -> &str;

    /// Run the pass. Returns a new `Uir` (or input semantically-cloned
    /// if no patterns matched). Returns `Err(PassError)` only on
    /// defensively-detected malformed input.
    fn run(&self, uir: &Uir) -> Result<Uir, PassError>;
}

/// The default pipeline of passes, applied in order.
///
/// Order matters: `EliminateDropout` MUST run before `FuseLinearRelu`
/// and `FuseLinearSoftmax` so that `linear → dropout → relu` and
/// `linear → dropout → softmax` collapse before the fusion attempt.
/// Reversed order leaves patterns unfused forever — the fusion passes
/// would see Linear's consumer as Dropout (not Relu/Softmax) and
/// decline to fuse, then `EliminateDropout` would remove the dropout,
/// leaving an unfused chain.
///
/// `FuseLinearRelu` runs before `FuseLinearSoftmax`; both are
/// independent of each other (they match disjoint post-ops), so
/// their relative order is a stability convention, not a correctness
/// requirement.
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![
        Box::new(eliminate_dropout::EliminateDropout),
        Box::new(fuse_linear_relu::FuseLinearRelu),
        Box::new(fuse_linear_softmax::FuseLinearSoftmax),
    ]
}

/// Run a sequence of passes, threading the UIR through each. Stops on
/// first error.
pub fn run_pipeline(uir: &Uir, passes: &[Box<dyn UirPass>]) -> Result<Uir, PassError> {
    let mut current = uir.clone();
    for pass in passes {
        current = pass.run(&current)?;
    }
    Ok(current)
}

/// Errors produced by a pass.
///
/// Invariant: every variant carries a `Span`. If a future variant cannot
/// reasonably point to a source location, the `span()` accessor migrates
/// to `Option<Span>` at that point — but that is a deliberate breaking
/// change, not an organic drift.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PassError {
    /// Defensive: pass found malformed input it can't handle. Should be
    /// unreachable if `ir::build` returned Ok. Carries the pass name +
    /// reason for diagnostics, plus a span pointing into the offending
    /// model.
    InvalidInput {
        pass: String,
        reason: String,
        span: Span,
    },
}

impl PassError {
    /// All current variants carry a span; this method returns it without
    /// `Option`. See enum doc-comment for migration plan.
    pub fn span(&self) -> Span {
        match self {
            PassError::InvalidInput { span, .. } => *span,
        }
    }
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::InvalidInput { pass, reason, .. } => {
                write!(f, "pass '{}' failed: {}", pass, reason)
            }
        }
    }
}

impl std::error::Error for PassError {}
