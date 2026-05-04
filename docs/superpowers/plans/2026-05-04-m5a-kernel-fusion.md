# M5a — Kernel Fusion Pass (linear → relu) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `compiler::passes` UIR-pass infrastructure and the first fusion pass `FuseLinearRelu` (linear-no-bias + single-consumer Linear → Relu). Profile asm-fusion via inline `fmax` recovers M4a's in-place relu. CLI gains `--no-fuse` flag with strict stdout/stderr discipline. Integration test confirms fused vs unfused is bit-identical.

**Architecture:** Three layers in dependency order. (1) UIR types extended with `PostOp` enum and `fused_post_ops` field on `NodeKind::Op`. (2) `compiler/src/passes/` module with `UirPass` trait + `FuseLinearRelu` impl + pipeline runner. (3) `profiles/arm64::emit_linear` consumes `fused_post_ops` and emits inline `fmax s0, s0, s4` before store; CLI calls `passes::run_pipeline` between `ir::build` and `profile.lower`.

**Tech Stack:** Rust 2021 (std-only for production crates; `libloading` 0.8 dev-dep already in place). AArch64 Mach-O assembly. AAPCS64 ABI.

**Source spec:** [`docs/superpowers/specs/2026-05-04-m5a-kernel-fusion-design.md`](../specs/2026-05-04-m5a-kernel-fusion-design.md). All architectural decisions and rationale live there. **If this plan disagrees with the spec, the spec wins.**

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5-kernel-fusion` (branch `claude/m5-kernel-fusion`, base `main` at `f3ce49c`).

**Project conventions** (`CLAUDE.md` + spec §11):
- `cargo fmt --all` before every commit (CI gates on `--check`).
- `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
- `cargo test --workspace` passes; test count goes up monotonically.
- Production crates strictly std-only.
- TDD: failing test first, minimal impl, verify pass, commit.

**Pre-task baseline:**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "BASELINE:", sum}'
# Expect: 148 (M4 fully shipped).
```

---

## File Structure

### Created

| Path | Responsibility | Created in task |
|---|---|---|
| `compiler/src/passes/mod.rs` | `UirPass` trait, `default_pipeline`, `run_pipeline`, `PassError`, module-level doc-comment | Task 3 |
| `compiler/src/passes/fuse_linear_relu.rs` | `FuseLinearRelu` impl + algorithm + inline `#[cfg(test)] mod tests` | Task 4 |
| `compiler/src/passes/tests.rs` | Pipeline-level unit tests | Task 3 |

### Modified

| Path | What changes | Tasks |
|---|---|---|
| `compiler/src/ir/types.rs` | New `pub enum PostOp { Relu }` `#[non_exhaustive]`; `NodeKind::Op` gains `fused_post_ops: Vec<PostOp>`; `Display for PostOp`; `Display for Node` renders optional `fused=[...]` suffix | Tasks 1, 2 |
| `compiler/src/ir/build.rs` | All `NodeKind::Op` constructions add `fused_post_ops: Vec::new()` | Task 1 |
| `compiler/src/ir/stdlib.rs` | New `pub fn linear_has_bias(attrs: &[OpAttr]) -> bool` (moved from `profiles/arm64::codegen`) | Task 1 |
| `compiler/src/ir/tests.rs` | Existing pattern matches use `..` to ignore new `fused_post_ops` field | Task 1 |
| `compiler/src/lib.rs` | `pub mod passes;` + `pub use passes::{...}` + `pub use ir::types::PostOp;` | Tasks 1, 3 |
| `profiles/arm64/src/types.rs` | New `LowerError::UnsupportedPostOp { op, span }` variant + Display + span() | Task 5 |
| `profiles/arm64/src/codegen.rs` | Drop local `linear_has_bias` (use `compiler::ir::linear_has_bias`); pass `node.source_span` and `fused_post_ops` to `emit_linear`; `?` propagation | Tasks 1, 5 |
| `profiles/arm64/src/ops/linear.rs` | `emit_linear` signature gains `node_span: Span` and `fused_post_ops: &[PostOp]`; returns `Result<String, LowerError>`; emits `fmov s4, wzr` + `fmax s0, s0, s4` inline when fused | Tasks 5, 6 |
| `profiles/arm64/src/tests.rs` | 3 new tests for fused asm shape | Task 6 |
| `profiles/arm64/tests/integration.rs` | New `fused_vs_unfused_classifier_match_numerically` test; existing M4b integration tests adapted to default-fused path | Tasks 9, 10 |
| `nflc/src/main.rs` | New `parse_compile_args` stateful parser; `--no-fuse` flag; `passes::run_pipeline` between build and lower; `note:` to stderr | Task 7 |
| `nflc/tests/cli_compile.rs` | New file with 3 CLI smoke tests | Task 8 |

### Deleted

Nothing.

---

## Verification approach

| Check | When | How |
|---|---|---|
| Build clean | Every task | `cargo build --workspace` |
| Fmt clean | Every task before commit | `cargo fmt --all` then verify `cargo fmt --all -- --check` |
| Clippy clean | Tasks 1, 3, 4, 5, 6, 7, 11 | `cargo clippy --workspace --all-targets -- -D warnings` |
| Tests pass | Every task | `cargo test --workspace` |
| Integration tests | Tasks 9, 10, 11 | `cargo test -p profiles-arm64 --test integration` |
| CLI smoke (default) | Task 11 | `cargo run --bin nflc -- compile tests/fixtures/classifier.nfl --profile arm64 -o /tmp/c.s 2>/tmp/c.stderr` — stderr has `note: applied passes`, /tmp/c.s has `fmax s0, s0, s4`, no `.Lrelu_` |
| CLI smoke (--no-fuse) | Task 11 | Same fixture with `--no-fuse` — stderr has `note: passes skipped`, asm has `.Lrelu_*` loops |
| stdout/stderr discipline | Task 11 | `cargo run … --profile arm64 2>/dev/null \| head -5` shows only asm directives |

---

## Task list

| # | Task | Mode | Commits |
|---|---|---|---|
| 1 | UIR types: `PostOp` enum + `fused_post_ops` field; relocate `linear_has_bias` | SUBAGENT | 1 |
| 2 | `Display` extension for `fused_post_ops` (`fused=[...]` suffix) | SUBAGENT | 1 |
| 3 | `passes` module skeleton + `UirPass` trait + pipeline-level tests | SUBAGENT | 1 |
| 4 | `FuseLinearRelu` pass + algorithm + 9 unit tests | SUBAGENT | 1 |
| 5 | `profiles/arm64` types + `emit_linear` signature change + sites updated | SUBAGENT | 1 |
| 6 | `emit_linear` asm fusion (fmov+fmax inline) + 3 unit tests | SUBAGENT | 1 |
| 7 | CLI `parse_compile_args` + `--no-fuse` + `run_pipeline` wiring | SUBAGENT | 1 |
| 8 | CLI smoke tests (3) | SUBAGENT | 1 |
| 9 | Integration test `fused_vs_unfused_classifier_match_numerically` | SUBAGENT | 1 |
| 10 | M4b integration tests adapt to default-fused path | INLINE (mechanical) | 1 |
| 11 | Closeout: DEVLOG + CLAUDE.md Current Status | INLINE | 1 |

**Total:** 11 tasks, 11 commits. Targets baseline 148 → 167 tests at end.

---

## Task 1: UIR types — `PostOp` enum + `fused_post_ops` field; relocate `linear_has_bias`

**Goal:** Extend `NodeKind::Op` with `fused_post_ops: Vec<PostOp>`. Define `PostOp` enum. Move `linear_has_bias` from `profiles/arm64::codegen` to `compiler::ir::stdlib` so passes (which live in `compiler/`) can use it. Existing tests pass with `..` patterns.

**Files:**
- Modify: `compiler/src/ir/types.rs`, `compiler/src/ir/build.rs`, `compiler/src/ir/stdlib.rs`, `compiler/src/ir/tests.rs`, `compiler/src/lib.rs`, `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Add `PostOp` enum to `compiler/src/ir/types.rs`**

After the existing `pub struct Shape` block (or before `NodeKind`), add:

```rust
/// Post-operations that fuse into a producer's output store.
///
/// `#[non_exhaustive]` — M5b/M6+ may add Gelu, Tanh, Sigmoid. Each variant
/// is meaningful as "applied to one element after the producer computes it,
/// before storing"; not all StdOps fit (Softmax needs row-context, Dropout
/// is no-op at inference, Linear can't post-op another Linear).
///
/// Keeping `PostOp` distinct from `StdOp` makes the constraint explicit at
/// type level: profiles can't mistakenly route a softmax through the
/// post-op machinery.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOp {
    /// Clamp negative values to zero (max(x, 0)). Equivalent to fusing a
    /// terminal-or-single-consumer Relu node into its producer.
    Relu,
}
```

- [ ] **Step 2: Extend `NodeKind::Op` in the same file**

Find the existing:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input { name: String },
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
    },
}
```

Replace with:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input { name: String },
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
        /// Fused post-operations, applied per-element after this op
        /// produces its output, before storing. Empty for un-fused or
        /// non-Linear ops.
        ///
        /// Populated only by `passes::FuseLinearRelu` (M5a) and future
        /// fusion passes. `compiler::ir::build` always sets this to
        /// `Vec::new()`.
        ///
        /// `Vec` rather than `Option` so M5b can express chains like
        /// `[BiasAdd, Relu]` should the need arise.
        fused_post_ops: Vec<PostOp>,
    },
}
```

- [ ] **Step 3: Move `linear_has_bias` to `compiler/src/ir/stdlib.rs`**

In `compiler/src/ir/stdlib.rs`, append:

```rust
/// True iff the op's attribute list includes `bias=true` (for `Linear`).
///
/// Used by the codegen profile to detect bias-add cases and by fusion
/// passes (M5a) to skip `linear[bias=true]` for Linear→Relu fusion in
/// the M5a slice.
pub fn linear_has_bias(attrs: &[crate::ir::types::OpAttr]) -> bool {
    use crate::ir::types::AttrValue;
    attrs.iter().any(|a| {
        a.name == "bias" && matches!(&a.value, AttrValue::Symbol(s) if s == "true")
    })
}
```

In `compiler/src/ir/mod.rs` (or wherever stdlib re-exports happen), ensure `linear_has_bias` is reachable as `compiler::ir::linear_has_bias`. Add to existing `pub use stdlib::*;` or explicit re-export:

```rust
pub use stdlib::linear_has_bias;
```

- [ ] **Step 4: Add `pub use ir::types::PostOp;` to `compiler/src/lib.rs`**

In `compiler/src/lib.rs`, after the existing `pub use ir::{...};` block, add:

```rust
pub use ir::types::PostOp;
```

- [ ] **Step 5: Update all `NodeKind::Op` constructions in `compiler/src/ir/build.rs`**

Find every `NodeKind::Op { op, operands, attrs }` (or similar field-naming) struct literal and add `fused_post_ops: Vec::new()`.

Search:
```bash
grep -n "NodeKind::Op {" compiler/src/ir/build.rs
```

For each hit (typically 2-3 sites), add the new field. Example:

```rust
// Before:
kind: NodeKind::Op {
    op: stdlib_op,
    operands: vec![input_id],
    attrs,
},

// After:
kind: NodeKind::Op {
    op: stdlib_op,
    operands: vec![input_id],
    attrs,
    fused_post_ops: Vec::new(),
},
```

- [ ] **Step 6: Adapt existing pattern matches in `compiler/src/ir/tests.rs`**

Find every `NodeKind::Op { op, operands, attrs } => {` or destructuring without `..`. Add `..` so the new field is ignored where the test doesn't care:

Search:
```bash
grep -n "NodeKind::Op {" compiler/src/ir/tests.rs
```

For each hit:

```rust
// Before:
let NodeKind::Op { op, operands, attrs } = &node.kind else { panic!() };

// After:
let NodeKind::Op { op, operands, attrs, .. } = &node.kind else { panic!() };
```

(Keep the same fields the test uses; just add `..` at the end.)

- [ ] **Step 7: Drop local `linear_has_bias` in `profiles/arm64/src/codegen.rs`**

In `profiles/arm64/src/codegen.rs`, find and DELETE the local `fn linear_has_bias` (currently around line 10). Update its sole caller (`classify_op` arm) to use the compiler-level helper.

Find:
```rust
fn linear_has_bias(attrs: &[compiler::OpAttr]) -> bool {
    attrs.iter().any(|a| {
        a.name == "bias" && matches!(&a.value, compiler::AttrValue::Symbol(s) if s == "true")
    })
}
```

Delete it. The single caller (`classify_op` checking Linear) replaces `linear_has_bias(attrs)` with `compiler::ir::linear_has_bias(attrs)`. Update the call site.

- [ ] **Step 8: Adapt existing pattern matches in `profiles/arm64`**

Some places in `profiles/arm64/src/codegen.rs` and `profiles/arm64/src/buffer.rs` destructure `NodeKind::Op`. Add `..` to the patterns:

```bash
grep -rn "NodeKind::Op {" profiles/arm64/src/
```

For each hit, ensure `..` is at the end of the pattern.

- [ ] **Step 9: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 148 tests still pass (no test count change — pure type extension + relocation).

- [ ] **Step 10: Commit**

```bash
git add compiler/ profiles/
git commit -m "feat(m5a/types): PostOp enum + fused_post_ops field on NodeKind::Op

Per spec §6: extend UIR types for fusion infrastructure.

- New pub enum PostOp { Relu, ... } with #[non_exhaustive] in
  compiler/src/ir/types.rs. Distinct from StdOp by design (see spec
  §6.1 reasoning: Softmax/Dropout/Linear don't fit as post-ops).
- NodeKind::Op gains fused_post_ops: Vec<PostOp> field. Empty by
  default; populated only by fusion passes.
- linear_has_bias relocated from profiles/arm64::codegen to
  compiler::ir::stdlib (fusion pass needs it; profile call site
  now uses the compiler-level fn).
- All existing NodeKind::Op constructions in ir::build set
  fused_post_ops: Vec::new() explicitly.
- Existing pattern matches in compiler/src/ir/tests.rs and
  profiles/arm64 use '..' to ignore the new field.
- pub use ir::types::PostOp added to compiler/src/lib.rs for
  downstream profile crates.

Display impls for the new field land in Task 2; the FuseLinearRelu
pass that populates it lands in Task 4. 148 tests still pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `Display` extension for `fused_post_ops` (`fused=[...]` suffix)

**Goal:** `Display for Node` renders `fused=[<list>]` suffix when `fused_post_ops` is non-empty. `Display for PostOp` prints lowercase variant name. UIR-print output for un-fused nodes unchanged (back-compat for M3c+ rendering).

**Files:**
- Modify: `compiler/src/ir/types.rs`

- [ ] **Step 1: Add failing test to `compiler/src/ir/tests.rs`**

Append:

```rust
#[test]
fn display_for_postop_lowercase() {
    use crate::ir::PostOp;
    assert_eq!(format!("{}", PostOp::Relu), "relu");
}

#[test]
fn display_for_node_renders_fused_post_ops_when_present() {
    use crate::ir::types::{Node, NodeKind, OpAttr, AttrValue, Type, Shape, PostOp};
    use crate::ir::stdlib::StdOp;
    use crate::ast::Span;

    let n = Node {
        kind: NodeKind::Op {
            op: StdOp::Linear,
            operands: vec![0],
            attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }],
            fused_post_ops: vec![PostOp::Relu],
        },
        ty: Type { name: "Tensor".into(), shape: Shape(vec![8, 2]) },
        source_span: Span::new(1, 1),
    };
    let rendered = format!("{}", n);
    assert!(rendered.contains("linear"));
    assert!(rendered.contains("operands=[n0]"));
    assert!(rendered.contains("attrs=[out_dim=2]"));
    assert!(rendered.contains("fused=[relu]"));
}

#[test]
fn display_for_node_omits_fused_when_empty() {
    use crate::ir::types::{Node, NodeKind, Type, Shape};
    use crate::ir::stdlib::StdOp;
    use crate::ast::Span;

    let n = Node {
        kind: NodeKind::Op {
            op: StdOp::Linear,
            operands: vec![0],
            attrs: vec![],
            fused_post_ops: vec![],
        },
        ty: Type { name: "Tensor".into(), shape: Shape(vec![8, 2]) },
        source_span: Span::new(1, 1),
    };
    let rendered = format!("{}", n);
    assert!(!rendered.contains("fused"), "empty fused_post_ops should NOT render 'fused' substring; got: {rendered}");
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test --lib display_for_postop_lowercase 2>&1 | tail -5
```

Expected: compile error — `Display` not implemented for `PostOp`.

- [ ] **Step 3: Add `Display for PostOp` in `compiler/src/ir/types.rs`**

After the `PostOp` enum:

```rust
impl std::fmt::Display for PostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PostOp::Relu => "relu",
        };
        write!(f, "{}", name)
    }
}
```

- [ ] **Step 4: Extend `Display for Node`'s Op arm**

Find the existing `impl std::fmt::Display for Node` (M3c-era). The Op arm currently looks like:

```rust
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            NodeKind::Input { name } => {
                write!(f, "input {:?}        :: {}", name, self.ty.shape)
            }
            NodeKind::Op { op, operands, attrs } => {
                let ops_s = operands.iter().map(|o| format!("n{}", o)).collect::<Vec<_>>().join(", ");
                write!(f, "{}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
                if !attrs.is_empty() {
                    let a = attrs.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                Ok(())
            }
        }
    }
}
```

Update the Op arm pattern + add fused suffix:

```rust
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            NodeKind::Input { name } => {
                write!(f, "input {:?}        :: {}", name, self.ty.shape)
            }
            NodeKind::Op { op, operands, attrs, fused_post_ops } => {
                let ops_s = operands.iter().map(|o| format!("n{}", o)).collect::<Vec<_>>().join(", ");
                write!(f, "{}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
                if !attrs.is_empty() {
                    let a = attrs.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                if !fused_post_ops.is_empty() {
                    let f_s = fused_post_ops.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
                    write!(f, "    fused=[{}]", f_s)?;
                }
                Ok(())
            }
        }
    }
}
```

- [ ] **Step 5: Verify PASS**

```bash
cargo test --lib display_for_postop_lowercase display_for_node_renders_fused_post_ops_when_present display_for_node_omits_fused_when_empty 2>&1 | tail -10
```

Expected: 3 new tests pass.

- [ ] **Step 6: Build + fmt + clippy + full test count**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 151 tests passing (148 + 3).

- [ ] **Step 7: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m5a/types): Display for PostOp + fused=[...] suffix on Node

Per spec §6.3: extend Display impls.

- Display for PostOp: lowercase variant name ('relu', matches StdOp
  Display convention).
- Display for Node Op arm: appends '    fused=[<list>]' suffix only
  when fused_post_ops is non-empty. Un-fused nodes render exactly
  as before — back-compat for M3c+ 'nflc parse <file> --uir' output.

3 new unit tests cover both branches (rendering present, omitted
when empty) and the PostOp lowercase contract.

151 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: `passes` module skeleton + `UirPass` trait + pipeline-level tests

**Goal:** Establish `compiler/src/passes/` infrastructure: `UirPass` trait, `PassError`, `default_pipeline()`, `run_pipeline()`. 3 pipeline-level unit tests using a synthetic identity pass. The actual `FuseLinearRelu` pass lands in Task 4.

**Files:**
- Create: `compiler/src/passes/mod.rs`, `compiler/src/passes/tests.rs`
- Modify: `compiler/src/lib.rs`

- [ ] **Step 1: Create `compiler/src/passes/mod.rs`**

```rust
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
//! M5a registers exactly one pass: `FuseLinearRelu`. M5b adds
//! `EliminateDropout`. Subsequent milestones add more.

use crate::ast::Span;
use crate::Uir;

pub mod fuse_linear_relu;

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
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![Box::new(fuse_linear_relu::FuseLinearRelu)]
}

/// Run a sequence of passes, threading the UIR through each. Stops on
/// first error.
pub fn run_pipeline(
    uir: &Uir,
    passes: &[Box<dyn UirPass>],
) -> Result<Uir, PassError> {
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
    InvalidInput { pass: String, reason: String, span: Span },
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
```

- [ ] **Step 2: Create stub `compiler/src/passes/fuse_linear_relu.rs`**

A stub so `mod fuse_linear_relu;` compiles; full impl lands in Task 4.

```rust
//! `linear → relu` fusion pass. See spec §7 for the algorithm.
//!
//! M5a Task 4 fills this in.

use super::{PassError, UirPass};
use crate::Uir;

pub struct FuseLinearRelu;

impl UirPass for FuseLinearRelu {
    fn name(&self) -> &str {
        "fuse_linear_relu"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        // Stub: identity transform. Task 4 replaces with real algorithm.
        Ok(uir.clone())
    }
}
```

- [ ] **Step 3: Create `compiler/src/passes/tests.rs`**

```rust
//! Pipeline-level tests for `compiler::passes`.

use super::{default_pipeline, run_pipeline, PassError, UirPass};
use crate::Uir;

/// Synthetic identity pass for testing the pipeline mechanics without
/// depending on any specific transformation.
struct IdentityPass {
    name: &'static str,
}

impl UirPass for IdentityPass {
    fn name(&self) -> &str {
        self.name
    }
    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        Ok(uir.clone())
    }
}

#[test]
fn default_pipeline_includes_fuse_linear_relu() {
    let pipeline = default_pipeline();
    let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
    assert!(
        names.contains(&"fuse_linear_relu"),
        "default_pipeline must include 'fuse_linear_relu'; got: {:?}",
        names
    );
}

#[test]
fn run_pipeline_threads_uir_through_passes() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let passes: Vec<Box<dyn UirPass>> = vec![
        Box::new(IdentityPass { name: "id_a" }),
        Box::new(IdentityPass { name: "id_b" }),
    ];

    let out = run_pipeline(&uir, &passes).expect("pipeline ok");
    // Identity passes preserve model count + node count.
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].nodes.len(), uir.models[0].nodes.len());
}

#[test]
fn empty_pipeline_returns_input_clone() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let out = run_pipeline(&uir, &[]).expect("empty pipeline ok");
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].name, uir.models[0].name);
}
```

- [ ] **Step 4: Wire `passes` module + re-exports in `compiler/src/lib.rs`**

After existing `pub mod ir;`:

```rust
pub mod passes;

pub use passes::{default_pipeline, run_pipeline, PassError, UirPass};
```

- [ ] **Step 5: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 154 tests passing (151 + 3 pipeline-level).

- [ ] **Step 6: Commit**

```bash
git add compiler/
git commit -m "feat(m5a/passes): UirPass trait + default_pipeline + run_pipeline

Per spec §5: establish UIR-pass infrastructure.

- compiler/src/passes/mod.rs: UirPass trait with mandatory name() +
  run(&Uir) -> Result<Uir, PassError>. default_pipeline() returns
  Vec<Box<dyn UirPass>>; run_pipeline threads UIR through each.
  PassError #[non_exhaustive] with InvalidInput variant carrying Span.
- Module-level doc-comment explains: what passes are, how to add a
  new one, why functional, what default_pipeline contains.
- compiler/src/passes/fuse_linear_relu.rs: stub identity pass with
  name() = 'fuse_linear_relu'. Real algorithm in Task 4.
- compiler/src/passes/tests.rs: 3 pipeline-level tests using a
  synthetic IdentityPass. Verifies default_pipeline contract,
  threading, empty-pipeline corner case.
- pub mod passes + pub use in compiler/src/lib.rs.

154 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: `FuseLinearRelu` pass + algorithm + 9 unit tests

**Goal:** Replace Task 3's stub with the real `FuseLinearRelu` algorithm per spec §7. 9 inline unit tests cover all edge cases (terminal, chain, multi-consumer-relu, multi-consumer-linear, bias=true skip, already-fused skip, empty UIR, NodeId remap, name contract).

**Files:**
- Modify: `compiler/src/passes/fuse_linear_relu.rs`

- [ ] **Step 1: Replace `compiler/src/passes/fuse_linear_relu.rs` with full impl**

```rust
//! `linear → relu` fusion pass (spec §7).
//!
//! Finds nodes matching the pattern:
//!   Linear (no bias=true, fused_post_ops empty, single consumer)
//!     → Relu (any consumer count)
//!
//! Rewrites the graph:
//!   - Linear gets `fused_post_ops: vec![PostOp::Relu]`.
//!   - Relu node is removed; references to it are remapped to the fused
//!     Linear's new NodeId.
//!
//! Functional: returns a fresh Uir with renumbered NodeIds.

use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind, PostOp};
use crate::ir::{linear_has_bias, StdOp};
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};

pub struct FuseLinearRelu;

impl UirPass for FuseLinearRelu {
    fn name(&self) -> &str {
        "fuse_linear_relu"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model)?);
        }
        Ok(Uir { models: new_models })
    }
}

fn fuse_one_model(model: &UirModel) -> Result<UirModel, PassError> {
    // Step 1: consumer counts.
    let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
    for node in &model.nodes {
        if let NodeKind::Op { operands, .. } = &node.kind {
            for &op_id in operands {
                *consumer_count.entry(op_id).or_insert(0) += 1;
            }
        }
    }
    *consumer_count.entry(model.output).or_insert(0) += 1;

    // Step 2: identify victims (Relu nodes that fold into producer Linear).
    let mut victim_to_producer: HashMap<NodeId, NodeId> = HashMap::new();
    for (relu_id, relu_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op { op: StdOp::Relu, operands, .. } = &relu_node.kind
        else {
            continue;
        };
        if operands.len() != 1 {
            continue;
        }
        let linear_id = operands[0];
        let linear_node = &model.nodes[linear_id];
        let NodeKind::Op {
            op: StdOp::Linear,
            attrs,
            fused_post_ops,
            ..
        } = &linear_node.kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion in M5a.
        }
        if linear_has_bias(attrs) {
            continue; // M5a scope: bias-aware fusion is M5b.
        }
        if *consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Relu).
        }
        victim_to_producer.insert(relu_id, linear_id);
    }

    let victims: HashSet<NodeId> = victim_to_producer.keys().copied().collect();
    let producers_of_victims: HashSet<NodeId> =
        victim_to_producer.values().copied().collect();

    // Step 3: build new model.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.iter().enumerate() {
        if victims.contains(&old_id) {
            // Skip pushing; map old victim id → producer's new id.
            let producer_old_id = victim_to_producer[&old_id];
            let producer_new_id = id_map[&producer_old_id];
            id_map.insert(old_id, producer_new_id);
            continue;
        }

        // Clone + remap operands.
        let mut new_node = node.clone();
        if let NodeKind::Op { operands, fused_post_ops, .. } = &mut new_node.kind {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
            if producers_of_victims.contains(&old_id) {
                fused_post_ops.push(PostOp::Relu);
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    // Step 4: remap inputs + output.
    let new_inputs: Vec<NodeId> = model.inputs.iter().map(|id| id_map[id]).collect();
    let new_output = id_map[&model.output];

    Ok(UirModel {
        name: model.name.clone(),
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
        source_span: model.source_span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uir;

    fn build(src: &str) -> Uir {
        let ast = crate::parse(src).expect("parse");
        crate::ir::build(&ast).expect("ir::build")
    }

    #[test]
    fn pass_name_is_stable() {
        assert_eq!(FuseLinearRelu.name(), "fuse_linear_relu");
    }

    #[test]
    fn empty_uir_passes_unchanged() {
        let uir = Uir { models: Vec::new() };
        let out = FuseLinearRelu.run(&uir).expect("ok");
        assert_eq!(out.models.len(), 0);
    }

    #[test]
    fn fuses_simple_linear_relu() {
        // Terminal: x -> linear[2] -> relu
        let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];

        // Original had 3 nodes (input, linear, relu); fused has 2 (input, fused-linear).
        assert_eq!(m.nodes.len(), 2, "expected 2 nodes; got: {:?}", m.nodes);

        // Node 1 is the fused Linear.
        let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!("expected Op node");
        };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);

        // model.output points at the fused Linear.
        assert_eq!(m.output, 1);
    }

    #[test]
    fn does_not_fuse_when_linear_has_multiple_consumers() {
        // x -> linear[3] -> [softmax, relu]    (linear has 2 consumers)
        // NFL grammar is single-pipeline, so build via two pipelines:
        //   x: Tensor[b, 3]
        //   x -> linear[3] -> softmax    (consumer 1)
        //   x -> linear[3] -> relu       (DIFFERENT linear; same x but separate)
        // To get genuinely-shared linear we need to construct UIR by hand.
        // Simpler test: use NFL with a model where one Linear is consumed by
        // both relu and softmax. Currently NFL pipelines are linear chains;
        // achieving "one node, two consumers" requires tagged variables which
        // isn't in v0.1. So we test the synthetic UIR path:
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        use crate::ast::Span;
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let linear_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                fused_post_ops: vec![],
            },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op { op: StdOp::Relu, operands: vec![1], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let softmax_n = Node {
            kind: NodeKind::Op { op: StdOp::Softmax, operands: vec![1], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };

        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, linear_n, relu_n, softmax_n],
            inputs: vec![0],
            output: 3,  // softmax is the terminal
            source_span: span,
        };
        let uir = Uir { models: vec![model] };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 4 nodes preserved (no fusion).
        assert_eq!(m.nodes.len(), 4);
        // Linear's fused_post_ops is still empty.
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
        assert!(fused_post_ops.is_empty());
    }

    #[test]
    fn fuses_when_relu_has_multiple_consumers() {
        // x -> linear[3] -> relu -> [linear[2] (= consumer A), linear[2] (= consumer B)]
        // Same hand-built UIR pattern. Linear has only Relu as consumer; Relu has 2 downstream.
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        use crate::ast::Span;
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let linear_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                fused_post_ops: vec![],
            },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op { op: StdOp::Relu, operands: vec![1], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let consumer_a = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![2], // consumes relu
                attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }],
                fused_post_ops: vec![],
            },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) },
            source_span: span,
        };
        let consumer_b = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![2], // consumes relu (shared)
                attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }],
                fused_post_ops: vec![],
            },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) },
            source_span: span,
        };

        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, linear_n, relu_n, consumer_a, consumer_b],
            inputs: vec![0],
            output: 4,  // consumer_b
            source_span: span,
        };
        let uir = Uir { models: vec![model] };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 4 nodes (relu removed).
        assert_eq!(m.nodes.len(), 4);
        // Both consumer_a and consumer_b operands now reference the fused linear (new id 1).
        let NodeKind::Op { operands: ca_ops, .. } = &m.nodes[2].kind else { panic!() };
        let NodeKind::Op { operands: cb_ops, .. } = &m.nodes[3].kind else { panic!() };
        assert_eq!(ca_ops, &vec![1usize], "consumer_a should remap to fused linear (id 1)");
        assert_eq!(cb_ops, &vec![1usize], "consumer_b should remap to fused linear (id 1)");
        // Fused linear has post-op set.
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    }

    #[test]
    fn fuses_chain_independently() {
        // x: Tensor[b, 4] -> linear[8] -> relu -> linear[2] -> relu
        let uir = build("model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // Original: input + linear[8] + relu + linear[2] + relu = 5 nodes.
        // After fusion: input + fused linear[8] + fused linear[2] = 3 nodes.
        assert_eq!(m.nodes.len(), 3);
        // Both Linears have fused_post_ops = [Relu].
        let NodeKind::Op { fused_post_ops: f1, .. } = &m.nodes[1].kind else { panic!() };
        let NodeKind::Op { fused_post_ops: f2, .. } = &m.nodes[2].kind else { panic!() };
        assert_eq!(f1, &vec![PostOp::Relu]);
        assert_eq!(f2, &vec![PostOp::Relu]);
    }

    #[test]
    fn does_not_fuse_when_linear_has_bias() {
        let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 3 nodes preserved.
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
        assert!(fused_post_ops.is_empty());
    }

    #[test]
    fn does_not_fuse_when_linear_already_fused() {
        // Hand-build UIR where Linear ALREADY has fused_post_ops = [Relu], followed by another Relu.
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        use crate::ast::Span;
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let pre_fused_linear = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                fused_post_ops: vec![PostOp::Relu],  // already fused
            },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op { op: StdOp::Relu, operands: vec![1], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, pre_fused_linear, relu_n],
            inputs: vec![0],
            output: 2,
            source_span: span,
        };
        let uir = Uir { models: vec![model] };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 3 nodes preserved (no double-fusion).
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
        // Still just one Relu in fused_post_ops (not [Relu, Relu]).
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    }

    #[test]
    fn does_not_fuse_when_relu_not_after_linear() {
        // Synthetic: softmax → relu (NFL grammar may not allow; we hand-build UIR).
        use crate::ir::types::{Node, NodeKind, Shape, Type};
        use crate::ast::Span;
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let softmax_n = Node {
            kind: NodeKind::Op { op: StdOp::Softmax, operands: vec![0], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op { op: StdOp::Relu, operands: vec![1], attrs: vec![], fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
            source_span: span,
        };
        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, softmax_n, relu_n],
            inputs: vec![0],
            output: 2,
            source_span: span,
        };
        let uir = Uir { models: vec![model] };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        // 3 nodes preserved (softmax → relu is not fusable; only Linear → Relu fuses in M5a).
        assert_eq!(out.models[0].nodes.len(), 3);
    }

    #[test]
    fn model_inputs_and_output_remapped() {
        // Simple: x -> linear[2] -> relu (terminal). After fusion: input(0) + fused_linear(1).
        let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // Old model.output was 2 (relu); after fusion, points at fused linear (new id 1).
        assert_eq!(m.output, 1);
        // Old model.inputs was [0] (input); preserved as [0].
        assert_eq!(m.inputs, vec![0]);
    }
}
```

- [ ] **Step 2: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 154 + 9 = 163 tests passing. (Actually 154 was after Task 3's pipeline-tests landed; Task 3's stub `FuseLinearRelu` was identity, contributing zero new tests. Task 4 adds 9 inline tests in `fuse_linear_relu.rs::tests` — total 163.)

- [ ] **Step 3: Commit**

```bash
git add compiler/src/passes/fuse_linear_relu.rs
git commit -m "feat(m5a/passes): FuseLinearRelu pass with full algorithm + 9 unit tests

Per spec §7: implement Linear→Relu fusion.

Algorithm (per UirModel):
1. Build consumer_count: HashMap<NodeId, usize>.
2. Identify victims: Relu nodes whose Linear operand has exactly one
   consumer (this Relu), no bias=true, and empty fused_post_ops.
3. Functional rebuild: skip victim nodes, copy + remap operands of
   surviving nodes, append PostOp::Relu to fused_post_ops of producers.
4. Remap model.inputs and model.output via id_map.

9 inline unit tests cover all spec edge cases:
- pass_name_is_stable (CLI contract)
- empty_uir_passes_unchanged (corner case)
- fuses_simple_linear_relu (terminal, basic)
- fuses_chain_independently (linear→relu→linear→relu, both fuse)
- fuses_when_relu_has_multiple_consumers (asymmetric rule: check
  Linear's consumer count, NOT Relu's)
- does_not_fuse_when_linear_has_multiple_consumers (other consumers
  expect pre-relu output)
- does_not_fuse_when_linear_has_bias (M5a scope)
- does_not_fuse_when_linear_already_fused (no double-fusion)
- does_not_fuse_when_relu_not_after_linear (only Linear→Relu)
- model_inputs_and_output_remapped (NodeId remap correctness)

163 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: `profiles/arm64` types + `emit_linear` signature change + sites updated

**Goal:** Add `LowerError::UnsupportedPostOp { op, span }` variant. Change `emit_linear` signature: add `node_span: Span` and `fused_post_ops: &[PostOp]` parameters; return `Result<String, LowerError>`. Update call site in `walk_model` to pass new args + use `?`. **No fusion behaviour yet** — Task 6 adds the actual asm output. After this task, asm for un-fused models is identical (signature change is plumbing only).

**Files:**
- Modify: `profiles/arm64/src/types.rs`, `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Add `UnsupportedPostOp` variant in `profiles/arm64/src/types.rs`**

Find the `LowerError` enum and append the variant:

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// (existing) Defensive guard for unsupported StdOps.
    #[allow(dead_code)]
    UnsupportedOp { op: String, span: compiler::ast::Span },
    /// (existing) Defensive guard for non-concrete shapes.
    ShapeNotConcrete { span: compiler::ast::Span },
    /// M5a: post-op variant not supported by this profile. Fires when a
    /// future PostOp variant lands in PostOp before this profile knows
    /// how to emit it.
    UnsupportedPostOp { op: String, span: compiler::ast::Span },
}
```

Update `Display for LowerError`:

```rust
impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } => {
                write!(f, "operation '{}' is not supported by the arm64 profile", op)
            }
            LowerError::ShapeNotConcrete { .. } => write!(
                f,
                "internal: UIR shape was not fully resolved before lowering"
            ),
            LowerError::UnsupportedPostOp { op, .. } => write!(
                f,
                "post-op '{}' is not supported by the arm64 profile",
                op
            ),
        }
    }
}
```

Update `LowerError::span()`:

```rust
impl LowerError {
    pub fn span(&self) -> compiler::ast::Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::UnsupportedPostOp { span, .. } => *span,
        }
    }
}
```

- [ ] **Step 2: Change `emit_linear` signature in `profiles/arm64/src/ops/linear.rs`**

Find the existing `pub fn emit_linear(...)` and update its signature:

```rust
use crate::buffer::BufferLoc;
use crate::types::LowerError;
use compiler::ast::Span;
use compiler::PostOp;

#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    model_idx: usize,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
    node_span: Span,
    fused_post_ops: &[PostOp],
) -> Result<String, LowerError> {
    // existing body, wrapped: every `s.push_str(...)` stays the same;
    // the function ends with `Ok(s)` instead of `s`.
    // PostOp dispatch lands in Task 6; for now, if fused_post_ops is non-empty,
    // emit nothing extra (un-fused behaviour preserved).

    /* ... existing matmul body ... */

    // Defensive: if Task 6 isn't done and fused_post_ops is non-empty, ignore for now.
    // Task 6 replaces this with real per-PostOp dispatch.
    let _ = fused_post_ops;
    let _ = node_span;

    Ok(s)
}
```

(Concretely: rename current `s.push_str(...) ... s` returning a `String` to the same body but ending in `Ok(s)`. Add the two new params; mark them unused with `let _ =` so clippy doesn't warn until Task 6 uses them.)

- [ ] **Step 3: Update call site in `walk_model` (`profiles/arm64/src/codegen.rs`)**

Find the `StdOp::Linear =>` arm in `walk_model`. Update the `emit_linear` call:

```rust
StdOp::Linear => {
    let in_shape = &model.nodes[operands[0]].ty.shape;
    let out_shape = &node.ty.shape;
    let b = in_shape.0[0];
    let k = in_shape.0[1];
    let n = out_shape.0[1];

    let src_loc = resolve_loc(&assignment.locs, operands[0]);
    let dst_loc = resolve_loc(&assignment.locs, node_idx);
    let weight_offset = sig
        .params_layout
        .iter()
        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
        .expect("LinearWeight slot must exist for this Linear")
        .offset;
    let bias_offset = sig
        .params_layout
        .iter()
        .find(|s| s.kind == ParamKind::LinearBias && s.origin_node == node_idx)
        .map(|s| s.offset);

    // M5a: read fused_post_ops from the node and pass through.
    let NodeKind::Op { fused_post_ops, .. } = &node.kind else { unreachable!() };

    body.push_str(&crate::ops::emit_linear(
        b,
        k,
        n,
        model_idx,
        linear_idx,
        src_loc,
        dst_loc,
        weight_offset,
        bias_offset,
        node.source_span,
        fused_post_ops,
    )?);
    linear_idx += 1;
}
```

Note: `walk_model` returns `Result<(String, FnSig), LowerError>` already (from M4b's softmax additions). The `?` propagates LowerError correctly.

- [ ] **Step 4: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 163 tests still passing. (Signature is plumbing only; behaviour unchanged.)

- [ ] **Step 5: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m5a/arm64): emit_linear sig change + UnsupportedPostOp variant

Per spec §8: prepare profile for PostOp consumption.

- profiles/arm64/src/types.rs: new LowerError::UnsupportedPostOp
  { op: String, span: Span }. Updated Display + span() arms.
- profiles/arm64/src/ops/linear.rs: emit_linear signature gains
  node_span: Span and fused_post_ops: &[PostOp]. Returns
  Result<String, LowerError>. Body unchanged (Task 6 adds asm
  fusion); new params marked unused for now.
- profiles/arm64/src/codegen.rs: walk_model::StdOp::Linear arm
  reads node.source_span and fused_post_ops, passes them, uses ?
  for Result propagation.

Behaviour unchanged for un-fused models. asm bit-identical to M4b.

163 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: `emit_linear` asm fusion (fmov + fmax inline) + 3 unit tests

**Goal:** Implement actual fusion in `emit_linear`. When `fused_post_ops` contains `PostOp::Relu`, materialise `s4 = 0.0` once at function-header time and emit `fmax s0, s0, s4` between bias-add (if any) and store. `_ =>` arm returns `LowerError::UnsupportedPostOp`. 3 new unit tests cover fused/un-fused asm shape.

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add 3 failing unit tests in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn fused_linear_relu_emits_fmax_before_store() {
    use crate::ir::PostOp;
    // Synthetic: hand-build UIR where Linear has fused_post_ops = [Relu].
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    // Inject PostOp::Relu manually (simulate post-pass UIR).
    let mut uir = uir;
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else {
        panic!("expected Op node");
    };
    fused_post_ops.push(PostOp::Relu);

    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // s4 materialised once.
    assert!(s.contains("fmov    s4, wzr"), "missing s4 zero materialisation:\n{s}");
    // fmax inline before store.
    assert!(s.contains("fmax    s0, s0, s4"), "missing inline fmax (relu):\n{s}");
}

#[test]
fn fused_linear_relu_no_separate_relu_loop() {
    use crate::ir::PostOp;
    // Same fixture as above. Asm must NOT contain a separate .Lrelu_*: label.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let mut uir = uir;
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else { panic!() };
    fused_post_ops.push(PostOp::Relu);

    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        !s.contains(".Lrelu_"),
        "fused linear+relu should NOT emit a separate relu loop:\n{s}"
    );
}

#[test]
fn unfused_linear_still_no_fmax() {
    // Linear without fused_post_ops: no fmax in asm.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(!s.contains("fmax"), "un-fused linear should NOT emit fmax:\n{s}");
}
```

- [ ] **Step 2: Verify FAIL (first two tests)**

```bash
cargo test -p profiles-arm64 fused_linear_relu_emits_fmax_before_store fused_linear_relu_no_separate_relu_loop 2>&1 | tail -10
```

Expected: both fail (asm doesn't yet contain `fmax`/`s4` materialisation).

The third test (`unfused_linear_still_no_fmax`) should already pass.

- [ ] **Step 3: Implement asm fusion in `profiles/arm64/src/ops/linear.rs`**

Replace the body of `emit_linear` with the version from spec §8.2 + §8.3:

```rust
#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    model_idx: usize,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
    node_span: Span,
    fused_post_ops: &[PostOp],
) -> Result<String, LowerError> {
    let lid = format!("{model_idx}_{linear_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}{}\n",
        if bias_offset.is_some() { " + bias" } else { "" },
        if !fused_post_ops.is_empty() { " + fused" } else { "" },
    ));

    // Materialise s4 = 0.0 once if any post-op needs it.
    let needs_zero = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
    if needs_zero {
        s.push_str("    fmov    s4, wzr\n");
    }

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&emit_imm32("x9", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str("    mov     x14, x1\n");
        } else {
            s.push_str(&emit_imm32("x9", boff));
            s.push_str("    add     x14, x1, x9, lsl #2\n");
        }
    }

    // ... existing matmul nested loops (i, j, k) — unchanged from M4b ...
    // (Reproduce verbatim from current emit_linear; only the post-k-loop section changes.)

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // Bias-add (if present).
    if bias_offset.is_some() {
        s.push_str("    ldr     s5, [x14, x4, lsl #2]\n");
        s.push_str("    fadd    s0, s0, s5\n");
    }

    // M5a NEW: post-ops inline.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => s.push_str("    fmax    s0, s0, s4\n"),
            // _ arm: required by #[non_exhaustive] PostOp.
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: format!("{post_op:?}").to_lowercase(),
                    span: node_span,
                });
            }
        }
    }

    // Store.
    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    Ok(s)
}
```

- [ ] **Step 4: Verify all tests PASS**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 166 tests passing (163 + 3 new).

- [ ] **Step 5: Build + fmt + clippy clean**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add profiles/arm64/src/ops/linear.rs profiles/arm64/src/tests.rs
git commit -m "feat(m5a/arm64): emit_linear consumes fused_post_ops — inline fmax

Per spec §8.2-§8.3: when fused_post_ops contains PostOp::Relu,
materialise s4 = 0.0 once at function-header (after pointer setup),
and emit 'fmax s0, s0, s4' inline between bias-add and store.

The matchexpression on PostOp must include a '_ =>' arm because
PostOp is #[non_exhaustive]; the arm returns
LowerError::UnsupportedPostOp { op: <lowercase Debug>, span: node_span }
so future PostOp variants (Gelu, Tanh, ...) compile cleanly but
produce a clear runtime error from the profile.

Recovers M4a's in-place relu pattern. asm shape: matmul body, then
'fadd s0, s0, s5' (if bias), then 'fmax s0, s0, s4' (if fused-relu),
then 'str s0, [x12, ...]'.

3 new unit tests:
- fused_linear_relu_emits_fmax_before_store
- fused_linear_relu_no_separate_relu_loop
- unfused_linear_still_no_fmax (back-compat)

166 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: CLI `parse_compile_args` + `--no-fuse` + `run_pipeline` wiring

**Goal:** Refactor `nflc compile` arg-parsing into a stateful `parse_compile_args`. Add `--no-fuse` flag. Wire `compiler::passes::run_pipeline` between `ir::build` and `profile.lower`. Print `note:` lines to **stderr** per spec §9.3.

**Files:**
- Modify: `nflc/src/main.rs`

- [ ] **Step 1: Add `parse_compile_args` helper at the top of `nflc/src/main.rs`**

After existing helper fns (e.g. `format_dims`, `format_arg`) but before `run_compile`:

```rust
struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_fuse: bool,
}

fn parse_compile_args(args: &[String]) -> Result<CompileArgs, String> {
    // args here is everything AFTER the "compile" subcommand keyword.
    // First positional: path. Then sweep flags.
    let mut iter = args.iter();
    let path = iter
        .next()
        .ok_or_else(|| "compile: missing <file.nfl>".to_string())?
        .clone();

    let mut profile: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut no_fuse = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--profile" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                profile = Some(v.clone());
            }
            "-o" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "-o requires a value".to_string())?;
                output = Some(PathBuf::from(v));
            }
            "--no-fuse" => {
                no_fuse = true;
            }
            other => {
                return Err(format!("unknown flag: {other}"));
            }
        }
    }

    let profile = profile.ok_or_else(|| "compile: missing --profile <name>".to_string())?;

    Ok(CompileArgs {
        path: PathBuf::from(path),
        profile,
        output,
        no_fuse,
    })
}
```

- [ ] **Step 2: Replace pattern-matched arms in `main()` with single dispatch**

Find the existing `match args.as_slice()` block. Find the multi-arm `compile` matches (they likely look like `[cmd, path, p_flag, p_name] if cmd == "compile" && p_flag == "--profile"` etc.). Replace ALL `compile` arms with one:

```rust
        [cmd, rest @ ..] if cmd == "compile" => {
            match parse_compile_args(rest) {
                Ok(parsed) => run_compile(parsed),
                Err(msg) => {
                    eprintln!("error: {}", msg);
                    print_usage();
                    ExitCode::FAILURE
                }
            }
        }
```

- [ ] **Step 3: Update `run_compile` to accept `CompileArgs` and call `passes::run_pipeline`**

Replace `fn run_compile(path: PathBuf, profile: String, out_path: Option<PathBuf>) -> ExitCode` with:

```rust
fn run_compile(args: CompileArgs) -> ExitCode {
    let CompileArgs { path, profile, output: out_path, no_fuse } = args;

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ast = match compiler::parse(&source) {
        Ok(a) => a,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
            return ExitCode::FAILURE;
        }
    };

    let uir = match compiler::ir::build(&ast) {
        Ok(u) => u,
        Err(e) => {
            let first = match &e.kind {
                compiler::BuildErrorKind::DuplicateModelName { first_span, .. } => {
                    Some((first_span.line, first_span.col))
                }
                _ => None,
            };
            let msg = e.to_string();
            render_error_with_snippet(&source, &path, e.line, e.col, &msg, first);
            return ExitCode::FAILURE;
        }
    };

    if profile != "arm64" {
        eprintln!("error: unknown profile '{}' (supported: arm64)", profile);
        return ExitCode::FAILURE;
    }

    // M5a: run UIR-passes pipeline (default), or skip if --no-fuse.
    let post_pass_uir = if no_fuse {
        eprintln!("note: passes skipped (--no-fuse)");
        uir
    } else {
        let pipeline = compiler::passes::default_pipeline();
        let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
        eprintln!("note: applied passes: {}", names.join(", "));
        match compiler::passes::run_pipeline(&uir, &pipeline) {
            Ok(u) => u,
            Err(e) => {
                let span = e.span();
                render_error_with_snippet(
                    &source,
                    &path,
                    span.line,
                    span.col,
                    &format!("{}", e),
                    None,
                );
                return ExitCode::FAILURE;
            }
        }
    };

    match profiles_arm64::lower(&post_pass_uir) {
        Ok(asm) => match out_path {
            Some(p) => match std::fs::write(&p, &asm.source) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: cannot write {}: {}", p.display(), e);
                    ExitCode::FAILURE
                }
            },
            None => {
                print!("{}", asm.source);
                ExitCode::SUCCESS
            }
        },
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(&source, &path, span.line, span.col, &format!("{}", e), None);
            ExitCode::FAILURE
        }
    }
}
```

- [ ] **Step 4: Update `print_usage` to mention `--no-fuse`**

```rust
fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-fuse]        Skip optimisation passes (debugging)");
}
```

- [ ] **Step 5: Build + fmt + clippy + smoke test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Smoke positive (default):
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m4a_fused.s 2>/tmp/m4a.stderr
echo "exit: $?"
cat /tmp/m4a.stderr
grep "fmax" /tmp/m4a_fused.s

# Smoke positive (--no-fuse):
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m4a_unfused.s --no-fuse 2>/tmp/m4a_nf.stderr
echo "exit: $?"
cat /tmp/m4a_nf.stderr
grep ".Lrelu_" /tmp/m4a_unfused.s
```

Expected:
- Default: exit 0, stderr has `note: applied passes: fuse_linear_relu`, asm contains `fmax`.
- `--no-fuse`: exit 0, stderr has `note: passes skipped (--no-fuse)`, asm contains `.Lrelu_*` labels.

- [ ] **Step 6: Run full test suite**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 166 tests still passing.

- [ ] **Step 7: Commit**

```bash
git add nflc/src/main.rs
git commit -m "feat(m5a/cli): parse_compile_args + --no-fuse flag + run_pipeline wiring

Per spec §9: extend nflc compile with fusion control.

- New parse_compile_args stateful parser (~30 lines): handles
  positional <file>, --profile <name>, optional -o <path>, optional
  --no-fuse. Returns CompileArgs struct or human-readable error.
  Replaces pattern-explosion of slice-position match arms.
- run_compile rewritten around CompileArgs.
- Default mode: applies compiler::passes::default_pipeline() between
  ir::build and profile.lower; emits 'note: applied passes: <list>'
  to stderr.
- --no-fuse mode: skips passes; emits 'note: passes skipped (--no-fuse)'
  to stderr; profile receives raw UIR from ir::build.
- Strict stdout/stderr discipline: stdout = asm only (or empty if -o);
  stderr = all notes/errors.
- print_usage updated to document --no-fuse.

166 tests pass; CLI smoke confirms both default and --no-fuse paths
produce correct asm shape.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: CLI smoke tests

**Goal:** 3 CLI smoke tests (per spec §9.5) that exercise the binary via `Command::new(env!("CARGO_BIN_EXE_nflc"))`.

**Files:**
- Create: `nflc/tests/cli_compile.rs`

- [ ] **Step 1: Create `nflc/tests/cli_compile.rs`**

```rust
//! CLI integration tests for `nflc compile`.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn compile_default_runs_fusion() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // stderr has the applied-passes note.
    assert!(
        stderr.contains("note: applied passes: fuse_linear_relu"),
        "stderr missing applied-passes note:\n{stderr}"
    );

    // stdout has fused asm: inline fmax, no separate relu loop.
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout missing inline fmax:\n{stdout}"
    );
    assert!(
        !stdout.contains(".Lrelu_"),
        "stdout has separate relu loop (fusion did NOT apply):\n{stdout}"
    );
}

#[test]
fn compile_with_no_fuse_skips_fusion() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--no-fuse",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: passes skipped (--no-fuse)"),
        "stderr missing passes-skipped note:\n{stderr}"
    );

    // Unfused asm: separate relu loop, no inline fmax.
    assert!(
        stdout.contains(".Lrelu_0_0:"),
        "stdout missing relu loop label (un-fused mode):\n{stdout}"
    );
    assert!(
        !stdout.contains("fmax    s0, s0, s4"),
        "stdout has inline fmax (fusion incorrectly applied in --no-fuse mode):\n{stdout}"
    );
}

#[test]
fn compile_unknown_flag_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--frobnicate",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown flag: --frobnicate") || stderr.contains("error:"),
        "stderr missing unknown-flag error:\n{stderr}"
    );
}
```

- [ ] **Step 2: Run + verify**

```bash
cargo test -p nflc --test cli_compile 2>&1 | tail -15
```

Expected: 3 tests passing.

- [ ] **Step 3: Full test suite**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 169 tests passing (166 + 3 CLI smoke).

- [ ] **Step 4: Commit**

```bash
git add nflc/tests/cli_compile.rs
git commit -m "test(m5a/cli): smoke tests for --no-fuse + unknown-flag rejection

Per spec §9.5: 3 CLI integration tests using
Command::new(env!(\"CARGO_BIN_EXE_nflc\")):

- compile_default_runs_fusion: default mode → stderr has 'applied
  passes: fuse_linear_relu'; stdout has inline 'fmax s0, s0, s4',
  no separate '.Lrelu_*' labels.
- compile_with_no_fuse_skips_fusion: --no-fuse → stderr has 'passes
  skipped'; stdout has '.Lrelu_0_0:' label, no inline fmax.
- compile_unknown_flag_rejected: --frobnicate → exit 1, stderr has
  unknown-flag error.

169 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Integration test `fused_vs_unfused_classifier_match_numerically`

**Goal:** End-to-end FFI test confirming numerical bit-equivalence between fused and unfused asm output for `classifier.nfl` (2 internal `linear→relu` fusions).

**Files:**
- Modify: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Add new test in `profiles/arm64/tests/integration.rs`**

Append:

```rust
#[test]
fn fused_vs_unfused_classifier_match_numerically() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/classifier.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();

    // Build BOTH paths.
    let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let fused_asm = profiles_arm64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_arm64::lower(&uir).expect("unfused lower");

    // Asm shape differs as expected.
    assert!(
        fused_asm.source.contains("fmax    s0, s0, s4"),
        "fused asm missing inline fmax"
    );
    assert!(
        !fused_asm.source.contains(".Lrelu_"),
        "fused asm should NOT have separate relu loops"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_0:"),
        "unfused asm missing first relu loop label"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_1:"),
        "unfused asm missing second relu loop label (classifier has 2 relus)"
    );

    // Compile both, run both with same input/params, compare numerically.
    let fused_dylib = common::compile_to_dylib(&fused_asm.source, "fused_classifier");
    let unfused_dylib = common::compile_to_dylib(&unfused_asm.source, "unfused_classifier");

    let fused_lib = unsafe { libloading::Library::new(&fused_dylib).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_dylib).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_Classifier") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_Classifier") }.unwrap();

    // Same deterministic input + params as classifier_runs_correctly test.
    let mut input = vec![0.0f32; 32 * 784];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; 535040];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }

    let mut fused_out = vec![0.0f32; 32 * 10];
    let mut unfused_out = vec![0.0f32; 32 * 10];

    unsafe {
        fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
        unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
    }

    // assert_eq! exact equality: f32 store+load is bit-preserving;
    // fusion only relocates WHERE relu is applied, not WHICH floats compute.
    for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "fused[{i}]={a} unfused[{i}]={b} — fusion changed numerics"
        );
    }
}
```

- [ ] **Step 2: Build + clippy + run integration test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p profiles-arm64 --test integration fused_vs_unfused_classifier_match_numerically 2>&1 | tail -10
```

Expected on aarch64 macOS: 1 test passing. On other hosts: 1 test "passing" with skip-message.

- [ ] **Step 3: Full test suite**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 170 tests passing (169 + 1 integration).

- [ ] **Step 4: Commit**

```bash
git add profiles/arm64/tests/integration.rs
git commit -m "test(m5a/integration): fused_vs_unfused_classifier numerical equivalence

Per spec §10: end-to-end FFI test confirming fusion preserves
numerics. Uses classifier.nfl (2 internal linear→relu fusions in
one function: linear[512]→relu and linear[256]→relu plus non-fused
softmax terminal).

Both fused and unfused paths build UIR, lower to asm, compile via
cc -shared -arch arm64, dlopen via libloading, call with same
deterministic input/params (matches classifier_runs_correctly test
data), compare outputs with assert_eq! (bit-exact, NOT epsilon).

Asm shape pre-asserts:
- Fused: contains 'fmax s0, s0, s4'; does NOT contain '.Lrelu_*'.
- Unfused: contains '.Lrelu_0_0:' and '.Lrelu_0_1:' (2 relus).

Bit-exactness rationale: f32 store+load is bit-preserving by IEEE
754 + AArch64 spec; fusion only relocates where relu applies, not
which floats are computed.

170 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: M4b integration tests adapt to default-fused path (INLINE)

**Goal:** Existing M4b integration tests (`tinymlp_full_with_softmax_runs_correctly`, `classifier_runs_correctly`, `pipeline_styles_runs_correctly`, `comments_runs_correctly`, `mixed_args_runs_correctly`, `m4a_no_softmax_still_runs`) currently call `profiles_arm64::lower(&uir)` directly on `ir::build`'s output. After M5a they should be calling `lower(&run_pipeline(&uir, &default_pipeline())?)` — the default-fused path. Numerical assertions continue to hold (per spec AC #9).

**Files:**
- Modify: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Update each M4b integration test to apply pass pipeline**

For each of the 6 existing tests (`tinymlp_full_with_softmax_runs_correctly`, `classifier_runs_correctly`, `pipeline_styles_runs_correctly`, `comments_runs_correctly`, `mixed_args_runs_correctly`, `m4a_no_softmax_still_runs`), find the section:

```rust
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");
```

Replace the third line with:

```rust
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_arm64::lower(&uir).expect("lower");
```

(Insert the `run_pipeline` call between `ir::build` and `lower`. Re-bind `uir` to the post-pass UIR.)

- [ ] **Step 2: Run all integration tests**

```bash
cargo test -p profiles-arm64 --test integration 2>&1 | tail -15
```

Expected: All integration tests pass — numerically the assertions hold (the integration tests' inner numerical checks are reference-Rust-compared, not asm-snapshot-compared, so the bit-identical fused-vs-unfused property of fusion preserves them).

- [ ] **Step 3: Full test suite + clippy**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 170 tests still passing (no count change — same tests, default-fused path).

- [ ] **Step 4: Commit**

```bash
git add profiles/arm64/tests/integration.rs
git commit -m "test(m5a/integration): switch M4b tests to default-fused path

Per spec AC #9: existing M4b integration tests now apply
compiler::passes::run_pipeline before lowering, matching the
default behaviour of 'nflc compile'. This exercises the fused
codegen path under the same numerical assertions as before.

Tests updated:
- tinymlp_full_with_softmax_runs_correctly (no fusion: linear→softmax)
- classifier_runs_correctly (2 internal fusions)
- pipeline_styles_runs_correctly (3 internal fusions, 1 per model)
- comments_runs_correctly (depends on shape)
- mixed_args_runs_correctly (no fusion: bias=true excluded)
- m4a_no_softmax_still_runs (terminal fusion: M4a-style inline relu)

The fused_vs_unfused_classifier_match_numerically test added in
Task 9 directly verifies that fusion preserves numerics; this
task's change makes the M4b tests run the same fused asm.

170 tests pass; clippy/fmt clean.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Closeout — DEVLOG + CLAUDE.md (INLINE)

**Goal:** Final verification + DEVLOG entry + CLAUDE.md "Current Status" update. Profile guide doc is M5c, not M5a — explicitly note in the DEVLOG.

**Files:**
- Modify: `DEVLOG.md`, `CLAUDE.md`

- [ ] **Step 1: Final verification**

```bash
cargo fmt --all -- --check
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'

# CLI smoke positive (default + --no-fuse).
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m5a_d.s 2>/tmp/m5a_d.err
echo "default exit: $? — stderr: $(cat /tmp/m5a_d.err)"
grep -E "fmax|Lrelu_" /tmp/m5a_d.s | head -3

cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --no-fuse -o /tmp/m5a_n.s 2>/tmp/m5a_n.err
echo "no-fuse exit: $? — stderr: $(cat /tmp/m5a_n.err)"
grep -E "fmax|Lrelu_" /tmp/m5a_n.s | head -3

# stdout/stderr discipline.
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 2>/dev/null | head -3
echo "(stdout above must show ONLY .globl / labels; no 'note:' lines)"
```

Expected: all checks pass; default produces `fmax`, no `.Lrelu_`; --no-fuse produces `.Lrelu_*`, no `fmax`; stdout-only-asm verified.

- [ ] **Step 2: Append M5a entry to `DEVLOG.md`**

Find the most recent entry. Insert above it (separated by `---`):

```markdown
---

## 2026-05-04 — Milestone 5a closed: kernel fusion (linear → relu) + UIR-pass framework

### What was done
- Introduced `compiler::passes` UIR-pass infrastructure: `UirPass` trait
  with mandatory `name()` + functional `run(&Uir) -> Result<Uir, PassError>`,
  `default_pipeline()`, `run_pipeline()`. `PassError` `#[non_exhaustive]`
  with `InvalidInput` variant carrying span.
- Implemented `FuseLinearRelu` pass — finds `Linear (no bias=true,
  no existing fused_post_ops, single consumer) → Relu`, merges via
  `Linear.fused_post_ops = vec![PostOp::Relu]`, removes Relu node, remaps
  references with fresh NodeIds. 9 inline unit tests cover all spec
  edge cases (terminal, chain, multi-consumer-relu allowed, multi-consumer-
  linear forbidden, bias-true skip, double-fusion skip, NodeId remap).
- Extended UIR types: new `pub enum PostOp { Relu }` `#[non_exhaustive]`,
  separate from `StdOp` by design (Softmax/Dropout/Linear don't fit as
  post-ops). `NodeKind::Op` gains `fused_post_ops: Vec<PostOp>` field.
  `Display for Node` renders optional `fused=[<list>]` suffix.
- Relocated `linear_has_bias` from `profiles/arm64::codegen` to
  `compiler::ir::stdlib` so passes can use it.
- Profile changes: `profiles/arm64::emit_linear` accepts `node_span`
  and `fused_post_ops`, returns `Result<String, LowerError>`. Materialises
  `s4 = 0.0` once if any `PostOp::Relu` in fused_post_ops. Emits
  `fmax s0, s0, s4` between bias-add and store. `_ =>` arm returns
  `LowerError::UnsupportedPostOp` (new variant for #[non_exhaustive]
  PostOp).
- CLI: refactored arg-parsing into `parse_compile_args` stateful parser.
  New `--no-fuse` flag. Default mode applies `passes::run_pipeline`
  between `ir::build` and `profile.lower`; `--no-fuse` skips. `note:`
  lines emit to stderr (strict stdout/stderr discipline: stdout = asm only,
  pipeable to `cc`).
- Integration test `fused_vs_unfused_classifier_match_numerically`
  exercises classifier.nfl (2 internal fusions) on both paths,
  asserts `assert_eq!` (bit-exact) on outputs. Existing M4b integration
  tests switched to default-fused path.
- 3 CLI smoke tests via `Command::new(env!("CARGO_BIN_EXE_nflc"))`:
  default-runs-fusion, --no-fuse-skips, unknown-flag-rejected.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-04-m5a-kernel-fusion-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-04-m5a-kernel-fusion.md` (11 tasks, 11
commits).

### Pre-decided architectural call
> **Fusion lives at UIR-pass level, not codegen-time peephole.** Two
> reasons (per user during brainstorming): visibility (consumer counts
> visible only on UIR — Linear→Relu fusion is safe iff Linear has one
> consumer, which is invisible to a peephole walking codegen dispatch
> arms) + profile isolation (`PROJECT_SPEC.md` design principle 3 —
> profiles consume already-fused graphs and emit accordingly; the
> fusion logic itself is profile-agnostic).
>
> Right separation of concerns: UIR-passes decide *what* fuses,
> codegen decides *how* to emit fused ops.

### Problems encountered
- (Fill in real issues found during implementation. If none: "None —
  implementation followed the plan straight through.")

### Known tech debt (carried forward)
1. `EliminateDropout` pass deferred to M5b. The dropout-as-noop alias
   in `buffer.rs::assign_buffers` (M4b) continues to handle dropout at
   profile level; M5b moves removal up to UIR-pass.
2. `linear[bias=true] → relu` fusion deferred to M5b. M5a's pass
   condition explicitly excludes `linear_has_bias` candidates.
3. `--passes=X,Y` filter syntax deferred to M5b. M5a only has the
   binary `--no-fuse` flag; `name()` foundation is in place.
4. Profile guide doc updates deferred to M5c. The fusion section,
   asm patterns, and CLI flag docs land in `docs/profile_guide/arm64.md`
   when M5c closes M5.
5. Snapshot tests via `insta` not introduced in M5a (substring asserts
   sufficient at this scope).

### Next step
**Milestone 5a complete.** Recovers M4a's in-place relu performance via
fusion infrastructure. M5b adds bias-aware fusion + `EliminateDropout` +
`--passes=X,Y` filter. M5c closes M5 with profile guide doc updates and
PROJECT_SPEC milestone close-out.

Brainstorming for M5b runs in a fresh worktree once main is updated
post-M5a-merge.
```

(Keep all existing entries intact; only insert above the most recent.)

- [ ] **Step 3: Update `CLAUDE.md` "Current Status"**

Find the existing block and replace its body:

```markdown
**Milestone 5a complete.** UIR-pass infrastructure shipped: `compiler::passes`
module with `UirPass` trait, `default_pipeline()`, `run_pipeline()`, and
`FuseLinearRelu` — the first fusion pass. Pass turns `Linear (no bias=true,
single consumer) → Relu` into `Linear { fused_post_ops: [Relu] }` with the
Relu node removed; profile/arm64 emits inline `fmax s0, s0, s4` before store
(recovers M4a's in-place relu performance).

CLI gains `--no-fuse` flag for verification. Strict stdout/stderr discipline:
stdout = asm only (pipeable to `cc`); stderr = `note:`/`error:` diagnostics.
`fused_vs_unfused_classifier_match_numerically` integration test confirms
fusion preserves numerics bit-exactly via `assert_eq!`.

Op coverage unchanged from M4 (linear ± bias, relu, dropout, softmax). Stack
allocation, non-leaf prologue, label namespacing — all unchanged.
**170 tests passing.** Build/clippy/fmt clean. CI green.

The immediate next step is **Milestone 5b — bias-aware fusion +
EliminateDropout pass**: lift the M5a `linear_has_bias` restriction (so
`linear[bias=true] → relu` fuses with bias-add inline), add
`EliminateDropout` pass (removes dropout from UIR using same NodeId-remap
mechanism), introduce `--passes=X,Y` CLI filter syntax. After M5b: `compiler::passes`
has 2 passes; profile guide doc updates land in M5c.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status
git commit -m "chore(m5a): close Milestone 5a — kernel fusion (linear → relu) shipped

Per spec §12 acceptance criteria — all met:
- cargo build/clippy/fmt --check clean across workspace
- 170 tests passing (baseline 148 + 22 new: 9 pass + 3 pipeline + 3 profile
  + 3 CLI + 1 integration + Display tests)
- CLI smoke positive (default fused) and --no-fuse (skip) both produce
  expected asm shape
- Stdout/stderr discipline verified
- All 5 M3 fixtures + M4a fixture compile under both modes
- Module-level doc-comment in compiler::passes covers what passes are,
  how to add new ones, why functional

DEVLOG documents the pre-decided architectural call (fusion at UIR-pass
level, not peephole) plus the M5a → M5b → M5c slicing roadmap.

CLAUDE.md Current Status reflects M5a complete; M5b (bias-aware fusion +
EliminateDropout + --passes filter) as next.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 11, M5a is complete by spec §12 acceptance criteria:

1. ✅ `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check` — all exit 0.
2. ✅ Pre-M5a tests preserved.
3. ✅ Pass-level unit tests pass (9 in `fuse_linear_relu`, 3 in `passes/tests.rs`).
4. ✅ Profile asm-level unit tests pass (3 in `profiles/arm64/src/tests.rs`).
5. ✅ Integration test `fused_vs_unfused_classifier_match_numerically` passes (or skips on non-aarch64).
6. ✅ CLI smoke positive (default).
7. ✅ CLI smoke positive (--no-fuse).
8. ✅ Stdout/stderr discipline verified.
9. ✅ All M3 fixtures + M4a fixture compile under both modes; M4b integration tests adapted to default-fused.
10. ✅ Module-level doc-comment in `compiler::passes`.

**After all tasks pass:** push `claude/m5-kernel-fusion` and open a PR against `main`. Title: "Implement Milestone 5a: kernel fusion pass (linear → relu) — UIR-pass framework". After merge, M5a closes; M5b begins with a fresh `superpowers:brainstorming` cycle.
