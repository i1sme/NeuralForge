# Milestone 6 — Attention-Pattern Fusion (`linear → softmax`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add row-wise softmax fusion to NeuralForge's M5 kernel-fusion pipeline. After M6, `linear → softmax` and `linear[bias=true] → softmax` patterns fuse into a single `emit_linear` call that materialises the matmul row, then runs three softmax passes (max → exp+sum → normalize) in-place — no cross-buffer round-trip, no separate softmax function call.

**Architecture:** Third `PostOp::SoftmaxRow` variant on the existing `PostOp` enum. New `FuseLinearSoftmax` pass parallel to `FuseLinearRelu` (registered third in `default_pipeline`). RowWise asm branch in `emit_linear` complements the existing Elementwise (`Relu`) branch — the dispatch lives in `emit_linear` itself. Test-helper extraction (`compiler/src/ir/test_utils.rs`) lands as the four-strikes trigger fires before the first M6 unit test is written.

**Tech Stack:** Rust 2021 edition, std-only at runtime (`libloading` and `cc` test-only dev-deps). Workspace: `compiler` (lib), `nflc` (bin), `profiles/arm64` (lib). Target: AArch64 / Apple Silicon. AAPCS64 calling convention.

**Spec:** [`docs/superpowers/specs/2026-05-05-m6-attention-fusion-design.md`](../specs/2026-05-05-m6-attention-fusion-design.md). All §-references in this plan refer to the spec.

---

## Pre-implementation findings

A reading pass over the M5b-current code (commits up through `c247dfa`) confirmed the spec's load-bearing assumptions and resolved one open contingency:

1. **§8 asm-sketch validates against `profiles/arm64/src/ops/softmax.rs::emit_softmax`.** s8 holds row-max, s9 holds row-sum, both callee-saved across `bl _expf`. x22/x23 carry src/dst pointers (also callee-saved). `compute_callee_saved` already saves d8/d9 + x19–x23 when any Softmax node is present; `compute_is_leaf` already returns `false` in the same condition.
2. **`tests/fixtures/classifier.nfl` final layer (`linear[output] -> softmax`) has `bias=false` (default).** M6 therefore MUST add a new fixture `tests/fixtures/softmax_with_bias.nfl` for the bias-aware FFI path. Spec's §13 R3 anticipated this.
3. **Test helpers (`input_node`, `op_node`) already exist as inline helpers in `compiler/src/passes/eliminate_dropout.rs`.** Extraction is a *promotion* (move to `compiler/src/ir/test_utils.rs`, expose `pub(crate)`) and a one-test migration in `compiler/src/passes/tests.rs::pipeline_eliminates_dropout_before_fusing_linear_relu`. `fuse_linear_relu` tests use the parser (`build("model M …")`) and don't need migration.
4. **Both `compute_is_leaf` and `compute_callee_saved` currently detect leaf-ness via `StdOp::Softmax` presence.** After M6 fusion, the standalone Softmax disappears but `bl _expf` lives inside the fused Linear. Both detectors must be extended to also inspect `fused_post_ops` for `PostOp::SoftmaxRow`.

These findings inform Tasks 1, 5, and 7. No spec amendment required.

---

## File map

**New files:**
- `compiler/src/ir/test_utils.rs` — `pub(crate)` UIR-construction helpers shared by pass tests.
- `compiler/src/passes/fuse_linear_softmax.rs` — the new pass module + inline tests.
- `tests/fixtures/softmax_with_bias.nfl` — fixture exercising the `linear[bias=true] → softmax` path.

**Modified files (compiler):**
- `compiler/src/ir/mod.rs` — declare `pub(crate) mod test_utils` (cfg(test) only).
- `compiler/src/ir/types.rs` — add `PostOp::SoftmaxRow` variant + `Display` arm.
- `compiler/src/passes/mod.rs` — add `pub mod fuse_linear_softmax;`, extend `default_pipeline()` to three passes.
- `compiler/src/passes/eliminate_dropout.rs` — replace inline helpers with imports from `crate::ir::test_utils`.
- `compiler/src/passes/tests.rs` — update `default_pipeline_is_canonical_order`; add `pipeline_eliminates_dropout_before_fusing_linear_softmax`; migrate `pipeline_eliminates_dropout_before_fusing_linear_relu` to use shared helpers.

**Modified files (profile arm64):**
- `profiles/arm64/src/buffer.rs` — extend `compute_is_leaf` and `compute_callee_saved` to detect `PostOp::SoftmaxRow` in any node's `fused_post_ops`.
- `profiles/arm64/src/ops/linear.rs` — add RowWise dispatch branch covering Phases 1–4. Surrounding Elementwise branch (existing M5b shape) untouched.
- `profiles/arm64/tests/integration.rs` — add `fused_vs_unfused_softmax_match_numerically`; harmonise three FFI tests' `params_floats` check from `debug_assert_eq!` → `assert_eq!` (OQ-5).

**Modified files (CLI / docs / status):**
- `nflc/tests/cli_compile.rs` — add `compile_with_passes_filter_only_fuse_linear_softmax_runs`.
- `docs/profile_guide/arm64.md` — §3 supported-ops row, new §4.10 (Fused linear → softmax row-wise), §5 errors annotation, §8 Limitations rewrite.
- `docs/language_reference/uir.md` — §2 `NodeKind::Op` mention `SoftmaxRow`.
- `PROJECT_SPEC.md` — milestones table M6 row → "complete".
- `CLAUDE.md` — "Current Status" rewrite reflecting M6.
- `DEVLOG.md` — new entry per project documentation protocol.

---

## Task overview

| # | Task | Approx. size |
|---|------|--------------|
| 1 | Test-helper extraction (promote + migrate cross-pass test) | small |
| 2 | `PostOp::SoftmaxRow` variant + Display | small |
| 3 | `FuseLinearSoftmax` pass module (5 unit tests + impl) | medium |
| 4 | Pipeline integration (default_pipeline + canonical-order test + cross-pass test 6) | small |
| 5 | arm64 leaf-detection + callee-saved updates | small |
| 6 | arm64 RowWise emit branch in `emit_linear` (Phases 1–4) | medium |
| 7 | `softmax_with_bias.nfl` fixture + FFI integration test | medium |
| 8 | OQ-5 `assert_eq!` harmonisation across three FFI tests | trivial |
| 9 | CLI smoke for `--passes fuse_linear_softmax` | small |
| 10 | Documentation updates (arm64.md + uir.md) | small |
| 11 | Closeout (DEVLOG + PROJECT_SPEC + CLAUDE.md + holistic review) | small |

**Sequencing constraints:**
- Task 1 must finish before any later task that constructs UIR by hand (Tasks 3, 4).
- Task 2 must finish before Task 3 (the pass references `PostOp::SoftmaxRow`).
- Task 4 depends on Task 3 (default_pipeline references the new pass).
- Task 5 must finish before Task 6 (compute_is_leaf detection is needed for fused softmax to assemble correctly).
- Task 6 must finish before Task 7 (FFI test exercises fused asm).
- Task 8 may interleave with Task 7 (operates on the same file).
- Tasks 9, 10 may run after Task 7.
- Task 11 is last.

---

## Task 1: Test-helper extraction (`compiler/src/ir/test_utils.rs`)

**Spec ref:** §10. Order of operations: extract first, migrate the M5b test that hand-builds verbose `Node` literals, *then* later tasks use the helpers from the start.

**Files:**
- Create: `compiler/src/ir/test_utils.rs`
- Modify: `compiler/src/ir/mod.rs`
- Modify: `compiler/src/passes/eliminate_dropout.rs`
- Modify: `compiler/src/passes/tests.rs:pipeline_eliminates_dropout_before_fusing_linear_relu`

- [ ] **Step 1: Read the existing inline helpers**

Read `compiler/src/passes/eliminate_dropout.rs`'s `#[cfg(test)] mod tests` block, find the `fn input_node(...)` and `fn op_node(...)` helpers. Note their exact signatures and bodies — these are the API to extract.

- [ ] **Step 2: Create `compiler/src/ir/test_utils.rs` with the promoted helpers**

```rust
//! Shared UIR-construction helpers for pass tests.
//!
//! Use these instead of hand-rolling `Node` literals. The functions construct
//! `Node`s with `source_span: Span::new(1, 1)` and `fused_post_ops: vec![]`
//! (when applicable) — defaults that suit pass tests where the source span
//! is irrelevant and post-ops are populated by the pass under test.

#![cfg(test)]

use crate::ast::Span;
use crate::ir::stdlib::StdOp;
use crate::ir::types::{
    AttrValue, Node, NodeId, NodeKind, OpAttr, Shape, Type,
};

/// Construct an `Input` node with the given name and shape.
pub(crate) fn input_node(name: &str, shape: Vec<u64>) -> Node {
    Node {
        kind: NodeKind::Input { name: name.into() },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(shape),
        },
        source_span: Span::new(1, 1),
    }
}

/// Construct an `Op` node with the given op kind, operand ids, attributes,
/// and output shape. `fused_post_ops` starts empty.
pub(crate) fn op_node(
    op: StdOp,
    operands: Vec<NodeId>,
    attrs: Vec<OpAttr>,
    shape: Vec<u64>,
) -> Node {
    Node {
        kind: NodeKind::Op {
            op,
            operands,
            attrs,
            fused_post_ops: vec![],
        },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(shape),
        },
        source_span: Span::new(1, 1),
    }
}

/// Convenience for the common `out_dim` integer attribute on `Linear`.
pub(crate) fn out_dim_attr(value: i64) -> OpAttr {
    OpAttr {
        name: "out_dim".into(),
        value: AttrValue::Integer(value),
    }
}

/// Convenience for the `rate` float attribute on `Dropout`.
pub(crate) fn rate_attr(value: f64) -> OpAttr {
    OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(value),
    }
}
```

(If the M5b inline helpers' signatures differ from the above, mirror them exactly — the reader of this plan should treat the existing M5b helpers as the source of truth and adapt the field-list above accordingly.)

- [ ] **Step 3: Wire `test_utils` into the `ir` module**

In `compiler/src/ir/mod.rs`, add the module declaration. Place it among existing `mod` lines:

```rust
#[cfg(test)]
pub(crate) mod test_utils;
```

- [ ] **Step 4: Run `cargo build --workspace` to confirm the new module compiles**

```sh
cargo build --workspace
```

Expected: clean build, no warnings.

- [ ] **Step 5: Migrate `eliminate_dropout.rs` to use shared helpers**

In `compiler/src/passes/eliminate_dropout.rs::tests`:
- DELETE the inline `fn input_node` and `fn op_node` (and any other shadowed helpers).
- ADD `use crate::ir::test_utils::{input_node, op_node, rate_attr, out_dim_attr};` near the top of the test module.
- All existing test bodies that called the old `input_node`/`op_node` functions now resolve to the shared ones — no test-body changes needed.

- [ ] **Step 6: Run the eliminate_dropout test suite**

```sh
cargo test -p compiler --lib passes::eliminate_dropout
```

Expected: all 8 existing eliminate_dropout tests pass (counts unchanged from M5b baseline).

- [ ] **Step 7: Migrate `pipeline_eliminates_dropout_before_fusing_linear_relu` to use shared helpers**

In `compiler/src/passes/tests.rs`, the existing test reads (verbatim):

```rust
#[test]
fn pipeline_eliminates_dropout_before_fusing_linear_relu() {
    use crate::ast::Span;
    use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, PostOp, Shape, Type};
    use crate::ir::StdOp;
    use crate::UirModel;

    let span = Span::new(1, 1);
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            Node { kind: NodeKind::Input { name: "x".into() }, ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) }, source_span: span },
            Node { kind: NodeKind::Op { op: StdOp::Linear, operands: vec![0], attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }], fused_post_ops: vec![] }, ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) }, source_span: span },
            Node { kind: NodeKind::Op { op: StdOp::Dropout, operands: vec![1], attrs: vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.5) }], fused_post_ops: vec![] }, ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) }, source_span: span },
            Node { kind: NodeKind::Op { op: StdOp::Relu, operands: vec![2], attrs: vec![], fused_post_ops: vec![] }, ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) }, source_span: span },
        ],
        inputs: vec![0],
        output: 3,
        source_span: span,
    };
    let uir = Uir { models: vec![model] };

    let out = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 2);
    assert!(matches!(m.nodes[0].kind, NodeKind::Input { .. }));
    let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    assert_eq!(m.output, 1);
    assert_eq!(m.inputs, vec![0]);
}
```

Replace it with:

```rust
#[test]
fn pipeline_eliminates_dropout_before_fusing_linear_relu() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 2]),
            op_node(StdOp::Relu, vec![2], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 3,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 2);
    assert!(matches!(m.nodes[0].kind, NodeKind::Input { .. }));
    let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    assert_eq!(m.output, 1);
    assert_eq!(m.inputs, vec![0]);
}
```

- [ ] **Step 8: Run the cross-pass tests**

```sh
cargo test -p compiler --lib passes::tests
```

Expected: all 5 cross-pass tests pass (`default_pipeline_is_canonical_order`, `run_pipeline_threads_uir_through_passes`, `empty_pipeline_returns_input_clone`, `pipeline_halts_on_first_error_and_propagates`, `pipeline_eliminates_dropout_before_fusing_linear_relu`).

- [ ] **Step 9: Run the full workspace tests + clippy**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green; total test count unchanged from M5c baseline.

- [ ] **Step 10: Commit**

```sh
git add compiler/src/ir/test_utils.rs compiler/src/ir/mod.rs compiler/src/passes/eliminate_dropout.rs compiler/src/passes/tests.rs
git commit -m "refactor(m6/test-utils): extract shared UIR helpers; migrate cross-pass test

Promotes the inline input_node()/op_node() helpers from
eliminate_dropout.rs to compiler/src/ir/test_utils.rs (pub(crate),
cfg(test) only). Migrates pipeline_eliminates_dropout_before_fusing_linear_relu
in passes/tests.rs to use them — that test was the worst offender for
verbose Node literals.

Per the M5c carry-forward debt list, the four-strikes trigger fires
on the first M6 hand-built UIR test; this refactor lands it before
that test is written, so M6's FuseLinearSoftmax tests use the helpers
from the start (rather than getting written with boilerplate and then
retroactively migrated).

No behavior change. fuse_linear_relu tests use the parser (build()) and
are left untouched."
```

---

## Task 2: `PostOp::SoftmaxRow` variant + Display

**Spec ref:** §4. Adds the third variant. Cascade arms in profile already use a wildcard (`_ => UnsupportedPostOp { ... }`), so they automatically protect against the new variant — but `Display` must be extended.

**Files:**
- Modify: `compiler/src/ir/types.rs:PostOp` (enum + Display impl)

- [ ] **Step 1: Read the current `PostOp` definition and Display impl**

In `compiler/src/ir/types.rs`, locate the `PostOp` enum and its `impl std::fmt::Display`. Confirm shape (today):

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOp {
    Relu,
}

impl std::fmt::Display for PostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PostOp::Relu => "relu",
        };
        write!(f, "{}", name)
    }
}
```

- [ ] **Step 2: Add a failing Display test for the new variant**

In `compiler/src/ir/tests.rs` (or wherever PostOp Display is tested today; if no test exists yet, add a new one in the most appropriate test module — likely `compiler/src/ir/tests.rs` if it exists, or create a `#[cfg(test)] mod` in `types.rs` itself). Add:

```rust
#[test]
fn post_op_softmax_row_displays_as_softmax_row() {
    use crate::ir::types::PostOp;
    assert_eq!(format!("{}", PostOp::SoftmaxRow), "softmax_row");
}
```

- [ ] **Step 3: Run the test to verify it fails**

```sh
cargo test -p compiler post_op_softmax_row_displays
```

Expected: FAIL — `error[E0599]: no variant or associated item named `SoftmaxRow` found for enum `PostOp``.

- [ ] **Step 4: Add the `SoftmaxRow` variant and extend Display**

In `compiler/src/ir/types.rs`, modify the `PostOp` enum:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOp {
    Relu,
    /// Row-wise softmax. Emit shape is structurally different from `Relu` —
    /// `emit_linear` materialises the full row first, then runs three sweeps
    /// (max → exp+sum → normalize) in-place. See `arm64.md` §4.10.
    SoftmaxRow,
}

impl std::fmt::Display for PostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PostOp::Relu => "relu",
            PostOp::SoftmaxRow => "softmax_row",
        };
        write!(f, "{}", name)
    }
}
```

- [ ] **Step 5: Run the failing test, expect pass**

```sh
cargo test -p compiler post_op_softmax_row_displays
```

Expected: PASS.

- [ ] **Step 6: Run the workspace test suite + clippy**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. The `profiles/arm64::ops::linear::emit_linear` post-op match has a wildcard arm (`_ => UnsupportedPostOp`), so adding `SoftmaxRow` does NOT break compilation; the wildcard catches it as an "unsupported post-op" until Task 6 adds the RowWise emit branch. Total test count = M5c baseline + 1.

- [ ] **Step 7: Commit**

```sh
git add compiler/src/ir/types.rs compiler/src/ir/tests.rs
git commit -m "feat(m6/postop): add PostOp::SoftmaxRow variant + Display

Third variant on the #[non_exhaustive] PostOp enum. Display renders
as 'softmax_row' (lowercase, snake_case to match Relu's 'relu').

The arm64 profile's emit_linear post-op match has a wildcard arm
returning LowerError::UnsupportedPostOp, so this commit does not
break codegen — fused-softmax-row paths simply hit the wildcard
until Task 6 adds the RowWise emit branch.

Variant doc-comment cross-references arm64.md §4.10 for the
structural difference from Elementwise post-ops (which Task 10
will add to that file)."
```

---

## Task 3: `FuseLinearSoftmax` pass module

**Spec ref:** §5 + §9 unit tests 1–5. TDD: write test 1, watch fail, implement minimum to pass; repeat for tests 2–5.

**Files:**
- Create: `compiler/src/passes/fuse_linear_softmax.rs`
- Modify: `compiler/src/passes/mod.rs` (add `pub mod fuse_linear_softmax;`)

- [ ] **Step 1: Stub the new module + register it**

Create `compiler/src/passes/fuse_linear_softmax.rs` with the minimum that compiles:

```rust
//! `FuseLinearSoftmax` UIR pass.
//!
//! Fuses `linear → softmax` and `linear[bias=true] → softmax` patterns
//! by appending `PostOp::SoftmaxRow` to the Linear's `fused_post_ops`
//! and removing the Softmax node. The arm64 profile's RowWise emit
//! branch (see `arm64.md` §4.10) consumes the fused result.
//!
//! See spec §5 for the full victim criteria.

use crate::ir::types::{NodeKind, PostOp};
use crate::ir::StdOp;
use crate::ir::Uir;
use crate::passes::{PassError, UirPass};

pub struct FuseLinearSoftmax;

impl UirPass for FuseLinearSoftmax {
    fn name(&self) -> &str {
        "fuse_linear_softmax"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        // Stub: pass-through. Real implementation arrives via TDD in
        // subsequent steps of this task.
        Ok(uir.clone())
    }
}

#[cfg(test)]
mod tests {
    // Tests filled in by the steps below.
}
```

In `compiler/src/passes/mod.rs`, add (in alphabetical order with other pass modules):

```rust
pub mod fuse_linear_softmax;
```

Run `cargo build --workspace`. Expected: clean.

- [ ] **Step 2: Write failing test 1 — `fuses_linear_softmax_no_bias`**

In `compiler/src/passes/fuse_linear_softmax.rs::tests`:

```rust
#[test]
fn fuses_linear_softmax_no_bias() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::ir::Uir;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 2,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 2);
    let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::SoftmaxRow]);
    assert_eq!(m.output, 1);
}
```

- [ ] **Step 3: Run test 1, expect fail**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::fuses_linear_softmax_no_bias
```

Expected: FAIL — pass is identity (pass-through), so `nodes.len()` is 3 not 2 and `fused_post_ops` is empty.

- [ ] **Step 4: Implement minimal pass that passes test 1**

In `compiler/src/passes/fuse_linear_softmax.rs::impl UirPass::run`, replace the stub body. Mirror `FuseLinearRelu`'s 3-step rebuild (read it as the canonical reference):

```rust
fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
    let mut new_models = Vec::with_capacity(uir.models.len());

    for model in &uir.models {
        // Step 1: count consumers of each node.
        let mut consumer_count = vec![0usize; model.nodes.len()];
        for node in &model.nodes {
            if let NodeKind::Op { operands, .. } = &node.kind {
                for &operand_id in operands {
                    consumer_count[operand_id] += 1;
                }
            }
        }

        // Step 2: identify Softmax victims whose producer is a fusable Linear.
        let mut is_victim = vec![false; model.nodes.len()];
        let mut producer_of: Vec<Option<usize>> = vec![None; model.nodes.len()];
        for (i, node) in model.nodes.iter().enumerate() {
            if let NodeKind::Op { op: StdOp::Softmax, operands, .. } = &node.kind {
                if operands.len() != 1 { continue; }
                let producer_id = operands[0];
                let producer = &model.nodes[producer_id];
                let NodeKind::Op { op: StdOp::Linear, fused_post_ops, .. } = &producer.kind else { continue; };
                if consumer_count[producer_id] != 1 { continue; }
                if !fused_post_ops.is_empty() { continue; }
                is_victim[i] = true;
                producer_of[i] = Some(producer_id);
            }
        }

        // Step 3: rebuild + remap.
        let mut id_map: Vec<usize> = vec![0; model.nodes.len()];
        let mut new_nodes = Vec::with_capacity(model.nodes.len());
        for (i, node) in model.nodes.iter().enumerate() {
            if is_victim[i] {
                // Map victim id to its producer's new id (set when producer was visited).
                let producer_id = producer_of[i].expect("victim has producer");
                id_map[i] = id_map[producer_id];
                continue;
            }
            id_map[i] = new_nodes.len();
            let mut new_node = node.clone();
            // Remap operands in Op nodes through id_map (only valid for already-visited operands).
            if let NodeKind::Op { operands, fused_post_ops, .. } = &mut new_node.kind {
                for operand_id in operands.iter_mut() {
                    *operand_id = id_map[*operand_id];
                }
                // If this is a Linear that produced a victim Softmax, append PostOp::SoftmaxRow.
                let this_old_id = i;
                let produced_victim = is_victim.iter().enumerate().any(|(victim_old_id, &is_v)| {
                    is_v && producer_of[victim_old_id] == Some(this_old_id)
                });
                if produced_victim {
                    fused_post_ops.push(PostOp::SoftmaxRow);
                }
            }
            new_nodes.push(new_node);
        }

        let new_inputs: Vec<usize> = model.inputs.iter().map(|&id| id_map[id]).collect();
        let new_output = id_map[model.output];

        new_models.push(crate::UirModel {
            name: model.name.clone(),
            nodes: new_nodes,
            inputs: new_inputs,
            output: new_output,
            source_span: model.source_span,
        });
    }

    Ok(Uir { models: new_models })
}
```

(If the existing `FuseLinearRelu` has a subtly different rebuild order — e.g. processes producers and victims in a different sequence — adapt to match. The important properties are: producer's new id is known when the victim is encountered, and the victim's consumers see the producer's new id when they look up `id_map[victim_id]`.)

- [ ] **Step 5: Run test 1, expect pass**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::fuses_linear_softmax_no_bias
```

Expected: PASS.

- [ ] **Step 6: Write failing test 2 — `fuses_linear_softmax_with_bias`**

```rust
#[test]
fn fuses_linear_softmax_with_bias() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{AttrValue, NodeKind, OpAttr, PostOp};
    use crate::ir::StdOp;
    use crate::ir::Uir;
    use crate::UirModel;

    let bias_attr = OpAttr { name: "bias".into(), value: AttrValue::Boolean(true) };
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2), bias_attr.clone()], vec![2, 2]),
            op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 2,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 2);
    let NodeKind::Op { op, attrs, fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(matches!(op, StdOp::Linear));
    // bias attr is preserved on the Linear
    assert!(attrs.iter().any(|a| a.name == "bias"));
    assert_eq!(fused_post_ops, &vec![PostOp::SoftmaxRow]);
}
```

(Adjust `AttrValue::Boolean` to whatever shape NFL's bias attr actually uses — peek at how `FuseLinearRelu`'s bias-aware test or the parser tests construct it.)

- [ ] **Step 7: Run test 2**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::fuses_linear_softmax_with_bias
```

Expected: PASS (the implementation in Step 4 is bias-agnostic — it does not inspect or filter on bias).

- [ ] **Step 8: Write failing test 3 — `does_not_fuse_when_post_ops_already_present`**

```rust
#[test]
fn does_not_fuse_when_post_ops_already_present() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{NodeKind, PostOp, Shape, Type};
    use crate::ir::StdOp;
    use crate::ir::Uir;
    use crate::UirModel;

    // Construct a Linear that already has fused_post_ops = [Relu], then a Softmax consumer.
    let mut linear = op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]);
    if let NodeKind::Op { fused_post_ops, .. } = &mut linear.kind {
        fused_post_ops.push(PostOp::Relu);
    }

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            linear,
            op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 2,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    // No fusion: graph shape preserved (3 nodes), Linear still has only [Relu], Softmax intact.
    assert_eq!(m.nodes.len(), 3);
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    assert!(matches!(m.nodes[2].kind, NodeKind::Op { op: StdOp::Softmax, .. }));
}
```

- [ ] **Step 9: Run test 3**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::does_not_fuse_when_post_ops_already_present
```

Expected: PASS — the implementation already includes `if !fused_post_ops.is_empty() { continue; }` (criterion 4).

- [ ] **Step 10: Write failing test 4 — `does_not_fuse_multi_consumer_linear`**

```rust
#[test]
fn does_not_fuse_multi_consumer_linear() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::ir::Uir;
    use crate::UirModel;

    // Linear feeds both Softmax and Relu — multi-consumer, must not fuse.
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
            op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        // Pick one as output; multi-consumer is the structural property.
        output: 2,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 4);
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(fused_post_ops.is_empty(), "Linear with multi-consumer must not be fused");
}
```

- [ ] **Step 11: Run test 4**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::does_not_fuse_multi_consumer_linear
```

Expected: PASS — the implementation has `if consumer_count[producer_id] != 1 { continue; }` (criterion 1).

- [ ] **Step 12: Write failing test 5 — `identity_when_no_softmax`**

```rust
#[test]
fn identity_when_no_softmax() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::NodeKind;
    use crate::ir::StdOp;
    use crate::ir::Uir;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 2,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    assert_eq!(m.nodes.len(), 3);
    assert!(matches!(m.nodes[2].kind, NodeKind::Op { op: StdOp::Relu, .. }));
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(fused_post_ops.is_empty());
}
```

- [ ] **Step 13: Run test 5**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::identity_when_no_softmax
```

Expected: PASS — no Softmax node, no victims, pass returns the input unchanged.

- [ ] **Step 14: Run all 5 unit tests + workspace clippy + tests**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p compiler --lib passes::fuse_linear_softmax
cargo test --workspace
```

Expected: 5 new tests PASS in `fuse_linear_softmax::tests`; full workspace tests still pass. Total test count = M5c baseline + 1 (Display test from Task 2) + 5 (this task) = baseline + 6.

- [ ] **Step 15: Commit**

```sh
git add compiler/src/passes/fuse_linear_softmax.rs compiler/src/passes/mod.rs
git commit -m "feat(m6/passes): FuseLinearSoftmax pass — bias-aware from day one

New UIR pass that fuses linear → softmax (and linear[bias=true] → softmax)
patterns by appending PostOp::SoftmaxRow to the Linear's fused_post_ops
and dropping the Softmax node. Mirrors FuseLinearRelu's structure:
single file with inline #[cfg(test)] mod tests, three-step rebuild
(victim identification → rebuild with id-remap → remap inputs/output).

Five unit tests pin the victim criteria:
  1. fuses_linear_softmax_no_bias
  2. fuses_linear_softmax_with_bias
  3. does_not_fuse_when_post_ops_already_present (criterion 4 — guards
     against [Relu, SoftmaxRow] stacking, whose emit shape is M7+ scope)
  4. does_not_fuse_multi_consumer_linear (criterion 1)
  5. identity_when_no_softmax

Pass not yet wired into default_pipeline — that lands in Task 4 of the
M6 plan together with the canonical-order test update."
```

---

## Task 4: Pipeline integration

**Spec ref:** §6 + §9 test 6.

**Files:**
- Modify: `compiler/src/passes/mod.rs:default_pipeline`
- Modify: `compiler/src/passes/tests.rs:default_pipeline_is_canonical_order`
- Modify: `compiler/src/passes/tests.rs` (add new pipeline test)

- [ ] **Step 1: Update `default_pipeline_is_canonical_order` to expect 3-element list (failing)**

In `compiler/src/passes/tests.rs`, replace the assertion:

```rust
#[test]
fn default_pipeline_is_canonical_order() {
    let pipeline = default_pipeline();
    let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
    assert_eq!(
        names,
        vec!["eliminate_dropout", "fuse_linear_relu", "fuse_linear_softmax"],
        "default_pipeline must run eliminate_dropout, then fuse_linear_relu, then fuse_linear_softmax; got: {:?}",
        names
    );
}
```

- [ ] **Step 2: Run the test, expect fail**

```sh
cargo test -p compiler --lib passes::tests::default_pipeline_is_canonical_order
```

Expected: FAIL — current `default_pipeline()` returns 2 passes; assertion sees `["eliminate_dropout", "fuse_linear_relu"]`.

- [ ] **Step 3: Extend `default_pipeline()` to register `FuseLinearSoftmax`**

In `compiler/src/passes/mod.rs::default_pipeline`:

```rust
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![
        Box::new(eliminate_dropout::EliminateDropout),
        Box::new(fuse_linear_relu::FuseLinearRelu),
        Box::new(fuse_linear_softmax::FuseLinearSoftmax),
    ]
}
```

- [ ] **Step 4: Re-run, expect pass**

```sh
cargo test -p compiler --lib passes::tests::default_pipeline_is_canonical_order
```

Expected: PASS.

- [ ] **Step 5: Add failing test 6 — `pipeline_eliminates_dropout_before_fusing_linear_softmax`**

In `compiler/src/passes/tests.rs` (cross-pass file, mirroring M5b's location convention):

```rust
#[test]
fn pipeline_eliminates_dropout_before_fusing_linear_softmax() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 2]),
            op_node(StdOp::Softmax, vec![2], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 3,
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir { models: vec![model] };

    let out = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let m = &out.models[0];

    // EliminateDropout collapses the chain to linear → softmax;
    // FuseLinearSoftmax then fuses, producing one Linear with [SoftmaxRow].
    assert_eq!(m.nodes.len(), 2);
    assert!(matches!(m.nodes[0].kind, NodeKind::Input { .. }));
    let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::SoftmaxRow]);
    assert_eq!(m.output, 1);
    assert_eq!(m.inputs, vec![0]);
}
```

- [ ] **Step 6: Run test 6, expect pass**

```sh
cargo test -p compiler --lib passes::tests::pipeline_eliminates_dropout_before_fusing_linear_softmax
```

Expected: PASS — `EliminateDropout` removes Dropout, then `FuseLinearSoftmax` fuses the resulting `linear → softmax`.

- [ ] **Step 7: Run full workspace tests + clippy**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Total = baseline + 6 (Tasks 2–3) + 1 (this task) = baseline + 7.

- [ ] **Step 8: Commit**

```sh
git add compiler/src/passes/mod.rs compiler/src/passes/tests.rs
git commit -m "feat(m6/pipeline): wire FuseLinearSoftmax into default_pipeline

Extends default_pipeline() to [EliminateDropout, FuseLinearRelu,
FuseLinearSoftmax]. EliminateDropout staying first preserves the
M5b invariant that linear → dropout → softmax collapses before
fusion is attempted.

Updates default_pipeline_is_canonical_order to assert the
three-element list. Adds pipeline_eliminates_dropout_before_fusing_linear_softmax
as a cross-pass integration test in passes/tests.rs (mirroring
the convention M5b set with the linear_relu version — cross-pass
tests live in passes/tests.rs, not in per-pass modules).

CLI --passes / --no-passes plumbing requires no code changes; the
filter reads pass names from default_pipeline() dynamically."
```

---

## Task 5: arm64 leaf-detection + callee-saved updates

**Spec ref:** §7 + §13 R2.

`compute_is_leaf` and `compute_callee_saved` currently inspect for `StdOp::Softmax` to decide whether `bl _expf` is invoked. After fusion, the Softmax disappears but the call lives inside the fused Linear's `fused_post_ops`. Both detectors must check both places.

**Files:**
- Modify: `profiles/arm64/src/buffer.rs:compute_is_leaf`
- Modify: `profiles/arm64/src/buffer.rs:compute_callee_saved`
- Modify: `profiles/arm64/src/buffer.rs::tests` (or `profiles/arm64/src/tests.rs` if leaf detection is tested there)

- [ ] **Step 1: Read the current detection logic**

In `profiles/arm64/src/buffer.rs`, locate `compute_is_leaf` and `compute_callee_saved`. Note exactly how they iterate model nodes and what they pattern-match on. The detection condition today is approximately:

```rust
model.nodes.iter().any(|n| matches!(n.kind, NodeKind::Op { op: StdOp::Softmax, .. }))
```

- [ ] **Step 2: Write a failing test for compute_is_leaf on a fused-softmax-row Linear**

Add to whichever `#[cfg(test)] mod tests` covers `compute_is_leaf`. If unsure, add it to `profiles/arm64/src/buffer.rs::tests`:

```rust
#[test]
fn is_leaf_false_for_fused_softmax_row_linear() {
    use compiler::ir::test_utils::{input_node, op_node, out_dim_attr};
    use compiler::ir::types::{NodeKind, PostOp};
    use compiler::ir::StdOp;
    use compiler::UirModel;

    // Construct a UIR where Softmax has already been fused into a Linear.
    let mut linear = op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]);
    if let NodeKind::Op { fused_post_ops, .. } = &mut linear.kind {
        fused_post_ops.push(PostOp::SoftmaxRow);
    }

    let model = UirModel {
        name: "M".into(),
        nodes: vec![input_node("x", vec![2, 3]), linear],
        inputs: vec![0],
        output: 1,
        source_span: compiler::ast::Span::new(1, 1),
    };

    assert!(!super::compute_is_leaf(&model),
        "a Linear carrying PostOp::SoftmaxRow still calls bl _expf — leaf must be false");
}
```

(If `compute_is_leaf` is not directly testable from outside the module, add the test inside `profiles/arm64/src/buffer.rs` as `#[cfg(test)] mod tests`. Visibility may need a `pub(crate) fn compute_is_leaf` — check the current visibility and adjust if needed.)

Note: the test imports `compiler::ir::test_utils::*`. For this to work, `test_utils` must be accessible to `profiles/arm64`'s test compilation. Since `test_utils` is `#[cfg(test)] pub(crate)`, it is *not* visible to other crates' tests. The simplest workaround is to construct the UIR with a `compiler::parse + compiler::ir::build` from a small NFL string that exercises the Linear+Softmax pattern, then run `FuseLinearSoftmax` on it before passing to `compute_is_leaf`. Rewrite the test:

```rust
#[test]
fn is_leaf_false_for_fused_softmax_row_linear() {
    use compiler::passes::{run_pipeline, default_pipeline};

    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let model = &fused.models[0];

    assert!(!super::compute_is_leaf(model),
        "a Linear carrying PostOp::SoftmaxRow still calls bl _expf — leaf must be false");
}
```

This avoids cross-crate test-helper visibility entirely.

- [ ] **Step 3: Run the test, expect fail**

```sh
cargo test -p profiles-arm64 --lib is_leaf_false_for_fused_softmax_row_linear
```

Expected: FAIL — current `compute_is_leaf` only checks for standalone `StdOp::Softmax`, sees none in the fused UIR, returns `true`.

- [ ] **Step 4: Extend `compute_is_leaf` to detect `PostOp::SoftmaxRow`**

Modify the detection condition. The exact form depends on the existing code shape; a representative replacement:

```rust
pub(crate) fn compute_is_leaf(model: &UirModel) -> bool {
    let calls_expf = model.nodes.iter().any(|n| match &n.kind {
        NodeKind::Op { op: StdOp::Softmax, .. } => true,
        NodeKind::Op { fused_post_ops, .. } => {
            fused_post_ops.iter().any(|po| matches!(po, PostOp::SoftmaxRow))
        }
        _ => false,
    });
    !calls_expf
}
```

Add the `PostOp` import at the top of the file if not already present.

- [ ] **Step 5: Re-run the test, expect pass**

```sh
cargo test -p profiles-arm64 --lib is_leaf_false_for_fused_softmax_row_linear
```

Expected: PASS.

- [ ] **Step 6: Apply the same extension to `compute_callee_saved`**

If `compute_callee_saved` uses similar detection logic — extend identically. Note: today `compute_callee_saved` returns `RegSet { d8_d9: has_softmax, x19_x23: has_softmax }`. After this change, both flags become true whenever EITHER a standalone Softmax exists OR any node has `PostOp::SoftmaxRow` in its `fused_post_ops`.

A representative replacement:

```rust
pub(crate) fn compute_callee_saved(model: &UirModel) -> RegSet {
    let needs_save = model.nodes.iter().any(|n| match &n.kind {
        NodeKind::Op { op: StdOp::Softmax, .. } => true,
        NodeKind::Op { fused_post_ops, .. } => {
            fused_post_ops.iter().any(|po| matches!(po, PostOp::SoftmaxRow))
        }
        _ => false,
    });
    RegSet { d8_d9: needs_save, x19_x23: needs_save }
}
```

- [ ] **Step 7: Add a callee_saved unit test**

```rust
#[test]
fn callee_saved_includes_d8_d9_for_fused_softmax_row() {
    use compiler::passes::{run_pipeline, default_pipeline};

    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let model = &fused.models[0];

    let regs = super::compute_callee_saved(model);
    assert!(regs.d8_d9, "fused-SoftmaxRow Linear needs d8/d9 saved");
    assert!(regs.x19_x23, "fused-SoftmaxRow Linear needs x19-x23 saved");
}
```

- [ ] **Step 8: Run all buffer tests + clippy + workspace tests**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p profiles-arm64
cargo test --workspace
```

Expected: all green. Total = baseline + 7 (prior tasks) + 2 (this task) = baseline + 9.

- [ ] **Step 9: Commit**

```sh
git add profiles/arm64/src/buffer.rs
git commit -m "fix(m6/buffer): leaf + callee-saved detection for fused SoftmaxRow

After M6 fuses linear → softmax, the standalone Softmax node disappears
but the bl _expf call lives inside the fused Linear's fused_post_ops.
Both compute_is_leaf and compute_callee_saved must inspect for
PostOp::SoftmaxRow in addition to standalone StdOp::Softmax — otherwise
the fused linear emits asm that calls bl _expf without saving lr/d8/d9
and crashes at runtime (ABI violation).

Two unit tests pin the new behaviour:
  - is_leaf_false_for_fused_softmax_row_linear
  - callee_saved_includes_d8_d9_for_fused_softmax_row

Both construct fused UIR through the parser + default_pipeline rather
than hand-built helpers (test_utils is pub(crate) compiler-internal,
not visible cross-crate)."
```

---

## Task 6: arm64 RowWise emit branch in `emit_linear`

**Spec ref:** §8 — full asm-sketch.

This is the largest task. It implements all four Phases of the RowWise emit shape inside `emit_linear`. Substring assertions on the emitted asm pin the structural invariants; bit-exact equivalence is verified later in Task 7's FFI integration test.

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs:emit_linear`
- Modify: `profiles/arm64/src/ops/linear.rs::tests` (or wherever `emit_linear` is tested today)

- [ ] **Step 1: Read the current `emit_linear` body and post-op dispatch**

Find the `for post_op in fused_post_ops { match post_op { ... } }` block. Note its exact location (approximately lines 105–120 in the M5b version) and the surrounding context — it sits AFTER the bias-add and BEFORE the `str s0, [..., x4, lsl #2]` final store inside the j-loop.

For RowWise, this in-loop dispatch cannot work. The RowWise branch must be invoked AFTER the j-loop completes for each row, not inside it.

- [ ] **Step 2: Write a failing substring test for the fused-softmax-row asm shape**

In `profiles/arm64/src/ops/linear.rs::tests` (or the closest existing test module):

```rust
#[test]
fn emit_linear_with_softmax_row_post_op_emits_three_phase_softmax() {
    use compiler::passes::{run_pipeline, default_pipeline};

    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let asm = crate::lower(&fused).expect("lower ok");

    // Phase 1 — matmul row + materialise to outbuf — uses the existing
    // M5b Elementwise loop bodies; we don't pin its substring beyond
    // verifying it exists.
    assert!(asm.source.contains("fmadd"), "Phase 1 matmul missing");

    // Phase 2 — row-max scan into s8.
    assert!(asm.source.contains("fmax    s8"), "Phase 2 row-max scan into s8 missing");

    // Phase 3 — exp(x - row_max) accumulated in s9, with bl _expf.
    assert!(asm.source.contains("bl      _expf"), "Phase 3 missing bl _expf");
    assert!(asm.source.contains("fadd    s9"), "Phase 3 sum accumulation in s9 missing");

    // Phase 4 — fdiv normalise.
    assert!(asm.source.contains("fdiv    s0, s0, s9"), "Phase 4 normalise missing");

    // No standalone softmax label — the fused Linear absorbed it.
    assert!(!asm.source.contains(".Lsoftmax_"),
        "fused asm must not contain a separate softmax loop label");
}
```

(Adjust `s0`/`s8`/`s9` register names to whatever the implementation chooses, as long as the spec's invariant holds — s8/s9 callee-saved for row max/sum, s0 working register.)

- [ ] **Step 3: Run the test, expect fail**

```sh
cargo test -p profiles-arm64 emit_linear_with_softmax_row_post_op_emits_three_phase_softmax
```

Expected: FAIL — the existing post-op match has a wildcard returning `LowerError::UnsupportedPostOp`, so `lower()` fails before any asm is emitted.

- [ ] **Step 4: Refactor `emit_linear` into Elementwise vs RowWise dispatch**

The change has two aspects:

**(a) Decide the post-op category before generating loops.** Add a helper at the top of `emit_linear`:

```rust
fn post_op_is_row_wise(po: &PostOp) -> bool {
    matches!(po, PostOp::SoftmaxRow)
}

let has_row_wise = fused_post_ops.iter().any(post_op_is_row_wise);
```

Reject mixed Elementwise+RowWise stacks (criterion 4 in the pass prevents this in practice, but defend against arbitrary callers):

```rust
if has_row_wise && fused_post_ops.len() > 1 {
    return Err(LowerError::UnsupportedPostOp {
        op: "stacked post-ops with RowWise variant".into(),
        span: node_span,
    });
}
```

**(b) Branch the post-op handling.** The current Elementwise inline (`fmax s0, s0, s4` for `Relu`) stays inside the j-loop. RowWise (the only case with `has_row_wise == true` in M6) extends the function with an after-the-j-loop tail. Concretely:

- INSIDE the j-loop, AFTER the bias-add, AFTER the post-op match: if the post-op was Elementwise, the existing inline applies. If it was RowWise (or empty with `has_row_wise == false`), no inline op is emitted; the matmul + bias result is stored as-is.
- AFTER the j-loop closes (still INSIDE the i-loop), if `has_row_wise`: emit the three RowWise phases.

A representative implementation skeleton (adapt to the actual existing structure):

```rust
for post_op in fused_post_ops {
    match post_op {
        PostOp::Relu => s.push_str("    fmax    s0, s0, s4\n"),
        PostOp::SoftmaxRow => {
            // Skipped here — handled after the j-loop closes (RowWise tail).
        }
        _ => return Err(LowerError::UnsupportedPostOp {
            op: post_op.to_string(),
            span: node_span,
        }),
    }
}

s.push_str(&format!("    str     s0, [x12, x4, lsl #2]\n"));
// j-loop tail (existing branch and increment)
s.push_str(&format!("    add     x4, x4, #1\n"));
s.push_str(&format!("    cmp     x4, #{n}\n"));
s.push_str(&format!("    b.lt    .Lmm_j_{linear_idx}\n"));

if has_row_wise {
    emit_row_wise_softmax_tail(&mut s, b, n, linear_idx, /* dst_loc args */);
}

// existing i-loop tail
```

- [ ] **Step 5: Implement the RowWise tail helper**

Add a helper function in `profiles/arm64/src/ops/linear.rs` (or a new sibling `softmax_row.rs` if size dictates):

```rust
/// Emit Phases 2–4 of the RowWise softmax tail.
/// Phase 1 (matmul + bias materialising the row) was emitted by the
/// caller's j-loop; this helper appends the three reduction sweeps,
/// all in-place over the same destination buffer.
fn emit_row_wise_softmax_tail(
    s: &mut String,
    _b: u64,
    n: u64,
    linear_idx: usize,
) {
    // x12 currently holds the dst buffer base; x3 holds i (row index);
    // we re-use them. Compute the row base offset as i * n into x6.
    s.push_str(&format!("    // ----- RowWise softmax tail (linear {linear_idx}) -----\n"));

    // Phase 2: row-max into s8.
    // Initialise from row[0] to avoid materialising -inf.
    s.push_str(&format!("    mov     x4, #0\n")); // j = 0
    s.push_str(&format!("    // load output[i, 0] into s8 (init)\n"));
    s.push_str(&format!("    mov     x6, x3\n"));
    s.push_str(&format!("    mov     x7, #{n}\n"));
    s.push_str(&format!("    mul     x6, x6, x7\n")); // x6 = i * n
    s.push_str(&format!("    add     x7, x12, x6, lsl #2\n")); // x7 = &output[i, 0]
    s.push_str(&format!("    ldr     s8, [x7]\n"));
    s.push_str(&format!("    mov     x4, #1\n")); // j = 1
    s.push_str(&format!(".Lsr_max_{linear_idx}:\n"));
    s.push_str(&format!("    ldr     s0, [x7, x4, lsl #2]\n"));
    s.push_str(&format!("    fmax    s8, s8, s0\n"));
    s.push_str(&format!("    add     x4, x4, #1\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.lt    .Lsr_max_{linear_idx}\n"));

    // Phase 3: exp(x - s8) accumulated in s9.
    s.push_str(&format!("    fmov    s9, wzr\n")); // sum = 0
    s.push_str(&format!("    mov     x4, #0\n"));
    s.push_str(&format!(".Lsr_exp_{linear_idx}:\n"));
    s.push_str(&format!("    ldr     s0, [x7, x4, lsl #2]\n"));
    s.push_str(&format!("    fsub    s0, s0, s8\n"));
    s.push_str(&format!("    bl      _expf\n"));
    s.push_str(&format!("    str     s0, [x7, x4, lsl #2]\n"));
    s.push_str(&format!("    fadd    s9, s9, s0\n"));
    s.push_str(&format!("    add     x4, x4, #1\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.lt    .Lsr_exp_{linear_idx}\n"));

    // Phase 4: normalise by s9.
    s.push_str(&format!("    mov     x4, #0\n"));
    s.push_str(&format!(".Lsr_norm_{linear_idx}:\n"));
    s.push_str(&format!("    ldr     s0, [x7, x4, lsl #2]\n"));
    s.push_str(&format!("    fdiv    s0, s0, s9\n"));
    s.push_str(&format!("    str     s0, [x7, x4, lsl #2]\n"));
    s.push_str(&format!("    add     x4, x4, #1\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.lt    .Lsr_norm_{linear_idx}\n"));
}
```

**Important caveats for the implementer:**

1. **Register conventions must match the rest of `emit_linear`.** This skeleton uses `x3` for i, `x4` for j, `x6/x7` for offset arithmetic, `x12` for dst — verify these against the actual existing `emit_linear`. If `x3`/`x4` are different in the surrounding code, RENAME consistently. The body of M5b's `emit_linear` is the source of truth.
2. **`x6` is scratched by `bl _expf`** (caller-saved). The implementation above recomputes the row pointer (`x7`) once at the start of the tail and re-uses it — `x7` is callee-saved (the function's prologue already saves x19–x23 because Task 5 made `compute_callee_saved` return true for fused-SoftmaxRow). Confirm `x7` is in fact callee-saved in the existing code; if not, use `x19`/`x20`/`x21` as needed.
3. **Bias path inside the j-loop is unaffected.** Phase 1 of RowWise = matmul + bias-add + `str s0, [...]` — the existing M5b chain. Only the post-op inline section was deferred for RowWise.
4. **Buffer materialisation:** the `dst_loc` materialised into `x12` already accounts for `BufferLoc::OutputReg` vs `StackOffset(...)`; the tail re-uses it.

- [ ] **Step 6: Run the substring test, expect pass**

```sh
cargo test -p profiles-arm64 emit_linear_with_softmax_row_post_op_emits_three_phase_softmax
```

Expected: PASS.

- [ ] **Step 7: Add an additional substring test for bias-aware row-wise emit**

```rust
#[test]
fn emit_linear_with_softmax_row_post_op_preserves_bias_add() {
    use compiler::passes::{run_pipeline, default_pipeline};

    // bias=true on the final linear.
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[out=2, bias=true] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let asm = crate::lower(&fused).expect("lower ok");

    // Phase 1 chain still emits matmul -> bias-add -> store.
    assert!(asm.source.contains("fadd    s0, s0, s5"), "bias-add missing in fused row-wise emit");
    // Phase 3 still calls _expf.
    assert!(asm.source.contains("bl      _expf"));
}
```

(Verify NFL syntax for `bias=true`. If the actual syntax differs — e.g. `linear[out=2, bias]` or `linear[2, bias=true]` — match it. Read `tests/fixtures/mixed_args.nfl` for a working example of bias-true syntax.)

- [ ] **Step 8: Run all profile tests + workspace clippy + tests**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p profiles-arm64
cargo test --workspace
```

Expected: all green. The Task 5 leaf-detection tests still pass (the fused asm now properly saves d8/d9 because Task 5 gave the right answer for `compute_callee_saved`). Total = baseline + 9 + 2 = baseline + 11.

- [ ] **Step 9: Commit**

```sh
git add profiles/arm64/src/ops/linear.rs
git commit -m "feat(m6/arm64): RowWise emit branch in emit_linear for SoftmaxRow

Adds a new dispatch branch in emit_linear that handles
PostOp::SoftmaxRow as a row-wise post-op (structurally different
from the existing Elementwise Relu inline). The j-loop still
materialises the full matmul+bias row, then a tail emits three
sweeps over the same buffer:

  - Phase 2: row-max scan accumulating into s8 (callee-saved across
    bl _expf in Phase 3).
  - Phase 3: exp(x - row_max) per element, in-place store, and sum
    accumulation into s9 (also callee-saved).
  - Phase 4: fdiv normalise using s9.

Phases 2-4 reuse the dst buffer that Phase 1 wrote to; no separate
softmax-output buffer is allocated by the caller (assign_buffers
sees no Softmax node post-fusion).

Two substring tests pin the structural invariants:
  - emit_linear_with_softmax_row_post_op_emits_three_phase_softmax
  - emit_linear_with_softmax_row_post_op_preserves_bias_add

Bit-exact equivalence with the unfused (--no-passes) path is
verified separately in Task 7's FFI integration test."
```

---

## Task 7: `softmax_with_bias.nfl` fixture + FFI integration test

**Spec ref:** §9 + §13 R3.

The classifier fixture's final `linear[output] -> softmax` has no bias (default `bias=false`). M6 needs a parallel fixture that exercises the bias-aware path through `linear[bias=true] -> softmax`.

**Files:**
- Create: `tests/fixtures/softmax_with_bias.nfl`
- Modify: `profiles/arm64/tests/integration.rs` (add `fused_vs_unfused_softmax_match_numerically`)

- [ ] **Step 1: Look up the bias=true syntax used by `mixed_args.nfl`**

Read `tests/fixtures/mixed_args.nfl` to see the exact NFL syntax for a `linear` with `bias=true`. Replicate that syntax in the new fixture.

- [ ] **Step 2: Create `tests/fixtures/softmax_with_bias.nfl`**

A minimal fixture exercising `linear[bias=true] -> softmax` as the final step:

```nfl
model SoftmaxWithBias [batch=4, input=8, output=3]:
    x: Tensor[batch, input]

    x -> linear[16] -> relu
      -> linear[out=output, bias=true] -> softmax
```

(Adjust to match the actual NFL grammar for `bias=true`. Keep dimensions small — the FFI test will allocate input/params/output arrays.)

Verify it parses:

```sh
cargo run -p nflc -- parse tests/fixtures/softmax_with_bias.nfl --uir
```

Expected: clean UIR rendering ending with a Linear (bias=true) feeding a Softmax.

- [ ] **Step 3: Write the failing FFI integration test**

In `profiles/arm64/tests/integration.rs`, add a new test mirroring the structure of `fused_vs_unfused_classifier_match_numerically` (read it for the exact pattern). Two scenarios — classifier (bias=false on final linear) and softmax_with_bias (bias=true). Combine into one test for cohesion or split into two — split for clarity:

```rust
#[test]
fn fused_vs_unfused_softmax_match_numerically() {
    if !cfg!(target_arch = "aarch64") { eprintln!("skip: requires aarch64"); return; }
    if !common::cc_available() { eprintln!("skip: requires cc"); return; }

    // Cover BOTH no-bias and bias-aware fused-softmax paths.
    for (fixture_path, fn_name, batch, input_dim, output_dim) in [
        ("../../tests/fixtures/classifier.nfl", "nfl_forward_Classifier", 32, 784, 10),
        ("../../tests/fixtures/softmax_with_bias.nfl", "nfl_forward_SoftmaxWithBias", 4, 8, 3),
    ] {
        let src = std::fs::read_to_string(fixture_path).unwrap_or_else(|e| panic!("{fixture_path}: {e}"));
        let ast = compiler::parse(&src).unwrap();
        let uir = compiler::ir::build(&ast).unwrap();

        let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline()).expect("pipeline ok");
        let fused_asm = profiles_arm64::lower(&fused_uir).expect("fused lower");
        let unfused_asm = profiles_arm64::lower(&uir).expect("unfused lower");

        // Asm structural validation:
        //   - fused: contains the Phase 3 bl _expf inside emit_linear's tail
        //   - unfused: still has separate softmax loops
        assert!(fused_asm.source.contains("bl      _expf"),
            "{fixture_path}: fused asm missing bl _expf in row-wise tail");
        assert!(!fused_asm.source.contains(".Lsoftmax_"),
            "{fixture_path}: fused asm should NOT have separate softmax loop labels");
        assert!(unfused_asm.source.contains(".Lsoftmax_") || unfused_asm.source.contains("bl      _expf"),
            "{fixture_path}: unfused asm should have softmax artefacts");

        let fused_dylib = common::compile_to_dylib(&fused_asm.source, &format!("fused_{}", fn_name.trim_start_matches("nfl_forward_")));
        let unfused_dylib = common::compile_to_dylib(&unfused_asm.source, &format!("unfused_{}", fn_name.trim_start_matches("nfl_forward_")));

        let fused_lib = unsafe { libloading::Library::new(&fused_dylib).unwrap() };
        let unfused_lib = unsafe { libloading::Library::new(&unfused_dylib).unwrap() };

        let fn_bytes = std::ffi::CString::new(fn_name).unwrap();
        let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            unsafe { fused_lib.get(fn_bytes.as_bytes_with_nul()).unwrap() };
        let unfused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            unsafe { unfused_lib.get(fn_bytes.as_bytes_with_nul()).unwrap() };

        let params_len = fused_asm.functions[0].params_floats;
        assert_eq!(params_len, unfused_asm.functions[0].params_floats,
            "{fixture_path}: fused/unfused param layout mismatch");

        let mut input = vec![0.0f32; batch * input_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i as f32) % 100.0) * 0.001;
        }
        let mut params = vec![0.0f32; params_len];
        for (i, v) in params.iter_mut().enumerate() {
            *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
        }

        let mut fused_out = vec![0.0f32; batch * output_dim];
        let mut unfused_out = vec![0.0f32; batch * output_dim];

        unsafe {
            fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
            unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
        }

        for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
            assert_eq!(*a, *b,
                "{fixture_path}: fused[{i}]={a} unfused[{i}]={b} — fusion changed numerics");
        }
    }
}
```

(Note: this test uses `assert_eq!` for `params_floats`, not `debug_assert_eq!` — that's the OQ-5 harmonisation, applied here from the start. Task 8 retro-fits the same change to M5a's and M5b's existing tests.)

- [ ] **Step 4: Run the test, expect pass**

```sh
cargo test -p profiles-arm64 --test integration fused_vs_unfused_softmax_match_numerically
```

Expected: PASS for both fixtures. If FAIL with a numeric divergence, that's the spec's R1 risk materialising — go back to Task 6 and verify the asm-sketch implementation. If FAIL with a SIGSEGV / SIGILL, that's likely a Task 5 issue (callee-saved or leaf detection wrong).

- [ ] **Step 5: Run the full workspace test suite + clippy**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Total = baseline + 11 + 1 (new FFI test) = baseline + 12.

- [ ] **Step 6: Commit**

```sh
git add tests/fixtures/softmax_with_bias.nfl profiles/arm64/tests/integration.rs
git commit -m "test(m6/integration): bit-exact fused-vs-unfused softmax FFI test

Adds tests/fixtures/softmax_with_bias.nfl as the bias-true companion
to tests/fixtures/classifier.nfl (whose final linear has bias=false).

The new fused_vs_unfused_softmax_match_numerically integration test
covers BOTH fixtures in one body — looping over (fixture, fn_name,
dims) tuples — to exercise both the no-bias and bias-aware paths
through the RowWise emit branch.

Bit-exact equality is asserted via assert_eq! (the OQ-5 carry-forward
from M5c — the params_floats agreement check uses assert_eq! from
this test's first commit; Task 8 of the M6 plan retro-fits the same
upgrade to M5a's classifier test and M5b's mixed_args test)."
```

---

## Task 8: OQ-5 `assert_eq!` harmonisation

**Spec ref:** §12 OQ-5, §14 Tests checkbox.

Three FFI integration tests now exist:
1. `fused_vs_unfused_classifier_match_numerically` (M5a)
2. `fused_vs_unfused_mixed_args_match_numerically` (M5b)
3. `fused_vs_unfused_softmax_match_numerically` (M6 — Task 7 above)

The first two used `debug_assert_eq!` for the `params_floats` agreement check — silently bypassing the assertion in release builds. M6 standardises all three on `assert_eq!`.

**Files:**
- Modify: `profiles/arm64/tests/integration.rs` (two existing tests; the third already uses `assert_eq!` per Task 7)

- [ ] **Step 1: Locate the two `debug_assert_eq!` call sites**

```sh
grep -n "debug_assert_eq!" profiles/arm64/tests/integration.rs
```

Expected: exactly two hits, one per existing fused/unfused test.

- [ ] **Step 2: Replace both with `assert_eq!`**

Edit `profiles/arm64/tests/integration.rs`:

In `fused_vs_unfused_classifier_match_numerically`, change:

```rust
let params_len = fused_asm.functions[0].params_floats;
debug_assert_eq!(params_len, unfused_asm.functions[0].params_floats);
```

to:

```rust
let params_len = fused_asm.functions[0].params_floats;
assert_eq!(params_len, unfused_asm.functions[0].params_floats,
    "fused/unfused params_floats disagree — pipeline changed param layout");
```

Apply the identical edit in `fused_vs_unfused_mixed_args_match_numerically`.

- [ ] **Step 3: Run the integration tests**

```sh
cargo test -p profiles-arm64 --test integration
```

Expected: all three FFI tests still pass — the assertion was already true at runtime in M5a/M5b (the pipeline does not change param layout); the upgrade simply enforces it in release builds too.

- [ ] **Step 4: Run workspace tests + clippy**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: green.

- [ ] **Step 5: Commit**

```sh
git add profiles/arm64/tests/integration.rs
git commit -m "test(m6/integration): harmonise params_floats agreement check (OQ-5)

All three fused_vs_unfused_*_match_numerically tests now use assert_eq!
(not debug_assert_eq!) for the params_floats agreement check between
fused and unfused asm. debug_assert_eq! is a no-op in release builds —
the agreement claim should hold unconditionally, so the upgrade is
correct.

This closes spec §12 OQ-5 and the corresponding §14 Done Criteria
checkbox. Carry-forward from M5c (DEVLOG-1)."
```

---

## Task 9: CLI smoke for `--passes fuse_linear_softmax`

**Spec ref:** §6 + §9 + §13 R-related (catches CLI registry update).

**Files:**
- Modify: `nflc/tests/cli_compile.rs`

- [ ] **Step 1: Read the M5b smoke template**

In `nflc/tests/cli_compile.rs`, locate `compile_with_passes_filter_runs_only_selected` (the M5b test that asserts `--passes fuse_linear_relu` runs only that pass). Note the assertion style.

- [ ] **Step 2: Write a failing smoke test for `--passes fuse_linear_softmax`**

```rust
#[test]
fn compile_with_passes_filter_only_fuse_linear_softmax_runs() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/classifier.nfl",
            "--profile", "arm64",
            "--passes", "fuse_linear_softmax",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(stderr.contains("note: applied passes: fuse_linear_softmax"),
        "stderr should announce only fuse_linear_softmax, got: {stderr}");
    assert!(!stderr.contains("eliminate_dropout"),
        "stderr should NOT mention eliminate_dropout under filter");
    assert!(!stderr.contains("fuse_linear_relu"),
        "stderr should NOT mention fuse_linear_relu under filter");

    // The fused row-wise softmax shape — a substring inside the asm —
    // pins that the pass actually ran end-to-end.
    assert!(stdout.contains("bl      _expf"),
        "fused asm should still call bl _expf inside emit_linear's RowWise tail");
    assert!(!stdout.contains(".Lsoftmax_"),
        "with fuse_linear_softmax applied, no separate softmax loop label should appear");
}
```

(The exact `note: applied passes:` format must match what `nflc compile` actually prints. M5b set this format; if it's different — `note: passes:`, `note: filter:`, etc. — match the existing convention exactly.)

- [ ] **Step 3: Run the test, expect pass**

```sh
cargo test -p nflc --test cli_compile compile_with_passes_filter_only_fuse_linear_softmax_runs
```

Expected: PASS — the dynamic pass registry already exposes `fuse_linear_softmax` as an available name (Task 4 added it to `default_pipeline`). The CLI required no code changes for this; this smoke test pins that fact.

- [ ] **Step 4: Run all CLI tests + workspace**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: green. Total = baseline + 12 (Tasks 1–7) + 1 (this task) = baseline + 13.

- [ ] **Step 5: Commit**

```sh
git add nflc/tests/cli_compile.rs
git commit -m "test(m6/cli): smoke test for --passes fuse_linear_softmax

Confirms the dynamic pass registry exposes fuse_linear_softmax as
an available name without CLI code changes (the M5b convention
of reading default_pipeline().iter().map(|p| p.name()) carries this
for free).

Asserts: (a) stderr announces 'applied passes: fuse_linear_softmax'
and only that, (b) no eliminate_dropout / fuse_linear_relu mentioned
under filter, (c) stdout asm contains bl _expf in the RowWise tail
and no .Lsoftmax_ label (proving the pass actually executed and the
RowWise emit branch took over)."
```

---

## Task 10: Documentation updates

**Spec ref:** §11.

**Files:**
- Modify: `docs/profile_guide/arm64.md`
- Modify: `docs/language_reference/uir.md`

- [ ] **Step 1: Update `docs/profile_guide/arm64.md` §3 supported-ops table**

Find the §3 supported-ops table. The existing `Softmax` row reads (M5b-current). Edit it to mention the M6 fused-vs-unfused split, mirroring how Linear / Relu / Dropout rows already document it:

```markdown
| `Softmax` | Inference 3-pass (max → exp → normalise). With `--no-passes` or `--passes` filter excluding `fuse_linear_softmax`: emitted as a standalone function via `emit_softmax`. Default pipeline (M6+): fused into the preceding Linear's emit_linear via `PostOp::SoftmaxRow` (row-wise tail; see §4.10). |
```

- [ ] **Step 2: Add new §4.10 "Fused linear → softmax (row-wise)"**

Insert after §4.9 (M5b's "Fused linear → relu" section). Paste the §8 asm-sketch from the spec and add the structural-difference note. Approximate length: 80–120 lines.

```markdown
### 4.10 Fused linear → softmax (row-wise)

When the M6 default pipeline applies `FuseLinearSoftmax`, a Linear node
carrying `fused_post_ops = [PostOp::SoftmaxRow]` triggers a different
`emit_linear` branch from §4.9's Elementwise Relu fusion. The row-wise
emit shape **cannot** inline the softmax computation per element —
softmax requires the row-max before any element can be exponentiated.

The structure is four phases nested inside the batch (i) outer loop:

1. **Phase 1 — matmul + bias-add** materialises the full output row
   `out[i, 0..N]` via the existing j-loop (same shape as §4.9's
   matmul body, but without the per-element post-op inline).
2. **Phase 2 — row-max scan** sweeps `out[i, 0..N]` accumulating the
   max into `s8` (callee-saved across `bl _expf`).
3. **Phase 3 — exp(x − s8) + sum** sweeps the row again, exponentiating
   each element via `bl _expf`, writing the result back in-place, and
   accumulating the sum into `s9` (also callee-saved).
4. **Phase 4 — normalise** sweeps a third time, dividing each element
   by `s9`.

The asm sketch (illustrative — actual register allocation may vary):

[paste the §8 asm-sketch from the spec]

#### Why row-wise differs from element-wise

Softmax is mathematically a row-wise reduction: every element of the
output depends on the row-max and the row-sum, which are not known
until the full row has been materialised. Attempting to inline softmax
per `(i, j)` like Relu's `fmax s0, s0, s4` would produce
mathematically wrong output. **Implementers extending this branch
must NOT attempt per-element softmax inlining.**

#### Memory and ABI

- The dst buffer is touched 6 times per row (1 write Phase 1, 1 read
  Phase 2, 1 read + 1 write Phase 3, 1 read + 1 write Phase 4) — all
  on the same buffer. No separate softmax-output buffer is allocated;
  `assign_buffers` does not see a Softmax node post-fusion.
- `bl _expf` is invoked N times per row, identical to unfused
  `emit_softmax`. `compute_is_leaf` returns false for any model
  containing a Linear with `PostOp::SoftmaxRow` in its `fused_post_ops`,
  driving prologue saves of `lr`/`x30` + callee-saved `d8`/`d9`.
- Row-max (`s8`) and row-sum (`s9`) live in callee-saved float
  registers for the duration of Phases 2–4. No additional stack slots.

#### Bias-aware fusion

`linear[bias=true] → softmax` fuses identically — Phase 1 includes the
existing `ldr s5, [x14, x4, lsl #2]` + `fadd s0, s0, s5` step before
the row materialises. Phases 2–4 are bias-independent.

#### Stacking with other post-ops

`FuseLinearSoftmax`'s pass-side criterion 4 (Linear's `fused_post_ops`
must be empty) prevents `[Relu, SoftmaxRow]` stacks from arising from
NFL v0.1 fixtures. The lowering rejects any stacked post-ops including
a RowWise variant by returning
`LowerError::UnsupportedPostOp { op: "stacked post-ops with RowWise variant", ... }`.
This is conservative; revisit if a future pattern legitimately needs
mixed elementwise + row-wise post-ops.
```

- [ ] **Step 3: Update §5 errors table**

Add or extend the `UnsupportedPostOp` row's description to mention `SoftmaxRow` as a concrete M6 implementation (no new variant; the existing wildcard now never fires for `SoftmaxRow` in default pipeline runs).

- [ ] **Step 4: Rewrite §8 Limitations**

Find the post-M5c "only Relu post-op fuses" line. Replace with:

```markdown
- Two `PostOp` variants are supported by `emit_linear`: `Relu`
  (Elementwise — inline `fmax s0, s0, s4` inside the j-loop;
  see §4.9) and `SoftmaxRow` (RowWise — three sweeps after the
  j-loop; see §4.10). Stacking variants is not supported and is
  guarded by `FuseLinearSoftmax` criterion 4 plus a defensive
  check in `emit_linear`.
- Graph-level dead-op elimination is limited to `EliminateDropout`.
  No general DCE pass.
- libm `expf` is the only `expf` source for `softmax` and
  `SoftmaxRow` post-op. Bare-metal targets requiring a
  Taylor/minimax `expf` are M7+ work (spec §12 OQ-3).
```

- [ ] **Step 5: Update `docs/language_reference/uir.md` §2**

Find `NodeKind::Op` rendering. Where the `fused_post_ops: Vec<PostOp>` field is described, list `SoftmaxRow` alongside `Relu` as a valid value:

```markdown
- `fused_post_ops: Vec<PostOp>` — post-operations folded into this
  node by passes such as `FuseLinearRelu` (`PostOp::Relu`) and
  `FuseLinearSoftmax` (`PostOp::SoftmaxRow`). Empty by default.
```

- [ ] **Step 6: Verify the docs compile (markdown render-check via grep)**

```sh
grep -nE "^#{1,6} " docs/profile_guide/arm64.md | head -30
grep -nE "^#{1,6} " docs/language_reference/uir.md | head -20
```

Expected: heading hierarchy looks consistent, no missing levels, §4.10 sits where §4.9 left off.

- [ ] **Step 7: Run workspace tests + clippy (smoke check no test references stale doc names)**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: green. Test count unchanged.

- [ ] **Step 8: Commit**

```sh
git add docs/profile_guide/arm64.md docs/language_reference/uir.md
git commit -m "docs(m6): arm64 §4.10 row-wise softmax + uir.md SoftmaxRow

Brings docs to M6 state:

  - arm64.md §3 Softmax row: fused-vs-unfused split documented in the
    same style M5b used for Linear / Relu / Dropout.
  - arm64.md §4.10 (new): Fused linear → softmax (row-wise) — the
    full asm sketch, the four-phase structure, the explicit warning
    that row-wise differs structurally from elementwise (do NOT
    inline softmax per element), memory and ABI notes, bias-aware
    fusion, and stacking constraints (criterion 4 + defensive
    UnsupportedPostOp).
  - arm64.md §5 errors: UnsupportedPostOp annotated with SoftmaxRow
    as the M6 concrete implementation.
  - arm64.md §8 Limitations: rewritten — two PostOp variants supported
    (Relu Elementwise + SoftmaxRow RowWise), no stacking, libm-only
    expf, EliminateDropout-only graph DCE.
  - uir.md §2: SoftmaxRow listed alongside Relu in the
    fused_post_ops field description."
```

---

## Task 11: Closeout (DEVLOG + PROJECT_SPEC + CLAUDE.md + holistic review)

**Spec ref:** §14 Process.

**Files:**
- Modify: `DEVLOG.md`
- Modify: `PROJECT_SPEC.md` (M6 row → "complete")
- Modify: `CLAUDE.md` ("Current Status" section)

- [ ] **Step 1: Run a holistic-review subagent dispatch**

Spawn a single subagent (general-purpose or Explore) with the prompt:

> Review the M6 implementation against the spec at
> `docs/superpowers/specs/2026-05-05-m6-attention-fusion-design.md`.
> Check: (a) every §-numbered requirement in the spec has corresponding
> code/tests, (b) docs in `docs/profile_guide/arm64.md` and
> `docs/language_reference/uir.md` reflect the implemented behaviour,
> (c) cross-cutting consistency — `#[non_exhaustive]` cascade arms in
> `profiles/arm64`, `Display` impls cover all enum variants, `passes`
> module exports are consistent with M5b's pattern, (d) M5c-style
> drift between code and PROJECT_SPEC.md / CLAUDE.md "Current Status".
> Report findings as a numbered list with citation paths and line
> numbers. Aim for a comprehensive scan; expect 5–15 findings of
> varying severity (the M5c review found 17 across less work).

Wait for the report. Triage findings into "close in M6c-style mini-task" vs "carry-forward to M7+".

- [ ] **Step 2: Address the M6-close findings**

For each finding to close in M6: edit the file, run `cargo fmt + clippy + test`, commit with message `chore(m6/holistic): close finding N — <one-line>`.

(This is iterative; repeat until the close-in-M6 list is empty.)

- [ ] **Step 3: Update `PROJECT_SPEC.md` M6 row to "complete"**

Find the milestones table row 6 (currently "Attention-pattern fusion (kernel fusion v2)"). Append " — **complete**" or rewrite the row text to reflect the shipped implementation, mirroring how M5 row was updated in M5c. Example replacement:

```markdown
| 6 | Attention-pattern fusion — kernel fusion v2 (complete) | `PostOp::SoftmaxRow` variant + `FuseLinearSoftmax` pass; `default_pipeline = [EliminateDropout, FuseLinearRelu, FuseLinearSoftmax]`; arm64 RowWise emit branch in `emit_linear` (4-phase: matmul row → row-max → exp+sum → normalize, in-place over the linear output buffer using callee-saved `s8`/`s9`); bit-exact equivalence proven via `fused_vs_unfused_softmax_match_numerically` on `classifier` (no-bias) + `softmax_with_bias` (bias-aware) fixtures; `compiler/src/ir/test_utils.rs` shared helpers; OQ-5 `assert_eq!` harmonisation across all three fused-vs-unfused tests |
```

- [ ] **Step 4: Rewrite the "Current Status" section in `CLAUDE.md`**

Find the "Current Status" section (M5c left it describing M5c-closeout state). Replace with a M6-closed equivalent, matching the level of detail M5c set:

- M6 close summary: SoftmaxRow + FuseLinearSoftmax + RowWise emit branch.
- Total test count: explicit number from the workspace.
- Mention of `compiler/src/ir/test_utils.rs` extraction.
- OQ-5 closed.
- Open M6-deferred carry-forward (OQ-1 through OQ-4, OQ-6) for M7+.
- "The immediate next step is **Milestone 7 — open scope**" with candidate directions from the spec's §12 (or pulled from M5c's list, minus the items M6 closed).

- [ ] **Step 5: Add a DEVLOG entry**

Append a new entry at the top (DEVLOG is reverse-chronological per the project protocol). Follow the standard format:

```markdown
## YYYY-MM-DD — Milestone 6 closed: attention-pattern fusion (`linear → softmax`)

### What was done
- [bullet list of major deliverables — PostOp variant, pass, RowWise emit branch, test_utils extraction, FFI test, fixture, docs, OQ-5 harmonisation]

### Decisions made
- [any architectural calls that diverged from the spec, or any spec amendments made during implementation]

### Problems encountered
- [Plan-phase Task 0 findings: classifier had no bias on final layer → fixture added; …]
- [Holistic-review findings: …]

### Known tech debt (carried forward to M7+)
- [OQ-1 FuseLinearPostOp consolidation, OQ-2 type-level distinction, OQ-3 bare-metal expf, OQ-4 BuildError::span()/Diagnostic, OQ-6 format!/to_string()]

### Next step
**Milestone 6 fully complete.** Brainstorm M7 in a fresh worktree.
Open scope; candidate directions (from spec §12 + M5c carry-forward):
1. [your priority ordering]
```

- [ ] **Step 6: Run all the gates one last time**

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: ALL green. Note the final test count for the DEVLOG entry.

- [ ] **Step 7: Commit closeout**

```sh
git add DEVLOG.md PROJECT_SPEC.md CLAUDE.md
git commit -m "chore(m6): close Milestone 6 — full cycle complete

PROJECT_SPEC.md milestones table M6 row → complete.
CLAUDE.md Current Status rewritten to reflect M6-closed reality.
DEVLOG.md entry per the project's documentation protocol.

What landed in M6:
- PostOp::SoftmaxRow variant (third on #[non_exhaustive] PostOp).
- compiler::passes::fuse_linear_softmax — bias-aware UIR pass.
- default_pipeline = [EliminateDropout, FuseLinearRelu, FuseLinearSoftmax].
- profiles/arm64::emit_linear — RowWise emit branch (4-phase
  in-place softmax over the linear output buffer, callee-saved
  s8/s9 spanning bl _expf).
- compute_is_leaf + compute_callee_saved extended to detect
  PostOp::SoftmaxRow in fused_post_ops.
- compiler/src/ir/test_utils.rs — shared UIR helpers, M5b cross-pass
  test migrated.
- tests/fixtures/softmax_with_bias.nfl — bias-aware FFI coverage.
- fused_vs_unfused_softmax_match_numerically — third FFI test;
  OQ-5 assert_eq! harmonisation applied across all three.
- CLI smoke for --passes fuse_linear_softmax.
- arm64.md §3 / §4.10 / §5 / §8 + uir.md §2 updated.

OQ-1 (FuseLinearPostOp), OQ-2 (PostOpKind), OQ-3 (bare-metal expf),
OQ-4 (BuildError::span/Diagnostic), and OQ-6 (format! style) carry
forward to M7+ per their spec-defined triggers."
```

- [ ] **Step 8: Done**

```sh
git log --oneline | head -25
cargo test --workspace 2>&1 | tail -5
```

Verify the commit history shows the M6 task sequence and the test count is the final number.

---

## Done. What's next?

Per the spec's §12, the carry-forward debt list for M7+ is:
- OQ-1 `FuseLinearPostOp` consolidation — fires on a third access pattern or a second RowWise post-op.
- OQ-2 type-level `PostOpKind` distinction — same trigger plus emit-shape divergence.
- OQ-3 bare-metal `expf` — fires on user-driven embedded need.
- OQ-4 `BuildError::span()` + `Diagnostic` trait — fires on a fourth error type or generic CLI rendering.
- OQ-6 `format!`/`to_string()` style consistency — fires on the next cascade-arm touch.

Test-helper extraction (OQ from M5c) is now closed by Task 1 — `compiler/src/ir/test_utils.rs` is the shared canonical location.

Brainstorm M7 in a fresh worktree.
