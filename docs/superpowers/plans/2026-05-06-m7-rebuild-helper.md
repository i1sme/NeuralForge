# Milestone 7 — Shared 3-Step Rebuild Helper Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the identical 3-step rebuild skeleton (identify victims → rebuild with id-remap → remap inputs/output) from three existing UIR passes (`EliminateDropout`, `FuseLinearRelu`, `FuseLinearSoftmax`) into a shared `compiler::passes::rewriter` helper. Each pass body shrinks from 70-100 lines to 15-25 lines while preserving identical behavior. Add one unit test (Task 2 from spec) closing M6's §8 invariant 6 coverage gap.

**Architecture:** Plan-as-data API. New `pub(crate) struct RewritePlan` holds three HashMaps — `consumer_count` (precomputed by `RewritePlan::new(&model)`), `victims` (declared by callers during victim identification), `producer_post_ops` (declared by callers when producers need PostOp mutation). New `pub(crate) fn rewrite_model(plan, model)` consumes both inputs and returns a fresh `UirModel`. No closures, no traits — caller populates plan via mutable borrow, hands it to `rewrite_model`. Callers preserve their `Result<UirModel, PassError>` per-pass shape via one-line `Ok(...)` wrap.

**Tech Stack:** Rust 2021, std-only at runtime. Workspace: `compiler` crate (where M7 work lives), `nflc` and `profiles-arm64` (untouched). Helper visibility `pub(crate)` — implementation detail of the `passes` module.

**Spec:** [`docs/superpowers/specs/2026-05-06-m7-rebuild-helper-design.md`](../specs/2026-05-06-m7-rebuild-helper-design.md). All §-references in this plan refer to the spec.

---

## Pre-implementation findings

A reading pass over the M6-current code (HEAD `2f95203` — M6 merge commit on origin/main) confirmed the spec's preconditions and load-bearing facts:

1. **Three pass bodies have identical 3-step rebuild skeleton.** Verified in `compiler/src/passes/eliminate_dropout.rs:50-114`, `compiler/src/passes/fuse_linear_relu.rs:43-141`, and `compiler/src/passes/fuse_linear_softmax.rs:37-130`. All three: walk nodes by old NodeId in topological order, branch on victim membership, clone non-victims with operand remap, optionally push PostOp to producers, then remap `model.inputs` and `model.output` through `id_map`.
2. **`compiler::ir::test_utils` exists with all four helpers.** Verified at `compiler/src/ir/test_utils.rs:13,26,48,56`: `pub(crate) fn input_node(name: &str, shape: Vec<u64>) -> Node`, `pub(crate) fn op_node(op: StdOp, operands: Vec<NodeId>, attrs: Vec<OpAttr>, shape: Vec<u64>) -> Node`, `pub(crate) fn out_dim_attr(value: u64) -> OpAttr`, `pub(crate) fn rate_attr(value: f64) -> OpAttr`. M6 Task 1 deliverable — M7 consumes, does not create.
3. **`fuse_linear_softmax::tests` has module-level `use crate::{Uir, UirModel};`** (line 7 of the tests block). Task 5's inline `use crate::{Uir, UirModel};` is intentional self-documenting redundancy, harmless.
4. **`Pass::run` trait signature:** `fn run(&self, uir: &Uir) -> Result<Uir, PassError>`. M7 does NOT change the trait. Per-pass `run` impls iterate `&uir.models`, clone each model (after Task 1 signature change), call the consuming `eliminate_one_model(model.clone())?` / `fuse_one_model(model.clone())?`.
5. **Workspace baseline: 202 tests, all green.** `cargo build --workspace` clean, `cargo clippy --workspace --all-targets -- -D warnings` clean, `cargo fmt --all -- --check` clean. M7 expected target: 208 (+5 helper unit tests + 1 invariant 6 test).

---

## File map

**New files:**
- `compiler/src/passes/rewriter.rs` — the helper module: `RewritePlan` struct + `rewrite_model` function + 5 inline unit tests.

**Modified files (compiler):**
- `compiler/src/passes/mod.rs` — add `pub(crate) mod rewriter;` declaration.
- `compiler/src/passes/eliminate_dropout.rs` — refactor `eliminate_one_model` to use helper; change signature to consume `UirModel`; update `Pass::run` to clone before passing; retire stale doc-comment about M7-deferred trigger; tighten imports.
- `compiler/src/passes/fuse_linear_relu.rs` — refactor `fuse_one_model`; change signature; update `Pass::run`; tighten imports.
- `compiler/src/passes/fuse_linear_softmax.rs` — refactor `fuse_one_model`; change signature; update `Pass::run`; tighten imports; add Task 2 unit test `leaves_linear_dropout_softmax_chain_untouched`.

**Modified files (closeout):**
- `PROJECT_SPEC.md` — add M7 row marked "complete"; relocate "Human-readable viewer v0.1" from M7 row to M8 row.
- `CLAUDE.md` — rewrite "Current Status" section reflecting M7 closure; update Design Principle 5 reference `(M7+)` → `(M8+)` for viewer tool.
- `DEVLOG.md` — add new M7 entry per project documentation protocol.

---

## Task overview

| # | Task | Approx. size |
|---|------|--------------|
| 1 | Create `rewriter.rs` helper module with 5 unit tests | medium |
| 2 | Migrate `EliminateDropout` (atomic unit 2) | small |
| 3 | Migrate `FuseLinearRelu` (atomic unit 3) | small |
| 4 | Migrate `FuseLinearSoftmax` (atomic unit 4) | small |
| 5 | Task 2 from spec — `leaves_linear_dropout_softmax_chain_untouched` invariant 6 unit test | trivial |
| 6 | Closeout (holistic review + PROJECT_SPEC + CLAUDE.md + DEVLOG) | small |

**Sequencing constraints:**
- Task 1 must finish before Tasks 2-4 (helper must exist before migrations).
- Task 2 (EliminateDropout) before Task 3 (FuseLinearRelu) before Task 4 (FuseLinearSoftmax) — migration order per spec §4.7 (simplest first → largest test surface → mirror).
- Task 5 can run in parallel with Tasks 2-4 in principle, but recommended after Task 4. Sequential execution is fine.
- Task 6 (closeout) is last — depends on all prior tasks being done.

**Atomic-task-pack convention (from spec §4.8):** Tasks 1-4 are four atomic commits with the workspace green between each. `cargo fmt + clippy + test --workspace` runs at the end of each task before commit.

---

## Task 1: Create `rewriter.rs` helper module

**Spec ref:** §5 (RewritePlan struct), §6 (rewrite_model function), §7 (file location & visibility), §9 (helper unit tests).

**Files:**
- Create: `compiler/src/passes/rewriter.rs`
- Modify: `compiler/src/passes/mod.rs` (add `pub(crate) mod rewriter;`)

- [ ] **Step 1: Create the helper module file**

Create `compiler/src/passes/rewriter.rs`:

```rust
//! Shared 3-step rebuild helper for UIR passes (M7).
//!
//! Three passes (`EliminateDropout`, `FuseLinearRelu`,
//! `FuseLinearSoftmax`) share an identical rebuild skeleton:
//!
//!   1. Identify victims (nodes that disappear in the new UIR).
//!   2. Rebuild + remap (clone non-victims, remap operand IDs through
//!      an id_map, optionally mutate kept producers, redirect victim
//!      references to other nodes' new IDs).
//!   3. Remap model.inputs and model.output through the same id_map.
//!
//! This module factors that skeleton out. Callers build a
//! `RewritePlan` (declared data, no closures), then call
//! `rewrite_model(plan, model)` which executes the plan and returns
//! a fresh `UirModel`.
//!
//! Plan-as-data design rationale: see spec §4.1 (rejected closure
//! and trait alternatives) and §4.5 (plain-`UirModel` return,
//! preconditions enforced by panic).

use crate::ir::types::{NodeKind, PostOp};
use crate::{NodeId, UirModel};
use std::collections::HashMap;

/// A plan for rewriting a `UirModel`: which nodes disappear (and
/// what their references redirect to), plus which surviving nodes
/// get post-op mutations.
///
/// Build a plan with `RewritePlan::new(&model)` (which precomputes
/// `consumer_count`), populate `victims` and `producer_post_ops`
/// during your pass's victim-identification logic, then hand the
/// plan to `rewrite_model(plan, model)`.
pub(crate) struct RewritePlan {
    /// node_id → number of consumers. Counts both `NodeKind::Op`
    /// operands referencing this node, and an extra `+1` if
    /// `model.output == node_id`. Precomputed by `new()`; passes
    /// that don't need consumer counts (e.g. `EliminateDropout`)
    /// simply don't reference this field.
    pub consumer_count: HashMap<NodeId, usize>,

    /// victim_id → redirect-target's old NodeId. The rewriter sets
    /// `id_map[victim] = id_map[victims[victim]]` when it encounters
    /// the victim during rebuild. The redirect target MUST appear
    /// earlier in topological order than the victim
    /// (target_old_id < victim_old_id).
    pub victims: HashMap<NodeId, NodeId>,

    /// producer_id → post-ops to push onto the producer's
    /// `fused_post_ops` AFTER its operands are remapped. Producer
    /// must be a non-victim `NodeKind::Op` node. Multiple PostOps
    /// push in vec order.
    pub producer_post_ops: HashMap<NodeId, Vec<PostOp>>,
}

impl RewritePlan {
    pub(crate) fn new(model: &UirModel) -> Self {
        let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
        for node in &model.nodes {
            if let NodeKind::Op { operands, .. } = &node.kind {
                for &op_id in operands {
                    *consumer_count.entry(op_id).or_insert(0) += 1;
                }
            }
        }
        *consumer_count.entry(model.output).or_insert(0) += 1;

        Self {
            consumer_count,
            victims: HashMap::new(),
            producer_post_ops: HashMap::new(),
        }
    }
}

/// Execute a rewrite plan against `model`. Returns a fresh `UirModel`
/// with renumbered NodeIds; both inputs are consumed.
///
/// Preconditions (caller's responsibility — violations cause
/// `id_map[…]` panics, NOT a `Result::Err`):
///   - `model.nodes` is in topological order (operand IDs <
///     consumer IDs). `compiler::ir::build` guarantees this.
///   - For every `(victim, target)` in `plan.victims`,
///     `target_old_id < victim_old_id`.
///   - Every key in `plan.producer_post_ops` is NOT a key in
///     `plan.victims`.
///   - Every key in `plan.producer_post_ops` references a node with
///     `NodeKind::Op` kind.
pub(crate) fn rewrite_model(plan: RewritePlan, model: UirModel) -> UirModel {
    let RewritePlan {
        consumer_count: _,
        victims,
        producer_post_ops,
    } = plan;

    let mut new_nodes = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.into_iter().enumerate() {
        if let Some(&target_old_id) = victims.get(&old_id) {
            // Victim: redirect its id_map entry to target's new id.
            let target_new_id = id_map[&target_old_id];
            id_map.insert(old_id, target_new_id);
            continue;
        }

        // Non-victim: take ownership, remap operands, optionally
        // append PostOps, push to new_nodes.
        let mut new_node = node;
        if let NodeKind::Op {
            operands,
            fused_post_ops,
            ..
        } = &mut new_node.kind
        {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
            if let Some(extras) = producer_post_ops.get(&old_id) {
                fused_post_ops.extend(extras.iter().copied());
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    let new_inputs: Vec<NodeId> = model.inputs.iter().map(|id| id_map[id]).collect();
    let new_output = id_map[&model.output];

    UirModel {
        name: model.name,
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
        source_span: model.source_span,
    }
}

#[cfg(test)]
mod tests {
    // Tests filled in by Steps 4-8 below.
}
```

- [ ] **Step 2: Wire the module into `passes::mod`**

In `compiler/src/passes/mod.rs`, add `pub(crate) mod rewriter;` immediately above the existing `pub mod` declarations:

```rust
pub(crate) mod rewriter;
```

The `pub(crate)` visibility (vs `pub` for actual passes) emphasises that `rewriter` is an internal utility, not an externally-visible pass.

- [ ] **Step 3: Verify the module compiles**

```sh
cargo build --workspace
```

Expected: clean build, no warnings.

- [ ] **Step 4: Add unit test #1 — `rewrite_model_with_empty_plan_is_identity`**

In `compiler/src/passes/rewriter.rs::tests`:

```rust
#[test]
fn rewrite_model_with_empty_plan_is_identity() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::NodeKind;
    use crate::ir::StdOp;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 1,
        source_span: crate::ast::Span::new(1, 1),
    };
    let plan = super::RewritePlan::new(&model);
    let out = super::rewrite_model(plan, model);

    assert_eq!(out.nodes.len(), 2);
    assert_eq!(out.output, 1);
    assert_eq!(out.inputs, vec![0]);
    assert!(matches!(out.nodes[0].kind, NodeKind::Input { .. }));
    let NodeKind::Op { op, .. } = &out.nodes[1].kind else {
        panic!("expected Op at index 1")
    };
    assert!(matches!(op, StdOp::Linear));
}
```

- [ ] **Step 5: Add unit test #2 — `rewrite_model_drops_victim_and_redirects_consumers`**

```rust
#[test]
fn rewrite_model_drops_victim_and_redirects_consumers() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::NodeKind;
    use crate::ir::StdOp;
    use crate::UirModel;

    // Topology: Input(0) → A=Linear(1) → B=Dropout(2) → C=Linear(3).
    // Plan: victim B(2) → redirect to A(1).
    // Expected: B disappears (3 nodes total), C's operand remaps to
    // A's new id (1 since Input=0, A=1).
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![1], vec![], vec![2, 2]),
            op_node(StdOp::Linear, vec![2], vec![out_dim_attr(2)], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 3,
        source_span: crate::ast::Span::new(1, 1),
    };
    let mut plan = super::RewritePlan::new(&model);
    plan.victims.insert(2, 1); // Dropout(2) → redirect to A(1)
    let out = super::rewrite_model(plan, model);

    assert_eq!(out.nodes.len(), 3); // Dropout dropped
    assert_eq!(out.output, 2); // C's new id (Input=0, A=1, C=2)
    let NodeKind::Op {
        operands: c_ops, ..
    } = &out.nodes[2].kind
    else {
        panic!("expected Op at index 2")
    };
    assert_eq!(c_ops, &vec![1usize]); // C's operand remapped to A's new id (1)
}
```

- [ ] **Step 6: Add unit test #3 — `rewrite_model_pushes_post_ops_to_producer`**

```rust
#[test]
fn rewrite_model_pushes_post_ops_to_producer() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
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
    let mut plan = super::RewritePlan::new(&model);
    plan.victims.insert(2, 1); // Relu → redirect to Linear
    plan.producer_post_ops
        .entry(1)
        .or_default()
        .push(PostOp::Relu);
    let out = super::rewrite_model(plan, model);

    assert_eq!(out.nodes.len(), 2);
    let NodeKind::Op { fused_post_ops, .. } = &out.nodes[1].kind else {
        panic!("expected Op at index 1")
    };
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
}
```

- [ ] **Step 7: Add unit test #4 — `rewrite_model_remaps_model_inputs_and_output`**

```rust
#[test]
fn rewrite_model_remaps_model_inputs_and_output() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::StdOp;
    use crate::UirModel;

    // Topology: Input(0), Input(1), A=Linear(2) using Input(0),
    //   B=Dropout(3) using A. Inputs=[0, 1], output=3 (Dropout victim).
    // Plan: victim B(3) → redirect to A(2).
    // Expected: out.inputs unchanged ([0, 1] — Input nodes are not
    //   victims), out.output = 2 (A's new id since Input(0)=0,
    //   Input(1)=1, A=2).
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            input_node("y", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![2], vec![], vec![2, 2]),
        ],
        inputs: vec![0, 1],
        output: 3,
        source_span: crate::ast::Span::new(1, 1),
    };
    let mut plan = super::RewritePlan::new(&model);
    plan.victims.insert(3, 2); // Dropout → A
    let out = super::rewrite_model(plan, model);

    assert_eq!(out.nodes.len(), 3); // Dropout dropped
    assert_eq!(out.inputs, vec![0, 1]); // Input nodes survive at same new ids
    assert_eq!(out.output, 2); // A's new id
}
```

- [ ] **Step 8: Add unit test #5 — `rewrite_plan_new_counts_consumers_correctly`**

```rust
#[test]
fn rewrite_plan_new_counts_consumers_correctly() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::StdOp;
    use crate::UirModel;

    // Topology: Input(0) → A=Linear(1), then both B=Dropout(2) and
    // C=Relu(3) consume A. model.output = C(3).
    // Expected counts:
    //   Input(0): 1 (consumed by A's operand list)
    //   A(1): 2 (consumed by B and C)
    //   B(2): absent from map (no consumers — B is an orphan node)
    //   C(3): 1 (model.output += 1)
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![1], vec![], vec![2, 2]),
            op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 3,
        source_span: crate::ast::Span::new(1, 1),
    };
    let plan = super::RewritePlan::new(&model);

    assert_eq!(plan.consumer_count.get(&0), Some(&1));
    assert_eq!(plan.consumer_count.get(&1), Some(&2));
    assert_eq!(plan.consumer_count.get(&2), None); // absent
    assert_eq!(plan.consumer_count.get(&3), Some(&1));
}
```

- [ ] **Step 9: Run all 5 helper unit tests**

```sh
cargo test -p compiler --lib passes::rewriter
```

Expected: 5 tests PASS.

- [ ] **Step 10: Run full workspace tests + clippy + fmt**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Total tests = 202 baseline + 5 new = 207. fmt + clippy clean.

- [ ] **Step 11: Commit (atomic unit 1)**

```sh
git add compiler/src/passes/rewriter.rs compiler/src/passes/mod.rs
git commit -m "feat(m7/rewriter): shared 3-step rebuild helper module

Adds compiler/src/passes/rewriter.rs implementing the plan-as-data
helper that the M5/M6 passes (EliminateDropout, FuseLinearRelu,
FuseLinearSoftmax) will migrate onto. Plan stores three HashMaps
(consumer_count precomputed by RewritePlan::new(), victims and
producer_post_ops declared by callers); rewrite_model consumes
both plan and model, returns a fresh UirModel with renumbered
NodeIds.

No Result wrapping — preconditions are caller's responsibility,
violations panic via id_map lookup. Same contract as the existing
per-pass functions (which keep their Result return for Pass::run
compatibility).

Five unit tests pin the helper's behavior independent of any
migrated pass.

Tasks 2-4 of the M7 plan migrate the three existing passes onto
this helper. Atomic-task-pack convention applies."
```

---

## Task 2: Migrate `EliminateDropout`

**Spec ref:** §4.7 migration order (simplest first), §8 migration shape (~70 → ~15 lines).

**Files:**
- Modify: `compiler/src/passes/eliminate_dropout.rs`

- [ ] **Step 1: Read the current `eliminate_one_model` to confirm shape**

The current function (lines 50-114 of `compiler/src/passes/eliminate_dropout.rs`) takes `model: &UirModel`, returns `Result<UirModel, PassError>`, identifies Dropout victims via HashSet, walks nodes with id_map, remaps operands, finally remaps inputs/output.

The current `Pass::run` impl (lines 27-33):
```rust
fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
    let mut new_models = Vec::with_capacity(uir.models.len());
    for model in &uir.models {
        new_models.push(eliminate_one_model(model)?);
    }
    Ok(Uir { models: new_models })
}
```

- [ ] **Step 2: Replace `eliminate_one_model` body to use the helper**

Replace the entire `fn eliminate_one_model(...) -> Result<UirModel, PassError> { ... }` (currently ~65 lines) with:

```rust
fn eliminate_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = super::rewriter::RewritePlan::new(&model);

    for (id, node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Dropout,
            operands,
            ..
        } = &node.kind
        else {
            continue;
        };
        // NFL grammar (§stdlib::Signature for `dropout`) guarantees
        // exactly one operand. Any future grammar / hand-built UIR
        // that violates this invariant will panic at the operands[0]
        // index access — same contract as the M5b/M6 inline code.
        debug_assert_eq!(
            operands.len(),
            1,
            "Dropout must have exactly one operand (NFL grammar invariant)"
        );
        plan.victims.insert(id, operands[0]);
    }

    Ok(super::rewriter::rewrite_model(plan, model))
}
```

Signature changes from `model: &UirModel` to `model: UirModel`. Body reads model through `model.nodes.iter()` (borrow ends before `rewrite_model` consumes model).

- [ ] **Step 3: Update `Pass::run` to clone each model before calling**

Replace the existing `Pass::run` impl block with:

```rust
impl UirPass for EliminateDropout {
    fn name(&self) -> &str {
        "eliminate_dropout"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(eliminate_one_model(model.clone())?);
        }
        Ok(Uir { models: new_models })
    }
}
```

The only difference vs current: `model` → `model.clone()` because `eliminate_one_model` now consumes its argument.

- [ ] **Step 4: Tighten imports**

Update the use list at the top of `compiler/src/passes/eliminate_dropout.rs` from:

```rust
use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind};
use crate::ir::StdOp;
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};
```

to:

```rust
use super::{PassError, UirPass};
use crate::ir::types::NodeKind;
use crate::ir::StdOp;
use crate::{Uir, UirModel};
```

(Drop `Node`, `NodeId`, `HashMap`, `HashSet` no longer needed.)

If clippy's unused-import lint catches anything else, drop those too. Run `cargo clippy --workspace --all-targets -- -D warnings` after this step to confirm.

- [ ] **Step 5: Update the stale doc-comment about "M7-deferred trigger"**

Find the doc-comment at lines 36-49 (above `eliminate_one_model`). Replace it with:

```rust
/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this.
///
/// Implementation: delegates the rebuild skeleton (identify victims →
/// rebuild with remap → remap inputs/output) to
/// `compiler::passes::rewriter`. Each Dropout becomes a victim that
/// redirects to its sole operand; no producer mutation. See spec
/// `docs/superpowers/specs/2026-05-06-m7-rebuild-helper-design.md`
/// §4 / §5 for the helper's design.
```

- [ ] **Step 6: Run all eliminate_dropout tests**

```sh
cargo test -p compiler --lib passes::eliminate_dropout
```

Expected: all 8 existing tests PASS without test-body modifications.

- [ ] **Step 7: Run full workspace tests + clippy + fmt**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: 207 tests pass. fmt + clippy clean.

- [ ] **Step 8: Commit (atomic unit 2)**

```sh
git add compiler/src/passes/eliminate_dropout.rs
git commit -m "refactor(m7/passes): migrate EliminateDropout onto shared rewriter

EliminateDropout::eliminate_one_model now delegates to
compiler::passes::rewriter::rewrite_model. Body shrinks from ~65
lines to ~15 lines: just the Dropout-victim identification loop.
Imports trimmed. Doc-comment retired — the M7-deferred trigger
fired and was closed by Task 1's helper.

Signature change: eliminate_one_model now consumes UirModel rather
than borrowing &UirModel. Pass::run clones each model before
calling. Total clone count per pass is unchanged from M6 behavior —
clone moves from inside the function body (per-node clone in the
rebuild loop) to once per model at the boundary.

All 8 eliminate_dropout unit tests pass without test-body
modifications."
```

---

## Task 3: Migrate `FuseLinearRelu`

**Spec ref:** §4.7 migration order, §8 migration shape (~100 → ~25 lines).

**Files:**
- Modify: `compiler/src/passes/fuse_linear_relu.rs`

- [ ] **Step 1: Read the current `fuse_one_model` to confirm shape**

The current function (lines 43-141) builds consumer_count, identifies Relu victims (sole-operand, single-consumer Linear producer with empty fused_post_ops), walks nodes with id_map, pushes `PostOp::Relu` to producers, remaps inputs/output.

- [ ] **Step 2: Replace `fuse_one_model` body to use the helper**

Replace the entire `fn fuse_one_model(...) -> Result<UirModel, PassError> { ... }` body (currently ~98 lines) with:

```rust
fn fuse_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = super::rewriter::RewritePlan::new(&model);

    for (relu_id, relu_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Relu,
            operands,
            ..
        } = &relu_node.kind
        else {
            continue;
        };
        if operands.len() != 1 {
            continue;
        }
        let linear_id = operands[0];
        let NodeKind::Op {
            op: StdOp::Linear,
            fused_post_ops,
            ..
        } = &model.nodes[linear_id].kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion.
        }
        if *plan.consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Relu).
        }
        plan.victims.insert(relu_id, linear_id);
        plan.producer_post_ops
            .entry(linear_id)
            .or_default()
            .push(PostOp::Relu);
    }

    Ok(super::rewriter::rewrite_model(plan, model))
}
```

Keep the existing doc-comment (lines 38-42) above the function.

- [ ] **Step 3: Update `Pass::run` to clone each model before calling**

```rust
impl UirPass for FuseLinearRelu {
    fn name(&self) -> &str {
        "fuse_linear_relu"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model.clone())?);
        }
        Ok(Uir { models: new_models })
    }
}
```

- [ ] **Step 4: Tighten imports**

Update use list at top of file to:

```rust
use super::{PassError, UirPass};
use crate::ir::types::{NodeKind, PostOp};
use crate::ir::StdOp;
use crate::{Uir, UirModel};
```

(Drop `Node`, `NodeId`, `HashMap`, `HashSet`.)

- [ ] **Step 5: Run all fuse_linear_relu tests**

```sh
cargo test -p compiler --lib passes::fuse_linear_relu
```

Expected: all 8 existing tests PASS without test-body modifications.

- [ ] **Step 6: Run full workspace tests + clippy + fmt**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: 207 tests pass. fmt + clippy clean.

- [ ] **Step 7: Commit (atomic unit 3)**

```sh
git add compiler/src/passes/fuse_linear_relu.rs
git commit -m "refactor(m7/passes): migrate FuseLinearRelu onto shared rewriter

FuseLinearRelu::fuse_one_model now delegates to
compiler::passes::rewriter::rewrite_model. Body shrinks from ~98
lines to ~28 lines: just the Relu-victim identification loop.
Imports trimmed.

Signature change: fuse_one_model now consumes UirModel; Pass::run
clones each model before calling. Same total clone count vs M6.

All 8 fuse_linear_relu unit tests pass without test-body
modifications, including the bias-aware path and multi-consumer
guard. The 6 cross-pass tests in passes/tests.rs and the 3 FFI
integration tests also pass — the migration is bit-exact
equivalent to the M5b/M6 code."
```

---

## Task 4: Migrate `FuseLinearSoftmax`

**Spec ref:** §4.7 migration order, §8 migration shape (~95 → ~25 lines).

**Files:**
- Modify: `compiler/src/passes/fuse_linear_softmax.rs`

- [ ] **Step 1: Read the current `fuse_one_model` to confirm shape**

The current function (lines 37-130) is structurally identical to FuseLinearRelu's — only `StdOp::Softmax` and `PostOp::SoftmaxRow` differ.

- [ ] **Step 2: Replace `fuse_one_model` body to use the helper**

Replace the entire body (currently ~93 lines) with:

```rust
fn fuse_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = super::rewriter::RewritePlan::new(&model);

    for (softmax_id, softmax_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Softmax,
            operands,
            ..
        } = &softmax_node.kind
        else {
            continue;
        };
        if operands.len() != 1 {
            continue;
        }
        let linear_id = operands[0];
        let NodeKind::Op {
            op: StdOp::Linear,
            fused_post_ops,
            ..
        } = &model.nodes[linear_id].kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion.
        }
        if *plan.consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Softmax).
        }
        plan.victims.insert(softmax_id, linear_id);
        plan.producer_post_ops
            .entry(linear_id)
            .or_default()
            .push(PostOp::SoftmaxRow);
    }

    Ok(super::rewriter::rewrite_model(plan, model))
}
```

Keep existing doc-comment (lines 32-36) above the function.

- [ ] **Step 3: Update `Pass::run` to clone each model before calling**

```rust
impl UirPass for FuseLinearSoftmax {
    fn name(&self) -> &str {
        "fuse_linear_softmax"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model.clone())?);
        }
        Ok(Uir { models: new_models })
    }
}
```

- [ ] **Step 4: Tighten imports**

Update use list at top of file to:

```rust
use super::{PassError, UirPass};
use crate::ir::types::{NodeKind, PostOp};
use crate::ir::StdOp;
use crate::{Uir, UirModel};
```

- [ ] **Step 5: Run all fuse_linear_softmax tests**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax
```

Expected: all 5 existing tests PASS without test-body modifications.

- [ ] **Step 6: Run full workspace tests + clippy + fmt**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: 207 tests pass. fmt + clippy clean.

The 6 cross-pass tests in `passes/tests.rs` (especially `pipeline_eliminates_dropout_before_fusing_linear_softmax`) and FFI integration test `fused_vs_unfused_softmax_match_numerically` are critical — they verify M6 attention-pattern fusion still produces bit-exact output post-migration.

- [ ] **Step 7: Commit (atomic unit 4)**

```sh
git add compiler/src/passes/fuse_linear_softmax.rs
git commit -m "refactor(m7/passes): migrate FuseLinearSoftmax onto shared rewriter

FuseLinearSoftmax::fuse_one_model now delegates to
compiler::passes::rewriter::rewrite_model. Body shrinks from ~93
lines to ~28 lines. Imports trimmed.

Signature change: fuse_one_model now consumes UirModel; Pass::run
clones each model before calling.

All 5 fuse_linear_softmax unit tests pass without test-body
modifications. The pipeline test
pipeline_eliminates_dropout_before_fusing_linear_softmax in
passes/tests.rs and the FFI integration test
fused_vs_unfused_softmax_match_numerically (bit-exact equivalence
on classifier.nfl + softmax_with_bias.nfl) both pass — the
migration preserves M6's attention-pattern fusion behavior
bit-exactly.

Three-of-three migrations complete. The shared rewriter helper
now serves all M5/M6 passes."
```

---

## Task 5: Spec §8 invariant 6 unit test

**Spec ref:** Spec §9 Task 2 (`leaves_linear_dropout_softmax_chain_untouched`). Closes M6 Finding #7.

**Files:**
- Modify: `compiler/src/passes/fuse_linear_softmax.rs::tests`

- [ ] **Step 1: Add the unit test**

In `compiler/src/passes/fuse_linear_softmax.rs::tests`, add this test after the existing `identity_when_no_softmax`:

```rust
#[test]
fn leaves_linear_dropout_softmax_chain_untouched() {
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
    use crate::ir::types::NodeKind;
    use crate::ir::StdOp;
    use crate::{Uir, UirModel};

    // Construct: Input → Linear → Dropout → Softmax.
    // Run ONLY FuseLinearSoftmax (NOT default_pipeline — that would
    // EliminateDropout first and remove the blocking Dropout).
    // Spec §8 invariant 6 (arm64.md §4.10 in M6): Linear's sole
    // consumer is Dropout, not Softmax → criterion 2 fails → no
    // fusion happens.
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
    let uir = Uir {
        models: vec![model],
    };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    // Untouched: 4 nodes preserved, Linear's fused_post_ops empty,
    // Softmax-node still exists at index 3.
    assert_eq!(m.nodes.len(), 4);
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
        panic!("expected Op at index 1")
    };
    assert!(fused_post_ops.is_empty(), "Linear should not be fused");
    assert!(matches!(
        m.nodes[3].kind,
        NodeKind::Op {
            op: StdOp::Softmax,
            ..
        }
    ));
}
```

The inline `use crate::{Uir, UirModel};` is intentional self-documenting redundancy — module-level convention already imports them, but explicit inline imports make this test compile-self-contained for spec readers.

- [ ] **Step 2: Run the new test**

```sh
cargo test -p compiler --lib passes::fuse_linear_softmax::tests::leaves_linear_dropout_softmax_chain_untouched
```

Expected: PASS.

- [ ] **Step 3: Run full workspace tests + clippy + fmt**

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: 208 tests pass (207 from Tasks 1-4 + 1 new). fmt + clippy clean.

- [ ] **Step 4: Commit**

```sh
git add compiler/src/passes/fuse_linear_softmax.rs
git commit -m "test(m7/fuse-linear-softmax): pin §8 invariant 6 (degradation case)

Adds leaves_linear_dropout_softmax_chain_untouched in
fuse_linear_softmax::tests. Closes M6 holistic-review Finding #7
(coverage gap for the spec §8 invariant 6 degradation case).

Test: Input → Linear → Dropout → Softmax UIR. Runs ONLY
FuseLinearSoftmax (NOT default_pipeline — which would
EliminateDropout first). Asserts: 4 nodes preserved, Linear's
fused_post_ops empty, Softmax-node intact.

The invariant: Linear's sole consumer is Dropout, not Softmax →
FuseLinearSoftmax criterion 2 (consumer-must-be-Softmax) fails →
no fusion happens. Documented in arm64.md §4.10 invariant 6 and
in compiler/src/passes/fuse_linear_softmax.rs's victim-criteria
loop, but lacked direct test coverage until M7.

Workspace 207 → 208."
```

---

## Task 6: Closeout

**Spec ref:** §10 documentation updates, §13 process checkboxes (holistic review + carry-forward re-evaluation).

**Files:**
- Modify: `PROJECT_SPEC.md`
- Modify: `CLAUDE.md`
- Modify: `DEVLOG.md`

- [ ] **Step 1: Run a holistic-review subagent dispatch**

Spawn a single subagent (general-purpose) with the prompt:

> Review the M7 implementation against the spec at
> `docs/superpowers/specs/2026-05-06-m7-rebuild-helper-design.md`.
> Check: (a) every §-numbered requirement in the spec has corresponding
> code/tests, (b) the four atomic-task-pack commits each leave the
> workspace green, (c) cross-cutting consistency — `#[non_exhaustive]`
> cascade arms unaffected, no stale "M6+ deferred" doc-comments left
> in `eliminate_dropout.rs`, no plan-language ("Task N" referring to
> M7 plan tasks) leaked into committed source comments, (d) M7
> carry-forward debt items (OQ-7, OQ-8, OQ-9 from spec §11) and
> M6/M5c carry-forwards still properly recorded in the upcoming
> DEVLOG entry. Report findings as a numbered list with citation
> paths and line numbers. Aim for 5-10 findings of varying severity.

Wait for the report. Triage findings into "close in M7 close-out" vs "carry-forward to M8+".

- [ ] **Step 2: Address close-in-M7 findings**

For each finding to close in M7: edit the file, run `cargo fmt + clippy + test --workspace`, commit with message `chore(m7/holistic): close finding N — <one-line>`.

Iterate until the close-in-M7 list is empty.

- [ ] **Step 3: Update `PROJECT_SPEC.md` milestone table**

Find the milestones table (around line 152-161). Currently the M7 row reads "Human-readable viewer v0.1". Replace with two rows:

```markdown
| 7 | Shared 3-step rebuild helper extraction (complete) | New `compiler/src/passes/rewriter.rs` (`pub(crate) struct RewritePlan` + `pub(crate) fn rewrite_model`); plan-as-data API (three HashMaps + one constructor); migration of three existing passes (`EliminateDropout`, `FuseLinearRelu`, `FuseLinearSoftmax`) onto the shared helper, each pass body shrinks from 70-100 lines to 15-25 lines; closes M6 holistic-review Finding #1 (three-strikes-then-refactor trigger fired in M6, deferred to M7); §8 invariant 6 unit test added (closes M6 Finding #7); atomic-task-pack convention demonstrated via 4 sequential clean commits |
| 8 | Human-readable viewer v0.1                     | Show UIR in annotated human-readable format       |
```

- [ ] **Step 4: Update `CLAUDE.md` "Current Status" section**

Find the "Current Status" section. Replace with the M7-closed equivalent. Match the M5c/M6 detail level — this is a closure milestone. Use the structure shown in the spec §10 (overview of helper, migration outcomes, test count, carry-forward candidate list).

Key facts to include:
- Helper at `compiler/src/passes/rewriter.rs`, `pub(crate)`.
- Three pass migrations completed; bodies shrink 70-100 → 15-25 lines.
- 208 tests passing (M5c 189 + M6 13 + M7 6).
- All gates clean.
- M7 only touched compiler-side code; arm64.md and uir.md unchanged.
- M8 candidate list (priority-ordered): OQ-7 per-pass Result cleanup, OQ-9 NodeMutation generalisation, OQ-8 lift to ir/, viewer (PROJECT_SPEC M8 row), attention extension (NFL v0.2), `FuseLinearPostOp` (M5c OQ-1), bare-metal `expf` (M5c OQ-3), `BuildError::span()` (M5c OQ-4).

- [ ] **Step 5: Update `CLAUDE.md` Design Principle 5**

Find Design Principle 5 in `CLAUDE.md`. The current text mentions the dedicated viewer tool with `(M7+)`. Replace with `(M8+)`. Match the existing word order; only change the milestone tag.

- [ ] **Step 6: Add a DEVLOG entry**

Append a new entry at the top of `DEVLOG.md` (after the format-spec block, before the M6 entry). Date: 2026-05-06. Standard sections: What was done, Decisions made, Problems encountered, Holistic review process, Known tech debt, Next step.

The entry must include:
- All key M7 deliverables (helper, three migrations, invariant test, doc-comment retirement).
- Pre-decided architectural calls from spec §4 (plan-as-data, no lifetime, consume-model, eager consumer_count, plain UirModel return, `victims` naming, migration order, atomic-task-pack convention).
- Workflow lessons (the "verify HEAD worktree state" lesson from spec brainstorm point-1 confusion).
- Carry-forward to M8+ (OQ-7, OQ-8, OQ-9 plus M5c/M6 inheritance).
- Next step pointing to M8 brainstorm in fresh worktree.

- [ ] **Step 7: Run all the gates one last time**

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: ALL green. Total = 208.

- [ ] **Step 8: Commit closeout**

```sh
git add DEVLOG.md PROJECT_SPEC.md CLAUDE.md
git commit -m "chore(m7): close Milestone 7 — full cycle complete

PROJECT_SPEC.md milestones table M7 row added marked complete;
'Human-readable viewer v0.1' relocated from M7 to M8 row.
CLAUDE.md Current Status rewritten to reflect M7-closed reality.
CLAUDE.md Design Principle 5 reference (M7+) → (M8+) for viewer tool.
DEVLOG.md entry per the project's documentation protocol.

What landed in M7 (across 5 task commits + drift-fixes + closeout):
- compiler/src/passes/rewriter.rs — pub(crate) struct RewritePlan +
  pub(crate) fn rewrite_model. Plan-as-data API; no closures, no
  traits; rewrite_model returns plain UirModel.
- 5 helper unit tests in rewriter.rs::tests.
- Three pass migrations: EliminateDropout, FuseLinearRelu,
  FuseLinearSoftmax. Each body shrinks 70-100 → 15-25 lines while
  preserving identical behavior.
- leaves_linear_dropout_softmax_chain_untouched test in
  fuse_linear_softmax::tests — closes M6 Finding #7.
- Stale eliminate_dropout.rs doc-comment about M7-deferred trigger
  retired.
- Atomic-task-pack convention demonstrated through 4 sequential
  clean commits.

Carry-forward to M8+ (recorded in DEVLOG):
- OQ-7 per-pass Result cleanup.
- OQ-8 lift rewriter to compiler/src/ir/.
- OQ-9 generalise producer_post_ops to enum NodeMutation.
- M5c/M6 carry-forwards continue per their respective triggers."
```

- [ ] **Step 9: Verify final history**

```sh
git log --oneline origin/main..HEAD
cargo test --workspace 2>&1 | grep "test result:"
```

Expected: M7 commits visible; total tests 208.

---

## Done. What's next?

Per the M7 spec §11, the carry-forward debt list for M8+ is:

- **OQ-7** per-pass `Result<UirModel, PassError>` cleanup — fires on first real `Err` or accumulated `Ok(...)` boilerplate discomfort.
- **OQ-8** lifting `rewriter.rs` to `compiler/src/ir/` — fires when non-pass UIR-rewrite consumer appears.
- **OQ-9** generalising `producer_post_ops` to `enum NodeMutation` — fires on fourth pass with non-PostOp producer mutation.
- **M5c/M6 carry-forward** (still open per respective triggers): OQ-1, OQ-2, OQ-3, OQ-4, OQ-6, M6 item 2 (`_expf` smoke), M6 item 4 (CLI smoke future-proofing).

**Brainstorm M8 in a fresh worktree** once M7 merges to main.
