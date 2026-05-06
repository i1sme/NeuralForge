# Milestone 7 — Shared 3-Step Rebuild Helper Extraction — Design

> **Status:** Brainstormed and approved 2026-05-06. To be implemented in
> the `claude/m7-rebuild-helper` worktree.
> **Source:** This spec captures the M7 brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M5/M6 shipped three UIR passes that share an identical 3-step rebuild
skeleton: `EliminateDropout` (M5b), `FuseLinearRelu` (M5a, bias-aware in
M5b), and `FuseLinearSoftmax` (M6). The skeleton is:

1. **Identify victims** (nodes that disappear in the new UIR).
2. **Rebuild + remap** (clone non-victims, remap operand IDs through an
   id_map, optionally mutate kept producers, redirect victim references
   to other nodes' new IDs).
3. **Remap inputs/output** (model.inputs and model.output threaded
   through the same id_map).

The "three strikes then refactor" trigger fired during M6 (the third
identical body shipped) but extraction was deferred to keep M6 focused.
M6 holistic-review Finding #1 and the M6 DEVLOG carry-forward record
explicitly anchor M7 on closing this debt.

M7 ships two atomic deliverables:

1. **Task 1 (anchor) — Shared 3-step rebuild helper.**
   New module `compiler/src/passes/rewriter.rs` exposing
   `pub(crate) struct RewritePlan` (data only — three HashMaps + one
   constructor that precomputes consumer counts) and
   `pub(crate) fn rewrite_model(plan, model) -> UirModel` (executes the
   plan against the model, returning a fresh `UirModel` with renumbered
   NodeIds). Migrate all three existing passes onto the helper. Each
   pass body shrinks from 70-100 lines to 15-25 lines.
2. **Task 2 — §8 invariant 6 unit test.**
   Add `leaves_linear_dropout_softmax_chain_untouched` to
   `compiler/src/passes/fuse_linear_softmax.rs::tests`. Pins the
   `FuseLinearSoftmax`-without-`EliminateDropout` degradation case
   that M6 documented in `arm64.md` §4.10 invariant 6 but did not
   cover with a direct test (M6 holistic-review Finding #7).

A third deliverable — the **atomic-task-pack convention** (M6 holistic-
review Finding #11) — applies as a process constraint at the
plan-write phase, not as a code task. The M7 plan explicitly marks
Task 1's four atomic units (helper-create, EliminateDropout migration,
FuseLinearRelu migration, FuseLinearSoftmax migration) as separate
commits with the workspace green between each.

M7 is compiler-side only. No `profiles/arm64` changes, no asm changes,
no documentation changes outside the standard closeout (`PROJECT_SPEC`
M7 row, `CLAUDE.md` Current Status, `DEVLOG.md` entry, plus the stale
`eliminate_dropout.rs:45-49` doc-comment about the M7-deferred trigger).

## 2. Goal

1. Add `compiler::passes::rewriter` module with two `pub(crate)` items:
   - `struct RewritePlan` holding precomputed `consumer_count` plus
     declared `victims` and `producer_post_ops` maps.
   - `fn rewrite_model(plan: RewritePlan, model: UirModel) -> UirModel`
     executing the plan.
2. Migrate `EliminateDropout::eliminate_one_model` onto the helper.
   Body shrinks; all 8 existing eliminate_dropout tests pass without
   changes to test bodies.
3. Migrate `FuseLinearRelu::fuse_one_model` onto the helper. Body
   shrinks; all 8 existing fuse_linear_relu tests pass without
   changes.
4. Migrate `FuseLinearSoftmax::fuse_one_model` onto the helper. Body
   shrinks; all 5 existing fuse_linear_softmax tests pass without
   changes.
5. Add 5 unit tests in `rewriter.rs::tests` providing direct coverage
   of the helper independent of the migrated passes.
6. Add Task 2's `leaves_linear_dropout_softmax_chain_untouched` test
   in `fuse_linear_softmax.rs::tests`.
7. Update closeout files: `PROJECT_SPEC.md` M7 row → "complete";
   `CLAUDE.md` Current Status rewritten; `CLAUDE.md` Design Principle
   5 reference `(M7+)` → `(M8+)`; `eliminate_dropout.rs:45-49`
   doc-comment retired (helper now exists, no longer "deferred to M7+").
   `DEVLOG.md` entry per project documentation protocol.
8. Holistic-review subagent dispatch before merge (M5c/M6 precedent).

`rewrite_model` returns plain `UirModel` — no `Result` wrapping. The
helper has no real `Err` cases; `Result<UirModel, PassError>` would be
preemptive pessimism contradicting the same YAGNI principle that ruled
out defensive runtime checks of the topological invariant (§4 below).
Callers wrap once at their `Pass::run` boundary.

## 3. Non-goals

The following are deliberately out of M7 scope:

- **Per-pass `Result<UirModel, PassError>` cleanup.** The existing
  `fuse_one_model` / `eliminate_one_model` per-pass functions return
  `Result` despite never producing `Err`. Same YAGNI debt as
  `rewrite_model`. Cleaning up per-pass return types is a separate
  refactor (touches `Pass::run` semantics; see Open Question OQ-7).
- **Generalising `producer_post_ops: Vec<PostOp>` to `enum NodeMutation`.**
  Today `Vec<PostOp>` covers all three passes' producer mutations
  (push `PostOp::Relu` or `PostOp::SoftmaxRow`). Generalisation
  triggered by a fourth pass requiring different mutation kinds (attr
  change, operand replacement). See OQ-9.
- **Lifting `rewriter.rs` to `compiler/src/ir/`.** Helper lives in
  `compiler/src/passes/` since passes are the only consumer today.
  Lift trigger is a non-pass UIR-rewrite consumer (UIR-build phase,
  viewer renderer, etc.). See OQ-8.
- **Defensive runtime validation of `RewritePlan` preconditions.**
  Topological order, redirect-target precedence, and producer/victim
  disjointness are documented preconditions enforced by panic, not
  by `Result::Err`. Same contract as current per-pass functions.
- **Performance benchmarks.** Helper-extracted code must produce
  identical `UirModel` outputs to the current code. Bit-exact FFI
  integration tests already pin this. Benchmarks are M8+ if anyone
  cares.
- **Migration of `fuse_linear_relu.rs` test bodies that hand-build
  `Node` literals.** M5b stable-surface tests; out of M7 scope.
  M7 only touches per-pass function bodies, not their tests.
- **Changes to `Pass::run` trait shape.** Trait stays
  `fn run(&self, uir: &Uir) -> Result<Uir, PassError>`. Per-pass
  `run` impls iterate `&uir.models`, clone each model, hand to the
  per-pass `fuse_one_model` / `eliminate_one_model` (which still
  return `Result`).
- **`profiles/arm64` changes, asm changes, documentation outside
  closeout.**
- **`BuildError::span()` accessor / `Diagnostic` trait** (M5c OQ-4).
  Independent M8+ candidate. Decided pre-brainstorm.

## 4. Pre-decided architectural calls

This section captures decisions made during the M7 brainstorm. Each is
a deliberate choice; alternatives are recorded so the reasoning
survives.

**4.1. RewritePlan-as-data over closure-based or trait-based APIs.**
- *Chosen:* `RewritePlan` struct with three plain HashMap fields plus
  a `new(&UirModel)` constructor. Pass code populates the maps directly
  via mutable borrow.
- *Rejected — closure-based* (`rewrite_model(model, is_victim_fn,
  mutate_producer_fn)`): closure boxing for producer mutation,
  state-tracking between closures via captures awkward.
- *Rejected — trait-based* (`RewriteRule` trait with per-node
  `classify(...) -> RewriteAction { Keep, Skip { target },
  KeepAndMutate { mutator } }`): heap allocation for boxed mutator
  closures, action enum largely overlaps the bookkeeping already
  inside `rewrite_model`.
- *Why data wins:* (a) all three current passes mentally build this
  decision-table — `RewritePlan` makes that implicit construction
  explicit; (b) plan is debuggable as a value (`dbg!(&plan)` works
  before `rewrite_model` runs); (c) no heap allocation in the hot
  path; (d) the plan-as-data shape is the natural form when the work
  decomposes cleanly into "decide what to do" then "do it."

**4.2. No lifetime parameter on `RewritePlan`.**
- *Chosen:* `struct RewritePlan { consumer_count, victims,
  producer_post_ops }` with no `'a`. Plan holds owned data; the
  `&UirModel` reference passed to `new()` is borrowed only during
  construction.
- *Rejected — `RewritePlan<'a>` holding `model: &'a UirModel`:*
  introduces lifetime gymnastics in struct definition, creates a
  fragile dependency (borrow lives as long as plan is reachable),
  blocks `rewrite_model(plan, model: UirModel)` move semantics.
- *Why no lifetime wins:* the plan only needs computed/declared
  data — caller already has the model accessible via its own
  reference, so the plan doesn't need to hold a borrow. Removing
  `'a` simplifies the struct definition without losing any
  functionality.

**4.3. `rewrite_model` consumes `model: UirModel` (Choice 2 in
brainstorm Q3).**
- *Chosen:* `fn rewrite_model(plan: RewritePlan, model: UirModel) ->
  UirModel`. Move semantics throughout — helper takes ownership of
  both inputs and returns a new `UirModel`.
- *Rejected — `model: &UirModel`:* helper would clone nodes
  internally as it walks (current per-pass-code style). Same total
  work, but signature reads as "borrow then produce" rather than
  "consume both then produce". The latter makes data flow explicit.
- *Caller pattern:* `Pass::run` receives `&Uir`, iterates
  `&uir.models`, calls `model.clone()` once per model, hands the
  owned `UirModel` to `fuse_one_model(model)` (which signature also
  becomes consuming). Total clone count per pass is unchanged from
  M5/M6 behavior.

**4.4. `consumer_count` always computed in `RewritePlan::new()`.**
- *Chosen:* Constructor unconditionally walks `model.nodes` to build
  `consumer_count: HashMap<NodeId, usize>`, including the `+1` for
  `model.output`. Passes that don't need it (EliminateDropout) just
  don't reference the field.
- *Rejected — lazy via `RefCell`/`OnceCell`:* O(N) walk is negligible
  (N ≤ ~20 for current fixtures; ~5-10 microseconds total).
  Lazy + memoization would add interior mutability or two-step setup
  with no real saving.
- *Why eager wins:* simpler API (caller never thinks about
  initialisation order), no interior mutability complexity, marginal
  cost. Revisit if N grows to thousands (large attention models).

**4.5. `rewrite_model` returns plain `UirModel`, not `Result`.**
- *Chosen:* Plain return. Helper has no real `Err` cases — all
  preconditions are caller's responsibility, violations panic via
  `id_map[…]` lookup failure.
- *Rejected — `Result<UirModel, PassError>` "reserved for future
  validation":* preemptive pessimism. Same YAGNI principle that ruled
  out defensive runtime checks.
- *Per-pass interface stays `Result`:* `fuse_one_model` /
  `eliminate_one_model` continue to return
  `Result<UirModel, PassError>` (matches `Pass::run` shape, eases
  `?`-propagation). The boundary is one-line `Ok(rewrite_model(plan,
  model))`. Per-pass cleanup is OQ-7, not M7 scope.

**4.6. Field name `victims`, not `redirects`.**
- *Chosen:* `victims: HashMap<NodeId, NodeId>` where key is the
  victim's old NodeId and value is the redirect-target's old NodeId.
- *Why `victims`:* matches the spec/code vocabulary ("victim
  identification step"). The keys are semantically the role
  (nodes-to-eliminate); the value tells you what their references
  redirect to.

**4.7. Migration order: EliminateDropout → FuseLinearRelu →
FuseLinearSoftmax.**
- *Why:* (a) EliminateDropout is simplest (no consumer-count
  constraint, no producer mutation) — lowest blast radius if helper
  has bugs. (b) FuseLinearRelu has the largest test surface (8 unit
  tests + cross-pass) — catches integration mismatches on the second
  migration. (c) FuseLinearSoftmax mirrors FuseLinearRelu —
  structurally identical migration, validated through the M6 unit
  tests.
- Workspace tests must pass after each migration. Atomic-task-pack
  convention applied (§4.8 below).

**4.8. Atomic-task-pack convention (M6 Finding #11) applied to
Task 1.**
- M7 has no asm-side ↔ pass-side mutual dependencies (M7 is
  compiler-side only), so the M6 commit-packing problem doesn't
  arise. But Task 1 itself decomposes into four atomic units that
  must each leave the workspace green:
  - Atomic 1: helper module created with internal unit tests.
  - Atomic 2: EliminateDropout migrated.
  - Atomic 3: FuseLinearRelu migrated.
  - Atomic 4: FuseLinearSoftmax migrated.
- Each atomic unit = one commit. `cargo test --workspace` clean
  between every commit. This is the convention applied
  forward-looking (M6 lessons-learned demonstration).

## 5. `RewritePlan` struct

```rust
// compiler/src/passes/rewriter.rs

use crate::ir::types::{NodeKind, PostOp};
use crate::{NodeId, UirModel};
use std::collections::HashMap;

/// A plan for rewriting a `UirModel`: which nodes disappear (and what
/// their references redirect to), plus which surviving nodes get
/// post-op mutations.
///
/// Build a plan with `RewritePlan::new(&model)` (which precomputes
/// `consumer_count`), populate `victims` and `producer_post_ops`
/// during your pass's victim-identification logic, then hand the plan
/// to `rewrite_model(plan, model)`.
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
```

## 6. `rewrite_model` function

```rust
/// Execute a rewrite plan against `model`. Returns a fresh `UirModel`
/// with renumbered NodeIds; both inputs are consumed.
///
/// Preconditions (caller's responsibility — violations cause
/// `id_map[…]` panics, NOT a `Result::Err`):
///   - `model.nodes` is in topological order: every operand NodeId is
///     strictly less than its consumer NodeId. `compiler::ir::build`
///     guarantees this.
///   - For every `(victim, target)` in `plan.victims`,
///     `target_old_id < victim_old_id` (target appears earlier in
///     topological order — its new id is known when victim is
///     encountered).
///   - Every key in `plan.producer_post_ops` is NOT a key in
///     `plan.victims` (a producer cannot be its own victim).
///   - Every key in `plan.producer_post_ops` references a node with
///     `NodeKind::Op` kind (not `NodeKind::Input`).
///
/// Behavior: walk `model.nodes` by old NodeId. For each node:
///   - If old_id is a victim, set
///     `id_map[old_id] = id_map[victims[old_id]]`.
///   - Else: take ownership of the node (move out of consumed model);
///     remap its operands via `id_map`; if old_id is in
///     `producer_post_ops`, append the listed PostOps to the new
///     node's `fused_post_ops`; push the node into `new_nodes`; set
///     `id_map[old_id] = new_nodes.len() - 1`.
/// Then remap `model.inputs` and `model.output` via `id_map`. Return
/// the new `UirModel`.
pub(crate) fn rewrite_model(plan: RewritePlan, model: UirModel) -> UirModel {
    // implementation per the behavior spec above
}
```

No `Result` return. Per §4.5, callers wrap with `Ok(...)` at their
`Pass::run` boundary.

## 7. File location & visibility

- **File:** `compiler/src/passes/rewriter.rs`. Single-file module,
  matching `fuse_linear_relu.rs` style.
- **Visibility:** `pub(crate)` for everything (the struct, both
  fields-visible-to-passes contract, the `new` constructor, the
  `rewrite_model` function). Helper is implementation detail of the
  `passes` module; nothing outside `compiler` consumes it.
- **Module declaration:** `compiler/src/passes/mod.rs` adds
  `pub(crate) mod rewriter;` immediately before the existing
  pass-module declarations (`pub mod eliminate_dropout;`, etc.) to
  emphasise it's an internal utility, not a pass.
- **Tests:** inline `#[cfg(test)] mod tests` within `rewriter.rs`
  itself. Plus indirect coverage via the migrated pass tests.

## 8. Migration shape

After Task 1, each per-pass function shrinks from explicit 3-step
rebuild logic to: "build plan → identify victims → call
`rewrite_model`".

**EliminateDropout** (current ~70 lines → ~15 lines):

```rust
fn eliminate_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = RewritePlan::new(&model);
    for (id, node) in model.nodes.iter().enumerate() {
        let NodeKind::Op { op: StdOp::Dropout, operands, .. } = &node.kind else {
            continue;
        };
        debug_assert_eq!(operands.len(), 1, "Dropout grammar invariant");
        plan.victims.insert(id, operands[0]);
        // No producer mutation — Dropout's operand is unchanged.
    }
    Ok(rewrite_model(plan, model))
}
```

`Pass::run` impl unchanged in shape: still iterates `&uir.models`,
clones each model (since `Pass::run` receives `&Uir`, the borrowed
`model` from the iter must be cloned to hand ownership to the
consuming `eliminate_one_model`), calls
`eliminate_one_model(model.clone())?`.

**FuseLinearRelu** (current ~100 lines → ~25 lines):

```rust
fn fuse_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = RewritePlan::new(&model);
    for (relu_id, relu_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op { op: StdOp::Relu, operands, .. } = &relu_node.kind else {
            continue;
        };
        if operands.len() != 1 { continue; }
        let linear_id = operands[0];
        let NodeKind::Op { op: StdOp::Linear, fused_post_ops, .. } =
            &model.nodes[linear_id].kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() { continue; }
        if *plan.consumer_count.get(&linear_id).unwrap_or(&0) != 1 { continue; }
        plan.victims.insert(relu_id, linear_id);
        plan.producer_post_ops.entry(linear_id).or_default().push(PostOp::Relu);
    }
    Ok(rewrite_model(plan, model))
}
```

**FuseLinearSoftmax** structurally identical to FuseLinearRelu — only
the matched op (`StdOp::Softmax`) and pushed PostOp (`SoftmaxRow`)
differ.

## 9. Test strategy

### Helper unit tests (5 in `rewriter.rs::tests`)

1. **`rewrite_model_with_empty_plan_is_identity`** — model with one
   `Linear` node; empty plan. Assert: `out.nodes.len() == 1`,
   `out.output == 0`, node structurally identical.
2. **`rewrite_model_drops_victim_and_redirects_consumers`** — model
   `Input → A → B → C` (output = C). Plan: victim B → redirect to A.
   Assert: `out.nodes.len() == 3`, `out.output == 2` (C's new id),
   AND `out.nodes[2].operands[0] == 1` (A's new id) — direct
   coverage of the operand-remap loop.
3. **`rewrite_model_pushes_post_ops_to_producer`** — model
   `Input → Linear → Relu`. Plan: victim Relu → Linear,
   `producer_post_ops[Linear] = vec![PostOp::Relu]`. Assert:
   Linear-node in result has `fused_post_ops == [Relu]`; Relu-node
   absent.
4. **`rewrite_model_remaps_model_inputs_and_output`** — model with
   multiple inputs and an output that passes through a victim chain.
   Assert: both `out.inputs` and `out.output` correctly remapped.
5. **`rewrite_plan_new_counts_consumers_correctly`** — model
   `Input → A → [B, C]` (A has two `Op` consumers) with
   `model.output = C`. Assert: `plan.consumer_count[A] == 2`,
   `plan.consumer_count[C] == 1` (the +1 from `model.output`).

### Integration coverage via existing pass tests

After migration, the following test counts must remain green without
test-body modifications:

- 8 `eliminate_dropout` unit tests.
- 8 `fuse_linear_relu` unit tests.
- 5 `fuse_linear_softmax` unit tests.
- 6 cross-pass tests in `passes/tests.rs`
  (`default_pipeline_is_canonical_order`,
  `run_pipeline_threads_uir_through_passes`,
  `empty_pipeline_returns_input_clone`,
  `pipeline_halts_on_first_error_and_propagates`,
  `pipeline_eliminates_dropout_before_fusing_linear_relu`,
  `pipeline_eliminates_dropout_before_fusing_linear_softmax`).
- 3 FFI integration tests
  (`fused_vs_unfused_classifier_match_numerically`,
  `fused_vs_unfused_mixed_args_match_numerically`,
  `fused_vs_unfused_softmax_match_numerically`).

If any existing test breaks, the migration is wrong. Test-body
preservation is a load-bearing M7 contract.

### Task 2: `leaves_linear_dropout_softmax_chain_untouched`

Location: `compiler/src/passes/fuse_linear_softmax.rs::tests`.

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
    let uir = Uir { models: vec![model] };

    let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
    let m = &out.models[0];

    // Untouched: 4 nodes preserved, Linear's fused_post_ops empty,
    // Softmax-node still exists.
    assert_eq!(m.nodes.len(), 4);
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(fused_post_ops.is_empty(), "Linear should not be fused");
    assert!(matches!(m.nodes[3].kind, NodeKind::Op { op: StdOp::Softmax, .. }));
}
```

Test count delta: +5 (helper unit) + 1 (invariant 6) = **+6**.
Workspace baseline 202 → M7 expected 208.

### Precondition: `compiler/src/ir/test_utils` exists

Task 2 imports `input_node`, `op_node`, `out_dim_attr`, `rate_attr`
from `crate::ir::test_utils`. This module was created in M6 Task 1
(commits `58d9b77` + `ff12ca6`); all four helpers are present as
`pub(crate)`. Verified in M7 worktree HEAD (`2f95203`). M7 is a
consumer of the M6 deliverable, not a creator.

## 10. Documentation updates

- **`PROJECT_SPEC.md` milestones table:** add M7 row marked
  "complete" describing the helper extraction. The current M7 row
  ("Human-readable viewer v0.1") relocates to M8. Result:
  - `| 7 | Shared 3-step rebuild helper extraction (complete) | …`
  - `| 8 | Human-readable viewer v0.1 | …`
- **`CLAUDE.md` "Current Status":** rewrite reflecting helper
  extraction + closure of M6 carry-forward Finding #1 + Task 2
  invariant 6 test added. Match M5c/M6 detail level — this is a
  closure milestone, not a feature milestone, so it's shorter than
  M6's status block.
- **`CLAUDE.md` Design Principle 5:** the existing reference
  `(M7+)` for the dedicated viewer tool becomes `(M8+)` after
  viewer relocation in `PROJECT_SPEC.md`.
- **`compiler/src/passes/eliminate_dropout.rs:45-49`:** the
  doc-comment about "deferred to M7+" becomes stale once the helper
  ships in M7. Replace with a forward-pointer to
  `compiler/src/passes/rewriter.rs` documenting the shared skeleton.
- **`DEVLOG.md`:** new entry per project documentation protocol.
  Sections: What was done, Decisions made, Problems encountered,
  Holistic review process, Known tech debt, Next step.

No other doc files affected. `arm64.md` and `uir.md` are unchanged
(M7 is compiler-side only, no asm or UIR-types changes).

## 11. Open Questions

Each entry includes a **concrete trigger**, not a vague "someday".

**OQ-7. Per-pass `Result<UirModel, PassError>` cleanup.**
*Trigger.* Either a real `Err`-case in pass-level logic, or
discomfort from `Ok(...)` boilerplate accumulates across many passes.
*Action.* Refactor per-pass `fuse_one_model` / `eliminate_one_model`
to return plain `UirModel`. `Pass::run` wraps once at the top.

**OQ-8. Lifting `rewriter.rs` to `compiler/src/ir/`.**
*Trigger.* A non-pass UIR-rewrite consumer appears (UIR-build phase
optimisation, viewer renderer that wants to project a fused view,
etc.).
*Action.* Move module to `compiler/src/ir/rewriter.rs`, change
visibility to `pub(crate)` under `ir/` so passes/ and ir/ both
consume it.

**OQ-9. Generalising `producer_post_ops` to `enum NodeMutation`.**
*Trigger.* A fourth pass requires producer mutation other than
"push PostOp to fused_post_ops" — e.g., attribute change, operand
replacement, type narrowing.
*Action.* Introduce `enum NodeMutation { PushPostOp(PostOp), … }`,
replace `producer_post_ops: HashMap<NodeId, Vec<PostOp>>` with
`producer_mutations: HashMap<NodeId, Vec<NodeMutation>>`. Migrate
existing pass call sites mechanically.

**Carried over from M5c/M6** (still open per their respective
triggers):
- **OQ-1 `FuseLinearPostOp` consolidation** (M6) — third access
  pattern OR second RowWise post-op.
- **OQ-2 type-level `PostOpKind` distinction** (M6) — same as OQ-1.
- **OQ-3 bare-metal `expf`** (M5c) — embedded MCU need.
- **OQ-4 `BuildError::span()` + `Diagnostic` trait** (M5c) — fourth
  error type or generic CLI rendering.
- **OQ-6 `format!`/`to_string()` style consistency** (M5c) — next
  cascade-arm touch.
- **M6 carry-forward item 2** (`_expf` AAPCS64 smoke test) — risk
  retired by FFI tests; smoke test is hygiene.
- **M6 carry-forward item 4** (CLI smoke future-proofing) —
  reactive trigger when an M8+ pass adds the substring
  "eliminate_dropout" to a dynamic available-passes listing.

## 12. Risks & Mitigations

**R1. Helper API doesn't fit one of the three passes.** Subtle
behavior in (e.g.) `FuseLinearRelu` not captured by the plan-as-data
model.
*Mitigation.* Migration order (§4.7) catches mismatches early.
EliminateDropout migration first (simplest, lowest blast radius),
FuseLinearRelu second (largest test surface — 8 unit tests + 6
cross-pass + 3 FFI integration). If discovered, adapt API before
continuing the third migration.

**R2. Migration changes node ordering** — helper produces a
different topological permutation than current code.
*Mitigation.* Spec §6 explicitly says `rewrite_model` walks
`0..model.nodes.len()` in original order — same iteration as the
current per-pass code. Helper unit test #2 directly pins this
invariant; FFI integration tests catch any divergence bit-exactly.

**R3. `consumer_count` precompute changes EliminateDropout
behavior.** Eager construction means EliminateDropout walks the
nodes once even though it doesn't use the data.
*Mitigation.* Zero risk — unused data has no semantic effect.
Verified by all 8 eliminate_dropout tests passing post-migration.

**R4. Same-pass invocation produces different result post-migration.**
This IS the load-bearing requirement, not a "risk."
*Mitigation.* If violated, the migration is wrong. The 21 unit
tests + 6 cross-pass tests + 3 FFI integration tests collectively
guarantee invariance.

**R5. Atomic-task-pack convention misapplied** — author skips
intermediate `cargo test --workspace` runs.
*Mitigation.* Plan-write phase enforces "every atomic unit runs
`cargo fmt + clippy + test --workspace` before commit." Same M5b
discipline. Holistic review at close-out catches drift.

**R6. M6 stale doc-comment in `eliminate_dropout.rs:45-49` not
updated.** The "deferred to M7+" wording becomes false once helper
ships.
*Mitigation.* Done Criteria §13 includes a checkbox for this update.

## 13. Done Criteria

M7 ready for holistic review and merge when **all** hold.

**Helper (Task 1).**
- [ ] `compiler/src/passes/rewriter.rs` exists with
      `pub(crate) struct RewritePlan` (fields:
      `consumer_count`, `victims`, `producer_post_ops`) +
      `pub(crate) fn rewrite_model(plan, model) -> UirModel`.
- [ ] `compiler/src/passes/mod.rs` declares
      `pub(crate) mod rewriter;`.
- [ ] All 5 helper unit tests in `rewriter.rs::tests` pass.
- [ ] `EliminateDropout::eliminate_one_model` migrated. Body
      shrinks from ~70 lines to ~15 lines. All 8 existing
      `eliminate_dropout` unit tests pass without test-body
      modification.
- [ ] `FuseLinearRelu::fuse_one_model` migrated. Body shrinks from
      ~100 lines to ~25 lines. All 8 existing `fuse_linear_relu`
      unit tests pass without test-body modification.
- [ ] `FuseLinearSoftmax::fuse_one_model` migrated. Body shrinks
      from ~95 lines to ~25 lines. All 5 existing
      `fuse_linear_softmax` unit tests pass without test-body
      modification.
- [ ] All 6 cross-pass tests in `passes/tests.rs` pass.
- [ ] All 3 FFI integration tests pass (bit-exact equivalence
      preserved post-migration).
- [ ] Atomic-task-pack convention demonstrated through 4
      sequential clean commits (helper, EliminateDropout migration,
      FuseLinearRelu migration, FuseLinearSoftmax migration).
      `cargo test --workspace` passes between every commit.

**Task 2.**
- [ ] `leaves_linear_dropout_softmax_chain_untouched` test added
      to `compiler/src/passes/fuse_linear_softmax.rs::tests` and
      passes.
- [ ] Test specifically runs `FuseLinearSoftmax.run()` only, NOT
      `default_pipeline()`.

**Precondition (already satisfied).**
- [ ] `compiler/src/ir/test_utils.rs` exists with `input_node`,
      `op_node`, `out_dim_attr`, `rate_attr` helpers (M6 Task 1
      deliverable, verified in M7 worktree HEAD).

**Tooling.**
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
      exits 0.
- [ ] `cargo test --workspace` passes; total = 202 baseline + 6
      new = 208.

**Documentation.**
- [ ] `PROJECT_SPEC.md` M7 row added marked "complete";
      "Human-readable viewer v0.1" relocated to M8 row.
- [ ] `CLAUDE.md` "Current Status" rewritten reflecting helper
      extraction + Task 2.
- [ ] `CLAUDE.md` Design Principle 5: `(M7+)` reference updated to
      `(M8+)` after viewer relocation.
- [ ] `compiler/src/passes/eliminate_dropout.rs:45-49` doc-comment
      updated — no longer "deferred to M7+", now references the
      shared `compiler/src/passes/rewriter.rs` helper.
- [ ] `DEVLOG.md` entry per project documentation protocol.

**Process.**
- [ ] Holistic-review subagent dispatch (M5c/M6 precedent) before
      merge.
- [ ] M5c/M6/M7 carry-forward items re-evaluated; whichever
      triggers fired during M7 are folded into the close-out diff.
