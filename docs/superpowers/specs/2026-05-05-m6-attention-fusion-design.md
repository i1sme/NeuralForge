# Milestone 6 — Attention-Pattern Fusion (`linear → softmax`) — Design

> **Status:** Brainstormed and approved 2026-05-05. To be implemented in a
> dedicated `claude/m6-…` worktree.
> **Source:** This spec captures the M6 brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M5 closed the UIR-pass framework: `UirPass` trait, `default_pipeline`,
`run_pipeline`, `PassError`, plus two passes (`EliminateDropout` and
`FuseLinearRelu`, the latter bias-aware after M5b). One `PostOp` variant
(`Relu`) is fused into `emit_linear` as a per-element `fmax` inside the
innermost matmul loop. Bit-exact equivalence between fused and unfused
output is pinned by FFI integration tests on classifier and mixed-args
fixtures.

M6 extends this one step:

- Adds a third `PostOp` variant — `SoftmaxRow` — for the row-wise softmax
  computation that follows a final-layer `linear` (with or without bias).
- Adds a new pass — `FuseLinearSoftmax` — parallel to `FuseLinearRelu`.
- Extends `profiles/arm64::emit_linear` with a row-wise emit branch:
  the j-loop matmul (+ optional bias-add) materialises the full row,
  then three sweeps over that row (max → exp+sum → normalize) run
  in-place, with `bl _expf` in the middle pass.
- Pipeline default becomes `[EliminateDropout, FuseLinearRelu,
  FuseLinearSoftmax]`. CLI `--passes` filter picks up the new pass
  automatically through the existing dynamic registry.

A bit-exact FFI integration test mirrors the M5a/M5b pattern. Test-helper
extraction (`compiler/src/ir/test_utils.rs`) lands during M6 as the fourth
hand-built UIR test fires the trigger from the M5c carry-forward debt.

The framing "attention-pattern fusion" is aspirational. NFL v0.1 cannot
yet express attention's Q/K/V projections, transpose, or scaled
dot-product. What M6 actually fuses today is the classifier-output
pattern (`final linear → softmax`). The same pattern *also* shows up in
attention's pre-softmax stage, hence the framing — but real attention
needs additional language and lowering work in later milestones.

## 2. Goal

1. Add `compiler::ir::PostOp::SoftmaxRow` (third variant of `PostOp`).
   Update `Display` and the `#[non_exhaustive]` cascade arms accordingly.
2. Implement `compiler::passes::fuse_linear_softmax` — bias-aware from
   the first commit; fuses `linear → softmax` and `linear[bias=true] →
   softmax`. Mirrors `fuse_linear_relu`'s file structure (single file,
   inline `#[cfg(test)] mod tests`).
3. Extend `default_pipeline()` to `[EliminateDropout, FuseLinearRelu,
   FuseLinearSoftmax]`. CLI `--passes` filter and `--no-passes` continue
   to work without code changes — they read the registry dynamically
   (M5b convention).
4. Extend `profiles/arm64::emit_linear` with a row-wise emit branch
   (full asm shape pinned in §8). Existing Elementwise branch (`Relu`)
   is unchanged.
5. Add FFI integration test `fused_vs_unfused_softmax_match_numerically`
   in `profiles/arm64/tests/integration.rs`. Bit-exact equivalence on
   the classifier fixture (and `softmax_with_bias.nfl` if R3 in §13
   forces a second fixture).
6. Extract `compiler/src/ir/test_utils.rs` reactively when the fourth
   hand-built UIR construct in M6 confirms the trigger from the M5c
   carry-forward debt list. Migrate M5b's three existing manual-UIR
   tests at extraction time.
7. Update documentation: `arm64.md` §3 / new §4.10 / §5 / §8;
   `uir.md` §2; `PROJECT_SPEC.md` M6 row → "complete";
   `CLAUDE.md` Current Status rewrite.

`emit_linear` differs structurally between Elementwise and RowWise
post-ops. **An implementer must not inline `SoftmaxRow` per-element by
analogy to `Relu`** — softmax requires the row-max before exponentiation,
which is not available inside the innermost j-loop. See §8 for the
asm-sketch and the explicit four-phase decomposition.

## 3. Non-goals

The following are deliberately scoped *out* of M6:

- **Decomposing softmax into UIR sub-ops** (`softmax_max`, `softmax_exp_sum`,
  `softmax_div`). Defer until NFL grows syntax in which softmax is no
  longer the final operator. Today's NFL v0.1 is inference-only and
  classifier-shaped; decomposing now would generalise on two examples
  (the same anti-pattern M5b avoided when declining `EliminateNoOp`).
- **Generalising `FuseLinearRelu` and `FuseLinearSoftmax` into a single
  parametrised `FuseLinearPostOp` pass.** Concrete trigger lives in §12
  (OQ-1).
- **Type-level access-pattern distinction** (`enum PostOpKind {
  Elementwise, RowWise }`). With two variants, dispatch via `match` in
  `emit_linear` is sufficient. Trigger in §12 (OQ-2).
- **Bare-metal `expf`** (Taylor / minimax polynomial without libm).
  Independent future milestone (§12 OQ-3).
- **Attention Q/K/V projections, scaled dot-product, masking, axis-N
  softmax.** Real attention requires NFL v0.2 grammar plus several
  lowering milestones. M6 only fuses what NFL can already express.
- **`BuildError::span()` + `Diagnostic` trait.** Carry-forward from M5c
  (§12 OQ-4); waits for a fourth error type or generic CLI rendering
  path.
- **`x86_64` / `riscv64` profiles.** Already marked `(future)` in the
  PROJECT_SPEC architecture profiles table after the pre-brainstorm fix
  (commit `2d550b8`).
- **Performance benchmarking / micro-architecture tuning.** Bit-exact
  equivalence is the M6 contract; throughput/latency comparisons are
  out of scope. Plan revisits this for any future SIMD profile.

## 4. `PostOp::SoftmaxRow` extension

```rust
#[non_exhaustive]
pub enum PostOp {
    Relu,
    SoftmaxRow, // M6
}
```

- `#[non_exhaustive]` was added in M5c. Wildcard arms in
  `profiles/arm64` (covering `LowerError::UnsupportedPostOp`) already
  protect against future variants. M6 adds exactly one variant; the
  wildcard remains as fallback.
- `Display for PostOp` covers the new variant: `SoftmaxRow`. The
  `nflc parse --uir` CLI renders `fused=[SoftmaxRow]` for M6-fused
  classifier final-layer Linears (Design Principle 5: every UIR
  extension stays inspectable through `Display`).
- The doc comment on `PostOp::SoftmaxRow` cross-references the row-wise
  emit shape: *"RowWise semantics — the emit branch in `emit_linear` is
  structurally different from `Relu`'s. See `arm64.md` §4.10."*
- Access-pattern distinction (Elementwise vs RowWise) is **not** encoded
  at the type level in M6. It is documented in `arm64.md` §4.10 and
  realised through the dispatch `match` inside `emit_linear`. The
  type-level promotion waits for a third access pattern or a second
  RowWise variant (§12 OQ-2).

## 5. `FuseLinearSoftmax` pass

- **File:** `compiler/src/passes/fuse_linear_softmax.rs`. Single file,
  inline `#[cfg(test)] mod tests`. Mirrors `fuse_linear_relu.rs`
  structurally.
- **Trait:** implements `UirPass` (introduced in M5a).
- **Fusion victim criteria** (all five must hold for a Linear node to
  be fused with its Softmax consumer):
  1. The Linear node has exactly one consumer in the UIR.
  2. That consumer is a Softmax node.
  3. The Softmax node has the Linear as its sole operand.
  4. The Linear's `fused_post_ops` is empty (i.e., no prior `Relu`
     was fused onto it). This guards against `[Relu, SoftmaxRow]`
     stacks, whose emit shape is not in M6 scope.
  5. Shape compatibility — Softmax's input shape matches Linear's
     output shape. This is a structural invariant of any well-formed
     UIR (the parser/validator enforces it for `op → softmax`
     chains); the pass treats it as a precondition rather than
     adding a runtime guard.
- **Bias-aware from the first commit.** No M5a-style
  `if linear.has_bias { continue; }` guard. M5b proved bias-aware
  fusion is mechanically the same — bias-handling already lives in
  `emit_linear`'s matmul→bias-add chain.
- **Transformation** (functional, returns a fresh `Uir`):
  - Drop the Softmax node.
  - Push `PostOp::SoftmaxRow` onto the Linear's `fused_post_ops`.
  - Remap consumers of the Softmax output (and `model.output` if the
    Softmax was terminal — true for the classifier fixture) to the
    Linear's output id.
- **`PassError` variants:** none added in M6. Either the pattern matches
  (transform applied) or it doesn't (no-op). All internal invariants
  use `unreachable!` for impossible UIR states (consistent with
  `FuseLinearRelu`'s style).
- **No interaction with `FuseLinearRelu` in M6.** The two passes' input
  patterns (`linear → relu` and `linear → softmax`) are disjoint in the
  fixtures NFL v0.1 produces. Criterion 4 (`fused_post_ops.is_empty()`)
  enforces this disjointness by construction even if a future fixture
  produces `linear → relu → softmax` — `FuseLinearRelu` runs first,
  yields a Linear with `fused = [Relu]`, then `FuseLinearSoftmax`'s
  criterion 4 rejects it. The Softmax remains a separate node and
  lowers via the unfused `emit_softmax`.

### Rejected alternative

- **Generalised `FuseLinearPostOp`** (subsuming `FuseLinearRelu`) was
  considered (option C2 in the brainstorming Q3). Rejected on the same
  grounds M5b rejected `EliminateNoOp`: two examples is too few to
  design the abstraction. Trigger captured in §12 OQ-1.

## 6. Pipeline integration & CLI

- **`default_pipeline()`** returns
  `[EliminateDropout, FuseLinearRelu, FuseLinearSoftmax]`.
  - `EliminateDropout` first so that `linear → dropout → softmax`
    collapses to `linear → softmax` before `FuseLinearSoftmax` sees
    it. This is an extension of the M5b invariant
    (`pipeline_eliminates_dropout_before_fusing_linear_relu`).
  - `FuseLinearRelu` before `FuseLinearSoftmax` is alphabetical and
    matches M5b's ordering style. The two passes' patterns are
    disjoint (§5 criterion 4), so the order is not load-bearing for
    correctness — but determinism keeps diagnostics readable.
- **CLI `--passes` filter:** automatic. The CLI reads pass names from
  `default_pipeline().iter().map(|p| p.name())` (M5b convention).
  `fuse_linear_softmax` enters the available-names list with no CLI
  code changes. Existing duplicate / unknown / mutually-exclusive
  validation continues to work.
- **CLI `--no-passes`:** automatic. Without passes, Softmax remains a
  standalone UIR node; `walk_model` in `profiles/arm64::codegen`
  dispatches it to the existing `emit_softmax` path (M4b). This is
  the unfused branch consumed by §9's FFI integration test.
- **Validation invariant** (encoded in the §6 pipeline-level unit
  test): after `default_pipeline()` on the classifier fixture's UIR,
  no Softmax node remains; the final Linear carries
  `fused_post_ops = [SoftmaxRow]`.

## 7. Buffer assignment & memory model

- M5b's `assign_buffers` (`profiles/arm64::buffer.rs`): each non-aliased
  UIR node gets its own static stack-allocated buffer; aliased nodes
  (Dropout in `--no-passes` mode) reuse the operand's buffer via
  `BufferLoc::Alias(operand_id)`.
- **Fused Linear (M5a/M5b convention).** A fused Linear node owns one
  output buffer used for both matmul and post-op output — the post-op
  is "in-place" by virtue of being inlined into the store. No new
  buffers, no `Alias`.
- **Fused Linear with `SoftmaxRow` (M6).** Same single-buffer story:
  one stack-allocated M×N buffer holds the matmul output, the row-max
  scan reads it, the exp pass overwrites it in place, the normalize
  pass overwrites it again. After fusion, the Softmax node is gone;
  `assign_buffers` never sees a separate softmax-output buffer.
- **Per-row scalars (row-max, row-sum):** held in callee-saved float
  registers `s8` / `s9` (AAPCS64 d8–d15 are callee-saved; `bl _expf`
  must preserve them). **No additional stack slots are required for
  these scalars.** The frame layout extension is exactly what unfused
  softmax already triggers (callee-saved area for `lr`/`x30` plus
  `d8`/`d9`) — `compute_is_leaf` returns `false` for any Linear
  carrying `SoftmaxRow` in `fused_post_ops`.
- **Memory savings vs `--no-passes`** (qualitative — exact touch-count
  comparison depends on `emit_softmax` layout, to be verified during
  implementation): fusion removes the cross-op handoff between
  `linear` and `softmax`, eliminates the function-call boundary, and
  potentially saves one M×N buffer if the unfused `emit_softmax`
  allocates its own output buffer. Concrete numbers go in DEVLOG once
  `emit_softmax` is read during implementation phase.

## 8. arm64 RowWise emit shape (asm-sketch)

`emit_linear` dispatches by inspecting the last entry of
`fused_post_ops`:

- empty → matmul (+ optional bias-add) only, no post-op (M4b shape).
- `[Relu]` → Elementwise branch (M5b shape: per-`(i,j)` `fmax` inlined
  before the store).
- `[SoftmaxRow]` → RowWise branch (new in M6, sketched below).
- anything else (stacked variants, future variants) →
  `LowerError::UnsupportedPostOp`. Criterion 4 in §5 prevents stacks
  from arising in M6.

The RowWise branch implements four sequential phases inside the
batch (i) outer loop. Register names below are illustrative; the
actual implementation reuses the register conventions and `emit_sp_*`
helpers established by M4b/M5b's `emit_linear`.

```text
; Outer loop: i in 0..M (batch)
.Lloop_i:

    ; ----- Phase 1: matmul row + optional bias-add -----
    ; The full j-loop materialises output[i, 0..N] before exiting.
    ; This is the same shape as M5b's emit_linear for empty post-ops,
    ; without the per-element post-op inline.
    mov     w_j, #0
.Lloop_j_matmul:
    fmov    s_acc, wzr
    mov     w_k, #0
.Lloop_k:
    ldr     s0, [<addr A[i, k]>]
    ldr     s1, [<addr B[k, j]>]
    fmadd   s_acc, s0, s1, s_acc
    add     w_k, w_k, #1
    cmp     w_k, K
    b.lt    .Lloop_k

    ; bias path — skipped if linear.has_bias = false
    ldr     s_b, [<addr bias[j]>]
    fadd    s_acc, s_acc, s_b

    str     s_acc, [<addr out[i, j]>]
    add     w_j, w_j, #1
    cmp     w_j, N
    b.lt    .Lloop_j_matmul

    ; ----- Phase 2: row-max scan into s8 -----
    ; s8 is callee-saved (AAPCS64), survives bl _expf in Phase 3.
    ; Init max = first element of the row to avoid an -inf literal.
    ldr     s8, [<addr out[i, 0]>]
    mov     w_j, #1
.Lloop_j_max:
    ldr     s0, [<addr out[i, j]>]
    fmax    s8, s8, s0
    add     w_j, w_j, #1
    cmp     w_j, N
    b.lt    .Lloop_j_max

    ; ----- Phase 3: exp(x - row_max) accumulated in s9 -----
    ; s9 is callee-saved. No per-iteration stack store/load.
    fmov    s9, wzr
    mov     w_j, #0
.Lloop_j_exp:
    ldr     s0, [<addr out[i, j]>]
    fsub    s0, s0, s8                ; s8 alive across bl
    bl      _expf                      ; clobbers s0..s7; s8/s9 preserved
    str     s0, [<addr out[i, j]>]    ; in-place
    fadd    s9, s9, s0
    add     w_j, w_j, #1
    cmp     w_j, N
    b.lt    .Lloop_j_exp

    ; ----- Phase 4: normalize using s9 (row sum) -----
    mov     w_j, #0
.Lloop_j_norm:
    ldr     s0, [<addr out[i, j]>]
    fdiv    s0, s0, s9
    str     s0, [<addr out[i, j]>]
    add     w_j, w_j, #1
    cmp     w_j, N
    b.lt    .Lloop_j_norm

    ; outer i++
    add     w_i, w_i, #1
    cmp     w_i, M
    b.lt    .Lloop_i
```

### Invariants encoded by the sketch

1. **Phase 1 finishes before Phase 2 starts.** The j-loop matmul fully
   materialises `out[i, 0..N]` before any reduction reads it. This is
   the structural difference from `Relu`'s emit (where the post-op is
   per-element inside the j-loop). An implementer attempting to inline
   `SoftmaxRow` per-element will produce mathematically wrong output —
   exponentiation requires the row-max, which is unavailable until
   the j-loop completes.
2. **`out` buffer is touched 6 times per row** in the fused version
   (Phase 1: 1 write; Phase 2: 1 read; Phase 3: 1 read + 1 write;
   Phase 4: 1 read + 1 write). The corresponding count for the
   unfused path will be measured during implementation when
   `profiles/arm64/src/ops/softmax.rs::emit_softmax` is read.
3. **`bl _expf`** is invoked N times per row, identical to M4b's
   unfused softmax. `is_leaf=false` for the fused Linear — the
   `compute_is_leaf` logic (M4b) extends to recognise `SoftmaxRow` in
   `fused_post_ops` (R2 in §13).
4. **No new stack slots** for row-max or row-sum. They live in
   `s8`/`s9` (callee-saved) for the duration of Phases 2–4 of each
   row iteration.
5. **No bias path** simply skips the `ldr s_b / fadd s_acc, s_acc,
   s_b` pair in Phase 1. Phases 2–4 are bias-independent.
6. **No `EliminateDropout`** (e.g. `--passes fuse_linear_softmax`
   alone): if the UIR still contains `linear → dropout → softmax`,
   `FuseLinearSoftmax`'s criterion 2 fails — Linear's sole
   consumer is Dropout, not Softmax. (Criterion 1, "exactly one
   consumer", is satisfied: Linear feeds only Dropout in this
   pattern.) The pattern is left untouched. Dropout stays as a
   `BufferLoc::Alias` (M5a fallback), Softmax as its own
   `emit_softmax` call. Correct degradation.

### Errors

- No new `LowerError` variants in M6.
- `LowerError::UnsupportedPostOp` wildcard covers any future
  `PostOp` variant; M6 does not exercise it.
- `LowerError::UnsupportedOp` (M5c-introduced) does not trigger —
  Softmax is a known op pre-fusion; post-fusion the standalone
  Softmax is gone.

## 9. Test strategy

**Per-pass unit tests** in
`compiler/src/passes/fuse_linear_softmax.rs` (`#[cfg(test)] mod
tests`). Minimum:

1. `fuses_linear_softmax_no_bias` — `linear → softmax` produces one
   Linear with `fused_post_ops = [SoftmaxRow]`.
2. `fuses_linear_softmax_with_bias` — `linear[bias=true] → softmax`
   produces one Linear with bias preserved and
   `fused_post_ops = [SoftmaxRow]`.
3. `does_not_fuse_when_post_ops_already_present` — Linear with
   `fused_post_ops = [Relu]` is left alone (criterion 4). M5b's
   post-merge holistic review caught a near-miss of an analogous
   case; this test pins the criterion explicitly.
4. `does_not_fuse_multi_consumer_linear` — Linear with two consumers
   (Softmax + something else) is left alone (criterion 1).
5. `identity_when_no_softmax` — UIR without Softmax is a pass no-op.

**Pipeline-integration test** in `compiler/src/passes/tests.rs`
(the cross-pass test file established by M5b — its
`pipeline_eliminates_dropout_before_fusing_linear_relu` lives
there, not in `fuse_linear_relu.rs`):

6. `pipeline_eliminates_dropout_before_fusing_linear_softmax` —
   `linear → dropout → softmax` after `[EliminateDropout,
   FuseLinearSoftmax]` produces one fused Linear. Mirror of M5b's
   `pipeline_eliminates_dropout_before_fusing_linear_relu`, kept
   in the same file to preserve the convention that cross-pass
   tests live in `passes/tests.rs`.

All six tests would construct UIR by hand under M5b's pre-helper
convention (tests 1–5 in `fuse_linear_softmax.rs`, test 6 in
`passes/tests.rs`). Combined with M5b's three existing manual-UIR
tests, the first M6 test is the four-strikes moment — and so the
helper extraction trigger fires *before* test 1 is written, not
after the sixth boilerplate paste. In practice the six tests are
written through the new helpers from the start; the boilerplate
is never duplicated. See §10 for the explicit order of operations.

**Existing test update** — `default_pipeline_is_canonical_order` in
`compiler/src/passes/tests.rs` (added in M5b) currently asserts
`["eliminate_dropout", "fuse_linear_relu"]` via `assert_eq!`. M6
extends this to `["eliminate_dropout", "fuse_linear_relu",
"fuse_linear_softmax"]`. Mechanical one-line edit; called out
explicitly here so the failing assertion doesn't read as a
regression during implementation.

**CLI smoke** in `nflc/tests/cli.rs`: extend the existing
`--passes` filter test with a `--passes fuse_linear_softmax` case.
Asserts the compiled assembly reflects the fused row-wise emit
shape (substring-style stdout-asm check, mirroring M5b's `fmax
s0, s0, s4` assertion style for fused-Relu). The exact substring
to assert is decided in plan-phase Task 0 once the §8 sketch is
verified against `emit_softmax`.

**FFI integration** in `profiles/arm64/tests/integration.rs`: new
`fused_vs_unfused_softmax_match_numerically`. Structurally a copy of
M5a's `_classifier_match_numerically` and M5b's
`_mixed_args_match_numerically`:

- Compile the classifier fixture (and `softmax_with_bias.nfl` if
  R3 in §13 forces a second fixture) twice — once with
  `default_pipeline()` (fused), once with `--no-passes` (unfused).
- Link both with `cc` + load through `libloading` (test-only dev-dep).
- Call both FFI functions with the same inputs.
- Assert element-wise bit-exact equality of outputs.

Bit-exact tolerance is justified: both branches use the same libm
`expf`, in the same order, with the same rounding mode. Any
divergence would indicate a real bug.

**Test-fixture decision** is deferred to implementation Task 0
(read the classifier fixture's final Linear, check the bias flag).
If bias is already true, the existing fixture covers both fusion
paths. If false, add `tests/fixtures/softmax_with_bias.nfl` as a
mirror of M5b's `mixed_args.nfl`.

## 10. Test-helper extraction

**Trigger:** the moment the first M6 unit test (`§9` test 1) is about
to be written as the fourth hand-built UIR construct in the
codebase. M5b shipped three such constructs (`fuse_linear_relu/tests`
×2, `eliminate_dropout/tests` ×1). The fourth fires the "three
strikes then refactor" rule per the M5c carry-forward debt list.

**Order of operations** (operational requirement that survives into
the writing-plans phase):

1. Extract `compiler/src/ir/test_utils.rs` based on the boilerplate
   pattern observed in M5b's three existing manual-UIR tests.
2. Migrate M5b's three existing tests (`fuse_linear_relu` ×2 +
   `eliminate_dropout` ×1) to use the helpers.
3. Then write M6 unit tests through the helpers from the start.

The alternative ordering ("write all M6 tests with boilerplate
first → extract → migrate everything") doubles the touches on every
M6 test without compensating benefit.

**Location:** `compiler/src/ir/test_utils.rs`. **Visibility:**
`pub(crate)`. Cross-crate access (e.g. from `profiles/arm64/tests/`)
is not required — integration tests construct UIR through the
parser, not by hand.

**Minimal API at extraction time** (driven by the three M5b use
cases plus the first M6 case):

```rust
pub(crate) fn input_node(name: &str, shape: Shape, dtype: Dtype) -> NodeKind { ... }
pub(crate) fn op_node(op: StdOp, args: Vec<NodeId>, shape: Shape, dtype: Dtype) -> NodeKind { ... }
```

Additional knobs (`fused_post_ops` injection, named-arg builders)
are added on demand by the fifth and later use cases. The API is
not designed speculatively.

**Conditional fallback:** if the M6 unit tests turn out to use the
parser instead of hand-built UIR (unlikely given the §9 test list,
but possible), the trigger does not fire. This is recorded
explicitly in DEVLOG as "trigger not fired in M6, deferred to M7+"
and does **not** block the M6 close-out. Acceptable outcome.

## 11. Documentation updates

- **`docs/profile_guide/arm64.md`:**
  - §3 supported-ops table: row `Softmax` extended with the
    fused-vs-`--no-passes` annotation, mirroring M5b's row format
    for Linear / Relu / Dropout.
  - **New §4.10 "Fused linear → softmax (row-wise)"** containing
    the §8 asm-sketch (or a slightly polished derivative) and an
    explicit contrast against §4.9 (Elementwise) explaining why
    the row-wise emit shape cannot be inlined per-element.
  - §5 errors table: row `UnsupportedPostOp` annotation cites
    `SoftmaxRow` as a concrete M6 implementation; no new variants.
  - §8 Limitations: rewrite the post-M5c "only `Relu` post-op
    fuses" line into "`Relu` (elementwise) and `SoftmaxRow`
    (row-wise) post-ops fuse; arbitrary stacking is gated by
    `FuseLinearSoftmax` criterion 4."
- **`docs/language_reference/uir.md`:**
  - §2 `NodeKind::Op` rendering — list `SoftmaxRow` alongside
    `Relu` as a valid value inside `fused_post_ops`.
- **`PROJECT_SPEC.md`:**
  - Milestones table M6 row → "complete" status after close-out
    (the row text was updated pre-brainstorm in commit `2d550b8`).
- **`CLAUDE.md`:**
  - "Current Status" section rewritten to reflect M6 closed,
    test-count update, mention of `SoftmaxRow` + `FuseLinearSoftmax`
    + helper extraction (or the explicit "trigger not fired" note).

## 12. Open Questions

Each entry includes a **concrete trigger** rather than a vague
"someday". Without a trigger an open-question item degrades into a
wishlist that no one rereads.

**OQ-1. `FuseLinearPostOp` consolidation.**
*Trigger.* Either (a) a third access pattern (neither Elementwise
nor RowWise — e.g. column-wise reduction) appears, or (b) a second
RowWise post-op (LayerNorm, attention-axis softmax, etc.) lands and
copy-paste between `FuseLinearRelu`, `FuseLinearSoftmax`, and the
hypothetical third pass becomes obvious pain.
*Action.* Generalise into a single parametrised
`FuseLinearPostOp` pass; `default_pipeline()` collapses to
`[EliminateDropout, FuseLinearPostOp]`.
*Precedent.* M5b refused to generalise `EliminateDropout` into
`EliminateNoOp` on two examples. Same lesson here.

**OQ-2. Type-level access-pattern distinction.**
*Trigger.* OQ-1's trigger plus the emit-shape divergence between
RowWise variants growing large enough that the `match` inside
`emit_linear` has more than two large arms.
*Action.* Introduce `enum PostOpKind { Elementwise, RowWise, ... }`,
associate each `PostOp` variant with one `PostOpKind`, and split
dispatch into Kind → Op two-tier.

**OQ-3. Bare-metal `expf`.**
*Trigger.* User-driven need for bare-metal arm64 deployment
(embedded MCU inference, etc.).
*Action.* New sub-profile `arm64-baremetal` (or feature flag) with
a minimax-polynomial `expf` replacing `bl _expf`. Trade-offs: more
code, ULP-loss, self-containment.
*Relation to M6.* Fused and unfused softmax depend on `bl _expf`
symmetrically — a bare-metal swap touches both branches identically;
fusion is neutral here.

**OQ-4. `BuildError::span()` + `Diagnostic` trait.**
*Trigger.* A fourth workspace error type, or a generic CLI
error-rendering path that suffers from per-type dispatch
duplication.
*Action.* Add `BuildError::span()`; introduce `pub trait
Diagnostic { fn span(&self) -> &Span; fn message(&self) -> String;
… }`; implement on all five error types.
*Carry-forward from M5c (Findings 1.2, 2.1).*

**OQ-5. `debug_assert_eq!` → `assert_eq!` in FFI integration tests.**
*Trigger.* Adding the third `fused_vs_unfused_*` integration test
(M6 *is* the third — M5a's `_classifier_match_numerically`, M5b's
`_mixed_args_match_numerically`, and M6's
`_softmax_match_numerically`).
*Action.* Harmonise all three tests on `assert_eq!` for the
`params_floats` agreement check. One-line edit per test.
*Sub-decision (resolved).* Bundle the fix into M6 implementation
phase rather than a separate close-out item — three tests
together is cheaper than returning later.

**OQ-6. `format!("{op}")` vs `op.to_string()` style.**
*Trigger.* The next cascade arm added to a `#[non_exhaustive]`
match. M6's criterion 4 closes off `PostOp` stacking, so M6 does
**not** add cascade arms; the trigger does not fire here.
*Action.* Harmonise on the next cascade-arm touch in M7+.

## 13. Risks & Mitigations

**R1. RowWise asm-sketch diverges from reality.** Frame layout,
register conventions, or `emit_softmax`'s actual structure differs
from the assumptions in §8.
*Mitigation.* Plan-phase Task 0 reads
`profiles/arm64/src/ops/softmax.rs` and updates the §8 sketch
before any `emit_linear` code is written. Standard M5b discipline.

**R2. `compute_is_leaf` for fused-`SoftmaxRow` Linears returns the
wrong value.** M4b's leaf-detection looks for standalone Softmax
nodes, not `SoftmaxRow` inside `fused_post_ops`.
*Mitigation.* Unit test `is_leaf_false_for_fused_softmax_row_linear`
in `profiles/arm64::buffer::tests` (or wherever
`compute_is_leaf` is tested). Test-driven; written before the
emit branch.

**R3. Classifier fixture's final Linear has `bias = false`.** The
bias-aware path is then not exercised by the FFI integration test.
*Mitigation.* Plan-phase Task 0 checks the bias flag (one-line
inspection). If false, add `tests/fixtures/softmax_with_bias.nfl`
mirroring M5b's `mixed_args.nfl`.

**R4. Helper-extraction trigger does not fire in M6.** E.g. the §9
unit tests end up using the parser rather than hand-built UIR.
*Mitigation.* Acceptable outcome. DEVLOG records "trigger not
fired, deferred to M7+". Does not block M6 close-out.

**R5. `bl _expf` in some libm clobbers `d8`/`d9` despite
AAPCS64.** Possible source: nonstandard libm, statically linked
musl, etc.
*Mitigation.* Smoke test in `profiles/arm64::asm::tests` ahead of
the integration test — direct `_expf` call with a pre/post check
that `s8` retains its expected value. If the smoke test fails,
fall back to stack slots (a two-line edit in the emit branch).
Risk is low: macOS libm and stock glibc both honour AAPCS64.

**R6. M5c carry-forward debt items collide with M6.** OQ-4 / OQ-5
/ OQ-6 triggers may fire mid-M6 implementation.
*Mitigation.* OQ-5 is explicitly bundled into M6 (decided). OQ-4
and OQ-6 do not fire on M6 paths — the M6 close-out re-evaluates
this and migrates any item whose trigger fired into the close-out
diff.

## 14. Done Criteria

M6 is ready for holistic review and merge when **all** of the
following hold.

**Functionality.**
- [ ] `compiler::ir::PostOp::SoftmaxRow` variant + `Display` impl.
- [ ] `compiler::passes::fuse_linear_softmax` module implements
      `UirPass`.
- [ ] `default_pipeline() == [EliminateDropout, FuseLinearRelu,
      FuseLinearSoftmax]`.
- [ ] CLI `--passes fuse_linear_softmax` works; `--no-passes`
      preserves existing behaviour.
- [ ] `profiles/arm64::emit_linear`'s RowWise branch implements the
      §8 asm-sketch (after Plan-phase Task 0 verification).

**Tests.**
- [ ] All §9 unit tests pass.
- [ ] FFI integration `fused_vs_unfused_softmax_match_numerically`
      bit-exact pass.
- [ ] CLI smoke for `--passes fuse_linear_softmax` passes.
- [ ] All three `fused_vs_unfused_*` integration tests use
      `assert_eq!` for the `params_floats` agreement check
      (§12 OQ-5 carry-forward, bundled into M6).
- [ ] Workspace test suite: M5c baseline preserved, total count
      monotonically increases by the new tests committed in this
      milestone.

**Helper extraction (conditional).**
- [ ] Trigger fired: `compiler/src/ir/test_utils.rs` exists, the
      §10 order of operations was followed, M5b's three tests are
      migrated; **or**
- [ ] Trigger not fired: explicit DEVLOG note, deferred to M7+.

**Documentation.**
- [ ] `docs/profile_guide/arm64.md` §3 / new §4.10 / §5 / §8
      updated.
- [ ] `docs/language_reference/uir.md` §2 mentions `SoftmaxRow`.
- [ ] `PROJECT_SPEC.md` M6 row marked "complete".
- [ ] `CLAUDE.md` "Current Status" section rewritten.

**Tooling.**
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
      exits 0.
- [ ] `cargo test --workspace` passes.

**Process.**
- [ ] Holistic-review subagent dispatch (M5c precedent) before
      merge.
- [ ] DEVLOG entry created per the project's documentation
      protocol.
- [ ] M5c carry-forward items (OQ-4 / OQ-5 / OQ-6) re-evaluated at
      close-out; whichever triggers fired during M6 are folded
      into the close-out diff.
