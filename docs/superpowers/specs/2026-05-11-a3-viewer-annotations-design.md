# M16 — A3: Profile-Level Viewer Annotations

**Date:** 2026-05-11
**Strategic axis:** Axis 2 follow-up — A3 (per Strategic Roadmap line in `PROJECT_SPEC.md`).
**Status:** Brainstorm complete; awaiting plan synthesis.

---

## 1. Goal & Non-Goals

### Goal

Ship a profile-aware UIR inspection tool — `nflc inspect <file.nfl> --profile <name>` —
that surfaces, for the **post-pass** UIR (the same graph that gets lowered to assembly):

- **Per-node:** `BufferLoc` (placement: `InputReg(i)` / `OutputReg` / `StackOffset(N)` /
  `Alias(nK)`), output buffer size in bytes, and — for `Linear` / `LayerNorm` ops —
  parameter floats consumed.
- **Per-model:** total stack frame bytes (16-byte aligned), callee-saved register set
  (textual rendering), leaf classification, total params/inputs/output floats.

A3 continues Axis 2 (modelling depth → tooling depth). A2 closed in M15 with the
last brick (FFN); A3 surfaces the per-emitter footprint richness that M13–M15 added
(`add`, `layernorm`, two `linear` in FFN, plus `relu` / `softmax` / `matmul` /
`mulscalar` from earlier milestones — each with distinct BufferLoc and stack-frame
behaviour).

A3 also unblocks Axis 3 (bare-metal `expf`): a structured `Inspection` schema makes
future `nflc inspect --diff before.s after.s` validation trivial — comparison
becomes structural field-equality, not text-grep over rendered output.

### Non-Goals (v1)

These are explicit non-goals to bound scope. Each is a viable follow-up but **not**
in M16:

- **Op-local scratch register footprint.** The `pushq %r12` / `popq %r12` pairs that
  individual emitters issue (e.g. M14 `emit_layernorm` op-local `%r12`/`%r13` for
  affine; M15 LH-4 cleanup added `%r15`/`%rbp` to x86_64). Surfacing these requires
  per-emitter declarative metadata (`fn op_local_saves() -> &[Reg]` on every emitter
  on both profiles — 8+ emitters × 2 profiles). Separate milestone.
- **`--diff before.s after.s`.** Axis 3 prereq tooling. Designed-for in §3 (shared
  `Inspection` schema), but the diff command itself ships separately.
- **`--node <id>` selector.** Single-node inspection drill-down. Future ergonomic.
- **`--format json`.** Machine-readable output. The first downstream consumer
  (Axis 3 diff) runs in-process and reads `Inspection` directly; external tooling
  would justify JSON later.
- **Cross-profile diff in a single command** (e.g. `nflc inspect --profile arm64,x86_64`).
  The schema supports it; the command surface waits for a real consumer.
- **Liveness-based buffer reuse in `assign_buffers`.** Today's allocator is monotonic
  (every non-aliased node gets a fresh stack slot). A3 reports what `assign_buffers`
  produces; it does not optimize the allocator. That is an Axis 1 / perf milestone.

---

## 2. CLI Surface

### New subcommand

```
nflc inspect <file.nfl> --profile <arm64|x86_64>
                        [--no-passes]
                        [--passes <comma-list>]
```

- **Default behaviour:** runs `compiler::passes::default_pipeline()` (parity with
  `nflc compile`). User sees the **post-pass** UIR — the topology that actually
  gets lowered.
- `--no-passes`: skip the pass pipeline. For debugging passes themselves
  ("what would `inspect` show if `FuseLinearRelu` were disabled?").
- `--passes <list>`: filter to listed passes (canonical order preserved).
  Validation logic shared with `nflc compile` — extracted into
  `parse_pass_args` helper in `nflc/src/main.rs` (see §4 Task 4).
- `--no-passes` and `--passes` mutually exclusive (mirror compile).

### Why a new subcommand (vs. `parse --uir-verbose --profile`)

Decided in brainstorm Q1.

1. `nflc parse` family is semantically "parse + UIR-render, no codegen runs". Adding
   `--profile` to it means `parse` silently invokes profile lowering analysis —
   violation of Design Principle #1 (explicit over implicit) inside the very tool
   that should embody it.
2. Separate command keeps growth path open for Axis 3 follow-ups
   (`--diff`, `--node`, `--format json`) without polluting `parse`.
3. Logs and onboarding benefit from the explicit boundary: `parse` = read-only
   over UIR; `inspect` = run profile analyzers and report.

### Print-usage line

`nflc/src/main.rs::print_usage()` gains:

```
  nflc inspect <file.nfl> --profile <arm64|x86_64>   Inspect post-pass UIR with profile annotations
                          [--no-passes]              Skip optimisation passes
                          [--passes <list>]          Run only listed passes (comma-separated)
```

---

## 3. Architecture

### 3.1 New types in `profile-api`

```rust
/// Profile-aware annotation of one Uir, returned by Profile::inspect.
/// One entry per UirModel in the input UIR, in declaration order.
#[derive(Debug, Clone)]
pub struct Inspection {
    pub functions: Vec<FnAnnotations>,
}

/// Annotation for one UirModel under one profile.
/// `nodes.len() == post_pass_model.nodes.len()` — strictly index-aligned with
/// the **post-pass** UirModel that gets lowered (see §6 Pass Interaction).
#[derive(Debug, Clone)]
pub struct FnAnnotations {
    pub fn_sig: FnSig,                    // pre-existing; reused as-is
    pub stack_bytes: usize,               // 16-byte aligned, == arm64/x86_64::asm prologue value
    pub callee_saved: Vec<String>,        // textual register names; per-profile rendering
    pub leaf: bool,                       // == !UirModel::calls_extern_math() for both profiles today
    pub nodes: Vec<NodeAnnotation>,
}

/// Per-node annotation. Index in FnAnnotations.nodes corresponds to NodeId
/// in the post-pass UirModel.
#[derive(Debug, Clone)]
pub struct NodeAnnotation {
    pub buffer_loc: BufferLoc,            // lifted from arm64::buffer to profile-api
    pub output_bytes: usize,              // == element_count * 4 (BYTES_PER_ELEMENT)
    pub params_floats: Option<usize>,     // Some for Linear (weights+bias) and LayerNorm (γ+β)
    pub extra_notes: Vec<String>,         // escape hatch — see Growth Rule below
}
```

### 3.2 `BufferLoc` lift to `profile-api`

**Pre-condition (verified in brainstorm):** `pub enum BufferLoc { InputReg(usize),
OutputReg, StackOffset(usize), Alias(NodeId) }` is **structurally bit-identical**
in `profiles/arm64/src/buffer.rs` and `profiles/x86_64/src/buffer.rs`. The two
copies differ only in doc-comment richness — x86_64's variants carry per-variant
`///` doc comments (referencing `INPUT_REGS[n_inputs + 1]`, `[%rsp + offset]`,
`codegen::resolve_loc`); arm64's enum is bare. Reconciliation is mechanical:
take the x86_64 doc-comments and place them on the lifted definition.

**Lift action:** define `BufferLoc` once in `profile-api/src/lib.rs` (or a new
`profile-api/src/inspection.rs` module); `pub use profile_api::BufferLoc` from
each profile's `buffer.rs`. `compiler::NodeId` is already a `profile-api`
dependency — no new crate dependencies.

### 3.3 `Profile` trait grows

```rust
pub trait Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;
    fn sym_prefix(&self) -> &'static str;

    /// M16 (A3): inspect the UIR under this profile, returning per-model and
    /// per-node annotations matching what lower() would produce.
    fn inspect(&self, uir: &Uir) -> Result<Inspection, LowerError>;
}
```

This satisfies the M9 trait-growth invariant ("trait grows by request, not by
anticipation") — `nflc inspect` is the real consumer. Errors share the existing
`LowerError` type because `inspect` runs the same `analyze()` preamble that
`lower` runs (§4 Task 1) and any failure surfaces there too.

### 3.4 Growth Rule for `NodeAnnotation`

`NodeAnnotation` extends only with fields that are **meaningful on both profiles**.
Profile-specific information (e.g. a profile that gains a unique footprint
dimension) goes into `extra_notes: Vec<String>` rather than as a top-level field.

**Why:** keeps the schema honest cross-profile. If we add `vector_lanes_used:
Option<u8>` for one SIMD profile and the other always returns `None`, the schema
becomes a bag of optional fields and consumer code degrades into per-profile
branching. `extra_notes` is the explicit escape hatch — costs nothing to ignore,
preserves rendering for the profiles that need it.

**Closes the door slightly, but not fully:** if a future field genuinely belongs
on both profiles but with profile-specific structural shape (not just textual),
that's the trigger to revisit this rule — at which point we'd consider a
`profile_extra: Box<dyn Any>` or per-profile-typed extension. Don't pre-build
that machinery now.

### 3.5 `ModelAnalysis` is per-profile private

The shared analysis preamble extracted in §4 Task 1 — `analyze(model: &UirModel)
-> Result<ModelAnalysis, LowerError>` — lives **inside each profile crate**, not
in `profile-api`. `ModelAnalysis` carries `AbiContext` (per-profile type from
`profiles/{arm64,x86_64}/src/abi.rs`), so it cannot be `profile-api`-public.

**Where:** `profiles/{arm64,x86_64}/src/codegen.rs` (or a new sibling module
`analysis.rs` if `codegen.rs` size warrants it — defer the split decision to
implementation).

```rust
// profiles/{arm64,x86_64}/src/codegen.rs (private)
struct ModelAnalysis {
    fn_sig: FnSig,
    assignment: BufferAssignment,
    callee_saved: RegSet,
    leaf: LeafKind,
    abi: AbiContext,
}

fn analyze(model: &UirModel) -> Result<ModelAnalysis, LowerError> { ... }
```

Both `walk_model` (existing) and the new per-profile `inspect_model` consume
`ModelAnalysis`. They diverge only in what they do with it: `walk_model` emits
asm; `inspect_model` packages an `FnAnnotations` and renders `RegSet → Vec<String>`
in profile-specific naming convention.

---

## 4. Implementation Tasks

Six commits, sequenced for bisectability. Each commit has an explicit
**bisect-claim** — a one-sentence invariant that holds at that commit.

### Task 1 — Extract `analyze()` from `walk_model` (both profiles)

- Pull the analysis preamble of `walk_model` (`assign_buffers`, `compute_callee_saved`,
  `compute_is_leaf`, `FnSig` construction, `AbiContext` setup, arity validation)
  into a private `analyze(model) -> Result<ModelAnalysis, LowerError>` function.
- `walk_model` becomes: `let a = analyze(model)?; emit_with(a, ...)`.
- Identical extraction shape on both profiles (different `RegSet` types but same
  control flow).
- **Tests changed:** ±0. No new tests, no test removed.
- **Bisect-claim:** "pure extract-method; asm output bit-identical for all 446
  tests; `cargo test --workspace` clean."

### Task 2 — Lift `BufferLoc` to `profile-api`

- Define `BufferLoc` in `profile-api/src/lib.rs` (copy x86_64's doc-comments —
  richer than arm64's bare enum).
- `pub use profile_api::BufferLoc` in `profiles/{arm64,x86_64}/src/buffer.rs`;
  remove the local definitions.
- All callsites continue to import via `crate::buffer::BufferLoc` — no callsite
  edits needed beyond the two `pub use` lines.
- **Tests changed:** ±0.
- **Bisect-claim:** "type relocation only; all callers re-import via existing path;
  446 tests clean."

### Task 3 — Add `Inspection` types + `Profile::inspect()` + per-profile impl

- New types in `profile-api/src/lib.rs` (or `profile-api/src/inspection.rs` if
  splitting helps): `Inspection`, `FnAnnotations`, `NodeAnnotation`.
- New trait method `Profile::inspect(&self, uir: &Uir) -> Result<Inspection, LowerError>`.
- Per-profile implementation: each profile's `lib.rs` adds `fn inspect`, calling
  the shared `analyze()` from Task 1, then walking `model.nodes` to build
  `Vec<NodeAnnotation>`. `params_floats` derived from `fn_sig.params_layout` by
  matching `origin_node`. `callee_saved: Vec<String>` rendered per-profile from
  `RegSet`:
  - arm64: `RegSet { d8_d9: true, x19_x23: true }` → `["d8-d9", "x19-x23"]`
  - x86_64: `RegSet { callee_saved_int: true }` → `["%rbx", "%r12-%r15"]`
  - Both empty → `[]`
- New unit tests in `profiles/{arm64,x86_64}/src/tests.rs` (or sibling): ~3 per
  profile asserting analyzer wire-up: leaf detection, alias placement, params
  count for Linear/LayerNorm. ≈6 new tests.
- **Tests changed:** +~6 unit.
- **Bisect-claim:** "new analysis API surfaced via Profile::inspect; no CLI yet;
  all old tests + 6 new pass."

### Task 4 — `nflc inspect` CLI + renderer

- New workspace crate `inspect-render/` (lib only): single public function
  `render_inspection(insp: &Inspection, header: RenderHeader) -> String` matching
  the format in §5. Add `inspect-render` to workspace `Cargo.toml` members and
  as a `nflc` dependency. Per Task 5 decision, this crate is also depended on
  by the per-profile integration tests in Task 5.
- `RenderHeader { source_path: &Path, profile: &str, applied_passes: &[&str] }` —
  keeps file-path / profile / pass-list out of the `Inspection` schema (those are
  CLI-invocation context, not analysis output).
- `nflc/src/main.rs` gains `inspect` subcommand dispatch:
  - Reuse `parse_compile_args` logic — refactor it into a shared
    `parse_pass_args(rest: &[String], cmd_name: &str) -> Result<PassArgs, String>`
    helper that both `compile` and `inspect` consume. (`compile` keeps its
    `output: Option<PathBuf>` field, which is `compile`-only.)
  - `run_inspect(args: InspectArgs) -> ExitCode` mirrors `run_compile` shape:
    parse → build UIR → run passes (or skip per `--no-passes` / filter per
    `--passes`) → call `profile_impl.inspect(&post_pass_uir)?` → render → print
    to stdout.
- New CLI smoke tests in `nflc/tests/inspect_cli.rs`: ~2 tests asserting `inspect`
  exits 0 and output contains key markers (`inspect-model`, `loc=`,
  `passes applied:`).
- **Tests changed:** +~2 CLI smoke.
- **Bisect-claim:** "renderer + CLI dispatch wired; goldens not captured yet;
  output stable for in-tree fixtures."

### Task 5 — Capture goldens from real runs

- Run `cargo run -p nflc -- inspect <fixture> --profile <name>` for each of the
  four selected fixtures × two profiles, redirect stdout to
  `profiles/{arm64,x86_64}/tests/inspect/<fixture>.expected.txt`.
- Add the integration test `profiles/{arm64,x86_64}/tests/inspect.rs` (one file
  per profile) that, for each fixture in the table:
  - Reads the fixture, parses, builds UIR, runs default passes, calls
    `Arm64Profile.inspect(&uir)?` (or x86_64), renders via
    `compiler-or-shared::render_inspection`, compares to the captured `.expected.txt`.
- Renderer reuse: `render_inspection` cannot live in `nflc` (which is a binary
  crate and not consumable by integration tests in `profiles/`). **Decision:
  new `inspect-render` workspace crate** (lib only) consumed by both `nflc`
  and per-profile integration tests. Rationale: `profile-api` is the
  schema + trait contract; rendering is formatting policy and has no business
  in the contract crate. One tiny new crate with single responsibility,
  one new line in workspace `Cargo.toml`. Alternative (rendering in
  `profile-api`) was considered and rejected for this reason.
- **Process rule (mandatory):** every number in every `.expected.txt` is captured
  from the actual `cargo run` output. **Zero hand-computed numbers.** This rule
  exists because hand-arithmetic in goldens (e.g. the 144 vs 160 stack-frame
  discrepancy in the brainstorm sketch) produces docs that lie.
- **Tests changed:** +~8 golden integration (4 fixtures × 2 profiles).
- **Bisect-claim:** "8 goldens captured from real runs; `cargo test --workspace`
  clean; first-time format-stability harness in place."

### Task 6 — Documentation & DEVLOG closure

- `docs/profile_guide/arm64.md` — new `## Inspection output` section: example
  invocation + sample output + field reference for `loc=` / `out=` / `params=`,
  `callee-saved` rendering for arm64 (`d8-d9`, `x19-x23`).
- `docs/profile_guide/x86_64.md` — parallel section. x86_64-specific
  `callee-saved` rendering (`%rbx`, `%r12-%r15`).
- `README.md` — repository map row for `nflc inspect` if existing repo-map style
  enumerates subcommands; verify against current README at implementation time.
- `CLAUDE.md` "Current Status" — bumped to M16 + A3 closure + new test count.
- `PROJECT_SPEC.md` — milestone table row for M16; Strategic Roadmap line gets
  A3 marked complete.
- `DEVLOG.md` — standard milestone-closure entry (What was done / Decisions /
  Problems / Next step).
- **Not touched:** `docs/language_reference/uir.md` — A3 is profile-level, not
  UIR-level. The viewer rendering documented there (`--uir`, `--uir-verbose`) is
  unchanged.
- **Tests changed:** ±0.
- **Bisect-claim:** "doc-only; no code changes; all tests still clean."

### Test-count trajectory

| After Task | Total | Δ |
|---|---|---|
| (M15 baseline) | 446 | — |
| 1 (extract `analyze()`) | 446 | 0 |
| 2 (lift `BufferLoc`) | 446 | 0 |
| 3 (`Inspection` + trait + impl) | ~452 | +6 |
| 4 (CLI + renderer) | ~454 | +2 |
| 5 (8 goldens) | ~462 | +8 |
| 6 (docs) | ~462 | 0 |

(Approximate counts; exact numbers verified per commit by `cargo test --workspace`.)

---

## 5. Output Format

Two-line per-node, six-line per-model summary, header with applied-pass list.
Visual continuity with `--uir-verbose` style — same column-aligned op header,
same indent-shifted continuation pattern (`-> fused: <name>` → `loc=...`).

### Sketch — `tiny_mlp.nfl --profile arm64` (compact case)

> **Note:** all numbers below are **placeholder** for the spec — actual values
> will be captured by Task 5 from the first real run. The spec validates
> *format*, not numeric values.

```
inspect tiny_mlp.nfl --profile arm64
  passes applied: FuseLinearRelu, EliminateDropout, FuseLinearSoftmaxRow

inspect-model TinyMLP
  inputs:        [n0]                32 floats (128 B)
  output:        n1                  16 floats (64 B)
  params:        8 floats            (32 B)
  stack frame:   0 bytes             (16-byte aligned)
  callee-saved:  [d8-d9, x19-x23]
  leaf:          no                  (calls _expf via fused softmax_row)

  nodes:
    n0  input "x"      :: Tensor[8, 4]
          loc=InputReg(0)        out=128 B
    n1  linear         :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]    fused=[softmax_row]
          loc=OutputReg          out=64 B    params=8 floats (32 B)
```

### Sketch — `transformer_block.nfl --profile arm64` (rich case)

> Numbers placeholder — see note above.

```
inspect transformer_block.nfl --profile arm64
  passes applied: FuseLinearRelu, EliminateDropout, FuseLinearSoftmaxRow

inspect-model TransformerBlock
  inputs:        [n0, n1, n2]        24 floats total (96 B; 32 B each)
  output:        n7                  8 floats (32 B)
  params:        84 floats           (336 B)
  stack frame:   <captured> bytes    (16-byte aligned)
  callee-saved:  []
  leaf:          yes

  nodes:
    n0  input "x"        :: Tensor[2, 4]
          loc=InputReg(0)        out=32 B
    n1  input "skip1"    :: Tensor[2, 4]
          loc=InputReg(1)        out=32 B
    n2  input "skip2"    :: Tensor[2, 4]
          loc=InputReg(2)        out=32 B
    n3  layernorm        :: Tensor[2, 4]    operands=[n0]    attrs=[affine=true]
          loc=StackOffset(0)     out=32 B    params=8 floats (32 B)
    n4  linear           :: Tensor[2, 8]    operands=[n3]    attrs=[out_dim=8, bias=true]    fused=[relu]
          loc=StackOffset(<N>)   out=64 B    params=40 floats (160 B)
    n5  linear           :: Tensor[2, 4]    operands=[n4]    attrs=[out_dim=4, bias=true]
          loc=StackOffset(<N>)   out=32 B    params=36 floats (144 B)
    n6  add              :: Tensor[2, 4]    operands=[n5, n1]
          loc=StackOffset(<N>)   out=32 B
    n7  add              :: Tensor[2, 4]    operands=[n6, n2]
          loc=OutputReg          out=32 B
```

### Format design choices (rationale, brief)

- **Header line + applied passes** — debuggability ("inspect ran with these
  passes; if `--no-passes` was passed, line reads `passes: skipped`").
- **6-line per-model summary** — fixed shape, "verdict at a glance" before any
  per-node detail.
- **Two-line per-node** — line 1 is `--uir-verbose`-style op summary (familiar);
  line 2 is the new annotation row (indent +6, `key=value` triples). Keeps both
  lines under typical terminal width on the common cases.
- **Textual `callee-saved` rendering** — accepts the lossy concession discussed
  in §3 in exchange for simple cross-profile readability.
- **No `--format json` in v1** — Axis 3 diff consumes `Inspection` directly
  in-process; external tooling case is hypothetical until proven.

---

## 6. Pass Interaction

### Default = post-pass

`nflc inspect` runs `compiler::passes::default_pipeline()` by default. This is
parity with `nflc compile`. The user sees the **same** UIR topology that gets
lowered to assembly:

- `FuseLinearRelu` collapsed `linear → relu` chains: the `relu` node is gone,
  the `linear` node carries `fused_post_ops=[Relu]`.
- `EliminateDropout` removed standalone dropout nodes (M5b).
- `FuseLinearSoftmaxRow` (M6) collapsed `linear → softmax` end-of-network into
  a single linear node carrying `fused_post_ops=[SoftmaxRow]`.

`FnAnnotations.nodes` is index-aligned with the **post-pass** `UirModel.nodes` —
not the source-as-written graph. This is critical: pre-pass alignment would
produce a report whose node IDs don't match what `lower()` actually compiles,
defeating the whole point of A3.

### Flags

- `--no-passes`: pre-pass UIR (debug). Useful for "what would inspect show if I
  disabled fusion?" type questions.
- `--passes <list>`: filter to listed passes (canonical order preserved).
  Validation logic shared with `nflc compile` via `parse_pass_args` helper.
- Both flags emit the same `note: applied passes: ...` and `note: pass order is
  canonical ...` divergence stderr notes that `compile` emits — consistent UX.

---

## 7. Test Strategy

Three orthogonal regression axes, no axis is tautological w.r.t. another.

### Axis 1 — Format stability (golden snapshots)

- **Where:** `profiles/{arm64,x86_64}/tests/inspect.rs` (one integration test
  file per profile, same pattern as existing `tests/integration.rs`).
- **Fixtures (4):**
  | Fixture | Why |
  |---|---|
  | `tiny_mlp.nfl` | Baseline: minimal post-fusion form (`linear` with `fused=[softmax_row]`); non-leaf. |
  | `transformer_block.nfl` | Rich: layernorm + FFN + dual residual; leaf; multi-input N=3. |
  | `self_attention.nfl` | Softmax-heavy + `matmul`; non-leaf; complex BufferLoc traversal. |
  | `dropout_only.nfl` | Edge: dropout-as-output. **Confirmed exists** at `tests/fixtures/dropout_only.nfl` (M3b fixture: `model OnlyDropout` with `x -> dropout[rate=0.1]`, dropout = output). Validates that the M8 dropout-as-output BufferLoc::OutputReg branch surfaces correctly post-pass (whatever EliminateDropout decides to do with this case — captured by golden, not hand-asserted). |
- **Files:** 4 fixtures × 2 profiles = 8 `.expected.txt` files in
  `profiles/{arm64,x86_64}/tests/inspect/<fixture>.expected.txt`.
- **Process rule:** zero hand-computed numbers. Every byte in every `.expected.txt`
  comes from `cargo run -p nflc -- inspect <fixture> --profile <name>` output,
  redirected once and committed.

### Axis 2 — Analyzer semantics (unit tests)

- **Where:** `profiles/{arm64,x86_64}/src/tests.rs` (or sibling `inspect_tests.rs`
  if file size warrants split).
- **Coverage (~3 per profile, ~6 total):**
  - Leaf detection: model with softmax → `inspect.functions[0].leaf == false`;
    model without → `true`.
  - Alias placement: model with `linear → relu` (post-fusion the relu is gone,
    but pre-fusion via `--no-passes` we should see `Alias(operands[0])` for
    relu; choose whichever path gives stable assertion).
  - Params count: `linear[out_dim=8, bias=true]` over input `[B, 4]` →
    `params_floats == Some(4*8 + 8)`.
- These target *analyzer correctness*, not format. They survive renderer changes.

### Axis 3 — CLI dispatch (smoke tests)

- **Where:** `nflc/tests/inspect_cli.rs` (new).
- **Coverage (~2):**
  - `nflc inspect tests/fixtures/tiny_mlp.nfl --profile arm64` exits 0; stdout
    contains `inspect-model TinyMLP`, `loc=`, `passes applied:`.
  - `nflc inspect tests/fixtures/tiny_mlp.nfl --profile arm64 --no-passes` exits
    0; stdout contains `passes: skipped` marker.
- These guard argument parsing, profile dispatch, and the `parse_pass_args`
  helper extraction.

### Drift-prevention by construction

The `analyze()` extract (Task 1) ensures `inspect()` and `lower()` consume
**the same analysis values** by structure — not by test. There is no
"`inspect`-vs-`lower` consistency" test because such a test would verify a
tautology: both methods literally call the same function and pull the same
fields. If a future refactor breaks this invariant (e.g. introduces a second
analyzer copy), the bisect-claim from Task 1 catches it as soon as any
existing test on either path fails.

---

## 8. Documentation Surface

| File | Change |
|---|---|
| `docs/profile_guide/arm64.md` | New `## Inspection output` section: example invocation + sample output + field reference for `loc=` / `out=` / `params=`; arm64 callee-saved rendering (`d8-d9`, `x19-x23`). |
| `docs/profile_guide/x86_64.md` | Parallel section. x86_64 callee-saved rendering (`%rbx`, `%r12-%r15`); SysV-specific notes on `inputs[i]` register mapping in BufferLoc. |
| `nflc/src/main.rs::print_usage()` | Add `inspect` subcommand block (see §2). |
| `README.md` | Repository-map row for `nflc inspect` (only if existing map enumerates subcommands; verify against current state at implementation time, do not over-document). |
| `CLAUDE.md` "Current Status" | Bump to M16; A3 marked closed; updated test count. |
| `PROJECT_SPEC.md` | New milestone table row for M16; Strategic Roadmap A3 marked complete. |
| `DEVLOG.md` | Standard closure entry per Documentation Protocol. |
| `docs/language_reference/uir.md` | **Not touched.** A3 is profile-level, not UIR-level. |

---

## 9. Verification Checklist (workspace gates)

Before each commit:

- `cargo fmt --all` — formatting canonical.
- `cargo clippy --workspace --all-targets -- -D warnings` — exits 0.
- `cargo test --workspace` — passes (count goes up monotonically per the
  trajectory in §4).

Before milestone closure:

- All 8 `.expected.txt` files present and non-empty.
- `nflc inspect` print-usage line shows on `nflc` (no args).
- `nflc compile` behaviour unchanged — bit-identical asm output for all M15
  fixtures (Task 1 invariant + Task 2 type relocation).

---

## 10. Open Questions (this spec)

None at brainstorm closure. Spec is fully specified.

If implementation surfaces a non-obvious choice (e.g. whether to split
`inspect_render` into a new workspace crate or fold into `profile-api`, see
Task 5), record it as a Known Latent Hazard / Open Question in `PROJECT_SPEC.md`
following project process — not by amending this spec post-hoc.
