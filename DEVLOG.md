# NeuralForge — Development Log

This file is the living record of the project. Every session gets an entry.
Entries are in reverse-chronological order (newest at the top).

Format for each entry:
```
## YYYY-MM-DD — <one-line summary>
### What was done
### Decisions made
### Problems encountered
### Next step
```

---

## 2026-05-04 — Milestone 4b closed: arm64 profile covers all 5 M3 fixtures end-to-end

### What was done
- Redesigned `FnSig` ABI: `weight_floats` removed, replaced by `params_floats`
  + `params_layout: Vec<ParamSlot>` with typed slots (`LinearWeight`,
  `LinearBias`). Generated functions take a single packed `params` buffer
  containing all weights and biases in topological UIR-node order.
  **This is a deliberate ABI break vs M4a** — see "ABI break callout" below.
- Added `profiles/arm64/src/buffer.rs`: `assign_buffers` (BufferLoc per node:
  InputReg / OutputReg / StackOffset / Alias), `compute_is_leaf`,
  `compute_callee_saved` (RegSet for d8/d9 + x19_x23). `BufferAssignment`
  carries 16-byte aligned total stack size.
- New prologue/epilogue helpers in `asm.rs`: `format_function_prologue` /
  `_epilogue` accept `LeafKind` + `RegSet` + intermediate-bytes. Conditional
  layers: callee-saved x19-x23 (iff softmax), callee-saved d8/d9 (iff
  softmax), non-leaf x29/x30 (iff bl present), sub/add sp (iff intermediates
  > 0). Large-immediate handling via shifted-12 or movz/movk + sub sp, sp,
  x9. New `emit_imm32` helper for arbitrary 32-bit immediate materialisation.
- Refactored `codegen.rs` body emission into `profiles/arm64/src/ops/`
  submodules (mod, linear, relu, softmax, dropout). Per-op emitters take
  `model_idx` + per-op counter for label namespacing across multi-model
  fixtures (e.g. pipeline_styles.nfl → labels like `.Lmm_i_<m>_<l>:`).
- New ops:
  - `linear[N, bias=true]`: matmul + bias-add inline after k-loop.
  - `dropout`: zero asm; `BufferLoc::Alias(operand)` propagation.
  - `softmax`: 3-pass numerically stable (max → exp+sum → normalize),
    `bl _expf`, callee-saved s8/s9 for max+sum, callee-saved x19-x23 for
    loop state across `bl _expf` (i, row_base, k, src ptr, dst ptr).
    `-inf` materialisation via `movz w0, #0x0000; movk w0, #0xFF80, lsl #16;
    fmov s8, w0` (since `fmov sN, #-inf` is invalid AArch64).
- Errors for `linear[N, bias=true]`, `dropout`, `softmax`, and duplicate
  model names are no longer emitted by the lowerer (all paths supported).
  Duplicate-model-name check moved up to `compiler::ir::build` as
  `BuildErrorKind::DuplicateModelName { name, first_span }`.
  `render_error_with_snippet` extended with optional `first_span` →
  emits trailing `note: previously defined at file:line:col` plain-text
  (single-snippet for M4b; rustc-style two-snippet upgrade is M4c-or-later).
- New fixture-driven integration tests via FFI: `tinymlp_full_with_softmax_runs_correctly`,
  `classifier_runs_correctly`, `pipeline_styles_runs_correctly`,
  `comments_runs_correctly`, `mixed_args_runs_correctly`. Plus
  `m4a_no_softmax_still_runs` adapted for the new ABI. All run on
  aarch64 macOS host; skip cleanly elsewhere.
- 2 reference-validation tests (`reference_softmax_stable_known_values`,
  `reference_bias_add_known_values`) pin hand-computed values so an
  asm-and-reference shared bug can't silently pass integration tests.
- `docs/profile_guide/arm64.md` extended (~270 lines added) with bias-add,
  softmax 3-pass, dropout aliasing, intermediate buffer pattern, non-leaf
  prologue with d8/d9 + x19-x23, per-model label namespacing, libm
  dependency note. Limitations greatly reduced.
- `docs/language_reference/uir.md` cross-links to the arm64 guide for both
  optional-attribute interpretation and dropout-as-noop semantics.
- `PROJECT_SPEC.md` milestones table M4 row updated to '4a + 4b complete';
  Architecture Profiles arm64 row expanded.

### ABI break callout

> **M4b deliberately broke the M4a public ABI of `FnSig`.** `weight_floats`
> field is gone; replaced by `params_floats` + `params_layout: Vec<ParamSlot>`.
> The generated `nfl_forward_*` C function signature changes the second
> parameter from `const float* weights` to `const float* params` (semantically
> the same buffer for M4a-compatible models — single LinearWeight slot — but
> renamed to reflect the more general layout).
>
> **Why deliberately:** the M4a name `weight_floats` would have been a lie
> the moment any M4b-supported model used `bias=true` (`params` then
> contains a LinearBias slot too). Renaming + restructuring at M4b is
> correct; retrofit-compat shims would have been worse.
>
> No external consumers exist (project is internal v0.1). Future readers of
> git history: this break was intentional, see
> `docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` §5.4.

### Critical bug caught + fixed during code review

Initial Task 8 (softmax) implementation kept loop state in caller-saved
registers (x3, x4, x5, x6, x11, x12) across `bl _expf`. Per AAPCS64, x0-x18
are caller-saved; `_expf` is allowed to clobber them. Apple libm `expf`
happens to preserve them today, but that's coincidence not contract — non-
Apple targets or libm updates would silently break.

Fix: moved loop state into callee-saved x19-x23 (i, row_base, k, src, dst);
x6 (element offset) is recomputed after each call. RegSet gained `x19_x23`
bit; prologue saves `x19, x20, x21, x22, x23` (two stp pairs + one str)
when softmax is present.

This is the kind of bug that could pass all integration tests on Apple
silicon but blow up on first Linux-arm64 CI run. Defensive fix landed in
the same Task 8 cycle as the spec/quality reviews.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-04-m4b-arm64-coverage.md` (12 tasks; ~21
commits including review-polish commits).

### Problems encountered
- Task 8 critical AAPCS64 violation (caller-saved registers across
  `bl _expf`) — caught by code review, fixed before integration tests ran.
- Task 9 implementer made signature changes to emit_linear/emit_relu (added
  `model_idx` for multi-model label namespacing) but session was
  interrupted before they could update softmax.rs and codegen.rs's
  dispatch arms. Resumed inline: completed the model_idx threading
  through softmax.rs + walk_uir/walk_model + the test assertions on
  label format.

### Known tech debt (carried forward)
1. Single-snippet rendering for `DuplicateModelName` (plain-text note for
   first_span). Two-snippet rustc-style upgrade is M4c-or-later.
2. Integration test tempdir not cleaned up (carried from M4a).
3. Performance: scalar code, mul-based indexing, no fusion, no SIMD. M5+.
4. `LowerError::UnsupportedOp` kept defensively (`#[allow(dead_code)]`);
   exercised by `unsupported_op_display_and_span_round_trip` to prevent
   Display/span impl rot.
5. Bare-metal arm64 target needs a separate profile (Taylor `exp` instead
   of libm). M7+.

### Next step
**Milestone 4 fully complete (4a + 4b).** All 5 M3 positive fixtures lower
end-to-end through the arm64 profile to runnable native code. 148 tests
pass; build/clippy/fmt/CI all clean.

The next milestone is **Milestone 5 — kernel fusion pass**: introduce an
optimisation pass on the UIR (or just-before-codegen) that fuses
`linear → relu` (and similar elementwise-after-matmul patterns) into a
single loop with the relu inlined into the matmul store. Recovers M4a's
in-place relu performance and sets up the framework for more aggressive
fusion (matmul→bias→relu→softmax_max etc.). Brainstorming for M5 runs in a
fresh worktree once main is updated post-M4b-merge.

---

## 2026-05-03 — CI workflow added; closes M3a tech-debt #3

### What was done
- Added `.github/workflows/ci.yml` with two jobs:
  - `unit` on `ubuntu-latest`: `cargo fmt --all -- --check`, `cargo clippy
    --workspace --all-targets -- -D warnings`, `cargo build --workspace`,
    `cargo test --workspace`. Integration test in `profiles/arm64/tests/`
    self-skips on x86_64, so unit-test count is just lexer/parser/ir/profile-unit.
  - `integration` on `macos-14` (Apple Silicon arm64): `cargo build --workspace`
    + `cargo test --workspace`. The full FFI integration test runs here
    (assembles via `cc`, dlopens .dylib, calls `nfl_forward_M4Demo` via
    libloading).
- Triggers: push to `main`, push to `claude/**`, PR to `main`.
- Uses `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2` for
  toolchain + cache.
- Added CI badge to `README.md`.
- Pre-CI: applied `cargo fmt --all` across workspace (117 sites in 20 files).
  Pure formatting, no semantic changes; committed separately so the CI PR
  is review-friendly.

### Decisions made
- **Format check IS in CI** (not just installed). Per the user's note:
  installing rustfmt but never running it is wasted seconds on every CI run.
  Project culture is zero-warnings; format is part of that.
- **Two jobs, not one matrix.** macOS arm64 is paid; ubuntu is free. Splitting
  jobs lets the cheap one fail-fast on lint/fmt without burning macOS minutes.
- **No nightly, no msrv matrix.** YAGNI for v0.1. Single `stable` toolchain.
- **No coverage.** YAGNI for v0.1. Tarpaulin/llvm-cov can come later if needed.

### Problems encountered
- 117 fmt-drift sites across 20 files when first checked. Resolved by running
  `cargo fmt --all` and committing as a separate `style:` commit before the
  CI YAML.

### Next step
**M3a tech-debt #3 closed.** CI now gates every push to main / `claude/*` and
every PR. The next milestone is **Milestone 4b** — bias=true in linear,
dropout (no-op pass-through), softmax (scalar exp). Brainstorming starts in a
fresh worktree once this CI PR merges to main.

---

## 2026-05-03 — Milestone 4a closed: arm64 scalar codegen — first machine-executable output

### What was done
- Workspace restructured into 3 crates: `compiler/` (lib only), `nflc/` (bin
  only), `profiles/arm64/` (lib only). Empty placeholder dirs
  `profiles/{generic,x86_64,riscv64}/` deleted. `compiler` package renamed
  from `nflc` to `compiler`. 25 mechanical `nflc::` → `compiler::` import
  rewrites across `nflc/src/main.rs`, `compiler/tests/uir_fixtures.rs`,
  `compiler/tests/fixtures.rs`. Stale `.gitkeep` markers removed.
- `profiles/arm64` lib crate. Public surface: `pub fn lower(uir: &Uir) ->
  Result<Asm, LowerError>`. Types: `Asm`, `FnSig`, `LowerError`
  (`#[non_exhaustive]`, 4 variants). Internal modules: `codegen.rs` (UIR
  walker, per-op emitters, classify_op upfront validation), `asm.rs`
  (function header/footer helpers + Mach-O symbol prefix), `tests.rs` (10
  unit tests).
- Lowering covers `linear[N]` without bias (matmul: 3 nested scalar loops
  with `fmadd`, `mul`-based index arithmetic), `relu` (separate elementwise
  loop with `fmov s4, wzr` once + `fmax s3, s3, s4` per element, in-place
  on `x2` output buffer), and `Input` (marker, no code).
- Errors for `linear[N, bias=true]`, `dropout`, `softmax`, and duplicate
  model names — all routed through M3c's `render_error_with_snippet` for
  CLI output.
- New `nflc compile <file.nfl> --profile <name> [-o <path>]` subcommand.
  Validates profile strictly (only `arm64` accepted in M4a). Default output
  goes to stdout; `-o` writes to a file.
- New fixture `tests/fixtures/m4_linear_relu.nfl` (the only positive
  fixture that doesn't terminate in `softmax`). UIR-build test mirrors the
  M3b per-fixture submodule style.
- End-to-end integration test: builds the M4a fixture's UIR, lowers to asm,
  assembles + links to a `.dylib` via `cc -shared -arch arm64`, dlopens via
  `libloading` (dev-dep, justified per spec §11), calls
  `nfl_forward_M4Demo` with deterministic input + weights, compares output
  against a pure-Rust matmul+relu reference. **Test passed first time with
  the planned `1e-5` tolerance — no FMA divergence flake.**
- New `docs/profile_guide/arm64.md` (217 lines): ABI, buffer layout,
  supported ops, asm patterns, error variants, recipes for adding new ops
  and new arch profiles, M4a limitations.
- `docs/language_reference/uir.md` cross-links to the arm64 guide for the
  optional-attribute interpretation.
- `PROJECT_SPEC.md` milestones table M4 row updated; "Architecture Profiles"
  table loses `generic` row, gains `arm64` row as M4 deliverable.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-03-m4a-arm64-codegen-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-03-m4a-arm64-codegen.md` (12 tasks, 13
commits — Task 1 split into restructure + cleanup-of-stale-`.gitkeep`).

### Project principle formalised in M4a spec §11

> **Dependency policy.** Production crates (`compiler`, `nflc`,
> `profiles/arm64` lib-target) — strict **std-only**. Adding a non-std
> production dep requires a separate explicit decision and PR.
> **Dev-dependencies** are admissible by need; M4a starts the list with
> `libloading` (used only in `profiles/arm64`'s integration test).

### Plan-bug discovered + fixed during execution
- The plan's NFL test strings used `"model M [b=2]: x: Tensor[b, 3]\n    ..."`
  (no `\n` after `:`). The parser requires `\n` after the model header
  before any body statement. Task 2 implementer caught this and fixed the
  test string to `"model M [b=2]:\n    x: Tensor[b, 3]\n    ..."`. Pattern
  propagated to Tasks 3-6 prompts. Behaviour under test unchanged.

### Problems encountered
- The empty `profiles/{generic,x86_64,riscv64}/` placeholder dirs each had
  a `.gitkeep` marker that `rmdir` couldn't remove. Solved with `git rm`
  on each `.gitkeep` (which also removes the now-empty dir).
- Two stale `.gitkeep` files (`profiles/.gitkeep`, `profiles/arm64/.gitkeep`)
  were caught by Task 1's reviewer; cleaned in commit `a317772` before
  proceeding.
- No FP divergence flake — `1e-5` tolerance was sufficient first try.

### Known tech debt (carried forward)
1. Model-name uniqueness check lives in `profiles/arm64::walk_uir` for now;
   spec §15 says move it up to `compiler::ir::build` in M4b.
2. Multi-Linear weight layout: M4a `FnSig.weight_floats` reports the total
   count only. M4b adds `weights_layout: Vec<WeightSlot>` with per-matrix
   offsets when multi-Linear models become lowerable (and need bias).
3. Integration test tempdir is left in `/tmp` after the test (no Drop-based
   cleanup). Acceptable for v0.1; revisit in M4c if it becomes noisy.
4. CI is still TODO (M3a tech-debt #3).
5. Performance: scalar code, `mul`-based indexing, no fusion, no SIMD. M5+.
6. `ShapeNotConcrete` reused for "no inputs" case in walk_model — semantically
   different from "shape unresolved". Add dedicated variant in M4b cleanup.

### Next step
**Milestone 4a complete.** First time NeuralForge produces real
machine-executable code: an `.s` text file → `.dylib` → callable function
that gives numerically correct output (matmul+relu of f32 inputs).

The immediate next step is **Milestone 4b — softmax + bias + dropout**:
- Lower `linear[N, bias=true]` (4-th `bias` parameter, `FnSig.weights_layout`).
- Lower `dropout` (no-op pass-through at inference).
- Lower `softmax` (scalar `exp` via Taylor series with range reduction OR
  link `expf` from libm).
- Result: all 5 M3 positive fixtures lower end-to-end.
- Move duplicate-model-name check up to `compiler::ir::build`.

Brainstorming for M4b runs in a fresh worktree once main is updated
post-M4a-merge.

---

## 2026-05-03 — Milestone 3c closed: UIR polish — Display impls + source-snippets + reference doc + clippy clean

### What was done
- Added `Display` impls for all UIR types (`Uir`, `UirModel`, `Node`, `Shape`,
  `OpAttr`, `AttrValue`) and for `StdOp`. Output content matches M3b's `print_uir`
  exactly apart from lowercase op names.
- Removed `print_uir`, `print_uir_node`, `format_uir_shape`, `format_uir_attr`
  free functions from `compiler/src/main.rs` (~50 lines deleted; replaced by one
  `print!("{}", uir)` line).
- Added `render_error_with_snippet` helper in `main.rs` (~20 lines, hand-rolled
  std-only). Routes all CLI error paths through it (parse, build, --tokens).
  Output mirrors rustc/cargo conventions (`error:` line, `--> file:line:col`
  pointer, `^` underline).
- Replaced `format!("{:?}", std_op)` with `format!("{}", std_op)` in
  `BuildError::invalid_attr_value`. Error messages now use lowercase op names
  (`dropout.rate` not `Dropout.rate`), matching the NFL source token names.
- Created `docs/language_reference/uir.md` (198 lines): UIR semantics, data
  shape, node kinds, stdlib ops, implicit semantics (incl. multi-pipeline
  convention from M3b open-Q4), CLI inspection format, v0.1 omissions list.
- Cleared all `cargo clippy` warnings: 4× `cloned_ref_to_slice_refs` →
  `std::slice::from_ref` (3 in tests.rs, 1 in build.rs — the build.rs site
  was discovered during the audit and not in the original plan), 1×
  `match_like_matches_macro` → `matches!`. M3a tech-debt #6 closed.
- Audited all enum variants for dead code by briefly enabling
  `#![deny(dead_code)]` at the crate root. Findings logged below.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-02-m3c-uir-polish-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3c-uir-polish.md` (7 tasks, 7 commits).

### Project principle formalised in M3c spec §2

> **Add code only when there's a real consumer.** Do not retain "for-future-use"
> variants/functions/types via `#[allow(dead_code)]`. Remove unused items when
> discovered; re-introduce with the first real use (with tests).

**Nuance:** "no real consumer" means *no caller at all*, not "unreached in current
tests". Defensively reachable code (constructed by guard helpers that protect
against future caller bugs) IS used and should be kept — documented with a
comment explaining the defensive role.

### Audit results
- `ShapeError::WrongInputCount` — KEPT. The audit (with `#![deny(dead_code)]`)
  did NOT flag it: `single_input()` constructs the variant, so it is genuinely
  reachable, not dead. The spec's prescription to add `#[allow(dead_code)]` was
  empirically unnecessary — would have been a no-op. Added a doc comment to the
  variant explaining its defensive role (catches the class of caller bug where a
  multi-input op slips into single-input shape inference; will be exercised for
  real in M5 when `add`/`concat` arrive).
- No other dead-code findings across the entire crate.

### Problems encountered
- The plan expected 4 clippy warnings; running clippy surfaced 5 (the extra one
  was a `cloned_ref_to_slice_refs` in `build.rs:191` at the `infer_output_shape`
  call site — not in tests.rs as the plan assumed). Fixed alongside the other 4.
- No other surprises. Implementation followed the plan closely.

### Known tech debt (carried forward to v0.2 / M4+)
1. M3a tech-debt items #1–#4 still apply (TypeExpr.name, Span start-only, no CI,
   crate version policy). v0.2.
2. AttrError + ShapeError still two enums. Unification is a v0.2 consideration.
3. Multi-error reporting — first-error-halt continues. v0.2.
4. No CI yet. Add as a small follow-up before M4 ships.
5. The `single_input` defensive guard's `WrongInputCount` path becomes
   exercised-for-real in M5 with multi-input ops.

### Next step
**Milestone 3 fully complete.** The UIR pipeline (lex → parse → build → CLI render)
is production-shaped and well-documented.

The immediate next milestone is **Milestone 4 — generic profile (scalar assembly
codegen)**: implement the first architecture profile that consumes the UIR and
emits scalar assembly for any POSIX target. This is the first time the project
produces actual machine-executable output. The first M4 decision is the assembly
flavour (AT&T x86-64 syntax for `as`, NASM, or LLVM textual IR as a stepping
stone) — to be resolved via a fresh `superpowers:brainstorming` cycle for M4.

---

## 2026-05-02 — Milestone 3b closed: UIR extended to all 5 fixtures + dropout validation + --uir CLI

### What was done
- Refactored `build_op` to take `&Shape` instead of `&[Node]` — eliminated the
  `Vec<Node>` clone in `build_model` (closes M3a tech-debt #5)
- Added `stdlib::validate_attrs` + `AttrError` (`OutOfRange`, `MissingAttr`); validates
  per-op value constraints (currently: dropout rate must be in [0, 1])
- Added `BuildErrorKind::InvalidAttrValue { op, attr, reason }` and wired
  `validate_attrs` into `build_op` between `resolve_args` and `infer_output_shape`
- Added `nflc parse <file> --uir` CLI flag with a compact textual UIR pretty-printer
  using `nN`-style node-id notation (matches what the M7 viewer will use)
- **Fix-up commit `7ad99f6`:** extended `resolve_args` to pre-resolve `Symbol(name)`
  args against `model_params` so `linear[output]` (where `output=10` is a param) builds
  to `linear[10]`. M3a missed this gap; classifier.nfl exposed it during Task 4 e2e.
- Restructured `compiler/tests/uir_tiny_mlp.rs` → `compiler/tests/uir_fixtures.rs`
  with submodules per fixture (`tiny_mlp`, `classifier`, `pipeline_styles`,
  `comments`, `mixed_args`, `negative`)
- 4 new positive integration tests cover the remaining M1 fixtures end-to-end
- New negative fixture `tests/fixtures/negative/dropout_rate_out_of_range.nfl`
  + integration test asserting `InvalidAttrValue` at line 6
- 102 tests passing (81 unit + 12 M2 integration + 9 M3 integration); zero warnings

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3b-uir-all-fixtures-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3b-uir-all-fixtures.md` (8 tasks, 9 commits with
the unplanned Symbol-resolution fix-up).

### Problems encountered
- **Plan defect found during Task 4 e2e verification.** The plan author (me) only
  considered M3a's symbolic-dim resolution as covering the params lookup; missed that
  positional Symbol args (e.g. `linear[output]` where `output` is a param) needed the
  same resolution. Fix-up commit `7ad99f6` extends `resolve_args` to pre-resolve
  `Symbol(name)` args against `model_params` HashMap. Caught by the implementer's
  diligent e2e check on classifier.nfl, not by unit tests (which used integer-only
  positionals). Two new unit tests added (`resolve_args_symbol_resolves_against_params`,
  `resolve_args_symbol_not_in_params_stays_symbol`).

### Known tech debt (carried forward — see spec §9)
1. **M3a tech-debt items #1-#4 still apply** (TypeExpr.name, Span start-only, no CI,
   crate version policy). M3b doesn't address them.
2. **AttrError and ShapeError are two separate enums in stdlib.** If the pattern
   grows, M3c can consider unifying into a single OpError enum.
3. **`--uir` printer lives in main.rs as free-function logic.** M3c moves it onto
   the UIR types as Display impls so libraries (test snapshot tools, IDE plugins,
   the M7 viewer) can consume it.
4. **Multi-pipeline behaviour in v0.1:** documented here that grammar permits
   multiple `pipeline_stmt`s but only the last's output becomes the model output.
   M3c will document this explicitly in `docs/language_reference/uir.md`.
5. **`format!("{:?}", std_op)` in the InvalidAttrValue message** uses Debug to
   render `StdOp` as `"Dropout"`. Good enough for v0.1; M3c may add `Display for StdOp`.
6. **Symbol-resolution placement** — currently in `resolve_args` as a pre-pass.
   Consider folding into a unified semantic-resolution pass when more symbol kinds
   appear (v0.2 may add other symbolic identifiers beyond model_params).

### Next step
Begin **Milestone 3c — UIR polish.** Adds: (1) viewer-friendly `Display` impls for
all UIR types (move `print_uir` from `main.rs` onto the types); (2) Ariadne-style
source-snippet error rendering; (3) `docs/language_reference/uir.md` documenting UIR
semantics including the multi-pipeline convention; (4) cleanup of clippy lints noted
in M3a tech-debt #6; (5) audit of unused enum variants. After M3c, Milestone 3 is
fully closed and we can begin **Milestone 4 — generic profile (scalar assembly
codegen)**.

---

## 2026-05-02 — Milestone 3a closed: UIR vertical-slice 1 shipped (tiny_mlp end-to-end)

### What was done
- Created `compiler/src/ir/` module with `mod`, `types`, `stdlib`, `build`, `error`,
  `tests` files (6 source files)
- Implemented index-based DAG (`Uir { models }`, `UirModel { nodes: Vec<Node> }`,
  `NodeId = usize`) per spec §5.1
- Defined stdlib for 4 operations (`Linear`, `Relu`, `Dropout`, `Softmax`) with per-op
  `signature()` and `infer_output_shape()` — all four reachable from `nflc::ir::*`
- Implemented `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>` covering
  symbolic-dim resolution, op binding, positional/named arg validation, and per-op
  shape inference
- Added integration test for `tests/fixtures/tiny_mlp.nfl` plus 3 negative inline tests
  (`UnknownOp`, `UnknownDim`, `ModelHasNoPipeline`)
- Re-exported `Uir`, `UirModel`, `Node`, `NodeId`, `NodeKind`, `OpAttr`, `AttrValue`,
  `Type`, `Shape`, `StdOp`, `BuildError`, `BuildErrorKind` from the crate root
- 88 tests passing (72 unit + 12 M2 integration + 4 M3a integration); zero warnings

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3a-uir-tiny-mlp-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3a-uir-tiny-mlp.md` (10 tasks, 10 commits).

### Problems encountered
- **Borrow-checker workaround in `build_model`.** Rust forbids passing both `&nodes`
  (read-only context for shape lookup in `build_op`) and `&mut nodes` (where `build_op`
  pushes the new node) simultaneously. Resolved by cloning a `Vec<Node>` snapshot
  before each `build_op` call. Cheap for tiny_mlp's ≤3 nodes; proper refactor is
  M3b's job (see tech-debt below).
- **`AttrValue::Symbol` is genuinely unused in M3a's tests** — only `bias=true` (in
  `mixed_args.nfl`, M3b territory) ever produces it. Caught and tracked in spec §9.1
  before implementation; no surprises in execution.

### Known tech debt (carried forward — see spec §9 plus this session's findings)
1. **`AttrValue::Symbol(String)` is unused in M3a tests.** Will be exercised in M3b
   when `mixed_args.nfl` is built. No `#[allow(dead_code)]` needed because the variant
   is reachable through the `pub use` chain at the crate root.
2. **`OpAttr.name` for positional args reuses `ArgSlot.name` from the signature.**
   Couples consumers to the slot-name string contract. No action in M3a.
3. **`Shape(Vec<u64>)` allocates per shape.** Acceptable for v0.1; revisit if
   profiling shows it matters.
4. **`Type.name` is always `"Tensor"` in v0.1.** Same tech-debt category as M2's
   `TypeExpr.name`. Becomes an `enum TypeKind` in v0.2.
5. **`build_model` clones `Vec<Node>` once per `build_op` call** to satisfy the
   borrow checker. Cheap for M3a's small graphs (≤3 nodes per model). M3b should
   refactor `build_op` to take `&Shape` instead of `&[Node]`, eliminating the clone.
6. **A few `cargo clippy` lints** are present but not blocking (the plan's bar is
   warning-free `cargo build`). Specifically: `&[input.clone()]` in stdlib tests
   triggers `cloned_ref_to_slice_refs`, and `match`-as-bool in `check_arg_type`
   triggers `match_like_matches_macro`. M3c can clean these up alongside the other
   polish items.

### Next step
Begin **Milestone 3b — extend UIR to all 5 fixtures.** Adds: multi-pipeline within a
single model, multi-model files (`pipeline_styles.nfl`), named args in real fixtures
(`dropout[rate=0.2]` from `classifier.nfl`, `linear[16, bias=true]` from
`mixed_args.nfl`), Float and Symbol AttrValue exercised by integration tests,
dropout-rate range validation, plus the `--uir` CLI flag for end-to-end inspection.
The data model and stdlib enum from M3a should not need extension; this is purely
incremental wiring + tests + the borrow-checker refactor mentioned in tech-debt #5.

---

## 2026-05-02 — Milestone 2 closed: NFL Parser prototype shipped (Rust, std-only)

### What was done
- Bootstrapped Cargo workspace at the repo root with member crate `nflc` (`compiler/`); std-only, edition 2021
- Implemented hand-written lexer (`compiler/src/lexer/`):
  - `tokens.rs` — `Token`, `TokenKind`, `LexError`
  - `mod.rs` — `lex(&str) -> Result<Vec<Token>, LexError>` with line-by-line scanning
  - `indent.rs` — `IndentStack` emitting virtual `Indent`/`Dedent` tokens
  - Comments, LF/CRLF newlines, pipeline-continuation rule (grammar §5.2), tab rejection
  - 26 unit tests
- Implemented hand-written recursive-descent parser (`compiler/src/parser/`):
  - One `parse_*` function per EBNF production: `parse_arg_value`, `parse_named_arg`,
    `parse_op_args`, `parse_operation`, `parse_pipeline_stmt`, `parse_dim`, `parse_dim_list`,
    `parse_type_expr`, `parse_variable_decl`, `parse_named_value`, `parse_model_params`,
    `parse_model_stmt`, `parse_model_body`, `parse_model_def`, `parse_nfl_source`
  - 24 unit tests
- Defined typed AST (`compiler/src/ast.rs`) with `Span` on every node
- Implemented `nflc parse <file>` CLI with `--tokens` flag for token-stream debug
- Library entry: `nflc::parse(&str) -> Result<NflSource, ParseError>` (lex + parse)
- Added 7 negative fixtures under `tests/fixtures/negative/`: tabs_in_indent,
  missing_colon, unclosed_bracket, empty_tensor, empty_op_args,
  named_before_positional, bad_dedent
- Integration tests (`compiler/tests/fixtures.rs`): 5 positive + 7 negative — all green
- Removed legacy empty `compiler/{lexer,parser,ir,passes}/` and `compiler/.gitkeep` —
  Rust convention is `compiler/src/<module>/`, the legacy stubs are no longer needed

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m2-parser-prototype-design.md` during brainstorming.
This session executed the plan in `docs/superpowers/plans/2026-05-02-m2-parser-prototype.md`
(20 tasks, 22 commits).

### Problems encountered
- **Plan defect found during Task 16 e2e verification.** `parse_pipeline_stmt` did not
  tolerate `Newline` between a step and the leading `->` of a continuation line, even
  though the lexer correctly suppressed `Indent`/`Dedent` for such lines. Symptom:
  classifier.nfl, pipeline_styles.nfl, mixed_args.nfl all failed to parse end-to-end
  while their unit tests (which used inline-only pipelines) passed. Fix: tolerate one
  `Newline` before each continuation `Arrow` in the parser loop. Committed as `dbb57b1`.
- **Same fix bundle:** `parse_model_body` did not tolerate blank/comment-only `Newline`
  between the model-header `:` `Newline` and the first content line's `Indent`. Symptom:
  comments.nfl failed (its first body line is a comment). Fix: `skip_newlines()` before
  `consume(Indent)`.
- **`unused_mut` ratchet during Task 4.** The plan's literal lex code had `let mut line`
  but never mutated it (newlines arrived in Task 5). Implementer removed `mut` to keep
  zero-warnings; restored it in Task 5 when newline handling landed. Cosmetic, no
  functional impact.
- **`#![allow(dead_code)]` was needed on `parser/mod.rs` until Task 15** wired
  `nflc::parse(&str)` to the `pub(crate)` `parse_*` chain. The plan's "remove on Task 10"
  was wrong — the `cargo build` (lib only, without tests) flagged the chain as unused
  until the public entry point existed. Task 15 removed the directive cleanly.

### Known tech debt (carried forward — see spec §9)
1. **`TypeExpr.name: String`** is fixed to `"Tensor"` for v0.1. When v0.2 introduces
   additional types this becomes either an `enum TypeKind` or a `String` validated by
   the semantic pass. Revisit at start of v0.2 grammar work.
2. **`Span` is start-only.** End-position is omitted in v0.1; add it when the first
   consumer (likely the M7 viewer) demands a full source range.
3. **No CI.** `cargo test` is run manually. Open a small follow-up PR to add a
   GitHub Actions workflow on stable Rust before M3 ships.
4. **Crate version `0.1.0` policy undecided.** Standard semver applies, but bump
   policy for the v0.x series should be agreed before v1.0.
5. **Lexer error formatting:** `LexError::UnknownChar { ch: b as char }` mis-renders
   non-ASCII bytes (e.g. UTF-8 sequences appear as Latin-1 fragments). Cosmetic;
   addresses when error reporting matures (v0.2 / Ariadne-style).
6. **`5.` and `.5` produce `UnknownChar` instead of `BadNumber`.** Spec §5.1 mentions
   `BadNumber` for these forms; current implementation rejects them via a different
   path. Acceptable for v0.1; clean up in v0.2.

### Next step
Begin **Milestone 3 — UIR prototype**: build the Universal IR (computation DAG) from
the AST. The 5 positive fixtures from this milestone parse cleanly and the AST types
are stable. The first M3 decision is the UIR's data shape (DAG node-and-edge
representation, sharing strategy, shape-inference timing) — to be resolved via a
fresh `superpowers:brainstorming` cycle for M3.

---

## 2026-05-02 — Milestone 1 closed: NFL Grammar v0.1 shipped

### What was done
- Wrote `language/grammar.ebnf` (formal ISO/IEC 14977 grammar, inference-only, 24 productions)
- Wrote `docs/language_reference/grammar.md` (human-readable reference, 9 sections, line-by-line walkthrough of `tests/fixtures/classifier.nfl`)
- Added 5 positive fixtures under `tests/fixtures/`: `classifier.nfl`, `tiny_mlp.nfl`,
  `pipeline_styles.nfl`, `comments.nfl`, `mixed_args.nfl`
- Verified all artefacts by manual review: reachability of every production from `nfl_source`,
  reference-doc coverage of every production, hand-trace of every fixture against the grammar

### Decisions made
None new. All design decisions for M1 were captured during brainstorming on 2026-05-02 (entry below)
and recorded in `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`. This session
executed the plan in `docs/superpowers/plans/2026-05-02-nfl-grammar-v0.1.md`.

### Problems encountered
- Verification pass found that the root production `nfl_source` was not named anywhere in
  the reference doc (every other production was covered). Fixed by adding a one-sentence
  mention in §1 Overview.
- A self-noted "spec discrepancy" (six vs seven `pipeline_step`s in a walkthrough) turned
  out to be a false alarm — the spec did not contain that walkthrough; it lives only in
  the reference doc, where the count was already correct.

### Next step
Begin **Milestone 2 — Parser prototype**: implement a parser that consumes `.nfl` files and
produces a typed AST. The 5 fixtures from this milestone become the initial test corpus.
The choice of implementation language (Rust / C++ / Python / …) is the first decision of
M2 — to be resolved via a fresh `superpowers:brainstorming` cycle for M2.

---

## 2026-05-02 — Brainstorming Milestone 1 (NFL Grammar v0.1); loss deferred to v0.2

### What was done
- Started brainstorming session for Milestone 1 using `superpowers:brainstorming` skill
- Confirmed scope (Milestone 1 only — formal EBNF grammar)
- Confirmed coverage baseline (the README example, modulo decisions below)
- Confirmed block structure (Python-style: significant indent, `:` opens, dedent closes)
- Resolved a loss-syntax ambiguity (see Decisions); updated `README.md` and `PROJECT_SPEC.md`

### Decisions made

**v0.1 grammar is inference-only; loss syntax deferred to v0.2.**
The original README example included `-> loss: CrossEntropy` as a pipeline terminator. This
made `->` ambiguous: in every other position it means "transform data through op", but in the
loss form it means "terminate the pipeline and bind a training loss". For a language whose
explicit goal is to be LLM-friendly, that dual meaning is a hazard.
Three alternatives were considered: (α) keep the form but mark it as a terminal production
in the grammar; (β) split `loss: TypeName` out as its own statement parallel to `x: Tensor[…]`;
(γ) treat `loss[CrossEntropy]` as a regular operation. The chosen option is to remove all
training syntax from v0.1 entirely — `->` retains a single meaning, the v0.1 spec stays
small, and a coherent training-syntax design (loss + optimiser + training loop hints) can
be done together in v0.2 instead of bolting on a special case now.

**Milestone 1 produces three artefacts, not just the grammar.**
Approach B was selected: `language/grammar.ebnf` (formal, ISO/IEC 14977) + `docs/language_reference/grammar.md`
(human-readable, with examples) + `tests/fixtures/*.nfl` (canonical valid programs).
Writing the reference doc forces ambiguities in the EBNF to surface; the fixtures become the
golden corpus the M2 parser will be tested against. No parser tooling is committed to at
this stage — fixtures are reviewed by hand for now.

**Block structure: Python-style with 4-space indent; tabs forbidden.**
Matches the README example aesthetic and is token-efficient. Tabs are rejected up front to
avoid the recurring tabs-vs-spaces ambiguity that bites LLM-generated code.

### Problems encountered
- None blocking. The loss-syntax ambiguity was caught and resolved during brainstorming,
  before any grammar was written.

### Next step
Finish the brainstorming design (grammar outline, fixtures plan, acceptance criteria),
write the spec to `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`,
then transition to `superpowers:writing-plans` to produce the implementation plan.

---

## 2026-05-02 — Project founded; architecture designed; initial files created

### What was done
- Conceived the NeuralForge project concept (NFL language + AOT compiler to assembly)
- Designed the full architecture: NFL → UIR → Architecture Profile → Assembly
- Created `PROJECT_SPEC.md` with complete design specification
- Created `CLAUDE.md` with context and workflow instructions for Claude Code + Superpowers
- Created `DEVLOG.md` (this file) and `README.md` for project onboarding
- Set up full directory structure:
  `compiler/`, `profiles/`, `language/`, `viewer/`, `tests/`, `docs/`

### Decisions made

**Language name: NeuralForge (NFL)**
Chosen for its directness — a forge that shapes neural networks.

**AOT compilation to assembly only**
No runtime, no interpreter, no JIT. The device receives a compiled binary.
Rationale: eliminates all framework overhead; suitable for edge devices.

**Universal IR (UIR) as the central abstraction**
All architecture-specific logic lives in profiles, not the language or core compiler.
Rationale: adding a new hardware target requires only a new profile.

**AI-native syntax design**
NFL is co-designed for LLM authoring — explicit shapes, left-to-right pipelines,
no ambiguity. Dual representation: compact for authoring, expanded for tooling.

**Human-readable viewer as a first-class component**
Every IR node must have a viewer rendering. AI-generated code must always be
inspectable by a human.

**Kernel fusion by default**
The compiler must attempt to fuse consecutive operations.
Rationale: memory bandwidth is the bottleneck in neural network inference.

**Initial target profiles: x86-64, arm64, riscv64, generic (scalar fallback)**
Chosen for maximum coverage of current hardware landscape.

**Documentation protocol**
Every session must produce a DEVLOG.md entry. Decisions must be logged with reasoning.

### Problems encountered
- None yet. This was a pure design session.

### Next step
Define the NFL grammar formally using EBNF notation (`language/grammar.ebnf`).
Start with the minimal subset needed for a simple feedforward network:
model declaration, tensor types, and the pipeline operator `->`.

---

*Add new entries above this line.*
