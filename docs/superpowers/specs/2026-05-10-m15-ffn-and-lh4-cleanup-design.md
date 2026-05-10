# Milestone 15 — A2 third brick (FFN) + LH-4 cleanup — Design Spec

**Date:** 2026-05-10
**Status:** approved (brainstorm complete; awaiting implementation plan)
**Predecessor:** M14 (LayerNorm + LH-1/2/3 cleanup) — merged at `b266b0e`
**Strategic position:** A2 third brick of "transformer block" axis (residual + LayerNorm + **FFN**); also closes LH-4 latent hazard opened in M14.

---

## 1. Goals & non-goals

### In scope

1. **LH-4 closure with runtime FFI evidence.** Relocate `%r8` (per-row src ptr) and `%r9` (per-row dst ptr) in `profiles/x86_64/src/ops/layernorm.rs` to free registers (`%r15` / `%rbp`). Close the §"Known Latent Hazards" entry opened in M14.
2. **A2 third brick — FFN feature.** Demonstrate `linear → relu → linear` as a compositional NFL pattern. No new `StdOp` variant, no IR changes, no codegen pattern — both `linear` and `relu` are existing emitters on both profiles.
3. **Two new positive fixtures:** `ffn.nfl` (N=1 baseline) and `transformer_block.nfl` (N=3 — combined runtime evidence for LH-4 + A2 transformer-block showcase).
4. **Four new FFI integration tests** (2 per profile) plus three new ABI-invariant unit tests in x86_64 (mirroring `emit_linear_n{2,3,4}_does_not_clobber_output_reg` precedent).

### Out of scope (explicit)

- New `StdOp::Ffn` variant — composition is the design (per `PROJECT_SPEC.md` §"Strategic Roadmap" Axis 2 "compositional op, no new codegen pattern").
- `arm64` codegen changes — FFN uses existing `linear`/`relu` emitters; arm64 `emit_layernorm` AAPCS64-clean (scratch in `x6`/`x9–x17` + `s0–s7`, no overlap with `x0–x4` input slots at supported N).
- Bench inclusion — FFN is chained matmul+relu, no orthogonal signal beyond existing `classifier` fixture.
- New docs structure — existing files (`DEVLOG.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `docs/profile_guide/x86_64.md`) are updated in place.
- Trigger-driven cleanup items (OQ-7/8/9, M5c OQ-4) — no triggers fire in M15, remain dormant.
- Negative fixtures for FFN — composition is covered by existing per-op negatives (`linear` shape mismatch, `relu` rank check, etc.); no new error surfaces.

---

## 2. LH-4 cleanup — `profiles/x86_64/src/ops/layernorm.rs`

### Background

The M14 entry in `PROJECT_SPEC.md` §"Known Latent Hazards":

> **LH-4** | `profiles/x86_64/src/ops/layernorm.rs` | N=3 (output_reg = `%r8`) or N=4 (output_reg = `%r9`) | `emit_layernorm` uses `%r8` (src row ptr) and `%r9` (dst row ptr) as per-row scratch — clobbers output_reg / input(N-1) at N≥3 | M14

Per the §LH rule and memory entry `feedback_triggered_cleanup.md`, the hazard must be closed in M15 (one-milestone budget; opener was M14).

### Register relocation

| Old register | New register | Justification |
|---|---|---|
| `%r8` (per-row src ptr) | **`%r15`** | Callee-saved; **NOT** in `compute_callee_saved` set → requires op-local `pushq %r15` / `popq %r15` (LH-2/3 precedent in `emit_linear`). Verified free in `layernorm.rs` (grep clean). |
| `%r9` (per-row dst ptr) | **`%rbp`** | Callee-saved; **always** pushed by function-level prologue (`profiles/x86_64/src/asm.rs:33` "always: push %rbp"; line 66–67) → body free without op-local push. Precedent: LH-1 in `emit_linear` (`%rcx` → `%rbp`); M13 `emit_matmul` (`%r9` → `%rbp` j-counter). NeuralForge prologue does **not** establish a frame pointer (no `movq %rsp, %rbp` after the push) — `%rbp` is purely a saved callee-saved scratch slot. |

### Push order discipline (load-bearing)

`materialise_ptr_with_rsp_bias` is called **after** all pushes complete (its `rsp_bias_bytes` parameter equals total pushed bytes). Push order must be exact for two reasons: (a) keep existing M14 push order (`%rbx`, `%r14`, optional `%r12`, `%r13`) intact as a relative subsequence — minimal diff, existing doc-comments stay accurate; (b) make `%r15` visually prominent as the M15 addition for future audit.

**Verbatim push order — no-affine path (3 pushes):**

```
pushq   %r15      # M15 LH-4 — first
pushq   %rbx      # existing M14
pushq   %r14      # existing M14
```

Pop order (strict LIFO reverse):

```
popq    %r14
popq    %rbx
popq    %r15
```

**Verbatim push order — affine path (5 pushes):**

```
pushq   %r15      # M15 LH-4 — first
pushq   %r12      # existing M14
pushq   %r13      # existing M14
pushq   %rbx      # existing M14
pushq   %r14      # existing M14
```

Pop order:

```
popq    %r14
popq    %rbx
popq    %r13
popq    %r12
popq    %r15
```

**Implementer rule:** if `%r15` ends up anywhere except first push / last pop, the diff against current `layernorm.rs` is wrong — revert and redo.

### Push strategy: unified, not conditional

Body unconditionally references `%r15` (relocated `%r8`) — there is no `N`-conditional branch in the emit path. Therefore `%r15` push/pop is unconditional (every `emit_layernorm` invocation pushes `%r15`).

A "conditional" alternative would emit two body code paths (one with `%r8` for N≤2, one with `%r15` for N≥3) — saves 1 push at N=1/N=2 (sub-percent perf, amortised across `b × d` inner iterations) at the cost of **two body code paths** in `emit_layernorm`. Rejected per design principle "explicit over implicit" + simpler-is-better.

### Push counts after fix

| Affine | Pre-M15 (current) | Post-M15 (after LH-4) |
|---|---|---|
| no-affine | 2 (`%rbx`, `%r14`) | **3** (`%r15`, `%rbx`, `%r14`) |
| affine | 4 (`%r12`, `%r13`, `%rbx`, `%r14`) | **5** (`%r15`, `%r12`, `%r13`, `%rbx`, `%r14`) |

### Stack alignment posture (preserved)

3 and 5 are odd — total `%rsp` displacement from function-frame stable point is `3·8 = 24` (no-affine) or `5·8 = 40` (affine), both ≡ 8 mod 16. Combined with `pushq %rbp` in prologue (+8, also odd), inside-body `%rsp` is **not** 16-byte aligned. This is **OK** because `emit_layernorm` is **leaf** (native `sqrtss`, no `call` instruction in body).

The M14 doc-comment `layernorm.rs:56-62` already documents this invariant; only the numerical counts need updating.

If a future fixture combines LayerNorm with Softmax (the only op emitting `call expf@PLT`) inside the same function, a one-time `subq $8, %rsp` adjustment outside the inner body restores alignment. This stays a future-foot-gun note in the doc-comment, not addressed in M15.

### Push-bytes constants

```rust
const OP_LOCAL_PUSH_BYTES_NO_AFFINE: usize = 3 * 8;   // was 2 * 8
const OP_LOCAL_PUSH_BYTES_AFFINE:    usize = 5 * 8;   // was 4 * 8
```

The existing `materialise_ptr_with_rsp_bias` `debug_assert` (`layernorm.rs:101-110`) continues to enforce `rsp_bias_bytes ∈ {NO_AFFINE, AFFINE}` — guards against future drift between push count and constant.

### Doc-comment updates (in same edit)

- "Register plan" table (lines 28–47): `%r8` row → `%r15` (callee-saved + op-local push/pop); `%r9` row → `%rbp` (callee-saved + function-level prologue handles).
- "Stack alignment invariant" block (lines 56–62): updated push counts (3/5 instead of 2/4); same leaf-only justification preserved.

### Unit tests (LH-4 specific) — `profiles/x86_64/src/tests.rs`

Three new tests, mirroring `emit_linear_n{2,3,4}_does_not_clobber_output_reg`:

- `emit_layernorm_n2_does_not_clobber_output_reg` — parametric guard (passes pre- and post-fix; output_reg = `%rcx` at N=2, never touched).
- `emit_layernorm_n3_does_not_clobber_output_reg` — **primary LH-4 unit test.** output_reg = `%r8`. Asm-shape check: emitted body must not contain `%r8` as a destination operand in `movq`/`leaq` instructions outside the prologue/epilogue.
- `emit_layernorm_n4_does_not_clobber_output_reg` — output_reg = `%r9`. Same asm-shape check excluding `%r9`.

Additional asm-shape assertion in each test: `pushq` substring count matches expected (3 for no-affine variant, 5 for affine — runs both `has_affine` cases per N where applicable).

---

## 3. Fixtures + reference impls + FFI tests

### 3.1 `tests/fixtures/ffn.nfl` — N=1 baseline FFN demonstrator

```nfl
# FFN as compositional NFL pattern — A2 third brick (M15).
#
# Pure composition: linear → relu → linear, both with bias.
# N=1 baseline — no multi-input ABI involvement, no LH surface.
# Demonstrates that FFN requires NO new StdOp / IR / codegen pattern;
# both `linear` and `relu` emitters already exist on both profiles
# (arm64 since M3, x86_64 since M9).

model Ffn [batch=2, dim=4, hidden=8]:
    x: Tensor[batch, dim]

    x -> linear[hidden, bias=true] -> relu -> linear[dim, bias=true]
```

**ABI mapping** (both profiles, N=1):

| Slot | x86_64 | arm64 |
|---|---|---|
| `x` (input 0) | `%rdi` | `x0` |
| params | `%rsi` | `x1` |
| output | `%rdx` | `x2` |

Standard N=1 ABI, no LH involvement.

**Param blob layout** (compute_offsets traversal order):

| Param | Source | Floats |
|---|---|---|
| LinearWeight #1 | first `linear[hidden=8, bias=true]` | dim × hidden = 32 |
| LinearBias #1 | first linear bias | hidden = 8 |
| LinearWeight #2 | second `linear[dim=4, bias=true]` | hidden × dim = 32 |
| LinearBias #2 | second linear bias | dim = 4 |
| **Total** | | **76** |

**Shape evolution:** `x: [2,4]` → `linear[8]` → `[2,8]` → `relu` → `[2,8]` → `linear[4]` → `[2,4]` (out).

### 3.2 `tests/fixtures/transformer_block.nfl` — N=3, combined LH-4 runtime evidence + A2 showcase

```nfl
# Pre-LN transformer block-style fixture, N=3 inputs.
#
# Combined purpose:
#   1. Runtime FFI evidence for LH-4 closure: LayerNorm at N=3 →
#      output_reg = %r8 on x86_64. Pre-T0, layernorm body would clobber
#      %r8 (per-row src ptr), corrupting output pointer → segfault or
#      wrong-output bit-mismatch vs Rust reference. Post-T0, %r8 untouched
#      (relocated to %r15), bit-exact match.
#   2. A2 third brick demonstration — full transformer-block composition
#      (LayerNorm + FFN + dual residual) end-to-end on both profiles.
#
# ABI slot layout (declaration order, SysV AMD64):
#   N=3: x (%rdi), skip1 (%rsi), skip2 (%rdx),
#        params (%rcx), out (%r8).  ← LH-4 condition: out = %r8

model TransformerBlock [batch=2, dim=4, hidden=8]:
    x: Tensor[batch, dim]
    skip1: Tensor[batch, dim]
    skip2: Tensor[batch, dim]

    x -> layernorm[affine=true]
      -> linear[hidden, bias=true]
      -> relu
      -> linear[dim, bias=true]
      -> add[skip1]
      -> add[skip2]
```

**ABI mapping** (SysV AMD64, N=3):

| Slot | Register | Notes |
|---|---|---|
| `x` (input 0) | `%rdi` | |
| `skip1` (input 1) | `%rsi` | |
| `skip2` (input 2) | `%rdx` | |
| params | `%rcx` | |
| **output** | **`%r8`** | **LH-4 trigger condition** |

**Param blob layout:**

| Param | Source | Floats |
|---|---|---|
| LayerNormScale (γ) | `layernorm[affine=true]` | dim = 4 |
| LayerNormBias (β) | `layernorm[affine=true]` | dim = 4 |
| LinearWeight #1 | first linear | dim × hidden = 32 |
| LinearBias #1 | first linear bias | hidden = 8 |
| LinearWeight #2 | second linear | hidden × dim = 32 |
| LinearBias #2 | second linear bias | dim = 4 |
| **Total** | | **84** |

**Shape evolution:** `[2,4]` → `layernorm` → `[2,4]` → `linear[8]` → `[2,8]` → `relu` → `[2,8]` → `linear[4]` → `[2,4]` → `add[skip1]` → `[2,4]` → `add[skip2]` → `[2,4]` (out).

The affine path triggers the **full 5-push block** (`%r15`/`%r12`/`%r13`/`%rbx`/`%r14`) → end-to-end check of push-bytes accounting + `materialise_ptr_with_rsp_bias` on realistic stack offsets.

### 3.3 Helper promotion (T1 prerequisite)

Currently in the workspace:
- `profiles/{arm64,x86_64}/tests/common/mod.rs` exports `cc_available`, `compile_to_dylib`/`compile_to_so`, **`layernorm_ref`** (M14 addition).
- `profiles/{arm64,x86_64}/tests/integration.rs` defines **file-local** `reference_matmul` (line 11/15) and `reference_bias_add` (line 25/29).

For M15, `ffn_ref` and `transformer_block_ref` need to compose `reference_matmul` + `reference_bias_add` + `layernorm_ref`. Therefore `reference_matmul` and `reference_bias_add` must be **promoted** from `integration.rs` file-local to `common/mod.rs` `pub fn` — per profile, separate copies (CLAUDE.md design principle 3 — profile isolation; no cross-profile sharing).

**Promotion caveat (mandatory for implementer):** before moving the function bodies verbatim, **inspect the existing signatures for test-specific quirks** (hardcoded shape assumptions, baked-in stride invariants, undocumented argument-order assumptions). If any quirk surfaces, generalise during the move — do **not** blindly copy. The promoted functions must work correctly for the new M15 shapes (`(B=2, K=4, N=8)` for FFN's first matmul; `(B=2, K=8, N=4)` for the second), not just the existing call-site shapes.

### 3.4 Per-profile reference impls

Per CLAUDE.md design principle 3 (profile isolation) and M14 `layernorm_ref` precedent, each profile gets its own copy.

**`ffn_ref`** in both `profiles/{arm64,x86_64}/tests/common/mod.rs`:

```rust
pub fn ffn_ref(
    input: &[f32],
    w1: &[f32], b1: &[f32],
    w2: &[f32], b2: &[f32],
    batch: usize, dim: usize, hidden: usize,
) -> Vec<f32> {
    // 1. linear #1: matmul + bias_add
    let mm1 = reference_matmul(input, w1, batch, dim, hidden);
    let mm1_b = reference_bias_add(&mm1, b1, hidden);
    // 2. relu (inline)
    let relu_out: Vec<f32> = mm1_b.iter().map(|&x| x.max(0.0)).collect();
    // 3. linear #2: matmul + bias_add
    let mm2 = reference_matmul(&relu_out, w2, batch, hidden, dim);
    reference_bias_add(&mm2, b2, dim)
}
```

**`transformer_block_ref`** in both `profiles/{arm64,x86_64}/tests/common/mod.rs`:

```rust
pub fn transformer_block_ref(
    input: &[f32], skip1: &[f32], skip2: &[f32],
    gamma: &[f32], beta: &[f32],
    w1: &[f32], b1: &[f32],
    w2: &[f32], b2: &[f32],
    batch: usize, dim: usize, hidden: usize,
) -> Vec<f32> {
    // 1. layernorm (M14-verified bit-exact ref)
    let ln = layernorm_ref(input, &[batch, dim], Some(gamma), Some(beta));
    // 2. FFN composition (reuse ffn_ref)
    let ffn_out = ffn_ref(&ln, w1, b1, w2, b2, batch, dim, hidden);
    // 3. add[skip1] (element-wise)
    let r1: Vec<f32> = ffn_out.iter().zip(skip1.iter()).map(|(&a, &b)| a + b).collect();
    // 4. add[skip2] (element-wise)
    r1.iter().zip(skip2.iter()).map(|(&a, &b)| a + b).collect()
}
```

**Helper-reuse rule (CRITICAL — load-bearing for both refs):** `ffn_ref` MUST call `reference_matmul` + `reference_bias_add` (after promotion). `transformer_block_ref` MUST call `layernorm_ref` (M14, in `common/mod.rs`) + `ffn_ref` (or its promoted helper components) + inline element-wise add. **Do NOT reimplement matmul reduction loop, bias add, or layernorm normalization.** Divergent reduction order produces 1+ ULP mismatches that fail bit-exact comparison and are deeply painful to debug. Existing helpers are M14-verified bit-exact against emitters — reuse them as-is.

### 3.5 FFI integration tests

| Test | Profile | OS coverage | Purpose |
|---|---|---|---|
| `ffn_ffi` | arm64 | macOS (CI + dev) | N=1 FFN bit-exact validation |
| `transformer_block_ffi` | arm64 | macOS | N=3 LayerNorm + FFN + dual residual; **implicit ABI audit for arm64 at N=3** |
| `ffn_ffi` | x86_64 | Linux CI; `#[cfg]` skipped on macOS | N=1 FFN bit-exact validation |
| **`transformer_block_ffi`** | **x86_64** | **Linux CI; `#[cfg]` skipped on macOS** | **LH-4 runtime evidence — empirical proof of T0 closure** |

Total: **4 new FFI integration tests.** Skip-on-macOS pattern follows M9 precedent (`#[cfg]` gating `target_os = "linux"` for x86_64-specific tests).

---

## 4. Task structure — 4 commits, single PR, strict order

Single-PR rationale (decided in brainstorm): M15 scope is materially smaller than M14 (no IR foundation, no per-profile codegen of a new `StdOp`, just register relocation in one file + 2 fixtures + helper promotion + tests). The 2-PR M14 split was driven by physically distinct themes (cleanup of `emit_linear` vs. IR foundation + 2-profile codegen of new `StdOp::LayerNorm`). M15's cleanup and feature form one coherent narrative ("register relocation → runtime proof"); separating them would be cargo-cult.

Commit order is **strict** for bisectability: each commit independently passes `cargo test --workspace`; reverting any single commit predictably impacts a known-bounded set of tests.

### T0 — `fix(m15): close LH-4 — relocate %r8/%r9 in x86_64 emit_layernorm`

**Edits in this commit:**

1. `profiles/x86_64/src/ops/layernorm.rs`:
   - Register relocation per §2 ("Register relocation"). Use the **verbatim push order** from §2 ("Push order discipline").
   - Update `OP_LOCAL_PUSH_BYTES_*` constants (3·8 / 5·8).
   - Update doc-comments: "Register plan" table, "Stack alignment invariant" block.
2. `profiles/x86_64/src/tests.rs`: 3 new ABI-invariant unit tests per §2 ("Unit tests").

**No fixture, no FFI test in T0.** Runtime evidence comes in T2.

**Acceptance:** `cargo build --workspace` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo fmt --all -- --check` + `cargo test --workspace` all green; 3 new unit tests pass.

### T1 — `feat(m15): A2 third brick — FFN as compositional NFL pattern`

**Edits in this commit:**

1. `tests/fixtures/ffn.nfl` — new file (§3.1 verbatim).
2. **Helper promotion (per profile, separate copies):**
   - `profiles/arm64/tests/common/mod.rs`: add `pub fn reference_matmul(...)` + `pub fn reference_bias_add(...)` (move from `integration.rs:11-30`, applying §3.3 caveat about test-specific quirks). Add `pub fn ffn_ref(...)` per §3.4.
   - `profiles/arm64/tests/integration.rs`: remove file-local `reference_matmul` / `reference_bias_add` definitions; update existing call sites to `common::reference_matmul` / `common::reference_bias_add` via `use common::{reference_matmul, reference_bias_add};` (or fully-qualified). Add `#[test] fn ffn_ffi()` that compiles `ffn.nfl`, FFI-calls, compares against `common::ffn_ref(...)`.
   - `profiles/x86_64/tests/common/mod.rs`: same promotion + `ffn_ref` (separate copy — NOT shared from arm64).
   - `profiles/x86_64/tests/integration.rs`: same removal/refactor + `#[test] fn ffn_ffi()` with `#[cfg]` skip-on-macOS pattern from M9.

**CRITICAL implementer requirement:** `ffn_ref` MUST compose existing promoted `reference_matmul` + `reference_bias_add` (do NOT rewrite matmul logic). See §3.4 helper-reuse rule.

**Acceptance:** 2 new FFI tests pass (arm64 + x86_64); existing tests (which referenced file-local `reference_matmul`/`reference_bias_add`) continue to pass after the move.

### T2 — `feat(m15): transformer_block fixture — LH-4 runtime evidence + A2 showcase`

**Edits in this commit:**

1. `tests/fixtures/transformer_block.nfl` — new file (§3.2 verbatim, including doc-comment about LH-4 trigger condition).
2. `profiles/arm64/tests/common/mod.rs`: add `pub fn transformer_block_ref(...)` per §3.4.
3. `profiles/arm64/tests/integration.rs`: add `#[test] fn transformer_block_ffi()` — bit-exact match.
4. `profiles/x86_64/tests/common/mod.rs`: add `transformer_block_ref` (separate copy).
5. `profiles/x86_64/tests/integration.rs`: add `#[test] fn transformer_block_ffi()` (Linux CI; `#[cfg]` skip-on-macOS).

**CRITICAL implementer requirement:** `transformer_block_ref` MUST compose `layernorm_ref` (M14) + `ffn_ref` (T1) + inline element-wise add. See §3.4 helper-reuse rule.

**Bisectability claim** (document in T2 commit message):

> Reverting **only** T0 (the LH-4 cleanup commit) on any tip-state after T2 lands → `transformer_block_ffi` on x86_64 fails on Linux CI with silent corruption (output_reg = `%r8` clobbered by per-row src ptr in `emit_layernorm`). This proves T0 is the load-bearing fix for LH-4 and that T2 is the empirical runtime evidence. Phrased compactly: T0 without T2 = closure by inspection only (asm-shape unit tests); T2 without T0 = runtime crash; T0+T2 together = LH-4 closed with runtime evidence.

**Acceptance:** 2 new FFI tests pass; specifically Linux CI confirms x86_64 `transformer_block_ffi` bit-exact match.

### T3 — `docs(m15): documentation closure`

**Edits in this commit:**

1. `DEVLOG.md` — M15 entry following M14 template:
   - **What was done:** T0–T3 bullet list with commit SHAs.
   - **Decisions made:** register relocation (`%r8`→`%r15`, `%r9`→`%rbp`) with §2 reasoning; unified push strategy; helper promotion (`reference_matmul`/`reference_bias_add` from `integration.rs` to `common/mod.rs`); FFN as compositional pattern (no new `StdOp`).
   - **Problems encountered:** placeholder ("none") unless something surfaces during execution; otherwise document.
   - **Next step:** M16 candidate (e.g. A3 — profile-level viewer annotations; or Axis 3 — bare-metal `expf`).
   - **ABI audit record (mandatory paper trail):** explicit per-emitter audit findings. Required listing:
     - `emit_layernorm` (x86_64) at N=3, N=4 — LH-4 closed in T0, asm-shape verified, runtime evidence in T2.
     - `emit_linear` (x86_64) at N=3, N=4 — reviewed, clean (LH-1/2/3 closed M14 commit `916e9c7`).
     - `emit_matmul` (x86_64) at N=3, N=4 — reviewed, clean (M13 Task 1 `%r9`→`%rbp` j-counter relocation).
     - **`emit_relu` (x86_64) at N=3 — reviewed, clean.** Empirically validated by T2 `transformer_block_ffi` (relu invocation occurs between two linears in N=3 model).
     - **`emit_add` (x86_64) at N=3 — reviewed, clean.** Empirically validated by T2 (two `add` invocations at N=3 in `transformer_block.nfl`).
     - `emit_mulscalar` (x86_64) at N=3, N=4 — reviewed, clean (single scratch register, no ABI overlap).
     - `emit_softmax` (x86_64) at N=3, N=4 — reviewed, clean (M10 spill of `%rdi`/`%rsi`/`%rdx` around `call expf@PLT`).
     - `emit_dropout` (x86_64) — pass-through marker, no scratch.
2. `PROJECT_SPEC.md`:
   - §"Milestones to date": new row "15 | A2 third brick — FFN compositional + LH-4 cleanup (complete) | …" with key decisions.
   - §"Strategic Roadmap" Axis 2: update describing A2 third brick (FFN) closed; A2 axis fully complete; open follow-ups updated.
   - §"Known Latent Hazards": **remove LH-4 row.** If table becomes empty, leave header + "(empty — all hazards closed at end of M15)" note.
3. `CLAUDE.md` "Current Status": update milestone marker (M14 → M15), test count, brief decisions summary.
4. `docs/profile_guide/x86_64.md`: section about `emit_layernorm` register allocation — update with post-LH-4 table (`%r15`/`%rbp` instead of `%r8`/`%r9`, push counts 3/5).

**Acceptance:** all docs synced; cargo gates remain green (docs don't affect build, but routine fmt/clippy/test re-verification).

---

## 5. ABI audit obligation

Per CLAUDE.md workflow §"ABI audit (x86_64)":

> When adding a new operation emitter OR when a milestone expands input arity, run an ABI audit across all x86_64 emitters in `profiles/x86_64/src/ops/`. For each emitter, verify that no ABI-argument register (from `AbiContext`) appears as a long-lived counter or scratch. Document any violations found as entries in `PROJECT_SPEC.md` §"Known Latent Hazards" before closing the milestone.

**M15 trigger:** `transformer_block.nfl` is the first fixture combining LayerNorm + FFN + Add at N=3. While no individual op is "new", the **arity expansion under this op set** triggers the audit obligation.

### Audit process (executed during T0/T2 prep, not a separate commit)

For each x86_64 emitter file in `profiles/x86_64/src/ops/`:

1. Read the file.
2. Identify all scratch register uses in body (post-prologue, pre-epilogue).
3. Cross-reference with `AbiContext::input_reg(0..N-1)` / `params_reg()` / `output_reg()` for N=3 and N=4 (the maximum N expanded by M12 + M13).
4. Flag any overlap as a candidate hazard.

### Expected findings (encoded in §4 T3 DEVLOG record)

| Emitter | Status | Note |
|---|---|---|
| `emit_layernorm` | LH-4 closed in T0 | Asm-shape verified (T0 unit tests); runtime evidence in T2 |
| `emit_linear` | clean | LH-1/2/3 closed M14 commit `916e9c7` |
| `emit_matmul` | clean | M13 Task 1 (`%r9`→`%rbp`) |
| `emit_relu` | **clean (M15 audit)** | Empirically validated by T2 — relu in `transformer_block.nfl` at N=3 |
| `emit_add` | **clean (M15 audit)** | Empirically validated by T2 — two `add` invocations at N=3 |
| `emit_mulscalar` | clean | Single scratch register, no ABI overlap |
| `emit_softmax` | clean | M10 spill around `call expf@PLT` |
| `emit_dropout` | clean | Pass-through marker |

**Paper trail rule:** every emitter must appear by name in the DEVLOG ABI audit record, even if "clean" — implicit clean-by-omission is not acceptable. The explicit listing protects against "we forgot to check X" failure mode.

---

## 6. Done definition (acceptance criteria for merge)

1. **All 4 commits land in PR** in order T0 → T1 → T2 → T3. Each commit independently passes `cargo test --workspace`.
2. **T2 commit message contains the bisectability claim** from §4.
3. **`cargo fmt --all -- --check`** clean for all 4 commits.
4. **`cargo clippy --workspace --all-targets -- -D warnings`** clean for all 4.
5. **Linux x86_64 CI runs all FFI tests** (including 2 new x86_64 FFI tests) — green.
6. **macOS arm64 dev/CI runs all arm64 FFI tests** (including 2 new arm64 FFI tests) — green.
7. **Docs updated** (T3): `DEVLOG.md` / `PROJECT_SPEC.md` / `CLAUDE.md` / `docs/profile_guide/x86_64.md` all synced.
8. **`PROJECT_SPEC.md` §"Known Latent Hazards" reflects LH-4 removal** (row deleted; table-empty note if applicable).
9. **Two-stage subagent review pattern** per memory M14 cadence + PR#33 reference shape: implementation → first review subagent → address feedback → second review subagent → merge.

---

## 7. References

### Memory entries consulted

- `feedback_triggered_cleanup.md` — once a PROJECT_SPEC trigger fires, close it before the next feature milestone. **Applied:** LH-4 (opened M14) is mandatory M15 scope, not optional.
- `feedback_review_deferral_verify_target.md` — verify deferral target before accepting reviewer carryover. **Applied:** verified that M15 (FFN feature) does not naturally touch `layernorm.rs`, so LH-4 cleanup must be explicit M15 scope, not a side-effect.
- `feedback_runtime_evidence_for_codegen.md` — asm-shape unit tests miss op-local push interactions; trust runtime FFI evidence on Linux x86_64 CI. **Applied:** T2 `transformer_block_ffi` on x86_64 is the runtime evidence requirement; T0 unit tests alone are insufficient.

### M14 precedent commits

- `b266b0e` — M14 merge.
- `916e9c7` — M14 LH-1/2/3 cleanup in x86_64 `emit_linear` (precedent for register relocation pattern).
- `7298f88` / `ec0659f` — M14 arm64 / x86_64 `emit_layernorm` (the file LH-4 lives in).
- M14 PR shapes: PR#31 (cleanup opener) + PR#32 (feature) — the 2-PR pattern that M15 deliberately does NOT follow.

### Spec sections relied upon

- `PROJECT_SPEC.md` §"Strategic Roadmap" Axis 2 — A2 transformer block decomposition (residual + LayerNorm + FFN).
- `PROJECT_SPEC.md` §"Known Latent Hazards" — LH-4 entry to close in M15.
- `CLAUDE.md` workflow §"Before any commit" + §"ABI audit (x86_64)" — process obligations.
- `CLAUDE.md` design principle 3 (profile isolation) — justifies per-profile reference impl duplication.

---

## 8. Open questions deferred

None. All design questions raised during brainstorming were resolved:

- FFN representation: pure NFL composition (no `StdOp::Ffn`).
- PR structure: single PR (not M14-style split), 4 commits T0→T1→T2→T3.
- LH-4 validation strategy: combined `transformer_block.nfl` (Option B from brainstorm) — one fixture serves as both LH-4 runtime evidence and A2 showcase.
- Register choice: `%r15` (op-local push/pop) and `%rbp` (function-level prologue handles).
- Push strategy: unified (always push `%r15`), not conditional.
- Push order: `%r15` first push / last pop, existing M14 pushes preserved as relative subsequence.
- Helper organisation: promote `reference_matmul` / `reference_bias_add` to `common/mod.rs` per profile; reuse rather than reimplement.
- ABI audit format: process obligation during T0/T2 prep, paper trail as explicit per-emitter listing in T3 DEVLOG.

---

*End of design spec. Implementation plan to follow via writing-plans skill upon approval.*
