# Bare-Metal `expf` (M17) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the libm `expf` call in softmax codegen with an inlined degree-7 Taylor polynomial on both profiles, removing NeuralForge's last runtime dependency.

**Architecture:** A shared per-profile `emit_exp_inline()` primitive (new `ops/exp.rs`) replaces the single `bl _expf` / `call expf@PLT` instruction at both softmax sites (standalone `emit_softmax`, fused `SoftmaxRow` tail in `emit_linear`). Cody-Waite range reduction → degree-7 Horner → `2^z` bit-trick reconstruction with a branchless underflow clamp. Polynomial constants live in a file-local `.rodata` pool emitted once per assembly file from `walk_uir`. The minimal-swap discipline (spec §1) leaves the softmax loop's register layout untouched; only the exp-pass body changes. Validation is two-layer: bit-exact asm vs a Rust port (`exp_ref`), and the port vs libm within ≤ 1 ulp.

**Tech Stack:** Rust (workspace crates `compiler`, `profiles-arm64`, `profiles-x86_64`, `profile-api`); AArch64 Mach-O + x86_64 Linux ELF assembly text; `cc` for FFI integration tests.

**Spec:** `docs/superpowers/specs/2026-05-29-bare-metal-expf-m17-design.md`

**Two spec refinements discovered during planning (reconcile in Task 10 doc-closure):**
1. **§3.3** — the `.rodata` pool is *pre-existing on x86_64* (layernorm, `profiles/x86_64/src/ops/layernorm.rs:309`) but *new on arm64* (arm64 layernorm uses inline `movz/movk/fmov`, `profiles/arm64/src/ops/layernorm.rs:44`). M17 introduces a Mach-O `.section __TEXT,__const` pool for arm64.
2. **§5.1 / §6.3** — current softmax FFI tests are tolerance-based (`abs() < 1e-4`, row-sum ≈ 1), not bit-exact-vs-libm. Layer 1 is therefore a **new** bit-exact FFI test, cleanest on a new isolated `tests/fixtures/softmax_only.nfl` fixture rather than by modifying existing tolerance tests.

**Shared numeric constants (used identically by emitters AND `exp_ref` ports — drift caught by the Task 5/9 bit-exact tests):**

```
LOG2E   = 1.4426950408889634
LN2_HI  = 0.693359375
LN2_LO  = -0.00021219444005469057
C0 = 1.0          C1 = 1.0          C2 = 0.5            C3 = 1.0/6.0
C4 = 1.0/24.0     C5 = 1.0/120.0    C6 = 1.0/720.0      C7 = 1.0/5040.0
```

Horner evaluates `exp(r)` high→low: `p = C7; p = p*r + C6; … ; p = p*r + C0`.
Reconstruction: `pow = (z+127 <= 0) ? 0.0 : from_bits((z+127)<<23)`; `result = p * pow`.

**Confirmed codebase conventions (verified against the tree):**
- FFI C signature is `unsafe extern "C" fn(*const f32 /*input*/, *const f32 /*params*/, *mut f32 /*output*/)`.
- arm64 dylib builder: `common::compile_to_dylib(asm, name)`; x86_64 so builder: `common::compile_to_so(asm, name)`.
- `deterministic_input(total)` is **file-local** in each `integration.rs` (NOT in `common`).
- Emitters are called via the re-export short path: `crate::ops::emit_<op>(…)`.
- Fixtures are read from `"../../tests/fixtures/<name>.nfl"` (crate cwd is `profiles/<p>/`).

---

## File Structure

**Created:**
- `profiles/arm64/src/ops/exp.rs` — `exp_pool_arm64()` (Task 3) + `emit_exp_inline()` (Task 4). Codegen primitive, not a `StdOp`.
- `profiles/x86_64/src/ops/exp.rs` — `exp_pool_x86_64()` (Task 7) + `emit_exp_inline()` (Task 8).
- `tests/fixtures/softmax_only.nfl` — isolated `input → softmax` fixture for the layer-1 bit-exact test.

**Modified:**
- `compiler/src/ir/types.rs`, `compiler/src/ir/tests.rs` — rename `calls_extern_math` → `has_softmax` + `--uir-verbose` label.
- `profiles/{arm64,x86_64}/src/buffer.rs`, `codegen.rs` — rename call sites + doc-comments + emit pool from `walk_uir`.
- `profile-api/src/lib.rs` — `FnAnnotations.leaf` doc-comment.
- `profiles/{arm64,x86_64}/src/ops/mod.rs`, `softmax.rs`, `linear.rs`, `exp.rs` — wire + swap call for inline.
- `profiles/{arm64,x86_64}/src/tests.rs` — flip asm-shape unit tests.
- `profiles/{arm64,x86_64}/tests/common/mod.rs` — add `exp_ref` + `softmax_ref`.
- `profiles/{arm64,x86_64}/tests/integration.rs` — layer-2 sweep + layer-1 bit-exact FFI + clamp FFI.
- `docs/profile_guide/{arm64,x86_64}.md`, `docs/language_reference/uir.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `DEVLOG.md`.

---

## Task 1: Rename `calls_extern_math` → `has_softmax`

Pure rename — no behavior change. The predicate already computes "model has standalone `StdOp::Softmax` or fused `PostOp::SoftmaxRow`"; only the name (and CLI label) lie after inlining.

**Files:** `compiler/src/ir/types.rs`, `compiler/src/ir/tests.rs`, `profiles/{arm64,x86_64}/src/buffer.rs`, `profiles/{arm64,x86_64}/src/codegen.rs`, `profile-api/src/lib.rs`, `docs/language_reference/uir.md`

- [ ] **Step 1: Write the failing test for the new verbose label**

In `compiler/src/ir/tests.rs`, rename `calls_extern_math_true_for_standalone_softmax`, `calls_extern_math_false_for_linear_only`, `calls_extern_math_true_for_fused_softmax_row` to `has_softmax_*` and update the method calls inside each. Then add:

```rust
#[test]
fn verbose_uir_uses_has_softmax_label() {
    let src = "model S [batch=1, k=3]:\n    x: Tensor[batch, k]\n    x -> softmax\n";
    let nfl = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("build");
    let rendered = format!("{}", compiler::ir::types::VerboseUir(&uir));
    assert!(rendered.contains("has-softmax: yes"), "got:\n{rendered}");
    assert!(!rendered.contains("calls-extern-math"), "stale label:\n{rendered}");
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p compiler verbose_uir_uses_has_softmax_label`
Expected: FAIL — method still `calls_extern_math`; rendered text still `calls-extern-math`.

- [ ] **Step 3: Rename the method + label in `types.rs`**

- `impl UirModel { pub fn calls_extern_math` → `pub fn has_softmax` (body identical).
- `impl Uir { pub fn calls_extern_math` → `pub fn has_softmax`; body `self.models.iter().any(UirModel::has_softmax)`.
- Update the `// M8: calls_extern_math predicate` section comment to describe `has_softmax`: "true iff the model contains softmax (standalone `StdOp::Softmax` or fused `PostOp::SoftmaxRow`) — the op whose codegen needs the callee-saved register regime."
- In `VerboseUir` and `VerboseModel` `Display` impls: `"  calls-extern-math: {}"` → `"  has-softmax: {}"`, and `.calls_extern_math()` → `.has_softmax()`.

- [ ] **Step 4: Update every other site (grep-driven)**

Run: `grep -rn "calls_extern_math\|calls-extern-math" --include="*.rs" --include="*.md" . | grep -v /target/`

Rename each. Expected sites:
- `profiles/arm64/src/buffer.rs` — `compute_is_leaf` (`!model.has_softmax()`), `compute_callee_saved` (`let has_softmax = model.has_softmax();`, fields `d8_d9: has_softmax, x19_x23: has_softmax`), RegSet/fn doc-comments reworded "iff any node calls `bl _expf`" → "iff the model has softmax (its loop holds state in these registers)".
- `profiles/x86_64/src/buffer.rs` — `assign_buffers` (`if model.has_softmax()`), `compute_callee_saved` (`model.has_softmax() || has_matmul(model)`), doc-comments reworded off "calls `expf@PLT`".
- `profiles/x86_64/src/codegen.rs` — `let leaf_bool = !model.has_softmax();` + comment.
- `profile-api/src/lib.rs` — `FnAnnotations.leaf` doc-comment "== `!UirModel::calls_extern_math()`" → "== `!UirModel::has_softmax()`".
- `docs/language_reference/uir.md` — "Viewing UIR" `calls-extern-math` example → `has-softmax`.

Reword `compute_is_leaf` doc (arm64): "Conservative: softmax models are non-leaf because their loop uses callee-saved registers and a frame, even though M17 inlined the exp. Precise leaf reclassification is M18." (Value stays `!has_softmax()`.)

- [ ] **Step 5: Run full suite + confirm no stale references**

Run: `cargo test --workspace 2>&1 | tail -20` → PASS.
Run: `grep -rn "calls_extern_math\|calls-extern-math" --include="*.rs" --include="*.md" . | grep -v /target/` → no output.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add -A
git commit -m "refactor(m17): rename calls_extern_math -> has_softmax (honest post-inline name)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: arm64 `exp_ref` + layer-2 accuracy sweep

Establish the reference algorithm and prove ≤ 1 ulp vs libm *before* any asm. "Widen LN2 split" tuning happens here if needed.

**Files:** `profiles/arm64/tests/common/mod.rs`, `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Write the failing sweep test**

In `profiles/arm64/tests/integration.rs`:

```rust
/// Layer 2 (spec §5.2): the Rust port must be within 1 ulp of libm over the
/// reachable softmax domain x ∈ [−80, 0]. Pure Rust; no asm, no FFI.
#[test]
fn exp_ref_within_one_ulp_of_libm() {
    let ulp_diff = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
    let mut x = -80.0_f32;
    while x <= 0.0 {
        let (got, want) = (common::exp_ref(x), x.exp());
        assert!(ulp_diff(got, want) <= 1,
            "x={x}: exp_ref={got} ({:#x}) libm={want} ({:#x}), {} ulp",
            got.to_bits(), want.to_bits(), ulp_diff(got, want));
        x += 0.0009765625; // 2^-10, exact step
    }
    for &x in &[0.0_f32, -std::f32::consts::LN_2, -1.0, -10.0, -50.0] {
        assert!(ulp_diff(common::exp_ref(x), x.exp()) <= 1, "x={x}");
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p profiles-arm64 --test integration exp_ref_within_one_ulp_of_libm`
Expected: FAIL — `common::exp_ref` does not exist.

- [ ] **Step 3: Implement `exp_ref` (arm64 — fused `mul_add`)**

In `profiles/arm64/tests/common/mod.rs`:

```rust
/// Reference f32 exp for x ≤ 0 — bit-exact match for the arm64 inline emitter.
/// Cody-Waite reduction + degree-7 Taylor (Horner) + 2^z.
///
/// CRITICAL: arm64 fuses multiply-accumulate (fmadd/fmsub), so every step uses
/// `f32::mul_add` (single rounding) to match the asm bit-for-bit. Do NOT rewrite
/// as separate `*`/`+` — that is the x86_64 variant. (Mirror of the per-profile
/// reference_matmul split, M15.)
pub fn exp_ref(x: f32) -> f32 {
    const LOG2E: f32 = 1.4426950408889634;
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -0.00021219444005469057;
    const C: [f32; 8] =
        [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0];
    let z = (x * LOG2E).round_ties_even() as i32; // fcvtns: nearest, ties-even
    let zf = z as f32;
    let r = (-zf).mul_add(LN2_HI, x); // x - zf*LN2_HI (fmsub, single rounding)
    let r = (-zf).mul_add(LN2_LO, r);
    let mut p = C[7];
    for k in (0..7).rev() {
        p = p.mul_add(r, C[k]); // p*r + C[k]
    }
    let zp = z + 127;
    let pow = if zp <= 0 { 0.0_f32 } else { f32::from_bits((zp as u32) << 23) };
    p * pow
}
```

- [ ] **Step 4: Run the sweep**

Run: `cargo test -p profiles-arm64 --test integration exp_ref_within_one_ulp_of_libm`
Expected: PASS. If a point reports 2 ulp, widen the reduction: add `LN2_LO2 = 1.0e-9` and a third `let r = (-zf).mul_add(LN2_LO2, r);` (mirror in the emitter later). Re-run until ≤ 1 ulp. Do NOT change the polynomial degree.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add -A
git commit -m "test(m17): arm64 exp_ref port + layer-2 <=1ulp sweep vs libm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: arm64 `.rodata` constant pool

A Mach-O `__const` pool emitted once per file (new mechanism for arm64). File-local labels — no global symbols. **No emitter wiring of `bl`-removal yet** (Task 4); this task only lands the pool, so its commit stays green.

**Files:** Create `profiles/arm64/src/ops/exp.rs`; modify `profiles/arm64/src/ops/mod.rs`, `profiles/arm64/src/codegen.rs`; test `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write the failing unit test (pool presence only)**

In `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn softmax_model_emits_local_exp_pool() {
    let src = "model S [batch=2, k=3]:\n    x: Tensor[batch, k]\n    x -> softmax\n";
    let uir = compiler::ir::build(&compiler::parse(src).unwrap()).unwrap();
    let asm = crate::lower(&uir).unwrap().source;
    assert!(asm.contains(".section __TEXT,__const"), "no const pool:\n{asm}");
    assert!(asm.contains(".Lexp_log2e:"), "no log2e constant:\n{asm}");
    assert!(asm.contains(".Lexp_c7:"), "no c7 constant:\n{asm}");
    assert_eq!(asm.matches(".Lexp_log2e:").count(), 1, "pool must be unique per file");
}
```

(The "no `expf` symbol" assertion belongs to Task 4 — the call is still present here.)

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p profiles-arm64 softmax_model_emits_local_exp_pool`
Expected: FAIL — no const pool yet.

- [ ] **Step 3: Create `profiles/arm64/src/ops/exp.rs` (pool only)**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Inline bare-metal `exp` for the softmax domain (x ≤ 0). Replaces the
//! M3-era `bl _expf`. See docs/superpowers/specs/2026-05-29-bare-metal-expf-m17-design.md.

/// f32 constants — MUST stay identical to `exp_ref` in
/// `profiles/arm64/tests/common/mod.rs` (drift caught by the Task 5 bit-exact test).
const LOG2E: f32 = 1.4426950408889634;
const LN2_HI: f32 = 0.693359375;
const LN2_LO: f32 = -0.00021219444005469057;
const C: [f32; 8] =
    [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0];

/// File-local Mach-O `__const` pool. Emitted ONCE per assembly file from
/// `walk_uir` when `uir.has_softmax()`. `.L`-local labels: one definition,
/// referenced from every `emit_exp_inline` site; locals do not collide across
/// separately-linked objects.
pub fn exp_pool_arm64() -> String {
    let mut s = String::new();
    s.push_str(".section __TEXT,__const\n");
    s.push_str(".p2align 2\n");
    s.push_str(&format!(".Lexp_log2e: .long 0x{:08x}\n", LOG2E.to_bits()));
    s.push_str(&format!(".Lexp_ln2hi: .long 0x{:08x}\n", LN2_HI.to_bits()));
    s.push_str(&format!(".Lexp_ln2lo: .long 0x{:08x}\n", LN2_LO.to_bits()));
    for (k, c) in C.iter().enumerate() {
        s.push_str(&format!(".Lexp_c{}: .long 0x{:08x}\n", k, c.to_bits()));
    }
    s
}
```

- [ ] **Step 4: Wire the module + emit pool from `walk_uir`**

In `profiles/arm64/src/ops/mod.rs`: add `pub mod exp;` and `pub use exp::exp_pool_arm64;` (do **not** re-export `emit_exp_inline` yet — it doesn't exist until Task 4; an unused re-export would trip `-D warnings`).

In `profiles/arm64/src/codegen.rs` `walk_uir`, after the model loop and before `Ok(Asm { … })`:

```rust
    if uir.has_softmax() {
        source.push('\n');
        source.push_str(&crate::ops::exp_pool_arm64());
    }
```

- [ ] **Step 5: Run the full arm64 suite + clippy**

Run: `cargo test -p profiles-arm64 2>&1 | tail -15`
Expected: PASS — the new pool test passes; the pool is appended (unreferenced labels are harmless) and existing softmax tests still pass (asm still calls `bl _expf`).
Run: `cargo clippy -p profiles-arm64 --all-targets -- -D warnings` → exit 0 (consts + pool fn both used).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add -A
git commit -m "feat(m17): arm64 .rodata exp constant pool (file-local labels)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: arm64 `emit_exp_inline` + wire into both sites + flip asm-shape tests

**Files:** `profiles/arm64/src/ops/exp.rs`, `profiles/arm64/src/ops/mod.rs`, `profiles/arm64/src/ops/softmax.rs`, `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Flip the asm-shape unit tests + add the no-`expf` assertion**

In `profiles/arm64/src/tests.rs`, find every test asserting `s.contains("bl      _expf")` (≈ lines 113/587/632) and change each to:

```rust
    assert!(!s.contains("bl      _expf"), "expf must be inlined now:\n{s}");
    assert!(s.contains("fcvtns"), "missing round-to-int (range reduction):\n{s}");
    assert!(s.contains(".Lexp_c7"), "missing Horner constant load:\n{s}");
```

Add the global no-libm assertion to `softmax_model_emits_local_exp_pool` (from Task 3):

```rust
    assert!(!asm.contains("expf"), "no libm expf symbol after inlining:\n{asm}");
```

Leave `matmul_does_not_call_extern_math` unchanged.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p profiles-arm64 2>&1 | tail -25`
Expected: FAIL — emitters still emit `bl _expf`, no `fcvtns`/`.Lexp_c7`.

- [ ] **Step 3: Add `emit_exp_inline` to `exp.rs` + re-export it**

Append to `profiles/arm64/src/ops/exp.rs`:

```rust
/// Emit AArch64 inline `exp` for x ≤ 0. Input in `s0`; result in `s0`.
///
/// Scratch (all non-loop-live; the softmax loop owns x19-x23/s8/s9, NOT
/// touched here): x9 (pool base), w11 (z), w12 (pow bits), s1-s5 (FP temps).
/// Branchless underflow clamp via `csel` — no labels, safe to inline at
/// multiple sites without unique suffixes.
pub fn emit_exp_inline() -> String {
    let mut s = String::new();
    s.push_str("    ; --- inline exp(x), x<=0 (M17) ---\n");
    s.push_str("    adrp    x9, .Lexp_log2e@PAGE\n");
    s.push_str("    ldr     s1, [x9, .Lexp_log2e@PAGEOFF]\n");
    s.push_str("    fmul    s2, s0, s1\n");
    s.push_str("    fcvtns  w11, s2\n"); // z = round-nearest-even
    s.push_str("    scvtf   s2, w11\n"); // zf
    s.push_str("    ldr     s1, [x9, .Lexp_ln2hi@PAGEOFF]\n");
    s.push_str("    fmsub   s3, s2, s1, s0\n"); // r = x - zf*ln2hi
    s.push_str("    ldr     s1, [x9, .Lexp_ln2lo@PAGEOFF]\n");
    s.push_str("    fmsub   s3, s2, s1, s3\n"); // r -= zf*ln2lo
    s.push_str("    ldr     s4, [x9, .Lexp_c7@PAGEOFF]\n"); // p = C7
    for k in (0..7).rev() {
        s.push_str(&format!("    ldr     s1, [x9, .Lexp_c{}@PAGEOFF]\n", k));
        s.push_str("    fmadd   s4, s4, s3, s1\n"); // p = p*r + C[k]
    }
    s.push_str("    add     w11, w11, #127\n");
    s.push_str("    lsl     w12, w11, #23\n");
    s.push_str("    cmp     w11, #0\n");
    s.push_str("    csel    w12, wzr, w12, le\n"); // z+127<=0 → pow bits 0
    s.push_str("    fmov    s5, w12\n");
    s.push_str("    fmul    s0, s4, s5\n"); // result = p * pow
    s.push_str("    ; --- end inline exp ---\n");
    s
}
```

In `profiles/arm64/src/ops/mod.rs` add `pub use exp::emit_exp_inline;` (now used → no warning).

- [ ] **Step 4: Replace the call in `emit_softmax`**

In `profiles/arm64/src/ops/softmax.rs` (Pass-2 loop) replace:

```rust
    s.push_str(&format!("    bl      {}expf\n", sym_prefix));
    // x6 must be recomputed: bl _expf may have clobbered it (caller-saved).
    s.push_str("    add     x6, x20, x21\n");
```

with:

```rust
    s.push_str(&crate::ops::emit_exp_inline());
    // x6 recomputed: emit_exp_inline clobbers scratch (x9/x11/x12/s1-s5).
    s.push_str("    add     x6, x20, x21\n");
```

(FFI save/restore around the loop is RETAINED per spec §3.4 — do not remove.) If clippy flags `sym_prefix` as now-unused in this fn, prefix the unused binding with `_` only if it is genuinely never referenced elsewhere in the fn.

- [ ] **Step 5: Replace the call in the fused tail**

In `profiles/arm64/src/ops/linear.rs` (`.Lfsmx_exp` loop) replace:

```rust
                s.push_str(&format!("    bl      {}expf\n", sym_prefix));
                // x6 may have been clobbered by _expf (caller-saved); recompute.
                s.push_str("    add     x6, x20, x21\n");
```

with:

```rust
                s.push_str(&crate::ops::emit_exp_inline());
                // x6 recomputed: emit_exp_inline clobbers scratch.
                s.push_str("    add     x6, x20, x21\n");
```

- [ ] **Step 6: Run unit tests + clippy**

Run: `cargo test -p profiles-arm64 2>&1 | tail -25` → PASS (flipped tests + pool no-`expf` assertion green).
Run: `cargo clippy -p profiles-arm64 --all-targets -- -D warnings` → exit 0.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add -A
git commit -m "feat(m17): arm64 inline exp — replace bl _expf with degree-7 Taylor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: arm64 layer-1 bit-exact FFI + underflow-clamp FFI

**Files:** Create `tests/fixtures/softmax_only.nfl`; modify `profiles/arm64/tests/common/mod.rs`, `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Create the isolated fixture**

`tests/fixtures/softmax_only.nfl`:

```
model SoftmaxOnly [batch=4, k=8]:
    x: Tensor[batch, k]

    x -> softmax
```

- [ ] **Step 2: Add `softmax_ref` to `common/mod.rs`**

```rust
/// Reference per-row stable softmax using `exp_ref` — bit-exact with the
/// emitter. Sequential max/sub/exp/sum/div mirror the asm pass order; only
/// `exp_ref` carries the fused-FMA divergence.
pub fn softmax_ref(input: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * k];
    for i in 0..b {
        let row = &input[i * k..(i + 1) * k];
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            m = m.max(v);
        }
        let mut sum = 0.0f32;
        for (j, &v) in row.iter().enumerate() {
            let e = exp_ref(v - m);
            out[i * k + j] = e;
            sum += e;
        }
        for j in 0..k {
            out[i * k + j] /= sum;
        }
    }
    out
}
```

- [ ] **Step 3: Write the failing bit-exact FFI test**

In `profiles/arm64/tests/integration.rs` (uses the file-local `deterministic_input`):

```rust
#[test]
fn softmax_only_ffi_bit_exact_vs_exp_ref() {
    if !common::cc_available() {
        eprintln!("skip: cc unavailable");
        return;
    }
    let (b, k) = (4usize, 8usize);
    let src = std::fs::read_to_string("../../tests/fixtures/softmax_only.nfl").unwrap();
    let uir = compiler::ir::build(&compiler::parse(&src).unwrap()).unwrap();
    let asm = profiles_arm64::lower(&uir).unwrap();
    let dylib = common::compile_to_dylib(&asm.source, "softmax_only");
    let input = deterministic_input(b * k);
    let params = vec![0.0f32; 1]; // softmax has no params; non-empty avoids a dangling ptr
    let mut output = vec![0.0f32; b * k];
    let lib = unsafe { libloading::Library::new(&dylib) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_SoftmaxOnly") }.unwrap();
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()) };
    let want = common::softmax_ref(&input, b, k);
    for (i, (g, w)) in output.iter().zip(want.iter()).enumerate() {
        assert!(g.to_bits() == w.to_bits(),
            "softmax[{i}]: asm={:#x} ref={:#x}", g.to_bits(), w.to_bits());
    }
}
```

- [ ] **Step 4: Run it**

Run: `cargo test -p profiles-arm64 --test integration softmax_only_ffi_bit_exact_vs_exp_ref`
Expected: PASS. If it fails, a constant in `exp.rs` vs `common/mod.rs` drifted — diff the two const blocks first.

- [ ] **Step 5: Write the underflow-clamp FFI test**

```rust
#[test]
fn softmax_only_ffi_underflow_clamp_agrees_with_libm() {
    if !common::cc_available() {
        eprintln!("skip: cc unavailable");
        return;
    }
    let (b, k) = (4usize, 8usize);
    let mut input = deterministic_input(b * k);
    input[0] = 120.0; // row-0 max
    for j in 1..k {
        input[j] = -120.0; // x - max ≈ -240 → z < -127 → flush
    }
    let src = std::fs::read_to_string("../../tests/fixtures/softmax_only.nfl").unwrap();
    let uir = compiler::ir::build(&compiler::parse(&src).unwrap()).unwrap();
    let asm = profiles_arm64::lower(&uir).unwrap();
    let dylib = common::compile_to_dylib(&asm.source, "softmax_only_clamp");
    let params = vec![0.0f32; 1];
    let mut output = vec![0.0f32; b * k];
    let lib = unsafe { libloading::Library::new(&dylib) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_SoftmaxOnly") }.unwrap();
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()) };
    for j in 1..k {
        assert_eq!(output[j].to_bits(), 0.0f32.to_bits(), "term {j} must flush to +0.0");
    }
    let row_sum: f32 = output[0..k].iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-6, "row must still sum to 1: {row_sum}");
}
```

- [ ] **Step 6: Run + commit**

Run: `cargo test -p profiles-arm64 --test integration 2>&1 | tail -15` → PASS (both new FFI tests + existing softmax tolerance tests green).

```bash
cargo fmt --all
git add -A
git commit -m "test(m17): arm64 layer-1 bit-exact FFI + underflow-clamp evidence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: x86_64 `exp_ref` + layer-2 sweep

Mirror of Task 2 with the no-FMA (separate `*`/`+`) variant.

**Files:** `profiles/x86_64/tests/common/mod.rs`, `profiles/x86_64/tests/integration.rs`

- [ ] **Step 1: Write the failing sweep test**

In `profiles/x86_64/tests/integration.rs`:

```rust
#[test]
fn exp_ref_within_one_ulp_of_libm() {
    let ulp_diff = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
    let mut x = -80.0_f32;
    while x <= 0.0 {
        let (got, want) = (common::exp_ref(x), x.exp());
        assert!(ulp_diff(got, want) <= 1, "x={x}: {} ulp", ulp_diff(got, want));
        x += 0.0009765625;
    }
    for &x in &[0.0_f32, -std::f32::consts::LN_2, -1.0, -10.0, -50.0] {
        assert!(ulp_diff(common::exp_ref(x), x.exp()) <= 1, "x={x}");
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p profiles-x86_64 --test integration exp_ref_within_one_ulp_of_libm`
Expected: FAIL — `common::exp_ref` does not exist.

- [ ] **Step 3: Implement `exp_ref` (x86_64 — separate `*`/`+`)**

In `profiles/x86_64/tests/common/mod.rs`:

```rust
/// Reference f32 exp for x ≤ 0 — bit-exact match for the x86_64 inline emitter.
///
/// CRITICAL: SSE2 has no scalar FMA, so every multiply-accumulate is a separate
/// `mulss`+`addss` (two roundings). This port uses separate `*` and `+`/`-` —
/// NOT `f32::mul_add`. (Mirror of the per-profile reference_matmul split.)
pub fn exp_ref(x: f32) -> f32 {
    const LOG2E: f32 = 1.4426950408889634;
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -0.00021219444005469057;
    const C: [f32; 8] =
        [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0];
    let z = (x * LOG2E).round_ties_even() as i32; // cvtss2si: ties-even
    let zf = z as f32;
    let r = x - zf * LN2_HI; // two roundings (mul then sub)
    let r = r - zf * LN2_LO;
    let mut p = C[7];
    for k in (0..7).rev() {
        p = p * r + C[k]; // mul then add — two roundings
    }
    let zp = z + 127;
    let pow = if zp <= 0 { 0.0_f32 } else { f32::from_bits((zp as u32) << 23) };
    p * pow
}
```

- [ ] **Step 4: Run the sweep**

Run: `cargo test -p profiles-x86_64 --test integration exp_ref_within_one_ulp_of_libm`
Expected: PASS. If a worst point hits 2 ulp (likelier here — no FMA), add `LN2_LO2 = 1.0e-9` + a third reduction step (mirror in the Task 8 emitter). Re-run until ≤ 1 ulp. (Pure Rust — runs on macOS and Linux.)

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add -A
git commit -m "test(m17): x86_64 exp_ref port (no-FMA) + layer-2 <=1ulp sweep

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: x86_64 `.rodata` constant pool

Follows the existing layernorm `.section .rodata` pattern. Pool-only (no `bl`-removal until Task 8) → green commit.

**Files:** Create `profiles/x86_64/src/ops/exp.rs`; modify `profiles/x86_64/src/ops/mod.rs`, `profiles/x86_64/src/codegen.rs`; test `profiles/x86_64/src/tests.rs`

- [ ] **Step 1: Write the failing unit test (pool presence only)**

In `profiles/x86_64/src/tests.rs`:

```rust
#[test]
fn softmax_model_emits_local_exp_pool() {
    let src = "model S [batch=2, k=3]:\n    x: Tensor[batch, k]\n    x -> softmax\n";
    let uir = compiler::ir::build(&compiler::parse(src).unwrap()).unwrap();
    let asm = crate::lower(&uir).unwrap().source;
    assert!(asm.contains(".section .rodata"), "no rodata pool:\n{asm}");
    assert!(asm.contains(".Lexp_log2e:"), "no log2e constant:\n{asm}");
    assert!(asm.contains(".Lexp_c7:"), "no c7 constant:\n{asm}");
    assert_eq!(asm.matches(".Lexp_log2e:").count(), 1, "pool must be unique per file");
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cargo test -p profiles-x86_64 softmax_model_emits_local_exp_pool`
Expected: FAIL — no pool yet.

- [ ] **Step 3: Create `profiles/x86_64/src/ops/exp.rs` (pool only)**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Inline bare-metal `exp` for the softmax domain (x ≤ 0) — x86_64 SSE2.
//! Replaces the M3-era `call expf@PLT`. See the M17 design spec.

const LOG2E: f32 = 1.4426950408889634;
const LN2_HI: f32 = 0.693359375;
const LN2_LO: f32 = -0.00021219444005469057;
const C: [f32; 8] =
    [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0];

/// File-local `.rodata` pool, emitted once per file from `walk_uir` when
/// `uir.has_softmax()`. Mirrors the layernorm pool pattern
/// (profiles/x86_64/src/ops/layernorm.rs). `.L`-local labels.
pub fn exp_pool_x86_64() -> String {
    let mut s = String::new();
    s.push_str(".section .rodata\n");
    s.push_str(".align 4\n");
    s.push_str(&format!(".Lexp_log2e: .long 0x{:08x}\n", LOG2E.to_bits()));
    s.push_str(&format!(".Lexp_ln2hi: .long 0x{:08x}\n", LN2_HI.to_bits()));
    s.push_str(&format!(".Lexp_ln2lo: .long 0x{:08x}\n", LN2_LO.to_bits()));
    for (k, c) in C.iter().enumerate() {
        s.push_str(&format!(".Lexp_c{}: .long 0x{:08x}\n", k, c.to_bits()));
    }
    s
}
```

- [ ] **Step 4: Wire module + emit pool (before the GNU-stack directive)**

In `profiles/x86_64/src/ops/mod.rs`: `pub mod exp;` + `pub use exp::exp_pool_x86_64;` (not `emit_exp_inline` yet).

In `profiles/x86_64/src/codegen.rs` `walk_uir`, insert before the existing `.note.GNU-stack` block:

```rust
    if uir.has_softmax() {
        source.push('\n');
        source.push_str(&crate::ops::exp_pool_x86_64());
    }
```

- [ ] **Step 5: Run full x86_64 suite + clippy + commit**

Run: `cargo test -p profiles-x86_64 2>&1 | tail -15` → PASS.
Run: `cargo clippy -p profiles-x86_64 --all-targets -- -D warnings` → exit 0.

```bash
cargo fmt --all
git add -A
git commit -m "feat(m17): x86_64 .rodata exp constant pool (file-local labels)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: x86_64 `emit_exp_inline` + wire into both sites + flip asm-shape tests

**Files:** `profiles/x86_64/src/ops/exp.rs`, `profiles/x86_64/src/ops/mod.rs`, `profiles/x86_64/src/ops/softmax.rs`, `profiles/x86_64/src/ops/linear.rs`, `profiles/x86_64/src/tests.rs`

- [ ] **Step 1: Flip the asm-shape unit tests + add no-`expf`**

In `profiles/x86_64/src/tests.rs`, change tests asserting `s.contains("call    expf@PLT")` (≈ lines 211/256/748/790) to:

```rust
    assert!(!s.contains("call    expf@PLT"), "expf must be inlined now:\n{s}");
    assert!(s.contains("cvtss2si"), "missing round-to-int (range reduction):\n{s}");
    assert!(s.contains(".Lexp_c7(%rip)"), "missing Horner constant load:\n{s}");
```

Rename `standalone_softmax_emits_three_pass_with_call_expf_plt` → `*_with_inline_exp` and `linear_softmax_fused_emits_row_wise_tail_with_call_expf_plt` → `*_with_inline_exp`. For the `*recompute %rax after call*` test (≈ line 307-315), the recompute is RETAINED but its `call expf@PLT` anchor is gone — re-anchor the `s.find(...)` on `.Lexp_c0(%rip)` (last pool load before the recompute) instead of `"call    expf@PLT"`; keep the row_max/row_sum slot-offset assertions. Add the no-`expf` global assertion to `softmax_model_emits_local_exp_pool`:

```rust
    assert!(!asm.contains("expf"), "no libm expf symbol after inlining:\n{asm}");
```

Leave `matmul_does_not_call_expf_plt` unchanged.

- [ ] **Step 2: Run to confirm they fail**

Run: `cargo test -p profiles-x86_64 2>&1 | tail -25`
Expected: FAIL — emitters still emit `call expf@PLT`.

- [ ] **Step 3: Add `emit_exp_inline` to `exp.rs` + re-export**

Append to `profiles/x86_64/src/ops/exp.rs`:

```rust
/// Emit x86_64 SSE2 inline `exp` for x ≤ 0. Input in `%xmm0`; result in `%xmm0`.
///
/// Scratch (all non-loop-live; the softmax loop owns %rbx/%r12-%r15 + stack
/// slots, NOT touched here): %eax (z), %ecx/%edx (pow bits), %xmm1-%xmm5.
/// Branchless underflow clamp via `cmovle` — no labels.
pub fn emit_exp_inline() -> String {
    let mut s = String::new();
    s.push_str("    # --- inline exp(x), x<=0 (M17) ---\n");
    s.push_str("    movss   .Lexp_log2e(%rip), %xmm1\n");
    s.push_str("    mulss   %xmm0, %xmm1\n");
    s.push_str("    cvtss2si %xmm1, %eax\n"); // z = nearest, ties-even
    s.push_str("    cvtsi2ss %eax, %xmm2\n"); // zf
    s.push_str("    movss   .Lexp_ln2hi(%rip), %xmm3\n");
    s.push_str("    mulss   %xmm2, %xmm3\n");
    s.push_str("    movss   %xmm0, %xmm5\n"); // preserve x
    s.push_str("    subss   %xmm3, %xmm5\n"); // x - zf*ln2hi
    s.push_str("    movss   .Lexp_ln2lo(%rip), %xmm3\n");
    s.push_str("    mulss   %xmm2, %xmm3\n");
    s.push_str("    subss   %xmm3, %xmm5\n"); // r
    s.push_str("    movss   .Lexp_c7(%rip), %xmm4\n"); // p = C7
    for k in (0..7).rev() {
        s.push_str("    mulss   %xmm5, %xmm4\n");
        s.push_str(&format!("    addss   .Lexp_c{}(%rip), %xmm4\n", k)); // p = p*r + C[k]
    }
    s.push_str("    addl    $127, %eax\n");
    s.push_str("    movl    %eax, %ecx\n");
    s.push_str("    shll    $23, %ecx\n");
    s.push_str("    xorl    %edx, %edx\n");
    s.push_str("    testl   %eax, %eax\n");
    s.push_str("    cmovle  %edx, %ecx\n"); // z+127<=0 → pow bits 0
    s.push_str("    movd    %ecx, %xmm5\n");
    s.push_str("    mulss   %xmm5, %xmm4\n"); // p * pow
    s.push_str("    movss   %xmm4, %xmm0\n");
    s.push_str("    # --- end inline exp ---\n");
    s
}
```

In `profiles/x86_64/src/ops/mod.rs` add `pub use exp::emit_exp_inline;`.

- [ ] **Step 4: Replace the call in `emit_softmax`**

In `profiles/x86_64/src/ops/softmax.rs` replace:

```rust
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax clobbered by call; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
```

with:

```rust
    s.push_str(&crate::ops::emit_exp_inline());
    // %rax recomputed: emit_exp_inline clobbers scratch (%eax/%ecx/%edx/%xmm1-5).
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
```

(FFI save/restore RETAINED per spec §3.4.)

- [ ] **Step 5: Replace the call in the fused tail**

In `profiles/x86_64/src/ops/linear.rs` (`emit_fused_softmax_tail`) replace:

```rust
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax was clobbered; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
```

with:

```rust
    s.push_str(&crate::ops::emit_exp_inline());
    // %rax recomputed: emit_exp_inline clobbers scratch.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
```

If `sym_prefix` becomes unused in `emit_fused_softmax_tail`, drop it from the signature + its single caller in `emit_linear`. Run `cargo build -p profiles-x86_64` and resolve any unused-parameter warning.

- [ ] **Step 6: Run unit tests + clippy + commit**

Run: `cargo test -p profiles-x86_64 2>&1 | tail -25` → PASS.
Run: `cargo clippy -p profiles-x86_64 --all-targets -- -D warnings` → exit 0.

```bash
cargo fmt --all
git add -A
git commit -m "feat(m17): x86_64 inline exp — replace call expf@PLT with degree-7 Taylor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: x86_64 layer-1 bit-exact FFI + clamp FFI + drop `-lm` (bare-metal proof)

FFI tests run on the Linux x86_64 CI leg — gate them `#[cfg(target_os = "linux")]` (matching `ffn_ffi` / `transformer_block_ffi`).

**Files:** `profiles/x86_64/tests/common/mod.rs`, `profiles/x86_64/tests/integration.rs` (reuse `tests/fixtures/softmax_only.nfl`)

- [ ] **Step 1: Add `softmax_ref` to x86_64 `common/mod.rs`**

Identical body to Task 5 Step 2 (the max/sub/sum/div are FMA-free), calling this crate's no-FMA `exp_ref`:

```rust
pub fn softmax_ref(input: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * k];
    for i in 0..b {
        let row = &input[i * k..(i + 1) * k];
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            m = m.max(v);
        }
        let mut sum = 0.0f32;
        for (j, &v) in row.iter().enumerate() {
            let e = exp_ref(v - m);
            out[i * k + j] = e;
            sum += e;
        }
        for j in 0..k {
            out[i * k + j] /= sum;
        }
    }
    out
}
```

- [ ] **Step 2: Write the failing bit-exact FFI test (Linux-only)**

In `profiles/x86_64/tests/integration.rs`:

```rust
#[cfg(target_os = "linux")]
#[test]
fn softmax_only_ffi_bit_exact_vs_exp_ref() {
    if !common::cc_available() {
        eprintln!("skip: cc unavailable");
        return;
    }
    let (b, k) = (4usize, 8usize);
    let src = std::fs::read_to_string("../../tests/fixtures/softmax_only.nfl").unwrap();
    let uir = compiler::ir::build(&compiler::parse(&src).unwrap()).unwrap();
    let asm = profiles_x86_64::lower(&uir).unwrap();
    let so = common::compile_to_so(&asm.source, "softmax_only");
    let input = deterministic_input(b * k);
    let params = vec![0.0f32; 1];
    let mut output = vec![0.0f32; b * k];
    let lib = unsafe { libloading::Library::new(&so) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_SoftmaxOnly") }.unwrap();
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()) };
    let want = common::softmax_ref(&input, b, k);
    for (i, (g, w)) in output.iter().zip(want.iter()).enumerate() {
        assert!(g.to_bits() == w.to_bits(),
            "softmax[{i}]: asm={:#x} ref={:#x}", g.to_bits(), w.to_bits());
    }
}
```

- [ ] **Step 3: Add the underflow-clamp FFI test (Linux-only)**

Same structure, gated `#[cfg(target_os = "linux")]`, `input[0]=120.0` / others `-120.0`, asserting flush-to-`0` for `j in 1..k` and `row_sum ≈ 1`. Use `common::compile_to_so(&asm.source, "softmax_only_clamp")`.

- [ ] **Step 4: Drop `-lm` from `compile_to_so` (bare-metal proof)**

In `profiles/x86_64/tests/common/mod.rs` `compile_to_so`, remove the `.args(["-lm"])` line. After M17, no emitted `.so` references libm, so every x86_64 FFI test must still link and pass with libm absent — direct evidence of bare-metal output.

- [ ] **Step 5: Run (Linux) + commit**

Run (Linux): `cargo test -p profiles-x86_64 --test integration 2>&1 | tail -20` → PASS (new FFI tests + all existing FFI tests link without `-lm`).
On macOS: `cargo test -p profiles-x86_64` (sweep + unit tests; FFI `#[cfg]`-skipped).

```bash
cargo fmt --all
git add -A
git commit -m "test(m17): x86_64 layer-1 bit-exact FFI + clamp + drop -lm (bare-metal proof)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Documentation + milestone closure + final gates

**Files:** `docs/profile_guide/{arm64,x86_64}.md`, `docs/language_reference/uir.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `DEVLOG.md`

- [ ] **Step 1: Profile guides**

Rewrite each softmax section: replace the `bl _expf` / `call expf@PLT` narrative with the inline algorithm (Cody-Waite → degree-7 Taylor Horner → `2^z` bit-trick + branchless clamp), the `.rodata`/`__const` file-local pool, and the scratch contract. State that FFI save/restore + the callee-saved prologue are RETAINED in M17 (M18 removes them). Note the x86_64 `-lm`-free link as bare-metal evidence.

- [ ] **Step 2: PROJECT_SPEC.md**

- Add the Milestone 17 row: "Axis 3 first leg — bare-metal inline `expf` (Cody-Waite + Taylor-7); `has_softmax` rename; two-layer validation (bit-exact vs `exp_ref` + ≤ 1 ulp vs libm); file-local `.rodata`/`__const` pool; x86_64 `.so` links without `-lm`. M18 records leaf-cleanup. Test count: 466 → <final>."
- §Decisions: add the `.rodata`/`__const` pool decision and the two-layer accuracy contract ("≤ 1 ulp, confirmed by sweep; widen LN2 split, not degree").
- Strategic Roadmap: Axis 3 — first leg closed M17; record M18 (softmax leaf-cleanup) as second leg with the 7-point deferral list from spec §9.
- §Known Latent Hazards: confirm stays empty.

- [ ] **Step 3: CLAUDE.md**

- "Current Status" → M17 complete + new test count.
- Repo structure: note the new `ops/exp.rs` primitive on both profiles.
- Update the arm64/x86_64 op-list lines: softmax now "inline bare-metal exp, no libm".

- [ ] **Step 4: DEVLOG.md**

Add the M17 entry (What was done / Decisions / Problems / Next step → M18 leaf-cleanup) above the most recent entry. Note the two spec refinements (arm64-new `.rodata`; new bit-exact test + `softmax_only.nfl` fixture).

- [ ] **Step 5: Final workspace gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | tail -25
grep -rn "calls_extern_math\|bl      _expf\|expf@PLT" --include="*.rs" . | grep -v /target/
```
Expected: fmt clean; clippy exit 0; all tests pass; grep returns nothing. Record the final macOS arm64 test count + the +Linux delta.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs(m17): close milestone — profile guides, spec, status, devlog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes (for the executor)

- **Constant drift is the top risk.** `exp.rs` (emitter) and `common/mod.rs` (`exp_ref`) each hold the 11 f32 literals; they MUST be textually identical. The Task 5/9 bit-exact FFI tests are the guard — if one fails, diff the two const blocks first.
- **Round-to-nearest must be ties-to-even on both sides.** `fcvtns` / `cvtss2si` are ties-even; ports use `round_ties_even()`. Never `f32::round()` (ties-away).
- **Per-profile FMA divergence is mandatory.** arm64 port uses `mul_add` (matches `fmadd`/`fmsub`); x86_64 port uses separate `*`/`+`/`-` (matches `mulss`+`addss`/`subss`).
- **Minimal-swap discipline:** do NOT touch the softmax loop's register set, FFI save/restore, the callee-saved prologue, or the leaf flag — those are M18. Only the exp-pass instruction block changes.
- **Assembler validates directives:** FFI tests assemble with real `cc`, so any `.section __TEXT,__const` / `@PAGE` / `(%rip)` mistake fails loudly at link/runtime.
- **If the ≤ 1 ulp sweep fails (Task 2/6):** widen the LN2 split (add `LN2_LO2`) in both the port and the emitter — never bump the polynomial degree (spec §5.2).
```
