# M14 Opener — Close LH-1/2/3 in x86_64 emit_linear — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all three Known Latent Hazards (LH-1, LH-2, LH-3) in `profiles/x86_64/src/ops/linear.rs` via uniform ABI-register relocation. Mandatory by §LH process — M14 main fixture `pre_ln_block.nfl` (Plan 2) at N=2 + linear-with-bias triggers LH-1; LH-2/3 closed proactively in same commit per memory rule "triggered cleanup is an obligation" and §LH "leaving an entry here longer than one milestone is a process failure".

**Architecture:**
- **LH-1** (N=2 + bias path j-counter aliases output_reg `%rcx`): relocate j-counter from `%rcx` to `%rbp`. M13 Task 1 precedent — `%rbp` callee-saved by unconditional function-level prologue (`pushq %rbp`/`popq %rbp`), never read by op bodies (verified by grep). Bias-add at N=2 reads `({output_reg=%rcx}, %rbp, 4)` correctly post-fix (was `(%rcx, %rcx, 4)` pre-fix — base aliased offset). Side-effect: M13 save block's `pushq %rcx`/`popq %rcx` becomes dead (j-counter no longer in %rcx, body never writes %rcx) and is removed.
- **LH-2** (N=3 src ptr scratch `%r8` aliases output_reg): relocate src ptr from `%r8` to `%r14`. `%r14` is callee-saved per SysV — op-local `pushq %r14` at body entry, `popq %r14` at body exit. M13 pre-Task-5 arm64 precedent for op-local save/restore (stp/ldp x3 inside emit_linear body, no function prologue change).
- **LH-3** (N=4 weight ptr scratch `%r9` aliases output_reg): relocate weight ptr from `%r9` to `%r15`. Same op-local push/pop pattern.

All three fixes preserve M12 §9.1 invariant ("op-emitter body must NOT touch any ABI argument register") at all N=1..4. `compute_callee_saved` is **NOT** modified — function-level prologue surface unchanged.

**Tech Stack:** Rust 2024 workspace; x86_64 SSE2 SysV; existing test harness in `profiles/x86_64/src/tests.rs`.

---

## Spec reference

This plan implements §3 of `docs/superpowers/specs/2026-05-10-m14-layernorm-design.md` (the opener cleanup). Per-section traceability:

| Spec section | Plan step |
|--------------|-----------|
| §3.1 (LH table restated) | Step 1 (audit) |
| §3.2 (constraints) | Steps 4, 7, 10 (each fix verified against constraints) |
| §3.3 (fix mechanism + preference order) | Steps 4, 7, 10 |
| §3.4 (test coverage — ABI-invariant unit tests) | Steps 2-3, 5-6, 8-9 |
| §3.5 (audit gate) | Step 14 |
| §3.6 (out of scope) | n/a — explicit non-changes documented in §"File Structure / Not modified" |

**Note on a tiny spec wording slip:** §3.5 audit gate item 2 says "Verify zero new `pushq` in op bodies". This is shorthand — §3.2 explicitly permits "Op-level push/pop". The intent is "function-level prologue surface unchanged"; the LH-2/3 fixes do add `pushq %r14`/`pushq %r15` inside the op body, which §3.2 allows. The audit step in this plan checks the correct invariant (function prologue unchanged).

---

## File Structure

### Modified
- **`profiles/x86_64/src/ops/linear.rs`** (the entire LH cleanup) — three classes of edit:
  1. `emit_linear` body register usage: j-counter `%rcx` → `%rbp` (six call sites); src ptr scratch `%r8` → `%r14` (two call sites + push/pop); weight ptr scratch `%r9` → `%r15` (three call sites + push/pop).
  2. M13 save block at lines 159-164 + 253-257: remove `pushq %rcx`/`popq %rcx` (dead post-fix); add `pushq %r14`/`pushq %r15` op-local saves and matching `popq %r15`/`popq %r14` restores.
  3. Module doc-comment lines 1-49: rewrite to reflect post-fix state (LH-1/2/3 closed, no longer "latent hazards remain").
- **`profiles/x86_64/src/tests.rs`** — add three new ABI-invariant tests (`emit_linear_n2_with_bias_does_not_alias_output_reg_in_body`, `emit_linear_n3_does_not_clobber_output_reg`, `emit_linear_n4_does_not_clobber_output_reg`).
- **`PROJECT_SPEC.md`** §"Known Latent Hazards" — remove rows LH-1, LH-2, LH-3 from the table. If the table becomes empty, retain the heading with a one-line comment "currently empty — populate as new latent hazards are discovered".

### Not modified (notable — out-of-scope per spec §3.6)
- `profiles/x86_64/src/buffer.rs` — `compute_callee_saved` and `compute_is_leaf` unchanged. Function-level callee-saved set driven only by other ops (matmul/softmax). The op-local `pushq %r14`/`pushq %r15` for LH-2/3 lives entirely inside `emit_linear`'s emitted asm string.
- `profiles/x86_64/src/abi.rs` — `AbiContext` unchanged.
- `profiles/x86_64/src/asm.rs` — function-level prologue/epilogue unchanged.
- `profiles/arm64/` — already fixed in M13 pre-Task-5 commit `c7fba5b` (stp/ldp x3/x4/x5 save/restore). No arm64 LH currently open.
- `profiles/x86_64/src/ops/matmul.rs` — already fixed in M13 Task 1 (commit `2d84281` predecessor — j-counter relocated `%r9` → `%rbp`).
- `tests/fixtures/` — no new positive fixtures. LH-1 will be validated end-to-end by Plan 2's `pre_ln_block.nfl` FFI integration test. LH-2/3 covered by ABI-invariant unit tests only (per spec §3.4 — N=3/4 + linear+bias is not an M14 use case, no new fixtures justified).

---

## Task 1: Close LH-1/2/3 in x86_64 emit_linear (single commit)

**Files:**
- Modify: `profiles/x86_64/src/ops/linear.rs`
- Test: `profiles/x86_64/src/tests.rs` (three new tests)
- Modify: `PROJECT_SPEC.md`

---

- [ ] **Step 1: Read current state of emit_linear and identify all register-conflict sites**

Read `profiles/x86_64/src/ops/linear.rs` in full. Confirm the following pre-fix register usage in the matmul body (between line 156 ABI save block and line 257 ABI restore):

| Register | Use site (line) | Conflict with output_reg at N |
|----------|-----------------|-------------------------------|
| `%rcx` | line 174 (j-counter init), 177 (cmp), 199 (offset arith), 214 (bias-add base offset), 239 (offset arith), 242 (incr) | N=2 (output_reg = INPUT_REGS[3] = `%rcx`) — **LH-1** |
| `%r8` | line 90 (`abi.materialise_ptr(src_loc, "%r8", ...)`), 193 (`movss (%r8, %rsi, 4), %xmm1`) | N=3 (output_reg = INPUT_REGS[4] = `%r8`) — **LH-2** |
| `%r9` | lines 95, 99 (weight base setup), 200 (`movss (%r9, %rsi, 4), %xmm2`) | N=4 (output_reg = INPUT_REGS[5] = `%r9`) — **LH-3** |

Read `profiles/x86_64/src/tests.rs` to find the existing ABI-invariant test pattern from commit `c993712` (search for tests asserting "no INPUT_REGS used as scratch"). Mirror that pattern in Steps 2, 5, 8.

Also confirm `compute_callee_saved` in `profiles/x86_64/src/buffer.rs` does NOT currently include `%r14` or `%r15` (these will be op-local push/pop in LH-2/3 fix, NOT added to function-level set).

**No code change in this step** — this is a read-only audit. Output: confirmed register-conflict map matching the table above.

- [ ] **Step 2: Write failing ABI-invariant test for LH-1 (N=2 + linear-with-bias)**

Add to `profiles/x86_64/src/tests.rs`:

```rust
#[test]
fn emit_linear_n2_with_bias_does_not_alias_output_reg_in_body() {
    // LH-1 regression guard.
    //
    // Pre-fix: at N=2, output_reg = %rcx (INPUT_REGS[3]). The j-counter
    // also lived in %rcx, so the bias-add expanded to
    // `movss (%rcx, %rcx, 4), %xmm5` — base aliased offset, wrong output.
    //
    // Post-fix: j-counter relocated to %rbp. Bias-add expands to
    // `movss (%rcx, %rbp, 4), %xmm5` — base = bias_base in %rcx, offset
    // = j-counter in %rbp. Correct.
    //
    // Pattern: lower a minimal N=2 + linear-with-bias UIR, extract the
    // matmul body (between `.Lmm_i_` and `.Lmm_i_end_` labels), assert
    // post-fix invariants on the body.

    let asm = lower_minimal_n2_linear_with_bias();  // helper from c993712 test infra
    let body = extract_matmul_body(&asm);

    // Pre-fix marker — must NOT appear:
    assert!(
        !body.contains("xorq    %rcx, %rcx"),
        "j-counter init must not be %rcx (would alias output_reg at N=2). Body:\n{body}"
    );
    assert!(
        !body.contains("(%rcx, %rcx,"),
        "bias-add base must not alias offset (LH-1 silent corruption pattern). Body:\n{body}"
    );

    // Post-fix marker — must appear:
    assert!(
        body.contains("xorq    %rbp, %rbp"),
        "j-counter should be relocated to %rbp (M13 Task 1 precedent). Body:\n{body}"
    );
    assert!(
        body.contains("(%rcx, %rbp, 4), %xmm5"),
        "bias-add should read from (output_reg=%rcx, j=%rbp). Body:\n{body}"
    );
}
```

If `lower_minimal_n2_linear_with_bias()` and `extract_matmul_body()` helpers don't exist, look at the existing tests in `profiles/x86_64/src/tests.rs` from commit `c993712` (search for "INPUT_REGS" or "abi" in test names) and either reuse their helpers or replicate the pattern inline. The minimal test fixture is a UIR with one model: input shape `[B, K]`, single `linear[N]` op with bias, output shape `[B, N]`. Use small dimensions (e.g. `B=2, K=2, N=2`) — only the asm shape matters, not numerical correctness.

- [ ] **Step 3: Run the new test, verify it FAILS pre-fix**

```bash
cargo test --package profiles-x86_64 emit_linear_n2_with_bias_does_not_alias_output_reg_in_body -- --nocapture
```

Expected: **FAIL**. The body still contains `xorq %rcx, %rcx` and `(%rcx, %rcx, 4)` patterns. This is the LH-1 bug. Confirms the test is exercising the right path.

- [ ] **Step 4: Apply LH-1 fix — relocate j-counter from %rcx to %rbp**

Edit `profiles/x86_64/src/ops/linear.rs`. Six call sites + the M13 save block:

```rust
// REMOVE line 163 (was: `s.push_str("    pushq   %rcx\n");`):
//   Reason: post-fix, %rcx is never written by the body. The M13 save
//   was added because j-counter clobbered %rcx; now obsolete. (M13 save
//   for %rdi and %rsi remains — those are still clobbered.)

// REMOVE line 254 (was: `s.push_str("    popq    %rcx\n");`):
//   Matching the line 163 removal — LIFO pair stays balanced.

// Line 173 — UPDATE comment:
//   was: "    // 3. Inner j-loop: %rcx = j, compared against n.\n"
//   now: "    // 3. Inner j-loop: %rbp = j, compared against n. M14 LH-1 fix:\n"
//        "    //    relocated from %rcx (which becomes output_reg at N=2)\n"
//        "    //    to %rbp (callee-saved by function-level prologue, never\n"
//        "    //    read by op bodies). M13 Task 1 precedent for emit_matmul.\n"

// Line 174 — REPLACE:
//   was: s.push_str("    xorq    %rcx, %rcx\n");
//   now: s.push_str("    xorq    %rbp, %rbp\n");

// Line 177 — REPLACE:
//   was: s.push_str("    cmpq    %r10, %rcx\n");
//   now: s.push_str("    cmpq    %r10, %rbp\n");

// Line 199 — REPLACE (inside k-loop offset arithmetic):
//   was: s.push_str("    addq    %rcx, %rsi\n"); // %rsi = kk*n + j
//   now: s.push_str("    addq    %rbp, %rsi\n"); // %rsi = kk*n + j

// Line 214 — REPLACE (bias-add):
//   was: s.push_str(&format!("    movss   ({}, %rcx, 4), %xmm5\n", output_reg));
//   now: s.push_str(&format!("    movss   ({}, %rbp, 4), %xmm5\n", output_reg));

// Line 239 — REPLACE (store offset arithmetic):
//   was: s.push_str("    addq    %rcx, %rsi\n");
//   now: s.push_str("    addq    %rbp, %rsi\n");

// Line 242 — REPLACE (j-counter increment):
//   was: s.push_str("    incq    %rcx\n");
//   now: s.push_str("    incq    %rbp\n");
```

Note: `%rbp` is callee-saved by the unconditional function prologue (`pushq %rbp`/`popq %rbp` in `asm.rs::format_function_prologue`). Inside the body it is wide-open scratch — the M13 Task 1 audit verified zero op-emitters read `%rbp`. The same property applies here.

- [ ] **Step 5: Run all tests, verify LH-1 test passes and existing tests still pass**

```bash
cargo test --package profiles-x86_64 -- --nocapture
```

Expected:
- `emit_linear_n2_with_bias_does_not_alias_output_reg_in_body` — **PASS**
- All pre-existing emit_linear tests — **PASS** (the fix preserves N=1 behavior bit-identically because at N=1, output_reg = `%rdx` ≠ `%rcx`, so the body's old %rcx use was already correct — the change to %rbp is semantic-preserving).
- All other workspace tests — **PASS**.

If any pre-existing emit_linear test fails, audit whether it asserted on the literal `%rcx` j-counter token. Such tests need to be updated to assert `%rbp` instead — the fix is correct, the tests need to track the new register choice.

- [ ] **Step 6: Write failing ABI-invariant test for LH-2 (N=3 src ptr scratch)**

Add to `profiles/x86_64/src/tests.rs`:

```rust
#[test]
fn emit_linear_n3_does_not_clobber_output_reg() {
    // LH-2 regression guard.
    //
    // Pre-fix: at N=3, output_reg = %r8 (INPUT_REGS[4]). emit_linear's
    // src_ptr materialise wrote to %r8 (line 90), destroying the FFI
    // output_reg. Subsequent ops in the same function would see a
    // garbage output pointer.
    //
    // Post-fix: src ptr scratch relocated to %r14 (callee-saved per SysV;
    // op-local pushq/popq inside emit_linear body — function-level
    // prologue unchanged).

    let asm = lower_minimal_n3_linear();  // helper or inline UIR
    let body = extract_matmul_body(&asm);

    // Pre-fix marker — must NOT appear. At N=3, output_reg = %r8;
    // `materialise_ptr(src_loc, "%r8", ...)` would emit a `movq ..., %r8`
    // (Input case) or `addq ..., %r8` (StackOffset case via leaq+rsp) — either
    // form writing to %r8 in the body is the LH-2 silent corruption.
    assert!(
        !body.contains(", %r8\n"),
        "src ptr scratch must not write to %r8 at N≥3 (output_reg alias). Body:\n{body}"
    );
    // Post-fix expectation:
    assert!(
        body.contains(", %r14\n"),
        "src ptr should be relocated to %r14. Body:\n{body}"
    );

    // Op-local save/restore must bracket the body:
    assert!(
        asm.contains("    pushq   %r14\n"),
        "op-local pushq %r14 must appear (callee-saved op-local save). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r14\n"),
        "op-local popq %r14 must appear (matching restore). Asm:\n{asm}"
    );
}
```

The negative-assertion form for `%r8` is simpler if you check the materialise sites specifically. The exact assertion form depends on existing test helper conventions — adapt to match them.

- [ ] **Step 7: Run new test, verify FAIL pre-fix**

```bash
cargo test --package profiles-x86_64 emit_linear_n3_does_not_clobber_output_reg -- --nocapture
```

Expected: **FAIL**. Body still contains `, %r8\n` (src ptr materialise) and lacks `, %r14\n` and `pushq %r14`.

- [ ] **Step 8: Apply LH-2 fix — relocate src ptr scratch %r8 → %r14 with op-local push/pop**

Edit `profiles/x86_64/src/ops/linear.rs`:

```rust
// Line 90 — REPLACE:
//   was: abi.materialise_ptr(src_loc, "%r8", &mut s); // src ptr
//   now: abi.materialise_ptr(src_loc, "%r14", &mut s); // src ptr (M14 LH-2: relocated from %r8 — output_reg at N=3)

// Line 193 — REPLACE (src load inside k-loop):
//   was: s.push_str("    movss   (%r8, %rsi, 4), %xmm1\n"); // xmm1 = src[i*k + kk]
//   now: s.push_str("    movss   (%r14, %rsi, 4), %xmm1\n"); // xmm1 = src[i*k + kk]

// Add op-local push %r14 IMMEDIATELY AFTER the M13 save block (after line 164,
// which after Step 4's edit is now `s.push_str("    pushq   %rsi\n");` since
// `pushq %rcx` was removed). The save block then becomes:
//
//   if save_abi {
//       s.push_str("    pushq   %rdi\n");
//       s.push_str("    pushq   %rsi\n");
//       // (pushq %rcx removed in Step 4 — j-counter no longer in %rcx)
//   }
//   // M14 LH-2/3: op-local save of callee-saved scratch used as src/weight ptrs.
//   // Lives inside op body — function-level compute_callee_saved unchanged.
//   s.push_str("    pushq   %r14\n");

// Add matching op-local pop %r14 IMMEDIATELY BEFORE the M13 restore block
// (before what is currently line 254 `popq %rsi` post-Step-4-edit). Restore
// block becomes:
//
//   // M14 LH-2/3 restore (LIFO of the entry op-local push):
//   s.push_str("    popq    %r14\n");
//   if save_abi {
//       s.push_str("    popq    %rsi\n");
//       s.push_str("    popq    %rdi\n");
//   }
```

**Important:** the LH-2 op-local push/pop is **unconditional** (not gated on N). At N=1, `%r8` was non-ABI scratch with no conflict — but uniform op-local save of the new register `%r14` is simpler than conditional code paths and adds only 2 instructions per linear op invocation (negligible). At N≥3, the save/restore is mandatory per SysV (`%r14` is callee-saved).

- [ ] **Step 9: Run tests, verify LH-2 test passes and all others remain green**

```bash
cargo test --package profiles-x86_64 -- --nocapture
```

Expected: LH-2 test **PASS**, LH-1 test from Step 5 still **PASS**, all existing tests **PASS**.

- [ ] **Step 10: Write failing ABI-invariant test for LH-3 (N=4 weight ptr scratch)**

Add to `profiles/x86_64/src/tests.rs`:

```rust
#[test]
fn emit_linear_n4_does_not_clobber_output_reg() {
    // LH-3 regression guard.
    //
    // Pre-fix: at N=4, output_reg = %r9 (INPUT_REGS[5]). emit_linear's
    // weight base setup (lines 95-101) wrote to %r9, destroying the FFI
    // output_reg.
    //
    // Post-fix: weight ptr scratch relocated to %r15 (callee-saved per
    // SysV; op-local pushq/popq inside emit_linear body — function-level
    // prologue unchanged).

    let asm = lower_minimal_n4_linear();  // helper or inline UIR
    let body = extract_matmul_body(&asm);

    // Pre-fix marker — must NOT appear. At N=4, output_reg = %r9; the
    // weight base setup (movq params_reg, %r9 OR leaq weight_offset(params_reg), %r9)
    // would clobber output_reg.
    assert!(
        !body.contains(", %r9\n"),
        "weight ptr scratch must not write to %r9 at N=4 (output_reg alias). Body:\n{body}"
    );
    // Post-fix expectation: weight ptr lives in %r15 (loads via `(%r15, %rsi, 4)`
    // for the inner k-loop weight read).
    assert!(
        body.contains("(%r15, ") || body.contains(", %r15\n"),
        "weight ptr should be relocated to %r15. Body:\n{body}"
    );

    // Op-local save/restore:
    assert!(
        asm.contains("    pushq   %r15\n") && asm.contains("    popq    %r15\n"),
        "op-local pushq/popq %r15 must bracket the body. Asm:\n{asm}"
    );
}
```

Adapt the assertion form to match existing test helper conventions in `profiles/x86_64/src/tests.rs`.

- [ ] **Step 11: Run new test, verify FAIL pre-fix**

```bash
cargo test --package profiles-x86_64 emit_linear_n4_does_not_clobber_output_reg -- --nocapture
```

Expected: **FAIL**. Body still uses `%r9` for weight base.

- [ ] **Step 12: Apply LH-3 fix — relocate weight ptr scratch %r9 → %r15 with op-local push/pop**

Edit `profiles/x86_64/src/ops/linear.rs`:

```rust
// Lines 95-102 — REPLACE the weight base setup block:
//   was:
//     if weight_offset == 0 {
//         s.push_str(&format!("    movq    {}, %r9\n", params_reg));
//     } else {
//         s.push_str(&format!(
//             "    leaq    {}({}), %r9\n",
//             weight_offset * 4,
//             params_reg
//         ));
//     }
//   now:
//     if weight_offset == 0 {
//         s.push_str(&format!("    movq    {}, %r15\n", params_reg));
//     } else {
//         s.push_str(&format!(
//             "    leaq    {}({}), %r15\n",
//             weight_offset * 4,
//             params_reg
//         ));
//     }
//   // M14 LH-3: relocated from %r9 (output_reg at N=4) to %r15
//   // (callee-saved, op-local push/pop below).

// Line 200 — REPLACE (weight load inside k-loop):
//   was: s.push_str("    movss   (%r9, %rsi, 4), %xmm2\n");
//   now: s.push_str("    movss   (%r15, %rsi, 4), %xmm2\n");

// Extend the LH-2/3 op-local save block (added in Step 8) to include %r15:
//
//   // M14 LH-2/3: op-local save of callee-saved scratch used as src/weight ptrs.
//   s.push_str("    pushq   %r14\n");
//   s.push_str("    pushq   %r15\n");
//
// Extend the matching op-local restore block (LIFO order):
//
//   // M14 LH-2/3 restore (LIFO of entry push):
//   s.push_str("    popq    %r15\n");
//   s.push_str("    popq    %r14\n");
```

- [ ] **Step 13: Run tests, verify LH-3 test passes and all others remain green**

```bash
cargo test --package profiles-x86_64 -- --nocapture
```

Expected: All three new ABI-invariant tests **PASS** (LH-1, LH-2, LH-3). All existing tests **PASS**.

- [ ] **Step 14: Audit gate — verify §3.5 invariants**

Run these greps and confirm:

```bash
# 1. No INPUT_REGS used as scratch/counter in emit_linear body.
#    INPUT_REGS = [%rdi, %rsi, %rdx, %rcx, %r8, %r9].
#    Allowed: as base/source in materialise, as bias_base read at line 214,
#    as part of params_reg/output_reg literal expansion. Forbidden: as
#    counter init (xorq %REG, %REG), as imul target, as scratch arith dst.
grep -nE "(xorq|movq|leaq|imulq|incq|addq|subq)\s+[^,]+,\s*%(rdi|rsi|rdx|rcx|r8|r9)\b" \
     profiles/x86_64/src/ops/linear.rs

# Expected: only ABI-context legitimate uses (e.g. `movq params_reg, output_reg`
# in bias_base setup at line 129, `movq %r10, %rax` in i-loop check, etc.).
# NO bare INPUT_REG-as-counter pattern. Each match must be auditable as
# "this is correct ABI-context use, not new scratch".
```

```bash
# 2. Function-level prologue surface unchanged. compute_callee_saved must
#    still report the same set as pre-fix (%rbx + %r12-%r15 conditional on
#    matmul/softmax usage, NOT extended for emit_linear).
grep -n "callee_saved\|RegSet" profiles/x86_64/src/buffer.rs | head -20

# Expected: no reference to LayerNorm or to LH cleanup. The set is unchanged.
```

```bash
# 3. Op-local push/pop balance. Inside emit_linear's emitted asm, any pushq
#    must have a matching popq before the function epilogue runs.
grep -cE 'pushq|popq' profiles/x86_64/src/ops/linear.rs

# Expected: equal counts of pushq and popq invocations in the source
# (pushq %rdi + pushq %rsi + pushq %r14 + pushq %r15 = 4 pushq sites;
#  matching 4 popq sites). The conditional `if save_abi` block does
# both push and pop conditionally, so balance is structural not numerical.
```

If any audit check fails, return to the offending step and re-verify the edit.

- [ ] **Step 15: Update emit_linear module doc-comment (lines 1-49)**

Replace the M13-era doc-comment (which describes LH-1/2/3 as "remaining latent hazards") with a post-fix doc-comment:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Linear (matmul + optional bias + fused PostOps) codegen — x86_64 SSE2.
//!
//! M12 multi-input ABI migration: data-flow accesses to the params and
//! output registers are routed through `AbiContext::params_reg()` /
//! `AbiContext::output_reg()`. For N=1 these resolve to `%rsi` / `%rdx`
//! — bit-identical to M3-M11. For N≥2 they shift (e.g. params → `%rdx`,
//! output → `%rcx` for N=2).
//!
//! M13 ABI-register save (N≥2): the inner k-loop scratch uses `%rsi` for
//! offset arithmetic and `%rdi` as the k-counter. At N=1 these are non-ABI
//! (params=`%rsi`, output=`%rdx`; `%rdi` is pure scratch). At N≥2:
//!   - `%rsi` becomes input(1) — body clobber would break the next op's
//!     read of the second input pointer.
//!   - `%rdi` is always input(0) — body clobber is invisible if no
//!     downstream emitter re-reads input(0).
//! Both saved via `pushq` at body entry, `popq` at body exit (N≥2 only).
//!
//! M14 LH-1/2/3 cleanup (commit `<this-commit>`): three latent hazards
//! resolved uniformly via ABI-register relocation. All scratch now lives
//! in non-INPUT_REGS scope at all N=1..4.
//!
//!   - LH-1 (was: N=2 + linear-with-bias). j-counter relocated `%rcx` →
//!     `%rbp`. `%rbp` is callee-saved by the unconditional function-level
//!     prologue (`pushq %rbp`/`popq %rbp` in `asm.rs::format_function_prologue`)
//!     and unread by op bodies. M13 Task 1 precedent for emit_matmul.
//!     Side-effect: M13 save block's `pushq %rcx`/`popq %rcx` removed
//!     (dead — body no longer writes %rcx).
//!
//!   - LH-2 (was: N=3 src ptr scratch %r8 = output_reg). src ptr scratch
//!     relocated `%r8` → `%r14`. `%r14` is callee-saved per SysV; op-local
//!     `pushq %r14`/`popq %r14` brackets the body. M13 pre-Task-5 arm64
//!     precedent for op-local save/restore. `compute_callee_saved` is
//!     NOT extended — function-level prologue unchanged.
//!
//!   - LH-3 (was: N=4 weight ptr scratch %r9 = output_reg). weight ptr
//!     scratch relocated `%r9` → `%r15`. Same op-local push/pop pattern
//!     as LH-2.
//!
//! No remaining latent hazards in this file. Future-proofing: any
//! addition of new scratch registers MUST verify against INPUT_REGS at
//! the highest supported N (currently N=4) before merging — see
//! ABI-invariant tests `emit_linear_n{2,3,4}_does_not_clobber_output_reg`
//! in `profiles/x86_64/src/tests.rs`.
//!
//! Cross-reference: same class of bug as M13 Task 1 (`emit_matmul`
//! `%r9` → `%rbp`) and the M13 pre-Task-5 arm64 emit_linear x3/x4/x5
//! stp/ldp fix. M14 closes the analogous x86_64 cases uniformly.
```

(The exact text of the new doc-comment may be tightened during commit-message authoring — the structure above is the required content.)

- [ ] **Step 16: Update PROJECT_SPEC.md §"Known Latent Hazards" — remove LH-1/2/3 rows**

Edit `PROJECT_SPEC.md` lines around 219-229. Remove these three rows from the table:

```markdown
| LH-1 | profiles/x86_64/src/ops/linear.rs | N=2 + linear with bias | bias-add reads j-counter as base address, wrong output (not SIGSEGV) | M13 |
| LH-2 | profiles/x86_64/src/ops/linear.rs | N=3, src ptr scratch %r8 == output_reg | src reads from wrong address | M13 |
| LH-3 | profiles/x86_64/src/ops/linear.rs | N=4, weight ptr scratch %r9 == output_reg | weight reads from wrong address | M13 |
```

If the table becomes empty (only the header row remains), retain the section heading and add a one-line comment immediately under the header:

```markdown
### Known Latent Hazards

Bugs that exist in the codebase but are not triggered by any current fixture.
Each entry must be resolved in the milestone whose fixture first exercises it.
Leaving an entry here longer than one milestone is a process failure.

*Currently empty — populate as new latent hazards are discovered.*

| # | Location | Condition | Symptom | Opened |
|---|----------|-----------|---------|--------|
```

(Keep the table header for future entries.)

- [ ] **Step 17: Run the full workspace gate — clippy, fmt, test**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All three commands must exit 0. Expected test count: 400 → 403 (three new ABI-invariant tests added; nothing removed).

If `cargo fmt --check` reports drift, run `cargo fmt --all` and re-run the check. If clippy reports warnings introduced by the changes, fix them inline.

- [ ] **Step 18: Commit**

Stage the three changed files and commit with the message below. Verify only the expected files are staged.

```bash
git status --short
# Expected output:
#  M PROJECT_SPEC.md
#  M profiles/x86_64/src/ops/linear.rs
#  M profiles/x86_64/src/tests.rs

git add PROJECT_SPEC.md profiles/x86_64/src/ops/linear.rs profiles/x86_64/src/tests.rs

git commit -m "$(cat <<'EOF'
fix(m14): close LH-1/2/3 in x86_64 emit_linear

Three latent hazards in profiles/x86_64/src/ops/linear.rs (documented
in PROJECT_SPEC §"Known Latent Hazards" since M13) closed uniformly
via ABI-register relocation:

  LH-1 (N=2 + bias): j-counter %rcx → %rbp. M13 Task 1 emit_matmul
  precedent — %rbp callee-saved by unconditional function prologue,
  unread by op bodies. M13 push/pop %rcx save block trimmed (dead
  post-fix).

  LH-2 (N=3 src ptr): %r8 → %r14. Op-local pushq/popq inside
  emit_linear body — compute_callee_saved unchanged, function-level
  prologue unchanged. M13 pre-Task-5 arm64 precedent for op-local
  callee-saved save/restore.

  LH-3 (N=4 weight ptr): %r9 → %r15. Same op-local pattern as LH-2.

All scratch now lives in non-INPUT_REGS scope at all N=1..4. M12 §9.1
invariant ("op body must NOT touch any ABI argument register") restored
uniformly across emit_linear.

Three new ABI-invariant unit tests in profiles/x86_64/src/tests.rs
guard against regression at N=2/3/4 (extends commit c993712 precedent
to complex emitter). Test count: 400 → 403.

Mandatory by §LH process — M14 fixture pre_ln_block.nfl (Plan 2 of
M14) at N=2 + linear-with-bias triggers LH-1 end-to-end. LH-2/3
closed proactively per memory rule "triggered cleanup is an
obligation" and §LH "leaving an entry here longer than one milestone
is a process failure" once the file is touched.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

git status --short
# Expected: clean tree (no output).
```

---

## Self-review checklist (run before declaring Task 1 complete)

- [ ] **Spec coverage:** Walk through spec §3 sub-sections (3.1, 3.2, 3.3, 3.4, 3.5, 3.6) — every requirement implemented or explicitly out-of-scope. Traceability table at top of this plan should match implementation.
- [ ] **No stale references:** grep `LH-1\|LH-2\|LH-3` through the codebase. Should appear only in the doc-comment cross-reference (Step 15) and in this commit's DEVLOG entry (Plan 2 Task 5 will write the DEVLOG entry — opener is referenced from there).
- [ ] **Tests run green:** `cargo test --workspace` exits 0. Three new tests pass. No pre-existing test regressed.
- [ ] **Audit gate clean:** §3.5 audit (Step 14) found zero violations.
- [ ] **Function-level prologue truly unchanged:** confirm no edits in `profiles/x86_64/src/buffer.rs` or `profiles/x86_64/src/asm.rs`.
- [ ] **PROJECT_SPEC table is consistent:** if LH-1/2/3 were the only entries, the §"Known Latent Hazards" header retains "currently empty" comment + table header. If other LH entries existed, only LH-1/2/3 rows removed.

---

## Dependencies & sequencing

- **Predecessor:** none in M14 (this is the opener). Latest dependency is M13 commit `2d84281` (`%rbp` j-counter precedent in `emit_matmul`).
- **Successor:** Plan 2 (M14 LayerNorm feature series). Plan 2's `pre_ln_block.nfl` FFI integration test validates LH-1 fix end-to-end (bit-exact compare against Rust reference at N=2 + linear-with-bias would corrupt pre-fix, green post-fix). Plan 2 must merge after this commit.
