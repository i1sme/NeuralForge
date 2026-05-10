# M15 — A2 Third Brick (FFN) + LH-4 Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close LH-4 latent hazard in `profiles/x86_64/src/ops/layernorm.rs` (relocate `%r8`→`%r15`, `%r9`→`%rbp`) and demonstrate A2 third brick — FFN as compositional NFL pattern (`linear → relu → linear`) — via two new fixtures and four new FFI integration tests.

**Architecture:** Single PR, four sequential commits T0→T1→T2→T3. T0 is the cleanup (asm-shape unit tests). T1 introduces `ffn.nfl` + promotes existing reference helpers to `common/mod.rs`. T2 introduces `transformer_block.nfl` (N=3, output_reg=%r8) which provides runtime FFI evidence for LH-4 closure on x86_64 Linux CI. T3 closes documentation.

**Tech Stack:** Rust 2021, cargo workspace; per-profile codegen crates (`profiles-arm64`, `profiles-x86_64`); FFI integration tests via `cc -shared -fPIC` + `libloading`; `compiler` crate for parse/IR/passes; `nflc` CLI is unaffected.

**Spec reference:** [`docs/superpowers/specs/2026-05-10-m15-ffn-and-lh4-cleanup-design.md`](../specs/2026-05-10-m15-ffn-and-lh4-cleanup-design.md) — read this first; the plan implements it.

**Branch:** `claude/stupefied-zhukovsky-59aaaf` (current worktree).

---

## File structure (all files touched across all tasks)

| Path | Task | Action | Responsibility |
|---|---|---|---|
| `profiles/x86_64/src/ops/layernorm.rs` | T0 | Modify | Register relocation %r8→%r15, %r9→%rbp; push order; constants; doc-comment |
| `profiles/x86_64/src/tests.rs` | T0 | Modify (append) | 3 new unit tests for LH-4 cleanup |
| `tests/fixtures/ffn.nfl` | T1 | Create | N=1 FFN baseline fixture |
| `profiles/arm64/tests/common/mod.rs` | T1 | Modify | Promote `reference_matmul/bias_add/relu`; add `ffn_ref` |
| `profiles/arm64/tests/integration.rs` | T1 | Modify | Remove file-local helpers (now in common); add `ffn_ffi` test |
| `profiles/x86_64/tests/common/mod.rs` | T1 | Modify | Same promotion + `ffn_ref` (separate copy, isolation principle) |
| `profiles/x86_64/tests/integration.rs` | T1 | Modify | Same removal + `ffn_ffi` test |
| `tests/fixtures/transformer_block.nfl` | T2 | Create | N=3 transformer block; LH-4 trigger |
| `profiles/arm64/tests/common/mod.rs` | T2 | Modify | Add `transformer_block_ref` |
| `profiles/arm64/tests/integration.rs` | T2 | Modify | Add `transformer_block_ffi` test |
| `profiles/x86_64/tests/common/mod.rs` | T2 | Modify | Add `transformer_block_ref` (separate copy) |
| `profiles/x86_64/tests/integration.rs` | T2 | Modify | Add `transformer_block_ffi` test |
| `DEVLOG.md` | T3 | Modify (prepend entry) | M15 entry per template; ABI audit paper trail |
| `PROJECT_SPEC.md` | T3 | Modify | §Milestones row 15; §Strategic Roadmap; remove LH-4 row |
| `CLAUDE.md` | T3 | Modify | "Current Status" → M15 |
| `docs/profile_guide/x86_64.md` | T3 | Modify | emit_layernorm register table post-LH-4 |

**Important: NO files in `profiles/x86_64/src/asm.rs`, `compiler/src/`, `nflc/`, `language/`, `bench/`** — out of scope.

---

## Pre-task baseline check

- [ ] **Step 0.1: Verify clean working tree on the M15 branch**

```bash
git status
git log --oneline -3
```

Expected: clean tree; HEAD is `095cfda docs(m15): address spec review` followed by `77e4904 docs(m15): brainstorm spec` (or later if more commits land).

- [ ] **Step 0.2: Verify all gates green pre-M15**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Expected: all four commands exit 0. Test count baseline: 441 on macOS arm64 (~444 on Linux x86_64 CI). If baseline is broken, STOP and fix before proceeding.

---

## Task 0: LH-4 cleanup in `profiles/x86_64/src/ops/layernorm.rs`

**Goal:** Relocate per-row scratch registers `%r8`→`%r15` (op-local pushq/popq) and `%r9`→`%rbp` (function-level prologue handles), update push-bytes constants, update doc-comment numbers, add three ABI-invariant unit tests.

**Files:**
- Modify: `profiles/x86_64/src/ops/layernorm.rs` (whole file edits in body block + constants + doc-comment lines 28–62 and 70–82)
- Modify: `profiles/x86_64/src/tests.rs` (append at end of file, after existing M14 layernorm tests)

**TDD ordering rationale:** Tests fail on current emit_layernorm output (uses %r8/%r9). After register relocation, tests pass. Same TDD pattern as M14 LH-1/2/3 cleanup commit `916e9c7`.

### Step 0.0: Read the design-spec §2 verbatim

Open [docs/superpowers/specs/2026-05-10-m15-ffn-and-lh4-cleanup-design.md §2](../specs/2026-05-10-m15-ffn-and-lh4-cleanup-design.md) and skim the "Push order discipline (load-bearing)" subsection — `%r15` MUST be first push / last pop in both the no-affine and affine paths. Wrong push order = silently wrong asm but unit tests will not catch the LIFO order, only the membership. **Implementer attention required here.**

### Step 0.1: Add three new unit tests to `profiles/x86_64/src/tests.rs`

The existing M14 tests for layernorm only cover N=1 and N=2 (lines 2050–2074). LH-4 condition is N=3 and N=4 — add three new tests mirroring the `emit_linear_n{2,3,4}_does_not_clobber_output_reg` pattern (lines 1902–2020).

Locate the end of the existing emit_layernorm section in `profiles/x86_64/src/tests.rs` (after the `emit_layernorm_x86_64_abi_clean_at_n2_with_affine` test, around line 2074). Append:

```rust
// ---- M15 LH-4 cleanup tests: emit_layernorm at N=2/3/4 -----------------------
//
// Mirrors the emit_linear_n{2,3,4}_does_not_clobber_output_reg pattern
// (M14 commit 916e9c7). Validates LH-4 closure: %r8 and %r9 must NOT
// appear as scratch destinations in the per-row body at N=3 (output_reg=%r8)
// or N=4 (output_reg=%r9). Post-fix expectation: per-row src ptr lives in
// %r15 (op-local pushq/popq); per-row dst ptr lives in %rbp (function-level
// prologue handles).

#[test]
fn emit_layernorm_n2_does_not_clobber_output_reg() {
    // Parametric guard. At N=2, output_reg=%rcx, never aliased by per-row
    // ptrs in any era (passes pre- and post-fix). Kept for coverage parity
    // across the supported N range, mirroring emit_linear pattern.
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs: 2 };
    let asm = emit_layernorm_x86_64_at(2, false);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

#[test]
fn emit_layernorm_n3_does_not_clobber_output_reg() {
    // Primary LH-4 unit test. At N=3, output_reg = %r8 (INPUT_REGS[4]).
    //
    // Pre-fix: emit_layernorm wrote per-row src ptr to %r8 inside the outer
    // row loop:
    //     leaq    (%rbx, %rax, 1), %r8       ← clobbers output_reg
    // and used it as src base in the three Pass loops:
    //     movss   (%r8, %rax, 4), %xmm6
    // Subsequent ops in the same function would see a corrupted output_reg.
    //
    // Post-fix: src ptr scratch relocated to %r15 (callee-saved per SysV;
    // op-local pushq %r15 / popq %r15 inside emit_layernorm body — function-
    // level prologue unchanged). Dst ptr scratch relocated to %rbp (function-
    // level prologue handles).

    let asm = emit_layernorm_x86_64_at(3, false);

    // Pre-fix marker — must NOT appear:
    assert!(
        !asm.contains(", %r8\n"),
        "src ptr scratch must not write to %r8 at N=3 (output_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r8,"),
        "src ptr scratch must not be used as base in indexed load at N=3. Asm:\n{asm}"
    );
    assert!(
        !asm.contains(", %r9\n"),
        "dst ptr scratch must not write to %r9 at N=3 (params_reg alias for next op). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r9,"),
        "dst ptr scratch must not be used as base in indexed store at N=3. Asm:\n{asm}"
    );

    // Post-fix expectations: src ptr in %r15 (op-local push/pop), dst ptr in %rbp.
    assert!(
        asm.contains("    pushq   %r15\n"),
        "op-local pushq %r15 must appear (callee-saved op-local save for src ptr). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r15\n"),
        "op-local popq %r15 must appear (matching restore). Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%r15,"),
        "src ptr should be %r15 (indexed load base). Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%rbp,"),
        "dst ptr should be %rbp (indexed store base). Asm:\n{asm}"
    );

    // Push count check: no-affine path = 3 op-local pushes (%r15, %rbx, %r14).
    let pushq_count = asm.matches("    pushq   ").count();
    let popq_count = asm.matches("    popq    ").count();
    assert!(
        pushq_count >= 3,
        "expected at least 3 op-local pushq in no-affine layernorm body, got {pushq_count}. Asm:\n{asm}"
    );
    assert_eq!(
        pushq_count, popq_count,
        "push/pop count mismatch — LIFO discipline broken. Asm:\n{asm}"
    );
}

#[test]
fn emit_layernorm_n4_does_not_clobber_output_reg() {
    // Secondary LH-4 unit test. At N=4, output_reg = %r9 (INPUT_REGS[5])
    // AND params_reg = %r8 (INPUT_REGS[4]). Both registers are ABI-occupied;
    // pre-fix, layernorm clobbered both. Post-fix, neither appears as
    // scratch destination in body.
    //
    // No N=4 runtime fixture in M15 (transformer_block.nfl is N=3).
    // Asm-shape closure follows M14 LH-2/3 precedent for emit_linear N=4
    // (four_input_matmul.nfl has no linear op, so emit_linear N=4 closure
    // was also asm-only).

    let asm = emit_layernorm_x86_64_at(4, true);  // affine: 5 op-local pushes

    // Pre-fix markers — must NOT appear:
    assert!(
        !asm.contains(", %r8\n"),
        "no scratch may write to %r8 at N=4 (params_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r8,"),
        "no scratch may use %r8 as base at N=4. Asm:\n{asm}"
    );
    assert!(
        !asm.contains(", %r9\n"),
        "no scratch may write to %r9 at N=4 (output_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r9,"),
        "no scratch may use %r9 as base at N=4. Asm:\n{asm}"
    );

    // Post-fix: %r15 + %rbp scratch present in affine path.
    assert!(
        asm.contains("    pushq   %r15\n") && asm.contains("    popq    %r15\n"),
        "op-local pushq/popq %r15 must bracket affine body. Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%r15,") && asm.contains("(%rbp,"),
        "src/dst ptrs should be %r15/%rbp. Asm:\n{asm}"
    );

    // Affine path = 5 op-local pushes (%r15, %r12, %r13, %rbx, %r14).
    let pushq_count = asm.matches("    pushq   ").count();
    let popq_count = asm.matches("    popq    ").count();
    assert!(
        pushq_count >= 5,
        "expected at least 5 op-local pushq in affine layernorm body, got {pushq_count}. Asm:\n{asm}"
    );
    assert_eq!(pushq_count, popq_count, "push/pop LIFO. Asm:\n{asm}");
}
```

### Step 0.2: Run the new tests — verify they FAIL on current emit_layernorm

```bash
cargo test -p profiles-x86_64 --lib emit_layernorm_n2_does_not_clobber_output_reg emit_layernorm_n3_does_not_clobber_output_reg emit_layernorm_n4_does_not_clobber_output_reg
```

Expected: **n2 PASS** (output_reg=%rcx unaffected), **n3 FAIL** (current emit uses %r8 as src ptr scratch — `, %r8` and `(%r8,` substrings present), **n4 FAIL** (same plus %r9 dst ptr).

If n3/n4 PASS unexpectedly, STOP — the test assertions are wrong (false-passing tests are worse than failing). Re-read Step 0.1 and the existing emit_layernorm.rs body to identify the exact substrings the body emits today.

### Step 0.3: Edit `profiles/x86_64/src/ops/layernorm.rs` — register relocation + push order + constants + doc-comment

**Edit 1 — push-bytes constants** (lines 70–82):

Locate:

```rust
const OP_LOCAL_PUSH_BYTES_AFFINE: usize = 4 * 8;
```

Change to:

```rust
const OP_LOCAL_PUSH_BYTES_AFFINE: usize = 5 * 8;
```

Locate:

```rust
const OP_LOCAL_PUSH_BYTES_NO_AFFINE: usize = 2 * 8;
```

Change to:

```rust
const OP_LOCAL_PUSH_BYTES_NO_AFFINE: usize = 3 * 8;
```

**Edit 2 — doc-comment "Register plan" table** (lines 28–47):

Locate the existing block:

```rust
//! Register plan (M14 spec §8, N=1..2 scope, finalized):
//!   %rax  = x_j (inner counter); also temp for byte-offset compute before counter use
//!   %r10  = bound scratch (clobbered every emit_imm32_to_r10 call)
//!   %r11  = x_i (outer counter)
//!   %r8   = x_in  (per-row input ptr;  recomputed per row) — free at N≤2
//!   %r9   = x_out (per-row output ptr; recomputed per row) — free at N≤2
//!   %rbx  = src_base (set ONCE; callee-saved + op-local push/pop)
//!   %r14  = dst_base (set ONCE; callee-saved + op-local push/pop)
//!   %r12  = x_gamma (γ base ptr) — affine only, callee-saved + op-local push/pop
//!   %r13  = x_beta  (β base ptr) — affine only, callee-saved + op-local push/pop
```

Replace the `%r8` and `%r9` lines with:

```rust
//! Register plan (M15 LH-4 closed, N=1..4 scope, finalized):
//!   %rax  = x_j (inner counter); also temp for byte-offset compute before counter use
//!   %r10  = bound scratch (clobbered every emit_imm32_to_r10 call)
//!   %r11  = x_i (outer counter)
//!   %r15  = x_in  (per-row input ptr;  recomputed per row) — callee-saved + op-local push/pop (M15 LH-4 — was %r8 pre-M15)
//!   %rbp  = x_out (per-row output ptr; recomputed per row) — callee-saved + function-level prologue handles (M15 LH-4 — was %r9 pre-M15)
//!   %rbx  = src_base (set ONCE; callee-saved + op-local push/pop)
//!   %r14  = dst_base (set ONCE; callee-saved + op-local push/pop)
//!   %r12  = x_gamma (γ base ptr) — affine only, callee-saved + op-local push/pop
//!   %r13  = x_beta  (β base ptr) — affine only, callee-saved + op-local push/pop
```

Also update the line:

```rust
//! All scratch in non-INPUT_REGS scope at N=1..2 (the M14 fixture range).
//! Higher-N (N=3..4) not validated; spec §8.7 documents the deferral.
```

Replace with:

```rust
//! All scratch in non-INPUT_REGS scope at N=1..4 (M15 closes LH-4 with
//! transformer_block.nfl runtime evidence at N=3; N=4 closure is asm-only,
//! mirroring M14 LH-2/3 precedent for emit_linear N=4).
```

**Edit 3 — doc-comment "Stack alignment invariant" block** (lines 56–62):

Locate:

```rust
//! Stack alignment invariant (M-future foot-gun): always-pushed %rbx +
//! %r14 add +16 bytes (pair preserves alignment), conditional %r12 + %r13
//! add +16 (pair preserves). `pushq %rbp` in function prologue is +8 → odd.
```

Replace with:

```rust
//! Stack alignment invariant (M-future foot-gun): unconditional pushes
//! %r15 + %rbx + %r14 add +24 bytes (3 pushes, no-affine path). Affine
//! path adds %r12 + %r13 → +40 bytes total (5 pushes). Both totals are
//! ≡ 8 mod 16 (odd-by-8). `pushq %rbp` in function prologue is also +8.
//! Inside-body %rsp is therefore NOT 16-byte aligned — OK because
//! emit_layernorm is leaf (native sqrtss, no `call` site in body).
```

**Edit 4 — doc-comment update for `OP_LOCAL_PUSH_BYTES_*` constants** (around lines 70–82):

Locate the doc-comment for `OP_LOCAL_PUSH_BYTES_AFFINE`:

```rust
/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is enabled (4 pushq: %r12, %r13, %rbx, %r14).
```

Replace `4 pushq` with `5 pushq` and update register list:

```rust
/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is enabled (5 pushq: %r15, %r12, %r13, %rbx, %r14).
```

Locate the doc-comment for `OP_LOCAL_PUSH_BYTES_NO_AFFINE`:

```rust
/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is disabled (2 pushq: %rbx, %r14). Same invariant as the affine const —
```

Replace `2 pushq: %rbx, %r14` with `3 pushq: %r15, %rbx, %r14`:

```rust
/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is disabled (3 pushq: %r15, %rbx, %r14). Same invariant as the affine const —
```

**Edit 5 — push block** (around lines 167–172):

Locate:

```rust
    if has_affine {
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
    }
    s.push_str("    pushq   %rbx\n");
    s.push_str("    pushq   %r14\n");
```

Replace with (note: `%r15` MUST be first push):

```rust
    s.push_str("    pushq   %r15\n");
    if has_affine {
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
    }
    s.push_str("    pushq   %rbx\n");
    s.push_str("    pushq   %r14\n");
```

**Edit 6 — pop block** (around lines 295–301):

Locate:

```rust
    // Op-local restores — strict LIFO of the entry pushes.
    s.push_str("    popq    %r14\n");
    s.push_str("    popq    %rbx\n");
    if has_affine {
        s.push_str("    popq    %r13\n");
        s.push_str("    popq    %r12\n");
    }
```

Replace with (note: `%r15` MUST be last pop):

```rust
    // Op-local restores — strict LIFO of the entry pushes.
    s.push_str("    popq    %r14\n");
    s.push_str("    popq    %rbx\n");
    if has_affine {
        s.push_str("    popq    %r13\n");
        s.push_str("    popq    %r12\n");
    }
    s.push_str("    popq    %r15\n");
```

**Edit 7 — body register relocation** (around lines 225–232 and inside the three Pass loops):

Locate this inner block (around lines 225–232):

```rust
    // Compute per-row pointers: %r8 = src_base + i*d*4; %r9 = dst_base + i*d*4.
    // Use %rax as transient byte-offset accumulator (then re-zeroed for Pass 1
    // counter use below — emit_imm32_to_r10 doesn't touch %rax).
    s.push_str(&emit_imm32_to_r10((d * 4) as u32));
    s.push_str("    movq    %r11, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * d * 4 (byte offset)
    s.push_str("    leaq    (%rbx, %rax, 1), %r8\n");
    s.push_str("    leaq    (%r14, %rax, 1), %r9\n");
```

Replace with:

```rust
    // Compute per-row pointers: %r15 = src_base + i*d*4; %rbp = dst_base + i*d*4.
    // Use %rax as transient byte-offset accumulator (then re-zeroed for Pass 1
    // counter use below — emit_imm32_to_r10 doesn't touch %rax).
    //
    // M15 LH-4: per-row scratch was %r8/%r9 pre-M15; relocated to %r15 (op-local
    // pushq/popq) and %rbp (function-level prologue handles) to avoid clobbering
    // output_reg at N=3 (=%r8) and params_reg/output_reg at N=4 (=%r8/%r9).
    s.push_str(&emit_imm32_to_r10((d * 4) as u32));
    s.push_str("    movq    %r11, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * d * 4 (byte offset)
    s.push_str("    leaq    (%rbx, %rax, 1), %r15\n");
    s.push_str("    leaq    (%r14, %rax, 1), %rbp\n");
```

**Edit 8 — Pass 1 inner load** (around line 241):

Locate `s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");` inside `.Lln_p1_{lid}:`. Replace `%r8` with `%r15`:

```rust
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
```

**Edit 9 — Pass 2 inner load** (around line 256):

Locate `s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");` inside `.Lln_p2_{lid}:`. Replace `%r8` with `%r15`:

```rust
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
```

**Edit 10 — Pass 3 inner load + store** (around lines 277, 286):

Locate inside `.Lln_p3_{lid}:`:

```rust
    s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");
```

Replace with:

```rust
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
```

And the store line:

```rust
    s.push_str("    movss   %xmm6, (%r9, %rax, 4)\n");
```

Replace with:

```rust
    s.push_str("    movss   %xmm6, (%rbp, %rax, 4)\n");
```

**Verification grep** (after all edits):

```bash
grep -nE "%r8|%r9" profiles/x86_64/src/ops/layernorm.rs
```

Expected: only doc-comment mentions of %r8/%r9 (in "M15 LH-4: per-row scratch was %r8/%r9 pre-M15" type comments), NO actual code emissions of `%r8` or `%r9`. Specifically, no `s.push_str(...%r8...)` or `s.push_str(...%r9...)` lines remain in the function body.

### Step 0.4: Run the three new unit tests — verify they now PASS

```bash
cargo test -p profiles-x86_64 --lib emit_layernorm_n2_does_not_clobber_output_reg emit_layernorm_n3_does_not_clobber_output_reg emit_layernorm_n4_does_not_clobber_output_reg
```

Expected: all three PASS.

### Step 0.5: Run all existing emit_layernorm tests — verify none regressed

```bash
cargo test -p profiles-x86_64 --lib emit_layernorm
```

Expected: all (existing M14 tests + 3 new M15 tests) PASS.

### Step 0.6: Run full workspace tests + lint + fmt gates

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Expected: all four exit 0; test count is +3 over baseline (441 → 444 on macOS arm64).

If `cargo fmt --check` fails, run `cargo fmt --all` and re-run check. If clippy complains about the new `match` in tests or new doc-comment, address inline (these are usually one-character lints).

### Step 0.7: Commit T0

```bash
git add profiles/x86_64/src/ops/layernorm.rs profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
fix(m15): close LH-4 — relocate %r8/%r9 in x86_64 emit_layernorm

Per-row src ptr scratch %r8 → %r15 (callee-saved, op-local pushq/popq —
%r15 first push / last pop, mirroring LH-2/3 pattern in emit_linear).
Per-row dst ptr scratch %r9 → %rbp (callee-saved, function-level prologue
already pushes %rbp — body free without op-local push, mirroring LH-1
pattern in emit_linear and M13 emit_matmul %rbp j-counter relocation).

Push counts: no-affine 2 → 3 (+%r15), affine 4 → 5 (+%r15). Both totals
remain odd → inside-body %rsp not 16-byte aligned. OK because
emit_layernorm is leaf (native sqrtss, no `call` site).

OP_LOCAL_PUSH_BYTES_NO_AFFINE: 2*8 → 3*8.
OP_LOCAL_PUSH_BYTES_AFFINE: 4*8 → 5*8.
materialise_ptr_with_rsp_bias debug_assert continues to enforce.

3 new ABI-invariant unit tests in profiles/x86_64/src/tests.rs:
- emit_layernorm_n2_does_not_clobber_output_reg (parametric guard)
- emit_layernorm_n3_does_not_clobber_output_reg (primary LH-4 test)
- emit_layernorm_n4_does_not_clobber_output_reg (asm-only N=4 closure)

Runtime FFI evidence for LH-4 lands in T2 (transformer_block.nfl at N=3,
output_reg=%r8). N=4 closure is asm-only — no current N=4 fixture invokes
emit_layernorm, mirroring M14 LH-2/3 precedent for emit_linear N=4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Verify commit lands:

```bash
git log --oneline -3
```

Expected: top commit message starts with `fix(m15): close LH-4 — relocate %r8/%r9`.

---

## Task 1: `ffn.nfl` + helper promotion + `ffn_ffi` tests

**Goal:** Create N=1 baseline FFN fixture; promote `reference_matmul`, `reference_bias_add`, **and** `reference_relu` from `integration.rs` file-local to `common/mod.rs` `pub fn` (per profile, separate copies — isolation principle); add `ffn_ref` composing the promoted helpers; add `ffn_ffi` integration test on both profiles.

**Files (all new content this task):**
- Create: `tests/fixtures/ffn.nfl`
- Modify: `profiles/arm64/tests/common/mod.rs` (add 4 pub fn)
- Modify: `profiles/arm64/tests/integration.rs` (remove 3 file-local fn; update use; add 1 test)
- Modify: `profiles/x86_64/tests/common/mod.rs` (add 4 pub fn — separate copies)
- Modify: `profiles/x86_64/tests/integration.rs` (remove 3 file-local fn; update use; add 1 test)

**TDD ordering rationale:** `ffn_ffi` test needs the fixture file to exist and `ffn_ref` helper to compile. Order: fixture → helpers → test.

### Step 1.1: Create `tests/fixtures/ffn.nfl`

```bash
ls tests/fixtures/ffn.nfl 2>/dev/null && echo "EXISTS — STOP" || echo "ok to create"
```

Expected: `ok to create`. If file already exists, STOP and investigate (concurrent work).

Create with content:

```
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

### Step 1.2: Verify fixture parses + lowers (smoke check, no commit yet)

```bash
cargo run -p nflc -- parse tests/fixtures/ffn.nfl --uir
```

Expected: prints UIR with two Linear nodes + one Relu node, total params = 76 floats. If parse fails, the fixture is malformed — fix per spec §3.1 verbatim.

```bash
cargo run -p nflc -- compile tests/fixtures/ffn.nfl --profile arm64 --output /tmp/ffn-arm64.s
```

Expected: emits assembly to `/tmp/ffn-arm64.s` without error. (x86_64 lower works the same on either OS — both lowerers are pure Rust; FFI test gating is host-only.)

### Step 1.3: Promote `reference_matmul`, `reference_bias_add`, `reference_relu` to `profiles/arm64/tests/common/mod.rs`

Open `profiles/arm64/tests/common/mod.rs`. After the existing `pub fn layernorm_ref(...)` (ends around line 98), append:

```rust
/// Reference matmul — naive `b × k` @ `k × n` → `b × n` with `f32::mul_add`
/// reduction. Promoted from integration.rs file-local in M15 to enable
/// reuse from `ffn_ref` and `transformer_block_ref`.
///
/// Reduction order is sequential left-to-right (`mul_add` accumulator) —
/// matches the emitter's scalar fmadd loop bit-exactly. Do NOT replace
/// with iterator-based fold under -O3 (auto-vec tree reduction breaks
/// bit-exact equivalence; same constraint as `layernorm_ref` above).
pub fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum = f32::mul_add(input[i * k + kk], weights[kk * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Reference bias add — broadcast `bias[n]` across `b` rows of `acc[b*n]`,
/// in place semantically (returns new vec, doesn't mutate input).
/// Promoted from integration.rs file-local in M15.
pub fn reference_bias_add(acc: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    let b = acc.len() / n;
    let mut out = acc.to_vec();
    for i in 0..b {
        for j in 0..n {
            out[i * n + j] += bias[j];
        }
    }
    out
}

/// Reference relu — element-wise max(x, 0.0). Promoted from
/// integration.rs file-local in M15.
pub fn reference_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x.max(0.0)).collect()
}

/// Reference FFN — composes `reference_matmul` + `reference_bias_add` +
/// `reference_relu` in the order `linear[w1, b1] → relu → linear[w2, b2]`.
///
/// Shapes: input `[batch, dim]` → matmul w1 `[dim, hidden]` → bias b1 → relu
///       → matmul w2 `[hidden, dim]` → bias b2 → output `[batch, dim]`.
///
/// CRITICAL (M15 helper-reuse rule, see design spec §3.4): this function
/// MUST compose the promoted primitives above. Do NOT inline a fresh matmul
/// or bias loop — divergent reduction order produces 1+ ULP mismatches that
/// fail bit-exact comparison and are deeply painful to debug.
pub fn ffn_ref(
    input: &[f32],
    w1: &[f32], b1: &[f32],
    w2: &[f32], b2: &[f32],
    batch: usize, dim: usize, hidden: usize,
) -> Vec<f32> {
    let mm1 = reference_matmul(input, w1, batch, dim, hidden);
    let mm1_b = reference_bias_add(&mm1, b1, hidden);
    let r1 = reference_relu(&mm1_b);
    let mm2 = reference_matmul(&r1, w2, batch, hidden, dim);
    reference_bias_add(&mm2, b2, dim)
}
```

### Step 1.4: Same promotion for `profiles/x86_64/tests/common/mod.rs`

Open `profiles/x86_64/tests/common/mod.rs`. After `pub fn layernorm_ref(...)`, append the **identical** four functions from Step 1.3 (verbatim copy — separate per-profile copy per design principle 3, NOT a shared module).

### Step 1.5: Update `profiles/arm64/tests/integration.rs` — remove file-local helpers, switch to `common::*`

Open `profiles/arm64/tests/integration.rs`. Locate lines 11–38 (the `reference_matmul`, `reference_bias_add`, `reference_relu` definitions). Delete them.

The file already starts with `mod common;` (line 5). Add a `use` statement after `mod common;`:

```rust
mod common;

use common::{reference_matmul, reference_bias_add, reference_relu};
```

(All existing call sites — `reference_matmul(&input, ...)`, `reference_bias_add(...)`, `reference_relu(...)` — work unchanged because of the `use` import; no body edits needed.)

### Step 1.6: Same removal/use for `profiles/x86_64/tests/integration.rs`

Open `profiles/x86_64/tests/integration.rs`. Locate lines 15–42 (same three functions). Delete them. The file already starts with `mod common;` (line 11). Add immediately after:

```rust
mod common;

use common::{reference_matmul, reference_bias_add, reference_relu};
```

### Step 1.7: Run existing integration tests on both profiles — verify no regression from the move

```bash
cargo test -p profiles-arm64 --test integration
```

Expected: all existing arm64 integration tests still pass (the move is invariance-preserving — same function bodies, just different module path).

```bash
cargo test -p profiles-x86_64 --test integration
```

Expected on macOS: tests skipped (`#![cfg(all(target_os = "linux", target_arch = "x86_64"))]` at top of file gates them out — 0 tests run, exit 0). On Linux x86_64: all existing tests pass.

If a test fails post-move, it likely means a call site references something other than the three promoted helpers — re-grep for `reference_matmul\|reference_bias_add\|reference_relu` in the file and confirm all references resolve via the `use` import.

### Step 1.8: Add `ffn_ffi` test in `profiles/arm64/tests/integration.rs`

Append at end of file:

```rust
// ─── M15 FFN integration tests ───────────────────────────────────────────────

#[test]
fn ffn_ffi() {
    // M15 A2 third brick — FFN as compositional NFL pattern.
    //
    // Fixture: tests/fixtures/ffn.nfl (N=1, dim=4, hidden=8).
    // Pipeline: x -> linear[hidden, bias=true] -> relu -> linear[dim, bias=true].
    // Expected: bit-exact match against common::ffn_ref.

    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/ffn.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Ffn");
    // params: w1 (4*8=32) + b1 (8) + w2 (8*4=32) + b2 (4) = 76 floats.
    assert_eq!(sig.params_floats, 76);

    let dylib_path = common::compile_to_dylib(&asm.source, "ffn");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Ffn").unwrap() };

    // batch=2, dim=4, hidden=8.
    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 38.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let w1 = &params[0..32];
    let b1 = &params[32..40];
    let w2 = &params[40..72];
    let b2 = &params[72..76];

    let expected = common::ffn_ref(&input, w1, b1, w2, b2, 2, 4, 8);

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "ffn[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

### Step 1.9: Add `ffn_ffi` test in `profiles/x86_64/tests/integration.rs`

Append at end of file (the file is already `#![cfg(all(target_os = "linux", target_arch = "x86_64"))]`-gated, so no per-test cfg needed):

```rust
// ─── M15 FFN integration tests ───────────────────────────────────────────────

#[test]
fn ffn_ffi() {
    // M15 A2 third brick — FFN as compositional NFL pattern (x86_64).
    // Linux x86_64 only; macOS skipped via file-level cfg.
    //
    // Fixture: tests/fixtures/ffn.nfl (N=1, dim=4, hidden=8).
    // Expected: bit-exact match against common::ffn_ref.

    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/ffn.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Ffn");
    assert_eq!(sig.params_floats, 76);

    let so_path = common::compile_to_so(&asm.source, "ffn");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Ffn").unwrap() };

    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 38.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let w1 = &params[0..32];
    let b1 = &params[32..40];
    let w2 = &params[40..72];
    let b2 = &params[72..76];

    let expected = common::ffn_ref(&input, w1, b1, w2, b2, 2, 4, 8);

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "ffn[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

### Step 1.10: Run new `ffn_ffi` test on arm64 (macOS dev)

```bash
cargo test -p profiles-arm64 --test integration ffn_ffi
```

Expected: PASS. If FAIL, inspect output:
- `nfl_forward_Ffn not found` → fixture model name mismatch (must be `Ffn` per `model Ffn` in fixture).
- `params_floats == 76` assertion fails → param blob layout mismatch; recheck fixture vs spec §3.1 table.
- `(a - b).abs() < 1e-3` fails → either `ffn_ref` composition wrong (re-check Step 1.3) or fixture lowering bug. Print first few `output[i]` and `expected[i]` to compare.

### Step 1.11: Run all gates pre-commit

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Expected: all green. Test count: macOS arm64 +1 (`ffn_ffi`) over T0 baseline.

(x86_64 `ffn_ffi` will pass on Linux CI; not validated locally on macOS by design.)

### Step 1.12: Commit T1

```bash
git add tests/fixtures/ffn.nfl \
        profiles/arm64/tests/common/mod.rs \
        profiles/arm64/tests/integration.rs \
        profiles/x86_64/tests/common/mod.rs \
        profiles/x86_64/tests/integration.rs

git commit -m "$(cat <<'EOF'
feat(m15): A2 third brick — FFN as compositional NFL pattern

New fixture tests/fixtures/ffn.nfl (N=1, dim=4, hidden=8): pure NFL
composition `linear → relu → linear` with bias on both linears. Demonstrates
that FFN requires NO new StdOp / IR / codegen pattern — both linear and
relu emitters already exist on both profiles since M3 (arm64) and M9 (x86_64).

Helper promotion (per profile, separate copies per design principle 3 —
isolation, no cross-profile sharing): `reference_matmul`, `reference_bias_add`,
`reference_relu` moved from integration.rs file-local to common/mod.rs as
`pub fn`. Verbatim signatures (clean, generic on shapes — no test-specific
quirks). New `pub fn ffn_ref` composes the promoted primitives — must NOT
reimplement matmul/bias/relu (helper-reuse rule, design spec §3.4).

Two new FFI integration tests (ffn_ffi on arm64 + x86_64). x86_64 test is
gated by file-level #![cfg(all(target_os = "linux", target_arch = "x86_64"))]
following M9 precedent.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

git log --oneline -3
```

Expected: top commit message starts with `feat(m15): A2 third brick — FFN`.

---

## Task 2: `transformer_block.nfl` + `transformer_block_ref` + `transformer_block_ffi` tests

**Goal:** Create N=3 transformer block fixture (LayerNorm + FFN + dual residual). Add `transformer_block_ref` composing `layernorm_ref` (M14) + `ffn_ref` (T1) + element-wise add. Add `transformer_block_ffi` integration test on both profiles. **The x86_64 test is the runtime FFI evidence for LH-4 closure.**

**Files:**
- Create: `tests/fixtures/transformer_block.nfl`
- Modify: `profiles/arm64/tests/common/mod.rs` (add 1 pub fn)
- Modify: `profiles/arm64/tests/integration.rs` (add 1 test)
- Modify: `profiles/x86_64/tests/common/mod.rs` (add 1 pub fn — separate copy)
- Modify: `profiles/x86_64/tests/integration.rs` (add 1 test)

### Step 2.1: Create `tests/fixtures/transformer_block.nfl`

```bash
ls tests/fixtures/transformer_block.nfl 2>/dev/null && echo "EXISTS — STOP" || echo "ok to create"
```

Expected: `ok to create`.

Content:

```
# transformer_block.nfl — N=3 LH-4 runtime evidence + A2 transformer-block showcase.
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
#
# Param blob (compute_offsets traversal order):
#   LayerNormScale (4 floats), LayerNormBias (4 floats),
#   LinearWeight (4*8=32), LinearBias (8),
#   LinearWeight (8*4=32), LinearBias (4)
#   Total: 84 floats.

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

### Step 2.2: Smoke check parse + lower on both profiles

```bash
cargo run -p nflc -- parse tests/fixtures/transformer_block.nfl --uir
cargo run -p nflc -- compile tests/fixtures/transformer_block.nfl --profile arm64 --output /tmp/tb-arm64.s
cargo run -p nflc -- compile tests/fixtures/transformer_block.nfl --profile x86_64 --output /tmp/tb-x86_64.s
```

Expected: all three exit 0; `params_floats` (in --uir output) = 84.

**LH-4 verification** — grep generated x86_64 asm for `%r8`/`%r9` clobbers in layernorm body:

```bash
grep -nE "leaq.*%r8\b|leaq.*%r9\b|, %r8$|, %r9$" /tmp/tb-x86_64.s
```

Expected: NO matches in the layernorm subsection (Lln_* labels). If any match appears in the layernorm body, T0 was incomplete — go back and re-verify Step 0.3 edits.

### Step 2.3: Add `transformer_block_ref` to `profiles/arm64/tests/common/mod.rs`

Append after `ffn_ref` (added in Step 1.3):

```rust
/// Reference transformer block — composes `layernorm_ref` + `ffn_ref` +
/// element-wise add. Mirrors the `transformer_block.nfl` fixture pipeline:
/// `x -> layernorm[affine] -> linear -> relu -> linear -> add[skip1] -> add[skip2]`.
///
/// CRITICAL (helper-reuse rule, design spec §3.4): this function MUST compose
/// `layernorm_ref` (M14, above) and `ffn_ref` (M15, above). Do NOT reimplement
/// LayerNorm normalization, matmul reduction, or bias add. The existing
/// helpers are M14-verified bit-exact against emitters; reuse them as-is.
pub fn transformer_block_ref(
    input: &[f32], skip1: &[f32], skip2: &[f32],
    gamma: &[f32], beta: &[f32],
    w1: &[f32], b1: &[f32],
    w2: &[f32], b2: &[f32],
    batch: usize, dim: usize, hidden: usize,
) -> Vec<f32> {
    // 1. layernorm[affine=true]
    let ln = layernorm_ref(input, &[batch, dim], Some(gamma), Some(beta));
    // 2. ffn (linear → relu → linear with bias on both)
    let ffn_out = ffn_ref(&ln, w1, b1, w2, b2, batch, dim, hidden);
    // 3. add[skip1] (element-wise)
    let r1: Vec<f32> = ffn_out
        .iter()
        .zip(skip1.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    // 4. add[skip2] (element-wise)
    r1.iter()
        .zip(skip2.iter())
        .map(|(&a, &b)| a + b)
        .collect()
}
```

### Step 2.4: Same `transformer_block_ref` in `profiles/x86_64/tests/common/mod.rs`

Append the **identical** function from Step 2.3 to `profiles/x86_64/tests/common/mod.rs` (verbatim copy — separate per-profile copy, isolation principle).

### Step 2.5: Add `transformer_block_ffi` test in `profiles/arm64/tests/integration.rs`

Append at end of file:

```rust
#[test]
fn transformer_block_ffi() {
    // M15 — N=3 transformer block (LayerNorm + FFN + dual residual).
    //
    // arm64 path: AAPCS64-clean by construction (M14 emit_layernorm uses
    // x6/x9-x17 + s0-s7 scratch, no overlap with x0-x4 inputs at N=3).
    // This test is the implicit ABI audit for arm64 emit_layernorm at N=3.

    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/transformer_block.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_TransformerBlock");
    // params: γ(4) + β(4) + w1(4*8=32) + b1(8) + w2(8*4=32) + b2(4) = 84.
    assert_eq!(sig.params_floats, 84);

    let dylib_path = common::compile_to_dylib(&asm.source, "transformer_block");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_TransformerBlock").unwrap() };

    // batch=2, dim=4 — three N=3 input tensors of [2,4] each.
    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut skip1 = vec![0.0f32; 2 * 4];
    for (i, v) in skip1.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.2;
    }
    let mut skip2 = vec![0.0f32; 2 * 4];
    for (i, v) in skip2.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 + 0.1;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 42.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(
            input.as_ptr(),
            skip1.as_ptr(),
            skip2.as_ptr(),
            params.as_ptr(),
            output.as_mut_ptr(),
        );
    }

    // Param blob slicing per compute_offsets traversal order.
    let gamma = &params[0..4];
    let beta = &params[4..8];
    let w1 = &params[8..40];
    let b1 = &params[40..48];
    let w2 = &params[48..80];
    let b2 = &params[80..84];

    let expected = common::transformer_block_ref(
        &input, &skip1, &skip2, gamma, beta, w1, b1, w2, b2, 2, 4, 8,
    );

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "transformer_block[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

### Step 2.6: Add `transformer_block_ffi` test in `profiles/x86_64/tests/integration.rs`

Append at end of file:

```rust
#[test]
fn transformer_block_ffi() {
    // M15 — N=3 transformer block (LayerNorm + FFN + dual residual).
    //
    // x86_64 / Linux only (file-level #![cfg]). THE LH-4 RUNTIME EVIDENCE
    // TEST. Pre-T0 (without LH-4 fix): layernorm body clobbers output_reg=%r8
    // → segfault or wrong-output bit-mismatch vs Rust reference. Post-T0:
    // %r8 untouched (relocated to %r15), bit-exact match.

    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/transformer_block.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_TransformerBlock");
    assert_eq!(sig.params_floats, 84);

    let so_path = common::compile_to_so(&asm.source, "transformer_block");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_TransformerBlock").unwrap() };

    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut skip1 = vec![0.0f32; 2 * 4];
    for (i, v) in skip1.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.2;
    }
    let mut skip2 = vec![0.0f32; 2 * 4];
    for (i, v) in skip2.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 + 0.1;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 42.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(
            input.as_ptr(),
            skip1.as_ptr(),
            skip2.as_ptr(),
            params.as_ptr(),
            output.as_mut_ptr(),
        );
    }

    let gamma = &params[0..4];
    let beta = &params[4..8];
    let w1 = &params[8..40];
    let b1 = &params[40..48];
    let w2 = &params[48..80];
    let b2 = &params[80..84];

    let expected = common::transformer_block_ref(
        &input, &skip1, &skip2, gamma, beta, w1, b1, w2, b2, 2, 4, 8,
    );

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "transformer_block[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

### Step 2.7: Run new `transformer_block_ffi` test on arm64

```bash
cargo test -p profiles-arm64 --test integration transformer_block_ffi
```

Expected: PASS. If FAIL:
- `nfl_forward_TransformerBlock not found` → fixture model name mismatch.
- `params_floats == 84` fails → param blob layout drift; recheck fixture vs spec §3.2.
- Bit-mismatch → if FAILS only on x86_64 path (above), suspect LH-4 incomplete. If FAILS on arm64, suspect arm64 emit_layernorm has an unrelated bug at N=3 (was previously only N=1/N=2 validated). Print first few `output[i]` and `expected[i]` to compare; cross-check with the M14 arm64 layernorm scratch register table.

### Step 2.8: Run all gates pre-commit

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Expected: all green. macOS arm64 test count: +1 over T1 (`transformer_block_ffi`).

### Step 2.9: Commit T2

```bash
git add tests/fixtures/transformer_block.nfl \
        profiles/arm64/tests/common/mod.rs \
        profiles/arm64/tests/integration.rs \
        profiles/x86_64/tests/common/mod.rs \
        profiles/x86_64/tests/integration.rs

git commit -m "$(cat <<'EOF'
feat(m15): transformer_block fixture — LH-4 runtime evidence + A2 showcase

N=3 transformer block fixture (tests/fixtures/transformer_block.nfl):
LayerNorm[affine=true] → linear[hidden, bias=true] → relu → linear[dim, bias=true]
→ add[skip1] → add[skip2]. Three model inputs (x, skip1, skip2) push N to 3,
making output_reg = %r8 on x86_64 — the exact LH-4 trigger condition.

`transformer_block_ref` composes layernorm_ref (M14) + ffn_ref (T1) +
inline element-wise add. Helper-reuse rule (design spec §3.4) — no
reimplementation of layernorm normalization or matmul reduction.

x86_64 transformer_block_ffi on Linux CI is the runtime FFI evidence for
LH-4 closure (memory `feedback_runtime_evidence_for_codegen.md` requirement).

Bisectability: reverting only T0 (the LH-4 cleanup commit) on any tip-state
after T2 lands → x86_64 transformer_block_ffi fails on Linux CI with silent
corruption (output_reg=%r8 clobbered by per-row src ptr in emit_layernorm).
T0 without T2 = closure by inspection only (asm-shape unit tests).
T2 without T0 = runtime crash. T0+T2 together = LH-4 closed with runtime
evidence.

arm64 transformer_block_ffi is the implicit ABI audit for arm64
emit_layernorm at N=3 (passes — AAPCS64 scratch x6/x9-x17 + s0-s7 has
no overlap with x0-x4 inputs at N=3).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

git log --oneline -3
```

Expected: top message starts with `feat(m15): transformer_block fixture — LH-4 runtime evidence`.

---

## Task 3: Documentation closure

**Goal:** Update `DEVLOG.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, and `docs/profile_guide/x86_64.md` per design spec §4 T3.

**Files:**
- Modify: `DEVLOG.md` (prepend M15 entry)
- Modify: `PROJECT_SPEC.md` (§Milestones row 15; §Strategic Roadmap; remove LH-4 row)
- Modify: `CLAUDE.md` ("Current Status" section update)
- Modify: `docs/profile_guide/x86_64.md` (emit_layernorm register table)

### Step 3.1: Prepend M15 entry to `DEVLOG.md`

Open `DEVLOG.md`. Locate the line `## 2026-05-10 — Milestone 14 closed: A2 second brick — LayerNorm + LH-1/2/3 cleanup + LH-4 entry` (around line 17). Insert the following M15 entry **before** that line, immediately after the `---` separator (around line 16):

```markdown
## 2026-05-10 — Milestone 15 closed: A2 third brick — FFN compositional + LH-4 cleanup

### What was done

- **T0 — LH-4 cleanup in x86_64 emit_layernorm** (commit `<T0_SHA>`): per-row
  src ptr scratch `%r8` → `%r15` (callee-saved, op-local pushq/popq — `%r15`
  first push / last pop, mirroring LH-2/3 pattern in emit_linear). Per-row
  dst ptr scratch `%r9` → `%rbp` (callee-saved, function-level prologue
  already pushes `%rbp` — body free without op-local push, mirroring LH-1
  pattern in emit_linear and M13 emit_matmul `%rbp` j-counter relocation).
  Push counts: no-affine 2 → 3 (+%r15), affine 4 → 5 (+%r15).
  `OP_LOCAL_PUSH_BYTES_NO_AFFINE`: 2*8 → 3*8.
  `OP_LOCAL_PUSH_BYTES_AFFINE`: 4*8 → 5*8.
  3 new ABI-invariant unit tests in `profiles/x86_64/src/tests.rs`:
  `emit_layernorm_n{2,3,4}_does_not_clobber_output_reg`.

- **T1 — A2 third brick — FFN compositional fixture** (commit `<T1_SHA>`):
  new `tests/fixtures/ffn.nfl` (N=1, dim=4, hidden=8). Pure NFL composition
  `linear → relu → linear`. Helper promotion: `reference_matmul`,
  `reference_bias_add`, `reference_relu` moved from `integration.rs`
  file-local to `common/mod.rs` `pub fn` (per profile, separate copies —
  isolation principle). New `pub fn ffn_ref` composes the promoted primitives.
  2 new FFI integration tests (`ffn_ffi` on arm64 + x86_64).

- **T2 — transformer_block fixture — LH-4 runtime evidence + A2 showcase**
  (commit `<T2_SHA>`): new `tests/fixtures/transformer_block.nfl` (N=3,
  output_reg=%r8 — exact LH-4 trigger). Pipeline: `layernorm[affine=true]
  → linear → relu → linear → add[skip1] → add[skip2]`. New
  `transformer_block_ref` composes `layernorm_ref` (M14) + `ffn_ref` (T1)
  + inline element-wise add. 2 new FFI tests (`transformer_block_ffi`).
  x86_64 test on Linux CI is the runtime FFI evidence for LH-4 closure.

- **T3 — documentation closure** (this commit): DEVLOG, PROJECT_SPEC
  (§Milestones row 15, §Strategic Roadmap update, LH-4 row removed),
  CLAUDE.md "Current Status", `docs/profile_guide/x86_64.md` register table.

- **Final test count: TODO_FILL_IN** (macOS arm64); ~TODO_FILL_IN on Linux x86_64 CI.

### Decisions made

- **Register relocation `%r8`→`%r15`, `%r9`→`%rbp`** for per-row layernorm
  scratch. `%r15` requires op-local pushq/popq (not in `compute_callee_saved`);
  `%rbp` reuses function-level prologue's unconditional push. Choices follow
  M14 LH-1/2/3 precedents in emit_linear (`%rbp` for LH-1, `%r14`/`%r15`
  op-local for LH-2/3).

- **Unified push strategy, not conditional.** `%r15` push is unconditional
  per emit_layernorm invocation (body always references `%r15`). Conditional
  alternative would require two body code paths; rejected as YAGNI. Push
  count is now 3 (no-affine) / 5 (affine).

- **Helper promotion applies to all three primitives** (matmul, bias_add,
  relu — not just two as design spec said). `reference_relu` was already
  file-local in both integration.rs files; promotion includes it for full
  reuse in `ffn_ref`.

- **FFN as compositional pattern, no new StdOp.** Per spec §"Strategic
  Roadmap" Axis 2: "compositional op, no new codegen pattern". Confirmed —
  both `linear` and `relu` already exist on both profiles. No IR changes.

- **`transformer_block_ref` reuses `layernorm_ref` + `ffn_ref` + inline
  add.** Helper-reuse rule (design spec §3.4) prevents numerical divergence
  between reference and emitter.

- **Single PR, 4 commits T0→T1→T2→T3** (not M14-style 2-PR split). M15
  scope materially smaller than M14 (no IR foundation, no per-profile
  codegen of a new StdOp, just register relocation + 2 fixtures + helper
  promotion). Cleanup and feature form one coherent narrative.

### Problems encountered

TODO_FILL_IN_DURING_EXEC — placeholder. If implementation surfaces problems,
document specifically. If clean run, replace with "None — TDD ordering
caught all issues at the unit-test stage; integration tests passed first try."

### ABI audit record (mandatory paper trail per design spec §5)

x86_64 emitters reviewed at N=3 and N=4 (M15 expanded arity via
`transformer_block.nfl`):

- `emit_layernorm` — **LH-4 closed in T0**. Asm-shape verified at N=2/3/4 by
  3 new unit tests; runtime FFI evidence at N=3 from T2 `transformer_block_ffi`
  on x86_64 Linux CI. N=4 closure asm-only (no current N=4 fixture invokes
  layernorm; mirroring M14 LH-2/3 precedent).
- `emit_linear` — clean. LH-1/2/3 closed in M14 commit `916e9c7`; ABI-invariant
  tests `emit_linear_n{2,3,4}_does_not_clobber_output_reg` continue to pass.
- `emit_matmul` — clean. M13 Task 1 closed N=4 hazard (`%r9` → `%rbp` j-counter).
- **`emit_relu` at N=3 — reviewed, clean.** Empirically validated by T2
  `transformer_block_ffi` (relu invocation between two linears, output_reg=%r8).
- **`emit_add` at N=3 — reviewed, clean.** Empirically validated by T2 (two
  `add` invocations after the FFN body, output_reg=%r8 across both).
- `emit_mulscalar` — clean. Single scratch register, no ABI overlap.
- `emit_softmax` — clean. M10 spill of `%rdi`/`%rsi`/`%rdx` around
  `call expf@PLT`.
- `emit_dropout` — clean. Pass-through marker, no scratch.

No new latent hazards surfaced. M15 closes the §"Known Latent Hazards"
table with no new entries.

### Next step

Choose next axis:
- **A3 — profile-level viewer annotations** (per-node footprint, stack frame,
  callee-saved set). Continues Axis 2 lineage.
- **Axis 3 — bare-metal `expf`**. Replace `bl _expf` / `call expf@PLT` with
  Taylor-series `expf` to remove libm dependency. Unlocks bare-metal targets.
- **A2 follow-up: training syntax (loss/optimiser)**. Requires NFL v0.3 —
  larger language milestone.

---
```

**Note:** the `<T0_SHA>`, `<T1_SHA>`, `<T2_SHA>` placeholders MUST be filled in with actual SHAs from `git log --oneline -5` before committing. Same for `TODO_FILL_IN` test counts. Run:

```bash
git log --oneline -5
```

Capture top-3 SHAs (T0, T1, T2 — T3 is the commit being made), and the test counts from `cargo test --workspace 2>&1 | grep "test result:" | tail -3`.

### Step 3.2: Update `PROJECT_SPEC.md`

**Edit 1 — §"Milestones to date" — add row 15.**

Locate the M14 row (line ~167 — starts with `| 14 | A2 second brick — LayerNorm`). Insert immediately after it (before the `---` separator that ends the table):

```
| 15 | A2 third brick — FFN compositional + LH-4 cleanup (complete) | LH-4 cleanup in x86_64 `emit_layernorm` (commit `<T0_SHA>`): per-row src ptr `%r8` → `%r15` (op-local pushq/popq); per-row dst ptr `%r9` → `%rbp` (function-level prologue handles). Push counts no-affine 2→3, affine 4→5. `OP_LOCAL_PUSH_BYTES_*` constants updated. 3 ABI-invariant unit tests `emit_layernorm_n{2,3,4}_does_not_clobber_output_reg`. A2 third brick: FFN as compositional NFL pattern (`linear → relu → linear`) — no new StdOp variant, no codegen changes. New fixtures `ffn.nfl` (N=1) and `transformer_block.nfl` (N=3 — exercises LH-4 condition output_reg=%r8 and validates closure via FFI on Linux x86_64 CI). Helper promotion: `reference_matmul/bias_add/relu` moved from `integration.rs` file-local to `common/mod.rs` `pub fn` per profile (isolation principle). 4 new FFI integration tests. ABI audit at N=3,4: all emitters clean. Test count: 441 → TODO_FILL_IN. |
```

**Edit 2 — §"Strategic Roadmap" — update Axis 2 paragraph.**

Locate (around lines 200-211):

```
- **Axis 2 — modelling depth.** M10 closed the first leg (NFL v0.2 self-attention).
  M12 closed A1 (multi-input ABI). M13 closed the M12→M13 priority signal
  (x86_64 N=4 + matmul gap via `%rbp` j-counter relocation) and shipped
  the first A2 brick (`StdOp::Add`, residual connections). M14 closed the
  A2 second brick (`StdOp::LayerNorm`) — single StdOp variant with internal
  3-pass codegen (mean → variance + inv_std → normalize + optional affine),
  mirroring Softmax-as-one-node. Native sqrt (`fsqrt` / `sqrtss`) — no libm
  dependency added. Open follow-ups: A2 FFN (`linear → relu → linear`,
  compositional op, no new codegen pattern, deferred to M15+),
  A3 — profile-level viewer annotations (per-node footprint, stack frame,
  callee-saved set).
```

Replace with:

```
- **Axis 2 — modelling depth.** M10 closed the first leg (NFL v0.2 self-attention).
  M12 closed A1 (multi-input ABI). M13 closed the M12→M13 priority signal
  (x86_64 N=4 + matmul gap via `%rbp` j-counter relocation) and shipped
  the first A2 brick (`StdOp::Add`, residual connections). M14 closed the
  A2 second brick (`StdOp::LayerNorm`) — single StdOp variant with internal
  3-pass codegen (mean → variance + inv_std → normalize + optional affine),
  mirroring Softmax-as-one-node. Native sqrt (`fsqrt` / `sqrtss`) — no libm
  dependency added. **M15 closed the A2 third brick — FFN as compositional
  NFL pattern (`linear → relu → linear`) — no new StdOp variant, no codegen
  changes. Demonstrated via `ffn.nfl` (N=1 baseline) and `transformer_block.nfl`
  (N=3, full transformer block with LayerNorm + FFN + dual residual). M15
  also closed LH-4 in x86_64 `emit_layernorm` (per-row scratch `%r8`/`%r9`
  → `%r15`/`%rbp`).** A2 axis is now complete (residual + LayerNorm + FFN
  all shipped on both profiles). Open follow-ups: A3 — profile-level viewer
  annotations (per-node footprint, stack frame, callee-saved set); A2-extended
  — training syntax (loss/optimiser) for NFL v0.3.
```

**Edit 3 — §"Known Latent Hazards" — remove LH-4 row.**

Locate (around line 232):

```
| LH-4 | profiles/x86_64/src/ops/layernorm.rs | N=3 (output_reg = %r8) or N=4 (output_reg = %r9) | emit_layernorm uses %r8 (src row ptr) and %r9 (dst row ptr) as per-row scratch — clobbers output_reg / input(N-1) at N≥3 | M14 |
```

Delete this entire line. The table is now empty (only header + LH-4 row existed). Replace the table body with a placeholder note immediately after the table header line:

```
| # | Location | Condition | Symptom | Opened |
|---|----------|-----------|---------|--------|
| (empty) | All hazards closed at end of M15. | | | |
```

(Or if you prefer keeping the column structure intact, leave the header rows in place and add a note line after the table:)

```
*(Table is empty as of M15 — all latent hazards closed.)*
```

Pick one. Either is acceptable per design spec §4 T3.

**Edit 4 — §175-179 update.** Locate the paragraph starting `M14 closed the A2 second brick...` (around line 175). Replace the closing two sentences (which mention LH-4 deferral and M15+ FFN) with M15-closed statements:

```
M14 closed the A2 second brick (LayerNorm) end-to-end on both profiles and the LH-1/2/3 latent hazard cleanup in x86_64 `emit_linear` (opener commit `916e9c7`). LayerNorm is a single StdOp variant with internal 3-pass codegen (mean → variance + inv_std → normalize + optional affine), modeled structurally after Softmax. Native `fsqrt`/`sqrtss` — no libm dependency added. Affine optionality via single Symbol toggle `layernorm[affine=true]`, mirroring `linear[bias=true]`. AAPCS64-safe register allocation on arm64 (s8–s15 callee-saved range intentionally avoided; `s_b` reuses `s2` after `s_inv_d` consumption to stay within s0–s7). Op-local `%r12`/`%r13` push/pop on x86_64 affine path — `compute_callee_saved` unchanged. **M15 closed LH-4 (per-row `%r8`/`%r9` scratch in x86_64 `emit_layernorm`) — relocated to `%r15` (op-local pushq/popq) and `%rbp` (function-level prologue handles). Runtime FFI evidence via new `transformer_block.nfl` fixture (N=3, output_reg=%r8) on Linux x86_64 CI.**

Strategic direction: see §"Strategic Roadmap" — A1 closed in M12, A2 first brick (`add`) closed in M13, A2 second brick (`layernorm`) closed in M14, **A2 third brick (FFN) closed in M15 — A2 axis fully complete**. Trigger-driven cleanup items (OQ-7, OQ-8, OQ-9, M5c OQ-4) live in §"Open Questions" / "Trigger-driven cleanup" and stay dormant. OQ-NEW closed in M9 (commit `a08fd24`). OQ-BENCH closed in M11 (commit `e7c29b8`).
```

(This replaces the old final two sentences of the M14 paragraph — the ones mentioning LH-4 deferral and "A2 third brick — FFN remains in M15+".)

### Step 3.3: Update `CLAUDE.md` "Current Status" section

Open `CLAUDE.md`. Locate the "Current Status" section (around lines 145–170 — starts with `## Current Status`). Replace with:

```markdown
## Current Status

**Milestone 15 complete. TODO_FILL_IN tests passing on macOS arm64 (~TODO_FILL_IN on Linux x86_64 CI with x86_64 FFI tests included).** All workspace gates clean
(`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check`, `cargo test --workspace`).

M15 closed the A2 third brick — FFN as compositional NFL pattern
(`linear → relu → linear`, no new StdOp variant, no codegen changes) — and
the LH-4 latent hazard cleanup in x86_64 `emit_layernorm` (per-row scratch
`%r8`/`%r9` → `%r15`/`%rbp`). A2 axis fully complete: residual + LayerNorm
+ FFN all shipped on both profiles. Two new positive fixtures: `ffn.nfl`
(N=1 baseline) and `transformer_block.nfl` (N=3 full transformer block,
runtime FFI evidence for LH-4 closure on Linux x86_64 CI). Helper
promotion: `reference_matmul`/`bias_add`/`relu` moved from `integration.rs`
file-local to `common/mod.rs` `pub fn` per profile.

Strategic direction: see `PROJECT_SPEC.md` §"Strategic Roadmap" — A1 closed
M12, A2 first brick (`add`) closed M13, A2 second brick (`layernorm`)
closed M14, A2 third brick (FFN) closed M15. **A2 axis fully complete.**
Next candidates: A3 — profile-level viewer annotations (per-node footprint,
stack frame, callee-saved set); Axis 3 — bare-metal `expf` to drop libm.
Trigger-driven cleanup (OQ-7, OQ-8, OQ-9, M5c OQ-4) stays dormant. §"Known
Latent Hazards" table empty as of end of M15.
```

(Fill in `TODO_FILL_IN` from `cargo test --workspace 2>&1 | grep "test result:" | tail -3` output.)

### Step 3.4: Update `docs/profile_guide/x86_64.md` — emit_layernorm register table

Open `docs/profile_guide/x86_64.md`. Search for the section about `emit_layernorm` register allocation (likely around the LayerNorm section added in M14 docs).

```bash
grep -n "emit_layernorm\|LayerNorm.*register" docs/profile_guide/x86_64.md
```

Locate the existing register table for `emit_layernorm` (M14 wrote one). It currently shows `%r8` and `%r9` as per-row src/dst ptrs. Replace those two rows with:

```
| Register | Role | Lifetime |
|---|---|---|
| `%rax` | inner counter `x_j` (and transient byte-offset accumulator) | per-row |
| `%r10` | `emit_imm32_to_r10` scratch | per-call |
| `%r11` | outer counter `x_i` | per-function |
| `%r15` | per-row src ptr (op-local pushq/popq, M15 LH-4 — was `%r8` pre-M15) | per-row |
| `%rbp` | per-row dst ptr (function-level prologue handles, M15 LH-4 — was `%r9` pre-M15) | per-row |
| `%rbx` | src_base (set ONCE; callee-saved + op-local push/pop) | per-function |
| `%r14` | dst_base (set ONCE; callee-saved + op-local push/pop) | per-function |
| `%r12` | γ base (affine only; callee-saved + op-local push/pop) | per-function |
| `%r13` | β base (affine only; callee-saved + op-local push/pop) | per-function |
| `%xmm0–8` | scratch FP (see source for detailed plan) | per-pass |
```

Add a short paragraph after the table:

```
**Push counts (M15-current):** no-affine 3 op-local pushes (`%r15`, `%rbx`, `%r14`); affine 5 op-local pushes (`%r15`, `%r12`, `%r13`, `%rbx`, `%r14`). `%r15` is always first push / last pop (load-bearing for `materialise_ptr_with_rsp_bias` push-bytes accounting).
```

(If the existing M14 doc has a "register plan" subsection mentioning the older `%r8`/`%r9` numbers, replace with the above table verbatim.)

### Step 3.5: Run all gates

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace
```

Expected: all green (docs changes don't affect compilation/tests). Capture final test count for filling `TODO_FILL_IN` placeholders in DEVLOG / CLAUDE.md.

### Step 3.6: Fill in placeholders

Re-edit `DEVLOG.md` and `CLAUDE.md`:
- `<T0_SHA>` / `<T1_SHA>` / `<T2_SHA>` → actual SHAs from `git log --oneline -5`
- `TODO_FILL_IN` test counts → from `cargo test --workspace 2>&1 | grep "test result:"` output

Re-run `cargo fmt --all` if needed.

### Step 3.7: Commit T3

```bash
git add DEVLOG.md PROJECT_SPEC.md CLAUDE.md docs/profile_guide/x86_64.md

git commit -m "$(cat <<'EOF'
docs(m15): documentation closure

DEVLOG: M15 entry per template — T0 LH-4 cleanup, T1 FFN fixture +
helper promotion, T2 transformer_block + runtime evidence, T3 docs.
Includes mandatory ABI audit paper trail with explicit per-emitter status
(emit_relu, emit_add at N=3 — reviewed clean per design spec §5).

PROJECT_SPEC: §Milestones row 15 added; §Strategic Roadmap Axis 2 updated
(A2 axis fully complete); §"Known Latent Hazards" — LH-4 row removed,
table now empty; M14 paragraph updated to mention M15 LH-4 closure.

CLAUDE.md "Current Status": M14 → M15; FFN/A2 axis-complete summary;
test count update.

docs/profile_guide/x86_64.md: emit_layernorm register table updated to
reflect post-LH-4 state (%r15/%rbp instead of %r8/%r9, push counts 3/5).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

git log --oneline -6
```

Expected: top message starts with `docs(m15): documentation closure`. Below it: T2, T1, T0, then the two pre-M15 spec commits.

---

## Final verification (after all 4 commits land)

- [ ] **Step 4.1: Confirm 4-commit sequence**

```bash
git log --oneline -6 main..HEAD
```

Expected output (in order, newest first):

```
<sha> docs(m15): documentation closure
<sha> feat(m15): transformer_block fixture — LH-4 runtime evidence + A2 showcase
<sha> feat(m15): A2 third brick — FFN as compositional NFL pattern
<sha> fix(m15): close LH-4 — relocate %r8/%r9 in x86_64 emit_layernorm
<sha> docs(m15): address spec review — N=4 evidence gap + FFI signatures
<sha> docs(m15): brainstorm spec — A2 third brick (FFN) + LH-4 cleanup
```

- [ ] **Step 4.2: Each commit independently passes gates** (sanity verify with `git rebase -x` style check if desired):

```bash
for sha in $(git log --oneline -4 --format=%H); do
    git checkout "$sha" -- .
    cargo test --workspace 2>&1 | tail -3
done
git checkout HEAD -- .
```

(Optional but recommended pre-push verification.)

- [ ] **Step 4.3: Final full test pass**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace 2>&1 | grep "test result:"
```

Expected: all green; test counts match the M15 entry in DEVLOG.

- [ ] **Step 4.4: Verify LH-4 row absent from PROJECT_SPEC**

```bash
grep -n "LH-4" PROJECT_SPEC.md
```

Expected: matches only in §"Milestones to date" row 15 narrative (e.g. "LH-4 cleanup in x86_64 `emit_layernorm`") and possibly historical references — NO row in §"Known Latent Hazards" table.

- [ ] **Step 4.5: Push branch + open PR**

```bash
git push -u origin claude/stupefied-zhukovsky-59aaaf
gh pr create --title "M15: A2 third brick (FFN) + LH-4 cleanup" --body "$(cat <<'EOF'
## Summary

Closes M15. Two coupled deliverables in one PR (4 commits, T0→T1→T2→T3):

- **LH-4 cleanup** in `profiles/x86_64/src/ops/layernorm.rs` — per-row scratch `%r8`→`%r15`, `%r9`→`%rbp`. Closes the M14-opened latent hazard. Asm-shape unit tests at N=2/3/4; runtime FFI evidence at N=3.
- **A2 third brick — FFN** as compositional NFL pattern (`linear → relu → linear`). No new StdOp / IR / codegen changes. Two new fixtures: `ffn.nfl` (N=1 baseline), `transformer_block.nfl` (N=3, exercises LH-4 condition + showcases full transformer block).

## Design / plan references

- Brainstorm spec: `docs/superpowers/specs/2026-05-10-m15-ffn-and-lh4-cleanup-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-10-m15-ffn-and-lh4-cleanup.md`

## Test plan

- [x] `cargo test --workspace` green on macOS arm64 (T0 unit tests + T1 `ffn_ffi` arm64 + T2 `transformer_block_ffi` arm64)
- [x] Linux x86_64 CI green (T0 unit tests + T1 `ffn_ffi` x86_64 + **T2 `transformer_block_ffi` x86_64** ← LH-4 runtime evidence)
- [x] `cargo clippy --workspace --all-targets -- -D warnings` green for all 4 commits
- [x] `cargo fmt --all -- --check` green for all 4 commits
- [x] PROJECT_SPEC §"Known Latent Hazards" table empty after M15
- [x] DEVLOG ABI audit paper trail includes `emit_relu` and `emit_add` at N=3 explicitly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed; CI starts.

---

## Self-review checklist (run after writing this plan)

1. **Spec coverage** — every §1–§7 of the design spec is implemented:
   - §1 Goals/Non-goals → Task headers + the file-structure table at top
   - §2 LH-4 cleanup (registers, push order, strategy, alignment, constants, doc-comment, unit tests, evidence-type) → Task 0 Steps 0.1–0.7
   - §3 Fixtures + reference impls + FFI tests + helper-reuse rule + extern "C" signatures → Task 1 (ffn) + Task 2 (transformer_block)
   - §4 Tasks T0–T3 with bisectability claim → Task 0/1/2/3 (claim in T2 commit message)
   - §5 ABI audit obligation + paper trail (emit_relu, emit_add explicit) → Task 3 Step 3.1 DEVLOG content
   - §6 Done definition (9 items) → Final verification Steps 4.1–4.5
   - §7 References → DEVLOG content + spec link at top of plan

2. **Placeholder scan** — only intentional `TODO_FILL_IN` (test counts, SHAs) and `<TX_SHA>` markers remain; explicitly flagged in Step 3.6 to be filled before T3 commit. NO content TODO/FIXME/TBD.

3. **Type consistency** — function names match across tasks:
   - `ffn_ref` (defined Step 1.3, called Step 1.8/1.9, called Step 2.3 inside `transformer_block_ref`)
   - `transformer_block_ref` (defined Step 2.3/2.4, called Step 2.5/2.6)
   - `reference_matmul/bias_add/relu` (promoted Step 1.3/1.4, used Step 1.5–1.6 + 2.3)
   - FFI fn types: `unsafe extern "C" fn(*const f32, *const f32, *mut f32)` for ffn (3 params), `unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32)` for transformer_block (5 params) — consistent across arm64 and x86_64.
