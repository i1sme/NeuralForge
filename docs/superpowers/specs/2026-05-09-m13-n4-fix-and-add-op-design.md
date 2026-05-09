# Milestone 13 — N=4 matmul fix + `add` op (A2 first brick) — Design

> Brainstormed: 2026-05-09
> Strategic axis: **Axis 2 — modelling depth** (PROJECT_SPEC §"Strategic Roadmap"). Continues the A2 (transformer block) axis whose first leg A1 (multi-input ABI) closed in M12. Closes the explicit M12→M13 priority signal (DEVLOG: "the most important bequest from M12 to M13") AND ships the first elementwise binary op needed for residual connections.
> Predecessor: M12 (multi-input ABI)
> Status: spec draft for plan synthesis

---

## 1. Overview

M13 has two deliverables, sequenced atomically:

1. **N=4 + matmul gap on x86_64** — closes the known M12 follow-up. `profiles/x86_64/src/ops/matmul.rs:128-139` currently rejects N=4 + matmul with `LowerError::UnsupportedOp` because the inner j-loop counter is hardcoded to `%r9`, which becomes `output_reg()` at N=4 (`AbiContext::output_reg` returns `INPUT_REGS[N+1]`, so at N=4 it's `INPUT_REGS[5] == "%r9"`). The j-counter clobbers the output pointer on the first iteration, producing silently wrong asm. M12 chose to reject at lowering rather than emit broken codegen; M13 reassigns the j-counter to a non-ABI scratch register and flips the rejection test into a positive integration test.

2. **`add` op (StdOp::Add)** — the first A2 brick. `add` is an elementwise binary tensor op with strict shape equality, surfaced in NFL as `a -> add[skip]` (positional Tensor arg named `"other"`, mirroring `Matmul`'s signature). It is the first operation that uses `ArgType::Tensor` outside of `Matmul`, validating the M10 machinery as a real reusable pattern. `add` enables residual connections, the foundational building block for all transformer-style architectures (A2 in M14+ adds LayerNorm and FFN on top of this).

The strategic claim being validated is that **A2 is decomposable into atomic ops**, not a monolithic "transformer block" milestone. M13 ships exactly one op (`add`); M14+ ships LayerNorm and FFN as separate composite ops (each likely a single `StdOp` variant with internal multi-pass codegen, mirroring how `Softmax` is one node, not "exp + sum + divide" decomposed). Scope discipline is the explicit goal: "transformer block" is not a milestone, it is an axis.

### Why these two together

The N=4 fix is a prerequisite for unblocking N=4 + matmul scenarios; future A2 work (e.g. multi-head attention with N=4 inputs Q/K/V/mask) needs it. Making the N=4 fix a standalone M13 would create a one-commit milestone for cosmetic reasons. Pairing it with the smallest possible A2 brick (`add`) gives M13 a coherent shape: "close M12's known gap, ship the next op."

### Non-goals (explicitly deferred)

- **LayerNorm** — deferred to M14. Requires mean/variance/sqrt/divide computation pattern not yet present in any codegen.
- **FFN** — deferred to M14+. Composes existing ops (`linear → activation → linear`); no new codegen, but waits behind LayerNorm for milestone packaging.
- **`sub`, `mul`, `div` elementwise** — not in M13. May never ship as standalone ops if LayerNorm/FFN encapsulate them. Three-strikes-then-refactor (M7 precedent): refactor `StdOp::Add` into a `BinaryOp` container only when a third elementwise sibling actually arrives with concrete differences.
- **Broadcasting** — not in M13 and not in NFL v0.x at all. Strict shape equality per design principle #1 ("Explicit over implicit"). Broadcasting belongs to NFL v0.3+ if ever.
- **Span on `ShapeError::AddShapeMismatch`** — follows existing `ShapeError` variant pattern (none of the 7 current variants carry `Span`; spans live on `BuildError` which wraps `ShapeError`). Adding span is the trigger for **M5c OQ-4** and would need to be applied to all 7 existing variants atomically, not piecemeal.
- **Kernel fusion involving `add`** — not in M13. Future fusion pass like `FuseLinearAdd` (residual add into linear) is plausible but waits for a real motivating fixture; current fusion passes (`FuseLinearRelu`, `FuseLinearSoftmax`, `EliminateDropout`) are unrelated.

---

## 2. Goals

Ship a single PR with **6 atomic commits** (option **P** from brainstorm Q5):

1. **Group A — N=4 + matmul fix on x86_64.** Reassign j-counter in `profiles/x86_64/src/ops/matmul.rs::emit_matmul` to a scratch register that is non-ABI at N=4. Remove the early `Err(LowerError::UnsupportedOp)` guard (`matmul.rs:134-139`). Flip `profiles/x86_64/src/tests.rs:1381` (`Group C Q11: N=4 rejection`) into a positive emit test. Update the module doc-comment table (`matmul.rs:42-58`) to reflect the new j-counter slot. Update `profiles/x86_64/src/ops/matmul.rs:23,26` "M12 caps N at 4" / "%r9 also caller-saved but transitions into ABI at N=4" wording to match the new reality.
2. **Group B — `add` op foundation.** `compiler/src/ir/stdlib.rs`: new `StdOp::Add` variant, `Signature` slot mirroring `Matmul` (positional `ArgSlot { name: "other", ty: Tensor, required: true }`, no named args), `infer_output_shape` arm doing strict shape equality + new `ShapeError::AddShapeMismatch { expected: Shape, got: Shape }`, `validate_attrs` no-op arm, `resolve("add") => Some(StdOp::Add)`, `Display for StdOp` "add" arm. Plus parser/builder unit tests pinning `a -> add[skip]` round-trip and the negative shape-mismatch case. **No emit_add yet — `walk_model` returns `Err(LowerError::UnsupportedOp)` for `StdOp::Add` until Group C/D land.**
3. **Group C — arm64 `emit_add`.** New `profiles/arm64/src/ops/add.rs::emit_add`. Flat elementwise loop: materialise `a_loc` and `other_loc` pointers, materialise `dst_loc`, loop over `total_elements = product(shape)` with `ldr s0, [a_ptr], #4` / `ldr s1, [other_ptr], #4` / `fadd s2, s0, s1` / `str s2, [dst_ptr], #4`. `walk_model` dispatch: shape inference passed from UIR, no FFI save/restore (no `bl _expf`), no scratch budget pressure. Unit test on emitted asm shape (analyser-style, like existing `emit_relu_emits_loop` tests).
4. **Group D — x86_64 `emit_add`.** Mirror of Group C for SysV. `profiles/x86_64/src/ops/add.rs::emit_add`. AT&T elementwise loop: `movss (a_ptr_reg), %xmm0` / `movss (other_ptr_reg), %xmm1` / `addss %xmm1, %xmm0` / `movss %xmm0, (dst_ptr_reg)`, where `a_ptr_reg` / `other_ptr_reg` / `dst_ptr_reg` are the materialised pointer registers from `{%rax, %r10, %r11}` per §5.4. Same structure as `emit_mulscalar` (M10) which is the closest existing template — flat scalar loop, no scratch pressure. Unit test analogous to Group C.
5. **Group E — integration fixtures + per-profile FFI tests.** Three fixtures:
   - **(i)** `tests/fixtures/residual_add.nfl` (form: `model Block: x: Tensor[batch, dim], skip: Tensor[batch, dim]; x -> linear[dim] -> relu -> add[skip]`) — positive `add` op end-to-end on both profiles.
   - **(ii)** N=4 + matmul fixture (name TBD by plan synthesis per §6.2) — closes Group A's N=4 fix end-to-end on x86_64. Optional regression sanity on arm64 since arm64 already supported N=4 + matmul at M12.
   - **(iii)** `tests/fixtures/profile-negative/add_shape_mismatch.nfl` — exercises `ShapeError::AddShapeMismatch`.

   Per-profile FFI integration tests in `profiles/{arm64,x86_64}/tests/integration.rs`: compile via cc + dlopen, call with random inputs / params, compare bit-exact against reference Rust impl.
6. **Group F — closure docs.** PROJECT_SPEC.md (M13 row in milestones table + Current Status bumped + Strategic Roadmap A2 annotation: "M13 closed N=4 gap + shipped first A2 brick `add`; A2 LayerNorm + FFN remain in M14+"), CLAUDE.md (Repository Structure tree gains `profiles/{arm64,x86_64}/src/ops/add.rs`, Current Status to M13), `docs/language_reference/grammar.md` (new `add` op in stdlib reference), `docs/profile_guide/{arm64,x86_64}.md` ("M13 ops" section + x86_64 N=4 fix note in matmul section), `docs/language_reference/uir.md` (new fixture in pretty-print examples if applicable), DEVLOG entry.

---

## 3. Group A — N=4 + matmul fix detailed

### 3.1 The bug, restated

At N=4 on x86_64, `AbiContext::output_reg()` returns `INPUT_REGS[5] = "%r9"` (the 6th SysV argument register, used for output). `emit_matmul`'s inner j-loop uses `%r9` as its counter, hardcoded across the inner loop. On the first j-iter, the counter overwrites the output pointer; subsequent inner-loop stores write to whatever value `j` has (zero, then 1, ...), corrupting memory and silently producing wrong results. `matmul.rs:134-139` therefore rejects N=4 at lowering.

### 3.2 Constraints on the fix

The plan must satisfy these invariants:

- **No new callee-saved registers added to the prologue.** `compute_callee_saved` already covers `%rbx, %r12-%r15` when matmul is present (`profiles/x86_64/src/buffer.rs::compute_callee_saved`). Expanding the prologue surface for matmul-only register pressure is rejected.
- **Matmul body must NOT touch any ABI argument register.** Spec §9.1 invariant from M12 — the body must remain ABI-clean so downstream emitters read input/params/output from those registers.
- **The j-counter slot must not collide with the M12 register layout** (`matmul.rs:42-58` table). %rax is clobbered by `imulq`. %r10 is clobbered by `emit_imm32_to_r10`. %r11 is the k_inner counter (innermost loop, changes per j). %rbx, %r12, %r13, %r14, %r15 are all in use.
- **The unit test `emit_matmul_body_contains_zero_pushq`** (`profiles/x86_64/src/tests.rs:1350`) must still hold — no new `pushq` in the matmul body. Stack-spill of the j-counter to a stack slot is acceptable IF the slot is allocated in the function-level prologue (not via `pushq` inside the body).

### 3.3 Strategy options for plan synthesis

Plan synthesis chooses one of these (or proposes another satisfying §3.2):

- **Option A.1 — j-counter in a stack slot.** Allocate one extra qword in the function-level prologue's local-variable region; use `movq` to/from `j_slot(%rsp)` per inner-loop arith. ~3 extra memory accesses per j-iter, OoO pipeline absorbs the latency. Conceptually simplest.
- **Option A.2 — j-counter in `%xmm9` (or another unused xmm).** SysV ABI marks all `%xmm0`-`%xmm15` as caller-saved; %xmm6/7/8 are used by matmul base-pointer storage. %xmm9 is free. `movd`/`movd` GPR↔XMM round-trip per arith op is fast on modern uarch but uglier asm.
- **Option A.3 — restructure j-loop to not need a separate counter.** Collapse j into an address-arithmetic increment (e.g. compute `j_ptr = b_slice + j_idx*4` differently so the loop iterates by pointer, not index). Most invasive but might eliminate the need for a j-counter register entirely.

Plan synthesis picks one, defends it under §3.2 constraints, and writes the chosen asm in the plan body. Brainstorm leaves this open — the choice is a performance/clarity trade-off best made with the surrounding emit code in front of you.

### 3.4 Test transition (Group A — unit-test-only)

`profiles/x86_64/src/tests.rs:1381` (`Group C Q11: N=4 rejection`) currently asserts `Err(LowerError::UnsupportedOp)` with a message containing "N=4" or "4 inputs". Group A flips this to a positive `Ok(asm)` test that pins the new j-counter slot in the emitted asm (e.g. `assert!(asm.contains("<chosen_j_register>"))` and `assert!(!asm.contains("%r9, %r9"))` to prove %r9 is no longer self-clobbered).

**FFI end-to-end coverage of N=4 + matmul lives in Group E §6.2**, not in Group A. This matches the M9/M10/M12 pattern: codegen-touching groups ship unit tests on emitted asm shape; integration fixtures + cc+dlopen+bit-exact tests are bundled in the dedicated integration group. Group A stays surgical — one file edit, one test flip.

---

## 4. Group B — `add` op foundation detailed

### 4.1 NFL surface (confirmed in brainstorm Q3)

```nfl
model ResidualBlock [batch=32, dim=512]:
    x: Tensor[batch, dim]
    skip: Tensor[batch, dim]

    x -> linear[dim] -> relu -> add[skip]
```

`add[skip]` parses identically to `matmul[other]` — positional Tensor arg. The parser already handles this via M10's `ArgType::Tensor` machinery; no parser changes required, only a new entry in the `StdOp` lookup tables.

### 4.2 `StdOp::Add` definition

```rust
pub enum StdOp {
    // ... existing variants ...
    /// Per-element tensor addition. Two tensor operands, strict shape
    /// equality (no broadcasting per design principle #1). Shape is
    /// preserved. New in M13 — first A2 brick (residual connections).
    Add,
}
```

`Signature` (mirrors `Matmul` minus the `transpose_b` named arg):

```rust
StdOp::Add => Signature {
    positional: &[ArgSlot {
        name: "other",
        ty: Tensor,
        required: true,
    }],
    named: &[],
},
```

`infer_output_shape`:

```rust
StdOp::Add => {
    if inputs.len() != 2 {
        return Err(ShapeError::WrongInputCount { expected: 2, actual: inputs.len() });
    }
    if inputs[0] != inputs[1] {
        return Err(ShapeError::AddShapeMismatch {
            expected: inputs[0].clone(),
            got: inputs[1].clone(),
        });
    }
    Ok(inputs[0].clone())
}
```

`validate_attrs` arm: `StdOp::Add => Ok(())` (no attrs). `resolve` arm: `"add" => Some(StdOp::Add)`. `Display`: `"add"`.

### 4.3 New `ShapeError` variant

```rust
/// Two `add` operands have different shapes. Strict equality required —
/// no broadcasting per design principle #1.
AddShapeMismatch {
    expected: Shape,
    got: Shape,
},
```

`Display`:

```rust
ShapeError::AddShapeMismatch { expected, got } => write!(
    f,
    "add operand shape mismatch: expected {}, got {} (no broadcasting)",
    expected, got
),
```

The `(no broadcasting)` suffix mirrors `LeadingDimMismatch`'s wording exactly — this is the established hint for "you cannot broadcast in NFL."

### 4.4 walk_model dispatch (Group B exit state)

In Group B, both profiles' `walk_model` dispatches `StdOp::Add` to `Err(LowerError::UnsupportedOp { op: "add (M13 codegen pending)".into(), span: node_span })`. This keeps Group B compilable without an emitter. Groups C and D replace the placeholder with the real call to `emit_add`.

### 4.5 Tests added in Group B

- `compiler/src/ir/tests/stdlib.rs` (or wherever StdOp tests live): positive `add[skip]` build round-trip; negative `AddShapeMismatch` (e.g. `[32, 512]` + `[32, 256]`).
- `compiler/src/parser/tests.rs`: parses `x -> add[skip]` to expected AST.
- No FFI tests in Group B (no codegen yet).

---

## 5. Groups C/D — codegen detailed

### 5.1 Codegen shape (both profiles)

Both `emit_add` are flat elementwise loops over `total_elements = shape.0.iter().product()`. Closest existing template is `emit_mulscalar` (M10) — also a flat scalar loop, also no scratch pressure, also no FFI save/restore. The only structural difference is: `mul_scalar` reads ONE input pointer + ONE pre-loaded scalar; `add` reads TWO input pointers (the pipeline carrier and the named-arg `other`).

### 5.2 BufferLoc for the `other` operand

`a_loc` (pipeline carrier) is whatever upstream produced — typically `BufferLoc::Stack(...)` after a `linear`. `other_loc` is the `BufferLoc` of the node referenced by `operands[1]`. If `skip` is a model input, `other_loc == BufferLoc::InputReg(skip_input_idx)`. The standard `materialise_ptr(loc, scratch_reg, &mut s)` machinery from M12 handles both cases — no special-casing required.

### 5.3 arm64 `emit_add` register budget

- `x9, x10, x11` (caller-saved scratch, non-ABI for N≤4): pointers to `a` / `other` / `dst` after materialise.
- `x12` (caller-saved scratch): element counter.
- `s0, s1, s2` (caller-saved FP): load-load-add-store cycle.
- No callee-saved touched. No FFI save/restore.

### 5.4 x86_64 `emit_add` register budget

- `%rax, %r10, %r11` (caller-saved non-ABI): pointers to `a` / `other` / `dst` after materialise. These three are free at all N ∈ [1, 4].
- **Loop counter must be non-ABI at all N ∈ [1, 4].** The full ABI footprint by arity (per `INPUT_REGS = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"]` at `profiles/x86_64/src/abi.rs:27`):

  | N | inputs | params | output | free non-ABI scratch |
  |---|--------|--------|--------|----------------------|
  | 1 | %rdi | %rsi | %rdx | %rcx, %r8, %r9, %rax, %r10, %r11 |
  | 2 | %rdi, %rsi | %rdx | %rcx | %r8, %r9, %rax, %r10, %r11 |
  | 3 | %rdi, %rsi, %rdx | %rcx | %r8 | %r9, %rax, %r10, %r11 |
  | 4 | %rdi, %rsi, %rdx, %rcx | %r8 | %r9 | %rax, %r10, %r11 |

  `%rcx` becomes ABI at N=2 (output) and stays ABI through N=4 — it is free **only** at N=1, never use it as `emit_add` scratch. The intersection of free non-ABI scratch across all N is `{%rax, %r10, %r11}` — three GP registers, exactly the materialised pointer set, leaving zero spare GPR for the loop counter.
- **Loop counter strategy** is plan-synthesis Q (§9 item 2). Three options: (a) stack slot allocated in prologue (cheapest, mirrors §3.3 Option A.1 for matmul); (b) collapse loop counter into pointer arithmetic (compare `dst_ptr` against `dst_end_ptr` precomputed from `total_elements`); (c) thread `AbiContext` and pick a register N-aware (ugly — emit_add becomes N-conditional). Brainstorm recommendation is (b) — pointer-bound comparison eliminates the counter entirely and is idiomatic for flat elementwise loops.
- `%xmm0, %xmm1` (caller-saved): load-load-addss-store cycle.
- No callee-saved touched. No FFI save/restore.

### 5.5 Group C/D tests

Per-profile unit test (analyser-style) on emitted asm — matches existing `emit_relu` / `emit_mulscalar` test patterns. Verifies:
- Loop structure (one cmp, one branch, one labelled body).
- Two input pointer loads per iter (this is what differentiates `add` from `mul_scalar`).
- One add instruction per iter (`fadd` arm64, `addss` x86_64).
- One store per iter.
- No `bl _expf` / `call expf@PLT` — `add` must not trigger callee-saved expansion.

---

## 6. Group E — integration tests

### 6.1 Positive fixture: `residual_add.nfl`

```nfl
model ResidualBlock [batch=2, dim=4]:
    x: Tensor[batch, dim]
    skip: Tensor[batch, dim]

    x -> linear[dim] -> relu -> add[skip]
```

Small dims for fast FFI execution. `batch=2, dim=4` is enough to exercise the loop multiple times without wall-clock cost.

### 6.2 N=4 + matmul positive fixture (closes Group A end-to-end)

**Owned by Group E** (this group), not Group A. Group A is unit-test-only (§3.4); end-to-end FFI coverage of the N=4 + matmul fix lands here alongside `residual_add` and the negative fixture so the full integration surface ships in one reviewable group.

Either extend `multi_input_attention.nfl` to a 4-input variant or add `tests/fixtures/four_input_matmul.nfl`. Plan synthesis decides; the constraint is that the fixture must exercise both N=4 input ABI mapping AND a matmul op (the bug surface). Per-profile FFI test added on x86_64 (where Group A's fix lives); arm64 regression sanity is optional — arm64 already supported N=4 + matmul at M12 and has no Group A change to validate.

### 6.3 Negative fixture: `add_shape_mismatch.nfl`

```nfl
model BadAdd [batch=2]:
    x: Tensor[batch, 4]
    skip: Tensor[batch, 8]

    x -> add[skip]
```

Builds successfully through parser, fails at IR shape inference with `ShapeError::AddShapeMismatch { expected: Shape([2, 4]), got: Shape([2, 8]) }`. Wired into the existing fixture-runner (same machinery as M10's negative matmul fixtures, `tests/fixtures/profile-negative/`).

### 6.4 FFI integration tests

Per-profile in `profiles/{arm64,x86_64}/tests/integration.rs`. Pattern matches existing M10/M12 multi-input tests:

1. Compile fixture → `.s` → `.dylib` via `cc + tempdir` helper.
2. dlopen, lookup `nfl_forward_<ModelName>`.
3. Generate random `x`, `skip`, `params` (Linear weights) with seeded RNG.
4. Call FFI function.
5. Compute reference output in pure Rust: `relu(x @ W + b) + skip` element-wise.
6. Assert bit-exact: `assert_eq!(ffi_output, reference_output)`.

---

## 7. Group F — documentation closure

### 7.1 PROJECT_SPEC.md

- Milestones table: new M13 row.
- Current Status: bumped to M13. Test count delta noted (estimate: 390 → ~410, plan synthesis confirms).
- Strategic Roadmap §"Axis 2 — modelling depth": A2 annotation updated to "M13 closed N=4 gap + shipped first A2 brick `add`; A2 LayerNorm + FFN remain in M14+." Known gap line removed (N=4 + matmul closed).

### 7.2 CLAUDE.md

- Repository Structure tree: `profiles/{arm64,x86_64}/src/ops/add.rs` added.
- Current Status: bumped to M13.

### 7.3 `docs/language_reference/grammar.md`

- New `add` op in stdlib reference. Same level of detail as `matmul` and `mul_scalar`.

### 7.4 `docs/profile_guide/arm64.md` and `docs/profile_guide/x86_64.md`

- New "M13 ops" section: brief description of `emit_add` codegen (loop shape, register choices, no FFI).
- x86_64 only: matmul section gets a "M13 N=4 fix" note explaining the j-counter reassignment.

### 7.5 `docs/language_reference/uir.md`

- If the `--uir-verbose` pretty-print needs adjustment for the new node kind, update the examples. Likely no change needed since `Add` uses the existing `NodeKind::Op` machinery.

### 7.6 DEVLOG.md

- Standard M13 entry: What was done, Decisions made, Problems encountered, Next step.

---

## 8. Open questions resolved by brainstorm

| # | Question | Resolution | Rationale |
|---|----------|------------|-----------|
| Q1 | M13 framing — surgical N=4 fix only, or themed milestone? | Themed: N=4 fix opener + `add` op | DEVLOG identified N=4 fix as M12→M13 priority signal; pairing with smallest A2 brick gives M13 coherent shape |
| Q2 | A2 scope inside M13 | `add` only; defer LayerNorm + FFN to M14+ | Scope discipline: "transformer block" is an axis, not a milestone |
| Q3 | NFL surface syntax for `add` | Option α: `a -> add[skip]` (positional Tensor arg, mirrors Matmul) | First real consumer of M10 `ArgType::Tensor` outside Matmul; minimal token overhead per AI-native principle |
| Q4 | UIR shape: flat `StdOp::Add` vs `BinaryOp` container | Flat `StdOp::Add` | Consistent with all 6 existing flat StdOp variants; three-strikes-then-refactor (M7 precedent); LayerNorm/FFN may never decompose into elementwise primitives |
| Q5 | Commit structure | Option P: 6 atomic groups | M7/M12 atomic-task-pack convention; per-group review surface preserved |

## 9. Open questions deferred to plan synthesis

- **N=4 j-counter register choice** (§3.3) — A.1 stack slot vs A.2 `%xmm9` vs A.3 j-loop restructure. Plan picks under §3.2 constraints.
- **`emit_add` x86_64 loop counter strategy** (§5.4) — intersection of free non-ABI scratch across N ∈ [1,4] is `{%rax, %r10, %r11}` and is fully consumed by the three materialised pointers, leaving zero spare GPR for a counter. Pick: (a) stack slot, (b) pointer-bound comparison eliminating the counter (brainstorm-recommended), or (c) N-aware register pick via `AbiContext`.
- **N=4 + matmul fixture name** (§6.2) — extend `multi_input_attention.nfl` or new `four_input_matmul.nfl`.
- **Test count target** (§7.1) — plan synthesis confirms after counting per-group test additions.

## 10. Constraints inherited from prior milestones

- **M7 atomic-task-pack convention** — each Group commits independently, compiles independently, and is independently reviewable.
- **M9 profile isolation** — Groups C and D touch only their respective profile crates; no cross-profile dependency.
- **M10 `ArgType::Tensor` machinery** — `add` is its first reuse; no parser/builder changes beyond the `StdOp` tables.
- **M11 §11.2 no-CI-artifact-sharing** — N/A for M13 (no bench changes).
- **M12 §9.1 ABI-clean matmul body** — Group A must preserve this invariant; the j-counter reassignment must not introduce ABI-register clobber.
- **M12 multi-input ABI** — `add` consumes a named tensor input; `BufferLoc::InputReg(usize)` and `materialise_ptr` already handle this; no new buffer machinery.

## 11. Trigger ledger interactions

- **OQ-7** (M7) — pass-level Err: not affected by M13 (no new passes).
- **OQ-8** (M7) — non-pass UIR-rewrite consumer: not affected by M13.
- **OQ-9** (M7) — fourth pass with non-PostOp producer mutation: not affected by M13.
- **M5c OQ-4** — `BuildError::span()` + `Diagnostic` trait: NOT triggered by `AddShapeMismatch` (intentionally — adding span only to one variant would violate the "all-or-nothing" principle implied by the trigger). M13's diagnostic ergonomics are no worse than M12's.

## 12. Acceptance criteria

M13 is complete when:

1. `cargo build --workspace` clean.
2. `cargo clippy --workspace --all-targets -- -D warnings` clean.
3. `cargo fmt --all -- --check` clean.
4. `cargo test --workspace` passes; test count strictly greater than 390.
5. `tests/fixtures/residual_add.nfl` compiles and runs bit-exact via FFI on both profiles.
6. `tests/fixtures/profile-negative/add_shape_mismatch.nfl` rejects with `ShapeError::AddShapeMismatch`.
7. N=4 + matmul fixture compiles and runs bit-exact via FFI on x86_64; the M12-era `Err(LowerError::UnsupportedOp)` rejection path no longer fires.
8. `profiles/x86_64/src/ops/matmul.rs::emit_matmul_body_contains_zero_pushq` unit test still holds.
9. Documentation Groups F items shipped (PROJECT_SPEC, CLAUDE.md, grammar.md, profile guides, DEVLOG entry).
10. Bench harness still builds and runs on both profiles (no Bench changes in M13, but build must not regress).
