# M13 — N=4 matmul fix + `add` op (A2 first brick) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the M12 follow-up gap (x86_64 `emit_matmul` rejects N=4 + matmul) AND ship `StdOp::Add` end-to-end on both profiles, enabling residual connections — the first A2 brick.

**Architecture:**
- **Group A** (Task 1) replaces `emit_matmul`'s hardcoded `%r9` j-counter with `%rbp` and removes the N=4 reject. Plan-synthesis option D (not in spec §3.3 enumeration but explicitly permitted by "or proposes another satisfying §3.2"): `%rbp` is callee-saved by the function-level prologue (`pushq %rbp` is unconditional in `asm.rs::format_function_prologue`) and is read by zero op-emitters today (verified by `grep -c rbp profiles/x86_64/src/ops/`). Inside the function body `%rbp` is wide-open scratch. This satisfies all four §3.2 constraints with zero prologue surface change.
- **Groups B-F** ship `StdOp::Add` as a flat StdOp variant (no `BinaryOp` container), surfaced in NFL as `a -> add[skip]` (positional Tensor arg, mirrors `Matmul`'s `other` slot). `emit_add` is a flat elementwise loop on both profiles, modeled after `emit_mulscalar` (M10). On x86_64, `emit_add` reuses the same `%rbp` scratch trick from Group A as its loop counter.

**Tech Stack:** Rust 2024 edition workspace; macOS arm64 (host) + Linux x86_64 (CI); `cc` + `libloading` for FFI integration tests.

---

## Spec reference

This plan implements `docs/superpowers/specs/2026-05-09-m13-n4-fix-and-add-op-design.md`. Per-section traceability:

| Spec section | Plan task |
|--------------|-----------|
| §2 #1 + §3 (Group A) | Task 1 |
| §2 #2 + §4 (Group B foundation) | Task 2 |
| §2 #3 + §5.1, §5.3 (Group C arm64 emit) | Task 3 |
| §2 #4 + §5.1, §5.4 (Group D x86_64 emit) | Task 4 |
| §2 #5 + §6 (Group E integration) | Task 5 |
| §2 #6 + §7 (Group F docs) | Task 6 |

---

## File Structure

### Created
- `profiles/arm64/src/ops/add.rs` — `pub fn emit_add(...)` for arm64.
- `profiles/x86_64/src/ops/add.rs` — `pub fn emit_add(...)` for x86_64.
- `tests/fixtures/residual_add.nfl` — positive `add` op fixture.
- `tests/fixtures/four_input_matmul.nfl` — positive N=4 + matmul fixture (closes Group A end-to-end).
- `tests/fixtures/negative/add_shape_mismatch.nfl` — IR-level rejection fixture.

**Note on negative-fixture dir:** spec §6.3 said `tests/fixtures/profile-negative/add_shape_mismatch.nfl`; the correct dir is `tests/fixtures/negative/` because `AddShapeMismatch` fires at IR build (compiler-level), not at lower (profile-level). The `profile-negative/` dir is reserved for `LowerError` fixtures (e.g. `too_many_inputs.nfl`). Verified by reading `compiler/tests/negative_fixtures.rs` (loops over `tests/fixtures/negative/` and asserts parse-or-build error).

### Modified
- `profiles/x86_64/src/ops/matmul.rs` — Task 1: replace `%r9` j-counter with `%rbp`; remove N=4 reject; update register-layout doc comment.
- `profiles/x86_64/src/tests.rs` — Task 1: flip `emit_matmul_rejects_n4_with_clear_error` (line 1383) into a positive `Ok(asm)` test that pins the new `%rbp` j-counter.
- `compiler/src/ir/stdlib.rs` — Task 2: add `StdOp::Add` variant + `Signature` + `infer_output_shape` arm + `validate_attrs` arm + `resolve` entry + `Display` arm + `ShapeError::AddShapeMismatch`.
- `profiles/arm64/src/ops/mod.rs` — Task 3: `pub mod add;`.
- `profiles/arm64/src/codegen.rs` — Task 3: extend `walk_model` match arms with `StdOp::Add` (and `classify_op` arm if needed); add per-model counter `add_idx`.
- `profiles/x86_64/src/ops/mod.rs` — Task 4: `pub mod add;`.
- `profiles/x86_64/src/codegen.rs` — Task 4: same as arm64 codegen.rs change.
- `profiles/arm64/tests/integration.rs` — Task 5: `residual_add_match_numerically` test.
- `profiles/x86_64/tests/integration.rs` — Task 5: `residual_add_match_numerically` + `four_input_matmul_match_numerically` tests.
- `PROJECT_SPEC.md` — Task 6: M13 row in milestones table; Current Status update; Strategic Roadmap A2 annotation; remove the M12 "Known gap" line about N=4 + matmul.
- `CLAUDE.md` — Task 6: tree gains `profiles/{arm64,x86_64}/src/ops/add.rs`; Current Status to M13.
- `DEVLOG.md` — Task 6: standard M13 entry.
- `docs/language_reference/grammar.md` — Task 6: `add` op stdlib reference.
- `docs/profile_guide/arm64.md` — Task 6: M13 ops section + brief `emit_add` description.
- `docs/profile_guide/x86_64.md` — Task 6: same as arm64.md + matmul section gets a "M13 N=4 fix" note explaining `%rbp` j-counter.

### Not modified (notable)
- `compiler/src/ir/build.rs` — `StdOp::Add` flows through the existing `build_op` machinery automatically once `StdOp::Add` is in the lookup tables; no per-op handling required.
- `compiler/src/parser/` — `add[skip]` parses identically to `matmul[other]`; the parser's positional-arg handler already resolves identifiers via `ArgType::Tensor`. No parser changes.
- `profile-api/` — no new error variants; `LowerError::UnsupportedOp` placeholder in Task 2 reuses the existing variant.
- `bench/` — no bench changes; M13 doesn't add bench fixtures.

---

## Task 1: Group A — Close N=4 + matmul gap on x86_64

**Files:**
- Modify: `profiles/x86_64/src/ops/matmul.rs:128-269` (remove N=4 reject + relocate j-counter)
- Modify: `profiles/x86_64/src/tests.rs:1381-1419` (flip rejection test → positive emit test)

**Strategy:** Replace `%r9` j-counter with `%rbp`. Justification:
- `%rbp` is callee-saved per SysV; the existing prologue unconditionally `pushq %rbp`s and the epilogue `popq %rbp`s. The body's clobber of `%rbp` is restored by the existing epilogue.
- No op-emitter currently reads `%rbp` (verified by `grep -rn rbp profiles/x86_64/src/ops/` — only doc comments in `matmul.rs:27-28` mention it as the "frame pointer" role).
- Satisfies all four §3.2 constraints: no new callee-saved (`%rbp` already saved); body remains ABI-clean (`%rbp` is not an ABI register); no collision with existing matmul layout; no new `pushq` in body (the `emit_matmul_body_contains_zero_pushq` invariant holds).
- Plan-synthesis option **D** beyond spec §3.3 enumeration (A.1 stack slot, A.2 `%xmm9`, A.3 loop restructure). Spec §3.3 explicitly permits "or proposes another satisfying §3.2".

- [ ] **Step 1.1: Read the existing test and emit code to confirm context**

Run: `sed -n '1381,1419p' profiles/x86_64/src/tests.rs`
Expected: see the `emit_matmul_rejects_n4_with_clear_error` test that asserts `Err(LowerError::UnsupportedOp)`.

Run: `sed -n '128,140p' profiles/x86_64/src/ops/matmul.rs`
Expected: see the early `if abi.n_inputs == 4 { return Err(...) }` guard.

- [ ] **Step 1.2: Write the failing test (positive emit at N=4)**

Replace the body of `emit_matmul_rejects_n4_with_clear_error` in `profiles/x86_64/src/tests.rs:1383-1419` with a positive emit test. Rename the function to `emit_matmul_accepts_n4_with_rbp_j_counter`. Keep the comment block but update it to reflect the new behavior.

```rust
// ---- Group A (M13): N=4 + matmul fix via %rbp j-counter --------------------

#[test]
fn emit_matmul_accepts_n4_with_rbp_j_counter() {
    // Group A (M13): the M12 reject path (commit 37868e5) blocked N=4 matmul
    // because %r9 was both the j-counter scratch and output_reg() at N=4.
    // M13 relocates the j-counter to %rbp (callee-saved by the function-level
    // prologue; no op-emitter reads it inside the body). emit_matmul must now
    // accept N=4 and emit asm using %rbp as the j-counter.
    use compiler::ast::Span;
    let abi = AbiContext { n_inputs: 4 };
    let span = Span::new(1, 1);
    let result = crate::ops::matmul::emit_matmul(
        &abi,
        /* leading_count */ 1,
        /* m */ 4,
        /* k */ 8,
        /* n */ 4,
        /* transpose_b */ false,
        /* model_idx */ 0,
        /* matmul_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* b_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
        span,
    );
    let asm = result.expect("emit_matmul must accept N=4 after M13 fix");
    // The j-counter init now writes to %rbp, not %r9.
    assert!(
        asm.contains("movq    $0, %rbp\n"),
        "expected j-counter init `movq $0, %rbp`; got:\n{asm}"
    );
    // Old %r9 j-counter init must be gone.
    assert!(
        !asm.contains("movq    $0, %r9\n"),
        "stale %r9 j-counter init must be removed; got:\n{asm}"
    );
    // %r9 is output_reg at N=4; it must NOT be written to by the matmul body.
    assert!(
        !asm.contains(", %r9\n"),
        "matmul body must not write to %r9 (output_reg at N=4); got:\n{asm}"
    );
}
```

- [ ] **Step 1.3: Run the test to verify it fails**

Run: `cargo test -p profiles-x86_64 --lib emit_matmul_accepts_n4_with_rbp_j_counter`
Expected: FAIL — emit_matmul still rejects N=4 with `LowerError::UnsupportedOp`.

- [ ] **Step 1.4: Remove the N=4 reject and replace `%r9` with `%rbp` in emit_matmul**

In `profiles/x86_64/src/ops/matmul.rs:128-139`, delete the early reject block:

```rust
    // Q11 (Group C): N=4 + matmul register collision.
    // ...8 lines deleted...
    if abi.n_inputs == 4 {
        return Err(LowerError::UnsupportedOp {
            op: "matmul at N=4 inputs on x86_64 (j-counter %r9 collides with output register; M13+ rework planned)".into(),
            span: node_span,
        });
    }
```

In `profiles/x86_64/src/ops/matmul.rs:201-258`, replace each occurrence of `%r9` (j-counter usage) with `%rbp`:

```rust
    // Inner j-loop ([0, N)). Counter %rbp (callee-saved by function-level
    // prologue; no op-emitter inside the body reads it. Replaces M12's
    // %r9 which collided with output_reg() at N=4.).
    s.push_str("    movq    $0, %rbp\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %rbp\n");
    s.push_str(&format!("    jge     .Lmm4d_j_end_{mid}\n"));
```

In the inner k-loop body's b_offset computation (lines 228-238), replace `%r9` with `%rbp`:

```rust
    if transpose_b {
        s.push_str(&emit_imm32_to_r10(k as u32));
        s.push_str("    movq    %rbp, %rax\n");
        s.push_str("    imulq   %r10, %rax\n"); // %rax = j * K
        s.push_str("    addq    %r11, %rax\n"); // %rax = j * K + k_inner
    } else {
        s.push_str(&emit_imm32_to_r10(n as u32));
        s.push_str("    movq    %r11, %rax\n");
        s.push_str("    imulq   %r10, %rax\n"); // %rax = k_inner * N
        s.push_str("    addq    %rbp, %rax\n"); // %rax = k_inner * N + j
    }
```

In the store back to DST_slice (lines 250-254), replace `%r9` with `%rbp`:

```rust
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %rbx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * N
    s.push_str("    addq    %rbp, %rax\n"); // %rax = i * N + j
    s.push_str("    movss   %xmm0, (%r14, %rax, 4)\n");
```

In the j++ tail (line 257), replace `%r9` with `%rbp`:

```rust
    // j++; j-loop tail.
    s.push_str("    addq    $1, %rbp\n");
    s.push_str(&format!("    jmp     .Lmm4d_j_{mid}\n"));
```

- [ ] **Step 1.5: Update the module-level doc comment**

In `profiles/x86_64/src/ops/matmul.rs:42-58`, update the register-layout table's j-counter row (lines 53-54) and the rationale paragraphs (lines 18-26, 27-31).

Lines 53-54 — replace:
```
//! | j counter          | %r9     | per i iter (caller-saved scratch; non-ABI for N≤3) |
```
with:
```
//! | j counter          | %rbp    | per i iter (callee-saved by function-level prologue; not read by any op body) |
```

Lines 27-31 — replace the "Callee-saved" entry to reflect the new role:
```
//! - **Callee-saved**: `%rbx`, `%rbp`, `%r12`, `%r13`, `%r14`, `%r15`.
//!   This profile saves `%rbp` unconditionally in the prologue (frame
//!   pointer role); the body is free to clobber it as scratch since no
//!   op-emitter reads `%rbp`. The other 5 are saved by the function-level
//!   prologue when `compute_callee_saved` returns true (= `model.calls_extern_math() OR has_matmul(model)`,
//!   per `buffer.rs`).
```

Lines 22-26 — update the M12 caps rationale to mention M13 closing the gap:
```
//! - **ABI argument** (one role each per arity): `%rdi`, `%rsi`,
//!   `%rdx`, `%rcx`, `%r8`, `%r9` — first 6 used. M12 capped N at 4
//!   (arity check in walk_model), so N+2 ≤ 6 — register-only. M13
//!   closed the N=4 + matmul subcase by relocating the j-counter from
//!   `%r9` (which becomes output_reg at N=4) to `%rbp`.
```

Lines 128-139 — the entire `if abi.n_inputs == 4 { return Err(...) }` block is gone (deleted in Step 1.4); the doc comment immediately above it should also be removed.

- [ ] **Step 1.6: Run the new test to verify it passes**

Run: `cargo test -p profiles-x86_64 --lib emit_matmul_accepts_n4_with_rbp_j_counter`
Expected: PASS.

- [ ] **Step 1.7: Run the full x86_64 test suite to verify no regression**

Run: `cargo test -p profiles-x86_64`
Expected: all tests pass. Confirm `emit_matmul_body_contains_zero_pushq` still passes (no new pushq in body).

- [ ] **Step 1.8: Run workspace gates**

Run in parallel:
- `cargo build --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`

Expected: all clean.

- [ ] **Step 1.9: Commit Group A**

```bash
git add profiles/x86_64/src/ops/matmul.rs profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
fix(m13): x86_64 emit_matmul N=4 + matmul gap — j-counter relocates to %rbp

Closes the M12→M13 priority signal: x86_64 emit_matmul previously
returned Err(LowerError::UnsupportedOp) at N=4 because the inner
j-loop counter was hardcoded to %r9, which becomes output_reg() at
N=4 (INPUT_REGS[5]). The first j-iter would clobber the output
pointer with the j index, producing silently wrong asm.

M13 fix (option D, beyond spec §3.3 enumeration): relocate j-counter
to %rbp. Justification:
- %rbp is callee-saved by the unconditional `pushq %rbp` in
  asm.rs::format_function_prologue; the body's clobber is restored
  by the existing epilogue's `popq %rbp`.
- No op-emitter reads %rbp inside the function body (grep verified).
- Satisfies all four §3.2 constraints: no new callee-saved
  (%rbp already saved); ABI-clean body (%rbp is not an ABI register);
  no new pushq inside body (the emit_matmul_body_contains_zero_pushq
  invariant holds); no collision with existing M12 register layout.

Test transition: emit_matmul_rejects_n4_with_clear_error → renamed
to emit_matmul_accepts_n4_with_rbp_j_counter; pins the new %rbp
slot in emitted asm, asserts %r9 no longer self-clobbered.

End-to-end FFI coverage of N=4 + matmul lands in Group E (Task 5)
via tests/fixtures/four_input_matmul.nfl per spec §6.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Group B — `StdOp::Add` foundation

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` — six edits (one per arm in `StdOp` enum, `resolve`, `signature`, `infer_output_shape`, `validate_attrs`, `Display`, plus new `ShapeError` variant).
- Modify: `compiler/src/ir/tests.rs` (or wherever StdOp builder tests live — locate via grep below).

**No code change needed in:** `compiler/src/parser/` (the existing positional-arg handler resolves identifiers via `ArgType::Tensor`, exactly as it does for `matmul[other]`).

**Walk_model dispatch:** Both profiles' `codegen.rs::walk_model` will receive `StdOp::Add` and have no match arm. Behavior at end of Task 2: `walk_model` returns `Err(LowerError::UnsupportedOp)` for `StdOp::Add` via the existing default handling (or we explicitly add a placeholder arm). Tasks 3 and 4 replace the placeholder with calls to `emit_add`.

- [ ] **Step 2.1: Locate StdOp builder tests**

Run: `grep -rn "StdOp::Matmul\|StdOp::MulScalar" compiler/src/ir/tests.rs compiler/src/ir/build.rs 2>/dev/null | head -20`
Expected: list of test sites and build-side dispatch sites.

Run: `grep -rn "infer_matmul_shape\|matmul_transpose_b" compiler/src/ir/ 2>/dev/null | head -10`
Expected: confirms shape inference helpers location.

- [ ] **Step 2.2: Write failing test for `StdOp::Add` build**

Append to `compiler/src/ir/tests.rs` (or create a new test module if no such file exists; structure mirrors existing `StdOp::Matmul` tests):

```rust
#[test]
fn build_add_op_two_input_model() {
    // M13 Group B: minimal positive case.
    // model AddDemo: x: Tensor[2, 4], skip: Tensor[2, 4]; x -> add[skip]
    let src = "model AddDemo:\n    x: Tensor[2, 4]\n    skip: Tensor[2, 4]\n\n    x -> add[skip]\n";
    let nfl = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let model = &uir.models[0];
    // Expect: 2 Input nodes (x, skip) + 1 Op node (add) = 3 nodes.
    assert_eq!(model.nodes.len(), 3, "expected 3 nodes");
    // Output node is the Add.
    let add_node = &model.nodes[model.output];
    use compiler::ir::stdlib::StdOp;
    use compiler::NodeKind;
    let NodeKind::Op { op, operands, .. } = &add_node.kind else {
        panic!("output is not an Op: {:?}", add_node.kind);
    };
    assert_eq!(*op, StdOp::Add, "output op must be StdOp::Add");
    assert_eq!(operands.len(), 2, "Add must have 2 operands");
    // Output shape preserved.
    assert_eq!(add_node.ty.shape.0, vec![2, 4]);
}

#[test]
fn build_add_op_rejects_shape_mismatch() {
    // M13 Group B: strict shape equality — no broadcasting.
    let src = "model BadAdd:\n    x: Tensor[2, 4]\n    skip: Tensor[2, 8]\n\n    x -> add[skip]\n";
    let nfl = compiler::parse(src).expect("parse");
    let result = compiler::ir::build(&nfl);
    let err = result.expect_err("expected build error for mismatched shapes");
    // The error should wrap ShapeError::AddShapeMismatch.
    let msg = format!("{err:?}");
    assert!(
        msg.contains("AddShapeMismatch"),
        "expected AddShapeMismatch error; got {msg}"
    );
}
```

- [ ] **Step 2.3: Run the failing tests**

Run: `cargo test -p compiler --lib build_add_op_two_input_model build_add_op_rejects_shape_mismatch`
Expected: FAIL — `add` is not a recognized op (`resolve("add")` returns `None`, parser/builder rejects with "unknown op").

- [ ] **Step 2.4: Add `StdOp::Add` to the enum**

In `compiler/src/ir/stdlib.rs`, modify the `StdOp` enum (line 11):

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    /// Matrix multiplication, rank ≥ 2 inputs. With `transpose_b=true`
    /// (named arg), the second operand's last two dims are interpreted
    /// transposed. New in M10.
    Matmul,
    /// Per-element multiply by a scalar literal. Shape is preserved.
    /// Scalar lives in `attrs` as an `AttrValue::Float(f64)`; codegen
    /// truncates to f32 at lowering time. New in M10.
    MulScalar,
    /// Per-element tensor addition. Two tensor operands, strict shape
    /// equality (no broadcasting per design principle #1). Shape is
    /// preserved. New in M13 — first A2 brick (residual connections).
    Add,
}
```

- [ ] **Step 2.5: Add `ShapeError::AddShapeMismatch`**

In `compiler/src/ir/stdlib.rs`, modify the `ShapeError` enum (line 51), append the new variant:

```rust
    /// Two `add` operands have different shapes. Strict equality required —
    /// no broadcasting per design principle #1. New in M13.
    AddShapeMismatch {
        expected: Shape,
        got: Shape,
    },
```

In the `Display for ShapeError` impl (line 95), append the new arm:

```rust
            ShapeError::AddShapeMismatch { expected, got } => write!(
                f,
                "add operand shape mismatch: expected {}, got {} (no broadcasting)",
                expected, got
            ),
```

- [ ] **Step 2.6: Add `resolve("add")`**

In `compiler/src/ir/stdlib.rs::resolve` (line 139), add the new arm before `_`:

```rust
pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear" => Some(StdOp::Linear),
        "relu" => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
        "matmul" => Some(StdOp::Matmul),
        "mul_scalar" => Some(StdOp::MulScalar),
        "add" => Some(StdOp::Add),
        _ => None,
    }
}
```

- [ ] **Step 2.7: Add `signature(StdOp::Add)`**

In `compiler/src/ir/stdlib.rs::signature` (line 151), add the new arm before the closing brace:

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

- [ ] **Step 2.8: Add `infer_output_shape(StdOp::Add)`**

In `compiler/src/ir/stdlib.rs::infer_output_shape` (line 205), add the new arm before the closing brace:

```rust
        StdOp::Add => {
            if inputs.len() != 2 {
                return Err(ShapeError::WrongInputCount {
                    expected: 2,
                    actual: inputs.len(),
                });
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

- [ ] **Step 2.9: Add `validate_attrs(StdOp::Add)`**

In `compiler/src/ir/stdlib.rs::validate_attrs` (line 377), update the `Linear | Relu | ... | MulScalar => Ok(())` arm to include `Add`:

```rust
        StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul | StdOp::MulScalar | StdOp::Add => Ok(()),
```

- [ ] **Step 2.10: Add `Display for StdOp` arm**

In `compiler/src/ir/stdlib.rs::Display for StdOp` (line 406), add the new arm:

```rust
        let name = match self {
            StdOp::Linear => "linear",
            StdOp::Relu => "relu",
            StdOp::Dropout => "dropout",
            StdOp::Softmax => "softmax",
            StdOp::Matmul => "matmul",
            StdOp::MulScalar => "mul_scalar",
            StdOp::Add => "add",
        };
```

- [ ] **Step 2.11: Run the new builder tests to verify they pass**

Run: `cargo test -p compiler --lib build_add_op_two_input_model build_add_op_rejects_shape_mismatch`
Expected: PASS.

- [ ] **Step 2.12: Run the full compiler test suite to confirm no regression**

Run: `cargo test -p compiler`
Expected: all pass. The new `StdOp::Add` variant is exhaustively matched in `infer_output_shape`, `validate_attrs`, `signature`, `resolve`, `Display` (5 sites + 2 new tests).

- [ ] **Step 2.13: Run workspace gates**

Run in parallel:
- `cargo build --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`

Expected: all clean. **Note:** profile crates may emit a clippy warning for non-exhaustive `StdOp` match if `walk_model` doesn't have an arm for `StdOp::Add`. If so, add a placeholder arm in both `profiles/{arm64,x86_64}/src/codegen.rs::classify_op` and `walk_model`:

```rust
// In classify_op (around the StdOp::Matmul / StdOp::MulScalar arms):
StdOp::Add => Ok(()), // M13: codegen lands in Task 3 (arm64) / Task 4 (x86_64)

// In walk_model match (around the existing op arms):
StdOp::Add => {
    return Err(LowerError::UnsupportedOp {
        op: "add (M13 codegen pending — Task 3/4)".into(),
        span: node.source_span,
    });
}
```

These placeholders get replaced in Tasks 3 and 4.

- [ ] **Step 2.14: Commit Group B**

```bash
git add compiler/src/ir/stdlib.rs compiler/src/ir/tests.rs profiles/arm64/src/codegen.rs profiles/x86_64/src/codegen.rs
git commit -m "$(cat <<'EOF'
feat(m13): StdOp::Add foundation — first A2 brick (no codegen yet)

NFL surface: `a -> add[skip]`. Positional Tensor arg "other",
mirrors Matmul. First real consumer of M10's ArgType::Tensor
machinery outside Matmul (StdOp::MulScalar uses Float, StdOp::Add
is the first elementwise binary op).

Flat StdOp::Add variant — no BinaryOp container per spec §4 / Q4.
Three-strikes-then-refactor: when sub/mul/div elementwise siblings
appear (M14+ LayerNorm/FFN may not need them; LayerNorm likely
ships as composite StdOp::LayerNorm), revisit grouping.

Shape semantics: strict equality. New ShapeError::AddShapeMismatch
{expected, got} (no Span — matches the 7 existing ShapeError
variants; Span addition is M5c OQ-4 trigger, not fired in M13).

walk_model dispatch placeholder: StdOp::Add returns
LowerError::UnsupportedOp on both profiles. Replaced with real
emit_add calls in Group C (arm64, Task 3) and Group D (x86_64,
Task 4).

Tests: build_add_op_two_input_model + build_add_op_rejects_shape_mismatch
(parser/builder coverage). FFI integration in Group E (Task 5).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Group C — arm64 `emit_add`

**Files:**
- Create: `profiles/arm64/src/ops/add.rs`
- Modify: `profiles/arm64/src/ops/mod.rs` (add `pub mod add;`)
- Modify: `profiles/arm64/src/codegen.rs` (add `walk_model` dispatch + per-model `add_idx` counter)
- Modify: `profiles/arm64/src/tests.rs` (unit tests for emit_add asm shape)

- [ ] **Step 3.1: Read template — `emit_mulscalar` arm64**

Run: `cat profiles/arm64/src/ops/mulscalar.rs`
Expected: confirm the flat-loop pattern (movz/movk/fmov scalar pre-load + ldr/fmul/str inner loop). emit_add will follow the same loop shell minus the scalar pre-load, plus a second materialise_ptr call for `other`.

- [ ] **Step 3.2: Write failing unit tests for `emit_add` arm64**

Append to `profiles/arm64/src/tests.rs`:

```rust
// ---- M13 Group C: emit_add arm64 -----------------------------------------

#[test]
fn emit_add_arm64_emits_three_pointer_loads_and_fadd() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 2 };
    let asm = crate::ops::add::emit_add(
        &abi,
        /* total_elements */ 8,
        /* model_idx */ 0,
        /* op_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* other_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
    );
    // Three pointers materialised.
    assert!(asm.contains("ldr     s0,"), "expected ldr s0 (a load); got:\n{asm}");
    assert!(asm.contains("ldr     s1,"), "expected ldr s1 (other load); got:\n{asm}");
    assert!(asm.contains("fadd    s2, s0, s1"), "expected fadd s2, s0, s1; got:\n{asm}");
    assert!(asm.contains("str     s2,"), "expected str s2 (dst store); got:\n{asm}");
}

#[test]
fn emit_add_arm64_no_callee_saved_or_ffi_save() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 2 };
    let asm = crate::ops::add::emit_add(
        &abi,
        16,
        0, 0,
        BufferLoc::InputReg(0),
        BufferLoc::InputReg(1),
        BufferLoc::OutputReg,
    );
    // No callee-saved register pushes (x19-x28 / d8-d15).
    for reg in &["x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"] {
        assert!(
            !asm.contains(&format!("str     {reg}")),
            "emit_add must not push callee-saved {reg}; got:\n{asm}"
        );
    }
    // No bl _expf (no FFI save needed).
    assert!(!asm.contains("bl      _expf"), "emit_add must not call _expf; got:\n{asm}");
}
```

- [ ] **Step 3.3: Run tests to verify they fail (no add.rs exists yet)**

Run: `cargo test -p profiles-arm64 --lib emit_add_arm64_emits_three_pointer_loads_and_fadd emit_add_arm64_no_callee_saved_or_ffi_save`
Expected: FAIL — `crate::ops::add` does not exist.

- [ ] **Step 3.4: Create `profiles/arm64/src/ops/add.rs`**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Add codegen — flat per-element tensor addition: dst[i] = a[i] + other[i].
//!
//! Closest existing template: `emit_mulscalar` (M10). Same flat loop
//! shell; differs in that `add` reads two input pointers (a, other)
//! instead of one input + a pre-loaded scalar.
//!
//! No FFI save/restore (no `bl _expf` call). No callee-saved register
//! usage. Three caller-saved scratch GPRs (x9/x10/x11) for the three
//! materialised pointers, x12 for the loop counter, x13 for the
//! immediate bound — all non-ABI on AAPCS64 for any N ≤ 4.
//!
//! M13 — first A2 brick (residual connections).

use crate::abi::AbiContext;
use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;

/// Emit AArch64 asm for `dst[i] = a[i] + other[i]` over `total_elements`.
#[allow(clippy::too_many_arguments)]
pub fn emit_add(
    abi: &AbiContext,
    total_elements: u64,
    model_idx: usize,
    op_idx: usize,
    a_loc: BufferLoc,
    other_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; add: total_elements={}\n",
        total_elements
    ));

    // Materialise three pointers. x9 = a, x10 = other, x11 = dst.
    abi.materialise_ptr(a_loc, "x9", &mut s);
    abi.materialise_ptr(other_loc, "x10", &mut s);
    abi.materialise_ptr(dst_loc, "x11", &mut s);

    // Counter x12 = 0; bound x13 = total_elements.
    s.push_str("    mov     x12, #0\n");
    s.push_str(&emit_imm32("x13", total_elements as usize));

    s.push_str(&format!(".Ladd_{mid}:\n"));
    s.push_str("    cmp     x12, x13\n");
    s.push_str(&format!("    b.ge    .Ladd_end_{mid}\n"));

    s.push_str("    ldr     s0, [x9, x12, lsl #2]\n");
    s.push_str("    ldr     s1, [x10, x12, lsl #2]\n");
    s.push_str("    fadd    s2, s0, s1\n");
    s.push_str("    str     s2, [x11, x12, lsl #2]\n");

    s.push_str("    add     x12, x12, #1\n");
    s.push_str(&format!("    b       .Ladd_{mid}\n"));
    s.push_str(&format!(".Ladd_end_{mid}:\n"));

    s
}
```

- [ ] **Step 3.5: Register the module in `profiles/arm64/src/ops/mod.rs`**

Add `pub mod add;` in alphabetical order (before `dropout`):

```rust
pub mod add;
pub mod dropout;
pub mod linear;
pub mod matmul;
pub mod mulscalar;
pub mod relu;
pub mod softmax;
```

- [ ] **Step 3.6: Wire `emit_add` into `walk_model` (arm64)**

In `profiles/arm64/src/codegen.rs::walk_model`, locate the per-op counter declarations (around line 148 — `let mut linear_idx = 0usize;` etc.). Add:

```rust
let mut add_idx = 0usize;
```

In the `match op { ... }` block (around `StdOp::MulScalar => { ... }`), add the new arm:

```rust
                StdOp::Add => {
                    let total_elements: u64 = node.ty.shape.0.iter().product();
                    let a_loc = assignment.locs[operands[0]];
                    let other_loc = assignment.locs[operands[1]];
                    let dst_loc = assignment.locs[node_idx];
                    body.push_str(&crate::ops::add::emit_add(
                        &abi,
                        total_elements,
                        model_idx,
                        add_idx,
                        a_loc,
                        other_loc,
                        dst_loc,
                    ));
                    add_idx += 1;
                }
```

Replace any Group B placeholder (`StdOp::Add => return Err(...)`) with the above.

- [ ] **Step 3.7: Run the unit tests to verify they pass**

Run: `cargo test -p profiles-arm64 --lib emit_add_arm64_emits_three_pointer_loads_and_fadd emit_add_arm64_no_callee_saved_or_ffi_save`
Expected: PASS.

- [ ] **Step 3.8: Run the full arm64 test suite + workspace gates**

Run in parallel:
- `cargo test -p profiles-arm64`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`

Expected: all clean.

- [ ] **Step 3.9: Commit Group C**

```bash
git add profiles/arm64/src/ops/add.rs profiles/arm64/src/ops/mod.rs profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m13): arm64 emit_add — flat elementwise tensor addition

New profiles/arm64/src/ops/add.rs::emit_add. Flat AArch64 loop:
  ldr s0, [a_ptr, idx, lsl #2]
  ldr s1, [other_ptr, idx, lsl #2]
  fadd s2, s0, s1
  str s2, [dst_ptr, idx, lsl #2]

Closest template: emit_mulscalar (M10). Differs in second
materialise_ptr call (other instead of pre-loaded scalar).

Register budget: x9/x10/x11 for the three materialised pointers
(caller-saved scratch, non-ABI for N≤4), x12 counter, x13
immediate bound. No callee-saved usage. No FFI save/restore.

walk_model dispatch wired in profiles/arm64/src/codegen.rs;
StdOp::Add Group B placeholder replaced with real emit_add call.

Per-emit unit tests verify: three pointer loads + fadd + str
present; no callee-saved pushes; no `bl _expf`.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Group D — x86_64 `emit_add`

**Files:**
- Create: `profiles/x86_64/src/ops/add.rs`
- Modify: `profiles/x86_64/src/ops/mod.rs` (add `pub mod add;`)
- Modify: `profiles/x86_64/src/codegen.rs` (walk_model dispatch + add_idx counter)
- Modify: `profiles/x86_64/src/tests.rs` (unit tests for emit_add asm shape)

**Strategy — loop counter:** Per spec §5.4, the intersection of free non-ABI scratch GPRs across N ∈ [1,4] is `{%rax, %r10, %r11}` — exactly the materialised pointer set. **Plan-synthesis pick: use `%rbp` as the loop counter** (same trick as Task 1's matmul fix). `%rbp` is callee-saved by the function-level prologue and unread by any op body. Zero extra instructions overhead, no save/restore, no stack-slot allocation.

This pick replaces spec §5.4's three options (a/b/c) and §9 item 2's open question with the same elegant solution Task 1 commits to. The plan effectively chooses option **D** (use the already-saved `%rbp` as universal scratch counter) for all M13 codegen.

- [ ] **Step 4.1: Read template — `emit_mulscalar` x86_64**

Run: `cat profiles/x86_64/src/ops/mulscalar.rs`
Expected: confirm the flat AT&T-syntax loop (movl scalar bits + movd into %xmm4 + flat movss/mulss/movss inner loop). emit_add will follow the same loop shell minus the scalar pre-load, plus a second materialise_ptr call for `other`, with `%rbp` replacing `%rcx` as the counter.

- [ ] **Step 4.2: Write failing unit tests for `emit_add` x86_64**

Append to `profiles/x86_64/src/tests.rs`:

```rust
// ---- M13 Group D: emit_add x86_64 ----------------------------------------

#[test]
fn emit_add_x86_64_emits_two_loads_one_addss_one_store() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 2 };
    let asm = crate::ops::add::emit_add(
        &abi,
        /* total_elements */ 8,
        /* model_idx */ 0,
        /* op_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* other_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
    );
    // Two scalar loads (one per input pointer).
    assert!(asm.contains("movss   (%rax,"), "expected movss from %rax (a_ptr); got:\n{asm}");
    assert!(asm.contains("movss   (%r10,"), "expected movss from %r10 (other_ptr); got:\n{asm}");
    // One addss.
    assert_eq!(
        asm.matches("addss").count(), 1,
        "expected exactly one addss; got:\n{asm}"
    );
    // One movss store to %r11.
    assert!(
        asm.contains("movss   %xmm0, (%r11,"),
        "expected movss store to %r11; got:\n{asm}"
    );
}

#[test]
fn emit_add_x86_64_uses_rbp_counter_no_pushq_no_rcx_clobber() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    // At N=2, %rcx = output_reg. emit_add must NOT clobber %rcx.
    let abi = AbiContext { n_inputs: 2 };
    let asm = crate::ops::add::emit_add(
        &abi,
        16,
        0, 0,
        BufferLoc::InputReg(0),
        BufferLoc::InputReg(1),
        BufferLoc::OutputReg,
    );
    // Counter init goes to %rbp.
    assert!(
        asm.contains("movq    $0, %rbp\n"),
        "expected counter init in %rbp; got:\n{asm}"
    );
    // No pushq/popq — %rbp is already saved by function-level prologue.
    assert!(
        !asm.contains("pushq"),
        "emit_add must not pushq inside body (rbp is preserved by prologue); got:\n{asm}"
    );
    assert!(
        !asm.contains("popq"),
        "emit_add must not popq inside body; got:\n{asm}"
    );
    // %rcx (ABI at N=2) is not written.
    assert!(
        !asm.contains(", %rcx\n"),
        "emit_add must not write to %rcx (ABI register at N≥2); got:\n{asm}"
    );
}

#[test]
fn emit_add_x86_64_no_callee_saved_or_ffi_save() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 1 };
    let asm = crate::ops::add::emit_add(
        &abi,
        4,
        0, 0,
        BufferLoc::InputReg(0),
        BufferLoc::InputReg(0),
        BufferLoc::OutputReg,
    );
    // No call to expf@PLT (no FFI save needed inside emit_add).
    assert!(!asm.contains("call    expf@PLT"), "emit_add must not call expf; got:\n{asm}");
    // No %rbx/%r12-%r15 writes (matmul-only callee-saved set).
    for reg in &["%rbx", "%r12", "%r13", "%r14", "%r15"] {
        assert!(
            !asm.contains(&format!(", {reg}\n")),
            "emit_add must not write to callee-saved {reg}; got:\n{asm}"
        );
    }
}
```

- [ ] **Step 4.3: Run tests to verify they fail (no add.rs yet)**

Run: `cargo test -p profiles-x86_64 --lib emit_add_x86_64_emits_two_loads_one_addss_one_store emit_add_x86_64_uses_rbp_counter_no_pushq_no_rcx_clobber emit_add_x86_64_no_callee_saved_or_ffi_save`
Expected: FAIL — `crate::ops::add` does not exist.

- [ ] **Step 4.4: Create `profiles/x86_64/src/ops/add.rs`**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Add codegen — x86_64 SSE2 AT&T-syntax flat per-element tensor addition.
//!
//!   dst[i] = a[i] + other[i]
//!
//! Closest existing template: `emit_mulscalar` (M10). Same flat loop
//! shell; differs in that `add` reads two input pointers (a, other)
//! instead of one input + a pre-loaded scalar, and uses `addss` instead
//! of `mulss`.
//!
//! ## Register budget (M13 spec §5.4)
//!
//! The intersection of free non-ABI scratch GPRs across N ∈ [1,4] is
//! {%rax, %r10, %r11} — exactly enough for the three materialised
//! pointers, leaving zero spare GPR for a counter. Plan-synthesis pick:
//! use `%rbp` as the counter. `%rbp` is callee-saved by the unconditional
//! `pushq %rbp` in `asm.rs::format_function_prologue`, and is read by
//! zero op-emitter bodies (verified at M13 plan synthesis). Inside the
//! function body, `%rbp` is wide-open scratch.
//!
//! This is the same trick used by M13 Group A (Task 1) for the matmul
//! j-counter at N=4. Both choices share the rationale: the prologue
//! already saves `%rbp`, no per-op save/restore needed.
//!
//! No FFI save/restore (no `bl _expf` / `call expf@PLT`). No additional
//! callee-saved register usage beyond `%rbp` (already saved).

use crate::abi::AbiContext;
use crate::buffer::BufferLoc;

/// Emit AT&T x86_64 asm for `dst[i] = a[i] + other[i]` over `total_elements`.
#[allow(clippy::too_many_arguments)]
pub fn emit_add(
    abi: &AbiContext,
    total_elements: u64,
    model_idx: usize,
    op_idx: usize,
    a_loc: BufferLoc,
    other_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # add: total_elements={}\n",
        total_elements
    ));

    // Materialise three pointers. %rax = a, %r10 = other, %r11 = dst.
    abi.materialise_ptr(a_loc, "%rax", &mut s);
    abi.materialise_ptr(other_loc, "%r10", &mut s);
    abi.materialise_ptr(dst_loc, "%r11", &mut s);

    // Loop counter %rbp = 0. (%rbp is callee-saved by the function-level
    // prologue; the body is free to clobber it.)
    s.push_str("    movq    $0, %rbp\n");
    s.push_str(&format!(".Ladd_{mid}:\n"));
    // cmpq with sign-extended 32-bit immediate. total_elements fits in
    // i32 for any practical NN size (max ~2^31 elements = ~8 GiB tensor).
    s.push_str(&format!("    cmpq    ${}, %rbp\n", total_elements));
    s.push_str(&format!("    jge     .Ladd_end_{mid}\n"));

    s.push_str("    movss   (%rax, %rbp, 4), %xmm0\n");
    s.push_str("    movss   (%r10, %rbp, 4), %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rbp, 4)\n");

    s.push_str("    addq    $1, %rbp\n");
    s.push_str(&format!("    jmp     .Ladd_{mid}\n"));
    s.push_str(&format!(".Ladd_end_{mid}:\n"));

    s
}
```

- [ ] **Step 4.5: Register the module in `profiles/x86_64/src/ops/mod.rs`**

Add `pub mod add;` in alphabetical order:

```rust
pub mod add;
pub mod dropout;
pub mod linear;
pub mod matmul;
pub mod mulscalar;
pub mod relu;
pub mod softmax;
```

- [ ] **Step 4.6: Wire `emit_add` into `walk_model` (x86_64)**

In `profiles/x86_64/src/codegen.rs::walk_model`, locate the per-op counter declarations (around line 148). Add:

```rust
let mut add_idx = 0usize;
```

In the `match op { ... }` block, add (or replace the Task 2 placeholder):

```rust
                StdOp::Add => {
                    let total_elements: u64 = node.ty.shape.0.iter().product();
                    let a_loc = assignment.locs[operands[0]];
                    let other_loc = assignment.locs[operands[1]];
                    let dst_loc = assignment.locs[node_idx];
                    body.push_str(&crate::ops::add::emit_add(
                        &abi,
                        total_elements,
                        model_idx,
                        add_idx,
                        a_loc,
                        other_loc,
                        dst_loc,
                    ));
                    add_idx += 1;
                }
```

- [ ] **Step 4.7: Run the unit tests to verify they pass**

Run: `cargo test -p profiles-x86_64 --lib emit_add_x86_64_emits_two_loads_one_addss_one_store emit_add_x86_64_uses_rbp_counter_no_pushq_no_rcx_clobber emit_add_x86_64_no_callee_saved_or_ffi_save`
Expected: PASS.

- [ ] **Step 4.8: Run the full x86_64 test suite + workspace gates**

Run in parallel:
- `cargo test -p profiles-x86_64`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`

Expected: all clean. Confirm `emit_matmul_body_contains_zero_pushq` (Task 1 invariant) still holds.

- [ ] **Step 4.9: Commit Group D**

```bash
git add profiles/x86_64/src/ops/add.rs profiles/x86_64/src/ops/mod.rs profiles/x86_64/src/codegen.rs profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m13): x86_64 emit_add — flat elementwise tensor addition with %rbp counter

New profiles/x86_64/src/ops/add.rs::emit_add. Flat AT&T loop:
  movss (%rax, %rbp, 4), %xmm0
  movss (%r10, %rbp, 4), %xmm1
  addss %xmm1, %xmm0
  movss %xmm0, (%r11, %rbp, 4)

Register budget: %rax/%r10/%r11 for materialised a/other/dst
pointers (intersection of free non-ABI scratch across N ∈ [1,4]
per spec §5.4). %rbp as loop counter — same trick as Task 1's
matmul j-counter fix: %rbp is callee-saved by the unconditional
prologue `pushq %rbp` and read by zero op bodies. Zero overhead
vs the spec's enumerated options (a/b/c).

Closest template: emit_mulscalar (M10). Differs in second
materialise_ptr call (other) and addss vs mulss.

walk_model dispatch wired; StdOp::Add Task 2 placeholder replaced
with real emit_add call.

Per-emit unit tests verify: two movss loads + one addss + one
store; counter init goes to %rbp; no pushq inside body; no %rcx
clobber (ABI at N≥2); no callee-saved %rbx/%r12-%r15 writes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Group E — Integration fixtures + per-profile FFI tests

**Files:**
- Create: `tests/fixtures/residual_add.nfl`
- Create: `tests/fixtures/four_input_matmul.nfl`
- Create: `tests/fixtures/negative/add_shape_mismatch.nfl`
- Modify: `profiles/arm64/tests/integration.rs` (add `residual_add_match_numerically`)
- Modify: `profiles/x86_64/tests/integration.rs` (add `residual_add_match_numerically` + `four_input_matmul_match_numerically`)

- [ ] **Step 5.1a: Verify linear bias default before writing fixture / reference**

Run: `grep -n "params_floats\|bias" profiles/arm64/tests/integration.rs | head -20`
Run: `grep -A3 'StdOp::Linear =>' compiler/src/ir/stdlib.rs | head -15`

Expected: confirm whether `linear[dim]` (no `bias=true` named arg) emits with bias OFF (default) or ON. As of M12: `linear[dim]` defaults to `bias=false` (per `stdlib.rs::signature`'s Linear arm — `bias` is `Symbol`, optional, `required: false`; `linear_has_bias` reads `bias=true` explicitly). So `params_floats == dim * dim` for the residual_add fixture (16 floats for dim=4), and the reference is `relu(x @ W) + skip` with NO bias term.

If grep reveals the default has changed (e.g. some prior milestone made `bias=true` the default), adjust the fixture's `linear[dim]` to `linear[dim, bias=false]`, the assertion to `assert_eq!(sig.params_floats, dim * dim, "...")`, and the reference to NOT include bias. Conversely, if the fixture intentionally uses bias, write `linear[dim, bias=true]`, assert `dim * dim + dim`, and add `+ b[j]` in the reference.

- [ ] **Step 5.1b: Write `tests/fixtures/residual_add.nfl`**

```
model ResidualBlock [batch=2, dim=4]:
    x: Tensor[batch, dim]
    skip: Tensor[batch, dim]

    x -> linear[dim] -> relu -> add[skip]
```

- [ ] **Step 5.2: Write `tests/fixtures/four_input_matmul.nfl`**

```
model FourInputMatmul [m=4, k=8, n=4]:
    a: Tensor[m, k]
    b: Tensor[k, n]
    c: Tensor[m, n]
    d: Tensor[m, n]

    a -> matmul[b] -> add[c] -> add[d]
```

This exercises both N=4 ABI mapping AND a matmul op (the bug surface). The `add[c] -> add[d]` chain also exercises emit_add at N=4 for free, validating Task 4 in a multi-input context.

- [ ] **Step 5.3: Write `tests/fixtures/negative/add_shape_mismatch.nfl`**

```
model BadAdd [batch=2]:
    x: Tensor[batch, 4]
    skip: Tensor[batch, 8]

    x -> add[skip]
```

This builds successfully through parser, fails at IR shape inference with `ShapeError::AddShapeMismatch { expected: Shape([2, 4]), got: Shape([2, 8]) }`. Picked up automatically by `compiler/tests/negative_fixtures.rs::all_negative_fixtures_reject` (which loops over `tests/fixtures/negative/` and asserts SOME error fires).

- [ ] **Step 5.4: Verify the negative fixture is auto-discovered**

Run: `cargo test -p compiler --test negative_fixtures all_negative_fixtures_reject`
Expected: PASS — the new `add_shape_mismatch.nfl` is included in the loop and asserted to error.

- [ ] **Step 5.5: Add `residual_add_match_numerically` to arm64 integration tests**

Append to `profiles/arm64/tests/integration.rs`:

```rust
// ---- M13 Group E: residual_add (StdOp::Add end-to-end) ------------------

#[test]
fn residual_add_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/residual_add.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 2, "residual_add has arity 2");
    assert_eq!(sig.inputs_floats[0], 2 * 4, "x is [2,4]=8 floats");
    assert_eq!(sig.inputs_floats[1], 2 * 4, "skip is [2,4]=8 floats");
    assert_eq!(sig.params_floats, 4 * 4, "linear weights only, 4x4=16 floats");

    let so_path = common::compile_to_dylib(&asm.source, "residual_add");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_ResidualBlock") }.expect("dlsym");

    let batch = 2usize;
    let dim = 4usize;
    let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.1).collect();
    let skip: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.07).collect();
    // Linear weights: 4x4 row-major.
    let weights: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.05).collect();
    let mut out = vec![0.0f32; batch * dim];

    unsafe {
        forward(x.as_ptr(), skip.as_ptr(), weights.as_ptr(), out.as_mut_ptr());
    }

    // Reference: relu(x @ W) + skip element-wise.
    let mut expected = vec![0.0f32; batch * dim];
    for b in 0..batch {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for kk in 0..dim {
                sum += x[b * dim + kk] * weights[kk * dim + j];
            }
            // ReLU
            let relu_out = if sum > 0.0 { sum } else { 0.0 };
            // Add skip
            expected[b * dim + j] = relu_out + skip[b * dim + j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}
```

**Note on helper name:** the arm64 integration test crate uses `compile_to_dylib` (per existing tests). Verify by `grep -n "compile_to_dylib\|compile_to_so" profiles/arm64/tests/common/mod.rs` and use whatever name matches (arm64 = `compile_to_dylib`, x86_64 = `compile_to_so` per spec §10's M9 profile-isolation pattern).

- [ ] **Step 5.6: Add `residual_add_match_numerically` to x86_64 integration tests**

Append to `profiles/x86_64/tests/integration.rs` — same as Step 5.5 but using `profiles_x86_64::lower`, the `compile_to_so` helper, and the SysV calling convention comment:

```rust
#[test]
fn residual_add_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/residual_add.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 2);
    assert_eq!(sig.params_floats, 16);

    let so_path = common::compile_to_so(&asm.source, "residual_add");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: x (%rdi), skip (%rsi), params (%rdx), out (%rcx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_ResidualBlock") }.expect("dlsym");

    let batch = 2usize;
    let dim = 4usize;
    let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.1).collect();
    let skip: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.07).collect();
    let weights: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.05).collect();
    let mut out = vec![0.0f32; batch * dim];

    unsafe {
        forward(x.as_ptr(), skip.as_ptr(), weights.as_ptr(), out.as_mut_ptr());
    }

    let mut expected = vec![0.0f32; batch * dim];
    for b in 0..batch {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for kk in 0..dim {
                sum += x[b * dim + kk] * weights[kk * dim + j];
            }
            let relu_out = if sum > 0.0 { sum } else { 0.0 };
            expected[b * dim + j] = relu_out + skip[b * dim + j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got}, want {want}"
        );
    }

    drop(lib);
}
```

- [ ] **Step 5.7: Add `four_input_matmul_match_numerically` to x86_64 integration tests**

Append to `profiles/x86_64/tests/integration.rs`:

```rust
// ---- M13 Group E: four_input_matmul (closes Group A end-to-end) -----

#[test]
fn four_input_matmul_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/four_input_matmul.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    // Pre-M13 this would fail with LowerError::UnsupportedOp ("matmul at N=4...").
    let asm = profiles_x86_64::lower(&uir).expect("lower (M13 closed N=4 + matmul gap)");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 4, "four_input_matmul has arity 4");
    let m = 4usize;
    let k = 8usize;
    let n = 4usize;
    assert_eq!(sig.inputs_floats[0], m * k);
    assert_eq!(sig.inputs_floats[1], k * n);
    assert_eq!(sig.inputs_floats[2], m * n);
    assert_eq!(sig.inputs_floats[3], m * n);

    let so_path = common::compile_to_so(&asm.source, "four_input_matmul");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: a (%rdi), b (%rsi), c (%rdx), d (%rcx), params (%r8, empty), out (%r9).
    type ForwardFn = unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32,
    );
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_FourInputMatmul") }.expect("dlsym");

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.07).collect();
    let c: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.03).collect();
    let d: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.02).collect();
    let params: Vec<f32> = vec![];
    let mut out = vec![0.0f32; m * n];

    unsafe {
        forward(
            a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr(),
            params.as_ptr(), out.as_mut_ptr(),
        );
    }

    // Reference: (a @ b) + c + d, separate mul + add per emit_matmul.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            expected[i * n + j] = sum + c[i * n + j] + d[i * n + j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}
```

**Note:** this test only runs on Linux x86_64 CI (where `cc` produces ELF) — macOS doesn't ship the x86_64 cross toolchain. The test gracefully early-returns on missing `cc` per the existing `common::cc_available()` pattern.

- [ ] **Step 5.8: Run all integration tests on the host platform**

Run: `cargo test -p profiles-arm64 --test integration residual_add`
Expected: PASS on macOS arm64 host.

Run: `cargo test -p profiles-x86_64 --test integration residual_add four_input_matmul`
Expected: skipped on macOS host (no `cc` for x86_64); PASS on Linux x86_64 CI.

- [ ] **Step 5.9: Run full workspace gates**

Run in parallel:
- `cargo build --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo test --workspace`

Expected: all clean. Test count strictly greater than 390 (M12 baseline). Counting per-group additions:
- Task 1: +1 test (positive emit at N=4 — replacement of negative test)
- Task 2: +2 tests (build_add positive + negative)
- Task 3: +2 tests (arm64 emit_add asm shape)
- Task 4: +3 tests (x86_64 emit_add asm shape + %rbp counter + no clobber)
- Task 5: +3 tests (residual_add arm64, residual_add x86_64, four_input_matmul x86_64) + 1 negative fixture auto-included
Total: ~12 new tests. Expected count: 390 → ~402.

- [ ] **Step 5.10: Commit Group E**

```bash
git add tests/fixtures/residual_add.nfl tests/fixtures/four_input_matmul.nfl tests/fixtures/negative/add_shape_mismatch.nfl profiles/arm64/tests/integration.rs profiles/x86_64/tests/integration.rs
git commit -m "$(cat <<'EOF'
feat(m13): integration fixtures + FFI tests — residual_add + four_input_matmul

Three new fixtures (spec §6):
- tests/fixtures/residual_add.nfl — positive add op end-to-end on
  both profiles (form: linear -> relu -> add[skip]).
- tests/fixtures/four_input_matmul.nfl — closes Group A (Task 1)
  end-to-end on x86_64. Form: a -> matmul[b] -> add[c] -> add[d];
  exercises N=4 ABI mapping AND matmul (the M12 bug surface) AND
  emit_add at N=4 in a single fixture.
- tests/fixtures/negative/add_shape_mismatch.nfl — IR-level
  rejection (ShapeError::AddShapeMismatch). Auto-discovered by
  compiler/tests/negative_fixtures.rs.

Per-profile FFI integration tests:
- residual_add_match_numerically (arm64 + x86_64): cc + dlopen +
  bit-exact comparison vs Rust reference (relu(x @ W) + skip).
- four_input_matmul_match_numerically (x86_64 only — arm64 already
  supported N=4 + matmul at M12 and has no Group A change to
  validate): cc + dlopen + bit-exact comparison vs Rust reference
  ((a @ b) + c + d).

Negative fixture lives under tests/fixtures/negative/ (compiler-
level error path), not profile-negative/ (which is reserved for
LowerError fixtures like too_many_inputs.nfl).

Test count: 390 → ~402 (12 new tests across Tasks 1-5).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Group F — Documentation closure

**Files:**
- Modify: `PROJECT_SPEC.md` (M13 row + Current Status + Strategic Roadmap A2 annotation + remove M12 known-gap line)
- Modify: `CLAUDE.md` (Repository Structure tree + Current Status)
- Modify: `docs/language_reference/grammar.md` (new `add` op stdlib reference)
- Modify: `docs/profile_guide/arm64.md` (M13 ops section)
- Modify: `docs/profile_guide/x86_64.md` (M13 ops section + matmul N=4 fix note)
- Modify: `DEVLOG.md` (M13 entry at top)

- [ ] **Step 6.1: Update `PROJECT_SPEC.md`**

In the milestones table (around line 162), append a new row:

```markdown
| 13 | N=4 + matmul fix + `add` op (A2 first brick) (complete) | x86_64 `emit_matmul` j-counter relocated from `%r9` to `%rbp` (callee-saved by unconditional prologue `pushq %rbp`; read by zero op-emitter bodies). Closes M12 known follow-up: N=4 + matmul now compiles and runs bit-exact. New `StdOp::Add` (flat variant; no BinaryOp container). NFL surface `a -> add[skip]` — first real consumer of M10's `ArgType::Tensor` outside Matmul. Strict shape equality (no broadcasting); new `ShapeError::AddShapeMismatch`. Both profiles ship `emit_add` (flat elementwise loop, modeled after `emit_mulscalar`); x86_64 reuses Task 1's `%rbp` scratch trick as loop counter. Three new fixtures: `residual_add.nfl` (positive both profiles), `four_input_matmul.nfl` (closes Group A end-to-end x86_64), `negative/add_shape_mismatch.nfl` (IR-level reject). Test count: 390 → ~402. |
```

In Current Status (around line 171), replace the M12 paragraph with M13:

```markdown
**Milestone 13 complete. ~402 tests passing on macOS arm64 (~410 on Linux x86_64 CI with x86_64 FFI tests included).** All workspace gates clean (`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `cargo test --workspace`).

M13 closed the M12→M13 priority signal (x86_64 `emit_matmul` rejected N=4 + matmul) and shipped the first A2 brick (`StdOp::Add`). Both fixes share a common register trick: `%rbp` is callee-saved by the unconditional function-level prologue `pushq %rbp` and is read by zero op-emitter bodies, so it serves as wide-open scratch inside the function — used as both the matmul j-counter (Group A) and the `emit_add` loop counter (Group D). This eliminates spec §3.3's three enumerated options (stack slot, `%xmm9`, j-loop restructure) in favor of a fourth simpler option that satisfies all §3.2 constraints. New fixtures: `tests/fixtures/{residual_add,four_input_matmul}.nfl` and `tests/fixtures/negative/add_shape_mismatch.nfl`.

Strategic direction: see §"Strategic Roadmap" — A1 closed in M12, A2 first brick (`add`) closed in M13. A2 LayerNorm + FFN remain in M14+ as separate composite ops (mirroring Softmax-as-one-node precedent). Trigger-driven cleanup items (OQ-7, OQ-8, OQ-9, M5c OQ-4) live in §"Open Questions" / "Trigger-driven cleanup" and stay dormant. OQ-NEW closed in M9 (commit `a08fd24`). OQ-BENCH closed in M11 (commit `e7c29b8`).
```

In Strategic Roadmap §"Axis 2" (around line 199), update the A2 line:

```markdown
- **Axis 2 — modelling depth.** M10 closed the first leg (NFL v0.2 self-attention).
  M12 closed A1 (multi-input ABI). M13 closed the M12→M13 priority signal
  (x86_64 N=4 + matmul gap via `%rbp` j-counter relocation) and shipped
  the first A2 brick (`StdOp::Add`, residual connections). Open follow-ups:
  A2 LayerNorm + FFN (separate composite ops, deferred to M14+),
  A3 — profile-level viewer annotations (per-node footprint, stack frame,
  callee-saved set).
```

Remove the "Known gap" line (was: "x86_64 `emit_matmul` rejects N=4+matmul...") — closed by M13.

- [ ] **Step 6.2: Update `CLAUDE.md`**

In the Repository Structure tree (around line 95), add `add.rs` to both ops dirs in alphabetical position:

```
│   │   │   ├── ops/
│   │   │   │   ├── mod.rs        ← per-op submodule entry + re-exports
│   │   │   │   ├── add.rs        ← emit_add (elementwise tensor add, M13)
│   │   │   │   ├── linear.rs     ← emit_linear (matmul ± bias) + materialise_ptr
```

Same for x86_64 ops dir.

In Current Status (around line 168), bump M12 → M13 with a one-paragraph summary:

```markdown
**Milestone 13 complete. ~402 tests passing on macOS arm64 (~410 on Linux x86_64 CI with x86_64 FFI tests included).** All workspace gates clean (`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `cargo test --workspace`).

M13 closed the x86_64 N=4 + matmul gap (j-counter relocated from `%r9` to `%rbp`) and shipped `StdOp::Add` (first A2 brick — residual connections). NFL surface `a -> add[skip]`. Both profiles ship `emit_add` (flat elementwise loop) reusing the same `%rbp` scratch trick. Three new fixtures.

Strategic direction: see `PROJECT_SPEC.md` §"Strategic Roadmap" — A1 closed M12, A2 first brick (`add`) closed M13; A2 LayerNorm + FFN remain in M14+. Trigger-driven cleanup (OQ-7, OQ-8, OQ-9, M5c OQ-4) stays dormant.
```

Also remove the "Known M12 follow-up" line about the N=4 + matmul rejection — the gap is closed.

- [ ] **Step 6.3: Update `docs/language_reference/grammar.md`**

Locate the stdlib reference section (search for `mul_scalar` reference). Add a new `add` subsection at the same level of detail as `matmul`:

```markdown
### `add[other]`

Per-element tensor addition. Adds the named `other` tensor to the
pipeline carrier element-wise. Strict shape equality required — no
broadcasting (per design principle #1, "Explicit over implicit").

**Signature:** positional `other: Tensor` (required); no named args.

**Shape inference:** input shape is preserved. Both operand shapes
must be exactly equal; otherwise `ShapeError::AddShapeMismatch` fires
at IR build time.

**Example:**

```nfl
model ResidualBlock [batch=32, dim=512]:
    x: Tensor[batch, dim]
    skip: Tensor[batch, dim]

    x -> linear[dim] -> relu -> add[skip]
```

**Codegen:** flat elementwise loop on both `arm64` and `x86_64`
profiles. No FFI dependency. New in M13.
```

- [ ] **Step 6.4: Update `docs/profile_guide/arm64.md`**

Append a new section "M13 ops":

```markdown
## M13 ops

### `emit_add` (`profiles/arm64/src/ops/add.rs`)

Flat elementwise tensor addition: `dst[i] = a[i] + other[i]` over
`total_elements = product(shape)`.

**Register layout:**
- `x9` — `a` pointer (caller-saved scratch, materialised via
  `AbiContext::materialise_ptr`).
- `x10` — `other` pointer (same).
- `x11` — `dst` pointer (same).
- `x12` — loop counter (caller-saved scratch).
- `x13` — total_elements bound (caller-saved scratch).
- `s0`, `s1`, `s2` — load-load-add-store scalar FP registers.

**No callee-saved register usage.** No FFI save/restore (no `bl _expf`).

Inner loop (per iter):
```
ldr     s0, [x9, x12, lsl #2]    ; load a[i]
ldr     s1, [x10, x12, lsl #2]   ; load other[i]
fadd    s2, s0, s1
str     s2, [x11, x12, lsl #2]   ; store dst[i]
add     x12, x12, #1
```

Closest existing template: `emit_mulscalar` (M10). The shell is
identical; `emit_add` reads two input pointers (a, other) where
`emit_mulscalar` reads one input + a pre-loaded scalar in `s4`.
```

- [ ] **Step 6.5: Update `docs/profile_guide/x86_64.md`**

Append the same "M13 ops" section as arm64 (with AT&T syntax + register names) — but with one critical addition explaining the `%rbp` counter trick:

```markdown
## M13 ops

### `emit_add` (`profiles/x86_64/src/ops/add.rs`)

Flat elementwise tensor addition: `dst[i] = a[i] + other[i]` over
`total_elements = product(shape)`.

**Register layout:**
- `%rax` — `a` pointer (caller-saved non-ABI for all N ∈ [1,4],
  materialised via `AbiContext::materialise_ptr`).
- `%r10` — `other` pointer (same).
- `%r11` — `dst` pointer (same).
- `%rbp` — loop counter. **Callee-saved by the function-level
  prologue's unconditional `pushq %rbp`** (asm.rs:66); the body is
  free to clobber it as scratch since no op-emitter reads `%rbp`.
- `%xmm0`, `%xmm1` — load-load-addss-store scalar FP registers.

**Why `%rbp` and not a regular GPR scratch?** The intersection of
free non-ABI scratch GPRs across N ∈ [1,4] is `{%rax, %r10, %r11}` —
exactly the materialised pointer set, leaving zero spare GPR for a
counter. `%rbp` is the simplest exception: it's already saved by the
prologue, restored by the epilogue, and read by zero op bodies.

Inner loop (per iter):
```
movss   (%rax, %rbp, 4), %xmm0   # load a[i]
movss   (%r10, %rbp, 4), %xmm1   # load other[i]
addss   %xmm1, %xmm0
movss   %xmm0, (%r11, %rbp, 4)   # store dst[i]
addq    $1, %rbp
```

### M13 N=4 + matmul fix (note in `emit_matmul`)

M12 (commit 37868e5) capped `emit_matmul` at N≤3 because the inner
j-loop counter was hardcoded to `%r9`, which becomes `output_reg()`
at N=4 (`INPUT_REGS[5]`). M13 (Task 1) relocates the j-counter to
`%rbp` for the same reason `emit_add` uses it: callee-saved by the
prologue, unread by op bodies. The reject path is removed; N=4 +
matmul now compiles and runs bit-exact (verified by
`tests/fixtures/four_input_matmul.nfl`).
```

- [ ] **Step 6.6: Update `DEVLOG.md`**

Prepend a new entry at the top (after the file header):

```markdown
## 2026-05-09 — Milestone 13 closed: N=4 + matmul fix + add op (A2 first brick)

### What was done

- **Group A — N=4 + matmul gap closed on x86_64.** `emit_matmul`'s
  inner j-loop counter relocated from `%r9` (which becomes
  `output_reg()` at N=4) to `%rbp` (callee-saved by unconditional
  prologue `pushq %rbp`; unread by op bodies). The M12 reject path
  removed. Test `emit_matmul_rejects_n4_with_clear_error` flipped
  to `emit_matmul_accepts_n4_with_rbp_j_counter`.
- **Group B — `StdOp::Add` foundation.** Flat StdOp variant + new
  `ShapeError::AddShapeMismatch` (no Span — pattern-consistent with
  the 7 existing variants; M5c OQ-4 not triggered). NFL surface
  `a -> add[skip]` — first real consumer of M10's `ArgType::Tensor`
  outside Matmul. Two builder tests added.
- **Group C — arm64 `emit_add`.** New `profiles/arm64/src/ops/add.rs`.
  Flat AArch64 loop modeled after `emit_mulscalar`. x9/x10/x11
  pointers, x12 counter, x13 bound. No FFI, no callee-saved.
- **Group D — x86_64 `emit_add`.** New `profiles/x86_64/src/ops/add.rs`.
  Flat AT&T loop. %rax/%r10/%r11 pointers, %rbp counter (same trick
  as Group A; the fourth simpler option beyond spec §5.4's
  enumerated a/b/c).
- **Group E — fixtures + FFI tests.** `residual_add.nfl` (positive
  both profiles), `four_input_matmul.nfl` (closes Group A
  end-to-end x86_64), `negative/add_shape_mismatch.nfl` (IR
  reject). Per-profile FFI integration tests bit-exact vs Rust
  reference.
- **Group F — docs.** PROJECT_SPEC.md M13 row + Current Status +
  Strategic Roadmap A2 annotation. CLAUDE.md tree + status.
  grammar.md `add` reference. profile_guide/{arm64,x86_64}.md M13
  ops sections.
- **Test count: 390 → ~402** (macOS arm64); ~410 on Linux x86_64 CI.

### Decisions made

- **`%rbp` over spec §3.3 enumerated options** for both Group A
  (matmul j-counter) and Group D (emit_add counter). The spec §3.3
  enumerated stack slot / `%xmm9` / loop restructure; plan synthesis
  discovered a fourth simpler option (`%rbp`) satisfying all four
  §3.2 constraints with zero prologue surface change. Rationale:
  `%rbp` is already saved/restored by the unconditional prologue
  `pushq %rbp` / epilogue `popq %rbp`, and grep across all op
  emitters confirmed zero reads of `%rbp` inside function bodies.
  Both Group A and Group D use the same trick — symmetric design.

- **Negative fixture in `tests/fixtures/negative/`**, not
  `profile-negative/`. Spec §6.3 originally said `profile-negative/`;
  plan synthesis corrected because `AddShapeMismatch` fires at IR
  build (compiler-level), not at lower (profile-level). The
  `profile-negative/` dir is reserved for `LowerError` fixtures
  (`too_many_inputs.nfl` is the existing example).

- **`four_input_matmul.nfl` form: `a -> matmul[b] -> add[c] -> add[d]`.**
  Single fixture exercising N=4 ABI mapping AND matmul (the M12
  bug surface) AND emit_add at N=4 in one go. Cheaper than two
  separate fixtures.

### Problems encountered

None — `%rbp` insight discovered during plan synthesis short-
circuited the deepest implementation question (j-counter strategy).
Both Groups A and D landed mechanically once the register choice
was fixed.

### Next step

A2 LayerNorm + FFN in M14. LayerNorm requires mean/variance/sqrt/
divide computation pattern not yet present in any codegen — likely
a single `StdOp::LayerNorm` with internal multi-pass codegen
(mirroring how `Softmax` is one node, not "exp + sum + divide"
decomposed). FFN composes existing ops (`linear → activation →
linear`). M13 added zero new codegen patterns; M14 will add at
least one (LayerNorm).

Trigger-driven cleanup status: OQ-7/8/9 + M5c OQ-4 still dormant
through M13 (no triggers fired). Per project memory rule
("triggered cleanup is an obligation"), monitor across M14
implementation.
```

- [ ] **Step 6.7: Run all workspace gates one final time**

Run in parallel:
- `cargo build --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo test --workspace`

Expected: all clean; final test count ≥ 402 on macOS arm64.

- [ ] **Step 6.8: Commit Group F**

```bash
git add PROJECT_SPEC.md CLAUDE.md DEVLOG.md docs/language_reference/grammar.md docs/profile_guide/arm64.md docs/profile_guide/x86_64.md
git commit -m "$(cat <<'EOF'
docs(m13): close M13 — PROJECT_SPEC, CLAUDE, DEVLOG, grammar, profile guides

PROJECT_SPEC.md:
- Milestones table: M13 row added.
- Current Status: bumped to M13 (390 → ~402 tests).
- Strategic Roadmap §Axis 2: A2 first brick closed; LayerNorm + FFN
  remain in M14+.
- Removed M12 "Known gap" line about N=4 + matmul rejection.

CLAUDE.md:
- Repository Structure tree: profiles/{arm64,x86_64}/src/ops/add.rs
  added in alphabetical order.
- Current Status: bumped to M13.
- Removed M12 follow-up note about x86_64 N=4 + matmul.

docs/language_reference/grammar.md:
- New `add[other]` stdlib reference. Same level of detail as
  `matmul` and `mul_scalar`. Notes strict shape equality and the
  "no broadcasting" design principle.

docs/profile_guide/arm64.md:
- New "M13 ops" section: emit_add register layout (x9/x10/x11
  pointers, x12 counter, x13 bound), inner-loop asm.

docs/profile_guide/x86_64.md:
- New "M13 ops" section: emit_add register layout. Explains the
  %rbp counter trick (free at all N ∈ [1,4] because callee-saved
  by unconditional prologue pushq %rbp).
- Matmul section gets "M13 N=4 + matmul fix" note explaining
  j-counter relocation %r9 → %rbp.

DEVLOG.md:
- Standard M13 entry: What was done (6 groups), Decisions made
  (%rbp option D + negative fixture dir + four_input_matmul form),
  Problems encountered (none — %rbp insight short-circuited the
  deepest open question), Next step (A2 LayerNorm + FFN in M14).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Spec coverage check

Spec §2 → Tasks:

| Spec §2 Group | Task | Status |
|---------------|------|--------|
| Group A — N=4 + matmul fix on x86_64 | Task 1 | ✓ |
| Group B — `StdOp::Add` foundation | Task 2 | ✓ |
| Group C — arm64 `emit_add` | Task 3 | ✓ |
| Group D — x86_64 `emit_add` | Task 4 | ✓ |
| Group E — integration fixtures + FFI tests | Task 5 | ✓ |
| Group F — closure docs | Task 6 | ✓ |

Spec §9 open questions → Plan resolutions:

| Spec §9 item | Plan resolution |
|--------------|-----------------|
| N=4 j-counter register choice | Task 1: `%rbp` (option D beyond spec enumeration) |
| `emit_add` x86_64 loop counter strategy | Task 4: `%rbp` (same trick) |
| N=4 + matmul fixture name | Task 5 Step 5.2: `four_input_matmul.nfl` |
| Test count target | Task 5 Step 5.9: 390 → ~402 (12 new tests counted per-group) |

Spec §12 acceptance criteria → Verification steps:

| AC | Verified by |
|----|-------------|
| 1. `cargo build --workspace` clean | Steps 1.8, 2.13, 3.8, 4.8, 5.9, 6.7 |
| 2. `cargo clippy ... -D warnings` clean | Steps 1.8, 2.13, 3.8, 4.8, 5.9, 6.7 |
| 3. `cargo fmt --all -- --check` clean | Steps 1.8, 2.13, 3.8, 4.8, 5.9, 6.7 |
| 4. `cargo test --workspace` passes; count > 390 | Step 5.9 (final count check) |
| 5. `residual_add.nfl` bit-exact FFI both profiles | Steps 5.5, 5.6 |
| 6. `add_shape_mismatch.nfl` rejects with `ShapeError::AddShapeMismatch` | Step 5.4 |
| 7. N=4 + matmul fixture compiles + runs bit-exact x86_64 | Step 5.7 |
| 8. `emit_matmul_body_contains_zero_pushq` invariant holds | Steps 1.7, 4.8 |
| 9. Documentation Group F shipped | Task 6 |
| 10. Bench harness still builds + runs | Step 5.9 (`cargo build --workspace` exercises bench) |

## Self-review notes

**Placeholder scan:** No "TBD"/"TODO"/"fill in details" in the plan body. Every code block contains the actual code an engineer needs.

**Type consistency check:**
- `emit_add` signature is identical on both profiles: `(abi: &AbiContext, total_elements: u64, model_idx: usize, op_idx: usize, a_loc: BufferLoc, other_loc: BufferLoc, dst_loc: BufferLoc) -> String`. (No `Result` return — neither emit can fail; matches `emit_mulscalar` template.)
- `walk_model` dispatch uses `add_idx` counter (per-model), matching the existing `linear_idx`, `relu_idx`, etc. naming convention.
- `ShapeError::AddShapeMismatch` field names (`expected`, `got`) match the spec §4.3 wording and the `Display` formatter.
- Test names follow the existing project convention: `<emit_function>_<assertion>` (e.g. `emit_matmul_body_contains_zero_pushq`).

**Spec coverage:** All §2 Groups have tasks. All §9 open questions have plan-level resolutions. All §12 acceptance criteria are verified by an explicit step.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-m13-n4-fix-and-add-op.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (Task 1 through Task 6), review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
