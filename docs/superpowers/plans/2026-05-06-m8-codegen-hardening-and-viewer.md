# M8 — ARM64 Codegen Hardening + Viewer v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two arm64 codegen bugs (dropout-as-output, dim-immediate encoding) and ship the viewer v0.1 (`--uir-verbose`) — three atomic commits in a single PR mirroring the M5/M6/M7 single-PR-with-atomic-task-pack convention.

**Architecture:** Each commit is independently green; no cross-commit dependencies. Commit 1 adds an `emit_dropout_copy` helper and a `BufferLoc::OutputReg` branch in `walk_model`'s `Dropout` arm. Commit 2 routes 17 immediate-emission sites in `linear.rs`/`relu.rs`/`softmax.rs` through the existing `emit_imm32` helper using two placement strategies (Group A hoist for bl-free loops, Group B re-materialise for bl-containing loops). Commit 3 adds three newtype wrappers (`VerboseUir`, `VerboseModel`, `VerboseNode`) with their own `Display` impls plus a `calls_extern_math` predicate and a `--uir-verbose` CLI flag.

**Tech Stack:** Rust workspace (3 crates: `compiler` lib, `nflc` bin, `profiles-arm64` lib), std-only production code, `cc` crate as dev-dep for FFI tests, AArch64 assembly target, libc dynamic loader for FFI.

**Spec:** [`docs/superpowers/specs/2026-05-06-m8-codegen-hardening-and-viewer-design.md`](../specs/2026-05-06-m8-codegen-hardening-and-viewer-design.md)

---

## Pre-implementation findings

- **`emit_imm32` already exists** at `profiles/arm64/src/asm.rs:103-116`. Signature: `pub fn emit_imm32(reg: &str, value: usize) -> String`. Internally emits `movz` + optional `movk`, asserts `value <= u32::MAX as usize`. Already used by `emit_linear` for `weight_offset`/`bias_offset`. Commit 2 reuses it; no new helper needed.
- **`materialise_ptr` uses `w10/x10` internally** (`linear.rs:250-254`) but ONLY for stack-offsets that don't fit shifted-imm12 (off > 16,773,120 OR off ≤ 4095 and not in shift range). For all current fixtures, `materialise_ptr` finishes before any cmp-bound hoist, so x10 is free for hoisting in matmul body Group A.
- **`NodeKind` is NOT `#[non_exhaustive]`** (`compiler/src/ir/types.rs`). Existing `Display for Node` (line 138) uses 2-arm exhaustive match without wildcard. New `calls_extern_math` matches that style.
- **arm64 unit-test convention:** `profiles/arm64/src/tests.rs::build_uir(src: &str)` parses NFL string + `compiler::ir::build` and returns `Uir`. Use this to drive emitters from realistic UIR rather than hand-rolled nodes.
- **arm64 FFI-test convention:** `profiles/arm64/tests/integration.rs` declares reference implementations as plain Rust fns at file top, then `#[test]` fns invoke `cc` + `dlopen` to compare emitted asm against reference. New FFI tests follow the existing `m6_*` test patterns at the bottom of that file.
- **CLI smoke convention:** `nflc/tests/cli_compile.rs` uses `Command::new(env!("CARGO_BIN_EXE_nflc"))` with `../tests/fixtures/...` paths (cargo runs integration tests with cwd at `nflc/`). `--uir-verbose` is a `parse` subcommand flag; tests live in a sibling file `nflc/tests/cli_parse.rs` (new).
- **Compiler-side test convention:** `compiler/src/ir/tests.rs` is the test module for the `ir` submodule. Predicate and `VerboseUir` snapshot tests go there.
- **`compiler::ir::test_utils` is `pub(crate)`** — not visible to other crates. Profile-side and snapshot tests build UIR from NFL strings via `compiler::parse` + `compiler::ir::build`, not via direct node construction.

---

## File map

### New files

| Path | Role |
|---|---|
| `tests/fixtures/dropout_only.nfl` | Single-node input + dropout, dropout IS model output. Triggers Commit 1's bug-fix path. |
| `tests/fixtures/large_classifier_k.nfl` | `k=8192 > 4095`. Triggers Commit 2's `cmp x5, #{k}` and `mov x8, #{k}` paths. |
| `tests/fixtures/large_classifier_n.nfl` | `out=5120 > 4095`. Triggers Commit 2's `cmp x4, #{n}` and softmax cmp paths. |
| `nflc/tests/cli_parse.rs` | CLI smoke tests for `nflc parse --uir-verbose` and the `--uir`/`--uir-verbose` mutual-exclusion check. |

### Modified files

| Path | Change |
|---|---|
| `profiles/arm64/src/ops/dropout.rs` | Add `pub fn emit_dropout_copy(...)`. |
| `profiles/arm64/src/ops/mod.rs` | Add `pub use dropout::emit_dropout_copy;`. |
| `profiles/arm64/src/codegen.rs` | Add `dropout_idx` counter; modify `StdOp::Dropout` arm to branch on `dst_loc`. |
| `profiles/arm64/src/ops/relu.rs` | Replace `cmp x9, #{total_floats}` with hoisted `emit_imm32`+register-form cmp. |
| `profiles/arm64/src/ops/linear.rs` | Replace 7 cmp + 4 mov immediate sites; add 3 hoists (b/n/k → x10/x15/x16) for matmul body, re-materialise pattern for RowWise softmax tail. |
| `profiles/arm64/src/ops/softmax.rs` | Replace 4 cmp + 1 mov immediate sites with re-materialise-at-loop-top pattern. |
| `profiles/arm64/src/tests.rs` | Add asm-shape positive checks for Commit 1 (`emit_dropout_copy`) and Commit 2 (movz+cmp register-form pairs across all emitters). |
| `profiles/arm64/tests/integration.rs` | Add FFI tests: `dropout_only_b2_k4_no_passes`, `dropout_only_b1_k8_no_passes`, `large_classifier_k_8192`, `large_classifier_n_5120`. |
| `compiler/src/ir/types.rs` | Add `calls_extern_math` methods on `Uir` and `UirModel`; add `VerboseUir`, `VerboseModel`, `VerboseNode` newtypes with `Display` impls. |
| `compiler/src/ir/tests.rs` | Add predicate unit tests (3 sub-cases) and `VerboseUir` snapshot test. |
| `nflc/src/main.rs` | Add `--uir-verbose` flag; mutual-exclusion check; rendering branch. Update help text. |
| `docs/language_reference/uir.md` | Add "Viewing UIR" section documenting `--uir` (compact) and `--uir-verbose` (annotated). |
| `docs/profile_guide/arm64.md` | Add two short paragraphs: "Dropout-as-output copy" and "Dim-immediate uniformity". |
| `PROJECT_SPEC.md` | Update M8 row in milestones table. |
| `CLAUDE.md` | Rewrite "Current Status"; bump Design Principle 5 reference `(M8+)` → `(M9+)`. |
| `DEVLOG.md` | Add closeout entry. |

### Unchanged

- `profiles/arm64/src/buffer.rs` (Commit 1 fix is in `codegen.rs`, not here).
- `profiles/arm64/src/asm.rs` (helper already exists).
- `language/grammar.ebnf` (no NFL grammar work).
- `Cargo.toml` workspace (no new crates, no new deps).

---

## Task overview

| # | Task | Commit |
|---|---|---|
| 1 | New fixture `dropout_only.nfl` | C1 |
| 2 | Add `emit_dropout_copy` + wire `Dropout` arm (asm-shape TDD) | C1 |
| 3 | FFI test `dropout_only_b2_k4_no_passes` | C1 |
| 4 | FFI test `dropout_only_b1_k8_no_passes` | C1 |
| 5 | Verify workspace; Commit 1 | C1 |
| 6 | Group A: `emit_relu` hoist (asm-shape TDD) | C2 |
| 7 | Group A: `emit_linear` matmul body — 3 hoists + 3 mov-reuses | C2 |
| 8 | Group B: `emit_softmax` standalone re-materialise | C2 |
| 9 | Group B: `emit_linear` RowWise softmax tail re-materialise | C2 |
| 10 | New fixtures `large_classifier_k.nfl` + `large_classifier_n.nfl` | C2 |
| 11 | FFI test `large_classifier_k_8192` | C2 |
| 12 | FFI test `large_classifier_n_5120` | C2 |
| 13 | Run full FFI suite; Commit 2 | C2 |
| 14 | `calls_extern_math` predicate + 3 unit sub-cases | C3 |
| 15 | `VerboseUir`/`VerboseModel`/`VerboseNode` newtypes + snapshot test | C3 |
| 16 | `--uir-verbose` CLI flag + smoke test | C3 |
| 17 | Mutual-exclusion `--uir`/`--uir-verbose` + smoke test | C3 |
| 18 | Update `docs/language_reference/uir.md` "Viewing UIR" section | C3 |
| 19 | Verify workspace; Commit 3 | C3 |
| 20 | Holistic review subagent dispatch | review |
| 21 | Conditional: `chore(m8/holistic)` close-in-M8 findings | review |
| 22 | Update `PROJECT_SPEC.md` M8 row | closeout |
| 23 | Update `CLAUDE.md` "Current Status" + Design Principle 5 | closeout |
| 24 | Update `DEVLOG.md` closeout entry | closeout |
| 25 | Update `docs/profile_guide/arm64.md` codegen-hardening section | closeout |
| 26 | Closeout commit + push branch + open PR | closeout |

---

## Task 1: New fixture `dropout_only.nfl`

**Files:**
- Create: `tests/fixtures/dropout_only.nfl`

- [ ] **Step 1: Create fixture**

Write to `tests/fixtures/dropout_only.nfl`:

```nfl
model OnlyDropout [b=2, k=4]:
    x: Tensor[b, k]
    x -> dropout[rate=0.1]
```

- [ ] **Step 2: Verify it parses**

Run: `cargo run --bin nflc -- parse tests/fixtures/dropout_only.nfl --uir`
Expected: Successful parse + UIR output showing 2 nodes (input n0, dropout n1) with `n1` as model output.

---

## Task 2: Add `emit_dropout_copy` + wire `Dropout` arm

**Files:**
- Create: `profiles/arm64/src/ops/dropout.rs` (modify — file exists as marker)
- Modify: `profiles/arm64/src/ops/mod.rs`
- Modify: `profiles/arm64/src/codegen.rs:111` (counter), `:166-169` (Dropout arm)
- Test: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing asm-shape test**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn dropout_as_output_emits_copy_loop() {
    let uir = build_uir(
        "model OnlyDropout [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        s.contains("; dropout-as-output:"),
        "missing dropout-as-output comment in:\n{s}"
    );
    assert!(
        s.contains(".Ldropout_0_0:"),
        "missing dropout loop label in:\n{s}"
    );
    assert!(
        s.contains("ldr     s3, [x11"),
        "missing s3 load from src ptr in:\n{s}"
    );
    assert!(
        s.contains("str     s3, [x12"),
        "missing s3 store to dst ptr in:\n{s}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --package profiles-arm64 --lib dropout_as_output_emits_copy_loop`
Expected: FAIL — generated asm currently emits nothing for the dropout (the bug). The first assertion fails.

- [ ] **Step 3: Implement `emit_dropout_copy`**

Replace contents of `profiles/arm64/src/ops/dropout.rs` with:

```rust
//! Dropout codegen.
//!
//! At inference, dropout is identity. The buffer-assignment first-pass
//! (`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand)` for
//! dropout nodes that are NOT the model output; in that case no asm is
//! emitted (downstream ops read from the operand's buffer directly).
//!
//! When a dropout node IS `model.output`, however, `assign_buffers`
//! returns `BufferLoc::OutputReg` (the caller's `x2` pointer). In that
//! case the operand's buffer must be explicitly copied into the output
//! buffer, since alias-redirection no longer applies. `emit_dropout_copy`
//! emits the float-by-float copy loop for that path.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for a dropout-as-output copy loop.
///
/// Mirror of `emit_relu`'s structure minus the zero-init and `fmax`:
/// element-wise load → store, no transformation. Used only when a
/// `Dropout` node is the model's output (see module-level doc).
pub fn emit_dropout_copy(
    total_floats: u64,
    model_idx: usize,
    dropout_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let did = format!("{model_idx}_{dropout_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; dropout-as-output: copy operand→output ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str(&emit_imm32("x10", total_floats as usize));
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Ldropout_{did}:\n"));
    s.push_str("    cmp     x9, x10\n");
    s.push_str(&format!("    b.ge    .Ldropout_end_{did}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Ldropout_{did}\n"));
    s.push_str(&format!(".Ldropout_end_{did}:\n"));
    s
}
```

- [ ] **Step 4: Re-export from `ops/mod.rs`**

In `profiles/arm64/src/ops/mod.rs`, add the re-export alongside the existing `pub use ...` lines:

```rust
pub use dropout::emit_dropout_copy;
```

- [ ] **Step 5: Wire `Dropout` arm in `codegen.rs::walk_model`**

In `profiles/arm64/src/codegen.rs`, near line 111 add a counter alongside the existing `linear_idx`/`relu_idx`/`softmax_idx`:

```rust
let mut dropout_idx = 0usize;
```

Replace the existing `StdOp::Dropout` arm (lines 166-169) with:

```rust
StdOp::Dropout => {
    let src_loc = resolve_loc(&assignment.locs, operands[0]);
    let dst_loc = resolve_loc(&assignment.locs, node_idx);
    if matches!(dst_loc, crate::buffer::BufferLoc::OutputReg) {
        // Bug-fix path (M8): dropout-as-output requires explicit copy
        // because BufferLoc::Alias redirection doesn't apply when this
        // node IS the output. See `ops/dropout.rs` module doc.
        let total: u64 = node.ty.shape.0.iter().product();
        body.push_str(&crate::ops::emit_dropout_copy(
            total, model_idx, dropout_idx, src_loc, dst_loc,
        ));
        dropout_idx += 1;
    }
    // else BufferLoc::Alias: no asm — downstream reads operand directly.
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --package profiles-arm64 --lib dropout_as_output_emits_copy_loop`
Expected: PASS.

- [ ] **Step 7: Run full workspace tests, expect no regressions**

Run: `cargo test --workspace`
Expected: All tests pass. Test count is 209 (208 + 1 new).

---

## Task 3: FFI test `dropout_only_b2_k4_no_passes`

**Files:**
- Test: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Write failing FFI test**

Append to `profiles/arm64/tests/integration.rs`. Pattern matches existing M4/M5/M6 tests (inline lowering, `common::compile_to_dylib` + `libloading::Library::new`). Note the `nfl_forward_<Model>` ABI is `(input, params, output)` — see e.g. `m4a_no_softmax_still_runs` at line 100.

```rust
// ---------------------------------------------------------------------------
// M8 fixture: dropout-only model (dropout IS model.output).
// Triggers the BufferLoc::OutputReg branch in walk_model::Dropout.
// ---------------------------------------------------------------------------

#[test]
fn dropout_only_b2_k4_no_passes() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/dropout_only.nfl")
        .expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    // No run_pipeline — exercise raw UIR (mirror of `--no-passes`).
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let dylib_path = common::compile_to_dylib(&asm.source, "dropout_only_b2_k4");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let params: [f32; 0] = [];
    let mut output = [0.0f32; 8];

    unsafe {
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib
            .get(b"nfl_forward_OnlyDropout\0")
            .expect("symbol not found");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    assert_eq!(
        output, input,
        "dropout-as-output must copy input verbatim; got {:?}",
        output
    );
}
```

- [ ] **Step 2: Run test, verify pass**

Run: `cargo test --package profiles-arm64 --test integration dropout_only_b2_k4_no_passes`
Expected: PASS. Output array equals input array bit-exact.

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Test count is 210.

---

## Task 4: FFI test `dropout_only_b1_k8_no_passes`

**Files:**
- Test: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Add second fixture variant inline**

Append to `profiles/arm64/tests/integration.rs` after the b=2,k=4 test. NFL source is inline (no separate fixture file — the model name `OnlyDropout1` differentiates the symbol).

```rust
#[test]
fn dropout_only_b1_k8_no_passes() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    // Same total floats (8) as b=2,k=4 but b=1 — closes single-row
    // coverage gap noted in the M8 audit.
    let nfl_src = "model OnlyDropout1 [b=1, k=8]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n";
    let ast = compiler::parse(nfl_src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let dylib_path = common::compile_to_dylib(&asm.source, "dropout_only_b1_k8");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");

    let input = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let params: [f32; 0] = [];
    let mut output = [0.0f32; 8];

    unsafe {
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib
            .get(b"nfl_forward_OnlyDropout1\0")
            .expect("symbol not found");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    assert_eq!(output, input, "b=1 dropout-as-output must copy verbatim");
}
```

- [ ] **Step 2: Run test, verify pass**

Run: `cargo test --package profiles-arm64 --test integration dropout_only_b1_k8_no_passes`
Expected: PASS.

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Test count is 211.

---

## Task 5: Verify workspace; Commit 1

**Files:** none (verification only)

- [ ] **Step 1: `cargo fmt --all`**

Run: `cargo fmt --all`
Expected: Format applied; subsequent `--check` is clean.

- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Exit 0, no warnings.

- [ ] **Step 3: `cargo test --workspace`**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 211.

- [ ] **Step 4: Stage and commit**

Run:

```bash
git add tests/fixtures/dropout_only.nfl \
        profiles/arm64/src/ops/dropout.rs \
        profiles/arm64/src/ops/mod.rs \
        profiles/arm64/src/codegen.rs \
        profiles/arm64/src/tests.rs \
        profiles/arm64/tests/integration.rs
```

Then commit:

```bash
git commit -m "$(cat <<'EOF'
feat(m8/arm64-fix): correct dropout-as-output codegen

When a Dropout node IS model.output, assign_buffers returns
BufferLoc::OutputReg, but walk_model previously emitted no asm for
StdOp::Dropout — leaving the caller's output buffer uninitialised.

Fix: branch on dst_loc in the Dropout arm. If OutputReg, emit a
copy loop via new ops/dropout.rs::emit_dropout_copy (mirror of
emit_relu minus the fmax). Otherwise (BufferLoc::Alias path),
continue emitting nothing — downstream ops read operand directly.

emit_dropout_copy uses emit_imm32 from birth, so Commit 2's
dim-immediate uniformity work patches exactly 17 pre-existing sites,
not 18.

Tests:
- profiles/arm64/src/tests.rs::dropout_as_output_emits_copy_loop
- profiles/arm64/tests/integration.rs::dropout_only_b2_k4_no_passes
- profiles/arm64/tests/integration.rs::dropout_only_b1_k8_no_passes
  (b=1 single-row variant closes M8-audit coverage gap)
- New fixture: tests/fixtures/dropout_only.nfl

Test count: 208 → 211.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify commit landed cleanly**

Run: `git log --oneline -1 && git status`
Expected: Most recent commit is the new one; working tree clean.

---

## Task 6: Group A — `emit_relu` hoist

**Files:**
- Modify: `profiles/arm64/src/ops/relu.rs`
- Test: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing positive-check test**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn relu_uses_register_form_cmp_with_hoisted_movz() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Hoisted materialise must appear AFTER the materialise_ptr lines
    // (which set up x11/x12) and BEFORE the .Lrelu_ label.
    let movz_pos = s
        .find("movz    x10, ")
        .expect("missing movz x10 hoist for relu loop bound");
    let label_pos = s
        .find(".Lrelu_0_0:")
        .expect("missing relu loop label");
    assert!(
        movz_pos < label_pos,
        "movz x10 must precede .Lrelu_ label (hoist outside loop)"
    );

    // Inside loop, cmp uses register form against x10.
    assert!(
        s.contains("cmp     x9, x10"),
        "cmp must use register form (x9, x10), not literal imm; full asm:\n{s}"
    );
    // Old literal-imm form must not appear for relu's bound.
    assert!(
        !s.contains("cmp     x9, #4"),
        "old literal-imm cmp must be replaced; full asm:\n{s}"
    );
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package profiles-arm64 --lib relu_uses_register_form_cmp_with_hoisted_movz`
Expected: FAIL — current emit_relu uses `cmp x9, #4` literal form.

- [ ] **Step 3: Patch `emit_relu`**

In `profiles/arm64/src/ops/relu.rs`, add the import at the top:

```rust
use crate::asm::emit_imm32;
```

Replace the body of `emit_relu` between the `materialise_ptr` calls and the loop label so it reads:

```rust
pub fn emit_relu(
    total_floats: u64,
    model_idx: usize,
    relu_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let rid = format!("{model_idx}_{relu_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; relu: copy-clamp from src to dst ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str(&emit_imm32("x10", total_floats as usize));
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str("    cmp     x9, x10\n");
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}
```

- [ ] **Step 4: Run new test, verify pass**

Run: `cargo test --package profiles-arm64 --lib relu_uses_register_form_cmp_with_hoisted_movz`
Expected: PASS.

- [ ] **Step 5: Update existing relu tests**

The existing tests `relu_emits_separate_loop_with_fmov_zero_and_fmax` and `relu_alone_after_matmul_does_not_break_existing_test` in `tests.rs` assert `cmp x9, #4`. Update those assertions to the new register form. Find them with:

Run: `grep -n 'cmp     x9, #4' profiles/arm64/src/tests.rs`

Replace each `assert!(s.contains("cmp     x9, #4"));` line with:

```rust
assert!(s.contains("cmp     x9, x10"));
```

- [ ] **Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 212.

---

## Task 7: Group A — `emit_linear` matmul body hoists + mov-reuses

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs:65-127`
- Test: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing positive-check test**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn linear_matmul_body_uses_hoisted_dim_registers() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Three hoists must appear before the i-loop label.
    let i_label_pos = s
        .find(".Lmm_i_0_0:")
        .expect("missing matmul i-loop label");
    for reg in ["x10", "x15", "x16"] {
        let movz = format!("movz    {}, ", reg);
        let pos = s
            .find(&movz)
            .unwrap_or_else(|| panic!("missing hoist for {reg}: \n{s}"));
        assert!(
            pos < i_label_pos,
            "{reg} hoist must precede .Lmm_i_ label"
        );
    }

    // Loop-bound cmps use register form.
    assert!(s.contains("cmp     x3, x10"), "i-loop cmp must use x10");
    assert!(s.contains("cmp     x4, x15"), "j-loop cmp must use x15");
    assert!(s.contains("cmp     x5, x16"), "k-loop cmp must use x16");

    // Mov-sites for stride reuse hoisted registers (no re-materialise).
    assert!(
        s.contains("mov     x8, x16"),
        "input-stride mov must reuse hoisted k (x16)"
    );
    assert!(
        s.contains("mov     x8, x15"),
        "output-stride mov must reuse hoisted n (x15)"
    );

    // Old literal-imm cmps must not appear for matmul bounds.
    for old in ["cmp     x3, #2", "cmp     x4, #2", "cmp     x5, #3"] {
        assert!(
            !s.contains(old),
            "old literal-imm cmp '{old}' must be removed"
        );
    }
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package profiles-arm64 --lib linear_matmul_body_uses_hoisted_dim_registers`
Expected: FAIL — current emit_linear uses literal-imm cmps and `mov x8, #{k}` etc.

- [ ] **Step 3: Patch matmul body in `emit_linear`**

In `profiles/arm64/src/ops/linear.rs`, locate the section starting at line 65 (`s.push_str("    mov     x3, #0\n");`). Replace the matmul body block (lines 65-127) — i.e. from the `mov x3, #0` line through the j-loop store site at line 127 — with the version below.

Note: the existing post-op code (line 106 onwards: `for post_op in fused_post_ops { ... PostOp::Relu => "fmax s0, s0, s4" ... }`) and bias-add (line 98-101) MUST remain UNCHANGED in their relative positions. Only the immediates change.

The replacement:

```rust
    // M8: hoist matmul loop bounds into x10/x15/x16 once, before the
    // i-loop label. All bl-free cmps inside use register form. Stride
    // movs reuse the hoisted registers (no re-materialise).
    s.push_str(&emit_imm32("x10", b as usize));
    s.push_str(&emit_imm32("x15", n as usize));
    s.push_str(&emit_imm32("x16", k as usize));

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str("    cmp     x3, x10\n");
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str("    cmp     x4, x15\n");
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str("    cmp     x5, x16\n");
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str("    mov     x8, x16\n");
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str("    mov     x8, x15\n");
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));
```

The bias-add block (line 98-101 in pre-patch) and the elementwise post-op loop (line 106-121 in pre-patch) are **not modified** and continue immediately after this block.

After the elementwise post-op loop, the store-site (pre-patch line 124-127) becomes:

```rust
    // Store after elementwise post-ops. Reuse hoisted x15 (= n).
    s.push_str("    mov     x8, x15\n");
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");
```

- [ ] **Step 4: Run new test, verify pass**

Run: `cargo test --package profiles-arm64 --lib linear_matmul_body_uses_hoisted_dim_registers`
Expected: PASS.

- [ ] **Step 5: Update existing linear tests**

Existing test `linear_emits_matmul_loops_with_fmadd` asserts `cmp x3, #2`, `cmp x4, #2`, `cmp x5, #3`. Update these to `x10`/`x15`/`x16` register form:

```rust
assert!(s.contains("cmp     x3, x10"));
assert!(s.contains("cmp     x4, x15"));
assert!(s.contains("cmp     x5, x16"));
```

Search for any other tests asserting on the old literal-imm cmps in matmul:

Run: `grep -n 'cmp     x[345], #' profiles/arm64/src/tests.rs`
Expected: After updates, no matches in matmul-body assertions.

- [ ] **Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 213.

---

## Task 8: Group B — `emit_softmax` standalone re-materialise

**Files:**
- Modify: `profiles/arm64/src/ops/softmax.rs`
- Test: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing positive-check test**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn softmax_standalone_uses_register_form_cmps_re_materialised() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> softmax\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // i-loop, max-loop, exp-loop, norm-loop — all four cmps register form.
    assert!(s.contains("cmp     x19, x10"), "i-loop cmp register form");
    // x21 is reused across max/exp/norm phases — find the cmp pattern.
    let count_x21_cmp_x10 = s.matches("cmp     x21, x10").count();
    assert_eq!(
        count_x21_cmp_x10, 3,
        "max/exp/norm phases must each cmp x21 against x10 (3 sites); got {count_x21_cmp_x10}\nfull asm:\n{s}"
    );

    // No literal-imm cmps for softmax bounds.
    assert!(
        !s.contains("cmp     x19, #2"),
        "old i-loop literal-imm cmp must be removed"
    );
    assert!(
        !s.contains("cmp     x21, #3"),
        "old phase-loop literal-imm cmps must be removed"
    );
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package profiles-arm64 --lib softmax_standalone_uses_register_form_cmps_re_materialised`
Expected: FAIL — softmax.rs currently emits literal-imm cmps.

- [ ] **Step 3: Patch `emit_softmax`**

Open `profiles/arm64/src/ops/softmax.rs`. Add the import at the top if not present:

```rust
use crate::asm::emit_imm32;
```

Inside `emit_softmax`, replace each `s.push_str(&format!("    cmp     x19, #{b}\n"));` site with the re-materialise pattern (movz/movk to x10 BEFORE the cmp, both inside the loop top — i.e. immediately after the loop label and before the cmp/b.ge):

For the i-loop (currently `cmp x19, #{b}` at line 46):

```rust
    s.push_str("    mov     x19, #0\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&emit_imm32("x10", b as usize));
    s.push_str("    cmp     x19, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_i_end_{sid}\n"));
```

For the max-phase loop (currently `cmp x21, #{k}` at line 59):

```rust
    s.push_str("    mov     x21, #0\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&emit_imm32("x10", k as usize));
    s.push_str("    cmp     x21, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_max_end_{sid}\n"));
```

Apply the same pattern to the exp-phase loop (`cmp x21, #{k}` at line 74) and the norm-phase loop (`cmp x21, #{k}` at line 91). For the mov-site at line 50 (`mov x8, #{k}`), replace with:

```rust
    s.push_str(&emit_imm32("x8", k as usize));
```

The full updated structure follows the existing softmax control flow; only the cmp-imm and mov-imm sites change.

- [ ] **Step 4: Run new test, verify pass**

Run: `cargo test --package profiles-arm64 --lib softmax_standalone_uses_register_form_cmps_re_materialised`
Expected: PASS.

- [ ] **Step 5: Update any existing softmax-cmp assertions**

Run: `grep -n 'cmp     x19, #\|cmp     x21, #' profiles/arm64/src/tests.rs`
Expected: All matches in softmax-related tests must be updated to register form.

For each match, change `cmp x19, #{b}` to `cmp x19, x10` and `cmp x21, #{k}` to `cmp x21, x10`.

- [ ] **Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 214.

---

## Task 9: Group B — `emit_linear` RowWise softmax tail re-materialise

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs:155-212` (RowWise softmax tail block)
- Test: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing positive-check test**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn linear_rowwise_softmax_tail_uses_re_materialised_cmps() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[4] -> softmax\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Pipeline applies fuse_linear_softmax → emits RowWise tail.
    assert!(
        s.contains("; fused softmax_row:"),
        "expected fused RowWise softmax tail; full asm:\n{s}"
    );

    // Re-materialise pattern: at each fsmx loop top, movz x10 then cmp.
    assert!(s.contains("cmp     x19, x10"), "fsmx i-loop cmp register form");
    let count_x21_cmp_x10 = s.matches("cmp     x21, x10").count();
    // 3 phase loops in the tail (max/exp/norm).
    assert_eq!(
        count_x21_cmp_x10, 3,
        "fsmx max/exp/norm cmps must each use register form; got {count_x21_cmp_x10}"
    );

    // No literal-imm fsmx cmps remain.
    assert!(
        !s.contains("cmp     x19, #2"),
        "old fsmx i-loop literal-imm cmp must be removed"
    );
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package profiles-arm64 --lib linear_rowwise_softmax_tail_uses_re_materialised_cmps`
Expected: FAIL.

- [ ] **Step 3: Patch RowWise softmax tail**

In `profiles/arm64/src/ops/linear.rs`, locate the `PostOp::SoftmaxRow` arm starting around line 142. The block from line 156 through line 212 emits the RowWise softmax tail.

Update the i-loop (currently lines 156-159):

```rust
                s.push_str("    mov     x19, #0\n");
                s.push_str(&format!(".Lfsmx_i_{lid}:\n"));
                s.push_str(&emit_imm32("x10", b as usize));
                s.push_str("    cmp     x19, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_i_end_{lid}\n"));
```

The mov-site (line 161, `mov x8, #{n}`) changes to:

```rust
                s.push_str(&emit_imm32("x8", n as usize));
```

The max-loop (lines 168-171), exp-loop (lines 181-184), and norm-loop (lines 198-201) each gain a `emit_imm32("x10", n as usize)` immediately after the label and before the cmp, with the cmp itself switched to register form (`cmp x21, x10`):

```rust
                // Max phase.
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_max_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_max_end_{lid}\n"));
                // ... unchanged body ...

                // Exp phase.
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_exp_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_exp_end_{lid}\n"));
                // ... unchanged body (includes bl _expf which clobbers x10 — reason for re-materialise) ...

                // Norm phase.
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_norm_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_norm_end_{lid}\n"));
                // ... unchanged body ...
```

All other lines in the RowWise tail (the `mov x22, x12`, the row-base computation, the body of each phase-loop including `bl _expf`) remain unchanged.

- [ ] **Step 4: Run new test, verify pass**

Run: `cargo test --package profiles-arm64 --lib linear_rowwise_softmax_tail_uses_re_materialised_cmps`
Expected: PASS.

- [ ] **Step 5: Update any existing RowWise tail assertions**

Run: `grep -n '\.Lfsmx_\|fused softmax_row' profiles/arm64/src/tests.rs`
For each test that asserts `cmp x19, #{b}` or `cmp x21, #{n}` patterns in the RowWise tail context, update to register form.

- [ ] **Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 215.

---

## Task 10: New fixtures `large_classifier_k.nfl` + `large_classifier_n.nfl`

**Files:**
- Create: `tests/fixtures/large_classifier_k.nfl`, `tests/fixtures/large_classifier_n.nfl`

- [ ] **Step 1: Create k-large fixture**

Write to `tests/fixtures/large_classifier_k.nfl`:

```nfl
model LargeK [b=2, k=8192, out=10]:
    x: Tensor[b, k]
    x -> linear[out] -> softmax
```

- [ ] **Step 2: Create n-large fixture**

Write to `tests/fixtures/large_classifier_n.nfl`:

```nfl
model LargeN [b=2, k=8, out=5120]:
    x: Tensor[b, k]
    x -> linear[out] -> softmax
```

- [ ] **Step 3: Verify both parse**

Run: `cargo run --bin nflc -- parse tests/fixtures/large_classifier_k.nfl --uir && cargo run --bin nflc -- parse tests/fixtures/large_classifier_n.nfl --uir`
Expected: Both parse successfully and emit UIR with `linear` + `softmax` nodes.

---

## Task 11: FFI test `large_classifier_k_8192`

**Files:**
- Test: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Write failing FFI test**

Append to `profiles/arm64/tests/integration.rs`:

```rust
// ---------------------------------------------------------------------------
// M8 fixture: dim > 4095 along k-axis. Triggers cmp x5, #{k} and
// mov x8, #{k} sites in matmul body — first test that actually exceeds
// ARM64 12-bit cmp immediate.
// ---------------------------------------------------------------------------

#[test]
fn large_classifier_k_8192() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    const B: usize = 2;
    const K: usize = 8192;
    const N: usize = 10;

    let src = std::fs::read_to_string("../../tests/fixtures/large_classifier_k.nfl")
        .expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir_pre = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir_pre, &compiler::passes::default_pipeline())
        .expect("pipeline");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let dylib_path = common::compile_to_dylib(&asm.source, "large_classifier_k");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");

    // Deterministic input: x[i, j] = (i * K + j) as f32 / 10000.0
    let input: Vec<f32> = (0..B * K).map(|i| (i as f32) / 10000.0).collect();
    // Deterministic weights: w[k, n] = ((k + n) % 7) as f32 / 100.0
    let weights: Vec<f32> = (0..K * N)
        .map(|i| {
            let kk = i / N;
            let nn = i % N;
            (((kk + nn) % 7) as f32) / 100.0
        })
        .collect();
    let mut output = vec![0.0f32; B * N];

    unsafe {
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib
            .get(b"nfl_forward_LargeK\0")
            .expect("symbol not found");
        forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
    }

    let matmul = reference_matmul(&input, &weights, B, K, N);
    let expected = reference_softmax_stable(&matmul, B, N);

    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "k=8192 output[{i}] = {got}, expected {want}"
        );
    }
}
```

- [ ] **Step 2: Run test, verify pass**

Run: `cargo test --package profiles-arm64 --test integration large_classifier_k_8192`
Expected: PASS. Output matches reference within 1e-3 tolerance (softmax + large-k accumulates float drift).

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 216.

---

## Task 12: FFI test `large_classifier_n_5120`

**Files:**
- Test: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Write failing FFI test**

Append to `profiles/arm64/tests/integration.rs`:

```rust
#[test]
fn large_classifier_n_5120() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    const B: usize = 2;
    const K: usize = 8;
    const N: usize = 5120;

    let src = std::fs::read_to_string("../../tests/fixtures/large_classifier_n.nfl")
        .expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir_pre = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir_pre, &compiler::passes::default_pipeline())
        .expect("pipeline");
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let dylib_path = common::compile_to_dylib(&asm.source, "large_classifier_n");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");

    let input: Vec<f32> = (0..B * K).map(|i| (i as f32) / 10.0).collect();
    let weights: Vec<f32> = (0..K * N)
        .map(|i| {
            let kk = i / N;
            let nn = i % N;
            (((kk + nn) % 5) as f32) / 100.0
        })
        .collect();
    let mut output = vec![0.0f32; B * N];

    unsafe {
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib
            .get(b"nfl_forward_LargeN\0")
            .expect("symbol not found");
        forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
    }

    let matmul = reference_matmul(&input, &weights, B, K, N);
    let expected = reference_softmax_stable(&matmul, B, N);

    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "n=5120 output[{i}] = {got}, expected {want}"
        );
    }
}
```

- [ ] **Step 2: Run test, verify pass**

Run: `cargo test --package profiles-arm64 --test integration large_classifier_n_5120`
Expected: PASS.

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 217.

---

## Task 13: Run full FFI suite; Commit 2

**Files:** none (verification only)

- [ ] **Step 1: Run all FFI tests bit-exact regression check**

Run: `cargo test --package profiles-arm64 --test integration`
Expected: All FFI tests pass — both M4/M5/M6/M7 carryover tests AND the new M8 large-dim tests. Bit-exact behaviour preserved on existing fixtures (Commit 2 changes only how immediates are materialised, not what computation runs).

- [ ] **Step 2: `cargo fmt --all`**

Run: `cargo fmt --all`

- [ ] **Step 3: `cargo clippy --workspace --all-targets -- -D warnings`**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Exit 0.

- [ ] **Step 4: `cargo test --workspace`**

Run: `cargo test --workspace`
Expected: All 217 tests pass.

- [ ] **Step 5: Stage and commit**

Run:

```bash
git add profiles/arm64/src/ops/relu.rs \
        profiles/arm64/src/ops/linear.rs \
        profiles/arm64/src/ops/softmax.rs \
        profiles/arm64/src/tests.rs \
        profiles/arm64/tests/integration.rs \
        tests/fixtures/large_classifier_k.nfl \
        tests/fixtures/large_classifier_n.nfl
```

Then commit:

```bash
git commit -m "$(cat <<'EOF'
feat(m8/arm64-fix): hoist dim immediates through emit_imm32

ARM64 cmp #imm encodes a 12-bit immediate (0-4095, optionally shifted
by 12); mov #imm encodes 16-bit (0-65535). Existing emitters used
literal #{dim} at 17 sites — silently broken on any production-scale
NN (transformer hidden_dim 4096+, LLM vocab 30k+, image classifier
10k classes). Current fixtures stayed within range by accident.

Fix: route all 17 sites through asm::emit_imm32 (movz + optional
movk) using two placement strategies:

  Group A (bl-free loops: emit_relu, emit_linear matmul body):
    hoist materialise once outside the loop label. Matmul body uses
    distinct registers per nesting level (x10 ← b, x15 ← n,
    x16 ← k); stride movs reuse the hoisted regs (mov x8, x16 etc).
    Inner-loop cmp uses register form — zero runtime cost.

  Group B (bl-containing loops: emit_softmax standalone, emit_linear
    RowWise softmax tail): re-materialise into x10 at each loop top
    (after label, before cmp). bl _expf clobbers caller-saved
    registers; hoisting outside would require expanding the
    callee-saved set in prologue/epilogue (out of scope). 1-2 movz
    per iter is < 1% overhead vs bl _expf cost.

Tests:
- profiles/arm64/src/tests.rs::relu_uses_register_form_cmp_with_hoisted_movz
- profiles/arm64/src/tests.rs::linear_matmul_body_uses_hoisted_dim_registers
- profiles/arm64/src/tests.rs::softmax_standalone_uses_register_form_cmps_re_materialised
- profiles/arm64/src/tests.rs::linear_rowwise_softmax_tail_uses_re_materialised_cmps
- profiles/arm64/tests/integration.rs::large_classifier_k_8192 (k > 4095)
- profiles/arm64/tests/integration.rs::large_classifier_n_5120 (n > 4095)
- New fixtures: tests/fixtures/large_classifier_{k,n}.nfl
- All existing FFI tests pass bit-exact (regression check).

Test count: 211 → 217.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Verify commit landed cleanly**

Run: `git log --oneline -2 && git status`
Expected: Commits 1 and 2 present; working tree clean.

---

## Task 14: `calls_extern_math` predicate + 3 unit sub-cases

**Files:**
- Modify: `compiler/src/ir/types.rs`
- Test: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Write failing predicate tests**

Append to `compiler/src/ir/tests.rs`:

```rust
#[test]
fn calls_extern_math_true_for_standalone_softmax() {
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    assert!(uir.calls_extern_math());
    assert!(uir.models[0].calls_extern_math());
}

#[test]
fn calls_extern_math_false_for_linear_only() {
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    assert!(!uir.calls_extern_math());
    assert!(!uir.models[0].calls_extern_math());
}

#[test]
fn calls_extern_math_true_for_fused_softmax_row() {
    // After default pipeline runs, linear→softmax fuses to
    // linear with PostOp::SoftmaxRow. Predicate must follow the fusion.
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[3] -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let fused = crate::passes::run_pipeline(&uir, &crate::passes::default_pipeline())
        .expect("pipeline");
    // Sanity: standalone softmax is gone, replaced by fused PostOp.
    let has_standalone_softmax = fused.models[0].nodes.iter().any(|n| match &n.kind {
        crate::ir::types::NodeKind::Op { op, .. } =>
            matches!(op, crate::ir::stdlib::StdOp::Softmax),
        _ => false,
    });
    assert!(!has_standalone_softmax, "fusion should have removed standalone softmax");
    assert!(fused.calls_extern_math());
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package compiler --lib calls_extern_math`
Expected: FAIL — `calls_extern_math` method does not exist yet.

- [ ] **Step 3: Implement predicate methods**

In `compiler/src/ir/types.rs`, add (after the existing `Display` impls or in any convenient location):

```rust
impl UirModel {
    /// True iff any operation in this model requires linking against
    /// external math (currently: standalone Softmax or fused SoftmaxRow).
    /// UIR-level predicate — does not depend on any profile.
    pub fn calls_extern_math(&self) -> bool {
        use crate::ir::stdlib::StdOp;
        use crate::ir::types::{NodeKind, PostOp};
        self.nodes.iter().any(|n| match &n.kind {
            NodeKind::Op { op, fused_post_ops, .. } => {
                matches!(op, StdOp::Softmax)
                    || fused_post_ops.iter().any(|p| matches!(p, PostOp::SoftmaxRow))
            }
            NodeKind::Input { .. } => false,
        })
    }
}

impl Uir {
    pub fn calls_extern_math(&self) -> bool {
        self.models.iter().any(UirModel::calls_extern_math)
    }
}
```

- [ ] **Step 4: Run new tests, verify pass**

Run: `cargo test --package compiler --lib calls_extern_math`
Expected: All 3 sub-cases PASS.

- [ ] **Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 220.

---

## Task 15: `VerboseUir`/`VerboseModel`/`VerboseNode` newtypes + snapshot test

**Files:**
- Modify: `compiler/src/ir/types.rs`
- Test: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Write failing snapshot test**

Append to `compiler/src/ir/tests.rs`:

```rust
#[test]
fn verbose_uir_snapshot_matches_expected_format() {
    use crate::ir::types::VerboseUir;

    // Pre-pass UIR — no run_pipeline. nflc parse --uir-verbose is the
    // parse subcommand, not compile, so the rendered UIR reflects
    // un-fused operations.
    let src = "model Demo [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> linear[3] -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");

    let rendered = format!("{}", VerboseUir(&uir));
    let expected = "\
uir-verbose summary
  models: 1
  total nodes: 3
  calls-extern-math: yes

uir-model Demo
  inputs: [n0]
  output: n2
  node count: 3
  calls-extern-math: yes

  n0: input \"x\"        :: [2, 4]
  n1: linear           :: [2, 3]    operands=[n0]    attrs=[out_dim=3]
  n2: softmax           :: [2, 3]    operands=[n1]
";
    assert_eq!(
        rendered, expected,
        "VerboseUir output drift — got:\n{rendered}\nexpected:\n{expected}"
    );
}
```

> **Implementation note:** The exact spacing in the expected string above
> mirrors the existing `Display for Node` and `Display for UirModel`
> formats (8-space pad after the op name and `::`, 4-space gap before
> `operands=[...]`). After implementing the newtypes in step 3, run
> the test and adjust the expected string ONLY to match the actual
> output, not the other way around. The test pins format; format
> evolves intentionally.

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package compiler --lib verbose_uir_snapshot_matches_expected_format`
Expected: FAIL — `VerboseUir` does not exist.

- [ ] **Step 3: Implement newtype wrappers**

In `compiler/src/ir/types.rs`, add (after the existing `Display for Uir` impl, at the bottom of the file or in a clearly delimited "// M8: verbose viewer wrappers" section):

```rust
// ----------------------------------------------------------------------------
// M8: verbose viewer wrappers
//
// Newtype pattern over plain methods. Idiomatic Rust composition:
// each wrapper has its own `Display` impl, so `write!(f, "{}",
// VerboseModel(m))` works inside the outer `VerboseUir` impl
// without any `fmt_verbose` boilerplate. Default `Display` for
// `Uir`/`UirModel`/`Node` is unchanged — the compact format used by
// `nflc parse --uir`.
// ----------------------------------------------------------------------------

pub struct VerboseUir<'a>(pub &'a Uir);

impl std::fmt::Display for VerboseUir<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_nodes: usize = self.0.models.iter().map(|m| m.nodes.len()).sum();
        writeln!(f, "uir-verbose summary")?;
        writeln!(f, "  models: {}", self.0.models.len())?;
        writeln!(f, "  total nodes: {}", total_nodes)?;
        writeln!(
            f,
            "  calls-extern-math: {}",
            if self.0.calls_extern_math() { "yes" } else { "no" }
        )?;
        writeln!(f)?;
        for m in &self.0.models {
            write!(f, "{}", VerboseModel(m))?;
        }
        Ok(())
    }
}

pub struct VerboseModel<'a>(pub &'a UirModel);

impl std::fmt::Display for VerboseModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.0;
        writeln!(f, "uir-model {}", m.name)?;
        let inputs = m
            .inputs
            .iter()
            .map(|i| format!("n{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "  inputs: [{}]", inputs)?;
        writeln!(f, "  output: n{}", m.output)?;
        writeln!(f, "  node count: {}", m.nodes.len())?;
        writeln!(
            f,
            "  calls-extern-math: {}",
            if m.calls_extern_math() { "yes" } else { "no" }
        )?;
        writeln!(f)?;
        for (i, node) in m.nodes.iter().enumerate() {
            write!(f, "  n{}: {}", i, VerboseNode(node))?;
        }
        Ok(())
    }
}

pub struct VerboseNode<'a>(pub &'a Node);

impl std::fmt::Display for VerboseNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0.kind {
            NodeKind::Input { name } => {
                writeln!(f, "input {:?}        :: {}", name, self.0.ty.shape)
            }
            NodeKind::Op {
                op,
                operands,
                attrs,
                fused_post_ops,
            } => {
                let ops_s = operands
                    .iter()
                    .map(|o| format!("n{}", o))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "{}           :: {}    operands=[{}]",
                    op, self.0.ty.shape, ops_s
                )?;
                if !attrs.is_empty() {
                    let a = attrs
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                writeln!(f)?;
                for p in fused_post_ops {
                    writeln!(f, "       -> fused: {}", p)?;
                }
                Ok(())
            }
        }
    }
}
```

- [ ] **Step 4: Run snapshot test**

Run: `cargo test --package compiler --lib verbose_uir_snapshot_matches_expected_format`
Expected: PASS — but if it fails on whitespace drift, look at the actual rendered output in the test failure message and update the `expected` literal in step 1's test to match (do not change the implementation to match a guessed format).

- [ ] **Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 221.

---

## Task 16: `--uir-verbose` CLI flag + smoke test

**Files:**
- Create: `nflc/tests/cli_parse.rs`
- Modify: `nflc/src/main.rs`

- [ ] **Step 1: Write failing CLI smoke test**

Create `nflc/tests/cli_parse.rs`:

```rust
//! CLI integration tests for `nflc parse` UIR rendering modes.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn parse_with_uir_verbose_renders_summary_and_extern_math() {
    let output = Command::new(nflc_bin())
        .args([
            "parse",
            "../tests/fixtures/classifier.nfl",
            "--uir-verbose",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("uir-verbose summary"),
        "stdout missing verbose header:\n{stdout}"
    );
    assert!(
        stdout.contains("calls-extern-math:"),
        "stdout missing calls-extern-math line:\n{stdout}"
    );
    assert!(
        stdout.contains("node count:"),
        "stdout missing node count line:\n{stdout}"
    );
}
```

> **Note:** Do not assert on `-> fused: <op>` here — `classifier.nfl`'s
> Display via VerboseUir runs PRE-PASS (parse subcommand, no fusion),
> so no fused post-ops will be present. The snapshot test in Task 15
> uses a hand-built UIR; the CLI smoke test here uses the parse path.

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package nflc --test cli_parse parse_with_uir_verbose_renders_summary_and_extern_math`
Expected: FAIL — flag `--uir-verbose` not recognised; nflc exits with error.

- [ ] **Step 3: Add flag to nflc**

In `nflc/src/main.rs`, locate the argument-parsing block (around line 90-150 where existing flags `--no-passes`, `--passes`, `--uir` are handled).

Add a flag variable near the existing ones:

```rust
let mut print_uir_verbose = false;
```

Add a match arm in the parse loop, parallel to the existing `--uir` arm:

```rust
"--uir-verbose" => {
    print_uir_verbose = true;
}
```

In the `parse` subcommand handler (search for where `print_uir` is consumed and a `Display`-rendering branch fires), add a sibling branch:

```rust
if print_uir {
    println!("{}", uir);
} else if print_uir_verbose {
    use compiler::ir::types::VerboseUir;
    println!("{}", VerboseUir(&uir));
}
```

In the help text (currently around lines 60-65, where `--no-passes` etc are documented), add a line for `--uir-verbose`:

```rust
println!("                          [--uir-verbose]    Print UIR with annotated metadata");
```

- [ ] **Step 4: Run smoke test, verify pass**

Run: `cargo test --package nflc --test cli_parse parse_with_uir_verbose_renders_summary_and_extern_math`
Expected: PASS.

- [ ] **Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 222.

---

## Task 17: Mutual-exclusion `--uir`/`--uir-verbose` + smoke test

**Files:**
- Modify: `nflc/src/main.rs`
- Test: `nflc/tests/cli_parse.rs`

- [ ] **Step 1: Write failing mutual-exclusion smoke test**

Append to `nflc/tests/cli_parse.rs`:

```rust
#[test]
fn parse_uir_and_uir_verbose_are_mutually_exclusive() {
    let output = Command::new(nflc_bin())
        .args([
            "parse",
            "../tests/fixtures/classifier.nfl",
            "--uir",
            "--uir-verbose",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(
        !output.status.success(),
        "expected failure exit but got success; full output: {:?}",
        output
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--uir") && stderr.contains("--uir-verbose"),
        "stderr must mention both flags in the mutual-exclusion error:\n{stderr}"
    );
}
```

- [ ] **Step 2: Run, verify fail**

Run: `cargo test --package nflc --test cli_parse parse_uir_and_uir_verbose_are_mutually_exclusive`
Expected: FAIL — currently both flags can coexist (each just sets a bool), so `nflc parse` exits 0.

- [ ] **Step 3: Add mutual-exclusion check**

In `nflc/src/main.rs`, locate the existing post-parse mutual-exclusion check (`--no-passes` and `--passes` are already mutually exclusive at line 138-141). Add a parallel check after parsing all flags:

```rust
if print_uir && print_uir_verbose {
    return Err("--uir and --uir-verbose are mutually exclusive".to_string());
}
```

- [ ] **Step 4: Run smoke test, verify pass**

Run: `cargo test --package nflc --test cli_parse parse_uir_and_uir_verbose_are_mutually_exclusive`
Expected: PASS.

- [ ] **Step 5: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass. Count: 223.

---

## Task 18: Update `docs/language_reference/uir.md` "Viewing UIR" section

**Files:**
- Modify: `docs/language_reference/uir.md`

- [ ] **Step 1: Inspect current uir.md structure**

Run: `grep "^#" docs/language_reference/uir.md | head -20`
Expected: Section list. Identify a natural insertion point (likely after the section that describes the UIR data model and before any internal implementation detail sections).

- [ ] **Step 2: Append "Viewing UIR" section**

Add to `docs/language_reference/uir.md` (insert at the natural location identified in Step 1):

```markdown
## Viewing UIR

The `nflc parse` subcommand exposes two human-readable rendering modes
for the UIR a source file produces:

- **`--uir`** — compact form. One line per node; shape, operands,
  attributes, and any `fused_post_ops` on a single line. Suitable
  for inspection of small models or quick debugging.

- **`--uir-verbose`** — annotated form. Adds a top-level summary
  block (model count, total node count, `calls-extern-math: yes/no`),
  a per-model summary block (node count, `calls-extern-math`), and
  breaks each fused post-op out onto its own indented line prefixed
  with `-> fused: <op>` for visibility. Suitable for understanding
  models with active fusion or models with unfamiliar structure.

Both modes are mutually exclusive — passing both flags in a single
invocation is a CLI error.

`calls-extern-math` reports whether the model contains any operation
that requires linking against external math at codegen time. In
NFL v0.1 this is true iff the model has a standalone `Softmax`
op or any node carrying `PostOp::SoftmaxRow` in `fused_post_ops`
(softmax requires `expf` from libm). The predicate is UIR-level —
profile-independent — and is also exposed as a method on `Uir` and
`UirModel` for programmatic use.

Both rendering modes consume the UIR as built by `compiler::ir::build`,
**before** any pass pipeline runs. To inspect the post-pipeline UIR,
use `nflc compile --profile <p>` and read the emitted assembly, or
extend `--uir-verbose` to compose with passes in a future milestone.

The dedicated viewer tool, when it ships, will consume the same
`Display`/`VerboseUir` output as a starting point.
```

- [ ] **Step 3: Verify the file renders cleanly**

Run: `grep -c "^## " docs/language_reference/uir.md`
Expected: Section count incremented by 1.

---

## Task 19: Verify workspace; Commit 3

**Files:** none (verification only)

- [ ] **Step 1: `cargo fmt --all`**

Run: `cargo fmt --all`

- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: Exit 0.

- [ ] **Step 3: `cargo test --workspace`**

Run: `cargo test --workspace`
Expected: All 223 tests pass.

- [ ] **Step 4: Stage and commit**

Run:

```bash
git add compiler/src/ir/types.rs \
        compiler/src/ir/tests.rs \
        nflc/src/main.rs \
        nflc/tests/cli_parse.rs \
        docs/language_reference/uir.md
```

Then commit:

```bash
git commit -m "$(cat <<'EOF'
feat(m8/viewer): UIR-verbose annotation mode

PROJECT_SPEC milestone row 8 deliverable: a human-readable viewer
for UIR. Existing `Display for Uir` plus `nflc parse --uir` provided
the compact baseline. v0.1 augments this with a verbose mode
(`--uir-verbose`) that surfaces UIR-level metadata and makes
fusion structure visually prominent.

Additions:
- compiler::ir::types::{VerboseUir, VerboseModel, VerboseNode}
  newtype wrappers, each with their own Display impl. Idiomatic Rust
  composition (write! threading); no API pollution on Uir/UirModel/Node.
- compiler::ir::types::Uir::calls_extern_math() and
  compiler::ir::types::UirModel::calls_extern_math() — UIR-level
  predicate (true iff standalone Softmax or fused SoftmaxRow present).
  Logic mirrors the existing profile-side `node_uses_softmax` in
  profiles/arm64/src/buffer.rs; deduplication is backlog OQ-NEW.
- nflc CLI: `--uir-verbose` flag on `parse` subcommand, mutually
  exclusive with `--uir`.

Verbose output adds: top-level summary (model count, total nodes,
calls-extern-math), per-model summary (node count, calls-extern-math),
fused post-ops on separate indented lines (`-> fused: <op>`).

Tests:
- compiler/src/ir/tests.rs::calls_extern_math_{true_for_standalone_softmax,
  false_for_linear_only, true_for_fused_softmax_row}
- compiler/src/ir/tests.rs::verbose_uir_snapshot_matches_expected_format
- nflc/tests/cli_parse.rs::parse_with_uir_verbose_renders_summary_and_extern_math
- nflc/tests/cli_parse.rs::parse_uir_and_uir_verbose_are_mutually_exclusive

docs/language_reference/uir.md gets a new "Viewing UIR" section.

Test count: 217 → 223.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Verify commit landed cleanly**

Run: `git log --oneline -3 && git status`
Expected: Three M8 feature commits in sequence; working tree clean.

---

## Task 20: Holistic review subagent dispatch

**Files:** none (review only)

- [ ] **Step 1: Dispatch holistic-review subagent**

Use the `Agent` tool with `subagent_type: Explore` (or general-purpose) to dispatch a single full-tree audit. Prompt the subagent with:

> Audit the full M8 worktree (`claude/sad-tesla-01188d`) for issues across
> spec, structure, cross-cutting consistency, docs, and process. Specifically:
>
> 1. Spec drift: does the actual code in `profiles/arm64/src/ops/`, `compiler/src/ir/types.rs`, `nflc/src/main.rs` match `docs/superpowers/specs/2026-05-06-m8-codegen-hardening-and-viewer-design.md`? List any divergence.
> 2. Structure: are the three M8 commits self-coherent (each ships independently green)? Do any imports leak between commits? Are tests for each commit actually under that commit's scope?
> 3. Cross-cutting: does `node_uses_softmax` in `profiles/arm64/src/buffer.rs:81-94` still exist (intentionally — backlog OQ-NEW)? Are #[non_exhaustive] markers on StdOp/PostOp still respected by all matches outside `compiler` crate? Do all five workspace error types still implement std::error::Error?
> 4. Docs: are CLAUDE.md, PROJECT_SPEC.md, DEVLOG.md, uir.md, arm64.md aligned with M8 changes? Anything stale (e.g. test count, principle 5 reference, milestone row text)?
> 5. Process: do the three feature commits follow the M5/M6/M7 atomic-task-pack convention (each green, each independently revertable, no cross-commit breaking changes)?
>
> Categorise findings as:
> - **close-in-M8** (small fixes, quick to address before close)
> - **carry-forward** (legitimate but out-of-scope, files in backlog)
> - **deviation-noted** (deliberate exception, no action needed)
>
> Cap report at ~600 words. Reference exact file:line locations for any drift findings.

- [ ] **Step 2: Review subagent report**

Read the report. For each finding, decide: close-in-M8, carry-forward, or deviation-noted.

- [ ] **Step 3: Record findings**

Note the close-in-M8 list for Task 21. Note carry-forward items for Task 23 (CLAUDE.md updates).

If no close-in-M8 findings → skip Task 21 directly to Task 22.

---

## Task 21 (conditional): `chore(m8/holistic)` close-in-M8 findings

**Files:** depends on findings (from Task 20 report)

Skip this task if Task 20 found no close-in-M8 issues.

- [ ] **Step 1: Apply each close-in-M8 fix**

For each finding from the report, apply the small fix inline. Each fix is typically < 20 lines. Common patterns from M5c/M6/M7 holistic reviews: stale doc comments referencing prior milestones, drift between spec and code on minor identifiers, missing `#[non_exhaustive]` wildcard arms in newly-added matches.

- [ ] **Step 2: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`**

Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: All clean.

- [ ] **Step 3: Stage and commit**

Stage only the files actually modified by the holistic-fixes. Commit:

```bash
git commit -m "$(cat <<'EOF'
chore(m8/holistic): close drift-fix findings before M8 closeout

Holistic review of M8 worktree (post-Commit 3) surfaced N findings;
M of them close-in-M8. Each fix is small (< 20 lines) and addresses:
[list each finding briefly]

Mirror of M7's 4974cd7 — same convention.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Replace the bullet list and counts with the actual findings.

---

## Task 22: Update `PROJECT_SPEC.md` M8 row

**Files:**
- Modify: `PROJECT_SPEC.md`

- [ ] **Step 1: Locate M8 row in milestones table**

Run: `grep -n "^| 8 " PROJECT_SPEC.md`
Expected: One match — the current M8 row reads `| 8 | Human-readable viewer v0.1 | Show UIR in annotated human-readable format |`.

- [ ] **Step 2: Replace with detailed description**

Replace the single-line M8 row with the multi-clause description following the M5/M6/M7 row granularity:

```markdown
| 8 | ARM64 codegen hardening + viewer v0.1 (complete) | Two arm64 codegen bugs closed: dropout-as-output now emits an explicit copy-loop via new `ops/dropout.rs::emit_dropout_copy` (BufferLoc::OutputReg branch in walk_model::Dropout); dim-immediate encoding routed uniformly through `asm::emit_imm32` across 17 sites (12 cmp + 5 mov), with hoist-outside-loop (Group A: relu, dropout-copy, matmul body) and re-materialise-at-loop-top (Group B: standalone softmax, fused RowWise tail) placement strategies; new fixtures `large_classifier_{k,n}.nfl` (k=8192 / out=5120) prove > 4095 dim now compiles. Viewer v0.1: `compiler::ir::types::{VerboseUir, VerboseModel, VerboseNode}` newtype wrappers + `Uir::calls_extern_math` / `UirModel::calls_extern_math` predicate; new `nflc parse --uir-verbose` flag (mutually exclusive with `--uir`) renders annotated UIR with top-level + per-model summary, fused post-ops on separate indented lines. `docs/language_reference/uir.md` gets new "Viewing UIR" section. Test count: 208 → 223. |
```

> **Note:** Adjust the test count if Task 21's close-in-M8 fixes added or removed any tests.

---

## Task 23: Update `CLAUDE.md` "Current Status" + Design Principle 5

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Design Principle 5 reference**

Run: `grep -n "(M8+)" CLAUDE.md`
Expected: One or more matches in the "Design Principles" section. Each `(M8+)` becomes `(M9+)` because viewer v0.1 has shipped.

For each match, replace inline.

- [ ] **Step 2: Rewrite "Current Status" section**

Locate the "Current Status" section. Replace with a description following the M7-closure template (look at existing CLAUDE.md "Current Status" pre-M8 for shape). New content should cover:

- M8 closed: codegen hardening (HIGH dropout-as-output + MEDIUM dim-immediate) + viewer v0.1.
- Test count: 208 → 223.
- 3-crate workspace unchanged.
- New carry-forward list: OQ-NEW (`node_uses_softmax` duplicate), OQ-7 (per-pass Result cleanup), OQ-8 (lift rewriter to compiler/ir/), OQ-9 (NodeMutation), profile-level viewer annotations, MACHO_SYM_PREFIX rename. Each with its trigger.
- Next steps: M9 brainstorm in fresh worktree once M8 merges.

- [ ] **Step 3: Verify no other stale (M8+) references remain**

Run: `grep -n "M8\|M9" CLAUDE.md | head -20`
Expected: All references reflect the new state (M8 complete, M9 next).

---

## Task 24: Update `DEVLOG.md` closeout entry

**Files:**
- Modify: `DEVLOG.md`

- [ ] **Step 1: Add new entry at the top of the log**

Insert at the top of `DEVLOG.md` (after the header and format-template section), preserving reverse-chronological order:

```markdown
## YYYY-MM-DD — Milestone 8 closed: arm64 codegen hardening + viewer v0.1

### What was done
- **`profiles/arm64/src/ops/dropout.rs`** — new `emit_dropout_copy`
  (mirror of `emit_relu` minus `fmax`). Triggered from a new
  `BufferLoc::OutputReg` branch in `codegen.rs::walk_model`'s
  `StdOp::Dropout` arm. Closes HIGH-severity bug: dropout placed at
  `model.output` previously left the caller's output buffer
  uninitialised.
- **`profiles/arm64/src/ops/{linear,relu,softmax}.rs`** — 17 immediate
  sites (12 cmp + 5 mov) routed through `asm::emit_imm32`. Two
  placement strategies: Group A hoist-outside-loop for bl-free
  emitters (relu, matmul body) with distinct registers per nesting
  level (x10/x15/x16); Group B re-materialise-at-loop-top for
  bl-containing emitters (standalone softmax, RowWise tail) where
  `bl _expf` clobbers caller-saved x10. Closes MEDIUM-severity bug:
  any production-scale dim (transformer hidden_dim 4096+, LLM vocab
  30k+, classifier with > 4095 classes) previously failed to
  assemble or failed silently.
- **`compiler/src/ir/types.rs`** — three newtype wrappers
  (`VerboseUir`, `VerboseModel`, `VerboseNode`), each with their own
  `Display` impl. Plus `calls_extern_math` predicate methods on
  `Uir` and `UirModel`. UIR-level predicate (no profile coupling).
  Default `Display` for the underlying types unchanged.
- **`nflc/src/main.rs`** — new `--uir-verbose` flag on `parse`
  subcommand, mutually exclusive with `--uir`. Help text updated.
- **`docs/language_reference/uir.md`** — new "Viewing UIR" section
  documenting both flags and the `calls-extern-math` semantics.
- **`docs/profile_guide/arm64.md`** — two short paragraphs:
  "Dropout-as-output copy" and "Dim-immediate uniformity".
- **New fixtures:** `tests/fixtures/{dropout_only,large_classifier_k,
  large_classifier_n}.nfl`.
- **6 new test groups, ~15 new tests:**
  asm-shape per emitter (Commit 1 + 4 in Commit 2), 4 FFI integration
  tests (2 dropout-only variants + 2 large_classifier), 3 predicate
  sub-cases, 1 verbose snapshot, 2 CLI smoke (verbose + mutual-exclusion).
  Test count 208 → 223.

### Decisions made
- **Single PR with 3 atomic feature commits + holistic-review +
  closeout commits**, mirroring M5/M6/M7. No cross-commit
  dependencies — each commit is independently revertable and green.
- **`emit_dropout_copy` uses `emit_imm32` from birth** — Commit 1's
  new emitter ships with the new pattern, so Commit 2 patches
  exactly 17 pre-existing sites, not 18. No "TODO patch in Commit 2"
  debt.
- **Mov-site replacement reuses hoisted registers in Group A** —
  `mov x8, x15` / `mov x8, x16` instead of re-materialising via
  `emit_imm32`. Principle: avoid illegal immediates, not "always
  call the helper". Single instruction vs 1-2 movz/movk.
- **Group B accepts 1-2 movz/movk per loop iteration** in
  bl-containing loops. Adding x10 to the prologue's callee-saved
  set was rejected as out-of-scope blast radius; `bl _expf` is
  hundreds of cycles, < 1% relative overhead.
- **Newtype wrappers over `fmt_verbose` methods** — idiomatic Rust
  composition, no API pollution. Default `Display` unchanged.
- **`calls_extern_math` placed on UIR side, predicate logic
  duplicated with profile-side `node_uses_softmax`.** Deduplication
  is backlog OQ-NEW; trigger is next predicate-logic change (e.g.,
  adding `tanh`-via-libm).
- **No new error variant for dim-out-of-range** — `emit_imm32` already
  asserts on u32::MAX, ~1000× any realistic NN dim. YAGNI.
- **`--uir-verbose` documented in `uir.md`** (UIR rendering interface)
  rather than `arm64.md` (which is profile-specific). Reasoning:
  viewer is profile-agnostic; `arm64.md` only gets the codegen-
  hardening section.

### Problems encountered
[Fill in during execution. Likely candidates:
 - Snapshot test whitespace drift in Task 15 — adjust expected literal
 - Existing test assertions on cmp x9, #4 etc that needed updating
   in Tasks 6/7/8/9 — caught by full-suite run
 - Holistic-review findings, if any]

### Next step
M9 brainstorming runs in a fresh worktree once M8 merges. Carry-
forward candidates: OQ-NEW (lift `node_uses_softmax` to single
source), OQ-7 (per-pass Result cleanup), OQ-8 (lift rewriter
to compiler/src/ir/), OQ-9 (NodeMutation generalisation),
profile-level viewer annotations, MACHO_SYM_PREFIX rename when
second profile starts, attention-pattern grammar (NFL v0.2),
bare-metal target, BuildError::span() + Diagnostic trait.
```

> **Note:** Replace `YYYY-MM-DD` with the actual closeout date. Replace the placeholder under "Problems encountered" with what actually happened during execution.

---

## Task 25: Update `docs/profile_guide/arm64.md` codegen-hardening section

**Files:**
- Modify: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Locate end of file**

Run: `wc -l docs/profile_guide/arm64.md && tail -5 docs/profile_guide/arm64.md`
Expected: Identify the last section heading; new content goes after.

- [ ] **Step 2: Append two short paragraphs**

At the end of `docs/profile_guide/arm64.md`, append:

```markdown
## M8 codegen hardening

### Dropout-as-output copy

Dropout is identity at inference time, and `assign_buffers`
returns `BufferLoc::Alias(operand)` for any Dropout node that is
NOT the model output — downstream ops read the operand's buffer
directly, no asm needed. When a Dropout node IS `model.output`,
however, alias-redirection no longer applies: `assign_buffers`
returns `BufferLoc::OutputReg` (the caller's `x2` pointer), and
codegen must explicitly copy the operand's buffer into it. The
`StdOp::Dropout` arm in `codegen.rs::walk_model` branches on
`dst_loc` and emits a copy-loop via `ops/dropout.rs::emit_dropout_copy`
in this case (mirror of `emit_relu`'s structure minus `fmax`).
This path is exercised only with `--no-passes` and dropout placed
at the model output; the default pipeline's `EliminateDropout` pass
removes the dropout before codegen sees it.

### Dim-immediate uniformity

ARM64 `cmp Xn, #imm` encodes a 12-bit immediate (0-4095, optionally
shifted by 12); `mov Xn, #imm` encodes 16-bit (0-65535). All loop-
bound and stride dimensions in matmul, relu, softmax, and the fused
RowWise softmax tail flow through `asm::emit_imm32` (movz + optional
movk) instead of literal-imm encoding. Two placement strategies:

- **Group A (bl-free loops)**: hoist materialise once before the
  loop label, register-form `cmp` inside. Matmul body uses three
  distinct registers (x10 ← b, x15 ← n, x16 ← k); stride-load movs
  reuse the hoisted regs (`mov x8, x16` etc) instead of re-
  materialising. Inner-loop cmp has zero runtime cost.
- **Group B (bl-containing loops)**: re-materialise into x10 at
  each loop top (after label, before cmp). `bl _expf` clobbers
  caller-saved registers including x10, so hoisting outside the
  loop is impossible without expanding the prologue's callee-saved
  set (deferred). 1-2 movz/movk per iteration is < 1% overhead vs
  the cost of `bl _expf` itself.

`emit_imm32` asserts `value <= u32::MAX as usize`, providing a
clear failure mode for hypothetical dimensions beyond 4 billion
elements (~1000× any realistic NN dim).
```

---

## Task 26: Closeout commit + push branch + open PR

**Files:** none (verification + git only)

- [ ] **Step 1: Final workspace verification**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: All clean.

- [ ] **Step 2: Stage doc updates**

Run:

```bash
git add PROJECT_SPEC.md CLAUDE.md DEVLOG.md docs/profile_guide/arm64.md
```

- [ ] **Step 3: Commit closeout**

```bash
git commit -m "$(cat <<'EOF'
chore(m8): close Milestone 8 — full cycle complete

M8 ships:
- Two arm64 codegen bugs fixed (dropout-as-output, dim-immediate
  uniformity across 17 sites in 5 emitters).
- Viewer v0.1 with --uir-verbose flag, calls_extern_math predicate,
  and three newtype wrappers (VerboseUir/Model/Node).
- Test count 208 → 223.

Documentation closeout:
- PROJECT_SPEC.md: M8 milestone row replaced with detailed
  description following M5/M6/M7 granularity.
- CLAUDE.md: "Current Status" rewritten reflecting M8 closure;
  Design Principle 5 reference (M8+) → (M9+).
- DEVLOG.md: M8 closeout entry under standard template.
- docs/language_reference/uir.md: new "Viewing UIR" section
  documenting --uir / --uir-verbose modes and calls-extern-math
  semantics. (Committed in feature commit; mentioned here for
  closeout completeness.)
- docs/profile_guide/arm64.md: two new paragraphs covering
  dropout-as-output copy and dim-immediate uniformity.

Carry-forward to M9+:
- OQ-NEW: lift node_uses_softmax → calls_extern_math single source
- OQ-7: per-pass Result cleanup (carried from M7)
- OQ-8: lift rewriter.rs to compiler/src/ir/ (carried from M7)
- OQ-9: NodeMutation generalisation (carried from M7)
- Profile-level viewer annotations
- MACHO_SYM_PREFIX rename (when second profile starts)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push branch**

Run: `git push -u origin claude/sad-tesla-01188d`
Expected: Push succeeds.

- [ ] **Step 5: Open PR via gh CLI**

Run:

```bash
gh pr create --title "M8: ARM64 codegen hardening + viewer v0.1" --body "$(cat <<'EOF'
## Summary

- **Commit 1 — `feat(m8/arm64-fix): correct dropout-as-output codegen`** — closes HIGH-severity bug where Dropout at `model.output` left the caller's buffer uninitialised. New `ops/dropout.rs::emit_dropout_copy` triggered from a `BufferLoc::OutputReg` branch in `walk_model`.
- **Commit 2 — `feat(m8/arm64-fix): hoist dim immediates through emit_imm32`** — closes MEDIUM-severity bug where 17 cmp/mov immediate sites used literal-imm encoding that fails on any production-scale dim. Routed uniformly through `asm::emit_imm32` with Group A (hoist outside loop) and Group B (re-materialise at loop top) placement.
- **Commit 3 — `feat(m8/viewer): UIR-verbose annotation mode`** — ships PROJECT_SPEC milestone row 8: `--uir-verbose` flag, `calls_extern_math` predicate, three newtype Display wrappers.
- **`chore(m8/holistic)`** — close-in-M8 holistic-review findings (if any).
- **`chore(m8): close Milestone 8`** — PROJECT_SPEC, CLAUDE.md, DEVLOG, arm64.md updates.

Test count: 208 → 223.

## Test plan

- [ ] `cargo fmt --all -- --check` clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo test --workspace` — all 223 tests pass
- [ ] `dropout_only_b{2_k4,1_k8}_no_passes` FFI tests verify dropout-as-output bug fix
- [ ] `large_classifier_{k_8192,n_5120}` FFI tests verify > 4095 dim handling
- [ ] All M4/M5/M6/M7 carryover FFI tests pass bit-exact (regression guarantee for Commit 2)
- [ ] CLI smoke tests verify `--uir-verbose` rendering and mutual-exclusion error

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Verify PR URL**

Output the PR URL from the gh command's stdout. Verify CI/checks are green on GitHub.

---

## Done. What's next?

Once the PR is merged into `main`, M9 brainstorming runs in a fresh worktree per the project protocol (CLAUDE.md "M{N} brainstorming runs in a fresh worktree once M{N-1} merges").

M9 candidates carried from M8 closeout:
- **OQ-NEW** — Lift `node_uses_softmax` (in `profiles/arm64/src/buffer.rs`) to call `Uir::calls_extern_math()` introduced in M8. Trigger: next change to either side's predicate logic.
- **OQ-7** — Per-pass `Result<UirModel, PassError>` cleanup. Trigger: first real `Err`-case in pass-level logic.
- **OQ-8** — Lift `compiler/src/passes/rewriter.rs` to `compiler/src/ir/`. Trigger: non-pass UIR-rewrite consumer appears.
- **OQ-9** — Generalise `producer_post_ops` to `enum NodeMutation`. Trigger: fourth pass needs non-PostOp producer mutation.
- **Profile-level viewer annotations** — per-node footprint, stack frame, callee-saved set. Trigger: user request OR x86_64 profile starts.
- **`MACHO_SYM_PREFIX` rename** — when second profile starts.
- **Attention-pattern extension** — Q/K/V projections, scaled dot-product, axis-N softmax. Requires NFL v0.2 grammar.
- **`FuseLinearPostOp` consolidation** (M5c OQ-1) — third access pattern OR second RowWise post-op.
- **Bare-metal target** (M5c OQ-3) — Taylor-series `expf`.
- **`BuildError::span()` + `Diagnostic` trait** (M5c OQ-4).
