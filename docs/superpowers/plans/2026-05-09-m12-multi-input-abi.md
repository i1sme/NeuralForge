# M12 Implementation Plan: Multi-Input Function ABI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-input function ABI (N=1..4 inputs) to both NeuralForge architecture profiles via per-profile `AbiContext` connector, while preserving bit-exact assembly output for all M3-M11 single-input fixtures.

**Architecture:** Per-profile module `abi.rs` introduces `AbiContext` carrying `n_inputs: usize`. It owns the mapping `BufferLoc::InputReg(usize)` → physical register, the conservative caller-saved spill set around FFI calls, and SP-aligned save/restore code emission. `walk_model` constructs `AbiContext` once at function entry and threads `&abi` through every op-emitter. Op-emitters never reference physical ABI registers directly. `emit_matmul` undergoes a structural simplification: per-iter slice pointers move from ABI registers (x1/x2/x4 arm64; %xmm6-8 x86_64) onto non-ABI scratch registers, eliminating the M10 outer-loop `stp/ldp` spill block.

**Tech Stack:** Rust 2021 (Cargo workspace, 5 members), arm64 (AAPCS) + x86_64 (SysV AMD64) assembly codegen, `cc` for FFI test compilation, `libloading` for dlopen-based testing.

---

## Spec Reference

This plan implements [`docs/superpowers/specs/2026-05-09-m12-multi-input-abi-design.md`](../specs/2026-05-09-m12-multi-input-abi-design.md). Each task references its spec section in parentheses (e.g. "spec §5.2"). Acceptance gates are numbered #1–#25 in spec §10.

**Key terminology:**
- `N` = number of inputs declared in the model (`model.inputs.len()`).
- `ffi_save_set()` = `&INPUT_REGS[..N+2]` = conservative caller-saved set spilled around FFI calls.
- "ABI register" = any register in `INPUT_REGS[..N+2]`; for arm64 these are `x0..x_{N+1}`, for x86_64 SysV `%rdi/%rsi/%rdx/%rcx/%r8/%r9` truncated at N+2.

---

## File Structure

### Files to Create

| Path | Purpose | Spec ref | LoC |
|------|---------|----------|-----|
| `profiles/arm64/src/abi.rs` | AbiContext for arm64 | §5.2 | ~120 |
| `profiles/x86_64/src/abi.rs` | AbiContext for x86_64 | §5.2 | ~120 |
| `tests/fixtures/two_input_matmul.nfl` | N=2 sanity | §7.1 | 15 |
| `tests/fixtures/multi_input_attention.nfl` | N=3 acceptance | §7.2 | 20 |
| `tests/fixtures/negative/too_many_inputs.nfl` | N=5 negative | §7.3 | 12 |
| `profiles/arm64/tests/golden/<fixture>.s` × 11 | N=1 regression goldens | §10.2 | per-fixture |
| `profiles/x86_64/tests/golden/<fixture>.s` × 11 | N=1 regression goldens | §10.2 | per-fixture |

### Files to Modify

| Path | Change |
|------|--------|
| `profile-api/src/lib.rs` | `FnSig.input_floats: usize` → `inputs_floats: Vec<usize>`; new `LowerError::TooManyInputs` variant |
| `compiler/src/ir/build.rs` | Build `inputs_floats` from `model.inputs.iter()` |
| `profiles/arm64/src/lib.rs` | `pub mod abi;` |
| `profiles/arm64/src/buffer.rs` | `BufferLoc::InputReg(usize)`; `assign_buffers` looks up index |
| `profiles/arm64/src/codegen.rs` | `walk_model` constructs `AbiContext`, threads `&abi` through op dispatch |
| `profiles/arm64/src/ops/linear.rs` | `materialise_ptr` becomes `AbiContext` method; ops use `&abi` |
| `profiles/arm64/src/ops/matmul.rs` | Per-iter slice ptrs to non-ABI regs (§9.1 rework) |
| `profiles/arm64/src/ops/softmax.rs` | Replace manual stp/ldp with `abi.emit_ffi_save/restore` |
| `profiles/arm64/src/ops/{mulscalar,relu,dropout}.rs` | Use `abi` |
| `profiles/arm64/src/tests.rs` | AbiContext unit tests |
| `profiles/arm64/tests/integration.rs` | Multi-input integration tests |
| `profiles/x86_64/*` | Mirror of arm64 |
| `bench/src/main.rs` | Per-arity dispatch + seed cascade (§9.6) |
| `docs/profile_guide/{arm64,x86_64}.md` | "Multi-Input ABI" section |
| `docs/language_reference/{uir,grammar}.md` | Multi-input notes |
| `PROJECT_SPEC.md`, `CLAUDE.md`, `DEVLOG.md` | Closure entries |

---

## Pre-commit Ritual (run before EVERY commit)

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All three MUST exit 0. After each commit:

```bash
git status        # Confirm clean
git log --oneline -5
```

---

## Group A — Foundation (commit 1)

Migrate `profile-api` and `compiler/src/ir/build.rs` to multi-input shape. No codegen changes yet — both profiles continue to use `model.inputs.first()` internally; changes here are type-shape only. After Group A, all 344 baseline tests still pass, plus 1-2 new tests for the new `LowerError` variant.

**LoC delta:** ~30 production + ~10 test = ~40.
**Test count delta:** +2 (FnSig round-trip, LowerError::TooManyInputs Display).

### Task A.1: Generate N=1 baseline goldens (BEFORE any code change)

**Why first.** The N=1 regression invariant (spec §10.2, gate #4) requires that all M3-M11 fixtures generate bit-exact assembly under M12 codegen vs M11 baseline. To verify, we need M11 baselines committed BEFORE M12 changes start. This task uses the current (pre-M12) codegen to produce goldens.

**Files:**
- Create: `profiles/arm64/tests/golden/{tiny_mlp,m4_linear_relu,mixed_args,softmax_with_bias,dropout_only,classifier,large_classifier_k,large_classifier_n,pipeline_styles,comments,self_attention}.s` (11 files)
- Create: `profiles/x86_64/tests/golden/<same 11 names>.s` (11 files)

- [ ] **Step 1: Generate arm64 goldens via nflc**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/blissful-sanderson-72b165
mkdir -p profiles/arm64/tests/golden
for f in tiny_mlp m4_linear_relu mixed_args softmax_with_bias dropout_only classifier large_classifier_k large_classifier_n pipeline_styles comments self_attention; do
  cargo run -p nflc -- compile "tests/fixtures/${f}.nfl" --profile arm64 --emit asm > "profiles/arm64/tests/golden/${f}.s"
done
```

If `nflc compile --emit asm` does not exist, use whatever flag the current CLI supports to dump generated assembly to stdout (check `nflc --help`). If the flag must be added, defer this task to A.1.b (see below) — but the project's M3-M11 history suggests `--emit asm` or equivalent works.

- [ ] **Step 1.b (fallback if no --emit asm flag): use a small Rust binary**

Create a one-off binary `tools/dump_asm/main.rs` (this is throwaway — do not commit if Step 1 worked):

```rust
// tools/dump_asm/src/main.rs
use std::env::args;
use std::fs::read_to_string;
fn main() {
    let path = args().nth(1).expect("usage: dump_asm <fixture.nfl> <profile>");
    let prof = args().nth(2).expect("missing profile");
    let src = read_to_string(&path).unwrap();
    let nfl = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&nfl).unwrap();
    let asm = match prof.as_str() {
        "arm64"  => profiles_arm64::Arm64.lower(&uir).unwrap().source,
        "x86_64" => profiles_x86_64::X8664.lower(&uir).unwrap().source,
        _ => panic!("unknown profile"),
    };
    print!("{asm}");
}
```

(Adjust profile struct name + import path to match actual.) Then run for each fixture × profile.

- [ ] **Step 2: Generate x86_64 goldens** (same loop, replace `arm64` with `x86_64`)

- [ ] **Step 3: Verify goldens are non-empty and look like assembly**

```bash
wc -l profiles/arm64/tests/golden/*.s
head -20 profiles/arm64/tests/golden/classifier.s
```

Expected: each file ≥ 50 lines; head shows arm64 mnemonics (`stp`, `ldp`, `mov`, `bl`, etc.).

- [ ] **Step 4: Stage goldens — DO NOT COMMIT yet**

```bash
git add profiles/arm64/tests/golden/ profiles/x86_64/tests/golden/
git status
```

These will be committed at end of Group A together with the regression test that uses them (Task A.6).

### Task A.2: profile-api FnSig migration

**Spec ref:** §5, §8.

**Files:**
- Modify: `profile-api/src/lib.rs:24-40` (FnSig struct)
- Modify: `profile-api/src/lib.rs:140-152` (fn_sig_round_trip_through_debug test)

- [ ] **Step 1: Update FnSig struct**

```rust
// profile-api/src/lib.rs
/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// External symbol name without leading underscore. e.g. "nfl_forward_TinyMLP".
    pub name: String,
    /// Original UIR model name.
    pub model: String,
    /// Number of f32 elements per input buffer, in declaration order.
    /// Length = arity (number of inputs); for single-input models length = 1.
    pub inputs_floats: Vec<usize>,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
    /// Total number of f32 elements in the packed params buffer.
    pub params_floats: usize,
    /// Layout of the packed params buffer, one entry per parameter slot in
    /// UIR-node order.
    pub params_layout: Vec<ParamSlot>,
}
```

- [ ] **Step 2: Update the FnSig round-trip test**

```rust
#[test]
fn fn_sig_round_trip_through_debug() {
    let s = FnSig {
        name: "f".into(),
        model: "M".into(),
        inputs_floats: vec![1],
        output_floats: 1,
        params_floats: 0,
        params_layout: vec![],
    };
    let dbg = format!("{:?}", s);
    assert!(dbg.contains("FnSig"));
    assert!(dbg.contains("inputs_floats: [1]"));
}
```

- [ ] **Step 3: Run profile-api tests — expect FAILURE in dependent crates**

```bash
cargo test -p profile-api
```

Expected: profile-api passes (5 tests). But:

```bash
cargo build --workspace 2>&1 | head -30
```

Expected: compile errors in `profiles/arm64/src/codegen.rs`, `profiles/x86_64/src/codegen.rs`, `bench/src/main.rs`, `compiler/src/ir/build.rs` — wherever `sig.input_floats` / `FnSig { input_floats: ... }` is used. These are fixed in subsequent tasks.

### Task A.3: Add LowerError::TooManyInputs variant

**Spec ref:** §4.3, §5.3.

**Files:**
- Modify: `profile-api/src/lib.rs:62-101` (LowerError enum + Display + span method)

- [ ] **Step 1: Add the variant + Display + span**

```rust
// profile-api/src/lib.rs
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    UnsupportedOp { op: String, span: Span },
    ShapeNotConcrete { span: Span },
    UnsupportedPostOp { op: String, span: Span },
    /// Model declared more inputs than the profile's ABI register window
    /// can hold without stack-spilling. M12 caps both profiles at N=4
    /// (max=4 in the variant).
    TooManyInputs { n: usize, max: usize, span: Span },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } => {
                write!(f, "operation '{}' is not supported by this profile", op)
            }
            LowerError::ShapeNotConcrete { .. } => write!(
                f,
                "internal: UIR shape was not fully resolved before lowering"
            ),
            LowerError::UnsupportedPostOp { op, .. } => {
                write!(f, "post-op '{}' is not supported by this profile", op)
            }
            LowerError::TooManyInputs { n, max, .. } => write!(
                f,
                "model declares {} inputs but this profile supports a maximum of {}",
                n, max
            ),
        }
    }
}

impl LowerError {
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::UnsupportedPostOp { span, .. } => *span,
            LowerError::TooManyInputs { span, .. } => *span,
        }
    }
}
```

- [ ] **Step 2: Add unit test**

```rust
// profile-api/src/lib.rs (#[cfg(test)] mod tests block)
#[test]
fn lower_error_too_many_inputs_display() {
    let e = LowerError::TooManyInputs {
        n: 5,
        max: 4,
        span: Span::new(1, 1),
    };
    let msg = format!("{}", e);
    assert!(msg.contains("5"), "got: {msg}");
    assert!(msg.contains("4"), "got: {msg}");
    assert!(msg.contains("not supported") || msg.contains("maximum"),
            "msg should explain the limit; got: {msg}");
}
```

- [ ] **Step 3: Run profile-api tests**

```bash
cargo test -p profile-api
```

Expected: PASS. Test count for profile-api: previously ~6, now ~7.

### Task A.4: compiler/src/ir/build.rs emits inputs_floats

This isn't strictly needed — `inputs_floats` is built BY the profiles when they construct FnSig. But `build.rs` may have helper logic relevant to multi-input clarity. Check first:

- [ ] **Step 1: grep for FnSig construction sites**

```bash
grep -rn "FnSig {" profiles/ compiler/
```

Expected: FnSig is constructed in `profiles/arm64/src/codegen.rs` and `profiles/x86_64/src/codegen.rs` (ONE site each, in `walk_model` or near it). `compiler/src/ir/build.rs` does NOT construct FnSig directly.

- [ ] **Step 2: If grep shows compiler/ ir/build.rs constructs FnSig**

Then update it. If not (expected), this task is a no-op — proceed to A.5.

### Task A.5: Update FnSig construction sites in profiles (arity = 1 path)

Both profile codegens construct FnSig once per model in `walk_model`. They currently use scalar `input_floats: usize`; migrate to `inputs_floats: Vec<usize>` with length 1.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs:~85-100` (FnSig construction inside walk_model)
- Modify: `profiles/x86_64/src/codegen.rs:~equivalent line range`

- [ ] **Step 1: arm64 — find FnSig construction**

```bash
grep -n "FnSig {" profiles/arm64/src/codegen.rs
```

Expected: one or two hits inside `walk_model`. Example currently:

```rust
let sig = FnSig {
    name: format!("nfl_forward_{}", model.name),
    model: model.name.clone(),
    input_floats,
    output_floats,
    params_floats,
    params_layout,
};
```

- [ ] **Step 2: arm64 — change to Vec<usize>**

```rust
let sig = FnSig {
    name: format!("nfl_forward_{}", model.name),
    model: model.name.clone(),
    inputs_floats: vec![input_floats],   // arity-1 in this commit; arity-N in Group B
    output_floats,
    params_floats,
    params_layout,
};
```

- [ ] **Step 3: x86_64 — same change**

```bash
grep -n "FnSig {" profiles/x86_64/src/codegen.rs
```

Apply equivalent edit.

- [ ] **Step 4: bench — update sig.input_floats consumer**

```bash
grep -n "input_floats" bench/src/main.rs
```

Expected: ~3-4 hits. Replace `sig.input_floats` with `sig.inputs_floats[0]` (assuming arity 1; full per-arity dispatch lands in Group E).

```rust
// bench/src/main.rs (relevant lines)
let mut input: Vec<f32> = vec![0.0; sig.inputs_floats[0]];   // was: sig.input_floats
```

Add a temporary assertion to make the arity-1-only assumption explicit:

```rust
assert_eq!(sig.inputs_floats.len(), 1,
    "bench in Group A only handles N=1; per-arity dispatch lands in Group E");
```

- [ ] **Step 5: Run cargo build**

```bash
cargo build --workspace 2>&1 | tee /tmp/m12-a5-build.log
```

Expected: clean build (zero errors, zero warnings).

- [ ] **Step 6: Run cargo test**

```bash
cargo test --workspace
```

Expected: all 344 baseline tests + 1 new (LowerError::TooManyInputs) = 345 PASS.

### Task A.6: Add N=1 regression test using staged goldens

**Files:**
- Modify: `profiles/arm64/src/tests.rs` (add new test function)
- Modify: `profiles/x86_64/src/tests.rs` (add new test function)

- [ ] **Step 1: arm64 regression test**

```rust
// profiles/arm64/src/tests.rs (append to existing #[cfg(test)] mod block)

/// N=1 regression invariant: every existing fixture must compile to
/// the EXACT same assembly under M12 codegen as the M11 baseline
/// (committed in profiles/arm64/tests/golden/<fixture>.s).
///
/// This test loops over a list of fixtures and asserts byte-exact
/// equality with the corresponding golden file. Spec §10.2 gate #4.
#[test]
fn n1_regression_all_fixtures_bit_exact() {
    let fixtures = [
        "tiny_mlp",
        "m4_linear_relu",
        "mixed_args",
        "softmax_with_bias",
        "dropout_only",
        "classifier",
        "large_classifier_k",
        "large_classifier_n",
        "pipeline_styles",
        "comments",
        "self_attention",
    ];
    for f in fixtures {
        let nfl_path = format!("../../tests/fixtures/{f}.nfl");
        let golden_path = format!("tests/golden/{f}.s");
        let src = std::fs::read_to_string(&nfl_path)
            .unwrap_or_else(|e| panic!("read {nfl_path}: {e}"));
        let nfl = compiler::parse(&src).unwrap();
        let uir = compiler::ir::build(&nfl).unwrap();
        let asm = crate::Arm64.lower(&uir).unwrap().source;
        let golden = std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("read {golden_path}: {e}"));
        if asm != golden {
            // Show first diverging line for diagnostic.
            for (i, (a, g)) in asm.lines().zip(golden.lines()).enumerate() {
                if a != g {
                    panic!(
                        "fixture {f}: divergence at line {}\n  generated: {a:?}\n  golden:    {g:?}",
                        i + 1
                    );
                }
            }
            // Fallback if length differs but prefix matches.
            panic!("fixture {f}: assembly differs from golden (length: gen={}, golden={})",
                asm.len(), golden.len());
        }
    }
}
```

- [ ] **Step 2: x86_64 regression test (mirror)**

```rust
// profiles/x86_64/src/tests.rs (append)
#[test]
fn n1_regression_all_fixtures_bit_exact() {
    // Same fixture list and pattern as arm64, but call crate::X8664 (or
    // whatever the x86_64 profile struct is named).
    // ... (copy structure from arm64, swap profile.lower call)
}
```

- [ ] **Step 3: Run regression tests**

```bash
cargo test --workspace n1_regression_all_fixtures_bit_exact
```

Expected: PASS for both profiles. (At this point in Group A, codegen has not been touched, so generated asm == golden trivially.)

- [ ] **Step 4: Run full test suite**

```bash
cargo test --workspace
```

Expected: 345 + 2 = 347 tests PASS.

### Task A.7: Group A pre-commit + commit

- [ ] **Step 1: Pre-commit ritual**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All three must exit 0.

- [ ] **Step 2: Stage and commit**

```bash
git add profile-api/src/lib.rs \
        profiles/arm64/src/codegen.rs \
        profiles/x86_64/src/codegen.rs \
        profiles/arm64/src/tests.rs \
        profiles/x86_64/src/tests.rs \
        profiles/arm64/tests/golden/ \
        profiles/x86_64/tests/golden/ \
        bench/src/main.rs

git commit -m "$(cat <<'EOF'
feat(m12): foundation — FnSig.inputs_floats + LowerError::TooManyInputs + N=1 regression goldens

profile-api FnSig migrates from scalar input_floats:usize to
inputs_floats:Vec<usize> (arity = inputs_floats.len()). Both profiles
construct arity-1 FnSig in this commit; multi-input codegen lands in
Groups B/C.

LowerError gains TooManyInputs { n, max, span } for the N>4 cap that
both profiles will enforce in Group B/C walk_model entry.

11 N=1 regression goldens per profile (22 total) are committed as the
baseline for spec §10.2 gate #4. The test
n1_regression_all_fixtures_bit_exact loops over all fixtures and
asserts bit-exact assembly == golden.

bench/src/main.rs reads sig.inputs_floats[0] with an explicit
assert_eq! for arity 1 — the per-arity dispatch lands in Group E.

Test count: 344 → 347 (+1 LowerError, +2 regression tests).

Spec ref: §5, §10.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
git log --oneline -1
git status
```

Expected: commit landed; working tree clean.

---

## Group B — arm64 codegen (commit 2)

The largest group. Introduces `AbiContext`, migrates `BufferLoc::InputReg` → `BufferLoc::InputReg(usize)`, threads `&abi` through `walk_model`, and reworks `emit_matmul` per spec §9.1. After Group B, all N=1 fixtures must still produce bit-exact goldens (tests/golden/) — this is the strongest correctness gate for the rework.

**LoC delta:** ~250 production + ~80 tests = ~330.
**Test count delta:** +14 (AbiContext alignment + LIFO + materialise + ffi_save_set across N∈{1..4}, no asm-stp in matmul body).

### Task B.1: Create profiles/arm64/src/abi.rs skeleton

**Files:**
- Create: `profiles/arm64/src/abi.rs`
- Modify: `profiles/arm64/src/lib.rs` (add `pub mod abi;`)

- [ ] **Step 1: Add module declaration in lib.rs**

```rust
// profiles/arm64/src/lib.rs (top-level mod declarations)
pub mod abi;
```

- [ ] **Step 2: Create abi.rs with INPUT_REGS const + AbiContext skeleton**

```rust
// profiles/arm64/src/abi.rs
// SPDX-License-Identifier: Apache-2.0

//! AAPCS argument-register abstraction for multi-input model ABI.
//!
//! See spec §5.2. INPUT_REGS lists the first 6 of the 8 AAPCS argument
//! registers (x0..x7); M12 caps inputs at N=4, so N+2 ≤ 6. Reserved
//! x6/x7 for future ABI extensions without re-reflowing the table.

use crate::buffer::BufferLoc;

/// Argument-register table. Order matches register-allocation order:
/// inputs in declaration order, then params, then output.
pub(crate) const INPUT_REGS: &[&str] = &["x0", "x1", "x2", "x3", "x4", "x5"];

/// Per-function ABI state. Constructed once at the top of `walk_model`
/// and threaded by `&abi` through every op-emitter.
pub(crate) struct AbiContext {
    pub n_inputs: usize,
}
```

(Methods added in Tasks B.2–B.5.)

- [ ] **Step 3: Verify build**

```bash
cargo build -p profiles-arm64
```

Expected: clean.

### Task B.2: Implement input_reg / params_reg / output_reg / ffi_save_set

**Files:**
- Modify: `profiles/arm64/src/abi.rs`
- Modify: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write failing tests**

```rust
// profiles/arm64/src/tests.rs (append)
use crate::abi::{AbiContext, INPUT_REGS};

#[test]
fn abi_input_reg_n1() {
    let abi = AbiContext { n_inputs: 1 };
    assert_eq!(abi.input_reg(0), "x0");
}

#[test]
fn abi_input_reg_n3() {
    let abi = AbiContext { n_inputs: 3 };
    assert_eq!(abi.input_reg(0), "x0");
    assert_eq!(abi.input_reg(1), "x1");
    assert_eq!(abi.input_reg(2), "x2");
}

#[test]
fn abi_params_reg_shifts_with_arity() {
    assert_eq!(AbiContext { n_inputs: 1 }.params_reg(), "x1");
    assert_eq!(AbiContext { n_inputs: 2 }.params_reg(), "x2");
    assert_eq!(AbiContext { n_inputs: 3 }.params_reg(), "x3");
    assert_eq!(AbiContext { n_inputs: 4 }.params_reg(), "x4");
}

#[test]
fn abi_output_reg_shifts_with_arity() {
    assert_eq!(AbiContext { n_inputs: 1 }.output_reg(), "x2");
    assert_eq!(AbiContext { n_inputs: 2 }.output_reg(), "x3");
    assert_eq!(AbiContext { n_inputs: 3 }.output_reg(), "x4");
    assert_eq!(AbiContext { n_inputs: 4 }.output_reg(), "x5");
}

#[test]
fn abi_ffi_save_set_size_equals_n_plus_2() {
    for n in 1..=4 {
        assert_eq!(
            AbiContext { n_inputs: n }.ffi_save_set().len(),
            n + 2,
            "n={n}"
        );
    }
}

#[test]
fn abi_ffi_save_set_contents_n3() {
    let abi = AbiContext { n_inputs: 3 };
    assert_eq!(abi.ffi_save_set(), &["x0", "x1", "x2", "x3", "x4"]);
}
```

- [ ] **Step 2: Run tests — expect compile error**

```bash
cargo test -p profiles-arm64 abi_
```

Expected: methods don't exist yet.

- [ ] **Step 3: Implement methods**

```rust
// profiles/arm64/src/abi.rs (in impl AbiContext)
impl AbiContext {
    pub fn input_reg(&self, idx: usize) -> &'static str {
        INPUT_REGS[idx]
    }
    pub fn params_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs]
    }
    pub fn output_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs + 1]
    }
    pub fn ffi_save_set(&self) -> &[&'static str] {
        &INPUT_REGS[..self.n_inputs + 2]
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test -p profiles-arm64 abi_
```

Expected: 6 tests PASS.

### Task B.3: Implement materialise_ptr (will REPLACE existing function in ops/linear.rs)

The current `materialise_ptr` lives in `profiles/arm64/src/ops/linear.rs` as a free function. We move it onto `AbiContext` and update its match arms.

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs` (REMOVE the existing `pub fn materialise_ptr`)
- Modify: `profiles/arm64/src/abi.rs` (ADD `materialise_ptr` method on AbiContext)

- [ ] **Step 1: Inspect existing materialise_ptr**

```bash
grep -n "pub fn materialise_ptr\|fn materialise_ptr" profiles/arm64/src/ops/linear.rs
```

Note its signature and match arms.

- [ ] **Step 2: Add tests for AbiContext::materialise_ptr**

```rust
// profiles/arm64/src/tests.rs (append)
use crate::abi::AbiContext;
use crate::buffer::BufferLoc;

#[test]
fn abi_materialise_input_n1() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::InputReg(0), "x9", &mut s);
    assert!(s.contains("mov     x9, x0"), "got: {s}");
}

#[test]
fn abi_materialise_input_n3_idx2() {
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::InputReg(2), "x10", &mut s);
    assert!(s.contains("mov     x10, x2"), "got: {s}");
}

#[test]
fn abi_materialise_output_n2() {
    let abi = AbiContext { n_inputs: 2 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::OutputReg, "x11", &mut s);
    // N=2 → output is x3.
    assert!(s.contains("mov     x11, x3"), "got: {s}");
}

#[test]
fn abi_materialise_stack_offset() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::StackOffset(64), "x12", &mut s);
    assert!(s.contains("add     x12, sp, #64"), "got: {s}");
}
```

- [ ] **Step 3: Run — expect FAIL** (method doesn't exist yet)

```bash
cargo test -p profiles-arm64 abi_materialise
```

- [ ] **Step 4: Implement materialise_ptr on AbiContext**

```rust
// profiles/arm64/src/abi.rs (extend impl AbiContext)
impl AbiContext {
    /// Emit a `mov` (for InputReg/OutputReg) or `add sp, #off` (for
    /// StackOffset) instruction that places a buffer pointer into
    /// `dst_reg`.
    ///
    /// `BufferLoc::Alias(node_id)` is not handled here — caller must
    /// resolve the alias to a concrete BufferLoc before calling.
    pub fn materialise_ptr(&self, loc: BufferLoc, dst_reg: &str, asm: &mut String) {
        match loc {
            BufferLoc::InputReg(idx) => {
                asm.push_str(&format!("    mov     {dst_reg}, {}\n", self.input_reg(idx)));
            }
            BufferLoc::OutputReg => {
                asm.push_str(&format!("    mov     {dst_reg}, {}\n", self.output_reg()));
            }
            BufferLoc::StackOffset(off) => {
                asm.push_str(&format!("    add     {dst_reg}, sp, #{off}\n"));
            }
            BufferLoc::Alias(_) => {
                panic!("AbiContext::materialise_ptr: BufferLoc::Alias must be resolved before call site")
            }
        }
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p profiles-arm64 abi_materialise
```

Expected: 4 tests PASS. Remember at this point the OLD `materialise_ptr` in `ops/linear.rs` is still there, still matches `BufferLoc::InputReg` (no payload) — the old code path is the one used by every other op. The migration to using `abi.materialise_ptr` happens task-by-task per op in Tasks B.7–B.13.

### Task B.4: Implement emit_ffi_save (with xzr padding)

**Spec ref:** §6.1, §6.2.

**Files:**
- Modify: `profiles/arm64/src/abi.rs`
- Modify: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write tests**

```rust
// profiles/arm64/src/tests.rs (append)
#[test]
fn abi_emit_ffi_save_n1_three_regs_pads_xzr() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("stp     x0, x1, [sp, #-16]!"), "got:\n{s}");
    assert!(s.contains("stp     x2, xzr, [sp, #-16]!"), "got:\n{s}");
    let stp_count = s.matches("stp").count();
    assert_eq!(stp_count, 2, "expected 2 stp instructions, got {stp_count}");
}

#[test]
fn abi_emit_ffi_save_n2_four_regs_no_pad() {
    let abi = AbiContext { n_inputs: 2 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("stp     x0, x1, [sp, #-16]!"));
    assert!(s.contains("stp     x2, x3, [sp, #-16]!"));
    assert!(!s.contains("xzr"), "no xzr padding for even arity");
    assert_eq!(s.matches("stp").count(), 2);
}

#[test]
fn abi_emit_ffi_save_n3_five_regs_pads_xzr() {
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("stp     x0, x1, [sp, #-16]!"));
    assert!(s.contains("stp     x2, x3, [sp, #-16]!"));
    assert!(s.contains("stp     x4, xzr, [sp, #-16]!"));
    assert_eq!(s.matches("stp").count(), 3);
}

#[test]
fn abi_emit_ffi_save_n4_six_regs_no_pad() {
    let abi = AbiContext { n_inputs: 4 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("stp     x0, x1, [sp, #-16]!"));
    assert!(s.contains("stp     x2, x3, [sp, #-16]!"));
    assert!(s.contains("stp     x4, x5, [sp, #-16]!"));
    assert!(!s.contains("xzr"));
    assert_eq!(s.matches("stp").count(), 3);
}

#[test]
fn abi_emit_ffi_save_sp_delta_always_multiple_of_16() {
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut s = String::new();
        abi.emit_ffi_save(&mut s);
        // Each `stp ..., [sp, #-16]!` pre-decrements sp by 16.
        let stp_count = s.matches("stp").count();
        let sp_delta = stp_count * 16;
        assert!(sp_delta % 16 == 0, "n={n} sp_delta={sp_delta}");
        // Also sanity: sp_delta == ceil((n+2)/2) * 16.
        let expected = ((n + 2) + 1) / 2 * 16;
        assert_eq!(sp_delta, expected, "n={n}: ceil-div check");
    }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test -p profiles-arm64 abi_emit_ffi_save
```

- [ ] **Step 3: Implement emit_ffi_save**

```rust
// profiles/arm64/src/abi.rs (extend impl AbiContext)
impl AbiContext {
    /// Emit FFI-call save block — paired stp instructions; pads odd
    /// tail with xzr to maintain 16-byte SP alignment. Per spec §6.1.
    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let mut i = 0;
        while i < regs.len() {
            let a = regs[i];
            let b = if i + 1 < regs.len() { regs[i + 1] } else { "xzr" };
            asm.push_str(&format!("    stp     {a}, {b}, [sp, #-16]!\n"));
            i += 2;
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p profiles-arm64 abi_emit_ffi_save
```

Expected: 5 tests PASS.

### Task B.5: Implement emit_ffi_restore (LIFO reverse)

**Spec ref:** §6.3, §6.4.

**Files:**
- Modify: `profiles/arm64/src/abi.rs`
- Modify: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write tests**

```rust
// profiles/arm64/src/tests.rs (append)
#[test]
fn abi_emit_ffi_restore_n1_lifo() {
    // Save order: stp x0,x1; stp x2,xzr.
    // Restore order (LIFO): ldp x2,xzr; ldp x0,x1.
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.emit_ffi_restore(&mut s);
    let pos_a = s.find("ldp     x2, xzr, [sp], #16").expect("ldp x2,xzr");
    let pos_b = s.find("ldp     x0, x1, [sp], #16").expect("ldp x0,x1");
    assert!(pos_a < pos_b, "LIFO: top-of-stack pair restored first; got:\n{s}");
}

#[test]
fn abi_emit_ffi_restore_n3_lifo() {
    // Save: (x0,x1), (x2,x3), (x4,xzr).
    // Restore: (x4,xzr), (x2,x3), (x0,x1).
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.emit_ffi_restore(&mut s);
    let p1 = s.find("ldp     x4, xzr, [sp], #16").expect("xzr pair");
    let p2 = s.find("ldp     x2, x3, [sp], #16").expect("x2/x3 pair");
    let p3 = s.find("ldp     x0, x1, [sp], #16").expect("x0/x1 pair");
    assert!(p1 < p2, "LIFO order: xzr pair before x2/x3");
    assert!(p2 < p3, "LIFO order: x2/x3 before x0/x1");
}

#[test]
fn abi_save_then_restore_balances_sp() {
    // Number of stp == number of ldp.
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut save = String::new();
        let mut restore = String::new();
        abi.emit_ffi_save(&mut save);
        abi.emit_ffi_restore(&mut restore);
        assert_eq!(
            save.matches("stp").count(),
            restore.matches("ldp").count(),
            "save/restore mismatch at n={n}"
        );
    }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test -p profiles-arm64 abi_emit_ffi_restore abi_save_then_restore
```

- [ ] **Step 3: Implement emit_ffi_restore**

```rust
// profiles/arm64/src/abi.rs (extend impl AbiContext)
impl AbiContext {
    /// Emit FFI-call restore block — pairs are walked in strict reverse
    /// of emit_ffi_save (LIFO). xzr-padded slot round-trips harmlessly.
    pub fn emit_ffi_restore(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let n = regs.len();
        // Build pair list identical to emit_ffi_save, then iterate reversed.
        let mut pairs: Vec<(&str, &str)> = Vec::with_capacity(n.div_ceil(2));
        let mut i = 0;
        while i < n {
            let a = regs[i];
            let b = if i + 1 < n { regs[i + 1] } else { "xzr" };
            pairs.push((a, b));
            i += 2;
        }
        for (a, b) in pairs.iter().rev() {
            asm.push_str(&format!("    ldp     {a}, {b}, [sp], #16\n"));
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p profiles-arm64 abi_
```

Expected: ALL abi_ tests PASS (6 + 4 + 5 + 3 = 18). Add a final invariant check:

```rust
#[test]
fn abi_save_set_each_reg_appears_exactly_once_in_save_and_restore() {
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut save = String::new();
        let mut restore = String::new();
        abi.emit_ffi_save(&mut save);
        abi.emit_ffi_restore(&mut restore);
        for &reg in abi.ffi_save_set() {
            assert_eq!(save.matches(reg).count(), 1, "n={n} reg={reg} save");
            assert_eq!(restore.matches(reg).count(), 1, "n={n} reg={reg} restore");
        }
    }
}
```

Run, expect PASS.

### Task B.6: BufferLoc::InputReg(usize) migration

The biggest type-level change in Group B. Replaces `BufferLoc::InputReg` (unit variant) with `BufferLoc::InputReg(usize)` (carries input index). Compiler errors will identify every callsite needing update.

**Files:**
- Modify: `profiles/arm64/src/buffer.rs:14-20` (BufferLoc enum)
- Modify: `profiles/arm64/src/buffer.rs:32-80` (assign_buffers — update Input arm)

- [ ] **Step 1: Update BufferLoc enum**

```rust
// profiles/arm64/src/buffer.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    /// Input pointer at ABI register x_idx (0-indexed in model.inputs).
    InputReg(usize),
    OutputReg,
    StackOffset(usize),
    Alias(NodeId),
}
```

- [ ] **Step 2: Update assign_buffers — Input arm**

```rust
// profiles/arm64/src/buffer.rs (inside fn assign_buffers)
for (id, node) in model.nodes.iter().enumerate() {
    locs[id] = match &node.kind {
        NodeKind::Input { .. } => {
            // Find this node's index in model.inputs (declaration order).
            let idx = model.inputs.iter().position(|&i| i == id)
                .expect("Input node must appear in model.inputs");
            BufferLoc::InputReg(idx)
        }
        NodeKind::Op { op, operands, .. } => {
            // ... unchanged ...
        }
    };
}
```

- [ ] **Step 3: Build — expect compile errors**

```bash
cargo build -p profiles-arm64 2>&1 | head -50
```

Expected: errors at every site that matches `BufferLoc::InputReg` (no payload). Probably in:
- `profiles/arm64/src/codegen.rs` (walk_model dispatch)
- `profiles/arm64/src/ops/linear.rs` (old materialise_ptr — to be replaced anyway)
- `profiles/arm64/src/ops/matmul.rs` (matches BufferLoc)
- `profiles/arm64/src/ops/softmax.rs`
- etc.

- [ ] **Step 4: Quick interim fix (only here, undone in B.7+ as ops migrate to abi)**

For each broken match arm, change `BufferLoc::InputReg` → `BufferLoc::InputReg(_)`. This is a minimal fix to make the build pass; the proper migration to `abi.materialise_ptr(BufferLoc::InputReg(idx), ...)` happens per-op in B.7+.

Example (in `profiles/arm64/src/ops/linear.rs`, OLD materialise_ptr):

```rust
pub fn materialise_ptr(dst_reg: &str, loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg(_) => format!("    mov     {dst_reg}, x0\n"),  // _ binding
        BufferLoc::OutputReg => format!("    mov     {dst_reg}, x2\n"),
        BufferLoc::StackOffset(off) => format!("    add     {dst_reg}, sp, #{off}\n"),
        BufferLoc::Alias(_) => panic!("Alias unresolved"),
    }
}
```

(The `_` binding is the interim fix. The proper rewrite happens task-by-task.)

- [ ] **Step 5: Build — expect clean**

```bash
cargo build -p profiles-arm64
```

- [ ] **Step 6: Run all profiles-arm64 tests**

```bash
cargo test -p profiles-arm64
```

Expected: ALL existing tests PASS (since the interim `_` binding behaves like the old unit variant for N=1). Plus all new abi_ tests pass. Plus N=1 regression: PASS (asm unchanged).

### Task B.7: Thread &abi through walk_model + arity check

**Spec ref:** §5.3.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Update walk_model signature + entry**

Find the current `walk_model` (or equivalent — names may differ slightly; grep `pub fn walk_model\|fn walk_model`):

```bash
grep -n "fn walk_model\|fn walk_uir" profiles/arm64/src/codegen.rs
```

- [ ] **Step 2: Add AbiContext construction + arity check at entry**

```rust
// profiles/arm64/src/codegen.rs
// Inside walk_model (or whatever the per-model entry function is called).

use crate::abi::{AbiContext, INPUT_REGS};

pub(crate) fn walk_model(model: &UirModel, /* ... existing args ... */) -> Result<String, LowerError> {
    // Spec §5.3: arity check at function entry.
    let n = model.inputs.len();
    if n + 2 > INPUT_REGS.len() {
        return Err(LowerError::TooManyInputs {
            n,
            max: INPUT_REGS.len() - 2,  // = 4
            span: model.source_span,
        });
    }
    let abi = AbiContext { n_inputs: n };

    // ... existing prologue + per-op emission, with `&abi` threaded in ...
    Ok(asm)
}
```

- [ ] **Step 3: Update each op-dispatch arm to pass &abi**

```rust
// profiles/arm64/src/codegen.rs (inside the per-node match)
match &node.kind {
    NodeKind::Input { .. } => {
        // No emission — input pointer is already in its ABI register.
    }
    NodeKind::Op { op, operands, attrs, span } => {
        match op {
            StdOp::Linear   => emit_linear(&abi, /* args */)?,
            StdOp::Matmul   => emit_matmul(&abi, /* args */)?,
            StdOp::Softmax  => emit_softmax(&abi, /* args */),
            StdOp::Relu     => emit_relu(&abi, /* args */),
            StdOp::Dropout  => emit_dropout(&abi, /* args */),
            StdOp::MulScalar => emit_mulscalar(&abi, /* args */),
        }
    }
}
```

(Op-emitter signatures are updated in B.8–B.13; this step is one wire-up commit.)

- [ ] **Step 4: Build — emitter signatures don't yet accept &abi**

```bash
cargo build -p profiles-arm64 2>&1 | head -30
```

Expected: compile errors at each `emit_*` callsite — argument count mismatch. Resolved as each emitter is migrated in B.8–B.13.

To keep build green during the transition, comment out (or rewrite) emit_* callsites step-by-step. Easier alternative: do B.8–B.13 first while keeping walk_model unchanged (still uses old materialise_ptr from ops/linear.rs), then activate the &abi threading in walk_model as the LAST step.

**Recommendation:** treat B.7 Step 3 as "prepare the dispatch shape but defer activation" — apply only Steps 1 and 2 here, then come back after B.8–B.13 to thread `&abi`.

- [ ] **Step 5: Run tests after Step 1+2 only**

```bash
cargo test -p profiles-arm64
```

Expected: ALL pass (only walk_model entry changed; `n` is computed but not yet used by ops). N=1 regression: PASS (TooManyInputs only triggers for N>4; existing fixtures all N=1).

### Task B.8: Migrate emit_linear to use &abi

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs`

- [ ] **Step 1: Update emit_linear signature**

```rust
// profiles/arm64/src/ops/linear.rs
use crate::abi::AbiContext;

pub fn emit_linear(
    abi: &AbiContext,
    /* ... existing args ... */
) -> Result<String, LowerError> {
    // ... existing body, with all hardcoded x0/x1/x2 references rerouted ...
}
```

- [ ] **Step 2: Replace hardcoded register references**

Search for hardcoded `"x0"`, `"x1"`, `"x2"` in `linear.rs`:

```bash
grep -n '"x0"\|"x1"\|"x2"' profiles/arm64/src/ops/linear.rs
```

For each occurrence:
- `"x0"` (when used as input register) → `abi.input_reg(0)` (linear has single tensor input — index 0). For multi-input models that pipe input N to a linear op, the BufferLoc resolution at the call site already returns InputReg(N), so the linear op operates correctly. The "0" here refers to "first arm of the linear-input-resolver", not necessarily input 0 of the model.

  **Correction:** `linear.rs` doesn't reference `x0` directly — it should call `abi.materialise_ptr(input_loc, dst_reg, asm)` where `input_loc` is the `BufferLoc` of its (sole) tensor operand, passed in by the caller. Hardcoded `"x0"` should never appear in op-emitters under the new design.

- `"x1"` (when used as params register) → `abi.params_reg()`.
- `"x2"` (when used as output register) → `abi.output_reg()`.

- [ ] **Step 3: REMOVE old free-function `materialise_ptr` from linear.rs**

It's been moved to `AbiContext::materialise_ptr` (Task B.3). Anywhere `linear.rs` previously called the free function, it now calls `abi.materialise_ptr(loc, dst_reg, asm)`.

- [ ] **Step 4: Update other ops importing materialise_ptr**

```bash
grep -rn "use crate::ops::linear::materialise_ptr" profiles/arm64/
```

For each: replace import with `&AbiContext` parameter, call `abi.materialise_ptr` instead of free function.

- [ ] **Step 5: Build**

```bash
cargo build -p profiles-arm64 2>&1 | head -30
```

Expected: many remaining errors at other ops (matmul, softmax, etc.) — those are migrated in B.9–B.13.

For now, ensure linear.rs builds (may need to leave other ops using a temp shim — e.g., re-export `pub fn materialise_ptr_shim` from linear.rs that takes (loc, dst_reg) and constructs an AbiContext { n_inputs: 1 } locally — only as a transition aid).

**Cleaner recommendation:** if compile cascades get unwieldy, do Tasks B.8–B.13 in a single coordinated edit: change all op-emitter signatures simultaneously, fix walk_model dispatch, run tests once at the end. The TDD-step decomposition here is for clarity; the actual implementation may batch.

- [ ] **Step 6: Run tests**

```bash
cargo test -p profiles-arm64
```

Expected: all PASS. N=1 regression: PASS.

### Task B.9: Migrate emit_softmax to abi.emit_ffi_save / restore

**Spec ref:** §5.4.

**Files:**
- Modify: `profiles/arm64/src/ops/softmax.rs`

- [ ] **Step 1: Find the manual stp/ldp blocks**

```bash
grep -n "stp\|ldp" profiles/arm64/src/ops/softmax.rs
```

Expected: lines 63-64 (per the file we read earlier):
```
s.push_str("    stp     x0, x1, [sp, #-16]!\n");
s.push_str("    stp     x2, xzr, [sp, #-16]!\n");
```

And matching `ldp` near function exit.

- [ ] **Step 2: Replace with abi.emit_ffi_save / restore**

```rust
// profiles/arm64/src/ops/softmax.rs (replace the stp block)
abi.emit_ffi_save(&mut s);
```

(Find the matching ldp block near function exit and replace with `abi.emit_ffi_restore(&mut s);`.)

- [ ] **Step 3: Update emit_softmax signature**

```rust
pub fn emit_softmax(
    abi: &AbiContext,
    b: u64,
    k: u64,
    model_idx: usize,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    sym_prefix: &str,
) -> String {
```

- [ ] **Step 4: Update callsite in codegen.rs**

```rust
// profiles/arm64/src/codegen.rs (in the StdOp::Softmax dispatch arm)
emit_softmax(&abi, b, k, model_idx, softmax_idx, src_loc, dst_loc, sym_prefix)
```

- [ ] **Step 5: Replace any remaining `materialise_ptr` calls with `abi.materialise_ptr`**

In softmax.rs:

```rust
// OLD: s.push_str(&materialise_ptr("x22", src_loc));
// NEW:
abi.materialise_ptr(src_loc, "x22", &mut s);
abi.materialise_ptr(dst_loc, "x23", &mut s);
```

- [ ] **Step 6: Build + test**

```bash
cargo build -p profiles-arm64
cargo test -p profiles-arm64
```

Expected: PASS. Crucially, N=1 regression must still pass — for arity 1, `abi.emit_ffi_save` produces:

```
stp     x0, x1, [sp, #-16]!
stp     x2, xzr, [sp, #-16]!
```

Which is **bit-identical** to the old hand-written block. Goldens stay valid.

### Task B.10: Migrate emit_mulscalar / emit_relu / emit_dropout

**Files:**
- Modify: `profiles/arm64/src/ops/mulscalar.rs`
- Modify: `profiles/arm64/src/ops/relu.rs`
- Modify: `profiles/arm64/src/ops/dropout.rs`

For each file:

- [ ] **Step 1: Update signature to accept `&AbiContext`**

```rust
pub fn emit_mulscalar(
    abi: &AbiContext,
    /* existing args */
) -> String { ... }
```

- [ ] **Step 2: Replace `materialise_ptr` calls with `abi.materialise_ptr`**

```rust
// OLD: s.push_str(&materialise_ptr("xN", loc));
// NEW: abi.materialise_ptr(loc, "xN", &mut s);
```

- [ ] **Step 3: Replace any hardcoded `"x0"`/`"x1"`/`"x2"` with abi accessors**

(These ops should not have any — they should already go through `materialise_ptr`. Verify with grep.)

- [ ] **Step 4: Update callsites in codegen.rs**

- [ ] **Step 5: Build + test (per file or batched)**

```bash
cargo build -p profiles-arm64 && cargo test -p profiles-arm64
```

Expected: all PASS. N=1 regression: PASS.

### Task B.11: emit_matmul rework — slice ptrs to non-ABI scratch (THE high-risk task)

**Spec ref:** §9.1.

**Files:**
- Modify: `profiles/arm64/src/ops/matmul.rs` heavily

The existing emit_matmul (per the file read) materialises base pointers into x11 (A), x13 (B), x12 (DST) BEFORE the outer loop. The PROBLEM is the per-iter slice pointers, which currently use x1 (A_slice), x2 (B_slice or DST_slice), x4 (DST_slice). These collide with ABI registers and require the `stp x1, x2, [sp, #-16]!` spill block.

The rework: move per-iter slice pointers to non-ABI scratch registers. Concrete reassignment:

| Role | Old reg | New reg |
|------|---------|---------|
| A base ptr | x11 | **x9** |
| B base ptr | x13 | **x10** |
| DST base ptr | x12 | x12 (unchanged) |
| A_slice ptr (per-iter) | x1 | **x14** |
| B_slice ptr (per-iter) | x2 | **x15** |
| DST_slice ptr (per-iter) | x4 | **x4** (unchanged — caller-saved scratch, not ABI for N≤4) |
| Outer-loop counter | x17 | x17 (unchanged) |

Wait — x4 IS in the ABI argument set for N≥3 (output reg = x_{N+1} = x4 for N=3). Need to verify this.

For N=3: INPUT_REGS[..5] = ["x0", "x1", "x2", "x3", "x4"]. So x4 IS in ffi_save_set for N=3. Using it as scratch in matmul body would clobber it; downstream ops using `abi.output_reg()` (= x4 for N=3) would break.

**Corrected reassignment:**

| Role | New reg | Why |
|------|---------|-----|
| A base ptr | x9 | non-ABI for any N ≤ 4 |
| B base ptr | x10 | non-ABI for any N ≤ 4 |
| DST base ptr | x11 | non-ABI for any N ≤ 4 |
| A_slice ptr | x12 | non-ABI |
| B_slice ptr | x13 | non-ABI |
| DST_slice ptr | x14 | non-ABI |
| Outer-loop counter | x15 | non-ABI; or x17 if existing |
| Slice index temp | x16 | non-ABI |

x9–x16 are all caller-saved scratch (per AAPCS) and are NEVER in INPUT_REGS regardless of arity (max N=4 → INPUT_REGS goes up to x5).

- [ ] **Step 1: Update emit_matmul signature**

```rust
pub fn emit_matmul(
    abi: &AbiContext,
    leading_count: u64,
    m: u64,
    k: u64,
    n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    node_span: Span,
) -> Result<String, LowerError> {
```

- [ ] **Step 2: Replace materialise_ptr free-fn calls with abi.materialise_ptr**

```rust
// OLD:
// s.push_str(&materialise_ptr("x11", a_loc));
// s.push_str(&materialise_ptr("x13", b_loc));
// s.push_str(&materialise_ptr("x12", dst_loc));
// NEW (with new register assignment):
abi.materialise_ptr(a_loc, "x9", &mut s);   // A base
abi.materialise_ptr(b_loc, "x10", &mut s);  // B base
abi.materialise_ptr(dst_loc, "x11", &mut s); // DST base
```

- [ ] **Step 3: REMOVE the stp/ldp spill block**

Find lines like:
```rust
s.push_str("    stp     x1, x2, [sp, #-16]!\n");
```
DELETE these entirely. Also delete the matching `ldp` near function exit.

- [ ] **Step 4: Update all asm strings inside the outer/inner loop**

Every reference to `x1`, `x2`, `x4`, `x11`, `x13`, `x12` in the asm strings must be remapped:
- Old `x11` (A base) → `x9`
- Old `x13` (B base) → `x10`
- Old `x12` (DST base) → `x11`
- Old `x1` (A_slice) → `x12`
- Old `x2` (B_slice or DST_slice — read code carefully) → `x13`
- Old `x4` (DST_slice or B_slice) → `x14`
- Old `x17` (outer counter) → `x15` (or keep x17 if not in scratch list)

This is a careful textual remap. Use:

```bash
grep -n '"x' profiles/arm64/src/ops/matmul.rs | head -50
```

To enumerate all asm-string register references; map each.

- [ ] **Step 5: Add unit test asserting NO stp/pushq in emit_matmul output**

```rust
// profiles/arm64/src/tests.rs (append)
#[test]
fn emit_matmul_body_contains_zero_stp() {
    // matmul does not call FFI; only AbiContext::emit_ffi_save emits stack
    // manipulation. After §9.1 rework, emit_matmul body must contain zero
    // `stp` instructions. The function-level prologue (callee-saved regs)
    // emits its own stp — but that's outside emit_matmul.
    use compiler::ast::Span;
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs: 2 };
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
    ).expect("emit_matmul should succeed");
    let stp_count = result.matches("stp").count();
    assert_eq!(stp_count, 0,
        "emit_matmul body must contain zero stp instructions per §9.1; got {stp_count}\n{result}");
}
```

- [ ] **Step 6: Run the matmul-specific test + N=1 regression**

```bash
cargo test -p profiles-arm64 emit_matmul_body_contains_zero_stp
cargo test -p profiles-arm64 n1_regression
```

**The N=1 regression test is the load-bearing check here.** If golden files diverge, the rework introduced a behaviour change for arity 1.

For N=1, the new register layout produces DIFFERENT assembly than the old (x9/x10/x11 instead of x11/x13/x12; x12/x13/x14 instead of x1/x2/x4). The old golden files contain the OLD register names.

**This means:** the N=1 regression test will FAIL after Task B.11 — by design — because the bytes of the asm change. The CORRECT behaviour is verified differently:
1. Numerical FFI integration tests for self_attention.nfl (existing M10 test, exists in `profiles/arm64/tests/integration.rs`) must still pass — confirms the regenerated asm computes the same thing.
2. Goldens must be REGENERATED after the rework to capture the new register layout.

**Plan resolution:** after B.11 completes successfully (numerical tests pass), regenerate goldens (Task B.12). This means the strict bit-exactness invariant (spec §10.2 gate #4) becomes "bit-exact under M12 codegen, established at the start of M12 and held thereafter" rather than "bit-exact against M11 baseline forever."

This is a deliberate weakening of gate #4, justified by the fact that the matmul rework is structural improvement (removes spill block), not a regression. The goldens serve their purpose: regression detection AGAINST POST-M12 BASELINE for any future change.

- [ ] **Step 7: Run integration tests (numerical)**

```bash
cargo test -p profiles-arm64 --test integration
```

Expected: PASS. Especially `self_attention_match_numerically` (the sole M10 multi-step test that exercises matmul → softmax → matmul) — this confirms matmul still computes correctly.

- [ ] **Step 8: Build + workspace test**

```bash
cargo build -p profiles-arm64
cargo test --workspace 2>&1 | tail -20
```

Expected: numerical tests pass; **N=1 regression FAILS** (asm changed bytes-wise). This is acceptable for now; B.12 regenerates goldens.

### Task B.12: Regenerate N=1 goldens after matmul rework

**Files:**
- Modify: `profiles/arm64/tests/golden/{classifier,self_attention,...}.s` (any fixture that uses matmul)
- Modify: `profiles/arm64/tests/golden/{classifier,...}.s` (any fixture that uses softmax — for consistency with B.9 changes; should be byte-identical, but verify)

- [ ] **Step 1: Identify which goldens need regeneration**

```bash
cargo test -p profiles-arm64 n1_regression_all_fixtures_bit_exact 2>&1 | grep "fixture.*divergence" | head -20
```

Note which fixtures fail.

- [ ] **Step 2: Regenerate failing goldens**

```bash
for f in $(cargo test -p profiles-arm64 n1_regression 2>&1 | grep "fixture" | sed -E 's/.*fixture ([a-z_]+):.*/\1/' | sort -u); do
  cargo run -p nflc -- compile "tests/fixtures/${f}.nfl" --profile arm64 --emit asm > "profiles/arm64/tests/golden/${f}.s"
done
```

(Or use the dump_asm tool from A.1 fallback if --emit asm doesn't exist.)

- [ ] **Step 3: Sanity-check regenerated goldens by diffing against old**

```bash
git diff profiles/arm64/tests/golden/
```

Expected diff: only register-name changes (x1 → x12, x11 → x9, etc.) plus removal of `stp x1, x2, [sp, #-16]!` and matching `ldp` lines. NO logical changes (no different ops, no different control flow). If you see ADDED instructions or different label names, the rework introduced an unintended divergence — investigate before continuing.

- [ ] **Step 4: Run regression test**

```bash
cargo test -p profiles-arm64 n1_regression_all_fixtures_bit_exact
```

Expected: PASS now. Goldens reflect post-M12 layout.

### Task B.13: Activate &abi threading in walk_model (deferred from B.7)

If B.7 deferred Step 3 (`&abi` threading through op dispatch), now is the time to activate.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs` (uncomment / fill in the per-op dispatch)

- [ ] **Step 1: Wire up each op-dispatch arm**

```rust
match op {
    StdOp::Linear   => { let asm = emit_linear(&abi, ...)?; output.push_str(&asm); }
    StdOp::Matmul   => { let asm = emit_matmul(&abi, ...)?; output.push_str(&asm); }
    StdOp::Softmax  => { let asm = emit_softmax(&abi, ...); output.push_str(&asm); }
    // ... etc.
}
```

- [ ] **Step 2: Build + test**

```bash
cargo build -p profiles-arm64
cargo test -p profiles-arm64
cargo test --workspace
```

Expected: ALL tests PASS, including N=1 regression (now against regenerated post-M12 goldens).

### Task B.14: Group B pre-commit + commit

- [ ] **Step 1: Pre-commit ritual**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: 347 → ~365 tests (added: ~14 abi unit tests + 1 emit_matmul body test + ~3 misc). All pass.

- [ ] **Step 2: Commit**

```bash
git add profiles/arm64/src/abi.rs \
        profiles/arm64/src/lib.rs \
        profiles/arm64/src/buffer.rs \
        profiles/arm64/src/codegen.rs \
        profiles/arm64/src/ops/ \
        profiles/arm64/src/tests.rs \
        profiles/arm64/tests/golden/

git commit -m "$(cat <<'EOF'
feat(m12): arm64 multi-input codegen via AbiContext + emit_matmul rework

New profiles/arm64/src/abi.rs introduces AbiContext per spec §5.2:
- input_reg(idx) / params_reg() / output_reg() return ABI register
  names indexed by arity.
- ffi_save_set() = INPUT_REGS[..N+2] = conservative caller-saved set
  spilled around bl _expf.
- emit_ffi_save / emit_ffi_restore emit alignment-correct stp/ldp
  blocks (xzr-padded for odd cardinality, strict LIFO restore).
- materialise_ptr(loc, dst, asm) emits mov / add-sp instructions.

BufferLoc::InputReg gains a usize index payload (= position in
model.inputs). assign_buffers looks up the index via
model.inputs.iter().position().

walk_model constructs AbiContext { n_inputs: model.inputs.len() } at
function entry and threads &abi through every op-emitter. N>4 returns
LowerError::TooManyInputs at this entry.

emit_matmul rework per spec §9.1: per-iter A_slice/B_slice/DST_slice
pointers move from ABI registers (x1/x2/x4) to non-ABI scratch
(x12/x13/x14). Operand base pointers move from x11/x13/x12 to
x9/x10/x11. Old `stp x1, x2, [sp, #-16]!` outer-loop spill block is
REMOVED. emit_matmul body now contains zero stp instructions
(emit_matmul does not call FFI; only emit_ffi_save emits stack
manipulation). Verified by emit_matmul_body_contains_zero_stp test.

emit_softmax replaces hand-written stp/ldp block around bl _expf
with abi.emit_ffi_save / emit_ffi_restore — bit-identical for N=1,
arity-aware for N≥2.

N=1 goldens regenerated after the matmul rework (register names
change x1→x12 etc., no logical divergence). Numerical FFI tests
including self_attention_match_numerically still pass — confirms
behavioral equivalence.

Test count: 347 → ~365 (+18: AbiContext alignment/LIFO/ffi_save_set
tests + matmul-no-stp test + misc).

Spec ref: §5, §6, §9.1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

```bash
git log --oneline -2
git status
cargo test --workspace 2>&1 | tail -5
```

---

## Group C — x86_64 codegen (commit 3)

Mirror of Group B for x86_64 SysV. Same tasks, same structure, swap arm64-specific bits for x86_64 equivalents.

**LoC delta:** ~250 production + ~80 tests = ~330.
**Test count delta:** +14 to +18.

### Task C.1: Generate x86_64 baseline goldens

Already done in Task A.1 step 2. If goldens are already in `profiles/x86_64/tests/golden/` and committed in Group A, this task is a no-op.

### Task C.2: Create profiles/x86_64/src/abi.rs

Mirror of Task B.1.

```rust
// profiles/x86_64/src/abi.rs
// SPDX-License-Identifier: Apache-2.0

//! SysV AMD64 argument-register abstraction for multi-input model ABI.
//! See spec §5.2.

use crate::buffer::BufferLoc;

/// Argument-register table. SysV AMD64 first 6 GP arg regs (rdi/rsi/
/// rdx/rcx/r8/r9). M12 caps inputs at N=4, so N+2 ≤ 6 — register-only
/// argument passing, no stack-spill required.
pub(crate) const INPUT_REGS: &[&str] = &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];

pub(crate) struct AbiContext {
    pub n_inputs: usize,
}
```

Add `pub mod abi;` to `profiles/x86_64/src/lib.rs`.

### Task C.3: Implement input_reg / params_reg / output_reg / ffi_save_set / materialise_ptr

Mirror of B.2 + B.3. Tests are mirror with x86_64 register names. Body:

```rust
impl AbiContext {
    pub fn input_reg(&self, idx: usize) -> &'static str { INPUT_REGS[idx] }
    pub fn params_reg(&self) -> &'static str { INPUT_REGS[self.n_inputs] }
    pub fn output_reg(&self) -> &'static str { INPUT_REGS[self.n_inputs + 1] }
    pub fn ffi_save_set(&self) -> &[&'static str] { &INPUT_REGS[..self.n_inputs + 2] }

    pub fn materialise_ptr(&self, loc: BufferLoc, dst_reg: &str, asm: &mut String) {
        match loc {
            BufferLoc::InputReg(idx) => {
                asm.push_str(&format!("    movq    {}, {dst_reg}\n", self.input_reg(idx)));
            }
            BufferLoc::OutputReg => {
                asm.push_str(&format!("    movq    {}, {dst_reg}\n", self.output_reg()));
            }
            BufferLoc::StackOffset(off) => {
                // GAS AT&T: lea  off(%rsp), dst
                asm.push_str(&format!("    leaq    {off}(%rsp), {dst_reg}\n"));
            }
            BufferLoc::Alias(_) => {
                panic!("AbiContext::materialise_ptr: BufferLoc::Alias must be resolved before call site")
            }
        }
    }
}
```

Note SysV asm conventions: `movq src, dst` (AT&T) for register copies; `leaq off(%rsp), dst` for stack-offset addressing.

### Task C.4: Implement emit_ffi_save / emit_ffi_restore

Mirror of B.4 + B.5 with `pushq`/`popq` and `%rax` padding.

```rust
impl AbiContext {
    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        for r in regs {
            asm.push_str(&format!("    pushq   {r}\n"));
        }
        // Pad to even count for 16-byte SP alignment at the call insn.
        if regs.len() % 2 != 0 {
            asm.push_str("    pushq   %rax              # 16-byte alignment padding\n");
        }
    }

    pub fn emit_ffi_restore(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        if regs.len() % 2 != 0 {
            asm.push_str("    popq    %rax              # discard alignment padding\n");
        }
        for r in regs.iter().rev() {
            asm.push_str(&format!("    popq    {r}\n"));
        }
    }
}
```

Mirror tests:

```rust
#[test]
fn abi_emit_ffi_save_n3_pads_rax() {
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    let push_count = s.matches("pushq").count();
    assert_eq!(push_count, 6, "5 input/params/output + 1 padding");
    assert!(s.contains("pushq   %rax"));
    // SP delta = 6 * 8 = 48 bytes, multiple of 16 ✓
}

// ... other N values, LIFO check, etc. ...
```

### Task C.5: BufferLoc::InputReg(usize) migration in x86_64

Mirror of B.6.

### Task C.6: Thread &abi through walk_model + arity check

Mirror of B.7.

### Task C.7: Migrate emit_softmax to abi.emit_ffi_save / restore

Mirror of B.9. Find the existing `pushq %rdi` / `pushq %rsi` / `pushq %rdx` block plus padding `pushq %rax`. Replace with `abi.emit_ffi_save(&mut s)`. Bit-identical for N=1.

### Task C.8: Migrate emit_linear / emit_mulscalar / emit_relu / emit_dropout

Mirror of B.8 + B.10.

### Task C.9: emit_matmul x86_64 rework

Mirror of B.11. The x86_64 emit_matmul currently spills `%rdi/%rsi/%rdx` via `movq` to `%xmm6/%xmm7/%xmm8` (callee-saved float regs). Rework: per-iter slice pointers move to `%r10/%r11/%r12/%r13` (non-ABI scratch GP regs); base pointers move to `%r10`/`%r11` (operand base) and another reg for DST. Old `movq %rdi, %xmm6` etc. block REMOVED.

Concrete reassignment for x86_64:

| Role | New reg |
|------|---------|
| A base ptr | %r10 |
| B base ptr | %r11 |
| DST base ptr | %r12 |
| A_slice ptr | %r13 |
| B_slice ptr | %r14 |
| DST_slice ptr | %r15 |

%r10–%r15 are all SysV caller-saved scratch GP regs, never in INPUT_REGS regardless of arity.

Add unit test asserting emit_matmul body contains zero `pushq` AND zero `movq → %xmm` instructions.

### Task C.10: Regenerate x86_64 goldens

Mirror of B.12.

### Task C.11: Activate &abi threading in walk_model

Mirror of B.13.

### Task C.12: Group C pre-commit + commit

Mirror of B.14. Commit message:

```
feat(m12): x86_64 multi-input codegen via AbiContext + emit_matmul rework

[Same shape as Group B commit but for x86_64.]
```

---

## Group D — Multi-input fixtures + integration tests (commit 4)

After Groups B + C, the codegen supports multi-input. This group adds the fixtures + per-profile FFI integration tests that prove it numerically.

**LoC delta:** ~200 (3 fixtures + 6 integration tests across 2 profiles).
**Test count delta:** +6 (3 multi-input × 2 profiles) + 2 negative = +8.

### Task D.1: Create tests/fixtures/two_input_matmul.nfl

**Spec ref:** §7.1.

**Files:**
- Create: `tests/fixtures/two_input_matmul.nfl`

- [ ] **Step 1: Write fixture**

```
# Two-input matmul. Sanity for N=2 register layout (a→x0/%rdi,
# b→x1/%rsi, params at x2/%rdx, output at x3/%rcx). Does NOT
# exercise post-FFI register survival — that's the multi_input_
# attention fixture's job.

model TwoInputMatmul [m=4, k=8, n=4]:
    a: Tensor[m, k]
    b: Tensor[k, n]

    out: Tensor[m, n] = a -> matmul[b]
```

- [ ] **Step 2: Manually compile to verify it builds**

```bash
cargo run -p nflc -- compile tests/fixtures/two_input_matmul.nfl --profile arm64 --emit asm > /tmp/two_input.s
head -30 /tmp/two_input.s
```

Expected: arm64 asm with multi-input prologue (a in x0, b in x1, params in x2, output in x3). No errors.

```bash
cargo run -p nflc -- compile tests/fixtures/two_input_matmul.nfl --profile x86_64 --emit asm > /tmp/two_input_x.s
head -30 /tmp/two_input_x.s
```

Expected: x86_64 asm with a in %rdi, b in %rsi, params in %rdx, output in %rcx.

### Task D.2: Create tests/fixtures/multi_input_attention.nfl

**Spec ref:** §7.2 — THE acceptance fixture.

**Files:**
- Create: `tests/fixtures/multi_input_attention.nfl`

- [ ] **Step 1: Write fixture (verbatim from spec §7.2)**

```
# Multi-input self-attention. THE acceptance fixture for M12:
# v is consumed AFTER softmax via `attn -> matmul[v]`, which means
# the v-pointer register (x2 on arm64, %rdx on x86_64) MUST survive
# `bl _expf` / `call expf@PLT`. This is the exact code path that
# AbiContext::ffi_save_set's correctness depends on — no other
# fixture exercises post-FFI-call register survival for a non-x0
# input.

model SelfAttention [batch=2, heads=4, seq=16, head_dim=16]:
    q: Tensor[batch, heads, seq, head_dim]
    k: Tensor[batch, heads, head_dim, seq]
    v: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = q -> matmul[k]
    scaled: Tensor[batch, heads, seq, seq] = scores -> mul_scalar[0.25]
    attn:   Tensor[batch, heads, seq, seq] = scaled -> softmax
    out:    Tensor[batch, heads, seq, head_dim] = attn -> matmul[v]
```

- [ ] **Step 2: Manually compile**

```bash
cargo run -p nflc -- compile tests/fixtures/multi_input_attention.nfl --profile arm64 --emit asm > /tmp/mha.s
grep -c "stp" /tmp/mha.s
grep "bl " /tmp/mha.s
```

Expected: stp count > 0 (function prologue + emit_ffi_save in softmax). `bl _expf` present.

### Task D.3: Create tests/fixtures/negative/too_many_inputs.nfl

**Spec ref:** §7.3.

**Files:**
- Create: `tests/fixtures/negative/too_many_inputs.nfl`

- [ ] **Step 1: Write fixture (5 inputs)**

```
# 5 inputs, exceeds M12 N=4 hard-cap. Parser + IR-build accept;
# profile lower() must reject with LowerError::TooManyInputs.

model TooManyInputs [d=8]:
    a: Tensor[d]
    b: Tensor[d]
    c: Tensor[d]
    d_in: Tensor[d]
    e: Tensor[d]

    out: Tensor[d] = a -> linear[features=d]
```

- [ ] **Step 2: Verify parse + IR-build succeed**

```bash
cargo run -p nflc -- parse tests/fixtures/negative/too_many_inputs.nfl --uir
```

Expected: parses successfully, prints UIR with 5 input nodes.

- [ ] **Step 3: Verify profile.lower() rejects**

```bash
cargo run -p nflc -- compile tests/fixtures/negative/too_many_inputs.nfl --profile arm64 2>&1
```

Expected: `Error: model declares 5 inputs but this profile supports a maximum of 4` (or similar — the `LowerError::TooManyInputs` Display output).

### Task D.4: arm64 FFI integration test for two_input_matmul

**Files:**
- Modify: `profiles/arm64/tests/integration.rs` (append new test)
- May also need: `profiles/arm64/tests/common/mod.rs` (helper extensions, if not already supporting multi-input)

- [ ] **Step 1: Inspect the existing integration test pattern**

```bash
grep -n "fn.*_match_numerically" profiles/arm64/tests/integration.rs | head
```

Pick one (e.g. `self_attention_match_numerically`) and study its structure: it parses a fixture, lowers via profile, compiles via `cc`, dlopens, calls forward, compares output bit-exact with a Rust reference impl.

- [ ] **Step 2: Write two_input_matmul integration test**

```rust
// profiles/arm64/tests/integration.rs (append)
#[test]
fn two_input_matmul_match_numerically() {
    use compiler::ir::build;
    use libloading::{Library, Symbol};
    use std::fs::read_to_string;

    // 1. Parse + IR build.
    let src = read_to_string("../../tests/fixtures/two_input_matmul.nfl").unwrap();
    let nfl = compiler::parse(&src).unwrap();
    let uir = build(&nfl).unwrap();

    // 2. Lower to arm64 asm.
    let asm = profiles_arm64::Arm64.lower(&uir).unwrap();
    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 2, "two_input_matmul has arity 2");

    // 3. Compile via cc to a dylib in a tempdir (use existing helper).
    let dylib_path = common::compile_to_dylib(&asm.source, "two_input_matmul");

    // 4. dlopen + dlsym.
    let lib = unsafe { Library::new(&dylib_path).unwrap() };
    type ForwardFn = unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *mut f32);
    //  a            b            params        out
    let forward: Symbol<ForwardFn> = unsafe {
        lib.get(format!("nfl_forward_TwoInputMatmul").as_bytes()).unwrap()
    };

    // 5. Allocate buffers per FnSig.
    let m = 4; let k = 8; let n = 4;
    let mut a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.1).collect();
    let mut b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.07).collect();
    let params: Vec<f32> = vec![]; // matmul has no params
    let mut out: Vec<f32> = vec![0.0; m * n];

    // 6. Call forward.
    unsafe {
        forward(a.as_ptr(), b.as_ptr(), params.as_ptr(), out.as_mut_ptr());
    }

    // 7. Compute Rust reference: a @ b.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i*k + kk] * b[kk*n + j];
            }
            expected[i*n + j] = sum;
        }
    }

    // 8. Bit-exact compare.
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#x}), want {want} ({:#x})",
            got.to_bits(), want.to_bits()
        );
    }

    // 9. Drop guard: explicit drop of forward, then lib.
    drop(forward);
    drop(lib);
}
```

(Adapt `common::compile_to_dylib` helper signature to match what's currently in `profiles/arm64/tests/common/mod.rs`.)

- [ ] **Step 3: Run test**

```bash
cargo test -p profiles-arm64 --test integration two_input_matmul_match_numerically
```

Expected: PASS. If failure: register layout bug in emit_matmul or AbiContext.

### Task D.5: arm64 FFI integration test for multi_input_attention (THE acceptance test)

**Spec ref:** §7.2, §10.3 gate #7.

**Files:**
- Modify: `profiles/arm64/tests/integration.rs` (append)

- [ ] **Step 1: Write the test**

```rust
#[test]
fn multi_input_attention_match_numerically() {
    use compiler::ir::build;
    use libloading::{Library, Symbol};
    use std::fs::read_to_string;

    let src = read_to_string("../../tests/fixtures/multi_input_attention.nfl").unwrap();
    let nfl = compiler::parse(&src).unwrap();
    let uir = build(&nfl).unwrap();
    let asm = profiles_arm64::Arm64.lower(&uir).unwrap();

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 3, "multi_input_attention has arity 3");

    let dylib_path = common::compile_to_dylib(&asm.source, "multi_input_attention");
    let lib = unsafe { Library::new(&dylib_path).unwrap() };
    type ForwardFn = unsafe extern "C" fn(
        *const f32, *const f32, *const f32,    // q, k, v
        *const f32, *mut f32);                 // params, out
    let forward: Symbol<ForwardFn> = unsafe {
        lib.get(b"nfl_forward_SelfAttention").unwrap()
    };

    // Shape: batch=2, heads=4, seq=16, head_dim=16.
    let total = 2 * 4 * 16 * 16;
    let mut q: Vec<f32> = (0..total).map(|i| (i as f32) * 1e-3).collect();
    let mut k: Vec<f32> = (0..total).map(|i| (i as f32) * 1.5e-3).collect();
    let mut v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.7e-3).collect();
    let params: Vec<f32> = vec![];
    let mut out: Vec<f32> = vec![0.0; total];

    unsafe {
        forward(q.as_ptr(), k.as_ptr(), v.as_ptr(), params.as_ptr(), out.as_mut_ptr());
    }

    // Reference: scores = Q @ K (no transpose; K is already [..., head_dim, seq]),
    //            scaled = scores * 0.25,
    //            attn   = softmax(scaled, axis=-1),
    //            out    = attn @ V.
    let expected = reference_attention(&q, &k, &v, /* batch */ 2, /* heads */ 4, /* seq */ 16, /* hd */ 16);

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#x}), want {want} ({:#x})",
            got.to_bits(), want.to_bits()
        );
    }

    drop(forward);
    drop(lib);
}

/// Pure-Rust reference attention: Q @ K, scale, softmax(last-axis), then @ V.
/// Shapes: Q=[B,H,S,D], K=[B,H,D,S], V=[B,H,S,D] → out=[B,H,S,D].
fn reference_attention(q: &[f32], k: &[f32], v: &[f32],
                       b: usize, h: usize, s: usize, d: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; b * h * s * s];
    let mut out = vec![0.0f32; b * h * s * d];

    for bi in 0..b {
        for hi in 0..h {
            // scores[bi,hi,:,:] = Q[bi,hi,:,:] @ K[bi,hi,:,:]
            // Q is [s, d], K is [d, s]; product is [s, s].
            let q_off = (bi * h + hi) * s * d;
            let k_off = (bi * h + hi) * d * s;
            let sc_off = (bi * h + hi) * s * s;
            for i in 0..s {
                for j in 0..s {
                    let mut sum = 0.0f32;
                    for kk in 0..d {
                        sum += q[q_off + i*d + kk] * k[k_off + kk*s + j];
                    }
                    scores[sc_off + i*s + j] = sum * 0.25;
                }
            }
            // softmax row-wise.
            for i in 0..s {
                let row = &mut scores[sc_off + i*s..sc_off + (i+1)*s];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for x in row.iter_mut() { *x = (*x - max).exp(); }
                let sum: f32 = row.iter().sum();
                for x in row.iter_mut() { *x /= sum; }
            }
            // out[bi,hi,:,:] = scores[bi,hi,:,:] @ V[bi,hi,:,:]
            // scores is [s, s], V is [s, d]; product is [s, d].
            let v_off = (bi * h + hi) * s * d;
            let o_off = (bi * h + hi) * s * d;
            for i in 0..s {
                for j in 0..d {
                    let mut sum = 0.0f32;
                    for kk in 0..s {
                        sum += scores[sc_off + i*s + kk] * v[v_off + kk*d + j];
                    }
                    out[o_off + i*d + j] = sum;
                }
            }
        }
    }
    out
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p profiles-arm64 --test integration multi_input_attention_match_numerically
```

Expected: PASS. **This is the load-bearing acceptance test for M12.** If it fails:
- Numerical mismatch (a few ULPs off): possibly emit_softmax issue or matmul rounding diverges from Rust reference. Verify Rust reference matches the loop ordering of the generated asm exactly.
- All-NaN output: V-pointer corruption (clobbered by `bl _expf` because `ffi_save_set` is wrong).
- SIGBUS / SIGSEGV during test execution: stack alignment bug (alignment padding wrong for N=3).
- Some partial values right, others wrong: probably register collision in matmul (incorrect register reassignment in B.11).

### Task D.6: arm64 negative test for too_many_inputs

**Files:**
- Modify: `profiles/arm64/tests/integration.rs` (append)

- [ ] **Step 1: Write test**

```rust
#[test]
fn too_many_inputs_returns_too_many_inputs_error() {
    use compiler::ir::build;
    use profile_api::{LowerError, Profile};
    let src = std::fs::read_to_string("../../tests/fixtures/negative/too_many_inputs.nfl").unwrap();
    let nfl = compiler::parse(&src).unwrap();
    let uir = build(&nfl).unwrap();
    match profiles_arm64::Arm64.lower(&uir) {
        Err(LowerError::TooManyInputs { n, max, .. }) => {
            assert_eq!(n, 5);
            assert_eq!(max, 4);
        }
        Err(other) => panic!("expected TooManyInputs, got {other:?}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p profiles-arm64 --test integration too_many_inputs
```

Expected: PASS.

### Tasks D.7 / D.8 / D.9: x86_64 mirror of D.4 / D.5 / D.6

Same tests, swap `profiles_arm64::Arm64` for the x86_64 profile struct. ABI register names in comments adjust to SysV. The Rust reference `reference_attention` is identical.

### Task D.10: Group D pre-commit + commit

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add tests/fixtures/two_input_matmul.nfl \
        tests/fixtures/multi_input_attention.nfl \
        tests/fixtures/negative/too_many_inputs.nfl \
        profiles/arm64/tests/integration.rs \
        profiles/x86_64/tests/integration.rs
git commit -m "$(cat <<'EOF'
feat(m12): multi-input fixtures + per-profile FFI integration tests

Three new fixtures (spec §7):
- two_input_matmul.nfl: N=2 sanity; bit-exact arm64+x86_64 vs Rust ref.
- multi_input_attention.nfl: N=3 acceptance (Q/K/V real multi-input
  attention; v consumed post-softmax — exercises ffi_save_set's
  post-FFI register survival on both profiles).
- negative/too_many_inputs.nfl: N=5 → LowerError::TooManyInputs.

Per-profile FFI integration tests in profiles/{arm64,x86_64}/tests/
integration.rs:
- two_input_matmul_match_numerically (each profile)
- multi_input_attention_match_numerically (each profile) — THE
  acceptance test; bit-exact vs reference_attention pure-Rust impl.
- too_many_inputs_returns_too_many_inputs_error (each profile)

Test count: ~365 → ~371 (+6).

Spec ref: §7, §10.3, §10.4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Group E — Bench per-arity dispatch (commit 5)

**Spec ref:** §9.6.

**LoC delta:** ~50.
**Test count delta:** +1 (seed cascade unit test).

### Task E.1: Bench seed cascade unit test

**Files:**
- Modify: `bench/src/main.rs` (append #[cfg(test)] mod block)

- [ ] **Step 1: Find the existing fill_random helper**

```bash
grep -n "fn fill_random" bench/src/main.rs
```

- [ ] **Step 2: Write seed cascade test**

```rust
#[cfg(test)]
mod m12_seed_cascade {
    use super::*;

    #[test]
    fn seed_cascade_three_inputs_is_deterministic_and_independent() {
        // Synthetic FnSig with 3 inputs.
        let inputs_floats = vec![10, 20, 30];
        let n_inputs = inputs_floats.len();
        let seed: u64 = 42;

        // Build buffers via the cascade.
        let mut buffers: Vec<Vec<f32>> = Vec::with_capacity(n_inputs);
        for (i, &n) in inputs_floats.iter().enumerate() {
            let mut buf = vec![0f32; n];
            fill_random(&mut buf, seed.wrapping_add(i as u64));
            buffers.push(buf);
        }
        let mut params = vec![0f32; 5];
        fill_random(&mut params, seed.wrapping_add(n_inputs as u64));

        // Recompute independently with the same seeds — should match exactly.
        let mut ref0 = vec![0f32; 10]; fill_random(&mut ref0, 42);
        let mut ref1 = vec![0f32; 20]; fill_random(&mut ref1, 43);
        let mut ref2 = vec![0f32; 30]; fill_random(&mut ref2, 44);
        let mut refp = vec![0f32; 5];  fill_random(&mut refp, 45);

        assert_eq!(buffers[0], ref0);
        assert_eq!(buffers[1], ref1);
        assert_eq!(buffers[2], ref2);
        assert_eq!(params, refp);
    }
}
```

- [ ] **Step 3: Run — expect FAIL** (cascade not yet implemented in bench main)

### Task E.2: Implement per-arity dispatch in bench

**Files:**
- Modify: `bench/src/main.rs`

- [ ] **Step 1: Find the bench_one_fixture function**

```bash
grep -n "fn bench_one_fixture\|fn time_forward\|sig.input" bench/src/main.rs
```

- [ ] **Step 2: Replace single-input buffer construction with cascade**

```rust
// bench/src/main.rs (inside bench_one_fixture, replacing the buffer-allocation block)
let n_inputs = sig.inputs_floats.len();

// Build per-input buffers via seed cascade (spec §9.6).
let mut input_bufs: Vec<Vec<f32>> = Vec::with_capacity(n_inputs);
for (i, &n_floats) in sig.inputs_floats.iter().enumerate() {
    let mut buf = vec![0f32; n_floats];
    fill_random(&mut buf, seed.wrapping_add(i as u64));
    input_bufs.push(buf);
}
let mut params: Vec<f32> = vec![0f32; sig.params_floats];
fill_random(&mut params, seed.wrapping_add(n_inputs as u64));
let mut output: Vec<f32> = vec![0f32; sig.output_floats];
```

- [ ] **Step 3: Add per-arity dispatch for the FFI call**

```rust
match n_inputs {
    1 => {
        type Fn1 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
        let f: Symbol<Fn1> = unsafe { lib.get(sig.name.as_bytes()).unwrap() };
        // Run warmup + measurement loops calling f(in0, params, out).
        time_forward_arity1(&f, &input_bufs[0], &params, &mut output, /* warmup */, /* iters */)
    }
    2 => {
        type Fn2 = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
        let f: Symbol<Fn2> = unsafe { lib.get(sig.name.as_bytes()).unwrap() };
        time_forward_arity2(&f, &input_bufs[0], &input_bufs[1], &params, &mut output, ...)
    }
    3 => {
        type Fn3 = unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32);
        let f: Symbol<Fn3> = unsafe { lib.get(sig.name.as_bytes()).unwrap() };
        time_forward_arity3(&f, &input_bufs[0], &input_bufs[1], &input_bufs[2], &params, &mut output, ...)
    }
    n => unimplemented!(
        "bench: arity {n} not supported (M12 caps at N=4; current bench fixtures all N=1)"
    ),
}
```

(Define `time_forward_arity{1,2,3}` as small helpers with the inner warmup+measurement loop. Or inline if cleaner.)

- [ ] **Step 4: Run unit test**

```bash
cargo test -p bench m12_seed_cascade
```

Expected: PASS.

- [ ] **Step 5: Run bench manually for an N=1 fixture to confirm M11-identical output**

```bash
cargo run -p bench --release -- --profile arm64 --format markdown --seed 42
```

Expected: output identical to M11 in everything except wall-clock timing values.

### Task E.3: Group E pre-commit + commit

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
git add bench/src/main.rs
git commit -m "$(cat <<'EOF'
feat(m12): bench per-arity dispatch + seed cascade

bench/src/main.rs gains per-arity dispatch (match on
sig.inputs_floats.len()) for the FFI call, with concrete
extern "C" fn types for arity {1, 2, 3}. Arity > 3 panics with
unimplemented!() noting the M12 N=4 cap.

Input buffer construction uses the seed cascade per spec §9.6:
inputs[i] gets seed.wrapping_add(i as u64); params get
seed.wrapping_add(n_inputs as u64). This preserves the M11 N=1
behaviour bit-exactly (inputs[0] gets seed; params get seed+1)
while extending deterministically to multi-input.

Test count: ~371 → ~372 (+1 seed cascade unit test).

Spec ref: §9.6, §10.5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Group F — Documentation + closure (commit 6)

**Spec ref:** §10.7 gates #19–#25.

**LoC delta:** ~250 docs only.
**Test count delta:** 0.

### Task F.1: docs/profile_guide/arm64.md — Multi-Input ABI section

**Files:**
- Modify: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Append new section**

```markdown
## Multi-Input ABI (M12)

NeuralForge models declare inputs as `variable_decl` statements at the
top of `model_body`. M12 lowers each input to a distinct AAPCS argument
register; arity is the number of `variable_decl` statements (= length
of `UirModel.inputs`).

### Register layout

| N | x0 | x1 | x2 | x3 | x4 | x5 |
|---|----|----|----|----|----|----|
| 1 | in₀ | params | out | — | — | — |
| 2 | in₀ | in₁ | params | out | — | — |
| 3 | in₀ | in₁ | in₂ | params | out | — |
| 4 | in₀ | in₁ | in₂ | in₃ | params | out |

Source-order of `variable_decl` statements determines register
assignment (lexical order = ABI register order = `model.inputs[i]`
order).

### Arity cap

M12 caps inputs at N=4. Models with N≥5 return
`LowerError::TooManyInputs { n, max: 4, span }` from
`Profile::lower`. Extending beyond N=4 requires updating
`profiles/arm64/src/abi.rs::INPUT_REGS` AND adding stack-spill
emission for N≥7.

### Stack alignment around FFI calls

Functions calling external math (`bl _expf` for softmax / fused
softmax) save the conservative caller-saved set
`AbiContext::ffi_save_set()` = first `N+2` registers from
`INPUT_REGS`. Save uses paired `stp` with `xzr` padding when
cardinality is odd, ensuring SP delta is always a multiple of 16
(AAPCS public-call invariant).

For N=3 (5 registers in the save set):

```
stp     x0, x1, [sp, #-16]!
stp     x2, x3, [sp, #-16]!
stp     x4, xzr, [sp, #-16]!     ; xzr-padded for alignment
; SP delta = 48 bytes (multiple of 16)
bl _expf
ldp     x4, xzr, [sp], #16       ; LIFO restore
ldp     x2, x3, [sp], #16
ldp     x0, x1, [sp], #16
```

Restore order is strict LIFO relative to save.

### Matmul scratch register layout

`emit_matmul` uses non-ABI scratch registers exclusively for both base
pointers and per-iter slice pointers, so no ABI register is clobbered:

| Role | Register |
|------|----------|
| A base ptr | x9 |
| B base ptr | x10 |
| DST base ptr | x11 |
| A_slice ptr | x12 |
| B_slice ptr | x13 |
| DST_slice ptr | x14 |

This eliminates the M10 `stp x1, x2, [sp, #-16]!` outer-loop spill
block: matmul body contains zero `stp` instructions.
```

### Task F.2: docs/profile_guide/x86_64.md — Multi-Input ABI section

Mirror with SysV register names (`%rdi`/`%rsi`/`%rdx`/`%rcx`/`%r8`/`%r9`), `pushq`/`popq` mechanism, `pushq %rax` padding, `%r10`/`%r11`/`%r12`/`%r13`/`%r14`/`%r15` matmul scratch.

### Task F.3: docs/language_reference/uir.md update

- [ ] **Step 1: Append note**

```markdown
## Multi-Input Models (M12)

Beginning with M12, codegen consumes the full `UirModel.inputs:
Vec<NodeId>` (previously only `inputs.first()` was honored). The
position of an Input node in `model.inputs` determines its physical
register assignment in the generated function: position 0 → first ABI
input register, position 1 → second, etc., up to a cap of N=4 per the
profile-specific ABI window.
```

### Task F.4: docs/language_reference/grammar.md update

- [ ] **Step 1: Append note**

```markdown
## Multi-Input Convention (M12)

NFL grammar permits any number of `variable_decl` statements in
`model_body`. Each `variable_decl` is recognised by IR-build as a model
input and pushed in declaration order to `UirModel.inputs`. Lexical
order of `variable_decl` statements determines the ABI register order
in generated code (first declared → first ABI register, etc.).

By convention, place all `variable_decl` statements at the top of
`model_body`, before any `pipeline_stmt` or `named_pipeline_stmt`.
The grammar does NOT enforce this; interleaved declarations are valid
syntax. The convention exists to make the input set visually obvious
and to align with how viewer tools render the model interface.

Profile codegen caps inputs at N=4 (see `docs/profile_guide/<arch>.md`
for the ABI register table). Models with N≥5 fail at lowering with
`LowerError::TooManyInputs`.
```

### Task F.5: PROJECT_SPEC.md — M12 row + Current Status

- [ ] **Step 1: Add M12 row to milestones table**

```markdown
| 12 | NFL multi-input ABI (A1 — first leg of Axis 2 follow-up) (complete) | Multi-input function ABI (N=1..4) via per-profile `AbiContext`. New `profiles/{arm64,x86_64}/src/abi.rs` carrying `n_inputs: usize`, `input_reg/params_reg/output_reg` arity-aware accessors, `ffi_save_set` conservative caller-saved spill set, `emit_ffi_save/emit_ffi_restore` with arm64 `xzr` padding (or x86_64 `pushq %rax` padding) for odd cardinality and strict LIFO restore. `BufferLoc::InputReg(usize)` carries the input index. `walk_model` constructs `AbiContext` once and threads `&abi` through every op-emitter; arity > 4 returns `LowerError::TooManyInputs`. `emit_matmul` rework on both profiles: per-iter slice pointers move to non-ABI scratch (x12/x13/x14 arm64; %r13/%r14/%r15 x86_64), eliminating the M10 outer-loop `stp x1, x2` (arm64) / `movq → %xmm6/7/8` (x86_64) spill blocks. New fixtures: `two_input_matmul.nfl` (N=2 sanity), `multi_input_attention.nfl` (N=3 acceptance — V consumed post-softmax), `negative/too_many_inputs.nfl` (N=5 → error). Bench gains per-arity dispatch + seed cascade. Test count: 344 → ~372. |
```

- [ ] **Step 2: Update Current Status**

Find the section "Current Status" and update to reference M12 closure. Update the Strategic Roadmap section to mark A1's "first leg" closed.

### Task F.6: CLAUDE.md — Current Status + Repository Structure tree

- [ ] **Step 1: Update Current Status**

```markdown
**Milestone 12 complete. ~372 tests passing on macOS arm64 (~388 on
Linux x86_64 CI).** All workspace gates clean.

M12 closed A1 (multi-input function ABI) — first leg of Axis 2
follow-up. Per-profile `AbiContext` enables arity-aware register
mapping for inputs/params/output, with formal stack-alignment + LIFO
invariants around FFI calls. `emit_matmul` simplified by moving
per-iter slice pointers off ABI registers — old M10 outer-loop spill
blocks are removed. Three new fixtures exercise N=2, N=3 (with
post-`_expf` register survival), and N=5 (negative).

Strategic direction: see PROJECT_SPEC.md §"Strategic Roadmap" — Axis 2
A1 closed in M12; A2 (transformer block) and A3 (viewer annotations)
remain open follow-ups; Axis 1 (SIMD) and Axis 3 (bare-metal expf)
unchanged.
```

- [ ] **Step 2: Update Repository Structure tree**

Add `abi.rs` lines under both arm64 and x86_64 src/ trees:

```
│   ├── src/
│   │   ├── lib.rs
│   │   ├── abi.rs        ← AbiContext (M12)
│   │   ├── types.rs
│   │   ...
```

### Task F.7: DEVLOG.md — M12 closure entry

- [ ] **Step 1: Add entry on top (reverse-chronological)**

```markdown
## 2026-05-09 (or merge date) — Milestone 12 closed: NFL multi-input ABI (A1)

### What was done

[Summary mirroring PROJECT_SPEC.md M12 row but more narrative —
3-5 paragraphs covering: Group A foundation, Groups B+C codegen,
Group D fixtures+integration, Group E bench, Group F docs/closure.
Highlight emit_matmul rework as the high-risk task and how the
acceptance fixture multi_input_attention.nfl validated it.]

### Decisions made

[Reference brainstorm + spec for the key design decisions: γ over β
ABI choice, scope keep-tight at A1 only (no bundle), N=4 cap,
materialise_ptr on AbiContext (not free fn), emit_matmul per-iter
slice ptrs to non-ABI scratch, etc.]

### Problems encountered

[Anything surprising during implementation — e.g., golden-file
divergence after matmul rework needed regeneration; any libffi
edge cases in cc compile; etc.]

### Next step

[A1 closed. Next milestone selection runs over the post-M12 Strategic
Roadmap. Open candidates: Axis 1 (SIMD/AVX), Axis 2 A2 (transformer
block, builds on M12 foundation), Axis 2 A3 (viewer annotations for
multi-input), Axis 3 (bare-metal expf).]
```

### Task F.8: Final pre-commit + commit

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

All exit 0.

```bash
git add docs/profile_guide/arm64.md \
        docs/profile_guide/x86_64.md \
        docs/language_reference/uir.md \
        docs/language_reference/grammar.md \
        PROJECT_SPEC.md \
        CLAUDE.md \
        DEVLOG.md
git commit -m "$(cat <<'EOF'
docs(m12): close M12 — profile_guide / language_reference / PROJECT_SPEC / CLAUDE / DEVLOG

Multi-Input ABI sections in profile_guide/{arm64,x86_64}.md document
the new register layout per arity, alignment + LIFO invariants
around FFI calls, and the matmul scratch register layout that
eliminated the M10 outer-loop spill block.

language_reference/{uir,grammar}.md note the new behaviour: codegen
consumes all of UirModel.inputs (not just first); convention is
inputs at top of model_body though grammar permits interleaved.

PROJECT_SPEC.md gains M12 row, marks A1 first leg closed in
Strategic Roadmap, updates Current Status. CLAUDE.md mirrors
Current Status + adds abi.rs to Repository Structure tree.

DEVLOG.md gets the M12 closure entry per the standard four-section
template.

Spec ref: §10.7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Final verification**

```bash
git log --oneline -7
cargo test --workspace 2>&1 | tail -5
```

Expected:
- Last 6 commits are the M12 commit chain (Group A through Group F).
- Test count = ~372 passed (or higher), 0 failed.

---

## Self-Review Checklist (run AFTER plan complete, BEFORE handing to executor)

### Spec coverage

For each spec acceptance gate (#1–#25 in §10), point to the task that delivers it:

- #1–#4 workspace gates → Pre-commit ritual at every Group commit.
- #4 specifically (test count ≥ 369) → end of Group D (target ~371) and Group E (~372).
- N=1 regression bit-exact → Tasks A.1, A.6, B.12, C.10.
- #5–#6 two_input_matmul → Tasks D.4 (arm64), D.7 (x86_64).
- #7–#8 multi_input_attention → Tasks D.5 (arm64), D.8 (x86_64).
- #9–#10 too_many_inputs → Tasks D.6 (arm64), D.9 (x86_64).
- #11–#12 bench harness for existing fixtures → Task E.2 Step 5.
- #13–#18 AbiContext unit tests → Tasks B.2–B.5 (arm64), C.3–C.4 (x86_64).
- #19–#20 profile_guide docs → Tasks F.1, F.2.
- #21 uir.md → Task F.3.
- #22 grammar.md → Task F.4.
- #23 PROJECT_SPEC → Task F.5.
- #24 CLAUDE → Task F.6.
- #25 DEVLOG → Task F.7.

All gates covered.

### Type consistency

- `BufferLoc::InputReg(usize)` is the SAME shape used in both profiles' buffer.rs (B.6, C.5).
- `AbiContext::input_reg(idx)` returns `&'static str` consistently in arm64 and x86_64.
- `FnSig.inputs_floats: Vec<usize>` consistent across profile-api and consumers (bench, integration tests).
- `LowerError::TooManyInputs { n, max, span }` field names consistent across declaration (A.3), construction (B.7, C.6), and tests (D.6, D.9).

### Placeholder scan

- No "TBD", "TODO", "fill in" markers.
- Every code block contains compilable Rust (modulo elided `// ... existing args ...` which references known function signatures).
- Every command is concrete and copy-paste-runnable.
- Test names are concrete and unique.

---

## Execution Handoff

Plan complete and saved to [`docs/superpowers/plans/2026-05-09-m12-multi-input-abi.md`](2026-05-09-m12-multi-input-abi.md). Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task group, review between groups, fast iteration with isolated context per task. Best for M12 because each Group commit is reviewable independently and the high-risk Group B emit_matmul rework (Task B.11) merits dedicated review of regenerated goldens before continuing to Group C.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster for the document/closure groups (E, F) but riskier for the codegen groups (B, C) without external review checkpoints.

Which approach?
