# M4a — `profiles/arm64` Scalar Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower `input → linear[N] → relu` UIR to native AArch64 assembly callable as a C function, end-to-end from `.nfl` source through `nflc compile` CLI to a runnable `.dylib` linked via `cc`.

**Architecture:** Refactor workspace into 3 crates (`compiler` lib / `nflc` bin / `profiles/arm64` lib), no cycles. `profiles/arm64::lower(&Uir) -> Result<Asm, LowerError>` walks UIR and emits AArch64 Mach-O assembly text. CLI wires it via new `compile` subcommand. Integration test assembles the asm with `cc -shared` and calls the function via `libloading` FFI from a Rust test, comparing against a pure-Rust matmul+relu reference.

**Tech Stack:** Rust 2021 (std-only for production crates; `libloading` 0.8 dev-dep for `profiles/arm64` integration tests only). AArch64 assembly (Mach-O syntax, AAPCS64 calling convention). `cc` (Apple clang) for assembling; `as`/`ld` are invoked by `cc` under the hood.

**Source spec:** [`docs/superpowers/specs/2026-05-03-m4a-arm64-codegen-design.md`](../specs/2026-05-03-m4a-arm64-codegen-design.md). All architectural decisions and rationale live there. **If anything in this plan disagrees with the spec, the spec wins.**

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m4-generic-profile` (branch `claude/m4-generic-profile`, base `main` at commit `4902f8d`).

**Project conventions** (`CLAUDE.md` + spec §11):
- Build must be **warning-free** at every commit: `cargo build --workspace`.
- Clippy must be clean: `cargo clippy --workspace --all-targets -- -D warnings`.
- Production crates strictly std-only. Dev-deps admissible by need; M4a starts the list with `libloading` (test-only).
- TDD: failing test first, minimal impl, then commit. Frequent commits.
- Each session ends with a `DEVLOG.md` entry.

**Pre-task baseline (recorded once before Task 1):**

```bash
# Run from worktree root.
# Record the actual numbers — the plan never references them as hard-coded; later AC checks just say "no regression".
cargo test 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "BASELINE TESTS PASSED:", sum}'
# Expect: 106 (per CLAUDE.md "Current Status"). If different, that's still the baseline; just record it.
```

---

## File Structure

### Created (during M4a)

| Path | Responsibility | Created in task |
|---|---|---|
| `nflc/Cargo.toml` | Bin-only crate manifest. `[[bin]] name = "nflc"`. `[dependencies] compiler = { path = "../compiler" }, profiles-arm64 = { path = "../profiles/arm64" }`. | Task 1 |
| `nflc/src/main.rs` | CLI dispatcher. Moved from `compiler/src/main.rs`, imports rewritten `nflc::*` → `compiler::*`. Adds new `compile` subcommand in Task 8. | Task 1 (move), Task 8 (add `compile`) |
| `profiles/arm64/Cargo.toml` | Lib-only crate manifest. `[dependencies] compiler = { path = "../../compiler" }`. `[dev-dependencies] libloading = "0.8"`. | Task 2 |
| `profiles/arm64/src/lib.rs` | `pub use` re-exports + `pub fn lower(uir: &Uir) -> Result<Asm, LowerError>` entry point. | Task 2 |
| `profiles/arm64/src/types.rs` | `Asm`, `FnSig`, `LowerError` (with `#[non_exhaustive]` and all variants from spec §5). | Task 2 |
| `profiles/arm64/src/asm.rs` | Low-level assembly building blocks: register names as constants, `format_function_header(&FnSig) -> String`, `format_function_footer() -> String`, label/symbol helpers. | Task 3 |
| `profiles/arm64/src/codegen.rs` | UIR walker: `walk_uir`, `walk_model`, `emit_op_linear`, `emit_op_relu`, error-classifying dispatch. | Tasks 3, 4, 5, 6 |
| `profiles/arm64/src/tests.rs` | Unit tests (~7-8) for lower output. | Tasks 3-6 |
| `profiles/arm64/tests/integration.rs` | End-to-end FFI test. Single test `tinymlp_no_softmax_runs_correctly`. | Task 9 |
| `profiles/arm64/tests/common/mod.rs` | Test helpers: `cc_available()`, `compile_to_dylib(asm, name) -> PathBuf`. | Task 9 |
| `tests/fixtures/m4_linear_relu.nfl` | M4a fixture: minimal model that lowers cleanly (no softmax, no bias). | Task 7 |
| `docs/profile_guide/arm64.md` | Profile guide: ABI, weight layout, supported ops in M4a, where to add a new op, where to add a new arch profile. | Task 10 |

### Modified

| Path | What changes | In task |
|---|---|---|
| `Cargo.toml` (root) | `members = ["compiler"]` → `members = ["compiler", "nflc", "profiles/arm64"]`. | Task 1 |
| `compiler/Cargo.toml` | Drop `[[bin]]` section. Rename `[package].name` from `"nflc"` to `"compiler"`. Update `description`. | Task 1 |
| `compiler/src/main.rs` | **Deleted** (content moves to `nflc/src/main.rs`). | Task 1 |
| `compiler/tests/uir_fixtures.rs` | All `use nflc::*;` → `use compiler::*;` (6 sites). | Task 1 |
| `compiler/tests/fixtures.rs` | All `use nflc::*;` → `use compiler::*;` (2 sites). | Task 1 |
| `compiler/tests/uir_fixtures.rs` (again) | New `mod m4_linear_relu` with the fixture's UIR-build assertion. | Task 7 |
| `docs/language_reference/uir.md` | 1-2 line cross-link note: `linear[N]` without `bias` attr is interpreted by codegen profiles; arm64-specific behaviour (no bias add) documented in `docs/profile_guide/arm64.md`. | Task 11 |
| `PROJECT_SPEC.md` | Milestones table: M4 row updated from "`generic` profile" to "`arm64` profile". "Architecture Profiles" table: drop `generic` row, add `arm64` row. | Task 11 |
| `CLAUDE.md` | "Repository Structure" → reflect 3-crate workspace. "Current Status" → M4a complete; M4b next. | Task 12 |
| `DEVLOG.md` | M4a closeout entry (top of file). | Task 12 |

### Deleted

| Path | Reason | Deleted in task |
|---|---|---|
| `profiles/generic/` (empty) | Name abandoned per spec §1. | Task 1 |
| `profiles/x86_64/` (empty) | YAGNI placeholder; M6 will create for real. | Task 1 |
| `profiles/riscv64/` (empty) | YAGNI placeholder. | Task 1 |

---

## Verification approach

| Check | When | How |
|---|---|---|
| `cargo build --workspace` warning-free | Every task | From worktree root. |
| `cargo clippy --workspace --all-targets -- -D warnings` clean | Tasks 1, 8, 9, 12 | After workspace restructure, after CLI wiring, after integration test, at closeout. |
| All tests pass | Every task | `cargo test --workspace`. Track count goes up monotonically. |
| Integration test runs (or skips cleanly on non-arm64) | Task 9, Task 12 | `cargo test -p profiles-arm64 --test integration`. |
| CLI smoke positive | Task 12 | `cargo run --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m4a.s` produces a valid `.s`; then `cc -shared -arch arm64 -o /tmp/m4a.dylib /tmp/m4a.s` succeeds. |
| CLI smoke negative | Task 12 | `cargo run --bin nflc -- compile tests/fixtures/tiny_mlp.nfl --profile arm64` exits 1 with `LowerError::UnsupportedOp { op: "softmax", ... }` rendered via `render_error_with_snippet`. |
| Unknown profile rejection | Task 8 | `cargo run --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile xyz` exits 1 with "unknown profile 'xyz' (supported: arm64)". |

---

## Task list

| # | Task | Mode | Commits |
|---|---|---|---|
| 1 | Workspace restructure (3 crates, move main.rs, rename package, fix imports, drop empty dirs) | INLINE (mechanical, many files) | 1 |
| 2 | `profiles/arm64` skeleton + types (`Asm`, `FnSig`, `LowerError`) | SUBAGENT | 1 |
| 3 | Linear codegen — function harness (`.globl`, label, prologue, ret) | SUBAGENT | 1 |
| 4 | Linear codegen — matmul body (3 nested loops with `fmadd`) | SUBAGENT | 1 |
| 5 | Relu codegen — separate elementwise loop (`fmov` zero + `fmax`) | SUBAGENT | 1 |
| 6 | Error variants (`LinearWithBias`, `UnsupportedOp` × 2, `DuplicateModelName`) | SUBAGENT | 1 |
| 7 | New M4a fixture + UIR-build test | INLINE (trivial) | 1 |
| 8 | CLI `nflc compile` subcommand | SUBAGENT | 1 |
| 9 | Integration test (cc + libloading FFI) | SUBAGENT | 1 |
| 10 | `docs/profile_guide/arm64.md` | SUBAGENT (prose) | 1 |
| 11 | Other doc updates (`uir.md` cross-link, `PROJECT_SPEC.md` milestones) | INLINE (small edits) | 1 |
| 12 | Closeout (DEVLOG, CLAUDE.md, final smoke + clippy) | INLINE | 1 |

**Total:** 12 tasks, 12 commits (or 13 if Task 1 splits the workspace move from the import sweep). ~10-15 new tests on top of baseline.

---

## Task 1: Workspace restructure

**Goal:** Three-crate workspace; existing tests still pass; no codegen yet.

**Files:**
- Create: `nflc/Cargo.toml`, `nflc/src/main.rs` (from move)
- Modify: `Cargo.toml` (root), `compiler/Cargo.toml`, `compiler/tests/uir_fixtures.rs`, `compiler/tests/fixtures.rs`
- Delete: `compiler/src/main.rs`, `profiles/generic/`, `profiles/x86_64/`, `profiles/riscv64/`

- [ ] **Step 1: Record baseline test count**

```bash
cargo test 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "BASELINE:", sum}'
```

Note the number; subsequent tasks must not regress below this. Expected: 106.

- [ ] **Step 2: Update root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = ["compiler", "nflc", "profiles/arm64"]
```

- [ ] **Step 3: Update `compiler/Cargo.toml`**

```toml
[package]
name = "compiler"
version = "0.1.0"
edition = "2021"
description = "NeuralForge compiler core — lexer, parser, UIR builder (lib only)"
license = "MIT OR Apache-2.0"

[dependencies]

[lib]
path = "src/lib.rs"
```

(Drop the `[[bin]]` section entirely. Rename package `nflc` → `compiler`.)

- [ ] **Step 4: Create `nflc/` crate directory and Cargo manifest**

```bash
mkdir -p nflc/src
```

`nflc/Cargo.toml`:

```toml
[package]
name = "nflc"
version = "0.1.0"
edition = "2021"
description = "NeuralForge Language Compiler — CLI binary"
license = "MIT OR Apache-2.0"

[dependencies]
compiler = { path = "../compiler" }
profiles-arm64 = { path = "../profiles/arm64" }

[[bin]]
name = "nflc"
path = "src/main.rs"
```

- [ ] **Step 5: Move `compiler/src/main.rs` → `nflc/src/main.rs`**

```bash
git mv compiler/src/main.rs nflc/src/main.rs
```

- [ ] **Step 6: Rewrite imports in `nflc/src/main.rs`**

In the moved file, replace every `nflc::` with `compiler::` (17 sites). Use sed:

```bash
sed -i '' 's/nflc::/compiler::/g' nflc/src/main.rs
```

Verify by grep:

```bash
grep -n "nflc::" nflc/src/main.rs
# Expected: no matches.
grep -n "compiler::" nflc/src/main.rs
# Expected: 17 matches.
```

- [ ] **Step 7: Rewrite imports in test files**

```bash
sed -i '' 's/use nflc::\*/use compiler::*/g' compiler/tests/uir_fixtures.rs compiler/tests/fixtures.rs
```

Verify:

```bash
grep -rn "nflc::" compiler/tests/ compiler/src/
# Expected: no matches.
```

- [ ] **Step 8: Create empty `profiles/arm64/` placeholder so workspace resolves**

The workspace will fail to load if `profiles/arm64/Cargo.toml` doesn't exist. Create the bare minimum so Step 9 can build; Task 2 fills in the real content.

```bash
mkdir -p profiles/arm64/src
```

`profiles/arm64/Cargo.toml` (placeholder — Task 2 expands):

```toml
[package]
name = "profiles-arm64"
version = "0.1.0"
edition = "2021"
description = "NeuralForge arm64 codegen profile"
license = "MIT OR Apache-2.0"

[dependencies]
```

`profiles/arm64/src/lib.rs` (placeholder):

```rust
//! NeuralForge arm64 scalar codegen profile (M4a).
```

- [ ] **Step 9: Delete the empty placeholder profile dirs**

```bash
rmdir profiles/generic profiles/x86_64 profiles/riscv64
```

(All three are empty per spec §1; `rmdir` succeeds. If non-empty due to drift, investigate before forcing.)

- [ ] **Step 10: Build + test the restructured workspace**

```bash
cargo build --workspace
```

Expected: zero warnings.

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TESTS PASSED:", sum}'
```

Expected: same as the baseline from Step 1 (106).

- [ ] **Step 11: Clippy clean**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: exit 0.

- [ ] **Step 12: Commit**

```bash
git add Cargo.toml compiler/Cargo.toml compiler/tests/ nflc/ profiles/arm64/
git rm -r profiles/generic profiles/x86_64 profiles/riscv64
git status
# Confirm: only the expected paths staged. compiler/src/main.rs should appear deleted (moved to nflc/).
git commit -m "refactor(m4a/workspace): split into 3 crates (compiler lib + nflc bin + profiles/arm64)

Per spec §4 — establishes the dependency graph compiler ← nflc and
compiler ← profiles/arm64 with no cycles. M5/M6 will add new
profile crates without touching existing ones.

Changes:
- Rename compiler package nflc → compiler. Drop its [[bin]] section.
- Create nflc/ crate (bin only) with main.rs moved from compiler/src/.
- Rewrite all 'nflc::' imports to 'compiler::' (17 sites in main.rs,
  6 sites in uir_fixtures.rs, 2 sites in fixtures.rs).
- Create profiles/arm64/ skeleton (Cargo.toml + empty lib.rs) so
  workspace resolves; Task 2 fills in real content.
- Delete empty placeholder dirs profiles/{generic,x86_64,riscv64}/.

cargo build --workspace clean, cargo test passes baseline count
(106), cargo clippy --workspace --all-targets -- -D warnings clean.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `profiles/arm64` types (`Asm`, `FnSig`, `LowerError`) + entry stub

**Goal:** Public types compile; `lower()` is a stub returning `LowerError::UnsupportedOp` for everything. One unit test asserts the stub behaviour.

**Files:**
- Create: `profiles/arm64/src/types.rs`, `profiles/arm64/src/tests.rs`
- Modify: `profiles/arm64/src/lib.rs`

- [ ] **Step 1: Write `profiles/arm64/src/types.rs`**

```rust
//! Public types for the arm64 codegen profile.

use compiler::ast::Span;

/// Generated assembly source plus per-function metadata.
#[derive(Debug, Clone)]
pub struct Asm {
    /// Full AArch64 Mach-O assembly source. UTF-8.
    pub source: String,
    /// One entry per UirModel in the input UIR, in declaration order.
    pub functions: Vec<FnSig>,
}

/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// External symbol name without leading underscore. e.g. "nfl_forward_TinyMLP".
    /// Mach-O linkers prepend the underscore; `dlsym` users pass this name verbatim.
    pub name: String,
    /// Original UIR model name.
    pub model: String,
    /// Number of f32 elements in the input buffer.
    pub input_floats: usize,
    /// Total number of f32 elements across all weight matrices, packed in
    /// UIR-node order. M4a always single-Linear so this equals the one
    /// matrix's element count.
    pub weight_floats: usize,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
}

/// Errors that can occur during lowering.
///
/// `#[non_exhaustive]` — variants representing deferred features
/// (`UnsupportedOp`, `LinearWithBias`) become unreachable as M4b/c add
/// coverage and may be removed at that point.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// Op is not supported in the current M4 slice.
    /// `op` is the lowercase token name (e.g. "softmax", "dropout").
    UnsupportedOp { op: String, span: Span },
    /// `linear[N, bias=true]` is not yet implemented (M4b).
    LinearWithBias { span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    /// Should be unreachable if the IR builder did its job.
    ShapeNotConcrete { span: Span },
    /// Two `UirModel`s share the same `name` — would emit duplicate
    /// `nfl_forward_<name>` symbols. M4b moves this check into `ir::build`.
    DuplicateModelName { name: String, span: Span },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedOp { op, .. } =>
                write!(f, "operation '{}' is not supported by the arm64 profile in M4a", op),
            LowerError::LinearWithBias { .. } =>
                write!(f, "linear[..., bias=true] is not yet implemented (M4b)"),
            LowerError::ShapeNotConcrete { .. } =>
                write!(f, "internal: UIR shape was not fully resolved before lowering"),
            LowerError::DuplicateModelName { name, .. } =>
                write!(f, "duplicate model name '{}': would emit conflicting symbols", name),
        }
    }
}

impl LowerError {
    /// Returns the source span associated with the error.
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::LinearWithBias { span } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::DuplicateModelName { span, .. } => *span,
        }
    }
}
```

- [ ] **Step 2: Write `profiles/arm64/src/lib.rs`**

```rust
//! NeuralForge arm64 scalar codegen profile (M4a).
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via [`lower`].

mod types;

pub use types::{Asm, FnSig, LowerError};

use compiler::{Uir, UirModel};

/// Lower a [`Uir`] to AArch64 assembly.
///
/// Returns [`LowerError`] on any unsupported op or structural problem.
/// See the M4a spec for op coverage details.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    // Stub: real impl arrives in Tasks 3-6.
    if let Some(model) = uir.models.first() {
        // Find the first op in the model and report it as unsupported,
        // so the stub at least returns a meaningful error per UIR.
        for node in &model.nodes {
            if let compiler::NodeKind::Op { op, .. } = &node.kind {
                return Err(LowerError::UnsupportedOp {
                    op: format!("{}", op),
                    span: node.source_span,
                });
            }
        }
    }
    Ok(Asm { source: String::new(), functions: Vec::new() })
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 3: Write the first unit test in `profiles/arm64/src/tests.rs`**

```rust
//! Unit tests for the arm64 codegen profile.

use super::*;

/// Build a UIR from a small NFL source string. Used by every test below.
fn build_uir(src: &str) -> compiler::Uir {
    let ast = compiler::parse(src).expect("parse");
    compiler::ir::build(&ast).expect("ir::build")
}

#[test]
fn empty_uir_lowers_to_empty_asm() {
    let uir = compiler::Uir { models: Vec::new() };
    let asm = lower(&uir).unwrap();
    assert!(asm.source.is_empty());
    assert!(asm.functions.is_empty());
}

#[test]
fn unsupported_op_returns_unsupported() {
    // tiny_mlp ends in softmax — not supported in M4a
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> softmax\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "softmax"));
}
```

- [ ] **Step 4: Build + test**

```bash
cargo build -p profiles-arm64
```

Expected: zero warnings.

```bash
cargo test -p profiles-arm64
```

Expected: 2 tests passing.

- [ ] **Step 5: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4a/arm64): scaffold types + lower() stub

Public API per spec §5: Asm, FnSig, LowerError (#[non_exhaustive],
4 variants). Display for LowerError, span() accessor.

lower() is a stub: returns UnsupportedOp for the first op of the
first model so callers immediately see what op blocked them.
Real codegen lands in Tasks 3-6.

2 unit tests cover empty-UIR and unsupported-op paths.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Linear codegen — function harness (`.globl`, label, prologue, ret)

**Goal:** `lower()` for a UIR containing a single `Input → Linear` model emits the function header, prologue, label, and `ret`. No matmul body yet — that's Task 4. Test asserts the symbol is correctly named and the function is well-formed.

**Files:**
- Create: `profiles/arm64/src/asm.rs`, `profiles/arm64/src/codegen.rs`
- Modify: `profiles/arm64/src/lib.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Write a failing test in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn linear_emits_function_with_correct_symbol_and_ret() {
    // model M [b=2]: x: Tensor[b, 3]
    //     x -> linear[2]
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");

    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M");
    assert_eq!(sig.model, "M");
    assert_eq!(sig.input_floats, 6);   // 2*3
    assert_eq!(sig.weight_floats, 6);  // 3*2
    assert_eq!(sig.output_floats, 4);  // 2*2

    let s = &asm.source;
    assert!(s.contains(".globl _nfl_forward_M"), "missing .globl in:\n{s}");
    assert!(s.contains("_nfl_forward_M:"), "missing label in:\n{s}");
    assert!(s.contains("ret"), "missing ret in:\n{s}");
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 linear_emits_function_with_correct_symbol_and_ret
```

Expected: FAIL — current stub returns `UnsupportedOp` for any op.

- [ ] **Step 3: Write `profiles/arm64/src/asm.rs`**

```rust
//! Low-level AArch64 assembly building blocks.
//!
//! Helpers that emit common instruction sequences and label/symbol formatting.
//! No UIR knowledge here — pure asm-string assembly.

use crate::FnSig;

/// Mach-O symbol prefix. Apple's `as` prepends `_` to C symbol names.
pub const MACHO_SYM_PREFIX: &str = "_";

/// Format the function header: directives + globl + alignment + label.
pub fn format_function_header(sig: &FnSig) -> String {
    let mut out = String::new();
    out.push_str(&format!(".globl {}{}\n", MACHO_SYM_PREFIX, sig.name));
    out.push_str(".p2align 2\n");
    out.push_str(&format!("{}{}:\n", MACHO_SYM_PREFIX, sig.name));
    out
}

/// Format the function epilogue: `ret`.
pub fn format_function_footer() -> String {
    "    ret\n".to_string()
}
```

- [ ] **Step 4: Write `profiles/arm64/src/codegen.rs`**

```rust
//! UIR → AArch64 asm walker.
//!
//! Per-op emitters land here as Tasks 3-5 progress.

use crate::asm;
use crate::{Asm, FnSig, LowerError};
use compiler::{NodeKind, StdOp, Uir, UirModel};

/// Walk the entire UIR, returning the combined asm source + per-model FnSigs.
pub fn walk_uir(uir: &Uir) -> Result<Asm, LowerError> {
    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for model in &uir.models {
        let (model_asm, sig) = walk_model(model)?;
        source.push_str(&model_asm);
        source.push('\n');
        functions.push(sig);
    }

    Ok(Asm { source, functions })
}

fn walk_model(model: &UirModel) -> Result<(String, FnSig), LowerError> {
    // Validate: every Op node must be a supported op. Walk first to surface
    // errors before emitting any asm.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // Compute ABI sizes from input + output shapes.
    let input_id = *model.inputs.first().ok_or_else(|| LowerError::ShapeNotConcrete {
        span: model.source_span,
    })?;
    let input_shape = &model.nodes[input_id].ty.shape;
    let output_shape = &model.nodes[model.output].ty.shape;
    let input_floats: usize = input_shape.0.iter().product::<u64>() as usize;
    let output_floats: usize = output_shape.0.iter().product::<u64>() as usize;

    // Sum weight sizes for all Linear ops in topological (UIR-node) order.
    let mut weight_floats: usize = 0;
    for (i, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op: StdOp::Linear, operands, .. } = &node.kind {
            // Input shape of this linear is the operand's shape; output rank-2 col is N.
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            let k = in_shape.0[in_shape.0.len() - 1] as usize;
            let n = out_shape.0[out_shape.0.len() - 1] as usize;
            weight_floats += k * n;
            let _ = i; // index reserved for future weight-layout metadata
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        weight_floats,
        output_floats,
    };

    let mut body = String::new();
    body.push_str(&asm::format_function_header(&sig));
    // Body emission (matmul, relu) lands in Tasks 4 and 5.
    body.push_str(&asm::format_function_footer());

    Ok((body, sig))
}

/// Validate that an op is supported in M4a; return error otherwise.
/// Linear with `bias=true` rejected; UnsupportedOp for softmax, dropout.
fn classify_op(
    op: StdOp,
    attrs: &[compiler::OpAttr],
    span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => {
            // bias=true is not yet supported.
            for a in attrs {
                if a.name == "bias" {
                    if let compiler::AttrValue::Symbol(s) = &a.value {
                        if s == "true" {
                            return Err(LowerError::LinearWithBias { span });
                        }
                    }
                }
            }
            Ok(())
        }
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Err(LowerError::UnsupportedOp { op: "dropout".into(), span }),
        StdOp::Softmax => Err(LowerError::UnsupportedOp { op: "softmax".into(), span }),
    }
}
```

- [ ] **Step 5: Wire codegen into `lib.rs`**

Replace `lib.rs`:

```rust
//! NeuralForge arm64 scalar codegen profile (M4a).
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via [`lower`].

mod asm;
mod codegen;
mod types;

pub use types::{Asm, FnSig, LowerError};

use compiler::Uir;

/// Lower a [`Uir`] to AArch64 assembly.
///
/// Returns [`LowerError`] on any unsupported op or structural problem.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    codegen::walk_uir(uir)
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 6: Verify the new test PASSES + previous 2 still pass**

```bash
cargo test -p profiles-arm64
```

Expected: 3 tests passing.

- [ ] **Step 7: Verify build + clippy still clean**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: zero warnings, exit 0.

- [ ] **Step 8: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4a/arm64): function harness — .globl/label/ret + FnSig sizes

Walker emits one extern function per UirModel with correctly-sized
FnSig.input_floats / weight_floats / output_floats computed from
UIR shapes (per spec §7).

Function body for now: just the header + ret. Matmul body lands in
Task 4. The walker DOES validate every op upfront via classify_op
so unsupported-op errors fire before any asm is emitted.

Adds asm.rs (low-level building blocks: function header/footer
helpers, MACHO_SYM_PREFIX) and codegen.rs (walk_uir, walk_model,
classify_op).

3 unit tests: empty UIR, unsupported-op (softmax), linear function
harness (asserts symbol name, sig sizes, .globl/label/ret presence).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Linear codegen — matmul body (3 nested loops with `fmadd`)

**Goal:** Linear[N] op (no bias) emits a working matmul: 3 nested scalar loops, FMADD accumulator. Output written to `output[i*N + j]`. Function still callable but doesn't include relu yet.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/tests.rs`

**Pseudocode the asm implements:**

```
for i in 0..B:
  for j in 0..N:
    sum = 0
    for k in 0..K:
      sum += input[i*K + k] * weights[k*N + j]
    output[i*N + j] = sum
```

**Asm template** (this is what the emitter generates; placeholders `<B>`, `<K>`, `<N>` are computed from the Linear op's input shape and `out_dim` attr):

```asm
    ; matmul: input [B,K] × weights [K,N] → output [B,N]
    ; x0 = input ptr, x1 = weights ptr, x2 = output ptr (so far unmodified)
    mov     x3, #0              ; i = 0
.Lmm_i_<lid>:
    cmp     x3, #<B>
    b.ge    .Lmm_i_end_<lid>

    mov     x4, #0              ; j = 0
.Lmm_j_<lid>:
    cmp     x4, #<N>
    b.ge    .Lmm_j_end_<lid>

    fmov    s0, wzr             ; sum = 0.0
    mov     x5, #0              ; k = 0
.Lmm_k_<lid>:
    cmp     x5, #<K>
    b.ge    .Lmm_k_end_<lid>

    ; load input[i*K + k]
    lsl     x6, x3, #<log2(K)?>  ; if K is power of 2 — use lsl; else mul
    add     x6, x6, x5           ; x6 = i*K + k
    ldr     s1, [x0, x6, lsl #2] ; s1 = input[i*K + k]

    ; load weights[k*N + j]
    lsl     x7, x5, #<log2(N)?>
    add     x7, x7, x4
    ldr     s2, [x1, x7, lsl #2]

    fmadd   s0, s1, s2, s0       ; sum += s1 * s2

    add     x5, x5, #1
    b       .Lmm_k_<lid>
.Lmm_k_end_<lid>:

    ; store output[i*N + j]
    lsl     x6, x3, #<log2(N)?>
    add     x6, x6, x4
    str     s0, [x2, x6, lsl #2]

    add     x4, x4, #1
    b       .Lmm_j_<lid>
.Lmm_j_end_<lid>:

    add     x3, x3, #1
    b       .Lmm_i_<lid>
.Lmm_i_end_<lid>:
```

**Index-arithmetic implementation note:** for the M4a fixture (B=8, K=4, N=2) all dims are powers of 2, so we _could_ use `lsl` for the multiplications. But **for portability across UIR shapes**, the M4a emitter uses generic `mul`-based arithmetic that works for any positive K, N:

```asm
    ; load input[i*K + k]
    mov     x_K, #<K>            ; or materialise once outside loops via ldr literal
    mul     x6, x3, x_K          ; x6 = i*K
    add     x6, x6, x5           ; x6 = i*K + k
    ldr     s1, [x0, x6, lsl #2]
```

This is slower (`mul` vs `lsl`) but correct for any K. Performance is M5+. The `<lid>` placeholder is a per-Linear-op suffix (e.g. `0` for the first Linear in the model, `1` for the second) so labels don't collide if M4b adds multi-Linear models.

- [ ] **Step 1: Write a failing test for matmul body in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn linear_emits_matmul_loops_with_fmadd() {
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Sanity: FMADD is the matmul accumulator.
    assert!(s.contains("fmadd"), "expected fmadd in:\n{s}");
    // Three loop labels (i, j, k) for the single Linear (label suffix 0).
    assert!(s.contains(".Lmm_i_0:"), "missing i-loop label in:\n{s}");
    assert!(s.contains(".Lmm_j_0:"), "missing j-loop label in:\n{s}");
    assert!(s.contains(".Lmm_k_0:"), "missing k-loop label in:\n{s}");
    // Comparison constants come from shapes.
    assert!(s.contains("cmp     x3, #2"), "missing i-bound (B=2) in:\n{s}");
    assert!(s.contains("cmp     x4, #2"), "missing j-bound (N=2) in:\n{s}");
    assert!(s.contains("cmp     x5, #3"), "missing k-bound (K=3) in:\n{s}");
    // Sum init.
    assert!(s.contains("fmov    s0, wzr"), "missing sum init in:\n{s}");
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 linear_emits_matmul_loops_with_fmadd
```

Expected: FAIL (no `fmadd` in current asm; only header+ret).

- [ ] **Step 3: Add matmul emitter in `profiles/arm64/src/codegen.rs`**

Append helper `emit_matmul`:

```rust
/// Emit the AArch64 matmul body for one Linear op.
///
/// Per spec §7: input is row-major [B, K], weights row-major [K, N],
/// output row-major [B, N]. ABI registers: x0=input, x1=weights, x2=output.
/// `linear_idx` is a unique-per-Linear suffix used in label names so
/// multiple Linear ops in one model don't collide on labels.
fn emit_matmul(b: u64, k: u64, n: u64, linear_idx: usize) -> String {
    let mut s = String::new();
    let lid = linear_idx;

    // Materialise K, N as scratch register values for index multiplications.
    s.push_str(&format!("    ; matmul: input [{b},{k}] × weights [{k},{n}] → output [{b},{n}]\n"));

    // Outer i loop
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    // j loop
    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    // sum = 0
    s.push_str("    fmov    s0, wzr\n");

    // k loop
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    // input[i*K + k]
    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x0, x6, lsl #2]\n");

    // weights[k*N + j]
    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x1, x7, lsl #2]\n");

    // sum += s1 * s2
    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // store output[i*N + j]
    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x2, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}
```

- [ ] **Step 4: Wire `emit_matmul` into `walk_model`**

Replace `walk_model`'s body-emission block (the lines between `body.push_str(&asm::format_function_header(&sig));` and `body.push_str(&asm::format_function_footer());`) with:

```rust
    let mut body = String::new();
    body.push_str(&asm::format_function_header(&sig));

    // Emit per-op asm, in topological (UIR-node) order.
    let mut linear_idx = 0usize;
    for node in &model.nodes {
        if let NodeKind::Op { op, operands, .. } = &node.kind {
            match op {
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    // shape is [batch, k] for input and [batch, n] for output
                    if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                        return Err(LowerError::ShapeNotConcrete { span: node.source_span });
                    }
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];
                    body.push_str(&emit_matmul(b, k, n, linear_idx));
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    // Relu emission lands in Task 5.
                }
                _ => unreachable!("classify_op should have caught this"),
            }
        }
    }

    body.push_str(&asm::format_function_footer());
```

- [ ] **Step 5: Verify all tests PASS**

```bash
cargo test -p profiles-arm64
```

Expected: 4 tests passing.

- [ ] **Step 6: Verify build + clippy still clean**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs
git commit -m "feat(m4a/arm64): matmul body — 3 nested loops with fmadd

Per spec §7 pseudocode: 'output[i*N + j] = sum_k(input[i*K + k] *
weights[k*N + j])'. Implemented as scalar nested loops with FMADD
accumulator; index arithmetic via mul (works for any K, N — not
tied to powers of 2). Per-Linear label suffix (.Lmm_i_<idx>:)
prevents collisions when M4b adds multi-Linear models.

Wired into walk_model: emits matmul for every Linear op in
topological order. Relu still no-op (Task 5). Function epilogue
unchanged (just 'ret').

1 new unit test: linear_emits_matmul_loops_with_fmadd. Asserts
on fmadd presence, all 3 loop labels, all 3 cmp bounds (B=2, N=2,
K=3 from the test UIR), and 'fmov s0, wzr' sum init.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Relu codegen — separate elementwise loop

**Goal:** Lower `linear → relu` UIR. Relu emits its own elementwise loop after the matmul. Operates in-place on the producer's output buffer (which for terminal-relu is the model's `output` buffer, i.e. via `x2`).

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/tests.rs`

**Asm template:**

```asm
    ; relu: in-place clamp on output buffer (size = <total>)
    fmov    s_zero, wzr             ; once outside the loop
    mov     x9, #0                   ; element index
.Lrelu_<rid>:
    cmp     x9, #<total>
    b.ge    .Lrelu_end_<rid>
    ldr     s3, [x2, x9, lsl #2]
    fmax    s3, s3, s_zero
    str     s3, [x2, x9, lsl #2]
    add     x9, x9, #1
    b       .Lrelu_<rid>
.Lrelu_end_<rid>:
```

For M4a, relu is always terminal (per the M4a fixture), so its input buffer IS the model output (`x2`). When M4b/M5 add intermediate buffers, the emitter will need to take an explicit "operand-buffer pointer" parameter; for M4a, hardcoded to `x2`.

**Picking `s_zero`:** AArch64 has 32 single-precision FP registers `s0..s31`. We use `s0`–`s3` for matmul (sum, ldr × 2), leaving `s4`–`s31` free. Pick `s4` for the persistent zero. Document in code comment.

- [ ] **Step 1: Write a failing test in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn relu_emits_separate_loop_with_fmov_zero_and_fmax() {
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Zero materialisation outside the loop.
    assert!(s.contains("fmov    s4, wzr"), "missing 'fmov s4, wzr' (zero materialisation) in:\n{s}");
    // The relu loop body uses fmax against s4.
    assert!(s.contains("fmax    s3, s3, s4"), "missing relu fmax in:\n{s}");
    // Loop label and bound (output total = 2*2 = 4).
    assert!(s.contains(".Lrelu_0:"), "missing relu loop label in:\n{s}");
    assert!(s.contains("cmp     x9, #4"), "missing relu element-count bound in:\n{s}");
    // Relu reads + writes via x2 (output buffer).
    assert!(s.contains("ldr     s3, [x2, x9, lsl #2]"), "missing relu load in:\n{s}");
    assert!(s.contains("str     s3, [x2, x9, lsl #2]"), "missing relu store in:\n{s}");
}

#[test]
fn relu_alone_after_matmul_does_not_break_existing_test() {
    // Sanity: matmul still emitted alongside relu.
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    assert!(asm.source.contains("fmadd"));
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 relu_emits_separate_loop
```

Expected: FAIL (no relu impl yet).

- [ ] **Step 3: Add `emit_relu` in `profiles/arm64/src/codegen.rs`**

Append:

```rust
/// Emit AArch64 elementwise relu over a buffer of `total_floats` f32 elements.
///
/// Operates in-place on the buffer pointed to by `x2` (the model output buffer).
/// In M4a this is always the producer's terminal output. M4b will generalise
/// to intermediate buffers when multi-stage Linear is added.
///
/// Uses `s4` for the persistent zero; `s3` for the per-element load/store.
/// `relu_idx` is a unique-per-Relu suffix for label naming.
fn emit_relu(total_floats: u64, relu_idx: usize) -> String {
    let mut s = String::new();
    let rid = relu_idx;

    s.push_str(&format!("    ; relu: in-place clamp on output buffer ({total_floats} elements)\n"));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x2, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x2, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));

    s
}
```

- [ ] **Step 4: Wire `emit_relu` into `walk_model`**

Replace the `StdOp::Relu => { /* Relu emission lands in Task 5. */ }` arm with:

```rust
                StdOp::Relu => {
                    // Operates in-place on the producer's output buffer.
                    // For M4a (terminal-relu only) this is the model output (x2).
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    body.push_str(&emit_relu(total, relu_idx));
                    relu_idx += 1;
                }
```

And add `let mut relu_idx = 0usize;` next to the existing `let mut linear_idx = 0usize;`.

- [ ] **Step 5: Verify all tests PASS**

```bash
cargo test -p profiles-arm64
```

Expected: 6 tests passing.

- [ ] **Step 6: Verify build + clippy clean**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs
git commit -m "feat(m4a/arm64): relu — separate elementwise loop with fmov+fmax

Per spec §6 (corrected from the user review): AArch64 fmax requires
both operands to be FP registers — wzr (integer zero) cannot pass
directly. Pattern is fmov s4, wzr once outside the loop, then
fmax s3, s3, s4 inside. Operates in-place on output buffer (x2).

Per-Relu label suffix (.Lrelu_<idx>:) mirrors the matmul pattern
to prepare for M4b's multi-Relu cases.

2 new unit tests: relu_emits_separate_loop_with_fmov_zero_and_fmax
covers the asm shape end-to-end; relu_alone_after_matmul... is a
sanity assert that the preceding matmul is still emitted.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Error variants — `LinearWithBias`, `UnsupportedOp` (softmax/dropout), `DuplicateModelName`

**Goal:** Each error path has a unit test; existing tests still pass.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add failing tests in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn linear_with_bias_returns_lower_error() {
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::LinearWithBias { .. }));
}

#[test]
fn dropout_returns_unsupported_op() {
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> linear[3] -> dropout[rate=0.2]\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "dropout"));
}

#[test]
fn softmax_returns_unsupported_op() {
    // softmax-only path
    let uir = build_uir("model M [b=2]: x: Tensor[b, 3]\n    x -> softmax\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "softmax"));
}

#[test]
fn duplicate_model_name_returns_error() {
    // Two models named "M" in one source.
    let src = "model M [b=2]: x: Tensor[b, 3]\n    x -> linear[2]\n\
               model M [b=2]: y: Tensor[b, 3]\n    y -> linear[2]\n";
    let uir = build_uir(src);
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::DuplicateModelName { ref name, .. } if name == "M"));
}
```

- [ ] **Step 2: Verify FAIL counts**

```bash
cargo test -p profiles-arm64 2>&1 | grep -E "FAILED|passed"
```

Expected: 4 FAILs (the 4 new tests). Existing 6 still pass.

The `linear_with_bias_returns_lower_error`, `dropout_returns_unsupported_op`, `softmax_returns_unsupported_op` tests should already PASS — Tasks 2-3 wired classify_op to handle these. Verify by running each individually:

```bash
cargo test -p profiles-arm64 linear_with_bias_returns_lower_error
cargo test -p profiles-arm64 dropout_returns_unsupported_op
cargo test -p profiles-arm64 softmax_returns_unsupported_op
```

If any fail, classify_op needs adjustment. If they all pass, the only real new test is `duplicate_model_name_returns_error`.

- [ ] **Step 3: Implement duplicate-model-name detection in `walk_uir`**

In `profiles/arm64/src/codegen.rs`, modify `walk_uir`:

```rust
pub fn walk_uir(uir: &Uir) -> Result<Asm, LowerError> {
    // First pass: detect duplicate model names.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for model in &uir.models {
        if !seen.insert(model.name.as_str()) {
            return Err(LowerError::DuplicateModelName {
                name: model.name.clone(),
                span: model.source_span,
            });
        }
    }

    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for model in &uir.models {
        let (model_asm, sig) = walk_model(model)?;
        source.push_str(&model_asm);
        source.push('\n');
        functions.push(sig);
    }

    Ok(Asm { source, functions })
}
```

- [ ] **Step 4: Verify all tests PASS**

```bash
cargo test -p profiles-arm64
```

Expected: 10 tests passing.

- [ ] **Step 5: Verify build + clippy clean**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs
git commit -m "feat(m4a/arm64): error coverage — bias, softmax, dropout, dup-name

Three error paths were already covered by classify_op (Task 3) but
hadn't been exercised by tests:
- LinearWithBias for linear[N, bias=true]
- UnsupportedOp { op: 'softmax' }
- UnsupportedOp { op: 'dropout' }

DuplicateModelName is new: walk_uir does a first pass over models
checking for name collisions before emitting anything. Per spec §15,
this check moves up to compiler::ir::build in M4b; for M4a it lives
in the lowerer.

4 new unit tests, one per error path. cargo test -p profiles-arm64
now at 10 passing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: New M4a fixture + UIR-build test

**Goal:** Add `tests/fixtures/m4_linear_relu.nfl` and a UIR-build test that mirrors the M3b style.

**Files:**
- Create: `tests/fixtures/m4_linear_relu.nfl`
- Modify: `compiler/tests/uir_fixtures.rs`

- [ ] **Step 1: Create the fixture**

`tests/fixtures/m4_linear_relu.nfl`:

```nfl
# M4a fixture — minimal lowerable model (no softmax, no bias).
# All 5 M3 positive fixtures end in softmax (M4b territory).
# This one exercises the M4a end-to-end path: linear + relu only.

model M4Demo [batch=8]:
    x: Tensor[batch, 4]

    x -> linear[2] -> relu
```

- [ ] **Step 2: Add the UIR-build test in `compiler/tests/uir_fixtures.rs`**

Append a new submodule (mirror the existing `mod tiny_mlp { ... }` style):

```rust
mod m4_linear_relu {
    use compiler::*;

    #[test]
    fn m4_linear_relu_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/m4_linear_relu.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 1);
        let m = &uir.models[0];
        assert_eq!(m.name, "M4Demo");

        // 1 input + 2 ops (linear, relu) = 3 nodes.
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 2);

        // Input shape: Tensor[8, 4] (batch=8, hidden=4).
        assert_eq!(m.nodes[0].ty.shape.0, vec![8, 4]);
        // Linear output: Tensor[8, 2].
        assert_eq!(m.nodes[1].ty.shape.0, vec![8, 2]);
        // Relu preserves shape.
        assert_eq!(m.nodes[2].ty.shape.0, vec![8, 2]);

        // Linear has no bias attr.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "out_dim");
        assert_eq!(attrs[0].value, AttrValue::Integer(2));
    }
}
```

- [ ] **Step 3: Verify the test passes**

```bash
cargo test --workspace m4_linear_relu_builds
```

Expected: 1 test passing.

- [ ] **Step 4: Run full test suite**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: baseline + 11 (10 unit tests in profiles/arm64 + 1 UIR-build for m4_linear_relu).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/m4_linear_relu.nfl compiler/tests/uir_fixtures.rs
git commit -m "test(m4a/fixture): add m4_linear_relu.nfl + UIR-build test

The 5 M3 positive fixtures all terminate in softmax, which M4a
doesn't yet lower. This minimal fixture exercises the M4a-supported
op set (linear without bias + relu) so the integration test in
Task 9 has something to compile end-to-end.

UIR-build test mirrors the M3b per-fixture submodule style in
compiler/tests/uir_fixtures.rs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: CLI `nflc compile` subcommand

**Goal:** Add `nflc compile <file.nfl> --profile <name> [-o <output.s>]` to the CLI. Routes through `profiles_arm64::lower`. All errors render via the existing `render_error_with_snippet` helper.

**Files:**
- Modify: `nflc/src/main.rs`

- [ ] **Step 1: Update CLI dispatch in `nflc/src/main.rs`**

Find the `match args.as_slice()` block in `main()` and add new arms for `compile`:

```rust
        [cmd, path, p_flag, p_name] if cmd == "compile" && p_flag == "--profile" => {
            run_compile(PathBuf::from(path), p_name.clone(), None)
        }
        [cmd, path, p_flag, p_name, o_flag, o_path]
            if cmd == "compile" && p_flag == "--profile" && o_flag == "-o" =>
        {
            run_compile(PathBuf::from(path), p_name.clone(), Some(PathBuf::from(o_path)))
        }
        [cmd] if cmd == "compile" => {
            eprintln!("error: 'compile' requires a file path and --profile");
            print_usage();
            ExitCode::FAILURE
        }
```

(Place these arms before the catch-all `_ =>`.)

- [ ] **Step 2: Update `print_usage` banner**

Replace the existing `print_usage` body with:

```rust
fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
}
```

- [ ] **Step 3: Add `run_compile` function**

Append:

```rust
fn run_compile(path: PathBuf, profile: String, out_path: Option<PathBuf>) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ast = match compiler::parse(&source) {
        Ok(a) => a,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
            return ExitCode::FAILURE;
        }
    };

    let uir = match compiler::ir::build(&ast) {
        Ok(u) => u,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
            return ExitCode::FAILURE;
        }
    };

    if profile != "arm64" {
        eprintln!("error: unknown profile '{}' (supported: arm64)", profile);
        return ExitCode::FAILURE;
    }

    match profiles_arm64::lower(&uir) {
        Ok(asm) => {
            match out_path {
                Some(p) => match std::fs::write(&p, &asm.source) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("error: cannot write {}: {}", p.display(), e);
                        ExitCode::FAILURE
                    }
                },
                None => {
                    print!("{}", asm.source);
                    ExitCode::SUCCESS
                }
            }
        }
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(&source, &path, span.line, span.col, &format!("{}", e));
            ExitCode::FAILURE
        }
    }
}
```

- [ ] **Step 4: Build + smoke test**

```bash
cargo build --workspace
```

Expected: zero warnings.

Smoke positive (will fail at lower step until Task 7's fixture exists; should fail with `parse error` on a non-existent file, or `LowerError::UnsupportedOp` on `tiny_mlp.nfl`):

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64
```

Expected: prints the asm to stdout, exit 0.

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/tiny_mlp.nfl --profile arm64
```

Expected: snippet error mentioning softmax, exit 1.

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile xyz
echo "exit code: $?"
```

Expected: "error: unknown profile 'xyz' (supported: arm64)", exit 1.

- [ ] **Step 5: Verify clippy clean**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 6: Verify all tests still pass**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: same as Task 7's count.

- [ ] **Step 7: Commit**

```bash
git add nflc/src/main.rs
git commit -m "feat(m4a/cli): nflc compile <file> --profile arm64 [-o <path>]

Per spec §8: new subcommand wiring compiler::parse → ir::build →
profiles_arm64::lower, with all errors routed through the existing
render_error_with_snippet helper from M3c.

--profile is required and rejected for any value other than 'arm64'
('explicit over implicit' per the brainstorming decision). When
M5/M6 add new profiles, the match arm extends.

-o defaults to stdout. Output is the raw asm.source from
profiles_arm64::Asm.

USAGE banner updated to include the new subcommand and to drop
the milestone-version annotation (less churn between milestones).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Integration test — end-to-end FFI via `cc` + `libloading`

**Goal:** Build the M4a fixture's UIR, lower to asm, assemble + link with `cc -shared` to a `.dylib`, dlopen + call via FFI from Rust, compare against pure-Rust reference.

**Files:**
- Create: `profiles/arm64/tests/integration.rs`, `profiles/arm64/tests/common/mod.rs`
- Modify: `profiles/arm64/Cargo.toml` (add `libloading` dev-dep)

- [ ] **Step 1: Add `libloading` as dev-dep in `profiles/arm64/Cargo.toml`**

Append:

```toml

[dev-dependencies]
libloading = "0.8"
```

- [ ] **Step 2: Write `profiles/arm64/tests/common/mod.rs`**

```rust
//! Shared helpers for arm64 integration tests.

use std::path::PathBuf;

/// Returns true if `cc` is on PATH and runs.
pub fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Assemble + link `asm_source` into a `.dylib` and return its path.
///
/// Tempdir under `std::env::temp_dir()/nflc-test-<pid>/` (left after
/// the test runs; OS or `tmpwatch` reclaims it eventually).
pub fn compile_to_dylib(asm_source: &str, name: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-test-{pid}"));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
        panic!("cannot create test tempdir {}: {e}", dir.display())
    });

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source).unwrap_or_else(|e| {
        panic!("cannot write {}: {e}", s_path.display())
    });

    let dylib_path = dir.join(format!("lib{name}.dylib"));
    let status = std::process::Command::new("cc")
        .args(["-shared", "-arch", "arm64", "-o"])
        .arg(&dylib_path)
        .arg(&s_path)
        .status()
        .expect("cc invocation failed");
    assert!(
        status.success(),
        "cc failed to assemble {} (status: {status})",
        s_path.display()
    );

    dylib_path
}
```

- [ ] **Step 3: Write `profiles/arm64/tests/integration.rs`**

```rust
//! M4a end-to-end integration test.

mod common;

#[test]
fn tinymlp_no_softmax_runs_correctly() {
    // Pre-flight gates.
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: integration test requires aarch64 host");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    // 1. Read fixture (path is relative to the integration-test crate root,
    //    which is profiles/arm64/. So the fixture is two dirs up.)
    let src = std::fs::read_to_string("../../tests/fixtures/m4_linear_relu.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");

    // 2. Lower.
    let asm = profiles_arm64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1, "one function expected");
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.input_floats, 32);   // 8*4
    assert_eq!(sig.weight_floats, 8);   // 4*2
    assert_eq!(sig.output_floats, 16);  // 8*2

    // 3. Assemble + link.
    let dylib_path = common::compile_to_dylib(&asm.source, "m4_linear_relu");

    // 4. dlopen + call via FFI.
    let lib = unsafe { libloading::Library::new(&dylib_path) }
        .expect("libloading: open dylib");
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_M4Demo") }
        .expect("dlsym: nfl_forward_M4Demo not found");

    // Deterministic test inputs.
    let mut input = [0.0f32; 32];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5; // mix of negatives + positives so relu has work
    }
    let mut weights = [0.0f32; 8];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16];

    unsafe { forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr()); }

    // 5. Compare against pure-Rust reference.
    let expected = reference_linear_relu(&input, &weights);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: asm got {a}, reference got {b}, diff {}",
            (a - b).abs()
        );
    }
}

/// Reference: matmul (input [B,K] × weights [K,N]) followed by elementwise relu.
/// Mirrors the asm spec exactly. B=8, K=4, N=2 hardcoded for the M4a fixture.
fn reference_linear_relu(input: &[f32; 32], weights: &[f32; 8]) -> [f32; 16] {
    const B: usize = 8;
    const K: usize = 4;
    const N: usize = 2;
    let mut out = [0.0f32; 16];
    for i in 0..B {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += input[i * K + k] * weights[k * N + j];
            }
            out[i * N + j] = sum.max(0.0);
        }
    }
    out
}
```

- [ ] **Step 4: Run the integration test**

```bash
cargo test -p profiles-arm64 --test integration
```

Expected on aarch64 Mac with cc: 1 test passing. On other hosts: 1 test passing with skip-message in stderr.

If it fails on the FP comparison, see spec §15 about FMA divergence — switching the reference to `f32::mul_add` is the documented workaround.

- [ ] **Step 5: Verify all tests + clippy clean**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
git add profiles/arm64/Cargo.toml profiles/arm64/tests/
git commit -m "test(m4a/arm64): end-to-end integration via cc + libloading

Per spec §9.2: builds M4a fixture UIR, lowers to asm, assembles +
links with cc -shared into a .dylib, dlopens via libloading
(dev-dep, justified per spec §11), calls nfl_forward_M4Demo via
FFI with deterministic input/weights, compares against a pure-Rust
matmul+relu reference.

Pre-flight gates: skip with logged reason on non-aarch64 hosts or
when cc is missing. Unit tests don't depend on cc — they run
anywhere cargo runs.

Reference function reference_linear_relu() is the executable spec
of what the asm must compute. Drift triggers test failure.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: `docs/profile_guide/arm64.md`

**Goal:** Profile-guide doc explaining the arm64 codegen for users and future contributors.

**Files:**
- Create: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Write `docs/profile_guide/arm64.md`**

```markdown
# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M4a complete (NFL v0.1). Lowers `linear[N]` (no bias) and `relu`
> to native AArch64 Mach-O assembly.
> **Authoritative source:** `profiles/arm64/src/` and the M4a spec under
> `docs/superpowers/specs/`.

The `arm64` profile is the first concrete codegen profile in NeuralForge. It
takes a `compiler::Uir` and emits AArch64 assembly (Mach-O syntax) callable as a
C function. M4a's scope is intentionally small — single-Linear models with
optional ReLU — so the rest of the pipeline (`nflc compile` CLI, integration
tests, FFI) is exercised end-to-end without getting blocked on transcendental
functions like `softmax`'s `exp`.

---

## 1. Calling convention (ABI)

For each `UirModel` in the input UIR, the profile emits one `extern "C"` function:

```c
void nfl_forward_<ModelName>(
    const float* input,
    const float* weights,
    float*       output
);
```

Standard AAPCS64: pointers in `x0`, `x1`, `x2`. Pure leaf function — no callee-saved registers touched, no stack frame, no calls into libc.

The symbol name in the asm is `_nfl_forward_<ModelName>` (Mach-O underscore prefix). C / FFI callers pass the underscore-less name to `dlsym`; the dynamic loader handles the prefix.

---

## 2. Buffer layout

All buffers are `f32`, row-major.

For an `input → linear[N] → relu` model where `input: Tensor[B, K]`:

| Buffer    | Size (f32 elements) | Layout                                                      |
|-----------|---------------------|-------------------------------------------------------------|
| `input`   | B × K               | `input[i * K + k]` for row i, column k.                     |
| `weights` | K × N               | `weights[k * N + j]` for row k, column j.                   |
| `output`  | B × N               | `output[i * N + j]` for row i, column j.                    |

Sizes are reported on the returned `FnSig` (`input_floats`, `weight_floats`, `output_floats`). The caller must allocate exactly these sizes. M4a does not perform any bounds checking — passing undersized buffers is undefined behaviour.

For models with multiple Linear ops (M4b+), `weights` is the **packed concatenation** of all weight matrices in UIR-node (topological) order. M4b adds `FnSig.weights_layout: Vec<WeightSlot>` so callers know each matrix's offset and size.

---

## 3. Supported ops in M4a

| StdOp                      | Supported | Notes                                                          |
|----------------------------|-----------|----------------------------------------------------------------|
| `Linear` (no `bias` attr)  | ✅        | Pure matmul. No bias add.                                     |
| `Linear` (`bias=true`)     | ❌ M4b    | Returns `LowerError::LinearWithBias`.                         |
| `Relu`                     | ✅        | Separate elementwise loop. Operates in-place on output buffer. |
| `Dropout`                  | ❌ M4b    | Returns `LowerError::UnsupportedOp { op: "dropout" }`.        |
| `Softmax`                  | ❌ M4b    | Returns `LowerError::UnsupportedOp { op: "softmax" }`.        |
| `Input`                    | ✅        | Marker only — maps to the input pointer.                      |

### Codegen-decision: `linear[N]` without `bias` attribute

Interpreted as **pure matmul, no bias add**. The NFL grammar marks `bias` as optional but doesn't commit a default. The arm64 profile treats absence of the `bias` attribute as `bias=false`. To get bias-add explicitly, write `linear[N, bias=true]` — which M4a rejects with `LowerError::LinearWithBias` and M4b implements.

---

## 4. Code-gen patterns

### 4.1 Matmul (Linear)

Three nested scalar loops. For `linear[N]` over input shape `[B, K]`:

```asm
    mov     x3, #0              ; i = 0
.Lmm_i_<idx>:
    cmp     x3, #B
    b.ge    .Lmm_i_end_<idx>

    mov     x4, #0              ; j = 0
.Lmm_j_<idx>:
    cmp     x4, #N
    b.ge    .Lmm_j_end_<idx>

    fmov    s0, wzr             ; sum = 0.0
    mov     x5, #0              ; k = 0
.Lmm_k_<idx>:
    cmp     x5, #K
    b.ge    .Lmm_k_end_<idx>

    mov     x8, #K              ; load input[i*K + k]
    mul     x6, x3, x8
    add     x6, x6, x5
    ldr     s1, [x0, x6, lsl #2]

    mov     x8, #N              ; load weights[k*N + j]
    mul     x7, x5, x8
    add     x7, x7, x4
    ldr     s2, [x1, x7, lsl #2]

    fmadd   s0, s1, s2, s0      ; sum += input * weight (single-rounding FMA)
    add     x5, x5, #1
    b       .Lmm_k_<idx>
.Lmm_k_end_<idx>:

    mov     x8, #N              ; store output[i*N + j]
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x2, x6, lsl #2]

    add     x4, x4, #1
    b       .Lmm_j_<idx>
.Lmm_j_end_<idx>:
    add     x3, x3, #1
    b       .Lmm_i_<idx>
.Lmm_i_end_<idx>:
```

`<idx>` is a per-Linear-op suffix so labels don't collide when M4b adds multi-Linear models.

Index arithmetic uses `mul` (not `lsl`) so the emitter is correct for any K, N — not tied to powers of 2. Performance is M5+ territory.

### 4.2 Relu

Separate elementwise loop. Operates in-place on the output buffer (M4a always
has Relu as the terminal op):

```asm
    fmov    s4, wzr             ; materialise 0.0 once outside the loop
                                ; (wzr is integer; AArch64 fmax requires both
                                ; operands in FP regs, so we can't pass wzr
                                ; directly to fmax)
    mov     x9, #0              ; element index
.Lrelu_<idx>:
    cmp     x9, #<total>        ; total = B*N for terminal-relu after linear[N]
    b.ge    .Lrelu_end_<idx>
    ldr     s3, [x2, x9, lsl #2]
    fmax    s3, s3, s4
    str     s3, [x2, x9, lsl #2]
    add     x9, x9, #1
    b       .Lrelu_<idx>
.Lrelu_end_<idx>:
```

When M4b adds multi-stage models (e.g. `linear → relu → linear`), `emit_relu`
will need an explicit "operand-buffer pointer" parameter so it can clamp
intermediate buffers, not just `x2`.

### 4.3 Function frame

Pure leaf function: just label + body + `ret`. No prologue, no epilogue, no stack frame, no callee-saved register handling.

```asm
.globl _nfl_forward_<ModelName>
.p2align 2
_nfl_forward_<ModelName>:
    ; <matmul + relu body>
    ret
```

---

## 5. Errors

`profiles_arm64::lower` returns `Result<Asm, LowerError>`. `LowerError` is
`#[non_exhaustive]`; consumers must keep a `_ => ...` arm. Variants in M4a:

| Variant                      | When                                                                     |
|------------------------------|--------------------------------------------------------------------------|
| `UnsupportedOp { op, span }` | Op isn't supported in the current slice (currently `softmax`, `dropout`). |
| `LinearWithBias { span }`    | `linear[N, bias=true]` — M4b adds support.                              |
| `ShapeNotConcrete { span }`  | Defensive: shape wasn't fully resolved by `ir::build`. Should be unreachable. |
| `DuplicateModelName { name, span }` | Two `UirModel`s share `name` — would produce conflicting symbols. M4b moves this check up to `ir::build`. |

The CLI (`nflc compile`) renders these via the existing `render_error_with_snippet` helper from M3c — same `error: ... --> file:line:col ... ^` format as parser/IR errors.

---

## 6. Adding a new op

To add an op to the `arm64` profile (e.g. `tanh`, `sigmoid`):

1. Add an arm in `profiles/arm64/src/codegen.rs::classify_op` returning `Ok(())` for the new op.
2. Add a per-op emitter, e.g. `fn emit_tanh(total_floats: u64, op_idx: usize) -> String`.
3. Add a dispatch arm in `walk_model`'s op-loop calling the new emitter.
4. Add unit tests in `profiles/arm64/src/tests.rs` asserting the asm contains the expected instructions.
5. Add an integration test if the op participates in end-to-end runnable code.
6. Update this doc's §3 table.

---

## 7. Adding a new architecture profile

To add a new profile (e.g. `x86_64`, `riscv64`):

1. Create `profiles/<arch>/Cargo.toml` mirroring `profiles/arm64/Cargo.toml`. `[dependencies] compiler = { path = "../../compiler" }`.
2. Add `"profiles/<arch>"` to the workspace `members` in `/Cargo.toml`.
3. Implement the same public surface as `profiles_arm64` — `pub fn lower(&Uir) -> Result<Asm, LowerError>` plus the `Asm`, `FnSig`, `LowerError` types. (M5+ may extract a shared `profile-api` crate when the second profile lands; for M4a that's premature.)
4. Add a dispatch arm in `nflc/src/main.rs::run_compile` for the new `--profile <arch>` value.
5. Mirror this guide as `docs/profile_guide/<arch>.md`.

---

## 8. Limitations (M4a)

Items deferred to M4b/c:

- No bias-add in `linear`.
- No `softmax`. Needs `exp()`; deferred to M4b (Taylor series or `expf` symbol).
- No `dropout`. Semantically identity at inference, but bundled to M4b.
- No multi-output models. Implicit-output convention (one output per model).
- No SIMD. Scalar instructions only. NEON / SVE are M5+ work.
- No optimisation passes. Three-nested-loop matmul; `mul` for indexing; per-element load/store. Performance is M5+.
- No CI configuration.
- Integration test runs only on aarch64 hosts with `cc` available; skips with logged reason elsewhere.
```

- [ ] **Step 2: Verify the doc renders + line count is reasonable**

```bash
wc -l docs/profile_guide/arm64.md
```

Expected: between 200 and 280 lines.

```bash
head -3 docs/profile_guide/arm64.md
```

Expected: title + status line.

- [ ] **Step 3: Commit**

```bash
git add docs/profile_guide/arm64.md
git commit -m "docs(m4a): arm64 profile guide

ABI, buffer layout, supported ops in M4a, codegen-decision for
linear[N]-without-bias, asm patterns for matmul + relu + function
frame, error variants, recipes for adding new ops and new arch
profiles, M4a limitations.

Mirrors the structure of M3c's docs/language_reference/uir.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Other doc updates (`uir.md`, `PROJECT_SPEC.md`)

**Goal:** Cross-link from `uir.md` to the new arm64 guide; update `PROJECT_SPEC.md` to reflect the M4 rename.

**Files:**
- Modify: `docs/language_reference/uir.md`, `PROJECT_SPEC.md`

- [ ] **Step 1: Add cross-link note in `docs/language_reference/uir.md`**

Find the paragraph in `## 4. Stdlib operations (v0.1)` near the `Linear` row, or in the `## 5. Implicit semantics` section. Append (after the existing table or as a new paragraph at the end of §4):

```markdown
### Codegen interpretation of optional attributes

NFL grammar marks some op arguments as optional (e.g. `Linear`'s `bias`).
Default behaviour is **codegen-profile-specific**: profiles document how they
treat absent optional attributes. The current arm64 profile (M4a) interprets
`linear[N]` without an explicit `bias` attribute as **no bias add** (pure
matmul). To get bias, write `linear[N, bias=true]` explicitly. See
[`docs/profile_guide/arm64.md`](../profile_guide/arm64.md) for details.
```

- [ ] **Step 2: Update `PROJECT_SPEC.md` milestones table**

Find the milestones table (around line 145-155):

```markdown
| 4 | `generic` profile                              | Generate scalar assembly for a matrix multiply    |
```

Replace with:

```markdown
| 4 | `arm64` profile                                | Generate scalar AArch64 assembly for `linear` + `relu` (host: Apple Silicon) |
```

- [ ] **Step 3: Update `PROJECT_SPEC.md` Architecture Profiles section**

Find the "Architecture Profiles" table (around line 65-71):

```markdown
| Profile     | Architecture       | Key capability              |
|-------------|--------------------|------------------------------|
| `generic`   | Any POSIX system   | Scalar fallback, no SIMD    |
```

Replace the `generic` row with:

```markdown
| `arm64`     | Apple Silicon / AArch64 POSIX | Scalar AArch64 assembly, no SIMD |
```

(Keep the `x86_64`, `arm64`-original, `riscv64` rows below if they exist — but the `arm64` row should now be the entry-point profile, not a future placeholder. If a duplicate `arm64` row exists from the original PROJECT_SPEC, merge it into this one.)

After the change, also append a note to the section explaining the rename:

```markdown
> Note: M4 was originally specced as a `generic` profile (LLVM IR or similar
> portable IR). During M4 brainstorming this was reframed: "generic" survives as
> the architectural _principle_ (profile isolation, swap-in profiles per target),
> not as a profile name. The first concrete profile is `arm64`, matching the
> host architecture for native execution.
```

- [ ] **Step 4: Verify both files are well-formed**

```bash
head -3 docs/language_reference/uir.md
wc -l docs/language_reference/uir.md
grep -A1 "Codegen interpretation" docs/language_reference/uir.md
grep "arm64" PROJECT_SPEC.md | head -5
```

- [ ] **Step 5: Commit**

```bash
git add docs/language_reference/uir.md PROJECT_SPEC.md
git commit -m "docs(m4a): cross-link uir.md → arm64 profile guide; update PROJECT_SPEC

uir.md: new subsection 'Codegen interpretation of optional attributes'
clarifies that absent optional attrs (like Linear's bias) are
codegen-profile-specific, with the arm64 profile treating no-bias
as pure matmul. Cross-links to docs/profile_guide/arm64.md.

PROJECT_SPEC.md:
- Milestones table M4 row: 'generic profile' → 'arm64 profile'
  with the M4a-specific scope (linear + relu, scalar AArch64).
- Architecture Profiles table: drop generic row, promote arm64
  from future placeholder to the M4 deliverable.
- Note explaining the rename: 'generic' lives as principle, not
  as profile name.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Closeout — DEVLOG, CLAUDE.md, final smoke + clippy

**Goal:** Record M4a in DEVLOG, update CLAUDE.md current-status + repo-structure, run all smoke tests one last time.

**Files:**
- Modify: `DEVLOG.md`, `CLAUDE.md`

- [ ] **Step 1: Final end-to-end verification**

```bash
cargo build --workspace
```

Expected: zero warnings.

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: exit 0.

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: baseline + 11 (10 unit tests in profiles/arm64 + 1 UIR-build test for m4_linear_relu) + 1 (integration). So baseline + 12.

If integration test was skipped (non-aarch64): baseline + 11. The skip is logged to stderr; check:

```bash
cargo test --workspace 2>&1 | grep -i skip
```

- [ ] **Step 2: CLI smoke positive (full pipeline through to .dylib)**

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m4a.s
echo "exit code: $?"
cat /tmp/m4a.s | head -10
cc -shared -arch arm64 -o /tmp/m4a.dylib /tmp/m4a.s
echo "cc exit code: $?"
nm /tmp/m4a.dylib | grep nfl_forward
```

Expected:
- `nflc` exits 0
- `/tmp/m4a.s` contains the asm header (`.globl _nfl_forward_M4Demo`, etc.)
- `cc` exits 0
- `nm` reports `_nfl_forward_M4Demo` symbol present.

- [ ] **Step 3: CLI smoke negative — softmax fixture**

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/tiny_mlp.nfl --profile arm64
echo "exit code: $?"
```

Expected: source-snippet error mentioning `softmax`, exit 1.

- [ ] **Step 4: CLI smoke — unknown profile**

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile xyz
echo "exit code: $?"
```

Expected: "error: unknown profile 'xyz' (supported: arm64)", exit 1.

If any check above fails, **do not commit** — fix first.

- [ ] **Step 5: Append M4a entry to `DEVLOG.md`**

Find the existing top entry (likely `## 2026-05-03 — Milestone 3c closed: ...` or similar) and use Edit to insert the new M4a entry above it (separated by `---`):

```markdown
---

## 2026-05-03 — Milestone 4a closed: arm64 scalar codegen — first machine-executable output

### What was done
- Workspace restructured into 3 crates: `compiler/` (lib only), `nflc/` (bin
  only), `profiles/arm64/` (lib only). Empty placeholder dirs
  `profiles/{generic,x86_64,riscv64}/` deleted. `compiler` package renamed
  from `nflc` to `compiler`. 25 mechanical `nflc::` → `compiler::` import
  rewrites across `nflc/src/main.rs`, `compiler/tests/uir_fixtures.rs`,
  `compiler/tests/fixtures.rs`.
- `profiles/arm64` lib crate. Public surface: `pub fn lower(uir: &Uir) ->
  Result<Asm, LowerError>`. Types: `Asm`, `FnSig`, `LowerError`
  (`#[non_exhaustive]`, 4 variants). Internal modules: `codegen.rs` (UIR
  walker, per-op emitters), `asm.rs` (function header/footer helpers),
  `tests.rs` (unit tests).
- Lowering covers `linear[N]` without bias (matmul: 3 nested scalar loops
  with `fmadd`), `relu` (separate elementwise loop with `fmov s4, wzr` once
  + `fmax s3, s3, s4` per element), and `Input` (marker, no code).
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
  against a pure-Rust matmul+relu reference. Pre-flight gates skip the test
  on non-aarch64 hosts or when `cc` is missing.
- New `docs/profile_guide/arm64.md` (~250 lines): ABI, buffer layout, supported
  ops, asm patterns, error variants, recipes for adding new ops and new arch
  profiles, M4a limitations.
- `docs/language_reference/uir.md` cross-links to the arm64 guide for the
  optional-attribute interpretation.
- `PROJECT_SPEC.md` milestones table M4 row updated; "Architecture Profiles"
  table loses `generic` row, gains `arm64` row as M4 deliverable.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-03-m4a-arm64-codegen-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-03-m4a-arm64-codegen.md` (12 tasks, 12 commits).

### Project principle formalised in M4a spec §11

> **Dependency policy.** Production crates (`compiler`, `nflc`,
> `profiles/arm64` lib-target) — strict **std-only**. Adding a non-std
> production dep requires a separate explicit decision and PR.
> **Dev-dependencies** are admissible by need; M4a starts the list with
> `libloading` (used only in `profiles/arm64`'s integration test).

### Problems encountered
- (Fill in real issues found during implementation. If none, write
  "None — implementation followed the plan straight through.")

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

### Next step
**Milestone 4a complete.** First time NeuralForge produces real
machine-executable code: an `.s` text file → `.dylib` → callable function.

The immediate next step is **Milestone 4b — softmax + bias + dropout**:
- Lower `linear[N, bias=true]` (4-th `bias` parameter, `FnSig.weights_layout`).
- Lower `dropout` (no-op pass-through at inference).
- Lower `softmax` (scalar `exp` via Taylor series with range reduction OR
  link `expf` from libm).
- Result: all 5 M3 positive fixtures lower end-to-end.
- Move duplicate-model-name check up to `compiler::ir::build`.

Brainstorming for M4b runs in a fresh worktree once main is updated post-M4a-merge.
```

(Keep all existing entries below intact — just insert the new entry above the most recent `---` boundary.)

- [ ] **Step 6: Update `CLAUDE.md` "Current Status"**

Find the existing "Current Status" section (M3c version) and replace its body with:

```markdown
**Milestone 4a complete.** First architecture profile shipped: `profiles/arm64`
lowers `input → linear[N] → relu` UIR to native AArch64 assembly callable as
a C function. End-to-end pipeline `NFL → AST → UIR → asm → .dylib → FFI` works
on Apple Silicon. New CLI subcommand `nflc compile <file> --profile arm64`.
3-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib) with no
dependency cycles. Production code stays std-only; `libloading` is a test-only
dev-dep. ~118 tests passing across lexer, parser, IR, profile codegen, and
the FFI integration test. Both `cargo build` and
`cargo clippy --workspace --all-targets -- -D warnings` clean.
`docs/profile_guide/arm64.md` documents the profile for users and contributors.

The immediate next step is **Milestone 4b** — add `bias=true` to linear,
implement `dropout` (inference no-op) and `softmax` (scalar `exp`). After
M4b all 5 M3 positive fixtures lower end-to-end.
```

(Test count "118" is approximate — replace with the actual count from Step 1's `cargo test` run.)

- [ ] **Step 7: Update `CLAUDE.md` "Repository Structure"**

Find the section that diagrams the repo layout (currently lists `compiler/` as a single crate). Replace the relevant block to reflect 3 crates:

```markdown
NeuralForge/
├── CLAUDE.md
├── PROJECT_SPEC.md
├── DEVLOG.md
│
├── Cargo.toml              ← workspace manifest (members = ["compiler", "nflc", "profiles/arm64"])
│
├── compiler/               ← `compiler` crate (lib only)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          ← public API: `compiler::parse(&str)`, `compiler::ir::build(&NflSource)`
│   │   ├── ast.rs          ← typed AST nodes (Span on every node)
│   │   ├── lexer/          ← tokeniser + INDENT/DEDENT machine
│   │   ├── parser/         ← recursive-descent parser
│   │   └── ir/             ← UIR types, builder, stdlib
│   └── tests/              ← integration tests (positive + negative fixtures)
│
├── nflc/                   ← `nflc` crate (bin only) — CLI dispatcher
│   ├── Cargo.toml
│   └── src/main.rs         ← `nflc parse|compile ...`
│
├── profiles/
│   └── arm64/              ← `profiles-arm64` crate (lib only)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs      ← `pub fn lower(&Uir) -> Result<Asm, LowerError>`
│       │   ├── types.rs    ← Asm, FnSig, LowerError
│       │   ├── asm.rs      ← low-level asm building blocks
│       │   ├── codegen.rs  ← UIR walker + per-op emitters
│       │   └── tests.rs    ← unit tests
│       └── tests/
│           ├── integration.rs    ← end-to-end FFI test
│           └── common/mod.rs     ← cc + tempdir helpers
│
├── language/
│   ├── grammar.ebnf        ← formal NFL grammar
│   └── stdlib/             ← (placeholder — operations live in compiler/src/ir/stdlib.rs for v0.1)
│
├── tests/
│   └── fixtures/           ← sample .nfl files used in tests
│
└── docs/
    ├── language_reference/
    │   ├── grammar.md
    │   └── uir.md
    └── profile_guide/
        └── arm64.md
```

- [ ] **Step 8: Commit closeout**

```bash
git add CLAUDE.md DEVLOG.md
git status
# Confirm: only the two .md files staged.
git commit -m "chore(m4a): close Milestone 4a — arm64 scalar codegen shipped

DEVLOG entry covers all 12 tasks, formalises the dependency policy
(production crates std-only, dev-deps pragmatic), records the
project's first machine-executable output, and notes M4b as the
next step (bias, dropout, softmax → all 5 M3 fixtures lower).

CLAUDE.md Current Status updated to reflect Milestone 4a complete
and Milestone 4b as next. Repository Structure updated to show the
3-crate workspace (compiler lib, nflc bin, profiles/arm64 lib).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 12, Milestone 4a is complete by spec acceptance criteria:

1. ✅ Workspace builds clean — Tasks 1-12.
2. ✅ Clippy clean — Task 1, verified at 8, 9, 12.
3. ✅ All pre-M4a tests still pass — verified at 1, 7, 9, 12.
4. ✅ M4a unit tests pass — Tasks 3-6.
5. ✅ M4a integration test passes (or skipped on non-aarch64) — Task 9, verified at 12.
6. ✅ `nflc compile` produces a valid `.s`; `cc` assembles it; `_nfl_forward_M4Demo` symbol present — verified at 12.
7. ✅ `nflc compile` on `tiny_mlp.nfl` exits 1 with snippet-formatted softmax error — verified at 12.
8. ✅ Unknown profile rejected with clear message — Task 8, verified at 12.
9. ✅ `docs/profile_guide/arm64.md` exists — Task 10.
10. ✅ DEVLOG entry — Task 12.
11. ✅ CLAUDE.md "Current Status" + "Repository Structure" updated — Task 12.
12. ✅ PROJECT_SPEC.md M4 rename — Task 11.

**After all tasks pass:** push `claude/m4-generic-profile` and open a PR against `main`. Title suggestion: "Implement Milestone 4a: arm64 scalar codegen — first machine-executable output". After merge, M4a is closed.

**Milestone 4b entry-point:** fresh `superpowers:brainstorming` cycle once main is updated post-M4a-merge. Decisions to make: `softmax`'s `exp` (Taylor / table / `expf` libm symbol), bias-add codegen pattern, weights_layout shape.
