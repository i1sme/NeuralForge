# Milestone 4a — `profiles/arm64` scalar codegen (vertical slice 1) — Design

> **Status:** Brainstormed and approved 2026-05-03. To be implemented in the
> `claude/m4-generic-profile` worktree.
> **Source:** This spec captures the brainstorming conversation. If something
> here disagrees with what was decided in the conversation, the conversation
> wins — file an amendment.

## 1. Overview

Milestone 4 in `PROJECT_SPEC.md` was originally named "**`generic` profile**: generate scalar assembly for a matrix multiply". Brainstorming reframed it:

- The "generic" name implied an architecture-neutral fallback. In practice that can only mean LLVM IR, which is _not_ assembly. The project principle is "pure assembly for the target", so a target-specific scalar profile is the right shape.
- The host machine is Apple Silicon (`uname -m == arm64`). Native AArch64 assembly runs directly via `as` + `cc`, no Rosetta layer, no LLVM dependency.
- Therefore: rename the milestone target from **`generic`** → **`arm64`**. "Generic" survives only as the architectural _principle_ (profile isolation, swap-in profiles per target), not as a profile name.

**M4 itself** is sliced (mirroring the M3 → 3a/3b/3c pattern that worked well):

| Slice | Scope                                                                                          |
|-------|-----------------------------------------------------------------------------------------------|
| M4a (this spec) | Minimal honest end-to-end: lower `input → linear[N] → relu` to AArch64 asm. Function callable via FFI. Test runs on host. |
| M4b   | `linear` with `bias=true`, `dropout` (no-op at inference), `softmax` (scalar `exp` via series or libm). Closes coverage of all 5 M3 fixtures.       |
| M4c   | Polish: profile-guide doc updates, snapshot tests for full asm fingerprint if needed, possibly an optimisation pass for trivially fold-able sequences. Closes M4. |

This document is M4a only.

## 2. Goal

Prove the full pipeline `NFL → AST → UIR → AArch64 assembly → executable code` end-to-end with the smallest UIR that exercises real codegen (matmul + elementwise non-linearity). All tests run natively on the host Mac.

## 3. Non-goals

- Performance optimisation. Scalar code, three nested loops, accept whatever cycles fall out.
- Cross-compilation. Host arch only.
- Bias addition in `linear`. `bias=true` returns `LowerError::LinearWithBias` and is implemented in M4b.
- `softmax`, `dropout`. Both return `LowerError::UnsupportedOp` and land in M4b.
- AOT executable. Output is one `.s` file containing one `extern "C"` function per `UirModel`. Caller assembles and links it.
- Multiple architectures. Only `arm64`. M5/M6 add more.
- New NFL syntax. Reuses M3 grammar verbatim. New fixture is needed only because all existing positive fixtures terminate in `softmax`.

## 4. Workspace restructure (3 crates)

```toml
# /Cargo.toml (new root)
[workspace]
members = ["compiler", "nflc", "profiles/arm64"]
resolver = "2"
```

| Crate              | Kind     | Depends on                       | Purpose                                         |
|--------------------|----------|----------------------------------|-------------------------------------------------|
| `compiler/`        | lib only | (no path-deps)                   | UIR types, parser, `ir::build`. Profile-agnostic. |
| `nflc/`            | bin only | `compiler` + `profiles-arm64`    | CLI dispatcher (`parse` / `compile` subcommands). |
| `profiles/arm64/`  | lib only | `compiler`                       | UIR → AArch64 asm codegen.                      |

### Dependency graph

```
profiles/arm64 ──→ compiler
nflc           ──→ compiler + profiles/arm64
```

No cycles. `compiler` knows nothing about specific profiles. M5/M6 add new workspace members without touching existing crates.

### File moves / removes

- `compiler/src/main.rs` is **deleted**. Its content moves into `nflc/src/main.rs` verbatim (only the import paths shift from `nflc::*` to `compiler::*` since the lib crate is now `compiler` not `nflc`).
- `compiler/Cargo.toml`: drop `[[bin]]` section if present; keep `[lib]`.
- Empty placeholder dirs `profiles/generic/`, `profiles/x86_64/`, `profiles/riscv64/` are **deleted**. The first two were YAGNI placeholders. `generic/` carried a name we explicitly abandoned. M6 (when it ships) will add `profiles/x86_64/` for real.

### Naming note

The CLI binary's crate name is `nflc` (matches the binary name). The library crate keeps the name `compiler` (as the workspace already names it). The `nflc` binary `[package].name = "nflc"` and `[[bin]] name = "nflc"`.

## 5. `profiles/arm64` public API

The crate exposes one entry point and three types:

```rust
// profiles/arm64/src/lib.rs

use compiler::{Uir, ast::Span};

pub fn lower(uir: &Uir) -> Result<Asm, LowerError>;

pub struct Asm {
    /// Full AArch64 Mach-O assembly source. UTF-8.
    pub source: String,
    /// One entry per UirModel in the input UIR.
    pub functions: Vec<FnSig>,
}

pub struct FnSig {
    /// External symbol name. e.g. "nfl_forward_TinyMLP" (without leading `_` —
    /// callers that need the Mach-O underscore prefix add it themselves).
    pub name: String,
    /// Original UIR model name (matches `UirModel.name`).
    pub model: String,
    /// Number of f32 elements in the input buffer.
    pub input_floats: usize,
    /// Total number of f32 elements across all weight matrices, packed in
    /// UIR-node order (topological by construction). For M4a this is always
    /// the single Linear's matrix size (= in_dim × out_dim).
    pub weight_floats: usize,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
}

#[non_exhaustive]
pub enum LowerError {
    /// Op is not supported in the current M4 slice. Op string is the
    /// lowercase token (e.g. "softmax", "dropout").
    UnsupportedOp { op: String, span: Span },
    /// `linear[N, bias=true]` is not yet implemented (M4b).
    LinearWithBias { span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved. Should
    /// be unreachable if the IR builder did its job. Carries the span of
    /// the offending node for diagnostics.
    ShapeNotConcrete { span: Span },
    /// Two `UirModel`s share the same `name`, so the lowerer would emit
    /// duplicate `nfl_forward_<name>` symbols. The IR builder doesn't enforce
    /// uniqueness yet — see §15. M4b moves the check up to `ir::build`.
    DuplicateModelName { name: String, span: Span },
}
```

`#[non_exhaustive]` on `LowerError` is mandatory — variants will be removed as M4b/c add coverage; consumers must keep a `_ => ...` arm.

`Asm` and `FnSig` carry no `Display` impl in M4a (the asm source is the
display). M4b/c may add `Display for FnSig` when the multi-Linear case needs
human-readable layout dumps.

## 6. Op coverage in M4a

| StdOp                      | M4a support | Codegen sketch                                                   |
|----------------------------|-------------|------------------------------------------------------------------|
| `Linear` (no `bias` attr)  | ✅          | Pure matmul: 3 nested scalar loops, `fmadd s0, s1, s2, s0` accumulator pattern, single-precision throughout. |
| `Linear` (`bias=true`)     | ❌          | `LowerError::LinearWithBias` (M4b adds bias-add).                |
| `Relu`                     | ✅          | Separate elementwise loop applying `fmax sN, sN, wzr` per element of its input buffer (operates in-place on the producer's output buffer). No fusion with the preceding op — fusion is M5 territory. |
| `Dropout`                  | ❌          | `LowerError::UnsupportedOp` (M4b: identity-pass at inference, +1 case). |
| `Softmax`                  | ❌          | `LowerError::UnsupportedOp` (M4b: scalar `exp` via Taylor series or libm). |
| `Input`                    | ✅          | Marker only — no code emitted; maps to the input pointer parameter. |

### Codegen-decision: `linear[N]` without `bias` attribute

Interpreted as **pure matmul, no bias add**. This is a codegen-level decision (the NFL grammar doesn't commit a default for the optional `bias` argument). Documented in:
- This spec (here).
- `docs/profile_guide/arm64.md` (the new profile guide).
- A 1-2 line note in `docs/language_reference/uir.md` cross-referencing the profile guide.

If the user wants bias, they write `linear[N, bias=true]` explicitly. M4a rejects this with `LowerError::LinearWithBias`; M4b implements it.

## 7. Generated function ABI

For every `UirModel` in the input UIR, lower emits one `extern "C"` function:

```c
void nfl_forward_<ModelName>(
    const float* input,     // input_floats × f32, row-major over UIR input shape
    const float* weights,   // packed: all Linear weight matrices in UIR-node order
    float*       output     // output_floats × f32, row-major over the model's terminal-node shape
);
```

- **f32 throughout.** Matches ML convention. AArch64 has native FP32 (`fmadd`, `fmax`) on `s0..s31`.
- **Standard AAPCS64.** Pointers in `x0`, `x1`, `x2`. Pure leaf function — no callee-saved registers touched, no stack frame needed.
- **Single packed `weights` pointer.** All Linear weight matrices concatenated in UIR-node order (which is topological by builder construction). For M4a there's exactly one Linear, so layout is trivial. M4b adds `weights_layout: Vec<WeightSlot>` to `FnSig` so callers of multi-Linear models know each matrix's offset and size.
- **No bias pointer in M4a.** Added in M4b as a 4th parameter `const float* biases` when `bias=true` becomes lowerable.
- **No globals, no `.bss`.** Pure function: input → output. Caller owns all buffers.
- **Unique symbol per model.** `nfl_forward_TinyMLP`, `nfl_forward_SingleLine`, etc. UIRs with multiple models (e.g., `pipeline_styles.nfl` has 3) emit 3 distinct symbols.

For TinyMLP-without-softmax (the M4a test fixture; see §10):
- `input` = 8×4 = 32 f32, indexed `input[i*4 + k]`
- `weights` = 4×2 = 8 f32, indexed `weights[k*2 + j]`
- `output` = 8×2 = 16 f32, indexed `output[i*2 + j]`

### Symbol decoration on Mach-O

macOS Mach-O prepends `_` to C symbol names in the asm (so `nfl_forward_M4Demo` is emitted as `_nfl_forward_M4Demo` in the `.s` text). The `FnSig.name` field stores the **C-level name without the underscore** so that callers using FFI from any language can construct the platform-correct symbol themselves. `compile_to_dylib` (test helper) and `dlsym` calls use the underscore-less name (the dynamic loader handles the prefix).

## 8. CLI changes (`nflc compile`)

New subcommand in `nflc/src/main.rs`:

```
nflc compile <file.nfl> --profile <name> [-o <output.s>]
```

Behaviour:
1. Read `.nfl` file (same error path as `parse`).
2. `compiler::parse` → AST. On error: `render_error_with_snippet`, exit 1.
3. `compiler::ir::build` → UIR. On error: `render_error_with_snippet`, exit 1.
4. Match `--profile`:
   - `arm64` → `profiles_arm64::lower(&uir)`. On error: `render_error_with_snippet` using the `LowerError`'s `span`, exit 1.
   - any other value → "error: unknown profile '<name>' (supported: arm64)", exit 1. **No default** — `--profile` is required for `compile`. Explicit over implicit.
5. Success: write `asm.source` to `-o <path>` if given, else stdout. Exit 0.

Existing subcommands (`nflc parse <file>`, `... --tokens`, `... --uir`) are preserved as-is in the move from `compiler/src/main.rs` to `nflc/src/main.rs`.

Updated USAGE banner:

```
nflc — NFL Compiler

USAGE:
  nflc parse   <file.nfl>                    Parse and pretty-print the AST
  nflc parse   <file.nfl> --tokens           Print the lexer's token stream
  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR
  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly
                          [-o <file.s>]      Output path (default: stdout)
```

## 9. Test strategy

Three layers, all in `profiles/arm64/`.

### 9.1. Unit tests (`profiles/arm64/src/tests.rs`)

Plain-string assertions on generated `.s`. Never invoke `as` / `cc`. Catch regressions in code-gen patterns. Approximately 6-8 tests:

- `matmul_emits_three_nested_loops` — checks for `fmadd`, the loop-counter increments, the `ret` epilogue.
- `function_symbol_format` — verifies `.globl _nfl_forward_M` and the leading underscore (Mach-O).
- `relu_uses_fmax_against_zero` — checks the emitted `fmax sN, sN, wzr` (or equivalent) when relu follows a linear.
- `linear_with_bias_returns_unsupported` — passes a UIR containing `linear[2, bias=true]`, asserts `Err(LowerError::LinearWithBias { .. })`.
- `softmax_returns_unsupported` — same shape, for softmax.
- `dropout_returns_unsupported` — same shape, for dropout.
- `multiple_models_emit_distinct_symbols` — pipeline_styles-shaped UIR (without the softmax tail) generates `nfl_forward_M1`, `nfl_forward_M2`, `nfl_forward_M3` distinct.
- `input_node_emits_no_code` — sanity: the `input` UIR node never produces an instruction.

### 9.2. Integration test (`profiles/arm64/tests/integration.rs`)

The end-to-end test. Single test:

```rust
#[test]
fn tinymlp_no_softmax_runs_correctly() {
    // Pre-flight: skip if not on aarch64 or `cc` missing.
    if !cfg!(target_arch = "aarch64") { return; }
    if !common::cc_available() { return; }

    // 1. Build UIR
    let src = std::fs::read_to_string("../../tests/fixtures/m4_linear_relu.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();

    // 2. Lower
    let asm = profiles_arm64::lower(&uir).unwrap();
    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");

    // 3. Assemble + link
    let dylib_path = common::compile_to_dylib(&asm.source, "m4_linear_relu");

    // 4. Load + call via FFI
    let lib = unsafe { libloading::Library::new(&dylib_path).unwrap() };
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_M4Demo").unwrap() };

    let input = deterministic_input();   // [f32; 32], values like (i as f32) / 32.0
    let weights = deterministic_weights(); // [f32; 8]
    let mut output = [0.0f32; 16];
    unsafe { forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr()); }

    // 5. Compare against pure-Rust reference
    let expected = reference_linear_relu(&input, &weights);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "output[{i}]: got {a}, expected {b}");
    }
}

fn reference_linear_relu(input: &[f32; 32], weights: &[f32; 8]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..8 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += input[i * 4 + k] * weights[k * 2 + j];
            }
            out[i * 2 + j] = sum.max(0.0);
        }
    }
    out
}
```

`reference_linear_relu` is the pure-Rust spec of what the asm must compute; if they ever drift, the test catches it.

### 9.3. Test helper (`profiles/arm64/tests/common/mod.rs`)

```rust
pub fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

pub fn compile_to_dylib(asm_source: &str, name: &str) -> std::path::PathBuf {
    use std::io::Write;
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-test-{pid}"));
    std::fs::create_dir_all(&dir).unwrap();

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source).unwrap();

    let dylib_path = dir.join(format!("lib{name}.dylib"));
    let status = std::process::Command::new("cc")
        .args(["-shared", "-arch", "arm64", "-o"])
        .arg(&dylib_path)
        .arg(&s_path)
        .status()
        .expect("cc invocation failed");
    assert!(status.success(), "cc failed to assemble {name}.s");

    dylib_path
}
```

`std`-only helper. No `tempfile` crate. Tempdir uses pid + name for uniqueness; left in `/tmp` after the test (the OS or `tmpwatch` handles cleanup eventually). Acceptable for v0.1; M4c may add explicit `Drop`-based cleanup if it becomes noisy.

### 9.4. Toolchain pre-flight & CI portability

Both pre-flight checks (`cfg!(target_arch = "aarch64")` and `cc_available()`) ensure the integration test gracefully skips on non-arm64 CI runners or systems without a C toolchain. Skip is reported as test-pass with a one-line `eprintln!` so CI logs are auditable. Unit tests have no such dependency — they run anywhere `cargo test` runs.

### 9.5. What we don't test in M4a

- Performance / cycle counts. M5+.
- Snapshot-of-full-asm via `insta` or similar. Substring checks in unit tests are sufficient at this scope; introducing a snapshot framework before the asm shape stabilises would create churn.
- Cross-compilation. Host arch only.
- Multiple inputs / fuzzing. One deterministic input set per integration test; sufficient at this scope.

## 10. New fixture

`tests/fixtures/m4_linear_relu.nfl`:

```nfl
# M4a fixture — minimal lowerable model.
# All 5 M3 positive fixtures end in softmax (M4b territory).
# This one exercises the M4a end-to-end path: linear + relu only.

model M4Demo [batch=8]:
    x: Tensor[batch, 4]

    x -> linear[2] -> relu
```

This fixture also gets a UIR-build test in `compiler/tests/uir_fixtures.rs` (mirrors the pattern of M3b's per-fixture submodules), and a parse test if the parser doesn't already cover this exact pattern.

## 11. Dependency policy

> **Production crates** (`compiler`, `nflc`, the lib-target of `profiles/arm64`) — strict **std-only**. Adding a non-std dep requires a separate explicit decision and PR.
>
> **Dev-dependencies** (anything under `[dev-dependencies]` in any crate's `Cargo.toml`) — admissible by necessity. M4a starts the list with `libloading` (used only by `profiles/arm64`'s integration test for FFI dlopen). Each subsequent dev-dep should still be justified, but doesn't reopen the policy.

This split exists so a future contributor doesn't read `libloading` in `Cargo.toml` and infer "this project allows external deps in production code". Production stays lean; tests are pragmatic.

## 12. Artifacts (created / modified / deleted)

### Created

| Path | Purpose |
|---|---|
| `Cargo.toml` (root) | Workspace manifest |
| `nflc/Cargo.toml` | Bin-crate manifest |
| `nflc/src/main.rs` | CLI (moved verbatim from `compiler/src/main.rs`, imports adjusted) |
| `profiles/arm64/Cargo.toml` | Lib-crate manifest. `[dependencies] compiler = { path = "../../compiler" }`. `[dev-dependencies] libloading = "0.8"`. |
| `profiles/arm64/src/lib.rs` | `pub use` re-exports + entry `pub fn lower` |
| `profiles/arm64/src/types.rs` | `Asm`, `FnSig`, `LowerError` |
| `profiles/arm64/src/codegen.rs` | UIR-walker emitting asm |
| `profiles/arm64/src/asm.rs` | Low-level asm primitives (FP register names, FMADD formatting, symbol-naming helpers) |
| `profiles/arm64/src/tests.rs` | Unit tests |
| `profiles/arm64/tests/integration.rs` | End-to-end FFI test |
| `profiles/arm64/tests/common/mod.rs` | `cc_available` + `compile_to_dylib` helpers |
| `tests/fixtures/m4_linear_relu.nfl` | M4a fixture |
| `docs/profile_guide/arm64.md` | Profile-guide doc (~150-200 lines): ABI, op coverage, weight layout, how to add an op |

### Modified

| Path | Change |
|---|---|
| `compiler/Cargo.toml` | Drop `[[bin]]` if present; lib only |
| `compiler/src/main.rs` | **Deleted** (moved to `nflc/`) |
| `compiler/tests/uir_fixtures.rs` | New test module for `m4_linear_relu` fixture |
| `docs/language_reference/uir.md` | 1-2 lines noting `linear[N]`-without-bias is a codegen-decision; cross-link to `arm64.md` |
| `PROJECT_SPEC.md` | Update milestones table (M4 = `arm64` profile, not `generic`); update "Architecture Profiles" section to drop the `generic` row, add `arm64` row |
| `CLAUDE.md` | "Current Status" → M4a complete; "Repository Structure" → reflect 3-crate workspace |
| `DEVLOG.md` | M4a closeout entry |

### Deleted

| Path | Reason |
|---|---|
| `profiles/generic/` | Name abandoned; concept lives in architecture, not as a profile |
| `profiles/x86_64/` | YAGNI placeholder; M6 will create it for real |
| `profiles/riscv64/` | YAGNI placeholder; future milestone will create it |

## 13. Acceptance criteria

1. Workspace has 3 members; `cargo build` from root builds all 3 with zero warnings.
2. `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
3. `cargo test --workspace` passes:
   - All 106 existing tests (unchanged).
   - The new fixture's UIR-build test in `compiler/tests/uir_fixtures.rs`.
   - All M4a unit tests in `profiles/arm64`.
   - The M4a integration test (or skipped with a logged reason on non-aarch64 hosts).
4. `nflc compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/out.s` produces a `.s` file. `cc -shared -arch arm64 -o /tmp/out.dylib /tmp/out.s` succeeds. The resulting dylib contains the `_nfl_forward_M4Demo` symbol.
5. `nflc compile tests/fixtures/tiny_mlp.nfl --profile arm64` exits 1 with `LowerError::UnsupportedOp { op: "softmax", ... }` rendered via the existing source-snippet error formatter.
6. `nflc compile <any.nfl> --profile <unknown>` exits 1 with a clear "unknown profile" message.
7. `docs/profile_guide/arm64.md` exists and documents: ABI, weight layout, supported ops in M4a, where to add a new op (in `codegen.rs`), where to add a new arch profile (mirror `profiles/arm64/`).
8. `DEVLOG.md` has the M4a close-out entry. `CLAUDE.md` "Current Status" is updated. `PROJECT_SPEC.md` reflects the M4 rename.

## 14. Vertical slicing — what comes after

| Slice | Content |
|---|---|
| **M4a** (this spec) | Workspace split + `linear[N]` (no bias) + `relu` + new fixture + integration test. |
| **M4b** | `linear[N, bias=true]` (4-th `bias` parameter, `FnSig.weights_layout` populated). `dropout` (no-op pass-through). `softmax` (scalar `exp` — Taylor series of degree 6-8 with range reduction OR libm `expf` symbol if linker accepts). All 5 M3 fixtures lower end-to-end. |
| **M4c** | Polish: full-asm snapshot tests if desired (`insta` as a dev-dep, justified), profile-guide doc finalisation, possibly a tiny optimisation pass (e.g. coalesce two consecutive `relu` into one — unlikely to matter, but a place to test the optimisation harness shape that M5 will use heavily). Closes M4. |

## 15. Open questions / risks

- **Mach-O calling convention edge cases.** Apple AArch64 is mostly AAPCS64 but with quirks (variadic functions, frame pointer requirements). M4a uses pure leaf functions with non-variadic signatures, so the standard register passing applies and we shouldn't need to set up a frame pointer. If `cc` complains during integration test, fall back to standard prologue (`stp x29, x30, [sp, #-16]!`) at the cost of a few extra instructions per function. Document the choice in `arm64.md`.
- **`f32` precision.** Reference Rust uses host `f32`; generated asm uses `s` registers (single-precision). They should match bit-exactly given identical operand order and the same FMA semantics. The integration test uses `1e-5` epsilon as a guard against any unexpected reordering — but in practice the test should pass with `0.0` epsilon. If not, investigate.
- **Symbol name collisions.** Two NFL models with the same name in the same source would produce duplicate `nfl_forward_X` symbols. The current IR builder doesn't enforce model-name uniqueness. M4a's lowerer detects this case and returns `LowerError::DuplicateModelName { name, span }` (added to the enum in §5). The proper fix — moving the check up to `compiler::ir::build` so it fails at IR-build time before any profile sees the UIR — is M4b's responsibility.
- **`cc` invocation portability.** macOS `cc` is clang. Linux `cc` is usually gcc but symlinked. Both should accept `-arch arm64` on a Mac, and `-x assembler` is universal. If we ever target Linux on aarch64, we may need to drop `-arch arm64` (Linux cc doesn't take it). M4a tests skip on non-Mac arm64, so this doesn't bite yet.

## 16. Out of M4a (explicit non-coverage)

Reiterating from §3 plus details:

- No bias-add → `LowerError::LinearWithBias`.
- No softmax → `LowerError::UnsupportedOp { op: "softmax" }`.
- No dropout → `LowerError::UnsupportedOp { op: "dropout" }` (even though semantically identity at inference; bundled to M4b).
- No multi-output models. The IR's implicit-output convention is one output per model; M4a respects that.
- No quantisation, no INT8, no BF16. f32 only.
- No SIMD. Scalar instructions only. SIMD is M6 territory (the explicit AVX-512 milestone in PROJECT_SPEC; AArch64 NEON would be a later sibling).
- No CI configuration. No `.github/` files. CI is M3a tech-debt #3 and remains so.
- No new build script (`build.rs`) anywhere. The integration test invokes `cc` at runtime, not at build time.

## 17. Sub-skill chain after this spec is approved

1. Spec self-review (inline placeholder/contradiction scan).
2. User reviews this spec file.
3. On approval → invoke `superpowers:writing-plans` to produce the M4a implementation plan in `docs/superpowers/plans/2026-05-03-m4a-arm64-codegen.md`.
4. Hybrid execution mode (per the project's prior pattern: subagent-driven for substantive logic tasks, inline for trivial scaffolding/cleanup).
5. PR against `main` when M4a is shippable.
