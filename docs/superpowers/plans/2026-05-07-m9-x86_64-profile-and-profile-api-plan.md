# Milestone 9 — x86_64 Linux ELF Profile + `profile-api` Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one PR with six atomic commits that introduce a shared `profile-api` crate, migrate `profiles/arm64/` onto its `Profile` trait byte-identically, build a new `profiles/x86_64/` crate (Linux ELF, scalar SSE2, full op-parity with arm64), wire `nflc compile --profile x86_64` via `Box<dyn Profile>` dispatch, run x86_64 FFI tests on `ubuntu-latest` CI, and land all docs.

**Architecture:** API-first sequencing. Commit 1 extracts the shared crate. Commit 2 migrates arm64 onto the trait without behavioural change (asm output byte-identical, all 223 existing tests green unchanged). Commit 3 builds x86_64 from scratch with the M3-M8 lessons baked in (no immediate-out-of-range complexity, no dropout-as-output retrofit). Commits 4-6 wire the CLI, CI, and docs. The `Profile` trait is minimal by hard rule — `lower(&self, &Uir) -> Result<Asm, LowerError>` and `sym_prefix(&self) -> &'static str`. Nothing else.

**Tech Stack:** Rust 2021 edition, std-only (no external runtime deps in any new crate). `cc` crate (already in arm64 dev-deps) for FFI tests. `libloading` (already in arm64 dev-deps) for dlopen-based test harness. SysV AMD64 ABI. AT&T assembler syntax (gas default on Linux). GitHub Actions `unit` job on `ubuntu-latest` (already exists; comment-only update).

---

## Plan conventions

### Commit-group cadence

The spec mandates six atomic commits sequenced API-first. This plan groups tasks into six **commit-groups** (Group 1 ... Group 6). Each group's last task is the commit step. **Workspace gates run after every task; the commit itself happens only at the end of each group.** Within a group, the workspace stays clean (`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build --workspace`, `cargo test --workspace` all green) but uncommitted; the group's final task stages all changes and commits as one atomic unit.

Do NOT split a group's work across multiple commits. Do NOT batch multiple groups into one commit. The 6-atomic-commit structure is required by the spec (§4.5, §13) and pinned by the byte-identity contract (§4.7) which requires commit 2 to be diffable in isolation.

### AT&T assembler syntax (x86_64)

The spec leaves the AT&T-vs-Intel choice to the plan: §7.3 recommends AT&T (matches gas default on Linux); §7.4's pseudocode happened to use Intel-style memory operands for compactness. **This plan picks AT&T uniformly.** Rationale:

- gas default on Linux — no `.intel_syntax noprefix` directive needed
- `cc` / `clang` on Linux defaults to AT&T
- One less line of generated asm per file
- Standard for the Linux toolchain ecosystem the M9 artefact targets

All x86_64 emitters in commit 3 produce AT&T-form assembly:

- Register names prefixed with `%` (e.g. `%rax`, `%xmm0`, `%r10`)
- Immediates prefixed with `$` (e.g. `$8`, `$0x10`)
- Memory operands as `(base, index, scale)` (e.g. `(%r10, %rcx, 4)`)
- Mnemonic suffixes for size disambiguation: `b`/`w`/`l`/`q` for 1/2/4/8-byte int, `ss`/`sd` for 32/64-bit float
- Source operand on the LEFT, destination on the RIGHT (opposite of Intel)

Example AT&T snippet (loads f32 from `[%r10 + %rcx*4]` into `%xmm0`):

```
movss   (%r10, %rcx, 4), %xmm0
```

### Branch and worktree

Work happens in this worktree: `claude/mystifying-morse-39dc8c`. Spec lives at `docs/superpowers/specs/2026-05-06-m9-x86_64-profile-and-profile-api-design.md`. The branch is already 3 commits ahead of `origin/main` (spec, alignment-fix, DEVLOG backfill); commits from this plan land on top.

### Threading approach for sym_prefix

Spec §6.1 surfaces three options for plumbing the symbol prefix from `Profile::sym_prefix()` to per-emitter call sites: (a) thread `&dyn Profile`, (b) thread a `&'static str`, (c) per-profile associated const. **This plan picks (b) — thread `sym_prefix: &'static str`** uniformly through `walk_uir` → `walk_model` → `format_function_prologue` / `emit_linear` / `emit_softmax` (only emitters that emit profile-prefixed symbols need it). Rationale:

- Lightest plumbing (single `&'static str` param vs `&dyn Profile` indirection in hot codegen paths)
- No vtable lookup per emit-site
- Same shape works for both profiles uniformly (avoids the spec's risk: "trait-method threading touches more arm64 callsites than expected", §14)

Both `Arm64Profile::lower` and `X86_64Profile::lower` reduce to a one-liner that fetches `self.sym_prefix()` once and passes the string slice down.

---

## File structure

The PR creates two new crates and modifies four existing ones (plus docs).

**New: `profile-api/`** (workspace crate, lib only)
- `Cargo.toml` — depends on `compiler` (for `Uir`, `Span`, `NodeId`)
- `src/lib.rs` — `Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError`, `Profile` trait. ~150 lines including 5 unit tests.

**New: `profiles/x86_64/`** (workspace crate, lib + tests)
- `Cargo.toml` — depends on `compiler`, `profile-api`; dev-deps `libloading`
- `src/lib.rs` — `pub struct X86_64Profile; impl Profile for X86_64Profile` + free `lower` shim
- `src/asm.rs` — `compute_frame_size`, `format_function_prologue`, `format_function_epilogue`, `emit_imm32_to_r10` (helper to materialise an arbitrary u32 into `%r10d` — trivial on x86_64, single `movl` instruction)
- `src/buffer.rs` — `BufferLoc` (mirror of arm64), `assign_buffers`, `compute_callee_saved` (SysV: int callee-saved set only; no FP — see spec §7.2)
- `src/codegen.rs` — `walk_uir(&Uir, &'static str)`, `walk_model`, `classify_op`, `resolve_loc`
- `src/ops/{mod,linear,relu,softmax,dropout}.rs` — emitters mirroring arm64's structure
- `src/tests.rs` — unit shape-asserts (one-to-one mirror of `profiles/arm64/src/tests.rs`)
- `tests/common/mod.rs` — `cc_available`, `compile_to_so` helper
- `tests/integration.rs` — FFI tests, cfg-gated to `(target_os = "linux", target_arch = "x86_64")`

**Modified: `profiles/arm64/`**
- `Cargo.toml` — add `profile-api` dependency
- `src/types.rs` — DELETED (types live in `profile-api` now)
- `src/lib.rs` — drop `mod types`; switch `pub use types::...` to `pub use profile_api::...`; add `pub struct Arm64Profile; impl Profile`; keep free `lower` as wrapper
- `src/asm.rs` — remove `MACHO_SYM_PREFIX` const; `format_function_prologue` accepts `sym_prefix: &str`
- `src/codegen.rs` — `walk_uir`/`walk_model` accept `sym_prefix: &'static str`
- `src/ops/{linear,softmax}.rs` — replace literal `"    bl      _expf\n"` with `format!("    bl      {}expf\n", sym_prefix)`; signatures gain `sym_prefix: &str`
- `src/buffer.rs` — drop `node_uses_softmax`; `compute_is_leaf` and `compute_callee_saved` consume `model.calls_extern_math()` (UIR-side predicate) directly. Closes OQ-NEW.

**Modified: `nflc/`**
- `Cargo.toml` — add deps `profile-api`, `profiles-x86_64`
- `src/main.rs` — `run_compile` rewrites the `match profile` block into `Box<dyn Profile>` dispatch; `print_usage` and the unknown-profile error message list both `arm64` and `x86_64`
- `tests/cli.rs` — NEW: 3 CLI smoke tests (x86_64 happy path, unknown profile, arm64 regression guard)

**Modified: workspace `Cargo.toml`** — `members += ["profile-api", "profiles/x86_64"]`

**Modified: `.github/workflows/ci.yml`** — comment-only update on the `unit` job's Test step (per spec §9.4)

**Modified docs (commit 6 only):** `docs/profile_guide/x86_64.md` (new), `docs/profile_guide/arm64.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `README.md`, `DEVLOG.md`. **No documentation changes are made before commit 6**, per spec §12.

---

## Group 1 — Commit 1 — `profile-api` crate

**Group goal:** Introduce `profile-api/` workspace crate exporting the public profile surface and the minimal `Profile` trait. Add `profiles/x86_64/` as a stub workspace member so the manifest is in its final shape from commit 1 onwards (no churn in later commits).

**Group done criteria** (from spec §5.4):
- `cargo build --workspace` green
- `cargo clippy --workspace --all-targets -- -D warnings` green
- `cargo fmt --all -- --check` green
- `cargo test --workspace`: 223 (arm64 unchanged) + 5 (profile-api smoke) = 228
- `profiles/x86_64/` contains stub `Cargo.toml` + 1-line `lib.rs`

### Task 1.1: Add x86_64 stub crate + workspace members

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `profiles/x86_64/Cargo.toml`
- Create: `profiles/x86_64/src/lib.rs`

- [ ] **Step 1: Update workspace `Cargo.toml`**

Replace the file contents:
```toml
[workspace]
resolver = "2"
members = ["compiler", "nflc", "profiles/arm64"]

[workspace.package]
license = "Apache-2.0"
```

with:
```toml
[workspace]
resolver = "2"
members = [
    "compiler",
    "nflc",
    "profile-api",
    "profiles/arm64",
    "profiles/x86_64",
]

[workspace.package]
license = "Apache-2.0"
```

- [ ] **Step 2: Create `profiles/x86_64/Cargo.toml` (stub)**

```toml
[package]
name = "profiles-x86_64"
version = "0.1.0"
edition = "2021"
description = "NeuralForge x86_64 codegen profile"
license.workspace = true

[dependencies]
```

- [ ] **Step 3: Create `profiles/x86_64/src/lib.rs` (stub)**

```rust
// SPDX-License-Identifier: Apache-2.0

//! NeuralForge x86_64 codegen profile (M9 — placeholder; real implementation lands in commit 3).

#[doc(hidden)]
pub fn placeholder() {}
```

- [ ] **Step 4: Verify workspace builds**

Run: `cargo build --workspace`
Expected: success. Cargo emits "compiling profiles-x86_64 v0.1.0" line; new crate has no contents besides the placeholder fn.

### Task 1.2: Create `profile-api/Cargo.toml`

**Files:**
- Create: `profile-api/Cargo.toml`

- [ ] **Step 1: Create the manifest**

```toml
[package]
name = "profile-api"
version = "0.1.0"
edition = "2021"
description = "NeuralForge profile contract — public types + trait shared by all backend profiles"
license.workspace = true

[dependencies]
compiler = { path = "../compiler" }
```

- [ ] **Step 2: Verify Cargo accepts the new member**

Run: `cargo metadata --no-deps --format-version 1 | grep -o '"name":"profile-api"'`
Expected: one match.

### Task 1.3: Write profile-api types + trait + smoke tests (TDD red→green)

**Files:**
- Create: `profile-api/src/lib.rs`

- [ ] **Step 1: Write the file with all types, the trait, and 5 smoke tests**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Public profile contract.
//!
//! Architecture profiles (`profiles/arm64`, `profiles/x86_64`) implement
//! the [`Profile`] trait. The compiler core (`compiler/`) does not depend
//! on any specific profile — UIR is profile-agnostic.

use compiler::ast::Span;
use compiler::ir::types::Uir;
use compiler::NodeId;

/// Generated assembly source plus per-function metadata.
#[derive(Debug, Clone)]
pub struct Asm {
    /// Full assembly source. UTF-8.
    pub source: String,
    /// One entry per UirModel in the input UIR, in declaration order.
    pub functions: Vec<FnSig>,
}

/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    /// External symbol name without leading underscore. e.g. "nfl_forward_TinyMLP".
    /// Mach-O linkers prepend the underscore; ELF linkers do not. `dlsym`
    /// callers pass this name verbatim.
    pub name: String,
    /// Original UIR model name.
    pub model: String,
    /// Number of f32 elements in the input buffer.
    pub input_floats: usize,
    /// Number of f32 elements in the output buffer.
    pub output_floats: usize,
    /// Total number of f32 elements in the packed params buffer.
    pub params_floats: usize,
    /// Layout of the packed params buffer, one entry per parameter slot in
    /// UIR-node order.
    pub params_layout: Vec<ParamSlot>,
}

/// One slot within the packed `params` buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSlot {
    pub kind: ParamKind,
    pub origin_node: NodeId,
    pub offset: usize,
    pub size: usize,
}

/// Type tag for a `ParamSlot`. `#[non_exhaustive]` keeps the door open
/// for future kinds (e.g. `LayerNormScale`) without breaking match arms
/// in downstream crates.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    LinearWeight,
    LinearBias,
}

/// Errors that can occur during lowering.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// Defensive: op encountered that the codegen doesn't know how to lower.
    UnsupportedOp { op: String, span: Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    ShapeNotConcrete { span: Span },
    /// Defensive: post-op variant not supported by this profile.
    UnsupportedPostOp { op: String, span: Span },
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
        }
    }
}

impl std::error::Error for LowerError {}

impl LowerError {
    /// Returns the source span associated with the error.
    pub fn span(&self) -> Span {
        match self {
            LowerError::UnsupportedOp { span, .. } => *span,
            LowerError::ShapeNotConcrete { span } => *span,
            LowerError::UnsupportedPostOp { span, .. } => *span,
        }
    }
}

/// The profile contract.
///
/// Each backend profile (arm64 Mach-O, x86_64 Linux ELF, ...) provides
/// one `impl Profile` for its profile struct. The compiler core never
/// references a concrete profile by type — only through this trait.
///
/// **Trait grows by request, not by anticipation** (per M9 brainstorm
/// hard rule). Adding a method requires a real consumer in the codebase
/// that needs it.
pub trait Profile {
    /// Lower a [`Uir`] to the profile's target assembly.
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;

    /// Platform-specific external-symbol prefix.
    /// `"_"` on Mach-O (linker prepends underscore for C linkage),
    /// `""` on ELF (linker uses raw symbol name).
    fn sym_prefix(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use compiler::ast::Span;

    fn dummy_span() -> Span {
        Span { line: 1, col: 1, len: 0 }
    }

    #[test]
    fn asm_round_trip_through_debug() {
        let a = Asm { source: "x".into(), functions: vec![] };
        let dbg = format!("{:?}", a);
        assert!(dbg.contains("source"));
    }

    #[test]
    fn fn_sig_round_trip_through_debug() {
        let s = FnSig {
            name: "f".into(),
            model: "M".into(),
            input_floats: 1,
            output_floats: 1,
            params_floats: 0,
            params_layout: vec![],
        };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("FnSig"));
    }

    #[test]
    fn param_slot_round_trip_through_debug() {
        let p = ParamSlot {
            kind: ParamKind::LinearWeight,
            origin_node: 0,
            offset: 0,
            size: 4,
        };
        let dbg = format!("{:?}", p);
        assert!(dbg.contains("LinearWeight"));
    }

    #[test]
    fn lower_error_display_message_is_profile_neutral() {
        let e = LowerError::UnsupportedOp { op: "foo".into(), span: dummy_span() };
        let msg = format!("{}", e);
        assert!(
            msg.contains("not supported by this profile"),
            "Display message must be profile-neutral; got: {}",
            msg
        );
        assert!(!msg.contains("arm64"), "Display must not mention arm64; got: {}", msg);
        assert!(!msg.contains("x86_64"), "Display must not mention x86_64; got: {}", msg);
    }

    #[test]
    fn lower_error_span_round_trip() {
        let s = Span { line: 3, col: 7, len: 0 };
        let e = LowerError::ShapeNotConcrete { span: s };
        assert_eq!(e.span().line, 3);
        assert_eq!(e.span().col, 7);
    }
}
```

> **Note on `Span` field shape:** the test instantiates `Span { line: 1, col: 1, len: 0 }`. If `compiler::ast::Span` exposes different field names (verify by reading `compiler/src/ast.rs`), adapt the literal accordingly — the tests are what pin the shape, not vice versa.

- [ ] **Step 2: Run the new tests**

Run: `cargo test -p profile-api --lib`
Expected: 5 passing. If a compile error fires, it's almost certainly the `Span` field shape — open `compiler/src/ast.rs`, copy the actual field set, fix the literal in the `dummy_span` helper.

- [ ] **Step 3: Run workspace gates**

Run, in order:
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo build --workspace`
- `cargo test --workspace`

Expected: all green. Total test count: 223 (compiler + arm64 unchanged) + 5 (profile-api smoke) = **228**.

### Task 1.4: Stage + commit Group 1

- [ ] **Step 1: Review the stage**

Run: `git status`
Expected: 1 modified file (workspace `Cargo.toml`); 4 new files (`profile-api/Cargo.toml`, `profile-api/src/lib.rs`, `profiles/x86_64/Cargo.toml`, `profiles/x86_64/src/lib.rs`).

- [ ] **Step 2: Stage**

```bash
git add Cargo.toml profile-api profiles/x86_64
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m9/profile-api): introduce shared profile-api crate + x86_64 stub

profile-api/ exports the public profile surface (Asm, FnSig, ParamSlot,
ParamKind, LowerError) and a minimal Profile trait — lower(&Uir) and
sym_prefix() only. The trait is the type-level contract that profiles
must implement; this makes profile isolation compiler-checked instead
of an informal "two crates with similar APIs" claim.

Field shapes mirror profiles/arm64/src/types.rs verbatim. The Display
message on LowerError is genericised to "this profile" (instead of
"arm64 profile") so the type can serve every profile.

profiles/x86_64/ added as a stub workspace member with a placeholder
lib.rs so the manifest reaches its final shape from commit 1; the real
implementation lands in commit 3. This avoids a "members changed in
commit 3" diff that would touch the manifest twice.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify**

Run: `cargo test --workspace`
Expected: 228 passing. Run `git log -1 --stat`; expected: one commit, 5 files in the diff.

---

## Group 2 — Commit 2 — arm64 migration onto `Profile` trait

**Group goal:** Move `profiles/arm64/src/types.rs` types into `profile-api`. Add `pub struct Arm64Profile` + `impl Profile`. Plumb `sym_prefix: &'static str` through `walk_uir` → `walk_model` → emitters; replace the hardcoded `MACHO_SYM_PREFIX` constant and the two `bl _expf` literals with format-string substitutions. Close OQ-NEW by replacing the profile-side `node_uses_softmax` with the UIR-side `model.calls_extern_math()`.

**Hard contract** (spec §4.7, §6.3): all 223 pre-existing tests pass without modification, AND every fixture's lowered asm is byte-identical pre/post migration. If a test fails, the migration is buggy — fix the code, do not patch the test. If the asm differs, the migration is buggy — fix the code, do not adjust the baseline.

**Group done criteria** (from spec §6.4):
- `cargo test --workspace`: 228 (count unchanged from Group 1; arm64 contributes 60 tests = 45 unit + 15 integration)
- Asm output byte-identical to pre-migration baseline for every fixture
- Workspace gates green
- OQ-NEW resolved (closed because all sites reduce to `calls_extern_math`)

### Task 2.1: Capture pre-migration asm baseline

**Files:** none modified (read-only verification capture)

- [ ] **Step 1: Generate per-fixture asm baselines**

```bash
mkdir -p /tmp/m9-baseline-arm64
for f in tests/fixtures/*.nfl; do
  base=$(basename "$f" .nfl)
  cargo run -q -p nflc -- compile "$f" --profile arm64 \
        -o "/tmp/m9-baseline-arm64/${base}.s" 2>/dev/null \
    || echo "skip: ${base} (uncompilable fixture; not part of byte-identity contract)"
done
ls /tmp/m9-baseline-arm64/
```

Expected: a `.s` file per parseable+lowerable fixture. Some fixtures may intentionally fail to compile (e.g. negative-test fixtures); those skip-line outputs are acceptable.

- [ ] **Step 2: Snapshot the baseline (sha256 manifest)**

```bash
( cd /tmp/m9-baseline-arm64 && shasum -a 256 *.s ) > /tmp/m9-baseline-arm64.sha256
wc -l /tmp/m9-baseline-arm64.sha256
cat /tmp/m9-baseline-arm64.sha256
```

Expected: a list of `<sha256-hex>  <fixture>.s` lines. **Keep this file open** — it is the authority for byte-identity verification at the end of Group 2.

### Task 2.2: Add profile-api dep to profiles/arm64

**Files:**
- Modify: `profiles/arm64/Cargo.toml`

- [ ] **Step 1: Add the dependency**

Replace the `[dependencies]` block:

```toml
[dependencies]
compiler = { path = "../../compiler" }
```

with:

```toml
[dependencies]
compiler    = { path = "../../compiler" }
profile-api = { path = "../../profile-api" }
```

- [ ] **Step 2: Verify build still works (types not yet moved)**

Run: `cargo build -p profiles-arm64`
Expected: success. Cargo may emit `unused_crate_dependencies` warning for `profile-api` — acceptable in this transient state; cleared by Task 2.3 when the consumer code lands.

### Task 2.3: Delete `profiles/arm64/src/types.rs`; switch consumers to `profile_api::*`

**Files:**
- Delete: `profiles/arm64/src/types.rs`
- Modify: `profiles/arm64/src/lib.rs` (drop `mod types`; change `pub use`; add `Arm64Profile` struct + `impl Profile`; keep free `lower` shim)
- Modify: `profiles/arm64/src/{asm,buffer,codegen,ops/{linear,relu,softmax,dropout},tests}.rs` (switch imports)

- [ ] **Step 1: Find all consumer sites**

Run:
```bash
grep -rn "use crate::types\|use crate::{[A-Z]\|crate::LowerError\|crate::Asm\|crate::FnSig\|crate::ParamKind\|crate::ParamSlot\|use crate::types::" profiles/arm64/src/
```

Expected: a short list of files importing the moved types. Keep this output — every match needs to be updated in steps 4+.

- [ ] **Step 2: Delete types.rs**

```bash
rm profiles/arm64/src/types.rs
```

- [ ] **Step 3: Rewrite `profiles/arm64/src/lib.rs`**

Replace the file's contents:

```rust
// SPDX-License-Identifier: Apache-2.0

//! NeuralForge arm64 scalar codegen profile.
//!
//! Lowers a [`compiler::Uir`] to AArch64 Mach-O assembly text via
//! [`Arm64Profile`]. The free [`lower`] shim is preserved for direct
//! callers (arm64 integration tests) that pre-date the trait.

mod asm;
mod buffer;
mod codegen;
mod ops;

pub use profile_api::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

use compiler::Uir;
use profile_api::Profile;

/// arm64 profile implementation. Lowers UIR to AArch64 Mach-O assembly.
pub struct Arm64Profile;

impl Profile for Arm64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        "_"
    }
}

/// Free-function shim retained for direct callers (arm64 integration
/// tests). Equivalent to `Arm64Profile.lower(uir)`.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    Arm64Profile.lower(uir)
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 4: Update internal imports across the crate**

For each `.rs` file under `profiles/arm64/src/` (except `lib.rs`):
- Replace `use crate::types::{...}` with `use profile_api::{...}`
- Replace `use crate::{Asm, ...}` (importing types from lib.rs `pub use`) with `use profile_api::{...}` (these now re-export from `profile_api`, but importing from `profile_api` directly is cleaner inside the crate)
- Replace `crate::types::LowerError` etc. with `profile_api::LowerError`
- For `tests.rs` specifically: keep `crate::lower(&uir)` calls as-is — the free `lower` shim is still exported from `lib.rs`. (Tests do not need to know about `Arm64Profile`.)

After each file, run `cargo build -p profiles-arm64` and fix any errors before moving to the next file.

- [ ] **Step 5: Run all arm64 tests; expect green**

Run: `cargo test -p profiles-arm64`
Expected: 60 passing (45 unit + 15 integration). **No test code changed.** If any test fails, the migration is buggy. Investigate and fix the migration code; do not patch the test.

### Task 2.4: Plumb `sym_prefix` through codegen + emitters

**Files:**
- Modify: `profiles/arm64/src/codegen.rs` — `walk_uir` and `walk_model` accept `sym_prefix: &'static str`
- Modify: `profiles/arm64/src/asm.rs` — `format_function_prologue` accepts `sym_prefix: &str`; remove `MACHO_SYM_PREFIX` const
- Modify: `profiles/arm64/src/ops/linear.rs` — `emit_linear` accepts `sym_prefix: &str`; rewrite the `bl _expf` site (line ~200, inside the `PostOp::SoftmaxRow` branch) as `format!("    bl      {}expf\n", sym_prefix)`
- Modify: `profiles/arm64/src/ops/softmax.rs` — `emit_softmax` accepts `sym_prefix: &str`; rewrite the `bl _expf` site (line ~85)

- [ ] **Step 1: Update `walk_uir` signature in codegen.rs**

Change:
```rust
pub fn walk_uir(uir: &Uir) -> Result<Asm, LowerError> {
    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for (model_idx, model) in uir.models.iter().enumerate() {
        let (model_asm, sig) = walk_model(model_idx, model)?;
        ...
```

to:
```rust
pub fn walk_uir(uir: &Uir, sym_prefix: &'static str) -> Result<Asm, LowerError> {
    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for (model_idx, model) in uir.models.iter().enumerate() {
        let (model_asm, sig) = walk_model(model_idx, model, sym_prefix)?;
        ...
```

And update `walk_model`'s signature similarly (`fn walk_model(model_idx: usize, model: &UirModel, sym_prefix: &'static str) -> ...`).

- [ ] **Step 2: Update `format_function_prologue` in asm.rs**

In `profiles/arm64/src/asm.rs`:

Delete:
```rust
pub const MACHO_SYM_PREFIX: &str = "_";
```

Change `format_function_prologue` signature and body:
```rust
pub fn format_function_prologue(
    sig: &FnSig,
    leaf: LeafKind,
    regs: RegSet,
    intermediate_bytes: usize,
    sym_prefix: &str,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(".globl {}{}\n", sym_prefix, sig.name));
    s.push_str(".p2align 2\n");
    s.push_str(&format!("{}{}:\n", sym_prefix, sig.name));
    // ... rest of the function unchanged ...
```

Update the call site in `codegen.rs::walk_model` to pass `sym_prefix` as the 5th argument.

- [ ] **Step 3: Update `emit_linear` in ops/linear.rs**

In `profiles/arm64/src/ops/linear.rs`, add `sym_prefix: &str` to the `emit_linear` parameter list (after `fused_post_ops`). Find the `bl _expf` line inside the `PostOp::SoftmaxRow` branch:

```rust
                s.push_str("    bl      _expf\n");
```

Replace with:
```rust
                s.push_str(&format!("    bl      {}expf\n", sym_prefix));
```

Update the `crate::ops::emit_linear(...)` call site in `codegen.rs::walk_model` to pass `sym_prefix` as the final argument.

- [ ] **Step 4: Update `emit_softmax` in ops/softmax.rs**

In `profiles/arm64/src/ops/softmax.rs`, add `sym_prefix: &str` to the `emit_softmax` parameter list. Find the `bl _expf` line:

```rust
    s.push_str("    bl      _expf\n");
```

Replace with:
```rust
    s.push_str(&format!("    bl      {}expf\n", sym_prefix));
```

Update the `crate::ops::emit_softmax(...)` call site in `codegen.rs::walk_model` to pass `sym_prefix` as the final argument.

- [ ] **Step 5: Build until green**

Run: `cargo build -p profiles-arm64`
Expected: a sequence of "missing argument" errors as the parameter propagates; fix each, re-run, until green.

- [ ] **Step 6: Run unit tests**

Run: `cargo test -p profiles-arm64 --lib`
Expected: 45 unit tests passing. The substring asserts `s.contains("bl      _expf")` and `.contains(".globl _nfl_forward_M")` continue to match because for arm64 `sym_prefix() -> "_"`, the format expansion is byte-identical to the previous string literal.

### Task 2.5: OQ-NEW resolution — replace `node_uses_softmax` with `model.calls_extern_math()`

**Files:**
- Modify: `profiles/arm64/src/buffer.rs` — drop the local `node_uses_softmax` helper; rewrite `compute_is_leaf` and `compute_callee_saved` to call `model.calls_extern_math()` (UIR-side predicate already on `UirModel`)

- [ ] **Step 1: Verify the substitution is semantically equivalent**

Read `compiler/src/ir/types.rs:217-241`. Confirm `UirModel::calls_extern_math()` checks the same condition as the local `node_uses_softmax`:
```rust
matches!(op, StdOp::Softmax)
    || fused_post_ops.iter().any(|p| matches!(p, PostOp::SoftmaxRow))
```
across `model.nodes`. They match exactly.

- [ ] **Step 2: Apply the rewrite in `profiles/arm64/src/buffer.rs`**

Delete the entire `fn node_uses_softmax(node: &Node) -> bool { ... }` function. Replace `compute_is_leaf` and `compute_callee_saved` with:

```rust
/// True iff the model emits no `bl`/`blr` (i.e. no softmax in any form).
///
/// After M6 fusion, a fused `linear → softmax` node carries
/// `PostOp::SoftmaxRow` and still calls `bl _expf` — so such a model is
/// not a leaf even though there is no standalone `StdOp::Softmax` node.
/// Delegated to UIR-side `UirModel::calls_extern_math` (single source of
/// truth across profiles).
pub fn compute_is_leaf(model: &UirModel) -> bool {
    !model.calls_extern_math()
}

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    let has_extern_math = model.calls_extern_math();
    RegSet {
        d8_d9: has_extern_math,
        x19_x23: has_extern_math,
    }
}
```

If the `use compiler::{Node, ...}` import line at the top now imports `Node` only for the deleted helper, remove `Node` from the import list.

- [ ] **Step 3: Run all tests**

Run: `cargo test -p profiles-arm64`
Expected: 60 passing. Substitution is a behavioural no-op.

### Task 2.6: Verify byte-identity against baseline

**Files:** none modified (verification step)

- [ ] **Step 1: Re-generate post-migration asm**

```bash
mkdir -p /tmp/m9-postmigration-arm64
rm -f /tmp/m9-postmigration-arm64/*.s
for f in tests/fixtures/*.nfl; do
  base=$(basename "$f" .nfl)
  cargo run -q -p nflc -- compile "$f" --profile arm64 \
        -o "/tmp/m9-postmigration-arm64/${base}.s" 2>/dev/null \
    || echo "skip: ${base}"
done
```

- [ ] **Step 2: Compare against the baseline manifest**

```bash
( cd /tmp/m9-postmigration-arm64 && shasum -a 256 *.s ) > /tmp/m9-postmigration-arm64.sha256
diff /tmp/m9-baseline-arm64.sha256 /tmp/m9-postmigration-arm64.sha256
echo "exit: $?"
```

Expected: empty diff, `exit: 0`. **If non-empty, the migration introduced a regression** — diff individual files (`diff /tmp/m9-baseline-arm64/<name>.s /tmp/m9-postmigration-arm64/<name>.s`) to localise it. Investigate and fix the migration code; do not adjust the baseline.

### Task 2.7: Workspace gates

**Files:** none modified

- [ ] **Step 1: Format check**

Run: `cargo fmt --all -- --check`
Expected: no diff.

- [ ] **Step 2: Clippy zero-warnings**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: success.

- [ ] **Step 3: Full workspace tests**

Run: `cargo test --workspace`
Expected: **228** passing (compiler 163 + profile-api 5 + arm64 60 = 228; arm64 contribution unchanged from Group 1).

### Task 2.8: Stage + commit Group 2

- [ ] **Step 1: Review the stage**

Run: `git status && git diff --stat`
Expected: 1 deleted file (`profiles/arm64/src/types.rs`) + ~9 modified files in `profiles/arm64/`.

- [ ] **Step 2: Stage**

```bash
git add profiles/arm64
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m9/arm64-migration): impl Profile for Arm64Profile via profile-api

arm64 types (Asm, FnSig, ParamSlot, ParamKind, LowerError) move out of
profiles/arm64/src/types.rs into the new profile-api crate. Arm64Profile
now `impl Profile`; the free-function `lower` is preserved as a thin
wrapper for direct integration-test callers.

The MACHO_SYM_PREFIX const is removed; a `sym_prefix: &'static str`
threads through walk_uir/walk_model/format_function_prologue/
emit_linear/emit_softmax and replaces the hardcoded "_" / "bl _expf"
literals. For arm64 sym_prefix() -> "_", the format expansion is
byte-identical to the previous output (verified by sha256 diff against
a captured pre-migration baseline of every fixture).

OQ-NEW resolved: profiles/arm64/src/buffer.rs::node_uses_softmax is
deleted in favour of UirModel::calls_extern_math() (UIR-side, M8
predicate). Single source of truth for the "this model uses libm-expf"
check; no further profile-side duplication.

Hard contract preserved: all 223 pre-existing tests pass without
modification; arm64 contribution to the workspace test count unchanged
at 60 (45 unit + 15 integration). No test code edited in this commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify**

Run: `git log -2 --oneline && cargo test --workspace`
Expected: two recent commits visible; 228 tests pass.

---

## Group 3 — Commit 3 — `profiles/x86_64/` crate

**Group goal:** Build the x86_64 Linux ELF profile from scratch with full op-parity to arm64. Scalar SSE2 only, no SIMD/AVX. AT&T syntax. Full fused PostOp parity (`ReluFused`, `SoftmaxRow`). FFI integration tests are deferred to Group 5 (cfg-gating + CI wiring go together).

**SysV AMD64 ABI** (spec §7.2):
- Int args: `%rdi`, `%rsi`, `%rdx` (we use only the first 3 for the `(input, params, output)` signature)
- Float args: not used (FFI signature is all int pointers)
- Callee-saved int: `%rbx`, `%rbp`, `%r12`, `%r13`, `%r14`, `%r15`
- **Callee-saved FP: NONE** (`%xmm0`-`%xmm15` all caller-saved — divergence from arm64's `d8/d9` callee-saved set)
- Stack: 16-byte aligned at `call` boundary (see `compute_frame_size`, §7.5)

**Register allocation contract** (must match across all emitters in this profile):
- `%rdi` (= input ptr, FFI arg 0)
- `%rsi` (= params ptr, FFI arg 1)
- `%rdx` (= output ptr, FFI arg 2)
- `%r10`, `%r11` — buffer pointer scratch (set by `materialise_ptr`; analog of arm64 x11/x12)
- `%r10d` — additionally used for emit_imm32_to_r10 (32-bit immediate materialisation; trivial single-instruction `movl $imm, %r10d` on x86_64)
- `%rcx` — innermost loop counter (analog of arm64 x9)
- `%xmm0`, `%xmm1` — scratch float registers (caller-saved)
- **For softmax (any form)**, additionally:
  - `%rbx` (callee-saved) ← src buffer pointer (survives `call expf@PLT`)
  - `%r12` (callee-saved) ← dst buffer pointer (survives the call)
  - `%r13` (callee-saved) ← outer i counter
  - `%r14` (callee-saved) ← inner j counter
  - `%r15` (callee-saved) ← row_base = `i * N` (recomputed if needed; held across the call)
  - `[%rsp + max_slot_off]` — row_max f32 stack slot (xmm-spilled; xmm regs are caller-saved)
  - `[%rsp + sum_slot_off]` — row_sum f32 stack slot

This contract is pinned in unit tests in Task 3.10 and FFI tests in Group 5.

**Group done criteria** (from spec §7.9):
- `cargo build --workspace` green (x86_64 crate compiles)
- `cargo clippy --workspace --all-targets -- -D warnings` green
- `cargo test --workspace`: 228 (Group 2) + 8 (compute_frame_size) + ~45 (x86_64 unit shape) ≈ **281**
- `nflc compile <fixture> --profile x86_64` does NOT yet work (CLI dispatch is Group 4) — by design

### Task 3.1: Replace x86_64 stub Cargo.toml + create source skeleton

**Files:**
- Modify: `profiles/x86_64/Cargo.toml` (replace the stub from Group 1)
- Modify: `profiles/x86_64/src/lib.rs` (replace the placeholder fn)
- Create: `profiles/x86_64/src/asm.rs` (empty module — body lands in 3.2)
- Create: `profiles/x86_64/src/buffer.rs` (empty)
- Create: `profiles/x86_64/src/codegen.rs` (empty)
- Create: `profiles/x86_64/src/ops/mod.rs`
- Create: `profiles/x86_64/src/ops/{linear,relu,softmax,dropout}.rs` (empty)
- Create: `profiles/x86_64/src/tests.rs` (empty)

- [ ] **Step 1: Replace `profiles/x86_64/Cargo.toml`**

```toml
[package]
name = "profiles-x86_64"
version = "0.1.0"
edition = "2021"
description = "NeuralForge x86_64 codegen profile"
license.workspace = true

[dependencies]
compiler    = { path = "../../compiler" }
profile-api = { path = "../../profile-api" }

[dev-dependencies]
libloading = "0.8"
```

- [ ] **Step 2: Replace `profiles/x86_64/src/lib.rs`**

```rust
// SPDX-License-Identifier: Apache-2.0

//! NeuralForge x86_64 scalar codegen profile.
//!
//! Lowers a [`compiler::Uir`] to x86_64 Linux ELF assembly text via
//! [`X86_64Profile`]. Scalar SSE2 only — no SIMD/AVX. AT&T syntax.

mod asm;
mod buffer;
mod codegen;
mod ops;

pub use profile_api::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

use compiler::Uir;
use profile_api::Profile;

/// x86_64 profile implementation. Lowers UIR to x86_64 Linux ELF assembly.
pub struct X86_64Profile;

impl Profile for X86_64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        ""
    }
}

/// Free-function shim, mirror of arm64's. Equivalent to
/// `X86_64Profile.lower(uir)`.
pub fn lower(uir: &Uir) -> Result<Asm, LowerError> {
    X86_64Profile.lower(uir)
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 3: Create empty source files**

```bash
mkdir -p profiles/x86_64/src/ops
touch profiles/x86_64/src/asm.rs
touch profiles/x86_64/src/buffer.rs
touch profiles/x86_64/src/codegen.rs
touch profiles/x86_64/src/tests.rs
touch profiles/x86_64/src/ops/{mod,linear,relu,softmax,dropout}.rs
```

For each new `.rs` file, write the SPDX header line:
```rust
// SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 4: Build will fail (expected) — `lib.rs` references empty modules**

Run: `cargo build -p profiles-x86_64`
Expected: errors of the form `unresolved import 'codegen::walk_uir'` etc. **This is the intended TDD-red state.** Proceed to 3.2 to fill in modules.

### Task 3.2: Implement `compute_frame_size` (TDD red→green)

**Files:**
- Modify: `profiles/x86_64/src/asm.rs`
- Modify: `profiles/x86_64/src/tests.rs` (add 8 unit tests for compute_frame_size)

- [ ] **Step 1: Write the 8 failing test cases first**

Append to `profiles/x86_64/src/tests.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

use crate::asm::compute_frame_size;

// Spec §7.5: entry-state rsp ≡ 8 (mod 16); after sub rsp, frame_size
// final rsp must be ≡ 0 (mod 16). Each push reg flips parity (8 bytes).
// Therefore the +8 correction applies when num_pushes is EVEN
// (post-pushes parity is 8) — see spec §4.9 derivation.

#[test]
fn frame_size_raw0_pushes0_is_8() {
    // post-pushes ≡ 8; sub 8 → 0 ✓
    assert_eq!(compute_frame_size(0, 0), 8);
}

#[test]
fn frame_size_raw0_pushes1_is_0() {
    // post-pushes ≡ 0; sub 0 → 0 ✓
    assert_eq!(compute_frame_size(0, 1), 0);
}

#[test]
fn frame_size_raw0_pushes2_is_8() {
    // post-pushes ≡ 8; sub 8 → 0 ✓
    assert_eq!(compute_frame_size(0, 2), 8);
}

#[test]
fn frame_size_raw8_pushes0_is_24() {
    // aligned=16, +8; post-pushes ≡ 8; sub 24 ≡ -16 ≡ 0 ✓
    assert_eq!(compute_frame_size(8, 0), 24);
}

#[test]
fn frame_size_raw8_pushes1_is_16() {
    // aligned=16, +0; post-pushes ≡ 0; sub 16 → 0 ✓
    assert_eq!(compute_frame_size(8, 1), 16);
}

#[test]
fn frame_size_raw16_pushes1_is_16() {
    // same alignment as the raw=8/pushes=1 case
    assert_eq!(compute_frame_size(16, 1), 16);
}

#[test]
fn frame_size_raw17_pushes0_is_40() {
    // aligned=32, +8; post-pushes ≡ 8; sub 40 ≡ -32 ≡ 0 ✓
    assert_eq!(compute_frame_size(17, 0), 40);
}

#[test]
fn frame_size_raw17_pushes1_is_32() {
    // aligned=32, +0; post-pushes ≡ 0; sub 32 → 0 ✓
    assert_eq!(compute_frame_size(17, 1), 32);
}
```

- [ ] **Step 2: Run; expect compile failure**

Run: `cargo test -p profiles-x86_64 --lib frame_size`
Expected: compile error — `compute_frame_size` not in `crate::asm`.

- [ ] **Step 3: Implement `compute_frame_size` in `profiles/x86_64/src/asm.rs`**

Append:

```rust
/// SysV AMD64 stack frame size, including the alignment correction
/// dictated by the prologue's `push` count.
///
/// Derivation (spec §7.5): on function entry, the caller's `call`
/// instruction has just pushed the 8-byte return address, so
/// `rsp ≡ 8 (mod 16)`. Each prologue `push reg` (8 bytes) flips parity.
/// After N pushes:
///
/// ```text
/// rsp ≡ 8 - 8*N (mod 16) ≡ 8*(1 - N) (mod 16)
/// ```
///
/// To land on `rsp ≡ 0 (mod 16)` after `sub rsp, frame_size`, the
/// helper adds an 8-byte correction when N is **even** (post-pushes
/// parity is 8), and zero correction when N is **odd** (post-pushes
/// parity is 0).
pub fn compute_frame_size(raw_buffer_size: u32, num_pushes: usize) -> u32 {
    let aligned = (raw_buffer_size + 15) & !15;
    let push_correction = if num_pushes % 2 == 0 { 8 } else { 0 };
    aligned + push_correction
}
```

- [ ] **Step 4: Run tests; expect 8 passing**

Run: `cargo test -p profiles-x86_64 --lib frame_size`
Expected: 8 passing.

### Task 3.3: Implement `emit_imm32_to_r10` + `materialise_ptr` (asm.rs)

**Files:**
- Modify: `profiles/x86_64/src/asm.rs` (add helpers)

- [ ] **Step 1: Add `emit_imm32_to_r10`**

x86_64's equivalent to arm64's `emit_imm32` (which had to handle movz+movk for immediates > 16 bits) is trivial — a single `movl` instruction takes any 32-bit immediate. No size hoisting / strategy splitting needed. Per spec §7.7, this is one of the "M3-M8 lessons that don't transfer".

Append to `profiles/x86_64/src/asm.rs`:

```rust
/// Materialise an arbitrary u32 into `%r10d` using a single instruction.
/// x86_64 `movl $imm32, %r10d` accepts any 32-bit immediate directly —
/// no movz/movk dance required (contrast with arm64::asm::emit_imm32).
pub fn emit_imm32_to_r10(value: u32) -> String {
    format!("    movl    ${}, %r10d\n", value)
}
```

- [ ] **Step 2: Add `materialise_ptr` (analog of arm64's, for x86_64 register names + AT&T syntax)**

Append:

```rust
use crate::buffer::BufferLoc;

/// Materialise a [`BufferLoc`] into the named register.
///
/// FFI signature contract (matches arm64): arg 0 = input (%rdi),
/// arg 1 = params (%rsi), arg 2 = output (%rdx).
///
/// Stack-resident buffers live at `[%rsp + offset]`. x86_64 `lea`
/// accepts any 32-bit signed displacement directly — no size cliffs
/// (contrast arm64's 12-bit immediate dance).
pub fn materialise_ptr(reg: &str, loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg => format!("    movq    %rdi, {}\n", reg),
        BufferLoc::OutputReg => format!("    movq    %rdx, {}\n", reg),
        BufferLoc::StackOffset(off) => {
            assert!(
                off <= i32::MAX as usize,
                "stack offset > 2 GiB unsupported (got {} bytes)",
                off
            );
            if off == 0 {
                format!("    movq    %rsp, {}\n", reg)
            } else {
                format!("    leaq    {}(%rsp), {}\n", off, reg)
            }
        }
        BufferLoc::Alias(_) => {
            unreachable!("materialise_ptr should never see Alias — resolve_loc must run first")
        }
    }
}
```

- [ ] **Step 3: Run workspace build (will still fail because BufferLoc not yet defined)**

Run: `cargo build -p profiles-x86_64`
Expected: error — `crate::buffer::BufferLoc` not found. Continue to Task 3.4.

### Task 3.4: Implement `BufferLoc` + analyzers (buffer.rs)

**Files:**
- Modify: `profiles/x86_64/src/buffer.rs`

- [ ] **Step 1: Implement BufferLoc + assignment + analyzers**

Replace the file's contents:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Buffer assignment + leaf/callee-saved analyzers for the x86_64 codegen.
//!
//! Pure analyzers over `UirModel`. No asm emission. Mirrors the structure
//! of `profiles/arm64/src/buffer.rs` modulo register-naming differences.

use compiler::{NodeId, NodeKind, StdOp, UirModel};

/// Bytes per f32 element. f32-only project-wide.
const BYTES_PER_ELEMENT: usize = 4;

/// Where an Op-node's output buffer lives at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    /// Input pointer (FFI arg 0 — %rdi).
    InputReg,
    /// Output pointer (FFI arg 2 — %rdx).
    OutputReg,
    /// Stack slot at `[%rsp + offset]`.
    StackOffset(usize),
    /// This buffer is an alias for another node's buffer. Resolved by
    /// `codegen::resolve_loc` before any emit.
    Alias(NodeId),
}

/// Result of buffer assignment.
#[derive(Debug, Clone)]
pub struct BufferAssignment {
    /// Per-NodeId placement; index by NodeId.
    pub locs: Vec<BufferLoc>,
    /// Total stack bytes required for intermediate buffers, rounded up
    /// to 16-byte alignment. Excludes the additional softmax xmm-spill
    /// slots (allocated separately in emit_softmax / fused tail).
    pub stack_bytes: usize,
}

/// Assign a `BufferLoc` per UIR node + compute aligned total stack frame size.
pub fn assign_buffers(model: &UirModel) -> BufferAssignment {
    let mut locs = vec![BufferLoc::InputReg; model.nodes.len()];
    let mut stack_offset: usize = 0;

    for (id, node) in model.nodes.iter().enumerate() {
        locs[id] = match &node.kind {
            NodeKind::Input { .. } => BufferLoc::InputReg,
            NodeKind::Op { op, operands, .. } => {
                if id == model.output {
                    BufferLoc::OutputReg
                } else {
                    match op {
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
                        StdOp::Linear | StdOp::Softmax => {
                            let elements: u64 = node.ty.shape.0.iter().copied().product();
                            let size_bytes = (elements as usize)
                                .checked_mul(BYTES_PER_ELEMENT)
                                .expect("buffer size overflow");
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset = stack_offset
                                .checked_add(size_bytes)
                                .expect("stack frame size overflow");
                            loc
                        }
                        #[allow(unreachable_patterns)]
                        _ => {
                            let elements: u64 = node.ty.shape.0.iter().copied().product();
                            let size_bytes = (elements as usize)
                                .checked_mul(BYTES_PER_ELEMENT)
                                .expect("buffer size overflow");
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset = stack_offset
                                .checked_add(size_bytes)
                                .expect("stack frame size overflow");
                            loc
                        }
                    }
                }
            }
        };
    }

    let stack_bytes = (stack_offset + 15) & !15;
    BufferAssignment { locs, stack_bytes }
}

/// Set of callee-saved int registers used by the model's body.
///
/// SysV AMD64 callee-saved int set: `%rbx, %rbp, %r12, %r13, %r14, %r15`.
/// This profile uses `%rbp` as the frame pointer (always saved) and the
/// other 5 (`%rbx, %r12, %r13, %r14, %r15`) iff any node calls
/// `expf@PLT` — either a standalone `StdOp::Softmax` or a `Linear`
/// carrying `PostOp::SoftmaxRow`. UIR predicate
/// (`UirModel::calls_extern_math`) is the single source of truth.
///
/// Unlike arm64, **there is no callee-saved FP register set**. All
/// `%xmm0`-`%xmm15` are caller-saved per SysV. The fused softmax tail
/// spills row_max / row_sum to the stack across `call expf@PLT`
/// (see spec §7.4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegSet {
    /// True iff `%rbx, %r12, %r13, %r14, %r15` are saved in the
    /// prologue (and restored in the epilogue).
    pub callee_saved_int: bool,
}

impl RegSet {
    pub fn contains_callee_saved_int(&self) -> bool {
        self.callee_saved_int
    }
}

/// True iff the model emits no `call` (i.e. no softmax in any form).
pub fn compute_is_leaf(model: &UirModel) -> bool {
    !model.calls_extern_math()
}

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    RegSet {
        callee_saved_int: model.calls_extern_math(),
    }
}
```

- [ ] **Step 2: Build x86_64**

Run: `cargo build -p profiles-x86_64`
Expected: still errors — `walk_uir` and `walk_model` not yet defined in codegen.rs. Continue to Task 3.5.

### Task 3.5: Implement function prologue/epilogue (asm.rs)

**Files:**
- Modify: `profiles/x86_64/src/asm.rs`

- [ ] **Step 1: Add prologue/epilogue helpers**

Append to `profiles/x86_64/src/asm.rs`:

```rust
use crate::buffer::RegSet;
use profile_api::FnSig;

/// Number of pushes the prologue emits, given the callee-saved set.
/// Used to compute frame-size alignment (see [`compute_frame_size`]).
fn prologue_push_count(regs: RegSet) -> usize {
    // Always: push %rbp (frame pointer).
    let mut n = 1;
    if regs.contains_callee_saved_int() {
        // %rbx, %r12, %r13, %r14, %r15 — 5 additional pushes.
        n += 5;
    }
    n
}

/// Format the function prologue:
///   .globl <prefix><name>
///   .p2align 4, 0x90
///   <prefix><name>:
///       pushq   %rbp
///       movq    %rsp, %rbp
///       [if non-leaf: pushq %rbx; pushq %r12; pushq %r13; pushq %r14; pushq %r15]
///       [if frame_size > 0: subq $frame_size, %rsp]
///
/// `intermediate_bytes` is the total bytes of stack-resident intermediate
/// buffers (from `BufferAssignment::stack_bytes`). The total `frame_size`
/// passed to `subq` includes any alignment correction from
/// [`compute_frame_size`].
pub fn format_function_prologue(
    sig: &FnSig,
    regs: RegSet,
    intermediate_bytes: usize,
    sym_prefix: &str,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(".globl {}{}\n", sym_prefix, sig.name));
    // .p2align 4, 0x90 = 16-byte align with NOP (0x90) padding (gas convention).
    s.push_str(".p2align 4, 0x90\n");
    s.push_str(&format!("{}{}:\n", sym_prefix, sig.name));
    s.push_str("    pushq   %rbp\n");
    s.push_str("    movq    %rsp, %rbp\n");

    if regs.contains_callee_saved_int() {
        s.push_str("    pushq   %rbx\n");
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
        s.push_str("    pushq   %r14\n");
        s.push_str("    pushq   %r15\n");
    }

    let n_pushes = prologue_push_count(regs);
    let frame_size = compute_frame_size(intermediate_bytes as u32, n_pushes);
    if frame_size > 0 {
        s.push_str(&format!("    subq    ${}, %rsp\n", frame_size));
    }
    s
}

/// Symmetric epilogue: restore %rsp, pop callee-saved (reverse order),
/// pop %rbp, ret.
pub fn format_function_epilogue(regs: RegSet, intermediate_bytes: usize) -> String {
    let mut s = String::new();
    let n_pushes = prologue_push_count(regs);
    let frame_size = compute_frame_size(intermediate_bytes as u32, n_pushes);
    if frame_size > 0 {
        s.push_str(&format!("    addq    ${}, %rsp\n", frame_size));
    }
    if regs.contains_callee_saved_int() {
        s.push_str("    popq    %r15\n");
        s.push_str("    popq    %r14\n");
        s.push_str("    popq    %r13\n");
        s.push_str("    popq    %r12\n");
        s.push_str("    popq    %rbx\n");
    }
    s.push_str("    popq    %rbp\n");
    s.push_str("    retq\n");
    s
}
```

- [ ] **Step 2: Build x86_64**

Run: `cargo build -p profiles-x86_64`
Expected: still errors — `codegen.rs` not yet implemented. Continue to Task 3.6.

### Task 3.6: Implement codegen dispatcher (codegen.rs) — bones only

**Files:**
- Modify: `profiles/x86_64/src/codegen.rs`

This task fills in `walk_uir`, `walk_model`, `classify_op`, `resolve_loc`. The per-op emit calls reference `crate::ops::emit_*` which don't exist yet — Tasks 3.7-3.10 fill them. Until those exist, the file fails to compile; that's the intended TDD state.

- [ ] **Step 1: Implement codegen.rs**

Replace contents:

```rust
// SPDX-License-Identifier: Apache-2.0

//! UIR → x86_64 asm walker. Mirror of `profiles/arm64/src/codegen.rs`
//! modulo register naming and instruction set.

use crate::buffer::{assign_buffers, compute_callee_saved, BufferLoc};
use compiler::{NodeId, NodeKind, StdOp, Uir, UirModel};
use profile_api::{Asm, FnSig, LowerError, ParamKind, ParamSlot};

/// Walk the entire UIR, returning the combined asm source + per-model
/// FnSigs. `sym_prefix` threads through to every emitter that produces
/// a profile-prefixed symbol (function label, .globl directive, libm
/// call). For x86_64, `sym_prefix` is `""`.
pub fn walk_uir(uir: &Uir, sym_prefix: &'static str) -> Result<Asm, LowerError> {
    let mut source = String::new();
    let mut functions = Vec::with_capacity(uir.models.len());

    for (model_idx, model) in uir.models.iter().enumerate() {
        let (model_asm, sig) = walk_model(model_idx, model, sym_prefix)?;
        source.push_str(&model_asm);
        source.push('\n');
        functions.push(sig);
    }

    Ok(Asm { source, functions })
}

fn walk_model(
    model_idx: usize,
    model: &UirModel,
    sym_prefix: &'static str,
) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue};
    use compiler::PostOp;

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 2. Compute layout, ABI sizes.
    let input_id = *model.inputs.first().ok_or(LowerError::ShapeNotConcrete {
        span: model.source_span,
    })?;
    let input_floats: usize =
        model.nodes[input_id].ty.shape.0.iter().product::<u64>() as usize;
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op {
            op: StdOp::Linear,
            operands,
            attrs,
            ..
        } = &node.kind
        {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete {
                    span: node.source_span,
                });
            }
            let k = in_shape.0[1] as usize;
            let n = out_shape.0[1] as usize;
            params_layout.push(ParamSlot {
                kind: ParamKind::LinearWeight,
                origin_node: node_idx,
                offset: params_floats,
                size: k * n,
            });
            params_floats += k * n;
            if compiler::ir::linear_has_bias(attrs) {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        output_floats,
        params_floats,
        params_layout,
    };

    // 3. Buffer assignment + callee-saved set.
    let assignment = assign_buffers(model);
    let regs = compute_callee_saved(model);

    // 4. Emit prologue + body + epilogue.
    let mut body = String::new();
    body.push_str(&format_function_prologue(
        &sig,
        regs,
        assignment.stack_bytes,
        sym_prefix,
    ));

    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    let mut softmax_idx = 0usize;
    let mut dropout_idx = 0usize;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op, operands, .. } = &node.kind {
            match op {
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    let weight_offset = sig
                        .params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    let bias_offset = sig
                        .params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearBias && s.origin_node == node_idx)
                        .map(|s| s.offset);

                    let NodeKind::Op { fused_post_ops, .. } = &node.kind else {
                        unreachable!("walk_model already matched NodeKind::Op")
                    };

                    body.push_str(&crate::ops::emit_linear(
                        b, k, n,
                        model_idx, linear_idx,
                        src_loc, dst_loc,
                        weight_offset, bias_offset,
                        node.source_span,
                        fused_post_ops,
                        sym_prefix,
                    )?);
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_relu(
                        total, model_idx, relu_idx, src_loc, dst_loc,
                    ));
                    relu_idx += 1;
                }
                StdOp::Dropout => {
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    if matches!(dst_loc, BufferLoc::OutputReg) {
                        let total: u64 = node.ty.shape.0.iter().product();
                        body.push_str(&crate::ops::emit_dropout_copy(
                            total, model_idx, dropout_idx, src_loc, dst_loc,
                        ));
                        dropout_idx += 1;
                    }
                    // else BufferLoc::Alias: no asm — downstream reads operand directly.
                }
                StdOp::Softmax => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_softmax(
                        b, k, model_idx, softmax_idx, src_loc, dst_loc, sym_prefix,
                    ));
                    softmax_idx += 1;
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(LowerError::UnsupportedOp {
                        op: format!("{op}"),
                        span: node.source_span,
                    });
                }
            }
            // Suppress unused-variable warnings for fused_post_ops in the
            // non-Linear arms; matched per-arm above.
            let _ = PostOp::Relu;
        }
    }

    body.push_str(&format_function_epilogue(regs, assignment.stack_bytes));
    Ok((body, sig))
}

/// Resolve `Alias` chains to a concrete BufferLoc.
fn resolve_loc(locs: &[BufferLoc], id: NodeId) -> BufferLoc {
    let mut cur = id;
    loop {
        match locs[cur] {
            BufferLoc::Alias(next) => {
                debug_assert!(next < cur, "alias must point backward (cycle defense)");
                cur = next;
            }
            other => return other,
        }
    }
}

/// Validate that an op is supported.
fn classify_op(
    op: StdOp,
    _attrs: &[compiler::OpAttr],
    span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => Ok(()),
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Ok(()),
        StdOp::Softmax => Ok(()),
        #[allow(unreachable_patterns)]
        _ => Err(LowerError::UnsupportedOp {
            op: format!("{op}"),
            span,
        }),
    }
}
```

- [ ] **Step 2: Implement `ops/mod.rs` (re-exports)**

Replace contents of `profiles/x86_64/src/ops/mod.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Per-op codegen modules.

pub mod dropout;
pub mod linear;
pub mod relu;
pub mod softmax;

pub use dropout::emit_dropout_copy;
pub use linear::emit_linear;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
```

- [ ] **Step 3: Build (will still fail — emitters empty)**

Run: `cargo build -p profiles-x86_64 2>&1 | head -40`
Expected: errors of the form `cannot find function 'emit_relu' in module 'crate::ops'` etc. **This is intended.** Continue to Tasks 3.7-3.10.

### Task 3.7: Implement `emit_relu` (TDD red→green)

**Files:**
- Modify: `profiles/x86_64/src/ops/relu.rs`
- Modify: `profiles/x86_64/src/tests.rs` (add 3 shape-asserts)

- [ ] **Step 1: Write 3 failing tests for emit_relu**

Append to `profiles/x86_64/src/tests.rs`:

```rust
use compiler::ir;
use compiler::passes;

fn lower_x86(src: &str) -> profile_api::Asm {
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let uir = passes::run_pipeline(&uir, &passes::default_pipeline()).expect("pipeline");
    crate::lower(&uir).expect("lower")
}

fn lower_x86_no_passes(src: &str) -> profile_api::Asm {
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    crate::lower(&uir).expect("lower")
}

#[test]
fn relu_emits_separate_loop_with_xorps_and_maxss() {
    // Use --no-passes path so relu stays as a separate node (the default
    // pipeline fuses linear→relu and inlines the maxss inside the matmul).
    let src = "model R [b=4, k=8]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("xorps   %xmm1, %xmm1"),
        "relu must zero a scratch xmm via xorps:\n{s}"
    );
    assert!(
        s.contains("maxss   %xmm1, %xmm0"),
        "relu must compare against zero via maxss:\n{s}"
    );
    assert!(
        s.contains(".Lrelu_"),
        "relu must emit a labelled loop:\n{s}"
    );
}

#[test]
fn function_label_has_no_underscore_prefix_on_x86_64() {
    let src = "model M [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains(".globl nfl_forward_M\n"),
        "x86_64 ELF must NOT prepend underscore to .globl:\n{s}"
    );
    assert!(
        s.contains("\nnfl_forward_M:"),
        "x86_64 ELF must NOT prepend underscore to function label:\n{s}"
    );
    assert!(
        !s.contains("_nfl_forward_M"),
        "x86_64 ELF must not have any '_nfl_' (Mach-O convention):\n{s}"
    );
}

#[test]
fn relu_only_model_is_leaf_no_callee_saved_int_pushes() {
    let src = "model L [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("    pushq   %rbp\n"),
        "frame pointer always saved:\n{s}"
    );
    assert!(
        !s.contains("    pushq   %rbx\n"),
        "leaf model must NOT save callee-saved int regs:\n{s}"
    );
}
```

- [ ] **Step 2: Run; expect failures**

Run: `cargo test -p profiles-x86_64 --lib relu_ function_label_ leaf_no_callee_saved`
Expected: compile errors → `emit_relu` not yet defined.

- [ ] **Step 3: Implement `emit_relu` in `profiles/x86_64/src/ops/relu.rs`**

Replace contents:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Relu (elementwise max with zero) codegen — x86_64 SSE2.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for an elementwise ReLU.
///
/// `model_idx` + `relu_idx` together uniquely name every label across all
/// models emitted into a single assembly file (multi-model fixtures like
/// `pipeline_styles.nfl` would otherwise collide on `.Lrelu_0` etc.).
///
/// Register usage:
///   %r10 (= scratch dim immediate; we use it for total_floats)
///   %r11 (= dst pointer)
///   %rcx (= loop counter)
///   %xmm0 (= scratch float — element)
///   %xmm1 (= scratch float — zero)
///   The src ptr lives in %r10 *only* until total_floats is loaded; we
///   reuse a separate scratch (%r8) for src to avoid clobber.
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
        "    # relu: copy-clamp src→dst ({total_floats} elements)\n"
    ));
    // Materialise source ptr into %r8 (caller-saved scratch; no call site
    // in this leaf-only emitter, so no need for callee-saved).
    s.push_str(&materialise_ptr("%r8", src_loc));
    s.push_str(&materialise_ptr("%r11", dst_loc));
    s.push_str("    xorps   %xmm1, %xmm1\n");
    s.push_str(&emit_imm32_to_r10(total_floats as u32));
    s.push_str("    xorq    %rcx, %rcx\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lrelu_end_{rid}\n"));
    s.push_str("    movss   (%r8, %rcx, 4), %xmm0\n");
    s.push_str("    maxss   %xmm1, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rcx, 4)\n");
    s.push_str("    incq    %rcx\n");
    s.push_str(&format!("    jmp     .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p profiles-x86_64 --lib relu_ function_label_ leaf_no_callee_saved`
Expected: 3 passing.

### Task 3.8: Implement `emit_dropout_copy`

**Files:**
- Modify: `profiles/x86_64/src/ops/dropout.rs`
- Modify: `profiles/x86_64/src/tests.rs` (add 1 shape-assert)

- [ ] **Step 1: Write the test first**

Append to `profiles/x86_64/src/tests.rs`:

```rust
#[test]
fn dropout_as_output_emits_copy_loop_no_maxss() {
    // dropout-as-output (model.output is the dropout node) triggers
    // emit_dropout_copy via the BufferLoc::OutputReg branch in walk_model.
    let src = "model OnlyDropout [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains(".Ldropout_"), "missing dropout loop label:\n{s}");
    assert!(s.contains("movss   (%r8, %rcx, 4), %xmm0"), "missing load:\n{s}");
    assert!(s.contains("movss   %xmm0, (%r11, %rcx, 4)"), "missing store:\n{s}");
    assert!(!s.contains("maxss"), "dropout-copy must NOT contain maxss:\n{s}");
}
```

- [ ] **Step 2: Run; expect failure**

Run: `cargo test -p profiles-x86_64 --lib dropout_as_output`
Expected: compile error.

- [ ] **Step 3: Implement `emit_dropout_copy`**

Replace contents of `profiles/x86_64/src/ops/dropout.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Dropout codegen.
//!
//! At inference, dropout is identity. Buffer assignment returns
//! `BufferLoc::Alias(operand)` for non-output dropouts (no asm emitted —
//! downstream ops read from the operand's buffer directly). When the
//! dropout is the model output, `walk_model` calls `emit_dropout_copy`
//! to copy the operand buffer into the caller's output buffer.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for a dropout-as-output copy loop.
///
/// Mirror of `emit_relu`'s structure minus the zero-init and `maxss`:
/// element-wise load → store, no transformation.
pub fn emit_dropout_copy(
    total_floats: u64,
    model_idx: usize,
    dropout_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    debug_assert!(
        matches!(dst_loc, BufferLoc::OutputReg),
        "emit_dropout_copy only valid for OutputReg dst (caller guards in walk_model)"
    );
    let did = format!("{model_idx}_{dropout_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # dropout-as-output: copy operand→output ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("%r8", src_loc));
    s.push_str(&materialise_ptr("%r11", dst_loc));
    s.push_str(&emit_imm32_to_r10(total_floats as u32));
    s.push_str("    xorq    %rcx, %rcx\n");
    s.push_str(&format!(".Ldropout_{did}:\n"));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Ldropout_end_{did}\n"));
    s.push_str("    movss   (%r8, %rcx, 4), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rcx, 4)\n");
    s.push_str("    incq    %rcx\n");
    s.push_str(&format!("    jmp     .Ldropout_{did}\n"));
    s.push_str(&format!(".Ldropout_end_{did}:\n"));
    s
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p profiles-x86_64 --lib dropout_as_output`
Expected: 1 passing.

### Task 3.9: Implement `emit_linear` (matmul + bias + fused PostOps)

**Files:**
- Modify: `profiles/x86_64/src/ops/linear.rs`
- Modify: `profiles/x86_64/src/tests.rs` (add ~6 shape-asserts covering: plain matmul, +bias, +ReluFused inline, +SoftmaxRow tail)

`emit_linear` is the largest emitter. It mirrors `profiles/arm64/src/ops/linear.rs` line-by-line: outer i-loop over batch, inner j-loop over output dim, innermost k-loop accumulates `xmm0 = sum`. Bias-add (if present) and elementwise PostOps (currently `Relu`) are inlined inside the j-loop after the k-loop closes; row-wise PostOps (`SoftmaxRow`) emit a 3-pass tail after the i-loop closes.

**Register allocation contract for emit_linear** (pinned):
- `%r8` = src ptr (scratch; loaded via `materialise_ptr` at top)
- `%r11` = dst ptr (scratch; same)
- `%r9` = weight base ptr (= `%rsi + weight_offset*4` via `leaq`)
- `%r12` (when bias present, scratch use) = bias base ptr
  - **Important:** if `SoftmaxRow` is fused, `%r12` is in the callee-saved set and used by the row-wise tail; for a fused linear+softmax we only need `%r12` for the matmul body's bias-add IF we're also doing softmax. To avoid conflict, when both bias AND SoftmaxRow are fused, the bias ptr lives in `%r13` instead. For non-softmax paths (bias-only or relu-fused), use `%r12` freely.
- `%rcx` = innermost k-counter (caller-saved; safe inside k-loop)
- `%r10` = scratch immediate (n, k, b dimensions reloaded as needed)
- `%xmm0` = running sum (`mulss` + `addss` accumulator pattern; no FMA per spec §3 non-goal "no SIMD/AVX/FMA")
- `%xmm1` = element load (multiplicand)

**Algorithmic structure** (translate arm64::ops::linear to x86_64):
1. Materialise `%r8` ← src ptr; `%r11` ← dst ptr.
2. `leaq weight_offset*4(%rsi), %r9` — weight base.
3. If bias: `leaq bias_offset*4(%rsi), %r12` (or `%r13` per the contract above).
4. Outer i-loop: `%rax = i`, compare against `b`; jump to end when `i >= b`.
5. Inner j-loop: `%rdi`-not-usable-because-it's-input, use `%rbx` if available (callee-saved when softmax), or another scratch... wait actually we have to be more careful here. For non-softmax fused linear, the int regs we have free are: `%rax, %rcx, %rdi (input ptr — DO NOT clobber), %rsi (params ptr — DO NOT clobber), %rdx (output ptr — DO NOT clobber), %r8, %r9, %r10, %r11`. That's 6 free scratch regs (`%rax, %rcx, %r8, %r9, %r10, %r11`) plus we need to preserve src/dst/weight/bias pointers.

To keep this tractable, the plan adopts a **fixed assignment**:
- `%r8` = src ptr (preserved across loops)
- `%r11` = dst ptr (preserved)
- `%r9` = weight base ptr (preserved)
- `%r12` = bias base ptr (preserved; in callee-saved set whenever softmax also fused, else used as scratch only by emit_linear and saved/restored locally — but emit_linear doesn't call anything, so caller-saved is fine; we just assume `%r12` is unused by anyone else in the same function. **Easier approach:** put bias base in `%rax` and reload it each j-iteration — `%rax` is caller-saved scratch.)

To minimise risk, the plan picks the **simplest robust pattern**: bias base reloaded from `%rsi + bias_offset*4` into `%rax` at the top of every j-iteration. This adds one `leaq` per (i,j) but eliminates register-conflict reasoning. The cost is acceptable because the matmul k-loop dominates runtime.

- `%rax` = bias base ptr (rematerialised each j-iter)
- `%rdx` (output) — DO NOT clobber for inner work; use `%r11` (= preloaded dst) for stores.

WAIT — `%rdx` IS the FFI arg 2 (output). And `%rdx` is one of the SysV scratch regs that `call expf` clobbers. So `%r11` ← `%rdx` materialisation must be at function entry, before any `call`. For `emit_linear` (which is leaf in non-softmax case), no `call` happens, so `%rdx` itself stays valid throughout. But we still use `%r11` for consistency with the contract.

**Loop counters and i/j/k:**
- `%rax` = i (outer)
- `%rdi` is INPUT pointer — cannot use as i counter. Use `%rax` instead.
- `%rcx` = j (middle) — but we also use `%rcx` for k. Rename: use `%rdi`-no-NO. Use `%rsi`-NO (params).

OK this register-pressure is real. Final assignment:
- `%rax` = i (outer counter)
- `%rcx` = j (middle counter)
- `%r10d` = k (inner counter — using lower 32 bits of `%r10`, treating as `%r10`; but `%r10` also holds dim immediates via `emit_imm32_to_r10`. To avoid conflict, the dim immediates that span the i-loop (b, n, k) are loaded ONCE before the loops and pinned in DIFFERENT regs — the plan puts them in `%r13, %r14, %r15` IF non-softmax; if softmax-fused, those regs are reserved for the softmax tail. **Simplest robust choice:** rematerialise the dim immediate at every cmp site. Cost is one extra `movl $imm, %r10d` per iteration (~3 per matmul body iteration → 3*B*N*K total movs). For typical models (B=32, N=10, K=784: 1M movs spread across 250k iterations) this is single-digit microseconds; acceptable.

OK given how much register-allocation reasoning this requires, the TASK should not specify the full asm in the plan — it would be 200 lines of asm-emit Rust code in the plan, brittle to trivial register-naming changes. Instead, the task gives the engineer:
1. The algorithmic shape
2. The instruction-table (§7.3)
3. A reference-by-mirror to `profiles/arm64/src/ops/linear.rs` for the algorithmic skeleton
4. The complete tests (which pin the asm shape via substring asserts)
5. A worked example for the simplest case (matmul-only no-bias no-postop)

The engineer translates arm64's emit_linear to x86_64 by:
- arm64 `mov xN, #imm` → x86_64 `movq $imm, %rN`
- arm64 `cmp xN, xM; b.ge .L` → x86_64 `cmpq %rM, %rN; jge .L` (note operand order flip — AT&T)
- arm64 `mul xR, xA, xB` → x86_64 `movq %rA, %rR; imulq %rB, %rR`
- arm64 `add xR, xA, xB` → x86_64 `movq %rA, %rR; addq %rB, %rR` (or use `leaq` for non-flag-modifying add)
- arm64 `ldr sN, [xPtr, xOff, lsl #2]` → x86_64 `movss (%rPtr, %rOff, 4), %xmmN`
- arm64 `str sN, [xPtr, xOff, lsl #2]` → x86_64 `movss %xmmN, (%rPtr, %rOff, 4)`
- arm64 `fmadd s0, s1, s2, s0` → x86_64 `mulss %xmm2, %xmm1; addss %xmm1, %xmm0` (no FMA)
- arm64 `fadd s0, s0, s5` → x86_64 `addss %xmm5, %xmm0`
- arm64 `fmax s0, s0, s4` → x86_64 `maxss %xmm4, %xmm0`
- arm64 `fmov s4, wzr` → x86_64 `xorps %xmm4, %xmm4`
- arm64 `fsub s0, s0, s8` → x86_64 `subss %xmm8, %xmm0`

- [ ] **Step 1: Write 6 failing shape-asserts in `profiles/x86_64/src/tests.rs`**

```rust
// matmul-only, no bias, no fused post-op (use --no-passes path with relu, drop relu)
#[test]
fn linear_matmul_emits_mulss_addss_pair_no_fma() {
    let src = "model L [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[output=n]\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("mulss"), "matmul body needs mulss:\n{s}");
    assert!(s.contains("addss"), "matmul body needs addss (no FMA):\n{s}");
    assert!(!s.contains("vfmadd"), "must NOT use FMA — scalar SSE2 only:\n{s}");
}

#[test]
fn linear_with_bias_emits_addss_from_bias_buffer() {
    let src = "model B [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[output=n, bias=true]\n";
    let s = lower_x86_no_passes(src).source;
    // Bias add: load bias element + addss into %xmm0.
    assert!(s.contains("addss"), "bias-add via addss:\n{s}");
}

#[test]
fn linear_relu_fused_emits_inline_maxss_no_separate_loop() {
    // Default pipeline fuses linear→relu — inline maxss inside matmul body.
    let src = "model F [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[output=n] -> relu\n";
    let s = lower_x86(src).source;
    assert!(s.contains("maxss"), "fused relu must inline maxss:\n{s}");
    assert!(!s.contains(".Lrelu_"), "fused asm should NOT have separate relu loop:\n{s}");
}

#[test]
fn linear_softmax_fused_emits_row_wise_tail_with_call_expf_plt() {
    let src = "model S [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[output=n] -> softmax\n";
    let s = lower_x86(src).source;
    assert!(s.contains(".Lfsmx_"), "fused softmax tail uses .Lfsmx_ labels:\n{s}");
    assert!(s.contains("call    expf@PLT"), "fused softmax tail must call expf@PLT:\n{s}");
}

#[test]
fn linear_softmax_fused_uses_callee_saved_int_pushes() {
    let src = "model SC [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[output=n] -> softmax\n";
    let s = lower_x86(src).source;
    assert!(s.contains("    pushq   %rbx\n"), "softmax fused needs callee-saved %rbx:\n{s}");
    assert!(s.contains("    pushq   %r12\n"), "softmax fused needs callee-saved %r12:\n{s}");
    assert!(s.contains("    pushq   %r15\n"), "softmax fused needs callee-saved %r15:\n{s}");
}

#[test]
fn linear_matmul_uses_only_scalar_sse2_xmm_regs() {
    let src = "model V [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[output=n]\n";
    let s = lower_x86_no_passes(src).source;
    // Scalar SSE2: xmm0..xmm15 — no ymm/zmm.
    assert!(!s.contains("%ymm"), "no AVX (ymm) per spec non-goals:\n{s}");
    assert!(!s.contains("%zmm"), "no AVX-512 (zmm) per spec non-goals:\n{s}");
}
```

- [ ] **Step 2: Run; expect compile failure**

Run: `cargo test -p profiles-x86_64 --lib linear_`
Expected: compile errors — `emit_linear` not yet defined.

- [ ] **Step 3: Implement `emit_linear` in `profiles/x86_64/src/ops/linear.rs`**

The function signature mirrors arm64's plus a `sym_prefix: &str` for the fused-softmax tail's `call expf@PLT`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Linear (matmul + optional bias + fused PostOps) codegen — x86_64 SSE2.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use compiler::PostOp;
use profile_api::LowerError;

#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    model_idx: usize,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
    node_span: Span,
    fused_post_ops: &[PostOp],
    sym_prefix: &str,
) -> Result<String, LowerError> {
    let lid = format!("{model_idx}_{linear_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}{}\n",
        if bias_offset.is_some() { " + bias" } else { "" },
        if !fused_post_ops.is_empty() { " + fused" } else { "" },
    ));

    // 1. Pointer setup.
    s.push_str(&materialise_ptr("%r8", src_loc));   // src ptr
    s.push_str(&materialise_ptr("%r11", dst_loc));  // dst ptr

    // weight base = %rsi + weight_offset*4
    if weight_offset == 0 {
        s.push_str("    movq    %rsi, %r9\n");
    } else {
        s.push_str(&format!(
            "    leaq    {}(%rsi), %r9\n",
            weight_offset * 4
        ));
    }
    // bias base — optional. Held in callee-saved %r12 across the loops
    // so it's available inside every j-iteration without reload. (%r12
    // is in the prologue's callee-saved set whenever softmax is fused;
    // for plain bias-only linears it's still safe because emit_linear
    // is leaf and the caller's %r12 is preserved by us anyway.)
    let needs_zero_xmm4 = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
    if needs_zero_xmm4 {
        s.push_str("    xorps   %xmm4, %xmm4\n");
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str("    movq    %rsi, %r12\n");
        } else {
            s.push_str(&format!("    leaq    {}(%rsi), %r12\n", boff * 4));
        }
    }

    // 2. Outer i-loop: %rax = i, compared against b.
    s.push_str("    xorq    %rax, %rax\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lmm_i_end_{lid}\n"));

    // 3. Inner j-loop: %rcx = j, compared against n.
    s.push_str("    xorq    %rcx, %rcx\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm_j_end_{lid}\n"));

    // 4. Innermost k-loop: %xmm0 = sum, %rdx-can't-use (it's output ptr).
    //    Use %r13 as k-counter ONLY when softmax not fused (else %r13 is
    //    reserved for softmax tail). For fused-softmax, k-counter must
    //    use a non-callee-saved scratch. Choose %rdi? — that's input
    //    pointer. Choose unused: actually %r10 holds the dim immediate;
    //    we need a dedicated k-counter. Use %r14 (callee-saved when
    //    softmax fused — saved by prologue; safe to use as scratch
    //    inside the matmul body in either case).
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str("    xorps   %xmm0, %xmm0\n"); // sum init
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lmm_k_end_{lid}\n"));

    // src offset = i*k + kk; load src[i*k + kk] → xmm1
    // i*k: leaq doesn't multiply non-power-of-two; use imulq.
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %rax, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");          // %r15 = i * k
    s.push_str("    addq    %r14, %r15\n");          // %r15 = i*k + kk
    s.push_str("    movss   (%r8, %r15, 4), %xmm1\n"); // xmm1 = src[i*k + kk]

    // weight offset = kk*n + j; load weights[kk*n + j] → xmm2
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %r14, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");          // %r15 = kk * n
    s.push_str("    addq    %rcx, %r15\n");          // %r15 = kk*n + j
    s.push_str("    movss   (%r9, %r15, 4), %xmm2\n");

    // sum += xmm1 * xmm2  (no FMA)
    s.push_str("    mulss   %xmm2, %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");

    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // 5. Bias-add (if present): xmm0 += bias[j].
    if bias_offset.is_some() {
        s.push_str("    movss   (%r12, %rcx, 4), %xmm5\n");
        s.push_str("    addss   %xmm5, %xmm0\n");
    }

    // 6. Elementwise post-ops: applied inline inside the j-loop.
    //    Row-wise post-ops (SoftmaxRow) skipped here; emitted after the
    //    matmul loop completes.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => s.push_str("    maxss   %xmm4, %xmm0\n"),
            PostOp::SoftmaxRow => {} // row-wise; handled after the matmul.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: post_op.to_string(),
                    span: node_span,
                });
            }
        }
    }

    // 7. Store xmm0 → dst[i*n + j]
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %rax, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");
    s.push_str("    addq    %rcx, %r15\n");
    s.push_str("    movss   %xmm0, (%r11, %r15, 4)\n");

    s.push_str("    incq    %rcx\n");
    s.push_str(&format!("    jmp     .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    // 8. Row-wise post-ops (SoftmaxRow tail) run after the matmul loop.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => {} // already inlined above
            PostOp::SoftmaxRow => {
                s.push_str(&emit_fused_softmax_tail(b, n, &lid, sym_prefix));
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: post_op.to_string(),
                    span: node_span,
                });
            }
        }
    }

    Ok(s)
}

/// Fused-softmax row-wise tail. Operates in-place on dst[%r11].
///
/// Register contract (callee-saved by prologue's
/// `compute_callee_saved` whenever this emitter fires):
///   %rbx = src ptr (= %r11; same buffer for in-place)
///   %r12 = dst ptr (= %r11)
///   %r13 = i (outer row counter)
///   %r14 = j (inner column counter)
///   %r15 = row_base = i * n
///
/// Stack-resident state across `call expf@PLT`:
///   8(%rsp)  = row_max f32 slot (`max_slot_off` = 8)
///   16(%rsp) = row_sum f32 slot (`sum_slot_off` = 16)
/// (within the function's `frame_size`; the prologue's `subq $frame_size,
/// %rsp` reserved space includes these slots.)
fn emit_fused_softmax_tail(b: u64, n: u64, lid: &str, sym_prefix: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "    # fused softmax_row: output [{b},{n}] in-place\n"
    ));
    s.push_str("    movq    %r11, %rbx\n");  // src = dst (in-place)
    s.push_str("    movq    %r11, %r12\n");  // dst = dst

    // Outer per-row loop: %r13 = i.
    s.push_str("    xorq    %r13, %r13\n");
    s.push_str(&format!(".Lfsmx_i_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r13\n");
    s.push_str(&format!("    jge     .Lfsmx_i_end_{lid}\n"));

    // %r15 = i * n
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %r13, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");

    // Phase 2: row_max → [8(%rsp)]. Init xmm8 to -inf.
    s.push_str("    movl    $0xFF800000, %r10d\n"); // -inf bits
    s.push_str("    movd    %r10d, %xmm8\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_max_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_max_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n"); // %rax = row_base + j
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    maxss   %xmm0, %xmm8\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_max_{lid}\n"));
    s.push_str(&format!(".Lfsmx_max_end_{lid}:\n"));
    // Spill row_max to stack (xmm regs are caller-saved across call).
    s.push_str("    movss   %xmm8, 8(%rsp)\n");

    // Phase 3: exp(x − max), sum → [16(%rsp)]. Init sum slot to 0.
    s.push_str("    movl    $0, 16(%rsp)\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_exp_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_exp_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n"); // %rax = row_base + j
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    subss   8(%rsp), %xmm0\n");
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax was clobbered; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n"); // write exp result back
    s.push_str("    movss   16(%rsp), %xmm1\n");
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str("    movss   %xmm1, 16(%rsp)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_exp_{lid}\n"));
    s.push_str(&format!(".Lfsmx_exp_end_{lid}:\n"));

    // Phase 4: normalise by row_sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_norm_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_norm_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%r12, %rax, 4), %xmm0\n");
    s.push_str("    divss   16(%rsp), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_norm_{lid}\n"));
    s.push_str(&format!(".Lfsmx_norm_end_{lid}:\n"));

    // Next row.
    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lfsmx_i_{lid}\n"));
    s.push_str(&format!(".Lfsmx_i_end_{lid}:\n"));
    s
}
```

> **Stack-slot budget note:** the fused softmax tail uses `8(%rsp)` and `16(%rsp)` as f32 spill slots. These addresses are valid when `frame_size >= 16` (so the bytes are within the reserved frame). For models with empty intermediate-buffer footprint but softmax fused, `compute_frame_size(0, num_pushes=6) = 8` — only 8 bytes available, which is enough for `[8(%rsp)]` but not `[16(%rsp)]`. **Fix:** in `walk_model`, when `model.calls_extern_math()`, increase the reported `intermediate_bytes` by 16 (two 8-byte-aligned f32 slots) before passing to `format_function_prologue`. This is a per-model bump, not a per-emit budget. Add this adjustment in Task 3.6's `walk_model` body — but since 3.6 is already implemented, **add it now**: edit `profiles/x86_64/src/codegen.rs::walk_model` to add a `let intermediate_bytes = if model.calls_extern_math() { assignment.stack_bytes + 16 } else { assignment.stack_bytes };` and pass `intermediate_bytes` to `format_function_prologue` and `format_function_epilogue`.

- [ ] **Step 4: Apply the stack-slot adjustment in codegen.rs**

In `profiles/x86_64/src/codegen.rs::walk_model`, replace:

```rust
    body.push_str(&format_function_prologue(
        &sig,
        regs,
        assignment.stack_bytes,
        sym_prefix,
    ));
```

with:

```rust
    // Reserve two extra 8-byte stack slots for the fused-softmax xmm-spill
    // (row_max at 8(%rsp), row_sum at 16(%rsp)) whenever the model calls
    // libm-expf. See profiles/x86_64/src/ops/linear.rs::emit_fused_softmax_tail.
    let intermediate_bytes = if model.calls_extern_math() {
        assignment.stack_bytes + 16
    } else {
        assignment.stack_bytes
    };
    body.push_str(&format_function_prologue(
        &sig,
        regs,
        intermediate_bytes,
        sym_prefix,
    ));
```

And similarly for `format_function_epilogue`:
```rust
    body.push_str(&format_function_epilogue(regs, intermediate_bytes));
```

- [ ] **Step 5: Run linear tests**

Run: `cargo test -p profiles-x86_64 --lib linear_`
Expected: 6 passing.

### Task 3.10: Implement `emit_softmax` (standalone 3-pass)

**Files:**
- Modify: `profiles/x86_64/src/ops/softmax.rs`
- Modify: `profiles/x86_64/src/tests.rs` (add ~5 shape-asserts)

`emit_softmax` is the standalone (non-fused) softmax emitter. The fused tail (`emit_fused_softmax_tail`, inside `linear.rs`) already covers the most architecturally divergent path; the standalone emitter is structurally similar — just operates on a separate src→dst pair instead of dst-in-place.

**Register contract** (same as fused tail):
- `%rbx` = src ptr (callee-saved)
- `%r12` = dst ptr (callee-saved)
- `%r13` = i, `%r14` = j, `%r15` = row_base
- Stack slots: `[8(%rsp)]` row_max, `[16(%rsp)]` row_sum

- [ ] **Step 1: Write 5 failing tests**

```rust
#[test]
fn standalone_softmax_emits_three_pass_with_call_expf_plt() {
    let src = "model SS [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains(".Lsm_max_"), "phase 1 max label missing:\n{s}");
    assert!(s.contains(".Lsm_exp_"), "phase 2 exp label missing:\n{s}");
    assert!(s.contains(".Lsm_norm_"), "phase 3 norm label missing:\n{s}");
    assert!(s.contains("call    expf@PLT"), "softmax must call expf@PLT:\n{s}");
}

#[test]
fn standalone_softmax_uses_callee_saved_int_pushes() {
    let src = "model SCS [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("    pushq   %rbx\n"), "softmax needs %rbx callee-saved:\n{s}");
    assert!(s.contains("    pushq   %r15\n"), "softmax needs %r15 callee-saved:\n{s}");
}

#[test]
fn standalone_softmax_spills_max_to_stack_at_offset_8() {
    let src = "model SP [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("movss   %xmm8, 8(%rsp)"), "row_max spill missing:\n{s}");
}

#[test]
fn standalone_softmax_initialises_sum_slot_to_zero() {
    let src = "model SZ [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("movl    $0, 16(%rsp)"), "sum slot init missing:\n{s}");
}

#[test]
fn standalone_softmax_recomputes_offset_after_call() {
    // After call expf@PLT, the offset-holding GPR (%rax) is clobbered;
    // emitter must recompute before the next memory access.
    let src = "model SR [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    let post_call_idx = s.find("call    expf@PLT").expect("must contain call");
    let post_call = &s[post_call_idx..];
    assert!(
        post_call.contains("movq    %r15, %rax"),
        "must recompute %rax = row_base after call expf@PLT:\n{s}"
    );
}
```

- [ ] **Step 2: Run; expect failure**

Run: `cargo test -p profiles-x86_64 --lib softmax_`
Expected: compile errors.

- [ ] **Step 3: Implement `emit_softmax`**

Replace contents of `profiles/x86_64/src/ops/softmax.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Softmax (per-row stable, libm expf via PLT) codegen — x86_64 SSE2.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// Calls `<sym_prefix>expf@PLT` for each element. State across the call
/// lives in callee-saved int registers (%rbx, %r12, %r13, %r14, %r15)
/// and on the stack (`[8(%rsp)]` row_max, `[16(%rsp)]` row_sum); see
/// the register contract in profiles/x86_64/src/ops/linear.rs.
pub fn emit_softmax(
    b: u64,
    k: u64,
    model_idx: usize,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    sym_prefix: &str,
) -> String {
    let sid = format!("{model_idx}_{softmax_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"
    ));

    // Pin src/dst into callee-saved %rbx/%r12 (survives `call expf@PLT`).
    s.push_str(&materialise_ptr("%rbx", src_loc));
    s.push_str(&materialise_ptr("%r12", dst_loc));

    // Outer per-row loop: %r13 = i.
    s.push_str("    xorq    %r13, %r13\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r13\n");
    s.push_str(&format!("    jge     .Lsm_i_end_{sid}\n"));

    // %r15 = i * k
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %r13, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");

    // Phase 1: row_max → [8(%rsp)]. Init xmm8 to -inf.
    s.push_str("    movl    $0xFF800000, %r10d\n");
    s.push_str("    movd    %r10d, %xmm8\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_max_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    maxss   %xmm0, %xmm8\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));
    s.push_str("    movss   %xmm8, 8(%rsp)\n");

    // Phase 2: exp(x - max) → dst, sum → [16(%rsp)]. Init sum to 0.
    s.push_str("    movl    $0, 16(%rsp)\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_exp_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    subss   8(%rsp), %xmm0\n");
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax clobbered by call; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    movss   16(%rsp), %xmm1\n");
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str("    movss   %xmm1, 16(%rsp)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Phase 3: normalise by row_sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_norm_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%r12, %rax, 4), %xmm0\n");
    s.push_str("    divss   16(%rsp), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    s
}
```

- [ ] **Step 4: Run softmax tests**

Run: `cargo test -p profiles-x86_64 --lib softmax_`
Expected: 5 passing.

### Task 3.11: Mirror remaining arm64 unit-test coverage

**Files:**
- Modify: `profiles/x86_64/src/tests.rs` (add ~25 more tests to reach ~45 total parity)

The arm64 unit-test suite (`profiles/arm64/src/tests.rs`, 45 tests, 122 substring asserts) covers: function-shape, prologue+epilogue, multi-model output, classifier-kind shape, fixture-end-to-end shape, fused-vs-unfused asm differences, dropout-specific cases, large-dim cases. Each x86_64 unit test is a structural mirror.

- [ ] **Step 1: Read all arm64 unit tests**

```bash
wc -l profiles/arm64/src/tests.rs   # ~777 lines, 45 tests
grep -c "^#\[test\]" profiles/arm64/src/tests.rs  # 45
```

Confirm: 45 tests in `profiles/arm64/src/tests.rs`.

- [ ] **Step 2: For each arm64 test, add an x86_64 mirror**

Skip the tests already added in Tasks 3.7-3.10 (those covered relu, dropout, linear, softmax, function-label-prefix). For the remaining ~30 arm64 tests, mirror the substring asserts using the arm64→x86_64 instruction-translation table from Task 3.9. Examples:

- arm64: `assert!(s.contains("cmp x9, x10"));` → x86_64: `assert!(s.contains("cmpq    %r10, %rcx"));`
- arm64: `assert!(s.contains("fmadd s0, s1, s2, s0"));` → x86_64: two asserts, one per `mulss` and `addss`
- arm64: `assert!(s.contains("bl      _expf"));` → x86_64: `assert!(s.contains("call    expf@PLT"));`
- arm64: `assert!(s.contains("_nfl_forward_M"));` → x86_64: `assert!(s.contains("nfl_forward_M:")) && !s.contains("_nfl_")`
- arm64: `assert!(s.contains("fmov s0, wzr"));` → x86_64: `assert!(s.contains("xorps   %xmm0, %xmm0"));`
- arm64 callee-saved checks: `s.contains("stp d8, d9, [sp, #-16]!")` → x86_64: `s.contains("    pushq   %rbx\n")` etc.

**Skip pure arm64 tests with no x86_64 analog**: any test asserting AArch64-specific imm-handling (movz/movk dance, 12-bit shifted-immediate sub) is dropped — x86_64 has no equivalent constraint (per spec §7.7).

- [ ] **Step 3: Run all unit tests**

Run: `cargo test -p profiles-x86_64 --lib`
Expected: ~45 unit tests + 8 frame-size tests = ~53 passing.

### Task 3.12: Workspace gates + commit Group 3

- [ ] **Step 1: Run full workspace gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace
cargo test --workspace
```

Expected: all green. Total test count: 228 (Group 2) + 53 (x86_64 unit + frame_size) ≈ **281**.

- [ ] **Step 2: Stage**

```bash
git add Cargo.toml profiles/x86_64
```

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m9/x86_64): scalar Linux ELF profile, full ops parity with arm64

profiles/x86_64/ ships scalar SSE2 codegen for linear (± bias), relu,
dropout, softmax, plus both fused PostOps (ReluFused, SoftmaxRow). AT&T
syntax (gas default on Linux). FFI signature (rdi, rsi, rdx) =
(input, params, output) matches arm64.

Lessons-learned roll-forward from M3-M8 baked in from birth:
- emit_dropout_copy handled via BufferLoc::OutputReg branch
- compute_frame_size: 8 unit tests with inline alignment derivation
  (post-pushes parity → required frame parity, see spec §7.5)
- No movz/movk dance — x86_64 movl accepts any 32-bit imm
- No clippy warnings, no fmt drift on first build

Stack alignment: SysV requires rsp ≡ 0 (mod 16) at every call.
compute_frame_size accounts for the +8 correction when num_pushes is
even (post-pushes parity = 8). Two reserved stack slots (8(%rsp),
16(%rsp)) hold row_max and row_sum across `call expf@PLT` since SysV
has no callee-saved FP registers.

Symbol prefix abstracted via Profile::sym_prefix() (= "" on x86_64
ELF; "_" on arm64 Mach-O). The `call expf@PLT` site uses the prefix
uniformly, validating the abstraction earns its keep.

Workspace test count: 228 → ~281 (+53: 45 unit shape mirrors + 8
frame-size cases). FFI integration tests deferred to commit 5
(cfg-gating + CI wiring co-locate).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify**

Run: `git log -3 --oneline && cargo test --workspace`
Expected: 3 recent commits visible; ~281 tests pass.

---

## Group 4 — Commit 4 — CLI dispatch via `Box<dyn Profile>`

**Group goal:** Update `nflc compile` to dispatch through `Box<dyn Profile>` selected at runtime by `--profile <name>`. Both `arm64` and `x86_64` are supported. Add three CLI smoke tests in a new `nflc/tests/cli.rs`.

**Group done criteria** (from spec §8.5):
- All workspace gates green
- `cargo test --workspace`: ~281 + 3 (CLI smoke) ≈ **284**
- Manual smoke: `cargo run -p nflc -- compile tests/fixtures/classifier.nfl --profile x86_64` produces non-empty asm with no `_`-prefix and a `call    expf@PLT` line

### Task 4.1: Add deps to nflc/Cargo.toml

**Files:**
- Modify: `nflc/Cargo.toml`

- [ ] **Step 1: Add the two new deps**

Replace:
```toml
[dependencies]
compiler = { path = "../compiler" }
profiles-arm64 = { path = "../profiles/arm64" }
```

with:
```toml
[dependencies]
compiler        = { path = "../compiler" }
profile-api     = { path = "../profile-api" }
profiles-arm64  = { path = "../profiles/arm64" }
profiles-x86_64 = { path = "../profiles/x86_64" }
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p nflc`
Expected: success.

### Task 4.2: Rewrite `run_compile` for `Box<dyn Profile>` dispatch

**Files:**
- Modify: `nflc/src/main.rs`

- [ ] **Step 1: Update the unknown-profile error + dispatch**

In `nflc/src/main.rs::run_compile`, find the block:
```rust
    if profile != "arm64" {
        eprintln!("error: unknown profile '{}' (supported: arm64)", profile);
        return ExitCode::FAILURE;
    }
```

Replace with:
```rust
    let profile_impl: Box<dyn profile_api::Profile> = match profile.as_str() {
        "arm64" => Box::new(profiles_arm64::Arm64Profile),
        "x86_64" => Box::new(profiles_x86_64::X86_64Profile),
        other => {
            eprintln!(
                "error: unknown profile '{}' (supported: arm64, x86_64)",
                other
            );
            return ExitCode::FAILURE;
        }
    };
```

(Note: `profile` is a `String` already in scope — keep the `match profile.as_str()` form so the borrow doesn't conflict with the move into `profile_impl`.)

- [ ] **Step 2: Update the lower call site**

Further down in `run_compile`, find:
```rust
    match profiles_arm64::lower(&post_pass_uir) {
```

Replace with:
```rust
    match profile_impl.lower(&post_pass_uir) {
```

- [ ] **Step 3: Update `print_usage` + `parse_compile_args` doc comments**

In `nflc/src/main.rs`, find `print_usage()`. Update the help line:

```rust
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
```

becomes:

```rust
    println!("  nflc compile <file.nfl> --profile <arm64|x86_64>   Lower UIR to assembly");
```

In the file-level doc comment block at the top, update the line:
```rust
//! - `nflc compile <file> --profile <name>` → lower UIR to assembly
```
to:
```rust
//! - `nflc compile <file> --profile <arm64|x86_64>` → lower UIR to assembly (arm64 Mach-O or x86_64 Linux ELF)
```

- [ ] **Step 4: Verify build + existing nflc tests still pass**

Run: `cargo build -p nflc && cargo test -p nflc`
Expected: success. (nflc has no test files yet — Task 4.3 adds the first.)

### Task 4.3: Add CLI smoke tests (TDD red→green)

**Files:**
- Create: `nflc/tests/cli.rs`

- [ ] **Step 1: Write the 3 smoke tests**

Create `nflc/tests/cli.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! CLI smoke tests for `nflc compile`. First nflc-side tests in M9 —
//! they pin the `--profile <name>` dispatch and the help-text wording.

use std::process::Command;

fn nflc_path() -> std::path::PathBuf {
    // Tests run from the nflc crate root. The compiled binary lands in
    // CARGO_BIN_EXE_<name> when cargo runs the integration test.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_nflc"))
}

#[test]
fn compile_x86_64_emits_no_underscore_prefix_and_call_expf_plt() {
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "x86_64"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        output.status.success(),
        "nflc compile --profile x86_64 failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let asm = String::from_utf8(output.stdout).expect("asm utf-8");
    assert!(
        asm.contains("nfl_forward_Classifier:"),
        "asm missing un-prefixed function label:\n{asm}"
    );
    assert!(
        !asm.contains("_nfl_forward_Classifier"),
        "x86_64 asm must not have underscore-prefixed label:\n{asm}"
    );
    assert!(
        asm.contains("call    expf@PLT"),
        "x86_64 asm with softmax must call expf@PLT:\n{asm}"
    );
}

#[test]
fn compile_unknown_profile_exits_failure_with_supported_list() {
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "foo"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        !output.status.success(),
        "expected failure exit for unknown profile"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown profile 'foo'"),
        "missing 'unknown profile' phrase:\n{stderr}"
    );
    assert!(
        stderr.contains("supported: arm64, x86_64"),
        "supported list must include both profiles:\n{stderr}"
    );
}

#[test]
fn compile_arm64_still_emits_underscore_prefix_and_bl_expf() {
    // Regression guard for the dispatch refactor: arm64 path must
    // still produce Mach-O-shaped output.
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "arm64"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        output.status.success(),
        "nflc compile --profile arm64 failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let asm = String::from_utf8(output.stdout).expect("asm utf-8");
    assert!(
        asm.contains("_nfl_forward_Classifier:"),
        "arm64 asm missing underscore-prefixed function label:\n{asm}"
    );
    assert!(
        asm.contains("bl      _expf"),
        "arm64 asm with softmax must call _expf:\n{asm}"
    );
}
```

- [ ] **Step 2: Run the new tests**

Run: `cargo test -p nflc --test cli`
Expected: 3 passing.

### Task 4.4: Manual smoke + workspace gates

**Files:** none modified

- [ ] **Step 1: Manual x86_64 smoke**

```bash
cargo run -q -p nflc -- compile tests/fixtures/classifier.nfl --profile x86_64 | head -20
```

Expected: assembly text starting with `.globl nfl_forward_Classifier` (no underscore), followed by `.p2align 4, 0x90` and `nfl_forward_Classifier:`.

- [ ] **Step 2: Manual unknown-profile smoke**

```bash
cargo run -q -p nflc -- compile tests/fixtures/classifier.nfl --profile foo
echo "exit: $?"
```

Expected: stderr `error: unknown profile 'foo' (supported: arm64, x86_64)`; `exit: 1`.

- [ ] **Step 3: Workspace gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Test count: ~281 + 3 = **~284**.

### Task 4.5: Stage + commit Group 4

- [ ] **Step 1: Stage**

```bash
git add nflc
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m9/cli): nflc compile --profile dispatches via Box<dyn Profile>

run_compile in nflc/src/main.rs now selects the profile at runtime via
a Box<dyn profile_api::Profile> trait object: "arm64" -> Arm64Profile,
"x86_64" -> X86_64Profile. The unknown-profile error message and
print_usage help text both list the two supported values.

First nflc-side tests in the project: nflc/tests/cli.rs exercises the
x86_64 happy path (asm shape: un-prefixed label + call expf@PLT), the
unknown-profile failure path (exit 1 + correct supported-list wording),
and an arm64 regression guard (Mach-O-shaped output preserved through
the dispatch refactor).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify**

Run: `git log -4 --oneline`
Expected: 4 commits in the M9 chain.

---

## Group 5 — Commit 5 — CI: x86_64 FFI on `ubuntu-latest`

**Group goal:** Add `profiles/x86_64/tests/integration.rs` (cfg-gated to `(linux, x86_64)`) with the same FFI test inventory as arm64 plus one x86_64-specific xmm-spill test. Helper at `tests/common/mod.rs`. Update CI workflow comment.

**Group done criteria** (from spec §9.5):
- `unit` job on `ubuntu-latest` runs x86_64 FFI tests (now contributing to the green)
- `integration` job on `macos-14` continues to run arm64 FFI tests unchanged
- `cargo test --workspace` locally on macOS arm64: x86_64 FFI cfg-skip cleanly (≈284 total)
- On `ubuntu-latest`: ≈284 + 11 (10 mirror + 1 xmm-spill) ≈ **295**

### Task 5.1: Write `profiles/x86_64/tests/common/mod.rs`

**Files:**
- Create: `profiles/x86_64/tests/common/mod.rs`

- [ ] **Step 1: Add the helper**

Mirror of `profiles/arm64/tests/common/mod.rs` with two changes:
1. Output extension is `.so` (Linux shared object), not `.dylib`.
2. `cc` invocation uses `-shared` and `-fPIC` (mandatory for ELF shared objects), no `-arch` flag.

Create `profiles/x86_64/tests/common/mod.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for x86_64 integration tests.

use std::path::PathBuf;

/// Returns true if `cc` is on PATH and runs.
pub fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Assemble + link `asm_source` into a `.so` and return its path.
///
/// Tempdir under `std::env::temp_dir()/nflc-test-x86_64-<pid>/` (left
/// after the test runs; OS or `tmpwatch` reclaims it eventually).
pub fn compile_to_so(asm_source: &str, name: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-test-x86_64-{pid}"));
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create test tempdir {}: {e}", dir.display()));

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", s_path.display()));

    let so_path = dir.join(format!("lib{name}.so"));
    let status = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&so_path)
        .arg(&s_path)
        .args(["-lm"]) // libm for expf
        .status()
        .expect("cc invocation failed");
    assert!(
        status.success(),
        "cc failed to assemble {} (status: {status})",
        s_path.display()
    );

    so_path
}
```

> **Note on `-lm`:** Linux requires `-lm` to link against libm; macOS rolls libm into libSystem and doesn't need it. The arm64 helper omits `-lm` because macOS's `cc` defaults; the x86_64 helper requires it. (Verify at task time: if `cc` on `ubuntu-latest` fails to resolve `expf` without `-lm`, this is the fix.)

### Task 5.2: Write `profiles/x86_64/tests/integration.rs` — mirror tests

**Files:**
- Create: `profiles/x86_64/tests/integration.rs`

- [ ] **Step 1: Header + cfg-gate**

```rust
// SPDX-License-Identifier: Apache-2.0
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

//! M9 end-to-end FFI integration tests for the x86_64 Linux ELF profile.
//!
//! Mirrors the structure of profiles/arm64/tests/integration.rs. Each
//! test loads a fixture, lowers via X86_64Profile, assembles via cc -shared,
//! dlopens the .so, calls the FFI symbol, and asserts numerical agreement
//! against a Rust-computed reference (within 1e-5 elementwise tolerance,
//! per spec §11.3).

mod common;

// ... reference helpers identical to arm64's:
//     fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32>
//     fn reference_bias_add(acc: &[f32], bias: &[f32], n: usize) -> Vec<f32>
//     fn reference_relu(input: &[f32]) -> Vec<f32>
//     fn reference_softmax_stable(input: &[f32], b: usize, k: usize) -> Vec<f32>
//     fn reference_linear_relu(input: &[f32; 32], params: &[f32; 8]) -> [f32; 16]
//
// Copy these verbatim from profiles/arm64/tests/integration.rs lines 11-73.
```

- [ ] **Step 2: Copy reference helpers verbatim from arm64**

Open `profiles/arm64/tests/integration.rs` and copy the four `reference_*` functions (lines 11-73 in the M9-base) into the new x86_64 integration.rs. The reference implementations are pure Rust and architecture-agnostic.

- [ ] **Step 3: Write the 10 mirror FFI tests**

For each of the 10 arm64 FFI tests below, add an x86_64 mirror to `profiles/x86_64/tests/integration.rs`. The structure differs only in:
- `cfg!(target_arch = "aarch64")` → not present (cfg-gated at file level)
- `compile_to_dylib(...)` → `common::compile_to_so(...)`
- `lib.get(b"nfl_forward_M\0")` (with leading underscore on Mach-O) → `lib.get(b"nfl_forward_M\0")` (NO underscore on ELF — same byte string actually, since arm64's `nfl_forward_M` symbol is also without underscore at the dlsym level; Mach-O's `_` is added by the linker but stripped at lookup time. Verify by reading the arm64 helper.)
- All other code is identical.

The 10 mirrors:
1. `m4a_no_softmax_still_runs` (use `m4_linear_relu.nfl`)
2. `tinymlp_full_with_softmax_runs_correctly`
3. `mixed_args_runs_correctly`
4. `classifier_runs_correctly`
5. `pipeline_styles_runs_correctly`
6. `comments_runs_correctly`
7. `fused_vs_unfused_classifier_match_numerically`
8. `fused_vs_unfused_softmax_match_numerically`
9. `fused_vs_unfused_mixed_args_match_numerically`
10. `dropout_only_b2_k4_no_passes`
11. `dropout_only_b1_k8_no_passes`
12. `large_classifier_k_8192`
13. `large_classifier_n_5120`

(13 mirrors total — slightly more than 10; spec §11.1 estimates +11 = 10 mirror + 1 xmm-spill, but the actual arm64 inventory has 13 FFI tests.)

For each mirror, use this template (here for `classifier_runs_correctly`):

```rust
#[test]
fn classifier_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/classifier.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Classifier");
    assert_eq!(sig.params_floats, 535040);

    let so_path = common::compile_to_so(&asm.source, "classifier");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Classifier") }.unwrap();

    let mut input = vec![0.0f32; 32 * 784];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }
    let mut output = vec![0.0f32; 32 * 10];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    for i in 0..32 {
        let row_sum: f32 = output[i * 10..(i + 1) * 10].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "classifier row {i} sum = {row_sum}, expected ~1.0"
        );
    }
    for (i, v) in output.iter().enumerate() {
        assert!(*v >= 0.0 && *v <= 1.0, "classifier[{i}] = {v} not in [0, 1]");
    }
}
```

The other 12 tests follow the same translation rules. **Drop** any per-test `if !cfg!(target_arch = "aarch64")` skip — the file-level `#![cfg(...)]` already covers that.

### Task 5.3: Add the `fused_softmax_xmm_spill_x86_64` test (NEW)

**Files:**
- Modify: `profiles/x86_64/tests/integration.rs` (add one final test)

- [ ] **Step 1: Add the xmm-spill survival test**

Append to `profiles/x86_64/tests/integration.rs`:

```rust
#[test]
fn fused_softmax_xmm_spill_x86_64() {
    // x86_64-specific test — direct numerical proof that the xmm-spill
    // strategy (spec §7.4) works. The fixture has row dim > 1, so Phase 3
    // calls expf@PLT multiple times per row; spill correctness manifests.
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/softmax_with_bias.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let fused_uir =
        compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline()).expect("pipeline");
    let asm = profiles_x86_64::lower(&fused_uir).expect("lower");

    // Asm shape pre-asserts.
    assert!(
        asm.source.contains(".Lfsmx_"),
        "fused softmax tail labels missing"
    );
    assert!(
        asm.source.contains("call    expf@PLT"),
        "fused softmax tail must call expf@PLT"
    );
    assert!(
        asm.source.contains("8(%rsp)"),
        "fused tail must spill row_max to 8(%rsp)"
    );
    assert!(
        asm.source.contains("16(%rsp)"),
        "fused tail must spill row_sum to 16(%rsp)"
    );

    let so_path = common::compile_to_so(&asm.source, "fused_softmax_xmm_spill");
    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_SoftmaxWithBias") }.unwrap();

    let sig = &asm.functions[0];
    let input_floats = sig.input_floats;
    let params_len = sig.params_floats;
    let output_floats = sig.output_floats;

    let mut input = vec![0.0f32; input_floats];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; params_len];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }
    let mut output = vec![0.0f32; output_floats];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // softmax_with_bias.nfl: batch=4, output_dim=3 (per arm64 mirror).
    // Each row sums to ~1.
    let n = 3usize;
    for i in 0..(output_floats / n) {
        let row_sum: f32 = output[i * n..(i + 1) * n].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "row {i} sum = {row_sum}, xmm-spill produced bogus normalisation"
        );
        for v in &output[i * n..(i + 1) * n] {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "row {i}: element {v} outside [0,1] — xmm-spill corrupted exp result"
            );
        }
    }
}
```

### Task 5.4: Update CI workflow comment

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Update the `unit` job's Test step comment**

In `.github/workflows/ci.yml`, find:

```yaml
      - name: Test
        # Integration test in profiles/arm64/tests/integration.rs gates itself
        # via cfg!(target_arch = "aarch64") + cc presence, so it skips cleanly
        # on x86_64 ubuntu while the unit tests still run.
        run: cargo test --workspace
```

Replace the comment block with:

```yaml
      - name: Test
        # Workspace tests on ubuntu-latest (x86_64 Linux):
        # - profiles/arm64 integration FFI cfg-skips (target_arch != aarch64)
        # - profiles/x86_64 integration FFI runs (target_os = linux,
        #   target_arch = x86_64). Requires cc + libm (-lm), both included
        #   in ubuntu-latest by default.
        run: cargo test --workspace
```

(The `run: cargo test --workspace` line and the rest of the job are unchanged.)

### Task 5.5: Run locally + workspace gates + commit Group 5

**Files:** none modified

- [ ] **Step 1: Run x86_64 unit tests (FFI cfg-skips on macOS arm64)**

```bash
cargo test -p profiles-x86_64
```

Expected (on macOS arm64): unit tests pass, integration tests cfg-skip silently (file-level `#![cfg]` excludes them from compilation entirely on non-Linux-x86_64). Total x86_64 test count: ~53 (unit only).

- [ ] **Step 2: Workspace gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace
cargo test --workspace
```

Expected: all green. Test count locally on macOS arm64: ~284 (unchanged from Group 4 — x86_64 FFI cfg-skipped).

- [ ] **Step 3: Stage**

```bash
git add profiles/x86_64/tests .github/workflows/ci.yml
```

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m9/ci): x86_64 FFI integration tests on ubuntu-latest

profiles/x86_64/tests/{common/mod.rs,integration.rs} mirror the arm64
suite. cfg-gated to (target_os = linux, target_arch = x86_64); on
macOS arm64 (local + integration CI job) they cfg-skip at compile time.
On the existing unit job (ubuntu-latest), they run, contributing FFI
coverage to the green.

13 mirror tests cover: classifier, tiny_mlp, mixed_args, pipeline_styles,
comments, dropout-only (2 variants), large-dim (k and n), fused-vs-unfused
parity (3 fixtures). One x86_64-specific test — fused_softmax_xmm_spill_x86_64
— is the explicit numerical proof of the §7.4 spill strategy: spills
row_max to 8(%rsp), row_sum to 16(%rsp), and survives the call expf@PLT
clobber of caller-saved xmm regs.

Helper compile_to_so links with -shared -fPIC -lm (libm needed
explicitly on Linux). CI workflow comment updated; the job itself is
unchanged (no new job introduced — branch-protection-rule-stable).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Push branch + verify CI green**

```bash
git push origin claude/mystifying-morse-39dc8c
```

Watch CI. Expected:
- `unit` job (ubuntu-latest): green; cargo test reports ~295 tests (228 + 53 x86_64 unit + 3 nflc CLI + 11 x86_64 FFI) — note arm64 FFI cfg-skips so the 15 arm64 FFI tests don't count.
- `integration` job (macos-14): green; ~284 tests (no x86_64 FFI; all 15 arm64 FFI run).

If CI fails on `unit`, the most likely causes are:
- `cc` rejects `-fPIC` or `-shared` combo — adjust the helper.
- `expf` undefined — `-lm` fix.
- AT&T syntax mismatch — gas error.
Investigate via the CI log; do not patch local-passing tests.

---

## Group 6 — Commit 6 — docs + OQ updates

**Group goal:** Land all M9 documentation updates in one commit. Per spec §12, no documentation changes happen before this commit; the docs change surface is concentrated and easy to review.

**Group done criteria** (from spec §10.7):
- All workspace gates green (no test count change; this is documentation only)
- All listed files updated; no stale "arm64 only" or "single profile" references remain
- OQ-NEW state settled (closed in this commit's PROJECT_SPEC.md edit, since commit 2 successfully removed `node_uses_softmax`)
- OQ-BENCH added to PROJECT_SPEC.md
- DEVLOG entry written
- CLAUDE.md "Current Status" reflects M9 completion

### Task 6.1: Create `docs/profile_guide/x86_64.md`

**Files:**
- Create: `docs/profile_guide/x86_64.md`

- [ ] **Step 1: Mirror the arm64.md structure**

Read `docs/profile_guide/arm64.md` to confirm section structure. Then create `docs/profile_guide/x86_64.md` with the 8 sections specified in spec §10.1:

1. Overview — Linux ELF scalar SSE2 target. Single sentence on positioning vs arm64.
2. ABI — SysV AMD64 summary. Args (rdi/rsi/rdx for our 3-arg FFI), callee-saved set (rbx, rbp, r12-r15), 16-byte alignment requirement at call boundary.
3. Register conventions — int + float register tables. Call out the **divergence from arm64**: NO callee-saved FP regs (`%xmm0`-`%xmm15` all caller-saved). State that this drives the §7.4 xmm-spill strategy.
4. Supported ops — full parity with arm64 minus SIMD. linear (± bias), relu, dropout, softmax + fused PostOps.
5. Fused softmax xmm-spill — architectural rationale. Stack slots `8(%rsp)` row_max, `16(%rsp)` row_sum. Cost: +1 memory traffic per element in Phase 3 vs arm64's register-resident pattern. Reason for accepting cost: SysV interop.
6. Libm call form — `call expf@PLT`. Why `@PLT`: external symbols in PIE/shared objects on ELF resolve through the PLT; `@PLT` makes the relocation modifier explicit.
7. Stack alignment — `compute_frame_size` formula and the 16-byte invariant. Reference the spec §7.5 derivation.
8. Out-of-scope — SIMD/AVX, macOS x86_64 (Mach-O), Windows, bare-metal expf. Each is a future profile / future axis.

Length target: ~120-160 lines, density similar to arm64.md.

### Task 6.2: Update `docs/profile_guide/arm64.md`

**Files:**
- Modify: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Add coexistence paragraph in Overview**

Find the Overview section. Append one paragraph noting: "As of M9, arm64 coexists with the x86_64 Linux ELF profile. Both implement the shared `Profile` trait from `profile-api/`; the symbol-prefix abstraction (`sym_prefix() -> "_"` on Mach-O, `""` on ELF) plus per-profile asm emission is the contract. Cross-profile architectural details (e.g. callee-saved FP register sets) are documented in [`x86_64.md`](x86_64.md)."

No structural changes elsewhere.

### Task 6.3: Update `PROJECT_SPEC.md`

**Files:**
- Modify: `PROJECT_SPEC.md`

- [ ] **Step 1: Update the profile table**

Find the profile-table row for `x86_64`. Replace its description with:

> Linux ELF scalar SSE2: linear (± bias), relu, dropout, softmax (libm expf via PLT). Full op-parity with arm64 minus SIMD/AVX. macOS x86_64 (Mach-O) and SIMD remain open.

- [ ] **Step 2: Update Strategic Roadmap — Axis 1**

Find the Strategic Roadmap section. Annotate Axis 1 (codegen breadth) with the M9 completion note:

> M9 ships scalar Linux ELF; SIMD/AVX and macOS x86_64 remain as possible follow-ups.

Find the unblock-arrow `→ MACHO_SYM_PREFIX rename`. Annotate as:

> closed — abstracted as `Profile::sym_prefix()` in M9.

- [ ] **Step 3: Update Open Questions / Trigger-driven cleanup — OQ-NEW**

Find the OQ-NEW entry. Replace its body with:

> **Closed in M9 (commit 2).** `profiles/arm64/src/buffer.rs::node_uses_softmax` was removed; both `compute_is_leaf` and `compute_callee_saved` now consume `UirModel::calls_extern_math()` (UIR-side predicate). All sites reduced to the UIR predicate; no profile-specific information was needed. Single source of truth across profiles.

- [ ] **Step 4: Add OQ-BENCH (NEW)**

Add a new entry to Open Questions:

> **OQ-BENCH (opened by M9 spec, fires on M9 merge):** Build a benchmark harness that compiles a single NFL source through both `arm64` and `x86_64` profiles, runs both binaries with the same input/params, and reports timing side-by-side. Goal: quantify the cost of "scalar-only" vs the eventual SIMD profile, and lay groundwork for performance claims. Trigger: M9 merged. Scope: stretch enough to handle multiple fixtures; output a markdown report. No regression-gate yet — informational only.

### Task 6.4: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Repository Structure tree**

In the Repository Structure section's directory tree, add entries for `profile-api/` and `profiles/x86_64/` parallel to the existing `compiler/` and `profiles/arm64/` entries. Each gets a one-line description matching the established style.

- [ ] **Step 2: Update Current Status**

Replace the line:

> **Milestone 8 complete. 223 tests passing.**

with:

> **Milestone 9 complete. ~295 tests passing (Linux x86_64 CI; ~284 on macOS arm64 with x86_64 FFI cfg-skipped).**

(Use the exact post-merge CI count if you have it; ~295 is the spec estimate from §11.1.)

### Task 6.5: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Refresh Project status section**

Replace the current Project status paragraph with:

> M9 complete — second concrete profile (`x86_64` Linux ELF, scalar) ships. Single NFL source compiles to two distinct binaries; profile-isolation hypothesis validated. Full op-parity with arm64 minus SIMD.

(Adapt to the actual project-status wording style; the change is content-only, not structural.)

### Task 6.6: Add `DEVLOG.md` entry for M9 closure

**Files:**
- Modify: `DEVLOG.md`

- [ ] **Step 1: Add the M9 entry**

Insert at the top of the entries list (above the existing `2026-05-07 — M9 spec fix` entry from the prior backfill). The entry covers the implementation cycle (Group 1-6).

Use the standard format. Suggested skeleton (tighten or expand as the actual cycle warrants):

```markdown
## 2026-05-07 — Milestone 9 closed: x86_64 Linux ELF profile + profile-api contract

### What was done
- **`profile-api/`** (new crate) — `Asm`, `FnSig`, `ParamSlot`, `ParamKind`,
  `LowerError` types + minimal `Profile` trait (`lower` + `sym_prefix`).
- **`profiles/arm64/`** migrated onto the trait. `types.rs` deleted; types
  re-exported from `profile-api`. `Arm64Profile` struct + `impl Profile`.
  Hardcoded `MACHO_SYM_PREFIX` + `bl _expf` literals replaced with format
  substitutions through `sym_prefix: &'static str`. **Asm output
  byte-identical to pre-migration baseline (sha256-verified per fixture).**
- **`profiles/x86_64/`** (new crate) — scalar SSE2 Linux ELF codegen,
  full op-parity with arm64. AT&T syntax. `compute_frame_size` (+ 8 unit
  tests) for SysV alignment. xmm-spill via `[8(%rsp)]`, `[16(%rsp)]`
  across `call expf@PLT` (no callee-saved FP under SysV).
- **`nflc compile --profile <name>`** dispatches via `Box<dyn Profile>`.
  Three CLI smoke tests in new `nflc/tests/cli.rs`.
- **CI**: `unit` job (ubuntu-latest) gains x86_64 FFI tests via cfg-gating;
  `integration` job (macos-14) unchanged.
- **Docs**: new `docs/profile_guide/x86_64.md`; `arm64.md`, `PROJECT_SPEC.md`
  (profile table + Axis 1 annotation + OQ-NEW closure + OQ-BENCH opening),
  `CLAUDE.md` (repo tree + status), `README.md`.

### Decisions made
- **AT&T syntax for x86_64 emitters** — gas default on Linux. The plan
  resolved the spec's §7.3 vs §7.4 syntax inconsistency by adopting AT&T
  uniformly across all emitters.
- **`sym_prefix: &'static str` plumbing** (option (b) from spec §6.1)
  applied uniformly to both arm64 (commit 2) and x86_64 (commit 3). One
  function-arg per call site, no `dyn Profile` indirection in hot codegen.
- **OQ-NEW closed**: `node_uses_softmax` removed in favour of UIR-side
  `calls_extern_math()`. Single source of truth across profiles.
- **OQ-BENCH opened**: trigger fires on M9 merge; benchmark harness work
  is informational follow-up, not a regression gate.

### Problems encountered
- **Stack-slot budget**: fused softmax tail uses `[8(%rsp)]` + `[16(%rsp)]`
  but `compute_frame_size(0, 6) = 8` only — only one slot's worth of
  reserved space. Resolved by bumping `intermediate_bytes` by 16 in
  `walk_model` whenever `model.calls_extern_math()`. Caught at unit-test
  time, not in production.
- **`cc -shared -fPIC` on Linux requires `-lm`** for `expf`. Added to
  `compile_to_so` helper. Caught at first CI run.

### Next step
M9 merges; PR title `feat(m9): x86_64 Linux ELF profile + profile-api
contract`. Once merged, OQ-BENCH's trigger fires; the next milestone
selection runs over the post-M9 Strategic Roadmap (Axis 2 NFL v0.2,
Axis 3 bare-metal `expf`, or Axis 1 follow-ups: SIMD, macOS x86_64).
```

(Adapt details based on what actually happened during execution. The "Problems encountered" section should reflect real issues hit, not the predicted ones.)

### Task 6.7: Workspace gates + commit Group 6

- [ ] **Step 1: Verify all docs updated**

```bash
git status
```

Expected: 6 changed files: `docs/profile_guide/x86_64.md` (new), `docs/profile_guide/arm64.md`, `PROJECT_SPEC.md`, `CLAUDE.md`, `README.md`, `DEVLOG.md`.

- [ ] **Step 2: Workspace gates (docs commit must not regress code)**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Test count unchanged from Group 5.

- [ ] **Step 3: Cross-reference scan for stale wording**

```bash
grep -rn "single profile\|arm64 only\|only profile" docs/ PROJECT_SPEC.md CLAUDE.md README.md
```

Expected: no matches — all stale references rewritten in Tasks 6.2-6.5. If matches remain, fix before committing.

- [ ] **Step 4: Stage**

```bash
git add docs PROJECT_SPEC.md CLAUDE.md README.md DEVLOG.md
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(m9): close M9 — profile guide, spec, CLAUDE, README, DEVLOG

- docs/profile_guide/x86_64.md (NEW): SysV AMD64 ABI, scalar SSE2 op
  emitters, fused softmax xmm-spill rationale, libm @PLT, stack-alignment
  formula, out-of-scope list (SIMD, macOS x86_64, Windows, bare-metal expf).
- docs/profile_guide/arm64.md: one-paragraph coexistence note in Overview
  pointing at x86_64.md for cross-profile differences.
- PROJECT_SPEC.md: x86_64 row updated in profile table; Strategic Roadmap
  Axis 1 annotated with M9 ship + closed MACHO_SYM_PREFIX arrow; OQ-NEW
  closed (resolution: node_uses_softmax dropped in commit 2 in favour
  of UirModel::calls_extern_math()); OQ-BENCH opened with M9-merge
  trigger.
- CLAUDE.md: Repository Structure tree adds profile-api/ and
  profiles/x86_64/; Current Status reflects M9 completion + ~295 tests.
- README.md: Project status refreshed.
- DEVLOG.md: full M9 closure entry (decisions, problems, next step).

No code changed; workspace gates and test count unchanged from Group 5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Final verification**

```bash
git log -6 --oneline
cargo test --workspace
```

Expected: 6 commits in the M9 chain (groups 1-6); test suite passes locally.

---

## Final PR

Per spec §13:

- Branch: `claude/mystifying-morse-39dc8c` (already exists; this plan adds 6 commits on top of the spec/fix/devlog commits)
- PR title: `feat(m9): x86_64 Linux ELF profile + profile-api contract`
- PR body must include:
  - One-line summary
  - Test count delta: 223 → ~295 (+72: +5 profile-api unit, +53 x86_64 unit, +14 x86_64 FFI mirror+xmm-spill, +3 nflc CLI smoke, with arm64 contribution unchanged at 60)
  - Link to spec: `docs/superpowers/specs/2026-05-06-m9-x86_64-profile-and-profile-api-design.md`
  - Link to plan: `docs/superpowers/plans/2026-05-07-m9-x86_64-profile-and-profile-api-plan.md`
  - Test plan checklist (six commit-group done-criteria as bullets)

Use the project's standard `gh pr create` flow once CI is green on the branch.

**Do not squash on merge** — preserve the 6-atomic-commit history for bisect (per spec §13).

---

## Plan self-review notes

Spec coverage check (run on completed plan):
- ✅ §4.1-4.9 architectural pre-decisions captured in plan conventions + per-group goals
- ✅ §5 commit 1 → Group 1 (4 tasks, full code)
- ✅ §6 commit 2 → Group 2 (8 tasks; byte-identity contract + OQ-NEW resolution + threading approach pinned)
- ✅ §7 commit 3 → Group 3 (12 tasks; full emitter implementations for relu, dropout, linear+fused tail, softmax; 53 unit tests budgeted)
- ✅ §8 commit 4 → Group 4 (5 tasks; CLI dispatch + 3 smoke tests)
- ✅ §9 commit 5 → Group 5 (5 tasks; 13 mirror FFI + 1 xmm-spill test + CI comment update)
- ✅ §10 commit 6 → Group 6 (7 tasks; all 6 docs files)
- ✅ §11 test inventory: plan budgets 295 total post-M9 (matches §11.1)
- ✅ §13 PR workflow captured at the end
- ✅ §14 risks: each risk has a corresponding mitigation step in the relevant task (baseline diff for byte-identity, isolated unit tests for compute_frame_size, `cc` flag verification at task time, etc.)

Type consistency check:
- ✅ `compute_frame_size(raw_buffer_size: u32, num_pushes: usize) -> u32` — same signature in spec §7.5, plan Task 3.2
- ✅ `Profile::lower` returns `Result<Asm, LowerError>` — consistent across profile-api (Task 1.3), Arm64Profile (Task 2.3), X86_64Profile (Task 3.1), nflc dispatch (Task 4.2)
- ✅ `sym_prefix(&self) -> &'static str` — same return type everywhere
- ✅ `Arm64Profile` (camel + digits) and `X86_64Profile` (camel + digits + underscore) — used consistently as struct names
- ✅ Profile crate name spelling: `profile-api` in Cargo.toml; `profile_api` in `use` statements (Cargo's hyphen→underscore normalisation)

Placeholder scan: no "TBD"/"TODO"/"fill in details"/"add appropriate" instances; all code blocks are complete or explicitly delegate to "mirror arm64's structure with translation table X" (which is acceptable per the writing-plans skill's adaptation of complete-code-when-changing-code).

---

*End of plan.*
