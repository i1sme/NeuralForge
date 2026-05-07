# Milestone 9 — x86_64 Linux ELF Profile + `profile-api` Contract — Design

> Brainstormed: 2026-05-06
> Strategic axis: **Axis 1 — codegen breadth** (Strategic Roadmap, `PROJECT_SPEC.md`)
> Predecessor: M8 (arm64 codegen hardening + viewer v0.1)
> Status: spec draft for plan synthesis

---

## 1. Overview

M9 delivers the first non-Mach-O concrete profile (`profiles/x86_64/`,
Linux ELF, scalar-only) with **full operations parity to arm64**, and lifts
the per-profile public surface (`Asm`, `FnSig`, `ParamSlot`, `ParamKind`,
`LowerError`, plus a new `Profile` trait) into a shared `profile-api/`
crate. The milestone validates the project's central architectural claim
— *profile isolation holds across OS boundaries* — by exercising it on a
target where Mach-O and ELF actually diverge: symbol prefix (`_` vs
empty), file format, libm call form (`bl _expf` vs `call expf@PLT`), and
position-independent code conventions.

The companion artefact is the `profile-api` crate. Without it, "isolation"
remains an informal claim about two independent crates with similar APIs;
the trait makes the contract type-level and compiler-checked. Per the
hard scoping constraint adopted during brainstorming, the trait is
minimal: `lower(&self, &Uir) -> Result<Asm, LowerError>` and
`sym_prefix(&self) -> &'static str`. Nothing more is added speculatively.

x86_64 ships scalar-only — `mulss`/`addss`/`maxss` and friends, no AVX,
no SIMD, no FMA. The benchmark question ("is the compiled output fast?")
is deferred to a separate trigger-driven follow-up (`OQ-BENCH`, opened
by this spec); M9 answers the prior question, *is the compiled output
correct on a second OS through a shared contract?*

---

## 2. Goal

Ship a single PR with six atomic commits that together:

1. Introduce `profile-api/` workspace crate exporting the public profile
   surface and a minimal `trait Profile`.
2. Migrate `profiles/arm64/` onto the trait without behavioural change
   (asm output byte-identical; all 223 existing tests green without
   modification).
3. Introduce `profiles/x86_64/` workspace crate, scalar-only Linux ELF
   target, full op-emitter parity (linear ± bias, relu, dropout,
   softmax) plus full fused PostOp parity (`ReluFused`, `SoftmaxRow`).
4. Update `nflc compile` to dispatch via `Box<dyn Profile>`, accepting
   `--profile arm64` or `--profile x86_64`.
5. Wire x86_64 FFI integration tests into the existing `unit` CI job on
   `ubuntu-latest`; arm64 FFI tests continue on the existing
   `integration` job on `macos-14`. No new CI job is introduced.
6. Land all documentation updates: new `docs/profile_guide/x86_64.md`,
   `PROJECT_SPEC.md` profile-table and Strategic Roadmap revisions,
   `OQ-NEW` closure (or principled re-open), new `OQ-BENCH` opening,
   `README.md` status refresh, `DEVLOG.md` entry, `CLAUDE.md`
   Repository Structure tree update.

The PR closes M9; no follow-up PR is needed for any in-scope concern.

---

## 3. Non-goals

The following are explicitly out of scope for M9. They are listed not as
"future work" but as decisions: each was raised during brainstorming and
deliberately rejected.

- **No SIMD / AVX / SSE-vectorisation.** x86_64 uses scalar SSE2 only
  (`mulss`, `addss`, `maxss`, etc.). FMA via `vfmadd231ss` is not used
  even where AVX is available. Vectorisation is a separate optimisation
  axis with its own benchmark surface; bundling it with isolation
  validation would conflate two independent claims.
- **No macOS x86_64 (Mach-O) support.** macOS x86_64 shares Mach-O
  conventions (`_` prefix, no PLT) with arm64 macOS, so it does not
  validate platform-isolation, only ISA-isolation. The whole point of
  Axis 1 is the platform-isolation claim. macOS x86_64 is rejected for
  M9 on this basis. Future profiles can add it cheaply once the trait
  is in place.
- **No Windows / no non-SysV ABI.** SysV AMD64 only.
- **No bare-metal `expf`.** Softmax continues to call libm
  (`bl _expf` on arm64-macOS, `call expf@PLT` on x86_64-linux). Removing
  the libm dependency is the explicit subject of Axis 3.
- **No NFL v0.2 grammar / no attention ops.** Axis 2.
- **No QEMU usermode for local testing.** CI-only execution
  (GitHub Actions `ubuntu-latest`). QEMU was rejected during brainstorming
  as a fragile dev-loop hard to document and harder to maintain.
- **No profile-level viewer annotations** (per-node footprint, stack
  frame, callee-saved register set). This was raised in M8 closeout as
  a sequel to viewer v0.1; it is its own milestone with its own scope.
- **No `Profile` trait extensions beyond `lower` + `sym_prefix`.** No
  `relocation_hints()`, no `library_names()`, no `target_triple()`, no
  associated types, no default implementations, no sealed traits. Trait
  grows by trigger (a real consumer needs it), not by anticipation.
- **No benchmark harness.** Single NFL into arm64 + x86_64 numerical
  parity is verified via FFI tests; timing comparison is the subject of
  the new `OQ-BENCH`, whose trigger fires when this PR merges.
- **No CI job rename / no new CI job.** Existing two-job matrix
  (`unit` on ubuntu-latest + `integration` on macos-14) covers full
  platform matrix via cfg-gating; renaming risks branch-protection
  rule drift.

---

## 4. Pre-decided architectural calls

These were settled during brainstorming and are recorded here so the
plan does not re-litigate them.

### 4.1 Strategic axis: **Axis 1 — codegen breadth**

Selected over Axis 2 (modelling depth — NFL v0.2 grammar) and Axis 3
(deployment reach — bare-metal `expf`). Rationale: profile isolation is
the only nontrivial architectural claim of the project; everything else
(correctness, fusion, UIR) is checkable inside one backend. Isolation
is not. Validating it earlier is cheaper than later — building Axis 2's
attention stack on top of an unvalidated isolation hypothesis would
force a more expensive retrofit if isolation leaks.

### 4.2 Target OS: **Linux ELF** (not macOS x86_64 Mach-O)

The platform-isolation claim only manifests on a non-Mach-O target.
Mach-O x86_64 would validate ISA-isolation but leave OS-isolation
untested; the `MACHO_SYM_PREFIX` rename would remain cosmetic. Linux
ELF forces the abstraction to be real (prefix divergence, PLT, ELF
relocations, different libm symbol form).

### 4.3 Operations surface: **full parity with arm64**

All four op emitters (linear ± bias, relu, dropout, softmax) plus both
fused PostOp branches (`ReluFused`, `SoftmaxRow`). Subset alternatives
were rejected because:

- Without softmax, x86_64 never exercises `call expf@PLT`, which is the
  exact site where the symbol-prefix abstraction earns its keep — a
  partial port would not validate the abstraction.
- Without the fused PostOp branches, the architecturally-most-interesting
  code path (in-place buffer reuse, callee-saved-FP-survival across
  external calls, RowWise softmax tail) lives in only one profile and
  the isolation claim covers only the easy half.
- The asymmetry "arm64 fuses, x86_64 doesn't" creates a deferred
  obligation that is exactly the kind of trigger-driven debt this
  project tries to avoid.

### 4.4 Contract shape: **Path B — shared `profile-api` crate with minimal trait**

A new crate at `profile-api/` exports the public surface
(`Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError`) and a trait:

```rust
pub trait Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;
    fn sym_prefix(&self) -> &'static str;
}
```

Path A (full duplication) was rejected because it leaves isolation as
informal "two crates with similar APIs", not a type-level contract.
Path C (shared types, free `lower` functions) was rejected because the
contract remains compiler-unchecked and the symbol-prefix abstraction
ends up as a duplicated `pub const` instead of a trait method.

The trait is minimal by hard constraint:

> Trait grows by request, not by anticipation.

`relocation_hints`, `library_names`, `target_triple`, options to
`lower`, default methods — all explicitly excluded from M9 even where
they "obviously" might be useful in M10. Each is added by its own
trigger when a real consumer materialises.

### 4.5 Sequencing: **API-first (Approach 1)**

Six atomic commits in the order: (1) `profile-api` extract, (2) arm64
migration, (3) x86_64 build, (4) CLI dispatch, (5) CI matrix, (6)
docs + OQ updates. Each commit leaves the workspace clean: `cargo fmt
--all -- --check`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo build --workspace`, `cargo test --workspace` all
green at every checkpoint.

Approach 2 (x86_64 standalone first, then extract) was rejected because
it creates type duplication exactly where Path B is meant to prevent it,
and risks x86_64 starting on a slightly-different shape that complicates
the eventual extract.

### 4.6 Dispatch: **`Box<dyn Profile>`**

`nflc` selects the profile at runtime from `--profile <name>`. Since
the type is unknown at compile time, a trait object is the semantically
correct shape; generics would require knowing `P: Profile` at the call
site, which is not the case here. Both profile crates are statically
linked into the `nflc` binary; the trait object dispatches the runtime
choice.

### 4.7 Hard invariant of commit 2 (arm64 migration)

All 223 existing tests pass without modification after commit 2. arm64
unit tests use `String::contains` substring asserts (122 of them in
`profiles/arm64/src/tests.rs`); the two hardcoded `_expf` callsites
([linear.rs:200](profiles/arm64/src/ops/linear.rs:200) and
[softmax.rs:85](profiles/arm64/src/ops/softmax.rs:85)) are rewritten as
`format!("\tbl {}expf\n", self.sym_prefix())`. For arm64 where
`sym_prefix() -> "_"`, the format expansion yields `"\tbl _expf\n"` —
byte-identical. Substring asserts continue to match.

If a test fails after commit 2:
- (a) substring assertion unexpectedly mis-matches → migration bug;
  investigate code, do not patch test.
- (b) register name / instruction / call target changed → migration
  bug; investigate code, do not patch test.

The commit 2 contract is *byte-identical asm output for arm64*. Tests
are not adjusted.

### 4.8 `call expf@PLT` (not bare `call expf`) on x86_64

x86_64's libm call uses `call expf@PLT`, not `call expf`. The M9
artefact is a `.so` shared library (for FFI tests via `cc -shared`);
external symbols in PIE / shared objects on Linux ELF resolve through
the procedure-linkage table, and an explicit `@PLT` relocation
modifier guarantees correct linker behaviour. `call expf` without
`@PLT` may work with specific linker configurations but is not
guaranteed — this is correctness, not tuning.

### 4.9 Stack alignment invariant in x86_64 prologue

`rsp` must be 16-byte aligned immediately before each `call`
instruction (SysV AMD64 §3.2.2). On function entry, the caller's
`call`-instruction has just pushed the 8-byte return address, so
`rsp ≡ 8 (mod 16)`. Each prologue `push reg` (8 bytes) flips parity:
after N pushes, `rsp ≡ 8 - 8*N (mod 16) ≡ 8*(1 - N) (mod 16)`. To land
on `rsp ≡ 0 (mod 16)` after `sub rsp, frame_size`, the helper must
add an 8-byte correction when N is **even** (post-pushes parity is 8),
and zero correction when N is **odd** (post-pushes parity is 0):

```
final_frame_size = round_up(raw_buffer_size, 16)
                 + (if num_pushes is even then 8 else 0)
```

Full derivation, formula, and the unit-test inventory live in §7.5.

Without this correction the first `call expf@PLT` SIGSEGVs because
libm's `expf` uses SSE instructions that assume aligned-load
preconditions. A helper `compute_frame_size(raw_buffer_size: u32,
num_pushes: usize) -> u32` encapsulates the logic and is unit-tested
in isolation.

---

## 5. Commit 1 — `profile-api` crate

### 5.1 Cargo.toml workspace update

`Cargo.toml` (workspace root):

```toml
[workspace]
members = [
    "compiler",
    "nflc",
    "profile-api",          # NEW
    "profiles/arm64",
    "profiles/x86_64",      # introduced in commit 3 but listed up-front
]
```

The `profiles/x86_64` member is added in this commit even though the
crate directory does not exist yet. Cargo errors on a missing member,
so we add `profiles/x86_64/Cargo.toml` + empty `src/lib.rs` in this
commit as a stub (one-line `pub fn placeholder() {}` to satisfy the
build); the real implementation lands in commit 3. This keeps the
workspace manifest stable from commit 1 onwards rather than churning
it across commits.

### 5.2 `profile-api/` crate contents

`profile-api/Cargo.toml`:
```toml
[package]
name = "profile-api"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
compiler = { path = "../compiler" }
```

`profile-api/src/lib.rs` (single file; no submodules in M9):

```rust
// SPDX-License-Identifier: Apache-2.0

//! Public profile contract.
//!
//! Architecture profiles (`profiles/arm64`, `profiles/x86_64`)
//! implement the `Profile` trait. The compiler core (`compiler/`)
//! does not depend on any specific profile — UIR is profile-agnostic.

use compiler::ir::types::Uir;

pub struct Asm {
    pub source: String,
    pub functions: Vec<FnSig>,
}

pub struct FnSig {
    pub name: String,
    pub model: String,
    pub input_floats: u32,
    pub params_floats: u32,
    pub output_floats: u32,
    pub params_layout: Vec<ParamSlot>,
}

pub struct ParamSlot {
    pub kind: ParamKind,
    pub offset: u32,
    pub size: u32,
    pub origin_node: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    LinearWeight,
    LinearBias,
}

#[derive(Debug)]
pub enum LowerError {
    // All variants migrated verbatim from
    // profiles/arm64/src/types.rs::LowerError. No new variants in M9;
    // no removed variants; no renames. The plan enumerates the exact
    // current set during commit 1.
}

impl LowerError {
    // span() implementation migrated verbatim from arm64.
    pub fn span(&self) -> compiler::Span { unimplemented!("see commit 1 plan") }
}

// Display + Error impls migrated verbatim from arm64.
impl std::fmt::Display for LowerError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!("see commit 1 plan")
    }
}
impl std::error::Error for LowerError {}

pub trait Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;
    fn sym_prefix(&self) -> &'static str;
}
```

The exact field set on `Asm`/`FnSig`/`ParamSlot` mirrors what
`profiles/arm64/src/types.rs` currently exports, verbatim. No fields
are added, removed, or renamed — this is a pure relocation.

### 5.3 Tests for commit 1

A small smoke test in `profile-api/src/lib.rs` confirming that the
crate compiles in isolation and that `Asm` / `FnSig` / `ParamSlot` /
`ParamKind` / `LowerError` round-trip through Debug. No trait impls
yet (those land in commits 2 and 3); no Profile-method tests.

### 5.4 Commit 1 done criteria

- `cargo build --workspace` green
- `cargo clippy --workspace --all-targets -- -D warnings` green
- `cargo fmt --all -- --check` green
- `cargo test --workspace`: 223 (arm64 unchanged) + ~5 (profile-api smoke) ≈ 228
- `profiles/x86_64/` contains stub `Cargo.toml` + 1-line `lib.rs`

---

## 6. Commit 2 — arm64 migration onto `Profile` trait

### 6.1 Files modified

- `profiles/arm64/Cargo.toml`: dependencies +=
  `profile-api = { path = "../../profile-api" }`
- `profiles/arm64/src/types.rs`: emptied (or deleted entirely; the
  types live in `profile-api` now). All `pub use` re-exports are
  removed; downstream consumers (`profiles/arm64/src/{asm,buffer,
  codegen,ops/*,tests}.rs`) update their imports to
  `use profile_api::{Asm, FnSig, ParamSlot, ParamKind, LowerError};`
- `profiles/arm64/src/lib.rs`: introduces `pub struct Arm64Profile;`
  and `impl Profile for Arm64Profile`. The free function
  `pub fn lower(&Uir) -> Result<Asm, LowerError>` is preserved as a
  thin wrapper that calls `Arm64Profile.lower(uir)` — this avoids
  breaking direct callers (the arm64 integration tests) inside the
  same commit. (The wrapper can be removed in a later cleanup, or
  retained indefinitely; spec leaves this open.)
- `profiles/arm64/src/ops/linear.rs:200`: `"\tbl _expf\n"` rewritten as
  `&format!("\tbl {}expf\n", profile.sym_prefix())`. Requires plumbing
  a `&Arm64Profile` (or its prefix string) through the call chain to
  the emit site. Mechanism options surfaced for the plan:
  (a) thread `&dyn Profile` through `walk_uir`/`walk_model`/`emit_*`;
  (b) thread a `sym_prefix: &'static str` (loosely-typed but ergonomic);
  (c) make `sym_prefix` an associated const on a per-profile struct
      and use it directly in a profile-specific code path.
  Approach (a) is the most uniform; (b) is the lightest. Plan picks one.
  **Whichever approach is chosen for arm64 in commit 2 is applied
  uniformly to x86_64 in commit 3** — the two profiles share the same
  internal plumbing pattern, not different ones.
- `profiles/arm64/src/ops/softmax.rs:85`: same rewrite as above.
- arm64 function label / `.globl` directive sites: same rewrite. The
  exact set of sites is enumerated in the plan via grep for `"_nfl_"`
  and `".globl _"` in `profiles/arm64/src/`.

### 6.2 OQ-NEW resolution sanity check

Between commit 2 and commit 3, the plan inserts a verification step:

1. Grep `profiles/arm64/src/` for `node_uses_softmax`. Each call site
   is examined: does it semantically reduce to `uir.calls_extern_math()`
   (the UIR-side predicate already on `Uir` and `UirModel`)?
2. If yes for all sites: `node_uses_softmax` is removed from
   `profiles/arm64/src/buffer.rs`. **OQ-NEW closes** as a side effect
   of commit 2 and is recorded as such in commit 6.
3. If any site needs profile-specific information not derivable from
   the UIR predicate: `node_uses_softmax` stays where it is; **OQ-NEW
   re-opens** with an updated trigger condition (e.g. "next addition
   to the profile-side predicate vocabulary"), recorded in commit 6.

The decision is made during commit 2 work, not pre-committed in this
spec.

### 6.3 Hard invariant of commit 2

All 223 pre-existing tests pass without modification (see §4.7). If a
test fails, the migration is buggy — investigate the code, do not
patch the test.

### 6.4 Commit 2 done criteria

- `cargo test --workspace`: 223 + ~5 (profile-api) ≈ 228, **count
  unchanged from commit 1; arm64 contribution unchanged at 60 tests
  (45 unit + 15 integration)**
- Asm output of every fixture is byte-identical to pre-migration
  (verified by running `nflc compile <fixture> --profile arm64` and
  diffing against a captured pre-migration baseline; baseline capture
  is a one-time step at the start of commit 2 work)
- Workspace gates green

---

## 7. Commit 3 — `profiles/x86_64/` crate

### 7.1 Crate skeleton

`profiles/x86_64/Cargo.toml`:
```toml
[package]
name = "profiles-x86_64"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
compiler    = { path = "../../compiler" }
profile-api = { path = "../../profile-api" }
```

Layout mirrors `profiles/arm64/`:

```
profiles/x86_64/
├── Cargo.toml
├── src/
│   ├── lib.rs        ← pub struct X86_64Profile; impl Profile { ... }
│   ├── asm.rs        ← prologue/epilogue helpers, compute_frame_size
│   ├── buffer.rs     ← BufferLoc, assign_buffers, compute_callee_saved (SysV-specific)
│   ├── codegen.rs    ← walk_uir/walk_model dispatcher + classify_op
│   ├── ops/
│   │   ├── mod.rs
│   │   ├── linear.rs
│   │   ├── relu.rs
│   │   ├── softmax.rs
│   │   └── dropout.rs
│   └── tests.rs      ← unit shape-asserts (mirror profiles/arm64/src/tests.rs)
└── tests/
    ├── common/mod.rs    ← compile_to_dylib helper (own copy; cc crate platform-aware)
    └── integration.rs   ← FFI tests, cfg-gated to (target_os=linux, target_arch=x86_64)
```

### 7.2 SysV AMD64 ABI implementation

Calling convention:
- Int args: `rdi, rsi, rdx, rcx, r8, r9` (vs arm64 `x0-x7`)
- Float args: `xmm0-xmm7` (vs arm64 `s0-s7`)
- Return: int → `rax`, float → `xmm0`
- Callee-saved int: `rbx, rbp, r12-r15`
- **Callee-saved FP: NONE** — all `xmm0-xmm15` are caller-saved
- Stack: 16-byte aligned at `call` boundary (see §4.9)

FFI signature contract (matches arm64):
- arg 0 (`rdi`): `*const f32` input buffer
- arg 1 (`rsi`): `*const f32` params buffer
- arg 2 (`rdx`): `*mut f32` output buffer

### 7.3 Per-op emit (instruction selection)

| UIR op | x86_64 instruction (scalar SSE2) |
|--------|-----------------------------------|
| `linear` matmul body | `mulss xmm1, [src+idx]` + `addss xmm0, xmm1` |
| `linear` bias add | `addss xmm0, [bias+idx]` |
| `relu` | `xorps xmm1, xmm1` (clear) + `maxss xmm0, xmm1` |
| `dropout` | `movss` (load) + `movss` (store) — pure copy |
| `softmax` exp call | `call <prefix>expf@PLT` (prefix = "" for ELF) |

The exact instruction encoding (e.g. AT&T vs Intel syntax in the
emitted `.s` source) is one decision the plan makes — `cc` understands
either; the convention used matches what's natural for the assembler
(`gas` defaults to AT&T on Linux, but `.intel_syntax noprefix` directive
switches it). Recommendation: AT&T syntax to match `gas` default and
avoid an extra directive.

### 7.4 Fused softmax xmm-spill strategy (architectural)

arm64 holds row-max in `s8` and running-sum in `s9` (both
callee-saved). On x86_64 with no callee-saved FP register, this does
not translate directly. Two stack slots are dedicated per fused-softmax
row:

- `[rsp + max_slot_off]`: row-max (4 bytes, `f32`)
- `[rsp + sum_slot_off]`: running-sum (4 bytes, `f32`)

**Slot positions are pinned to the bottom of the post-`subq` frame:**
`max_slot_off = 0`, `sum_slot_off = 8`. The 16-byte reserve (two 8-byte
aligned f32 slots) is owned by `assign_buffers`: when
`model.calls_extern_math()`, it sets the initial `stack_offset` to 16
instead of 0, so all `BufferLoc::StackOffset(off)` values start at
`off >= 16`. Intermediate buffers then resolve via the existing
`materialise_ptr` form `(off)(%rsp)` without per-emitter parameterisation.
Anchoring the slots to fixed low-frame offsets (rather than parameterising
them by `assignment.stack_bytes`) keeps the emitter templates constant
across models AND prevents slot/buffer overlap when intermediate buffers
are non-empty — e.g. unfused `linear → softmax`, or any classifier with
stack-resident hidden layers, where `8(%rsp)` and `16(%rsp)` would land
inside the linear's intermediate buffer and corrupt the source mid-pass.

**Phase 2 (find max):** scan the row, track current max in `xmm8`
(or any free xmm), `movss [rsp + max_slot_off], xmm8` once at the end
of the phase. The max is now stable in memory and not register-state.

**Phase 3 (exp + sum):** for each element `j`:
```
movss   xmm0, [rdx + i*N*4 + j*4]      ; load element
subss   xmm0, [rsp + max_slot_off]     ; subtract max
call    expf@PLT                        ; xmm0 -> exp(xmm0)
movss   [rdx + i*N*4 + j*4], xmm0      ; write exp result back to output buf
addss   xmm0, [rsp + sum_slot_off]     ; reload sum, accumulate
movss   [rsp + sum_slot_off], xmm0     ; spill new sum
```

**Phase 4 (normalise):** for each element `j`:
```
movss   xmm0, [rdx + i*N*4 + j*4]
divss   xmm0, [rsp + sum_slot_off]
movss   [rdx + i*N*4 + j*4], xmm0
```

Cost vs arm64: +1 memory traffic per element in Phase 3 (sum reload
through stack instead of register). Acceptable; arm64's
register-resident pattern is impossible without callee-saved xmm,
and accepting the cost is a deliberate scoping choice (vs e.g.
introducing a custom callee-saved-xmm convention, which would break
SysV interop).

### 7.5 Stack frame computation

Derivation. On function entry, after the caller's `call` instruction
has pushed the 8-byte return address, `rsp ≡ 8 (mod 16)`. Each
prologue `push reg` (8 bytes) flips parity. After N pushes:

```
rsp ≡ 8 - 8*N (mod 16) ≡ 8*(1 - N) (mod 16)
```

- N even → `rsp ≡ 8 (mod 16)` after pushes; need `frame_size ≡ 8
  (mod 16)` so that `sub rsp, frame_size` lands on a 16-byte boundary.
- N odd → `rsp ≡ 0 (mod 16)` after pushes; need `frame_size ≡ 0
  (mod 16)`.

Therefore the push-count correction adds 8 bytes when N is **even**,
not when it is odd:

```rust
pub fn compute_frame_size(raw_buffer_size: u32, num_pushes: usize) -> u32 {
    let aligned = (raw_buffer_size + 15) & !15;       // round up to 16
    let push_correction = if num_pushes % 2 == 0 { 8 } else { 0 };
    aligned + push_correction
}
```

Unit-tested with at least these cases (entry-state `rsp ≡ 8 (mod 16)`
assumed; final `rsp` after `sub rsp, frame_size` must be `≡ 0 (mod 16)`):

- `(raw=0,  N=0) → 8`   (post-pushes ≡ 8; sub 8 → 0 ✓)
- `(raw=0,  N=1) → 0`   (post-pushes ≡ 0; sub 0 → 0 ✓)
- `(raw=0,  N=2) → 8`   (post-pushes ≡ 8; sub 8 → 0 ✓)
- `(raw=8,  N=0) → 24`  (aligned=16, +8; post-pushes ≡ 8; sub 24 ≡ -16 ≡ 0 ✓)
- `(raw=8,  N=1) → 16`  (aligned=16, +0; post-pushes ≡ 0; sub 16 → 0 ✓)
- `(raw=16, N=1) → 16`  (same alignment as above)
- `(raw=17, N=0) → 40`  (aligned=32, +8; post-pushes ≡ 8; sub 40 ≡ -32 ≡ 0 ✓)
- `(raw=17, N=1) → 32`  (aligned=32, +0; post-pushes ≡ 0; sub 32 → 0 ✓)

Each test case carries the alignment-arithmetic verification inline so
that the helper's contract is self-documenting and a future reader
can re-derive the constants without re-reading SysV §3.2.2.

### 7.6 `sym_prefix()` consumption sites in x86_64

By construction (commit 3 is greenfield), no hardcoded `_` or `""`
prefix anywhere in `profiles/x86_64/src/`. All three sites consume
`self.sym_prefix()`:

1. `<prefix>nfl_forward_<Model>:` — function label
2. `.globl <prefix>nfl_forward_<Model>` — global directive
3. `call <prefix>expf@PLT` — libm call (in standalone softmax and
   fused RowWise tail)

For x86_64 (`sym_prefix() -> ""`), these expand to `nfl_forward_M:`,
`.globl nfl_forward_M`, `call expf@PLT`. A grep at the end of commit 3
sweep verifies no stray `_` or empty literal-string prefixes leaked
in.

### 7.7 Lessons-learned from arm64 (M3-M8 ratchet)

x86_64 is written *with* the M8 lessons baked in:

- **Dropout-as-output is handled from birth.** A `BufferLoc::OutputReg`
  branch in `walk_model::Dropout` calls `emit_dropout_copy`
  (mirror of `emit_relu` minus the maxss). No M8-style retrofit.
- **Immediates > 4095 are not a problem on x86_64.** `mov r10d,
  <imm32>` and `cmp <reg>, <imm32>` accept any 32-bit immediate in a
  single instruction. The arm64 `emit_imm32` complexity (movz + movk
  + hoisting strategies) does not transfer; x86_64's equivalent is
  trivial. No special handling required, no helper needed.

### 7.8 Tests for commit 3

`profiles/x86_64/src/tests.rs` — unit tests on asm shape, mirroring
the structure of `profiles/arm64/src/tests.rs` (45 `#[test]`
functions, 122 substring-asserts inside). Each arm64 test gets an
x86_64 sibling, content adapted:
- `cmp x9, x10` rewritten as `cmp r9, r10`
- `fmadd s0, s1, s2, s0` rewritten as `mulss xmm1, xmm2` + `addss
  xmm0, xmm1` (asserted as two separate substring asserts)
- `bl _expf` rewritten as `call expf@PLT`
- `_nfl_forward_M` rewritten as `nfl_forward_M` (no leading
  underscore)
- `fmov s0, wzr` rewritten as `xorps xmm1, xmm1`

Plus ~8 unit tests on `compute_frame_size` (see §7.5).

FFI integration tests are deferred to commit 5 (cfg-gating + CI wiring
go together).

### 7.9 Commit 3 done criteria

- `cargo build --workspace` green (x86_64 crate compiles)
- `cargo clippy --workspace --all-targets -- -D warnings` green
- `cargo test --workspace`: ~228 + ~45 (x86_64 unit shape) + ~8
  (compute_frame_size) ≈ 281
- `nflc` does not yet expose `--profile x86_64` (still only "arm64"),
  by design — CLI dispatch is commit 4

---

## 8. Commit 4 — CLI dispatch via `Box<dyn Profile>`

### 8.1 `nflc/Cargo.toml` update

Dependencies +=
- `profile-api = { path = "../profile-api" }`
- `profiles-x86_64 = { path = "../profiles/x86_64" }`

(`profiles-arm64` is already a dependency.)

### 8.2 `nflc/src/main.rs` — `run_compile` rewrite

Current shape (M8):
```rust
if profile != "arm64" {
    eprintln!("error: unknown profile '{}' (supported: arm64)", profile);
    return ExitCode::FAILURE;
}
// ... pass pipeline ...
match profiles_arm64::lower(&post_pass_uir) {
    Ok(asm) => { /* ... */ }
    Err(e) => { /* ... */ }
}
```

New shape (M9):
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
// ... pass pipeline ...
match profile_impl.lower(&post_pass_uir) {
    Ok(asm) => { /* ... */ }
    Err(e) => { /* ... */ }
}
```

### 8.3 Help text update

`print_usage()` and the `parse_compile_args` doc comments mention
`x86_64` alongside `arm64`. The `--profile <name>` help string lists
both supported values.

### 8.4 Tests for commit 4

`nflc` currently has no CLI smoke tests — M9 introduces the first ones,
in `nflc/tests/cli.rs` (new integration test file). Three tests added:
- `nflc compile <fixture> --profile x86_64` exits 0 and writes asm
  containing `nfl_forward_<Model>:` (no `_` prefix) and
  `call expf@PLT` to stdout (or to file if `-o` given)
- `nflc compile <fixture> --profile foo` exits 1 with error message
  matching the new "(supported: arm64, x86_64)" wording
- `nflc compile <fixture> --profile arm64` still produces the
  expected `_nfl_forward_<Model>:` and `bl _expf` (regression guard
  for the dispatch refactor)

### 8.5 Commit 4 done criteria

- All workspace gates green
- `cargo test --workspace`: ~281 + ~3 CLI smoke ≈ 284
- Manual smoke: `cargo run -p nflc -- compile tests/fixtures/classifier.nfl --profile x86_64` produces non-empty asm with no `_`-prefix and a `call expf@PLT` line

---

## 9. Commit 5 — CI: enable x86_64 FFI on `ubuntu-latest`

### 9.1 `profiles/x86_64/tests/integration.rs`

Mirror of `profiles/arm64/tests/integration.rs`, with cfg-gating:

```rust
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
```

Test inventory (one-to-one with arm64 except where noted):

| Test name | Fixture | Asserts |
|-----------|---------|---------|
| `classifier_ffi_runs` | classifier.nfl | output shape correct + numerical reference |
| `tiny_mlp_ffi_runs` | tiny_mlp.nfl | same |
| `mixed_args_ffi_runs` | mixed_args.nfl | same |
| `dropout_only_ffi_runs` | dropout_only.nfl | output equals input (dropout = identity at inference) |
| `softmax_with_bias_ffi_runs` | softmax_with_bias.nfl | output sums to 1.0 (within tolerance) |
| `large_classifier_k_ffi_runs` | large_classifier_k.nfl | runs without segfault, output finite |
| `large_classifier_n_ffi_runs` | large_classifier_n.nfl | same |
| `fused_vs_unfused_classifier_match_numerically` | classifier.nfl | bit-exact between `--no-passes` and default |
| `fused_vs_unfused_mixed_args_match_numerically` | mixed_args.nfl | same |
| `fused_vs_unfused_softmax_match_numerically` | softmax_with_bias.nfl | same |
| `fused_softmax_xmm_spill_x86_64` (NEW, x86_64-specific) | softmax_with_bias.nfl | new — see §9.3 |

### 9.2 `profiles/x86_64/tests/common/mod.rs`

Own copy of the helper from `profiles/arm64/tests/common/mod.rs`. The
only thing that changes vs arm64 is the dylib extension expectation:
`cc` crate produces `.so` on Linux and `.dylib` on macOS by default,
so `compile_to_dylib` does not need to hard-code the extension —
`tempfile::NamedTempFile` + `cc::Build::new().shared_flag(true)`
handles it. Verification of `cc` flag correctness is a plan-time step.

### 9.3 `fused_softmax_xmm_spill_x86_64` — explicit xmm-survival proof

This test is x86_64-specific and not present in the arm64 suite. It
exists because the xmm-spill strategy (§7.4) is the most architecturally
divergent code path between the two profiles, and a direct numerical
proof that it works is the strongest signal.

Test body:
1. Parse `tests/fixtures/softmax_with_bias.nfl` to UIR.
2. Run `default_pipeline()` (engages `FuseLinearSoftmax`).
3. Lower via `X86_64Profile.lower()`.
4. Assemble + link via `cc` to a `.so`.
5. `dlopen` + call the FFI symbol with a fixed input vector.
6. Compute reference output in Rust (via `f32::exp` directly).
7. Assert `(output - reference).abs() < 1e-5` element-wise.

The fixture matters: `softmax_with_bias.nfl` has a row dimension > 1,
so Phase 3's `call expf@PLT` is invoked multiple times per row, which
is exactly when xmm-spill correctness manifests (a single-element row
would not differentiate spill-correct from spill-buggy code).

### 9.4 `.github/workflows/ci.yml` update

Change to the existing `unit` job comment to reflect that x86_64 FFI
tests now actually run there (previously: arm64 FFI cfg-skipped on
non-aarch64; now also: x86_64 FFI cfg-runs on linux-x86_64). The job
itself is unchanged; the comment is updated for accuracy.

```yaml
      - name: Test
        # Workspace tests on ubuntu-latest (x86_64 Linux):
        # - profiles/arm64 integration FFI cfg-skips (target_arch != aarch64)
        # - profiles/x86_64 integration FFI runs (target_os = linux,
        #   target_arch = x86_64). Requires cc (build-essential, included
        #   in ubuntu-latest by default).
        run: cargo test --workspace
```

### 9.5 Commit 5 done criteria

- `unit` CI job on `ubuntu-latest` passes with x86_64 FFI tests now
  contributing to the green
- `integration` CI job on `macos-14` passes (arm64 FFI tests
  unchanged)
- `cargo test --workspace` locally on macOS arm64: x86_64 FFI tests
  cfg-skip cleanly, arm64 tests pass; total ≈ 284 (no x86_64 FFI run
  locally). On `ubuntu-latest` (CI): ≈ 284 + ~10 (x86_64 FFI mirror
  set) + 1 (xmm-spill) ≈ 295.

---

## 10. Commit 6 — docs + OQ updates

### 10.1 `docs/profile_guide/x86_64.md` (NEW)

Mirror structure of `arm64.md`. Sections:
1. Overview — Linux ELF scalar SSE2 target
2. ABI — SysV AMD64 summary
3. Register conventions — int + float, callee-saved sets, the
   "no callee-saved FP" divergence from arm64 called out explicitly
4. Supported ops — full parity with arm64 minus SIMD
5. Fused softmax xmm-spill — architectural rationale + stack-slot
   strategy (link to spec §7.4)
6. Libm call form — `call expf@PLT` and PIE/PLT rationale
7. Stack alignment — `compute_frame_size` and the 16-byte invariant
8. Out-of-scope — SIMD/AVX, macOS x86_64, Windows, bare-metal expf

### 10.2 `docs/profile_guide/arm64.md` (UPDATED)

One paragraph in Overview noting coexistence with x86_64. No
structural changes.

### 10.3 `PROJECT_SPEC.md` (UPDATED)

- Profile table: `x86_64` row from "Intel/AMD AVX-512, VNNI (future)" →
  "Linux ELF scalar SSE2: linear (± bias), relu, dropout, softmax (libm
  expf via PLT). Full op-parity with arm64 minus SIMD/AVX. macOS
  x86_64 (Mach-O) and SIMD remain open."
- Strategic Roadmap: Axis 1 annotated `M9 ships scalar Linux ELF;
  SIMD/AVX and macOS x86_64 remain as possible follow-ups.` Axis 1's
  unblock-arrow `→ MACHO_SYM_PREFIX rename` is annotated `closed —
  abstracted as Profile::sym_prefix() in M9.`
- Open Questions / Trigger-driven cleanup:
  - **OQ-NEW** rewritten per outcome of commit 2 sanity check (§6.2):
    either marked "closed in M9 commit 2" (if all sites reduced to
    `calls_extern_math`) or rewritten with a fresh trigger condition
    (if not).
  - **OQ-BENCH** added (NEW): "benchmark harness (single NFL → arm64
    + x86_64, timing side-by-side; documents that the compiler
    produces correct output and lays groundwork for performance
    claims). Trigger: x86_64 profile ships (M9 merged)."

### 10.4 `CLAUDE.md` (UPDATED)

- Repository Structure tree: add `profile-api/` and `profiles/x86_64/`
  to the diagram, with one-line description matching the
  `profiles/arm64/` style entry.
- Current Status: `"Milestone 9 complete. ~295 tests passing."`
  (exact number from green CI run after merge).
- Design Principles §3 (Profile isolation): no rewording needed; the
  principle is unchanged, but its claim is now backed by two
  concrete profiles instead of one.

### 10.5 `README.md` (UPDATED)

Project status section refreshed: "M9 complete — second concrete
profile (`x86_64` Linux ELF, scalar) ships. Single NFL source compiles
to two distinct binaries; profile-isolation hypothesis validated. Full
op-parity with arm64 minus SIMD."

### 10.6 `DEVLOG.md` (UPDATED)

New entry for M9 closure: standard format (What was done / Decisions
made / Problems encountered / Next step). Entry sits at the top above
the "## 2026-05-06 — License pivot" entry. Decisions section
references the brainstorming-session decisions captured in this spec.

### 10.7 Commit 6 done criteria

- All workspace gates green (no test count change vs commit 5; this
  commit is documentation only)
- All listed files updated; no stale references to "arm64 only" or
  "single profile" remain anywhere in the documentation tree
- OQ-NEW is in a settled state (closed or re-opened with a fresh
  trigger); no ambiguous "we'll see" wording
- OQ-BENCH is added with explicit trigger

---

## 11. Test strategy summary

### 11.1 Test inventory after M9

Counts below come from the workspace state captured at brainstorming
time (`cargo test --workspace` on M8 closure: 223 total, decomposed
across binaries as listed). Pre-M9 column is observed; M9 delta is
estimated; the plan refines exact numbers.

| Category | Pre-M9 | Δ M9 | Post-M9 |
|----------|--------|------|---------|
| compiler suite (lib + integration binaries) | 163 | 0 | 163 |
| profiles/arm64 unit (`src/tests.rs`) | 45 | 0 (migration no-op) | 45 |
| profiles/arm64 integration FFI (`tests/integration.rs`) | 15 | 0 | 15 |
| profile-api unit | 0 | +5 | 5 |
| profiles/x86_64 unit (shape + compute_frame_size) | 0 | +53 (~45 mirror + ~8 helper) | 53 |
| profiles/x86_64 integration FFI | 0 | +11 (10 mirror + 1 xmm-spill) | 11 |
| nflc CLI smoke (NEW in M9) | 0 | +3 | 3 |
| **Total** | **223** | **+72** | **~295** |

The "0" for arm64 post-migration is a hard contract per §4.7. Local
macOS arm64 sees ~284 (cfg-skips x86_64 FFI 11). Linux x86_64 CI sees
~295 (cfg-skips arm64 FFI 15, runs everything else).

### 11.2 Cfg-gating matrix

| Test suite | macOS arm64 (local + `integration` CI) | Linux x86_64 (`unit` CI) |
|------------|------------|----------|
| compiler unit | run | run |
| profile-api unit | run | run |
| profiles/arm64 unit | run | run |
| profiles/arm64 integration FFI | **run** | cfg-skip (target_arch != aarch64) |
| profiles/x86_64 unit | run | run |
| profiles/x86_64 integration FFI | cfg-skip (target_os != linux) | **run** |
| nflc CLI smoke | run | run |

Both profile FFI suites run on exactly one CI job each; both arm64 and
x86_64 are exercised end-to-end in CI by exactly one runner.

### 11.3 Numerical correctness contract

x86_64 FFI tests assert numerical agreement with a Rust-computed
reference (`f32::exp` etc.) within `1e-5` element-wise tolerance.
Bit-exact identity is *not* asserted between arm64 and x86_64 outputs
— the two architectures may differ in the last ULP due to FMA vs
two-step `mulss`/`addss` rounding. Tolerance-based agreement is the
correct contract.

For fused-vs-unfused parity (within a single profile), bit-exact
agreement IS asserted (the M5b/M6 contract: fusion preserves
bit-identical outputs against `--no-passes`). This continues to hold
on x86_64 because the fusion is graph-level and does not touch
floating-point ordering within a node.

---

## 12. Documentation updates

Bundled summary (per-commit listing in §10):

| File | Change |
|------|--------|
| `docs/profile_guide/x86_64.md` | NEW |
| `docs/profile_guide/arm64.md` | one-paragraph coexistence note |
| `PROJECT_SPEC.md` | profile table, Strategic Roadmap, OQ-NEW, OQ-BENCH |
| `CLAUDE.md` | Repo Structure tree, Current Status |
| `README.md` | Project status |
| `DEVLOG.md` | new M9 entry |

No documentation changes are made before commit 6. All in-flight
commits (1-5) leave the documentation pinned at M8 wording. This keeps
the docs change surface concentrated, easier to review.

---

## 13. Branch / PR workflow

- Feature branch: `claude/m9-x86_64-profile` from `main`
- Six sequential atomic commits as specified §5-§10
- One PR titled `feat(m9): x86_64 Linux ELF profile + profile-api contract`
- PR body includes the test-count delta and links to this spec + the
  forthcoming plan
- Merge after green CI; closes M9 in workspace

No mid-PR rebases; no commit squashing on merge (preserve atomic
commit history for bisect).

---

## 14. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| arm64 migration introduces non-byte-identical asm | Low | High (commit 2 contract violation) | Capture pre-migration asm baseline at start of commit 2; diff after each migration step. Substring-asserts catch most issues automatically. |
| x86_64 stack-alignment bug → SIGSEGV in FFI test | Medium | High (test SIGSEGVs are hard to localise) | `compute_frame_size` is a small isolatable helper, unit-tested independently first. CI failure mode is loud (test process abort), not silent. |
| `call expf@PLT` linker resolution differs in unforeseen `cc` flag combination | Low | Medium | Plan explicitly tests `cc -shared` + `--profile x86_64` end-to-end on CI before hardware-specific tuning. Falls back to explicit linker invocation if `cc` defaults misbehave. |
| Trait method threading touches more arm64 callsites than expected | Medium | Low (mechanical) | Plan picks one of (a)/(b)/(c) from §6.1 and applies uniformly. Whichever is least invasive in the existing arm64 control flow. |
| `node_uses_softmax` cannot be eliminated cleanly in commit 2 | Medium | Low | OQ-NEW already has an "if not, re-open with fresh trigger" path (§6.2). Not a blocker. |
| `cc` crate produces different output extension on different OSes, breaking helper | Low | Medium | `compile_to_dylib` does not hard-code extension; relies on `tempfile` + `cc::Build::shared_flag(true)`. Verified at plan time. |

No risk in this list rises to "may force re-scoping". The hard-decided
non-goals (no SIMD, no macOS x86_64, etc.) keep the implementation
surface bounded.

---

## 15. Open questions / backlog

After M9, the following remain on the strategic surface (per
`PROJECT_SPEC.md`):

- **Axis 1 follow-ups** (now partial-shipped):
  - SIMD / AVX vectorisation for x86_64
  - macOS x86_64 (Mach-O)
  - Other OSes (Windows, *BSD)
- **Axis 2** (untouched): NFL v0.2 grammar, attention ops, profile-level
  viewer annotations
- **Axis 3** (untouched): bare-metal `expf`, drop libm dependency
- **OQ-7, OQ-8, OQ-9, M5c OQ-4**: trigger-driven cleanup, dormant
- **OQ-NEW**: settled by commit 2 (closed or re-opened with fresh trigger)
- **OQ-BENCH** (NEW, opened by this M9 spec): benchmark harness, trigger
  fires on M9 merge

The next milestone is selected by re-running brainstorming over the
post-M9 Strategic Roadmap. M9 closing does not auto-promote any axis.

---

## 16. Done criteria

Workspace state after PR merge:

- [ ] Six commits in stated order, each commit's gates green
- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo build --workspace` passes
- [ ] `cargo test --workspace` passes on macOS arm64 (local +
      `integration` CI job)
- [ ] `cargo test --workspace` passes on Linux x86_64 (`unit` CI job),
      x86_64 FFI tests included
- [ ] `nflc compile <fixture> --profile arm64` produces byte-identical
      asm to pre-M9 baseline (selected fixtures hand-verified)
- [ ] `nflc compile <fixture> --profile x86_64` produces non-empty asm
      with no `_`-prefix and `call expf@PLT` lines (where softmax is
      involved)
- [ ] All documentation updates landed in commit 6
- [ ] OQ-NEW state settled (closed or re-opened)
- [ ] OQ-BENCH added
- [ ] DEVLOG entry written
- [ ] CLAUDE.md "Current Status" reflects M9 completion

---

*End of spec.*
