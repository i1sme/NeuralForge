# Milestone 8 — ARM64 Codegen Hardening + Viewer v0.1 — Design

> **Status:** Brainstormed and approved 2026-05-06. To be implemented in
> the `claude/sad-tesla-01188d` worktree.
> **Source:** This spec captures the M8 brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M7 closed the M6 holistic-review carry-forward (shared `RewritePlan`
helper). With pass-side cleanup done, M8 turns to two distinct concerns
discovered during a pre-M8 audit of the arm64 profile:

1. **Two real arm64 codegen bugs** that affect correctness on inputs
   our existing test fixtures happen not to exercise:
   - `Dropout` placed at `model.output` produces an uninitialised
     output buffer (HIGH severity).
   - `cmp Xn, #{dim}` and `mov X8, #{dim}` emit literal immediates
     larger than the ARM64 architectural limits (12-bit for `cmp`,
     16-bit for `mov`); current fixtures stay within range by accident
     (MEDIUM severity).
2. **The PROJECT_SPEC milestones row 8 deliverable** — "Human-readable
   viewer v0.1: Show UIR in annotated human-readable format". The
   existing `Display for Uir` plus `nflc parse --uir` provide the
   baseline; v0.1 augments this with a verbose mode that surfaces
   model-level metadata and makes fusion structure visually
   prominent.

M8 ships three atomic commits in one PR, mirroring the M5/M6/M7
single-PR-with-atomic-task-pack convention.

## 2. Goal

Close two arm64 codegen bugs (dropout-as-output, dim-immediate
encoding) and ship the v0.1 human-readable viewer (`--uir-verbose`).
Each commit is independently green; the PR carries no cross-commit
dependencies. End state:

- `nflc compile dropout_only.nfl --no-passes --profile arm64` produces
  asm that correctly copies the dropout input to the output buffer.
- `nflc compile large_classifier.nfl --profile arm64` (with `k > 4095`
  AND a separate fixture with `n > 4095`) assembles, links, and
  produces bit-exact output identical to a reference computation.
- `nflc parse classifier.nfl --uir-verbose` renders an annotated UIR
  including `calls-extern-math: yes/no`, model node counts, and an
  explicit `-> fused: <op>` indent marker for fused post-ops.
- All 208 existing tests + ~14–15 new tests pass.
- `cargo build --workspace`, `cargo clippy --workspace --all-targets
  -- -D warnings`, `cargo fmt --all -- --check`, and `cargo test
  --workspace` are all clean.

## 3. Non-goals

These are explicitly out of scope for M8 (with where they live in
backlog noted):

- **Profile-level annotations in viewer** — per-node footprint in
  bytes, total stack frame size, callee-saved register set. These
  require running arm64 analyzers and would couple the viewer to a
  specific profile, breaking architecture-agnostic UIR rendering.
  *Backlog: M9+ as a separate `--arm64-analyze` flag or similar.*
- **O(L²) parameter-slot lookup** in `codegen.rs::walk_model`
  ([codegen.rs:124-134](../../profiles/arm64/src/codegen.rs)). Two
  `.iter().find(...)` per Linear node. Negligible at v0.1 scale.
  *Backlog: M9+ when transformer-scale models (≥ 50 layers) become
  fixtures.*
- **Dropout FFI tests beyond the output-position case** — the
  bug-trigger is the only path that needs new coverage. Existing
  `mixed_args.nfl` / `tiny_mlp.nfl` continue to exercise the
  `BufferLoc::Alias` path implicitly.
- **`MACHO_SYM_PREFIX` renaming** in [`asm.rs:6`](../../profiles/arm64/src/asm.rs).
  No second profile yet; speculative concern.
- **`LowerError::DimensionOutOfRange`** — `emit_imm32` already
  asserts `value <= u32::MAX`, which is ~1000× any realistic NN
  dimension. No new error variant needed.
- **`resolve_loc` debug-only assertion** ([codegen.rs:217](../../profiles/arm64/src/codegen.rs)).
  UIR builder's monotonic-NodeId invariant guarantees backward-
  pointing aliases by construction; the `debug_assert!` is
  appropriate, not a release-mode bug.
- **Refactoring `node_uses_softmax`** out of `profiles/arm64/src/buffer.rs`
  to call the new `Uir::calls_extern_math()`. Logic is duplicated
  intentionally; *backlog OQ-NEW: lift to single source of truth on
  next predicate-logic change (e.g. when `tanh`-via-libm lands).*

## 4. Pre-decided architectural calls

These were settled during brainstorming and are inputs to writing-
plans, not open questions:

- **Single PR, three atomic commits**. Order: HIGH → MEDIUM → feature.
  No cross-commit dependencies.
- **Commit 1 (`dropout-as-output`)** — fix lives in `codegen.rs` arm
  + new `emit_dropout_copy` in `ops/dropout.rs`. `buffer.rs` is
  unchanged; the existing `BufferLoc::OutputReg` decision stands and
  codegen handles the copy.
- **Commit 1 emitter uses `emit_imm32` from birth.** No
  "patched in Commit 2" debt; the new emitter ships with the new
  pattern. Commit 2 patches exactly 17 pre-existing sites (12 cmp
  + 5 mov), not 18.
- **Commit 2 placement strategy** — uniform path through
  `emit_imm32`, but two placements:
  - Group A (bl-free loops: relu, dropout-copy, matmul body): hoist
    materialise once outside the loop label, register-form `cmp`
    inside.
  - Group B (bl-containing loops: standalone softmax, fused RowWise
    softmax tail): re-materialise at loop top after the label and
    before the `cmp`. `bl _expf` clobbers caller-saved registers,
    so hoisting outside is impossible without adding callee-saved
    state to the prologue (out of scope).
- **Commit 2 mov-site replacement** — when a hoisted register
  already holds the needed value (matmul body: `x15` = n, `x16` = k),
  use `mov x8, x15` / `mov x8, x16` for the stride-load instead of
  re-materialising via `emit_imm32`. Principle: avoid illegal
  immediates, not "always call the helper".
- **Commit 3 uses newtype pattern** — `VerboseUir<'a>(pub &'a Uir)`,
  `VerboseModel<'a>`, `VerboseNode<'a>`, each with their own `Display`
  impl. The default `Display` for `Uir`/`UirModel`/`Node` is
  unchanged. Reasoning: idiomatic Rust composition, no API pollution
  on core types.
- **`calls_extern_math` is a method on `UirModel` and `Uir`** in
  `compiler/src/ir/types.rs`. Two-arm exhaustive `match` on
  `NodeKind` (Input + Op) — `NodeKind` is not `#[non_exhaustive]`,
  matching the style of the existing `Display for Node`
  ([types.rs:138](../../compiler/src/ir/types.rs)).
- **`--uir-verbose` is mutually exclusive with `--uir`.** Both off →
  no UIR printing. Both on → CLI error. Implemented in the existing
  argument-parsing block in `nflc/src/main.rs`.

## 5. Commit 1 — dropout-as-output codegen fix

**Bug.** When a `Dropout` node IS `model.output` (id == model.output),
`assign_buffers` ([buffer.rs:38-42](../../profiles/arm64/src/buffer.rs))
returns `BufferLoc::OutputReg` (write to caller's `x2` pointer). But
[codegen.rs:166-169](../../profiles/arm64/src/codegen.rs) emits ZERO
asm for `StdOp::Dropout`, so the output buffer is never written. The
caller observes uninitialised memory.

This only triggers with `--no-passes` (since `EliminateDropout` would
otherwise remove the dropout) AND with dropout placed at the model's
output position. Semantically odd but legal NFL; codegen must be
correct regardless.

**Fix.** In `codegen.rs::walk_model`, branch on `dst_loc` for the
`StdOp::Dropout` arm. If `BufferLoc::OutputReg`, emit a copy-loop via
new helper `emit_dropout_copy` in `ops/dropout.rs`. Otherwise (the
`BufferLoc::Alias` path), continue emitting nothing.

**`emit_dropout_copy` signature:**

```rust
pub fn emit_dropout_copy(
    total_floats: u64,
    model_idx: usize,
    dropout_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String
```

**Generated asm** — direct mirror of `emit_relu` minus the zero-init
and `fmax` (load → store, no clamp):

```text
    ; dropout-as-output: copy operand→output ({total} elements)
    <materialise_ptr "x11" src_loc>
    <materialise_ptr "x12" dst_loc>
    <emit_imm32 "x10" total_floats as usize>
    mov     x9, #0
.Ldropout_{model_idx}_{dropout_idx}:
    cmp     x9, x10
    b.ge    .Ldropout_end_{model_idx}_{dropout_idx}
    ldr     s3, [x11, x9, lsl #2]
    str     s3, [x12, x9, lsl #2]
    add     x9, x9, #1
    b       .Ldropout_{model_idx}_{dropout_idx}
.Ldropout_end_{model_idx}_{dropout_idx}:
```

Registers: `x9` (counter), `x10` (loop-bound, hoisted via `emit_imm32`),
`x11` (src ptr), `x12` (dst ptr), `s3` (scratch). All caller-saved; no
conflict with softmax callee-saved set (x19-x23, d8-d9). No `bl` —
leaf-safe.

**`codegen.rs` changes:**

```rust
// Add counter near the existing linear_idx/relu_idx/softmax_idx (line ~111):
let mut dropout_idx = 0usize;

// Replace the existing Dropout arm (lines 166-169):
StdOp::Dropout => {
    let src_loc = resolve_loc(&assignment.locs, operands[0]);
    let dst_loc = resolve_loc(&assignment.locs, node_idx);
    if matches!(dst_loc, crate::buffer::BufferLoc::OutputReg) {
        let total: u64 = node.ty.shape.0.iter().product();
        body.push_str(&crate::ops::emit_dropout_copy(
            total, model_idx, dropout_idx, src_loc, dst_loc,
        ));
        dropout_idx += 1;
    }
    // else BufferLoc::Alias: no asm, downstream reads operand directly.
}
```

`pub use dropout::emit_dropout_copy;` added to `ops/mod.rs`.

**New fixture `tests/fixtures/dropout_only.nfl`:**

```nfl
model OnlyDropout [b=2, k=4]:
    x: Tensor[b, k]
    x -> dropout[rate=0.1]
```

**Tests:**

1. **Asm-shape unit** in `profiles/arm64/src/tests.rs` — build UIR
   for `dropout_only`, call `lower()` with `no-passes` semantics
   (skip pipeline), assert `Asm.source` contains the substrings:
   `; dropout-as-output:`, `.Ldropout_0_0:`, `ldr     s3, [x11`,
   `str     s3, [x12`.
2. **FFI integration** in `profiles/arm64/tests/integration.rs` —
   compile `dropout_only.nfl` with `--no-passes`, link, call
   `nfl_forward_OnlyDropout(input, output, params)` with two
   variants:
   - `b=2, k=4` (8 floats): bug-trigger case.
   - `b=1, k=8` (8 floats): single-row variant — closes the audit
     coverage gap on `b=1`.
   Assert: output ≡ input bit-exact in both.
3. **Regression** — existing FFI tests using fixtures with dropout
   NOT at output (`mixed_args.nfl`) continue to pass without any
   asm change for their Dropout nodes. No new test needed; existing
   suite already covers.

## 6. Commit 2 — dim-immediate uniform encoding

**Bug.** ARM64 `cmp Xn, #imm` encodes a 12-bit immediate (0-4095,
optionally shifted by 12). `mov Xn, #imm` encodes a 16-bit immediate
(0-65535, optionally shifted). The arm64 emitters use literal
immediates for loop-bound dims (`b`, `n`, `k`, `total_floats`) at 17
sites:

- **12 `cmp` sites** (loop bounds): [`linear.rs:67,72,78,158,170,183,200`](../../profiles/arm64/src/ops/linear.rs),
  [`relu.rs:27`](../../profiles/arm64/src/ops/relu.rs),
  [`softmax.rs:46,59,74,91`](../../profiles/arm64/src/ops/softmax.rs).
- **5 `mov` sites** (stride values): [`linear.rs:81,86,124,161`](../../profiles/arm64/src/ops/linear.rs),
  [`softmax.rs:50`](../../profiles/arm64/src/ops/softmax.rs).

Current fixtures stay within both ranges (max `n=512`, max `k=784`,
max `b=32`). Any production-scale model — transformer hidden_dim
4096+, LLM vocab 30k+, image classifier 10k classes — fails to
assemble or fails silently.

**Fix.** Uniform path through `emit_imm32`
([asm.rs:103](../../profiles/arm64/src/asm.rs)) which materialises any
`u32` value via `movz` + optional `movk`. Two placement patterns:

**Group A — bl-free loops (single hoist outside loop label):**

| Emitter | Sites | Hoist target | Placement |
|---|---|---|---|
| `emit_relu` | 1 cmp | x10 ← total_floats | Before `mov x9, #0` |
| `emit_linear` matmul body | 3 cmps + 3 movs | x10 ← b, x15 ← n, x16 ← k | Before `mov x3, #0` (i-loop init) |

In matmul, the 3 mov sites become `mov x8, x15` (where stride is `n`)
and `mov x8, x16` (where stride is `k`) — reuse the hoisted register
instead of re-materialising. Single instruction, exact-same effect.

`emit_dropout_copy` (Commit 1) is also Group A but ships correct
from birth — no Commit 2 patch needed.

**Group B — bl-containing loops (re-materialise at loop top):**

| Emitter | Sites | Pattern |
|---|---|---|
| `emit_linear` RowWise softmax tail | 4 cmps + 1 mov | re-materialise after each loop label, before cmp |
| `emit_softmax` standalone | 4 cmps + 1 mov | re-materialise after each loop label, before cmp |

`bl _expf` clobbers caller-saved x10. Materialise pattern at each
loop top:

```text
.Lfsmx_exp_{lid}:
    movz    x10, #lo                  ; new (skipped if value == 0)
    movk    x10, #hi, lsl #16         ; new (skipped if hi == 0)
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_{lid}
    ...
    bl      _expf                     ; clobbers x10
    ...
    b       .Lfsmx_exp_{lid}          ; back to top, materialise re-runs
```

Cost: 1-2 instructions per iter; `bl _expf` is hundreds of cycles.
< 1% relative overhead. Avoiding this would require adding x10 to
the callee-saved set in the prologue/epilogue — out-of-scope blast
radius.

The mov-site at [`linear.rs:161`](../../profiles/arm64/src/ops/linear.rs)
sits inside the i-loop body, before any `bl` in that iteration (the
`bl _expf` is in the inner exp-phase loop that follows). Each i-loop
iteration begins with this mov, so `emit_imm32("x8", n)` emitted
in-place is correct: it runs once per row, the previous iteration's
`bl _expf` is already complete.

**Type conversions.** `Shape.0: Vec<u64>`, dim values cast to
`usize` via `as usize`. Safe on 64-bit hosts (project's only target).
`emit_imm32` asserts `value <= u32::MAX as usize` — fires before
any realistic NN dim hits the limit.

**Register allocation summary.**

In matmul body (Group A):
- `x3, x4, x5` — counters (existing)
- `x8` — stride (existing, now `mov x8, x15` / `mov x8, x16`)
- `x9` — emit_imm32 temp for setup (existing, weight/bias offset)
- `x10` — hoisted b (new)
- `x11, x12` — src/dst ptr (existing)
- `x13, x14` — weight/bias ptr (existing)
- `x15` — hoisted n (new)
- `x16` — hoisted k (new)

x10/x15/x16 are caller-saved, untouched by `materialise_ptr` (which
runs before the hoist) and by anything else in matmul body
(no `bl` calls).

In RowWise tail / standalone softmax (Group B):
- Each `cmp` site materialises its bound into x10 at the loop top
  (after label, before cmp). The mov-site at [`linear.rs:161`](../../profiles/arm64/src/ops/linear.rs)
  uses x8 directly via `emit_imm32("x8", n)`.

**New fixtures:**

- `tests/fixtures/large_classifier_k.nfl`:

  ```nfl
  model LargeK [b=2, k=8192, out=10]:
      x: Tensor[b, k]
      x -> linear[out] -> softmax
  ```

  Tests `cmp x5, #{k}` site (k > 4095) and `mov x8, #{k}` stride.

- `tests/fixtures/large_classifier_n.nfl`:

  ```nfl
  model LargeN [b=2, k=8, out=5120]:
      x: Tensor[b, k]
      x -> linear[out] -> softmax
  ```

  Tests `cmp x4, #{n}` (n > 4095) and `mov x8, #{n}` stride. Also
  exercises softmax loops with `cmp x21, #{n=5120}`.

**Tests:**

1. **Asm-shape unit (positive checks)** in `tests.rs` — for each
   emitter assert specific `movz xT, #...` lines appear immediately
   before the corresponding `cmp Xc, xT` lines. Use `find()` with
   ordering checks rather than negative regex (less brittle, doesn't
   false-positive on `cmp x9, #0` and similar legal small-imm
   sites).
2. **FFI integration (large_classifier_k + large_classifier_n)** —
   compile, link, call, compare bit-exact against numpy-precomputed
   reference output. First tests that actually exceed 4095.
3. **Regression** — all existing FFI tests must produce
   bit-identical numerical output. Asm changes (3 instr extra per
   materialise vs 1), behaviour does not.

## 7. Commit 3 — viewer v0.1 (`--uir-verbose`)

**Goal.** Augment the existing `nflc parse --uir` rendering with a
verbose mode that exposes UIR-level metadata and visually emphasises
fusion structure. Default `--uir` rendering is unchanged.

**`compiler/src/ir/types.rs` additions:**

```rust
pub struct VerboseUir<'a>(pub &'a Uir);
pub struct VerboseModel<'a>(pub &'a UirModel);
pub struct VerboseNode<'a>(pub &'a Node);

impl std::fmt::Display for VerboseUir<'_> { /* see below */ }
impl std::fmt::Display for VerboseModel<'_> { /* see below */ }
impl std::fmt::Display for VerboseNode<'_> { /* see below */ }

impl UirModel {
    pub fn calls_extern_math(&self) -> bool {
        self.nodes.iter().any(|n| match &n.kind {
            NodeKind::Op { op, fused_post_ops, .. } =>
                matches!(op, StdOp::Softmax)
                    || fused_post_ops.iter().any(|p| matches!(p, PostOp::SoftmaxRow)),
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

`NodeKind` is not `#[non_exhaustive]`; the 2-arm match style matches
existing `Display for Node` ([types.rs:138](../../compiler/src/ir/types.rs)).
No wildcard.

**Verbose output format** (example, `classifier.nfl`):

```text
uir-verbose summary
  models: 1
  total nodes: 5
  calls-extern-math: yes

uir-model Classifier
  inputs: [n0]
  output: n4
  node count: 5
  calls-extern-math: yes

  n0: input "x"           :: [32, 784]
  n1: linear              :: [32, 512]   operands=[n0]   attrs=[bias=true]
       -> fused: Relu
  n2: linear              :: [32, 256]   operands=[n1]
       -> fused: Relu
  n3: linear              :: [32, 10]    operands=[n2]
  n4: softmax             :: [32, 10]    operands=[n3]
```

Differences vs default `Display for Uir`:
1. **Top-level summary** (4 lines): models count, total nodes,
   `calls-extern-math: yes/no`.
2. **Per-model summary** (2 added lines after inputs/output):
   `node count: N`, `calls-extern-math: yes/no`.
3. **Fused post-ops on separate indented line** prefixed with
   `-> fused: <op>` (ASCII, project convention). The default form
   `fused=[Relu]` is preserved as a single-line attribute in the
   non-verbose `Display`; verbose mode breaks it out.
4. **Blank lines** between top-level summary and models, and between
   models, for readability.

**`nflc/src/main.rs` changes:**

```rust
// Add to argument-parsing block (~line 107):
"--uir-verbose" => { print_uir_verbose = true; }

// Mutual-exclusion check (after parsing all args):
if print_uir && print_uir_verbose {
    return Err("--uir and --uir-verbose are mutually exclusive".to_string());
}

// In parse-subcommand handler (after UIR build):
if print_uir {
    println!("{}", uir);
} else if print_uir_verbose {
    println!("{}", VerboseUir(&uir));
}
```

Help text in [`main.rs:60`](../../nflc/src/main.rs) area: add a line for
`--uir-verbose` parallel to the existing `--uir`.

**Tests:**

1. **`calls_extern_math` predicate unit** in `compiler/src/ir/types.rs::tests`
   — three sub-cases: model with standalone `Softmax` → true; without →
   false; with fused `SoftmaxRow` only → true.
2. **`VerboseUir` snapshot** in same module — build UIR from
   `tests/fixtures/classifier.nfl` (or hand-built equivalent), format
   via `VerboseUir`, compare against literal expected-string. Pins
   format; future intentional changes update expected.
3. **CLI smoke** in `nflc/tests/cli.rs` (or extending the existing
   smoke harness) — invoke `nflc parse classifier.nfl --uir-verbose`
   via `Command::new`, assert stdout contains `uir-verbose summary`,
   `calls-extern-math: yes`, `-> fused:`.
4. **Mutual exclusion smoke** — `nflc parse f.nfl --uir --uir-verbose`
   exits with non-zero status and prints the mutual-exclusion error.
5. **Regression** — existing `--uir` smoke test continues to pass
   unchanged.

## 8. Test strategy summary

| Category | Commit 1 | Commit 2 | Commit 3 |
|---|---|---|---|
| Unit asm/output | ✓ shape pin (1) | ✓ positive movz+cmp pairs (multiple) | ✓ verbose snapshot (1) |
| Predicate unit | — | — | ✓ calls_extern_math (3 sub-cases) |
| FFI integration | ✓ dropout_only b=2 + b=1 (2) | ✓ large_classifier_k + large_classifier_n (2) | — |
| CLI smoke | — | — | ✓ --uir-verbose + mutual-exclusion (2) |
| Regressions | ✓ existing dropout-as-Alias paths | ✓ all existing FFI bit-exact | ✓ existing --uir smoke |

Approximate new test count: 14–15 (counting positive-check assertions
per emitter as separate tests). Total moves from 208 → ~222–223,
maintaining the CLAUDE.md monotonic-growth requirement.

## 9. Documentation updates

| File | Change |
|---|---|
| `DEVLOG.md` | New entry `## 2026-MM-DD — Milestone 8 closed: arm64 codegen hardening + viewer v0.1`, standard template (What was done / Decisions / Problems / Next step). |
| `PROJECT_SPEC.md` | M8 row in milestones table — extend "Show UIR in annotated human-readable format" to a multi-clause line covering the codegen fixes + viewer v0.1, mirroring the granularity of M5/M6/M7 rows. |
| `CLAUDE.md` "Current Status" | Full rewrite reflecting M8 closure. New carry-forward list: OQ-NEW (`node_uses_softmax` duplicate), OQ-7/8/9 if not closed, plus any new candidates surfaced during M8. |
| `CLAUDE.md` Design Principle 5 | `(M8+)` → `(M9+)` — viewer v0.1 has shipped. |
| `docs/language_reference/uir.md` | New section "Viewing UIR" — describes `--uir` (compact, default `Display`) and `--uir-verbose` (annotated mode, `calls-extern-math` predicate, `-> fused:` indent marker). UIR semantics unchanged; this section documents the rendering interface. |
| `docs/profile_guide/arm64.md` | Two new short paragraphs in the existing structure: "Dropout-as-output copy" describing the codegen branch, "Dim-immediate uniformity" describing Group A vs Group B materialisation strategy. |
| `language/grammar.ebnf` | Unchanged (no NFL grammar work). |
| `Cargo.toml` workspace | Unchanged (no new crates). |

## 10. Branch / PR workflow

- **Worktree:** `claude/sad-tesla-01188d` (current).
- **Branch name:** `claude/sad-tesla-01188d` (matches worktree).
- **Commit order:** Commit 1 → Commit 2 → Commit 3, each green
  workspace-wide before the next.
- **Holistic review** after Commit 3 — single subagent dispatch,
  full-tree audit (spec / structure / cross-cutting / docs / process).
  Any close-in-M8 findings get a separate `chore(m8/holistic):
  close drift-fix findings before M8 closeout` commit, mirroring
  M7's `4974cd7` pattern.
- **Final commit:** `chore(m8): close Milestone 8 — full cycle complete`
  with PROJECT_SPEC + CLAUDE.md + DEVLOG updates.
- **PR title:** `M8: ARM64 codegen hardening + viewer v0.1` (or
  similar short form).
- **PR body:** links to commits 1-3 + holistic + closeout, summary
  of changes, list of new fixtures, test count delta.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Commit 2 introduces a numerical regression | Bit-exact existing FFI test suite gates merge. Any divergence blocks. |
| Commit 3 verbose-format snapshot becomes stale | Snapshot tests are easy-update — one-line change in expected-string. Not lock-in; format can evolve in M9+. |
| Atomic commits break under rebase | Commits are independent (no shared diff). Recreating is straightforward; nothing is lost. |
| Post-merge bug surfaces in Commit 2 | `git revert <hash>` restores literal-imm form. Commits 1/3 do not depend on Commit 2 and remain valid. |
| Holistic review uncovers carry-forward debt | Standard process: close-in-M8 if quick (< 20 lines, single concern), else carry to M9+ with explicit trigger documented. |

## 12. Open questions / backlog

**Closed during brainstorming** (decisions documented above):
- Commit 1 fix location (codegen.rs vs buffer.rs) — codegen.rs.
- Commit 2 placement strategy (uniform vs conditional) — uniform via
  `emit_imm32`, two placements (Group A hoist vs Group B re-materialise).
- Commit 3 architecture (newtype vs method) — newtype.
- Predicate naming (`leaf` vs `calls_extern_math`) —
  `calls_extern_math` (UIR-native).
- Viewer scope (UIR-level vs profile-level) — UIR-level only for v0.1.

**Inline decisions during implementation** (not blockers for plan):
- Exact wording of `arm64.md` paragraphs (one short paragraph each
  for the two changes; not a full new §).
- Exact register choice in standalone softmax for Group B (likely x10
  uniformly; confirm during code-write that no inner-iteration
  dependency exists).

**Carried to M9+ backlog:**
- **OQ-NEW** — Lift `node_uses_softmax` (in `profiles/arm64/src/buffer.rs`)
  to call `Uir::calls_extern_math()` introduced in Commit 3.
  *Trigger:* next change to either side's predicate logic (e.g.,
  adding `tanh`-via-libm support).
- **OQ-7** — Per-pass `Result<UirModel, PassError>` cleanup. From M7.
  *Trigger:* first real `Err`-case in pass-level logic.
- **OQ-8** — Lift `compiler/src/passes/rewriter.rs` to `compiler/src/ir/`.
  From M7. *Trigger:* non-pass UIR-rewrite consumer appears.
- **OQ-9** — Generalise `producer_post_ops: Vec<PostOp>` to
  `enum NodeMutation`. From M7. *Trigger:* fourth pass needs
  non-PostOp producer mutation.
- **Profile-level viewer annotations** — per-node footprint, stack
  frame, callee-saved set. *Trigger:* user request OR x86_64 profile
  appearing (validates the profile-agnostic split).
- **`MACHO_SYM_PREFIX` rename** — `ARM64_SYM_PREFIX` or per-OS
  abstraction. *Trigger:* second profile (x86_64 or riscv64) starts.

## 13. Done criteria

- [ ] Commit 1 `feat(m8/arm64-fix): correct dropout-as-output codegen`
  pushed; workspace green; 2 new FFI tests pass.
- [ ] Commit 2 `feat(m8/arm64-fix): hoist dim immediates through emit_imm32`
  pushed; workspace green; 2 new FFI tests pass; all existing FFI
  tests bit-exact.
- [ ] Commit 3 `feat(m8/viewer): UIR-verbose annotation mode`
  pushed; workspace green; CLI smoke + verbose snapshot + predicate
  unit tests pass.
- [ ] Holistic review subagent dispatched; any close-in-M8 findings
  resolved in `chore(m8/holistic)` commit.
- [ ] Closeout commit `chore(m8): close Milestone 8 — full cycle
  complete` updates PROJECT_SPEC, CLAUDE.md, DEVLOG, uir.md,
  arm64.md.
- [ ] PR opened, all CI checks green, ready for merge.
- [ ] Test count: 208 → 222–223.
- [ ] `cargo build --workspace`, `cargo clippy --workspace --all-targets
  -- -D warnings`, `cargo fmt --all -- --check`, `cargo test --workspace`
  all clean on closeout commit.
