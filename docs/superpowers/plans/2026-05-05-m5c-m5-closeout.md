# M5c — Milestone 5 close-out (docs sync + small consistency fixes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (Task 1) or superpowers:executing-plans (Tasks 2–5) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Milestone 5. Apply the 17 findings from the M5b post-merge holistic review: 4 small Rust consistency fixes (~6 lines total, including the `#[non_exhaustive]`-cascade in `codegen.rs`) and 13 documentation corrections across `PROJECT_SPEC.md`, `docs/profile_guide/arm64.md`, `docs/language_reference/uir.md`, and `CLAUDE.md`. After M5c: PROJECT_SPEC milestone table accurately reflects M5 closed; the M4b-era profile guide is M5b-current; `uir.md` matches the post-M5a UIR shape; CLAUDE.md "Current Status" reflects M5 closed.

**Architecture:** No new architecture. M5c is *consistency work* — every change closes a specific drift between code and its supporting documentation, or between two related types (e.g., `PassError`/`LowerError` gaining `impl std::error::Error` to match `BuildError`/`ParseError`/`LexError`).

**Tech Stack:** Rust 2021, three-crate workspace (unchanged). Markdown for docs. No new dependencies.

**Source punch-list:** Holistic review verdict on M5b PR (2026-05-05). 17 findings classified as CLOSE-IN-M5C (Option B scope). Findings 1.2 (shared Diagnostic trait), 2.1 (BuildError span() accessor), 4.1 (test-helper extraction), 6.1 (pass struct visibility), DEVLOG-1 (debug_assert_eq! → assert_eq!) explicitly deferred to M6+.

**Baseline at branch cut:** 189 tests (post-M5b-merge `60cccb9`). **Target:** 189 (no test changes; code fixes are consistency-level, no behaviour change).

---

## File map

Files this plan touches:

- **Modify** `compiler/src/passes/mod.rs` — add `impl std::error::Error for PassError {}`. (Task 1, Finding 1.1)
- **Modify** `profiles/arm64/src/types.rs` — add `impl std::error::Error for LowerError {}`. (Task 1, Finding 1.1)
- **Modify** `nflc/src/main.rs` — change `&e.message` → `&e.to_string()` on line 253 (ParseError rendering call site). (Task 1, Finding 2.2)
- **Modify** `compiler/src/ir/stdlib.rs` — add `#[non_exhaustive]` to `pub enum StdOp`. (Task 1, Finding 5.1)
- **Modify** `profiles/arm64/src/codegen.rs` — add `_` wildcard arm to `walk_model::match op { ... }` to cover the new `#[non_exhaustive]` constraint. (Task 1, transitive of Finding 5.1)
- **Modify** `PROJECT_SPEC.md` — milestones-table M5 row update + Open Questions cleanup. (Task 2, Findings 3.1, 3.2, 3.3)
- **Modify** `docs/profile_guide/arm64.md` — status header, §3 Relu fusion note, new §4 fused linear→relu pattern, §5 errors table `UnsupportedPostOp`, §8 Limitations rewrite. (Task 3, Findings 7.1, 7.2, 7.3, 7.4)
- **Modify** `docs/language_reference/uir.md` — line 17 `profiles/generic/` → `profiles/arm64/`, §2 `NodeKind::Op` add `fused_post_ops`, §2 immutability rationale (M5 mutation claim), §7 mutation→functional. (Task 4, Findings 7.5, 7.6, 7.7)
- **Modify** `CLAUDE.md` — Principle 5 viewer caveat, "What NOT to Do" viewer line caveat, "Adding a new architecture profile" recipe `profiles/generic/` → `profiles/arm64/`, "Current Status" updated to reflect M5 closed. (Task 5 closeout, Findings 8.1, 8.2)
- **Modify** `DEVLOG.md` — append M5c entry above the M5b entry. (Task 5)

Files NOT touched in M5c:
- `compiler/src/ir/build.rs`, `parser/`, `lexer/` — no findings.
- `compiler/src/passes/fuse_linear_relu.rs`, `compiler/src/passes/eliminate_dropout.rs` — no findings.
- `profiles/arm64/src/ops/`, `profiles/arm64/src/buffer.rs` — no findings.
- `profiles/arm64/tests/integration.rs`, `nflc/tests/cli_compile.rs`, `compiler/src/passes/tests.rs` — no findings (test-helper extraction explicitly deferred to M6+).
- `language/grammar.ebnf`, `docs/language_reference/grammar.md` — no findings (NFL v0.1 grammar frozen since M1).

---

## Task overview

| # | Task | Mode | Findings closed | Net tests |
|---|---|---|---|---|
| 1 | Code consistency fixes | SUBAGENT | 1.1, 2.2, 5.1 (+ codegen.rs cascade) | 0 |
| 2 | `PROJECT_SPEC.md` updates | INLINE | 3.1, 3.2, 3.3 | 0 |
| 3 | `docs/profile_guide/arm64.md` updates | INLINE | 7.1, 7.2, 7.3, 7.4 | 0 |
| 4 | `docs/language_reference/uir.md` updates | INLINE | 7.5, 7.6, 7.7 | 0 |
| 5 | Closeout — `CLAUDE.md` + `DEVLOG.md` | INLINE | 8.1, 8.2 + M5 close | 0 |
|   | **Total** | | **13 findings (+ 4 deferred to M6+)** | **0** |

Test count: 189 → 189. Code fixes are consistency-level, no behaviour change. `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `cargo test --workspace` must remain clean throughout.

---

## Task 1: Code consistency fixes

**Goal:** Close holistic-review code findings 1.1, 2.2, and 5.1 in a single atomic commit. The `#[non_exhaustive]` addition on `StdOp` requires a downstream wildcard arm in `profiles/arm64/src/codegen.rs::walk_model` to keep the build clean (cross-crate exhaustiveness rule). Bundling all four sub-changes in one commit ensures every commit on the branch leaves `cargo build` green.

**Files:**
- Modify: `compiler/src/passes/mod.rs` (add `impl Error`)
- Modify: `profiles/arm64/src/types.rs` (add `impl Error`)
- Modify: `nflc/src/main.rs` (1-line consistency fix)
- Modify: `compiler/src/ir/stdlib.rs` (`#[non_exhaustive]`)
- Modify: `profiles/arm64/src/codegen.rs` (wildcard arm)

- [ ] **Step 1: Add `impl std::error::Error for PassError`**

In `compiler/src/passes/mod.rs`, locate the existing `impl std::fmt::Display for PassError` block. Append a `std::error::Error` impl directly after it:

```rust
impl std::error::Error for PassError {}
```

The `std::error::Error` trait requires only `Debug + Display` as supertraits, both of which `PassError` already satisfies (`#[derive(Debug)]` on the enum + the `Display` impl above). The empty body uses default trait methods (`source()` returns `None`, etc.), which is correct: `PassError::InvalidInput` carries `String`-typed reason, no inner error to chain.

- [ ] **Step 2: Add `impl std::error::Error for LowerError`**

In `profiles/arm64/src/types.rs`, locate the existing `impl std::fmt::Display for LowerError` block. Append a `std::error::Error` impl directly after it:

```rust
impl std::error::Error for LowerError {}
```

Same rationale as Step 1. After this, all five workspace error types (`BuildError`, `ParseError`, `LexError`, `PassError`, `LowerError`) implement `std::error::Error` consistently.

- [ ] **Step 3: Change `&e.message` → `&e.to_string()` in `nflc/src/main.rs:253`**

Locate line 253 of `nflc/src/main.rs` (in the `parse` subcommand's `Err(e)` arm where `compiler::parse` returns `Err`). The current line:

```rust
render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
```

Change to:

```rust
render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
```

`ParseError` implements `Display`, and `Display` for `ParseError` returns `self.message` (verify by reading the existing `Display` impl). This is functionally equivalent but matches the call-site style used for the other four error types: every other `render_error_with_snippet` call uses `&format!("{}", e)` or `&e.to_string()`. After this change, all five rendering call sites in `main.rs` use the same pattern.

- [ ] **Step 4: Add `#[non_exhaustive]` to `pub enum StdOp` in `compiler/src/ir/stdlib.rs`**

Locate the existing `pub enum StdOp` declaration (around line 7-13 of `compiler/src/ir/stdlib.rs`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
}
```

Add `#[non_exhaustive]` directly above the existing attributes:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
}
```

This is forward-compat-only: external crates (i.e., `profiles/arm64`) must now add a wildcard arm to any `match op` block. Same-crate matches (within `compiler/src/ir/stdlib.rs`'s own `signature`, `infer_output_shape`, `validate_attrs`, and `Display for StdOp`) are unaffected — Rust does not enforce `#[non_exhaustive]` within the defining crate.

After adding this, `cargo build --workspace` will fail in `profiles/arm64/src/codegen.rs::walk_model` because the `match op` block has no wildcard arm. Fix in Step 5 (must land in the same commit).

- [ ] **Step 5: Add wildcard arm to `walk_model::match op { ... }` in `profiles/arm64/src/codegen.rs`**

Locate the `match op` block in `walk_model` (around lines 113-185 of `profiles/arm64/src/codegen.rs`). It currently has four arms (`StdOp::Linear`, `StdOp::Relu`, `StdOp::Dropout`, `StdOp::Softmax`) and no wildcard. After Step 4's `#[non_exhaustive]`, this block won't compile.

Add a wildcard arm at the end of the match (just before the closing `}` on line ~185):

```rust
                StdOp::Softmax => {
                    // ... existing softmax arm ...
                    softmax_idx += 1;
                }
                // M5c: #[non_exhaustive] on StdOp requires a wildcard
                // arm. Future ops (e.g. Tanh, Gelu, Embedding) will
                // route here until codegen learns them. Returning
                // LowerError::UnsupportedOp keeps the failure mode
                // graceful (the existing variant is `#[allow(dead_code)]`
                // — this arm makes it live).
                _ => {
                    return Err(crate::types::LowerError::UnsupportedOp {
                        op: format!("{op}"),
                        span: node.source_span,
                    });
                }
            }
```

Two things to note:
1. The `LowerError::UnsupportedOp { op: String, span: Span }` variant already exists in `profiles/arm64/src/types.rs` and was marked `#[allow(dead_code)]` (M4b legacy — defensive variant for "future ops before codegen catches up"). Adding the wildcard arm makes the variant live, so the `#[allow(dead_code)]` attribute on `UnsupportedOp` may now be removable.
2. **Remove the `#[allow(dead_code)]` attribute on `LowerError::UnsupportedOp`** in `profiles/arm64/src/types.rs` if it's present (it was added in M4b when no code path constructed the variant). After Step 5, the variant is reachable in principle, so the attribute is unneeded.

Verify by grep:

```bash
grep -n "allow(dead_code)" profiles/arm64/src/types.rs
```

If the attribute appears on `UnsupportedOp`, remove it. If not, skip this sub-step.

The `format!("{op}")` call uses `Display for StdOp` which returns the lowercase variant name (e.g., `"linear"`, `"future_op"`). For variants outside the current four, the implementer of the new variant is expected to also extend `Display for StdOp` — that's the M6+ author's responsibility.

- [ ] **Step 6: Verify build + clippy + fmt + tests**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && cargo fmt --all && cargo build --workspace 2>&1 | tail -3 && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && echo "---TESTS---" && cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: `TOTAL: 189`. No test count change — these are all consistency fixes.

If clippy complains about the `format!("{op}")` (e.g., `clippy::uninlined_format_args` or `clippy::useless_format`), use the inlined form `format!("{op}")` (already current) or fall back to `op.to_string()` if clippy prefers it. Either is fine.

- [ ] **Step 7: Verify the `_` arm is genuinely unreachable today (sanity check)**

The `_` arm we added is unreachable on today's `StdOp` (Linear, Relu, Dropout, Softmax — all explicitly matched). Confirm by checking all integration tests still pass:

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && cargo test -p profiles-arm64 --test integration 2>&1 | tail -10
```

Expected: all integration tests pass. The `_` arm is defensive future-proofing, not a behaviour change.

If clippy emits an `unreachable_patterns` warning on the `_` arm (because all four `StdOp` variants are explicitly matched), suppress with `#[allow(unreachable_patterns)]` directly on the arm — same pattern M5a Task 6 used for `PostOp`'s wildcard:

```rust
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(crate::types::LowerError::UnsupportedOp { ... });
                }
```

This is consistent with how `profiles/arm64/src/ops/linear.rs` handles the `#[non_exhaustive] PostOp` wildcard.

- [ ] **Step 8: Commit**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && git add compiler/src/passes/mod.rs profiles/arm64/src/types.rs nflc/src/main.rs compiler/src/ir/stdlib.rs profiles/arm64/src/codegen.rs && git commit -m "$(cat <<'EOF'
chore(m5c): cross-cutting consistency fixes (Findings 1.1, 2.2, 5.1)

Holistic-review punch-list, three small consistency improvements:

Finding 1.1 — error trait conformance:
- impl std::error::Error for PassError (compiler/src/passes/mod.rs)
- impl std::error::Error for LowerError (profiles/arm64/src/types.rs)

After this, all five workspace error types (BuildError, ParseError,
LexError, PassError, LowerError) implement std::error::Error
uniformly. Enables ? -based propagation into Box<dyn Error>
chains if M6+ adds such context. Both impls are zero-method
(Debug + Display already satisfy the bound).

Finding 2.2 — call-site rendering consistency:
- nflc/src/main.rs:253: &e.message → &e.to_string() (ParseError
  call site). Now all five render_error_with_snippet call sites
  use the same idiom (&format!("{e}") or &e.to_string()).
  Functionally identical (Display for ParseError delegates to
  self.message); pure visual consistency.

Finding 5.1 — forward-compat for StdOp:
- #[non_exhaustive] added to compiler::ir::stdlib::StdOp.
- Cascade: profiles/arm64/src/codegen.rs::walk_model match block
  needed a wildcard arm. Routes future ops to existing
  LowerError::UnsupportedOp variant (M4b-era, was #[allow(dead_code)]
  — now live, attribute removed).

PostOp was made #[non_exhaustive] in M5a; StdOp matched in M5c
for symmetry. M6+ ops (Tanh, Gelu, attention) won't break
downstream match users in external crates.

189 tests pass; clippy/fmt clean.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `PROJECT_SPEC.md` updates

**Goal:** Close findings 3.1, 3.2, 3.3. Update the milestones table M5 row to reflect what shipped (5a + 5b + 5c). Retire two answered questions from the Open Questions section.

**Files:**
- Modify: `PROJECT_SPEC.md`

This task is INLINE: pure markdown edits, no testable behaviour, low risk.

- [ ] **Step 1: Update the M5 milestone row (lines 158)**

In `PROJECT_SPEC.md`, locate the milestones table (lines 152-160). The current M5 row:

```markdown
| 5 | Kernel fusion pass                             | Fuse linear+activation in the IR optimiser        |
```

Replace with:

```markdown
| 5 | Kernel fusion + UIR-pass framework (5a + 5b + 5c complete) | UIR-pass infrastructure (`UirPass` trait, `default_pipeline`, `run_pipeline`, `PassError`); two passes shipped — `FuseLinearRelu` (bias-aware: fuses `linear → relu` and `linear[bias=true] → relu`) and `EliminateDropout` (removes inference-time-noop Dropout); CLI gains `--no-passes` and `--passes <list>` filter; bit-exact equivalence proven via `fused_vs_unfused_*_match_numerically` integration tests on classifier and mixed_args fixtures |
```

This combines findings 3.1 (completion marker + sub-milestone breakdown) and 3.3 (goal expansion to include UIR-pass framework, EliminateDropout, CLI flags).

- [ ] **Step 2: Clean up Open Questions section (lines 164-170)**

In `PROJECT_SPEC.md`, locate the Open Questions section. Currently:

```markdown
## Open Questions

- Final syntax decisions for NFL (to be designed incrementally)
- Memory model: static allocation only, or dynamic?
- Training syntax design: when and how to introduce loss/optimiser constructs (planned for v0.2)
- How profiles handle quantisation (INT8, FP16, BF16)?
- Distribution format for compiled binaries
```

Two questions are now answered:
1. *"Final syntax decisions for NFL"* — NFL v0.1 grammar is frozen since M1 (`language/grammar.ebnf`). Future syntax (v0.2+) is a separate concern, not an open question for v0.1.
2. *"Memory model: static allocation only, or dynamic?"* — M4 implicitly decided: static stack-allocated intermediate buffers (see `profiles/arm64::buffer.rs::assign_buffers` and the M4b spec). No heap. The decision is documented in the M4b DEVLOG.

Replace the section with:

```markdown
## Open Questions

- Training syntax design: when and how to introduce loss/optimiser constructs (planned for v0.2)
- How profiles handle quantisation (INT8, FP16, BF16)?
- Distribution format for compiled binaries

## Decisions (formerly open, now resolved)

- **NFL v0.1 grammar** — frozen as of M1 (`language/grammar.ebnf`). Future syntax extensions land in NFL v0.2+ as separate language milestones.
- **Memory model** — static stack-allocated intermediate buffers, no heap. Established by M4b (`profiles/arm64::buffer.rs::assign_buffers`); applies to all v1 profiles.
```

This closes findings 3.2 (retire answered questions) cleanly. The "Decisions" sub-section preserves the historical record without leaving the questions as open-looking-but-actually-resolved.

- [ ] **Step 3: Verify markdown renders correctly**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && head -200 PROJECT_SPEC.md | tail -60
```

Visual check: M5 row is one line in the table (does not break table formatting), Open Questions has 3 remaining items, Decisions section has 2 items.

- [ ] **Step 4: Commit**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && git add PROJECT_SPEC.md && git commit -m "$(cat <<'EOF'
docs(m5c): update PROJECT_SPEC for M5 close-out (Findings 3.1, 3.2, 3.3)

Holistic-review punch-list:

Finding 3.1 + 3.3 — milestones-table M5 row:
The original goal "Fuse linear+activation in the IR optimiser" was
too narrow for what shipped across 5a+5b+5c. Updated row to:
- Mark M5 complete (5a + 5b + 5c).
- Describe the actual deliverables: UIR-pass framework (UirPass
  trait, default_pipeline, run_pipeline, PassError), two passes
  (FuseLinearRelu bias-aware + EliminateDropout), --no-passes /
  --passes CLI, bit-exact integration tests.

Finding 3.2 — Open Questions cleanup:
Two of five questions were resolved during M1-M4. Moved to a new
"Decisions (formerly open, now resolved)" sub-section that
preserves the historical record:
- NFL v0.1 grammar frozen at M1.
- Memory model: static stack allocation, established by M4b.

The other three questions remain genuinely open (training syntax,
quantisation, distribution format).

Finding 3.4 from the holistic review (claimed "M5 introduces
mutation" text in PROJECT_SPEC.md §4) was a false positive — that
text doesn't exist in PROJECT_SPEC.md. The actual mutation drift
is in docs/language_reference/uir.md (Findings 7.6, 7.7) and
addressed in Task 4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `docs/profile_guide/arm64.md` updates

**Goal:** Close findings 7.1, 7.2, 7.3, 7.4. The profile guide is M4b-era; M5a/M5b shipped meaningful changes (kernel fusion default-on, `UnsupportedPostOp` error variant, fmax-fmadd inline pattern). The `"No fusion"` claim in §8 Limitations is the most embarrassing — it actively contradicts what default `nflc compile` does.

**Files:**
- Modify: `docs/profile_guide/arm64.md`

This task is INLINE: pure markdown edits, fairly extensive. Five sub-changes.

- [ ] **Step 1: Update status header (line 1-9)**

Current header:

```markdown
# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M4b complete (NFL v0.1). Lowers `linear` (with or without
> `bias=true`), `relu`, `dropout` (no-op pass-through at inference), and
> `softmax` (numerically stable 3-pass via libm `expf`) to native AArch64
> Mach-O assembly. All 5 M3 positive fixtures + the M4a fixture run
> end-to-end via FFI.
> **Authoritative source:** `profiles/arm64/src/` and the M4a/M4b specs under
> `docs/superpowers/specs/`.
```

Replace with:

```markdown
# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M5b complete (NFL v0.1). Lowers `linear` (with or without
> `bias=true`), `relu`, `dropout` (no-op pass-through at inference), and
> `softmax` (numerically stable 3-pass via libm `expf`) to native AArch64
> Mach-O assembly. The compiler runs the default UIR-pass pipeline
> (`EliminateDropout` + `FuseLinearRelu`) before lowering, so
> dropout-containing models reach the profile already with dropout
> removed, and `linear → relu` (with or without bias) reach as a
> single fused Linear with `fused_post_ops: [Relu]`. All 5 M3
> positive fixtures + the M4a fixture run end-to-end via FFI; bit-exact
> equivalence between fused and unfused asm proven on classifier.nfl
> and mixed_args.nfl integration tests.
> **Authoritative source:** `profiles/arm64/src/` and the M4a/M4b/M5a/M5b
> specs under `docs/superpowers/specs/`.
```

- [ ] **Step 2: §3 Supported ops table — add Relu fusion note (line 91-101)**

Locate §3 (around line 91-101). The current Relu row says:

```markdown
| `Relu`                     | ✅        | Separate elementwise loop, copy-with-clamp from src to dst.   |
```

The Relu row in M5b only describes the *unfused* form. Replace the Relu and Linear rows with these (note: also tweak the Linear-no-bias and Linear-bias rows to mention fused-relu):

```markdown
| `Linear` (no `bias` attr)  | ✅        | Pure matmul. With `fused_post_ops: [Relu]` (default-pipeline output of `linear → relu`): adds inline `fmax s0, s0, s4` post-op before store — see §4.9. |
| `Linear` (`bias=true`)     | ✅        | Matmul + per-output bias-add inline. With `fused_post_ops: [Relu]` (default-pipeline output of `linear[bias=true] → relu`): bias-add then inline `fmax` then store — see §4.9. |
| `Relu`                     | ✅        | Standalone (only in `--no-passes` mode, or `--passes` filter excluding `fuse_linear_relu`): separate elementwise loop, copy-with-clamp src→dst (§4.2). Default mode: fused into preceding Linear via `FuseLinearRelu` UIR pass — see §4.9. |
| `Dropout`                  | ✅        | Standalone (only in `--no-passes` mode, or `--passes` filter excluding `eliminate_dropout`): no asm, `BufferLoc::Alias(operand)` propagation (§4.5). Default mode: removed from UIR by `EliminateDropout` UIR pass before reaching the profile. |
```

This addresses Finding 7.1 and threads the same message through the Linear and Dropout rows — a reader looking up any one row sees the M5 fusion context.

- [ ] **Step 3: New §4.9 — fused linear→relu codegen pattern**

Append new section §4.9 between the existing §4.8 (line ~308) and §5 (line ~309). Insert after §4.8's closing paragraph (just before `---` on line ~309):

```markdown
### 4.9 Fused linear → relu (with optional bias-add)

When the compiler's `FuseLinearRelu` UIR pass identifies a
`linear → relu` (or `linear[bias=true] → relu`) pattern with the
linear having a single consumer, it merges them into a single Linear
node with `fused_post_ops: vec![PostOp::Relu]`. The `emit_linear`
emitter consumes that field and produces:

```asm
    ; once at function-header time (before the matmul i-loop):
    fmov    s4, wzr             ; materialise 0.0 — needed by fmax post-op below

    ; ... (matmul i/j/k loops, accumulating sum in s0) ...
    ; ... (k-loop end) ...

    ; bias-add (if bias_offset.is_some()) — same as §4.3:
    ldr     s5, [x14, x4, lsl #2]
    fadd    s0, s0, s5

    ; M5a NEW: post-ops inline, between bias-add and store.
    ; For PostOp::Relu, the implementation emits one fmax per element:
    fmax    s0, s0, s4          ; relu — clamps negative to 0.0

    ; ... (store + j/i increments) ...
```

Order is fixed: `matmul → bias-add (if any) → post-ops → store`.
This recovers M4a's in-place relu pattern and saves one
intermediate buffer round-trip vs the unfused `Linear → Relu` chain
(§4.1 + §4.2).

The `fmov s4, wzr` materialisation happens **once** at function-header
time, conditional on `fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu))`
— not per-element. AArch64 `fmax` requires both operands in FP regs,
so `wzr` must be moved through `s4` first.

The post-op match block in `ops/linear.rs` is `#[allow(unreachable_patterns)]`-
wildcarded against future `PostOp` variants (see §5 for `LowerError::UnsupportedPostOp`).

```

This addresses Finding 7.2.

- [ ] **Step 4: §5 errors table — add `UnsupportedPostOp` row (line 311-323)**

Locate §5 errors table. Currently has two variants (`UnsupportedOp`, `ShapeNotConcrete`). Add a third row:

```markdown
| `UnsupportedOp { op, span }` | Defensive: codegen doesn't know how to lower `op`. All M4b ops are supported; this fires only if M5+ adds a new op before codegen catches up. M5c made `StdOp` `#[non_exhaustive]`, so this variant became reachable through the wildcard arm in `walk_model`. |
| `ShapeNotConcrete { span }`  | Defensive: shape wasn't fully resolved by `ir::build`. Should be unreachable.                    |
| `UnsupportedPostOp { op, span }` | M5a: post-op variant not supported by this profile. Fires when a future `PostOp` variant lands in `compiler::PostOp` before this profile knows how to emit it (e.g., `Tanh`, `Gelu`). The post-op match in `ops/linear.rs` has a wildcard arm that returns this variant; same forward-compat pattern as `UnsupportedOp`. |
```

This addresses Finding 7.3. Also annotates `UnsupportedOp` with the M5c StdOp `#[non_exhaustive]` change so the reader sees both variants are now reachable through `#[non_exhaustive]` cascades.

- [ ] **Step 5: §8 Limitations rewrite (line 383-394)**

Current §8:

```markdown
## 8. Limitations (M4b)

- **No SIMD.** Scalar throughout. NEON is M5+/M6.
- **No fusion.** `linear → relu` emits two separate loops. Fusion is M5.
- **No optimisation passes.** Three-nested-loop matmul; `mul` for indexing;
  per-element load/store; `bl _expf` per softmax element. Performance is M5+.
- **No bare-metal target.** Requires libm at link time.
- **Single-snippet error rendering for duplicate-model-name.** The
  `note: previously defined at` line is plain text, not a second `^`
  snippet. Multi-snippet (rustc-style) upgrade is M4c-or-later.
- **Integration tests run only on aarch64 hosts with `cc` available.**
  Skip with logged reason elsewhere.
```

Replace with:

```markdown
## 8. Limitations (M5b)

- **No SIMD.** Scalar throughout. NEON is M6+.
- **No matmul tiling / cache blocking.** Three-nested-loop matmul;
  `mul` for indexing; per-element load/store. Performance optimisation
  is M6+.
- **`bl _expf` per softmax element.** No batched / vectorised exp.
  M6+.
- **No bare-metal target.** Requires libm at link time. M7+ for a
  Taylor-series-`exp`-based bare-metal profile.
- **Single-snippet error rendering for duplicate-model-name.** The
  `note: previously defined at` line is plain text, not a second `^`
  snippet. Multi-snippet (rustc-style) upgrade is M4c-or-later (still
  applies).
- **Integration tests run only on aarch64 hosts with `cc` available.**
  Skip with logged reason elsewhere.
- **Only `linear → relu` and `linear[bias=true] → relu` fuse.**
  Other elementwise patterns (`linear → tanh`, `linear → gelu`, etc.)
  require new `PostOp` variants in `compiler::PostOp` and corresponding
  emit branches in `emit_linear`. M6+.
- **No graph-level dead-code elimination beyond `EliminateDropout`.**
  Other no-op shapes (e.g. `linear[out_dim=K] → linear[out_dim=N]` collapsing
  via matmul-of-matmul) are M6+.
```

Removed claims:
- `"No fusion. linear → relu emits two separate loops. Fusion is M5."` — false after M5a/M5b.
- `"No optimisation passes."` — false; two passes ship by default.

Added forward-looking limitations that are accurate as-of M5b:
- Only `Relu` post-op fuses — no other elementwise.
- No additional graph-DCE besides dropout elimination.

This addresses Finding 7.4.

- [ ] **Step 6: Verify markdown renders correctly**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && wc -l docs/profile_guide/arm64.md && grep -n "^## " docs/profile_guide/arm64.md && grep -n "^### " docs/profile_guide/arm64.md
```

Expected: section headers in order — `## 1.`, `## 2.`, `## 3.`, `## 4.` (with §4.1-4.9 sub-sections), `## 5.`, `## 5.5.`, `## 6.`, `## 7.`, `## 8.`. The new §4.9 should appear in the §### grep.

- [ ] **Step 7: Commit**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && git add docs/profile_guide/arm64.md && git commit -m "$(cat <<'EOF'
docs(m5c): bring profile_guide/arm64.md to M5b state (Findings 7.1, 7.2, 7.3, 7.4)

Holistic-review punch-list — the profile guide was M4b-era and
contradicted what default `nflc compile` does post-M5a/M5b.

Five sub-changes:

7.1 — §3 supported-ops table: extended Linear and Relu rows to
note that `linear → relu` and `linear[bias=true] → relu` fuse by
default (FuseLinearRelu UIR pass), and that standalone Relu only
appears in --no-passes / --passes <without fuse> modes. Same
update on Dropout: standalone only when EliminateDropout is filtered
out.

7.2 — new §4.9 "Fused linear → relu (with optional bias-add)":
documents the `fmov s4, wzr` once + inline `fmax s0, s0, s4`
asm shape, the matmul → bias-add → post-op → store ordering,
and the #[allow(unreachable_patterns)] wildcard for future PostOp
variants.

7.3 — §5 errors table: added `UnsupportedPostOp { op, span }`
row (M5a, M6+ Tanh/Gelu trigger). Annotated `UnsupportedOp` with
the M5c StdOp #[non_exhaustive] change so readers see both forward-
compat error paths.

7.4 — §8 Limitations rewrite: removed false claims ("No fusion",
"No optimisation passes"). Added accurate M5b limitations
(only Relu post-op, no graph-DCE beyond EliminateDropout).

Status header bumped from M4b to M5b complete.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `docs/language_reference/uir.md` updates

**Goal:** Close findings 7.5, 7.6, 7.7. The UIR reference is M3c-era; M5a added `fused_post_ops` to `NodeKind::Op` and chose functional (not in-place mutation) for passes. Both drifts mislead a reader using the doc to understand UIR.

**Files:**
- Modify: `docs/language_reference/uir.md`

INLINE. Three sub-changes.

- [ ] **Step 1: Line 17 — `profiles/generic/` → `profiles/arm64/`**

Locate line 17 of `uir.md`:

```markdown
The Universal IR is a typed computation graph that the NFL compiler produces
between parsing and codegen. It is the input to architecture profiles
(`profiles/generic/` and friends, M4+) and to optimisation passes (kernel fusion,
M5).
```

Replace with:

```markdown
The Universal IR is a typed computation graph that the NFL compiler produces
between parsing and codegen. It is the input to architecture profiles
(`profiles/arm64/` is the first concrete one, M4+) and to optimisation passes
(`compiler::passes::default_pipeline()` runs `EliminateDropout` then
`FuseLinearRelu`, M5+).
```

This addresses Finding 7.7 (`profiles/generic/` never existed) and adds accurate post-M5b context.

- [ ] **Step 2: §2 `NodeKind::Op` struct — add `fused_post_ops` field (line 56-59)**

Locate §2 Data shape, the `NodeKind` enum (around lines 56-60):

```markdown
pub enum NodeKind {
    Input { name: String },
    Op { op: StdOp, operands: Vec<NodeId>, attrs: Vec<OpAttr> },
}
```

Replace with:

```markdown
pub enum NodeKind {
    Input { name: String },
    Op {
        op: StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
        // M5a: post-ops fused into this op's emitter (currently
        // FuseLinearRelu sets this to `vec![PostOp::Relu]` on a
        // Linear it has fused with a downstream Relu; otherwise
        // empty).
        fused_post_ops: Vec<PostOp>,
    },
}
```

Also, the surrounding text at line 62-65 says:

```markdown
**Why index-based?** Easy to clone, easy to traverse (just iterate `nodes`), easy
to mutate (M5 fusion will replace nodes by id), easy to share subexpressions
(multiple nodes can reference the same `NodeId`). Standard compiler-textbook
choice.

**Why immutable in v0.1?** The builder never modifies a node after pushing it.
M5 will introduce mutation when fusion lands.
```

This is the M3c-era claim that M5 will mutate. M5 chose functional (immutable Uir → fresh Uir). Replace both paragraphs with:

```markdown
**Why index-based?** Easy to clone, easy to traverse (just iterate `nodes`),
easy to share subexpressions (multiple nodes can reference the same `NodeId`).
Standard compiler-textbook choice. UIR-passes (M5+) take an immutable `&Uir`
and return a fresh `Uir` with the transformation applied — see §7 below.

**Why immutable?** The builder never modifies a node after pushing it. UIR-passes
preserve the immutability contract: each pass returns a freshly-numbered `Uir`
(NodeIds renumbered 0..N), with operands and `model.inputs`/`model.output`
remapped through an internal id_map. Consumers can hold a `&Uir` reference
across multiple passes by re-binding through `run_pipeline`'s output. No
in-place mutation; no tombstones; no stale-NodeId hazards.
```

This addresses Finding 7.5 and partially Finding 7.6 (the §2 paragraph also wrongly claimed M5-mutation; fixed here).

- [ ] **Step 3: §7 — fix the "Mutation API" item (line 209-210)**

Locate §7 What v0.1 doesn't have, the "Mutation API" bullet:

```markdown
- **Mutation API.** `Uir` is immutable-by-construction in v0.1. M5 (kernel fusion)
  introduces mutation.
```

Replace with:

```markdown
- **Mutation API.** `Uir` is immutable-by-construction. M5+ UIR-passes
  preserve this — each pass produces a fresh `Uir` (NodeIds renumbered
  0..N, references remapped), not in-place edits. See
  `compiler::passes::run_pipeline` and the per-pass doc-comments in
  `compiler/src/passes/`.
```

This addresses Finding 7.6.

- [ ] **Step 4: Verify markdown renders correctly**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && wc -l docs/language_reference/uir.md && grep -n "^## " docs/language_reference/uir.md && grep -n "fused_post_ops\|profiles/generic\|profiles/arm64\|will introduce mutation" docs/language_reference/uir.md
```

Expected: no remaining `profiles/generic` or `will introduce mutation` matches; one `fused_post_ops` match in the new struct rendering.

- [ ] **Step 5: Commit**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && git add docs/language_reference/uir.md && git commit -m "$(cat <<'EOF'
docs(m5c): bring uir.md to post-M5a UIR shape (Findings 7.5, 7.6, 7.7)

Holistic-review punch-list — the UIR reference was M3c-era and
described an outdated UIR shape + an outdated M5-mutation plan.

Three corrections:

7.5 — §2 `NodeKind::Op` struct rendering: added the
`fused_post_ops: Vec<PostOp>` field (introduced M5a) with a
short comment explaining how FuseLinearRelu populates it.
The surrounding paragraph "Why index-based" / "Why immutable in
v0.1" was M3c-era; rewrote to reflect that M5+ passes are
functional (fresh Uir per pass), not in-place mutators.

7.6 — §7 "Mutation API" item: replaced "M5 (kernel fusion)
introduces mutation" with accurate description of the functional
pass model. Cites compiler::passes::run_pipeline as the entry
point.

7.7 — §1 introduction: replaced reference to non-existent
`profiles/generic/` with `profiles/arm64/` (the first concrete
profile) and added pipeline-default-passes context.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Closeout — `CLAUDE.md` + `DEVLOG.md` (M5 fully closed)

**Goal:** Final M5c task. Close findings 8.1 (CLAUDE.md viewer caveat) and 8.2 (`profiles/generic/` reference). Update CLAUDE.md "Current Status" to reflect M5 closed (5a + 5b + 5c). Append M5c entry to DEVLOG. After this commit, the entire M5 cycle is documented and merged.

**Files:**
- Modify: `CLAUDE.md` — Principle 5 + "What NOT to Do" line (Finding 8.1), "Adding a new architecture profile" recipe (Finding 8.2), "Current Status" section (M5c-aware update).
- Modify: `DEVLOG.md` — append M5c entry above M5b.

INLINE. Pure markdown closeout commit.

- [ ] **Step 1: Final pre-closeout verification**

Before any docs changes, verify the codebase state is clean:

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && cargo fmt --all -- --check 2>&1 | tail -3 && cargo build --workspace 2>&1 | tail -3 && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: `TOTAL: 189`. All checks green. Tasks 1-4 should not have introduced any regression.

- [ ] **Step 2: Update CLAUDE.md Principle 5 (line 98-99)**

Locate Design Principle 5 in `CLAUDE.md`:

```markdown
5. **Human oversight.** The viewer layer always exists. Any compiler output must be inspectable by
   a human using the viewer tool.
```

Replace with:

```markdown
5. **Human oversight.** Every compiler output must be inspectable by a human. Until
   the dedicated viewer tool ships (M7+), the `nflc parse <file.nfl> --uir` CLI
   provides human-readable UIR pretty-printing via `Display for Uir`, including
   M5a's `fused=[<list>]` suffix for fused operations. New UIR fields and node
   kinds must extend the `Display` impls so this CLI rendering stays complete.
```

This closes Finding 8.1 by replacing "the viewer layer always exists" (false — `viewer/` is `.gitkeep`-only) with "every output must be inspectable", an actually-followable principle that names the current rendering tool (`nflc parse --uir`).

- [ ] **Step 3: Update CLAUDE.md "What NOT to Do" line 208**

Locate the "What NOT to Do" section, line 208:

```markdown
- Do not skip viewer support — every new IR node must have a viewer rendering
```

Replace with:

```markdown
- Do not skip human-readable rendering — every new IR node, field, or NodeKind
  variant must extend the `Display` impls in `compiler/src/ir/types.rs` so the
  `nflc parse --uir` CLI continues to render the full UIR shape. The dedicated
  viewer tool (M7+) will consume the same `Display` output as a starting point.
```

This rephrases the rule in terms of the actual tooling that exists.

- [ ] **Step 4: Update CLAUDE.md "Adding a new architecture profile" recipe (line 149)**

Locate line 149:

```markdown
2. Implement the profile interface (see `profiles/generic/` as reference)
```

Replace with:

```markdown
2. Implement the profile interface (see `profiles/arm64/` as the canonical reference: `pub fn lower(&Uir) -> Result<Asm, LowerError>` plus the `Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError` types)
```

This closes Finding 8.2.

- [ ] **Step 5: Update CLAUDE.md "Current Status" (lines 156-198)**

Replace the entire Current Status block (lines 156 to ~198) with M5-closed content:

```markdown
## Current Status

**Milestone 5 fully complete (5a + 5b + 5c).** UIR-pass infrastructure ships
two passes: `EliminateDropout` (removes dropout nodes from the graph at
inference time) and `FuseLinearRelu` (bias-aware — fuses `linear → relu`
and `linear[bias=true] → relu`). `default_pipeline()` runs them in canonical
order `[EliminateDropout, FuseLinearRelu]` so that `linear → dropout → relu`
patterns collapse and fuse end-to-end.

CLI: `nflc compile` runs the default pipeline between `ir::build` and
profile lowering. `--no-passes` skips the pipeline; `--passes <list>` runs a
filtered subset (canonical order enforced regardless of user-typed order,
with a stderr `note:` when they diverge). Mutually exclusive flags. All flag
validation uses the dynamic `default_pipeline()` registry, so M6+ pass
additions surface in error messages automatically.

Profile (`profiles/arm64`) is unchanged from M4b in source, but consumes the
fused UIR by default. `emit_linear` stacks `matmul → bias-add → fmax (post-op)
→ store` in one block; `BufferLoc::Alias(operand)` stays as the fallback for
Dropout in `--no-passes` and exclude-eliminate_dropout filter modes.

Op coverage: linear (± bias), relu, dropout, softmax (NFL v0.1 inference-only).
Two FFI integration tests pin bit-exact equivalence between fused and unfused
asm — `fused_vs_unfused_classifier_match_numerically` (M5a) and
`fused_vs_unfused_mixed_args_match_numerically` (M5b).

Cross-cutting consistency: all five workspace error types (`BuildError`,
`ParseError`, `LexError`, `PassError`, `LowerError`) implement
`std::error::Error`. `StdOp` and `PostOp` are both `#[non_exhaustive]`. The
profile-side `match op` block has a wildcard arm routing future ops to
`LowerError::UnsupportedOp`.

3-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib).
Production code std-only; `libloading` is a test-only dev-dep. **189 tests
passing** across lexer, parser, IR, passes (11 fusion + 8 dropout +
5 pipeline-level), profile codegen, CLI smoke (8), reference-validation,
and FFI integration. `cargo build --workspace`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo fmt --all -- --check` all clean.
CI green.

Documentation: `docs/profile_guide/arm64.md` covers the M5b-current asm
patterns including fused linear→relu (§4.9) and the `UnsupportedPostOp` /
`UnsupportedOp` error variants. `docs/language_reference/uir.md` reflects
the `fused_post_ops` field on `NodeKind::Op` and the functional-pass model.
`PROJECT_SPEC.md` milestones table marks M5 complete.

The immediate next step is **Milestone 6 — open scope**. Candidate
directions documented in DEVLOG (M5b/M5c carried-forward tech debt):
- **Test-helper extraction** (`compiler/src/ir/test_utils.rs` for the
  hand-built UIR pattern that hit "three strikes" in M5b).
- **Bare-metal target** (Taylor-series `expf` for softmax, no libm
  dependency) as a second arm64-flavoured profile.
- **Attention-pattern fusion** (`linear → softmax_max`, `linear → bias →
  softmax`) — requires a third PostOp variant and possibly a softmax-aware
  fusion pass, which would naturally trigger the M5b-deferred shared
  victim/remap helper extraction.
- **x86_64 profile** (AVX-512 / VNNI for matmul).
- **`BuildError::span()` accessor + shared `Diagnostic` trait** if a
  fourth error type appears or generic CLI rendering arrives (M5c findings
  1.2, 2.1).

M6 brainstorming runs in a fresh worktree once M5c is merged.
```

This is the largest single edit in M5c. It replaces the M5b-era Current Status with M5-closed-era content, accurately summarising every M5 deliverable plus the M5c consistency improvements, and pointing to M6 candidate directions.

- [ ] **Step 6: Append M5c entry to DEVLOG.md**

In `DEVLOG.md`, find the most recent entry (M5b, dated 2026-05-05). Insert ABOVE it (separated by `---`):

```markdown
---

## 2026-05-05 — Milestone 5c closed: M5 cycle close-out (docs sync + small consistency fixes)

### What was done
- Applied 13 of 17 findings from the M5b post-merge holistic review
  (Option B scope from the brainstorming session). 4 findings explicitly
  deferred to M6+ (1.2 shared `Diagnostic` trait, 2.1 `BuildError::span()`
  accessor, 4.1 test-helper extraction, 6.1 pass struct visibility,
  DEVLOG-1 `debug_assert_eq!` → `assert_eq!`).
- Code consistency (4 small Rust changes, ~6 lines total):
  - `impl std::error::Error for PassError` (`compiler/src/passes/mod.rs`).
  - `impl std::error::Error for LowerError` (`profiles/arm64/src/types.rs`).
  - All five workspace error types now implement `std::error::Error`
    uniformly (`BuildError`, `ParseError`, `LexError`, `PassError`,
    `LowerError`).
  - `nflc/src/main.rs:253` — `&e.message` → `&e.to_string()` for
    `ParseError` rendering call-site consistency.
  - `#[non_exhaustive]` on `compiler::ir::stdlib::StdOp`. Cascade:
    `profiles/arm64/src/codegen.rs::walk_model` `match op { ... }`
    block needed a wildcard arm — added one routing future ops to
    `LowerError::UnsupportedOp` (which lost its `#[allow(dead_code)]`
    attribute, since the wildcard makes the variant reachable). Same
    forward-compat pattern as M5a's `PostOp` `#[non_exhaustive]` +
    `emit_linear` post-op-match wildcard.
- `PROJECT_SPEC.md`:
  - Milestones table M5 row updated to "5a + 5b + 5c complete" with
    accurate description of UIR-pass framework + two passes + CLI
    flags + bit-exact integration tests.
  - Open Questions section: retired two answered questions (NFL v0.1
    grammar frozen at M1; static stack memory model decided at M4b).
    Moved to a new "Decisions (formerly open, now resolved)"
    sub-section preserving the historical record.
- `docs/profile_guide/arm64.md` brought from M4b-era to M5b-current:
  - Status header updated to M5b complete.
  - §3 supported-ops table: Linear/Relu/Dropout rows extended to
    document their default-fused vs `--no-passes` behavior.
  - New §4.9 "Fused linear → relu (with optional bias-add)"
    documenting the `fmov s4, wzr` once + inline `fmax s0, s0, s4`
    asm shape, the `matmul → bias-add → post-op → store` ordering,
    and the wildcard for future `PostOp` variants.
  - §5 errors table: added `UnsupportedPostOp` row (M5a) + annotated
    `UnsupportedOp` with the M5c `StdOp` `#[non_exhaustive]` change.
  - §8 Limitations rewrite: removed false claims ("No fusion", "No
    optimisation passes"); added accurate M5b limitations (only
    `Relu` post-op fuses; no graph-DCE beyond `EliminateDropout`).
- `docs/language_reference/uir.md` brought from M3c-era to M5a-current:
  - §1: `profiles/generic/` (never existed) replaced with `profiles/arm64/`
    + post-M5b pipeline-default-passes context.
  - §2 `NodeKind::Op` struct rendering: added the `fused_post_ops:
    Vec<PostOp>` field with comment.
  - §2 immutability rationale rewritten to describe the functional
    pass model (M5+ passes return fresh `Uir`, not in-place edits).
  - §7 "Mutation API" item: replaced "M5 introduces mutation" with
    accurate description of the functional pass model.
- `CLAUDE.md`:
  - Design Principle 5 ("Human oversight"): replaced false "viewer
    always exists" with accurate "every output must be inspectable;
    `nflc parse --uir` is the current renderer until the M7+ viewer
    tool ships". The `viewer/` directory is currently a `.gitkeep`
    placeholder.
  - "What NOT to Do" line about viewer: rephrased to cite the
    `Display` impls in `compiler/src/ir/types.rs` as the actual
    rendering surface to keep extending.
  - "Adding a new architecture profile" recipe: replaced the
    `profiles/generic/` reference (deleted before M4a shipped) with
    `profiles/arm64/` + the actual public surface to replicate.
  - "Current Status" section: rewritten to reflect M5 fully closed
    (5a + 5b + 5c), the consistency improvements from M5c, and the
    open M6 candidate directions documented in this DEVLOG entry.

### Decisions made
None new. M5c is purely drift-fix execution against the holistic-review
punch-list. No architectural calls were made — the punch-list IS the
spec, and Option B (drift-fix only, no test-helper extraction yet) was
chosen with the user before plan-writing.

### Holistic review process — worth recording for M6+
The M5b post-merge holistic review (single thorough subagent dispatch,
spec/structure/cross-cutting/docs/PR-body scan) found 17 findings vs.
the per-task reviews' typical 1-3 findings each. Of the 17:
- 13 were close-in-M5C (this milestone).
- 4 are deferred M6+ items.
- Almost half the findings were docs drift (4 in `arm64.md`, 3 in
  `uir.md`, 2 in `CLAUDE.md`, 3 in `PROJECT_SPEC.md`) — the kind of
  drift per-task reviews systematically don't catch because each task
  reviews "did the code match the plan", not "did the docs catch up".

Decision for M6+ workflow: schedule a holistic review at every
milestone close-out, not just at v1 stability. Cost: one subagent
dispatch (~5 min). Benefit: catches docs drift early, while context
is fresh.

### Problems encountered
- One holistic-review finding (3.4: claimed `PROJECT_SPEC.md §4`
  Compiler Pipeline diagram says "M5 introduces mutation") was a false
  positive — that text doesn't exist in `PROJECT_SPEC.md`. The actual
  mutation drift is in `docs/language_reference/uir.md` (closed by
  Findings 7.6, 7.7 in this milestone). Reviewer probably conflated
  the two files. Test count and other code findings were all
  verifiable.

### Known tech debt (carried forward to M6+)
1. **Test-helper extraction** (`compiler/src/ir/test_utils.rs`):
   `op_node` / `input_node` private helpers. The "three strikes" rule
   fired with the third hand-built UIR test in M5b's
   `pipeline_eliminates_dropout_before_fusing_linear_relu`. Holistic
   review confirmed the threshold is met. Deferred to M6+ as the
   first task because M6+ may surface a fourth use case that informs
   the helper API shape (e.g., attention-pattern tests).
2. **`BuildError::span()` accessor** to match `PassError`/`LowerError`'s
   `span()` API. Non-breaking addition (`line`/`col` flat fields stay).
3. **Shared `Diagnostic` trait** for the five error types. Defer until
   either a fourth error type appears or the CLI acquires a generic
   error-rendering path that currently duplicates per-type dispatch.
4. **Pass struct visibility** (`EliminateDropout`, `FuseLinearRelu` →
   `pub(crate)`?). Leave `pub` until v1 stability commitment forces a
   decision.
5. **`debug_assert_eq!` → `assert_eq!`** for the FnSig `params_floats`
   agreement check in both `fused_vs_unfused_*_match_numerically`
   integration tests. Pre-existing pattern; pre-M5b. Harden when next
   integration test is added (M6+).
6. **Holistic-review false-positive auditing process** — find a way
   to spot-check reviewer claims against actual file content before
   integrating findings. Mitigates the rare 3.4-style conflation.

### Next step
**Milestone 5 fully complete.** Brainstorm M6 in a fresh worktree once
M5c merges. Open scope; candidate directions (in priority order based
on user-feedback signal):
1. Test-helper extraction (~30 lines, M6 task 1) — closes the longest-
   standing M5-era tech debt and creates a shared primitive M6+ tests
   can build on.
2. Attention-pattern fusion (`linear → softmax_max`, `linear → bias →
   softmax`) — requires a third `PostOp` variant and possibly a
   softmax-aware fusion pass.
3. Bare-metal target (Taylor-series `expf` for softmax, no libm).
4. x86_64 profile (AVX-512 / VNNI for matmul).
5. `BuildError::span()` + shared `Diagnostic` trait if a fourth error
   type appears.
```

This is a long DEVLOG entry, but M5c closes the entire M5 cycle and the entry needs to do double duty: document M5c itself AND wrap up the M5 cycle for the M6 brainstorm.

- [ ] **Step 7: Final verification (post-edits)**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && cargo fmt --all -- --check 2>&1 | tail -3 && cargo build --workspace 2>&1 | tail -3 && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3 && cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: `TOTAL: 189`. All checks green. Docs edits don't trigger any code-side regression.

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && grep -rn "no_fuse\|--no-fuse\|profiles/generic" nflc/src/ profiles/ compiler/src/ docs/ CLAUDE.md PROJECT_SPEC.md 2>/dev/null
```

Expected: no matches. Both legacy-flag and never-existed-profile-dir references cleaned up across code and docs.

- [ ] **Step 8: Commit closeout**

```bash
cd /Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m5c-m5-closeout && git add CLAUDE.md DEVLOG.md && git commit -m "$(cat <<'EOF'
chore(m5c): close Milestone 5 — full cycle complete (5a + 5b + 5c)

Per holistic-review punch-list (Findings 8.1, 8.2) plus M5 close-out:

CLAUDE.md:
- Design Principle 5: replaced false "viewer always exists" with
  accurate "every output must be inspectable; nflc parse --uir is
  the current renderer until M7+ viewer ships". `viewer/` directory
  is currently `.gitkeep` placeholder; principle as stated since M2
  was unfollowable.
- "What NOT to Do" viewer line: rephrased to cite Display impls in
  compiler/src/ir/types.rs as the rendering surface to keep extending.
- "Adding a new architecture profile" recipe: profiles/generic/
  (deleted before M4a) → profiles/arm64/ + actual public surface.
- "Current Status" section: rewritten to reflect M5 fully closed
  (5a + 5b + 5c), M5c consistency improvements, and M6+ candidate
  directions.

DEVLOG.md:
- New M5c entry above M5b. Documents the 13 findings closed (with
  cite to the punch-list), the 6 carried-forward to M6+ items, the
  one false-positive review finding (3.4), and the workflow note
  that holistic-review-per-milestone catches drift per-task reviews
  systematically miss.
- M6 candidate directions ordered by priority signal: test-helper
  extraction (#1, the longest-standing M5-era debt), attention-pattern
  fusion, bare-metal target, x86_64 profile.

189 tests pass (no count change in M5c — consistency fixes only).
clippy/fmt clean. grep confirms zero `no_fuse`, `--no-fuse`, or
`profiles/generic` substring across code + docs.

M5 fully closed. M6 brainstorm runs in a fresh worktree
post-M5c-merge.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done. What's next?

After Task 5, M5c is complete:

1. ✅ All 13 in-scope holistic-review findings closed.
2. ✅ 4 deferred findings explicitly carried-forward to M6+ in DEVLOG.
3. ✅ All five workspace error types implement `std::error::Error`.
4. ✅ `StdOp` is `#[non_exhaustive]` with downstream wildcard arm.
5. ✅ `PROJECT_SPEC.md` milestones table marks M5 complete.
6. ✅ `docs/profile_guide/arm64.md` reflects M5b state including fused linear→relu pattern.
7. ✅ `docs/language_reference/uir.md` reflects M5a's `fused_post_ops` and the functional-pass model.
8. ✅ `CLAUDE.md` Principles + recipes accurate; "Current Status" reflects M5 closed.
9. ✅ `cargo build/clippy/fmt --check` clean.
10. ✅ 189 tests pass (no count change).

**After all tasks pass:** push `claude/m5c-m5-closeout` and open a PR against `main`. Title: `Implement Milestone 5c: M5 close-out documentation + small consistency fixes`. After merge, M5 is fully closed; M6 begins with a fresh `superpowers:brainstorming` cycle once main is updated.
