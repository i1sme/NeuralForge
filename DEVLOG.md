# NeuralForge — Development Log

This file is the living record of the project. Every session gets an entry.
Entries are in reverse-chronological order (newest at the top).

Format for each entry:
```
## YYYY-MM-DD — <one-line summary>
### What was done
### Decisions made
### Problems encountered
### Next step
```

---

## 2026-05-06 — Public-release hygiene: drop stale Gmail contact from DEVLOG

### What was done
- Removed one sentence from the earlier 2026-05-06 "License adoption:
  AGPL-3.0-only + CLA" entry that recorded `me.its1984@gmail.com` as the
  commercial-licensing contact. The contact context evaporated when the same
  day's later entry pivoted the project to pure Apache-2.0 (no dual-license,
  no commercial-by-request channel), so the email was a stale leftover that
  would only confuse a public reader.
- Verified via `grep -rIn "me.its1984\|gmail"` across the worktree: no
  remaining matches in tracked files.

### Decisions made
- Deleted the sentence outright rather than redacting (`[redacted]`). The
  surrounding paragraph still describes the historical AGPL-era README/CLA
  shape; removing one line keeps the historical record coherent without
  leaving a placeholder that begs explanation. **Why:** the only purpose of
  the email was to be a contact channel, and that channel no longer exists.
- Did not touch the rest of the AGPL adoption entry — it remains an accurate
  log of what shipped that day, even though the next same-day entry undid
  most of it. The DEVLOG is a chronological record, not a current-state doc.
- Did not redact the `me.its18@yandex.ru` author/committer email from git
  history. That email is already published in every existing commit's
  metadata, and rewriting history to scrub it would invalidate every
  outstanding clone, branch, and PR reference. Accepting it as part of the
  project's public identity going forward.

### Problems encountered
- None.

### Next step
- Public-release blockers closed (Apache-2.0 LICENSE merged, README present,
  no Gmail in tracked files, no Gmail in git history). Repo can be flipped to
  public when the user is ready. Strategic roadmap selection (per
  `PROJECT_SPEC.md` §"Strategic Roadmap") is unrelated to this and remains
  the next *engineering* step.

---

## 2026-05-06 — License pivot: AGPL-3.0-only → Apache-2.0

### What was done
- **`LICENSE`** — AGPL-3.0 text overwritten with canonical
  Apache-2.0 text from `apache.org/licenses/LICENSE-2.0.txt`
  (661 → 202 lines).
- **`CONTRIBUTING.md`** — simplified: the four-clause CLA from
  the AGPL adoption is dropped. Apache-2.0 §5 ("Submission of
  Contributions") implicitly licenses contributions under the
  same terms; no separate CLA is required at this stage. The
  commercial-relicensing grant (clause 2 in the previous CLA)
  is removed entirely — there is no dual-licensing model under
  pure Apache-2.0. Development-workflow pointer to `CLAUDE.md`
  retained.
- **`README.md`** `## License` section rewritten — dual-license
  description, "What AGPL covers, and what it does not"
  paragraph (GCC/LLVM compiler-output precedent), and
  commercial-use contact section all removed. Replaced with a
  single-license description for Apache-2.0 that explains the §3
  patent-grant rationale (the reason for choosing Apache-2.0
  over MIT specifically). `## Contributing` section updated to
  match (no CLA referenced).
- **`Cargo.toml`** (workspace root) — `[workspace.package].license`
  changed from `"AGPL-3.0-only"` to `"Apache-2.0"`. Per-crate
  inheritance via `license.workspace = true` unchanged (still
  three crates, still inheriting one source of truth).
- **SPDX one-liner sweep** — `// SPDX-License-Identifier:
  AGPL-3.0-only` replaced with `// SPDX-License-Identifier:
  Apache-2.0` across all 39 `.rs` files in the workspace
  (`sed -i ''` via `find -exec`). 39/39 verified post-sweep.
- Workspace gates verified clean post-change: `cargo fmt --all
  -- --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace` (223/223 passing — no
  behaviour regressions).

### Decisions made

**Pivot rationale: AGPL+commercial was premature optimization.**
For a 0.1-pre-alpha project with zero public users, AGPL+commercial
dual-licensing pays a real, immediate cost (corporate-blacklist
adoption friction — Google internally bans AGPL, many corporate
`LICENSE_POLICY` files explicitly forbid it; contributor CLA
friction; "premature monetization" pattern-matching by readers,
who associate AGPL+commercial with mature MongoDB-class projects
defending accumulated value, not pre-alpha projects with no users
yet) to defend a commercial revenue stream that does not exist.
The barrier activates legal-review **before** anyone has tried
the technology, so it can pre-emptively close the very adoption
channel that would generate eventual commercial demand.

The standard path for early-stage infrastructure projects is the
opposite ordering: permissive licensing first, build adoption,
re-evaluate licensing later if real commercial interest emerges
and there is an actual community for the rationale to address.
Redis (MIT → BSL), MongoDB (AGPL → SSPL), and Elasticsearch
(Apache → SSPL) all monetised already-accumulated open-source
impact; none defended speculative future revenue.

AGPL §13 (the SaaS clause, the licence's headline feature) almost
never applies to a compiler — compilers are local tools, not
network services — so the practical copyleft surface AGPL added
was narrow to begin with. The price/value ratio at this stage
was poor.

**Apache-2.0 over MIT.** Apache §3 grants an explicit patent
license from contributors. MIT is silent on patents. For an
infrastructure compiler where codegen algorithms may carry patent
claims, the explicit grant forecloses the scenario where a
contributor (or their employer) later asserts patents over
contributions they made. Standard choice for compiler/runtime
projects (LLVM, Apache TVM, Bazel, etc.).

**No CLA at this stage.** Apache-2.0 §5 ("Submission of
Contributions") states that any contribution intentionally
submitted for inclusion is licensed under the Apache-2.0 terms
unless the contributor states otherwise. For a project not
pursuing dual-licensing, this implicit grant is sufficient. A
formal CLA can be added later if the project's needs change.

### Problems encountered

PR #18 (the AGPL adoption) had already been merged to `main`
before this pivot conversation occurred. The pivot is therefore
implemented as a **forward license-change PR** rather than a
close-and-replace of an unmerged PR — this PR overwrites the
AGPL artefacts on `main` with Apache-2.0 equivalents (LICENSE
whole-file overwrite; SPDX headers sed-replaced across all 39
.rs files; Cargo.toml `workspace.package.license` string change;
README §License rewritten; CONTRIBUTING.md simplified).

Branch `claude/agpl-license` and PR #18 remain on the remote as
a historical record of the AGPL approach and the reasoning that
led to it, plus this DEVLOG entry preserves the pivot rationale
for future contributors who might wonder "why did we change
license once already".

Minor tooling friction during execution: macOS `xargs sed -i ''`
and `find -exec sed -i '' '...' {} \;` chained with
`&& grep -c ...` exited 1 silently because `grep -c` returns
non-zero when there are zero matches (which is the *desired*
post-sweep state: no AGPL headers remain). Verification was
restructured to use `awk` counters instead of `grep -c` chained
with `&&`.

### Next step

After merge of this license-pivot PR, M9 brainstorming begins in
a fresh worktree. Original axis-selection question stands: codegen
breadth (x86_64 profile) vs modelling depth (NFL v0.2 grammar →
attention) vs deployment reach (bare-metal `expf`).

---

## 2026-05-06 — README "Project status" refresh: M5 → M8 catch-up before public surface

### What was done
- **`README.md` `## Project status`** — rewritten end-to-end. Was
  pinned at M5 closure (M5a + 5b + 5c) with `linear → relu` as the
  fusion frontier; now reflects M8 closure: three passes shipped
  (`EliminateDropout`, `FuseLinearRelu`, `FuseLinearSoftmax`),
  large-dim immediate hoisting through a single emit helper, and
  viewer v0.1. Test-count claim moved 189 → 223 (verified via
  `cargo test --workspace` summed across suites).
- **`README.md` next-milestone pointer** — replaced the
  "Next: Milestone 6 — attention-pattern fusion" line (and its
  link to `docs/superpowers/specs/2026-05-05-m6-attention-fusion-design.md`)
  with a cross-link to `PROJECT_SPEC.md#strategic-roadmap`. The
  README now describes M9 as a scope-selection step over three open
  axes (codegen breadth / modelling depth / deployment reach) rather
  than naming a specific milestone.
- **`README.md` CLI bullet + Build & try block** — `--uir-verbose`
  added to both the bullet under Project status and the parse
  example block under Build & try.
- **`README.md` Core principles `Human oversight` bullet** — was
  "with a dedicated viewer tool planned for M7+"; now reads "viewer
  v0.1 ships today via `nflc parse --uir` (compact) and
  `nflc parse --uir-verbose` (annotated), with a dedicated
  standalone viewer tool reserved for future profile-level
  annotation work".
- **`README.md` Repository map `viewer/` row** — same drift fix:
  dropped the now-stale `(M7+)` parenthetical, renamed to "future
  standalone viewer tool", and added the `--uir-verbose` rendering
  alongside `--uir`.
- **Workspace gates** — re-ran `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test --workspace`. All green; 223 tests pass. No-op as
  expected since no code paths changed.
- **Rebase onto post-PR #18 main** — PR #18 (AGPL-3.0 + CLA) merged
  while this PR was open. Rebased rather than merge-committed:
  README auto-merged cleanly (License/Contributing block from PR
  #18 sits at the end, my four section edits sit in the middle, no
  textual overlap); DEVLOG required a hand resolve since both PRs
  added a top-of-file entry — chose `--ours` (post-PR-#18 baseline)
  and prepended this entry above the License-adoption entry per the
  reverse-chronological convention.

### Decisions made

**README points to `PROJECT_SPEC.md` §Strategic Roadmap instead of
naming the next milestone directly.**
The previous README named M6 explicitly and linked to a single
design spec. After M6 → M7 → M8 all shipped, that wording was
exactly the failure mode we just fixed. M9's defining property is
that *which* milestone it becomes is itself a decision (which axis
seeds the brainstorm) — the README cannot pre-name it without
re-introducing the same staleness. Linking to the spec section
keeps the README evergreen across milestone transitions and
concentrates roadmap churn in one place.

**Repository-map `viewer/` row updated alongside the four
explicitly-listed README sections.**
The user enumerated four sections to refresh and asked for tight
scope. Line 54 was not enumerated, but its `(M7+)` reference is the
same class of staleness as the Core-principles bullet (milestone
numbering frozen at write time) and its viewer phrasing should
match the new "future standalone viewer tool" framing decided
under task 4. Updating it preserves internal consistency without
touching unrelated sections; leaving it would have shipped a
README with two different stories about the viewer in two adjacent
sections.

**Rebase strategy: take main's DEVLOG (`--ours`) and prepend my
entry, rather than merging two top-of-file entries by hand.**
Both PR #18 and this PR added a new entry to the same insertion
point at the top of DEVLOG, so git produced overlapping conflict
blocks. Reading from the post-PR-#18 file gives a clean baseline
where PR #18's License entry, separators, and downstream entries
are already structurally correct; prepending this entry above it
re-establishes the reverse-chronological order without needing to
hand-reassemble separator markers around two simultaneously-edited
blocks. Lower error surface than line-by-line marker resolution.

### Problems encountered
- Merge conflict on rebase against post-PR #18 main, as expected.
  Resolved as described in "What was done" / "Decisions made"; no
  semantic change to either entry's content.

### Next step
Strategic-roadmap selection for M9 remains the next substantive
decision (see `PROJECT_SPEC.md` §"Strategic Roadmap"). This PR
also closes the explicit follow-up flagged in PR #18's
"Out-of-scope follow-up" note (README "Project status" staleness
called out before going public), so the public-surface readiness
checklist is unblocked.

---

## 2026-05-06 — License adoption: AGPL-3.0-only + CLA, open-source release prep

### What was done
- **`LICENSE`** — canonical GNU Affero General Public License v3.0
  text (661 lines) fetched from `gnu.org/licenses/agpl-3.0.txt`.
- **`CONTRIBUTING.md`** (new) — four-clause Contributor License
  Agreement covering: (1) AGPL-3.0 license grant on contributions,
  (2) commercial-relicensing grant to the project owner, (3)
  future-patches clause explicitly covering follow-up commits,
  force-pushes, amendments, rebases, and review-fixups within the
  same PR, branch, or revision history, (4) original-work
  attestation. Plus brief development-workflow pointer to
  `CLAUDE.md`.
- **`README.md`** — new `## License` and `## Contributing`
  sections explaining: dual-license model (AGPL-3.0 + commercial-
  by-request), explicit GCC/LLVM precedent that AGPL does **not**
  extend to compiler output, attribution-as-courtesy etiquette
  request (link back to repo), copyright assertion `Copyright (C)
  2026 Arsenii Voloshyn`.
- **`Cargo.toml`** — replaced `license = "MIT OR Apache-2.0"`
  placeholder in 3 crate manifests (`compiler`, `nflc`,
  `profiles-arm64`) with `license.workspace = true`. Added a new
  `[workspace.package]` block to root `Cargo.toml` with
  `license = "AGPL-3.0-only"` as the single source of truth.
- **SPDX one-liner sweep across all 39 `.rs` files** —
  `// SPDX-License-Identifier: AGPL-3.0-only` prepended (with a
  blank-line separator) to every Rust source. Idempotent —
  files already containing an SPDX header are skipped.
- Workspace gates verified clean post-change: `cargo fmt --all
  -- --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo test --workspace` (223/223 passing — no
  regressions from M8 baseline).

### Decisions made

**License choice: `AGPL-3.0-only` (not `-or-later`).** The
"-only" suffix pins the license to v3 specifically. Rationale:
the dual-licensing strategy depends on the project owner
controlling relicensing decisions; "-or-later" would let the
FSF effectively modify the project's license terms by publishing
a future GPL version, which dilutes that control. FSF
traditionally recommends "-or-later" for community goodwill;
for a commercial dual-license, "-only" is the safer pin.

**CLA via `CONTRIBUTING.md`, not via DCO or external CLA-bot.**
A single-paragraph CLA in a markdown doc is sufficient for an
early-stage one-person OSS project. Upgrading to DCO (sign-off
line on each commit) or a CLA-bot becomes worth the friction
only when the project actually starts attracting external
contributions. Clause 3 ("future patches") is the crucial piece:
without it, a contributor could open a PR, agree to the CLA,
then push fixup commits during review that are technically not
bound by the agreement. With clause 3, every commit on the same
branch/PR is bound by the same grant.

**SPDX one-liner per file (not full GPL-boilerplate header).**
`SPDX-License-Identifier` is the modern Rust/kernel convention —
machine-parseable, single line of noise per file, sufficient as
a per-file license declaration. The full 15-line GPL boilerplate
header is required only when no LICENSE file is present, which
is not the case here.

**No §7 binding attribution clause; attribution is README
etiquette only.** AGPL §4 already mandates copyright preservation;
adding a §7 additional term ("Required attribution: visible
link to upstream") would create an irremovable obligation that
some downstream users would refuse, with no proportional benefit.
The README's "please link back" courtesy is honoured by most
downstream users in practice without legal coercion.

**Compiler output is explicitly NOT covered by AGPL — documented
in README.** Per GNU FAQ ("the output of a program is not, in
general, covered by the copyright on the code of the program")
and the GCC/LLVM precedent. Documenting this explicitly in the
README sets honest expectations: the AGPL barrier applies to
those who fork, embed, or service-host the compiler, not to
those who use vanilla NeuralForge to compile their own
proprietary networks. The dual-licensing model targets the
former group; the latter group is a free user.

**`[workspace.package]` inheritance over per-crate license
duplication.** Root `Cargo.toml` is the single source of truth;
each crate's `license.workspace = true` inherits. New crates
added to the workspace inherit automatically. Slightly more
upfront mechanism, lower long-term maintenance cost.

### Problems encountered
- None. SPDX sweep ran cleanly; all 39 files already lacked an
  SPDX header (no idempotency conflicts). `cargo fmt --check`
  accepted the SPDX comment format unchanged (single-line `//`
  comment + blank-line separator + existing content).

### Next step
After merge of this licensing PR, M9 brainstorming begins in a
fresh worktree. Original axis-selection question stands: codegen
breadth (x86_64 profile) vs modelling depth (NFL v0.2 grammar →
attention) vs deployment reach (bare-metal `expf`).

Out-of-scope follow-up surfaced during this session: `README.md`
"Project status" section is stale — references "M5 fully closed"
and "Next: Milestone 6", but the actual state is M8 closed.
Worth a small refresh PR before the repo goes public, but not
part of this licensing PR — different concern, different scope.

---

## 2026-05-06 — M9 framing: Strategic Roadmap added; carry-forward list split into axes vs trigger-driven OQs

### What was done
- **`PROJECT_SPEC.md`** — new `## Strategic Roadmap` section
  inserted between First Milestones and Open Questions. Three
  open strategic axes presented as a dependency graph:
  `x86_64 profile → MACHO_SYM_PREFIX rename`;
  `NFL v0.2 grammar → attention ops → profile-level viewer annotations`;
  `bare-metal expf → drop libm dependency`. Each axis annotated with
  one paragraph (codegen breadth / modelling depth / deployment reach).
- **`PROJECT_SPEC.md`** — `## Open Questions` restructured into
  two subsections: `### Design questions` (existing 3 bullets
  unchanged: training syntax, quantisation, distribution format)
  and `### Trigger-driven cleanup` (5 OQs migrated from CLAUDE.md
  carry-forward list — OQ-NEW, OQ-7, OQ-8, OQ-9, M5c OQ-4 — each
  with its trigger condition explicit).
- **`CLAUDE.md`** `## Current Status` — trimmed from ~80-line M8
  summary + 9-item carry-forward list down to one factual state
  line (`Milestone 8 complete. 223 tests passing.`) + workspace-gate
  one-liner + pointer to `PROJECT_SPEC.md` §Strategic Roadmap and
  §Open Questions / Trigger-driven cleanup.

### Decisions made

**Roadmap = dependency graph, not a task plan with deadlines.**
The artefact is literally three rows of "what unlocks what". No
sequencing across axes, no estimates, no scope checklists.
Rationale: without this framing the project risks burning M9, M10,
M11 on trigger-driven cleanup and emerging three iterations later
with the same fundamental capabilities — that is maintenance, not
strategic progress. Choosing the next milestone means choosing
which axis to advance, which is a deliberate decision rather than
a "what's interesting today" pick.

**Trigger-driven OQs (OQ-NEW, OQ-7, OQ-8, OQ-9, M5c OQ-4) stay
out of the roadmap, in `## Open Questions / ### Trigger-driven
cleanup`.** They activate on their own trigger condition (next
predicate change, first real `Err`-case, fourth-pass non-PostOp
mutation, etc.) and explicitly should not be planned in advance —
that would defeat the trigger mechanism. Putting them in the
strategic roadmap would conflate "we choose to do this" with
"this fires when X happens".

**Roadmap lives in `PROJECT_SPEC.md`, not a separate
`ROADMAP.md`.** The spec is already the single source of truth
for what the project is and where it's heading; splitting the
roadmap into a second document creates a synchronisation surface
that has to be maintained when strategy shifts. Alternative C
(replacing the carry-forward list in CLAUDE.md "Current Status")
was rejected because Current Status is a *where we are now*
snapshot, not a *where we're going* document — mixing them loses
both signals.

**`CLAUDE.md` "Current Status" keeps one factual state line in
addition to the spec pointer.** Without this, the next session's
context-load loses the instant answer to "where are we now"
(would require an extra read of the spec or `git log` to
reconstruct). The pointer-only design was rejected for that
reason.

**M5c OQ-4 (`BuildError::span()` + `Diagnostic` trait) classified
as trigger-driven cleanup**, with a soft trigger
("error-reporting ergonomics become a real pain point").
Justification: it does not fit any of the three strategic axes
and is not transformative on its own — closer in shape to OQ-7/8/9
than to attention-grammar / x86_64 / bare-metal. Risk: if
diagnostics never become painful, this stays dormant
indefinitely. Acceptable: that is exactly the trigger semantic.

### Problems encountered
- None blocking. Pure planning / repo-bookkeeping session.
- One classification ambiguity (M5c OQ-4) resolved as above.

### Next step
Brainstorm M9 = pick one of the three strategic axes. The
structure for that brainstorm is now constrained: not "what's
interesting today" but "which axis advances first, given the
dependency graph". The trade-off to surface during brainstorming:
unlocking-power (axis 2 has the deepest dependency chain so
delivers the most leverage per milestone — grammar + UIR + arm64
codegen + viewer in one direction) versus blast-radius
information value (axis 1 forces the first real
profile-isolation test, which validates the riskiest design
assumption in the project). Axis 3 (bare-metal `expf`) is the
smallest and most self-contained — a good fit if the next
milestone needs to be small.

---

## 2026-05-06 — Milestone 8 closed: arm64 codegen hardening + viewer v0.1

### What was done
- **`profiles/arm64/src/ops/dropout.rs`** — new `emit_dropout_copy`
  (mirror of `emit_relu` minus `fmax`). Triggered from a new
  `BufferLoc::OutputReg` branch in `codegen.rs::walk_model`'s
  `StdOp::Dropout` arm. Closes HIGH-severity bug: dropout placed
  at `model.output` previously left the caller's output buffer
  uninitialised. `debug_assert!` guards the OutputReg invariant
  at the function top.
- **`profiles/arm64/src/ops/{linear,relu,softmax}.rs`** — 17
  immediate sites (12 cmp + 5 mov) routed through `asm::emit_imm32`.
  Two placement strategies: Group A hoist-outside-loop for bl-free
  emitters (relu, matmul body) with distinct registers per nesting
  level (x10/x15/x16); Group B re-materialise-at-loop-top for
  bl-containing emitters (standalone softmax, RowWise softmax tail)
  where `bl _expf` clobbers caller-saved x10. Closes MEDIUM-severity
  bug: any production-scale dim (transformer hidden_dim 4096+, LLM
  vocab 30k+, classifier with > 4095 classes) previously failed to
  assemble or failed silently.
- **`compiler/src/ir/types.rs`** — three newtype wrappers
  (`VerboseUir`, `VerboseModel`, `VerboseNode`), each with their
  own `Display` impl. Plus `calls_extern_math` predicate methods
  on `Uir` and `UirModel`. UIR-level predicate (no profile
  coupling). Default `Display` for the underlying types unchanged.
- **`nflc/src/main.rs`** — new `--uir-verbose` flag on `parse`
  subcommand, mutually exclusive with `--uir`. Help text updated.
- **`docs/language_reference/uir.md`** — new "Viewing UIR" section
  (§7) documenting both flags and the `calls-extern-math` semantics.
- **`docs/profile_guide/arm64.md`** — new "M8 codegen hardening"
  section: dropout-as-output copy + dim-immediate uniformity.
- **New fixtures:** `tests/fixtures/{dropout_only,large_classifier_k,
  large_classifier_n}.nfl`.
- **15 new tests:** asm-shape positive checks (1 dropout-copy + 4
  Group A/B), 4 FFI integration (2 dropout-only variants + 2
  large_classifier), 3 predicate sub-cases, 1 verbose snapshot, 2
  CLI smoke (verbose render + mutual-exclusion). Test count
  208 → 223.

### Decisions made
- **Single PR with 3 atomic feature commits + holistic-review +
  closeout commits**, mirroring M5/M6/M7. No cross-commit
  dependencies.
- **`emit_dropout_copy` uses `emit_imm32` from birth** — Commit 1's
  new emitter ships with the new pattern, so Commit 2 patches
  exactly 17 pre-existing sites, not 18. No "TODO patch in
  Commit 2" debt.
- **Mov-site replacement reuses hoisted registers in Group A** —
  `mov x8, x15` / `mov x8, x16` instead of re-materialising via
  `emit_imm32`. Principle: avoid illegal immediates, not "always
  call the helper".
- **Group B accepts 1-2 movz/movk per loop iteration** in
  bl-containing loops. Adding x10 to the prologue's callee-saved
  set was rejected as out-of-scope blast radius; `bl _expf` is
  hundreds of cycles, < 1% relative overhead.
- **Newtype wrappers over `fmt_verbose` methods** — idiomatic Rust
  composition, no API pollution. Default `Display` unchanged.
- **`calls_extern_math` placed on UIR side, predicate logic
  duplicated with profile-side `node_uses_softmax`.** Deduplication
  is backlog OQ-NEW; trigger is next predicate-logic change.
- **No new error variant for dim-out-of-range** — `emit_imm32`
  already asserts on u32::MAX, ~1000× any realistic NN dim.
  YAGNI.
- **`--uir-verbose` documented in `uir.md`** (UIR rendering
  interface) rather than `arm64.md` (profile-specific). Reasoning:
  viewer is profile-agnostic.

### Problems encountered
- **Plan self-review caught two errors before commit:** FFI ABI
  signature was `(input, output, params)` in plan draft (wrong);
  actual ABI is `(input, params, output)`. Plan also referenced
  non-existent `common::cc_link` / `common::lower_fixture` helpers;
  existing convention is inline `read_to_string` + parse + build +
  run_pipeline + lower + `compile_to_dylib` + `libloading::Library`.
  Both fixed in plan-review commit before execution.
- **Code-quality reviewer flagged `dst_loc` as structurally
  redundant in `emit_dropout_copy`** — caller always guards on
  `BufferLoc::OutputReg`, making `materialise_ptr` always emit
  the same fixed instruction. Resolved by adding
  `debug_assert!(matches!(dst_loc, BufferLoc::OutputReg), ...)`
  at the function top to document the invariant without
  removing future flexibility. Quality reviewer's negative-`fmax`
  test assertion also added.
- **Existing tests pinned literal-imm cmps** — `relu_emits_separate
  _loop_with_fmov_zero_and_fmax` and `linear_emits_matmul_loops_
  with_fmadd` had assertions on `cmp x9, #4` / `cmp x3, #2` etc.
  Updated to register-form (`cmp x9, x10` / `cmp x3, x10` etc.) as
  part of Tasks 6-7. Caught by full workspace test runs.
- **fmt drift caught at Task 5** — Tasks 3 and 4 implementer
  subagents didn't run `cargo fmt --all` before reporting. Pre-
  Commit-1 fmt-check failed. Resolved by running fmt then
  proceeding to commit.
- **Holistic-review subagent caught 3 close-in findings** (CLAUDE.md
  Design Principle 5 stale "Until…ships (M8+)" hedge,
  CLAUDE.md "What NOT to Do" stale `(M7+)` label, uir.md status
  header still saying "Milestone 6 complete"). All fixed in
  `729a3e9 chore(m8/holistic)` before docs closeout.

### Next step
M9 brainstorming runs in a fresh worktree once M8 merges. Carry-
forward candidates: OQ-NEW (lift `node_uses_softmax` to single
source via `calls_extern_math`), OQ-7 (per-pass Result cleanup),
OQ-8 (lift rewriter to compiler/src/ir/), OQ-9 (NodeMutation
generalisation), profile-level viewer annotations,
MACHO_SYM_PREFIX rename when second profile starts, attention-
pattern grammar (NFL v0.2), bare-metal target,
BuildError::span() + Diagnostic trait.

---

## 2026-05-06 — Milestone 7 closed: shared 3-step rebuild helper extraction

### What was done
- **`compiler/src/passes/rewriter.rs`** — new shared helper module
  (`pub(crate)`). `RewritePlan` struct with `#[derive(Debug)]` holds
  three HashMaps (`consumer_count`, `victims`, `producer_post_ops`);
  constructor `RewritePlan::new(&model)` precomputes `consumer_count`.
  Function `rewrite_model(plan, model) -> UirModel` (plain return, no
  `Result`) walks `model.nodes` by old NodeId in topological order,
  branches on victim membership (redirects via `id_map`), takes
  ownership of non-victims and remaps their operands, optionally
  appends PostOps to producers, finally remaps `model.inputs` and
  `model.output`.
- **5 helper unit tests** in `rewriter.rs::tests` — pin behavior
  independent of any migrated pass: identity-on-empty-plan, victim
  drop + consumer redirect (4-node topology demonstrates direct
  operand-remap, not just model.output), PostOp push to producer,
  `model.inputs`/output remap, consumer count precomputation
  (including the absent-from-map orphan case).
- **Three pass migrations** (atomic units 2-4 of the M7 atomic-task-
  pack convention):
  - `EliminateDropout::eliminate_one_model` shrinks ~65 → ~26 lines.
    No producer mutation — just Dropout victims redirecting to their
    sole operand.
  - `FuseLinearRelu::fuse_one_model` shrinks ~99 → ~39 lines. All
    five victim criteria preserved (Relu kind, single operand, Linear
    producer, empty `fused_post_ops`, single consumer); pushes
    `PostOp::Relu` to producer.
  - `FuseLinearSoftmax::fuse_one_model` shrinks ~94 → ~39 lines.
    Mirror of FuseLinearRelu; pushes `PostOp::SoftmaxRow`.
  - All three pass functions changed signature `&UirModel` → `UirModel`
    (consume); `Pass::run` clones each model before calling.
  - All 21 per-pass unit tests (8 dropout + 11 relu + 5 softmax) +
    6 cross-pass tests + 3 FFI integration tests pass without test-
    body modifications. Bit-exact behavior preservation verified.
- **§8 invariant 6 unit test** —
  `leaves_linear_dropout_softmax_chain_untouched` in
  `fuse_linear_softmax::tests`. Closes M6 holistic-review Finding #7
  (coverage gap for the "FuseLinearSoftmax-without-EliminateDropout"
  degradation case).
- **`eliminate_dropout.rs:36-49` doc-comment** retired — the
  M7-deferred trigger fired and was closed by Task 1's helper.
  Replaced with a forward-pointer to `compiler::passes::rewriter`.
- **Drift-fix commit** (`4974cd7`) closed two close-in-M7 holistic-
  review findings: stale "step 3" doc-comment references in
  fuse_linear_relu.rs and fuse_linear_softmax.rs (replaced with
  forward-pointers to `rewriter::rewrite_model`); three
  `#[allow(dead_code)]` attributes in rewriter.rs removed (no longer
  needed after Tasks 2-4 wired up all callers).
- **Documentation closeout:** `PROJECT_SPEC.md` M7 row added marked
  "complete"; "Human-readable viewer v0.1" relocated from M7 to M8.
  `CLAUDE.md` "Current Status" rewritten reflecting M7 closure +
  carry-forward candidate list. `CLAUDE.md` Design Principle 5
  reference `(M7+)` → `(M8+)`.

### Decisions made
- **Plan-as-data API** chosen over closure-based or trait-based
  alternatives. Plan reads naturally as "decision-table the code
  already implicitly built"; debuggable as a value (`dbg!(&plan)`
  works because `RewritePlan` derives `Debug`); no heap allocation
  in the hot path. See spec §4.1 for rejected alternatives.
- **No lifetime parameter on `RewritePlan`** — struct holds only
  computed/declared data, not borrows. The `&UirModel` reference
  passed to `new()` is borrowed only during construction.
- **`rewrite_model` consumes both `plan` and `model`** (move
  semantics). `Pass::run` clones each model at the boundary before
  handing ownership to the consuming per-pass function.
- **`consumer_count` always computed in `new()`** (eager, simple)
  rather than lazy. EliminateDropout pays the O(N) walk it doesn't
  use — negligible cost. FuseLinearRelu/FuseLinearSoftmax both read
  the field for the single-consumer guard.
- **`rewrite_model` returns plain `UirModel`** — no `Result`
  wrapping. Helper has no real `Err` cases; preconditions are
  caller's responsibility, violations panic via `id_map[…]` lookup.
  Same YAGNI principle that ruled out defensive runtime checks.
  Per-pass functions retain `Result` for `Pass::run` compatibility
  (one-line `Ok(...)` wrap at the boundary).
- **Field name `victims`** (not `redirects`) — matches spec/code
  vocabulary; the keys ARE victims, the values tell what their
  references redirect to.
- **Migration order EliminateDropout → FuseLinearRelu →
  FuseLinearSoftmax** — simplest first (lowest blast radius if
  helper has bugs), largest test surface second (catches integration
  mismatches), mirror third.
- **Atomic-task-pack convention applied** — 4 sequential clean
  commits (helper-create + three migrations) with workspace green
  between each. Demonstrates M6 holistic-review Finding #11.

### Problems encountered
- **Brainstorm point-1 confusion:** during spec review on the M7
  worktree, an early review reading was based on M5c-stale `origin/main`
  state and flagged `FuseLinearSoftmax` as "doesn't exist yet". M6 had
  already merged via PR #14 (`2f95203`); the M7 worktree HEAD had the
  file. Resolution: when reviewing brainstorms on feature branches,
  explicitly verify "what's in HEAD worktree" rather than relying on
  a main-branch state mental model.
- **Plan draft used `redirects` field name** before the user proposed
  `victims` (more semantic). Renamed throughout.
- **Plan draft had `RewritePlan<'a>` with `model: &'a UirModel`
  field** — unnecessary lifetime. Caught and dropped before plan-write.
- **Plan draft `rewrite_model` returned `Result`** — same YAGNI debt
  as defensive runtime checks. Switched to plain `UirModel` before
  plan-write.
- **Helper unit test #2 originally used topology `Input → A → B`**
  (no Op-level operand consumer of A) — would pass even if the
  rewriter's operand-remap loop had a bug. Topology corrected to
  `Input → A → B → C` before plan-write.
- **§9 Task 2 test missed `use crate::Uir;` in inline imports** —
  would have compiled via module-level convention, but spec wasn't
  self-documenting. Fixed before plan-write.
- **§8 migration prose said `eliminate_one_model(model)?`** instead
  of `eliminate_one_model(model.clone())?` after the consume-model
  signature change. E0507 risk. Fixed before plan-write.

### Holistic review process — worth recording for M8+
The M7 holistic review (single subagent dispatch, spec / structure /
cross-cutting / docs / process scan) found 13 findings — slightly
fewer than M6's 15. Of the 13:
- 6 close-in-M7 (4 docs closeout + 2 drift-fix in `4974cd7`).
- 0 carry-forward to M8+.
- 7 acceptable deviations or process notes (line-count drift, plan
  test-count drift, commit-message typo, M5b version tags as historical
  pinpoints, profile unchanged, Debug-derive-beyond-template,
  atomic-task-pack process success).

Decision for M8+ continues: holistic review at every milestone close-out.
Cost ~5 minutes of subagent time; benefit: catches docs drift early.

### Known tech debt (carried forward to M8+)
1. **OQ-7 — Per-pass `Result<UirModel, PassError>` cleanup.** The
   per-pass `eliminate_one_model`/`fuse_one_model` functions return
   `Result` despite never producing `Err`. Same YAGNI debt as the
   M7-resolved `rewrite_model` Result. *Trigger:* first real
   `Err`-case in pass-level logic, OR discomfort from `Ok(...)`
   boilerplate accumulates across many passes. *Action:* refactor
   per-pass to plain `UirModel`; `Pass::run` wraps once.
2. **OQ-8 — Lifting `rewriter.rs` to `compiler/src/ir/`.**
   *Trigger:* a non-pass UIR-rewrite consumer (UIR-build phase
   optimisation, viewer renderer). *Action:* move module, change
   visibility.
3. **OQ-9 — Generalising `producer_post_ops: Vec<PostOp>` to
   `enum NodeMutation`.** *Trigger:* fourth pass needs producer
   mutation other than PostOp-push. *Action:* introduce
   `enum NodeMutation`, replace map value type.
4. **Carried over from M5c/M6** (still open per their respective
   triggers): OQ-1 (`FuseLinearPostOp` consolidation), OQ-2
   (type-level `PostOpKind`), OQ-3 (bare-metal `expf`), OQ-4
   (`BuildError::span()` + `Diagnostic` trait), OQ-6
   (`format!`/`to_string()` style consistency), M6 carry-forward
   item 2 (`_expf` AAPCS64 smoke test), M6 carry-forward item 4
   (CLI smoke future-proofing).

### Next step
**Milestone 7 fully complete.** Brainstorm M8 in a fresh worktree
once M7 merges. Open scope; candidate directions:
1. **OQ-7 per-pass `Result` cleanup** — small, decisive YAGNI
   closeout matching M7's helper-side change.
2. **Human-readable viewer v0.1** (PROJECT_SPEC M8 row).
3. **OQ-9 `NodeMutation` generalisation** — fires when a fourth
   pass needs non-PostOp mutation.
4. **`FuseLinearPostOp` consolidation** (OQ-1) — fires on next
   RowWise post-op.
5. **Bare-metal target** (OQ-3).
6. **`BuildError::span()` + `Diagnostic` trait** (OQ-4).
7. **Attention-pattern extension** — biggest scope; needs NFL v0.2.

---

## 2026-05-05 — Milestone 6 closed: attention-pattern fusion (`linear → softmax`)

### What was done
- **`PostOp::SoftmaxRow` variant** on the `#[non_exhaustive]` `compiler::ir::PostOp` enum. `Display` renders as `softmax_row` (lowercase snake_case, matching `Relu => "relu"` convention).
- **`compiler::passes::FuseLinearSoftmax` pass** — bias-aware from day one, parallel to `FuseLinearRelu`. Mirrors the canonical 3-step rebuild pattern (consumer count → victim identification → rebuild + remap). 5 unit tests pin all 5 victim criteria (single-consumer Linear, Softmax sole consumer, Softmax has Linear as sole operand, Linear's `fused_post_ops` empty, identity-when-no-Softmax). Cross-pass pipeline test `pipeline_eliminates_dropout_before_fusing_linear_softmax` confirms `linear → dropout → softmax` collapses through `EliminateDropout` then fuses.
- **`default_pipeline()` extended** to `[EliminateDropout, FuseLinearRelu, FuseLinearSoftmax]`. CLI `--passes <list>` and `--no-passes` work without code changes (the filter reads pass names dynamically).
- **arm64 RowWise emit branch** in `profiles/arm64::ops::linear::emit_linear`. After the existing matmul i-loop completes (writing the full M×N output), a separate i-loop runs Phases 2-4 of softmax in-place: row-max scan into `s8` (callee-saved), exp(x − s8) per element with `bl _expf` and sum-accumulate into `s9` (also callee-saved), normalise by `s9`. Labels prefixed `.Lfsmx_*` to avoid collision with the standalone-softmax `.Lsm_*` labels. Caller-saved `x6` is recomputed after each `bl _expf`.
- **`profiles/arm64::buffer::node_uses_softmax(node)`** — shared helper used by `compute_is_leaf` and `compute_callee_saved` to detect both standalone `StdOp::Softmax` and `Linear` with `PostOp::SoftmaxRow` in `fused_post_ops`. Both analysers return the correct answer (non-leaf, d8/d9 + x19-x23 saved) for fused-softmax-row Linears.
- **Shared test helpers** (`compiler/src/ir/test_utils.rs`, `pub(crate)`, `cfg(test)`): `input_node`, `op_node`, `out_dim_attr`, `rate_attr`. Promoted from inline in `eliminate_dropout.rs` (where they had been since M5b). Migrated the cross-pass test `pipeline_eliminates_dropout_before_fusing_linear_relu` and all 8 `eliminate_dropout` unit tests to use the shared module. `fuse_linear_relu` tests (which use the parser via `build("model M …")`) deliberately not migrated — out of M6 scope.
- **New fixture** `tests/fixtures/softmax_with_bias.nfl` — minimal (batch=4, input=8, output=3) with `linear[output, bias=true] -> softmax` as the final step. Exercises the bias-aware path through the RowWise tail.
- **FFI integration test** `fused_vs_unfused_softmax_match_numerically` in `profiles/arm64/tests/integration.rs`. Loops over `classifier.nfl` (no-bias) + `softmax_with_bias.nfl` (bias-aware). Compiles fused (`default_pipeline`) and unfused (`--no-passes`) asm. Links via `cc + libloading` and asserts bit-exact element-wise equality. Uses `assert_eq!` (not `debug_assert_eq!`) for `params_floats` agreement (OQ-5 fix applied from this test's first commit) and `FnSig`-driven buffer sizing (defensive cross-check against fixture-tuple constants).
- **OQ-5 harmonisation** retro-fitted to the M5a `fused_vs_unfused_classifier_match_numerically` and M5b `fused_vs_unfused_mixed_args_match_numerically` tests: both `debug_assert_eq!` instances replaced with `assert_eq!` (`debug_assert_eq!` is a no-op in release builds; the agreement claim should hold unconditionally).
- **CLI smoke** `compile_with_passes_filter_only_fuse_linear_softmax_runs` in `nflc/tests/cli_compile.rs` — confirms the dynamic pass registry exposes `fuse_linear_softmax` without CLI code changes; pins the stderr `note: applied passes:` format and the stdout asm shape (presence of `bl _expf` and `.Lfsmx_*` labels; absence of standalone `.Lsm_*`).
- **Documentation:** `docs/profile_guide/arm64.md` §3 / §4.10 (new) / §5 / §8 brought to M6 state. `docs/language_reference/uir.md` §2 mentions `SoftmaxRow` alongside `Relu` in the `fused_post_ops` field description. `PROJECT_SPEC.md` M6 row marks "complete". `CLAUDE.md` "Current Status" rewritten.
- **Drift-fixes from the M6 holistic review** (commit `a535184`): 6 close-in-M6 findings — stale doc-comment about helper-extraction trigger in `eliminate_dropout.rs`, "Pass N" → "Phase N" rename in `linear.rs` softmax tail, "Task N" plan-language → "M6" in source comments, M4b-era `RegSet` doc-comments updated, `node_uses_softmax` match overlap eliminated, two `#[allow(unreachable_patterns)]` comment wordings harmonised. None functional.

### Decisions made
- **Tasks 4+5+6 packed into a single commit** (`609eede`). Plan implicitly assumed independent tasks; in practice the asm-side dependency forced a combined commit (Task 4 alone would leave the workspace red because `default_pipeline()` includes `FuseLinearSoftmax` once the pass exists, and the asm-side `LowerError::UnsupportedPostOp` for `SoftmaxRow` triggers if the RowWise emit branch isn't there). A 4-test follow-up commit (`838cb7d`) added the per-task unit tests the combined commit skipped (`is_leaf_false_for_fused_softmax_row_linear`, `callee_saved_includes_d8_d9_for_fused_softmax_row`, `emit_linear_with_softmax_row_post_op_emits_three_phase_softmax`, `emit_linear_with_softmax_row_post_op_preserves_bias_add`).
- **Two acceptable spec deviations**, both correctly reflected in arm64.md §4.10:
  - **Two-pass i-loop structure**: full matmul i-loop completes (writes the entire M×N output buffer), then a separate i-loop runs Phases 2-4. The spec sketch implied a per-row interleaved structure. The two-pass form is simpler to reason about (no save/restore dance for `x3` around the tail) at the cost of cache locality for large M (negligible for the typical NFL fixtures, M ≤ 32).
  - **`-inf` bit-pattern init for `s8`** (`movz w0, #0x0000; movk w0, #0xFF80, lsl #16; fmov s8, w0`) instead of `s8 = row[0]`. Mirrors the canonical `emit_softmax` pattern (consistency wins).
- **No defensive `emit_linear` stacking check.** The plan briefly mentioned a defensive `has_row_wise && fused_post_ops.len() > 1 → UnsupportedPostOp` check inside `emit_linear`. Not implemented — the pass-level `FuseLinearSoftmax` criterion 4 (`fused_post_ops.is_empty()`) is the only guard against `[Relu, SoftmaxRow]` stacks. Single source of truth; documented in arm64.md §4.10 + the `PostOp::SoftmaxRow` doc-comment.
- **Helper-extraction order of operations.** §10 of the spec was followed: extract `compiler/src/ir/test_utils.rs` BEFORE writing any M6 unit tests, migrate the existing M5b cross-pass test that hand-built verbose `Node` literals, only then write M6 tests through the shared helpers. Avoided the alternative ordering's double-touch hazard.

### Problems encountered
- **Plan-phase fixture audit (R3 from the spec):** the `classifier.nfl` final layer (`linear[output] -> softmax`) has `bias=false` (NFL default), so the bias-aware path through the RowWise tail isn't exercised by the existing fixture. Resolved by adding `tests/fixtures/softmax_with_bias.nfl` as a parallel small fixture covering `linear[output, bias=true] -> softmax`. The new FFI integration test loops over both fixtures.
- **Cross-crate test_utils visibility:** `compiler::ir::test_utils` is `#[cfg(test)] pub(crate)`, making it invisible to `profiles/arm64`'s test compilation. The unit tests in `profiles/arm64::buffer::tests` and `profiles/arm64::ops::linear::tests` couldn't import the shared helpers directly; they construct fused UIR via `compiler::parse + ir::build + run_pipeline(default_pipeline)` instead. Acceptable workaround; cross-crate exposure of the helpers is a future-decision item if `profiles/x86_64` ever adds equivalent tests.
- **Plan's draft `AttrValue::Boolean` for the bias attribute** was wrong: `compiler::ir::types::AttrValue` has `Integer(u64)`, `Float(f64)`, and `Symbol(String)` variants, no `Boolean`. The implementation uses `AttrValue::Symbol("true".into())` to mirror the existing convention (also confirmed via `linear_has_bias()` in `stdlib.rs`).
- **Plan-language drift in source comments:** several "Task 5" / "Task 6" plan references leaked into committed source. Caught by the M6 holistic review (Finding #4) and renamed to "M6" in commit `a535184`.

### Holistic review process — worth recording for M7+
The M6 post-merge holistic review (single thorough subagent dispatch, spec / structure / cross-cutting / docs / process scan) found 15 findings vs. the per-task reviews' typical 1-3 findings each. Of the 15:
- 7 close-in-M6 (landed in commit `a535184`).
- 5 carry-forward to M7+ (recorded below).
- 2 acceptable deviations (no action; documented in this entry).
- 1 process finding for M7+ planning (atomic-task-pack convention).

Decision for M7+: continue the holistic-review-per-milestone pattern. The cost (one subagent dispatch ~5 min) consistently catches drift the per-task reviews miss.

### Known tech debt (carried forward to M7+)
1. **Shared 3-step rebuild helper extraction.** Three identical bodies now exist in `eliminate_dropout.rs`, `fuse_linear_relu.rs`, `fuse_linear_softmax.rs`. The "three strikes" trigger has fired but extraction was deferred to keep M6 focused. M7+ first task candidate.
2. **`_expf` AAPCS64 smoke test** (spec §13 R5): direct unit test in `profiles/arm64::asm::tests` pinning that `_expf` preserves d8/d9. The FFI integration test covers this transitively; the explicit smoke test is low-priority hygiene.
3. **§8 invariant 6 unit test** (Finding #7): a small unit test for the "(--passes fuse_linear_softmax alone leaves linear → dropout → softmax untouched)" degradation case. Logic verified by code review; coverage gap is trivially closeable.
4. **CLI smoke test future-proofing** (Finding #8): the `!stderr.contains("eliminate_dropout")` assertion in `compile_with_passes_filter_only_fuse_linear_softmax_runs` would break if a future M7+ pass adds the substring to a dynamic available-passes listing. Switch to `!stderr.contains("note: applied passes: eliminate_dropout")` style when the test becomes brittle.
5. **Plan-convention for atomic task packs** (Finding #11): when a feature pack has mutual asm-side ↔ pass-side dependencies that would leave the workspace red mid-implementation, the plan should explicitly mark those tasks as "atomic / single commit" up-front. Apply this convention from M7's plan.
6. **Carried over from M5c** (still open):
   - **OQ-1 `FuseLinearPostOp` consolidation** — fires on a third access pattern OR a second RowWise post-op.
   - **OQ-2 type-level `PostOpKind` distinction** — same trigger plus emit-shape divergence between RowWise variants.
   - **OQ-3 bare-metal `expf`** — fires on user-driven embedded need.
   - **OQ-4 `BuildError::span()` accessor + shared `Diagnostic` trait** — fires on a fourth error type or generic CLI rendering path.
   - **OQ-6 `format!`/`to_string()` style consistency** — fires on the next cascade-arm touch.

### Next step
**Milestone 6 fully complete.** Brainstorm M7 in a fresh worktree once M6 merges. Open scope; candidate directions (priority-ordered from the holistic review):
1. **Shared 3-step rebuild helper extraction** (the M6-deferred trigger; ~30-50 lines, decisive small win).
2. **Attention-pattern extension** — Q/K/V projections, scaled dot-product, axis-N softmax. Requires NFL v0.2 grammar work first; biggest scope.
3. **`FuseLinearPostOp` consolidation** (OQ-1) — fires when the next RowWise post-op (LayerNorm, attention-axis softmax) lands.
4. **Bare-metal target** (OQ-3) — Taylor-series `expf`, second arm64 sub-profile.
5. **`BuildError::span()` + `Diagnostic` trait** (OQ-4) — landed if a fourth error type appears or the CLI gains generic error rendering.

---

## 2026-05-05 — Milestone 5c closed: M5 cycle close-out (docs sync + small consistency fixes)

### What was done
- Applied 13 of 17 findings from the M5b post-merge holistic review
  (Option B scope from the brainstorming session). 4 holistic-review
  findings explicitly deferred to M6+ (1.2 shared `Diagnostic` trait,
  2.1 `BuildError::span()` accessor, 4.1 test-helper extraction, 6.1
  pass struct visibility). 13 + 4 = 17 ✓. DEVLOG-1
  (`debug_assert_eq!` → `assert_eq!`) is pre-existing M5b tech debt
  also carried forward — listed in §"Known tech debt" but not part
  of the holistic-review punch-list arithmetic.
- Code consistency (3 small Rust changes + cascade fixes the plan
  didn't anticipate):
  - `impl std::error::Error for PassError` (`compiler/src/passes/mod.rs`).
  - `impl std::error::Error for LowerError` (`profiles/arm64/src/types.rs`).
  - All five workspace error types now implement `std::error::Error`
    uniformly (`BuildError`, `ParseError`, `LexError`, `PassError`,
    `LowerError`).
  - `nflc/src/main.rs` — four `&e.message` call sites (lines 253, 338,
    343, 369) → `&e.to_string()` for `render_error_with_snippet`
    consistency. The plan only cited line 253; code review caught the
    other three.
  - `#[non_exhaustive]` on `compiler::ir::stdlib::StdOp`. Cascade
    surfaced THREE locations needing wildcard arms (the plan named one):
    `profiles/arm64/src/codegen.rs::walk_model` (`LowerError::UnsupportedOp`),
    `profiles/arm64/src/codegen.rs::classify_op` (also `UnsupportedOp`),
    and `profiles/arm64/src/buffer.rs::assign_buffers` (defaults to
    stack-allocated, identical to `Linear|Softmax` arm — `classify_op`
    rejects unknown ops downstream so this allocation is harmless).
    `LowerError::UnsupportedOp` lost its `#[allow(dead_code)]` attribute
    (M4b-era — variant is now reachable through two cascade arms).
  - All three wildcard arms use `#[allow(unreachable_patterns)]` —
    same pattern M5a's `emit_linear` uses for the `PostOp` wildcard.
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
    §3 heading also lost the "in M4b" version tag (it was stale; the
    table content carries sub-milestone provenance per row).
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
    open M6 candidate directions.

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
  the two files.
- Task 1 plan under-specified Finding 5.1 cascade scope (named only
  `walk_model`, missed `classify_op` and `buffer.rs::assign_buffers`).
  Implementer caught all three when `cargo build` failed; commit
  message was updated to document the actual file count (6 files,
  not the planned 5).
- Task 1 plan under-specified Finding 2.2 scope (named only line 253,
  missed three more `&e.message` call sites at lines 338, 343, 369).
  Code review caught the gap; followup commit closed it.
- Both gaps were "punch-list-cited line was the obvious one; the rest
  needed a grep". Lesson for M6+ planning: when applying findings with
  citations to a single line, verify with `grep` that the cited
  instance is the only one before scoping the task.

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
6. **Holistic-review false-positive auditing** — find a way to
   spot-check reviewer claims against actual file content before
   integrating findings. Mitigates the rare 3.4-style conflation.
7. **`format!("{op}")` vs `op.to_string()` style consistency** in the
   profile's wildcard arms. M5a's `emit_linear` `PostOp` wildcard uses
   `to_string()`; M5c's three new `StdOp` wildcards use `format!`.
   Both work; harmonise in M6+ when the cascade arms get touched
   again.

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

---

## 2026-05-05 — Milestone 5b closed: bias-aware fusion + EliminateDropout + --passes filter

### What was done
- Lifted M5a's `if linear_has_bias { continue; }` guard in
  `FuseLinearRelu`. `linear[bias=true] → relu` now fuses into a single
  `emit_linear` block that stacks `matmul → bias-add → fmax → store`.
  No profile-side changes — `emit_linear` already supported the asm
  shape; only the pass-level rejection blocked it.
- Added `compiler::passes::eliminate_dropout::EliminateDropout` —
  a new UIR-pass that removes every Dropout node from the graph,
  remapping consumers and `model.inputs` / `model.output` to the
  dropout's operand. Functional 3-step rebuild (identify victims →
  rebuild with id-remap → remap inputs/output). 8 inline unit tests
  cover terminal-dropout, internal-dropout, chained dropouts,
  multi-consumer dropout, identity-when-no-dropout, and explicit
  inputs/output remap correctness.
- `default_pipeline()` now registers BOTH passes in canonical order
  `[EliminateDropout, FuseLinearRelu]`. Order matters: without it,
  `linear → dropout → relu` patterns would never fuse. The doc-comment
  documents the dependency. M6+ may revisit if a third pass needs
  non-trivial coordination.
- New end-to-end pipeline test
  `pipeline_eliminates_dropout_before_fusing_linear_relu` proves the
  order-dependency: hand-built UIR `linear → dropout → relu` collapses
  to two nodes (input + fused linear with `fused_post_ops == [Relu]`)
  through the full pipeline.
- CLI: `--no-fuse` renamed to `--no-passes`. Clean break — no alias,
  no `#[allow(dead_code)]` shim, `grep "no_fuse|--no-fuse"` against
  `nflc/src/`, `compiler/src/`, and `profiles/` confirms zero residue.
  New `--passes <list>` flag for filtered runs: comma-separated,
  validated against the dynamic `default_pipeline()` registry, mutually
  exclusive with `--no-passes`, emits a stderr `note:` when user-typed
  order diverges from canonical.
- 4 new CLI smoke tests cover `--passes` filter, unknown-name
  rejection (with dynamic available list), order-divergence warning,
  and mutually-exclusive interaction. `compile_with_no_fuse_skips_fusion`
  renamed to `compile_with_no_passes_skips_pipeline` (assertion strings
  updated to new flag/note shape).
- Integration test `fused_vs_unfused_mixed_args_match_numerically`
  proves bit-exact equivalence for the bias-aware case using
  `mixed_args.nfl` (which has `linear[16, bias=true] → relu` as its
  first internal layer). Mirrors M5a's classifier test pattern with
  one additional load-bearing pre-assert (`fadd s0, s0, s5` inside
  fused linear) — pins that bias-add survives the lift.
- Existing M4b/M5a integration tests (`mixed_args_runs_correctly`,
  `classifier_runs_correctly`, `fused_vs_unfused_classifier_match_numerically`,
  others) continue to pass without changes — the pipeline-order
  switch is automatic via M5a Task 10's adaptation.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-05-m5b-bias-fusion-eliminate-dropout-design.md`
during brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-05-m5b-bias-fusion-eliminate-dropout.md`
(7 tasks, ~14 commits including review-driven polish).

### Pre-decided architectural calls (from spec §4)
1. **Pipeline order: `[EliminateDropout, FuseLinearRelu]`.** Hardcoded
   in `default_pipeline()` with explanatory comment. Fixed-point /
   dependency-declaration deferred to M6+ when a third pass with
   non-trivial coordination lands.
2. **Profile keeps `BufferLoc::Alias(operand)` for Dropout.** Fallback
   for `--no-passes` mode; profile remains complete relative to its
   input grammar. A verification tool that fails on valid UIR isn't a
   verification tool — it's a trap.
3. **`--no-fuse` removed without alias.** v0 has no external consumers;
   backward-compat aliases here would be cargo-cult.
4. **`--passes` is filter-only, canonical order enforced.** Reorder
   mode (B-variant) deferred to M6+ if a real research case demands it.
5. **No shared helper for victim/remap pattern.** Two passes duplicate
   the rebuild skeleton intentionally — "three strikes then refactor"
   rule. EliminateDropout's doc-comment flags this for the M6+ author.

### Problems encountered
- None blocking. The spec went through five review rounds during
  brainstorming (user caught three placeholder/contradiction issues,
  one E0505 borrow-checker bug in the pseudocode, and one
  cross-reference typo before the plan was written). All five fixed
  inline before implementation began.
- Implementation surfaced one emergent breakage at Task 3: the M5a
  CLI smoke test `compile_default_runs_fusion` asserted the
  single-pass `applied passes:` string, which broke once
  `default_pipeline()` grew a second pass. Implementer fixed inline
  to keep the workspace green at every commit (sensible deviation
  from the original task scope).
- Test count finished at 188 (matches plan target exactly: 173 + 15).

### Known tech debt (carried forward)
1. **Profile guide doc updates** (`docs/profile_guide/arm64.md`):
   bias-aware fusion section, `--no-passes` / `--passes` documentation,
   EliminateDropout removal note. → **M5c**.
2. **`PROJECT_SPEC.md` milestones table** close-out for M5 → **M5c**.
3. **Pass-shared helper for victim/remap pattern** — defer to M6+ when
   the third pass with the same structural pattern lands ("three
   strikes then refactor"). DEVLOG and EliminateDropout doc-comment
   flag the rationale.
4. **`--passes` reorder mode (B-variant)** — only if research / debugging
   case demands it. M6+.
5. **Pass dependency declaration / fixed-point iteration** — when a
   third pass with non-trivial coordination lands. M6+.
6. **Snapshot tests via `insta`** — substring asserts continue to suffice.
7. **`debug_assert_eq!` for fused/unfused FnSig agreement** — currently
   in both M5a and M5b integration tests; would be strictly safer as
   `assert_eq!` (fires once per test invocation). Not a regression;
   noted by code review as a pre-existing pattern.

### Next step
**Milestone 5b complete.** M5 remains technically open until M5c lands
the documentation: profile guide updates for bias-aware fusion +
EliminateDropout + the new CLI flags, plus the PROJECT_SPEC milestones
close-out. M5c is small (docs only, no code changes) and can be a
single-commit milestone.

After M5c: brainstorm M6 in a fresh worktree once main is updated
post-M5b-merge. M6 is open territory — possible directions: bare-metal
target, attention-pattern fusion (`linear → softmax_max`), x86_64
profile, or pass-helper extraction triggered by a third pass with the
same victim/remap structural pattern.

---

## 2026-05-04 — Milestone 5a closed: kernel fusion (linear → relu) + UIR-pass framework

### What was done
- Introduced `compiler::passes` UIR-pass infrastructure: `UirPass` trait
  with mandatory `name()` + functional `run(&Uir) -> Result<Uir, PassError>`,
  `default_pipeline()`, `run_pipeline()`. `PassError` `#[non_exhaustive]`
  with `InvalidInput` variant carrying span; `span()` accessor returns
  `Span` directly with a documented migration plan if a future variant
  ever lacks one.
- Implemented `FuseLinearRelu` pass — finds `Linear (no bias=true,
  no existing fused_post_ops, single consumer) → Relu`, merges via
  `Linear.fused_post_ops = vec![PostOp::Relu]`, removes Relu node, remaps
  references with fresh NodeIds via per-model functional rebuild. 10 inline
  unit tests cover all spec edge cases (terminal, chain, multi-consumer-
  relu allowed, multi-consumer-linear forbidden, bias-true skip, double-
  fusion skip, softmax→relu skip, NodeId remap).
- Extended UIR types: new `pub enum PostOp { Relu }` `#[non_exhaustive]`,
  separate from `StdOp` by design (Softmax/Dropout/Linear don't fit as
  post-ops). `NodeKind::Op` gains `fused_post_ops: Vec<PostOp>` field.
  `Display for Node` renders optional `fused=[<list>]` suffix only when
  non-empty (back-compat for M3c+ `nflc parse <file> --uir` output).
- Relocated `linear_has_bias` from `profiles/arm64::codegen` to
  `compiler::ir::stdlib` so passes can use it.
- Profile changes: `profiles/arm64::emit_linear` accepts `node_span`
  and `fused_post_ops`, returns `Result<String, LowerError>`. Materialises
  `s4 = 0.0` once if any `PostOp::Relu` in `fused_post_ops`. Emits
  `fmax s0, s0, s4` between bias-add and store. The required catch-all
  arm on the `match post_op` (mandatory for `#[non_exhaustive]` PostOp)
  returns `LowerError::UnsupportedPostOp` (new variant).
- CLI: refactored arg-parsing into `parse_compile_args` stateful parser
  (replaces the 3-arm slice-position match). New `--no-fuse` flag.
  Default mode applies `passes::run_pipeline` between `ir::build` and
  `profile.lower`; `--no-fuse` skips. `note:` lines emit to **stderr**
  only after the pipeline succeeds (strict stdout/stderr discipline:
  stdout = asm only, pipeable to `cc`).
- Integration test `fused_vs_unfused_classifier_match_numerically`
  exercises `classifier.nfl` (2 internal fusions) on both paths,
  asserts `assert_eq!` (bit-exact, not epsilon) on all 320 output
  elements. Existing M4b integration tests switched to the default-fused
  path; numerical assertions hold automatically by bit-exactness.
- 3 CLI smoke tests via `Command::new(env!("CARGO_BIN_EXE_nflc"))`:
  default-runs-fusion, --no-fuse-skips, unknown-flag-rejected.
- `parse_compile_args` rejects flag-as-path early (e.g.
  `nflc compile --no-fuse --profile arm64` errors with a clear message
  rather than producing a confusing `cannot read --no-fuse`).

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-04-m5a-kernel-fusion-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-04-m5a-kernel-fusion.md` (11 tasks,
22 commits including review-driven polish).

### Pre-decided architectural call
> **Fusion lives at UIR-pass level, not codegen-time peephole.** Two
> reasons (per user during brainstorming): visibility (consumer counts
> are visible only on the UIR — Linear→Relu fusion is safe iff Linear has
> exactly one consumer, which is invisible to a peephole walking codegen
> dispatch arms) + profile isolation (`PROJECT_SPEC.md` design principle 3
> — profiles consume already-fused graphs and emit accordingly; the
> fusion logic itself is profile-agnostic).
>
> Right separation of concerns: UIR-passes decide *what* fuses;
> codegen decides *how* to emit fused ops.

### Problems encountered
- None blocking. A handful of plan-text rough edges were caught by code
  reviewers and fixed inline:
  - Task 7 plan emitted the `note: applied passes` line *before* running
    the pipeline; on a pass error, the user would see a misleading
    success-style message followed by the error. Moved the note into the
    `Ok` arm.
  - Task 8 plan used a loose `|| stderr.contains("error:")` fallback in
    the unknown-flag assertion that would silently pass for any error
    path. Tightened to the strict `unknown flag: --frobnicate` substring.
  - Task 4 plan prose said "9 tests" but enumerated 10; implementer
    delivered all 10 (correct), and the plan's count was an undercount.
  - Task 9 plan hardcoded the params length (`535040`); switched to
    `fused_asm.functions[0].params_floats` so the test follows the
    fixture if it ever changes.
- Test count grew slightly past the plan target (173 vs. 170) because
  review polish added two defensive tests:
  `pipeline_halts_on_first_error_and_propagates` (Task 3 review N-2) and
  `unsupported_post_op_display_and_span_round_trip` (Task 5 review N-3).

### Known tech debt (carried forward)
1. `EliminateDropout` pass deferred to M5b. The dropout-as-noop alias in
   `buffer.rs::assign_buffers` (M4b) continues to handle dropout at
   profile level; M5b moves removal up to UIR-pass.
2. `linear[bias=true] → relu` fusion deferred to M5b. M5a's pass
   condition explicitly excludes `linear_has_bias` candidates.
3. `--passes=X,Y` filter syntax deferred to M5b. M5a only has the
   binary `--no-fuse` flag; `name()` foundation is in place.
4. Profile guide doc updates deferred to M5c. The fusion section,
   asm patterns, and CLI flag docs land in `docs/profile_guide/arm64.md`
   when M5c closes M5.
5. Snapshot tests via `insta` not introduced in M5a (substring asserts
   sufficient at this scope).

### Next step
**Milestone 5a complete.** Recovers M4a's in-place relu performance via
a pass-based fusion infrastructure that future passes (`EliminateDropout`,
bias-aware fusion, M6+ multi-pattern fusion) can extend without changing
the profile contract.

M5b adds bias-aware fusion + `EliminateDropout` + `--passes=X,Y` filter.
M5c closes M5 with profile guide doc updates and PROJECT_SPEC milestone
close-out. Brainstorming for M5b runs in a fresh worktree once main is
updated post-M5a-merge.

---

## 2026-05-04 — Milestone 4b closed: arm64 profile covers all 5 M3 fixtures end-to-end

### What was done
- Redesigned `FnSig` ABI: `weight_floats` removed, replaced by `params_floats`
  + `params_layout: Vec<ParamSlot>` with typed slots (`LinearWeight`,
  `LinearBias`). Generated functions take a single packed `params` buffer
  containing all weights and biases in topological UIR-node order.
  **This is a deliberate ABI break vs M4a** — see "ABI break callout" below.
- Added `profiles/arm64/src/buffer.rs`: `assign_buffers` (BufferLoc per node:
  InputReg / OutputReg / StackOffset / Alias), `compute_is_leaf`,
  `compute_callee_saved` (RegSet for d8/d9 + x19_x23). `BufferAssignment`
  carries 16-byte aligned total stack size.
- New prologue/epilogue helpers in `asm.rs`: `format_function_prologue` /
  `_epilogue` accept `LeafKind` + `RegSet` + intermediate-bytes. Conditional
  layers: callee-saved x19-x23 (iff softmax), callee-saved d8/d9 (iff
  softmax), non-leaf x29/x30 (iff bl present), sub/add sp (iff intermediates
  > 0). Large-immediate handling via shifted-12 or movz/movk + sub sp, sp,
  x9. New `emit_imm32` helper for arbitrary 32-bit immediate materialisation.
- Refactored `codegen.rs` body emission into `profiles/arm64/src/ops/`
  submodules (mod, linear, relu, softmax, dropout). Per-op emitters take
  `model_idx` + per-op counter for label namespacing across multi-model
  fixtures (e.g. pipeline_styles.nfl → labels like `.Lmm_i_<m>_<l>:`).
- New ops:
  - `linear[N, bias=true]`: matmul + bias-add inline after k-loop.
  - `dropout`: zero asm; `BufferLoc::Alias(operand)` propagation.
  - `softmax`: 3-pass numerically stable (max → exp+sum → normalize),
    `bl _expf`, callee-saved s8/s9 for max+sum, callee-saved x19-x23 for
    loop state across `bl _expf` (i, row_base, k, src ptr, dst ptr).
    `-inf` materialisation via `movz w0, #0x0000; movk w0, #0xFF80, lsl #16;
    fmov s8, w0` (since `fmov sN, #-inf` is invalid AArch64).
- Errors for `linear[N, bias=true]`, `dropout`, `softmax`, and duplicate
  model names are no longer emitted by the lowerer (all paths supported).
  Duplicate-model-name check moved up to `compiler::ir::build` as
  `BuildErrorKind::DuplicateModelName { name, first_span }`.
  `render_error_with_snippet` extended with optional `first_span` →
  emits trailing `note: previously defined at file:line:col` plain-text
  (single-snippet for M4b; rustc-style two-snippet upgrade is M4c-or-later).
- New fixture-driven integration tests via FFI: `tinymlp_full_with_softmax_runs_correctly`,
  `classifier_runs_correctly`, `pipeline_styles_runs_correctly`,
  `comments_runs_correctly`, `mixed_args_runs_correctly`. Plus
  `m4a_no_softmax_still_runs` adapted for the new ABI. All run on
  aarch64 macOS host; skip cleanly elsewhere.
- 2 reference-validation tests (`reference_softmax_stable_known_values`,
  `reference_bias_add_known_values`) pin hand-computed values so an
  asm-and-reference shared bug can't silently pass integration tests.
- `docs/profile_guide/arm64.md` extended (~270 lines added) with bias-add,
  softmax 3-pass, dropout aliasing, intermediate buffer pattern, non-leaf
  prologue with d8/d9 + x19-x23, per-model label namespacing, libm
  dependency note. Limitations greatly reduced.
- `docs/language_reference/uir.md` cross-links to the arm64 guide for both
  optional-attribute interpretation and dropout-as-noop semantics.
- `PROJECT_SPEC.md` milestones table M4 row updated to '4a + 4b complete';
  Architecture Profiles arm64 row expanded.

### ABI break callout

> **M4b deliberately broke the M4a public ABI of `FnSig`.** `weight_floats`
> field is gone; replaced by `params_floats` + `params_layout: Vec<ParamSlot>`.
> The generated `nfl_forward_*` C function signature changes the second
> parameter from `const float* weights` to `const float* params` (semantically
> the same buffer for M4a-compatible models — single LinearWeight slot — but
> renamed to reflect the more general layout).
>
> **Why deliberately:** the M4a name `weight_floats` would have been a lie
> the moment any M4b-supported model used `bias=true` (`params` then
> contains a LinearBias slot too). Renaming + restructuring at M4b is
> correct; retrofit-compat shims would have been worse.
>
> No external consumers exist (project is internal v0.1). Future readers of
> git history: this break was intentional, see
> `docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` §5.4.

### Critical bug caught + fixed during code review

Initial Task 8 (softmax) implementation kept loop state in caller-saved
registers (x3, x4, x5, x6, x11, x12) across `bl _expf`. Per AAPCS64, x0-x18
are caller-saved; `_expf` is allowed to clobber them. Apple libm `expf`
happens to preserve them today, but that's coincidence not contract — non-
Apple targets or libm updates would silently break.

Fix: moved loop state into callee-saved x19-x23 (i, row_base, k, src, dst);
x6 (element offset) is recomputed after each call. RegSet gained `x19_x23`
bit; prologue saves `x19, x20, x21, x22, x23` (two stp pairs + one str)
when softmax is present.

This is the kind of bug that could pass all integration tests on Apple
silicon but blow up on first Linux-arm64 CI run. Defensive fix landed in
the same Task 8 cycle as the spec/quality reviews.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-04-m4b-arm64-coverage.md` (12 tasks; ~21
commits including review-polish commits).

### Problems encountered
- Task 8 critical AAPCS64 violation (caller-saved registers across
  `bl _expf`) — caught by code review, fixed before integration tests ran.
- Task 9 implementer made signature changes to emit_linear/emit_relu (added
  `model_idx` for multi-model label namespacing) but session was
  interrupted before they could update softmax.rs and codegen.rs's
  dispatch arms. Resumed inline: completed the model_idx threading
  through softmax.rs + walk_uir/walk_model + the test assertions on
  label format.

### Known tech debt (carried forward)
1. Single-snippet rendering for `DuplicateModelName` (plain-text note for
   first_span). Two-snippet rustc-style upgrade is M4c-or-later.
2. Integration test tempdir not cleaned up (carried from M4a).
3. Performance: scalar code, mul-based indexing, no fusion, no SIMD. M5+.
4. `LowerError::UnsupportedOp` kept defensively (`#[allow(dead_code)]`);
   exercised by `unsupported_op_display_and_span_round_trip` to prevent
   Display/span impl rot.
5. Bare-metal arm64 target needs a separate profile (Taylor `exp` instead
   of libm). M7+.

### Next step
**Milestone 4 fully complete (4a + 4b).** All 5 M3 positive fixtures lower
end-to-end through the arm64 profile to runnable native code. 148 tests
pass; build/clippy/fmt/CI all clean.

The next milestone is **Milestone 5 — kernel fusion pass**: introduce an
optimisation pass on the UIR (or just-before-codegen) that fuses
`linear → relu` (and similar elementwise-after-matmul patterns) into a
single loop with the relu inlined into the matmul store. Recovers M4a's
in-place relu performance and sets up the framework for more aggressive
fusion (matmul→bias→relu→softmax_max etc.). Brainstorming for M5 runs in a
fresh worktree once main is updated post-M4b-merge.

---

## 2026-05-03 — CI workflow added; closes M3a tech-debt #3

### What was done
- Added `.github/workflows/ci.yml` with two jobs:
  - `unit` on `ubuntu-latest`: `cargo fmt --all -- --check`, `cargo clippy
    --workspace --all-targets -- -D warnings`, `cargo build --workspace`,
    `cargo test --workspace`. Integration test in `profiles/arm64/tests/`
    self-skips on x86_64, so unit-test count is just lexer/parser/ir/profile-unit.
  - `integration` on `macos-14` (Apple Silicon arm64): `cargo build --workspace`
    + `cargo test --workspace`. The full FFI integration test runs here
    (assembles via `cc`, dlopens .dylib, calls `nfl_forward_M4Demo` via
    libloading).
- Triggers: push to `main`, push to `claude/**`, PR to `main`.
- Uses `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2` for
  toolchain + cache.
- Added CI badge to `README.md`.
- Pre-CI: applied `cargo fmt --all` across workspace (117 sites in 20 files).
  Pure formatting, no semantic changes; committed separately so the CI PR
  is review-friendly.

### Decisions made
- **Format check IS in CI** (not just installed). Per the user's note:
  installing rustfmt but never running it is wasted seconds on every CI run.
  Project culture is zero-warnings; format is part of that.
- **Two jobs, not one matrix.** macOS arm64 is paid; ubuntu is free. Splitting
  jobs lets the cheap one fail-fast on lint/fmt without burning macOS minutes.
- **No nightly, no msrv matrix.** YAGNI for v0.1. Single `stable` toolchain.
- **No coverage.** YAGNI for v0.1. Tarpaulin/llvm-cov can come later if needed.

### Problems encountered
- 117 fmt-drift sites across 20 files when first checked. Resolved by running
  `cargo fmt --all` and committing as a separate `style:` commit before the
  CI YAML.

### Next step
**M3a tech-debt #3 closed.** CI now gates every push to main / `claude/*` and
every PR. The next milestone is **Milestone 4b** — bias=true in linear,
dropout (no-op pass-through), softmax (scalar exp). Brainstorming starts in a
fresh worktree once this CI PR merges to main.

---

## 2026-05-03 — Milestone 4a closed: arm64 scalar codegen — first machine-executable output

### What was done
- Workspace restructured into 3 crates: `compiler/` (lib only), `nflc/` (bin
  only), `profiles/arm64/` (lib only). Empty placeholder dirs
  `profiles/{generic,x86_64,riscv64}/` deleted. `compiler` package renamed
  from `nflc` to `compiler`. 25 mechanical `nflc::` → `compiler::` import
  rewrites across `nflc/src/main.rs`, `compiler/tests/uir_fixtures.rs`,
  `compiler/tests/fixtures.rs`. Stale `.gitkeep` markers removed.
- `profiles/arm64` lib crate. Public surface: `pub fn lower(uir: &Uir) ->
  Result<Asm, LowerError>`. Types: `Asm`, `FnSig`, `LowerError`
  (`#[non_exhaustive]`, 4 variants). Internal modules: `codegen.rs` (UIR
  walker, per-op emitters, classify_op upfront validation), `asm.rs`
  (function header/footer helpers + Mach-O symbol prefix), `tests.rs` (10
  unit tests).
- Lowering covers `linear[N]` without bias (matmul: 3 nested scalar loops
  with `fmadd`, `mul`-based index arithmetic), `relu` (separate elementwise
  loop with `fmov s4, wzr` once + `fmax s3, s3, s4` per element, in-place
  on `x2` output buffer), and `Input` (marker, no code).
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
  against a pure-Rust matmul+relu reference. **Test passed first time with
  the planned `1e-5` tolerance — no FMA divergence flake.**
- New `docs/profile_guide/arm64.md` (217 lines): ABI, buffer layout,
  supported ops, asm patterns, error variants, recipes for adding new ops
  and new arch profiles, M4a limitations.
- `docs/language_reference/uir.md` cross-links to the arm64 guide for the
  optional-attribute interpretation.
- `PROJECT_SPEC.md` milestones table M4 row updated; "Architecture Profiles"
  table loses `generic` row, gains `arm64` row as M4 deliverable.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-03-m4a-arm64-codegen-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-03-m4a-arm64-codegen.md` (12 tasks, 13
commits — Task 1 split into restructure + cleanup-of-stale-`.gitkeep`).

### Project principle formalised in M4a spec §11

> **Dependency policy.** Production crates (`compiler`, `nflc`,
> `profiles/arm64` lib-target) — strict **std-only**. Adding a non-std
> production dep requires a separate explicit decision and PR.
> **Dev-dependencies** are admissible by need; M4a starts the list with
> `libloading` (used only in `profiles/arm64`'s integration test).

### Plan-bug discovered + fixed during execution
- The plan's NFL test strings used `"model M [b=2]: x: Tensor[b, 3]\n    ..."`
  (no `\n` after `:`). The parser requires `\n` after the model header
  before any body statement. Task 2 implementer caught this and fixed the
  test string to `"model M [b=2]:\n    x: Tensor[b, 3]\n    ..."`. Pattern
  propagated to Tasks 3-6 prompts. Behaviour under test unchanged.

### Problems encountered
- The empty `profiles/{generic,x86_64,riscv64}/` placeholder dirs each had
  a `.gitkeep` marker that `rmdir` couldn't remove. Solved with `git rm`
  on each `.gitkeep` (which also removes the now-empty dir).
- Two stale `.gitkeep` files (`profiles/.gitkeep`, `profiles/arm64/.gitkeep`)
  were caught by Task 1's reviewer; cleaned in commit `a317772` before
  proceeding.
- No FP divergence flake — `1e-5` tolerance was sufficient first try.

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
6. `ShapeNotConcrete` reused for "no inputs" case in walk_model — semantically
   different from "shape unresolved". Add dedicated variant in M4b cleanup.

### Next step
**Milestone 4a complete.** First time NeuralForge produces real
machine-executable code: an `.s` text file → `.dylib` → callable function
that gives numerically correct output (matmul+relu of f32 inputs).

The immediate next step is **Milestone 4b — softmax + bias + dropout**:
- Lower `linear[N, bias=true]` (4-th `bias` parameter, `FnSig.weights_layout`).
- Lower `dropout` (no-op pass-through at inference).
- Lower `softmax` (scalar `exp` via Taylor series with range reduction OR
  link `expf` from libm).
- Result: all 5 M3 positive fixtures lower end-to-end.
- Move duplicate-model-name check up to `compiler::ir::build`.

Brainstorming for M4b runs in a fresh worktree once main is updated
post-M4a-merge.

---

## 2026-05-03 — Milestone 3c closed: UIR polish — Display impls + source-snippets + reference doc + clippy clean

### What was done
- Added `Display` impls for all UIR types (`Uir`, `UirModel`, `Node`, `Shape`,
  `OpAttr`, `AttrValue`) and for `StdOp`. Output content matches M3b's `print_uir`
  exactly apart from lowercase op names.
- Removed `print_uir`, `print_uir_node`, `format_uir_shape`, `format_uir_attr`
  free functions from `compiler/src/main.rs` (~50 lines deleted; replaced by one
  `print!("{}", uir)` line).
- Added `render_error_with_snippet` helper in `main.rs` (~20 lines, hand-rolled
  std-only). Routes all CLI error paths through it (parse, build, --tokens).
  Output mirrors rustc/cargo conventions (`error:` line, `--> file:line:col`
  pointer, `^` underline).
- Replaced `format!("{:?}", std_op)` with `format!("{}", std_op)` in
  `BuildError::invalid_attr_value`. Error messages now use lowercase op names
  (`dropout.rate` not `Dropout.rate`), matching the NFL source token names.
- Created `docs/language_reference/uir.md` (198 lines): UIR semantics, data
  shape, node kinds, stdlib ops, implicit semantics (incl. multi-pipeline
  convention from M3b open-Q4), CLI inspection format, v0.1 omissions list.
- Cleared all `cargo clippy` warnings: 4× `cloned_ref_to_slice_refs` →
  `std::slice::from_ref` (3 in tests.rs, 1 in build.rs — the build.rs site
  was discovered during the audit and not in the original plan), 1×
  `match_like_matches_macro` → `matches!`. M3a tech-debt #6 closed.
- Audited all enum variants for dead code by briefly enabling
  `#![deny(dead_code)]` at the crate root. Findings logged below.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-02-m3c-uir-polish-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3c-uir-polish.md` (7 tasks, 7 commits).

### Project principle formalised in M3c spec §2

> **Add code only when there's a real consumer.** Do not retain "for-future-use"
> variants/functions/types via `#[allow(dead_code)]`. Remove unused items when
> discovered; re-introduce with the first real use (with tests).

**Nuance:** "no real consumer" means *no caller at all*, not "unreached in current
tests". Defensively reachable code (constructed by guard helpers that protect
against future caller bugs) IS used and should be kept — documented with a
comment explaining the defensive role.

### Audit results
- `ShapeError::WrongInputCount` — KEPT. The audit (with `#![deny(dead_code)]`)
  did NOT flag it: `single_input()` constructs the variant, so it is genuinely
  reachable, not dead. The spec's prescription to add `#[allow(dead_code)]` was
  empirically unnecessary — would have been a no-op. Added a doc comment to the
  variant explaining its defensive role (catches the class of caller bug where a
  multi-input op slips into single-input shape inference; will be exercised for
  real in M5 when `add`/`concat` arrive).
- No other dead-code findings across the entire crate.

### Problems encountered
- The plan expected 4 clippy warnings; running clippy surfaced 5 (the extra one
  was a `cloned_ref_to_slice_refs` in `build.rs:191` at the `infer_output_shape`
  call site — not in tests.rs as the plan assumed). Fixed alongside the other 4.
- No other surprises. Implementation followed the plan closely.

### Known tech debt (carried forward to v0.2 / M4+)
1. M3a tech-debt items #1–#4 still apply (TypeExpr.name, Span start-only, no CI,
   crate version policy). v0.2.
2. AttrError + ShapeError still two enums. Unification is a v0.2 consideration.
3. Multi-error reporting — first-error-halt continues. v0.2.
4. No CI yet. Add as a small follow-up before M4 ships.
5. The `single_input` defensive guard's `WrongInputCount` path becomes
   exercised-for-real in M5 with multi-input ops.

### Next step
**Milestone 3 fully complete.** The UIR pipeline (lex → parse → build → CLI render)
is production-shaped and well-documented.

The immediate next milestone is **Milestone 4 — generic profile (scalar assembly
codegen)**: implement the first architecture profile that consumes the UIR and
emits scalar assembly for any POSIX target. This is the first time the project
produces actual machine-executable output. The first M4 decision is the assembly
flavour (AT&T x86-64 syntax for `as`, NASM, or LLVM textual IR as a stepping
stone) — to be resolved via a fresh `superpowers:brainstorming` cycle for M4.

---

## 2026-05-02 — Milestone 3b closed: UIR extended to all 5 fixtures + dropout validation + --uir CLI

### What was done
- Refactored `build_op` to take `&Shape` instead of `&[Node]` — eliminated the
  `Vec<Node>` clone in `build_model` (closes M3a tech-debt #5)
- Added `stdlib::validate_attrs` + `AttrError` (`OutOfRange`, `MissingAttr`); validates
  per-op value constraints (currently: dropout rate must be in [0, 1])
- Added `BuildErrorKind::InvalidAttrValue { op, attr, reason }` and wired
  `validate_attrs` into `build_op` between `resolve_args` and `infer_output_shape`
- Added `nflc parse <file> --uir` CLI flag with a compact textual UIR pretty-printer
  using `nN`-style node-id notation (matches what the M7 viewer will use)
- **Fix-up commit `7ad99f6`:** extended `resolve_args` to pre-resolve `Symbol(name)`
  args against `model_params` so `linear[output]` (where `output=10` is a param) builds
  to `linear[10]`. M3a missed this gap; classifier.nfl exposed it during Task 4 e2e.
- Restructured `compiler/tests/uir_tiny_mlp.rs` → `compiler/tests/uir_fixtures.rs`
  with submodules per fixture (`tiny_mlp`, `classifier`, `pipeline_styles`,
  `comments`, `mixed_args`, `negative`)
- 4 new positive integration tests cover the remaining M1 fixtures end-to-end
- New negative fixture `tests/fixtures/negative/dropout_rate_out_of_range.nfl`
  + integration test asserting `InvalidAttrValue` at line 6
- 102 tests passing (81 unit + 12 M2 integration + 9 M3 integration); zero warnings

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3b-uir-all-fixtures-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3b-uir-all-fixtures.md` (8 tasks, 9 commits with
the unplanned Symbol-resolution fix-up).

### Problems encountered
- **Plan defect found during Task 4 e2e verification.** The plan author (me) only
  considered M3a's symbolic-dim resolution as covering the params lookup; missed that
  positional Symbol args (e.g. `linear[output]` where `output` is a param) needed the
  same resolution. Fix-up commit `7ad99f6` extends `resolve_args` to pre-resolve
  `Symbol(name)` args against `model_params` HashMap. Caught by the implementer's
  diligent e2e check on classifier.nfl, not by unit tests (which used integer-only
  positionals). Two new unit tests added (`resolve_args_symbol_resolves_against_params`,
  `resolve_args_symbol_not_in_params_stays_symbol`).

### Known tech debt (carried forward — see spec §9)
1. **M3a tech-debt items #1-#4 still apply** (TypeExpr.name, Span start-only, no CI,
   crate version policy). M3b doesn't address them.
2. **AttrError and ShapeError are two separate enums in stdlib.** If the pattern
   grows, M3c can consider unifying into a single OpError enum.
3. **`--uir` printer lives in main.rs as free-function logic.** M3c moves it onto
   the UIR types as Display impls so libraries (test snapshot tools, IDE plugins,
   the M7 viewer) can consume it.
4. **Multi-pipeline behaviour in v0.1:** documented here that grammar permits
   multiple `pipeline_stmt`s but only the last's output becomes the model output.
   M3c will document this explicitly in `docs/language_reference/uir.md`.
5. **`format!("{:?}", std_op)` in the InvalidAttrValue message** uses Debug to
   render `StdOp` as `"Dropout"`. Good enough for v0.1; M3c may add `Display for StdOp`.
6. **Symbol-resolution placement** — currently in `resolve_args` as a pre-pass.
   Consider folding into a unified semantic-resolution pass when more symbol kinds
   appear (v0.2 may add other symbolic identifiers beyond model_params).

### Next step
Begin **Milestone 3c — UIR polish.** Adds: (1) viewer-friendly `Display` impls for
all UIR types (move `print_uir` from `main.rs` onto the types); (2) Ariadne-style
source-snippet error rendering; (3) `docs/language_reference/uir.md` documenting UIR
semantics including the multi-pipeline convention; (4) cleanup of clippy lints noted
in M3a tech-debt #6; (5) audit of unused enum variants. After M3c, Milestone 3 is
fully closed and we can begin **Milestone 4 — generic profile (scalar assembly
codegen)**.

---

## 2026-05-02 — Milestone 3a closed: UIR vertical-slice 1 shipped (tiny_mlp end-to-end)

### What was done
- Created `compiler/src/ir/` module with `mod`, `types`, `stdlib`, `build`, `error`,
  `tests` files (6 source files)
- Implemented index-based DAG (`Uir { models }`, `UirModel { nodes: Vec<Node> }`,
  `NodeId = usize`) per spec §5.1
- Defined stdlib for 4 operations (`Linear`, `Relu`, `Dropout`, `Softmax`) with per-op
  `signature()` and `infer_output_shape()` — all four reachable from `nflc::ir::*`
- Implemented `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>` covering
  symbolic-dim resolution, op binding, positional/named arg validation, and per-op
  shape inference
- Added integration test for `tests/fixtures/tiny_mlp.nfl` plus 3 negative inline tests
  (`UnknownOp`, `UnknownDim`, `ModelHasNoPipeline`)
- Re-exported `Uir`, `UirModel`, `Node`, `NodeId`, `NodeKind`, `OpAttr`, `AttrValue`,
  `Type`, `Shape`, `StdOp`, `BuildError`, `BuildErrorKind` from the crate root
- 88 tests passing (72 unit + 12 M2 integration + 4 M3a integration); zero warnings

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3a-uir-tiny-mlp-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3a-uir-tiny-mlp.md` (10 tasks, 10 commits).

### Problems encountered
- **Borrow-checker workaround in `build_model`.** Rust forbids passing both `&nodes`
  (read-only context for shape lookup in `build_op`) and `&mut nodes` (where `build_op`
  pushes the new node) simultaneously. Resolved by cloning a `Vec<Node>` snapshot
  before each `build_op` call. Cheap for tiny_mlp's ≤3 nodes; proper refactor is
  M3b's job (see tech-debt below).
- **`AttrValue::Symbol` is genuinely unused in M3a's tests** — only `bias=true` (in
  `mixed_args.nfl`, M3b territory) ever produces it. Caught and tracked in spec §9.1
  before implementation; no surprises in execution.

### Known tech debt (carried forward — see spec §9 plus this session's findings)
1. **`AttrValue::Symbol(String)` is unused in M3a tests.** Will be exercised in M3b
   when `mixed_args.nfl` is built. No `#[allow(dead_code)]` needed because the variant
   is reachable through the `pub use` chain at the crate root.
2. **`OpAttr.name` for positional args reuses `ArgSlot.name` from the signature.**
   Couples consumers to the slot-name string contract. No action in M3a.
3. **`Shape(Vec<u64>)` allocates per shape.** Acceptable for v0.1; revisit if
   profiling shows it matters.
4. **`Type.name` is always `"Tensor"` in v0.1.** Same tech-debt category as M2's
   `TypeExpr.name`. Becomes an `enum TypeKind` in v0.2.
5. **`build_model` clones `Vec<Node>` once per `build_op` call** to satisfy the
   borrow checker. Cheap for M3a's small graphs (≤3 nodes per model). M3b should
   refactor `build_op` to take `&Shape` instead of `&[Node]`, eliminating the clone.
6. **A few `cargo clippy` lints** are present but not blocking (the plan's bar is
   warning-free `cargo build`). Specifically: `&[input.clone()]` in stdlib tests
   triggers `cloned_ref_to_slice_refs`, and `match`-as-bool in `check_arg_type`
   triggers `match_like_matches_macro`. M3c can clean these up alongside the other
   polish items.

### Next step
Begin **Milestone 3b — extend UIR to all 5 fixtures.** Adds: multi-pipeline within a
single model, multi-model files (`pipeline_styles.nfl`), named args in real fixtures
(`dropout[rate=0.2]` from `classifier.nfl`, `linear[16, bias=true]` from
`mixed_args.nfl`), Float and Symbol AttrValue exercised by integration tests,
dropout-rate range validation, plus the `--uir` CLI flag for end-to-end inspection.
The data model and stdlib enum from M3a should not need extension; this is purely
incremental wiring + tests + the borrow-checker refactor mentioned in tech-debt #5.

---

## 2026-05-02 — Milestone 2 closed: NFL Parser prototype shipped (Rust, std-only)

### What was done
- Bootstrapped Cargo workspace at the repo root with member crate `nflc` (`compiler/`); std-only, edition 2021
- Implemented hand-written lexer (`compiler/src/lexer/`):
  - `tokens.rs` — `Token`, `TokenKind`, `LexError`
  - `mod.rs` — `lex(&str) -> Result<Vec<Token>, LexError>` with line-by-line scanning
  - `indent.rs` — `IndentStack` emitting virtual `Indent`/`Dedent` tokens
  - Comments, LF/CRLF newlines, pipeline-continuation rule (grammar §5.2), tab rejection
  - 26 unit tests
- Implemented hand-written recursive-descent parser (`compiler/src/parser/`):
  - One `parse_*` function per EBNF production: `parse_arg_value`, `parse_named_arg`,
    `parse_op_args`, `parse_operation`, `parse_pipeline_stmt`, `parse_dim`, `parse_dim_list`,
    `parse_type_expr`, `parse_variable_decl`, `parse_named_value`, `parse_model_params`,
    `parse_model_stmt`, `parse_model_body`, `parse_model_def`, `parse_nfl_source`
  - 24 unit tests
- Defined typed AST (`compiler/src/ast.rs`) with `Span` on every node
- Implemented `nflc parse <file>` CLI with `--tokens` flag for token-stream debug
- Library entry: `nflc::parse(&str) -> Result<NflSource, ParseError>` (lex + parse)
- Added 7 negative fixtures under `tests/fixtures/negative/`: tabs_in_indent,
  missing_colon, unclosed_bracket, empty_tensor, empty_op_args,
  named_before_positional, bad_dedent
- Integration tests (`compiler/tests/fixtures.rs`): 5 positive + 7 negative — all green
- Removed legacy empty `compiler/{lexer,parser,ir,passes}/` and `compiler/.gitkeep` —
  Rust convention is `compiler/src/<module>/`, the legacy stubs are no longer needed

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m2-parser-prototype-design.md` during brainstorming.
This session executed the plan in `docs/superpowers/plans/2026-05-02-m2-parser-prototype.md`
(20 tasks, 22 commits).

### Problems encountered
- **Plan defect found during Task 16 e2e verification.** `parse_pipeline_stmt` did not
  tolerate `Newline` between a step and the leading `->` of a continuation line, even
  though the lexer correctly suppressed `Indent`/`Dedent` for such lines. Symptom:
  classifier.nfl, pipeline_styles.nfl, mixed_args.nfl all failed to parse end-to-end
  while their unit tests (which used inline-only pipelines) passed. Fix: tolerate one
  `Newline` before each continuation `Arrow` in the parser loop. Committed as `dbb57b1`.
- **Same fix bundle:** `parse_model_body` did not tolerate blank/comment-only `Newline`
  between the model-header `:` `Newline` and the first content line's `Indent`. Symptom:
  comments.nfl failed (its first body line is a comment). Fix: `skip_newlines()` before
  `consume(Indent)`.
- **`unused_mut` ratchet during Task 4.** The plan's literal lex code had `let mut line`
  but never mutated it (newlines arrived in Task 5). Implementer removed `mut` to keep
  zero-warnings; restored it in Task 5 when newline handling landed. Cosmetic, no
  functional impact.
- **`#![allow(dead_code)]` was needed on `parser/mod.rs` until Task 15** wired
  `nflc::parse(&str)` to the `pub(crate)` `parse_*` chain. The plan's "remove on Task 10"
  was wrong — the `cargo build` (lib only, without tests) flagged the chain as unused
  until the public entry point existed. Task 15 removed the directive cleanly.

### Known tech debt (carried forward — see spec §9)
1. **`TypeExpr.name: String`** is fixed to `"Tensor"` for v0.1. When v0.2 introduces
   additional types this becomes either an `enum TypeKind` or a `String` validated by
   the semantic pass. Revisit at start of v0.2 grammar work.
2. **`Span` is start-only.** End-position is omitted in v0.1; add it when the first
   consumer (likely the M7 viewer) demands a full source range.
3. **No CI.** `cargo test` is run manually. Open a small follow-up PR to add a
   GitHub Actions workflow on stable Rust before M3 ships.
4. **Crate version `0.1.0` policy undecided.** Standard semver applies, but bump
   policy for the v0.x series should be agreed before v1.0.
5. **Lexer error formatting:** `LexError::UnknownChar { ch: b as char }` mis-renders
   non-ASCII bytes (e.g. UTF-8 sequences appear as Latin-1 fragments). Cosmetic;
   addresses when error reporting matures (v0.2 / Ariadne-style).
6. **`5.` and `.5` produce `UnknownChar` instead of `BadNumber`.** Spec §5.1 mentions
   `BadNumber` for these forms; current implementation rejects them via a different
   path. Acceptable for v0.1; clean up in v0.2.

### Next step
Begin **Milestone 3 — UIR prototype**: build the Universal IR (computation DAG) from
the AST. The 5 positive fixtures from this milestone parse cleanly and the AST types
are stable. The first M3 decision is the UIR's data shape (DAG node-and-edge
representation, sharing strategy, shape-inference timing) — to be resolved via a
fresh `superpowers:brainstorming` cycle for M3.

---

## 2026-05-02 — Milestone 1 closed: NFL Grammar v0.1 shipped

### What was done
- Wrote `language/grammar.ebnf` (formal ISO/IEC 14977 grammar, inference-only, 24 productions)
- Wrote `docs/language_reference/grammar.md` (human-readable reference, 9 sections, line-by-line walkthrough of `tests/fixtures/classifier.nfl`)
- Added 5 positive fixtures under `tests/fixtures/`: `classifier.nfl`, `tiny_mlp.nfl`,
  `pipeline_styles.nfl`, `comments.nfl`, `mixed_args.nfl`
- Verified all artefacts by manual review: reachability of every production from `nfl_source`,
  reference-doc coverage of every production, hand-trace of every fixture against the grammar

### Decisions made
None new. All design decisions for M1 were captured during brainstorming on 2026-05-02 (entry below)
and recorded in `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`. This session
executed the plan in `docs/superpowers/plans/2026-05-02-nfl-grammar-v0.1.md`.

### Problems encountered
- Verification pass found that the root production `nfl_source` was not named anywhere in
  the reference doc (every other production was covered). Fixed by adding a one-sentence
  mention in §1 Overview.
- A self-noted "spec discrepancy" (six vs seven `pipeline_step`s in a walkthrough) turned
  out to be a false alarm — the spec did not contain that walkthrough; it lives only in
  the reference doc, where the count was already correct.

### Next step
Begin **Milestone 2 — Parser prototype**: implement a parser that consumes `.nfl` files and
produces a typed AST. The 5 fixtures from this milestone become the initial test corpus.
The choice of implementation language (Rust / C++ / Python / …) is the first decision of
M2 — to be resolved via a fresh `superpowers:brainstorming` cycle for M2.

---

## 2026-05-02 — Brainstorming Milestone 1 (NFL Grammar v0.1); loss deferred to v0.2

### What was done
- Started brainstorming session for Milestone 1 using `superpowers:brainstorming` skill
- Confirmed scope (Milestone 1 only — formal EBNF grammar)
- Confirmed coverage baseline (the README example, modulo decisions below)
- Confirmed block structure (Python-style: significant indent, `:` opens, dedent closes)
- Resolved a loss-syntax ambiguity (see Decisions); updated `README.md` and `PROJECT_SPEC.md`

### Decisions made

**v0.1 grammar is inference-only; loss syntax deferred to v0.2.**
The original README example included `-> loss: CrossEntropy` as a pipeline terminator. This
made `->` ambiguous: in every other position it means "transform data through op", but in the
loss form it means "terminate the pipeline and bind a training loss". For a language whose
explicit goal is to be LLM-friendly, that dual meaning is a hazard.
Three alternatives were considered: (α) keep the form but mark it as a terminal production
in the grammar; (β) split `loss: TypeName` out as its own statement parallel to `x: Tensor[…]`;
(γ) treat `loss[CrossEntropy]` as a regular operation. The chosen option is to remove all
training syntax from v0.1 entirely — `->` retains a single meaning, the v0.1 spec stays
small, and a coherent training-syntax design (loss + optimiser + training loop hints) can
be done together in v0.2 instead of bolting on a special case now.

**Milestone 1 produces three artefacts, not just the grammar.**
Approach B was selected: `language/grammar.ebnf` (formal, ISO/IEC 14977) + `docs/language_reference/grammar.md`
(human-readable, with examples) + `tests/fixtures/*.nfl` (canonical valid programs).
Writing the reference doc forces ambiguities in the EBNF to surface; the fixtures become the
golden corpus the M2 parser will be tested against. No parser tooling is committed to at
this stage — fixtures are reviewed by hand for now.

**Block structure: Python-style with 4-space indent; tabs forbidden.**
Matches the README example aesthetic and is token-efficient. Tabs are rejected up front to
avoid the recurring tabs-vs-spaces ambiguity that bites LLM-generated code.

### Problems encountered
- None blocking. The loss-syntax ambiguity was caught and resolved during brainstorming,
  before any grammar was written.

### Next step
Finish the brainstorming design (grammar outline, fixtures plan, acceptance criteria),
write the spec to `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`,
then transition to `superpowers:writing-plans` to produce the implementation plan.

---

## 2026-05-02 — Project founded; architecture designed; initial files created

### What was done
- Conceived the NeuralForge project concept (NFL language + AOT compiler to assembly)
- Designed the full architecture: NFL → UIR → Architecture Profile → Assembly
- Created `PROJECT_SPEC.md` with complete design specification
- Created `CLAUDE.md` with context and workflow instructions for Claude Code + Superpowers
- Created `DEVLOG.md` (this file) and `README.md` for project onboarding
- Set up full directory structure:
  `compiler/`, `profiles/`, `language/`, `viewer/`, `tests/`, `docs/`

### Decisions made

**Language name: NeuralForge (NFL)**
Chosen for its directness — a forge that shapes neural networks.

**AOT compilation to assembly only**
No runtime, no interpreter, no JIT. The device receives a compiled binary.
Rationale: eliminates all framework overhead; suitable for edge devices.

**Universal IR (UIR) as the central abstraction**
All architecture-specific logic lives in profiles, not the language or core compiler.
Rationale: adding a new hardware target requires only a new profile.

**AI-native syntax design**
NFL is co-designed for LLM authoring — explicit shapes, left-to-right pipelines,
no ambiguity. Dual representation: compact for authoring, expanded for tooling.

**Human-readable viewer as a first-class component**
Every IR node must have a viewer rendering. AI-generated code must always be
inspectable by a human.

**Kernel fusion by default**
The compiler must attempt to fuse consecutive operations.
Rationale: memory bandwidth is the bottleneck in neural network inference.

**Initial target profiles: x86-64, arm64, riscv64, generic (scalar fallback)**
Chosen for maximum coverage of current hardware landscape.

**Documentation protocol**
Every session must produce a DEVLOG.md entry. Decisions must be logged with reasoning.

### Problems encountered
- None yet. This was a pure design session.

### Next step
Define the NFL grammar formally using EBNF notation (`language/grammar.ebnf`).
Start with the minimal subset needed for a simple feedforward network:
model declaration, tensor types, and the pipeline operator `->`.

---

*Add new entries above this line.*
