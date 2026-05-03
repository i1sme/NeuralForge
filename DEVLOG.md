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
