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
