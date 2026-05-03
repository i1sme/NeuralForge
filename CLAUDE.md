# NeuralForge — Claude Context File

This file gives Claude (and the Superpowers plugin) full context about the NeuralForge project.
Read this before writing any code, planning any feature, or running any skill.

---

## What is NeuralForge?

NeuralForge is a domain-specific language (NFL) and ahead-of-time (AOT) compiler stack for
neural networks. It compiles high-level network definitions down to pure assembly for the target
hardware — no runtime, no interpreter, no framework overhead.

**The full stack:**
```
NFL (NeuralForge Language)  ← human / LLM writes this
        ↓
Universal IR (UIR)          ← compiler-internal graph representation
        ↓
Architecture Profile        ← translates UIR to target assembly
        ↓
Assembly binary             ← loaded directly onto device
```

---

## Repository Structure

```
NeuralForge/
├── CLAUDE.md               ← you are here
├── PROJECT_SPEC.md         ← full design specification
│
├── Cargo.toml              ← workspace manifest (members = ["compiler"], more added per milestone)
│
├── compiler/               ← `nflc` crate (Cargo workspace member)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          ← public API: `nflc::parse(&str)` etc.
│   │   ├── main.rs         ← `nflc` CLI binary
│   │   ├── ast.rs          ← typed AST nodes (Span on every node)
│   │   ├── lexer/          ← tokeniser + INDENT/DEDENT machine
│   │   └── parser/         ← recursive-descent parser, one fn per EBNF production
│   └── tests/              ← integration tests (positive + negative fixtures)
│
│   (ir/ and passes/ modules will live under compiler/src/ in M3+)
│
├── profiles/
│   ├── generic/            ← scalar fallback, any POSIX target
│   ├── x86_64/             ← Intel/AMD with AVX-512
│   ├── arm64/              ← Apple M-series, mobile (NEON/SVE/AMX)
│   └── riscv64/            ← RISC-V with RVV
│
├── language/
│   ├── grammar.ebnf        ← formal NFL grammar
│   └── stdlib/             ← built-in operations (linear, conv, attention…)
│
├── viewer/                 ← human-readable renderer for UIR and assembly
│
├── tests/
│   ├── unit/               ← per-module unit tests
│   ├── integration/        ← end-to-end compile-and-run tests
│   └── fixtures/           ← sample .nfl files used in tests
│
└── docs/
    ├── language_reference/ ← NFL syntax reference
    └── profile_guide/      ← how to write a new architecture profile
```

---

## Design Principles (Non-Negotiable)

1. **Explicit over implicit.** Shapes, types, and data flow are always declared. Nothing is inferred
   silently. `Tensor[32, 512]` not `Tensor`.

2. **Assembly output only.** The compiler never emits a higher-level language. The target device
   receives assembled binary — nothing else.

3. **Profile isolation.** Each architecture profile is self-contained. Changes to one profile must
   not affect others. The language and compiler core are hardware-agnostic.

4. **AI-native syntax.** NFL is designed so LLMs can write and read it with minimal token overhead.
   Regular grammar, no exceptions, left-to-right pipeline notation.

5. **Human oversight.** The viewer layer always exists. Any compiler output must be inspectable by
   a human using the viewer tool.

6. **Kernel fusion by default.** The compiler must attempt to fuse consecutive elementwise
   operations. Unfused sequences are a performance bug.

---

## Key Concepts to Understand

### Universal IR (UIR)
A directed acyclic graph (DAG) where:
- Nodes = operations (linear, relu, softmax, loss…)
- Edges = tensors flowing between operations
- Every node carries explicit shape and dtype metadata

### Kernel Fusion
Merging `A → B → C` (three memory round-trips) into a single fused kernel (one round-trip).
This is the single biggest performance win in the compiler.

### Architecture Profile
A module that receives UIR as input and emits assembly as output.
It knows how to map abstract operations (e.g. `matmul[A, B]`) to hardware-specific instructions
(e.g. AVX-512 VNNI intrinsics for x86-64).

---

## Development Workflow

### When implementing a new feature:
1. Write a failing test first (red)
2. Write the minimum code to make it pass (green)
3. Refactor without breaking tests (refactor)
4. Update PROJECT_SPEC.md if the design changed

### When adding a new operation to NFL:
1. Add it to `language/grammar.ebnf`
2. Add a parser rule in `compiler/parser/`
3. Add a UIR node type in `compiler/ir/`
4. Add lowering logic in each relevant profile
5. Add a test fixture in `tests/fixtures/`
6. Add an integration test

### When adding a new architecture profile:
1. Create `profiles/<name>/` directory
2. Implement the profile interface (see `profiles/generic/` as reference)
3. Add the profile to the compiler's profile registry
4. Write integration tests using `tests/fixtures/`
5. Document hardware-specific decisions in `docs/profile_guide/`

---

## Current Status

**Milestone 3 fully complete.** The UIR pipeline is production-shaped:
`nflc::ir::build(&NflSource)` turns parsed AST into a typed Universal IR,
`nflc parse <file> --uir` renders it via `Display` impls, and errors carry
source-snippet pointers with `^` markers (rustc-style). All 5 M1 positive
fixtures build to UIR; the M3b negative fixture correctly fails at the right
stage. 106 tests passing across lexer, parser, IR, and integration. Both
`cargo build` and `cargo clippy --all-targets -- -D warnings` are clean.
`docs/language_reference/uir.md` documents UIR semantics for contributors.

The immediate next step is **Milestone 4 — generic profile**: implement the
first architecture profile that consumes the UIR and emits scalar assembly for
any POSIX target. This is the first time NeuralForge produces real
machine-executable output. The first M4 decision is the assembly flavour
(AT&T `as`, NASM, or LLVM textual IR as a stepping stone) — to be resolved via
a fresh `superpowers:brainstorming` cycle.

---

## What NOT to Do

- Do not add a runtime or interpreter — output is always compiled assembly
- Do not add Python bindings or framework wrappers in v1
- Do not let a profile depend on another profile's internals
- Do not use implicit shape broadcasting — all shapes must be explicit
- Do not skip viewer support — every new IR node must have a viewer rendering

---

## Documentation Protocol (MANDATORY)

After every working session — whether you wrote code, designed something, or just discussed
a decision — you MUST update `DEVLOG.md` before finishing.

### What to log in DEVLOG.md:

```
## YYYY-MM-DD — <one-line summary of what happened>

### What was done
- Bullet list of concrete work completed

### Decisions made
- Each decision + the reasoning behind it
- If a design changed from PROJECT_SPEC.md, note it here and update the spec too

### Problems encountered
- Any blockers, surprises, or unresolved questions

### Next step
- The single most important thing to do next
```

### Rules:
- **Never skip the log.** Even a 5-minute session that only answered a question gets a short entry.
- **Be specific.** "Worked on parser" is bad. "Added rule for pipeline operator `->` in parser/pipeline.py" is good.
- **Log decisions, not just actions.** Future contributors need to know *why*, not just *what*.
- **Update PROJECT_SPEC.md** if any decision changes or extends the original design, then reference that change in the log.
- **Keep CLAUDE.md's "Current Status" section up to date** — it should always reflect where the project actually is.

---

## Asking for Help

If uncertain about a design decision, consult `PROJECT_SPEC.md` first.
If the spec doesn't answer it, add an entry to the "Open Questions" section there before
implementing anything.
