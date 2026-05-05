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
├── Cargo.toml              ← workspace manifest (members = ["compiler", "nflc", "profiles/arm64"])
│
├── compiler/               ← `compiler` crate (lib only)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          ← public API: `compiler::parse(&str)`, `compiler::ir::build(&NflSource)`
│   │   ├── ast.rs          ← typed AST nodes (Span on every node)
│   │   ├── lexer/          ← tokeniser + INDENT/DEDENT machine
│   │   ├── parser/         ← recursive-descent parser, one fn per EBNF production
│   │   └── ir/             ← UIR types, builder, stdlib
│   └── tests/              ← integration tests (positive + negative fixtures)
│
├── nflc/                   ← `nflc` crate (bin only) — CLI dispatcher
│   ├── Cargo.toml
│   └── src/main.rs         ← `nflc parse|compile ...`
│
├── profiles/
│   └── arm64/              ← `profiles-arm64` crate (lib only) — first concrete codegen profile
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs      ← `pub fn lower(&Uir) -> Result<Asm, LowerError>`
│       │   ├── types.rs    ← Asm, FnSig, ParamSlot, ParamKind, LowerError
│       │   ├── asm.rs      ← prologue/epilogue + emit_sp_* + emit_imm32 helpers
│       │   ├── buffer.rs   ← BufferLoc, assign_buffers, compute_is_leaf, compute_callee_saved
│       │   ├── codegen.rs  ← walk_uir/walk_model dispatcher + classify_op
│       │   ├── ops/
│       │   │   ├── mod.rs        ← per-op submodule entry + re-exports
│       │   │   ├── linear.rs     ← emit_linear (matmul ± bias) + materialise_ptr
│       │   │   ├── relu.rs       ← emit_relu (elementwise copy-clamp)
│       │   │   ├── softmax.rs    ← emit_softmax (3-pass + bl _expf)
│       │   │   └── dropout.rs    ← marker (no emitter — aliasing only)
│       │   └── tests.rs    ← unit tests on asm shape + analyzers
│       └── tests/
│           ├── integration.rs    ← end-to-end FFI tests for all 5 M3 fixtures + M4a
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
    ├── language_reference/ ← NFL syntax reference (grammar.md, uir.md)
    └── profile_guide/      ← per-profile docs (arm64.md)
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

5. **Human oversight.** Every compiler output must be inspectable by a human. Until
   the dedicated viewer tool ships (M7+), the `nflc parse <file.nfl> --uir` CLI
   provides human-readable UIR pretty-printing via `Display for Uir`, including
   M5a's `fused=[<list>]` suffix for fused operations. New UIR fields and node
   kinds must extend the `Display` impls so this CLI rendering stays complete.

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

### Before any commit (zero-warnings culture):
1. `cargo fmt --all` — keep formatting in canonical form. CI gates on `--check`,
   so drift accumulates into noisy "style:" commits if not done per-session.
2. `cargo clippy --workspace --all-targets -- -D warnings` — must exit 0.
3. `cargo test --workspace` — must pass (test count goes up monotonically).

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
2. Implement the profile interface (see `profiles/arm64/` as the canonical reference: `pub fn lower(&Uir) -> Result<Asm, LowerError>` plus the `Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError` types)
3. Add the profile to the compiler's profile registry
4. Write integration tests using `tests/fixtures/`
5. Document hardware-specific decisions in `docs/profile_guide/`

---

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

Cross-cutting consistency (M5c): all five workspace error types (`BuildError`,
`ParseError`, `LexError`, `PassError`, `LowerError`) implement
`std::error::Error`. `StdOp` and `PostOp` are both `#[non_exhaustive]`. The
profile-side `match op` blocks (in `walk_model`, `classify_op`, and
`assign_buffers`) have wildcard arms routing future ops to
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

---

## What NOT to Do

- Do not add a runtime or interpreter — output is always compiled assembly
- Do not add Python bindings or framework wrappers in v1
- Do not let a profile depend on another profile's internals
- Do not use implicit shape broadcasting — all shapes must be explicit
- Do not skip human-readable rendering — every new IR node, field, or NodeKind
  variant must extend the `Display` impls in `compiler/src/ir/types.rs` so the
  `nflc parse --uir` CLI continues to render the full UIR shape. The dedicated
  viewer tool (M7+) will consume the same `Display` output as a starting point.

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
