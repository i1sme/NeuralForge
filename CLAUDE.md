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
├── Cargo.toml              ← workspace manifest (members = ["bench", "compiler", "nflc", "profile-api", "profiles/arm64", "profiles/x86_64"])
│
├── bench/                  ← `bench` crate (bin only) — OQ-BENCH harness
│   ├── Cargo.toml
│   ├── src/main.rs         ← single-file harness
│   └── results/            ← committed cross-profile reports
│       └── <YYYY-MM-DD>.md ← lands as a post-merge follow-up commit
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
├── profile-api/            ← shared Profile contract — types + trait, lifted from arm64 in M9
│
├── profiles/
│   ├── arm64/              ← `profiles-arm64` crate (lib only) — first concrete codegen profile
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs      ← `pub fn lower(&Uir) -> Result<Asm, LowerError>`
│   │   │   ├── types.rs    ← Asm, FnSig, ParamSlot, ParamKind, LowerError
│   │   │   ├── asm.rs      ← prologue/epilogue + emit_sp_* + emit_imm32 helpers
│   │   │   ├── abi.rs      ← AbiContext (n_inputs, input_reg/params_reg/output_reg, ffi_save/restore, M12)
│   │   │   ├── buffer.rs   ← BufferLoc, assign_buffers, compute_is_leaf, compute_callee_saved
│   │   │   ├── codegen.rs  ← walk_uir/walk_model dispatcher + classify_op
│   │   │   ├── ops/
│   │   │   │   ├── mod.rs        ← per-op submodule entry + re-exports
│   │   │   │   ├── add.rs        ← emit_add (elementwise tensor add, M13)
│   │   │   │   ├── layernorm.rs  ← emit_layernorm (3-pass mean/var/normalize, optional affine, native fsqrt; M14)
│   │   │   │   ├── linear.rs     ← emit_linear (matmul ± bias) + materialise_ptr
│   │   │   │   ├── matmul.rs     ← emit_matmul (rank ≥ 2, optional transpose_b, M10; scratch rework M12)
│   │   │   │   ├── mulscalar.rs  ← emit_mulscalar (scalar pre-load + flat loop, M10)
│   │   │   │   ├── relu.rs       ← emit_relu (elementwise copy-clamp)
│   │   │   │   ├── softmax.rs    ← emit_softmax (3-pass + bl _expf)
│   │   │   │   └── dropout.rs    ← marker (no emitter — aliasing only)
│   │   │   └── tests.rs    ← unit tests on asm shape + analyzers
│   │   └── tests/
│   │       ├── integration.rs    ← end-to-end FFI tests for all 5 M3 fixtures + M4a + M10 self_attention + M12 multi-input
│   │       └── common/mod.rs     ← cc + tempdir helpers
│   └── x86_64/             ← Linux ELF scalar SSE2 codegen profile, M9
│       └── src/
│           ├── abi.rs      ← AbiContext (SysV AMD64 variant, M12)
│           └── ops/        ← add.rs (M13), layernorm.rs (M14; 3-pass SysV native sqrtss, op-local %r12/%r13 for affine; M15 LH-4 closed — %r15/%rbp scratch), linear.rs, matmul.rs (M10; callee-saved scratch rework M12; %rbp j-counter fix M13), mulscalar.rs (M10), relu.rs, softmax.rs, dropout.rs
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
    └── profile_guide/      ← per-profile docs (arm64.md, x86_64.md)
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

5. **Human oversight.** Every compiler output must be inspectable by a human.
   `nflc parse <file.nfl> --uir` (compact) and `--uir-verbose` (annotated,
   M8+) provide human-readable UIR pretty-printing. New UIR fields and node
   kinds must extend the `Display` impls in `compiler/src/ir/types.rs` so
   this CLI rendering stays complete.

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

> **Before starting brainstorm for any milestone, review `PROJECT_SPEC.md` §"Known Latent Hazards" — if the milestone's fixtures could trigger any entry, resolving it is mandatory scope, not optional.**

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
5. **ABI audit (x86_64):** When adding a new operation emitter OR when a milestone expands input arity, run an ABI audit across all x86_64 emitters in `profiles/x86_64/src/ops/`. For each emitter, verify that no ABI-argument register (from `AbiContext`) appears as a long-lived counter or scratch. Document any violations found as entries in `PROJECT_SPEC.md` §"Known Latent Hazards" before closing the milestone.

### When adding a new operation to NFL:
1. Add it to `language/grammar.ebnf`
2. Add a parser rule in `compiler/parser/`
3. Add a UIR node type in `compiler/ir/`
4. Add lowering logic in each relevant profile
5. Add a test fixture in `tests/fixtures/`
6. Add an integration test

### When adding a new architecture profile:
1. Create `profiles/<name>/` directory
2. Implement the `Profile` trait from `profile-api/` (see `profiles/arm64/` and `profiles/x86_64/` as canonical references: `impl Profile` with `lower(&Uir) -> Result<Asm, LowerError>` and `sym_prefix() -> &'static str`)
3. Add the profile to `nflc compile --profile` dispatch in `nflc/src/main.rs`
4. Write integration tests using `tests/fixtures/`
5. Document hardware-specific decisions in `docs/profile_guide/<name>.md`

---

## Current Status

**Milestone 15 complete. 446 tests passing on macOS arm64 (~448 on Linux x86_64 CI with x86_64 FFI tests included).** All workspace gates clean
(`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check`, `cargo test --workspace`).

M15 closed the A2 third brick — FFN as compositional NFL pattern
(`linear → relu → linear`, no new StdOp variant, no codegen changes) — and
the LH-4 latent hazard cleanup in x86_64 `emit_layernorm` (per-row scratch
`%r8`/`%r9` → `%r15`/`%rbp`). A2 axis fully complete: residual + LayerNorm
+ FFN all shipped on both profiles. Two new positive fixtures: `ffn.nfl`
(N=1 baseline) and `transformer_block.nfl` (N=3 full transformer block,
runtime FFI evidence for LH-4 closure on Linux x86_64 CI). Helper
promotion: `reference_matmul`/`bias_add`/`relu` moved from `integration.rs`
file-local to `common/mod.rs` `pub fn` per profile.

Strategic direction: see `PROJECT_SPEC.md` §"Strategic Roadmap" — A1 closed
M12, A2 first brick (`add`) closed M13, A2 second brick (`layernorm`)
closed M14, A2 third brick (FFN) closed M15. **A2 axis fully complete.**
Next candidates: A3 — profile-level viewer annotations (per-node footprint,
stack frame, callee-saved set); Axis 3 — bare-metal `expf` to drop libm.
Trigger-driven cleanup (OQ-7, OQ-8, OQ-9, M5c OQ-4) stays dormant. §"Known
Latent Hazards" table empty as of end of M15.

---

## What NOT to Do

- Do not add a runtime or interpreter — output is always compiled assembly
- Do not add Python bindings or framework wrappers in v1
- Do not let a profile depend on another profile's internals
- Do not use implicit shape broadcasting — all shapes must be explicit
- Do not skip human-readable rendering — every new IR node, field, or NodeKind
  variant must extend the `Display` impls in `compiler/src/ir/types.rs` so the
  `nflc parse --uir` CLI continues to render the full UIR shape. The dedicated
  viewer tool (M9+) will consume the same `Display`/`VerboseUir` output as a starting point.

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
