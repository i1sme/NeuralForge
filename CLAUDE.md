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
2. Implement the profile interface (see `profiles/generic/` as reference)
3. Add the profile to the compiler's profile registry
4. Write integration tests using `tests/fixtures/`
5. Document hardware-specific decisions in `docs/profile_guide/`

---

## Current Status

**Milestone 5a complete.** UIR-pass infrastructure shipped: `compiler::passes`
module with `UirPass` trait, `default_pipeline()`, `run_pipeline()`, and
`FuseLinearRelu` — the first fusion pass. Pass turns `Linear (no bias=true,
single consumer) → Relu` into `Linear { fused_post_ops: [Relu] }` with the
Relu node removed and references remapped via fresh NodeIds. Profile/arm64
emits `fmov s4, wzr` once at function-header time and inline `fmax s0, s0, s4`
before store (recovers M4a's in-place relu performance).

CLI `nflc compile` runs `default_pipeline()` between `ir::build` and
`profiles_arm64::lower` by default; `--no-fuse` flag skips it for verification.
Strict stdout/stderr discipline: stdout = asm only (pipeable to `cc`); stderr =
`note:`/`error:` diagnostics. The applied-passes note is emitted only on
pipeline success.

Op coverage unchanged from M4 (linear ± bias, relu, dropout, softmax).
ABI unchanged. Stack allocation, non-leaf prologue, label namespacing — all
unchanged. The `fused_vs_unfused_classifier_match_numerically` integration
test confirms fusion preserves numerics bit-exactly via `assert_eq!` on
all 320 output elements of `classifier.nfl`.

3-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib).
Production code std-only; `libloading` is a test-only dev-dep. **173 tests
passing** across lexer, parser, IR, passes (10 fusion + 4 pipeline-level),
profile codegen (3 fusion + existing M4b), CLI smoke (3), reference-
validation, and FFI integration. `cargo build --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, and
`cargo fmt --all -- --check` are clean. CI green.

The immediate next step is **Milestone 5b — bias-aware fusion +
EliminateDropout pass**: lift the M5a `linear_has_bias` restriction so
`linear[bias=true] → relu` fuses (with bias-add inline before the post-op),
add `EliminateDropout` pass (removes dropout from UIR via the same NodeId-
remap mechanism), introduce `--passes=X,Y` CLI filter syntax. After M5b:
`compiler::passes` has 2 passes; profile guide doc updates land in M5c.

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
