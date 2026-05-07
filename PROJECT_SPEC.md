# NeuralForge — Project Specification

> **Status:** Early design / pre-alpha  
> **Version:** 0.1  
> **Language:** English (code, docs, comments)

---

## Vision

NeuralForge is a domain-specific language (DSL) and ahead-of-time (AOT) compiler stack designed
specifically for building and deploying neural networks. Its core philosophy:

- **Write once, run anywhere** — the language is not tied to any hardware architecture
- **Assembly-level output** — no runtime overhead; the compiled binary is pure assembly for the target
- **AI-native design** — syntax and semantics optimised to be written and read by both humans and LLMs
- **Human oversight** — a viewer layer makes AI-generated or compiler-generated code inspectable by humans
- **Extensible to any hardware** — new architectures can be supported by adding a profile, without changing the language

---

## Core Components

### 1. NeuralForge Language (NFL)

A high-level DSL for describing neural network architectures, training logic, and inference pipelines.

Design goals:
- **Explicit over implicit** — shapes, types, and data flow are always declared, never inferred silently
- **Regular syntax** — one way to express each concept; no historical exceptions
- **Compositional** — operations chain left-to-right in a readable pipeline style
- **Token-efficient** — compact but unambiguous; low cognitive load for LLMs and humans alike
- **Self-documenting** — code carries enough context to be understood without external documentation

Rough sketch of what NFL code looks like:

```nfl
model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu
      -> dropout[rate=0.2]
      -> linear[256] -> relu
      -> linear[output] -> softmax
```

> **Note:** the example above describes the inference (forward) path only. Training syntax —
> loss specification, optimisers — is intentionally deferred to NFL v0.2 to keep the v0.1
> grammar focused on a single concept: a pipeline `->` is always a data transformation,
> never a control statement.

### 2. Universal Intermediate Representation (UIR)

An architecture-agnostic IR that the NFL compiler produces before targeting specific hardware.

Properties:
- Represents the computation graph explicitly (nodes = operations, edges = data flow)
- Carries shape and type information at every node
- Enables cross-operation analysis: kernel fusion, operation reordering, memory layout optimisation

### 3. Architecture Profiles

Profiles translate UIR into assembly for a specific hardware target.

Initial target profiles:
| Profile     | Architecture       | Key capability              |
|-------------|--------------------|-----------------------------|
| `arm64`     | Apple Silicon / AArch64 POSIX | Scalar AArch64 assembly: linear (± bias), relu, dropout (no-op pass-through), softmax (libm `expf`). All 5 M3 fixtures lower end-to-end (M4a + M4b). NEON / SVE / AMX in later slices. |
| `x86_64`    | Intel / AMD (Linux ELF) | Linux ELF scalar SSE2: linear (± bias), relu, dropout, softmax (libm expf via PLT). Full op-parity with arm64 minus SIMD/AVX. macOS x86_64 (Mach-O) and SIMD remain open. |
| `riscv64`   | RISC-V             | RVV vector extension (future) |

A profile is a self-contained module. Adding support for a new architecture means writing a new profile — the language and compiler core remain unchanged.

> Note: M4 was originally specced as a `generic` profile (LLVM IR or similar
> portable IR). During M4 brainstorming this was reframed: "generic" survives as
> the architectural _principle_ (profile isolation, swap-in profiles per target),
> not as a profile name. The first concrete profile is `arm64`, matching the
> host architecture for native execution.

### 4. Compiler Pipeline

```
NFL source
    │
    ▼
Lexer / Parser
    │
    ▼
Typed AST
    │
    ▼
Universal IR (UIR)
    │  ← optimisation passes here:
    │     - kernel fusion
    │     - tiling / blocking
    │     - dead operation elimination
    ▼
Architecture Profile
    │
    ▼
Assembly output (.asm)
    │
    ▼
Assembled binary → loaded directly onto target device
```

### 5. Human-Readable Viewer

A tool (CLI and/or IDE extension) that renders assembly or UIR back into annotated, human-readable form. Intended for:
- Inspecting AI-generated NFL code
- Debugging compiler output
- Understanding what the compiler actually produced

---

## Key Optimisations

### Kernel Fusion
Multiple consecutive operations (e.g. `linear → bias → relu`) are merged into a single assembly pass, eliminating redundant reads and writes to memory.

### Tiling / Cache Blocking
Matrices are split into blocks sized to fit CPU/GPU cache levels, dramatically reducing memory bandwidth pressure.

### Operation Scheduling
The compiler analyses the computation graph and reorders operations to maximise data locality and minimise cache evictions.

---

## AI-Native Design Principles

NeuralForge is designed so that LLMs can write, read, and reason about NFL code efficiently:

1. **Explicit shapes everywhere** — `Tensor[32, 512]` not `Tensor`
2. **Left-to-right pipeline notation** — matches natural reading order
3. **No overloaded symbols** — each operator has one meaning
4. **Canonical form** — the compiler normalises all code to a standard format
5. **Dual representation** — compact form for authoring, expanded form for tooling and AI processing

---

## What NeuralForge Is NOT

- Not a general-purpose programming language
- Not a Python library or framework wrapper
- Not tied to CUDA, ROCm, or any proprietary runtime
- Not a model zoo or pre-trained model format

---

## First Milestones

| # | Milestone                                      | Goal                                                                       |
|---|------------------------------------------------|----------------------------------------------------------------------------|
| 1 | Language spec v0.1                             | Define NFL syntax formally (EBNF grammar) — inference-only; training in v0.2 |
| 2 | Parser prototype                               | Parse a simple feedforward network definition                              |
| 3 | UIR prototype                                  | Produce a computation graph from parsed AST       |
| 4 | `arm64` profile (4a + 4b complete)             | Generate scalar AArch64 assembly for all 5 M3 fixtures end-to-end (linear ± bias, relu, dropout, softmax via libm expf) |
| 5 | Kernel fusion + UIR-pass framework (5a + 5b + 5c complete) | UIR-pass infrastructure (`UirPass` trait, `default_pipeline`, `run_pipeline`, `PassError`); two passes shipped — `FuseLinearRelu` (bias-aware: fuses `linear → relu` and `linear[bias=true] → relu`) and `EliminateDropout` (removes inference-time-noop Dropout); CLI gains `--no-passes` and `--passes <list>` filter; bit-exact equivalence proven via `fused_vs_unfused_*_match_numerically` integration tests on classifier and mixed_args fixtures |
| 6 | Attention-pattern fusion — kernel fusion v2 (complete) | `PostOp::SoftmaxRow` variant + `FuseLinearSoftmax` pass; `default_pipeline = [EliminateDropout, FuseLinearRelu, FuseLinearSoftmax]`; arm64 RowWise emit branch in `emit_linear` (matmul i-loop A writes the M×N output, then a separate i-loop B runs Phases 2-4: row-max → exp+sum → normalise, in-place over the linear output buffer using callee-saved s8/s9 surviving `bl _expf`); bit-exact equivalence proven via `fused_vs_unfused_softmax_match_numerically` on `classifier` (no-bias) + `softmax_with_bias` (bias-aware) fixtures; `compiler/src/ir/test_utils.rs` shared helpers extracted; OQ-5 `assert_eq!` harmonisation across all three fused-vs-unfused FFI tests |
| 7 | Shared 3-step rebuild helper extraction (complete) | New `compiler/src/passes/rewriter.rs` (`pub(crate) struct RewritePlan` + `pub(crate) fn rewrite_model`); plan-as-data API (three HashMaps + one constructor that precomputes `consumer_count`); migration of three existing passes (`EliminateDropout`, `FuseLinearRelu`, `FuseLinearSoftmax`) onto the shared helper, each pass body shrinks ~60% (70-100 → 26-39 lines); closes M6 holistic-review Finding #1 (three-strikes-then-refactor trigger fired in M6, deferred to M7); §8 invariant 6 unit test added (closes M6 Finding #7); atomic-task-pack convention demonstrated via 4 sequential clean commits |
| 8 | ARM64 codegen hardening + viewer v0.1 (complete) | Two arm64 codegen bugs closed: dropout-as-output now emits an explicit copy-loop via new `ops/dropout.rs::emit_dropout_copy` (BufferLoc::OutputReg branch in walk_model::Dropout); dim-immediate encoding routed uniformly through `asm::emit_imm32` across 17 sites (12 cmp + 5 mov), with hoist-outside-loop (Group A: relu, dropout-copy, matmul body) and re-materialise-at-loop-top (Group B: standalone softmax, fused RowWise tail) placement strategies; new fixtures `large_classifier_{k,n}.nfl` (k=8192 / out=5120) prove > 4095 dim now compiles. Viewer v0.1: `compiler::ir::types::{VerboseUir, VerboseModel, VerboseNode}` newtype wrappers + `Uir::calls_extern_math` / `UirModel::calls_extern_math` predicate; new `nflc parse --uir-verbose` flag (mutually exclusive with `--uir`) renders annotated UIR with top-level + per-model summary, fused post-ops on separate indented lines. `docs/language_reference/uir.md` gets new "Viewing UIR" section. Test count: 208 → 223. |

---

## Strategic Roadmap

A dependency graph (not a schedule) of the open strategic axes. Each row shows
what unlocks what; choosing the next milestone means choosing one axis to
advance. Trigger-driven cleanup is intentionally excluded — it activates on
its own trigger condition and lives under "Open Questions" below.

```
x86_64 profile [M9 complete] → MACHO_SYM_PREFIX rename [closed — abstracted as Profile::sym_prefix() in M9]
NFL v0.2 grammar → attention ops → profile-level viewer annotations
bare-metal expf → drop libm dependency
```

- **Axis 1 — codegen breadth.** Adding a second concrete profile (x86_64)
  validates the profile-isolation principle; the per-OS symbol-prefix rename
  falls out as a natural consequence of the work, not as a separately-scheduled
  milestone. M9 ships scalar Linux ELF; SIMD/AVX and macOS x86_64 remain as
  possible follow-ups. `MACHO_SYM_PREFIX rename` closed — abstracted as
  `Profile::sym_prefix()` in M9.
- **Axis 2 — modelling depth.** NFL v0.2 grammar unblocks attention patterns
  (Q/K/V projections, scaled dot-product, axis-N softmax). Profile-level viewer
  annotations (per-node footprint, stack frame, callee-saved set) follow once a
  non-trivial attention model lowers end-to-end, and double as a profile-agnostic
  split validation.
- **Axis 3 — deployment reach.** Replacing the `bl _expf` libm call with a
  Taylor-series `expf` removes the only runtime dependency, unlocking bare-metal
  targets.

---

## Open Questions

### Design questions
- Training syntax design: when and how to introduce loss/optimiser constructs (planned for v0.2)
- How profiles handle quantisation (INT8, FP16, BF16)?
- Distribution format for compiled binaries

### Trigger-driven cleanup
Items raised during a milestone that intentionally do not get scheduled — they
activate when their trigger condition fires.

- **OQ-NEW** — **Closed in M9 (commit `a08fd24`).** `profiles/arm64/src/buffer.rs::node_uses_softmax` was removed; both `compute_is_leaf` and `compute_callee_saved` now consume `UirModel::calls_extern_math()` (UIR-side predicate). All sites reduced to the UIR predicate; no profile-specific information was needed. Single source of truth across profiles.
- **OQ-7** (M7) — per-pass `eliminate_one_model` / `fuse_one_model` return `Result<UirModel, PassError>` despite never producing `Err`. *Trigger: first real `Err`-case in pass-level logic.*
- **OQ-8** (M7) — `compiler/src/passes/rewriter.rs` could lift to `compiler/src/ir/`. *Trigger: a non-pass UIR-rewrite consumer appears.*
- **OQ-9** (M7) — `producer_post_ops: Vec<PostOp>` could generalise to `enum NodeMutation`. *Trigger: a fourth pass needs non-PostOp producer mutation.*
- **M5c OQ-4** — `BuildError::span()` + `Diagnostic` trait for richer error reporting. *Trigger: error-reporting ergonomics become a real pain point in a downstream milestone.*
- **OQ-BENCH** (opened by M9 spec, fires on M9 merge) — Build a benchmark harness that compiles a single NFL source through both `arm64` and `x86_64` profiles, runs both binaries with the same input/params, and reports timing side-by-side. Goal: quantify the cost of "scalar-only" vs the eventual SIMD profile, and lay groundwork for performance claims. *Trigger: M9 merged. Scope: stretch enough to handle multiple fixtures; output a markdown report. No regression-gate yet — informational only.*

## Decisions (formerly open, now resolved)

- **NFL v0.1 grammar** — frozen as of M1 (`language/grammar.ebnf`). Future syntax extensions land in NFL v0.2+ as separate language milestones.
- **Memory model** — static stack-allocated intermediate buffers, no heap. Established by M4b (`profiles/arm64::buffer.rs::assign_buffers`); applies to all v1 profiles.

---

*This document evolves with the project. Update it as decisions are made.*
