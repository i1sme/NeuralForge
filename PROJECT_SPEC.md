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
| `arm64`     | Apple Silicon / AArch64 POSIX | Scalar AArch64 assembly: linear (± bias), relu, dropout (no-op pass-through), softmax (libm `expf`, rank ≥ 2 since M10), matmul (rank ≥ 2, optional `transpose_b`, since M10), mul_scalar (since M10). All 5 M3 fixtures + the M10 self_attention fixture lower end-to-end. NEON / SVE / AMX in later slices. |
| `x86_64`    | Intel / AMD (Linux ELF) | Linux ELF scalar SSE2: linear (± bias), relu, dropout, softmax (libm expf via PLT, rank ≥ 2 since M10), matmul (rank ≥ 2, optional `transpose_b`, since M10 — `mulss + addss`, no FMA), mul_scalar (since M10). Full op-parity with arm64 minus SIMD/AVX. macOS x86_64 (Mach-O) and SIMD remain open. |
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
| 9 | x86_64 Linux ELF profile + profile-api contract (complete) | x86_64 scalar SSE2 codegen with full op-parity with arm64 (linear ± bias, relu, fused relu, softmax_row, dropout alias). `Profile` trait in new `profile-api` crate abstracts the profile contract (`lower` + `sym_prefix`); both `profiles-arm64` and `profiles-x86_64` implement it. SysV AMD64 ABI compliance: prologue/epilogue saves callee-saved registers conditionally (`%rbp` always; `%rbx/%r12–%r15` when `calls_extern_math()`); matmul body uses only caller-saved registers (`%rdi/%rsi/%rdx` as k-counter/scratch/bias-base). `docs/profile_guide/x86_64.md` added. Test count: 223 → 284. |
| 10 | NFL v0.2 self-attention + 4D codegen (complete) | NFL grammar v0.2 — new `named_pipeline_stmt = identifier , ":" , type_expr , "=" , identifier , pipeline_chain` production with one-token lookahead disambiguation from `variable_decl`. UIR: `ArgType::Tensor` + `resolve_args` cascade through `build_op` / `build_model`. Two new stdlib ops: `StdOp::Matmul` (rank ≥ 2 inputs, optional `transpose_b`, four new `ShapeError` variants) and `StdOp::MulScalar` (per-element scalar multiply, shape-preserving). New `BuildErrorKind::DeclaredShapeMismatch` for the named-pipeline declared-vs-inferred shape check. `Softmax` rank tightened to ≥ 2 (any-rank, last-axis). Both profiles ship `emit_matmul` (outer-loop wrapper over `leading_count`; arm64 FMA inner triple-loop, x86_64 `mulss + addss` — intentional ISA divergence) + `emit_mulscalar` (scalar pre-load + flat in-place loop) + softmax dispatch generalised to `b = product(shape[..-1]), k = shape[-1]`. arm64's `emit_softmax` spills `x0/x1/x2` via `stp/ldp` around `bl _expf` and `emit_matmul` spills `x1/x2` via `stp/ldp` around the outer loop; x86_64's `emit_softmax` spills `%rdi/%rsi/%rdx` via `pushq`/`popq` and `emit_matmul` spills `%rdi/%rsi/%rdx` via `movq` to/from `%xmm6/%xmm7/%xmm8`. End-to-end self-attention fixture compiles + runs bit-exact per-profile via FFI on both profiles. Test count: 284 → 331. |
| 11 | OQ-BENCH harness — close the M9-merge trigger (complete) | New `bench/` workspace crate (`cargo run -p bench --release -- --profile {arm64|x86_64} --format {markdown|github-summary} [--seed N]`) compiling 3 fixtures (`classifier`, `large_classifier_k`, `self_attention`) through host-native profile, timing 10 warmup + 100 measurement FFI calls, reporting median + p95 µs to stdout / Job Summary. New `.github/workflows/bench.yml` with 2-leg matrix (`macos-14` arm64 + `ubuntu-latest` x86_64); each leg writes to `$GITHUB_STEP_SUMMARY` (no artifact sharing, no aggregator). Cross-profile combined report composed manually post-CI to `bench/results/<date>.md`. Test count: 331 → 344. |
| 12 | NFL multi-input ABI (A1 — first leg of Axis 2 follow-up) (complete) | Multi-input function ABI (N=1..4) via per-profile `AbiContext` connector. New `profiles/{arm64,x86_64}/src/abi.rs` carrying `n_inputs: usize`, arity-aware `input_reg/params_reg/output_reg/ffi_save_set/materialise_ptr` accessors, alignment-correct `emit_ffi_save/emit_ffi_restore` (xzr / pushq %rax padding for odd cardinality, strict LIFO restore). `BufferLoc::InputReg(usize)` carries input index. `walk_model` constructs AbiContext once and threads `&abi` through every op-emitter; arity > 4 returns `LowerError::TooManyInputs`. `emit_matmul` rework on both profiles per spec §9.1: per-iter slice pointers move off ABI registers (arm64: x12/x13/x14 in place of x1/x2/x4; x86_64: callee-saved %rbx/%r12-%r15 via extended `compute_callee_saved` trigger), eliminating M10 outer-loop spill blocks. New fixtures: `two_input_matmul.nfl` (N=2 sanity), `multi_input_attention.nfl` (N=3 acceptance — V consumed post-softmax), `tests/fixtures/profile-negative/too_many_inputs.nfl` (N=5 → LowerError). Bench `bench/src/main.rs` gains per-arity dispatch + seed cascade. Test count: 344 → 390. **Known follow-up:** x86_64 `emit_matmul` currently rejects N=4 (j-counter %r9 collides with output_reg); rework deferred to M13+. |
| 13 | N=4 + matmul fix + `add` op (A2 first brick) (complete) | x86_64 `emit_matmul` j-counter relocated from `%r9` to `%rbp` (callee-saved by unconditional prologue `pushq %rbp`; read by zero op-emitter bodies). Closes M12 known follow-up: N=4 + matmul now compiles and runs bit-exact. New `StdOp::Add` (flat variant; no BinaryOp container). NFL surface `a -> add[skip]` — first real consumer of M10's `ArgType::Tensor` outside Matmul. Strict shape equality (no broadcasting); new `ShapeError::AddShapeMismatch`. Both profiles ship `emit_add` (flat elementwise loop, modeled after `emit_mulscalar`); x86_64 reuses Task 1's `%rbp` scratch trick as loop counter. **Pre-Task-5 fix:** arm64 `emit_linear` ABI register clobber for N≥2 closed via stp/ldp save/restore of x3 (and x4 at N≥3, x5 at N≥4) around the i-loop body — same class of bug as Task 1 but resolved differently because emit_linear's bias paths and fused PostOp::SoftmaxRow dispatch already saturate x9-x16. Three new fixtures: `residual_add.nfl` (positive both profiles), `four_input_matmul.nfl` (closes Group A end-to-end x86_64), `negative/add_shape_mismatch.nfl` (IR-level reject). Test count: 390 → 400. |
| 14 | A2 second brick — LayerNorm + LH-1/2/3 cleanup (complete) | LH-1/2/3 cleanup in x86_64 `emit_linear` (commit `916e9c7`): j-counter `%rcx` → `%rbp` (LH-1); src-ptr scratch `%r8` → op-local `pushq %r14` (LH-2); weight-ptr scratch `%r9` → op-local `pushq %r15` (LH-3). New `StdOp::LayerNorm` — single StdOp variant with internal 3-pass codegen (mean → variance + inv_std → normalize + optional affine). Native `fsqrt` on arm64, `sqrtss` on x86_64; no libm dependency. Affine toggle `layernorm[affine=true]` mirrors `linear[bias=true]`. `ParamKind` extended: `LayerNormScale` (γ) before `LayerNormBias` (β) — contract. arm64: leaf function, s0–s7 scratch only (s8–s15 avoided), `s_b` reuses `s2` after `s_inv_d` consumption. x86_64: op-local `pushq %r12/%r13` when affine; `compute_callee_saved` unchanged. LH-4 logged (N=3..4 %r8/%r9 reuse in x86_64 emit_layernorm, deferred). New fixtures: `layernorm_no_affine.nfl`, `layernorm_affine.nfl`, `pre_ln_block.nfl` (N=2, validates LH-1 closure end-to-end), `negative/layernorm_rank_too_low.nfl`. Test count: 400 → 441. |
| 15 | A2 third brick — FFN compositional + LH-4 cleanup (complete) | LH-4 cleanup in x86_64 `emit_layernorm` (commit `e35dfaa`): per-row src ptr `%r8` → `%r15` (op-local pushq/popq); per-row dst ptr `%r9` → `%rbp` (function-level prologue handles). Push counts no-affine 2→3, affine 4→5. `OP_LOCAL_PUSH_BYTES_*` constants updated. 3 ABI-invariant unit tests `emit_layernorm_n{2,3,4}_does_not_clobber_output_reg`. A2 third brick: FFN as compositional NFL pattern (`linear → relu → linear`) — no new StdOp variant, no codegen changes. New fixtures `ffn.nfl` (N=1) and `transformer_block.nfl` (N=3 — exercises LH-4 condition output_reg=%r8 and validates closure via FFI on Linux x86_64 CI). Helper promotion: `reference_matmul/bias_add/relu` moved from `integration.rs` file-local to `common/mod.rs` `pub fn` per profile (isolation principle). Per-profile divergent `reference_matmul` body (arm64 `fmadd` / x86_64 `mulss+addss`). 4 new FFI integration tests with bit-exact `to_bits()` comparison. ABI audit at N=3,4: all emitters clean. Test count: 441 → 446 (macOS arm64); ~448 on Linux x86_64 CI. |

---

## Current Status

**Milestone 15 complete. 446 tests passing on macOS arm64 (~448 on Linux x86_64 CI with x86_64 FFI tests included).** All workspace gates clean (`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check`, `cargo test --workspace`).

M14 closed the A2 second brick (LayerNorm) end-to-end on both profiles and the LH-1/2/3 latent hazard cleanup in x86_64 `emit_linear` (opener commit `916e9c7`). LayerNorm is a single StdOp variant with internal 3-pass codegen (mean → variance + inv_std → normalize + optional affine), modeled structurally after Softmax. Native `fsqrt`/`sqrtss` — no libm dependency added. Affine optionality via single Symbol toggle `layernorm[affine=true]`, mirroring `linear[bias=true]`. AAPCS64-safe register allocation on arm64 (s8–s15 callee-saved range intentionally avoided; `s_b` reuses `s2` after `s_inv_d` consumption to stay within s0–s7). Op-local `%r12`/`%r13` push/pop on x86_64 affine path — `compute_callee_saved` unchanged. **M15 closed LH-4 (per-row `%r8`/`%r9` scratch in x86_64 `emit_layernorm`) — relocated to `%r15` (op-local pushq/popq) and `%rbp` (function-level prologue handles). Runtime FFI evidence via new `transformer_block.nfl` fixture (N=3, output_reg=%r8) on Linux x86_64 CI.**

Strategic direction: see §"Strategic Roadmap" — A1 closed in M12, A2 first brick (`add`) closed in M13, A2 second brick (`layernorm`) closed in M14, **A2 third brick (FFN) closed in M15 — A2 axis fully complete**. Trigger-driven cleanup items (OQ-7, OQ-8, OQ-9, M5c OQ-4) live in §"Open Questions" / "Trigger-driven cleanup" and stay dormant. OQ-NEW closed in M9 (commit `a08fd24`). OQ-BENCH closed in M11 (commit `e7c29b8`).

---

## Strategic Roadmap

A dependency graph (not a schedule) of the open strategic axes. Each row shows
what unlocks what; choosing the next milestone means choosing one axis to
advance. Trigger-driven cleanup is intentionally excluded — it activates on
its own trigger condition and lives under "Open Questions" below.

```
x86_64 profile [M9 complete] → MACHO_SYM_PREFIX rename [closed — abstracted as Profile::sym_prefix() in M9]
NFL v0.2 self-attention [complete in M10] → multi-input grammar A1 [closed M12] → transformer block A2 (residual + LayerNorm + FFN) → profile-level viewer annotations A3
bare-metal expf → drop libm dependency
```

- **Axis 1 — codegen breadth.** Adding a second concrete profile (x86_64)
  validates the profile-isolation principle; the per-OS symbol-prefix rename
  falls out as a natural consequence of the work, not as a separately-scheduled
  milestone. M9 ships scalar Linux ELF; SIMD/AVX and macOS x86_64 remain as
  possible follow-ups. `MACHO_SYM_PREFIX rename` closed — abstracted as
  `Profile::sym_prefix()` in M9.
- **Axis 2 — modelling depth.** M10 closed the first leg (NFL v0.2 self-attention).
  M12 closed A1 (multi-input ABI). M13 closed the M12→M13 priority signal
  (x86_64 N=4 + matmul gap via `%rbp` j-counter relocation) and shipped
  the first A2 brick (`StdOp::Add`, residual connections). M14 closed the
  A2 second brick (`StdOp::LayerNorm`) — single StdOp variant with internal
  3-pass codegen (mean → variance + inv_std → normalize + optional affine),
  mirroring Softmax-as-one-node. Native sqrt (`fsqrt` / `sqrtss`) — no libm
  dependency added. **M15 closed the A2 third brick — FFN as compositional
  NFL pattern (`linear → relu → linear`) — no new StdOp variant, no codegen
  changes. Demonstrated via `ffn.nfl` (N=1 baseline) and `transformer_block.nfl`
  (N=3, full transformer block with LayerNorm + FFN + dual residual). M15
  also closed LH-4 in x86_64 `emit_layernorm` (per-row scratch `%r8`/`%r9`
  → `%r15`/`%rbp`).** A2 axis is now complete (residual + LayerNorm + FFN
  all shipped on both profiles). Open follow-ups: A3 — profile-level viewer
  annotations (per-node footprint, stack frame, callee-saved set); A2-extended
  — training syntax (loss/optimiser) for NFL v0.3.
- **Axis 3 — deployment reach.** Replacing the `bl _expf` libm call with a
  Taylor-series `expf` removes the only runtime dependency, unlocking bare-metal
  targets.

---

## Open Questions

### Design questions
- Training syntax design: when and how to introduce loss/optimiser constructs (planned for v0.2)
- How profiles handle quantisation (INT8, FP16, BF16)?
- Distribution format for compiled binaries

### Known Latent Hazards

Bugs that exist in the codebase but are not triggered by any current fixture.
Each entry must be resolved in the milestone whose fixture first exercises it.
Leaving an entry here longer than one milestone is a process failure.

| # | Location | Condition | Symptom | Opened |
|---|----------|-----------|---------|--------|

*(Table is empty as of M15 — all latent hazards closed.)*

### Trigger-driven cleanup
Items raised during a milestone that intentionally do not get scheduled — they
activate when their trigger condition fires.

- **OQ-NEW** — **Closed in M9 (commit `a08fd24`).** `profiles/arm64/src/buffer.rs::node_uses_softmax` was removed; both `compute_is_leaf` and `compute_callee_saved` now consume `UirModel::calls_extern_math()` (UIR-side predicate). All sites reduced to the UIR predicate; no profile-specific information was needed. Single source of truth across profiles.
- **OQ-7** (M7) — per-pass `eliminate_one_model` / `fuse_one_model` return `Result<UirModel, PassError>` despite never producing `Err`. *Trigger: first real `Err`-case in pass-level logic.*
- **OQ-8** (M7) — `compiler/src/passes/rewriter.rs` could lift to `compiler/src/ir/`. *Trigger: a non-pass UIR-rewrite consumer appears.*
- **OQ-9** (M7) — `producer_post_ops: Vec<PostOp>` could generalise to `enum NodeMutation`. *Trigger: a fourth pass needs non-PostOp producer mutation.*
- **M5c OQ-4** — `BuildError::span()` + `Diagnostic` trait for richer error reporting. *Trigger: error-reporting ergonomics become a real pain point in a downstream milestone.*
- **OQ-BENCH** — **Closed in M11 (commit `e7c29b8`).** Bench harness shipped as the `bench/` workspace crate; CI workflow `.github/workflows/bench.yml` writes per-profile Job Summaries on `macos-14` (arm64) and `ubuntu-latest` (x86_64). Cross-profile comparison is composed manually into `bench/results/<date>.md` after each run (no aggregator job — preserves M10 §11.2 rule). Three fixtures (`classifier` / `large_classifier_k` / `self_attention`) chosen for orthogonal signals (matmul-mass / large-K inner-loop accumulator / expf-dominated dispatch overhead). Methodology: 10 warmup + 100 measurement, median + p95.

## Decisions (formerly open, now resolved)

- **NFL v0.1 grammar** — frozen as of M1 (`language/grammar.ebnf`). Future syntax extensions land in NFL v0.2+ as separate language milestones.
- **Memory model** — static stack-allocated intermediate buffers, no heap. Established by M4b (`profiles/arm64::buffer.rs::assign_buffers`); applies to all v1 profiles.

---

*This document evolves with the project. Update it as decisions are made.*
