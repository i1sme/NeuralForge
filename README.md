# NeuralForge

[![CI](https://github.com/i1sme/NeuralForge/actions/workflows/ci.yml/badge.svg)](https://github.com/i1sme/NeuralForge/actions/workflows/ci.yml)

A domain-specific language and ahead-of-time compiler for neural networks.
Write your network in NFL (NeuralForge Language), compile it to assembly,
load it directly onto any device.

---

## The idea in one paragraph

Most neural network frameworks run on a runtime (Python, CUDA libraries, etc.) which adds significant overhead.
NeuralForge takes a different approach: you describe your network in a high-level language designed for
clarity and AI-native authoring, and the compiler produces pure assembly for your target hardware.
No interpreter. No framework. Just the network, compiled.

---

## How it works

```
You write NFL         →   Compiler produces UIR   →   Profile lowers to assembly   →   Binary runs on device
(high-level DSL)          (hardware-agnostic IR)       (hardware-specific code)
```

A simple network in NFL looks like this:

```nfl
model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu
      -> dropout[rate=0.2]
      -> linear[256] -> relu
      -> linear[output] -> softmax
```

---

## Repository map

| Path | What's in it |
|------|-------------|
| `PROJECT_SPEC.md` | Full design specification — read this first |
| `DEVLOG.md` | Chronological record of all work and decisions |
| `CLAUDE.md` | Context file for Claude Code (AI development assistant) |
| `compiler/` | `compiler` crate — lexer, parser, AST, Universal IR, optimisation passes |
| `inspect-render/` | `inspect-render` crate — formats `profile_api::Inspection` as human-readable text for `nflc inspect` and per-profile golden tests |
| `nflc/` | `nflc` crate — CLI binary (`nflc parse`, `nflc compile`, `nflc inspect`) |
| `profile-api/` | `profile-api` crate — shared `Profile` trait, `Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError` |
| `profiles/arm64/` | `profiles-arm64` crate — AArch64 / Apple Silicon code generator |
| `profiles/x86_64/` | `profiles-x86_64` crate — Linux ELF scalar SSE2 code generator |
| `bench/` | `bench` crate — OQ-BENCH harness (`cargo run -p bench --release -- --profile {arm64\|x86_64}`); per-profile median + p95 µs across `classifier` / `large_classifier_k` / `self_attention` |
| `language/` | NFL grammar (`grammar.ebnf`, frozen at v0.1; v0.2 named-pipeline extension since M10) |
| `tests/fixtures/` | Sample `.nfl` files used in integration tests |
| `docs/` | Language reference (`grammar.md`, `uir.md`) and profile guide (`arm64.md`, `x86_64.md`) |
| `viewer/` | Reserved for a future standalone viewer tool; UIR rendering is via `nflc parse --uir` / `--uir-verbose`; profile-level annotation is now live via `nflc inspect` (M16) |

---

## Where to start

**To understand the project:** read `PROJECT_SPEC.md` — it has the full
architecture, design principles, and open questions.

**To contribute:** read `CONTRIBUTING.md` for the development workflow,
then `DEVLOG.md` to understand what has been done and why. Every significant
design decision is recorded there with its reasoning.

---

## Project status

**Milestone 17 complete** — Axis 3 first leg: softmax's libm `expf`
(`bl _expf` / `call expf@PLT`) is replaced by an **inlined degree-7 Taylor
polynomial** (Cody-Waite range reduction → Horner → `2^z` bit-trick + branchless
underflow clamp) on both profiles, removing NeuralForge's **last runtime
dependency** — the x86_64 `.so` now links without `-lm`. Correctness rests on a
two-layer contract: the emitted asm is bit-exact against a Rust `exp_ref` port,
and `exp_ref` is within **≤ 1 ulp** of libm across the softmax domain. The
now-misnamed predicate `calls_extern_math` was renamed `has_softmax`. Per the
minimal-swap discipline, the FFI save/restore and callee-saved prologue are
retained for now; their removal (softmax leaf-cleanup) is the recorded **M18**
follow-up. The §"Known Latent Hazards" table remains empty.

(Prior milestone — M16: `nflc inspect --profile <arm64|x86_64>` ships
profile-level viewer annotations; Axis 2 fully complete.)

What's working today:

- **NFL v0.1 grammar** (frozen since M1) + **v0.2 named-pipeline extension**
  (since M10) for self-attention-style multi-stage fixtures with declared
  intermediate shapes
- **Lexer, parser, typed AST, Universal IR (UIR)** with optimiser passes
- **Multi-input ABI** (since M12, A1) — models with up to N=4 input tensors
  lower correctly under SysV AMD64 / AAPCS64 calling conventions; all
  per-emitter scratch is non-INPUT_REGS at all supported N
- **Stdlib operations on both profiles:** `linear` (± bias), `relu`,
  `dropout` (no-op pass-through), `softmax` (inline bare-metal exp — degree-7 Taylor, no libm, M17; rank ≥ 2),
  `matmul` (rank ≥ 2, optional `transpose_b`, M10), `mul_scalar` (M10),
  `add` (residual connections, M13), `layernorm` (3-pass mean/var/normalize,
  optional affine, native `fsqrt` / `sqrtss` — no libm dependency, M14)
- **FFN as compositional NFL pattern** — `linear → relu → linear` (M15), no
  new StdOp variant. Demonstrated via `tests/fixtures/ffn.nfl` (N=1) and
  `tests/fixtures/transformer_block.nfl` (N=3 — full pre-LN block:
  `x -> layernorm[affine=true] -> linear -> relu -> linear -> add[skip1] -> add[skip2]`)
- **`profile-api/`** — shared `Profile` trait, `Asm`, `FnSig`, `ParamSlot`,
  `ParamKind`, `LowerError` types; both profiles implement the trait
- **AArch64 scalar code generation** (`profiles/arm64/`): all stdlib ops
  above; AAPCS64-clean register allocation; `fmadd` single-rounding matmul
- **x86_64 Linux ELF scalar SSE2 code generation** (`profiles/x86_64/`):
  full op-parity with arm64; AT&T syntax; SysV AMD64 ABI; ABI-invariant
  unit tests at N=2/3/4 for every emitter; xmm-spill strategy for softmax
  row state (no callee-saved FP registers under SysV)
- **UIR-pass framework** with `EliminateDropout`, `FuseLinearRelu`, and
  `FuseLinearSoftmax`
- **CLI:** `nflc parse` (with `--uir` compact and `--uir-verbose` annotated
  rendering), `nflc compile --profile <arm64|x86_64>`, and `nflc inspect --profile <arm64|x86_64>`
- **Bit-exact FFI integration tests** with `to_bits()` comparison
  (M14 layernorm precedent); per-profile divergent `reference_matmul` /
  `exp_ref` bodies match each emitter's rounding semantics (arm64 `fmadd`
  single-rounding; x86_64 `mulss + addss` two-rounding). M17 adds a two-layer
  numeric contract for the inline `exp`: asm bit-exact vs the Rust `exp_ref`
  port, plus an `exp_ref`-vs-libm ≤ 1 ulp sweep over the softmax domain
- **OQ-BENCH harness** (`bench/` crate, M11) — per-profile median + p95 µs
  across `classifier` / `large_classifier_k` / `self_attention` fixtures;
  CI workflow `.github/workflows/bench.yml` writes per-profile Job
  Summaries on `macos-14` (arm64) and `ubuntu-latest` (x86_64)
- **Viewer v0.1:** `nflc parse --uir-verbose` renders annotated UIR with
  top-level and per-model summaries, and fused post-ops on indented lines
- **472 tests passing on macOS arm64 (~476 on Linux x86_64 CI)**; CI green;
  `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace` all clean

Active development continues along three strategic axes — codegen breadth
(SIMD/AVX vectorisation still open), modelling depth (A2 axis fully closed
in M15; A3 — profile-level viewer annotations closed in M16; Axis 2 fully
complete), and deployment reach (bare-metal inline `expf` — first leg closed in
M17, removing the last libm dependency; M18 softmax leaf-cleanup next) — tracked in
[`PROJECT_SPEC.md` §"Strategic Roadmap"](PROJECT_SPEC.md#strategic-roadmap).

NFL training syntax (loss, optimiser) remains deferred to v0.3.

---

## Build & try

The workspace is pure Rust, std-only at runtime (`libloading` is a
test-only dev-dependency). Build the CLI:

```sh
cargo build --release -p nflc
```

Parse an NFL file and print the AST or UIR:

```sh
cargo run -p nflc -- parse tests/fixtures/classifier.nfl
cargo run -p nflc -- parse tests/fixtures/classifier.nfl --uir
cargo run -p nflc -- parse tests/fixtures/classifier.nfl --uir-verbose
```

Compile to assembly (specify target profile):

```sh
# AArch64 / Apple Silicon (default profile)
cargo run -p nflc -- compile tests/fixtures/classifier.nfl --profile arm64 > out_arm64.s

# x86_64 Linux ELF (M9)
cargo run -p nflc -- compile tests/fixtures/classifier.nfl --profile x86_64 > out_x86_64.s
```

Inspect or filter optimisation passes:

```sh
# skip the entire pipeline (Dropout stays as a buffer alias, no fusion)
cargo run -p nflc -- compile foo.nfl --no-passes

# run only the linear+relu fusion pass
cargo run -p nflc -- compile foo.nfl --passes fuse_linear_relu
```

Run the full test suite:

```sh
cargo test --workspace
```

The arm64 profile targets Apple Silicon and AArch64 POSIX hosts. The x86_64
profile targets Linux ELF. NEON / SVE / AVX vectorisation and a RISC-V profile
are future work.

---

## Core principles

- **Assembly output only** — the device receives a compiled binary, not interpreted code
- **Explicit over implicit** — shapes and types are always declared, never inferred silently
- **Profile isolation** — each hardware target is a self-contained module
- **AI-native syntax** — NFL is designed to be written and read by both humans and LLMs
- **Human oversight** — every compiler output is inspectable; viewer v0.1 ships today via `nflc parse --uir` (compact) and `nflc parse --uir-verbose` (annotated), with profile-level annotation now live via `nflc inspect` (M16), and a fuller standalone viewer tool still reserved for future UI work

---

## License

NeuralForge is licensed under the [Apache License, Version 2.0](LICENSE).

You may use, modify, and distribute this software freely under the
terms of the Apache License. The license includes an explicit patent
grant from contributors (Apache 2.0 §3) — meaningful for an
infrastructure compiler where codegen algorithms may carry patent
claims, where a permissive license like MIT would be silent on
patents.

If you use NeuralForge in a public project, please link back to
[https://github.com/i1sme/NeuralForge](https://github.com/i1sme/NeuralForge).
This is a courtesy request, not a legal requirement beyond what the
Apache License already mandates (preservation of copyright notices
and license text in redistributions).

Copyright (C) 2026 Arsenii Voloshyn.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.
NeuralForge is licensed under Apache 2.0; per §5 of the license,
contributions are implicitly licensed under the same terms — no
separate Contributor License Agreement is required.

---

*NeuralForge is an open design. See `PROJECT_SPEC.md` for open questions and future directions.*
