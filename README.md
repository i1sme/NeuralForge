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
| `nflc/` | `nflc` crate — CLI binary (`nflc parse`, `nflc compile`) |
| `profile-api/` | `profile-api` crate — shared `Profile` trait, `Asm`, `FnSig`, `ParamSlot`, `ParamKind`, `LowerError` |
| `profiles/arm64/` | `profiles-arm64` crate — AArch64 / Apple Silicon code generator |
| `profiles/x86_64/` | `profiles-x86_64` crate — Linux ELF scalar SSE2 code generator (M9) |
| `language/` | NFL grammar (`grammar.ebnf`, frozen at v0.1) |
| `tests/fixtures/` | Sample `.nfl` files used in integration tests |
| `docs/` | Language reference (`grammar.md`, `uir.md`) and profile guide (`arm64.md`, `x86_64.md`) |
| `viewer/` | Reserved for a future standalone viewer tool; rendering today is via `nflc parse --uir` (compact) and `nflc parse --uir-verbose` (annotated) |

---

## Where to start

**To understand the project:** read `PROJECT_SPEC.md` — it has the full
architecture, design principles, and open questions.

**To contribute:** read `CONTRIBUTING.md` for the development workflow,
then `DEVLOG.md` to understand what has been done and why. Every significant
design decision is recorded there with its reasoning.

---

## Project status

**Milestone 9 complete** — second concrete profile (`x86_64` Linux ELF, scalar)
ships. A single NFL source now compiles to two distinct binaries via
`nflc compile --profile arm64` and `nflc compile --profile x86_64`; the
profile-isolation hypothesis is validated. Full op-parity with arm64 minus SIMD.

What's working today:

- Lexer, parser, typed AST, Universal IR (UIR)
- **`profile-api/`** — shared `Profile` trait, `Asm`, `FnSig`, `ParamSlot`,
  `ParamKind`, `LowerError` types; both profiles implement the trait
- **AArch64 scalar code generation** (`profiles/arm64/`): `linear` (with or
  without bias), `relu`, `dropout`, `softmax` (libm `expf`); large-dimension
  immediates routed uniformly through `emit_imm32`
- **x86_64 Linux ELF scalar code generation** (`profiles/x86_64/`): full
  op-parity with arm64; AT&T syntax; `call expf@PLT`; SysV AMD64 ABI;
  xmm-spill strategy for `row_max`/`row_sum` across `call expf@PLT` (no
  callee-saved FP registers under SysV)
- UIR-pass framework with three passes shipped — `EliminateDropout`,
  `FuseLinearRelu`, and `FuseLinearSoftmax`
- CLI: `nflc parse` (with `--uir` compact and `--uir-verbose` annotated
  rendering) and `nflc compile --profile <arm64|x86_64>`
- Bit-exact fused-vs-unfused FFI integration tests across all
  fusion-eligible fixtures; x86_64 FFI tests run on ubuntu-latest CI
- Viewer v0.1: `nflc parse --uir-verbose` renders annotated UIR with
  top-level and per-model summaries, and fused post-ops on indented lines
- 284 tests passing on macOS arm64 (~300 on Linux x86_64 CI); CI green;
  `cargo fmt`, `cargo clippy -D warnings`, `cargo test --workspace` all clean

Active development continues along three strategic axes — codegen breadth
(x86_64 ships in M9; SIMD/AVX still open), modelling depth (NFL v0.2 /
attention), and deployment reach (bare-metal `expf`) — tracked in
[`PROJECT_SPEC.md` §"Strategic Roadmap"](PROJECT_SPEC.md#strategic-roadmap).

NFL training syntax (loss, optimiser) is deferred to v0.2.

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
- **Human oversight** — every compiler output is inspectable; viewer v0.1 ships today via `nflc parse --uir` (compact) and `nflc parse --uir-verbose` (annotated), with a dedicated standalone viewer tool reserved for future profile-level annotation work

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
