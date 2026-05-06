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
| `profiles/arm64/` | `profiles-arm64` crate — AArch64 / Apple Silicon code generator |
| `language/` | NFL grammar (`grammar.ebnf`, frozen at v0.1) |
| `tests/fixtures/` | Sample `.nfl` files used in integration tests |
| `docs/` | Language reference (`grammar.md`, `uir.md`) and profile guide (`arm64.md`) |
| `viewer/` | Reserved for a future standalone viewer tool; rendering today is via `nflc parse --uir` (compact) and `nflc parse --uir-verbose` (annotated) |

---

## Where to start

**To understand the project:**
1. Read `PROJECT_SPEC.md` — it has the full architecture, design principles, and milestones
2. Read `DEVLOG.md` — it tells you what has been done and what comes next

**To contribute or continue development:**
1. Read `CLAUDE.md` — it explains the development workflow and non-negotiable rules
2. Check `DEVLOG.md` for the latest "Next step" entry
3. Follow the TDD workflow: red → green → refactor

**To understand a design decision:**
Look it up in `DEVLOG.md`. Every significant decision is recorded there with its reasoning.

---

## Project status

**Milestone 8 fully closed.** The compiler stack is end-to-end working for
inference-only NFL v0.1, with a complete arm64 profile, a multi-pass
kernel-fusion pipeline that includes the attention-shape `linear → softmax`
pattern, hardened large-dimension codegen, and a v0.1 UIR viewer.

What's working today:

- Lexer, parser, typed AST, Universal IR (UIR)
- AArch64 scalar code generation: `linear` (with or without bias), `relu`,
  `dropout`, `softmax` (libm `expf`); large-dimension immediates routed
  uniformly through a single emit helper so dims above the 12-bit cmp
  range now compile cleanly
- UIR-pass framework with three passes shipped — `EliminateDropout`
  (removes inference-time-noop Dropout), `FuseLinearRelu` (bias-aware
  fusion of `linear → relu`), and `FuseLinearSoftmax` (attention-pattern
  fusion of `linear → softmax` into a row-wise emit branch)
- CLI: `nflc parse` (with `--uir` compact and `--uir-verbose` annotated
  rendering) and `nflc compile` (with `--no-passes` and `--passes <list>`
  filters)
- Bit-exact fused-vs-unfused FFI integration tests across all
  fusion-eligible fixtures
- Viewer v0.1: `nflc parse --uir-verbose` renders annotated UIR with
  top-level and per-model summaries, and fused post-ops on indented lines
- 223 tests passing across the workspace; CI green; `cargo fmt`,
  `cargo clippy -D warnings`, `cargo test --workspace` all clean

Next: scope for **Milestone 9** is decided by selecting one of three open
axes — codegen breadth, modelling depth, or deployment reach — described
in [`PROJECT_SPEC.md` §"Strategic Roadmap"](PROJECT_SPEC.md#strategic-roadmap).
The chosen axis seeds a fresh brainstorming round rather than picking from
a flat list of follow-ups.

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

Compile to AArch64 assembly:

```sh
cargo run -p nflc -- compile tests/fixtures/classifier.nfl > out.s
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

The arm64 profile targets Apple Silicon and AArch64 POSIX hosts. NEON / SVE
vectorisation, an x86_64 profile, and a RISC-V profile are future work.

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
