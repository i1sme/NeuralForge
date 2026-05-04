# NeuralForge

[![CI](https://github.com/i1sme/NeuralForge/actions/workflows/ci.yml/badge.svg)](https://github.com/i1sme/NeuralForge/actions/workflows/ci.yml)

A domain-specific language and ahead-of-time compiler for neural networks.
Write your network in NFL (NeuralForge Language), compile it to assembly, load it directly onto any device.

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
| `compiler/` | Lexer, parser, IR, optimisation passes |
| `profiles/` | Architecture-specific code generators (x86-64, arm64, riscv64, generic) |
| `language/` | NFL grammar (EBNF) and standard library of operations |
| `viewer/` | Human-readable renderer for compiler output |
| `tests/` | Unit tests, integration tests, and NFL fixture files |
| `docs/` | Language reference and profile writing guide |

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

Early design phase. The architecture is defined; implementation has not started yet.

The next concrete step is defining the NFL grammar formally (`language/grammar.ebnf`).
The v0.1 grammar covers inference (forward pass) only — training syntax (loss, optimizer)
is planned for v0.2.

---

## Core principles

- **Assembly output only** — the device receives a compiled binary, not interpreted code
- **Explicit over implicit** — shapes and types are always declared, never inferred silently
- **Profile isolation** — each hardware target is a self-contained module
- **AI-native syntax** — NFL is designed to be written and read by both humans and LLMs
- **Human oversight** — all compiler output is inspectable via the viewer tool

---

*NeuralForge is an open design. See `PROJECT_SPEC.md` for open questions and future directions.*
