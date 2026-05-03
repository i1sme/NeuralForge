# NFL UIR Prototype — Vertical Slice 1 (Milestone 3a) — Design Spec

> **Status:** Approved (brainstorming output, 2026-05-02)
> **Authoritative for:** Milestone 3a implementation
> **Source skill:** `superpowers:brainstorming`
> **Next skill:** `superpowers:writing-plans`
> **Builds on:** [M2 parser spec](./2026-05-02-m2-parser-prototype-design.md) and the AST it defines.
> **Followed by:** M3b (extend to all 5 fixtures), M3c (polish + viewer-friendly Display).

---

## 1. Context

Milestone 2 produced a working parser: `nflc::parse(&str) -> Result<NflSource, ParseError>`.
The output is a typed AST that mirrors the EBNF grammar (`language/grammar.ebnf`).

**Milestone 3** turns the AST into the **Universal IR (UIR)** — a typed computation
graph that subsequent milestones consume:
- M4 (`generic` profile) walks the UIR and emits scalar assembly.
- M5 (kernel fusion pass) traverses and rewrites the UIR.
- M7 (viewer) renders the UIR back to readable text.

Per the brainstorming, M3 is split into **three vertical slices** so each ships working
software:

| | Scope | Demos |
|---|---|---|
| **M3a (this spec)** | End-to-end AST→UIR for `tiny_mlp.nfl` | Library + tests build a UIR for one canonical fixture |
| **M3b (next)** | Extend to all 5 M1 fixtures (symbolic dims at scale, named args, multi-pipeline, multi-model) | All M1 fixtures build to UIR |
| **M3c (then)** | Polish: viewer-friendly `Display` impls, error-message quality, additional negative tests, `--uir` CLI flag | UIR is human-inspectable |

This document covers **M3a only**. M3b and M3c get their own brainstorming/spec/plan
cycles when M3a closes.

**Reading order for context:**
1. `CLAUDE.md` — project rules, especially "When implementing a new feature" (TDD)
2. `language/grammar.ebnf` — the source grammar this UIR mirrors
3. `compiler/src/ast.rs` — the AST that is M3's input
4. `tests/fixtures/tiny_mlp.nfl` — the only fixture M3a must build to UIR
5. `docs/superpowers/specs/2026-05-02-m2-parser-prototype-design.md` — for AST shape and conventions

---

## 2. Scope

### In scope (Milestone 3a)

- New module `compiler/src/ir/` with `mod.rs`, `types.rs`, `stdlib.rs`, `build.rs`,
  `error.rs`, `tests.rs`
- Index-based DAG data shape (`Uir { models: Vec<UirModel> }`,
  `UirModel { nodes: Vec<Node>, ... }`, `NodeId = usize`)
- Stdlib via `enum StdOp { Linear, Relu, Dropout, Softmax }` with `signature()` and
  `infer_output_shape()` per op (for M3a we exercise `Linear` and `Softmax` only;
  `Relu` and `Dropout` are defined but not exercised — their tests live in M3b)
- AST→UIR builder (`nflc::ir::build(&NflSource) -> Result<Uir, BuildError>`) covering:
  - Symbolic dim resolution against `model_params` (e.g., `Tensor[batch, 4]` with
    `model X [batch=8]:` resolves to `Shape(vec![8, 4])`)
  - Op-name binding via `stdlib::resolve` (`linear` → `StdOp::Linear`, etc.)
  - Positional arg validation against op signature (count, type)
  - Per-op shape inference via `infer_output_shape`
  - Model output = "value of the last operation of the **last** `pipeline_stmt` in the
    body" (tracked explicitly via `last_pipeline_output: Option<NodeId>`, **not** as
    `nodes.len() - 1`)
- `BuildError` with `BuildErrorKind` covering all error paths reachable from M3a
  (`UnknownOp`, `UnknownDim`, `UnknownVariable`, `ArgCountMismatch`,
  `ArgTypeMismatch`, `MissingRequiredArg`, `UnexpectedNamedArg`, `ShapeMismatch`,
  `ModelHasNoPipeline`)
- ~15 unit tests + 1 integration test (`tiny_mlp_builds`) + a few inline negative tests
  (`UnknownOp`, `UnknownDim`, `ModelHasNoPipeline`)
- DEVLOG entry, CLAUDE.md "Current Status" updated

### Out of scope — deferred to **M3b**

- Multi-pipeline (more than one `pipeline_stmt` in a single model body)
- Multi-model files (`pipeline_styles.nfl` has 3 models)
- Named args in the live tests (e.g. `dropout[rate=0.2]`) — `resolve_args` already
  supports them in M3a; M3b adds tests
- Float / Symbol `AttrValue` exercised in real fixtures
- Dropout-rate range validation (`0..=1`)
- `nflc parse <file> --uir` CLI flag

### Out of scope — deferred to **M3c**

- Viewer-friendly `Display` impls for UIR nodes (CLAUDE.md "viewer support" requirement)
- Polished, source-snippet-style error messages (Ariadne-style)
- Additional negative test fixtures specific to UIR build phase
- Documentation in `docs/language_reference/` describing UIR semantics

### Out of scope — deferred further

- UIR mutation / fusion API (Milestone 5)
- Codegen consumption (Milestone 4)
- Multi-error reporting (continues from M2's first-error-halt policy until M2.5/v0.2)

---

## 3. Deliverables

| Path | Purpose |
|---|---|
| `compiler/src/ir/mod.rs` | Module root; declares submodules; re-exports public items; defines `pub fn build` |
| `compiler/src/ir/types.rs` | `Uir`, `UirModel`, `Node`, `NodeId`, `NodeKind`, `OpAttr`, `AttrValue`, `Type`, `Shape` |
| `compiler/src/ir/stdlib.rs` | `StdOp`, `Signature`, `ArgSlot`, `ArgType`, `resolve()`, `signature()`, `infer_output_shape()`, `ShapeError` |
| `compiler/src/ir/build.rs` | `build_model`, `build_op`, `resolve_type`, `resolve_args` (private helpers) |
| `compiler/src/ir/error.rs` | `BuildError`, `BuildErrorKind`, `Display` impl |
| `compiler/src/ir/tests.rs` | `#[cfg(test)]` unit tests for the helpers and stdlib |
| `compiler/tests/uir_tiny_mlp.rs` | Integration test: AST→UIR end-to-end on `tiny_mlp.nfl` plus a few inline negative cases |

**Modify:**

| Path | Change |
|---|---|
| `compiler/src/lib.rs` | Add `pub mod ir;` and re-exports `pub use ir::{Uir, UirModel, Node, NodeId, NodeKind, OpAttr, AttrValue, Type, Shape, StdOp, BuildError, BuildErrorKind};`. No new top-level `nflc::build` function in M3a — callers use `ir::build(&ast)`. |
| `DEVLOG.md` | Append M3a close-out entry at the top |
| `CLAUDE.md` | Update "Current Status" → "M3a complete; M3b next (extend UIR to all 5 fixtures)" |

**Do NOT touch in M3a:**
- `compiler/src/main.rs` — `--uir` CLI flag belongs to M3b
- `compiler/src/{ast,lexer,parser}/...` — frozen public surface from M2
- M2 spec, plan, fixtures, integration tests — frozen

---

## 4. Architecture

### Module layout

```
compiler/src/ir/                         (NEW)
├── mod.rs                               public API + module decls
├── types.rs                             core data structures (no logic)
├── stdlib.rs                            StdOp + signature + shape inference
├── build.rs                             AST→UIR builder + helpers
├── error.rs                             BuildError + Display
└── tests.rs                             unit tests
```

The `ir` name (rather than `uir`) matches `CLAUDE.md`'s original "Repository Structure"
diagram. The type is still `Uir` to match the project's stable terminology
("Universal IR" in `PROJECT_SPEC.md`).

### Crate dependencies

Still **zero external dependencies**. M3a uses `std::collections::HashMap` only. Same
hand-written, std-only philosophy as M1/M2.

### Visibility

- `pub` on the module root, the public type names, and `build`
- `pub(super)` on internal helpers (`resolve_type`, `resolve_args`, `build_op`,
  `build_model`)
- Private fields where lifetime/refactor risk is low (e.g. `Shape(pub Vec<u64>)` keeps
  the field public for direct access in tests; this can be tightened in v0.2)

---

## 5. Components

### 5.1 UIR types (`compiler/src/ir/types.rs`)

```rust
//! UIR data types — index-based DAG of typed nodes.

use crate::ast::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Uir {
    pub models: Vec<UirModel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UirModel {
    pub name: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<NodeId>,
    pub output: NodeId,
    pub source_span: Span,
}

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    pub ty: Type,
    pub source_span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input { name: String },
    Op { op: super::stdlib::StdOp, operands: Vec<NodeId>, attrs: Vec<OpAttr> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpAttr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Integer(u64),
    Float(f64),
    Symbol(String),                        // see Open Q1
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub name: String,                      // always "Tensor" in v0.1
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn rank(&self) -> usize { self.0.len() }
}
```

### 5.2 Stdlib (`compiler/src/ir/stdlib.rs`)

```rust
//! Standard library of NFL operations.

use super::types::{AttrValue, OpAttr, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp { Linear, Relu, Dropout, Softmax }

pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear"  => Some(StdOp::Linear),
        "relu"    => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
        _ => None,
    }
}

pub struct Signature {
    pub positional: &'static [ArgSlot],
    pub named: &'static [ArgSlot],
}

pub struct ArgSlot {
    pub name: &'static str,
    pub ty: ArgType,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType { Integer, Float, Symbol }

pub fn signature(op: StdOp) -> Signature {
    use ArgType::*;
    match op {
        StdOp::Linear  => Signature {
            positional: &[ArgSlot { name: "out_dim", ty: Integer, required: true }],
            named:      &[ArgSlot { name: "bias",    ty: Symbol,  required: false }],
        },
        StdOp::Relu    => Signature { positional: &[], named: &[] },
        StdOp::Dropout => Signature {
            positional: &[],
            named:      &[ArgSlot { name: "rate", ty: Float, required: true }],
        },
        StdOp::Softmax => Signature { positional: &[], named: &[] },
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    WrongInputCount { expected: usize, actual: usize },
    WrongRank { expected: usize, actual: usize, dim_index: Option<usize> },
    MissingAttr { name: &'static str },
}

pub fn infer_output_shape(
    op: StdOp,
    inputs: &[Shape],
    attrs: &[OpAttr],
) -> Result<Shape, ShapeError> {
    match op {
        StdOp::Linear => {
            let input = single_input(inputs)?;
            require_rank(input, 2)?;
            let out_dim = get_int_attr(attrs, "out_dim")?;
            Ok(Shape(vec![input.0[0], out_dim]))
        }
        StdOp::Relu | StdOp::Softmax | StdOp::Dropout => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
    }
}

fn single_input(inputs: &[Shape]) -> Result<&Shape, ShapeError> {
    if inputs.len() != 1 {
        Err(ShapeError::WrongInputCount { expected: 1, actual: inputs.len() })
    } else { Ok(&inputs[0]) }
}

fn require_rank(s: &Shape, expected: usize) -> Result<(), ShapeError> {
    if s.rank() == expected { Ok(()) }
    else { Err(ShapeError::WrongRank { expected, actual: s.rank(), dim_index: None }) }
}

fn get_int_attr(attrs: &[OpAttr], name: &'static str) -> Result<u64, ShapeError> {
    attrs.iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value { AttrValue::Integer(n) => Some(n), _ => None })
        .ok_or(ShapeError::MissingAttr { name })
}
```

### 5.3 Builder (`compiler/src/ir/build.rs`)

Public API: `pub fn build(ast: &NflSource) -> Result<Uir, BuildError>` is exposed via
`mod.rs`; the implementation lives here in `build_model` + helpers.

Algorithm per model:

1. **Index params:** `params: HashMap<&str, u64>` from `ast_model.params`.
2. **Walk body, accumulating `nodes: Vec<Node>` and `env: HashMap<String, NodeId>`:**
   - `VariableDecl`: resolve `TypeExpr` → `Shape` (substituting symbolic dims via
     `params`); push `Node { kind: NodeKind::Input { name }, ty: Type { ..., shape } }`;
     record `env[name] = nodes.len() - 1`; record `inputs.push(id)`.
   - `Pipeline`:
     - Look up `p.source` in `env` (`UnknownVariable` if absent).
     - For each `op_ast` in `p.steps`:
       - Resolve op name via `stdlib::resolve` (`UnknownOp` if absent).
       - `resolve_args` → `Vec<OpAttr>` (validates positional/named count & type).
       - `infer_output_shape` from current input shape → output `Shape` (`ShapeMismatch`
         on `ShapeError`).
       - Push `Node { kind: NodeKind::Op { ... }, ty: Type { ..., shape: out } }`.
     - After all steps, record `last_pipeline_output = Some(<last node id>)`.
3. **Determine model output:** `output = last_pipeline_output.ok_or(ModelHasNoPipeline)?`.
4. Return `UirModel { name, nodes, inputs, output, source_span }`.

### 5.4 `BuildError` (`compiler/src/ir/error.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct BuildError {
    pub message: String,
    pub line: u32,
    pub col: u32,
    pub kind: BuildErrorKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BuildErrorKind {
    UnknownOp { name: String },
    UnknownDim { name: String },
    UnknownVariable { name: String },
    ArgCountMismatch { expected: usize, actual: usize },
    ArgTypeMismatch { slot: String, expected: String, actual: String },
    MissingRequiredArg { slot: String },
    UnexpectedNamedArg { name: String },
    ShapeMismatch { detail: String },
    ModelHasNoPipeline { name: String },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for BuildError {}
```

Constructors (`BuildError::unknown_op(&name, span)`, etc.) compose `message` from
`kind` and attach `(line, col)` from the AST's `Span`.

---

## 6. Testing strategy

### 6.1 Unit tests (`compiler/src/ir/tests.rs`)

| Helper / target | Tests |
|---|---|
| `resolve_type` | (a) all-integer dims, (b) symbolic dim with valid lookup, (c) symbolic dim with invalid lookup → `UnknownDim` |
| `resolve_args` | (a) positional only matches, (b) missing required positional → `ArgCountMismatch` or `MissingRequiredArg`, (c) extra positional → `ArgCountMismatch`, (d) type mismatch → `ArgTypeMismatch`, (e) named-only matches |
| `stdlib::resolve` | (a) each known op resolves, (b) unknown name → `None` |
| `stdlib::signature` | each op returns expected `Signature` |
| `stdlib::infer_output_shape` | (a) Linear with rank-2 input + `out_dim` → correct shape, (b) Linear with rank-1 input → `WrongRank`, (c) Relu/Softmax/Dropout pass shape unchanged |
| `BuildError` | each variant constructs and Display-renders human-readably |

Target: ~15 tests.

### 6.2 Integration test (`compiler/tests/uir_tiny_mlp.rs`)

```rust
//! End-to-end: lex + parse + build → assert UIR shape for tiny_mlp.

use nflc::*;
use nflc::ir;

#[test]
fn tiny_mlp_builds() {
    let src = std::fs::read_to_string("../tests/fixtures/tiny_mlp.nfl")
        .expect("fixture readable");
    let ast = parse(&src).expect("parses");
    let uir = ir::build(&ast).expect("builds");

    assert_eq!(uir.models.len(), 1);
    let m = &uir.models[0];
    assert_eq!(m.name, "TinyMLP");

    // 3 nodes total: input x, op linear, op softmax.
    assert_eq!(m.nodes.len(), 3);
    assert_eq!(m.inputs, vec![0]);
    assert_eq!(m.output, 2);

    // Node 0: Input "x", Tensor[8, 4]
    assert!(matches!(&m.nodes[0].kind, NodeKind::Input { name } if name == "x"));
    assert_eq!(m.nodes[0].ty.shape.0, vec![8, 4]);

    // Node 1: Linear[2], operands=[0], shape Tensor[8, 2]
    let NodeKind::Op { op, operands, attrs } = &m.nodes[1].kind else { panic!() };
    assert_eq!(*op, StdOp::Linear);
    assert_eq!(operands.as_slice(), &[0]);
    assert_eq!(m.nodes[1].ty.shape.0, vec![8, 2]);
    let AttrValue::Integer(out_dim) = attrs[0].value else { panic!() };
    assert_eq!(out_dim, 2);
    assert_eq!(attrs[0].name, "out_dim");

    // Node 2: Softmax, operands=[1], shape Tensor[8, 2]
    let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else { panic!() };
    assert_eq!(*op, StdOp::Softmax);
    assert_eq!(operands.as_slice(), &[1]);
    assert_eq!(m.nodes[2].ty.shape.0, vec![8, 2]);
}

#[test]
fn unknown_op_errors() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> mystery\n";
    let ast = parse(src).expect("parses");
    let err = ir::build(&ast).expect_err("must fail");
    assert!(matches!(err.kind, BuildErrorKind::UnknownOp { .. }));
}

#[test]
fn unknown_dim_errors() {
    let src = "model X [batch=8]:\n    x: Tensor[zzz, 4]\n    x -> softmax\n";
    let ast = parse(src).expect("parses");
    let err = ir::build(&ast).expect_err("must fail");
    assert!(matches!(err.kind, BuildErrorKind::UnknownDim { .. }));
}

#[test]
fn model_has_no_pipeline_errors() {
    let src = "model X [a=1]:\n    x: Tensor[a, 1]\n";
    let ast = parse(src).expect("parses");
    let err = ir::build(&ast).expect_err("must fail");
    assert!(matches!(err.kind, BuildErrorKind::ModelHasNoPipeline { .. }));
}
```

### 6.3 Test runner

`cargo test` from the repo root or `compiler/`. Expected total after M3a: **~78 tests**
(50 unit-from-M2 + ~15 unit-new + 12 integration-from-M2 + 4 new integration cases).

---

## 7. Acceptance criteria

Milestone 3a is **complete** when all of the following hold:

1. **Workspace builds clean** — `cargo build` from repo root: zero errors, zero warnings.
2. **Module `compiler/src/ir/`** exists with all six files (mod, types, stdlib, build,
   error, tests). All non-trivial.
3. **Public API:** `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>` is callable
   from outside the crate; `Uir`, `UirModel`, `Node`, `NodeId`, `NodeKind`, `OpAttr`,
   `AttrValue`, `Type`, `Shape`, `StdOp`, `BuildError`, `BuildErrorKind` are re-exported
   at the crate root.
4. **`cargo test` is green:** 50+12 prior + ~15 unit + 4 integration = ~81 tests, all
   passing. The integration test `tiny_mlp_builds` asserts the exact UIR structure
   above.
5. **Negative inline tests:** `unknown_op_errors`, `unknown_dim_errors`,
   `model_has_no_pipeline_errors` all pass.
6. **DEVLOG entry for M3a close** with the standard format (What was done /
   Decisions made / Problems encountered / Known tech debt / Next step → M3b).
7. **`CLAUDE.md` "Current Status"** updated to reflect M3a complete and M3b
   (extend to all 5 fixtures) as next.
8. **No changes to `compiler/src/main.rs`** — CLI extension is M3b.

---

## 8. Deferred items

### Deferred to **Milestone 3b** (next vertical slice)

- `--uir` CLI flag (`nflc parse <file> --uir`)
- Multi-pipeline within a model body (more than one `pipeline_stmt`)
- Multi-model files (`pipeline_styles.nfl`)
- Named args in real fixtures (`dropout[rate=0.2]` from `classifier.nfl`,
  `linear[16, bias=true]` from `mixed_args.nfl`)
- Float and Symbol `AttrValue` exercised in real fixtures
- Dropout-rate range validation (`0..=1`)
- Building `classifier.nfl`, `pipeline_styles.nfl`, `comments.nfl`, `mixed_args.nfl`
  end-to-end via integration tests

### Deferred to **Milestone 3c** (final M3 polish slice)

- Viewer-friendly `Display` impls for UIR nodes
- Polished, source-snippet-style error messages
- `docs/language_reference/uir.md` documenting UIR semantics
- Additional negative test fixtures specific to UIR build phase
- Audit and remove any genuinely-unused code from `AttrValue::Symbol` etc.

### Deferred further (M4 / M5 / v0.2+)

- UIR mutation API for fusion (M5)
- Codegen consumption (M4)
- Multi-error reporting (M2.5 / v0.2)
- Property-based testing, fuzzing (v0.2+)

---

## 9. Open questions / known tech debt

These are NOT blockers for M3a implementation, but **must** be logged in the DEVLOG
entry that closes M3a:

1. **`AttrValue::Symbol(String)` is potentially dead code in M3a.** It is only
   needed for `bias=true` (in `mixed_args.nfl`, M3b territory). Decision: keep the
   variant declared (so M3b doesn't have to refactor `AttrValue`), but mark with
   `#[allow(dead_code)]` if the lint fires; remove the directive in M3b at first use.

2. **`OpAttr.name` for positional args reuses `ArgSlot::name` from the signature.**
   This couples consumers to the slot-name string contract. A future refactor to
   trait-based stdlib (if it ever happens) would change this contract — flag, but no
   action in M3a.

3. **`Shape(Vec<u64>)` allocates per shape.** Tensor shapes in practice are short
   (2–4 dims). Heap overhead is acceptable for v0.1. Revisit if profiling shows it
   matters.

4. **`Type.name` is always `"Tensor"` in v0.1.** Same tech-debt category as M2's
   `TypeExpr.name` (spec §9.1). Becomes an `enum TypeKind` in v0.2 when more types
   appear.

5. **Implicit-output convention is documented in M2 spec §5.5 (Pipelines) and the
   grammar comments.** M3a's builder enforces it correctly via
   `last_pipeline_output: Option<NodeId>`. If the convention ever changes (v0.2
   training syntax, multi-output models), this is the load-bearing line.

---

## 10. Transition

After this spec is reviewed and approved by the user, transition to the
`superpowers:writing-plans` skill to produce a step-by-step implementation plan
covering all the deliverables in §3, written for an engineer with zero project
context. Implementation itself happens in a later `superpowers:executing-plans` (or
`superpowers:subagent-driven-development`) cycle.
