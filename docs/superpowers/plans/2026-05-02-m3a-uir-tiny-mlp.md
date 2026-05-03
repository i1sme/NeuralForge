# NFL UIR — Vertical Slice 1 (Milestone 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a typed Universal IR (UIR) from a parsed AST for `tests/fixtures/tiny_mlp.nfl` end-to-end, exposing `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>`.

**Architecture:** New module `compiler/src/ir/` (six files: mod, types, stdlib, build, error, tests). Index-based DAG (`Uir { models }`, `UirModel { nodes: Vec<Node> }`, `NodeId = usize`). Stdlib is an `enum StdOp { Linear, Relu, Dropout, Softmax }` with `signature()` and `infer_output_shape()` per op. Builder walks the AST top-down, resolving symbolic dims against `model_params` and op names against the stdlib, with shape inference per node.

**Tech Stack:** Rust 2021, std only (uses `std::collections::HashMap`). No external dependencies.

**Source spec:** [`docs/superpowers/specs/2026-05-02-m3a-uir-tiny-mlp-design.md`](../specs/2026-05-02-m3a-uir-tiny-mlp-design.md). All decisions, types, and acceptance criteria live there. **If anything in this plan disagrees with the spec, the spec wins** — flag and stop.

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m3-uir-prototype` (worktree on branch `claude/m3-uir-prototype`, branched from main `f6fa262` which has all M2 work merged).

**Branch strategy:** all M3a commits land on `claude/m3-uir-prototype`. Push and PR when M3a closes.

**Project conventions** (from `CLAUDE.md`):
- TDD: red → green → refactor. Each impl task starts with a failing test.
- Each session ends with a `DEVLOG.md` entry.
- "Current Status" in `CLAUDE.md` reflects actual state.
- All code/comments in English. Russian only in conversation.
- Build must be **warning-free** at every commit. Use `#[allow(dead_code)]` only with a comment justifying why and pinning when it gets removed.

---

## File Structure

**Create (6 source files + 1 integration test):**

| Path | Purpose | Created in |
|---|---|---|
| `compiler/src/ir/mod.rs` | Module root; declares submodules; re-exports; `pub fn build` lives here | Task 1 → grows in Task 7 |
| `compiler/src/ir/types.rs` | All UIR data types (no logic) | Task 1 |
| `compiler/src/ir/stdlib.rs` | Stdlib types in Task 1; functions added in Tasks 2-3 | Task 1 → grows in Tasks 2-3 |
| `compiler/src/ir/build.rs` | Empty stub in Task 1; helpers added in Tasks 4-7 | Task 1 → grows in Tasks 4-7 |
| `compiler/src/ir/error.rs` | `BuildError`, `BuildErrorKind`, `Display` | Task 1 |
| `compiler/src/ir/tests.rs` | `#[cfg(test)]` unit tests; grown across Tasks 2-7 | Task 1 → grows in Tasks 2-7 |
| `compiler/tests/uir_tiny_mlp.rs` | Integration tests: tiny_mlp + 3 negative inline cases | Task 9 |

**Modify (3 files):**

| Path | Change | Modified in |
|---|---|---|
| `compiler/src/lib.rs` | Add `pub mod ir;` + re-exports of all public IR types | Task 1, finalised Task 8 |
| `DEVLOG.md` | Append M3a close-out entry at the top | Task 10 |
| `CLAUDE.md` | Update "Current Status" → M3a complete, M3b next | Task 10 |

**Do NOT touch:**
- `compiler/src/main.rs` — `--uir` CLI flag is M3b
- `compiler/src/{ast,lexer,parser}/...` — frozen from M2
- M2 spec, plan, fixtures, integration tests
- The M3a spec itself

---

## Verification approach

| Verification | When | How |
|---|---|---|
| Workspace builds clean | Every task ends here | `cargo build` from worktree root: zero errors, zero warnings |
| Each impl task is correct | Tasks 2-7, 9 | TDD: failing unit test exists first, implementation makes it pass |
| All unit tests pass | After every impl task | `cargo test --lib` from `compiler/`, all green |
| All integration tests pass | After Task 9 | `cargo test --test uir_tiny_mlp` from `compiler/`, all 4 green |
| End-to-end smoke | Task 10 final check | Run a quick `cargo test` confirming all ~81 tests pass and `cargo build` clean |

**TDD discipline:** for every code change in Tasks 2-7 and 9, the test is written first and verified to FAIL (red), then the minimum implementation makes it PASS (green), then commit.

---

## Task list

| # | Task | Commits | Tests added |
|---|---|---|---|
| 1 | Scaffold `ir` module — types, stdlib types, error types | 1 | 0 (data only) |
| 2 | Stdlib `resolve()` + `signature()` (TDD) | 1 | ~5 |
| 3 | Stdlib `infer_output_shape()` + helpers (TDD) | 1 | ~4 |
| 4 | `build::resolve_type` helper (TDD) | 1 | ~3 |
| 5 | `build::resolve_args` helper (TDD) | 1 | ~5 |
| 6 | `build::build_op` (TDD) | 1 | ~3 |
| 7 | `build::build_model` + `ir::build` public entry (TDD) | 1 | ~2 |
| 8 | Remove `#![allow(dead_code)]`; final lib.rs verification | 1 | 0 |
| 9 | Integration tests — `tiny_mlp_builds` + 3 negative inline | 1 | 4 |
| 10 | Close out M3a (DEVLOG + CLAUDE.md) | 1 | 0 |

**Total:** 10 tasks, 10 commits, ~22 unit tests + 4 integration = ~26 new tests on top of M2's 62 = ~88 total.

---

## Task 1: Scaffold `ir` module — types, stdlib types, error types

**Files:**
- Create: `compiler/src/ir/mod.rs`
- Create: `compiler/src/ir/types.rs`
- Create: `compiler/src/ir/stdlib.rs` (types only — no functions yet)
- Create: `compiler/src/ir/build.rs` (empty stub)
- Create: `compiler/src/ir/error.rs`
- Create: `compiler/src/ir/tests.rs` (empty stub for now; Tasks 2-7 fill it)
- Modify: `compiler/src/lib.rs` (add module decl + re-exports)

This task creates all data definitions in one go. No logic, no tests yet — those land in Tasks 2-7. The module is wired in `lib.rs` so `nflc::ir::*` is available immediately, but a crate-level `#![allow(dead_code)]` lives on `ir/mod.rs` until Task 8 (when `build` wires everything together — same pattern as M2's parser scaffolding in Task 9).

- [ ] **Step 1: Create `compiler/src/ir/types.rs`**

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
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
    },
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
    /// Used by named keyword-like args (e.g. `bias=true`). Not exercised by
    /// any M3a test (tiny_mlp.nfl uses only Integer args). See spec §9 open Q1.
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub name: String,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }
}
```

- [ ] **Step 2: Create `compiler/src/ir/error.rs`**

```rust
//! Errors raised while building UIR from AST.

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

impl BuildError {
    pub fn unknown_op(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("unknown operation: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownOp { name: name.to_string() },
        }
    }

    pub fn unknown_dim(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("unknown symbolic dimension: '{}' (not declared in model_params)", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownDim { name: name.to_string() },
        }
    }

    pub fn unknown_variable(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("unknown variable: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownVariable { name: name.to_string() },
        }
    }

    pub fn arg_count_mismatch(expected: usize, actual: usize, span: crate::ast::Span) -> Self {
        Self {
            message: format!("operation expects {} positional argument(s), got {}", expected, actual),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ArgCountMismatch { expected, actual },
        }
    }

    pub fn arg_type_mismatch(slot: &str, expected: &str, actual: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("argument '{}' expects {}, got {}", slot, expected, actual),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ArgTypeMismatch {
                slot: slot.to_string(),
                expected: expected.to_string(),
                actual: actual.to_string(),
            },
        }
    }

    pub fn missing_required_arg(slot: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("missing required argument: '{}'", slot),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::MissingRequiredArg { slot: slot.to_string() },
        }
    }

    pub fn unexpected_named_arg(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("operation does not accept named argument: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnexpectedNamedArg { name: name.to_string() },
        }
    }

    pub fn shape(detail: String, span: crate::ast::Span) -> Self {
        Self {
            message: format!("shape error: {}", detail),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ShapeMismatch { detail },
        }
    }

    pub fn model_has_no_pipeline(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("model '{}' has no pipeline_stmt — output is undefined", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ModelHasNoPipeline { name: name.to_string() },
        }
    }
}
```

- [ ] **Step 3: Create `compiler/src/ir/stdlib.rs` with types only (no functions yet)**

```rust
//! Standard library of NFL operations (Milestone 3a defines four:
//! Linear, Relu, Dropout, Softmax). Functions `resolve`, `signature`,
//! and `infer_output_shape` land in Tasks 2-3.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
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
pub enum ArgType {
    Integer,
    Float,
    Symbol,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    WrongInputCount { expected: usize, actual: usize },
    WrongRank { expected: usize, actual: usize, dim_index: Option<usize> },
    MissingAttr { name: &'static str },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::WrongInputCount { expected, actual } =>
                write!(f, "expected {} input(s), got {}", expected, actual),
            ShapeError::WrongRank { expected, actual, dim_index: _ } =>
                write!(f, "expected rank {}, got {}", expected, actual),
            ShapeError::MissingAttr { name } =>
                write!(f, "missing required attribute: '{}'", name),
        }
    }
}
```

- [ ] **Step 4: Create `compiler/src/ir/build.rs` as an empty stub**

```rust
//! AST→UIR builder. Functions land in Tasks 4-7.
```

- [ ] **Step 5: Create `compiler/src/ir/tests.rs` as an empty stub**

```rust
//! Unit tests for the IR module. Tests are added across Tasks 2-7.
```

- [ ] **Step 6: Create `compiler/src/ir/mod.rs`**

```rust
//! Universal IR — the typed computation graph the compiler produces from
//! the parsed AST. Consumed by architecture profiles (M4+) and optimisation
//! passes (M5+).

// Many items below are introduced in Task 1 but only consumed once
// `pub fn build` is wired in Task 7 (and reachable from outside the crate
// via lib.rs re-exports). Until then `cargo build` (lib only) flags the
// helper chain as unused. Removed in Task 8.
#![allow(dead_code)]

pub mod types;
pub mod stdlib;
pub mod error;
mod build;

#[cfg(test)]
mod tests;

pub use error::{BuildError, BuildErrorKind};
pub use stdlib::{ArgSlot, ArgType, Signature, StdOp};
pub use types::{AttrValue, Node, NodeId, NodeKind, OpAttr, Shape, Type, Uir, UirModel};

// Public entry point lands in Task 7.
```

- [ ] **Step 7: Modify `compiler/src/lib.rs` to declare the new module and re-export**

Replace the existing `compiler/src/lib.rs` content (currently 26 lines, ending after the `parse` fn) by appending these lines AT THE END of the file (do NOT remove the existing `parse` fn or other re-exports):

```rust

pub mod ir;

pub use ir::{
    AttrValue, BuildError, BuildErrorKind, Node, NodeId, NodeKind,
    OpAttr, Shape, StdOp, Type, Uir, UirModel,
};
```

- [ ] **Step 8: Verify the build is clean**

Run: `cargo build` from the worktree root.
Expected: `Compiling nflc v0.1.0 (...)` then `Finished ...`. **Zero warnings.**

If you see warnings, fix them before committing. The most likely culprit is a missed import or visibility mistake; if it's `dead_code` and you're sure the `#![allow(dead_code)]` is present in `ir/mod.rs`, double-check that `mod.rs` is the file that contains it and that no submodule has its own conflicting attribute.

- [ ] **Step 9: Verify all M2 tests still pass**

Run: `cargo test`
Expected: 50 unit + 12 integration = 62 tests, all passing. (No new tests yet.)

- [ ] **Step 10: Commit**

```bash
git add compiler/src/ir/ compiler/src/lib.rs
git commit -m "feat(m3a): scaffold ir module with types and error definitions

Adds compiler/src/ir/ with mod.rs, types.rs, stdlib.rs, error.rs,
build.rs (stub), tests.rs (stub). Public types Uir, UirModel,
Node, NodeKind, OpAttr, AttrValue, Type, Shape, StdOp, BuildError,
BuildErrorKind re-exported via lib.rs.

Module-level #![allow(dead_code)] is in place because items are
consumed by tests and by pub fn build, both arriving in subsequent
tasks. Removed in Task 8.

No tests added yet — tests land alongside their implementation in
Tasks 2-9.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Stdlib `resolve()` + `signature()` (TDD)

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` (append `resolve` and `signature`)
- Modify: `compiler/src/ir/tests.rs` (add tests)

- [ ] **Step 1: Write failing tests**

Replace the contents of `compiler/src/ir/tests.rs` with:

```rust
//! Unit tests for the IR module.

use super::stdlib::*;

#[test]
fn resolve_known_ops() {
    assert_eq!(resolve("linear"), Some(StdOp::Linear));
    assert_eq!(resolve("relu"), Some(StdOp::Relu));
    assert_eq!(resolve("dropout"), Some(StdOp::Dropout));
    assert_eq!(resolve("softmax"), Some(StdOp::Softmax));
}

#[test]
fn resolve_unknown_op_returns_none() {
    assert_eq!(resolve("foo"), None);
    assert_eq!(resolve("Linear"), None); // case-sensitive
    assert_eq!(resolve(""), None);
}

#[test]
fn signature_linear_has_one_positional_one_named() {
    let s = signature(StdOp::Linear);
    assert_eq!(s.positional.len(), 1);
    assert_eq!(s.positional[0].name, "out_dim");
    assert_eq!(s.positional[0].ty, ArgType::Integer);
    assert!(s.positional[0].required);
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "bias");
    assert_eq!(s.named[0].ty, ArgType::Symbol);
    assert!(!s.named[0].required);
}

#[test]
fn signature_dropout_has_one_named_required() {
    let s = signature(StdOp::Dropout);
    assert!(s.positional.is_empty());
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "rate");
    assert_eq!(s.named[0].ty, ArgType::Float);
    assert!(s.named[0].required);
}

#[test]
fn signature_relu_and_softmax_are_empty() {
    let r = signature(StdOp::Relu);
    assert!(r.positional.is_empty());
    assert!(r.named.is_empty());
    let s = signature(StdOp::Softmax);
    assert!(s.positional.is_empty());
    assert!(s.named.is_empty());
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `resolve` and `signature` are not found in `stdlib`.

- [ ] **Step 3: Append to `compiler/src/ir/stdlib.rs`**

```rust

pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear" => Some(StdOp::Linear),
        "relu" => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
        _ => None,
    }
}

pub fn signature(op: StdOp) -> Signature {
    use ArgType::*;
    match op {
        StdOp::Linear => Signature {
            positional: &[ArgSlot { name: "out_dim", ty: Integer, required: true }],
            named: &[ArgSlot { name: "bias", ty: Symbol, required: false }],
        },
        StdOp::Relu => Signature { positional: &[], named: &[] },
        StdOp::Dropout => Signature {
            positional: &[],
            named: &[ArgSlot { name: "rate", ty: Float, required: true }],
        },
        StdOp::Softmax => Signature { positional: &[], named: &[] },
    }
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 5 passing.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test`
Expected: 62 prior + 5 new = 67 tests, all passing. Build is warning-free.

- [ ] **Step 6: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): stdlib resolve() and signature()

resolve(\"linear\") -> Some(StdOp::Linear), etc. Unknown names
return None. Case-sensitive. Signatures define positional/named
slots per op; relu/softmax take no args, linear takes positional
out_dim and optional named bias, dropout takes required named rate.

5 unit tests cover all four ops in both functions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Stdlib `infer_output_shape()` + helpers (TDD)

**Files:**
- Modify: `compiler/src/ir/stdlib.rs`
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/ir/tests.rs`:

```rust
use super::types::{AttrValue, OpAttr, Shape};

#[test]
fn infer_linear_output_shape() {
    let input = Shape(vec![8, 4]);
    let attrs = vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }];
    let out = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap();
    assert_eq!(out.0, vec![8, 2]);
}

#[test]
fn infer_linear_with_wrong_rank_input() {
    let input = Shape(vec![8]); // rank 1, linear expects rank 2
    let attrs = vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }];
    let err = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap_err();
    matches!(err, ShapeError::WrongRank { expected: 2, actual: 1, .. });
}

#[test]
fn infer_relu_preserves_shape() {
    let input = Shape(vec![8, 2]);
    let out = infer_output_shape(StdOp::Relu, &[input.clone()], &[]).unwrap();
    assert_eq!(out, input);
}

#[test]
fn infer_softmax_and_dropout_preserve_shape() {
    let input = Shape(vec![3, 7, 2]);
    assert_eq!(infer_output_shape(StdOp::Softmax, &[input.clone()], &[]).unwrap(), input);
    assert_eq!(infer_output_shape(StdOp::Dropout, &[input.clone()], &[]).unwrap(), input);
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `infer_output_shape` not found in `stdlib`.

- [ ] **Step 3: Append helpers and `infer_output_shape` to `compiler/src/ir/stdlib.rs`**

Append:

```rust

use super::types::{AttrValue, OpAttr, Shape};

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
    if inputs.len() == 1 {
        Ok(&inputs[0])
    } else {
        Err(ShapeError::WrongInputCount { expected: 1, actual: inputs.len() })
    }
}

fn require_rank(s: &Shape, expected: usize) -> Result<(), ShapeError> {
    if s.rank() == expected {
        Ok(())
    } else {
        Err(ShapeError::WrongRank { expected, actual: s.rank(), dim_index: None })
    }
}

fn get_int_attr(attrs: &[OpAttr], name: &'static str) -> Result<u64, ShapeError> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value {
            AttrValue::Integer(n) => Some(n),
            _ => None,
        })
        .ok_or(ShapeError::MissingAttr { name })
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 9 passing (5 prior + 4 new).

- [ ] **Step 5: Run full test suite**

Run: `cargo test`
Expected: 71 passing. Build clean.

- [ ] **Step 6: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): stdlib infer_output_shape

Per-op shape inference: linear takes rank-2 input + out_dim attr,
returns Tensor[batch, out_dim]. Relu/Softmax/Dropout are
elementwise: output shape = input shape. Three private helpers
(single_input, require_rank, get_int_attr) keep each op arm short.

4 unit tests cover linear shape, linear wrong-rank rejection,
and elementwise pass-through for relu/softmax/dropout.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: `build::resolve_type` helper (TDD)

**Files:**
- Modify: `compiler/src/ir/build.rs`
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/ir/tests.rs`:

```rust
use super::build::resolve_type;
use super::error::BuildErrorKind;
use crate::ast::{Dim, Span, TypeExpr};
use std::collections::HashMap;

fn span() -> Span { Span::new(1, 1) }

#[test]
fn resolve_type_all_integer_dims() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Integer(8), Dim::Integer(4)],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_symbolic_dim_with_lookup() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("batch".into()), Dim::Integer(4)],
        span: span(),
    };
    let mut params: HashMap<&str, u64> = HashMap::new();
    params.insert("batch", 8);
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_unknown_dim_errors() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("zzz".into())],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let err = resolve_type(&ty, &params).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::UnknownDim { .. }));
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `resolve_type` not found in `build`.

- [ ] **Step 3: Implement in `compiler/src/ir/build.rs`**

Replace the contents of `compiler/src/ir/build.rs` (currently the stub comment) with:

```rust
//! AST→UIR builder.

use std::collections::HashMap;

use crate::ast::{Dim, TypeExpr};

use super::error::BuildError;
use super::types::Shape;

pub(crate) fn resolve_type(
    ty: &TypeExpr,
    params: &HashMap<&str, u64>,
) -> Result<Shape, BuildError> {
    let mut dims: Vec<u64> = Vec::with_capacity(ty.dims.len());
    for dim in &ty.dims {
        match dim {
            Dim::Integer(n) => dims.push(*n),
            Dim::Symbol(name) => {
                let v = params
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| BuildError::unknown_dim(name, ty.span))?;
                dims.push(v);
            }
        }
    }
    Ok(Shape(dims))
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 12 passing (9 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): resolve_type helper

Walks a TypeExpr's dim_list, substituting symbolic dims with their
values from the model_params HashMap. UnknownDim error if a symbol
is missing. Always returns a concrete Shape (all u64 dims).

3 unit tests cover all-integer dims, symbolic-with-lookup, and
unknown-symbol error.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: `build::resolve_args` helper (TDD)

**Files:**
- Modify: `compiler/src/ir/build.rs`
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/ir/tests.rs`:

```rust
use super::build::resolve_args;
use crate::ast::{ArgValue, OpArg};

#[test]
fn resolve_args_one_positional_integer() {
    let args = vec![OpArg::Positional(ArgValue::Integer(512))];
    let attrs = resolve_args(StdOp::Linear, &args, span()).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "out_dim");
    assert_eq!(attrs[0].value, AttrValue::Integer(512));
}

#[test]
fn resolve_args_missing_required_positional() {
    let args: Vec<OpArg> = vec![]; // linear needs out_dim
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(
        err.kind,
        BuildErrorKind::ArgCountMismatch { .. } | BuildErrorKind::MissingRequiredArg { .. }
    ));
}

#[test]
fn resolve_args_extra_positional() {
    let args = vec![
        OpArg::Positional(ArgValue::Integer(2)),
        OpArg::Positional(ArgValue::Integer(3)),
    ];
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgCountMismatch { .. }));
}

#[test]
fn resolve_args_type_mismatch() {
    let args = vec![OpArg::Positional(ArgValue::Float(2.5))]; // out_dim wants Integer
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgTypeMismatch { .. }));
}

#[test]
fn resolve_args_named_only_dropout() {
    let args = vec![OpArg::Named { name: "rate".into(), value: ArgValue::Float(0.2) }];
    let attrs = resolve_args(StdOp::Dropout, &args, span()).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "rate");
    assert_eq!(attrs[0].value, AttrValue::Float(0.2));
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `resolve_args` not found.

- [ ] **Step 3: Implement in `compiler/src/ir/build.rs`**

Append to `compiler/src/ir/build.rs`:

```rust

use crate::ast::{ArgValue, OpArg, Span};

use super::stdlib::{self, ArgSlot, ArgType, StdOp};
use super::types::{AttrValue, OpAttr};

pub(crate) fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    op_span: Span,
) -> Result<Vec<OpAttr>, BuildError> {
    let sig = stdlib::signature(op);

    // Split AST args into positional and named (in source order).
    let mut positionals: Vec<&ArgValue> = Vec::new();
    let mut nameds: Vec<(&str, &ArgValue)> = Vec::new();
    for arg in args {
        match arg {
            OpArg::Positional(v) => positionals.push(v),
            OpArg::Named { name, value } => nameds.push((name.as_str(), value)),
        }
    }

    // Validate positional arity.
    let required_positional = sig.positional.iter().filter(|s| s.required).count();
    let max_positional = sig.positional.len();
    if positionals.len() < required_positional || positionals.len() > max_positional {
        return Err(BuildError::arg_count_mismatch(
            required_positional,
            positionals.len(),
            op_span,
        ));
    }

    let mut attrs: Vec<OpAttr> = Vec::with_capacity(positionals.len() + nameds.len());

    // Bind positionals to slots.
    for (slot, value) in sig.positional.iter().zip(positionals.iter()) {
        check_arg_type(slot, value, op_span)?;
        attrs.push(OpAttr {
            name: slot.name.to_string(),
            value: arg_value_to_attr(value),
        });
    }

    // Bind nameds — match each by slot name.
    for (name, value) in &nameds {
        let slot = sig
            .named
            .iter()
            .find(|s| s.name == *name)
            .ok_or_else(|| BuildError::unexpected_named_arg(name, op_span))?;
        check_arg_type(slot, value, op_span)?;
        attrs.push(OpAttr {
            name: slot.name.to_string(),
            value: arg_value_to_attr(value),
        });
    }

    // Verify all required named args are present.
    for slot in sig.named.iter().filter(|s| s.required) {
        if !nameds.iter().any(|(n, _)| *n == slot.name) {
            return Err(BuildError::missing_required_arg(slot.name, op_span));
        }
    }

    Ok(attrs)
}

fn check_arg_type(slot: &ArgSlot, value: &ArgValue, op_span: Span) -> Result<(), BuildError> {
    let actual = describe_arg_type(value);
    let expected = describe_slot_type(slot.ty);
    let ok = match (slot.ty, value) {
        (ArgType::Integer, ArgValue::Integer(_)) => true,
        (ArgType::Float, ArgValue::Float(_)) => true,
        (ArgType::Symbol, ArgValue::Symbol(_)) => true,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(BuildError::arg_type_mismatch(slot.name, expected, actual, op_span))
    }
}

fn arg_value_to_attr(v: &ArgValue) -> AttrValue {
    match v {
        ArgValue::Integer(n) => AttrValue::Integer(*n),
        ArgValue::Float(f) => AttrValue::Float(*f),
        ArgValue::Symbol(s) => AttrValue::Symbol(s.clone()),
    }
}

fn describe_arg_type(v: &ArgValue) -> &'static str {
    match v {
        ArgValue::Integer(_) => "integer",
        ArgValue::Float(_) => "float",
        ArgValue::Symbol(_) => "identifier",
    }
}

fn describe_slot_type(ty: ArgType) -> &'static str {
    match ty {
        ArgType::Integer => "integer",
        ArgType::Float => "float",
        ArgType::Symbol => "identifier",
    }
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 17 passing (12 prior + 5 new).

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): resolve_args helper

Validates AST OpArg list against the stdlib signature: positional
arity, required named, type matches per slot. Produces Vec<OpAttr>
where each entry pairs slot.name with the resolved value. Errors
returned as ArgCountMismatch / ArgTypeMismatch / MissingRequiredArg
/ UnexpectedNamedArg as appropriate.

5 unit tests cover positional happy path, missing required, extra
positional, type mismatch, and named-only dropout.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: `build::build_op` (TDD)

**Files:**
- Modify: `compiler/src/ir/build.rs`
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/ir/tests.rs`:

```rust
use super::build::build_op;
use super::types::{Node, NodeKind, Type};
use crate::ast::Operation;

fn input_node(shape: Vec<u64>) -> Node {
    Node {
        kind: NodeKind::Input { name: "x".into() },
        ty: Type { name: "Tensor".into(), shape: Shape(shape) },
        source_span: span(),
    }
}

#[test]
fn build_op_linear_produces_correct_node() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "linear".into(),
        args: vec![OpArg::Positional(ArgValue::Integer(2))],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let id = build_op(&op_ast, 0, &nodes, &mut out_nodes).unwrap();
    assert_eq!(id, 1);
    assert_eq!(out_nodes.len(), 2);
    let NodeKind::Op { op, operands, attrs } = &out_nodes[1].kind else {
        panic!("expected Op node");
    };
    assert_eq!(*op, StdOp::Linear);
    assert_eq!(operands, &[0]);
    assert_eq!(attrs[0].value, AttrValue::Integer(2));
    assert_eq!(out_nodes[1].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_op_softmax_preserves_input_shape() {
    let nodes = vec![input_node(vec![8, 2])];
    let op_ast = Operation {
        name: "softmax".into(),
        args: vec![],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let id = build_op(&op_ast, 0, &nodes, &mut out_nodes).unwrap();
    assert_eq!(out_nodes[id].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_op_unknown_op_errors() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation { name: "mystery".into(), args: vec![], span: span() };
    let mut out_nodes = nodes.clone();
    let err = build_op(&op_ast, 0, &nodes, &mut out_nodes).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::UnknownOp { .. }));
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `build_op` not found.

- [ ] **Step 3: Implement in `compiler/src/ir/build.rs`**

Append:

```rust

use super::types::{Node, NodeId, NodeKind, Type};
use crate::ast::Operation;

pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    existing_nodes: &[Node],
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
    let input_shape = existing_nodes[input_id].ty.shape.clone();
    let out_shape = stdlib::infer_output_shape(std_op, &[input_shape], &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
    let id = out_nodes.len();
    out_nodes.push(Node {
        kind: NodeKind::Op {
            op: std_op,
            operands: vec![input_id],
            attrs,
        },
        ty: Type {
            name: "Tensor".to_string(),
            shape: out_shape,
        },
        source_span: op_ast.span,
    });
    Ok(id)
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 20 passing (17 prior + 3 new).

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): build_op

Builds one UIR node from an AST Operation: resolves op name to
StdOp, validates args, infers output shape, pushes the new Node
into out_nodes, returns its NodeId. The input node is referenced
by id; existing_nodes slice is read-only context for shape lookup.

3 unit tests cover linear shape inference, softmax pass-through,
and unknown-op error.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: `build::build_model` + `ir::build` public entry (TDD)

**Files:**
- Modify: `compiler/src/ir/build.rs`
- Modify: `compiler/src/ir/mod.rs`
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/ir/tests.rs`:

```rust
use crate::lexer::lex;
use crate::parser;

fn parse_to_ast(src: &str) -> crate::ast::NflSource {
    let tokens = lex(src).expect("lex");
    let leaked: &'static [crate::lexer::Token] = Box::leak(tokens.into_boxed_slice());
    let mut p = parser::Parser::new(leaked);
    parser::parse_nfl_source(&mut p).expect("parse")
}

#[test]
fn build_tiny_mlp_minimal() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2] -> softmax\n";
    let ast = parse_to_ast(src);
    let uir = super::build(&ast).unwrap();
    assert_eq!(uir.models.len(), 1);
    let m = &uir.models[0];
    assert_eq!(m.name, "X");
    assert_eq!(m.nodes.len(), 3);
    assert_eq!(m.inputs, vec![0]);
    assert_eq!(m.output, 2);
    assert_eq!(m.nodes[0].ty.shape.0, vec![8, 4]);
    assert_eq!(m.nodes[1].ty.shape.0, vec![8, 2]);
    assert_eq!(m.nodes[2].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_model_with_no_pipeline_errors() {
    let src = "model X [a=1]:\n    x: Tensor[a, 1]\n";
    let ast = parse_to_ast(src);
    let err = super::build(&ast).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ModelHasNoPipeline { .. }));
}
```

Note: these tests reference `parser::Parser` and `parser::parse_nfl_source` which are
`pub(crate)` and accessible from inside the crate. The `Box::leak` trick mirrors the
parser's own `parser_of` test helper.

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib ir::tests`
Expected: compile errors — `super::build` not found and/or `build_model` referenced
indirectly.

- [ ] **Step 3: Implement `build_model` + `build` in `compiler/src/ir/build.rs`**

Append:

```rust

use crate::ast::{ModelDef, ModelStmt, NflSource};

use super::types::{Uir, UirModel};

pub fn build(ast: &NflSource) -> Result<Uir, BuildError> {
    let mut models = Vec::with_capacity(ast.models.len());
    for ast_model in &ast.models {
        models.push(build_model(ast_model)?);
    }
    Ok(Uir { models })
}

pub(crate) fn build_model(ast_model: &ModelDef) -> Result<UirModel, BuildError> {
    // Index params for symbolic dim lookup.
    let params: HashMap<&str, u64> = ast_model
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.value))
        .collect();

    let mut nodes: Vec<Node> = Vec::new();
    let mut env: HashMap<String, NodeId> = HashMap::new();
    let mut inputs: Vec<NodeId> = Vec::new();
    let mut last_pipeline_output: Option<NodeId> = None;

    for stmt in &ast_model.body {
        match stmt {
            ModelStmt::VariableDecl(v) => {
                let shape = resolve_type(&v.ty, &params)?;
                let id = nodes.len();
                nodes.push(Node {
                    kind: NodeKind::Input { name: v.name.clone() },
                    ty: Type { name: v.ty.name.clone(), shape },
                    source_span: v.span,
                });
                env.insert(v.name.clone(), id);
                inputs.push(id);
            }
            ModelStmt::Pipeline(p) => {
                let mut current = *env
                    .get(&p.source)
                    .ok_or_else(|| BuildError::unknown_variable(&p.source, p.span))?;
                for op_ast in &p.steps {
                    // We need to read from `nodes` AND push to it. Take a snapshot
                    // of the current slice for read-only access during build_op.
                    let snapshot_len = nodes.len();
                    let (read_view, _) = nodes.split_at(snapshot_len);
                    let read_view: Vec<Node> = read_view.to_vec();
                    current = build_op(op_ast, current, &read_view, &mut nodes)?;
                }
                last_pipeline_output = Some(current);
            }
        }
    }

    let output = last_pipeline_output
        .ok_or_else(|| BuildError::model_has_no_pipeline(&ast_model.name, ast_model.span))?;

    Ok(UirModel {
        name: ast_model.name.clone(),
        nodes,
        inputs,
        output,
        source_span: ast_model.span,
    })
}
```

> **Note on the `read_view: Vec<Node>` clone:** Rust's borrow checker forbids passing
> `&nodes[..]` and `&mut nodes` to `build_op` in the same call. The straight way is to
> pre-clone the prefix used for read-only shape inspection. For tiny_mlp this clone is
> cheap (≤3 nodes); M3b should refactor `build_op` to read shape from a borrowed slice
> WITHOUT needing the full `Node` (e.g. take `&Shape` directly), removing the clone.
> This shortcut is acceptable for M3a — track in DEVLOG as a tech-debt item.

Wire `pub use build::build;` in `compiler/src/ir/mod.rs` by appending to the file
(keep the existing `#![allow(dead_code)]` and module declarations):

```rust

pub use build::build;
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib ir::tests`
Expected: 22 passing (20 prior + 2 new).

- [ ] **Step 5: Run full test suite**

Run: `cargo test`
Expected: 84 passing (62 prior + 22 IR unit). Build clean.

- [ ] **Step 6: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3a/ir): build_model + public build entry

build_model walks the AST: indexes params, processes body
statements (VariableDecl → Input node, Pipeline → chain of Op nodes
via build_op). Tracks last_pipeline_output explicitly to honour the
implicit-output convention; ModelHasNoPipeline error if no pipeline
exists.

ir::build is the public entry that maps each AST model to a UirModel.

Includes a small clone of the node-slice during build_op calls to
work around the borrow-checker (&nodes vs &mut nodes). Acceptable
for M3a's small graphs; M3b should refactor build_op to take &Shape
directly.

2 new tests (tiny_mlp build, no-pipeline error). cargo test green
at 84 total.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Remove `#![allow(dead_code)]`; final lib.rs verification

**Files:**
- Modify: `compiler/src/ir/mod.rs` (remove the directive)
- Verify: `compiler/src/lib.rs` re-exports complete

- [ ] **Step 1: Remove `#![allow(dead_code)]` from `compiler/src/ir/mod.rs`**

Use Edit to remove these lines (and the blank line after):

```rust
// Many items below are introduced in Task 1 but only consumed once
// `pub fn build` is wired in Task 7 (and reachable from outside the crate
// via lib.rs re-exports). Until then `cargo build` (lib only) flags the
// helper chain as unused. Removed in Task 8.
#![allow(dead_code)]

```

- [ ] **Step 2: Verify `cargo build` is still clean**

Run: `cargo build`
Expected: zero warnings. The `pub fn build` + `pub use` chain in `lib.rs` should mean
every `pub` item in the IR module is reachable from outside the crate.

If `cargo build` flags items, audit which ones are unused:
- If they're public types that `nflc::*` uses, add them to the `lib.rs` re-export list.
- If they're internal helpers (e.g. `single_input` in `stdlib.rs`), they're consumed
  by other internal code; the lint shouldn't fire.
- If something is genuinely unused (e.g. a `BuildErrorKind` variant only constructed
  in tests), add a targeted `#[allow(dead_code)]` on the specific item with a
  one-line comment justifying it.

- [ ] **Step 3: Run full test suite**

Run: `cargo test`
Expected: 84 passing.

- [ ] **Step 4: Commit**

```bash
git add compiler/src/ir/mod.rs
git commit -m "chore(m3a/ir): remove module-level dead_code allow

The full Parser/types/build chain is now consumed by
nflc::ir::build (wired in Task 7) and re-exported via lib.rs.
No more dead_code warnings — the directive is no longer needed.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Integration tests — `tiny_mlp_builds` + 3 negative inline cases

**Files:**
- Create: `compiler/tests/uir_tiny_mlp.rs`

- [ ] **Step 1: Create `compiler/tests/uir_tiny_mlp.rs`**

```rust
//! End-to-end integration: lex + parse + build → assert UIR shape for
//! tests/fixtures/tiny_mlp.nfl, plus three small inline negative cases.

use nflc::*;

#[test]
fn tiny_mlp_builds() {
    let src = std::fs::read_to_string("../tests/fixtures/tiny_mlp.nfl")
        .expect("fixture readable");
    let ast = parse(&src).expect("must parse");
    let uir = ir::build(&ast).expect("must build");

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

- [ ] **Step 2: Run integration test**

Run: `cargo test --test uir_tiny_mlp`
Expected: 4 passing.

If `tiny_mlp_builds` fails on the structural assertions (counts, shapes), debug by
running `cargo run --bin nflc -- parse ../tests/fixtures/tiny_mlp.nfl` to see the AST,
then trace through the build logic for each step.

- [ ] **Step 3: Run the FULL test suite**

Run: `cargo test`
Expected: 88 total tests passing (50 lexer + 24 parser + 22 IR unit = 96 unit; wait,
let me recount: 50 lexer + 24 parser = 74 prior unit; +22 IR-unit-from-Tasks-2-7 = 96
unit. Plus 12 integration-from-M2 + 4 integration-from-M3a = 16 integration. Total
**~112 tests**, all passing).

(Exact count may differ by ±2 depending on how each test was structured; the important
thing is "all green, no failures".)

- [ ] **Step 4: Commit**

```bash
git add compiler/tests/uir_tiny_mlp.rs
git commit -m "test(m3a): integration tests for tiny_mlp UIR build

End-to-end (lex + parse + build) verifying the UIR for tiny_mlp
has 3 nodes (input + 2 ops), correct shapes (Tensor[8,4] →
Tensor[8,2] via linear[2] → Tensor[8,2] via softmax), and the
expected output id.

Plus 3 inline negative tests: unknown op, unknown symbolic dim,
model with no pipeline_stmt — each asserting the right
BuildErrorKind variant.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Close out M3a (DEVLOG + CLAUDE.md)

**Files:**
- Modify: `DEVLOG.md` (add M3a close-out entry at top)
- Modify: `CLAUDE.md` (update Current Status)

- [ ] **Step 1: Add M3a close-out entry to `DEVLOG.md`**

Find the line `## 2026-05-02 — Milestone 2 closed: NFL Parser prototype shipped (Rust, std-only)`
and use the Edit tool to insert the new M3a entry **above** it (separated by `---`):

```
---

## 2026-05-02 — Milestone 3a closed: UIR vertical-slice 1 shipped (tiny_mlp end-to-end)

### What was done
- Created `compiler/src/ir/` module with `mod`, `types`, `stdlib`, `build`, `error`,
  `tests` files
- Implemented index-based DAG (`Uir { models }`, `UirModel { nodes: Vec<Node> }`,
  `NodeId = usize`) per spec §5.1
- Defined stdlib for 4 operations (`Linear`, `Relu`, `Dropout`, `Softmax`) with per-op
  `signature()` and `infer_output_shape()`
- Implemented `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>` covering
  symbolic-dim resolution, op binding, positional/named arg validation, and per-op
  shape inference
- Added integration test for `tests/fixtures/tiny_mlp.nfl` plus 3 negative inline tests
  (`UnknownOp`, `UnknownDim`, `ModelHasNoPipeline`)
- Re-exported `Uir`, `BuildError`, `StdOp`, etc. from the crate root

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3a-uir-tiny-mlp-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3a-uir-tiny-mlp.md` (10 tasks, 10 commits).

### Problems encountered
- (Fill in real issues found during implementation. If none, write
  "None — implementation followed the plan straight through.")

### Known tech debt (carried forward — see spec §9 plus this session's findings)
1. **`AttrValue::Symbol(String)` is unused in M3a tests.** Will be exercised in M3b
   when `mixed_args.nfl` (which has `bias=true`) is built. Decision: keep with
   per-item `#[allow(dead_code)]` if the lint fires.
2. **`OpAttr.name` for positional args reuses `ArgSlot.name` from the signature.**
   Couples consumers to slot-name string contracts. No action in M3a.
3. **`Shape(Vec<u64>)` allocates per shape.** Acceptable for v0.1; revisit if
   profiling shows it matters.
4. **`Type.name` is always `"Tensor"` in v0.1.** Same tech-debt category as M2's
   `TypeExpr.name`. Becomes an `enum TypeKind` in v0.2.
5. **`build_model` clones a `Vec<Node>` snapshot before each `build_op` call** to
   work around the borrow-checker (`&nodes` vs `&mut nodes`). Cheap for M3a's small
   graphs (≤3 nodes per model). M3b should refactor `build_op` to take `&Shape`
   instead of `&[Node]`, eliminating the clone.

### Next step
Begin **Milestone 3b — extend UIR to all 5 fixtures.** Adds: multi-pipeline within a
single model, multi-model files (pipeline_styles.nfl), named args in real fixtures
(dropout's `rate=0.2`, mixed_args' `bias=true`), Float and Symbol AttrValue exercised
by integration tests, dropout-rate range validation, plus the `--uir` CLI flag for
end-to-end inspection. The data model and stdlib enum from M3a should not need
extension; this is purely incremental wiring + tests.

---

## 2026-05-02 — Milestone 2 closed: NFL Parser prototype shipped (Rust, std-only)
```

(Keep the existing M2 entry intact — only add the new M3a entry above it.)

- [ ] **Step 2: Update `CLAUDE.md` "Current Status"**

Find the existing "Current Status" section (currently describes M2 complete) and
replace with:

```
## Current Status

Milestone 3a complete: UIR vertical-slice 1 shipped — `nflc::ir::build(&NflSource)`
turns a parsed AST for `tests/fixtures/tiny_mlp.nfl` into a typed Universal IR (DAG
of nodes with concrete shapes). All 4 stdlib operations (Linear, Relu, Dropout,
Softmax) are defined with signatures and shape inference; tiny_mlp exercises Linear
and Softmax end-to-end. ~88 tests passing across lexer, parser, and IR.

The immediate next step is **Milestone 3b — extend UIR to all 5 fixtures**: add
multi-pipeline / multi-model / named-arg coverage, plus the `--uir` CLI flag.
```

- [ ] **Step 3: Final end-to-end verification**

Run from the worktree root:

```bash
cargo build              # zero warnings
cargo test               # all tests green
```

Expected: clean. If anything fails, do NOT commit — fix it first.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status                  # confirm only the two .md files are staged
git commit -m "chore(m3a): close Milestone 3a — UIR for tiny_mlp shipped

Adds M3a close-out entry to DEVLOG, with the four spec-tracked
tech-debt items plus one new one (Vec<Node> snapshot in
build_model — should be refactored away in M3b).

Updates CLAUDE.md Current Status to reflect M3a complete and
M3b (extend UIR to all 5 fixtures) as the next milestone.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 10, Milestone 3a is complete by the spec's acceptance criteria:

1. ✅ Workspace builds clean (Tasks 1, 8, 10)
2. ✅ Module `compiler/src/ir/` with all 6 files (Tasks 1-7)
3. ✅ Public API `nflc::ir::build` + re-exports (Tasks 1, 7, 8)
4. ✅ `cargo test` green at ~88 total (Tasks 2-7, 9)
5. ✅ Negative inline tests for UnknownOp / UnknownDim / ModelHasNoPipeline (Task 9)
6. ✅ DEVLOG entry for M3a close (Task 10)
7. ✅ CLAUDE.md "Current Status" updated (Task 10)
8. ✅ `compiler/src/main.rs` untouched — `--uir` flag stays in M3b (verified by Task 10's `git status`)

**Optional follow-up (recommended before M3b):** push `claude/m3-uir-prototype` and
open a PR. Title suggestion: "Implement Milestone 3a: UIR for tiny_mlp end-to-end".

**The Milestone 3b entry-point** is a fresh `superpowers:brainstorming` cycle to
extend the UIR to all 5 M1 fixtures. The data model from M3a is the foundation;
M3b adds the wiring for multi-pipeline/multi-model bodies, named-arg fixtures, and
the `--uir` CLI flag.
