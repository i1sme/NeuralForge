# NFL UIR — Vertical Slice 2 (Milestone 3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `nflc::ir::build` end-to-end coverage from `tiny_mlp.nfl` to all 5 M1 positive fixtures, add dropout-rate range validation, the `--uir` CLI flag, and refactor `build_op` to take `&Shape` (closes M3a tech-debt #5).

**Architecture:** No structural changes to the `compiler/src/ir/` module. The 8 deliverables fall into three groups: (1) **Refactor** — `build_op` signature change; (2) **New logic** — `stdlib::validate_attrs` + `AttrError`, `BuildErrorKind::InvalidAttrValue`, `--uir` CLI flag with `print_uir` formatter; (3) **Tests** — restructure existing `uir_tiny_mlp.rs` → `uir_fixtures.rs` with submodules per fixture, plus 5 new integration tests + 1 negative fixture.

**Tech Stack:** Rust 2021, std only. No new external dependencies.

**Source spec:** [`docs/superpowers/specs/2026-05-02-m3b-uir-all-fixtures-design.md`](../specs/2026-05-02-m3b-uir-all-fixtures-design.md). All decisions, types, acceptance criteria live there. **If anything in this plan disagrees with the spec, the spec wins** — flag and stop.

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m3b-uir-all-fixtures` (worktree on branch `claude/m3b-uir-all-fixtures`, branched from main `1e3174f` which has all M3a work merged).

**Branch strategy:** all M3b commits land on `claude/m3b-uir-all-fixtures`. Push and PR when M3b closes.

**Project conventions** (from `CLAUDE.md`):
- TDD: red → green → refactor for impl tasks.
- Each session ends with a `DEVLOG.md` entry.
- Build must be **warning-free** at every commit.

---

## File Structure

**Modify (5 source files + 2 docs):**

| Path | Change | Modified in |
|---|---|---|
| `compiler/src/ir/build.rs` | `build_op` signature → `&Shape`; `build_model` cleanup; wire `validate_attrs` call | Tasks 1, 3 |
| `compiler/src/ir/stdlib.rs` | Add `validate_attrs`, `AttrError`, `get_float_attr` | Task 2 |
| `compiler/src/ir/error.rs` | Add `BuildErrorKind::InvalidAttrValue` + `BuildError::invalid_attr_value(...)` | Task 3 |
| `compiler/src/ir/tests.rs` | Update `build_op_*` tests for new signature; add `validate_attrs` tests | Tasks 1, 2 |
| `compiler/src/main.rs` | Add `--uir` arg arm + `run_build_uir` + `print_uir` + helpers | Task 4 |
| `DEVLOG.md` | Append M3b close-out entry at top | Task 8 |
| `CLAUDE.md` | Update "Current Status" | Task 8 |

**Create (1 negative fixture):**

| Path | Purpose | Created in |
|---|---|---|
| `tests/fixtures/negative/dropout_rate_out_of_range.nfl` | Negative fixture with `dropout[rate=1.5]` | Task 7 |

**Rename + restructure:**

| Old | New | Why | When |
|---|---|---|---|
| `compiler/tests/uir_tiny_mlp.rs` | `compiler/tests/uir_fixtures.rs` (with `mod tiny_mlp { ... }` wrapper around the 4 existing tests, plus 5 new submodules) | Consolidates UIR integration tests | Task 5 (rename) → Task 6 (4 fixture mods) → Task 7 (negative mod) |

**Do NOT touch:**
- `compiler/src/ir/types.rs` — unchanged
- `compiler/src/ir/mod.rs` — unchanged (public API surface stable)
- `compiler/src/{ast,lexer,parser}/` — frozen since M2
- M2 fixtures and integration tests — frozen
- M3a spec / plan / DEVLOG entry — frozen

---

## Verification approach

| Verification | When | How |
|---|---|---|
| `cargo build` warning-free | Every task ends here | From worktree root |
| Each impl task correct | Tasks 2, 3, 4, 6, 7 | TDD: failing test exists first |
| All unit + integration tests | After every commit | `cargo test`, all green |
| CLI manual smoke (positive) | Task 4 + Task 8 | `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir` prints UIR, exit 0 |
| CLI manual smoke (negative) | Task 8 | `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir` prints error, exit 1 |

---

## Task list

| # | Task | Mode | Tests added |
|---|---|---|---|
| 1 | Refactor `build_op` signature (`&[Node]` → `&Shape`); update `build_model` and 3 existing tests | INLINE | 0 (signature change only) |
| 2 | `stdlib::validate_attrs` + `AttrError` (TDD) | SUBAGENT | ~5 |
| 3 | `BuildErrorKind::InvalidAttrValue` + wire `validate_attrs` into `build_op` (TDD) | SUBAGENT | ~2 |
| 4 | `--uir` CLI flag + `print_uir` formatter | SUBAGENT | (manual verify) |
| 5 | Rename `uir_tiny_mlp.rs` → `uir_fixtures.rs`; wrap existing 4 tests in `mod tiny_mlp { }` | INLINE | 0 (restructure) |
| 6 | Add 4 fixture integration tests (classifier, pipeline_styles, comments, mixed_args) | SUBAGENT | 4 |
| 7 | Negative fixture file + `mod negative { }` integration test | SUBAGENT | 1 |
| 8 | Closeout — final cargo test, DEVLOG entry, CLAUDE.md Current Status | INLINE | 0 |

**Total:** 8 tasks, ~8-10 commits, ~12 new tests on top of M3a's 88 = ~98-100 total.

---

## Task 1: Refactor `build_op` signature (INLINE)

**Files:**
- Modify: `compiler/src/ir/build.rs` (`build_op` signature; `build_model` inner loop)
- Modify: `compiler/src/ir/tests.rs` (3 existing `build_op_*` tests update their callers)

This is a mechanical refactor: change one parameter, update three callers. No new functionality.

- [ ] **Step 1: Update `build_op` signature in `compiler/src/ir/build.rs`**

Find the existing `build_op`:

```rust
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
    /* ... */
}
```

Replace with:

```rust
pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    input_shape: &Shape,
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
    let out_shape = stdlib::infer_output_shape(std_op, &[input_shape.clone()], &attrs)
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

- [ ] **Step 2: Update `build_model` inner loop in `compiler/src/ir/build.rs`**

Find the existing inner loop (inside the `Pipeline` arm):

```rust
for op_ast in &p.steps {
    let read_view: Vec<Node> = nodes.clone();
    current = build_op(op_ast, current, &read_view, &mut nodes)?;
}
```

Replace with:

```rust
for op_ast in &p.steps {
    let input_shape = nodes[current].ty.shape.clone();
    current = build_op(op_ast, current, &input_shape, &mut nodes)?;
}
```

- [ ] **Step 3: Update the 3 `build_op_*` tests in `compiler/src/ir/tests.rs`**

Find each of the three test bodies that currently end with:

```rust
let mut out_nodes = nodes.clone();
let id = build_op(&op_ast, 0, &nodes, &mut out_nodes).unwrap();
```

Replace with:

```rust
let mut out_nodes = nodes.clone();
let input_shape = nodes[0].ty.shape.clone();
let id = build_op(&op_ast, 0, &input_shape, &mut out_nodes).unwrap();
```

(For all three: `build_op_linear_produces_correct_node`, `build_op_softmax_preserves_input_shape`, `build_op_unknown_op_errors`. The `nodes` and `out_nodes` locals stay; only the call site changes.)

- [ ] **Step 4: Verify build + tests**

Run: `cargo build` — zero warnings.
Run: `cargo test` — 88 passing (no count change; same tests, same behaviour).

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "refactor(m3b/ir): build_op takes &Shape instead of &[Node]

Closes M3a tech-debt #5. The Vec<Node> snapshot in build_model's
inner loop is gone; instead, the caller extracts the input shape
directly from nodes[current].ty.shape. No behavioural change; all
88 tests pass.

The clone shrinks from a full Vec<Node> (every node so far) to one
Shape (a short Vec<u64>). Borrow checker is satisfied because
&input_shape lives independently of &mut nodes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `stdlib::validate_attrs` + `AttrError` (SUBAGENT, TDD)

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` (append `AttrError`, `validate_attrs`, `get_float_attr`)
- Modify: `compiler/src/ir/tests.rs` (append ~5 tests)

- [ ] **Step 1: Append failing tests to `compiler/src/ir/tests.rs`**

```rust

use super::stdlib::{validate_attrs, AttrError};

#[test]
fn validate_attrs_dropout_in_range_succeeds() {
    let attrs = vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.0) }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
    let attrs = vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.5) }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
    let attrs = vec![OpAttr { name: "rate".into(), value: AttrValue::Float(1.0) }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
}

#[test]
fn validate_attrs_dropout_out_of_range_errors() {
    let attrs = vec![OpAttr { name: "rate".into(), value: AttrValue::Float(1.5) }];
    let err = validate_attrs(StdOp::Dropout, &attrs).unwrap_err();
    assert!(matches!(err, AttrError::OutOfRange { name: "rate", .. }));
    let attrs = vec![OpAttr { name: "rate".into(), value: AttrValue::Float(-0.1) }];
    let err = validate_attrs(StdOp::Dropout, &attrs).unwrap_err();
    assert!(matches!(err, AttrError::OutOfRange { name: "rate", .. }));
}

#[test]
fn validate_attrs_dropout_missing_rate_errors() {
    let err = validate_attrs(StdOp::Dropout, &[]).unwrap_err();
    assert!(matches!(err, AttrError::MissingAttr { name: "rate" }));
}

#[test]
fn validate_attrs_other_ops_no_op() {
    assert!(validate_attrs(StdOp::Linear, &[]).is_ok());
    assert!(validate_attrs(StdOp::Relu, &[]).is_ok());
    assert!(validate_attrs(StdOp::Softmax, &[]).is_ok());
}

#[test]
fn attr_error_displays_human_message() {
    let err = AttrError::OutOfRange { name: "rate", value: 1.5, min: 0.0, max: 1.0 };
    let msg = format!("{err}");
    assert!(msg.contains("rate") && msg.contains("1.5"), "got: {msg}");
}
```

- [ ] **Step 2: Verify FAIL**

Run: `cargo test --lib ir::tests` — compile errors: `validate_attrs` and `AttrError` not in scope.

- [ ] **Step 3: Append to `compiler/src/ir/stdlib.rs`**

```rust

#[derive(Debug, Clone, PartialEq)]
pub enum AttrError {
    OutOfRange { name: &'static str, value: f64, min: f64, max: f64 },
    MissingAttr { name: &'static str },
}

impl std::fmt::Display for AttrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrError::OutOfRange { name, value, min, max } =>
                write!(f, "attribute '{}' value {} is outside [{}, {}]", name, value, min, max),
            AttrError::MissingAttr { name } =>
                write!(f, "missing required attribute: '{}'", name),
        }
    }
}

pub fn validate_attrs(op: StdOp, attrs: &[OpAttr]) -> Result<(), AttrError> {
    match op {
        StdOp::Dropout => {
            let rate = get_float_attr(attrs, "rate")?;
            if !(0.0..=1.0).contains(&rate) {
                return Err(AttrError::OutOfRange {
                    name: "rate",
                    value: rate,
                    min: 0.0,
                    max: 1.0,
                });
            }
            Ok(())
        }
        StdOp::Linear | StdOp::Relu | StdOp::Softmax => Ok(()),
    }
}

fn get_float_attr(attrs: &[OpAttr], name: &'static str) -> Result<f64, AttrError> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value {
            AttrValue::Float(f) => Some(f),
            _ => None,
        })
        .ok_or(AttrError::MissingAttr { name })
}
```

- [ ] **Step 4: Verify PASS — 88 + 5 = 93 tests passing.**

Run: `cargo test`. Zero warnings.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3b/ir): stdlib validate_attrs + AttrError

Per-op value-range validation. Dropout's 'rate' attribute must be
in [0.0, 1.0]; other ops have no value constraints today (linear's
out_dim is already an Integer per signature, no further validation
needed). AttrError variants: OutOfRange, MissingAttr.

5 unit tests: dropout in-range, dropout out-of-range (both ends),
dropout missing rate, no-op for linear/relu/softmax, Display output
contains the offending name and value.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: `BuildErrorKind::InvalidAttrValue` + wire into `build_op` (SUBAGENT, TDD)

**Files:**
- Modify: `compiler/src/ir/error.rs` (new variant + constructor)
- Modify: `compiler/src/ir/build.rs` (wire `validate_attrs` call)
- Modify: `compiler/src/ir/tests.rs` (~2 tests)

- [ ] **Step 1: Append failing tests to `compiler/src/ir/tests.rs`**

```rust

#[test]
fn build_op_dropout_out_of_range_errors() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "dropout".into(),
        args: vec![OpArg::Named { name: "rate".into(), value: ArgValue::Float(1.5) }],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let err = build_op(&op_ast, 0, &input_shape, &mut out_nodes).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::InvalidAttrValue { .. }));
}

#[test]
fn build_op_dropout_in_range_succeeds() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "dropout".into(),
        args: vec![OpArg::Named { name: "rate".into(), value: ArgValue::Float(0.5) }],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let id = build_op(&op_ast, 0, &input_shape, &mut out_nodes).unwrap();
    assert_eq!(out_nodes[id].ty.shape.0, vec![8, 4]);
}
```

- [ ] **Step 2: Verify FAIL — `BuildErrorKind::InvalidAttrValue` not found.**

- [ ] **Step 3: Add `InvalidAttrValue` variant + constructor to `compiler/src/ir/error.rs`**

In the `BuildErrorKind` enum (alongside the existing variants), add:

```rust
    InvalidAttrValue { op: String, attr: String, reason: String },
```

In the `impl BuildError` block, add:

```rust
    pub fn invalid_attr_value(op: &str, attr: &str, reason: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("invalid value for {}.{}: {}", op, attr, reason),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::InvalidAttrValue {
                op: op.to_string(),
                attr: attr.to_string(),
                reason: reason.to_string(),
            },
        }
    }
```

- [ ] **Step 4: Wire `validate_attrs` into `build_op` in `compiler/src/ir/build.rs`**

Find the body of `build_op`:

```rust
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
    let out_shape = stdlib::infer_output_shape(std_op, &[input_shape.clone()], &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
```

Insert the validate_attrs call between `resolve_args` and `infer_output_shape`:

```rust
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
    stdlib::validate_attrs(std_op, &attrs).map_err(|e| {
        let attr_name = match &e {
            stdlib::AttrError::OutOfRange { name, .. } => *name,
            stdlib::AttrError::MissingAttr { name } => *name,
        };
        BuildError::invalid_attr_value(
            &format!("{:?}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
    })?;
    let out_shape = stdlib::infer_output_shape(std_op, &[input_shape.clone()], &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
```

- [ ] **Step 5: Verify PASS — 95 tests passing (93 + 2 new).**

Run: `cargo test`. Zero warnings.

- [ ] **Step 6: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3b/ir): InvalidAttrValue variant + wire validate_attrs into build_op

Adds BuildErrorKind::InvalidAttrValue { op, attr, reason } with
constructor BuildError::invalid_attr_value(...). build_op now calls
stdlib::validate_attrs after resolve_args, before infer_output_shape,
so per-op value constraints (e.g. dropout rate in [0,1]) are checked
during UIR construction.

The std_op's name in error messages is rendered via Debug
(\"Dropout\") for now; M3c may add a proper Display impl for StdOp.

2 new tests: dropout out-of-range rejected with InvalidAttrValue,
dropout in-range builds successfully and preserves shape.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: `--uir` CLI flag + `print_uir` formatter (SUBAGENT)

**Files:**
- Modify: `compiler/src/main.rs` (new flag arm + `run_build_uir` + `print_uir` + helpers + usage text)

- [ ] **Step 1: Read current `compiler/src/main.rs` to know the existing arg-matching layout**

Run: `cat compiler/src/main.rs | head -40`

Confirm the existing `match args.as_slice()` arms include `[]`, `[cmd] if cmd == "parse"`, `[cmd, path] if cmd == "parse"`, `[cmd, path, flag] if cmd == "parse" && flag == "--tokens"`, plus the catch-all.

- [ ] **Step 2: Add the new arm and the new `run_build_uir` + printer functions**

In `compiler/src/main.rs`:

(a) Add the new arm immediately after the `--tokens` arm:

```rust
        [cmd, path, flag] if cmd == "parse" && flag == "--uir" => {
            run_build_uir(PathBuf::from(path))
        }
```

(b) Update `print_usage` to mention the new flag. Replace its body with:

```rust
fn print_usage() {
    println!("nflc — NFL Compiler (Milestone 3b)");
    println!();
    println!("USAGE:");
    println!("  nflc parse <file.nfl>            Parse and pretty-print the AST");
    println!("  nflc parse <file.nfl> --tokens   Print the lexer's token stream");
    println!("  nflc parse <file.nfl> --uir      Build and pretty-print the UIR");
}
```

(c) Append the new functions at the bottom of the file:

```rust
fn run_build_uir(path: PathBuf) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };
    match nflc::parse(&source) {
        Ok(ast) => match nflc::ir::build(&ast) {
            Ok(uir) => {
                print_uir(&uir);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
            ExitCode::FAILURE
        }
    }
}

fn print_uir(uir: &nflc::Uir) {
    for m in &uir.models {
        println!("uir-model {}", m.name);
        println!("  inputs: [{}]",
            m.inputs.iter().map(|id| format!("n{}", id)).collect::<Vec<_>>().join(", "));
        println!("  output: n{}", m.output);
        for (i, node) in m.nodes.iter().enumerate() {
            print_uir_node(i, node);
        }
        println!();
    }
}

fn print_uir_node(id: usize, node: &nflc::Node) {
    let ty = format!("Tensor[{}]", format_uir_shape(&node.ty.shape));
    match &node.kind {
        nflc::NodeKind::Input { name } => {
            println!("  n{}: input {:?}        :: {}", id, name, ty);
        }
        nflc::NodeKind::Op { op, operands, attrs } => {
            let operands_s = operands.iter()
                .map(|o| format!("n{}", o))
                .collect::<Vec<_>>()
                .join(", ");
            let mut line = format!(
                "  n{}: {:?}           :: {}    operands=[{}]",
                id, op, ty, operands_s,
            );
            if !attrs.is_empty() {
                let attrs_s = attrs.iter()
                    .map(format_uir_attr)
                    .collect::<Vec<_>>()
                    .join(", ");
                line.push_str(&format!("    attrs=[{}]", attrs_s));
            }
            println!("{}", line);
        }
    }
}

fn format_uir_shape(shape: &nflc::Shape) -> String {
    shape.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
}

fn format_uir_attr(a: &nflc::OpAttr) -> String {
    match &a.value {
        nflc::AttrValue::Integer(n) => format!("{}={}", a.name, n),
        nflc::AttrValue::Float(f) => format!("{}={}", a.name, f),
        nflc::AttrValue::Symbol(s) => format!("{}={}", a.name, s),
    }
}
```

- [ ] **Step 3: Verify build is warning-free**

Run: `cargo build --bin nflc` — zero warnings.

- [ ] **Step 4: Manual end-to-end smoke**

Run: `cargo run --bin nflc -- parse tests/fixtures/tiny_mlp.nfl --uir`
Expected output similar to:

```
uir-model TinyMLP
  inputs: [n0]
  output: n2
  n0: input "x"        :: Tensor[8, 4]
  n1: Linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]
  n2: Softmax          :: Tensor[8, 2]    operands=[n1]
```

(The exact spacing of the Op label is debug-`{:?}`-rendered, so it's `Linear`/`Softmax`/`Dropout`. M3c will polish.)

Run: `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir`
Expected: 8 nodes printed (1 input + 7 ops), exit 0.

If the second one errors, debug — it means the M3a build pipeline can't yet handle classifier (named float arg or multi-line pipeline). It SHOULD work; M3a's `resolve_args` already supports named args.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/main.rs
git commit -m "feat(m3b/cli): nflc parse <file> --uir

Lexes, parses, builds the UIR, and prints a compact textual format:
- 'uir-model NAME' header per model with inputs/output node ids
- One line per node: 'nN: <kind> :: Tensor[...] operands=[...] attrs=[...]'

Operands and inputs use 'nN' notation matching what the M7 viewer
will use. Op kind is rendered via Debug for now ('Linear', 'Softmax',
etc); M3c will move this onto Display impls on the UIR types.

Errors render as 'error: <msg> at <file>:<line>:<col>' on stderr,
exit 1 — same convention as the existing parse subcommand.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Restructure `uir_tiny_mlp.rs` → `uir_fixtures.rs` (INLINE)

**Files:**
- Rename: `compiler/tests/uir_tiny_mlp.rs` → `compiler/tests/uir_fixtures.rs`
- Modify: the renamed file — wrap the 4 existing tests in `mod tiny_mlp { }`

- [ ] **Step 1: Move the file**

```bash
git mv compiler/tests/uir_tiny_mlp.rs compiler/tests/uir_fixtures.rs
```

- [ ] **Step 2: Wrap the existing 4 tests in `mod tiny_mlp { }`**

The current contents are:

```rust
//! End-to-end integration: lex + parse + build → assert UIR shape for
//! tests/fixtures/tiny_mlp.nfl, plus three small inline negative cases.

use nflc::*;

#[test]
fn tiny_mlp_builds() { /* ... */ }

#[test]
fn unknown_op_errors() { /* ... */ }

#[test]
fn unknown_dim_errors() { /* ... */ }

#[test]
fn model_has_no_pipeline_errors() { /* ... */ }
```

Replace with (header + opening `mod`, body wrapped, closing `}`):

```rust
//! End-to-end integration tests for the UIR builder.
//!
//! One submodule per fixture; `mod negative` for cross-cutting rejection cases.

mod tiny_mlp {
    use nflc::*;

    #[test]
    fn tiny_mlp_builds() { /* unchanged body from M3a */ }

    #[test]
    fn unknown_op_errors() { /* unchanged */ }

    #[test]
    fn unknown_dim_errors() { /* unchanged */ }

    #[test]
    fn model_has_no_pipeline_errors() { /* unchanged */ }
}
```

(Copy the existing test bodies verbatim into the new `mod tiny_mlp { }` block. The `use nflc::*;` moves inside the mod since it now needs to live there.)

- [ ] **Step 3: Verify**

Run: `cargo test --test uir_fixtures` — 4 tests pass under `tiny_mlp::*`.
Run: `cargo test` — full suite, same count as before (95).

- [ ] **Step 4: Commit**

```bash
git add compiler/tests/
git commit -m "refactor(m3b/tests): rename uir_tiny_mlp.rs to uir_fixtures.rs

Renames the M3a integration test file in preparation for adding
4 new fixture submodules in Task 6 plus a 'negative' submodule in
Task 7. The 4 existing tests move into 'mod tiny_mlp { }' wrapper;
no behavioural change.

Mirrors the parser's compiler/tests/fixtures.rs layout (mod
positive { } / mod negative { }).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Add 4 fixture integration tests (SUBAGENT)

**Files:**
- Modify: `compiler/tests/uir_fixtures.rs` (append 4 new submodules)

- [ ] **Step 1: Append 4 new submodules to `compiler/tests/uir_fixtures.rs`**

After the existing `mod tiny_mlp { }` block, append:

```rust

mod classifier {
    use nflc::*;

    #[test]
    fn classifier_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/classifier.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 1);
        let m = &uir.models[0];
        assert_eq!(m.name, "Classifier");

        // Body: 1 input + 7 ops (linear, relu, dropout, linear, relu, linear, softmax)
        // = 8 nodes.
        assert_eq!(m.nodes.len(), 8);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 7);

        // Input: Tensor[32, 784] (batch=32, input=784).
        assert_eq!(m.nodes[0].ty.shape.0, vec![32, 784]);

        // Final output: Tensor[32, 10] (output=10).
        assert_eq!(m.nodes[7].ty.shape.0, vec![32, 10]);

        // Spot-check the dropout node (n3) has its named float arg.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[3].kind else { panic!() };
        assert_eq!(*op, StdOp::Dropout);
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "rate");
        let AttrValue::Float(rate) = attrs[0].value else { panic!() };
        assert!((rate - 0.2).abs() < 1e-9);
    }
}

mod pipeline_styles {
    use nflc::*;

    #[test]
    fn pipeline_styles_three_models() {
        let src = std::fs::read_to_string("../tests/fixtures/pipeline_styles.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 3);
        assert_eq!(uir.models[0].name, "SingleLine");
        assert_eq!(uir.models[1].name, "PerStepWrap");
        assert_eq!(uir.models[2].name, "MixedWrap");

        // All three models have the same pipeline shape:
        //   x: Tensor[batch=4, input=10]
        //   x -> linear[8] -> relu -> linear[output=2] -> softmax
        // = 1 input + 4 ops = 5 nodes.
        for m in &uir.models {
            assert_eq!(m.nodes.len(), 5, "model {}", m.name);
            assert_eq!(m.inputs, vec![0]);
            assert_eq!(m.output, 4);
            assert_eq!(m.nodes[0].ty.shape.0, vec![4, 10]);
            assert_eq!(m.nodes[4].ty.shape.0, vec![4, 2]);
        }
    }
}

mod comments {
    use nflc::*;

    #[test]
    fn comments_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/comments.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];
        assert_eq!(m.name, "Commented");
        // Body: 1 input + 4 ops (linear[16], relu, linear[output=2], softmax) = 5 nodes.
        assert_eq!(m.nodes.len(), 5);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 4);
        assert_eq!(m.nodes[4].ty.shape.0, vec![4, 2]);
    }
}

mod mixed_args {
    use nflc::*;

    #[test]
    fn mixed_args_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/mixed_args.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];

        // First op: linear[16, bias=true] — positional Integer + named Symbol.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(attrs.len(), 2);
        // Positional out_dim = 16
        assert_eq!(attrs[0].name, "out_dim");
        assert_eq!(attrs[0].value, AttrValue::Integer(16));
        // Named bias = true (Symbol)
        assert_eq!(attrs[1].name, "bias");
        assert_eq!(attrs[1].value, AttrValue::Symbol("true".into()));
    }
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test --test uir_fixtures` — all 8 (4 from tiny_mlp + 4 new) tests pass.

If any of `classifier_builds` / `pipeline_styles_three_models` / `comments_builds` /
`mixed_args_builds` fails on shape or count, debug by running:

```bash
cargo run --bin nflc -- parse tests/fixtures/<name>.nfl
```

to see the parsed AST, then:

```bash
cargo run --bin nflc -- parse tests/fixtures/<name>.nfl --uir
```

to see the built UIR. Compare against the assertions.

- [ ] **Step 3: Run full test suite**

Run: `cargo test` — ~99 tests passing (95 prior + 4 new).
Build: `cargo build` — zero warnings.

- [ ] **Step 4: Commit**

```bash
git add compiler/tests/uir_fixtures.rs
git commit -m "test(m3b): UIR integration tests for 4 remaining M1 fixtures

classifier  — 8-node UIR with named Float attr (rate=0.2 on dropout)
pipeline_styles — 3 UirModels, all same shape (5 nodes each)
comments    — comments stripped, 5-node UIR
mixed_args  — Linear with positional Integer + named Symbol (bias=true)

Together these prove the M3a UIR builder handles every construct the
M2 parser accepts in positive fixtures: symbolic dim resolution,
named args (Float and Symbol), multi-model files, multi-line pipeline
continuation, comments at every legal position.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Negative fixture + integration test (SUBAGENT)

**Files:**
- Create: `tests/fixtures/negative/dropout_rate_out_of_range.nfl`
- Modify: `compiler/tests/uir_fixtures.rs` (append `mod negative { }`)

- [ ] **Step 1: Create the negative fixture**

Create `tests/fixtures/negative/dropout_rate_out_of_range.nfl` with EXACT content (the dropout call must end up on line 6):

```nfl
# NEGATIVE: dropout rate 1.5 is outside the valid [0, 1] range.
# Expected: BuildError InvalidAttrValue at line 6.

model X [batch=8]:
    x: Tensor[batch, 4]
    x -> linear[2] -> dropout[rate=1.5] -> softmax
```

- [ ] **Step 2: Verify the fixture parses but fails to build**

Run: `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl`
Expected: AST printed (parsing succeeds), exit 0.

Run: `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir`
Expected: stderr contains `error: invalid value for Dropout.rate: ...`, exit 1.
Note the line number reported (should be 6, the dropout's line).

- [ ] **Step 3: Append `mod negative { }` to `compiler/tests/uir_fixtures.rs`**

After the four fixture mods from Task 6, append:

```rust

mod negative {
    use nflc::*;

    #[test]
    fn dropout_rate_out_of_range_rejected() {
        let src = std::fs::read_to_string(
            "../tests/fixtures/negative/dropout_rate_out_of_range.nfl"
        ).expect("fixture readable");
        let ast = parse(&src).expect("parses");
        let err = ir::build(&ast).expect_err("must fail");
        assert!(matches!(err.kind, BuildErrorKind::InvalidAttrValue { .. }));
        assert_eq!(err.line, 6, "dropout call is on line 6 of the fixture");
    }
}
```

- [ ] **Step 4: Run integration tests**

Run: `cargo test --test uir_fixtures` — 9 tests pass (4 tiny_mlp + 4 fixture + 1 negative).

If `dropout_rate_out_of_range_rejected` fails because the line is not 6, the
implementation may report the dropout's column/line slightly differently — adjust
the assertion to whichever line `cargo run --bin nflc -- parse … --uir` actually
reports (the fixture's `# Expected:` comment should be updated to match).

- [ ] **Step 5: Run full test suite**

Run: `cargo test` — ~100 tests passing (99 + 1 new).
Build: `cargo build` — zero warnings.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/negative/ compiler/tests/uir_fixtures.rs
git commit -m "test(m3b): negative fixture for out-of-range dropout rate

Adds tests/fixtures/negative/dropout_rate_out_of_range.nfl
(dropout[rate=1.5] — outside [0, 1]) and a corresponding
integration test in mod negative { } that asserts the
BuildError is InvalidAttrValue at line 6.

Verifies the validate_attrs → build_op → BuildError chain
end-to-end: lex + parse succeed, ir::build rejects with the
right error variant at the right position.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Closeout (INLINE)

**Files:**
- Modify: `DEVLOG.md` (M3b close-out entry at top)
- Modify: `CLAUDE.md` (Current Status)

- [ ] **Step 1: Final end-to-end verification**

Run from worktree root:

```bash
cargo build               # zero warnings
cargo test                # all ~100 tests pass
```

Smoke the CLI on a representative positive and the negative fixture:

```bash
cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir
# Expected: 8-node UIR, exit 0

cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir
# Expected: error: invalid value for Dropout.rate: … at <file>:6:<col> on stderr, exit 1
echo "exit code: $?"      # Expect 1
```

If anything fails, do NOT commit — fix it first.

- [ ] **Step 2: Append M3b close-out entry to `DEVLOG.md`**

Find the line `## 2026-05-02 — Milestone 3a closed: UIR vertical-slice 1 shipped (tiny_mlp end-to-end)` and use the Edit tool to insert above it (separated by `---`):

```
---

## 2026-05-02 — Milestone 3b closed: UIR extended to all 5 fixtures + dropout validation + --uir CLI

### What was done
- Refactored `build_op` to take `&Shape` instead of `&[Node]`; eliminated the
  Vec<Node> clone in `build_model` (closes M3a tech-debt #5)
- Added `stdlib::validate_attrs` + `AttrError` (`OutOfRange`, `MissingAttr`); validates
  per-op value constraints (currently: dropout rate must be in [0, 1])
- Added `BuildErrorKind::InvalidAttrValue { op, attr, reason }` and wired
  `validate_attrs` into `build_op` between `resolve_args` and `infer_output_shape`
- Added `nflc parse <file> --uir` CLI flag with a compact textual UIR pretty-printer
  using `nN`-style node-id notation (matches what the M7 viewer will use)
- Restructured `compiler/tests/uir_tiny_mlp.rs` → `compiler/tests/uir_fixtures.rs`
  with submodules per fixture (`tiny_mlp`, `classifier`, `pipeline_styles`,
  `comments`, `mixed_args`, `negative`)
- 4 new positive integration tests cover the remaining M1 fixtures end-to-end
- New negative fixture `tests/fixtures/negative/dropout_rate_out_of_range.nfl`
  + integration test asserting `InvalidAttrValue` at the dropout's line
- ~100 tests passing total; zero warnings

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m3b-uir-all-fixtures-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3b-uir-all-fixtures.md` (8 tasks, ~9 commits).

### Problems encountered
- (Fill in real issues found during implementation. If none, write
  "None — implementation followed the plan straight through.")

### Known tech debt (carried forward — see spec §9)
1. **M3a tech-debt items #1-#4 still apply** (TypeExpr.name, Span start-only, no CI,
   crate version policy). M3b doesn't address them.
2. **AttrError and ShapeError are two separate enums in stdlib.** If the pattern
   grows, M3c can consider unifying into a single OpError enum.
3. **`--uir` printer lives in main.rs as free-function logic.** M3c moves it onto
   the UIR types as Display impls so libraries (test snapshot tools, IDE plugins,
   the M7 viewer) can consume it.
4. **Multi-pipeline behaviour in v0.1:** documented here that grammar permits
   multiple `pipeline_stmt`s but only the last's output becomes the model output.
   M3c will document this explicitly in `docs/language_reference/uir.md`.
5. **`format!("{:?}", std_op)` in the InvalidAttrValue message** uses Debug to
   render `StdOp` as `"Dropout"`. Good enough for v0.1; M3c may add `Display for StdOp`.

### Next step
Begin **Milestone 3c — UIR polish.** Adds: (1) viewer-friendly `Display` impls for
all UIR types (move `print_uir` from `main.rs` onto the types); (2) Ariadne-style
source-snippet error rendering; (3) `docs/language_reference/uir.md` documenting UIR
semantics including the multi-pipeline convention; (4) cleanup of clippy lints noted
in M3a tech-debt #6; (5) audit of unused enum variants. After M3c, Milestone 3 is
fully closed and we can begin **Milestone 4 — generic profile (scalar assembly
codegen)**.
```

(Keep the existing M3a entry intact — only add the new M3b entry above it.)

- [ ] **Step 3: Update `CLAUDE.md` "Current Status"**

Find the existing "Current Status" section (currently describes M3a complete) and
replace with:

```
## Current Status

Milestone 3b complete: UIR extended to all 5 M1 fixtures end-to-end. The full UIR
pipeline (lex + parse + build + optional CLI render) is now production-shaped:
`nflc::ir::build` covers symbolic dims, named args (Float and Symbol), multi-pipeline,
multi-model files, comments, and per-op value validation (dropout rate ∈ [0, 1]).
`nflc parse <file> --uir` prints a compact textual UIR. ~100 tests passing.

The immediate next step is **Milestone 3c — UIR polish**: viewer-friendly Display
impls, Ariadne-style errors, language-reference doc for the UIR. After M3c the full
Milestone 3 closes and Milestone 4 (generic profile codegen) begins.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status                  # confirm only the two .md files staged
git commit -m "chore(m3b): close Milestone 3b — UIR for all 5 fixtures + validation + --uir CLI

Adds M3b close-out entry to DEVLOG with the four spec-tracked
tech-debt items plus one new (Debug leak in InvalidAttrValue
message — fixes in M3c).

Updates CLAUDE.md Current Status to reflect M3b complete and
M3c (UIR polish) as next.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 8, Milestone 3b is complete by the spec's acceptance criteria:

1. ✅ Workspace builds clean (Tasks 1, 8)
2. ✅ `build_op` signature is `(&Operation, NodeId, &Shape, &mut Vec<Node>)` (Task 1)
3. ✅ `stdlib::validate_attrs` + `AttrError` exist (Task 2)
4. ✅ `BuildErrorKind::InvalidAttrValue` + constructor exist (Task 3)
5. ✅ `compiler/tests/uir_fixtures.rs` with 6 submodules (Tasks 5, 6, 7)
6. ✅ All 5 new fixture-tests pass (Tasks 6, 7)
7. ✅ Negative fixture file + correct line assertion (Task 7)
8. ✅ CLI smoke positive (Tasks 4, 8)
9. ✅ CLI smoke negative (Task 8)
10. ✅ `cargo test` green at ~100 (Task 8)
11. ✅ DEVLOG entry (Task 8)
12. ✅ CLAUDE.md updated (Task 8)

**Optional follow-up:** push `claude/m3b-uir-all-fixtures` and open a PR.

**The Milestone 3c entry-point** is a fresh `superpowers:brainstorming` cycle for the
final M3 polish slice: Display impls for UIR types (replacing the `main.rs` printer),
Ariadne-style error rendering, language-reference doc, clippy cleanup. M3c is the
last vertical slice before Milestone 3 fully closes and Milestone 4 (generic profile
codegen) begins.
