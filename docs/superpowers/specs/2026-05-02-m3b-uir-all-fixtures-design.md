# NFL UIR — Vertical Slice 2 (Milestone 3b) — Design Spec

> **Status:** Approved (brainstorming output, 2026-05-02)
> **Authoritative for:** Milestone 3b implementation
> **Source skill:** `superpowers:brainstorming`
> **Next skill:** `superpowers:writing-plans`
> **Builds on:** [M3a UIR vertical-slice 1 spec](./2026-05-02-m3a-uir-tiny-mlp-design.md) and the `nflc::ir` module it shipped.
> **Followed by:** M3c (polish: viewer-friendly Display impls, Ariadne-style errors, additional reference docs).

---

## 1. Context

Milestone 3a shipped `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>` end-to-end
for `tests/fixtures/tiny_mlp.nfl` only. The data model (typed DAG, four stdlib ops with
signatures and shape inference) is in place; this slice **extends UIR coverage to all
five M1 positive fixtures plus the borrow-checker tech-debt cleanup left behind by M3a**.

Per the brainstorming, M3b is the second of three vertical slices for Milestone 3:

| | Status | Scope |
|---|---|---|
| **M3a** | Complete | tiny_mlp.nfl end-to-end; ir module scaffolded |
| **M3b (this spec)** | This brainstorming | Cover the other 4 fixtures + dropout-rate validation + `--uir` CLI flag + build_op refactor |
| **M3c (next)** | Future | Polish: viewer-friendly `Display`, Ariadne-style errors, reference doc |

**Reading order for context:**
1. M3a spec (`docs/superpowers/specs/2026-05-02-m3a-uir-tiny-mlp-design.md`) — UIR data model and stdlib design
2. M3a DEVLOG entry (2026-05-02 second-to-top entry) — known tech debt items #1-#6
3. `language/grammar.ebnf` — for the constructs M3b's new fixtures exercise (multi-line pipeline continuation, named args, multi-model files)
4. The five positive fixtures under `tests/fixtures/` — the corpus M3b must build to UIR

---

## 2. Scope

### In scope (Milestone 3b)

| # | Item | Type |
|---|---|---|
| 1 | Refactor `build_op` to take `&Shape` instead of `&[Node]` (eliminates `Vec<Node>` clone in `build_model`; closes M3a tech-debt #5) | Refactor |
| 2 | Multi-pipeline within a model body (already works; needs explicit test) | Test |
| 3 | Multi-model files (already works; integration test for `pipeline_styles.nfl` with 3 models) | Test |
| 4 | Named args in real fixtures (`dropout[rate=0.2]` from classifier; already works via `resolve_args`) | Test |
| 5 | Float `AttrValue` exercised by an integration test (covered by classifier's dropout rate) | Test |
| 6 | Symbol `AttrValue` exercised by an integration test (covered by mixed_args' `bias=true`) | Test |
| 7 | Dropout-rate range validation (`0..=1`) — new `stdlib::validate_attrs` + new `BuildErrorKind::InvalidAttrValue` | New logic |
| 8 | `nflc parse <file> --uir` CLI flag — new `run_build_uir` + textual UIR pretty-printer | New logic |

### Out of scope — deferred to **M3c**

- Viewer-friendly `Display` impls for UIR nodes (the `--uir` printer in M3b lives in `main.rs`; M3c moves it onto the types as `Display`)
- Polished, source-snippet-style error messages (Ariadne-style)
- `docs/language_reference/uir.md` documenting UIR semantics including the multi-pipeline convention
- Cleanup of small clippy lints noted in M3a DEVLOG tech-debt #6
- Audit of any remaining unused enum variants

### Out of scope — deferred further

- UIR mutation API for fusion (Milestone 5)
- Codegen consumption (Milestone 4 — first profile)
- Multi-error reporting (M2.5 / v0.2)
- Property-based testing, fuzzing (v0.2+)

---

## 3. Deliverables

**Modify:**

| Path | Change |
|---|---|
| `compiler/src/ir/build.rs` | Update `build_op` signature (`&[Node]` → `&Shape`); simplify `build_model` (remove `Vec<Node>` clone in inner loop); wire `validate_attrs` call into `build_op` after `resolve_args` |
| `compiler/src/ir/stdlib.rs` | Add `pub fn validate_attrs(op: StdOp, attrs: &[OpAttr]) -> Result<(), AttrError>` and `enum AttrError`; add helper `get_float_attr` |
| `compiler/src/ir/error.rs` | Add `BuildErrorKind::InvalidAttrValue { op, attr, reason }` variant + `BuildError::invalid_attr_value(...)` constructor |
| `compiler/src/ir/tests.rs` | Update build_op tests to new signature; add ~5 new tests for `validate_attrs` happy/sad paths and per-op no-ops |
| `compiler/src/main.rs` | Add `[cmd, path, flag] if cmd == "parse" && flag == "--uir"` arm; new `run_build_uir(path)` fn; new `print_uir(&Uir)` + helper formatters |
| `DEVLOG.md` | Append M3b close-out entry at the top |
| `CLAUDE.md` | Update "Current Status" → M3b complete, M3c next |

**Create:**

| Path | Purpose |
|---|---|
| `tests/fixtures/negative/dropout_rate_out_of_range.nfl` | Negative fixture: `dropout[rate=1.5]` — must produce `InvalidAttrValue` at line 6 |

**Rename + restructure:**

| Old path | New path | Why |
|---|---|---|
| `compiler/tests/uir_tiny_mlp.rs` | `compiler/tests/uir_fixtures.rs` | Consolidates all UIR integration tests under one file with submodules per fixture; mirrors the parser's `compiler/tests/fixtures.rs` layout. The 4 existing tests move into `mod tiny_mlp { }`; new `mod classifier { }`, `mod pipeline_styles { }`, `mod comments { }`, `mod mixed_args { }`, `mod negative { }` are added. |

**Do NOT touch:**

- `compiler/src/ir/types.rs` — unchanged from M3a (data model is stable)
- `compiler/src/{ast,lexer,parser}/` — frozen since M2
- M2 fixtures and integration tests — frozen
- M3a spec, plan, DEVLOG entry — frozen

---

## 4. Architecture

No structural changes. The `compiler/src/ir/` module shape from M3a remains:

```
compiler/src/ir/
├── mod.rs                               (no changes — public API unchanged)
├── types.rs                             (no changes)
├── stdlib.rs                            (+ validate_attrs, + AttrError, + get_float_attr)
├── build.rs                             (build_op signature change; build_model cleanup)
├── error.rs                             (+ InvalidAttrValue variant + constructor)
└── tests.rs                             (build_op tests updated; + 5 validate_attrs tests)
```

The public API surface remains exactly: `nflc::ir::build(&NflSource) -> Result<Uir, BuildError>`.
M3b adds one new `BuildErrorKind` variant; existing variants and the `Uir` shape are
unchanged.

---

## 5. Components

### 5.1 `build_op` signature change

**Before (M3a):**
```rust
pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    existing_nodes: &[Node],
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    let input_shape = existing_nodes[input_id].ty.shape.clone();
    /* ... */
}
```

**After (M3b):**
```rust
pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    input_shape: &Shape,
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    /* ... uses input_shape directly ... */
}
```

`build_model` becomes:
```rust
for op_ast in &p.steps {
    let input_shape = nodes[current].ty.shape.clone();
    current = build_op(op_ast, current, &input_shape, &mut nodes)?;
}
```

The clone shrinks from `Vec<Node>` (every node so far) to one `Shape` (a short
`Vec<u64>`). The borrow checker is satisfied because `&input_shape` lives independently
of `&mut nodes`.

### 5.2 `validate_attrs` and `AttrError`

In `compiler/src/ir/stdlib.rs`:

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
                    name: "rate", value: rate, min: 0.0, max: 1.0,
                });
            }
            Ok(())
        }
        StdOp::Linear | StdOp::Relu | StdOp::Softmax => Ok(()),
    }
}

fn get_float_attr(attrs: &[OpAttr], name: &'static str) -> Result<f64, AttrError> {
    attrs.iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value { AttrValue::Float(f) => Some(f), _ => None })
        .ok_or(AttrError::MissingAttr { name })
}
```

Validation runs **between** `resolve_args` (which guarantees the named slot exists with
the right type) and `infer_output_shape` (which guarantees rank/inputs are correct).
Value-range validation is its own concern — it does not belong in shape inference.

### 5.3 `BuildErrorKind::InvalidAttrValue` and `build_op` wiring

In `compiler/src/ir/error.rs`:

```rust
pub enum BuildErrorKind {
    /* ... existing variants ... */
    InvalidAttrValue { op: String, attr: String, reason: String },
}

impl BuildError {
    pub fn invalid_attr_value(op: &str, attr: &str, reason: &str, span: Span) -> Self {
        Self {
            message: format!("invalid value for {}.{}: {}", op, attr, reason),
            line: span.line, col: span.col,
            kind: BuildErrorKind::InvalidAttrValue {
                op: op.to_string(),
                attr: attr.to_string(),
                reason: reason.to_string(),
            },
        }
    }
}
```

In `build_op` (after `resolve_args`, before `infer_output_shape`):

```rust
let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
stdlib::validate_attrs(std_op, &attrs).map_err(|e| match e {
    AttrError::OutOfRange { name, .. } =>
        BuildError::invalid_attr_value(&format!("{:?}", std_op), name, &format!("{e}"), op_ast.span),
    AttrError::MissingAttr { name } =>
        BuildError::invalid_attr_value(&format!("{:?}", std_op), name, &format!("{e}"), op_ast.span),
})?;
let out_shape = stdlib::infer_output_shape(std_op, &[input_shape.clone()], &attrs)
    .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
```

(The `format!("{:?}", std_op)` produces `"Dropout"` etc. — fine for v0.1; M3c may
introduce a proper `Display` impl on `StdOp`.)

### 5.4 `--uir` CLI flag

In `compiler/src/main.rs`:

```rust
[cmd, path, flag] if cmd == "parse" && flag == "--uir" => {
    run_build_uir(PathBuf::from(path))
}
```

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
            Ok(uir) => { print_uir(&uir); ExitCode::SUCCESS }
            Err(e) => {
                eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
            ExitCode::FAILURE
        }
    }
}
```

`print_uir` produces a compact textual format:

```
uir-model TinyMLP [batch=8]
  inputs: [n0]
  output: n2
  n0: input "x"        :: Tensor[8, 4]
  n1: linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]
  n2: softmax          :: Tensor[8, 2]    operands=[n1]
```

Implementation: `print_uir(&Uir)` iterates models and calls `print_model(&UirModel)`,
which prints the header and then each `print_node(id, &Node)`. Helpers
`format_shape(&Shape) -> String`, `format_attr(&OpAttr) -> String`, and similar are
small. Total CLI addition: ~50 lines.

The `print_usage` function is updated to mention the new flag.

---

## 6. Testing strategy

### 6.1 Unit tests (`compiler/src/ir/tests.rs`)

Updates to existing `build_op` tests: change `&nodes` → `&Shape` in callers (3 tests
from M3a Task 6).

New tests (~5):

```rust
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
```

Plus the existing build_op tests are reworked to call `build_op(&op_ast, 0, &Shape(vec![..]), &mut out_nodes)`.

### 6.2 Integration tests (`compiler/tests/uir_fixtures.rs`)

The existing `compiler/tests/uir_tiny_mlp.rs` is **renamed** to
`compiler/tests/uir_fixtures.rs` and the four existing tests move into a `tiny_mlp`
submodule. Five new submodules are added:

```rust
mod tiny_mlp {
    // existing 4 tests from M3a, unchanged in semantics
}

mod classifier {
    use nflc::*;

    #[test]
    fn classifier_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/classifier.nfl").unwrap();
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 1);
        let m = &uir.models[0];
        assert_eq!(m.name, "Classifier");

        // Body: 1 input + 7 ops = 8 nodes.
        assert_eq!(m.nodes.len(), 8);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 7);

        // Input node: Tensor[32, 784] (batch=32, input=784).
        assert_eq!(m.nodes[0].ty.shape.0, vec![32, 784]);

        // Final output: Tensor[32, 10] (output=10).
        assert_eq!(m.nodes[7].ty.shape.0, vec![32, 10]);

        // Spot-check the dropout node has Float arg.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[3].kind else { panic!() };
        assert_eq!(*op, StdOp::Dropout);
        let AttrValue::Float(rate) = attrs[0].value else { panic!() };
        assert!((rate - 0.2).abs() < 1e-9);
    }
}

mod pipeline_styles {
    use nflc::*;

    #[test]
    fn pipeline_styles_three_models() {
        let src = std::fs::read_to_string("../tests/fixtures/pipeline_styles.nfl").unwrap();
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 3);
        assert_eq!(uir.models[0].name, "SingleLine");
        assert_eq!(uir.models[1].name, "PerStepWrap");
        assert_eq!(uir.models[2].name, "MixedWrap");

        // All three have the same shape: 1 input + 4 ops = 5 nodes.
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
        let src = std::fs::read_to_string("../tests/fixtures/comments.nfl").unwrap();
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];
        assert_eq!(m.name, "Commented");
        // Body: 1 input + 4 ops (linear, relu, linear, softmax).
        assert_eq!(m.nodes.len(), 5);
        assert_eq!(m.nodes[4].ty.shape.0, vec![4, 2]);
    }
}

mod mixed_args {
    use nflc::*;

    #[test]
    fn mixed_args_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/mixed_args.nfl").unwrap();
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];
        // First op linear[16, bias=true]: positional 16 (Integer) + named bias=true (Symbol).
        let NodeKind::Op { attrs, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].value, AttrValue::Integer(16));
        let OpAttr { name, value } = &attrs[1];
        assert_eq!(name, "bias");
        assert_eq!(*value, AttrValue::Symbol("true".into()));
    }
}

mod negative {
    use nflc::*;

    #[test]
    fn dropout_rate_out_of_range_rejected() {
        let src = std::fs::read_to_string(
            "../tests/fixtures/negative/dropout_rate_out_of_range.nfl"
        ).unwrap();
        let ast = parse(&src).expect("parses");
        let err = ir::build(&ast).expect_err("must fail");
        assert!(matches!(err.kind, BuildErrorKind::InvalidAttrValue { .. }));
        assert_eq!(err.line, 6, "dropout is on line 6");
    }
}
```

### 6.3 Negative fixture

`tests/fixtures/negative/dropout_rate_out_of_range.nfl`:

```nfl
# NEGATIVE: dropout rate 1.5 is outside the valid [0, 1] range.
# Expected: BuildError InvalidAttrValue at line 6.

model X [batch=8]:
    x: Tensor[batch, 4]
    x -> linear[2] -> dropout[rate=1.5] -> softmax
```

### 6.4 Test runner

`cargo test` from repo root runs everything. Expected total after M3b:
- 50 lexer/parser unit (unchanged from M2)
- 22 IR unit prior + ~5 new = ~27 IR unit
- 12 M2 integration (unchanged)
- 4 M3a integration (move into mod tiny_mlp; semantics unchanged)
- 5 new integration (classifier, pipeline_styles, comments, mixed_args, negative)
- **Total: ~98 tests, all passing**

---

## 7. Acceptance criteria

Milestone 3b is **complete** when all of the following hold:

1. **Workspace builds clean** — `cargo build`: zero errors, zero warnings.
2. **`build_op` signature is `(&Operation, NodeId, &Shape, &mut Vec<Node>)`** and the
   inner `Vec<Node>` clone in `build_model` is gone (replaced by a per-step `Shape`
   clone, which is short).
3. **`stdlib::validate_attrs` exists** and is called inside `build_op` between
   `resolve_args` and `infer_output_shape`. `AttrError` enum exists with `OutOfRange`
   and `MissingAttr` variants.
4. **`BuildErrorKind::InvalidAttrValue { op, attr, reason }`** exists with
   `BuildError::invalid_attr_value(op, attr, reason, span)` constructor.
5. **`compiler/tests/uir_fixtures.rs`** exists with submodules: `tiny_mlp`,
   `classifier`, `pipeline_styles`, `comments`, `mixed_args`, `negative`.
   `compiler/tests/uir_tiny_mlp.rs` is removed (renamed).
6. **All 5 new fixture-tests pass:** `classifier_builds`, `pipeline_styles_three_models`,
   `comments_builds`, `mixed_args_builds`, `dropout_rate_out_of_range_rejected`.
7. **The negative fixture file** `tests/fixtures/negative/dropout_rate_out_of_range.nfl`
   exists with a `# NEGATIVE` header and triggers `InvalidAttrValue` at line 6.
8. **CLI smoke (positive):** `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir`
   prints a readable UIR tree (8 nodes, named float arg shown as `rate=0.2`), exit 0.
9. **CLI smoke (negative):** `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir`
   prints `error: invalid value for Dropout.rate: ... at <path>:6:<col>` to stderr, exit 1.
10. **`cargo test` is green** — ~98 tests passing total (50+27 unit + 12+4+5 integration).
11. **DEVLOG entry for M3b close** with the standard format (What was done /
    Decisions made / Problems encountered / Known tech debt / Next step → M3c).
12. **`CLAUDE.md` "Current Status"** updated to reflect M3b complete and M3c (polish)
    as next.

---

## 8. Deferred items

### Deferred to **Milestone 3c** (final M3 polish slice)

- Viewer-friendly `Display` impls for UIR nodes (move the `print_uir` logic from
  `main.rs` onto `Display for Uir` / `Display for Node` etc.)
- Polished, source-snippet-style error messages (Ariadne-style)
- `docs/language_reference/uir.md` documenting UIR semantics, including the explicit
  multi-pipeline convention ("v0.1: independent computations, only the last
  pipeline_stmt's output becomes the model output")
- Cleanup of small clippy lints noted in M3a DEVLOG tech-debt #6
- Audit and remove any genuinely-unused enum variants

### Deferred further (M4 / M5 / v0.2+)

- UIR mutation API for fusion (M5)
- Codegen consumption (M4)
- Multi-error reporting (M2.5 / v0.2)
- Property-based testing, fuzzing (v0.2+)

---

## 9. Open questions / known tech debt

These are NOT blockers for M3b implementation, but **must** be logged in the DEVLOG
entry that closes M3b:

1. **M3a tech-debt items #1-#4 still apply:** `TypeExpr.name` String, `Span` start-only,
   no CI, crate version policy. M3b doesn't address them.

2. **`AttrError` and `ShapeError` are two separate enums in stdlib.** Each represents
   a different per-op failure mode (value validation vs. shape inference). If the
   pattern grows (e.g., op-specific arity rules beyond what `resolve_args` handles),
   M3c can consider unifying into a single `OpError` enum.

3. **`--uir` printer lives in `main.rs`** as free-function logic. M3c moves it onto
   the UIR types as `Display` impls, which makes it consumable from libraries (e.g.,
   future test snapshot tools, IDE plugins, the M7 viewer).

4. **Multi-pipeline behaviour in v0.1:** the grammar permits multiple `pipeline_stmt`s
   in a single model body, but only the last one's output becomes the model output.
   The `last_pipeline_output: Option<NodeId>` tracking from M3a already implements
   this. **Document this convention explicitly in `docs/language_reference/uir.md` as
   part of M3c** so contributors don't accidentally write models with two pipelines
   and wonder why the first is ignored.

5. **`format!("{:?}", std_op)` in `BuildError::invalid_attr_value`** uses `Debug`
   to render `StdOp` as `"Dropout"`, etc. This is good enough for v0.1 error messages
   but is technically a `Debug`-leak. M3c may add a proper `Display for StdOp`.

---

## 10. Transition

After this spec is reviewed and approved by the user, transition to the
`superpowers:writing-plans` skill to produce a step-by-step implementation plan
covering all the deliverables in §3, written for an engineer with zero project
context. Implementation itself happens in a later `superpowers:executing-plans` (or
`superpowers:subagent-driven-development`) cycle.
