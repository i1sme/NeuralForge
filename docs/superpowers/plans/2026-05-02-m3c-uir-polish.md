# NFL UIR — Vertical Slice 3 (Milestone 3c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the M3 UIR — move CLI formatting onto types as `Display` impls, hand-roll a source-snippet error formatter, document UIR semantics, audit unused code, fix clippy lints. Closes Milestone 3.

**Architecture:** Pure refactor + docs + cleanup. No new logic, no new public types. Move `print_uir` machinery from `main.rs` onto `Display` impls in `compiler/src/ir/types.rs` and `stdlib.rs`. Add `render_error_with_snippet` in `main.rs` and route all CLI error paths through it. Replace `format!("{:?}", std_op)` with `format!("{}", std_op)` in `build.rs` (changes one error message text from `"Dropout.rate"` → `"dropout.rate"` — no test asserts on the text).

**Tech Stack:** Rust 2021, std only. No new dependencies (chose hand-roll over `ariadne`).

**Source spec:** [`docs/superpowers/specs/2026-05-02-m3c-uir-polish-design.md`](../specs/2026-05-02-m3c-uir-polish-design.md). Decisions, code samples, rationale all live there. **If anything in this plan disagrees with the spec, the spec wins.**

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m3c-uir-polish` (branch `claude/m3c-uir-polish`, branched off `claude/m3b-uir-all-fixtures` at `d974e07` — M3b PR #5 not yet merged at branch-creation time).

**Branch strategy:** all M3c commits land on `claude/m3c-uir-polish`. After M3b PR #5 merges, M3c PR will be reviewable on top of merged main. Push and PR when M3c closes.

**Project conventions** (from `CLAUDE.md`):
- Build must be **warning-free** at every commit. Plus M3c also enforces **clippy-warning-free** as an acceptance criterion.
- Each session ends with a `DEVLOG.md` entry.
- "Add code only when there's a real consumer" — formalised in M3c spec §2.

---

## File Structure

**Modify (5 source files + 2 docs):**

| Path | Change | Modified in |
|---|---|---|
| `compiler/src/ir/types.rs` | + 6 `Display` impls (Uir, UirModel, Node, Shape, OpAttr, AttrValue) | Task 1 |
| `compiler/src/ir/stdlib.rs` | + `Display for StdOp` | Task 2 |
| `compiler/src/ir/build.rs` | `format!("{:?}", std_op)` → `format!("{}", std_op)`; convert `match` → `matches!` in `check_arg_type` | Tasks 2, 4 |
| `compiler/src/ir/tests.rs` | + ~3 Display roundtrip tests; 3× `&[input.clone()]` → `std::slice::from_ref(&input)` | Tasks 1, 4 |
| `compiler/src/main.rs` | Remove `print_uir` / `print_uir_node` / `format_uir_*` free functions; replace with `println!("{}", uir)`; add `render_error_with_snippet` helper; route all error paths through it | Task 3 |
| `DEVLOG.md` | M3c close-out entry | Task 7 |
| `CLAUDE.md` | Update "Current Status" → M3 fully complete | Task 7 |

**Create (1 doc):**

| Path | Purpose | Created in |
|---|---|---|
| `docs/language_reference/uir.md` | UIR semantics reference (~150-200 lines) | Task 6 |

**Do NOT touch:**
- M2 fixtures, parser, lexer — frozen
- M3a/M3b specs, plans, DEVLOG entries — frozen
- M3a/M3b integration test fixtures — frozen
- `compiler/src/ir/error.rs`, `compiler/src/ir/mod.rs` — no changes needed

---

## Verification approach

| Verification | When | How |
|---|---|---|
| `cargo build` warning-free | Every task | From worktree root |
| `cargo clippy --all-targets -- -D warnings` clean | Tasks 4, 7 | After clippy fixes land + final verification |
| All tests pass | Every task | `cargo test`, all green |
| CLI smoke positive | Task 7 | `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir` produces same content as M3b, exit 0 |
| CLI smoke negative w/ snippet | Task 7 | `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir` shows `^` source-snippet, exit 1 |

---

## Task list

| # | Task | Mode | Commits |
|---|---|---|---|
| 1 | `Display` impls for UIR types (Uir, UirModel, Node, Shape, OpAttr, AttrValue) + 3 roundtrip tests | INLINE | 1 |
| 2 | `Display for StdOp` + replace `{:?}` in `BuildError::invalid_attr_value` message | INLINE | 1 |
| 3 | Hand-rolled source-snippet error formatter in main.rs; replace `print_uir` machinery with `println!("{}", uir)` | INLINE | 1 |
| 4 | Clippy cleanup (3× `cloned_ref_to_slice_refs` + 1× `match_like_matches_macro`) | INLINE | 1 |
| 5 | Unused-variant audit (per §5.5 of spec) | INLINE | 1 (or 0 if no removals) |
| 6 | `docs/language_reference/uir.md` | INLINE | 1 |
| 7 | Closeout: final verification, DEVLOG, CLAUDE.md | INLINE | 1 |

**Total:** 7 tasks, 6-7 commits, ~3 new tests on top of M3b's 102 = ~105 total.

All tasks INLINE — milestone is small (mostly mechanical), per-task subagent ceremony would dominate the actual work. Hybrid-mode user choice can be invoked at orchestration time if desired.

---

## Task 1: `Display` impls for UIR types + roundtrip tests

**Files:**
- Modify: `compiler/src/ir/types.rs` (append 6 Display impls)
- Modify: `compiler/src/ir/tests.rs` (append 3 roundtrip tests)

- [ ] **Step 1: Add 3 failing roundtrip tests to `compiler/src/ir/tests.rs`**

Append at the end of the file:

```rust

#[test]
fn shape_displays_as_tensor_with_dims() {
    let s = Shape(vec![32, 784]);
    assert_eq!(format!("{}", s), "Tensor[32, 784]");
}

#[test]
fn attrvalue_displays_each_variant() {
    assert_eq!(format!("{}", AttrValue::Integer(42)), "42");
    assert_eq!(format!("{}", AttrValue::Float(0.5)), "0.5");
    assert_eq!(format!("{}", AttrValue::Symbol("true".into())), "true");
}

#[test]
fn opattr_displays_name_equals_value() {
    let a = OpAttr { name: "out_dim".into(), value: AttrValue::Integer(512) };
    assert_eq!(format!("{}", a), "out_dim=512");
    let b = OpAttr { name: "rate".into(), value: AttrValue::Float(0.2) };
    assert_eq!(format!("{}", b), "rate=0.2");
}
```

- [ ] **Step 2: Verify FAIL**

Run: `cargo test --lib ir::tests`
Expected: 3 compile errors — `Display` not implemented for `Shape`, `AttrValue`, `OpAttr`.

- [ ] **Step 3: Add 6 Display impls to `compiler/src/ir/types.rs`**

Append at the end of the file (after `impl Span`):

```rust

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims: Vec<String> = self.0.iter().map(|d| d.to_string()).collect();
        write!(f, "Tensor[{}]", dims.join(", "))
    }
}

impl std::fmt::Display for AttrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrValue::Integer(n) => write!(f, "{}", n),
            AttrValue::Float(v) => write!(f, "{}", v),
            AttrValue::Symbol(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for OpAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            NodeKind::Input { name } => {
                write!(f, "input {:?}        :: {}", name, self.ty.shape)
            }
            NodeKind::Op { op, operands, attrs } => {
                let ops_s = operands
                    .iter()
                    .map(|o| format!("n{}", o))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
                if !attrs.is_empty() {
                    let a = attrs
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for UirModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "uir-model {}", self.name)?;
        let inputs = self
            .inputs
            .iter()
            .map(|i| format!("n{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "  inputs: [{}]", inputs)?;
        writeln!(f, "  output: n{}", self.output)?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "  n{}: {}", i, node)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Uir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for m in &self.models {
            writeln!(f, "{}", m)?;
        }
        Ok(())
    }
}
```

> **Note:** `Node`'s `Display` references `op` (a `StdOp`) via `{}` — but `Display for StdOp` lands in Task 2. So this code WILL fail to compile until Task 2. **Adjust:** in Task 1, temporarily use `{:?}` for the `op` in `Node`'s Display, then change to `{}` in Task 2 alongside the new `Display for StdOp` impl. Specifically, the Op arm becomes:
>
> ```rust
> write!(f, "{:?}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
> ```
>
> for Task 1, and changes to `{}` in Task 2.

- [ ] **Step 4: Verify PASS — 3 new tests + Display compiles**

Run: `cargo test`
Expected: 105 tests passing (102 prior + 3 new). Build warning-free.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3c/ir): Display impls for UIR types

Adds Display for Uir, UirModel, Node, Shape, OpAttr, AttrValue.
Output format identical to M3b's print_uir/print_uir_node free
functions in main.rs (those will be removed in Task 3).

Display for StdOp lands in Task 2; Node's Display temporarily uses
{:?} for the op field until then.

3 new roundtrip tests cover Shape, AttrValue (all 3 variants),
OpAttr (with Integer and Float values).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `Display for StdOp` + message format change

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` (append Display impl)
- Modify: `compiler/src/ir/build.rs` (one-line format string change)
- Modify: `compiler/src/ir/types.rs` (change `{:?}` → `{}` in `Node`'s Display Op arm)

- [ ] **Step 1: Add a failing test to `compiler/src/ir/tests.rs`**

Append:

```rust

#[test]
fn stdop_displays_lowercase_name() {
    assert_eq!(format!("{}", StdOp::Linear), "linear");
    assert_eq!(format!("{}", StdOp::Relu), "relu");
    assert_eq!(format!("{}", StdOp::Dropout), "dropout");
    assert_eq!(format!("{}", StdOp::Softmax), "softmax");
}
```

- [ ] **Step 2: Verify FAIL**

Run: `cargo test --lib ir::tests::stdop_displays_lowercase_name`
Expected: compile error — `Display` not implemented for `StdOp`.

- [ ] **Step 3: Append `Display for StdOp` to `compiler/src/ir/stdlib.rs`**

Append at the end of the file:

```rust

impl std::fmt::Display for StdOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            StdOp::Linear => "linear",
            StdOp::Relu => "relu",
            StdOp::Dropout => "dropout",
            StdOp::Softmax => "softmax",
        };
        write!(f, "{}", name)
    }
}
```

- [ ] **Step 4: Update Node's Display in `compiler/src/ir/types.rs` to use `{}` for op**

In the `Display for Node` impl, change the Op arm's first `write!` from:

```rust
write!(f, "{:?}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
```

to:

```rust
write!(f, "{}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
```

- [ ] **Step 5: Update `BuildError::invalid_attr_value` message in `compiler/src/ir/build.rs`**

Find the call inside `build_op`:

```rust
        BuildError::invalid_attr_value(
            &format!("{:?}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
```

Change to:

```rust
        BuildError::invalid_attr_value(
            &format!("{}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
```

- [ ] **Step 6: Verify PASS**

Run: `cargo test`
Expected: 106 tests passing (105 + 1 new). Build clean.

The `dropout_rate_out_of_range_rejected` integration test still passes — it asserts on `BuildErrorKind::InvalidAttrValue` variant and `err.line`, not message text.

- [ ] **Step 7: Commit**

```bash
git add compiler/src/ir/
git commit -m "feat(m3c/ir): Display for StdOp + lowercase op name in errors

Adds Display impl: 'linear', 'relu', 'dropout', 'softmax' (matches
NFL source-token names — what the user wrote).

Updates BuildError::invalid_attr_value to use {} instead of {:?},
so error messages now read 'invalid value for dropout.rate: ...'
instead of 'invalid value for Dropout.rate: ...'. More consistent
with how the user wrote the operation name.

Also updates Node's Display (Task 1) to use {} for the op field
now that Display for StdOp exists.

1 new roundtrip test for all 4 StdOp variants. The
dropout_rate_out_of_range integration test still passes (asserts
on variant + line, not message text).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Hand-rolled source-snippet error formatter + main.rs cleanup

**Files:**
- Modify: `compiler/src/main.rs` (remove `print_uir` / `print_uir_node` / `format_uir_shape` / `format_uir_attr`; replace with `println!("{}", uir)`; add `render_error_with_snippet`; route all error paths through it)

- [ ] **Step 1: Add `render_error_with_snippet` helper to `compiler/src/main.rs`**

Add this function (place before `print_usage` or wherever helper functions live):

```rust
/// Render an error with a source-snippet pointer. Output:
///
/// ```text
/// error: <message>
///   --> <path>:<line>:<col>
///    |
/// 12 |     x -> dropout[rate=1.5] -> softmax
///    |                       ^
/// ```
fn render_error_with_snippet(
    source: &str,
    path: &std::path::Path,
    line: u32,
    col: u32,
    message: &str,
) {
    eprintln!("error: {}", message);
    eprintln!("  --> {}:{}:{}", path.display(), line, col);
    let line_idx = line.saturating_sub(1) as usize;
    if let Some(src_line) = source.lines().nth(line_idx) {
        let prefix = line.to_string();
        let pad = " ".repeat(prefix.len());
        eprintln!("{}  |", pad);
        eprintln!("{} | {}", prefix, src_line);
        let mut underline = String::with_capacity(col as usize);
        for _ in 1..col {
            underline.push(' ');
        }
        underline.push('^');
        eprintln!("{}  | {}", pad, underline);
    }
}
```

- [ ] **Step 2: Replace `print_uir` machinery in `run_build_uir`**

Find the success branch of `run_build_uir`:

```rust
            Ok(uir) => {
                print_uir(&uir);
                ExitCode::SUCCESS
            }
```

Replace with:

```rust
            Ok(uir) => {
                print!("{}", uir);
                ExitCode::SUCCESS
            }
```

- [ ] **Step 3: Route all error paths through `render_error_with_snippet`**

In `run_parse`:

```rust
            Err(e) => {
                eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
                ExitCode::FAILURE
            }
```

Becomes (in the `--tokens` lex-error path):

```rust
            Err(e) => {
                let (line, col) = e.position();
                render_error_with_snippet(&source, &path, line, col, &format!("{}", e));
                ExitCode::FAILURE
            }
```

And in the parse-error path (no `--tokens`):

```rust
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
                ExitCode::FAILURE
            }
```

In `run_build_uir`, both error branches (parse + build) route through the same helper:

```rust
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
                ExitCode::FAILURE
            }
```

- [ ] **Step 4: Remove the now-unused free functions**

Delete these functions from `compiler/src/main.rs`:

- `fn print_uir(uir: &nflc::Uir)`
- `fn print_uir_node(id: usize, node: &nflc::Node)`
- `fn format_uir_shape(shape: &nflc::Shape) -> String`
- `fn format_uir_attr(a: &nflc::OpAttr) -> String`

- [ ] **Step 5: Verify build + tests + manual smoke**

Run: `cargo build`
Expected: zero warnings.

Run: `cargo test`
Expected: still 106 passing.

Run: `cargo run --bin nflc -- parse tests/fixtures/tiny_mlp.nfl --uir`
Expected output:

```
uir-model TinyMLP
  inputs: [n0]
  output: n2
  n0: input "x"        :: Tensor[8, 4]
  n1: linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]
  n2: softmax           :: Tensor[8, 2]    operands=[n1]
```

Note `linear` and `softmax` are now lowercase (Task 2 effect). Otherwise identical to M3b output.

Run: `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir`
Expected (with snippet):

```
error: invalid value for dropout.rate: attribute 'rate' value 1.5 is outside [0, 1]
  --> tests/fixtures/negative/dropout_rate_out_of_range.nfl:6:23
  |
6 |     x -> linear[2] -> dropout[rate=1.5] -> softmax
  |                       ^
```

- [ ] **Step 6: Commit**

```bash
git add compiler/src/main.rs
git commit -m "feat(m3c/cli): hand-rolled source-snippet error renderer + Display via println!

Removes ~50 lines of free-function UIR printing (print_uir,
print_uir_node, format_uir_shape, format_uir_attr) — replaced
by 'print!(\"{}\", uir)' now that Display impls exist on the
UIR types (Tasks 1-2).

Adds render_error_with_snippet helper (~20 lines, std-only):
shows the offending source line plus a '^' marker under the
column. Used by all CLI error paths (parse, build, --tokens).

Format mirrors rustc/cargo errors:
  error: <message>
    --> <path>:<line>:<col>
     |
  12 |     <source line>
     |          ^

Net: less code, more responsibility on the types where it
belongs. UIR output content unchanged from M3b.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Clippy cleanup

**Files:**
- Modify: `compiler/src/ir/tests.rs` (3× `cloned_ref_to_slice_refs`)
- Modify: `compiler/src/ir/build.rs` (1× `match_like_matches_macro` in `check_arg_type`)

- [ ] **Step 1: Run clippy to confirm baseline lints**

Run: `cargo clippy --all-targets`
Expected: 5 warnings — 3× `cloned_ref_to_slice_refs` (in `infer_relu_preserves_shape` and `infer_softmax_and_dropout_preserve_shape`), 1× `match_like_matches_macro` (in `check_arg_type`), and 1 duplicate (clippy counts the second softmax/dropout assert separately).

- [ ] **Step 2: Fix `cloned_ref_to_slice_refs` in tests.rs**

Three sites in `compiler/src/ir/tests.rs`. Replace each `&[input.clone()]` with `std::slice::from_ref(&input)`:

```rust
// Before:
let out = infer_output_shape(StdOp::Relu, &[input.clone()], &[]).unwrap();
// After:
let out = infer_output_shape(StdOp::Relu, std::slice::from_ref(&input), &[]).unwrap();
```

```rust
// Before (Softmax):
assert_eq!(infer_output_shape(StdOp::Softmax, &[input.clone()], &[]).unwrap(), input);
// After:
assert_eq!(infer_output_shape(StdOp::Softmax, std::slice::from_ref(&input), &[]).unwrap(), input);
```

```rust
// Before (Dropout):
assert_eq!(infer_output_shape(StdOp::Dropout, &[input.clone()], &[]).unwrap(), input);
// After:
assert_eq!(infer_output_shape(StdOp::Dropout, std::slice::from_ref(&input), &[]).unwrap(), input);
```

- [ ] **Step 3: Fix `match_like_matches_macro` in build.rs**

In `compiler/src/ir/build.rs`, find `check_arg_type`:

```rust
fn check_arg_type(slot: &ArgSlot, value: &ArgValue, op_span: Span) -> Result<(), BuildError> {
    let actual = describe_arg_type(value);
    let expected = describe_slot_type(slot.ty);
    let ok = match (slot.ty, value) {
        (ArgType::Integer, ArgValue::Integer(_)) => true,
        (ArgType::Float, ArgValue::Float(_)) => true,
        (ArgType::Symbol, ArgValue::Symbol(_)) => true,
        _ => false,
    };
    /* ... */
}
```

Replace the `let ok = match` block with:

```rust
    let ok = matches!(
        (slot.ty, value),
        (ArgType::Integer, ArgValue::Integer(_))
            | (ArgType::Float, ArgValue::Float(_))
            | (ArgType::Symbol, ArgValue::Symbol(_))
    );
```

- [ ] **Step 4: Verify clippy clean + tests pass**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: exit 0, no warnings.

Run: `cargo test`
Expected: still 106 passing.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/ir/
git commit -m "fix(m3c): clippy cleanup — std::slice::from_ref + matches!

Two clippy lint categories from M3a tech-debt #6:

1. cloned_ref_to_slice_refs — three sites in ir/tests.rs that
   built &[input.clone()] when std::slice::from_ref(&input) is
   semantically identical and avoids the clone.

2. match_like_matches_macro — check_arg_type's let-ok match block
   becomes the matches!() macro form clippy prefers.

cargo clippy --all-targets -- -D warnings now exits 0. All 106
tests still pass; behaviour unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Unused-variant audit

**Files:** none modified unless audit finds genuinely dead items.

This is an inspection task. The spec §5.5 already classified `ShapeError::WrongInputCount` as **defensively reachable → KEEP**. This task verifies that classification AND inspects all other variants.

- [ ] **Step 1: Temporarily add `#![deny(dead_code)]` at the top of each module**

For each of these modules, add `#![deny(dead_code)]` as the first non-doc line:
- `compiler/src/ir/types.rs`
- `compiler/src/ir/stdlib.rs`
- `compiler/src/ir/build.rs`
- `compiler/src/ir/error.rs`
- `compiler/src/lexer/tokens.rs`
- `compiler/src/lexer/mod.rs`
- `compiler/src/parser/mod.rs`
- `compiler/src/ast.rs`

- [ ] **Step 2: Run `cargo build --tests`**

Run: `cargo build --tests 2>&1 | grep -E "error\[E0|warning" | head -30`
Capture every reported `dead_code` error. For each item, classify per spec §2:

- **Truly dead** (no constructor/no caller anywhere): mark for removal
- **Defensively reachable** (constructor exists in a guard helper, no test fires): mark for keep + add justification comment

Expected items the audit will surface (best guess; the actual run may differ):
- `ShapeError::WrongInputCount` — DEFENSIVE (kept; spec-decided)
- `Span::new` — used in tests via helpers (alive)
- Possibly `Token::new` — used (alive)
- Possibly some `BuildErrorKind` constructors that aren't fired in tests but ARE used by builder — alive
- Possibly some helper fns

- [ ] **Step 3: Apply audit decisions**

For each "defensively reachable" item, add an `#[allow(dead_code)]` attribute on it directly (not module-wide) with a one-line justification comment explaining why the guard exists:

```rust
#[allow(dead_code)] // Defensive: catches caller bug if multi-input op slips into single-input infer path. Exercised in M5+.
WrongInputCount { expected: usize, actual: usize },
```

For each "truly dead" item, REMOVE it. (None expected for v0.1 today.)

- [ ] **Step 4: Remove the temporary `#![deny(dead_code)]` directives**

Remove the `#![deny(dead_code)]` lines added in Step 1 from all 8 modules. (The per-item `#[allow]` attributes added in Step 3 stay.)

- [ ] **Step 5: Verify build clean + tests pass**

Run: `cargo build && cargo test`
Expected: zero warnings, 106 passing.

- [ ] **Step 6: Commit (only if items changed)**

If Step 3 modified any file:

```bash
git add compiler/src/
git commit -m "chore(m3c): audit unused enum variants — kept defensively-reachable ones

Per the project principle in spec §2 ('Add code only when there's
a real consumer'), audited every public/private enum variant and
function for dead_code. Findings:

- ShapeError::WrongInputCount: KEPT (defensive guard in
  single_input helper). Annotated with #[allow(dead_code)] +
  justifying comment. Exercised by M5+'s multi-input ops.
- (Other findings or 'no other items found'.)

cargo build clean, cargo test still 106 passing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

If Step 3 found nothing to change beyond what was already classified:

```bash
git status
# (Should show no changes if audit confirmed everything as-is.)
# Skip the commit; record audit results in DEVLOG (Task 7) instead.
```

---

## Task 6: `docs/language_reference/uir.md`

**Files:**
- Create: `docs/language_reference/uir.md`

- [ ] **Step 1: Create `docs/language_reference/uir.md`**

```markdown
# NFL Universal IR (UIR) — Language Reference

> **Status:** Defines the UIR as of NFL v0.1 (Milestone 3 complete).
> **Authoritative source:** `compiler/src/ir/` and the M3a/M3b/M3c specs under
> `docs/superpowers/specs/`.

This document explains what the UIR is, how it's structured, and the rules the
builder enforces when constructing it from the AST. If this doc and the source
disagree, the source wins — file an issue.

---

## 1. Overview

The Universal IR is a typed computation graph that the NFL compiler produces
between parsing and codegen. It is the input to architecture profiles
(`profiles/generic/` and friends, M4+) and to optimisation passes (kernel fusion,
M5).

```
NFL source  ──lex──>  Tokens  ──parse──>  AST  ──build──>  UIR  ──codegen──>  assembly
                                                  ▲                ▲
                                                  M3              M4 (next)
```

The UIR is hardware-agnostic. All architecture-specific decisions live in
profiles, which consume the UIR.

---

## 2. Data shape

The UIR is an **index-based DAG**:

```rust
pub struct Uir {
    pub models: Vec<UirModel>,
}

pub struct UirModel {
    pub name: String,
    pub nodes: Vec<Node>,    // index = NodeId = usize
    pub inputs: Vec<NodeId>, // entry points (variable_decls)
    pub output: NodeId,      // implicit-output convention
    pub source_span: Span,
}

pub type NodeId = usize;

pub struct Node {
    pub kind: NodeKind,
    pub ty:   Type,           // Tensor type with concrete shape
    pub source_span: Span,
}

pub enum NodeKind {
    Input { name: String },
    Op { op: StdOp, operands: Vec<NodeId>, attrs: Vec<OpAttr> },
}
```

**Why index-based?** Easy to clone, easy to traverse (just iterate `nodes`), easy
to mutate (M5 fusion will replace nodes by id), easy to share subexpressions
(multiple nodes can reference the same `NodeId`). Standard compiler-textbook
choice.

**Why immutable in v0.1?** The builder never modifies a node after pushing it.
M5 will introduce mutation when fusion lands.

---

## 3. Node kinds

### Input

`NodeKind::Input { name }` — corresponds to a `variable_decl` in the AST. The
`Type` carries the resolved shape (no symbolic dims; symbols already substituted
against `model_params`).

Example: `x: Tensor[batch, 4]` in a model with `[batch=8]` becomes:
```
n0: input "x"        :: Tensor[8, 4]
```

### Op

`NodeKind::Op { op, operands, attrs }` — applies an operation from the stdlib.
- `op` is a `StdOp` enum variant (resolved from the AST identifier).
- `operands` are `NodeId`s referencing the inputs (one for v0.1's single-input ops).
- `attrs` are validated, type-resolved arguments (positional and named, in the
  signature's slot order).

Example: `linear[16, bias=true]` becomes:
```
n1: linear           :: Tensor[8, 16]    operands=[n0]    attrs=[out_dim=16, bias=true]
```

---

## 4. Stdlib operations (v0.1)

Four operations are recognised:

| StdOp | Signature | Output shape |
|---|---|---|
| `Linear`  | positional `out_dim: Integer` (required), named `bias: Symbol` (optional) | `Tensor[input.batch, out_dim]` |
| `Relu`    | no args | input shape (elementwise) |
| `Dropout` | named `rate: Float` (required, must be `0..=1`) | input shape (elementwise) |
| `Softmax` | no args | input shape (elementwise) |

Adding a new op = new `StdOp` variant + new arms in `signature()` and
`infer_output_shape()` in `compiler/src/ir/stdlib.rs`.

---

## 5. Implicit semantics (rules the builder enforces)

These are NOT enforced by the grammar — they're checked when AST → UIR.

1. **Symbolic dims in `Tensor[…]`** must reference an identifier in `model_params`.
   `Tensor[batch, 4]` requires `batch=N` declared in the model header. Failing the
   lookup raises `BuildErrorKind::UnknownDim`.

2. **Symbolic args in `op[name]`** (positional Symbol args) are likewise resolved
   against `model_params`. `linear[output]` where `output=10` is a model param
   becomes `linear[10]` semantically. Symbols that don't match a param stay as
   Symbols and are subject to the slot's type check.

3. **Operation names** must resolve to a `StdOp` (currently linear/relu/dropout/
   softmax). Unknown names raise `BuildErrorKind::UnknownOp`.

4. **First identifier of a `pipeline_stmt`** must reference a previously declared
   variable. Otherwise `BuildErrorKind::UnknownVariable`.

5. **Per-op value validation** runs after argument type-resolution but before
   shape inference. Currently only `Dropout`'s `rate` is validated (∈ [0, 1]).
   Failures raise `BuildErrorKind::InvalidAttrValue`.

6. **Implicit output convention.** A model's output is the value produced by the
   last operation of the **last** `pipeline_stmt` in its body — tracked
   explicitly via `last_pipeline_output: Option<NodeId>`. Models with no
   `pipeline_stmt` raise `BuildErrorKind::ModelHasNoPipeline`.

7. **Multi-pipeline convention (v0.1).** The grammar permits multiple
   `pipeline_stmt`s in a single model body, but only the last one's output
   becomes the model output. Earlier pipelines are independent computations with
   no consumer in v0.1. (A future v0.2 may introduce pipeline output binding.)

---

## 6. CLI inspection

`nflc parse <file.nfl> --uir` lexes, parses, builds, and prints the UIR:

```
$ nflc parse tests/fixtures/tiny_mlp.nfl --uir
uir-model TinyMLP
  inputs: [n0]
  output: n2
  n0: input "x"        :: Tensor[8, 4]
  n1: linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]
  n2: softmax           :: Tensor[8, 2]    operands=[n1]
```

`nN` notation for node IDs is used everywhere (in `inputs`, `output`, and
`operands` lists). Op kind is rendered via `Display for StdOp` (lowercase, matching
the source token).

Errors are rendered with a source-snippet pointer:

```
$ nflc parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir
error: invalid value for dropout.rate: attribute 'rate' value 1.5 is outside [0, 1]
  --> tests/fixtures/negative/dropout_rate_out_of_range.nfl:6:23
   |
6  |     x -> linear[2] -> dropout[rate=1.5] -> softmax
   |                       ^
```

---

## 7. What v0.1 doesn't have

Listed here so contributors don't accidentally rely on absent features:

- **Multi-output models.** A model has effectively one output (the last pipeline's
  last op).
- **Pipeline output binding.** Pipelines don't bind their output to a name that
  later pipelines can reference. Multi-pipeline bodies have only one consumer
  (the implicit-output convention).
- **Mutation API.** `Uir` is immutable-by-construction in v0.1. M5 (kernel fusion)
  introduces mutation.
- **Profile-specific lowering.** All profile work is M4+.
- **Multi-error reporting.** First error halts the build. v0.2.
- **Source-snippet errors with multi-line context, color, or labels.** M3c's
  hand-rolled formatter is single-line, monochrome. v0.2+ may upgrade.
- **Custom operations.** No syntax for declaring user-defined ops. v0.2+.
- **Training syntax** (loss, optimiser). NFL v0.1 is inference-only.
```

- [ ] **Step 2: Verify the doc renders**

Run: `wc -l docs/language_reference/uir.md`
Expected: between 150 and 250 lines.

Run: `head -3 docs/language_reference/uir.md`
Expected: title + status line.

(No tests for the doc; it's prose. Spot-check that the code blocks and CLI examples
match what M3c produces.)

- [ ] **Step 3: Commit**

```bash
git add docs/language_reference/uir.md
git commit -m "docs(m3c): UIR language reference

Documents what the UIR is, its data shape, node kinds, stdlib ops,
implicit semantics (incl. the multi-pipeline convention from M3b
open-Q4), CLI inspection format, and the v0.1 omissions list.

~150-250 lines of prose, mirrors the structure of M1's
docs/language_reference/grammar.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Closeout — final verification, DEVLOG, CLAUDE.md

**Files:**
- Modify: `DEVLOG.md` (add M3c close-out at top)
- Modify: `CLAUDE.md` (Current Status → M3 fully complete)

- [ ] **Step 1: Final end-to-end verification**

Run from worktree root:

```bash
cargo build                      # zero warnings
cargo clippy --all-targets -- -D warnings   # exits 0
cargo test                       # 106 tests, all green
```

Smoke positive + negative:

```bash
cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir
# Expected: 8-node UIR, lowercase op names (linear/dropout/softmax), exit 0

cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir
# Expected: source-snippet error with ^ marker, exit 1
echo "exit code: $?"             # Expect 1
```

If anything fails, do NOT commit — fix it first.

- [ ] **Step 2: Append M3c close-out entry to `DEVLOG.md`**

Find the existing line `## 2026-05-02 — Milestone 3b closed: …` and use the Edit tool
to insert the new M3c entry above it (separated by `---`):

```
---

## 2026-05-02 — Milestone 3c closed: UIR polish — Display impls + source-snippets + reference doc + clippy clean

### What was done
- Added `Display` impls for all UIR types (`Uir`, `UirModel`, `Node`, `Shape`,
  `OpAttr`, `AttrValue`) and for `StdOp`. Output content matches M3b's `print_uir`
  exactly.
- Removed `print_uir`, `print_uir_node`, `format_uir_shape`, `format_uir_attr`
  free functions from `compiler/src/main.rs` (~50 lines deleted; replaced by one
  `print!("{}", uir)` line).
- Added `render_error_with_snippet` helper in `main.rs` (~20 lines, hand-rolled
  std-only). Routes all CLI error paths through it. Output matches rustc/cargo
  conventions (`error:` line, `--> file:line:col` pointer, `^` underline).
- Replaced `format!("{:?}", std_op)` with `format!("{}", std_op)` in
  `BuildError::invalid_attr_value`. Error messages now use lowercase op names
  ('dropout.rate' not 'Dropout.rate'), matching the NFL source token names.
- Created `docs/language_reference/uir.md` (~150-250 lines): UIR semantics,
  multi-pipeline convention, CLI inspection format, v0.1 omissions list.
- Cleared all `cargo clippy` warnings: 3× `cloned_ref_to_slice_refs` →
  `std::slice::from_ref`, 1× `match_like_matches_macro` → `matches!`. M3a
  tech-debt #6 closed.
- Audited all enum variants for dead code per project principle. Findings logged
  below.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-02-m3c-uir-polish-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m3c-uir-polish.md` (7 tasks, 6-7 commits).

### Project principle formalised in M3c spec §2

> **Add code only when there's a real consumer.** Do not retain "for-future-use"
> variants/functions/types via `#[allow(dead_code)]`. Remove unused items when
> discovered; re-introduce with the first real use (with tests).

**Nuance:** "no real consumer" means *no caller at all*, not "unreached in current
tests". Defensively reachable code (constructed by guard helpers that protect
against future caller bugs) IS used and should be kept — annotated with
`#[allow(dead_code)]` + a one-line comment explaining the defensive role.

### Audit results
- `ShapeError::WrongInputCount` — KEPT (defensive guard in `single_input` helper;
  catches the class of caller bug where a multi-input op slips into single-input
  shape inference. Exercised for real in M5 when `add`/`concat` arrive).
- (Other findings TBD by audit run; populate this list during Task 5.)

### Problems encountered
- (Fill in real issues found during implementation. If none, write
  "None — implementation followed the plan straight through.")

### Known tech debt (carried forward to v0.2 / M4+)
1. M3a tech-debt items #1-#4 still apply (TypeExpr.name, Span start-only, no CI,
   crate version policy). v0.2.
2. AttrError + ShapeError still two enums. Unification is a v0.2 consideration.
3. Multi-error reporting — first-error-halt continues. v0.2.
4. No CI yet. Add as a small follow-up before M4 ships.
5. The `single_input` defensive guard's `WrongInputCount` path becomes
   exercised-for-real in M5 with multi-input ops.

### Next step
**Milestone 3 fully complete.** The UIR pipeline (lex → parse → build → CLI render)
is production-shaped and well-documented.

The immediate next milestone is **Milestone 4 — generic profile (scalar assembly
codegen)**: implement the first architecture profile that consumes the UIR and
emits scalar assembly for any POSIX target. This is the first time the project
produces actual machine-executable output. The first M4 decision is the assembly
flavour (AT&T x86-64 syntax for `as`, NASM, or LLVM textual IR as a stepping
stone) — to be resolved via a fresh `superpowers:brainstorming` cycle for M4.

---

## 2026-05-02 — Milestone 3b closed: ...
```

(Keep the existing M3b entry intact — only add the new M3c entry above it.)

- [ ] **Step 3: Update `CLAUDE.md` "Current Status"**

Find the existing "Current Status" section (M3b version) and replace with:

```
## Current Status

**Milestone 3 fully complete.** The UIR pipeline is production-shaped:
`nflc::ir::build(&NflSource)` turns parsed AST into a typed Universal IR,
`nflc parse <file> --uir` renders it via `Display` impls, and errors carry
source-snippets with `^` markers. All 5 M1 positive fixtures build to UIR,
all 7 M2 + 1 M3b negative fixtures correctly fail at the right stage.
~106 tests passing across lexer, parser, IR, and integration. `cargo build`
and `cargo clippy --all-targets -- -D warnings` both clean. `docs/language_reference/uir.md`
documents UIR semantics for contributors.

The immediate next step is **Milestone 4 — generic profile**: implement the
first architecture profile that consumes the UIR and emits scalar assembly for
any POSIX target. This is the first time NeuralForge produces real
machine-executable output.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status                       # confirm only the two .md files staged
git commit -m "chore(m3c): close Milestone 3c — UIR polish shipped; M3 fully complete

Adds M3c close-out entry to DEVLOG with the project principle
formalised ('Add code only when there's a real consumer'),
audit results (WrongInputCount kept defensively-reachable), and
the M3 → M4 transition note.

Updates CLAUDE.md Current Status to reflect Milestone 3 fully
complete and Milestone 4 (generic profile / scalar assembly
codegen) as the next milestone — the first time NeuralForge
will produce real machine-executable output.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 7, Milestone 3c (and all of Milestone 3) is complete by the spec's
acceptance criteria:

1. ✅ Workspace builds clean — Tasks 1-7
2. ✅ Clippy clean — Task 4, verified Task 7
3. ✅ Display impls for 7 types — Tasks 1, 2
4. ✅ `main.rs` shrunk: no `print_uir` machinery, just `print!("{}", uir)` — Task 3
5. ✅ `render_error_with_snippet` wired — Task 3
6. ✅ `docs/language_reference/uir.md` exists — Task 6
7. ✅ Audit complete + documented — Tasks 5, 7
8. ✅ ~106 tests passing — Tasks 1-4, verified Task 7
9. ✅ CLI smoke positive — Tasks 3, 7
10. ✅ CLI smoke negative w/ snippet — Tasks 3, 7
11. ✅ DEVLOG entry — Task 7
12. ✅ CLAUDE.md updated — Task 7

**Optional follow-up (recommended before M4):** push `claude/m3c-uir-polish` and
open a PR. Title suggestion: "Implement Milestone 3c: UIR polish — Display impls,
source-snippets, reference doc". After merge, M3 is fully closed.

**The Milestone 4 entry-point** is a fresh `superpowers:brainstorming` cycle to
design the generic profile. First M4 decision is assembly target syntax (AT&T
GAS, Intel NASM, or LLVM IR as a stepping stone).
