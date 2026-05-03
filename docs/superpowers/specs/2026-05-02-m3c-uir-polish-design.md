# NFL UIR — Vertical Slice 3 (Milestone 3c) — Design Spec

> **Status:** Approved (brainstorming output, 2026-05-02)
> **Authoritative for:** Milestone 3c implementation
> **Source skill:** `superpowers:brainstorming`
> **Next skill:** `superpowers:writing-plans`
> **Builds on:** [M3a](./2026-05-02-m3a-uir-tiny-mlp-design.md), [M3b](./2026-05-02-m3b-uir-all-fixtures-design.md)
> **Closes:** Milestone 3 (Universal IR). After M3c lands, the project moves to **M4 — generic profile (scalar assembly codegen)**.

---

## 1. Context

Milestone 3 was decomposed into three vertical slices:
- **M3a** (shipped): UIR end-to-end for `tiny_mlp.nfl`; module scaffolded
- **M3b** (shipped, PR #5 open): UIR for all 5 M1 fixtures + dropout validation + `--uir` CLI
- **M3c (this spec)**: UIR polish — final slice that closes Milestone 3

After M3c the codebase has a fully-shaped UIR pipeline that's clean, well-documented, and lint-free, ready to be consumed by the first architecture profile in M4.

**Reading order for context:**
1. M3a spec — UIR data model and stdlib design
2. M3b spec — extension to all fixtures + Symbol-resolution fix-up
3. M3b DEVLOG entry (2026-05-02 second-from-top) — known tech debt items #1-#6
4. `compiler/src/main.rs` (M3b) — the `print_uir` / `print_uir_node` / `format_uir_*` free-function logic that M3c moves onto types
5. `compiler/src/ir/{types,stdlib,build,error}.rs` — current code state to polish

---

## 2. Scope

### In scope (Milestone 3c)

| # | Item | Type |
|---|---|---|
| 1 | `Display` impls for UIR types: `Uir`, `UirModel`, `Node`, `Shape`, `OpAttr`, `AttrValue` | Refactor |
| 2 | `Display for StdOp` (replaces `format!("{:?}", std_op)` in `BuildError::invalid_attr_value` messages) | Refactor |
| 3 | Hand-rolled source-snippet error formatter (no external `ariadne` crate) | New logic |
| 4 | `docs/language_reference/uir.md` documenting UIR semantics + multi-pipeline convention | Docs |
| 5 | Clippy cleanup (3× `cloned_ref_to_slice_refs`, 1× `match_like_matches_macro`) | Cleanup |
| 6 | Audit unused enum variants — REMOVE rather than `#[allow(dead_code)]` | Cleanup |

### Project principle (formalised in this spec)

> **Add code only when there's a real consumer.** Do not retain "for-future-use"
> variants/functions/types via `#[allow(dead_code)]`. Remove unused items when
> discovered; re-introduce with the first real use (with tests). Speculative code
> violates "explicit over implicit" and rots the codebase.

This applies project-wide. **Important nuance:** "no real consumer" means *no caller
at all*, not "unreached in current tests". A variant constructed by a defensive
helper that protects against future caller bugs (e.g., `single_input` returning
`WrongInputCount` if a future op passes the wrong number of inputs) IS a real
consumer — defensive guards are intentional code, not dead code. The audit must
distinguish:

- **No-consumer dead code** — nothing constructs/calls it anywhere → remove
- **Defensively-reachable code** — constructed by a guard helper, just not fired by
  current tests → keep, document in DEVLOG why

In M3c the audit will pass through every variant and document each as one or the
other.

### Out of scope — deferred further

- **Ariadne crate dependency** — chose hand-roll instead (preserves "std-only" philosophy)
- **AttrError + ShapeError unification** — only 2 enums today, no growth signal; v0.2
- **Multi-error reporting** — first-error-halt continues; v0.2
- **CI / GitHub Actions** — small follow-up PR before M4 lands (not blocking M3c)
- **M3a tech-debt items #1-#4** (TypeExpr.name, Span start-only, no CI, crate version)
  — discuss before v1.0
- **Property-based testing, fuzzing** — v0.2+

---

## 3. Deliverables

**Modify:**

| Path | Change |
|---|---|
| `compiler/src/ir/types.rs` | Add `Display` impls for `Uir`, `UirModel`, `Node`, `Shape`, `OpAttr`, `AttrValue` |
| `compiler/src/ir/stdlib.rs` | Add `Display for StdOp`; remove `ShapeError::WrongInputCount` variant + its construction site (in `single_input` helper); the `single_input` check for `inputs.len() != 1` becomes a `panic!` (lexer/parser invariant — caller always passes single input in v0.1) OR is removed entirely if dead too |
| `compiler/src/ir/build.rs` | Replace `format!("{:?}", std_op)` with `format!("{}", std_op)` in `BuildError::invalid_attr_value`; convert `check_arg_type`'s `match` to `matches!` macro |
| `compiler/src/ir/tests.rs` | 3× `&[input.clone()]` → `std::slice::from_ref(&input)` in `infer_softmax_and_dropout_preserve_shape` and `infer_relu_preserves_shape` |
| `compiler/src/main.rs` | Remove `print_uir`, `print_uir_node`, `format_uir_shape`, `format_uir_attr` free functions; replace with `println!("{}", uir)` in `run_build_uir`; add `render_error_with_snippet` helper; route all error paths through it |
| `DEVLOG.md` | Append M3c close-out entry (with audit results) |
| `CLAUDE.md` | Update "Current Status" → M3 fully complete, M4 next |

**Create:**

| Path | Purpose |
|---|---|
| `docs/language_reference/uir.md` | UIR semantics reference (~150-200 lines) |

**Do NOT touch:**
- M2 fixtures, parser, lexer — frozen
- M3a/M3b specs, plans, DEVLOG entries — frozen
- M3a/M3b integration test fixtures — frozen
- Public API surface of `nflc::ir` — `Display` impls are additions, not breaking changes; the `BuildError::invalid_attr_value` message text change is acceptable (no consumer asserts on exact text)

---

## 4. Architecture

No structural changes. Module layout from M3a/M3b is stable:

```
compiler/src/ir/
├── mod.rs                               unchanged
├── types.rs                             + Display impls
├── stdlib.rs                            + Display for StdOp; - WrongInputCount
├── build.rs                             text-only changes (format string + matches!)
├── error.rs                             unchanged
└── tests.rs                             clippy-style fixes
```

The CLI (`compiler/src/main.rs`) shrinks: ~50 lines of free-function printing replaced
by ~3 lines of `println!("{}", …)`. The new `render_error_with_snippet` adds ~20 lines.
Net: fewer lines, more responsibility on the types themselves (where it belongs).

---

## 5. Components

### 5.1 `Display` impls

`compiler/src/ir/types.rs` gains six `Display` impls. Output format is identical to
M3b's `print_uir` so all M3a/M3b CLI smoke checks remain valid:

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
            AttrValue::Float(v)   => write!(f, "{}", v),
            AttrValue::Symbol(s)  => write!(f, "{}", s),
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
                let ops_s = operands.iter()
                    .map(|o| format!("n{}", o)).collect::<Vec<_>>().join(", ");
                write!(f, "{}           :: {}    operands=[{}]", op, self.ty.shape, ops_s)?;
                if !attrs.is_empty() {
                    let a = attrs.iter()
                        .map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
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
        let inputs = self.inputs.iter()
            .map(|i| format!("n{}", i)).collect::<Vec<_>>().join(", ");
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

### 5.2 `Display for StdOp`

`compiler/src/ir/stdlib.rs`:

```rust
impl std::fmt::Display for StdOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            StdOp::Linear  => "linear",
            StdOp::Relu    => "relu",
            StdOp::Dropout => "dropout",
            StdOp::Softmax => "softmax",
        };
        write!(f, "{}", name)
    }
}
```

In `compiler/src/ir/build.rs`, the `BuildError::invalid_attr_value(...)` call inside
`build_op` changes:

```rust
// Before (M3b):
BuildError::invalid_attr_value(&format!("{:?}", std_op), attr_name, ...)
// After (M3c):
BuildError::invalid_attr_value(&format!("{}", std_op), attr_name, ...)
```

Effect: error message becomes `"invalid value for dropout.rate: ..."` (lowercase,
matches the source NFL token name) instead of `"invalid value for Dropout.rate: ..."`.

The `dropout_rate_out_of_range_rejected` test asserts only on `BuildErrorKind` variant
and `err.line`, not on message content — no test update needed.

### 5.3 Hand-rolled source-snippet error formatter

In `compiler/src/main.rs`, add:

```rust
/// Render an error with a source-snippet pointer. Output:
///
/// ```
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
        for _ in 1..col { underline.push(' '); }
        underline.push('^');
        eprintln!("{}  | {}", pad, underline);
    }
}
```

All error paths in `run_parse` and `run_build_uir` switch to this:

```rust
// Before (M3b):
eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
// After (M3c):
render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
```

### 5.4 Clippy cleanup

`compiler/src/ir/tests.rs` — three `&[input.clone()]` sites become
`std::slice::from_ref(&input)` (no allocation, semantically identical).

`compiler/src/ir/build.rs` `check_arg_type`:

```rust
// Before:
let ok = match (slot.ty, value) {
    (ArgType::Integer, ArgValue::Integer(_)) => true,
    (ArgType::Float, ArgValue::Float(_)) => true,
    (ArgType::Symbol, ArgValue::Symbol(_)) => true,
    _ => false,
};
// After:
let ok = matches!(
    (slot.ty, value),
    (ArgType::Integer, ArgValue::Integer(_))
        | (ArgType::Float, ArgValue::Float(_))
        | (ArgType::Symbol, ArgValue::Symbol(_)),
);
```

After both changes, `cargo clippy --all-targets -- -D warnings` exits 0.

### 5.5 Unused-variant audit

Strategy: temporarily add `#![deny(dead_code)]` at the top of each module, run
`cargo build --tests`, classify each finding per §2's nuance:

- **Truly dead** (no constructor/no caller anywhere) → remove
- **Defensively reachable** (constructor exists in a guard helper but no test fires
  the guard) → keep, justify in DEVLOG

**Initial classification of `ShapeError::WrongInputCount`:**

- Constructor: `single_input` helper in `stdlib.rs` returns it when `inputs.len() != 1`
- All v0.1 ops are single-input → `single_input` is always called with a one-element
  slice in current code → guard never fires in tests
- BUT: the guard catches a class of caller bugs (a future `build_op` change that
  accidentally passes 2 inputs to a single-input op would error gracefully rather
  than panic on `&inputs[0]` index)
- **Decision: KEEP** as defensively reachable. Document this classification in DEVLOG.

Other variants/functions go through the same classification. The audit's findings
(every removed item, every kept item with justification) land in the M3c DEVLOG entry
under "Audit results" — making the policy auditable for future contributors.

---

## 6. Testing strategy

### Existing tests

All 102 tests must pass unchanged after M3c. The Display refactor doesn't change the
CLI output content (only moves the formatting code), and the message-text change in
`InvalidAttrValue` doesn't break any test that asserts on `BuildErrorKind` variant or
position.

### New tests (~3-5)

In `compiler/src/ir/tests.rs`, add small Display-roundtrip tests:

```rust
#[test]
fn shape_displays_as_tensor_with_dims() {
    let s = Shape(vec![32, 784]);
    assert_eq!(format!("{}", s), "Tensor[32, 784]");
}

#[test]
fn stdop_displays_lowercase_name() {
    assert_eq!(format!("{}", StdOp::Linear), "linear");
    assert_eq!(format!("{}", StdOp::Dropout), "dropout");
}

#[test]
fn attrvalue_displays_each_variant() {
    assert_eq!(format!("{}", AttrValue::Integer(42)), "42");
    assert_eq!(format!("{}", AttrValue::Float(0.5)), "0.5");
    assert_eq!(format!("{}", AttrValue::Symbol("true".into())), "true");
}
```

(Display for `Uir`, `UirModel`, `Node` is exercised by the existing CLI smoke checks
in §7.9-§7.10 — they confirm the visible output is correct.)

---

## 7. Acceptance criteria

Milestone 3c is **complete** when all of the following hold:

1. **Workspace builds clean** — `cargo build` zero errors, zero warnings.
2. **Clippy clean** — `cargo clippy --all-targets -- -D warnings` exits 0.
3. **`Display` impls** exist for `Uir`, `UirModel`, `Node`, `Shape`, `OpAttr`, `AttrValue`, `StdOp`.
4. **`compiler/src/main.rs`** no longer contains `print_uir`, `print_uir_node`,
   `format_uir_shape`, or `format_uir_attr` — replaced by `println!("{}", uir)`.
5. **`render_error_with_snippet`** is called from all CLI error paths (`run_parse`
   error branches, `run_build_uir` error branches).
6. **`docs/language_reference/uir.md`** exists, ~150-200 lines, covers the seven
   sections from §3.1.
7. **Audit complete and documented** — DEVLOG lists every item the audit examined,
   classified as "removed (truly dead)" or "kept (defensively reachable, justification:
   …)". Initial classification per §5.5 is `ShapeError::WrongInputCount` → KEEP; the
   audit must verify this and inspect all other variants the same way.
8. **`cargo test`** is green at ~105 tests (102 prior + ~3 new Display tests).
9. **CLI smoke positive:** `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --uir`
   produces output identical to M3b (8-node UIR with named float `rate=0.2`,
   resolved `out_dim=10`), exit 0.
10. **CLI smoke negative:** `cargo run --bin nflc -- parse tests/fixtures/negative/dropout_rate_out_of_range.nfl --uir`
    produces source-snippet output (with `^` under the offending value), exit 1.
11. **DEVLOG entry for M3c close** with the project-principle note (Add code only
    when there's a real consumer) and the audit results.
12. **`CLAUDE.md` "Current Status"** reflects M3 fully complete; M4 (generic profile)
    as next.

---

## 8. Deferred items

### Deferred to v0.2+ (or future milestones)

- AttrError + ShapeError unification (no growth signal yet)
- Multi-error reporting (continues from M2's first-error-halt)
- Property-based testing, fuzzing
- CI / GitHub Actions (small follow-up PR before M4 ships)
- All M3a tech-debt items #1-#4 (TypeExpr.name String, Span start-only, no CI, crate version)

### Deferred to M4 (next milestone)

- Generic profile — first scalar-assembly codegen consumer of the UIR

### Deferred to M5+

- UIR mutation API for kernel fusion
- Multi-input ops (`add`, `concat`) — at which point the `single_input` guard's
  `WrongInputCount` path will finally be exercised by tests (rather than just
  defensively reachable as it is in v0.1)

---

## 9. Open questions / known tech debt

These are NOT blockers for M3c implementation. **Must** be logged in the DEVLOG
entry that closes M3c so they remain visible:

1. **Multi-error reporting** still deferred. v0.2 question.
2. **AttrError + ShapeError** still two enums. Unification is a v0.2 consideration if
   pattern grows.
3. **No CI** — separate small PR before M4.
4. **`single_input` defensive guard** — kept (per §5.5). M5 will exercise it for
   real when multi-input ops (`add`, `concat`) appear. The guard's error variant
   `WrongInputCount` continues to be defensively reachable until then.

---

## 10. Transition

After this spec is reviewed and approved, transition to `superpowers:writing-plans`
for the implementation plan. M3c is small (6 items, mostly mechanical refactors) so
the plan should be ~6-8 tasks.
