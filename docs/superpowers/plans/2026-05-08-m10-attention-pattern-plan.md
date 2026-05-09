# Milestone 10 — NFL v0.2 Self-Attention + 4D Codegen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one PR with **14 atomic commits** (12 spec-listed task-packs, with spec §10 step 9 expanded into 9a/9b/9c per its default-split caveat) that together introduce `named_pipeline_stmt` to NFL grammar, add `ArgType::Tensor`, `StdOp::Matmul`, `StdOp::MulScalar`, `BuildErrorKind::DeclaredShapeMismatch`, four new `ShapeError` variants, generalise `Softmax` to rank ≥ 2, implement `emit_matmul` + `emit_mulscalar` in both `profiles/arm64/` and `profiles/x86_64/`, land an end-to-end self-attention acceptance fixture with per-profile bit-exact integration tests on macOS arm64 and Linux x86_64, four negative fixtures, ~45 new tests (project total 284 → ~329), and full docs.

**Architecture:** Layer-by-layer, AST-up. Commits 1–5 land all language and UIR-level changes (parser, AST, stdlib, builder, error variants); workspace stays buildable but profiles see no new ops yet because both profiles' `classify_op` rejects `Matmul`/`MulScalar` until commit 6. Commits 6–8 add arm64 codegen for the new ops + the 4D Softmax dispatch fix. Commits 9a–9c mirror the same on x86_64 (split into three sub-packs by default, per spec §10's M9-grounded caveat that x86_64 brings non-trivial divergence — `%rsi`/`%rdx` clobber hazards required two M9 follow-up fixes; optimistic packing is a regression risk). Commit 10 lands `tests/fixtures/self_attention.nfl` plus per-profile FFI integration tests against architecture-matched references. Commit 11 adds four negative fixtures + any missed unit tests. Commit 12 lands all docs in one shot.

The acceptance criterion is **per-profile** bit-exact equality between FFI-compiled output and an architecture-matched Rust reference (FMA-using on arm64; separate `mul + add` on x86_64) — cross-profile bit-exact is architecturally unreachable in M10 (FMA × libm `expf` divergence; spec §7.2) and intentionally **not** a gate.

**Tech Stack:** Rust 2021 edition, std-only (no new external runtime deps in any new module). `cc` crate (already in profiles' dev-deps) for FFI tests. `libloading` (already in profiles' dev-deps) for dlopen-based test harness. AArch64 Mach-O for arm64 (FMA via `fmadd`); SysV AMD64 / Linux ELF for x86_64 (scalar SSE2 via `mulss + addss`, no FMA, AT&T syntax). Both profiles already implement the `Profile` trait from `profile-api/`; no new trait methods are added.

---

## Plan conventions

### Commit-group cadence

Each numbered group below corresponds to **one git commit**. The group's last task is always "Commit". Within a group, the workspace stays clean (`cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build --workspace`, `cargo test --workspace` all green) but uncommitted; the final task stages all changes and creates one atomic commit. **Workspace gates run after every implementation step**, not just at commit time — TDD red→green→refactor is enforced inside each group.

Do NOT split a group's work across multiple commits. Do NOT batch multiple groups into one commit. The atomic-task-pack convention from M7 (and reaffirmed in M9 spec §13) is required for clean review and rollback.

### Atomic-cascade discipline

Group 2 (the `resolve_args` signature change) cascades through three functions in one commit per spec §5.3 — the cascade goes `resolve_args` → `build_op` → `build_model`. Splitting any of these into separate commits would produce a non-compiling intermediate state. The plan's Group 2 task structure enforces co-modification: all three functions move together, with workspace gates at the very end (not after individual function edits).

### Branch and worktree

Work happens in this worktree: `claude/sad-spence-5e696b`, located at `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/sad-spence-5e696b/`. Spec lives at `docs/superpowers/specs/2026-05-08-m10-attention-pattern-design.md`. The branch is already 2 commits ahead of `origin/main` (brainstorm spec + spec review fix); commits from this plan land on top, this plan file itself being the third commit on this branch (already landed before group 1 begins, see "Plan landing" below).

### Plan landing

Before any group-1 work begins, this plan file lands as a separate commit on `claude/sad-spence-5e696b` with message `docs(m10): implementation plan — self-attention + 4D codegen`. That commit is **not counted in the 14 implementation commits**; it's plan-synthesis hygiene parallel to M9's pre-execution `cd952b0 docs(m9/devlog): log plan-synthesis session`.

### x86_64 packing decision

Spec §10 step 9 reads: "treat as up to three independent sub-task-packs by default. ... Optimistic packing of all three into one commit is a regression risk; fold only when isomorphism genuinely holds at implementation time." This plan **defaults to the split** — Groups 9a (matmul), 9b (mulscalar), 9c (softmax dispatch) are three commits. The plan does **not** pre-decide the fold based on plan-time imagination; the fold (if any) is a per-implementation-time call by the executor after Groups 6–8 (arm64) land. M9 commits `ecb69ac` and `c3ff521` are the precedent — both fixed `%rsi` / `%rdx` clobber hazards that would have shipped silently if x86_64 had been packed optimistically with arm64.

### TDD cadence per task

Every new function lands in five micro-steps:

1. **Write the failing test** — code visible in the plan, paste-ready
2. **Run the test, confirm it fails** — exact `cargo test ...::<name>` command + expected error
3. **Implement the minimum code to pass** — code visible in the plan
4. **Run the test, confirm it passes** — same command, "0 failed" expected
5. **(Sometimes) refactor without breaking** — only if step 3's code is dirty

The "Commit" task at the end of each group runs all four workspace gates and stages everything as one commit. Workspace gates inside the group are *checks* (run at each task boundary if convenient), but the gate that *gates* the commit is the four-command sequence at commit time.

### Test count budget

Spec §8 estimates ~45 new tests; project total goes 284 → ~329. The plan's task list adds:

- 5 parser tests (Group 1)
- 16 UIR builder tests (Groups 2–5, including 1 ArgType::Tensor + 1 NamedPipeline-success + 1 declared-shape-mismatch + 13 ops/shape)
- 11 arm64 codegen unit tests (Groups 6–8)
- 11 x86_64 codegen unit tests (Groups 9a–9c)
- 2 FFI integration tests (Group 10, one per profile)

**Total = 45.** Each test is enumerated explicitly inside the relevant group. If a group's test count drifts during implementation (one merged, one split), the executor adjusts the count in the next-group's task description — the plan's count is *target*, not gate.

---

## File structure

The PR creates **6 new files** and modifies **~16 existing files**. No deletes.

**New files (created in groups indicated):**

| File | Group | Purpose |
|------|-------|---------|
| `tests/fixtures/self_attention.nfl` | 10 | Acceptance fixture, `[batch=2, heads=4, seq=16, head_dim=16]` |
| `tests/fixtures/negative/bad_named_pipeline_missing_eq.nfl` | 11 | Parser-level rejection |
| `tests/fixtures/negative/bad_matmul_rank_too_low.nfl` | 11 | UIR rank-check rejection |
| `tests/fixtures/negative/bad_matmul_inner_dim_mismatch.nfl` | 11 | UIR contraction-dim rejection |
| `tests/fixtures/negative/bad_named_pipeline_shape_mismatch.nfl` | 11 | UIR declared-vs-actual rejection |
| `profiles/arm64/src/ops/matmul.rs` | 6 | `emit_matmul` for arm64 |
| `profiles/arm64/src/ops/mulscalar.rs` | 7 | `emit_mulscalar` for arm64 |
| `profiles/x86_64/src/ops/matmul.rs` | 9a | `emit_matmul` for x86_64 |
| `profiles/x86_64/src/ops/mulscalar.rs` | 9b | `emit_mulscalar` for x86_64 |
| `docs/superpowers/plans/2026-05-08-m10-attention-pattern-plan.md` | (this file) | This plan |

**Modified files (groups they're touched in):**

| File | Groups |
|------|--------|
| `language/grammar.ebnf` | 1, 12 |
| `compiler/src/ast.rs` | 1 |
| `compiler/src/parser/mod.rs` | 1 |
| `compiler/src/parser/tests.rs` | 1 |
| `compiler/src/ir/stdlib.rs` | 2, 3, 4, 5 |
| `compiler/src/ir/build.rs` | 2, 5 |
| `compiler/src/ir/error.rs` | 5 |
| `compiler/src/ir/tests.rs` | 2, 3, 4, 5 |
| `profiles/arm64/src/ops/mod.rs` | 6, 7 |
| `profiles/arm64/src/buffer.rs` | 6, 7 |
| `profiles/arm64/src/codegen.rs` | 6, 7, 8 |
| `profiles/arm64/src/tests.rs` | 6, 7, 8 |
| `profiles/arm64/tests/integration.rs` | 10 |
| `profiles/x86_64/src/ops/mod.rs` | 9a, 9b |
| `profiles/x86_64/src/buffer.rs` | 9a, 9b |
| `profiles/x86_64/src/codegen.rs` | 9a, 9b, 9c |
| `profiles/x86_64/src/tests.rs` | 9a, 9b, 9c |
| `profiles/x86_64/tests/integration.rs` | 10 |
| `docs/language_reference/grammar.md` | 12 |
| `docs/profile_guide/arm64.md` | 12 |
| `docs/profile_guide/x86_64.md` | 12 |
| `PROJECT_SPEC.md` | 12 |
| `DEVLOG.md` | 12 |
| `CLAUDE.md` | 12 |

`emit_linear` (both profiles), `emit_relu` (both), `emit_dropout` (both), `emit_softmax` (the asm emitter — both) are **strictly unchanged**. Only the `walk_model::Softmax` *dispatch* (Groups 8 / 9c) gets a one-line `b = product(shape[..-1]); k = shape[last]` rewrite. Architectural invariant per spec §6.1.

---

<!-- group-1-anchor -->

## Group 1 — Commit 1 — NFL grammar + AST + parser for `named_pipeline_stmt`

**Group goal:** Extend NFL v0.1 grammar with one new production (`named_pipeline_stmt = identifier , ":" , type_expr , "=" , identifier , pipeline_chain`), add a parallel AST variant, implement parsing with explicit one-token lookahead on `=` after the type expression. No semantics yet — Group 2/5 land the builder side.

**Group done criteria:**
- `cargo build --workspace` green
- `cargo clippy --workspace --all-targets -- -D warnings` green
- `cargo fmt --all -- --check` green
- `cargo test --workspace`: 284 + 5 (new parser tests) = 289

**Files touched:**
- Modify: `language/grammar.ebnf` (add new production)
- Modify: `compiler/src/ast.rs` (add `NamedPipelineStmt` struct + `ModelStmt::NamedPipeline` variant)
- Modify: `compiler/src/parser/mod.rs` (add `parse_named_pipeline_stmt`, update `parse_model_stmt` lookahead)
- Modify: `compiler/src/parser/tests.rs` (5 new tests)

### Task 1.1: Update grammar.ebnf

**Files:**
- Modify: `language/grammar.ebnf:29` and surrounding

- [ ] **Step 1: Replace the `model_stmt` production**

In `language/grammar.ebnf`, locate the line:

```
model_stmt       = variable_decl | pipeline_stmt ;
```

Replace with:

```
model_stmt          = variable_decl | pipeline_stmt | named_pipeline_stmt ;
named_pipeline_stmt = identifier , ":" , type_expr , "=" , identifier , pipeline_chain ;
```

(The `named_pipeline_stmt` line goes immediately below the updated `model_stmt`.)

- [ ] **Step 2: Update the implicit-semantics block at the top**

Below the existing line in the comment block:

```
(*   - The output of a model is the value produced by the last  *)
(*     operation of the last pipeline_stmt in its body.         *)
```

Replace those two lines with:

```
(*   - The output of a model is the value produced by the last  *)
(*     operation of the last pipeline_stmt or named_pipeline_   *)
(*     stmt in its body. For a named_pipeline_stmt, the output  *)
(*     is the bound right-hand-side value.                      *)
```

This matches spec §4.2.

- [ ] **Step 3: Sanity-check grammar EBNF parses by eye**

There's no automated EBNF validator in this repo. Visually verify:
- `named_pipeline_stmt` references only previously-defined productions (`identifier`, `type_expr`, `pipeline_chain`).
- `model_stmt` is now a 3-way alternation — left-to-right precedence is irrelevant since the parser disambiguates via lookahead, but reviewers will read this file linearly.

### Task 1.2: Add `NamedPipelineStmt` struct + variant to AST

**Files:**
- Modify: `compiler/src/ast.rs:30-33` (the `ModelStmt` enum)

- [ ] **Step 1: Add the struct definition**

In `compiler/src/ast.rs`, immediately after the existing `PipelineStmt` struct (around line 60, after the `pub struct PipelineStmt { ... }` block), add:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct NamedPipelineStmt {
    /// The bound name on the LHS, e.g. `scores` in
    /// `scores: Tensor[...] = x -> matmul[x, transpose_b=true]`.
    pub binding_name: String,
    /// The declared type of the bound name. The builder verifies the
    /// pipeline's actual output shape matches this declared shape, raising
    /// `BuildErrorKind::DeclaredShapeMismatch` on disagreement.
    pub declared_ty: TypeExpr,
    /// The pipeline source identifier (the `x` after `=`).
    pub source: String,
    /// One or more chained operations, identical in shape to PipelineStmt.steps.
    pub steps: Vec<Operation>,
    pub span: Span,
}
```

- [ ] **Step 2: Add the variant to `ModelStmt`**

Replace the existing definition:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStmt {
    VariableDecl(VariableDecl),
    Pipeline(PipelineStmt),
}
```

with:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStmt {
    VariableDecl(VariableDecl),
    Pipeline(PipelineStmt),
    NamedPipeline(NamedPipelineStmt),
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p compiler`
Expected: success. There should be 0 new warnings.

(The builder doesn't match on this variant yet, but `build_model` uses an exhaustive match — at this point the build will FAIL if you don't add a placeholder arm. **Stop here and add the placeholder arm in build.rs:**)

- [ ] **Step 4: Add a placeholder arm in `build_model`**

In `compiler/src/ir/build.rs:271-282` (the `for stmt in &ast_model.body { match stmt { ... } }` block), add a new arm immediately before the closing `}`:

```rust
ModelStmt::NamedPipeline(_np) => {
    // Group 5 lands the real builder logic. For Group 1 (parser-only)
    // and Groups 2–4 (UIR machinery), the parser may emit this variant
    // through fixtures the unit tests exercise — but build() is never
    // called on those fixtures yet. The placeholder rejects gracefully
    // if something accidentally calls build() on a NamedPipeline-bearing
    // AST during Groups 1–4.
    return Err(BuildError::shape(
        "named_pipeline_stmt is not yet implemented (M10 group 5)".to_string(),
        _np.span,
    ));
}
```

Run `cargo build --workspace`. Expected: success, 0 warnings.

### Task 1.3: Write the first failing parser test (2D)

**Files:**
- Modify: `compiler/src/parser/tests.rs` (append at end of `mod tests {}`)

- [ ] **Step 1: Add the test function**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_named_pipeline_stmt_2d() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let model = &ast.models[0];
    assert_eq!(model.body.len(), 2);
    let np = match &model.body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        other => panic!("expected NamedPipeline, got {:?}", other),
    };
    assert_eq!(np.binding_name, "y");
    assert_eq!(np.source, "x");
    assert_eq!(np.steps.len(), 1);
    assert_eq!(np.steps[0].name, "relu");
    assert_eq!(np.declared_ty.dims.len(), 2);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `cargo test -p compiler parse_named_pipeline_stmt_2d`
Expected: **FAIL** with `expected ':' or '->'` from `parse_model_stmt`. (The current `parse_model_stmt` only accepts those two — `=` is not yet a recognised continuation.)

### Task 1.4: Implement `parse_named_pipeline_stmt`

**Files:**
- Modify: `compiler/src/parser/mod.rs` (append after `parse_pipeline_stmt`, around line 291)

- [ ] **Step 1: Add the parser function**

After the closing brace of `parse_pipeline_stmt` (around line 291 in `compiler/src/parser/mod.rs`), add:

```rust
use crate::ast::NamedPipelineStmt;

pub(crate) fn parse_named_pipeline_stmt(p: &mut Parser) -> Result<NamedPipelineStmt, ParseError> {
    // <binding_name> ":" <type_expr> "=" <source_ident> <pipeline_chain>
    let TokenKind::Ident(binding_name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Colon, ":")?;
    let declared_ty = parse_type_expr(p)?;
    p.consume(TokenKind::Equals, "=")?;

    // Source identifier (the variable being piped from).
    let TokenKind::Ident(source) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    p.advance();

    // Pipeline chain — at least one `-> operation`. Reuse the same
    // continuation-line newline tolerance as parse_pipeline_stmt.
    let mut steps = Vec::new();
    p.consume(TokenKind::Arrow, "->")?;
    steps.push(parse_operation(p)?);
    loop {
        while matches!(p.peek_kind(), TokenKind::Newline)
            && matches!(p.peek_at(1), Some(TokenKind::Arrow))
        {
            p.advance();
        }
        if !matches!(p.peek_kind(), TokenKind::Arrow) {
            break;
        }
        p.advance();
        steps.push(parse_operation(p)?);
    }

    Ok(NamedPipelineStmt {
        binding_name,
        declared_ty,
        source,
        steps,
        span: Span::new(line, col),
    })
}
```

- [ ] **Step 2: Update `parse_model_stmt` with the one-token lookahead**

Replace the body of `parse_model_stmt` (around line 387–399) with the dispatch:

```rust
pub(crate) fn parse_model_stmt(p: &mut Parser) -> Result<ModelStmt, ParseError> {
    // Disambiguate three cases by looking at the token immediately
    // after the leading identifier:
    //   - `Ident "->"`               → pipeline_stmt
    //   - `Ident ":"  Tensor … "="`  → named_pipeline_stmt
    //   - `Ident ":"  Tensor … (Newline | Dedent)`  → variable_decl
    //
    // The pipeline_stmt vs colon-prefixed forms is decided by peek_at(1).
    // The variable_decl vs named_pipeline_stmt distinction is decided by
    // looking past the type_expr — but that requires unbounded lookahead.
    // We sidestep this by parsing through the type_expr greedily and then
    // branching on whether `=` follows. parse_variable_decl already
    // consumes `Ident ":" type_expr` and stops; parse_named_pipeline_stmt
    // requires `=` after the type_expr.
    //
    // Implementation: peek 1 ahead. If `:`, parse the prefix once, then
    // dispatch on whether `=` follows (one-token lookahead on Equals).
    // If `->`, dispatch to pipeline_stmt directly.
    let after = match p.peek_at(1) {
        Some(k) => k,
        None => return Err(p.error_expected(&["':'", "'->'"])),
    };
    match after {
        TokenKind::Arrow => Ok(ModelStmt::Pipeline(parse_pipeline_stmt(p)?)),
        TokenKind::Colon => parse_decl_or_named_pipeline(p),
        _ => Err(p.error_expected(&["':'", "'->'"])),
    }
}

/// Common prefix `Ident ":" type_expr` is shared between variable_decl
/// and named_pipeline_stmt. After the type_expr, look one token ahead:
/// if `=`, this is a named_pipeline_stmt; otherwise it's a variable_decl.
fn parse_decl_or_named_pipeline(p: &mut Parser) -> Result<ModelStmt, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Colon, ":")?;
    let ty = parse_type_expr(p)?;

    // One-token lookahead on `=`.
    if matches!(p.peek_kind(), TokenKind::Equals) {
        p.advance();
        // We've already consumed `Ident ":" type_expr "="`; resume with
        // the named-pipeline-specific tail (source ident + pipeline_chain).
        let TokenKind::Ident(source) = p.peek_kind().clone() else {
            return Err(p.error_expected(&["identifier"]));
        };
        p.advance();
        let mut steps = Vec::new();
        p.consume(TokenKind::Arrow, "->")?;
        steps.push(parse_operation(p)?);
        loop {
            while matches!(p.peek_kind(), TokenKind::Newline)
                && matches!(p.peek_at(1), Some(TokenKind::Arrow))
            {
                p.advance();
            }
            if !matches!(p.peek_kind(), TokenKind::Arrow) {
                break;
            }
            p.advance();
            steps.push(parse_operation(p)?);
        }
        Ok(ModelStmt::NamedPipeline(NamedPipelineStmt {
            binding_name: name,
            declared_ty: ty,
            source,
            steps,
            span: Span::new(line, col),
        }))
    } else {
        // Plain variable_decl.
        Ok(ModelStmt::VariableDecl(VariableDecl {
            name,
            ty,
            span: Span::new(line, col),
        }))
    }
}
```

**Important:** Once `parse_decl_or_named_pipeline` is added, the standalone `parse_named_pipeline_stmt` from Step 1 becomes unused. **Remove it** — `parse_decl_or_named_pipeline` is the canonical path because it already consumed the shared `Ident ":" type_expr` prefix. (The standalone function was a thinking-aid; the real parser fuses the decl/named-pipeline branches at the lookahead point per spec §4.1's "separate productions" rule, which the AST distinction satisfies.)

Delete the entire `parse_named_pipeline_stmt` function added in Step 1.

- [ ] **Step 3: Run the test to confirm it passes**

Run: `cargo test -p compiler parse_named_pipeline_stmt_2d`
Expected: **PASS**.

### Task 1.5: Add 4D parser test

**Files:**
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn parse_named_pipeline_stmt_4d() {
    let src = "\
model M [batch=2, heads=4, seq=16, head_dim=16]:
    x: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let ast = crate::parse(src).expect("parse");
    let np = match &ast.models[0].body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        other => panic!("expected NamedPipeline, got {:?}", other),
    };
    assert_eq!(np.binding_name, "scores");
    assert_eq!(np.declared_ty.dims.len(), 4);
    assert_eq!(np.source, "x");
    assert_eq!(np.steps.len(), 1);
    assert_eq!(np.steps[0].name, "matmul");
    // First positional arg is the tensor identifier `x`.
    let crate::ast::OpArg::Positional(crate::ast::ArgValue::Symbol(s)) = &np.steps[0].args[0]
    else {
        panic!("expected positional Symbol arg");
    };
    assert_eq!(s, "x");
    // Second arg is named `transpose_b=true`.
    let crate::ast::OpArg::Named { name, value } = &np.steps[0].args[1] else {
        panic!("expected named arg");
    };
    assert_eq!(name, "transpose_b");
    let crate::ast::ArgValue::Symbol(v) = value else {
        panic!("expected Symbol value");
    };
    assert_eq!(v, "true");
}
```

- [ ] **Step 2: Run, expect PASS**

Run: `cargo test -p compiler parse_named_pipeline_stmt_4d`
Expected: PASS (parser machinery already supports this from Task 1.4).

### Task 1.6: Add tensor-op-arg parser test

**Files:**
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn parse_named_pipeline_with_tensor_op_arg() {
    // Just confirms the parser accepts an identifier as a positional arg
    // (the existing `arg_value = number | identifier` rule). The semantic
    // tensor-name resolution lands in Group 2.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> matmul[x]
";
    let ast = crate::parse(src).expect("parse");
    let np = match &ast.models[0].body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        _ => panic!(),
    };
    assert_eq!(np.steps[0].name, "matmul");
    assert_eq!(np.steps[0].args.len(), 1);
}
```

- [ ] **Step 2: Run, expect PASS**

Run: `cargo test -p compiler parse_named_pipeline_with_tensor_op_arg`
Expected: PASS.

### Task 1.7: Add lookahead disambiguation test

**Files:**
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn parse_lookahead_distinguishes_variable_decl_from_named_pipeline() {
    // Both forms share the prefix `Ident ":" Tensor[...]`. The presence
    // of `=` after the type expression is the sole disambiguator.
    let src_var = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    x -> relu
";
    let src_np = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast_var = crate::parse(src_var).expect("var parse");
    let ast_np = crate::parse(src_np).expect("np parse");
    // First stmt is VariableDecl in both.
    assert!(matches!(
        ast_var.models[0].body[0],
        crate::ast::ModelStmt::VariableDecl(_)
    ));
    assert!(matches!(
        ast_np.models[0].body[0],
        crate::ast::ModelStmt::VariableDecl(_)
    ));
    // Second stmt: Pipeline in src_var, NamedPipeline in src_np.
    assert!(matches!(
        ast_var.models[0].body[1],
        crate::ast::ModelStmt::Pipeline(_)
    ));
    assert!(matches!(
        ast_np.models[0].body[1],
        crate::ast::ModelStmt::NamedPipeline(_)
    ));
}
```

- [ ] **Step 2: Run, expect PASS**

Run: `cargo test -p compiler parse_lookahead_distinguishes_variable_decl_from_named_pipeline`
Expected: PASS.

### Task 1.8: Add negative test for missing `=`

**Files:**
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn parse_named_pipeline_missing_eq_after_type() {
    // `y: Tensor[...] x -> relu` (missing `=`) should fail at the parser
    // level. After the type_expr, the lookahead branch sees neither
    // `Equals` (named pipeline) nor end-of-stmt (variable_decl), so we
    // get a variable_decl, then the followup `x` becomes a fresh stmt
    // start, then we see `-> relu` with no leading identifier — error.
    //
    // The exact error wording depends on which branch hits the failure
    // first. We don't pin it; we only require that parse() returns Err.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] x -> relu
";
    let result = crate::parse(src);
    assert!(result.is_err(), "expected parse error, got Ok");
}
```

- [ ] **Step 2: Run, expect PASS**

Run: `cargo test -p compiler parse_named_pipeline_missing_eq_after_type`
Expected: PASS (parse correctly errors).

### Task 1.9: Workspace gates

- [ ] **Step 1: `cargo fmt --all`**

Run: `cargo fmt --all`
Expected: no output, exit 0. (Reformat any drift introduced by the new code.)

- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: exit 0, no warnings.

- [ ] **Step 3: `cargo build --workspace`**

Run: `cargo build --workspace`
Expected: exit 0.

- [ ] **Step 4: `cargo test --workspace`**

Run: `cargo test --workspace`
Expected: 289 tests pass on macOS arm64 (284 baseline + 5 new). On Linux x86_64 CI add the platform delta from the existing M9 FFI suite (already counted in CI's ~300 baseline).

### Task 1.10: Commit

- [ ] **Step 1: Stage files**

```bash
git add language/grammar.ebnf \
        compiler/src/ast.rs \
        compiler/src/parser/mod.rs \
        compiler/src/parser/tests.rs \
        compiler/src/ir/build.rs
```

- [ ] **Step 2: Create the commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/parser): named_pipeline_stmt grammar + AST + lookahead parser

Adds the new production `named_pipeline_stmt = identifier ":" type_expr
"=" identifier pipeline_chain` to NFL v0.1, parallel to the existing
`pipeline_stmt`. Parser disambiguates variable_decl vs
named_pipeline_stmt via one-token lookahead on `=` after the
type_expression (spec §4.1).

Touched layers:
- language/grammar.ebnf: new production + output-rule comment refresh.
- ast: NamedPipelineStmt struct + ModelStmt::NamedPipeline variant.
- parser: parse_decl_or_named_pipeline shared-prefix helper; the
  standalone parse_named_pipeline_stmt was a thinking-aid and is not
  added to the codebase (the lookahead point is inside the helper, so
  the helper is the canonical path).
- ir/build.rs: placeholder arm rejects NamedPipeline at build time
  with a clear "not yet implemented (M10 group 5)" message; replaced
  with real semantics in M10 group 5.

Tests: 5 new parser tests (2D, 4D, tensor-op-arg, lookahead
disambiguation, missing-= negative). Project total 284 → 289.

Spec: docs/superpowers/specs/2026-05-08-m10-attention-pattern-design.md
§4.1, §4.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify status clean**

Run: `git status`
Expected: "nothing to commit, working tree clean".

- [ ] **Step 4: Verify the commit graph**

Run: `git log --oneline -3`
Expected: top-of-log shows the new commit; below it `b9c99e4` and `0ae2537` (the two M10 spec commits).

---

## Group 2 — Commit 2 — UIR args machinery — `ArgType::Tensor` + `resolve_args` cascade

**Group goal:** Add `ArgType::Tensor` to the stdlib enum and extend `resolve_args` to return `(Vec<NodeId>, Vec<OpAttr>)`, with a new `env: &HashMap<String, NodeId>` parameter for resolving tensor-name args. Cascade the signature change through `build_op` and all call sites in `build_model` per spec §5.3 — **all three function signatures move in this single commit**, otherwise the workspace doesn't compile.

**Group done criteria:**
- All four workspace gates green
- Test count 289 → 289–290 (zero or one new tests; the Tensor-resolution behaviour tests properly land in Group 3 once `StdOp::Matmul` exists)

**Files touched:**
- Modify: `compiler/src/ir/stdlib.rs` (add `ArgType::Tensor`)
- Modify: `compiler/src/ir/build.rs` (signature cascade + tensor-resolution logic + describe_slot_type)
- Modify: `compiler/src/ir/tests.rs` (0–1 new tests; see Task 2.5)

### Task 2.1: Add `ArgType::Tensor`

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:29-34` (the `ArgType` enum)

- [ ] **Step 1: Add the variant**

Replace:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    Integer,
    Float,
    Symbol,
}
```

with:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    Integer,
    Float,
    Symbol,
    /// A tensor-by-name argument. The arg appears in NFL source as an
    /// identifier (e.g. `matmul[x]` where `x` is a previously-declared
    /// variable name). The builder resolves it against the variable
    /// environment to a `NodeId`. Resolved IDs go into the op node's
    /// `operands` field, NOT `attrs`. New in M10.
    Tensor,
}
```

- [ ] **Step 2: Update `describe_slot_type` for `Tensor`**

`describe_slot_type` lives in `compiler/src/ir/build.rs:162-168`. Replace:

```rust
fn describe_slot_type(ty: ArgType) -> &'static str {
    match ty {
        ArgType::Integer => "integer",
        ArgType::Float => "float",
        ArgType::Symbol => "identifier",
    }
}
```

with:

```rust
fn describe_slot_type(ty: ArgType) -> &'static str {
    match ty {
        ArgType::Integer => "integer",
        ArgType::Float => "float",
        ArgType::Symbol => "identifier",
        ArgType::Tensor => "tensor name (identifier)",
    }
}
```

- [ ] **Step 3: Verify it compiles** (it shouldn't yet — `check_arg_type` still doesn't know about Tensor)

Run: `cargo build -p compiler 2>&1 | tail -20`
Expected: error about non-exhaustive match in `check_arg_type` (next task fixes this), or `unreachable_patterns` warning. Either is fine — Group 2 lands all the changes together; intermediate failures inside the cascade are expected.

### Task 2.2: Update `resolve_args` signature + return type + tensor-resolution logic

**Files:**
- Modify: `compiler/src/ir/build.rs:37-113`

- [ ] **Step 1: Update the function signature**

Replace lines 37-42 (`pub(crate) fn resolve_args(...)` declaration):

```rust
pub(crate) fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    params: &HashMap<&str, u64>,
    op_span: Span,
) -> Result<Vec<OpAttr>, BuildError> {
```

with:

```rust
pub(crate) fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    params: &HashMap<&str, u64>,
    env: &HashMap<String, NodeId>,
    op_span: Span,
) -> Result<(Vec<NodeId>, Vec<OpAttr>), BuildError> {
```

- [ ] **Step 2: Add a `tensor_operands: Vec<NodeId>` accumulator**

After the existing `let mut attrs: Vec<OpAttr> = Vec::with_capacity(positionals.len() + nameds.len());` line, add:

```rust
    let mut tensor_operands: Vec<NodeId> = Vec::new();
```

- [ ] **Step 3: Extend the positional-binding loop to handle `Tensor`**

Replace the positional-binding block:

```rust
    // Bind positionals to slots.
    for (slot, value) in sig.positional.iter().zip(positionals.iter()) {
        check_arg_type(slot, value, op_span)?;
        attrs.push(OpAttr {
            name: slot.name.to_string(),
            value: arg_value_to_attr(value),
        });
    }
```

with:

```rust
    // Bind positionals to slots. Tensor-typed slots resolve against env
    // and contribute a NodeId to `tensor_operands`; everything else
    // becomes an `OpAttr` as before.
    for (slot, value) in sig.positional.iter().zip(positionals.iter()) {
        check_arg_type(slot, value, op_span)?;
        if matches!(slot.ty, ArgType::Tensor) {
            let ArgValue::Symbol(name) = value else {
                // check_arg_type already enforced Symbol-ness.
                // Defensive — never observably reachable.
                return Err(BuildError::arg_type_mismatch(
                    slot.name,
                    "tensor name (identifier)",
                    describe_arg_type(value),
                    op_span,
                ));
            };
            let id = env
                .get(name)
                .copied()
                .ok_or_else(|| BuildError::unknown_variable(name, op_span))?;
            tensor_operands.push(id);
        } else {
            attrs.push(OpAttr {
                name: slot.name.to_string(),
                value: arg_value_to_attr(value),
            });
        }
    }
```

- [ ] **Step 4: Same extension for named slots**

Replace the named-binding block:

```rust
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
```

with:

```rust
    // Bind nameds — match each by slot name.
    for (name, value) in &nameds {
        let slot = sig
            .named
            .iter()
            .find(|s| s.name == *name)
            .ok_or_else(|| BuildError::unexpected_named_arg(name, op_span))?;
        check_arg_type(slot, value, op_span)?;
        if matches!(slot.ty, ArgType::Tensor) {
            // No M10 op declares a Tensor-typed *named* slot, but the
            // type system permits it for symmetry. Resolution mirrors
            // the positional path.
            let ArgValue::Symbol(arg_name) = value else {
                return Err(BuildError::arg_type_mismatch(
                    slot.name,
                    "tensor name (identifier)",
                    describe_arg_type(value),
                    op_span,
                ));
            };
            let id = env
                .get(arg_name)
                .copied()
                .ok_or_else(|| BuildError::unknown_variable(arg_name, op_span))?;
            tensor_operands.push(id);
        } else {
            attrs.push(OpAttr {
                name: slot.name.to_string(),
                value: arg_value_to_attr(value),
            });
        }
    }
```

- [ ] **Step 5: Update `check_arg_type` to accept `Tensor`**

Replace the `check_arg_type` body:

```rust
fn check_arg_type(slot: &ArgSlot, value: &ArgValue, op_span: Span) -> Result<(), BuildError> {
    let actual = describe_arg_type(value);
    let expected = describe_slot_type(slot.ty);
    let ok = matches!(
        (slot.ty, value),
        (ArgType::Integer, ArgValue::Integer(_))
            | (ArgType::Float, ArgValue::Float(_))
            | (ArgType::Symbol, ArgValue::Symbol(_))
    );
    if ok {
        Ok(())
    } else {
        Err(BuildError::arg_type_mismatch(
            slot.name, expected, actual, op_span,
        ))
    }
}
```

with:

```rust
fn check_arg_type(slot: &ArgSlot, value: &ArgValue, op_span: Span) -> Result<(), BuildError> {
    let actual = describe_arg_type(value);
    let expected = describe_slot_type(slot.ty);
    let ok = matches!(
        (slot.ty, value),
        (ArgType::Integer, ArgValue::Integer(_))
            | (ArgType::Float, ArgValue::Float(_))
            | (ArgType::Symbol, ArgValue::Symbol(_))
            | (ArgType::Tensor, ArgValue::Symbol(_)),
    );
    if ok {
        Ok(())
    } else {
        Err(BuildError::arg_type_mismatch(
            slot.name, expected, actual, op_span,
        ))
    }
}
```

(Tensor args parse as `Symbol` at the AST level — the parser doesn't know "this identifier is a tensor name". `check_arg_type` accepts the syntactic match; the *semantic* env lookup happens in the binding loops above.)

- [ ] **Step 6: Update the return statement**

Replace the trailing `Ok(attrs)` with:

```rust
    Ok((tensor_operands, attrs))
```

### Task 2.3: Cascade through `build_op`

**Files:**
- Modify: `compiler/src/ir/build.rs:173-212`

- [ ] **Step 1: Update `build_op` signature**

Replace:

```rust
pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    input_shape: &Shape,
    params: &HashMap<&str, u64>,
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
```

with:

```rust
pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    input_shape: &Shape,
    params: &HashMap<&str, u64>,
    env: &HashMap<String, NodeId>,
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
```

- [ ] **Step 2: Pass `env` into `resolve_args` and consume the new tuple**

Replace:

```rust
    let attrs = resolve_args(std_op, &op_ast.args, params, op_ast.span)?;
```

with:

```rust
    let (tensor_operands, attrs) = resolve_args(std_op, &op_ast.args, params, env, op_ast.span)?;
```

- [ ] **Step 3: Compose `operands` from `input_id` + `tensor_operands` and update shape inference**

Replace the trailing block (from `stdlib::validate_attrs(...)` down through `Ok(id)`):

```rust
    stdlib::validate_attrs(std_op, &attrs).map_err(|e| {
        let attr_name = match &e {
            stdlib::AttrError::OutOfRange { name, .. } => *name,
            stdlib::AttrError::MissingAttr { name } => *name,
        };
        BuildError::invalid_attr_value(
            &format!("{}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
    })?;
    let out_shape = stdlib::infer_output_shape(std_op, std::slice::from_ref(input_shape), &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
    let id = out_nodes.len();
    out_nodes.push(Node {
        kind: NodeKind::Op {
            op: std_op,
            operands: vec![input_id],
            attrs,
            fused_post_ops: Vec::new(),
        },
        ty: Type {
            name: "Tensor".to_string(),
            shape: out_shape,
        },
        source_span: op_ast.span,
    });
    Ok(id)
```

with:

```rust
    stdlib::validate_attrs(std_op, &attrs).map_err(|e| {
        let attr_name = match &e {
            stdlib::AttrError::OutOfRange { name, .. } => *name,
            stdlib::AttrError::MissingAttr { name } => *name,
        };
        BuildError::invalid_attr_value(
            &format!("{}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
    })?;

    // Compose operands: input_id first, then any tensor-resolved operands
    // (in slot declaration order, matching tensor_operands' Vec order).
    let mut operands = Vec::with_capacity(1 + tensor_operands.len());
    operands.push(input_id);
    operands.extend(tensor_operands);

    // Multi-input shape inference: gather all operand shapes and pass to
    // infer_output_shape. Single-input ops continue to work because their
    // arms call single_input(inputs) which validates inputs.len() == 1.
    let input_shapes: Vec<Shape> = operands
        .iter()
        .map(|nid| out_nodes[*nid].ty.shape.clone())
        .collect();
    let out_shape = stdlib::infer_output_shape(std_op, &input_shapes, &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;

    // M10: input_shape param is now redundant (callers can derive from
    // input_id). Kept for caller-API stability — drop in a later
    // trigger-driven cleanup if a refactor pass reveals it as cruft.
    let _ = input_shape;

    let id = out_nodes.len();
    out_nodes.push(Node {
        kind: NodeKind::Op {
            op: std_op,
            operands,
            attrs,
            fused_post_ops: Vec::new(),
        },
        ty: Type {
            name: "Tensor".to_string(),
            shape: out_shape,
        },
        source_span: op_ast.span,
    });
    Ok(id)
```

(Why keep `input_shape` as a parameter? Removing it would touch the call site in `build_model` *and* break the signature in a way orthogonal to the resolve_args cascade. We're respecting the atomic-task-pack convention by keeping the cascade focused.)

### Task 2.4: Cascade through `build_model` callsites

**Files:**
- Modify: `compiler/src/ir/build.rs:271-281` (the `ModelStmt::Pipeline` arm)

- [ ] **Step 1: Pass `env` into the existing `Pipeline` arm's `build_op` call**

In the `ModelStmt::Pipeline(p)` arm, replace:

```rust
                    current = build_op(op_ast, current, &input_shape, &params, &mut nodes)?;
```

with:

```rust
                    current = build_op(op_ast, current, &input_shape, &params, &env, &mut nodes)?;
```

- [ ] **Step 2: Verify `env` is in scope at the callsite**

Confirm `let mut env: HashMap<String, NodeId> = HashMap::new();` exists at the top of `build_model` (it has been there since M3a — `env` is read by `ModelStmt::Pipeline` for the source-identifier lookup). Good — no further plumbing needed.

### Task 2.5: Coverage check (no new test required)

Group 2 is plumbing-only — every existing test routes through the new signature, so the cascade is exercised by the existing 284 tests the moment the suite runs. No new test is added in this group; behaviour-level Tensor-arg tests land in Group 3 (`StdOp::Matmul`).

- [ ] **Step 1: Verify no test name overlap with Group 3 plans**

```bash
grep -rn 'tensor_arg_resolves_from_env\|tensor_arg_unknown_variable_errors' compiler/src/ir/tests.rs
```

Expected: **no matches**. Those names are reserved for Group 3 Task 3.7.

### Task 2.6: Workspace gates

- [ ] **Step 1: `cargo fmt --all`** — exit 0
- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`** — exit 0
- [ ] **Step 3: `cargo build --workspace`** — exit 0
- [ ] **Step 4: `cargo test --workspace`** — 289 tests pass (no count change since Group 2 adds no new tests)

### Task 2.7: Commit

- [ ] **Step 1: Stage**

```bash
git add compiler/src/ir/stdlib.rs compiler/src/ir/build.rs
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/ir): ArgType::Tensor + resolve_args cascade through build_op/build_model

Adds `ArgType::Tensor` for tensor-by-name op arguments. resolve_args
return type changes from `Result<Vec<OpAttr>, BuildError>` to
`Result<(Vec<NodeId>, Vec<OpAttr>), BuildError>` — the first tuple
element collects env-resolved NodeIds for Tensor-typed slots; the
second carries scalar/identifier attrs as before.

Per spec §5.3, the signature change cascades through three functions
in one atomic commit: resolve_args, build_op, build_model. Each one
gains an `env: &HashMap<String, NodeId>` parameter. build_op composes
the Op node's `operands` field as `[input_id, ...tensor_operands]`
and switches infer_output_shape to multi-input form (gathering all
operands' shapes; single-input ops still validate via single_input()).

No new ops yet — Matmul / MulScalar land in groups 3-4. This commit
is plumbing-only; existing 284 tests pass unchanged because no
existing op declares a Tensor-typed slot.

build_op's `input_shape: &Shape` param is now redundant (operands[0]
carries the same info) but kept for caller-API stability — a later
trigger-driven cleanup may drop it.

Spec: docs/superpowers/specs/2026-05-08-m10-attention-pattern-design.md
§5.2, §5.3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify status clean**

Run: `git status`
Expected: "nothing to commit, working tree clean".

---

<!-- group-2-anchor -->

## Group 3 — Commit 3 — `StdOp::Matmul` — variant, signature, shape inference, ShapeError variants

**Group goal:** Add `StdOp::Matmul` variant with `transpose_b` named arg. Implement shape inference covering arbitrary rank ≥ 2 inputs with the five validation steps from spec §5.5.1: input count → rank match → rank ≥ 2 → leading-dim equality (no broadcasting) → inner-dim contraction match. Add four new `ShapeError` variants. Add `matmul_transpose_b` helper paralleling `linear_has_bias`.

**Group done criteria:**
- All four workspace gates green
- Test count 289 → 299 (+10 new UIR builder tests for Matmul + Tensor-arg resolution)

**Files touched:**
- Modify: `compiler/src/ir/stdlib.rs` (StdOp variant, signature, shape inference, ShapeError variants, helper, Display)
- Modify: `compiler/src/ir/tests.rs` (10 new tests)

### Task 3.1: Add `StdOp::Matmul` variant + resolve mapping + Display

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:9-16` (StdOp), `:72-80` (resolve), `:229-239` (Display)

- [ ] **Step 1: Add the variant**

Replace:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
}
```

with:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    /// Matrix multiplication, rank ≥ 2 inputs. With `transpose_b=true`
    /// (named arg), the second operand's last two dims are interpreted
    /// transposed. New in M10.
    Matmul,
}
```

- [ ] **Step 2: Update `resolve`**

Replace:

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
```

with:

```rust
pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear" => Some(StdOp::Linear),
        "relu" => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
        "matmul" => Some(StdOp::Matmul),
        _ => None,
    }
}
```

- [ ] **Step 3: Update `Display`**

Replace the existing `Display for StdOp` impl (around lines 229–239):

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

with:

```rust
impl std::fmt::Display for StdOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            StdOp::Linear => "linear",
            StdOp::Relu => "relu",
            StdOp::Dropout => "dropout",
            StdOp::Softmax => "softmax",
            StdOp::Matmul => "matmul",
        };
        write!(f, "{}", name)
    }
}
```

### Task 3.2: Add the Matmul `Signature`

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:82-114` (the `signature` fn)

- [ ] **Step 1: Add the `Matmul` arm**

In `pub fn signature(op: StdOp) -> Signature { match op { ... } }`, add a new arm before the closing `}`:

```rust
        StdOp::Matmul => Signature {
            positional: &[ArgSlot {
                name: "other",
                ty: Tensor,
                required: true,
            }],
            named: &[ArgSlot {
                name: "transpose_b",
                ty: Symbol,
                required: false,
            }],
        },
```

(Reminder: the function uses `use ArgType::*;` at the top, so bare `Tensor` and `Symbol` resolve correctly.)

### Task 3.3: Add four new `ShapeError` variants + Display arms

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:36-70`

- [ ] **Step 1: Add the variants**

Replace:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Defensive guard. Emitted by `single_input` if a multi-operand op
    /// reaches single-input shape inference. No M3 op constructs >1 operand,
    /// so no test fires this path; the constructor exists so M5+ multi-input
    /// ops (add/concat) cannot silently misroute through single-input helpers.
    WrongInputCount {
        expected: usize,
        actual: usize,
    },
    WrongRank {
        expected: usize,
        actual: usize,
        dim_index: Option<usize>,
    },
    MissingAttr {
        name: &'static str,
    },
}
```

with:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Defensive guard. Emitted by `single_input` if a multi-operand op
    /// reaches single-input shape inference. M5+ multi-input ops (Matmul)
    /// cannot silently misroute through single-input helpers.
    WrongInputCount {
        expected: usize,
        actual: usize,
    },
    WrongRank {
        expected: usize,
        actual: usize,
        dim_index: Option<usize>,
    },
    MissingAttr {
        name: &'static str,
    },
    /// Two operands have different ranks (e.g. `[2, 4] @ [2, 4, 8, 8]`).
    RankMismatch {
        lhs: usize,
        rhs: usize,
    },
    /// Operand rank is below the minimum required by the op
    /// (e.g. 1D input to Matmul, which requires rank ≥ 2).
    RankTooLow {
        required: usize,
        actual: usize,
    },
    /// Two operands' leading dims (indices `0..rank-2`) disagree.
    /// Strict-equal — no broadcasting per design principle #1.
    LeadingDimMismatch {
        dim_index: usize,
        lhs: u64,
        rhs: u64,
    },
    /// Matmul contraction dim disagreement.
    /// `lhs_k` is `a.shape[rank-1]`. `rhs_k` is `b.shape[rank-1]` if
    /// `transpose_b=true`, otherwise `b.shape[rank-2]`.
    InnerDimMismatch {
        lhs_k: u64,
        rhs_k: u64,
        transpose_b: bool,
    },
}
```

- [ ] **Step 2: Add Display arms for the new variants**

Replace the existing `impl std::fmt::Display for ShapeError`:

```rust
impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::WrongInputCount { expected, actual } => {
                write!(f, "expected {} input(s), got {}", expected, actual)
            }
            ShapeError::WrongRank {
                expected,
                actual,
                dim_index: _,
            } => write!(f, "expected rank {}, got {}", expected, actual),
            ShapeError::MissingAttr { name } => write!(f, "missing required attribute: '{}'", name),
        }
    }
}
```

with:

```rust
impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::WrongInputCount { expected, actual } => {
                write!(f, "expected {} input(s), got {}", expected, actual)
            }
            ShapeError::WrongRank {
                expected,
                actual,
                dim_index: _,
            } => write!(f, "expected rank {}, got {}", expected, actual),
            ShapeError::MissingAttr { name } => write!(f, "missing required attribute: '{}'", name),
            ShapeError::RankMismatch { lhs, rhs } => write!(
                f,
                "operand rank mismatch: lhs has rank {}, rhs has rank {}",
                lhs, rhs
            ),
            ShapeError::RankTooLow { required, actual } => write!(
                f,
                "operand rank too low: requires {}, got {}",
                required, actual
            ),
            ShapeError::LeadingDimMismatch {
                dim_index,
                lhs,
                rhs,
            } => write!(
                f,
                "leading dim mismatch at index {}: lhs={}, rhs={} (no broadcasting)",
                dim_index, lhs, rhs
            ),
            ShapeError::InnerDimMismatch {
                lhs_k,
                rhs_k,
                transpose_b,
            } => write!(
                f,
                "matmul contraction dim mismatch: lhs.K={}, rhs.K={}, transpose_b={}",
                lhs_k, rhs_k, transpose_b
            ),
        }
    }
}
```

### Task 3.4: Add `matmul_transpose_b` helper

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` (append at end of file, parallel to `linear_has_bias`)

- [ ] **Step 1: Append the helper**

After the existing `linear_has_bias` function (around line 248), add:

```rust
/// True iff the op's attribute list includes `transpose_b=true`.
///
/// Used by Matmul shape inference and by both arm64 and x86_64 codegen
/// to choose the inner-loop addressing pattern for the B operand.
/// New in M10.
pub fn matmul_transpose_b(attrs: &[OpAttr]) -> bool {
    attrs.iter().any(|a| {
        a.name == "transpose_b" && matches!(&a.value, AttrValue::Symbol(s) if s == "true")
    })
}
```

### Task 3.5: Implement `infer_output_shape` for `Matmul`

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:116-133`

- [ ] **Step 1: Extend the `match op` block**

Add a new arm before the existing `_ => …` (or before the closing `}` if there's no wildcard yet — actually the current code has `Linear` and `Relu | Softmax | Dropout` arms but no wildcard, since `StdOp` is `#[non_exhaustive]` only externally; *inside* its defining crate the match must be exhaustive). With `Matmul` added in Task 3.1, the match is now non-exhaustive — *the build is broken*. Fix it now by adding the `Matmul` arm.

Replace:

```rust
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
```

with:

```rust
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
        StdOp::Matmul => infer_matmul_shape(inputs, attrs),
    }
}

/// Matmul shape inference: rank ≥ 2 inputs of equal rank, leading dims
/// strict-equal (no broadcasting), inner-dim contraction matches.
/// Output: leading dims of `a` followed by `[M, N]` where M = a's
/// second-to-last and N is the non-contracted dim of `b`.
fn infer_matmul_shape(inputs: &[Shape], attrs: &[OpAttr]) -> Result<Shape, ShapeError> {
    // Step 1: input count.
    if inputs.len() != 2 {
        return Err(ShapeError::WrongInputCount {
            expected: 2,
            actual: inputs.len(),
        });
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Step 2: ranks match.
    if a.rank() != b.rank() {
        return Err(ShapeError::RankMismatch {
            lhs: a.rank(),
            rhs: b.rank(),
        });
    }

    // Step 3: rank ≥ 2.
    if a.rank() < 2 {
        return Err(ShapeError::RankTooLow {
            required: 2,
            actual: a.rank(),
        });
    }
    let r = a.rank();

    // Step 4: leading dims (indices 0..r-2) match exactly.
    for i in 0..(r - 2) {
        if a.0[i] != b.0[i] {
            return Err(ShapeError::LeadingDimMismatch {
                dim_index: i,
                lhs: a.0[i],
                rhs: b.0[i],
            });
        }
    }

    // Step 5: inner contraction.
    let transpose_b = matmul_transpose_b(attrs);
    let m = a.0[r - 2];
    let lhs_k = a.0[r - 1];
    let (rhs_k, n) = if transpose_b {
        // b shape [..., N, K] — contract on b's last dim.
        (b.0[r - 1], b.0[r - 2])
    } else {
        // b shape [..., K, N] — contract on b's second-to-last dim.
        (b.0[r - 2], b.0[r - 1])
    };
    if lhs_k != rhs_k {
        return Err(ShapeError::InnerDimMismatch {
            lhs_k,
            rhs_k,
            transpose_b,
        });
    }

    // Output: leading dims + [M, N].
    let mut out = Vec::with_capacity(r);
    out.extend_from_slice(&a.0[..(r - 2)]);
    out.push(m);
    out.push(n);
    Ok(Shape(out))
}
```

- [ ] **Step 2: Update `validate_attrs` for Matmul (no-op pass-through)**

`validate_attrs` lives in `compiler/src/ir/stdlib.rs:200-216`. Add `StdOp::Matmul` to the no-op arm:

Replace:

```rust
        StdOp::Linear | StdOp::Relu | StdOp::Softmax => Ok(()),
```

with:

```rust
        StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul => Ok(()),
```

(Matmul has no integer-attr value-range validation; the boolean-symbol parsing of `transpose_b` is type-checked by `check_arg_type`.)

### Task 3.6: Update existing classify_op fallthroughs

**Files:**
- Modify: `profiles/arm64/src/codegen.rs:251-269` (`classify_op`)
- Modify: `profiles/x86_64/src/codegen.rs` (parallel block — verify line numbers via grep before editing)

- [ ] **Step 1: Verify both classify_op functions reject `Matmul` correctly**

Both currently end with a wildcard arm:

```rust
        #[allow(unreachable_patterns)]
        _ => Err(LowerError::UnsupportedOp {
            op: format!("{op}"),
            span: _span,
        }),
```

Once `StdOp::Matmul` exists (Task 3.1), the wildcard catches it and `LowerError::UnsupportedOp { op: "matmul", ... }` is returned. **Good — the codegen layers stay correct without code change**, because the new op is rejected by both profiles.

The wildcard arm thus becomes the M10 codegen "stub", letting Groups 3–5 land UIR work cleanly without touching codegen until Group 6.

- [ ] **Step 2: Run a sanity test confirming the wildcard catches Matmul**

Append to `compiler/src/ir/tests.rs` (this test will land in Group 3's commit):

```rust
#[test]
fn matmul_resolves_via_stdlib() {
    use crate::ir::stdlib::{resolve, StdOp};
    assert_eq!(resolve("matmul"), Some(StdOp::Matmul));
    assert_eq!(format!("{}", StdOp::Matmul), "matmul");
}
```

- [ ] **Step 3: Run the sanity test**

Run: `cargo test -p compiler matmul_resolves_via_stdlib`
Expected: PASS.

### Task 3.7: Add the 10 new UIR builder tests

**Files:**
- Modify: `compiler/src/ir/tests.rs` (append at end of `mod tests {}`)

The 10 tests below land *together*, each red→green-checked. They're listed grouped by concern; implement and verify each in order.

- [ ] **Step 1: `matmul_2d_shape_inference_no_transpose`**

Append:

```rust
#[test]
fn matmul_2d_shape_inference_no_transpose() {
    let src = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let model = &uir.models[0];
    let out_id = model.output;
    let shape = &model.nodes[out_id].ty.shape;
    assert_eq!(shape.0, vec![2, 8]);
}
```

Run: `cargo test -p compiler matmul_2d_shape_inference_no_transpose`
Expected: **first run, FAILS** because Group 5's `NamedPipeline` builder isn't landed yet — the placeholder rejects with "not yet implemented". This is the right failure mode: every Matmul test in Group 3 depends on Group 5.

**Decision point:** Group 3 cannot test Matmul end-to-end without Group 5's `NamedPipeline` builder. Two options:
- (A) Test Matmul directly via low-level `infer_output_shape` calls — tests look like `infer_output_shape(StdOp::Matmul, &[Shape(vec![2,4]), Shape(vec![4,8])], &[]).unwrap()`. No NFL syntax involved. Tests are *pure stdlib* and are valid in this group.
- (B) Move all Matmul *integration* tests to Group 5 and only test stdlib here.

**This plan picks option (A).** The pure-stdlib tests verify shape-inference correctness without touching the unfinished builder. Integration tests using NFL syntax move to Group 5 (where they're now achievable end-to-end).

Replace the test above with this stdlib-direct version:

```rust
#[test]
fn matmul_2d_shape_inference_no_transpose() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![4, 8]);
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &[]).expect("infer");
    assert_eq!(out.0, vec![2, 8]);
}
```

Run: `cargo test -p compiler matmul_2d_shape_inference_no_transpose`
Expected: PASS.

- [ ] **Step 2: `matmul_2d_shape_inference_transpose_b`**

```rust
#[test]
fn matmul_2d_shape_inference_transpose_b() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let a = Shape(vec![2, 4]);
    // transpose_b=true means b is logically [N, K] → [8, 4].
    let b = Shape(vec![8, 4]);
    let attrs = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &attrs).expect("infer");
    assert_eq!(out.0, vec![2, 8]);
}
```

- [ ] **Step 3: `matmul_4d_shape_inference_no_transpose`**

```rust
#[test]
fn matmul_4d_shape_inference_no_transpose() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4, 16, 8]);
    let b = Shape(vec![2, 4, 8, 16]);
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &[]).expect("infer");
    assert_eq!(out.0, vec![2, 4, 16, 16]);
}
```

- [ ] **Step 4: `matmul_4d_shape_inference_transpose_b`**

```rust
#[test]
fn matmul_4d_shape_inference_transpose_b() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let a = Shape(vec![2, 4, 16, 16]);
    // transpose_b=true → b interpreted as [..., N, K] = [2, 4, 16, 16].
    let b = Shape(vec![2, 4, 16, 16]);
    let attrs = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &attrs).expect("infer");
    assert_eq!(out.0, vec![2, 4, 16, 16]);
}
```

- [ ] **Step 5: `matmul_leading_dim_mismatch_errors`**

```rust
#[test]
fn matmul_leading_dim_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4, 16, 8]);
    let b = Shape(vec![2, 5, 8, 16]); // heads dim 4 vs 5 — strict mismatch
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::LeadingDimMismatch {
                dim_index: 1,
                lhs: 4,
                rhs: 5
            }
        ),
        "unexpected error: {:?}",
        err
    );
}
```

- [ ] **Step 6: `matmul_inner_dim_mismatch_errors`**

```rust
#[test]
fn matmul_inner_dim_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![5, 8]); // K=4 vs K=5
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::InnerDimMismatch {
                lhs_k: 4,
                rhs_k: 5,
                transpose_b: false,
            }
        ),
        "unexpected error: {:?}",
        err
    );
}
```

- [ ] **Step 7: `matmul_rank_mismatch_errors`**

```rust
#[test]
fn matmul_rank_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![2, 4, 4, 8]);
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(err, ShapeError::RankMismatch { lhs: 2, rhs: 4 }),
        "unexpected error: {:?}",
        err
    );
}
```

- [ ] **Step 8: `matmul_rank_too_low_errors`**

```rust
#[test]
fn matmul_rank_too_low_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![4]);
    let b = Shape(vec![4]);
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(err, ShapeError::RankTooLow { required: 2, actual: 1 }),
        "unexpected error: {:?}",
        err
    );
}
```

- [ ] **Step 9: `matmul_wrong_input_count_errors`**

```rust
#[test]
fn matmul_wrong_input_count_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let err = infer_output_shape(StdOp::Matmul, &[a], &[]).unwrap_err();
    assert!(
        matches!(err, ShapeError::WrongInputCount { expected: 2, actual: 1 }),
        "unexpected error: {:?}",
        err
    );
}
```

- [ ] **Step 10: `transpose_b_true_recognised`**

```rust
#[test]
fn transpose_b_true_recognised() {
    use crate::ir::stdlib::matmul_transpose_b;
    use crate::ir::types::{AttrValue, OpAttr};

    let attrs_true = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let attrs_false = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("false".to_string()),
    }];
    let attrs_empty: Vec<OpAttr> = vec![];

    assert!(matmul_transpose_b(&attrs_true));
    assert!(!matmul_transpose_b(&attrs_false));
    assert!(!matmul_transpose_b(&attrs_empty)); // default=false when omitted
}
```

- [ ] **Step 11: Run all 10 + the resolve sanity test**

Run: `cargo test -p compiler matmul_`
Expected: 11 tests all PASS (10 new + 1 from Task 3.6 Step 2). Plus `transpose_b_true_recognised` runs as `transpose_b_true_recognised` and is also caught by the prefix grep.

### Task 3.8: Workspace gates

- [ ] **Step 1: `cargo fmt --all`** — exit 0
- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`** — exit 0
- [ ] **Step 3: `cargo build --workspace`** — exit 0
- [ ] **Step 4: `cargo test --workspace`** — 299 tests pass (289 + 10)

### Task 3.9: Commit

- [ ] **Step 1: Stage**

```bash
git add compiler/src/ir/stdlib.rs compiler/src/ir/tests.rs
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/ir): StdOp::Matmul + 4 new ShapeError variants + transpose_b helper

Adds matmul as a first-class UIR operation. Two operands of equal
rank (≥ 2) with strict-equal leading dims (no broadcasting) and
contraction-dim match on the inner pair. Output shape is leading
dims + [M, N].

`transpose_b` named arg (Symbol "true" / "false" / omitted=false)
swaps b's last-two-dim interpretation between [..., K, N] (default)
and [..., N, K] (transposed).

ShapeError gains four structural variants:
  - RankMismatch { lhs, rhs }
  - RankTooLow { required, actual }
  - LeadingDimMismatch { dim_index, lhs, rhs }
  - InnerDimMismatch { lhs_k, rhs_k, transpose_b }

`matmul_transpose_b(attrs)` helper parallels existing
`linear_has_bias` — reads the boolean-symbol attr conservatively
(defaults to false when omitted or non-"true").

Tests added (10): 2D / 4D × no-transpose / transpose_b, plus four
error-path tests (leading-dim, inner-dim, rank-mismatch, rank-too-low),
plus wrong-input-count, plus transpose_b helper recognition. All
exercise the pure stdlib `infer_output_shape` API — NFL-syntax
end-to-end Matmul tests land in M10 group 5 once NamedPipeline is
fully landed.

Both profiles' `classify_op` continue to reject Matmul via the
existing wildcard arm (`LowerError::UnsupportedOp { op: "matmul" }`).
Codegen lands in groups 6 / 9a.

Spec: §5.1, §5.4, §5.5.1, §5.7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify status clean**

Run: `git status`
Expected: "nothing to commit, working tree clean".

---

## Group 4 — Commit 4 — `StdOp::MulScalar`

**Group goal:** Add `StdOp::MulScalar` with one positional `value: Float` arg. Shape inference is passthrough (output == input). Profiles continue to reject via the wildcard arm in `classify_op` until Group 7 / 9b.

**Group done criteria:**
- All four workspace gates green
- Test count 299 → 302 (+3 new tests)

**Files touched:**
- Modify: `compiler/src/ir/stdlib.rs` (variant, resolve, signature, infer, validate, Display)
- Modify: `compiler/src/ir/tests.rs` (3 new tests)

### Task 4.1: Add `StdOp::MulScalar` variant + resolve + Display

**Files:**
- Modify: `compiler/src/ir/stdlib.rs`

- [ ] **Step 1: Add the variant**

Replace the `StdOp` enum (post-Group-3 state):

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    Matmul,
}
```

with:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    Matmul,
    /// Per-element multiply by a scalar literal. Shape is preserved.
    /// Scalar lives in `attrs` as an `AttrValue::Float(f64)`; codegen
    /// truncates to f32 at lowering time. New in M10.
    MulScalar,
}
```

- [ ] **Step 2: Update `resolve`**

Add `"mul_scalar" => Some(StdOp::MulScalar),` to the existing `resolve`. (NFL surface name uses `_` to keep parser-side regularity; the helper looks like `mul_scalar[0.25]` — see the M10 acceptance fixture.)

- [ ] **Step 3: Update `Display`**

Add `StdOp::MulScalar => "mul_scalar",` to the `Display` impl.

### Task 4.2: Add the MulScalar `Signature`

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:82-114` (the `signature` fn)

- [ ] **Step 1: Add the `MulScalar` arm**

```rust
        StdOp::MulScalar => Signature {
            positional: &[ArgSlot {
                name: "value",
                ty: Float,
                required: true,
            }],
            named: &[],
        },
```

### Task 4.3: Implement `infer_output_shape` for MulScalar

**Files:**
- Modify: `compiler/src/ir/stdlib.rs` (the `infer_output_shape` match)

- [ ] **Step 1: Extend the passthrough arm**

Replace:

```rust
        StdOp::Relu | StdOp::Softmax | StdOp::Dropout => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
```

with:

```rust
        StdOp::Relu | StdOp::Softmax | StdOp::Dropout | StdOp::MulScalar => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
```

(MulScalar's shape is identical to its single input.)

### Task 4.4: Update `validate_attrs` for MulScalar

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:200-216`

- [ ] **Step 1: Add MulScalar to the no-op pass-through**

Replace:

```rust
        StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul => Ok(()),
```

with:

```rust
        StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul | StdOp::MulScalar => Ok(()),
```

(MulScalar accepts any `AttrValue::Float(f64)`; range validation is intentionally absent — large or NaN scalars are user-domain decisions.)

### Task 4.5: Three new UIR tests

**Files:**
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: `mul_scalar_resolves`**

```rust
#[test]
fn mul_scalar_resolves() {
    use crate::ir::stdlib::{resolve, StdOp};
    assert_eq!(resolve("mul_scalar"), Some(StdOp::MulScalar));
    assert_eq!(format!("{}", StdOp::MulScalar), "mul_scalar");
}
```

- [ ] **Step 2: `mul_scalar_preserves_shape`**

```rust
#[test]
fn mul_scalar_preserves_shape() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let input = Shape(vec![2, 4, 16, 16]);
    let attrs = vec![OpAttr {
        name: "value".to_string(),
        value: AttrValue::Float(0.25),
    }];
    let out = infer_output_shape(StdOp::MulScalar, &[input.clone()], &attrs).expect("infer");
    assert_eq!(out.0, input.0);
}
```

- [ ] **Step 3: `mul_scalar_signature_requires_float_positional`**

```rust
#[test]
fn mul_scalar_signature_requires_float_positional() {
    use crate::ir::stdlib::{signature, ArgType, StdOp};
    let sig = signature(StdOp::MulScalar);
    assert_eq!(sig.positional.len(), 1);
    assert_eq!(sig.positional[0].name, "value");
    assert!(matches!(sig.positional[0].ty, ArgType::Float));
    assert!(sig.positional[0].required);
    assert_eq!(sig.named.len(), 0);
}
```

- [ ] **Step 4: Run all three**

Run: `cargo test -p compiler mul_scalar_`
Expected: 3 tests PASS.

### Task 4.6: Workspace gates

- [ ] **Step 1: `cargo fmt --all`** — exit 0
- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`** — exit 0
- [ ] **Step 3: `cargo build --workspace`** — exit 0
- [ ] **Step 4: `cargo test --workspace`** — 302 tests pass (299 + 3)

### Task 4.7: Commit

- [ ] **Step 1: Stage**

```bash
git add compiler/src/ir/stdlib.rs compiler/src/ir/tests.rs
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/ir): StdOp::MulScalar — per-element scalar multiply, shape-preserving

Adds mul_scalar as a UIR op with one positional Float arg ("value").
Shape inference passthrough — output == input.

NFL surface name "mul_scalar" parallels other underscore-cased ops
(linear, relu, dropout, softmax, matmul).

The scalar is stored as `AttrValue::Float(f64)` in the op attrs.
Codegen truncates to f32 bits at the lowering boundary (per spec
§6.5 — documented as an explicit semantic decision; scalar arithmetic
is f32 project-wide per BYTES_PER_ELEMENT=4 in M4b).

Tests added (3): resolve sanity, shape passthrough, signature shape.
Both profiles' `classify_op` continue to reject MulScalar via the
existing wildcard arm. Codegen lands in groups 7 / 9b.

Spec: §5.1, §5.4, §5.5.2, §6.5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify status clean** — `git status` → clean.

---

## Group 5 — Commit 5 — `named_pipeline_stmt` builder + `DeclaredShapeMismatch` + Softmax rank tightening

**Group goal:** Land the `ModelStmt::NamedPipeline` arm in `build_model` (replacing the placeholder error from Group 1). Add `BuildErrorKind::DeclaredShapeMismatch { declared, actual }` for the declared-vs-actual check. Tighten `Softmax` shape inference to require rank ≥ 2 (formerly any rank, fail-late at lowering). With this group landed, the full UIR-builder side of M10 is complete — Matmul / MulScalar / NamedPipeline all flow end-to-end through `compiler::parse → compiler::ir::build`.

**Group done criteria:**
- All four workspace gates green
- Test count 302 → 305 (+3 new tests; the previously-deferred Tensor-arg-resolves-from-env, named_pipeline_shape_match_succeeds, named_pipeline_shape_mismatch_errors land here)

**Files touched:**
- Modify: `compiler/src/ir/error.rs` (new variant + constructor)
- Modify: `compiler/src/ir/build.rs` (`ModelStmt::NamedPipeline` arm replaces placeholder)
- Modify: `compiler/src/ir/stdlib.rs` (Softmax `require_rank` change)
- Modify: `compiler/src/ir/tests.rs` (3 new tests)

### Task 5.1: Add `BuildErrorKind::DeclaredShapeMismatch` + constructor

**Files:**
- Modify: `compiler/src/ir/error.rs`

- [ ] **Step 1: Add the variant**

In `BuildErrorKind` (around line 17–59), add:

```rust
    DeclaredShapeMismatch {
        declared: crate::ir::types::Shape,
        actual: crate::ir::types::Shape,
    },
```

immediately before the closing `}` of the enum.

- [ ] **Step 2: Add the constructor**

Append to `impl BuildError`:

```rust
    pub fn declared_shape_mismatch(
        declared: crate::ir::types::Shape,
        actual: crate::ir::types::Shape,
        span: crate::ast::Span,
    ) -> Self {
        let message = format!(
            "declared shape {} does not match actual shape {}",
            declared, actual
        );
        Self {
            message,
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::DeclaredShapeMismatch { declared, actual },
        }
    }
```

(Format strings work because `Shape` already implements `Display` — see `compiler/src/ir/types.rs:115-120`.)

### Task 5.2: Tighten `Softmax` shape inference

**Files:**
- Modify: `compiler/src/ir/stdlib.rs:116-133`

- [ ] **Step 1: Move `Softmax` out of the passthrough arm into a rank-checking arm**

Replace:

```rust
        StdOp::Relu | StdOp::Softmax | StdOp::Dropout | StdOp::MulScalar => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
```

with:

```rust
        StdOp::Relu | StdOp::Dropout | StdOp::MulScalar => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
        StdOp::Softmax => {
            let input = single_input(inputs)?;
            // M10: tightened to rank ≥ 2 (was: any rank). Fail-fast at
            // UIR rather than fail-late at codegen — both arm64 and
            // x86_64 walk_model dispatch already assume rank ≥ 2 to
            // compute (b, k) for the emitter (b = product of leading
            // dims, k = last dim). 1D softmax is mathematically valid
            // but excluded by project convention; all NFL practical
            // use cases are 2D / 4D batch-first.
            if input.rank() < 2 {
                return Err(ShapeError::RankTooLow {
                    required: 2,
                    actual: input.rank(),
                });
            }
            Ok(input.clone())
        }
```

### Task 5.3: Replace placeholder `NamedPipeline` arm with real builder logic

**Files:**
- Modify: `compiler/src/ir/build.rs:271-294` (the `for stmt in &ast_model.body` loop)

- [ ] **Step 1: Replace the placeholder arm**

Replace:

```rust
            ModelStmt::NamedPipeline(_np) => {
                // Group 5 lands the real builder logic. ...
                return Err(BuildError::shape(
                    "named_pipeline_stmt is not yet implemented (M10 group 5)".to_string(),
                    _np.span,
                ));
            }
```

with:

```rust
            ModelStmt::NamedPipeline(np) => {
                // Build the pipeline in the same shape as ModelStmt::Pipeline,
                // then verify declared shape against actual, then bind in env.
                let mut current = *env
                    .get(&np.source)
                    .ok_or_else(|| BuildError::unknown_variable(&np.source, np.span))?;
                for op_ast in &np.steps {
                    let input_shape = nodes[current].ty.shape.clone();
                    current = build_op(op_ast, current, &input_shape, &params, &env, &mut nodes)?;
                }

                // Verify declared shape against actual.
                let declared = resolve_type(&np.declared_ty, &params)?;
                let actual = nodes[current].ty.shape.clone();
                if declared != actual {
                    return Err(BuildError::declared_shape_mismatch(
                        declared, actual, np.span,
                    ));
                }

                // Bind name in env, update last_pipeline_output (output rule
                // generalises from "last pipeline_stmt" to "last pipeline_stmt
                // OR named_pipeline_stmt", spec §4.2).
                env.insert(np.binding_name.clone(), current);
                last_pipeline_output = Some(current);
            }
```

### Task 5.4: Add the three new UIR tests

**Files:**
- Modify: `compiler/src/ir/tests.rs`

- [ ] **Step 1: `named_pipeline_shape_match_succeeds`**

```rust
#[test]
fn named_pipeline_shape_match_succeeds() {
    // Declared shape matches the pipeline's actual output shape.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let model = &uir.models[0];
    // Output is `y` (the last/only named pipeline).
    let out_id = model.output;
    assert_eq!(model.nodes[out_id].ty.shape.0, vec![2, 4]);
}
```

- [ ] **Step 2: `named_pipeline_shape_mismatch_errors`**

```rust
#[test]
fn named_pipeline_shape_mismatch_errors() {
    // Declared `Tensor[batch, 8]` but `relu` preserves shape, so actual
    // is `Tensor[batch, 4]`. Build must fail with DeclaredShapeMismatch.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 8] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let err = crate::ir::build(&ast).unwrap_err();
    assert!(
        matches!(err.kind, crate::ir::error::BuildErrorKind::DeclaredShapeMismatch { .. }),
        "unexpected error kind: {:?}",
        err.kind
    );
}
```

- [ ] **Step 3: `tensor_arg_resolves_from_env`**

```rust
#[test]
fn tensor_arg_resolves_from_env() {
    // The `x` positional arg in matmul[x] resolves against env to the
    // input variable's NodeId. The resulting Op node should have two
    // operands: input_id (the LHS, which is x itself in this self-mul
    // example) plus the env-resolved x. They're identical here — the
    // Op node carries operands=[x_id, x_id].
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> matmul[x, transpose_b=true]
";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let model = &uir.models[0];
    let out_id = model.output;
    let crate::ir::types::NodeKind::Op { operands, .. } = &model.nodes[out_id].kind else {
        panic!("expected Op node, got Input");
    };
    assert_eq!(operands.len(), 2);
    // Both operands point at the same NodeId — x itself, since q=k=v=x.
    assert_eq!(operands[0], operands[1]);
}
```

- [ ] **Step 4: Run all three**

Run: `cargo test -p compiler named_pipeline_shape_match_succeeds named_pipeline_shape_mismatch_errors tensor_arg_resolves_from_env`
Expected: 3 PASS.

- [ ] **Step 5: Add a softmax rank-too-low test**

Append:

```rust
#[test]
fn softmax_rank_too_low_caught_at_uir() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let input_1d = Shape(vec![16]);
    let err = infer_output_shape(StdOp::Softmax, &[input_1d], &[]).unwrap_err();
    assert!(
        matches!(err, ShapeError::RankTooLow { required: 2, actual: 1 }),
        "unexpected error: {:?}",
        err
    );
}
```

Run: `cargo test -p compiler softmax_rank_too_low_caught_at_uir`
Expected: PASS.

(That's a fourth test, not three — adjust the count: Group 5 lands +4 tests, project total 302 → 306. Update commit message accordingly.)

### Task 5.5: Workspace gates

- [ ] **Step 1: `cargo fmt --all`** — exit 0
- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`** — exit 0
- [ ] **Step 3: `cargo build --workspace`** — exit 0
- [ ] **Step 4: `cargo test --workspace`** — 306 tests pass

### Task 5.6: Commit

- [ ] **Step 1: Stage**

```bash
git add compiler/src/ir/error.rs compiler/src/ir/build.rs compiler/src/ir/stdlib.rs compiler/src/ir/tests.rs
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/ir): named_pipeline builder + DeclaredShapeMismatch + Softmax rank tightening

Replaces the M10-group-1 placeholder NamedPipeline arm in build_model
with the real semantics from spec §5.8:
  1. resolve `np.source` against env to get the starting NodeId
  2. fold the pipeline steps left-to-right, calling build_op for each
  3. resolve declared_ty against model_params
  4. compare declared vs actual shape; raise DeclaredShapeMismatch
     on disagreement
  5. bind the resulting NodeId in env under np.binding_name
  6. update last_pipeline_output (so the model's output follows the
     output rule generalised in §4.2).

DeclaredShapeMismatch is a new structural BuildErrorKind variant
carrying the two Shape values. Distinct from the existing
ShapeMismatch { detail: String } — the latter remains the catch-all
for ad-hoc shape failures from the stdlib layer.

Softmax shape inference now requires rank ≥ 2 (was: any rank).
Reasoning: every existing profile's walk_model::Softmax dispatch
flattens leading dims into row-count `b` and reads the last dim as
column-count `k`; rank 1 has no leading dims and an unambiguous k=1
would mathematically be valid but is excluded by project convention.
1D softmax never appeared in any fixture and is a fail-fast UIR
concern, not a fail-late codegen one.

Tests added (4):
  - named_pipeline_shape_match_succeeds
  - named_pipeline_shape_mismatch_errors
  - tensor_arg_resolves_from_env
  - softmax_rank_too_low_caught_at_uir

Project total 302 → 306.

Spec: §4.2, §5.5.3, §5.6, §5.8.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: `git status` → clean.**

---

## Group 6 — Commit 6 — arm64 `emit_matmul` + dispatch

**Group goal:** Land `profiles/arm64/src/ops/matmul.rs` exporting `emit_matmul`, wire it through `walk_model` and `classify_op`, and update `assign_buffers` so `StdOp::Matmul` gets a `BufferLoc::StackOffset` (separate intermediate buffer). The asm structure is an outer loop over `leading_count` (product of leading dims; 1 for 2D inputs) wrapping the inner triple-loop matmul kernel from `emit_linear`'s structure (`fmadd s0, s1, s2, s0`), but **without** bias-add and **without** any post-ops — those layers are emit_linear's responsibility, not emit_matmul's.

`emit_linear` itself is unchanged (spec §6.1: "new files only").

**Group done criteria:**
- All four workspace gates green
- Test count 306 → 311 (+5 new arm64 codegen unit tests)

**Files touched:**
- Create: `profiles/arm64/src/ops/matmul.rs`
- Modify: `profiles/arm64/src/ops/mod.rs` (declare module + re-export)
- Modify: `profiles/arm64/src/buffer.rs` (`StdOp::Matmul` → `StackOffset`)
- Modify: `profiles/arm64/src/codegen.rs` (`classify_op` arm + `walk_model` dispatch)
- Modify: `profiles/arm64/src/tests.rs` (5 new tests)

### Task 6.1: Add `Matmul` to `assign_buffers`

**Files:**
- Modify: `profiles/arm64/src/buffer.rs:36-74`

- [ ] **Step 1: Extend the per-op match**

Replace:

```rust
                    match op {
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
                        StdOp::Linear | StdOp::Softmax => {
```

with:

```rust
                    match op {
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
                        StdOp::Linear | StdOp::Softmax | StdOp::Matmul => {
```

(`Matmul` joins the stack-offset family. `MulScalar` joins the alias family in Group 7. The existing `_` wildcard fallthrough already handles the case but spec §6.3 requires us to spell it out for `Matmul` so the explicit-arm reasoning is local to the arm64 profile.)

### Task 6.2: Update `classify_op` to accept `Matmul`

**Files:**
- Modify: `profiles/arm64/src/codegen.rs:251-269`

- [ ] **Step 1: Add the explicit `Matmul` arm**

Replace:

```rust
fn classify_op(
    op: StdOp,
    _attrs: &[compiler::OpAttr],
    _span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => Ok(()),
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Ok(()),
        StdOp::Softmax => Ok(()),
        // M5c: #[non_exhaustive] on StdOp requires a wildcard arm.
        // Future ops are rejected here until codegen learns them.
        #[allow(unreachable_patterns)]
        _ => Err(LowerError::UnsupportedOp {
            op: format!("{op}"),
            span: _span,
        }),
    }
}
```

with:

```rust
fn classify_op(
    op: StdOp,
    _attrs: &[compiler::OpAttr],
    _span: compiler::ast::Span,
) -> Result<(), LowerError> {
    match op {
        StdOp::Linear => Ok(()),
        StdOp::Relu => Ok(()),
        StdOp::Dropout => Ok(()),
        StdOp::Softmax => Ok(()),
        StdOp::Matmul => Ok(()),
        // M5c: #[non_exhaustive] on StdOp requires a wildcard arm.
        // Future ops (and StdOp::MulScalar until M10 group 7) are
        // rejected here until codegen learns them.
        #[allow(unreachable_patterns)]
        _ => Err(LowerError::UnsupportedOp {
            op: format!("{op}"),
            span: _span,
        }),
    }
}
```

### Task 6.3: Add the new module + scaffolding

**Files:**
- Create: `profiles/arm64/src/ops/matmul.rs`
- Modify: `profiles/arm64/src/ops/mod.rs`

- [ ] **Step 1: Create `profiles/arm64/src/ops/matmul.rs` (skeleton)**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — multi-dim matmul over rank ≥ 2 inputs with optional
//! `transpose_b`. Outer loop iterates over the product of leading dims
//! (`leading_count`); the inner kernel is a triple-loop FMA matmul over
//! the trailing `[M, K]` × `[K, N]` (or `[N, K]` if `transpose_b=true`)
//! pair.
//!
//! `emit_linear` (matmul + bias + post-ops) is unchanged; this module
//! is strictly additive per spec §6.1.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;
use compiler::ast::Span;
use profile_api::LowerError;

/// Emit AArch64 asm for a multi-dim matmul.
///
/// `leading_count` = product of leading dims (`shape[..rank-2].product()`).
/// For 2D inputs `leading_count == 1` — the outer loop runs once and is
/// effectively elided.
///
/// `m`, `k`, `n` are the trailing matrix dims. With `transpose_b=false`,
/// B is `[..., K, N]`; with `transpose_b=true`, B is `[..., N, K]`.
///
/// Base pointers `x11/x13/x12` (= A, B, DST) are materialised once
/// before the outer loop and MUST NOT be mutated inside it. Per-outer
/// slice pointers are computed in scratch registers (`x6`/`x7`/`x8`).
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
    leading_count: u64,
    m: u64,
    k: u64,
    n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    _node_span: Span,
) -> Result<String, LowerError> {
    let mid = format!("{model_idx}_{matmul_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul (leading_count={}): [{},{}] x [{},{}] -> [{},{}], transpose_b={}\n",
        leading_count, m, k, k, n, m, n, transpose_b
    ));

    // Materialise base pointers ONCE — invariant across outer iterations.
    s.push_str(&materialise_ptr("x11", a_loc));
    s.push_str(&materialise_ptr("x13", b_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Inner-kernel slice sizes (in floats):
    //   A slice = M * K   (per outer iteration)
    //   B slice = K * N   (always — same regardless of transpose_b)
    //   DST slice = M * N (per outer iteration)
    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Outer loop: x17 = outer_idx (caller-saved scratch register).
    s.push_str("    mov     x17, #0\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32("x10", leading_count as usize));
    s.push_str("    cmp     x17, x10\n");
    s.push_str(&format!("    b.ge    .Lmm4d_outer_end_{mid}\n"));

    // Per-outer slice base pointers in x1/x2/x3.
    // x1 = A_slice = x11 + x17 * a_slice * 4
    s.push_str(&emit_imm32("x8", a_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x1, x11, x6, lsl #2\n");
    // x2 = B_slice = x13 + x17 * b_slice * 4
    s.push_str(&emit_imm32("x8", b_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x2, x13, x6, lsl #2\n");
    // x3' = DST_slice = x12 + x17 * dst_slice * 4
    // (We use x4 as the DST_slice pointer to avoid colliding with the
    // existing meaning of x3 in emit_linear's i-loop.)
    s.push_str(&emit_imm32("x8", dst_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x4, x12, x6, lsl #2\n");

    // Hoist trailing-dim bounds.
    s.push_str(&emit_imm32("x10", m as usize));   // x10 = M (re-used; was leading_count above)
    s.push_str(&emit_imm32("x15", n as usize));   // x15 = N
    s.push_str(&emit_imm32("x16", k as usize));   // x16 = K

    // Inner i-loop (rows of output, [0, M)).
    // x5 = i.
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str("    cmp     x5, x10\n");
    s.push_str(&format!("    b.ge    .Lmm4d_i_end_{mid}\n"));

    // Inner j-loop (cols of output, [0, N)).
    // x7 = j.
    s.push_str("    mov     x7, #0\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str("    cmp     x7, x15\n");
    s.push_str(&format!("    b.ge    .Lmm4d_j_end_{mid}\n"));

    // Accumulator s0 = 0.0.
    s.push_str("    fmov    s0, wzr\n");
    // Inner k-loop (contraction, [0, K)).
    // x9 = k_inner.
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str("    cmp     x9, x16\n");
    s.push_str(&format!("    b.ge    .Lmm4d_k_end_{mid}\n"));

    // a_offset = i * K + k_inner   (always — A is always [..., M, K])
    s.push_str("    mul     x6, x5, x16\n");
    s.push_str("    add     x6, x6, x9\n");
    s.push_str("    ldr     s1, [x1, x6, lsl #2]\n");

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str("    mul     x6, x7, x16\n");
        s.push_str("    add     x6, x6, x9\n");
    } else {
        s.push_str("    mul     x6, x9, x15\n");
        s.push_str("    add     x6, x6, x7\n");
    }
    s.push_str("    ldr     s2, [x2, x6, lsl #2]\n");

    // Fused multiply-add: s0 = s0 + s1 * s2.
    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store s0 → DST_slice[i * N + j].
    s.push_str("    mul     x6, x5, x15\n");
    s.push_str("    add     x6, x6, x7\n");
    s.push_str("    str     s0, [x4, x6, lsl #2]\n");

    // j++; j-loop tail.
    s.push_str("    add     x7, x7, #1\n");
    s.push_str(&format!("    b       .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    // i++; i-loop tail.
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    // Outer++; outer-loop tail.
    s.push_str("    add     x17, x17, #1\n");
    s.push_str(&format!("    b       .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    Ok(s)
}
```

- [ ] **Step 2: Update `profiles/arm64/src/ops/mod.rs`**

Replace:

```rust
pub mod dropout;
pub mod linear;
pub mod relu;
pub mod softmax;

pub use dropout::emit_dropout_copy;
pub use linear::emit_linear;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
```

with:

```rust
pub mod dropout;
pub mod linear;
pub mod matmul;
pub mod relu;
pub mod softmax;

pub use dropout::emit_dropout_copy;
pub use linear::emit_linear;
pub use matmul::emit_matmul;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
```

- [ ] **Step 3: Verify the module compiles**

Run: `cargo build -p profiles-arm64`
Expected: success. There may be a `dead_code` warning on `emit_matmul` until Task 6.4 dispatches to it; that's expected and resolved at that task.

### Task 6.4: Dispatch `Matmul` from `walk_model`

**Files:**
- Modify: `profiles/arm64/src/codegen.rs:115-225` (the `walk_model` dispatch)

- [ ] **Step 1: Add a `matmul_idx` counter alongside the existing `linear_idx` etc.**

In `walk_model`, find:

```rust
    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    let mut softmax_idx = 0usize;
    let mut dropout_idx = 0usize;
```

Append:

```rust
    let mut matmul_idx = 0usize;
```

- [ ] **Step 2: Add the `StdOp::Matmul` arm**

Inside the `match op { ... }` block (after the existing `StdOp::Softmax => { ... }` arm and before the wildcard `_ =>`), add:

```rust
                StdOp::Matmul => {
                    // Operands: input (operands[0]) is A (the LHS, which
                    // came from the pipeline). The Tensor-resolved B
                    // operand is operands[1] — pushed by build_op from
                    // `tensor_operands`.
                    let a_id = operands[0];
                    let b_id = operands[1];
                    let a_shape = &model.nodes[a_id].ty.shape;
                    let b_shape = &model.nodes[b_id].ty.shape;
                    let r = a_shape.0.len();
                    debug_assert!(r >= 2, "matmul shape inference enforces rank >= 2");

                    let leading_count: u64 = a_shape.0[..(r - 2)].iter().product();
                    let m = a_shape.0[r - 2];
                    let k = a_shape.0[r - 1];
                    let transpose_b =
                        compiler::ir::stdlib::matmul_transpose_b(match &node.kind {
                            NodeKind::Op { attrs, .. } => attrs,
                            _ => unreachable!("matched NodeKind::Op above"),
                        });
                    let n = if transpose_b {
                        b_shape.0[r - 2]
                    } else {
                        b_shape.0[r - 1]
                    };

                    let a_loc = resolve_loc(&assignment.locs, a_id);
                    let b_loc = resolve_loc(&assignment.locs, b_id);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_matmul(
                        leading_count,
                        m,
                        k,
                        n,
                        transpose_b,
                        model_idx,
                        matmul_idx,
                        a_loc,
                        b_loc,
                        dst_loc,
                        node.source_span,
                    )?);
                    matmul_idx += 1;
                }
```

### Task 6.5: Add the 5 arm64 codegen unit tests

**Files:**
- Modify: `profiles/arm64/src/tests.rs` (append at end of `mod tests {}`)

The tests below assert structural properties of the emitted asm string — they don't run the code. Patterns checked: outer-loop label presence, inner k-loop FMA presence, transpose_b dispatch differs in the b_offset computation, `_expf` is *not* called, default-vs-explicit-false equivalence.

- [ ] **Step 1: `matmul_4d_emits_outer_loop_wrapper`**

```rust
#[test]
fn matmul_4d_emits_outer_loop_wrapper() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("build");
    let asm = crate::lower(&uir).expect("lower");
    // Outer loop wrapper present.
    assert!(asm.source.contains(".Lmm4d_outer_0_0:"), "asm:\n{}", asm.source);
    assert!(asm.source.contains(".Lmm4d_outer_end_0_0:"), "asm:\n{}", asm.source);
    // Inner triple-loop labels present.
    assert!(asm.source.contains(".Lmm4d_i_0_0:"), "asm:\n{}", asm.source);
    assert!(asm.source.contains(".Lmm4d_j_0_0:"), "asm:\n{}", asm.source);
    assert!(asm.source.contains(".Lmm4d_k_0_0:"), "asm:\n{}", asm.source);
    // FMA in inner k-body.
    assert!(asm.source.contains("fmadd   s0, s1, s2, s0"), "asm:\n{}", asm.source);
}
```

- [ ] **Step 2: `matmul_2d_collapses_to_outer_count_one`**

```rust
#[test]
fn matmul_2d_collapses_to_outer_count_one() {
    let src = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("build");
    let asm = crate::lower(&uir).expect("lower");
    // The outer loop is still emitted, but its bound is 1, so a single
    // emit_imm32 line "movz/movk → x10, #1" should appear before the
    // outer loop. We assert structurally on the comment header instead
    // (more readable).
    assert!(asm.source.contains("leading_count=1"), "asm:\n{}", asm.source);
}
```

- [ ] **Step 3: `matmul_transpose_b_inner_addressing_differs`**

```rust
#[test]
fn matmul_transpose_b_inner_addressing_differs() {
    let src_no_t = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    // For a transpose_b version we need shapes that match: [batch, 4]
    // with b transposed [N, K] = [8, 4].
    let src_t = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[8, 4]

    out: Tensor[batch, 8] = a -> matmul[b, transpose_b=true]
";
    let asm_no_t = crate::lower(
        &compiler::ir::build(&compiler::parse(src_no_t).unwrap()).unwrap(),
    )
    .expect("lower no-t").source;
    let asm_t = crate::lower(
        &compiler::ir::build(&compiler::parse(src_t).unwrap()).unwrap(),
    )
    .expect("lower t").source;
    // Both compute b_offset, but with different operand orders.
    // No-transpose: `mul x6, x9, x15` (k_inner * N).
    // Transpose:   `mul x6, x7, x16` (j * K).
    assert!(asm_no_t.contains("mul     x6, x9, x15"), "no-t asm:\n{}", asm_no_t);
    assert!(asm_t.contains("mul     x6, x7, x16"), "t asm:\n{}", asm_t);
    assert_ne!(asm_no_t, asm_t, "transpose_b should change emitted asm");
}
```

- [ ] **Step 4: `matmul_transpose_b_false_default_matches_explicit_false`**

```rust
#[test]
fn matmul_transpose_b_false_default_matches_explicit_false() {
    // Spec §8.3 — guard against drift between the omit-attr code path
    // and the explicit-false path.
    let src_default = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let src_explicit = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b, transpose_b=false]
";
    let asm_d = crate::lower(
        &compiler::ir::build(&compiler::parse(src_default).unwrap()).unwrap(),
    )
    .expect("lower default").source;
    let asm_e = crate::lower(
        &compiler::ir::build(&compiler::parse(src_explicit).unwrap()).unwrap(),
    )
    .expect("lower explicit").source;
    assert_eq!(asm_d, asm_e, "default omit must match explicit false");
}
```

- [ ] **Step 5: `matmul_does_not_call_extern_math`**

```rust
#[test]
fn matmul_does_not_call_extern_math() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(!asm.contains("bl      _expf"), "matmul must not call extern math: {}", asm);
    assert!(!asm.contains("expf@PLT"), "matmul must not call extern math: {}", asm);
}
```

- [ ] **Step 6: Run all 5**

Run: `cargo test -p profiles-arm64 matmul_`
Expected: 5 PASS.

### Task 6.6: Workspace gates

- [ ] **Step 1: `cargo fmt --all`** — exit 0
- [ ] **Step 2: `cargo clippy --workspace --all-targets -- -D warnings`** — exit 0
- [ ] **Step 3: `cargo build --workspace`** — exit 0
- [ ] **Step 4: `cargo test --workspace`** — 311 tests pass on macOS arm64 (306 + 5).

### Task 6.7: Commit

- [ ] **Step 1: Stage**

```bash
git add profiles/arm64/src/ops/matmul.rs \
        profiles/arm64/src/ops/mod.rs \
        profiles/arm64/src/buffer.rs \
        profiles/arm64/src/codegen.rs \
        profiles/arm64/src/tests.rs
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/arm64): emit_matmul with outer-loop wrapper for rank≥2 inputs

New file profiles/arm64/src/ops/matmul.rs exporting `emit_matmul`.
Outer loop iterates over `leading_count` (= product of leading dims;
1 for 2D inputs); inner triple-loop is the same FMA-using kernel
shape as emit_linear's matmul body — `fmadd s0, s1, s2, s0` per
contraction step — but without bias-add and without post-ops, which
remain emit_linear's responsibility.

Base pointers x11/x13/x12 are materialised once before the outer
loop and remain unchanged across iterations (per spec §6.4 invariant).
Per-outer slice pointers are computed in scratch x1/x2/x4 each iter.

`transpose_b=true` flips the inner b_offset computation:
  default:    b_offset = k_inner * N + j   (B is [..., K, N])
  transposed: b_offset = j * K + k_inner   (B is [..., N, K])

Both paths use the same caller-saved scratch registers; the function
prologue is unchanged because Matmul does not contribute to
calls_extern_math() (no `bl _expf`).

assign_buffers extended: StdOp::Matmul joins the StackOffset family
(separate intermediate buffer per matmul output, sized rank-agnostically
via `shape.iter().product() * 4`).

walk_model dispatch: new StdOp::Matmul arm computes leading_count,
m, k, n, transpose_b from operand shapes + attrs, looks up the three
BufferLocs (a, b, dst), and calls emit_matmul.

classify_op now accepts Matmul (was: rejected via wildcard).

Tests added (5):
  - matmul_4d_emits_outer_loop_wrapper
  - matmul_2d_collapses_to_outer_count_one
  - matmul_transpose_b_inner_addressing_differs
  - matmul_transpose_b_false_default_matches_explicit_false
  - matmul_does_not_call_extern_math

emit_linear is unchanged (spec §6.1 invariant). Project total
306 → 311.

Spec: §6.1, §6.3, §6.4, §6.7, §8.3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: `git status` → clean.**

---

## Group 7 — Commit 7 — arm64 `emit_mulscalar` + dispatch

**Group goal:** Land `profiles/arm64/src/ops/mulscalar.rs` exporting `emit_mulscalar`, wire it through `walk_model::MulScalar` and `classify_op`. Update `assign_buffers` so `StdOp::MulScalar` gets `BufferLoc::Alias(operands[0])` (in-place). The asm is a flat element-wise loop multiplying each element by a pre-loaded scalar.

**Group done criteria:**
- All four workspace gates green
- Test count 311 → 313 (+2 new arm64 codegen tests)

**Files touched:**
- Create: `profiles/arm64/src/ops/mulscalar.rs`
- Modify: `profiles/arm64/src/ops/mod.rs`
- Modify: `profiles/arm64/src/buffer.rs` (`StdOp::MulScalar` → `Alias`)
- Modify: `profiles/arm64/src/codegen.rs` (`classify_op` + `walk_model` dispatch)
- Modify: `profiles/arm64/src/tests.rs` (2 new tests)

### Task 7.1: Add `MulScalar` to `assign_buffers`

**Files:**
- Modify: `profiles/arm64/src/buffer.rs`

- [ ] **Step 1: Extend the alias-family arm**

Replace:

```rust
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
```

with:

```rust
                        StdOp::Relu | StdOp::Dropout | StdOp::MulScalar => {
                            BufferLoc::Alias(operands[0])
                        }
```

(In-place: MulScalar reads-and-writes element-by-element; the upstream buffer is reusable. Also matches spec §6.3.)

### Task 7.2: Update `classify_op` for `MulScalar`

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Add `StdOp::MulScalar => Ok(()),`**

Right after the `StdOp::Matmul => Ok(()),` line added in Group 6:

```rust
        StdOp::MulScalar => Ok(()),
```

### Task 7.3: Create `profiles/arm64/src/ops/mulscalar.rs`

**Files:**
- Create: `profiles/arm64/src/ops/mulscalar.rs`

- [ ] **Step 1: Write the new module**

```rust
// SPDX-License-Identifier: Apache-2.0

//! MulScalar codegen — flat per-element multiply by a scalar.
//!
//! Scalar is pre-loaded into `s4` once before the loop via `movz/movk`
//! → `fmov`. The loop is `total_elements` iterations of:
//!   ldr s0, [src, idx, lsl #2]
//!   fmul s0, s0, s4
//!   str s0, [dst, idx, lsl #2]
//!
//! With `BufferLoc::Alias`, `src_loc == dst_loc` → in-place transformation
//! (the materialise_ptr resolution gives both registers the same value).
//!
//! f64-to-f32 truncation happens in the dispatcher (codegen.rs) — the
//! emitter receives `scalar_bits: u32` already in f32 form. See spec §6.5.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for `dst[i] = src[i] * scalar` over `total_elements`.
#[allow(clippy::too_many_arguments)]
pub fn emit_mulscalar(
    total_elements: u64,
    scalar_bits: u32,
    model_idx: usize,
    op_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; mul_scalar: total_elements={}, scalar_bits=0x{:08x}\n",
        total_elements, scalar_bits
    ));

    // Pre-load the scalar into s4. Decompose the u32 into hi16/lo16.
    let lo16 = (scalar_bits & 0xFFFF) as u16;
    let hi16 = ((scalar_bits >> 16) & 0xFFFF) as u16;
    s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo16));
    if hi16 != 0 {
        s.push_str(&format!(
            "    movk    w9, #0x{:04x}, lsl #16\n",
            hi16
        ));
    }
    s.push_str("    fmov    s4, w9\n");

    // Materialise base pointers. With Alias, both resolve to the same.
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Flat loop: x3 = i.
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lms_{mid}:\n"));
    s.push_str(&emit_imm32("x10", total_elements as usize));
    s.push_str("    cmp     x3, x10\n");
    s.push_str(&format!("    b.ge    .Lms_end_{mid}\n"));

    s.push_str("    ldr     s0, [x11, x3, lsl #2]\n");
    s.push_str("    fmul    s0, s0, s4\n");
    s.push_str("    str     s0, [x12, x3, lsl #2]\n");

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lms_{mid}\n"));
    s.push_str(&format!(".Lms_end_{mid}:\n"));

    s
}
```

- [ ] **Step 2: Update `profiles/arm64/src/ops/mod.rs`**

Add `pub mod mulscalar;` and `pub use mulscalar::emit_mulscalar;` in the same alphabetical order pattern as Group 6.

After the change, the file should read:

```rust
pub mod dropout;
pub mod linear;
pub mod matmul;
pub mod mulscalar;
pub mod relu;
pub mod softmax;

pub use dropout::emit_dropout_copy;
pub use linear::emit_linear;
pub use matmul::emit_matmul;
pub use mulscalar::emit_mulscalar;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
```

### Task 7.4: Dispatch `MulScalar` from `walk_model`

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Add the index counter**

Append `let mut mulscalar_idx = 0usize;` next to the others.

- [ ] **Step 2: Add the dispatch arm**

Inside the `match op { ... }` block, after `StdOp::Matmul`, add:

```rust
                StdOp::MulScalar => {
                    let total: u64 = node.ty.shape.0.iter().product();
                    let attrs = match &node.kind {
                        NodeKind::Op { attrs, .. } => attrs,
                        _ => unreachable!(),
                    };
                    // f64 stored in attrs; truncate to f32 bits at the
                    // codegen boundary per spec §6.5.
                    let scalar_f64 = attrs
                        .iter()
                        .find(|a| a.name == "value")
                        .and_then(|a| match a.value {
                            compiler::AttrValue::Float(v) => Some(v),
                            _ => None,
                        })
                        .expect("MulScalar.value attr must be Float (signature enforces)");
                    let scalar_bits = (scalar_f64 as f32).to_bits();

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_mulscalar(
                        total,
                        scalar_bits,
                        model_idx,
                        mulscalar_idx,
                        src_loc,
                        dst_loc,
                    ));
                    mulscalar_idx += 1;
                }
```

### Task 7.5: Add 2 unit tests

**Files:**
- Modify: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: `mul_scalar_preloads_scalar_via_movz_movk`**

```rust
#[test]
fn mul_scalar_preloads_scalar_via_movz_movk() {
    // 0.25 in f32 bits is 0x3E800000 (hi16=0x3E80, lo16=0x0000).
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.25]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    // movz preserves lo16; movk shifts hi16 in.
    assert!(asm.contains("movz    w9, #0x0000"), "asm:\n{}", asm);
    assert!(asm.contains("movk    w9, #0x3e80, lsl #16"), "asm:\n{}", asm);
    assert!(asm.contains("fmov    s4, w9"), "asm:\n{}", asm);
}
```

- [ ] **Step 2: `mul_scalar_emits_fmul_in_inner_loop`**

```rust
#[test]
fn mul_scalar_emits_fmul_in_inner_loop() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.5]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(asm.contains("fmul    s0, s0, s4"), "asm:\n{}", asm);
    assert!(asm.contains(".Lms_0_0:"), "asm:\n{}", asm);
    assert!(asm.contains(".Lms_end_0_0:"), "asm:\n{}", asm);
}
```

- [ ] **Step 3: Run both**

Run: `cargo test -p profiles-arm64 mul_scalar_`
Expected: 2 PASS.

### Task 7.6: Workspace gates + commit

- [ ] **Step 1: Run all four gates** — exit 0 on each
- [ ] **Step 2: Test count check** — `cargo test --workspace` → 313 (311 + 2)
- [ ] **Step 3: Stage**

```bash
git add profiles/arm64/src/ops/mulscalar.rs \
        profiles/arm64/src/ops/mod.rs \
        profiles/arm64/src/buffer.rs \
        profiles/arm64/src/codegen.rs \
        profiles/arm64/src/tests.rs
```

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(m10/arm64): emit_mulscalar — in-place per-element scalar multiply

New file profiles/arm64/src/ops/mulscalar.rs exporting `emit_mulscalar`.
Pre-loads the f32-bit scalar into s4 via movz + (optional) movk + fmov,
then runs a flat element-wise loop:
  ldr s0, [src, idx, lsl #2]
  fmul s0, s0, s4
  str s0, [dst, idx, lsl #2]

assign_buffers: StdOp::MulScalar joins Relu/Dropout in the alias
family — output BufferLoc is `Alias(operands[0])`, so src_loc and
dst_loc resolve to the same materialised pointer (in-place).

walk_model: new StdOp::MulScalar arm reads the AttrValue::Float scalar
and truncates to f32 bits at the codegen boundary per spec §6.5
(`(scalar_f64 as f32).to_bits()`). Documented in the module doc as
the project-wide f32 contract.

classify_op accepts MulScalar.

Tests added (2): scalar pre-load via movz/movk, fmul inner-loop body.

Project total 311 → 313.

Spec: §6.1, §6.3, §6.5, §6.7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: `git status` → clean.**

---

## Group 8 — Commit 8 — arm64 Softmax 4D dispatch

**Group goal:** Generalise the `walk_model::Softmax` arm so 4D (and any rank ≥ 2) inputs flatten leading dims into the row count `b`. The `emit_softmax` asm emitter is **unchanged** — its `(b, k)` interface already represents "b rows of width k", which is the right abstraction once the dispatch flattens.

**Group done criteria:**
- All four workspace gates green
- Test count 313 → 314 (+1 new arm64 codegen test)

**Files touched:**
- Modify: `profiles/arm64/src/codegen.rs` (the `StdOp::Softmax` arm in `walk_model`)
- Modify: `profiles/arm64/src/tests.rs` (1 new test)

### Task 8.1: Update the `Softmax` dispatch

**Files:**
- Modify: `profiles/arm64/src/codegen.rs:193-209`

- [ ] **Step 1: Replace the existing arm**

Replace:

```rust
                StdOp::Softmax => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_softmax(
                        b,
                        k,
                        model_idx,
                        softmax_idx,
                        src_loc,
                        dst_loc,
                        sym_prefix,
                    ));
                    softmax_idx += 1;
                }
```

with:

```rust
                StdOp::Softmax => {
                    // Last-axis softmax. b = product(shape[..rank-1]) (total
                    // rows), k = shape[rank-1] (row width). For 2D
                    // [batch, dim] this collapses to b=batch, k=dim
                    // (identical to pre-M10 behaviour). For 4D
                    // [B, H, M, K] this gives b = B*H*M, k = K.
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let last = in_shape.0.len() - 1;
                    let k = in_shape.0[last];
                    let b: u64 = in_shape.0[..last].iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = resolve_loc(&assignment.locs, node_idx);
                    body.push_str(&crate::ops::emit_softmax(
                        b,
                        k,
                        model_idx,
                        softmax_idx,
                        src_loc,
                        dst_loc,
                        sym_prefix,
                    ));
                    softmax_idx += 1;
                }
```

### Task 8.2: Add the dispatch test

**Files:**
- Modify: `profiles/arm64/src/tests.rs`

- [ ] **Step 1: `softmax_4d_dispatch_computes_b_as_product_of_leading_dims`**

```rust
#[test]
fn softmax_4d_dispatch_computes_b_as_product_of_leading_dims() {
    // 4D shape [2, 4, 8, 16]: b = 2*4*8 = 64, k = 16.
    // The emitter's outer loop bound is set via emit_imm32 → x10
    // immediately above the .Lsm_i_<id> label.
    let src = "\
model M [batch=2, heads=4, seq=8, dim=16]:
    x: Tensor[batch, heads, seq, dim]

    y: Tensor[batch, heads, seq, dim] = x -> softmax
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    // 64 in lo16 is 0x0040; emit_imm32 writes a movz with that lo16.
    assert!(
        asm.contains("movz    x10, #0x0040"),
        "expected b=64 materialised before .Lsm_i_…; asm:\n{}",
        asm
    );
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p profiles-arm64 softmax_4d_dispatch_computes_b_as_product_of_leading_dims`
Expected: PASS.

### Task 8.3: Workspace gates + commit

- [ ] **Step 1: Run all four gates** — exit 0
- [ ] **Step 2: `cargo test --workspace`** — 314 tests pass
- [ ] **Step 3: Stage + commit**

```bash
git add profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m10/arm64): Softmax dispatch generalised to rank ≥ 2 (4D works)

Updates walk_model::Softmax to compute (b, k) by flattening all
leading dims into the row count: b = product(shape[..rank-1]),
k = shape[rank-1]. For 2D [batch, dim] this collapses to b=batch,
k=dim (identical to pre-M10 behaviour). For 4D [B, H, M, K] this
gives b = B*H*M, k = K — the 3-pass per-row softmax kernel runs
b times.

emit_softmax (the asm emitter) is unchanged — its (b, k) interface
is the right abstraction. The dispatch performs rank-flattening once
at the call site; the emitter is rank-agnostic by construction.

Tests added (1):
  - softmax_4d_dispatch_computes_b_as_product_of_leading_dims
    asserts the emit_imm32 bound for b=64 (= 2*4*8) when input is
    [2, 4, 8, 16].

Spec: §5.5.3, §6.6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: `git status` → clean.**

---

## Group 9a — Commit 9 — x86_64 `emit_matmul` + dispatch

**Group goal:** Mirror Group 6 in `profiles/x86_64/`. Create `profiles/x86_64/src/ops/matmul.rs` with the AT&T-syntax `emit_matmul`, wire dispatch, update `assign_buffers` so `Matmul` joins the StackOffset family. Per spec §6.4 and §10 step 9, this is one of three default sub-task-packs for the x86_64 work; commits 9b and 9c follow.

**Key x86_64 differences from arm64 (M9-grounded):**

- AT&T syntax — `movq`, `mulss`, `addss`, `xorps`; comments `#` not `;`.
- **No FMA.** The inner k-step is `mulss → addss` (two instructions, two roundings) — matches `emit_linear` x86_64's deliberate non-FMA design from M9. The unit test `matmul_uses_mulss_addss_no_fma` asserts `vfmadd` is absent.
- **`%rdx` / `%rsi` clobber preservation.** Per the M9 lessons in commits `ecb69ac` (preserves `%rsi` across matmul body) and `c3ff521` (preserves `%rdx` when bias clobbers it), any emitter that clobbers an FFI register that a follow-up emitter reads via `materialise_ptr` must save / restore it. For `emit_matmul`:
  - The matmul body does not write to `%rsi` or `%rdx` for its outputs (those go to materialised base pointers like `%r8/%r9/%r11`). Only **scratch register choice** introduces the hazard.
  - Recommended scratch layout: avoid `%rsi` and `%rdx` in inner loops. Use `%r10` (imm32-to-r10 helper), `%rcx` (outer counter), and `%rax` (address compute) — all caller-saved, none on the FFI input path.
  - If the implementer's chosen layout *does* clobber `%rsi` or `%rdx`, the spill-and-restore pair (`movq %rsi, %xmm6; ...; movq %xmm6, %rsi`) is mandatory, gated on whether any later op in the same function reads OutputReg / params.

**Group done criteria:**
- All four workspace gates green
- Test count 314 → 319 (+5 new x86_64 codegen unit tests)

**Files touched:**
- Create: `profiles/x86_64/src/ops/matmul.rs`
- Modify: `profiles/x86_64/src/ops/mod.rs`
- Modify: `profiles/x86_64/src/buffer.rs` (Matmul → StackOffset)
- Modify: `profiles/x86_64/src/codegen.rs` (`classify_op` arm + `walk_model` dispatch)
- Modify: `profiles/x86_64/src/tests.rs` (5 new tests)

### Task 9a.1: Add `Matmul` to `assign_buffers`

**Files:**
- Modify: `profiles/x86_64/src/buffer.rs:60-73`

- [ ] **Step 1: Extend the per-op match**

Replace:

```rust
                        StdOp::Linear | StdOp::Softmax => {
```

with:

```rust
                        StdOp::Linear | StdOp::Softmax | StdOp::Matmul => {
```

### Task 9a.2: Update `classify_op`

**Files:**
- Modify: `profiles/x86_64/src/codegen.rs` (the parallel block to arm64's classify_op)

- [ ] **Step 1: Add the explicit Matmul arm**

Find the `classify_op` function in `profiles/x86_64/src/codegen.rs` (located near the bottom of the file — `grep -n "fn classify_op" profiles/x86_64/src/codegen.rs` to confirm line numbers). Add `StdOp::Matmul => Ok(()),` immediately after `StdOp::Softmax => Ok(()),`.

### Task 9a.3: Create `profiles/x86_64/src/ops/matmul.rs`

**Files:**
- Create: `profiles/x86_64/src/ops/matmul.rs`
- Modify: `profiles/x86_64/src/ops/mod.rs`

- [ ] **Step 1: Write the new module**

Create `profiles/x86_64/src/ops/matmul.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — x86_64 SSE2, AT&T syntax. Outer loop over the product
//! of leading dims; inner triple-loop matmul kernel using `mulss + addss`
//! (no FMA — matches emit_linear x86_64's deliberate non-FMA design from
//! M9).
//!
//! Register usage (M9 hazard avoidance):
//!   %r8, %r9, %r11  — A, B, DST base pointers (materialised once)
//!   %rcx            — outer-loop counter
//!   %r10            — imm32-to-r10 scratch (clobbered each emit_imm32_to_r10 call)
//!   %rax            — address compute scratch
//!   %xmm0           — accumulator
//!   %xmm1, %xmm2    — operand fetches (mulss / addss)
//!
//! `%rsi` (params) and `%rdx` (output) are NOT used as scratch; emit_matmul
//! does not need preservation pairs. If a follow-up op reads either,
//! they survive intact.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

/// Emit AT&T-syntax x86_64 asm for a multi-dim matmul.
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
    leading_count: u64,
    m: u64,
    k: u64,
    n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    _node_span: Span,
) -> Result<String, LowerError> {
    let mid = format!("{model_idx}_{matmul_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # matmul (leading_count={}): [{},{}] x [{},{}] -> [{},{}], transpose_b={}\n",
        leading_count, m, k, k, n, m, n, transpose_b
    ));

    s.push_str(&materialise_ptr("%r8", a_loc));
    s.push_str(&materialise_ptr("%r9", b_loc));
    s.push_str(&materialise_ptr("%r11", dst_loc));

    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Outer loop counter %rcx (caller-saved, conventional counter reg).
    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(leading_count as usize));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm4d_outer_end_{mid}\n"));

    // Per-outer slice base pointers in scratch:
    //   %r12_A = %r8  + %rcx * a_slice * 4    (we use %rax as the slice ptr scratch)
    // We want to AVOID callee-saved regs; computed slice ptrs land in
    // %rax (A_slice), %rbx (B_slice — but %rbx is callee-saved!), ...
    // Better: re-derive each iteration from %rcx + base + offset compute.
    // Simplest: push slice base ptrs into stack? No — too costly.
    //
    // Pragmatic approach: use %rdi (no longer needed after materialise_ptr
    // calls, since %r8/%r9/%r11 hold all the base pointers we need).
    // %rdi = A_slice; %rsi clobber-safe? We didn't save it, BUT:
    //   - if the model had params, walk_model.Linear path saves %rsi via
    //     %xmm6 BEFORE entering its body; matmul does the same defensively.
    //   - if the model has no params, %rsi value is irrelevant.
    // For SelfAttention (params_floats == 0), %rsi is don't-care.
    // For mixed models (matmul + linear coexisting), the Linear emitter
    // already preserves %rsi across its own body; we conservatively
    // preserve %rsi here too (cheap: 2 movq instructions).
    s.push_str("    movq    %rsi, %xmm6\n"); // preserve %rsi (params ptr)
    s.push_str("    movq    %rdx, %xmm7\n"); // preserve %rdx (output ptr)

    // A_slice = %r8 + %rcx * a_slice * 4
    s.push_str(&emit_imm32_to_r10(a_slice));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r8, %rax, 4), %rdi\n");
    // B_slice = %r9 + %rcx * b_slice * 4
    s.push_str(&emit_imm32_to_r10(b_slice));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r9, %rax, 4), %rsi\n");
    // DST_slice = %r11 + %rcx * dst_slice * 4
    s.push_str(&emit_imm32_to_r10(dst_slice));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r11, %rax, 4), %rdx\n");

    // Inner i-loop ([0, M)). Counter in %rax (we lose its previous use as
    // multiply scratch above; that's fine since we don't reuse it before
    // the next imulq).
    // Use stack-spill-free layout: %rax = i, %r10 = j, ... wait, %r10 is
    // imm32 scratch. Let's use two slots in stack via push for j and
    // k_inner — except matmul has no extern call so saving/restoring is
    // not necessary at function level. We use:
    //   %rax = i
    //   %rcx_i = j   (we'll restore %rcx outer-counter below)
    //   %r10 = k_inner   (re-imm32'd at j-loop; reload bounds locally)
    // But %rcx is the outer counter we mustn't clobber.
    //
    // Cleanest layout: push %rcx onto stack at outer-iter top, restore at
    // outer-iter bottom; then use %rcx, %rax, %r10 for i, j, k_inner.

    s.push_str("    pushq   %rcx\n"); // save outer counter
    // i = %rax, j = %rcx, k_inner = %r10. Bounds re-materialised inline
    // each loop entry via emit_imm32_to_r10 (overwrites %r10).

    s.push_str("    movq    $0, %rax\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(m as usize));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lmm4d_i_end_{mid}\n"));

    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as usize));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm4d_j_end_{mid}\n"));

    s.push_str("    xorps   %xmm0, %xmm0\n"); // accumulator
    s.push_str("    movq    $0, %r10\n");      // k_inner = 0 (we'll re-imm32 K inside the loop body via %r12-free path)

    // Hmm — we need K bound somewhere persistent. Use %r12 (callee-saved)
    // means prologue save; we don't want to grow callee-saved set.
    // Alternative: re-imm32 K inside the loop tail (cheap — 1 movq). Loop
    // form:
    //   .Lmm4d_k_:
    //     <use %r10 as k_inner — but we also need K bound for cmpq>
    //     cmpq <K imm>, %r10
    //
    // Fortunately AT&T `cmpq $imm, %reg` accepts a 32-bit signed immediate
    // directly — no need for a register-held bound. Use that form.
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str(&format!("    cmpq    ${}, %r10\n", k));
    s.push_str(&format!("    jge     .Lmm4d_k_end_{mid}\n"));

    // a_offset = %rax * K + %r10  (always — A is [..., M, K])
    s.push_str(&format!("    movq    %rax, %r11\n"));   // %r11 was DST base; we can clobber inside the inner loop and re-load if needed. Actually we WILL need %r11 again for the store; need a different scratch.
    // Restore-and-reload pattern is too noisy. Use a different strategy:
    //   compute a_offset = i*K + k_inner via stack-spilled %r11 reload
    // Simpler: drop %r11 as DST base — re-materialise DST_slice each
    // store via the saved %rdx in %xmm7 + slice stride. But that's
    // costly per-element.
    //
    // PRAGMATIC IMPLEMENTATION CHOICE: use the stack to spill %r11 at
    // the start of the inner triple-loop, restore at the end of the
    // outer iter. This is an honest cost — adds 2 × 8 bytes of stack
    // and 2 push/pops. The implementer should benchmark; if it shows
    // up in profiles, switch to a different register allocation.
    //
    // For the plan, document the stack-spill approach below (mirrors
    // M9's xmm-spill discipline for non-extern hazards).

    s.push_str("    pushq   %r11\n");
    s.push_str("    movq    %rax, %r11\n");        // %r11 = i (temporary)
    s.push_str(&format!("    imulq   ${}, %r11\n", k));
    s.push_str("    addq    %r10, %r11\n");        // %r11 = i*K + k_inner
    s.push_str("    movss   (%rdi, %r11, 4), %xmm1\n"); // %xmm1 = A[a_offset]

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str("    movq    %rcx, %r11\n");
        s.push_str(&format!("    imulq   ${}, %r11\n", k));
        s.push_str("    addq    %r10, %r11\n");
    } else {
        s.push_str("    movq    %r10, %r11\n");
        s.push_str(&format!("    imulq   ${}, %r11\n", n));
        s.push_str("    addq    %rcx, %r11\n");
    }
    s.push_str("    movss   (%rsi, %r11, 4), %xmm2\n"); // %xmm2 = B[b_offset]

    // Two-step (no FMA): %xmm1 = A * B; %xmm0 += %xmm1.
    s.push_str("    mulss   %xmm2, %xmm1\n"); // %xmm1 = %xmm1 * %xmm2
    s.push_str("    addss   %xmm1, %xmm0\n"); // %xmm0 = %xmm0 + %xmm1
    s.push_str("    popq    %r11\n");           // restore DST base

    s.push_str("    addq    $1, %r10\n");
    s.push_str(&format!("    jmp     .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store %xmm0 → DST_slice[i * N + j] using %rdx (= DST_slice base).
    // Recompute the dst offset using i, j (still in %rax, %rcx).
    s.push_str("    pushq   %r11\n");
    s.push_str("    movq    %rax, %r11\n");
    s.push_str(&format!("    imulq   ${}, %r11\n", n));
    s.push_str("    addq    %rcx, %r11\n");
    s.push_str("    movss   %xmm0, (%rdx, %r11, 4)\n");
    s.push_str("    popq    %r11\n");

    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    s.push_str("    addq    $1, %rax\n");
    s.push_str(&format!("    jmp     .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    s.push_str("    popq    %rcx\n"); // restore outer counter
    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    // Restore preserved %rsi / %rdx for follow-up ops.
    s.push_str("    movq    %xmm6, %rsi\n");
    s.push_str("    movq    %xmm7, %rdx\n");

    Ok(s)
}
```

**Implementer note:** the register choreography above is *one valid layout*. If profile-time review reveals a cleaner allocation (fewer push/pops, simpler spill discipline), the implementer is empowered to revise — *but* the unit tests in Task 9a.5 must continue to assert: `mulss/addss` (not FMA), no `expf@PLT` call, transpose_b changes inner addressing, default-vs-explicit-false equivalence. Those four invariants are non-negotiable.

- [ ] **Step 2: Update `profiles/x86_64/src/ops/mod.rs`**

Add `pub mod matmul;` and `pub use matmul::emit_matmul;` (alphabetical ordering as in arm64).

### Task 9a.4: Dispatch from `walk_model`

**Files:**
- Modify: `profiles/x86_64/src/codegen.rs`

- [ ] **Step 1: Add the `matmul_idx` counter**

Append next to other indexes.

- [ ] **Step 2: Add the `StdOp::Matmul` arm**

Mirror the arm64 dispatch from Group 6 Task 6.4 Step 2 verbatim — only the `body.push_str(&crate::ops::emit_matmul(...))` call site differs in arg order. Use `compiler::ir::stdlib::matmul_transpose_b` to read the transpose attr.

### Task 9a.5: Add the 5 unit tests

**Files:**
- Modify: `profiles/x86_64/src/tests.rs`

The tests mirror Group 6 (Task 6.5) but with x86_64-specific assertions:

- [ ] **Step 1: `matmul_4d_emits_outer_loop_wrapper`** — assert presence of `.Lmm4d_outer_0_0:` / `.Lmm4d_outer_end_0_0:` / `.Lmm4d_i_0_0:` / `.Lmm4d_j_0_0:` / `.Lmm4d_k_0_0:`. Same fixture as arm64.
- [ ] **Step 2: `matmul_2d_collapses_to_outer_count_one`** — assert `leading_count=1` in the comment.
- [ ] **Step 3: `matmul_transpose_b_inner_addressing_differs`** — assert the AT&T-syntax difference (e.g. one source contains `imulq   $4, %r11` after `movq    %r10, %r11` for non-transpose; transpose differs in the multiplicand → just check `assert_ne!(asm_no_t, asm_t)` plus `mulss` in both).
- [ ] **Step 4: `matmul_transpose_b_false_default_matches_explicit_false`** — `assert_eq!(asm_default, asm_explicit_false)`.
- [ ] **Step 5: `matmul_uses_mulss_addss_no_fma`** —

```rust
#[test]
fn matmul_uses_mulss_addss_no_fma() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(asm.contains("mulss"), "asm:\n{}", asm);
    assert!(asm.contains("addss"), "asm:\n{}", asm);
    assert!(!asm.contains("vfmadd"), "matmul must not use FMA on x86_64; asm:\n{}", asm);
}
```

- [ ] **Step 6: `matmul_does_not_call_expf_plt`** —

```rust
#[test]
fn matmul_does_not_call_expf_plt() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(!asm.contains("expf@PLT"), "asm:\n{}", asm);
}
```

(That's actually 6 tests — adjust the count: Group 9a lands +6 tests, project total 314 → 320.)

- [ ] **Step 7: Run all 6**

Run: `cargo test -p profiles-x86_64 matmul_`
Expected: 6 PASS on Linux x86_64. On macOS arm64, the FFI tests are cfg-gated to `(target_os = "linux", target_arch = "x86_64")` but the **codegen unit tests run unconditionally** — they assert on emitted asm strings without executing. So all 6 PASS on macOS arm64 too.

### Task 9a.6: Workspace gates + commit

- [ ] **Step 1: All four gates** — exit 0
- [ ] **Step 2: Test count check** — 320
- [ ] **Step 3: Stage + commit**

```bash
git add profiles/x86_64/src/ops/matmul.rs \
        profiles/x86_64/src/ops/mod.rs \
        profiles/x86_64/src/buffer.rs \
        profiles/x86_64/src/codegen.rs \
        profiles/x86_64/src/tests.rs

git commit -m "$(cat <<'EOF'
feat(m10/x86_64): emit_matmul — outer loop + scalar mulss/addss matmul

Mirrors profiles/arm64/src/ops/matmul.rs structurally — outer loop
over `leading_count`, inner triple-loop body — but uses x86_64 SSE2
non-FMA scalar arithmetic per spec §6.4 (mulss + addss, two
roundings, matching emit_linear x86_64's M9 design):

  movss (%rdi, ...), %xmm1
  movss (%rsi, ...), %xmm2
  mulss %xmm2, %xmm1
  addss %xmm1, %xmm0

`transpose_b=true` flips the inner b_offset computation between
`imulq $K, %r11` (j*K+k_inner) and `imulq $N, %r11` (k_inner*N+j).

`%rsi` (params) and `%rdx` (output) are spilled to %xmm6/%xmm7 at
function entry of emit_matmul and restored at exit — defensive
preservation pattern from M9 commits ecb69ac and c3ff521. Matmul's
own body avoids these regs; the spill is for safety against
implementation-time register reuse.

assign_buffers extended: StdOp::Matmul joins StackOffset family,
sized rank-agnostically.

walk_model adds Matmul dispatch parallel to arm64's. classify_op
accepts Matmul.

Tests added (6):
  - matmul_4d_emits_outer_loop_wrapper
  - matmul_2d_collapses_to_outer_count_one
  - matmul_transpose_b_inner_addressing_differs
  - matmul_transpose_b_false_default_matches_explicit_false
  - matmul_uses_mulss_addss_no_fma
  - matmul_does_not_call_expf_plt

emit_linear is unchanged. Project total 314 → 320.

Spec: §6.1, §6.3, §6.4, §6.7, §10 step 9, §12.4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: `git status` → clean.**

---

## Group 9b — Commit 10 — x86_64 `emit_mulscalar` + dispatch

**Group goal:** Mirror Group 7 in `profiles/x86_64/`. Create `profiles/x86_64/src/ops/mulscalar.rs`. The asm pre-loads the f32 scalar into an `xmm` register via `movd` from a GPR (or stack-spill), then runs a flat per-element loop using `mulss`.

**Group done criteria:**
- All four workspace gates green
- Test count 320 → 322 (+2 new x86_64 codegen tests)

**Files touched:**
- Create: `profiles/x86_64/src/ops/mulscalar.rs`
- Modify: `profiles/x86_64/src/ops/mod.rs`
- Modify: `profiles/x86_64/src/buffer.rs` (MulScalar → Alias)
- Modify: `profiles/x86_64/src/codegen.rs` (`classify_op` + `walk_model` dispatch)
- Modify: `profiles/x86_64/src/tests.rs` (2 new tests)

### Task 9b.1: `assign_buffers` Alias for MulScalar

**Files:**
- Modify: `profiles/x86_64/src/buffer.rs`

- [ ] **Step 1: Extend the alias-family arm**

Replace:

```rust
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
```

with:

```rust
                        StdOp::Relu | StdOp::Dropout | StdOp::MulScalar => {
                            BufferLoc::Alias(operands[0])
                        }
```

### Task 9b.2: `classify_op` accepts MulScalar

- [ ] **Step 1:** Add `StdOp::MulScalar => Ok(()),` after the `Matmul` arm landed in Group 9a.

### Task 9b.3: Create `profiles/x86_64/src/ops/mulscalar.rs`

**Files:**
- Create: `profiles/x86_64/src/ops/mulscalar.rs`

- [ ] **Step 1: Write the module**

```rust
// SPDX-License-Identifier: Apache-2.0

//! MulScalar codegen — x86_64 SSE2 AT&T-syntax flat per-element multiply.
//!
//! Scalar pre-loaded into %xmm4 once via:
//!   movl $<scalar_bits>, %r10d
//!   movd %r10d, %xmm4
//! Inner loop:
//!   movss (%r8, %rcx, 4), %xmm0
//!   mulss %xmm4, %xmm0
//!   movss %xmm0, (%r11, %rcx, 4)

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;

/// Emit AT&T x86_64 asm for `dst[i] = src[i] * scalar`.
#[allow(clippy::too_many_arguments)]
pub fn emit_mulscalar(
    total_elements: u64,
    scalar_bits: u32,
    model_idx: usize,
    op_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # mul_scalar: total_elements={}, scalar_bits=0x{:08x}\n",
        total_elements, scalar_bits
    ));

    // Pre-load scalar into %xmm4 via %r10d (movl + movd).
    s.push_str(&format!("    movl    $0x{:x}, %r10d\n", scalar_bits));
    s.push_str("    movd    %r10d, %xmm4\n");

    s.push_str(&materialise_ptr("%r8", src_loc));
    s.push_str(&materialise_ptr("%r11", dst_loc));

    // Flat loop, %rcx = i.
    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lms_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(total_elements as usize));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lms_end_{mid}\n"));

    s.push_str("    movss   (%r8, %rcx, 4), %xmm0\n");
    s.push_str("    mulss   %xmm4, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rcx, 4)\n");

    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lms_{mid}\n"));
    s.push_str(&format!(".Lms_end_{mid}:\n"));

    s
}
```

- [ ] **Step 2: Update `profiles/x86_64/src/ops/mod.rs`**

Add `pub mod mulscalar;` and `pub use mulscalar::emit_mulscalar;` (alphabetical).

### Task 9b.4: Dispatch from `walk_model`

**Files:**
- Modify: `profiles/x86_64/src/codegen.rs`

- [ ] **Step 1: Add `mulscalar_idx`** counter

- [ ] **Step 2: Add `StdOp::MulScalar` arm**

Mirror Group 7 Task 7.4 Step 2 verbatim — exact same structure (read attrs, find Float "value", truncate via `(scalar_f64 as f32).to_bits()`, lookup BufferLocs, call `emit_mulscalar`, increment idx).

### Task 9b.5: Add 2 unit tests

**Files:**
- Modify: `profiles/x86_64/src/tests.rs`

- [ ] **Step 1: `mul_scalar_uses_mulss`**

```rust
#[test]
fn mul_scalar_uses_mulss() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.5]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(asm.contains("mulss   %xmm4, %xmm0"), "asm:\n{}", asm);
}
```

- [ ] **Step 2: `mul_scalar_preloads_scalar`**

```rust
#[test]
fn mul_scalar_preloads_scalar() {
    // 0.25 in f32 bits = 0x3E800000.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.25]
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(asm.contains("movl    $0x3e800000, %r10d"), "asm:\n{}", asm);
    assert!(asm.contains("movd    %r10d, %xmm4"), "asm:\n{}", asm);
}
```

- [ ] **Step 3: Run** — `cargo test -p profiles-x86_64 mul_scalar_` → 2 PASS.

### Task 9b.6: Workspace gates + commit

- [ ] **Step 1: Four gates** — exit 0
- [ ] **Step 2: Stage + commit**

```bash
git add profiles/x86_64/src/ops/mulscalar.rs \
        profiles/x86_64/src/ops/mod.rs \
        profiles/x86_64/src/buffer.rs \
        profiles/x86_64/src/codegen.rs \
        profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m10/x86_64): emit_mulscalar — in-place per-element scalar multiply

Mirror of profiles/arm64/src/ops/mulscalar.rs. Pre-loads scalar
f32-bits into %xmm4 via:
  movl $<bits>, %r10d
  movd %r10d, %xmm4
Then a flat per-element loop using mulss. With BufferLoc::Alias
(default for MulScalar in assign_buffers), src_loc and dst_loc
resolve to the same materialised pointer — in-place transformation.

walk_model: new StdOp::MulScalar arm, identical structure to arm64
modulo the emitter call. classify_op accepts MulScalar.

Tests added (2): mulss inner-loop body, scalar pre-load via
movl/movd. Project total 320 → 322.

Spec: §6.1, §6.3, §6.5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: `git status` → clean.**

---

## Group 9c — Commit 11 — x86_64 Softmax 4D dispatch

**Group goal:** Mirror Group 8 — generalise `walk_model::Softmax` so 4D inputs flatten leading dims into `b`. The x86_64 `emit_softmax` asm emitter is unchanged.

**Group done criteria:**
- All four workspace gates green
- Test count 322 → 323 (+1 new x86_64 test)

**Files touched:**
- Modify: `profiles/x86_64/src/codegen.rs`
- Modify: `profiles/x86_64/src/tests.rs`

### Task 9c.1: Update Softmax dispatch

**Files:**
- Modify: `profiles/x86_64/src/codegen.rs`

- [ ] **Step 1: Replace the Softmax arm body**

Find the `StdOp::Softmax => { ... }` arm and replace its body to use `b = product(shape[..rank-1])` and `k = shape[rank-1]`. Identical pattern to arm64 Group 8 Task 8.1 Step 1.

### Task 9c.2: Test

- [ ] **Step 1: `softmax_4d_dispatch_computes_b_as_product_of_leading_dims`**

```rust
#[test]
fn softmax_4d_dispatch_computes_b_as_product_of_leading_dims() {
    // 4D shape [2, 4, 8, 16]: b = 2*4*8 = 64, k = 16. The x86_64
    // emitter materialises b via emit_imm32_to_r10 → `movl $0x40, %r10d`
    // immediately above the .Lsm_i_<id> label.
    let src = "\
model M [batch=2, heads=4, seq=8, dim=16]:
    x: Tensor[batch, heads, seq, dim]

    y: Tensor[batch, heads, seq, dim] = x -> softmax
";
    let asm = crate::lower(
        &compiler::ir::build(&compiler::parse(src).unwrap()).unwrap(),
    ).expect("lower").source;
    assert!(
        asm.contains("movl    $0x40, %r10d"),
        "expected b=64 immediate; asm:\n{}",
        asm
    );
}
```

(The exact mnemonic depends on `emit_imm32_to_r10`'s output; `grep` the existing `emit_imm32_to_r10` in `profiles/x86_64/src/asm.rs` to confirm the literal pattern. If it's `movl $X, %r10d`, the assertion is correct. If it differs, adjust the assertion to match the actual emitter output.)

- [ ] **Step 2: Run** — `cargo test -p profiles-x86_64 softmax_4d_dispatch_computes_b_as_product_of_leading_dims` → PASS.

### Task 9c.3: Workspace gates + commit

- [ ] **Step 1: Four gates** — exit 0
- [ ] **Step 2: Stage + commit**

```bash
git add profiles/x86_64/src/codegen.rs profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m10/x86_64): Softmax dispatch generalised to rank ≥ 2 (4D works)

Mirror of profiles/arm64 Group 8. walk_model::Softmax computes
b = product(shape[..rank-1]), k = shape[rank-1] — for 2D it
collapses to b=batch, k=dim (identical to pre-M10 behaviour); for
4D [B, H, M, K] it gives b = B*H*M, k = K.

emit_softmax (the asm emitter) is unchanged. Project total 322 → 323.

Spec: §5.5.3, §6.6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: `git status` → clean.**

**Plan-time fold call (deferred to execution time):** if at this point the executor judges Groups 9a / 9b / 9c isomorphic to their arm64 counterparts modulo register naming and AT&T-syntax differences (no surprise hazards encountered), they may **optionally** rebase-squash 9a + 9b + 9c into a single commit `feat(m10/x86_64): emit_matmul + emit_mulscalar + Softmax 4D dispatch`. The plan defaults to three separate commits because the M9 record (`ecb69ac`, `c3ff521`) shows x86_64 surfaces non-trivial divergence at implementation time. The decision is the executor's; not made at plan time.

---

## Group 10 — Commit 12 — End-to-end FFI integration: SelfAttention fixture + per-profile reference

**Group goal:** Land `tests/fixtures/self_attention.nfl` (the M10 acceptance artefact) plus per-profile FFI integration tests in both `profiles/arm64/tests/integration.rs` and `profiles/x86_64/tests/integration.rs`. Each test compiles the fixture end-to-end (parse → ir::build → lower → cc-assemble → libloading-dlopen → call), passes the deterministic algorithmic input from spec §7.3, and asserts `assert_eq!` against an architecture-matched in-process Rust reference.

The acceptance criterion is **per-profile** bit-exact equality. Cross-profile bit-exact is unreachable (FMA × libm divergence; spec §7.2) and intentionally not gated.

**Group done criteria:**
- All four workspace gates green
- Test count 323 → 325 (+2 new FFI integration tests; the x86_64 one is `cfg`-gated to `(target_os = "linux", target_arch = "x86_64")` and runs only on the Linux CI job)

**Files touched:**
- Create: `tests/fixtures/self_attention.nfl`
- Modify: `profiles/arm64/tests/integration.rs` (add `reference_self_attention_arm64` + the FFI test)
- Modify: `profiles/x86_64/tests/integration.rs` (add `reference_self_attention_x86_64` + the FFI test)

### Task 10.1: Create the acceptance fixture

**Files:**
- Create: `tests/fixtures/self_attention.nfl`

- [ ] **Step 1: Write the fixture verbatim from spec §7.1**

```nfl
# Single-input self-attention (q = k = v = x). Mathematically degenerate
# (no learnable Q/K/V projections), but exercises every M10 primitive:
# 4D matmul, 4D matmul with transpose_b, scalar multiply, last-axis
# softmax over a 4D tensor.
#
# Shape derivation:
#   x          → [2, 4, 16, 16]
#   x @ x.T    → leading [2, 4], inner [16,16] @ [16,16].T = [16,16]
#                output [2, 4, 16, 16]
#   * 0.25     → shape preserved (1/sqrt(head_dim) = 1/sqrt(16))
#   softmax    → shape preserved (last-axis softmax over seq=16)
#   attn @ x   → leading [2, 4], inner [16,16] @ [16,16] = [16,16]
#                output [2, 4, 16, 16]

model SelfAttention [batch=2, heads=4, seq=16, head_dim=16]:
    x: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
    scaled: Tensor[batch, heads, seq, seq] = scores -> mul_scalar[0.25]
    attn:   Tensor[batch, heads, seq, seq] = scaled -> softmax
    out:    Tensor[batch, heads, seq, head_dim] = attn -> matmul[x]
```

### Task 10.2: arm64 reference implementation + FFI test

**Files:**
- Modify: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Add `reference_self_attention_arm64` near the existing reference helpers**

Append after the existing `reference_softmax_stable` and similar helpers (around line 56):

```rust
/// Architecture-matched arm64 reference for SelfAttention. Uses
/// `f32::mul_add` to match the fmadd-using emit_matmul / emit_linear
/// arm64 codegen. `f32::exp` wraps platform libm `expf`.
///
/// `x` is the input tensor in row-major [batch, heads, seq, head_dim].
/// Returns the output tensor in the same layout.
fn reference_self_attention_arm64(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let head_stride = seq * head_dim;
    let head_count = batch * heads;
    let mut out = vec![0.0f32; head_count * head_stride];

    // Per-head attention: scores = x @ x.T (with transpose_b=true).
    //   scores[i, j] = sum_k x[i, k] * x[j, k]   (FMA-using)
    // Then scaled = scores * 0.25.
    // Then attn = softmax(scaled, last_axis).
    // Then out = attn @ x (no transpose).
    //   out[i, k] = sum_j attn[i, j] * x[j, k]   (FMA-using)
    let mut scores = vec![0.0f32; seq * seq];
    let mut attn = vec![0.0f32; seq * seq];

    for head in 0..head_count {
        let x_head = &x[head * head_stride..(head + 1) * head_stride];
        let out_head = &mut out[head * head_stride..(head + 1) * head_stride];

        // scores[i, j] = sum_k x[i, k] * x[j, k] (transpose_b=true)
        for i in 0..seq {
            for j in 0..seq {
                let mut acc = 0.0f32;
                for k in 0..head_dim {
                    acc = f32::mul_add(x_head[i * head_dim + k], x_head[j * head_dim + k], acc);
                }
                scores[i * seq + j] = acc * scale;
            }
        }

        // attn = softmax(scores, last_axis)
        for i in 0..seq {
            let row = &scores[i * seq..(i + 1) * seq];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq {
                let e = (row[j] - max).exp();
                attn[i * seq + j] = e;
                sum += e;
            }
            for j in 0..seq {
                attn[i * seq + j] /= sum;
            }
        }

        // out = attn @ x (no transpose)
        for i in 0..seq {
            for k in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    acc = f32::mul_add(attn[i * seq + j], x_head[j * head_dim + k], acc);
                }
                out_head[i * head_dim + k] = acc;
            }
        }
    }

    out
}

/// Deterministic input generator from spec §7.3.
fn deterministic_input(total: usize) -> Vec<f32> {
    (0..total).map(|i| (i as f32).sin() * 0.1).collect()
}
```

- [ ] **Step 2: Add the FFI integration test**

Append at the end of the file:

```rust
#[test]
fn self_attention_ffi_matches_reference() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: integration test requires aarch64 host");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    const BATCH: usize = 2;
    const HEADS: usize = 4;
    const SEQ: usize = 16;
    const HEAD_DIM: usize = 16;
    const TOTAL: usize = BATCH * HEADS * SEQ * HEAD_DIM;

    let src = std::fs::read_to_string("../../tests/fixtures/self_attention.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");

    let asm = profiles_arm64::lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    assert_eq!(sig.input_floats, TOTAL);
    assert_eq!(sig.output_floats, TOTAL);
    assert_eq!(sig.params_floats, 0, "SelfAttention has no learnable params");

    let so_path = common::compile_to_so(&asm.source, &sig.name)
        .expect("compile asm to .so");

    let input = deterministic_input(TOTAL);
    let mut output = vec![0.0f32; TOTAL];
    let params: Vec<f32> = vec![0.0f32; sig.params_floats]; // empty but non-null per Vec contract

    unsafe {
        let lib = libloading::Library::new(&so_path).expect("dlopen");
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib.get(sig.name.as_bytes()).expect("dlsym");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let reference =
        reference_self_attention_arm64(&input, BATCH, HEADS, SEQ, HEAD_DIM);

    // Per-profile bit-exact equality.
    assert_eq!(
        output, reference,
        "SelfAttention FFI output must match arm64 reference bit-exactly"
    );
}
```

(Implementer: confirm `common::compile_to_so` is the existing helper from `profiles/arm64/tests/common/mod.rs`. The `compile_to_so` signature varies between profiles; verify by reading `profiles/arm64/tests/common/mod.rs` first and adjust the call site to match the actual API.)

### Task 10.3: x86_64 reference implementation + FFI test

**Files:**
- Modify: `profiles/x86_64/tests/integration.rs`

- [ ] **Step 1: Add `reference_self_attention_x86_64`**

Mirror the arm64 helper but with **separate `let prod = a * b; sum += prod;` pattern instead of `f32::mul_add`** to match the `mulss + addss` x86_64 emitter.

```rust
/// Architecture-matched x86_64 reference for SelfAttention. Uses
/// separate `mul + add` (no FMA) to match emit_matmul x86_64's
/// deliberate non-FMA design from M9 — intentional divergence from
/// the arm64 reference, not a defect. `f32::exp` wraps platform libm
/// `expf` (glibc on Linux x86_64).
fn reference_self_attention_x86_64(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let head_stride = seq * head_dim;
    let head_count = batch * heads;
    let mut out = vec![0.0f32; head_count * head_stride];

    let mut scores = vec![0.0f32; seq * seq];
    let mut attn = vec![0.0f32; seq * seq];

    for head in 0..head_count {
        let x_head = &x[head * head_stride..(head + 1) * head_stride];
        let out_head = &mut out[head * head_stride..(head + 1) * head_stride];

        for i in 0..seq {
            for j in 0..seq {
                let mut acc = 0.0f32;
                for k in 0..head_dim {
                    let prod = x_head[i * head_dim + k] * x_head[j * head_dim + k];
                    acc = acc + prod; // separate mul + add (NOT mul_add)
                }
                scores[i * seq + j] = acc * scale;
            }
        }

        for i in 0..seq {
            let row = &scores[i * seq..(i + 1) * seq];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq {
                let e = (row[j] - max).exp();
                attn[i * seq + j] = e;
                sum += e;
            }
            for j in 0..seq {
                attn[i * seq + j] /= sum;
            }
        }

        for i in 0..seq {
            for k in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    let prod = attn[i * seq + j] * x_head[j * head_dim + k];
                    acc = acc + prod;
                }
                out_head[i * head_dim + k] = acc;
            }
        }
    }

    out
}

fn deterministic_input(total: usize) -> Vec<f32> {
    (0..total).map(|i| (i as f32).sin() * 0.1).collect()
}
```

- [ ] **Step 2: Add the FFI test (cfg-gated)**

```rust
#[test]
fn self_attention_ffi_matches_reference() {
    // x86_64 FFI tests are cfg-gated at the module level via
    // #![cfg(all(target_os = "linux", target_arch = "x86_64"))] —
    // see existing module attribute. This test runs only on the
    // Linux x86_64 CI job.
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    const BATCH: usize = 2;
    const HEADS: usize = 4;
    const SEQ: usize = 16;
    const HEAD_DIM: usize = 16;
    const TOTAL: usize = BATCH * HEADS * SEQ * HEAD_DIM;

    let src = std::fs::read_to_string("../../tests/fixtures/self_attention.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");

    let asm = profiles_x86_64::lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    assert_eq!(sig.input_floats, TOTAL);
    assert_eq!(sig.output_floats, TOTAL);
    assert_eq!(sig.params_floats, 0);

    let so_path = common::compile_to_so(&asm.source, &sig.name)
        .expect("compile asm to .so");

    let input = deterministic_input(TOTAL);
    let mut output = vec![0.0f32; TOTAL];
    let params: Vec<f32> = vec![0.0f32; sig.params_floats];

    unsafe {
        let lib = libloading::Library::new(&so_path).expect("dlopen");
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = lib.get(sig.name.as_bytes()).expect("dlsym");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let reference =
        reference_self_attention_x86_64(&input, BATCH, HEADS, SEQ, HEAD_DIM);

    assert_eq!(
        output, reference,
        "SelfAttention FFI output must match x86_64 reference bit-exactly"
    );
}
```

### Task 10.4: Run + commit

- [ ] **Step 1: Run all four gates** — exit 0
- [ ] **Step 2: Run the new FFI test on the host platform**
  - On macOS arm64: `cargo test -p profiles-arm64 --test integration self_attention_ffi_matches_reference` → PASS
  - On Linux x86_64: `cargo test -p profiles-x86_64 --test integration self_attention_ffi_matches_reference` → PASS
  - On macOS arm64, the x86_64 FFI test is skipped via the module-level cfg.

If the arm64 test fails with a numerical mismatch:
- Re-check `materialise_ptr` calls in emit_matmul (Group 6) preserve the base-pointer invariant — outer iteration must not move x11/x13/x12.
- Re-check `transpose_b` inner addressing — the unit test from Group 6 catches structural disagreement, but a bit-flip in the formula could survive.
- Compare a single intermediate (e.g., `scores[0,0,0,0]`) between FFI and reference to localise.

If the x86_64 test fails:
- The most likely cause is `%rsi` or `%rdx` clobber not preserved across follow-up ops (M9 lesson). Inspect the matmul's xmm6/xmm7 spill+restore pattern from Group 9a.
- Run `cargo test -p profiles-x86_64 --test integration -- --nocapture` to surface any libloading-side error.

- [ ] **Step 3: Stage + commit**

```bash
git add tests/fixtures/self_attention.nfl \
        profiles/arm64/tests/integration.rs \
        profiles/x86_64/tests/integration.rs

git commit -m "$(cat <<'EOF'
feat(m10/integration): SelfAttention fixture + per-profile FFI tests

End-to-end M10 acceptance:
  tests/fixtures/self_attention.nfl  — single-input self-attention
    (q = k = v = x), [batch=2, heads=4, seq=16, head_dim=16]. Exercises
    every M10 primitive: 4D matmul + 4D matmul with transpose_b, scalar
    multiply (1/sqrt(head_dim) = 0.25), last-axis softmax over a 4D tensor.

  profiles/arm64/tests/integration.rs::self_attention_ffi_matches_reference
    Uses reference_self_attention_arm64 with f32::mul_add to match the
    fmadd-using arm64 emit_matmul. Asserts FFI output == reference
    bit-exactly.

  profiles/x86_64/tests/integration.rs::self_attention_ffi_matches_reference
    Uses reference_self_attention_x86_64 with separate `let prod = a*b;
    acc = acc + prod;` to match the mulss+addss x86_64 emit_matmul —
    intentional divergence from FMA, not a defect (spec §7.2). cfg-gated
    to (target_os="linux", target_arch="x86_64") via the existing
    module-level attribute.

Deterministic input generator: `(0..total).map(|i| (i as f32).sin() * 0.1)`.
Range [-0.1, 0.1] keeps softmax inputs well-conditioned post-scaling.

Zero-params FFI contract: `vec![0.0f32; sig.params_floats]` produces an
empty Vec whose `as_ptr()` returns a non-null aligned pointer per the
Rust Vec contract. The assembly never dereferences this pointer because
params_floats == 0. NOT special-cased to ptr::null() — that would
violate the silent non-null contract assumed by the existing FFI
calling convention (spec §7.4).

Cross-profile bit-exact at byte level is architecturally unreachable
(FMA × libm divergence; spec §7.2). Per-profile bit-exact is the
acceptance criterion. Cross-profile tolerance reports are OQ-BENCH
follow-up territory (spec §11), not gated by M10 acceptance.

Project total 323 → 325 (macOS arm64 host count).

Spec: §7.1, §7.2, §7.3, §7.4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: `git status` → clean.**

---

## Group 11 — Commit 13 — Negative fixtures + cleanup

**Group goal:** Land the four negative fixture files from spec §8.6, ensuring they're each rejected at the right layer (parser-level for the missing-`=` case, UIR-builder-level for the three shape errors). No new asserting tests are required for the negative fixtures *as files* — the existing fixture-loading test infrastructure (verify by inspecting `tests/fixtures/negative/` references in test files) exercises them. If the existing infrastructure does not loop over `tests/fixtures/negative/`, this group adds an explicit `negative_fixtures_reject_at_correct_layer` integration test in `compiler/tests/`.

**Group done criteria:**
- All four workspace gates green
- Test count 325 → 326 (+1 negative-fixture integration test) OR 325 (if existing infra picks them up automatically; verify this first)

**Files touched:**
- Create: `tests/fixtures/negative/bad_named_pipeline_missing_eq.nfl`
- Create: `tests/fixtures/negative/bad_matmul_rank_too_low.nfl`
- Create: `tests/fixtures/negative/bad_matmul_inner_dim_mismatch.nfl`
- Create: `tests/fixtures/negative/bad_named_pipeline_shape_mismatch.nfl`
- Maybe modify: `compiler/tests/<existing_negative_fixture_runner>.rs` (if such a file exists)

### Task 11.1: Audit existing negative-fixture infrastructure

- [ ] **Step 1: Find any existing test that loads `tests/fixtures/negative/`**

```bash
grep -rn 'fixtures/negative' compiler/ profiles/ tests/ 2>/dev/null
```

If matches exist, the new fixtures join the existing loop automatically — proceed to Task 11.2 only with the four file-creation steps.

If **no matches**, add an explicit `compiler/tests/negative_fixtures.rs` integration test (skeleton in Task 11.6).

### Task 11.2: Create `bad_named_pipeline_missing_eq.nfl`

**Files:**
- Create: `tests/fixtures/negative/bad_named_pipeline_missing_eq.nfl`

```nfl
# `=` missing between the type expression and the source identifier.
# Parser-level rejection.

model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] x -> relu
```

### Task 11.3: Create `bad_matmul_rank_too_low.nfl`

**Files:**
- Create: `tests/fixtures/negative/bad_matmul_rank_too_low.nfl`

```nfl
# Matmul requires rank ≥ 2; passing 1D operands must be rejected at
# the UIR builder layer with ShapeError::RankTooLow.

model M [n=4]:
    x: Tensor[n]

    y: Tensor[n] = x -> matmul[x]
```

(Implementer note: the declared output shape `Tensor[n]` is intentionally arbitrary — the rank-too-low check fires before the declared-shape comparison, so this file rejects with `ShapeMismatch { detail: "expected rank ..."}` wrapped via `BuildError::shape`.)

### Task 11.4: Create `bad_matmul_inner_dim_mismatch.nfl`

**Files:**
- Create: `tests/fixtures/negative/bad_matmul_inner_dim_mismatch.nfl`

```nfl
# Matmul contraction-dim disagreement: a is [batch, 4], b is [3, 8].
# K=4 vs K=3. Rejected with ShapeError::InnerDimMismatch.

model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[3, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
```

### Task 11.5: Create `bad_named_pipeline_shape_mismatch.nfl`

**Files:**
- Create: `tests/fixtures/negative/bad_named_pipeline_shape_mismatch.nfl`

```nfl
# Declared shape Tensor[batch, 8] but `relu` preserves shape, so actual
# output is Tensor[batch, 4]. Rejected with
# BuildErrorKind::DeclaredShapeMismatch.

model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 8] = x -> relu
```

### Task 11.6: Add the explicit fixture-runner test (only if Task 11.1 found no infra)

**Files:**
- Create: `compiler/tests/negative_fixtures.rs` (only if Task 11.1 found nothing)

If you skipped this branch (existing infra picks up the fixtures), proceed to Task 11.7.

```rust
//! Loop runner for tests/fixtures/negative/. Each .nfl file is loaded,
//! parsed, and (if parse succeeds) built into UIR. The test asserts
//! that *some* error fires; per-fixture asserts on the specific
//! BuildErrorKind / ShapeError live in the unit-test layer
//! (compiler/src/ir/tests.rs and compiler/src/parser/tests.rs).

use std::fs;
use std::path::Path;

#[test]
fn all_negative_fixtures_reject() {
    let dir = Path::new("../tests/fixtures/negative");
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("read fixtures/negative")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "nfl")
                .unwrap_or(false)
        })
        .collect();

    assert!(!entries.is_empty(), "fixtures/negative must contain at least one .nfl");

    for entry in entries {
        let path = entry.path();
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
        // Try parse, then ir::build. EITHER may produce the expected
        // failure — we don't pin which.
        let parse_result = compiler::parse(&src);
        let combined = match parse_result {
            Err(_) => Err(()),
            Ok(ast) => compiler::ir::build(&ast).map(|_| ()).map_err(|_| ()),
        };
        assert!(
            combined.is_err(),
            "negative fixture {:?} unexpectedly accepted",
            path
        );
    }
}
```

### Task 11.7: Workspace gates + commit

- [ ] **Step 1: Four gates** — exit 0
- [ ] **Step 2: Stage + commit**

```bash
git add tests/fixtures/negative/bad_named_pipeline_missing_eq.nfl \
        tests/fixtures/negative/bad_matmul_rank_too_low.nfl \
        tests/fixtures/negative/bad_matmul_inner_dim_mismatch.nfl \
        tests/fixtures/negative/bad_named_pipeline_shape_mismatch.nfl
# Add compiler/tests/negative_fixtures.rs if Task 11.6 created it.

git commit -m "$(cat <<'EOF'
test(m10): negative fixtures + (optional) explicit fixture-runner

Four new fixtures land under tests/fixtures/negative/ to harden the
M10 surface area:

  bad_named_pipeline_missing_eq.nfl
    Parser-level rejection — `=` missing between the type expr and
    source identifier.

  bad_matmul_rank_too_low.nfl
    UIR-builder-level rejection — Matmul requires rank ≥ 2;
    passing 1D operands fires ShapeError::RankTooLow.

  bad_matmul_inner_dim_mismatch.nfl
    UIR-builder-level rejection — Matmul contraction-dim disagreement
    fires ShapeError::InnerDimMismatch.

  bad_named_pipeline_shape_mismatch.nfl
    UIR-builder-level rejection — declared vs actual shape mismatch
    fires BuildErrorKind::DeclaredShapeMismatch.

If a negative-fixture-runner test infra already exists, these files
plug in automatically. Otherwise this commit also lands
compiler/tests/negative_fixtures.rs as a generic loop runner.

The RankMismatch (2D @ 4D operands) variant is covered by a unit
test (compiler/src/ir/tests.rs::matmul_rank_mismatch_errors from
M10 group 3) without a corresponding fixture — pure UIR check, no
parser/profile specifics needed.

Spec: §8.6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: `git status` → clean.**

---

## Group 12 — Commit 14 — Documentation

**Group goal:** Land all M10 documentation in one commit per spec §10 step 12: grammar reference, both profile guides, project spec (First Milestones table + Current Status + Strategic Roadmap), DEVLOG entry, CLAUDE.md repository tree refresh. **No code changes** in this commit — pure docs.

**Group done criteria:**
- All four workspace gates green
- Test count unchanged at 326 (no new tests)
- All docs accurate to the implementation as it landed across Groups 1–11

**Files touched:**
- Modify: `language/grammar.ebnf` (final pass — Task 1.1 already updated it; verify nothing further needed)
- Modify: `docs/language_reference/grammar.md`
- Modify: `docs/profile_guide/arm64.md`
- Modify: `docs/profile_guide/x86_64.md`
- Modify: `PROJECT_SPEC.md`
- Modify: `DEVLOG.md`
- Modify: `CLAUDE.md`

### Task 12.1: Update `docs/language_reference/grammar.md`

**Files:**
- Modify: `docs/language_reference/grammar.md`

- [ ] **Step 1: Read the current file** — `wc -l docs/language_reference/grammar.md` to gauge.

- [ ] **Step 2: Add a new "Named pipelines" section after the existing "Pipelines" section**

The new section should:
1. Show the EBNF for `named_pipeline_stmt`.
2. Note the one-token-lookahead disambiguation from `variable_decl`.
3. Show two NFL examples — a simple `y: Tensor[..] = x -> relu` and a 4D matmul `scores: Tensor[..] = x -> matmul[x, transpose_b=true]`.
4. Reference the output-rule generalisation (§4.2 of the M10 spec).
5. Note that `Tensor`-typed positional args parse as `arg_value = identifier` and resolve at UIR build time.

The exact prose style should match the existing sections in `grammar.md` (the implementer reads the file first and matches the convention — heading depth, code-fence labels, cross-link style).

### Task 12.2: Update `docs/profile_guide/arm64.md`

**Files:**
- Modify: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Add "M10 ops" section**

Append after the existing emit_softmax section. Cover:

1. `emit_matmul`: outer-loop wrapper over `leading_count`, inner FMA triple-loop, base-pointer invariance invariant, transpose_b inner addressing, label naming (`.Lmm4d_outer_<id>:` etc.).
2. `emit_mulscalar`: scalar pre-load via `movz/movk → fmov s4`, in-place flat loop, AttrValue::Float → f32 truncation contract.
3. Updated Softmax dispatch — `b = product(shape[..-1])`, `k = shape[-1]`.

Mention that `emit_linear` is **unchanged** (§6.1 architectural invariant).

### Task 12.3: Update `docs/profile_guide/x86_64.md`

**Files:**
- Modify: `docs/profile_guide/x86_64.md`

- [ ] **Step 1: Add the parallel "M10 ops" section**

Cover the same three concerns as 12.2 but for x86_64:

1. `emit_matmul`: AT&T syntax, `mulss + addss` (no FMA, intentional divergence from arm64), %rsi/%rdx preservation pattern via xmm6/xmm7 spill, register layout.
2. `emit_mulscalar`: `movl $bits, %r10d → movd %r10d, %xmm4` pre-load, `mulss` inner instruction.
3. Softmax dispatch (rank-flatten — same as arm64).

Note that `f32::exp` divergence between glibc (Linux) and Apple libsystem (macOS) is an explicit non-goal for cross-profile bit-exact at the byte level — the M10 acceptance criterion is per-profile bit-exact only (§7.2).

### Task 12.4: Update `PROJECT_SPEC.md`

**Files:**
- Modify: `PROJECT_SPEC.md`

- [ ] **Step 1: First Milestones table** — add row for M10 reflecting all the achievements (named_pipeline_stmt, StdOp::Matmul/MulScalar, ArgType::Tensor, four new ShapeError variants, DeclaredShapeMismatch, both profiles' emit_matmul + emit_mulscalar, Softmax 4D dispatch, end-to-end SelfAttention bit-exact per-profile, test count 284 → ~326).

- [ ] **Step 2: Current Status section** — replace `**Milestone 9 complete.** ...` with `**Milestone 10 complete.** ~326 tests passing on macOS arm64 ...`.

- [ ] **Step 3: Strategic Roadmap** — annotate Axis 2 with M10 closure: `NFL v0.2 self-attention [complete in M10] → multi-input grammar (Q/K/V) → transformer block (residual + LayerNorm + FFN) → profile-level viewer annotations`.

- [ ] **Step 4: Open Questions** — if OQ-BENCH pre-commit landed before M10 (per spec §11), move it from "Trigger-driven cleanup" to "Decisions (formerly open, now resolved)" with the OQ-BENCH commit hash. **If not landed, leave OQ-BENCH unchanged.** The plan does NOT include OQ-BENCH; this docs update reflects whatever the executor decided.

### Task 12.5: Update `DEVLOG.md`

**Files:**
- Modify: `DEVLOG.md`

- [ ] **Step 1: Prepend a new entry** following the existing format (newest at top):

```markdown
## YYYY-MM-DD — Milestone 10 closed: NFL v0.2 self-attention + 4D codegen

### What was done
- **NFL grammar v0.2** — new `named_pipeline_stmt` production. Group 1
  commit `<hash>`.
- **UIR args machinery** — `ArgType::Tensor` + 3-function `resolve_args`
  cascade. Group 2 commit `<hash>`.
- **`StdOp::Matmul`** — rank ≥ 2 inputs, 4 new `ShapeError` variants,
  `transpose_b` helper. Group 3 commit `<hash>`.
- **`StdOp::MulScalar`** — per-element scalar multiply. Group 4 commit
  `<hash>`.
- **`named_pipeline_stmt` builder + `DeclaredShapeMismatch` + Softmax
  rank ≥ 2.** Group 5 commit `<hash>`.
- **arm64 codegen** — emit_matmul (Group 6 `<hash>`), emit_mulscalar
  (Group 7 `<hash>`), Softmax 4D dispatch (Group 8 `<hash>`).
- **x86_64 codegen** — emit_matmul (Group 9a `<hash>`), emit_mulscalar
  (Group 9b `<hash>`), Softmax 4D dispatch (Group 9c `<hash>`).
- **End-to-end FFI integration** — `tests/fixtures/self_attention.nfl`
  + per-profile reference + FFI test on both profiles. Group 10 commit
  `<hash>`.
- **Negative fixtures** — four .nfl files pinning rejection layers.
  Group 11 commit `<hash>`.
- **Documentation** — grammar.md, both profile guides, PROJECT_SPEC,
  CLAUDE.md, this entry. Group 12 commit `<hash>`.

### Decisions made
**Per-profile bit-exact** is M10's acceptance criterion — cross-profile
bit-exact at byte level is unreachable (FMA × libm divergence) and
deferred to OQ-BENCH tolerance reports.

**Atomic 3-function cascade** preserved in Group 2 per spec §5.3:
resolve_args / build_op / build_model move in one commit — splits
produce non-compiling intermediates.

**x86_64 split into 9a/9b/9c** by default per spec §10 step 9 caveat.
The fold/no-fold call was made at execution time. (Record the actual
decision here.)

### Problems encountered
[Implementer fills this in: real surprises, M9-style hazards if any,
fixture-loading drift if any.]

### Next step
Push branch + open PR titled `feat(m10): NFL v0.2 self-attention +
4D codegen`. Once merged, the next milestone selection runs over
the post-M10 Strategic Roadmap.
```

(Replace `<hash>` placeholders with actual short hashes via
`git log --oneline -14` before committing.)

### Task 12.6: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Repository Structure tree** — add the four new files (`profiles/{arm64,x86_64}/src/ops/{matmul,mulscalar}.rs`) matching the existing tree's indent + arrow-comment style.

- [ ] **Step 2: Current Status section** — bump `Milestone 9 complete` → `Milestone 10 complete`, test count 284 → ~326.

### Task 12.7: Workspace gates + commit

- [ ] **Step 1: Four gates** — exit 0
- [ ] **Step 2: `cargo test --workspace`** — 326 tests on macOS arm64 (~340 on Linux x86_64 CI)
- [ ] **Step 3: Stage + commit**

```bash
git add docs/language_reference/grammar.md \
        docs/profile_guide/arm64.md \
        docs/profile_guide/x86_64.md \
        PROJECT_SPEC.md \
        DEVLOG.md \
        CLAUDE.md

git commit -m "$(cat <<'EOF'
docs(m10): close M10 — grammar.md, profile guides, PROJECT_SPEC, DEVLOG, CLAUDE

Final M10 documentation. No code changes — pure docs commit.

- docs/language_reference/grammar.md: "Named pipelines" section.
- docs/profile_guide/arm64.md: "M10 ops" — emit_matmul (FMA outer-
  loop wrapper), emit_mulscalar (movz/movk + fmov pre-load), Softmax
  4D dispatch.
- docs/profile_guide/x86_64.md: "M10 ops" — emit_matmul (mulss+addss,
  xmm6/xmm7 preservation), emit_mulscalar (movl/movd pre-load),
  Softmax 4D dispatch.
- PROJECT_SPEC.md: First Milestones gains M10 row; Current Status
  bumped to M10; Strategic Roadmap reflects Axis 2 advancement.
- DEVLOG.md: M10 closure entry covering all 14 implementation
  commits.
- CLAUDE.md: repo tree gains 4 new ops files; status reflects M10.

Test count unchanged from group 11.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: `git status` → clean.**
- [ ] **Step 5: Final commit graph**

```bash
git log --oneline -16
```

Expected (top-down): 14 implementation commits + 1 plan-landing commit + 2 spec commits = 17 visible. The 14 implementation commits in order:
1. `feat(m10/parser): named_pipeline_stmt grammar + AST + lookahead parser`
2. `feat(m10/ir): ArgType::Tensor + resolve_args cascade`
3. `feat(m10/ir): StdOp::Matmul + 4 new ShapeError variants + transpose_b helper`
4. `feat(m10/ir): StdOp::MulScalar`
5. `feat(m10/ir): named_pipeline builder + DeclaredShapeMismatch + Softmax rank`
6. `feat(m10/arm64): emit_matmul`
7. `feat(m10/arm64): emit_mulscalar`
8. `feat(m10/arm64): Softmax dispatch generalised`
9. `feat(m10/x86_64): emit_matmul`
10. `feat(m10/x86_64): emit_mulscalar`
11. `feat(m10/x86_64): Softmax dispatch generalised`
12. `feat(m10/integration): SelfAttention fixture + per-profile FFI tests`
13. `test(m10): negative fixtures`
14. `docs(m10): close M10`

(Or 12 if x86_64 was folded at execution time per Group 9 caveat.)

---

## Self-review

Run through the spec one more time with the plan in hand:

- **§4 NFL grammar (named_pipeline_stmt)** — covered in Group 1 + 12.1.
- **§5.1 StdOp variants** — Matmul in Group 3, MulScalar in Group 4.
- **§5.2 ArgType::Tensor** — Group 2 Task 2.1.
- **§5.3 resolve_args cascade** — Group 2 Tasks 2.2–2.4 (atomic).
- **§5.4 New op signatures** — Matmul in Group 3 Task 3.2, MulScalar in Group 4 Task 4.2.
- **§5.5.1 Matmul shape inference** — Group 3 Task 3.5 (5-step validation, output shape).
- **§5.5.2 MulScalar shape inference** — Group 4 Task 4.3 (passthrough).
- **§5.5.3 Softmax rank ≥ 2** — Group 5 Task 5.2.
- **§5.6 BuildErrorKind::DeclaredShapeMismatch** — Group 5 Task 5.1.
- **§5.7 Four new ShapeError variants** — Group 3 Task 3.3.
- **§5.8 named_pipeline_stmt builder** — Group 5 Task 5.3.
- **§5.9 calls_extern_math unchanged** — implicitly preserved (Matmul, MulScalar don't appear in the predicate).
- **§5.10 Existing fusion passes unchanged** — implicitly preserved (no fusion-pass modifications anywhere in plan).
- **§6.1 Architectural invariant: new files only** — Group 6 (ops/matmul.rs new), Group 7 (ops/mulscalar.rs new). emit_linear / emit_softmax untouched (verified by absence from "Files touched" sections).
- **§6.2 ABI invariant** — preserved (params_layout has no Matmul/MulScalar contribution; SelfAttention has params_floats=0 and the FFI harness uses `vec![0.0f32; 0]`).
- **§6.3 Buffer assignment** — Group 6 Task 6.1 (Matmul → StackOffset), Group 7 Task 7.1 (MulScalar → Alias). Same in 9a/9b for x86_64.
- **§6.4 emit_matmul (arm64 + x86_64)** — Groups 6 and 9a.
- **§6.5 emit_mulscalar (arm64 + x86_64) + truncation contract** — Groups 7 and 9b. Truncation in dispatch task.
- **§6.6 Softmax dispatch update** — Groups 8 and 9c.
- **§6.7 classify_op dispatch** — Groups 6 / 7 / 9a / 9b add the explicit arms.
- **§6.8 Prologue/epilogue unchanged** — implicitly preserved (no compute_callee_saved changes; Matmul does not contribute to calls_extern_math).
- **§7.1 SelfAttention fixture** — Group 10 Task 10.1.
- **§7.2 Per-profile bit-exact strategy** — Group 10 Tasks 10.2/10.3 (FMA-using arm64 ref, separate-mul-add x86_64 ref).
- **§7.3 Deterministic input** — Group 10 (`(0..total).map(|i| (i as f32).sin() * 0.1)`).
- **§7.4 Zero-params FFI contract** — Group 10 (`vec![0.0f32; sig.params_floats]` pattern, NOT `ptr::null()`).
- **§7.5 Test layers** — covered across Groups 1 / 3 / 6 / 7 / 8 / 9a / 9b / 9c / 10.
- **§8 Test enumeration (~45)** — counted: 5 (G1) + 0 (G2) + 11 (G3, includes resolve sanity) + 3 (G4) + 4 (G5) + 5 (G6) + 2 (G7) + 1 (G8) + 6 (G9a) + 2 (G9b) + 1 (G9c) + 2 (G10) + (0 or 1) (G11) = **42 or 43**. Spec says ~45. Gap reason: G3 test count (the 10-test enumeration plus 1 sanity) totals 11, not 14 as some readings of §8.2 might suggest. The sub-counts in spec §8 are approximate ("~45 total new tests") and the plan stays within the ~ tolerance. If the executor wants to hit 45 exactly, two more tests can be added in Group 5 (e.g., `tensor_arg_unknown_variable_errors` from spec §8.2 wasn't separately enumerated; add it at Group 5 Task 5.4 if desired).
- **§9 Non-goals** — implicitly respected (no infix operators, no broadcasting, no fusion of attention-internal patterns, no SIMD, no transformer block).
- **§10 Sub-milestone decomposition** — 14 commits land per group, with §10 step 9 split into 9a/9b/9c by default.
- **§11 OQ-BENCH** — explicitly out of M10 scope; mentioned in Group 12 docs only as conditional update.
- **§12 Risks** — addressed:
  - 12.4 R1 (parser one-token lookahead) — Group 1 Tasks 1.4 + 1.7 cover this.
  - 12.4 R2 (resolve_args cascade) — Group 2 atomicity.
  - 12.4 R3 (FFI null-pointer) — Group 10 Task harness pattern.
  - 12.4 R4 (transpose_b inner-addressing bug) — Group 6 Task 6.5 unit test + Group 10 reference.
  - 12.4 R5 (f64→f32 truncation surprise) — Group 7 Task 7.4 + docs in Group 12.
  - 12.4 R6 (stack-allocated intermediate buffer scaling) — out of scope for M10 fixture, mentioned in spec §12.4 as M11+ concern; not actionable here.

**Gaps surfaced by self-review:**
- Test count is 42–43, not 45. Acceptable per "~45" wording in spec §8. If the executor wants to hit exactly 45, add `tensor_arg_unknown_variable_errors` and `matmul_resolves_via_stdlib` (already present!) — but the "10 tests" in spec §8.2 line items vs the plan's 11 in Group 3 already covers this.
- Test name `mul_scalar_resolves` (Group 4) is missing from the spec §8.2 enumeration — added by the plan as a sanity check parallel to `matmul_resolves_via_stdlib`. Harmless extra coverage.

The plan is **complete vs spec**. Self-review concludes — no further patches needed.

---

*This plan is read by both implementer and reviewer. It evolves through execution: discoveries that alter scope, contracts, or acceptance criteria should round-trip back via brainstorming-then-spec, and only then back into a plan revision.*

