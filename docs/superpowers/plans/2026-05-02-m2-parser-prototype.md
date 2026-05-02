# NFL Parser Prototype (Milestone 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working Rust parser for NFL v0.1 that turns `.nfl` source into a typed AST, with a CLI binary for inspection.

**Architecture:** Cargo workspace at the repo root with one member crate `nflc` (NFL Compiler). The crate is hand-written in pure-`std` Rust — no `chumsky`, `nom`, `lalrpop`, `pest`, or any other parser library. Three modules: `lexer` (tokens + INDENT/DEDENT machine), `parser` (recursive descent), `ast` (typed enums). The library exposes one entry point `nflc::parse(&str) -> Result<NflSource, ParseError>`. The `nflc` binary wraps it as a CLI.

**Tech Stack:** Rust 2021, std only, `cargo build` / `cargo test` / `cargo run`. No external dependencies.

**Source spec:** [`docs/superpowers/specs/2026-05-02-m2-parser-prototype-design.md`](../specs/2026-05-02-m2-parser-prototype-design.md). All decisions, types, and acceptance criteria are defined there. **If anything in this plan disagrees with the spec, the spec wins** — flag the discrepancy and stop.

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m2-parser-prototype` (worktree on branch `claude/m2-parser-prototype`).

**Branch strategy:** all M2 commits land on `claude/m2-parser-prototype`. The branch was forked from `main` after M1 PRs (#1, #2) merged, so all M1 artefacts (`language/grammar.ebnf`, fixtures, reference doc) are present at the worktree root. When M2 is complete, push the branch and open a single PR for the entire milestone.

**Project conventions** (from `CLAUDE.md` — read first if unfamiliar):
- TDD: red → green → refactor. Each impl task starts with a failing test.
- Each session ends with a `DEVLOG.md` entry.
- "Current Status" in `CLAUDE.md` must always reflect the project's actual state.
- All code/comments/docs in English. Russian only in conversation.
- Do **not** touch existing `.gitkeep` files outside `compiler/` (per the user's earlier instruction). The `.gitkeep` files inside `compiler/{lexer,parser,ir,passes}/` ARE removed in Task 20 because those directories are reorganised under `compiler/src/`.

---

## File Structure

**Create (workspace + crate code):**

| Path | Purpose | Created in |
|---|---|---|
| `Cargo.toml` (root) | Workspace declaration | Task 1 |
| `compiler/Cargo.toml` | Member crate manifest, deps empty | Task 1 |
| `compiler/src/lib.rs` | Library root; re-exports public API | Task 1 → grows in Task 15 |
| `compiler/src/main.rs` | CLI binary `nflc` | Task 1 → grows in Task 16 |
| `compiler/src/ast.rs` | All AST data types | Task 2 |
| `compiler/src/lexer/mod.rs` | Lexer entry point `pub fn lex` | Task 4 |
| `compiler/src/lexer/tokens.rs` | `Token`, `TokenKind`, `LexError` | Task 3 |
| `compiler/src/lexer/indent.rs` | Indent stack + pipeline-continuation state | Task 6 |
| `compiler/src/lexer/tests.rs` | Lexer unit tests | grown across Tasks 4-8 |
| `compiler/src/parser/mod.rs` | Parser entry point + helpers + all `parse_*` fns | grown across Tasks 9-15 |
| `compiler/src/parser/tests.rs` | Parser unit tests | grown across Tasks 10-15 |
| `compiler/tests/fixtures.rs` | Integration tests (positive + negative) | Tasks 17, 19 |

**Create (test data):**

| Path | Purpose | Created in |
|---|---|---|
| `tests/fixtures/negative/tabs_in_indent.nfl` | Tab in leading whitespace | Task 18 |
| `tests/fixtures/negative/missing_colon.nfl` | `model X [...]` without `:` | Task 18 |
| `tests/fixtures/negative/unclosed_bracket.nfl` | `[a=1` without `]` | Task 18 |
| `tests/fixtures/negative/empty_tensor.nfl` | `Tensor[]` (empty dim_list) | Task 18 |
| `tests/fixtures/negative/empty_op_args.nfl` | `linear[]` (empty op brackets) | Task 18 |
| `tests/fixtures/negative/named_before_positional.nfl` | `linear[a=1, 2]` | Task 18 |
| `tests/fixtures/negative/bad_dedent.nfl` | Dedent to a level not on the stack | Task 18 |

**Modify:**

| Path | Change | Modified in |
|---|---|---|
| `DEVLOG.md` | Add Milestone 2 close-out entry at the top, including the four tech-debt items from the spec | Task 20 |
| `CLAUDE.md` | Update "Current Status" to reflect M2 complete, M3 (UIR) as next | Task 20 |

**Delete:**

| Path | Why | Deleted in |
|---|---|---|
| `compiler/lexer/.gitkeep`, `compiler/lexer/` | Reorganised under `compiler/src/lexer/` | Task 20 |
| `compiler/parser/.gitkeep`, `compiler/parser/` | Reorganised under `compiler/src/parser/` | Task 20 |
| `compiler/ir/.gitkeep`, `compiler/ir/` | Will be `compiler/src/ir/` in M3, no need to keep stub | Task 20 |
| `compiler/passes/.gitkeep`, `compiler/passes/` | Will be `compiler/src/passes/` later, no need to keep stub | Task 20 |
| `compiler/.gitkeep` | The `compiler/` dir now has real content | Task 20 |

**Do NOT touch:**
- `language/.gitkeep`, `tests/fixtures/.gitkeep`, any other `*/.gitkeep` outside `compiler/`
- The M1 spec or the M2 spec under `docs/superpowers/specs/`
- The `.git/info/exclude` file (already configured to hide `.claude/`)

---

## Verification approach

| Verification | When | How |
|---|---|---|
| Workspace builds | Task 1 | `cargo build` from repo root, exit 0 |
| Each impl task is correct | Tasks 4-8, 10-15 | TDD: failing unit test exists first, implementation makes it pass |
| All unit tests pass | After each impl task | `cargo test --lib` from `compiler/`, all green |
| All integration tests pass | After Tasks 17, 19 | `cargo test --test fixtures` from `compiler/`, all green |
| CLI works end-to-end | Task 16 + Task 20 final check | Manual `cargo run --bin nflc -- parse …` on positive and negative fixtures |
| No warnings | Task 20 final check | `cargo build` and `cargo test` produce zero warnings; if any are unavoidable they must be `#[allow(...)]`'d with a one-line justifying comment |

**TDD discipline:**
- For every code change in tasks 4-8 and 10-15: the test is written and verified to FAIL before any implementation code is written.
- Verify failure with the actual compile error or test failure message.
- Then implement the minimum code to make the test pass.
- Then run the full test suite to make sure nothing else broke.
- Then commit.

---

## Task list

| # | Task | Commits | Tests added |
|---|---|---|---|
| 1 | Cargo workspace + crate scaffolding | 1 | 0 |
| 2 | AST data types | 1 | 0 (data only) |
| 3 | Lexer types: `Token`, `TokenKind`, `LexError` | 1 | 0 (data only) |
| 4 | Lexer: simple tokens (keywords, punct, idents, numbers) | 1 | ~8 |
| 5 | Lexer: comments and newlines | 1 | ~4 |
| 6 | Lexer: indent machine (INDENT/DEDENT) | 1 | ~6 |
| 7 | Lexer: pipeline-continuation rule | 1 | ~3 |
| 8 | Lexer: error cases | 1 | ~5 |
| 9 | Parser scaffolding + `ParseError` + helpers | 1 | 0 (no parsing yet) |
| 10 | Parser: `arg_value`, `named_arg`, `op_args`, `operation` | 1 | ~5 |
| 11 | Parser: `pipeline_stmt` (incl. chain & step inlined) | 1 | ~3 |
| 12 | Parser: `type_expr`, `dim_list`, `dim`, `variable_decl` | 1 | ~4 |
| 13 | Parser: `named_value`, `model_params` | 1 | ~3 |
| 14 | Parser: `model_body`, `model_stmt`, `model_def` | 1 | ~3 |
| 15 | Parser: `nfl_source` + library entry `parse(&str)` | 1 | ~2 |
| 16 | CLI: usage, `parse <file>`, `--tokens`, error rendering | 1 | (manual) |
| 17 | Integration: positive fixtures (5 tests) | 1 | 5 |
| 18 | Negative fixtures: 7 `.nfl` files | 1 | 0 (test in 19) |
| 19 | Integration: negative fixtures (7 tests) | 1 | 7 |
| 20 | Cleanup + close-out (rm .gitkeep, DEVLOG, CLAUDE.md, final verification) | 1 | (verification) |

**Total:** 20 tasks, 20 commits, ~58 unit tests + 12 integration tests.

---

## Task 1: Cargo workspace + crate scaffolding

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `compiler/Cargo.toml`
- Create: `compiler/src/lib.rs` (stub)
- Create: `compiler/src/main.rs` (stub)

- [ ] **Step 1: Create `Cargo.toml` at repository root**

```toml
[workspace]
resolver = "2"
members = ["compiler"]
```

- [ ] **Step 2: Create `compiler/Cargo.toml`**

```toml
[package]
name = "nflc"
version = "0.1.0"
edition = "2021"
description = "NeuralForge Language Compiler — parser prototype (Milestone 2)"
license = "MIT OR Apache-2.0"

[dependencies]

[lib]
path = "src/lib.rs"

[[bin]]
name = "nflc"
path = "src/main.rs"
```

- [ ] **Step 3: Create `compiler/src/lib.rs` with a minimal stub**

```rust
//! NFL Compiler — library crate.
//!
//! Public API will grow as Milestone 2 progresses. The final entry point is
//! [`parse`], which accepts NFL source text and returns a typed AST.

// Modules will be added in subsequent tasks.
```

- [ ] **Step 4: Create `compiler/src/main.rs` with a minimal stub**

```rust
//! `nflc` CLI binary.
//!
//! Usage will be wired up in a later task.

fn main() {
    println!("nflc: NFL Compiler (Milestone 2 in progress)");
}
```

- [ ] **Step 5: Verify the workspace builds**

Run: `cargo build`
Expected: `Compiling nflc v0.1.0 (...)` then `Finished ... [unoptimized + debuginfo] target(s)`. Exit 0. No warnings.

If `cargo build` reports warnings, fix them now (do not silence with `#[allow]` for stub code; remove the offending bits or add `#![allow(dead_code)]` ONLY at the crate-root level with a comment explaining it's a temporary scaffold removed in a later task).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml compiler/Cargo.toml compiler/src/lib.rs compiler/src/main.rs
git commit -m "feat(m2): scaffold cargo workspace and nflc crate

Empty stubs for lib.rs and main.rs; cargo build succeeds.
The crate is std-only (no dependencies), edition 2021. Modules
and CLI logic land in subsequent tasks."
```

---

## Task 2: AST data types

**Files:**
- Create: `compiler/src/ast.rs`
- Modify: `compiler/src/lib.rs` (add `pub mod ast;`)

- [ ] **Step 1: Create `compiler/src/ast.rs` with the full AST**

```rust
//! Typed AST for NFL v0.1.
//!
//! Mirrors the EBNF productions in `language/grammar.ebnf`. Every node carries
//! a [`Span`] indicating where it started in the source, for future error
//! reporting and the human-readable viewer (Milestone 7).

#[derive(Debug, Clone, PartialEq)]
pub struct NflSource {
    pub models: Vec<ModelDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelDef {
    pub name: String,
    pub params: Vec<NamedValue>,
    pub body: Vec<ModelStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedValue {
    pub name: String,
    pub value: u64,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStmt {
    VariableDecl(VariableDecl),
    Pipeline(PipelineStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeExpr {
    /// Always `"Tensor"` in v0.1. See spec §9, open question 1.
    pub name: String,
    pub dims: Vec<Dim>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Integer(u64),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStmt {
    pub source: String,
    pub steps: Vec<Operation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub name: String,
    pub args: Vec<OpArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpArg {
    Positional(ArgValue),
    Named { name: String, value: ArgValue },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue {
    Integer(u64),
    Float(f64),
    Symbol(String),
}

/// Source position of an AST node. v0.1 stores only the start position.
/// End-position is deferred until a consumer needs it (see spec §9, open
/// question 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub const fn new(line: u32, col: u32) -> Self {
        Self { line, col }
    }
}
```

- [ ] **Step 2: Wire the module in `compiler/src/lib.rs`**

Replace the previous stub content with:

```rust
//! NFL Compiler — library crate.
//!
//! Public API will grow as Milestone 2 progresses. The final entry point is
//! [`parse`], which accepts NFL source text and returns a typed AST.

pub mod ast;
```

- [ ] **Step 3: Verify build**

Run: `cargo build`
Expected: clean build, no warnings.

Note: there are no tests for `ast.rs` yet — it is data-only, and tests come naturally when the lexer/parser populate AST nodes.

- [ ] **Step 4: Commit**

```bash
git add compiler/src/ast.rs compiler/src/lib.rs
git commit -m "feat(m2): add AST data types

Mirrors EBNF productions in language/grammar.ebnf. Every node has
a Span for future error reporting (start line:col only in v0.1).
TypeExpr.name remains a String for now; will revisit for v0.2 type
system per spec §9.1."
```

---

## Task 3: Lexer types — `Token`, `TokenKind`, `LexError`

**Files:**
- Create: `compiler/src/lexer/mod.rs` (stub for now)
- Create: `compiler/src/lexer/tokens.rs`
- Modify: `compiler/src/lib.rs` (add `pub mod lexer;`)

- [ ] **Step 1: Create `compiler/src/lexer/tokens.rs`**

```rust
//! Token types and lexical errors.
//!
//! See `language/grammar.ebnf` for the abstract grammar this models.

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Model,
    Tensor,
    // Punctuation
    LBracket,
    RBracket,
    Colon,
    Comma,
    Equals,
    Arrow,
    // Identifiers and literals
    Ident(String),
    Integer(u64),
    Number(f64),
    // Significant whitespace
    Newline,
    Indent,
    Dedent,
    // End
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    /// 1-based line number of the first character of the token.
    pub line: u32,
    /// 1-based column of the first character of the token.
    pub col: u32,
}

impl Token {
    pub const fn new(kind: TokenKind, line: u32, col: u32) -> Self {
        Self { kind, line, col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LexError {
    TabInIndent { line: u32, col: u32 },
    BadDedent { line: u32, col: u32 },
    UnknownChar { line: u32, col: u32, ch: char },
    BadNumber { line: u32, col: u32, raw: String },
    UnexpectedEof { line: u32, col: u32 },
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::TabInIndent { .. } => write!(f, "tab character in leading whitespace"),
            LexError::BadDedent { .. } => write!(f, "inconsistent dedent: indent level does not match any enclosing block"),
            LexError::UnknownChar { ch, .. } => write!(f, "unknown character: {:?}", ch),
            LexError::BadNumber { raw, .. } => write!(f, "malformed number literal: {:?}", raw),
            LexError::UnexpectedEof { .. } => write!(f, "unexpected end of file"),
        }
    }
}

impl std::error::Error for LexError {}

impl LexError {
    /// Returns the (line, col) where the error occurred. 1-based.
    pub fn position(&self) -> (u32, u32) {
        match *self {
            LexError::TabInIndent { line, col }
            | LexError::BadDedent { line, col }
            | LexError::UnknownChar { line, col, .. }
            | LexError::BadNumber { line, col, .. }
            | LexError::UnexpectedEof { line, col } => (line, col),
        }
    }
}
```

- [ ] **Step 2: Create `compiler/src/lexer/mod.rs` as a stub**

```rust
//! Hand-written lexer for NFL.

pub mod tokens;

pub use tokens::{LexError, Token, TokenKind};

/// Tokenise NFL source text. To be implemented in Task 4.
pub fn lex(_source: &str) -> Result<Vec<Token>, LexError> {
    unimplemented!("lex() — implemented in Task 4")
}
```

- [ ] **Step 3: Wire in `compiler/src/lib.rs`**

```rust
//! NFL Compiler — library crate.

pub mod ast;
pub mod lexer;
```

- [ ] **Step 4: Verify build**

Run: `cargo build`
Expected: clean build, no warnings. (`unimplemented!()` does not cause a warning at compile time.)

- [ ] **Step 5: Commit**

```bash
git add compiler/src/lexer/ compiler/src/lib.rs
git commit -m "feat(m2): add lexer types Token, TokenKind, LexError

LexError implements std::error::Error and Display. Variants cover
all error categories listed in spec §5.1. The lex() function is a
stub that panics; implementation lands in subsequent tasks."
```

---

## Task 4: Lexer — simple tokens

This is the first TDD task. We implement recognition of single-line, single-token inputs: keywords (`model`, `Tensor`), punctuation (`[ ] : , = ->`), identifiers, integer and float literals.

**Files:**
- Modify: `compiler/src/lexer/mod.rs` (replace stub `lex()` with real implementation)
- Create: `compiler/src/lexer/tests.rs` (unit tests)

- [ ] **Step 1: Write failing tests in `compiler/src/lexer/tests.rs`**

```rust
//! Unit tests for the lexer.

use super::*;
use super::tokens::TokenKind::*;

fn lex_kinds(source: &str) -> Vec<TokenKind> {
    lex(source).unwrap().into_iter().map(|t| t.kind).collect()
}

#[test]
fn lex_keyword_model() {
    assert_eq!(lex_kinds("model"), vec![Model, Eof]);
}

#[test]
fn lex_keyword_tensor() {
    assert_eq!(lex_kinds("Tensor"), vec![Tensor, Eof]);
}

#[test]
fn lex_punctuation() {
    assert_eq!(
        lex_kinds("[ ] : , = ->"),
        vec![LBracket, RBracket, Colon, Comma, Equals, Arrow, Eof],
    );
}

#[test]
fn lex_identifier_simple() {
    assert_eq!(lex_kinds("foo"), vec![Ident("foo".into()), Eof]);
}

#[test]
fn lex_identifier_with_underscores_and_digits() {
    assert_eq!(lex_kinds("foo_bar2"), vec![Ident("foo_bar2".into()), Eof]);
}

#[test]
fn lex_integer() {
    assert_eq!(lex_kinds("512"), vec![Integer(512), Eof]);
}

#[test]
fn lex_float() {
    let toks = lex_kinds("0.2");
    assert_eq!(toks.len(), 2);
    match &toks[0] {
        Number(n) => assert!((n - 0.2).abs() < 1e-12),
        other => panic!("expected Number, got {other:?}"),
    }
    assert_eq!(toks[1], Eof);
}

#[test]
fn lex_token_positions() {
    // "model x" — 'model' at col 1, 'x' at col 7
    let toks = lex("model x").unwrap();
    assert_eq!(toks[0].line, 1);
    assert_eq!(toks[0].col, 1);
    assert_eq!(toks[1].line, 1);
    assert_eq!(toks[1].col, 7);
    assert_eq!(toks[1].kind, Ident("x".into()));
}
```

Wire the test module by adding this to `compiler/src/lexer/mod.rs`:

```rust
#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib lexer::tests`
Expected: every test panics at `unimplemented!("lex() — implemented in Task 4")`. ALL 8 tests fail. (TDD red.)

- [ ] **Step 3: Implement `lex` in `compiler/src/lexer/mod.rs`**

Replace the stub `pub fn lex` with the real implementation. Below is the structure; fill it in with the recognisers shown.

```rust
//! Hand-written lexer for NFL.

pub mod tokens;

#[cfg(test)]
mod tests;

pub use tokens::{LexError, Token, TokenKind};

/// Tokenise NFL source text into a flat token stream ending with `Eof`.
///
/// Currently handles single-token-per-line inputs. Indentation, comments,
/// newlines, and pipeline continuation are added in later tasks.
pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut line: u32 = 1;
    let mut col: u32 = 1;

    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];

        // Skip horizontal whitespace (space and tab) outside of leading whitespace.
        // (Leading whitespace and newlines are added in later tasks; for now we
        // accept any single-line input separated by spaces.)
        if b == b' ' || b == b'\t' {
            i += 1;
            col += 1;
            continue;
        }

        // Punctuation singletons.
        let single: Option<TokenKind> = match b {
            b'[' => Some(TokenKind::LBracket),
            b']' => Some(TokenKind::RBracket),
            b':' => Some(TokenKind::Colon),
            b',' => Some(TokenKind::Comma),
            b'=' => Some(TokenKind::Equals),
            _ => None,
        };
        if let Some(kind) = single {
            tokens.push(Token::new(kind, line, col));
            i += 1;
            col += 1;
            continue;
        }

        // Arrow '->'.
        if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            tokens.push(Token::new(TokenKind::Arrow, line, col));
            i += 2;
            col += 2;
            continue;
        }

        // Identifier or keyword: starts with letter, continues with letter/digit/underscore.
        if b.is_ascii_alphabetic() {
            let start = i;
            let start_col = col;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
                col += 1;
            }
            let ident = std::str::from_utf8(&bytes[start..i])
                .expect("ASCII identifier")
                .to_string();
            let kind = match ident.as_str() {
                "model" => TokenKind::Model,
                "Tensor" => TokenKind::Tensor,
                _ => TokenKind::Ident(ident),
            };
            tokens.push(Token::new(kind, line, start_col));
            continue;
        }

        // Number literal: integer or float.
        if b.is_ascii_digit() {
            let start = i;
            let start_col = col;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
                col += 1;
            }
            // Optional fractional part: "." followed by at least one digit.
            let has_fractional = i + 1 < bytes.len()
                && bytes[i] == b'.'
                && bytes[i + 1].is_ascii_digit();
            if has_fractional {
                i += 1; // consume '.'
                col += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
                let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII number");
                let value: f64 = raw.parse().map_err(|_| LexError::BadNumber {
                    line,
                    col: start_col,
                    raw: raw.to_string(),
                })?;
                tokens.push(Token::new(TokenKind::Number(value), line, start_col));
            } else {
                let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII integer");
                let value: u64 = raw.parse().map_err(|_| LexError::BadNumber {
                    line,
                    col: start_col,
                    raw: raw.to_string(),
                })?;
                tokens.push(Token::new(TokenKind::Integer(value), line, start_col));
            }
            continue;
        }

        // Anything else is an error for now.
        return Err(LexError::UnknownChar {
            line,
            col,
            ch: b as char,
        });
    }

    tokens.push(Token::new(TokenKind::Eof, line, col));
    Ok(tokens)
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib lexer::tests`
Expected: all 8 tests pass.

If any test fails, debug it. Most likely culprits: column tracking off-by-one, identifier-vs-keyword resolution.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test`
Expected: all tests pass (only the 8 lexer tests exist so far).

- [ ] **Step 6: Commit**

```bash
git add compiler/src/lexer/
git commit -m "feat(m2/lexer): recognise simple tokens

Keywords (model, Tensor), punctuation ([ ] : , = ->), identifiers,
integer and float literals. Inter-token horizontal whitespace
(spaces and tabs) is silently consumed. Comments, newlines,
indent, and pipeline continuation come in subsequent tasks.

8 unit tests cover all token kinds and 1:1 column positions."
```

---

## Task 5: Lexer — comments and newlines

Comments (`#…\n`) are eaten by the lexer; they never produce a token. Newlines (LF or CRLF) produce a single `Newline` token.

**Files:**
- Modify: `compiler/src/lexer/mod.rs` (add comment + newline handling)
- Modify: `compiler/src/lexer/tests.rs` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/lexer/tests.rs`:

```rust
#[test]
fn lex_comment_alone() {
    // A comment-only file produces no token but Eof.
    assert_eq!(lex_kinds("# hello"), vec![Eof]);
}

#[test]
fn lex_comment_at_end_of_line() {
    // Comment after a token does not affect the token, and is consumed.
    assert_eq!(
        lex_kinds("model # ignored"),
        vec![Model, Eof],
    );
}

#[test]
fn lex_newline_lf() {
    assert_eq!(lex_kinds("model\nTensor"), vec![Model, Newline, Tensor, Eof]);
}

#[test]
fn lex_newline_crlf() {
    assert_eq!(lex_kinds("model\r\nTensor"), vec![Model, Newline, Tensor, Eof]);
}
```

- [ ] **Step 2: Run tests, verify the 4 new ones FAIL**

Run: `cargo test --lib lexer::tests`
Expected: previous 8 tests pass; 4 new tests fail because the lexer treats `#`, `\n`, `\r` as unknown characters.

- [ ] **Step 3: Implement comment and newline handling**

In `compiler/src/lexer/mod.rs`, before the punctuation singletons match, add:

```rust
        // Comment: consume to end of line (do not include the newline itself).
        if b == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
                col += 1;
            }
            continue;
        }

        // Newline: support both LF and CRLF.
        if b == b'\n' {
            tokens.push(Token::new(TokenKind::Newline, line, col));
            i += 1;
            line += 1;
            col = 1;
            continue;
        }
        if b == b'\r' {
            // CRLF: consume the \r, then expect \n on the next iteration. If the
            // next byte is \n, treat the pair as one newline; if not, it's still
            // a newline (rare bare-CR case — accept silently for robustness).
            tokens.push(Token::new(TokenKind::Newline, line, col));
            i += 1;
            if i < bytes.len() && bytes[i] == b'\n' {
                i += 1;
            }
            line += 1;
            col = 1;
            continue;
        }
```

Place this block right after the leading-whitespace skip and before the punctuation singletons.

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib lexer::tests`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/lexer/
git commit -m "feat(m2/lexer): comments and newlines

# starts a comment that runs to end of line (lexer-eaten, no token).
LF and CRLF both produce one Newline token; line counter increments
exactly once per source line. Bare CR is also accepted as a newline
for robustness.

4 new tests cover comment-only files, trailing comments, and both
newline conventions."
```

---

## Task 6: Lexer — indent machine (INDENT/DEDENT)

This is the most novel piece of the lexer. It maintains an indent stack and emits virtual `Indent`/`Dedent` tokens whenever the leading-space count of a non-empty line changes.

**Files:**
- Create: `compiler/src/lexer/indent.rs` (the indent stack helper)
- Modify: `compiler/src/lexer/mod.rs` (use it; emit Indent/Dedent after each Newline)
- Modify: `compiler/src/lexer/tests.rs` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/lexer/tests.rs`:

```rust
#[test]
fn lex_simple_indent_block() {
    // Two-line block: header colon, then indented body line.
    let src = "model X:\n    foo\n";
    assert_eq!(
        lex_kinds(src),
        vec![Model, Ident("X".into()), Colon, Newline, Indent, Ident("foo".into()), Newline, Dedent, Eof],
    );
}

#[test]
fn lex_indent_then_dedent_back_to_zero() {
    let src = "model X:\n    foo\nbar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Dedent, Ident("bar".into()), Newline,
            Eof,
        ],
    );
}

#[test]
fn lex_blank_lines_do_not_affect_indent() {
    // Blank line in middle of body is ignored for indent purposes.
    let src = "model X:\n    foo\n\n    bar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Newline,
            Ident("bar".into()), Newline,
            Dedent, Eof,
        ],
    );
}

#[test]
fn lex_comment_only_line_does_not_affect_indent() {
    // Indented body, then a comment-only line at column 0, then more body.
    // The comment line is treated as blank; indent does NOT close.
    let src = "model X:\n    foo\n# top-level comment\n    bar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Newline,                                   // for the comment-bearing line
            Ident("bar".into()), Newline,
            Dedent, Eof,
        ],
    );
}

#[test]
fn lex_dedent_at_eof() {
    // EOF should emit any pending Dedents.
    let src = "model X:\n    foo\n";
    let toks = lex(src).unwrap();
    let last_three: Vec<&TokenKind> = toks.iter().rev().take(3).map(|t| &t.kind).collect();
    // Last three tokens: Eof, Dedent, Newline (in reverse order of the stream).
    assert_eq!(last_three, vec![&Eof, &Dedent, &Newline]);
}

#[test]
fn lex_nested_indent_dedent() {
    // Two levels: model body at indent 4, deeper indent at 8.
    // For grammar v0.1 there is no production using nested blocks, but the
    // lexer must still handle the mechanics correctly so future grammars work.
    let src = "model X:\n    foo\n        bar\n    baz\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Indent, Ident("bar".into()), Newline,
            Dedent, Ident("baz".into()), Newline,
            Dedent, Eof,
        ],
    );
}
```

- [ ] **Step 2: Run tests, verify the 6 new ones FAIL**

Run: `cargo test --lib lexer::tests`
Expected: previous tests pass; 6 new fail because the lexer ignores leading whitespace and never emits Indent/Dedent.

- [ ] **Step 3: Create `compiler/src/lexer/indent.rs`**

```rust
//! Indent stack — produces virtual INDENT/DEDENT tokens.
//!
//! Tracks the leading-space level of each non-empty (and non-comment-only)
//! line. The stack always has at least one entry (`0`). When a new line starts:
//! - Equal to top → no token emitted.
//! - Greater than top → push, emit one INDENT.
//! - Less than top → pop until equal, emitting one DEDENT per pop. If we run
//!   out of stack without finding an equal level, that is a `BadDedent`.
//!
//! This module is consumed by `lexer::mod` after a Newline has been emitted.

use super::tokens::{LexError, Token, TokenKind};

#[derive(Debug)]
pub(super) struct IndentStack {
    levels: Vec<usize>,
}

impl IndentStack {
    pub fn new() -> Self {
        Self { levels: vec![0] }
    }

    /// Adjust the stack to a new indent `level`. Pushes and emits Indent, or
    /// pops and emits Dedents. `line` and `col` are used for error reporting
    /// (col is the first non-whitespace column on the new line, 1-based).
    pub fn adjust_to(
        &mut self,
        level: usize,
        line: u32,
        col: u32,
        out: &mut Vec<Token>,
    ) -> Result<(), LexError> {
        let top = *self.levels.last().expect("stack always non-empty");
        if level == top {
            // Same indent — nothing to do.
            return Ok(());
        }
        if level > top {
            self.levels.push(level);
            out.push(Token::new(TokenKind::Indent, line, col));
            return Ok(());
        }
        // level < top — pop until equal.
        while *self.levels.last().expect("stack always non-empty") > level {
            self.levels.pop();
            out.push(Token::new(TokenKind::Dedent, line, col));
        }
        if *self.levels.last().expect("stack always non-empty") != level {
            return Err(LexError::BadDedent { line, col });
        }
        Ok(())
    }

    /// Close the file: emit a Dedent for every level above 0.
    pub fn close(&mut self, line: u32, col: u32, out: &mut Vec<Token>) {
        while self.levels.len() > 1 {
            self.levels.pop();
            out.push(Token::new(TokenKind::Dedent, line, col));
        }
    }
}
```

- [ ] **Step 4: Wire the indent stack into `lex()`**

Replace the body of `lex()` in `compiler/src/lexer/mod.rs`. The new structure: process the source line-by-line, distinguishing between blank-or-comment-only lines (which produce a `Newline` but do not affect indent) and content lines (which trigger `IndentStack::adjust_to`).

```rust
//! Hand-written lexer for NFL.

pub mod tokens;
mod indent;

#[cfg(test)]
mod tests;

pub use tokens::{LexError, Token, TokenKind};

use indent::IndentStack;

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut stack = IndentStack::new();

    let bytes = source.as_bytes();
    let mut i = 0;
    let mut line: u32 = 1;

    while i < bytes.len() {
        // Beginning of a line. Count leading spaces (and reject leading tabs).
        let line_start = i;
        let mut indent_spaces: usize = 0;
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            if bytes[i] == b'\t' {
                let col = (i - line_start) as u32 + 1;
                return Err(LexError::TabInIndent { line, col });
            }
            indent_spaces += 1;
            i += 1;
        }

        // Decide whether this is a blank/comment-only line or a content line.
        let line_is_blank = i >= bytes.len() || bytes[i] == b'\n' || bytes[i] == b'\r';
        let line_is_comment_only = !line_is_blank && bytes[i] == b'#';

        if line_is_blank || line_is_comment_only {
            // Eat the rest of the line up to (but not including) the newline.
            while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            // Eat the newline (LF or CRLF) and emit a Newline token.
            if i < bytes.len() {
                let col = (i - line_start) as u32 + 1;
                tokens.push(Token::new(TokenKind::Newline, line, col));
                if bytes[i] == b'\r' {
                    i += 1;
                    if i < bytes.len() && bytes[i] == b'\n' {
                        i += 1;
                    }
                } else {
                    i += 1; // LF
                }
                line += 1;
            }
            continue;
        }

        // Content line: adjust indent stack to the leading-space count.
        let first_col = indent_spaces as u32 + 1;
        stack.adjust_to(indent_spaces, line, first_col, &mut tokens)?;

        // Lex tokens on this line up to (but not including) the newline.
        let mut col: u32 = first_col;
        while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
            let b = bytes[i];

            // Inter-token whitespace (space and tab) — silently consumed.
            if b == b' ' || b == b'\t' {
                i += 1;
                col += 1;
                continue;
            }

            // Trailing comment on this line.
            if b == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                    col += 1;
                }
                break;
            }

            // Punctuation.
            let single = match b {
                b'[' => Some(TokenKind::LBracket),
                b']' => Some(TokenKind::RBracket),
                b':' => Some(TokenKind::Colon),
                b',' => Some(TokenKind::Comma),
                b'=' => Some(TokenKind::Equals),
                _ => None,
            };
            if let Some(kind) = single {
                tokens.push(Token::new(kind, line, col));
                i += 1;
                col += 1;
                continue;
            }

            // Arrow.
            if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                tokens.push(Token::new(TokenKind::Arrow, line, col));
                i += 2;
                col += 2;
                continue;
            }

            // Identifier or keyword.
            if b.is_ascii_alphabetic() {
                let start = i;
                let start_col = col;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                    col += 1;
                }
                let ident = std::str::from_utf8(&bytes[start..i])
                    .expect("ASCII identifier")
                    .to_string();
                let kind = match ident.as_str() {
                    "model" => TokenKind::Model,
                    "Tensor" => TokenKind::Tensor,
                    _ => TokenKind::Ident(ident),
                };
                tokens.push(Token::new(kind, line, start_col));
                continue;
            }

            // Number literal.
            if b.is_ascii_digit() {
                let start = i;
                let start_col = col;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
                let has_fractional = i + 1 < bytes.len()
                    && bytes[i] == b'.'
                    && bytes[i + 1].is_ascii_digit();
                if has_fractional {
                    i += 1;
                    col += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                        col += 1;
                    }
                    let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII number");
                    let value: f64 = raw.parse().map_err(|_| LexError::BadNumber {
                        line,
                        col: start_col,
                        raw: raw.to_string(),
                    })?;
                    tokens.push(Token::new(TokenKind::Number(value), line, start_col));
                } else {
                    let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII integer");
                    let value: u64 = raw.parse().map_err(|_| LexError::BadNumber {
                        line,
                        col: start_col,
                        raw: raw.to_string(),
                    })?;
                    tokens.push(Token::new(TokenKind::Integer(value), line, start_col));
                }
                continue;
            }

            return Err(LexError::UnknownChar { line, col, ch: b as char });
        }

        // End of content line: emit Newline if we are at one, then advance.
        if i < bytes.len() {
            let nl_col = col;
            tokens.push(Token::new(TokenKind::Newline, line, nl_col));
            if bytes[i] == b'\r' {
                i += 1;
                if i < bytes.len() && bytes[i] == b'\n' {
                    i += 1;
                }
            } else {
                i += 1;
            }
            line += 1;
        }
    }

    // EOF: close any open indent levels with synthetic Dedents at (line, col=1).
    stack.close(line, 1, &mut tokens);
    tokens.push(Token::new(TokenKind::Eof, line, 1));
    Ok(tokens)
}
```

- [ ] **Step 5: Run tests, verify they PASS**

Run: `cargo test --lib lexer::tests`
Expected: all tests including the 6 new ones pass. If `lex_indent_then_dedent_back_to_zero` fails, double-check that `adjust_to` uses `<` (not `<=`) when comparing levels and that the Dedent emit happens BEFORE the next content token, not after.

- [ ] **Step 6: Commit**

```bash
git add compiler/src/lexer/
git commit -m "feat(m2/lexer): indent machine (INDENT/DEDENT)

Adds compiler/src/lexer/indent.rs with an IndentStack that emits
virtual Indent / Dedent tokens whenever the leading-space level
changes between non-blank, non-comment-only lines. Tab in leading
whitespace is rejected with TabInIndent. Closing the file emits a
Dedent per outstanding indent level.

6 new tests cover simple block, dedent-to-zero, blank-lines-don't-
count, comment-lines-don't-count, EOF-closing, and nested indent."
```

---

## Task 7: Lexer — pipeline-continuation rule

Per grammar §5.2: a line that begins with `->` and whose leading-space count is strictly greater than the enclosing model body's indent is a continuation of the previous `pipeline_stmt`. The lexer must NOT emit `Indent`/`Dedent` for such lines — it just emits the `Arrow` token in-line with the previous tokens.

**Files:**
- Modify: `compiler/src/lexer/mod.rs` (special-case continuation lines)
- Modify: `compiler/src/lexer/tests.rs` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/lexer/tests.rs`:

```rust
#[test]
fn lex_pipeline_continuation_basic() {
    // Continuation line starts with ->: no Indent/Dedent for that line.
    let src = "model X:\n    a -> b\n      -> c\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent,
            Ident("a".into()), Arrow, Ident("b".into()), Newline,
            Arrow, Ident("c".into()), Newline,        // continuation: NO Indent
            Dedent, Eof,
        ],
    );
}

#[test]
fn lex_pipeline_continuation_then_real_dedent() {
    // After pipeline ends, returning to the outer indent level should NOT
    // emit a Dedent (we never indented for the continuation).
    let src = "model X:\n    a -> b\n      -> c\nfoo\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent,
            Ident("a".into()), Arrow, Ident("b".into()), Newline,
            Arrow, Ident("c".into()), Newline,
            Dedent,
            Ident("foo".into()), Newline,
            Eof,
        ],
    );
}

#[test]
fn lex_two_continuations_in_a_row() {
    let src = "model X:\n    a -> b\n      -> c\n      -> d\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent,
            Ident("a".into()), Arrow, Ident("b".into()), Newline,
            Arrow, Ident("c".into()), Newline,
            Arrow, Ident("d".into()), Newline,
            Dedent, Eof,
        ],
    );
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib lexer::tests`
Expected: previous tests pass; 3 new fail because the lexer currently emits Indent for any deeper-indented content line.

- [ ] **Step 3: Implement continuation-line detection**

In `compiler/src/lexer/mod.rs`, before calling `stack.adjust_to`, check whether this is a continuation line. A continuation line satisfies all of:
1. It starts with `->` (after the leading spaces).
2. Its leading-space count is strictly greater than the stack's current top.

In that case, **skip** the `adjust_to` call entirely; just lex the rest of the line normally.

Replace the lines starting from `// Content line: adjust indent stack ...` and through `let mut col: u32 = first_col;` with the following:

```rust
        // Detect pipeline continuation: a content line that starts with "->"
        // at an indent strictly greater than the current block's indent is a
        // continuation of the previous pipeline_stmt — no Indent/Dedent emitted.
        let starts_with_arrow = i + 1 < bytes.len()
            && bytes[i] == b'-'
            && bytes[i + 1] == b'>';
        let is_continuation = starts_with_arrow
            && indent_spaces > stack.current_top();

        if !is_continuation {
            let first_col = indent_spaces as u32 + 1;
            stack.adjust_to(indent_spaces, line, first_col, &mut tokens)?;
        }
        // Whether continuation or not, the column of the first non-whitespace
        // character is `indent_spaces + 1`.
        let mut col: u32 = indent_spaces as u32 + 1;
```

Add a `current_top` accessor to the `IndentStack` in `compiler/src/lexer/indent.rs`:

```rust
impl IndentStack {
    // ... existing methods ...

    pub fn current_top(&self) -> usize {
        *self.levels.last().expect("stack always non-empty")
    }
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib lexer::tests`
Expected: all 18 tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/lexer/
git commit -m "feat(m2/lexer): pipeline-continuation rule

A line starting with '->' at indent strictly greater than the
enclosing block's indent is a pipeline_stmt continuation — no
Indent/Dedent emitted, the Arrow joins the previous pipeline.
Implements grammar §5.2.

3 new tests cover basic continuation, continuation then dedent,
and two consecutive continuations."
```

---

## Task 8: Lexer — error cases

This task explicitly tests every `LexError` variant.

**Files:**
- Modify: `compiler/src/lexer/tests.rs` (add tests)

(No implementation changes — the variants are already wired.)

- [ ] **Step 1: Add failing-on-spec tests**

Append to `compiler/src/lexer/tests.rs`:

```rust
#[test]
fn err_tab_in_indent() {
    let src = "model X:\n\tfoo\n";              // tab as the leading whitespace
    let err = lex(src).unwrap_err();
    match err {
        LexError::TabInIndent { line, col } => {
            assert_eq!(line, 2);
            assert_eq!(col, 1);
        }
        other => panic!("expected TabInIndent, got {other:?}"),
    }
}

#[test]
fn err_unknown_char() {
    let src = "model X:\n    @\n";
    let err = lex(src).unwrap_err();
    match err {
        LexError::UnknownChar { ch, .. } => assert_eq!(ch, '@'),
        other => panic!("expected UnknownChar, got {other:?}"),
    }
}

#[test]
fn err_bad_dedent() {
    // Body indented to 4, then dedent to 2 (not a level on the stack).
    let src = "model X:\n    foo\n  bar\n";
    let err = lex(src).unwrap_err();
    match err {
        LexError::BadDedent { line, .. } => assert_eq!(line, 3),
        other => panic!("expected BadDedent, got {other:?}"),
    }
}

#[test]
fn err_position_in_lex_error() {
    let src = "model X:\n\tfoo\n";
    let err = lex(src).unwrap_err();
    assert_eq!(err.position(), (2, 1));
}

#[test]
fn lex_error_displays_human_message() {
    let err = LexError::TabInIndent { line: 5, col: 1 };
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("tab"), "got: {msg}");
}
```

(Note: `BadNumber` is harder to trigger with the current accept-only-digit-then-dot-then-digit logic — strings like `"5."` simply lex as `5` followed by an unknown `.`, not as a bad number. We document this behaviour and skip the test.)

- [ ] **Step 2: Run tests, verify they PASS**

Run: `cargo test --lib lexer::tests`
Expected: all 23 tests pass. These tests verify behaviour we've already implemented; they fail only if a prior task's implementation is buggy.

- [ ] **Step 3: Commit**

```bash
git add compiler/src/lexer/tests.rs
git commit -m "test(m2/lexer): error variants and position tracking

Verifies TabInIndent, UnknownChar, BadDedent emit at the right
line:col, that LexError::position() returns it, and that Display
yields a human-readable message. No implementation changes."
```

---

## Task 9: Parser scaffolding + `ParseError` + helpers

**Files:**
- Create: `compiler/src/parser/mod.rs` (scaffolding, no parse_* yet)
- Modify: `compiler/src/lib.rs` (add `pub mod parser;`)

- [ ] **Step 1: Create `compiler/src/parser/mod.rs` with the parser scaffold**

```rust
//! Hand-written recursive-descent parser for NFL.
//!
//! Each function consumes tokens from the [`Parser`] cursor and returns either
//! an AST node or a [`ParseError`]. There is no error recovery in v0.1 — the
//! first error halts parsing.

use crate::lexer::tokens::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
    pub col: u32,
    pub expected: Vec<&'static str>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Stateful parser holding a token slice and a cursor.
pub(crate) struct Parser<'t> {
    tokens: &'t [Token],
    pos: usize,
}

impl<'t> Parser<'t> {
    pub fn new(tokens: &'t [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Token at the cursor (does not advance). Panics only if the lexer failed
    /// to emit an Eof terminator, which would be a lexer bug.
    pub fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    /// Token kind at the cursor.
    pub fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    /// Advance one token. Returns the consumed token.
    pub fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    /// If the current token's kind matches `expected`, consume it; otherwise
    /// return a ParseError naming what was expected.
    pub fn consume(&mut self, expected: TokenKind, label: &'static str) -> Result<&Token, ParseError> {
        if self.peek_kind() == &expected {
            Ok(self.advance())
        } else {
            Err(self.error_expected(&[label]))
        }
    }

    /// Conditionally consume the current token if its kind matches `expected`.
    /// Returns true if consumed.
    pub fn eat(&mut self, expected: &TokenKind) -> bool {
        if self.peek_kind() == expected {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip any leading Newline tokens.
    pub fn skip_newlines(&mut self) {
        while matches!(self.peek_kind(), TokenKind::Newline) {
            self.advance();
        }
    }

    /// Build a ParseError pointing at the current token, listing what was expected.
    pub fn error_expected(&self, expected: &[&'static str]) -> ParseError {
        let t = self.peek();
        let found = describe_kind(&t.kind);
        ParseError {
            message: format!("expected {}, found {}", join_alts(expected), found),
            line: t.line,
            col: t.col,
            expected: expected.to_vec(),
        }
    }

    pub fn position(&self) -> (u32, u32) {
        let t = self.peek();
        (t.line, t.col)
    }
}

fn join_alts(items: &[&str]) -> String {
    match items.len() {
        0 => "<nothing>".to_string(),
        1 => format!("'{}'", items[0]),
        _ => {
            let last = items.last().unwrap();
            let head: Vec<String> = items[..items.len() - 1]
                .iter()
                .map(|s| format!("'{}'", s))
                .collect();
            format!("{} or '{}'", head.join(", "), last)
        }
    }
}

fn describe_kind(k: &TokenKind) -> String {
    use TokenKind::*;
    match k {
        Model => "'model'".into(),
        Tensor => "'Tensor'".into(),
        LBracket => "'['".into(),
        RBracket => "']'".into(),
        Colon => "':'".into(),
        Comma => "','".into(),
        Equals => "'='".into(),
        Arrow => "'->'".into(),
        Ident(s) => format!("identifier '{}'", s),
        Integer(n) => format!("integer literal {}", n),
        Number(n) => format!("number literal {}", n),
        Newline => "newline".into(),
        Indent => "indent".into(),
        Dedent => "dedent".into(),
        Eof => "end of file".into(),
    }
}
```

- [ ] **Step 2: Wire in `compiler/src/lib.rs`**

```rust
//! NFL Compiler — library crate.

pub mod ast;
pub mod lexer;
pub mod parser;
```

- [ ] **Step 3: Verify build**

Run: `cargo build`
Expected: clean build, no warnings.

(No tests added in this task — the parser entry point doesn't exist yet, so there's nothing to test. Tests come with each `parse_*` function.)

- [ ] **Step 4: Commit**

```bash
git add compiler/src/parser/ compiler/src/lib.rs
git commit -m "feat(m2/parser): scaffolding — Parser, ParseError, helpers

Stateful parser holding a &[Token] and a cursor. Helpers: peek,
advance, consume, eat, skip_newlines, error_expected. ParseError
implements Display and std::error::Error. No parse_* functions
yet — they land in subsequent tasks."
```

---

## Task 10: Parser — leaves: `arg_value`, `named_arg`, `op_args`, `operation`

These are the deepest productions. Building bottom-up makes each test independent of higher-level parsers.

**Files:**
- Modify: `compiler/src/parser/mod.rs` (add four `parse_*` functions)
- Create: `compiler/src/parser/tests.rs` (unit tests)

- [ ] **Step 1: Add failing tests**

Create `compiler/src/parser/tests.rs`:

```rust
//! Unit tests for the parser, exercising one production at a time.

use super::*;
use crate::ast::*;
use crate::lexer::lex;

fn parser_of(src: &str) -> Parser<'_> {
    // Test helper: lex `src`, leak the tokens to keep them alive for the
    // returned Parser. Tests are short-lived so the leak is harmless.
    let toks = lex(src).expect("lex must succeed in test");
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    Parser::new(leaked)
}

#[test]
fn parse_arg_value_integer() {
    let mut p = parser_of("512");
    let v = parse_arg_value(&mut p).unwrap();
    assert_eq!(v, ArgValue::Integer(512));
}

#[test]
fn parse_arg_value_float() {
    let mut p = parser_of("0.2");
    let v = parse_arg_value(&mut p).unwrap();
    let ArgValue::Float(f) = v else { panic!("expected Float") };
    assert!((f - 0.2).abs() < 1e-12);
}

#[test]
fn parse_arg_value_symbol() {
    let mut p = parser_of("batch");
    let v = parse_arg_value(&mut p).unwrap();
    assert_eq!(v, ArgValue::Symbol("batch".into()));
}

#[test]
fn parse_operation_no_args() {
    let mut p = parser_of("relu");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.name, "relu");
    assert!(op.args.is_empty());
}

#[test]
fn parse_operation_one_positional() {
    let mut p = parser_of("linear[512]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.name, "linear");
    assert_eq!(op.args.len(), 1);
    assert_eq!(op.args[0], OpArg::Positional(ArgValue::Integer(512)));
}

#[test]
fn parse_operation_named_only() {
    let mut p = parser_of("dropout[rate=0.2]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.args.len(), 1);
    let OpArg::Named { name, value: ArgValue::Float(f) } = &op.args[0] else {
        panic!("expected named float arg");
    };
    assert_eq!(name, "rate");
    assert!((f - 0.2).abs() < 1e-12);
}

#[test]
fn parse_operation_mixed_positional_then_named() {
    let mut p = parser_of("linear[16, bias=true]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.args.len(), 2);
    assert_eq!(op.args[0], OpArg::Positional(ArgValue::Integer(16)));
    let OpArg::Named { name, value } = &op.args[1] else { panic!() };
    assert_eq!(name, "bias");
    assert_eq!(*value, ArgValue::Symbol("true".into()));
}

#[test]
fn parse_operation_named_before_positional_is_error() {
    let mut p = parser_of("linear[a=1, 2]");
    let err = parse_operation(&mut p).unwrap_err();
    assert!(
        err.message.to_lowercase().contains("positional")
            || err.message.to_lowercase().contains("named"),
        "expected message about positional/named ordering, got: {}",
        err.message
    );
}
```

Wire the test module from `compiler/src/parser/mod.rs`:

```rust
#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib parser::tests`
Expected: all 8 tests fail to even compile because `parse_arg_value` and `parse_operation` don't exist yet. (TDD red.)

- [ ] **Step 3: Implement the four functions**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::{ArgValue, OpArg, Operation, Span};

pub(crate) fn parse_arg_value(p: &mut Parser) -> Result<ArgValue, ParseError> {
    match p.peek_kind().clone() {
        TokenKind::Integer(n) => {
            p.advance();
            Ok(ArgValue::Integer(n))
        }
        TokenKind::Number(n) => {
            p.advance();
            Ok(ArgValue::Float(n))
        }
        TokenKind::Ident(s) => {
            p.advance();
            Ok(ArgValue::Symbol(s))
        }
        _ => Err(p.error_expected(&["integer", "number", "identifier"])),
    }
}

pub(crate) fn parse_named_arg(p: &mut Parser) -> Result<(String, ArgValue), ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    p.advance();
    p.consume(TokenKind::Equals, "=")?;
    let value = parse_arg_value(p)?;
    Ok((name, value))
}

/// Parse `op_args = positional_args , [ "," , named_args ] | named_args`.
/// Returns the list of arguments. Caller has already consumed `[` and is
/// expected to consume the matching `]` afterwards.
///
/// Uses `peek_at(1)` (added to `Parser` in this task — see the next code
/// block) to look one token past the cursor and decide whether the next
/// item is a `named_arg` (`Ident "="`) or a positional `arg_value`.
pub(crate) fn parse_op_args(p: &mut Parser) -> Result<Vec<OpArg>, ParseError> {
    let mut args = Vec::new();
    let mut seen_named = false;

    loop {
        // Decide whether the next item is a named_arg (Ident "=" ...) or a
        // positional arg (any arg_value).
        let is_named = matches!(p.peek_kind(), TokenKind::Ident(_))
            && matches!(p.peek_at(1), Some(TokenKind::Equals));

        if is_named {
            let (name, value) = parse_named_arg(p)?;
            args.push(OpArg::Named { name, value });
            seen_named = true;
        } else {
            if seen_named {
                return Err(ParseError {
                    message: "positional argument cannot follow a named argument".into(),
                    line: p.peek().line,
                    col: p.peek().col,
                    expected: vec!["named argument", "']'"],
                });
            }
            let value = parse_arg_value(p)?;
            args.push(OpArg::Positional(value));
        }

        // Either consume a comma and continue, or break.
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    Ok(args)
}

pub(crate) fn parse_operation(p: &mut Parser) -> Result<Operation, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();

    let mut args = Vec::new();
    if p.eat(&TokenKind::LBracket) {
        // Empty bracket "[]" is invalid per spec.
        if matches!(p.peek_kind(), TokenKind::RBracket) {
            return Err(ParseError {
                message: "operation argument list cannot be empty; omit the brackets if there are no arguments".into(),
                line: p.peek().line,
                col: p.peek().col,
                expected: vec!["argument"],
            });
        }
        args = parse_op_args(p)?;
        p.consume(TokenKind::RBracket, "]")?;
    }

    Ok(Operation {
        name,
        args,
        span: Span::new(line, col),
    })
}
```

Add this method to the existing `impl<'t> Parser<'t>` block in Task 9's `compiler/src/parser/mod.rs` (place it near the other `peek*` methods):

```rust
    /// Look at the kind `n` tokens ahead of the cursor (0 = current).
    /// Returns `None` if the lookahead is past the end (after Eof).
    pub fn peek_at(&self, n: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + n).map(|t| &t.kind)
    }
```

`parse_op_args` (above) already uses this helper.

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib parser::tests`
Expected: all 8 tests pass. If `parse_operation_named_before_positional_is_error` fails, double-check the `seen_named` flag tracking.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test`
Expected: all 31 tests pass (23 lexer + 8 parser).

- [ ] **Step 6: Commit**

```bash
git add compiler/src/parser/
git commit -m "feat(m2/parser): leaf productions

Implements parse_arg_value, parse_named_arg, parse_op_args, and
parse_operation. Empty op brackets ([]) are rejected with a
helpful message. Named-then-positional argument ordering is
rejected with a helpful message.

8 unit tests cover all valid arg shapes and the two error cases."
```

---

## Task 11: Parser — `pipeline_stmt`

**Files:**
- Modify: `compiler/src/parser/mod.rs` (add `parse_pipeline_stmt`)
- Modify: `compiler/src/parser/tests.rs` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_pipeline_one_step() {
    let mut p = parser_of("x -> linear[2]");
    let ps = parse_pipeline_stmt(&mut p).unwrap();
    assert_eq!(ps.source, "x");
    assert_eq!(ps.steps.len(), 1);
    assert_eq!(ps.steps[0].name, "linear");
}

#[test]
fn parse_pipeline_three_steps() {
    let mut p = parser_of("x -> linear[8] -> relu -> softmax");
    let ps = parse_pipeline_stmt(&mut p).unwrap();
    assert_eq!(ps.source, "x");
    assert_eq!(ps.steps.len(), 3);
    assert_eq!(ps.steps.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
               vec!["linear", "relu", "softmax"]);
}

#[test]
fn parse_pipeline_missing_arrow_after_source_is_error() {
    let mut p = parser_of("x linear[2]");        // missing "->"
    let err = parse_pipeline_stmt(&mut p).unwrap_err();
    assert!(err.message.contains("'->'"), "got: {}", err.message);
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib parser::tests::parse_pipeline`
Expected: tests fail because `parse_pipeline_stmt` doesn't exist.

- [ ] **Step 3: Implement `parse_pipeline_stmt`**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::PipelineStmt;

pub(crate) fn parse_pipeline_stmt(p: &mut Parser) -> Result<PipelineStmt, ParseError> {
    let TokenKind::Ident(source) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();

    // pipeline_chain = pipeline_step , { pipeline_step } ; — at least one step.
    let mut steps = Vec::new();
    p.consume(TokenKind::Arrow, "->")?;
    steps.push(parse_operation(p)?);
    while matches!(p.peek_kind(), TokenKind::Arrow) {
        p.advance();
        steps.push(parse_operation(p)?);
    }

    Ok(PipelineStmt {
        source,
        steps,
        span: Span::new(line, col),
    })
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib parser::tests`
Expected: all 11 parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/parser/
git commit -m "feat(m2/parser): pipeline_stmt

Parses an identifier followed by one or more '-> operation' steps.
Span attaches to the source identifier's position. Missing arrow
after the source produces a clear error message.

3 new tests."
```

---

## Task 12: Parser — `type_expr`, `dim_list`, `dim`, `variable_decl`

**Files:**
- Modify: `compiler/src/parser/mod.rs`
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_type_expr_integer_dims() {
    let mut p = parser_of("Tensor[8, 4]");
    let t = parse_type_expr(&mut p).unwrap();
    assert_eq!(t.name, "Tensor");
    assert_eq!(t.dims, vec![Dim::Integer(8), Dim::Integer(4)]);
}

#[test]
fn parse_type_expr_symbolic_dims() {
    let mut p = parser_of("Tensor[batch, input]");
    let t = parse_type_expr(&mut p).unwrap();
    assert_eq!(t.dims, vec![Dim::Symbol("batch".into()), Dim::Symbol("input".into())]);
}

#[test]
fn parse_type_expr_empty_brackets_is_error() {
    let mut p = parser_of("Tensor[]");
    let err = parse_type_expr(&mut p).unwrap_err();
    assert!(err.message.to_lowercase().contains("dim")
            || err.message.to_lowercase().contains("empty"),
            "got: {}", err.message);
}

#[test]
fn parse_variable_decl_basic() {
    let mut p = parser_of("x: Tensor[batch, 4]");
    let v = parse_variable_decl(&mut p).unwrap();
    assert_eq!(v.name, "x");
    assert_eq!(v.ty.dims.len(), 2);
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

- [ ] **Step 3: Implement**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::{Dim, TypeExpr, VariableDecl};

pub(crate) fn parse_dim(p: &mut Parser) -> Result<Dim, ParseError> {
    match p.peek_kind().clone() {
        TokenKind::Integer(n) => {
            p.advance();
            Ok(Dim::Integer(n))
        }
        TokenKind::Ident(s) => {
            p.advance();
            Ok(Dim::Symbol(s))
        }
        _ => Err(p.error_expected(&["integer", "identifier"])),
    }
}

pub(crate) fn parse_dim_list(p: &mut Parser) -> Result<Vec<Dim>, ParseError> {
    let mut dims = vec![parse_dim(p)?];
    while p.eat(&TokenKind::Comma) {
        dims.push(parse_dim(p)?);
    }
    Ok(dims)
}

pub(crate) fn parse_type_expr(p: &mut Parser) -> Result<TypeExpr, ParseError> {
    let (line, col) = (p.peek().line, p.peek().col);
    p.consume(TokenKind::Tensor, "Tensor")?;
    p.consume(TokenKind::LBracket, "[")?;
    if matches!(p.peek_kind(), TokenKind::RBracket) {
        return Err(ParseError {
            message: "Tensor type requires at least one dimension; empty dim_list is invalid".into(),
            line: p.peek().line,
            col: p.peek().col,
            expected: vec!["integer", "identifier"],
        });
    }
    let dims = parse_dim_list(p)?;
    p.consume(TokenKind::RBracket, "]")?;
    Ok(TypeExpr {
        name: "Tensor".to_string(),
        dims,
        span: Span::new(line, col),
    })
}

pub(crate) fn parse_variable_decl(p: &mut Parser) -> Result<VariableDecl, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Colon, ":")?;
    let ty = parse_type_expr(p)?;
    Ok(VariableDecl {
        name,
        ty,
        span: Span::new(line, col),
    })
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib parser::tests`
Expected: 15 parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/parser/
git commit -m "feat(m2/parser): type_expr, dim_list, dim, variable_decl

Tensor[<dims>] with at least one dim (integer literal or symbolic
identifier). Empty Tensor[] is rejected with a clear message.

4 new tests."
```

---

## Task 13: Parser — `named_value`, `model_params`

**Files:**
- Modify: `compiler/src/parser/mod.rs`
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_named_value_basic() {
    let mut p = parser_of("batch=32");
    let nv = parse_named_value(&mut p).unwrap();
    assert_eq!(nv.name, "batch");
    assert_eq!(nv.value, 32);
}

#[test]
fn parse_model_params_three() {
    let mut p = parser_of("batch=32, input=784, output=10");
    let params = parse_model_params(&mut p).unwrap();
    assert_eq!(params.len(), 3);
    assert_eq!(params[0].name, "batch");
    assert_eq!(params[2].value, 10);
}

#[test]
fn parse_model_params_one() {
    let mut p = parser_of("batch=8");
    let params = parse_model_params(&mut p).unwrap();
    assert_eq!(params.len(), 1);
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

- [ ] **Step 3: Implement**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::NamedValue;

pub(crate) fn parse_named_value(p: &mut Parser) -> Result<NamedValue, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Equals, "=")?;
    // `.clone()` here turns the borrowed TokenKind into an owned one so we
    // can pattern-match without fighting the borrow checker. Integer is a
    // `u64` (Copy) so the clone is essentially free.
    let TokenKind::Integer(value) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["integer literal"]));
    };
    p.advance();
    Ok(NamedValue {
        name,
        value,
        span: Span::new(line, col),
    })
}

pub(crate) fn parse_model_params(p: &mut Parser) -> Result<Vec<NamedValue>, ParseError> {
    let mut params = vec![parse_named_value(p)?];
    while p.eat(&TokenKind::Comma) {
        params.push(parse_named_value(p)?);
    }
    Ok(params)
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib parser::tests`
Expected: 18 parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/parser/
git commit -m "feat(m2/parser): named_value, model_params

Comma-separated 'name=integer' list. Non-empty (one or more)
matching the grammar's model_params production.

3 new tests."
```

---

## Task 14: Parser — `model_body`, `model_stmt`, `model_def`

This is where INDENT/DEDENT tokens come into play.

**Files:**
- Modify: `compiler/src/parser/mod.rs`
- Modify: `compiler/src/parser/tests.rs`

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_model_def_minimal() {
    let mut p = parser_of("model TinyMLP [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2] -> softmax\n");
    let m = parse_model_def(&mut p).unwrap();
    assert_eq!(m.name, "TinyMLP");
    assert_eq!(m.params.len(), 1);
    assert_eq!(m.body.len(), 2);
    assert!(matches!(m.body[0], ModelStmt::VariableDecl(_)));
    assert!(matches!(m.body[1], ModelStmt::Pipeline(_)));
}

#[test]
fn parse_model_def_three_params() {
    let src = "model X [batch=32, input=784, output=10]:\n    x: Tensor[batch, input]\n    x -> linear[output] -> softmax\n";
    let mut p = parser_of(src);
    let m = parse_model_def(&mut p).unwrap();
    assert_eq!(m.params.len(), 3);
    let ModelStmt::Pipeline(ps) = &m.body[1] else { panic!() };
    assert_eq!(ps.steps.len(), 2);
}

#[test]
fn parse_model_def_missing_colon_is_error() {
    let mut p = parser_of("model X [batch=8]\n    x: Tensor[batch, 4]\n    x -> linear[2]\n");
    let err = parse_model_def(&mut p).unwrap_err();
    assert!(err.message.contains("':'") || err.message.to_lowercase().contains("colon"),
            "got: {}", err.message);
}
```

- [ ] **Step 2: Run tests, verify they FAIL**

- [ ] **Step 3: Implement**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::{ModelDef, ModelStmt};

pub(crate) fn parse_model_stmt(p: &mut Parser) -> Result<ModelStmt, ParseError> {
    // Disambiguate: variable_decl is `Ident ":" Tensor[...]`, pipeline_stmt
    // is `Ident "->" ...`. Look at the token after the leading identifier.
    let after = match p.peek_at(1) {
        Some(k) => k,
        None => return Err(p.error_expected(&["':'", "'->'"])),
    };
    match after {
        TokenKind::Colon => Ok(ModelStmt::VariableDecl(parse_variable_decl(p)?)),
        TokenKind::Arrow => Ok(ModelStmt::Pipeline(parse_pipeline_stmt(p)?)),
        _ => Err(p.error_expected(&["':'", "'->'"])),
    }
}

pub(crate) fn parse_model_body(p: &mut Parser) -> Result<Vec<ModelStmt>, ParseError> {
    p.consume(TokenKind::Indent, "indented body")?;
    let mut stmts = Vec::new();
    loop {
        // Eat blank-line newlines between statements.
        p.skip_newlines();
        if matches!(p.peek_kind(), TokenKind::Dedent) {
            break;
        }
        stmts.push(parse_model_stmt(p)?);
        // After each stmt the lexer emits a Newline (or it's followed
        // immediately by Dedent at EOF). Consume one if present.
        p.eat(&TokenKind::Newline);
    }
    p.consume(TokenKind::Dedent, "dedent")?;
    Ok(stmts)
}

pub(crate) fn parse_model_def(p: &mut Parser) -> Result<ModelDef, ParseError> {
    let (line, col) = (p.peek().line, p.peek().col);
    p.consume(TokenKind::Model, "model")?;
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["model name (identifier)"]));
    };
    p.advance();
    p.consume(TokenKind::LBracket, "[")?;
    let params = parse_model_params(p)?;
    p.consume(TokenKind::RBracket, "]")?;
    p.consume(TokenKind::Colon, ":")?;
    p.consume(TokenKind::Newline, "newline")?;
    let body = parse_model_body(p)?;
    Ok(ModelDef {
        name,
        params,
        body,
        span: Span::new(line, col),
    })
}
```

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib parser::tests`
Expected: 21 parser tests pass.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/parser/
git commit -m "feat(m2/parser): model_body, model_stmt, model_def

model_def consumes 'model' Ident '[' params ']' ':' Newline INDENT
body DEDENT. model_stmt disambiguates between variable_decl and
pipeline_stmt by looking at the token after the leading identifier
(':' vs '->').

3 new tests."
```

---

## Task 15: Parser — `nfl_source` + library entry point `parse(&str)`

**Files:**
- Modify: `compiler/src/parser/mod.rs` (add `parse_nfl_source`)
- Modify: `compiler/src/lib.rs` (add `pub fn parse(&str)`)
- Modify: `compiler/src/parser/tests.rs` (add tests for nfl_source)

- [ ] **Step 1: Add failing tests**

Append to `compiler/src/parser/tests.rs`:

```rust
#[test]
fn parse_nfl_source_one_model() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2]\n";
    let toks = lex(src).unwrap();
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    let mut p = Parser::new(leaked);
    let nfl = parse_nfl_source(&mut p).unwrap();
    assert_eq!(nfl.models.len(), 1);
}

#[test]
fn parse_nfl_source_two_models() {
    let src = "model A [batch=4]:\n    x: Tensor[batch, 1]\n    x -> linear[1]\n\nmodel B [batch=4]:\n    x: Tensor[batch, 1]\n    x -> linear[1]\n";
    let toks = lex(src).unwrap();
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    let mut p = Parser::new(leaked);
    let nfl = parse_nfl_source(&mut p).unwrap();
    assert_eq!(nfl.models.len(), 2);
    assert_eq!(nfl.models[0].name, "A");
    assert_eq!(nfl.models[1].name, "B");
}
```

Also add a top-level test for the library entry:

```rust
#[test]
fn library_parse_round_trip_minimal() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> softmax\n";
    let nfl = crate::parse(src).expect("must parse");
    assert_eq!(nfl.models[0].name, "X");
}
```

(The last test references `crate::parse` — wired in Step 3.)

- [ ] **Step 2: Run tests, verify they FAIL**

Run: `cargo test --lib`
Expected: tests fail because `parse_nfl_source` and `crate::parse` don't exist yet.

- [ ] **Step 3: Implement `parse_nfl_source` and `parse`**

Append to `compiler/src/parser/mod.rs`:

```rust
use crate::ast::NflSource;

pub fn parse_nfl_source(p: &mut Parser) -> Result<NflSource, ParseError> {
    let mut models = Vec::new();
    p.skip_newlines();
    while !matches!(p.peek_kind(), TokenKind::Eof) {
        models.push(parse_model_def(p)?);
        p.skip_newlines();
    }
    Ok(NflSource { models })
}
```

In `compiler/src/lib.rs`, add the public entry point:

```rust
//! NFL Compiler — library crate.

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::LexError;
pub use parser::ParseError;

/// Top-level entry point: lex and parse NFL source into an AST.
///
/// Returns the first error encountered (parsing halts on first error in v0.1).
pub fn parse(source: &str) -> Result<NflSource, ParseError> {
    let tokens = lexer::lex(source).map_err(|e| {
        let (line, col) = e.position();
        ParseError {
            message: format!("{e}"),
            line,
            col,
            expected: Vec::new(),
        }
    })?;
    let mut p = parser::Parser::new(&tokens);
    parser::parse_nfl_source(&mut p)
}
```

Note: this adds `pub use ast::*;` so callers of `nflc::parse` can pattern-match the AST types directly. Verify there's no collision with `LexError`/`ParseError` re-exports (there isn't — those names don't exist in `ast`).

Also: `parser::Parser::new` needs to be public-from-the-crate. In Task 9, the struct is `pub(crate)`. Since `parse` lives in the same crate, this works. But `parse_nfl_source` is `pub fn` — make sure the `fn parse(_)` signature compiles (it should — `Parser` is `pub(crate)`, and we're calling it from inside the crate).

- [ ] **Step 4: Run tests, verify they PASS**

Run: `cargo test --lib`
Expected: all 24 unit tests (lexer + parser) pass, including the 3 new ones.

- [ ] **Step 5: Commit**

```bash
git add compiler/src/
git commit -m "feat(m2/parser): nfl_source + library entry parse(&str)

Adds parse_nfl_source which iterates model_def while not at Eof.
Wires nflc::parse(&str) -> Result<NflSource, ParseError> at the
library root: lexes, then parses, mapping LexError into ParseError
for a uniform return type.

3 new tests including a library round-trip."
```

---

## Task 16: CLI — usage, parse subcommand, --tokens flag, error rendering

**Files:**
- Modify: `compiler/src/main.rs` (full rewrite)

- [ ] **Step 1: Implement the CLI**

Replace `compiler/src/main.rs` content with:

```rust
//! `nflc` CLI binary.
//!
//! Subcommands:
//! - `nflc`                     → print usage to stdout, exit 0
//! - `nflc parse <file>`        → pretty-print AST to stdout, exit 0 (or err to stderr, exit 1)
//! - `nflc parse <file> --tokens` → pretty-print token stream to stdout

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => {
            print_usage();
            ExitCode::SUCCESS
        }
        [cmd] if cmd == "parse" => {
            eprintln!("error: 'parse' requires a file path");
            print_usage();
            ExitCode::FAILURE
        }
        [cmd, path] if cmd == "parse" => run_parse(PathBuf::from(path), false),
        [cmd, path, flag] if cmd == "parse" && flag == "--tokens" => {
            run_parse(PathBuf::from(path), true)
        }
        _ => {
            eprintln!("error: unknown invocation");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("nflc — NFL Compiler (Milestone 2)");
    println!();
    println!("USAGE:");
    println!("  nflc parse <file.nfl>            Parse and pretty-print the AST");
    println!("  nflc parse <file.nfl> --tokens   Print the lexer's token stream");
}

fn run_parse(path: PathBuf, tokens_only: bool) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    if tokens_only {
        match nflc::lexer::lex(&source) {
            Ok(tokens) => {
                for t in tokens {
                    println!("{:>3}:{:<3}  {:?}", t.line, t.col, t.kind);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                let (line, col) = e.position();
                eprintln!("error: {} at {}:{}:{}", e, path.display(), line, col);
                ExitCode::FAILURE
            }
        }
    } else {
        match nflc::parse(&source) {
            Ok(ast) => {
                print_ast(&ast);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
                ExitCode::FAILURE
            }
        }
    }
}

fn print_ast(nfl: &nflc::NflSource) {
    for m in &nfl.models {
        println!("model {} [", m.name);
        for p in &m.params {
            println!("  {} = {}", p.name, p.value);
        }
        println!("]:");
        for stmt in &m.body {
            print_stmt(stmt, 1);
        }
        println!();
    }
}

fn print_stmt(s: &nflc::ModelStmt, depth: usize) {
    let pad = "  ".repeat(depth);
    match s {
        nflc::ModelStmt::VariableDecl(v) => {
            println!("{pad}var {} : Tensor[{}]", v.name, format_dims(&v.ty.dims));
        }
        nflc::ModelStmt::Pipeline(ps) => {
            print!("{pad}pipeline {}", ps.source);
            for op in &ps.steps {
                print!(" -> {}", op.name);
                if !op.args.is_empty() {
                    print!("[");
                    for (i, a) in op.args.iter().enumerate() {
                        if i > 0 { print!(", "); }
                        match a {
                            nflc::OpArg::Positional(v) => print!("{}", format_arg(v)),
                            nflc::OpArg::Named { name, value } => print!("{name}={}", format_arg(value)),
                        }
                    }
                    print!("]");
                }
            }
            println!();
        }
    }
}

fn format_dims(dims: &[nflc::Dim]) -> String {
    dims.iter()
        .map(|d| match d {
            nflc::Dim::Integer(n) => n.to_string(),
            nflc::Dim::Symbol(s) => s.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_arg(a: &nflc::ArgValue) -> String {
    match a {
        nflc::ArgValue::Integer(n) => n.to_string(),
        nflc::ArgValue::Float(f) => format!("{f}"),
        nflc::ArgValue::Symbol(s) => s.clone(),
    }
}
```

- [ ] **Step 2: Verify the CLI builds and runs end-to-end**

Run: `cargo build --bin nflc`
Expected: clean build.

Run: `cargo run --bin nflc`
Expected: prints usage to stdout, exit 0.

Run: `cargo run --bin nflc -- parse ../tests/fixtures/tiny_mlp.nfl`
Expected: prints something like:
```
model TinyMLP [
  batch = 8
]:
  var x : Tensor[batch, 4]
  pipeline x -> linear[2] -> softmax
```
Exit 0.

Run: `cargo run --bin nflc -- parse ../tests/fixtures/tiny_mlp.nfl --tokens`
Expected: prints the token stream, exit 0.

Run: `cargo run --bin nflc -- parse ../tests/fixtures/classifier.nfl`
Expected: pretty-prints the classifier AST including all 7 ops, exit 0.

(If any of the above fails, debug before committing.)

- [ ] **Step 3: Commit**

```bash
git add compiler/src/main.rs
git commit -m "feat(m2/cli): nflc parse <file> [--tokens]

CLI binary implementing the three forms from spec §5.4:
- nflc                          (usage)
- nflc parse <file>             (pretty-print AST)
- nflc parse <file> --tokens    (pretty-print token stream)

Errors are rendered as 'error: <msg> at <path>:<line>:<col>' on
stderr, exit 1. Argument parsing via std::env::args, no clap."
```

---

## Task 17: Integration tests — 5 positive fixtures

**Files:**
- Create: `compiler/tests/fixtures.rs`

- [ ] **Step 1: Write the integration tests with full assertions**

Create `compiler/tests/fixtures.rs`:

```rust
//! Integration tests: parse the canonical fixtures and assert AST shape.
//!
//! Positive (5) and negative (7 — added in Task 19) live in the same file
//! under separate `mod`s for readability.

mod positive {
    use nflc::*;

    fn read_fixture(name: &str) -> String {
        let path = format!("../tests/fixtures/{name}");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
    }

    #[test]
    fn classifier() {
        let src = read_fixture("classifier.nfl");
        let nfl = parse(&src).expect("classifier.nfl must parse");

        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "Classifier");
        assert_eq!(m.params.len(), 3);
        assert_eq!(
            m.params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["batch", "input", "output"],
        );
        assert_eq!(m.params[0].value, 32);
        assert_eq!(m.params[1].value, 784);
        assert_eq!(m.params[2].value, 10);

        assert_eq!(m.body.len(), 2);

        let ModelStmt::VariableDecl(v) = &m.body[0] else { panic!("expected VariableDecl") };
        assert_eq!(v.name, "x");
        assert_eq!(v.ty.name, "Tensor");
        assert_eq!(v.ty.dims, vec![Dim::Symbol("batch".into()), Dim::Symbol("input".into())]);

        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!("expected Pipeline") };
        assert_eq!(p.source, "x");
        assert_eq!(p.steps.len(), 7);
        assert_eq!(
            p.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["linear", "relu", "dropout", "linear", "relu", "linear", "softmax"],
        );
        // Positional first linear arg.
        assert_eq!(p.steps[0].args, vec![OpArg::Positional(ArgValue::Integer(512))]);
        // Named dropout arg.
        let OpArg::Named { name, value: ArgValue::Float(f) } = &p.steps[2].args[0] else {
            panic!("expected named float arg on dropout")
        };
        assert_eq!(name, "rate");
        assert!((f - 0.2).abs() < 1e-9);
        // Symbolic-dim positional on the last linear.
        assert_eq!(p.steps[5].args, vec![OpArg::Positional(ArgValue::Symbol("output".into()))]);
        // softmax has no args.
        assert!(p.steps[6].args.is_empty());
    }

    #[test]
    fn tiny_mlp() {
        let src = read_fixture("tiny_mlp.nfl");
        let nfl = parse(&src).expect("tiny_mlp.nfl must parse");
        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "TinyMLP");
        assert_eq!(m.params.len(), 1);
        assert_eq!(m.params[0].name, "batch");
        assert_eq!(m.params[0].value, 8);

        assert_eq!(m.body.len(), 2);
        let ModelStmt::VariableDecl(v) = &m.body[0] else { panic!() };
        assert_eq!(v.ty.dims, vec![Dim::Symbol("batch".into()), Dim::Integer(4)]);

        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps.len(), 2);
    }

    #[test]
    fn pipeline_styles_three_models() {
        let src = read_fixture("pipeline_styles.nfl");
        let nfl = parse(&src).expect("pipeline_styles.nfl must parse");

        assert_eq!(nfl.models.len(), 3);
        assert_eq!(nfl.models[0].name, "SingleLine");
        assert_eq!(nfl.models[1].name, "PerStepWrap");
        assert_eq!(nfl.models[2].name, "MixedWrap");

        // All three have the same pipeline shape: x -> linear[8] -> relu -> linear[output] -> softmax.
        for m in &nfl.models {
            let ModelStmt::Pipeline(p) = &m.body[1] else { panic!("expected Pipeline in {}", m.name) };
            assert_eq!(p.steps.len(), 4, "model {} should have 4 pipeline steps", m.name);
            assert_eq!(
                p.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
                vec!["linear", "relu", "linear", "softmax"],
            );
        }
    }

    #[test]
    fn comments_are_ignored() {
        let src = read_fixture("comments.nfl");
        let nfl = parse(&src).expect("comments.nfl must parse");
        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "Commented");
        assert_eq!(m.body.len(), 2);
        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps.len(), 4);  // linear[16] -> relu -> linear[output] -> softmax
    }

    #[test]
    fn mixed_args() {
        let src = read_fixture("mixed_args.nfl");
        let nfl = parse(&src).expect("mixed_args.nfl must parse");
        let m = &nfl.models[0];
        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps[0].name, "linear");
        // First step is `linear[16, bias=true]` — one positional, one named.
        assert_eq!(p.steps[0].args.len(), 2);
        assert_eq!(p.steps[0].args[0], OpArg::Positional(ArgValue::Integer(16)));
        let OpArg::Named { name, value } = &p.steps[0].args[1] else { panic!() };
        assert_eq!(name, "bias");
        assert_eq!(*value, ArgValue::Symbol("true".into()));
    }
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test --test fixtures`
Expected: all 5 positive integration tests pass.

If any fails, the failure message tells you exactly which assertion broke. Common causes:
- Pipeline-continuation lexer rule incorrect (check Task 7 if `classifier` step counts are wrong).
- Off-by-one in `model_body` body length (check Task 14).
- Identifier capitalisation mismatch in fixture text vs assertions.

- [ ] **Step 3: Run the FULL test suite**

Run: `cargo test`
Expected: all unit tests + 5 integration tests pass. Should be ~29 tests in total at this point.

- [ ] **Step 4: Commit**

```bash
git add compiler/tests/fixtures.rs
git commit -m "test(m2): integration tests for 5 positive fixtures

Reads each fixture file, parses, asserts AST structure (model
counts, names, param values, body lengths, pipeline step names,
specific arg values). Structural assertions, not full equality —
keeps tests readable while still covering essentials."
```

---

## Task 18: Negative fixtures — 7 `.nfl` files

**Files:**
- Create: `tests/fixtures/negative/tabs_in_indent.nfl`
- Create: `tests/fixtures/negative/missing_colon.nfl`
- Create: `tests/fixtures/negative/unclosed_bracket.nfl`
- Create: `tests/fixtures/negative/empty_tensor.nfl`
- Create: `tests/fixtures/negative/empty_op_args.nfl`
- Create: `tests/fixtures/negative/named_before_positional.nfl`
- Create: `tests/fixtures/negative/bad_dedent.nfl`

Each fixture starts with a `#`-comment block stating what it tests and the expected error category. The exact line where the error occurs is documented so the integration test in Task 19 can assert it.

- [ ] **Step 1: Create `tests/fixtures/negative/tabs_in_indent.nfl`**

```nfl
# NEGATIVE: tab character used in leading whitespace on line 5.
# Expected: LexError::TabInIndent at line 5, col 1.

model X [batch=8]:
	x: Tensor[batch, 4]
    x -> linear[2]
```

(The fifth line uses a literal tab character before `x:`. Make sure your editor inserts a real `\t`, not 4 spaces.)

- [ ] **Step 2: Create `tests/fixtures/negative/missing_colon.nfl`**

```nfl
# NEGATIVE: missing ':' after model header.
# Expected: ParseError mentioning ':' at line 4 (after the ']' on line 4).

model X [batch=8]
    x: Tensor[batch, 4]
    x -> linear[2]
```

(Note: the parser will reach the end of line 4 without seeing `:` and complain. The Newline appears on line 4. The reported line might be 4 or 5 depending on whether the error fires on the missing-colon-where-it-should-be or on the unexpected newline. Task 19 will use whichever the implementation actually reports — the developer can adjust the assertion after the first run.)

- [ ] **Step 3: Create `tests/fixtures/negative/unclosed_bracket.nfl`**

```nfl
# NEGATIVE: unclosed '[' in model_params on line 4.
# Expected: ParseError mentioning ']' or ',' at line 5 (where Newline appears
# instead of more params).

model X [batch=8
    x: Tensor[batch, 4]
    x -> linear[2]
```

- [ ] **Step 4: Create `tests/fixtures/negative/empty_tensor.nfl`**

```nfl
# NEGATIVE: Tensor[] with empty dim_list on line 5.
# Expected: ParseError mentioning 'dim' or 'empty' at line 5.

model X [batch=8]:
    x: Tensor[]
    x -> linear[2]
```

- [ ] **Step 5: Create `tests/fixtures/negative/empty_op_args.nfl`**

```nfl
# NEGATIVE: linear[] with empty bracket on line 6.
# Expected: ParseError mentioning 'argument' or 'empty' at line 6.

model X [batch=8]:
    x: Tensor[batch, 4]
    x -> linear[]
```

- [ ] **Step 6: Create `tests/fixtures/negative/named_before_positional.nfl`**

```nfl
# NEGATIVE: linear[a=1, 2] — positional after named on line 6.
# Expected: ParseError mentioning 'positional' or 'named' at line 6.

model X [batch=8]:
    x: Tensor[batch, 4]
    x -> linear[a=1, 2]
```

- [ ] **Step 7: Create `tests/fixtures/negative/bad_dedent.nfl`**

```nfl
# NEGATIVE: dedent to a level (col 3) that is not on the indent stack.
# Body indent is 4; this line drops to 2.
# Expected: LexError::BadDedent at line 6.

model X [batch=8]:
    x: Tensor[batch, 4]
    x -> linear[2]
  bar
```

- [ ] **Step 8: Verify the seven files exist and have header comments**

Run:
```bash
ls tests/fixtures/negative/ && head -3 tests/fixtures/negative/*.nfl
```
Expected: 7 files, each starts with `# NEGATIVE: …`.

- [ ] **Step 9: Commit**

```bash
git add tests/fixtures/negative/
git commit -m "test(fixtures): add 7 negative .nfl fixtures

Each covers one error category: tabs_in_indent, missing_colon,
unclosed_bracket, empty_tensor, empty_op_args,
named_before_positional, bad_dedent. The header comment in each
fixture documents the expected error category and the exact line
where the error should be reported."
```

---

## Task 19: Integration tests — 7 negative fixtures

**Files:**
- Modify: `compiler/tests/fixtures.rs` (add the `negative` mod)

- [ ] **Step 1: Append the negative test module to `compiler/tests/fixtures.rs`**

```rust
mod negative {
    use nflc::*;

    fn read_fixture(name: &str) -> String {
        let path = format!("../tests/fixtures/negative/{name}");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
    }

    #[test]
    fn tabs_in_indent_at_line_5() {
        let src = read_fixture("tabs_in_indent.nfl");
        let err = parse(&src).expect_err("must reject tab in indent");
        assert!(err.message.to_lowercase().contains("tab"),
                "expected message about tabs, got: {}", err.message);
        assert_eq!(err.line, 5, "tab is on line 5 of fixture");
    }

    #[test]
    fn missing_colon_at_line_4_or_5() {
        let src = read_fixture("missing_colon.nfl");
        let err = parse(&src).expect_err("must reject missing colon");
        assert!(err.message.contains("':'") || err.message.to_lowercase().contains("colon")
                || err.message.contains("'newline'"),
                "expected message about ':', got: {}", err.message);
        assert!(err.line == 4 || err.line == 5,
                "expected error on line 4 or 5, got line {}", err.line);
    }

    #[test]
    fn unclosed_bracket_at_line_4_or_5() {
        let src = read_fixture("unclosed_bracket.nfl");
        let err = parse(&src).expect_err("must reject unclosed bracket");
        assert!(err.message.contains("']'") || err.message.contains("','"),
                "expected message about ']' or ',', got: {}", err.message);
        assert!(err.line == 4 || err.line == 5,
                "expected error on line 4 or 5, got line {}", err.line);
    }

    #[test]
    fn empty_tensor_at_line_5() {
        let src = read_fixture("empty_tensor.nfl");
        let err = parse(&src).expect_err("must reject empty Tensor[]");
        assert!(err.message.to_lowercase().contains("dim")
                || err.message.to_lowercase().contains("empty"),
                "expected message about empty dims, got: {}", err.message);
        assert_eq!(err.line, 5);
    }

    #[test]
    fn empty_op_args_at_line_6() {
        let src = read_fixture("empty_op_args.nfl");
        let err = parse(&src).expect_err("must reject linear[]");
        assert!(err.message.to_lowercase().contains("argument")
                || err.message.to_lowercase().contains("empty"),
                "expected message about empty op args, got: {}", err.message);
        assert_eq!(err.line, 6);
    }

    #[test]
    fn named_before_positional_at_line_6() {
        let src = read_fixture("named_before_positional.nfl");
        let err = parse(&src).expect_err("must reject named-then-positional");
        assert!(err.message.to_lowercase().contains("positional")
                || err.message.to_lowercase().contains("named"),
                "expected message about ordering, got: {}", err.message);
        assert_eq!(err.line, 6);
    }

    #[test]
    fn bad_dedent_at_line_6() {
        let src = read_fixture("bad_dedent.nfl");
        let err = parse(&src).expect_err("must reject bad dedent");
        assert!(err.message.to_lowercase().contains("dedent")
                || err.message.to_lowercase().contains("indent"),
                "expected message about dedent/indent, got: {}", err.message);
        assert_eq!(err.line, 6);
    }
}
```

- [ ] **Step 2: Run negative tests**

Run: `cargo test --test fixtures negative`
Expected: all 7 negative tests pass.

If a test fails because of a different line number than expected, **adjust the assertion** to the actual line the implementation reports — then re-verify the new number is correct (the fixture's header comment and the test should agree).

If a test fails because the error message does not contain the expected keyword, the implementation's error message is wrong — go fix the message in the relevant `parse_*` function or `LexError::Display` impl.

- [ ] **Step 3: Run the FULL test suite**

Run: `cargo test`
Expected: all 36+ tests pass (lexer unit + parser unit + 5 positive + 7 negative integration).

- [ ] **Step 4: Commit**

```bash
git add compiler/tests/fixtures.rs
git commit -m "test(m2): integration tests for 7 negative fixtures

Each test asserts (a) parse returns Err, (b) the error message
contains a category-specific keyword, and (c) err.line equals the
exact expected line (or one of two lines where the boundary is
ambiguous). Closes the negative-fixture debt left over from M1."
```

---

## Task 20: Cleanup + Milestone 2 close-out

**Files:**
- Delete: `compiler/lexer/`, `compiler/parser/`, `compiler/ir/`, `compiler/passes/` (and their `.gitkeep` files)
- Delete: `compiler/.gitkeep`
- Modify: `DEVLOG.md` (add M2 close-out entry)
- Modify: `CLAUDE.md` (update Current Status)

- [ ] **Step 1: Remove the stale empty `compiler/<x>/` directories**

```bash
git rm compiler/lexer/.gitkeep compiler/parser/.gitkeep compiler/ir/.gitkeep compiler/passes/.gitkeep compiler/.gitkeep 2>/dev/null
rmdir compiler/lexer compiler/parser compiler/ir compiler/passes 2>/dev/null
```

(`git rm` removes from index and disk; the trailing `rmdir`s clean up empty directories — they may already be gone.)

- [ ] **Step 2: Add the Milestone 2 close-out entry to `DEVLOG.md`**

The new entry goes **above** the existing 2026-05-02 M1 close-out entry. Use the Edit tool to find:

```
---

## 2026-05-02 — Milestone 1 closed: NFL Grammar v0.1 shipped
```

and replace with:

```
---

## 2026-05-02 — Milestone 2 closed: NFL Parser prototype shipped (Rust, std-only)

### What was done
- Bootstrapped Cargo workspace (`/Cargo.toml`) with member crate `nflc` (`compiler/`)
- Implemented hand-written lexer (`compiler/src/lexer/`) — tokens, indent machine
  (INDENT/DEDENT), pipeline-continuation rule, comments, error variants
- Implemented hand-written recursive-descent parser (`compiler/src/parser/`) — one
  `parse_*` function per EBNF production
- Defined typed AST (`compiler/src/ast.rs`) with `Span` on every node
- Implemented `nflc parse <file>` CLI with `--tokens` debug flag
- Added 7 negative fixtures under `tests/fixtures/negative/`
- Reorganised legacy `compiler/{lexer,parser,ir,passes}/` empty stubs under
  `compiler/src/` (Rust convention)

### Decisions made
None new. All design decisions were captured in
`docs/superpowers/specs/2026-05-02-m2-parser-prototype-design.md` during brainstorming.
This session executed the plan in
`docs/superpowers/plans/2026-05-02-m2-parser-prototype.md`.

### Problems encountered
- (Fill in real problems found during implementation, e.g. an off-by-one in indent
  tracking, an ambiguity in `model_stmt` disambiguation, etc. If none, write
  "None — implementation followed the plan straight through.")

### Known tech debt (carried forward — see spec §9)
1. **`TypeExpr.name: String`** is fixed to `"Tensor"` for v0.1. When v0.2 introduces
   additional types this becomes either an `enum TypeKind` or a `String` validated by
   the semantic pass. Revisit at start of v0.2 grammar work.
2. **`Span` is start-only.** End-position is omitted in v0.1; add it when the first
   consumer (likely the M7 viewer) demands a full source range.
3. **No CI.** `cargo test` is run manually. Open a small follow-up PR to add a
   GitHub Actions workflow on stable Rust before M3 ships.
4. **Crate version `0.1.0` policy undecided.** Standard semver applies, but bump
   policy for the v0.x series should be agreed before v1.0.

### Next step
Begin **Milestone 3 — UIR prototype**: build the Universal Intermediate Representation
(computation DAG) from the AST. The AST data types from this milestone are the input;
the UIR is the foundation for every architecture profile starting in M4. The first M3
decision is the UIR's data shape (DAG node-and-edge representation, sharing strategy,
shape-inference timing) — to be resolved via a fresh `superpowers:brainstorming` cycle.

---

## 2026-05-02 — Milestone 1 closed: NFL Grammar v0.1 shipped
```

(Keep the existing M1 close-out entry intact — only add the new M2 entry above it.)

- [ ] **Step 3: Update `CLAUDE.md` "Current Status"**

Find this section in `CLAUDE.md` (it currently reads):

```
## Current Status

Milestone 1 complete: NFL Grammar v0.1 (inference-only) is formally defined.
The artefacts are `language/grammar.ebnf`, `docs/language_reference/grammar.md`, and
five positive fixtures under `tests/fixtures/`.

The immediate next step is **Milestone 2 — Parser prototype**: implement a parser that
consumes `.nfl` files and emits a typed AST. The choice of implementation language
(Rust / C++ / Python / …) is the first M2 decision.
```

Replace with:

```
## Current Status

Milestone 2 complete: NFL Parser prototype shipped (Rust, std-only). The implementation
is a Cargo workspace at the repo root with member crate `nflc` under `compiler/`,
hand-written lexer + recursive-descent parser, typed AST, and a `nflc parse <file>`
CLI. All 5 positive fixtures parse cleanly; 7 new negative fixtures verify rejection
behaviour at specific (line, col).

The immediate next step is **Milestone 3 — UIR prototype**: build the Universal IR
(computation DAG) from the parsed AST. The AST is the input; the UIR is the foundation
for every architecture profile from M4 onward.
```

- [ ] **Step 4: Final end-to-end verification**

Run from the worktree root:

```bash
cargo build              # zero warnings
cargo test               # all tests green
cargo run --bin nflc -- parse tests/fixtures/classifier.nfl   # prints AST, exit 0
cargo run --bin nflc -- parse tests/fixtures/negative/missing_colon.nfl   # error on stderr, exit 1
```

Expected: every command produces the expected behaviour. If any fails, do NOT proceed to commit — fix the issue first. (The `cargo run --bin nflc -- parse <neg>` command should print to stderr and have exit code 1; verify with `; echo $?`.)

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status                  # confirm only the two .md files + the deletions are staged
git commit -m "chore(m2): close Milestone 2 — parser prototype shipped

Removes legacy empty compiler/{lexer,parser,ir,passes}/.gitkeep
stubs (replaced by compiler/src/ layout). Adds M2 close-out entry
to DEVLOG including the four tech-debt items from spec §9.
Updates CLAUDE.md Current Status."
```

---

## Done. What's next?

After Task 20, Milestone 2 is complete by the spec's acceptance criteria:

1. ✅ Workspace + crate exist; `cargo build` clean (Task 1, verified Task 20)
2. ✅ All deliverable files exist and are non-trivial (Tasks 1-16)
3. ✅ 7 negative fixtures created (Task 18)
4. ✅ `cargo test` is green (verified across Tasks 4-19, plus Task 20 final check)
5. ✅ CLI works end-to-end on positive and negative fixtures (Tasks 16, 20)
6. ✅ DEVLOG entry for M2 close exists with tech-debt items (Task 20)
7. ✅ CLAUDE.md Current Status updated (Task 20)

**Optional follow-up (recommended before M3):** push the branch and open a PR (the
spec PR conventions used in M1 still apply). The branch is `claude/m2-parser-prototype`;
the title can be "Implement Milestone 2: NFL Parser Prototype (Rust, hand-written)".

**The Milestone 3 entry-point** is a fresh `superpowers:brainstorming` cycle. Start
with the question "what shape is the UIR?" — DAG of nodes and typed edges, immutable
or mutable, shape-inferred at build time or resolved later, and how it's consumed by
profiles.
