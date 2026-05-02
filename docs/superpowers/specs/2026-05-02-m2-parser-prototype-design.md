# NFL Parser Prototype (Milestone 2) — Design Spec

> **Status:** Approved (brainstorming output, 2026-05-02)
> **Authoritative for:** Milestone 2 implementation
> **Source skill:** `superpowers:brainstorming`
> **Next skill:** `superpowers:writing-plans`
> **Builds on:** [`docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`](./2026-05-02-nfl-grammar-v0.1-design.md) (M1 grammar)

---

## 1. Context

Milestone 1 produced the formal NFL v0.1 grammar (`language/grammar.ebnf`), a human-readable
reference (`docs/language_reference/grammar.md`), and 5 positive `.nfl` fixtures in
`tests/fixtures/`. There is no executable code yet — the grammar is documentation-grade.

**Milestone 2 turns NFL from a paper language into an executable one.** It implements a
parser that reads `.nfl` source and produces a typed AST, plus a thin CLI to run it.
This is the foundation for Milestone 3 (UIR builder) and everything downstream.

**Reading order:**
1. `CLAUDE.md` — project rules; especially the dev workflow and "When adding a new operation"
2. `language/grammar.ebnf` — the source of truth this parser implements
3. `docs/language_reference/grammar.md` — human-friendly explanation of the grammar
4. `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md` — the M1 design (rationale)

---

## 2. Scope

### In scope (Milestone 2)

- Hand-written **lexer** (`compiler/src/lexer/`) that tokenises `.nfl` source, handles
  significant indentation (INDENT/DEDENT virtual tokens), pipeline-continuation lines,
  comments, and tracks position
- Hand-written **recursive-descent parser** (`compiler/src/parser/`) that consumes the
  token stream and produces a typed AST
- **AST data model** (`compiler/src/ast.rs`) — Rust enums/structs mirroring the EBNF
  productions, with a `Span` on every node
- **CLI binary `nflc`** (`compiler/src/main.rs`) supporting `nflc parse <file>` and
  `nflc parse <file> --tokens`
- **Test corpus:**
  - All 5 positive fixtures from M1 must parse to expected ASTs (structural assertions)
  - 7 new **negative** fixtures under `tests/fixtures/negative/` cover the main error
    categories; each is rejected with a specific (line, col) and an error message
    containing a keyword
- Cargo workspace at the repository root; `compiler/` is the first member crate

### Out of scope (deferred)

- **UIR construction** (computation graph from AST) — Milestone 3
- **Semantic analysis:** symbolic-dim resolution against `model_params`, operation-name
  binding to a (yet-unwritten) stdlib, type checking — Milestone 3
- **Multi-error reporting** — first error halts in v0.1; M2.5+ for parser-recovery
- **Property-based testing, fuzzing, performance benchmarks** — v0.2+
- **Snapshot testing libraries (`insta`)** — chosen hand-written stdlib-only style
- **Ariadne-style rich source-snippet errors** — v0.2+
- **CI / GitHub Actions** — separate small follow-up after M2 closes

---

## 3. Deliverables

| Path | Purpose |
|---|---|
| `Cargo.toml` (root) | Workspace declaration; lists `compiler` as the first member |
| `compiler/Cargo.toml` | Crate manifest; name `nflc`, edition 2021, `[dependencies]` empty |
| `compiler/src/lib.rs` | Library root; re-exports `parse(&str) -> Result<NflSource, ParseError>` |
| `compiler/src/main.rs` | CLI binary `nflc`; argv parsing via `std::env::args` |
| `compiler/src/ast.rs` | All AST data types (see §5.3) |
| `compiler/src/lexer/mod.rs` | Lexer entry point; `pub fn lex(src: &str) -> Result<Vec<Token>, LexError>` |
| `compiler/src/lexer/tokens.rs` | `Token`, `TokenKind`, `Span`-equivalent for tokens |
| `compiler/src/lexer/indent.rs` | Indent-stack machine and pipeline-continuation state |
| `compiler/src/lexer/tests.rs` | `#[cfg(test)]` unit tests for the lexer |
| `compiler/src/parser/mod.rs` | Parser entry point; ~17 `parse_*` functions, one per EBNF production |
| `compiler/src/parser/tests.rs` | `#[cfg(test)]` unit tests for the parser |
| `compiler/tests/fixtures.rs` | Integration tests: positive (5) + negative (7) fixtures |
| `tests/fixtures/negative/tabs_in_indent.nfl` | Tab in leading whitespace |
| `tests/fixtures/negative/missing_colon.nfl` | `model X [...]` without `:` |
| `tests/fixtures/negative/unclosed_bracket.nfl` | `[a=1` without `]` |
| `tests/fixtures/negative/empty_tensor.nfl` | `Tensor[]` (empty dim list) |
| `tests/fixtures/negative/empty_op_args.nfl` | `linear[]` (use bare `linear` instead) |
| `tests/fixtures/negative/named_before_positional.nfl` | `linear[a=1, 2]` |
| `tests/fixtures/negative/bad_dedent.nfl` | Dedent to a level not on the stack |

**Pre-existing files reorganised:**
- `compiler/lexer/.gitkeep`, `compiler/parser/.gitkeep`, `compiler/ir/.gitkeep`,
  `compiler/passes/.gitkeep` are removed; the directories themselves vanish (Rust expects
  `src/` layout). Other `.gitkeep` files (e.g. `tests/fixtures/.gitkeep`,
  `language/.gitkeep`, `docs/language_reference/.gitkeep`) are left untouched per the
  user's earlier instruction.

---

## 4. Architecture

### 4.1 Workspace layout

```
NeuralForge/
├── Cargo.toml                      # workspace root: members = ["compiler"]
├── compiler/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                  # pub fn parse() — library entry
│   │   ├── main.rs                 # CLI binary `nflc`
│   │   ├── ast.rs
│   │   ├── lexer/{mod, tokens, indent, tests}.rs
│   │   └── parser/{mod, tests}.rs
│   └── tests/fixtures.rs           # integration tests
├── language/grammar.ebnf           # M1 — unchanged
├── docs/language_reference/        # M1 — unchanged
├── tests/fixtures/                 # M1 positives — unchanged
│   └── negative/                   # NEW for M2
└── ... (DEVLOG, CLAUDE, PROJECT_SPEC, README — text updates only)
```

The workspace is created from day 1 even with one member, because M4 will add
`profiles/generic/` as the second crate. Adding members to an existing workspace is
zero-cost; refactoring a single-crate project into a workspace later is not.

### 4.2 Crate naming

- Crate name: `nflc` (NFL Compiler — short, no collisions)
- Binary name: also `nflc`
- Module organisation inside `nflc`: `lexer`, `parser`, `ast` at the crate root

### 4.3 Dependencies

`compiler/Cargo.toml`'s `[dependencies]` section is **empty**. The entire crate is
implemented against `std` only. This matches the user's "hand-written everything" choice
and the project principle "no runtime, no framework".

---

## 5. Components

### 5.1 Lexer

**Contract:**
```rust
pub fn lex(source: &str) -> Result<Vec<Token>, LexError>
```

**Token shape:**
```rust
pub struct Token {
    pub kind: TokenKind,
    pub line: u32,            // 1-based
    pub col:  u32,            // 1-based, position of first char
}

pub enum TokenKind {
    // Keywords
    Model,                    // "model"
    Tensor,                   // "Tensor"
    // Punctuation
    LBracket, RBracket,       // [ ]
    Colon, Comma, Equals,     // : , =
    Arrow,                    // ->
    // Identifiers and literals
    Ident(String),
    Integer(u64),
    Number(f64),
    // Significant whitespace
    Newline,
    Indent,                   // virtual
    Dedent,                   // virtual
    // End
    Eof,
}
```

**Indent handling:** the lexer maintains an indent stack `Vec<usize>` initially `[0]`.
After every `Newline`, the lexer counts leading spaces of the next non-empty line and:
- equal to top of stack → no token emitted
- strictly greater → push, emit one `Indent`
- strictly smaller → pop until equal, emit one `Dedent` per pop; if no equal level is
  found → `LexError::BadDedent`

**Pipeline-continuation rule** (§5.2 of the grammar): when the lexer sees a non-empty
line beginning with `->` whose leading-space count is strictly greater than the
*currently-enclosing model body's* indent, it emits the `Arrow` token in-line with the
preceding `pipeline_stmt` and does **not** emit `Indent`/`Dedent` for that line. This
needs a small amount of context (am I currently inside a pipeline_stmt?) — implemented
as a single bool flag in the lexer state.

**Comments:** `#` to end-of-line. Eaten by the lexer; never produces a token. They do
not affect indentation (a comment-only line is treated as blank for indent purposes).

**`LexError` variants:**
- `TabInIndent { line, col }`
- `BadDedent { line, col }`
- `UnknownChar { line, col, char }`
- `BadNumber { line, col, raw }` (e.g. `5.` or `.5`)
- `UnexpectedEof { line, col }`

**Whitespace details (explicit to avoid ambiguity):**
- Tab characters **outside** leading whitespace are treated as inter-token spacing and
  silently consumed (no token emitted). Only tabs **in leading whitespace** are an error.
- Both `\n` (LF) and `\r\n` (CRLF) produce a single `Newline` token. The lexer accepts
  either; output ASTs do not preserve which was used.
- A line that contains only whitespace and/or a comment is treated as blank — it does
  not affect the indent stack.

### 5.2 Parser

**Contract:**
```rust
pub fn parse(tokens: &[Token]) -> Result<NflSource, ParseError>
```

The parser is a struct holding `&[Token]` and a position cursor. It exposes private
helpers `peek()`, `advance()`, `consume(expected: TokenKind)`, `consume_ident()`,
`error_expected(items: &[&str])`. There is one public-by-default `parse_*` function
per EBNF production:

```
parse_nfl_source       parse_pipeline_stmt
parse_model_def        parse_pipeline_chain   (inlined into pipeline_stmt)
parse_model_params     parse_pipeline_step    (inlined too)
parse_named_value      parse_operation
parse_model_body       parse_op_args
parse_model_stmt       parse_named_arg
parse_variable_decl    parse_arg_value
parse_type_expr
parse_dim_list
parse_dim
```

(`pipeline_chain` and `pipeline_step` are tiny enough to inline into `parse_pipeline_stmt` for readability — counted as part of the same function.)

**`ParseError` shape:**
```rust
pub struct ParseError {
    pub message:  String,                // "expected ']', found ';'"
    pub line:     u32,
    pub col:      u32,
    pub expected: Vec<&'static str>,     // ["]", ","]
}
```

`expected` is filled when the parser knows multiple alternatives could have been
accepted; useful for richer error messages later.

**Recovery:** none in v0.1. The first `Err` propagates up and halts parsing.
Multi-error reporting and recovery strategies belong to a later milestone.

### 5.3 AST data model

```rust
pub struct NflSource {
    pub models: Vec<ModelDef>,
}

pub struct ModelDef {
    pub name:   String,
    pub params: Vec<NamedValue>,
    pub body:   Vec<ModelStmt>,
    pub span:   Span,
}

pub struct NamedValue {
    pub name:  String,
    pub value: u64,
    pub span:  Span,
}

pub enum ModelStmt {
    VariableDecl(VariableDecl),
    Pipeline(PipelineStmt),
}

pub struct VariableDecl {
    pub name: String,
    pub ty:   TypeExpr,
    pub span: Span,
}

pub struct TypeExpr {
    pub name: String,                    // always "Tensor" in v0.1 (see Open Q1)
    pub dims: Vec<Dim>,
    pub span: Span,
}

pub enum Dim {
    Integer(u64),
    Symbol(String),                      // identifier referring to a model_param
}

pub struct PipelineStmt {
    pub source: String,
    pub steps:  Vec<Operation>,
    pub span:   Span,
}

pub struct Operation {
    pub name: String,
    pub args: Vec<OpArg>,                // empty if op was bare
    pub span: Span,
}

pub enum OpArg {
    Positional(ArgValue),
    Named { name: String, value: ArgValue },
}

pub enum ArgValue {
    Integer(u64),
    Float(f64),
    Symbol(String),                      // identifier (e.g. `true`, `batch`)
}

pub struct Span {
    pub line: u32,
    pub col:  u32,
}
```

`Span` is start-only in v0.1 (`(line, col)`). End-position is deferred — see Open Q2.

### 5.4 CLI

The `nflc` binary accepts these forms:

| Invocation | Behaviour |
|---|---|
| `nflc` | Print usage to stdout, exit 0 |
| `nflc parse <path>` | Read file, lex+parse, pretty-print AST to stdout, exit 0 |
| `nflc parse <path> --tokens` | Read file, lex only, print tokens (debug aid) |
| (any error) | Print `error: <message>  at <path>:<line>:<col>` to stderr, exit 1 |

Argument parsing via `std::env::args` (no `clap`). AST pretty-printing via a small
`Display` impl on each AST node (or a dedicated `printer.rs` if it grows).

---

## 6. Testing strategy

### 6.1 Unit tests

In `#[cfg(test)] mod tests` blocks within `compiler/src/lexer/tests.rs` and
`compiler/src/parser/tests.rs`. Cover:

- **Lexer:** every token kind on a synthetic single-token input; multi-token sequences;
  indent/dedent on hand-crafted indent variations; comment-eating at every position;
  tab rejection; pipeline-continuation correctness
- **Parser:** every `parse_*` function on hand-crafted token vectors; success cases
  (returns expected AST) and failure cases (returns expected `ParseError`)

Target: ~30-50 unit tests across both modules.

### 6.2 Integration tests

In `compiler/tests/fixtures.rs`:

```rust
mod positive {
    use nflc::*;

    #[test]
    fn classifier() {
        let src = std::fs::read_to_string("../tests/fixtures/classifier.nfl").unwrap();
        let ast = parse(&src).expect("classifier.nfl must parse");
        assert_eq!(ast.models.len(), 1);
        let m = &ast.models[0];
        assert_eq!(m.name, "Classifier");
        assert_eq!(m.params.len(), 3);
        // ... ~10-15 structural asserts per fixture
    }

    #[test] fn tiny_mlp() { /* ... */ }
    #[test] fn pipeline_styles_three_models() { /* ... */ }
    #[test] fn comments_are_ignored() { /* ... */ }
    #[test] fn mixed_args() { /* ... */ }
}

mod negative {
    use nflc::*;

    #[test]
    fn tabs_in_indent_at_specific_line() {
        let src = std::fs::read_to_string(
            "../tests/fixtures/negative/tabs_in_indent.nfl").unwrap();
        let err = parse(&src).unwrap_err();
        assert!(err.message.to_lowercase().contains("tab"),
                "expected tab-related error, got: {}", err.message);
        assert_eq!(err.line, 5);                 // exact line, not >=1
        // (col exact-or-range up to author's discretion per fixture)
    }

    #[test] fn missing_colon_at_specific_line() { /* ... */ }
    #[test] fn unclosed_bracket_at_specific_line() { /* ... */ }
    #[test] fn empty_tensor_at_specific_line() { /* ... */ }
    #[test] fn empty_op_args_at_specific_line() { /* ... */ }
    #[test] fn named_before_positional_at_specific_line() { /* ... */ }
    #[test] fn bad_dedent_at_specific_line() { /* ... */ }
}
```

**Assertion style:**
- Positive: structural — assert on counts, names, types, key values; not full equality
- Negative: assert (a) `parse` returned `Err`; (b) message contains a keyword
  (case-insensitive); (c) **exact** `err.line` (the fixture is tiny and deterministic,
  so the author knows precisely where the error is)

### 6.3 Test runner

`cargo test` runs everything. No additional tooling.

---

## 7. Acceptance criteria

Milestone 2 is **complete** when all of the following hold:

1. **Workspace + crate exist:**
   - `Cargo.toml` (workspace) and `compiler/Cargo.toml` (member) are valid; `cargo build`
     in the repo root completes without errors or warnings (warnings except for `dead_code`
     in genuinely-unused cases must be fixed, not silenced)
   - `compiler/Cargo.toml`'s `[dependencies]` is empty

2. **All deliverable files exist** (see §3) and are non-trivial (no empty stubs)

3. **Negative fixtures created:** 7 files under `tests/fixtures/negative/`, each with a
   header comment explaining what it tests

4. **`cargo test` is green:**
   - 30-50 unit tests passing across `lexer::tests` and `parser::tests`
   - 5 positive integration tests passing
   - 7 negative integration tests passing, each asserting a specific line number

5. **CLI works end-to-end:**
   - `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl` prints a readable
     AST tree, exit 0
   - `cargo run --bin nflc -- parse tests/fixtures/classifier.nfl --tokens` prints the
     token stream, exit 0
   - `cargo run --bin nflc -- parse tests/fixtures/negative/missing_colon.nfl` prints
     `error: ... at <file>:<line>:<col>` to stderr, exit 1

6. **DEVLOG entry for M2 close** — including:
   - The known tech-debt items from §9 (`TypeExpr.name`, `Span.end`, no CI)
   - Brief summary of what landed
   - "Next step: Milestone 3 (UIR prototype)"

7. **`CLAUDE.md` "Current Status"** updated: M2 complete, M3 (UIR) as next

The "real" semantic correctness (symbolic-dim resolution, type-checking) is a Milestone
3 acceptance criterion, not Milestone 2.

---

## 8. Deferred items

### Deferred to Milestone 3
- UIR (Universal IR) construction from AST
- Semantic analysis (symbolic-dim resolution against model_params, operation-name
  binding, type-checking)
- Stdlib of operations (currently the parser accepts any identifier; semantics will
  validate)

### Deferred to Milestone 2.5 (or whenever a real need appears)
- Multi-error reporting (parser keeps going after errors)
- Richer error messages (Ariadne-style with source snippets)
- CI pipeline (GitHub Actions running `cargo test`)
- Negative fixtures for additional categories beyond the 7 chosen

### Deferred to v0.2+
- Property-based testing (`proptest` / `quickcheck`)
- Fuzzing (`cargo-fuzz`)
- Performance benchmarks
- Snapshot testing libraries (consciously chosen against — hand-written style)

---

## 9. Open questions / known tech debt

These are NOT blockers for M2 implementation, but **must** be logged in the DEVLOG entry
that closes M2 so they remain visible to future work:

1. **`TypeExpr.name: String` is a string until v0.2.** The grammar only permits
   `Tensor[…]` in v0.1, so the field is effectively a constant `"Tensor"`. When v0.2
   introduces additional types, this will either become an `enum TypeKind` or remain
   a `String` validated by the semantic pass. No action needed in M2 — just a note.

2. **`Span` is start-only in v0.1.** We store `(line, col)` for the start of each AST
   node. End-position is omitted to keep v0.1 small. Add `end_line`, `end_col` when the
   first consumer (likely the M7 viewer or a refactoring tool) demands a full range.

3. **No CI.** `cargo test` is run manually. A small follow-up PR can add a GitHub Actions
   workflow (matrix on stable Rust, possibly nightly) once M2 has shipped.

4. **Crate version `0.1.0`.** Bump policy for future versions is not decided. Standard
   semver applies, but with a v0.x crate we have flexibility. Decide before v1.0.

---

## 10. Transition

After this spec is reviewed and approved by the user, transition to the
`superpowers:writing-plans` skill to produce a step-by-step implementation plan
covering all the deliverables in §3, written for an engineer with zero project context.
Implementation itself is the subject of a later `superpowers:executing-plans` cycle.
