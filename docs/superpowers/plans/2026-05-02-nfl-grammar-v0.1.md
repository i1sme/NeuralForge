# NFL Grammar v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the seven Milestone 1 artefacts that formalise NFL v0.1 — the EBNF grammar, a human-readable reference, and a five-file positive-fixture corpus — and close Milestone 1 with the project's documentation protocol.

**Architecture:** All deliverables are static documentation/data files. The grammar is written in ISO/IEC 14977 EBNF; the reference doc walks each production with examples; the fixtures are valid `.nfl` programs that exercise every production. There is no executable code in this milestone — verification is manual hand-tracing.

**Tech Stack:** ISO/IEC 14977 EBNF, Markdown, NFL (the language being defined). No build tools, test runners, or runtime dependencies.

**Source spec:** [`docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`](../specs/2026-05-02-nfl-grammar-v0.1-design.md). All content below is derived from there. **If anything in this plan disagrees with the spec, the spec wins** — flag the discrepancy and stop.

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/ecstatic-gagarin-13ff2a` (a worktree on branch `claude/ecstatic-gagarin-13ff2a`).

**Branch strategy:** Continue on the same branch as the spec PR ([i1sme/NeuralForge#1](https://github.com/i1sme/NeuralForge/pull/1)). Implementation commits will join that PR; the PR title may be updated when M1 closes.

**Project conventions** (from `CLAUDE.md` — read it first if unfamiliar):
- Each session ends with a `DEVLOG.md` entry (newest at the top, between the `---` and the previous entry).
- "Current Status" in `CLAUDE.md` must always reflect the project's actual state.
- All artefacts are in English. Russian only in conversation with the user.
- Do NOT touch `.gitkeep` files — leave them in place per the user's earlier instruction.

---

## File Structure

**Create (7 new files, all task-attributed below):**

| Path | Purpose | Created in |
|---|---|---|
| `language/grammar.ebnf` | Formal grammar in ISO/IEC 14977 EBNF | Task 1 |
| `tests/fixtures/classifier.nfl` | Canonical demo (README example minus loss) | Task 2 |
| `tests/fixtures/tiny_mlp.nfl` | Smallest valid NFL program | Task 2 |
| `tests/fixtures/pipeline_styles.nfl` | Three valid formattings of the same network | Task 2 |
| `tests/fixtures/comments.nfl` | Comments at every legal position | Task 2 |
| `tests/fixtures/mixed_args.nfl` | Operation with mixed positional + named args | Task 2 |
| `docs/language_reference/grammar.md` | Human-readable reference, one section per production | Task 3 |

**Modify (2 files):**

| Path | Change | Modified in |
|---|---|---|
| `DEVLOG.md` | Add Milestone 1 close-out entry at top (above existing entries) | Task 5 |
| `CLAUDE.md` | Update "Current Status" to reflect M1 complete and M2 (Parser prototype) as next | Task 5 |

**Do NOT touch:**
- Any existing `*/.gitkeep` files
- The spec file in `docs/superpowers/specs/` (treat as immutable)

---

## Verification approach (no parser yet)

There is no parser in M1. Verification is manual:

1. **Internal consistency** (Task 4): every grammar production is reachable from `nfl_source`; every referenced production is defined; the reference doc covers every production with at least one example; no contradictions between grammar comments and reference doc.
2. **Hand-tracing** (embedded in Task 3 for `classifier.nfl`; spot-checked for the other four fixtures in Task 4): walk the grammar productions top-down against each fixture's tokens and confirm a valid match exists.

Real automated parser-based validation is a Milestone 2 deliverable.

If hand-tracing surfaces a fixture-vs-grammar mismatch: **fix the fixture**, not the grammar — the grammar is the authoritative artefact, and changes to it require returning to the spec process.

---

## Task list (5 tasks, ~7 commits)

| # | Task | Commits |
|---|---|---|
| 1 | Create the formal grammar (`language/grammar.ebnf`) | 1 |
| 2 | Create the five `.nfl` fixtures | 1 |
| 3 | Create the reference doc (`docs/language_reference/grammar.md`) | 1 |
| 4 | Self-consistency review (verification pass) | 0 or 1 (only if review surfaces fixes) |
| 5 | Close out Milestone 1 (DEVLOG + CLAUDE.md Current Status) | 1 |

---

### Task 1: Create the formal grammar

**Files:**
- Create: `language/grammar.ebnf`

- [ ] **Step 1: Create the file with the full grammar**

Write `language/grammar.ebnf` with this exact content (copied verbatim from spec §5):

```ebnf
(* ============================================================ *)
(* NeuralForge Language v0.1 — formal grammar                   *)
(* Notation: ISO/IEC 14977 EBNF                                 *)
(*                                                              *)
(* Scope: inference-only.                                       *)
(* Training syntax (loss/optimiser) will arrive in v0.2.        *)
(*                                                              *)
(* Implicit semantics (NOT enforced by this grammar):           *)
(*   - The output of a model is the value produced by the last  *)
(*     operation of the last pipeline_stmt in its body.         *)
(*   - Symbolic dims in Tensor[…] must reference an identifier  *)
(*     declared in model_params.                                *)
(*   - Operation names must resolve against the stdlib.         *)
(*   - The first identifier of a pipeline_stmt must reference   *)
(*     a previously-declared variable.                          *)
(* ============================================================ *)

(* === Top-level structure ====================================== *)
nfl_source       = { newline } , { model_def , { newline } } ;

model_def        = "model" , identifier , "[" , model_params , "]" , ":" , newline
                 , INDENT , model_body , DEDENT ;

model_params     = named_value , { "," , named_value } ;
named_value      = identifier , "=" , integer ;

(* === Model body =============================================== *)
model_body       = model_stmt , { newline , model_stmt } ;
model_stmt       = variable_decl | pipeline_stmt ;

variable_decl    = identifier , ":" , type_expr ;

type_expr        = "Tensor" , "[" , dim_list , "]" ;
dim_list         = dim , { "," , dim } ;
dim              = integer | identifier ;

(* === Pipelines ================================================ *)
pipeline_stmt    = identifier , pipeline_chain ;
pipeline_chain   = pipeline_step , { pipeline_step } ;
pipeline_step    = "->" , operation ;
(* A line break inside a pipeline_chain is permitted iff the     *)
(* next non-empty line begins with "->" and its leading-space    *)
(* count is strictly greater than the enclosing model_body's     *)
(* indent. The lexer attaches such continuation lines to the     *)
(* current pipeline_stmt and does NOT emit INDENT/DEDENT for     *)
(* them — they are not their own block.                          *)

(* === Operations =============================================== *)
operation        = identifier , [ "[" , op_args , "]" ] ;
op_args          = positional_args , [ "," , named_args ]
                 | named_args ;
positional_args  = arg_value , { "," , arg_value } ;
named_args       = named_arg , { "," , named_arg } ;
named_arg        = identifier , "=" , arg_value ;
arg_value        = number | identifier ;

(* === Lexical ================================================== *)
identifier       = letter , { letter | digit | "_" } ;
letter           = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j"
                 | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t"
                 | "u" | "v" | "w" | "x" | "y" | "z"
                 | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J"
                 | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T"
                 | "U" | "V" | "W" | "X" | "Y" | "Z" ;
digit            = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
integer          = digit , { digit } ;
number           = integer , [ "." , integer ] ;

(* === Whitespace and tokens not shown above ==================== *)
(* newline    : LF or CRLF; multiple consecutive newlines are    *)
(*              equivalent to one.                               *)
(* INDENT     : virtual token emitted by the lexer when the      *)
(*              leading-space count of a line exceeds the        *)
(*              current block's indent level by exactly 4.       *)
(* DEDENT     : virtual token emitted when leading space drops   *)
(*              back to or below an enclosing block's level.     *)
(* comment    : "#" followed by any characters up to (but not    *)
(*              including) the next newline; eaten by the lexer  *)
(*              and not present in the parse tree.               *)
(* tab in leading whitespace : lexical error.                    *)
```

- [ ] **Step 2: Sanity-check the file**

Run: `wc -l language/grammar.ebnf`
Expected: between 65 and 80 lines.

Run: `grep -c '^[a-z_]* *=' language/grammar.ebnf`
Expected: `24` — the count of EBNF productions defined (`nfl_source`, `model_def`, `model_params`, `named_value`, `model_body`, `model_stmt`, `variable_decl`, `type_expr`, `dim_list`, `dim`, `pipeline_stmt`, `pipeline_chain`, `pipeline_step`, `operation`, `op_args`, `positional_args`, `named_args`, `named_arg`, `arg_value`, `identifier`, `letter`, `digit`, `integer`, `number`).

> **Heads-up:** that grep counts left-hand-sides starting at column 0 with the `name = ` pattern. The number above is the count of *unique production names*. Continuation lines (e.g. for `letter` and `op_args`) start with whitespace and don't match. As long as every production from spec §5 is present, getting an exact 24 is the goal.

If the grep count differs from your visual count of productions, re-read the file and reconcile.

- [ ] **Step 3: Commit**

```bash
git add language/grammar.ebnf
git commit -m "feat(language): add NFL v0.1 formal grammar (EBNF)

Implements the grammar specified in
docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md §5.
ISO/IEC 14977 notation; inference-only scope; explicit comments
listing implicit semantics that are NOT grammatically enforced."
```

---

### Task 2: Create the five `.nfl` fixtures

**Files:**
- Create: `tests/fixtures/classifier.nfl`
- Create: `tests/fixtures/tiny_mlp.nfl`
- Create: `tests/fixtures/pipeline_styles.nfl`
- Create: `tests/fixtures/comments.nfl`
- Create: `tests/fixtures/mixed_args.nfl`

> **Note on whitespace:** all five fixtures use 4-space indentation for model bodies, and 6-space (4 + 2) indentation for pipeline-continuation lines starting with `->`. Tabs are forbidden. Make sure your editor is configured for spaces.

- [ ] **Step 1: Create `tests/fixtures/classifier.nfl`**

Exact content:

```nfl
# Canonical demo from README/PROJECT_SPEC.
# Exercises: model_def with three params, variable_decl with symbolic dims,
# multi-line pipeline_chain, operations with no/positional/named args.

model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu
      -> dropout[rate=0.2]
      -> linear[256] -> relu
      -> linear[output] -> softmax
```

- [ ] **Step 2: Create `tests/fixtures/tiny_mlp.nfl`**

Exact content:

```nfl
# Minimal valid NFL: one model_param, one variable_decl with all-integer dims,
# one pipeline_stmt with two ops.

model TinyMLP [batch=8]:
    x: Tensor[batch, 4]

    x -> linear[2] -> softmax
```

- [ ] **Step 3: Create `tests/fixtures/pipeline_styles.nfl`**

Exact content:

```nfl
# Three valid formattings of the same network. Each sub-model is a separate
# model_def; all three are valid and equivalent semantically. Stress-tests the
# pipeline-continuation lexer rules (single line, per-step wrap, mixed wrap).

model SingleLine [batch=4, input=10, output=2]:
    x: Tensor[batch, input]
    x -> linear[8] -> relu -> linear[output] -> softmax

model PerStepWrap [batch=4, input=10, output=2]:
    x: Tensor[batch, input]
    x -> linear[8]
      -> relu
      -> linear[output]
      -> softmax

model MixedWrap [batch=4, input=10, output=2]:
    x: Tensor[batch, input]
    x -> linear[8] -> relu
      -> linear[output] -> softmax
```

- [ ] **Step 4: Create `tests/fixtures/comments.nfl`**

Exact content:

```nfl
# Top-of-file comment.
# Exercises comments at every legal position the lexer must handle.

# Comment immediately before a model.
model Commented [batch=4, input=8, output=2]:
    # Comment as the first line of a model body.
    x: Tensor[batch, input]      # Trailing comment on a variable_decl.

    # Blank-line comment between declarations.

    x -> linear[16]              # Trailing comment on a pipeline step.
      -> relu                    # Trailing comment on a continuation line.
      -> linear[output] -> softmax
# Trailing top-level comment after the model.
```

- [ ] **Step 5: Create `tests/fixtures/mixed_args.nfl`**

Exact content:

```nfl
# Operation with mixed arguments: one positional then one named.
# This is the only fixture exercising the `op_args = positional_args , "," , named_args`
# alternation. (Operations with this signature are hypothetical in v0.1; the stdlib
# may or may not include them — that's a Milestone 2 decision. Grammatically valid.)

model MixedArgs [batch=4, input=8, output=2]:
    x: Tensor[batch, input]

    x -> linear[16, bias=true]
      -> relu
      -> linear[output] -> softmax
```

- [ ] **Step 6: Sanity-check the fixtures**

Run: `ls -1 tests/fixtures/*.nfl | wc -l`
Expected: `5`.

Run: `grep -l $'\t' tests/fixtures/*.nfl || echo "no tabs — good"`
Expected: `no tabs — good` (no fixture contains a literal tab character).

Run: `head -1 tests/fixtures/*.nfl`
Expected: each fixture's first line starts with `# ` (a comment header).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/classifier.nfl tests/fixtures/tiny_mlp.nfl \
        tests/fixtures/pipeline_styles.nfl tests/fixtures/comments.nfl \
        tests/fixtures/mixed_args.nfl
git commit -m "test(fixtures): add 5 NFL v0.1 positive fixtures

classifier.nfl  — canonical demo (README example minus loss)
tiny_mlp.nfl    — smallest valid NFL program
pipeline_styles — three formattings of the same network
comments.nfl    — comments at every legal position
mixed_args.nfl  — operation with mixed positional + named args

Together exercises every production in language/grammar.ebnf.
Negative fixtures are deferred to Milestone 2 (when there is a
parser to reject them against)."
```

---

### Task 3: Create the human-readable reference doc

**Files:**
- Create: `docs/language_reference/grammar.md`

This is the longest single task. The structure follows spec §6 exactly. Length target: ~250-350 Markdown lines.

- [ ] **Step 1: Create `docs/language_reference/grammar.md` with this exact content**

```markdown
# NFL v0.1 — Language Reference

> **Status:** Defines NFL as of grammar v0.1 (Milestone 1).
> **Authoritative grammar:** [`language/grammar.ebnf`](../../language/grammar.ebnf).
> **Scope:** inference-only. Training syntax (loss, optimiser) is planned for v0.2.

This document is the human-facing companion to the formal EBNF grammar. Each section
follows the same top-down order as the grammar file. Every production has at least one
example. If this document and the grammar disagree, the grammar wins — file an issue
and we will reconcile.

---

## 1. Overview

NFL (NeuralForge Language) is a domain-specific language for describing neural networks
that compile ahead-of-time to assembly. v0.1 covers the **inference** path only:
declaring a model, its inputs, and the chain of operations that produces an output.

A complete v0.1 NFL file consists of one or more **model definitions**. Each model has
a name, parameters in square brackets, a typed input declaration, and a pipeline of
operations.

```nfl
model TinyMLP [batch=8]:
    x: Tensor[batch, 4]

    x -> linear[2] -> softmax
```

The output of the model is whatever the last operation in the pipeline produces — in
the example above, the softmax of a 2-element vector. This is **implicit**: the grammar
does not mark any expression as "the output"; the convention is "last operation of the
last pipeline_stmt in the model body".

---

## 2. Lexical structure

### 2.1 Identifiers

```ebnf
identifier = letter , { letter | digit | "_" } ;
```

An identifier starts with a letter (`a-z` or `A-Z`) and continues with letters, digits,
or underscores. Examples: `x`, `linear`, `Tensor`, `Classifier`, `model_a`, `Conv2D`.

**Stylistic guidance** (the grammar does not enforce these — a future linter will):

- `snake_case` for variables and operations: `x`, `hidden`, `linear`, `dropout`
- `PascalCase` for type names and model names: `Tensor`, `Classifier`, `Conv2D`

### 2.2 Numbers

```ebnf
integer = digit , { digit } ;
number  = integer , [ "." , integer ] ;
```

`512`, `0`, `42` are valid integers. `0.2`, `3.14` are valid floats. `5.` and `.5` are
**not** valid in v0.1 — both sides of the decimal point must have at least one digit.

No hexadecimal (`0x`), no exponent notation (`1e6`), no underscore separators (`1_000_000`)
in v0.1. Add them later if a use case appears.

### 2.3 Comments

A comment starts with `#` and runs to the end of the line. Comments are removed by the
lexer and do not appear in the parse tree.

```nfl
# This is a comment.
model X [batch=1]:        # Trailing comment on a model declaration.
    # Comment on its own line inside the model body.
    x: Tensor[batch, 1]
```

There are no block comments in v0.1.

### 2.4 Whitespace and indentation

NFL uses **significant indentation** (Python-style):

- A `:` at the end of a line opens a block. The next non-empty line must be indented
  by exactly 4 spaces more than the opening line.
- The block ends when a line returns to the opening line's indent (or shallower).
- Tabs in leading whitespace are a lexical error. Use spaces only.

Pipeline continuation lines (lines starting with `->` after the first step of a pipeline)
have a different rule — see §5.

Multiple consecutive blank lines are equivalent to one. They are visual separators only.

---

## 3. Top-level: model declarations

### 3.1 `model_def`

```ebnf
model_def    = "model" , identifier , "[" , model_params , "]" , ":" , newline
             , INDENT , model_body , DEDENT ;
```

A model definition begins with the keyword `model`, followed by the model's name (an
identifier, conventionally PascalCase), a parameter list in square brackets, and a
colon. The body is indented under the colon.

```nfl
model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu -> linear[output] -> softmax
```

A file may contain any number of model definitions (zero or more). Each is independent.

### 3.2 `model_params`

```ebnf
model_params = named_value , { "," , named_value } ;
named_value  = identifier , "=" , integer ;
```

The parameter list is one or more `name=integer` pairs separated by commas. Parameters
are identifiers bound to integer values; they can be referenced by name as symbolic
dimensions in `Tensor[…]` types and as integer arguments to operations.

```nfl
model A [batch=32]:                   # one parameter
model B [batch=32, input=784]:        # two parameters
model C [batch=32, input=784, output=10]:    # three parameters
```

`model_params` is non-empty: every model has at least one parameter. (A 0-parameter
model is not useful for v0.1's scope and is reserved for a future revision.)

---

## 4. Variable declarations and tensor types

### 4.1 `variable_decl`

```ebnf
variable_decl = identifier , ":" , type_expr ;
```

A variable declaration binds a name to a tensor type. In v0.1 every model needs at least
one such declaration: the input.

```nfl
x: Tensor[batch, 784]
```

### 4.2 `type_expr` and `dim_list`

```ebnf
type_expr = "Tensor" , "[" , dim_list , "]" ;
dim_list  = dim , { "," , dim } ;
dim       = integer | identifier ;
```

The only type in v0.1 is `Tensor[…]`, parameterised by a comma-separated list of one or
more dimensions. Each dimension is either an integer literal (`784`) or an identifier
(`batch`) that refers to a model parameter.

```nfl
Tensor[8, 4]          # all-integer dims
Tensor[batch, input]  # all-symbolic dims (must match model_params)
Tensor[batch, 64]     # mixed
```

**Symbolic-dim resolution is semantic**, not grammatical. The grammar accepts any
identifier here; a later semantic pass (Milestone 2/3) will check that each symbolic
dim refers to an existing model parameter.

`Tensor[]` (empty dim list) is invalid — every tensor has at least one dimension.

---

## 5. Pipelines

### 5.1 `pipeline_stmt`

```ebnf
pipeline_stmt  = identifier , pipeline_chain ;
pipeline_chain = pipeline_step , { pipeline_step } ;
pipeline_step  = "->" , operation ;
```

A pipeline statement starts with an identifier (the source — typically a previously-
declared variable) and is followed by one or more `-> operation` steps:

```nfl
x -> linear[10]                         # one step
x -> linear[10] -> relu                 # two steps
x -> linear[10] -> relu -> softmax      # three steps
```

The first identifier of a pipeline must reference a variable that was declared earlier
in the same model body. This is a **semantic** requirement, not enforced by the grammar.

### 5.2 Multi-line pipeline continuation

A line break inside a pipeline is permitted **if and only if** the next non-empty line
starts with `->` and has a leading-space count strictly greater than the indent of the
enclosing model body. Such continuation lines belong to the same pipeline statement and
do **not** open a new block (no INDENT/DEDENT is emitted for them).

```nfl
model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu          # model body indent = 4
      -> dropout[rate=0.2]            # continuation indent = 6 (any depth > 4 works)
      -> linear[256] -> relu
      -> linear[output] -> softmax
```

Mixing wrap styles (some steps inline, others wrapped) is allowed (see fixture
`tests/fixtures/pipeline_styles.nfl`).

### 5.3 Implicit model output

The value produced by the last operation of the **last** `pipeline_stmt` in a model
body is implicitly the model's output. There is no explicit `output:` declaration in
v0.1.

---

## 6. Operations and arguments

### 6.1 `operation`

```ebnf
operation = identifier , [ "[" , op_args , "]" ] ;
```

An operation is an identifier (the operation's name, conventionally snake_case),
optionally followed by a bracketed argument list. If the operation takes no arguments,
omit the brackets entirely — do not write empty brackets `[]`.

```nfl
relu                          # no arguments
linear[512]                   # one positional argument
dropout[rate=0.2]             # one named argument
linear[16, bias=true]         # mixed: positional then named
```

Operation names are identifiers; resolving them to actual operations (in the stdlib or
elsewhere) is a semantic step, not part of the grammar.

### 6.2 `op_args` — three valid forms

```ebnf
op_args         = positional_args , [ "," , named_args ]
                | named_args ;
positional_args = arg_value , { "," , arg_value } ;
named_args      = named_arg , { "," , named_arg } ;
named_arg       = identifier , "=" , arg_value ;
arg_value       = number | identifier ;
```

Three argument shapes are valid:

| Form | Example |
|---|---|
| Positional only | `op[a, b, c]` |
| Named only | `op[name=v]` or `op[a=1, b=2]` |
| Positional then named (in that order) | `op[a, name=v]` or `op[a, b, x=1, y=2]` |

**Named-then-positional is invalid** (`op[a=1, b]` will not parse). Positional arguments
must always come before named ones.

`arg_value` is either a number (`512`, `0.2`) or an identifier (`true`, `batch`).
Booleans-as-keywords (`true`, `false`) are just identifiers in v0.1; the stdlib decides
whether a given operation accepts them.

---

## 7. Implicit semantics (NOT enforced by the grammar)

These rules are documented here and as comments in `language/grammar.ebnf`. They are
checked by later compiler passes, not by the parser.

| Rule | Where it is checked |
|---|---|
| The output of a model is the last operation of the last `pipeline_stmt` | Convention; documented here |
| A symbolic dim in `Tensor[…]` must reference an existing `model_param` | Semantic analysis (M2/M3) |
| The first identifier of a `pipeline_stmt` must reference a previously-declared variable | Semantic analysis (M2) |
| Operation names must resolve against the stdlib | Stdlib resolution (M2/M3) |
| Operation argument count and types must match the operation's signature | Type checker (M3) |

---

## 8. What is intentionally absent in v0.1

These constructs are **not** part of NFL v0.1 and will not parse:

- **Training syntax**: no loss specification, no optimiser declaration, no training-loop
  hints. Planned for v0.2 as a coherent group.
- **Multi-output models**: a model body has effectively one pipeline (any further
  pipelines are syntactically permitted by the grammar but the implicit-output
  convention only treats the last one as the output).
- **Custom operations**: there is no syntax for declaring user-defined operations in
  NFL itself. v0.1 operations must come from the stdlib (M2+).
- **Control flow**: no conditionals, no loops at the NFL level.
- **String literals**: not present in v0.1.
- **Quantisation directives** (INT8/FP16/BF16): future versions.

The omissions are deliberate and informed by the brainstorming spec — see
`docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md` for rationale.

---

## 9. A complete annotated example

Here is the canonical fixture `tests/fixtures/classifier.nfl` walked line-by-line.

```nfl
# Canonical demo from README/PROJECT_SPEC.
# Exercises: model_def with three params, variable_decl with symbolic dims,
# multi-line pipeline_chain, operations with no/positional/named args.

model Classifier [batch=32, input=784, output=10]:
    x: Tensor[batch, input]

    x -> linear[512] -> relu
      -> dropout[rate=0.2]
      -> linear[256] -> relu
      -> linear[output] -> softmax
```

| Line(s) | What the grammar matches |
|---|---|
| Lines 1-3 | Three `comment` tokens, eaten by the lexer. |
| Blank line 4 | A `newline`. |
| Line 5 | Begins a `model_def`: keyword `model`, identifier `Classifier`, `[`, `model_params` (three `named_value`s `batch=32`, `input=784`, `output=10`), `]`, `:`, `newline`. The lexer emits `INDENT` for the next line. |
| Line 6 | A `model_stmt` of the `variable_decl` form: identifier `x`, `:`, `type_expr` `Tensor[batch, input]` (a `dim_list` of two symbolic `dim`s). |
| Blank line 7 | A `newline`; ignored as a separator. |
| Lines 8-11 | A single `pipeline_stmt`: identifier `x`, then a `pipeline_chain` of six `pipeline_step`s. Lines 9-11 are continuation lines (start with `->` at indent 6 > model_body indent 4); they belong to the same pipeline. The six steps are: `linear[512]` (positional), `relu` (no args), `dropout[rate=0.2]` (named), `linear[256]` (positional), `relu` (no args), `linear[output]` (positional with symbolic-dim ref), `softmax` (no args). |
| End of file | Lexer emits `DEDENT` to close the model body. |

The implicit output is the value produced by `softmax` — an unknown-shape tensor whose
true shape is determined by semantic analysis (which would resolve `output` to `10` and
therefore type the output as `Tensor[batch, 10]`).
```

- [ ] **Step 2: Sanity-check the reference doc**

Run: `wc -l docs/language_reference/grammar.md`
Expected: ~250-350 lines.

Run: `grep -c '^## ' docs/language_reference/grammar.md`
Expected: `9` — nine top-level sections.

Run: `grep -c '```ebnf' docs/language_reference/grammar.md`
Expected: `≥ 8` — every grammar production is shown in an `ebnf` code block at least once.

- [ ] **Step 3: Commit**

```bash
git add docs/language_reference/grammar.md
git commit -m "docs(language): add NFL v0.1 language reference

Human-readable companion to language/grammar.ebnf. Each grammar
production has a section with an EBNF excerpt and at least one
example. Includes the implicit-semantics list and a line-by-line
walkthrough of tests/fixtures/classifier.nfl."
```

---

### Task 4: Self-consistency review (verification pass)

This task does not (usually) produce a commit. Its purpose is to verify that the three
deliverables produced in Tasks 1-3 are mutually consistent and complete. If issues are
found, fix them inline and add a single follow-up commit at the end.

- [ ] **Step 1: Productions reachability check**

For each production defined in `language/grammar.ebnf`, confirm it is reachable from
`nfl_source` (directly or transitively). The full set of productions:

```
nfl_source        ←  ROOT
model_def         ←  used by nfl_source
model_params      ←  used by model_def
named_value       ←  used by model_params
model_body        ←  used by model_def
model_stmt        ←  used by model_body
variable_decl     ←  used by model_stmt
type_expr         ←  used by variable_decl
dim_list          ←  used by type_expr
dim               ←  used by dim_list
pipeline_stmt     ←  used by model_stmt
pipeline_chain    ←  used by pipeline_stmt
pipeline_step     ←  used by pipeline_chain
operation         ←  used by pipeline_step
op_args           ←  used by operation
positional_args   ←  used by op_args
named_args        ←  used by op_args
named_arg         ←  used by named_args
arg_value         ←  used by positional_args, named_arg
identifier        ←  used in many places
letter            ←  used by identifier
digit             ←  used by identifier, integer
integer           ←  used by named_value, dim, number
number            ←  used by arg_value
```

Confirm every production above is both **defined** in the file and **reachable**. If
any are missing, add them; if any are defined but unreachable, either remove or wire
them up.

- [ ] **Step 2: Reference-doc coverage check**

For each production above, confirm `docs/language_reference/grammar.md` mentions it
(typically inside an `ebnf` code block). Use:

```bash
for prod in nfl_source model_def model_params named_value model_body model_stmt \
            variable_decl type_expr dim_list dim pipeline_stmt pipeline_chain \
            pipeline_step operation op_args positional_args named_args named_arg \
            arg_value identifier integer number; do
  count=$(grep -c "$prod" docs/language_reference/grammar.md)
  echo "$count  $prod"
done
```

Every production should have count ≥ 1. (Lexical productions `letter` and `digit` are
acceptable to omit from the reference if they're covered in prose under "Identifiers"
and "Numbers" — exercise judgement.)

- [ ] **Step 3: Hand-trace each fixture against the grammar**

For each fixture, mentally walk top-down through the grammar productions and verify
the fixture matches. The classifier fixture is already walked through in
`docs/language_reference/grammar.md` §9 — confirm it's accurate. For the other four
fixtures, do the walk in your head (or on paper) and confirm:

| Fixture | What to confirm |
|---|---|
| `tiny_mlp.nfl` | Single `model_def`; one `variable_decl`; one `pipeline_stmt` with two steps; all dims are integer literals. |
| `pipeline_styles.nfl` | Three `model_def`s; each has the same `variable_decl` and a 4-step pipeline; the three differ only in pipeline-line wrapping. All three must accept under §5.2 continuation rule. |
| `comments.nfl` | Comments at every position do not affect parsing — every `#…` is a lexer-eaten comment. The model_body still parses as `variable_decl` then `pipeline_stmt`. |
| `mixed_args.nfl` | The `linear[16, bias=true]` operation must match `op_args = positional_args , "," , named_args`. `bias=true` is `named_arg = identifier "=" arg_value` where `arg_value = identifier` (`true` is an identifier here, not a keyword). |

If any fixture fails to trace cleanly, **the fixture is wrong** — fix the fixture, not
the grammar. (Grammar changes require returning to the spec process.)

- [ ] **Step 4: Cross-check grammar comments vs reference doc**

The preamble comment in `language/grammar.ebnf` lists 4 implicit-semantics rules. The
reference doc §7 should list the same set. Read both and reconcile any differences
(usually by extending the table in §7 — the grammar comment is the source of truth for
what is "implicit semantics").

- [ ] **Step 5: If any fixes were made, commit them**

```bash
git diff --stat
# If non-empty:
git add -p           # stage only the review-driven fixes
git commit -m "fix(m1): reconcile grammar/reference/fixtures after review"
```

If `git diff --stat` is empty, skip the commit.

---

### Task 5: Close out Milestone 1

**Files:**
- Modify: `DEVLOG.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a Milestone 1 close-out entry to `DEVLOG.md`**

The new entry goes **above** the existing 2026-05-02 brainstorming entry, separated by
the standard `---` line. Use the Edit tool: find the string

```
---

## 2026-05-02 — Brainstorming Milestone 1 (NFL Grammar v0.1); loss deferred to v0.2
```

and replace with:

```
---

## 2026-05-02 — Milestone 1 closed: NFL Grammar v0.1 shipped

### What was done
- Wrote `language/grammar.ebnf` (formal ISO/IEC 14977 grammar, inference-only)
- Wrote `docs/language_reference/grammar.md` (human-readable reference, one section per production)
- Added 5 positive fixtures under `tests/fixtures/`: `classifier.nfl`, `tiny_mlp.nfl`,
  `pipeline_styles.nfl`, `comments.nfl`, `mixed_args.nfl`
- Verified all artefacts by manual hand-tracing per the M1 acceptance criteria in the spec

### Decisions made
None new. All design decisions for M1 were captured during brainstorming on 2026-05-02 (entry below)
and recorded in `docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`. This session
executed the plan in `docs/superpowers/plans/2026-05-02-nfl-grammar-v0.1.md`.

### Problems encountered
- None. (If you found and fixed issues during the Task 4 review, list them here instead of "none".)

### Next step
Begin **Milestone 2 — Parser prototype**: implement a parser that consumes `.nfl` files and
produces a typed AST. The 5 fixtures from this milestone become the initial test corpus.
The choice of implementation language (Rust / C++ / Python / …) is the first decision of
M2 — to be resolved via a fresh `superpowers:brainstorming` cycle for M2.

---

## 2026-05-02 — Brainstorming Milestone 1 (NFL Grammar v0.1); loss deferred to v0.2
```

(Note: keep the *existing* brainstorming entry intact — only add the new close-out
entry above it. The Edit tool's old_string includes the brainstorming entry's heading
so the new entry is inserted just above it.)

- [ ] **Step 2: Update `CLAUDE.md` "Current Status"**

Find this section in `CLAUDE.md`:

```
## Current Status

Early design phase. Nothing is implemented yet.

The immediate next step is: **define the NFL grammar formally** (EBNF) and build a parser
prototype that handles a simple feedforward network definition.
```

Replace with:

```
## Current Status

Milestone 1 complete: NFL Grammar v0.1 (inference-only) is formally defined.
The artefacts are `language/grammar.ebnf`, `docs/language_reference/grammar.md`, and
five positive fixtures under `tests/fixtures/`.

The immediate next step is **Milestone 2 — Parser prototype**: implement a parser that
consumes `.nfl` files and emits a typed AST. The choice of implementation language
(Rust / C++ / Python / …) is the first M2 decision.
```

- [ ] **Step 3: Sanity-check the modifications**

Run: `git diff --stat`
Expected:
```
 CLAUDE.md  | 7 ++++---
 DEVLOG.md  | 24 ++++++++++++++++++++++++   (line counts may differ slightly)
```

Run: `head -25 DEVLOG.md`
Expected: see the new "Milestone 1 closed" entry as the topmost dated entry.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git commit -m "chore(m1): close Milestone 1 — NFL Grammar v0.1 shipped

Updates DEVLOG with the M1 close-out entry and CLAUDE.md's
Current Status to reflect M1 complete and M2 (Parser prototype)
as the next milestone."
```

---

## Done. What's next?

After Task 5, Milestone 1 is complete by the spec's acceptance criteria:

1. ✅ All 7 artefacts exist (Tasks 1, 2, 3)
2. ✅ Grammar internally consistent (Task 4 step 1)
3. ✅ Reference doc complete (Task 4 step 2)
4. ✅ Each fixture hand-traceable (Task 4 step 3, plus the §9 walkthrough for classifier)
5. ✅ DEVLOG entry added (Task 5 step 1)
6. ✅ CLAUDE.md "Current Status" updated (Task 5 step 2)

**Optional follow-up:** push commits and update PR [i1sme/NeuralForge#1](https://github.com/i1sme/NeuralForge/pull/1) — the existing PR will pick up these new commits. You may want to retitle the PR to reflect that it now ships M1 implementation and not just the spec, or open a new PR if your team's convention prefers separate spec/implementation PRs.

**The Milestone 2 entry-point** is a fresh `superpowers:brainstorming` cycle to design the parser — start with the question "implementation language for the compiler?" since that decision drives nearly everything in M2.
