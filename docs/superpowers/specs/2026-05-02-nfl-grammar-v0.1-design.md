# NFL Grammar v0.1 — Design Spec

> **Status:** Approved (brainstorming output, 2026-05-02)
> **Authoritative for:** Milestone 1 implementation
> **Source skill:** `superpowers:brainstorming`
> **Next skill:** `superpowers:writing-plans`

---

## 1. Context

NeuralForge is in pure design phase: directory structure exists, no code yet.
Per `PROJECT_SPEC.md`, the first concrete deliverable is **Milestone 1 — Language spec v0.1**:
a formal definition of NFL syntax in EBNF.

This spec captures the decisions made during the 2026-05-02 brainstorming session and
serves as the single source of truth for the Milestone 1 implementation plan that follows.

**Reading order for context:**
1. `CLAUDE.md` — project rules, design principles, documentation protocol
2. `PROJECT_SPEC.md` — full architecture and milestone roadmap
3. `DEVLOG.md` — chronological decisions (this spec corresponds to the 2026-05-02 entry
   "Brainstorming Milestone 1 (NFL Grammar v0.1); loss deferred to v0.2")

---

## 2. Scope

### In scope (v0.1)

- Formal EBNF grammar covering the **inference** (forward-pass) syntax of NFL
- Constructs needed for the canonical README example (modulo loss): `model` declaration with
  parameters, variable declarations with `Tensor[…]` types, pipeline operator `->`, operations
  with positional and named arguments, comments, multi-line pipeline continuation
- Human-readable language reference document with examples
- A small corpus of valid `.nfl` fixtures exercising every grammar production

### Out of scope (deferred)

- **Training syntax** (loss specification, optimisers, training-loop hints) — planned for **v0.2**.
  Reason: keeping `->` with a single, uniform meaning ("transform data through op")
  is more important than fitting one extra construct into v0.1; a coherent training-syntax
  design (loss + optimiser + training-loop hints together) is best done as one piece in v0.2.
- **Parser implementation** — Milestone 2.
- **Semantic analysis / type checking** — Milestone 2/3 (e.g. resolving symbolic dims,
  validating operation signatures against the stdlib).
- **Negative fixtures** (programs that should be rejected) — defer to Milestone 2 when there
  is a parser to actually test rejection against.
- **Multi-output models, control flow, custom operations, quantisation directives** — future versions.

---

## 3. Deliverables

| Path | Purpose |
|---|---|
| `language/grammar.ebnf` | Formal grammar in ISO/IEC 14977 EBNF |
| `docs/language_reference/grammar.md` | Human-readable reference: each construct explained with examples |
| `tests/fixtures/classifier.nfl` | Canonical demo (README example, sans loss) |
| `tests/fixtures/tiny_mlp.nfl` | Minimal valid NFL: one input, one linear, one softmax |
| `tests/fixtures/pipeline_styles.nfl` | Same network in three valid formattings (whitespace stress test) |
| `tests/fixtures/comments.nfl` | Comments at every legal position |
| `tests/fixtures/mixed_args.nfl` | Operation with mixed positional + named arguments |

---

## 4. Language design decisions for v0.1

| Aspect | Decision | Rationale |
|---|---|---|
| **Block structure** | Python-style: `:` opens, indented body, dedent closes | Matches README example; token-efficient |
| **Indent unit** | 4 spaces; tabs forbidden in leading whitespace | Avoids tabs-vs-spaces ambiguity; LLM-stable |
| **Line comments** | `#` to end of line; no block comments | Matches Python aesthetic; minimal lexer |
| **Identifier casing (style)** | `snake_case` for variables/operations, `PascalCase` for types/models | Stylistic guidance only — grammar accepts any `[A-Za-z][A-Za-z0-9_]*`; linter (future) enforces case |
| **Numeric literals** | Integer (`512`) and float (`0.2`); no hex, no exponent, no `_` separators | YAGNI |
| **String literals** | Not present in v0.1 | No use case yet |
| **Operation arg syntax** | `op[positional, named=value]`; bare `op` when there are no args | Matches README; no empty `[]` noise |
| **Pipeline operator** | `->` between expressions; multi-line continuation by starting next line with `->` at deeper indent | Matches README |
| **`Tensor` type** | `Tensor[dim, dim, …]` where `dim` is integer literal or identifier | Symbolic dims reference `model_params` (resolved semantically, not by grammar) |
| **Implicit model output** | The value of the last operation of the last `pipeline_stmt` IS the model's output | Documented in reference doc; not enforced by grammar |
| **Training syntax (loss / optimiser)** | Excluded from v0.1; deferred to v0.2 | Keeps `->` with one meaning; v0.2 designs training holistically |
| **EBNF dialect** | ISO/IEC 14977 | Standard, widely understood, parser-tooling-agnostic; comments via `(* ... *)` |

---

## 5. Grammar (full EBNF)

This is the exact content to be placed in `language/grammar.ebnf`:

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

---

## 6. Reference doc structure (`docs/language_reference/grammar.md`)

The reference is the human-facing companion to `grammar.ebnf`. It follows the same
top-down order. For each grammar production it provides: what the construct is, an
EBNF excerpt, and at least one example. Sections:

1. **Overview** — what NFL is, what v0.1 covers, how to read this doc
2. **Lexical structure** — identifiers, numbers, comments, whitespace, indent rules
3. **Top-level: model declarations** — `model_def`, `model_params`, with examples
4. **Variable declarations and tensor types** — `variable_decl`, `type_expr`, symbolic dims
5. **Pipelines** — `pipeline_stmt`, the `->` operator, multi-line continuation
6. **Operations and arguments** — `operation`, positional / named / mixed forms
7. **Implicit semantics** — what is NOT enforced by the grammar (output convention,
   symbolic-dim resolution, stdlib resolution)
8. **What is intentionally absent in v0.1** — pointer to "Out of scope" with reasons
9. **A complete annotated example** — `classifier.nfl` walked line-by-line

Length target: ~150-250 lines of Markdown. Each section ≤ ~30 lines.

---

## 7. Fixtures

All fixtures are valid NFL programs. Each begins with a `#` comment block stating the
fixture's role.

### `tests/fixtures/classifier.nfl`

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

### `tests/fixtures/tiny_mlp.nfl`

```nfl
# Minimal valid NFL: one model_param, one variable_decl with all-integer dims,
# one pipeline_stmt with two ops.

model TinyMLP [batch=8]:
    x: Tensor[batch, 4]

    x -> linear[2] -> softmax
```

### `tests/fixtures/pipeline_styles.nfl`

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

### `tests/fixtures/comments.nfl`

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

### `tests/fixtures/mixed_args.nfl`

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

> **Note on `bias=true`:** `true` is just an identifier here per the grammar (`arg_value = number | identifier`).
> Booleans-as-keywords are a v0.2 question; for v0.1 the grammar treats them as identifiers and
> the stdlib (when designed in M2) decides whether `bias` accepts the identifier `true`.

---

## 8. Acceptance criteria for Milestone 1

Milestone 1 is **complete** when all of the following hold:

1. **All seven artefacts from Section 3 exist** and are non-empty
2. **Grammar is internally consistent:**
   - Every referenced production is defined
   - Every defined production is reachable from `nfl_source`
   - No production contradicts the textual description in the reference doc
3. **Reference doc is complete:**
   - Each EBNF production has a corresponding section with at least one example
   - Implicit-semantics list (output, symbolic-dim resolution, stdlib resolution, pipeline-source binding) is present and matches the grammar's preamble comment
4. **Each fixture can be hand-traced through the grammar.** A short walkthrough for
   `classifier.nfl` is included in the reference doc; the rest are reviewed but not
   walked through inline
5. **`DEVLOG.md`** has a closing entry for Milestone 1 referencing this spec
6. **`CLAUDE.md` "Current Status"** is updated to reflect Milestone 1 complete and Milestone 2
   (Parser prototype) as the next step

The "real" end-to-end test — feeding fixtures through a parser — is intentionally a
**Milestone 2** acceptance criterion, not Milestone 1.

---

## 9. Deferred items and open questions

### Deferred to Milestone 2
- Negative fixtures (programs that should be rejected by the parser)
- Automated parser-driven validation of all positive fixtures
- Semantic resolution: model_param references in `Tensor[…]`, operation-name binding to stdlib
- Type checker: `linear[N]` requires integer >0; `dropout[rate=…]` requires float ∈ [0,1]; etc.

### Deferred to NFL v0.2
- Training syntax: loss specification, optimiser declaration, training-loop hints
- Multi-output models (multiple `pipeline_stmt`s in one body)
- Custom operations (user-defined ops in NFL itself)
- Control flow (conditional sub-graphs)
- Quantisation directives (INT8, FP16, BF16)

### Open questions (non-blocking for M1)
- None. All blocking questions for v0.1 grammar were resolved during brainstorming.

---

## 10. Transition

After this spec is reviewed and approved by the user, transition to the
`superpowers:writing-plans` skill to produce a step-by-step implementation plan
covering all seven artefacts above. Implementation itself is the subject of a
later `superpowers:executing-plans` cycle.
