# NFL v0.1 / v0.2 — Language Reference

> **Status:** Defines NFL through grammar v0.2 (Milestone 10).
> **Authoritative grammar:** [`language/grammar.ebnf`](../../language/grammar.ebnf).
> **Scope:** inference-only. Training syntax (loss, optimiser) is still planned for a
> future grammar revision; M10's v0.2 added named pipelines (§5.4) and tensor-typed
> positional arguments (§5.4 / §6.2) for self-attention patterns.

This document is the human-facing companion to the formal EBNF grammar. Each section
follows the same top-down order as the grammar file. Every production has at least one
example. If this document and the grammar disagree, the grammar wins — file an issue
and we will reconcile.

---

## 1. Overview

NFL (NeuralForge Language) is a domain-specific language for describing neural networks
that compile ahead-of-time to assembly. v0.1 covers the **inference** path only:
declaring a model, its inputs, and the chain of operations that produces an output.

A complete v0.1 NFL file (the grammar's root production `nfl_source`) consists of zero
or more **model definitions**, separated by optional blank lines. Each model has a name,
parameters in square brackets, a typed input declaration, and a pipeline of operations.

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
model A [batch=32]:                          # one parameter
model B [batch=32, input=784]:               # two parameters
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

In v0.2 (M10) the same rule applies to a `named_pipeline_stmt` — see §5.4: the last
statement's right-hand-side bound value (rather than the original source identifier)
is the model's output.

### 5.4 Named pipelines (v0.2, M10)

```ebnf
named_pipeline_stmt = identifier , ":" , type_expr , "=" , identifier , pipeline_chain ;
```

A **named pipeline** binds a pipeline result to a named, declared-shape variable so it
can be referenced by later statements in the same model body. Unlike `pipeline_stmt`,
which is anonymous (its value is only addressable as the implicit model output), a
`named_pipeline_stmt` produces an in-scope identifier that subsequent statements may
read — exactly the affordance attention patterns need (Q/K/V projections, attention
scores, attention output) without inventing a new construct.

```nfl
q: Tensor[8, 64]   = x -> linear[64]
k: Tensor[8, 64]   = x -> linear[64]
scores: Tensor[8, 8] = q -> matmul[k, transpose_b=true]
```

#### Lookahead disambiguation

`variable_decl` and `named_pipeline_stmt` share the same first three tokens
(`identifier`, `":"`, `type_expr`). The parser disambiguates with one extra token of
lookahead **after** the closing `]` of the `type_expr`:

| Next token | Production |
|---|---|
| `=` | `named_pipeline_stmt` |
| anything else (typically `newline`) | `variable_decl` |

If the parser sees `=`, it consumes it, expects an `identifier` (the source), and then
parses a `pipeline_chain` exactly as in §5.1. Otherwise the statement is closed as a
plain `variable_decl`.

#### Two examples

A 2D classifier with named intermediates:

```nfl
model NamedClassifier [batch=8, input=4, hidden=16, output=2]:
    x: Tensor[batch, input]

    h: Tensor[batch, hidden] = x -> linear[hidden] -> relu
    h -> linear[output] -> softmax
```

A 4D self-attention sketch (Q/K/V over `[batch, heads, seq, dim]`):

```nfl
model SelfAttention [batch=2, heads=4, seq=8, dim=16, scale_int=4]:
    x: Tensor[batch, heads, seq, dim]

    q: Tensor[batch, heads, seq, dim] = x -> linear[dim]
    k: Tensor[batch, heads, seq, dim] = x -> linear[dim]
    v: Tensor[batch, heads, seq, dim] = x -> linear[dim]
    scores: Tensor[batch, heads, seq, seq] = q -> matmul[k, transpose_b=true]
    scaled: Tensor[batch, heads, seq, seq] = scores -> mul_scalar[scale_int]
    attn:   Tensor[batch, heads, seq, seq] = scaled -> softmax
    attn -> matmul[v]
```

#### Tensor-typed positional arguments

Note the `matmul[k, transpose_b=true]` and `matmul[v]` operations above. The first
positional argument is an `identifier`, not a number — the grammar already accepts
this via `arg_value = number | identifier` (see §6.2). What is new in v0.2 is that
the stdlib operation `matmul` declares its first positional parameter as a `Tensor`
rather than an integer. Resolution from "identifier-token" to "the actual tensor
node in the UIR graph" happens at UIR-build time (`ArgType::Tensor` resolution); the
grammar itself is unchanged.

#### Output-rule generalisation

The §5.3 implicit-output rule generalises naturally: the last statement of a model
body is the implicit output regardless of whether it is a `pipeline_stmt` (anonymous,
output is the chain's terminal value) or a `named_pipeline_stmt` (output is the
right-hand-side bound value). A model body may freely intermix the two forms.

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

In v0.2 (M10) `arg_value = identifier` also covers tensor-typed positional arguments —
e.g. the second matmul operand in `matmul[k, transpose_b=true]`. The grammar is
unchanged; the `ArgType::Tensor` machinery in the UIR builder resolves the identifier
to the matching `NodeId` at build time. See §5.4 for the language-level perspective.

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
- **Multi-output models**: a model body has effectively one output — the implicit-
  output convention only treats the **last** statement (whether `pipeline_stmt` or
  `named_pipeline_stmt`) as the model's output. Earlier `named_pipeline_stmt` bindings
  are in scope for *internal* references but are not externally exposed.
- **Custom operations**: there is no syntax for declaring user-defined operations in
  NFL itself. v0.1 operations must come from the stdlib (M2+).
- **Control flow**: no conditionals, no loops at the NFL level.
- **String literals**: not present in v0.1.
- **Quantisation directives** (INT8/FP16/BF16): future versions.

The omissions are deliberate and informed by the brainstorming spec — see
[`docs/superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md`](../superpowers/specs/2026-05-02-nfl-grammar-v0.1-design.md)
for rationale.

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
| Lines 8-11 | A single `pipeline_stmt`: identifier `x`, then a `pipeline_chain` of seven `pipeline_step`s. Lines 9-11 are continuation lines (start with `->` at indent 6 > model_body indent 4); they belong to the same pipeline. The seven steps are: `linear[512]` (positional), `relu` (no args), `dropout[rate=0.2]` (named), `linear[256]` (positional), `relu` (no args), `linear[output]` (positional with symbolic-dim ref), `softmax` (no args). |
| End of file | Lexer emits `DEDENT` to close the model body. |

The implicit output is the value produced by `softmax` — an unknown-shape tensor whose
true shape is determined by semantic analysis (which would resolve `output` to `10` and
therefore type the output as `Tensor[batch, 10]`).
