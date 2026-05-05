# NFL Universal IR (UIR) — Language Reference

> **Status:** Defines the UIR as of NFL v0.1 (Milestone 6 complete).
> **Authoritative source:** `compiler/src/ir/` and the M3a/M3b/M3c/M5a/M6 specs under
> `docs/superpowers/specs/`.

This document explains what the UIR is, how it's structured, and the rules the
builder enforces when constructing it from the AST. If this doc and the source
disagree, the source wins — file an issue.

---

## 1. Overview

The Universal IR is a typed computation graph that the NFL compiler produces
between parsing and codegen. It is the input to architecture profiles
(`profiles/arm64/` is the first concrete one, M4+) and to optimisation passes
(`compiler::passes::default_pipeline()` runs `EliminateDropout` then
`FuseLinearRelu` then `FuseLinearSoftmax`, M5+/M6+).

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
    Op {
        op: StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
        // M5a+: post-ops fused into this op's emitter. Set by:
        //   FuseLinearRelu → vec![PostOp::Relu] (linear→relu)
        //   FuseLinearSoftmax → vec![PostOp::SoftmaxRow] (linear→softmax, M6+)
        // Empty for un-fused nodes.
        fused_post_ops: Vec<PostOp>,
    },
}
```

**Why index-based?** Easy to clone, easy to traverse (just iterate `nodes`),
easy to share subexpressions (multiple nodes can reference the same `NodeId`).
Standard compiler-textbook choice. UIR-passes (M5+) take an immutable `&Uir`
and return a fresh `Uir` with the transformation applied — see §7 below.

**Why immutable?** The builder never modifies a node after pushing it. UIR-passes
preserve the immutability contract: each pass returns a freshly-numbered `Uir`
(NodeIds renumbered 0..N), with operands and `model.inputs`/`model.output`
remapped through an internal id_map. Consumers can hold a `&Uir` reference
across multiple passes by re-binding through `run_pipeline`'s output. No
in-place mutation; no tombstones; no stale-NodeId hazards.

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

`NodeKind::Op { op, operands, attrs, fused_post_ops }` — applies an operation
from the stdlib.
- `op` is a `StdOp` enum variant (resolved from the AST identifier).
- `operands` are `NodeId`s referencing the inputs (one for v0.1's single-input ops).
- `attrs` are validated, type-resolved arguments (positional and named, in the
  signature's slot order).
- `fused_post_ops: Vec<PostOp>` — post-operations folded into this
  node by passes such as `FuseLinearRelu` (`PostOp::Relu`) and
  `FuseLinearSoftmax` (`PostOp::SoftmaxRow`). Empty by default.
  `Display` rendering: lowercase snake_case (`relu`, `softmax_row`).
  Renders as `fused=[<list>]` suffix in the CLI pretty-print (§6).

Example: `linear[16, bias=true]` becomes:
```
n1: linear           :: Tensor[8, 16]    operands=[n0]    attrs=[out_dim=16, bias=true]
```

---

## 4. Stdlib operations (v0.1)

Four operations are recognised:

| StdOp     | Signature                                                                | Output shape                       |
|-----------|--------------------------------------------------------------------------|-------------------------------------|
| `Linear`  | positional `out_dim: Integer` (required), named `bias: Symbol` (optional) | `Tensor[input.batch, out_dim]`     |
| `Relu`    | no args                                                                  | input shape (elementwise)          |
| `Dropout` | named `rate: Float` (required, must be `0..=1`)                          | input shape (elementwise)          |
| `Softmax` | no args                                                                  | input shape (elementwise)          |

Adding a new op = new `StdOp` variant + new arms in `signature()` and
`infer_output_shape()` in `compiler/src/ir/stdlib.rs`.

### Codegen interpretation of optional attributes

NFL grammar marks some op arguments as optional (e.g. `Linear`'s `bias`).
Default behaviour is **codegen-profile-specific**: profiles document how they
treat absent optional attributes. The current arm64 profile (M4b) interprets
`linear[N]` without an explicit `bias` attribute as **no bias add** (pure
matmul). To get bias, write `linear[N, bias=true]` explicitly. See
[`docs/profile_guide/arm64.md`](../profile_guide/arm64.md) §3 + §4.3 for the
exact codegen patterns.

### Dropout at inference

NFL v0.1 is inference-only and `dropout` behaves as **identity** at run time
(no random masking). Codegen profiles implement this by aliasing the dropout
node's output buffer to its operand's, emitting no asm. See
[`docs/profile_guide/arm64.md`](../profile_guide/arm64.md) §4.5 for the
arm64-profile-specific implementation.

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
- **Mutation API.** `Uir` is immutable-by-construction. M5+ UIR-passes
  preserve this — each pass produces a fresh `Uir` (NodeIds renumbered
  0..N, references remapped), not in-place edits. See
  `compiler::passes::run_pipeline` and the per-pass doc-comments in
  `compiler/src/passes/`.
- **Profile-specific lowering.** All profile work is M4+.
- **Multi-error reporting.** First error halts the build. v0.2.
- **Source-snippet errors with multi-line context, color, or labels.** M3c's
  hand-rolled formatter is single-line, monochrome. v0.2+ may upgrade.
- **Custom operations.** No syntax for declaring user-defined ops. v0.2+.
- **Training syntax** (loss, optimiser). NFL v0.1 is inference-only.
