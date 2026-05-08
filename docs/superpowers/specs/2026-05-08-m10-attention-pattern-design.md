# Milestone 10 — NFL v0.2 Self-Attention + 4D Codegen — Design

> Brainstormed: 2026-05-08
> Strategic axis: **Axis 2 — modelling depth** (Strategic Roadmap, `PROJECT_SPEC.md`)
> Predecessor: M9 (x86_64 Linux ELF profile + `profile-api` contract)
> Status: spec draft for plan synthesis

---

## 1. Overview

M10 introduces self-attention as a first-class workload in NFL by adding
the minimum-viable set of language and codegen primitives required to
express and execute a self-attention block end-to-end on both arm64
(macOS Mach-O) and x86_64 (Linux ELF) profiles. The acceptance artefact
is a single NFL fixture whose compiled output passes per-profile
bit-exact comparison against an architecture-matched reference
implementation.

The strategic claim being validated is **vertical stack-isolation**: a
new neural-network construction (attention-pattern operations) extends
each compiler layer (lexer, parser, AST, UIR, both profile codegens)
through localised, type-level changes — without leaking concerns
between layers. M9 proved horizontal isolation (two profiles, one UIR);
M10 is the first milestone that *uses* that isolation as a foundation
rather than validating it.

The companion non-deliverable is the OQ-BENCH harness, landed as a
**separate single-day commit before M10 implementation begins**. It
formally closes the OQ-BENCH trigger opened by M9, produces baseline
scalar-FFN timings on existing fixtures, and seeds infrastructure for
post-M10 cross-profile numerical reports. OQ-BENCH does *not* extend
M10's scope.

Strategically, M10 promotes NeuralForge from "feedforward MLP compiler"
to "compiler that understands attention" — the minimum credibility
threshold for a project claiming to be a neural network compiler in
2026.

---

## 2. Goals

Ship a single PR with ~12 atomic commits (final count delegated to
`writing-plans`) that together:

1. Extend NFL v0.1 grammar with `named_pipeline_stmt` (typed
   intermediate-tensor binding via `name : Type = source -> chain`).
2. Introduce `ArgType::Tensor` and extend `resolve_args` to return
   `(Vec<NodeId>, Vec<OpAttr>)`, enabling tensor-by-name op arguments.
3. Add two new UIR `StdOp` variants — `Matmul` (with `transpose_b`
   attribute) and `MulScalar` (with `value` attribute) — with shape
   inference covering arbitrary rank-≥-2 inputs and four new
   `ShapeError` variants.
4. Generalise `Softmax` shape inference to require rank ≥ 2;
   profile-side dispatch flattens leading dims into the row count.
5. Add `BuildErrorKind::DeclaredShapeMismatch { declared, actual }`
   for `named_pipeline_stmt` declared-vs-actual shape verification.
6. Implement `emit_matmul` and `emit_mulscalar` in both
   `profiles/arm64/` and `profiles/x86_64/`, leaving `emit_linear`
   and the `emit_softmax` asm emitter unchanged.
7. Land acceptance fixture `tests/fixtures/self_attention.nfl` with
   `[batch=2, heads=4, seq=16, head_dim=16]` shape and per-profile
   FFI integration tests against architecture-matched references.
8. Add four negative parser/builder fixtures plus ~38 unit tests
   across parser, UIR, codegen, and integration layers (~43 total
   new tests; project total 284 → ~327).
9. Update `language/grammar.ebnf`, `docs/language_reference/grammar.md`,
   `docs/profile_guide/{arm64,x86_64}.md`, `PROJECT_SPEC.md` (First
   Milestones table + Current Status + Strategic Roadmap revision),
   `DEVLOG.md`, and `CLAUDE.md` Repository Structure tree.

The PR closes M10. No follow-up PR is required for any in-scope
concern.

---

## 3. Strategic Positioning

### 3.1 What M10 Proves

- **Stack-isolation, vertically.** New ops extend each compiler layer
  through localised, type-level changes: lexer is unchanged; parser
  gains one new production with one-token lookahead; AST gains one
  variant; UIR gains two `StdOp` variants and one `ArgType` variant;
  each profile gains two new emitter files. No layer leaks
  responsibilities into another.
- **Profile-isolation as foundation, not as hypothesis.** M9
  validated that two profiles can independently implement the same
  UIR contract. M10 *uses* that contract — both profiles
  independently implement Matmul and MulScalar, with no shared
  state, no cross-profile signalling, no back-channels. This is the
  first milestone since M9 to exercise the trait surface from
  multiple consumers simultaneously.
- **ABI invariant `(input, params, output)` preserved.** New ops
  are added without disturbing the three-pointer FFI calling
  convention. SelfAttention has zero learnable parameters in M10
  (no learnable Q/K/V projections); the FFI harness passes a
  non-null aligned pointer to a zero-length params buffer, which
  the assembly never dereferences (Section 7.4).
- **NeuralForge is an NN compiler, not an MLP compiler.** Attention
  over tokens compiles and executes — the minimum credibility
  threshold for the project category in 2026.

### 3.2 What M10 Does Not Prove

- **Performance competitiveness.** No SIMD on either profile; no
  fusion of attention-internal ops. Benchmarking is OQ-BENCH
  territory.
- **Full NFL v0.2 grammar.** No reshape op, no multi-input model,
  no expression syntax, no general broadcasting. Only the
  primitives required to express self-attention.
- **A complete transformer block.** Residual, layer-norm, FFN are
  not in scope; SelfAttention is the smallest meaningful attention
  construct, not a full block.
- **Cross-profile bit-exact equality at the byte level.** M9
  deliberately diverged FMA usage (arm64 `fmadd` single-rounding,
  x86_64 `mulss + addss` two-rounding); combined with libm `expf`
  divergence (Apple libsystem vs glibc), cross-profile bit-exact
  is architecturally unreachable without harmonisation work
  outside M10. Acceptance is **per-profile** bit-exact against an
  architecture-matched reference; cross-profile numerical agreement
  is measured with tolerance in OQ-BENCH (informational, not a
  gate).

---

## 4. NFL Grammar Extensions

### 4.1 New Production: `named_pipeline_stmt`

```ebnf
model_stmt          = variable_decl | pipeline_stmt | named_pipeline_stmt ;
named_pipeline_stmt = identifier , ":" , type_expr , "=" , identifier , pipeline_chain ;
```

The new production binds the result of a pipeline to a typed name.
After execution, the bound name is available as an identifier in
subsequent statements.

**Parser dispatch — explicit one-token lookahead requirement.** The
parser MUST distinguish `variable_decl` (`identifier ":" type_expr`
with no `=`) from `named_pipeline_stmt` (`identifier ":" type_expr
"=" ...`) using one-token lookahead on `=` after the type expression.
`named_pipeline_stmt` is a *separate production* in the grammar and
a *separate AST variant* — it is NOT a branch inside `variable_decl`
parsing. Folding the two into a single parsing function with a
conditional `=` branch produces a tangled AST and obscures the
language structure. The parser unit test
`parse_lookahead_distinguishes_variable_decl_from_named_pipeline`
(Section 8.1) guards against the merged form.

### 4.2 Output Rule (Explicit)

> The output of a model is the value produced by the last operation
> of the last `model_stmt` in its body. If the last `model_stmt` is
> a `named_pipeline_stmt`, the output is the bound right-hand-side
> value (i.e., the final operation's result of its `pipeline_chain`).

This rule generalises the v0.1 wording "last operation of the last
`pipeline_stmt`" to cover named bindings.

### 4.3 Tensor-By-Name Op Arguments (No Grammar Change)

The existing rule `arg_value = number | identifier` already permits
identifiers as op arguments. M10 introduces stdlib-side semantics:
an op that declares a `Tensor`-typed slot expects its arg to resolve
against the previously-declared variable environment.

Example:
```nfl
scores: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
```

- `x` (positional arg of `matmul`) — resolved as previously-declared
  variable name; passed as second operand.
- `transpose_b=true` (named arg) — `true` parses as
  `ArgValue::Symbol("true")` per existing grammar; stdlib interprets
  the symbol as a boolean (parallel to the existing `linear_has_bias`
  pattern for `bias=true`).

### 4.4 What Is Not Added

- Infix operators (`@`, `*`, `.T`).
- Expression syntax of any kind.
- Multi-input model declarations (separate `q`, `k`, `v` as named
  inputs).
- Boolean literals as a grammar token (handled at stdlib
  interpretation level).
- Tensor reshape op.

Rationale: design principle #4 ("Regular grammar, no exceptions").
Pipeline-as-canonical-form remains the single way to express
computation; named bindings are a strictly additive extension that
does not introduce a parallel expression grammar.

---

## 5. UIR Extensions

### 5.1 New `StdOp` Variants

```rust
#[non_exhaustive]
pub enum StdOp {
    // existing:
    Linear, Relu, Dropout, Softmax,
    // new in M10:
    Matmul,
    MulScalar,
}
```

`StdOp` remains `Copy`. Variant-specific data lives in the existing
`attrs: Vec<OpAttr>` field on `NodeKind::Op`, consistent with how
`Linear`'s `out_dim` is carried.

### 5.2 New `ArgType::Tensor`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    Integer, Float, Symbol,
    Tensor,  // new in M10
}
```

A `Tensor`-typed slot expects an `ArgValue::Symbol(name)` that
resolves against the variable environment to a `NodeId`. The
resolved `NodeId` is included in the operation's
`operands: Vec<NodeId>`; it does NOT appear in the operation's
`attrs`.

### 5.3 `resolve_args` Signature Change (Breaking Internal API)

```rust
// before:
fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    params: &HashMap<&str, u64>,
    op_span: Span,
) -> Result<Vec<OpAttr>, BuildError>;

// after:
fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    params: &HashMap<&str, u64>,
    env: &HashMap<String, NodeId>,    // new in M10 — for Tensor arg resolution
    op_span: Span,
) -> Result<(Vec<NodeId>, Vec<OpAttr>), BuildError>;
```

The first component of the returned tuple contains all
`Tensor`-resolved operand IDs in declaration order; the second
contains scalar/identifier attrs as before. The new `env` parameter
maps variable names to `NodeId`s (the same `env` already maintained
by `build_model`).

**Implementer note (load-bearing):** all current callsites of
`resolve_args` (currently only `build_op` in
`compiler/src/ir/build.rs`) MUST be updated atomically with the
signature change. A missed callsite produces a compile error
without context. The atomic-task-pack convention from M7 applies —
the signature change and its callsite updates land in one commit.

### 5.4 New Op Signatures

```rust
StdOp::Matmul => Signature {
    positional: &[ArgSlot {
        name: "other",
        ty: ArgType::Tensor,
        required: true,
    }],
    named: &[ArgSlot {
        name: "transpose_b",
        ty: ArgType::Symbol,
        required: false,
    }],
},
StdOp::MulScalar => Signature {
    positional: &[ArgSlot {
        name: "value",
        ty: ArgType::Float,
        required: true,
    }],
    named: &[],
},
```

Helper for `transpose_b` reads parallels existing `linear_has_bias`:

```rust
pub fn matmul_transpose_b(attrs: &[OpAttr]) -> bool {
    attrs.iter().any(|a|
        a.name == "transpose_b"
        && matches!(&a.value, AttrValue::Symbol(s) if s == "true")
    )
}
```

### 5.5 Shape Inference

#### 5.5.1 Matmul

```
inputs[0] = a, shape [..., M, K_a]
inputs[1] = b, shape [..., K_b, N]   if !transpose_b
            or [..., N, K_b]         if transpose_b
output       shape [..., M, N]
```

Validation steps (in order):

1. `inputs.len() == 2` else `WrongInputCount`.
2. `a.rank() == b.rank()` else `RankMismatch`.
3. `a.rank() >= 2` else `RankTooLow { required: 2, actual: a.rank() }`.
4. Leading dims (indices `0..rank-2`) match exactly between a and b
   — no broadcasting (`LeadingDimMismatch { dim_index, lhs, rhs }`).
5. `K_a == K_b` else `InnerDimMismatch { lhs_k, rhs_k, transpose_b }`.

Output shape: `[..., M, N]` where leading dims are inherited from
`a`.

**Strict no-broadcasting** is a principled position, not a
limitation. Per design principle #1 ("Nothing is inferred silently"),
implicit shape transformations are excluded.

#### 5.5.2 MulScalar

Single input (else `WrongInputCount`). Output shape == input shape.

#### 5.5.3 Softmax (Updated)

Input shape rank check tightened: rank ≥ 2 required (else
`RankTooLow`). This change is fail-fast in UIR rather than fail-late
in lowering — each profile previously embedded the same implicit
assumption. 1D softmax is mathematically valid but excluded by
project convention: all practical NFL use cases are 2D and 4D
batch-first; rank ≥ 2 reflects "batch dimension required."

Output shape == input shape (unchanged).

### 5.6 `BuildErrorKind` Additions

```rust
pub enum BuildErrorKind {
    // existing variants ...
    DeclaredShapeMismatch { declared: Shape, actual: Shape },
}
```

Constructor: `BuildError::declared_shape_mismatch(declared, actual, span)`.

This variant is *additional to* (not replacing) the existing
`ShapeMismatch { detail: String }` (used by `BuildError::shape`).
The new variant is structural — it carries the two `Shape` values —
for the specific case of `named_pipeline_stmt` declared-vs-actual
verification. Generic shape-related errors continue to use the
existing `ShapeMismatch` variant.

All `BuildError` variants already carry span via the flat
`line: u32, col: u32` fields on the outer struct;
`DeclaredShapeMismatch` follows the same pattern (no precedent
created — span is universal).

### 5.7 `ShapeError` Additions

```rust
pub enum ShapeError {
    // existing variants ...
    RankMismatch { lhs: usize, rhs: usize },
    RankTooLow { required: usize, actual: usize },
    LeadingDimMismatch { dim_index: usize, lhs: u64, rhs: u64 },
    InnerDimMismatch { lhs_k: u64, rhs_k: u64, transpose_b: bool },
}
```

### 5.8 `named_pipeline_stmt` Builder Logic

In `compiler/src/ir/build.rs::build_model`, a new arm in the
`model_stmt` match:

```rust
ModelStmt::NamedPipeline(np) => {
    // 1. Build pipeline as for an ordinary pipeline_stmt.
    let mut current = *env.get(&np.source)
        .ok_or_else(|| BuildError::unknown_variable(&np.source, np.span))?;
    for op_ast in &np.steps {
        let input_shape = nodes[current].ty.shape.clone();
        current = build_op(op_ast, current, &input_shape, &params, &mut nodes)?;
    }
    // 2. Verify declared shape against actual.
    let declared = resolve_type(&np.declared_ty, &params)?;
    let actual = nodes[current].ty.shape.clone();
    if declared != actual {
        return Err(BuildError::declared_shape_mismatch(
            declared, actual, np.span,
        ));
    }
    // 3. Bind name in env.
    env.insert(np.binding_name.clone(), current);
    // 4. Update last_pipeline_output.
    last_pipeline_output = Some(current);
}
```

Output rule (Section 4.2) is preserved automatically:
`last_pipeline_output` continues to denote the model's output.

### 5.9 `calls_extern_math` Predicate

Unchanged. `Matmul` and `MulScalar` are pure scalar arithmetic;
neither calls libm. `Softmax` remains in the predicate's positive
set; 4D softmax still calls `_expf` per element (just over more
rows).

### 5.10 Existing Fusion Passes

`FuseLinearRelu` and `FuseLinearSoftmax` are unchanged. They operate
on `Linear` (which remains 2D-only via `require_rank(input, 2)`).
SelfAttention contains no `Linear` ops, so neither pass fires on the
attention path. This is intentional — fusion of attention-internal
patterns (Matmul → MulScalar → Softmax → Matmul) is explicitly
deferred (Section 9).

---

## 6. Lowering Plan

### 6.1 Architectural Invariant: New Files Only

`emit_linear` (both profiles) is **unchanged**. The accumulated
M3-M8 layering — bias-add, fused `PostOp::Relu`, fused
`PostOp::SoftmaxRow` row-wise tail — is preserved exactly as-is.
The cost of generalising `emit_linear` to handle
activation-as-second-operand and absent-bias cases (the kind of
branching conditional logic this project deliberately avoids) is
paid by separate emitter files.

| Layer | New file (per profile) | Existing file (unchanged) |
|-------|-----------------------|---------------------------|
| Codegen | `ops/matmul.rs` (`emit_matmul`) | `ops/linear.rs` (`emit_linear`) |
| Codegen | `ops/mulscalar.rs` (`emit_mulscalar`) | — |
| Codegen | — | `ops/softmax.rs` — interface unchanged; only dispatch updated |
| Codegen | — | `ops/relu.rs`, `ops/dropout.rs` (unchanged) |

**Future consideration:** an extracted `emit_matmul_inner_k_loop`
helper shared between `emit_linear` and `emit_matmul` is an M11+
candidate, activated when SIMD vectorisation lands (Axis 1). Until
SIMD, the inner-loop bodies are not similar enough across linear
and matmul to justify the abstraction. After SIMD they will diverge
again (matmul SIMD vectorisation differs from
bias-add-followed-by-relu linear SIMD), so the helper is
value-neutral until the trigger fires.

### 6.2 ABI Invariant Preserved

`FnSig` shape `(input_ptr, params_ptr, output_ptr)` is unchanged.
For SelfAttention:

- `params_floats == 0`, `params_layout == []` (no `Linear` op).
- `input_floats == output_floats == batch * heads * seq * head_dim`
  (4D shape product). The existing `walk_model` code computes both
  via `shape.0.iter().product()` — already rank-agnostic.

### 6.3 Buffer Assignment

In `profiles/arm64/src/buffer.rs::assign_buffers`, extend the
per-op match:

- `StdOp::Matmul` joins `StdOp::Linear | StdOp::Softmax` — gets a
  `BufferLoc::StackOffset` (separate intermediate buffer).
- `StdOp::MulScalar` joins `StdOp::Relu | StdOp::Dropout` — gets a
  `BufferLoc::Alias(operands[0])` (in-place transformation).

Buffer size computation (`shape.iter().product() * 4 bytes`) is
already rank-agnostic; 4D shapes work without further changes.

Parallel change in `profiles/x86_64/src/buffer.rs`.

### 6.4 `emit_matmul` (arm64)

Signature:

```rust
pub fn emit_matmul(
    leading_count: u64,    // product of leading dims; 1 for 2D inputs
    m: u64, k: u64, n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    node_span: Span,
) -> Result<String, LowerError>
```

Structure:

1. Materialise three base pointers via `materialise_ptr`: `x11=A`,
   `x13=B`, `x12=DST`. **Invariant: `x11/x13/x12` are original base
   pointers and MUST remain unchanged across outer-loop iterations.**
   Inner code computes per-iteration slice pointers in scratch
   registers (`x6/x7/x8` as in `emit_linear`).
2. Emit outer loop over `leading_count` (counter `x17`); inside,
   compute `slice_offset = outer_idx * (slice_size_in_floats)` and
   add to base pointers into per-iteration registers.
3. Inner triple-loop (i ∈ [0,M), j ∈ [0,N), k ∈ [0,K)) — same
   FMA-using structure as `emit_linear` k-loop
   (`fmadd s0, s1, s2, s0`); no bias-add, no post-ops.
4. **Transpose-b inner addressing.** When `transpose_b == true`,
   the B inner-load addresses `B[j, k]` instead of `B[k, j]`:
   - `transpose_b = false`: `b_offset = k_inner * N + j` (B is
     `[..., K, N]`).
   - `transpose_b = true`:  `b_offset = j * K + k_inner` (B is
     `[..., N, K]`).

Performance note — outer loop costs ~3 pointer-additions per
iteration plus an `emit_imm32` for the slice stride (or hoisted
outside). For SelfAttention dimensions
(`leading_count = batch * heads = 8`), the outer wrapper executes 8
times; each iteration runs an `M*N*K = 4096`-FMA inner kernel.
Outer-loop overhead is roughly `3/4096` of inner work — negligible.

**No callee-saved register expansion.** The outer-loop counter
(`x17`) is a caller-saved scratch register. `RegSet` remains driven
solely by `calls_extern_math()`, to which `Matmul` does not
contribute.

Parallel implementation in
`profiles/x86_64/src/ops/matmul.rs::emit_matmul`:

- SysV AMD64 ABI: `%rdi` = input, `%rsi` = params, `%rdx` = output.
- Inner kernel uses `mulss + addss` (no FMA), consistent with
  `emit_linear` x86_64's deliberate non-FMA design from M9. The
  unit test `matmul_uses_mulss_addss_no_fma` (Section 8.4) asserts
  the absence of `vfmadd`.
- Outer-loop counter in caller-saved GPR (e.g., `%r10` or `%r11`).

### 6.5 `emit_mulscalar` (arm64)

Signature:

```rust
pub fn emit_mulscalar(
    total_elements: u64,
    scalar_bits: u32,        // f32-as-u32 bit pattern
    model_idx: usize,
    op_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String
```

Structure:

1. Materialise the scalar bit pattern into `s4` once before the
   loop:
   ```
   movz w9, #<lo16>
   movk w9, #<hi16>, lsl #16    ; only if hi16 != 0
   fmov s4, w9
   ```
2. Materialise src/dst pointers via `materialise_ptr`.
3. Flat loop over `total_elements`:
   ```
   ldr s0, [x11, x3, lsl #2]
   fmul s0, s0, s4
   str s0, [x12, x3, lsl #2]
   ```

When `dst_loc == src_loc` (the `Alias` case, the normal one), the
loop is in-place. The emitter does not branch on this case —
`materialise_ptr` resolves both into the same register.

**Scalar truncation contract (semantic boundary).** UIR stores
`mul_scalar`'s scalar as `AttrValue::Float(f64)`. Lowering converts
to f32 bits at the codegen boundary:

```rust
let scalar_f32 = float_attr_value as f32;
let scalar_bits = scalar_f32.to_bits();
```

For values like `0.25` the f64→f32 conversion is exact. For other
values the truncation is the intended behaviour — assembly
arithmetic is f32. This is documented as an explicit semantic
decision, not an implementation detail.

Parallel implementation in `profiles/x86_64/src/ops/mulscalar.rs`:

- Scalar pre-loaded into a free `xmm` register via stack spill or
  `movd` + `movss` from a GPR — pattern follows existing
  `emit_linear` x86_64 conventions.
- Inner instruction: `mulss <scalar_xmm>, <element_xmm>`.

### 6.6 Softmax Dispatch Update

The `emit_softmax` asm emitter (both profiles) — interface
unchanged: `(b, k, ...)`. The change is in the `walk_model::Softmax`
arm in `codegen.rs`:

```rust
StdOp::Softmax => {
    let in_shape = &model.nodes[operands[0]].ty.shape;
    // Last axis is the softmax axis; preceding dims flatten into row count.
    let last_idx = in_shape.0.len() - 1;
    let k = in_shape.0[last_idx];
    let b: u64 = in_shape.0[..last_idx].iter().product();
    // ... existing call to emit_softmax(b, k, ...)
}
```

For 2D `[batch, dim]`: `b = shape[0]`, `k = shape[1]` — identical to
current behaviour. For 4D `[B, H, M, K]`: `b = B*H*M`, `k = K`. The
3-pass per-row softmax kernel runs `b` times — no asm changes
required.

The same single-line change applies to the `Softmax` dispatch in
`profiles/x86_64/src/codegen.rs`.

The `(b, k)` interface is the right abstraction: `b` = total rows
(product of leading dims), `k` = row width (last dim). The dispatch
performs the rank-flattening once at the call site; the emitter is
rank-agnostic by construction. Passing the full shape into the
emitter would be a misallocation of responsibility.

### 6.7 `classify_op` Dispatch

In both profiles' `classify_op` (codegen.rs validation), add:

```rust
StdOp::Matmul => Ok(()),
StdOp::MulScalar => Ok(()),
```

The wildcard arm continues rejecting future unsupported ops via
`LowerError::UnsupportedOp`.

### 6.8 Prologue/Epilogue

Unchanged on both profiles. New ops do not call extern math, do
not require state preservation across calls, and do not need
additional callee-saved registers. `compute_callee_saved` continues
delegating to `UirModel::calls_extern_math()`.

---

## 7. Acceptance Fixture and Test Plan

### 7.1 Fixture: `tests/fixtures/self_attention.nfl`

```nfl
model SelfAttention [batch=2, heads=4, seq=16, head_dim=16]:
    x: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
    scaled: Tensor[batch, heads, seq, seq] = scores -> mul_scalar[0.25]
    attn:   Tensor[batch, heads, seq, seq] = scaled -> softmax
    out:    Tensor[batch, heads, seq, head_dim] = attn -> matmul[x]
```

Shape derivation:

- `x` → `[2, 4, 16, 16]`.
- `x @ x.T` → leading `[2, 4]`, inner
  `[16, 16] @ [16, 16].T = [16, 16]` with M=K=N=16. Output
  `[2, 4, 16, 16]`.
- `* 0.25` → shape preserved.
- `softmax` → shape preserved (last-axis softmax over `seq=16`).
- `attn @ x` → leading `[2, 4]`, inner
  `[16, 16] @ [16, 16] = [16, 16]`. Output `[2, 4, 16, 16]`.

Scale value: `1/sqrt(head_dim) = 1/sqrt(16) = 0.25` (precomputed).

Self-attention with `q = k = v = x` is mathematically degenerate
(no learnable Q/K/V projections), but the operation graph exercises
every new primitive: 4D matmul (twice), 4D matmul with
`transpose_b`, scalar multiply, last-axis softmax over a 4D tensor.
Compile correctness is validated regardless of semantic novelty.

### 7.2 Per-Profile Bit-Exact Strategy

Cross-profile bit-exact equivalence at the byte level is
**architecturally unreachable** in M10:

- arm64 `emit_linear` and `emit_matmul` use `fmadd` (single-rounding
  fused multiply-add).
- x86_64 `emit_linear` and `emit_matmul` use `mulss + addss`
  separately (two roundings, no FMA).
- `_expf` (libm) implementations differ between Apple libsystem
  (macOS arm64) and glibc (Linux x86_64).

These divergences are intentional design decisions from M9 and are
preserved in M10. The acceptance criterion is **per-profile
bit-exact against an architecture-matched reference**:

- `profiles/arm64/tests/integration.rs::reference_self_attention_arm64`
  uses `f32::mul_add` (matches arm64 `fmadd`) plus `f32::exp`
  wrapping platform libm.
- `profiles/x86_64/tests/integration.rs::reference_self_attention_x86_64`
  uses explicit two-step `let prod = a * b; sum = sum + prod;`
  (matches x86_64 `mulss + addss`) plus `f32::exp` wrapping platform
  libm. Comment in source: "uses separate mul+add to match x86_64
  emitter; intentional divergence from FMA, not a defect."

Within each profile's CI runner (macOS arm64 for arm64,
ubuntu-latest for x86_64), the FFI-compiled output and the
reference-implementation output match `assert_eq!` bit-exactly.

Cross-profile numerical agreement is measured by OQ-BENCH harness
(Section 11) as informational tolerance reports — not an M10
acceptance criterion.

### 7.3 Deterministic Input

Algorithmic generator, no `rand` crate dependency:

```rust
let total = batch * heads * seq * head_dim;
let input: Vec<f32> = (0..total)
    .map(|i| (i as f32).sin() * 0.1)
    .collect();
```

Range `[-0.1, 0.1]` keeps softmax inputs (after matmul + 0.25 scale)
in the well-conditioned numerically-stable range.

### 7.4 Test Harness — Zero-Params FFI Contract

For models with `params_floats == 0`, the test harness uses the
existing M5/M9 pattern:

```rust
let params = vec![0.0f32; sig.params_floats];
unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }
```

When `sig.params_floats == 0`, this produces an empty `Vec<f32>`.
Rust ABI guarantees that `Vec::as_ptr()` on an empty `Vec` returns
a non-null aligned ("dangling-but-aligned") pointer per the `Vec`
contract. The assembly never dereferences this pointer because
`params_floats == 0` and no params lookups occur.

**Implementer note:** the harness MUST NOT special-case
`params_floats == 0` with `ptr::null()` — that would violate the
non-null contract assumed (silently) by the existing FFI calling
convention. Just use the existing pattern; it is already correct.

### 7.5 Test Layers

| Layer | Location | Validates |
|-------|----------|-----------|
| Lexer / parser unit | `compiler/src/parser/tests.rs` | `named_pipeline_stmt` parses; lookahead on `=` distinguishes from `variable_decl`; AST contains correct `NamedPipeline` variant |
| AST / UIR builder unit | `compiler/src/ir/tests.rs` | `Matmul` / `MulScalar` nodes built; shape inference covers transpose, leading-dim mismatch, contraction mismatch, rank checks; `DeclaredShapeMismatch` fires correctly |
| Profile codegen unit | `profiles/{arm64,x86_64}/src/tests.rs` | `emit_matmul` produces outer-loop wrapper; transpose addressing differs; `emit_mulscalar` produces flat loop with scalar pre-load; `Softmax` dispatch computes `b = product(shape[..-1])` |
| Profile FFI integration | `profiles/{arm64,x86_64}/tests/integration.rs` | SelfAttention compiles end-to-end; FFI output matches per-profile reference bit-exactly |

Detailed enumeration in Section 8.

---

## 8. Test Enumeration

Estimated ~43 new tests; project total 284 → ~327. In line with M9
delta (+61) and M8 delta (+15).

### 8.1 Parser Tests (`compiler/src/parser/tests.rs`)

- `parse_named_pipeline_stmt_2d`
- `parse_named_pipeline_stmt_4d`
- `parse_named_pipeline_with_tensor_op_arg`
- `parse_lookahead_distinguishes_variable_decl_from_named_pipeline`
- `parse_named_pipeline_missing_eq_after_type` (negative)

### 8.2 UIR Builder Tests (`compiler/src/ir/tests.rs`)

- `matmul_2d_shape_inference_no_transpose`
- `matmul_2d_shape_inference_transpose_b`
- `matmul_4d_shape_inference_no_transpose`
- `matmul_4d_shape_inference_transpose_b`
- `matmul_leading_dim_mismatch_errors`
- `matmul_inner_dim_mismatch_errors`
- `matmul_rank_mismatch_errors` (2D @ 4D)
- `matmul_rank_too_low_errors` (1D input)
- `mul_scalar_preserves_shape`
- `softmax_rank_too_low_caught_at_uir`
- `named_pipeline_shape_match_succeeds`
- `named_pipeline_shape_mismatch_errors`
- `tensor_arg_resolves_from_env`
- `tensor_arg_unknown_variable_errors`
- `transpose_b_true_recognised`

### 8.3 arm64 Codegen Tests (`profiles/arm64/src/tests.rs`)

- `matmul_4d_emits_outer_loop_wrapper`
- `matmul_2d_collapses_to_outer_count_one`
- `matmul_transpose_b_inner_addressing_differs`
- `matmul_uses_fmadd_native_to_arm64` (consistency with
  `emit_linear`)
- `matmul_does_not_call_extern_math`
- `mul_scalar_preloads_scalar_via_movz_movk`
- `mul_scalar_emits_fmul_in_inner_loop`
- `softmax_4d_dispatch_computes_b_as_product_of_leading_dims`
- `prologue_unchanged_for_self_attention`
- `asymmetric_matmul_shape_M_neq_K_neq_N` (M=4, K=8, N=2;
  general-case `transpose_b` coverage)

### 8.4 x86_64 Codegen Tests (`profiles/x86_64/src/tests.rs`)

Parallel set, mirroring 8.3:

- `matmul_4d_emits_outer_loop_wrapper`
- `matmul_2d_collapses_to_outer_count_one`
- `matmul_transpose_b_inner_addressing_differs`
- `matmul_uses_mulss_addss_no_fma` (consistency with `emit_linear`)
- `matmul_does_not_call_expf_plt`
- `mul_scalar_uses_mulss`
- `mul_scalar_preloads_scalar`
- `softmax_4d_dispatch_computes_b_as_product_of_leading_dims`
- `prologue_unchanged_for_self_attention`
- `asymmetric_matmul_shape_M_neq_K_neq_N`

### 8.5 Integration FFI Tests

- `profiles/arm64/tests/integration.rs::self_attention_ffi_matches_reference`
- `profiles/x86_64/tests/integration.rs::self_attention_ffi_matches_reference`

Each compiles `tests/fixtures/self_attention.nfl`, assembles, links,
loads via `libloading`, calls with the deterministic input from
Section 7.3, and asserts `assert_eq!` against
`reference_self_attention_<arch>`.

### 8.6 Negative Fixtures (`tests/fixtures/negative/`)

- `bad_named_pipeline_missing_eq.nfl` — parser-level rejection.
- `bad_matmul_rank_too_low.nfl` — UIR rank-check rejection.
- `bad_matmul_inner_dim_mismatch.nfl` — UIR contraction-dim
  rejection.
- `bad_named_pipeline_shape_mismatch.nfl` — UIR
  declared-vs-actual rejection.

`RankMismatch` (2D @ 4D operands) is covered by UIR builder unit
test without a corresponding fixture — pure UIR check, no
parser/profile specifics needed.

---

## 9. Non-Goals (Explicit)

### 9.1 Grammar Extensions Deferred

- Infix operators (`@`, `*`, `.T`) as expression syntax.
- General expression grammar (binary ops, function calls,
  parentheses).
- Multi-input model declarations (separate `q`, `k`, `v`).
- Tensor reshape op (would require UIR-level dynamic shape
  tracking).
- Boolean literals as a grammar token.
- Expression evaluation / constant folding for `mul_scalar`
  operand.

### 9.2 UIR / Compiler Features Deferred

- Broadcasting in `Matmul` (strict-equal leading dims is the
  invariant).
- Fusion passes for attention-internal patterns
  (`Matmul → MulScalar → Softmax → Matmul`).
- Generalisation of `StdOp::Linear` to handle activation @
  activation.
- Multi-output ops.

### 9.3 Codegen Changes Deferred

- Modifications to `emit_linear`, `emit_relu`, `emit_dropout`, or
  `emit_softmax` (the asm emitter; only the dispatch into Softmax
  changes).
- Shared `emit_matmul_inner_k_loop` helper (M11+ candidate;
  activates with SIMD).
- `RegSet` extension / new callee-saved registers.
- Prologue/epilogue changes.
- ABI / `FnSig` changes.

### 9.4 Attention Variants Out of Scope

- Q/K/V learnable projections.
- Multi-head via explicit reshape.
- Causal / masked attention.
- Rotary position embedding (RoPE) and other positional encodings.
- Grouped Query Attention / Multi-Query Attention.
- Cross-attention (would require multi-input model).
- FlashAttention / tiled attention.

### 9.5 Transformer Block Components Out of Scope

- Residual connections (would require pointwise add op).
- Layer normalisation.
- FFN block bundling into the acceptance fixture.
- Per-attention dropout.

### 9.6 Performance / Numerics Deferred

- SIMD on either profile (Axis 1).
- Bare-metal `expf` replacement (Axis 3).
- Cross-profile bit-exact byte-level equality (architecturally
  unreachable; tolerance reporting in OQ-BENCH).
- New quantisation kinds (INT8 / FP16 / BF16).

### 9.7 Adjacent Strategic Axes

Open and orthogonal to M10 — explicit non-decisions, not closures:

- **Axis 1** — codegen breadth (SIMD, additional architectures,
  macOS x86_64 Mach-O).
- **Axis 3** — deployment reach (bare-metal expf, drop libm
  dependency).

---

## 10. Sub-Milestone Decomposition

M10 is **monolithic** — a single PR. Splitting M10 into
sub-milestones would require independently shippable deliverables,
but every layer (grammar, UIR, codegen) is mutually dependent for
the acceptance test to compile and run.

Atomic task-pack sequence (per the convention from M7; final count
and ordering finalised by `writing-plans`):

1. Grammar + AST: lexer/parser/AST for `named_pipeline_stmt`, plus
   parser unit tests.
2. UIR builder — args machinery: `ArgType::Tensor`, `resolve_args`
   signature change, env lookup for tensor args, builder unit
   tests. (All `resolve_args` callsites updated in the same commit
   per Section 5.3.)
3. UIR builder — `StdOp::Matmul`: variant, signature, shape
   inference, new `ShapeError` variants, unit tests.
4. UIR builder — `StdOp::MulScalar`: variant, signature, shape
   passthrough, unit tests.
5. UIR builder — `named_pipeline_stmt` consumption:
   `BuildError::DeclaredShapeMismatch`, builder logic, Softmax
   rank-check tightening, unit tests.
6. arm64 codegen — `emit_matmul`: new `ops/matmul.rs`, dispatch,
   `classify_op` extension, codegen unit tests.
7. arm64 codegen — `emit_mulscalar`: new `ops/mulscalar.rs`,
   dispatch, unit tests.
8. arm64 codegen — Softmax dispatch generalisation:
   `walk_model::Softmax` arm updated for `b = product(shape[..-1])`.
9. x86_64 codegen — `Matmul` + `MulScalar` + Softmax dispatch
   (parallel to steps 6-8; possibly one task-pack given
   isomorphism).
10. Integration FFI: `self_attention.nfl` fixture, per-profile
    reference implementations, FFI tests on both profiles.
11. Negative fixtures + final cleanup: four rejection fixtures,
    any missed unit tests.
12. Documentation: `language/grammar.ebnf`,
    `docs/language_reference/grammar.md`,
    `docs/profile_guide/{arm64,x86_64}.md`, `PROJECT_SPEC.md`
    (First Milestones table + Current Status + Strategic Roadmap
    revision), `DEVLOG.md`, `CLAUDE.md` Repository Structure tree.

Estimated ~12 task-packs — close to M9 size.

---

## 11. OQ-BENCH Pre-Commit (Separate from M10)

Single-day commit landed **before** M10 implementation begins.
Closes the OQ-BENCH trigger formally and provides scalar-FFN
baseline numbers.

### 11.1 Scope

1. **Harness:** new `tools/bench/` (or `xtask/bench/`, depending
   on workspace convention — implementer to verify). CLI driver
   compiles one NFL fixture through both profiles, assembles,
   links, runs with deterministic input, measures wall-clock
   timing, emits markdown report.
2. **Markdown report:** `bench/results-2026-05-08.md` with
   side-by-side timings for fixtures `classifier`,
   `softmax_with_bias`, `mixed_args`, `tiny_mlp`. Each row reports
   arm64 timing, x86_64 timing, and a per-profile output-tolerance
   column (max abs diff vs the in-process Rust neutral reference;
   see 11.2).
3. **`PROJECT_SPEC.md` update:** OQ-BENCH entry moves from
   "Trigger-driven cleanup" to "Decisions (formerly open, now
   resolved)" with a reference to the commit.

### 11.2 Cross-Profile Comparison Implementation

Avoid CI artifact sharing (arm64 outputs → ubuntu job dependency).
Instead, **use a pure-Rust neutral reference implementation as
proxy** for one side of the comparison:

- Each per-profile CI job runs its FFI-compiled binary AND
  computes a "neutral" Rust reference. The output-tolerance
  column compares the FFI output to this in-process reference
  within the job. The reference style is documented per fixture
  (FMA-using vs separate mul+add).
- Cross-profile delta is then derived offline (or in a
  documentation step) by combining the two reports — not gated
  by CI orchestration.

This avoids artifact-sharing complexity while still producing a
meaningful "FMA divergence × libm divergence" tolerance signal
per profile.

### 11.3 Out of Scope for OQ-BENCH

- Regression-gate in CI (numbers informational, not
  fail-conditions).
- Benchmarking M10-new ops (`Matmul`, `MulScalar`) — they don't
  exist at OQ-BENCH commit time.
- Tolerance gates (cross-profile) — useful post-M10, not part of
  OQ-BENCH initial commit.
- Benchmarking under multiple compiler optimisation levels.

### 11.4 Estimated Size

One day, possibly 1.5 with markdown polish: harness driver, four
fixture runs, report.

### 11.5 Post-M10 Reuse

After M10 lands, the OQ-BENCH harness is naturally extended to run
the SelfAttention fixture and produce the cross-profile tolerance
report that Section 7.2 references — but this extension is *not*
part of the OQ-BENCH initial commit.

---

## 12. Open Questions / Risk Assessment

### 12.1 Resolved during Brainstorm

- **Axis selection** — Axis 2 (modelling depth). Settled.
- **Scope** — Option C (minimal primitive set for attention).
  Settled.
- **Acceptance criterion** — Option α (single-input self-attention,
  q = k = v = x). Settled, primarily on ABI-invariant grounds.
- **Matmul vs Linear NodeKind** — separate `StdOp::Matmul`.
  Settled, on design-principle and lowering-cleanliness grounds.
- **Cross-profile bit-exact** — deferred to OQ-BENCH tolerance
  reports. Settled, on architectural-unreachability grounds (FMA
  divergence + libm divergence).

### 12.2 Trigger-Driven Items Opened by M10

None. M10 closes the OQ-BENCH trigger via the pre-commit; no new
triggers are speculatively opened by this spec.

### 12.3 Items Deferred to M11+ Decision Point

- Whether M11 continues Axis 2 (multi-input grammar + Q/K/V
  projections, unblocking residual + LayerNorm + FFN for a
  transformer block) or pivots to Axis 1 (SIMD on existing
  profiles, OQ-BENCH numbers as the selection signal) or Axis 3
  (bare-metal expf).
- Extraction of `emit_matmul_inner_k_loop` shared helper —
  activates with SIMD landing on either profile.

### 12.4 Risks

- **Risk: parser one-token lookahead implementation.** If
  `named_pipeline_stmt` and `variable_decl` are merged at the
  parser level (one function with a conditional `=` branch), the
  AST shape becomes ambiguous and downstream consumers must
  disambiguate.
  *Mitigation:* spec mandates separate productions and separate
  AST variants; parser unit test
  `parse_lookahead_distinguishes_variable_decl_from_named_pipeline`
  guards against the merged form.
- **Risk: `resolve_args` signature change missed callsites.** A
  silent compile error without context.
  *Mitigation:* atomic-task-pack convention (Task 2 includes the
  signature change AND all callsite updates in one commit).
- **Risk: FFI harness adopts `ptr::null()` for `params_floats == 0`.**
  Violates the silent non-null contract.
  *Mitigation:* spec Section 7.4 documents the contract; existing
  M9 harness pattern `vec![0.0f32; sig.params_floats]` already
  meets it. Reviewer guards against regressions.
- **Risk: `transpose_b` inner-addressing bug** producing wrong
  result without a crash.
  *Mitigation:* unit test
  `matmul_transpose_b_inner_addressing_differs` asserts on the
  emitted asm pattern; integration test catches numerical
  wrongness via reference comparison.
- **Risk: scalar f64→f32 truncation surprises.** A future user
  writes `mul_scalar[3.141592653589793]` and is surprised that
  arithmetic uses the f32 truncation `3.1415927`.
  *Mitigation:* spec Section 6.5 documents the truncation
  contract; profile guide documentation (`docs/profile_guide/*.md`)
  records it. NFL is f32-only project-wide (per
  `BYTES_PER_ELEMENT = 4` in M4b); this is consistent with
  language-level expectations.

---

*This spec evolves through plan synthesis (`writing-plans` skill).
Implementation-time discoveries that alter scope, contracts, or
acceptance criteria should round-trip back to this document via the
same brainstorm-then-spec process.*
