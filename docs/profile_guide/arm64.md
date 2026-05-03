# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M4a complete (NFL v0.1). Lowers `linear[N]` (no bias) and `relu`
> to native AArch64 Mach-O assembly.
> **Authoritative source:** `profiles/arm64/src/` and the M4a spec under
> `docs/superpowers/specs/`.

The `arm64` profile is the first concrete codegen profile in NeuralForge. It
takes a `compiler::Uir` and emits AArch64 assembly (Mach-O syntax) callable as a
C function. M4a's scope is intentionally small — single-Linear models with
optional ReLU — so the rest of the pipeline (`nflc compile` CLI, integration
tests, FFI) is exercised end-to-end without getting blocked on transcendental
functions like `softmax`'s `exp`.

---

## 1. Calling convention (ABI)

For each `UirModel` in the input UIR, the profile emits one `extern "C"` function:

```c
void nfl_forward_<ModelName>(
    const float* input,
    const float* weights,
    float*       output
);
```

Standard AAPCS64: pointers in `x0`, `x1`, `x2`. Pure leaf function — no callee-saved registers touched, no stack frame, no calls into libc.

The symbol name in the asm is `_nfl_forward_<ModelName>` (Mach-O underscore prefix). C / FFI callers pass the underscore-less name to `dlsym`; the dynamic loader handles the prefix.

---

## 2. Buffer layout

All buffers are `f32`, row-major.

For an `input → linear[N] → relu` model where `input: Tensor[B, K]`:

| Buffer    | Size (f32 elements) | Layout                                                      |
|-----------|---------------------|-------------------------------------------------------------|
| `input`   | B × K               | `input[i * K + k]` for row i, column k.                     |
| `weights` | K × N               | `weights[k * N + j]` for row k, column j.                   |
| `output`  | B × N               | `output[i * N + j]` for row i, column j.                    |

Sizes are reported on the returned `FnSig` (`input_floats`, `weight_floats`, `output_floats`). The caller must allocate exactly these sizes. M4a does not perform any bounds checking — passing undersized buffers is undefined behaviour.

For models with multiple Linear ops (M4b+), `weights` is the **packed concatenation** of all weight matrices in UIR-node (topological) order. M4b adds `FnSig.weights_layout: Vec<WeightSlot>` so callers know each matrix's offset and size.

---

## 3. Supported ops in M4a

| StdOp                      | Supported | Notes                                                          |
|----------------------------|-----------|----------------------------------------------------------------|
| `Linear` (no `bias` attr)  | ✅        | Pure matmul. No bias add.                                     |
| `Linear` (`bias=true`)     | ❌ M4b    | Returns `LowerError::LinearWithBias`.                         |
| `Relu`                     | ✅        | Separate elementwise loop. Operates in-place on output buffer. |
| `Dropout`                  | ❌ M4b    | Returns `LowerError::UnsupportedOp { op: "dropout" }`.        |
| `Softmax`                  | ❌ M4b    | Returns `LowerError::UnsupportedOp { op: "softmax" }`.        |
| `Input`                    | ✅        | Marker only — maps to the input pointer.                      |

### Codegen-decision: `linear[N]` without `bias` attribute

Interpreted as **pure matmul, no bias add**. The NFL grammar marks `bias` as optional but doesn't commit a default. The arm64 profile treats absence of the `bias` attribute as `bias=false`. To get bias-add explicitly, write `linear[N, bias=true]` — which M4a rejects with `LowerError::LinearWithBias` and M4b implements.

---

## 4. Code-gen patterns

### 4.1 Matmul (Linear)

Three nested scalar loops. For `linear[N]` over input shape `[B, K]`:

```asm
    mov     x3, #0              ; i = 0
.Lmm_i_<idx>:
    cmp     x3, #B
    b.ge    .Lmm_i_end_<idx>

    mov     x4, #0              ; j = 0
.Lmm_j_<idx>:
    cmp     x4, #N
    b.ge    .Lmm_j_end_<idx>

    fmov    s0, wzr             ; sum = 0.0
    mov     x5, #0              ; k = 0
.Lmm_k_<idx>:
    cmp     x5, #K
    b.ge    .Lmm_k_end_<idx>

    mov     x8, #K              ; load input[i*K + k]
    mul     x6, x3, x8
    add     x6, x6, x5
    ldr     s1, [x0, x6, lsl #2]

    mov     x8, #N              ; load weights[k*N + j]
    mul     x7, x5, x8
    add     x7, x7, x4
    ldr     s2, [x1, x7, lsl #2]

    fmadd   s0, s1, s2, s0      ; sum += input * weight (single-rounding FMA)
    add     x5, x5, #1
    b       .Lmm_k_<idx>
.Lmm_k_end_<idx>:

    mov     x8, #N              ; store output[i*N + j]
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x2, x6, lsl #2]

    add     x4, x4, #1
    b       .Lmm_j_<idx>
.Lmm_j_end_<idx>:
    add     x3, x3, #1
    b       .Lmm_i_<idx>
.Lmm_i_end_<idx>:
```

`<idx>` is a per-Linear-op suffix so labels don't collide when M4b adds multi-Linear models.

Index arithmetic uses `mul` (not `lsl`) so the emitter is correct for any K, N — not tied to powers of 2. Performance is M5+ territory.

### 4.2 Relu

Separate elementwise loop. Operates in-place on the output buffer (M4a always
has Relu as the terminal op):

```asm
    fmov    s4, wzr             ; materialise 0.0 once outside the loop
                                ; (wzr is integer; AArch64 fmax requires both
                                ; operands in FP regs, so we can't pass wzr
                                ; directly to fmax)
    mov     x9, #0              ; element index
.Lrelu_<idx>:
    cmp     x9, #<total>        ; total = B*N for terminal-relu after linear[N]
    b.ge    .Lrelu_end_<idx>
    ldr     s3, [x2, x9, lsl #2]
    fmax    s3, s3, s4
    str     s3, [x2, x9, lsl #2]
    add     x9, x9, #1
    b       .Lrelu_<idx>
.Lrelu_end_<idx>:
```

When M4b adds multi-stage models (e.g. `linear → relu → linear`), `emit_relu`
will need an explicit "operand-buffer pointer" parameter so it can clamp
intermediate buffers, not just `x2`.

### 4.3 Function frame

Pure leaf function: just label + body + `ret`. No prologue, no epilogue, no stack frame, no callee-saved register handling.

```asm
.globl _nfl_forward_<ModelName>
.p2align 2
_nfl_forward_<ModelName>:
    ; <matmul + relu body>
    ret
```

---

## 5. Errors

`profiles_arm64::lower` returns `Result<Asm, LowerError>`. `LowerError` is
`#[non_exhaustive]`; consumers must keep a `_ => ...` arm. Variants in M4a:

| Variant                      | When                                                                     |
|------------------------------|--------------------------------------------------------------------------|
| `UnsupportedOp { op, span }` | Op isn't supported in the current slice (currently `softmax`, `dropout`). |
| `LinearWithBias { span }`    | `linear[N, bias=true]` — M4b adds support.                              |
| `ShapeNotConcrete { span }`  | Defensive: shape wasn't fully resolved by `ir::build`. Should be unreachable. |
| `DuplicateModelName { name, span }` | Two `UirModel`s share `name` — would produce conflicting symbols. M4b moves this check up to `ir::build`. |

The CLI (`nflc compile`) renders these via the existing `render_error_with_snippet` helper from M3c — same `error: ... --> file:line:col ... ^` format as parser/IR errors.

---

## 6. Adding a new op

To add an op to the `arm64` profile (e.g. `tanh`, `sigmoid`):

1. Add an arm in `profiles/arm64/src/codegen.rs::classify_op` returning `Ok(())` for the new op.
2. Add a per-op emitter, e.g. `fn emit_tanh(total_floats: u64, op_idx: usize) -> String`.
3. Add a dispatch arm in `walk_model`'s op-loop calling the new emitter.
4. Add unit tests in `profiles/arm64/src/tests.rs` asserting the asm contains the expected instructions.
5. Add an integration test if the op participates in end-to-end runnable code.
6. Update this doc's §3 table.

---

## 7. Adding a new architecture profile

To add a new profile (e.g. `x86_64`, `riscv64`):

1. Create `profiles/<arch>/Cargo.toml` mirroring `profiles/arm64/Cargo.toml`. `[dependencies] compiler = { path = "../../compiler" }`.
2. Add `"profiles/<arch>"` to the workspace `members` in `/Cargo.toml`.
3. Implement the same public surface as `profiles_arm64` — `pub fn lower(&Uir) -> Result<Asm, LowerError>` plus the `Asm`, `FnSig`, `LowerError` types. (M5+ may extract a shared `profile-api` crate when the second profile lands; for M4a that's premature.)
4. Add a dispatch arm in `nflc/src/main.rs::run_compile` for the new `--profile <arch>` value.
5. Mirror this guide as `docs/profile_guide/<arch>.md`.

---

## 8. Limitations (M4a)

Items deferred to M4b/c:

- No bias-add in `linear`.
- No `softmax`. Needs `exp()`; deferred to M4b (Taylor series or `expf` symbol).
- No `dropout`. Semantically identity at inference, but bundled to M4b.
- No multi-output models. Implicit-output convention (one output per model).
- No SIMD. Scalar instructions only. NEON / SVE are M5+ work.
- No optimisation passes. Three-nested-loop matmul; `mul` for indexing; per-element load/store. Performance is M5+.
- No CI configuration.
- Integration test runs only on aarch64 hosts with `cc` available; skips with logged reason elsewhere.
