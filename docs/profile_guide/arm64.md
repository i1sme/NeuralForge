# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M17 complete (bare-metal inline exp). Lowers `linear` (with or without
> `bias=true`), `relu`, `dropout` (no-op pass-through at inference, or
> explicit copy-loop when dropout IS model output — see §M8),
> `softmax` (numerically stable 3-pass, inline bare-metal exp — Cody-Waite
> + degree-7 Taylor, no libm; generalised to rank ≥ 2 in M10), `linear → softmax` fused via `PostOp::SoftmaxRow`,
> the two M10 attention-pattern ops `matmul` (multi-dim, optional
> `transpose_b`) and `mul_scalar` (scalar pre-load + flat loop),
> `add` (elementwise tensor addition, M13), and `layernorm` (3-pass
> per-row normalization + optional affine, native `fsqrt`, M14) to
> native AArch64 Mach-O assembly. The compiler runs the default
> UIR-pass pipeline (`EliminateDropout` + `FuseLinearRelu` +
> `FuseLinearSoftmax`) before lowering. All 5 M3 positive fixtures +
> M4a + the M10 self-attention fixture run end-to-end via FFI;
> bit-exact equivalence (per-profile) proven on classifier,
> mixed_args, self_attention, and M14 LayerNorm integration tests. M8
> added dim-immediate uniformity via `asm::emit_imm32` and the
> dropout-as-output copy-loop fix; M10 added the multi-dim matmul
> outer-loop wrapper and the post-Group-10 softmax FFI-register
> preservation (`stp/ldp x0/x1/x2`).
> **Authoritative source:** `profiles/arm64/src/` and the M4a/M4b/M5a/M5b/M6/M7/M8/M14
> specs under `docs/superpowers/specs/`.

The `arm64` profile is the first concrete codegen profile in NeuralForge. It
takes a `compiler::Uir` and emits AArch64 assembly (Mach-O syntax) callable as a
C function. M4a shipped the minimal honest end-to-end path (linear+relu); M4b
extends to all 5 M3 positive fixtures by adding bias, dropout, and softmax,
plus the supporting infrastructure for multi-stage models (intermediate stack
buffers, non-leaf prologue, per-model label namespacing, packed `params` buffer
with typed slot metadata).

As of M9, arm64 coexists with the x86_64 Linux ELF profile. Both implement the
shared `Profile` trait from `profile-api/`; the symbol-prefix abstraction
(`sym_prefix() -> "_"` on Mach-O, `""` on ELF) plus per-profile asm emission is
the contract. Cross-profile architectural differences (notably: callee-saved FP
register sets) are documented in [`x86_64.md`](x86_64.md).

---

## 1. Calling convention (ABI)

For each `UirModel` in the input UIR, the profile emits one `extern "C"` function:

```c
void nfl_forward_<ModelName>(
    const float* input,
    const float* params,    // packed: weights + biases of all Linear nodes
    float*       output
);
```

Standard AAPCS64: pointers in `x0`, `x1`, `x2`. Function may be leaf (no
frame) or non-leaf (softmax — conservatively non-leaf in M17 because the
loop holds state in callee-saved registers; precise leaf reclassification
is M18); the prologue/epilogue is built conditionally per model based on
which ops are present.

The symbol name in the asm is `_nfl_forward_<ModelName>` (Mach-O underscore
prefix). C / FFI callers pass the underscore-less name to `dlsym`; the
dynamic loader handles the prefix.

---

## 2. Buffer layout

All buffers are `f32`, row-major.

| Buffer    | Size (f32 elements)               | Layout                                          |
|-----------|-----------------------------------|-------------------------------------------------|
| `input`   | sum over `input.shape`            | `input[i * K + k]` for row i, col k.            |
| `params`  | `FnSig.params_floats`             | Packed slots — see §2.5 below.                   |
| `output`  | sum over terminal-node shape      | `output[i * N + j]` for row i, col j.           |

Sizes are reported on the returned `FnSig` (`input_floats`, `params_floats`,
`output_floats`). The caller must allocate exactly these sizes. The profile
does not perform any bounds checking — passing undersized buffers is
undefined behaviour.

### 2.5. `params` buffer layout

`params` is a single packed float buffer holding all Linear weights and
biases for the model, in topological (UIR-node) order. For each `Linear`
node:

1. The weight matrix slot — `ParamKind::LinearWeight`, size `K * N`.
2. (If `bias=true`) the bias vector slot — `ParamKind::LinearBias`, size `N`.

Slot offsets and sizes are exposed via `FnSig.params_layout: Vec<ParamSlot>`.
Each `ParamSlot` carries:

- `kind: ParamKind` — what the slot holds.
- `origin_node: NodeId` — which UIR node owns the slot.
- `offset: usize` — start position in the params buffer, in f32 elements.
- `size: usize` — slot length, in f32 elements.

Callers use this metadata to serialise their model checkpoint into the right
offsets. Example for `classifier.nfl` (3 Linear, no bias):

```
slot 0: LinearWeight  (node 1, offset=0,        size=784*512)
slot 1: LinearWeight  (node 4, offset=401408,   size=512*256)
slot 2: LinearWeight  (node 6, offset=532480,   size=256*10)
total params_floats = 535040
```

`ParamKind` is `#[non_exhaustive]`. M5+ ops introduce new variants
(`NormGamma`, `EmbeddingTable`, `AttnQ/K/V/O`, …) without breaking
downstream `match` consumers.

---

## 3. Supported ops

| StdOp                      | Supported | Notes                                                          |
|----------------------------|-----------|----------------------------------------------------------------|
| `Linear` (no `bias` attr)  | ✅        | Pure matmul. With `fused_post_ops: [Relu]` (default-pipeline output of `linear → relu`): adds inline `fmax s0, s0, s4` post-op before store — see §4.9. |
| `Linear` (`bias=true`)     | ✅        | Matmul + per-output bias-add inline. With `fused_post_ops: [Relu]` (default-pipeline output of `linear[bias=true] → relu`): bias-add then inline `fmax` then store — see §4.9. |
| `Relu`                     | ✅        | Standalone (only in `--no-passes` mode, or `--passes` filter excluding `fuse_linear_relu`): separate elementwise loop, copy-with-clamp src→dst (§4.2). Default mode: fused into preceding Linear via `FuseLinearRelu` UIR pass — see §4.9. |
| `Dropout`                  | ✅        | Standalone (only in `--no-passes` mode, or `--passes` filter excluding `eliminate_dropout`): no asm, `BufferLoc::Alias(operand)` propagation (§4.5). Default mode: removed from UIR by `EliminateDropout` UIR pass before reaching the profile. |
| `Softmax`                  | ✅        | Numerically stable 3-pass (max → inline-exp → normalise), inline bare-metal exp (M17: no `bl _expf`). With `--no-passes` or `--passes` filter excluding `fuse_linear_softmax`: emitted as a standalone function via `emit_softmax` (labels `.Lsm_*`; see §4.4). Default pipeline (M6+): fused into the preceding Linear's `emit_linear` via `PostOp::SoftmaxRow` (row-wise tail; labels `.Lfsmx_*`; see §4.10). |
| `Add`                      | ✅        | Flat elementwise loop (M13). See §M13 ops. |
| `LayerNorm` (no `affine`)  | ✅        | 3-pass per-row: mean → variance + inv_std → normalize. Native `fsqrt`. Leaf (M14). See §M14 ops. |
| `LayerNorm` (`affine=true`) | ✅       | 3-pass + per-element γ/β affine transform. `s_b` reuses `s2` (see §M14 ops). |
| `Input`                    | ✅        | Marker only — `BufferLoc::InputReg` (`x0`).                   |

### Codegen-decision: `linear[N]` without `bias` attribute

Interpreted as **pure matmul, no bias add**. The NFL grammar marks `bias` as
optional but doesn't commit a default. The arm64 profile treats absence of
the `bias` attribute as `bias=false`. To get bias-add explicitly, write
`linear[N, bias=true]`.

---

## 4. Codegen patterns

### 4.1 Matmul (Linear without bias)

Three nested scalar loops. For `linear[N]` over input shape `[B, K]`:

```asm
    mov     x3, #0              ; i = 0
.Lmm_i_<m>_<l>:
    cmp     x3, #B
    b.ge    .Lmm_i_end_<m>_<l>

    mov     x4, #0              ; j = 0
.Lmm_j_<m>_<l>:
    cmp     x4, #N
    b.ge    .Lmm_j_end_<m>_<l>

    fmov    s0, wzr             ; sum = 0.0
    mov     x5, #0              ; k = 0
.Lmm_k_<m>_<l>:
    cmp     x5, #K
    b.ge    .Lmm_k_end_<m>_<l>

    mov     x8, #K              ; load input[i*K + k]
    mul     x6, x3, x8
    add     x6, x6, x5
    ldr     s1, [x11, x6, lsl #2]   ; x11 = src pointer

    mov     x8, #N              ; load weights[k*N + j]
    mul     x7, x5, x8
    add     x7, x7, x4
    ldr     s2, [x13, x7, lsl #2]   ; x13 = weight pointer (= params + offset)

    fmadd   s0, s1, s2, s0      ; sum += input * weight (single-rounding FMA)
    add     x5, x5, #1
    b       .Lmm_k_<m>_<l>
.Lmm_k_end_<m>_<l>:

    mov     x8, #N              ; store output[i*N + j]
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]   ; x12 = dst pointer

    add     x4, x4, #1
    b       .Lmm_j_<m>_<l>
.Lmm_j_end_<m>_<l>:
    add     x3, x3, #1
    b       .Lmm_i_<m>_<l>
.Lmm_i_end_<m>_<l>:
```

`<m>` is `model_idx` (per-model namespacing — see §4.8); `<l>` is per-Linear
within the model. `x11`/`x12`/`x13` are materialised once at the top of the
emitter from `BufferLoc`s + `weight_offset` (see `materialise_ptr` in
`ops/linear.rs`).

Index arithmetic uses `mul` (not `lsl`) so the emitter is correct for any
K, N — not tied to powers of 2. Performance is M5+ territory.

### 4.2 Relu

Separate elementwise loop. Copies from `src` (x11) to `dst` (x12) with
elementwise clamp:

```asm
    fmov    s4, wzr             ; materialise 0.0 once outside the loop
                                ; (wzr is integer; AArch64 fmax requires both
                                ; operands in FP regs, so we can't pass wzr
                                ; directly to fmax)
    mov     x9, #0              ; element index
.Lrelu_<m>_<r>:
    cmp     x9, #<total>        ; total = product of buffer shape
    b.ge    .Lrelu_end_<m>_<r>
    ldr     s3, [x11, x9, lsl #2]
    fmax    s3, s3, s4
    str     s3, [x12, x9, lsl #2]
    add     x9, x9, #1
    b       .Lrelu_<m>_<r>
.Lrelu_end_<m>_<r>:
```

Source and destination buffers are resolved per-node via `assign_buffers`
+ `resolve_loc`. M4a's in-place optimisation (writing back to operand
buffer) was dropped in M4b for cleaner buffer accounting; future fusion
pass will restore it.

### 4.3 Bias-add (inline in `linear[N, bias=true]`)

After the k-loop accumulates `s0 = sum`, before the output store:

```asm
    ldr     s5, [x14, x4, lsl #2]    ; bias[j], with x14 = params + bias_offset*4
    fadd    s0, s0, s5
```

`x14` is set up once at the top of the linear emitter when
`bias_offset.is_some()`. Bias offset is looked up from `sig.params_layout`
by the dispatcher in `walk_model`.

### 4.4 Softmax (per-row 3-pass, inline bare-metal exp — M17)

Per row `i`, three passes over `K` elements:

1. **Max scan:** `s8 = max(row)`, initialised to `-inf`.
2. **Exp + sum:** for each `k`, `output[i,k] = inline_exp(input[i,k] - s8)` and
   `s9 += output[i,k]`.
3. **Normalize:** for each `k`, `output[i,k] /= s9`.

`s8` (per-row max) and `s9` (per-row sum) are AAPCS64 callee-saved (lower
64 bits of `v8`/`v9`). Function prologue saves `d8`/`d9` when
`compute_callee_saved` returns `RegSet { d8_d9: true }`.

**Inline exp algorithm (M17).** No `bl _expf` — the exponentiation is
inlined via `emit_exp_inline()` from `ops/exp.rs`:

1. **Cody-Waite range reduction:** `z = round_ties_even(x · LOG2E)` via
   `fcvtns`; `zf = scvtf(z)`; `r = (x − zf·LN2_HI) − zf·LN2_LO` via two
   `fmsub` (fused, single rounding each).
2. **Degree-7 Taylor Horner:** `p = C7; p = p*r + C6; ...; p = p*r + C0`
   (7× `fmadd`). Coefficients `1/k!` are loaded from the file-local
   `.section __TEXT,__const` pool (see §4.4.1).
3. **`2^z` reconstruction:** `bits = (z+127) << 23` via `add`/`lsl`;
   `pow = fmov(bits → f32)`. Branchless underflow clamp: if `z+127 ≤ 0`
   then `pow = 0.0` (via `csel wzr`). Result: `s0 = p * pow`.

**Scratch contract.** `emit_exp_inline` uses: `x9` (pool base), `w11`
(z), `w12` (pow bits), `s1`–`s5` (FP temps). All are caller-saved and
non-loop-live. The inline helper does NOT touch the softmax loop's
callee-saved state (`x19`–`x23`, `s8`, `s9`).

**Loop state in callee-saved registers.** The loop state (`i`, `row_base`,
`k`, `src` pointer, `dst` pointer) lives in `x19`–`x23` because the inline
exp body clobbers caller-saved registers. The element offset (`x6`) is
recomputed after each call. The function prologue saves `x19`–`x23` when
`RegSet.x19_x23` is set (true iff softmax is present in the model).

**FFI save/restore (retained in M17).** `abi.emit_ffi_save` /
`emit_ffi_restore` spill the ABI argument registers (inputs + params +
output pointers) across the inline exp's scratch usage so downstream
emitters can re-materialise pointers. The spill block is RETAINED
unchanged in M17; its removal (along with the callee-saved prologue
contribution) is **M18** (softmax leaf-cleanup).

#### 4.4.1 File-local `.section __TEXT,__const` pool

The 11 `f32` constants (3 reduction + 8 Taylor coefficients) are emitted
once per assembly file from `walk_uir` when `uir.has_softmax()`, under
file-local `.L`-prefixed labels:

```asm
.section __TEXT,__const
.p2align 2
.Lexp_log2e: .long 0x3fb8aa3b    ; LOG2E = log2(e)
.Lexp_ln2hi: .long 0x3f318000    ; LN2_HI (exactly representable)
.Lexp_ln2lo: .long 0xb95e8083    ; LN2_LO (two-part split for cancellation)
.Lexp_c0:    .long 0x3f800000    ; C0 = 1.0
.Lexp_c1:    .long 0x3f800000    ; C1 = 1.0
.Lexp_c2:    .long 0x3f000000    ; C2 = 0.5
.Lexp_c3:    .long 0x3e2aaaab    ; C3 = 1/6
.Lexp_c4:    .long 0x3d2aaaab    ; C4 = 1/24
.Lexp_c5:    .long 0x3c088889    ; C5 = 1/120
.Lexp_c6:    .long 0x3ab60b61    ; C6 = 1/720
.Lexp_c7:    .long 0x39500d01    ; C7 = 1/5040
```

File-local labels do not collide when linking multiple NeuralForge object
files together. The pool is emitted once regardless of how many softmax
models appear in the file (the guard is at the `walk_uir` level). Constants
are loaded via `adrp`/`ldr` with `@PAGE`/`@PAGEOFF` relocations.

**`-inf` materialisation.** `fmov sN, #-inf` is invalid (8-bit FP-immediate
encoding doesn't include ±inf). The portable pattern is to load the bit
pattern (`0xFF800000` for f32) into a GPR and `fmov sN, wN`:

```asm
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16   ; w0 = 0xFF800000 = f32 -inf
    fmov    s8, w0
```

### 4.5 Dropout (aliasing, no asm; or copy-loop when model output)

Dropout at inference is identity. The buffer-assignment first-pass
(`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand_id)` for
dropout nodes. **No asm is emitted** for dropout — the dispatcher's
`StdOp::Dropout =>` arm is empty. Downstream ops reading dropout's output
resolve the alias chain through `resolve_loc` to the operand's actual
`BufferLoc`. Exception (M8): when a Dropout node IS `model.output`,
`assign_buffers` returns `BufferLoc::OutputReg` (the caller's `x2` pointer)
instead, and codegen emits an explicit copy-loop via
`ops/dropout.rs::emit_dropout_copy`. See the M8 codegen hardening section
for details.

### 4.6 Intermediate buffers (stack-allocated)

Non-terminal `Linear` and `Softmax` nodes whose results are consumed by
another op get a stack slot. The function prologue does
`sub sp, sp, #N` (with `N` rounded up to 16-byte alignment). The epilogue
does `add sp, sp, #N`. For sizes that don't fit a single 12-bit immediate,
the codegen uses the shifted-by-12 form (multiples of 4096) or
`movz/movk + sub sp, sp, x9` for the general case.

The largest M4b fixture (classifier with batch=32) needs ~97KB of stack —
well under macOS default thread stack of 8MB.

### 4.7 Non-leaf prologue/epilogue

The pre-emission analyzers `compute_is_leaf` and `compute_callee_saved`
classify each function. M4b has three layers conditionally included:

- **Callee-saved integer regs** — emitted iff `RegSet.x19_x23` (set when
  softmax is present):
  ```asm
  stp     x19, x20, [sp, #-16]!
  stp     x21, x22, [sp, #-16]!
  str     x23, [sp, #-16]!
  ```
  Symmetric reverse on the epilogue.

- **Callee-saved FP** — emitted iff `RegSet.d8_d9` (set when softmax is
  present):
  ```asm
  stp     d8, d9, [sp, #-16]!
  ```
  Symmetric reverse on the epilogue.

- **Non-leaf frame** — emitted iff `LeafKind::NonLeaf` (set when softmax is
  present, since softmax is the only op that emits `bl`):
  ```asm
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp
  ```
  Symmetric reverse on the epilogue.

Leaf functions with no intermediates (e.g., a single Linear terminal) emit
just `ret` — zero overhead. Each layer is independently included based on
its analyzer's output.

### 4.8 Per-model label namespacing

Multi-model fixtures (e.g. `pipeline_styles.nfl` with 3 models in one .s
file) would collide on labels like `.Lmm_i_0:` if every model used the
same naming. Each per-op emitter takes both `model_idx` and `op_idx`,
producing labels of the form:

- `.Lmm_i_<m>_<l>:`, `.Lmm_j_<m>_<l>:`, `.Lmm_k_<m>_<l>:` — Linear loops.
- `.Lrelu_<m>_<r>:` — Relu loop.
- `.Lsm_i_<m>_<s>:`, `.Lsm_max_<m>_<s>:`, `.Lsm_exp_<m>_<s>:`,
  `.Lsm_norm_<m>_<s>:` — Softmax passes.

For single-model fixtures the `model_idx` is always `0`, so labels look
like `.Lmm_i_0_0:`.

### 4.9 Fused linear → relu (with optional bias-add)

When the compiler's `FuseLinearRelu` UIR pass identifies a
`linear → relu` (or `linear[bias=true] → relu`) pattern with the
linear having a single consumer, it merges them into a single Linear
node with `fused_post_ops: vec![PostOp::Relu]`. The `emit_linear`
emitter consumes that field and produces:

```asm
    ; once at function-header time (before the matmul i-loop):
    fmov    s4, wzr             ; materialise 0.0 — needed by fmax post-op below

    ; ... (matmul i/j/k loops, accumulating sum in s0) ...
    ; ... (k-loop end) ...

    ; bias-add (if bias_offset.is_some()) — same as §4.3:
    ldr     s5, [x14, x4, lsl #2]
    fadd    s0, s0, s5

    ; M5a NEW: post-ops inline, between bias-add and store.
    ; For PostOp::Relu, the implementation emits one fmax per element:
    fmax    s0, s0, s4          ; relu — clamps negative to 0.0

    ; ... (store + j/i increments) ...
```

Order is fixed: `matmul → bias-add (if any) → post-ops → store`.
This recovers M4a's in-place relu pattern and saves one
intermediate buffer round-trip vs the unfused `Linear → Relu` chain
(§4.1 + §4.2).

The `fmov s4, wzr` materialisation happens **once** at function-header
time, conditional on `fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu))`
— not per-element. AArch64 `fmax` requires both operands in FP regs,
so `wzr` must be moved through `s4` first.

The post-op match block in `ops/linear.rs` is `#[allow(unreachable_patterns)]`-
wildcarded against future `PostOp` variants (see §5 for `LowerError::UnsupportedPostOp`).

### 4.10 Fused linear → softmax (row-wise)

When the compiler's `FuseLinearSoftmax` UIR pass identifies a
`linear → softmax` (or `linear[bias=true] → softmax`) pattern with the
linear having a single consumer (and an empty `fused_post_ops` — criterion 4
prevents `[Relu, SoftmaxRow]` stacks at the pass level), it merges them into a
single Linear node with `fused_post_ops: vec![PostOp::SoftmaxRow]`.

**Structural difference from elementwise post-ops.** Unlike `PostOp::Relu`
(which can be inlined element-by-element inside the j-loop), softmax requires
the full row max to be known before any element can be exponentiated. This
means the row-wise tail CANNOT be inlined inside the matmul j-loop. The
implementation uses a **two-pass i-loop structure**:

1. **Phase 1 (i-loop A — the matmul loop):** Runs first and writes the
   complete `[B, N]` output matrix to the dst buffer. For `PostOp::SoftmaxRow`,
   the inline post-op slot inside the j-loop is empty (`SoftmaxRow => {}`).
   Bias-add (if `bias=true`) still happens in this phase, before the store.
2. **Phases 2-4 (i-loop B — separate, runs after matmul completes):** A second
   i-loop sweeps each row for the three softmax passes.

**IMPLEMENTERS: do NOT attempt per-element softmax inlining inside the j-loop.**
The row max is not available until the entire row has been written by Phase 1.
Any attempt to fuse Phases 2-4 into the matmul j-loop will compute incorrect
results.

**Register convention (Phases 2-4).** All registers below are AAPCS64
callee-saved, saved by the function prologue when `compute_callee_saved`
returns `RegSet { d8_d9: true, x19_x23: true }`:

| Register | Role                                                       |
|----------|------------------------------------------------------------|
| `x19`    | outer row index `i`                                        |
| `x20`    | row base offset `i * N` (element units)                    |
| `x21`    | inner column index `j`                                     |
| `x22`    | src pointer (= `x12`, the matmul dst — in-place)           |
| `x23`    | dst pointer (= `x12`, same buffer — in-place)              |
| `s8`     | per-row maximum (callee-saved FP; saved as `d8`)           |
| `s9`     | per-row sum (callee-saved FP; saved as `d9`)               |

`x12` (the matmul dst pointer) is set before the matmul loop and is still
valid when Phases 2-4 begin. `x22` and `x23` are both set to `x12` at the
start of the softmax i-loop; the output buffer is touched in-place throughout.

**ABI notes.**
- `compute_is_leaf` returns `false` (i.e., `LeafKind::NonLeaf`) for any model
  containing `PostOp::SoftmaxRow` in `fused_post_ops`. Conservative in M17:
  the inline exp no longer emits `bl _expf`, but the loop still holds state
  in callee-saved registers and a non-leaf frame. Precise leaf reclassification
  is **M18**.
- `compute_callee_saved` requests `{ d8_d9: true, x19_x23: true }` — same
  as standalone `softmax`.
- `x6` (element offset, caller-saved) is recomputed after each `emit_exp_inline`
  call because the helper clobbers caller-saved registers (`x9/w11/w12/s1-s5`).

**Memory access per row.**
The dst buffer is accessed 6 times per row element across the four phases:

| Phase | Access     | Count per element |
|-------|------------|-------------------|
| 1     | write       | 1                 |
| 2     | read (max)  | 1                 |
| 3     | read + write (exp, in-place) | 2  |
| 4     | read + write (normalise, in-place) | 2 |

No separate softmax buffer is allocated. All phases share the single matmul
output buffer.

**`-inf` initialisation.** The row max `s8` is initialised to negative infinity
using the bit-pattern load (`0xFF800000`), not from `row[0]`. This matches
`emit_softmax` (§4.4) and avoids an off-by-one if the first element is the
maximum:

```asm
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16   ; w0 = 0xFF800000 = f32 -inf
    fmov    s8, w0
```

**Abbreviated asm sketch.** The label suffix `{lid}` is `{model_idx}_{linear_idx}`.

```asm
; ── Phase 1: matmul + optional bias-add (i-loop A) ──────────────────────────
; Standard M5b emit_linear shape: nested i / j / k loops.
; Writes out[i, 0..N] for each row i. PostOp::SoftmaxRow is a no-op here
; (the inline slot inside the j-loop is empty); store happens normally.

; ── Phases 2-4: row-wise softmax tail (i-loop B, after matmul) ───────────────
    mov     x22, x12               ; src ptr = dst ptr (in-place)
    mov     x23, x12               ; dst ptr = same buffer

    mov     x19, #0                ; i = 0  (callee-saved)
.Lfsmx_i_{lid}:
    cmp     x19, #B
    b.ge    .Lfsmx_i_end_{lid}

    mov     x8, #N
    mul     x20, x19, x8           ; x20 = i * N  (row offset in elements)

    ; ── Phase 2: row-max → s8 ────────────────────────────────────────────────
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16   ; s8 = -inf (bit pattern 0xFF800000)
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_{lid}:
    cmp     x21, #N
    b.ge    .Lfsmx_max_end_{lid}
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_{lid}
.Lfsmx_max_end_{lid}:

    ; ── Phase 3: exp(x - s8) in-place, sum → s9 ─────────────────────────────
    fmov    s9, wzr                ; s9 = 0.0
    mov     x21, #0
.Lfsmx_exp_{lid}:
    cmp     x21, #N
    b.ge    .Lfsmx_exp_end_{lid}
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    ; --- inline exp(x), x<=0 (M17) ---   ; (see ops/exp.rs emit_exp_inline)
    ; ... Cody-Waite reduction + degree-7 Horner + 2^z bit-trick ...
    ; clobbers x9/w11/w12/s1-s5 (non-loop-live caller-saved)
    ; --- end inline exp ---
    add     x6, x20, x21          ; x6 is caller-saved; recompute after inline exp
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lfsmx_exp_{lid}
.Lfsmx_exp_end_{lid}:

    ; ── Phase 4: normalise by s9 ─────────────────────────────────────────────
    mov     x21, #0
.Lfsmx_norm_{lid}:
    cmp     x21, #N
    b.ge    .Lfsmx_norm_end_{lid}
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_{lid}
.Lfsmx_norm_end_{lid}:

    add     x19, x19, #1
    b       .Lfsmx_i_{lid}
.Lfsmx_i_end_{lid}:
```

**Bias-aware fusion.** The `linear[bias=true] → softmax` pattern fuses
identically: Phase 1 includes the bias-add step (as in §4.3) before the store.
Phases 2-4 are unchanged — they operate on the post-bias output.

**Stacking constraints.** `FuseLinearSoftmax` criterion 4 requires the Linear's
`fused_post_ops` to be empty before the pass will fuse a Softmax onto it. This
is the only guard against `[Relu, SoftmaxRow]` stacks — there is no defensive
check inside `emit_linear`. The pass-level criterion is sufficient because
`FuseLinearRelu` runs before `FuseLinearSoftmax` in `default_pipeline()`, and a
Linear that has already been tagged with `[Relu]` fails criterion 4 and is left
alone by `FuseLinearSoftmax`.

**Label namespace.** Fused-softmax labels use the `.Lfsmx_*` prefix to avoid
collision with standalone `.Lsm_*` labels from `emit_softmax` if both are
present in the same model (e.g. a model with two softmax layers where only
one is preceded by a fusable Linear).

---

## 5. Errors

`profiles_arm64::lower` returns `Result<Asm, LowerError>`. `LowerError` is
`#[non_exhaustive]`; consumers must keep a `_ => ...` arm. Variants in M6:

| Variant                      | When                                                                                              |
|------------------------------|---------------------------------------------------------------------------------------------------|
| `UnsupportedOp { op, span }` | Defensive: codegen doesn't know how to lower `op`. All M5b ops are supported; M5c made `StdOp` `#[non_exhaustive]`, so this variant is now reachable through the wildcard arm in `walk_model` and `classify_op` for any future `StdOp` variant before codegen catches up. |
| `ShapeNotConcrete { span }`  | Defensive: shape wasn't fully resolved by `ir::build`. Should be unreachable.                    |
| `UnsupportedPostOp { op, span }` | M5a: post-op variant not supported by this profile. M6 added `PostOp::SoftmaxRow` as a concrete, handled implementation — this variant never fires for `SoftmaxRow` in the default pipeline. The wildcard arm in `ops/linear.rs` remains as a forward-compat guard: it fires for any future `PostOp` variant (e.g., `Tanh`, `Gelu`) that lands in `compiler::PostOp` before this profile catches up. Same pattern as `UnsupportedOp`. |

Duplicate model name detection moved up to `compiler::ir::build` in M4b
(see `BuildErrorKind::DuplicateModelName`); profiles no longer see
duplicate-name UIRs.

The CLI (`nflc compile`) renders these via the `render_error_with_snippet`
helper from M3c — same `error: ... --> file:line:col ... ^` format as
parser/IR errors. For `BuildErrorKind::DuplicateModelName` the helper also
emits a trailing `note: previously defined at <file>:<line>:<col>`
plain-text line.

### 5.5. Runtime dependency: libm (removed in M17)

**M17: softmax no longer calls libm.** The `bl _expf` instruction was
replaced by `emit_exp_inline()` — a Cody-Waite + degree-7 Taylor
polynomial inlined directly at each exp site. The constant pool lives in
`.section __TEXT,__const` (see §4.4.1). No external symbol is referenced
and no `-lm` link flag is needed. A compiled arm64 softmax binary is now
genuinely bare-metal.

The FFI save/restore block and the callee-saved prologue contribution are
RETAINED in M17 — their removal is deferred to **M18** (softmax
leaf-cleanup), which will also flip `compute_is_leaf` to `true` for
softmax models and update the inspect goldens.

---

## 6. Adding a new op

To add an op to the `arm64` profile (e.g. `tanh`, `sigmoid`):

1. Add an arm in `profiles/arm64/src/codegen.rs::classify_op` returning
   `Ok(())` for the new op (or returning a `LowerError` if it should be
   rejected).
2. Add a per-op emitter as `profiles/arm64/src/ops/<op>.rs` exposing
   `pub fn emit_<op>(b: u64, ..., model_idx: usize, op_idx: usize, ...)`.
3. Re-export the emitter from `profiles/arm64/src/ops/mod.rs`.
4. Add a dispatch arm in `walk_model`'s op-loop calling the new emitter,
   passing `model_idx` and the per-op counter.
5. If the op needs callee-saved registers or a `bl`, update
   `compute_is_leaf` / `compute_callee_saved` in `buffer.rs`.
6. If the op is in-place (like Relu) or no-op (like Dropout), update
   `assign_buffers` in `buffer.rs` to return the appropriate `BufferLoc`.
7. Add unit tests in `profiles/arm64/src/tests.rs` asserting the asm
   contains the expected instructions.
8. Add an integration test in `profiles/arm64/tests/integration.rs` if the
   op participates in end-to-end runnable code.
9. Update this doc's §3 table.

---

## 7. Adding a new architecture profile

To add a new profile (e.g. `x86_64`, `riscv64`):

1. Create `profiles/<arch>/Cargo.toml` mirroring `profiles/arm64/Cargo.toml`.
   `[dependencies] compiler = { path = "../../compiler" }`.
2. Add `"profiles/<arch>"` to the workspace `members` in `/Cargo.toml`.
3. Implement the same public surface as `profiles_arm64` —
   `pub fn lower(&Uir) -> Result<Asm, LowerError>` plus the `Asm`,
   `FnSig`, `ParamSlot`, `ParamKind`, `LowerError` types. (M5+ may extract
   a shared `profile-api` crate when the second profile lands; for M4b
   that's premature.)
4. Add a dispatch arm in `nflc/src/main.rs::run_compile` for the new
   `--profile <arch>` value.
5. Mirror this guide as `docs/profile_guide/<arch>.md`.

---

## 8. Limitations (M6)

- **No SIMD.** Scalar throughout. NEON is M7+.
- **No matmul tiling / cache blocking.** Three-nested-loop matmul;
  `mul` for indexing; per-element load/store. Performance optimisation
  is M7+.
- **No batched / vectorised exp.** The inline exp is scalar (one `s0`
  element at a time); SIMD or batched evaluation is M7+/NEON.
- **Two `PostOp` variants are supported by `emit_linear`: `Relu`
  (Elementwise — inline `fmax s0, s0, s4` inside the j-loop;
  see §4.9) and `SoftmaxRow` (RowWise — three sweeps after the
  matmul i-loop; see §4.10). Stacking variants (e.g. `[Relu, SoftmaxRow]`)
  is prevented at the pass level by `FuseLinearSoftmax` criterion 4
  (the Linear's `fused_post_ops` must be empty before the pass will
  fuse a Softmax onto it).**
- **Graph-level dead-op elimination is limited to `EliminateDropout`.**
  No general DCE pass. Other no-op shapes (e.g. `linear[out_dim=K] →
  linear[out_dim=N]` collapsing via matmul-of-matmul) are M7+.
- **No softmax leaf optimization.** Softmax models are conservatively
  classified as non-leaf in M17 (the loop still uses callee-saved regs
  and a frame). Precise leaf reclassification is M18.
- **Single-snippet error rendering for duplicate-model-name.** The
  `note: previously defined at` line is plain text, not a second `^`
  snippet. Multi-snippet (rustc-style) upgrade is M4c-or-later (still
  applies).
- **Integration tests run only on aarch64 hosts with `cc` available.**
  Skip with logged reason elsewhere.

---

## M8 codegen hardening

### Dropout-as-output copy

Dropout is identity at inference time, and `assign_buffers`
returns `BufferLoc::Alias(operand)` for any Dropout node that is
NOT the model output — downstream ops read the operand's buffer
directly, no asm needed. When a Dropout node IS `model.output`,
however, alias-redirection no longer applies: `assign_buffers`
returns `BufferLoc::OutputReg` (the caller's `x2` pointer), and
codegen must explicitly copy the operand's buffer into it. The
`StdOp::Dropout` arm in `codegen.rs::walk_model` branches on
`dst_loc` and emits a copy-loop via `ops/dropout.rs::emit_dropout_copy`
in this case (mirror of `emit_relu`'s structure minus `fmax`).
This path is exercised only with `--no-passes` and dropout placed
at the model output; the default pipeline's `EliminateDropout` pass
removes the dropout before codegen sees it.

### Dim-immediate uniformity

ARM64 `cmp Xn, #imm` encodes a 12-bit immediate (0-4095, optionally
shifted by 12); `mov Xn, #imm` encodes 16-bit (0-65535). All loop-
bound and stride dimensions in matmul, relu, softmax, and the fused
RowWise softmax tail flow through `asm::emit_imm32` (movz + optional
movk) instead of literal-imm encoding. Two placement strategies:

- **Group A (bl-free loops)**: hoist materialise once before the
  loop label, register-form `cmp` inside. Matmul body uses three
  distinct registers (x10 ← b, x15 ← n, x16 ← k); stride-load movs
  reuse the hoisted regs (`mov x8, x16` etc) instead of re-
  materialising. Inner-loop cmp has zero runtime cost.
- **Group B (inline-exp loops)**: re-materialise into x10 at
  each loop top (after label, before cmp). The inline exp (M17)
  clobbers caller-saved registers including x10, so hoisting
  outside the loop is impossible without expanding the prologue's
  callee-saved set (deferred to M18). 1-2 movz/movk per iteration
  is < 1% overhead vs the cost of the exp body itself.

`emit_imm32` asserts `value <= u32::MAX as usize`, providing a
clear failure mode for hypothetical dimensions beyond 4 billion
elements (~1000× any realistic NN dim).

---

## M10 ops

M10 adds two new ops (`StdOp::Matmul`, `StdOp::MulScalar`) and generalises
softmax dispatch from rank-2 to rank ≥ 2. The new ops live in their own
`ops/` modules; the existing `emit_linear` is **unchanged** (§6.1
architectural invariant — adding multi-dim matmul does NOT alter the
hot path for `linear`-as-2D).

### `emit_matmul` — multi-dim matmul over rank ≥ 2 inputs

Source: `profiles/arm64/src/ops/matmul.rs`. Emitted for every `StdOp::Matmul`
node — typically two per attention block (Q · Kᵀ for scores, attn · V for
the attended values).

**Outer-loop wrapper.** `Matmul` accepts inputs of any rank ≥ 2; the
trailing two dims `[..., M, K]` × `[..., K, N]` form the inner kernel,
and the leading dims (count = `leading_count = product(shape[..rank-2])`)
are iterated with an outer loop. For 2D inputs `leading_count == 1` and
the outer loop runs once (no measurable overhead vs an unwrapped 2D
matmul).

```asm
    ; per-outer-iteration setup: a_slice = M*K, b_slice = K*N,
    ; dst_slice = M*N (in floats); slice base ptrs = base + idx*slice*4.
    mov     x17, #0                ; outer_idx
.Lmm4d_outer_<m>_<l>:
    cmp     x17, #leading_count
    b.ge    .Lmm4d_outer_end_<m>_<l>
    ; ... A_slice ptr -> x1, B_slice ptr -> x2, DST_slice ptr -> x4 ...
    ; ... triple-nested i/j/k FMA matmul over the slice ...
    add     x17, x17, #1
    b       .Lmm4d_outer_<m>_<l>
.Lmm4d_outer_end_<m>_<l>:
```

**FMA inner triple-loop.** Inside each outer iteration, the kernel is
the standard scalar AArch64 matmul (cf. §4.1):

```asm
    fmadd   s0, s1, s2, s0      ; sum += a[i,k] * b[k,j], single rounding
```

For `transpose_b=true`, the b-pointer load uses `[k, j]` indexing as
`weights[j*K + k]` (i.e. swap the roles of `K` and `N` in the b-stride
calculation). This is the canonical attention-scores pattern
(`q · kᵀ`).

**Base-pointer invariance invariant.** The materialised base pointers
`x11` (A), `x13` (B), `x12` (DST) are emitted **once** before the outer
loop and MUST NOT be mutated inside it. Per-outer-iteration slice
pointers go into `x1`, `x2`, `x4`. The choice of `x1`/`x2` (the FFI
params/output registers!) is deliberate scratch reuse. To preserve the
FFI calling convention for downstream emitters (e.g. a subsequent
`emit_linear` or `emit_matmul` that re-materialises from the original
`x1`/`x2`), `emit_matmul` spills `x1`/`x2` to the stack via
`stp/ldp` around the outer loop:

```asm
    stp     x1, x2, [sp, #-16]!     ; spill at function-body entry
    ; ... outer loop body ...
    ldp     x1, x2, [sp], #16       ; restore at function-body exit
```

This idiom is the M10 application of the M9 lesson: any emitter that
clobbers FFI input registers must save/restore them around its body.
The fix landed via fixup `00b6f82`.

### `emit_mulscalar` — flat per-element scalar multiply

Source: `profiles/arm64/src/ops/mulscalar.rs`. Emitted for every
`StdOp::MulScalar` node — one per attention block (the `1/√d` scaling
of the `Q · Kᵀ` scores before softmax).

**Pre-loaded scalar via `movz/movk + fmov`.** The scalar (an f32 bit
pattern computed by the dispatcher from the AttrValue) is materialised
into `s4` *once* before the loop:

```asm
    movz    w9, #<lo16>             ; lo 16 bits of f32 bit pattern
    movk    w9, #<hi16>, lsl #16    ; hi 16 bits (skipped if zero)
    fmov    s4, w9                  ; scalar -> s4
```

**In-place flat loop.** With `BufferLoc::Alias` plumbing (the same
machinery that makes `Dropout` a no-op) `src_loc == dst_loc`, so the
loop reads and writes the same buffer:

```asm
    ldr     s0, [x11, x3, lsl #2]
    fmul    s0, s0, s4
    str     s0, [x12, x3, lsl #2]   ; x12 == x11 under Alias
```

**`AttrValue::Float → f32` truncation contract.** `AttrValue::Float`
holds an `f64`. The dispatcher (`codegen.rs`) is responsible for
truncating to f32 (`val as f32`), then transmuting to `u32` bits via
`f32::to_bits` and passing the `u32` to `emit_mulscalar`. The emitter
itself never sees the `f64` — this keeps the per-op file pure asm
formatting and concentrates lossy-conversion semantics in one place
(spec §6.5).

### Softmax dispatch — generalised to rank ≥ 2

`emit_softmax` itself is structurally unchanged at the body level (the
3-pass max → exp → normalise is per-row, regardless of rank), but the
dispatcher in `codegen.rs::walk_model` now computes:

- `b = product(shape[..rank-1])` — total number of rows across all leading dims.
- `k = shape[rank-1]` — row width (last dim).

For 2D inputs this collapses to the M3/M4b shape (`b = batch`,
`k = features`). For the 4D attention case the leading dims (batch,
heads, seq) flatten into a single outer counter — softmax is applied
to the **last axis only**, which is the standard transformer
convention.

**Post-Group-10 codegen fix.** During M10 end-to-end FFI integration
(commit `feb65de`, Group 10) we discovered that `emit_softmax` itself
was clobbering `x0`/`x1`/`x2` (the FFI input/params/output registers)
because `bl _expf` was caller-saved per AAPCS64 — caller-saved spans
include `x0..x18`. Any downstream emitter that re-materialises from
`InputReg`/`OutputReg` after softmax (e.g. attention's second
`emit_matmul` reading from the same input `x` as the first) was
silently miscompiling. Fix: `emit_softmax` now spills `x0/x1/x2` via
two `stp` pairs at function-body entry and matches with two `ldp`
pairs at exit — same pattern as `emit_matmul`. This is the M9 lesson
(commit `ecb69ac`, `c3ff521`) re-applied to the only standalone op
that emits a call. **M17 retained this spill block** (the inline exp
still clobbers caller-saved scratch; the block's purpose is unchanged).
Its removal is M18.

### What is unchanged

- `emit_linear` body is byte-identical to M9. `linear`-as-2D models
  (every M3-M9 fixture) lower exactly as before.
- `emit_relu`, `emit_dropout_copy`, the prologue/epilogue analyzers,
  the buffer-allocation strategy, and the `compute_callee_saved` /
  `compute_is_leaf` predicates — all unchanged. M10 added two new ops
  and one dispatcher generalisation; the surrounding scaffolding is
  the same.

---

## Multi-Input ABI (M12)

M12 adds per-profile `AbiContext` that extends the calling convention to N∈{1..4}
inputs, enabling models that take separate `q`/`k`/`v` tensors (or any other
multi-input pattern) as distinct ABI arguments without packing them into a single
buffer.

### Register layout (AAPCS64)

Under AAPCS64, pointer arguments are passed in `x0..x5` in order. The M12
convention assigns registers as follows for N inputs:

| N | x0  | x1  | x2  | x3  | x4     | x5  |
|---|-----|-----|-----|-----|--------|-----|
| 1 | in₀ | params | out | — | — | — |
| 2 | in₀ | in₁ | params | out | — | — |
| 3 | in₀ | in₁ | in₂ | params | out | — |
| 4 | in₀ | in₁ | in₂ | in₃ | params | out |

`params` is always the last input-class register; `out` follows immediately after.
N=1 is the legacy single-input layout and is fully backward-compatible with all
pre-M12 fixtures.

The C prototype for N=3 looks like:

```c
void nfl_forward_<ModelName>(
    const float* in0,
    const float* in1,
    const float* in2,
    const float* params,
    float*       output
);
```

### `AbiContext` — per-arity accessor struct

`profiles/arm64/src/abi.rs` exports `AbiContext { n_inputs: usize }` with the
following arity-aware accessors:

- `input_reg(i: usize) -> &'static str` — returns `"x0"` through `"x3"` for
  `i` in `0..n_inputs`. Panics if `i >= n_inputs`.
- `params_reg() -> &'static str` — returns `"x1"` through `"x4"` (= `x[n_inputs]`).
- `output_reg() -> &'static str` — returns `"x2"` through `"x5"` (= `x[n_inputs+1]`).
- `ffi_save_set() -> &[&'static str]` — returns the slice of registers that
  must be preserved across a multi-emitter body (all ABI regs that `emit_matmul`
  might clobber when reusing them as per-iteration slice pointers).
- `materialise_ptr(b: &mut String, dst: &'static str, loc: &BufferLoc, ...)` —
  emits the materialise-pointer sequence appropriate for the given `BufferLoc`
  variant (`InputReg(i)` loads from `input_reg(i)`; `ParamsReg` loads from
  `params_reg()`; `OutputReg` loads from `output_reg()`).

`walk_model` constructs exactly one `AbiContext { n_inputs: model.inputs.len() }`
and threads `&abi` through every op-emitter call. Models with more than 4 inputs
return `Err(LowerError::TooManyInputs { n: model.inputs.len() })`.

### `BufferLoc::InputReg(usize)` — input index as buffer location

`BufferLoc::InputReg` now carries an index (instead of being a unit variant).
`assign_buffers` maps `model.inputs[i]` → `BufferLoc::InputReg(i)`. The index
is forwarded to `abi.materialise_ptr(...)` so each input gets the correct ABI
register without any per-emitter hardcoding.

### Stack alignment for FFI calls

When `ffi_save_set()` is non-empty, `emit_ffi_save` spills each register in
the set using `stp` pairs. If the set has an **odd** cardinality (so pairing
would leave one register without a partner), an extra `str xzr, [sp, #-8]!`
padding store is prepended to maintain the 16-byte aligned SP invariant. The
SP delta across the entire save/restore block is always a multiple of 16.

`emit_ffi_restore` reverses `emit_ffi_save` in strict **LIFO** order: the
last pair saved is the first pair restored via `ldp`, mirroring the descending-
stack `stp x, y, [sp, #-16]!` / ascending `ldp x, y, [sp], #16` idiom.

### `emit_matmul` scratch register layout (M12 rework)

In M10, `emit_matmul` reused `x1`/`x2`/`x4` (FFI ABI registers) as per-outer-
iteration slice pointers, then spilled them via `stp x1, x2 / stp x4, xzr` at
function-body entry. This spill block is **removed** in M12.

M12 moves the per-outer-iteration slice pointers to scratch registers that are
always outside the ABI window regardless of N:

| Register | Role in `emit_matmul` |
|----------|-----------------------|
| `x9`     | A-base pointer (materialised once before outer loop) |
| `x10`    | B-base pointer (materialised once before outer loop) |
| `x11`    | DST-base pointer (materialised once before outer loop) |
| `x12`    | A_slice pointer (per outer-iteration) |
| `x13`    | B_slice pointer (per outer-iteration) |
| `x14`    | DST_slice pointer (per outer-iteration) |
| `x15`    | outer loop index |
| `x16`    | leading_count immediate |
| `x17`    | inner loop indices / scratch |

Because `x9`–`x17` are all caller-saved under AAPCS64 and none of them overlap
with the ABI argument window (`x0`–`x5`), no save/restore block is needed around
the outer loop. The materialise-ptr-first ordering rule (applied at `walk_model`
call-site level) ensures base pointers are loaded before any slice arithmetic
clobbers scratch registers.

The old `stp x1, x2, [sp, #-16]!` / `ldp x1, x2, [sp], #16` outer-loop spill
block from M10 is fully eliminated. Any model containing `emit_matmul` now emits
fewer instructions per matmul op.

---

## M13 ops

### `emit_add` (`profiles/arm64/src/ops/add.rs`)

Flat elementwise tensor addition: `dst[i] = a[i] + other[i]` over
`total_elements = product(shape)`.

**Register layout:**
- `x9` — `a` pointer (caller-saved scratch, materialised via
  `AbiContext::materialise_ptr`).
- `x10` — `other` pointer (same).
- `x11` — `dst` pointer (same).
- `x12` — loop counter (caller-saved scratch).
- `x13` — total_elements bound (caller-saved scratch).
- `s0`, `s1`, `s2` — load-load-add-store scalar FP registers.

**No callee-saved register usage.** No FFI save/restore (no `bl _expf`).

Inner loop (per iter):
```
ldr     s0, [x9, x12, lsl #2]    ; load a[i]
ldr     s1, [x10, x12, lsl #2]   ; load other[i]
fadd    s2, s0, s1
str     s2, [x11, x12, lsl #2]   ; store dst[i]
add     x12, x12, #1
```

Closest existing template: `emit_mulscalar` (M10). The shell is
identical; `emit_add` reads two input pointers (a, other) where
`emit_mulscalar` reads one input + a pre-loaded scalar in `s4`.

### M13 emit_linear ABI register save (N≥2)

Pre-Task-5 fix: `emit_linear` previously used `x3`/`x4`/`x5` as
i/j/k loop counters, which silently overlap with ABI argument
registers at N≥2 (output_reg = `INPUT_REGS[n_inputs+1]`; at N=2
that's `x3`, at N=3 `x4`, at N=4 `x5`). M12 missed this because all
M12 multi-input fixtures used matmul-only; M13's `residual_add.nfl`
is the first multi-input fixture with a `linear` op and surfaced the
bug via SIGSEGV in the FFI test.

The fix uses `stp`/`ldp` save/restore around the i-loop body: at
N=2 push `(x3, xzr)`; at N=3 push `(x3, x4)`; at N=4 push `(x3, x4)`
plus `(x5, xzr)`. Restore in strict LIFO order. Inner-loop body is
unchanged.

Save/restore was chosen over relocating counters to `x9`-`x15`
because `emit_linear`'s bias paths and fused `PostOp::SoftmaxRow`
dispatch already touch `x9`-`x16` extensively — a counter rename
would cascade through too many sites. Trade-off: 2-4 extra
instructions per linear op invocation at N≥2 vs a smaller, lower-
risk diff.

Cross-reference: same class of bug as Task 1's x86_64 `emit_matmul`
fix (j-counter `%r9` collided with output_reg at N=4); resolved on
x86_64 via register relocation (`%rbp`), on arm64 via save/restore.

---

## M14 ops

### `emit_layernorm` (`profiles/arm64/src/ops/layernorm.rs`)

Layer normalization: per-row 3-pass kernel (mean → variance + inv_std →
normalize + optional affine). Native `fsqrt` — no FFI dependency. Leaf
function (no `bl` calls).

**Register plan:**

| Register | Role                                                                    |
|----------|-------------------------------------------------------------------------|
| `x6`     | Bound scratch (clobbered by every `emit_imm32` call)                   |
| `x9`     | Per-row input pointer (`x_in`) — recomputed at top of each row         |
| `x10`    | Per-row output pointer (`x_out`) — recomputed at top of each row       |
| `x11`    | Inner loop counter (`x_j`)                                              |
| `x12`    | Outer loop counter (`x_i`)                                              |
| `x13`    | γ base pointer (affine only)                                            |
| `x14`    | β base pointer (affine only)                                            |
| `x16`    | `src_base` (materialised once via `materialise_ptr`; lives entire function) |
| `x17`    | `dst_base` (materialised once via `materialise_ptr`; lives entire function) |
| `s0`     | `s_acc` (accumulator); reused as `s_var` at end of Pass 2              |
| `s1`     | `s_mean` (live Pass 2 + Pass 3)                                         |
| `s2`     | `s_inv_d` (1/D constant); **reused as `s_b`** in Pass 3 affine path; rematerialised inline (`movz/movk/fmov`) at end of each row's Pass 3 — 3 instructions/row, identical encoding cost to the entry materialisation |
| `s3`     | `s_eps` (1e-5; live across outer batch loop)                            |
| `s4`     | `s_one` (1.0; live across outer batch loop)                             |
| `s5`     | `s_inv_std` (Q4 constraint — held through Pass 3; not recomputed per element) |
| `s6`     | `s_t` (per-element temp)                                                |
| `s7`     | `s_g` (γⱼ load — affine only)                                          |

**AAPCS64 register safety note.** All scratch registers are in the caller-saved
ranges: `x6`, `x9`–`x17` (GPRs), `s0`–`s7` (FP, lower 32 bits of `v0`–`v7`).
Registers `s8`–`s15` (lower 32 bits of `v8`–`v15`) are callee-saved per
AAPCS64 §6.1.2 and are **intentionally avoided** — writing them in a leaf
function without `stp`/`ldp` save/restore would silently corrupt the caller's
`v8`–`v15`. The `s_b` slot (β load in affine path) reuses `s2` after
`s_inv_d` consumption to stay within `s0`–`s7`, at the cost of a 3-instruction
reload of `s2` (= `s_inv_d`) at the end of each row's Pass 3 when affine is
enabled.

**Note on `x16`/`x17`:** AAPCS64 designates these as intra-procedure-call
scratch (IP0/IP1), used by linker stubs across `bl` calls. Because
`emit_layernorm` has no `bl` calls (leaf function), `x16`/`x17` are
effectively free for op use.

**3-pass structure:**

1. **Pass 1 — mean:** Load each element, accumulate into `s0` (`s_acc`), then
   `s_mean = s_acc * s_inv_d`. `fmadd` not used — accumulation is a plain
   `fadd` to keep the pass structurally parallel with Pass 2.
2. **Pass 2 — variance + inv_std:** For each element, compute `(xⱼ − μ)²`
   and accumulate. Then `s_var = s_acc * s_inv_d + s_eps`; `s_var = fsqrt(s_var)`;
   `s_inv_std = s_one / s_var` (single divide after sqrt; not per-element).
3. **Pass 3 — normalize + optional affine:** For each element,
   `s_t = (xⱼ − s_mean) * s_inv_std`; if affine: `s_t = s_t * γⱼ + βⱼ`.
   Store `s_t` to output.

**Implicit cost — `s_inv_d` rematerialisation.** When affine is enabled, `s2`
(`s_inv_d`) is consumed as `s_b` during the Pass 3 affine multiply-accumulate.
Rematerialisation of `s2` at the end of each row's Pass 3 costs 3 instructions
(`emit_f32_const` emits `movz w9, #lo` / optional `movk w9, #hi, lsl #16` /
`fmov s2, w9`). This is negligible vs the O(D) per-row work.

**Constants materialisation.** `s_eps` (1e-5 as f32 bits), `s_one` (1.0 as f32
bits), and `s_inv_d` (1.0/D as f32 bits) are materialised inline once before
the outer batch loop via the private `emit_f32_const` helper (`movz w9, #lo`
+ optional `movk w9, #hi, lsl #16` + `fmov s_dst, w9`). No `.rodata` pool, no
`adrp`/`ldr` chain — mirrors the inline materialisation pattern softmax uses
for its `-inf` constant. The `w9` GPR is safe as a temporary at the
materialisation sites (caller-saved; only briefly live for the `fmov` bridge).

**Leaf function discipline.** No `bl` — no FFI save/restore, no non-leaf frame
record, no callee-saved register save. `compute_is_leaf` returns
`LeafKind::Leaf` for any model that contains only LayerNorm (and no Softmax).
`compute_callee_saved` is unchanged by M14.

**Validated at N=1..2.** M14 fixtures (`layernorm_no_affine.nfl`,
`layernorm_affine.nfl`, `pre_ln_block.nfl`) use N=1 and N=2. Higher-N is
structurally safe on arm64 because the register plan lives entirely in
`x6`/`x9`–`x17` + `s0`–`s7`, which never overlap the ABI argument window
(`x0`–`x5`) at any N ∈ {1..4}.

---

## Inspection output (M16 / A3)

`nflc inspect <file.nfl> --profile arm64` runs the same per-profile
analyzers that `nflc compile` runs (`assign_buffers`,
`compute_callee_saved`, `compute_is_leaf`), packages the result as a
structured `profile_api::Inspection`, and renders it to text. Both
commands run the default pass pipeline by default; pass `--no-passes`
or `--passes <list>` to skip / filter (same semantics as `compile`).

The renderer produces a header line + per-model summary + per-node table.

Field reference:
- **`loc=`** — output buffer placement: `InputReg(i)` (the i-th input's
  ABI register, mapped via `AbiContext::input_reg(i)`), `OutputReg`
  (the output ABI register at `INPUT_REGS[n_inputs + 1]`),
  `StackOffset(N)` (`[sp + N]` in the model's intermediate frame), or
  `Alias(nK)` (consumer reads from node K's buffer directly — no asm
  emitted for this node by `assign_buffers`-aliased ops like `relu`,
  `dropout`, `mul_scalar`).
- **`out=`** — logical output bytes (`element_count * 4`). Aliased
  nodes still report logical bytes; physical placement is captured by
  `loc=`.
- **`params=`** — for `Linear` and `LayerNorm[affine=true]`: floats
  consumed from the packed `params` buffer (weights + bias for Linear,
  γ + β for LayerNorm). Other ops omit this field.
- **`callee-saved`** (per-model) — registers saved in the prologue for
  this function. `d8-d9` and `x19-x23` appear when `UirModel::has_softmax()`
  (standalone Softmax or fused SoftmaxRow), whose loop holds state in
  those registers; empty otherwise.
- **`leaf`** — `yes` iff `!UirModel::has_softmax()`. Conservative:
  softmax models stay non-leaf through M17's exp-inline (precise
  reclassification is M18). Drives whether `x29`/`x30` are saved.

See `docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md`
for the full schema and design rationale.
