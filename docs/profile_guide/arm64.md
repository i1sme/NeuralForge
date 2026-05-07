# `arm64` Profile — AArch64 Scalar Codegen

> **Status:** M8 complete (NFL v0.1). Lowers `linear` (with or without
> `bias=true`), `relu`, `dropout` (no-op pass-through at inference, or
> explicit copy-loop when dropout IS model output — see §M8),
> `softmax` (numerically stable 3-pass via libm `expf`), and now
> `linear → softmax` fused via `PostOp::SoftmaxRow` to native AArch64
> Mach-O assembly. The compiler runs the default UIR-pass pipeline
> (`EliminateDropout` + `FuseLinearRelu` + `FuseLinearSoftmax`) before
> lowering, so dropout-containing models reach the profile already with
> dropout removed, `linear → relu` (with or without bias) fuse to a
> single Linear with `fused_post_ops: [Relu]`, and `linear → softmax`
> (with or without bias) fuse to a single Linear with
> `fused_post_ops: [SoftmaxRow]`. All 5 M3 positive fixtures + the M4a
> fixture run end-to-end via FFI; bit-exact equivalence between fused
> and unfused asm proven on classifier.nfl and mixed_args.nfl
> integration tests. M8 adds dim-immediate uniformity via
> `asm::emit_imm32` (production-scale dims now compile correctly) and
> the dropout-as-output copy-loop fix.
> **Authoritative source:** `profiles/arm64/src/` and the M4a/M4b/M5a/M5b/M6/M7/M8
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
`bl`) or non-leaf (softmax → `bl _expf`); the prologue/epilogue is built
conditionally per model based on which ops are present.

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
| `Softmax`                  | ✅        | Numerically stable 3-pass (max → exp → normalise), `bl _expf` from libm. With `--no-passes` or `--passes` filter excluding `fuse_linear_softmax`: emitted as a standalone function via `emit_softmax` (labels `.Lsm_*`; see §4.4). Default pipeline (M6+): fused into the preceding Linear's `emit_linear` via `PostOp::SoftmaxRow` (row-wise tail; labels `.Lfsmx_*`; see §4.10). |
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

### 4.4 Softmax (per-row 3-pass, libm `expf`)

Per row `i`, three passes over `K` elements:

1. **Max scan:** `s8 = max(row)`, initialised to `-inf`.
2. **Exp + sum:** for each `k`, `output[i,k] = expf(input[i,k] - s8)` and
   `s9 += output[i,k]`.
3. **Normalize:** for each `k`, `output[i,k] /= s9`.

`s8` (per-row max) and `s9` (per-row sum) are AAPCS64 callee-saved (lower
64 bits of `v8`/`v9`). Function prologue saves `d8`/`d9` when
`compute_callee_saved` returns `RegSet { d8_d9: true }`.

**Caller-saved x-registers across `bl _expf`.** Per AAPCS64, `x0..x18` are
caller-saved and may be clobbered by `_expf`. Loop state that must survive
the call (`i`, `row_base`, `k`, `src` pointer, `dst` pointer) lives in
callee-saved `x19`..`x23`. The element offset (`x6`) is recomputed after
each call. The function prologue saves `x19`..`x23` when `RegSet.x19_x23`
is set (true iff softmax is present in the model).

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
  containing `PostOp::SoftmaxRow` in `fused_post_ops`, because Phase 3 emits
  `bl _expf`.
- `compute_callee_saved` requests `{ d8_d9: true, x19_x23: true }` — same
  as standalone `softmax`.
- `x6` (element offset, caller-saved) is recomputed after each `bl _expf`
  call because `_expf` may clobber all caller-saved registers (`x0..x18`).

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
    bl      _expf                  ; clobbers x0..x18, s0..s7
    add     x6, x20, x21          ; x6 is caller-saved; recompute after bl
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

### 5.5. Runtime dependency: libm

The softmax codegen emits `bl _expf`, which resolves to libm's `expf`
symbol at link time. On macOS and Linux, `cc` links libm by default. Bare-
metal targets without libm need a separate profile (M7+) — Taylor-series
`exp` implementation is reserved for that profile. The `arm64` profile
assumes POSIX with libm.

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
- **`bl _expf` per softmax element.** No batched / vectorised exp.
  M7+.
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
- **libm `expf` is the only `expf` source** for both standalone `softmax`
  and `SoftmaxRow` post-op. Bare-metal targets requiring a
  Taylor/minimax `expf` are M7+ work (spec §12 OQ-3).
- **No bare-metal target.** Requires libm at link time. M7+ for a
  Taylor-series-`exp`-based bare-metal profile.
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
- **Group B (bl-containing loops)**: re-materialise into x10 at
  each loop top (after label, before cmp). `bl _expf` clobbers
  caller-saved registers including x10, so hoisting outside the
  loop is impossible without expanding the prologue's callee-saved
  set (deferred). 1-2 movz/movk per iteration is < 1% overhead vs
  the cost of `bl _expf` itself.

`emit_imm32` asserts `value <= u32::MAX as usize`, providing a
clear failure mode for hypothetical dimensions beyond 4 billion
elements (~1000× any realistic NN dim).
