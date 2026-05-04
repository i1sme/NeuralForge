# Milestone 4b — `profiles/arm64` op coverage (vertical slice 2) — Design

> **Status:** Brainstormed and approved 2026-05-04. To be implemented in the
> `claude/m4b-arm64-coverage` worktree.
> **Source:** This spec captures the M4b brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M4a shipped the first concrete codegen profile (`profiles/arm64`) with the
minimal honest end-to-end path: `input → linear[N] → relu` lowered to native
AArch64 assembly, callable via FFI through `cc -shared` + `dlopen`. PR #8 +
PR #9 (CI) merged into main; baseline 118 tests green.

M4b extends `profiles/arm64` to cover **all five M3 positive fixtures**:
adds `linear` with `bias=true`, `dropout` (no-op pass-through at inference),
and `softmax` (numerically stable 3-pass with libm `expf`). Closes M4 functional
coverage; M4c handles polish (profile-guide updates, snapshot tests if needed,
maybe a small optimisation pass).

**M4b also introduces** four cross-cutting infrastructure changes that any
M4b op needs:
- New ABI: single packed `params` buffer with typed slot metadata (`FnSig.params_layout`).
- Stack-allocated intermediate buffers + buffer-aliasing for in-place ops.
- Non-leaf function prologue/epilogue + callee-saved register analysis.
- Move duplicate-model-name check from `profiles/arm64::walk_uir` up to `compiler::ir::build`.

## 2. Goal

Lower all five M3 positive fixtures (`tiny_mlp`, `classifier`, `pipeline_styles`,
`comments`, `mixed_args`) end-to-end to AArch64 assembly that runs natively on
the host (Apple Silicon arm64). Each fixture's integration test compiles via `cc`,
loads via `libloading`, and produces output bit-close to a pure-Rust reference.

## 3. Non-goals

- **Performance optimisation.** Same scalar nested-loop scheme as M4a; bias-add
  inline; softmax 3-pass. No SIMD, no fusion, no register coalescing. M5+.
- **Bare-metal targets.** `bl _expf` requires libm at link time. Bare-metal /
  no-libc targets get a separate profile (M7+); `expf`-via-Taylor is reserved
  for that profile. The `arm64` profile assumes POSIX with libm.
- **Multi-output models.** Implicit-output convention (one output per model)
  unchanged from M4a.
- **Caller-allocated workspace.** Intermediate buffers are stack-allocated
  inside the function. No 4th workspace pointer in the ABI.
- **Polymorphic function shape per model.** All `nfl_forward_*` functions
  share the same 3-pointer signature `(input, params, output)`, regardless of
  whether the model uses bias, has multiple Linears, etc. Layout details
  travel via `FnSig.params_layout`.
- **Multi-snippet error rendering** (rustc-style "first defined here" /
  "redefined here" with two `^` snippets). M4b uses one snippet + plain-text
  `note: previously defined at line:col`. Multi-snippet upgrade is M4c-or-later.
- **New NFL syntax.** Reuses M3 grammar verbatim. All five existing M3
  fixtures lower; no new fixture file is needed (mixed_args carries the
  `bias=true` coverage).
- **CI changes.** PR #9 already established the workflow. M4b's new tests
  ride on it.

## 4. Workspace structure (no change vs M4a)

3-crate workspace established in M4a stays intact:

```
compiler/      lib only — UIR types, parser, ir::build (+ duplicate-name check now)
nflc/          bin only — CLI (parse + compile subcommands)
profiles/arm64/ lib only — codegen (extended in M4b)
```

No new crates. No new path dependencies. Workspace `members` unchanged.

## 5. ABI redesign — single packed `params` buffer

### 5.1 Function signature

For every `UirModel`:

```c
void nfl_forward_<ModelName>(
    const float* input,
    const float* params,    // packed: weights + biases of all Linear nodes, in UIR-node order
    float*       output
);
```

Standard AAPCS64. Pointers in `x0`, `x1`, `x2`. Possibly non-leaf (see §7).
Identical shape regardless of bias presence, op count, or model complexity.

### 5.2 `FnSig` redesign

```rust
pub struct FnSig {
    pub name: String,
    pub model: String,
    pub input_floats: usize,
    pub output_floats: usize,
    pub params_floats: usize,             // sum of all params_layout slot sizes
    pub params_layout: Vec<ParamSlot>,    // typed; topological by UIR-node order
}

pub struct ParamSlot {
    pub kind: ParamKind,
    pub origin_node: NodeId,              // which UIR node owns this slot
    pub offset: usize,                    // in float-elements from start of params buffer
    pub size: usize,                      // in float-elements
}

#[non_exhaustive]
pub enum ParamKind {
    LinearWeight,                          // matrix [in_dim, out_dim], row-major
    LinearBias,                            // vector [out_dim]
    // future: NormGamma, NormBeta, EmbeddingTable, AttnQ/K/V/O, ...
}
```

`#[non_exhaustive]` on `ParamKind` is mandatory: M5+ ops introduce new param
kinds without breaking downstream `match` consumers.

### 5.3 Layout order

Topological by UIR-node order. For each `Linear` node in order:
1. Emit `LinearWeight` slot (size = `K * N`).
2. If the Linear has `bias=true`: emit `LinearBias` slot (size = `N`).

Other op kinds (Relu, Dropout, Softmax) contribute zero slots.

**Example — classifier with all three Linears using `bias=true`:**

```
slot 0: LinearWeight  (node 1, offset=0,        size=784*512)
slot 1: LinearBias    (node 1, offset=401408,   size=512)
slot 2: LinearWeight  (node 4, offset=401920,   size=512*256)
slot 3: LinearBias    (node 4, offset=533952,   size=256)
slot 4: LinearWeight  (node 6, offset=534208,   size=256*10)
slot 5: LinearBias    (node 6, offset=536768,   size=10)
total params_floats = 536778
```

### 5.4 Breaking change vs M4a

**M4b deliberately breaks the M4a ABI.** `FnSig.weight_floats` is removed;
`FnSig.params_floats` + `FnSig.params_layout` replace it. Generated functions
no longer take a `weights` pointer — they take `params` (which contains the
same floats for bias-free single-Linear models, just renamed).

The project is internal v0.1; no external consumers exist, so no compatibility
shim is needed. **The M4b closeout DEVLOG entry must explicitly mark this
ABI break and state the rationale**, so a future reader of the git history
sees that the change was intentional, not accidental.

The M4a integration test (`tinymlp_no_softmax_runs_correctly`) is renamed and
rewritten to call `forward(input, params, output)` against the new ABI. The
underlying buffer contents are unchanged for the M4a fixture (one Linear, no
bias → params buffer = same float layout as old weights buffer).

## 6. Intermediate buffers — stack-allocated + aliasing

### 6.1 Allocation rule

Every Op-node whose result is consumed by another Op-node (i.e., non-terminal)
gets a buffer for its output. The codegen first-pass walks the UIR computing
buffer assignment per node:

```rust
enum BufferLoc {
    InputReg,                  // x0 — for the model Input node
    OutputReg,                 // x2 — for the terminal node
    StackOffset(usize),        // sp + offset — for non-terminal Op outputs
    Alias(NodeId),             // resolves to another node's BufferLoc
}
```

The `assign_buffers(&UirModel) -> Vec<BufferLoc>` function (one entry per
node, indexed by NodeId) is the ground truth consumed by every op-emitter.

### 6.2 Aliasing rules

**Ops that don't allocate a new buffer** (their output reuses the operand's buffer):
- `Relu`: operates in-place on operand's buffer (same shape). `BufferLoc::Alias(operand_id)`.
- `Dropout`: at inference, identity. Zero instructions emitted; `BufferLoc::Alias(operand_id)`.

**Ops that always allocate** (their output shape may differ or they need fresh storage):
- `Linear`: output `[B, N]` ≠ input `[B, K]` in general. Stack slot.
- `Softmax`: writes new probabilities. If terminal → `OutputReg`. If
  non-terminal (hypothetical M5+) → stack slot. (M4b: always terminal.)

### 6.3 Stack frame management

When at least one node maps to `StackOffset(_)`, the function reserves stack
space in the prologue. AAPCS64 requires 16-byte SP alignment at all `bl`
boundaries — so the total intermediate size is rounded up to 16 bytes:

```asm
; Prologue (when intermediates > 0):
sub     sp, sp, #<aligned_total_bytes>

; Epilogue (when intermediates > 0):
add     sp, sp, #<aligned_total_bytes>
```

For the largest M4b fixture (classifier, batch=32, hidden=512+256+10):
non-terminal Linear nodes need stack slots — n1 (32*512=16384 floats), n4
(32*256=8192 floats), n6 (32*10=320 floats; n6 is non-terminal because softmax
consumes it). Relu and dropout alias their operand's slot, so contribute zero.
Softmax writes to `OutputReg`. Total intermediate = 16384 + 8192 + 320 = 24896
floats = 99584 bytes ≈ 97KB. Well under macOS default thread stack of 8MB.

**Large-immediate `sub sp` handling.** AArch64 `sub` immediate is 12-bit
optionally shifted by 12 (so 0..4095 or 0..16,773,120 in steps of 4096).
For frame sizes that don't fit (like 99584 = not a multiple of 4096 and >4095),
the codegen materialises the size into a scratch GPR first:
```asm
mov     w8, #<low_16_bits>
movk    w8, #<high_16_bits>, lsl #16   ; only if needed
sub     sp, sp, x8
```
For frame sizes that DO fit in a single immediate (multiple of 4096, or ≤4095),
emit the direct form `sub sp, sp, #<imm>`. Implementer chooses based on the
computed aligned size.

### 6.4 Resolving `Alias` chains

When a downstream emitter asks for a buffer location, it follows `Alias`
indirections until it reaches `InputReg`/`OutputReg`/`StackOffset(_)`. Cycles
are impossible by construction (UIR is a DAG; each Alias points to an earlier
node).

## 7. Non-leaf codegen — prologue/epilogue + callee-saved register analysis

M4a functions were pure leaf (no `bl`/`blr`). M4b's softmax emits `bl _expf`,
making any model with softmax non-leaf. Two consequences:

1. **Link register `x30` must be preserved.** `bl` overwrites `x30`; the
   function's eventual `ret` reads from `x30`. Without saving, `ret` jumps
   to garbage.
2. **Caller-saved FP registers are clobbered across `bl _expf`.** Per AAPCS64,
   `s0–s7` and `v16–v31` are caller-saved. Any softmax state that must
   survive the call (per-row `max`, accumulating `sum`) goes in callee-saved
   `s8` and `s9` (lower 64 bits of `v8`/`v9`).
3. **Callee-saved use is two-way.** If `nfl_forward` writes to `s8`/`s9`, it
   must save and restore them for *its own caller*.

### 7.1 Pre-emission analysis

Two analyzers run before any asm is emitted, both walking the model's nodes:

```rust
fn compute_is_leaf(&UirModel) -> bool;
// For M4b: returns false iff any node is Softmax. Other ops emit no `bl`.

fn compute_callee_saved(&UirModel) -> RegSet;
// For M4b: contains {d8, d9} iff any node is Softmax. Other ops use only
// caller-saved registers and re-materialise state per iteration.
```

(`RegSet` is a small bit-set type; for M4b it tracks just `d8` and `d9`,
but generalises for M5+ ops.)

### 7.2 Prologue / epilogue templates

**Leaf, no intermediates** (M4a-style — TinyMLP-no-softmax):
```asm
; (no prologue)
; <body>
ret
```

**Leaf, with intermediates** (hypothetical: multi-Linear model with no softmax):
```asm
sub     sp, sp, #<aligned_intermediates>
; <body>
add     sp, sp, #<aligned_intermediates>
ret
```

**Non-leaf, no callee-saved registers** (hypothetical: a `bl` op that doesn't
need callee-saved FP state — none in M4b but possible in M5+):
```asm
stp     x29, x30, [sp, #-16]!
mov     x29, sp
; <body>
ldp     x29, x30, [sp], #16
ret
```

**Non-leaf with callee-saved FP** (M4b: any model with softmax):
```asm
stp     d8, d9,   [sp, #-16]!
stp     x29, x30, [sp, #-16]!
mov     x29, sp
sub     sp, sp, #<aligned_intermediates>     ; if intermediates > 0
; <body>
add     sp, sp, #<aligned_intermediates>     ; if intermediates > 0
ldp     x29, x30, [sp], #16
ldp     d8, d9,   [sp], #16
ret
```

Each layer (callee-saved FP, frame pointer + LR, SP adjustment) is conditionally
included based on the analyzers' output. Unused layers contribute zero overhead.

### 7.3 Stack alignment invariants

- Each `stp` / `ldp` pair is exactly 16 bytes → preserves alignment.
- `sub sp, sp, #<aligned>` rounds intermediate-buffer total up to a multiple
  of 16 → preserves alignment.
- At the moment of any `bl` instruction, SP is 16-byte aligned. AAPCS64-compliant.

## 8. Op coverage in M4b

| StdOp | M4b support | Codegen sketch |
|---|---|---|
| `Linear` no `bias` | ✅ (M4a) | matmul: 3 nested scalar loops with `fmadd`. |
| `Linear` `bias=true` | ✅ **new** | matmul + bias-add: after `fmadd` k-loop (s0 = sum), load `bias[j]` from params, `fadd s0, s0, s_bias`, then store. |
| `Relu` | ✅ (M4a) | Elementwise loop with `fmov s4, wzr` once + `fmax sN, sN, s4` per element. In-place on operand buffer. |
| `Dropout` | ✅ **new (no-op)** | Zero instructions emitted. `BufferLoc::Alias(operand_id)`. Reads of dropout's output resolve through the alias to the operand's buffer. |
| `Softmax` | ✅ **new** | Per-row 3-pass: max, exp+sum, normalize. Uses `bl _expf` per element. State in callee-saved `s8`/`s9`. |
| `Input` | ✅ (M4a) | Marker only — maps to `BufferLoc::InputReg` (`x0`). |

### 8.1 `linear[N, bias=true]` — bias-add codegen

After the `k`-loop accumulates `s0 = sum`, between the k-loop end and the
output store, insert:

```asm
.Lmm_k_end_<idx>:
    ; (s0 = matmul accumulator)
    ; Load bias[j] from params buffer
    mov     x8, #<bias_slot_offset_in_floats>
    add     x10, x1, x8, lsl #2          ; x10 = params + bias_offset*4
    ldr     s5, [x10, x4, lsl #2]        ; s5 = bias[j]
    fadd    s0, s0, s5                   ; sum += bias[j]
    ; Then the existing store:
    mov     x8, #<n>
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x_dst, x6, lsl #2]
```

The `bias_slot_offset_in_floats` is a constant per Linear node — looked up
from the precomputed `params_layout` (the `LinearBias` slot whose
`origin_node` matches this Linear). `x_dst` is the destination buffer pointer
(may be `x2` if terminal, or computed from `sp + offset` otherwise; see §6).

### 8.2 `softmax` — numerically stable 3-pass

Input shape `[B, K]`. Normalize per row (last axis).

```
for i in 0..B:                         # per-row outer loop
    # Pass 1: find max of row
    max = -inf
    for k in 0..K:
        if input[i*K + k] > max: max = input[i*K + k]

    # Pass 2: compute exp(input - max), write to output, accumulate sum
    sum = 0.0
    for k in 0..K:
        e = expf(input[i*K + k] - max)
        output[i*K + k] = e
        sum += e

    # Pass 3: normalize
    for k in 0..K:
        output[i*K + k] = output[i*K + k] / sum
```

**Materialising `-inf` for the max init.** AArch64 `fmov` immediate uses an
8-bit FP encoding that does NOT include ±inf, ±0, NaN, or denormals. So
`fmov s8, #-inf` is invalid — assembler will reject it. The portable
pattern is to load the bit pattern of `-inf` (`0xFF800000` for f32) into a
GPR and `fmov` from GPR to FP register (this `fmov` variant moves bits
literally, no immediate restriction):

```asm
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16   ; w0 = 0xFF800000 = f32 -inf
    fmov    s8, w0
```

(`movz` zeros the destination then sets the low 16 bits; `movk` keeps
existing bits and updates one 16-bit chunk. Apple's `as` may also accept
the equivalent `mov w0, #0xFF800000` syntax; the `movz`/`movk` pair is the
explicit-and-portable form.) Sum init (`s9 = 0.0`) uses the same trick as
M4a's relu: `fmov s9, wzr` (the integer-zero-register source is a special
case the encoder permits).

**Asm sketch (per-row body):**

```asm
    ; Pass 1: max into s8 (callee-saved)
    ; ... materialise -inf into s8 (see above), then loop fmax over input row ...

    ; Pass 2: exp loop with bl _expf
    ; ... loop body:
    ldr     s0, [x_in, x_k, lsl #2]    ; arg in s0 (caller-saved, we're about to bl anyway)
    fsub    s0, s0, s8                  ; s0 = input[k] - max
    bl      _expf                       ; s0 = expf(s0); clobbers s0-s7, v16-v31
    str     s0, [x_out, x_k, lsl #2]
    fadd    s9, s9, s0                  ; sum += s0
    ; ... loop maintenance ...

    ; Pass 3: normalize loop
    ; ... loop body:
    ldr     s0, [x_out, x_k, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x_out, x_k, lsl #2]
    ; ... loop maintenance ...
```

`s8` holds per-row `max`; `s9` holds per-row `sum`. Both reset at the start of
each row's outer iteration (since they're row-local).

### 8.3 `dropout` — no-op pass-through

Codegen for dropout: emit zero instructions. The buffer-assignment first-pass
sets `BufferLoc[dropout_node] = Alias(operand_node)`. Any downstream op
reading dropout's output resolves the chain through `assign_buffers`'s
output, ending at the operand's actual `BufferLoc`. Documented in
`docs/profile_guide/arm64.md`.

## 9. Move duplicate-model-name check to `compiler::ir::build`

Per M4a §15, `LowerError::DuplicateModelName` lives in `profiles/arm64::walk_uir`
as a temporary measure. It's a profile-agnostic invariant (any future profile
also can't emit duplicate symbols), so it belongs in `compiler::ir::build`.

### 9.1 Changes

1. **`compiler/src/ir/build.rs`:** after building all models, iterate them
   and reject duplicates:
   ```rust
   let mut seen: HashMap<&str, Span> = HashMap::new();
   for model in &uir.models {
       if let Some(prev_span) = seen.get(model.name.as_str()) {
           return Err(BuildError::duplicate_model_name(
               model.name.clone(),
               *prev_span,                        // first span
               model.source_span,                  // current (redefinition) span
           ));
       }
       seen.insert(&model.name, model.source_span);
   }
   ```

2. **`compiler/src/ir/error.rs`:** add variant:
   ```rust
   BuildErrorKind::DuplicateModelName {
       name: String,
       first_span: Span,                          // location of first definition
   },
   ```
   `BuildError`'s `line`/`col` fields carry the *redefinition* span (so
   `render_error_with_snippet` points at the conflict, not the original).

3. **`profiles/arm64/src/codegen.rs`:** delete the duplicate-name check from
   `walk_uir`. It's now unreachable (build already filters).

4. **`profiles/arm64/src/types.rs`:** delete `LowerError::DuplicateModelName`.
   `#[non_exhaustive]` makes the removal non-breaking for downstream `match`
   consumers (they're already required to keep a `_ =>` arm).

5. **Test moves:**
   - Add `duplicate_model_name_at_build_time` in `compiler/src/ir/tests.rs`.
   - Delete `duplicate_model_name_returns_error` from `profiles/arm64/src/tests.rs`.

### 9.2 Diagnostic format (M4b minimal version)

```
error: duplicate model name 'M': would emit conflicting symbols
  --> tests/fixtures/dup.nfl:5:7
   |
5  |     model M [...]:
   |           ^
note: previously defined at tests/fixtures/dup.nfl:1:7
```

`render_error_with_snippet` in `nflc/src/main.rs` accepts an optional
`first_span: Option<(u32, u32)>` argument: when present, appends a
`note: previously defined at <file>:<line>:<col>` plain-text line after the
snippet block. This is the M4b interim solution; M4c may upgrade to a true
two-snippet rustc-style render.

## 10. CLI changes

**None.** `nflc parse <file>`, `nflc parse <file> --tokens`, `nflc parse <file> --uir`,
`nflc compile <file> --profile arm64 [-o <path>]` work as in M4a. The CLI
becomes more useful (more fixtures lower successfully, fewer `LowerError::UnsupportedOp`
returns) but its surface is unchanged.

## 11. Test strategy

### 11.1 Unit tests (`profiles/arm64/src/tests.rs`)

Add ~10 new unit tests on top of M4a's set. Each asserts on the asm-string
shape (no execution):

| Test | Validates |
|---|---|
| `linear_with_bias_emits_bias_add` | `linear[N, bias=true]` → asm contains `fadd s0, s0, ...` after k-loop, before store. |
| `dropout_emits_no_code` | `linear → dropout → linear` → no instructions specifically attributable to dropout (no new label, no extra ops between the two linear blocks). |
| `softmax_emits_three_passes` | `linear → softmax` → asm contains max-loop, exp-loop with `bl _expf`, normalize-loop. |
| `non_leaf_function_saves_x29_x30` | Model with softmax → prologue has `stp x29, x30, [sp, #-16]!`. |
| `softmax_function_saves_d8_d9` | Model with softmax → prologue has `stp d8, d9, [sp, #-16]!`. |
| `leaf_function_no_prologue` | M4a-style model (no softmax, no intermediates) → no `stp`/`ldp`, no `sub sp` — clean `ret`. |
| `multiple_linears_packed_params_layout` | UIR with 3 Linear → `params_layout.len() == 3`, slots in topological order, correct offset arithmetic. |
| `linear_bias_packed_layout` | `linear[N, bias=true]` (one Linear) → 2 slots: `LinearWeight` then `LinearBias` immediately after. |
| `intermediate_buffers_allocated_on_stack` | Multi-stage model → prologue has `sub sp, sp, #N`, epilogue has `add sp, sp, #N` with N=aligned-up size. |
| `dup_name_now_at_build_time` | `compiler::ir::build` returns `BuildErrorKind::DuplicateModelName`; `profiles_arm64::lower` is never reached. |

### 11.2 Integration tests (`profiles/arm64/tests/integration.rs`)

End-to-end: build UIR → lower → cc-assemble → libloading dlopen → call → compare.
All 5 M3 fixtures + the M4a fixture under the new ABI.

| Test | Fixture | What it validates |
|---|---|---|
| `tinymlp_full_with_softmax_runs_correctly` | `tiny_mlp.nfl` | `linear → softmax`, new ABI, callee-saved + non-leaf prologue. |
| `classifier_runs_correctly` | `classifier.nfl` | 3 Linear + 2 relu + dropout + softmax — most complex realistic path. ~96KB stack frame exercised. |
| `pipeline_styles_runs_correctly` | `pipeline_styles.nfl` | 3 models in one file → 3 distinct symbols, 3 distinct `nfl_forward_*` calls in the test. |
| `comments_runs_correctly` | `comments.nfl` | Sanity (comments don't alter codegen, but it's a free coverage point). |
| `mixed_args_runs_correctly` | `mixed_args.nfl` | `linear[16, bias=true]` end-to-end — bias-add path verified live. The only M3 fixture with `bias=true`, which makes it the canonical bias-end-to-end test (no new fixture needed). |
| `m4a_no_softmax_still_runs` | `m4_linear_relu.nfl` | M4a fixture preserved, runs under the new ABI (`forward(input, params, output)`). Replaces the M4a-era `tinymlp_no_softmax_runs_correctly`. |

### 11.3 Reference function validation (mandatory pattern)

**Every reference function with non-trivial logic gets its own unit test
against hand-computed values.** Without this, a bug in the reference function
goes undetected when the asm has the same bug.

Required for M4b:
- `reference_softmax_stable_known_values` — `softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]`.
- `reference_bias_add_known_values` — `[1,2,3] + [0.5, -1, 2.5] = [1.5, 1, 5.5]`.

`reference_matmul` and `reference_relu` are trivial enough that hand-checked
inputs in the integration tests provide adequate coverage. (Implementer may
add explicit tests if uncertain; not required.)

### 11.4 Test count budget

Baseline (after M4a + CI): **118**. After M4b: ~134 (10 new unit + 6 new
integration + 2 reference-validation). The plan should record actual baseline
at task-start and assert no regression, not hard-code 134.

### 11.5 Pre-flight & CI portability

Same as M4a §9.4. Integration tests gate on `cfg!(target_arch = "aarch64")`
and `cc_available()`; skip cleanly with logged reason on Linux ubuntu CI.
Unit tests run anywhere.

## 12. Existing fixtures used (no new fixtures)

All five M3 positive fixtures (`tiny_mlp`, `classifier`, `pipeline_styles`,
`comments`, `mixed_args`) plus the M4a fixture (`m4_linear_relu`) cover the
M4b op set. **`mixed_args.nfl` carries the `bias=true` coverage**
(`linear[16, bias=true]`) — verified during brainstorming.

## 13. Dependency policy

**Production crates** (`compiler`, `nflc`, `profiles/arm64` lib-target):
strictly std-only. Unchanged from M4a §11.

**Linkage:** `cc` links against libm by default on macOS and Linux; `expf`
resolves through that. No new Rust dependency. The runtime requirement
(libm available at link time) is documented in `docs/profile_guide/arm64.md`.
Bare-metal targets without libm are out of scope for this profile (M7+ may
add a `bare-metal-arm64` profile that uses Taylor for `exp` instead).

**Dev-dependencies:** unchanged. `libloading 0.8` from M4a remains the only
non-std dep, used only in `profiles/arm64`'s integration test.

## 14. Artifacts (created / modified / deleted)

### Created

| Path | Purpose |
|---|---|
| `profiles/arm64/src/ops/mod.rs` | Per-op codegen submodule entry; re-exports `emit_*`. |
| `profiles/arm64/src/ops/linear.rs` | `emit_linear(...)` — matmul + optional bias-add. |
| `profiles/arm64/src/ops/relu.rs` | `emit_relu(...)` — moved from M4a's `codegen.rs`. |
| `profiles/arm64/src/ops/softmax.rs` | `emit_softmax(...)` — 3-pass with `bl _expf`. |
| `profiles/arm64/src/ops/dropout.rs` | Marker / no-op (or omitted; aliasing is decided in buffer.rs). |
| `profiles/arm64/src/buffer.rs` | First-pass analyzers: `assign_buffers`, `compute_is_leaf`, `compute_callee_saved`. Defines `BufferLoc`, `RegSet`. |

### Modified

| Path | Change |
|---|---|
| `compiler/src/ir/build.rs` | Add duplicate-model-name check after model-build loop. |
| `compiler/src/ir/error.rs` | Add `BuildErrorKind::DuplicateModelName { name, first_span }`. |
| `compiler/src/ir/tests.rs` | Add `duplicate_model_name_at_build_time` test. |
| `nflc/src/main.rs` | Extend `render_error_with_snippet` to accept optional `first_span` and emit a trailing `note: previously defined at line:col` line when present. |
| `profiles/arm64/src/lib.rs` | Update `pub use` exports for refactored types (`ParamSlot`, `ParamKind`, etc). Stay private if not needed externally. |
| `profiles/arm64/src/types.rs` | Drop `weight_floats` from `FnSig`; add `params_floats`, `params_layout`. Add `ParamSlot`, `ParamKind` (`#[non_exhaustive]`). Drop `LowerError::DuplicateModelName`. |
| `profiles/arm64/src/codegen.rs` | Refactor `walk_uir`/`walk_model` to use `buffer.rs` analyzers and dispatch to `ops/*` modules. Remove the M4a-style monolithic Linear/Relu emitters (now live in `ops/`). |
| `profiles/arm64/src/asm.rs` | Add `format_function_prologue(LeafKind, RegSet, intermediate_bytes) -> String` and `format_function_epilogue(...)`. Replace M4a's fixed harness helpers. |
| `profiles/arm64/src/tests.rs` | Add the 10 new unit tests from §11.1. Existing tests adapt to the new `FnSig` shape and the new ops/ module structure. |
| `profiles/arm64/tests/integration.rs` | Replace M4a's single integration test with the 6 from §11.2. Add the reference-validation tests from §11.3. |
| `profiles/arm64/tests/common/mod.rs` | Possibly extend with multi-symbol helpers for `pipeline_styles` (which produces 3 symbols in one .dylib). |
| `docs/profile_guide/arm64.md` | Extend with: bias-add codegen, softmax 3-pass, libm dependency note, intermediate buffer allocation, callee-saved + non-leaf prologue, new ABI. Limitations section reflects M4b's coverage. |
| `docs/language_reference/uir.md` | One-line note: dropout at inference is identity (codegen treats it as buffer aliasing). |
| `PROJECT_SPEC.md` | Milestones table M4 row → "4a + 4b complete". Architecture Profiles `arm64` row → expanded capability description. |
| `CLAUDE.md` | Current Status → M4 fully complete; M5 next. Repo Structure may need a small update for `ops/` and `buffer.rs`. |
| `DEVLOG.md` | M4b closeout entry with **explicit ABI-break note** (`weight_floats` → `params_floats` + layout) and rationale. |

### Deleted

Nothing.

## 15. Vertical slicing — monolithic M4b

M4b is shipped as a **single PR** (no M4b1/M4b2 split). Reasons:
- Plumbing-only (M4b1) state would change the public ABI without delivering
  new ops — a non-shippable interim.
- All three new ops (bias, dropout, softmax) need the same plumbing
  (params layout, intermediate buffers, leaf/non-leaf analysis).
- M4a was the same shape (~12 tasks, mixed plumbing + features) and merged
  cleanly in one PR.

**Task ordering inside the monolithic plan** (per user guidance during
brainstorming — infrastructure first, ops in increasing complexity):

1. `params_layout` + `FnSig` refactor (the deliberate ABI-break commit).
2. Buffer-assignment analyzer (`buffer.rs`: `BufferLoc`, `is_leaf`, `callee_saved`).
3. Move duplicate-name check from `profiles/arm64::walk_uir` to
   `compiler::ir::build` + extend `render_error_with_snippet` to support
   the `first_span` note.
4. `dropout` aliasing (trivial — zero asm; only `buffer.rs` table changes).
5. `linear[N, bias=true]` (matmul + bias-add). Mixed_args fixture lights up.
6. `softmax` (3-pass + `bl _expf` + non-leaf prologue + d8/d9 save).
7. Tests, profile-guide doc updates, DEVLOG/CLAUDE.md/PROJECT_SPEC updates,
   final smoke + clippy + closeout.

If softmax (task 6) hits an unexpected blocker, everything before it is
already green; the issue is isolated. Subagent-driven workflow with per-task
review enforces this discipline automatically.

## 16. Open questions / risks

- **`bl _expf` symbol resolution on Linux CI runner.** macOS clang links
  libm by default; Linux gcc/clang typically need `-lm`. Our test helper
  invokes `cc -shared -arch arm64 -o foo.dylib foo.s`. On Linux this would
  need adjustment (no `-arch arm64` on Linux cc; possibly `-lm` explicit).
  The integration test self-skips on non-aarch64 hosts (Linux ubuntu CI),
  so this doesn't bite M4b — but if a future Linux-arm64 CI runner is added,
  the test helper will need conditional flags. Note in the profile guide.

- **FMA divergence (carried from M4a §15).** Asm uses `fmadd`; Rust reference
  uses `*` then `+`. For M4b's integration tests with classifier (768→512→256→10
  pipeline of multiplications), accumulated divergence could be larger than M4a's
  8×4×2 case. If `1e-5` epsilon flakes, switch the reference to `f32::mul_add`
  to match the asm bit-exactly. Documented in the spec; implementer applies the
  workaround if needed.

- **`expf` precision differences across libm implementations.** Apple's libm
  `expf` may differ in ULPs from glibc's. Same `1e-5` epsilon strategy applies;
  if the integration test ever runs on Linux arm64, the reference can call
  Rust's `f32::exp` to match libm's contract closely enough for these
  tolerances.

- **Stack-frame size for very large models.** classifier's ~96KB frame is
  fine on macOS (8MB stack). Future fixtures with larger batch sizes or
  hidden dims could push past 1MB; probably still fine but worth tracking.
  No mitigation needed for M4b; document in profile guide.

- **`d8`/`d9` save covers only lower 64 bits.** AAPCS64 says lower 64 bits
  of `v8`–`v15` are callee-saved. We only use `s8`/`s9` (lower 32 bits of
  `v8`/`v9`) — well within the contract. No issue, but worth a comment in
  the prologue helper for future readers.

## 17. Acceptance criteria

1. **Build clean across the workspace.** `cargo build --workspace` exits 0
   with zero warnings.
2. **Clippy clean.** `cargo clippy --workspace --all-targets -- -D warnings`
   exits 0.
3. **Format clean.** `cargo fmt --all -- --check` exits 0.
4. **All pre-M4b tests still pass** (currently 118; plan should capture
   actual count and assert no regression).
5. **All 10 M4b unit tests + 6 integration tests pass** (or skip cleanly on
   non-aarch64 hosts with logged reason).
6. **`nflc compile` produces valid asm for all 5 M3 fixtures + the M4a
   fixture.** Each can be assembled with `cc -shared -arch arm64` and the
   expected `nfl_forward_*` symbol(s) appear in the resulting dylib.
7. **`compiler::ir::build` rejects duplicate model names** with a snippet +
   `note: previously defined at line:col`.
8. **CI is green on the PR's first run** (or fixed-then-green within the PR).
9. **`docs/profile_guide/arm64.md` updated** to reflect M4b coverage,
   including the libm dependency note.
10. **DEVLOG entry exists with explicit ABI-break note.** CLAUDE.md
    Current Status reflects M4 fully complete; PROJECT_SPEC milestones table
    updated.

## 18. Out of M4b (explicit non-coverage)

- No SIMD. Scalar throughout. NEON is M5+/M6.
- No fusion. `linear → relu` still emits two separate loops. Fusion is M5.
- No optimisation passes of any kind.
- No new fixtures. All 5 M3 + 1 M4a fixture cover the M4b op set.
- No new CLI subcommands.
- No new architecture profiles (still `arm64` only).
- No coverage tooling.
- No bare-metal target. Profile assumes POSIX with libm.
- No multi-snippet ("first defined here / redefined here" with two `^`)
  rendering. M4b uses single snippet + plain-text note. M4c may upgrade.
- No CI configuration changes (PR #9 already established).

## 19. Sub-skill chain after this spec is approved

1. Spec self-review (placeholder/contradiction scan).
2. User reviews this spec file.
3. On approval → invoke `superpowers:writing-plans` to produce
   `docs/superpowers/plans/2026-05-04-m4b-arm64-coverage.md`.
4. Subagent-driven execution mode (per the project's prior pattern); per-task
   review between tasks; INLINE for trivial tasks like fixture-creation
   (none in M4b — no new fixtures) and DEVLOG/CLAUDE.md/PROJECT_SPEC updates.
5. PR against `main` when M4b is shippable. CI gates on first push.
