# M17 — Axis 3: Bare-Metal `expf` (inline), first leg

**Date:** 2026-05-29
**Strategic axis:** Axis 3 — deployment reach (per Strategic Roadmap line in `PROJECT_SPEC.md`). First of two legs; the second (softmax leaf-cleanup) is recorded as M18 in §9.
**Status:** Brainstorm complete; awaiting plan synthesis.

---

## 1. Goal & Non-Goals

### Goal

Replace the libm `expf` call in softmax codegen with an **inlined, bare-metal**
`f32` exponential on both profiles, removing NeuralForge's last runtime
dependency. After M17 a softmax-bearing binary links against nothing — it is
genuinely bare-metal.

`expf` is consumed from exactly two emit sites per profile, both inside softmax:

- standalone `StdOp::Softmax` exp-pass — `profiles/{arm64,x86_64}/src/ops/softmax.rs`;
- fused `PostOp::SoftmaxRow` row-wise tail — `profiles/{arm64,x86_64}/src/ops/linear.rs`.

Both currently emit `bl _expf` (arm64) / `call expf@PLT` (x86_64). M17 replaces
that single instruction (plus its post-call scratch recompute) with an inline
polynomial expansion produced by a shared per-profile helper.

### Scope discipline — this is the **minimal swap**

M17 does **not** touch the softmax loop's register allocation. The loop keeps
using its current callee-saved state registers (arm64 `x19-x23` / `s8` / `s9`;
x86_64 `%rbx` / `%r12-%r15` + `row_max`/`row_sum` stack slots) byte-for-byte.
The prologue keeps saving them — still correct, because the loop genuinely uses
(clobbers) those registers; the *justification* shifts from "across the libm
call" to "the loop body uses them." Only the exp-pass body changes.

The one honest cleanup that **is** in M17: the misnamed predicate
`calls_extern_math()` is renamed to `has_softmax()` (§4) — after inlining, the
old name (and its `--uir-verbose` label) would be factually false, violating
Design Principle 5 (human-inspectable output).

### Non-Goals — deferred to M18 (see §9)

Each is real, contained follow-up work, explicitly **out** of M17:

- Moving softmax loop state from callee-saved to caller-saved registers.
- Dropping the FFI save/restore around the (now-removed) call.
- Removing softmax's contribution to the callee-saved prologue (arm64
  `d8-d9`/`x19-x23`; x86_64 softmax half of `callee_saved_int`).
- Moving x86_64 `row_max`/`row_sum` from stack slots into xmm registers.
- Flipping `compute_is_leaf` to `true` for softmax models + the leaner prologue
  (this is where the inspect goldens change).
- The unambiguous bench speedup (M17 is dependency-removal; net per-call cycle
  delta is measured, not promised — see §5.3).

### Non-Goals — beyond both M17 and M18

- A **general-purpose** `expf` over the full `f32` range (overflow → `+inf`,
  positive-argument handling). Softmax feeds `exp` only the non-positive
  argument `x − row_max ≤ 0`, so output is `(0, 1]` and overflow is
  unreachable. A general `expf` would ship **untested** guard branches (the
  tolerance sweep only covers the reachable domain), which is worse than not
  shipping them. When a future op (GELU, SiLU, cross-entropy loss) needs the
  full range, generalisation is a concrete diff with its own tests.
- Any public `expf` symbol. The helper is purely inline; with no `bl`/`call`
  there is no symbol to export.

---

## 2. Algorithm

`exp(x)` for `x ≤ 0`, `f32`. Three stages, all single-precision.

### 2.1 Range reduction (Cody-Waite)

```
z   = round_to_nearest( x · LOG2E )          // i32, round-ties-to-even
zf  = (f32) z
r   = (x − zf·LN2_HI) − zf·LN2_LO            // |r| ≤ ln2/2 ≈ 0.34657
```

Constants (file-local `.rodata`, §3.3):

| Name      | Value             | Note                                         |
|-----------|-------------------|----------------------------------------------|
| `LOG2E`   | `1.4426950409`    | `log2(e)`, as `f32`                          |
| `LN2_HI`  | `0.693359375`     | exactly representable in `f32` (`0x3F318000`) |
| `LN2_LO`  | `−2.12194440e-4`  | `LN2_HI + LN2_LO = ln2` to extra precision   |

The two-step subtraction (subtract `zf·LN2_HI` first, then `zf·LN2_LO`)
cancels the rounding error of a single `zf·ln2` product. **`LN2_LO` is the
tuning lever for the ≤ 1 ulp gate (§5.2):** if the empirical sweep exceeds
1 ulp, the reduction is widened (e.g. a three-part split adding `LN2_LO2`),
**not** the polynomial degree.

### 2.2 Polynomial — Taylor degree 7, Horner

`exp(r) ≈ Σ_{k=0}^{7} rᵏ / k!` over the reduced interval. Coefficients are
`1/k!` — trivially correct and self-documenting (the chief reason this beats a
minimax fit: the Rust reference port in §5.1 reads as fractions checkable by
eye, keeping *both* sides of the bit-exact contract auditable). On
`|r| ≤ ln2/2` the degree-7 remainder is below `f32` epsilon (< 1 ulp on
`exp(r)` itself).

Horner form (coefficients high → low: `1/5040, 1/720, 1/120, 1/24, 1/6, 1/2, 1, 1`):

```
p = 1/5040
p = p·r + 1/720
p = p·r + 1/120
p = p·r + 1/24
p = p·r + 1/6
p = p·r + 1/2
p = p·r + 1
p = p·r + 1          // p = exp(r)
```

Seven multiply-accumulate steps. On arm64 these are `fmadd` (fused); on x86_64
they are separate `mulss` + `addss` (SSE2 has no scalar FMA). This ISA
divergence is mirrored in the reference ports (§5.1).

### 2.3 Reconstruction

```
exp(x) = exp(r) · 2^z

2^z built by integer exponent insertion:
    bits   = (z + 127) << 23           // i32 → f32 bit pattern
    pow    = reinterpret_f32(bits)
    result = p · pow

underflow clamp:  z ≤ −127  →  result = +0.0
```

For the softmax domain `z ≤ 0`. The clamp flushes results below the smallest
normal `f32` (`2^−126`) to `+0.0`; this is correct for softmax because such
terms are negligible against the max term (`exp(0) = 1`) and the row is divided
by a sum `≥ 1`. The clamp is **not** exercised by the §5.2 accuracy sweep
(restricted to `x ∈ [−80, 0]`, where `z ≥ −115`, comfortably normal); it is
covered separately by a runtime flush-agreement assertion (§6.3).

---

## 3. Per-profile codegen

### 3.1 Drop-in shape

A shared per-profile helper

```
emit_exp_inline(arg_reg, /* scratch contract */ …) -> String
```

replaces the single `bl`/`call` line (and its post-call `x6`/`%rax` recompute)
at both sites. Input argument arrives in `s0` / `%xmm0`; the result is left in
the same register — identical to the call's calling convention, so the
surrounding loop is untouched.

Scratch budget (minimal-swap constraint): the **retained** FFI save (§3.4) has
already spilled the ABI-argument registers (`x0-x2` / `%rdi`/`%rsi`/`%rdx`) for
the whole softmax duration, so the helper may freely use them plus other
non-loop-live caller-saved registers (e.g. `x9`). The helper must **not** touch
the loop-live set (arm64 `x19-x23`/`s8`/`s9`; x86_64 `%rbx`/`%r12-%r15` and the
`row_max`/`row_sum` stack slots).

### 3.2 Key instructions

| Stage             | arm64                              | x86_64 (SSE2)                       |
|-------------------|------------------------------------|-------------------------------------|
| `z = round(x·LOG2E)` | `fmul` + `fcvtns` (→ int, ties-even) | `mulss` + `cvtss2si` (→ int, ties-even) |
| `zf = (f32) z`    | `scvtf`                            | `cvtsi2ss`                          |
| reduction `r`     | `fmsub` (fused)                    | `mulss` + `subss` (two roundings)   |
| Horner            | 7× `fmadd`                         | 7× (`mulss` + `addss`)              |
| `2^z`             | `add`/`lsl`/`fmov` (int→f32 bits)  | `addl`/`shll`/`movd`                |
| clamp + multiply  | `cmp` + `fcsel` + `fmul`           | `cmpl` + conditional + `mulss`      |

Exact register assignments and instruction ordering are finalised in the
implementation plan; the spec fixes behaviour, the scratch contract, and the
instruction *families* (so the reference ports in §5.1 can mirror rounding).

### 3.3 Constant pool

The 11 `f32` constants (3 reduction + 8 Taylor) live in a `.rodata` literal
pool emitted **once per assembly file** from the `lower()` driver, guarded by
`uir.has_softmax()`. Loads: arm64 `adrp` + `ldr`; x86_64 `movss sym(%rip)`.

**Pool labels are file-local** (e.g. `.Lexp_c_log2e`, not a global symbol) so
that linking several NeuralForge object files together cannot collide. Emitting
the pool once (not per emit site, not per model) avoids duplicate definitions
when a file contains multiple softmax-bearing models.

Inline-immediate materialisation (`movz`/`movk` + `fmov` per constant per
element) was rejected: ~30 extra instructions per element would likely make
M17 *slower* than the libm call it removes — a regression merged for the sake
of bare-metal, which is not worth merging. The `.rodata` pool is standard on
both ISAs and orthogonal to register layout (so it does not disturb the M17/M18
boundary).

**Correction found during plan synthesis:** the pool is *pre-existing* on
x86_64 — `emit_layernorm` already emits a `.section .rodata` pool with
`.L`-local labels (`profiles/x86_64/src/ops/layernorm.rs`) — but *new* on arm64,
whose `emit_layernorm` materialises its 3 constants inline via `movz`/`movk`/
`fmov` (`profiles/arm64/src/ops/layernorm.rs`). M17 therefore introduces a
Mach-O `.section __TEXT,__const` pool for arm64, referenced via `adrp`/`ldr`.

### 3.4 FFI save/restore and recompute — retained, repurposed

Because the inline helper uses the ABI-argument registers as scratch, the
existing `emit_ffi_save` / `emit_ffi_restore` are **still required** in M17 —
they now protect the ABI arguments across the helper's scratch usage instead of
across the call. They are emitted unconditionally inside the emitters (not
predicate-gated), so the code is unchanged; only the doc-comments are reworded.
The post-call `x6`/`%rax` recompute likewise stays (the helper clobbers
scratch). Their removal is M18 work.

---

## 4. Predicate rename: `calls_extern_math()` → `has_softmax()`

The predicate's implementation (`compiler/src/ir/types.rs`) already computes
exactly "the model contains a standalone `StdOp::Softmax` or a fused
`PostOp::SoftmaxRow`" — only its **name** mentions extern math. After inlining,
nothing calls extern math; the name (and its CLI label) become false.

- **Rename both** `UirModel::calls_extern_math` and `Uir::calls_extern_math` →
  `has_softmax`. Name describes *what* is detected (profile-neutral, fitting a
  predicate that lives in the hardware-agnostic compiler core), not *why* it
  matters (the consequence differs per profile). `has_X` is the idiomatic Rust
  boolean-predicate form.
- **Doc-comments** at the definition and all six consumer sites
  (arm64 `buffer.rs` `compute_is_leaf` / `compute_callee_saved` / `RegSet`;
  x86_64 `buffer.rs` `assign_buffers` / `compute_callee_saved`; x86_64
  `codegen.rs` `leaf_bool`) are reworded from "across `bl _expf` / `expf@PLT`"
  to "the softmax loop holds state in these registers."
- **`--uir-verbose` label** `calls-extern-math: yes/no` → `has-softmax: yes/no`
  (two `Display` sites in `types.rs`). **Mandatory in M17, not deferrable:** a
  `calls-extern-math: yes` line on a post-inline softmax model is a factual lie
  in human-inspectable output (Design Principle 5).
- **Sweep targets:** `docs/language_reference/uir.md` ("Viewing UIR" section)
  and any `.rs` test asserting the label string. The method-level tests
  `calls_extern_math_*` in `compiler/src/ir/tests.rs` rename with the method.

**No register-layout cascade — this is what makes the M17/M18 split viable.**
The rename moves zero register assignments and zero buffer offsets: the
predicate's value is unchanged for every model (same implementation), so
`compute_callee_saved` and `assign_buffers` produce identical output, and the
softmax loop keeps its current register set. The rename is purely a method
name + doc-comments + one CLI label. Verified by inspecting all six consumers:
none derives a register or offset *from the name* — each only branches on the
boolean. Were the rename to force the loop's state off callee-saved registers,
the split would be illusory and M18 would have to land in one shot; it does
not.

### Leaf classification stays conservative (intentional)

`compute_is_leaf` remains `!has_softmax()`, so softmax models are still
classified non-leaf in M17. This is the distinction accepted during
brainstorming: `calls_extern_math = true` post-inline would be *factually
wrong*, whereas `leaf = false` is *suboptimal-but-correct* (saving the link
register when it is not clobbered is safe). The `compute_is_leaf` doc-comment is
reworded to state the conservatism explicitly and point at M18 for the precise
reclassification. **Consequence:** the M16 inspect goldens (which assert `leaf`
and `callee-saved`) are **byte-identical** in M17 — a clean test boundary.

---

## 5. Validation contract

Two independent layers (the agreed Q1 contract). Layer 1 anchors against
regression; layer 2 anchors against truth.

### 5.1 Layer 1 — bit-exact asm vs Rust port

A new reference `exp_ref(x: f32) -> f32` lives in each profile's
`tests/common/mod.rs` (per-profile copy — the isolation pattern established by
M15's `reference_matmul`). It reproduces §2 step-for-step: `round_ties_even`
for `z`, the two-step reduction, the degree-7 Horner, `2^z` via
`f32::from_bits(((z + 127) as u32) << 23)`, the `z ≤ −127` flush.

**Per-profile FMA divergence applies uniformly** — wherever the asm fuses a
multiply-accumulate, the port uses `f32::mul_add`; wherever it does not, the
port uses separate `*`/`+`/`-`:

- **arm64** port: `f32::mul_add` for the reduction (`fmsub`) **and** every
  Horner step (`fmadd`).
- **x86_64** port: separate `*` then `−`/`+` for reduction and Horner (SSE2 has
  no scalar FMA).

**Correction found during plan synthesis:** the current softmax FFI tests are
tolerance-based (`abs() < 1e-4`, row-sum ≈ 1) — there is no libm-`f32::exp`
reference to "swap." Layer 1 is therefore a **new** bit-exact FFI test, anchored
on a new isolated fixture `tests/fixtures/softmax_only.nfl` (`input → softmax`,
no surrounding ops), so the asm output equals `softmax_ref` — a reference
softmax composing sequential max/sub/sum/div with `exp_ref` — bit-for-bit via
`to_bits()`. The isolated fixture pins exactly the changed code; the existing
tolerance tests stay as complementary coverage. (This refines §6.3's original
"no new fixture" stance.)

### 5.2 Layer 2 — `exp_ref` vs libm, ≤ 1 ulp

A new pure-Rust test (runs on both platforms) sweeps `exp_ref` against libm
`f32::exp` over `x ∈ [−80, 0]` (dense sweep + structural points `0, −ln2, −1,
−10, −50`) and asserts a ulp distance `≤ 1`. Because layer 1 pins asm ≡
`exp_ref` bit-for-bit, this transitively bounds **asm** to ≤ 1 ulp of libm.

**Accuracy statement (exact wording for the spec / docs):** *"≤ 1 ulp,
confirmed by sweep; widen the LN2 split (not the polynomial degree) if a point
exceeds it."* This is deliberately **not** phrased as a degree-7-in-isolation
guarantee. The polynomial is sub-ulp on the reduced interval, but end-to-end
error also absorbs ~0.5 ulp from the reduction and ~0.5 ulp from the final
multiply — which degree does not control. The honest path (mirroring M16's
"zero hand-computed numbers — measured only" golden rule) is measure-then-tune:
if the x86_64 (no-FMA) sweep shows, say, 1.2 ulp at worst points, the fix is a
wider `LN2` split, and the ≤ 1 ulp acceptance criterion is held fixed.

### 5.3 Performance posture

M17's guaranteed, measurable result is **dependency removal**. With the
`.rodata` pool (§3.3) M17 is expected to be roughly perf-neutral per call; the
unambiguous speedup (leaf prologue, no spills, hoisted constants) is M18. The
OQ-BENCH `self_attention` fixture (expf-dominated) is the instrument; before/
after numbers are reported, not promised.

---

## 6. Test plan

### 6.1 asm-shape unit tests — flip

The ~6 positive tests per profile that assert `bl _expf` / `call expf@PLT`
(arm64 `tests.rs` ≈ lines 113/587/632; x86_64 `tests.rs` ≈ 211/256/748/790)
flip to assert the **absence** of the call **and** the presence of inline
markers: the file-local pool label, the round-to-int (`fcvtns`/`cvtss2si`), the
Horner chain, the `2^z` reconstruction. The negative tests
(`matmul_does_not_call_expf*`) stay green trivially (still no call).

### 6.2 New unit test — pool locality

Assert the `.rodata` pool is emitted with the correct constants under
**file-local** labels and that no global `expf` symbol is referenced anywhere
in the output.

### 6.3 FFI — underflow-clamp runtime evidence

**Correction found during plan synthesis** (refines the original "no new
fixture" stance): M17 adds one small fixture, `tests/fixtures/softmax_only.nfl`
(`input → softmax`), introduced for the layer-1 bit-exact test (§5.1). The clamp
test **reuses** it with a wide-logit-spread input row that drives `x − row_max`
far negative (`z < −127`), asserting flush-to-`0` (and that the row still sums
to 1). No *other* new fixture is needed — the existing softmax-bearing fixtures
(`classifier`, `self_attention`, `softmax_with_bias`) continue to cover the
emitter via their tolerance tests.

### 6.4 Accuracy sweep

The layer-2 test of §5.2.

---

## 7. Code structure

- New module `profiles/{arm64,x86_64}/src/ops/exp.rs` — a codegen **primitive**,
  not a `StdOp`. Holds `emit_exp_inline` and the constant-pool emitter. Wired
  into `ops/mod.rs` (submodule + re-exports).
- `softmax.rs` and `linear.rs` (fused tail) call `emit_exp_inline` in place of
  the `bl`/`call` line.
- `lower()` emits the `.rodata` pool once when `uir.has_softmax()`.
- A small `asm.rs` helper may host the pool-section boilerplate if it keeps
  `exp.rs` focused; decided in the plan.

---

## 8. Documentation

- `docs/profile_guide/arm64.md`, `docs/profile_guide/x86_64.md` — rewrite the
  softmax sections: inline algorithm, `.rodata` pool, register usage; drop the
  `bl _expf` / `expf@PLT` narrative.
- `docs/language_reference/uir.md` — `has-softmax` label in "Viewing UIR."
- `PROJECT_SPEC.md` — Milestone 17 table row; §Decisions (the `.rodata`-pool
  choice and the two-layer accuracy contract); Strategic Roadmap (Axis 3 first
  leg closed in M17, M18 recorded as second leg); §Known Latent Hazards stays
  empty (M17 opens no bug — `leaf = false` is deferred optimisation, the clamp
  is tested).
- `CLAUDE.md` — Current Status → M17; note the new `ops/exp.rs` primitive.
- `DEVLOG.md` — M17 entry.

---

## 9. M18 — recorded scope (the deferral)

Recorded here and in `PROJECT_SPEC.md` so it survives between sessions.

**M18 — Axis 3 second leg: softmax leaf-cleanup.** Now that no call remains,
make the codegen honest and lean:

1. Move softmax loop state from callee-saved to caller-saved registers (safe —
   no call to clobber them), keeping clear of the ABI-argument registers.
2. Drop `emit_ffi_save` / `emit_ffi_restore` and the scratch recompute at the
   softmax sites.
3. Remove softmax's contribution to the callee-saved prologue (arm64
   `d8-d9`/`x19-x23`; x86_64 the softmax half of `callee_saved_int` — matmul's
   `has_matmul` trigger is independent and stays).
4. Move x86_64 `row_max`/`row_sum` from stack slots into xmm registers; drop the
   16-byte reserve in `assign_buffers`.
5. Flip `compute_is_leaf` to `true` for softmax models + emit the leaner
   leaf prologue. **This is where the M16 inspect goldens change.**
6. Re-evaluate whether `has_softmax()` is still needed (it may collapse if no
   regime depends on it after the above).
7. Measure the bench speedup (leaf + no spills + hoisted constants) on the
   `self_attention` fixture.

The M17/M18 boundary is clean precisely because M17 leaves register layout and
leaf classification untouched: M17 changes the exp-pass body + the predicate
name + asm-shape/FFI tests; M18 changes register layout + leaf flag + inspect
goldens. No overlap.

---

## 10. Decisions & rationale (brainstorm trail)

| # | Fork | Choice | Why |
|---|------|--------|-----|
| D1 | Next axis | Axis 3 (M17) | Roadmap + DEVLOG "Next step" both point here; M16 unblocked structural validation; narrow, self-contained scope. |
| D2 | Validation contract | Two layers: bit-exact vs Rust port **and** tolerance vs libm | Layer 1 catches codegen regression (M15 convention); layer 2 is the independent accuracy oracle. Neither alone suffices: layer 1 alone can't catch a wrong coefficient; layer 2 alone drops the bit-exact regression anchor. |
| D3 | Domain | Softmax-tailored (`x ≤ 0`), no general `expf` | YAGNI; softmax is the only consumer; a general version ships untested overflow/NaN branches (the sweep can't reach them) — worse than omitting them. |
| D4 | Cleanup scope | Split: minimal swap (M17) + leaf-cleanup (M18) | Two distinct goals (drop libm vs honest leaf codegen); separate diffs stay reviewable. Viable **because** the predicate rename does not cascade (verified, §4). |
| D5 | Algorithm | Cody-Waite reduction + Taylor degree 7 + `2^z` bit-trick | `1/k!` coefficients keep **both** sides of the bit-exact contract auditable; minimax would move "magic numbers" from asm into the reference, weakening layer 1. Degree-7 is sub-ulp on the reduced interval; the extra term vs minimax is lost in load/store noise. |
| D6 | Constants | `.rodata` literal pool, file-local labels | Standard on both ISAs; inline immediates would likely make M17 slower than libm. Local labels avoid multi-object link collisions. Orthogonal to register layout. |
| D7 | Predicate | Rename to `has_softmax()` + change `--uir-verbose` label in M17 | Name describes what is detected (profile-neutral). The CLI label must change in M17 — a false `calls-extern-math: yes` violates Design Principle 5. |
| D8 | Accuracy gate | Hard ≤ 1 ulp, "measure → widen LN2 split" | Declaring ≤ 1 ulp before measuring is an over-claim; measure-then-tune (M16 golden-rule ethos) earns the same strong statement honestly. Fix is in the reduction, not the degree. |

---

## 11. Acceptance criteria

- [ ] No `bl _expf` / `call expf@PLT` anywhere in emitted output; no global
      `expf` symbol referenced.
- [ ] `emit_exp_inline` shared between standalone softmax and fused tail on both
      profiles.
- [ ] `.rodata` constant pool emitted once per file under file-local labels.
- [ ] `calls_extern_math` renamed to `has_softmax` everywhere; `--uir-verbose`
      prints `has-softmax`; no stale references in code, tests, or docs.
- [ ] Layer 1: softmax FFI tests bit-exact (`to_bits()`) against `exp_ref`.
- [ ] Layer 2: `exp_ref` within ≤ 1 ulp of libm over `x ∈ [−80, 0]`
      (LN2 split widened if needed to hold the bound).
- [ ] Underflow clamp exercised by a runtime FFI assertion (flush-to-0
      agreement with libm).
- [ ] M16 inspect goldens unchanged (leaf / callee-saved untouched in M17).
- [ ] M18 scope recorded in `PROJECT_SPEC.md`.
- [ ] All workspace gates green: `cargo fmt --all -- --check`,
      `cargo clippy --workspace --all-targets -- -D warnings`,
      `cargo test --workspace`.
</content>
</invoke>
