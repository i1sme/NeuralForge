# Milestone 14 — LayerNorm (A2 second brick) + LH-1/2/3 cleanup — Design

> Brainstormed: 2026-05-10
> Strategic axis: **Axis 2 — modelling depth** (PROJECT_SPEC §"Strategic Roadmap"). Continues the A2 (transformer block) axis whose first brick `add` (residual connections) closed in M13. Ships A2 second brick — LayerNorm — as a single `StdOp::LayerNorm` variant with internal multi-pass codegen (mirroring how `Softmax` is one node, not "exp + sum + divide" decomposed). FFN remains in M15+ as A2 third brick.
> Predecessor: M13 (N=4 + matmul fix on x86_64; `add` op as A2 first brick)
> Status: spec draft for plan synthesis

---

## 1. Overview

M14 has two deliverables, sequenced atomically:

1. **LH-1/2/3 cleanup on x86_64 `emit_linear`** — opener commit. `profiles/x86_64/src/ops/linear.rs` carries three documented latent hazards (PROJECT_SPEC §"Known Latent Hazards", opened in M13): LH-1 (N=2 + linear-with-bias), LH-2 (N=3 src ptr scratch %r8 alias), LH-3 (N=4 weight ptr scratch %r9 alias). All three are ABI-register conflicts at higher N — the same class of bug closed by M13 Task 1's `%r9 → %rbp` j-counter relocation in `emit_matmul`. M14's main integration fixture (`pre_ln_block.nfl`) at N=2 with `linear[w, b]` triggers LH-1, making closure mandatory by §LH process. LH-2/3 close in the same commit by proactive cleanup (memory rule: "triggered cleanup is an obligation" — once we touch the file, close all hazards).

2. **`StdOp::LayerNorm`** — A2 second brick. LayerNorm is normalization over the last dimension of each row (mean, variance, normalize), with optional learnable affine transform (γ scale, β bias). Surfaced in NFL as `x -> layernorm` (no affine) or `x -> layernorm[affine=true]` (with affine), mirroring the existing `linear` / `linear[bias=true]` opt-in pattern. Codegen is **3-pass per row** (mean → variance + inv_std → normalize + optional affine), modeled structurally after the existing `Softmax` 3-pass emitter (max-find → exp+sum → divide). Native hardware `sqrt` instruction on both profiles — **no libm dependency added**, keeping `expf` as the only remaining libm call (Strategic Roadmap Axis 3 alignment).

The strategic claim being validated is that **A2 decomposes into atomic ops with one new codegen pattern per milestone**: M13 shipped `add` (flat elementwise, no new pattern), M14 ships `layernorm` (3-pass with native sqrt + optional params), M15+ ships `ffn` (compositional, no new pattern). This keeps milestone scope at "one principled new codegen surface" — the rhythm M9-M13 established.

### Why these two together

LH-1 closure is mandatory regardless because M14's main integration fixture triggers it. LH-2/3 are defensively closed in the same commit because (a) same file, (b) same class of bug, (c) same fix mechanism (`%rbp` relocation), (d) §LH rule "leaving an entry here longer than one milestone is a process failure" applies the moment we touch the file. Splitting LH-2/3 into a separate milestone would be procedural noise without risk reduction.

### Non-goals (explicitly deferred)

- **FFN (Feed-Forward Network)** — deferred to M15. Composition of existing ops (`linear → relu → linear`); no new codegen pattern. M9-M13 cadence has been "one major new codegen surface per milestone" — bundling FFN with LayerNorm would oversize M14 (M10/Softmax-equivalent + M9-additional-emitter combined).
- **fp16/bf16/INT8 LayerNorm** — belongs to PROJECT_SPEC §"Open Questions" / quantisation. M14 is fp32-only.
- **Tunable `eps`** — hardcoded compile-time constant `1e-5` in codegen, no NFL surface. Softmax precedent: `axis=last` is hardcoded with no NFL surface either, by universal convention. Reversible to NFL surface via signature extension if quantisation milestones require tunable eps; B→C breaking change avoided by deferring.
- **Tunable `axis`** — hardcoded `last dim`, no NFL surface. Same rationale as eps — Softmax precedent.
- **RMSNorm** — different operation (γ-only, no centering). Not on the roadmap.
- **BatchNorm** — different statistics (per-channel across batch, not per-row), not relevant for transformer architectures.
- **Higher-N (N≥3) LayerNorm fixtures** — M14 validates emit_layernorm at N=1 and N=2 only. Higher arity requires callee-saved migration of row pointers on x86_64 (at N=4, %r8 and %r9 both become ABI-bound). Documented as deferred per §LH process: triggering fixture in M-future opens an LH entry, closure mandatory in that milestone.
- **Adding LayerNorm to OQ-BENCH harness** — bench fixtures expand based on profiling evidence (orthogonal signal), not enthusiasm at op landing time. M11 prior art: 3 fixtures chosen for orthogonal coverage. Defer to M-future after real-world use.
- **Span on any new shape error** — M14 needs no new `ShapeError` variants (existing `RankTooLow` covers the only failure mode, mirroring Softmax). M5c OQ-4 (Span on `BuildError`) remains dormant.

---

## 2. Goals

Ship a single PR with **5 atomic commits** plus a separate **opener cleanup commit** at the front:

1. **Commit 0 — `fix(m14): close LH-1/2/3 in x86_64 emit_linear`.** Audit `profiles/x86_64/src/ops/linear.rs` for ABI-register conflicts at N=2/3/4. Relocate j-counter and scratch pointers from INPUT_REGS-aliased registers to `%rbp` (callee-saved by unconditional prologue, never read by op bodies — M13 Task 1 precedent). Fallback for paths where `%rbp` is already in use: `pushq %r12` / `popq %r12` save-restore (M13 pre-Task-5 arm64 precedent). Add ABI-invariant unit tests for emit_linear at N=2/3/4 to `profiles/x86_64/src/tests.rs` (extends commit `c993712`'s coverage — that commit covered simple-loop emitters; emit_linear with its complex bias and PostOp::SoftmaxRow dispatch was not in scope at the time). PROJECT_SPEC §"Known Latent Hazards": remove rows LH-1, LH-2, LH-3.

2. **Commit 1 — `feat(m14): StdOp::LayerNorm foundation`.** `compiler/src/ir/stdlib.rs`: new `StdOp::LayerNorm` variant; `Signature` with `positional: []` and `named: [{ name: "affine", ty: Symbol, required: false }]` (mirrors `linear`'s `bias` slot exactly); `infer_output_shape` arm: input rank ≥ 2 (Softmax precedent — design constraint, not mathematical limit), output shape == input shape (identity); `validate_attrs` joins the `Ok(())` group with Linear/Relu/Softmax/etc; `resolve("layernorm") => Some(StdOp::LayerNorm)`; `Display for StdOp` "layernorm" arm. `profile-api/src/lib.rs`: `ParamKind` gains `LayerNormScale` and `LayerNormBias` variants. Both profiles: stub `emit_layernorm` returning `LowerError::UnsupportedOp { op: "layernorm".to_string(), span }` plus `walk_model` dispatch entry plus `ParamSlot` allocation logic (when `affine=true`: push `LayerNormScale` then `LayerNormBias`, both with shape `[input.last_dim]`; order is contract — see §5.3). `compiler/src/ir/types.rs`: extend `Display` impl for the new variant (CLAUDE.md mandate: every new IR variant must extend Display so `nflc parse --uir` rendering stays complete). Builder/IR unit tests: signature shape, `infer_output_shape` positive (rank=2,3,4), negative (RankTooLow). **Workspace stays green at this commit** (cargo build OK, cargo test OK — only compiler-side tests, profile FFI tests not added until Commit 4).

3. **Commit 2 — `feat(m14): arm64 emit_layernorm`.** New file `profiles/arm64/src/ops/layernorm.rs`. 3-pass codegen per §6 sketch. Native `fsqrt s, s` for sqrt. Leaf function (no FFI, no callee-saved promotion). Register plan from §7 (s_b reuses s2 after s_inv_d consumption — AAPCS64-safe; v8–v15 are callee-saved and intentionally avoided). Unit tests on emitted asm shape (analyser-style, mirroring `emit_softmax_emits_three_passes` precedent): assert exactly one `fsqrt`, exactly one `fdiv`, three loop labels per row.

4. **Commit 3 — `feat(m14): x86_64 emit_layernorm`.** New file `profiles/x86_64/src/ops/layernorm.rs`. 3-pass codegen mirroring Commit 2 in AT&T syntax. Native `sqrtss %xmm, %xmm` for sqrt. Op-local `pushq %r12` / `pushq %r13` at the **start of emit_layernorm body** (and `popq` mirror at end) **only when `affine=true`** — `compute_callee_saved` is NOT extended for LayerNorm; the affine save/restore is bracketed inside the op's emitted asm only. This preserves §3.2's invariant uniformly across the milestone (same mechanism as M13 pre-Task-5 arm64 emit_linear stp/ldp save/restore). Register plan from §8. Unit tests analogous to Commit 2.

5. **Commit 4 — `feat(m14): layernorm fixtures + per-profile FFI integration tests`.** Three positive fixtures (`tests/fixtures/{layernorm_no_affine,layernorm_affine,pre_ln_block}.nfl`) plus one negative (`tests/fixtures/negative/layernorm_rank_too_low.nfl`). Per-profile FFI integration tests in `profiles/{arm64,x86_64}/tests/integration.rs`: compile via `cc`, dlopen, call with random inputs/params, bit-exact compare against scalar Rust reference impl (sequential reduction — see §9.4 for auto-vectorization gotcha). Negative integration test in `compiler/tests/`: parse + IR build returns `ShapeError::RankTooLow`.

6. **Commit 5 — `docs(m14): documentation closure`.** DEVLOG entry; PROJECT_SPEC.md (Current Status, Strategic Roadmap A2 annotation); CLAUDE.md (Repository Structure tree, Current Status); `docs/language_reference/grammar.md` (`layernorm` op reference); `docs/language_reference/uir.md` (StdOp::LayerNorm entry); `docs/profile_guide/{arm64,x86_64}.md` (M14 ops sections + LH cleanup note on x86_64). See §10 for full inventory.

---

## 3. LH-1/2/3 opener (Commit 0) detailed

### 3.1 The bugs, restated

All three LH live in `profiles/x86_64/src/ops/linear.rs`. All three are **ABI-register conflicts** triggered when `output_reg() == INPUT_REGS[n_inputs+1]` aliases a register that emit_linear uses as scratch or counter.

| LH | N | Aliasing register | Symptom |
|----|---|-------------------|---------|
| LH-1 | 2 | `%rcx` (= INPUT_REGS[3] = output_reg at N=2) — used by bias-add path as j-counter or via j-counter aliasing base | bias-add reads j-counter as base address; silent corruption (correct shapes, wrong values; not SIGSEGV) |
| LH-2 | 3 | `%r8` (= INPUT_REGS[4] = output_reg at N=3) — used as src ptr scratch | src reads from wrong address |
| LH-3 | 4 | `%r9` (= INPUT_REGS[5] = output_reg at N=4) — used as weight ptr scratch | weight reads from wrong address |

The same class of bug as M12→M13's emit_matmul `%r9` j-counter (Task 1). M13 closed it in matmul; LH-1/2/3 are the analogous unfixed cases in linear.

### 3.2 Constraints on the fix

- **No new callee-saved registers added to function-level prologue.** `compute_callee_saved` in `profiles/x86_64/src/buffer.rs` already covers the matmul/softmax callee-saved set. Op-level push/pop (M13 emit_linear arm64 precedent) is acceptable; function-level prologue expansion is rejected.
- **Op body must NOT touch any ABI argument register** (M12 §9.1 invariant). All scratch/counter relocations must land on registers in non-INPUT_REGS scope.
- **Existing emit_linear unit tests continue to pass.** Adding new ABI-invariant tests at N=2/3/4 is mandatory (see §3.4); pre-existing tests at N=1 must remain green.

### 3.3 Fix mechanism (precedent-aligned)

Preference order for each LH:

1. **`%rbp` relocation** — first choice. `%rbp` is callee-saved by the unconditional prologue (`pushq %rbp` / `popq %rbp`); zero op bodies in the codebase read it (verified by M13 grep audit; re-verified at audit gate per §3.5). Adds zero new push/pop in op body.

2. **Op-level `pushq %r12` / `popq %r12` save-restore** — fallback when `%rbp` is already in use in some emit_linear code path (e.g., bias path may already use `%rbp` for something else — to be confirmed at audit). Symmetric with M13 arm64 emit_linear pre-Task-5 fix that used `stp/ldp x3` save-restore. 2 instructions overhead per affected linear op.

3. **Stack slot in function-level prologue** — last resort if both above are unavailable. ~3 extra memory accesses per inner-loop iteration (load/op/store), OoO pipeline absorbs.

Plan synthesis selects the final distribution per LH after reading the current state of `linear.rs`. Spec fixes only the constraint that all relocations land in non-INPUT_REGS scope and follow the preference order above.

### 3.4 Test coverage

- **ABI-invariant unit tests** in `profiles/x86_64/src/tests.rs` for emit_linear at N=2, N=3, N=4. Test pattern (extending commit `c993712`): lower a fixture at the target N, grep emitted asm body, assert no `INPUT_REGS[i]` for `i ∈ 0..n_inputs+2` appears as a counter/scratch in op body (only allowed in ABI context setup at function boundary). Three new tests, one per N. **Primary correctness signal for LH-2/3** (no FFI fixtures at N=3/4 — see below).

- **Pre-existing FFI integration coverage for LH-1**: `pre_ln_block.nfl` (M14 main fixture, Commit 4) is N=2 with `linear[w, b]`. Pre-fix, LH-1 corrupts output silently; bit-exact compare against Rust reference catches it immediately. Post-fix, green.

- **No new N=3/4 FFI fixtures.** N=3/4 + linear-with-bias is not an M14 use case. Adding fixtures only to defensively cover LH-2/3 would be scope creep. ABI-invariant unit tests above are sufficient.

### 3.5 Audit gate before closing opener

Before merging Commit 0:

1. Full `grep "INPUT_REGS\["` and `grep "%r[89]\|%rcx\|%rdx\|%rsi\|%rdi"` through `profiles/x86_64/src/ops/linear.rs` confirming all relocated scratches landed in `%rbp` / `%r10` / `%r11` / callee-saved range.
2. Verify zero new `pushq` in op bodies (function-level prologue unchanged).
3. Run `cargo test --workspace` — all existing tests green plus three new ABI-invariant tests pass.
4. PROJECT_SPEC §"Known Latent Hazards" table edited: LH-1, LH-2, LH-3 rows removed. If table becomes empty, header retained with comment "currently empty — populate as new latent hazards are discovered".

### 3.6 Out of scope for opener

- **arm64 emit_linear** — already fixed in M13 pre-Task-5 commit `c7fba5b` (stp/ldp x3/x4/x5 save/restore). No arm64 LH currently open.
- **emit_matmul on either profile** — already closed (M13 Task 1 for x86_64; arm64 was already correct).
- **Other simple-loop ops** (add, relu, mulscalar, dropout, softmax) — verified clean by `c993712` ABI-invariant tests.

---

## 4. NFL grammar surface

### 4.1 Surface forms

```nfl
# without affine — no params allocated
x -> layernorm

# with affine — γ (scale) and β (bias) allocated as auto-params
x -> layernorm[affine=true]
```

`affine` is a `Symbol` named arg, `required: false` — directly mirrors `linear[bias=true]`. No new grammar machinery; existing parser handles arity-overloaded brackets via the named-arg machinery.

### 4.2 Why this shape (not `[γ, β]`)

Two-positional-slots form `layernorm[γ, β]` was rejected during brainstorm:

1. **Type system mismatch.** γ and β are not NFL-bound identifiers (no `let γ = ...` declaration), not numeric literals, not `Tensor` references. Forcing them as positional `Symbol`s introduces "Symbol-as-keyword" semantics (parser must match literal name `gamma`/`beta`) — a new pattern with no precedent in the codebase.

2. **Signature should reflect configuration space.** γ and β are structurally bundled — there is no valid configuration where one exists without the other (γ-only is RMSNorm, a different operation). A two-slot signature visually suggests independent toggle, which the design forbids. Single `affine=true` toggle is honest about the single bit of decision.

3. **Direct precedent.** `linear[bias=true]` already established the pattern: optional affine-like params controlled by a single `Symbol` toggle, with shapes inferred and storage auto-allocated. M14 reuses the pattern verbatim.

### 4.3 Why no `eps` or `axis` in NFL surface

**Softmax precedent.** Softmax does not expose `dim`/`axis`; reduction over the last dim is hardcoded by universal convention. ε=1e-5 in fp32 LayerNorm is similarly universal across PyTorch / JAX / TF / all transformer literature.

**Design Principle 1 (explicit-over-implicit) does not apply.** That principle protects shapes, types, and dataflow from implicit inference. ε is a math-stability constant, not shape/type/dataflow. axis is a semantic op choice (Softmax made the same choice without surface).

**YAGNI + reversibility.** Adding NFL surface for ε now would establish the first numeric-with-default precedent in NFL (`dropout[rate]` is `required: true`, the only existing optional named arg `linear[bias]` is a Symbol toggle). Hardcoded → exposed is a one-line signature change later; exposed → hardcoded is a breaking change for all existing fixtures.

### 4.4 Default = no affine — by design

`layernorm` (bare, no brackets) does **not** allocate γ/β. This is intentional opt-in, mirroring `linear` defaulting to no bias. Rationale: explicit-over-implicit (Design Principle 1) — affine adds learnable parameters and a Pass 3 multiply-add; default-on would silently allocate params for users who wrote the bare form. Document in `grammar.md` and in `StdOp::LayerNorm` enum doc comment so the opt-in is visible and not mistaken for an oversight.

### 4.5 Grammar EBNF — no changes required

Generic op-with-bracket form (`<ident>[<args>]`) already covers both `layernorm` (no brackets) and `layernorm[affine=true]`. Zero EBNF additions.

### 4.6 Implementation-time check

Verify in `compiler/src/parser/` (or wherever Symbol-resolution happens): if `linear[bias=false]` is currently handled by `Symbol == "true"` check (anything else → no bias), then `layernorm[affine=false]` falls through automatically as no-affine, no extra branch needed. If a different mechanism is in use, an explicit branch in profile codegen `has_attr(node, "affine", "true")` is required.

---

## 5. UIR layer

### 5.1 `StdOp::LayerNorm` variant

```rust
// in compiler/src/ir/stdlib.rs
#[non_exhaustive]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    Matmul,
    MulScalar,
    Add,
    LayerNorm,   // M14 — A2 second brick
}

// in resolve():
"layernorm" => Some(StdOp::LayerNorm),

// in signature():
StdOp::LayerNorm => Signature {
    positional: &[],
    named: &[ArgSlot { name: "affine", ty: Symbol, required: false }],
},

// in infer_output_shape():
StdOp::LayerNorm => {
    let input = single_input(inputs)?;
    if input.rank() < 2 {
        return Err(ShapeError::RankTooLow {
            required: 2,
            actual: input.rank(),
        });
    }
    Ok(input.clone())   // identity — LayerNorm preserves shape
}

// in validate_attrs() — joins existing Ok-group:
StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul
    | StdOp::MulScalar | StdOp::Add | StdOp::LayerNorm => Ok(()),

// in Display for StdOp:
StdOp::LayerNorm => "layernorm",
```

### 5.2 No new `ShapeError` variants

Single-input, identity-shape op. The only failure mode is rank < 2, covered by existing `RankTooLow`. M5c OQ-4 (Span on shape errors) does not trigger.

### 5.3 `ParamKind` extension (in `profile-api/src/lib.rs`)

```rust
pub enum ParamKind {
    LinearWeight,
    LinearBias,
    LayerNormScale,   // M14 — γ, shape [input.last_dim]
    LayerNormBias,    // M14 — β, shape [input.last_dim]
}
```

**ParamSlot allocation order is a contract.** When `affine=true`, `walk_model` (in each profile's `codegen.rs`) MUST push `LayerNormScale` before `LayerNormBias`. Pass 3 emitter reads from `params_layout` by indexed `find` (precedent: `emit_linear` finds `LinearWeight` then `LinearBias` in that order, see `profiles/arm64/src/codegen.rs:158, 164`). If the order silently changes, weights and biases swap with no compile-time or runtime error — a silent correctness bug. Document the contract:

- In `profile-api/src/lib.rs` enum doc: "`LayerNormScale` MUST be pushed before `LayerNormBias` in `params_layout`. Mirror of `LinearWeight` / `LinearBias` ordering."
- In each profile's `walk_model` LayerNorm dispatch arm: inline comment "// γ before β — contract; see ParamKind doc."

### 5.4 ParamSlot allocation in `walk_model` (both profiles, symmetric)

```rust
// pseudo, in codegen::walk_model NodeKind::StdOp(StdOp::LayerNorm) arm:
if has_attr(node, "affine", "true") {
    let last_dim = input_shape.0.last().copied().unwrap();
    // Contract: γ before β.
    params_layout.push(ParamSlot {
        kind: ParamKind::LayerNormScale,
        origin_node: node_idx,
        shape: vec![last_dim],
    });
    params_layout.push(ParamSlot {
        kind: ParamKind::LayerNormBias,
        origin_node: node_idx,
        shape: vec![last_dim],
    });
}
// affine=false (or absent): zero ParamSlots — no params blob entries for this node.
```

### 5.5 Implementation-time check — `ParamKind` exhaustiveness

Verify whether `ParamKind` carries `#[non_exhaustive]` or not. Downstream uses appear to be equality matches (`.find(|s| s.kind == ParamKind::LinearWeight)`), not exhaustive `match` arms — adding variants does not require catch-all updates. If a `match` is found, add arms for the new variants in the same Commit 1.

### 5.6 No changes to `compute_is_leaf` / `compute_callee_saved` (either side)

`UirModel::calls_extern_math()` predicate stays bound only to Softmax (via `expf`). LayerNorm uses native sqrt, no FFI, leaf function. Same situation as `add` in M13.

**Profile-side `compute_callee_saved` is NOT extended for LayerNorm.** The x86_64 affine path's %r12/%r13 use is bracketed by op-local `pushq`/`popq` inside the emit_layernorm body (see §8.4) — function-level prologue is unchanged. This is uniform with §3.2's invariant for the LH cleanup commit.

### 5.7 No changes to `assign_buffers`

LayerNorm output shape == input shape, single output buffer of standard size. No scratch buffers allocated on the heap — μ, σ², inv_std live entirely in float registers across the per-row passes.

### 5.8 No changes to `LowerError` / `AbiContext` / `FnSig`

Existing variants suffice. Stub emit_layernorm in Commit 1 returns `LowerError::UnsupportedOp { op: "layernorm".to_string(), span: node.span }`. AbiContext's existing N-arity machinery (M12) covers N=1..2 cleanly. FnSig sees only an additional `ParamSlot` count change when `affine=true`.

---

## 6. Codegen pass structure (canonical)

### 6.1 Math

```
For each row of length D:
  μ        = (1/D) · Σ x_i
  σ²       = (1/D) · Σ (x_i − μ)²
  inv_std  = 1 / sqrt(σ² + ε)              # ε = 1e-5 hardcoded
  y_i      = (x_i − μ) · inv_std · γ_i + β_i      (with affine)
           = (x_i − μ) · inv_std                  (without affine)
```

### 6.2 Why 3-pass (rejected: 2-pass fused, Welford 1-pass)

**3-pass chosen over 2-pass-fused (sum + sum-of-squares simultaneous).** 2-pass fused computes `σ² = E[x²] − (E[x])²`, which suffers catastrophic cancellation when `|μ|` is comparable to `σ`. For pre-LN transformer activations this is "ok in practice", but that is exactly the class of caveat the project avoids elsewhere. Future LayerNorm use in debugging or arbitrary contexts would surface a numerical bug that is hard to diagnose. 3-pass eliminates the failure mode entirely: each pass computes a clean quantity with no cancellation. The ~25% bandwidth saving of 2-pass-fused is theoretical for typical D ≤ 1024 (rows fit in L1; re-reads are essentially free).

**3-pass chosen over Welford (1-pass online).** Welford updates μ and an M2 accumulator incrementally per element. Numerically most stable but per-element arithmetic is heavier (running mean + M2), and a separate normalize pass is still required → effectively 2-pass with heavier Pass 1. No advantage over 3-pass for fp32 transformer use; added complexity.

**3-pass mirrors Softmax.** Softmax is also 3-pass (max-find, exp+sum, divide). Reading `emit_softmax` and `emit_layernorm` side-by-side, a developer sees the identical skeleton. AI-native principle (regular structure, no exceptions) at the source level. M9-M13 cadence has consistently chosen "follow established project structure over micro-optimization without evidence".

### 6.3 Why native `sqrt` (rejected: libm `sqrtf`)

Both target ISAs ship single-instruction IEEE-754 correctly-rounded `sqrt`:
- arm64: `fsqrt s, s` (~10-20 cycle latency on M-series, throughput ~1/4 cycles)
- x86_64 SSE2: `sqrtss %xmm, %xmm` (~12-20 cycle latency, throughput ~1/6)

Using libm `sqrtf` would require:
- New `bl _sqrtf` (arm64) or `call sqrtf@PLT` (x86_64) emission
- Extending `UirModel::calls_extern_math()` predicate
- Extending `compute_callee_saved` and `compute_is_leaf` for LayerNorm
- ABI dance around the call (caller-saved register preservation)

All for a function call where a 1-instruction primitive exists. **Strategic Roadmap Axis 3 alignment**: `expf` is currently the only libm dependency; the bare-metal-expf milestone closes that. Adding `sqrtf` is anti-progress on the same axis. Native sqrt is unambiguously the right call — no aspect of the design favors libm here.

`expf` ≠ `sqrtf` in this respect: `expf` has no single-instruction equivalent (Taylor series or libm), so libm is the pragmatic choice there. `sqrtf` is structurally different.

### 6.4 inv_std hoisting — explicit emitter constraint

`inv_std = 1.0 / sqrtf(σ² + ε)` is computed **once** at the end of Pass 2 and held in a single scalar register through all of Pass 3. Pass 3 inner loop uses multiplication `(x_j − μ) · inv_std`, never division. The `fdiv`/`divss` for the reciprocal lives outside the hot loop.

- arm64 sequence (end of Pass 2): `fsqrt s_var, s_var` → `fmov s_one, #1.0` → `fdiv s_inv_std, s_one, s_var`
- x86_64 sequence: `sqrtss %xmm_var, %xmm_var` → `movss .Lone(%rip), %xmm_inv_std` → `divss %xmm_var, %xmm_inv_std`

**Explicit emitter constraint** (call out in code comment in `emit_layernorm` Pass 3 setup): "Pass 3 inner loop must contain zero `fdiv`/`divss` instructions; inv_std is hoisted from Pass 2 end." This prevents future maintainers from accidentally regressing to per-element division when refactoring.

### 6.5 Canonical pseudo-asm sketch (AArch64-flavored)

```asm
# pre-loop hoisted constants (loaded once per function call):
adrp  x_const, .Lconsts; add x_const, x_const, :lo12:.Lconsts
ldr   s_inv_d,  [x_const, #0]    # 1.0/D, precomputed at compile time
ldr   s_eps,    [x_const, #4]    # 1e-5
ldr   s_one,    [x_const, #8]    # 1.0

# === outer row loop: i in 0..B (B = product of leading dims, compile-time) ===
mov   x_i, #0
.Lrow_loop:
  # row pointers via shift-add (D known at compile time)
  add   x_in,  input_reg,  x_i, lsl #LOG2_D_BYTES
  add   x_out, output_reg, x_i, lsl #LOG2_D_BYTES

  # === Pass 1: μ = (1/D) · Σ x_j ===
  fmov  s_acc, wzr
  mov   x_j, #0
.Lp1:
    ldr   s_t, [x_in, x_j, lsl #2]
    fadd  s_acc, s_acc, s_t
    add   x_j, x_j, #1
    cmp   x_j, #D
    b.lt  .Lp1
  fmul  s_mean, s_acc, s_inv_d

  # === Pass 2: σ² = (1/D) · Σ (x_j − μ)²; inv_std = 1/sqrt(σ²+ε) ===
  fmov  s_acc, wzr
  mov   x_j, #0
.Lp2:
    ldr   s_t, [x_in, x_j, lsl #2]
    fsub  s_t, s_t, s_mean
    fmul  s_t, s_t, s_t
    fadd  s_acc, s_acc, s_t
    add   x_j, x_j, #1
    cmp   x_j, #D
    b.lt  .Lp2
  fmul  s_var, s_acc, s_inv_d
  fadd  s_var, s_var, s_eps
  fsqrt s_var, s_var
  fdiv  s_inv_std, s_one, s_var      # ← computed ONCE; held through Pass 3

  # === Pass 3: y_j = (x_j − μ) · inv_std [· γ_j + β_j] ===
  mov   x_j, #0
.Lp3:
    ldr   s_t, [x_in, x_j, lsl #2]
    fsub  s_t, s_t, s_mean
    fmul  s_t, s_t, s_inv_std        # ← fmul, NOT fdiv
    # === IF has_affine (compile-time branch in emitter): ===
    ldr   s_g, [x_gamma, x_j, lsl #2]
    fmul  s_t, s_t, s_g
    ldr   s_b, [x_beta,  x_j, lsl #2]
    fadd  s_t, s_t, s_b
    # === END affine ===
    str   s_t, [x_out, x_j, lsl #2]
    add   x_j, x_j, #1
    cmp   x_j, #D
    b.lt  .Lp3

  add  x_i, x_i, #1
  cmp  x_i, #B
  b.lt .Lrow_loop
```

### 6.6 Register allocation contract (M13 lesson generalized)

All scratch in `emit_layernorm` MUST live in non-INPUT_REGS scope:

| Profile | GPR scratch range | Float scratch |
|---------|-------------------|---------------|
| arm64   | x9–x16 (caller-saved, never ABI args) | s0–s7 (free in op body after args consumed) |
| x86_64  | %r10, %r11 (caller-saved, never ABI args) + %rbp (saved by prologue) for counters | %xmm0–%xmm7 (caller-saved) |

**Never use `INPUT_REGS[0..6]` in op body.** This is the M12 §9.1 invariant + the M13 lesson (LH-1/2/3 surfaced because emit_linear violated it). Fresh emit_layernorm code is written with this contract from the start, making LH-class bugs structurally impossible in the new code.

### 6.7 Single emitter function with compile-time affine branch

`emit_layernorm(asm, abi, node, has_affine: bool, params_layout)` — one function, with the Pass 3 affine block emitted conditionally based on `has_affine`. Mirrors `emit_linear`'s `if has_bias` conditional.

`has_affine` is determined at the dispatcher level (`walk_model` / `codegen.rs`) by `has_attr(node, "affine", "true")` and passed into the emitter. The emitter itself has no NFL grammar awareness.

---

## 7. arm64 emitter (`profiles/arm64/src/ops/layernorm.rs`)

### 7.1 Final register plan (AAPCS64-corrected)

| Purpose | Register | Live range |
|---------|----------|------------|
| `x_in` (row input ptr) | x9 | per-row, recomputed at row start |
| `x_out` (row output ptr) | x10 | per-row, recomputed at row start |
| `x_j` (inner counter) | x11 | per-pass |
| `x_i` (outer counter) | x12 | through outer row loop |
| `x_gamma` (γ base ptr, if affine) | x13 | through Pass 3 |
| `x_beta` (β base ptr, if affine) | x14 | through Pass 3 |
| `s_acc` (sum / sum-of-sq accumulator) | s0 | per-pass (clobbered between Pass 1 and Pass 2) |
| `s_var` (σ², σ²+ε, sqrt(σ²+ε)) | s0 (reuses `s_acc` — dead after Pass 2's final `fmul s_var, s_acc, s_inv_d`) | brief, only at end of Pass 2 (4 instructions) |
| `s_mean` | s1 | live Pass 2 + Pass 3 |
| `s_inv_d` (1/D constant) | s2 | live until end of Pass 2; **reused as `s_b` in Pass 3 affine** |
| `s_eps` (1e-5) | s3 | live through Pass 2 only |
| `s_one` (1.0) | s4 | live through Pass 2 only |
| `s_inv_std` | s5 | live through Pass 3 (§6.4 constraint) |
| `s_t` (per-element temp) | s6 | inner-loop temp |
| `s_g` (γ_j load) | s7 | inner-loop affine |
| `s_b` (β_j load) | **s2 (reused)** | inner-loop affine |

### 7.2 AAPCS64 callee-saved constraint — why s_b reuses s2, not s8

AAPCS64 §6.1.2 marks `v0–v7` (and thus `s0–s7`) caller-saved (free for op use), and `v8–v15` (and `s8–s15`) callee-saved (lower 64 bits must be preserved across function returns, regardless of leaf status). Writing `s8` in `emit_layernorm` without `stp/ldp d8` save-restore would silently corrupt `v8` of the caller — exactly an LH-class bug, in the float register file.

The fix avoids both `s8` and stack save-restore by reusing a dead constant register. After Pass 2 completes, `s_inv_d` (s2), `s_eps` (s3), and `s_one` (s4) are all consumed and dead — three free registers in the safe `s0–s7` range. `s_b` lands in `s2`. Live registers in Pass 3 inner loop: `s_mean` (s1), `s_inv_std` (s5), `s_t` (s6), `s_g` (s7), `s_b` (s2) = 5 simultaneously live. Comfortably under the 8-register caller-saved budget.

### 7.3 Implicit cost — constant reload at row start

Because `s_inv_d` (s2) is consumed by Pass 3 (now holding `s_b`), the next row's Pass 1 needs `s_inv_d` re-loaded from `.rodata`. Three additional instructions per row at outer-loop top:

```asm
adrp  x_const, .Lconsts; add x_const, x_const, :lo12:.Lconsts
ldr   s_inv_d, [x_const, #0]
```

(`s_eps` and `s_one` are also re-loaded if the same scheme is applied uniformly; alternatively, `s_eps` and `s_one` stay live across the outer loop in s3/s4 and only `s_inv_d` reloads.) Cost: 3-9 instructions × B rows. Negligible vs the per-row O(D) work. Document the implicit cost in `docs/profile_guide/arm64.md` so it does not surprise future readers.

### 7.4 Sqrt and reciprocal sequence

```asm
fsqrt s_var, s_var          # σ² + ε → sqrt(σ² + ε)
fmov  s_one, #1.0           # constant 1.0 (or use the live s4 if reload-strategy keeps it)
fdiv  s_inv_std, s_one, s_var
```

One `fsqrt`, one `fmov`, one `fdiv` — all once per row, all outside Pass 3 hot loop.

### 7.5 Constants in `.rodata`

```asm
# .section directive shown ELF-flavored for canonical readability;
# actual macOS arm64 emission uses Mach-O `.section __TEXT,__const`.
# Match existing arm64 .rodata convention at implementation time —
# see Softmax constants emission (`profiles/arm64/src/ops/softmax.rs`)
# for the canonical reference.
.section .rodata, "a"
.align 4
.Lconsts:
    .float  0.03125     # 1.0/D for D=32 (precomputed at lowering time)
    .float  1e-5        # ε
    .float  1.0         # used for reciprocal of sqrt
```

D varies by fixture; constants are emitted per-function based on the input shape. Compile-time computation of `1.0/D` avoids any runtime division.

### 7.6 No FFI, leaf function

`compute_is_leaf` returns true (LayerNorm has no `bl _expf`, no other extern calls). `compute_callee_saved` is unchanged on arm64 (no callee-saved registers added by LayerNorm — x9-x16 are caller-saved per AAPCS64 §6.1.1).

### 7.7 Unit tests (in `profiles/arm64/src/ops/layernorm.rs` or `profiles/arm64/src/tests.rs`)

- `emit_layernorm_no_affine_emits_three_passes` — assert exactly 3 loop labels (`.Lp1`, `.Lp2`, `.Lp3`), exactly one `fsqrt`, exactly one `fdiv`, zero `bl` (leaf).
- `emit_layernorm_affine_emits_three_passes_with_gamma_beta_loads` — additionally assert two `ldr` from γ/β base pointers in Pass 3 body, plus one `fmul` (γ scale) and one `fadd` (β bias).
- `emit_layernorm_uses_only_safe_float_registers` — grep emitted asm; assert no `s8`–`s15` writes (AAPCS64 callee-saved guard).
- `emit_layernorm_param_slot_order` — for an affine fixture, verify `params_layout` contains `LayerNormScale` at index `i` and `LayerNormBias` at index `i+1` (contract from §5.3).

---

## 8. x86_64 emitter (`profiles/x86_64/src/ops/layernorm.rs`)

### 8.1 Final register plan (N=1..2 scope)

| Purpose | Register | Notes |
|---------|----------|-------|
| `x_in` | %r8 | free at N≤2 (becomes output_reg at N=3, params_reg at N=4 — out of M14 scope) |
| `x_out` | %r9 | free at N≤2 (becomes output_reg at N=4 — out of M14 scope) |
| `x_j` | %r10 | caller-saved, never ABI |
| `x_i` | %r11 | caller-saved, never ABI |
| `x_gamma` | %r12 | **callee-saved** — `pushq %r12` / `popq %r12` in op prologue ONLY when has_affine |
| `x_beta` | %r13 | **callee-saved** — `pushq %r13` / `popq %r13` in op prologue ONLY when has_affine |
| `s_acc` | %xmm0 | per-pass |
| `s_mean` | %xmm1 | live Pass 2 + Pass 3 |
| `s_inv_d`, `s_eps`, `s_one` | %xmm2..%xmm4 | hoisted constants |
| `s_inv_std` | %xmm5 | §6.4 constraint |
| `s_t` | %xmm6 | inner-loop temp |
| `s_g`, `s_b` | %xmm7, %xmm8 | inner-loop affine |

`%xmm8` is fine on x86_64 (caller-saved per SysV) — no AAPCS64-style callee-saved concern in the xmm range. Constant reload trick from arm64 §7.2 is not needed here.

### 8.2 Sqrt and reciprocal sequence (AT&T)

```asm
sqrtss  %xmm_var, %xmm_var               # σ² + ε → sqrt(σ² + ε)
movss   .Lone(%rip), %xmm_inv_std        # load 1.0
divss   %xmm_var, %xmm_inv_std           # %xmm_inv_std /= %xmm_var → 1/sqrt(...)
```

One `sqrtss`, one `movss`, one `divss` — all once per row, outside Pass 3 hot loop.

### 8.3 Constants in `.rodata`

```asm
.section .rodata
.align 4
.Linv_d:    .float  0.03125     # 1.0/D
.Leps:      .float  1e-5
.Lone:      .float  1.0
```

Loaded RIP-relative: `movss .Linv_d(%rip), %xmm2` etc. Simpler than arm64's `adrp + add + ldr` sequence.

### 8.4 Affine-only op-local save/restore (NOT `compute_callee_saved` extension)

```asm
# Inside emit_layernorm emitted body, at the very start, ONLY when has_affine == true:
pushq %r12        # γ base pointer (callee-saved per SysV)
pushq %r13        # β base pointer

# ... Passes 1-3 ...

# Inside emit_layernorm emitted body, at the very end, ONLY when has_affine == true:
popq  %r13
popq  %r12
```

When `has_affine == false`, no push/pop — leaf function, cleaner.

**`compute_callee_saved` is NOT modified for LayerNorm.** The push/pop pairs above live entirely inside the asm string emitted by `emit_layernorm`, bracketing the affine path only. The function-level prologue/epilogue (driven by `compute_callee_saved` in `profiles/x86_64/src/buffer.rs`) is unchanged. This preserves §3.2's invariant ("No new callee-saved registers added to function-level prologue. Op-level push/pop acceptable; function-level prologue expansion is rejected") uniformly across the milestone — exact same mechanism as M13's pre-Task-5 arm64 fix in emit_linear (stp/ldp x3/x4/x5 save/restore inside op body, no function prologue change). A `compute_callee_saved` extension would have required `linear`-style helpers (`node_uses_layernorm_affine`) and a function-level prologue surface change — explicitly rejected for symmetry with the milestone-wide invariant.

**Stack alignment invariant (M-future foot-gun guard).** SysV ABI requires 16-byte stack alignment at `call` sites. Two op-local `pushq` add 16 bytes — alignment-preserving IF the function prologue's push count is even (`pushq %rbp` alone is +8 → odd). For M14 fixtures this is moot — none co-locate LayerNorm-with-affine with Softmax (the only op that emits `call expf@PLT` and thus the only op where stack alignment matters at a call site). If a future fixture combines LayerNorm-with-affine and Softmax in the same function, verify the function prologue's pushq count remains even at the `call expf@PLT` site, or add a one-time `subq $8, %rsp` / `addq $8, %rsp` adjustment around the affine push pair.

### 8.5 No FFI; leaf when no affine; op-local save/restore when affine

`compute_is_leaf` returns true (no `call expf@PLT`, no other extern). `compute_callee_saved` is **unchanged** by LayerNorm — function-level callee-saved set is driven entirely by other ops (matmul/softmax). The affine path's %r12/%r13 use is op-local (see §8.4), invisible to function-level prologue/epilogue.

### 8.6 Unit tests (mirror §7.7)

Same four test categories adapted to AT&T:
- Three loop labels, one `sqrtss`, one `divss`, zero `call`.
- Affine variant adds two `movss` from γ/β plus one `mulss` (γ) one `addss` (β).
- Caller-saved check is moot on x86_64 (no callee-saved xmm range), but assert no use of `INPUT_REGS[0..n_inputs+2]` as scratch (LH-class structural guard).
- ParamSlot order check.

### 8.7 Higher-N future-proofing — explicitly deferred

At N=3, %r8 becomes `output_reg`. At N=4, %r8 = `params_reg` and %r9 = `output_reg`. The current `x_in` / `x_out` plan breaks. Solution path (deferred to triggering fixture): migrate row pointers to callee-saved %rbx/%r14 (op-local push/pop in emit_layernorm body, same mechanism as §8.4) or hold them as stack slots reloaded at each pass start.

Document in `docs/profile_guide/x86_64.md` LayerNorm section: "validated for N=1..2 in M14; higher arity requires callee-saved migration of row pointers, deferred to triggering fixture per §LH process". If a M-future fixture introduces N≥3 + LayerNorm, an LH entry opens at that point and closure becomes mandatory in the milestone where the fixture lands.

---

## 9. Fixtures and test plan

### 9.1 Positive fixtures (in `tests/fixtures/`)

```nfl
# tests/fixtures/layernorm_no_affine.nfl — N=1, minimal sanity
fn forward(x: Tensor[8, 32]) -> Tensor[8, 32]:
    let normalized = x -> layernorm
    return normalized
```

```nfl
# tests/fixtures/layernorm_affine.nfl — N=1, with γ/β params
fn forward(x: Tensor[8, 32]) -> Tensor[8, 32]:
    let normalized = x -> layernorm[affine=true]
    return normalized
```

```nfl
# tests/fixtures/pre_ln_block.nfl — N=2, transformer-block integration
# Triggers LH-1 (N=2 + linear-with-bias); validates Commit 0 fix end-to-end
fn block(x: Tensor[8, 32], skip: Tensor[8, 32]) -> Tensor[8, 64]:
    let merged = x -> add[skip]
    let normalized = merged -> layernorm[affine=true]
    let projected = normalized -> linear[64, bias=true]
    return projected
```

(Exact `linear[64, bias=true]` vs `linear[out_dim=64, bias=true]` syntax: implementation-time check against existing `linear` fixtures — do not guess at spec time.)

### 9.2 Negative fixture (in `tests/fixtures/negative/`)

```nfl
# tests/fixtures/negative/layernorm_rank_too_low.nfl
# Expects ShapeError::RankTooLow at IR build (rank<2 reject, Softmax precedent)
fn forward(x: Tensor[32]) -> Tensor[32]:
    let normalized = x -> layernorm
    return normalized
```

Lives in `tests/fixtures/negative/` (compiler-level reject), **not** `profile-negative/` (lowering-level reject). `RankTooLow` fires at IR build, before any profile sees the node. M13 `add_shape_mismatch.nfl` precedent.

### 9.3 Test infrastructure layers

| Layer | Location | Coverage | Count |
|-------|----------|----------|-------|
| Builder/IR unit | `compiler/tests/` (or `compiler/src/ir/` inline) | signature shape, infer_output_shape positive (rank=2,3,4), infer_output_shape negative (RankTooLow), validate_attrs Ok | 4-5 |
| Codegen unit (per profile) | `profiles/{arm64,x86_64}/src/tests.rs` | emit_layernorm asm shape (no-affine, with-affine), ParamSlot allocation order (γ before β), 3-pass structural assertions (count of `fsqrt`/`sqrtss`, count of `fdiv`/`divss`, AAPCS64 register guard on arm64) | 3-4 × 2 = 6-8 |
| ABI-invariant unit (x86_64) | `profiles/x86_64/src/tests.rs` | emit_linear at N=2/3/4 — no INPUT_REGS used as scratch (LH-1/2/3 closure verification, §3.4) | 3 |
| FFI integration | `profiles/{arm64,x86_64}/tests/integration.rs` | bit-exact lowered asm vs Rust reference, all 3 positive fixtures × 2 profiles | 6 |
| Negative integration | `compiler/tests/` | parse + IR build on `layernorm_rank_too_low.nfl` returns `ShapeError::RankTooLow` | 1 |

**Total: 19-23 new tests.** Test count: 400 → ~419-423 on macOS arm64; +6 (x86_64 FFI) on Linux CI → ~426.

### 9.4 Bit-exact reference impl and the auto-vectorization gotcha

Reference Rust impl (in `profiles/{arm64,x86_64}/tests/common/mod.rs` or integration.rs — duplication acceptable per profile-isolation principle, or extract to a shared test util if convenient at implementation time):

```rust
fn layernorm_ref(input: &[f32], shape: &[usize],
                 gamma: Option<&[f32]>, beta: Option<&[f32]>) -> Vec<f32> {
    let d = *shape.last().unwrap();
    let n = shape.iter().take(shape.len() - 1).product::<usize>();
    let mut out = Vec::with_capacity(input.len());
    for r in 0..n {
        let row = &input[r * d..(r + 1) * d];
        // Sequential reduction — DO NOT use .iter().sum::<f32>() because
        // LLVM under -O3 may auto-vectorize it into a SIMD tree-reduction,
        // changing the order of float additions and breaking bit-exact
        // equivalence with the scalar 3-pass asm. Explicit `for` + `+=`
        // keeps reduction strictly left-to-right; LLVM does not reorder
        // f32 adds without -ffast-math (which Rust does not enable).
        let inv_d = 1.0_f32 / d as f32;  // bit-exact match with emitter's compile-time `1.0/D` constant in .rodata
        let mut sum = 0.0_f32;
        for &x in row { sum += x; }
        let mean = sum * inv_d;
        let mut sumsq = 0.0_f32;
        for &x in row { sumsq += (x - mean) * (x - mean); }
        let var = sumsq * inv_d;
        let inv_std = 1.0_f32 / (var + 1e-5_f32).sqrt();
        for (i, &x) in row.iter().enumerate() {
            let n = (x - mean) * inv_std;
            let val = match (gamma, beta) {
                (Some(g), Some(b)) => n * g[i] + b[i],
                _ => n,
            };
            out.push(val);
        }
    }
    out
}
```

**Critical implementation-time gate.** If a future maintainer rewrites the reference to use `.iter().sum::<f32>()` for "cleanliness", bit-exact compares may pass on some platforms and fail on others depending on whether LLVM chose to vectorize. Inline comment in the reference impl explaining the constraint is mandatory.

### 9.5 Edge case: D=1

`σ² = 0`, `inv_std = 1/sqrt(ε) ≈ 316.23`, output = `(x − μ) · inv_std · γ + β = 0 · inv_std · γ + β = β` (or 0 without affine). Mathematically defined, handled correctly by ε-padding without special-case in the emitter. No M14 fixture exercises D=1; not pre-engineered. Implicitly covered if a future fixture has D=1.

### 9.6 Bench scope decision — defer to M-future

M14 does **not** add LayerNorm to `bench/src/main.rs`. Three reasons:

1. **M11 prior art** — bench fixtures chosen for orthogonal signals (matmul-mass / large-K accumulator / expf-dominated dispatch). Expansion requires evidence of non-redundant signal, not enthusiasm at op landing.
2. **Memory rule "bench variance threshold"** — fixtures with `p95/median > 1.3×` are discounted from strategic arguments. A fresh op without baseline has unknown ratio; cannot make the case yet.
3. **Decoupled cadence** — `bench/results/<YYYY-MM-DD>.md` lands as post-merge follow-up commits. Bench fixture additions follow real-world signal accumulation, not milestone closing.

If M-future profiling shows LayerNorm as a meaningful tracked signal, add to `bench/` in a separate commit then.

---

## 10. Documentation deliverables (Commit 5)

All updates land in a single `docs(m14)` commit at the end of the feature series (M13 Task 6 precedent).

### 10.1 `DEVLOG.md` — new entry at top

Standard 4-section format (`What was done` / `Decisions made` / `Problems encountered` / `Next step`). See appendix §A for full skeleton.

### 10.2 `PROJECT_SPEC.md` — three targeted updates

| Section | Change |
|---------|--------|
| Current Status (line 170+) | Replace M13 paragraph with M14 paragraph: bumped test count, opener LH closure, LayerNorm landing, updated strategic direction line |
| Strategic Roadmap §"Axis 2" (line 199+) | Add: "M14 closed A2 second brick (LayerNorm). FFN — A2 third brick — remains in M15+." |
| §"Known Latent Hazards" (line 219+) | Remove rows LH-1, LH-2, LH-3. If table empty, retain header with comment "currently empty — populate as new latent hazards are discovered" |

### 10.3 `CLAUDE.md` — two targeted updates

| Section | Change |
|---------|--------|
| Repository Structure tree | Add `layernorm.rs` to `profiles/{arm64,x86_64}/src/ops/` subsections with brief annotation "M14: 3-pass mean/var/normalize, optional affine" |
| Current Status | Replace M13 paragraph with M14 paragraph (test count, key decisions referenced to DEVLOG) |

### 10.4 `docs/language_reference/grammar.md` — new `layernorm` section

```markdown
### `layernorm`

Layer normalization. Two forms:

- `x -> layernorm` — normalize without affine. Output shape == input shape.
- `x -> layernorm[affine=true]` — normalize and apply learnable γ (scale) and β (bias).
  γ and β shapes are both `[input.last_dim]`, allocated in params blob automatically.

**Constraints:**
- Input rank must be ≥ 2. Reduction is over the last dimension (per-row).
  This is a design constraint mirroring `softmax`, not a mathematical limit.
- `eps = 1e-5` and reduction axis (last dim) are hardcoded compile-time defaults.
  Tunable variants are deferred until quantisation milestones.

**Default:** no affine (`affine` is opt-in like `linear[bias=true]`, by design
principle "explicit over implicit").
```

### 10.5 `docs/language_reference/uir.md`

- Add `StdOp::LayerNorm` to enum reference.
- Document: input arity 1, attrs: optional `affine: Symbol`, output: identity shape, params: `[LayerNormScale, LayerNormBias]` in this order when `affine=true` (contract).

### 10.6 `docs/profile_guide/arm64.md` — new M14 ops section

- Register plan table from §7.1.
- 3-pass structure brief.
- "Native fsqrt; leaf function; no FFI."
- Implicit cost note: "constants `s_inv_d`, `s_eps`, `s_one` re-loaded from .rodata at top of each row's Pass 1, ~3 instructions × 3 constants overhead per row, negligible vs O(D) per-row work."
- AAPCS64 note: "emit_layernorm intentionally avoids `s8`–`s15` (callee-saved); `s_b` reuses `s2` after `s_inv_d` consumption to stay within the safe `s0`–`s7` range."

### 10.7 `docs/profile_guide/x86_64.md` — M14 ops section + cleanup section

Cleanup section (separate from LayerNorm):
- "M14 cleanup: LH-1/2/3 closed in emit_linear via %rbp relocation, uniform mechanism with M13 Task 1 emit_matmul fix."
- List of relocated scratches per LH.

LayerNorm section:
- Register plan table from §8.1.
- "Native sqrtss; leaf when affine=false."
- "Op-local %r12/%r13 push/pop INSIDE emit_layernorm body when affine=true (compute_callee_saved unchanged). Same mechanism as M13 pre-Task-5 arm64 emit_linear stp/ldp fix."
- "Validated for N=1..2 in M14 fixtures; higher N requires callee-saved migration of row pointers (deferred per §LH process)."

### 10.8 Viewer impact — none

The viewer (M9+) consumes the `Display` impl from `compiler/src/ir/types.rs`. The `Display` arm for `StdOp::LayerNorm` is added in Commit 1 (foundation), per CLAUDE.md mandate "every new IR node, field, or NodeKind variant must extend the Display impls". No separate viewer code change required.

### 10.9 Implementation-time docs gates

Before merging Commit 5:
1. All `.md` files preview-rendered (markdown linter clean).
2. `cargo doc --no-deps --workspace` builds without warnings.
3. `nflc parse <fixture> --uir` and `nflc parse <fixture> --uir-verbose` on each M14 fixture produce readable output for the `layernorm` node (no `<unrendered>` placeholders or panic).

---

## 11. Open Questions and Implementation-time Checks

Items deferred from spec to implementation plan because they require reading current code state. None block the spec; all are tactical.

### 11.1 `affine=false` Symbol resolution semantics

If `linear[bias=false]` currently works through a `Symbol == "true"` check (anything else, including literal "false", treats as opt-out), then `layernorm[affine=false]` falls through automatically as no-affine. If a different mechanism is in use, an explicit branch in profile codegen (`has_attr(node, "affine", "true")`) is required. Verify in `compiler/src/parser/` or wherever Symbol-resolution happens.

### 11.2 `ParamKind` exhaustiveness audit

Verify whether `ParamKind` carries `#[non_exhaustive]`. Downstream uses appear to be equality matches (`.find(|s| s.kind == ParamKind::LinearWeight)`), not exhaustive `match` arms. If a `match` is found, add arms for `LayerNormScale` and `LayerNormBias` in Commit 1.

### 11.3 `LowerError::UnsupportedOp` for stub

Commit 1 stubs return `LowerError::UnsupportedOp { op: "layernorm".to_string(), span: node.span }`. Verify that the message is informative enough at this transient state and that no test expects a specific message format. Replace with real implementation in Commits 2 and 3.

### 11.4 `linear[N, bias=true]` syntax in fixtures

The `pre_ln_block.nfl` fixture in §9.1 uses `linear[64, bias=true]`. Verify against existing `linear` fixtures whether the positional `out_dim` is named (`linear[out_dim=64, bias=true]`) or unnamed (`linear[64, bias=true]`). Adjust fixture accordingly.

### 11.5 Audit gate before closing LH opener (Commit 0)

See §3.5 for full checklist. Key items: grep `INPUT_REGS\[` and ABI register tokens through `linear.rs`; verify zero new `pushq` in op bodies; PROJECT_SPEC §"Known Latent Hazards" table edited.

### 11.6 `%rbp` availability per LH path

Plan synthesis must determine whether `%rbp` is available in each LH's specific code path (bias path for LH-1, src ptr path for LH-2, weight ptr path for LH-3). If `%rbp` is already in use in some path, the fallback (`pushq %r12` save-restore) applies. Final distribution per LH determined by audit, not pre-decided in spec.

### 11.7 Reference impl placement

`layernorm_ref` is mathematically identical for both profiles. May be duplicated in each profile's `tests/common/mod.rs` (profile isolation) or extracted to a shared test util. Both are acceptable per project conventions; choice is at implementation time based on convenience.

### 11.8 Constants reload strategy on arm64

§7.3 describes reloading `s_inv_d` per row from `.rodata`. Two sub-strategies are valid:
- (a) Reload all three constants (`s_inv_d`, `s_eps`, `s_one`) per row — uniform but 9 extra instructions/row.
- (b) Reload only `s_inv_d` (which is reused as `s_b` in Pass 3); keep `s_eps` and `s_one` live across the outer loop in s3/s4 — 3 extra instructions/row.

Choose (b) for the small win; (a) is acceptable if it simplifies emitter structure. Negligible runtime difference either way.

---

## Appendix A — DEVLOG entry skeleton

```markdown
## 2026-05-XX — Milestone 14 closed: LayerNorm (A2 second brick) + LH-1/2/3 cleanup

### What was done
- Opener (Commit 0): closed LH-1/2/3 in profiles/x86_64/src/ops/linear.rs via %rbp
  relocation (precedent M13 Task 1). ABI-invariant unit tests for emit_linear at
  N=2/3/4 added (extends c993712 precedent to complex ops).
- StdOp::LayerNorm UIR variant + signature (named `affine` Symbol toggle,
  required:false, mirrors linear[bias=true]) + shape inference (identity, rank ≥ 2
  mirroring Softmax). ParamKind extended with LayerNormScale + LayerNormBias
  variants in profile-api.
- arm64 emit_layernorm: 3-pass codegen (mean → variance+inv_std → normalize+optional
  affine), native fsqrt, leaf function, scratch in x9-x16 + s0-s7 (s_b reuses s2
  after s_inv_d consumption — AAPCS64-safe; v8-v15 intentionally avoided).
- x86_64 emit_layernorm: same 3-pass, native sqrtss, op-local %r12/%r13
  pushq/popq inside emit_layernorm body when affine=true (compute_callee_saved
  unchanged — same mechanism as M13 pre-Task-5 arm64 emit_linear stp/ldp fix).
- Fixtures: layernorm_no_affine.nfl, layernorm_affine.nfl, pre_ln_block.nfl
  (transformer-block N=2 integration, triggers LH-1 closure validation). Negative:
  tests/fixtures/negative/layernorm_rank_too_low.nfl.
- Test count: 400 → ~420 (macOS arm64); ~426 (Linux x86_64 CI).

### Decisions made
- 3-pass over 2-pass-fused — Softmax precedent + numerical robustness
  (no E[x²]−(E[x])² cancellation). 25% bandwidth saving of fused 2-pass absorbed
  by L1 for typical D ≤ 1024.
- Native fsqrt/sqrtss over libm sqrtf — zero FFI surface added, aligns with
  Strategic Roadmap Axis 3 ("drop libm dependency"). expf remains the only libm
  dependency.
- eps=1e-5 and axis=last hardcoded in codegen, no NFL surface — Softmax precedent
  (axis is implicit there too). Reversible to NFL surface via signature extension
  if M-future quantisation requires tunable eps.
- Default = no affine — explicit-over-implicit, mirrors linear[bias=true] opt-in
  semantics. layernorm without brackets does not allocate γ/β.
- LH-1/2/3 closed in one opener commit — same class of bug, same mechanism.
  Mandatory by §LH (LH-1) + proactive cleanup (LH-2/3).
- s_b reuses s2 (not s8) on arm64 — AAPCS64 v8-v15 are callee-saved; writing s8
  without save/restore would silently corrupt caller's v8.

### Problems encountered
- (filled in during implementation if any surface)

### Next step
A2 third brick — FFN (Feed-Forward Network) in M15. Compositional op
(linear → relu → linear), no new codegen pattern required. A2 fully closed
with FFN landing.

Trigger-driven cleanup status: OQ-7/8/9 + M5c OQ-4 still dormant through M14
(no triggers fired).
```
