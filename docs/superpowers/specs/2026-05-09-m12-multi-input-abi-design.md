# Milestone 12 — Multi-Input ABI (A1) — Design

> Brainstormed: 2026-05-09
> Strategic axis: **Axis 2 — modelling depth** (PROJECT_SPEC §"Strategic Roadmap"). First leg of A1 follow-up to M10's NFL v0.2 self-attention. Closes the gap between "IR supports multi-input" (already true) and "codegen + FFI surface support multi-input" (the M12 deliverable).
> Predecessor: M11 (OQ-BENCH harness)
> Status: spec draft for plan synthesis

---

## 1. Overview

M12 completes the multi-input dataflow that NFL v0.2 began in M10. The
NFL grammar, AST, and IR build pipeline already accept multiple
`variable_decl` statements per model — `compiler/src/ir/build.rs:319-334`
pushes every `VariableDecl` into `UirModel.inputs: Vec<NodeId>` without
restriction. The constraint preventing real multi-input use lives entirely
in the FFI ABI surface: `profile-api::FnSig.input_floats: usize` is
scalar, both architecture profiles take `model.inputs.first()` and
hardcode `x0` (arm64) / `%rdi` (x86_64) as the single input register, the
bench harness assumes a single input pointer per fixture, and integration
tests are written against a single-input `extern "C"` signature.

M12 evolves this surface to support up to four inputs via a per-arity
expanded register-args ABI (option **γ** from brainstorm Q2). Choice of γ
over the array-of-pointers alternative (option β) is grounded in three
correctness arguments surfaced during brainstorm:

1. **Zero migration cost for N=1 (the existing fixtures).** γ keeps the
   exact 3-register ABI for single-input models. β would require a
   one-off rewrite of every M3-M11 FFI test call site for no functional
   gain.
2. **Honest arity surface.** γ encodes arity in the function type. β
   hides arity in a runtime array length, which only relocates the
   complexity from type system to data structure.
3. **M10 register-preservation invariant preserved.** In β, `x0` becomes
   a metadata pointer (`*const *const f32`); the M10 spill logic in
   `emit_softmax` and `emit_matmul` would need to be rewritten because
   it currently treats x0 as input data. γ leaves N=1 codegen
   bit-identical to M11.

The deliverable shape is intentionally tight: one new per-profile module
(`abi.rs`), parameterization of all op-emitters by an `AbiContext`,
arity-aware FFI-call save/restore with mandatory 16-byte stack alignment,
and three new fixtures (one for N=2 sanity, one as the N=3 acceptance
fixture, one negative for N=5 → `LowerError::TooManyInputs`).

The strategic claim being validated is that **A1's ABI work is M7-sized
once the IR-side groundwork from M10 is honored**, not M9-sized as
originally framed in the M11 handoff primer. The handoff was conservative
because the grammar+IR readiness was not yet confirmed; brainstorm
discovered it in `compiler/src/ir/build.rs`.

---

## 2. Goals

Ship a single PR with ~6 atomic commits (final count delegated to
`writing-plans`):

1. **Foundation commit** — `profile-api/src/lib.rs`: `FnSig.input_floats:
   usize` → `inputs_floats: Vec<usize>`, new `LowerError::TooManyInputs
   { n: usize, max: usize, span: Span }` variant, `compiler/src/ir/build.rs`
   builds `inputs_floats` from `UirModel.inputs.iter()`.
2. **arm64 codegen commit** — new `profiles/arm64/src/abi.rs` with
   `AbiContext`, `BufferLoc::Input(usize)` change, `walk_model` threads
   `&abi` through every `emit_*` call. All ops migrated to use
   `abi.materialise_ptr` / `abi.input_reg` / `abi.params_reg` /
   `abi.output_reg` / `abi.emit_ffi_save` / `abi.emit_ffi_restore`.
   `emit_matmul` operand-loading rework (§5.1).
3. **x86_64 codegen commit** — mirror of #2 for SysV ABI.
4. **Fixtures + integration commit** — three new `.nfl` fixtures
   (`two_input_matmul.nfl`, `multi_input_attention.nfl`,
   `negative/too_many_inputs.nfl`); per-profile FFI integration tests
   that compile each via cc + dlopen + bit-exact compare against
   reference Rust impl.
5. **Bench commit** — `bench/src/main.rs` per-arity dispatch (match on
   `sig.inputs_floats.len()`); existing 3 bench fixtures (all N=1) run
   identically.
6. **Closure commit** — docs (`profile_guide/{arm64,x86_64}.md` get
   "Multi-input ABI" sections; `language_reference/grammar.md` clarifies
   multi-input semantics; `language_reference/uir.md` notes that
   codegen now honors all of `UirModel.inputs` not just the first);
   `PROJECT_SPEC.md` M12 row + Current Status + A1 closed in Strategic
   Roadmap; `CLAUDE.md` Current Status + Repository Structure tree gains
   `abi.rs`; `DEVLOG.md` M12 closure entry.

The merge of the PR closes M12.

---

## 3. Strategic Positioning

### 3.1 What M12 Proves

- **A1 closes A2's blocker.** Real Q/K/V multi-input is now expressible
  end-to-end. M13's transformer block (residual + LayerNorm + FFN) can
  build on this without re-litigating the ABI.
- **Stack alignment + LIFO discipline are formal invariants, not
  conventions.** `AbiContext::emit_ffi_save` emits a SP delta that is
  always a multiple of 16, padding with `xzr` (arm64) or `pushq %rax`
  (x86_64) when `ffi_save_set().len()` is odd. Acceptance gate #6
  (multi_input_attention bit-exact) silently fails if alignment is
  wrong — the underlying `_expf` will SIGBUS / #GP from a misaligned
  `movaps`.
- **Profile isolation holds under ABI surface change.** `AbiContext` is
  per-profile; profile-api gains only `inputs_floats: Vec<usize>` and
  one `LowerError` variant. No architecture-specific information leaks
  into the shared crate.
- **Test count grows monotonically.** 344 → ≥369 (floor +25, ceiling
  +35; final count from how many M3-M11 regression goldens are
  formalised as Rust tests vs CI shell checks).

### 3.2 What M12 Does Not Prove

- **Performance equivalence on multi-input models.** No bench fixture
  for N=3 in M12 — the bench harness gains per-arity dispatch but only
  the existing 3 (all N=1) fixtures remain. Benchmarking real Q/K/V is
  a follow-up and depends on whether N=3 numbers are useful given M11's
  variance discipline.
- **Stack-spilled inputs.** N>4 returns `TooManyInputs`. Implementing
  N≥5 requires SysV stack-spill on x86_64 — deferred until a real
  use-case (e.g., transformer attention with mask + position-bias as
  separate inputs) demands it.
- **Add operation, LayerNorm, FFN.** All three will be needed for A2's
  transformer block. They are M13-or-later work; M12 stays atomic on
  ABI.
- **Liveness-based spill optimisation.** `emit_ffi_save` always spills
  the entire ABI argument set (`ffi_save_set()`). A liveness analysis
  could spill only registers actually live across the call (~2 fewer
  registers in some cases). Deferred until measurable cost appears —
  current overhead per softmax row is ≤ 4 instructions.

### 3.3 Why Not Bundle With A2 Or A3

Brainstorm Q3 considered (b) bundling A1 with a partial A2 slice (e.g.,
`add` for residual connections), and (c) bundling A1 with A3 viewer
annotations. Both rejected on three correctness grounds:

- **`BufferLoc::Input` becomes `BufferLoc::Input(usize)` — a type-level
  change.** If new ops are added simultaneously, debugging which
  emitter introduced a buffer-assignment bug is harder.
- **`emit_softmax` and `emit_matmul` spill logic must cover N+2 registers
  (not 3).** Bundled changes obscure spill bugs that only manifest on
  specific op sequences.
- **Acceptance fixture must specifically read a non-x0 input AFTER an
  external call.** A bundled milestone risks accidentally writing
  fixtures that don't exercise the post-FFI register-survival path.
  M12's `multi_input_attention.nfl` is purpose-built: `out = attn ->
  matmul[v]` reads `v` (in x2 / %rdx) after `bl _expf`.

---

## 4. ABI Choice (γ Confirmed)

### 4.1 Register Layout

For a model with `N` inputs (declared in source-order via `variable_decl`
statements), function arity is `N + 2`. All arguments fit in registers;
M12 caps N at 4 to keep both profiles within their register-only argument
windows.

**arm64 (AAPCS):**

| N | x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 |
|---|----|----|----|----|----|----|----|----|
| 1 | in₀ | params | out | — | — | — | — | — |
| 2 | in₀ | in₁ | params | out | — | — | — | — |
| 3 | in₀ | in₁ | in₂ | params | out | — | — | — |
| 4 | in₀ | in₁ | in₂ | in₃ | params | out | — | — |

**x86_64 (SysV AMD64):**

| N | %rdi | %rsi | %rdx | %rcx | %r8 | %r9 |
|---|------|------|------|------|-----|-----|
| 1 | in₀ | params | out | — | — | — |
| 2 | in₀ | in₁ | params | out | — | — |
| 3 | in₀ | in₁ | in₂ | params | out | — |
| 4 | in₀ | in₁ | in₂ | in₃ | params | out |

`out` is `*mut f32`; all others are `*const f32`. Source-order of
`variable_decl` statements determines register assignment (lexical order
in source = ABI register order = `model.inputs[i]` order).

### 4.2 N=1 Backwards Compatibility

For N=1 the register layout is identical to M11 (`x0`=in, `x1`=params,
`x2`=out / `%rdi`=in, `%rsi`=params, `%rdx`=out). All M3-M11 fixtures
must produce **bit-exact** assembly under the M12 codegen. This is
acceptance gate #4 (§9).

The mechanism that delivers this invariant: `AbiContext { n_inputs: 1 }`
returns `input_reg(0) = "x0"`, `params_reg() = "x1"`, `output_reg() =
"x2"`, and `ffi_save_set() = &["x0", "x1", "x2"]`. Every existing
hardcoded reference to `x0`/`x1`/`x2` in op-emitters is replaced by
the corresponding `abi.*_reg()` call; for N=1 this evaluates to the same
string literal.

### 4.3 N>4 Rejection

```rust
pub enum LowerError {
    UnsupportedOp { op: String, span: Span },
    ShapeNotConcrete { span: Span },
    UnsupportedPostOp { op: String, span: Span },
    TooManyInputs { n: usize, max: usize, span: Span },
}
```

`max` is always 4 for both profiles in M12. The structurally identical
shape on both profiles preserves the profile-isolation principle: error
display ("model has N inputs but profile supports max M") is profile-
neutral. Negative test fixture `tests/fixtures/negative/too_many_inputs.nfl`
declares 5 inputs and is exercised on both profiles in integration
tests.

---

## 5. Architectural Connector — `AbiContext`

### 5.1 BufferLoc enum change (per-profile)

```rust
// profiles/{arm64,x86_64}/src/buffer.rs
pub enum BufferLoc {
    Input(usize),               // was: Input. usize = position in model.inputs.
    Params,
    Output,
    Stack { offset: u32 },
}
```

`assign_buffers` for an `Input`-kind node looks up the node's position
in `model.inputs` and emits `BufferLoc::Input(idx)`. All other variants
unchanged. The Rust compiler enforces exhaustive `match` updates across
buffer.rs, codegen.rs, and op modules — no silent miss possible.

### 5.2 AbiContext (per-profile, new file)

**`profiles/arm64/src/abi.rs`:**

```rust
const INPUT_REGS: &[&str] = &["x0", "x1", "x2", "x3", "x4", "x5"];
// First 6 of the AAPCS x0-x7 argument registers. M12 caps N+2 ≤ 6
// (i.e., N ≤ 4); reserved x6, x7 for future ABI extensions without
// re-reflowing this table.

pub(crate) struct AbiContext {
    pub n_inputs: usize,
}

impl AbiContext {
    pub fn input_reg(&self, idx: usize) -> &'static str {
        INPUT_REGS[idx]
    }
    pub fn params_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs]
    }
    pub fn output_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs + 1]
    }

    /// All ABI-argument registers in use by this function. Equal to
    /// `INPUT_REGS[0..n_inputs+2]`. This is the conservative caller-
    /// saved set to spill across any FFI call (`bl _expf` today;
    /// generalises to any future external call).
    pub fn ffi_save_set(&self) -> &[&'static str] {
        &INPUT_REGS[..self.n_inputs + 2]
    }

    /// Materialise a BufferLoc into a register.
    /// - Input(i) / Params / Output: a `mov dst_reg, ABI_reg`.
    /// - Stack { offset }: an `add dst_reg, sp, #offset`.
    pub fn materialise_ptr(&self, loc: BufferLoc, dst_reg: &str, asm: &mut String) {
        match loc {
            BufferLoc::Input(i)         => *asm += &format!("    mov {dst_reg}, {}\n", self.input_reg(i)),
            BufferLoc::Params           => *asm += &format!("    mov {dst_reg}, {}\n", self.params_reg()),
            BufferLoc::Output           => *asm += &format!("    mov {dst_reg}, {}\n", self.output_reg()),
            BufferLoc::Stack { offset } => *asm += &format!("    add {dst_reg}, sp, #{offset}\n"),
        }
    }

    /// Emit FFI-call save block. Always pushes pairs (stp); pads the
    /// last odd register with xzr to maintain 16-byte SP alignment.
    /// Total SP delta is always a multiple of 16 bytes.
    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let mut i = 0;
        while i < regs.len() {
            let a = regs[i];
            let b = if i + 1 < regs.len() { regs[i + 1] } else { "xzr" };
            *asm += &format!("    stp {a}, {b}, [sp, #-16]!\n");
            i += 2;
        }
    }

    /// Emit FFI-call restore block in strict LIFO order relative to
    /// emit_ffi_save. xzr-padded slot round-trips harmlessly (xzr is
    /// the zero register; restoring into it is a write-discard).
    pub fn emit_ffi_restore(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let n = regs.len();
        let mut pairs: Vec<(&str, &str)> = Vec::with_capacity(n.div_ceil(2));
        let mut i = 0;
        while i < n {
            let a = regs[i];
            let b = if i + 1 < n { regs[i + 1] } else { "xzr" };
            pairs.push((a, b));
            i += 2;
        }
        for (a, b) in pairs.iter().rev() {
            *asm += &format!("    ldp {a}, {b}, [sp], #16\n");
        }
    }
}
```

**`profiles/x86_64/src/abi.rs`:**

```rust
const INPUT_REGS: &[&str] = &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];
// All 6 SysV AMD64 GP argument registers. M12 caps N+2 ≤ 6 (i.e.,
// N ≤ 4). N=5+ requires stack-spill and is deferred.

pub(crate) struct AbiContext {
    pub n_inputs: usize,
}

impl AbiContext {
    // input_reg / params_reg / output_reg / ffi_save_set / materialise_ptr
    // structurally identical to arm64 above.

    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        for r in regs {
            *asm += &format!("    pushq {r}\n");
        }
        // Pad to even number of pushes for 16-byte SP alignment at the
        // call instruction (SysV requires (rsp+8) % 16 == 0 entering
        // the callee; each pushq is 8 bytes; even count keeps rsp
        // aligned to 16 across our save block).
        if regs.len() % 2 != 0 {
            *asm += "    pushq %rax           # 16-byte alignment padding\n";
        }
    }

    pub fn emit_ffi_restore(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        if regs.len() % 2 != 0 {
            *asm += "    popq %rax            # discard alignment padding\n";
        }
        for r in regs.iter().rev() {
            *asm += &format!("    popq {r}\n");
        }
    }
}
```

### 5.3 Threading through walk_model

```rust
// profiles/arm64/src/codegen.rs (schematic)
pub(crate) fn walk_model(model: &UirModel, ...) -> Result<String, LowerError> {
    let n = model.inputs.len();
    if n + 2 > INPUT_REGS.len() {
        return Err(LowerError::TooManyInputs {
            n,
            max: INPUT_REGS.len() - 2,
            span: model.source_span,
        });
    }
    let abi = AbiContext { n_inputs: n };

    emit_prologue(&abi, asm);
    for &node_id in &model.execution_order {
        match &model.nodes[node_id].kind {
            NodeKind::Linear { .. }    => emit_linear(&abi, ..., asm),
            NodeKind::Softmax { .. }   => emit_softmax(&abi, ..., asm),
            NodeKind::Matmul { .. }    => emit_matmul(&abi, ..., asm),
            NodeKind::MulScalar { .. } => emit_mulscalar(&abi, ..., asm),
            NodeKind::Relu { .. }      => emit_relu(&abi, ..., asm),
            NodeKind::Dropout { .. }   => emit_dropout(&abi, ..., asm),
            NodeKind::Input { .. }     => { /* no-op; ABI register already holds it */ }
        }
    }
    emit_epilogue(&abi, asm);
    Ok(asm)
}
```

`AbiContext` is constructed once at function entry and threaded by
`&abi` to every emitter. This is the single point of truth; emitters
hold no architecture-specific knowledge about which register holds what.

### 5.4 emit_softmax with AbiContext

```rust
fn emit_softmax(abi: &AbiContext, ..., asm: &mut String) {
    // ... phase 1 (row-max), phase 2 (set up exp loop) ...

    abi.emit_ffi_save(asm);                      // arity-aware spill
    *asm += "    bl _expf\n";                    // arm64; "call expf@PLT" on x86_64
    abi.emit_ffi_restore(asm);                   // strict LIFO restore

    // ... phase 3 (sum), phase 4 (normalise) ...
}
```

The call to `bl _expf` is wrapped by `emit_ffi_save` / `emit_ffi_restore`.
Phase 2's loop counters and accumulators use callee-saved registers
(s8/s9 on arm64 today, per M10's pattern; carried forward unchanged).
Phase 3 and phase 4 read input pointers via `abi.materialise_ptr` —
which after restore returns to its pre-call value.

---

## 6. Stack Alignment + LIFO Discipline (Hard Invariant)

### 6.1 Alignment rule

```
ffi_save_set.len() = N + 2

arm64:
    pair_count = (N + 2 + 1) / 2          // ceil-div by 2
    sp_delta   = pair_count * 16          // always multiple of 16 ✓
    if (N + 2) is odd:
        last pair = (ffi_save_set[-1], "xzr")
    else:
        last pair = (ffi_save_set[-2], ffi_save_set[-1])

x86_64:
    push_count = N + 2
    if push_count is odd:
        push_count += 1                   // pushq %rax padding
    sp_delta   = push_count * 8           // always multiple of 16 ✓
```

### 6.2 Concrete examples

**N=1 (current M3-M11 behaviour, regression-protected):**

arm64 (3 regs, odd):
```
stp x0, x1, [sp, #-16]!
stp x2, xzr, [sp, #-16]!
; sp_delta = 32 bytes
```

x86_64 (3 regs, odd):
```
pushq %rdi
pushq %rsi
pushq %rdx
pushq %rax            # padding
; sp_delta = 32 bytes
```

**N=2 (4 regs, even — no padding):**

arm64:
```
stp x0, x1, [sp, #-16]!
stp x2, x3, [sp, #-16]!
; sp_delta = 32 bytes
```

x86_64:
```
pushq %rdi
pushq %rsi
pushq %rdx
pushq %rcx
; sp_delta = 32 bytes
```

**N=3 (5 regs, odd — the acceptance fixture's case):**

arm64:
```
stp x0, x1, [sp, #-16]!
stp x2, x3, [sp, #-16]!
stp x4, xzr, [sp, #-16]!
; sp_delta = 48 bytes
```

x86_64:
```
pushq %rdi
pushq %rsi
pushq %rdx
pushq %rcx
pushq %r8
pushq %rax            # padding
; sp_delta = 48 bytes
```

**N=4 (6 regs, even — no padding):**

arm64:
```
stp x0, x1, [sp, #-16]!
stp x2, x3, [sp, #-16]!
stp x4, x5, [sp, #-16]!
; sp_delta = 48 bytes
```

x86_64:
```
pushq %rdi
pushq %rsi
pushq %rdx
pushq %rcx
pushq %r8
pushq %r9
; sp_delta = 48 bytes
```

### 6.3 LIFO restore

`emit_ffi_restore` walks pairs / pushes in **reverse** order from
`emit_ffi_save`. For arm64 N=3:

```
ldp x4, xzr, [sp], #16    ; undo pair 2
ldp x2, x3, [sp], #16     ; undo pair 1
ldp x0, x1, [sp], #16     ; undo pair 0
```

For x86_64 N=3:

```
popq %rax                 ; discard padding
popq %r8
popq %rcx
popq %rdx
popq %rsi
popq %rdi
```

### 6.4 Invariants (unit-tested)

For N ∈ {1, 2, 3, 4} on each profile:

1. SP delta from `emit_ffi_save` is a positive multiple of 16.
2. SP delta from `emit_ffi_restore` exactly negates that of
   `emit_ffi_save` (save+restore returns SP to its starting value).
3. Pair / push order in `emit_ffi_restore` is the strict reverse of
   `emit_ffi_save` (LIFO).
4. Every register in `ffi_save_set()` appears exactly once in the save
   block and exactly once in the restore block.
5. `xzr` (arm64) / `%rax` (x86_64) padding appears iff
   `ffi_save_set().len()` is odd.

---

## 7. Acceptance Fixtures

### 7.1 `tests/fixtures/two_input_matmul.nfl` (N=2 sanity)

```
# Two-input matmul. Sanity for N=2 register layout (a→x0/%rdi,
# b→x1/%rsi, params at x2/%rdx, output at x3/%rcx). Does NOT
# exercise post-FFI register survival — that's the multi_input_
# attention fixture's job. This fixture exists purely to verify
# arity=2 codegen path in isolation, with no _expf-call coupling.

model TwoInputMatmul [m=4, k=8, n=4]:
    a: Tensor[m, k]
    b: Tensor[k, n]

    out: Tensor[m, n] = a -> matmul[b]
```

Per-profile FFI integration test: build via cc + dlopen, fill `a` and
`b` with deterministic seeded random data, call forward, compare
output with reference Rust matmul implementation. Bit-exact match
required.

### 7.2 `tests/fixtures/multi_input_attention.nfl` (N=3 acceptance)

```
# Multi-input self-attention. THE acceptance fixture for M12:
# v is consumed AFTER softmax via `attn -> matmul[v]`, which means
# the v-pointer register (x2 on arm64, %rdx on x86_64) MUST survive
# `bl _expf` / `call expf@PLT`. This is the exact code path that
# AbiContext::ffi_save_set's correctness depends on — no other
# fixture exercises post-FFI-call register survival for a non-x0
# input.
#
# Contrast with M10's self_attention.nfl: that used a single input
# x where q=k=v=x, so the v-pointer was effectively x0, which
# already survived M10's spill.

model SelfAttention [batch=2, heads=4, seq=16, head_dim=16]:
    q: Tensor[batch, heads, seq, head_dim]
    k: Tensor[batch, heads, head_dim, seq]
    v: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = q -> matmul[k]
    scaled: Tensor[batch, heads, seq, seq] = scores -> mul_scalar[0.25]
    attn:   Tensor[batch, heads, seq, seq] = scaled -> softmax
    out:    Tensor[batch, heads, seq, head_dim] = attn -> matmul[v]
```

Per-profile FFI integration test: same shape as 7.1 but with three
seeded random inputs. Compute reference output in Rust (Q@Kᵀ via
matmul, scale, softmax over last axis, attn@V via matmul). Bit-exact
match required.

The test silently fails (numerical mismatch) if any of:
- v-pointer register clobbered by `_expf` (ffi_save_set incomplete).
- `_expf` SIGBUS / #GP from misaligned SP (alignment padding wrong).
- LIFO restore order wrong (one of the pointers restored to the
  wrong slot, e.g. x2 ↔ x4 swap).

Any of these bugs surfaces as either a numerical mismatch or a process
crash on the very first FFI call. Both manifest cleanly in test
output.

### 7.3 `tests/fixtures/negative/too_many_inputs.nfl` (N=5 negative)

```
# 5 inputs, exceeds M12 N=4 hard-cap on both profiles. Parser must
# accept (grammar allows any number of variable_decls); IR build
# must accept (UirModel.inputs is a Vec<NodeId> with no limit);
# profile lower() must reject with LowerError::TooManyInputs.
#
# This fixture is the regression test that the cap is enforced
# uniformly on both profiles.

model TooManyInputs [d=8]:
    a: Tensor[d]
    b: Tensor[d]
    c: Tensor[d]
    d_in: Tensor[d]
    e: Tensor[d]

    out: Tensor[d] = a -> linear[features=d]
```

Per-profile integration test: `parse + build_ir` succeeds; `profile.lower(&uir)`
returns `Err(LowerError::TooManyInputs { n: 5, max: 4, .. })`. Both
profiles must produce structurally identical errors (same `n`, same
`max`, span pointing to model declaration).

---

## 8. Ripple Map

| Layer / file | Change | LoC |
|---|---|---|
| `profile-api/src/lib.rs` | `FnSig.input_floats: usize` → `inputs_floats: Vec<usize>`; new `LowerError::TooManyInputs` variant; `Display` for new variant | ~15 |
| `compiler/src/ir/build.rs` | `inputs_floats` collected via `model.inputs.iter().map(|id| float_count(model.nodes[id].ty)).collect()` | ~5 |
| `profiles/arm64/src/abi.rs` | **new** — `AbiContext`, `INPUT_REGS`, full save/restore + materialise_ptr per §5.2 | ~120 |
| `profiles/arm64/src/buffer.rs` | `BufferLoc::Input` → `BufferLoc::Input(usize)`; `assign_buffers` looks up index in `model.inputs` | ~15 |
| `profiles/arm64/src/codegen.rs` | `walk_model` constructs `AbiContext`, threads `&abi` through op dispatch; N>4 → `TooManyInputs` | ~30 |
| `profiles/arm64/src/ops/linear.rs` | replace hardcoded `x0`/`x1`/`x2` with `abi.materialise_ptr` / `abi.input_reg(idx)` / `abi.params_reg()` / `abi.output_reg()` | ~25 |
| `profiles/arm64/src/ops/matmul.rs` | same + **operand-loading rework** (§9.1): materialise all operand pointers into scratch registers (x9–x15) before any internal spill | ~50 |
| `profiles/arm64/src/ops/softmax.rs` | replace manual `stp/ldp` block around `bl _expf` with `abi.emit_ffi_save` / `abi.emit_ffi_restore` | ~20 |
| `profiles/arm64/src/ops/mulscalar.rs` | `abi.materialise_ptr` for input/output | ~15 |
| `profiles/arm64/src/ops/relu.rs` | same | ~15 |
| `profiles/arm64/src/ops/dropout.rs` | `emit_dropout_copy` (BufferLoc::Output path) uses `abi.output_reg()` | ~10 |
| `profiles/arm64/src/asm.rs` | unchanged — prologue/epilogue callee-saved set is driven by `calls_extern_math()`, not by input register count | 0 |
| `profiles/arm64/src/tests.rs` | unit tests for `AbiContext`: alignment balance for N∈{1..4}, LIFO restore, `ffi_save_set` cardinality, `materialise_ptr` output for each variant | ~80 |
| `profiles/x86_64/src/abi.rs` | **new** — mirror of arm64 for SysV, `pushq`/`popq` with `%rax` padding | ~120 |
| `profiles/x86_64/src/buffer.rs` | mirror | ~15 |
| `profiles/x86_64/src/codegen.rs` | mirror | ~30 |
| `profiles/x86_64/src/ops/{linear,matmul,softmax,mulscalar,relu,dropout}.rs` | mirror of arm64 op changes | ~150 |
| `profiles/x86_64/src/tests.rs` | mirror of arm64 unit tests | ~80 |
| `profiles/{arm64,x86_64}/tests/integration.rs` | new tests: `two_input_matmul_match_numerically`, `multi_input_attention_match_numerically`, `too_many_inputs_returns_error` | ~120 |
| `tests/fixtures/two_input_matmul.nfl` | **new** | 15 |
| `tests/fixtures/multi_input_attention.nfl` | **new** | 20 |
| `tests/fixtures/negative/too_many_inputs.nfl` | **new** | 12 |
| `bench/src/main.rs` | per-arity dispatch; existing 3 fixtures all N=1, run identically | ~50 |
| `docs/profile_guide/arm64.md` | new "Multi-Input ABI" section: register tables (§4.1) + alignment rule (§6.1) + LIFO note (§6.3) | ~40 |
| `docs/profile_guide/x86_64.md` | mirror for SysV | ~40 |
| `docs/language_reference/uir.md` | clarification: `model.inputs: Vec<NodeId>` is now consumed in full by codegen, not just the first | ~10 |
| `docs/language_reference/grammar.md` | note: grammar already supports multi-input via multiple `variable_decl` statements; lexical declaration order = ABI register order; convention is to declare all inputs at top of model body | ~15 |
| `PROJECT_SPEC.md` | M12 row in milestones table; Current Status updated; Strategic Roadmap marks A1's "first leg" closed | ~30 |
| `CLAUDE.md` | Current Status → M12; Repository Structure tree gains `abi.rs` files | ~10 |
| `DEVLOG.md` | M12 closure entry | ~50 |
| **Total LoC** | (production + tests + docs, rough) | **~1200** |

---

## 9. Risks + Mitigations

### 9.1 emit_matmul operand-loading rework (highest risk)

**Risk.** Existing M10 `emit_matmul` (arm64) spills `x1`/`x2` via `stp/ldp`
around the outer loop, treating them as scratch inside the loop. This
assumes `x1` = params and `x2` = output. With N≥2 multi-input, `x1`
holds a **second input pointer** (e.g., `k` in `q -> matmul[k]`), and
the inner loop needs to read from `K[*, j]` via this pointer. Spilling
`x1` and reusing it as scratch destroys the operand pointer.

**Mitigation — emit_matmul ordering rule.** The **first** operation in
`emit_matmul`, before any `stp`/`pushq` or other stack manipulation,
is `abi.materialise_ptr` for both matmul operands (left, right) into
dedicated scratch registers: `x9`/`x10` on arm64, `%r10`/`%r11` on
x86_64. After this materialisation, the inner loop reads operand
pointers exclusively from these scratch registers, never from ABI
argument registers. The original ABI argument registers (`x0`/`x1` on
arm64, `%rdi`/`%rsi` on x86_64 for the N=2 case) are now free for use
as inner-loop scratch (counters, address arithmetic).

**Consequence: the old M10 `stp x1, x2, [sp, #-16]!` (arm64) and
`movq → %xmm6/%xmm7/%xmm8` (x86_64) blocks around the outer loop are
REMOVED, not adapted.** They existed in M10 to protect `params` and
`output` pointers from being clobbered by inner-loop scratch use.
Under the new rule, inner-loop scratch comes from the freed input ABI
registers (`x0..x_{N-1}` on arm64; `%rdi..%r_{N-1}` on x86_64);
`params` (in `x_N` / `%r_N`) and `output` (in `x_{N+1}` / `%r_{N+1}`)
are never touched by the inner loop. The structure of `emit_matmul`
therefore SIMPLIFIES — fewer instructions emitted, no stack manipulation
across the outer loop.

The acceptance fixture `multi_input_attention.nfl` exercises this path:
`q -> matmul[k]` reads `k` from `x1` / `%rsi` originally, but under the
rework, the `k`-pointer is materialised into `x10` / `%r11` before the
inner loop and read from there. The N=2 sanity fixture
(`two_input_matmul.nfl`) catches the same hazard in the simpler case
where there is no softmax to mask the failure mode.

**Verification.** The N=3 acceptance test would fail numerically if
matmul corrupts the `k` or `v` pointer; the N=2 sanity fixture catches
`b`-pointer corruption pre-softmax. Unit tests on the emitted asm
shape additionally check that `emit_matmul` for N≥1 contains no `stp`
(arm64) or `pushq` (x86_64) instructions in its body — the only
stack-manipulation instructions a multi-input matmul function should
emit are those produced by `AbiContext::emit_ffi_save` /
`emit_ffi_restore`, and matmul itself does not call FFI.

### 9.2 BufferLoc::Input(usize) match-arm misses

**Risk.** Adding a `usize` payload to `BufferLoc::Input` requires
updating every `match` over `BufferLoc` in the codebase. Some sites
might do `BufferLoc::Input => ...` without a binding — the Rust
compiler will flag these, but a careless implementation could insert
`BufferLoc::Input(_) => ...` ignoring the index when it should be using
it.

**Mitigation.** The plan must explicitly enumerate all `match BufferLoc`
sites in arm64 and x86_64 codebases (grep `match.*BufferLoc` →
expected count documented). Any `Input(_)` in op-emitters is a code
smell to be reviewed: in M12, `materialise_ptr(BufferLoc::Input(idx))`
is the canonical path, and ops should use it rather than match on
`BufferLoc` themselves.

### 9.3 Bench harness per-arity dispatch — bounded but explicit

**Risk.** `match sig.inputs_floats.len() { 1 => ..., 2 => ..., 3 => ...,
n => unimplemented!("...") }` is enumeration up to a current cap. If
a future fixture has N=4 and bench is forgotten, it silently fails on
that fixture.

**Mitigation.** The `n => unimplemented!()` arm produces a clear panic
at runtime for any unhandled arity. Bench's existing 3 fixtures are
all N=1 and continue to work; N=2 and N=3 paths are unused at M12
merge but are present in code for symmetry. The plan must include
a unit test that exercises each arity branch with a synthetic FnSig
to confirm dispatch logic.

### 9.4 N=1 silent regression

**Risk.** A bug in `AbiContext` for N=1 case (e.g., off-by-one in
`INPUT_REGS` indexing, or an alignment-padding regression) could
produce subtly wrong assembly for all M3-M11 fixtures. Existing FFI
tests would still pass numerically because the bug might happen to
work (e.g., wrong register but holding the right pointer by accident
due to caller convention), only to fail unpredictably later.

**Mitigation.** Acceptance gate #4 (§10.4) requires that all M3-M11
fixtures produce **bit-exact** assembly under M12 codegen vs M11
baseline. Implementation: golden-file diff in CI, or hash comparison
against pinned baseline hashes for each fixture's compiled assembly.

### 9.5 Documentation drift

**Risk.** New ABI tables in `profile_guide/{arm64,x86_64}.md` could
fall out of sync with `INPUT_REGS` table in code if someone extends
the cap later (e.g., N=5+ via stack-spill).

**Mitigation.** ABI table in docs is a copy of `INPUT_REGS` with one
row per N. The doc explicitly states "M12 caps N at 4; extending
beyond requires updating this table AND `INPUT_REGS` AND adding
stack-spill emission." Future-N work is gated on this checklist.

### 9.6 Bench seeding cascade for multi-input determinism

**Invariant.** Bench harness output must be bit-identical across hosts
for a given `--seed` value. M11 established this for N=1 with a single
`fill_random(buf, seed)` call. M12's per-arity dispatch must extend
this to multi-input fixtures via a **seed cascade**: each input buffer
gets `seed.wrapping_add(i as u64)` where `i` is its index in
`sig.inputs_floats`; the params buffer gets
`seed.wrapping_add(sig.inputs_floats.len() as u64)`.

```rust
let n_inputs = sig.inputs_floats.len();
let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(n_inputs);
for (i, &n_floats) in sig.inputs_floats.iter().enumerate() {
    let mut buf = vec![0f32; n_floats];
    fill_random(&mut buf, seed.wrapping_add(i as u64));
    inputs.push(buf);
}
let mut params = vec![0f32; sig.params_floats];
fill_random(&mut params, seed.wrapping_add(n_inputs as u64));
```

For the existing N=1 bench fixtures: `inputs[0]` gets `seed`, `params`
gets `seed.wrapping_add(1)` — bit-identical to M11 behaviour. For a
hypothetical N=3 fixture (none in M12 itself, but the dispatch code
must handle it correctly): q gets `seed`, k gets `seed.wrapping_add(1)`,
v gets `seed.wrapping_add(2)`, params get `seed.wrapping_add(3)`.

**Why `wrapping_add`.** Avoid panic on hypothetical overflow; real
bench seeds are small (default 42) and production never approaches
`u64::MAX`. The wrapping form is the idiomatic Rust pattern for "I do
not care about overflow here."

**Verification.** Unit test in `bench`: for a synthetic `FnSig` with
`inputs_floats = vec![10, 20, 30]`, the buffer-construction step
produces three input buffers and one params buffer whose contents
match four independent `fill_random(_, 42)`, `fill_random(_, 43)`,
`fill_random(_, 44)`, `fill_random(_, 45)` calls (no aliasing between
buffers; the cascade is sequential and deterministic).

---

## 10. Acceptance Gates

For the M12 PR to merge, all of the following must pass.

### 10.1 Workspace gates

1. `cargo fmt --all -- --check` — exit 0.
2. `cargo clippy --workspace --all-targets -- -D warnings` — exit 0.
3. `cargo build --workspace` — exit 0.
4. `cargo test --workspace` — exit 0; test count ≥ 369.

### 10.2 N=1 regression (bit-exact against post-M12 golden baseline)

For each existing M3-M11 fixture in `tests/fixtures/` (`tiny_mlp.nfl`,
`m4_linear_relu.nfl`, `mixed_args.nfl`, `softmax_with_bias.nfl`,
`dropout_only.nfl`, `classifier.nfl`, `large_classifier_k.nfl`,
`large_classifier_n.nfl`, `pipeline_styles.nfl`, `comments.nfl`,
`self_attention.nfl`):

- arm64 generated assembly == post-M12 golden baseline (bit-exact).
- x86_64 generated assembly == post-M12 golden baseline (bit-exact).

The post-M12 golden baseline is established in plan Task B.12 (and
mirror C.10 for x86_64), AFTER the `emit_matmul` rework lands. The
rework intentionally changes scratch-register assignment (`x9`/`x10`/
`x11` in place of M11's `x11`/`x13`/`x12` on arm64; `%r10`/`%r11`/
`%r12` in place of prior `%xmm6`/`%xmm7`/`%xmm8` spill on x86_64) —
register renames are expected and intentional, NOT a regression.

Acceptable diff between M11-era assembly and post-M12 goldens: only
register-name renames AND removal of the old outer-loop spill block
(`stp x1, x2, [sp, #-16]!` arm64; `movq → %xmm6/%xmm7/%xmm8` x86_64).
Any change to instruction count outside that single removed block,
any change to loop structure, any change to function prologue or
epilogue, any change to label naming — these are regressions and
block merge.

**Register-cascade-induced changes within `emit_matmul` body are
permitted** when justified by §9.1's ABI-register-avoidance
requirements. Specifically: when the new operand-pointer layout
consumes the non-ABI scratch range (`x6`-`x17` on arm64; `%r9`/`%r10`/
`%r11` on x86_64 — only three caller-saved non-ABI regs remain for
N=3 after `%rdi`/`%rsi`/`%rdx`/`%rcx`/`%r8` are occupied by ABI),
bound and stride emissions that previously hoisted outside loops may
move inline if no free scratch register remains. The justification
must appear in the rework's commit message and module doc.

This relaxation reflects an implementation-discovered constraint, not
a loosening of the regression invariant in the abstract: any
*non-forced* drift in instruction count or structure is still a
regression. Reviewers must verify that the cascade is actually
register-pressure-driven (e.g., document which non-ABI scratch slots
are occupied and why no spare register exists for the hoist).

Numerical correctness for every N=1 model is verified independently
through the existing M3-M11 integration test re-run on both profiles
(no fixture's `_match_numerically` test loosens its acceptance).
Bit-exact assembly identity is the secondary safeguard against silent
codegen drift; numerical correctness is the primary correctness gate.

Once the post-M12 golden baseline is committed (Task B.12 / C.10),
gate #4 holds the new baseline as the bit-exact reference for any
future change. Future milestones must regenerate goldens only when
they introduce intentional codegen changes; review of golden diffs
is a hard checkpoint in any plan that touches op-emitters.

Implementation: golden files at
`profiles/{arm64,x86_64}/tests/golden/<fixture>.s` checked in
(post-M12 baseline); a regression test
(`n1_regression_all_fixtures_bit_exact`) asserts `generated ==
golden` for each fixture.

### 10.3 Multi-input fixtures (numerical correctness)

5. arm64: `two_input_matmul_match_numerically` passes.
6. x86_64: `two_input_matmul_match_numerically` passes.
7. arm64: `multi_input_attention_match_numerically` passes.
8. x86_64: `multi_input_attention_match_numerically` passes.

Acceptance criterion for #7 + #8: bit-exact match against reference Rust
implementation. The Rust reference computes `(softmax((Q @ Kᵀ) * 0.25))
@ V` element-by-element with the same f32 ordering as the generated
assembly (same loop order over (batch, heads, seq, head_dim)).

### 10.4 Negative test (error path)

9. arm64: `too_many_inputs_returns_error` — `profile.lower(&uir)` returns
   `Err(LowerError::TooManyInputs { n: 5, max: 4, span: ... })`.
10. x86_64: same as #9. Identical `n` and `max` values; span pointing
    to the same source location.

### 10.5 Bench harness

11. `cargo run -p bench --release -- --profile arm64 --format markdown
    --seed 42` runs all 3 existing fixtures (all N=1) and produces
    output formatted identically to M11's pre-M12 output (modulo
    timing variance, which is informational).
12. Same for `--profile x86_64` (on x86_64 hosts).

### 10.6 AbiContext unit tests

For each profile, for N ∈ {1, 2, 3, 4}:

13. `AbiContext { n_inputs: N }.ffi_save_set().len() == N + 2`.
14. `AbiContext { n_inputs: N }.input_reg(i)` returns `INPUT_REGS[i]`
    for all `i ∈ 0..N`.
15. `emit_ffi_save(&mut s)` produces output whose total SP delta
    (counted from `stp .. !` arm64 / `pushq` x86_64 instructions)
    is divisible by 16.
16. `emit_ffi_restore(&mut s)` produces output whose pair / push
    sequence is the strict reverse of `emit_ffi_save`.
17. For odd N+2 (i.e., N ∈ {1, 3}): `emit_ffi_save` includes one
    `xzr` (arm64) or one `pushq %rax` (x86_64) padding entry.
18. For even N+2 (i.e., N ∈ {2, 4}): `emit_ffi_save` includes no
    padding.

### 10.7 Documentation

19. `docs/profile_guide/arm64.md` contains a "Multi-Input ABI"
    section with the AAPCS register table from §4.1, alignment rule
    from §6.1, and LIFO note from §6.3.
20. `docs/profile_guide/x86_64.md` contains the equivalent for
    SysV.
21. `docs/language_reference/uir.md` notes that codegen now consumes
    all of `UirModel.inputs` (clarifying behavioural change from M10
    where only `inputs.first()` was honored).
22. `docs/language_reference/grammar.md` notes the multi-input
    convention (declaration order = ABI register order; convention is
    inputs at top of model body).
23. `PROJECT_SPEC.md` M12 row added; Current Status updated; Strategic
    Roadmap marks A1 first leg closed.
24. `CLAUDE.md` Current Status updated to M12; Repository Structure
    tree gains `profiles/{arm64,x86_64}/src/abi.rs`.
25. `DEVLOG.md` has a chronological M12 closure entry.

---

## 11. Out of Scope

Items intentionally excluded from M12, with the rationale:

- **`add` operation (binary elementwise).** Required for residual
  connections in M13's transformer block. Excluded from M12 per
  brainstorm Q3 to keep M12 atomic on ABI. Becomes early M13 work.
- **LayerNorm, FFN.** Same reasoning; M13+.
- **Stack-spilled inputs (N ≥ 5).** Would require SysV stack-spill on
  x86_64 + corresponding stack frame adjustments on arm64. Deferred
  until a use case demands it (currently unanticipated within A2).
- **Liveness-based FFI spill optimisation.** `ffi_save_set()` always
  returns the full ABI argument set. Liveness analysis could spill
  fewer registers in some cases, saving ~2 instructions per softmax
  row. Cost is below measurement noise; deferred until a real
  regression appears.
- **Bench fixture for N=3 (real Q/K/V multi-input attention).** Bench
  harness gains per-arity dispatch logic but no new fixtures. Reason:
  M11 variance discipline (p95/median ≤ 1.3×) requires multi-run for
  small-µs fixtures. Adding a noisy single-sample fixture undermines
  bench's stable-anchor invariant. A multi-run methodology change is a
  separate (smaller) milestone if/when the data is needed.
- **Extending the `Profile` trait.** New methods (e.g., for ABI
  introspection from outside profile crates) violate the M9 hard rule
  "trait grows by request, not by anticipation." `AbiContext` is
  per-profile module-internal; no consumer outside profile crates
  needs ABI register names.
- **Viewer annotations for multi-input shape (A3).** Excluded per
  brainstorm Q3. M13+.

---

## 12. Open Questions

None blocking spec completion. Items below are nice-to-have research
follow-ups recorded for visibility.

- **Q12.1.** Should `AbiContext` be promoted to a per-profile trait
  (profile-internal, not in profile-api) so future profiles
  (e.g., RISC-V) reuse the abstraction? Premature now — single concrete
  use per profile, no abstraction call yet. Re-evaluate if a third
  profile lands.

- **Q12.2.** Should N=1 regression goldens be hash-only or full-text?
  Full-text golden files (~100KB across all fixtures) make diff review
  meaningful when a regression triggers; hash-only is more compact but
  uninformative on failure. Plan synthesis to decide based on existing
  golden infrastructure (none exists pre-M12; this is M12's choice to
  make).

- **Q12.3.** Should the N=2 sanity fixture survive after M13's `add`
  op lands? `two_input_matmul.nfl` is a placeholder for "any binary-
  input op test"; once `add` exists, residual-style fixtures may be
  more representative. Decision deferred to M13's brainstorm.

---

## 13. Sizing Summary

| Metric | Estimate |
|---|---|
| LoC (production + tests + docs) | ~1200 |
| New files | 5 (2 abi.rs modules, 3 .nfl fixtures) |
| Workspace crates touched | 5 (profile-api, compiler, profiles/arm64, profiles/x86_64, bench) |
| Test count delta | +25 to +35 |
| Atomic commits | ~6 |

**M12 is M7-sized**, not M9. Originally framed M9-sized in the M11
handoff because grammar+IR readiness was unverified; brainstorm
discovered `compiler/src/ir/build.rs` already supports multi-input,
collapsing the grammar+AST+IR work the original framing assumed. M12
is the codegen + ABI + FFI surface delta only.

The closest sibling milestone is **M7** (rebuild-helper extraction):
single architectural connector (`AbiContext` here, `RewritePlan` there),
ripple through all op modules, clear acceptance fixture, atomic commits
on coherent surface.
