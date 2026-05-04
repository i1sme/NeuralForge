# Milestone 5a — Kernel Fusion Pass (linear → relu) — Design

> **Status:** Brainstormed and approved 2026-05-04. To be implemented in the
> `claude/m5-kernel-fusion` worktree.
> **Source:** This spec captures the M5a brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M4 closed with all 5 M3 positive fixtures lowering end-to-end through the
arm64 profile. One regression vs M4a's minimum slice: relu in M4b is emitted
as a separate elementwise loop after matmul (so a copy-with-clamp from
intermediate stack buffer to output buffer), whereas M4a had relu fused
inline into matmul's store instruction (one fewer memory round-trip).

M5 introduces the **UIR-level optimisation pass** infrastructure that
recovers M4a's in-place relu and sets up the framework for further fusion
patterns. M5a is the minimum honest end-to-end slice: pass framework
+ one fusion pattern (`linear → relu`) + profile asm-fusion + a CLI
flag to disable fusion for verification.

**Pre-decided architectural call:** fusion lives in **UIR-level passes**,
not codegen-time peephole. Two reasons:

1. **Visibility.** Codegen-time peephole sees only local context (two
   adjacent nodes). UIR-level pass sees the entire graph and can decide
   based on shapes, types, **consumer counts**. Linear→Relu fusion is
   safe only when Linear has exactly one consumer (the Relu) — that
   constraint is invisible to a peephole optimiser walking dispatch arms.
2. **Profile isolation** (per `PROJECT_SPEC.md` design principle 3).
   With fusion at UIR-pass level, `profiles/arm64` receives a
   partially-fused graph and just emits whatever the graph says. The
   profile knows nothing about fusion logic — only about codegen.
   `compiler/passes/` is the place that decides *what* fuses; profiles
   decide *how* to emit fused ops.

This is the right separation of concerns: **UIR-passes decide what,
codegen decides how.**

## 2. Goal

Introduce `compiler::passes` infrastructure (UirPass trait, registry,
pipeline runner, PassError type). Implement `FuseLinearRelu` — a pass
that finds the pattern `Linear (without bias=true) → Relu` where the
Linear has exactly one consumer (the Relu), merges them by setting
`Linear.fused_post_ops = vec![PostOp::Relu]`, and removes the Relu node
from the graph. `profiles/arm64::emit_linear` consumes `fused_post_ops`
and emits inline `fmax s0, s0, s4` before the store instruction
(recovering M4a's in-place relu). CLI gains `--no-fuse` flag for
verification — `fused_vs_unfused_classifier_match_numerically`
integration test confirms numerical output is bit-identical between
fused and unfused asm.

## 3. Non-goals

- **`linear[bias=true] → relu` fusion. M5b** lifts the bias restriction
  (bias is internal to Linear in the UIR; fusion logic is orthogonal,
  but enabled only after M5a proves the framework works for the simpler
  case).
- **Other post-ops (Gelu, Tanh, Sigmoid, etc).** Not in NFL stdlib;
  added if/when needed. M5a's `PostOp` enum carries `Relu` only.
- **`EliminateDropout` pass.** **M5b**. Dropout-elimination is a
  *graph mutation* (node disappears entirely), distinct from fusion
  (node merges into producer). Same NodeId-remap mechanism, but
  different pattern logic. M5b inherits M5a's framework.
- **Multi-pattern passes** in one trait impl. Each pass handles one
  pattern. Composition via the pipeline.
- **Pass dependency declaration.** Passes today are independent;
  `default_pipeline()` returns them in a hardcoded order. M6+ may
  introduce dependency / ordering specifications.
- **Numerical comparison automated** beyond what the integration test
  covers. CI green is the contract; no separate fuzzer.
- **Performance benchmarking infrastructure.** Standalone milestone.
- **Bare-metal target / second profile / x86_64.** M6+ / M7+.

## 4. Pre-decided architecture

Fusion lives in **`compiler/src/passes/`**. UIR-level transformation,
profile-agnostic. `nflc compile`'s pipeline is:

```
NFL source → parse → AST → ir::build → UIR
                                         ↓
                           passes::run_pipeline(uir, default_pipeline())
                                         ↓
                                    fused UIR
                                         ↓
                            profiles::<arch>::lower(uir) → asm
```

When `--no-fuse` is set, `passes::run_pipeline` is skipped; profile
receives raw UIR.

## 5. Pass interface — `compiler::passes`

### 5.1. The `UirPass` trait

```rust
// compiler/src/passes/mod.rs
use crate::Uir;
use crate::ast::Span;

pub mod fuse_linear_relu;

#[cfg(test)]
mod tests;

/// A UIR-level optimisation pass.
///
/// Passes are functional: they take an immutable `&Uir` and return a fresh
/// `Uir` with the transformation applied. NodeIds in the new graph are
/// freshly numbered 0..N; references (operands, model.inputs, model.output)
/// are remapped during reconstruction. This guarantees no stale-NodeId
/// hazards for downstream consumers (codegen, viewer, future passes).
pub trait UirPass {
    /// Stable identifier for CLI flags ("--passes=..."), error messages,
    /// log lines. Snake_case, matching the convention used for
    /// `Display for StdOp` ("linear", "relu", etc).
    fn name(&self) -> &str;

    /// Run the pass. Returns a new `Uir` (or the input semantically-cloned
    /// if no patterns matched). Returns `Err(PassError)` only on
    /// defensively-detected malformed input.
    fn run(&self, uir: &Uir) -> Result<Uir, PassError>;
}

/// The default pipeline of passes, applied in order.
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![Box::new(fuse_linear_relu::FuseLinearRelu)]
}

/// Run a sequence of passes, threading the UIR through each. Stops on first
/// error.
pub fn run_pipeline(
    uir: &Uir,
    passes: &[Box<dyn UirPass>],
) -> Result<Uir, PassError> {
    let mut current = uir.clone();
    for pass in passes {
        current = pass.run(&current)?;
    }
    Ok(current)
}

/// Errors produced by a pass.
///
/// Invariant: every variant carries a `Span`. If a future variant cannot
/// reasonably point to a source location, the `span()` accessor migrates
/// to `Option<Span>` at that point — but that is a deliberate breaking
/// change, not an organic drift.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PassError {
    /// Defensive: pass found malformed input it can't handle. Should be
    /// unreachable if `ir::build` returned Ok. Carries the pass name +
    /// reason for diagnostics, plus a span pointing into the offending
    /// model.
    InvalidInput { pass: String, reason: String, span: Span },
}

impl PassError {
    /// All current variants carry a span; this method returns it without
    /// `Option`. See enum doc-comment for migration plan.
    pub fn span(&self) -> Span {
        match self {
            PassError::InvalidInput { span, .. } => *span,
        }
    }
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassError::InvalidInput { pass, reason, .. } => {
                write!(f, "pass '{}' failed: {}", pass, reason)
            }
        }
    }
}
```

### 5.2. Re-exports from `compiler/src/lib.rs`

```rust
pub mod passes;

pub use passes::{default_pipeline, run_pipeline, PassError, UirPass};
```

### 5.3. Key contracts

- **`name()` mandatory**, never empty, never changes once shipped (CLI flags
  depend on the value).
- **`run` is pure functional**: takes `&Uir`, returns owned `Uir`. The
  original is untouched; safe to call from tests for property-based
  comparison.
- **`run_pipeline` clones once** at the start, then threads. If a pass
  returns an error, downstream passes don't run.
- **`PassError`** `#[non_exhaustive]`: M5b/M6+ may add variants without
  breaking downstream consumers. Each variant must carry a `Span`.

## 6. Fusion representation in UIR

### 6.1. New `PostOp` enum

```rust
// compiler/src/ir/types.rs

/// Post-operations that fuse into a producer's output store.
///
/// `#[non_exhaustive]` — M5b/M6+ may add variants. Each represents a
/// per-element transformation applied after the producer computes a value
/// but before it stores. Not every StdOp fits as a post-op:
///
/// - Softmax needs row-context (max + sum across siblings).
/// - Dropout is no-op at inference (handled by EliminateDropout in M5b).
/// - Linear can't post-op another Linear (that's `linear → linear`,
///   different op chain, no fusion semantics).
///
/// Keeping `PostOp` distinct from `StdOp` makes the constraint explicit
/// at type level: profiles can't mistakenly route a softmax through the
/// post-op machinery.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOp {
    /// Clamp negative values to zero. Equivalent to fusing a
    /// terminal-or-single-consumer Relu node into its producer.
    Relu,
}
```

### 6.2. `NodeKind::Op` extension

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input { name: String },
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
        /// Fused post-operations, applied per-element after this op
        /// produces its output, before storing. Empty for un-fused or
        /// non-Linear ops.
        ///
        /// Populated only by `passes::FuseLinearRelu` (M5a) and future
        /// fusion passes. `compiler::ir::build` always sets this to
        /// `Vec::new()`.
        ///
        /// `Vec` rather than `Option` so M5b can express chains like
        /// `[BiasAdd, Relu]` should the need arise. Empty in M5a's
        /// fused output (only `[Relu]` for fused Linears; nothing for
        /// other ops).
        fused_post_ops: Vec<PostOp>,
    },
}
```

### 6.3. Display extension

`Display for Node` keeps M3c's single-line format:

```
n0: input "x"        :: Tensor[8, 4]
n1: linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]
```

Adds an optional `    fused=[<list>]` suffix when `fused_post_ops` is
non-empty:

```
n1: linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]    fused=[relu]
```

Suffix omitted entirely for empty `fused_post_ops`. `Display for PostOp`
prints lowercase variant name (`Relu` → `"relu"`), matching
`Display for StdOp`.

### 6.4. `compiler::ir::build` changes

Every existing `NodeKind::Op { ... }` construction site adds
`fused_post_ops: Vec::new()`. `ir::build` is the **only** producer of
`fused_post_ops: Vec::new()`; passes are the only thing that can set it
non-empty. This keeps the invariant testable: pre-pass UIR has all-empty
`fused_post_ops`; post-pass UIR may have non-empty.

### 6.5. Profile contract for `PostOp` exhaustiveness

Profiles match on `PostOp` to dispatch asm emission. Per the
`#[non_exhaustive]` rule, profiles **MUST** include a `_ => Err(...)`
arm that returns a structured error (not `unreachable!()`, not silent
skip). Adding `PostOp::Gelu` in M6 must compile cleanly in
`profiles/arm64`, but a user attempting to lower a Gelu-fused model
through the arm64 profile must receive a clean
`error: post-op 'gelu' not supported by arm64 profile`. See §7 for the
specific `LowerError::UnsupportedPostOp` variant added to
`profiles/arm64`.

## 7. Pass — `FuseLinearRelu`

```rust
// compiler/src/passes/fuse_linear_relu.rs
use super::{PassError, UirPass};
use crate::{NodeKind, StdOp, Uir, UirModel};

pub struct FuseLinearRelu;

impl UirPass for FuseLinearRelu {
    fn name(&self) -> &str {
        "fuse_linear_relu"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model)?);
        }
        Ok(Uir { models: new_models })
    }
}

fn fuse_one_model(model: &UirModel) -> Result<UirModel, PassError> {
    /* algorithm in §7.1 */
}
```

### 7.1. Algorithm

```
INPUT:  &UirModel (immutable)
OUTPUT: UirModel (fresh NodeIds)

Step 1. Build consumer_count: HashMap<NodeId, usize>
        For each node in model.nodes:
          For each operand_id in node.kind's operands:
            consumer_count[operand_id] += 1
        Treat model.output as having +1 consumer (terminal use).
        (model.inputs are not consumers; inputs are consumed but
        InputReg/OutputReg roles are about pointers, not graph edges.)

Step 2. Identify fusion victims: HashSet<NodeId> + HashMap<victim → producer>
        For each node `relu_node` at index `relu_id`:
          IF relu_node.kind == Op { op: Relu, operands: [linear_id], .. }
          AND model.nodes[linear_id].kind == Op { op: Linear, attrs, fused_post_ops, .. }
          AND fused_post_ops.is_empty()                    ; no double-fusion
          AND !linear_has_bias(attrs)                      ; M5a only; M5b lifts
          AND consumer_count[linear_id] == 1               ; Linear's only consumer is this Relu
          THEN
            Mark relu_id as victim.
            Record victim_to_producer[relu_id] = linear_id.

Step 3. Build new model:
        new_nodes: Vec<Node> = Vec::new()
        id_map: HashMap<old_id, new_id> = HashMap::new()

        FOR (old_id, node) IN model.nodes.iter().enumerate():
          IF old_id IS victim:
            ; Skip pushing this node. Map its old_id to the producer's
            ; new_id (any reference to this Relu's output now goes to
            ; the fused Linear's output).
            producer_old_id = victim_to_producer[old_id]
            id_map[old_id] = id_map[producer_old_id]   ; producer was visited earlier
          ELSE:
            ; Clone node, remap operands.
            new_node = node.clone()
            FOR operand IN new_node.kind.operands_mut():
              operand = id_map[operand]
            ; If THIS node is the producer for any victim Relu, append
            ; PostOp::Relu to its fused_post_ops.
            IF this old_id is a producer in victim_to_producer:
              new_node.kind.fused_post_ops_mut().push(PostOp::Relu)
            new_id = new_nodes.len()
            new_nodes.push(new_node)
            id_map[old_id] = new_id

Step 4. Remap model.inputs and model.output:
        new_inputs = model.inputs.iter().map(|&id| id_map[id]).collect()
        new_output = id_map[model.output]

Step 5. Return UirModel { name, nodes: new_nodes, inputs: new_inputs, output: new_output, source_span }
```

The `linear_has_bias` helper from `profiles/arm64::codegen` becomes a
library fn in `compiler::ir::stdlib` (since both passes and profiles need
it). Move during M5a; not new logic, just relocation.

### 7.2. Edge cases

| Pattern | Behaviour | Reason |
|---|---|---|
| `linear → relu` (relu is terminal) | **Fuse.** New terminal = fused Linear. | One consumer of Linear; Relu is the sole consumer. |
| `linear → relu → linear → relu` (chain) | **Both fuse independently.** | Each Linear has exactly one consumer (its Relu). The intermediate Relu's consumer (next Linear) is irrelevant — what matters is that *Linear* has only one consumer. |
| `linear → relu` where relu has multiple consumers | **Fuse.** | Symmetric: every consumer of Relu's output expects the relu'd value. After fusion, the fused Linear's output IS the relu'd value. Same semantics. |
| `linear` has multiple consumers (one of which is relu) | **Don't fuse.** | Other consumers expect *pre-relu* (raw matmul) output. Fusing would change their input. |
| `linear[bias=true] → relu` | **Don't fuse in M5a.** Returns unchanged in pass output. M5b lifts the restriction. |
| `linear → dropout → relu` | **Don't fuse.** Dropout intervenes between Linear and Relu — they're not adjacent. M5b's `EliminateDropout` (which removes dropout from graph) followed by `FuseLinearRelu` would handle this transitively in a multi-pass pipeline. |
| `softmax → relu` (hypothetical, currently invalid grammar) | **Don't fuse.** Producer must be Linear. `PostOp::Relu` only attaches to Linear in M5a. |
| Empty UIR / no fusable patterns | **Identity transform.** Returns clone of input. |
| UIR has zero models | **Identity.** `Uir { models: vec![] }`. |

### 7.3. Pass-level unit tests

In `compiler/src/passes/fuse_linear_relu.rs` inline `#[cfg(test)] mod tests`:

| Test | Validates |
|---|---|
| `fuses_simple_linear_relu` | Terminal `linear → relu` → fused Linear with `fused_post_ops: [Relu]`, Relu node gone, NodeIds 0..N-1 dense. |
| `does_not_fuse_when_linear_has_multiple_consumers` | `linear → [relu, softmax]` → no fusion (Linear has 2 consumers). |
| `fuses_chain_independently` | `linear[8] → relu → linear[2] → relu` → both Linears fused. |
| `does_not_fuse_when_relu_not_after_linear` | `softmax → relu` (synthetic UIR; grammar may not allow) → not fusable. |
| `does_not_fuse_when_linear_already_fused` | Linear already with `fused_post_ops: [Relu]` followed by another Relu → no double-fusion. |
| `does_not_fuse_when_linear_has_bias` | `linear[bias=true] → relu` → no fusion (M5a scope). |
| `empty_uir_passes_unchanged` | `Uir { models: [] }` → identity. |
| `model_inputs_and_output_remapped` | After fusion in a model whose terminal was Relu, the new model.output points to the fused Linear's new NodeId; model.inputs are remapped if needed. |
| `pass_name_is_stable` | `FuseLinearRelu.name() == "fuse_linear_relu"` (locks the CLI contract). |

### 7.4. Pipeline-level unit tests

In `compiler/src/passes/tests.rs`:

| Test | Validates |
|---|---|
| `default_pipeline_includes_fuse_linear_relu` | `default_pipeline()` returns a Vec containing exactly one pass whose `name()` is `"fuse_linear_relu"`. |
| `run_pipeline_threads_uir_through_passes` | With a synthetic Vec of 2 mock passes (`IdentityPass`, `IdentityPass`), output equals the input model count etc. — pipeline doesn't drop nodes spontaneously. |
| `empty_pipeline_returns_input_clone` | `run_pipeline(&uir, &[])` → returns `uir.clone()`. |

## 8. Profile changes — `profiles/arm64::emit_linear`

### 8.1. Signature change

```rust
// profiles/arm64/src/ops/linear.rs
use crate::buffer::BufferLoc;
use crate::types::LowerError;                  // M5a: now needed for UnsupportedPostOp
use compiler::{ast::Span, PostOp};

#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    model_idx: usize,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
    node_span: Span,                           // NEW — for error span
    fused_post_ops: &[PostOp],                 // NEW — fusion info
) -> Result<String, LowerError> {              // NEW — was String
    /* implementation in §8.2 */
}
```

### 8.2. Asm changes

**(a) Materialise `s_zero` once** at the top of the function body, but
ONLY if any post-op needs it (today: `PostOp::Relu` does):

```asm
    ; (existing prologue, materialise_ptr x11/x12/x13/x14)
    fmov    s4, wzr                   ; s_zero, only if fused_post_ops contains a zero-needing PostOp
    mov     x3, #0
.Lmm_i_<m>_<l>:
    ; ... matmul nested loops ...
```

The `s4` register choice is consistent with M4a's relu emitter (which used
`s4` for its pre-loop zero materialisation). matmul body uses
`s0`/`s1`/`s2`/`s5` — `s4` is a free callee-saved-equivalent slot for
this purpose (caller-saved per AAPCS64, but matmul never needs to
preserve it across calls — there are no calls in matmul body).

**(b) Apply post-ops** between the k-loop's accumulator finalisation and
the store, **after bias-add** if bias is present:

```asm
.Lmm_k_end_<m>_<l>:
    ; (s0 = matmul accumulator)
    ; bias-add if present (already in M4b):
    ldr     s5, [x14, x4, lsl #2]    ; bias[j]
    fadd    s0, s0, s5
    ; M5a: NEW — apply each post-op in order
    fmax    s0, s0, s4               ; PostOp::Relu (s4 = 0.0 from prologue)
    ; (existing store):
    mov     x8, #<n>
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
```

### 8.3. The `_ =>` arm

```rust
let needs_zero = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
if needs_zero {
    s.push_str("    fmov    s4, wzr\n");
}

// ... matmul body up to k-loop end ...

// After bias-add (if any), before store:
for post_op in fused_post_ops {
    match post_op {
        PostOp::Relu => s.push_str("    fmax    s0, s0, s4\n"),
        // _ arm: required by #[non_exhaustive] PostOp.
        _ => {
            return Err(LowerError::UnsupportedPostOp {
                op: format!("{post_op:?}").to_lowercase(),
                span: node_span,
            });
        }
    }
}
```

`format!("{post_op:?}")` uses `Debug` derive; lowercased output gives
`"relu"`, `"gelu"`, etc. — matching `Display for StdOp` convention.

### 8.4. New `LowerError` variant

```rust
// profiles/arm64/src/types.rs

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// (existing) Defensive guard for unsupported StdOps.
    #[allow(dead_code)]
    UnsupportedOp { op: String, span: compiler::ast::Span },
    /// (existing) Defensive guard for non-concrete shapes.
    ShapeNotConcrete { span: compiler::ast::Span },
    /// M5a: post-op variant not supported by this profile. Fires when a
    /// future PostOp variant lands in PostOp before this profile knows
    /// how to emit it.
    UnsupportedPostOp { op: String, span: compiler::ast::Span },
}
```

`Display`/`span()` arms updated.

### 8.5. `walk_model` change

```rust
// profiles/arm64/src/codegen.rs

StdOp::Linear => {
    // ... existing setup ...
    let NodeKind::Op { fused_post_ops, .. } = &node.kind else { unreachable!() };
    body.push_str(&crate::ops::emit_linear(
        b, k, n, model_idx, linear_idx,
        src_loc, dst_loc,
        weight_offset, bias_offset,
        node.source_span,                    // NEW
        fused_post_ops,                      // NEW
    )?);                                     // NEW — `?` because Result
    linear_idx += 1;
}
```

### 8.6. Profile-level unit tests

In `profiles/arm64/src/tests.rs`, 3 new tests:

| Test | Validates |
|---|---|
| `fused_linear_relu_emits_fmax_before_store` | Synthetic UIR with `Linear { fused_post_ops: [Relu] }` → asm has `fmov s4, wzr` (once) and `fmax s0, s0, s4` between matmul and store. |
| `fused_linear_relu_no_separate_relu_loop` | Same → asm does NOT contain `.Lrelu_<m>_<r>:` label (relu is inline, not a separate elementwise loop). |
| `unfused_linear_still_no_fmax` | UIR with `Linear { fused_post_ops: [] }` → asm does NOT contain `fmax` (back-compat for un-fused models). |

## 9. CLI — `--no-fuse` flag

### 9.1. Subcommand syntax

```
nflc compile <file.nfl> --profile <name> [-o <output.s>] [--no-fuse]
```

### 9.2. Behaviour

- **Default:** `compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())` runs between `ir::build` and `profile.lower`. The fused UIR is what reaches the profile.
- **`--no-fuse`:** `passes::run_pipeline` is skipped; profile receives raw UIR (post-`ir::build`, pre-passes).

### 9.3. Output contract — stdout/stderr discipline

**Strict invariant: stdout = asm source ONLY (or empty if `-o <path>`); stderr = ALL diagnostic output (notes, warnings, errors).**

Rationale: `nflc compile <file> --profile arm64 | as -o <obj>` is a real
shell pipeline. Mixing diagnostics into stdout would break it.

Specifically:
- `note: applied passes: fuse_linear_relu` → **stderr** (default mode).
- `note: passes skipped (--no-fuse)` → **stderr** (when `--no-fuse`).
- `error: ...`, `note: previously defined at ...` (existing render) → **stderr**.
- The `.s` source: **stdout** (or written to `-o <path>`).

Verification: `cargo run --bin nflc -- compile <file> --profile arm64 2>/dev/null | head -5`
must show only asm directives, no `note:` lines.

### 9.4. Arg-parsing refactor

Current M4b `nflc compile` arg-parsing uses pattern-matching on
`args.as_slice()` slice positions. With `--no-fuse` added (optional, no
value), the combinatorial explosion gets unmanageable (8+ arms).

Introduce a minimal stateful parser in `nflc/src/main.rs`:

```rust
struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_fuse: bool,
}

fn parse_compile_args(args: &[String]) -> Result<CompileArgs, String> {
    // First positional: path. Then sweep flags:
    //   --profile <name>  (required, positional value)
    //   -o <path>         (optional, positional value)
    //   --no-fuse         (optional, no value)
    // Returns CompileArgs or human-readable error string for unknown flag,
    // missing required, missing value after flag, or duplicate flag.
}
```

~30 lines, but eliminates pattern explosion and enables future flags
(`--passes=...` from M5b, `--emit=tokens|ast|uir|asm` if ever added)
without rewriting the dispatcher.

### 9.5. CLI tests

In `nflc/src/main.rs`'s test module (or a new
`nflc/tests/cli_compile.rs`):

| Test | Validates |
|---|---|
| `compile_default_runs_fusion` | Smoke-style assertion using `std::process::Command::new(env!("CARGO_BIN_EXE_nflc"))`: `nflc compile m4_linear_relu.nfl --profile arm64` → stderr contains `applied passes: fuse_linear_relu`; stdout contains `fmax    s0, s0, s4` and does NOT contain `.Lrelu_`. |
| `compile_with_no_fuse_skips_fusion` | Same fixture with `--no-fuse` → stderr contains `passes skipped (--no-fuse)`; stdout has `.Lrelu_0_0:` (M4b-style separate loop). |
| `compile_unknown_flag_rejected` | `nflc compile ... --frobnicate` → exit 1, stderr has clear "unknown flag" message. |

## 10. Testing strategy summary

Layered testing per § 7, § 8, § 9:

- **Pass-level unit tests** (inline `#[cfg(test)] mod tests` inside `compiler/src/passes/fuse_linear_relu.rs`, 9 tests): pattern detection, NodeId remap, edge cases.
- **Pipeline-level unit tests** (compiler/src/passes/tests.rs, 3 tests): pipeline mechanics with mock passes.
- **Profile asm-level unit tests** (profiles/arm64/src/tests.rs, 3 tests): asm shape with/without fusion.
- **CLI-level smoke tests** (nflc, 3 tests): default + `--no-fuse` + unknown-flag rejection.
- **Integration test (FFI numerical correctness)** in
  `profiles/arm64/tests/integration.rs`:
  `fused_vs_unfused_classifier_match_numerically` — the strongest
  guarantee. Uses **`classifier.nfl`** (which exercises 2 independent
  fusions: `linear[512]→relu` and `linear[256]→relu` plus non-fused
  softmax). Runs the model through both fused and unfused paths,
  compiles each with `cc -shared -arch arm64`, dlopens both, calls with
  the same deterministic input/params, asserts `assert_eq!` (bit-exact)
  on outputs. Bit-exactness is valid because intermediate `str + ldr`
  on f32 is bit-preserving — fusion just relocates *where* relu is
  applied, doesn't change *which* floats get computed.

**Test count target:** baseline 148 → ~167 (148 + 9 pass-level + 3
pipeline + 3 profile asm + 3 CLI + 1 integration = 19 new tests).

**Negative tests:** none new. `PassError::InvalidInput` is defensive,
`#[allow(dead_code)]`, mirroring M4b convention for `LowerError::UnsupportedOp`.

**Snapshot tests:** explicitly NOT introduced. Substring assertions are
sufficient at M5a's scope. Snapshot framework (`insta` or similar) is
M5c-or-later territory.

## 11. Vertical slicing

| Slice | Content |
|---|---|
| **M5a (this spec)** | Pass framework + `FuseLinearRelu` (linear-no-bias only, single-consumer-only) + profile asm fusion + `--no-fuse` CLI + integration tests. |
| **M5b** | Lift bias restriction in `FuseLinearRelu` (so `linear[bias=true]→relu` fuses). Add `EliminateDropout` pass (removes dropout from graph using same NodeId-remap mechanism). Add `--passes=X,Y` CLI filter syntax. |
| **M5c** | `docs/profile_guide/arm64.md` updates (fusion section, asm patterns, CLI flags). `docs/language_reference/uir.md` extension on PostOp + fused_post_ops. `PROJECT_SPEC.md` milestone close-out. DEVLOG. Optional: snapshot tests via `insta` if asm-shape stabilised; benchmark sketches. |

**Size estimate:** M5a ~10-12 tasks (mirroring M4a). M5b ~6-8. M5c ~3-5.

## 12. Acceptance criteria

1. **Build/lint/format clean.** `cargo build --workspace`,
   `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo fmt --all -- --check` all exit 0.
2. **Pre-M5a tests preserved.** Plan records actual baseline
   (currently 148); no regression.
3. **Pass-level unit tests pass** (9 in `fuse_linear_relu`, 3 in
   `passes/tests.rs`).
4. **Profile asm-level unit tests pass** (3 new in
   `profiles/arm64/src/tests.rs`).
5. **Integration test passes** on aarch64 (skips cleanly elsewhere):
   `fused_vs_unfused_classifier_match_numerically` confirms
   (a) asm shapes differ as expected (fused: inline `fmax s0, s0, s4`
   without `.Lrelu_*` labels; unfused: separate `.Lrelu_<m>_<r>:`
   loops), (b) numerical output is **bit-identical** (`assert_eq!`,
   not epsilon).
6. **CLI smoke positive (default mode):**
   `nflc compile tests/fixtures/classifier.nfl --profile arm64` → exit 0;
   stderr contains `note: applied passes: fuse_linear_relu`; stdout =
   pure asm (pipeable to `cc -shared -arch arm64 -x assembler -o
   /tmp/classifier.dylib -`); `nm` reports `_nfl_forward_Classifier`.
7. **CLI smoke positive (`--no-fuse`):** same fixture with `--no-fuse`
   → stderr contains `note: passes skipped (--no-fuse)`; stdout asm has
   `.Lrelu_*` loops (M4b shape).
8. **Stdout/stderr discipline verified:**
   `cargo run … --profile arm64 2>/dev/null | head -5` shows only asm
   directives, no `note:` lines.
9. **All 5 M3 fixtures + M4a fixture continue to compile** under both
   default and `--no-fuse` modes. Existing M4b integration tests
   (`tinymlp_full_with_softmax_runs_correctly`, `classifier_runs_correctly`,
   `pipeline_styles_runs_correctly`, `comments_runs_correctly`,
   `mixed_args_runs_correctly`, `m4a_no_softmax_still_runs`) switch to
   default-fused path and continue numerically passing.
   **Implication:** for softmax-terminal models (tiny_mlp, classifier,
   pipeline_styles, comments, mixed_args), no Linear→Relu fusion
   candidate exists at the terminal — softmax is the terminal — so
   those models' `fused_post_ops` lists stay empty. Their asm outputs
   in default-fused mode are bit-identical to M4b. Only the M4a fixture
   (`m4_linear_relu.nfl`, terminal Relu) actually differs in default-fused
   mode (recovers M4a's inline relu).
10. **Module-level doc-comment in `compiler::passes`** explains: what
    passes are, how to add a new one, why functional, what `default_pipeline()`
    contains.

## 13. Artifacts

### 13.1. Created

| Path | Purpose |
|---|---|
| `compiler/src/passes/mod.rs` | `UirPass` trait, `default_pipeline`, `run_pipeline`, `PassError`, module-level doc-comment (per AC #10). |
| `compiler/src/passes/fuse_linear_relu.rs` | `FuseLinearRelu` impl + algorithm + inline `#[cfg(test)] mod tests` (9 tests per §7.3). |
| `compiler/src/passes/tests.rs` | Pipeline-level unit tests (3 tests per §7.4). |

### 13.2. Modified

| Path | Change |
|---|---|
| `compiler/src/lib.rs` | `pub mod passes;` + `pub use passes::{default_pipeline, run_pipeline, PassError, UirPass};`. Plus `pub use ir::types::PostOp;` (for profiles' use). |
| `compiler/src/ir/types.rs` | (a) New `pub enum PostOp { Relu }` with `#[non_exhaustive]`, derives, doc-comment; (b) `NodeKind::Op` gains `fused_post_ops: Vec<PostOp>` field; (c) `Display for Node` renders optional `fused=[...]` suffix; (d) `Display for PostOp` impl. |
| `compiler/src/ir/build.rs` | All `NodeKind::Op { ... }` constructions add `fused_post_ops: Vec::new()`. |
| `compiler/src/ir/stdlib.rs` | New `pub fn linear_has_bias(attrs: &[OpAttr]) -> bool` (moved from `profiles/arm64::codegen`). |
| `compiler/src/ir/tests.rs` | Existing UIR unit tests adapt to new field — pattern matches use `..` to ignore `fused_post_ops` where irrelevant. |
| `profiles/arm64/src/types.rs` | New `LowerError::UnsupportedPostOp { op: String, span: compiler::ast::Span }` variant; Display + span() arms updated. |
| `profiles/arm64/src/codegen.rs` | `linear_has_bias` removed (now in `compiler::ir::stdlib`); imports updated. `walk_model::StdOp::Linear` arm reads `node.source_span` and `fused_post_ops`, passes them to `emit_linear`, uses `?` propagation. |
| `profiles/arm64/src/ops/linear.rs` | `emit_linear` signature extended with `node_span: Span` and `fused_post_ops: &[PostOp]`. Returns `Result<String, LowerError>`. Materialises `s4` if any `PostOp::Relu` in fused_post_ops; emits `fmax s0, s0, s4` inline before store. `_ =>` arm returns `LowerError::UnsupportedPostOp`. |
| `profiles/arm64/src/tests.rs` | 3 new tests per §8.6. |
| `profiles/arm64/tests/integration.rs` | New `fused_vs_unfused_classifier_match_numerically` test per §10. |
| `nflc/src/main.rs` | `parse_compile_args` minimal stateful parser (§9.4). `--no-fuse` flag handling. `compiler::passes::run_pipeline` invocation between `ir::build` and `profile.lower` (skipped if `no_fuse`). `note:` lines to stderr. |
| `nflc/tests/cli_compile.rs` (new file or appended to existing tests) | 3 CLI smoke tests per §9.5. |

### 13.3. Deleted

Nothing.

## 14. Open questions / risks

- **`linear_has_bias` move from `profiles/arm64::codegen` to `compiler::ir::stdlib`.**
  Pure relocation — same logic, same signature. No risk if both
  call sites (the pass and the profile) are updated atomically. Handled
  by the plan's task ordering.

- **`Uir: Clone` requirement.** `run_pipeline` clones the input UIR
  once at start, then passes mutate-and-replace via threading. Existing
  derive on `Uir`/`UirModel`/`Node` already includes `Clone` (verified
  in M3 source); if any future field is added without `Clone`,
  pass infrastructure breaks. Module-level doc-comment in
  `compiler::passes` flags this contract.

- **Unfused `emit_relu` becomes nearly-dead code.** After M5a, all
  `linear → relu` patterns fuse, so unfused `Relu` nodes only appear
  in pathological grammars (e.g., `softmax → relu`, hypothetical). The
  separate `emit_relu` emitter stays as defensive coverage but rarely
  fires in real fixtures. M5b's `EliminateDropout` may not change this;
  M6+ might. No action needed for M5a; document in profile guide
  during M5c.

- **`assert_eq!` exact-equality in fused-vs-unfused integration test.**
  Relies on f32 store+load bit-preservation (which is guaranteed by
  IEEE 754 + AArch64 f32 load/store semantics). If a future
  hypothetical platform does denormal flushing or other non-standard
  f32 behaviour at memory boundaries, the assertion would need to relax
  to `< 1e-6` epsilon. Today the macOS aarch64 platform of the CI
  runner does NOT do this; assertion holds.

- **`stderr` discipline depends on `eprintln!` consistently.** Existing
  `render_error_with_snippet` already uses `eprintln!`. `note:` lines
  for passes also use `eprintln!`. CLI tests verify
  `2>/dev/null | head` returns clean asm.

## 15. Sub-skill chain after this spec is approved

1. Spec self-review (placeholder/contradiction/scope/ambiguity scan).
2. User reviews this spec file.
3. On approval → invoke `superpowers:writing-plans` to produce
   `docs/superpowers/plans/2026-05-04-m5a-kernel-fusion.md`.
4. Subagent-driven execution mode (proven through M4a + M4b); per-task
   spec + code-quality review; INLINE for trivial tasks.
5. PR against `main` when M5a is shippable. CI gates on first push.
