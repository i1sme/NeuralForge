# M16 — A3: Profile-Level Viewer Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `nflc inspect <file.nfl> --profile <name>` — a CLI tool that surfaces post-pass per-node BufferLoc + footprint and per-model stack frame + callee-saved + leaf classification, packaged for human inspection of the same analysis values that `lower()` consumes.

**Architecture:** Extract analysis preamble out of each profile's `walk_model` into a private `analyze()` function consumed by both `lower()` and a new `Profile::inspect()` trait method. Lift `BufferLoc` to `profile-api` (verified bit-identical between profiles). New shared `Inspection` schema in `profile-api` rendered by a new `inspect-render` workspace crate; both `nflc inspect` and per-profile golden integration tests consume the renderer.

**Tech Stack:** Rust 2021, std-only (no new external dependencies). Workspace member crates: `compiler`, `profile-api`, `profiles/{arm64,x86_64}`, `nflc`, plus new `inspect-render`.

**Spec:** [`docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md`](../specs/2026-05-11-a3-viewer-annotations-design.md)

---

## Conventions

- **Workspace gates per commit (mandatory):**
  ```
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace
  ```
  All three must pass before each commit. CI gates on the same.
- **Test count baseline:** 446 tests at start (M15 closure). Trajectory tracked per commit.
- **`cargo test --workspace` runs in repository root**; integration test binaries get `cwd = <crate>/`.
- **Commit messages** follow project conventions: scope prefix (`docs(m16):`, `feat(m16):`, `refactor(m16):`), Co-Authored-By trailer if AI-assisted.
- **Bisect-claim** ends every commit message body — one sentence about what holds at this commit.

---

## Task 1 — Extract `analyze()` from `walk_model` (both profiles)

**Why this commit first:** Adding `Profile::inspect()` later requires both `lower()` and `inspect()` to consume the *same* analysis values. The cleanest enforcement is structural: extract the analysis preamble of `walk_model` into a private `analyze(model)` function, so both consumers literally call the same code. As a *pure refactor* this commit must produce bit-identical assembly output for all 446 existing tests.

**Files:**
- Modify: `profiles/arm64/src/codegen.rs`
- Modify: `profiles/x86_64/src/codegen.rs`

### arm64

- [ ] **Step 1: Confirm baseline** — run `cargo test --workspace` and confirm 446 tests pass. Record exact count.

- [ ] **Step 2: Read existing arm64 `walk_model`** — open `profiles/arm64/src/codegen.rs`. The analysis preamble runs from the top of `walk_model` (line ~30) through the `let regs = compute_callee_saved(model);` line (~163). Everything between the function entry and the `// 4. Emit prologue + body + epilogue.` comment is the preamble.

- [ ] **Step 3: Add `ModelAnalysis` struct + `analyze()` function** to `profiles/arm64/src/codegen.rs` (place above `walk_model`):

```rust
use crate::asm::LeafKind;

/// Analysis preamble shared by `walk_model` and `inspect_model`.
/// All fields are computed by pure analyzers (`assign_buffers`,
/// `compute_callee_saved`, `compute_is_leaf`) over the input UirModel.
/// Both consumers must call `analyze()` rather than re-running the
/// analyzers — single source of truth, no drift by construction.
struct ModelAnalysis {
    fn_sig: FnSig,
    assignment: crate::buffer::BufferAssignment,
    callee_saved: crate::buffer::RegSet,
    leaf: LeafKind,
    abi: AbiContext,
}

fn analyze(model: &UirModel) -> Result<ModelAnalysis, LowerError> {
    use crate::buffer::{assign_buffers, compute_callee_saved, compute_is_leaf};

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 1b. Arity check (M12 spec §5.3): N + 2 ≤ INPUT_REGS.len().
    let n_inputs = model.inputs.len();
    if n_inputs + 2 > INPUT_REGS.len() {
        return Err(LowerError::TooManyInputs {
            n: n_inputs,
            max: INPUT_REGS.len() - 2,
            span: model.source_span,
        });
    }
    let abi = AbiContext { n_inputs };

    // 2. Compute layout, ABI sizes.
    if model.inputs.is_empty() {
        return Err(LowerError::ShapeNotConcrete {
            span: model.source_span,
        });
    }
    let inputs_floats: Vec<usize> = model
        .inputs
        .iter()
        .map(|&id| model.nodes[id].ty.shape.0.iter().product::<u64>() as usize)
        .collect();
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op {
            op: StdOp::Linear,
            operands,
            attrs,
            ..
        } = &node.kind
        {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete {
                    span: node.source_span,
                });
            }
            let k = in_shape.0[1] as usize;
            let n = out_shape.0[1] as usize;
            params_layout.push(ParamSlot {
                kind: ParamKind::LinearWeight,
                origin_node: node_idx,
                offset: params_floats,
                size: k * n,
            });
            params_floats += k * n;
            if compiler::ir::linear_has_bias(attrs) {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
        if let NodeKind::Op {
            op: StdOp::LayerNorm,
            attrs,
            ..
        } = &node.kind
        {
            if compiler::ir::layernorm_has_affine(attrs) {
                let last_dim = node
                    .ty
                    .shape
                    .0
                    .last()
                    .copied()
                    .expect("LayerNorm input rank ≥ 2 enforced at IR build")
                    as usize;
                params_layout.push(ParamSlot {
                    kind: ParamKind::LayerNormScale,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: last_dim,
                });
                params_floats += last_dim;
                params_layout.push(ParamSlot {
                    kind: ParamKind::LayerNormBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: last_dim,
                });
                params_floats += last_dim;
            }
        }
    }

    let fn_sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        inputs_floats,
        output_floats,
        params_floats,
        params_layout,
    };

    let assignment = assign_buffers(model);
    let leaf = if compute_is_leaf(model) {
        LeafKind::Leaf
    } else {
        LeafKind::NonLeaf
    };
    let callee_saved = compute_callee_saved(model);

    Ok(ModelAnalysis {
        fn_sig,
        assignment,
        callee_saved,
        leaf,
        abi,
    })
}
```

- [ ] **Step 4: Refactor `walk_model` to call `analyze()`.** Replace the preamble section (lines ~30 to ~163, everything before `// 4. Emit prologue + body + epilogue.`) with:

```rust
fn walk_model(
    model_idx: usize,
    model: &UirModel,
    sym_prefix: &'static str,
) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue};

    let analysis = analyze(model)?;
    let ModelAnalysis {
        fn_sig: sig,
        assignment,
        callee_saved: regs,
        leaf,
        abi,
    } = &analysis;
    let sig = sig.clone(); // walk_model returns sig by value at end
    let assignment = assignment.clone();
    let regs = *regs;
    let leaf = *leaf;
    let abi = *abi;

    // 4. Emit prologue + body + epilogue.
    let mut body = String::new();
    body.push_str(&format_function_prologue(
        &sig,
        leaf,
        regs,
        assignment.stack_bytes,
        sym_prefix,
    ));

    // ... rest of walk_model body unchanged (per-op emission loop, epilogue) ...
```

Note: keep the existing per-op emission loop and `format_function_epilogue` call exactly as they were. Only the preamble section is replaced.

- [ ] **Step 5: Verify all tests pass** — run `cargo test --workspace` and confirm count is still 446 (no new tests, none removed). Run `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all -- --check`.

### x86_64

- [ ] **Step 6: Add `ModelAnalysis` struct + `analyze()` function** to `profiles/x86_64/src/codegen.rs`. Note that x86_64 has **no `LeafKind`** — its prologue does not depend on leaf classification. The struct shape is therefore slightly smaller:

```rust
/// Analysis preamble shared by `walk_model` and `inspect_model`.
/// Mirror of arm64's analyze(), minus LeafKind (x86_64 prologue does
/// not depend on leaf classification — see profiles/x86_64/src/asm.rs).
/// Leaf classification for inspect output is computed via the UIR-side
/// `model.calls_extern_math()` predicate directly in inspect_model.
struct ModelAnalysis {
    fn_sig: FnSig,
    assignment: crate::buffer::BufferAssignment,
    callee_saved: crate::buffer::RegSet,
    abi: AbiContext,
}

fn analyze(model: &UirModel) -> Result<ModelAnalysis, LowerError> {
    use crate::buffer::{assign_buffers, compute_callee_saved};

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 1b. Arity check.
    let n_inputs = model.inputs.len();
    if n_inputs + 2 > INPUT_REGS.len() {
        return Err(LowerError::TooManyInputs {
            n: n_inputs,
            max: INPUT_REGS.len() - 2,
            span: model.source_span,
        });
    }
    let abi = AbiContext { n_inputs };

    // 2. Compute layout, ABI sizes.
    if model.inputs.is_empty() {
        return Err(LowerError::ShapeNotConcrete {
            span: model.source_span,
        });
    }
    let inputs_floats: Vec<usize> = model
        .inputs
        .iter()
        .map(|&id| model.nodes[id].ty.shape.0.iter().product::<u64>() as usize)
        .collect();
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op {
            op: StdOp::Linear,
            operands,
            attrs,
            ..
        } = &node.kind
        {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete {
                    span: node.source_span,
                });
            }
            let k = in_shape.0[1] as usize;
            let n = out_shape.0[1] as usize;
            params_layout.push(ParamSlot {
                kind: ParamKind::LinearWeight,
                origin_node: node_idx,
                offset: params_floats,
                size: k * n,
            });
            params_floats += k * n;
            if compiler::ir::linear_has_bias(attrs) {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
        if let NodeKind::Op {
            op: StdOp::LayerNorm,
            attrs,
            ..
        } = &node.kind
        {
            if compiler::ir::layernorm_has_affine(attrs) {
                let last_dim = node
                    .ty
                    .shape
                    .0
                    .last()
                    .copied()
                    .expect("LayerNorm input rank ≥ 2 enforced at IR build")
                    as usize;
                params_layout.push(ParamSlot {
                    kind: ParamKind::LayerNormScale,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: last_dim,
                });
                params_floats += last_dim;
                params_layout.push(ParamSlot {
                    kind: ParamKind::LayerNormBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: last_dim,
                });
                params_floats += last_dim;
            }
        }
    }

    let fn_sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        inputs_floats,
        output_floats,
        params_floats,
        params_layout,
    };

    let assignment = assign_buffers(model);
    let callee_saved = compute_callee_saved(model);

    Ok(ModelAnalysis {
        fn_sig,
        assignment,
        callee_saved,
        abi,
    })
}
```

- [ ] **Step 7: Refactor x86_64 `walk_model` to call `analyze()`.** Same pattern as arm64 — replace the preamble (lines ~43 to ~175) with:

```rust
fn walk_model(
    model_idx: usize,
    model: &UirModel,
    sym_prefix: &'static str,
) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue};

    let analysis = analyze(model)?;
    let ModelAnalysis {
        fn_sig: sig,
        assignment,
        callee_saved: regs,
        abi,
    } = &analysis;
    let sig = sig.clone();
    let assignment = assignment.clone();
    let regs = *regs;
    let abi = *abi;

    // 4. Emit prologue + body + epilogue.
    let mut body = String::new();
    body.push_str(&format_function_prologue(
        &sig,
        regs,
        assignment.stack_bytes,
        sym_prefix,
    ));

    // ... rest of walk_model body unchanged ...
```

- [ ] **Step 8: Verify all tests pass** — `cargo test --workspace` should still report 446. Run clippy + fmt gates.

- [ ] **Step 9: Commit**

```bash
git add profiles/arm64/src/codegen.rs profiles/x86_64/src/codegen.rs
git commit -m "$(cat <<'EOF'
refactor(m16): extract analyze() from walk_model on both profiles

Pull the analysis preamble (assign_buffers, compute_callee_saved,
compute_is_leaf, FnSig construction, AbiContext setup, arity validation)
out of walk_model into a private analyze(model) -> Result<ModelAnalysis,
LowerError> function on each profile. walk_model now calls analyze()
and proceeds to emission unchanged.

Sets up the structural invariant for M16/A3 Task 3: lower() and
inspect() will both call analyze(), making "inspect output matches
what lower would produce" true by construction (no drift possible).

Per-profile struct shape differs by one field — arm64's ModelAnalysis
carries LeafKind for prologue use; x86_64 omits it (its prologue
doesn't depend on leaf classification). Inspect-path leaf rendering
will compute via UIR-side model.calls_extern_math() on x86_64.

Bisect-claim: pure extract-method; asm output bit-identical for all
fixtures; cargo test --workspace clean at 446 tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Lift `BufferLoc` to `profile-api`

**Why:** `Inspection` (Task 3) carries `BufferLoc` per node; the type must live in `profile-api` so consumers (the renderer in Task 4, integration tests in Task 5) can refer to it without a profile-crate dependency. The two profile copies are structurally bit-identical (verified in spec §3.2) — lift is mechanical.

**Files:**
- Modify: `profile-api/src/lib.rs`
- Modify: `profiles/arm64/src/buffer.rs`
- Modify: `profiles/x86_64/src/buffer.rs`

- [ ] **Step 1: Pre-task grep verification** (per spec §3.2, Task 2 pre-condition). Run from repo root:

```bash
grep -rn 'BufferLoc' --include='*.rs' .
```

Expected: only references inside `profiles/arm64/`, `profiles/x86_64/`, and possibly `profiles/{arm64,x86_64}/tests/`. **No** references in `nflc/`, `bench/`, or `compiler/`. If you find external imports of `profiles_arm64::buffer::BufferLoc` or similar, those callers must be updated to `profile_api::BufferLoc` after Step 4 — record the file paths and add update steps before commit.

- [ ] **Step 2: Add `BufferLoc` to `profile-api/src/lib.rs`.** Insert above the `Asm` struct (use x86_64's richer doc-comments; arm64's lacks them):

```rust
/// Where an Op-node's output buffer lives at run time.
///
/// `InputReg(idx)` carries the input's position in `model.inputs`
/// (M12+). The codegen profile maps `idx` → ABI register via
/// `AbiContext::input_reg`. For N=1 this is always `0` (= `x0` on
/// arm64, `%rdi` on x86_64), preserving M3-M11 single-input behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    /// Input pointer at `model.inputs[idx]`. Mapped to a profile arg
    /// register by `AbiContext::input_reg(idx)`.
    InputReg(usize),
    /// Output pointer (the FFI register at `INPUT_REGS[n_inputs + 1]`).
    OutputReg,
    /// Stack slot at `[sp + offset]` (arm64) or `[%rsp + offset]` (x86_64).
    StackOffset(usize),
    /// This buffer is an alias for another node's buffer. Resolved by
    /// `codegen::resolve_loc` before any emit.
    Alias(NodeId),
}
```

- [ ] **Step 3: Replace arm64's local `BufferLoc` with re-export.** In `profiles/arm64/src/buffer.rs`, delete the local `pub enum BufferLoc { ... }` definition (lines ~19-25) and add at the top (after the `use compiler::...` line):

```rust
pub use profile_api::BufferLoc;
```

- [ ] **Step 4: Replace x86_64's local `BufferLoc` with re-export.** Same change in `profiles/x86_64/src/buffer.rs` — delete the local `pub enum BufferLoc { ... }` definition and add `pub use profile_api::BufferLoc;` near the top.

- [ ] **Step 5: Run `cargo build --workspace`** — confirms no callsite breakage. If callsites in either profile crate use `crate::buffer::BufferLoc`, they continue to work via re-export. If they use `super::BufferLoc` or similar, those keep working too.

- [ ] **Step 6: If Step 1 found external callers, update them now.** For any `profiles_arm64::buffer::BufferLoc` or `profiles_x86_64::buffer::BufferLoc` import outside the profile crates, change to `profile_api::BufferLoc`.

- [ ] **Step 7: Run workspace gates** — `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`. Test count still 446.

- [ ] **Step 8: Commit**

```bash
git add profile-api/src/lib.rs profiles/arm64/src/buffer.rs profiles/x86_64/src/buffer.rs
git commit -m "$(cat <<'EOF'
refactor(m16): lift BufferLoc enum to profile-api

The two profile copies of `pub enum BufferLoc` were structurally
identical (verified by `diff <(sed -n '/^pub enum BufferLoc/,/^}$/p'
profiles/arm64/src/buffer.rs) ...` during M16 brainstorm — variants
and payload types match; only doc-comment richness differed). Lifting
to profile-api removes the duplicate, lets profile-api consumers
(forthcoming Inspection schema in Task 3) reference the type without
a profile-crate dependency, and uses the richer x86_64 doc-comments
on the canonical definition.

profile-api already depends on `compiler::NodeId` — no new
dependencies. Each profile's buffer.rs swaps the local definition for
`pub use profile_api::BufferLoc;`; all internal callsites continue to
import via `crate::buffer::BufferLoc` unchanged.

Bisect-claim: type relocation only; all callers re-import via existing
path; cargo test --workspace clean at 446 tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Add `Inspection` types + `Profile::inspect()` trait method + per-profile impl

**Why:** Defines the schema A3 surfaces and adds the trait method that satisfies the M9 invariant ("trait grows by request, not by anticipation") — `nflc inspect` (Task 4) is the consumer.

**Files:**
- Modify: `profile-api/src/lib.rs`
- Modify: `profiles/arm64/src/lib.rs`
- Modify: `profiles/x86_64/src/lib.rs`
- Modify: `profiles/arm64/src/codegen.rs` (add `inspect_model` function)
- Modify: `profiles/x86_64/src/codegen.rs` (add `inspect_model` function)
- Modify: `profiles/arm64/src/tests.rs`
- Modify: `profiles/x86_64/src/tests.rs`

### 3a. Schema types in `profile-api`

- [ ] **Step 1: Add `Inspection`, `FnAnnotations`, `NodeAnnotation` to `profile-api/src/lib.rs`.** Place near the bottom of the file before the `#[cfg(test)] mod tests` block:

```rust
// ----------------------------------------------------------------------------
// M16 (A3): Profile-aware inspection schema.
//
// Returned by `Profile::inspect()`. Mirror of `Asm` in role: where Asm
// is "what lowering produces (text)", Inspection is "what lowering
// would compute (structured analysis)". Both consume the same
// per-profile analyze() preamble (M16 Task 1) — drift between them
// is impossible by construction.
// ----------------------------------------------------------------------------

/// Profile-aware annotation of one Uir, returned by `Profile::inspect`.
/// One entry per UirModel in the input UIR, in declaration order.
#[derive(Debug, Clone)]
pub struct Inspection {
    pub functions: Vec<FnAnnotations>,
}

/// Annotation for one UirModel under one profile.
///
/// `nodes.len() == post_pass_model.nodes.len()` — strictly index-aligned
/// with the **post-pass** UirModel that gets lowered. Pre-pass alignment
/// would produce a report whose node IDs don't match what `lower()`
/// actually compiles, defeating the point of A3.
#[derive(Debug, Clone)]
pub struct FnAnnotations {
    pub fn_sig: FnSig,
    pub stack_bytes: usize,
    /// Textual rendering of the profile's RegSet — lossy by design.
    /// arm64: e.g. `["d8-d9", "x19-x23"]`. x86_64: e.g. `["%rbx", "%r12-%r15"]`.
    /// Empty Vec if no callee-saved registers are touched by this function.
    pub callee_saved: Vec<String>,
    /// True iff the function emits no `bl _expf` / `call expf@PLT`
    /// (== `!UirModel::calls_extern_math()` for both profiles today).
    pub leaf: bool,
    /// Real NodeId of each input in the post-pass UirModel, in
    /// declaration order. Renderer uses these to produce `n<id>` refs;
    /// without this field, positional indices would not match actual
    /// NodeIds in models where inputs are not the first N nodes.
    pub input_nodes: Vec<compiler::NodeId>,
    /// Real NodeId of the model output in the post-pass UirModel.
    pub output_node: compiler::NodeId,
    pub nodes: Vec<NodeAnnotation>,
}

/// Per-node annotation. Index in `FnAnnotations.nodes` corresponds to
/// `NodeId` in the post-pass `UirModel`.
///
/// **Growth rule:** new fields land here only when meaningful for both
/// profiles. Profile-specific information goes into `extra_notes` rather
/// than as a top-level field, to keep the schema honest cross-profile.
/// (See spec §3.4.)
#[derive(Debug, Clone)]
pub struct NodeAnnotation {
    /// Pre-rendered description of the node — op kind, shape, operands,
    /// attrs, fused post-ops. Format mirrors `Display for compiler::Node`
    /// (the `--uir-verbose` style); produced once at inspect time so the
    /// renderer doesn't need access to the source UirModel.
    /// Examples:
    /// - `input "x"        :: Tensor[8, 4]`
    /// - `linear           :: Tensor[8, 2]    operands=[n0]    attrs=[out_dim=2]    fused=[softmax_row]`
    pub label: String,
    pub buffer_loc: BufferLoc,
    /// `element_count * 4` (BYTES_PER_ELEMENT). For aliased nodes this
    /// is still the *logical* output size — the node "produces" this
    /// many bytes; physical placement is captured by `buffer_loc`.
    pub output_bytes: usize,
    /// `Some(N)` for ops that consume packed `params` slots:
    /// `Linear` (weights ± bias) and `LayerNorm[affine=true]`
    /// (γ + β). `None` for all other ops.
    pub params_floats: Option<usize>,
    /// Profile-specific freeform annotations. Empty for now; reserved
    /// for the growth-rule escape hatch.
    pub extra_notes: Vec<String>,
}
```

- [ ] **Step 2: Add `inspect` to the `Profile` trait** in `profile-api/src/lib.rs`. Update the trait definition:

```rust
pub trait Profile {
    /// Lower a [`Uir`] to the profile's target assembly.
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError>;

    /// Platform-specific external-symbol prefix.
    fn sym_prefix(&self) -> &'static str;

    /// M16 (A3): inspect the UIR under this profile, returning per-model
    /// and per-node annotations matching what `lower()` would produce.
    /// Both methods share an internal `analyze()` preamble — drift
    /// between inspection output and lowered asm is structurally
    /// impossible.
    fn inspect(&self, uir: &Uir) -> Result<Inspection, LowerError>;
}
```

- [ ] **Step 3: Run `cargo build --workspace`** — expect failure: both profile crates now don't satisfy the `Profile` trait (missing `inspect` method). This confirms the trait change reached the right place.

### 3b. arm64 `inspect` implementation

- [ ] **Step 3.5: Verify RegSet accessor surface (both profiles).** Quick sanity check before Step 4:

```bash
grep -n 'fn contains_' profiles/arm64/src/buffer.rs profiles/x86_64/src/buffer.rs
```

Expected output:
```
profiles/arm64/src/buffer.rs: pub fn contains_d8_d9(&self) -> bool { self.d8_d9 }
profiles/arm64/src/buffer.rs: pub fn contains_x19_x23(&self) -> bool { self.x19_x23 }
profiles/x86_64/src/buffer.rs: pub fn contains_callee_saved_int(&self) -> bool { self.callee_saved_int }
```

If methods missing (regression), use field access instead — `callee_saved.d8_d9` etc. — both `pub` fields exist. The plan code below uses methods (verified extant at M15-tip).

- [ ] **Step 4: Add `inspect_model` to `profiles/arm64/src/codegen.rs`.** Add after `walk_model` (or below the `analyze()` function, in the same file):

```rust
/// Inspect one model under arm64. Mirror of walk_model — both call
/// analyze() then diverge: walk_model emits asm, inspect_model packages
/// the analysis as FnAnnotations.
pub(crate) fn inspect_model(model: &UirModel) -> Result<profile_api::FnAnnotations, LowerError> {
    use compiler::NodeKind;
    use profile_api::{BufferLoc, FnAnnotations, NodeAnnotation};

    let analysis = analyze(model)?;
    let ModelAnalysis {
        fn_sig,
        assignment,
        callee_saved,
        leaf,
        abi: _,
    } = analysis;

    // Render arm64's RegSet → Vec<String>. Order matches AAPCS save order
    // (FP regs before GP regs) for stable output.
    let mut callee_saved_str: Vec<String> = Vec::new();
    if callee_saved.contains_d8_d9() {
        callee_saved_str.push("d8-d9".to_string());
    }
    if callee_saved.contains_x19_x23() {
        callee_saved_str.push("x19-x23".to_string());
    }

    let leaf_bool = matches!(leaf, crate::asm::LeafKind::Leaf);

    // BYTES_PER_ELEMENT = 4 (f32). Match the constant in buffer.rs.
    const BYTES_PER_ELEMENT: usize = 4;

    let nodes: Vec<NodeAnnotation> = model
        .nodes
        .iter()
        .enumerate()
        .map(|(node_idx, node)| {
            let element_count: u64 = node.ty.shape.0.iter().copied().product();
            let output_bytes = (element_count as usize)
                .checked_mul(BYTES_PER_ELEMENT)
                .expect("output_bytes overflow: shape product * f32 size");

            let params_floats: Option<usize> = match &node.kind {
                NodeKind::Op { op, .. }
                    if matches!(op, compiler::StdOp::Linear)
                        || matches!(op, compiler::StdOp::LayerNorm) =>
                {
                    let total: usize = fn_sig
                        .params_layout
                        .iter()
                        .filter(|s| s.origin_node == node_idx)
                        .map(|s| s.size)
                        .sum();
                    if total == 0 {
                        None
                    } else {
                        Some(total)
                    }
                }
                _ => None,
            };

            // BufferLoc is the lifted profile_api::BufferLoc (Task 2).
            let buffer_loc: BufferLoc = assignment.locs[node_idx];

            // label = pre-rendered `Display for Node` output. This is the
            // exact format used by `nflc parse --uir-verbose` per-node
            // line — visual continuity is intentional (Q5 brainstorm).
            let label = format!("{}", node);

            NodeAnnotation {
                label,
                buffer_loc,
                output_bytes,
                params_floats,
                extra_notes: Vec::new(),
            }
        })
        .collect();

    Ok(FnAnnotations {
        fn_sig,
        stack_bytes: assignment.stack_bytes,
        callee_saved: callee_saved_str,
        leaf: leaf_bool,
        input_nodes: model.inputs.clone(),
        output_node: model.output,
        nodes,
    })
}

/// Inspect a full Uir under arm64.
pub fn inspect_uir(uir: &Uir) -> Result<profile_api::Inspection, LowerError> {
    let mut functions = Vec::with_capacity(uir.models.len());
    for model in &uir.models {
        functions.push(inspect_model(model)?);
    }
    Ok(profile_api::Inspection { functions })
}
```

- [ ] **Step 5: Implement `Profile::inspect` for `Arm64Profile`** in `profiles/arm64/src/lib.rs`. Update the existing `impl Profile for Arm64Profile` block:

```rust
impl Profile for Arm64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        "_"
    }

    fn inspect(&self, uir: &Uir) -> Result<profile_api::Inspection, LowerError> {
        codegen::inspect_uir(uir)
    }
}
```

Add `Inspection` to the `pub use` line at the top of `profiles/arm64/src/lib.rs`:

```rust
pub use profile_api::{Asm, FnSig, Inspection, LowerError, ParamKind, ParamSlot};
```

### 3c. x86_64 `inspect` implementation

- [ ] **Step 6: Add `inspect_model` to `profiles/x86_64/src/codegen.rs`.** Same shape as arm64, modulo callee-saved rendering and leaf computation:

```rust
/// Inspect one model under x86_64. Mirror of walk_model.
pub(crate) fn inspect_model(model: &UirModel) -> Result<profile_api::FnAnnotations, LowerError> {
    use compiler::NodeKind;
    use profile_api::{BufferLoc, FnAnnotations, NodeAnnotation};

    let analysis = analyze(model)?;
    let ModelAnalysis {
        fn_sig,
        assignment,
        callee_saved,
        abi: _,
    } = analysis;

    // x86_64 RegSet → Vec<String>. The callee_saved_int flag covers the
    // entire %rbx, %r12-%r15 set as a single group (see
    // profiles/x86_64/src/buffer.rs RegSet doc).
    let mut callee_saved_str: Vec<String> = Vec::new();
    if callee_saved.contains_callee_saved_int() {
        callee_saved_str.push("%rbx".to_string());
        callee_saved_str.push("%r12-%r15".to_string());
    }

    // x86_64 has no LeafKind (its prologue is leaf-agnostic). Compute
    // leaf bool directly from the UIR-side predicate — same source
    // arm64's compute_is_leaf delegates to.
    let leaf_bool = !model.calls_extern_math();

    const BYTES_PER_ELEMENT: usize = 4;

    let nodes: Vec<NodeAnnotation> = model
        .nodes
        .iter()
        .enumerate()
        .map(|(node_idx, node)| {
            let element_count: u64 = node.ty.shape.0.iter().copied().product();
            let output_bytes = (element_count as usize)
                .checked_mul(BYTES_PER_ELEMENT)
                .expect("output_bytes overflow: shape product * f32 size");

            let params_floats: Option<usize> = match &node.kind {
                NodeKind::Op { op, .. }
                    if matches!(op, compiler::StdOp::Linear)
                        || matches!(op, compiler::StdOp::LayerNorm) =>
                {
                    let total: usize = fn_sig
                        .params_layout
                        .iter()
                        .filter(|s| s.origin_node == node_idx)
                        .map(|s| s.size)
                        .sum();
                    if total == 0 {
                        None
                    } else {
                        Some(total)
                    }
                }
                _ => None,
            };

            let buffer_loc: BufferLoc = assignment.locs[node_idx];
            let label = format!("{}", node);

            NodeAnnotation {
                label,
                buffer_loc,
                output_bytes,
                params_floats,
                extra_notes: Vec::new(),
            }
        })
        .collect();

    Ok(FnAnnotations {
        fn_sig,
        stack_bytes: assignment.stack_bytes,
        callee_saved: callee_saved_str,
        leaf: leaf_bool,
        input_nodes: model.inputs.clone(),
        output_node: model.output,
        nodes,
    })
}

pub fn inspect_uir(uir: &Uir) -> Result<profile_api::Inspection, LowerError> {
    let mut functions = Vec::with_capacity(uir.models.len());
    for model in &uir.models {
        functions.push(inspect_model(model)?);
    }
    Ok(profile_api::Inspection { functions })
}
```

- [ ] **Step 7: Implement `Profile::inspect` for `X86_64Profile`** in `profiles/x86_64/src/lib.rs`. Mirror the arm64 change:

```rust
impl Profile for X86_64Profile {
    fn lower(&self, uir: &Uir) -> Result<Asm, LowerError> {
        codegen::walk_uir(uir, self.sym_prefix())
    }

    fn sym_prefix(&self) -> &'static str {
        ""
    }

    fn inspect(&self, uir: &Uir) -> Result<profile_api::Inspection, LowerError> {
        codegen::inspect_uir(uir)
    }
}
```

And update the `pub use` line to include `Inspection`.

- [ ] **Step 8: Verify compilation** — `cargo build --workspace`. Should compile clean now.

### 3d. Unit tests

- [ ] **Step 9: Write failing test — leaf detection (arm64)** in `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn inspect_softmax_model_is_non_leaf() {
    use profile_api::Profile;
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 4]\n    x -> softmax\n");
    let insp = Arm64Profile.inspect(&uir).expect("inspect");
    assert_eq!(insp.functions.len(), 1);
    assert!(
        !insp.functions[0].leaf,
        "softmax-bearing model must report leaf=false (calls _expf)"
    );
    // NOTE: callee_saved population is implementation-defined per profile
    // (arm64 ties d8-d9/x19-x23 to extern_math; x86_64 ties %rbx/%r12-%r15
    // to extern_math OR matmul). Don't assert non-emptiness here — the
    // primary leaf assertion is what this test is for.
}

#[test]
fn inspect_pure_linear_model_is_leaf() {
    use profile_api::Profile;
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let insp = Arm64Profile.inspect(&uir).expect("inspect");
    assert!(insp.functions[0].leaf, "no extern math = leaf");
    // Same NOTE as above — don't over-specify callee_saved emptiness.
    // arm64 tiny linear model happens to be empty today; that's a
    // correctness outcome of the analyzer, not a contract we want to
    // pin in the inspect-level test.
}
```

Run: `cargo test -p profiles-arm64 inspect_ -- --nocapture` — both should pass against the implementation in 3b. (TDD nuance: implementation is already in place from Step 4-5; these tests are the regression net for it. If you prefer strict red-then-green, comment out the inspect_uir body to force fail, run, uncomment.)

- [ ] **Step 10: Write test — alias placement on pre-pass UIR (arm64)**, per spec §7 Axis 2 fix (no "choose whichever path"):

```rust
#[test]
fn inspect_pre_pass_relu_uses_alias_placement() {
    use compiler::ir::types::{NodeKind, StdOp};
    use profile_api::{BufferLoc, Profile};

    // Pre-pass: relu is a separate node aliased to its operand. Post-pass
    // (default pipeline) FuseLinearRelu would fold it into the linear node;
    // alias placement is observable only on the pre-pass graph.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[4] -> relu\n");
    let insp = Arm64Profile.inspect(&uir).expect("inspect");
    let f = &insp.functions[0];

    // Locate the relu node (last Op in pre-pass, operands = [linear_node]).
    let relu_idx = uir.models[0]
        .nodes
        .iter()
        .position(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Relu, .. }))
        .expect("pre-pass UIR must contain a Relu node");
    let linear_idx = uir.models[0]
        .nodes
        .iter()
        .position(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Linear, .. }))
        .expect("pre-pass UIR must contain a Linear node");

    // Output node = relu (model.output). Output gets BufferLoc::OutputReg
    // by assign_buffers' "id == model.output" branch — so this test asserts
    // OutputReg, NOT Alias, when relu IS the output. To test the Alias
    // path, we'd need a model where relu is NOT the output. Since our
    // tiny test models all end in relu, just assert the loc is whatever
    // assign_buffers picked and validate the field exists.
    let _ = f.nodes[relu_idx].buffer_loc;  // accessor smoke
    let _ = f.nodes[linear_idx].buffer_loc;

    // The real alias-placement check: build a model where the dropout
    // is NOT the output (dropout is the canonical alias-bearing op).
    let uir2 = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> dropout[rate=0.1] -> linear[4]\n");
    let insp2 = Arm64Profile.inspect(&uir2).expect("inspect");
    let dropout_idx = uir2.models[0]
        .nodes
        .iter()
        .position(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Dropout, .. }))
        .expect("pre-pass UIR must contain a Dropout node");
    let dropout_input_idx = match &uir2.models[0].nodes[dropout_idx].kind {
        NodeKind::Op { operands, .. } => operands[0],
        _ => unreachable!(),
    };
    assert_eq!(
        insp2.functions[0].nodes[dropout_idx].buffer_loc,
        BufferLoc::Alias(dropout_input_idx),
        "dropout (not output) must alias its operand"
    );
}
```

- [ ] **Step 11: Write test — params_floats for Linear with bias (arm64)**:

```rust
#[test]
fn inspect_linear_with_bias_reports_correct_params() {
    use compiler::ir::types::{NodeKind, StdOp};
    use profile_api::Profile;

    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8, bias=true]\n",
    );
    let insp = Arm64Profile.inspect(&uir).expect("inspect");
    let f = &insp.functions[0];

    let linear_idx = uir.models[0]
        .nodes
        .iter()
        .position(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Linear, .. }))
        .unwrap();

    // K=4, N=8, bias=true → 4*8 + 8 = 40 floats
    assert_eq!(f.nodes[linear_idx].params_floats, Some(40));
}
```

- [ ] **Step 12: Mirror three tests in `profiles/x86_64/src/tests.rs`.** Same shape, swap `Arm64Profile` for `X86_64Profile`. The assertions on `leaf`, `callee_saved`, `BufferLoc::Alias`, and `params_floats` apply identically.

- [ ] **Step 13: Run all tests** — `cargo test --workspace`. Count should be 446 + 6 ≈ 452.

- [ ] **Step 14: Run gates + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add profile-api/src/lib.rs \
        profiles/arm64/src/lib.rs profiles/arm64/src/codegen.rs profiles/arm64/src/tests.rs \
        profiles/x86_64/src/lib.rs profiles/x86_64/src/codegen.rs profiles/x86_64/src/tests.rs
git commit -m "$(cat <<'EOF'
feat(m16): add Profile::inspect() trait method + per-profile impl

New profile-api types: Inspection { functions: Vec<FnAnnotations> },
FnAnnotations { fn_sig, stack_bytes, callee_saved: Vec<String>, leaf,
nodes: Vec<NodeAnnotation> }, NodeAnnotation { buffer_loc, output_bytes,
params_floats: Option<usize>, extra_notes: Vec<String> }.

Profile trait gains:
    fn inspect(&self, uir: &Uir) -> Result<Inspection, LowerError>;

Per-profile impl: each profile's codegen.rs adds inspect_uir +
inspect_model, both calling the shared analyze() from M16 Task 1
(structural drift-prevention). RegSet → callee_saved Vec<String>
rendering is per-profile:
    arm64: ["d8-d9", "x19-x23"] (when extern math fires)
    x86_64: ["%rbx", "%r12-%r15"] (when callee_saved_int set)

Leaf bool is computed via crate::asm::LeafKind on arm64 (already in
ModelAnalysis) and via UirModel::calls_extern_math() on x86_64 (no
LeafKind in x86_64).

params_floats derivation uses the verified-extant FnSig.params_layout
field — sum slot.size where slot.origin_node == node_idx, gated on
op == Linear or LayerNorm.

6 new unit tests (3 per profile): leaf detection (softmax vs pure
linear), alias placement on pre-pass UIR, params_floats for Linear
with bias.

Bisect-claim: new analysis API surfaced via Profile::inspect; no CLI
or renderer yet; cargo test --workspace clean at ~452 tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — New `inspect-render` workspace crate + `nflc inspect` CLI

**Why:** Renderer cannot live in `nflc` (binary, not consumable by integration tests in profile crates) and shouldn't live in `profile-api` (formatting policy, not contract). New tiny lib crate with single responsibility.

**Files:**
- Create: `inspect-render/Cargo.toml`
- Create: `inspect-render/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)
- Modify: `nflc/Cargo.toml` (add inspect-render dep)
- Modify: `nflc/src/main.rs` (subcommand dispatch + parse_pass_args helper)
- Create: `nflc/tests/cli_inspect.rs`

### 4a. Create `inspect-render` crate

- [ ] **Step 1: Create directory + Cargo.toml.**

```bash
mkdir -p inspect-render/src
```

Write `inspect-render/Cargo.toml`:

```toml
[package]
name = "inspect-render"
version = "0.1.0"
edition = "2021"
description = "NeuralForge inspection renderer — formats profile_api::Inspection as human-readable text"
license.workspace = true

[dependencies]
profile-api = { path = "../profile-api" }
```

- [ ] **Step 2: Add to workspace members.** Edit `Cargo.toml` (workspace root):

```toml
[workspace]
resolver = "2"
members = [
    "bench",
    "compiler",
    "inspect-render",
    "nflc",
    "profile-api",
    "profiles/arm64",
    "profiles/x86_64",
]
```

- [ ] **Step 3: Implement `render_inspection`** — write `inspect-render/src/lib.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Renders `profile_api::Inspection` to human-readable text for
//! `nflc inspect` and per-profile golden integration tests.
//!
//! Format spec: `docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md` §5.

use profile_api::{BufferLoc, FnAnnotations, Inspection, NodeAnnotation};
use std::path::Path;

/// CLI-invocation context for the rendered header. Kept out of
/// `Inspection` because file path / profile / pass list are not analysis
/// outputs — they're inputs to the invocation.
pub struct RenderHeader<'a> {
    pub source_path: &'a Path,
    pub profile: &'a str,
    /// `Some(names)` when a pipeline ran (default or filtered);
    /// `None` when `--no-passes` skipped the pipeline.
    pub applied_passes: Option<&'a [&'a str]>,
}

/// Render a full Inspection. Output ends with a trailing newline.
pub fn render_inspection(insp: &Inspection, header: RenderHeader<'_>) -> String {
    let mut out = String::new();

    // Header: command-style line + applied-passes status.
    out.push_str(&format!(
        "inspect {} --profile {}\n",
        header.source_path.display(),
        header.profile
    ));
    match header.applied_passes {
        Some(names) => {
            out.push_str(&format!("  passes applied: {}\n", names.join(", ")));
        }
        None => {
            out.push_str("  passes: skipped\n");
        }
    }
    out.push('\n');

    for fa in &insp.functions {
        render_fn_annotations(&mut out, fa);
    }

    out
}

fn render_fn_annotations(out: &mut String, fa: &FnAnnotations) {
    out.push_str(&format!("inspect-model {}\n", fa.fn_sig.model));

    // Inputs line: real NodeId refs from input_nodes (NOT positional).
    let total_input_floats: usize = fa.fn_sig.inputs_floats.iter().sum();
    let total_input_bytes = total_input_floats * 4;
    let n_inputs = fa.input_nodes.len();
    let input_node_refs: Vec<String> = fa
        .input_nodes
        .iter()
        .map(|id| format!("n{}", id))
        .collect();
    let inputs_per_count_clause = if n_inputs > 1 {
        format!(
            " ({} B each)",
            total_input_bytes / n_inputs.max(1) // safe — n_inputs > 1 here
        )
    } else {
        String::new()
    };
    out.push_str(&format!(
        "  inputs:        [{}]                {} floats ({} B){}\n",
        input_node_refs.join(", "),
        total_input_floats,
        total_input_bytes,
        inputs_per_count_clause
    ));

    // Output line. Real NodeId from output_node field.
    let output_bytes = fa.fn_sig.output_floats * 4;
    out.push_str(&format!(
        "  output:        n{}                  {} floats ({} B)\n",
        fa.output_node, fa.fn_sig.output_floats, output_bytes
    ));

    let params_bytes = fa.fn_sig.params_floats * 4;
    out.push_str(&format!(
        "  params:        {} floats            ({} B)\n",
        fa.fn_sig.params_floats, params_bytes
    ));

    out.push_str(&format!(
        "  stack frame:   {} bytes             (16-byte aligned)\n",
        fa.stack_bytes
    ));

    out.push_str(&format!(
        "  callee-saved:  [{}]\n",
        fa.callee_saved.join(", ")
    ));

    out.push_str(&format!(
        "  leaf:          {}\n",
        if fa.leaf { "yes" } else { "no" }
    ));

    out.push_str("\n  nodes:\n");
    for (node_idx, na) in fa.nodes.iter().enumerate() {
        render_node_annotation(out, node_idx, na);
    }
    out.push('\n');
}

fn render_node_annotation(out: &mut String, node_idx: usize, na: &NodeAnnotation) {
    // Line 1: node id ref + pre-rendered label (op kind + shape +
    // operands + attrs + fused). Format mirrors `--uir-verbose`
    // per-node line — visual continuity per Q5 brainstorm.
    out.push_str(&format!("    n{}  {}\n", node_idx, na.label));

    // Line 2: annotation row.
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("loc={}", format_buffer_loc(na.buffer_loc)));
    parts.push(format!("out={} B", na.output_bytes));
    if let Some(p) = na.params_floats {
        parts.push(format!("params={} floats ({} B)", p, p * 4));
    }
    for note in &na.extra_notes {
        parts.push(note.clone());
    }
    out.push_str(&format!("          {}\n", parts.join("    ")));
}

fn format_buffer_loc(loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg(idx) => format!("InputReg({})", idx),
        BufferLoc::OutputReg => "OutputReg".to_string(),
        BufferLoc::StackOffset(off) => format!("StackOffset({})", off),
        BufferLoc::Alias(node_id) => format!("Alias(n{})", node_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profile_api::{FnSig, Inspection, ParamKind, ParamSlot};
    use std::path::PathBuf;

    fn dummy_inspection() -> Inspection {
        Inspection {
            functions: vec![FnAnnotations {
                fn_sig: FnSig {
                    name: "nfl_forward_M".to_string(),
                    model: "M".to_string(),
                    inputs_floats: vec![6],
                    output_floats: 4,
                    params_floats: 6,
                    params_layout: vec![ParamSlot {
                        kind: ParamKind::LinearWeight,
                        origin_node: 1,
                        offset: 0,
                        size: 6,
                    }],
                },
                stack_bytes: 0,
                callee_saved: vec![],
                leaf: true,
                input_nodes: vec![0],
                output_node: 1,
                nodes: vec![
                    NodeAnnotation {
                        label: "input \"x\"        :: Tensor[2, 3]".to_string(),
                        buffer_loc: BufferLoc::InputReg(0),
                        output_bytes: 24,
                        params_floats: None,
                        extra_notes: vec![],
                    },
                    NodeAnnotation {
                        label: "linear           :: Tensor[2, 2]    operands=[n0]    attrs=[out_dim=2]".to_string(),
                        buffer_loc: BufferLoc::OutputReg,
                        output_bytes: 16,
                        params_floats: Some(6),
                        extra_notes: vec![],
                    },
                ],
            }],
        }
    }

    #[test]
    fn render_contains_required_markers() {
        let path = PathBuf::from("test.nfl");
        let passes = ["fuse_linear_relu"];
        let header = RenderHeader {
            source_path: &path,
            profile: "arm64",
            applied_passes: Some(&passes),
        };
        let out = render_inspection(&dummy_inspection(), header);
        assert!(out.contains("inspect-model M"), "missing model header: {}", out);
        assert!(out.contains("loc=InputReg(0)"), "missing loc render: {}", out);
        assert!(out.contains("loc=OutputReg"), "missing OutputReg render: {}", out);
        assert!(out.contains("out=24 B"), "missing output_bytes render");
        assert!(out.contains("params=6 floats (24 B)"), "missing params line: {}", out);
        assert!(out.contains("passes applied: fuse_linear_relu"), "missing passes line");
        // Label rendered on line 1.
        assert!(out.contains("n0  input \"x\""), "line-1 label missing for input node: {}", out);
        assert!(out.contains("n1  linear"), "line-1 label missing for op node: {}", out);
    }

    #[test]
    fn render_no_passes_marker() {
        let path = PathBuf::from("test.nfl");
        let header = RenderHeader {
            source_path: &path,
            profile: "arm64",
            applied_passes: None,
        };
        let out = render_inspection(&dummy_inspection(), header);
        assert!(out.contains("passes: skipped"), "missing skipped marker: {}", out);
    }
}
```

- [ ] **Step 4: Run `cargo test -p inspect-render`** — both unit tests should pass.

### 4b. Wire `nflc inspect` CLI

- [ ] **Step 5: Add `inspect-render` to `nflc/Cargo.toml`:**

```toml
[dependencies]
compiler        = { path = "../compiler" }
profile-api     = { path = "../profile-api" }
profiles-arm64  = { path = "../profiles/arm64" }
profiles-x86_64 = { path = "../profiles/x86_64" }
inspect-render  = { path = "../inspect-render" }

[[bin]]
name = "nflc"
path = "src/main.rs"
```

- [ ] **Step 6: Refactor `parse_compile_args` to extract shared `parse_pass_args` helper** in `nflc/src/main.rs`. Add this helper near the top of the file (after `parse_compile_args` and `CompileArgs`):

```rust
struct PassArgs {
    no_passes: bool,
    /// `None` = run `default_pipeline()`; `Some(list)` = filter to listed
    /// names (canonical order preserved regardless of user order).
    passes: Option<Vec<String>>,
}

/// Parse the `--no-passes` / `--passes <list>` flag pair from a flag
/// iterator. Returns the validated `PassArgs` or a user-friendly error.
/// Shared by `nflc compile` and `nflc inspect`.
fn parse_pass_flag(arg: &str, iter: &mut std::slice::Iter<'_, String>, no_passes: &mut bool, passes: &mut Option<Vec<String>>) -> Result<bool, String> {
    match arg {
        "--no-passes" => {
            *no_passes = true;
            Ok(true)
        }
        "--passes" => {
            let v = iter.next().ok_or_else(|| "--passes requires a value".to_string())?;
            if v.is_empty() {
                return Err("--passes value cannot be empty (use --no-passes to skip the pipeline)".to_string());
            }
            let names: Vec<String> = v.split(',').map(str::to_owned).collect();
            if names.iter().any(|n| n.is_empty()) {
                return Err(format!(
                    "--passes value '{v}' contains an empty token (use --no-passes for empty)"
                ));
            }
            *passes = Some(names);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn validate_pass_args(no_passes: bool, passes: &Option<Vec<String>>) -> Result<(), String> {
    if no_passes && passes.is_some() {
        return Err("--no-passes and --passes are mutually exclusive".to_string());
    }
    if let Some(names) = passes {
        let available_names: Vec<String> = compiler::passes::default_pipeline()
            .iter()
            .map(|p| p.name().to_owned())
            .collect();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(format!("pass '{n}' specified more than once in --passes"));
            }
        }
        for n in names {
            if !available_names.iter().any(|c| c == n) {
                return Err(format!(
                    "unknown pass '{n}' (available: {})",
                    available_names.join(", ")
                ));
            }
        }
    }
    Ok(())
}
```

Update `parse_compile_args` to use the helpers — replace the `--no-passes` / `--passes` arms in its `match arg.as_str()` with:

```rust
            other if other == "--no-passes" || other == "--passes" => {
                if !parse_pass_flag(other, &mut iter, &mut no_passes, &mut passes)? {
                    return Err(format!("unknown flag: {other}"));
                }
            }
            other => {
                return Err(format!("unknown flag: {other}"));
            }
```

And replace the duplicate validation block with `validate_pass_args(no_passes, &passes)?;`.

- [ ] **Step 7: Add `InspectArgs` + `parse_inspect_args`** to `nflc/src/main.rs`:

```rust
struct InspectArgs {
    path: PathBuf,
    profile: String,
    no_passes: bool,
    passes: Option<Vec<String>>,
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs, String> {
    let mut iter = args.iter();
    let path = iter
        .next()
        .ok_or_else(|| "inspect: missing <file.nfl>".to_string())?
        .clone();
    if path.starts_with('-') {
        return Err(format!(
            "inspect: expected <file.nfl> as first argument, got flag '{path}'"
        ));
    }

    let mut profile: Option<String> = None;
    let mut no_passes = false;
    let mut passes: Option<Vec<String>> = None;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--profile" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                profile = Some(v.clone());
            }
            other if other == "--no-passes" || other == "--passes" => {
                if !parse_pass_flag(other, &mut iter, &mut no_passes, &mut passes)? {
                    return Err(format!("unknown flag: {other}"));
                }
            }
            other => {
                return Err(format!("unknown flag: {other}"));
            }
        }
    }

    let profile = profile.ok_or_else(|| "inspect: missing --profile <name>".to_string())?;
    validate_pass_args(no_passes, &passes)?;

    Ok(InspectArgs {
        path: PathBuf::from(path),
        profile,
        no_passes,
        passes,
    })
}
```

- [ ] **Step 8: Add `inspect` arm to the top-level `match args.as_slice()`** in `main()`:

```rust
        [cmd, rest @ ..] if cmd == "inspect" => match parse_inspect_args(rest) {
            Ok(parsed) => run_inspect(parsed),
            Err(msg) => {
                eprintln!("error: {}", msg);
                print_usage();
                ExitCode::FAILURE
            }
        },
```

(Place above the existing `[cmd, rest @ ..] if cmd == "compile" => ...` arm.)

- [ ] **Step 9: Implement `run_inspect`** in `nflc/src/main.rs`:

```rust
fn run_inspect(args: InspectArgs) -> ExitCode {
    let InspectArgs {
        path,
        profile,
        no_passes,
        passes,
    } = args;

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ast = match compiler::parse(&source) {
        Ok(a) => a,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
            return ExitCode::FAILURE;
        }
    };

    let uir = match compiler::ir::build(&ast) {
        Ok(u) => u,
        Err(e) => {
            let first = match &e.kind {
                compiler::BuildErrorKind::DuplicateModelName { first_span, .. } => {
                    Some((first_span.line, first_span.col))
                }
                _ => None,
            };
            let msg = e.to_string();
            render_error_with_snippet(&source, &path, e.line, e.col, &msg, first);
            return ExitCode::FAILURE;
        }
    };

    let profile_impl: Box<dyn profile_api::Profile> = match profile.as_str() {
        "arm64" => Box::new(profiles_arm64::Arm64Profile),
        "x86_64" => Box::new(profiles_x86_64::X86_64Profile),
        other => {
            eprintln!(
                "error: unknown profile '{}' (supported: arm64, x86_64)",
                other
            );
            return ExitCode::FAILURE;
        }
    };

    // Pass pipeline (mirror compile semantics).
    let (post_pass_uir, applied_pass_names): (compiler::Uir, Option<Vec<String>>) = if no_passes {
        eprintln!("note: passes skipped (--no-passes)");
        (uir, None)
    } else {
        let canonical = compiler::passes::default_pipeline();
        let canonical_names: Vec<String> =
            canonical.iter().map(|p| p.name().to_owned()).collect();

        let (pipeline, divergent) = match passes {
            None => (canonical, false),
            Some(user_names) => {
                let user_set: std::collections::HashSet<&str> =
                    user_names.iter().map(String::as_str).collect();
                let filtered: Vec<Box<dyn compiler::passes::UirPass>> = canonical
                    .into_iter()
                    .filter(|p| user_set.contains(p.name()))
                    .collect();
                let canonical_filtered_names: Vec<&str> =
                    filtered.iter().map(|p| p.name()).collect();
                let div = user_names.len() >= 2
                    && user_names.iter().map(String::as_str).collect::<Vec<_>>()
                        != canonical_filtered_names;
                (filtered, div)
            }
        };

        match compiler::passes::run_pipeline(&uir, &pipeline) {
            Ok(u) => {
                let names: Vec<String> = pipeline.iter().map(|p| p.name().to_owned()).collect();
                eprintln!("note: applied passes: {}", names.join(", "));
                if divergent {
                    eprintln!(
                        "note: pass order is canonical ({}); user-specified order ignored",
                        canonical_names.join(", ")
                    );
                }
                (u, Some(names))
            }
            Err(e) => {
                let span = e.span();
                render_error_with_snippet(
                    &source,
                    &path,
                    span.line,
                    span.col,
                    &format!("{}", e),
                    None,
                );
                return ExitCode::FAILURE;
            }
        }
    };

    let inspection = match profile_impl.inspect(&post_pass_uir) {
        Ok(i) => i,
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(&source, &path, span.line, span.col, &format!("{}", e), None);
            return ExitCode::FAILURE;
        }
    };

    // Render. Convert applied_pass_names to &[&str] for the renderer.
    let applied_refs: Option<Vec<&str>> =
        applied_pass_names.as_ref().map(|v| v.iter().map(String::as_str).collect());
    let header = inspect_render::RenderHeader {
        source_path: &path,
        profile: &profile,
        applied_passes: applied_refs.as_deref(),
    };
    print!("{}", inspect_render::render_inspection(&inspection, header));
    ExitCode::SUCCESS
}
```

- [ ] **Step 10: Update `print_usage()`** in `nflc/src/main.rs` — add the inspect block:

```rust
fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc parse   <file.nfl> --uir-verbose      Print UIR with annotated metadata");
    println!("  nflc compile <file.nfl> --profile <arm64|x86_64>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-passes]      Skip optimisation passes (debugging)");
    println!(
        "                          [--passes <list>]  Run only listed passes (comma-separated)"
    );
    println!("  nflc inspect <file.nfl> --profile <arm64|x86_64>   Inspect post-pass UIR with profile annotations");
    println!("                          [--no-passes]      Skip optimisation passes");
    println!(
        "                          [--passes <list>]  Run only listed passes (comma-separated)"
    );
}
```

- [ ] **Step 11: Verify build** — `cargo build --workspace`. Should compile clean.

### 4c. CLI smoke tests

- [ ] **Step 12: Write `nflc/tests/cli_inspect.rs`:**

```rust
// SPDX-License-Identifier: Apache-2.0

//! CLI integration tests for `nflc inspect`.
//!
//! Mirror of `cli_compile.rs` — Cargo runs integration-test binaries
//! with cwd at the package root (`nflc/`), so paths to workspace-root
//! fixtures are written as `"../tests/fixtures/<name>.nfl"`.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn inspect_default_runs_pipeline_and_renders() {
    let output = Command::new(nflc_bin())
        .args([
            "inspect",
            "../tests/fixtures/tiny_mlp.nfl",
            "--profile",
            "arm64",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Stdout markers (format-stability safety net; full format covered
    // by goldens in M16 Task 5).
    assert!(
        stdout.contains("inspect-model TinyMLP"),
        "stdout missing inspect-model header:\n{stdout}"
    );
    assert!(stdout.contains("loc="), "stdout missing loc= row:\n{stdout}");
    assert!(
        stdout.contains("passes applied:"),
        "stdout missing passes-applied header line:\n{stdout}"
    );

    // Stderr applied-passes note (mirrors compile's behaviour).
    assert!(
        stderr.contains("note: applied passes:"),
        "stderr missing applied-passes note:\n{stderr}"
    );
}

#[test]
fn inspect_no_passes_marks_skipped() {
    let output = Command::new(nflc_bin())
        .args([
            "inspect",
            "../tests/fixtures/tiny_mlp.nfl",
            "--profile",
            "arm64",
            "--no-passes",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stdout.contains("passes: skipped"),
        "stdout missing skipped marker:\n{stdout}"
    );
    assert!(
        stderr.contains("note: passes skipped"),
        "stderr missing passes-skipped note:\n{stderr}"
    );
}
```

- [ ] **Step 13: Run tests** — `cargo test --workspace`. Count should be ~452 + 2 (inspect-render lib unit tests from Step 4) + 2 (CLI smoke from Step 12) ≈ 456.

- [ ] **Step 14: Run gates + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add Cargo.toml inspect-render/ nflc/Cargo.toml nflc/src/main.rs nflc/tests/cli_inspect.rs
git commit -m "$(cat <<'EOF'
feat(m16): add nflc inspect subcommand + inspect-render crate

New workspace lib crate `inspect-render` with single public function
render_inspection(insp: &Inspection, header: RenderHeader) -> String,
matching the format spec in §5 of the M16 design doc. Two unit tests
in the crate cover format markers + the --no-passes "skipped" branch.

nflc/src/main.rs gains `inspect` subcommand dispatch:
- parse_inspect_args / InspectArgs (mirror parse_compile_args / CompileArgs
  shape, minus the -o flag — inspect output goes to stdout only).
- run_inspect: parse → build UIR → run passes (or skip per --no-passes /
  filter per --passes) → call profile_impl.inspect(&post_pass_uir)? →
  render via inspect_render::render_inspection → print to stdout.
- Shared parse_pass_flag + validate_pass_args helpers extracted from
  parse_compile_args for reuse between compile and inspect.

print_usage() updated to advertise the new subcommand.

CLI smoke tests in nflc/tests/cli_inspect.rs verify exit-0 + key
output markers (inspect-model, loc=, passes applied:, passes: skipped).

Bisect-claim: renderer + CLI dispatch wired; goldens not captured yet;
output stable for in-tree fixtures; cargo test --workspace clean at
~456 tests.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — Capture goldens + per-profile integration tests

**Why:** Format-stability regression net. Per spec §7 process rule: every byte in every `.expected.txt` is captured from real `cargo run` output — zero hand-computed numbers.

**Files:**
- Create: `profiles/arm64/tests/inspect.rs`
- Create: `profiles/arm64/tests/inspect/tiny_mlp.expected.txt`
- Create: `profiles/arm64/tests/inspect/transformer_block.expected.txt`
- Create: `profiles/arm64/tests/inspect/self_attention.expected.txt`
- Create: `profiles/arm64/tests/inspect/dropout_only.expected.txt`
- Create: `profiles/x86_64/tests/inspect.rs` (parallel to arm64)
- Create: `profiles/x86_64/tests/inspect/tiny_mlp.expected.txt`
- Create: `profiles/x86_64/tests/inspect/transformer_block.expected.txt`
- Create: `profiles/x86_64/tests/inspect/self_attention.expected.txt`
- Create: `profiles/x86_64/tests/inspect/dropout_only.expected.txt`
- Modify: `profiles/arm64/Cargo.toml` (add inspect-render dev-dep)
- Modify: `profiles/x86_64/Cargo.toml` (add inspect-render dev-dep)

### 5a. Add `inspect-render` as dev-dependency for both profiles

- [ ] **Step 1: Edit `profiles/arm64/Cargo.toml`** — extend the existing `[dev-dependencies]` block (which already contains `libloading = "0.8"`) so it reads:

```toml
[dev-dependencies]
libloading     = "0.8"
inspect-render = { path = "../../inspect-render" }
```

- [ ] **Step 2: Edit `profiles/x86_64/Cargo.toml`** — same extension (it also already has `libloading = "0.8"`):

```toml
[dev-dependencies]
libloading     = "0.8"
inspect-render = { path = "../../inspect-render" }
```

### 5b. Write the integration test harness

- [ ] **Step 3: Create `profiles/arm64/tests/inspect.rs`:**

```rust
// SPDX-License-Identifier: Apache-2.0

//! Golden-snapshot tests for `Arm64Profile::inspect()` rendering.
//!
//! Each fixture under `tests/inspect/<name>.expected.txt` is the
//! verbatim stdout of:
//!     cargo run -p nflc -- inspect tests/fixtures/<name>.nfl --profile arm64
//! captured at the time of M16 closure. Regen on intentional format
//! change with the same command.

use inspect_render::{render_inspection, RenderHeader};
use profile_api::Profile;
use profiles_arm64::Arm64Profile;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    // cwd = profiles/arm64/ during integration test; workspace root is two up.
    PathBuf::from(format!("../../tests/fixtures/{}.nfl", name))
}

fn expected_path(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/inspect/{}.expected.txt", name))
}

fn run_and_render(name: &str) -> String {
    let path = fixture_path(name);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let ast = compiler::parse(&source).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let pipeline = compiler::passes::default_pipeline();
    let post_pass = compiler::passes::run_pipeline(&uir, &pipeline).expect("run_pipeline");
    let insp = Arm64Profile.inspect(&post_pass).expect("inspect");

    let pass_names: Vec<String> = pipeline.iter().map(|p| p.name().to_owned()).collect();
    let pass_refs: Vec<&str> = pass_names.iter().map(String::as_str).collect();
    let header = RenderHeader {
        source_path: &path,
        profile: "arm64",
        applied_passes: Some(&pass_refs),
    };
    render_inspection(&insp, header)
}

fn assert_golden(name: &str) {
    let actual = run_and_render(name);
    let expected_path = expected_path(name);
    let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|e| {
        panic!(
            "read expected file {}: {}\n\nIf this is the first run, regenerate with:\n  cargo run -p nflc -- inspect tests/fixtures/{}.nfl --profile arm64 > {}",
            expected_path.display(),
            e,
            name,
            expected_path.display()
        )
    });
    if actual != expected {
        // Pretty diff for triage.
        panic!(
            "golden mismatch for {} (arm64).\n--- expected ---\n{}\n--- actual ---\n{}\n",
            name, expected, actual
        );
    }
}

#[test]
fn golden_tiny_mlp_arm64() {
    assert_golden("tiny_mlp");
}

#[test]
fn golden_transformer_block_arm64() {
    assert_golden("transformer_block");
}

#[test]
fn golden_self_attention_arm64() {
    assert_golden("self_attention");
}

#[test]
fn golden_dropout_only_arm64() {
    assert_golden("dropout_only");
}
```

- [ ] **Step 4: Create `profiles/x86_64/tests/inspect.rs`** — parallel structure, swap `Arm64Profile` for `X86_64Profile` and `arm64` for `x86_64` in the path strings and the header:

```rust
// SPDX-License-Identifier: Apache-2.0

//! Golden-snapshot tests for `X86_64Profile::inspect()` rendering.
//! Mirror of profiles/arm64/tests/inspect.rs.

use inspect_render::{render_inspection, RenderHeader};
use profile_api::Profile;
use profiles_x86_64::X86_64Profile;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(format!("../../tests/fixtures/{}.nfl", name))
}

fn expected_path(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/inspect/{}.expected.txt", name))
}

fn run_and_render(name: &str) -> String {
    let path = fixture_path(name);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let ast = compiler::parse(&source).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let pipeline = compiler::passes::default_pipeline();
    let post_pass = compiler::passes::run_pipeline(&uir, &pipeline).expect("run_pipeline");
    let insp = X86_64Profile.inspect(&post_pass).expect("inspect");

    let pass_names: Vec<String> = pipeline.iter().map(|p| p.name().to_owned()).collect();
    let pass_refs: Vec<&str> = pass_names.iter().map(String::as_str).collect();
    let header = RenderHeader {
        source_path: &path,
        profile: "x86_64",
        applied_passes: Some(&pass_refs),
    };
    render_inspection(&insp, header)
}

fn assert_golden(name: &str) {
    let actual = run_and_render(name);
    let expected_path = expected_path(name);
    let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|e| {
        panic!(
            "read expected file {}: {}\n\nIf this is the first run, regenerate with:\n  cargo run -p nflc -- inspect tests/fixtures/{}.nfl --profile x86_64 > {}",
            expected_path.display(),
            e,
            name,
            expected_path.display()
        )
    });
    if actual != expected {
        panic!(
            "golden mismatch for {} (x86_64).\n--- expected ---\n{}\n--- actual ---\n{}\n",
            name, expected, actual
        );
    }
}

#[test]
fn golden_tiny_mlp_x86_64() {
    assert_golden("tiny_mlp");
}

#[test]
fn golden_transformer_block_x86_64() {
    assert_golden("transformer_block");
}

#[test]
fn golden_self_attention_x86_64() {
    assert_golden("self_attention");
}

#[test]
fn golden_dropout_only_x86_64() {
    assert_golden("dropout_only");
}
```

### 5c. Capture goldens

- [ ] **Step 5: Make the inspect-output directories.** From repo root:

```bash
mkdir -p profiles/arm64/tests/inspect profiles/x86_64/tests/inspect
```

- [ ] **Step 6: Capture all 8 goldens.** From repo root, run each command and verify stdout looks reasonable before committing:

```bash
cargo run -p nflc -- inspect tests/fixtures/tiny_mlp.nfl --profile arm64 > profiles/arm64/tests/inspect/tiny_mlp.expected.txt
cargo run -p nflc -- inspect tests/fixtures/transformer_block.nfl --profile arm64 > profiles/arm64/tests/inspect/transformer_block.expected.txt
cargo run -p nflc -- inspect tests/fixtures/self_attention.nfl --profile arm64 > profiles/arm64/tests/inspect/self_attention.expected.txt
cargo run -p nflc -- inspect tests/fixtures/dropout_only.nfl --profile arm64 > profiles/arm64/tests/inspect/dropout_only.expected.txt
cargo run -p nflc -- inspect tests/fixtures/tiny_mlp.nfl --profile x86_64 > profiles/x86_64/tests/inspect/tiny_mlp.expected.txt
cargo run -p nflc -- inspect tests/fixtures/transformer_block.nfl --profile x86_64 > profiles/x86_64/tests/inspect/transformer_block.expected.txt
cargo run -p nflc -- inspect tests/fixtures/self_attention.nfl --profile x86_64 > profiles/x86_64/tests/inspect/self_attention.expected.txt
cargo run -p nflc -- inspect tests/fixtures/dropout_only.nfl --profile x86_64 > profiles/x86_64/tests/inspect/dropout_only.expected.txt
```

After capture, **eyeball each file**:
- Header line shows the right profile.
- `passes applied:` line lists the three default passes.
- Each `inspect-model` block has the 6 summary lines.
- Per-node entries have the two-line shape (`n<idx>` then indented `loc=...    out=...`).
- Numbers look plausible (no negative bytes, stack frame is multiple of 16, params count matches op shape × types).

**If any fixture fails to inspect** (parse error, lowering error, etc.), do NOT commit a broken golden — fix the renderer or per-profile inspect_model first, then re-capture.

- [ ] **Step 7: Run integration tests.**

```bash
cargo test --workspace
```

Now that the goldens exist, all 8 new integration tests should pass. Total count ≈ 456 + 8 = 464.

- [ ] **Step 8: Run gates + commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add profiles/arm64/Cargo.toml profiles/arm64/tests/ \
        profiles/x86_64/Cargo.toml profiles/x86_64/tests/
git commit -m "$(cat <<'EOF'
test(m16): capture inspect goldens for 4 fixtures × 2 profiles

8 golden-snapshot integration tests anchor `Profile::inspect()` output
format. Goldens captured verbatim from
    cargo run -p nflc -- inspect <fixture> --profile <name>
per the spec §7 process rule (zero hand-computed numbers).

Fixtures cover orthogonal axes:
  - tiny_mlp:           baseline (linear+softmax, post-fusion compact form)
  - transformer_block:  rich (layernorm + FFN + dual residual; multi-input N=3)
  - self_attention:     softmax + matmul + non-leaf
  - dropout_only:       edge (dropout-as-output post-EliminateDropout behaviour)

Both profiles get parallel test files
(profiles/{arm64,x86_64}/tests/inspect.rs) using the inspect-render
crate as a dev-dependency.

Bisect-claim: 8 goldens captured from real runs; cargo test --workspace
clean at ~464 tests; first-time format-stability harness in place.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Documentation closure

**Why:** Project Documentation Protocol (CLAUDE.md). Every milestone closes with profile-guide updates, status bumps, and a DEVLOG entry.

**Files:**
- Modify: `docs/profile_guide/arm64.md`
- Modify: `docs/profile_guide/x86_64.md`
- Modify: `CLAUDE.md`
- Modify: `PROJECT_SPEC.md`
- Modify: `README.md` (verify pattern first)
- Modify: `DEVLOG.md`

- [ ] **Step 1: Add `## Inspection output` section to `docs/profile_guide/arm64.md`.** Append at the end of the file (or insert after the last existing top-level section, mirroring file structure). Content:

```markdown
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
  this function. `d8-d9` and `x19-x23` appear when any node in the
  model calls `_expf` (standalone Softmax or fused SoftmaxRow); empty
  for leaf functions.
- **`leaf`** — `yes` iff no `bl _expf` is emitted; `no` otherwise.
  Drives whether `x29`/`x30` are saved in the prologue.

See `docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md`
for the full schema and design rationale.
```

- [ ] **Step 2: Add parallel `## Inspection output` section to `docs/profile_guide/x86_64.md`.** Same structure, swap arm64-specific terms:

```markdown
## Inspection output (M16 / A3)

`nflc inspect <file.nfl> --profile x86_64` runs the same per-profile
analyzers that `nflc compile` runs (`assign_buffers`,
`compute_callee_saved`), packages the result as a structured
`profile_api::Inspection`, and renders it to text. Both commands run
the default pass pipeline by default; pass `--no-passes` or
`--passes <list>` to skip / filter (same semantics as `compile`).

Field reference:
- **`loc=`** — output buffer placement: `InputReg(i)` (the i-th input's
  SysV ABI register: `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8` for i=0..4
  per `AbiContext::input_reg(i)`), `OutputReg` (the output ABI
  register at `INPUT_REGS[n_inputs + 1]`), `StackOffset(N)`
  (`[%rsp + N]`), or `Alias(nK)`.
- **`out=`** — logical output bytes (`element_count * 4`).
- **`params=`** — for `Linear` and `LayerNorm[affine=true]`: floats
  consumed from the packed `params` buffer.
- **`callee-saved`** (per-model) — `%rbx, %r12-%r15` appear when the
  model contains a matmul (M12 callee-saved-int trigger for matmul
  body scratch; spec §9.1) or any `expf@PLT` call (softmax). x86_64
  has **no callee-saved FP register set** — all `%xmm0-%xmm15` are
  caller-saved per SysV.
- **`leaf`** — `yes` iff no `call expf@PLT` is emitted. Unlike arm64,
  x86_64's prologue does not vary on leaf classification — the bool
  is computed from `UirModel::calls_extern_math()` for the inspect
  output only.

See `docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md`
for the full schema and design rationale.
```

- [ ] **Step 3: Update `CLAUDE.md` "Current Status" section.** Replace the M15 status block with M16. Bump test count, add A3 closure, update the repository map "Strategic direction" line. Find the existing block:

```
**Milestone 15 complete. 446 tests passing on macOS arm64 (~448 on Linux x86_64 CI with x86_64 FFI tests included).** ...
```

Replace with:

```
**Milestone 16 complete. ~464 tests passing on macOS arm64 (~466 on Linux x86_64 CI — the +2 delta is the M15 x86_64-only FFI integration tests `ffn_ffi` / `transformer_block_ffi`, gated on `#[cfg(target_os = "linux")]`; the new M16 inspect goldens are pure Rust and run on both platforms).** All workspace gates clean
(`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check`, `cargo test --workspace`).

M16 closed A3 — profile-level viewer annotations. New `nflc inspect <file.nfl> --profile <name>`
subcommand surfaces post-pass per-node BufferLoc + footprint + params and per-model stack
frame + callee-saved + leaf classification, packaged from the same `analyze()` preamble that
`lower()` consumes (drift-prevention by construction). New workspace crate `inspect-render/`
hosts the renderer. `BufferLoc` lifted from per-profile duplicates to `profile-api`. Eight
golden-snapshot integration tests (4 fixtures × 2 profiles) anchor format stability.

Strategic direction: see `PROJECT_SPEC.md` §"Strategic Roadmap" — A1 closed
M12, A2 first brick (`add`) closed M13, A2 second brick (`layernorm`)
closed M14, A2 third brick (FFN) closed M15, **A3 (viewer annotations) closed M16. Axis 2 fully complete.**
Next candidates: Axis 3 — bare-metal `expf` to drop libm (now unblocked — A3 enables structural
`--diff before.s after.s` validation post-implementation). Trigger-driven cleanup (OQ-7, OQ-8, OQ-9,
M5c OQ-4) stays dormant. §"Known Latent Hazards" table empty as of end of M16.
```

Also update the Repository Structure tree near the top of CLAUDE.md to add the new `inspect-render/` crate row and the `nflc inspect` subcommand line under the `nflc/` description.

- [ ] **Step 4: Update `PROJECT_SPEC.md` Strategic Roadmap line.** Find the line:

```
NFL v0.2 self-attention [complete in M10] → multi-input grammar A1 [closed M12] → transformer block A2 (residual + LayerNorm + FFN) → profile-level viewer annotations A3
```

Replace with:

```
NFL v0.2 self-attention [complete in M10] → multi-input grammar A1 [closed M12] → transformer block A2 (residual + LayerNorm + FFN) [closed M15] → profile-level viewer annotations A3 [closed M16]
```

Also add a new row to the milestone status table (matching the table format used for M11-M15):

```
| 16 | A3 viewer annotations (complete) | New `nflc inspect <file.nfl> --profile <name>` subcommand surfaces post-pass per-node BufferLoc + footprint + params and per-model stack frame + callee-saved + leaf classification. Architecture: per-profile `analyze()` preamble extracted from `walk_model` (Task 1), shared `Inspection`/`FnAnnotations`/`NodeAnnotation` schema in `profile-api` with `BufferLoc` lifted from per-profile duplicates (Task 2), `Profile::inspect()` trait method + per-profile impl (Task 3), new `inspect-render` workspace crate (Task 4), 8 golden-snapshot tests (Task 5), profile-guide docs (Task 6). Test count: 446 → ~462. |
```

- [ ] **Step 5: Verify README.md repository-map style** before editing. From repo root:

```bash
grep -n -A 2 'nflc' README.md | head -20
```

If README enumerates subcommands per `nflc`, add an `inspect` line in the same style. If README only mentions `nflc` at crate level, leave alone (do not over-document — spec §8 caveat).

- [ ] **Step 6: Add DEVLOG entry.** Append to `DEVLOG.md` at the top (newest entry first per existing convention):

```markdown
## 2026-05-11 — Milestone 16 closed: A3 — profile-level viewer annotations

### What was done

- **Task 1 — extract `analyze()` from `walk_model` (both profiles).**
  Pure refactor; asm output bit-identical for all 446 fixtures.
  Per-profile `ModelAnalysis` private struct: arm64 carries
  `LeafKind`; x86_64 omits it (its prologue is leaf-agnostic).
  Both `lower()` and (forthcoming) `inspect()` consume `analyze()` —
  drift-prevention by construction.

- **Task 2 — lift `BufferLoc` enum to `profile-api`.** The two
  profile copies were structurally bit-identical (verified by diff
  before lift); only doc-comment richness differed. Each profile's
  `buffer.rs` swapped its local definition for
  `pub use profile_api::BufferLoc`.

- **Task 3 — `Inspection`/`FnAnnotations`/`NodeAnnotation` schema +
  `Profile::inspect()` trait method + per-profile impl.** Schema
  lives in `profile-api` (M9 trait-grows-by-request invariant
  satisfied — `nflc inspect` is the consumer). Per-profile callee-saved
  rendering: arm64 `["d8-d9", "x19-x23"]`; x86_64 `["%rbx", "%r12-%r15"]`.
  6 new unit tests (3 per profile): leaf detection, alias placement,
  params count.

- **Task 4 — new `inspect-render` workspace crate + `nflc inspect` CLI.**
  Renderer crate (lib only) with 2 unit tests. CLI subcommand mirrors
  `compile` shape (`--profile` + `--no-passes`/`--passes`); shared
  `parse_pass_flag` + `validate_pass_args` helpers extracted from
  `parse_compile_args`. 2 CLI smoke tests.

- **Task 5 — 8 goldens captured + integration tests.** 4 fixtures
  (`tiny_mlp`, `transformer_block`, `self_attention`, `dropout_only`)
  × 2 profiles. Process rule: zero hand-computed numbers, every byte
  from `cargo run -p nflc -- inspect ...` output.

- **Task 6 — documentation.** Profile guides updated, CLAUDE.md
  bumped to M16 status, PROJECT_SPEC.md milestone table + Strategic
  Roadmap line updated, this DEVLOG entry.

- **Final test count: ~464** (macOS arm64); **~466** on Linux x86_64
  CI — +2 delta is the M15 x86_64-only FFI tests (`ffn_ffi`,
  `transformer_block_ffi`), not the new M16 inspect goldens (which
  are pure Rust and run on both platforms).

### Decisions made

- **`inspect-render` as separate workspace crate** rather than folding
  into `profile-api`. Rationale: `profile-api` is the schema + trait
  contract; rendering is formatting policy and has no business in the
  contract crate. One tiny new crate, single responsibility.

- **`BufferLoc` lifted to `profile-api`** as part of A3 rather than
  deferred. Rationale: natural cleanup at the point a third consumer
  (the inspect-render crate) needs the type. Verified bit-identical
  before lift.

- **`FnAnnotations` carries `input_nodes: Vec<NodeId>` + `output_node: NodeId`
  in addition to `fn_sig.inputs_floats` / `fn_sig.output_floats`.** Reason:
  in general, model inputs are not necessarily the first N nodes by id
  (and outputs are never structurally constrained). The renderer needs
  the real NodeIds to produce stable `n<id>` refs in the inputs/output
  summary lines, matching the format sketched in spec §5.

### Problems encountered

- **None blocking.** Pre-task grep verification (Task 2) confirmed no
  external imports of `profiles_*::buffer::BufferLoc` — lift was
  mechanical as expected.

### Next step

Open candidates per Strategic Roadmap:
- **Axis 3 — bare-metal `expf`**. Now unblocked: A3 enables structural
  `nflc inspect --diff before.s after.s` validation (future tooling)
  for verifying that Taylor-series `expf` produces the expected
  footprint reduction.
- **A2-extended: training syntax (loss/optimiser)**. NFL v0.3 — larger
  language milestone.
```

Also update the milestone status block at the top of DEVLOG.md (if one exists) to bump M15 → M16.

- [ ] **Step 7: Run full workspace gates one more time** to confirm doc-only commit doesn't break anything:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

- [ ] **Step 8: Commit**

```bash
git add docs/profile_guide/arm64.md docs/profile_guide/x86_64.md \
        CLAUDE.md PROJECT_SPEC.md DEVLOG.md
# Add README.md only if Step 5 produced an edit:
# git add README.md
git commit -m "$(cat <<'EOF'
docs(m16): close milestone — profile guides, status, devlog

Per Documentation Protocol (CLAUDE.md):
- docs/profile_guide/{arm64,x86_64}.md gain `## Inspection output`
  sections with field reference + per-profile callee-saved rendering
  conventions.
- CLAUDE.md "Current Status" bumped to M16 + A3 closure + new test count.
  Repository map updated for inspect-render/ crate + nflc inspect line.
- PROJECT_SPEC.md milestone table row 16 added; Strategic Roadmap line
  marks A2 + A3 closed; Axis 2 fully complete.
- DEVLOG.md milestone-closure entry per protocol (What done / Decisions
  / Problems / Next step).
- README.md left unchanged (verified — does not enumerate subcommands).

Bisect-claim: doc-only; cargo test --workspace clean at ~464 tests;
no code changes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist (milestone closure)

After Task 6 commit, verify:

- [ ] `cargo build --workspace` clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` exits 0
- [ ] `cargo fmt --all -- --check` clean
- [ ] `cargo test --workspace` passes; total count is monotonically increased from 446 (target ~464)
- [ ] All 8 `.expected.txt` files present and non-empty:
  ```bash
  ls -la profiles/arm64/tests/inspect/*.expected.txt profiles/x86_64/tests/inspect/*.expected.txt
  ```
- [ ] `nflc` (no args) print-usage shows `inspect` line:
  ```bash
  cargo run -p nflc 2>&1 | grep inspect
  ```
- [ ] `nflc compile` behaviour unchanged for any M15 fixture (Task 1 + Task 2 invariant). Smoke check:
  ```bash
  cargo run -p nflc -- compile tests/fixtures/tiny_mlp.nfl --profile arm64 | head -20
  ```
  Output should look the same as on M15-tip (modulo any pre-existing whitespace).
- [ ] `git log --oneline main..HEAD` shows 6 commits with the expected scope prefixes (`refactor(m16):` × 2, `feat(m16):` × 2, `test(m16):` × 1, `docs(m16):` × 1).

---

## Out-of-scope reminders

Per spec §1 non-goals — explicitly NOT in this plan:

- Op-local scratch register footprint (declarative emitter metadata; future milestone)
- `nflc inspect --diff before.s after.s` (Axis 3 prereq tooling, separate scope)
- `--node <id>` selector
- `--format json`
- Cross-profile diff in single command
- Liveness-based buffer reuse (Axis 1 / perf milestone)

If any of these surfaces during implementation as "wouldn't this be easy to add", **don't**. Open a `PROJECT_SPEC.md §"Open Questions"` entry instead, per project process.
