# M4b — `profiles/arm64` Op Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `profiles/arm64` to lower all 5 M3 positive fixtures end-to-end. Adds `linear[N, bias=true]`, `dropout` (no-op aliasing), and `softmax` (3-pass numerically stable, libm `expf`). Re-architects `FnSig` around a single packed `params` buffer with typed slot metadata; introduces stack-allocated intermediate buffers; adds non-leaf prologue/epilogue with callee-saved register analysis; moves duplicate-model-name check from `profiles/arm64::walk_uir` up to `compiler::ir::build`.

**Architecture:** Three layers in dependency order. (1) ABI redesign — `FnSig` carries `params_layout: Vec<ParamSlot>` instead of `weight_floats`. (2) Buffer-assignment + leaf-analysis infrastructure in `profiles/arm64/src/buffer.rs`; conditional prologue/epilogue helpers in `asm.rs`. (3) New op emitters in `profiles/arm64/src/ops/{linear, relu, softmax, dropout}.rs` (refactor of M4a code + new ops). Six integration tests exercise all 5 M3 fixtures + the renamed M4a fixture.

**Tech Stack:** Rust 2021 (std-only for production crates; `libloading` 0.8 dev-dep already in place from M4a). AArch64 Mach-O assembly, AAPCS64 ABI, `bl _expf` from libm (linked by default by `cc`).

**Source spec:** [`docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md`](../specs/2026-05-04-m4b-arm64-coverage-design.md). All architectural decisions and rationale live there. **If this plan disagrees with the spec, the spec wins.**

**Working directory:** `/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/m4b-arm64-coverage` (branch `claude/m4b-arm64-coverage`, base `main` at commit `fc55419`).

**Project conventions** (`CLAUDE.md` + spec §13):
- `cargo fmt --all` before every commit (CI gates on `--check`).
- `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
- `cargo test --workspace` passes; test count goes up monotonically.
- Production crates strictly std-only.
- TDD: failing test first, minimal impl, verify pass, commit.

**Pre-task baseline:**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "BASELINE:", sum}'
# Expect: 118.
```

---

## File Structure

### Created

| Path | Responsibility | Created in task |
|---|---|---|
| `profiles/arm64/src/buffer.rs` | `BufferLoc`, `BufferAssignment`, `assign_buffers`, `compute_is_leaf`, `compute_callee_saved`, `RegSet` | Task 2 |
| `profiles/arm64/src/ops/mod.rs` | Submodule entry; re-exports `emit_*` | Task 5 |
| `profiles/arm64/src/ops/linear.rs` | `emit_linear` — matmul (Task 5) + bias-add (Task 7) | Task 5, extended Task 7 |
| `profiles/arm64/src/ops/relu.rs` | `emit_relu` — moved from M4a's `codegen.rs` | Task 5 |
| `profiles/arm64/src/ops/softmax.rs` | `emit_softmax` — 3-pass with `bl _expf` | Task 8 |
| `profiles/arm64/src/ops/dropout.rs` | Marker module — dropout has no emitter (aliasing only) | Task 6 |

### Modified

| Path | What changes | Tasks |
|---|---|---|
| `profiles/arm64/src/types.rs` | Drop `weight_floats`; add `params_floats`, `params_layout`, `ParamSlot`, `ParamKind`. Drop `LowerError::DuplicateModelName` | 1, 4 |
| `profiles/arm64/src/codegen.rs` | Layout computation (1); wire buffer.rs (3); ops/* dispatch (5); skip dropout (6); pass bias info (7); dispatch softmax (8) | 1, 3, 5, 6, 7, 8 |
| `profiles/arm64/src/asm.rs` | Replace `format_function_header`/`_footer` with `format_function_prologue`/`_epilogue` | Task 3 |
| `profiles/arm64/src/lib.rs` | `mod buffer;` (2), `mod ops;` (5), `pub use` updates (1, 5) | 1, 2, 5 |
| `profiles/arm64/src/tests.rs` | M4a tests adapted to new ABI (1); 10 new unit tests across 2-8 | 1-8 |
| `profiles/arm64/tests/integration.rs` | Rename M4a integration test under new ABI (1); 5 new + 2 reference-validation (9) | 1, 9 |
| `compiler/src/ir/build.rs` | Duplicate-model-name check after model-build loop | Task 4 |
| `compiler/src/ir/error.rs` | New `BuildErrorKind::DuplicateModelName { name, first_span }` | Task 4 |
| `compiler/src/ir/tests.rs` | New `duplicate_model_name_at_build_time` test | Task 4 |
| `nflc/src/main.rs` | `render_error_with_snippet` accepts optional `first_span` | Task 4 |
| `docs/profile_guide/arm64.md` | Extend per spec §11.5 | Task 10 |
| `docs/language_reference/uir.md` | One-line dropout-as-noop note | Task 11 |
| `PROJECT_SPEC.md` | M4 row → "4a + 4b complete" | Task 11 |
| `CLAUDE.md` | Current Status; repo structure adds `ops/`, `buffer.rs` | Task 12 |
| `DEVLOG.md` | M4b closeout with explicit ABI-break note | Task 12 |

### Deleted

Nothing.

---

## Verification approach

| Check | When | How |
|---|---|---|
| Build clean | Every task | `cargo build --workspace` |
| Fmt clean | Every task before commit | `cargo fmt --all` then verify `cargo fmt --all -- --check` exits 0 |
| Clippy clean | Tasks 1, 3, 4, 7, 8, 9, 12 | `cargo clippy --workspace --all-targets -- -D warnings` |
| Tests pass | Every task | `cargo test --workspace` |
| Integration tests | Tasks 1, 9, 12 | `cargo test -p profiles-arm64 --test integration` |
| CLI smoke (5 M3 + M4a fixtures) | Task 12 | `cargo run --bin nflc -- compile <fixture> --profile arm64 -o /tmp/out.s` succeeds; `cc -shared -arch arm64 -o /tmp/out.dylib /tmp/out.s` succeeds |

---

## Task list

| # | Task | Mode | Commits |
|---|---|---|---|
| 1 | ABI refactor: `FnSig` redesign + layout computation + M4a test rewrite | SUBAGENT | 1 |
| 2 | `buffer.rs` analyzers | SUBAGENT | 1 |
| 3 | Wire `buffer.rs` into `walk_model` + new prologue/epilogue helpers | SUBAGENT | 1 |
| 4 | Move dup-name check to `compiler::ir::build` + extend snippet renderer | SUBAGENT | 1 |
| 5 | Refactor `codegen.rs` body emission into `ops/{mod, linear, relu}.rs` | SUBAGENT | 1 |
| 6 | Dropout aliasing | INLINE (trivial) | 1 |
| 7 | `linear[N, bias=true]` bias-add inline | SUBAGENT | 1 |
| 8 | `softmax` — 3-pass + `bl _expf` + `-inf` materialisation | SUBAGENT | 1 |
| 9 | Integration tests: 5 M3 fixtures + M4a fixture + 2 reference-validation tests | SUBAGENT | 1 |
| 10 | `docs/profile_guide/arm64.md` extension | SUBAGENT (prose) | 1 |
| 11 | `uir.md` cross-link + `PROJECT_SPEC.md` update | INLINE | 1 |
| 12 | Closeout — DEVLOG (with ABI-break note) + CLAUDE.md + final smoke | INLINE | 1 |

**Total:** 12 tasks, 12 commits. Targets baseline 118 → ~134 tests at end.

---

## Task 1: ABI refactor — `FnSig` redesign + M4a test rewrite

**Goal:** `FnSig` carries `params_layout`; layout computed in `walk_model`; M4a integration test runs under new ABI. Existing M4a behaviour preserved (asm content unchanged for M4a fixture; only the metadata field names change).

**Files:**
- Modify: `profiles/arm64/src/types.rs`, `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/lib.rs`, `profiles/arm64/src/tests.rs`, `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Replace `FnSig` and add `ParamSlot`/`ParamKind` in `profiles/arm64/src/types.rs`**

Find the existing `FnSig` struct and replace it (and append the new types after):

```rust
/// ABI metadata for one generated function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnSig {
    pub name: String,
    pub model: String,
    pub input_floats: usize,
    pub output_floats: usize,
    pub params_floats: usize,
    pub params_layout: Vec<ParamSlot>,
}

/// One slot within the packed `params` buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSlot {
    pub kind: ParamKind,
    pub origin_node: compiler::NodeId,
    pub offset: usize,
    pub size: usize,
}

/// Type tag for a `ParamSlot`. `#[non_exhaustive]` per spec §5.2.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    LinearWeight,
    LinearBias,
}
```

- [ ] **Step 2: Update `pub use` in `profiles/arm64/src/lib.rs`**

```rust
pub use types::{Asm, FnSig, LowerError, ParamKind, ParamSlot};
```

- [ ] **Step 3: Compute `params_layout` in `walk_model` in `profiles/arm64/src/codegen.rs`**

Find the section where `FnSig` is constructed (currently uses `weight_floats`). Replace the layout-computation block with:

```rust
    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op: StdOp::Linear, operands, attrs } = &node.kind {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete { span: node.source_span });
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

            // Bias-slot pre-allocation: detection runs now, codegen lands in Task 7.
            let has_bias = attrs.iter().any(|a| {
                a.name == "bias"
                    && matches!(&a.value, compiler::AttrValue::Symbol(s) if s == "true")
            });
            if has_bias {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        output_floats,
        params_floats,
        params_layout,
    };
```

(Add the `ParamKind` and `ParamSlot` imports at the top of `codegen.rs`: `use crate::types::{ParamKind, ParamSlot};` if not already pulled in by the module's `use` group.)

- [ ] **Step 4: Update M4a unit test in `profiles/arm64/src/tests.rs`**

Replace the body of `linear_emits_function_with_correct_symbol_and_ret`:

```rust
#[test]
fn linear_emits_function_with_correct_symbol_and_ret() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");

    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M");
    assert_eq!(sig.model, "M");
    assert_eq!(sig.input_floats, 6);
    assert_eq!(sig.params_floats, 6);
    assert_eq!(sig.output_floats, 4);

    assert_eq!(sig.params_layout.len(), 1);
    let slot = &sig.params_layout[0];
    assert_eq!(slot.kind, ParamKind::LinearWeight);
    assert_eq!(slot.offset, 0);
    assert_eq!(slot.size, 6);
    assert_eq!(slot.origin_node, 1);

    let s = &asm.source;
    assert!(s.contains(".globl _nfl_forward_M"));
    assert!(s.contains("_nfl_forward_M:"));
    assert!(s.contains("ret"));
}
```

(Other M4a tests don't touch `FnSig` fields and stay unchanged.)

- [ ] **Step 5: Update integration test in `profiles/arm64/tests/integration.rs`**

Rename `tinymlp_no_softmax_runs_correctly` → `m4a_no_softmax_still_runs` and update to assert against new `FnSig` shape; rename `weights` parameter name to `params` (function body unchanged):

```rust
#[test]
fn m4a_no_softmax_still_runs() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: integration test requires aarch64 host");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/m4_linear_relu.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");

    let asm = profiles_arm64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.input_floats, 32);
    assert_eq!(sig.params_floats, 8);
    assert_eq!(sig.output_floats, 16);

    let dylib_path = common::compile_to_dylib(&asm.source, "m4a_linear_relu");

    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_M4Demo") }.expect("dlsym");

    let mut input = [0.0f32; 32];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5;
    }
    let mut params = [0.0f32; 8];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16];
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

    let expected = reference_linear_relu(&input, &params);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: got {a}, expected {b}"
        );
    }
}

fn reference_linear_relu(input: &[f32; 32], params: &[f32; 8]) -> [f32; 16] {
    const B: usize = 8;
    const K: usize = 4;
    const N: usize = 2;
    let mut out = [0.0f32; 16];
    for i in 0..B {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += input[i * K + k] * params[k * N + j];
            }
            out[i * N + j] = sum.max(0.0);
        }
    }
    out
}
```

- [ ] **Step 6: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: zero warnings, exit 0, 118 tests pass.

- [ ] **Step 7: Commit**

```bash
git add profiles/
git commit -m "feat(m4b/abi): redesign FnSig around packed params + ParamSlot layout

Per spec §5: deliberate ABI break vs M4a. FnSig.weight_floats removed;
replaced by params_floats + params_layout: Vec<ParamSlot>. Each
ParamSlot has kind (LinearWeight | LinearBias), origin_node, offset,
size. Slots emitted in topological UIR-node order.

For M4a-compatible models (single Linear, no bias), params_layout
has one LinearWeight slot — same float content as the old weights
buffer, just renamed in the ABI.

walk_model computes layout in a first pass; bias-slot detection is
included now even though codegen for bias-add lands in Task 7
(isolation). emit_matmul still uses old offset arithmetic; Task 7
extends it for bias.

ParamKind is #[non_exhaustive] — M5+ adds NormGamma, EmbeddingTable,
etc. without breaking downstream match consumers.

M4a integration test renamed tinymlp_no_softmax_runs_correctly →
m4a_no_softmax_still_runs and updated to call forward(input, params,
output) — same buffer contents, renamed parameter.

Baseline 118 tests preserved.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: `buffer.rs` analyzers

**Goal:** Standalone module defining buffer-assignment + leaf/callee-saved analyzers. Pure data structures + functions; no integration with codegen yet (Task 3 wires). Unit tests cover analyzer behaviour directly.

**Files:**
- Create: `profiles/arm64/src/buffer.rs`
- Modify: `profiles/arm64/src/lib.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add failing tests in `profiles/arm64/src/tests.rs`**

Append:

```rust
use super::buffer::{
    assign_buffers, compute_callee_saved, compute_is_leaf, BufferLoc,
};

#[test]
fn assign_buffers_input_node_is_input_reg() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[0], BufferLoc::InputReg));
}

#[test]
fn assign_buffers_terminal_node_is_output_reg() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    let last = assignment.locs.last().unwrap();
    assert!(matches!(last, BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_relu_aliases_operand() {
    // input → linear → relu (terminal-relu)
    // n0 input, n1 linear (non-terminal), n2 relu (terminal)
    // Expected: n2 → OutputReg (terminal wins over alias rule); n1 → StackOffset
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_intermediate_relu_aliases_operand() {
    // input → linear → relu → linear → relu (terminal). Intermediate relu (n2)
    // aliases linear (n1). The terminal relu (n4) is OutputReg.
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::Alias(1)));
    assert!(matches!(assignment.locs[3], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[4], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_stack_bytes_is_aligned() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(assignment.stack_bytes > 0);
    assert_eq!(assignment.stack_bytes % 16, 0, "stack must be 16-aligned");
}

#[test]
fn compute_is_leaf_true_for_no_softmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    assert!(compute_is_leaf(&uir.models[0]));
}

#[test]
fn compute_is_leaf_false_when_softmax_present() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n");
    assert!(!compute_is_leaf(&uir.models[0]));
}

#[test]
fn compute_callee_saved_includes_d8_d9_when_softmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(regs.contains_d8_d9());
}

#[test]
fn compute_callee_saved_empty_for_leaf() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(!regs.contains_d8_d9());
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 buffer 2>&1 | tail -10
```

Expected: compile errors — module `buffer` doesn't exist.

- [ ] **Step 3: Create `profiles/arm64/src/buffer.rs`**

```rust
//! Buffer assignment + leaf/callee-saved analyzers for the arm64 codegen.
//!
//! Pure analyzers over `UirModel`. No asm emission. Consumed by `codegen.rs`
//! in Task 3.

use compiler::{NodeId, NodeKind, StdOp, UirModel};

/// Where an Op-node's output buffer lives at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    InputReg,
    OutputReg,
    StackOffset(usize),
    Alias(NodeId),
}

/// Result of buffer assignment.
pub struct BufferAssignment {
    pub locs: Vec<BufferLoc>,
    /// Total stack bytes required, rounded up to 16-byte alignment.
    pub stack_bytes: usize,
}

/// Assign a `BufferLoc` per UIR node + compute aligned total stack frame size.
pub fn assign_buffers(model: &UirModel) -> BufferAssignment {
    let mut locs = vec![BufferLoc::InputReg; model.nodes.len()];
    let mut stack_offset: usize = 0;

    for (id, node) in model.nodes.iter().enumerate() {
        locs[id] = match &node.kind {
            NodeKind::Input { .. } => BufferLoc::InputReg,
            NodeKind::Op { op, operands, .. } => {
                if id == model.output {
                    BufferLoc::OutputReg
                } else {
                    match op {
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
                        StdOp::Linear | StdOp::Softmax => {
                            let size_bytes =
                                node.ty.shape.0.iter().product::<u64>() as usize * 4;
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset += size_bytes;
                            loc
                        }
                    }
                }
            }
        };
    }

    let stack_bytes = (stack_offset + 15) & !15;
    BufferAssignment { locs, stack_bytes }
}

/// True iff the model emits no `bl`/`blr` (i.e. no softmax in M4b).
pub fn compute_is_leaf(model: &UirModel) -> bool {
    !model
        .nodes
        .iter()
        .any(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Softmax, .. }))
}

/// Set of callee-saved registers used by the model's body. M4b: `{d8, d9}`
/// iff softmax is present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegSet {
    pub d8_d9: bool,
}

impl RegSet {
    pub fn contains_d8_d9(&self) -> bool {
        self.d8_d9
    }
}

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    RegSet {
        d8_d9: model
            .nodes
            .iter()
            .any(|n| matches!(&n.kind, NodeKind::Op { op: StdOp::Softmax, .. })),
    }
}
```

- [ ] **Step 4: Add `mod buffer;` in `profiles/arm64/src/lib.rs`**

After existing `mod` declarations (e.g., after `mod codegen;`):

```rust
mod buffer;
```

(Internal visibility is sufficient; tests use `super::buffer::*`.)

- [ ] **Step 5: Build + fmt + test**

```bash
cargo fmt --all
cargo build --workspace
cargo test -p profiles-arm64 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "PROFILES TOTAL:", sum}'
```

Expected: zero warnings; profiles/arm64 unit test count goes from M4a's ~10 to ~19.

- [ ] **Step 6: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4b/buffer): buffer-assignment + leaf/callee-saved analyzers

Per spec §6 + §7: pure analyzers over UirModel. No codegen
integration yet (Task 3 wires walk_model to use these).

- BufferLoc enum: InputReg | OutputReg | StackOffset(bytes) | Alias(NodeId).
- BufferAssignment { locs, stack_bytes }: stack_bytes already
  rounded to 16-byte alignment per AAPCS64.
- assign_buffers walks nodes in topological order. Terminal node →
  OutputReg. Relu/Dropout → Alias(operand). Linear/Softmax (non-terminal)
  → StackOffset growing offset. Input → InputReg.
- compute_is_leaf: false iff any Softmax.
- compute_callee_saved → RegSet { d8_d9: bool }: d8_d9 set iff Softmax.

9 new unit tests cover input/terminal/aliasing/stack-aligned/leaf/
callee-saved cases. Tests verify analyzer behaviour without touching
asm output.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Wire `buffer.rs` into `walk_model` + new prologue/epilogue helpers

**Goal:** `walk_model` uses `assign_buffers`, `compute_is_leaf`, `compute_callee_saved` to drive prologue/epilogue + per-op buffer-pointer computation. Asm output for M4a fixture changes shape (relu now does load-from-source/store-to-output rather than in-place); existing M4a tests are updated.

**Files:**
- Modify: `profiles/arm64/src/asm.rs`, `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Replace asm helpers in `profiles/arm64/src/asm.rs`**

Replace the entire file contents:

```rust
//! Low-level AArch64 assembly building blocks.

use crate::buffer::RegSet;
use crate::FnSig;

pub const MACHO_SYM_PREFIX: &str = "_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafKind {
    Leaf,
    NonLeaf,
}

/// Format the function header (.globl + alignment + label) plus prologue.
///
/// Per spec §7.2:
/// - callee-saved FP regs (d8/d9) saved first if `regs.contains_d8_d9()`.
/// - x29/x30 saved iff non-leaf, and frame pointer set.
/// - sp adjusted for intermediate buffers (multiple of 16, may need
///   movz/movk + sub for sizes that don't fit in a 12-bit immediate).
pub fn format_function_prologue(
    sig: &FnSig,
    leaf: LeafKind,
    regs: RegSet,
    intermediate_bytes: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(".globl {}{}\n", MACHO_SYM_PREFIX, sig.name));
    s.push_str(".p2align 2\n");
    s.push_str(&format!("{}{}:\n", MACHO_SYM_PREFIX, sig.name));

    if regs.contains_d8_d9() {
        s.push_str("    stp     d8, d9, [sp, #-16]!\n");
    }
    if leaf == LeafKind::NonLeaf {
        s.push_str("    stp     x29, x30, [sp, #-16]!\n");
        s.push_str("    mov     x29, sp\n");
    }
    if intermediate_bytes > 0 {
        s.push_str(&emit_sp_sub(intermediate_bytes));
    }
    s
}

/// Symmetric epilogue.
pub fn format_function_epilogue(
    leaf: LeafKind,
    regs: RegSet,
    intermediate_bytes: usize,
) -> String {
    let mut s = String::new();
    if intermediate_bytes > 0 {
        s.push_str(&emit_sp_add(intermediate_bytes));
    }
    if leaf == LeafKind::NonLeaf {
        s.push_str("    ldp     x29, x30, [sp], #16\n");
    }
    if regs.contains_d8_d9() {
        s.push_str("    ldp     d8, d9, [sp], #16\n");
    }
    s.push_str("    ret\n");
    s
}

/// Emit `sub sp, sp, #N` correctly for any 16-aligned N.
///
/// `sub` immediate is 12-bit (0..4095) optionally shifted by 12 (0..16,773,120
/// in steps of 4096). For sizes that don't fit, materialise N into x9 first.
pub fn emit_sp_sub(n_bytes: usize) -> String {
    if n_bytes <= 4095 {
        format!("    sub     sp, sp, #{}\n", n_bytes)
    } else if n_bytes <= 16_773_120 && n_bytes % 4096 == 0 {
        format!("    sub     sp, sp, #{}, lsl #12\n", n_bytes / 4096)
    } else {
        let lo = (n_bytes & 0xFFFF) as u16;
        let hi = ((n_bytes >> 16) & 0xFFFF) as u16;
        let mut s = String::new();
        s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo));
        if hi != 0 {
            s.push_str(&format!("    movk    w9, #0x{:04x}, lsl #16\n", hi));
        }
        s.push_str("    sub     sp, sp, x9\n");
        s
    }
}

/// Symmetric `add sp, sp, #N`.
pub fn emit_sp_add(n_bytes: usize) -> String {
    if n_bytes <= 4095 {
        format!("    add     sp, sp, #{}\n", n_bytes)
    } else if n_bytes <= 16_773_120 && n_bytes % 4096 == 0 {
        format!("    add     sp, sp, #{}, lsl #12\n", n_bytes / 4096)
    } else {
        let lo = (n_bytes & 0xFFFF) as u16;
        let hi = ((n_bytes >> 16) & 0xFFFF) as u16;
        let mut s = String::new();
        s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo));
        if hi != 0 {
            s.push_str(&format!("    movk    w9, #0x{:04x}, lsl #16\n", hi));
        }
        s.push_str("    add     sp, sp, x9\n");
        s
    }
}
```

- [ ] **Step 2: Wire prologue/epilogue + buffer pointers in `profiles/arm64/src/codegen.rs`**

Replace `walk_model` with the buffer-aware version. Find the current `fn walk_model(model: &UirModel) -> Result<(String, FnSig), LowerError>` and replace its body (everything from prologue emission through epilogue emission). Keep the layout-computation block from Task 1 intact at the top:

```rust
fn walk_model(model: &UirModel) -> Result<(String, FnSig), LowerError> {
    use crate::asm::{format_function_epilogue, format_function_prologue, LeafKind};
    use crate::buffer::{assign_buffers, compute_callee_saved, compute_is_leaf, BufferLoc};

    // 1. Validate ops upfront.
    for node in &model.nodes {
        if let NodeKind::Op { op, attrs, .. } = &node.kind {
            classify_op(*op, attrs, node.source_span)?;
        }
    }

    // 2. Compute layout, ABI sizes (kept from Task 1).
    let input_id = *model.inputs.first().ok_or_else(|| LowerError::ShapeNotConcrete {
        span: model.source_span,
    })?;
    let input_floats: usize =
        model.nodes[input_id].ty.shape.0.iter().product::<u64>() as usize;
    let output_floats: usize =
        model.nodes[model.output].ty.shape.0.iter().product::<u64>() as usize;

    let mut params_layout: Vec<ParamSlot> = Vec::new();
    let mut params_floats: usize = 0;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op: StdOp::Linear, operands, attrs } = &node.kind {
            let in_shape = &model.nodes[operands[0]].ty.shape;
            let out_shape = &node.ty.shape;
            if in_shape.0.len() != 2 || out_shape.0.len() != 2 {
                return Err(LowerError::ShapeNotConcrete { span: node.source_span });
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
            let has_bias = attrs.iter().any(|a| {
                a.name == "bias"
                    && matches!(&a.value, compiler::AttrValue::Symbol(s) if s == "true")
            });
            if has_bias {
                params_layout.push(ParamSlot {
                    kind: ParamKind::LinearBias,
                    origin_node: node_idx,
                    offset: params_floats,
                    size: n,
                });
                params_floats += n;
            }
        }
    }

    let sig = FnSig {
        name: format!("nfl_forward_{}", model.name),
        model: model.name.clone(),
        input_floats,
        output_floats,
        params_floats,
        params_layout,
    };

    // 3. Buffer assignment + leaf analysis.
    let assignment = assign_buffers(model);
    let leaf = if compute_is_leaf(model) { LeafKind::Leaf } else { LeafKind::NonLeaf };
    let regs = compute_callee_saved(model);

    // 4. Emit prologue + body + epilogue.
    let mut body = String::new();
    body.push_str(&format_function_prologue(&sig, leaf, regs, assignment.stack_bytes));

    // Per-op emission (Tasks 5-8 refactor this dispatch into ops/*).
    let mut linear_idx = 0usize;
    let mut relu_idx = 0usize;
    for (node_idx, node) in model.nodes.iter().enumerate() {
        if let NodeKind::Op { op, operands, .. } = &node.kind {
            match op {
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];

                    // Resolve buffer pointers via assignment.
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    let weight_offset = sig.params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    body.push_str(&emit_matmul_with_locs(
                        b, k, n, linear_idx, src_loc, dst_loc, weight_offset,
                    ));
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    body.push_str(&emit_relu_with_locs(total, relu_idx, src_loc, dst_loc));
                    relu_idx += 1;
                }
                _ => unreachable!("classify_op should have caught this"),
            }
        }
    }

    body.push_str(&format_function_epilogue(leaf, regs, assignment.stack_bytes));
    Ok((body, sig))
}

/// Resolve `Alias` chains to a concrete BufferLoc.
fn resolve_loc(locs: &[BufferLoc], id: NodeId) -> BufferLoc {
    let mut cur = id;
    loop {
        match locs[cur] {
            BufferLoc::Alias(next) => cur = next,
            other => return other,
        }
    }
}
```

(Add `use compiler::NodeId;` near the top of `codegen.rs` if not already there.)

- [ ] **Step 3: Update `emit_matmul` and `emit_relu` to take buffer-pointer args**

Replace the existing `emit_matmul(b, k, n, linear_idx)` with `emit_matmul_with_locs`:

```rust
fn emit_matmul_with_locs(
    b: u64,
    k: u64,
    n: u64,
    linear_idx: usize,
    src_loc: crate::buffer::BufferLoc,
    dst_loc: crate::buffer::BufferLoc,
    weight_offset: usize,
) -> String {
    use crate::buffer::BufferLoc;
    let lid = linear_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]\n"
    ));

    // Materialise src and dst pointers into x_src, x_dst registers.
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    // Weight pointer = x1 (params) + weight_offset*4
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&format!("    mov     x9, #{}\n", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    let _ = BufferLoc::InputReg; // import placeholder; remove if Rust complains
    s
}

fn emit_relu_with_locs(
    total_floats: u64,
    relu_idx: usize,
    src_loc: crate::buffer::BufferLoc,
    dst_loc: crate::buffer::BufferLoc,
) -> String {
    let rid = relu_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; relu: copy-clamp from src to dst ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}

/// Materialise a `BufferLoc` into a GPR (e.g. x11, x12).
fn materialise_ptr(reg: &str, loc: crate::buffer::BufferLoc) -> String {
    use crate::buffer::BufferLoc;
    match loc {
        BufferLoc::InputReg => format!("    mov     {}, x0\n", reg),
        BufferLoc::OutputReg => format!("    mov     {}, x2\n", reg),
        BufferLoc::StackOffset(off) => {
            if off == 0 {
                format!("    mov     {}, sp\n", reg)
            } else if off <= 4095 {
                format!("    add     {}, sp, #{}\n", reg, off)
            } else {
                let lo = (off & 0xFFFF) as u16;
                let hi = ((off >> 16) & 0xFFFF) as u16;
                let mut s = String::new();
                s.push_str(&format!("    movz    w10, #0x{:04x}\n", lo));
                if hi != 0 {
                    s.push_str(&format!("    movk    w10, #0x{:04x}, lsl #16\n", hi));
                }
                s.push_str(&format!("    add     {}, sp, x10\n", reg));
                s
            }
        }
        BufferLoc::Alias(_) => unreachable!("alias must be resolved by caller"),
    }
}
```

(Delete the old `emit_matmul` and `emit_relu` functions.)

- [ ] **Step 4: Update existing M4a unit tests in `profiles/arm64/src/tests.rs`**

The tests `linear_emits_matmul_loops_with_fmadd`, `relu_emits_separate_loop_with_fmov_zero_and_fmax`, and `relu_alone_after_matmul_does_not_break_existing_test` assert on `[x2, ..., lsl #2]` patterns. Now linear writes via `x12` (= dst pointer), relu reads via `x11` and writes via `x12`. Update assertions:

```rust
#[test]
fn linear_emits_matmul_loops_with_fmadd() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("fmadd"), "expected fmadd in:\n{s}");
    assert!(s.contains(".Lmm_i_0:"));
    assert!(s.contains(".Lmm_j_0:"));
    assert!(s.contains(".Lmm_k_0:"));
    assert!(s.contains("cmp     x3, #2"));
    assert!(s.contains("cmp     x4, #2"));
    assert!(s.contains("cmp     x5, #3"));
    assert!(s.contains("fmov    s0, wzr"));
    // Destination is x12 (materialised dst pointer), not raw x2.
    assert!(s.contains("str     s0, [x12,"));
}

#[test]
fn relu_emits_separate_loop_with_fmov_zero_and_fmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("fmov    s4, wzr"));
    assert!(s.contains("fmax    s3, s3, s4"));
    assert!(s.contains(".Lrelu_0:"));
    assert!(s.contains("cmp     x9, #4"));
    // Relu now uses materialised src/dst pointers.
    assert!(s.contains("ldr     s3, [x11,"));
    assert!(s.contains("str     s3, [x12,"));
}

#[test]
fn relu_alone_after_matmul_does_not_break_existing_test() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    assert!(asm.source.contains("fmadd"));
}
```

- [ ] **Step 5: Add new unit tests for prologue/epilogue shape**

Append to `profiles/arm64/src/tests.rs`:

```rust
#[test]
fn leaf_function_no_prologue() {
    // input → linear → relu (terminal): leaf, no intermediates beyond OutputReg
    // wait — n1 (linear non-terminal) → StackOffset, so intermediates > 0.
    // For a true leaf-no-prologue case, use just input → linear (terminal).
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Leaf, no intermediates → no stp, no sub sp, no ldp.
    assert!(!s.contains("stp"), "leaf-no-intermediates should have no stp:\n{s}");
    assert!(!s.contains("ldp"));
    assert!(!s.contains("sub     sp"));
}

#[test]
fn intermediate_buffers_allocated_on_stack() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("sub     sp, sp,"), "expected sub sp in:\n{s}");
    assert!(s.contains("add     sp, sp,"), "expected add sp in:\n{s}");
}
```

- [ ] **Step 6: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: zero warnings; total tests = baseline + 11 (9 new buffer tests from Task 2 + 2 new prologue tests in this task).

- [ ] **Step 7: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4b/codegen): wire buffer.rs + new prologue/epilogue helpers

walk_model now uses assign_buffers, compute_is_leaf, compute_callee_saved
to drive prologue/epilogue construction and per-op buffer pointer
materialisation.

asm.rs gains:
- LeafKind { Leaf, NonLeaf }
- format_function_prologue (.globl + label + conditional callee-saved
  + conditional non-leaf frame + conditional sub sp)
- format_function_epilogue (symmetric)
- emit_sp_sub / emit_sp_add (handle small immediate, shifted, and
  materialised-via-x9 forms per spec §6.3 large-immediate handling)

emit_matmul → emit_matmul_with_locs: takes BufferLoc args + weight
slot offset; materialises src/dst pointers into x11/x12 and weight
pointer into x13. emit_relu → emit_relu_with_locs: copy-with-clamp
from src (x11) to dst (x12). The M4a in-place optimisation for
terminal-relu-after-linear is dropped; future fusion pass restores it.

M4a unit tests adapted to new asm shape (load/store via x11/x12 not
raw x2). Two new tests: leaf_function_no_prologue,
intermediate_buffers_allocated_on_stack.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Move duplicate-model-name check to `compiler::ir::build` + extend snippet renderer

**Goal:** `compiler::ir::build` rejects duplicate model names with a `BuildErrorKind::DuplicateModelName { name, first_span }`. `nflc compile` renders the error with a `note: previously defined at line:col` plain-text line. `LowerError::DuplicateModelName` removed (`#[non_exhaustive]` makes it non-breaking).

**Files:**
- Modify: `compiler/src/ir/build.rs`, `compiler/src/ir/error.rs`, `compiler/src/ir/tests.rs`, `nflc/src/main.rs`, `profiles/arm64/src/types.rs`, `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add new `BuildErrorKind` variant in `compiler/src/ir/error.rs`**

Find the `BuildErrorKind` enum and add the variant:

```rust
    DuplicateModelName {
        name: String,
        first_span: crate::ast::Span,
    },
```

Update `Display for BuildErrorKind` (or wherever `BuildError`'s message is formatted) to handle the new variant — for example, in the `to_message` / `Display` arm:

```rust
            BuildErrorKind::DuplicateModelName { name, .. } => write!(
                f,
                "duplicate model name '{}': would emit conflicting symbols",
                name
            ),
```

Add a constructor matching the existing pattern (e.g. if there's `BuildError::shape(...)`, add `BuildError::duplicate_model_name(...)`):

```rust
impl BuildError {
    pub fn duplicate_model_name(
        name: String,
        first_span: crate::ast::Span,
        current_span: crate::ast::Span,
    ) -> Self {
        Self {
            kind: BuildErrorKind::DuplicateModelName { name, first_span },
            line: current_span.line,
            col: current_span.col,
            message: String::new(), // filled by Display when rendered
        }
    }
}
```

(Adapt to match the existing `BuildError`'s constructor convention; if `message` is computed eagerly in other constructors, do the same here.)

- [ ] **Step 2: Add the duplicate check in `compiler/src/ir/build.rs`**

After the loop that builds all models (typically near the end of the public `build` function, before the final `Ok(Uir { models })`), insert:

```rust
    // Reject duplicate model names — would produce conflicting symbols at codegen time.
    let mut seen: std::collections::HashMap<String, crate::ast::Span> =
        std::collections::HashMap::new();
    for model in &models {
        if let Some(prev_span) = seen.get(&model.name) {
            return Err(BuildError::duplicate_model_name(
                model.name.clone(),
                *prev_span,
                model.source_span,
            ));
        }
        seen.insert(model.name.clone(), model.source_span);
    }
```

- [ ] **Step 3: Add unit test in `compiler/src/ir/tests.rs`**

Append:

```rust
#[test]
fn duplicate_model_name_at_build_time() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n\
               model M [b=2]:\n    y: Tensor[b, 3]\n    y -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let err = crate::ir::build(&ast).expect_err("must fail");
    match err.kind {
        crate::ir::BuildErrorKind::DuplicateModelName { ref name, .. } => {
            assert_eq!(name, "M");
        }
        _ => panic!("expected DuplicateModelName, got {:?}", err.kind),
    }
}
```

- [ ] **Step 4: Extend `render_error_with_snippet` in `nflc/src/main.rs`**

Change the function signature to accept an optional `first_span`:

```rust
fn render_error_with_snippet(
    source: &str,
    path: &Path,
    line: u32,
    col: u32,
    message: &str,
    first_span: Option<(u32, u32)>,
) {
    eprintln!("error: {}", message);
    eprintln!("  --> {}:{}:{}", path.display(), line, col);
    let line_idx = (line as usize).saturating_sub(1);
    let lines: Vec<&str> = source.lines().collect();
    if let Some(src_line) = lines.get(line_idx) {
        let pad = " ".repeat(line.to_string().len());
        eprintln!("{} |", pad);
        eprintln!("{} | {}", line, src_line);
        let mut underline = String::new();
        for _ in 1..col {
            underline.push(' ');
        }
        underline.push('^');
        eprintln!("{} | {}", pad, underline);
    }
    if let Some((fl, fc)) = first_span {
        eprintln!("note: previously defined at {}:{}:{}", path.display(), fl, fc);
    }
}
```

Update all existing call sites (in `run_parse` and `run_build_uir`/`run_compile`) to pass `None` for the new arg:

```rust
render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
```

In `run_compile`, when handling a build error, check for `DuplicateModelName` and pass the `first_span`:

```rust
        Err(e) => {
            let first = match &e.kind {
                compiler::BuildErrorKind::DuplicateModelName { first_span, .. } => {
                    Some((first_span.line, first_span.col))
                }
                _ => None,
            };
            let msg = if e.message.is_empty() {
                format!("{}", e.kind)
            } else {
                e.message.clone()
            };
            render_error_with_snippet(&source, &path, e.line, e.col, &msg, first);
            return ExitCode::FAILURE;
        }
```

(Adjust if `BuildError` already exposes `Display` — use `format!("{}", e)` instead.)

- [ ] **Step 5: Remove `LowerError::DuplicateModelName`**

In `profiles/arm64/src/types.rs`:
- Delete the `DuplicateModelName { name: String, span: Span }` variant from the `LowerError` enum.
- Remove its arm from the `Display for LowerError` match.
- Remove its arm from the `LowerError::span()` match.

In `profiles/arm64/src/codegen.rs`:
- Find the duplicate-name check at the start of `walk_uir` and delete it (the IR builder now catches duplicates upstream; lowerer is unreachable on duplicates).

- [ ] **Step 6: Remove the corresponding test in `profiles/arm64/src/tests.rs`**

Delete `duplicate_model_name_returns_error` — its scenario is now covered by `duplicate_model_name_at_build_time` in compiler tests.

- [ ] **Step 7: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: net +0 (one test moves from profiles/arm64 to compiler) — same total as Task 3.

- [ ] **Step 8: Smoke test the diagnostic output**

Create a temporary fixture and verify the error format:

```bash
cat > /tmp/dup.nfl << 'EOF'
model M [b=2]:
    x: Tensor[b, 3]
    x -> linear[2]

model M [b=2]:
    y: Tensor[b, 3]
    y -> linear[2]
EOF

cargo run --quiet --bin nflc -- compile /tmp/dup.nfl --profile arm64 2>&1
echo "exit: $?"
```

Expected output (exact line numbers depend on file):
```
error: duplicate model name 'M': would emit conflicting symbols
  --> /tmp/dup.nfl:5:7
  |
5 | model M [b=2]:
  |       ^
note: previously defined at /tmp/dup.nfl:1:7
exit: 1
```

- [ ] **Step 9: Commit**

```bash
git add compiler/ nflc/ profiles/
git commit -m "refactor(m4b): move dup-model-name check to compiler::ir::build

Per spec §9: profile-agnostic invariant belongs in the IR builder,
not in profiles/arm64. compiler::ir::build now rejects duplicate
model names with BuildErrorKind::DuplicateModelName { name, first_span }.

The error carries first_span (location of the original definition)
so the diagnostic can show 'note: previously defined at ...' after
the snippet pointing at the redefinition.

render_error_with_snippet in nflc/src/main.rs accepts optional
first_span: Option<(u32, u32)> and emits the note line when present.
M4b uses one snippet + plain-text note; rustc-style two-snippet
upgrade is deferred to M4c.

LowerError::DuplicateModelName removed from profiles/arm64
(#[non_exhaustive] makes the removal non-breaking). One test moved
from profiles/arm64 to compiler/src/ir/tests.rs.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Refactor `codegen.rs` body emission into `ops/{mod, linear, relu}.rs`

**Goal:** Pure code-move refactor. `codegen.rs` becomes a slim dispatcher; `emit_matmul_with_locs` moves to `ops/linear.rs`, `emit_relu_with_locs` moves to `ops/relu.rs`. No behaviour change; all tests pass unchanged.

**Files:**
- Create: `profiles/arm64/src/ops/mod.rs`, `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/ops/relu.rs`
- Modify: `profiles/arm64/src/lib.rs`, `profiles/arm64/src/codegen.rs`

- [ ] **Step 1: Create `profiles/arm64/src/ops/mod.rs`**

```rust
//! Per-op codegen modules.

pub mod linear;
pub mod relu;

pub use linear::emit_linear;
pub use relu::emit_relu;
```

- [ ] **Step 2: Create `profiles/arm64/src/ops/linear.rs`**

Move `emit_matmul_with_locs` here, renaming to `emit_linear`. Bias-add is added in Task 7; this version is matmul-only:

```rust
//! Linear (matmul + optional bias-add) codegen.

use crate::buffer::BufferLoc;

pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
) -> String {
    let lid = linear_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]\n"
    ));

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&format!("    mov     x9, #{}\n", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}

/// Materialise a `BufferLoc` into a GPR.
pub(crate) fn materialise_ptr(reg: &str, loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg => format!("    mov     {}, x0\n", reg),
        BufferLoc::OutputReg => format!("    mov     {}, x2\n", reg),
        BufferLoc::StackOffset(off) => {
            if off == 0 {
                format!("    mov     {}, sp\n", reg)
            } else if off <= 4095 {
                format!("    add     {}, sp, #{}\n", reg, off)
            } else {
                let lo = (off & 0xFFFF) as u16;
                let hi = ((off >> 16) & 0xFFFF) as u16;
                let mut s = String::new();
                s.push_str(&format!("    movz    w10, #0x{:04x}\n", lo));
                if hi != 0 {
                    s.push_str(&format!("    movk    w10, #0x{:04x}, lsl #16\n", hi));
                }
                s.push_str(&format!("    add     {}, sp, x10\n", reg));
                s
            }
        }
        BufferLoc::Alias(_) => unreachable!("alias must be resolved by caller"),
    }
}
```

- [ ] **Step 3: Create `profiles/arm64/src/ops/relu.rs`**

Move `emit_relu_with_locs` here, renaming to `emit_relu`:

```rust
//! Relu (elementwise max with zero) codegen.

use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

pub fn emit_relu(
    total_floats: u64,
    relu_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let rid = relu_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; relu: copy-clamp from src to dst ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}
```

- [ ] **Step 4: Update `profiles/arm64/src/lib.rs`**

Add module declaration:

```rust
mod ops;
```

- [ ] **Step 5: Update `profiles/arm64/src/codegen.rs` to dispatch via `ops::*`**

Delete `emit_matmul_with_locs`, `emit_relu_with_locs`, and `materialise_ptr` from `codegen.rs` (they live in `ops/` now). Update the dispatch arms in `walk_model`:

```rust
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    let weight_offset = sig.params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    body.push_str(&crate::ops::emit_linear(
                        b, k, n, linear_idx, src_loc, dst_loc, weight_offset,
                    ));
                    linear_idx += 1;
                }
                StdOp::Relu => {
                    let buf_shape = &node.ty.shape;
                    let total: u64 = buf_shape.0.iter().product();
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    body.push_str(&crate::ops::emit_relu(total, relu_idx, src_loc, dst_loc));
                    relu_idx += 1;
                }
```

- [ ] **Step 6: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: same total as Task 4 (pure refactor, no test changes).

- [ ] **Step 7: Commit**

```bash
git add profiles/arm64/
git commit -m "refactor(m4b/ops): split codegen.rs body emission into ops/

Pure code-move. emit_matmul_with_locs → ops/linear.rs::emit_linear.
emit_relu_with_locs → ops/relu.rs::emit_relu. materialise_ptr lives
in ops/linear.rs as pub(crate), shared with relu.

ops/mod.rs declares the submodules and re-exports emit_*. codegen.rs
walk_model now dispatches via crate::ops::emit_linear / emit_relu.

Setup for Task 6 (dropout marker), Task 7 (bias-add inline in
emit_linear), Task 8 (ops/softmax.rs).

No behaviour change; all tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Dropout aliasing (INLINE)

**Goal:** Dropout nodes don't emit any asm; their output buffer is the operand's buffer (already covered by `assign_buffers` returning `Alias(operand)` for dropout — Task 2). `walk_model`'s dispatch must skip dropout. `classify_op` must accept it. `LowerError::UnsupportedOp` for dropout no longer fires.

**Files:**
- Create: `profiles/arm64/src/ops/dropout.rs` (marker)
- Modify: `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/ops/mod.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Create marker module `profiles/arm64/src/ops/dropout.rs`**

```rust
//! Dropout codegen.
//!
//! At inference, dropout is identity. The buffer-assignment first-pass
//! (`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand)` for
//! dropout nodes; therefore no asm is emitted. This module exists as a
//! marker so the ops/ directory has parallel structure for all StdOps.
```

- [ ] **Step 2: Add `pub mod dropout;` in `profiles/arm64/src/ops/mod.rs`**

```rust
pub mod dropout;
pub mod linear;
pub mod relu;

pub use linear::emit_linear;
pub use relu::emit_relu;
```

- [ ] **Step 3: Accept Dropout in `classify_op` (`profiles/arm64/src/codegen.rs`)**

Find `fn classify_op(...)` and change the `StdOp::Dropout =>` arm from returning `LowerError::UnsupportedOp` to returning `Ok(())`:

```rust
        StdOp::Dropout => Ok(()),
```

- [ ] **Step 4: Add Dropout dispatch arm in `walk_model`**

In the per-op `match op` block in `walk_model`, add a no-op arm for `StdOp::Dropout`:

```rust
                StdOp::Dropout => {
                    // No asm emitted: BufferLoc::Alias(operand) ensures downstream
                    // ops read from the operand's buffer directly.
                }
```

- [ ] **Step 5: Update tests in `profiles/arm64/src/tests.rs`**

Replace `dropout_returns_unsupported_op` with positive test `dropout_emits_no_code`:

```rust
#[test]
fn dropout_emits_no_code() {
    // input → linear → dropout → linear (terminal-linear)
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> dropout[rate=0.2] -> linear[2]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Two linear matmuls, no dropout-specific instructions or labels.
    assert!(s.contains(".Lmm_i_0:"));
    assert!(s.contains(".Lmm_i_1:"));
    assert!(!s.contains("dropout"), "asm must not mention dropout literally:\n{s}");
}
```

- [ ] **Step 6: Build + fmt + test**

```bash
cargo fmt --all
cargo build --workspace
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

- [ ] **Step 7: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4b/ops): dropout aliasing — no asm, buffer reuse

Per spec §8.3: dropout at inference is identity. classify_op accepts
Dropout. walk_model has a no-op dispatch arm. Buffer assignment
already returns Alias(operand) for dropout in Task 2's assign_buffers.

ops/dropout.rs is a marker module (no emit_* function) so the
ops/ directory has parallel structure across all StdOps.

Test moved from negative (returns_unsupported_op) to positive
(emits_no_code).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: `linear[N, bias=true]` bias-add inline

**Goal:** `emit_linear` accepts an optional `bias_offset: Option<usize>`. When present, after the k-loop sum is in `s0`, load `bias[j]` and add. `classify_op` accepts `bias=true`. `LowerError::LinearWithBias` no longer fires; remove the variant.

**Files:**
- Modify: `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/types.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add failing test in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn linear_with_bias_emits_bias_add() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // After the k-loop end, before the store, expect bias load + fadd.
    assert!(s.contains("fadd    s0, s0,"), "expected fadd s0, s0, ... in:\n{s}");
}

#[test]
fn linear_bias_packed_layout() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n",
    );
    let asm = lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    // Two slots: LinearWeight (size 6) then LinearBias (size 2) immediately after.
    assert_eq!(sig.params_layout.len(), 2);
    assert_eq!(sig.params_layout[0].kind, ParamKind::LinearWeight);
    assert_eq!(sig.params_layout[0].size, 6);
    assert_eq!(sig.params_layout[1].kind, ParamKind::LinearBias);
    assert_eq!(sig.params_layout[1].size, 2);
    assert_eq!(sig.params_layout[1].offset, 6);
    assert_eq!(sig.params_floats, 8);
}
```

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 linear_with_bias_emits_bias_add 2>&1 | tail -5
cargo test -p profiles-arm64 linear_bias_packed_layout 2>&1 | tail -5
```

Expected: both fail. The first because `emit_linear` doesn't emit `fadd`. The second because `classify_op` rejects `bias=true` (LinearWithBias error).

- [ ] **Step 3: Accept `bias=true` in `classify_op` (`profiles/arm64/src/codegen.rs`)**

Find the `StdOp::Linear =>` arm in `classify_op` and replace its `LinearWithBias` rejection with acceptance:

```rust
        StdOp::Linear => Ok(()),
```

(Bias is now a supported case; no rejection.)

- [ ] **Step 4: Extend `emit_linear` in `profiles/arm64/src/ops/linear.rs` to accept bias**

Change the function signature and add bias-add inline:

```rust
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
) -> String {
    let lid = linear_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}\n",
        if bias_offset.is_some() { " + bias" } else { "" }
    ));

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&format!("    mov     x9, #{}\n", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str("    mov     x14, x1\n");
        } else {
            s.push_str(&format!("    mov     x9, #{}\n", boff));
            s.push_str("    add     x14, x1, x9, lsl #2\n");
        }
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // Bias-add (if present) before the store: load bias[j], fadd into s0.
    if bias_offset.is_some() {
        s.push_str("    ldr     s5, [x14, x4, lsl #2]\n");
        s.push_str("    fadd    s0, s0, s5\n");
    }

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}
```

- [ ] **Step 5: Pass `bias_offset` from `walk_model`**

In `profiles/arm64/src/codegen.rs`'s `StdOp::Linear =>` arm, look up the bias slot and pass it:

```rust
                StdOp::Linear => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let out_shape = &node.ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let n = out_shape.0[1];

                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    let weight_offset = sig.params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearWeight && s.origin_node == node_idx)
                        .expect("LinearWeight slot must exist for this Linear")
                        .offset;
                    let bias_offset = sig.params_layout
                        .iter()
                        .find(|s| s.kind == ParamKind::LinearBias && s.origin_node == node_idx)
                        .map(|s| s.offset);
                    body.push_str(&crate::ops::emit_linear(
                        b, k, n, linear_idx, src_loc, dst_loc, weight_offset, bias_offset,
                    ));
                    linear_idx += 1;
                }
```

- [ ] **Step 6: Remove `LowerError::LinearWithBias`**

In `profiles/arm64/src/types.rs`:
- Delete the `LinearWithBias { span: Span }` variant from `LowerError`.
- Remove its arm from `Display for LowerError`.
- Remove its arm from `LowerError::span()`.

In `profiles/arm64/src/tests.rs`: delete `linear_with_bias_returns_lower_error` (test now obsolete; positive coverage replaces it).

- [ ] **Step 7: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: linear_with_bias_emits_bias_add and linear_bias_packed_layout pass.

- [ ] **Step 8: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4b/ops): linear[N, bias=true] — bias-add inline after k-loop

Per spec §8.1: emit_linear now accepts bias_offset: Option<usize>.
When present, after the k-loop accumulator s0 holds the matmul sum,
load bias[j] from params + bias_offset and fadd into s0 before the
store.

walk_model looks up the bias slot from sig.params_layout (already
emitted in Task 1's layout pass) and passes its offset through.

classify_op no longer rejects bias=true. LowerError::LinearWithBias
removed (#[non_exhaustive] makes it non-breaking).

Mixed_args fixture (linear[16, bias=true]) is now lowerable end-to-end;
integration test in Task 9 exercises it.

Two new unit tests: linear_with_bias_emits_bias_add (asm contains
fadd s0, s0, ...) and linear_bias_packed_layout (FnSig has 2 slots
in correct order with correct offsets).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: `softmax` — 3-pass + `bl _expf` + `-inf` materialisation

**Goal:** `emit_softmax` produces a per-row 3-pass: max → exp+sum → normalize. Uses `bl _expf`. Per-row state in callee-saved s8 (max) and s9 (sum). Function becomes non-leaf; prologue/epilogue infrastructure from Task 3 + leaf analysis from Task 2 handle the saving.

**Files:**
- Create: `profiles/arm64/src/ops/softmax.rs`
- Modify: `profiles/arm64/src/ops/mod.rs`, `profiles/arm64/src/codegen.rs`, `profiles/arm64/src/types.rs`, `profiles/arm64/src/tests.rs`

- [ ] **Step 1: Add failing tests in `profiles/arm64/src/tests.rs`**

Append:

```rust
#[test]
fn softmax_emits_three_passes() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Pass 2 has the bl _expf.
    assert!(s.contains("bl      _expf"), "expected 'bl _expf' in:\n{s}");
    // Pass 3 has the divide.
    assert!(s.contains("fdiv"), "expected fdiv (normalize pass) in:\n{s}");
}

#[test]
fn softmax_function_saves_d8_d9() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("stp     d8, d9, [sp, #-16]!"), "missing d8/d9 prologue:\n{s}");
    assert!(s.contains("ldp     d8, d9, [sp], #16"), "missing d8/d9 epilogue:\n{s}");
}

#[test]
fn non_leaf_function_saves_x29_x30() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("stp     x29, x30, [sp, #-16]!"));
    assert!(s.contains("ldp     x29, x30, [sp], #16"));
}
```

Also delete `softmax_returns_unsupported_op` (its negative scenario is replaced).

- [ ] **Step 2: Verify FAIL**

```bash
cargo test -p profiles-arm64 softmax 2>&1 | tail -10
```

Expected: tests fail because softmax still returns `UnsupportedOp`.

- [ ] **Step 3: Create `profiles/arm64/src/ops/softmax.rs`**

```rust
//! Softmax (per-row stable, libm expf) codegen.

use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// Uses `bl _expf` (libm). State in callee-saved s8 (per-row max), s9
/// (per-row sum). The function-level prologue (Task 3) handles d8/d9 save
/// and frame setup based on `compute_callee_saved` + `compute_is_leaf`.
pub fn emit_softmax(
    b: u64,
    k: u64,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let sid = softmax_idx;
    let mut s = String::new();
    s.push_str(&format!("    ; softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"));

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Outer per-row loop: x3 = i.
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lsm_i_end_{sid}\n"));

    // Compute row base offsets in elements: x4 = i * k.
    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x4, x3, x8\n");

    // Pass 1: max into s8. Initialise to -inf.
    s.push_str("    movz    w0, #0x0000\n");
    s.push_str("    movk    w0, #0xFF80, lsl #16\n");
    s.push_str("    fmov    s8, w0\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_max_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");
    s.push_str("    fmax    s8, s8, s1\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));

    // Pass 2: exp(x - max) → output, accumulate sum into s9.
    s.push_str("    fmov    s9, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_exp_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s0, [x11, x6, lsl #2]\n");
    s.push_str("    fsub    s0, s0, s8\n");
    s.push_str("    bl      _expf\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");
    s.push_str("    fadd    s9, s9, s0\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Pass 3: normalize.
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_norm_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s0, [x12, x6, lsl #2]\n");
    s.push_str("    fdiv    s0, s0, s9\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    // Next row.
    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    s
}
```

- [ ] **Step 4: Add `pub mod softmax;` in `profiles/arm64/src/ops/mod.rs`**

```rust
pub mod dropout;
pub mod linear;
pub mod relu;
pub mod softmax;

pub use linear::emit_linear;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
```

- [ ] **Step 5: Accept Softmax in `classify_op` and dispatch in `walk_model`**

In `profiles/arm64/src/codegen.rs`:

```rust
        StdOp::Softmax => Ok(()),
```

(Replace the existing `LowerError::UnsupportedOp` arm.)

In `walk_model`'s op-dispatch match, add:

```rust
                StdOp::Softmax => {
                    let in_shape = &model.nodes[operands[0]].ty.shape;
                    let b = in_shape.0[0];
                    let k = in_shape.0[1];
                    let src_loc = resolve_loc(&assignment.locs, operands[0]);
                    let dst_loc = assignment.locs[node_idx];
                    body.push_str(&crate::ops::emit_softmax(b, k, softmax_idx, src_loc, dst_loc));
                    softmax_idx += 1;
                }
```

Add `let mut softmax_idx = 0usize;` next to the existing `linear_idx`/`relu_idx` declarations.

- [ ] **Step 6: Remove `LowerError::UnsupportedOp` if no consumers remain**

After Tasks 6-8, the only `UnsupportedOp` paths were softmax + dropout — both now supported. The `UnsupportedOp` variant has no remaining call sites in M4b. Per spec §5.2, `#[non_exhaustive]` makes removal non-breaking. Either:

(a) Leave `UnsupportedOp` in place — defensive guard for future M5+ ops that might land before the codegen is updated. Keep its `Display` arm. Document with a `#[allow(dead_code)]` + comment per the M3c project principle.

(b) Remove it entirely.

**Decision:** keep with `#[allow(dead_code)]` + doc comment, mirroring the M3c handling of `ShapeError::WrongInputCount`. M5 will likely add a new op that needs it before codegen catches up.

In `profiles/arm64/src/types.rs`:

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LowerError {
    /// Defensive: op encountered that the codegen doesn't know how to lower.
    /// All M4b ops (linear/relu/dropout/softmax with or without bias) are
    /// supported; this variant exists as a guard for M5+ ops landing before
    /// codegen catches up. Annotate per the M3c project principle.
    #[allow(dead_code)]
    UnsupportedOp { op: String, span: compiler::ast::Span },
    /// Defensive: UIR contained a shape that wasn't fully resolved.
    ShapeNotConcrete { span: compiler::ast::Span },
}
```

- [ ] **Step 7: Build + fmt + clippy + test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: softmax tests pass.

- [ ] **Step 8: Commit**

```bash
git add profiles/arm64/
git commit -m "feat(m4b/ops): softmax 3-pass + bl _expf + -inf materialisation

Per spec §8.2: per-row pass-1 finds max (callee-saved s8), pass-2
computes expf(x - max) via bl _expf and accumulates sum (s9), pass-3
divides each element by sum.

-inf materialisation per spec §8 detail: 'fmov s8, #-inf' is invalid
(8-bit FP-immediate encoding doesn't include ±inf), so use
'movz w0, #0x0000; movk w0, #0xFF80, lsl #16; fmov s8, w0' to load
the bit pattern.

s8 (max) and s9 (sum) are AAPCS64 callee-saved (lower 64 bits of
v8/v9). The function-level prologue (Task 3) saves d8/d9 and
x29/x30 because compute_callee_saved + compute_is_leaf return
{d8_d9: true, leaf: false} for any model with softmax.

LowerError::UnsupportedOp kept (#[allow(dead_code)] + doc comment)
as defensive guard for M5+ ops that might land before codegen.

3 new unit tests: softmax_emits_three_passes,
softmax_function_saves_d8_d9, non_leaf_function_saves_x29_x30.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Integration tests — 5 M3 fixtures + M4a fixture + 2 reference-validation tests

**Goal:** Each M3 positive fixture runs end-to-end via FFI and matches a Rust reference. Reference functions for non-trivial logic (softmax, bias-add) get hand-computed validation tests.

**Files:**
- Modify: `profiles/arm64/tests/integration.rs`, `profiles/arm64/tests/common/mod.rs`

- [ ] **Step 1: Add reference functions in `profiles/arm64/tests/integration.rs`**

After the existing `reference_linear_relu`, append:

```rust
fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += input[i * k + kk] * weights[kk * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn reference_bias_add(acc: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    let b = acc.len() / n;
    let mut out = acc.to_vec();
    for i in 0..b {
        for j in 0..n {
            out[i * n + j] += bias[j];
        }
    }
    out
}

fn reference_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x.max(0.0)).collect()
}

fn reference_softmax_stable(input: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * k];
    for i in 0..b {
        let row = &input[i * k..(i + 1) * k];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for kk in 0..k {
            let e = (row[kk] - max).exp();
            out[i * k + kk] = e;
            sum += e;
        }
        for kk in 0..k {
            out[i * k + kk] /= sum;
        }
    }
    out
}
```

- [ ] **Step 2: Add reference-validation tests**

Append to `profiles/arm64/tests/integration.rs`:

```rust
#[test]
fn reference_softmax_stable_known_values() {
    let input = [1.0f32, 2.0, 3.0];
    let output = reference_softmax_stable(&input, 1, 3);
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    assert!((output[0] - 0.0900).abs() < 1e-4, "got {}", output[0]);
    assert!((output[1] - 0.2447).abs() < 1e-4, "got {}", output[1]);
    assert!((output[2] - 0.6652).abs() < 1e-4, "got {}", output[2]);
}

#[test]
fn reference_bias_add_known_values() {
    let acc = [1.0f32, 2.0, 3.0];
    let bias = [0.5f32, -1.0, 2.5];
    let out = reference_bias_add(&acc, &bias, 3);
    assert_eq!(out, vec![1.5, 1.0, 5.5]);
}
```

- [ ] **Step 3: Add integration test for `tiny_mlp.nfl` (linear → softmax)**

```rust
#[test]
fn tinymlp_full_with_softmax_runs_correctly() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/tiny_mlp.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");
    let dylib_path = common::compile_to_dylib(&asm.source, "tinymlp_softmax");

    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_TinyMLP") }.unwrap();

    let mut input = [0.0f32; 32]; // batch=8, hidden=4
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5;
    }
    let mut params = [0.0f32; 8]; // 4*2
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16]; // 8*2
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

    let intermediate = reference_matmul(&input, &params, 8, 4, 2);
    let expected = reference_softmax_stable(&intermediate, 8, 2);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "tinymlp[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

- [ ] **Step 4: Add integration test for `mixed_args.nfl` (bias=true)**

```rust
#[test]
fn mixed_args_runs_correctly() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/mixed_args.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");

    // Confirm layout: linear[16, bias=true] + linear[output=2] (no bias) + softmax.
    // params: weight(8*16=128) + bias(16) + weight(16*2=32) = 176 floats.
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_MixedArgs");
    assert_eq!(sig.params_floats, 8 * 16 + 16 + 16 * 2);

    let dylib_path = common::compile_to_dylib(&asm.source, "mixed_args");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_MixedArgs") }.unwrap();

    // batch=4, input=8, output=2
    let mut input = vec![0.0f32; 4 * 8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.8;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 50.0) * 0.01;
    }
    let mut output = vec![0.0f32; 4 * 2];
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

    // Reference: matmul → bias → relu → matmul → softmax
    let weights1 = &params[0..128];
    let bias1 = &params[128..144];
    let weights2 = &params[144..176];

    let mm1 = reference_matmul(&input, weights1, 4, 8, 16);
    let mm1_b = reference_bias_add(&mm1, bias1, 16);
    let r1 = reference_relu(&mm1_b);
    let mm2 = reference_matmul(&r1, weights2, 4, 16, 2);
    let expected = reference_softmax_stable(&mm2, 4, 2);

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "mixed_args[{i}]: asm got {a}, ref got {b}"
        );
    }
}
```

- [ ] **Step 5: Add integration test for `classifier.nfl`**

```rust
#[test]
fn classifier_runs_correctly() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/classifier.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Classifier");
    // batch=32, input=784, hidden=512+256, output=10.
    // Linears: 784*512 + 512*256 + 256*10 = 401408 + 131072 + 2560 = 535040
    assert_eq!(sig.params_floats, 535040);

    let dylib_path = common::compile_to_dylib(&asm.source, "classifier");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_Classifier") }.unwrap();

    // Use small deterministic values to avoid NaN from huge accumulators.
    let mut input = vec![0.0f32; 32 * 784];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }
    let mut output = vec![0.0f32; 32 * 10];
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

    // Sanity: each row of output sums to ~1 (softmax property).
    for i in 0..32 {
        let row_sum: f32 = output[i * 10..(i + 1) * 10].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "classifier row {i} sum = {row_sum}, expected ~1.0"
        );
    }

    // Sanity: all output elements in [0, 1].
    for (i, v) in output.iter().enumerate() {
        assert!(
            *v >= 0.0 && *v <= 1.0,
            "classifier[{i}] = {v} not in [0, 1]"
        );
    }
}
```

- [ ] **Step 6: Add integration tests for `pipeline_styles.nfl` and `comments.nfl`**

```rust
#[test]
fn pipeline_styles_runs_correctly() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/pipeline_styles.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");

    // Three models with same signature shape.
    assert_eq!(asm.functions.len(), 3);
    let names: Vec<&str> = asm.functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["nfl_forward_SingleLine", "nfl_forward_PerStepWrap", "nfl_forward_MixedWrap"]
    );

    let dylib_path = common::compile_to_dylib(&asm.source, "pipeline_styles");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();

    // Each model: batch=4, input=10, linear[8] -> relu -> linear[output=2] -> softmax
    // params = 10*8 + 8*2 = 96 floats.
    let mut input = vec![0.0f32; 4 * 10];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.5;
    }
    let mut params = vec![0.0f32; 96];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 48.0) * 0.01;
    }

    for name in &names {
        let sym_bytes = format!("{}\0", name).into_bytes();
        let forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = unsafe { lib.get(&sym_bytes) }.unwrap();
        let mut output = vec![0.0f32; 4 * 2];
        unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

        // Sanity: rows sum to ~1.
        for i in 0..4 {
            let row_sum: f32 = output[i * 2..(i + 1) * 2].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-3,
                "pipeline {name} row {i} sum = {row_sum}"
            );
        }
    }
}

#[test]
fn comments_runs_correctly() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/comments.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Commented");

    let dylib_path = common::compile_to_dylib(&asm.source, "comments");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_Commented") }.unwrap();

    let mut input = vec![0.0f32; sig.input_floats];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.0;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 10.0) * 0.05;
    }
    let mut output = vec![0.0f32; sig.output_floats];
    unsafe { forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr()); }

    // Final op is softmax → rows sum to ~1.
    let last_dim = sig.output_floats / 4; // batch=4 (per fixture header)
    for i in 0..4 {
        let row_sum: f32 = output[i * last_dim..(i + 1) * last_dim].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "comments row {i} sum = {row_sum}"
        );
    }
}
```

- [ ] **Step 7: Build + fmt + clippy + test (integration only)**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p profiles-arm64 --test integration 2>&1 | tail -20
```

Expected: all 6 integration tests pass on aarch64; skip cleanly elsewhere. Plus 2 reference-validation tests that always run.

If FMA divergence flakes any test (especially classifier with deeper composition), switch the reference's matmul accumulator to `f32::mul_add(input[i], weight[k], sum)` per spec §16. Document the change in the test commit.

- [ ] **Step 8: Full test suite verification**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: baseline + 11 new unit + 6 integration + 2 reference = baseline + 19 — actual count varies depending on running totals from prior tasks; track the diff from baseline 118.

- [ ] **Step 9: Commit**

```bash
git add profiles/arm64/
git commit -m "test(m4b): integration tests for all 5 M3 fixtures + reference validation

5 new integration tests (tiny_mlp, classifier, pipeline_styles,
comments, mixed_args) on top of the M4a fixture. Each builds UIR,
lowers to asm, assembles via cc -shared -arch arm64, dlopens via
libloading, calls nfl_forward_* with deterministic input/params,
compares against pure-Rust reference (or for classifier — sanity
checks that softmax rows sum to ~1 and outputs are in [0,1]).

reference_matmul, reference_bias_add, reference_relu,
reference_softmax_stable cover the per-op references.

reference_softmax_stable_known_values and
reference_bias_add_known_values validate the references themselves
against hand-computed values per spec §11.3 — without these, an
asm-and-reference shared bug could silently pass all integration
tests.

pipeline_styles tests three symbols from one .dylib (multi-model
NFL files).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: `docs/profile_guide/arm64.md` extension

**Goal:** Update the profile guide to reflect M4b coverage. Sections to update: ABI (params buffer + ParamSlot/ParamKind), supported ops (full table now), codegen patterns (bias-add, softmax 3-pass, dropout aliasing, intermediate buffers, non-leaf prologue with d8/d9 saves, libm dependency), limitations (greatly reduced from M4a list).

**Files:**
- Modify: `docs/profile_guide/arm64.md`

- [ ] **Step 1: Read current `docs/profile_guide/arm64.md` to understand structure**

Run:
```bash
cat docs/profile_guide/arm64.md | head -50
```

- [ ] **Step 2: Update §1 (ABI / Calling convention)**

Replace the function-signature block to use `params` instead of `weights`:

```markdown
For each `UirModel` in the input UIR, the profile emits one `extern "C"` function:

```c
void nfl_forward_<ModelName>(
    const float* input,
    const float* params,    // packed: weights + biases of all Linear nodes
    float*       output
);
```
```

(Surrounding prose stays — pointers in x0/x1/x2, AAPCS64, etc.)

- [ ] **Step 3: Add §2.5 Buffer layout — new subsection on `params` packing**

Insert after the existing §2 (or wherever buffer-layout discussion lives):

```markdown
### `params` buffer layout

`params` is a single packed float buffer holding all Linear weights and biases
for the model, in topological (UIR-node) order. For each `Linear` node:
1. The weight matrix slot — `LinearWeight`, size `K * N`.
2. (If `bias=true`) the bias vector slot — `LinearBias`, size `N`.

Slot offsets and sizes are exposed via `FnSig.params_layout: Vec<ParamSlot>`.
Each `ParamSlot` carries `kind: ParamKind`, `origin_node: NodeId`, `offset`
and `size` (both in float-elements). Callers use this metadata to
serialise their model checkpoint into the right offsets.

`ParamKind` is `#[non_exhaustive]`. Future ops (normalisation, embedding,
attention) introduce new variants without breaking downstream consumers.
```

- [ ] **Step 4: Update §3 (Supported ops table)**

Replace the table with all M4b-supported ops:

```markdown
| StdOp                      | Supported | Notes                                                    |
|----------------------------|-----------|----------------------------------------------------------|
| `Linear` (no `bias` attr)  | ✅        | Pure matmul.                                            |
| `Linear` (`bias=true`)     | ✅        | Matmul + per-output bias-add inline.                    |
| `Relu`                     | ✅        | Separate elementwise loop, copy-with-clamp.             |
| `Dropout`                  | ✅        | No-op at inference: `BufferLoc::Alias(operand)`.        |
| `Softmax`                  | ✅        | Numerically stable 3-pass, `bl _expf` from libm.        |
| `Input`                    | ✅        | Marker only — `BufferLoc::InputReg` (`x0`).             |
```

- [ ] **Step 5: Update §4 (Codegen patterns) — add bias, softmax, dropout, intermediate buffers, prologue**

Add subsections after the existing matmul + relu patterns:

```markdown
### 4.3 Bias-add (inline in `linear[N, bias=true]`)

After the k-loop accumulates `s0 = sum`, before the output store:

```asm
    ldr     s5, [x14, x4, lsl #2]    ; bias[j], with x14 = params + bias_offset*4
    fadd    s0, s0, s5
```

`x14` is set up once at the top of the linear emitter when `bias_offset.is_some()`.

### 4.4 Softmax (per-row 3-pass, libm `expf`)

Per row `i`, three passes over `K` elements:
1. `s8 = max(row)` — initialised to `-inf` via `movz/movk + fmov`.
2. For each `k`: `output[i,k] = expf(input[i,k] - s8)`, accumulate `s9 += output[i,k]`.
3. For each `k`: `output[i,k] /= s9`.

`s8` and `s9` are AAPCS64 callee-saved (lower 64 bits of v8/v9). Function prologue
saves d8/d9 when `compute_callee_saved` returns `RegSet { d8_d9: true }`.

`-inf` materialisation: `fmov sN, #-inf` is invalid (8-bit FP-immediate doesn't
encode ±inf). The portable pattern is to load the bit pattern (0xFF800000 for
f32) into a GPR and `fmov sN, wN`:

```asm
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16   ; w0 = 0xFF800000 = f32 -inf
    fmov    s8, w0
```

### 4.5 Dropout (aliasing, no asm)

Dropout at inference is identity. The buffer-assignment first-pass
(`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand_id)` for
dropout nodes. No asm is emitted. Downstream ops reading dropout's output
resolve the alias chain to the operand's actual `BufferLoc` (via
`resolve_loc` in `codegen.rs`).

### 4.6 Intermediate buffers (stack-allocated)

Non-terminal Linear and Softmax nodes whose results are consumed by another op
get a stack slot. The function prologue does `sub sp, sp, #N` (with `N`
rounded up to 16-byte alignment). The epilogue does `add sp, sp, #N`. For
sizes that don't fit a single 12-bit immediate, the codegen uses
`movz/movk + sub sp, sp, x9` instead.

The largest M4b fixture (classifier with batch=32) needs ~97KB of stack —
well under macOS default thread stack of 8MB.

### 4.7 Non-leaf prologue/epilogue

The pre-emission analyzers `compute_is_leaf` and `compute_callee_saved`
classify each function. M4b has two layers conditionally included:

- **Callee-saved FP** (`stp d8, d9, [sp, #-16]!` … `ldp d8, d9, [sp], #16`) —
  emitted iff softmax is present.
- **Non-leaf frame** (`stp x29, x30, [sp, #-16]!; mov x29, sp` … `ldp x29, x30, [sp], #16`) —
  emitted iff softmax is present (since softmax emits `bl _expf`).

Leaf functions with no intermediates (e.g., a single Linear terminal) emit
just `ret` — zero overhead.
```

- [ ] **Step 6: Update §5 (Errors)**

Trim the variants table to current state:

```markdown
| Variant                      | When                                                                |
|------------------------------|---------------------------------------------------------------------|
| `UnsupportedOp { op, span }` | Defensive: codegen doesn't know how to lower `op`. All M4b ops are supported; this fires only if M5+ adds a new op before codegen catches up. |
| `ShapeNotConcrete { span }`  | Defensive: shape wasn't fully resolved by `ir::build`. Should be unreachable. |
```

(Drop `LinearWithBias`, `DuplicateModelName` — both removed in Tasks 7 / 4.)

- [ ] **Step 7: Add §5.5 — libm dependency note**

```markdown
### Runtime dependency: libm

The softmax codegen emits `bl _expf`, which resolves to libm's `expf` symbol
at link time. On macOS and Linux, `cc` links libm by default. Bare-metal
targets without libm need a separate profile (M7+) — Taylor-series `exp`
implementation is reserved for that profile. The `arm64` profile assumes
POSIX with libm.
```

- [ ] **Step 8: Update §8 (Limitations) — drop most M4a items**

```markdown
## 8. Limitations (M4b)

- No SIMD. Scalar throughout. NEON is M5+/M6.
- No fusion. `linear → relu` emits two separate loops. Fusion is M5.
- No optimisation passes.
- No bare-metal target. Requires libm at link time.
- Single-snippet error rendering for duplicate-model-name (the `note: previously defined at`
  is plain text, not a second `^` snippet). Multi-snippet upgrade is M4c-or-later.
- Integration test runs only on aarch64 hosts with `cc` available; skips
  with logged reason elsewhere.
```

- [ ] **Step 9: Verify the doc renders**

```bash
wc -l docs/profile_guide/arm64.md
```

Expected: between 250 and 350 lines.

- [ ] **Step 10: Commit**

```bash
git add docs/profile_guide/arm64.md
git commit -m "docs(m4b): extend arm64 profile guide for full M4b coverage

Per spec §11.5:
- ABI section: forward(input, params, output); ParamSlot/ParamKind
  packed layout; #[non_exhaustive] note.
- Supported ops table: all 6 ops now ✅ (was 3).
- Codegen patterns added: bias-add inline, softmax 3-pass with -inf
  materialisation snippet, dropout aliasing, stack-allocated
  intermediate buffers, conditional non-leaf prologue with d8/d9.
- Errors table trimmed (LinearWithBias, DuplicateModelName removed).
- New §5.5: libm runtime dependency, bare-metal carve-out for M7+.
- Limitations greatly reduced from M4a's list.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: `uir.md` cross-link + `PROJECT_SPEC.md` update (INLINE)

**Goal:** Cross-link from `uir.md` to the dropout-as-noop codegen note. Update `PROJECT_SPEC.md` milestones table and Architecture Profiles table.

**Files:**
- Modify: `docs/language_reference/uir.md`, `PROJECT_SPEC.md`

- [ ] **Step 1: Add dropout-as-noop note in `docs/language_reference/uir.md`**

Find the section listing stdlib operations or implicit semantics. Append a paragraph:

```markdown
### Dropout at inference

Dropout in NFL v0.1 is inference-only and behaves as identity at run time
(no random masking). Codegen profiles implement this by aliasing the dropout
node's output buffer to its operand's, emitting no asm. See
[`docs/profile_guide/arm64.md`](../profile_guide/arm64.md) §4.5 for the
profile-specific implementation.
```

- [ ] **Step 2: Update `PROJECT_SPEC.md` Milestones table**

Find the M4 row and update:

```markdown
| 4 | `arm64` profile (4a + 4b complete) | Generate scalar AArch64 assembly for all 5 M3 fixtures end-to-end (linear ± bias, relu, dropout, softmax via libm expf) |
```

- [ ] **Step 3: Update `PROJECT_SPEC.md` Architecture Profiles table**

Update the `arm64` row:

```markdown
| `arm64`     | Apple Silicon / AArch64 POSIX | Scalar AArch64 (linear, relu, dropout, softmax). Stack-allocated intermediates. libm `expf` for softmax. |
```

- [ ] **Step 4: Verify docs**

```bash
grep -A1 "Dropout at inference" docs/language_reference/uir.md
grep "arm64" PROJECT_SPEC.md | head -5
```

- [ ] **Step 5: Commit**

```bash
git add docs/language_reference/uir.md PROJECT_SPEC.md
git commit -m "docs(m4b): uir.md dropout note + PROJECT_SPEC milestone updates

uir.md: new 'Dropout at inference' subsection clarifies semantics
and cross-links to arm64 profile guide §4.5.

PROJECT_SPEC.md:
- Milestones table M4 row → '4a + 4b complete' with full coverage
  description.
- Architecture Profiles arm64 row → expanded capability description
  (all 5 M3 fixtures end-to-end, libm expf, stack-allocated
  intermediates).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Closeout — DEVLOG (with ABI-break note) + CLAUDE.md + final smoke (INLINE)

**Goal:** Final verification across the workspace. DEVLOG entry with explicit ABI-break note. CLAUDE.md "Current Status" + Repository Structure updated.

**Files:**
- Modify: `DEVLOG.md`, `CLAUDE.md`

- [ ] **Step 1: Final end-to-end verification**

```bash
cargo fmt --all -- --check
cargo build --workspace 2>&1 | tail -3
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: zero diff from fmt, zero warnings, exit 0, ~134 tests pass.

- [ ] **Step 2: CLI smoke positive — all 5 M3 fixtures + M4a fixture**

```bash
for fix in tiny_mlp classifier pipeline_styles comments mixed_args m4_linear_relu; do
    echo "=== $fix.nfl ==="
    cargo run --quiet --bin nflc -- compile tests/fixtures/$fix.nfl --profile arm64 -o /tmp/$fix.s
    echo "nflc exit: $?"
    cc -shared -arch arm64 -o /tmp/$fix.dylib /tmp/$fix.s
    echo "cc exit: $?"
    nm /tmp/$fix.dylib | grep nfl_forward
done
```

Expected for each: nflc exits 0, cc exits 0, the right `_nfl_forward_*` symbol(s) appear.

- [ ] **Step 3: CLI smoke — unknown profile rejection**

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/tiny_mlp.nfl --profile xyz 2>&1
echo "exit: $?"
```

Expected: "error: unknown profile 'xyz' (supported: arm64)", exit 1.

If any verification step fails, **do not commit** — fix it first.

- [ ] **Step 4: Add M4b entry to `DEVLOG.md`**

Find the most recent entry (the `## 2026-05-03 — CI workflow added` one or whatever's at top). Insert above it:

```markdown
---

## 2026-05-04 — Milestone 4b closed: arm64 profile covers all 5 M3 fixtures end-to-end

### What was done
- Redesigned `FnSig` ABI: `weight_floats` removed, replaced by `params_floats`
  + `params_layout: Vec<ParamSlot>` with typed slots (`LinearWeight`,
  `LinearBias`). Generated functions take a single packed `params` buffer
  containing all weights and biases in topological UIR-node order.
  **This is a deliberate ABI break vs M4a** — see "Decisions made" below.
- Added `profiles/arm64/src/buffer.rs`: `assign_buffers` (BufferLoc per node:
  InputReg / OutputReg / StackOffset / Alias), `compute_is_leaf`,
  `compute_callee_saved` (RegSet for d8/d9), `BufferAssignment` carries
  16-byte aligned total stack size.
- New prologue/epilogue helpers in `asm.rs`: `format_function_prologue` /
  `_epilogue` accept `LeafKind` + `RegSet` + intermediate-bytes. Conditional
  layers: callee-saved d8/d9 (iff softmax), non-leaf x29/x30 (iff bl
  present), sub/add sp (iff intermediates > 0). Large-immediate handling
  via movz/movk + sub sp, sp, x9.
- Refactored `codegen.rs` body emission into `profiles/arm64/src/ops/`
  submodules (mod, linear, relu, softmax, dropout). Pure code-move +
  per-op extension.
- New ops:
  - `linear[N, bias=true]`: matmul + bias-add inline after k-loop.
  - `dropout`: zero asm; `BufferLoc::Alias(operand)` propagation.
  - `softmax`: 3-pass numerically stable (max → exp+sum → normalize),
    `bl _expf`, callee-saved s8/s9, `-inf` materialisation via movz/movk +
    fmov from GPR.
- Moved duplicate-model-name check from `profiles/arm64::walk_uir` up to
  `compiler::ir::build`. New `BuildErrorKind::DuplicateModelName { name,
  first_span }`. `render_error_with_snippet` extended with optional
  `first_span` → emits `note: previously defined at file:line:col` plain
  text after the snippet (single-snippet for M4b; rustc-style two-snippet
  upgrade is M4c-or-later).
- Removed `LowerError::DuplicateModelName` and `LowerError::LinearWithBias`
  (both `#[non_exhaustive]` removals are non-breaking). Kept
  `LowerError::UnsupportedOp` with `#[allow(dead_code)]` + doc comment per
  the M3c project principle (defensive guard for M5+ ops landing before
  codegen).
- 6 integration tests (tiny_mlp, classifier, pipeline_styles, comments,
  mixed_args, m4a fixture) + 2 reference-validation tests
  (reference_softmax_stable_known_values, reference_bias_add_known_values).
  All run on aarch64 macOS host; skip cleanly elsewhere.
- `docs/profile_guide/arm64.md` extended with bias-add, softmax 3-pass,
  dropout aliasing, intermediate buffer pattern, non-leaf prologue, libm
  dependency note. Limitations greatly reduced.
- `docs/language_reference/uir.md` cross-links to the new dropout-as-noop
  codegen detail. `PROJECT_SPEC.md` milestones table M4 row updated;
  Architecture Profiles arm64 row expanded.

### Decisions made
None new from spec — all captured in
`docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` during
brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-04-m4b-arm64-coverage.md` (12 tasks, 12 commits).

### ABI break callout

> **M4b deliberately broke the M4a public ABI of `FnSig`.** `weight_floats`
> field is gone; replaced by `params_floats` + `params_layout: Vec<ParamSlot>`.
> The generated `nfl_forward_*` C function signature changes the second
> parameter from `const float* weights` to `const float* params` (semantically
> the same buffer for M4a-compatible models — single LinearWeight slot — but
> renamed to reflect the more general layout).
>
> **Why deliberately:** the M4a name `weight_floats` would have been a lie
> the moment any M4b-supported model used `bias=true` (`params` then contains
> a LinearBias slot too). Renaming + restructuring at M4b is correct;
> retrofit-compat shims would have been worse.
>
> No external consumers exist (project is internal v0.1). Future readers of
> git history: this break was intentional, see `docs/superpowers/specs/2026-05-04-m4b-arm64-coverage-design.md` §5.4.

### Problems encountered
- (Fill in actual issues found during implementation. If none: "None —
  implementation followed the plan straight through.")

### Known tech debt (carried forward)
1. Single-snippet rendering for `DuplicateModelName` (plain-text note for
   first_span). Two-snippet rustc-style upgrade is M4c-or-later.
2. Integration test tempdir not cleaned up (carried from M4a). Acceptable;
   revisit in M4c if noisy.
3. Performance: scalar code, mul-based indexing, no fusion, no SIMD. M5+.
4. `LowerError::UnsupportedOp` kept defensively; will be exercised when M5+
   adds new ops.
5. Bare-metal arm64 target needs a separate profile (Taylor `exp` instead
   of libm). M7+.

### Next step
**Milestone 4 fully complete (4a + 4b).** All 5 M3 positive fixtures lower
end-to-end through the arm64 profile to runnable native code.

The next milestone is **Milestone 5 — kernel fusion pass**: introduce an
optimisation pass on the UIR (or just-before-codegen) that fuses
`linear → relu` (and similar elementwise-after-matmul patterns) into a
single loop with the relu inlined into the matmul store. Recovers M4a's
in-place relu performance and sets up the framework for more aggressive
fusion (matmul→bias→relu→softmax_max etc.). Brainstorming starts in a
fresh worktree once main is updated post-M4b-merge.
```

- [ ] **Step 5: Update `CLAUDE.md` "Current Status"**

Replace the existing Current Status block with:

```markdown
**Milestone 4 fully complete (4a + 4b).** The arm64 codegen profile lowers
all 5 M3 positive fixtures (`tiny_mlp`, `classifier`, `pipeline_styles`,
`comments`, `mixed_args`) plus the M4a-era `m4_linear_relu` fixture
end-to-end to native AArch64 assembly callable as a C function.

Op coverage: `linear` (with or without `bias=true`), `relu`, `dropout`
(no-op pass-through at inference), `softmax` (numerically stable 3-pass via
libm `expf`). ABI: single packed `params` buffer with typed slot metadata
(`FnSig.params_layout: Vec<ParamSlot>`). Stack-allocated intermediate
buffers; conditional non-leaf prologue with d8/d9 callee-saved when
softmax is present.

3-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib).
Production code stays std-only; `libloading` is the only test-only dev-dep.
~134 tests passing across lexer, parser, IR, profile codegen, FFI integration.
`cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
and `cargo fmt --all -- --check` all clean. CI green on every push.

`docs/profile_guide/arm64.md` documents the profile for users and contributors.

The immediate next step is **Milestone 5 — kernel fusion pass**: introduce
optimisation passes that fuse `linear → relu` and similar patterns,
recovering the in-place performance the M4a fixture had.
```

(Test count "134" is approximate — replace with the actual count from Step 1's `cargo test` run.)

- [ ] **Step 6: Update `CLAUDE.md` Repository Structure**

Find the `profiles/arm64/` block and update to reflect new files:

```markdown
├── profiles/
│   └── arm64/              ← `profiles-arm64` crate (lib only)
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs      ← `pub fn lower(&Uir) -> Result<Asm, LowerError>`
│       │   ├── types.rs    ← Asm, FnSig, ParamSlot, ParamKind, LowerError
│       │   ├── asm.rs      ← prologue/epilogue + sp helpers
│       │   ├── buffer.rs   ← BufferLoc, assign_buffers, leaf/callee-saved
│       │   ├── codegen.rs  ← walk_uir/walk_model dispatcher
│       │   ├── ops/
│       │   │   ├── mod.rs
│       │   │   ├── linear.rs    ← emit_linear (matmul ± bias)
│       │   │   ├── relu.rs      ← emit_relu
│       │   │   ├── softmax.rs   ← emit_softmax (3-pass + bl _expf)
│       │   │   └── dropout.rs   ← marker (no emitter)
│       │   └── tests.rs    ← unit tests
│       └── tests/
│           ├── integration.rs    ← end-to-end FFI tests for all M3 fixtures
│           └── common/mod.rs     ← cc + tempdir helpers
```

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md DEVLOG.md
git status
git commit -m "chore(m4b): close Milestone 4b — arm64 profile full M3 coverage

Per spec §17 acceptance criteria — all met:
- cargo build/clippy/fmt --check clean across workspace
- All 5 M3 fixtures + M4a fixture compile and run via FFI on host
- Duplicate model name rejected at ir::build with snippet+note
- libm expf for softmax (Apple/Linux default link)
- profile_guide/arm64.md updated, PROJECT_SPEC.md M4 row updated

DEVLOG includes the explicit ABI-break callout per spec §5.4 — for
future readers: weight_floats → params_floats + params_layout was
intentional, not accidental.

CLAUDE.md Current Status reflects Milestone 4 fully complete;
Repository Structure shows the 3-crate workspace + ops/ submodule
layout. Next milestone: M5 (kernel fusion pass).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Done. What's next?

After Task 12, M4b is complete by spec acceptance criteria:
1. ✅ `cargo build --workspace` clean.
2. ✅ `cargo clippy --workspace --all-targets -- -D warnings` clean.
3. ✅ `cargo fmt --all -- --check` clean.
4. ✅ All baseline tests still pass.
5. ✅ All 10 M4b unit tests + 6 integration tests pass.
6. ✅ `nflc compile` produces valid asm for all 5 M3 + M4a fixtures.
7. ✅ `compiler::ir::build` rejects duplicate model names with snippet + note.
8. ✅ CI green on PR.
9. ✅ `docs/profile_guide/arm64.md` updated.
10. ✅ DEVLOG entry with ABI-break callout. CLAUDE.md + PROJECT_SPEC updated.

**Push + PR.** Title suggestion: "Implement Milestone 4b: arm64 profile covers all 5 M3 fixtures end-to-end (ABI break)". After merge, Milestone 4 fully closes.

**Milestone 5 entry-point:** fresh `superpowers:brainstorming` cycle once main is updated post-M4b-merge. Decisions to make: where in the pipeline does fusion live (UIR-level pass vs codegen-time peephole)? Which patterns to fuse first (linear+relu, linear+bias+relu, softmax-max-pass-1)?
