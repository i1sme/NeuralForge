# M5b — Bias-Aware Fusion + EliminateDropout + `--passes` filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift M5a's `linear[bias=true]` fusion restriction (1-line pass change), add `EliminateDropout` UIR-pass that removes Dropout nodes before fusion, rename `--no-fuse` → `--no-passes`, add `--passes <list>` filter with canonical-order enforcement.

**Architecture:** Two compiler-side changes (extend `FuseLinearRelu`, add `EliminateDropout`) keep the `compiler::passes` infrastructure unchanged but add one entry to `default_pipeline` in canonical order. Profile (`profiles/arm64`) requires zero source changes — `emit_linear` already stacks bias-add + post-op + store correctly. CLI gains a stateful filter while preserving M5a's stdout/stderr discipline.

**Tech Stack:** Rust 2021, three-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib). Tests: `cargo test --workspace`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-05-m5b-bias-fusion-eliminate-dropout-design.md`

**Baseline at branch cut:** 173 tests. **Target:** 188 (+15 net per spec §11.6).

---

## File map

Files this plan touches:

- **Modify** `compiler/src/passes/fuse_linear_relu.rs` — delete bias-guard; invert `does_not_fuse_when_linear_has_bias` test; add `fuses_chain_with_bias` test. (Task 1)
- **Create** `compiler/src/passes/eliminate_dropout.rs` — new pass module with full impl + 8 unit tests. (Task 2)
- **Modify** `compiler/src/passes/mod.rs` — add `pub mod eliminate_dropout;` and update `default_pipeline()` to register both passes in canonical order with explanatory comment. (Task 2 adds module decl; Task 3 updates default_pipeline)
- **Modify** `compiler/src/passes/tests.rs` — rename existing pipeline-registry test, add order-dependency end-to-end test. (Task 3)
- **Modify** `nflc/src/main.rs` — rename `--no-fuse` → `--no-passes`; add `--passes <list>` parsing + validation + filter logic + order-divergence note; update `print_usage`. (Task 4)
- **Modify** `nflc/tests/cli_compile.rs` — rename M5a `--no-fuse` test; add 4 new smoke tests for `--passes` and mutually-exclusive case. (Task 5)
- **Modify** `profiles/arm64/tests/integration.rs` — add `fused_vs_unfused_mixed_args_match_numerically` integration test. (Task 6)
- **Modify** `DEVLOG.md` — closeout entry for M5b. (Task 7)
- **Modify** `CLAUDE.md` — `Current Status` reflects M5b complete + new flag names. (Task 7)

Two files in the spec but **not modified** in M5b code: `profiles/arm64/src/ops/linear.rs`, `profiles/arm64/src/buffer.rs`. Profile requires zero source changes — bias-aware codegen already works.

---

## Task overview

| # | Task | Mode | Net tests |
|---|---|---|---|
| 1 | Bias-aware fusion in FuseLinearRelu | SUBAGENT | +1 |
| 2 | EliminateDropout pass module | SUBAGENT | +8 |
| 3 | default_pipeline ordering + order-dependency test | SUBAGENT | +1 |
| 4 | CLI rename + `--passes` filter | SUBAGENT | 0 |
| 5 | CLI smoke tests | SUBAGENT | +4 |
| 6 | Bias-aware FFI integration test | SUBAGENT | +1 |
| 7 | Closeout — DEVLOG + CLAUDE.md | INLINE | 0 |
|   | **Total** | | **+15** |

After Task 7: 173 + 15 = 188. AC #2 requires `≥ baseline + new committed`, not exact.

---

## Task 1: Bias-aware fusion in `FuseLinearRelu`

**Goal:** Allow `linear[bias=true] → relu` to fuse. The arm64 `emit_linear` already stacks `matmul → bias-add → fmax → store` correctly when both `bias_offset.is_some()` AND `fused_post_ops == [Relu]`; this task removes the gate that blocks `FuseLinearRelu` from setting `fused_post_ops` on a `bias=true` Linear.

**Files:**
- Modify: `compiler/src/passes/fuse_linear_relu.rs`

- [ ] **Step 1: Invert the existing M5a test `does_not_fuse_when_linear_has_bias`**

In `compiler/src/passes/fuse_linear_relu.rs::tests`, locate the existing test:

```rust
#[test]
fn does_not_fuse_when_linear_has_bias() {
    let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> relu\n");
    let out = FuseLinearRelu.run(&uir).expect("ok");
    let m = &out.models[0];
    // 3 nodes preserved.
    assert_eq!(m.nodes.len(), 3);
    let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else { panic!() };
    assert!(fused_post_ops.is_empty());
}
```

Replace it (rename + invert assertions) with:

```rust
#[test]
fn fuses_when_linear_has_bias() {
    // M5b: bias-aware fusion. Linear[bias=true] → Relu now fuses.
    // The asm path for fused-bias-relu already worked in M5a; only
    // the pass-level guard `if linear_has_bias { continue; }` blocked
    // it. After M5b lifts that guard, this case fuses.
    let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> relu\n");
    let out = FuseLinearRelu.run(&uir).expect("ok");
    let m = &out.models[0];

    // Original: 3 nodes (input, linear, relu); fused: 2 (input, fused linear).
    assert_eq!(m.nodes.len(), 2, "expected 2 nodes; got: {:?}", m.nodes);

    let NodeKind::Op { op, fused_post_ops, attrs, .. } = &m.nodes[1].kind else {
        panic!("expected Op node");
    };
    assert_eq!(*op, StdOp::Linear);
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    // bias=true preserved on the fused Linear (fusion does not strip the bias attribute;
    // emit_linear reads it to decide whether to emit the bias-add inline before fmax).
    assert!(crate::ir::linear_has_bias(attrs));

    // model.output points at the fused Linear.
    assert_eq!(m.output, 1);
}
```

- [ ] **Step 2: Add a new test `fuses_chain_with_bias`**

Append after the renamed test:

```rust
#[test]
fn fuses_chain_with_bias() {
    // Chain where the second linear has bias=true. Both should fuse
    // independently — covers that bias-aware fusion composes with
    // multi-linear chains (the fusion of one Linear doesn't disable
    // the next Linear's fusion candidacy).
    let uir = build(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[4] -> relu -> linear[2, bias=true] -> relu\n"
    );
    let out = FuseLinearRelu.run(&uir).expect("ok");
    let m = &out.models[0];

    // Original: 5 nodes (input, linear[4], relu, linear[2,bias], relu).
    // After fusion: 3 nodes (input, fused linear[4], fused linear[2,bias]).
    assert_eq!(m.nodes.len(), 3);

    // First fused linear: no bias, has Relu post-op.
    let NodeKind::Op { fused_post_ops: f1, attrs: a1, .. } = &m.nodes[1].kind else {
        panic!("expected Op at n1")
    };
    assert_eq!(f1, &vec![PostOp::Relu]);
    assert!(!crate::ir::linear_has_bias(a1));

    // Second fused linear: bias=true, has Relu post-op.
    let NodeKind::Op { fused_post_ops: f2, attrs: a2, .. } = &m.nodes[2].kind else {
        panic!("expected Op at n2")
    };
    assert_eq!(f2, &vec![PostOp::Relu]);
    assert!(crate::ir::linear_has_bias(a2));
}
```

- [ ] **Step 3: Verify both new/inverted tests FAIL**

Run:
```bash
cargo test --lib fuses_when_linear_has_bias fuses_chain_with_bias 2>&1 | tail -10
```

Expected: both tests fail. The bias-guard `if linear_has_bias(attrs) { continue; }` still blocks fusion, so:
- `fuses_when_linear_has_bias`: assertion `assert_eq!(m.nodes.len(), 2)` fails — actual is 3.
- `fuses_chain_with_bias`: assertion `assert_eq!(m.nodes.len(), 3)` fails — actual is 4 (the `linear[2, bias=true]` doesn't fuse, only the first `linear[4] → relu` does).

- [ ] **Step 4: Delete the bias guard**

Locate this block in `fuse_one_model` (around lines 81-83 of the M5a-final state):

```rust
        if linear_has_bias(attrs) {
            continue; // M5a scope: bias-aware fusion is M5b.
        }
```

Delete those 3 lines.

After deletion, the surrounding `let NodeKind::Op { ... attrs, fused_post_ops, .. }` destructure binds `attrs` but the binding is now unused (the `if linear_has_bias(attrs)` was its only consumer). Rust will emit `unused variable: attrs`. Fix: drop the `attrs` field from the destructure pattern. The block becomes:

```rust
        let NodeKind::Op {
            op: StdOp::Linear,
            fused_post_ops,
            ..
        } = &linear_node.kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion in M5a.
        }
        if *consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Relu).
        }
        victim_to_producer.insert(relu_id, linear_id);
    }
```

- [ ] **Step 5: Verify the new tests PASS + full workspace clean**

```bash
cargo test --lib fuses_when_linear_has_bias fuses_chain_with_bias 2>&1 | tail -10
```

Expected: both pass.

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 174 (baseline 173 + 1 new chain test; the bias test was a rename so no count change).

- [ ] **Step 6: Commit**

```bash
git add compiler/src/passes/fuse_linear_relu.rs
git commit -m "$(cat <<'EOF'
feat(m5b/passes): bias-aware fusion in FuseLinearRelu

Per spec §7.1: lift the M5a `if linear_has_bias { continue; }` guard.
The arm64 emit_linear already stacks matmul → bias-add → fmax → store
correctly for fused Linears with both bias_offset.is_some() and
fused_post_ops == [Relu]; the only blocker was the pass-level rejection.

After this change: linear[bias=true] → relu fuses, computing
y = relu(x*W + b) in one pass (no intermediate buffer between bias-add
and relu).

Tests:
- Inverted does_not_fuse_when_linear_has_bias → fuses_when_linear_has_bias
  (asserts the post-fusion shape: 2 nodes, fused_post_ops == [Relu],
  bias=true preserved in attrs).
- New fuses_chain_with_bias: linear[4] → relu → linear[2, bias=true] → relu
  → 3 nodes, both Linears fused independently.

The destructure pattern in fuse_one_model drops the now-unused `attrs`
binding (was only used by the deleted bias check).

174 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `EliminateDropout` pass module

**Goal:** New `compiler::passes::eliminate_dropout::EliminateDropout` pass. Removes every Dropout node from the UIR, remapping consumers (and `model.inputs` / `model.output` if applicable) to the dropout's operand. Functional: returns a fresh `Uir` with NodeIds renumbered 0..N. 8 inline unit tests cover all spec edge cases.

**Files:**
- Create: `compiler/src/passes/eliminate_dropout.rs`
- Modify: `compiler/src/passes/mod.rs` (add `pub mod eliminate_dropout;` line; do NOT yet add to `default_pipeline()` — that's Task 3)

- [ ] **Step 1: Create `compiler/src/passes/eliminate_dropout.rs` with full impl + tests**

The file is new. Full content (copy verbatim):

```rust
//! `dropout` elimination pass (M5b).
//!
//! At inference time, dropout is a no-op (it only randomises during
//! training, which NFL v0.1 does not support). This pass removes every
//! Dropout node from the UIR, remapping its consumers to the dropout's
//! operand. After this pass, the `BufferLoc::Alias(operand)` machinery
//! in `profiles/arm64::buffer.rs` is unreachable in default mode (still
//! reachable for `--no-passes` and `--passes` filters that exclude this
//! pass). Profile remains complete relative to its input grammar — see
//! spec §4.2.
//!
//! Functional: returns a fresh Uir with renumbered NodeIds.

use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind};
use crate::ir::StdOp;
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};

pub struct EliminateDropout;

impl UirPass for EliminateDropout {
    fn name(&self) -> &str {
        "eliminate_dropout"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(eliminate_one_model(model)?);
        }
        Ok(Uir { models: new_models })
    }
}

/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this.
///
/// Note: this 3-step skeleton (identify victims → rebuild with remap →
/// remap inputs/output) echoes `FuseLinearRelu::fuse_one_model`, which
/// has an extra leading consumer-count step (FuseLinearRelu's victim
/// criterion 5 — single-consumer Linear — needs the precomputed map;
/// EliminateDropout has no consumer-count constraint and can skip it).
/// Extraction into a shared helper is deferred to M6+ when a third pass
/// with the same rebuild pattern lands ("three strikes then refactor").
fn eliminate_one_model(model: &UirModel) -> Result<UirModel, PassError> {
    // Step 1: identify victims (every Dropout node).
    let victims: HashSet<NodeId> = model
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(id, node)| match &node.kind {
            NodeKind::Op { op: StdOp::Dropout, .. } => Some(id),
            _ => None,
        })
        .collect();

    // Step 2: build new model — skip victims, remap operands.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.iter().enumerate() {
        if victims.contains(&old_id) {
            // Dropout's operand becomes Dropout's "result" id-wise.
            // NFL grammar guarantees Dropout has exactly one operand.
            let operand_old_id = match &node.kind {
                NodeKind::Op { operands, .. } => operands[0],
                _ => unreachable!("victim must be Op (filter-step established this)"),
            };
            let operand_new_id = id_map[&operand_old_id];
            id_map.insert(old_id, operand_new_id);
            continue;
        }

        let mut new_node = node.clone();
        if let NodeKind::Op { operands, .. } = &mut new_node.kind {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    // Step 3: remap inputs + output.
    let new_inputs: Vec<NodeId> = model.inputs.iter().map(|id| id_map[id]).collect();
    let new_output = id_map[&model.output];

    Ok(UirModel {
        name: model.name.clone(),
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
        source_span: model.source_span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::types::{AttrValue, OpAttr, Shape, Type};
    use crate::Uir;

    /// Build a Node with `NodeKind::Op` and a tensor type. Local helper
    /// used by the hand-built UIR tests in this module. Kept private —
    /// per spec §4.5, no shared helper between EliminateDropout and
    /// FuseLinearRelu (rule of three).
    fn op_node(op: StdOp, operands: Vec<NodeId>, attrs: Vec<OpAttr>, shape: Vec<u64>) -> Node {
        Node {
            kind: NodeKind::Op { op, operands, attrs, fused_post_ops: vec![] },
            ty: Type { name: "Tensor".into(), shape: Shape(shape) },
            source_span: Span::new(1, 1),
        }
    }

    fn input_node(name: &str, shape: Vec<u64>) -> Node {
        Node {
            kind: NodeKind::Input { name: name.into() },
            ty: Type { name: "Tensor".into(), shape: Shape(shape) },
            source_span: Span::new(1, 1),
        }
    }

    #[test]
    fn pass_name_is_stable() {
        assert_eq!(EliminateDropout.name(), "eliminate_dropout");
    }

    #[test]
    fn empty_uir_passes_unchanged() {
        let uir = Uir { models: Vec::new() };
        let out = EliminateDropout.run(&uir).expect("ok");
        assert_eq!(out.models.len(), 0);
    }

    #[test]
    fn removes_terminal_dropout() {
        // input → linear → dropout    (output IS the dropout)
        // After: input → linear   (output is the linear's new id)
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }],
                    vec![2, 2],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![1],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.5) }],
                    vec![2, 2],
                ),
            ],
            inputs: vec![0],
            output: 2, // dropout
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 2, "dropout node should be removed");
        assert_eq!(m.output, 1, "output should remap to linear's new id");
        assert_eq!(m.inputs, vec![0]);
        // Surviving linear has its operand still pointing at input (id 0).
        let NodeKind::Op { operands, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(operands, &vec![0]);
    }

    #[test]
    fn removes_internal_dropout() {
        // input → linear → dropout → softmax    (terminal softmax)
        // After: input → linear → softmax
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                    vec![2, 3],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![1],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.3) }],
                    vec![2, 3],
                ),
                op_node(StdOp::Softmax, vec![2], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 3, // softmax
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3, "dropout removed; 3 survivors");
        // Softmax (now id 2) reads linear (now id 1) directly.
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else { panic!() };
        assert!(matches!(op, StdOp::Softmax));
        assert_eq!(operands, &vec![1usize]);
        assert_eq!(m.output, 2);
    }

    #[test]
    fn removes_chained_dropouts() {
        // input → linear → dropout → dropout → relu
        // After: input → linear → relu  (both dropouts collapsed)
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                    vec![2, 3],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![1],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.2) }],
                    vec![2, 3],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![2],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.4) }],
                    vec![2, 3],
                ),
                op_node(StdOp::Relu, vec![3], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 4, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3, "both dropouts removed");
        // Relu (now id 2) reads linear (now id 1) directly — both dropouts collapsed.
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else { panic!() };
        assert!(matches!(op, StdOp::Relu));
        assert_eq!(operands, &vec![1usize]);
        assert_eq!(m.output, 2);
    }

    #[test]
    fn preserves_when_no_dropout() {
        // input → linear → relu (no dropout). Pass acts as identity.
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                    vec![2, 3],
                ),
                op_node(StdOp::Relu, vec![1], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 2);
        // NodeIds renumbered 0..N (identity here since no nodes were dropped).
        let NodeKind::Op { op: op1, .. } = &m.nodes[1].kind else { panic!() };
        assert!(matches!(op1, StdOp::Linear));
        let NodeKind::Op { op: op2, operands, .. } = &m.nodes[2].kind else { panic!() };
        assert!(matches!(op2, StdOp::Relu));
        assert_eq!(operands, &vec![1usize]);
    }

    #[test]
    fn multi_consumer_dropout() {
        // input → linear → dropout → { relu, softmax }   (dropout has two consumers)
        // After: input → linear → { relu, softmax }   (both consumers remap to linear)
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                    vec![2, 3],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![1],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.5) }],
                    vec![2, 3],
                ),
                op_node(StdOp::Relu, vec![2], vec![], vec![2, 3]), // consumer A: reads dropout
                op_node(StdOp::Softmax, vec![2], vec![], vec![2, 3]), // consumer B: reads dropout
            ],
            inputs: vec![0],
            output: 3, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 4, "dropout removed; 4 survivors");
        // Both relu and softmax now read linear (id 1) directly.
        let NodeKind::Op { op: op_a, operands: ops_a, .. } = &m.nodes[2].kind else { panic!() };
        let NodeKind::Op { op: op_b, operands: ops_b, .. } = &m.nodes[3].kind else { panic!() };
        assert!(matches!(op_a, StdOp::Relu));
        assert!(matches!(op_b, StdOp::Softmax));
        assert_eq!(ops_a, &vec![1usize], "relu should remap dropout-operand to linear");
        assert_eq!(ops_b, &vec![1usize], "softmax should remap dropout-operand to linear");
        // Output (was relu's old id 3) → relu's new id 2.
        assert_eq!(m.output, 2);
    }

    #[test]
    fn model_inputs_and_output_remapped() {
        // input → linear → dropout → relu (terminal relu).
        // Defensive: explicit assertions on both inputs and output remap correctness.
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(3) }],
                    vec![2, 3],
                ),
                op_node(
                    StdOp::Dropout,
                    vec![1],
                    vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.1) }],
                    vec![2, 3],
                ),
                op_node(StdOp::Relu, vec![2], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 3, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir { models: vec![model] };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];

        // Input id 0 was Input — preserved as 0 in output.
        assert_eq!(m.inputs, vec![0], "inputs should remap through id_map (identity for input nodes)");
        // Output was relu's old id 3 — should remap to relu's new id 2.
        assert_eq!(m.output, 2);
        // Verify the structure is intact.
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else { panic!() };
        assert!(matches!(op, StdOp::Relu));
        assert_eq!(operands, &vec![1usize]);
    }
}
```

- [ ] **Step 2: Wire the new module in `compiler/src/passes/mod.rs`**

Find the existing `pub mod fuse_linear_relu;` line and add `eliminate_dropout` ABOVE it (alphabetical order helps future readers; not required by Rust):

```rust
pub mod eliminate_dropout;
pub mod fuse_linear_relu;
```

Do NOT change `default_pipeline()` in this task — that's Task 3.

- [ ] **Step 3: Build + verify all 8 unit tests pass**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --lib eliminate_dropout 2>&1 | tail -15
```

Expected: 8 tests pass (`pass_name_is_stable`, `empty_uir_passes_unchanged`, `removes_terminal_dropout`, `removes_internal_dropout`, `removes_chained_dropouts`, `preserves_when_no_dropout`, `multi_consumer_dropout`, `model_inputs_and_output_remapped`).

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 174 + 8 = 182.

- [ ] **Step 4: Commit**

```bash
git add compiler/src/passes/eliminate_dropout.rs compiler/src/passes/mod.rs
git commit -m "$(cat <<'EOF'
feat(m5b/passes): EliminateDropout pass with full algorithm + 8 unit tests

Per spec §7.2: new compiler::passes::eliminate_dropout module.

Algorithm (per UirModel, 3-step skeleton):
1. Identify victims — every Dropout node.
2. Functional rebuild — skip victims, copy + remap operands of
   surviving nodes, map each victim's old NodeId to its operand's
   new NodeId (no producer mutation, unlike FuseLinearRelu).
3. Remap model.inputs and model.output via id_map.

Returns a fresh Uir with NodeIds renumbered 0..N. NFL grammar
guarantees Dropout has exactly one operand; the catch-all
unreachable! arm covers the filter-step contract.

Doc-comment on eliminate_one_model documents two invariants:
- Topological-order precondition (same as FuseLinearRelu).
- 3-step structure echoes FuseLinearRelu's 4-step (FuseLinearRelu
  has an extra consumer-count precompute that EliminateDropout
  doesn't need); shared helper deferred to M6+ when a third pass
  with the same rebuild pattern lands ("three strikes then
  refactor" — see spec §4.5).

8 inline unit tests:
- pass_name_is_stable (CLI contract)
- empty_uir_passes_unchanged (corner case)
- removes_terminal_dropout (model.output IS dropout)
- removes_internal_dropout (linear → dropout → softmax)
- removes_chained_dropouts (two dropouts collapse)
- preserves_when_no_dropout (identity case)
- multi_consumer_dropout (relu + softmax both reading dropout
  → both remap to linear)
- model_inputs_and_output_remapped (defensive coverage)

Module wired in passes/mod.rs (`pub mod eliminate_dropout;`).
default_pipeline() update is deferred to the next task so that
this commit isolates the pass implementation from the pipeline
ordering decision.

182 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `default_pipeline` ordering + order-dependency test

**Goal:** Register `EliminateDropout` in `default_pipeline()` BEFORE `FuseLinearRelu` (canonical order — see spec §4.1). Update the existing pipeline-registry test to assert this order. Add a new end-to-end test proving the order matters: a synthetic UIR `linear → dropout → relu` collapses to two nodes (input + fused linear with `fused_post_ops == [Relu]`) when run through the full pipeline.

**Files:**
- Modify: `compiler/src/passes/mod.rs` (update `default_pipeline()` body + comment)
- Modify: `compiler/src/passes/tests.rs` (rename + extend existing test, add order-dependency test)

- [ ] **Step 1: Update `default_pipeline()` in `compiler/src/passes/mod.rs`**

Replace the existing `default_pipeline()`:

```rust
/// The default pipeline of passes, applied in order.
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![Box::new(fuse_linear_relu::FuseLinearRelu)]
}
```

with:

```rust
/// The default pipeline of passes, applied in order.
///
/// Order matters: `EliminateDropout` MUST run before `FuseLinearRelu`
/// so that `linear → dropout → relu` collapses to `linear → relu`
/// before the fusion attempt. Reversed order leaves the pattern
/// unfused forever — `FuseLinearRelu` would see Linear's consumer
/// as Dropout (not Relu) and decline to fuse, then `EliminateDropout`
/// would remove the dropout, leaving an unfused `linear → relu`.
///
/// M6+ may introduce a fixed-point iteration or dependency-declaration
/// mechanism if a third pass with non-trivial coordination lands.
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![
        Box::new(eliminate_dropout::EliminateDropout),
        Box::new(fuse_linear_relu::FuseLinearRelu),
    ]
}
```

- [ ] **Step 2: Update the existing pipeline-registry test in `compiler/src/passes/tests.rs`**

Locate the existing test (M5a):

```rust
#[test]
fn default_pipeline_includes_fuse_linear_relu() {
    let pipeline = default_pipeline();
    let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
    assert!(
        names.contains(&"fuse_linear_relu"),
        "default_pipeline must include 'fuse_linear_relu'; got: {:?}",
        names
    );
}
```

Replace it with:

```rust
#[test]
fn default_pipeline_is_canonical_order() {
    // M5b: default_pipeline now contains two passes in canonical order.
    // EliminateDropout MUST come before FuseLinearRelu so that
    // `linear → dropout → relu` patterns can fuse (the dropout has to
    // be removed first for the Linear's consumer to become the Relu).
    let pipeline = default_pipeline();
    let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
    assert_eq!(
        names,
        vec!["eliminate_dropout", "fuse_linear_relu"],
        "default_pipeline must run eliminate_dropout before fuse_linear_relu; got: {:?}",
        names
    );
}
```

- [ ] **Step 3: Add the order-dependency end-to-end test**

Append to `compiler/src/passes/tests.rs`:

```rust
#[test]
fn pipeline_eliminates_dropout_before_fusing_linear_relu() {
    // Load-bearing test for spec §4.1: hand-build a synthetic UIR
    // `linear → dropout → relu` and run the full default pipeline.
    // Expected: 2 nodes (input + fused linear with fused_post_ops==[Relu]).
    // This proves end-to-end that EliminateDropout runs first AND
    // that FuseLinearRelu picks up the resulting linear→relu pattern.
    use crate::ast::Span;
    use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, PostOp, Shape, Type};
    use crate::ir::StdOp;
    use crate::UirModel;

    let span = Span::new(1, 1);
    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            Node {
                kind: NodeKind::Input { name: "x".into() },
                ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 3]) },
                source_span: span,
            },
            Node {
                kind: NodeKind::Op {
                    op: StdOp::Linear,
                    operands: vec![0],
                    attrs: vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }],
                    fused_post_ops: vec![],
                },
                ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) },
                source_span: span,
            },
            Node {
                kind: NodeKind::Op {
                    op: StdOp::Dropout,
                    operands: vec![1],
                    attrs: vec![OpAttr { name: "rate".into(), value: AttrValue::Float(0.5) }],
                    fused_post_ops: vec![],
                },
                ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) },
                source_span: span,
            },
            Node {
                kind: NodeKind::Op {
                    op: StdOp::Relu,
                    operands: vec![2],
                    attrs: vec![],
                    fused_post_ops: vec![],
                },
                ty: Type { name: "Tensor".into(), shape: Shape(vec![2, 2]) },
                source_span: span,
            },
        ],
        inputs: vec![0],
        output: 3, // relu
        source_span: span,
    };
    let uir = Uir { models: vec![model] };

    let out = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let m = &out.models[0];

    // After EliminateDropout: 3 nodes (input, linear, relu).
    // After FuseLinearRelu: 2 nodes (input, fused linear).
    assert_eq!(
        m.nodes.len(),
        2,
        "expected 2 nodes (input + fused linear); got: {:?}",
        m.nodes
    );

    // The fused linear has fused_post_ops == [Relu].
    let NodeKind::Op { op, fused_post_ops, .. } = &m.nodes[1].kind else {
        panic!("expected Op at n1");
    };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);

    // model.output points at the fused linear.
    assert_eq!(m.output, 1);
    assert_eq!(m.inputs, vec![0]);
}
```

- [ ] **Step 4: Build + verify**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --lib default_pipeline_is_canonical_order pipeline_eliminates_dropout_before_fusing_linear_relu 2>&1 | tail -10
```

Expected: both pass.

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 182 + 1 = 183 (the rename of the existing test doesn't change the count; only the new order-dependency test adds one).

- [ ] **Step 5: Commit**

```bash
git add compiler/src/passes/mod.rs compiler/src/passes/tests.rs
git commit -m "$(cat <<'EOF'
feat(m5b/passes): default_pipeline runs EliminateDropout before FuseLinearRelu

Per spec §4.1: register EliminateDropout in canonical order before
FuseLinearRelu. The doc-comment on default_pipeline() explains why
the order matters (and what would go wrong with the reverse order),
plus the deferral of fixed-point/dependency-declaration to M6+.

Tests:
- Renamed default_pipeline_includes_fuse_linear_relu →
  default_pipeline_is_canonical_order. Asserts the exact
  [eliminate_dropout, fuse_linear_relu] sequence (vec equality, not
  just contains).
- New pipeline_eliminates_dropout_before_fusing_linear_relu (load-
  bearing end-to-end test): hand-built UIR linear → dropout → relu,
  full pipeline run, asserts result has 2 nodes (input + fused linear
  with fused_post_ops == [Relu]) and model.output points at the fused
  linear. Proves both passes coordinate correctly under default_pipeline.

183 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI rename `--no-fuse` → `--no-passes` + `--passes <list>` filter

**Goal:** Replace `--no-fuse` with `--no-passes` (no alias). Add `--passes <list>` flag with comma-separated parsing, validation (unknown / empty / duplicate / mutually-exclusive-with-`--no-passes`), filter logic that preserves canonical order, and a stderr `note:` when the user-typed order diverges from canonical.

**Files:**
- Modify: `nflc/src/main.rs`

This is the largest task. After it, every M5a `--no-fuse` site is gone (no alias, no `#[allow(dead_code)]` shims), and `--passes` is the new finer-grained control.

- [ ] **Step 1: Rename `CompileArgs.no_fuse` → `no_passes` + add `passes` field**

Find the existing struct (around the top of `nflc/src/main.rs`):

```rust
struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_fuse: bool,
}
```

Replace it with:

```rust
struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_passes: bool,             // renamed from no_fuse (M5b §4.3)
    passes: Option<Vec<String>>, // None = default; Some(list) = filter (M5b §4.4)
}
```

- [ ] **Step 2: Update `parse_compile_args` to accept new flags + validate**

Find the existing `parse_compile_args`. It has a `while let Some(arg) = iter.next()` loop matching `--profile`, `-o`, `--no-fuse`, and an `other` fallback.

Replace the entire function body (signature unchanged: `fn parse_compile_args(args: &[String]) -> Result<CompileArgs, String>`):

```rust
fn parse_compile_args(args: &[String]) -> Result<CompileArgs, String> {
    let mut iter = args.iter();
    let path = iter
        .next()
        .ok_or_else(|| "compile: missing <file.nfl>".to_string())?
        .clone();
    if path.starts_with('-') {
        return Err(format!(
            "compile: expected <file.nfl> as first argument, got flag '{path}'"
        ));
    }

    let mut profile: Option<String> = None;
    let mut output: Option<PathBuf> = None;
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
            "-o" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "-o requires a value".to_string())?;
                output = Some(PathBuf::from(v));
            }
            "--no-passes" => {
                no_passes = true;
            }
            "--passes" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--passes requires a value".to_string())?;
                if v.is_empty() {
                    return Err(
                        "--passes value cannot be empty (use --no-passes to skip the pipeline)"
                            .to_string(),
                    );
                }
                // Strict split on `,` — no whitespace trimming. Users invoke
                // as --passes a,b or --passes "a,b" (no spaces inside).
                let names: Vec<String> = v.split(',').map(str::to_owned).collect();
                if names.iter().any(|n| n.is_empty()) {
                    return Err(format!(
                        "--passes value '{v}' contains an empty token (use --no-passes for empty)"
                    ));
                }
                passes = Some(names);
            }
            other => {
                return Err(format!("unknown flag: {other}"));
            }
        }
    }

    let profile = profile.ok_or_else(|| "compile: missing --profile <name>".to_string())?;

    // Mutually exclusive: --no-passes and --passes can't coexist.
    if no_passes && passes.is_some() {
        return Err("--no-passes and --passes are mutually exclusive".to_string());
    }

    // Validate --passes content against the canonical pass registry.
    if let Some(ref names) = passes {
        let canonical_names: Vec<String> = compiler::passes::default_pipeline()
            .iter()
            .map(|p| p.name().to_owned())
            .collect();

        // Duplicate check.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(format!("pass '{n}' specified more than once in --passes"));
            }
        }

        // Unknown-name check (dynamic available list).
        for n in names {
            if !canonical_names.iter().any(|c| c == n) {
                return Err(format!(
                    "unknown pass '{n}' (available: {})",
                    canonical_names.join(", ")
                ));
            }
        }
    }

    Ok(CompileArgs {
        path: PathBuf::from(path),
        profile,
        output,
        no_passes,
        passes,
    })
}
```

- [ ] **Step 3: Update `run_compile` to consume the new fields + emit notes**

Find the existing `run_compile` and locate the M5a "M5a: run UIR-passes pipeline" block (after `if profile != "arm64"` and before `match profiles_arm64::lower(&post_pass_uir)`).

Replace the destructure at the top:

```rust
fn run_compile(args: CompileArgs) -> ExitCode {
    let CompileArgs {
        path,
        profile,
        output: out_path,
        no_fuse,
    } = args;
```

with (note `no_passes` and `passes`):

```rust
fn run_compile(args: CompileArgs) -> ExitCode {
    let CompileArgs {
        path,
        profile,
        output: out_path,
        no_passes,
        passes,
    } = args;
```

Then replace the M5a pipeline block:

```rust
    // M5a: run UIR-passes pipeline (default), or skip if --no-fuse.
    let post_pass_uir = if no_fuse {
        eprintln!("note: passes skipped (--no-fuse)");
        uir
    } else {
        let pipeline = compiler::passes::default_pipeline();
        match compiler::passes::run_pipeline(&uir, &pipeline) {
            Ok(u) => {
                let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
                eprintln!("note: applied passes: {}", names.join(", "));
                u
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
```

with the M5b filtered version (full block):

```rust
    // M5b: run UIR-passes pipeline with optional filter, or skip
    // entirely if --no-passes. See spec §9.3.
    let post_pass_uir = if no_passes {
        eprintln!("note: passes skipped (--no-passes)");
        uir
    } else {
        let canonical = compiler::passes::default_pipeline();
        // Own the names (Vec<String>, not Vec<&str>) so the borrow on
        // `canonical` doesn't outlive the move into either match arm —
        // E0505 if Vec<&str> were used here (see spec §9.3).
        let canonical_names: Vec<String> =
            canonical.iter().map(|p| p.name().to_owned()).collect();

        let (pipeline, divergent) = match passes {
            None => (canonical, false),
            Some(user_names) => {
                // Filter canonical to retain only user-named passes,
                // preserving canonical order.
                let user_set: std::collections::HashSet<&str> =
                    user_names.iter().map(String::as_str).collect();
                let filtered: Vec<Box<dyn compiler::passes::UirPass>> = canonical
                    .into_iter()
                    .filter(|p| user_set.contains(p.name()))
                    .collect();
                let canonical_filtered_names: Vec<&str> =
                    filtered.iter().map(|p| p.name()).collect();
                // Order divergence: only meaningful when len >= 2.
                // user_names is Vec<String> (owned), canonical_filtered_names
                // is Vec<&str> (borrowed). Project user_names through
                // String::as_str into a Vec<&str> for type-aligned `!=`.
                let div = user_names.len() >= 2
                    && user_names.iter().map(String::as_str).collect::<Vec<_>>()
                        != canonical_filtered_names;
                (filtered, div)
            }
        };

        match compiler::passes::run_pipeline(&uir, &pipeline) {
            Ok(u) => {
                // Applied-note emitted only on success (M5a polish kept).
                let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
                eprintln!("note: applied passes: {}", names.join(", "));
                if divergent {
                    eprintln!(
                        "note: pass order is canonical ({}); user-specified order ignored",
                        canonical_names.join(", ")
                    );
                }
                u
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
```

- [ ] **Step 4: Update `print_usage`**

Find the existing `print_usage` function:

```rust
fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-fuse]        Skip optimisation passes (debugging)");
}
```

Replace with:

```rust
fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-passes]      Skip optimisation passes (debugging)");
    println!("                          [--passes <list>]  Run only listed passes (comma-separated)");
}
```

- [ ] **Step 5: Verify no `--no-fuse` substring remains in `nflc/src/`**

```bash
grep -rn "no_fuse\|--no-fuse" nflc/src/
```

Expected: empty output (no matches anywhere in `nflc/src/`). If anything remains — a missed call site, a leftover comment, a doc string — fix it before continuing.

The CLI smoke test file (`nflc/tests/cli_compile.rs`) still mentions `--no-fuse` in the test name from M5a; that gets renamed in Task 5.

- [ ] **Step 6: Smoke-test the binary manually**

The full unit/integration test suite for the CLI is in Task 5. This step is a quick sanity check before committing.

Default mode:

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m5b_default.s 2>/tmp/m5b_default.err
echo "exit: $?"
cat /tmp/m5b_default.err
grep "fmax\|Lrelu_" /tmp/m5b_default.s | head -3
```

Expected: exit 0; stderr contains `note: applied passes: eliminate_dropout, fuse_linear_relu`; asm has `fmax s0, s0, s4`.

`--no-passes`:

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --no-passes -o /tmp/m5b_nopasses.s 2>/tmp/m5b_nopasses.err
echo "exit: $?"
cat /tmp/m5b_nopasses.err
grep "fmax\|Lrelu_" /tmp/m5b_nopasses.s | head -3
```

Expected: exit 0; stderr contains `note: passes skipped (--no-passes)`; asm has `.Lrelu_*:` labels (separate relu loop).

`--passes <single>`:

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --passes fuse_linear_relu -o /tmp/m5b_filter.s 2>/tmp/m5b_filter.err
echo "exit: $?"
cat /tmp/m5b_filter.err
grep "fmax\|Lrelu_" /tmp/m5b_filter.s | head -3
```

Expected: exit 0; stderr contains `note: applied passes: fuse_linear_relu` (no `eliminate_dropout`); asm has `fmax s0, s0, s4` (since FuseLinearRelu still runs and m4_linear_relu has no dropout).

`--passes <reverse order>`:

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --passes fuse_linear_relu,eliminate_dropout -o /tmp/m5b_reverse.s 2>/tmp/m5b_reverse.err
echo "exit: $?"
cat /tmp/m5b_reverse.err
```

Expected: exit 0; stderr contains BOTH `note: applied passes: eliminate_dropout, fuse_linear_relu` AND `note: pass order is canonical (eliminate_dropout, fuse_linear_relu); user-specified order ignored`.

Unknown name:

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --passes foo 2>/tmp/m5b_err.err
echo "exit: $?"
cat /tmp/m5b_err.err
```

Expected: exit 1; stderr contains `unknown pass 'foo'` AND `available: eliminate_dropout, fuse_linear_relu`.

If any of the smoke checks above fail, investigate. Don't commit.

- [ ] **Step 7: Workspace clean + test suite**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 183 tests still passing. Existing M5a CLI tests (`compile_default_runs_fusion`, `compile_with_no_fuse_skips_fusion`, `compile_unknown_flag_rejected`) — note: `compile_with_no_fuse_skips_fusion` will start FAILING here because the rename to `--no-passes` means `--no-fuse` is now an unknown flag. **This is expected.** Task 5 renames the test. To keep the workspace green between Task 4 and Task 5, this commit and the Task 5 commit can land without intermediate `cargo test --workspace` passing — OR Task 4 can include the test rename as a single atomic step.

To keep CI green at every commit, include the M5a test rename in this commit. Add a sub-step:

- [ ] **Step 7a: Rename `compile_with_no_fuse_skips_fusion` → `compile_with_no_passes_skips_pipeline`**

In `nflc/tests/cli_compile.rs`, find the existing test:

```rust
#[test]
fn compile_with_no_fuse_skips_fusion() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--no-fuse",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: passes skipped (--no-fuse)"),
        "stderr missing passes-skipped note:\n{stderr}"
    );

    // Unfused asm: separate relu loop, no inline fmax.
    assert!(
        stdout.contains(".Lrelu_0_0:"),
        "stdout missing relu loop label (un-fused mode):\n{stdout}"
    );
    assert!(
        !stdout.contains("fmax    s0, s0, s4"),
        "stdout has inline fmax (fusion incorrectly applied in --no-fuse mode):\n{stdout}"
    );
}
```

Replace with:

```rust
#[test]
fn compile_with_no_passes_skips_pipeline() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
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
        stderr.contains("note: passes skipped (--no-passes)"),
        "stderr missing passes-skipped note:\n{stderr}"
    );
    // Successful skip mode does NOT emit the applied-passes note.
    assert!(
        !stderr.contains("note: applied passes:"),
        "stderr should not contain 'applied passes' when passes are skipped:\n{stderr}"
    );

    // Unfused asm: separate relu loop, no inline fmax.
    assert!(
        stdout.contains(".Lrelu_0_0:"),
        "stdout missing relu loop label (un-fused mode):\n{stdout}"
    );
    assert!(
        !stdout.contains("fmax    s0, s0, s4"),
        "stdout has inline fmax (fusion incorrectly applied in --no-passes mode):\n{stdout}"
    );
}
```

Re-run:

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 183.

- [ ] **Step 8: Commit**

```bash
git add nflc/src/main.rs nflc/tests/cli_compile.rs
git commit -m "$(cat <<'EOF'
feat(m5b/cli): rename --no-fuse → --no-passes, add --passes <list> filter

Per spec §9: extend nflc compile with finer-grained pipeline
control after M5b adds a second pass.

CompileArgs:
- no_fuse: bool → no_passes: bool (rename, no alias).
- New passes: Option<Vec<String>>. None = run default_pipeline; Some
  = filter to listed names (canonical order preserved).

parse_compile_args validates:
- --no-passes and --passes are mutually exclusive (error).
- --passes value cannot be empty or contain empty tokens (error,
  steers user to --no-passes).
- Names must match the dynamic registered set (default_pipeline()
  is the source of truth — error message lists available passes
  derived at runtime, not hardcoded, so M6+ pass additions surface
  automatically).
- Duplicate names in --passes (error).

run_compile filter logic:
- If --no-passes, skip the pipeline entirely; emit
  'note: passes skipped (--no-passes)' to stderr.
- If --passes <subset>, filter canonical pipeline (preserve order),
  run filtered set, emit 'note: applied passes: <ran-list>' to
  stderr.
- If user-typed order ≠ canonical (and len >= 2), additionally
  emit 'note: pass order is canonical (...); user-specified order
  ignored' to stderr — prevents confused-debugging-session
  'why didn't fusion happen?' mistakes.
- Stdout discipline preserved: asm only.

Code uses Vec<String> (owned) for canonical_names to avoid an E0505
where the Vec<&str> would borrow `canonical` past its move into
either match arm. Inline comment explains the trap (spec §9.3).

print_usage updated. --no-fuse fully removed: no aliases, no
#[allow(dead_code)] shims, no leftover substrings in nflc/src/
(verified via grep). M5a's compile_with_no_fuse_skips_fusion
test renamed to compile_with_no_passes_skips_pipeline (assertions
updated to new flag/note strings).

183 tests pass; CLI smoke verifies all four paths.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI smoke tests

**Goal:** Add 4 new CLI integration tests covering `--passes` filter, unknown name, order divergence warning, and mutually-exclusive interaction with `--no-passes`. The 5th smoke test (the rename of `compile_with_no_fuse_skips_fusion`) was done in Task 4 to keep the workspace green between commits.

**Files:**
- Modify: `nflc/tests/cli_compile.rs`

- [ ] **Step 1: Add `compile_with_passes_filter_runs_only_selected`**

Append to `nflc/tests/cli_compile.rs`:

```rust
#[test]
fn compile_with_passes_filter_runs_only_selected() {
    // --passes fuse_linear_relu against m4_linear_relu.nfl (which has
    // no dropout). The filter exercise is purely about pipeline
    // selection: stderr should show only the named pass; asm should
    // still contain inline fmax (since FuseLinearRelu is in the
    // filtered set).
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_relu",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: applied passes: fuse_linear_relu"),
        "stderr should show only fuse_linear_relu in applied list:\n{stderr}"
    );
    // The other registered pass should NOT appear in the applied list.
    assert!(
        !stderr.contains("note: applied passes: eliminate_dropout"),
        "stderr should NOT have eliminate_dropout in applied list when filtered:\n{stderr}"
    );
    // Fusion still applied.
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout should have inline fmax (fusion in filtered set):\n{stdout}"
    );
}
```

- [ ] **Step 2: Add `compile_with_passes_unknown_name_rejected`**

```rust
#[test]
fn compile_with_passes_unknown_name_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "foo",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Strict: must mention BOTH the offending name AND an "available:"
    // listing. The exact contents of the available list are dynamic
    // (M6+ may add passes); substring match on "available:" keeps the
    // test resilient.
    assert!(
        stderr.contains("unknown pass 'foo'"),
        "stderr missing unknown-pass error for 'foo':\n{stderr}"
    );
    assert!(
        stderr.contains("available:"),
        "stderr missing 'available:' substring (dynamic list):\n{stderr}"
    );
}
```

- [ ] **Step 3: Add `compile_with_passes_order_warning`**

```rust
#[test]
fn compile_with_passes_order_warning() {
    // User writes the two passes in REVERSE of canonical order.
    // CLI should still produce correct asm (canonical order applied)
    // AND emit a divergence note so the user knows their order was
    // overridden.
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_relu,eliminate_dropout",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // BOTH notes must appear: applied-passes (in canonical order) AND
    // the divergence warning. They are separate eprintln! calls per
    // spec §9.3 — substring checks are independent.
    assert!(
        stderr.contains("note: applied passes: eliminate_dropout, fuse_linear_relu"),
        "stderr missing canonical-order applied-passes note:\n{stderr}"
    );
    assert!(
        stderr.contains("user-specified order ignored"),
        "stderr missing order-divergence warning:\n{stderr}"
    );
    // Stdout still has the expected fused-asm shape (canonical order
    // produces the same asm as the no-flag default).
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout missing inline fmax (canonical order should still fuse):\n{stdout}"
    );
}
```

- [ ] **Step 4: Add `compile_no_passes_and_passes_rejected`**

```rust
#[test]
fn compile_no_passes_and_passes_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--no-passes",
            "--passes",
            "fuse_linear_relu",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "stderr missing mutually-exclusive error:\n{stderr}"
    );
}
```

- [ ] **Step 5: Build + verify all CLI tests pass**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p nflc --test cli_compile 2>&1 | tail -15
```

Expected: 7 tests pass total in `cli_compile.rs`:

- (existing M5a) `compile_default_runs_fusion`
- (renamed in Task 4) `compile_with_no_passes_skips_pipeline`
- (existing M5a) `compile_unknown_flag_rejected`
- (NEW) `compile_with_passes_filter_runs_only_selected`
- (NEW) `compile_with_passes_unknown_name_rejected`
- (NEW) `compile_with_passes_order_warning`
- (NEW) `compile_no_passes_and_passes_rejected`

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 183 + 4 = 187.

- [ ] **Step 6: Commit**

```bash
git add nflc/tests/cli_compile.rs
git commit -m "$(cat <<'EOF'
test(m5b/cli): smoke tests for --passes filter + mutually-exclusive case

Per spec §11.4: 4 new CLI integration tests using
Command::new(env!("CARGO_BIN_EXE_nflc")).

- compile_with_passes_filter_runs_only_selected: --passes fuse_linear_relu
  → stderr applied-note shows only that pass; stdout has inline fmax
  (fusion still applied since FuseLinearRelu is in the filtered set);
  eliminate_dropout absent from applied note.
- compile_with_passes_unknown_name_rejected: --passes foo → exit 1;
  stderr has BOTH 'unknown pass 'foo'' AND 'available:' (substring
  check on the dynamic list keeps the test resilient to M6+ pass
  additions).
- compile_with_passes_order_warning: --passes fuse_linear_relu,eliminate_dropout
  (reverse) → exit 0; stderr has BOTH the canonical-order applied
  note AND 'user-specified order ignored' divergence warning;
  stdout shape equivalent to default-mode fused asm.
- compile_no_passes_and_passes_rejected: --no-passes --passes <list>
  → exit 1; stderr contains 'mutually exclusive'.

The compile_with_no_passes_skips_pipeline rename happened in
Task 4 to keep the workspace green between commits.

187 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Bias-aware FFI integration test

**Goal:** Mirror M5a's `fused_vs_unfused_classifier_match_numerically` for the bias-aware case. Compile `mixed_args.nfl` (which has `linear[16, bias=true] → relu` as its first internal layer) via both pipeline-on and raw-UIR paths, FFI-call both with deterministic input/params, assert bit-exact `assert_eq!` on output.

**Files:**
- Modify: `profiles/arm64/tests/integration.rs`

- [ ] **Step 1: Append the new test**

In `profiles/arm64/tests/integration.rs`, append (after the M5a `fused_vs_unfused_classifier_match_numerically` test):

```rust
#[test]
fn fused_vs_unfused_mixed_args_match_numerically() {
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

    // Build BOTH paths.
    let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let fused_asm = profiles_arm64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_arm64::lower(&uir).expect("unfused lower");

    // Asm shape pre-asserts.
    assert!(
        fused_asm.source.contains("fmax    s0, s0, s4"),
        "fused asm missing inline fmax"
    );
    // mixed_args.nfl has `linear[16, bias=true] → relu` as the first
    // fusion candidate; the fused asm must still contain bias-add
    // (fadd s0, s0, s5) immediately before the fmax (within one
    // emit_linear, not in a separate function).
    assert!(
        fused_asm.source.contains("fadd    s0, s0, s5"),
        "fused asm missing bias-add (fadd s0, s0, s5):\n{}",
        fused_asm.source
    );
    assert!(
        !fused_asm.source.contains(".Lrelu_"),
        "fused asm should NOT have separate relu loops"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_0:"),
        "unfused asm missing relu loop label"
    );

    // Compile both, run both with same input/params, compare numerically.
    let fused_dylib = common::compile_to_dylib(&fused_asm.source, "fused_mixed_args");
    let unfused_dylib = common::compile_to_dylib(&unfused_asm.source, "unfused_mixed_args");

    let fused_lib = unsafe { libloading::Library::new(&fused_dylib).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_dylib).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();

    // Same deterministic input + params formula as the classifier
    // integration test. mixed_args has batch=4, input=8, output=2.
    let params_len = fused_asm.functions[0].params_floats;
    debug_assert_eq!(
        params_len, unfused_asm.functions[0].params_floats,
        "fused/unfused FnSig params_floats must agree"
    );

    let mut input = vec![0.0f32; 4 * 8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; params_len];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }

    let mut fused_out = vec![0.0f32; 4 * 2];
    let mut unfused_out = vec![0.0f32; 4 * 2];

    unsafe {
        fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
        unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
    }

    // assert_eq! exact equality: f32 store+load is bit-preserving;
    // bias-aware fusion (matmul → bias-add → fmax → store) only
    // relocates WHERE relu/bias is applied, not WHICH floats compute.
    for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "fused[{i}]={a} unfused[{i}]={b} — bias-aware fusion changed numerics"
        );
    }
}
```

- [ ] **Step 2: Build + run the new integration test**

```bash
cargo fmt --all
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p profiles-arm64 --test integration fused_vs_unfused_mixed_args_match_numerically 2>&1 | tail -10
```

Expected on aarch64 macOS (current host): 1 test passes. On non-aarch64: skips with stderr message.

- [ ] **Step 3: Full workspace test count**

```bash
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: 187 + 1 = 188.

- [ ] **Step 4: Commit**

```bash
git add profiles/arm64/tests/integration.rs
git commit -m "$(cat <<'EOF'
test(m5b/integration): fused_vs_unfused_mixed_args numerical equivalence

Per spec §11.5 + AC #4: end-to-end FFI test confirming
bias-aware fusion preserves numerics. Uses mixed_args.nfl
which has linear[16, bias=true] → relu as the internal fusion
candidate.

Both fused and unfused paths build UIR, lower to asm, compile via
cc -shared -arch arm64, dlopen via libloading, call with same
deterministic input/params, compare outputs with assert_eq!
(bit-exact, NOT epsilon).

Asm shape pre-asserts:
- Fused: contains 'fmax s0, s0, s4' AND 'fadd s0, s0, s5' (bias
  inside one emit_linear); does NOT contain '.Lrelu_*'.
- Unfused: contains '.Lrelu_0_0:'.

The 'fadd' assertion is critical for this test — it pins that
bias-add did NOT regress when the fusion guard was lifted in
Task 1. Without this, a pass-side bug that stripped bias=true
during fusion would still pass the bit-exact equality check
(both paths would have no bias) but produce wrong values vs.
the spec's y = relu(x*W + b).

Bit-exactness rationale: f32 store+load is bit-preserving
(IEEE 754 + AArch64); fusion only relocates where bias-add and
relu apply, not which floats compute.

188 tests pass.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Closeout — DEVLOG + CLAUDE.md (INLINE)

**Goal:** Final verification + DEVLOG entry + CLAUDE.md "Current Status" update. M5c (profile guide doc + PROJECT_SPEC milestones close-out) remains the next milestone.

**Files:**
- Modify: `DEVLOG.md` — append M5b entry (newest at top, above the M5a entry).
- Modify: `CLAUDE.md` — replace "Current Status" body to reflect M5b complete.

This task is INLINE (executed directly in the controller session, no subagent dispatch).

- [ ] **Step 1: Final verification**

```bash
cargo fmt --all -- --check
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace 2>&1 | grep "^test result" | awk '{sum+=$4} END {print "TOTAL:", sum}'
```

Expected: all checks pass; total tests = 188 (or higher if implementer added defensive tests).

```bash
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 -o /tmp/m5b_d.s 2>/tmp/m5b_d.err
echo "default exit: $? — stderr:" && cat /tmp/m5b_d.err
grep -E "fmax|Lrelu_" /tmp/m5b_d.s | head -3

cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 --no-passes -o /tmp/m5b_n.s 2>/tmp/m5b_n.err
echo "no-passes exit: $? — stderr:" && cat /tmp/m5b_n.err
grep -E "fmax|Lrelu_" /tmp/m5b_n.s | head -3

# Stdout discipline.
cargo run --quiet --bin nflc -- compile tests/fixtures/m4_linear_relu.nfl --profile arm64 2>/dev/null | head -3
echo "(stdout above must show ONLY .globl / labels; no 'note:' lines)"

# --no-fuse fully removed.
grep -rn "no_fuse\|--no-fuse" nflc/src/ profiles/ compiler/src/ || echo "no matches — clean"
```

Expected: all checks green; default produces `fmax`, no `.Lrelu_`; `--no-passes` produces `.Lrelu_*`, no `fmax`; stdout shows only asm; grep produces no matches in src/ trees (DEVLOG history is allowed to mention `--no-fuse` since it's a historical record).

- [ ] **Step 2: Append M5b entry to `DEVLOG.md`**

Find the most recent entry (M5a, dated 2026-05-04). Insert above it (separated by `---`):

```markdown
---

## 2026-05-05 — Milestone 5b closed: bias-aware fusion + EliminateDropout + --passes filter

### What was done
- Lifted M5a's `if linear_has_bias { continue; }` guard in
  `FuseLinearRelu`. `linear[bias=true] → relu` now fuses into a single
  `emit_linear` block that stacks `matmul → bias-add → fmax → store`.
  No profile-side changes — `emit_linear` already supported the asm
  shape; only the pass-level rejection blocked it.
- Added `compiler::passes::eliminate_dropout::EliminateDropout` —
  a new UIR-pass that removes every Dropout node from the graph,
  remapping consumers and `model.inputs` / `model.output` to the
  dropout's operand. Functional 3-step rebuild (identify victims →
  rebuild with id-remap → remap inputs/output). 8 inline unit tests
  cover terminal-dropout, internal-dropout, chained dropouts,
  multi-consumer dropout, identity-when-no-dropout, and explicit
  inputs/output remap correctness.
- `default_pipeline()` now registers BOTH passes in canonical order
  `[EliminateDropout, FuseLinearRelu]`. Order matters: without it,
  `linear → dropout → relu` patterns would never fuse. The doc-comment
  documents the dependency. M6+ may revisit if a third pass needs
  non-trivial coordination.
- New end-to-end pipeline test
  `pipeline_eliminates_dropout_before_fusing_linear_relu` proves the
  order-dependency: hand-built UIR `linear → dropout → relu` collapses
  to two nodes (input + fused linear with `fused_post_ops == [Relu]`)
  through the full pipeline.
- CLI: `--no-fuse` renamed to `--no-passes`. Clean break — no alias,
  no `#[allow(dead_code)]` shim, `grep` against `nflc/src/` confirms
  zero residue. New `--passes <list>` flag for filtered runs:
  comma-separated, validated against the dynamic
  `default_pipeline()` registry, mutually exclusive with `--no-passes`,
  emits a stderr `note:` when user-typed order diverges from canonical.
- Integration test `fused_vs_unfused_mixed_args_match_numerically`
  proves bit-exact equivalence for the bias-aware case using
  `mixed_args.nfl` (which has `linear[16, bias=true] → relu` as its
  first internal layer). Mirrors M5a's classifier test pattern.
- Existing M4b/M5a integration tests (`mixed_args_runs_correctly`,
  `classifier_runs_correctly`, `fused_vs_unfused_classifier_match_numerically`,
  others) continue to pass without changes — the pipeline-order
  switch automatic via M5a Task 10's adaptation.

### Decisions made
None new. All design decisions captured in
`docs/superpowers/specs/2026-05-05-m5b-bias-fusion-eliminate-dropout-design.md`
during brainstorming. This session executed the plan in
`docs/superpowers/plans/2026-05-05-m5b-bias-fusion-eliminate-dropout.md`
(7 tasks, ~12 commits with review-driven polish).

### Pre-decided architectural calls (from spec §4)
1. **Pipeline order: `[EliminateDropout, FuseLinearRelu]`.** Hardcoded
   in `default_pipeline()` with explanatory comment. Fixed-point /
   dependency-declaration deferred to M6+ when a third pass with
   non-trivial coordination lands.
2. **Profile keeps `BufferLoc::Alias(operand)` for Dropout.** Fallback
   for `--no-passes` mode; profile remains complete relative to its
   input grammar. A verification tool that fails on valid UIR isn't a
   verification tool — it's a trap.
3. **`--no-fuse` removed without alias.** v0 has no external consumers;
   backward-compat aliases here would be cargo-cult.
4. **`--passes` is filter-only, canonical order enforced.** Reorder
   mode (B-variant) deferred to M6+ if a real research case demands it.
5. **No shared helper for victim/remap pattern.** Two passes duplicate
   the rebuild skeleton intentionally — "three strikes then refactor"
   rule. EliminateDropout's doc-comment flags this for the M6+ author.

### Problems encountered
- None blocking. The spec went through five review rounds (user caught
  three placeholder/contradiction issues, one E0505 borrow-checker bug
  in the pseudocode, and one cross-reference typo before the plan was
  written). All five fixed inline before implementation began.
- Test count finished at 188 (matches plan target exactly: 173 + 15).

### Known tech debt (carried forward)
1. **Profile guide doc updates** (`docs/profile_guide/arm64.md`):
   bias-aware fusion section, `--no-passes` / `--passes` documentation,
   EliminateDropout removal note. → **M5c**.
2. **`PROJECT_SPEC.md` milestones table** close-out for M5 → **M5c**.
3. **Pass-shared helper for victim/remap pattern** — defer to M6+ when
   the third pass with the same structural pattern lands ("three
   strikes then refactor"). DEVLOG and EliminateDropout doc-comment
   flag the rationale.
4. **`--passes` reorder mode (B-variant)** — only if research / debugging
   case demands it. M6+.
5. **Pass dependency declaration / fixed-point iteration** — when a
   third pass with non-trivial coordination lands. M6+.
6. **Snapshot tests via `insta`** — substring asserts continue to suffice.

### Next step
**Milestone 5b complete.** M5 remains technically open until M5c lands
the documentation: profile guide updates for bias-aware fusion +
EliminateDropout + the new CLI flags, plus the PROJECT_SPEC milestones
close-out. M5c is small (docs only, no code changes) and can be a
single-commit milestone.

After M5c: brainstorm M6 in a fresh worktree once main is updated
post-M5b-merge. M6 is open territory — possible directions: bare-metal
target, attention-pattern fusion (`linear → softmax_max`), x86_64
profile, or pass-helper extraction triggered by a third pass.
```

(Keep all existing entries intact; only insert above the most recent.)

- [ ] **Step 3: Update `CLAUDE.md` "Current Status"**

Find the existing block (M5a-version) and replace its body:

```markdown
**Milestone 5b complete.** UIR-pass infrastructure ships two passes:
`EliminateDropout` (removes dropout nodes from the graph at inference
time) and `FuseLinearRelu` (now bias-aware — fuses both `linear → relu`
and `linear[bias=true] → relu`). `default_pipeline()` runs them in
canonical order `[EliminateDropout, FuseLinearRelu]` so that
`linear → dropout → relu` patterns collapse and fuse end-to-end.

CLI: `--no-fuse` renamed to `--no-passes` (clean break, no alias).
New `--passes <list>` filter accepts a comma-separated subset of pass
names; canonical order is enforced regardless of user-typed order, with
a stderr `note:` when they diverge. Mutually exclusive with `--no-passes`.
All flag validation uses the dynamic `default_pipeline()` registry, so
M6+ pass additions surface in error messages automatically.

Profile (`profiles/arm64`) requires zero changes for M5b. `emit_linear`
already stacks `matmul → bias-add → fmax → store` correctly, and the
`BufferLoc::Alias(operand)` machinery for Dropout stays as the fallback
path for `--no-passes` and `--passes` filters that exclude
`eliminate_dropout`.

Op coverage unchanged from M4 (linear ± bias, relu, dropout, softmax).
The `fused_vs_unfused_mixed_args_match_numerically` integration test
confirms bias-aware fusion preserves numerics bit-exactly via
`assert_eq!` on every output element.

3-crate workspace (`compiler` lib, `nflc` bin, `profiles/arm64` lib).
Production code std-only; `libloading` is a test-only dev-dep. **188 tests
passing** across lexer, parser, IR, passes (10 fusion + 8 dropout +
5 pipeline-level), profile codegen, CLI smoke (7), reference-validation,
and FFI integration. `cargo build --workspace`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo fmt --all -- --check` are clean.
CI green.

The immediate next step is **Milestone 5c — M5 close-out documentation**:
update `docs/profile_guide/arm64.md` with the bias-aware fusion section,
`--no-passes` / `--passes` flag documentation, and the EliminateDropout
removal note; update `PROJECT_SPEC.md`'s milestones table to mark M5
fully complete. M5c is docs-only (no code changes), single-commit scope.
```

- [ ] **Step 4: Commit closeout**

```bash
git add CLAUDE.md DEVLOG.md
git status
git commit -m "$(cat <<'EOF'
chore(m5b): close Milestone 5b — bias-aware fusion + EliminateDropout shipped

Per spec §10 acceptance criteria — all met:
- cargo build/clippy/fmt --check clean across workspace
- 188 tests passing (baseline 173 + 15 new: 1 bias-aware unit
  + 8 EliminateDropout unit + 1 pipeline-order end-to-end + 4 CLI
  smoke + 1 FFI integration)
- CLI smoke positive (default fused, --no-passes, --passes filter,
  --passes invalid, mutually-exclusive) all behave per spec
- Stdout/stderr discipline preserved from M5a
- All M3 fixtures + M4a fixture compile under both modes; M4b/M5a
  integration tests (mixed_args_runs_correctly, classifier-runs,
  fused_vs_unfused_classifier) continue to pass
- fused_vs_unfused_mixed_args_match_numerically confirms bit-exact
  equivalence for bias-aware case
- pipeline_eliminates_dropout_before_fusing_linear_relu proves
  order-dependency end-to-end
- --no-fuse fully removed (verified by grep — no shims, no aliases)
- Module-level doc-comment in compiler::passes already covers the
  pass infrastructure (M5a artefact)

DEVLOG documents:
- Pre-decided architectural calls from §4 of the spec
- "Three strikes then refactor" deferral for the shared helper
- M5b → M5c slicing (docs-only milestone for M5 close-out)

CLAUDE.md Current Status reflects M5b complete; M5c (profile guide
doc + PROJECT_SPEC close-out) as next.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Done. What's next?

After Task 7, M5b is complete by spec §10 acceptance criteria:

1. ✅ `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all -- --check` — all exit 0.
2. ✅ Test count = 188 (baseline 173 + 15 new) — meets `≥ baseline + new committed`.
3. ✅ CLI smoke covers all 5 cases (default, --no-passes, --passes filter, --passes invalid, mutually exclusive).
4. ✅ `fused_vs_unfused_mixed_args_match_numerically` passes — bit-exact bias-aware fusion equivalence.
5. ✅ `pipeline_eliminates_dropout_before_fusing_linear_relu` passes — end-to-end order-dependency proof.
6. ✅ `--no-fuse` fully removed (verified by `grep -rn "no_fuse\|--no-fuse" nflc/src/`).
7. ✅ M5a `fused_vs_unfused_classifier_match_numerically` continues to pass (regression check).
8. ✅ DEVLOG entry written.
9. ✅ CLAUDE.md "Current Status" updated.

**After all tasks pass:** push `claude/m5b-bias-aware-fusion` and open a PR against `main`. Title suggestion: `Implement Milestone 5b: bias-aware fusion + EliminateDropout pass + --passes CLI filter`. After merge, M5b closes; M5c (docs-only close-out for M5) begins.
