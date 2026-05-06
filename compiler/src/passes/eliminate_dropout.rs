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
use crate::ir::types::NodeKind;
use crate::ir::StdOp;
use crate::{Uir, UirModel};

pub struct EliminateDropout;

impl UirPass for EliminateDropout {
    fn name(&self) -> &str {
        "eliminate_dropout"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(eliminate_one_model(model.clone())?);
        }
        Ok(Uir { models: new_models })
    }
}

/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this.
///
/// Implementation: delegates the rebuild skeleton (identify victims →
/// rebuild with remap → remap inputs/output) to
/// `compiler::passes::rewriter`. Each Dropout becomes a victim that
/// redirects to its sole operand; no producer mutation. See spec
/// `docs/superpowers/specs/2026-05-06-m7-rebuild-helper-design.md`
/// §4 / §5 for the helper's design.
fn eliminate_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = super::rewriter::RewritePlan::new(&model);

    for (id, node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Dropout,
            operands,
            ..
        } = &node.kind
        else {
            continue;
        };
        // NFL grammar (§stdlib::Signature for `dropout`) guarantees
        // exactly one operand. Any future grammar / hand-built UIR
        // that violates this invariant will panic at the operands[0]
        // index access.
        debug_assert_eq!(
            operands.len(),
            1,
            "Dropout must have exactly one operand (NFL grammar invariant)"
        );
        plan.victims.insert(id, operands[0]);
    }

    Ok(super::rewriter::rewrite_model(plan, model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
    use crate::Uir;

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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2, // dropout
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 2, "dropout node should be removed");
        assert_eq!(m.output, 1, "output should remap to linear's new id");
        assert_eq!(m.inputs, vec![0]);
        // Surviving linear has its operand still pointing at input (id 0).
        let NodeKind::Op { operands, .. } = &m.nodes[1].kind else {
            panic!()
        };
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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(3)], vec![2, 3]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.3)], vec![2, 3]),
                op_node(StdOp::Softmax, vec![2], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 3, // softmax
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3, "dropout removed; 3 survivors");
        // Softmax (now id 2) reads linear (now id 1) directly.
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else {
            panic!()
        };
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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(3)], vec![2, 3]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.2)], vec![2, 3]),
                op_node(StdOp::Dropout, vec![2], vec![rate_attr(0.4)], vec![2, 3]),
                op_node(StdOp::Relu, vec![3], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 4, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3, "both dropouts removed");
        // Relu (now id 2) reads linear (now id 1) directly — both dropouts collapsed.
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else {
            panic!()
        };
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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(3)], vec![2, 3]),
                op_node(StdOp::Relu, vec![1], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 2);
        // NodeIds renumbered 0..N (identity here since no nodes were dropped).
        let NodeKind::Op { op: op1, .. } = &m.nodes[1].kind else {
            panic!()
        };
        assert!(matches!(op1, StdOp::Linear));
        let NodeKind::Op {
            op: op2, operands, ..
        } = &m.nodes[2].kind
        else {
            panic!()
        };
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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(3)], vec![2, 3]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 3]),
                op_node(StdOp::Relu, vec![2], vec![], vec![2, 3]), // consumer A: reads dropout
                op_node(StdOp::Softmax, vec![2], vec![], vec![2, 3]), // consumer B: reads dropout
            ],
            inputs: vec![0],
            output: 3, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];
        assert_eq!(m.nodes.len(), 4, "dropout removed; 4 survivors");
        // Both relu and softmax now read linear (id 1) directly.
        let NodeKind::Op {
            op: op_a,
            operands: ops_a,
            ..
        } = &m.nodes[2].kind
        else {
            panic!()
        };
        let NodeKind::Op {
            op: op_b,
            operands: ops_b,
            ..
        } = &m.nodes[3].kind
        else {
            panic!()
        };
        assert!(matches!(op_a, StdOp::Relu));
        assert!(matches!(op_b, StdOp::Softmax));
        assert_eq!(
            ops_a,
            &vec![1usize],
            "relu should remap dropout-operand to linear"
        );
        assert_eq!(
            ops_b,
            &vec![1usize],
            "softmax should remap dropout-operand to linear"
        );
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
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(3)], vec![2, 3]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.1)], vec![2, 3]),
                op_node(StdOp::Relu, vec![2], vec![], vec![2, 3]),
            ],
            inputs: vec![0],
            output: 3, // relu
            source_span: Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = EliminateDropout.run(&uir).expect("ok");
        let m = &out.models[0];

        // Input id 0 was Input — preserved as 0 in output.
        assert_eq!(
            m.inputs,
            vec![0],
            "inputs should remap through id_map (identity for input nodes)"
        );
        // Output was relu's old id 3 — should remap to relu's new id 2.
        assert_eq!(m.output, 2);
        // Verify the structure is intact.
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else {
            panic!()
        };
        assert!(matches!(op, StdOp::Relu));
        assert_eq!(operands, &vec![1usize]);
    }
}
