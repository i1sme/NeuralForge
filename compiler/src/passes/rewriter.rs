//! Shared 3-step rebuild helper for UIR passes (M7).
//!
//! Three passes (`EliminateDropout`, `FuseLinearRelu`,
//! `FuseLinearSoftmax`) share an identical rebuild skeleton:
//!
//!   1. Identify victims (nodes that disappear in the new UIR).
//!   2. Rebuild + remap (clone non-victims, remap operand IDs through
//!      an id_map, optionally mutate kept producers, redirect victim
//!      references to other nodes' new IDs).
//!   3. Remap model.inputs and model.output through the same id_map.
//!
//! This module factors that skeleton out. Callers build a
//! `RewritePlan` (declared data, no closures), then call
//! `rewrite_model(plan, model)` which executes the plan and returns
//! a fresh `UirModel`.
//!
//! Plan-as-data design rationale: see spec §4.1 (rejected closure
//! and trait alternatives) and §4.5 (plain-`UirModel` return,
//! preconditions enforced by panic).

use crate::ir::types::{NodeKind, PostOp};
use crate::{NodeId, UirModel};
use std::collections::HashMap;

/// A plan for rewriting a `UirModel`: which nodes disappear (and
/// what their references redirect to), plus which surviving nodes
/// get post-op mutations.
///
/// Build a plan with `RewritePlan::new(&model)` (which precomputes
/// `consumer_count`), populate `victims` and `producer_post_ops`
/// during your pass's victim-identification logic, then hand the
/// plan to `rewrite_model(plan, model)`.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct RewritePlan {
    /// node_id → number of consumers. Counts both `NodeKind::Op`
    /// operands referencing this node, and an extra `+1` if
    /// `model.output == node_id`. Precomputed by `new()`; passes
    /// that don't need consumer counts (e.g. `EliminateDropout`)
    /// simply don't reference this field.
    pub consumer_count: HashMap<NodeId, usize>,

    /// victim_id → redirect-target's old NodeId. The rewriter sets
    /// `id_map[victim] = id_map[victims[victim]]` when it encounters
    /// the victim during rebuild. The redirect target MUST appear
    /// earlier in topological order than the victim
    /// (target_old_id < victim_old_id).
    pub victims: HashMap<NodeId, NodeId>,

    /// producer_id → post-ops to push onto the producer's
    /// `fused_post_ops` AFTER its operands are remapped. Producer
    /// must be a non-victim `NodeKind::Op` node. Multiple PostOps
    /// push in vec order.
    pub producer_post_ops: HashMap<NodeId, Vec<PostOp>>,
}

impl RewritePlan {
    #[allow(dead_code)]
    pub(crate) fn new(model: &UirModel) -> Self {
        let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
        for node in &model.nodes {
            if let NodeKind::Op { operands, .. } = &node.kind {
                for &op_id in operands {
                    *consumer_count.entry(op_id).or_insert(0) += 1;
                }
            }
        }
        *consumer_count.entry(model.output).or_insert(0) += 1;

        Self {
            consumer_count,
            victims: HashMap::new(),
            producer_post_ops: HashMap::new(),
        }
    }
}

/// Execute a rewrite plan against `model`. Returns a fresh `UirModel`
/// with renumbered NodeIds; both inputs are consumed.
///
/// Preconditions (caller's responsibility — violations cause
/// `id_map[…]` panics, NOT a `Result::Err`):
///   - `model.nodes` is in topological order (operand IDs <
///     consumer IDs). `compiler::ir::build` guarantees this.
///   - For every `(victim, target)` in `plan.victims`,
///     `target_old_id < victim_old_id`.
///   - Every key in `plan.producer_post_ops` is NOT a key in
///     `plan.victims`.
///   - Every key in `plan.producer_post_ops` references a node with
///     `NodeKind::Op` kind.
#[allow(dead_code)]
pub(crate) fn rewrite_model(plan: RewritePlan, model: UirModel) -> UirModel {
    // consumer_count is used by callers during plan population
    // (e.g. fuse_linear_relu's single-consumer victim guard); the
    // rewrite step itself doesn't need it.
    let RewritePlan {
        consumer_count: _,
        victims,
        producer_post_ops,
    } = plan;

    let mut new_nodes = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.into_iter().enumerate() {
        if let Some(&target_old_id) = victims.get(&old_id) {
            // Victim: redirect its id_map entry to target's new id.
            let target_new_id = id_map[&target_old_id];
            id_map.insert(old_id, target_new_id);
            continue;
        }

        // Non-victim: take ownership, remap operands, optionally
        // append PostOps, push to new_nodes.
        let mut new_node = node;
        if let NodeKind::Op {
            operands,
            fused_post_ops,
            ..
        } = &mut new_node.kind
        {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
            if let Some(extras) = producer_post_ops.get(&old_id) {
                fused_post_ops.extend(extras.iter().copied());
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    let new_inputs: Vec<NodeId> = model.inputs.iter().map(|id| id_map[id]).collect();
    let new_output = id_map[&model.output];

    UirModel {
        name: model.name,
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
        source_span: model.source_span,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rewrite_model_with_empty_plan_is_identity() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
        use crate::ir::types::NodeKind;
        use crate::ir::StdOp;
        use crate::UirModel;

        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 1,
            source_span: crate::ast::Span::new(1, 1),
        };
        let plan = super::RewritePlan::new(&model);
        let out = super::rewrite_model(plan, model);

        assert_eq!(out.nodes.len(), 2);
        assert_eq!(out.output, 1);
        assert_eq!(out.inputs, vec![0]);
        assert!(matches!(out.nodes[0].kind, NodeKind::Input { .. }));
        let NodeKind::Op { op, .. } = &out.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert!(matches!(op, StdOp::Linear));
    }

    #[test]
    fn rewrite_model_drops_victim_and_redirects_consumers() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
        use crate::ir::types::NodeKind;
        use crate::ir::StdOp;
        use crate::UirModel;

        // Topology: Input(0) → A=Linear(1) → B=Dropout(2) → C=Linear(3).
        // Plan: victim B(2) → redirect to A(1).
        // Expected: B disappears (3 nodes total), C's operand remaps to
        // A's new id (1 since Input=0, A=1).
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Dropout, vec![1], vec![], vec![2, 2]),
                op_node(StdOp::Linear, vec![2], vec![out_dim_attr(2)], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 3,
            source_span: crate::ast::Span::new(1, 1),
        };
        let mut plan = super::RewritePlan::new(&model);
        plan.victims.insert(2, 1); // Dropout(2) → redirect to A(1)
        let out = super::rewrite_model(plan, model);

        assert_eq!(out.nodes.len(), 3); // Dropout dropped
        assert_eq!(out.output, 2); // C's new id (Input=0, A=1, C=2)
        let NodeKind::Op {
            operands: c_ops, ..
        } = &out.nodes[2].kind
        else {
            panic!("expected Op at index 2")
        };
        assert_eq!(c_ops, &vec![1usize]); // C's operand remapped to A's new id (1)
    }

    #[test]
    fn rewrite_model_pushes_post_ops_to_producer() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
        use crate::ir::types::{NodeKind, PostOp};
        use crate::ir::StdOp;
        use crate::UirModel;

        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: crate::ast::Span::new(1, 1),
        };
        let mut plan = super::RewritePlan::new(&model);
        plan.victims.insert(2, 1); // Relu → redirect to Linear
        plan.producer_post_ops
            .entry(1)
            .or_default()
            .push(PostOp::Relu);
        let out = super::rewrite_model(plan, model);

        assert_eq!(out.nodes.len(), 2);
        let NodeKind::Op { fused_post_ops, .. } = &out.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    }

    #[test]
    fn rewrite_model_remaps_model_inputs_and_output() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
        use crate::ir::StdOp;
        use crate::UirModel;

        // Topology: Input(0), Input(1), A=Linear(2) using Input(0),
        //   B=Dropout(3) using A. Inputs=[0, 1], output=3 (Dropout victim).
        // Plan: victim B(3) → redirect to A(2).
        // Expected: out.inputs unchanged ([0, 1] — Input nodes are not
        //   victims), out.output = 2 (A's new id since Input(0)=0,
        //   Input(1)=1, A=2).
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                input_node("y", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Dropout, vec![2], vec![], vec![2, 2]),
            ],
            inputs: vec![0, 1],
            output: 3,
            source_span: crate::ast::Span::new(1, 1),
        };
        let mut plan = super::RewritePlan::new(&model);
        plan.victims.insert(3, 2); // Dropout → A
        let out = super::rewrite_model(plan, model);

        assert_eq!(out.nodes.len(), 3); // Dropout dropped
        assert_eq!(out.inputs, vec![0, 1]); // Input nodes survive at same new ids
        assert_eq!(out.output, 2); // A's new id
    }

    #[test]
    fn rewrite_plan_new_counts_consumers_correctly() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
        use crate::ir::StdOp;
        use crate::UirModel;

        // Topology: Input(0) → A=Linear(1), then both B=Dropout(2) and
        // C=Relu(3) consume A. model.output = C(3).
        // Expected counts:
        //   Input(0): 1 (consumed by A's operand list)
        //   A(1): 2 (consumed by B and C)
        //   B(2): absent from map (no consumers — B is an orphan node)
        //   C(3): 1 (model.output += 1)
        //
        // B is intentionally orphan to demonstrate the absent-entry
        // contract — the consumer_count map only has entries for
        // nodes referenced by at least one operand list or as
        // model.output. Graph validity is not required by new().
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Dropout, vec![1], vec![], vec![2, 2]),
                op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 3,
            source_span: crate::ast::Span::new(1, 1),
        };
        let plan = super::RewritePlan::new(&model);

        assert_eq!(plan.consumer_count.get(&0), Some(&1));
        assert_eq!(plan.consumer_count.get(&1), Some(&2));
        assert_eq!(plan.consumer_count.get(&2), None); // absent
        assert_eq!(plan.consumer_count.get(&3), Some(&1));
    }
}
