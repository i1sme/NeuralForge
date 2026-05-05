//! `FuseLinearSoftmax` UIR pass.
//!
//! Fuses `linear → softmax` and `linear[bias=true] → softmax` patterns
//! by appending `PostOp::SoftmaxRow` to the Linear's `fused_post_ops`
//! and removing the Softmax node. The arm64 profile's RowWise emit
//! branch (see `arm64.md` §4.10) consumes the fused result.
//!
//! See spec §5 for the full victim criteria.

use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind, PostOp};
use crate::ir::StdOp;
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};

pub struct FuseLinearSoftmax;

impl UirPass for FuseLinearSoftmax {
    fn name(&self) -> &str {
        "fuse_linear_softmax"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model)?);
        }
        Ok(Uir { models: new_models })
    }
}

/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this. Violations cause `id_map[…]` panics in step 3,
/// not a `PassError` — defensive checks would be belt-and-suspenders
/// for an invariant the type system can't (yet) express.
fn fuse_one_model(model: &UirModel) -> Result<UirModel, PassError> {
    // Step 1: consumer counts.
    let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
    for node in &model.nodes {
        if let NodeKind::Op { operands, .. } = &node.kind {
            for &op_id in operands {
                *consumer_count.entry(op_id).or_insert(0) += 1;
            }
        }
    }
    *consumer_count.entry(model.output).or_insert(0) += 1;

    // Step 2: identify victims (Softmax nodes that fold into producer Linear).
    let mut victim_to_producer: HashMap<NodeId, NodeId> = HashMap::new();
    for (softmax_id, softmax_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Softmax,
            operands,
            ..
        } = &softmax_node.kind
        else {
            continue;
        };
        if operands.len() != 1 {
            continue;
        }
        let linear_id = operands[0];
        let linear_node = &model.nodes[linear_id];
        let NodeKind::Op {
            op: StdOp::Linear,
            fused_post_ops,
            ..
        } = &linear_node.kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion.
        }
        if *consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Softmax).
        }
        victim_to_producer.insert(softmax_id, linear_id);
    }

    let victims: HashSet<NodeId> = victim_to_producer.keys().copied().collect();
    let producers_of_victims: HashSet<NodeId> = victim_to_producer.values().copied().collect();

    // Step 3: build new model.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.iter().enumerate() {
        if victims.contains(&old_id) {
            // Skip pushing; map old victim id → producer's new id.
            let producer_old_id = victim_to_producer[&old_id];
            let producer_new_id = id_map[&producer_old_id];
            id_map.insert(old_id, producer_new_id);
            continue;
        }

        // Clone + remap operands.
        let mut new_node = node.clone();
        if let NodeKind::Op {
            operands,
            fused_post_ops,
            ..
        } = &mut new_node.kind
        {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
            if producers_of_victims.contains(&old_id) {
                fused_post_ops.push(PostOp::SoftmaxRow);
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    // Step 4: remap inputs + output.
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
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::{Uir, UirModel};

    #[test]
    fn fuses_linear_softmax_no_bias() {
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: crate::ast::Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        assert_eq!(m.nodes.len(), 2);
        let NodeKind::Op {
            op, fused_post_ops, ..
        } = &m.nodes[1].kind
        else {
            panic!("expected Op node at index 1")
        };
        assert!(matches!(op, StdOp::Linear));
        assert_eq!(fused_post_ops, &vec![PostOp::SoftmaxRow]);
        assert_eq!(m.output, 1);
    }

    #[test]
    fn fuses_linear_softmax_with_bias() {
        use crate::ir::types::{AttrValue, OpAttr};

        let bias_attr = OpAttr {
            name: "bias".into(),
            value: AttrValue::Symbol("true".into()),
        };
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(
                    StdOp::Linear,
                    vec![0],
                    vec![out_dim_attr(2), bias_attr],
                    vec![2, 2],
                ),
                op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: crate::ast::Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        assert_eq!(m.nodes.len(), 2);
        let NodeKind::Op {
            op,
            attrs,
            fused_post_ops,
            ..
        } = &m.nodes[1].kind
        else {
            panic!("expected Op node at index 1")
        };
        assert!(matches!(op, StdOp::Linear));
        // bias attr is preserved on the fused Linear
        assert!(attrs.iter().any(|a| a.name == "bias"));
        assert_eq!(fused_post_ops, &vec![PostOp::SoftmaxRow]);
    }

    #[test]
    fn does_not_fuse_when_post_ops_already_present() {
        // Construct a Linear that already has fused_post_ops = [Relu], then a Softmax consumer.
        let mut linear = op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]);
        if let NodeKind::Op { fused_post_ops, .. } = &mut linear.kind {
            fused_post_ops.push(PostOp::Relu);
        }

        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                linear,
                op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: crate::ast::Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
        assert!(matches!(
            m.nodes[2].kind,
            NodeKind::Op {
                op: StdOp::Softmax,
                ..
            }
        ));
    }

    #[test]
    fn does_not_fuse_multi_consumer_linear() {
        // Linear feeds both Softmax and Relu — multi-consumer, must not fuse.
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Softmax, vec![1], vec![], vec![2, 2]),
                op_node(StdOp::Relu, vec![1], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 2,
            source_span: crate::ast::Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        assert_eq!(m.nodes.len(), 4);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert!(
            fused_post_ops.is_empty(),
            "Linear with multi-consumer must not be fused"
        );
    }

    #[test]
    fn identity_when_no_softmax() {
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
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        assert_eq!(m.nodes.len(), 3);
        assert!(matches!(
            m.nodes[2].kind,
            NodeKind::Op {
                op: StdOp::Relu,
                ..
            }
        ));
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert!(fused_post_ops.is_empty());
    }
}
