//! `linear → relu` fusion pass (spec §7).
//!
//! Finds nodes matching the pattern:
//!   Linear (no bias=true, fused_post_ops empty, single consumer)
//!     → Relu (any consumer count)
//!
//! Rewrites the graph:
//!   - Linear gets `fused_post_ops: vec![PostOp::Relu]`.
//!   - Relu node is removed; references to it are remapped to the fused
//!     Linear's new NodeId.
//!
//! Functional: returns a fresh Uir with renumbered NodeIds.

use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind, PostOp};
use crate::ir::{linear_has_bias, StdOp};
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};

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

    // Step 2: identify victims (Relu nodes that fold into producer Linear).
    let mut victim_to_producer: HashMap<NodeId, NodeId> = HashMap::new();
    for (relu_id, relu_node) in model.nodes.iter().enumerate() {
        let NodeKind::Op {
            op: StdOp::Relu,
            operands,
            ..
        } = &relu_node.kind
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
            attrs,
            fused_post_ops,
            ..
        } = &linear_node.kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion in M5a.
        }
        if linear_has_bias(attrs) {
            continue; // M5a scope: bias-aware fusion is M5b.
        }
        if *consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Relu).
        }
        victim_to_producer.insert(relu_id, linear_id);
    }

    let victims: HashSet<NodeId> = victim_to_producer.keys().copied().collect();
    // M5a invariant: each Linear in `victim_to_producer.values()` appears
    // exactly once (each Linear has at most one consuming Relu, by the
    // single-consumer guard). Hence pushing one `PostOp::Relu` per
    // matched producer is correct. M5b multi-victim fusion will need a
    // count-per-producer to push the right number of post-ops.
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
                fused_post_ops.push(PostOp::Relu);
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
    use crate::Uir;

    fn build(src: &str) -> Uir {
        let ast = crate::parse(src).expect("parse");
        crate::ir::build(&ast).expect("ir::build")
    }

    #[test]
    fn pass_name_is_stable() {
        assert_eq!(FuseLinearRelu.name(), "fuse_linear_relu");
    }

    #[test]
    fn empty_uir_passes_unchanged() {
        let uir = Uir { models: Vec::new() };
        let out = FuseLinearRelu.run(&uir).expect("ok");
        assert_eq!(out.models.len(), 0);
    }

    #[test]
    fn fuses_simple_linear_relu() {
        // Terminal: x -> linear[2] -> relu
        let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];

        // Original had 3 nodes (input, linear, relu); fused has 2 (input, fused-linear).
        assert_eq!(m.nodes.len(), 2, "expected 2 nodes; got: {:?}", m.nodes);

        // Node 1 is the fused Linear.
        let NodeKind::Op {
            op, fused_post_ops, ..
        } = &m.nodes[1].kind
        else {
            panic!("expected Op node");
        };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);

        // model.output points at the fused Linear.
        assert_eq!(m.output, 1);
    }

    #[test]
    fn does_not_fuse_when_linear_has_multiple_consumers() {
        // x -> linear[3] -> [softmax, relu]    (linear has 2 consumers)
        // Hand-built UIR (NFL grammar can't express shared producer).
        use crate::ast::Span;
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let linear_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr {
                    name: "out_dim".into(),
                    value: AttrValue::Integer(3),
                }],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Relu,
                operands: vec![1],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let softmax_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Softmax,
                operands: vec![1],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };

        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, linear_n, relu_n, softmax_n],
            inputs: vec![0],
            output: 3, // softmax is the terminal
            source_span: span,
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 4 nodes preserved (no fusion).
        assert_eq!(m.nodes.len(), 4);
        // Linear's fused_post_ops is still empty.
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!()
        };
        assert!(fused_post_ops.is_empty());
    }

    #[test]
    fn fuses_when_relu_has_multiple_consumers() {
        // x -> linear[3] -> relu -> [linear[2] (= consumer A), linear[2] (= consumer B)]
        // Linear has only Relu as consumer; Relu has 2 downstream.
        use crate::ast::Span;
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let linear_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr {
                    name: "out_dim".into(),
                    value: AttrValue::Integer(3),
                }],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Relu,
                operands: vec![1],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let consumer_a = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![2], // consumes relu
                attrs: vec![OpAttr {
                    name: "out_dim".into(),
                    value: AttrValue::Integer(2),
                }],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 2]),
            },
            source_span: span,
        };
        let consumer_b = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![2], // consumes relu (shared)
                attrs: vec![OpAttr {
                    name: "out_dim".into(),
                    value: AttrValue::Integer(2),
                }],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 2]),
            },
            source_span: span,
        };

        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, linear_n, relu_n, consumer_a, consumer_b],
            inputs: vec![0],
            output: 4, // consumer_b
            source_span: span,
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 4 nodes (relu removed).
        assert_eq!(m.nodes.len(), 4);
        // Both consumer_a and consumer_b operands now reference the fused linear (new id 1).
        let NodeKind::Op {
            operands: ca_ops, ..
        } = &m.nodes[2].kind
        else {
            panic!()
        };
        let NodeKind::Op {
            operands: cb_ops, ..
        } = &m.nodes[3].kind
        else {
            panic!()
        };
        assert_eq!(
            ca_ops,
            &vec![1usize],
            "consumer_a should remap to fused linear (id 1)"
        );
        assert_eq!(
            cb_ops,
            &vec![1usize],
            "consumer_b should remap to fused linear (id 1)"
        );
        // Fused linear has post-op set.
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!()
        };
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    }

    #[test]
    fn fuses_chain_independently() {
        // x: Tensor[b, 4] -> linear[8] -> relu -> linear[2] -> relu
        let uir = build(
            "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
        );
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // Original: input + linear[8] + relu + linear[2] + relu = 5 nodes.
        // After fusion: input + fused linear[8] + fused linear[2] = 3 nodes.
        assert_eq!(m.nodes.len(), 3);
        // Both Linears have fused_post_ops = [Relu].
        let NodeKind::Op {
            fused_post_ops: f1, ..
        } = &m.nodes[1].kind
        else {
            panic!()
        };
        let NodeKind::Op {
            fused_post_ops: f2, ..
        } = &m.nodes[2].kind
        else {
            panic!()
        };
        assert_eq!(f1, &vec![PostOp::Relu]);
        assert_eq!(f2, &vec![PostOp::Relu]);
    }

    #[test]
    fn does_not_fuse_when_linear_has_bias() {
        let uir =
            build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 3 nodes preserved.
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!()
        };
        assert!(fused_post_ops.is_empty());
    }

    #[test]
    fn does_not_fuse_when_linear_already_fused() {
        // Hand-build UIR where Linear ALREADY has fused_post_ops = [Relu], followed by another Relu.
        use crate::ast::Span;
        use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, Shape, Type};
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let pre_fused_linear = Node {
            kind: NodeKind::Op {
                op: StdOp::Linear,
                operands: vec![0],
                attrs: vec![OpAttr {
                    name: "out_dim".into(),
                    value: AttrValue::Integer(3),
                }],
                fused_post_ops: vec![PostOp::Relu], // already fused
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Relu,
                operands: vec![1],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, pre_fused_linear, relu_n],
            inputs: vec![0],
            output: 2,
            source_span: span,
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // 3 nodes preserved (no double-fusion).
        assert_eq!(m.nodes.len(), 3);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!()
        };
        // Still just one Relu in fused_post_ops (not [Relu, Relu]).
        assert_eq!(fused_post_ops, &vec![PostOp::Relu]);
    }

    #[test]
    fn does_not_fuse_when_relu_not_after_linear() {
        // Synthetic: softmax → relu (NFL grammar may not allow; we hand-build UIR).
        use crate::ast::Span;
        use crate::ir::types::{Node, NodeKind, Shape, Type};
        let span = Span::new(1, 1);

        let input_n = Node {
            kind: NodeKind::Input { name: "x".into() },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let softmax_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Softmax,
                operands: vec![0],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let relu_n = Node {
            kind: NodeKind::Op {
                op: StdOp::Relu,
                operands: vec![1],
                attrs: vec![],
                fused_post_ops: vec![],
            },
            ty: Type {
                name: "Tensor".into(),
                shape: Shape(vec![2, 3]),
            },
            source_span: span,
        };
        let model = UirModel {
            name: "M".into(),
            nodes: vec![input_n, softmax_n, relu_n],
            inputs: vec![0],
            output: 2,
            source_span: span,
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = FuseLinearRelu.run(&uir).expect("ok");
        // 3 nodes preserved (softmax → relu is not fusable; only Linear → Relu fuses in M5a).
        assert_eq!(out.models[0].nodes.len(), 3);
    }

    #[test]
    fn model_inputs_and_output_remapped() {
        // Simple: x -> linear[2] -> relu (terminal). After fusion: input(0) + fused_linear(1).
        let uir = build("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
        let out = FuseLinearRelu.run(&uir).expect("ok");
        let m = &out.models[0];
        // Old model.output was 2 (relu); after fusion, points at fused linear (new id 1).
        assert_eq!(m.output, 1);
        // Old model.inputs was [0] (input); preserved as [0].
        assert_eq!(m.inputs, vec![0]);
    }
}
