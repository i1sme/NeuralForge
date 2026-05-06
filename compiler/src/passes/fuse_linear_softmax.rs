//! `FuseLinearSoftmax` UIR pass.
//!
//! Fuses `linear → softmax` and `linear[bias=true] → softmax` patterns
//! by appending `PostOp::SoftmaxRow` to the Linear's `fused_post_ops`
//! and removing the Softmax node. The arm64 profile's RowWise emit
//! branch (see `arm64.md` §4.10) consumes the fused result.
//!
//! See spec §5 for the full victim criteria.

use super::{PassError, UirPass};
use crate::ir::types::{NodeKind, PostOp};
use crate::ir::StdOp;
use crate::{Uir, UirModel};

pub struct FuseLinearSoftmax;

impl UirPass for FuseLinearSoftmax {
    fn name(&self) -> &str {
        "fuse_linear_softmax"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(fuse_one_model(model.clone())?);
        }
        Ok(Uir { models: new_models })
    }
}

/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this. Violations cause `id_map[…]` panics inside
/// `compiler::passes::rewriter::rewrite_model`, not a `PassError` —
/// defensive checks would be belt-and-suspenders for an invariant
/// the type system can't (yet) express.
fn fuse_one_model(model: UirModel) -> Result<UirModel, PassError> {
    let mut plan = super::rewriter::RewritePlan::new(&model);

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
        let NodeKind::Op {
            op: StdOp::Linear,
            fused_post_ops,
            ..
        } = &model.nodes[linear_id].kind
        else {
            continue;
        };
        if !fused_post_ops.is_empty() {
            continue; // No double-fusion.
        }
        if *plan.consumer_count.get(&linear_id).unwrap_or(&0) != 1 {
            continue; // Linear must have exactly one consumer (this Softmax).
        }
        plan.victims.insert(softmax_id, linear_id);
        plan.producer_post_ops
            .entry(linear_id)
            .or_default()
            .push(PostOp::SoftmaxRow);
    }

    Ok(super::rewriter::rewrite_model(plan, model))
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

    #[test]
    fn leaves_linear_dropout_softmax_chain_untouched() {
        use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
        use crate::ir::types::NodeKind;
        use crate::ir::StdOp;
        use crate::{Uir, UirModel};

        // Construct: Input → Linear → Dropout → Softmax.
        // Run ONLY FuseLinearSoftmax (NOT default_pipeline — that would
        // EliminateDropout first and remove the blocking Dropout).
        // Spec §8 invariant 6 (arm64.md §4.10 in M6): Linear's sole
        // consumer is Dropout, not Softmax → criterion 2 fails → no
        // fusion happens.
        let model = UirModel {
            name: "M".into(),
            nodes: vec![
                input_node("x", vec![2, 3]),
                op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
                op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 2]),
                op_node(StdOp::Softmax, vec![2], vec![], vec![2, 2]),
            ],
            inputs: vec![0],
            output: 3,
            source_span: crate::ast::Span::new(1, 1),
        };
        let uir = Uir {
            models: vec![model],
        };

        let out = super::FuseLinearSoftmax.run(&uir).expect("pass ok");
        let m = &out.models[0];

        // Untouched: 4 nodes preserved, Linear's fused_post_ops empty,
        // Softmax-node still exists at index 3.
        assert_eq!(m.nodes.len(), 4);
        let NodeKind::Op { fused_post_ops, .. } = &m.nodes[1].kind else {
            panic!("expected Op at index 1")
        };
        assert!(fused_post_ops.is_empty(), "Linear should not be fused");
        assert!(matches!(
            m.nodes[3].kind,
            NodeKind::Op {
                op: StdOp::Softmax,
                ..
            }
        ));
    }
}
