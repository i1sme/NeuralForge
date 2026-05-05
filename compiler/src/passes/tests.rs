//! Pipeline-level tests for `compiler::passes`.

use super::{default_pipeline, run_pipeline, PassError, UirPass};
use crate::Uir;

/// Synthetic identity pass for testing the pipeline mechanics without
/// depending on any specific transformation.
struct IdentityPass {
    name: &'static str,
}

impl UirPass for IdentityPass {
    fn name(&self) -> &str {
        self.name
    }
    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        Ok(uir.clone())
    }
}

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

#[test]
fn run_pipeline_threads_uir_through_passes() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let passes: Vec<Box<dyn UirPass>> = vec![
        Box::new(IdentityPass { name: "id_a" }),
        Box::new(IdentityPass { name: "id_b" }),
    ];

    let out = run_pipeline(&uir, &passes).expect("pipeline ok");
    // Identity passes preserve model count + node count.
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].nodes.len(), uir.models[0].nodes.len());
}

#[test]
fn empty_pipeline_returns_input_clone() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let out = run_pipeline(&uir, &[]).expect("empty pipeline ok");
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].name, uir.models[0].name);
}

/// Synthetic always-failing pass: lets us verify pipeline halts on
/// `Err` and propagates the error unchanged. Without this test, a
/// future refactor of `run_pipeline` could accidentally swallow
/// errors and only Task 4's tests would (incidentally) catch it.
struct FailPass;

impl UirPass for FailPass {
    fn name(&self) -> &str {
        "fail"
    }
    fn run(&self, _uir: &Uir) -> Result<Uir, PassError> {
        Err(PassError::InvalidInput {
            pass: "fail".into(),
            reason: "synthetic".into(),
            span: crate::ast::Span::new(1, 1),
        })
    }
}

#[test]
fn pipeline_halts_on_first_error_and_propagates() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let passes: Vec<Box<dyn UirPass>> = vec![
        Box::new(FailPass),
        Box::new(IdentityPass {
            name: "should_not_run",
        }),
    ];
    let err = run_pipeline(&uir, &passes).expect_err("expected pipeline error");
    match err {
        PassError::InvalidInput { pass, reason, .. } => {
            assert_eq!(pass, "fail");
            assert_eq!(reason, "synthetic");
        }
    }
}

#[test]
fn pipeline_eliminates_dropout_before_fusing_linear_relu() {
    // Load-bearing test for spec §4.1: hand-build a synthetic UIR
    // `linear → dropout → relu` and run the full default pipeline.
    // Expected: 2 nodes (input + fused linear with fused_post_ops==[Relu]).
    // This proves end-to-end that EliminateDropout runs first AND
    // that FuseLinearRelu picks up the resulting linear→relu pattern.
    use crate::ir::test_utils::{input_node, op_node, out_dim_attr, rate_attr};
    use crate::ir::types::{NodeKind, PostOp};
    use crate::ir::StdOp;
    use crate::UirModel;

    let model = UirModel {
        name: "M".into(),
        nodes: vec![
            input_node("x", vec![2, 3]),
            op_node(StdOp::Linear, vec![0], vec![out_dim_attr(2)], vec![2, 2]),
            op_node(StdOp::Dropout, vec![1], vec![rate_attr(0.5)], vec![2, 2]),
            op_node(StdOp::Relu, vec![2], vec![], vec![2, 2]),
        ],
        inputs: vec![0],
        output: 3, // relu
        source_span: crate::ast::Span::new(1, 1),
    };
    let uir = Uir {
        models: vec![model],
    };

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

    // n0 is still the original Input node (renumber is identity here
    // because no input nodes were victims). Defensive check — guards
    // against a future pass accidentally reordering nodes.
    assert!(
        matches!(m.nodes[0].kind, NodeKind::Input { .. }),
        "n0 must be the Input node after pipeline; got: {:?}",
        m.nodes[0].kind
    );

    // The fused linear has fused_post_ops == [Relu].
    let NodeKind::Op {
        op, fused_post_ops, ..
    } = &m.nodes[1].kind
    else {
        panic!("expected Op at n1");
    };
    assert!(matches!(op, StdOp::Linear));
    assert_eq!(fused_post_ops, &vec![PostOp::Relu]);

    // model.output points at the fused linear.
    assert_eq!(m.output, 1);
    assert_eq!(m.inputs, vec![0]);
}
