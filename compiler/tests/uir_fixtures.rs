//! End-to-end integration tests for the UIR builder.
//!
//! One submodule per fixture; `mod negative` for cross-cutting rejection cases.

mod tiny_mlp {
    use nflc::*;

    #[test]
    fn tiny_mlp_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/tiny_mlp.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 1);
        let m = &uir.models[0];
        assert_eq!(m.name, "TinyMLP");

        // 3 nodes total: input x, op linear, op softmax.
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 2);

        // Node 0: Input "x", Tensor[8, 4]
        assert!(matches!(&m.nodes[0].kind, NodeKind::Input { name } if name == "x"));
        assert_eq!(m.nodes[0].ty.shape.0, vec![8, 4]);

        // Node 1: Linear[2], operands=[0], shape Tensor[8, 2]
        let NodeKind::Op { op, operands, attrs } = &m.nodes[1].kind else { panic!() };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(operands.as_slice(), &[0]);
        assert_eq!(m.nodes[1].ty.shape.0, vec![8, 2]);
        let AttrValue::Integer(out_dim) = attrs[0].value else { panic!() };
        assert_eq!(out_dim, 2);
        assert_eq!(attrs[0].name, "out_dim");

        // Node 2: Softmax, operands=[1], shape Tensor[8, 2]
        let NodeKind::Op { op, operands, .. } = &m.nodes[2].kind else { panic!() };
        assert_eq!(*op, StdOp::Softmax);
        assert_eq!(operands.as_slice(), &[1]);
        assert_eq!(m.nodes[2].ty.shape.0, vec![8, 2]);
    }

    #[test]
    fn unknown_op_errors() {
        let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> mystery\n";
        let ast = parse(src).expect("parses");
        let err = ir::build(&ast).expect_err("must fail");
        assert!(matches!(err.kind, BuildErrorKind::UnknownOp { .. }));
    }

    #[test]
    fn unknown_dim_errors() {
        let src = "model X [batch=8]:\n    x: Tensor[zzz, 4]\n    x -> softmax\n";
        let ast = parse(src).expect("parses");
        let err = ir::build(&ast).expect_err("must fail");
        assert!(matches!(err.kind, BuildErrorKind::UnknownDim { .. }));
    }

    #[test]
    fn model_has_no_pipeline_errors() {
        let src = "model X [a=1]:\n    x: Tensor[a, 1]\n";
        let ast = parse(src).expect("parses");
        let err = ir::build(&ast).expect_err("must fail");
        assert!(matches!(err.kind, BuildErrorKind::ModelHasNoPipeline { .. }));
    }
}
