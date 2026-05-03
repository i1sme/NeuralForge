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

mod classifier {
    use nflc::*;

    #[test]
    fn classifier_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/classifier.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 1);
        let m = &uir.models[0];
        assert_eq!(m.name, "Classifier");

        // Body: 1 input + 7 ops (linear, relu, dropout, linear, relu, linear, softmax)
        // = 8 nodes.
        assert_eq!(m.nodes.len(), 8);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 7);

        // Input: Tensor[32, 784] (batch=32, input=784).
        assert_eq!(m.nodes[0].ty.shape.0, vec![32, 784]);

        // Final output: Tensor[32, 10] (output=10).
        assert_eq!(m.nodes[7].ty.shape.0, vec![32, 10]);

        // Spot-check the dropout node (n3) has its named float arg.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[3].kind else { panic!() };
        assert_eq!(*op, StdOp::Dropout);
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "rate");
        let AttrValue::Float(rate) = attrs[0].value else { panic!() };
        assert!((rate - 0.2).abs() < 1e-9);
    }
}

mod pipeline_styles {
    use nflc::*;

    #[test]
    fn pipeline_styles_three_models() {
        let src = std::fs::read_to_string("../tests/fixtures/pipeline_styles.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        assert_eq!(uir.models.len(), 3);
        assert_eq!(uir.models[0].name, "SingleLine");
        assert_eq!(uir.models[1].name, "PerStepWrap");
        assert_eq!(uir.models[2].name, "MixedWrap");

        // All three models have the same pipeline shape:
        //   x: Tensor[batch=4, input=10]
        //   x -> linear[8] -> relu -> linear[output=2] -> softmax
        // = 1 input + 4 ops = 5 nodes.
        for m in &uir.models {
            assert_eq!(m.nodes.len(), 5, "model {}", m.name);
            assert_eq!(m.inputs, vec![0]);
            assert_eq!(m.output, 4);
            assert_eq!(m.nodes[0].ty.shape.0, vec![4, 10]);
            assert_eq!(m.nodes[4].ty.shape.0, vec![4, 2]);
        }
    }
}

mod comments {
    use nflc::*;

    #[test]
    fn comments_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/comments.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];
        assert_eq!(m.name, "Commented");
        // Body: 1 input + 4 ops (linear[16], relu, linear[output=2], softmax) = 5 nodes.
        assert_eq!(m.nodes.len(), 5);
        assert_eq!(m.inputs, vec![0]);
        assert_eq!(m.output, 4);
        assert_eq!(m.nodes[4].ty.shape.0, vec![4, 2]);
    }
}

mod mixed_args {
    use nflc::*;

    #[test]
    fn mixed_args_builds() {
        let src = std::fs::read_to_string("../tests/fixtures/mixed_args.nfl")
            .expect("fixture readable");
        let ast = parse(&src).expect("must parse");
        let uir = ir::build(&ast).expect("must build");

        let m = &uir.models[0];

        // First op: linear[16, bias=true] — positional Integer + named Symbol.
        let NodeKind::Op { op, attrs, .. } = &m.nodes[1].kind else { panic!() };
        assert_eq!(*op, StdOp::Linear);
        assert_eq!(attrs.len(), 2);
        // Positional out_dim = 16
        assert_eq!(attrs[0].name, "out_dim");
        assert_eq!(attrs[0].value, AttrValue::Integer(16));
        // Named bias = true (Symbol)
        assert_eq!(attrs[1].name, "bias");
        assert_eq!(attrs[1].value, AttrValue::Symbol("true".into()));
    }
}
