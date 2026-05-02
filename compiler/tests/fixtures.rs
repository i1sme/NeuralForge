//! Integration tests: parse the canonical fixtures and assert AST shape.
//!
//! Positive (5) and negative (7 — added in Task 19) live in the same file
//! under separate `mod`s for readability.

mod positive {
    use nflc::*;

    fn read_fixture(name: &str) -> String {
        let path = format!("../tests/fixtures/{name}");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
    }

    #[test]
    fn classifier() {
        let src = read_fixture("classifier.nfl");
        let nfl = parse(&src).expect("classifier.nfl must parse");

        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "Classifier");
        assert_eq!(m.params.len(), 3);
        assert_eq!(
            m.params.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["batch", "input", "output"],
        );
        assert_eq!(m.params[0].value, 32);
        assert_eq!(m.params[1].value, 784);
        assert_eq!(m.params[2].value, 10);

        assert_eq!(m.body.len(), 2);

        let ModelStmt::VariableDecl(v) = &m.body[0] else { panic!("expected VariableDecl") };
        assert_eq!(v.name, "x");
        assert_eq!(v.ty.name, "Tensor");
        assert_eq!(v.ty.dims, vec![Dim::Symbol("batch".into()), Dim::Symbol("input".into())]);

        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!("expected Pipeline") };
        assert_eq!(p.source, "x");
        assert_eq!(p.steps.len(), 7);
        assert_eq!(
            p.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["linear", "relu", "dropout", "linear", "relu", "linear", "softmax"],
        );
        // Positional first linear arg.
        assert_eq!(p.steps[0].args, vec![OpArg::Positional(ArgValue::Integer(512))]);
        // Named dropout arg.
        let OpArg::Named { name, value: ArgValue::Float(f) } = &p.steps[2].args[0] else {
            panic!("expected named float arg on dropout")
        };
        assert_eq!(name, "rate");
        assert!((f - 0.2).abs() < 1e-9);
        // Symbolic-dim positional on the last linear.
        assert_eq!(p.steps[5].args, vec![OpArg::Positional(ArgValue::Symbol("output".into()))]);
        // softmax has no args.
        assert!(p.steps[6].args.is_empty());
    }

    #[test]
    fn tiny_mlp() {
        let src = read_fixture("tiny_mlp.nfl");
        let nfl = parse(&src).expect("tiny_mlp.nfl must parse");
        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "TinyMLP");
        assert_eq!(m.params.len(), 1);
        assert_eq!(m.params[0].name, "batch");
        assert_eq!(m.params[0].value, 8);

        assert_eq!(m.body.len(), 2);
        let ModelStmt::VariableDecl(v) = &m.body[0] else { panic!() };
        assert_eq!(v.ty.dims, vec![Dim::Symbol("batch".into()), Dim::Integer(4)]);

        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps.len(), 2);
    }

    #[test]
    fn pipeline_styles_three_models() {
        let src = read_fixture("pipeline_styles.nfl");
        let nfl = parse(&src).expect("pipeline_styles.nfl must parse");

        assert_eq!(nfl.models.len(), 3);
        assert_eq!(nfl.models[0].name, "SingleLine");
        assert_eq!(nfl.models[1].name, "PerStepWrap");
        assert_eq!(nfl.models[2].name, "MixedWrap");

        // All three have the same pipeline shape: x -> linear[8] -> relu -> linear[output] -> softmax.
        for m in &nfl.models {
            let ModelStmt::Pipeline(p) = &m.body[1] else { panic!("expected Pipeline in {}", m.name) };
            assert_eq!(p.steps.len(), 4, "model {} should have 4 pipeline steps", m.name);
            assert_eq!(
                p.steps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
                vec!["linear", "relu", "linear", "softmax"],
            );
        }
    }

    #[test]
    fn comments_are_ignored() {
        let src = read_fixture("comments.nfl");
        let nfl = parse(&src).expect("comments.nfl must parse");
        assert_eq!(nfl.models.len(), 1);
        let m = &nfl.models[0];
        assert_eq!(m.name, "Commented");
        assert_eq!(m.body.len(), 2);
        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps.len(), 4);  // linear[16] -> relu -> linear[output] -> softmax
    }

    #[test]
    fn mixed_args() {
        let src = read_fixture("mixed_args.nfl");
        let nfl = parse(&src).expect("mixed_args.nfl must parse");
        let m = &nfl.models[0];
        let ModelStmt::Pipeline(p) = &m.body[1] else { panic!() };
        assert_eq!(p.steps[0].name, "linear");
        // First step is `linear[16, bias=true]` — one positional, one named.
        assert_eq!(p.steps[0].args.len(), 2);
        assert_eq!(p.steps[0].args[0], OpArg::Positional(ArgValue::Integer(16)));
        let OpArg::Named { name, value } = &p.steps[0].args[1] else { panic!() };
        assert_eq!(name, "bias");
        assert_eq!(*value, ArgValue::Symbol("true".into()));
    }
}
