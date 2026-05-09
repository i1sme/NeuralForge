// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the parser, exercising one production at a time.

use super::*;
use crate::ast::*;
use crate::lexer::lex;

fn parser_of(src: &str) -> Parser<'_> {
    // Test helper: lex `src`, leak the tokens to keep them alive for the
    // returned Parser. Tests are short-lived so the leak is harmless.
    let toks = lex(src).expect("lex must succeed in test");
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    Parser::new(leaked)
}

#[test]
fn parse_arg_value_integer() {
    let mut p = parser_of("512");
    let v = parse_arg_value(&mut p).unwrap();
    assert_eq!(v, ArgValue::Integer(512));
}

#[test]
fn parse_arg_value_float() {
    let mut p = parser_of("0.2");
    let v = parse_arg_value(&mut p).unwrap();
    let ArgValue::Float(f) = v else {
        panic!("expected Float")
    };
    assert!((f - 0.2).abs() < 1e-12);
}

#[test]
fn parse_arg_value_symbol() {
    let mut p = parser_of("batch");
    let v = parse_arg_value(&mut p).unwrap();
    assert_eq!(v, ArgValue::Symbol("batch".into()));
}

#[test]
fn parse_operation_no_args() {
    let mut p = parser_of("relu");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.name, "relu");
    assert!(op.args.is_empty());
}

#[test]
fn parse_operation_one_positional() {
    let mut p = parser_of("linear[512]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.name, "linear");
    assert_eq!(op.args.len(), 1);
    assert_eq!(op.args[0], OpArg::Positional(ArgValue::Integer(512)));
}

#[test]
fn parse_operation_named_only() {
    let mut p = parser_of("dropout[rate=0.2]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.args.len(), 1);
    let OpArg::Named {
        name,
        value: ArgValue::Float(f),
    } = &op.args[0]
    else {
        panic!("expected named float arg");
    };
    assert_eq!(name, "rate");
    assert!((f - 0.2).abs() < 1e-12);
}

#[test]
fn parse_operation_mixed_positional_then_named() {
    let mut p = parser_of("linear[16, bias=true]");
    let op = parse_operation(&mut p).unwrap();
    assert_eq!(op.args.len(), 2);
    assert_eq!(op.args[0], OpArg::Positional(ArgValue::Integer(16)));
    let OpArg::Named { name, value } = &op.args[1] else {
        panic!()
    };
    assert_eq!(name, "bias");
    assert_eq!(*value, ArgValue::Symbol("true".into()));
}

#[test]
fn parse_operation_named_before_positional_is_error() {
    let mut p = parser_of("linear[a=1, 2]");
    let err = parse_operation(&mut p).unwrap_err();
    assert!(
        err.message.to_lowercase().contains("positional")
            || err.message.to_lowercase().contains("named"),
        "expected message about positional/named ordering, got: {}",
        err.message
    );
}

#[test]
fn parse_pipeline_one_step() {
    let mut p = parser_of("x -> linear[2]");
    let ps = parse_pipeline_stmt(&mut p).unwrap();
    assert_eq!(ps.source, "x");
    assert_eq!(ps.steps.len(), 1);
    assert_eq!(ps.steps[0].name, "linear");
}

#[test]
fn parse_pipeline_three_steps() {
    let mut p = parser_of("x -> linear[8] -> relu -> softmax");
    let ps = parse_pipeline_stmt(&mut p).unwrap();
    assert_eq!(ps.source, "x");
    assert_eq!(ps.steps.len(), 3);
    assert_eq!(
        ps.steps.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
        vec!["linear", "relu", "softmax"]
    );
}

#[test]
fn parse_pipeline_missing_arrow_after_source_is_error() {
    let mut p = parser_of("x linear[2]"); // missing "->"
    let err = parse_pipeline_stmt(&mut p).unwrap_err();
    assert!(err.message.contains("'->'"), "got: {}", err.message);
}

#[test]
fn parse_type_expr_integer_dims() {
    let mut p = parser_of("Tensor[8, 4]");
    let t = parse_type_expr(&mut p).unwrap();
    assert_eq!(t.name, "Tensor");
    assert_eq!(t.dims, vec![Dim::Integer(8), Dim::Integer(4)]);
}

#[test]
fn parse_type_expr_symbolic_dims() {
    let mut p = parser_of("Tensor[batch, input]");
    let t = parse_type_expr(&mut p).unwrap();
    assert_eq!(
        t.dims,
        vec![Dim::Symbol("batch".into()), Dim::Symbol("input".into())]
    );
}

#[test]
fn parse_type_expr_empty_brackets_is_error() {
    let mut p = parser_of("Tensor[]");
    let err = parse_type_expr(&mut p).unwrap_err();
    assert!(
        err.message.to_lowercase().contains("dim") || err.message.to_lowercase().contains("empty"),
        "got: {}",
        err.message
    );
}

#[test]
fn parse_variable_decl_basic() {
    let mut p = parser_of("x: Tensor[batch, 4]");
    let v = parse_variable_decl(&mut p).unwrap();
    assert_eq!(v.name, "x");
    assert_eq!(v.ty.dims.len(), 2);
}

#[test]
fn parse_named_value_basic() {
    let mut p = parser_of("batch=32");
    let nv = parse_named_value(&mut p).unwrap();
    assert_eq!(nv.name, "batch");
    assert_eq!(nv.value, 32);
}

#[test]
fn parse_model_params_three() {
    let mut p = parser_of("batch=32, input=784, output=10");
    let params = parse_model_params(&mut p).unwrap();
    assert_eq!(params.len(), 3);
    assert_eq!(params[0].name, "batch");
    assert_eq!(params[2].value, 10);
}

#[test]
fn parse_model_params_one() {
    let mut p = parser_of("batch=8");
    let params = parse_model_params(&mut p).unwrap();
    assert_eq!(params.len(), 1);
}

#[test]
fn parse_model_def_minimal() {
    let mut p = parser_of(
        "model TinyMLP [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2] -> softmax\n",
    );
    let m = parse_model_def(&mut p).unwrap();
    assert_eq!(m.name, "TinyMLP");
    assert_eq!(m.params.len(), 1);
    assert_eq!(m.body.len(), 2);
    assert!(matches!(m.body[0], ModelStmt::VariableDecl(_)));
    assert!(matches!(m.body[1], ModelStmt::Pipeline(_)));
}

#[test]
fn parse_model_def_three_params() {
    let src = "model X [batch=32, input=784, output=10]:\n    x: Tensor[batch, input]\n    x -> linear[output] -> softmax\n";
    let mut p = parser_of(src);
    let m = parse_model_def(&mut p).unwrap();
    assert_eq!(m.params.len(), 3);
    let ModelStmt::Pipeline(ps) = &m.body[1] else {
        panic!()
    };
    assert_eq!(ps.steps.len(), 2);
}

#[test]
fn parse_model_def_missing_colon_is_error() {
    let mut p = parser_of("model X [batch=8]\n    x: Tensor[batch, 4]\n    x -> linear[2]\n");
    let err = parse_model_def(&mut p).unwrap_err();
    assert!(
        err.message.contains("':'") || err.message.to_lowercase().contains("colon"),
        "got: {}",
        err.message
    );
}

#[test]
fn parse_nfl_source_one_model() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2]\n";
    let toks = lex(src).unwrap();
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    let mut p = Parser::new(leaked);
    let nfl = parse_nfl_source(&mut p).unwrap();
    assert_eq!(nfl.models.len(), 1);
}

#[test]
fn parse_nfl_source_two_models() {
    let src = "model A [batch=4]:\n    x: Tensor[batch, 1]\n    x -> linear[1]\n\nmodel B [batch=4]:\n    x: Tensor[batch, 1]\n    x -> linear[1]\n";
    let toks = lex(src).unwrap();
    let leaked: &'static [Token] = Box::leak(toks.into_boxed_slice());
    let mut p = Parser::new(leaked);
    let nfl = parse_nfl_source(&mut p).unwrap();
    assert_eq!(nfl.models.len(), 2);
    assert_eq!(nfl.models[0].name, "A");
    assert_eq!(nfl.models[1].name, "B");
}

#[test]
fn library_parse_round_trip_minimal() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> softmax\n";
    let nfl = crate::parse(src).expect("must parse");
    assert_eq!(nfl.models[0].name, "X");
}

#[test]
fn parse_named_pipeline_stmt_2d() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let model = &ast.models[0];
    assert_eq!(model.body.len(), 2);
    let np = match &model.body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        other => panic!("expected NamedPipeline, got {:?}", other),
    };
    assert_eq!(np.binding_name, "y");
    assert_eq!(np.source, "x");
    assert_eq!(np.steps.len(), 1);
    assert_eq!(np.steps[0].name, "relu");
    assert_eq!(np.declared_ty.dims.len(), 2);
}

#[test]
fn parse_named_pipeline_stmt_4d() {
    let src = "\
model M [batch=2, heads=4, seq=16, head_dim=16]:
    x: Tensor[batch, heads, seq, head_dim]

    scores: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let ast = crate::parse(src).expect("parse");
    let np = match &ast.models[0].body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        other => panic!("expected NamedPipeline, got {:?}", other),
    };
    assert_eq!(np.binding_name, "scores");
    assert_eq!(np.declared_ty.dims.len(), 4);
    assert_eq!(np.source, "x");
    assert_eq!(np.steps.len(), 1);
    assert_eq!(np.steps[0].name, "matmul");
    // First positional arg is the tensor identifier `x`.
    let crate::ast::OpArg::Positional(crate::ast::ArgValue::Symbol(s)) = &np.steps[0].args[0]
    else {
        panic!("expected positional Symbol arg");
    };
    assert_eq!(s, "x");
    // Second arg is named `transpose_b=true`.
    let crate::ast::OpArg::Named { name, value } = &np.steps[0].args[1] else {
        panic!("expected named arg");
    };
    assert_eq!(name, "transpose_b");
    let crate::ast::ArgValue::Symbol(v) = value else {
        panic!("expected Symbol value");
    };
    assert_eq!(v, "true");
}

#[test]
fn parse_named_pipeline_with_tensor_op_arg() {
    // Just confirms the parser accepts an identifier as a positional arg
    // (the existing `arg_value = number | identifier` rule). The semantic
    // tensor-name resolution lands in Group 2.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> matmul[x]
";
    let ast = crate::parse(src).expect("parse");
    let np = match &ast.models[0].body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        _ => panic!(),
    };
    assert_eq!(np.steps[0].name, "matmul");
    assert_eq!(np.steps[0].args.len(), 1);
    let crate::ast::OpArg::Positional(crate::ast::ArgValue::Symbol(s)) = &np.steps[0].args[0]
    else {
        panic!("expected positional Symbol arg");
    };
    assert_eq!(s, "x");
}

#[test]
fn parse_lookahead_distinguishes_variable_decl_from_named_pipeline() {
    // Both forms share the prefix `Ident ":" Tensor[...]`. The presence
    // of `=` after the type expression is the sole disambiguator.
    let src_var = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    x -> relu
";
    let src_np = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast_var = crate::parse(src_var).expect("var parse");
    let ast_np = crate::parse(src_np).expect("np parse");
    // First stmt is VariableDecl in both.
    assert!(matches!(
        ast_var.models[0].body[0],
        crate::ast::ModelStmt::VariableDecl(_)
    ));
    assert!(matches!(
        ast_np.models[0].body[0],
        crate::ast::ModelStmt::VariableDecl(_)
    ));
    // Second stmt: Pipeline in src_var, NamedPipeline in src_np.
    assert!(matches!(
        ast_var.models[0].body[1],
        crate::ast::ModelStmt::Pipeline(_)
    ));
    assert!(matches!(
        ast_np.models[0].body[1],
        crate::ast::ModelStmt::NamedPipeline(_)
    ));
}

#[test]
fn parse_named_pipeline_missing_eq_after_type() {
    // `y: Tensor[...] x -> relu` (missing `=`) should fail at the parser
    // level. After the type_expr, the lookahead branch sees neither
    // `Equals` (named pipeline) nor end-of-stmt (variable_decl), so we
    // get a variable_decl, then the followup `x` becomes a fresh stmt
    // start, then we see `-> relu` with no leading identifier — error.
    //
    // The exact error wording depends on which branch hits the failure
    // first. We don't pin it; we only require that parse() returns Err.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] x -> relu
";
    let result = crate::parse(src);
    assert!(result.is_err(), "expected parse error, got Ok");
}

#[test]
fn parse_named_pipeline_multi_step() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 2] = x -> linear[2] -> relu
";
    let ast = crate::parse(src).expect("parse");
    let np = match &ast.models[0].body[1] {
        crate::ast::ModelStmt::NamedPipeline(np) => np,
        other => panic!("expected NamedPipeline, got {:?}", other),
    };
    assert_eq!(np.steps.len(), 2);
    assert_eq!(np.steps[0].name, "linear");
    assert_eq!(np.steps[1].name, "relu");
}

#[test]
fn parse_named_pipeline_missing_source_after_eq() {
    // After `=`, parser expects an identifier (the source). `->` instead is
    // a parse error.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = -> relu
";
    assert!(crate::parse(src).is_err());
}

#[test]
fn parse_named_pipeline_missing_arrow_after_source() {
    // After the source identifier, parser expects `->`. A bare identifier
    // (no chain) is a parse error.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x relu
";
    assert!(crate::parse(src).is_err());
}
