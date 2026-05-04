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
