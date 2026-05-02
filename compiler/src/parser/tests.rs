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
    let ArgValue::Float(f) = v else { panic!("expected Float") };
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
    let OpArg::Named { name, value: ArgValue::Float(f) } = &op.args[0] else {
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
    let OpArg::Named { name, value } = &op.args[1] else { panic!() };
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
    assert_eq!(ps.steps.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
               vec!["linear", "relu", "softmax"]);
}

#[test]
fn parse_pipeline_missing_arrow_after_source_is_error() {
    let mut p = parser_of("x linear[2]");        // missing "->"
    let err = parse_pipeline_stmt(&mut p).unwrap_err();
    assert!(err.message.contains("'->'"), "got: {}", err.message);
}
