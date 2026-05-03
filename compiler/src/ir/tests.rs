//! Unit tests for the IR module.

use super::stdlib::*;
use super::types::{AttrValue, OpAttr, Shape};

#[test]
fn resolve_known_ops() {
    assert_eq!(resolve("linear"), Some(StdOp::Linear));
    assert_eq!(resolve("relu"), Some(StdOp::Relu));
    assert_eq!(resolve("dropout"), Some(StdOp::Dropout));
    assert_eq!(resolve("softmax"), Some(StdOp::Softmax));
}

#[test]
fn resolve_unknown_op_returns_none() {
    assert_eq!(resolve("foo"), None);
    assert_eq!(resolve("Linear"), None); // case-sensitive
    assert_eq!(resolve(""), None);
}

#[test]
fn signature_linear_has_one_positional_one_named() {
    let s = signature(StdOp::Linear);
    assert_eq!(s.positional.len(), 1);
    assert_eq!(s.positional[0].name, "out_dim");
    assert_eq!(s.positional[0].ty, ArgType::Integer);
    assert!(s.positional[0].required);
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "bias");
    assert_eq!(s.named[0].ty, ArgType::Symbol);
    assert!(!s.named[0].required);
}

#[test]
fn signature_dropout_has_one_named_required() {
    let s = signature(StdOp::Dropout);
    assert!(s.positional.is_empty());
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "rate");
    assert_eq!(s.named[0].ty, ArgType::Float);
    assert!(s.named[0].required);
}

#[test]
fn signature_relu_and_softmax_are_empty() {
    let r = signature(StdOp::Relu);
    assert!(r.positional.is_empty());
    assert!(r.named.is_empty());
    let s = signature(StdOp::Softmax);
    assert!(s.positional.is_empty());
    assert!(s.named.is_empty());
}

#[test]
fn infer_linear_output_shape() {
    let input = Shape(vec![8, 4]);
    let attrs = vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }];
    let out = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap();
    assert_eq!(out.0, vec![8, 2]);
}

#[test]
fn infer_linear_with_wrong_rank_input() {
    let input = Shape(vec![8]); // rank 1, linear expects rank 2
    let attrs = vec![OpAttr { name: "out_dim".into(), value: AttrValue::Integer(2) }];
    let err = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap_err();
    matches!(err, ShapeError::WrongRank { expected: 2, actual: 1, .. });
}

#[test]
fn infer_relu_preserves_shape() {
    let input = Shape(vec![8, 2]);
    let out = infer_output_shape(StdOp::Relu, &[input.clone()], &[]).unwrap();
    assert_eq!(out, input);
}

#[test]
fn infer_softmax_and_dropout_preserve_shape() {
    let input = Shape(vec![3, 7, 2]);
    assert_eq!(infer_output_shape(StdOp::Softmax, &[input.clone()], &[]).unwrap(), input);
    assert_eq!(infer_output_shape(StdOp::Dropout, &[input.clone()], &[]).unwrap(), input);
}

use super::build::resolve_type;
use super::error::BuildErrorKind;
use crate::ast::{Dim, Span, TypeExpr};
use std::collections::HashMap;

fn span() -> Span { Span::new(1, 1) }

#[test]
fn resolve_type_all_integer_dims() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Integer(8), Dim::Integer(4)],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_symbolic_dim_with_lookup() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("batch".into()), Dim::Integer(4)],
        span: span(),
    };
    let mut params: HashMap<&str, u64> = HashMap::new();
    params.insert("batch", 8);
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_unknown_dim_errors() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("zzz".into())],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let err = resolve_type(&ty, &params).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::UnknownDim { .. }));
}

use super::build::resolve_args;
use crate::ast::{ArgValue, OpArg};

#[test]
fn resolve_args_one_positional_integer() {
    let args = vec![OpArg::Positional(ArgValue::Integer(512))];
    let attrs = resolve_args(StdOp::Linear, &args, span()).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "out_dim");
    assert_eq!(attrs[0].value, AttrValue::Integer(512));
}

#[test]
fn resolve_args_missing_required_positional() {
    let args: Vec<OpArg> = vec![]; // linear needs out_dim
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(
        err.kind,
        BuildErrorKind::ArgCountMismatch { .. } | BuildErrorKind::MissingRequiredArg { .. }
    ));
}

#[test]
fn resolve_args_extra_positional() {
    let args = vec![
        OpArg::Positional(ArgValue::Integer(2)),
        OpArg::Positional(ArgValue::Integer(3)),
    ];
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgCountMismatch { .. }));
}

#[test]
fn resolve_args_type_mismatch() {
    let args = vec![OpArg::Positional(ArgValue::Float(2.5))]; // out_dim wants Integer
    let err = resolve_args(StdOp::Linear, &args, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgTypeMismatch { .. }));
}

#[test]
fn resolve_args_named_only_dropout() {
    let args = vec![OpArg::Named { name: "rate".into(), value: ArgValue::Float(0.2) }];
    let attrs = resolve_args(StdOp::Dropout, &args, span()).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "rate");
    assert_eq!(attrs[0].value, AttrValue::Float(0.2));
}
