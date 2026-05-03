//! Unit tests for the IR module.

use super::stdlib::*;

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
