//! Unit tests for the arm64 codegen profile.

use super::*;

/// Build a UIR from a small NFL source string. Used by every test below.
fn build_uir(src: &str) -> compiler::Uir {
    let ast = compiler::parse(src).expect("parse");
    compiler::ir::build(&ast).expect("ir::build")
}

#[test]
fn empty_uir_lowers_to_empty_asm() {
    let uir = compiler::Uir { models: Vec::new() };
    let asm = lower(&uir).unwrap();
    assert!(asm.source.is_empty());
    assert!(asm.functions.is_empty());
}

#[test]
fn unsupported_op_returns_unsupported() {
    // tiny_mlp ends in softmax — not supported in M4a
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> softmax\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "softmax"));
}
