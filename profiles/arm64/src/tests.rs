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

#[test]
fn linear_emits_function_with_correct_symbol_and_ret() {
    // model M [b=2]: x: Tensor[b, 3]
    //     x -> linear[2]
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");

    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M");
    assert_eq!(sig.model, "M");
    assert_eq!(sig.input_floats, 6);   // 2*3
    assert_eq!(sig.weight_floats, 6);  // 3*2
    assert_eq!(sig.output_floats, 4);  // 2*2

    let s = &asm.source;
    assert!(s.contains(".globl _nfl_forward_M"), "missing .globl in:\n{s}");
    assert!(s.contains("_nfl_forward_M:"), "missing label in:\n{s}");
    assert!(s.contains("ret"), "missing ret in:\n{s}");
}

#[test]
fn linear_emits_matmul_loops_with_fmadd() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Sanity: FMADD is the matmul accumulator.
    assert!(s.contains("fmadd"), "expected fmadd in:\n{s}");
    // Three loop labels (i, j, k) for the single Linear (label suffix 0).
    assert!(s.contains(".Lmm_i_0:"), "missing i-loop label in:\n{s}");
    assert!(s.contains(".Lmm_j_0:"), "missing j-loop label in:\n{s}");
    assert!(s.contains(".Lmm_k_0:"), "missing k-loop label in:\n{s}");
    // Comparison constants come from shapes.
    assert!(s.contains("cmp     x3, #2"), "missing i-bound (B=2) in:\n{s}");
    assert!(s.contains("cmp     x4, #2"), "missing j-bound (N=2) in:\n{s}");
    assert!(s.contains("cmp     x5, #3"), "missing k-bound (K=3) in:\n{s}");
    // Sum init.
    assert!(s.contains("fmov    s0, wzr"), "missing sum init in:\n{s}");
}
