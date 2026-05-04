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
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");

    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M");
    assert_eq!(sig.model, "M");
    assert_eq!(sig.input_floats, 6);
    assert_eq!(sig.params_floats, 6);
    assert_eq!(sig.output_floats, 4);

    assert_eq!(sig.params_layout.len(), 1);
    let slot = &sig.params_layout[0];
    assert_eq!(slot.kind, ParamKind::LinearWeight);
    assert_eq!(slot.offset, 0);
    assert_eq!(slot.size, 6);
    assert_eq!(slot.origin_node, 1);

    let s = &asm.source;
    assert!(s.contains(".globl _nfl_forward_M"));
    assert!(s.contains("_nfl_forward_M:"));
    assert!(s.contains("ret"));
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
    assert!(
        s.contains("cmp     x3, #2"),
        "missing i-bound (B=2) in:\n{s}"
    );
    assert!(
        s.contains("cmp     x4, #2"),
        "missing j-bound (N=2) in:\n{s}"
    );
    assert!(
        s.contains("cmp     x5, #3"),
        "missing k-bound (K=3) in:\n{s}"
    );
    // Sum init.
    assert!(s.contains("fmov    s0, wzr"), "missing sum init in:\n{s}");
}

#[test]
fn relu_emits_separate_loop_with_fmov_zero_and_fmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Zero materialisation outside the loop.
    assert!(
        s.contains("fmov    s4, wzr"),
        "missing 'fmov s4, wzr' (zero materialisation) in:\n{s}"
    );
    // The relu loop body uses fmax against s4.
    assert!(
        s.contains("fmax    s3, s3, s4"),
        "missing relu fmax in:\n{s}"
    );
    // Loop label and bound (output total = 2*2 = 4).
    assert!(s.contains(".Lrelu_0:"), "missing relu loop label in:\n{s}");
    assert!(
        s.contains("cmp     x9, #4"),
        "missing relu element-count bound in:\n{s}"
    );
    // Relu reads + writes via x2 (output buffer).
    assert!(
        s.contains("ldr     s3, [x2, x9, lsl #2]"),
        "missing relu load in:\n{s}"
    );
    assert!(
        s.contains("str     s3, [x2, x9, lsl #2]"),
        "missing relu store in:\n{s}"
    );
}

#[test]
fn relu_alone_after_matmul_does_not_break_existing_test() {
    // Sanity: matmul still emitted alongside relu.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    assert!(asm.source.contains("fmadd"));
}

#[test]
fn linear_with_bias_returns_lower_error() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::LinearWithBias { .. }));
}

#[test]
fn dropout_returns_unsupported_op() {
    let uir =
        build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> dropout[rate=0.2]\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "dropout"));
}

#[test]
fn softmax_returns_unsupported_op() {
    // softmax-only path
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> softmax\n");
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::UnsupportedOp { ref op, .. } if op == "softmax"));
}

#[test]
fn duplicate_model_name_returns_error() {
    // Two models named "M" in one source.
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n\
               model M [b=2]:\n    y: Tensor[b, 3]\n    y -> linear[2]\n";
    let uir = build_uir(src);
    let err = lower(&uir).unwrap_err();
    assert!(matches!(err, LowerError::DuplicateModelName { ref name, .. } if name == "M"));
}
