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
    assert!(s.contains("fmadd"), "expected fmadd in:\n{s}");
    // Labels include model_idx prefix: model 0, linear 0 → "0_0".
    assert!(s.contains(".Lmm_i_0_0:"));
    assert!(s.contains(".Lmm_j_0_0:"));
    assert!(s.contains(".Lmm_k_0_0:"));
    assert!(s.contains("cmp     x3, x10"));
    assert!(s.contains("cmp     x4, x15"));
    assert!(s.contains("cmp     x5, x16"));
    assert!(s.contains("fmov    s0, wzr"));
    // Destination is x12 (materialised dst pointer), not raw x2.
    assert!(s.contains("str     s0, [x12,"));
}

#[test]
fn relu_emits_separate_loop_with_fmov_zero_and_fmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("fmov    s4, wzr"));
    assert!(s.contains("fmax    s3, s3, s4"));
    assert!(s.contains(".Lrelu_0_0:"));
    assert!(s.contains("cmp     x9, x10"));
    // Relu now uses materialised src/dst pointers.
    assert!(s.contains("ldr     s3, [x11,"));
    assert!(s.contains("str     s3, [x12,"));
}

#[test]
fn relu_alone_after_matmul_does_not_break_existing_test() {
    // Sanity: matmul still emitted alongside relu.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    assert!(asm.source.contains("fmadd"));
}

#[test]
fn dropout_emits_no_code() {
    // input → linear → dropout → linear (terminal-linear). Dropout has no
    // dispatch arm that emits asm; its BufferLoc::Alias(operand) propagates.
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> dropout[rate=0.2] -> linear[2]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Two linear matmuls present (model 0 → "0_0" and "0_1").
    assert!(s.contains(".Lmm_i_0_0:"));
    assert!(s.contains(".Lmm_i_0_1:"));
    // No dropout-specific instructions or labels.
    assert!(
        !s.contains("dropout"),
        "asm must not mention dropout literally:\n{s}"
    );
}

#[test]
fn softmax_emits_three_passes() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Key instructions present.
    assert!(s.contains("bl      _expf"), "expected 'bl _expf' in:\n{s}");
    assert!(
        s.contains("fdiv"),
        "expected fdiv (normalize pass) in:\n{s}"
    );
    assert!(s.contains("fmax    s8,"), "expected fmax (max pass)");
    assert!(
        s.contains("fsub    s0, s0, s8"),
        "expected fsub (max-subtract)"
    );
    assert!(
        s.contains("fadd    s9, s9, s0"),
        "expected fadd (sum accumulate)"
    );
    // -inf materialisation present.
    assert!(s.contains("movz    w0, #0x0000"));
    assert!(s.contains("movk    w0, #0xFF80, lsl #16"));
    assert!(s.contains("fmov    s8, w0"));

    // Pass ordering: max → exp → norm. Labels include model_idx prefix → "0_0".
    let max_label = s.find(".Lsm_max_0_0:").expect("max label");
    let exp_label = s.find(".Lsm_exp_0_0:").expect("exp label");
    let norm_label = s.find(".Lsm_norm_0_0:").expect("norm label");
    assert!(max_label < exp_label, "max must precede exp");
    assert!(exp_label < norm_label, "exp must precede norm");
}

#[test]
fn softmax_function_saves_d8_d9_and_x19_x23() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Callee-saved FP (s8/s9 for max+sum).
    assert!(
        s.contains("stp     d8, d9, [sp, #-16]!"),
        "missing d8/d9 prologue:\n{s}"
    );
    assert!(
        s.contains("ldp     d8, d9, [sp], #16"),
        "missing d8/d9 epilogue:\n{s}"
    );
    // Callee-saved integer regs (x19-x23 for softmax loop state across bl _expf).
    assert!(
        s.contains("stp     x19, x20, [sp, #-16]!"),
        "missing x19/x20 prologue:\n{s}"
    );
    assert!(
        s.contains("stp     x21, x22, [sp, #-16]!"),
        "missing x21/x22 prologue:\n{s}"
    );
    assert!(
        s.contains("str     x23, [sp, #-16]!"),
        "missing x23 prologue:\n{s}"
    );
    assert!(
        s.contains("ldr     x23, [sp], #16"),
        "missing x23 epilogue:\n{s}"
    );
    assert!(
        s.contains("ldp     x21, x22, [sp], #16"),
        "missing x21/x22 epilogue:\n{s}"
    );
    assert!(
        s.contains("ldp     x19, x20, [sp], #16"),
        "missing x19/x20 epilogue:\n{s}"
    );
}

#[test]
fn non_leaf_function_saves_x29_x30() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("stp     x29, x30, [sp, #-16]!"));
    assert!(s.contains("ldp     x29, x30, [sp], #16"));
}

// ── buffer analyzer tests ────────────────────────────────────────────────────

use super::buffer::{assign_buffers, compute_callee_saved, compute_is_leaf, BufferLoc};

#[test]
fn assign_buffers_input_node_is_input_reg() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[0], BufferLoc::InputReg));
}

#[test]
fn assign_buffers_terminal_node_is_output_reg() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    let last = assignment.locs.last().unwrap();
    assert!(matches!(last, BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_relu_aliases_operand() {
    // input → linear → relu (terminal-relu)
    // n0 input, n1 linear (non-terminal), n2 relu (terminal)
    // Expected: n2 → OutputReg (terminal wins over alias rule); n1 → StackOffset
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_intermediate_relu_aliases_operand() {
    // input → linear → relu → linear → relu (terminal). Intermediate relu (n2)
    // aliases linear (n1). The terminal relu (n4) is OutputReg.
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::Alias(1)));
    assert!(matches!(assignment.locs[3], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[4], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_stack_bytes_is_aligned() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(assignment.stack_bytes > 0);
    assert_eq!(assignment.stack_bytes % 16, 0, "stack must be 16-aligned");
}

#[test]
fn compute_is_leaf_true_for_no_softmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    assert!(compute_is_leaf(&uir.models[0]));
}

#[test]
fn compute_is_leaf_false_when_softmax_present() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n");
    assert!(!compute_is_leaf(&uir.models[0]));
}

#[test]
fn compute_callee_saved_includes_d8_d9_when_softmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(regs.contains_d8_d9());
}

#[test]
fn compute_callee_saved_empty_for_leaf() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(!regs.contains_d8_d9());
}

#[test]
fn assign_buffers_stack_bytes_rounds_non_aligned_total_up() {
    // Use a model where the unaligned total is NOT already 16-aligned, so the
    // round-up math actually does work. Tensor[1, 2] -> linear[3] -> linear[3]:
    //   n0 input (no slot), n1 linear (1*3=3 floats=12 bytes, non-terminal),
    //   n2 linear (terminal -> OutputReg, no slot)
    // Total raw stack = 12 bytes; rounded up to 16.
    let uir = build_uir("model M [b=1]:\n    x: Tensor[b, 2]\n    x -> linear[3] -> linear[3]\n");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert_eq!(
        assignment.stack_bytes, 16,
        "12 raw bytes should round up to 16"
    );
}

#[test]
fn leaf_function_no_prologue() {
    // input → linear (terminal): leaf, no intermediates.
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // Leaf, no intermediates → no stp, no sub sp, no ldp.
    assert!(
        !s.contains("stp"),
        "leaf-no-intermediates should have no stp:\n{s}"
    );
    assert!(!s.contains("ldp"));
    assert!(!s.contains("sub     sp"));
}

#[test]
fn intermediate_buffers_allocated_on_stack() {
    let uir = build_uir(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(s.contains("sub     sp, sp,"), "expected sub sp in:\n{s}");
    assert!(s.contains("add     sp, sp,"), "expected add sp in:\n{s}");
}

#[test]
fn linear_with_bias_emits_bias_add() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    // After the k-loop end, before the store, expect bias load + fadd.
    assert!(
        s.contains("fadd    s0, s0,"),
        "expected fadd s0, s0, ... in:\n{s}"
    );
}

#[test]
fn linear_bias_packed_layout() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n");
    let asm = lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    // Two slots: LinearWeight (size 6) then LinearBias (size 2) immediately after.
    assert_eq!(sig.params_layout.len(), 2);
    assert_eq!(sig.params_layout[0].kind, ParamKind::LinearWeight);
    assert_eq!(sig.params_layout[0].size, 6);
    assert_eq!(sig.params_layout[1].kind, ParamKind::LinearBias);
    assert_eq!(sig.params_layout[1].size, 2);
    assert_eq!(sig.params_layout[1].offset, 6);
    assert_eq!(sig.params_floats, 8);
}

// ── emit_sp_sub / emit_sp_add branch coverage ────────────────────────────────

use super::asm::{emit_sp_add, emit_sp_sub};

#[test]
fn emit_sp_sub_small_immediate() {
    let s = emit_sp_sub(80);
    assert_eq!(s, "    sub     sp, sp, #80\n");
}

#[test]
fn emit_sp_sub_shifted_12_for_4096_multiple() {
    let s = emit_sp_sub(8192);
    // 8192 = 2*4096 → "sub sp, sp, #2, lsl #12"
    assert_eq!(s, "    sub     sp, sp, #2, lsl #12\n");
}

#[test]
fn emit_sp_sub_movz_movk_for_general_case() {
    // 99584 = 0x18500 → lo=0x8500, hi=0x0001
    let s = emit_sp_sub(99584);
    assert!(s.contains("movz    w9, #0x8500"));
    assert!(s.contains("movk    w9, #0x0001, lsl #16"));
    assert!(s.contains("sub     sp, sp, x9"));
}

#[test]
fn emit_sp_add_small_immediate() {
    let s = emit_sp_add(80);
    assert_eq!(s, "    add     sp, sp, #80\n");
}

#[test]
fn emit_sp_add_shifted_12_for_4096_multiple() {
    let s = emit_sp_add(8192);
    assert_eq!(s, "    add     sp, sp, #2, lsl #12\n");
}

#[test]
fn emit_sp_add_movz_movk_for_general_case() {
    let s = emit_sp_add(99584);
    assert!(s.contains("movz    w9, #0x8500"));
    assert!(s.contains("movk    w9, #0x0001, lsl #16"));
    assert!(s.contains("add     sp, sp, x9"));
}

// ── RegSet x19_x23 flag tests ────────────────────────────────────────────────

#[test]
fn compute_callee_saved_includes_x19_x23_when_softmax() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(regs.contains_x19_x23());
}

#[test]
fn compute_callee_saved_no_x19_x23_for_leaf() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(!regs.contains_x19_x23());
}

// ── UnsupportedOp display and span round-trip ────────────────────────────────

#[test]
fn unsupported_op_display_and_span_round_trip() {
    let span = compiler::ast::Span::new(1, 1);
    let e = LowerError::UnsupportedOp {
        op: "future_op".into(),
        span,
    };
    let msg = e.to_string();
    assert!(
        msg.contains("future_op"),
        "Display should mention op name; got: {msg}"
    );
    assert_eq!(e.span().line, span.line);
    assert_eq!(e.span().col, span.col);
}

#[test]
fn unsupported_post_op_display_and_span_round_trip() {
    let span = compiler::ast::Span::new(7, 3);
    let e = LowerError::UnsupportedPostOp {
        op: "future_post_op".into(),
        span,
    };
    let msg = e.to_string();
    assert!(
        msg.contains("future_post_op"),
        "Display should mention post-op name; got: {msg}"
    );
    assert!(
        msg.contains("post-op"),
        "Display should clearly mark this as a post-op error; got: {msg}"
    );
    assert_eq!(e.span().line, span.line);
    assert_eq!(e.span().col, span.col);
}

#[test]
fn fused_linear_relu_emits_fmax_before_store() {
    use compiler::{NodeKind, PostOp};
    // Synthetic: hand-build UIR where Linear has fused_post_ops = [Relu].
    let mut uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else {
        panic!("expected Op node");
    };
    fused_post_ops.push(PostOp::Relu);

    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // s4 materialised once.
    assert!(
        s.contains("fmov    s4, wzr"),
        "missing s4 zero materialisation:\n{s}"
    );
    // fmax inline before store.
    assert!(
        s.contains("fmax    s0, s0, s4"),
        "missing inline fmax (relu):\n{s}"
    );
}

#[test]
fn fused_linear_relu_no_separate_relu_loop() {
    use compiler::{NodeKind, PostOp};
    // Same fixture as above. Asm must NOT contain a separate .Lrelu_*: label.
    let mut uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else {
        panic!()
    };
    fused_post_ops.push(PostOp::Relu);

    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        !s.contains(".Lrelu_"),
        "fused linear+relu should NOT emit a separate relu loop:\n{s}"
    );
}

#[test]
fn unfused_linear_still_no_fmax() {
    // Linear without fused_post_ops: no fmax AND no s4 zero-materialisation.
    // The two assertions together pin the un-fused asm shape: no post-op
    // path is taken at all (neither header materialisation nor inline op).
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        !s.contains("fmax"),
        "un-fused linear should NOT emit fmax:\n{s}"
    );
    assert!(
        !s.contains("fmov    s4, wzr"),
        "un-fused linear should NOT materialise s4 zero (only fused-relu needs it):\n{s}"
    );
}

// ── M6 analyzer tests: PostOp::SoftmaxRow via default pipeline ───────────────

#[test]
fn is_leaf_false_for_fused_softmax_row_linear() {
    use compiler::passes::{default_pipeline, run_pipeline};

    // Construct a fused linear → softmax UIR via the parser + default pipeline.
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let model = &fused.models[0];

    assert!(
        !super::buffer::compute_is_leaf(model),
        "a Linear carrying PostOp::SoftmaxRow still calls bl _expf — leaf must be false"
    );
}

#[test]
fn callee_saved_includes_d8_d9_for_fused_softmax_row() {
    use compiler::passes::{default_pipeline, run_pipeline};

    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let model = &fused.models[0];

    let regs = super::buffer::compute_callee_saved(model);
    assert!(regs.d8_d9, "fused-SoftmaxRow Linear needs d8/d9 saved");
    assert!(regs.x19_x23, "fused-SoftmaxRow Linear needs x19-x23 saved");
}

// ── M6 asm-shape tests: four-phase softmax tail via default pipeline ──────────

#[test]
fn emit_linear_with_softmax_row_post_op_emits_three_phase_softmax() {
    use compiler::passes::{default_pipeline, run_pipeline};

    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let asm = crate::lower(&fused).expect("lower ok");
    let s = &asm.source;

    // Phase 1 — matmul. Some fmadd must appear.
    assert!(s.contains("fmadd"), "Phase 1 matmul missing:\n{s}");

    // Phase 2 — row-max scan into s8.
    assert!(
        s.contains("fmax    s8, s8, s1"),
        "Phase 2 row-max scan into s8 missing:\n{s}"
    );

    // Phase 3 — exp(x - max), sum into s9, with bl _expf.
    assert!(
        s.contains("bl      _expf"),
        "Phase 3 missing bl _expf:\n{s}"
    );
    assert!(
        s.contains("fadd    s9, s9, s0"),
        "Phase 3 sum accumulation in s9 missing:\n{s}"
    );

    // Phase 4 — normalise by s9.
    assert!(
        s.contains("fdiv    s0, s0, s9"),
        "Phase 4 normalise missing:\n{s}"
    );

    // Fused asm uses .Lfsmx_* labels; no standalone .Lsm_* labels expected.
    assert!(
        !s.contains(".Lsm_"),
        "fused asm must not emit standalone softmax .Lsm_* labels:\n{s}"
    );
    assert!(
        s.contains(".Lfsmx_"),
        "fused asm must use .Lfsmx_* labels for the inlined softmax tail:\n{s}"
    );
}

#[test]
fn emit_linear_with_softmax_row_post_op_preserves_bias_add() {
    use compiler::passes::{default_pipeline, run_pipeline};

    // bias=true on the linear; confirmed against tests/fixtures/mixed_args.nfl syntax.
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> softmax\n";
    let ast = compiler::parse(src).expect("parse ok");
    let uir = compiler::ir::build(&ast).expect("build ok");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline ok");
    let asm = crate::lower(&fused).expect("lower ok");
    let s = &asm.source;

    // Phase 1 still emits matmul → bias-add. The bias-add is `fadd s0, s0, s5`
    // per the existing M5b emit_linear shape.
    assert!(
        s.contains("fadd    s0, s0, s5"),
        "bias-add missing in fused row-wise emit:\n{s}"
    );
    // Phase 3 still calls _expf.
    assert!(
        s.contains("bl      _expf"),
        "fused softmax tail missing bl _expf:\n{s}"
    );
}

#[test]
fn dropout_as_output_emits_copy_loop() {
    let uir = build_uir(
        "model OnlyDropout [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n",
    );
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        s.contains("; dropout-as-output:"),
        "missing dropout-as-output comment in:\n{s}"
    );
    assert!(
        s.contains(".Ldropout_0_0:"),
        "missing dropout loop label in:\n{s}"
    );
    assert!(
        s.contains("ldr     s3, [x11"),
        "missing s3 load from src ptr in:\n{s}"
    );
    assert!(
        s.contains("str     s3, [x12"),
        "missing s3 store to dst ptr in:\n{s}"
    );
    assert!(
        !s.contains("fmax"),
        "dropout copy must not clamp (no fmax expected in identity copy):\n{s}"
    );
}

#[test]
fn relu_uses_register_form_cmp_with_hoisted_movz() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Hoisted materialise must appear AFTER the materialise_ptr lines
    // (which set up x11/x12) and BEFORE the .Lrelu_ label.
    let movz_pos = s
        .find("movz    x10, ")
        .expect("missing movz x10 hoist for relu loop bound");
    let label_pos = s.find(".Lrelu_0_0:").expect("missing relu loop label");
    assert!(
        movz_pos < label_pos,
        "movz x10 must precede .Lrelu_ label (hoist outside loop)"
    );

    // Inside loop, cmp uses register form against x10.
    assert!(
        s.contains("cmp     x9, x10"),
        "cmp must use register form (x9, x10), not literal imm; full asm:\n{s}"
    );
    // Old literal-imm form must not appear for relu's bound.
    assert!(
        !s.contains("cmp     x9, #4"),
        "old literal-imm cmp must be replaced; full asm:\n{s}"
    );
}

#[test]
fn linear_matmul_body_uses_hoisted_dim_registers() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // Three hoists must appear before the i-loop label.
    let i_label_pos = s.find(".Lmm_i_0_0:").expect("missing matmul i-loop label");
    for reg in ["x10", "x15", "x16"] {
        let movz = format!("movz    {}, ", reg);
        let pos = s
            .find(&movz)
            .unwrap_or_else(|| panic!("missing hoist for {reg}: \n{s}"));
        assert!(pos < i_label_pos, "{reg} hoist must precede .Lmm_i_ label");
    }

    // Loop-bound cmps use register form.
    assert!(s.contains("cmp     x3, x10"), "i-loop cmp must use x10");
    assert!(s.contains("cmp     x4, x15"), "j-loop cmp must use x15");
    assert!(s.contains("cmp     x5, x16"), "k-loop cmp must use x16");

    // Mov-sites for stride reuse hoisted registers (no re-materialise).
    assert!(
        s.contains("mov     x8, x16"),
        "input-stride mov must reuse hoisted k (x16)"
    );
    assert!(
        s.contains("mov     x8, x15"),
        "output-stride mov must reuse hoisted n (x15)"
    );

    // Old literal-imm cmps must not appear for matmul bounds.
    for old in ["cmp     x3, #2", "cmp     x4, #2", "cmp     x5, #3"] {
        assert!(
            !s.contains(old),
            "old literal-imm cmp '{old}' must be removed"
        );
    }
}

#[test]
fn softmax_standalone_uses_register_form_cmps_re_materialised() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;

    // i-loop, max-loop, exp-loop, norm-loop — all four cmps register form.
    assert!(s.contains("cmp     x19, x10"), "i-loop cmp register form");
    // x21 is reused across max/exp/norm phases — find the cmp pattern.
    let count_x21_cmp_x10 = s.matches("cmp     x21, x10").count();
    assert_eq!(
        count_x21_cmp_x10, 3,
        "max/exp/norm phases must each cmp x21 against x10 (3 sites); got {count_x21_cmp_x10}\nfull asm:\n{s}"
    );

    // No literal-imm cmps for softmax bounds.
    assert!(
        !s.contains("cmp     x19, #2"),
        "old i-loop literal-imm cmp must be removed"
    );
    assert!(
        !s.contains("cmp     x21, #3"),
        "old phase-loop literal-imm cmps must be removed"
    );
}

#[test]
fn linear_rowwise_softmax_tail_uses_re_materialised_cmps() {
    use compiler::passes::{default_pipeline, run_pipeline};
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[4] -> softmax\n";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline");
    let asm = lower(&fused).expect("lower");
    let s = &asm.source;

    // Pipeline applies fuse_linear_softmax → emits RowWise tail.
    assert!(
        s.contains("; fused softmax_row:"),
        "expected fused RowWise softmax tail; full asm:\n{s}"
    );

    // Re-materialise pattern: at each fsmx loop top, movz x10 then cmp.
    assert!(
        s.contains("cmp     x19, x10"),
        "fsmx i-loop cmp register form"
    );
    let count_x21_cmp_x10 = s.matches("cmp     x21, x10").count();
    // 3 phase loops in the tail (max/exp/norm) — each uses cmp x21, x10.
    // BUT: standalone softmax (already patched in Task 8) also uses cmp x21, x10
    // at 3 sites. So if this test fixture builds asm with both standalone softmax
    // AND fused RowWise tail in the same model, the count would be 6.
    // The fixture above uses linear → softmax which fuses fully; no standalone
    // softmax should remain. So expect exactly 3 fsmx cmps.
    assert_eq!(
        count_x21_cmp_x10, 3,
        "fsmx max/exp/norm cmps must each use register form (3 sites); got {count_x21_cmp_x10}\nfull asm:\n{s}"
    );

    // No literal-imm fsmx cmps remain.
    assert!(
        !s.contains("cmp     x19, #2"),
        "old fsmx i-loop literal-imm cmp must be removed"
    );
}
