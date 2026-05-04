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
    assert!(s.contains(".Lmm_i_0:"));
    assert!(s.contains(".Lmm_j_0:"));
    assert!(s.contains(".Lmm_k_0:"));
    assert!(s.contains("cmp     x3, #2"));
    assert!(s.contains("cmp     x4, #2"));
    assert!(s.contains("cmp     x5, #3"));
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
    assert!(s.contains(".Lrelu_0:"));
    assert!(s.contains("cmp     x9, #4"));
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
    // Two linear matmuls present.
    assert!(s.contains(".Lmm_i_0:"));
    assert!(s.contains(".Lmm_i_1:"));
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
    // Pass 2 has the bl _expf.
    assert!(s.contains("bl      _expf"), "expected 'bl _expf' in:\n{s}");
    // Pass 3 has the divide.
    assert!(
        s.contains("fdiv"),
        "expected fdiv (normalize pass) in:\n{s}"
    );
}

#[test]
fn softmax_function_saves_d8_d9() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> softmax\n");
    let asm = lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        s.contains("stp     d8, d9, [sp, #-16]!"),
        "missing d8/d9 prologue:\n{s}"
    );
    assert!(
        s.contains("ldp     d8, d9, [sp], #16"),
        "missing d8/d9 epilogue:\n{s}"
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
