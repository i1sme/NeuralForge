// SPDX-License-Identifier: Apache-2.0

use crate::asm::{compute_frame_size, emit_imm32_to_r10};
use crate::buffer::{assign_buffers, compute_callee_saved, BufferLoc};
use crate::LowerError;
use compiler::ir;
use compiler::passes;
use profile_api::Profile;

#[allow(dead_code)]
fn lower_x86(src: &str) -> profile_api::Asm {
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let uir = passes::run_pipeline(&uir, &passes::default_pipeline()).expect("pipeline");
    crate::lower(&uir).expect("lower")
}

#[allow(dead_code)]
fn lower_x86_no_passes(src: &str) -> profile_api::Asm {
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    crate::lower(&uir).expect("lower")
}

// Spec §7.5: entry-state rsp ≡ 8 (mod 16); after sub rsp, frame_size
// final rsp must be ≡ 0 (mod 16). Each push reg flips parity (8 bytes).
// Therefore the +8 correction applies when num_pushes is EVEN
// (post-pushes parity is 8) — see spec §4.9 derivation.

#[test]
fn frame_size_raw0_pushes0_is_8() {
    // post-pushes ≡ 8; sub 8 → 0 ✓
    assert_eq!(compute_frame_size(0, 0), 8);
}

#[test]
fn frame_size_raw0_pushes1_is_0() {
    // post-pushes ≡ 0; sub 0 → 0 ✓
    assert_eq!(compute_frame_size(0, 1), 0);
}

#[test]
fn frame_size_raw0_pushes2_is_8() {
    // post-pushes ≡ 8; sub 8 → 0 ✓
    assert_eq!(compute_frame_size(0, 2), 8);
}

#[test]
fn frame_size_raw8_pushes0_is_24() {
    // aligned=16, +8; post-pushes ≡ 8; sub 24 ≡ -16 ≡ 0 ✓
    assert_eq!(compute_frame_size(8, 0), 24);
}

#[test]
fn frame_size_raw8_pushes1_is_16() {
    // aligned=16, +0; post-pushes ≡ 0; sub 16 → 0 ✓
    assert_eq!(compute_frame_size(8, 1), 16);
}

#[test]
fn frame_size_raw16_pushes1_is_16() {
    // same alignment as the raw=8/pushes=1 case
    assert_eq!(compute_frame_size(16, 1), 16);
}

#[test]
fn frame_size_raw17_pushes0_is_40() {
    // aligned=32, +8; post-pushes ≡ 8; sub 40 ≡ -32 ≡ 0 ✓
    assert_eq!(compute_frame_size(17, 0), 40);
}

#[test]
fn frame_size_raw17_pushes1_is_32() {
    // aligned=32, +0; post-pushes ≡ 0; sub 32 → 0 ✓
    assert_eq!(compute_frame_size(17, 1), 32);
}

// ── asm helper coverage ──────────────────────────────────────────────────────

#[test]
fn emit_imm32_to_r10_formats_correctly() {
    let s = emit_imm32_to_r10(42);
    assert_eq!(s, "    movl    $42, %r10d\n");
}

#[test]
fn emit_imm32_to_r10_large_value() {
    let s = emit_imm32_to_r10(0xDEAD_BEEF);
    assert!(
        s.contains("3735928559"),
        "expected decimal repr of 0xDEADBEEF"
    );
}

#[test]
fn relu_emits_separate_loop_with_xorps_and_maxss() {
    // Use --no-passes path so relu stays as a separate node (the default
    // pipeline fuses linear→relu and inlines the maxss inside the matmul).
    let src = "model R [b=4, k=8]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("xorps   %xmm1, %xmm1"),
        "relu must zero a scratch xmm via xorps:\n{s}"
    );
    assert!(
        s.contains("maxss   %xmm1, %xmm0"),
        "relu must compare against zero via maxss:\n{s}"
    );
    assert!(
        s.contains(".Lrelu_"),
        "relu must emit a labelled loop:\n{s}"
    );
}

#[test]
fn function_label_has_no_underscore_prefix_on_x86_64() {
    let src = "model M [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains(".globl nfl_forward_M\n"),
        "x86_64 ELF must NOT prepend underscore to .globl:\n{s}"
    );
    assert!(
        s.contains("\nnfl_forward_M:"),
        "x86_64 ELF must NOT prepend underscore to function label:\n{s}"
    );
    assert!(
        !s.contains("_nfl_forward_M"),
        "x86_64 ELF must not have any '_nfl_' (Mach-O convention):\n{s}"
    );
}

#[test]
fn relu_only_model_is_leaf_no_callee_saved_int_pushes() {
    let src = "model L [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> relu\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("    pushq   %rbp\n"),
        "frame pointer always saved:\n{s}"
    );
    assert!(
        !s.contains("    pushq   %rbx\n"),
        "leaf model must NOT save callee-saved int regs:\n{s}"
    );
}

#[test]
fn dropout_as_output_emits_copy_loop_no_maxss() {
    // dropout-as-output (model.output is the dropout node) triggers
    // emit_dropout_copy via the BufferLoc::OutputReg branch in walk_model.
    let src = "model OnlyDropout [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains(".Ldropout_"), "missing dropout loop label:\n{s}");
    assert!(
        s.contains("movss   (%rax, %rbp, 4), %xmm0"),
        "missing load:\n{s}"
    );
    assert!(
        s.contains("movss   %xmm0, (%r11, %rbp, 4)"),
        "missing store:\n{s}"
    );
    assert!(
        !s.contains("maxss"),
        "dropout-copy must NOT contain maxss:\n{s}"
    );
}

#[test]
fn linear_matmul_emits_mulss_addss_pair_no_fma() {
    let src = "model L [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[n]\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("mulss"), "matmul body needs mulss:\n{s}");
    assert!(
        s.contains("addss"),
        "matmul body needs addss (no FMA):\n{s}"
    );
    assert!(
        !s.contains("vfmadd"),
        "must NOT use FMA — scalar SSE2 only:\n{s}"
    );
}

#[test]
fn linear_with_bias_emits_addss_from_bias_buffer() {
    let src = "model B [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[n, bias=true]\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains("addss"), "bias-add via addss:\n{s}");
}

#[test]
fn linear_relu_fused_emits_inline_maxss_no_separate_loop() {
    // Default pipeline fuses linear→relu — inline maxss inside matmul body.
    let src = "model F [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[n] -> relu\n";
    let s = lower_x86(src).source;
    assert!(s.contains("maxss"), "fused relu must inline maxss:\n{s}");
    assert!(
        !s.contains(".Lrelu_"),
        "fused asm should NOT have separate relu loop:\n{s}"
    );
}

#[test]
fn linear_softmax_fused_emits_row_wise_tail_with_call_expf_plt() {
    let src = "model S [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[n] -> softmax\n";
    let s = lower_x86(src).source;
    assert!(
        s.contains(".Lfsmx_"),
        "fused softmax tail uses .Lfsmx_ labels:\n{s}"
    );
    assert!(
        s.contains("call    expf@PLT"),
        "fused softmax tail must call expf@PLT:\n{s}"
    );
}

#[test]
fn linear_softmax_fused_uses_callee_saved_int_pushes() {
    let src = "model SC [b=2, k=4, n=3]:\n    x: Tensor[b, k]\n    x -> linear[n] -> softmax\n";
    let s = lower_x86(src).source;
    assert!(
        s.contains("    pushq   %rbx\n"),
        "softmax fused needs callee-saved %rbx:\n{s}"
    );
    assert!(
        s.contains("    pushq   %r12\n"),
        "softmax fused needs callee-saved %r12:\n{s}"
    );
    assert!(
        s.contains("    pushq   %r15\n"),
        "softmax fused needs callee-saved %r15:\n{s}"
    );
}

#[test]
fn linear_matmul_uses_only_scalar_sse2_xmm_regs() {
    let src = "model V [b=2, k=4, n=2]:\n    x: Tensor[b, k]\n    x -> linear[n]\n";
    let s = lower_x86_no_passes(src).source;
    // Scalar SSE2: xmm0..xmm15 — no ymm/zmm.
    assert!(!s.contains("%ymm"), "no AVX (ymm) per spec non-goals:\n{s}");
    assert!(
        !s.contains("%zmm"),
        "no AVX-512 (zmm) per spec non-goals:\n{s}"
    );
}

// ── Task 3.10: standalone softmax asm-shape tests ────────────────────────────

#[test]
fn standalone_softmax_emits_three_pass_with_call_expf_plt() {
    let src = "model SS [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(s.contains(".Lsm_max_"), "phase 1 max label missing:\n{s}");
    assert!(s.contains(".Lsm_exp_"), "phase 2 exp label missing:\n{s}");
    assert!(s.contains(".Lsm_norm_"), "phase 3 norm label missing:\n{s}");
    assert!(
        s.contains("call    expf@PLT"),
        "softmax must call expf@PLT:\n{s}"
    );
}

#[test]
fn standalone_softmax_uses_callee_saved_int_pushes() {
    let src = "model SCS [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("    pushq   %rbx\n"),
        "softmax needs %rbx callee-saved:\n{s}"
    );
    assert!(
        s.contains("    pushq   %r15\n"),
        "softmax needs %r15 callee-saved:\n{s}"
    );
}

#[test]
fn standalone_softmax_spills_max_to_stack_at_offset_32() {
    // assign_buffers reserves bytes 0..15 for the two xmm-spill slots
    // when has_softmax; row_max sits at offset 0, row_sum at 8
    // (spec §7.4). Standalone softmax model has no intermediate buffers,
    // so the reserve is the only stack content other than alignment pad.
    //
    // M10: emit_softmax pushes %rdi/%rsi/%rdx + padding (32 bytes) at
    // entry to preserve FFI input regs across `call expf@PLT` for any
    // downstream emitter (matmul-after-softmax in self_attention). The
    // 32-byte push shifts the row_max slot from (%rsp) to 32(%rsp).
    let src = "model SP [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("movss   %xmm8, 32(%rsp)"),
        "row_max spill missing at offset 32 (post-FFI-push):\n{s}"
    );
}

#[test]
fn standalone_softmax_initialises_sum_slot_to_zero() {
    // M10: post-FFI-push offset shifted from 8(%rsp) → 40(%rsp).
    let src = "model SZ [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    assert!(
        s.contains("movl    $0, 40(%rsp)"),
        "sum slot init missing at offset 40 (post-FFI-push):\n{s}"
    );
}

#[test]
fn standalone_softmax_recomputes_offset_after_call() {
    // After call expf@PLT, the offset-holding GPR (%rax) is clobbered;
    // emitter must recompute before the next memory access.
    let src = "model SR [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> softmax\n";
    let s = lower_x86_no_passes(src).source;
    let post_call_idx = s.find("call    expf@PLT").expect("must contain call");
    let post_call = &s[post_call_idx..];
    assert!(
        post_call.contains("movq    %r15, %rax"),
        "must recompute %rax = row_base after call expf@PLT:\n{s}"
    );
}

#[test]
fn softmax_4d_dispatch_computes_b_as_product_of_leading_dims() {
    // 4D shape [2, 4, 8, 16]: b = 2*4*8 = 64, k = 16. The x86_64
    // emitter materialises b via emit_imm32_to_r10 → `movl $64, %r10d`
    // immediately above the .Lsm_i_<id> label. emit_imm32_to_r10 prints
    // the immediate in decimal (see profiles/x86_64/src/asm.rs:28), so
    // 64 appears as `$64`, not `$0x40`.
    let src = "\
model M [batch=2, heads=4, seq=8, dim=16]:
    x: Tensor[batch, heads, seq, dim]

    y: Tensor[batch, heads, seq, dim] = x -> softmax
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;
    assert!(
        asm.contains("movl    $64, %r10d"),
        "expected b=64 immediate; asm:\n{}",
        asm
    );
}

// ── Task 3.11: mirrors of arm64 unit-test coverage ───────────────────────────

#[test]
fn empty_uir_lowers_to_empty_asm() {
    let uir = compiler::Uir { models: Vec::new() };
    let asm = crate::lower(&uir).unwrap();
    assert!(asm.source.is_empty());
    assert!(asm.functions.is_empty());
}

#[test]
fn linear_emits_function_with_correct_symbol_and_ret() {
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");

    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M");
    assert_eq!(sig.model, "M");
    assert_eq!(sig.inputs_floats, vec![6]);
    assert_eq!(sig.params_floats, 6);
    assert_eq!(sig.output_floats, 4);

    assert_eq!(sig.params_layout.len(), 1);
    let slot = &sig.params_layout[0];
    use crate::ParamKind;
    assert_eq!(slot.kind, ParamKind::LinearWeight);
    assert_eq!(slot.offset, 0);
    assert_eq!(slot.size, 6);
    assert_eq!(slot.origin_node, 1);

    let s = &asm.source;
    assert!(s.contains(".globl nfl_forward_M"));
    assert!(s.contains("nfl_forward_M:"));
    assert!(s.contains("retq"));
}

#[test]
fn linear_emits_matmul_loops_with_movss_no_fmadd() {
    // x86_64 uses mulss+addss — no fmadd.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    // Labels use model_idx prefix: model 0, linear 0 → "0_0".
    assert!(s.contains(".Lmm_i_0_0:"), "i-loop label missing:\n{s}");
    assert!(s.contains(".Lmm_j_0_0:"), "j-loop label missing:\n{s}");
    assert!(s.contains(".Lmm_k_0_0:"), "k-loop label missing:\n{s}");
    assert!(s.contains("movss"), "load via movss missing:\n{s}");
    assert!(s.contains("mulss"), "multiply via mulss missing:\n{s}");
    assert!(!s.contains("fmadd"), "x86_64 must not use fmadd:\n{s}");
    // Destination written via movss store.
    assert!(
        s.contains("movss   %xmm0,"),
        "store via movss missing:\n{s}"
    );
}

#[test]
fn relu_alone_after_matmul_does_not_break_matmul() {
    // Sanity: matmul still emitted alongside relu.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    assert!(
        asm.source.contains("mulss"),
        "matmul must survive alongside relu"
    );
}

#[test]
fn dropout_emits_no_code_between_two_linears() {
    // input → linear → dropout → linear (terminal-linear). Dropout carries
    // BufferLoc::Alias and emits nothing; only two linear matmuls appear.
    let ast = compiler::parse(
        "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[3] -> dropout[rate=0.2] -> linear[2]\n",
    )
    .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    // Two linear matmuls present (model 0 → "0_0" and "0_1").
    assert!(s.contains(".Lmm_i_0_0:"), "first matmul missing:\n{s}");
    assert!(s.contains(".Lmm_i_0_1:"), "second matmul missing:\n{s}");
    // No dropout-specific instructions or labels.
    assert!(
        !s.contains("dropout"),
        "asm must not mention dropout literally:\n{s}"
    );
}

// ── buffer analyzer mirrors ───────────────────────────────────────────────────

#[test]
fn assign_buffers_input_node_is_input_reg() {
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[0], BufferLoc::InputReg(0)));
}

#[test]
fn assign_buffers_terminal_node_is_output_reg() {
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    let last = assignment.locs.last().unwrap();
    assert!(matches!(last, BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_relu_aliases_operand() {
    // n0 input, n1 linear (non-terminal), n2 relu (terminal).
    // Expected: n2 → OutputReg; n1 → StackOffset.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_intermediate_relu_aliases_operand() {
    // input → linear → relu → linear → relu (terminal).
    // Intermediate relu (n2) aliases linear (n1); terminal relu (n4) is OutputReg.
    let ast = compiler::parse(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    )
    .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(matches!(assignment.locs[1], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[2], BufferLoc::Alias(1)));
    assert!(matches!(assignment.locs[3], BufferLoc::StackOffset(_)));
    assert!(matches!(assignment.locs[4], BufferLoc::OutputReg));
}

#[test]
fn assign_buffers_stack_bytes_is_16_aligned() {
    let ast = compiler::parse(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    )
    .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert!(assignment.stack_bytes > 0);
    assert_eq!(assignment.stack_bytes % 16, 0, "stack must be 16-aligned");
}

#[test]
fn compute_is_leaf_true_for_linear_relu() {
    // No extern math call (no softmax) → callee_saved_int is false → leaf.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> relu\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(
        !regs.contains_callee_saved_int(),
        "linear+relu is a leaf — no callee-saved int regs needed"
    );
}

#[test]
fn compute_is_leaf_false_when_softmax_present() {
    // Softmax → calls expf@PLT → callee_saved_int true → non-leaf.
    let ast =
        compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n")
            .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let regs = compute_callee_saved(&uir.models[0]);
    assert!(
        regs.contains_callee_saved_int(),
        "softmax requires callee-saved int regs (not a leaf)"
    );
}

#[test]
fn assign_buffers_stack_bytes_rounds_non_aligned_total_up() {
    // Tensor[1, 2] -> linear[3] -> linear[3]:
    //   n0 input (no slot), n1 linear (1*3=3 floats=12 bytes, non-terminal),
    //   n2 linear (terminal → OutputReg, no slot)
    // Total raw stack = 12 bytes; rounded up to 16.
    let ast =
        compiler::parse("model M [b=1]:\n    x: Tensor[b, 2]\n    x -> linear[3] -> linear[3]\n")
            .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    let assignment = assign_buffers(model);
    assert_eq!(
        assignment.stack_bytes, 16,
        "12 raw bytes should round up to 16"
    );
}

#[test]
fn leaf_function_no_sub_rsp_for_leaf_no_intermediates() {
    // input → linear (terminal): leaf, no intermediates, no sub rsp.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    // No callee-saved int registers pushed (no %rbx).
    assert!(
        !s.contains("    pushq   %rbx\n"),
        "leaf-no-intermediates should not save %rbx:\n{s}"
    );
    // No stack frame sub needed for zero intermediate bytes (modulo alignment).
    // A leaf with no intermediates has stack_bytes=0 → frame_size from
    // alignment correction only; with 1 push (%rbp) → odd → no correction.
    assert!(
        !s.contains("    subq    $"),
        "leaf with no intermediate buffers should emit no subq:\n{s}"
    );
}

#[test]
fn intermediate_buffers_allocated_on_stack() {
    let ast = compiler::parse(
        "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8] -> relu -> linear[2] -> relu\n",
    )
    .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        s.contains("    subq    $"),
        "expected subq to open frame:\n{s}"
    );
    assert!(
        s.contains("    addq    $"),
        "expected addq to close frame:\n{s}"
    );
}

#[test]
fn linear_bias_packed_layout() {
    let ast =
        compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true]\n")
            .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    use crate::{ParamKind, ParamSlot};
    // Two slots: LinearWeight (size 6) then LinearBias (size 2) at offset 6.
    assert_eq!(sig.params_layout.len(), 2);
    assert_eq!(sig.params_layout[0].kind, ParamKind::LinearWeight);
    assert_eq!(sig.params_layout[0].size, 6);
    assert_eq!(sig.params_layout[1].kind, ParamKind::LinearBias);
    assert_eq!(sig.params_layout[1].size, 2);
    assert_eq!(sig.params_layout[1].offset, 6);
    assert_eq!(sig.params_floats, 8);
    // Suppress unused import warning.
    let _: Option<&ParamSlot> = None;
}

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
fn fused_linear_relu_emits_maxss_before_store() {
    use compiler::{NodeKind, PostOp};
    // Hand-build UIR where Linear has fused_post_ops = [Relu].
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let mut uir = ir::build(&ast).expect("ir::build");
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else {
        panic!("expected Op node");
    };
    fused_post_ops.push(PostOp::Relu);

    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;

    // xmm4 zeroed once (materialised outside inner loop in linear.rs).
    assert!(
        s.contains("xorps   %xmm4, %xmm4"),
        "missing xmm4 zero materialisation:\n{s}"
    );
    // maxss inline before store.
    assert!(
        s.contains("maxss   %xmm4, %xmm0"),
        "missing inline maxss (relu):\n{s}"
    );
}

#[test]
fn fused_linear_relu_no_separate_relu_loop() {
    use compiler::{NodeKind, PostOp};
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let mut uir = ir::build(&ast).expect("ir::build");
    let m = &mut uir.models[0];
    let NodeKind::Op { fused_post_ops, .. } = &mut m.nodes[1].kind else {
        panic!()
    };
    fused_post_ops.push(PostOp::Relu);

    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        !s.contains(".Lrelu_"),
        "fused linear+relu should NOT emit a separate relu loop:\n{s}"
    );
}

#[test]
fn unfused_linear_still_no_maxss() {
    // Linear without fused_post_ops: no maxss AND no xmm4 zero-materialisation.
    let ast = compiler::parse("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n")
        .expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let asm = crate::lower(&uir).expect("lower");
    let s = &asm.source;
    assert!(
        !s.contains("maxss"),
        "un-fused linear should NOT emit maxss:\n{s}"
    );
    assert!(
        !s.contains("xorps   %xmm4, %xmm4"),
        "un-fused linear should NOT materialise xmm4 zero:\n{s}"
    );
}

#[test]
fn is_leaf_false_for_fused_softmax_row_linear() {
    use compiler::passes::{default_pipeline, run_pipeline};
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline");
    let model = &fused.models[0];
    let regs = compute_callee_saved(model);
    assert!(
        regs.contains_callee_saved_int(),
        "a Linear carrying PostOp::SoftmaxRow still calls expf@PLT — not a leaf"
    );
}

#[test]
fn emit_linear_with_softmax_row_post_op_emits_three_phase_softmax() {
    use compiler::passes::{default_pipeline, run_pipeline};
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2] -> softmax\n";
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline");
    let asm = crate::lower(&fused).expect("lower");
    let s = &asm.source;

    // Phase 1 — matmul: mulss must appear.
    assert!(s.contains("mulss"), "Phase 1 matmul missing:\n{s}");

    // Phase 2 — row-max scan: maxss on xmm8.
    assert!(
        s.contains("maxss   %xmm0, %xmm8"),
        "Phase 2 row-max scan into xmm8 missing:\n{s}"
    );

    // Phase 3 — exp(x - max), sum, with call expf@PLT.
    assert!(
        s.contains("call    expf@PLT"),
        "Phase 3 missing call expf@PLT:\n{s}"
    );
    assert!(
        s.contains("addss   %xmm0, %xmm1"),
        "Phase 3 sum accumulation missing:\n{s}"
    );

    // Phase 4 — normalise by row_sum.
    assert!(
        s.contains("divss   8(%rsp), %xmm0"),
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
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2, bias=true] -> softmax\n";
    let ast = compiler::parse(src).expect("parse");
    let uir = ir::build(&ast).expect("ir::build");
    let fused = run_pipeline(&uir, &default_pipeline()).expect("pipeline");
    let asm = crate::lower(&fused).expect("lower");
    let s = &asm.source;

    // Phase 1 still emits matmul → bias-add via addss.
    assert!(
        s.contains("addss   %xmm5, %xmm0"),
        "bias-add missing in fused row-wise emit:\n{s}"
    );
    // Phase 3 still calls expf@PLT.
    assert!(
        s.contains("call    expf@PLT"),
        "fused softmax tail missing call expf@PLT:\n{s}"
    );
}

// ---- Group 9a: emit_matmul (x86_64) -----------------------------------------

#[test]
fn matmul_4d_emits_outer_loop_wrapper() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("build");
    let asm = crate::lower(&uir).expect("lower");
    // Outer loop wrapper present.
    assert!(
        asm.source.contains(".Lmm4d_outer_0_0:"),
        "asm:\n{}",
        asm.source
    );
    assert!(
        asm.source.contains(".Lmm4d_outer_end_0_0:"),
        "asm:\n{}",
        asm.source
    );
    // Inner triple-loop labels present.
    assert!(asm.source.contains(".Lmm4d_i_0_0:"), "asm:\n{}", asm.source);
    assert!(asm.source.contains(".Lmm4d_j_0_0:"), "asm:\n{}", asm.source);
    assert!(asm.source.contains(".Lmm4d_k_0_0:"), "asm:\n{}", asm.source);
}

#[test]
fn matmul_2d_collapses_to_outer_count_one() {
    let src = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("build");
    let asm = crate::lower(&uir).expect("lower");
    // The outer loop is still emitted, but its bound is 1; assert
    // structurally on the comment header (more readable).
    assert!(
        asm.source.contains("leading_count=1"),
        "asm:\n{}",
        asm.source
    );
}

#[test]
fn matmul_transpose_b_inner_addressing_differs() {
    let src_no_t = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    // For a transpose_b version we need shapes that match: [batch, 4]
    // with b transposed [N, K] = [8, 4].
    let src_t = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[8, 4]

    out: Tensor[batch, 8] = a -> matmul[b, transpose_b=true]
";
    let asm_no_t = crate::lower(&compiler::ir::build(&compiler::parse(src_no_t).unwrap()).unwrap())
        .expect("lower no-t")
        .source;
    let asm_t = crate::lower(&compiler::ir::build(&compiler::parse(src_t).unwrap()).unwrap())
        .expect("lower t")
        .source;
    // Both must use mulss (no FMA on x86_64).
    assert!(asm_no_t.contains("mulss"), "no-t asm:\n{}", asm_no_t);
    assert!(asm_t.contains("mulss"), "t asm:\n{}", asm_t);
    // M13 register layout: k_inner counter %r11, j counter %rbp.
    // Transpose flips inner b_offset computation:
    //   no-transpose: `movq    %r11, %rax` (k_inner * N + j; %rax = k_inner)
    //   transpose:    `movq    %rbp, %rax` (j * K + k_inner;  %rax = j)
    assert!(
        asm_no_t.contains("movq    %r11, %rax"),
        "no-t asm should compute b_offset from %r11 (k_inner):\n{}",
        asm_no_t
    );
    assert!(
        asm_t.contains("movq    %rbp, %rax"),
        "t asm should compute b_offset from %rbp (j):\n{}",
        asm_t
    );
    assert_ne!(asm_no_t, asm_t, "transpose_b should change emitted asm");
}

#[test]
fn matmul_transpose_b_false_default_matches_explicit_false() {
    // Spec §8.3 — guard against drift between the omit-attr code path
    // and the explicit-false path.
    let src_default = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let src_explicit = "\
model M [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b, transpose_b=false]
";
    let asm_d = crate::lower(&compiler::ir::build(&compiler::parse(src_default).unwrap()).unwrap())
        .expect("lower default")
        .source;
    let asm_e =
        crate::lower(&compiler::ir::build(&compiler::parse(src_explicit).unwrap()).unwrap())
            .expect("lower explicit")
            .source;
    assert_eq!(asm_d, asm_e, "default omit must match explicit false");
}

#[test]
fn matmul_uses_mulss_addss_no_fma() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;
    assert!(asm.contains("mulss"), "asm:\n{}", asm);
    assert!(asm.contains("addss"), "asm:\n{}", asm);
    assert!(
        !asm.contains("vfmadd"),
        "matmul must not use FMA on x86_64; asm:\n{}",
        asm
    );
}

#[test]
fn matmul_does_not_call_expf_plt() {
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;
    assert!(!asm.contains("expf@PLT"), "asm:\n{}", asm);
}

#[test]
fn matmul_preserves_ffi_register_invariants_no_spill() {
    // M12 (spec §9.1) regression test, INVERTED from M11: emit_matmul
    // must NOT spill any ABI argument register, because under the
    // multi-input ABI %rsi/%rdx/%rcx/%r8 may hold input pointers
    // downstream emitters need to read intact.
    //
    // The pre-M12 spill pair `movq %rdi, %xmm8` / `movq %xmm8, %rdi`
    // (and the matching %rsi/%rdx pairs) is gone; per-iter slice
    // pointers move to non-ABI scratch (callee-saved %r12/%r13/%r14).
    // emit_matmul does not call FFI, so no stack manipulation should
    // appear in its body. Asserted by `emit_matmul_body_contains_zero_pushq`
    // below; this test guards the legacy fixture-shape model end-to-end.
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;

    // The matmul body should contain no spill of an ABI arg register
    // INTO an %xmm scratch reg. The function-level prologue may push
    // callee-saved %rbx/%r12-%r15 (because matmul triggers callee-saved
    // per `compute_callee_saved`), but no `movq %rdi, %xmm…`/etc.
    // The matmul-only fixture above is leaf w.r.t. FFI (no softmax),
    // so no FFI-call spill block either.
    assert!(
        !asm.contains("movq    %rdi, %xmm"),
        "M12 emit_matmul must not spill %rdi; asm:\n{}",
        asm
    );
    assert!(
        !asm.contains("movq    %rsi, %xmm"),
        "M12 emit_matmul must not spill %rsi; asm:\n{}",
        asm
    );
    assert!(
        !asm.contains("movq    %rdx, %xmm"),
        "M12 emit_matmul must not spill %rdx; asm:\n{}",
        asm
    );
    assert!(
        !asm.contains("movq    %xmm8, %rdi"),
        "M12 emit_matmul must not restore %rdi; asm:\n{}",
        asm
    );
    assert!(
        !asm.contains("movq    %xmm6, %rsi"),
        "M12 emit_matmul must not restore %rsi; asm:\n{}",
        asm
    );
    assert!(
        !asm.contains("movq    %xmm7, %rdx"),
        "M12 emit_matmul must not restore %rdx; asm:\n{}",
        asm
    );
}

#[test]
fn mul_scalar_uses_mulss() {
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.5]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;
    assert!(asm.contains("mulss   %xmm4, %xmm0"), "asm:\n{}", asm);
}

#[test]
fn mul_scalar_preloads_scalar() {
    // 0.25 in f32 bits = 0x3E800000.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> mul_scalar[0.25]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;
    assert!(asm.contains("movl    $0x3e800000, %r10d"), "asm:\n{}", asm);
    assert!(asm.contains("movd    %r10d, %xmm4"), "asm:\n{}", asm);
}

/// N=1 regression invariant: every existing fixture must compile to
/// the EXACT same assembly as the committed goldens in tests/golden/
///
/// This test loops over a list of fixtures and asserts byte-exact
/// equality with the corresponding golden file. Spec §10.2 gate #4.
///
/// Mirrors the exact nflc compile pipeline: parse → build → default
/// passes → lower. Golden files were generated by nflc compile with
/// defaults (passes enabled).
#[test]
fn n1_regression_all_fixtures_bit_exact() {
    use compiler::passes::{default_pipeline, run_pipeline};

    let fixtures = [
        "tiny_mlp",
        "m4_linear_relu",
        "mixed_args",
        "softmax_with_bias",
        "dropout_only",
        "classifier",
        "large_classifier_k",
        "large_classifier_n",
        "pipeline_styles",
        "comments",
        "self_attention",
    ];
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    for f in fixtures {
        let nfl_path = format!("{manifest_dir}/../../tests/fixtures/{f}.nfl");
        let golden_path = format!("{manifest_dir}/tests/golden/{f}.s");
        let src =
            std::fs::read_to_string(&nfl_path).unwrap_or_else(|e| panic!("read {nfl_path}: {e}"));
        let nfl = compiler::parse(&src).unwrap();
        let uir = compiler::ir::build(&nfl).unwrap();
        let uir =
            run_pipeline(&uir, &default_pipeline()).unwrap_or_else(|e| panic!("pipeline {f}: {e}"));
        let asm = crate::X86_64Profile.lower(&uir).unwrap().source;
        let golden = std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("read {golden_path}: {e}"));
        if asm != golden {
            // Show first diverging line for diagnostic.
            for (i, (a, g)) in asm.lines().zip(golden.lines()).enumerate() {
                if a != g {
                    panic!(
                        "fixture {f}: divergence at line {}\n  generated: {a:?}\n  golden:    {g:?}",
                        i + 1
                    );
                }
            }
            // Fallback if length differs but prefix matches.
            panic!(
                "fixture {f}: assembly differs from golden (length: gen={}, golden={})",
                asm.len(),
                golden.len()
            );
        }
    }
}

// ---------------------------------------------------------------------
// M12 AbiContext unit tests — alignment, LIFO, ffi_save_set, materialise_ptr.
// ---------------------------------------------------------------------

use crate::abi::AbiContext;

#[test]
fn abi_input_reg_n1() {
    let abi = AbiContext { n_inputs: 1 };
    assert_eq!(abi.input_reg(0), "%rdi");
}

#[test]
fn abi_input_reg_n3() {
    let abi = AbiContext { n_inputs: 3 };
    assert_eq!(abi.input_reg(0), "%rdi");
    assert_eq!(abi.input_reg(1), "%rsi");
    assert_eq!(abi.input_reg(2), "%rdx");
}

#[test]
fn abi_params_reg_shifts_with_arity() {
    assert_eq!(AbiContext { n_inputs: 1 }.params_reg(), "%rsi");
    assert_eq!(AbiContext { n_inputs: 2 }.params_reg(), "%rdx");
    assert_eq!(AbiContext { n_inputs: 3 }.params_reg(), "%rcx");
    assert_eq!(AbiContext { n_inputs: 4 }.params_reg(), "%r8");
}

#[test]
fn abi_output_reg_shifts_with_arity() {
    assert_eq!(AbiContext { n_inputs: 1 }.output_reg(), "%rdx");
    assert_eq!(AbiContext { n_inputs: 2 }.output_reg(), "%rcx");
    assert_eq!(AbiContext { n_inputs: 3 }.output_reg(), "%r8");
    assert_eq!(AbiContext { n_inputs: 4 }.output_reg(), "%r9");
}

#[test]
fn abi_ffi_save_set_size_equals_n_plus_2() {
    for n in 1..=4 {
        assert_eq!(
            AbiContext { n_inputs: n }.ffi_save_set().len(),
            n + 2,
            "n={n}"
        );
    }
}

#[test]
fn abi_ffi_save_set_contents_n3() {
    let abi = AbiContext { n_inputs: 3 };
    assert_eq!(abi.ffi_save_set(), &["%rdi", "%rsi", "%rdx", "%rcx", "%r8"]);
}

#[test]
fn abi_materialise_input_n1() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::InputReg(0), "%r10", &mut s);
    assert!(s.contains("movq    %rdi, %r10"), "got: {s}");
}

#[test]
fn abi_materialise_input_n3_idx2() {
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::InputReg(2), "%r11", &mut s);
    assert!(s.contains("movq    %rdx, %r11"), "got: {s}");
}

#[test]
fn abi_materialise_output_n2() {
    let abi = AbiContext { n_inputs: 2 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::OutputReg, "%r12", &mut s);
    // N=2 → output is %rcx (= INPUT_REGS[2 + 1]).
    assert!(s.contains("movq    %rcx, %r12"), "got: {s}");
}

#[test]
fn abi_materialise_stack_offset_small() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::StackOffset(64), "%r12", &mut s);
    assert!(s.contains("leaq    64(%rsp), %r12"), "got: {s}");
}

#[test]
fn abi_materialise_stack_offset_zero_uses_movq_rsp() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.materialise_ptr(BufferLoc::StackOffset(0), "%r12", &mut s);
    assert!(s.contains("movq    %rsp, %r12"), "got: {s}");
}

#[test]
fn abi_emit_ffi_save_n1_three_regs_pads_rax() {
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("pushq   %rdi"), "got:\n{s}");
    assert!(s.contains("pushq   %rsi"), "got:\n{s}");
    assert!(s.contains("pushq   %rdx"), "got:\n{s}");
    assert!(s.contains("pushq   %rax"), "got:\n{s}");
    let push_count = s.matches("pushq").count();
    assert_eq!(
        push_count, 4,
        "3 input/params/output + 1 padding; got {push_count}"
    );
}

#[test]
fn abi_emit_ffi_save_n2_four_regs_no_pad() {
    let abi = AbiContext { n_inputs: 2 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("pushq   %rdi"));
    assert!(s.contains("pushq   %rsi"));
    assert!(s.contains("pushq   %rdx"));
    assert!(s.contains("pushq   %rcx"));
    // No padding push when arity is even.
    assert!(
        !s.contains("padding"),
        "no %rax padding for even arity; got:\n{s}"
    );
    assert_eq!(s.matches("pushq").count(), 4);
}

#[test]
fn abi_emit_ffi_save_n3_five_regs_pads_rax() {
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("pushq   %rdi"));
    assert!(s.contains("pushq   %rsi"));
    assert!(s.contains("pushq   %rdx"));
    assert!(s.contains("pushq   %rcx"));
    assert!(s.contains("pushq   %r8"));
    assert!(s.contains("pushq   %rax"));
    assert_eq!(
        s.matches("pushq").count(),
        6,
        "5 input/params/output + 1 padding"
    );
}

#[test]
fn abi_emit_ffi_save_n4_six_regs_no_pad() {
    let abi = AbiContext { n_inputs: 4 };
    let mut s = String::new();
    abi.emit_ffi_save(&mut s);
    assert!(s.contains("pushq   %rdi"));
    assert!(s.contains("pushq   %rsi"));
    assert!(s.contains("pushq   %rdx"));
    assert!(s.contains("pushq   %rcx"));
    assert!(s.contains("pushq   %r8"));
    assert!(s.contains("pushq   %r9"));
    assert!(!s.contains("padding"), "no padding for even arity");
    assert_eq!(s.matches("pushq").count(), 6);
}

#[test]
fn abi_emit_ffi_save_sp_delta_always_multiple_of_16() {
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut s = String::new();
        abi.emit_ffi_save(&mut s);
        // Each `pushq` decrements rsp by 8.
        let push_count = s.matches("pushq").count();
        let sp_delta = push_count * 8;
        assert!(sp_delta.is_multiple_of(16), "n={n} sp_delta={sp_delta}");
        // Also: push_count == n+2 + (1 if odd else 0).
        let expected = (n + 2) + if (n + 2).is_multiple_of(2) { 0 } else { 1 };
        assert_eq!(push_count, expected, "n={n}: parity check");
    }
}

#[test]
fn abi_emit_ffi_restore_n1_lifo() {
    // Save order: pushq %rdi; pushq %rsi; pushq %rdx; pushq %rax (pad).
    // Restore order (LIFO): popq %rax (discard pad); popq %rdx; popq %rsi; popq %rdi.
    let abi = AbiContext { n_inputs: 1 };
    let mut s = String::new();
    abi.emit_ffi_restore(&mut s);
    let pos_pad = s.find("popq    %rax").expect("popq %rax (pad)");
    let pos_rdx = s.find("popq    %rdx").expect("popq %rdx");
    let pos_rsi = s.find("popq    %rsi").expect("popq %rsi");
    let pos_rdi = s.find("popq    %rdi").expect("popq %rdi");
    assert!(
        pos_pad < pos_rdx,
        "LIFO: padding pop comes first; got:\n{s}"
    );
    assert!(pos_rdx < pos_rsi, "LIFO: %rdx pop before %rsi");
    assert!(pos_rsi < pos_rdi, "LIFO: %rsi pop before %rdi");
}

#[test]
fn abi_emit_ffi_restore_n3_lifo() {
    // Save: pushq %rdi/%rsi/%rdx/%rcx/%r8 + pushq %rax (pad).
    // Restore (LIFO): popq %rax (pad); popq %r8; popq %rcx; popq %rdx; popq %rsi; popq %rdi.
    let abi = AbiContext { n_inputs: 3 };
    let mut s = String::new();
    abi.emit_ffi_restore(&mut s);
    let pad = s.find("popq    %rax").expect("padding pop");
    let r8 = s.find("popq    %r8").expect("%r8 pop");
    let rcx = s.find("popq    %rcx").expect("%rcx pop");
    let rdx = s.find("popq    %rdx").expect("%rdx pop");
    let rsi = s.find("popq    %rsi").expect("%rsi pop");
    let rdi = s.find("popq    %rdi").expect("%rdi pop");
    assert!(pad < r8, "LIFO: pad before %r8");
    assert!(r8 < rcx, "LIFO: %r8 before %rcx");
    assert!(rcx < rdx, "LIFO: %rcx before %rdx");
    assert!(rdx < rsi, "LIFO: %rdx before %rsi");
    assert!(rsi < rdi, "LIFO: %rsi before %rdi");
}

#[test]
fn abi_save_then_restore_balances_sp() {
    // Number of pushq == number of popq.
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut save = String::new();
        let mut restore = String::new();
        abi.emit_ffi_save(&mut save);
        abi.emit_ffi_restore(&mut restore);
        assert_eq!(
            save.matches("pushq").count(),
            restore.matches("popq").count(),
            "save/restore mismatch at n={n}"
        );
    }
}

#[test]
fn abi_save_set_each_reg_appears_exactly_once_in_save_and_restore() {
    for n in 1..=4 {
        let abi = AbiContext { n_inputs: n };
        let mut save = String::new();
        let mut restore = String::new();
        abi.emit_ffi_save(&mut save);
        abi.emit_ffi_restore(&mut restore);
        for &reg in abi.ffi_save_set() {
            assert_eq!(save.matches(reg).count(), 1, "n={n} reg={reg} save");
            assert_eq!(restore.matches(reg).count(), 1, "n={n} reg={reg} restore");
        }
    }
}

#[test]
fn emit_matmul_body_contains_zero_pushq() {
    // Spec §9.1: emit_matmul does not call FFI; only AbiContext::emit_ffi_save
    // emits stack manipulation. After the M12 rework, emit_matmul body must
    // contain zero `pushq` instructions. The function-level prologue (callee-
    // saved regs %rbx/%r12-%r15) emits its own pushq — but that's outside
    // emit_matmul. Mirror of arm64's `emit_matmul_body_contains_zero_stp`.
    use compiler::ast::Span;
    let abi = AbiContext { n_inputs: 2 };
    let span = Span::new(0, 0);
    let result = crate::ops::matmul::emit_matmul(
        &abi,
        /* leading_count */ 1,
        /* m */ 4,
        /* k */ 8,
        /* n */ 4,
        /* transpose_b */ false,
        /* model_idx */ 0,
        /* matmul_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* b_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
        span,
    )
    .expect("emit_matmul should succeed");
    let pushq_count = result.matches("pushq").count();
    assert_eq!(
        pushq_count, 0,
        "emit_matmul body must contain zero pushq instructions per §9.1; got {pushq_count}\n{result}"
    );
}

// ---- M13 Group D: emit_add x86_64 ----------------------------------------

#[test]
fn emit_add_x86_64_emits_two_loads_one_addss_one_store() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 2 };
    let asm = crate::ops::add::emit_add(
        &abi,
        /* total_elements */ 8,
        /* model_idx */ 0,
        /* op_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* other_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
    );
    // Two scalar loads (one per input pointer).
    assert!(
        asm.contains("movss   (%rax,"),
        "expected movss from %rax (a_ptr); got:\n{asm}"
    );
    assert!(
        asm.contains("movss   (%r10,"),
        "expected movss from %r10 (other_ptr); got:\n{asm}"
    );
    // One addss.
    assert_eq!(
        asm.matches("addss").count(),
        1,
        "expected exactly one addss; got:\n{asm}"
    );
    // One movss store to %r11.
    assert!(
        asm.contains("movss   %xmm0, (%r11,"),
        "expected movss store to %r11; got:\n{asm}"
    );
}

#[test]
fn emit_add_x86_64_uses_rbp_counter_no_pushq_no_abi_clobber() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    // emit_add must preserve all ABI argument registers across its body
    // (M12 §9.1 invariant — downstream emitters re-read input/params/output
    // from those registers). %rbp is callee-saved by the function-level
    // prologue and unread by op bodies, so emit_add uses it as the counter.
    //
    // Cover the full N ∈ [2, 4] range; at each arity, output_reg lands on
    // a different ABI register (%rcx at N=2, %r8 at N=3, %r9 at N=4) so the
    // "no ABI clobber" invariant must hold across all of them.
    for n_inputs in [2, 3, 4] {
        let abi = AbiContext { n_inputs };
        let asm = crate::ops::add::emit_add(
            &abi,
            16,
            0,
            0,
            BufferLoc::InputReg(0),
            BufferLoc::InputReg(1),
            BufferLoc::OutputReg,
        );
        // Counter init goes to %rbp.
        assert!(
            asm.contains("movq    $0, %rbp\n"),
            "N={n_inputs}: expected counter init in %rbp; got:\n{asm}"
        );
        // No pushq/popq — %rbp is already saved by function-level prologue.
        assert!(
            !asm.contains("pushq"),
            "N={n_inputs}: emit_add must not pushq inside body; got:\n{asm}"
        );
        assert!(
            !asm.contains("popq"),
            "N={n_inputs}: emit_add must not popq inside body; got:\n{asm}"
        );
        // No ABI argument register is written across N ∈ [2, 4]. Tracks
        // INPUT_REGS = [%rdi, %rsi, %rdx, %rcx, %r8, %r9]; the first
        // n_inputs+2 are reserved at each arity.
        for reg in &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"] {
            assert!(
                !asm.contains(&format!(", {reg}\n")),
                "N={n_inputs}: emit_add must not write to ABI register {reg}; got:\n{asm}"
            );
        }
    }
}

#[test]
fn emit_add_x86_64_no_callee_saved_or_ffi_save() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    let abi = AbiContext { n_inputs: 1 };
    let asm = crate::ops::add::emit_add(
        &abi,
        4,
        0,
        0,
        BufferLoc::InputReg(0),
        BufferLoc::InputReg(0),
        BufferLoc::OutputReg,
    );
    // No call to expf@PLT (no FFI save needed inside emit_add).
    assert!(
        !asm.contains("call    expf@PLT"),
        "emit_add must not call expf; got:\n{asm}"
    );
    // No %rbx/%r12-%r15 writes (matmul-only callee-saved set).
    for reg in &["%rbx", "%r12", "%r13", "%r14", "%r15"] {
        assert!(
            !asm.contains(&format!(", {reg}\n")),
            "emit_add must not write to callee-saved {reg}; got:\n{asm}"
        );
    }
}

// ---- M13 PR-fix / M14 LH-1 update: emit_linear x86_64 ABI register save at N≥2 -----

#[test]
fn emit_linear_x86_64_save_block_balances_at_all_n() {
    // M13 (PR follow-up): x86_64 emit_linear's body clobbers %rdi/%rsi
    // (k-counter, offset scratch). At N=1 these are non-ABI; at N≥2 they
    // overlap with input(0)/input(1). The fix: pushq save at body entry,
    // popq restore at body exit, conditional on n_inputs ≥ 2.
    //
    // M14 LH-1 update: %rcx save REMOVED from the block (j-counter relocated
    // to %rbp by M14 LH-1 fix; body no longer writes %rcx). The op-local
    // push/pop for %r14 and %r15 (LH-2/3) are added unconditionally.
    //
    // This test verifies push/pop balance at every N ∈ [1, 4] and pins
    // the exact register set for N≥2 (post-M14):
    //   N=1 → 2 pushq/popq pairs (op-local %r14/%r15 only; no ABI save needed).
    //   N=2..4 → 4 pushq + 4 popq (ABI save %rdi/%rsi + op-local %r14/%r15).
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    use compiler::ast::Span;
    use compiler::PostOp;
    let cases = [(1usize, 2usize), (2, 4), (3, 4), (4, 4)];
    for &(n_inputs, expected_pushes) in &cases {
        let abi = AbiContext { n_inputs };
        let post: Vec<PostOp> = vec![];
        let asm = crate::ops::emit_linear(
            &abi,
            /* b */ 2,
            /* k */ 4,
            /* n */ 4,
            /* model_idx */ 0,
            /* linear_idx */ 0,
            /* src_loc */ BufferLoc::InputReg(0),
            /* dst_loc */ BufferLoc::OutputReg,
            /* weight_offset */ 0,
            /* bias_offset */ None,
            /* node_span */ Span::new(1, 1),
            /* fused_post_ops */ &post,
            /* sym_prefix */ "",
        )
        .expect("emit_linear must succeed");
        let pushq_count = asm.matches("    pushq   ").count();
        let popq_count = asm.matches("    popq    ").count();
        assert_eq!(
            pushq_count, expected_pushes,
            "N={n_inputs}: expected {expected_pushes} pushq; got {pushq_count}\n{asm}"
        );
        assert_eq!(
            popq_count, expected_pushes,
            "N={n_inputs}: expected {expected_pushes} popq; got {popq_count}\n{asm}"
        );
        // Op-local %r14/%r15 save/restore — always present (M14 LH-2/3).
        for reg in &["%r14", "%r15"] {
            assert!(
                asm.contains(&format!("    pushq   {reg}\n")),
                "N={n_inputs}: expected `pushq {reg}` (op-local LH-2/3 save); got:\n{asm}"
            );
            assert!(
                asm.contains(&format!("    popq    {reg}\n")),
                "N={n_inputs}: expected `popq {reg}` (op-local LH-2/3 restore); got:\n{asm}"
            );
        }
        if n_inputs >= 2 {
            // ABI save: %rdi and %rsi (M14: %rcx removed — j-counter now %rbp).
            for reg in &["%rdi", "%rsi"] {
                assert!(
                    asm.contains(&format!("    pushq   {reg}\n")),
                    "N={n_inputs}: expected `pushq {reg}`; got:\n{asm}"
                );
                assert!(
                    asm.contains(&format!("    popq    {reg}\n")),
                    "N={n_inputs}: expected `popq {reg}`; got:\n{asm}"
                );
            }
            // %rcx must NOT be in the save block post-M14 LH-1 fix.
            assert!(
                !asm.contains("    pushq   %rcx\n"),
                "N={n_inputs}: %rcx must no longer be in the save block (M14 LH-1 fix); got:\n{asm}"
            );
            // LIFO check: pop %rsi before %rdi (reverse of push order %rdi, %rsi).
            let pop_rsi = asm.find("    popq    %rsi\n").expect("popq %rsi");
            let pop_rdi = asm.find("    popq    %rdi\n").expect("popq %rdi");
            assert!(
                pop_rsi < pop_rdi,
                "N={n_inputs}: popq order must be LIFO (%rsi before %rdi); got {pop_rsi}/{pop_rdi}\n{asm}"
            );
        }
    }
}

// ---- Group A (M13): N=4 + matmul fix via %rbp j-counter --------------------

#[test]
fn emit_matmul_accepts_n4_with_rbp_j_counter() {
    // Group A (M13): the M12 reject path (commit 37868e5) blocked N=4 matmul
    // because %r9 was both the j-counter scratch and output_reg() at N=4.
    // M13 relocates the j-counter to %rbp (callee-saved by the function-level
    // prologue; no op-emitter reads it inside the body). emit_matmul must now
    // accept N=4 and emit asm using %rbp as the j-counter.
    use compiler::ast::Span;
    let abi = AbiContext { n_inputs: 4 };
    let span = Span::new(1, 1);
    let result = crate::ops::matmul::emit_matmul(
        &abi,
        /* leading_count */ 1,
        /* m */ 4,
        /* k */ 8,
        /* n */ 4,
        /* transpose_b */ false,
        /* model_idx */ 0,
        /* matmul_idx */ 0,
        /* a_loc */ BufferLoc::InputReg(0),
        /* b_loc */ BufferLoc::InputReg(1),
        /* dst_loc */ BufferLoc::OutputReg,
        span,
    );
    let asm = result.expect("emit_matmul must accept N=4 after M13 fix");
    // The j-counter init now writes to %rbp, not %r9.
    assert!(
        asm.contains("movq    $0, %rbp\n"),
        "expected j-counter init `movq $0, %rbp`; got:\n{asm}"
    );
    // Old %r9 j-counter init must be gone.
    assert!(
        !asm.contains("movq    $0, %r9\n"),
        "stale %r9 j-counter init must be removed; got:\n{asm}"
    );
    // %r9 is output_reg at N=4; it must NOT be written to by the matmul body.
    assert!(
        !asm.contains(", %r9\n"),
        "matmul body must not write to %r9 (output_reg at N=4); got:\n{asm}"
    );
}

// ---- Group C Q5: compute_callee_saved matmul-only branch ------------------

#[test]
fn compute_callee_saved_fires_for_matmul_only_no_softmax_model() {
    // Q5 (Group C): the has_matmul branch of compute_callee_saved was only
    // exercised end-to-end through self_attention.nfl which has BOTH softmax
    // AND matmul. This test verifies the has_matmul trigger fires independently
    // for a matmul-only model with no softmax.
    let src = "\
model MatmulOnly [batch=2]:
    a: Tensor[batch, 4]
    b: Tensor[4, 8]

    out: Tensor[batch, 8] = a -> matmul[b]
";
    let ast = compiler::parse(src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];
    // Precondition: model has no softmax node.
    let has_softmax = model.nodes.iter().any(|n| {
        matches!(
            n.kind,
            compiler::NodeKind::Op {
                op: compiler::StdOp::Softmax,
                ..
            }
        )
    });
    assert!(
        !has_softmax,
        "fixture must not contain softmax for this test to be meaningful"
    );
    let regs = compute_callee_saved(model);
    assert!(
        regs.callee_saved_int,
        "matmul-only model must trigger callee-saved register save (has_matmul branch)"
    );
}

// ─── M13 PR follow-up: ABI-clean invariant for x86_64 simple-loop emitters ──
//
// PR #28 closed a systemic class of bug: emit_relu/dropout/mulscalar/linear
// all used %rcx as a loop counter, which is the output_reg at N=2 (and shifts
// to other ABI roles at N=3, N=4). The bug was invisible until residual_add
// (N=2 with linear+relu+add) surfaced it via SIGSEGV on Linux x86_64.
//
// To prevent recurrence, every simple-loop emitter gets a parametric "no ABI
// clobber" test for each n_inputs ∈ {1, 2, 3, 4}. The test calls the emitter,
// then asserts no ABI argument register (from `abi.ffi_save_set()` =
// INPUT_REGS[..n_inputs+2] = inputs ∪ {params, output}) appears as a write
// destination in the emitted asm. The pattern `, %<reg>\n` matches AT&T
// destination operands (`movq <src>, <dst>` and `xorq <reg>, <reg>` forms).
//
// Single-operand instructions (`incq %reg`, `pushq %reg`) are not pattern-
// matched, but they only matter as a source-of-clobber if a prior `movq` or
// `xorq` already wrote to the register — which the pattern catches. So the
// `, %<reg>\n` check is a sufficient first-write detector.

#[cfg(test)]
fn assert_emit_abi_clean(emitter: &str, asm: &str, abi: &crate::abi::AbiContext) {
    for reg in abi.ffi_save_set() {
        assert!(
            !asm.contains(&format!(", {reg}\n")),
            "{emitter} at N={n} writes to ABI register {reg}; got:\n{asm}",
            n = abi.n_inputs,
        );
    }
}

// ── emit_relu ───────────────────────────────────────────────────────────────

#[cfg(test)]
fn emit_relu_at(n_inputs: usize) -> String {
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs };
    crate::ops::emit_relu(
        &abi,
        /* total_floats */ 8,
        /* model_idx */ 0,
        /* relu_idx */ 0,
        /* src_loc */ BufferLoc::StackOffset(0),
        /* dst_loc */ BufferLoc::OutputReg,
    )
}

#[test]
fn emit_relu_abi_clean_at_n1() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    assert_emit_abi_clean("emit_relu", &emit_relu_at(1), &abi);
}
#[test]
fn emit_relu_abi_clean_at_n2() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    assert_emit_abi_clean("emit_relu", &emit_relu_at(2), &abi);
}
#[test]
fn emit_relu_abi_clean_at_n3() {
    let abi = crate::abi::AbiContext { n_inputs: 3 };
    assert_emit_abi_clean("emit_relu", &emit_relu_at(3), &abi);
}
#[test]
fn emit_relu_abi_clean_at_n4() {
    let abi = crate::abi::AbiContext { n_inputs: 4 };
    assert_emit_abi_clean("emit_relu", &emit_relu_at(4), &abi);
}

// ── emit_dropout_copy ───────────────────────────────────────────────────────

#[cfg(test)]
fn emit_dropout_copy_at(n_inputs: usize) -> String {
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs };
    crate::ops::dropout::emit_dropout_copy(
        &abi,
        /* total_floats */ 8,
        /* model_idx */ 0,
        /* dropout_idx */ 0,
        /* src_loc */ BufferLoc::StackOffset(0),
        /* dst_loc */ BufferLoc::OutputReg,
    )
}

#[test]
fn emit_dropout_copy_abi_clean_at_n1() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    assert_emit_abi_clean("emit_dropout_copy", &emit_dropout_copy_at(1), &abi);
}
#[test]
fn emit_dropout_copy_abi_clean_at_n2() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    assert_emit_abi_clean("emit_dropout_copy", &emit_dropout_copy_at(2), &abi);
}
#[test]
fn emit_dropout_copy_abi_clean_at_n3() {
    let abi = crate::abi::AbiContext { n_inputs: 3 };
    assert_emit_abi_clean("emit_dropout_copy", &emit_dropout_copy_at(3), &abi);
}
#[test]
fn emit_dropout_copy_abi_clean_at_n4() {
    let abi = crate::abi::AbiContext { n_inputs: 4 };
    assert_emit_abi_clean("emit_dropout_copy", &emit_dropout_copy_at(4), &abi);
}

// ── emit_mulscalar ──────────────────────────────────────────────────────────

#[cfg(test)]
fn emit_mulscalar_at(n_inputs: usize) -> String {
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs };
    crate::ops::emit_mulscalar(
        &abi,
        /* total_elements */ 8,
        /* scalar_bits */ 0,
        /* model_idx */ 0,
        /* op_idx */ 0,
        /* src_loc */ BufferLoc::StackOffset(0),
        /* dst_loc */ BufferLoc::OutputReg,
    )
}

#[test]
fn emit_mulscalar_abi_clean_at_n1() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    assert_emit_abi_clean("emit_mulscalar", &emit_mulscalar_at(1), &abi);
}
#[test]
fn emit_mulscalar_abi_clean_at_n2() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    assert_emit_abi_clean("emit_mulscalar", &emit_mulscalar_at(2), &abi);
}
#[test]
fn emit_mulscalar_abi_clean_at_n3() {
    let abi = crate::abi::AbiContext { n_inputs: 3 };
    assert_emit_abi_clean("emit_mulscalar", &emit_mulscalar_at(3), &abi);
}
#[test]
fn emit_mulscalar_abi_clean_at_n4() {
    let abi = crate::abi::AbiContext { n_inputs: 4 };
    assert_emit_abi_clean("emit_mulscalar", &emit_mulscalar_at(4), &abi);
}

// ── emit_add ────────────────────────────────────────────────────────────────

#[cfg(test)]
fn emit_add_at(n_inputs: usize) -> String {
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs };
    crate::ops::emit_add(
        &abi,
        /* total_elements */ 8,
        /* model_idx */ 0,
        /* op_idx */ 0,
        /* a_loc */ BufferLoc::StackOffset(0),
        /* other_loc */ BufferLoc::InputReg(0),
        /* dst_loc */ BufferLoc::OutputReg,
    )
}

#[test]
fn emit_add_abi_clean_at_n1() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    assert_emit_abi_clean("emit_add", &emit_add_at(1), &abi);
}
#[test]
fn emit_add_abi_clean_at_n2() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    assert_emit_abi_clean("emit_add", &emit_add_at(2), &abi);
}
#[test]
fn emit_add_abi_clean_at_n3() {
    let abi = crate::abi::AbiContext { n_inputs: 3 };
    assert_emit_abi_clean("emit_add", &emit_add_at(3), &abi);
}
#[test]
fn emit_add_abi_clean_at_n4() {
    let abi = crate::abi::AbiContext { n_inputs: 4 };
    assert_emit_abi_clean("emit_add", &emit_add_at(4), &abi);
}

// ─── M14 Plan 1: LH-1/2/3 ABI-invariant regression guards for emit_linear ───
//
// Three latent hazards existed in emit_linear at N=2/3/4 where scratch
// registers aliased ABI argument registers (output_reg at those N values).
// These tests guard against regression after the M14 LH-1/2/3 cleanup.
//
// Test structure: call emit_linear directly at the triggering N, extract
// the matmul body (between .Lmm_i_ and .Lmm_i_end_ labels), assert
// post-fix invariants on that body.

#[cfg(test)]
fn emit_linear_at(n_inputs: usize, bias: bool) -> String {
    use crate::abi::AbiContext;
    use compiler::PostOp;
    let abi = AbiContext { n_inputs };
    let post: Vec<PostOp> = vec![];
    crate::ops::emit_linear(
        &abi,
        /* b */ 2,
        /* k */ 2,
        /* n */ 2,
        /* model_idx */ 0,
        /* linear_idx */ 0,
        /* src_loc */ BufferLoc::InputReg(0),
        /* dst_loc */ BufferLoc::OutputReg,
        /* weight_offset */ 0,
        /* bias_offset */ if bias { Some(0) } else { None },
        /* node_span */ compiler::ast::Span::new(1, 1),
        /* fused_post_ops */ &post,
        /* sym_prefix */ "",
    )
    .expect("emit_linear must succeed")
}

/// Extract the matmul body: text between the `.Lmm_i_` loop label and
/// the `.Lmm_i_end_` label (exclusive). This is the region where
/// j-counter and scratch pointer conflicts manifest.
#[cfg(test)]
fn extract_linear_matmul_body(asm: &str) -> &str {
    // The outer i-loop starts at ".Lmm_i_0_0:\n" and ends at ".Lmm_i_end_0_0:\n".
    let start_label = ".Lmm_i_0_0:";
    let end_label = ".Lmm_i_end_0_0:";
    let start = asm
        .find(start_label)
        .expect("matmul body start label not found")
        + start_label.len();
    let end = asm
        .find(end_label)
        .expect("matmul body end label not found");
    &asm[start..end]
}

#[test]
fn emit_linear_n2_with_bias_does_not_alias_output_reg_in_body() {
    // LH-1 regression guard.
    //
    // Pre-fix: at N=2, output_reg = %rcx (INPUT_REGS[3]). The j-counter
    // also lived in %rcx, so the bias-add expanded to
    // `movss (%rcx, %rcx, 4), %xmm5` — base aliased offset, wrong output.
    //
    // Post-fix: j-counter relocated to %rbp. Bias-add expands to
    // `movss (%rcx, %rbp, 4), %xmm5` — base = bias_base in %rcx, offset
    // = j-counter in %rbp. Correct.
    //
    // Pattern: lower a minimal N=2 + linear-with-bias UIR, extract the
    // matmul body (between `.Lmm_i_` and `.Lmm_i_end_` labels), assert
    // post-fix invariants on the body.

    let asm = emit_linear_at(2, true);
    let body = extract_linear_matmul_body(&asm);

    // Pre-fix marker — must NOT appear:
    assert!(
        !body.contains("xorq    %rcx, %rcx"),
        "j-counter init must not be %rcx (would alias output_reg at N=2). Body:\n{body}"
    );
    assert!(
        !body.contains("(%rcx, %rcx,"),
        "bias-add base must not alias offset (LH-1 silent corruption pattern). Body:\n{body}"
    );

    // Post-fix marker — must appear:
    assert!(
        body.contains("xorq    %rbp, %rbp"),
        "j-counter should be relocated to %rbp (M13 Task 1 precedent). Body:\n{body}"
    );
    assert!(
        body.contains("(%rcx, %rbp, 4), %xmm5"),
        "bias-add should read from (output_reg=%rcx, j=%rbp). Body:\n{body}"
    );
}

#[test]
fn emit_linear_n3_does_not_clobber_output_reg() {
    // LH-2 regression guard.
    //
    // Pre-fix: at N=3, output_reg = %r8 (INPUT_REGS[4]). emit_linear's
    // src_ptr materialise wrote to %r8 (via materialise_ptr call), destroying the FFI
    // output_reg. Subsequent ops in the same function would see a
    // garbage output pointer.
    //
    // Post-fix: src ptr scratch relocated to %r14 (callee-saved per SysV;
    // op-local pushq/popq inside emit_linear body — function-level
    // prologue unchanged).

    let asm = emit_linear_at(3, false);
    let body = extract_linear_matmul_body(&asm);

    // Pre-fix marker — must NOT appear. At N=3, output_reg = %r8;
    // `materialise_ptr(src_loc, "%r8", ...)` would emit a `movq ..., %r8`
    // (write to %r8) which is the LH-2 silent corruption. Also check the
    // body uses no %r8 base in indexed loads.
    assert!(
        !body.contains(", %r8\n") && !body.contains("(%r8,"),
        "src ptr scratch must not use %r8 at N=3 (output_reg alias). Body:\n{body}"
    );
    // Post-fix expectation: src loads use (%r14, ...) indexed addressing.
    assert!(
        body.contains("(%r14,"),
        "src ptr should be relocated to %r14 (indexed load). Body:\n{body}"
    );

    // Op-local save/restore must appear in full asm:
    assert!(
        asm.contains("    pushq   %r14\n"),
        "op-local pushq %r14 must appear (callee-saved op-local save). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r14\n"),
        "op-local popq %r14 must appear (matching restore). Asm:\n{asm}"
    );
    // Full asm must show %r14 init (materialise src ptr, before the loop body).
    assert!(
        asm.contains(", %r14\n"),
        "src ptr materialise must write to %r14 (full asm). Asm:\n{asm}"
    );
}

#[test]
fn emit_linear_n4_does_not_clobber_output_reg() {
    // LH-3 regression guard.
    //
    // Pre-fix: at N=4, output_reg = %r9 (INPUT_REGS[5]). emit_linear's
    // weight base setup (via movq/leaq into %r9) wrote to %r9, destroying the FFI
    // output_reg.
    //
    // Post-fix: weight ptr scratch relocated to %r15 (callee-saved per
    // SysV; op-local pushq/popq inside emit_linear body — function-level
    // prologue unchanged).

    let asm = emit_linear_at(4, false);
    let body = extract_linear_matmul_body(&asm);

    // Pre-fix marker — must NOT appear. At N=4, output_reg = %r9; the
    // weight base setup (movq params_reg, %r9 OR leaq weight_offset(params_reg), %r9)
    // would clobber output_reg.
    assert!(
        !body.contains(", %r9\n"),
        "weight ptr scratch must not write to %r9 at N=4 (output_reg alias). Body:\n{body}"
    );
    // Post-fix expectation: weight ptr lives in %r15.
    assert!(
        body.contains("(%r15, ") || body.contains(", %r15\n"),
        "weight ptr should be relocated to %r15. Body:\n{body}"
    );

    // Op-local save/restore:
    assert!(
        asm.contains("    pushq   %r15\n") && asm.contains("    popq    %r15\n"),
        "op-local pushq/popq %r15 must bracket the body. Asm:\n{asm}"
    );
}

// ── emit_layernorm ──────────────────────────────────────────────────────────

#[cfg(test)]
fn emit_layernorm_x86_64_at(n_inputs: usize, has_affine: bool) -> String {
    use crate::abi::AbiContext;
    use compiler::ast::Span;
    let abi = AbiContext { n_inputs };
    let (gamma_offset, beta_offset) = if has_affine {
        (Some(0usize), Some(32usize))
    } else {
        (None, None)
    };
    crate::ops::emit_layernorm(
        &abi,
        /* b */ 8,
        /* d */ 32,
        /* model_idx */ 0,
        /* layernorm_idx */ 0,
        /* src_loc */ BufferLoc::InputReg(0),
        /* dst_loc */ BufferLoc::OutputReg,
        gamma_offset,
        beta_offset,
        Span::new(1, 1),
    )
    .expect("emit succeeds")
}

#[test]
fn emit_layernorm_x86_64_abi_clean_at_n1_no_affine() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    let asm = emit_layernorm_x86_64_at(1, false);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

#[test]
fn emit_layernorm_x86_64_abi_clean_at_n1_with_affine() {
    let abi = crate::abi::AbiContext { n_inputs: 1 };
    let asm = emit_layernorm_x86_64_at(1, true);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

#[test]
fn emit_layernorm_x86_64_abi_clean_at_n2_no_affine() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    let asm = emit_layernorm_x86_64_at(2, false);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

#[test]
fn emit_layernorm_x86_64_abi_clean_at_n2_with_affine() {
    let abi = crate::abi::AbiContext { n_inputs: 2 };
    let asm = emit_layernorm_x86_64_at(2, true);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

// ---- M14 Task 3: emit_layernorm x86_64 unit tests ----------------------------

#[test]
fn emit_layernorm_x86_64_no_affine_emits_three_passes_with_native_sqrtss() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    use compiler::ast::Span;

    let abi = AbiContext { n_inputs: 1 };
    let asm = crate::ops::emit_layernorm(
        &abi,
        8,
        32,
        0,
        0,
        BufferLoc::InputReg(0),
        BufferLoc::OutputReg,
        None,
        None,
        Span::new(1, 1),
    )
    .expect("no-affine emit should succeed");

    // Three pass labels.
    assert!(asm.matches(".Lln_p1_").count() >= 2);
    assert!(asm.matches(".Lln_p2_").count() >= 2);
    assert!(asm.matches(".Lln_p3_").count() >= 2);

    // Native sqrtss — no `call sqrtf@PLT`.
    assert!(asm.contains("sqrtss"));
    assert!(!asm.contains("sqrtf@PLT"));
    assert!(
        !asm.contains("    call    "),
        "leaf function — no `call` instruction"
    );

    // divss for inv_std reciprocal — exactly one, OUTSIDE Pass 3.
    // Use "\n.Lln_p3_end_" to find the actual end LABEL, not the `jge`
    // branch-target substring that appears inside the loop body earlier.
    let p3_label = asm.find(".Lln_p3_").expect("Pass 3 label");
    let p3_end = asm
        .find("\n.Lln_p3_end_")
        .expect("Pass 3 end label (newline-prefixed to skip branch target)");
    let p3_body = &asm[p3_label..p3_end];
    assert!(
        !p3_body.contains("divss"),
        "Pass 3 hot loop must contain ZERO divss (Q4 constraint); body:\n{p3_body}"
    );

    // Always-present op-local saves for src/dst base ptrs (callee-saved per SysV).
    assert!(
        asm.contains("pushq   %rbx") && asm.contains("popq    %rbx"),
        "no-affine path must still push %rbx for src base ptr scratch"
    );
    assert!(
        asm.contains("pushq   %r14") && asm.contains("popq    %r14"),
        "no-affine path must still push %r14 for dst base ptr scratch"
    );
    // Affine-only saves must NOT appear in no-affine path:
    assert!(
        !asm.contains("pushq   %r12"),
        "no-affine must not push %r12 (γ ptr)"
    );
    assert!(
        !asm.contains("pushq   %r13"),
        "no-affine must not push %r13 (β ptr)"
    );
}

#[test]
fn emit_layernorm_x86_64_affine_emits_gamma_beta_loads_and_callee_saved_pushes() {
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    use compiler::ast::Span;

    let abi = AbiContext { n_inputs: 1 };
    let asm = crate::ops::emit_layernorm(
        &abi,
        /* b = */ 8,
        /* d = */ 32,
        /* model_idx = */ 0,
        /* layernorm_idx = */ 0,
        BufferLoc::InputReg(0),
        BufferLoc::OutputReg,
        /* gamma_offset = */ Some(0),
        /* beta_offset = */ Some(32), // β follows γ at offset 32 floats
        Span::new(1, 1),
    )
    .expect("affine emit should succeed");

    // Op-local callee-saved push/pop for affine path — must bracket the body.
    assert!(
        asm.contains("    pushq   %r12\n"),
        "affine path must push %r12 (op-local save of γ base ptr). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    pushq   %r13\n"),
        "affine path must push %r13 (op-local save of β base ptr). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r12\n"),
        "affine path must pop %r12 (matching restore). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r13\n"),
        "affine path must pop %r13 (matching restore). Asm:\n{asm}"
    );

    // Pass 3 must contain γ and β loads from %r12 and %r13.
    // Use "\n.Lln_p3_end_" to find the actual end LABEL, not the `jge`
    // branch-target substring that appears inside the loop body earlier.
    let p3_label = asm.find(".Lln_p3_").expect("Pass 3 label");
    let p3_end = asm
        .find("\n.Lln_p3_end_")
        .expect("Pass 3 end label (newline-prefixed to skip branch target)");
    let p3_body = &asm[p3_label..p3_end];

    assert!(
        p3_body.contains("movss   (%r12, "),
        "Pass 3 must load γ_j from %r12 base; body:\n{p3_body}"
    );
    assert!(
        p3_body.contains("movss   (%r13, "),
        "Pass 3 must load β_j from %r13 base; body:\n{p3_body}"
    );
    assert!(
        p3_body.contains("mulss"),
        "Pass 3 affine must include γ multiply (mulss)"
    );
    assert!(
        p3_body.contains("addss"),
        "Pass 3 affine must include β add (addss)"
    );

    // Q4 constraint persists in affine path — Pass 3 hot loop has zero divss.
    assert!(
        !p3_body.contains("divss"),
        "Pass 3 hot loop must contain ZERO divss (Q4 constraint). Body:\n{p3_body}"
    );
}

#[test]
fn emit_layernorm_x86_64_does_not_extend_function_level_callee_saved_set() {
    // The op-local %r12/%r13 push/pop lives entirely inside emit_layernorm's
    // emitted asm. compute_callee_saved (in buffer.rs) must NOT report
    // callee_saved_int == true for a LayerNorm-only function — the function-
    // level prologue should NOT push the 5-register callee-saved block.
    // (M13 pre-Task-5 arm64 emit_linear stp/ldp precedent applied to x86_64.)
    //
    // Two-pronged check:
    //   1. compute_callee_saved returns callee_saved_int=false (analyzer level).
    //   2. The full lowered asm contains the op-local pushq %r12/%r13 (body
    //      level), but the function prologue block (up to the first pushq from
    //      emit_layernorm) is only "pushq %rbp; movq %rsp, %rbp" — no 5-reg block.
    let nfl_src = "model M [b=2, d=4]:\n    x: Tensor[b, d]\n    x -> layernorm[affine=true]\n";
    let ast = compiler::parse(nfl_src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let model = &uir.models[0];

    // Prong 1: analyzer must not trigger callee-saved-int promotion.
    let regs = compute_callee_saved(model);
    assert!(
        !regs.contains_callee_saved_int(),
        "compute_callee_saved must return callee_saved_int=false for LayerNorm-only model; \
         function-level prologue should not push %rbx/%r12-%r15"
    );

    // Prong 2: full asm contains op-local pushq %r12/%r13 from emit_layernorm body.
    let asm = crate::X86_64Profile.lower(&uir).expect("lower").source;
    assert!(
        asm.contains("    pushq   %r12\n"),
        "Op-local pushq %r12 must appear in full asm (from emit_layernorm body). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    pushq   %r13\n"),
        "Op-local pushq %r13 must appear in full asm (from emit_layernorm body). Asm:\n{asm}"
    );
    // The function-level prologue (per asm.rs::format_function_prologue with
    // callee_saved_int=false) emits only: pushq %rbp + movq %rsp, %rbp.
    // Verify the 5-register block (%rbx, %r12, %r13, %r14, %r15) is absent
    // from the top of the asm (before the first emit_layernorm label).
    let body_start = asm
        .find(".Lln_row_")
        .expect("outer row loop label must exist");
    // The function header is everything before the outer row loop. The op-local
    // pushes appear between the function entry and .Lln_row_ — that is expected
    // and correct. What must NOT appear: "pushq   %rbx\n    pushq   %r12\n" as
    // the function-level 5-register block (pushq %rbx is the first push in the
    // function-level callee-saved block per asm.rs:70; op-local block starts
    // with pushq %r15 instead). Post-M15 the op-local block DOES push %r15, so
    // checking for %r15 alone would false-positive — check the function-level
    // sentinel pattern instead.
    let header = &asm[..body_start];
    assert!(
        !header.contains("    pushq   %rbx\n    pushq   %r12\n"),
        "Function-level prologue must NOT push the 5-register callee-saved block \
         (%rbx/%r12-%r15 sequence — signals compute_callee_saved returned true). \
         Header:\n{header}"
    );
}

#[test]
fn emit_layernorm_x86_64_affine_allocates_scale_before_bias_in_params_layout() {
    // ParamSlot order contract: γ (LayerNormScale) must appear before β
    // (LayerNormBias) in params_layout. Mirrors arm64's ParamSlot order test.
    use crate::ParamKind;
    let nfl_src = "model M [b=2, d=4]:\n    x: Tensor[b, d]\n    x -> layernorm[affine=true]\n";
    let ast = compiler::parse(nfl_src).expect("parse should succeed");
    let uir = compiler::ir::build(&ast).expect("ir::build should succeed");

    let asm = crate::X86_64Profile
        .lower(&uir)
        .expect("lower should succeed");

    let sig = &asm.functions[0];
    let scale_idx = sig
        .params_layout
        .iter()
        .position(|s| s.kind == ParamKind::LayerNormScale)
        .expect("LayerNormScale must be in params_layout when affine=true");
    let bias_idx = sig
        .params_layout
        .iter()
        .position(|s| s.kind == ParamKind::LayerNormBias)
        .expect("LayerNormBias must be in params_layout when affine=true");

    assert!(
        scale_idx < bias_idx,
        "Contract violation: γ (LayerNormScale at idx {scale_idx}) must come before \
         β (LayerNormBias at idx {bias_idx}) in params_layout. Layout: {:?}",
        sig.params_layout
    );
}

// ---- M15 LH-4 cleanup tests: emit_layernorm at N=2/3/4 -----------------------
//
// Mirrors the emit_linear_n{2,3,4}_does_not_clobber_output_reg pattern
// (M14 commit 916e9c7). Validates LH-4 closure: %r8 and %r9 must NOT
// appear as scratch destinations in the per-row body at N=3 (output_reg=%r8)
// or N=4 (output_reg=%r9). Post-fix expectation: per-row src ptr lives in
// %r15 (op-local pushq/popq); per-row dst ptr lives in %rbp (function-level
// prologue handles).

#[test]
fn emit_layernorm_n2_does_not_clobber_output_reg() {
    // Parametric guard. At N=2, output_reg=%rcx, never aliased by per-row
    // ptrs in any era (passes pre- and post-fix). Kept for coverage parity
    // across the supported N range, mirroring emit_linear pattern.
    use crate::abi::AbiContext;
    let abi = AbiContext { n_inputs: 2 };
    let asm = emit_layernorm_x86_64_at(2, false);
    assert_emit_abi_clean("emit_layernorm", &asm, &abi);
}

#[test]
fn emit_layernorm_n3_does_not_clobber_output_reg() {
    // Primary LH-4 unit test. At N=3, output_reg = %r8 (INPUT_REGS[4]).
    //
    // Pre-fix: emit_layernorm wrote per-row src ptr to %r8 inside the outer
    // row loop:
    //     leaq    (%rbx, %rax, 1), %r8       ← clobbers output_reg
    // and used it as src base in the three Pass loops:
    //     movss   (%r8, %rax, 4), %xmm6
    // Subsequent ops in the same function would see a corrupted output_reg.
    //
    // Post-fix: src ptr scratch relocated to %r15 (callee-saved per SysV;
    // op-local pushq %r15 / popq %r15 inside emit_layernorm body — function-
    // level prologue unchanged). Dst ptr scratch relocated to %rbp (function-
    // level prologue handles).

    let asm = emit_layernorm_x86_64_at(3, false);

    // Pre-fix marker — must NOT appear:
    assert!(
        !asm.contains(", %r8\n"),
        "src ptr scratch must not write to %r8 at N=3 (output_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r8,"),
        "src ptr scratch must not be used as base in indexed load at N=3. Asm:\n{asm}"
    );
    assert!(
        !asm.contains(", %r9\n"),
        "dst ptr scratch must not write to %r9 at N=3 (params_reg alias for next op). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r9,"),
        "dst ptr scratch must not be used as base in indexed store at N=3. Asm:\n{asm}"
    );

    // Post-fix expectations: src ptr in %r15 (op-local push/pop), dst ptr in %rbp.
    assert!(
        asm.contains("    pushq   %r15\n"),
        "op-local pushq %r15 must appear (callee-saved op-local save for src ptr). Asm:\n{asm}"
    );
    assert!(
        asm.contains("    popq    %r15\n"),
        "op-local popq %r15 must appear (matching restore). Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%r15,"),
        "src ptr should be %r15 (indexed load base). Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%rbp,"),
        "dst ptr should be %rbp (indexed store base). Asm:\n{asm}"
    );

    // Push count check: no-affine path = 3 op-local pushes (%r15, %rbx, %r14).
    let pushq_count = asm.matches("    pushq   ").count();
    let popq_count = asm.matches("    popq    ").count();
    assert!(
        pushq_count >= 3,
        "expected at least 3 op-local pushq in no-affine layernorm body, got {pushq_count}. Asm:\n{asm}"
    );
    assert_eq!(
        pushq_count, popq_count,
        "push/pop count mismatch — LIFO discipline broken. Asm:\n{asm}"
    );
}

#[test]
fn emit_layernorm_n4_does_not_clobber_output_reg() {
    // Secondary LH-4 unit test. At N=4, output_reg = %r9 (INPUT_REGS[5])
    // AND params_reg = %r8 (INPUT_REGS[4]). Both registers are ABI-occupied;
    // pre-fix, layernorm clobbered both. Post-fix, neither appears as
    // scratch destination in body.
    //
    // No N=4 runtime fixture in M15 (transformer_block.nfl is N=3).
    // Asm-shape closure follows M14 LH-2/3 precedent for emit_linear N=4
    // (four_input_matmul.nfl has no linear op, so emit_linear N=4 closure
    // was also asm-only).

    let asm = emit_layernorm_x86_64_at(4, true); // affine: 5 op-local pushes

    // Pre-fix markers — must NOT appear:
    assert!(
        !asm.contains(", %r8\n"),
        "no scratch may write to %r8 at N=4 (params_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r8,"),
        "no scratch may use %r8 as base at N=4. Asm:\n{asm}"
    );
    assert!(
        !asm.contains(", %r9\n"),
        "no scratch may write to %r9 at N=4 (output_reg alias). Asm:\n{asm}"
    );
    assert!(
        !asm.contains("(%r9,"),
        "no scratch may use %r9 as base at N=4. Asm:\n{asm}"
    );

    // Post-fix: %r15 + %rbp scratch present in affine path.
    assert!(
        asm.contains("    pushq   %r15\n") && asm.contains("    popq    %r15\n"),
        "op-local pushq/popq %r15 must bracket affine body. Asm:\n{asm}"
    );
    assert!(
        asm.contains("(%r15,") && asm.contains("(%rbp,"),
        "src/dst ptrs should be %r15/%rbp. Asm:\n{asm}"
    );

    // Affine path = 5 op-local pushes (%r15, %r12, %r13, %rbx, %r14).
    let pushq_count = asm.matches("    pushq   ").count();
    let popq_count = asm.matches("    popq    ").count();
    assert!(
        pushq_count >= 5,
        "expected at least 5 op-local pushq in affine layernorm body, got {pushq_count}. Asm:\n{asm}"
    );
    assert_eq!(pushq_count, popq_count, "push/pop LIFO. Asm:\n{asm}");
}

// ── M16 (A3): Profile::inspect() unit tests ─────────────────────────────────

fn build_uir(src: &str) -> compiler::Uir {
    let ast = compiler::parse(src).expect("parse");
    compiler::ir::build(&ast).expect("ir::build")
}

#[test]
fn inspect_softmax_model_is_non_leaf() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 4]\n    x -> softmax\n");
    let insp = crate::X86_64Profile.inspect(&uir).expect("inspect");
    assert_eq!(insp.functions.len(), 1);
    assert!(
        !insp.functions[0].leaf,
        "softmax-bearing model must report leaf=false (calls expf@PLT)"
    );
}

#[test]
fn inspect_pure_linear_model_is_leaf() {
    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n");
    let insp = crate::X86_64Profile.inspect(&uir).expect("inspect");
    assert!(insp.functions[0].leaf, "no extern math = leaf");
}

#[test]
fn inspect_pre_pass_dropout_uses_alias_placement() {
    use compiler::{NodeKind, StdOp};
    use profile_api::BufferLoc;

    // Pre-pass: dropout is the canonical alias-bearing op. The model
    // below has dropout in the middle (NOT as the output), so
    // assign_buffers takes the Alias branch.
    let uir =
        build_uir("model M [b=2]:\n    x: Tensor[b, 3]\n    x -> dropout[rate=0.1] -> linear[4]\n");
    let insp = crate::X86_64Profile.inspect(&uir).expect("inspect");
    let dropout_idx = uir.models[0]
        .nodes
        .iter()
        .position(|n| {
            matches!(
                &n.kind,
                NodeKind::Op {
                    op: StdOp::Dropout,
                    ..
                }
            )
        })
        .expect("pre-pass UIR must contain a Dropout node");
    let dropout_input_idx = match &uir.models[0].nodes[dropout_idx].kind {
        NodeKind::Op { operands, .. } => operands[0],
        _ => unreachable!(),
    };
    assert_eq!(
        insp.functions[0].nodes[dropout_idx].buffer_loc,
        BufferLoc::Alias(dropout_input_idx),
        "dropout (not output) must alias its operand"
    );
}

#[test]
fn inspect_linear_with_bias_reports_correct_params() {
    use compiler::{NodeKind, StdOp};

    let uir = build_uir("model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[8, bias=true]\n");
    let insp = crate::X86_64Profile.inspect(&uir).expect("inspect");
    let f = &insp.functions[0];

    let linear_idx = uir.models[0]
        .nodes
        .iter()
        .position(|n| {
            matches!(
                &n.kind,
                NodeKind::Op {
                    op: StdOp::Linear,
                    ..
                }
            )
        })
        .unwrap();

    // K=4, N=8, bias=true → 4*8 + 8 = 40 floats
    assert_eq!(f.nodes[linear_idx].params_floats, Some(40));
}

// ── M17 Task 7: x86_64 .rodata exp constant pool ─────────────────────────────

#[test]
fn softmax_model_emits_local_exp_pool() {
    let src = "model S [batch=2, k=3]:\n    x: Tensor[batch, k]\n    x -> softmax\n";
    let uir = compiler::ir::build(&compiler::parse(src).unwrap()).unwrap();
    let asm = crate::lower(&uir).unwrap().source;
    assert!(asm.contains(".section .rodata"), "no rodata pool:\n{asm}");
    assert!(asm.contains(".Lexp_log2e:"), "no log2e constant:\n{asm}");
    assert!(asm.contains(".Lexp_c7:"), "no c7 constant:\n{asm}");
    assert_eq!(
        asm.matches(".Lexp_log2e:").count(),
        1,
        "pool must be unique per file"
    );
}
