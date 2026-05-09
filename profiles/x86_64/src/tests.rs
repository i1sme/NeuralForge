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
        s.contains("movss   (%rax, %rcx, 4), %xmm0"),
        "missing load:\n{s}"
    );
    assert!(
        s.contains("movss   %xmm0, (%r11, %rcx, 4)"),
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
    // when calls_extern_math; row_max sits at offset 0, row_sum at 8
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

// ---- M13 PR-fix: emit_linear x86_64 ABI register save at N≥2 -----------

#[test]
fn emit_linear_x86_64_save_block_balances_at_all_n() {
    // M13 (PR follow-up): x86_64 emit_linear's body clobbers %rdi/%rsi/%rcx
    // (k-counter, offset scratch, j-counter). At N=1 these are non-ABI; at
    // N≥2 they overlap with input(0)/input(1)/output_reg respectively
    // (and shift further at N=3, N=4). The fix: pushq save at body entry,
    // popq restore at body exit, conditional on n_inputs ≥ 2.
    //
    // This test verifies push/pop balance at every N ∈ [1, 4] and pins
    // the exact register set for N≥2:
    //   N=1 → 0 pushq/popq pairs (no save needed).
    //   N=2..4 → 3 pushq + 3 popq, in LIFO order (push %rdi, %rsi, %rcx;
    //            pop %rcx, %rsi, %rdi).
    use crate::abi::AbiContext;
    use crate::buffer::BufferLoc;
    use compiler::ast::Span;
    use compiler::PostOp;
    let cases = [(1usize, 0usize), (2, 3), (3, 3), (4, 3)];
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
        if n_inputs >= 2 {
            // Specific register set + LIFO ordering.
            for reg in &["%rdi", "%rsi", "%rcx"] {
                assert!(
                    asm.contains(&format!("    pushq   {reg}\n")),
                    "N={n_inputs}: expected `pushq {reg}`; got:\n{asm}"
                );
                assert!(
                    asm.contains(&format!("    popq    {reg}\n")),
                    "N={n_inputs}: expected `popq {reg}`; got:\n{asm}"
                );
            }
            // LIFO check: pop order must be %rcx, %rsi, %rdi (reverse of
            // push order %rdi, %rsi, %rcx). Verify by relative position.
            let pop_rcx = asm.find("    popq    %rcx\n").expect("popq %rcx");
            let pop_rsi = asm.find("    popq    %rsi\n").expect("popq %rsi");
            let pop_rdi = asm.find("    popq    %rdi\n").expect("popq %rdi");
            assert!(
                pop_rcx < pop_rsi && pop_rsi < pop_rdi,
                "N={n_inputs}: popq order must be LIFO (%rcx < %rsi < %rdi); got {pop_rcx}/{pop_rsi}/{pop_rdi}\n{asm}"
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
