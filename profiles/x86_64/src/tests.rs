// SPDX-License-Identifier: Apache-2.0

use crate::asm::{compute_frame_size, emit_imm32_to_r10, materialise_ptr};
use crate::buffer::{assign_buffers, compute_callee_saved, BufferLoc};
use crate::LowerError;
use compiler::ir;
use compiler::passes;

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
fn materialise_ptr_input_reg() {
    let s = materialise_ptr("%rax", BufferLoc::InputReg);
    assert_eq!(s, "    movq    %rdi, %rax\n");
}

#[test]
fn materialise_ptr_output_reg() {
    let s = materialise_ptr("%rbx", BufferLoc::OutputReg);
    assert_eq!(s, "    movq    %rdx, %rbx\n");
}

#[test]
fn materialise_ptr_stack_offset_zero() {
    let s = materialise_ptr("%rax", BufferLoc::StackOffset(0));
    assert_eq!(s, "    movq    %rsp, %rax\n");
}

#[test]
fn materialise_ptr_stack_offset_nonzero() {
    let s = materialise_ptr("%rax", BufferLoc::StackOffset(16));
    assert_eq!(s, "    leaq    16(%rsp), %rax\n");
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
        s.contains("movss   (%r8, %rcx, 4), %xmm0"),
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
    assert_eq!(sig.input_floats, 6);
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
    assert!(matches!(assignment.locs[0], BufferLoc::InputReg));
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
    // Transpose flips inner b_offset computation:
    //   no-transpose: `movq    %r10, %r11` then `imulq   $N, %r11` (k_inner * N + j)
    //   transpose:    `movq    %rcx, %r11` then `imulq   $K, %r11` (j * K + k_inner)
    assert!(
        asm_no_t.contains("movq    %r10, %r11"),
        "no-t asm should compute b_offset from %r10 (k_inner):\n{}",
        asm_no_t
    );
    assert!(
        asm_t.contains("movq    %rcx, %r11"),
        "t asm should compute b_offset from %rcx (j):\n{}",
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
fn matmul_preserves_ffi_register_invariants_rdi_rsi_rdx() {
    // Critical regression test: emit_matmul x86_64 must not leak its scratch
    // use of %rdi/%rsi/%rdx as A_slice/B_slice/DST_slice pointers into the
    // FFI register state visible to downstream emitters (per SysV AMD64 ABI,
    // %rdi=input, %rsi=params, %rdx=output).
    //
    // The fix is to spill all three to %xmm6/%xmm7/%xmm8 at function entry
    // and restore at exit. This test asserts all three pairs are present.
    let src = "\
model M [batch=2, heads=4, seq=4, head_dim=4]:
    x: Tensor[batch, heads, seq, head_dim]

    out: Tensor[batch, heads, seq, seq] = x -> matmul[x, transpose_b=true]
";
    let asm = crate::lower(&compiler::ir::build(&compiler::parse(src).unwrap()).unwrap())
        .expect("lower")
        .source;

    // Entry spills.
    assert!(
        asm.contains("movq    %rdi, %xmm8"),
        "missing %rdi spill; asm:\n{}",
        asm
    );
    assert!(
        asm.contains("movq    %rsi, %xmm6"),
        "missing %rsi spill; asm:\n{}",
        asm
    );
    assert!(
        asm.contains("movq    %rdx, %xmm7"),
        "missing %rdx spill; asm:\n{}",
        asm
    );
    // Exit restores.
    assert!(
        asm.contains("movq    %xmm8, %rdi"),
        "missing %rdi restore; asm:\n{}",
        asm
    );
    assert!(
        asm.contains("movq    %xmm6, %rsi"),
        "missing %rsi restore; asm:\n{}",
        asm
    );
    assert!(
        asm.contains("movq    %xmm7, %rdx"),
        "missing %rdx restore; asm:\n{}",
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
