// SPDX-License-Identifier: Apache-2.0

//! Softmax (per-row stable, libm expf via PLT) codegen — x86_64 SSE2.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// Calls `<sym_prefix>expf@PLT` for each element. State across the call
/// lives in callee-saved int registers (%rbx, %r12, %r13, %r14, %r15)
/// and on the stack (`(%rsp)` row_max, `8(%rsp)` row_sum; reserved by
/// `assign_buffers` per spec §7.4); see the register contract in
/// profiles/x86_64/src/ops/linear.rs.
///
/// FFI register preservation (M10): `call expf@PLT` clobbers caller-saved
/// regs including %rdi (input), %rsi (params), %rdx (output) per SysV
/// AMD64. Any downstream emitter that reads any of these via
/// `materialise_ptr` would otherwise see garbage. Spec
/// §[register preservation]: any emitter that clobbers an FFI register
/// must save/restore. Self-attention exercises this: matmul → softmax →
/// matmul where the second matmul's `materialise_ptr("%r9", InputReg)`
/// emits `movq %rdi, %r9` after softmax has clobbered %rdi.
///
/// Strategy: at function entry, push %rdi/%rsi/%rdx (plus padding %rax
/// for 16-byte alignment) onto the stack — 32 bytes total. The body's
/// existing (%rsp) / 8(%rsp) row_max/row_sum slots shift to 32(%rsp) /
/// 40(%rsp). Restore at exit. xmm-spill is unavailable here because all
/// xmm regs are caller-saved on SysV AMD64 and `call expf@PLT` may
/// clobber any of them.
pub fn emit_softmax(
    b: u64,
    k: u64,
    model_idx: usize,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    sym_prefix: &str,
) -> String {
    let sid = format!("{model_idx}_{softmax_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"
    ));

    // Pin src/dst into callee-saved %rbx/%r12 (survives `call expf@PLT`).
    // MUST happen before the FFI-reg push below: `materialise_ptr` for any
    // `BufferLoc::StackOffset(off)` emits `leaq off(%rsp), %rbx` against
    // the current %rsp. Pushing first would shift %rsp and corrupt the
    // intermediate-buffer offsets baked in at assign_buffers time.
    s.push_str(&materialise_ptr("%rbx", src_loc));
    s.push_str(&materialise_ptr("%r12", dst_loc));

    // Spill FFI input regs %rdi (input ptr), %rsi (params ptr), %rdx
    // (output ptr) — survive `call expf@PLT`. Restored at function exit
    // before any downstream emitter reads them via materialise_ptr.
    // The 4th push is padding (16-byte alignment requirement on call
    // boundaries); %rax is just a convenient scratch reg to push. Total
    // 32 bytes pushed → softmax body's row_max/row_sum slots shift from
    // (%rsp)/8(%rsp) to 32(%rsp)/40(%rsp).
    s.push_str("    pushq   %rdi\n");
    s.push_str("    pushq   %rsi\n");
    s.push_str("    pushq   %rdx\n");
    s.push_str("    pushq   %rax\n"); // padding for 16-byte alignment

    // Outer per-row loop: %r13 = i.
    s.push_str("    xorq    %r13, %r13\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r13\n");
    s.push_str(&format!("    jge     .Lsm_i_end_{sid}\n"));

    // %r15 = i * k
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %r13, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");

    // Phase 1: row_max → 32(%rsp). Init xmm8 to -inf.
    // (Was (%rsp) before the FFI-reg push above shifted offsets by 32.)
    s.push_str("    movl    $0xFF800000, %r10d\n");
    s.push_str("    movd    %r10d, %xmm8\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_max_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    maxss   %xmm0, %xmm8\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));
    s.push_str("    movss   %xmm8, 32(%rsp)\n");

    // Phase 2: exp(x - max) → dst, sum → 40(%rsp). Init sum to 0.
    // (Was 8(%rsp) before the FFI-reg push above shifted offsets by 32.)
    s.push_str("    movl    $0, 40(%rsp)\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_exp_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    subss   32(%rsp), %xmm0\n");
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax clobbered by call; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    movss   40(%rsp), %xmm1\n");
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str("    movss   %xmm1, 40(%rsp)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Phase 3: normalise by row_sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_norm_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%r12, %rax, 4), %xmm0\n");
    s.push_str("    divss   40(%rsp), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    // Restore FFI input regs — must match the 4-push sequence above.
    // Order is reversed (LIFO): %rax (padding) first, then %rdx/%rsi/%rdi.
    s.push_str("    popq    %rax\n"); // discard padding
    s.push_str("    popq    %rdx\n");
    s.push_str("    popq    %rsi\n");
    s.push_str("    popq    %rdi\n");

    s
}
