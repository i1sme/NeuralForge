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
    s.push_str(&materialise_ptr("%rbx", src_loc));
    s.push_str(&materialise_ptr("%r12", dst_loc));

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

    // Phase 1: row_max → (%rsp). Init xmm8 to -inf.
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
    s.push_str("    movss   %xmm8, (%rsp)\n");

    // Phase 2: exp(x - max) → dst, sum → 8(%rsp). Init sum to 0.
    s.push_str("    movl    $0, 8(%rsp)\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_exp_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    subss   (%rsp), %xmm0\n");
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax clobbered by call; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    movss   8(%rsp), %xmm1\n");
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str("    movss   %xmm1, 8(%rsp)\n");
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
    s.push_str("    divss   8(%rsp), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    s
}
