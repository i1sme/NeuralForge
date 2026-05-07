// SPDX-License-Identifier: Apache-2.0

//! Linear (matmul + optional bias + fused PostOps) codegen — x86_64 SSE2.

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use compiler::PostOp;
use profile_api::LowerError;

#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    model_idx: usize,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
    bias_offset: Option<usize>,
    node_span: Span,
    fused_post_ops: &[PostOp],
    sym_prefix: &str,
) -> Result<String, LowerError> {
    let lid = format!("{model_idx}_{linear_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}{}\n",
        if bias_offset.is_some() { " + bias" } else { "" },
        if !fused_post_ops.is_empty() {
            " + fused"
        } else {
            ""
        },
    ));

    // 1. Pointer setup.
    s.push_str(&materialise_ptr("%r8", src_loc)); // src ptr
    s.push_str(&materialise_ptr("%r11", dst_loc)); // dst ptr

    // weight base = %rsi + weight_offset*4
    if weight_offset == 0 {
        s.push_str("    movq    %rsi, %r9\n");
    } else {
        s.push_str(&format!("    leaq    {}(%rsi), %r9\n", weight_offset * 4));
    }
    let needs_zero_xmm4 = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
    if needs_zero_xmm4 {
        s.push_str("    xorps   %xmm4, %xmm4\n");
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str("    movq    %rsi, %r12\n");
        } else {
            s.push_str(&format!("    leaq    {}(%rsi), %r12\n", boff * 4));
        }
    }

    // 2. Outer i-loop: %rax = i, compared against b.
    s.push_str("    xorq    %rax, %rax\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lmm_i_end_{lid}\n"));

    // 3. Inner j-loop: %rcx = j, compared against n.
    s.push_str("    xorq    %rcx, %rcx\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm_j_end_{lid}\n"));

    // 4. Innermost k-loop: %xmm0 = sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str("    xorps   %xmm0, %xmm0\n"); // sum init
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lmm_k_end_{lid}\n"));

    // src offset = i*k + kk; load src[i*k + kk] → xmm1
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %rax, %r15\n");
    s.push_str("    imulq   %r10, %r15\n"); // %r15 = i * k
    s.push_str("    addq    %r14, %r15\n"); // %r15 = i*k + kk
    s.push_str("    movss   (%r8, %r15, 4), %xmm1\n"); // xmm1 = src[i*k + kk]

    // weight offset = kk*n + j; load weights[kk*n + j] → xmm2
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %r14, %r15\n");
    s.push_str("    imulq   %r10, %r15\n"); // %r15 = kk * n
    s.push_str("    addq    %rcx, %r15\n"); // %r15 = kk*n + j
    s.push_str("    movss   (%r9, %r15, 4), %xmm2\n");

    // sum += xmm1 * xmm2  (no FMA)
    s.push_str("    mulss   %xmm2, %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");

    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // 5. Bias-add (if present): xmm0 += bias[j].
    if bias_offset.is_some() {
        s.push_str("    movss   (%r12, %rcx, 4), %xmm5\n");
        s.push_str("    addss   %xmm5, %xmm0\n");
    }

    // 6. Elementwise post-ops: applied inline inside the j-loop.
    //    Row-wise post-ops (SoftmaxRow) skipped here; emitted after the
    //    matmul loop completes.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => s.push_str("    maxss   %xmm4, %xmm0\n"),
            PostOp::SoftmaxRow => {} // row-wise; handled after the matmul.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: post_op.to_string(),
                    span: node_span,
                });
            }
        }
    }

    // 7. Store xmm0 → dst[i*n + j]
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %rax, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");
    s.push_str("    addq    %rcx, %r15\n");
    s.push_str("    movss   %xmm0, (%r11, %r15, 4)\n");

    s.push_str("    incq    %rcx\n");
    s.push_str(&format!("    jmp     .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    // 8. Row-wise post-ops (SoftmaxRow tail) run after the matmul loop.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => {} // already inlined above
            PostOp::SoftmaxRow => {
                s.push_str(&emit_fused_softmax_tail(b, n, &lid, sym_prefix));
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: post_op.to_string(),
                    span: node_span,
                });
            }
        }
    }

    Ok(s)
}

/// Fused-softmax row-wise tail. Operates in-place on dst[%r11].
///
/// Register contract (callee-saved by prologue's
/// `compute_callee_saved` whenever this emitter fires):
///   %rbx = src ptr (= %r11; same buffer for in-place)
///   %r12 = dst ptr (= %r11)
///   %r13 = i (outer row counter)
///   %r14 = j (inner column counter)
///   %r15 = row_base = i * n
///
/// Stack-resident state across `call expf@PLT`:
///   (%rsp)  = row_max f32 slot (offset 0)
///   8(%rsp) = row_sum f32 slot (offset 8)
/// The 16-byte spill region is reserved at the bottom of the frame by
/// `assign_buffers` whenever `model.calls_extern_math()` (spec §7.4);
/// `BufferAssignment::stack_bytes` already accounts for it, so the
/// prologue's `subq $frame_size, %rsp` covers both slots and any
/// intermediate buffers without per-emitter parameterisation.
fn emit_fused_softmax_tail(b: u64, n: u64, lid: &str, sym_prefix: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "    # fused softmax_row: output [{b},{n}] in-place\n"
    ));
    s.push_str("    movq    %r11, %rbx\n"); // src = dst (in-place)
    s.push_str("    movq    %r11, %r12\n"); // dst = dst

    // Outer per-row loop: %r13 = i.
    s.push_str("    xorq    %r13, %r13\n");
    s.push_str(&format!(".Lfsmx_i_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r13\n");
    s.push_str(&format!("    jge     .Lfsmx_i_end_{lid}\n"));

    // %r15 = i * n
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %r13, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");

    // Phase 2: row_max → (%rsp). Init xmm8 to -inf.
    s.push_str("    movl    $0xFF800000, %r10d\n"); // -inf bits
    s.push_str("    movd    %r10d, %xmm8\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_max_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_max_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n"); // %rax = row_base + j
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    maxss   %xmm0, %xmm8\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_max_{lid}\n"));
    s.push_str(&format!(".Lfsmx_max_end_{lid}:\n"));
    // Spill row_max to stack slot 0 (xmm regs are caller-saved across call).
    s.push_str("    movss   %xmm8, (%rsp)\n");

    // Phase 3: exp(x − max), sum → 8(%rsp). Init sum slot to 0.
    s.push_str("    movl    $0, 8(%rsp)\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_exp_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_exp_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n"); // %rax = row_base + j
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    subss   (%rsp), %xmm0\n");
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax was clobbered; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n"); // write exp result back
    s.push_str("    movss   8(%rsp), %xmm1\n");
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str("    movss   %xmm1, 8(%rsp)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_exp_{lid}\n"));
    s.push_str(&format!(".Lfsmx_exp_end_{lid}:\n"));

    // Phase 4: normalise by row_sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lfsmx_norm_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lfsmx_norm_end_{lid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%r12, %rax, 4), %xmm0\n");
    s.push_str("    divss   8(%rsp), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lfsmx_norm_{lid}\n"));
    s.push_str(&format!(".Lfsmx_norm_end_{lid}:\n"));

    // Next row.
    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lfsmx_i_{lid}\n"));
    s.push_str(&format!(".Lfsmx_i_end_{lid}:\n"));
    s
}
