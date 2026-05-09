// SPDX-License-Identifier: Apache-2.0

//! Linear (matmul + optional bias-add) codegen.

use crate::abi::AbiContext;
use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use compiler::PostOp;
use profile_api::LowerError;

/// Emit AArch64 asm for a linear layer (matmul + optional bias-add).
///
/// `model_idx` and `linear_idx` together uniquely name every label in the
/// output file, which is critical when multiple models share one assembly
/// source (e.g. pipeline_styles.nfl with 3 model definitions).
///
/// M6 added `node_span` and `fused_post_ops` and wired the `PostOp::SoftmaxRow` dispatch (see `arm64.md` §4.10).
#[allow(clippy::too_many_arguments)]
pub fn emit_linear(
    abi: &AbiContext,
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
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}{}\n",
        if bias_offset.is_some() { " + bias" } else { "" },
        if !fused_post_ops.is_empty() {
            " + fused"
        } else {
            ""
        },
    ));

    // Materialise s4 = 0.0 once if any post-op needs it (currently only Relu does).
    let needs_zero = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
    if needs_zero {
        s.push_str("    fmov    s4, wzr\n");
    }

    abi.materialise_ptr(src_loc, "x11", &mut s);
    abi.materialise_ptr(dst_loc, "x12", &mut s);
    let params_reg = abi.params_reg();
    if weight_offset == 0 {
        s.push_str(&format!("    mov     x13, {}\n", params_reg));
    } else {
        s.push_str(&emit_imm32("x9", weight_offset));
        s.push_str(&format!("    add     x13, {}, x9, lsl #2\n", params_reg));
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str(&format!("    mov     x14, {}\n", params_reg));
        } else {
            s.push_str(&emit_imm32("x9", boff));
            s.push_str(&format!("    add     x14, {}, x9, lsl #2\n", params_reg));
        }
    }

    // M8: hoist matmul loop bounds into x10/x15/x16 once, before the
    // i-loop label. All bl-free cmps inside use register form. Stride
    // movs reuse the hoisted registers (no re-materialise).
    s.push_str(&emit_imm32("x10", b as usize));
    s.push_str(&emit_imm32("x15", n as usize));
    s.push_str(&emit_imm32("x16", k as usize));

    // M13 ABI-register save: emit_linear uses x3/x4/x5 as i/j/k loop
    // counters. For N≥2, x3..x5 overlap with ABI argument registers
    // (INPUT_REGS[3..5] hold ABI slots used by downstream emitters).
    // Save any conflicting registers in the ABI range with push/pop pairs
    // that are balanced within this emitter (sp is correctly restored
    // before the emitter returns). For N=1, no overlap — no save needed.
    //
    // Overlap rule: INPUT_REGS = [x0..x5]; ABI uses INPUT_REGS[..n+2].
    //   x3 (INPUT_REGS[3]) conflicts when n_inputs >= 2.
    //   x4 (INPUT_REGS[4]) conflicts when n_inputs >= 3.
    //   x5 (INPUT_REGS[5]) conflicts when n_inputs >= 4.
    //
    // We push pairs (aligned to 16 bytes); a single conflicting register
    // is padded with xzr. Save order: x3, then x4, then x5 (from outer
    // loop to inner); restore in LIFO order.
    let save_x3 = abi.n_inputs >= 2;
    let save_x4 = abi.n_inputs >= 3;
    let save_x5 = abi.n_inputs >= 4;
    if save_x3 && save_x4 {
        s.push_str("    stp     x3, x4, [sp, #-16]!\n");
    } else if save_x3 {
        s.push_str("    stp     x3, xzr, [sp, #-16]!\n");
    }
    if save_x5 {
        s.push_str("    stp     x5, xzr, [sp, #-16]!\n");
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str("    cmp     x3, x10\n");
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str("    cmp     x4, x15\n");
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str("    cmp     x5, x16\n");
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str("    mov     x8, x16\n");
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str("    mov     x8, x15\n");
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // Bias-add (if present) BEFORE post-ops.
    if bias_offset.is_some() {
        s.push_str("    ldr     s5, [x14, x4, lsl #2]\n");
        s.push_str("    fadd    s0, s0, s5\n");
    }

    // Elementwise post-ops: applied inline inside the j-loop, element by element.
    // Row-wise post-ops (SoftmaxRow) are skipped here and emitted after the
    // matmul loop completes (see the second post-op dispatch block below).
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => s.push_str("    fmax    s0, s0, s4\n"),
            // SoftmaxRow is row-wise; handled after the matmul loops.
            PostOp::SoftmaxRow => {}
            // PostOp is `#[non_exhaustive]`; wildcard required.
            // Drop the `#[allow]` when a third `PostOp` variant lands.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(LowerError::UnsupportedPostOp {
                    op: post_op.to_string(),
                    span: node_span,
                });
            }
        }
    }

    // Store after elementwise post-ops. Reuse hoisted x15 (= n).
    s.push_str("    mov     x8, x15\n");
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    // M13 ABI-register restore (LIFO mirror of the save block above).
    if save_x5 {
        s.push_str("    ldp     x5, xzr, [sp], #16\n");
    }
    if save_x3 && save_x4 {
        s.push_str("    ldp     x3, x4, [sp], #16\n");
    } else if save_x3 {
        s.push_str("    ldp     x3, xzr, [sp], #16\n");
    }

    // Row-wise post-ops run after the full matmul loop completes. These
    // require the entire output row to be written before they can proceed.
    for post_op in fused_post_ops {
        match post_op {
            PostOp::Relu => {} // already emitted inline above
            PostOp::SoftmaxRow => {
                // Emit 3-pass stable softmax over the output buffer [b, n] in-place.
                // x12 still holds the dst pointer (set before the matmul loop).
                // Uses the same callee-saved register convention as emit_softmax:
                //   x19=i, x20=row_base, x21=j, x22=src(=dst), x23=dst(=dst)
                //   s8=max, s9=sum  (callee-saved; saved by the function prologue)
                // "fsmx" prefix (fused-softmax) avoids label collision with any
                // standalone .Lsm_* labels emitted by emit_softmax in the same model.
                s.push_str(&format!(
                    "    ; fused softmax_row: output [{b},{n}] in-place\n"
                ));
                s.push_str("    mov     x22, x12\n");
                s.push_str("    mov     x23, x12\n");

                s.push_str("    mov     x19, #0\n");
                s.push_str(&format!(".Lfsmx_i_{lid}:\n"));
                s.push_str(&emit_imm32("x10", b as usize));
                s.push_str("    cmp     x19, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_i_end_{lid}\n"));

                s.push_str(&emit_imm32("x8", n as usize));
                s.push_str("    mul     x20, x19, x8\n");

                // Phase 2: row-max → s8.
                s.push_str("    movz    w0, #0x0000\n");
                s.push_str("    movk    w0, #0xFF80, lsl #16\n");
                s.push_str("    fmov    s8, w0\n");
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_max_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_max_end_{lid}\n"));
                s.push_str("    add     x6, x20, x21\n");
                s.push_str("    ldr     s1, [x22, x6, lsl #2]\n");
                s.push_str("    fmax    s8, s8, s1\n");
                s.push_str("    add     x21, x21, #1\n");
                s.push_str(&format!("    b       .Lfsmx_max_{lid}\n"));
                s.push_str(&format!(".Lfsmx_max_end_{lid}:\n"));

                // Phase 3: exp(x − max), sum → s9.
                s.push_str("    fmov    s9, wzr\n");
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_exp_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_exp_end_{lid}\n"));
                s.push_str("    add     x6, x20, x21\n");
                s.push_str("    ldr     s0, [x22, x6, lsl #2]\n");
                s.push_str("    fsub    s0, s0, s8\n");
                s.push_str(&format!("    bl      {}expf\n", sym_prefix));
                // x6 may have been clobbered by _expf (caller-saved); recompute.
                s.push_str("    add     x6, x20, x21\n");
                s.push_str("    str     s0, [x23, x6, lsl #2]\n");
                s.push_str("    fadd    s9, s9, s0\n");
                s.push_str("    add     x21, x21, #1\n");
                s.push_str(&format!("    b       .Lfsmx_exp_{lid}\n"));
                s.push_str(&format!(".Lfsmx_exp_end_{lid}:\n"));

                // Phase 4: normalise by s9.
                s.push_str("    mov     x21, #0\n");
                s.push_str(&format!(".Lfsmx_norm_{lid}:\n"));
                s.push_str(&emit_imm32("x10", n as usize));
                s.push_str("    cmp     x21, x10\n");
                s.push_str(&format!("    b.ge    .Lfsmx_norm_end_{lid}\n"));
                s.push_str("    add     x6, x20, x21\n");
                s.push_str("    ldr     s0, [x23, x6, lsl #2]\n");
                s.push_str("    fdiv    s0, s0, s9\n");
                s.push_str("    str     s0, [x23, x6, lsl #2]\n");
                s.push_str("    add     x21, x21, #1\n");
                s.push_str(&format!("    b       .Lfsmx_norm_{lid}\n"));
                s.push_str(&format!(".Lfsmx_norm_end_{lid}:\n"));

                s.push_str("    add     x19, x19, #1\n");
                s.push_str(&format!("    b       .Lfsmx_i_{lid}\n"));
                s.push_str(&format!(".Lfsmx_i_end_{lid}:\n"));
            }
            // PostOp is `#[non_exhaustive]`; wildcard required for cross-crate
            // match completeness. Drop the `#[allow]` when a third `PostOp` variant lands.
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
