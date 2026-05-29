// SPDX-License-Identifier: Apache-2.0

//! Linear (matmul + optional bias + fused PostOps) codegen — x86_64 SSE2.
//!
//! M12 multi-input ABI migration: data-flow accesses to the params and
//! output registers are routed through `AbiContext::params_reg()` /
//! `AbiContext::output_reg()`. For N=1 these resolve to `%rsi` / `%rdx`
//! — bit-identical to M3-M11. For N≥2 they shift (e.g. params → `%rdx`,
//! output → `%rcx` for N=2).
//!
//! M13 ABI-register save (N≥2): the inner k-loop scratch uses `%rsi` for
//! offset arithmetic and `%rdi` as the k-counter. At N=1 these are non-ABI
//! (params=`%rsi`, output=`%rdx`; `%rdi` is pure scratch). At N≥2:
//!
//!   - `%rsi` becomes input(1) — body clobber would break the next op's
//!     read of the second input pointer.
//!   - `%rdi` is always input(0) — body clobber is invisible if no
//!     downstream emitter re-reads input(0).
//!
//! Both (`%rdi`/`%rsi`) saved via `pushq` at body entry, `popq` at exit (N≥2 only).
//! `%r14` and `%r15` (LH-2/3 op-local scratch) are saved unconditionally at
//! all N — see the LH-2/3 bullets below.
//!
//! LH-1/2/3 cleanup: three latent hazards resolved uniformly via
//! ABI-register relocation. All scratch now lives in non-INPUT_REGS scope
//! at all N=1..4.
//!
//!   - LH-1 (was: N=2 + linear-with-bias). j-counter relocated `%rcx` →
//!     `%rbp`. `%rbp` is callee-saved by the unconditional function-level
//!     prologue (`pushq %rbp`/`popq %rbp` in `asm.rs::format_function_prologue`)
//!     and unread by op bodies. M13 Task 1 precedent for emit_matmul.
//!     Side-effect: M13 save block's `pushq %rcx`/`popq %rcx` removed
//!     (dead — body no longer writes %rcx).
//!
//!   - LH-2 (was: N=3 src ptr scratch %r8 = output_reg). src ptr scratch
//!     relocated `%r8` → `%r14`. `%r14` is callee-saved per SysV; op-local
//!     `pushq %r14`/`popq %r14` brackets the body. M13 pre-Task-5 arm64
//!     precedent for op-local save/restore. `compute_callee_saved` is
//!     NOT extended — function-level prologue unchanged.
//!
//!   - LH-3 (was: N=4 weight ptr scratch %r9 = output_reg). weight ptr
//!     scratch relocated `%r9` → `%r15`. Same op-local push/pop pattern
//!     as LH-2.
//!
//! No remaining latent hazards in this file. Future-proofing: any
//! addition of new scratch registers MUST verify against INPUT_REGS at
//! the highest supported N (currently N=4) before merging — see
//! ABI-invariant tests `emit_linear_n{2,3,4}_does_not_clobber_output_reg`
//! in `profiles/x86_64/src/tests.rs`.
//!
//! Cross-reference: same class of bug as M13 Task 1 (`emit_matmul`
//! `%r9` → `%rbp`) and the M13 pre-Task-5 arm64 emit_linear x3/x4/x5
//! stp/ldp fix. M14 closes the analogous x86_64 cases uniformly.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use compiler::PostOp;
use profile_api::LowerError;

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
        "    # matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]{}{}\n",
        if bias_offset.is_some() { " + bias" } else { "" },
        if !fused_post_ops.is_empty() {
            " + fused"
        } else {
            ""
        },
    ));

    let params_reg = abi.params_reg();
    let output_reg = abi.output_reg();

    // 1. Pointer setup.
    // src ptr scratch: %r14 avoids output_reg alias at N=3 (callee-saved; op-local save below).
    abi.materialise_ptr(src_loc, "%r14", &mut s); // src ptr
    abi.materialise_ptr(dst_loc, "%r11", &mut s); // dst ptr

    // weight base = params_reg + weight_offset*4
    // weight ptr scratch: %r15 avoids output_reg alias at N=4 (callee-saved; op-local save below).
    if weight_offset == 0 {
        s.push_str(&format!("    movq    {}, %r15\n", params_reg));
    } else {
        s.push_str(&format!(
            "    leaq    {}({}), %r15\n",
            weight_offset * 4,
            params_reg
        ));
    }
    let needs_zero_xmm4 = fused_post_ops.iter().any(|p| matches!(p, PostOp::Relu));
    if needs_zero_xmm4 {
        s.push_str("    xorps   %xmm4, %xmm4\n");
    }
    // Save the FFI output pointer (output_reg) into %xmm7 BEFORE the
    // bias-base setup overwrites it. A subsequent op in the same function
    // (e.g. standalone emit_softmax following an unfused linear-with-bias)
    // calls abi.materialise_ptr(OutputReg, ...) which reads output_reg;
    // if we don't restore it, the destination would point at the bias
    // buffer instead of the caller's output buffer, and the standalone-
    // softmax writes would land in the wrong place (manifesting as zeroed
    // output — exactly what `fused_vs_unfused_softmax_match_numerically`
    // caught).
    //
    // Skip when SoftmaxRow is fused — that's the LAST op in the model,
    // and no follow-up consumer needs output_reg. Saving across
    // `call expf@PLT` is also pointless (xmm7 is caller-saved under SysV).
    let has_softmax_row = fused_post_ops
        .iter()
        .any(|p| matches!(p, PostOp::SoftmaxRow));
    let preserve_output_ptr = bias_offset.is_some() && !has_softmax_row;
    if preserve_output_ptr {
        s.push_str(&format!("    movq    {}, %xmm7\n", output_reg));
    }
    if let Some(boff) = bias_offset {
        if boff == 0 {
            s.push_str(&format!("    movq    {}, {}\n", params_reg, output_reg));
        } else {
            s.push_str(&format!(
                "    leaq    {}({}), {}\n",
                boff * 4,
                params_reg,
                output_reg
            ));
        }
    }

    // Save params ptr (params_reg) into %xmm6 BEFORE the matmul body
    // clobbers it as offset scratch. The next linear in the same function
    // (e.g. linear[1] → linear[2] in a multi-layer model) reads
    // params_reg at the top of its own emit_linear (`leaq weight_offset
    // (params_reg), %r9`); if we don't preserve it, that read produces a
    // wild pointer and SIGSEGVs on the first weight load.
    //
    // Skip the save when SoftmaxRow is fused — that's the LAST linear in
    // the model (softmax is always terminal), so no follow-up emit_linear
    // needs params_reg. Saving across the `call expf@PLT` is also pointless
    // because xmm6 is caller-saved under SysV; the call would clobber it.
    let preserve_params_ptr = !has_softmax_row;
    if preserve_params_ptr {
        s.push_str(&format!("    movq    {}, %xmm6\n", params_reg));
    }

    // ABI-register save (N≥2 only). See module doc-comment for the
    // full rationale. The matching pop block lives right after the
    // matmul i-loop end label, before the SoftmaxRow tail.
    // %rcx save omitted — j-counter relocated to %rbp, body no longer writes %rcx.
    let save_abi = abi.n_inputs >= 2;
    if save_abi {
        s.push_str("    pushq   %rdi\n");
        s.push_str("    pushq   %rsi\n");
    }
    // Op-local save of callee-saved scratch used as src/weight ptrs. Lives
    // inside op body — function-level compute_callee_saved unchanged.
    // arm64 pre-Task-5 precedent for op-local save/restore.
    s.push_str("    pushq   %r14\n");
    s.push_str("    pushq   %r15\n");

    // 2. Outer i-loop: %rax = i, compared against b.
    s.push_str("    xorq    %rax, %rax\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lmm_i_end_{lid}\n"));

    // 3. Inner j-loop: %rbp = j, compared against n. j-counter in %rbp
    //    avoids output_reg alias at N=2 (%rcx); %rbp is callee-saved by
    //    function-level prologue, never read by op bodies.
    s.push_str("    xorq    %rbp, %rbp\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %rbp\n");
    s.push_str(&format!("    jge     .Lmm_j_end_{lid}\n"));

    // 4. Innermost k-loop: %xmm0 = sum.
    s.push_str("    xorq    %rdi, %rdi\n");
    s.push_str("    xorps   %xmm0, %xmm0\n"); // sum init
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %rdi\n");
    s.push_str(&format!("    jge     .Lmm_k_end_{lid}\n"));

    // src offset = i*k + kk; load src[i*k + kk] → xmm1
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %rax, %rsi\n");
    s.push_str("    imulq   %r10, %rsi\n"); // %rsi = i * k
    s.push_str("    addq    %rdi, %rsi\n"); // %rsi = i*k + kk
    s.push_str("    movss   (%r14, %rsi, 4), %xmm1\n"); // xmm1 = src[i*k + kk]

    // weight offset = kk*n + j; load weights[kk*n + j] → xmm2
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %rdi, %rsi\n");
    s.push_str("    imulq   %r10, %rsi\n"); // %rsi = kk * n
    s.push_str("    addq    %rbp, %rsi\n"); // %rsi = kk*n + j
    s.push_str("    movss   (%r15, %rsi, 4), %xmm2\n");

    // sum += xmm1 * xmm2  (no FMA)
    s.push_str("    mulss   %xmm2, %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");

    s.push_str("    incq    %rdi\n");
    s.push_str(&format!("    jmp     .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    // 5. Bias-add (if present): xmm0 += bias[j]. Bias base lives in
    // output_reg (re-purposed as scratch for the duration of the body —
    // restored from %xmm7 below).
    if bias_offset.is_some() {
        s.push_str(&format!("    movss   ({}, %rbp, 4), %xmm5\n", output_reg));
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
    s.push_str("    movq    %rax, %rsi\n");
    s.push_str("    imulq   %r10, %rsi\n");
    s.push_str("    addq    %rbp, %rsi\n");
    s.push_str("    movss   %xmm0, (%r11, %rsi, 4)\n");

    s.push_str("    incq    %rbp\n");
    s.push_str(&format!("    jmp     .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    // Op-local restore (LIFO of entry op-local push):
    s.push_str("    popq    %r15\n");
    s.push_str("    popq    %r14\n");
    // ABI-register restore (LIFO of the entry save block). Runs BEFORE the
    // SoftmaxRow tail so any `call expf@PLT` in the tail sees a properly-
    // aligned RSP and uncorrupted ABI registers. %rcx pop omitted —
    // j-counter relocated to %rbp, body no longer writes %rcx.
    if save_abi {
        s.push_str("    popq    %rsi\n");
        s.push_str("    popq    %rdi\n");
    }

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

    // 9. Restore params ptr (params_reg) from %xmm6 so the next emit_linear
    //    in the same function (in multi-layer models) reads the correct
    //    pointer at the top of its weight-base setup. No-op if we didn't
    //    save above (SoftmaxRow case — this is the last linear).
    if preserve_params_ptr {
        s.push_str(&format!("    movq    %xmm6, {}\n", params_reg));
    }

    // 10. Restore output ptr (output_reg) from %xmm7 so a subsequent op
    //     (e.g. standalone emit_softmax following unfused linear-with-bias)
    //     can re-materialise OutputReg via `movq output_reg, ...`. No-op if
    //     we didn't save above (no bias OR SoftmaxRow fused).
    if preserve_output_ptr {
        s.push_str(&format!("    movq    %xmm7, {}\n", output_reg));
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
/// `assign_buffers` whenever `model.has_softmax()` (spec §7.4);
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
