// SPDX-License-Identifier: Apache-2.0

//! LayerNorm codegen — x86_64 SSE2 (M14).
//!
//! Per-row 3-pass: mean → variance + inv_std → normalize + optional affine.
//! Native `sqrtss` — no libm call. No-affine path: 2 op-local callee-saved
//! pushes (%rbx, %r14 for src/dst base ptrs). With-affine adds 2 more
//! (%r12, %r13 for γ/β base ptrs). All op-local — function-level
//! `compute_callee_saved` unchanged (M13 pre-Task-5 arm64 precedent). No
//! FFI calls (native `sqrtss`).
//!
//! Register plan (M14 spec §8, N=1..2 scope, finalized):
//!   %rax  = x_j (inner counter); also temp for byte-offset compute before counter use
//!   %r10  = bound scratch (clobbered every emit_imm32_to_r10 call)
//!   %r11  = x_i (outer counter)
//!   %r8   = x_in  (per-row input ptr;  recomputed per row) — free at N≤2
//!   %r9   = x_out (per-row output ptr; recomputed per row) — free at N≤2
//!   %rbx  = src_base (set ONCE; callee-saved + op-local push/pop)
//!   %r14  = dst_base (set ONCE; callee-saved + op-local push/pop)
//!   %r12  = x_gamma (γ base ptr) — affine only, callee-saved + op-local push/pop
//!   %r13  = x_beta  (β base ptr) — affine only, callee-saved + op-local push/pop
//!   %xmm0 = s_acc; reused for s_var at end of Pass 2
//!   %xmm1 = s_mean (live Pass 2 + Pass 3)
//!   %xmm2 = s_inv_d (1/D constant; live across outer loop)
//!   %xmm3 = s_eps (1e-5; live across outer loop)
//!   %xmm4 = s_one (1.0; live across outer loop)
//!   %xmm5 = s_inv_std (Q4 constraint — held through Pass 3)
//!   %xmm6 = s_t (per-element temp)
//!   %xmm7 = s_g (γ_j load — affine only)
//!   %xmm8 = s_b (β_j load — affine only; xmm8 caller-saved on SysV, no AAPCS-style concern)
//!
//! All scratch in non-INPUT_REGS scope at N=1..2 (the M14 fixture range).
//! Higher-N (N=3..4) not validated; spec §8.7 documents the deferral.
//!
//! Why %rax for the inner counter (not %r10): emit_imm32_to_r10 clobbers
//! %r10 every call, so %r10 cannot hold a long-lived counter. %rax is
//! caller-saved scratch (free for op use without save/restore) and is
//! NOT touched by emit_imm32_to_r10.
//!
//! Stack alignment invariant (M-future foot-gun): always-pushed %rbx +
//! %r14 add +16 bytes (pair preserves alignment), conditional %r12 + %r13
//! add +16 (pair preserves). `pushq %rbp` in function prologue is +8 → odd.
//! M14 fixtures never co-locate LayerNorm with Softmax (the only op
//! emitting `call expf@PLT`), so misalignment never reaches a `call` site.
//! If a future fixture combines them, add a one-time `subq $8, %rsp`
//! adjustment outside the inner body.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

#[allow(clippy::too_many_arguments)]
pub fn emit_layernorm(
    abi: &AbiContext,
    b: u64,
    d: u64,
    model_idx: usize,
    layernorm_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    gamma_offset: Option<usize>,
    beta_offset: Option<usize>,
    _node_span: Span,
) -> Result<String, LowerError> {
    let lid = format!("{model_idx}_{layernorm_idx}");
    let has_affine = gamma_offset.is_some();
    debug_assert_eq!(
        has_affine,
        beta_offset.is_some(),
        "γ/β offsets must be both Some or both None"
    );

    let mut s = String::new();
    s.push_str(&format!(
        "    # layernorm (3-pass, {}affine): input [{b},{d}] -> output [{b},{d}]\n",
        if has_affine { "with " } else { "no-" },
    ));

    // Op-local callee-saved save. Push order: γ/β first if affine, then
    // base pointers. LIFO pop at function end mirrors this exactly.
    if has_affine {
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
    }
    s.push_str("    pushq   %rbx\n");
    s.push_str("    pushq   %r14\n");

    // Materialise base pointers ONCE before the outer loop.
    abi.materialise_ptr(src_loc, "%rbx", &mut s);
    abi.materialise_ptr(dst_loc, "%r14", &mut s);

    // Affine: materialise γ/β base pointers from params_reg into %r12/%r13.
    if has_affine {
        let g_off = gamma_offset.unwrap();
        let b_off = beta_offset.unwrap();
        let params_reg = abi.params_reg();
        if g_off == 0 {
            s.push_str(&format!("    movq    {}, %r12\n", params_reg));
        } else {
            s.push_str(&format!(
                "    leaq    {}({}), %r12\n",
                g_off * 4,
                params_reg
            ));
        }
        if b_off == 0 {
            s.push_str(&format!("    movq    {}, %r13\n", params_reg));
        } else {
            s.push_str(&format!(
                "    leaq    {}({}), %r13\n",
                b_off * 4,
                params_reg
            ));
        }
    }

    // Pre-loop hoisted constants — RIP-relative loads from .rodata pool
    // (emitted at end of this function).
    let inv_d_bits = (1.0_f32 / d as f32).to_bits();
    let eps_bits = 1e-5_f32.to_bits();
    let one_bits = 1.0_f32.to_bits();
    s.push_str(&format!(
        "    # constants: inv_d=0x{:08x}, eps=0x{:08x}, one=0x{:08x}\n",
        inv_d_bits, eps_bits, one_bits
    ));
    s.push_str(&format!("    movss   .Lln_inv_d_{lid}(%rip), %xmm2\n"));
    s.push_str(&format!("    movss   .Lln_eps_{lid}(%rip), %xmm3\n"));
    s.push_str(&format!("    movss   .Lln_one_{lid}(%rip), %xmm4\n"));

    // === outer row loop: i in 0..b ===
    s.push_str("    xorq    %r11, %r11\n");
    s.push_str(&format!(".Lln_row_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r11\n");
    s.push_str(&format!("    jge     .Lln_row_end_{lid}\n"));

    // Compute per-row pointers: %r8 = src_base + i*d*4; %r9 = dst_base + i*d*4.
    // Use %rax as transient byte-offset accumulator (then re-zeroed for Pass 1
    // counter use below — emit_imm32_to_r10 doesn't touch %rax).
    s.push_str(&emit_imm32_to_r10((d * 4) as u32));
    s.push_str("    movq    %r11, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * d * 4 (byte offset)
    s.push_str("    leaq    (%rbx, %rax, 1), %r8\n");
    s.push_str("    leaq    (%r14, %rax, 1), %r9\n");

    // === Pass 1: μ = (1/D) · Σ x_j ===
    s.push_str("    xorps   %xmm0, %xmm0\n"); // s_acc = 0
    s.push_str("    xorq    %rax, %rax\n"); // x_j = 0 (counter)
    s.push_str(&format!(".Lln_p1_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(d as u32)); // bound = d → %r10
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lln_p1_end_{lid}\n"));
    s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");
    s.push_str("    addss   %xmm6, %xmm0\n");
    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lln_p1_{lid}\n"));
    s.push_str(&format!(".Lln_p1_end_{lid}:\n"));
    s.push_str("    mulss   %xmm2, %xmm0\n"); // %xmm0 *= s_inv_d → s_mean (in %xmm0)
    s.push_str("    movss   %xmm0, %xmm1\n"); // copy s_mean to %xmm1 (stable through Pass 2/3)

    // === Pass 2: σ² = (1/D) · Σ (x_j − μ)²; inv_std = 1/sqrt(σ² + ε) ===
    s.push_str("    xorps   %xmm0, %xmm0\n"); // s_acc = 0
    s.push_str("    xorq    %rax, %rax\n");
    s.push_str(&format!(".Lln_p2_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(d as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lln_p2_end_{lid}\n"));
    s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");
    s.push_str("    subss   %xmm1, %xmm6\n"); // %xmm6 -= mean
    s.push_str("    mulss   %xmm6, %xmm6\n"); // %xmm6 = (x − μ)²
    s.push_str("    addss   %xmm6, %xmm0\n");
    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lln_p2_{lid}\n"));
    s.push_str(&format!(".Lln_p2_end_{lid}:\n"));
    s.push_str("    mulss   %xmm2, %xmm0\n"); // s_var = s_acc * s_inv_d
    s.push_str("    addss   %xmm3, %xmm0\n"); // s_var += s_eps
    s.push_str("    sqrtss  %xmm0, %xmm0\n"); // s_var = sqrt(σ² + ε)
    s.push_str("    movss   %xmm4, %xmm5\n"); // %xmm5 = 1.0 (copy of s_one)
    s.push_str("    divss   %xmm0, %xmm5\n"); // s_inv_std = s_one / s_var
                                              //         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Q4 constraint: ONE divss per row,
                                              //         OUTSIDE Pass 3. Pass 3 hot loop uses mulss %xmm5, ...

    // === Pass 3: y_j = (x_j − μ) · inv_std [· γ_j + β_j] ===
    s.push_str("    xorq    %rax, %rax\n");
    s.push_str(&format!(".Lln_p3_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(d as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lln_p3_end_{lid}\n"));
    s.push_str("    movss   (%r8, %rax, 4), %xmm6\n");
    s.push_str("    subss   %xmm1, %xmm6\n"); // %xmm6 -= mean
    s.push_str("    mulss   %xmm5, %xmm6\n"); // %xmm6 *= s_inv_std (NOT divss — Q4)
    if has_affine {
        s.push_str("    movss   (%r12, %rax, 4), %xmm7\n"); // γ_j → %xmm7
        s.push_str("    mulss   %xmm7, %xmm6\n");
        s.push_str("    movss   (%r13, %rax, 4), %xmm8\n"); // β_j → %xmm8
        s.push_str("    addss   %xmm8, %xmm6\n");
    }
    s.push_str("    movss   %xmm6, (%r9, %rax, 4)\n");
    s.push_str("    incq    %rax\n");
    s.push_str(&format!("    jmp     .Lln_p3_{lid}\n"));
    s.push_str(&format!(".Lln_p3_end_{lid}:\n"));

    s.push_str("    incq    %r11\n");
    s.push_str(&format!("    jmp     .Lln_row_{lid}\n"));
    s.push_str(&format!(".Lln_row_end_{lid}:\n"));

    // Op-local restores — strict LIFO of the entry pushes.
    s.push_str("    popq    %r14\n");
    s.push_str("    popq    %rbx\n");
    if has_affine {
        s.push_str("    popq    %r13\n");
        s.push_str("    popq    %r12\n");
    }

    // Per-function .rodata constants pool. The `.text` directive after the
    // constants restores the assembler to the .text section before
    // format_function_epilogue appends popq/retq — so the epilogue lands in
    // .text as required. No further asm.rs changes needed.
    s.push_str(".section .rodata\n");
    s.push_str(".align 4\n");
    s.push_str(&format!(".Lln_inv_d_{lid}: .long 0x{:08x}\n", inv_d_bits));
    s.push_str(&format!(".Lln_eps_{lid}:   .long 0x{:08x}\n", eps_bits));
    s.push_str(&format!(".Lln_one_{lid}:   .long 0x{:08x}\n", one_bits));
    s.push_str(".text\n");

    Ok(s)
}
