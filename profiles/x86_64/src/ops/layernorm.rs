// SPDX-License-Identifier: Apache-2.0

//! LayerNorm codegen — x86_64 SSE2 (M14).
//!
//! Per-row 3-pass: mean → variance + inv_std → normalize + optional affine.
//! Native `sqrtss` — no libm call. No-affine path: 3 op-local callee-saved
//! pushes (%r15, %rbx, %r14 — M15 LH-4 adds %r15 for per-row src ptr). With-
//! affine adds 2 more (%r12, %r13 for γ/β base ptrs) → 5 total. All op-local —
//! function-level `compute_callee_saved` unchanged (M13 pre-Task-5 arm64
//! precedent). No FFI calls (native `sqrtss`).
//!
//! Pointer materialisation order (M14 bugfix): the op-local `pushq` block
//! comes FIRST (saves caller's callee-saved values), then base pointers are
//! materialised AFTER pushes WITH stack-offset bias compensation. For
//! `BufferLoc::StackOffset(N)`, the helper `materialise_ptr_with_rsp_bias`
//! emits `leaq (N + push_bytes)(%rsp), <reg>` so the stack-relative address
//! recovers the function-frame %rsp. Two-property fix:
//!
//!   1. push first → caller's %rbx/%r14/%r12/%r13 preserved on stack
//!   2. materialise with bias → correct buffer addresses despite shifted %rsp
//!
//! Pre-bug-fix attempt (commit 65c24b6) tried materialise-then-push pattern
//! mirroring emit_linear, but that corrupts caller's callee-saved registers
//! (push saves the OVERWRITTEN values, not caller's), surfacing as a Rust-side
//! FFI test failure when the caller's %rbx/%r12/%r14 hold loop state. The
//! current order is the correct fix.
//!
//! Register plan (M15 LH-4 closed, N=1..4 scope, finalized):
//!   %rax  = x_j (inner counter); also temp for byte-offset compute before counter use
//!   %r10  = bound scratch (clobbered every emit_imm32_to_r10 call)
//!   %r11  = x_i (outer counter)
//!   %r15  = x_in  (per-row input ptr;  recomputed per row) — callee-saved + op-local push/pop (M15 LH-4 — was %r8 pre-M15)
//!   %rbp  = x_out (per-row output ptr; recomputed per row) — callee-saved + function-level prologue handles (M15 LH-4 — was %r9 pre-M15)
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
//! All scratch in non-INPUT_REGS scope at N=1..4 (M15 closes LH-4 with
//! transformer_block.nfl runtime evidence at N=3; N=4 closure is asm-only,
//! mirroring M14 LH-2/3 precedent for emit_linear N=4).
//!
//! Why %rax for the inner counter (not %r10): emit_imm32_to_r10 clobbers
//! %r10 every call, so %r10 cannot hold a long-lived counter. %rax is
//! caller-saved scratch (free for op use without save/restore) and is
//! NOT touched by emit_imm32_to_r10.
//!
//! Stack alignment invariant (M-future foot-gun): unconditional pushes
//! %r15 + %rbx + %r14 add +24 bytes (3 pushes, no-affine path). Affine
//! path adds %r12 + %r13 → +40 bytes total (5 pushes). Both totals are
//! ≡ 8 mod 16 (odd-by-8). `pushq %rbp` in function prologue is also +8.
//! Inside-body %rsp is therefore NOT 16-byte aligned — OK because
//! emit_layernorm is leaf (native sqrtss, no `call` site in body).

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is enabled (5 pushq: %r15, %r12, %r13, %rbx, %r14).
///
/// MUST equal 8 × number of `pushq` instructions emitted in the affine save
/// block of `emit_layernorm`. If a future change adds/removes a pushq without
/// updating this constant, `materialise_ptr_with_rsp_bias`'s debug_assert will
/// fire — the address compensation depends on this number being exact.
const OP_LOCAL_PUSH_BYTES_AFFINE: usize = 5 * 8;

/// Bytes pushed to stack by the op-local callee-saved save block when affine
/// is disabled (3 pushq: %r15, %rbx, %r14). Same invariant as the affine const —
/// keep in sync with the actual push count in `emit_layernorm`.
const OP_LOCAL_PUSH_BYTES_NO_AFFINE: usize = 3 * 8;

/// Materialise a `BufferLoc` into `dst_reg`, with `rsp_bias_bytes` added to
/// stack-relative offsets to compensate for op-local `pushq` instructions
/// that have decremented `%rsp` since the function-frame base. For non-stack
/// locations (`InputReg`/`OutputReg`), delegates to `abi.materialise_ptr`
/// (no bias needed — those use ABI argument registers, not `%rsp`).
///
/// `rsp_bias_bytes` MUST match one of the two `OP_LOCAL_PUSH_BYTES_*` consts
/// — guarded by debug_assert so a divergence between push count and bias
/// fires loudly in dev/test rather than producing silent buffer-address
/// corruption.
fn materialise_ptr_with_rsp_bias(
    abi: &AbiContext,
    loc: BufferLoc,
    dst_reg: &str,
    rsp_bias_bytes: usize,
    s: &mut String,
) {
    debug_assert!(
        rsp_bias_bytes == OP_LOCAL_PUSH_BYTES_AFFINE
            || rsp_bias_bytes == OP_LOCAL_PUSH_BYTES_NO_AFFINE,
        "rsp_bias_bytes ({}) must match one of the OP_LOCAL_PUSH_BYTES_* consts \
         (affine={}, no-affine={}). If push count in emit_layernorm changed, \
         update the const to match.",
        rsp_bias_bytes,
        OP_LOCAL_PUSH_BYTES_AFFINE,
        OP_LOCAL_PUSH_BYTES_NO_AFFINE,
    );
    match loc {
        BufferLoc::StackOffset(n) => {
            let adjusted = n + rsp_bias_bytes;
            if adjusted == 0 {
                s.push_str(&format!("    movq    %rsp, {}\n", dst_reg));
            } else {
                s.push_str(&format!("    leaq    {}(%rsp), {}\n", adjusted, dst_reg));
            }
        }
        _ => abi.materialise_ptr(loc, dst_reg, s),
    }
}

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

    // Op-local callee-saved save: push CALLER's values FIRST so they can be
    // restored at body exit. The body overwrites %rbx/%r14 (and %r12/%r13 if
    // affine) below; the LIFO pop at body exit restores caller's values.
    //
    // Stack-relative materialise compensation: each pushq decrements %rsp by 8.
    // After this block, %rsp is `op_local_push_bytes` lower than at function-
    // frame stable point. `materialise_ptr` for BufferLoc::StackOffset(N) emits
    // `leaq N(%rsp), <reg>` using current %rsp — to recover the caller's
    // intended buffer address we must add `op_local_push_bytes` to N. Pre-fix,
    // this adjustment was missing → silent buffer-address corruption when
    // src_loc / dst_loc was a StackOffset (e.g. pre_ln_block fixture).
    let op_local_push_bytes = if has_affine {
        OP_LOCAL_PUSH_BYTES_AFFINE
    } else {
        OP_LOCAL_PUSH_BYTES_NO_AFFINE
    };
    s.push_str("    pushq   %r15\n");
    if has_affine {
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
    }
    s.push_str("    pushq   %rbx\n");
    s.push_str("    pushq   %r14\n");

    // Materialise base pointers AFTER the pushes, with stack-offset adjustment.
    materialise_ptr_with_rsp_bias(abi, src_loc, "%rbx", op_local_push_bytes, &mut s);
    materialise_ptr_with_rsp_bias(abi, dst_loc, "%r14", op_local_push_bytes, &mut s);

    // Affine: materialise γ/β base pointers from params_reg into %r12/%r13.
    // params_reg is one of the ABI input registers (%rsi at N=1, %rdx at N=2,
    // etc.) — never %rsp-relative — so no offset adjustment needed.
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

    // Compute per-row pointers: %r15 = src_base + i*d*4; %rbp = dst_base + i*d*4.
    // Use %rax as transient byte-offset accumulator (then re-zeroed for Pass 1
    // counter use below — emit_imm32_to_r10 doesn't touch %rax).
    //
    // M15 LH-4: per-row scratch was %r8/%r9 pre-M15; relocated to %r15 (op-local
    // pushq/popq) and %rbp (function-level prologue handles) to avoid clobbering
    // output_reg at N=3 (=%r8) and params_reg/output_reg at N=4 (=%r8/%r9).
    s.push_str(&emit_imm32_to_r10((d * 4) as u32));
    s.push_str("    movq    %r11, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * d * 4 (byte offset)
    s.push_str("    leaq    (%rbx, %rax, 1), %r15\n");
    s.push_str("    leaq    (%r14, %rax, 1), %rbp\n");

    // === Pass 1: μ = (1/D) · Σ x_j ===
    s.push_str("    xorps   %xmm0, %xmm0\n"); // s_acc = 0
    s.push_str("    xorq    %rax, %rax\n"); // x_j = 0 (counter)
    s.push_str(&format!(".Lln_p1_{lid}:\n"));
    s.push_str(&emit_imm32_to_r10(d as u32)); // bound = d → %r10
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lln_p1_end_{lid}\n"));
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
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
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
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
    s.push_str("    movss   (%r15, %rax, 4), %xmm6\n");
    s.push_str("    subss   %xmm1, %xmm6\n"); // %xmm6 -= mean
    s.push_str("    mulss   %xmm5, %xmm6\n"); // %xmm6 *= s_inv_std (NOT divss — Q4)
    if has_affine {
        s.push_str("    movss   (%r12, %rax, 4), %xmm7\n"); // γ_j → %xmm7
        s.push_str("    mulss   %xmm7, %xmm6\n");
        s.push_str("    movss   (%r13, %rax, 4), %xmm8\n"); // β_j → %xmm8
        s.push_str("    addss   %xmm8, %xmm6\n");
    }
    s.push_str("    movss   %xmm6, (%rbp, %rax, 4)\n");
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
    s.push_str("    popq    %r15\n");

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
