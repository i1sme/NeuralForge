// SPDX-License-Identifier: Apache-2.0

//! LayerNorm codegen — arm64 (M14).
//!
//! Per-row 3-pass: mean (Σ x_i, /D) → variance + inv_std (Σ (x_i−μ)²,
//! /D, +ε, fsqrt, 1/sqrt) → normalize + optional affine
//! ((x_i−μ) · inv_std [· γ_i + β_i]).
//!
//! Native `fsqrt s, s` — no libm call. Leaf function (no FFI save/restore).
//!
//! Register plan (AAPCS64-safe; all scratch in caller-saved range):
//!   x6   = bound scratch (clobbered by every emit_imm32 call)
//!   x9   = x_in  (per-row input ptr;  recomputed at top of each row)
//!   x10  = x_out (per-row output ptr; recomputed at top of each row)
//!   x11  = x_j (inner counter)
//!   x12  = x_i (outer counter)
//!   x13  = x_gamma (γ base ptr — affine only)
//!   x14  = x_beta  (β base ptr — affine only)
//!   x16  = src_base (set ONCE via materialise_ptr; lives entire function)
//!   x17  = dst_base (set ONCE via materialise_ptr; lives entire function)
//!
//! Float registers (all in caller-saved s0–s7; s8–s15 intentionally avoided):
//!   s0   = s_acc (accumulator per-pass; reused as s_var at end of Pass 2)
//!   s1   = s_mean (live Pass 2 + Pass 3)
//!   s2   = s_inv_d (1/D constant; REUSED as s_b in Pass 3 affine — see below)
//!   s3   = s_eps (1e-5; live across outer batch loop)
//!   s4   = s_one (1.0;  live across outer batch loop)
//!   s5   = s_inv_std (held through Pass 3; Q4 constraint — NOT recomputed)
//!   s6   = s_t (per-element temp)
//!   s7   = s_g (γ_j load — affine only)
//!
//! s_b reuses s2 (s_inv_d) in Pass 3 affine body:
//!   After Pass 2, s_inv_d is consumed; s2 is dead. Pass 3 uses it for
//!   β_j loads, keeping the float register count within s0–s7. When
//!   affine=true, s2 is reloaded from inline constant at the end of each
//!   Pass 3 (top of next row's Pass 1 needs s_inv_d again).
//!
//! AAPCS64 callee-saved constraint — why s_b reuses s2, not s8:
//!   v8–v15 (= s8–s15) are callee-saved (lower 64 bits). Writing them in
//!   a leaf function without stp/ldp d8 save would silently corrupt the
//!   caller's v8–v15 — an LH-class bug in the float register file.
//!   Reusing dead s2 avoids both s8 usage and any stack manipulation.
//!
//! Constants (1/D, 1e-5, 1.0) are materialised ONCE before the outer loop
//! via movz/movk/fmov (inline; no .rodata pool needed). On the affine path,
//! s2 (s_inv_d) is reloaded at the end of Pass 3 with the same 3-instruction
//! sequence (cost: 3 instructions per row × B rows — negligible vs O(D) work).
//!
//! Note on x16/x17: AAPCS64 marks these "intra-procedure call scratch"
//! (IP0/IP1) — caller-saved, used by linker stubs across `bl` calls.
//! emit_layernorm has NO `bl` calls (leaf function), so x16/x17 are
//! effectively free for op use.
//!
//! M13 ABI register-conflict lesson generalized: all scratch lives in
//! non-INPUT_REGS scope from the start. LH-class bugs structurally
//! impossible in fresh emit_layernorm code at any N=1..4.

use crate::abi::AbiContext;
use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

/// Materialise a single f32 constant (given as its bit-pattern) into a
/// scalar float register, using a GPR scratch register as an intermediary.
/// Emits: movz <scratch_gpr>, #lo16; [movk ...]; fmov <freg>, <scratch_gpr_w>.
fn emit_f32_const(freg: &str, scratch_gpr_w: &str, bits: u32) -> String {
    let lo = (bits & 0xFFFF) as u16;
    let hi = ((bits >> 16) & 0xFFFF) as u16;
    let mut s = String::new();
    s.push_str(&format!("    movz    {}, #0x{:04x}\n", scratch_gpr_w, lo));
    if hi != 0 {
        s.push_str(&format!(
            "    movk    {}, #0x{:04x}, lsl #16\n",
            scratch_gpr_w, hi
        ));
    }
    s.push_str(&format!("    fmov    {}, {}\n", freg, scratch_gpr_w));
    s
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
        "γ and β offsets must be both Some or both None — order contract enforced upstream"
    );

    let mut s = String::new();
    s.push_str(&format!(
        "    ; layernorm (3-pass, {}affine): input [{b},{d}] -> output [{b},{d}]\n",
        if has_affine { "with " } else { "no-" },
    ));

    // Materialise base pointers ONCE before the outer loop. x16/x17 are
    // safe in this leaf function (no `bl` to trigger linker stubs).
    abi.materialise_ptr(src_loc, "x16", &mut s);
    abi.materialise_ptr(dst_loc, "x17", &mut s);

    // Affine: materialise γ and β base pointers from params_reg into x13/x14.
    // x6 used as transient byte-offset scratch for non-zero offsets.
    if has_affine {
        let g_off = gamma_offset.unwrap();
        let b_off = beta_offset.unwrap();
        let params_reg = abi.params_reg();
        if g_off == 0 {
            s.push_str(&format!("    mov     x13, {}\n", params_reg));
        } else {
            s.push_str(&emit_imm32("x6", g_off * 4));
            s.push_str(&format!("    add     x13, {}, x6\n", params_reg));
        }
        if b_off == 0 {
            s.push_str(&format!("    mov     x14, {}\n", params_reg));
        } else {
            s.push_str(&emit_imm32("x6", b_off * 4));
            s.push_str(&format!("    add     x14, {}, x6\n", params_reg));
        }
    }

    // Pre-loop hoisted constants — materialised ONCE before outer loop.
    // Layout:
    //   s2 = s_inv_d  = 1.0 / D  (compile-time constant)
    //   s3 = s_eps    = 1e-5
    //   s4 = s_one    = 1.0
    //
    // We use w9 as the scratch GPR for fmov (caller-saved, not used as
    // a live pointer or counter at this point). Note: emit_imm32 targets
    // x6 (x6 is the bound scratch); w9 is the fmov bridge for constants.
    //
    // s3 and s4 stay live through the entire outer loop (no affine clobber).
    // s2 (s_inv_d) is REUSED as s_b in Pass 3 affine — see below.
    let inv_d_bits = (1.0_f32 / d as f32).to_bits();
    let eps_bits = 1e-5_f32.to_bits();
    let one_bits = 1.0_f32.to_bits();
    s.push_str(&format!(
        "    ; constants: inv_d=0x{:08x}, eps=0x{:08x}, one=0x{:08x}\n",
        inv_d_bits, eps_bits, one_bits
    ));
    s.push_str(&emit_f32_const("s2", "w9", inv_d_bits)); // s_inv_d
    s.push_str(&emit_f32_const("s3", "w9", eps_bits)); // s_eps
    s.push_str(&emit_f32_const("s4", "w9", one_bits)); // s_one

    // === outer row loop: i in 0..b ===
    s.push_str("    mov     x12, #0\n");
    s.push_str(&format!(".Lln_row_{lid}:\n"));
    s.push_str(&emit_imm32("x6", b as usize));
    s.push_str("    cmp     x12, x6\n");
    s.push_str(&format!("    b.ge    .Lln_row_end_{lid}\n"));

    // Compute per-row pointers: x9 = src_base + i*d*4; x10 = dst_base + i*d*4.
    // x6 holds the byte stride temporarily (i * d * 4).
    s.push_str(&emit_imm32("x6", (d as usize) * 4));
    s.push_str("    mul     x6, x12, x6\n"); // x6 = i * d * 4 (byte offset)
    s.push_str("    add     x9,  x16, x6\n");
    s.push_str("    add     x10, x17, x6\n");

    // === Pass 1: μ = (1/D) · Σ x_j ===
    s.push_str("    fmov    s0, wzr\n"); // s_acc = 0
    s.push_str("    mov     x11, #0\n"); // x_j = 0
    s.push_str(&format!(".Lln_p1_{lid}:\n"));
    s.push_str(&emit_imm32("x6", d as usize));
    s.push_str("    cmp     x11, x6\n");
    s.push_str(&format!("    b.ge    .Lln_p1_end_{lid}\n"));
    s.push_str("    ldr     s6, [x9, x11, lsl #2]\n");
    s.push_str("    fadd    s0, s0, s6\n");
    s.push_str("    add     x11, x11, #1\n");
    s.push_str(&format!("    b       .Lln_p1_{lid}\n"));
    s.push_str(&format!(".Lln_p1_end_{lid}:\n"));
    s.push_str("    fmul    s1, s0, s2\n"); // s_mean = s_acc * s_inv_d

    // === Pass 2: σ² = (1/D) · Σ (x_j − μ)²; inv_std = 1/sqrt(σ² + ε) ===
    s.push_str("    fmov    s0, wzr\n"); // s_acc = 0
    s.push_str("    mov     x11, #0\n");
    s.push_str(&format!(".Lln_p2_{lid}:\n"));
    s.push_str(&emit_imm32("x6", d as usize));
    s.push_str("    cmp     x11, x6\n");
    s.push_str(&format!("    b.ge    .Lln_p2_end_{lid}\n"));
    s.push_str("    ldr     s6, [x9, x11, lsl #2]\n");
    s.push_str("    fsub    s6, s6, s1\n");
    s.push_str("    fmul    s6, s6, s6\n");
    s.push_str("    fadd    s0, s0, s6\n");
    s.push_str("    add     x11, x11, #1\n");
    s.push_str(&format!("    b       .Lln_p2_{lid}\n"));
    s.push_str(&format!(".Lln_p2_end_{lid}:\n"));
    // s_acc (s0) is now Σ (x_j − μ)² — compute variance + inv_std.
    s.push_str("    fmul    s0, s0, s2\n"); // s_var = s_acc * s_inv_d
    s.push_str("    fadd    s0, s0, s3\n"); // s_var += s_eps
    s.push_str("    fsqrt   s0, s0\n"); // s_var = sqrt(σ² + ε)  — native, no libm
    s.push_str("    fdiv    s5, s4, s0\n"); // s_inv_std = s_one / s_var
                                            //         ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Q4 constraint: ONE fdiv per row,
                                            //         OUTSIDE Pass 3. Pass 3 hot loop uses fmul s_t, s_t, s_inv_std.

    // === Pass 3: y_j = (x_j − μ) · inv_std [· γ_j + β_j] ===
    // Pass 3 inner loop must contain ZERO fdiv — inv_std is pre-computed above.
    s.push_str("    mov     x11, #0\n");
    s.push_str(&format!(".Lln_p3_{lid}:\n"));
    s.push_str(&emit_imm32("x6", d as usize));
    s.push_str("    cmp     x11, x6\n");
    s.push_str(&format!("    b.ge    .Lln_p3_end_{lid}\n"));
    s.push_str("    ldr     s6, [x9, x11, lsl #2]\n");
    s.push_str("    fsub    s6, s6, s1\n");
    s.push_str("    fmul    s6, s6, s5\n"); // s_t *= s_inv_std (NOT fdiv — Q4)
    if has_affine {
        s.push_str("    ldr     s7, [x13, x11, lsl #2]\n"); // γ_j → s_g
        s.push_str("    fmul    s6, s6, s7\n");
        s.push_str("    ldr     s2, [x14, x11, lsl #2]\n"); // β_j → s_b (REUSES s2/s_inv_d)
        s.push_str("    fadd    s6, s6, s2\n");
    }
    s.push_str("    str     s6, [x10, x11, lsl #2]\n");
    s.push_str("    add     x11, x11, #1\n");
    s.push_str(&format!("    b       .Lln_p3_{lid}\n"));
    s.push_str(&format!(".Lln_p3_end_{lid}:\n"));

    // Reload s_inv_d (s2) for next row — only when affine, because Pass 3
    // affine clobbered s2 with β_j loads. For no-affine, s2 still holds
    // the original constant (never touched by Pass 3). Reload uses the same
    // inline materialisation (3 instructions per row × B rows — negligible
    // vs O(D) per-row work).
    if has_affine {
        s.push_str(&emit_f32_const("s2", "w9", inv_d_bits));
    }

    s.push_str("    add     x12, x12, #1\n");
    s.push_str(&format!("    b       .Lln_row_{lid}\n"));
    s.push_str(&format!(".Lln_row_end_{lid}:\n"));

    Ok(s)
}
