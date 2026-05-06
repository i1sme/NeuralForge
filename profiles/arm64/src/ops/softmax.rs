// SPDX-License-Identifier: Apache-2.0

//! Softmax (per-row stable, libm expf) codegen.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// `model_idx` and `softmax_idx` together uniquely name every label across
/// all models emitted into a single assembly file (multi-model fixtures like
/// `pipeline_styles.nfl` would otherwise collide on `.Lsm_i_0` etc.).
///
/// Uses `bl _expf` (libm). State across the call is held in callee-saved
/// registers so that `_expf` cannot clobber it:
///   x19 = i (outer row counter)
///   x20 = row base = i * k (element index)
///   x21 = j (inner column counter)
///   x22 = src base pointer
///   x23 = dst base pointer
/// x6 (element offset = x20 + x21) is recomputed each iteration — it is
/// scratch and need not survive the call.
///
/// Callee-saved s8 (per-row max) and s9 (per-row sum) are handled by the
/// function-level prologue via `compute_callee_saved` / `d8_d9` in RegSet.
/// The function-level prologue also saves x19-x23 when `x19_x23` is set.
pub fn emit_softmax(
    b: u64,
    k: u64,
    model_idx: usize,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let sid = format!("{model_idx}_{softmax_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"
    ));

    // Materialise src/dst into callee-saved x22/x23 so they survive bl _expf.
    s.push_str(&materialise_ptr("x22", src_loc));
    s.push_str(&materialise_ptr("x23", dst_loc));

    // Outer per-row loop: x19 = i.
    s.push_str("    mov     x19, #0\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&emit_imm32("x10", b as usize));
    s.push_str("    cmp     x19, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_i_end_{sid}\n"));

    // Compute row base offsets in elements: x20 = i * k.
    s.push_str(&emit_imm32("x8", k as usize));
    s.push_str("    mul     x20, x19, x8\n");

    // Pass 1: max into s8. Initialise to -inf.
    s.push_str("    movz    w0, #0x0000\n");
    s.push_str("    movk    w0, #0xFF80, lsl #16\n");
    s.push_str("    fmov    s8, w0\n");
    s.push_str("    mov     x21, #0\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&emit_imm32("x10", k as usize));
    s.push_str("    cmp     x21, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_max_end_{sid}\n"));
    s.push_str("    add     x6, x20, x21\n");
    s.push_str("    ldr     s1, [x22, x6, lsl #2]\n");
    s.push_str("    fmax    s8, s8, s1\n");
    s.push_str("    add     x21, x21, #1\n");
    s.push_str(&format!("    b       .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));

    // Pass 2: exp(x - max) -> output, accumulate sum into s9.
    // All live state (x19, x20, x21, x22, x23, s8, s9) is in callee-saved
    // registers, so bl _expf cannot clobber it per AAPCS64.
    s.push_str("    fmov    s9, wzr\n");
    s.push_str("    mov     x21, #0\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&emit_imm32("x10", k as usize));
    s.push_str("    cmp     x21, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_exp_end_{sid}\n"));
    s.push_str("    add     x6, x20, x21\n");
    s.push_str("    ldr     s0, [x22, x6, lsl #2]\n");
    s.push_str("    fsub    s0, s0, s8\n");
    s.push_str("    bl      _expf\n");
    // x6 must be recomputed: bl _expf may have clobbered it (caller-saved).
    s.push_str("    add     x6, x20, x21\n");
    s.push_str("    str     s0, [x23, x6, lsl #2]\n");
    s.push_str("    fadd    s9, s9, s0\n");
    s.push_str("    add     x21, x21, #1\n");
    s.push_str(&format!("    b       .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Pass 3: normalize.
    s.push_str("    mov     x21, #0\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&emit_imm32("x10", k as usize));
    s.push_str("    cmp     x21, x10\n");
    s.push_str(&format!("    b.ge    .Lsm_norm_end_{sid}\n"));
    s.push_str("    add     x6, x20, x21\n");
    s.push_str("    ldr     s0, [x23, x6, lsl #2]\n");
    s.push_str("    fdiv    s0, s0, s9\n");
    s.push_str("    str     s0, [x23, x6, lsl #2]\n");
    s.push_str("    add     x21, x21, #1\n");
    s.push_str(&format!("    b       .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    // Next row.
    s.push_str("    add     x19, x19, #1\n");
    s.push_str(&format!("    b       .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    s
}
