//! Softmax (per-row stable, libm expf) codegen.

use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// Uses `bl _expf` (libm). State in callee-saved s8 (per-row max), s9
/// (per-row sum). The function-level prologue (Task 3) handles d8/d9 save
/// and frame setup based on `compute_callee_saved` + `compute_is_leaf`.
pub fn emit_softmax(
    b: u64,
    k: u64,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let sid = softmax_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"
    ));

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Outer per-row loop: x3 = i.
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lsm_i_end_{sid}\n"));

    // Compute row base offsets in elements: x4 = i * k.
    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x4, x3, x8\n");

    // Pass 1: max into s8. Initialise to -inf.
    s.push_str("    movz    w0, #0x0000\n");
    s.push_str("    movk    w0, #0xFF80, lsl #16\n");
    s.push_str("    fmov    s8, w0\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_max_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");
    s.push_str("    fmax    s8, s8, s1\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));

    // Pass 2: exp(x - max) -> output, accumulate sum into s9.
    s.push_str("    fmov    s9, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_exp_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s0, [x11, x6, lsl #2]\n");
    s.push_str("    fsub    s0, s0, s8\n");
    s.push_str("    bl      _expf\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");
    s.push_str("    fadd    s9, s9, s0\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Pass 3: normalize.
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lsm_norm_end_{sid}\n"));
    s.push_str("    add     x6, x4, x5\n");
    s.push_str("    ldr     s0, [x12, x6, lsl #2]\n");
    s.push_str("    fdiv    s0, s0, s9\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    // Next row.
    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    s
}
