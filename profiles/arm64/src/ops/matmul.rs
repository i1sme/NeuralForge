// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — multi-dim matmul over rank ≥ 2 inputs with optional
//! `transpose_b`. Outer loop iterates over the product of leading dims
//! (`leading_count`); the inner kernel is a triple-loop FMA matmul over
//! the trailing `[M, K]` × `[K, N]` (or `[N, K]` if `transpose_b=true`)
//! pair.
//!
//! Spills `x1` (params) and `x2` (output) to the stack via `stp`/`ldp` —
//! the matmul body uses them as scratch for per-iter A_slice and B_slice
//! pointers, then restores both before the function continues. This
//! preserves the AArch64 FFI calling convention for downstream emitters
//! (e.g. a subsequent `emit_linear` reading `x1` for params, or any
//! emitter writing to `x2` for output).
//!
//! `emit_linear` (matmul + bias + post-ops) is unchanged; this module
//! is strictly additive per spec §6.1.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;
use compiler::ast::Span;
use profile_api::LowerError;

/// Emit AArch64 asm for a multi-dim matmul.
///
/// `leading_count` = product of leading dims (`shape[..rank-2].product()`).
/// For 2D inputs `leading_count == 1` — the outer loop runs once and is
/// effectively elided.
///
/// `m`, `k`, `n` are the trailing matrix dims. With `transpose_b=false`,
/// B is `[..., K, N]`; with `transpose_b=true`, B is `[..., N, K]`.
///
/// Base pointers `x11/x13/x12` (= A, B, DST) are materialised once
/// before the outer loop and MUST NOT be mutated inside it. Per-outer
/// slice base pointers go into `x1`, `x2`, `x4` (with `x1` and `x2`
/// stack-spilled around the outer loop to preserve the FFI param/output
/// pointers for downstream emitters).
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
    leading_count: u64,
    m: u64,
    k: u64,
    n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    _node_span: Span,
) -> Result<String, LowerError> {
    let mid = format!("{model_idx}_{matmul_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul (leading_count={}): [{},{}] x [{},{}] -> [{},{}], transpose_b={}\n",
        leading_count, m, k, k, n, m, n, transpose_b
    ));

    // Materialise base pointers ONCE — invariant across outer iterations.
    s.push_str(&materialise_ptr("x11", a_loc));
    s.push_str(&materialise_ptr("x13", b_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Inner-kernel slice sizes (in floats):
    //   A slice = M * K   (per outer iteration)
    //   B slice = K * N   (always — same regardless of transpose_b)
    //   DST slice = M * N (per outer iteration)
    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Spill x1 (params ptr) and x2 (output ptr) to the stack — emit_matmul
    // uses them as scratch for per-iter slice pointers. Restored at the end
    // so downstream emitters see the original FFI register state.
    s.push_str("    stp     x1, x2, [sp, #-16]!\n");

    // Outer loop: x17 = outer_idx (caller-saved scratch register).
    s.push_str("    mov     x17, #0\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32("x10", leading_count as usize));
    s.push_str("    cmp     x17, x10\n");
    s.push_str(&format!("    b.ge    .Lmm4d_outer_end_{mid}\n"));

    // Per-outer slice base pointers go into x1, x2, x4. Note: x1 and x2
    // are AArch64 FFI input regs (params ptr / output ptr); we spilled
    // them via `stp` above, so reusing them as scratch here is safe — the
    // matching `ldp` after the outer loop restores both before any
    // downstream emitter runs. We use x4 (rather than x3) for DST_slice to
    // avoid colliding with the existing meaning of x3 in emit_linear's
    // i-loop.
    // x1 = A_slice = x11 + x17 * a_slice * 4
    s.push_str(&emit_imm32("x8", a_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x1, x11, x6, lsl #2\n");
    // x2 = B_slice = x13 + x17 * b_slice * 4
    s.push_str(&emit_imm32("x8", b_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x2, x13, x6, lsl #2\n");
    // x4 = DST_slice = x12 + x17 * dst_slice * 4
    s.push_str(&emit_imm32("x8", dst_slice));
    s.push_str("    mul     x6, x17, x8\n");
    s.push_str("    add     x4, x12, x6, lsl #2\n");

    // Hoist trailing-dim bounds.
    s.push_str(&emit_imm32("x10", m as usize)); // x10 = M (re-used; was leading_count above)
    s.push_str(&emit_imm32("x15", n as usize)); // x15 = N
    s.push_str(&emit_imm32("x16", k as usize)); // x16 = K

    // Inner i-loop (rows of output, [0, M)).
    // x5 = i.
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str("    cmp     x5, x10\n");
    s.push_str(&format!("    b.ge    .Lmm4d_i_end_{mid}\n"));

    // Inner j-loop (cols of output, [0, N)).
    // x7 = j.
    s.push_str("    mov     x7, #0\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str("    cmp     x7, x15\n");
    s.push_str(&format!("    b.ge    .Lmm4d_j_end_{mid}\n"));

    // Accumulator s0 = 0.0.
    s.push_str("    fmov    s0, wzr\n");
    // Inner k-loop (contraction, [0, K)).
    // x9 = k_inner.
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str("    cmp     x9, x16\n");
    s.push_str(&format!("    b.ge    .Lmm4d_k_end_{mid}\n"));

    // a_offset = i * K + k_inner   (always — A is always [..., M, K])
    s.push_str("    mul     x6, x5, x16\n");
    s.push_str("    add     x6, x6, x9\n");
    s.push_str("    ldr     s1, [x1, x6, lsl #2]\n");

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str("    mul     x6, x7, x16\n");
        s.push_str("    add     x6, x6, x9\n");
    } else {
        s.push_str("    mul     x6, x9, x15\n");
        s.push_str("    add     x6, x6, x7\n");
    }
    s.push_str("    ldr     s2, [x2, x6, lsl #2]\n");

    // Fused multiply-add: s0 = s0 + s1 * s2.
    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store s0 → DST_slice[i * N + j].
    s.push_str("    mul     x6, x5, x15\n");
    s.push_str("    add     x6, x6, x7\n");
    s.push_str("    str     s0, [x4, x6, lsl #2]\n");

    // j++; j-loop tail.
    s.push_str("    add     x7, x7, #1\n");
    s.push_str(&format!("    b       .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    // i++; i-loop tail.
    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    // Outer++; outer-loop tail.
    s.push_str("    add     x17, x17, #1\n");
    s.push_str(&format!("    b       .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    // Restore x1 (params ptr) and x2 (output ptr) — must match the
    // `stp ..., [sp, #-16]!` above so the AArch64 FFI register state is
    // preserved for downstream emitters.
    s.push_str("    ldp     x1, x2, [sp], #16\n");

    Ok(s)
}
