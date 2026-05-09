// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — multi-dim matmul over rank ≥ 2 inputs with optional
//! `transpose_b`. Outer loop iterates over the product of leading dims
//! (`leading_count`); the inner kernel is a triple-loop FMA matmul over
//! the trailing `[M, K]` × `[K, N]` (or `[N, K]` if `transpose_b=true`)
//! pair.
//!
//! M12 (multi-input ABI) rework — spec §9.1: matmul body must NOT
//! touch any ABI argument register (`x0`..`x_{N+1}` for an N-input
//! function), since `x_0`..`x_{N-1}` hold input pointers, `x_N` holds
//! the params pointer, and `x_{N+1}` holds the output pointer — all
//! read by other emitters before/after. The pre-M12 emitter spilled
//! `x1`/`x2` (params/output for N=1) to the stack with `stp`/`ldp`
//! around the outer loop and reused them as scratch — that pattern
//! breaks for N≥2 where `x1` holds a SECOND input pointer the inner
//! loop must read.
//!
//! New register layout — entirely within non-ABI caller-saved scratch
//! (`x6`..`x17` minus reserved `x18`):
//!
//! | Role                    | Reg | Lifetime           |
//! |-------------------------|-----|--------------------|
//! | A base ptr              | x9  | full function      |
//! | B base ptr              | x10 | full function      |
//! | DST base ptr            | x11 | full function      |
//! | A_slice ptr             | x12 | per outer iter     |
//! | B_slice ptr             | x13 | per outer iter     |
//! | DST_slice ptr           | x14 | per outer iter     |
//! | outer counter           | x15 | full function      |
//! | i counter               | x16 | per outer iter     |
//! | j counter               | x17 | per i iter         |
//! | k_inner counter         | x7  | per j iter         |
//! | addr-arith scratch      | x6  | per k iter         |
//! | bound/stride emit temp  | x8  | per cmp / per use  |
//!
//! Bounds (M, N, K, leading_count) and strides (K, N) are emitted
//! inline at each cmp / address-arithmetic site rather than hoisted
//! into dedicated bound registers. This trades a few extra
//! `emit_imm32` calls for register-pressure relief — necessary because
//! reserving `x0`..`x_{N+1}` for downstream-readable ABI registers
//! limits scratch to `x6`..`x17`, which is just enough for the 6 base/
//! slice ptrs + 4 counters + 2 scratches.
//!
//! The pre-M12 outer-loop `stp x1, x2, [sp, #-16]!` spill block is
//! REMOVED. emit_matmul body now contains zero `stp` instructions
//! (matmul does not call FFI; only `AbiContext::emit_ffi_save` emits
//! stack manipulation). Verified by `emit_matmul_body_contains_zero_stp`
//! unit test.

use crate::abi::AbiContext;
use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
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
/// Base pointers are loaded into `x9`/`x10`/`x11` (= A, B, DST)
/// once via `abi.materialise_ptr` before the outer loop. The inner
/// kernel uses only `x6`..`x17` scratch — no ABI argument register
/// is ever touched by the matmul body, so `x0`..`x_{N+1}` survive
/// across the call site for downstream emitters.
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
    abi: &AbiContext,
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
    // x9 = A base, x10 = B base, x11 = DST base. All non-ABI scratch.
    abi.materialise_ptr(a_loc, "x9", &mut s);
    abi.materialise_ptr(b_loc, "x10", &mut s);
    abi.materialise_ptr(dst_loc, "x11", &mut s);

    // Inner-kernel slice sizes (in floats):
    //   A slice = M * K   (per outer iteration)
    //   B slice = K * N   (always — same regardless of transpose_b)
    //   DST slice = M * N (per outer iteration)
    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Outer loop: x15 = outer_idx (caller-saved scratch, never in INPUT_REGS).
    s.push_str("    mov     x15, #0\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32("x8", leading_count as usize));
    s.push_str("    cmp     x15, x8\n");
    s.push_str(&format!("    b.ge    .Lmm4d_outer_end_{mid}\n"));

    // Per-outer slice base pointers go into x12, x13, x14. All non-ABI
    // scratch — no stack spill needed (M11's `stp x1, x2, [sp, #-16]!`
    // block is gone). x12 = A_slice = x9 + x15 * a_slice * 4.
    s.push_str(&emit_imm32("x8", a_slice));
    s.push_str("    mul     x6, x15, x8\n");
    s.push_str("    add     x12, x9, x6, lsl #2\n");
    // x13 = B_slice = x10 + x15 * b_slice * 4.
    s.push_str(&emit_imm32("x8", b_slice));
    s.push_str("    mul     x6, x15, x8\n");
    s.push_str("    add     x13, x10, x6, lsl #2\n");
    // x14 = DST_slice = x11 + x15 * dst_slice * 4.
    s.push_str(&emit_imm32("x8", dst_slice));
    s.push_str("    mul     x6, x15, x8\n");
    s.push_str("    add     x14, x11, x6, lsl #2\n");

    // Inner i-loop (rows of output, [0, M)).
    // x16 = i counter. Bound M is emitted inline at cmp.
    s.push_str("    mov     x16, #0\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str(&emit_imm32("x8", m as usize));
    s.push_str("    cmp     x16, x8\n");
    s.push_str(&format!("    b.ge    .Lmm4d_i_end_{mid}\n"));

    // Inner j-loop (cols of output, [0, N)).
    // x17 = j counter. Bound N is emitted inline at cmp.
    s.push_str("    mov     x17, #0\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str(&emit_imm32("x8", n as usize));
    s.push_str("    cmp     x17, x8\n");
    s.push_str(&format!("    b.ge    .Lmm4d_j_end_{mid}\n"));

    // Accumulator s0 = 0.0.
    s.push_str("    fmov    s0, wzr\n");
    // Inner k-loop (contraction, [0, K)).
    // x7 = k_inner counter. Bound K is emitted inline at cmp.
    s.push_str("    mov     x7, #0\n");
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str(&emit_imm32("x8", k as usize));
    s.push_str("    cmp     x7, x8\n");
    s.push_str(&format!("    b.ge    .Lmm4d_k_end_{mid}\n"));

    // a_offset = i * K + k_inner   (always — A is always [..., M, K]).
    // x8 = K stride; x6 = scratch offset.
    s.push_str(&emit_imm32("x8", k as usize));
    s.push_str("    mul     x6, x16, x8\n");
    s.push_str("    add     x6, x6, x7\n");
    s.push_str("    ldr     s1, [x12, x6, lsl #2]\n");

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str(&emit_imm32("x8", k as usize));
        s.push_str("    mul     x6, x17, x8\n");
        s.push_str("    add     x6, x6, x7\n");
    } else {
        s.push_str(&emit_imm32("x8", n as usize));
        s.push_str("    mul     x6, x7, x8\n");
        s.push_str("    add     x6, x6, x17\n");
    }
    s.push_str("    ldr     s2, [x13, x6, lsl #2]\n");

    // Fused multiply-add: s0 = s0 + s1 * s2.
    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x7, x7, #1\n");
    s.push_str(&format!("    b       .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store s0 → DST_slice[i * N + j].
    s.push_str(&emit_imm32("x8", n as usize));
    s.push_str("    mul     x6, x16, x8\n");
    s.push_str("    add     x6, x6, x17\n");
    s.push_str("    str     s0, [x14, x6, lsl #2]\n");

    // j++; j-loop tail.
    s.push_str("    add     x17, x17, #1\n");
    s.push_str(&format!("    b       .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    // i++; i-loop tail.
    s.push_str("    add     x16, x16, #1\n");
    s.push_str(&format!("    b       .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    // Outer++; outer-loop tail. No stack restore — the M11 `ldp x1, x2`
    // block is gone (matmul body never spilled in M12).
    s.push_str("    add     x15, x15, #1\n");
    s.push_str(&format!("    b       .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    Ok(s)
}
