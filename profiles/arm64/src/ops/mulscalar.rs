// SPDX-License-Identifier: Apache-2.0

//! MulScalar codegen — flat per-element multiply by a scalar.
//!
//! Scalar is pre-loaded into `s4` once before the loop via `movz/movk`
//! → `fmov`. The loop is `total_elements` iterations of:
//!   ldr s0, [src, idx, lsl #2]
//!   fmul s0, s0, s4
//!   str s0, [dst, idx, lsl #2]
//!
//! With `BufferLoc::Alias`, `src_loc == dst_loc` → in-place transformation
//! (the materialise_ptr resolution gives both registers the same value).
//!
//! f64-to-f32 truncation happens in the dispatcher (codegen.rs) — the
//! emitter receives `scalar_bits: u32` already in f32 form. See spec §6.5.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for `dst[i] = src[i] * scalar` over `total_elements`.
#[allow(clippy::too_many_arguments)]
pub fn emit_mulscalar(
    total_elements: u64,
    scalar_bits: u32,
    model_idx: usize,
    op_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    ; mul_scalar: total_elements={}, scalar_bits=0x{:08x}\n",
        total_elements, scalar_bits
    ));

    // Pre-load the scalar into s4. Decompose the u32 into hi16/lo16.
    let lo16 = (scalar_bits & 0xFFFF) as u16;
    let hi16 = ((scalar_bits >> 16) & 0xFFFF) as u16;
    s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo16));
    if hi16 != 0 {
        s.push_str(&format!("    movk    w9, #0x{:04x}, lsl #16\n", hi16));
    }
    s.push_str("    fmov    s4, w9\n");

    // Materialise base pointers. With Alias, both resolve to the same.
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));

    // Flat loop: x3 = i.
    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lms_{mid}:\n"));
    s.push_str(&emit_imm32("x10", total_elements as usize));
    s.push_str("    cmp     x3, x10\n");
    s.push_str(&format!("    b.ge    .Lms_end_{mid}\n"));

    s.push_str("    ldr     s0, [x11, x3, lsl #2]\n");
    s.push_str("    fmul    s0, s0, s4\n");
    s.push_str("    str     s0, [x12, x3, lsl #2]\n");

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lms_{mid}\n"));
    s.push_str(&format!(".Lms_end_{mid}:\n"));

    s
}
