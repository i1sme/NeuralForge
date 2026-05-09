// SPDX-License-Identifier: Apache-2.0

//! Add codegen — flat per-element tensor addition: dst[i] = a[i] + other[i].
//!
//! Closest existing template: `emit_mulscalar` (M10). Same flat loop
//! shell; differs in that `add` reads two input pointers (a, other)
//! instead of one input + a pre-loaded scalar.
//!
//! No FFI save/restore (no `bl _expf` call). No callee-saved register
//! usage. Three caller-saved scratch GPRs (x9/x10/x11) for the three
//! materialised pointers, x12 for the loop counter, x13 for the
//! immediate bound — all non-ABI on AAPCS64 for any N ≤ 4.
//!
//! M13 — first A2 brick (residual connections).

use crate::abi::AbiContext;
use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;

/// Emit AArch64 asm for `dst[i] = a[i] + other[i]` over `total_elements`.
#[allow(clippy::too_many_arguments)]
pub fn emit_add(
    abi: &AbiContext,
    total_elements: u64,
    model_idx: usize,
    op_idx: usize,
    a_loc: BufferLoc,
    other_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let mid = format!("{model_idx}_{op_idx}");
    let mut s = String::new();
    s.push_str(&format!("    ; add: total_elements={}\n", total_elements));

    // Materialise three pointers. x9 = a, x10 = other, x11 = dst.
    abi.materialise_ptr(a_loc, "x9", &mut s);
    abi.materialise_ptr(other_loc, "x10", &mut s);
    abi.materialise_ptr(dst_loc, "x11", &mut s);

    // Counter x12 = 0; bound x13 = total_elements (hoisted outside loop).
    s.push_str("    mov     x12, #0\n");
    s.push_str(&emit_imm32("x13", total_elements as usize));

    s.push_str(&format!(".Ladd_{mid}:\n"));
    s.push_str("    cmp     x12, x13\n");
    s.push_str(&format!("    b.ge    .Ladd_end_{mid}\n"));

    s.push_str("    ldr     s0, [x9, x12, lsl #2]\n");
    s.push_str("    ldr     s1, [x10, x12, lsl #2]\n");
    s.push_str("    fadd    s2, s0, s1\n");
    s.push_str("    str     s2, [x11, x12, lsl #2]\n");

    s.push_str("    add     x12, x12, #1\n");
    s.push_str(&format!("    b       .Ladd_{mid}\n"));
    s.push_str(&format!(".Ladd_end_{mid}:\n"));

    s
}
