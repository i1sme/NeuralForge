// SPDX-License-Identifier: Apache-2.0

//! MulScalar codegen — x86_64 SSE2 AT&T-syntax flat per-element multiply.
//!
//! Scalar pre-loaded into %xmm4 once via:
//!   movl $<scalar_bits>, %r10d
//!   movd %r10d, %xmm4
//! Inner loop:
//!   movss (%r8, %rcx, 4), %xmm0
//!   mulss %xmm4, %xmm0
//!   movss %xmm0, (%r11, %rcx, 4)
//!
//! With `BufferLoc::Alias`, `src_loc == dst_loc` → in-place transformation
//! (the materialise_ptr resolution gives both registers the same value).
//!
//! f64-to-f32 truncation happens in the dispatcher (codegen.rs) — the
//! emitter receives `scalar_bits: u32` already in f32 form. See spec §6.5.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;

/// Emit AT&T x86_64 asm for `dst[i] = src[i] * scalar`.
#[allow(clippy::too_many_arguments)]
pub fn emit_mulscalar(
    abi: &AbiContext,
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
        "    # mul_scalar: total_elements={}, scalar_bits=0x{:08x}\n",
        total_elements, scalar_bits
    ));

    // Pre-load scalar into %xmm4 via %r10d (movl + movd).
    s.push_str(&format!("    movl    $0x{:x}, %r10d\n", scalar_bits));
    s.push_str("    movd    %r10d, %xmm4\n");

    abi.materialise_ptr(src_loc, "%r8", &mut s);
    abi.materialise_ptr(dst_loc, "%r11", &mut s);

    // Flat loop, %rcx = i.
    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lms_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(total_elements as u32));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lms_end_{mid}\n"));

    s.push_str("    movss   (%r8, %rcx, 4), %xmm0\n");
    s.push_str("    mulss   %xmm4, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rcx, 4)\n");

    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lms_{mid}\n"));
    s.push_str(&format!(".Lms_end_{mid}:\n"));

    s
}
