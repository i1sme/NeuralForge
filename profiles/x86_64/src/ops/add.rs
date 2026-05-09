// SPDX-License-Identifier: Apache-2.0

//! Add codegen — x86_64 SSE2 AT&T-syntax flat per-element tensor addition.
//!
//!   dst[i] = a[i] + other[i]
//!
//! Closest existing template: `emit_mulscalar` (M10). Same flat loop
//! shell; differs in that `add` reads two input pointers (a, other)
//! instead of one input + a pre-loaded scalar, and uses `addss` instead
//! of `mulss`.
//!
//! ## Register budget (M13 spec §5.4)
//!
//! The intersection of free non-ABI scratch GPRs across N ∈ [1,4] is
//! {%rax, %r10, %r11} — exactly enough for the three materialised
//! pointers, leaving zero spare GPR for a counter. Plan-synthesis pick:
//! use `%rbp` as the counter. `%rbp` is callee-saved by the unconditional
//! `pushq %rbp` in `asm.rs::format_function_prologue`, and is read by
//! zero op-emitter bodies (verified at M13 plan synthesis). Inside the
//! function body, `%rbp` is wide-open scratch.
//!
//! This is the same trick used by M13 Group A (Task 1) for the matmul
//! j-counter at N=4. Both choices share the rationale: the prologue
//! already saves `%rbp`, no per-op save/restore needed.
//!
//! No FFI save/restore (no `call expf@PLT`). No additional callee-saved
//! register usage beyond `%rbp` (already saved).

use crate::abi::AbiContext;
use crate::buffer::BufferLoc;

/// Emit AT&T x86_64 asm for `dst[i] = a[i] + other[i]` over `total_elements`.
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
    s.push_str(&format!("    # add: total_elements={}\n", total_elements));

    // Materialise three pointers. %rax = a, %r10 = other, %r11 = dst.
    abi.materialise_ptr(a_loc, "%rax", &mut s);
    abi.materialise_ptr(other_loc, "%r10", &mut s);
    abi.materialise_ptr(dst_loc, "%r11", &mut s);

    // Loop counter %rbp = 0. (%rbp is callee-saved by the function-level
    // prologue; the body is free to clobber it.)
    s.push_str("    movq    $0, %rbp\n");
    s.push_str(&format!(".Ladd_{mid}:\n"));
    // cmpq with sign-extended 32-bit immediate. total_elements fits in
    // i32 for any practical NN size (max ~2^31 elements = ~8 GiB tensor).
    s.push_str(&format!("    cmpq    ${}, %rbp\n", total_elements));
    s.push_str(&format!("    jge     .Ladd_end_{mid}\n"));

    s.push_str("    movss   (%rax, %rbp, 4), %xmm0\n");
    s.push_str("    movss   (%r10, %rbp, 4), %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rbp, 4)\n");

    s.push_str("    addq    $1, %rbp\n");
    s.push_str(&format!("    jmp     .Ladd_{mid}\n"));
    s.push_str(&format!(".Ladd_end_{mid}:\n"));

    s
}
