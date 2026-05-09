// SPDX-License-Identifier: Apache-2.0

//! Relu (elementwise max with zero) codegen — x86_64 SSE2.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for an elementwise ReLU.
///
/// `model_idx` + `relu_idx` together uniquely name every label across all
/// models emitted into a single assembly file (multi-model fixtures like
/// `pipeline_styles.nfl` would otherwise collide on `.Lrelu_0` etc.).
///
/// Register usage:
///   %rax (= src pointer)
///   %r11 (= dst pointer)
///   %r10d (= total_floats — written via emit_imm32_to_r10)
///   %rbp (= loop counter)
///   %xmm0 (= scratch float — element)
///   %xmm1 (= scratch float — zero)
pub fn emit_relu(
    abi: &AbiContext,
    total_floats: u64,
    model_idx: usize,
    relu_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let rid = format!("{model_idx}_{relu_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # relu: copy-clamp src→dst ({total_floats} elements)\n"
    ));
    abi.materialise_ptr(src_loc, "%rax", &mut s);
    abi.materialise_ptr(dst_loc, "%r11", &mut s);
    s.push_str("    xorps   %xmm1, %xmm1\n");
    s.push_str(&emit_imm32_to_r10(total_floats as u32));
    s.push_str("    xorq    %rbp, %rbp\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str("    cmpq    %r10, %rbp\n");
    s.push_str(&format!("    jge     .Lrelu_end_{rid}\n"));
    s.push_str("    movss   (%rax, %rbp, 4), %xmm0\n");
    s.push_str("    maxss   %xmm1, %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rbp, 4)\n");
    s.push_str("    incq    %rbp\n");
    s.push_str(&format!("    jmp     .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}
