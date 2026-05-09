// SPDX-License-Identifier: Apache-2.0

//! Dropout codegen.
//!
//! At inference, dropout is identity. Buffer assignment returns
//! `BufferLoc::Alias(operand)` for non-output dropouts (no asm emitted —
//! downstream ops read from the operand's buffer directly). When the
//! dropout is the model output, `walk_model` calls `emit_dropout_copy`
//! to copy the operand buffer into the caller's output buffer.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for a dropout-as-output copy loop.
///
/// Mirror of `emit_relu`'s structure minus the zero-init and `maxss`:
/// element-wise load → store, no transformation.
pub fn emit_dropout_copy(
    abi: &AbiContext,
    total_floats: u64,
    model_idx: usize,
    dropout_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    debug_assert!(
        matches!(dst_loc, BufferLoc::OutputReg),
        "emit_dropout_copy only valid for OutputReg dst (caller guards in walk_model)"
    );
    let did = format!("{model_idx}_{dropout_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # dropout-as-output: copy operand→output ({total_floats} elements)\n"
    ));
    abi.materialise_ptr(src_loc, "%rax", &mut s);
    abi.materialise_ptr(dst_loc, "%r11", &mut s);
    s.push_str(&emit_imm32_to_r10(total_floats as u32));
    s.push_str("    xorq    %rcx, %rcx\n");
    s.push_str(&format!(".Ldropout_{did}:\n"));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Ldropout_end_{did}\n"));
    s.push_str("    movss   (%rax, %rcx, 4), %xmm0\n");
    s.push_str("    movss   %xmm0, (%r11, %rcx, 4)\n");
    s.push_str("    incq    %rcx\n");
    s.push_str(&format!("    jmp     .Ldropout_{did}\n"));
    s.push_str(&format!(".Ldropout_end_{did}:\n"));
    s
}
