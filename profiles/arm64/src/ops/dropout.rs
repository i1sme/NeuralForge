// SPDX-License-Identifier: AGPL-3.0-only

//! Dropout codegen.
//!
//! At inference, dropout is identity. The buffer-assignment first-pass
//! (`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand)` for
//! dropout nodes that are NOT the model output; in that case no asm is
//! emitted (downstream ops read from the operand's buffer directly).
//!
//! When a dropout node IS `model.output`, however, `assign_buffers`
//! returns `BufferLoc::OutputReg` (the caller's `x2` pointer). In that
//! case the operand's buffer must be explicitly copied into the output
//! buffer, since alias-redirection no longer applies. `emit_dropout_copy`
//! emits the float-by-float copy loop for that path.

use crate::asm::emit_imm32;
use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

/// Emit AArch64 asm for a dropout-as-output copy loop.
///
/// Mirror of `emit_relu`'s structure minus the zero-init and `fmax`:
/// element-wise load → store, no transformation. Used only when a
/// `Dropout` node is the model's output (see module-level doc).
pub fn emit_dropout_copy(
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
        "    ; dropout-as-output: copy operand→output ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str(&emit_imm32("x10", total_floats as usize));
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Ldropout_{did}:\n"));
    s.push_str("    cmp     x9, x10\n");
    s.push_str(&format!("    b.ge    .Ldropout_end_{did}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Ldropout_{did}\n"));
    s.push_str(&format!(".Ldropout_end_{did}:\n"));
    s
}
