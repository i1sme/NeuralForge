// SPDX-License-Identifier: Apache-2.0

//! Low-level AArch64 assembly building blocks.

use crate::buffer::RegSet;
use crate::FnSig;

pub const MACHO_SYM_PREFIX: &str = "_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafKind {
    Leaf,
    NonLeaf,
}

/// Format the function header (.globl + alignment + label) plus prologue.
///
/// Per spec §7.2:
/// - callee-saved FP regs (d8/d9) saved first if `regs.contains_d8_d9()`.
/// - x29/x30 saved iff non-leaf, and frame pointer set.
/// - sp adjusted for intermediate buffers (multiple of 16, may need
///   movz/movk + sub for sizes that don't fit in a 12-bit immediate).
pub fn format_function_prologue(
    sig: &FnSig,
    leaf: LeafKind,
    regs: RegSet,
    intermediate_bytes: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(".globl {}{}\n", MACHO_SYM_PREFIX, sig.name));
    s.push_str(".p2align 2\n");
    s.push_str(&format!("{}{}:\n", MACHO_SYM_PREFIX, sig.name));

    if regs.contains_x19_x23() {
        s.push_str("    stp     x19, x20, [sp, #-16]!\n");
        s.push_str("    stp     x21, x22, [sp, #-16]!\n");
        s.push_str("    str     x23, [sp, #-16]!\n");
    }
    if regs.contains_d8_d9() {
        s.push_str("    stp     d8, d9, [sp, #-16]!\n");
    }
    if leaf == LeafKind::NonLeaf {
        s.push_str("    stp     x29, x30, [sp, #-16]!\n");
        s.push_str("    mov     x29, sp\n");
    }
    if intermediate_bytes > 0 {
        s.push_str(&emit_sp_sub(intermediate_bytes));
    }
    s
}

/// Symmetric epilogue.
pub fn format_function_epilogue(leaf: LeafKind, regs: RegSet, intermediate_bytes: usize) -> String {
    let mut s = String::new();
    if intermediate_bytes > 0 {
        s.push_str(&emit_sp_add(intermediate_bytes));
    }
    if leaf == LeafKind::NonLeaf {
        s.push_str("    ldp     x29, x30, [sp], #16\n");
    }
    if regs.contains_d8_d9() {
        s.push_str("    ldp     d8, d9, [sp], #16\n");
    }
    if regs.contains_x19_x23() {
        s.push_str("    ldr     x23, [sp], #16\n");
        s.push_str("    ldp     x21, x22, [sp], #16\n");
        s.push_str("    ldp     x19, x20, [sp], #16\n");
    }
    s.push_str("    ret\n");
    s
}

/// Emit `sub sp, sp, #N` correctly for any 16-aligned N.
///
/// `sub` immediate is 12-bit (0..4095) optionally shifted by 12 (0..16,773,120
/// in steps of 4096). For sizes that don't fit, materialise N into x9 first.
pub fn emit_sp_sub(n_bytes: usize) -> String {
    assert!(
        n_bytes <= u32::MAX as usize,
        "frame > 4 GiB unsupported in M4b (got {} bytes)",
        n_bytes
    );
    if n_bytes <= 4095 {
        format!("    sub     sp, sp, #{}\n", n_bytes)
    } else if n_bytes <= 16_773_120 && n_bytes.is_multiple_of(4096) {
        format!("    sub     sp, sp, #{}, lsl #12\n", n_bytes / 4096)
    } else {
        let lo = (n_bytes & 0xFFFF) as u16;
        let hi = ((n_bytes >> 16) & 0xFFFF) as u16;
        let mut s = String::new();
        s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo));
        if hi != 0 {
            s.push_str(&format!("    movk    w9, #0x{:04x}, lsl #16\n", hi));
        }
        s.push_str("    sub     sp, sp, x9\n");
        s
    }
}

/// Materialise an arbitrary 32-bit unsigned value into a GPR using movz/movk.
///
/// AArch64 `mov Xn, #imm` only encodes values whose bit pattern fits a 16-bit
/// immediate (optionally shifted). For larger values (e.g. param offsets > 65535)
/// we must use movz + optional movk.
pub fn emit_imm32(reg: &str, value: usize) -> String {
    assert!(
        value <= u32::MAX as usize,
        "immediate > 32 bits unsupported here (got {value})"
    );
    let lo = (value & 0xFFFF) as u16;
    let hi = ((value >> 16) & 0xFFFF) as u16;
    let mut s = String::new();
    s.push_str(&format!("    movz    {}, #0x{:04x}\n", reg, lo));
    if hi != 0 {
        s.push_str(&format!("    movk    {}, #0x{:04x}, lsl #16\n", reg, hi));
    }
    s
}

/// Symmetric `add sp, sp, #N`.
pub fn emit_sp_add(n_bytes: usize) -> String {
    assert!(
        n_bytes <= u32::MAX as usize,
        "frame > 4 GiB unsupported in M4b (got {} bytes)",
        n_bytes
    );
    if n_bytes <= 4095 {
        format!("    add     sp, sp, #{}\n", n_bytes)
    } else if n_bytes <= 16_773_120 && n_bytes.is_multiple_of(4096) {
        format!("    add     sp, sp, #{}, lsl #12\n", n_bytes / 4096)
    } else {
        let lo = (n_bytes & 0xFFFF) as u16;
        let hi = ((n_bytes >> 16) & 0xFFFF) as u16;
        let mut s = String::new();
        s.push_str(&format!("    movz    w9, #0x{:04x}\n", lo));
        if hi != 0 {
            s.push_str(&format!("    movk    w9, #0x{:04x}, lsl #16\n", hi));
        }
        s.push_str("    add     sp, sp, x9\n");
        s
    }
}
