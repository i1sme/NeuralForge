// SPDX-License-Identifier: Apache-2.0

//! AAPCS argument-register abstraction for the multi-input model ABI.
//!
//! `INPUT_REGS` lists the first 6 of the 8 AAPCS argument registers
//! (`x0`..`x7`); M12 caps the number of model inputs at N=4, so
//! `N + 2 ≤ 6` (inputs + params + output). The reserved `x6`/`x7` slots
//! leave room for future ABI extensions without re-flowing this table.
//!
//! `AbiContext` is constructed once per function at the top of
//! `walk_model` and threaded by `&abi` through every op-emitter. It is
//! the single point of truth about register layout — emitters never
//! hardcode `x0`/`x1`/`x2`. See spec `docs/superpowers/specs/2026-05-09-m12-multi-input-abi-design.md` §5.
//!
//! Stack-alignment around FFI calls is the second responsibility:
//! `emit_ffi_save` / `emit_ffi_restore` produce alignment-correct
//! `stp`/`ldp` blocks (with `xzr` padding for odd cardinality), in
//! strict LIFO order. See spec §6.

use crate::buffer::BufferLoc;

/// Argument-register table. Order matches register-allocation order:
/// inputs in declaration order (indices `0..n_inputs`), then params at
/// `n_inputs`, then output at `n_inputs + 1`.
pub(crate) const INPUT_REGS: &[&str] = &["x0", "x1", "x2", "x3", "x4", "x5"];

/// Per-function ABI state. Constructed once at the top of `walk_model`
/// and threaded by `&abi` through every op-emitter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AbiContext {
    pub n_inputs: usize,
}

impl AbiContext {
    /// Register holding the i-th input pointer (0-indexed within
    /// `model.inputs`). For N=3, `input_reg(2) == "x2"`.
    pub fn input_reg(&self, idx: usize) -> &'static str {
        INPUT_REGS[idx]
    }

    /// Register holding the params pointer. Always `INPUT_REGS[n_inputs]`.
    pub fn params_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs]
    }

    /// Register holding the output pointer. Always `INPUT_REGS[n_inputs + 1]`.
    pub fn output_reg(&self) -> &'static str {
        INPUT_REGS[self.n_inputs + 1]
    }

    /// All ABI-argument registers in use by this function. Equals
    /// `INPUT_REGS[..n_inputs + 2]`. This is the conservative
    /// caller-saved set we spill across any FFI call (`bl _expf` today;
    /// generalises to any future external call).
    pub fn ffi_save_set(&self) -> &[&'static str] {
        &INPUT_REGS[..self.n_inputs + 2]
    }

    /// Emit a single instruction placing a buffer pointer into `dst_reg`.
    /// - `BufferLoc::InputReg(i)` → `mov dst_reg, x_i`.
    /// - `BufferLoc::OutputReg`   → `mov dst_reg, x_{n+1}`.
    /// - `BufferLoc::StackOffset(off)` → `add dst_reg, sp, #off` (with
    ///   appropriate large-immediate decomposition for offsets that
    ///   cannot be encoded in a single `add`).
    /// - `BufferLoc::Alias(_)` panics — caller must resolve aliases
    ///   before calling.
    pub fn materialise_ptr(&self, loc: BufferLoc, dst_reg: &str, asm: &mut String) {
        match loc {
            BufferLoc::InputReg(idx) => {
                asm.push_str(&format!(
                    "    mov     {}, {}\n",
                    dst_reg,
                    self.input_reg(idx)
                ));
            }
            BufferLoc::OutputReg => {
                asm.push_str(&format!("    mov     {}, {}\n", dst_reg, self.output_reg()));
            }
            BufferLoc::StackOffset(off) => {
                assert!(
                    off <= u32::MAX as usize,
                    "stack offset > 4 GiB unsupported in M4b (got {} bytes)",
                    off
                );
                if off == 0 {
                    asm.push_str(&format!("    mov     {}, sp\n", dst_reg));
                } else if off <= 4095 {
                    asm.push_str(&format!("    add     {}, sp, #{}\n", dst_reg, off));
                } else if off <= 16_773_120 && off.is_multiple_of(4096) {
                    asm.push_str(&format!(
                        "    add     {}, sp, #{}, lsl #12\n",
                        dst_reg,
                        off / 4096
                    ));
                } else {
                    let lo = (off & 0xFFFF) as u16;
                    let hi = ((off >> 16) & 0xFFFF) as u16;
                    asm.push_str(&format!("    movz    w10, #0x{:04x}\n", lo));
                    if hi != 0 {
                        asm.push_str(&format!("    movk    w10, #0x{:04x}, lsl #16\n", hi));
                    }
                    asm.push_str(&format!("    add     {}, sp, x10\n", dst_reg));
                }
            }
            BufferLoc::Alias(_) => {
                panic!("AbiContext::materialise_ptr: BufferLoc::Alias must be resolved before call site")
            }
        }
    }

    /// Emit FFI-call save block — paired `stp` instructions; pads odd
    /// cardinality with `xzr` to maintain 16-byte SP alignment. SP delta
    /// is always a positive multiple of 16. Per spec §6.1.
    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let mut i = 0;
        while i < regs.len() {
            let a = regs[i];
            let b = if i + 1 < regs.len() {
                regs[i + 1]
            } else {
                "xzr"
            };
            asm.push_str(&format!("    stp     {}, {}, [sp, #-16]!\n", a, b));
            i += 2;
        }
    }

    /// Emit FFI-call restore block — pairs walked in strict LIFO order
    /// (reverse of `emit_ffi_save`). The `xzr`-padded slot round-trips
    /// harmlessly: `xzr` is the zero register, so `ldp ..., xzr, ...`
    /// is a write-discard. Per spec §6.3.
    pub fn emit_ffi_restore(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        let n = regs.len();
        let mut pairs: Vec<(&str, &str)> = Vec::with_capacity(n.div_ceil(2));
        let mut i = 0;
        while i < n {
            let a = regs[i];
            let b = if i + 1 < n { regs[i + 1] } else { "xzr" };
            pairs.push((a, b));
            i += 2;
        }
        for (a, b) in pairs.iter().rev() {
            asm.push_str(&format!("    ldp     {}, {}, [sp], #16\n", a, b));
        }
    }
}
