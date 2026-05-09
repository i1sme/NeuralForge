// SPDX-License-Identifier: Apache-2.0

//! SysV AMD64 argument-register abstraction for the multi-input model ABI.
//!
//! `INPUT_REGS` lists all 6 SysV AMD64 GP argument registers (`%rdi`,
//! `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`); M12 caps the number of model
//! inputs at N=4, so `N + 2 ≤ 6` (inputs + params + output) and the
//! arity is always register-only — no stack-spill required.
//!
//! `AbiContext` is constructed once per function at the top of
//! `walk_model` and threaded by `&abi` through every op-emitter. It is
//! the single point of truth about register layout — emitters never
//! hardcode `%rdi`/`%rsi`/`%rdx`. See spec
//! `docs/superpowers/specs/2026-05-09-m12-multi-input-abi-design.md`
//! §5.2.
//!
//! Stack-alignment around FFI calls is the second responsibility:
//! `emit_ffi_save` / `emit_ffi_restore` produce alignment-correct
//! `pushq`/`popq` blocks (with `pushq %rax` padding for odd
//! cardinality), in strict LIFO order. See spec §6.

use crate::buffer::BufferLoc;

/// Argument-register table. Order matches register-allocation order:
/// inputs in declaration order (indices `0..n_inputs`), then params at
/// `n_inputs`, then output at `n_inputs + 1`.
pub(crate) const INPUT_REGS: &[&str] = &["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"];

/// Per-function ABI state. Constructed once at the top of `walk_model`
/// and threaded by `&abi` through every op-emitter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AbiContext {
    pub n_inputs: usize,
}

impl AbiContext {
    /// Register holding the i-th input pointer (0-indexed within
    /// `model.inputs`). For N=3, `input_reg(2) == "%rdx"`.
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
    /// caller-saved set we spill across any FFI call (`call expf@PLT`
    /// today; generalises to any future external call).
    pub fn ffi_save_set(&self) -> &[&'static str] {
        &INPUT_REGS[..self.n_inputs + 2]
    }

    /// Emit a single instruction placing a buffer pointer into `dst_reg`.
    /// - `BufferLoc::InputReg(i)` → `movq %r_i, dst_reg`.
    /// - `BufferLoc::OutputReg`   → `movq %r_{n+1}, dst_reg`.
    /// - `BufferLoc::StackOffset(off)` → `leaq off(%rsp), dst_reg`
    ///   (or `movq %rsp, dst_reg` for offset 0).
    /// - `BufferLoc::Alias(_)` panics — caller must resolve aliases
    ///   before calling.
    pub fn materialise_ptr(&self, loc: BufferLoc, dst_reg: &str, asm: &mut String) {
        match loc {
            BufferLoc::InputReg(idx) => {
                asm.push_str(&format!(
                    "    movq    {}, {}\n",
                    self.input_reg(idx),
                    dst_reg
                ));
            }
            BufferLoc::OutputReg => {
                asm.push_str(&format!("    movq    {}, {}\n", self.output_reg(), dst_reg));
            }
            BufferLoc::StackOffset(off) => {
                assert!(
                    off <= i32::MAX as usize,
                    "stack offset > 2 GiB unsupported (got {} bytes)",
                    off
                );
                if off == 0 {
                    asm.push_str(&format!("    movq    %rsp, {}\n", dst_reg));
                } else {
                    asm.push_str(&format!("    leaq    {}(%rsp), {}\n", off, dst_reg));
                }
            }
            BufferLoc::Alias(_) => {
                panic!("AbiContext::materialise_ptr: BufferLoc::Alias must be resolved before call site")
            }
        }
    }

    /// Emit FFI-call save block — sequential `pushq` per register, with
    /// `pushq %rax` padding for odd cardinality to maintain 16-byte SP
    /// alignment at the call instruction. SP delta is always a
    /// multiple of 16. Per spec §6.1.
    ///
    /// Why padding: SysV AMD64 requires `(rsp + 8) % 16 == 0` entering
    /// the callee. Each `pushq` is 8 bytes; pushing an even total keeps
    /// rsp parity unchanged across the save block, so an odd number of
    /// register pushes needs one extra dummy push (`pushq %rax`) to
    /// restore the alignment expected at the call site.
    pub fn emit_ffi_save(&self, asm: &mut String) {
        let regs = self.ffi_save_set();
        for r in regs {
            asm.push_str(&format!("    pushq   {}\n", r));
        }
        if !regs.len().is_multiple_of(2) {
            asm.push_str("    pushq   %rax              # 16-byte alignment padding\n");
        }
    }

    /// Emit FFI-call restore block in strict LIFO order relative to
    /// `emit_ffi_save`. The asymmetry vs `emit_ffi_save`: pushes go in
    /// forward order, but pops must walk in reverse. The padding
    /// `pushq %rax` is the LAST instruction in the save block (when
    /// odd), so it's the FIRST `popq %rax` in the restore block.
    /// Per spec §6.3.
    pub fn emit_ffi_restore(&self, asm: &mut String) {
        // emit_ffi_save's order is "pushq r0..rN; (pushq %rax if odd)".
        // emit_ffi_restore must reverse exactly: "popq %rax (if odd);
        // popq rN..r0". Iterating forward like save would give the
        // wrong order — pops have to walk reverse, hence this dedicated
        // function shape rather than reusing the save loop.
        let regs = self.ffi_save_set();
        if !regs.len().is_multiple_of(2) {
            asm.push_str("    popq    %rax              # discard alignment padding\n");
        }
        for r in regs.iter().rev() {
            asm.push_str(&format!("    popq    {}\n", r));
        }
    }
}
