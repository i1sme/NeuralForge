// SPDX-License-Identifier: Apache-2.0

//! Low-level x86_64 assembly building blocks (AT&T syntax).

use crate::buffer::RegSet;
use profile_api::FnSig;

/// SysV AMD64 stack frame size, including the alignment correction
/// dictated by the prologue's `push` count.
///
/// Derivation (spec §7.5): on function entry, the caller's `call`
/// instruction has just pushed the 8-byte return address, so
/// `rsp ≡ 8 (mod 16)`. Each prologue `push reg` (8 bytes) flips parity.
/// After N pushes, `rsp ≡ 8 - 8*N (mod 16) ≡ 8*(1 - N) (mod 16)`.
/// To land on `rsp ≡ 0 (mod 16)` after `sub rsp, frame_size`, the
/// helper adds an 8-byte correction when N is **even** (post-pushes
/// parity is 8), zero correction when N is **odd**.
pub fn compute_frame_size(raw_buffer_size: u32, num_pushes: usize) -> u32 {
    let aligned = (raw_buffer_size + 15) & !15;
    let push_correction = if num_pushes.is_multiple_of(2) { 8 } else { 0 };
    aligned + push_correction
}

/// Materialise an arbitrary u32 into `%r10d` using a single instruction.
/// x86_64 `movl $imm32, %r10d` accepts any 32-bit immediate directly —
/// no movz/movk dance required (contrast arm64::asm::emit_imm32).
pub fn emit_imm32_to_r10(value: u32) -> String {
    format!("    movl    ${}, %r10d\n", value)
}

/// Number of pushes the prologue emits, given the callee-saved set.
fn prologue_push_count(regs: RegSet) -> usize {
    let mut n = 1; // always: push %rbp
    if regs.contains_callee_saved_int() {
        n += 5; // %rbx, %r12, %r13, %r14, %r15
    }
    n
}

/// Format the function prologue (AT&T syntax):
///   .globl <prefix><name>
///   .p2align 4, 0x90
///   <prefix><name>:
///       pushq   %rbp
///       movq    %rsp, %rbp
///       [if non-leaf: pushq %rbx; pushq %r12; pushq %r13; pushq %r14; pushq %r15]
///       [if frame_size > 0: subq $frame_size, %rsp]
///
/// `intermediate_bytes` is the total bytes that need to live on the
/// stack frame: stack-resident intermediate buffers plus the 16-byte
/// fused-softmax xmm-spill reserve when applicable. Pass
/// `BufferAssignment::stack_bytes` directly — `assign_buffers` already
/// folds the spill reserve into that value when the model calls
/// libm-expf (spec §7.4). The total `frame_size` passed to `subq`
/// adds any alignment correction from [`compute_frame_size`].
pub fn format_function_prologue(
    sig: &FnSig,
    regs: RegSet,
    intermediate_bytes: usize,
    sym_prefix: &str,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(".globl {}{}\n", sym_prefix, sig.name));
    s.push_str(".p2align 4, 0x90\n");
    s.push_str(&format!("{}{}:\n", sym_prefix, sig.name));
    s.push_str("    pushq   %rbp\n");
    s.push_str("    movq    %rsp, %rbp\n");

    if regs.contains_callee_saved_int() {
        s.push_str("    pushq   %rbx\n");
        s.push_str("    pushq   %r12\n");
        s.push_str("    pushq   %r13\n");
        s.push_str("    pushq   %r14\n");
        s.push_str("    pushq   %r15\n");
    }

    let n_pushes = prologue_push_count(regs);
    let frame_size = compute_frame_size(intermediate_bytes as u32, n_pushes);
    if frame_size > 0 {
        s.push_str(&format!("    subq    ${}, %rsp\n", frame_size));
    }
    s
}

/// Symmetric epilogue: restore %rsp, pop callee-saved (reverse order),
/// pop %rbp, ret.
pub fn format_function_epilogue(regs: RegSet, intermediate_bytes: usize) -> String {
    let mut s = String::new();
    let n_pushes = prologue_push_count(regs);
    let frame_size = compute_frame_size(intermediate_bytes as u32, n_pushes);
    if frame_size > 0 {
        s.push_str(&format!("    addq    ${}, %rsp\n", frame_size));
    }
    if regs.contains_callee_saved_int() {
        s.push_str("    popq    %r15\n");
        s.push_str("    popq    %r14\n");
        s.push_str("    popq    %r13\n");
        s.push_str("    popq    %r12\n");
        s.push_str("    popq    %rbx\n");
    }
    s.push_str("    popq    %rbp\n");
    s.push_str("    retq\n");
    s
}
