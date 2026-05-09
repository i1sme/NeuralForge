// SPDX-License-Identifier: Apache-2.0

//! Softmax (per-row stable, libm expf via PLT) codegen — x86_64 SSE2.

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;

/// Emit x86_64 asm for softmax over `[b, k]` shape (per-row normalize).
///
/// Calls `<sym_prefix>expf@PLT` for each element. State across the call
/// lives in callee-saved int registers (%rbx, %r12, %r13, %r14, %r15)
/// and on the stack (`(%rsp)` row_max, `8(%rsp)` row_sum; reserved by
/// `assign_buffers` per spec §7.4); see the register contract in
/// profiles/x86_64/src/ops/linear.rs.
///
/// FFI register preservation (M12): `call expf@PLT` clobbers caller-
/// saved regs including the SysV argument registers (input(s), params,
/// output) per AMD64 ABI. Any downstream emitter that reads any of
/// these via `abi.materialise_ptr` would otherwise see garbage. Spec
/// §5.4: any emitter that clobbers an ABI register that a follow-up
/// emitter reads must save/restore. Self-attention exercises this:
/// matmul → softmax → matmul where the second matmul's
/// `abi.materialise_ptr(InputReg(_), …)` reads %rdi/%rsi after
/// softmax's `call expf@PLT` has clobbered them.
///
/// Strategy: at function entry, push the full `ffi_save_set()`
/// (= INPUT_REGS[..N+2]) onto the stack, with `pushq %rax` padding
/// for odd cardinality. For N=1 this is `pushq %rdi/%rsi/%rdx/%rax`
/// — bit-identical to M11. For N=3 the body's row_max/row_sum slots
/// shift from `(%rsp)/8(%rsp)` to `48(%rsp)/56(%rsp)` (6 pushes ×
/// 8 bytes). The arithmetic for slot offsets follows the formula
/// `slot_offset = ceil((N+2) / 2) * 16` — derivable from
/// `ffi_save_set().len()` rounded up to even cardinality times 8.
#[allow(clippy::too_many_arguments)]
pub fn emit_softmax(
    abi: &AbiContext,
    b: u64,
    k: u64,
    model_idx: usize,
    softmax_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    sym_prefix: &str,
) -> String {
    let sid = format!("{model_idx}_{softmax_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # softmax (3-pass): input [{b},{k}] -> output [{b},{k}]\n"
    ));

    // Pin src/dst into callee-saved %rbx/%r12 (survives `call expf@PLT`).
    // MUST happen before the FFI-reg push below: `materialise_ptr` for any
    // `BufferLoc::StackOffset(off)` emits `leaq off(%rsp), %rbx` against
    // the current %rsp. Pushing first would shift %rsp and corrupt the
    // intermediate-buffer offsets baked in at assign_buffers time.
    abi.materialise_ptr(src_loc, "%rbx", &mut s);
    abi.materialise_ptr(dst_loc, "%r12", &mut s);

    // Spill the full ABI argument set (inputs + params + output) so they
    // survive `call expf@PLT`. Arity-aware via AbiContext (spec §5.4,
    // §6.1). For N=1 this emits `pushq %rdi; pushq %rsi; pushq %rdx;
    // pushq %rax` (with %rax as alignment padding) — bit-identical to
    // the M11 hand-written block.
    abi.emit_ffi_save(&mut s);

    // Spill SP delta — used to compute the row_max / row_sum stack slot
    // offsets after the FFI-reg push. SP delta is always a multiple of
    // 16 bytes (16-byte alignment invariant); for N=1 → 32, N=3 → 48.
    let push_count = abi.ffi_save_set().len()
        + if abi.ffi_save_set().len().is_multiple_of(2) {
            0
        } else {
            1
        };
    let sp_shift = push_count * 8;
    let row_max_slot = sp_shift; // (was 0(%rsp))
    let row_sum_slot = sp_shift + 8; // (was 8(%rsp))

    // Outer per-row loop: %r13 = i.
    s.push_str("    xorq    %r13, %r13\n");
    s.push_str(&format!(".Lsm_i_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(b as u32));
    s.push_str("    cmpq    %r10, %r13\n");
    s.push_str(&format!("    jge     .Lsm_i_end_{sid}\n"));

    // %r15 = i * k
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %r13, %r15\n");
    s.push_str("    imulq   %r10, %r15\n");

    // Phase 1: row_max → row_max_slot(%rsp). Init xmm8 to -inf.
    s.push_str("    movl    $0xFF800000, %r10d\n");
    s.push_str("    movd    %r10d, %xmm8\n");
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_max_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_max_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str("    maxss   %xmm0, %xmm8\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_max_{sid}\n"));
    s.push_str(&format!(".Lsm_max_end_{sid}:\n"));
    s.push_str(&format!("    movss   %xmm8, {}(%rsp)\n", row_max_slot));

    // Phase 2: exp(x - max) → dst, sum → row_sum_slot(%rsp). Init sum to 0.
    s.push_str(&format!("    movl    $0, {}(%rsp)\n", row_sum_slot));
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_exp_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_exp_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%rbx, %rax, 4), %xmm0\n");
    s.push_str(&format!("    subss   {}(%rsp), %xmm0\n", row_max_slot));
    s.push_str(&format!("    call    {}expf@PLT\n", sym_prefix));
    // %rax clobbered by call; recompute.
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str(&format!("    movss   {}(%rsp), %xmm1\n", row_sum_slot));
    s.push_str("    addss   %xmm0, %xmm1\n");
    s.push_str(&format!("    movss   %xmm1, {}(%rsp)\n", row_sum_slot));
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_exp_{sid}\n"));
    s.push_str(&format!(".Lsm_exp_end_{sid}:\n"));

    // Phase 3: normalise by row_sum.
    s.push_str("    xorq    %r14, %r14\n");
    s.push_str(&format!(".Lsm_norm_{sid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r14\n");
    s.push_str(&format!("    jge     .Lsm_norm_end_{sid}\n"));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    addq    %r14, %rax\n");
    s.push_str("    movss   (%r12, %rax, 4), %xmm0\n");
    s.push_str(&format!("    divss   {}(%rsp), %xmm0\n", row_sum_slot));
    s.push_str("    movss   %xmm0, (%r12, %rax, 4)\n");
    s.push_str("    incq    %r14\n");
    s.push_str(&format!("    jmp     .Lsm_norm_{sid}\n"));
    s.push_str(&format!(".Lsm_norm_end_{sid}:\n"));

    s.push_str("    incq    %r13\n");
    s.push_str(&format!("    jmp     .Lsm_i_{sid}\n"));
    s.push_str(&format!(".Lsm_i_end_{sid}:\n"));

    // Restore the ABI argument set — strict LIFO order to match
    // `abi.emit_ffi_save` above. For N=1 this emits `popq %rax;
    // popq %rdx; popq %rsi; popq %rdi` (bit-identical to M11).
    abi.emit_ffi_restore(&mut s);

    s
}
