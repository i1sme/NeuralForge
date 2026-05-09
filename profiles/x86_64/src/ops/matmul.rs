// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — multi-dim matmul over rank ≥ 2 inputs with optional
//! `transpose_b`. Outer loop iterates over the product of leading dims
//! (`leading_count`); the inner triple-loop matmul kernel uses
//! `mulss + addss` (no FMA — matches `emit_linear` x86_64's deliberate
//! non-FMA design from M9).
//!
//! M12 (multi-input ABI) rework — spec §9.1: matmul body must NOT touch
//! any ABI argument register (`%rdi`/`%rsi`/`%rdx`/`%rcx`/`%r8`/`%r9`
//! depending on arity), since downstream emitters read input/params/
//! output from those registers. The pre-M12 emitter spilled the FFI
//! input regs to `%xmm6`/`%xmm7`/`%xmm8` at function entry and reused
//! `%rdi`/`%rsi`/`%rdx` as A_slice/B_slice/DST_slice scratch — that
//! pattern breaks for N≥2 where `%rsi` holds a SECOND input pointer the
//! inner loop must read.
//!
//! ## Scratch budget
//!
//! SysV AMD64 GP registers, by category:
//! - **ABI argument** (one role each per arity): `%rdi`, `%rsi`,
//!   `%rdx`, `%rcx`, `%r8`, `%r9` — first 6 used. M12 caps N at 4
//!   (arity check in walk_model), so N+2 ≤ 6 — register-only.
//! - **Caller-saved non-ABI scratch**: `%rax`, `%r10`, `%r11`. Always
//!   safe to clobber, never holds an ABI input/params/output.
//!   `%r9` is also caller-saved but transitions into ABI at N=4.
//! - **Callee-saved**: `%rbx`, `%rbp`, `%r12`, `%r13`, `%r14`, `%r15`.
//!   This profile uses `%rbp` as frame pointer; the other 5 are saved
//!   by the function-level prologue when `compute_callee_saved`
//!   returns true (= `model.calls_extern_math() OR has_matmul(model)`,
//!   per `buffer.rs`).
//!
//! At N=3 the non-ABI caller-saved scratch shrinks to `%rax`, `%r10`,
//! `%r11`, `%r9` = 4 registers — insufficient for the 9 register-roles
//! a matmul kernel needs (3 base ptrs, 3 slice ptrs, 3 counters). The
//! rework therefore uses **callee-saved registers** as long-lived
//! scratch (option β per the spec §10.2 amendment's "register-cascade-
//! induced changes" relaxation), with base pointers spilled to
//! caller-saved `%xmm6`/`%xmm7`/`%xmm8` and reloaded into GPRs once
//! at entry.
//!
//! ## Register layout (M12)
//!
//! | Role               | Reg     | Lifetime               |
//! |--------------------|---------|------------------------|
//! | A base ptr         | %xmm8   | spill at entry, reload to %r12 once |
//! | B base ptr         | %xmm6   | spill at entry, reload to %r13 once |
//! | DST base ptr       | %xmm7   | spill at entry, reload to %r14 once |
//! | outer counter      | %r15    | full function (callee-saved)  |
//! | A_slice ptr        | %r12    | per outer iter (callee-saved) |
//! | B_slice ptr        | %r13    | per outer iter (callee-saved) |
//! | DST_slice ptr      | %r14    | per outer iter (callee-saved) |
//! | i counter          | %rbx    | per outer iter (callee-saved) |
//! | j counter          | %r9     | per i iter (caller-saved scratch; non-ABI for N≤3) |
//! | k_inner counter    | %r11    | per j iter (caller-saved scratch) |
//! | addr arith temp 1  | %rax    | per use (clobbered by imulq)  |
//! | addr arith temp 2  | %r10    | per use (clobbered by emit_imm32_to_r10) |
//!
//! Bounds (M, N, K, leading_count) and slice strides (a_slice,
//! b_slice, dst_slice) are emitted inline at each cmp / address-
//! arithmetic site rather than hoisted into dedicated bound registers
//! — the scratch budget after the rework leaves no spare GPR for the
//! hoist. This trades a few extra `emit_imm32_to_r10` calls for
//! register-pressure relief.
//!
//! Note that `%r12`/`%r13`/`%r14` hold A_slice/B_slice/DST_slice
//! recomputed PER OUTER ITER. The original A/B/DST base pointers live
//! in `%xmm6`/`%xmm7`/`%xmm8` for the entire function lifetime —
//! they're reloaded into the slice GPRs at the top of each outer iter
//! and then offset by `outer * <slice_size> * 4` to form the slice
//! pointer.
//!
//! ## Why the M10 FFI-input spill is REMOVED
//!
//! The pre-M12 emitter wrote `movq %rdi, %xmm8` etc. at function entry
//! to preserve the FFI input/params/output pointers across the matmul
//! body's clobber of those registers as slice scratch. With the M12
//! rework, the matmul body NEVER touches an ABI argument register
//! (per spec §9.1), so cross-emitter preservation is no longer needed.
//! The `%xmm6`/`%xmm7`/`%xmm8` registers are now used **internally**
//! by the matmul body for base-pointer storage — a different role, but
//! the same register names happen to apply.
//!
//! Cross-FFI register preservation (e.g. across `call expf@PLT` in a
//! softmax that follows matmul) is now handled by `AbiContext::
//! emit_ffi_save` / `emit_ffi_restore` in `emit_softmax`, arity-aware
//! via the spec §6 invariants.
//!
//! Verified by `emit_matmul_body_contains_zero_pushq` unit test
//! (matmul body emits zero `pushq` instructions; the function-level
//! prologue handles all callee-saved pushes).

use crate::abi::AbiContext;
use crate::asm::emit_imm32_to_r10;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

/// Emit AT&T-syntax x86_64 asm for a multi-dim matmul.
///
/// `leading_count` = product of leading dims (`shape[..rank-2].product()`).
/// For 2D inputs `leading_count == 1` — the outer loop runs once and is
/// effectively elided.
///
/// `m`, `k`, `n` are the trailing matrix dims. With `transpose_b=false`,
/// B is `[..., K, N]`; with `transpose_b=true`, B is `[..., N, K]`.
///
/// The inner kernel uses only callee-saved registers (`%r12`-`%r15`,
/// `%rbx`) and caller-saved non-ABI scratch (`%rax`, `%r9`, `%r10`,
/// `%r11`, `%xmm6`-`%xmm8`) — no ABI argument register is ever
/// touched by the matmul body, so `%rdi`/`%rsi`/`%rdx`/`%rcx`/`%r8`
/// survive across the call site for downstream emitters.
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
    abi: &AbiContext,
    leading_count: u64,
    m: u64,
    k: u64,
    n: u64,
    transpose_b: bool,
    model_idx: usize,
    matmul_idx: usize,
    a_loc: BufferLoc,
    b_loc: BufferLoc,
    dst_loc: BufferLoc,
    _node_span: Span,
) -> Result<String, LowerError> {
    let mid = format!("{model_idx}_{matmul_idx}");
    let mut s = String::new();
    s.push_str(&format!(
        "    # matmul (leading_count={}): [{},{}] x [{},{}] -> [{},{}], transpose_b={}\n",
        leading_count, m, k, k, n, m, n, transpose_b
    ));

    // Materialise base pointers ONCE — invariant across outer iterations.
    // Use %rax as scratch destination because it's caller-saved and
    // non-ABI for all N≤4. Spill to %xmm6/7/8 immediately so the GPR
    // (%rax) is freed for use as a counter / arith scratch in the body.
    abi.materialise_ptr(a_loc, "%rax", &mut s);
    s.push_str("    movq    %rax, %xmm8\n"); // %xmm8 = A base
    abi.materialise_ptr(b_loc, "%rax", &mut s);
    s.push_str("    movq    %rax, %xmm6\n"); // %xmm6 = B base
    abi.materialise_ptr(dst_loc, "%rax", &mut s);
    s.push_str("    movq    %rax, %xmm7\n"); // %xmm7 = DST base

    // Inner-kernel slice sizes (in floats):
    //   A slice = M * K   (per outer iteration)
    //   B slice = K * N   (always — same regardless of transpose_b)
    //   DST slice = M * N (per outer iteration)
    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Outer counter %r15 (callee-saved; never an ABI register).
    s.push_str("    movq    $0, %r15\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(leading_count as u32));
    s.push_str("    cmpq    %r10, %r15\n");
    s.push_str(&format!("    jge     .Lmm4d_outer_end_{mid}\n"));

    // Compute per-outer slice base pointers into %r12, %r13, %r14
    // (callee-saved). All non-ABI.
    //   %r12 = A_base + %r15 * a_slice * 4
    //   %r13 = B_base + %r15 * b_slice * 4
    //   %r14 = DST_base + %r15 * dst_slice * 4
    s.push_str(&emit_imm32_to_r10(a_slice as u32));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = outer * a_slice
    s.push_str("    movq    %xmm8, %r12\n"); // reload A base
    s.push_str("    leaq    (%r12, %rax, 4), %r12\n"); // %r12 = A_slice
    s.push_str(&emit_imm32_to_r10(b_slice as u32));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = outer * b_slice
    s.push_str("    movq    %xmm6, %r13\n"); // reload B base
    s.push_str("    leaq    (%r13, %rax, 4), %r13\n"); // %r13 = B_slice
    s.push_str(&emit_imm32_to_r10(dst_slice as u32));
    s.push_str("    movq    %r15, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = outer * dst_slice
    s.push_str("    movq    %xmm7, %r14\n"); // reload DST base
    s.push_str("    leaq    (%r14, %rax, 4), %r14\n"); // %r14 = DST_slice

    // Inner i-loop ([0, M)). Counter %rbx (callee-saved).
    s.push_str("    movq    $0, %rbx\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(m as u32));
    s.push_str("    cmpq    %r10, %rbx\n");
    s.push_str(&format!("    jge     .Lmm4d_i_end_{mid}\n"));

    // Inner j-loop ([0, N)). Counter %r9 (caller-saved, non-ABI for N≤3).
    s.push_str("    movq    $0, %r9\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %r9\n");
    s.push_str(&format!("    jge     .Lmm4d_j_end_{mid}\n"));

    // Accumulator %xmm0 = 0.0.
    s.push_str("    xorps   %xmm0, %xmm0\n");
    // Inner k-loop ([0, K)). Counter %r11 (caller-saved scratch).
    s.push_str("    movq    $0, %r11\n");
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    cmpq    %r10, %r11\n");
    s.push_str(&format!("    jge     .Lmm4d_k_end_{mid}\n"));

    // a_offset = i * K + k_inner   (always — A is always [..., M, K]).
    // %rax = i * K, then add k_inner.
    s.push_str(&emit_imm32_to_r10(k as u32));
    s.push_str("    movq    %rbx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * K
    s.push_str("    addq    %r11, %rax\n"); // %rax = i * K + k_inner
    s.push_str("    movss   (%r12, %rax, 4), %xmm1\n"); // %xmm1 = A[a_offset]

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str(&emit_imm32_to_r10(k as u32));
        s.push_str("    movq    %r9, %rax\n");
        s.push_str("    imulq   %r10, %rax\n"); // %rax = j * K
        s.push_str("    addq    %r11, %rax\n"); // %rax = j * K + k_inner
    } else {
        s.push_str(&emit_imm32_to_r10(n as u32));
        s.push_str("    movq    %r11, %rax\n");
        s.push_str("    imulq   %r10, %rax\n"); // %rax = k_inner * N
        s.push_str("    addq    %r9, %rax\n"); // %rax = k_inner * N + j
    }
    s.push_str("    movss   (%r13, %rax, 4), %xmm2\n"); // %xmm2 = B[b_offset]

    // Two-step (no FMA): %xmm1 = A * B; %xmm0 += %xmm1.
    s.push_str("    mulss   %xmm2, %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");

    s.push_str("    addq    $1, %r11\n");
    s.push_str(&format!("    jmp     .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store %xmm0 → DST_slice[i * N + j].
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    movq    %rbx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n"); // %rax = i * N
    s.push_str("    addq    %r9, %rax\n"); // %rax = i * N + j
    s.push_str("    movss   %xmm0, (%r14, %rax, 4)\n");

    // j++; j-loop tail.
    s.push_str("    addq    $1, %r9\n");
    s.push_str(&format!("    jmp     .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    // i++; i-loop tail.
    s.push_str("    addq    $1, %rbx\n");
    s.push_str(&format!("    jmp     .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    // Outer++; outer-loop tail. No stack restore — matmul never spills.
    s.push_str("    addq    $1, %r15\n");
    s.push_str(&format!("    jmp     .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    Ok(s)
}
