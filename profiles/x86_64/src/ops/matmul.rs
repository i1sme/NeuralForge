// SPDX-License-Identifier: Apache-2.0

//! Matmul codegen — x86_64 SSE2, AT&T syntax. Outer loop over the product
//! of leading dims; inner triple-loop matmul kernel using `mulss + addss`
//! (no FMA — matches emit_linear x86_64's deliberate non-FMA design from
//! M9).
//!
//! Register usage (M9 hazard avoidance):
//!   %r8, %r9, %r11  — A, B, DST base pointers (materialised once)
//!   %rcx            — outer-loop counter / inner j-counter (saved across via push)
//!   %r10            — imm32-to-r10 scratch (clobbered each emit_imm32_to_r10 call)
//!   %rax            — i-counter / address compute scratch
//!   %rdi, %rsi, %rdx — per-outer A/B/DST slice pointers (clobbered;
//!                      %rdi, %rsi, %rdx are spilled to %xmm8/%xmm6/%xmm7
//!                      and restored at function exit so downstream
//!                      emitters see the original FFI register state)
//!   %xmm0           — accumulator
//!   %xmm1, %xmm2    — operand fetches (mulss / addss)
//!   %xmm6, %xmm7, %xmm8 — defensive spill of %rsi / %rdx / %rdi
//!
//! `%rdi` (input), `%rsi` (params), `%rdx` (output) preservation: see plan
//! §9a and the M9 fixes in commits ecb69ac (preserves %rsi across
//! emit_linear's matmul body) and c3ff521 (preserves %rdx when bias
//! clobbers it). The %rdi spill closes the same hazard for the input
//! pointer, surfaced by the M10 attention fixture (two consecutive
//! matmuls both reading the same `x` InputReg base).

use crate::asm::{emit_imm32_to_r10, materialise_ptr};
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
#[allow(clippy::too_many_arguments)]
pub fn emit_matmul(
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

    // Defensive preservation of FFI registers %rdi (input), %rsi
    // (params), and %rdx (output). Mirrors the M9 fixes (ecb69ac,
    // c3ff521): emit_matmul reuses %rdi, %rsi, %rdx as
    // per-outer-iteration A_slice/B_slice/DST_slice scratch pointers,
    // so any downstream emitter (e.g. another matmul, a softmax, a
    // linear) that reads any of these after this body would otherwise
    // see garbage. The xmm-spill triple is the canonical M9 idiom —
    // cheap (3 movq instructions at entry, 3 at exit) and free of
    // stack frame impact.
    //
    // The %rdi spill is the M10 fixup: the attention fixture has two
    // matmuls that both materialise B from `x` (InputReg → %rdi). The
    // first matmul's outer loop overwrote %rdi with an A_slice
    // pointer, so the second matmul's `materialise_ptr("%r9", InputReg)`
    // copied garbage into %r9 and silently miscompiled.
    s.push_str("    movq    %rdi, %xmm8\n"); // preserve %rdi (input ptr)
    s.push_str("    movq    %rsi, %xmm6\n"); // preserve %rsi (params ptr)
    s.push_str("    movq    %rdx, %xmm7\n"); // preserve %rdx (output ptr)

    // Materialise base pointers ONCE — invariant across outer iterations.
    s.push_str(&materialise_ptr("%r8", a_loc));
    s.push_str(&materialise_ptr("%r9", b_loc));
    s.push_str(&materialise_ptr("%r11", dst_loc));

    // Inner-kernel slice sizes (in floats):
    //   A slice = M * K   (per outer iteration)
    //   B slice = K * N   (always — same regardless of transpose_b)
    //   DST slice = M * N (per outer iteration)
    let a_slice = m as usize * k as usize;
    let b_slice = k as usize * n as usize;
    let dst_slice = m as usize * n as usize;

    // Outer loop counter %rcx (caller-saved, conventional counter reg).
    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lmm4d_outer_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(leading_count as u32));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm4d_outer_end_{mid}\n"));

    // Per-outer slice base pointers go into %rdi (A_slice), %rsi
    // (B_slice), %rdx (DST_slice). These three FFI regs are now scratch
    // — the matmul body's only output is the in-buffer store via %rdx,
    // and the original %rdi/%rsi/%rdx are preserved in
    // %xmm8/%xmm6/%xmm7 for function-exit restore.
    //
    // A_slice = %r8 + %rcx * a_slice * 4
    s.push_str(&emit_imm32_to_r10(a_slice as u32));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r8, %rax, 4), %rdi\n");
    // B_slice = %r9 + %rcx * b_slice * 4
    s.push_str(&emit_imm32_to_r10(b_slice as u32));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r9, %rax, 4), %rsi\n");
    // DST_slice = %r11 + %rcx * dst_slice * 4
    s.push_str(&emit_imm32_to_r10(dst_slice as u32));
    s.push_str("    movq    %rcx, %rax\n");
    s.push_str("    imulq   %r10, %rax\n");
    s.push_str("    leaq    (%r11, %rax, 4), %rdx\n");

    // Save the outer counter on the stack so we can reuse %rcx as the
    // inner j-counter. Restored at the bottom of each outer iter.
    s.push_str("    pushq   %rcx\n");

    // Inner i-loop ([0, M)). Counter in %rax.
    s.push_str("    movq    $0, %rax\n");
    s.push_str(&format!(".Lmm4d_i_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(m as u32));
    s.push_str("    cmpq    %r10, %rax\n");
    s.push_str(&format!("    jge     .Lmm4d_i_end_{mid}\n"));

    // Inner j-loop ([0, N)). Counter in %rcx.
    s.push_str("    movq    $0, %rcx\n");
    s.push_str(&format!(".Lmm4d_j_{mid}:\n"));
    s.push_str(&emit_imm32_to_r10(n as u32));
    s.push_str("    cmpq    %r10, %rcx\n");
    s.push_str(&format!("    jge     .Lmm4d_j_end_{mid}\n"));

    // Accumulator init.
    s.push_str("    xorps   %xmm0, %xmm0\n");
    // Inner k-loop ([0, K)). Counter in %r10. Bound K is materialised
    // inline as an immediate in the cmpq — `cmpq $K, %r10` accepts a
    // 32-bit signed immediate directly, so no register-held bound is
    // needed.
    s.push_str("    movq    $0, %r10\n");
    s.push_str(&format!(".Lmm4d_k_{mid}:\n"));
    s.push_str(&format!("    cmpq    ${}, %r10\n", k));
    s.push_str(&format!("    jge     .Lmm4d_k_end_{mid}\n"));

    // a_offset = i * K + k_inner   (always — A is always [..., M, K])
    // Spill DST base (%r11) to the stack so we can use it as offset
    // scratch; restore after the load. A 2 push/pop pair per inner
    // iteration is the simplest correct allocation here — we have no
    // free GPR that doesn't conflict with an active loop counter
    // (%rax = i, %rcx = j, %r10 = k_inner) or with %rdi/%rsi/%rdx
    // (slice base pointers) or with the materialised A/B/DST bases
    // (%r8/%r9/%r11).
    s.push_str("    pushq   %r11\n");
    s.push_str("    movq    %rax, %r11\n");
    s.push_str(&format!("    imulq   ${}, %r11\n", k));
    s.push_str("    addq    %r10, %r11\n");
    s.push_str("    movss   (%rdi, %r11, 4), %xmm1\n"); // %xmm1 = A[a_offset]

    // b_offset depends on transpose_b:
    //   false: b_offset = k_inner * N + j   (B is [..., K, N])
    //   true:  b_offset = j * K + k_inner   (B is [..., N, K])
    if transpose_b {
        s.push_str("    movq    %rcx, %r11\n");
        s.push_str(&format!("    imulq   ${}, %r11\n", k));
        s.push_str("    addq    %r10, %r11\n");
    } else {
        s.push_str("    movq    %r10, %r11\n");
        s.push_str(&format!("    imulq   ${}, %r11\n", n));
        s.push_str("    addq    %rcx, %r11\n");
    }
    s.push_str("    movss   (%rsi, %r11, 4), %xmm2\n"); // %xmm2 = B[b_offset]

    // Two-step (no FMA): %xmm1 = A * B; %xmm0 += %xmm1.
    s.push_str("    mulss   %xmm2, %xmm1\n");
    s.push_str("    addss   %xmm1, %xmm0\n");
    s.push_str("    popq    %r11\n");

    s.push_str("    addq    $1, %r10\n");
    s.push_str(&format!("    jmp     .Lmm4d_k_{mid}\n"));
    s.push_str(&format!(".Lmm4d_k_end_{mid}:\n"));

    // Store %xmm0 → DST_slice[i*N + j] using %rdx (= DST_slice base).
    // Spill %r11 again as offset scratch.
    s.push_str("    pushq   %r11\n");
    s.push_str("    movq    %rax, %r11\n");
    s.push_str(&format!("    imulq   ${}, %r11\n", n));
    s.push_str("    addq    %rcx, %r11\n");
    s.push_str("    movss   %xmm0, (%rdx, %r11, 4)\n");
    s.push_str("    popq    %r11\n");

    // j++; j-loop tail.
    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lmm4d_j_{mid}\n"));
    s.push_str(&format!(".Lmm4d_j_end_{mid}:\n"));

    // i++; i-loop tail.
    s.push_str("    addq    $1, %rax\n");
    s.push_str(&format!("    jmp     .Lmm4d_i_{mid}\n"));
    s.push_str(&format!(".Lmm4d_i_end_{mid}:\n"));

    // Restore the outer counter, increment, continue.
    s.push_str("    popq    %rcx\n");
    s.push_str("    addq    $1, %rcx\n");
    s.push_str(&format!("    jmp     .Lmm4d_outer_{mid}\n"));
    s.push_str(&format!(".Lmm4d_outer_end_{mid}:\n"));

    // Restore preserved %rdi / %rsi / %rdx for follow-up ops.
    s.push_str("    movq    %xmm8, %rdi\n"); // restore %rdi (input ptr)
    s.push_str("    movq    %xmm6, %rsi\n");
    s.push_str("    movq    %xmm7, %rdx\n");

    Ok(s)
}
