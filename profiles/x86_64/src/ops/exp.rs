// SPDX-License-Identifier: Apache-2.0

//! Inline bare-metal `exp` for the softmax domain (x ≤ 0) — x86_64 SSE2.
//! Replaces the M3-era `call expf@PLT`. See the M17 design spec.

/// f32 constants — MUST stay identical to `exp_ref` in
/// `profiles/x86_64/tests/common/mod.rs` (drift caught by the Task 5 bit-exact test).
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const LOG2E: f32 = 1.4426950408889634;
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const LN2_HI: f32 = 0.693359375;
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const LN2_LO: f32 = -0.00021219444005469057;
#[allow(clippy::excessive_precision, clippy::approx_constant)]
const C: [f32; 8] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
];

/// Emit x86_64 SSE2 inline `exp` for x ≤ 0. Input in `%xmm0`; result in `%xmm0`.
///
/// Scratch (all non-loop-live; the softmax loop owns %rbx/%r12-%r15 + stack
/// slots, NOT touched here): %eax (z), %ecx/%edx (pow bits), %xmm1-%xmm5.
/// Branchless underflow clamp via `cmovle` — no labels.
pub fn emit_exp_inline() -> String {
    let mut s = String::new();
    s.push_str("    # --- inline exp(x), x<=0 (M17) ---\n");
    s.push_str("    movss   .Lexp_log2e(%rip), %xmm1\n");
    s.push_str("    mulss   %xmm0, %xmm1\n");
    s.push_str("    cvtss2si %xmm1, %eax\n");
    s.push_str("    cvtsi2ss %eax, %xmm2\n");
    s.push_str("    movss   .Lexp_ln2hi(%rip), %xmm3\n");
    s.push_str("    mulss   %xmm2, %xmm3\n");
    s.push_str("    movss   %xmm0, %xmm5\n");
    s.push_str("    subss   %xmm3, %xmm5\n");
    s.push_str("    movss   .Lexp_ln2lo(%rip), %xmm3\n");
    s.push_str("    mulss   %xmm2, %xmm3\n");
    s.push_str("    subss   %xmm3, %xmm5\n");
    s.push_str("    movss   .Lexp_c7(%rip), %xmm4\n");
    for k in (0..7).rev() {
        s.push_str("    mulss   %xmm5, %xmm4\n");
        s.push_str(&format!("    addss   .Lexp_c{}(%rip), %xmm4\n", k));
    }
    s.push_str("    addl    $127, %eax\n");
    s.push_str("    movl    %eax, %ecx\n");
    s.push_str("    shll    $23, %ecx\n");
    s.push_str("    xorl    %edx, %edx\n");
    s.push_str("    testl   %eax, %eax\n");
    s.push_str("    cmovle  %edx, %ecx\n");
    s.push_str("    movd    %ecx, %xmm5\n");
    s.push_str("    mulss   %xmm5, %xmm4\n");
    s.push_str("    movss   %xmm4, %xmm0\n");
    s.push_str("    # --- end inline exp ---\n");
    s
}

/// File-local `.rodata` pool, emitted once per file from `walk_uir` when
/// `uir.has_softmax()`. Mirrors the layernorm pool pattern
/// (profiles/x86_64/src/ops/layernorm.rs). `.L`-local labels.
pub fn exp_pool_x86_64() -> String {
    let mut s = String::new();
    s.push_str(".section .rodata\n");
    s.push_str(".align 4\n");
    s.push_str(&format!(".Lexp_log2e: .long 0x{:08x}\n", LOG2E.to_bits()));
    s.push_str(&format!(".Lexp_ln2hi: .long 0x{:08x}\n", LN2_HI.to_bits()));
    s.push_str(&format!(".Lexp_ln2lo: .long 0x{:08x}\n", LN2_LO.to_bits()));
    for (k, c) in C.iter().enumerate() {
        s.push_str(&format!(".Lexp_c{}: .long 0x{:08x}\n", k, c.to_bits()));
    }
    s
}
