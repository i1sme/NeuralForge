// SPDX-License-Identifier: Apache-2.0

//! Inline bare-metal `exp` for the softmax domain (x ≤ 0). Replaces the
//! M3-era `bl _expf`. See docs/superpowers/specs/2026-05-29-bare-metal-expf-m17-design.md.

/// f32 constants — MUST stay identical to `exp_ref` in
/// `profiles/arm64/tests/common/mod.rs` (drift caught by the Task 5 bit-exact test).
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

/// Emit AArch64 inline `exp` for x ≤ 0. Input in `s0`; result in `s0`.
///
/// Scratch (all non-loop-live; the softmax loop owns x19-x23/s8/s9, NOT
/// touched here): x9 (pool base), w11 (z), w12 (pow bits), s1-s5 (FP temps).
/// Branchless underflow clamp via `csel` — no labels, safe to inline at
/// multiple sites without unique suffixes.
pub fn emit_exp_inline() -> String {
    let mut s = String::new();
    s.push_str("    ; --- inline exp(x), x<=0 (M17) ---\n");
    s.push_str("    adrp    x9, .Lexp_log2e@PAGE\n");
    s.push_str("    ldr     s1, [x9, .Lexp_log2e@PAGEOFF]\n");
    s.push_str("    fmul    s2, s0, s1\n");
    s.push_str("    fcvtns  w11, s2\n");
    s.push_str("    scvtf   s2, w11\n");
    s.push_str("    ldr     s1, [x9, .Lexp_ln2hi@PAGEOFF]\n");
    s.push_str("    fmsub   s3, s2, s1, s0\n");
    s.push_str("    ldr     s1, [x9, .Lexp_ln2lo@PAGEOFF]\n");
    s.push_str("    fmsub   s3, s2, s1, s3\n");
    s.push_str("    ldr     s4, [x9, .Lexp_c7@PAGEOFF]\n");
    for k in (0..7).rev() {
        s.push_str(&format!("    ldr     s1, [x9, .Lexp_c{}@PAGEOFF]\n", k));
        s.push_str("    fmadd   s4, s4, s3, s1\n");
    }
    s.push_str("    add     w11, w11, #127\n");
    s.push_str("    lsl     w12, w11, #23\n");
    s.push_str("    cmp     w11, #0\n");
    s.push_str("    csel    w12, wzr, w12, le\n");
    s.push_str("    fmov    s5, w12\n");
    s.push_str("    fmul    s0, s4, s5\n");
    s.push_str("    ; --- end inline exp ---\n");
    s
}

/// File-local Mach-O `__const` pool. Emitted ONCE per assembly file from
/// `walk_uir` when `uir.has_softmax()`. `.L`-local labels: one definition,
/// referenced from every `emit_exp_inline` site; locals do not collide across
/// separately-linked objects.
pub fn exp_pool_arm64() -> String {
    let mut s = String::new();
    s.push_str(".section __TEXT,__const\n");
    s.push_str(".p2align 2\n");
    s.push_str(&format!(".Lexp_log2e: .long 0x{:08x}\n", LOG2E.to_bits()));
    s.push_str(&format!(".Lexp_ln2hi: .long 0x{:08x}\n", LN2_HI.to_bits()));
    s.push_str(&format!(".Lexp_ln2lo: .long 0x{:08x}\n", LN2_LO.to_bits()));
    for (k, c) in C.iter().enumerate() {
        s.push_str(&format!(".Lexp_c{}: .long 0x{:08x}\n", k, c.to_bits()));
    }
    s
}
