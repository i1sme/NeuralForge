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
