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
