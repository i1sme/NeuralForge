// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for x86_64 integration tests.

use std::path::PathBuf;

/// Returns true if `cc` is on PATH and runs.
pub fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Assemble + link `asm_source` into a `.so` and return its path.
///
/// Tempdir under `std::env::temp_dir()/nflc-test-x86_64-<pid>/` (left
/// after the test runs; OS or `tmpwatch` reclaims it eventually).
pub fn compile_to_so(asm_source: &str, name: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-test-x86_64-{pid}"));
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create test tempdir {}: {e}", dir.display()));

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", s_path.display()));

    let so_path = dir.join(format!("lib{name}.so"));
    let status = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&so_path)
        .arg(&s_path)
        .status()
        .expect("cc invocation failed");
    assert!(
        status.success(),
        "cc failed to assemble {} (status: {status})",
        s_path.display()
    );

    so_path
}

/// Reference layernorm — bit-exact match with emitter at all D.
///
/// CRITICAL: uses sequential `for` reduction with `+=`. Do NOT replace
/// with `.iter().sum::<f32>()` — under -O3 LLVM may auto-vectorize
/// the sum into a SIMD tree-reduction, changing the order of float
/// additions and breaking bit-exact equivalence with the scalar 3-pass
/// asm. Sequential `for` keeps reduction strictly left-to-right; LLVM
/// does NOT reorder f32 adds without `-ffast-math` (Rust does not
/// enable it).
///
/// Uses pre-computed `inv_d = 1.0/d` and multiplication (`sum * inv_d`)
/// to match the emitter's compile-time `1.0/D` constant in `.rodata` —
/// `sum / d as f32` would diverge by 1 ULP for non-power-of-2 D.
pub fn layernorm_ref(
    input: &[f32],
    shape: &[usize],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
) -> Vec<f32> {
    let d = *shape.last().unwrap();
    let n = shape.iter().take(shape.len() - 1).product::<usize>();
    let inv_d = 1.0_f32 / d as f32;
    let eps = 1e-5_f32;

    let mut out = Vec::with_capacity(input.len());
    for r in 0..n {
        let row = &input[r * d..(r + 1) * d];

        let mut sum = 0.0_f32;
        for &x in row {
            sum += x;
        }
        let mean = sum * inv_d;

        let mut sumsq = 0.0_f32;
        for &x in row {
            sumsq += (x - mean) * (x - mean);
        }
        let var = sumsq * inv_d;
        let inv_std = 1.0_f32 / (var + eps).sqrt();

        for (i, &x) in row.iter().enumerate() {
            let normalized = (x - mean) * inv_std;
            let val = match (gamma, beta) {
                (Some(g), Some(b)) => normalized * g[i] + b[i],
                _ => normalized,
            };
            out.push(val);
        }
    }
    out
}

/// Reference matmul — naive `b × k` @ `k × n` → `b × n` with **non-FMA**
/// (`+= a * b`) reduction matching the x86_64 SSE2 emitter's `mulss + addss`
/// two-rounding pattern. Promoted from integration.rs file-local in M15
/// with x86_64-specific divergence from arm64 (which uses `f32::mul_add`
/// matching its `fmadd` single-rounding emitter).
///
/// Bit-exact equivalence with x86_64 `emit_linear` body (`mulss %xmm2, %xmm1;
/// addss %xmm1, %xmm0` per inner k-iteration) requires non-FMA reduction.
/// Using `f32::mul_add` here would produce ≤0.5 ULP per-element divergence
/// from the emitter and break `to_bits()` comparison in FFI tests.
///
/// Rust's `+`/`*` operators do NOT contract to FMA at default codegen
/// settings — strict IEEE 754 with two roundings, matching the SSE2 emitter.
pub fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += input[i * k + kk] * weights[kk * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Reference f32 exp for x ≤ 0 — bit-exact match for the x86_64 inline emitter.
/// Cody-Waite range reduction + degree-7 Taylor (Horner) + 2^z.
///
/// CRITICAL: SSE2 has no scalar FMA, so every multiply-accumulate is a separate
/// `mulss`+`addss` (two roundings). This port uses separate `*` and `+`/`-` —
/// NOT `f32::mul_add`. (Mirror of the per-profile reference_matmul split, M15.)
///
/// Constants are INTENTIONALLY spelled out at full double precision before
/// truncation to f32 — they must match the emitter literals exactly.
/// Clippy lints are suppressed so the bit-patterns are preserved as-is.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub fn exp_ref(x: f32) -> f32 {
    const LOG2E: f32 = 1.4426950408889634;
    const LN2_HI: f32 = 0.693359375;
    const LN2_LO: f32 = -0.00021219444005469057;
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
    let z = (x * LOG2E).round_ties_even() as i32; // cvtss2si: ties-even
    let zf = z as f32;
    let r = x - zf * LN2_HI; // two roundings (mul then sub)
    let r = r - zf * LN2_LO;
    let mut p = C[7];
    for k in (0..7).rev() {
        p = p * r + C[k]; // mul then add — two roundings
    }
    let zp = z + 127;
    let pow = if zp <= 0 {
        0.0_f32
    } else {
        f32::from_bits((zp as u32) << 23)
    };
    p * pow
}

/// Reference bias add — broadcast `bias[n]` across `b` rows of `acc[b*n]`,
/// in place semantically (returns new vec, doesn't mutate input).
/// Promoted from integration.rs file-local in M15.
pub fn reference_bias_add(acc: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    let b = acc.len() / n;
    let mut out = acc.to_vec();
    for i in 0..b {
        for j in 0..n {
            out[i * n + j] += bias[j];
        }
    }
    out
}

/// Reference relu — element-wise max(x, 0.0). Promoted from
/// integration.rs file-local in M15.
pub fn reference_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x.max(0.0)).collect()
}

/// Reference FFN — composes `reference_matmul` + `reference_bias_add` +
/// `reference_relu` in the order `linear[w1, b1] → relu → linear[w2, b2]`.
///
/// Shapes: input `[batch, dim]` → matmul w1 `[dim, hidden]` → bias b1 → relu
///       → matmul w2 `[hidden, dim]` → bias b2 → output `[batch, dim]`.
///
/// CRITICAL (M15 helper-reuse rule, see design spec §3.4): this function
/// MUST compose the promoted primitives above. Do NOT inline a fresh matmul
/// or bias loop — divergent reduction order produces 1+ ULP mismatches that
/// fail bit-exact comparison and are deeply painful to debug.
#[allow(clippy::too_many_arguments)]
pub fn ffn_ref(
    input: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
    dim: usize,
    hidden: usize,
) -> Vec<f32> {
    let mm1 = reference_matmul(input, w1, batch, dim, hidden);
    let mm1_b = reference_bias_add(&mm1, b1, hidden);
    let r1 = reference_relu(&mm1_b);
    let mm2 = reference_matmul(&r1, w2, batch, hidden, dim);
    reference_bias_add(&mm2, b2, dim)
}

/// Reference transformer block — composes `layernorm_ref` + `ffn_ref` +
/// element-wise add. Mirrors the `transformer_block.nfl` fixture pipeline:
/// `x -> layernorm[affine] -> linear -> relu -> linear -> add[skip1] -> add[skip2]`.
///
/// CRITICAL (helper-reuse rule, design spec §3.4): this function MUST compose
/// `layernorm_ref` (M14, above) and `ffn_ref` (M15, above). Do NOT reimplement
/// LayerNorm normalization, matmul reduction, or bias add. The existing
/// helpers are M14-verified bit-exact against emitters; reuse them as-is.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_ref(
    input: &[f32],
    skip1: &[f32],
    skip2: &[f32],
    gamma: &[f32],
    beta: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
    dim: usize,
    hidden: usize,
) -> Vec<f32> {
    // 1. layernorm[affine=true]
    let ln = layernorm_ref(input, &[batch, dim], Some(gamma), Some(beta));
    // 2. ffn (linear → relu → linear with bias on both)
    let ffn_out = ffn_ref(&ln, w1, b1, w2, b2, batch, dim, hidden);
    // 3. add[skip1] (element-wise)
    let r1: Vec<f32> = ffn_out
        .iter()
        .zip(skip1.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    // 4. add[skip2] (element-wise)
    r1.iter().zip(skip2.iter()).map(|(&a, &b)| a + b).collect()
}
