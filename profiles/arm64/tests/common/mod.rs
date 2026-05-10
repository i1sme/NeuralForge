// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for arm64 integration tests.

use std::path::PathBuf;

/// Returns true if `cc` is on PATH and runs.
pub fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Assemble + link `asm_source` into a `.dylib` and return its path.
///
/// Tempdir under `std::env::temp_dir()/nflc-test-<pid>/` (left after
/// the test runs; OS or `tmpwatch` reclaims it eventually).
pub fn compile_to_dylib(asm_source: &str, name: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-test-{pid}"));
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("cannot create test tempdir {}: {e}", dir.display()));

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", s_path.display()));

    let dylib_path = dir.join(format!("lib{name}.dylib"));
    let status = std::process::Command::new("cc")
        .args(["-shared", "-arch", "arm64", "-o"])
        .arg(&dylib_path)
        .arg(&s_path)
        .status()
        .expect("cc invocation failed");
    assert!(
        status.success(),
        "cc failed to assemble {} (status: {status})",
        s_path.display()
    );

    dylib_path
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

/// Reference matmul — naive `b × k` @ `k × n` → `b × n` with `f32::mul_add`
/// reduction. Promoted from integration.rs file-local in M15 to enable
/// reuse from `ffn_ref` and `transformer_block_ref`.
///
/// Reduction order is sequential left-to-right (`mul_add` accumulator) —
/// matches the emitter's scalar fmadd loop bit-exactly. Do NOT replace
/// with iterator-based fold under -O3 (auto-vec tree reduction breaks
/// bit-exact equivalence; same constraint as `layernorm_ref` above).
pub fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum = f32::mul_add(input[i * k + kk], weights[kk * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
    out
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
