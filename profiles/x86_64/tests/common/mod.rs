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
        .args(["-lm"])
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
