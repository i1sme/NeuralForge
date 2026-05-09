// SPDX-License-Identifier: Apache-2.0

//! NeuralForge bench harness (OQ-BENCH closure, M11).
//!
//! Compiles fixed NFL fixtures through the host-native profile,
//! times warmup × 10 + measurement × 100 FFI calls, emits markdown
//! to stdout (intended for `$GITHUB_STEP_SUMMARY` in CI).

fn main() {
    eprintln!("bench: skeleton — wiring lands in M11 group B");
    std::process::exit(0);
}
