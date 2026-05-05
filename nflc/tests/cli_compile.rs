//! CLI integration tests for `nflc compile`.
//!
//! Cargo runs integration-test binaries with cwd at the package root
//! (`nflc/`), so paths to workspace-root fixtures are written as
//! `"../tests/fixtures/<name>.nfl"`.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn compile_default_runs_fusion() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // stderr has the applied-passes note (M5b: two passes in canonical order).
    assert!(
        stderr.contains("note: applied passes: eliminate_dropout, fuse_linear_relu"),
        "stderr missing applied-passes note:\n{stderr}"
    );

    // stdout has fused asm: inline fmax, no separate relu loop.
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout missing inline fmax:\n{stdout}"
    );
    assert!(
        !stdout.contains(".Lrelu_"),
        "stdout has separate relu loop (fusion did NOT apply):\n{stdout}"
    );
}

#[test]
fn compile_with_no_passes_skips_pipeline() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--no-passes",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: passes skipped (--no-passes)"),
        "stderr missing passes-skipped note:\n{stderr}"
    );
    // Successful skip mode does NOT emit the applied-passes note.
    assert!(
        !stderr.contains("note: applied passes:"),
        "stderr should not contain 'applied passes' when passes are skipped:\n{stderr}"
    );

    // Unfused asm: separate relu loop, no inline fmax.
    assert!(
        stdout.contains(".Lrelu_0_0:"),
        "stdout missing relu loop label (un-fused mode):\n{stdout}"
    );
    assert!(
        !stdout.contains("fmax    s0, s0, s4"),
        "stdout has inline fmax (fusion incorrectly applied in --no-passes mode):\n{stdout}"
    );
}

#[test]
fn compile_unknown_flag_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--frobnicate",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    // Strict substring: must mention BOTH the diagnostic kind and the
    // offending flag. A loose `|| contains("error:")` would pass for any
    // error path (parse error, missing file, unknown profile…), defeating
    // the test's purpose of pinning unknown-flag detection specifically.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown flag: --frobnicate"),
        "stderr missing unknown-flag error for '--frobnicate':\n{stderr}"
    );
}
