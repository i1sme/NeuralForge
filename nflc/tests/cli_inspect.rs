// SPDX-License-Identifier: Apache-2.0

//! CLI integration tests for `nflc inspect`.
//!
//! Mirror of `cli_compile.rs` — Cargo runs integration-test binaries
//! with cwd at the package root (`nflc/`), so paths to workspace-root
//! fixtures are written as `"../tests/fixtures/<name>.nfl"`.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn inspect_default_runs_pipeline_and_renders() {
    let output = Command::new(nflc_bin())
        .args([
            "inspect",
            "../tests/fixtures/tiny_mlp.nfl",
            "--profile",
            "arm64",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Stdout markers (format-stability safety net; full format covered
    // by goldens in M16 Task 5).
    assert!(
        stdout.contains("inspect-model TinyMLP"),
        "stdout missing inspect-model header:\n{stdout}"
    );
    assert!(
        stdout.contains("loc="),
        "stdout missing loc= row:\n{stdout}"
    );
    assert!(
        stdout.contains("passes applied:"),
        "stdout missing passes-applied header line:\n{stdout}"
    );

    // Stderr applied-passes note (mirrors compile's behaviour).
    assert!(
        stderr.contains("note: applied passes:"),
        "stderr missing applied-passes note:\n{stderr}"
    );
}

#[test]
fn inspect_no_passes_marks_skipped() {
    let output = Command::new(nflc_bin())
        .args([
            "inspect",
            "../tests/fixtures/tiny_mlp.nfl",
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
        stdout.contains("passes: skipped"),
        "stdout missing skipped marker:\n{stdout}"
    );
    assert!(
        stderr.contains("note: passes skipped"),
        "stderr missing passes-skipped note:\n{stderr}"
    );
}
