// SPDX-License-Identifier: AGPL-3.0-only

//! CLI integration tests for `nflc parse` UIR rendering modes.

use std::process::Command;

fn nflc_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nflc")
}

#[test]
fn parse_with_uir_verbose_renders_summary_and_extern_math() {
    let output = Command::new(nflc_bin())
        .args(["parse", "../tests/fixtures/classifier.nfl", "--uir-verbose"])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("uir-verbose summary"),
        "stdout missing verbose header:\n{stdout}"
    );
    assert!(
        stdout.contains("calls-extern-math:"),
        "stdout missing calls-extern-math line:\n{stdout}"
    );
    assert!(
        stdout.contains("node count:"),
        "stdout missing node count line:\n{stdout}"
    );
}

#[test]
fn parse_uir_and_uir_verbose_are_mutually_exclusive() {
    let output = Command::new(nflc_bin())
        .args([
            "parse",
            "../tests/fixtures/classifier.nfl",
            "--uir",
            "--uir-verbose",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(
        !output.status.success(),
        "expected failure exit but got success; full output: {:?}",
        output
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--uir") && stderr.contains("--uir-verbose"),
        "stderr must mention both flags in the mutual-exclusion error:\n{stderr}"
    );
}
