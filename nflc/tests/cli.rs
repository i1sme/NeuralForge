// SPDX-License-Identifier: Apache-2.0

//! CLI smoke tests for `nflc compile`. First nflc-side tests in M9 —
//! they pin the `--profile <name>` dispatch and the help-text wording.

use std::process::Command;

fn nflc_path() -> std::path::PathBuf {
    // Tests run from the nflc crate root. The compiled binary lands in
    // CARGO_BIN_EXE_<name> when cargo runs the integration test.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_nflc"))
}

#[test]
fn compile_x86_64_emits_no_underscore_prefix_and_call_expf_plt() {
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "x86_64"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        output.status.success(),
        "nflc compile --profile x86_64 failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let asm = String::from_utf8(output.stdout).expect("asm utf-8");
    assert!(
        asm.contains("nfl_forward_Classifier:"),
        "asm missing un-prefixed function label:\n{asm}"
    );
    assert!(
        !asm.contains("_nfl_forward_Classifier"),
        "x86_64 asm must not have underscore-prefixed label:\n{asm}"
    );
    assert!(
        asm.contains("call    expf@PLT"),
        "x86_64 asm with softmax must call expf@PLT:\n{asm}"
    );
}

#[test]
fn compile_unknown_profile_exits_failure_with_supported_list() {
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "foo"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        !output.status.success(),
        "expected failure exit for unknown profile"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown profile 'foo'"),
        "missing 'unknown profile' phrase:\n{stderr}"
    );
    assert!(
        stderr.contains("supported: arm64, x86_64"),
        "supported list must include both profiles:\n{stderr}"
    );
}

#[test]
fn compile_arm64_still_emits_underscore_prefix_and_bl_expf() {
    // Regression guard for the dispatch refactor: arm64 path must
    // still produce Mach-O-shaped output.
    let fixture = "../tests/fixtures/classifier.nfl";
    let output = Command::new(nflc_path())
        .args(["compile", fixture, "--profile", "arm64"])
        .output()
        .expect("nflc invocation failed");
    assert!(
        output.status.success(),
        "nflc compile --profile arm64 failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let asm = String::from_utf8(output.stdout).expect("asm utf-8");
    assert!(
        asm.contains("_nfl_forward_Classifier:"),
        "arm64 asm missing underscore-prefixed function label:\n{asm}"
    );
    assert!(
        asm.contains("bl      _expf"),
        "arm64 asm with softmax must call _expf:\n{asm}"
    );
}
