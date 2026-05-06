// SPDX-License-Identifier: AGPL-3.0-only

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

#[test]
fn compile_with_passes_filter_runs_only_selected() {
    // --passes fuse_linear_relu against m4_linear_relu.nfl (which has
    // no dropout). The filter exercise is purely about pipeline
    // selection: stderr should show only the named pass; asm should
    // still contain inline fmax (since FuseLinearRelu is in the
    // filtered set).
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_relu",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: applied passes: fuse_linear_relu"),
        "stderr should show only fuse_linear_relu in applied list:\n{stderr}"
    );
    // Specifically: the canonical two-pass form (which would start
    // 'note: applied passes: eliminate_dropout, fuse_linear_relu') must
    // not appear when the filter retained only fuse_linear_relu. We pin
    // the exact prefix `note: applied passes: eliminate_dropout` rather
    // than the bare name `eliminate_dropout` because the latter could
    // also match unrelated diagnostic text in future stderr additions.
    assert!(
        !stderr.contains("note: applied passes: eliminate_dropout"),
        "stderr applied-passes note should NOT start with eliminate_dropout when filter retained only fuse_linear_relu:\n{stderr}"
    );
    // Fusion still applied.
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout should have inline fmax (fusion in filtered set):\n{stdout}"
    );
}

#[test]
fn compile_with_passes_unknown_name_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "foo",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Strict: must mention BOTH the offending name AND an "available:"
    // listing. The exact contents of the available list are dynamic
    // (M6+ may add passes); substring match on "available:" keeps the
    // test resilient.
    assert!(
        stderr.contains("unknown pass 'foo'"),
        "stderr missing unknown-pass error for 'foo':\n{stderr}"
    );
    assert!(
        stderr.contains("available:"),
        "stderr missing 'available:' substring (dynamic list):\n{stderr}"
    );
}

#[test]
fn compile_with_passes_order_warning() {
    // User writes the two passes in REVERSE of canonical order.
    // CLI should still produce correct asm (canonical order applied)
    // AND emit a divergence note so the user knows their order was
    // overridden.
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_relu,eliminate_dropout",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // BOTH notes must appear: applied-passes (in canonical order) AND
    // the divergence warning. They are separate eprintln! calls per
    // spec §9.3 — substring checks are independent.
    assert!(
        stderr.contains("note: applied passes: eliminate_dropout, fuse_linear_relu"),
        "stderr missing canonical-order applied-passes note:\n{stderr}"
    );
    assert!(
        stderr.contains("user-specified order ignored"),
        "stderr missing order-divergence warning:\n{stderr}"
    );
    // Stdout still has the expected fused-asm shape (canonical order
    // produces the same asm as the no-flag default).
    assert!(
        stdout.contains("fmax    s0, s0, s4"),
        "stdout missing inline fmax (canonical order should still fuse):\n{stdout}"
    );
}

#[test]
fn compile_no_passes_and_passes_rejected() {
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--no-passes",
            "--passes",
            "fuse_linear_relu",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "stderr missing mutually-exclusive error:\n{stderr}"
    );
}

#[test]
fn compile_with_passes_filter_only_fuse_linear_softmax_runs() {
    // --passes fuse_linear_softmax against classifier.nfl, which ends
    // with `linear[output] -> softmax`.  Only fuse_linear_softmax runs;
    // eliminate_dropout and fuse_linear_relu must be absent from the
    // applied-passes note.  Asm confirms the RowWise fused tail was
    // emitted (.Lfsmx_* labels, bl _expf) and the standalone softmax
    // path (.Lsm_*) was NOT taken.
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/classifier.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_softmax",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(output.status.success(), "exit failure: {:?}", output);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("note: applied passes: fuse_linear_softmax"),
        "stderr should announce only fuse_linear_softmax:\n{stderr}"
    );
    assert!(
        !stderr.contains("eliminate_dropout"),
        "stderr should NOT mention eliminate_dropout under filter:\n{stderr}"
    );
    assert!(
        !stderr.contains("fuse_linear_relu"),
        "stderr should NOT mention fuse_linear_relu under filter:\n{stderr}"
    );

    // Fused RowWise tail: bl _expf and .Lfsmx_* labels must be present.
    assert!(
        stdout.contains("bl      _expf"),
        "fused asm should call bl _expf inside the RowWise softmax tail:\n{stdout}"
    );
    assert!(
        stdout.contains(".Lfsmx_"),
        "fused asm should use .Lfsmx_* labels for the inlined softmax tail:\n{stdout}"
    );
    // No standalone softmax path should appear (fusion replaced it).
    assert!(
        !stdout.contains(".Lsm_"),
        "with fuse_linear_softmax applied, no standalone .Lsm_* label should appear:\n{stdout}"
    );
}

#[test]
fn compile_with_passes_duplicate_name_rejected() {
    // Companion test for the duplicate-name guard in parse_compile_args
    // (--passes a,a → error "pass 'a' specified more than once").
    // Spec §11.4 didn't list this case, but the validation code exists
    // and shipping it without a smoke test creates a documentation gap
    // for future readers (holistic-review N-tier finding pre-merge).
    let output = Command::new(nflc_bin())
        .args([
            "compile",
            "../tests/fixtures/m4_linear_relu.nfl",
            "--profile",
            "arm64",
            "--passes",
            "fuse_linear_relu,fuse_linear_relu",
        ])
        .output()
        .expect("failed to run nflc");

    assert!(!output.status.success(), "expected failure exit");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("specified more than once"),
        "stderr missing duplicate-pass error:\n{stderr}"
    );
    assert!(
        stderr.contains("fuse_linear_relu"),
        "stderr should name the duplicated pass:\n{stderr}"
    );
}
