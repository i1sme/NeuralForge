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
