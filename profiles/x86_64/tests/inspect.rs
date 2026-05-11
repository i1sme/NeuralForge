// SPDX-License-Identifier: Apache-2.0

//! Golden-snapshot tests for `X86_64Profile::inspect()` rendering.
//! Mirror of profiles/arm64/tests/inspect.rs.

use inspect_render::{render_inspection, RenderHeader};
use profile_api::Profile;
use profiles_x86_64::X86_64Profile;
use std::path::PathBuf;

/// On-disk path used to read the fixture.
fn read_path(name: &str) -> PathBuf {
    PathBuf::from(format!("../../tests/fixtures/{}.nfl", name))
}

/// Stable workspace-relative path for the rendered header — keeps
/// goldens cwd-independent.
fn header_path(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/fixtures/{}.nfl", name))
}

fn expected_path(name: &str) -> PathBuf {
    PathBuf::from(format!("tests/inspect/{}.expected.txt", name))
}

fn run_and_render(name: &str) -> String {
    let read = read_path(name);
    let source =
        std::fs::read_to_string(&read).unwrap_or_else(|e| panic!("read {}: {}", read.display(), e));
    let ast = compiler::parse(&source).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let pipeline = compiler::passes::default_pipeline();
    let post_pass = compiler::passes::run_pipeline(&uir, &pipeline).expect("run_pipeline");
    let insp = X86_64Profile.inspect(&post_pass).expect("inspect");

    let pass_names: Vec<String> = pipeline.iter().map(|p| p.name().to_owned()).collect();
    let pass_refs: Vec<&str> = pass_names.iter().map(String::as_str).collect();
    let header_path = header_path(name);
    let header = RenderHeader {
        source_path: &header_path,
        profile: "x86_64",
        applied_passes: Some(&pass_refs),
    };
    render_inspection(&insp, header)
}

fn assert_golden(name: &str) {
    let actual = run_and_render(name);
    let expected_path = expected_path(name);
    let expected = std::fs::read_to_string(&expected_path).unwrap_or_else(|e| {
        panic!(
            "read expected file {}: {}\n\nIf this is the first run, regenerate with:\n  cargo run -p nflc -- inspect tests/fixtures/{}.nfl --profile x86_64 > {}",
            expected_path.display(),
            e,
            name,
            expected_path.display()
        )
    });
    if actual != expected {
        panic!(
            "golden mismatch for {} (x86_64).\n--- expected ---\n{}\n--- actual ---\n{}\n",
            name, expected, actual
        );
    }
}

#[test]
fn golden_tiny_mlp_x86_64() {
    assert_golden("tiny_mlp");
}

#[test]
fn golden_transformer_block_x86_64() {
    assert_golden("transformer_block");
}

#[test]
fn golden_self_attention_x86_64() {
    assert_golden("self_attention");
}

#[test]
fn golden_dropout_only_x86_64() {
    assert_golden("dropout_only");
}
