// SPDX-License-Identifier: Apache-2.0

//! Loop runner for tests/fixtures/negative/. Each .nfl file is loaded,
//! parsed, and (if parse succeeds) built into UIR. The test asserts
//! that *some* error fires; per-fixture asserts on the specific
//! BuildErrorKind / ShapeError live in the unit-test layer
//! (compiler/src/ir/tests.rs and compiler/src/parser/tests.rs).

use std::fs;
use std::path::Path;

#[test]
fn all_negative_fixtures_reject() {
    let dir = Path::new("../tests/fixtures/negative");
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("read fixtures/negative")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "nfl").unwrap_or(false))
        .collect();

    assert!(
        !entries.is_empty(),
        "fixtures/negative must contain at least one .nfl"
    );

    for entry in entries {
        let path = entry.path();
        let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
        // Try parse, then ir::build. EITHER may produce the expected
        // failure — we don't pin which.
        let parse_result = compiler::parse(&src);
        let combined = match parse_result {
            Err(_) => Err(()),
            Ok(ast) => compiler::ir::build(&ast).map(|_| ()).map_err(|_| ()),
        };
        assert!(
            combined.is_err(),
            "negative fixture {:?} unexpectedly accepted",
            path
        );
    }
}

/// M14 negative fixture: layernorm on rank-1 input must be rejected at IR
/// build with a ShapeMismatch error (detail contains "rank too low").
///
/// Mirrors the Softmax precedent: rank check fires before any profile sees
/// the node, so this test lives in compiler/tests/ (not profile-negative/).
#[test]
fn layernorm_rank_too_low_rejected_at_ir_build() {
    let src = fs::read_to_string("../tests/fixtures/negative/layernorm_rank_too_low.nfl")
        .expect("layernorm_rank_too_low.nfl fixture missing");

    let ast = compiler::parse(&src).expect("parse must succeed (syntax is valid)");
    let err = compiler::ir::build(&ast).expect_err("rank-1 layernorm must be rejected at IR build");

    let err_str = format!("{err}");
    assert!(
        err_str.contains("rank too low"),
        "expected error message to contain 'rank too low'; got: {err_str}"
    );
}
