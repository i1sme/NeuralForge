//! Pipeline-level tests for `compiler::passes`.

use super::{default_pipeline, run_pipeline, PassError, UirPass};
use crate::Uir;

/// Synthetic identity pass for testing the pipeline mechanics without
/// depending on any specific transformation.
struct IdentityPass {
    name: &'static str,
}

impl UirPass for IdentityPass {
    fn name(&self) -> &str {
        self.name
    }
    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        Ok(uir.clone())
    }
}

#[test]
fn default_pipeline_includes_fuse_linear_relu() {
    let pipeline = default_pipeline();
    let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
    assert!(
        names.contains(&"fuse_linear_relu"),
        "default_pipeline must include 'fuse_linear_relu'; got: {:?}",
        names
    );
}

#[test]
fn run_pipeline_threads_uir_through_passes() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let passes: Vec<Box<dyn UirPass>> = vec![
        Box::new(IdentityPass { name: "id_a" }),
        Box::new(IdentityPass { name: "id_b" }),
    ];

    let out = run_pipeline(&uir, &passes).expect("pipeline ok");
    // Identity passes preserve model count + node count.
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].nodes.len(), uir.models[0].nodes.len());
}

#[test]
fn empty_pipeline_returns_input_clone() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let out = run_pipeline(&uir, &[]).expect("empty pipeline ok");
    assert_eq!(out.models.len(), uir.models.len());
    assert_eq!(out.models[0].name, uir.models[0].name);
}

/// Synthetic always-failing pass: lets us verify pipeline halts on
/// `Err` and propagates the error unchanged. Without this test, a
/// future refactor of `run_pipeline` could accidentally swallow
/// errors and only Task 4's tests would (incidentally) catch it.
struct FailPass;

impl UirPass for FailPass {
    fn name(&self) -> &str {
        "fail"
    }
    fn run(&self, _uir: &Uir) -> Result<Uir, PassError> {
        Err(PassError::InvalidInput {
            pass: "fail".into(),
            reason: "synthetic".into(),
            span: crate::ast::Span::new(1, 1),
        })
    }
}

#[test]
fn pipeline_halts_on_first_error_and_propagates() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("ir::build");

    let passes: Vec<Box<dyn UirPass>> = vec![
        Box::new(FailPass),
        Box::new(IdentityPass {
            name: "should_not_run",
        }),
    ];
    let err = run_pipeline(&uir, &passes).expect_err("expected pipeline error");
    match err {
        PassError::InvalidInput { pass, reason, .. } => {
            assert_eq!(pass, "fail");
            assert_eq!(reason, "synthetic");
        }
    }
}
