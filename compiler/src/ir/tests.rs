// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the IR module.

use super::stdlib::*;
use super::types::{AttrValue, OpAttr, Shape};

#[test]
fn resolve_known_ops() {
    assert_eq!(resolve("linear"), Some(StdOp::Linear));
    assert_eq!(resolve("relu"), Some(StdOp::Relu));
    assert_eq!(resolve("dropout"), Some(StdOp::Dropout));
    assert_eq!(resolve("softmax"), Some(StdOp::Softmax));
}

#[test]
fn resolve_unknown_op_returns_none() {
    assert_eq!(resolve("foo"), None);
    assert_eq!(resolve("Linear"), None); // case-sensitive
    assert_eq!(resolve(""), None);
}

#[test]
fn signature_linear_has_one_positional_one_named() {
    let s = signature(StdOp::Linear);
    assert_eq!(s.positional.len(), 1);
    assert_eq!(s.positional[0].name, "out_dim");
    assert_eq!(s.positional[0].ty, ArgType::Integer);
    assert!(s.positional[0].required);
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "bias");
    assert_eq!(s.named[0].ty, ArgType::Symbol);
    assert!(!s.named[0].required);
}

#[test]
fn signature_dropout_has_one_named_required() {
    let s = signature(StdOp::Dropout);
    assert!(s.positional.is_empty());
    assert_eq!(s.named.len(), 1);
    assert_eq!(s.named[0].name, "rate");
    assert_eq!(s.named[0].ty, ArgType::Float);
    assert!(s.named[0].required);
}

#[test]
fn signature_relu_and_softmax_are_empty() {
    let r = signature(StdOp::Relu);
    assert!(r.positional.is_empty());
    assert!(r.named.is_empty());
    let s = signature(StdOp::Softmax);
    assert!(s.positional.is_empty());
    assert!(s.named.is_empty());
}

#[test]
fn infer_linear_output_shape() {
    let input = Shape(vec![8, 4]);
    let attrs = vec![OpAttr {
        name: "out_dim".into(),
        value: AttrValue::Integer(2),
    }];
    let out = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap();
    assert_eq!(out.0, vec![8, 2]);
}

#[test]
fn infer_linear_with_wrong_rank_input() {
    let input = Shape(vec![8]); // rank 1, linear expects rank 2
    let attrs = vec![OpAttr {
        name: "out_dim".into(),
        value: AttrValue::Integer(2),
    }];
    let err = infer_output_shape(StdOp::Linear, &[input], &attrs).unwrap_err();
    matches!(
        err,
        ShapeError::WrongRank {
            expected: 2,
            actual: 1,
            ..
        }
    );
}

#[test]
fn infer_relu_preserves_shape() {
    let input = Shape(vec![8, 2]);
    let out = infer_output_shape(StdOp::Relu, std::slice::from_ref(&input), &[]).unwrap();
    assert_eq!(out, input);
}

#[test]
fn infer_softmax_and_dropout_preserve_shape() {
    let input = Shape(vec![3, 7, 2]);
    assert_eq!(
        infer_output_shape(StdOp::Softmax, std::slice::from_ref(&input), &[]).unwrap(),
        input
    );
    assert_eq!(
        infer_output_shape(StdOp::Dropout, std::slice::from_ref(&input), &[]).unwrap(),
        input
    );
}

use super::build::resolve_type;
use super::error::BuildErrorKind;
use crate::ast::{Dim, Span, TypeExpr};
use std::collections::HashMap;

fn span() -> Span {
    Span::new(1, 1)
}

#[test]
fn resolve_type_all_integer_dims() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Integer(8), Dim::Integer(4)],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_symbolic_dim_with_lookup() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("batch".into()), Dim::Integer(4)],
        span: span(),
    };
    let mut params: HashMap<&str, u64> = HashMap::new();
    params.insert("batch", 8);
    let shape = resolve_type(&ty, &params).unwrap();
    assert_eq!(shape.0, vec![8, 4]);
}

#[test]
fn resolve_type_unknown_dim_errors() {
    let ty = TypeExpr {
        name: "Tensor".into(),
        dims: vec![Dim::Symbol("zzz".into())],
        span: span(),
    };
    let params: HashMap<&str, u64> = HashMap::new();
    let err = resolve_type(&ty, &params).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::UnknownDim { .. }));
}

use super::build::resolve_args;
use crate::ast::{ArgValue, OpArg};

#[test]
fn resolve_args_one_positional_integer() {
    let args = vec![OpArg::Positional(ArgValue::Integer(512))];
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let (operands, attrs) =
        resolve_args(StdOp::Linear, &args, &HashMap::new(), &env, span()).unwrap();
    assert!(operands.is_empty());
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "out_dim");
    assert_eq!(attrs[0].value, AttrValue::Integer(512));
}

#[test]
fn resolve_args_missing_required_positional() {
    let args: Vec<OpArg> = vec![]; // linear needs out_dim
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let err = resolve_args(StdOp::Linear, &args, &HashMap::new(), &env, span()).unwrap_err();
    assert!(matches!(
        err.kind,
        BuildErrorKind::ArgCountMismatch { .. } | BuildErrorKind::MissingRequiredArg { .. }
    ));
}

#[test]
fn resolve_args_extra_positional() {
    let args = vec![
        OpArg::Positional(ArgValue::Integer(2)),
        OpArg::Positional(ArgValue::Integer(3)),
    ];
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let err = resolve_args(StdOp::Linear, &args, &HashMap::new(), &env, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgCountMismatch { .. }));
}

#[test]
fn resolve_args_type_mismatch() {
    let args = vec![OpArg::Positional(ArgValue::Float(2.5))]; // out_dim wants Integer
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let err = resolve_args(StdOp::Linear, &args, &HashMap::new(), &env, span()).unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::ArgTypeMismatch { .. }));
}

#[test]
fn resolve_args_named_only_dropout() {
    let args = vec![OpArg::Named {
        name: "rate".into(),
        value: ArgValue::Float(0.2),
    }];
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let (operands, attrs) =
        resolve_args(StdOp::Dropout, &args, &HashMap::new(), &env, span()).unwrap();
    assert!(operands.is_empty());
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "rate");
    assert_eq!(attrs[0].value, AttrValue::Float(0.2));
}

#[test]
fn resolve_args_symbol_resolves_against_params() {
    // linear[output] where `output` is a model_param (e.g., output=10).
    // Should resolve Symbol("output") → Integer(10) and pass type-check.
    let args = vec![OpArg::Positional(ArgValue::Symbol("output".into()))];
    let mut params: HashMap<&str, u64> = HashMap::new();
    params.insert("output", 10);
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let (operands, attrs) = resolve_args(StdOp::Linear, &args, &params, &env, span()).unwrap();
    assert!(operands.is_empty());
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "out_dim");
    assert_eq!(attrs[0].value, AttrValue::Integer(10));
}

#[test]
fn resolve_args_symbol_not_in_params_stays_symbol() {
    // bias=true where `true` is NOT a param: stays as Symbol, passes Symbol slot.
    let args = vec![
        OpArg::Positional(ArgValue::Integer(16)),
        OpArg::Named {
            name: "bias".into(),
            value: ArgValue::Symbol("true".into()),
        },
    ];
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let (operands, attrs) =
        resolve_args(StdOp::Linear, &args, &HashMap::new(), &env, span()).unwrap();
    assert!(operands.is_empty());
    assert_eq!(attrs.len(), 2);
    assert_eq!(attrs[1].name, "bias");
    assert_eq!(attrs[1].value, AttrValue::Symbol("true".into()));
}

use super::build::build_op;
use super::types::{Node, NodeKind, Type};
use crate::ast::Operation;

fn input_node(shape: Vec<u64>) -> Node {
    Node {
        kind: NodeKind::Input { name: "x".into() },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(shape),
        },
        source_span: span(),
    }
}

#[test]
fn build_op_linear_produces_correct_node() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "linear".into(),
        args: vec![OpArg::Positional(ArgValue::Integer(2))],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let id = build_op(
        &op_ast,
        0,
        &input_shape,
        &HashMap::new(),
        &env,
        &mut out_nodes,
    )
    .unwrap();
    assert_eq!(id, 1);
    assert_eq!(out_nodes.len(), 2);
    let NodeKind::Op {
        op,
        operands,
        attrs,
        ..
    } = &out_nodes[1].kind
    else {
        panic!("expected Op node");
    };
    assert_eq!(*op, StdOp::Linear);
    assert_eq!(operands, &[0]);
    assert_eq!(attrs[0].value, AttrValue::Integer(2));
    assert_eq!(out_nodes[1].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_op_softmax_preserves_input_shape() {
    let nodes = vec![input_node(vec![8, 2])];
    let op_ast = Operation {
        name: "softmax".into(),
        args: vec![],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let id = build_op(
        &op_ast,
        0,
        &input_shape,
        &HashMap::new(),
        &env,
        &mut out_nodes,
    )
    .unwrap();
    assert_eq!(out_nodes[id].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_op_unknown_op_errors() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "mystery".into(),
        args: vec![],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let err = build_op(
        &op_ast,
        0,
        &input_shape,
        &HashMap::new(),
        &env,
        &mut out_nodes,
    )
    .unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::UnknownOp { .. }));
}

use crate::lexer::lex;
use crate::parser;

fn parse_to_ast(src: &str) -> crate::ast::NflSource {
    let tokens = lex(src).expect("lex");
    let leaked: &'static [crate::lexer::Token] = Box::leak(tokens.into_boxed_slice());
    let mut p = parser::Parser::new(leaked);
    parser::parse_nfl_source(&mut p).expect("parse")
}

#[test]
fn build_tiny_mlp_minimal() {
    let src = "model X [batch=8]:\n    x: Tensor[batch, 4]\n    x -> linear[2] -> softmax\n";
    let ast = parse_to_ast(src);
    let uir = super::build(&ast).unwrap();
    assert_eq!(uir.models.len(), 1);
    let m = &uir.models[0];
    assert_eq!(m.name, "X");
    assert_eq!(m.nodes.len(), 3);
    assert_eq!(m.inputs, vec![0]);
    assert_eq!(m.output, 2);
    assert_eq!(m.nodes[0].ty.shape.0, vec![8, 4]);
    assert_eq!(m.nodes[1].ty.shape.0, vec![8, 2]);
    assert_eq!(m.nodes[2].ty.shape.0, vec![8, 2]);
}

#[test]
fn build_model_with_no_pipeline_errors() {
    let src = "model X [a=1]:\n    x: Tensor[a, 1]\n";
    let ast = parse_to_ast(src);
    let err = super::build(&ast).unwrap_err();
    assert!(matches!(
        err.kind,
        BuildErrorKind::ModelHasNoPipeline { .. }
    ));
}

use super::stdlib::{validate_attrs, AttrError};

#[test]
fn validate_attrs_dropout_in_range_succeeds() {
    let attrs = vec![OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(0.0),
    }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
    let attrs = vec![OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(0.5),
    }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
    let attrs = vec![OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(1.0),
    }];
    assert!(validate_attrs(StdOp::Dropout, &attrs).is_ok());
}

#[test]
fn validate_attrs_dropout_out_of_range_errors() {
    let attrs = vec![OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(1.5),
    }];
    let err = validate_attrs(StdOp::Dropout, &attrs).unwrap_err();
    assert!(matches!(err, AttrError::OutOfRange { name: "rate", .. }));
    let attrs = vec![OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(-0.1),
    }];
    let err = validate_attrs(StdOp::Dropout, &attrs).unwrap_err();
    assert!(matches!(err, AttrError::OutOfRange { name: "rate", .. }));
}

#[test]
fn validate_attrs_dropout_missing_rate_errors() {
    let err = validate_attrs(StdOp::Dropout, &[]).unwrap_err();
    assert!(matches!(err, AttrError::MissingAttr { name: "rate" }));
}

#[test]
fn validate_attrs_other_ops_no_op() {
    assert!(validate_attrs(StdOp::Linear, &[]).is_ok());
    assert!(validate_attrs(StdOp::Relu, &[]).is_ok());
    assert!(validate_attrs(StdOp::Softmax, &[]).is_ok());
}

#[test]
fn attr_error_displays_human_message() {
    let err = AttrError::OutOfRange {
        name: "rate",
        value: 1.5,
        min: 0.0,
        max: 1.0,
    };
    let msg = format!("{err}");
    assert!(msg.contains("rate") && msg.contains("1.5"), "got: {msg}");
}

#[test]
fn build_op_dropout_out_of_range_errors() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "dropout".into(),
        args: vec![OpArg::Named {
            name: "rate".into(),
            value: ArgValue::Float(1.5),
        }],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let err = build_op(
        &op_ast,
        0,
        &input_shape,
        &HashMap::new(),
        &env,
        &mut out_nodes,
    )
    .unwrap_err();
    assert!(matches!(err.kind, BuildErrorKind::InvalidAttrValue { .. }));
}

#[test]
fn build_op_dropout_in_range_succeeds() {
    let nodes = vec![input_node(vec![8, 4])];
    let op_ast = Operation {
        name: "dropout".into(),
        args: vec![OpArg::Named {
            name: "rate".into(),
            value: ArgValue::Float(0.5),
        }],
        span: span(),
    };
    let mut out_nodes = nodes.clone();
    let input_shape = nodes[0].ty.shape.clone();
    let env: HashMap<String, super::types::NodeId> = HashMap::new();
    let id = build_op(
        &op_ast,
        0,
        &input_shape,
        &HashMap::new(),
        &env,
        &mut out_nodes,
    )
    .unwrap();
    assert_eq!(out_nodes[id].ty.shape.0, vec![8, 4]);
}

#[test]
fn shape_displays_as_tensor_with_dims() {
    let s = Shape(vec![32, 784]);
    assert_eq!(format!("{}", s), "Tensor[32, 784]");
}

#[test]
fn attrvalue_displays_each_variant() {
    assert_eq!(format!("{}", AttrValue::Integer(42)), "42");
    assert_eq!(format!("{}", AttrValue::Float(0.5)), "0.5");
    assert_eq!(format!("{}", AttrValue::Symbol("true".into())), "true");
}

#[test]
fn opattr_displays_name_equals_value() {
    let a = OpAttr {
        name: "out_dim".into(),
        value: AttrValue::Integer(512),
    };
    assert_eq!(format!("{}", a), "out_dim=512");
    let b = OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(0.2),
    };
    assert_eq!(format!("{}", b), "rate=0.2");
}

#[test]
fn stdop_displays_lowercase_name() {
    assert_eq!(format!("{}", StdOp::Linear), "linear");
    assert_eq!(format!("{}", StdOp::Relu), "relu");
    assert_eq!(format!("{}", StdOp::Dropout), "dropout");
    assert_eq!(format!("{}", StdOp::Softmax), "softmax");
}

#[test]
fn duplicate_model_name_at_build_time() {
    let src = "model M [b=2]:\n    x: Tensor[b, 3]\n    x -> linear[2]\n\
               model M [b=2]:\n    y: Tensor[b, 3]\n    y -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let err = crate::ir::build(&ast).expect_err("must fail");
    // err.line/col point at the REDEFINITION (line 4 of the source).
    assert_eq!(err.line, 4, "err.line should point at the redefinition");
    match err.kind {
        crate::ir::BuildErrorKind::DuplicateModelName {
            ref name,
            first_span,
        } => {
            assert_eq!(name, "M");
            // first_span points at the ORIGINAL definition (line 1).
            assert_eq!(
                first_span.line, 1,
                "first_span should point at the original"
            );
        }
        _ => panic!("expected DuplicateModelName, got {:?}", err.kind),
    }
}

#[test]
fn display_for_postop_lowercase() {
    use crate::ir::PostOp;
    assert_eq!(format!("{}", PostOp::Relu), "relu");
}

#[test]
fn post_op_softmax_row_displays_as_softmax_row() {
    use crate::ir::PostOp;
    assert_eq!(format!("{}", PostOp::SoftmaxRow), "softmax_row");
}

#[test]
fn display_for_node_renders_fused_post_ops_when_present() {
    use crate::ast::Span;
    use crate::ir::stdlib::StdOp;
    use crate::ir::types::{AttrValue, Node, NodeKind, OpAttr, PostOp, Shape, Type};

    let n = Node {
        kind: NodeKind::Op {
            op: StdOp::Linear,
            operands: vec![0],
            attrs: vec![OpAttr {
                name: "out_dim".into(),
                value: AttrValue::Integer(2),
            }],
            fused_post_ops: vec![PostOp::Relu],
        },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(vec![8, 2]),
        },
        source_span: Span::new(1, 1),
    };
    let rendered = format!("{}", n);
    assert!(rendered.contains("linear"));
    assert!(rendered.contains("operands=[n0]"));
    assert!(rendered.contains("attrs=[out_dim=2]"));
    assert!(rendered.contains("fused=[relu]"));
}

#[test]
fn display_for_node_omits_fused_when_empty() {
    use crate::ast::Span;
    use crate::ir::stdlib::StdOp;
    use crate::ir::types::{Node, NodeKind, Shape, Type};

    let n = Node {
        kind: NodeKind::Op {
            op: StdOp::Linear,
            operands: vec![0],
            attrs: vec![],
            fused_post_ops: vec![],
        },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(vec![8, 2]),
        },
        source_span: Span::new(1, 1),
    };
    let rendered = format!("{}", n);
    assert_eq!(
        rendered.find("fused"),
        None,
        "empty fused_post_ops should NOT render 'fused' substring; got: {rendered}"
    );
}

#[test]
fn calls_extern_math_true_for_standalone_softmax() {
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    assert!(uir.calls_extern_math());
    assert!(uir.models[0].calls_extern_math());
}

#[test]
fn calls_extern_math_false_for_linear_only() {
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[2]\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    assert!(!uir.calls_extern_math());
    assert!(!uir.models[0].calls_extern_math());
}

#[test]
fn calls_extern_math_true_for_fused_softmax_row() {
    // After default pipeline runs, linear→softmax fuses to
    // linear with PostOp::SoftmaxRow. Predicate must follow the fusion.
    let src = "model M [b=2]:\n    x: Tensor[b, 4]\n    x -> linear[3] -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let fused =
        crate::passes::run_pipeline(&uir, &crate::passes::default_pipeline()).expect("pipeline");
    // Sanity: standalone softmax is gone, replaced by fused PostOp.
    let has_standalone_softmax = fused.models[0].nodes.iter().any(|n| match &n.kind {
        crate::ir::types::NodeKind::Op { op, .. } => {
            matches!(op, crate::ir::stdlib::StdOp::Softmax)
        }
        _ => false,
    });
    assert!(
        !has_standalone_softmax,
        "fusion should have removed standalone softmax"
    );
    assert!(fused.calls_extern_math());
}

#[test]
fn verbose_uir_snapshot_matches_expected_format() {
    use crate::ir::types::VerboseUir;

    // Pre-pass UIR — no run_pipeline. nflc parse --uir-verbose is the
    // parse subcommand, not compile, so the rendered UIR reflects
    // un-fused operations.
    let src = "model Demo [b=2, k=4]:\n    x: Tensor[b, k]\n    x -> linear[3] -> softmax\n";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");

    let rendered = format!("{}", VerboseUir(&uir));

    // Pin key annotations rather than exact whitespace — Display impl
    // for Node has its own padding that we mirror, but exact char
    // counts are fragile. Assert the structural elements.
    assert!(
        rendered.contains("uir-verbose summary"),
        "missing top-level header:\n{rendered}"
    );
    assert!(rendered.contains("models: 1"), "missing models count");
    assert!(rendered.contains("total nodes: 3"), "missing total nodes");
    assert!(
        rendered.contains("calls-extern-math: yes"),
        "missing top-level extern-math line"
    );
    assert!(rendered.contains("uir-model Demo"), "missing model header");
    assert!(rendered.contains("inputs: [n0]"), "missing inputs");
    assert!(rendered.contains("output: n2"), "missing output");
    assert!(
        rendered.contains("node count: 3"),
        "missing per-model node count"
    );

    // Model has 3 nodes: input n0, linear n1, softmax n2.
    assert!(rendered.contains("n0:"), "missing n0 line");
    assert!(rendered.contains("n1:"), "missing n1 line");
    assert!(rendered.contains("n2:"), "missing n2 line");
    assert!(rendered.contains("linear"), "missing linear op");
    assert!(rendered.contains("softmax"), "missing softmax op");

    // No fusion — pre-pass UIR — so no `-> fused:` line should appear.
    assert!(
        !rendered.contains("-> fused:"),
        "pre-pass UIR must not have fused post-ops"
    );
}

#[test]
fn matmul_resolves_via_stdlib() {
    use crate::ir::stdlib::{resolve, StdOp};
    assert_eq!(resolve("matmul"), Some(StdOp::Matmul));
    assert_eq!(format!("{}", StdOp::Matmul), "matmul");
}

#[test]
fn matmul_2d_shape_inference_no_transpose() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![4, 8]);
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &[]).expect("infer");
    assert_eq!(out.0, vec![2, 8]);
}

#[test]
fn matmul_2d_shape_inference_transpose_b() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let a = Shape(vec![2, 4]);
    // transpose_b=true means b is logically [N, K] → [8, 4].
    let b = Shape(vec![8, 4]);
    let attrs = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &attrs).expect("infer");
    assert_eq!(out.0, vec![2, 8]);
}

#[test]
fn matmul_4d_shape_inference_no_transpose() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4, 16, 8]);
    let b = Shape(vec![2, 4, 8, 16]);
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &[]).expect("infer");
    assert_eq!(out.0, vec![2, 4, 16, 16]);
}

#[test]
fn matmul_4d_shape_inference_transpose_b() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let a = Shape(vec![2, 4, 16, 16]);
    // transpose_b=true → b interpreted as [..., N, K] = [2, 4, 16, 16].
    let b = Shape(vec![2, 4, 16, 16]);
    let attrs = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let out = infer_output_shape(StdOp::Matmul, &[a, b], &attrs).expect("infer");
    assert_eq!(out.0, vec![2, 4, 16, 16]);
}

#[test]
fn matmul_leading_dim_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4, 16, 8]);
    let b = Shape(vec![2, 5, 8, 16]); // heads dim 4 vs 5 — strict mismatch
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::LeadingDimMismatch {
                dim_index: 1,
                lhs: 4,
                rhs: 5
            }
        ),
        "unexpected error: {:?}",
        err
    );
}

#[test]
fn matmul_inner_dim_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![5, 8]); // K=4 vs K=5
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::InnerDimMismatch {
                lhs_k: 4,
                rhs_k: 5,
                transpose_b: false,
            }
        ),
        "unexpected error: {:?}",
        err
    );
}

#[test]
fn matmul_rank_mismatch_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let b = Shape(vec![2, 4, 4, 8]);
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(err, ShapeError::RankMismatch { lhs: 2, rhs: 4 }),
        "unexpected error: {:?}",
        err
    );
}

#[test]
fn matmul_rank_too_low_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![4]);
    let b = Shape(vec![4]);
    let err = infer_output_shape(StdOp::Matmul, &[a, b], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::RankTooLow {
                required: 2,
                actual: 1
            }
        ),
        "unexpected error: {:?}",
        err
    );
}

#[test]
fn matmul_wrong_input_count_errors() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let a = Shape(vec![2, 4]);
    let err = infer_output_shape(StdOp::Matmul, &[a], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::WrongInputCount {
                expected: 2,
                actual: 1
            }
        ),
        "unexpected error: {:?}",
        err
    );
}

#[test]
fn transpose_b_true_recognised() {
    use crate::ir::stdlib::matmul_transpose_b;
    use crate::ir::types::{AttrValue, OpAttr};

    let attrs_true = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    let attrs_false = vec![OpAttr {
        name: "transpose_b".to_string(),
        value: AttrValue::Symbol("false".to_string()),
    }];
    let attrs_empty: Vec<OpAttr> = vec![];

    assert!(matmul_transpose_b(&attrs_true));
    assert!(!matmul_transpose_b(&attrs_false));
    assert!(!matmul_transpose_b(&attrs_empty)); // default=false when omitted
}

#[test]
fn mul_scalar_resolves() {
    use crate::ir::stdlib::{resolve, StdOp};
    assert_eq!(resolve("mul_scalar"), Some(StdOp::MulScalar));
    assert_eq!(format!("{}", StdOp::MulScalar), "mul_scalar");
}

#[test]
fn mul_scalar_preserves_shape() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::{AttrValue, OpAttr, Shape};
    let input = Shape(vec![2, 4, 16, 16]);
    let attrs = vec![OpAttr {
        name: "value".to_string(),
        value: AttrValue::Float(0.25),
    }];
    let out =
        infer_output_shape(StdOp::MulScalar, std::slice::from_ref(&input), &attrs).expect("infer");
    assert_eq!(out.0, input.0);
}

#[test]
fn mul_scalar_signature_requires_float_positional() {
    use crate::ir::stdlib::{signature, ArgType, StdOp};
    let sig = signature(StdOp::MulScalar);
    assert_eq!(sig.positional.len(), 1);
    assert_eq!(sig.positional[0].name, "value");
    assert!(matches!(sig.positional[0].ty, ArgType::Float));
    assert!(sig.positional[0].required);
    assert_eq!(sig.named.len(), 0);
}

#[test]
fn named_pipeline_shape_match_succeeds() {
    // Declared shape matches the pipeline's actual output shape.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 4] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let model = &uir.models[0];
    // Output is `y` (the last/only named pipeline).
    let out_id = model.output;
    assert_eq!(model.nodes[out_id].ty.shape.0, vec![2, 4]);
}

#[test]
fn named_pipeline_shape_mismatch_errors() {
    // Declared `Tensor[batch, 8]` but `relu` preserves shape, so actual
    // is `Tensor[batch, 4]`. Build must fail with DeclaredShapeMismatch.
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, 8] = x -> relu
";
    let ast = crate::parse(src).expect("parse");
    let err = crate::ir::build(&ast).unwrap_err();
    assert!(
        matches!(
            err.kind,
            crate::ir::error::BuildErrorKind::DeclaredShapeMismatch { .. }
        ),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
fn tensor_arg_resolves_from_env() {
    // The `x` positional arg in matmul[x] resolves against env to the
    // input variable's NodeId. The resulting Op node should have two
    // operands: input_id (the LHS, which is x itself in this self-mul
    // example) plus the env-resolved x. They're identical here — the
    // Op node carries operands=[x_id, x_id].
    //
    // With x: Tensor[batch, 4] and matmul[x, transpose_b=true]:
    //   LHS = x = [batch, 4] (M=batch, K=4)
    //   RHS = x = [batch, 4], transpose_b=true → contract on last dim
    //          (rhs_K = 4) so N = batch (rhs second-to-last)
    //   output = [M, N] = [batch, batch]
    let src = "\
model M [batch=2]:
    x: Tensor[batch, 4]

    y: Tensor[batch, batch] = x -> matmul[x, transpose_b=true]
";
    let ast = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&ast).expect("build");
    let model = &uir.models[0];
    let out_id = model.output;
    let crate::ir::types::NodeKind::Op { operands, .. } = &model.nodes[out_id].kind else {
        panic!("expected Op node, got Input");
    };
    assert_eq!(operands.len(), 2);
    // Both operands point at the same NodeId — x itself, since q=k=v=x.
    assert_eq!(operands[0], operands[1]);
}

#[test]
fn softmax_rank_too_low_caught_at_uir() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;
    let input_1d = Shape(vec![16]);
    let err = infer_output_shape(StdOp::Softmax, &[input_1d], &[]).unwrap_err();
    assert!(
        matches!(
            err,
            ShapeError::RankTooLow {
                required: 2,
                actual: 1
            }
        ),
        "unexpected error: {:?}",
        err
    );
}

// M14: StdOp::LayerNorm IR foundation tests.

#[test]
fn layernorm_resolves_and_has_no_positional_one_optional_named() {
    use crate::ir::stdlib::{resolve, signature, ArgType, StdOp};

    let op = resolve("layernorm").expect("layernorm should resolve");
    assert_eq!(op, StdOp::LayerNorm);

    let sig = signature(StdOp::LayerNorm);
    assert_eq!(sig.positional.len(), 0, "layernorm has no positional args");
    assert_eq!(
        sig.named.len(),
        1,
        "layernorm has exactly one named arg (affine)"
    );
    assert_eq!(sig.named[0].name, "affine");
    assert_eq!(sig.named[0].ty, ArgType::Symbol);
    assert!(
        !sig.named[0].required,
        "affine is opt-in (default = no affine)"
    );
}

#[test]
fn layernorm_infer_shape_is_identity_at_rank_2_and_3() {
    use crate::ir::stdlib::{infer_output_shape, StdOp};
    use crate::ir::types::Shape;

    // Rank 2: [B, D]
    let inputs_2d = vec![Shape(vec![8, 32])];
    let out =
        infer_output_shape(StdOp::LayerNorm, &inputs_2d, &[]).expect("rank-2 input should succeed");
    assert_eq!(out, Shape(vec![8, 32]), "shape should be identity");

    // Rank 3: [B, S, D] — common in transformer use
    let inputs_3d = vec![Shape(vec![2, 16, 64])];
    let out =
        infer_output_shape(StdOp::LayerNorm, &inputs_3d, &[]).expect("rank-3 input should succeed");
    assert_eq!(out, Shape(vec![2, 16, 64]));
}

#[test]
fn layernorm_infer_shape_rejects_rank_below_2() {
    use crate::ir::stdlib::{infer_output_shape, ShapeError, StdOp};
    use crate::ir::types::Shape;

    let inputs_1d = vec![Shape(vec![32])];
    let err = infer_output_shape(StdOp::LayerNorm, &inputs_1d, &[])
        .expect_err("rank-1 input should be rejected");

    match err {
        ShapeError::RankTooLow { required, actual } => {
            assert_eq!(required, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected RankTooLow, got {other:?}"),
    }
}

#[test]
fn layernorm_has_affine_recognises_affine_true() {
    use crate::ir::stdlib::layernorm_has_affine;
    use crate::ir::types::{AttrValue, OpAttr};
    let attrs = vec![OpAttr {
        name: "affine".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    assert!(layernorm_has_affine(&attrs));
}

#[test]
fn layernorm_has_affine_rejects_absent_or_false() {
    use crate::ir::stdlib::layernorm_has_affine;
    use crate::ir::types::{AttrValue, OpAttr};
    assert!(!layernorm_has_affine(&[]));
    let false_attrs = vec![OpAttr {
        name: "affine".to_string(),
        value: AttrValue::Symbol("false".to_string()),
    }];
    assert!(!layernorm_has_affine(&false_attrs));
    let other_name = vec![OpAttr {
        name: "bias".to_string(),
        value: AttrValue::Symbol("true".to_string()),
    }];
    assert!(!layernorm_has_affine(&other_name));
}

// M13 Group B: StdOp::Add IR foundation tests.

#[test]
fn build_add_op_two_input_model() {
    // M13 Group B: minimal positive case.
    // NFL requires [param=val] generic params on model; literal Tensor
    // dims are fine when mixed with a named param (b=2, d=4).
    let src = "model AddDemo [b=2, d=4]:\n    x: Tensor[b, d]\n    skip: Tensor[b, d]\n\n    x -> add[skip]\n";
    let nfl = crate::parse(src).expect("parse");
    let uir = crate::ir::build(&nfl).expect("ir::build");
    let model = &uir.models[0];
    // Expect: 2 Input nodes (x, skip) + 1 Op node (add) = 3 nodes.
    assert_eq!(model.nodes.len(), 3, "expected 3 nodes");
    // Output node is the Add.
    let add_node = &model.nodes[model.output];
    use crate::ir::stdlib::StdOp;
    use crate::ir::types::NodeKind;
    let NodeKind::Op { op, operands, .. } = &add_node.kind else {
        panic!("output is not an Op: {:?}", add_node.kind);
    };
    assert_eq!(*op, StdOp::Add, "output op must be StdOp::Add");
    assert_eq!(operands.len(), 2, "Add must have 2 operands");
    // Output shape preserved.
    assert_eq!(add_node.ty.shape.0, vec![2, 4]);
}

#[test]
fn build_add_op_rejects_shape_mismatch() {
    // M13 Group B: strict shape equality — no broadcasting.
    // x is [b, d1] = [2, 4], skip is [b, d2] = [2, 8] — mismatch.
    // ShapeError::AddShapeMismatch is surfaced as BuildErrorKind::ShapeMismatch
    // whose `detail` field carries the Display string from AddShapeMismatch.
    let src = "model BadAdd [b=2, d1=4, d2=8]:\n    x: Tensor[b, d1]\n    skip: Tensor[b, d2]\n\n    x -> add[skip]\n";
    let nfl = crate::parse(src).expect("parse");
    let result = crate::ir::build(&nfl);
    let err = result.expect_err("expected build error for mismatched shapes");
    assert!(
        matches!(
            &err.kind,
            crate::ir::error::BuildErrorKind::ShapeMismatch { detail }
                if detail.contains("add operand shape mismatch")
        ),
        "expected ShapeMismatch with 'add operand shape mismatch'; got {err:?}"
    );
}
