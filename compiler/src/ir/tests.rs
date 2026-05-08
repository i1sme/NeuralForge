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
