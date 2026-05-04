//! AST→UIR builder.

use std::collections::HashMap;

use crate::ast::{Dim, TypeExpr};

use super::error::BuildError;
use super::types::Shape;

pub(crate) fn resolve_type(
    ty: &TypeExpr,
    params: &HashMap<&str, u64>,
) -> Result<Shape, BuildError> {
    let mut dims: Vec<u64> = Vec::with_capacity(ty.dims.len());
    for dim in &ty.dims {
        match dim {
            Dim::Integer(n) => dims.push(*n),
            Dim::Symbol(name) => {
                let v = params
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| BuildError::unknown_dim(name, ty.span))?;
                dims.push(v);
            }
        }
    }
    Ok(Shape(dims))
}

use crate::ast::{ArgValue, OpArg, Span};

use super::stdlib::{self, ArgSlot, ArgType, StdOp};
use super::types::{AttrValue, OpAttr};

pub(crate) fn resolve_args(
    op: StdOp,
    args: &[OpArg],
    params: &HashMap<&str, u64>,
    op_span: Span,
) -> Result<Vec<OpAttr>, BuildError> {
    let sig = stdlib::signature(op);

    // Pre-resolve Symbol args against model params: if a positional or named arg
    // is `Symbol(name)` and `name` is a model_param, substitute with `Integer(value)`.
    // Non-matching symbols (e.g. `bias=true`) stay as Symbol — type-check decides.
    let resolved: Vec<OpArg> = args
        .iter()
        .map(|arg| match arg {
            OpArg::Positional(v) => OpArg::Positional(resolve_arg_value(v, params)),
            OpArg::Named { name, value } => OpArg::Named {
                name: name.clone(),
                value: resolve_arg_value(value, params),
            },
        })
        .collect();

    // Split AST args into positional and named (in source order).
    let mut positionals: Vec<&ArgValue> = Vec::new();
    let mut nameds: Vec<(&str, &ArgValue)> = Vec::new();
    for arg in &resolved {
        match arg {
            OpArg::Positional(v) => positionals.push(v),
            OpArg::Named { name, value } => nameds.push((name.as_str(), value)),
        }
    }

    // Validate positional arity.
    let required_positional = sig.positional.iter().filter(|s| s.required).count();
    let max_positional = sig.positional.len();
    if positionals.len() < required_positional || positionals.len() > max_positional {
        return Err(BuildError::arg_count_mismatch(
            required_positional,
            positionals.len(),
            op_span,
        ));
    }

    let mut attrs: Vec<OpAttr> = Vec::with_capacity(positionals.len() + nameds.len());

    // Bind positionals to slots.
    for (slot, value) in sig.positional.iter().zip(positionals.iter()) {
        check_arg_type(slot, value, op_span)?;
        attrs.push(OpAttr {
            name: slot.name.to_string(),
            value: arg_value_to_attr(value),
        });
    }

    // Bind nameds — match each by slot name.
    for (name, value) in &nameds {
        let slot = sig
            .named
            .iter()
            .find(|s| s.name == *name)
            .ok_or_else(|| BuildError::unexpected_named_arg(name, op_span))?;
        check_arg_type(slot, value, op_span)?;
        attrs.push(OpAttr {
            name: slot.name.to_string(),
            value: arg_value_to_attr(value),
        });
    }

    // Verify all required named args are present.
    for slot in sig.named.iter().filter(|s| s.required) {
        if !nameds.iter().any(|(n, _)| *n == slot.name) {
            return Err(BuildError::missing_required_arg(slot.name, op_span));
        }
    }

    Ok(attrs)
}

fn check_arg_type(slot: &ArgSlot, value: &ArgValue, op_span: Span) -> Result<(), BuildError> {
    let actual = describe_arg_type(value);
    let expected = describe_slot_type(slot.ty);
    let ok = matches!(
        (slot.ty, value),
        (ArgType::Integer, ArgValue::Integer(_))
            | (ArgType::Float, ArgValue::Float(_))
            | (ArgType::Symbol, ArgValue::Symbol(_))
    );
    if ok {
        Ok(())
    } else {
        Err(BuildError::arg_type_mismatch(
            slot.name, expected, actual, op_span,
        ))
    }
}

fn arg_value_to_attr(v: &ArgValue) -> AttrValue {
    match v {
        ArgValue::Integer(n) => AttrValue::Integer(*n),
        ArgValue::Float(f) => AttrValue::Float(*f),
        ArgValue::Symbol(s) => AttrValue::Symbol(s.clone()),
    }
}

/// Pre-resolve a Symbol arg against model_params. If the symbol matches a
/// param name, return `Integer(value)`; otherwise return a clone (Symbol stays
/// Symbol — could be a keyword like `true`).
fn resolve_arg_value(v: &ArgValue, params: &HashMap<&str, u64>) -> ArgValue {
    match v {
        ArgValue::Symbol(name) => match params.get(name.as_str()) {
            Some(&value) => ArgValue::Integer(value),
            None => ArgValue::Symbol(name.clone()),
        },
        other => other.clone(),
    }
}

fn describe_arg_type(v: &ArgValue) -> &'static str {
    match v {
        ArgValue::Integer(_) => "integer",
        ArgValue::Float(_) => "float",
        ArgValue::Symbol(_) => "identifier",
    }
}

fn describe_slot_type(ty: ArgType) -> &'static str {
    match ty {
        ArgType::Integer => "integer",
        ArgType::Float => "float",
        ArgType::Symbol => "identifier",
    }
}

use super::types::{Node, NodeId, NodeKind, Type};
use crate::ast::Operation;

pub(crate) fn build_op(
    op_ast: &Operation,
    input_id: NodeId,
    input_shape: &Shape,
    params: &HashMap<&str, u64>,
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, params, op_ast.span)?;
    stdlib::validate_attrs(std_op, &attrs).map_err(|e| {
        let attr_name = match &e {
            stdlib::AttrError::OutOfRange { name, .. } => *name,
            stdlib::AttrError::MissingAttr { name } => *name,
        };
        BuildError::invalid_attr_value(
            &format!("{}", std_op),
            attr_name,
            &format!("{e}"),
            op_ast.span,
        )
    })?;
    let out_shape = stdlib::infer_output_shape(std_op, std::slice::from_ref(input_shape), &attrs)
        .map_err(|e| BuildError::shape(format!("{e}"), op_ast.span))?;
    let id = out_nodes.len();
    out_nodes.push(Node {
        kind: NodeKind::Op {
            op: std_op,
            operands: vec![input_id],
            attrs,
        },
        ty: Type {
            name: "Tensor".to_string(),
            shape: out_shape,
        },
        source_span: op_ast.span,
    });
    Ok(id)
}

use crate::ast::{ModelDef, ModelStmt, NflSource};

use super::types::{Uir, UirModel};

pub fn build(ast: &NflSource) -> Result<Uir, BuildError> {
    let mut models = Vec::with_capacity(ast.models.len());
    for ast_model in &ast.models {
        models.push(build_model(ast_model)?);
    }
    Ok(Uir { models })
}

pub(crate) fn build_model(ast_model: &ModelDef) -> Result<UirModel, BuildError> {
    // Index params for symbolic dim lookup.
    let params: HashMap<&str, u64> = ast_model
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.value))
        .collect();

    let mut nodes: Vec<Node> = Vec::new();
    let mut env: HashMap<String, NodeId> = HashMap::new();
    let mut inputs: Vec<NodeId> = Vec::new();
    let mut last_pipeline_output: Option<NodeId> = None;

    for stmt in &ast_model.body {
        match stmt {
            ModelStmt::VariableDecl(v) => {
                let shape = resolve_type(&v.ty, &params)?;
                let id = nodes.len();
                nodes.push(Node {
                    kind: NodeKind::Input {
                        name: v.name.clone(),
                    },
                    ty: Type {
                        name: v.ty.name.clone(),
                        shape,
                    },
                    source_span: v.span,
                });
                env.insert(v.name.clone(), id);
                inputs.push(id);
            }
            ModelStmt::Pipeline(p) => {
                let mut current = *env
                    .get(&p.source)
                    .ok_or_else(|| BuildError::unknown_variable(&p.source, p.span))?;
                for op_ast in &p.steps {
                    let input_shape = nodes[current].ty.shape.clone();
                    current = build_op(op_ast, current, &input_shape, &params, &mut nodes)?;
                }
                last_pipeline_output = Some(current);
            }
        }
    }

    let output = last_pipeline_output
        .ok_or_else(|| BuildError::model_has_no_pipeline(&ast_model.name, ast_model.span))?;

    Ok(UirModel {
        name: ast_model.name.clone(),
        nodes,
        inputs,
        output,
        source_span: ast_model.span,
    })
}
