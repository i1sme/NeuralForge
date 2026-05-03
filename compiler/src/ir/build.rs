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
    op_span: Span,
) -> Result<Vec<OpAttr>, BuildError> {
    let sig = stdlib::signature(op);

    // Split AST args into positional and named (in source order).
    let mut positionals: Vec<&ArgValue> = Vec::new();
    let mut nameds: Vec<(&str, &ArgValue)> = Vec::new();
    for arg in args {
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
    let ok = match (slot.ty, value) {
        (ArgType::Integer, ArgValue::Integer(_)) => true,
        (ArgType::Float, ArgValue::Float(_)) => true,
        (ArgType::Symbol, ArgValue::Symbol(_)) => true,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(BuildError::arg_type_mismatch(slot.name, expected, actual, op_span))
    }
}

fn arg_value_to_attr(v: &ArgValue) -> AttrValue {
    match v {
        ArgValue::Integer(n) => AttrValue::Integer(*n),
        ArgValue::Float(f) => AttrValue::Float(*f),
        ArgValue::Symbol(s) => AttrValue::Symbol(s.clone()),
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
    existing_nodes: &[Node],
    out_nodes: &mut Vec<Node>,
) -> Result<NodeId, BuildError> {
    let std_op = stdlib::resolve(&op_ast.name)
        .ok_or_else(|| BuildError::unknown_op(&op_ast.name, op_ast.span))?;
    let attrs = resolve_args(std_op, &op_ast.args, op_ast.span)?;
    let input_shape = existing_nodes[input_id].ty.shape.clone();
    let out_shape = stdlib::infer_output_shape(std_op, &[input_shape], &attrs)
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
