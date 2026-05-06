// SPDX-License-Identifier: Apache-2.0

//! Shared UIR-construction helpers for pass tests.
//!
//! Use these instead of hand-rolling `Node` literals. The functions construct
//! `Node`s with `source_span: Span::new(1, 1)` and `fused_post_ops: vec![]`
//! (when applicable) — defaults that suit pass tests where the source span
//! is irrelevant and post-ops are populated by the pass under test.

use crate::ast::Span;
use crate::ir::stdlib::StdOp;
use crate::ir::types::{AttrValue, Node, NodeId, NodeKind, OpAttr, Shape, Type};

/// Construct an `Input` node with the given name and shape.
pub(crate) fn input_node(name: &str, shape: Vec<u64>) -> Node {
    Node {
        kind: NodeKind::Input { name: name.into() },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(shape),
        },
        source_span: Span::new(1, 1),
    }
}

/// Construct an `Op` node with the given op kind, operand ids, attributes,
/// and output shape. `fused_post_ops` starts empty.
pub(crate) fn op_node(
    op: StdOp,
    operands: Vec<NodeId>,
    attrs: Vec<OpAttr>,
    shape: Vec<u64>,
) -> Node {
    Node {
        kind: NodeKind::Op {
            op,
            operands,
            attrs,
            fused_post_ops: vec![],
        },
        ty: Type {
            name: "Tensor".into(),
            shape: Shape(shape),
        },
        source_span: Span::new(1, 1),
    }
}

/// Convenience for the common `out_dim` integer attribute on `Linear`.
pub(crate) fn out_dim_attr(value: u64) -> OpAttr {
    OpAttr {
        name: "out_dim".into(),
        value: AttrValue::Integer(value),
    }
}

/// Convenience for the `rate` float attribute on `Dropout`.
pub(crate) fn rate_attr(value: f64) -> OpAttr {
    OpAttr {
        name: "rate".into(),
        value: AttrValue::Float(value),
    }
}
