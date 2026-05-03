//! UIR data types — index-based DAG of typed nodes.

use crate::ast::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Uir {
    pub models: Vec<UirModel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UirModel {
    pub name: String,
    pub nodes: Vec<Node>,
    pub inputs: Vec<NodeId>,
    pub output: NodeId,
    pub source_span: Span,
}

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    pub ty: Type,
    pub source_span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input { name: String },
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpAttr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Integer(u64),
    Float(f64),
    /// Used by named keyword-like args (e.g. `bias=true`). Not exercised by
    /// any M3a test (tiny_mlp.nfl uses only Integer args). See spec §9 open Q1.
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub name: String,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }
}
