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

/// Post-operations that fuse into a producer's output store.
///
/// `#[non_exhaustive]` — M5b/M6+ may add Gelu, Tanh, Sigmoid. Each variant
/// is meaningful as "applied to one element after the producer computes it,
/// before storing"; not all StdOps fit (Softmax needs row-context, Dropout
/// is no-op at inference, Linear can't post-op another Linear).
///
/// Keeping `PostOp` distinct from `StdOp` makes the constraint explicit at
/// type level: profiles can't mistakenly route a softmax through the
/// post-op machinery.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostOp {
    /// Clamp negative values to zero (max(x, 0)). Equivalent to fusing a
    /// terminal-or-single-consumer Relu node into its producer.
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input {
        name: String,
    },
    Op {
        op: super::stdlib::StdOp,
        operands: Vec<NodeId>,
        attrs: Vec<OpAttr>,
        /// Fused post-operations, applied per-element after this op
        /// produces its output, before storing. Empty for un-fused or
        /// non-Linear ops.
        ///
        /// Populated only by `passes::FuseLinearRelu` (M5a) and future
        /// fusion passes. `compiler::ir::build` always sets this to
        /// `Vec::new()`.
        ///
        /// `Vec` rather than `Option` so M5b can express chains like
        /// `[BiasAdd, Relu]` should the need arise.
        fused_post_ops: Vec<PostOp>,
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

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims: Vec<String> = self.0.iter().map(|d| d.to_string()).collect();
        write!(f, "Tensor[{}]", dims.join(", "))
    }
}

impl std::fmt::Display for AttrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrValue::Integer(n) => write!(f, "{}", n),
            AttrValue::Float(v) => write!(f, "{}", v),
            AttrValue::Symbol(s) => write!(f, "{}", s),
        }
    }
}

impl std::fmt::Display for OpAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            NodeKind::Input { name } => {
                write!(f, "input {:?}        :: {}", name, self.ty.shape)
            }
            NodeKind::Op {
                op,
                operands,
                attrs,
                ..
            } => {
                let ops_s = operands
                    .iter()
                    .map(|o| format!("n{}", o))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "{}           :: {}    operands=[{}]",
                    op, self.ty.shape, ops_s
                )?;
                if !attrs.is_empty() {
                    let a = attrs
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for UirModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "uir-model {}", self.name)?;
        let inputs = self
            .inputs
            .iter()
            .map(|i| format!("n{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "  inputs: [{}]", inputs)?;
        writeln!(f, "  output: n{}", self.output)?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f, "  n{}: {}", i, node)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Uir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for m in &self.models {
            writeln!(f, "{}", m)?;
        }
        Ok(())
    }
}
