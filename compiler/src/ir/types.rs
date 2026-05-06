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
    /// Clamp negative values to zero (`max(x, 0)`), applied per-element to
    /// the producer's output before store.
    Relu,
    /// Row-wise softmax. Emit shape is structurally different from `Relu` —
    /// `emit_linear` materialises the full row first, then runs three sweeps
    /// (max → exp+sum → normalize) in-place. See `arm64.md` §4.10.
    SoftmaxRow,
}

impl std::fmt::Display for PostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PostOp::Relu => "relu",
            PostOp::SoftmaxRow => "softmax_row",
        };
        write!(f, "{}", name)
    }
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
                fused_post_ops,
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
                if !fused_post_ops.is_empty() {
                    let fused_s = fused_post_ops
                        .iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "    fused=[{}]", fused_s)?;
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

// ----------------------------------------------------------------------------
// M8: calls_extern_math predicate (Task 14)
//
// UIR-level predicate: true iff any operation requires linking against
// external math. Currently: standalone Softmax or fused SoftmaxRow.
// Does not depend on any profile.
// ----------------------------------------------------------------------------

impl UirModel {
    /// True iff any operation in this model requires linking against
    /// external math (currently: standalone Softmax or fused SoftmaxRow).
    /// UIR-level predicate — does not depend on any profile.
    pub fn calls_extern_math(&self) -> bool {
        use crate::ir::stdlib::StdOp;
        self.nodes.iter().any(|n| match &n.kind {
            NodeKind::Op {
                op, fused_post_ops, ..
            } => {
                matches!(op, StdOp::Softmax)
                    || fused_post_ops
                        .iter()
                        .any(|p| matches!(p, PostOp::SoftmaxRow))
            }
            NodeKind::Input { .. } => false,
        })
    }
}

impl Uir {
    pub fn calls_extern_math(&self) -> bool {
        self.models.iter().any(UirModel::calls_extern_math)
    }
}

// ----------------------------------------------------------------------------
// M8: verbose viewer wrappers (Task 15)
//
// Newtype pattern over plain methods. Idiomatic Rust composition:
// each wrapper has its own `Display` impl, so `write!(f, "{}",
// VerboseModel(m))` works inside the outer `VerboseUir` impl
// without any `fmt_verbose` boilerplate. Default `Display` for
// `Uir`/`UirModel`/`Node` is unchanged — the compact format used by
// `nflc parse --uir`.
// ----------------------------------------------------------------------------

pub struct VerboseUir<'a>(pub &'a Uir);

impl std::fmt::Display for VerboseUir<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_nodes: usize = self.0.models.iter().map(|m| m.nodes.len()).sum();
        writeln!(f, "uir-verbose summary")?;
        writeln!(f, "  models: {}", self.0.models.len())?;
        writeln!(f, "  total nodes: {}", total_nodes)?;
        writeln!(
            f,
            "  calls-extern-math: {}",
            if self.0.calls_extern_math() {
                "yes"
            } else {
                "no"
            }
        )?;
        writeln!(f)?;
        for m in &self.0.models {
            write!(f, "{}", VerboseModel(m))?;
        }
        Ok(())
    }
}

pub struct VerboseModel<'a>(pub &'a UirModel);

impl std::fmt::Display for VerboseModel<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.0;
        writeln!(f, "uir-model {}", m.name)?;
        let inputs = m
            .inputs
            .iter()
            .map(|i| format!("n{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(f, "  inputs: [{}]", inputs)?;
        writeln!(f, "  output: n{}", m.output)?;
        writeln!(f, "  node count: {}", m.nodes.len())?;
        writeln!(
            f,
            "  calls-extern-math: {}",
            if m.calls_extern_math() { "yes" } else { "no" }
        )?;
        writeln!(f)?;
        for (i, node) in m.nodes.iter().enumerate() {
            write!(f, "  n{}: {}", i, VerboseNode(node))?;
        }
        Ok(())
    }
}

pub struct VerboseNode<'a>(pub &'a Node);

impl std::fmt::Display for VerboseNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0.kind {
            NodeKind::Input { name } => {
                writeln!(f, "input {:?}        :: {}", name, self.0.ty.shape)
            }
            NodeKind::Op {
                op,
                operands,
                attrs,
                fused_post_ops,
            } => {
                let ops_s = operands
                    .iter()
                    .map(|o| format!("n{}", o))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "{}           :: {}    operands=[{}]",
                    op, self.0.ty.shape, ops_s
                )?;
                if !attrs.is_empty() {
                    let a = attrs
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    write!(f, "    attrs=[{}]", a)?;
                }
                writeln!(f)?;
                for p in fused_post_ops {
                    writeln!(f, "       -> fused: {}", p)?;
                }
                Ok(())
            }
        }
    }
}
