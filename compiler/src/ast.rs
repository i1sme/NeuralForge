//! Typed AST for NFL v0.1.
//!
//! Mirrors the EBNF productions in `language/grammar.ebnf`. Every node carries
//! a [`Span`] indicating where it started in the source, for future error
//! reporting and the human-readable viewer (Milestone 7).

#[derive(Debug, Clone, PartialEq)]
pub struct NflSource {
    pub models: Vec<ModelDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelDef {
    pub name: String,
    pub params: Vec<NamedValue>,
    pub body: Vec<ModelStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedValue {
    pub name: String,
    pub value: u64,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStmt {
    VariableDecl(VariableDecl),
    Pipeline(PipelineStmt),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeExpr {
    /// Always `"Tensor"` in v0.1. See spec §9, open question 1.
    pub name: String,
    pub dims: Vec<Dim>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Integer(u64),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStmt {
    pub source: String,
    pub steps: Vec<Operation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub name: String,
    pub args: Vec<OpArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpArg {
    Positional(ArgValue),
    Named { name: String, value: ArgValue },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue {
    Integer(u64),
    Float(f64),
    Symbol(String),
}

/// Source position of an AST node. v0.1 stores only the start position.
/// End-position is deferred until a consumer needs it (see spec §9, open
/// question 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub const fn new(line: u32, col: u32) -> Self {
        Self { line, col }
    }
}
