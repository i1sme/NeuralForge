//! NFL Compiler — library crate.

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use lexer::LexError;
pub use parser::ParseError;

/// Top-level entry point: lex and parse NFL source into an AST.
///
/// Returns the first error encountered (parsing halts on first error in v0.1).
pub fn parse(source: &str) -> Result<NflSource, ParseError> {
    let tokens = lexer::lex(source).map_err(|e| {
        let (line, col) = e.position();
        ParseError {
            message: format!("{e}"),
            line,
            col,
            expected: Vec::new(),
        }
    })?;
    let mut p = parser::Parser::new(&tokens);
    parser::parse_nfl_source(&mut p)
}

pub mod ir;

pub use ir::{
    AttrValue, BuildError, BuildErrorKind, Node, NodeId, NodeKind, OpAttr, PostOp, Shape, StdOp,
    Type, Uir, UirModel,
};
