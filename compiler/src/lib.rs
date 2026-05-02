//! NFL Compiler — library crate.
//!
//! Public API will grow as Milestone 2 progresses. The final entry point is
//! [`parse`], which accepts NFL source text and returns a typed AST.

pub mod ast;
pub mod lexer;
pub mod parser;
