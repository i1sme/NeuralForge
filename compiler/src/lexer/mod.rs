//! Hand-written lexer for NFL.

pub mod tokens;

pub use tokens::{LexError, Token, TokenKind};

/// Tokenise NFL source text. To be implemented in Task 4.
pub fn lex(_source: &str) -> Result<Vec<Token>, LexError> {
    unimplemented!("lex() — implemented in Task 4")
}
