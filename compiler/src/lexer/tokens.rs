// SPDX-License-Identifier: Apache-2.0

//! Token types and lexical errors.
//!
//! See `language/grammar.ebnf` for the abstract grammar this models.

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Model,
    Tensor,
    // Punctuation
    LBracket,
    RBracket,
    Colon,
    Comma,
    Equals,
    Arrow,
    // Identifiers and literals
    Ident(String),
    Integer(u64),
    Number(f64),
    // Significant whitespace
    Newline,
    Indent,
    Dedent,
    // End
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    /// 1-based line number of the first character of the token.
    pub line: u32,
    /// 1-based column of the first character of the token.
    pub col: u32,
}

impl Token {
    pub const fn new(kind: TokenKind, line: u32, col: u32) -> Self {
        Self { kind, line, col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LexError {
    TabInIndent { line: u32, col: u32 },
    BadDedent { line: u32, col: u32 },
    UnknownChar { line: u32, col: u32, ch: char },
    BadNumber { line: u32, col: u32, raw: String },
    UnexpectedEof { line: u32, col: u32 },
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::TabInIndent { .. } => write!(f, "tab character in leading whitespace"),
            LexError::BadDedent { .. } => write!(
                f,
                "inconsistent dedent: indent level does not match any enclosing block"
            ),
            LexError::UnknownChar { ch, .. } => write!(f, "unknown character: {:?}", ch),
            LexError::BadNumber { raw, .. } => write!(f, "malformed number literal: {:?}", raw),
            LexError::UnexpectedEof { .. } => write!(f, "unexpected end of file"),
        }
    }
}

impl std::error::Error for LexError {}

impl LexError {
    /// Returns the (line, col) where the error occurred. 1-based.
    pub fn position(&self) -> (u32, u32) {
        match *self {
            LexError::TabInIndent { line, col }
            | LexError::BadDedent { line, col }
            | LexError::UnknownChar { line, col, .. }
            | LexError::BadNumber { line, col, .. }
            | LexError::UnexpectedEof { line, col } => (line, col),
        }
    }
}
