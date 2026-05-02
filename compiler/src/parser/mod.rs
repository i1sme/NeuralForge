//! Hand-written recursive-descent parser for NFL.
//!
//! Each function consumes tokens from the [`Parser`] cursor and returns either
//! an AST node or a [`ParseError`]. There is no error recovery in v0.1 — the
//! first error halts parsing.

// Scaffolding lands ahead of its first consumer. The `parse_*` functions
// added in Task 10 onwards will call into Parser/ParseError/describe_kind/
// join_alts; until then the dead-code lint would fire. Remove this
// directive once `parse_arg_value` (Task 10) wires the chain.
#![allow(dead_code)]

use crate::lexer::tokens::{Token, TokenKind};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
    pub col: u32,
    pub expected: Vec<&'static str>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Stateful parser holding a token slice and a cursor.
pub(crate) struct Parser<'t> {
    tokens: &'t [Token],
    pos: usize,
}

impl<'t> Parser<'t> {
    pub fn new(tokens: &'t [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Token at the cursor (does not advance). Panics only if the lexer failed
    /// to emit an Eof terminator, which would be a lexer bug.
    pub fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    /// Token kind at the cursor.
    pub fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    /// Look at the kind `n` tokens ahead of the cursor (0 = current).
    /// Returns `None` if the lookahead is past the end (after Eof).
    pub fn peek_at(&self, n: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + n).map(|t| &t.kind)
    }

    /// Advance one token. Returns the consumed token.
    pub fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    /// If the current token's kind matches `expected`, consume it; otherwise
    /// return a ParseError naming what was expected.
    pub fn consume(&mut self, expected: TokenKind, label: &'static str) -> Result<&Token, ParseError> {
        if self.peek_kind() == &expected {
            Ok(self.advance())
        } else {
            Err(self.error_expected(&[label]))
        }
    }

    /// Conditionally consume the current token if its kind matches `expected`.
    /// Returns true if consumed.
    pub fn eat(&mut self, expected: &TokenKind) -> bool {
        if self.peek_kind() == expected {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip any leading Newline tokens.
    pub fn skip_newlines(&mut self) {
        while matches!(self.peek_kind(), TokenKind::Newline) {
            self.advance();
        }
    }

    /// Build a ParseError pointing at the current token, listing what was expected.
    pub fn error_expected(&self, expected: &[&'static str]) -> ParseError {
        let t = self.peek();
        let found = describe_kind(&t.kind);
        ParseError {
            message: format!("expected {}, found {}", join_alts(expected), found),
            line: t.line,
            col: t.col,
            expected: expected.to_vec(),
        }
    }

    pub fn position(&self) -> (u32, u32) {
        let t = self.peek();
        (t.line, t.col)
    }
}

fn join_alts(items: &[&str]) -> String {
    match items.len() {
        0 => "<nothing>".to_string(),
        1 => format!("'{}'", items[0]),
        _ => {
            let last = items.last().unwrap();
            let head: Vec<String> = items[..items.len() - 1]
                .iter()
                .map(|s| format!("'{}'", s))
                .collect();
            format!("{} or '{}'", head.join(", "), last)
        }
    }
}

fn describe_kind(k: &TokenKind) -> String {
    use TokenKind::*;
    match k {
        Model => "'model'".into(),
        Tensor => "'Tensor'".into(),
        LBracket => "'['".into(),
        RBracket => "']'".into(),
        Colon => "':'".into(),
        Comma => "','".into(),
        Equals => "'='".into(),
        Arrow => "'->'".into(),
        Ident(s) => format!("identifier '{}'", s),
        Integer(n) => format!("integer literal {}", n),
        Number(n) => format!("number literal {}", n),
        Newline => "newline".into(),
        Indent => "indent".into(),
        Dedent => "dedent".into(),
        Eof => "end of file".into(),
    }
}
