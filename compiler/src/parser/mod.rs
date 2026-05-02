//! Hand-written recursive-descent parser for NFL.
//!
//! Each function consumes tokens from the [`Parser`] cursor and returns either
//! an AST node or a [`ParseError`]. There is no error recovery in v0.1 — the
//! first error halts parsing.

#[cfg(test)]
mod tests;

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

use crate::ast::{ArgValue, OpArg, Operation, Span};

pub(crate) fn parse_arg_value(p: &mut Parser) -> Result<ArgValue, ParseError> {
    match p.peek_kind().clone() {
        TokenKind::Integer(n) => {
            p.advance();
            Ok(ArgValue::Integer(n))
        }
        TokenKind::Number(n) => {
            p.advance();
            Ok(ArgValue::Float(n))
        }
        TokenKind::Ident(s) => {
            p.advance();
            Ok(ArgValue::Symbol(s))
        }
        _ => Err(p.error_expected(&["integer", "number", "identifier"])),
    }
}

pub(crate) fn parse_named_arg(p: &mut Parser) -> Result<(String, ArgValue), ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    p.advance();
    p.consume(TokenKind::Equals, "=")?;
    let value = parse_arg_value(p)?;
    Ok((name, value))
}

/// Parse `op_args = positional_args , [ "," , named_args ] | named_args`.
/// Returns the list of arguments. Caller has already consumed `[` and is
/// expected to consume the matching `]` afterwards.
///
/// Uses `peek_at(1)` to look one token past the cursor and decide whether
/// the next item is a `named_arg` (`Ident "="`) or a positional `arg_value`.
pub(crate) fn parse_op_args(p: &mut Parser) -> Result<Vec<OpArg>, ParseError> {
    let mut args = Vec::new();
    let mut seen_named = false;

    loop {
        // Decide whether the next item is a named_arg (Ident "=" ...) or a
        // positional arg (any arg_value).
        let is_named = matches!(p.peek_kind(), TokenKind::Ident(_))
            && matches!(p.peek_at(1), Some(TokenKind::Equals));

        if is_named {
            let (name, value) = parse_named_arg(p)?;
            args.push(OpArg::Named { name, value });
            seen_named = true;
        } else {
            if seen_named {
                return Err(ParseError {
                    message: "positional argument cannot follow a named argument".into(),
                    line: p.peek().line,
                    col: p.peek().col,
                    expected: vec!["named argument", "']'"],
                });
            }
            let value = parse_arg_value(p)?;
            args.push(OpArg::Positional(value));
        }

        // Either consume a comma and continue, or break.
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    Ok(args)
}

pub(crate) fn parse_operation(p: &mut Parser) -> Result<Operation, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();

    let mut args = Vec::new();
    if p.eat(&TokenKind::LBracket) {
        // Empty bracket "[]" is invalid per spec.
        if matches!(p.peek_kind(), TokenKind::RBracket) {
            return Err(ParseError {
                message: "operation argument list cannot be empty; omit the brackets if there are no arguments".into(),
                line: p.peek().line,
                col: p.peek().col,
                expected: vec!["argument"],
            });
        }
        args = parse_op_args(p)?;
        p.consume(TokenKind::RBracket, "]")?;
    }

    Ok(Operation {
        name,
        args,
        span: Span::new(line, col),
    })
}
