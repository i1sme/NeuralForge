// SPDX-License-Identifier: Apache-2.0

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
    pub fn consume(
        &mut self,
        expected: TokenKind,
        label: &'static str,
    ) -> Result<&Token, ParseError> {
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

    #[allow(dead_code)]
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

use crate::ast::PipelineStmt;

/// `pipeline_chain = pipeline_step , { pipeline_step }`. Caller is positioned
/// at the first `->`. Tolerates a single Newline between steps when the next
/// line begins with `->` (continuation-line per grammar §5.2).
fn parse_pipeline_chain(p: &mut Parser) -> Result<Vec<Operation>, ParseError> {
    let mut steps = Vec::new();
    p.consume(TokenKind::Arrow, "->")?;
    steps.push(parse_operation(p)?);
    loop {
        while matches!(p.peek_kind(), TokenKind::Newline)
            && matches!(p.peek_at(1), Some(TokenKind::Arrow))
        {
            p.advance();
        }
        if !matches!(p.peek_kind(), TokenKind::Arrow) {
            break;
        }
        p.advance();
        steps.push(parse_operation(p)?);
    }
    Ok(steps)
}

pub(crate) fn parse_pipeline_stmt(p: &mut Parser) -> Result<PipelineStmt, ParseError> {
    let TokenKind::Ident(source) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();

    // pipeline_chain = pipeline_step , { pipeline_step } ; — at least one step.
    // Continuation lines: the lexer suppresses INDENT/DEDENT when a line starts
    // with `->` at deeper indent (grammar §5.2), but it still emits a Newline
    // between the previous step and the continuation. parse_pipeline_chain
    // tolerates that Newline so the chain is parsed as a single pipeline_stmt.
    let steps = parse_pipeline_chain(p)?;

    Ok(PipelineStmt {
        source,
        steps,
        span: Span::new(line, col),
    })
}

use crate::ast::{Dim, TypeExpr, VariableDecl};

pub(crate) fn parse_dim(p: &mut Parser) -> Result<Dim, ParseError> {
    match p.peek_kind().clone() {
        TokenKind::Integer(n) => {
            p.advance();
            Ok(Dim::Integer(n))
        }
        TokenKind::Ident(s) => {
            p.advance();
            Ok(Dim::Symbol(s))
        }
        _ => Err(p.error_expected(&["integer", "identifier"])),
    }
}

pub(crate) fn parse_dim_list(p: &mut Parser) -> Result<Vec<Dim>, ParseError> {
    let mut dims = vec![parse_dim(p)?];
    while p.eat(&TokenKind::Comma) {
        dims.push(parse_dim(p)?);
    }
    Ok(dims)
}

pub(crate) fn parse_type_expr(p: &mut Parser) -> Result<TypeExpr, ParseError> {
    let (line, col) = (p.peek().line, p.peek().col);
    p.consume(TokenKind::Tensor, "Tensor")?;
    p.consume(TokenKind::LBracket, "[")?;
    if matches!(p.peek_kind(), TokenKind::RBracket) {
        return Err(ParseError {
            message: "Tensor type requires at least one dimension; empty dim_list is invalid"
                .into(),
            line: p.peek().line,
            col: p.peek().col,
            expected: vec!["integer", "identifier"],
        });
    }
    let dims = parse_dim_list(p)?;
    p.consume(TokenKind::RBracket, "]")?;
    Ok(TypeExpr {
        name: "Tensor".to_string(),
        dims,
        span: Span::new(line, col),
    })
}

pub(crate) fn parse_variable_decl(p: &mut Parser) -> Result<VariableDecl, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Colon, ":")?;
    let ty = parse_type_expr(p)?;
    Ok(VariableDecl {
        name,
        ty,
        span: Span::new(line, col),
    })
}

use crate::ast::NamedValue;

pub(crate) fn parse_named_value(p: &mut Parser) -> Result<NamedValue, ParseError> {
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    let (line, col) = (p.peek().line, p.peek().col);
    p.advance();
    p.consume(TokenKind::Equals, "=")?;
    // `.clone()` here turns the borrowed TokenKind into an owned one so we
    // can pattern-match without fighting the borrow checker. Integer is a
    // `u64` (Copy) so the clone is essentially free.
    let TokenKind::Integer(value) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["integer literal"]));
    };
    p.advance();
    Ok(NamedValue {
        name,
        value,
        span: Span::new(line, col),
    })
}

pub(crate) fn parse_model_params(p: &mut Parser) -> Result<Vec<NamedValue>, ParseError> {
    let mut params = vec![parse_named_value(p)?];
    while p.eat(&TokenKind::Comma) {
        params.push(parse_named_value(p)?);
    }
    Ok(params)
}

use crate::ast::{ModelDef, ModelStmt, NamedPipelineStmt};

pub(crate) fn parse_model_stmt(p: &mut Parser) -> Result<ModelStmt, ParseError> {
    // Disambiguate three cases by looking at the token immediately
    // after the leading identifier:
    //   - `Ident "->"`               → pipeline_stmt
    //   - `Ident ":"  Tensor … "="`  → named_pipeline_stmt
    //   - `Ident ":"  Tensor … (Newline | Dedent)`  → variable_decl
    //
    // The pipeline_stmt vs colon-prefixed forms is decided by peek_at(1).
    // The variable_decl vs named_pipeline_stmt distinction is decided by
    // looking past the type_expr — but that requires unbounded lookahead.
    // We sidestep this by parsing through the type_expr greedily and then
    // branching on whether `=` follows. parse_variable_decl already
    // consumes `Ident ":" type_expr` and stops; parse_named_pipeline_stmt
    // requires `=` after the type_expr.
    //
    // Implementation: peek 1 ahead. If `:`, parse the prefix once, then
    // dispatch on whether `=` follows (one-token lookahead on Equals).
    // If `->`, dispatch to pipeline_stmt directly.
    let after = match p.peek_at(1) {
        Some(k) => k,
        None => return Err(p.error_expected(&["':'", "'->'"])),
    };
    match after {
        TokenKind::Arrow => Ok(ModelStmt::Pipeline(parse_pipeline_stmt(p)?)),
        TokenKind::Colon => parse_decl_or_named_pipeline(p),
        _ => Err(p.error_expected(&["':'", "'->'"])),
    }
}

/// Common prefix `Ident ":" type_expr` is shared between variable_decl
/// and named_pipeline_stmt. We optimistically call `parse_variable_decl`
/// to consume the prefix, then look one token ahead: if `=`, the prefix
/// was actually the head of a named_pipeline_stmt and we promote the
/// parsed `VariableDecl` into a `NamedPipelineStmt`; otherwise we keep
/// the variable_decl as-is.
fn parse_decl_or_named_pipeline(p: &mut Parser) -> Result<ModelStmt, ParseError> {
    let decl = parse_variable_decl(p)?;

    // One-token lookahead on `=`.
    if !matches!(p.peek_kind(), TokenKind::Equals) {
        return Ok(ModelStmt::VariableDecl(decl));
    }
    p.advance();

    // Source identifier (the variable being piped from).
    let TokenKind::Ident(source) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["identifier"]));
    };
    p.advance();

    // Pipeline chain — at least one `-> operation`. Reuse the same
    // continuation-line newline tolerance as parse_pipeline_stmt.
    let steps = parse_pipeline_chain(p)?;

    Ok(ModelStmt::NamedPipeline(NamedPipelineStmt {
        binding_name: decl.name,
        declared_ty: decl.ty,
        source,
        steps,
        span: decl.span,
    }))
}

pub(crate) fn parse_model_body(p: &mut Parser) -> Result<Vec<ModelStmt>, ParseError> {
    // Tolerate blank or comment-only lines between the model header's Newline
    // and the first content line — the lexer emits Newlines for those without
    // affecting the indent stack, so the Indent appears AFTER one or more
    // Newlines rather than immediately.
    p.skip_newlines();
    p.consume(TokenKind::Indent, "indented body")?;
    let mut stmts = Vec::new();
    loop {
        // Eat blank-line newlines between statements.
        p.skip_newlines();
        if matches!(p.peek_kind(), TokenKind::Dedent) {
            break;
        }
        stmts.push(parse_model_stmt(p)?);
        // Per EBNF `model_body = model_stmt , { newline , model_stmt }`,
        // successive statements must be separated by a newline. The body
        // also ends with a Dedent (block close); we accept that as a
        // terminator without requiring a preceding Newline because the
        // lexer is allowed to elide the final Newline at EOF.
        match p.peek_kind() {
            TokenKind::Newline => {
                p.advance();
            }
            TokenKind::Dedent | TokenKind::Eof => {}
            _ => {
                return Err(p.error_expected(&["newline", "dedent"]));
            }
        }
    }
    p.consume(TokenKind::Dedent, "dedent")?;
    Ok(stmts)
}

pub(crate) fn parse_model_def(p: &mut Parser) -> Result<ModelDef, ParseError> {
    let (line, col) = (p.peek().line, p.peek().col);
    p.consume(TokenKind::Model, "model")?;
    let TokenKind::Ident(name) = p.peek_kind().clone() else {
        return Err(p.error_expected(&["model name (identifier)"]));
    };
    p.advance();
    p.consume(TokenKind::LBracket, "[")?;
    let params = parse_model_params(p)?;
    p.consume(TokenKind::RBracket, "]")?;
    p.consume(TokenKind::Colon, ":")?;
    p.consume(TokenKind::Newline, "newline")?;
    let body = parse_model_body(p)?;
    Ok(ModelDef {
        name,
        params,
        body,
        span: Span::new(line, col),
    })
}

use crate::ast::NflSource;

pub(crate) fn parse_nfl_source(p: &mut Parser) -> Result<NflSource, ParseError> {
    let mut models = Vec::new();
    p.skip_newlines();
    while !matches!(p.peek_kind(), TokenKind::Eof) {
        models.push(parse_model_def(p)?);
        p.skip_newlines();
    }
    Ok(NflSource { models })
}
