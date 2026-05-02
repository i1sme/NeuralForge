//! Unit tests for the lexer.

use super::*;
use super::tokens::TokenKind::*;

fn lex_kinds(source: &str) -> Vec<TokenKind> {
    lex(source).unwrap().into_iter().map(|t| t.kind).collect()
}

#[test]
fn lex_keyword_model() {
    assert_eq!(lex_kinds("model"), vec![Model, Eof]);
}

#[test]
fn lex_keyword_tensor() {
    assert_eq!(lex_kinds("Tensor"), vec![Tensor, Eof]);
}

#[test]
fn lex_punctuation() {
    assert_eq!(
        lex_kinds("[ ] : , = ->"),
        vec![LBracket, RBracket, Colon, Comma, Equals, Arrow, Eof],
    );
}

#[test]
fn lex_identifier_simple() {
    assert_eq!(lex_kinds("foo"), vec![Ident("foo".into()), Eof]);
}

#[test]
fn lex_identifier_with_underscores_and_digits() {
    assert_eq!(lex_kinds("foo_bar2"), vec![Ident("foo_bar2".into()), Eof]);
}

#[test]
fn lex_integer() {
    assert_eq!(lex_kinds("512"), vec![Integer(512), Eof]);
}

#[test]
fn lex_float() {
    let toks = lex_kinds("0.2");
    assert_eq!(toks.len(), 2);
    match &toks[0] {
        Number(n) => assert!((n - 0.2).abs() < 1e-12),
        other => panic!("expected Number, got {other:?}"),
    }
    assert_eq!(toks[1], Eof);
}

#[test]
fn lex_token_positions() {
    // "model x" — 'model' at col 1, 'x' at col 7
    let toks = lex("model x").unwrap();
    assert_eq!(toks[0].line, 1);
    assert_eq!(toks[0].col, 1);
    assert_eq!(toks[1].line, 1);
    assert_eq!(toks[1].col, 7);
    assert_eq!(toks[1].kind, Ident("x".into()));
}

#[test]
fn lex_comment_alone() {
    // A comment-only file produces no token but Eof.
    assert_eq!(lex_kinds("# hello"), vec![Eof]);
}

#[test]
fn lex_comment_at_end_of_line() {
    // Comment after a token does not affect the token, and is consumed.
    assert_eq!(
        lex_kinds("model # ignored"),
        vec![Model, Eof],
    );
}

#[test]
fn lex_newline_lf() {
    assert_eq!(lex_kinds("model\nTensor"), vec![Model, Newline, Tensor, Eof]);
}

#[test]
fn lex_newline_crlf() {
    assert_eq!(lex_kinds("model\r\nTensor"), vec![Model, Newline, Tensor, Eof]);
}
