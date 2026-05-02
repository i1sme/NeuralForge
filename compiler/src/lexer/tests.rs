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

#[test]
fn lex_simple_indent_block() {
    // Two-line block: header colon, then indented body line.
    let src = "model X:\n    foo\n";
    assert_eq!(
        lex_kinds(src),
        vec![Model, Ident("X".into()), Colon, Newline, Indent, Ident("foo".into()), Newline, Dedent, Eof],
    );
}

#[test]
fn lex_indent_then_dedent_back_to_zero() {
    let src = "model X:\n    foo\nbar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Dedent, Ident("bar".into()), Newline,
            Eof,
        ],
    );
}

#[test]
fn lex_blank_lines_do_not_affect_indent() {
    // Blank line in middle of body is ignored for indent purposes.
    let src = "model X:\n    foo\n\n    bar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Newline,
            Ident("bar".into()), Newline,
            Dedent, Eof,
        ],
    );
}

#[test]
fn lex_comment_only_line_does_not_affect_indent() {
    // Indented body, then a comment-only line at column 0, then more body.
    // The comment line is treated as blank; indent does NOT close.
    let src = "model X:\n    foo\n# top-level comment\n    bar\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Newline,                                   // for the comment-bearing line
            Ident("bar".into()), Newline,
            Dedent, Eof,
        ],
    );
}

#[test]
fn lex_dedent_at_eof() {
    // EOF should emit any pending Dedents.
    let src = "model X:\n    foo\n";
    let toks = lex(src).unwrap();
    let last_three: Vec<&TokenKind> = toks.iter().rev().take(3).map(|t| &t.kind).collect();
    // Last three tokens: Eof, Dedent, Newline (in reverse order of the stream).
    assert_eq!(last_three, vec![&Eof, &Dedent, &Newline]);
}

#[test]
fn lex_nested_indent_dedent() {
    // Two levels: model body at indent 4, deeper indent at 8.
    // For grammar v0.1 there is no production using nested blocks, but the
    // lexer must still handle the mechanics correctly so future grammars work.
    let src = "model X:\n    foo\n        bar\n    baz\n";
    assert_eq!(
        lex_kinds(src),
        vec![
            Model, Ident("X".into()), Colon, Newline,
            Indent, Ident("foo".into()), Newline,
            Indent, Ident("bar".into()), Newline,
            Dedent, Ident("baz".into()), Newline,
            Dedent, Eof,
        ],
    );
}
