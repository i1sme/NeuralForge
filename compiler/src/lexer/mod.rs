//! Hand-written lexer for NFL.

pub mod tokens;
mod indent;

#[cfg(test)]
mod tests;

pub use tokens::{LexError, Token, TokenKind};

use indent::IndentStack;

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut stack = IndentStack::new();

    let bytes = source.as_bytes();
    let mut i = 0;
    let mut line: u32 = 1;

    while i < bytes.len() {
        // Beginning of a line. Count leading spaces (and reject leading tabs).
        let line_start = i;
        let mut indent_spaces: usize = 0;
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            if bytes[i] == b'\t' {
                let col = (i - line_start) as u32 + 1;
                return Err(LexError::TabInIndent { line, col });
            }
            indent_spaces += 1;
            i += 1;
        }

        // Decide whether this is a blank/comment-only line or a content line.
        let line_is_blank = i >= bytes.len() || bytes[i] == b'\n' || bytes[i] == b'\r';
        let line_is_comment_only = !line_is_blank && bytes[i] == b'#';

        if line_is_blank || line_is_comment_only {
            // Eat the rest of the line up to (but not including) the newline.
            while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            // Eat the newline (LF or CRLF) and emit a Newline token.
            if i < bytes.len() {
                let col = (i - line_start) as u32 + 1;
                tokens.push(Token::new(TokenKind::Newline, line, col));
                if bytes[i] == b'\r' {
                    i += 1;
                    if i < bytes.len() && bytes[i] == b'\n' {
                        i += 1;
                    }
                } else {
                    i += 1; // LF
                }
                line += 1;
            }
            continue;
        }

        // Content line: adjust indent stack to the leading-space count.
        let first_col = indent_spaces as u32 + 1;
        stack.adjust_to(indent_spaces, line, first_col, &mut tokens)?;

        // Lex tokens on this line up to (but not including) the newline.
        let mut col: u32 = first_col;
        while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
            let b = bytes[i];

            // Inter-token whitespace (space and tab) — silently consumed.
            if b == b' ' || b == b'\t' {
                i += 1;
                col += 1;
                continue;
            }

            // Trailing comment on this line.
            if b == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                    col += 1;
                }
                break;
            }

            // Punctuation.
            let single = match b {
                b'[' => Some(TokenKind::LBracket),
                b']' => Some(TokenKind::RBracket),
                b':' => Some(TokenKind::Colon),
                b',' => Some(TokenKind::Comma),
                b'=' => Some(TokenKind::Equals),
                _ => None,
            };
            if let Some(kind) = single {
                tokens.push(Token::new(kind, line, col));
                i += 1;
                col += 1;
                continue;
            }

            // Arrow.
            if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                tokens.push(Token::new(TokenKind::Arrow, line, col));
                i += 2;
                col += 2;
                continue;
            }

            // Identifier or keyword.
            if b.is_ascii_alphabetic() {
                let start = i;
                let start_col = col;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                    col += 1;
                }
                let ident = std::str::from_utf8(&bytes[start..i])
                    .expect("ASCII identifier")
                    .to_string();
                let kind = match ident.as_str() {
                    "model" => TokenKind::Model,
                    "Tensor" => TokenKind::Tensor,
                    _ => TokenKind::Ident(ident),
                };
                tokens.push(Token::new(kind, line, start_col));
                continue;
            }

            // Number literal.
            if b.is_ascii_digit() {
                let start = i;
                let start_col = col;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
                let has_fractional = i + 1 < bytes.len()
                    && bytes[i] == b'.'
                    && bytes[i + 1].is_ascii_digit();
                if has_fractional {
                    i += 1;
                    col += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                        col += 1;
                    }
                    let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII number");
                    let value: f64 = raw.parse().map_err(|_| LexError::BadNumber {
                        line,
                        col: start_col,
                        raw: raw.to_string(),
                    })?;
                    tokens.push(Token::new(TokenKind::Number(value), line, start_col));
                } else {
                    let raw = std::str::from_utf8(&bytes[start..i]).expect("ASCII integer");
                    let value: u64 = raw.parse().map_err(|_| LexError::BadNumber {
                        line,
                        col: start_col,
                        raw: raw.to_string(),
                    })?;
                    tokens.push(Token::new(TokenKind::Integer(value), line, start_col));
                }
                continue;
            }

            return Err(LexError::UnknownChar { line, col, ch: b as char });
        }

        // End of content line: emit Newline if we are at one, then advance.
        if i < bytes.len() {
            let nl_col = col;
            tokens.push(Token::new(TokenKind::Newline, line, nl_col));
            if bytes[i] == b'\r' {
                i += 1;
                if i < bytes.len() && bytes[i] == b'\n' {
                    i += 1;
                }
            } else {
                i += 1;
            }
            line += 1;
        }
    }

    // EOF: close any open indent levels with synthetic Dedents at (line, col=1).
    stack.close(line, 1, &mut tokens);
    tokens.push(Token::new(TokenKind::Eof, line, 1));
    Ok(tokens)
}
