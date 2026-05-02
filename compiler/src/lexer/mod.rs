//! Hand-written lexer for NFL.

pub mod tokens;

#[cfg(test)]
mod tests;

pub use tokens::{LexError, Token, TokenKind};

/// Tokenise NFL source text into a flat token stream ending with `Eof`.
///
/// Currently handles single-token-per-line inputs. Indentation, comments,
/// newlines, and pipeline continuation are added in later tasks.
pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let line: u32 = 1;
    let mut col: u32 = 1;

    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];

        // Skip horizontal whitespace (space and tab) outside of leading whitespace.
        // (Leading whitespace and newlines are added in later tasks; for now we
        // accept any single-line input separated by spaces.)
        if b == b' ' || b == b'\t' {
            i += 1;
            col += 1;
            continue;
        }

        // Punctuation singletons.
        let single: Option<TokenKind> = match b {
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

        // Arrow '->'.
        if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            tokens.push(Token::new(TokenKind::Arrow, line, col));
            i += 2;
            col += 2;
            continue;
        }

        // Identifier or keyword: starts with letter, continues with letter/digit/underscore.
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

        // Number literal: integer or float.
        if b.is_ascii_digit() {
            let start = i;
            let start_col = col;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
                col += 1;
            }
            // Optional fractional part: "." followed by at least one digit.
            let has_fractional = i + 1 < bytes.len()
                && bytes[i] == b'.'
                && bytes[i + 1].is_ascii_digit();
            if has_fractional {
                i += 1; // consume '.'
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

        // Anything else is an error for now.
        return Err(LexError::UnknownChar {
            line,
            col,
            ch: b as char,
        });
    }

    tokens.push(Token::new(TokenKind::Eof, line, col));
    Ok(tokens)
}
