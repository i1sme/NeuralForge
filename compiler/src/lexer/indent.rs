// SPDX-License-Identifier: Apache-2.0

//! Indent stack — produces virtual INDENT/DEDENT tokens.
//!
//! Tracks the leading-space level of each non-empty (and non-comment-only)
//! line. The stack always has at least one entry (`0`). When a new line starts:
//! - Equal to top → no token emitted.
//! - Greater than top → push, emit one INDENT.
//! - Less than top → pop until equal, emitting one DEDENT per pop. If we run
//!   out of stack without finding an equal level, that is a `BadDedent`.
//!
//! This module is consumed by `lexer::mod` after a Newline has been emitted.

use super::tokens::{LexError, Token, TokenKind};

#[derive(Debug)]
pub(super) struct IndentStack {
    levels: Vec<usize>,
}

impl IndentStack {
    pub fn new() -> Self {
        Self { levels: vec![0] }
    }

    /// Adjust the stack to a new indent `level`. Pushes and emits Indent, or
    /// pops and emits Dedents. `line` and `col` are used for error reporting
    /// (col is the first non-whitespace column on the new line, 1-based).
    pub fn adjust_to(
        &mut self,
        level: usize,
        line: u32,
        col: u32,
        out: &mut Vec<Token>,
    ) -> Result<(), LexError> {
        let top = *self.levels.last().expect("stack always non-empty");
        if level == top {
            // Same indent — nothing to do.
            return Ok(());
        }
        if level > top {
            self.levels.push(level);
            out.push(Token::new(TokenKind::Indent, line, col));
            return Ok(());
        }
        // level < top — pop until equal.
        while *self.levels.last().expect("stack always non-empty") > level {
            self.levels.pop();
            out.push(Token::new(TokenKind::Dedent, line, col));
        }
        if *self.levels.last().expect("stack always non-empty") != level {
            return Err(LexError::BadDedent { line, col });
        }
        Ok(())
    }

    /// Close the file: emit a Dedent for every level above 0.
    pub fn close(&mut self, line: u32, col: u32, out: &mut Vec<Token>) {
        while self.levels.len() > 1 {
            self.levels.pop();
            out.push(Token::new(TokenKind::Dedent, line, col));
        }
    }

    pub fn current_top(&self) -> usize {
        *self.levels.last().expect("stack always non-empty")
    }
}
