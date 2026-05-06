// SPDX-License-Identifier: Apache-2.0

//! Errors raised while building UIR from AST.

#[derive(Debug, Clone, PartialEq)]
pub struct BuildError {
    pub message: String,
    pub line: u32,
    pub col: u32,
    pub kind: BuildErrorKind,
}

/// `#[non_exhaustive]` — adding new variants is non-breaking for downstream
/// `match` consumers. The `nflc` CLI and any future profile crates that
/// inspect `BuildErrorKind` (e.g. to extract a `first_span` for diagnostics)
/// must keep a `_ => ...` arm.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum BuildErrorKind {
    UnknownOp {
        name: String,
    },
    UnknownDim {
        name: String,
    },
    UnknownVariable {
        name: String,
    },
    ArgCountMismatch {
        expected: usize,
        actual: usize,
    },
    ArgTypeMismatch {
        slot: String,
        expected: String,
        actual: String,
    },
    MissingRequiredArg {
        slot: String,
    },
    UnexpectedNamedArg {
        name: String,
    },
    ShapeMismatch {
        detail: String,
    },
    ModelHasNoPipeline {
        name: String,
    },
    InvalidAttrValue {
        op: String,
        attr: String,
        reason: String,
    },
    DuplicateModelName {
        name: String,
        first_span: crate::ast::Span,
    },
}

impl std::fmt::Display for BuildErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildErrorKind::DuplicateModelName { name, .. } => write!(
                f,
                "duplicate model name '{}': would emit conflicting symbols",
                name
            ),
            // All other variants rely on the pre-filled message in BuildError.
            _ => Ok(()),
        }
    }
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            write!(f, "{}", self.kind)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for BuildError {}

impl BuildError {
    pub fn unknown_op(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("unknown operation: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownOp {
                name: name.to_string(),
            },
        }
    }

    pub fn unknown_dim(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!(
                "unknown symbolic dimension: '{}' (not declared in model_params)",
                name
            ),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownDim {
                name: name.to_string(),
            },
        }
    }

    pub fn unknown_variable(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("unknown variable: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnknownVariable {
                name: name.to_string(),
            },
        }
    }

    pub fn arg_count_mismatch(expected: usize, actual: usize, span: crate::ast::Span) -> Self {
        Self {
            message: format!(
                "operation expects {} positional argument(s), got {}",
                expected, actual
            ),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ArgCountMismatch { expected, actual },
        }
    }

    pub fn arg_type_mismatch(
        slot: &str,
        expected: &str,
        actual: &str,
        span: crate::ast::Span,
    ) -> Self {
        Self {
            message: format!("argument '{}' expects {}, got {}", slot, expected, actual),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ArgTypeMismatch {
                slot: slot.to_string(),
                expected: expected.to_string(),
                actual: actual.to_string(),
            },
        }
    }

    pub fn missing_required_arg(slot: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("missing required argument: '{}'", slot),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::MissingRequiredArg {
                slot: slot.to_string(),
            },
        }
    }

    pub fn unexpected_named_arg(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("operation does not accept named argument: '{}'", name),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::UnexpectedNamedArg {
                name: name.to_string(),
            },
        }
    }

    pub fn shape(detail: String, span: crate::ast::Span) -> Self {
        Self {
            message: format!("shape error: {}", detail),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ShapeMismatch { detail },
        }
    }

    pub fn model_has_no_pipeline(name: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!(
                "model '{}' has no pipeline_stmt — output is undefined",
                name
            ),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::ModelHasNoPipeline {
                name: name.to_string(),
            },
        }
    }

    pub fn invalid_attr_value(op: &str, attr: &str, reason: &str, span: crate::ast::Span) -> Self {
        Self {
            message: format!("invalid value for {}.{}: {}", op, attr, reason),
            line: span.line,
            col: span.col,
            kind: BuildErrorKind::InvalidAttrValue {
                op: op.to_string(),
                attr: attr.to_string(),
                reason: reason.to_string(),
            },
        }
    }

    pub fn duplicate_model_name(
        name: String,
        first_span: crate::ast::Span,
        current_span: crate::ast::Span,
    ) -> Self {
        let message = format!(
            "duplicate model name '{}': would emit conflicting symbols",
            name
        );
        Self {
            message,
            line: current_span.line,
            col: current_span.col,
            kind: BuildErrorKind::DuplicateModelName { name, first_span },
        }
    }
}
