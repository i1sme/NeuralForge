// SPDX-License-Identifier: Apache-2.0

//! Standard library of NFL operations (Milestone 3a defines four:
//! Linear, Relu, Dropout, Softmax). Functions `resolve`, `signature`,
//! and `infer_output_shape` land in Tasks 2-3.

use super::types::{AttrValue, OpAttr, Shape};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
    /// Matrix multiplication, rank ≥ 2 inputs. With `transpose_b=true`
    /// (named arg), the second operand's last two dims are interpreted
    /// transposed. New in M10.
    Matmul,
}

pub struct Signature {
    pub positional: &'static [ArgSlot],
    pub named: &'static [ArgSlot],
}

pub struct ArgSlot {
    pub name: &'static str,
    pub ty: ArgType,
    pub required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    Integer,
    Float,
    Symbol,
    /// A tensor-by-name argument. The arg appears in NFL source as an
    /// identifier (e.g. `matmul[x]` where `x` is a previously-declared
    /// variable name). The builder resolves it against the variable
    /// environment to a `NodeId`. Resolved IDs go into the op node's
    /// `operands` field, NOT `attrs`. New in M10.
    Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Defensive guard. Emitted by `single_input` if a multi-operand op
    /// reaches single-input shape inference. M5+ multi-input ops (Matmul)
    /// cannot silently misroute through single-input helpers.
    WrongInputCount {
        expected: usize,
        actual: usize,
    },
    WrongRank {
        expected: usize,
        actual: usize,
        dim_index: Option<usize>,
    },
    MissingAttr {
        name: &'static str,
    },
    /// Two operands have different ranks (e.g. `[2, 4] @ [2, 4, 8, 8]`).
    RankMismatch {
        lhs: usize,
        rhs: usize,
    },
    /// Operand rank is below the minimum required by the op
    /// (e.g. 1D input to Matmul, which requires rank ≥ 2).
    RankTooLow {
        required: usize,
        actual: usize,
    },
    /// Two operands' leading dims (indices `0..rank-2`) disagree.
    /// Strict-equal — no broadcasting per design principle #1.
    LeadingDimMismatch {
        dim_index: usize,
        lhs: u64,
        rhs: u64,
    },
    /// Matmul contraction dim disagreement.
    /// `lhs_k` is `a.shape[rank-1]`. `rhs_k` is `b.shape[rank-1]` if
    /// `transpose_b=true`, otherwise `b.shape[rank-2]`.
    InnerDimMismatch {
        lhs_k: u64,
        rhs_k: u64,
        transpose_b: bool,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::WrongInputCount { expected, actual } => {
                write!(f, "expected {} input(s), got {}", expected, actual)
            }
            ShapeError::WrongRank {
                expected,
                actual,
                dim_index: _,
            } => write!(f, "expected rank {}, got {}", expected, actual),
            ShapeError::MissingAttr { name } => write!(f, "missing required attribute: '{}'", name),
            ShapeError::RankMismatch { lhs, rhs } => write!(
                f,
                "operand rank mismatch: lhs has rank {}, rhs has rank {}",
                lhs, rhs
            ),
            ShapeError::RankTooLow { required, actual } => write!(
                f,
                "operand rank too low: requires {}, got {}",
                required, actual
            ),
            ShapeError::LeadingDimMismatch {
                dim_index,
                lhs,
                rhs,
            } => write!(
                f,
                "leading dim mismatch at index {}: lhs={}, rhs={} (no broadcasting)",
                dim_index, lhs, rhs
            ),
            ShapeError::InnerDimMismatch {
                lhs_k,
                rhs_k,
                transpose_b,
            } => write!(
                f,
                "matmul contraction dim mismatch: lhs.K={}, rhs.K={}, transpose_b={}",
                lhs_k, rhs_k, transpose_b
            ),
        }
    }
}

pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear" => Some(StdOp::Linear),
        "relu" => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
        "matmul" => Some(StdOp::Matmul),
        _ => None,
    }
}

pub fn signature(op: StdOp) -> Signature {
    use ArgType::*;
    match op {
        StdOp::Linear => Signature {
            positional: &[ArgSlot {
                name: "out_dim",
                ty: Integer,
                required: true,
            }],
            named: &[ArgSlot {
                name: "bias",
                ty: Symbol,
                required: false,
            }],
        },
        StdOp::Relu => Signature {
            positional: &[],
            named: &[],
        },
        StdOp::Dropout => Signature {
            positional: &[],
            named: &[ArgSlot {
                name: "rate",
                ty: Float,
                required: true,
            }],
        },
        StdOp::Softmax => Signature {
            positional: &[],
            named: &[],
        },
        StdOp::Matmul => Signature {
            positional: &[ArgSlot {
                name: "other",
                ty: Tensor,
                required: true,
            }],
            named: &[ArgSlot {
                name: "transpose_b",
                ty: Symbol,
                required: false,
            }],
        },
    }
}

pub fn infer_output_shape(
    op: StdOp,
    inputs: &[Shape],
    attrs: &[OpAttr],
) -> Result<Shape, ShapeError> {
    match op {
        StdOp::Linear => {
            let input = single_input(inputs)?;
            require_rank(input, 2)?;
            let out_dim = get_int_attr(attrs, "out_dim")?;
            Ok(Shape(vec![input.0[0], out_dim]))
        }
        StdOp::Relu | StdOp::Softmax | StdOp::Dropout => {
            let input = single_input(inputs)?;
            Ok(input.clone())
        }
        StdOp::Matmul => infer_matmul_shape(inputs, attrs),
    }
}

/// Matmul shape inference: rank ≥ 2 inputs of equal rank, leading dims
/// strict-equal (no broadcasting), inner-dim contraction matches.
/// Output: leading dims of `a` followed by `[M, N]` where M = a's
/// second-to-last and N is the non-contracted dim of `b`.
fn infer_matmul_shape(inputs: &[Shape], attrs: &[OpAttr]) -> Result<Shape, ShapeError> {
    // Step 1: input count.
    if inputs.len() != 2 {
        return Err(ShapeError::WrongInputCount {
            expected: 2,
            actual: inputs.len(),
        });
    }
    let a = &inputs[0];
    let b = &inputs[1];

    // Step 2: ranks match.
    if a.rank() != b.rank() {
        return Err(ShapeError::RankMismatch {
            lhs: a.rank(),
            rhs: b.rank(),
        });
    }

    // Step 3: rank ≥ 2.
    if a.rank() < 2 {
        return Err(ShapeError::RankTooLow {
            required: 2,
            actual: a.rank(),
        });
    }
    let r = a.rank();

    // Step 4: leading dims (indices 0..r-2) match exactly.
    for i in 0..(r - 2) {
        if a.0[i] != b.0[i] {
            return Err(ShapeError::LeadingDimMismatch {
                dim_index: i,
                lhs: a.0[i],
                rhs: b.0[i],
            });
        }
    }

    // Step 5: inner contraction.
    let transpose_b = matmul_transpose_b(attrs);
    let m = a.0[r - 2];
    let lhs_k = a.0[r - 1];
    let (rhs_k, n) = if transpose_b {
        // b shape [..., N, K] — contract on b's last dim.
        (b.0[r - 1], b.0[r - 2])
    } else {
        // b shape [..., K, N] — contract on b's second-to-last dim.
        (b.0[r - 2], b.0[r - 1])
    };
    if lhs_k != rhs_k {
        return Err(ShapeError::InnerDimMismatch {
            lhs_k,
            rhs_k,
            transpose_b,
        });
    }

    // Output: leading dims + [M, N].
    let mut out = Vec::with_capacity(r);
    out.extend_from_slice(&a.0[..(r - 2)]);
    out.push(m);
    out.push(n);
    Ok(Shape(out))
}

fn single_input(inputs: &[Shape]) -> Result<&Shape, ShapeError> {
    if inputs.len() == 1 {
        Ok(&inputs[0])
    } else {
        Err(ShapeError::WrongInputCount {
            expected: 1,
            actual: inputs.len(),
        })
    }
}

fn require_rank(s: &Shape, expected: usize) -> Result<(), ShapeError> {
    if s.rank() == expected {
        Ok(())
    } else {
        Err(ShapeError::WrongRank {
            expected,
            actual: s.rank(),
            dim_index: None,
        })
    }
}

fn get_int_attr(attrs: &[OpAttr], name: &'static str) -> Result<u64, ShapeError> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value {
            AttrValue::Integer(n) => Some(n),
            _ => None,
        })
        .ok_or(ShapeError::MissingAttr { name })
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrError {
    OutOfRange {
        name: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    MissingAttr {
        name: &'static str,
    },
}

impl std::fmt::Display for AttrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrError::OutOfRange {
                name,
                value,
                min,
                max,
            } => write!(
                f,
                "attribute '{}' value {} is outside [{}, {}]",
                name, value, min, max
            ),
            AttrError::MissingAttr { name } => write!(f, "missing required attribute: '{}'", name),
        }
    }
}

pub fn validate_attrs(op: StdOp, attrs: &[OpAttr]) -> Result<(), AttrError> {
    match op {
        StdOp::Dropout => {
            let rate = get_float_attr(attrs, "rate")?;
            if !(0.0..=1.0).contains(&rate) {
                return Err(AttrError::OutOfRange {
                    name: "rate",
                    value: rate,
                    min: 0.0,
                    max: 1.0,
                });
            }
            Ok(())
        }
        StdOp::Linear | StdOp::Relu | StdOp::Softmax | StdOp::Matmul => Ok(()),
    }
}

fn get_float_attr(attrs: &[OpAttr], name: &'static str) -> Result<f64, AttrError> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value {
            AttrValue::Float(f) => Some(f),
            _ => None,
        })
        .ok_or(AttrError::MissingAttr { name })
}

impl std::fmt::Display for StdOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            StdOp::Linear => "linear",
            StdOp::Relu => "relu",
            StdOp::Dropout => "dropout",
            StdOp::Softmax => "softmax",
            StdOp::Matmul => "matmul",
        };
        write!(f, "{}", name)
    }
}

/// True iff the op's attribute list includes `bias=true`.
///
/// Used by the arm64 codegen profile to detect bias-add cases and by
/// kernel-fusion passes that need to inspect the Linear's bias presence.
pub fn linear_has_bias(attrs: &[OpAttr]) -> bool {
    attrs
        .iter()
        .any(|a| a.name == "bias" && matches!(&a.value, AttrValue::Symbol(s) if s == "true"))
}

/// True iff the op's attribute list includes `transpose_b=true`.
///
/// Used by Matmul shape inference and by both arm64 and x86_64 codegen
/// to choose the inner-loop addressing pattern for the B operand.
/// New in M10.
pub fn matmul_transpose_b(attrs: &[OpAttr]) -> bool {
    attrs
        .iter()
        .any(|a| a.name == "transpose_b" && matches!(&a.value, AttrValue::Symbol(s) if s == "true"))
}
