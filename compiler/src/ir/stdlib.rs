//! Standard library of NFL operations (Milestone 3a defines four:
//! Linear, Relu, Dropout, Softmax). Functions `resolve`, `signature`,
//! and `infer_output_shape` land in Tasks 2-3.

use super::types::{AttrValue, OpAttr, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdOp {
    Linear,
    Relu,
    Dropout,
    Softmax,
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// Defensive guard. Emitted by `single_input` if a multi-operand op
    /// reaches single-input shape inference. No M3 op constructs >1 operand,
    /// so no test fires this path; the constructor exists so M5+ multi-input
    /// ops (add/concat) cannot silently misroute through single-input helpers.
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
        }
    }
}

pub fn resolve(name: &str) -> Option<StdOp> {
    match name {
        "linear" => Some(StdOp::Linear),
        "relu" => Some(StdOp::Relu),
        "dropout" => Some(StdOp::Dropout),
        "softmax" => Some(StdOp::Softmax),
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
    }
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
        StdOp::Linear | StdOp::Relu | StdOp::Softmax => Ok(()),
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
        };
        write!(f, "{}", name)
    }
}

/// True iff the op's attribute list includes `bias=true` (for `Linear`).
///
/// Used by the codegen profile to detect bias-add cases and by fusion
/// passes (M5a) to skip `linear[bias=true]` for Linear→Relu fusion in
/// the M5a slice.
pub fn linear_has_bias(attrs: &[crate::ir::types::OpAttr]) -> bool {
    use crate::ir::types::AttrValue;
    attrs
        .iter()
        .any(|a| a.name == "bias" && matches!(&a.value, AttrValue::Symbol(s) if s == "true"))
}
