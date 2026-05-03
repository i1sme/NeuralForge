//! Standard library of NFL operations (Milestone 3a defines four:
//! Linear, Relu, Dropout, Softmax). Functions `resolve`, `signature`,
//! and `infer_output_shape` land in Tasks 2-3.

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
    WrongInputCount { expected: usize, actual: usize },
    WrongRank { expected: usize, actual: usize, dim_index: Option<usize> },
    MissingAttr { name: &'static str },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::WrongInputCount { expected, actual } =>
                write!(f, "expected {} input(s), got {}", expected, actual),
            ShapeError::WrongRank { expected, actual, dim_index: _ } =>
                write!(f, "expected rank {}, got {}", expected, actual),
            ShapeError::MissingAttr { name } =>
                write!(f, "missing required attribute: '{}'", name),
        }
    }
}
