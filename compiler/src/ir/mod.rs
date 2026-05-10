// SPDX-License-Identifier: Apache-2.0

//! Universal IR — the typed computation graph the compiler produces from
//! the parsed AST. Consumed by architecture profiles (M4+) and optimisation
//! passes (M5+).

mod build;
pub mod error;
pub mod stdlib;
pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod test_utils;

pub use error::{BuildError, BuildErrorKind};
pub use stdlib::{layernorm_has_affine, linear_has_bias, ArgSlot, ArgType, Signature, StdOp};
pub use types::{AttrValue, Node, NodeId, NodeKind, OpAttr, PostOp, Shape, Type, Uir, UirModel};

pub use build::build;
