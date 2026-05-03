//! Universal IR — the typed computation graph the compiler produces from
//! the parsed AST. Consumed by architecture profiles (M4+) and optimisation
//! passes (M5+).

pub mod types;
pub mod stdlib;
pub mod error;
mod build;

#[cfg(test)]
mod tests;

pub use error::{BuildError, BuildErrorKind};
pub use stdlib::{ArgSlot, ArgType, Signature, StdOp};
pub use types::{AttrValue, Node, NodeId, NodeKind, OpAttr, Shape, Type, Uir, UirModel};

pub use build::build;
