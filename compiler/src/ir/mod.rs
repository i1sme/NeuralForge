//! Universal IR — the typed computation graph the compiler produces from
//! the parsed AST. Consumed by architecture profiles (M4+) and optimisation
//! passes (M5+).

// Many items below are introduced in Task 1 but only consumed once
// `pub fn build` is wired in Task 7 (and reachable from outside the crate
// via lib.rs re-exports). Until then `cargo build` (lib only) flags the
// helper chain as unused. Removed in Task 8.
#![allow(dead_code)]

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
