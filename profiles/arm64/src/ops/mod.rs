//! Per-op codegen modules.

pub mod dropout;
pub mod linear;
pub mod relu;

pub use linear::emit_linear;
pub use relu::emit_relu;
