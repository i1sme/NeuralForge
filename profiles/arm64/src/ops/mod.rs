//! Per-op codegen modules.

pub mod dropout;
pub mod linear;
pub mod relu;
pub mod softmax;

pub use linear::emit_linear;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
