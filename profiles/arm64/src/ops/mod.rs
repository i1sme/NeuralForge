// SPDX-License-Identifier: Apache-2.0

//! Per-op codegen modules.

pub mod add;
pub mod dropout;
pub mod linear;
pub mod matmul;
pub mod mulscalar;
pub mod relu;
pub mod softmax;

pub use add::emit_add;
pub use dropout::emit_dropout_copy;
pub use linear::emit_linear;
pub use matmul::emit_matmul;
pub use mulscalar::emit_mulscalar;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
