// SPDX-License-Identifier: Apache-2.0

//! Per-op codegen modules.

pub mod add;
pub mod dropout;
pub mod exp; // M17: bare-metal expf constant pool
pub mod layernorm; // M14
pub mod linear;
pub mod matmul;
pub mod mulscalar;
pub mod relu;
pub mod softmax;

pub use add::emit_add;
pub use dropout::emit_dropout_copy;
pub use exp::exp_pool_arm64; // M17
pub use layernorm::emit_layernorm; // M14
pub use linear::emit_linear;
pub use matmul::emit_matmul;
pub use mulscalar::emit_mulscalar;
pub use relu::emit_relu;
pub use softmax::emit_softmax;
