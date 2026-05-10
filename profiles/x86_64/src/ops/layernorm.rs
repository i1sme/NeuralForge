// SPDX-License-Identifier: Apache-2.0

//! LayerNorm codegen — x86_64 (M14).
//!
//! 3-pass per row (mean → variance + inv_std → normalize + optional affine).
//! Native `sqrtss` — no libm call. AT&T syntax, SysV AMD64 ABI.
//!
//! Stub — real implementation lands in a follow-up commit.

use crate::abi::AbiContext;
use crate::buffer::BufferLoc;
use compiler::ast::Span;
use profile_api::LowerError;

#[allow(clippy::too_many_arguments)]
pub fn emit_layernorm(
    _abi: &AbiContext,
    _b: u64,
    _d: u64,
    _model_idx: usize,
    _layernorm_idx: usize,
    _src_loc: BufferLoc,
    _dst_loc: BufferLoc,
    _gamma_offset: Option<usize>,
    _beta_offset: Option<usize>,
    node_span: Span,
) -> Result<String, LowerError> {
    Err(LowerError::UnsupportedOp {
        op: "layernorm".to_string(),
        span: node_span,
    })
}
