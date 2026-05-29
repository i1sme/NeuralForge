// SPDX-License-Identifier: Apache-2.0

//! Buffer assignment + leaf/callee-saved analyzers for the x86_64 codegen.
//!
//! Pure analyzers over `UirModel`. No asm emission. Mirrors the structure
//! of `profiles/arm64/src/buffer.rs` modulo register-naming differences.

use compiler::{NodeKind, StdOp, UirModel};
pub use profile_api::BufferLoc;

/// Bytes per f32 element. f32-only project-wide.
const BYTES_PER_ELEMENT: usize = 4;

/// Result of buffer assignment.
#[derive(Debug, Clone)]
pub struct BufferAssignment {
    /// Per-NodeId placement; index by NodeId.
    pub locs: Vec<BufferLoc>,
    /// Total stack bytes required for the function's frame (intermediate
    /// buffers + xmm-spill reserve), rounded up to 16-byte alignment.
    /// **Includes** the 16-byte fused-softmax reserve when
    /// `model.has_softmax()` is true (spec §7.4): in that case
    /// the reserve sits at offsets `0..15` and intermediate buffers
    /// start at `off >= 16`, so emitters can address the row_max /
    /// row_sum slots at fixed `(%rsp)` / `8(%rsp)` regardless of the
    /// model's intermediate-buffer footprint.
    pub stack_bytes: usize,
}

/// Assign a `BufferLoc` per UIR node + compute aligned total stack frame size.
pub fn assign_buffers(model: &UirModel) -> BufferAssignment {
    let mut locs = vec![BufferLoc::InputReg(0); model.nodes.len()];
    // Reserve 16 bytes at the bottom of the frame for fused-softmax
    // xmm-spill slots (row_max at 0(%rsp), row_sum at 8(%rsp)) when
    // the model has softmax. Slots live at fixed offsets 0/8 (NOT
    // parameterised by stack_bytes), so all subsequent intermediate
    // buffers shift up by 16 — preventing slot/buffer overlap when
    // intermediate buffers are non-empty (spec §7.4).
    let mut stack_offset: usize = if model.has_softmax() { 16 } else { 0 };

    for (id, node) in model.nodes.iter().enumerate() {
        locs[id] = match &node.kind {
            NodeKind::Input { .. } => {
                // Find this node's index in `model.inputs` (declaration
                // order). This index is later mapped to an ABI register
                // by `AbiContext::input_reg(idx)` in the codegen pass.
                let idx = model
                    .inputs
                    .iter()
                    .position(|&i| i == id)
                    .expect("Input node must appear in model.inputs");
                BufferLoc::InputReg(idx)
            }
            NodeKind::Op { op, operands, .. } => {
                if id == model.output {
                    BufferLoc::OutputReg
                } else {
                    match op {
                        StdOp::Relu | StdOp::Dropout | StdOp::MulScalar => {
                            BufferLoc::Alias(operands[0])
                        }
                        StdOp::Linear | StdOp::Softmax | StdOp::Matmul => {
                            let elements: u64 = node.ty.shape.0.iter().copied().product();
                            let size_bytes = (elements as usize)
                                .checked_mul(BYTES_PER_ELEMENT)
                                .expect("buffer size overflow");
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset = stack_offset
                                .checked_add(size_bytes)
                                .expect("stack frame size overflow");
                            loc
                        }
                        #[allow(unreachable_patterns)]
                        _ => {
                            let elements: u64 = node.ty.shape.0.iter().copied().product();
                            let size_bytes = (elements as usize)
                                .checked_mul(BYTES_PER_ELEMENT)
                                .expect("buffer size overflow");
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset = stack_offset
                                .checked_add(size_bytes)
                                .expect("stack frame size overflow");
                            loc
                        }
                    }
                }
            }
        };
    }

    let stack_bytes = (stack_offset + 15) & !15;
    BufferAssignment { locs, stack_bytes }
}

/// Set of callee-saved int registers used by the model's body.
///
/// SysV AMD64 callee-saved int set: `%rbx, %rbp, %r12, %r13, %r14, %r15`.
/// This profile uses `%rbp` as the frame pointer (always saved) and the
/// other 5 (`%rbx, %r12, %r13, %r14, %r15`) iff:
///  - the model has softmax (`StdOp::Softmax` or `Linear` carrying
///    `PostOp::SoftmaxRow`) — its loop holds state in these registers, OR
///  - the model contains an `StdOp::Matmul` (M12 — `emit_matmul` body
///    uses callee-saved scratch to avoid touching ABI argument
///    registers per spec §9.1).
///
/// Both triggers fire independently: a model with neither softmax nor
/// matmul gets a leaf prologue (no callee-saved pushes); a model with
/// either OR both triggers gets the full 5-register save block. For
/// the M3-M11 fixture set, only `self_attention.nfl` has matmul AND
/// it also has softmax — so adding the matmul trigger does not change
/// any pre-M12 fixture's prologue (the softmax trigger already fires
/// for it). New M12 multi-input fixtures (`two_input_matmul.nfl`)
/// have matmul-only and rely on this trigger.
///
/// Unlike arm64, **there is no callee-saved FP register set**. All
/// `%xmm0`-`%xmm15` are caller-saved per SysV. The fused softmax tail
/// spills row_max / row_sum to the stack across the inline exp's
/// scratch usage (M17; the stack slots are retained, removed in M18
/// — see spec §7.4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegSet {
    /// True iff `%rbx, %r12, %r13, %r14, %r15` are saved in the
    /// prologue (and restored in the epilogue).
    pub callee_saved_int: bool,
}

impl RegSet {
    pub fn contains_callee_saved_int(&self) -> bool {
        self.callee_saved_int
    }
}

/// True iff the model contains at least one matmul op. M12: triggers
/// callee-saved-int register saves so `emit_matmul` body can use
/// `%r12`/`%r13`/`%r14`/`%r15`/`%rbx` as long-lived scratch (per spec
/// §9.1, matmul body must not touch ABI argument registers, leaving
/// only 3-4 caller-saved non-ABI GP regs at N=3 — insufficient for the
/// 3 base + 3 slice + 3 counter register-roles, hence callee-saved
/// fallback).
fn has_matmul(model: &UirModel) -> bool {
    model.nodes.iter().any(|n| {
        matches!(
            n.kind,
            NodeKind::Op {
                op: StdOp::Matmul,
                ..
            }
        )
    })
}

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    RegSet {
        callee_saved_int: model.has_softmax() || has_matmul(model),
    }
}
