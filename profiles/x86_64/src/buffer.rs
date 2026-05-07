// SPDX-License-Identifier: Apache-2.0

//! Buffer assignment + leaf/callee-saved analyzers for the x86_64 codegen.
//!
//! Pure analyzers over `UirModel`. No asm emission. Mirrors the structure
//! of `profiles/arm64/src/buffer.rs` modulo register-naming differences.

use compiler::{NodeId, NodeKind, StdOp, UirModel};

/// Bytes per f32 element. f32-only project-wide.
const BYTES_PER_ELEMENT: usize = 4;

/// Where an Op-node's output buffer lives at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    /// Input pointer (FFI arg 0 — %rdi).
    InputReg,
    /// Output pointer (FFI arg 2 — %rdx).
    OutputReg,
    /// Stack slot at `[%rsp + offset]`.
    StackOffset(usize),
    /// This buffer is an alias for another node's buffer. Resolved by
    /// `codegen::resolve_loc` before any emit.
    Alias(NodeId),
}

/// Result of buffer assignment.
#[derive(Debug, Clone)]
pub struct BufferAssignment {
    /// Per-NodeId placement; index by NodeId.
    pub locs: Vec<BufferLoc>,
    /// Total stack bytes required for the function's frame (intermediate
    /// buffers + xmm-spill reserve), rounded up to 16-byte alignment.
    /// **Includes** the 16-byte fused-softmax reserve when
    /// `model.calls_extern_math()` is true (spec §7.4): in that case
    /// the reserve sits at offsets `0..15` and intermediate buffers
    /// start at `off >= 16`, so emitters can address the row_max /
    /// row_sum slots at fixed `(%rsp)` / `8(%rsp)` regardless of the
    /// model's intermediate-buffer footprint.
    pub stack_bytes: usize,
}

/// Assign a `BufferLoc` per UIR node + compute aligned total stack frame size.
pub fn assign_buffers(model: &UirModel) -> BufferAssignment {
    let mut locs = vec![BufferLoc::InputReg; model.nodes.len()];
    // Reserve 16 bytes at the bottom of the frame for fused-softmax
    // xmm-spill slots (row_max at 0(%rsp), row_sum at 8(%rsp)) when
    // any node calls libm-expf. Slots live at fixed offsets 0/8 (NOT
    // parameterised by stack_bytes), so all subsequent intermediate
    // buffers shift up by 16 — preventing slot/buffer overlap when
    // intermediate buffers are non-empty (spec §7.4).
    let mut stack_offset: usize = if model.calls_extern_math() { 16 } else { 0 };

    for (id, node) in model.nodes.iter().enumerate() {
        locs[id] = match &node.kind {
            NodeKind::Input { .. } => BufferLoc::InputReg,
            NodeKind::Op { op, operands, .. } => {
                if id == model.output {
                    BufferLoc::OutputReg
                } else {
                    match op {
                        StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0]),
                        StdOp::Linear | StdOp::Softmax => {
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
/// other 5 (`%rbx, %r12, %r13, %r14, %r15`) iff any node calls
/// `expf@PLT` — either a standalone `StdOp::Softmax` or a `Linear`
/// carrying `PostOp::SoftmaxRow`. UIR predicate
/// (`UirModel::calls_extern_math`) is the single source of truth.
///
/// Unlike arm64, **there is no callee-saved FP register set**. All
/// `%xmm0`-`%xmm15` are caller-saved per SysV. The fused softmax tail
/// spills row_max / row_sum to the stack across `call expf@PLT`
/// (see spec §7.4).
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

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    RegSet {
        callee_saved_int: model.calls_extern_math(),
    }
}
