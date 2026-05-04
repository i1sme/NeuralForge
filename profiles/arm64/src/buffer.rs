//! Buffer assignment + leaf/callee-saved analyzers for the arm64 codegen.
//!
//! Pure analyzers over `UirModel`. No asm emission. Consumed by `codegen.rs`
//! in Task 3.

use compiler::{NodeId, NodeKind, StdOp, UirModel};

/// Where an Op-node's output buffer lives at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLoc {
    InputReg,
    OutputReg,
    StackOffset(usize),
    Alias(NodeId),
}

/// Result of buffer assignment.
pub struct BufferAssignment {
    pub locs: Vec<BufferLoc>,
    /// Total stack bytes required, rounded up to 16-byte alignment.
    pub stack_bytes: usize,
}

/// Assign a `BufferLoc` per UIR node + compute aligned total stack frame size.
pub fn assign_buffers(model: &UirModel) -> BufferAssignment {
    let mut locs = vec![BufferLoc::InputReg; model.nodes.len()];
    let mut stack_offset: usize = 0;

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
                            let size_bytes = node.ty.shape.0.iter().product::<u64>() as usize * 4;
                            let loc = BufferLoc::StackOffset(stack_offset);
                            stack_offset += size_bytes;
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

/// True iff the model emits no `bl`/`blr` (i.e. no softmax in M4b).
pub fn compute_is_leaf(model: &UirModel) -> bool {
    !model.nodes.iter().any(|n| {
        matches!(
            &n.kind,
            NodeKind::Op {
                op: StdOp::Softmax,
                ..
            }
        )
    })
}

/// Set of callee-saved registers used by the model's body. M4b: `{d8, d9}`
/// iff softmax is present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegSet {
    pub d8_d9: bool,
}

impl RegSet {
    pub fn contains_d8_d9(&self) -> bool {
        self.d8_d9
    }
}

pub fn compute_callee_saved(model: &UirModel) -> RegSet {
    RegSet {
        d8_d9: model.nodes.iter().any(|n| {
            matches!(
                &n.kind,
                NodeKind::Op {
                    op: StdOp::Softmax,
                    ..
                }
            )
        }),
    }
}
