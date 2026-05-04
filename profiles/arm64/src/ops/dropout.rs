//! Dropout codegen.
//!
//! At inference, dropout is identity. The buffer-assignment first-pass
//! (`buffer.rs::assign_buffers`) returns `BufferLoc::Alias(operand)` for
//! dropout nodes; therefore no asm is emitted. This module exists as a
//! marker so the ops/ directory has parallel structure for all StdOps.
