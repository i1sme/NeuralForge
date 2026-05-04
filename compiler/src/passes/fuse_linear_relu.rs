//! `linear → relu` fusion pass. See spec §7 for the algorithm.
//!
//! M5a Task 4 fills this in.

use super::{PassError, UirPass};
use crate::Uir;

pub struct FuseLinearRelu;

impl UirPass for FuseLinearRelu {
    fn name(&self) -> &str {
        "fuse_linear_relu"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        // Stub: identity transform. Task 4 replaces with real algorithm.
        Ok(uir.clone())
    }
}
