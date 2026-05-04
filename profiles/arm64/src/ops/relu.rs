//! Relu (elementwise max with zero) codegen.

use crate::buffer::BufferLoc;
use crate::ops::linear::materialise_ptr;

pub fn emit_relu(
    total_floats: u64,
    relu_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
) -> String {
    let rid = relu_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; relu: copy-clamp from src to dst ({total_floats} elements)\n"
    ));
    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    s.push_str("    fmov    s4, wzr\n");
    s.push_str("    mov     x9, #0\n");
    s.push_str(&format!(".Lrelu_{rid}:\n"));
    s.push_str(&format!("    cmp     x9, #{total_floats}\n"));
    s.push_str(&format!("    b.ge    .Lrelu_end_{rid}\n"));
    s.push_str("    ldr     s3, [x11, x9, lsl #2]\n");
    s.push_str("    fmax    s3, s3, s4\n");
    s.push_str("    str     s3, [x12, x9, lsl #2]\n");
    s.push_str("    add     x9, x9, #1\n");
    s.push_str(&format!("    b       .Lrelu_{rid}\n"));
    s.push_str(&format!(".Lrelu_end_{rid}:\n"));
    s
}
