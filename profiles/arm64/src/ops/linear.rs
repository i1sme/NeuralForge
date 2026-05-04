//! Linear (matmul + optional bias-add) codegen.

use crate::buffer::BufferLoc;

pub fn emit_linear(
    b: u64,
    k: u64,
    n: u64,
    linear_idx: usize,
    src_loc: BufferLoc,
    dst_loc: BufferLoc,
    weight_offset: usize,
) -> String {
    let lid = linear_idx;
    let mut s = String::new();
    s.push_str(&format!(
        "    ; matmul: input [{b},{k}] x weights [{k},{n}] -> output [{b},{n}]\n"
    ));

    s.push_str(&materialise_ptr("x11", src_loc));
    s.push_str(&materialise_ptr("x12", dst_loc));
    if weight_offset == 0 {
        s.push_str("    mov     x13, x1\n");
    } else {
        s.push_str(&format!("    mov     x9, #{}\n", weight_offset));
        s.push_str("    add     x13, x1, x9, lsl #2\n");
    }

    s.push_str("    mov     x3, #0\n");
    s.push_str(&format!(".Lmm_i_{lid}:\n"));
    s.push_str(&format!("    cmp     x3, #{b}\n"));
    s.push_str(&format!("    b.ge    .Lmm_i_end_{lid}\n"));

    s.push_str("    mov     x4, #0\n");
    s.push_str(&format!(".Lmm_j_{lid}:\n"));
    s.push_str(&format!("    cmp     x4, #{n}\n"));
    s.push_str(&format!("    b.ge    .Lmm_j_end_{lid}\n"));

    s.push_str("    fmov    s0, wzr\n");
    s.push_str("    mov     x5, #0\n");
    s.push_str(&format!(".Lmm_k_{lid}:\n"));
    s.push_str(&format!("    cmp     x5, #{k}\n"));
    s.push_str(&format!("    b.ge    .Lmm_k_end_{lid}\n"));

    s.push_str(&format!("    mov     x8, #{k}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x5\n");
    s.push_str("    ldr     s1, [x11, x6, lsl #2]\n");

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x7, x5, x8\n");
    s.push_str("    add     x7, x7, x4\n");
    s.push_str("    ldr     s2, [x13, x7, lsl #2]\n");

    s.push_str("    fmadd   s0, s1, s2, s0\n");

    s.push_str("    add     x5, x5, #1\n");
    s.push_str(&format!("    b       .Lmm_k_{lid}\n"));
    s.push_str(&format!(".Lmm_k_end_{lid}:\n"));

    s.push_str(&format!("    mov     x8, #{n}\n"));
    s.push_str("    mul     x6, x3, x8\n");
    s.push_str("    add     x6, x6, x4\n");
    s.push_str("    str     s0, [x12, x6, lsl #2]\n");

    s.push_str("    add     x4, x4, #1\n");
    s.push_str(&format!("    b       .Lmm_j_{lid}\n"));
    s.push_str(&format!(".Lmm_j_end_{lid}:\n"));

    s.push_str("    add     x3, x3, #1\n");
    s.push_str(&format!("    b       .Lmm_i_{lid}\n"));
    s.push_str(&format!(".Lmm_i_end_{lid}:\n"));

    s
}

/// Materialise a `BufferLoc` into a GPR (e.g. x11, x12). pub(crate) so relu.rs uses it too.
pub(crate) fn materialise_ptr(reg: &str, loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg => format!("    mov     {}, x0\n", reg),
        BufferLoc::OutputReg => format!("    mov     {}, x2\n", reg),
        BufferLoc::StackOffset(off) => {
            assert!(
                off <= u32::MAX as usize,
                "stack offset > 4 GiB unsupported in M4b (got {} bytes)",
                off
            );
            if off == 0 {
                format!("    mov     {}, sp\n", reg)
            } else if off <= 4095 {
                format!("    add     {}, sp, #{}\n", reg, off)
            } else if off <= 16_773_120 && off.is_multiple_of(4096) {
                format!("    add     {}, sp, #{}, lsl #12\n", reg, off / 4096)
            } else {
                let lo = (off & 0xFFFF) as u16;
                let hi = ((off >> 16) & 0xFFFF) as u16;
                let mut s = String::new();
                s.push_str(&format!("    movz    w10, #0x{:04x}\n", lo));
                if hi != 0 {
                    s.push_str(&format!("    movk    w10, #0x{:04x}, lsl #16\n", hi));
                }
                s.push_str(&format!("    add     {}, sp, x10\n", reg));
                s
            }
        }
        BufferLoc::Alias(_) => unreachable!("alias must be resolved by caller"),
    }
}
