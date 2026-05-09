.globl _nfl_forward_SelfAttention
.p2align 2
_nfl_forward_SelfAttention:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #4, lsl #12
    ; matmul (leading_count=8): [16,16] x [16,16] -> [16,16], transpose_b=true
    mov     x9, x0
    mov     x10, x0
    mov     x11, sp
    mov     x15, #0
.Lmm4d_outer_0_0:
    movz    x8, #0x0008
    cmp     x15, x8
    b.ge    .Lmm4d_outer_end_0_0
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x12, x9, x6, lsl #2
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x13, x10, x6, lsl #2
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x14, x11, x6, lsl #2
    mov     x16, #0
.Lmm4d_i_0_0:
    movz    x8, #0x0010
    cmp     x16, x8
    b.ge    .Lmm4d_i_end_0_0
    mov     x17, #0
.Lmm4d_j_0_0:
    movz    x8, #0x0010
    cmp     x17, x8
    b.ge    .Lmm4d_j_end_0_0
    fmov    s0, wzr
    mov     x7, #0
.Lmm4d_k_0_0:
    movz    x8, #0x0010
    cmp     x7, x8
    b.ge    .Lmm4d_k_end_0_0
    movz    x8, #0x0010
    mul     x6, x16, x8
    add     x6, x6, x7
    ldr     s1, [x12, x6, lsl #2]
    movz    x8, #0x0010
    mul     x6, x17, x8
    add     x6, x6, x7
    ldr     s2, [x13, x6, lsl #2]
    fmadd   s0, s1, s2, s0
    add     x7, x7, #1
    b       .Lmm4d_k_0_0
.Lmm4d_k_end_0_0:
    movz    x8, #0x0010
    mul     x6, x16, x8
    add     x6, x6, x17
    str     s0, [x14, x6, lsl #2]
    add     x17, x17, #1
    b       .Lmm4d_j_0_0
.Lmm4d_j_end_0_0:
    add     x16, x16, #1
    b       .Lmm4d_i_0_0
.Lmm4d_i_end_0_0:
    add     x15, x15, #1
    b       .Lmm4d_outer_0_0
.Lmm4d_outer_end_0_0:
    ; mul_scalar: total_elements=2048, scalar_bits=0x3e800000
    movz    w9, #0x0000
    movk    w9, #0x3e80, lsl #16
    fmov    s4, w9
    mov     x11, sp
    mov     x12, sp
    mov     x3, #0
.Lms_0_0:
    movz    x10, #0x0800
    cmp     x3, x10
    b.ge    .Lms_end_0_0
    ldr     s0, [x11, x3, lsl #2]
    fmul    s0, s0, s4
    str     s0, [x12, x3, lsl #2]
    add     x3, x3, #1
    b       .Lms_0_0
.Lms_end_0_0:
    ; softmax (3-pass): input [128,16] -> output [128,16]
    mov     x22, sp
    add     x23, sp, #2, lsl #12
    stp     x0, x1, [sp, #-16]!
    stp     x2, xzr, [sp, #-16]!
    mov     x19, #0
.Lsm_i_0_0:
    movz    x10, #0x0080
    cmp     x19, x10
    b.ge    .Lsm_i_end_0_0
    movz    x8, #0x0010
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lsm_max_0_0:
    movz    x10, #0x0010
    cmp     x21, x10
    b.ge    .Lsm_max_end_0_0
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lsm_max_0_0
.Lsm_max_end_0_0:
    fmov    s9, wzr
    mov     x21, #0
.Lsm_exp_0_0:
    movz    x10, #0x0010
    cmp     x21, x10
    b.ge    .Lsm_exp_end_0_0
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    add     x6, x20, x21
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lsm_exp_0_0
.Lsm_exp_end_0_0:
    mov     x21, #0
.Lsm_norm_0_0:
    movz    x10, #0x0010
    cmp     x21, x10
    b.ge    .Lsm_norm_end_0_0
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lsm_norm_0_0
.Lsm_norm_end_0_0:
    add     x19, x19, #1
    b       .Lsm_i_0_0
.Lsm_i_end_0_0:
    ldp     x2, xzr, [sp], #16
    ldp     x0, x1, [sp], #16
    ; matmul (leading_count=8): [16,16] x [16,16] -> [16,16], transpose_b=false
    add     x9, sp, #2, lsl #12
    mov     x10, x0
    mov     x11, x2
    mov     x15, #0
.Lmm4d_outer_0_1:
    movz    x8, #0x0008
    cmp     x15, x8
    b.ge    .Lmm4d_outer_end_0_1
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x12, x9, x6, lsl #2
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x13, x10, x6, lsl #2
    movz    x8, #0x0100
    mul     x6, x15, x8
    add     x14, x11, x6, lsl #2
    mov     x16, #0
.Lmm4d_i_0_1:
    movz    x8, #0x0010
    cmp     x16, x8
    b.ge    .Lmm4d_i_end_0_1
    mov     x17, #0
.Lmm4d_j_0_1:
    movz    x8, #0x0010
    cmp     x17, x8
    b.ge    .Lmm4d_j_end_0_1
    fmov    s0, wzr
    mov     x7, #0
.Lmm4d_k_0_1:
    movz    x8, #0x0010
    cmp     x7, x8
    b.ge    .Lmm4d_k_end_0_1
    movz    x8, #0x0010
    mul     x6, x16, x8
    add     x6, x6, x7
    ldr     s1, [x12, x6, lsl #2]
    movz    x8, #0x0010
    mul     x6, x7, x8
    add     x6, x6, x17
    ldr     s2, [x13, x6, lsl #2]
    fmadd   s0, s1, s2, s0
    add     x7, x7, #1
    b       .Lmm4d_k_0_1
.Lmm4d_k_end_0_1:
    movz    x8, #0x0010
    mul     x6, x16, x8
    add     x6, x6, x17
    str     s0, [x14, x6, lsl #2]
    add     x17, x17, #1
    b       .Lmm4d_j_0_1
.Lmm4d_j_end_0_1:
    add     x16, x16, #1
    b       .Lmm4d_i_0_1
.Lmm4d_i_end_0_1:
    add     x15, x15, #1
    b       .Lmm4d_outer_0_1
.Lmm4d_outer_end_0_1:
    add     sp, sp, #4, lsl #12
    ldp     x29, x30, [sp], #16
    ldp     d8, d9, [sp], #16
    ldr     x23, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

