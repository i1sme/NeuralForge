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
    ; --- inline exp(x), x<=0 (M17) ---
    adrp    x9, .Lexp_log2e@PAGE
    ldr     s1, [x9, .Lexp_log2e@PAGEOFF]
    fmul    s2, s0, s1
    fcvtns  w11, s2
    scvtf   s2, w11
    ldr     s1, [x9, .Lexp_ln2hi@PAGEOFF]
    fmsub   s3, s2, s1, s0
    ldr     s1, [x9, .Lexp_ln2lo@PAGEOFF]
    fmsub   s3, s2, s1, s3
    ldr     s4, [x9, .Lexp_c7@PAGEOFF]
    ldr     s1, [x9, .Lexp_c6@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c5@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c4@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c3@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c2@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c1@PAGEOFF]
    fmadd   s4, s4, s3, s1
    ldr     s1, [x9, .Lexp_c0@PAGEOFF]
    fmadd   s4, s4, s3, s1
    add     w11, w11, #127
    lsl     w12, w11, #23
    cmp     w11, #0
    csel    w12, wzr, w12, le
    fmov    s5, w12
    fmul    s0, s4, s5
    ; --- end inline exp ---
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


.section __TEXT,__const
.p2align 2
.Lexp_log2e: .long 0x3fb8aa3b
.Lexp_ln2hi: .long 0x3f318000
.Lexp_ln2lo: .long 0xb95e8083
.Lexp_c0: .long 0x3f800000
.Lexp_c1: .long 0x3f800000
.Lexp_c2: .long 0x3f000000
.Lexp_c3: .long 0x3e2aaaab
.Lexp_c4: .long 0x3d2aaaab
.Lexp_c5: .long 0x3c088889
.Lexp_c6: .long 0x3ab60b61
.Lexp_c7: .long 0x39500d01
