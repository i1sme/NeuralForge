.globl _nfl_forward_LargeN
.p2align 2
_nfl_forward_LargeN:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    ; matmul: input [2,8] x weights [8,5120] -> output [2,5120] + fused
    mov     x11, x0
    mov     x12, x2
    mov     x13, x1
    movz    x10, #0x0002
    movz    x15, #0x1400
    movz    x16, #0x0008
    mov     x3, #0
.Lmm_i_0_0:
    cmp     x3, x10
    b.ge    .Lmm_i_end_0_0
    mov     x4, #0
.Lmm_j_0_0:
    cmp     x4, x15
    b.ge    .Lmm_j_end_0_0
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_0_0:
    cmp     x5, x16
    b.ge    .Lmm_k_end_0_0
    mov     x8, x16
    mul     x6, x3, x8
    add     x6, x6, x5
    ldr     s1, [x11, x6, lsl #2]
    mov     x8, x15
    mul     x7, x5, x8
    add     x7, x7, x4
    ldr     s2, [x13, x7, lsl #2]
    fmadd   s0, s1, s2, s0
    add     x5, x5, #1
    b       .Lmm_k_0_0
.Lmm_k_end_0_0:
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_0_0
.Lmm_j_end_0_0:
    add     x3, x3, #1
    b       .Lmm_i_0_0
.Lmm_i_end_0_0:
    ; fused softmax_row: output [2,5120] in-place
    mov     x22, x12
    mov     x23, x12
    mov     x19, #0
.Lfsmx_i_0_0:
    movz    x10, #0x0002
    cmp     x19, x10
    b.ge    .Lfsmx_i_end_0_0
    movz    x8, #0x1400
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_0_0:
    movz    x10, #0x1400
    cmp     x21, x10
    b.ge    .Lfsmx_max_end_0_0
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_0_0
.Lfsmx_max_end_0_0:
    fmov    s9, wzr
    mov     x21, #0
.Lfsmx_exp_0_0:
    movz    x10, #0x1400
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_0_0
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
    b       .Lfsmx_exp_0_0
.Lfsmx_exp_end_0_0:
    mov     x21, #0
.Lfsmx_norm_0_0:
    movz    x10, #0x1400
    cmp     x21, x10
    b.ge    .Lfsmx_norm_end_0_0
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_0_0
.Lfsmx_norm_end_0_0:
    add     x19, x19, #1
    b       .Lfsmx_i_0_0
.Lfsmx_i_end_0_0:
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
