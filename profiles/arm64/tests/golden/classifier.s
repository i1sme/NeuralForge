.globl _nfl_forward_Classifier
.p2align 2
_nfl_forward_Classifier:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #24, lsl #12
    ; matmul: input [32,784] x weights [784,512] -> output [32,512] + fused
    fmov    s4, wzr
    mov     x11, x0
    mov     x12, sp
    mov     x13, x1
    movz    x10, #0x0020
    movz    x15, #0x0200
    movz    x16, #0x0310
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
    fmax    s0, s0, s4
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
    ; matmul: input [32,512] x weights [512,256] -> output [32,256] + fused
    fmov    s4, wzr
    mov     x11, sp
    add     x12, sp, #16, lsl #12
    movz    x9, #0x2000
    movk    x9, #0x0006, lsl #16
    add     x13, x1, x9, lsl #2
    movz    x10, #0x0020
    movz    x15, #0x0100
    movz    x16, #0x0200
    mov     x3, #0
.Lmm_i_0_1:
    cmp     x3, x10
    b.ge    .Lmm_i_end_0_1
    mov     x4, #0
.Lmm_j_0_1:
    cmp     x4, x15
    b.ge    .Lmm_j_end_0_1
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_0_1:
    cmp     x5, x16
    b.ge    .Lmm_k_end_0_1
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
    b       .Lmm_k_0_1
.Lmm_k_end_0_1:
    fmax    s0, s0, s4
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_0_1
.Lmm_j_end_0_1:
    add     x3, x3, #1
    b       .Lmm_i_0_1
.Lmm_i_end_0_1:
    ; matmul: input [32,256] x weights [256,10] -> output [32,10] + fused
    add     x11, sp, #16, lsl #12
    mov     x12, x2
    movz    x9, #0x2000
    movk    x9, #0x0008, lsl #16
    add     x13, x1, x9, lsl #2
    movz    x10, #0x0020
    movz    x15, #0x000a
    movz    x16, #0x0100
    mov     x3, #0
.Lmm_i_0_2:
    cmp     x3, x10
    b.ge    .Lmm_i_end_0_2
    mov     x4, #0
.Lmm_j_0_2:
    cmp     x4, x15
    b.ge    .Lmm_j_end_0_2
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_0_2:
    cmp     x5, x16
    b.ge    .Lmm_k_end_0_2
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
    b       .Lmm_k_0_2
.Lmm_k_end_0_2:
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_0_2
.Lmm_j_end_0_2:
    add     x3, x3, #1
    b       .Lmm_i_0_2
.Lmm_i_end_0_2:
    ; fused softmax_row: output [32,10] in-place
    mov     x22, x12
    mov     x23, x12
    mov     x19, #0
.Lfsmx_i_0_2:
    movz    x10, #0x0020
    cmp     x19, x10
    b.ge    .Lfsmx_i_end_0_2
    movz    x8, #0x000a
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_0_2:
    movz    x10, #0x000a
    cmp     x21, x10
    b.ge    .Lfsmx_max_end_0_2
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_0_2
.Lfsmx_max_end_0_2:
    fmov    s9, wzr
    mov     x21, #0
.Lfsmx_exp_0_2:
    movz    x10, #0x000a
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_0_2
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    add     x6, x20, x21
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lfsmx_exp_0_2
.Lfsmx_exp_end_0_2:
    mov     x21, #0
.Lfsmx_norm_0_2:
    movz    x10, #0x000a
    cmp     x21, x10
    b.ge    .Lfsmx_norm_end_0_2
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_0_2
.Lfsmx_norm_end_0_2:
    add     x19, x19, #1
    b       .Lfsmx_i_0_2
.Lfsmx_i_end_0_2:
    add     sp, sp, #24, lsl #12
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
