.globl _nfl_forward_SingleLine
.p2align 2
_nfl_forward_SingleLine:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #128
    ; matmul: input [4,10] x weights [10,8] -> output [4,8] + fused
    fmov    s4, wzr
    mov     x11, x0
    mov     x12, sp
    mov     x13, x1
    movz    x10, #0x0004
    movz    x15, #0x0008
    movz    x16, #0x000a
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
    ; matmul: input [4,8] x weights [8,2] -> output [4,2] + fused
    mov     x11, sp
    mov     x12, x2
    movz    x9, #0x0050
    add     x13, x1, x9, lsl #2
    movz    x10, #0x0004
    movz    x15, #0x0002
    movz    x16, #0x0008
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
    ; fused softmax_row: output [4,2] in-place
    mov     x22, x12
    mov     x23, x12
    mov     x19, #0
.Lfsmx_i_0_1:
    movz    x10, #0x0004
    cmp     x19, x10
    b.ge    .Lfsmx_i_end_0_1
    movz    x8, #0x0002
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_0_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_max_end_0_1
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_0_1
.Lfsmx_max_end_0_1:
    fmov    s9, wzr
    mov     x21, #0
.Lfsmx_exp_0_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_0_1
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    add     x6, x20, x21
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lfsmx_exp_0_1
.Lfsmx_exp_end_0_1:
    mov     x21, #0
.Lfsmx_norm_0_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_norm_end_0_1
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_0_1
.Lfsmx_norm_end_0_1:
    add     x19, x19, #1
    b       .Lfsmx_i_0_1
.Lfsmx_i_end_0_1:
    add     sp, sp, #128
    ldp     x29, x30, [sp], #16
    ldp     d8, d9, [sp], #16
    ldr     x23, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

.globl _nfl_forward_PerStepWrap
.p2align 2
_nfl_forward_PerStepWrap:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #128
    ; matmul: input [4,10] x weights [10,8] -> output [4,8] + fused
    fmov    s4, wzr
    mov     x11, x0
    mov     x12, sp
    mov     x13, x1
    movz    x10, #0x0004
    movz    x15, #0x0008
    movz    x16, #0x000a
    mov     x3, #0
.Lmm_i_1_0:
    cmp     x3, x10
    b.ge    .Lmm_i_end_1_0
    mov     x4, #0
.Lmm_j_1_0:
    cmp     x4, x15
    b.ge    .Lmm_j_end_1_0
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_1_0:
    cmp     x5, x16
    b.ge    .Lmm_k_end_1_0
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
    b       .Lmm_k_1_0
.Lmm_k_end_1_0:
    fmax    s0, s0, s4
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_1_0
.Lmm_j_end_1_0:
    add     x3, x3, #1
    b       .Lmm_i_1_0
.Lmm_i_end_1_0:
    ; matmul: input [4,8] x weights [8,2] -> output [4,2] + fused
    mov     x11, sp
    mov     x12, x2
    movz    x9, #0x0050
    add     x13, x1, x9, lsl #2
    movz    x10, #0x0004
    movz    x15, #0x0002
    movz    x16, #0x0008
    mov     x3, #0
.Lmm_i_1_1:
    cmp     x3, x10
    b.ge    .Lmm_i_end_1_1
    mov     x4, #0
.Lmm_j_1_1:
    cmp     x4, x15
    b.ge    .Lmm_j_end_1_1
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_1_1:
    cmp     x5, x16
    b.ge    .Lmm_k_end_1_1
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
    b       .Lmm_k_1_1
.Lmm_k_end_1_1:
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_1_1
.Lmm_j_end_1_1:
    add     x3, x3, #1
    b       .Lmm_i_1_1
.Lmm_i_end_1_1:
    ; fused softmax_row: output [4,2] in-place
    mov     x22, x12
    mov     x23, x12
    mov     x19, #0
.Lfsmx_i_1_1:
    movz    x10, #0x0004
    cmp     x19, x10
    b.ge    .Lfsmx_i_end_1_1
    movz    x8, #0x0002
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_1_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_max_end_1_1
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_1_1
.Lfsmx_max_end_1_1:
    fmov    s9, wzr
    mov     x21, #0
.Lfsmx_exp_1_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_1_1
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    add     x6, x20, x21
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lfsmx_exp_1_1
.Lfsmx_exp_end_1_1:
    mov     x21, #0
.Lfsmx_norm_1_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_norm_end_1_1
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_1_1
.Lfsmx_norm_end_1_1:
    add     x19, x19, #1
    b       .Lfsmx_i_1_1
.Lfsmx_i_end_1_1:
    add     sp, sp, #128
    ldp     x29, x30, [sp], #16
    ldp     d8, d9, [sp], #16
    ldr     x23, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

.globl _nfl_forward_MixedWrap
.p2align 2
_nfl_forward_MixedWrap:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    str     x23, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #128
    ; matmul: input [4,10] x weights [10,8] -> output [4,8] + fused
    fmov    s4, wzr
    mov     x11, x0
    mov     x12, sp
    mov     x13, x1
    movz    x10, #0x0004
    movz    x15, #0x0008
    movz    x16, #0x000a
    mov     x3, #0
.Lmm_i_2_0:
    cmp     x3, x10
    b.ge    .Lmm_i_end_2_0
    mov     x4, #0
.Lmm_j_2_0:
    cmp     x4, x15
    b.ge    .Lmm_j_end_2_0
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_2_0:
    cmp     x5, x16
    b.ge    .Lmm_k_end_2_0
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
    b       .Lmm_k_2_0
.Lmm_k_end_2_0:
    fmax    s0, s0, s4
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_2_0
.Lmm_j_end_2_0:
    add     x3, x3, #1
    b       .Lmm_i_2_0
.Lmm_i_end_2_0:
    ; matmul: input [4,8] x weights [8,2] -> output [4,2] + fused
    mov     x11, sp
    mov     x12, x2
    movz    x9, #0x0050
    add     x13, x1, x9, lsl #2
    movz    x10, #0x0004
    movz    x15, #0x0002
    movz    x16, #0x0008
    mov     x3, #0
.Lmm_i_2_1:
    cmp     x3, x10
    b.ge    .Lmm_i_end_2_1
    mov     x4, #0
.Lmm_j_2_1:
    cmp     x4, x15
    b.ge    .Lmm_j_end_2_1
    fmov    s0, wzr
    mov     x5, #0
.Lmm_k_2_1:
    cmp     x5, x16
    b.ge    .Lmm_k_end_2_1
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
    b       .Lmm_k_2_1
.Lmm_k_end_2_1:
    mov     x8, x15
    mul     x6, x3, x8
    add     x6, x6, x4
    str     s0, [x12, x6, lsl #2]
    add     x4, x4, #1
    b       .Lmm_j_2_1
.Lmm_j_end_2_1:
    add     x3, x3, #1
    b       .Lmm_i_2_1
.Lmm_i_end_2_1:
    ; fused softmax_row: output [4,2] in-place
    mov     x22, x12
    mov     x23, x12
    mov     x19, #0
.Lfsmx_i_2_1:
    movz    x10, #0x0004
    cmp     x19, x10
    b.ge    .Lfsmx_i_end_2_1
    movz    x8, #0x0002
    mul     x20, x19, x8
    movz    w0, #0x0000
    movk    w0, #0xFF80, lsl #16
    fmov    s8, w0
    mov     x21, #0
.Lfsmx_max_2_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_max_end_2_1
    add     x6, x20, x21
    ldr     s1, [x22, x6, lsl #2]
    fmax    s8, s8, s1
    add     x21, x21, #1
    b       .Lfsmx_max_2_1
.Lfsmx_max_end_2_1:
    fmov    s9, wzr
    mov     x21, #0
.Lfsmx_exp_2_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_exp_end_2_1
    add     x6, x20, x21
    ldr     s0, [x22, x6, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    add     x6, x20, x21
    str     s0, [x23, x6, lsl #2]
    fadd    s9, s9, s0
    add     x21, x21, #1
    b       .Lfsmx_exp_2_1
.Lfsmx_exp_end_2_1:
    mov     x21, #0
.Lfsmx_norm_2_1:
    movz    x10, #0x0002
    cmp     x21, x10
    b.ge    .Lfsmx_norm_end_2_1
    add     x6, x20, x21
    ldr     s0, [x23, x6, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x6, lsl #2]
    add     x21, x21, #1
    b       .Lfsmx_norm_2_1
.Lfsmx_norm_end_2_1:
    add     x19, x19, #1
    b       .Lfsmx_i_2_1
.Lfsmx_i_end_2_1:
    add     sp, sp, #128
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
