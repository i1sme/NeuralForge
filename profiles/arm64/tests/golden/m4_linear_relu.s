.globl _nfl_forward_M4Demo
.p2align 2
_nfl_forward_M4Demo:
    ; matmul: input [8,4] x weights [4,2] -> output [8,2] + fused
    fmov    s4, wzr
    mov     x11, x0
    mov     x12, x2
    mov     x13, x1
    movz    x10, #0x0008
    movz    x15, #0x0002
    movz    x16, #0x0004
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
    ret

