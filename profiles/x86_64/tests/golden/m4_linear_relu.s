.globl nfl_forward_M4Demo
.p2align 4, 0x90
nfl_forward_M4Demo:
    pushq   %rbp
    movq    %rsp, %rbp
    # matmul: input [8,4] x weights [4,2] -> output [8,2] + fused
    movq    %rdi, %r8
    movq    %rdx, %r11
    movq    %rsi, %r9
    xorps   %xmm4, %xmm4
    movq    %rsi, %xmm6
    xorq    %rax, %rax
.Lmm_i_0_0:
    movl    $8, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_0
    xorq    %rcx, %rcx
.Lmm_j_0_0:
    movl    $2, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm_j_end_0_0
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_0:
    movl    $4, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_0
    movl    $4, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rdi, %rsi
    movss   (%r8, %rsi, 4), %xmm1
    movl    $2, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rcx, %rsi
    movss   (%r9, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_0
.Lmm_k_end_0_0:
    maxss   %xmm4, %xmm0
    movl    $2, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rcx, %rsi
    movss   %xmm0, (%r11, %rsi, 4)
    incq    %rcx
    jmp     .Lmm_j_0_0
.Lmm_j_end_0_0:
    incq    %rax
    jmp     .Lmm_i_0_0
.Lmm_i_end_0_0:
    movq    %xmm6, %rsi
    popq    %rbp
    retq


.section .note.GNU-stack,"",@progbits
