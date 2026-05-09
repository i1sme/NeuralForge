.globl nfl_forward_MixedArgs
.p2align 4, 0x90
nfl_forward_MixedArgs:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $280, %rsp
    # matmul: input [4,8] x weights [8,16] -> output [4,16] + bias + fused
    movq    %rdi, %r8
    leaq    16(%rsp), %r11
    movq    %rsi, %r9
    xorps   %xmm4, %xmm4
    movq    %rdx, %xmm7
    leaq    512(%rsi), %rdx
    movq    %rsi, %xmm6
    xorq    %rax, %rax
.Lmm_i_0_0:
    movl    $4, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_0
    xorq    %rcx, %rcx
.Lmm_j_0_0:
    movl    $16, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm_j_end_0_0
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_0:
    movl    $8, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_0
    movl    $8, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rdi, %rsi
    movss   (%r8, %rsi, 4), %xmm1
    movl    $16, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rcx, %rsi
    movss   (%r9, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_0
.Lmm_k_end_0_0:
    movss   (%rdx, %rcx, 4), %xmm5
    addss   %xmm5, %xmm0
    maxss   %xmm4, %xmm0
    movl    $16, %r10d
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
    movq    %xmm7, %rdx
    # matmul: input [4,16] x weights [16,2] -> output [4,2] + fused
    leaq    16(%rsp), %r8
    movq    %rdx, %r11
    leaq    576(%rsi), %r9
    xorq    %rax, %rax
.Lmm_i_0_1:
    movl    $4, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_1
    xorq    %rcx, %rcx
.Lmm_j_0_1:
    movl    $2, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm_j_end_0_1
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_1:
    movl    $16, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_1
    movl    $16, %r10d
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
    jmp     .Lmm_k_0_1
.Lmm_k_end_0_1:
    movl    $2, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rcx, %rsi
    movss   %xmm0, (%r11, %rsi, 4)
    incq    %rcx
    jmp     .Lmm_j_0_1
.Lmm_j_end_0_1:
    incq    %rax
    jmp     .Lmm_i_0_1
.Lmm_i_end_0_1:
    # fused softmax_row: output [4,2] in-place
    movq    %r11, %rbx
    movq    %r11, %r12
    xorq    %r13, %r13
.Lfsmx_i_0_1:
    movl    $4, %r10d
    cmpq    %r10, %r13
    jge     .Lfsmx_i_end_0_1
    movl    $2, %r10d
    movq    %r13, %r15
    imulq   %r10, %r15
    movl    $0xFF800000, %r10d
    movd    %r10d, %xmm8
    xorq    %r14, %r14
.Lfsmx_max_0_1:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_max_end_0_1
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    maxss   %xmm0, %xmm8
    incq    %r14
    jmp     .Lfsmx_max_0_1
.Lfsmx_max_end_0_1:
    movss   %xmm8, (%rsp)
    movl    $0, 8(%rsp)
    xorq    %r14, %r14
.Lfsmx_exp_0_1:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_exp_end_0_1
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    subss   (%rsp), %xmm0
    call    expf@PLT
    movq    %r15, %rax
    addq    %r14, %rax
    movss   %xmm0, (%r12, %rax, 4)
    movss   8(%rsp), %xmm1
    addss   %xmm0, %xmm1
    movss   %xmm1, 8(%rsp)
    incq    %r14
    jmp     .Lfsmx_exp_0_1
.Lfsmx_exp_end_0_1:
    xorq    %r14, %r14
.Lfsmx_norm_0_1:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_norm_end_0_1
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%r12, %rax, 4), %xmm0
    divss   8(%rsp), %xmm0
    movss   %xmm0, (%r12, %rax, 4)
    incq    %r14
    jmp     .Lfsmx_norm_0_1
.Lfsmx_norm_end_0_1:
    incq    %r13
    jmp     .Lfsmx_i_0_1
.Lfsmx_i_end_0_1:
    addq    $280, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    retq


.section .note.GNU-stack,"",@progbits
