.globl nfl_forward_SelfAttention
.p2align 4, 0x90
nfl_forward_SelfAttention:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $16408, %rsp
    # matmul (leading_count=8): [16,16] x [16,16] -> [16,16], transpose_b=true
    movq    %rdi, %xmm8
    movq    %rsi, %xmm6
    movq    %rdx, %xmm7
    movq    %rdi, %r8
    movq    %rdi, %r9
    leaq    16(%rsp), %r11
    movq    $0, %rcx
.Lmm4d_outer_0_0:
    movl    $8, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm4d_outer_end_0_0
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r8, %rax, 4), %rdi
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r9, %rax, 4), %rsi
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r11, %rax, 4), %rdx
    pushq   %rcx
    movq    $0, %rax
.Lmm4d_i_0_0:
    movl    $16, %r10d
    cmpq    %r10, %rax
    jge     .Lmm4d_i_end_0_0
    movq    $0, %rcx
.Lmm4d_j_0_0:
    movl    $16, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm4d_j_end_0_0
    xorps   %xmm0, %xmm0
    movq    $0, %r10
.Lmm4d_k_0_0:
    cmpq    $16, %r10
    jge     .Lmm4d_k_end_0_0
    pushq   %r11
    movq    %rax, %r11
    imulq   $16, %r11
    addq    %r10, %r11
    movss   (%rdi, %r11, 4), %xmm1
    movq    %rcx, %r11
    imulq   $16, %r11
    addq    %r10, %r11
    movss   (%rsi, %r11, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    popq    %r11
    addq    $1, %r10
    jmp     .Lmm4d_k_0_0
.Lmm4d_k_end_0_0:
    pushq   %r11
    movq    %rax, %r11
    imulq   $16, %r11
    addq    %rcx, %r11
    movss   %xmm0, (%rdx, %r11, 4)
    popq    %r11
    addq    $1, %rcx
    jmp     .Lmm4d_j_0_0
.Lmm4d_j_end_0_0:
    addq    $1, %rax
    jmp     .Lmm4d_i_0_0
.Lmm4d_i_end_0_0:
    popq    %rcx
    addq    $1, %rcx
    jmp     .Lmm4d_outer_0_0
.Lmm4d_outer_end_0_0:
    movq    %xmm8, %rdi
    movq    %xmm6, %rsi
    movq    %xmm7, %rdx
    # mul_scalar: total_elements=2048, scalar_bits=0x3e800000
    movl    $0x3e800000, %r10d
    movd    %r10d, %xmm4
    leaq    16(%rsp), %r8
    leaq    16(%rsp), %r11
    movq    $0, %rcx
.Lms_0_0:
    movl    $2048, %r10d
    cmpq    %r10, %rcx
    jge     .Lms_end_0_0
    movss   (%r8, %rcx, 4), %xmm0
    mulss   %xmm4, %xmm0
    movss   %xmm0, (%r11, %rcx, 4)
    addq    $1, %rcx
    jmp     .Lms_0_0
.Lms_end_0_0:
    # softmax (3-pass): input [128,16] -> output [128,16]
    leaq    16(%rsp), %rbx
    leaq    8208(%rsp), %r12
    pushq   %rdi
    pushq   %rsi
    pushq   %rdx
    pushq   %rax
    xorq    %r13, %r13
.Lsm_i_0_0:
    movl    $128, %r10d
    cmpq    %r10, %r13
    jge     .Lsm_i_end_0_0
    movl    $16, %r10d
    movq    %r13, %r15
    imulq   %r10, %r15
    movl    $0xFF800000, %r10d
    movd    %r10d, %xmm8
    xorq    %r14, %r14
.Lsm_max_0_0:
    movl    $16, %r10d
    cmpq    %r10, %r14
    jge     .Lsm_max_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    maxss   %xmm0, %xmm8
    incq    %r14
    jmp     .Lsm_max_0_0
.Lsm_max_end_0_0:
    movss   %xmm8, 32(%rsp)
    movl    $0, 40(%rsp)
    xorq    %r14, %r14
.Lsm_exp_0_0:
    movl    $16, %r10d
    cmpq    %r10, %r14
    jge     .Lsm_exp_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    subss   32(%rsp), %xmm0
    call    expf@PLT
    movq    %r15, %rax
    addq    %r14, %rax
    movss   %xmm0, (%r12, %rax, 4)
    movss   40(%rsp), %xmm1
    addss   %xmm0, %xmm1
    movss   %xmm1, 40(%rsp)
    incq    %r14
    jmp     .Lsm_exp_0_0
.Lsm_exp_end_0_0:
    xorq    %r14, %r14
.Lsm_norm_0_0:
    movl    $16, %r10d
    cmpq    %r10, %r14
    jge     .Lsm_norm_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%r12, %rax, 4), %xmm0
    divss   40(%rsp), %xmm0
    movss   %xmm0, (%r12, %rax, 4)
    incq    %r14
    jmp     .Lsm_norm_0_0
.Lsm_norm_end_0_0:
    incq    %r13
    jmp     .Lsm_i_0_0
.Lsm_i_end_0_0:
    popq    %rax
    popq    %rdx
    popq    %rsi
    popq    %rdi
    # matmul (leading_count=8): [16,16] x [16,16] -> [16,16], transpose_b=false
    movq    %rdi, %xmm8
    movq    %rsi, %xmm6
    movq    %rdx, %xmm7
    leaq    8208(%rsp), %r8
    movq    %rdi, %r9
    movq    %rdx, %r11
    movq    $0, %rcx
.Lmm4d_outer_0_1:
    movl    $8, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm4d_outer_end_0_1
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r8, %rax, 4), %rdi
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r9, %rax, 4), %rsi
    movl    $256, %r10d
    movq    %rcx, %rax
    imulq   %r10, %rax
    leaq    (%r11, %rax, 4), %rdx
    pushq   %rcx
    movq    $0, %rax
.Lmm4d_i_0_1:
    movl    $16, %r10d
    cmpq    %r10, %rax
    jge     .Lmm4d_i_end_0_1
    movq    $0, %rcx
.Lmm4d_j_0_1:
    movl    $16, %r10d
    cmpq    %r10, %rcx
    jge     .Lmm4d_j_end_0_1
    xorps   %xmm0, %xmm0
    movq    $0, %r10
.Lmm4d_k_0_1:
    cmpq    $16, %r10
    jge     .Lmm4d_k_end_0_1
    pushq   %r11
    movq    %rax, %r11
    imulq   $16, %r11
    addq    %r10, %r11
    movss   (%rdi, %r11, 4), %xmm1
    movq    %r10, %r11
    imulq   $16, %r11
    addq    %rcx, %r11
    movss   (%rsi, %r11, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    popq    %r11
    addq    $1, %r10
    jmp     .Lmm4d_k_0_1
.Lmm4d_k_end_0_1:
    pushq   %r11
    movq    %rax, %r11
    imulq   $16, %r11
    addq    %rcx, %r11
    movss   %xmm0, (%rdx, %r11, 4)
    popq    %r11
    addq    $1, %rcx
    jmp     .Lmm4d_j_0_1
.Lmm4d_j_end_0_1:
    addq    $1, %rax
    jmp     .Lmm4d_i_0_1
.Lmm4d_i_end_0_1:
    popq    %rcx
    addq    $1, %rcx
    jmp     .Lmm4d_outer_0_1
.Lmm4d_outer_end_0_1:
    movq    %xmm8, %rdi
    movq    %xmm6, %rsi
    movq    %xmm7, %rdx
    addq    $16408, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    retq


.section .note.GNU-stack,"",@progbits
