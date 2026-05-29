.globl nfl_forward_Classifier
.p2align 4, 0x90
nfl_forward_Classifier:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $98328, %rsp
    # matmul: input [32,784] x weights [784,512] -> output [32,512] + fused
    movq    %rdi, %r14
    leaq    16(%rsp), %r11
    movq    %rsi, %r15
    xorps   %xmm4, %xmm4
    movq    %rsi, %xmm6
    pushq   %r14
    pushq   %r15
    xorq    %rax, %rax
.Lmm_i_0_0:
    movl    $32, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_0
    xorq    %rbp, %rbp
.Lmm_j_0_0:
    movl    $512, %r10d
    cmpq    %r10, %rbp
    jge     .Lmm_j_end_0_0
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_0:
    movl    $784, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_0
    movl    $784, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rdi, %rsi
    movss   (%r14, %rsi, 4), %xmm1
    movl    $512, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   (%r15, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_0
.Lmm_k_end_0_0:
    maxss   %xmm4, %xmm0
    movl    $512, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   %xmm0, (%r11, %rsi, 4)
    incq    %rbp
    jmp     .Lmm_j_0_0
.Lmm_j_end_0_0:
    incq    %rax
    jmp     .Lmm_i_0_0
.Lmm_i_end_0_0:
    popq    %r15
    popq    %r14
    movq    %xmm6, %rsi
    # matmul: input [32,512] x weights [512,256] -> output [32,256] + fused
    leaq    16(%rsp), %r14
    leaq    65552(%rsp), %r11
    leaq    1605632(%rsi), %r15
    xorps   %xmm4, %xmm4
    movq    %rsi, %xmm6
    pushq   %r14
    pushq   %r15
    xorq    %rax, %rax
.Lmm_i_0_1:
    movl    $32, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_1
    xorq    %rbp, %rbp
.Lmm_j_0_1:
    movl    $256, %r10d
    cmpq    %r10, %rbp
    jge     .Lmm_j_end_0_1
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_1:
    movl    $512, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_1
    movl    $512, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rdi, %rsi
    movss   (%r14, %rsi, 4), %xmm1
    movl    $256, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   (%r15, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_1
.Lmm_k_end_0_1:
    maxss   %xmm4, %xmm0
    movl    $256, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   %xmm0, (%r11, %rsi, 4)
    incq    %rbp
    jmp     .Lmm_j_0_1
.Lmm_j_end_0_1:
    incq    %rax
    jmp     .Lmm_i_0_1
.Lmm_i_end_0_1:
    popq    %r15
    popq    %r14
    movq    %xmm6, %rsi
    # matmul: input [32,256] x weights [256,10] -> output [32,10] + fused
    leaq    65552(%rsp), %r14
    movq    %rdx, %r11
    leaq    2129920(%rsi), %r15
    pushq   %r14
    pushq   %r15
    xorq    %rax, %rax
.Lmm_i_0_2:
    movl    $32, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_2
    xorq    %rbp, %rbp
.Lmm_j_0_2:
    movl    $10, %r10d
    cmpq    %r10, %rbp
    jge     .Lmm_j_end_0_2
    xorq    %rdi, %rdi
    xorps   %xmm0, %xmm0
.Lmm_k_0_2:
    movl    $256, %r10d
    cmpq    %r10, %rdi
    jge     .Lmm_k_end_0_2
    movl    $256, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rdi, %rsi
    movss   (%r14, %rsi, 4), %xmm1
    movl    $10, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   (%r15, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_2
.Lmm_k_end_0_2:
    movl    $10, %r10d
    movq    %rax, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   %xmm0, (%r11, %rsi, 4)
    incq    %rbp
    jmp     .Lmm_j_0_2
.Lmm_j_end_0_2:
    incq    %rax
    jmp     .Lmm_i_0_2
.Lmm_i_end_0_2:
    popq    %r15
    popq    %r14
    # fused softmax_row: output [32,10] in-place
    movq    %r11, %rbx
    movq    %r11, %r12
    xorq    %r13, %r13
.Lfsmx_i_0_2:
    movl    $32, %r10d
    cmpq    %r10, %r13
    jge     .Lfsmx_i_end_0_2
    movl    $10, %r10d
    movq    %r13, %r15
    imulq   %r10, %r15
    movl    $0xFF800000, %r10d
    movd    %r10d, %xmm8
    xorq    %r14, %r14
.Lfsmx_max_0_2:
    movl    $10, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_max_end_0_2
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    maxss   %xmm0, %xmm8
    incq    %r14
    jmp     .Lfsmx_max_0_2
.Lfsmx_max_end_0_2:
    movss   %xmm8, (%rsp)
    movl    $0, 8(%rsp)
    xorq    %r14, %r14
.Lfsmx_exp_0_2:
    movl    $10, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_exp_end_0_2
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
    jmp     .Lfsmx_exp_0_2
.Lfsmx_exp_end_0_2:
    xorq    %r14, %r14
.Lfsmx_norm_0_2:
    movl    $10, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_norm_end_0_2
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%r12, %rax, 4), %xmm0
    divss   8(%rsp), %xmm0
    movss   %xmm0, (%r12, %rax, 4)
    incq    %r14
    jmp     .Lfsmx_norm_0_2
.Lfsmx_norm_end_0_2:
    incq    %r13
    jmp     .Lfsmx_i_0_2
.Lfsmx_i_end_0_2:
    addq    $98328, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    retq


.section .rodata
.align 4
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

.section .note.GNU-stack,"",@progbits
