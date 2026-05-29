.globl nfl_forward_TinyMLP
.p2align 4, 0x90
nfl_forward_TinyMLP:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $24, %rsp
    # matmul: input [8,4] x weights [4,2] -> output [8,2] + fused
    movq    %rdi, %r14
    movq    %rdx, %r11
    movq    %rsi, %r15
    pushq   %r14
    pushq   %r15
    xorq    %rax, %rax
.Lmm_i_0_0:
    movl    $8, %r10d
    cmpq    %r10, %rax
    jge     .Lmm_i_end_0_0
    xorq    %rbp, %rbp
.Lmm_j_0_0:
    movl    $2, %r10d
    cmpq    %r10, %rbp
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
    movss   (%r14, %rsi, 4), %xmm1
    movl    $2, %r10d
    movq    %rdi, %rsi
    imulq   %r10, %rsi
    addq    %rbp, %rsi
    movss   (%r15, %rsi, 4), %xmm2
    mulss   %xmm2, %xmm1
    addss   %xmm1, %xmm0
    incq    %rdi
    jmp     .Lmm_k_0_0
.Lmm_k_end_0_0:
    movl    $2, %r10d
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
    # fused softmax_row: output [8,2] in-place
    movq    %r11, %rbx
    movq    %r11, %r12
    xorq    %r13, %r13
.Lfsmx_i_0_0:
    movl    $8, %r10d
    cmpq    %r10, %r13
    jge     .Lfsmx_i_end_0_0
    movl    $2, %r10d
    movq    %r13, %r15
    imulq   %r10, %r15
    movl    $0xFF800000, %r10d
    movd    %r10d, %xmm8
    xorq    %r14, %r14
.Lfsmx_max_0_0:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_max_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    maxss   %xmm0, %xmm8
    incq    %r14
    jmp     .Lfsmx_max_0_0
.Lfsmx_max_end_0_0:
    movss   %xmm8, (%rsp)
    movl    $0, 8(%rsp)
    xorq    %r14, %r14
.Lfsmx_exp_0_0:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_exp_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%rbx, %rax, 4), %xmm0
    subss   (%rsp), %xmm0
    # --- inline exp(x), x<=0 (M17) ---
    movss   .Lexp_log2e(%rip), %xmm1
    mulss   %xmm0, %xmm1
    cvtss2si %xmm1, %eax
    cvtsi2ss %eax, %xmm2
    movss   .Lexp_ln2hi(%rip), %xmm3
    mulss   %xmm2, %xmm3
    movss   %xmm0, %xmm5
    subss   %xmm3, %xmm5
    movss   .Lexp_ln2lo(%rip), %xmm3
    mulss   %xmm2, %xmm3
    subss   %xmm3, %xmm5
    movss   .Lexp_c7(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c6(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c5(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c4(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c3(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c2(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c1(%rip), %xmm4
    mulss   %xmm5, %xmm4
    addss   .Lexp_c0(%rip), %xmm4
    addl    $127, %eax
    movl    %eax, %ecx
    shll    $23, %ecx
    xorl    %edx, %edx
    testl   %eax, %eax
    cmovle  %edx, %ecx
    movd    %ecx, %xmm5
    mulss   %xmm5, %xmm4
    movss   %xmm4, %xmm0
    # --- end inline exp ---
    movq    %r15, %rax
    addq    %r14, %rax
    movss   %xmm0, (%r12, %rax, 4)
    movss   8(%rsp), %xmm1
    addss   %xmm0, %xmm1
    movss   %xmm1, 8(%rsp)
    incq    %r14
    jmp     .Lfsmx_exp_0_0
.Lfsmx_exp_end_0_0:
    xorq    %r14, %r14
.Lfsmx_norm_0_0:
    movl    $2, %r10d
    cmpq    %r10, %r14
    jge     .Lfsmx_norm_end_0_0
    movq    %r15, %rax
    addq    %r14, %rax
    movss   (%r12, %rax, 4), %xmm0
    divss   8(%rsp), %xmm0
    movss   %xmm0, (%r12, %rax, 4)
    incq    %r14
    jmp     .Lfsmx_norm_0_0
.Lfsmx_norm_end_0_0:
    incq    %r13
    jmp     .Lfsmx_i_0_0
.Lfsmx_i_end_0_0:
    addq    $24, %rsp
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
