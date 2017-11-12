
bin/clang++-3.8_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

00000000004006c8 <_init>:
  4006c8:	48 83 ec 08          	sub    $0x8,%rsp
  4006cc:	48 8b 05 25 19 20 00 	mov    0x201925(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4006d3:	48 85 c0             	test   %rax,%rax
  4006d6:	74 05                	je     4006dd <_init+0x15>
  4006d8:	e8 b3 00 00 00       	callq  400790 <sqrt@plt+0x10>
  4006dd:	48 83 c4 08          	add    $0x8,%rsp
  4006e1:	c3                   	retq   

Disassembly of section .plt:

00000000004006f0 <printf@plt-0x10>:
  4006f0:	ff 35 12 19 20 00    	pushq  0x201912(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006f6:	ff 25 14 19 20 00    	jmpq   *0x201914(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400700 <printf@plt>:
  400700:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400706:	68 00 00 00 00       	pushq  $0x0
  40070b:	e9 e0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400710 <_ZNSt8ios_base4InitC1Ev@plt>:
  400710:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400716:	68 01 00 00 00       	pushq  $0x1
  40071b:	e9 d0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400720 <__libc_start_main@plt>:
  400720:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400726:	68 02 00 00 00       	pushq  $0x2
  40072b:	e9 c0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400730 <__cxa_atexit@plt>:
  400730:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400736:	68 03 00 00 00       	pushq  $0x3
  40073b:	e9 b0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400740 <_ZNSt8ios_base4InitD1Ev@plt>:
  400740:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400746:	68 04 00 00 00       	pushq  $0x4
  40074b:	e9 a0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400750 <srand@plt>:
  400750:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400756:	68 05 00 00 00       	pushq  $0x5
  40075b:	e9 90 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400760:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400766:	68 06 00 00 00       	pushq  $0x6
  40076b:	e9 80 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400770 <rand@plt>:
  400770:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400776:	68 07 00 00 00       	pushq  $0x7
  40077b:	e9 70 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400780 <sqrt@plt>:
  400780:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400786:	68 08 00 00 00       	pushq  $0x8
  40078b:	e9 60 ff ff ff       	jmpq   4006f0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400790 <.plt.got>:
  400790:	ff 25 62 18 20 00    	jmpq   *0x201862(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400796:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007a0 <_GLOBAL__sub_I_aos.cc>:
  4007a0:	50                   	push   %rax
  4007a1:	bf 90 20 60 00       	mov    $0x602090,%edi
  4007a6:	e8 65 ff ff ff       	callq  400710 <_ZNSt8ios_base4InitC1Ev@plt>
  4007ab:	bf 40 07 40 00       	mov    $0x400740,%edi
  4007b0:	be 90 20 60 00       	mov    $0x602090,%esi
  4007b5:	ba 68 20 60 00       	mov    $0x602068,%edx
  4007ba:	58                   	pop    %rax
  4007bb:	e9 70 ff ff ff       	jmpq   400730 <__cxa_atexit@plt>

00000000004007c0 <_start>:
  4007c0:	31 ed                	xor    %ebp,%ebp
  4007c2:	49 89 d1             	mov    %rdx,%r9
  4007c5:	5e                   	pop    %rsi
  4007c6:	48 89 e2             	mov    %rsp,%rdx
  4007c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4007cd:	50                   	push   %rax
  4007ce:	54                   	push   %rsp
  4007cf:	49 c7 c0 50 10 40 00 	mov    $0x401050,%r8
  4007d6:	48 c7 c1 e0 0f 40 00 	mov    $0x400fe0,%rcx
  4007dd:	48 c7 c7 a0 0d 40 00 	mov    $0x400da0,%rdi
  4007e4:	e8 37 ff ff ff       	callq  400720 <__libc_start_main@plt>
  4007e9:	f4                   	hlt    
  4007ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007f0 <deregister_tm_clones>:
  4007f0:	b8 7f 20 60 00       	mov    $0x60207f,%eax
  4007f5:	55                   	push   %rbp
  4007f6:	48 2d 78 20 60 00    	sub    $0x602078,%rax
  4007fc:	48 83 f8 0e          	cmp    $0xe,%rax
  400800:	48 89 e5             	mov    %rsp,%rbp
  400803:	76 1b                	jbe    400820 <deregister_tm_clones+0x30>
  400805:	b8 00 00 00 00       	mov    $0x0,%eax
  40080a:	48 85 c0             	test   %rax,%rax
  40080d:	74 11                	je     400820 <deregister_tm_clones+0x30>
  40080f:	5d                   	pop    %rbp
  400810:	bf 78 20 60 00       	mov    $0x602078,%edi
  400815:	ff e0                	jmpq   *%rax
  400817:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40081e:	00 00 
  400820:	5d                   	pop    %rbp
  400821:	c3                   	retq   
  400822:	0f 1f 40 00          	nopl   0x0(%rax)
  400826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40082d:	00 00 00 

0000000000400830 <register_tm_clones>:
  400830:	be 78 20 60 00       	mov    $0x602078,%esi
  400835:	55                   	push   %rbp
  400836:	48 81 ee 78 20 60 00 	sub    $0x602078,%rsi
  40083d:	48 c1 fe 03          	sar    $0x3,%rsi
  400841:	48 89 e5             	mov    %rsp,%rbp
  400844:	48 89 f0             	mov    %rsi,%rax
  400847:	48 c1 e8 3f          	shr    $0x3f,%rax
  40084b:	48 01 c6             	add    %rax,%rsi
  40084e:	48 d1 fe             	sar    %rsi
  400851:	74 15                	je     400868 <register_tm_clones+0x38>
  400853:	b8 00 00 00 00       	mov    $0x0,%eax
  400858:	48 85 c0             	test   %rax,%rax
  40085b:	74 0b                	je     400868 <register_tm_clones+0x38>
  40085d:	5d                   	pop    %rbp
  40085e:	bf 78 20 60 00       	mov    $0x602078,%edi
  400863:	ff e0                	jmpq   *%rax
  400865:	0f 1f 00             	nopl   (%rax)
  400868:	5d                   	pop    %rbp
  400869:	c3                   	retq   
  40086a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400870 <__do_global_dtors_aux>:
  400870:	80 3d 09 18 20 00 00 	cmpb   $0x0,0x201809(%rip)        # 602080 <completed.7585>
  400877:	75 11                	jne    40088a <__do_global_dtors_aux+0x1a>
  400879:	55                   	push   %rbp
  40087a:	48 89 e5             	mov    %rsp,%rbp
  40087d:	e8 6e ff ff ff       	callq  4007f0 <deregister_tm_clones>
  400882:	5d                   	pop    %rbp
  400883:	c6 05 f6 17 20 00 01 	movb   $0x1,0x2017f6(%rip)        # 602080 <completed.7585>
  40088a:	f3 c3                	repz retq 
  40088c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400890 <frame_dummy>:
  400890:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400895:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400899:	75 05                	jne    4008a0 <frame_dummy+0x10>
  40089b:	eb 93                	jmp    400830 <register_tm_clones>
  40089d:	0f 1f 00             	nopl   (%rax)
  4008a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4008a5:	48 85 c0             	test   %rax,%rax
  4008a8:	74 f1                	je     40089b <frame_dummy+0xb>
  4008aa:	55                   	push   %rbp
  4008ab:	48 89 e5             	mov    %rsp,%rbp
  4008ae:	ff d0                	callq  *%rax
  4008b0:	5d                   	pop    %rbp
  4008b1:	e9 7a ff ff ff       	jmpq   400830 <register_tm_clones>
  4008b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008bd:	00 00 00 

00000000004008c0 <_Z11r_float2intf>:
  4008c0:	66 0f 7e c0          	movd   %xmm0,%eax
  4008c4:	c3                   	retq   
  4008c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4008cc:	00 00 00 00 

00000000004008d0 <_Z13instantiationv>:
  4008d0:	53                   	push   %rbx
  4008d1:	48 83 ec 30          	sub    $0x30,%rsp
  4008d5:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4008da:	e8 71 fe ff ff       	callq  400750 <srand@plt>
  4008df:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4008e4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008eb:	00 00 00 00 00 
  4008f0:	e8 7b fe ff ff       	callq  400770 <rand@plt>
  4008f5:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008f8:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008fc:	f2 0f 10 05 64 07 00 	movsd  0x764(%rip),%xmm0        # 401068 <_IO_stdin_used+0x8>
  400903:	00 
  400904:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400908:	f2 0f 10 05 60 07 00 	movsd  0x760(%rip),%xmm0        # 401070 <_IO_stdin_used+0x10>
  40090f:	00 
  400910:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400914:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400918:	f2 0f 10 05 58 07 00 	movsd  0x758(%rip),%xmm0        # 401078 <_IO_stdin_used+0x18>
  40091f:	00 
  400920:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400924:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  40092a:	e8 41 fe ff ff       	callq  400770 <rand@plt>
  40092f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400932:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400936:	f2 0f 5e 0d 2a 07 00 	divsd  0x72a(%rip),%xmm1        # 401068 <_IO_stdin_used+0x8>
  40093d:	00 
  40093e:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400942:	f2 0f 10 05 36 07 00 	movsd  0x736(%rip),%xmm0        # 401080 <_IO_stdin_used+0x20>
  400949:	00 
  40094a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40094e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400954:	e8 17 fe ff ff       	callq  400770 <rand@plt>
  400959:	0f 57 c0             	xorps  %xmm0,%xmm0
  40095c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400960:	f2 0f 5e 05 00 07 00 	divsd  0x700(%rip),%xmm0        # 401068 <_IO_stdin_used+0x8>
  400967:	00 
  400968:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40096c:	f2 0f 58 05 0c 07 00 	addsd  0x70c(%rip),%xmm0        # 401080 <_IO_stdin_used+0x20>
  400973:	00 
  400974:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40097a:	e8 f1 fd ff ff       	callq  400770 <rand@plt>
  40097f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400982:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400986:	f2 0f 5e 0d da 06 00 	divsd  0x6da(%rip),%xmm1        # 401068 <_IO_stdin_used+0x8>
  40098d:	00 
  40098e:	f2 0f 10 05 f2 06 00 	movsd  0x6f2(%rip),%xmm0        # 401088 <_IO_stdin_used+0x28>
  400995:	00 
  400996:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40099a:	f2 0f 5e 0d d6 06 00 	divsd  0x6d6(%rip),%xmm1        # 401078 <_IO_stdin_used+0x18>
  4009a1:	00 
  4009a2:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  4009a8:	e8 c3 fd ff ff       	callq  400770 <rand@plt>
  4009ad:	0f 57 c0             	xorps  %xmm0,%xmm0
  4009b0:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  4009b4:	f2 0f 5e 05 ac 06 00 	divsd  0x6ac(%rip),%xmm0        # 401068 <_IO_stdin_used+0x8>
  4009bb:	00 
  4009bc:	f2 0f 58 05 c4 06 00 	addsd  0x6c4(%rip),%xmm0        # 401088 <_IO_stdin_used+0x28>
  4009c3:	00 
  4009c4:	f2 0f 5e 05 ac 06 00 	divsd  0x6ac(%rip),%xmm0        # 401078 <_IO_stdin_used+0x18>
  4009cb:	00 
  4009cc:	48 8b 05 9d 16 20 00 	mov    0x20169d(%rip),%rax        # 602070 <Body_arena_head>
  4009d3:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  4009d9:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  4009dd:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  4009e3:	f2 0f 11 48 08       	movsd  %xmm1,0x8(%rax)
  4009e8:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  4009ee:	f2 0f 11 48 10       	movsd  %xmm1,0x10(%rax)
  4009f3:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  4009f9:	f2 0f 11 48 18       	movsd  %xmm1,0x18(%rax)
  4009fe:	f2 0f 11 40 20       	movsd  %xmm0,0x20(%rax)
  400a03:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
  400a07:	66 0f 11 40 28       	movupd %xmm0,0x28(%rax)
  400a0c:	48 83 c0 38          	add    $0x38,%rax
  400a10:	48 89 05 59 16 20 00 	mov    %rax,0x201659(%rip)        # 602070 <Body_arena_head>
  400a17:	ff cb                	dec    %ebx
  400a19:	0f 85 d1 fe ff ff    	jne    4008f0 <_Z13instantiationv+0x20>
  400a1f:	48 83 c4 30          	add    $0x30,%rsp
  400a23:	5b                   	pop    %rbx
  400a24:	c3                   	retq   
  400a25:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400a2c:	00 00 00 00 

0000000000400a30 <_Z14run_simulationv>:
  400a30:	55                   	push   %rbp
  400a31:	41 57                	push   %r15
  400a33:	41 56                	push   %r14
  400a35:	41 55                	push   %r13
  400a37:	41 54                	push   %r12
  400a39:	53                   	push   %rbx
  400a3a:	48 83 ec 28          	sub    $0x28,%rsp
  400a3e:	45 31 f6             	xor    %r14d,%r14d
  400a41:	f2 0f 10 25 47 06 00 	movsd  0x647(%rip),%xmm4        # 401090 <_IO_stdin_used+0x30>
  400a48:	00 
  400a49:	f2 0f 10 2d 47 06 00 	movsd  0x647(%rip),%xmm5        # 401098 <_IO_stdin_used+0x38>
  400a50:	00 
  400a51:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400a58:	0f 1f 84 00 00 00 00 
  400a5f:	00 
  400a60:	31 c0                	xor    %eax,%eax
  400a62:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400a69:	1f 84 00 00 00 00 00 
  400a70:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
  400a74:	66 0f 11 80 c8 20 60 	movupd %xmm0,0x6020c8(%rax)
  400a7b:	00 
  400a7c:	66 0f 11 80 00 21 60 	movupd %xmm0,0x602100(%rax)
  400a83:	00 
  400a84:	66 0f 11 80 38 21 60 	movupd %xmm0,0x602138(%rax)
  400a8b:	00 
  400a8c:	66 0f 11 80 70 21 60 	movupd %xmm0,0x602170(%rax)
  400a93:	00 
  400a94:	66 0f 11 80 a8 21 60 	movupd %xmm0,0x6021a8(%rax)
  400a9b:	00 
  400a9c:	66 0f 11 80 e0 21 60 	movupd %xmm0,0x6021e0(%rax)
  400aa3:	00 
  400aa4:	66 0f 11 80 18 22 60 	movupd %xmm0,0x602218(%rax)
  400aab:	00 
  400aac:	66 0f 11 80 50 22 60 	movupd %xmm0,0x602250(%rax)
  400ab3:	00 
  400ab4:	48 05 c0 01 00 00    	add    $0x1c0,%rax
  400aba:	48 3d 00 d6 06 00    	cmp    $0x6d600,%rax
  400ac0:	41 bf 00 00 00 00    	mov    $0x0,%r15d
  400ac6:	41 bc 00 00 00 00    	mov    $0x0,%r12d
  400acc:	75 a2                	jne    400a70 <_Z14run_simulationv+0x40>
  400ace:	66 90                	xchg   %ax,%ax
  400ad0:	4d 6b ec 38          	imul   $0x38,%r12,%r13
  400ad4:	48 c7 c3 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rbx
  400adb:	4c 89 fd             	mov    %r15,%rbp
  400ade:	66 90                	xchg   %ax,%ax
  400ae0:	48 85 ed             	test   %rbp,%rbp
  400ae3:	0f 84 a7 00 00 00    	je     400b90 <_Z14run_simulationv+0x160>
  400ae9:	66 41 0f 10 9d a8 20 	movupd 0x6020a8(%r13),%xmm3
  400af0:	60 00 
  400af2:	66 0f 10 83 a8 f6 66 	movupd 0x66f6a8(%rbx),%xmm0
  400af9:	00 
  400afa:	66 0f 5c d8          	subpd  %xmm0,%xmm3
  400afe:	66 0f 28 c3          	movapd %xmm3,%xmm0
  400b02:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400b06:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400b0a:	66 0f c6 c9 01       	shufpd $0x1,%xmm1,%xmm1
  400b0f:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400b13:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400b17:	0f 57 c0             	xorps  %xmm0,%xmm0
  400b1a:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400b1e:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400b22:	7b 25                	jnp    400b49 <_Z14run_simulationv+0x119>
  400b24:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400b28:	66 0f 29 5c 24 10    	movapd %xmm3,0x10(%rsp)
  400b2e:	e8 4d fc ff ff       	callq  400780 <sqrt@plt>
  400b33:	66 0f 28 5c 24 10    	movapd 0x10(%rsp),%xmm3
  400b39:	f2 0f 10 2d 57 05 00 	movsd  0x557(%rip),%xmm5        # 401098 <_IO_stdin_used+0x38>
  400b40:	00 
  400b41:	f2 0f 10 25 47 05 00 	movsd  0x547(%rip),%xmm4        # 401090 <_IO_stdin_used+0x30>
  400b48:	00 
  400b49:	f2 0f 10 8b a0 f6 66 	movsd  0x66f6a0(%rbx),%xmm1
  400b50:	00 
  400b51:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b55:	f2 41 0f 59 8d a0 20 	mulsd  0x6020a0(%r13),%xmm1
  400b5c:	60 00 
  400b5e:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400b62:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400b66:	f2 0f 58 d5          	addsd  %xmm5,%xmm2
  400b6a:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400b6e:	0f 16 c9             	movlhps %xmm1,%xmm1
  400b71:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  400b75:	0f 16 c0             	movlhps %xmm0,%xmm0
  400b78:	66 0f 5e d8          	divpd  %xmm0,%xmm3
  400b7c:	66 0f 10 83 c8 f6 66 	movupd 0x66f6c8(%rbx),%xmm0
  400b83:	00 
  400b84:	66 0f 58 c3          	addpd  %xmm3,%xmm0
  400b88:	66 0f 11 83 c8 f6 66 	movupd %xmm0,0x66f6c8(%rbx)
  400b8f:	00 
  400b90:	48 83 c5 c8          	add    $0xffffffffffffffc8,%rbp
  400b94:	48 83 c3 38          	add    $0x38,%rbx
  400b98:	0f 85 42 ff ff ff    	jne    400ae0 <_Z14run_simulationv+0xb0>
  400b9e:	49 ff c4             	inc    %r12
  400ba1:	49 83 c7 38          	add    $0x38,%r15
  400ba5:	48 c7 c0 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rax
  400bac:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400bb3:	0f 85 17 ff ff ff    	jne    400ad0 <_Z14run_simulationv+0xa0>
  400bb9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400bc0:	66 0f 10 88 c8 f6 66 	movupd 0x66f6c8(%rax),%xmm1
  400bc7:	00 
  400bc8:	66 0f 28 05 f0 04 00 	movapd 0x4f0(%rip),%xmm0        # 4010c0 <_IO_stdin_used+0x60>
  400bcf:	00 
  400bd0:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400bd4:	66 0f 59 cb          	mulpd  %xmm3,%xmm1
  400bd8:	f2 0f 10 80 a0 f6 66 	movsd  0x66f6a0(%rax),%xmm0
  400bdf:	00 
  400be0:	0f 16 c0             	movlhps %xmm0,%xmm0
  400be3:	66 0f 5e c8          	divpd  %xmm0,%xmm1
  400be7:	66 0f 10 80 b8 f6 66 	movupd 0x66f6b8(%rax),%xmm0
  400bee:	00 
  400bef:	66 0f 58 c1          	addpd  %xmm1,%xmm0
  400bf3:	66 0f 11 80 b8 f6 66 	movupd %xmm0,0x66f6b8(%rax)
  400bfa:	00 
  400bfb:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400bff:	66 0f 59 d3          	mulpd  %xmm3,%xmm2
  400c03:	66 0f 10 88 a8 f6 66 	movupd 0x66f6a8(%rax),%xmm1
  400c0a:	00 
  400c0b:	66 0f 58 ca          	addpd  %xmm2,%xmm1
  400c0f:	66 0f 11 88 a8 f6 66 	movupd %xmm1,0x66f6a8(%rax)
  400c16:	00 
  400c17:	f2 0f 10 15 61 04 00 	movsd  0x461(%rip),%xmm2        # 401080 <_IO_stdin_used+0x20>
  400c1e:	00 
  400c1f:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  400c23:	77 0a                	ja     400c2f <_Z14run_simulationv+0x1ff>
  400c25:	66 0f 2e 0d 73 04 00 	ucomisd 0x473(%rip),%xmm1        # 4010a0 <_IO_stdin_used+0x40>
  400c2c:	00 
  400c2d:	76 14                	jbe    400c43 <_Z14run_simulationv+0x213>
  400c2f:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400c33:	66 0f 57 15 95 04 00 	xorpd  0x495(%rip),%xmm2        # 4010d0 <_IO_stdin_used+0x70>
  400c3a:	00 
  400c3b:	66 0f 13 90 b8 f6 66 	movlpd %xmm2,0x66f6b8(%rax)
  400c42:	00 
  400c43:	66 0f c6 c9 01       	shufpd $0x1,%xmm1,%xmm1
  400c48:	f2 0f 10 15 30 04 00 	movsd  0x430(%rip),%xmm2        # 401080 <_IO_stdin_used+0x20>
  400c4f:	00 
  400c50:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  400c54:	77 0a                	ja     400c60 <_Z14run_simulationv+0x230>
  400c56:	66 0f 2e 0d 42 04 00 	ucomisd 0x442(%rip),%xmm1        # 4010a0 <_IO_stdin_used+0x40>
  400c5d:	00 
  400c5e:	76 15                	jbe    400c75 <_Z14run_simulationv+0x245>
  400c60:	66 0f c6 c0 01       	shufpd $0x1,%xmm0,%xmm0
  400c65:	66 0f 57 05 63 04 00 	xorpd  0x463(%rip),%xmm0        # 4010d0 <_IO_stdin_used+0x70>
  400c6c:	00 
  400c6d:	66 0f 13 80 c0 f6 66 	movlpd %xmm0,0x66f6c0(%rax)
  400c74:	00 
  400c75:	48 83 c0 38          	add    $0x38,%rax
  400c79:	0f 85 41 ff ff ff    	jne    400bc0 <_Z14run_simulationv+0x190>
  400c7f:	41 ff c6             	inc    %r14d
  400c82:	41 83 fe 05          	cmp    $0x5,%r14d
  400c86:	0f 85 d4 fd ff ff    	jne    400a60 <_Z14run_simulationv+0x30>
  400c8c:	48 83 c4 28          	add    $0x28,%rsp
  400c90:	5b                   	pop    %rbx
  400c91:	41 5c                	pop    %r12
  400c93:	41 5d                	pop    %r13
  400c95:	41 5e                	pop    %r14
  400c97:	41 5f                	pop    %r15
  400c99:	5d                   	pop    %rbp
  400c9a:	c3                   	retq   
  400c9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ca0 <_Z10run_simplev>:
  400ca0:	31 c0                	xor    %eax,%eax
  400ca2:	66 0f 28 05 16 04 00 	movapd 0x416(%rip),%xmm0        # 4010c0 <_IO_stdin_used+0x60>
  400ca9:	00 
  400caa:	66 0f 28 0d 0e 04 00 	movapd 0x40e(%rip),%xmm1        # 4010c0 <_IO_stdin_used+0x60>
  400cb1:	00 
  400cb2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400cb9:	1f 84 00 00 00 00 00 
  400cc0:	31 c9                	xor    %ecx,%ecx
  400cc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400cc9:	1f 84 00 00 00 00 00 
  400cd0:	48 6b d1 38          	imul   $0x38,%rcx,%rdx
  400cd4:	66 0f 10 92 c8 20 60 	movupd 0x6020c8(%rdx),%xmm2
  400cdb:	00 
  400cdc:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  400ce0:	f2 0f 10 9a a0 20 60 	movsd  0x6020a0(%rdx),%xmm3
  400ce7:	00 
  400ce8:	0f 16 db             	movlhps %xmm3,%xmm3
  400ceb:	66 0f 5e d3          	divpd  %xmm3,%xmm2
  400cef:	66 0f 10 9a b8 20 60 	movupd 0x6020b8(%rdx),%xmm3
  400cf6:	00 
  400cf7:	66 0f 10 a2 a8 20 60 	movupd 0x6020a8(%rdx),%xmm4
  400cfe:	00 
  400cff:	be 64 00 00 00       	mov    $0x64,%esi
  400d04:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400d0b:	00 00 00 00 00 
  400d10:	66 0f 58 da          	addpd  %xmm2,%xmm3
  400d14:	66 0f 28 ea          	movapd %xmm2,%xmm5
  400d18:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  400d1c:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  400d20:	66 0f 58 dc          	addpd  %xmm4,%xmm3
  400d24:	66 0f 28 e2          	movapd %xmm2,%xmm4
  400d28:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  400d2c:	66 0f 59 e9          	mulpd  %xmm1,%xmm5
  400d30:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  400d34:	66 0f 28 f2          	movapd %xmm2,%xmm6
  400d38:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  400d3c:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  400d40:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  400d44:	66 0f 28 da          	movapd %xmm2,%xmm3
  400d48:	66 0f 58 de          	addpd  %xmm6,%xmm3
  400d4c:	66 0f 59 f1          	mulpd  %xmm1,%xmm6
  400d50:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  400d54:	66 0f 28 e3          	movapd %xmm3,%xmm4
  400d58:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  400d5c:	66 0f 58 e6          	addpd  %xmm6,%xmm4
  400d60:	83 c6 fb             	add    $0xfffffffb,%esi
  400d63:	75 ab                	jne    400d10 <_Z10run_simplev+0x70>
  400d65:	66 0f 11 9a b8 20 60 	movupd %xmm3,0x6020b8(%rdx)
  400d6c:	00 
  400d6d:	66 0f 11 a2 a8 20 60 	movupd %xmm4,0x6020a8(%rdx)
  400d74:	00 
  400d75:	48 ff c1             	inc    %rcx
  400d78:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  400d7f:	0f 85 4b ff ff ff    	jne    400cd0 <_Z10run_simplev+0x30>
  400d85:	ff c0                	inc    %eax
  400d87:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  400d8c:	0f 85 2e ff ff ff    	jne    400cc0 <_Z10run_simplev+0x20>
  400d92:	c3                   	retq   
  400d93:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400d9a:	84 00 00 00 00 00 

0000000000400da0 <main>:
  400da0:	55                   	push   %rbp
  400da1:	41 57                	push   %r15
  400da3:	41 56                	push   %r14
  400da5:	41 55                	push   %r13
  400da7:	41 54                	push   %r12
  400da9:	53                   	push   %rbx
  400daa:	50                   	push   %rax
  400dab:	e8 b0 f9 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400db0:	48 89 c5             	mov    %rax,%rbp
  400db3:	e8 18 fb ff ff       	callq  4008d0 <_Z13instantiationv>
  400db8:	e8 a3 f9 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400dbd:	48 89 c3             	mov    %rax,%rbx
  400dc0:	48 29 eb             	sub    %rbp,%rbx
  400dc3:	e8 98 f9 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400dc8:	48 89 c5             	mov    %rax,%rbp
  400dcb:	e8 60 fc ff ff       	callq  400a30 <_Z14run_simulationv>
  400dd0:	e8 8b f9 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400dd5:	49 89 c7             	mov    %rax,%r15
  400dd8:	49 29 ef             	sub    %rbp,%r15
  400ddb:	31 ed                	xor    %ebp,%ebp
  400ddd:	e8 7e f9 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400de2:	49 89 c6             	mov    %rax,%r14
  400de5:	66 0f 28 05 d3 02 00 	movapd 0x2d3(%rip),%xmm0        # 4010c0 <_IO_stdin_used+0x60>
  400dec:	00 
  400ded:	66 0f 28 0d cb 02 00 	movapd 0x2cb(%rip),%xmm1        # 4010c0 <_IO_stdin_used+0x60>
  400df4:	00 
  400df5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400dfc:	00 00 00 00 
  400e00:	31 c0                	xor    %eax,%eax
  400e02:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e09:	1f 84 00 00 00 00 00 
  400e10:	48 6b c8 38          	imul   $0x38,%rax,%rcx
  400e14:	66 0f 10 91 c8 20 60 	movupd 0x6020c8(%rcx),%xmm2
  400e1b:	00 
  400e1c:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  400e20:	f2 0f 10 99 a0 20 60 	movsd  0x6020a0(%rcx),%xmm3
  400e27:	00 
  400e28:	0f 16 db             	movlhps %xmm3,%xmm3
  400e2b:	66 0f 5e d3          	divpd  %xmm3,%xmm2
  400e2f:	66 0f 10 99 b8 20 60 	movupd 0x6020b8(%rcx),%xmm3
  400e36:	00 
  400e37:	66 0f 10 a1 a8 20 60 	movupd 0x6020a8(%rcx),%xmm4
  400e3e:	00 
  400e3f:	ba 64 00 00 00       	mov    $0x64,%edx
  400e44:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e4b:	00 00 00 00 00 
  400e50:	66 0f 58 da          	addpd  %xmm2,%xmm3
  400e54:	66 0f 28 ea          	movapd %xmm2,%xmm5
  400e58:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  400e5c:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  400e60:	66 0f 58 dc          	addpd  %xmm4,%xmm3
  400e64:	66 0f 28 e2          	movapd %xmm2,%xmm4
  400e68:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  400e6c:	66 0f 59 e9          	mulpd  %xmm1,%xmm5
  400e70:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  400e74:	66 0f 28 f2          	movapd %xmm2,%xmm6
  400e78:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  400e7c:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  400e80:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  400e84:	66 0f 28 da          	movapd %xmm2,%xmm3
  400e88:	66 0f 58 de          	addpd  %xmm6,%xmm3
  400e8c:	66 0f 59 f1          	mulpd  %xmm1,%xmm6
  400e90:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  400e94:	66 0f 28 e3          	movapd %xmm3,%xmm4
  400e98:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  400e9c:	66 0f 58 e6          	addpd  %xmm6,%xmm4
  400ea0:	83 c2 fb             	add    $0xfffffffb,%edx
  400ea3:	75 ab                	jne    400e50 <main+0xb0>
  400ea5:	66 0f 11 99 b8 20 60 	movupd %xmm3,0x6020b8(%rcx)
  400eac:	00 
  400ead:	66 0f 11 a1 a8 20 60 	movupd %xmm4,0x6020a8(%rcx)
  400eb4:	00 
  400eb5:	48 ff c0             	inc    %rax
  400eb8:	48 3d 40 1f 00 00    	cmp    $0x1f40,%rax
  400ebe:	0f 85 4c ff ff ff    	jne    400e10 <main+0x70>
  400ec4:	ff c5                	inc    %ebp
  400ec6:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
  400ecc:	0f 85 2e ff ff ff    	jne    400e00 <main+0x60>
  400ed2:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  400ed9:	de 1b 43 
  400edc:	48 89 d8             	mov    %rbx,%rax
  400edf:	48 f7 e9             	imul   %rcx
  400ee2:	49 89 d4             	mov    %rdx,%r12
  400ee5:	4c 89 e0             	mov    %r12,%rax
  400ee8:	48 c1 e8 3f          	shr    $0x3f,%rax
  400eec:	49 c1 fc 12          	sar    $0x12,%r12
  400ef0:	49 01 c4             	add    %rax,%r12
  400ef3:	4c 89 f8             	mov    %r15,%rax
  400ef6:	48 f7 e9             	imul   %rcx
  400ef9:	49 89 d5             	mov    %rdx,%r13
  400efc:	4c 89 e8             	mov    %r13,%rax
  400eff:	48 c1 e8 3f          	shr    $0x3f,%rax
  400f03:	49 c1 fd 12          	sar    $0x12,%r13
  400f07:	49 01 c5             	add    %rax,%r13
  400f0a:	e8 51 f8 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f0f:	4c 29 f0             	sub    %r14,%rax
  400f12:	48 89 04 24          	mov    %rax,(%rsp)
  400f16:	bd 0b 00 00 00       	mov    $0xb,%ebp
  400f1b:	31 db                	xor    %ebx,%ebx
  400f1d:	41 be b0 20 60 00    	mov    $0x6020b0,%r14d
  400f23:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400f2a:	84 00 00 00 00 00 
  400f30:	f2 41 0f 10 46 f8    	movsd  -0x8(%r14),%xmm0
  400f36:	f2 41 0f 10 0e       	movsd  (%r14),%xmm1
  400f3b:	f2 0f 5a d0          	cvtsd2ss %xmm0,%xmm2
  400f3f:	66 0f 7e d0          	movd   %xmm2,%eax
  400f43:	01 e8                	add    %ebp,%eax
  400f45:	0f 57 d2             	xorps  %xmm2,%xmm2
  400f48:	f2 0f 5a d1          	cvtsd2ss %xmm1,%xmm2
  400f4c:	66 0f 7e d5          	movd   %xmm2,%ebp
  400f50:	01 c5                	add    %eax,%ebp
  400f52:	48 63 c5             	movslq %ebp,%rax
  400f55:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  400f5c:	48 89 c1             	mov    %rax,%rcx
  400f5f:	48 c1 e9 3f          	shr    $0x3f,%rcx
  400f63:	48 c1 f8 33          	sar    $0x33,%rax
  400f67:	01 c8                	add    %ecx,%eax
  400f69:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  400f70:	48 83 fb 09          	cmp    $0x9,%rbx
  400f74:	77 0f                	ja     400f85 <main+0x1e5>
  400f76:	bf e0 10 40 00       	mov    $0x4010e0,%edi
  400f7b:	b0 02                	mov    $0x2,%al
  400f7d:	48 89 de             	mov    %rbx,%rsi
  400f80:	e8 7b f7 ff ff       	callq  400700 <printf@plt>
  400f85:	44 29 fd             	sub    %r15d,%ebp
  400f88:	48 ff c3             	inc    %rbx
  400f8b:	49 83 c6 38          	add    $0x38,%r14
  400f8f:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  400f96:	75 98                	jne    400f30 <main+0x190>
  400f98:	48 8b 04 24          	mov    (%rsp),%rax
  400f9c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  400fa3:	de 1b 43 
  400fa6:	48 f7 e9             	imul   %rcx
  400fa9:	48 89 d0             	mov    %rdx,%rax
  400fac:	48 c1 e8 3f          	shr    $0x3f,%rax
  400fb0:	48 c1 fa 12          	sar    $0x12,%rdx
  400fb4:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  400fb8:	bf f5 10 40 00       	mov    $0x4010f5,%edi
  400fbd:	31 c0                	xor    %eax,%eax
  400fbf:	4c 89 e6             	mov    %r12,%rsi
  400fc2:	4c 89 ea             	mov    %r13,%rdx
  400fc5:	41 89 e8             	mov    %ebp,%r8d
  400fc8:	e8 33 f7 ff ff       	callq  400700 <printf@plt>
  400fcd:	31 c0                	xor    %eax,%eax
  400fcf:	48 83 c4 08          	add    $0x8,%rsp
  400fd3:	5b                   	pop    %rbx
  400fd4:	41 5c                	pop    %r12
  400fd6:	41 5d                	pop    %r13
  400fd8:	41 5e                	pop    %r14
  400fda:	41 5f                	pop    %r15
  400fdc:	5d                   	pop    %rbp
  400fdd:	c3                   	retq   
  400fde:	66 90                	xchg   %ax,%ax

0000000000400fe0 <__libc_csu_init>:
  400fe0:	41 57                	push   %r15
  400fe2:	41 56                	push   %r14
  400fe4:	41 89 ff             	mov    %edi,%r15d
  400fe7:	41 55                	push   %r13
  400fe9:	41 54                	push   %r12
  400feb:	4c 8d 25 e6 0d 20 00 	lea    0x200de6(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  400ff2:	55                   	push   %rbp
  400ff3:	48 8d 2d ee 0d 20 00 	lea    0x200dee(%rip),%rbp        # 601de8 <__init_array_end>
  400ffa:	53                   	push   %rbx
  400ffb:	49 89 f6             	mov    %rsi,%r14
  400ffe:	49 89 d5             	mov    %rdx,%r13
  401001:	4c 29 e5             	sub    %r12,%rbp
  401004:	48 83 ec 08          	sub    $0x8,%rsp
  401008:	48 c1 fd 03          	sar    $0x3,%rbp
  40100c:	e8 b7 f6 ff ff       	callq  4006c8 <_init>
  401011:	48 85 ed             	test   %rbp,%rbp
  401014:	74 20                	je     401036 <__libc_csu_init+0x56>
  401016:	31 db                	xor    %ebx,%ebx
  401018:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40101f:	00 
  401020:	4c 89 ea             	mov    %r13,%rdx
  401023:	4c 89 f6             	mov    %r14,%rsi
  401026:	44 89 ff             	mov    %r15d,%edi
  401029:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40102d:	48 83 c3 01          	add    $0x1,%rbx
  401031:	48 39 eb             	cmp    %rbp,%rbx
  401034:	75 ea                	jne    401020 <__libc_csu_init+0x40>
  401036:	48 83 c4 08          	add    $0x8,%rsp
  40103a:	5b                   	pop    %rbx
  40103b:	5d                   	pop    %rbp
  40103c:	41 5c                	pop    %r12
  40103e:	41 5d                	pop    %r13
  401040:	41 5e                	pop    %r14
  401042:	41 5f                	pop    %r15
  401044:	c3                   	retq   
  401045:	90                   	nop
  401046:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40104d:	00 00 00 

0000000000401050 <__libc_csu_fini>:
  401050:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401054 <_fini>:
  401054:	48 83 ec 08          	sub    $0x8,%rsp
  401058:	48 83 c4 08          	add    $0x8,%rsp
  40105c:	c3                   	retq   
