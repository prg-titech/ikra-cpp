
bin/clang++-3.8_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

0000000000400700 <_init>:
  400700:	48 83 ec 08          	sub    $0x8,%rsp
  400704:	48 8b 05 ed 18 20 00 	mov    0x2018ed(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40070b:	48 85 c0             	test   %rax,%rax
  40070e:	74 05                	je     400715 <_init+0x15>
  400710:	e8 bb 00 00 00       	callq  4007d0 <sqrt@plt+0x10>
  400715:	48 83 c4 08          	add    $0x8,%rsp
  400719:	c3                   	retq   

Disassembly of section .plt:

0000000000400720 <printf@plt-0x10>:
  400720:	ff 35 e2 18 20 00    	pushq  0x2018e2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400726:	ff 25 e4 18 20 00    	jmpq   *0x2018e4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40072c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400730 <printf@plt>:
  400730:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400736:	68 00 00 00 00       	pushq  $0x0
  40073b:	e9 e0 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400740 <memset@plt>:
  400740:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400746:	68 01 00 00 00       	pushq  $0x1
  40074b:	e9 d0 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400750 <_ZNSt8ios_base4InitC1Ev@plt>:
  400750:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400756:	68 02 00 00 00       	pushq  $0x2
  40075b:	e9 c0 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400760 <__libc_start_main@plt>:
  400760:	ff 25 ca 18 20 00    	jmpq   *0x2018ca(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400766:	68 03 00 00 00       	pushq  $0x3
  40076b:	e9 b0 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400770 <__cxa_atexit@plt>:
  400770:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400776:	68 04 00 00 00       	pushq  $0x4
  40077b:	e9 a0 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400780 <_ZNSt8ios_base4InitD1Ev@plt>:
  400780:	ff 25 ba 18 20 00    	jmpq   *0x2018ba(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400786:	68 05 00 00 00       	pushq  $0x5
  40078b:	e9 90 ff ff ff       	jmpq   400720 <_init+0x20>

0000000000400790 <srand@plt>:
  400790:	ff 25 b2 18 20 00    	jmpq   *0x2018b2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400796:	68 06 00 00 00       	pushq  $0x6
  40079b:	e9 80 ff ff ff       	jmpq   400720 <_init+0x20>

00000000004007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  4007a0:	ff 25 aa 18 20 00    	jmpq   *0x2018aa(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  4007a6:	68 07 00 00 00       	pushq  $0x7
  4007ab:	e9 70 ff ff ff       	jmpq   400720 <_init+0x20>

00000000004007b0 <rand@plt>:
  4007b0:	ff 25 a2 18 20 00    	jmpq   *0x2018a2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  4007b6:	68 08 00 00 00       	pushq  $0x8
  4007bb:	e9 60 ff ff ff       	jmpq   400720 <_init+0x20>

00000000004007c0 <sqrt@plt>:
  4007c0:	ff 25 9a 18 20 00    	jmpq   *0x20189a(%rip)        # 602060 <_GLOBAL_OFFSET_TABLE_+0x60>
  4007c6:	68 09 00 00 00       	pushq  $0x9
  4007cb:	e9 50 ff ff ff       	jmpq   400720 <_init+0x20>

Disassembly of section .plt.got:

00000000004007d0 <.plt.got>:
  4007d0:	ff 25 22 18 20 00    	jmpq   *0x201822(%rip)        # 601ff8 <_DYNAMIC+0x200>
  4007d6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007e0 <_GLOBAL__sub_I_soa.cc>:
  4007e0:	50                   	push   %rax
  4007e1:	bf 90 20 60 00       	mov    $0x602090,%edi
  4007e6:	e8 65 ff ff ff       	callq  400750 <_ZNSt8ios_base4InitC1Ev@plt>
  4007eb:	bf 80 07 40 00       	mov    $0x400780,%edi
  4007f0:	be 90 20 60 00       	mov    $0x602090,%esi
  4007f5:	ba 70 20 60 00       	mov    $0x602070,%edx
  4007fa:	58                   	pop    %rax
  4007fb:	e9 70 ff ff ff       	jmpq   400770 <__cxa_atexit@plt>

0000000000400800 <_start>:
  400800:	31 ed                	xor    %ebp,%ebp
  400802:	49 89 d1             	mov    %rdx,%r9
  400805:	5e                   	pop    %rsi
  400806:	48 89 e2             	mov    %rsp,%rdx
  400809:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40080d:	50                   	push   %rax
  40080e:	54                   	push   %rsp
  40080f:	49 c7 c0 10 15 40 00 	mov    $0x401510,%r8
  400816:	48 c7 c1 a0 14 40 00 	mov    $0x4014a0,%rcx
  40081d:	48 c7 c7 50 12 40 00 	mov    $0x401250,%rdi
  400824:	e8 37 ff ff ff       	callq  400760 <__libc_start_main@plt>
  400829:	f4                   	hlt    
  40082a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400830 <deregister_tm_clones>:
  400830:	b8 7f 20 60 00       	mov    $0x60207f,%eax
  400835:	55                   	push   %rbp
  400836:	48 2d 78 20 60 00    	sub    $0x602078,%rax
  40083c:	48 83 f8 0e          	cmp    $0xe,%rax
  400840:	48 89 e5             	mov    %rsp,%rbp
  400843:	76 1b                	jbe    400860 <deregister_tm_clones+0x30>
  400845:	b8 00 00 00 00       	mov    $0x0,%eax
  40084a:	48 85 c0             	test   %rax,%rax
  40084d:	74 11                	je     400860 <deregister_tm_clones+0x30>
  40084f:	5d                   	pop    %rbp
  400850:	bf 78 20 60 00       	mov    $0x602078,%edi
  400855:	ff e0                	jmpq   *%rax
  400857:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40085e:	00 00 
  400860:	5d                   	pop    %rbp
  400861:	c3                   	retq   
  400862:	0f 1f 40 00          	nopl   0x0(%rax)
  400866:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40086d:	00 00 00 

0000000000400870 <register_tm_clones>:
  400870:	be 78 20 60 00       	mov    $0x602078,%esi
  400875:	55                   	push   %rbp
  400876:	48 81 ee 78 20 60 00 	sub    $0x602078,%rsi
  40087d:	48 c1 fe 03          	sar    $0x3,%rsi
  400881:	48 89 e5             	mov    %rsp,%rbp
  400884:	48 89 f0             	mov    %rsi,%rax
  400887:	48 c1 e8 3f          	shr    $0x3f,%rax
  40088b:	48 01 c6             	add    %rax,%rsi
  40088e:	48 d1 fe             	sar    %rsi
  400891:	74 15                	je     4008a8 <register_tm_clones+0x38>
  400893:	b8 00 00 00 00       	mov    $0x0,%eax
  400898:	48 85 c0             	test   %rax,%rax
  40089b:	74 0b                	je     4008a8 <register_tm_clones+0x38>
  40089d:	5d                   	pop    %rbp
  40089e:	bf 78 20 60 00       	mov    $0x602078,%edi
  4008a3:	ff e0                	jmpq   *%rax
  4008a5:	0f 1f 00             	nopl   (%rax)
  4008a8:	5d                   	pop    %rbp
  4008a9:	c3                   	retq   
  4008aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008b0 <__do_global_dtors_aux>:
  4008b0:	80 3d c9 17 20 00 00 	cmpb   $0x0,0x2017c9(%rip)        # 602080 <completed.7585>
  4008b7:	75 11                	jne    4008ca <__do_global_dtors_aux+0x1a>
  4008b9:	55                   	push   %rbp
  4008ba:	48 89 e5             	mov    %rsp,%rbp
  4008bd:	e8 6e ff ff ff       	callq  400830 <deregister_tm_clones>
  4008c2:	5d                   	pop    %rbp
  4008c3:	c6 05 b6 17 20 00 01 	movb   $0x1,0x2017b6(%rip)        # 602080 <completed.7585>
  4008ca:	f3 c3                	repz retq 
  4008cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008d0 <frame_dummy>:
  4008d0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  4008d5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4008d9:	75 05                	jne    4008e0 <frame_dummy+0x10>
  4008db:	eb 93                	jmp    400870 <register_tm_clones>
  4008dd:	0f 1f 00             	nopl   (%rax)
  4008e0:	b8 00 00 00 00       	mov    $0x0,%eax
  4008e5:	48 85 c0             	test   %rax,%rax
  4008e8:	74 f1                	je     4008db <frame_dummy+0xb>
  4008ea:	55                   	push   %rbp
  4008eb:	48 89 e5             	mov    %rsp,%rbp
  4008ee:	ff d0                	callq  *%rax
  4008f0:	5d                   	pop    %rbp
  4008f1:	e9 7a ff ff ff       	jmpq   400870 <register_tm_clones>
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_Z11r_float2intf>:
  400900:	66 0f 7e c0          	movd   %xmm0,%eax
  400904:	c3                   	retq   
  400905:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40090c:	00 00 00 00 

0000000000400910 <_Z15Body_initializemddddd>:
  400910:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  400917:	60 00 
  400919:	f2 0f 11 0c fd a0 1a 	movsd  %xmm1,0x611aa0(,%rdi,8)
  400920:	61 00 
  400922:	f2 0f 11 14 fd a0 14 	movsd  %xmm2,0x6214a0(,%rdi,8)
  400929:	62 00 
  40092b:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  400932:	63 00 
  400934:	f2 0f 11 24 fd a0 08 	movsd  %xmm4,0x6408a0(,%rdi,8)
  40093b:	64 00 
  40093d:	48 c7 04 fd a0 02 65 	movq   $0x0,0x6502a0(,%rdi,8)
  400944:	00 00 00 00 00 
  400949:	48 c7 04 fd a0 fc 65 	movq   $0x0,0x65fca0(,%rdi,8)
  400950:	00 00 00 00 00 
  400955:	c3                   	retq   
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <_Z14Body_add_forcemm>:
  400960:	41 56                	push   %r14
  400962:	53                   	push   %rbx
  400963:	48 83 ec 18          	sub    $0x18,%rsp
  400967:	49 89 f6             	mov    %rsi,%r14
  40096a:	48 89 fb             	mov    %rdi,%rbx
  40096d:	4c 39 f3             	cmp    %r14,%rbx
  400970:	0f 84 cb 00 00 00    	je     400a41 <_Z14Body_add_forcemm+0xe1>
  400976:	f2 42 0f 10 24 f5 a0 	movsd  0x611aa0(,%r14,8),%xmm4
  40097d:	1a 61 00 
  400980:	f2 0f 5c 24 dd a0 1a 	subsd  0x611aa0(,%rbx,8),%xmm4
  400987:	61 00 
  400989:	f2 42 0f 10 1c f5 a0 	movsd  0x6214a0(,%r14,8),%xmm3
  400990:	14 62 00 
  400993:	f2 0f 5c 1c dd a0 14 	subsd  0x6214a0(,%rbx,8),%xmm3
  40099a:	62 00 
  40099c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4009a0:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  4009a4:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4009a8:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  4009ac:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4009b0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4009b3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4009b7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4009bb:	7b 21                	jnp    4009de <_Z14Body_add_forcemm+0x7e>
  4009bd:	66 0f 28 c1          	movapd %xmm1,%xmm0
  4009c1:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  4009c7:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  4009cd:	e8 ee fd ff ff       	callq  4007c0 <sqrt@plt>
  4009d2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4009d8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4009de:	f2 0f 10 0c dd a0 20 	movsd  0x6020a0(,%rbx,8),%xmm1
  4009e5:	60 00 
  4009e7:	f2 0f 59 0d 39 0b 00 	mulsd  0xb39(%rip),%xmm1        # 401528 <_IO_stdin_used+0x8>
  4009ee:	00 
  4009ef:	f2 42 0f 59 0c f5 a0 	mulsd  0x6020a0(,%r14,8),%xmm1
  4009f6:	20 60 00 
  4009f9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4009fd:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400a01:	f2 0f 58 15 27 0b 00 	addsd  0xb27(%rip),%xmm2        # 401530 <_IO_stdin_used+0x10>
  400a08:	00 
  400a09:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400a0d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  400a11:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  400a15:	f2 0f 58 24 dd a0 02 	addsd  0x6502a0(,%rbx,8),%xmm4
  400a1c:	65 00 
  400a1e:	f2 0f 11 24 dd a0 02 	movsd  %xmm4,0x6502a0(,%rbx,8)
  400a25:	65 00 
  400a27:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  400a2b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  400a2f:	f2 0f 58 1c dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm3
  400a36:	65 00 
  400a38:	f2 0f 11 1c dd a0 fc 	movsd  %xmm3,0x65fca0(,%rbx,8)
  400a3f:	65 00 
  400a41:	48 83 c4 18          	add    $0x18,%rsp
  400a45:	5b                   	pop    %rbx
  400a46:	41 5e                	pop    %r14
  400a48:	c3                   	retq   
  400a49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a50 <_Z21Body_add_force_to_allm>:
  400a50:	41 57                	push   %r15
  400a52:	41 56                	push   %r14
  400a54:	53                   	push   %rbx
  400a55:	48 83 ec 10          	sub    $0x10,%rsp
  400a59:	49 89 ff             	mov    %rdi,%r15
  400a5c:	4d 89 fe             	mov    %r15,%r14
  400a5f:	49 f7 de             	neg    %r14
  400a62:	48 c7 c3 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rbx
  400a69:	f2 0f 10 1d b7 0a 00 	movsd  0xab7(%rip),%xmm3        # 401528 <_IO_stdin_used+0x8>
  400a70:	00 
  400a71:	f2 0f 10 25 b7 0a 00 	movsd  0xab7(%rip),%xmm4        # 401530 <_IO_stdin_used+0x10>
  400a78:	00 
  400a79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400a80:	49 8d 04 1e          	lea    (%r14,%rbx,1),%rax
  400a84:	48 3d c0 e0 ff ff    	cmp    $0xffffffffffffe0c0,%rax
  400a8a:	0f 84 d1 00 00 00    	je     400b61 <_Z21Body_add_force_to_allm+0x111>
  400a90:	f2 42 0f 10 34 fd a0 	movsd  0x611aa0(,%r15,8),%xmm6
  400a97:	1a 61 00 
  400a9a:	f2 0f 5c 34 dd a0 14 	subsd  0x6214a0(,%rbx,8),%xmm6
  400aa1:	62 00 
  400aa3:	f2 42 0f 10 2c fd a0 	movsd  0x6214a0(,%r15,8),%xmm5
  400aaa:	14 62 00 
  400aad:	f2 0f 5c 2c dd a0 0e 	subsd  0x630ea0(,%rbx,8),%xmm5
  400ab4:	63 00 
  400ab6:	66 0f 28 c6          	movapd %xmm6,%xmm0
  400aba:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400abe:	66 0f 28 cd          	movapd %xmm5,%xmm1
  400ac2:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400ac6:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400aca:	0f 57 c0             	xorps  %xmm0,%xmm0
  400acd:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400ad1:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400ad5:	7b 2f                	jnp    400b06 <_Z21Body_add_force_to_allm+0xb6>
  400ad7:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400adb:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  400ae1:	f2 0f 11 34 24       	movsd  %xmm6,(%rsp)
  400ae6:	e8 d5 fc ff ff       	callq  4007c0 <sqrt@plt>
  400aeb:	f2 0f 10 34 24       	movsd  (%rsp),%xmm6
  400af0:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  400af6:	f2 0f 10 25 32 0a 00 	movsd  0xa32(%rip),%xmm4        # 401530 <_IO_stdin_used+0x10>
  400afd:	00 
  400afe:	f2 0f 10 1d 22 0a 00 	movsd  0xa22(%rip),%xmm3        # 401528 <_IO_stdin_used+0x8>
  400b05:	00 
  400b06:	f2 0f 10 0c dd a0 1a 	movsd  0x611aa0(,%rbx,8),%xmm1
  400b0d:	61 00 
  400b0f:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400b13:	f2 42 0f 59 0c fd a0 	mulsd  0x6020a0(,%r15,8),%xmm1
  400b1a:	20 60 00 
  400b1d:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400b21:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400b25:	f2 0f 58 d4          	addsd  %xmm4,%xmm2
  400b29:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400b2d:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  400b31:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400b35:	f2 0f 58 34 dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm6
  400b3c:	65 00 
  400b3e:	f2 0f 11 34 dd a0 fc 	movsd  %xmm6,0x65fca0(,%rbx,8)
  400b45:	65 00 
  400b47:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  400b4b:	f2 0f 5e e8          	divsd  %xmm0,%xmm5
  400b4f:	f2 0f 58 2c dd a0 f6 	addsd  0x66f6a0(,%rbx,8),%xmm5
  400b56:	66 00 
  400b58:	f2 0f 11 2c dd a0 f6 	movsd  %xmm5,0x66f6a0(,%rbx,8)
  400b5f:	66 00 
  400b61:	48 ff c3             	inc    %rbx
  400b64:	0f 85 16 ff ff ff    	jne    400a80 <_Z21Body_add_force_to_allm+0x30>
  400b6a:	48 83 c4 10          	add    $0x10,%rsp
  400b6e:	5b                   	pop    %rbx
  400b6f:	41 5e                	pop    %r14
  400b71:	41 5f                	pop    %r15
  400b73:	c3                   	retq   
  400b74:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b7b:	00 00 00 00 00 

0000000000400b80 <_Z16Body_reset_forcem>:
  400b80:	48 c7 04 fd a0 02 65 	movq   $0x0,0x6502a0(,%rdi,8)
  400b87:	00 00 00 00 00 
  400b8c:	48 c7 04 fd a0 fc 65 	movq   $0x0,0x65fca0(,%rdi,8)
  400b93:	00 00 00 00 00 
  400b98:	c3                   	retq   
  400b99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ba0 <_Z11Body_updatemd>:
  400ba0:	f2 0f 10 14 fd a0 02 	movsd  0x6502a0(,%rdi,8),%xmm2
  400ba7:	65 00 
  400ba9:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  400bad:	f2 0f 10 1c fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm3
  400bb4:	60 00 
  400bb6:	f2 0f 5e d3          	divsd  %xmm3,%xmm2
  400bba:	f2 0f 58 14 fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm2
  400bc1:	63 00 
  400bc3:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  400bca:	63 00 
  400bcc:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  400bd3:	65 00 
  400bd5:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400bd9:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400bdd:	f2 0f 58 0c fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm1
  400be4:	64 00 
  400be6:	f2 0f 11 0c fd a0 08 	movsd  %xmm1,0x6408a0(,%rdi,8)
  400bed:	64 00 
  400bef:	66 0f 28 da          	movapd %xmm2,%xmm3
  400bf3:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400bf7:	f2 0f 58 1c fd a0 1a 	addsd  0x611aa0(,%rdi,8),%xmm3
  400bfe:	61 00 
  400c00:	f2 0f 11 1c fd a0 1a 	movsd  %xmm3,0x611aa0(,%rdi,8)
  400c07:	61 00 
  400c09:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400c0d:	f2 0f 58 04 fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm0
  400c14:	62 00 
  400c16:	f2 0f 11 04 fd a0 14 	movsd  %xmm0,0x6214a0(,%rdi,8)
  400c1d:	62 00 
  400c1f:	f2 0f 10 25 11 09 00 	movsd  0x911(%rip),%xmm4        # 401538 <_IO_stdin_used+0x18>
  400c26:	00 
  400c27:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  400c2b:	77 0a                	ja     400c37 <_Z11Body_updatemd+0x97>
  400c2d:	66 0f 2e 1d 0b 09 00 	ucomisd 0x90b(%rip),%xmm3        # 401540 <_IO_stdin_used+0x20>
  400c34:	00 
  400c35:	76 11                	jbe    400c48 <_Z11Body_updatemd+0xa8>
  400c37:	66 0f 57 15 31 09 00 	xorpd  0x931(%rip),%xmm2        # 401570 <_IO_stdin_used+0x50>
  400c3e:	00 
  400c3f:	66 0f 13 14 fd a0 0e 	movlpd %xmm2,0x630ea0(,%rdi,8)
  400c46:	63 00 
  400c48:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  400c4c:	77 0a                	ja     400c58 <_Z11Body_updatemd+0xb8>
  400c4e:	66 0f 2e 05 ea 08 00 	ucomisd 0x8ea(%rip),%xmm0        # 401540 <_IO_stdin_used+0x20>
  400c55:	00 
  400c56:	76 11                	jbe    400c69 <_Z11Body_updatemd+0xc9>
  400c58:	66 0f 57 0d 10 09 00 	xorpd  0x910(%rip),%xmm1        # 401570 <_IO_stdin_used+0x50>
  400c5f:	00 
  400c60:	66 0f 13 0c fd a0 08 	movlpd %xmm1,0x6408a0(,%rdi,8)
  400c67:	64 00 
  400c69:	c3                   	retq   
  400c6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c70 <_Z26Body_codegen_simple_updatemd>:
  400c70:	0f 16 c0             	movlhps %xmm0,%xmm0
  400c73:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  400c7a:	65 00 
  400c7c:	66 0f 16 0c fd a0 02 	movhpd 0x6502a0(,%rdi,8),%xmm1
  400c83:	65 00 
  400c85:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  400c89:	f2 0f 10 14 fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm2
  400c90:	60 00 
  400c92:	0f 16 d2             	movlhps %xmm2,%xmm2
  400c95:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  400c99:	f2 0f 10 1c fd a0 14 	movsd  0x6214a0(,%rdi,8),%xmm3
  400ca0:	62 00 
  400ca2:	66 0f 16 1c fd a0 1a 	movhpd 0x611aa0(,%rdi,8),%xmm3
  400ca9:	61 00 
  400cab:	f2 0f 10 14 fd a0 08 	movsd  0x6408a0(,%rdi,8),%xmm2
  400cb2:	64 00 
  400cb4:	66 0f 16 14 fd a0 0e 	movhpd 0x630ea0(,%rdi,8),%xmm2
  400cbb:	63 00 
  400cbd:	b8 64 00 00 00       	mov    $0x64,%eax
  400cc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400cc9:	1f 84 00 00 00 00 00 
  400cd0:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400cd4:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400cd8:	66 0f 59 e2          	mulpd  %xmm2,%xmm4
  400cdc:	66 0f 58 e3          	addpd  %xmm3,%xmm4
  400ce0:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400ce4:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400ce8:	66 0f 59 da          	mulpd  %xmm2,%xmm3
  400cec:	66 0f 58 dc          	addpd  %xmm4,%xmm3
  400cf0:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400cf4:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400cf8:	66 0f 59 e2          	mulpd  %xmm2,%xmm4
  400cfc:	66 0f 58 e3          	addpd  %xmm3,%xmm4
  400d00:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400d04:	66 0f 28 e8          	movapd %xmm0,%xmm5
  400d08:	66 0f 59 ea          	mulpd  %xmm2,%xmm5
  400d0c:	66 0f 58 ec          	addpd  %xmm4,%xmm5
  400d10:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400d14:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400d18:	66 0f 59 da          	mulpd  %xmm2,%xmm3
  400d1c:	66 0f 58 dd          	addpd  %xmm5,%xmm3
  400d20:	83 c0 fb             	add    $0xfffffffb,%eax
  400d23:	75 ab                	jne    400cd0 <_Z26Body_codegen_simple_updatemd+0x60>
  400d25:	66 0f 17 14 fd a0 0e 	movhpd %xmm2,0x630ea0(,%rdi,8)
  400d2c:	63 00 
  400d2e:	66 0f 13 14 fd a0 08 	movlpd %xmm2,0x6408a0(,%rdi,8)
  400d35:	64 00 
  400d37:	66 0f 17 1c fd a0 1a 	movhpd %xmm3,0x611aa0(,%rdi,8)
  400d3e:	61 00 
  400d40:	66 0f 13 1c fd a0 14 	movlpd %xmm3,0x6214a0(,%rdi,8)
  400d47:	62 00 
  400d49:	c3                   	retq   
  400d4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400d50 <_Z13instantiationv>:
  400d50:	53                   	push   %rbx
  400d51:	48 83 ec 30          	sub    $0x30,%rsp
  400d55:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400d5a:	e8 31 fa ff ff       	callq  400790 <srand@plt>
  400d5f:	48 c7 c3 00 06 ff ff 	mov    $0xffffffffffff0600,%rbx
  400d66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d6d:	00 00 00 
  400d70:	e8 3b fa ff ff       	callq  4007b0 <rand@plt>
  400d75:	0f 57 c9             	xorps  %xmm1,%xmm1
  400d78:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400d7c:	f2 0f 10 05 c4 07 00 	movsd  0x7c4(%rip),%xmm0        # 401548 <_IO_stdin_used+0x28>
  400d83:	00 
  400d84:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400d88:	f2 0f 10 05 c0 07 00 	movsd  0x7c0(%rip),%xmm0        # 401550 <_IO_stdin_used+0x30>
  400d8f:	00 
  400d90:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400d94:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400d98:	f2 0f 10 05 b8 07 00 	movsd  0x7b8(%rip),%xmm0        # 401558 <_IO_stdin_used+0x38>
  400d9f:	00 
  400da0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400da4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  400daa:	e8 01 fa ff ff       	callq  4007b0 <rand@plt>
  400daf:	0f 57 c9             	xorps  %xmm1,%xmm1
  400db2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400db6:	f2 0f 5e 0d 8a 07 00 	divsd  0x78a(%rip),%xmm1        # 401548 <_IO_stdin_used+0x28>
  400dbd:	00 
  400dbe:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400dc2:	f2 0f 10 05 6e 07 00 	movsd  0x76e(%rip),%xmm0        # 401538 <_IO_stdin_used+0x18>
  400dc9:	00 
  400dca:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400dce:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400dd4:	e8 d7 f9 ff ff       	callq  4007b0 <rand@plt>
  400dd9:	0f 57 c0             	xorps  %xmm0,%xmm0
  400ddc:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400de0:	f2 0f 5e 05 60 07 00 	divsd  0x760(%rip),%xmm0        # 401548 <_IO_stdin_used+0x28>
  400de7:	00 
  400de8:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400dec:	f2 0f 58 05 44 07 00 	addsd  0x744(%rip),%xmm0        # 401538 <_IO_stdin_used+0x18>
  400df3:	00 
  400df4:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  400dfa:	e8 b1 f9 ff ff       	callq  4007b0 <rand@plt>
  400dff:	0f 57 c9             	xorps  %xmm1,%xmm1
  400e02:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400e06:	f2 0f 5e 0d 3a 07 00 	divsd  0x73a(%rip),%xmm1        # 401548 <_IO_stdin_used+0x28>
  400e0d:	00 
  400e0e:	f2 0f 10 05 4a 07 00 	movsd  0x74a(%rip),%xmm0        # 401560 <_IO_stdin_used+0x40>
  400e15:	00 
  400e16:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400e1a:	f2 0f 5e 0d 36 07 00 	divsd  0x736(%rip),%xmm1        # 401558 <_IO_stdin_used+0x38>
  400e21:	00 
  400e22:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400e28:	e8 83 f9 ff ff       	callq  4007b0 <rand@plt>
  400e2d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400e30:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400e34:	f2 0f 5e 05 0c 07 00 	divsd  0x70c(%rip),%xmm0        # 401548 <_IO_stdin_used+0x28>
  400e3b:	00 
  400e3c:	f2 0f 58 05 1c 07 00 	addsd  0x71c(%rip),%xmm0        # 401560 <_IO_stdin_used+0x40>
  400e43:	00 
  400e44:	f2 0f 5e 05 0c 07 00 	divsd  0x70c(%rip),%xmm0        # 401558 <_IO_stdin_used+0x38>
  400e4b:	00 
  400e4c:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  400e52:	f2 0f 11 8b a0 1a 61 	movsd  %xmm1,0x611aa0(%rbx)
  400e59:	00 
  400e5a:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  400e60:	f2 0f 11 8b a0 14 62 	movsd  %xmm1,0x6214a0(%rbx)
  400e67:	00 
  400e68:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  400e6e:	f2 0f 11 8b a0 0e 63 	movsd  %xmm1,0x630ea0(%rbx)
  400e75:	00 
  400e76:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  400e7c:	f2 0f 11 8b a0 08 64 	movsd  %xmm1,0x6408a0(%rbx)
  400e83:	00 
  400e84:	f2 0f 11 83 a0 02 65 	movsd  %xmm0,0x6502a0(%rbx)
  400e8b:	00 
  400e8c:	48 c7 83 a0 fc 65 00 	movq   $0x0,0x65fca0(%rbx)
  400e93:	00 00 00 00 
  400e97:	48 c7 83 a0 f6 66 00 	movq   $0x0,0x66f6a0(%rbx)
  400e9e:	00 00 00 00 
  400ea2:	48 83 c3 08          	add    $0x8,%rbx
  400ea6:	0f 85 c4 fe ff ff    	jne    400d70 <_Z13instantiationv+0x20>
  400eac:	48 83 c4 30          	add    $0x30,%rsp
  400eb0:	5b                   	pop    %rbx
  400eb1:	c3                   	retq   
  400eb2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400eb9:	1f 84 00 00 00 00 00 

0000000000400ec0 <_Z14run_simulationv>:
  400ec0:	41 57                	push   %r15
  400ec2:	41 56                	push   %r14
  400ec4:	53                   	push   %rbx
  400ec5:	48 83 ec 20          	sub    $0x20,%rsp
  400ec9:	45 31 f6             	xor    %r14d,%r14d
  400ecc:	0f 1f 40 00          	nopl   0x0(%rax)
  400ed0:	bf a0 02 65 00       	mov    $0x6502a0,%edi
  400ed5:	31 f6                	xor    %esi,%esi
  400ed7:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400edc:	e8 5f f8 ff ff       	callq  400740 <memset@plt>
  400ee1:	bf a0 fc 65 00       	mov    $0x65fca0,%edi
  400ee6:	31 f6                	xor    %esi,%esi
  400ee8:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400eed:	e8 4e f8 ff ff       	callq  400740 <memset@plt>
  400ef2:	45 31 ff             	xor    %r15d,%r15d
  400ef5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400efc:	00 00 00 00 
  400f00:	31 db                	xor    %ebx,%ebx
  400f02:	f2 0f 10 1d 1e 06 00 	movsd  0x61e(%rip),%xmm3        # 401528 <_IO_stdin_used+0x8>
  400f09:	00 
  400f0a:	f2 0f 10 25 1e 06 00 	movsd  0x61e(%rip),%xmm4        # 401530 <_IO_stdin_used+0x10>
  400f11:	00 
  400f12:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400f19:	1f 84 00 00 00 00 00 
  400f20:	49 39 df             	cmp    %rbx,%r15
  400f23:	0f 84 d3 00 00 00    	je     400ffc <_Z14run_simulationv+0x13c>
  400f29:	f2 42 0f 10 34 fd a0 	movsd  0x611aa0(,%r15,8),%xmm6
  400f30:	1a 61 00 
  400f33:	f2 0f 5c 34 dd a0 1a 	subsd  0x611aa0(,%rbx,8),%xmm6
  400f3a:	61 00 
  400f3c:	f2 42 0f 10 2c fd a0 	movsd  0x6214a0(,%r15,8),%xmm5
  400f43:	14 62 00 
  400f46:	f2 0f 5c 2c dd a0 14 	subsd  0x6214a0(,%rbx,8),%xmm5
  400f4d:	62 00 
  400f4f:	66 0f 28 c6          	movapd %xmm6,%xmm0
  400f53:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400f57:	66 0f 28 cd          	movapd %xmm5,%xmm1
  400f5b:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400f5f:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400f63:	0f 57 c0             	xorps  %xmm0,%xmm0
  400f66:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400f6a:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400f6e:	7b 31                	jnp    400fa1 <_Z14run_simulationv+0xe1>
  400f70:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400f74:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400f7a:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400f80:	e8 3b f8 ff ff       	callq  4007c0 <sqrt@plt>
  400f85:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400f8b:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400f91:	f2 0f 10 25 97 05 00 	movsd  0x597(%rip),%xmm4        # 401530 <_IO_stdin_used+0x10>
  400f98:	00 
  400f99:	f2 0f 10 1d 87 05 00 	movsd  0x587(%rip),%xmm3        # 401528 <_IO_stdin_used+0x8>
  400fa0:	00 
  400fa1:	f2 0f 10 0c dd a0 20 	movsd  0x6020a0(,%rbx,8),%xmm1
  400fa8:	60 00 
  400faa:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400fae:	f2 42 0f 59 0c fd a0 	mulsd  0x6020a0(,%r15,8),%xmm1
  400fb5:	20 60 00 
  400fb8:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400fbc:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400fc0:	f2 0f 58 d4          	addsd  %xmm4,%xmm2
  400fc4:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400fc8:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  400fcc:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400fd0:	f2 0f 58 34 dd a0 02 	addsd  0x6502a0(,%rbx,8),%xmm6
  400fd7:	65 00 
  400fd9:	f2 0f 11 34 dd a0 02 	movsd  %xmm6,0x6502a0(,%rbx,8)
  400fe0:	65 00 
  400fe2:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  400fe6:	f2 0f 5e e8          	divsd  %xmm0,%xmm5
  400fea:	f2 0f 58 2c dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm5
  400ff1:	65 00 
  400ff3:	f2 0f 11 2c dd a0 fc 	movsd  %xmm5,0x65fca0(,%rbx,8)
  400ffa:	65 00 
  400ffc:	48 ff c3             	inc    %rbx
  400fff:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401006:	0f 85 14 ff ff ff    	jne    400f20 <_Z14run_simulationv+0x60>
  40100c:	49 ff c7             	inc    %r15
  40100f:	48 c7 c0 00 06 ff ff 	mov    $0xffffffffffff0600,%rax
  401016:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  40101d:	0f 85 dd fe ff ff    	jne    400f00 <_Z14run_simulationv+0x40>
  401023:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40102a:	84 00 00 00 00 00 
  401030:	f2 0f 10 88 a0 fc 65 	movsd  0x65fca0(%rax),%xmm1
  401037:	00 
  401038:	f2 0f 10 05 10 05 00 	movsd  0x510(%rip),%xmm0        # 401550 <_IO_stdin_used+0x30>
  40103f:	00 
  401040:	66 0f 28 e0          	movapd %xmm0,%xmm4
  401044:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  401048:	f2 0f 10 90 a0 1a 61 	movsd  0x611aa0(%rax),%xmm2
  40104f:	00 
  401050:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  401054:	f2 0f 58 88 a0 08 64 	addsd  0x6408a0(%rax),%xmm1
  40105b:	00 
  40105c:	f2 0f 11 88 a0 08 64 	movsd  %xmm1,0x6408a0(%rax)
  401063:	00 
  401064:	f2 0f 10 80 a0 f6 66 	movsd  0x66f6a0(%rax),%xmm0
  40106b:	00 
  40106c:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  401070:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  401074:	f2 0f 58 80 a0 02 65 	addsd  0x6502a0(%rax),%xmm0
  40107b:	00 
  40107c:	f2 0f 11 80 a0 02 65 	movsd  %xmm0,0x6502a0(%rax)
  401083:	00 
  401084:	66 0f 28 d9          	movapd %xmm1,%xmm3
  401088:	f2 0f 59 dc          	mulsd  %xmm4,%xmm3
  40108c:	f2 0f 58 98 a0 14 62 	addsd  0x6214a0(%rax),%xmm3
  401093:	00 
  401094:	f2 0f 11 98 a0 14 62 	movsd  %xmm3,0x6214a0(%rax)
  40109b:	00 
  40109c:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4010a0:	f2 0f 59 d4          	mulsd  %xmm4,%xmm2
  4010a4:	f2 0f 58 90 a0 0e 63 	addsd  0x630ea0(%rax),%xmm2
  4010ab:	00 
  4010ac:	f2 0f 11 90 a0 0e 63 	movsd  %xmm2,0x630ea0(%rax)
  4010b3:	00 
  4010b4:	f2 0f 10 25 7c 04 00 	movsd  0x47c(%rip),%xmm4        # 401538 <_IO_stdin_used+0x18>
  4010bb:	00 
  4010bc:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  4010c0:	77 0a                	ja     4010cc <_Z14run_simulationv+0x20c>
  4010c2:	66 0f 2e 1d 76 04 00 	ucomisd 0x476(%rip),%xmm3        # 401540 <_IO_stdin_used+0x20>
  4010c9:	00 
  4010ca:	76 10                	jbe    4010dc <_Z14run_simulationv+0x21c>
  4010cc:	66 0f 57 0d 9c 04 00 	xorpd  0x49c(%rip),%xmm1        # 401570 <_IO_stdin_used+0x50>
  4010d3:	00 
  4010d4:	66 0f 13 88 a0 08 64 	movlpd %xmm1,0x6408a0(%rax)
  4010db:	00 
  4010dc:	f2 0f 10 0d 54 04 00 	movsd  0x454(%rip),%xmm1        # 401538 <_IO_stdin_used+0x18>
  4010e3:	00 
  4010e4:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  4010e8:	77 0a                	ja     4010f4 <_Z14run_simulationv+0x234>
  4010ea:	66 0f 2e 15 4e 04 00 	ucomisd 0x44e(%rip),%xmm2        # 401540 <_IO_stdin_used+0x20>
  4010f1:	00 
  4010f2:	76 10                	jbe    401104 <_Z14run_simulationv+0x244>
  4010f4:	66 0f 57 05 74 04 00 	xorpd  0x474(%rip),%xmm0        # 401570 <_IO_stdin_used+0x50>
  4010fb:	00 
  4010fc:	66 0f 13 80 a0 02 65 	movlpd %xmm0,0x6502a0(%rax)
  401103:	00 
  401104:	48 83 c0 08          	add    $0x8,%rax
  401108:	0f 85 22 ff ff ff    	jne    401030 <_Z14run_simulationv+0x170>
  40110e:	41 ff c6             	inc    %r14d
  401111:	41 83 fe 05          	cmp    $0x5,%r14d
  401115:	0f 85 b5 fd ff ff    	jne    400ed0 <_Z14run_simulationv+0x10>
  40111b:	48 83 c4 20          	add    $0x20,%rsp
  40111f:	5b                   	pop    %rbx
  401120:	41 5e                	pop    %r14
  401122:	41 5f                	pop    %r15
  401124:	c3                   	retq   
  401125:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40112c:	00 00 00 00 

0000000000401130 <_Z10run_simplev>:
  401130:	31 c0                	xor    %eax,%eax
  401132:	66 0f 28 05 46 04 00 	movapd 0x446(%rip),%xmm0        # 401580 <_IO_stdin_used+0x60>
  401139:	00 
  40113a:	66 0f 28 0d 3e 04 00 	movapd 0x43e(%rip),%xmm1        # 401580 <_IO_stdin_used+0x60>
  401141:	00 
  401142:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401149:	1f 84 00 00 00 00 00 
  401150:	31 c9                	xor    %ecx,%ecx
  401152:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401159:	1f 84 00 00 00 00 00 
  401160:	f2 0f 10 14 cd a0 fc 	movsd  0x65fca0(,%rcx,8),%xmm2
  401167:	65 00 
  401169:	66 0f 16 14 cd a0 02 	movhpd 0x6502a0(,%rcx,8),%xmm2
  401170:	65 00 
  401172:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  401176:	f2 0f 10 1c cd a0 20 	movsd  0x6020a0(,%rcx,8),%xmm3
  40117d:	60 00 
  40117f:	0f 16 db             	movlhps %xmm3,%xmm3
  401182:	66 0f 5e d3          	divpd  %xmm3,%xmm2
  401186:	f2 0f 10 24 cd a0 14 	movsd  0x6214a0(,%rcx,8),%xmm4
  40118d:	62 00 
  40118f:	66 0f 16 24 cd a0 1a 	movhpd 0x611aa0(,%rcx,8),%xmm4
  401196:	61 00 
  401198:	f2 0f 10 1c cd a0 08 	movsd  0x6408a0(,%rcx,8),%xmm3
  40119f:	64 00 
  4011a1:	66 0f 16 1c cd a0 0e 	movhpd 0x630ea0(,%rcx,8),%xmm3
  4011a8:	63 00 
  4011aa:	ba 64 00 00 00       	mov    $0x64,%edx
  4011af:	90                   	nop
  4011b0:	66 0f 58 da          	addpd  %xmm2,%xmm3
  4011b4:	66 0f 28 ea          	movapd %xmm2,%xmm5
  4011b8:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  4011bc:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  4011c0:	66 0f 58 dc          	addpd  %xmm4,%xmm3
  4011c4:	66 0f 28 e2          	movapd %xmm2,%xmm4
  4011c8:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  4011cc:	66 0f 59 e9          	mulpd  %xmm1,%xmm5
  4011d0:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  4011d4:	66 0f 28 f2          	movapd %xmm2,%xmm6
  4011d8:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  4011dc:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  4011e0:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  4011e4:	66 0f 28 da          	movapd %xmm2,%xmm3
  4011e8:	66 0f 58 de          	addpd  %xmm6,%xmm3
  4011ec:	66 0f 59 f1          	mulpd  %xmm1,%xmm6
  4011f0:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  4011f4:	66 0f 28 e3          	movapd %xmm3,%xmm4
  4011f8:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  4011fc:	66 0f 58 e6          	addpd  %xmm6,%xmm4
  401200:	83 c2 fb             	add    $0xfffffffb,%edx
  401203:	75 ab                	jne    4011b0 <_Z10run_simplev+0x80>
  401205:	66 0f 17 1c cd a0 0e 	movhpd %xmm3,0x630ea0(,%rcx,8)
  40120c:	63 00 
  40120e:	66 0f 13 1c cd a0 08 	movlpd %xmm3,0x6408a0(,%rcx,8)
  401215:	64 00 
  401217:	66 0f 17 24 cd a0 1a 	movhpd %xmm4,0x611aa0(,%rcx,8)
  40121e:	61 00 
  401220:	66 0f 13 24 cd a0 14 	movlpd %xmm4,0x6214a0(,%rcx,8)
  401227:	62 00 
  401229:	48 ff c1             	inc    %rcx
  40122c:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401233:	0f 85 27 ff ff ff    	jne    401160 <_Z10run_simplev+0x30>
  401239:	ff c0                	inc    %eax
  40123b:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  401240:	0f 85 0a ff ff ff    	jne    401150 <_Z10run_simplev+0x20>
  401246:	c3                   	retq   
  401247:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40124e:	00 00 

0000000000401250 <main>:
  401250:	55                   	push   %rbp
  401251:	41 57                	push   %r15
  401253:	41 56                	push   %r14
  401255:	41 55                	push   %r13
  401257:	41 54                	push   %r12
  401259:	53                   	push   %rbx
  40125a:	50                   	push   %rax
  40125b:	e8 40 f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401260:	48 89 c3             	mov    %rax,%rbx
  401263:	e8 e8 fa ff ff       	callq  400d50 <_Z13instantiationv>
  401268:	e8 33 f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40126d:	49 89 c7             	mov    %rax,%r15
  401270:	49 29 df             	sub    %rbx,%r15
  401273:	e8 28 f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401278:	48 89 c3             	mov    %rax,%rbx
  40127b:	e8 40 fc ff ff       	callq  400ec0 <_Z14run_simulationv>
  401280:	e8 1b f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401285:	49 89 c4             	mov    %rax,%r12
  401288:	49 29 dc             	sub    %rbx,%r12
  40128b:	31 db                	xor    %ebx,%ebx
  40128d:	e8 0e f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401292:	48 89 c5             	mov    %rax,%rbp
  401295:	66 0f 28 05 e3 02 00 	movapd 0x2e3(%rip),%xmm0        # 401580 <_IO_stdin_used+0x60>
  40129c:	00 
  40129d:	66 0f 28 0d db 02 00 	movapd 0x2db(%rip),%xmm1        # 401580 <_IO_stdin_used+0x60>
  4012a4:	00 
  4012a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4012ac:	00 00 00 00 
  4012b0:	31 c0                	xor    %eax,%eax
  4012b2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4012b9:	1f 84 00 00 00 00 00 
  4012c0:	f2 0f 10 14 c5 a0 fc 	movsd  0x65fca0(,%rax,8),%xmm2
  4012c7:	65 00 
  4012c9:	66 0f 16 14 c5 a0 02 	movhpd 0x6502a0(,%rax,8),%xmm2
  4012d0:	65 00 
  4012d2:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  4012d6:	f2 0f 10 1c c5 a0 20 	movsd  0x6020a0(,%rax,8),%xmm3
  4012dd:	60 00 
  4012df:	0f 16 db             	movlhps %xmm3,%xmm3
  4012e2:	66 0f 5e d3          	divpd  %xmm3,%xmm2
  4012e6:	f2 0f 10 24 c5 a0 14 	movsd  0x6214a0(,%rax,8),%xmm4
  4012ed:	62 00 
  4012ef:	66 0f 16 24 c5 a0 1a 	movhpd 0x611aa0(,%rax,8),%xmm4
  4012f6:	61 00 
  4012f8:	f2 0f 10 1c c5 a0 08 	movsd  0x6408a0(,%rax,8),%xmm3
  4012ff:	64 00 
  401301:	66 0f 16 1c c5 a0 0e 	movhpd 0x630ea0(,%rax,8),%xmm3
  401308:	63 00 
  40130a:	b9 64 00 00 00       	mov    $0x64,%ecx
  40130f:	90                   	nop
  401310:	66 0f 58 da          	addpd  %xmm2,%xmm3
  401314:	66 0f 28 ea          	movapd %xmm2,%xmm5
  401318:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  40131c:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  401320:	66 0f 58 dc          	addpd  %xmm4,%xmm3
  401324:	66 0f 28 e2          	movapd %xmm2,%xmm4
  401328:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  40132c:	66 0f 59 e9          	mulpd  %xmm1,%xmm5
  401330:	66 0f 58 eb          	addpd  %xmm3,%xmm5
  401334:	66 0f 28 f2          	movapd %xmm2,%xmm6
  401338:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  40133c:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  401340:	66 0f 58 e5          	addpd  %xmm5,%xmm4
  401344:	66 0f 28 da          	movapd %xmm2,%xmm3
  401348:	66 0f 58 de          	addpd  %xmm6,%xmm3
  40134c:	66 0f 59 f1          	mulpd  %xmm1,%xmm6
  401350:	66 0f 58 f4          	addpd  %xmm4,%xmm6
  401354:	66 0f 28 e3          	movapd %xmm3,%xmm4
  401358:	66 0f 59 e1          	mulpd  %xmm1,%xmm4
  40135c:	66 0f 58 e6          	addpd  %xmm6,%xmm4
  401360:	83 c1 fb             	add    $0xfffffffb,%ecx
  401363:	75 ab                	jne    401310 <main+0xc0>
  401365:	66 0f 17 1c c5 a0 0e 	movhpd %xmm3,0x630ea0(,%rax,8)
  40136c:	63 00 
  40136e:	66 0f 13 1c c5 a0 08 	movlpd %xmm3,0x6408a0(,%rax,8)
  401375:	64 00 
  401377:	66 0f 17 24 c5 a0 1a 	movhpd %xmm4,0x611aa0(,%rax,8)
  40137e:	61 00 
  401380:	66 0f 13 24 c5 a0 14 	movlpd %xmm4,0x6214a0(,%rax,8)
  401387:	62 00 
  401389:	48 ff c0             	inc    %rax
  40138c:	48 3d 40 1f 00 00    	cmp    $0x1f40,%rax
  401392:	0f 85 28 ff ff ff    	jne    4012c0 <main+0x70>
  401398:	ff c3                	inc    %ebx
  40139a:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  4013a0:	0f 85 0a ff ff ff    	jne    4012b0 <main+0x60>
  4013a6:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  4013ad:	de 1b 43 
  4013b0:	4c 89 f8             	mov    %r15,%rax
  4013b3:	48 f7 e9             	imul   %rcx
  4013b6:	49 89 d7             	mov    %rdx,%r15
  4013b9:	4c 89 f8             	mov    %r15,%rax
  4013bc:	48 c1 e8 3f          	shr    $0x3f,%rax
  4013c0:	49 c1 ff 12          	sar    $0x12,%r15
  4013c4:	49 01 c7             	add    %rax,%r15
  4013c7:	4c 89 e0             	mov    %r12,%rax
  4013ca:	48 f7 e9             	imul   %rcx
  4013cd:	49 89 d5             	mov    %rdx,%r13
  4013d0:	4c 89 e8             	mov    %r13,%rax
  4013d3:	48 c1 e8 3f          	shr    $0x3f,%rax
  4013d7:	49 c1 fd 12          	sar    $0x12,%r13
  4013db:	49 01 c5             	add    %rax,%r13
  4013de:	e8 bd f3 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4013e3:	49 89 c4             	mov    %rax,%r12
  4013e6:	49 29 ec             	sub    %rbp,%r12
  4013e9:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4013ee:	31 db                	xor    %ebx,%ebx
  4013f0:	f2 0f 10 04 dd a0 1a 	movsd  0x611aa0(,%rbx,8),%xmm0
  4013f7:	61 00 
  4013f9:	f2 0f 5a c8          	cvtsd2ss %xmm0,%xmm1
  4013fd:	66 0f 7e c8          	movd   %xmm1,%eax
  401401:	01 e8                	add    %ebp,%eax
  401403:	f2 0f 10 0c dd a0 14 	movsd  0x6214a0(,%rbx,8),%xmm1
  40140a:	62 00 
  40140c:	f2 0f 5a d1          	cvtsd2ss %xmm1,%xmm2
  401410:	66 0f 7e d5          	movd   %xmm2,%ebp
  401414:	01 c5                	add    %eax,%ebp
  401416:	48 63 c5             	movslq %ebp,%rax
  401419:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  401420:	48 89 c1             	mov    %rax,%rcx
  401423:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401427:	48 c1 f8 33          	sar    $0x33,%rax
  40142b:	01 c8                	add    %ecx,%eax
  40142d:	44 69 f0 87 d6 12 00 	imul   $0x12d687,%eax,%r14d
  401434:	48 83 fb 09          	cmp    $0x9,%rbx
  401438:	77 0f                	ja     401449 <main+0x1f9>
  40143a:	bf 90 15 40 00       	mov    $0x401590,%edi
  40143f:	b0 02                	mov    $0x2,%al
  401441:	48 89 de             	mov    %rbx,%rsi
  401444:	e8 e7 f2 ff ff       	callq  400730 <printf@plt>
  401449:	44 29 f5             	sub    %r14d,%ebp
  40144c:	48 ff c3             	inc    %rbx
  40144f:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401456:	75 98                	jne    4013f0 <main+0x1a0>
  401458:	4c 89 e0             	mov    %r12,%rax
  40145b:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401462:	de 1b 43 
  401465:	48 f7 e9             	imul   %rcx
  401468:	48 89 d0             	mov    %rdx,%rax
  40146b:	48 c1 e8 3f          	shr    $0x3f,%rax
  40146f:	48 c1 fa 12          	sar    $0x12,%rdx
  401473:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401477:	bf a5 15 40 00       	mov    $0x4015a5,%edi
  40147c:	31 c0                	xor    %eax,%eax
  40147e:	4c 89 fe             	mov    %r15,%rsi
  401481:	4c 89 ea             	mov    %r13,%rdx
  401484:	41 89 e8             	mov    %ebp,%r8d
  401487:	e8 a4 f2 ff ff       	callq  400730 <printf@plt>
  40148c:	31 c0                	xor    %eax,%eax
  40148e:	48 83 c4 08          	add    $0x8,%rsp
  401492:	5b                   	pop    %rbx
  401493:	41 5c                	pop    %r12
  401495:	41 5d                	pop    %r13
  401497:	41 5e                	pop    %r14
  401499:	41 5f                	pop    %r15
  40149b:	5d                   	pop    %rbp
  40149c:	c3                   	retq   
  40149d:	0f 1f 00             	nopl   (%rax)

00000000004014a0 <__libc_csu_init>:
  4014a0:	41 57                	push   %r15
  4014a2:	41 56                	push   %r14
  4014a4:	41 89 ff             	mov    %edi,%r15d
  4014a7:	41 55                	push   %r13
  4014a9:	41 54                	push   %r12
  4014ab:	4c 8d 25 26 09 20 00 	lea    0x200926(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  4014b2:	55                   	push   %rbp
  4014b3:	48 8d 2d 2e 09 20 00 	lea    0x20092e(%rip),%rbp        # 601de8 <__init_array_end>
  4014ba:	53                   	push   %rbx
  4014bb:	49 89 f6             	mov    %rsi,%r14
  4014be:	49 89 d5             	mov    %rdx,%r13
  4014c1:	4c 29 e5             	sub    %r12,%rbp
  4014c4:	48 83 ec 08          	sub    $0x8,%rsp
  4014c8:	48 c1 fd 03          	sar    $0x3,%rbp
  4014cc:	e8 2f f2 ff ff       	callq  400700 <_init>
  4014d1:	48 85 ed             	test   %rbp,%rbp
  4014d4:	74 20                	je     4014f6 <__libc_csu_init+0x56>
  4014d6:	31 db                	xor    %ebx,%ebx
  4014d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4014df:	00 
  4014e0:	4c 89 ea             	mov    %r13,%rdx
  4014e3:	4c 89 f6             	mov    %r14,%rsi
  4014e6:	44 89 ff             	mov    %r15d,%edi
  4014e9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4014ed:	48 83 c3 01          	add    $0x1,%rbx
  4014f1:	48 39 eb             	cmp    %rbp,%rbx
  4014f4:	75 ea                	jne    4014e0 <__libc_csu_init+0x40>
  4014f6:	48 83 c4 08          	add    $0x8,%rsp
  4014fa:	5b                   	pop    %rbx
  4014fb:	5d                   	pop    %rbp
  4014fc:	41 5c                	pop    %r12
  4014fe:	41 5d                	pop    %r13
  401500:	41 5e                	pop    %r14
  401502:	41 5f                	pop    %r15
  401504:	c3                   	retq   
  401505:	90                   	nop
  401506:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40150d:	00 00 00 

0000000000401510 <__libc_csu_fini>:
  401510:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401514 <_fini>:
  401514:	48 83 ec 08          	sub    $0x8,%rsp
  401518:	48 83 c4 08          	add    $0x8,%rsp
  40151c:	c3                   	retq   
