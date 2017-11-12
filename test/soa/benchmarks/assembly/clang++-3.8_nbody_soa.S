
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
  40080f:	49 c7 c0 e0 13 40 00 	mov    $0x4013e0,%r8
  400816:	48 c7 c1 70 13 40 00 	mov    $0x401370,%rcx
  40081d:	48 c7 c7 80 11 40 00 	mov    $0x401180,%rdi
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
  4009e7:	f2 0f 59 0d 09 0a 00 	mulsd  0xa09(%rip),%xmm1        # 4013f8 <_IO_stdin_used+0x8>
  4009ee:	00 
  4009ef:	f2 42 0f 59 0c f5 a0 	mulsd  0x6020a0(,%r14,8),%xmm1
  4009f6:	20 60 00 
  4009f9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4009fd:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400a01:	f2 0f 58 15 f7 09 00 	addsd  0x9f7(%rip),%xmm2        # 401400 <_IO_stdin_used+0x10>
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
  400a69:	f2 0f 10 1d 87 09 00 	movsd  0x987(%rip),%xmm3        # 4013f8 <_IO_stdin_used+0x8>
  400a70:	00 
  400a71:	f2 0f 10 25 87 09 00 	movsd  0x987(%rip),%xmm4        # 401400 <_IO_stdin_used+0x10>
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
  400af6:	f2 0f 10 25 02 09 00 	movsd  0x902(%rip),%xmm4        # 401400 <_IO_stdin_used+0x10>
  400afd:	00 
  400afe:	f2 0f 10 1d f2 08 00 	movsd  0x8f2(%rip),%xmm3        # 4013f8 <_IO_stdin_used+0x8>
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
  400c1f:	f2 0f 10 25 e1 07 00 	movsd  0x7e1(%rip),%xmm4        # 401408 <_IO_stdin_used+0x18>
  400c26:	00 
  400c27:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  400c2b:	77 0a                	ja     400c37 <_Z11Body_updatemd+0x97>
  400c2d:	66 0f 2e 1d db 07 00 	ucomisd 0x7db(%rip),%xmm3        # 401410 <_IO_stdin_used+0x20>
  400c34:	00 
  400c35:	76 11                	jbe    400c48 <_Z11Body_updatemd+0xa8>
  400c37:	66 0f 57 15 01 08 00 	xorpd  0x801(%rip),%xmm2        # 401440 <_IO_stdin_used+0x50>
  400c3e:	00 
  400c3f:	66 0f 13 14 fd a0 0e 	movlpd %xmm2,0x630ea0(,%rdi,8)
  400c46:	63 00 
  400c48:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  400c4c:	77 0a                	ja     400c58 <_Z11Body_updatemd+0xb8>
  400c4e:	66 0f 2e 05 ba 07 00 	ucomisd 0x7ba(%rip),%xmm0        # 401410 <_IO_stdin_used+0x20>
  400c55:	00 
  400c56:	76 11                	jbe    400c69 <_Z11Body_updatemd+0xc9>
  400c58:	66 0f 57 0d e0 07 00 	xorpd  0x7e0(%rip),%xmm1        # 401440 <_IO_stdin_used+0x50>
  400c5f:	00 
  400c60:	66 0f 13 0c fd a0 08 	movlpd %xmm1,0x6408a0(,%rdi,8)
  400c67:	64 00 
  400c69:	c3                   	retq   
  400c6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c70 <_Z26Body_codegen_simple_updatemd>:
  400c70:	f2 0f 10 0c fd a0 02 	movsd  0x6502a0(,%rdi,8),%xmm1
  400c77:	65 00 
  400c79:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400c7d:	f2 0f 10 14 fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm2
  400c84:	60 00 
  400c86:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400c8a:	f2 0f 58 0c fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm1
  400c91:	63 00 
  400c93:	f2 0f 11 0c fd a0 0e 	movsd  %xmm1,0x630ea0(,%rdi,8)
  400c9a:	63 00 
  400c9c:	f2 0f 10 1c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm3
  400ca3:	65 00 
  400ca5:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400ca9:	f2 0f 5e da          	divsd  %xmm2,%xmm3
  400cad:	f2 0f 58 1c fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm3
  400cb4:	64 00 
  400cb6:	f2 0f 11 1c fd a0 08 	movsd  %xmm3,0x6408a0(,%rdi,8)
  400cbd:	64 00 
  400cbf:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400cc3:	f2 0f 58 0c fd a0 1a 	addsd  0x611aa0(,%rdi,8),%xmm1
  400cca:	61 00 
  400ccc:	f2 0f 11 0c fd a0 1a 	movsd  %xmm1,0x611aa0(,%rdi,8)
  400cd3:	61 00 
  400cd5:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400cd9:	f2 0f 58 1c fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm3
  400ce0:	62 00 
  400ce2:	f2 0f 11 1c fd a0 14 	movsd  %xmm3,0x6214a0(,%rdi,8)
  400ce9:	62 00 
  400ceb:	c3                   	retq   
  400cec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cf0 <_Z13instantiationv>:
  400cf0:	53                   	push   %rbx
  400cf1:	48 83 ec 30          	sub    $0x30,%rsp
  400cf5:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400cfa:	e8 91 fa ff ff       	callq  400790 <srand@plt>
  400cff:	48 c7 c3 00 06 ff ff 	mov    $0xffffffffffff0600,%rbx
  400d06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d0d:	00 00 00 
  400d10:	e8 9b fa ff ff       	callq  4007b0 <rand@plt>
  400d15:	0f 57 c9             	xorps  %xmm1,%xmm1
  400d18:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400d1c:	f2 0f 10 05 f4 06 00 	movsd  0x6f4(%rip),%xmm0        # 401418 <_IO_stdin_used+0x28>
  400d23:	00 
  400d24:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400d28:	f2 0f 10 05 f0 06 00 	movsd  0x6f0(%rip),%xmm0        # 401420 <_IO_stdin_used+0x30>
  400d2f:	00 
  400d30:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400d34:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400d38:	f2 0f 10 05 e8 06 00 	movsd  0x6e8(%rip),%xmm0        # 401428 <_IO_stdin_used+0x38>
  400d3f:	00 
  400d40:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400d44:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  400d4a:	e8 61 fa ff ff       	callq  4007b0 <rand@plt>
  400d4f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400d52:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400d56:	f2 0f 5e 0d ba 06 00 	divsd  0x6ba(%rip),%xmm1        # 401418 <_IO_stdin_used+0x28>
  400d5d:	00 
  400d5e:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400d62:	f2 0f 10 05 9e 06 00 	movsd  0x69e(%rip),%xmm0        # 401408 <_IO_stdin_used+0x18>
  400d69:	00 
  400d6a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400d6e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400d74:	e8 37 fa ff ff       	callq  4007b0 <rand@plt>
  400d79:	0f 57 c0             	xorps  %xmm0,%xmm0
  400d7c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400d80:	f2 0f 5e 05 90 06 00 	divsd  0x690(%rip),%xmm0        # 401418 <_IO_stdin_used+0x28>
  400d87:	00 
  400d88:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400d8c:	f2 0f 58 05 74 06 00 	addsd  0x674(%rip),%xmm0        # 401408 <_IO_stdin_used+0x18>
  400d93:	00 
  400d94:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  400d9a:	e8 11 fa ff ff       	callq  4007b0 <rand@plt>
  400d9f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400da2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400da6:	f2 0f 5e 0d 6a 06 00 	divsd  0x66a(%rip),%xmm1        # 401418 <_IO_stdin_used+0x28>
  400dad:	00 
  400dae:	f2 0f 10 05 7a 06 00 	movsd  0x67a(%rip),%xmm0        # 401430 <_IO_stdin_used+0x40>
  400db5:	00 
  400db6:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400dba:	f2 0f 5e 0d 66 06 00 	divsd  0x666(%rip),%xmm1        # 401428 <_IO_stdin_used+0x38>
  400dc1:	00 
  400dc2:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400dc8:	e8 e3 f9 ff ff       	callq  4007b0 <rand@plt>
  400dcd:	0f 57 c0             	xorps  %xmm0,%xmm0
  400dd0:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400dd4:	f2 0f 5e 05 3c 06 00 	divsd  0x63c(%rip),%xmm0        # 401418 <_IO_stdin_used+0x28>
  400ddb:	00 
  400ddc:	f2 0f 58 05 4c 06 00 	addsd  0x64c(%rip),%xmm0        # 401430 <_IO_stdin_used+0x40>
  400de3:	00 
  400de4:	f2 0f 5e 05 3c 06 00 	divsd  0x63c(%rip),%xmm0        # 401428 <_IO_stdin_used+0x38>
  400deb:	00 
  400dec:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  400df2:	f2 0f 11 8b a0 1a 61 	movsd  %xmm1,0x611aa0(%rbx)
  400df9:	00 
  400dfa:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  400e00:	f2 0f 11 8b a0 14 62 	movsd  %xmm1,0x6214a0(%rbx)
  400e07:	00 
  400e08:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  400e0e:	f2 0f 11 8b a0 0e 63 	movsd  %xmm1,0x630ea0(%rbx)
  400e15:	00 
  400e16:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  400e1c:	f2 0f 11 8b a0 08 64 	movsd  %xmm1,0x6408a0(%rbx)
  400e23:	00 
  400e24:	f2 0f 11 83 a0 02 65 	movsd  %xmm0,0x6502a0(%rbx)
  400e2b:	00 
  400e2c:	48 c7 83 a0 fc 65 00 	movq   $0x0,0x65fca0(%rbx)
  400e33:	00 00 00 00 
  400e37:	48 c7 83 a0 f6 66 00 	movq   $0x0,0x66f6a0(%rbx)
  400e3e:	00 00 00 00 
  400e42:	48 83 c3 08          	add    $0x8,%rbx
  400e46:	0f 85 c4 fe ff ff    	jne    400d10 <_Z13instantiationv+0x20>
  400e4c:	48 83 c4 30          	add    $0x30,%rsp
  400e50:	5b                   	pop    %rbx
  400e51:	c3                   	retq   
  400e52:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e59:	1f 84 00 00 00 00 00 

0000000000400e60 <_Z14run_simulationv>:
  400e60:	41 57                	push   %r15
  400e62:	41 56                	push   %r14
  400e64:	53                   	push   %rbx
  400e65:	48 83 ec 20          	sub    $0x20,%rsp
  400e69:	45 31 f6             	xor    %r14d,%r14d
  400e6c:	0f 1f 40 00          	nopl   0x0(%rax)
  400e70:	bf a0 02 65 00       	mov    $0x6502a0,%edi
  400e75:	31 f6                	xor    %esi,%esi
  400e77:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400e7c:	e8 bf f8 ff ff       	callq  400740 <memset@plt>
  400e81:	bf a0 fc 65 00       	mov    $0x65fca0,%edi
  400e86:	31 f6                	xor    %esi,%esi
  400e88:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400e8d:	e8 ae f8 ff ff       	callq  400740 <memset@plt>
  400e92:	45 31 ff             	xor    %r15d,%r15d
  400e95:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400e9c:	00 00 00 00 
  400ea0:	31 db                	xor    %ebx,%ebx
  400ea2:	f2 0f 10 1d 4e 05 00 	movsd  0x54e(%rip),%xmm3        # 4013f8 <_IO_stdin_used+0x8>
  400ea9:	00 
  400eaa:	f2 0f 10 25 4e 05 00 	movsd  0x54e(%rip),%xmm4        # 401400 <_IO_stdin_used+0x10>
  400eb1:	00 
  400eb2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400eb9:	1f 84 00 00 00 00 00 
  400ec0:	49 39 df             	cmp    %rbx,%r15
  400ec3:	0f 84 d3 00 00 00    	je     400f9c <_Z14run_simulationv+0x13c>
  400ec9:	f2 42 0f 10 34 fd a0 	movsd  0x611aa0(,%r15,8),%xmm6
  400ed0:	1a 61 00 
  400ed3:	f2 0f 5c 34 dd a0 1a 	subsd  0x611aa0(,%rbx,8),%xmm6
  400eda:	61 00 
  400edc:	f2 42 0f 10 2c fd a0 	movsd  0x6214a0(,%r15,8),%xmm5
  400ee3:	14 62 00 
  400ee6:	f2 0f 5c 2c dd a0 14 	subsd  0x6214a0(,%rbx,8),%xmm5
  400eed:	62 00 
  400eef:	66 0f 28 c6          	movapd %xmm6,%xmm0
  400ef3:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400ef7:	66 0f 28 cd          	movapd %xmm5,%xmm1
  400efb:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400eff:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400f03:	0f 57 c0             	xorps  %xmm0,%xmm0
  400f06:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400f0a:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400f0e:	7b 31                	jnp    400f41 <_Z14run_simulationv+0xe1>
  400f10:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400f14:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400f1a:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400f20:	e8 9b f8 ff ff       	callq  4007c0 <sqrt@plt>
  400f25:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400f2b:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400f31:	f2 0f 10 25 c7 04 00 	movsd  0x4c7(%rip),%xmm4        # 401400 <_IO_stdin_used+0x10>
  400f38:	00 
  400f39:	f2 0f 10 1d b7 04 00 	movsd  0x4b7(%rip),%xmm3        # 4013f8 <_IO_stdin_used+0x8>
  400f40:	00 
  400f41:	f2 0f 10 0c dd a0 20 	movsd  0x6020a0(,%rbx,8),%xmm1
  400f48:	60 00 
  400f4a:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400f4e:	f2 42 0f 59 0c fd a0 	mulsd  0x6020a0(,%r15,8),%xmm1
  400f55:	20 60 00 
  400f58:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400f5c:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400f60:	f2 0f 58 d4          	addsd  %xmm4,%xmm2
  400f64:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400f68:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  400f6c:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400f70:	f2 0f 58 34 dd a0 02 	addsd  0x6502a0(,%rbx,8),%xmm6
  400f77:	65 00 
  400f79:	f2 0f 11 34 dd a0 02 	movsd  %xmm6,0x6502a0(,%rbx,8)
  400f80:	65 00 
  400f82:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  400f86:	f2 0f 5e e8          	divsd  %xmm0,%xmm5
  400f8a:	f2 0f 58 2c dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm5
  400f91:	65 00 
  400f93:	f2 0f 11 2c dd a0 fc 	movsd  %xmm5,0x65fca0(,%rbx,8)
  400f9a:	65 00 
  400f9c:	48 ff c3             	inc    %rbx
  400f9f:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  400fa6:	0f 85 14 ff ff ff    	jne    400ec0 <_Z14run_simulationv+0x60>
  400fac:	49 ff c7             	inc    %r15
  400faf:	48 c7 c0 00 06 ff ff 	mov    $0xffffffffffff0600,%rax
  400fb6:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400fbd:	0f 85 dd fe ff ff    	jne    400ea0 <_Z14run_simulationv+0x40>
  400fc3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400fca:	84 00 00 00 00 00 
  400fd0:	f2 0f 10 88 a0 fc 65 	movsd  0x65fca0(%rax),%xmm1
  400fd7:	00 
  400fd8:	f2 0f 10 05 40 04 00 	movsd  0x440(%rip),%xmm0        # 401420 <_IO_stdin_used+0x30>
  400fdf:	00 
  400fe0:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400fe4:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400fe8:	f2 0f 10 90 a0 1a 61 	movsd  0x611aa0(%rax),%xmm2
  400fef:	00 
  400ff0:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400ff4:	f2 0f 58 88 a0 08 64 	addsd  0x6408a0(%rax),%xmm1
  400ffb:	00 
  400ffc:	f2 0f 11 88 a0 08 64 	movsd  %xmm1,0x6408a0(%rax)
  401003:	00 
  401004:	f2 0f 10 80 a0 f6 66 	movsd  0x66f6a0(%rax),%xmm0
  40100b:	00 
  40100c:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  401010:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  401014:	f2 0f 58 80 a0 02 65 	addsd  0x6502a0(%rax),%xmm0
  40101b:	00 
  40101c:	f2 0f 11 80 a0 02 65 	movsd  %xmm0,0x6502a0(%rax)
  401023:	00 
  401024:	66 0f 28 d9          	movapd %xmm1,%xmm3
  401028:	f2 0f 59 dc          	mulsd  %xmm4,%xmm3
  40102c:	f2 0f 58 98 a0 14 62 	addsd  0x6214a0(%rax),%xmm3
  401033:	00 
  401034:	f2 0f 11 98 a0 14 62 	movsd  %xmm3,0x6214a0(%rax)
  40103b:	00 
  40103c:	66 0f 28 d0          	movapd %xmm0,%xmm2
  401040:	f2 0f 59 d4          	mulsd  %xmm4,%xmm2
  401044:	f2 0f 58 90 a0 0e 63 	addsd  0x630ea0(%rax),%xmm2
  40104b:	00 
  40104c:	f2 0f 11 90 a0 0e 63 	movsd  %xmm2,0x630ea0(%rax)
  401053:	00 
  401054:	f2 0f 10 25 ac 03 00 	movsd  0x3ac(%rip),%xmm4        # 401408 <_IO_stdin_used+0x18>
  40105b:	00 
  40105c:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  401060:	77 0a                	ja     40106c <_Z14run_simulationv+0x20c>
  401062:	66 0f 2e 1d a6 03 00 	ucomisd 0x3a6(%rip),%xmm3        # 401410 <_IO_stdin_used+0x20>
  401069:	00 
  40106a:	76 10                	jbe    40107c <_Z14run_simulationv+0x21c>
  40106c:	66 0f 57 0d cc 03 00 	xorpd  0x3cc(%rip),%xmm1        # 401440 <_IO_stdin_used+0x50>
  401073:	00 
  401074:	66 0f 13 88 a0 08 64 	movlpd %xmm1,0x6408a0(%rax)
  40107b:	00 
  40107c:	f2 0f 10 0d 84 03 00 	movsd  0x384(%rip),%xmm1        # 401408 <_IO_stdin_used+0x18>
  401083:	00 
  401084:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  401088:	77 0a                	ja     401094 <_Z14run_simulationv+0x234>
  40108a:	66 0f 2e 15 7e 03 00 	ucomisd 0x37e(%rip),%xmm2        # 401410 <_IO_stdin_used+0x20>
  401091:	00 
  401092:	76 10                	jbe    4010a4 <_Z14run_simulationv+0x244>
  401094:	66 0f 57 05 a4 03 00 	xorpd  0x3a4(%rip),%xmm0        # 401440 <_IO_stdin_used+0x50>
  40109b:	00 
  40109c:	66 0f 13 80 a0 02 65 	movlpd %xmm0,0x6502a0(%rax)
  4010a3:	00 
  4010a4:	48 83 c0 08          	add    $0x8,%rax
  4010a8:	0f 85 22 ff ff ff    	jne    400fd0 <_Z14run_simulationv+0x170>
  4010ae:	41 ff c6             	inc    %r14d
  4010b1:	41 83 fe 05          	cmp    $0x5,%r14d
  4010b5:	0f 85 b5 fd ff ff    	jne    400e70 <_Z14run_simulationv+0x10>
  4010bb:	48 83 c4 20          	add    $0x20,%rsp
  4010bf:	5b                   	pop    %rbx
  4010c0:	41 5e                	pop    %r14
  4010c2:	41 5f                	pop    %r15
  4010c4:	c3                   	retq   
  4010c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4010cc:	00 00 00 00 

00000000004010d0 <_Z10run_simplev>:
  4010d0:	31 c0                	xor    %eax,%eax
  4010d2:	66 0f 28 05 76 03 00 	movapd 0x376(%rip),%xmm0        # 401450 <_IO_stdin_used+0x60>
  4010d9:	00 
  4010da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4010e0:	31 c9                	xor    %ecx,%ecx
  4010e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4010e9:	1f 84 00 00 00 00 00 
  4010f0:	66 0f 28 89 a0 02 65 	movapd 0x6502a0(%rcx),%xmm1
  4010f7:	00 
  4010f8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4010fc:	66 0f 28 91 a0 20 60 	movapd 0x6020a0(%rcx),%xmm2
  401103:	00 
  401104:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  401108:	66 0f 58 89 a0 0e 63 	addpd  0x630ea0(%rcx),%xmm1
  40110f:	00 
  401110:	66 0f 29 89 a0 0e 63 	movapd %xmm1,0x630ea0(%rcx)
  401117:	00 
  401118:	66 0f 28 99 a0 fc 65 	movapd 0x65fca0(%rcx),%xmm3
  40111f:	00 
  401120:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  401124:	66 0f 5e da          	divpd  %xmm2,%xmm3
  401128:	66 0f 58 99 a0 08 64 	addpd  0x6408a0(%rcx),%xmm3
  40112f:	00 
  401130:	66 0f 29 99 a0 08 64 	movapd %xmm3,0x6408a0(%rcx)
  401137:	00 
  401138:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  40113c:	66 0f 58 89 a0 1a 61 	addpd  0x611aa0(%rcx),%xmm1
  401143:	00 
  401144:	66 0f 29 89 a0 1a 61 	movapd %xmm1,0x611aa0(%rcx)
  40114b:	00 
  40114c:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  401150:	66 0f 58 99 a0 14 62 	addpd  0x6214a0(%rcx),%xmm3
  401157:	00 
  401158:	66 0f 29 99 a0 14 62 	movapd %xmm3,0x6214a0(%rcx)
  40115f:	00 
  401160:	48 83 c1 10          	add    $0x10,%rcx
  401164:	48 81 f9 00 fa 00 00 	cmp    $0xfa00,%rcx
  40116b:	75 83                	jne    4010f0 <_Z10run_simplev+0x20>
  40116d:	ff c0                	inc    %eax
  40116f:	3d 50 c3 00 00       	cmp    $0xc350,%eax
  401174:	0f 85 66 ff ff ff    	jne    4010e0 <_Z10run_simplev+0x10>
  40117a:	c3                   	retq   
  40117b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401180 <main>:
  401180:	55                   	push   %rbp
  401181:	41 57                	push   %r15
  401183:	41 56                	push   %r14
  401185:	41 55                	push   %r13
  401187:	41 54                	push   %r12
  401189:	53                   	push   %rbx
  40118a:	50                   	push   %rax
  40118b:	e8 10 f6 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401190:	48 89 c5             	mov    %rax,%rbp
  401193:	e8 58 fb ff ff       	callq  400cf0 <_Z13instantiationv>
  401198:	e8 03 f6 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40119d:	49 89 c6             	mov    %rax,%r14
  4011a0:	49 29 ee             	sub    %rbp,%r14
  4011a3:	e8 f8 f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4011a8:	48 89 c3             	mov    %rax,%rbx
  4011ab:	e8 b0 fc ff ff       	callq  400e60 <_Z14run_simulationv>
  4011b0:	e8 eb f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4011b5:	48 89 c5             	mov    %rax,%rbp
  4011b8:	48 29 dd             	sub    %rbx,%rbp
  4011bb:	31 db                	xor    %ebx,%ebx
  4011bd:	e8 de f5 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4011c2:	49 89 c7             	mov    %rax,%r15
  4011c5:	66 0f 28 05 83 02 00 	movapd 0x283(%rip),%xmm0        # 401450 <_IO_stdin_used+0x60>
  4011cc:	00 
  4011cd:	0f 1f 00             	nopl   (%rax)
  4011d0:	31 c0                	xor    %eax,%eax
  4011d2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4011d9:	1f 84 00 00 00 00 00 
  4011e0:	66 0f 28 88 a0 02 65 	movapd 0x6502a0(%rax),%xmm1
  4011e7:	00 
  4011e8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4011ec:	66 0f 28 90 a0 20 60 	movapd 0x6020a0(%rax),%xmm2
  4011f3:	00 
  4011f4:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  4011f8:	66 0f 58 88 a0 0e 63 	addpd  0x630ea0(%rax),%xmm1
  4011ff:	00 
  401200:	66 0f 29 88 a0 0e 63 	movapd %xmm1,0x630ea0(%rax)
  401207:	00 
  401208:	66 0f 28 98 a0 fc 65 	movapd 0x65fca0(%rax),%xmm3
  40120f:	00 
  401210:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  401214:	66 0f 5e da          	divpd  %xmm2,%xmm3
  401218:	66 0f 58 98 a0 08 64 	addpd  0x6408a0(%rax),%xmm3
  40121f:	00 
  401220:	66 0f 29 98 a0 08 64 	movapd %xmm3,0x6408a0(%rax)
  401227:	00 
  401228:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  40122c:	66 0f 58 88 a0 1a 61 	addpd  0x611aa0(%rax),%xmm1
  401233:	00 
  401234:	66 0f 29 88 a0 1a 61 	movapd %xmm1,0x611aa0(%rax)
  40123b:	00 
  40123c:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  401240:	66 0f 58 98 a0 14 62 	addpd  0x6214a0(%rax),%xmm3
  401247:	00 
  401248:	66 0f 29 98 a0 14 62 	movapd %xmm3,0x6214a0(%rax)
  40124f:	00 
  401250:	48 83 c0 10          	add    $0x10,%rax
  401254:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  40125a:	75 84                	jne    4011e0 <main+0x60>
  40125c:	ff c3                	inc    %ebx
  40125e:	81 fb 50 c3 00 00    	cmp    $0xc350,%ebx
  401264:	0f 85 66 ff ff ff    	jne    4011d0 <main+0x50>
  40126a:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401271:	de 1b 43 
  401274:	4c 89 f0             	mov    %r14,%rax
  401277:	48 f7 e9             	imul   %rcx
  40127a:	49 89 d6             	mov    %rdx,%r14
  40127d:	4c 89 f0             	mov    %r14,%rax
  401280:	48 c1 e8 3f          	shr    $0x3f,%rax
  401284:	49 c1 fe 12          	sar    $0x12,%r14
  401288:	49 01 c6             	add    %rax,%r14
  40128b:	48 89 e8             	mov    %rbp,%rax
  40128e:	48 f7 e9             	imul   %rcx
  401291:	49 89 d5             	mov    %rdx,%r13
  401294:	4c 89 e8             	mov    %r13,%rax
  401297:	48 c1 e8 3f          	shr    $0x3f,%rax
  40129b:	49 c1 fd 12          	sar    $0x12,%r13
  40129f:	49 01 c5             	add    %rax,%r13
  4012a2:	e8 f9 f4 ff ff       	callq  4007a0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012a7:	49 89 c4             	mov    %rax,%r12
  4012aa:	4d 29 fc             	sub    %r15,%r12
  4012ad:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4012b2:	31 db                	xor    %ebx,%ebx
  4012b4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4012bb:	00 00 00 00 00 
  4012c0:	f2 0f 10 04 dd a0 1a 	movsd  0x611aa0(,%rbx,8),%xmm0
  4012c7:	61 00 
  4012c9:	f2 0f 5a c8          	cvtsd2ss %xmm0,%xmm1
  4012cd:	66 0f 7e c8          	movd   %xmm1,%eax
  4012d1:	01 e8                	add    %ebp,%eax
  4012d3:	f2 0f 10 0c dd a0 14 	movsd  0x6214a0(,%rbx,8),%xmm1
  4012da:	62 00 
  4012dc:	f2 0f 5a d1          	cvtsd2ss %xmm1,%xmm2
  4012e0:	66 0f 7e d5          	movd   %xmm2,%ebp
  4012e4:	01 c5                	add    %eax,%ebp
  4012e6:	48 63 c5             	movslq %ebp,%rax
  4012e9:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  4012f0:	48 89 c1             	mov    %rax,%rcx
  4012f3:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4012f7:	48 c1 f8 33          	sar    $0x33,%rax
  4012fb:	01 c8                	add    %ecx,%eax
  4012fd:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  401304:	48 83 fb 09          	cmp    $0x9,%rbx
  401308:	77 0f                	ja     401319 <main+0x199>
  40130a:	bf 60 14 40 00       	mov    $0x401460,%edi
  40130f:	b0 02                	mov    $0x2,%al
  401311:	48 89 de             	mov    %rbx,%rsi
  401314:	e8 17 f4 ff ff       	callq  400730 <printf@plt>
  401319:	44 29 fd             	sub    %r15d,%ebp
  40131c:	48 ff c3             	inc    %rbx
  40131f:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401326:	75 98                	jne    4012c0 <main+0x140>
  401328:	4c 89 e0             	mov    %r12,%rax
  40132b:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401332:	de 1b 43 
  401335:	48 f7 e9             	imul   %rcx
  401338:	48 89 d0             	mov    %rdx,%rax
  40133b:	48 c1 e8 3f          	shr    $0x3f,%rax
  40133f:	48 c1 fa 12          	sar    $0x12,%rdx
  401343:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401347:	bf 75 14 40 00       	mov    $0x401475,%edi
  40134c:	31 c0                	xor    %eax,%eax
  40134e:	4c 89 f6             	mov    %r14,%rsi
  401351:	4c 89 ea             	mov    %r13,%rdx
  401354:	41 89 e8             	mov    %ebp,%r8d
  401357:	e8 d4 f3 ff ff       	callq  400730 <printf@plt>
  40135c:	31 c0                	xor    %eax,%eax
  40135e:	48 83 c4 08          	add    $0x8,%rsp
  401362:	5b                   	pop    %rbx
  401363:	41 5c                	pop    %r12
  401365:	41 5d                	pop    %r13
  401367:	41 5e                	pop    %r14
  401369:	41 5f                	pop    %r15
  40136b:	5d                   	pop    %rbp
  40136c:	c3                   	retq   
  40136d:	0f 1f 00             	nopl   (%rax)

0000000000401370 <__libc_csu_init>:
  401370:	41 57                	push   %r15
  401372:	41 56                	push   %r14
  401374:	41 89 ff             	mov    %edi,%r15d
  401377:	41 55                	push   %r13
  401379:	41 54                	push   %r12
  40137b:	4c 8d 25 56 0a 20 00 	lea    0x200a56(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  401382:	55                   	push   %rbp
  401383:	48 8d 2d 5e 0a 20 00 	lea    0x200a5e(%rip),%rbp        # 601de8 <__init_array_end>
  40138a:	53                   	push   %rbx
  40138b:	49 89 f6             	mov    %rsi,%r14
  40138e:	49 89 d5             	mov    %rdx,%r13
  401391:	4c 29 e5             	sub    %r12,%rbp
  401394:	48 83 ec 08          	sub    $0x8,%rsp
  401398:	48 c1 fd 03          	sar    $0x3,%rbp
  40139c:	e8 5f f3 ff ff       	callq  400700 <_init>
  4013a1:	48 85 ed             	test   %rbp,%rbp
  4013a4:	74 20                	je     4013c6 <__libc_csu_init+0x56>
  4013a6:	31 db                	xor    %ebx,%ebx
  4013a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4013af:	00 
  4013b0:	4c 89 ea             	mov    %r13,%rdx
  4013b3:	4c 89 f6             	mov    %r14,%rsi
  4013b6:	44 89 ff             	mov    %r15d,%edi
  4013b9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4013bd:	48 83 c3 01          	add    $0x1,%rbx
  4013c1:	48 39 eb             	cmp    %rbp,%rbx
  4013c4:	75 ea                	jne    4013b0 <__libc_csu_init+0x40>
  4013c6:	48 83 c4 08          	add    $0x8,%rsp
  4013ca:	5b                   	pop    %rbx
  4013cb:	5d                   	pop    %rbp
  4013cc:	41 5c                	pop    %r12
  4013ce:	41 5d                	pop    %r13
  4013d0:	41 5e                	pop    %r14
  4013d2:	41 5f                	pop    %r15
  4013d4:	c3                   	retq   
  4013d5:	90                   	nop
  4013d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4013dd:	00 00 00 

00000000004013e0 <__libc_csu_fini>:
  4013e0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004013e4 <_fini>:
  4013e4:	48 83 ec 08          	sub    $0x8,%rsp
  4013e8:	48 83 c4 08          	add    $0x8,%rsp
  4013ec:	c3                   	retq   
