
bin/g++_nbody_ikracpp:     file format elf64-x86-64


Disassembly of section .init:

0000000000400718 <_init>:
  400718:	48 83 ec 08          	sub    $0x8,%rsp
  40071c:	48 8b 05 d5 18 20 00 	mov    0x2018d5(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  400723:	48 85 c0             	test   %rax,%rax
  400726:	74 05                	je     40072d <_init+0x15>
  400728:	e8 c3 00 00 00       	callq  4007f0 <sqrt@plt+0x10>
  40072d:	48 83 c4 08          	add    $0x8,%rsp
  400731:	c3                   	retq   

Disassembly of section .plt:

0000000000400740 <memset@plt-0x10>:
  400740:	ff 35 c2 18 20 00    	pushq  0x2018c2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400746:	ff 25 c4 18 20 00    	jmpq   *0x2018c4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40074c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400750 <memset@plt>:
  400750:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400756:	68 00 00 00 00       	pushq  $0x0
  40075b:	e9 e0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400760 <__printf_chk@plt>:
  400760:	ff 25 ba 18 20 00    	jmpq   *0x2018ba(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400766:	68 01 00 00 00       	pushq  $0x1
  40076b:	e9 d0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400770 <_ZNSt8ios_base4InitC1Ev@plt>:
  400770:	ff 25 b2 18 20 00    	jmpq   *0x2018b2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400776:	68 02 00 00 00       	pushq  $0x2
  40077b:	e9 c0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400780 <__libc_start_main@plt>:
  400780:	ff 25 aa 18 20 00    	jmpq   *0x2018aa(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400786:	68 03 00 00 00       	pushq  $0x3
  40078b:	e9 b0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400790 <__cxa_atexit@plt>:
  400790:	ff 25 a2 18 20 00    	jmpq   *0x2018a2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400796:	68 04 00 00 00       	pushq  $0x4
  40079b:	e9 a0 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007a0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4007a0:	ff 25 9a 18 20 00    	jmpq   *0x20189a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4007a6:	68 05 00 00 00       	pushq  $0x5
  4007ab:	e9 90 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007b0 <srand@plt>:
  4007b0:	ff 25 92 18 20 00    	jmpq   *0x201892(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  4007b6:	68 06 00 00 00       	pushq  $0x6
  4007bb:	e9 80 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  4007c0:	ff 25 8a 18 20 00    	jmpq   *0x20188a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  4007c6:	68 07 00 00 00       	pushq  $0x7
  4007cb:	e9 70 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007d0 <rand@plt>:
  4007d0:	ff 25 82 18 20 00    	jmpq   *0x201882(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  4007d6:	68 08 00 00 00       	pushq  $0x8
  4007db:	e9 60 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007e0 <sqrt@plt>:
  4007e0:	ff 25 7a 18 20 00    	jmpq   *0x20187a(%rip)        # 602060 <_GLOBAL_OFFSET_TABLE_+0x60>
  4007e6:	68 09 00 00 00       	pushq  $0x9
  4007eb:	e9 50 ff ff ff       	jmpq   400740 <_init+0x28>

Disassembly of section .plt.got:

00000000004007f0 <.plt.got>:
  4007f0:	ff 25 02 18 20 00    	jmpq   *0x201802(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  4007f6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400800 <main>:
  400800:	41 56                	push   %r14
  400802:	41 55                	push   %r13
  400804:	bf 50 0a 40 00       	mov    $0x400a50,%edi
  400809:	41 54                	push   %r12
  40080b:	55                   	push   %rbp
  40080c:	41 bd 98 1a 61 00    	mov    $0x611a98,%r13d
  400812:	53                   	push   %rbx
  400813:	45 31 e4             	xor    %r12d,%r12d
  400816:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  40081c:	48 83 ec 10          	sub    $0x10,%rsp
  400820:	48 c7 05 55 18 20 00 	movq   $0x0,0x201855(%rip)        # 602080 <__Bodydata_buffer>
  400827:	00 00 00 00 
  40082b:	48 c7 05 52 18 20 00 	movq   $0x0,0x201852(%rip)        # 602088 <__Bodydata_buffer+0x8>
  400832:	00 00 00 00 
  400836:	e8 75 0a 00 00       	callq  4012b0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40083b:	bf 90 0c 40 00       	mov    $0x400c90,%edi
  400840:	48 89 c3             	mov    %rax,%rbx
  400843:	e8 68 0a 00 00       	callq  4012b0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400848:	bf c0 0b 40 00       	mov    $0x400bc0,%edi
  40084d:	48 89 c5             	mov    %rax,%rbp
  400850:	e8 5b 0a 00 00       	callq  4012b0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400855:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  40085b:	49 89 c0             	mov    %rax,%r8
  40085e:	66 90                	xchg   %ax,%ax
  400860:	66 0f ef d2          	pxor   %xmm2,%xmm2
  400864:	f2 41 0f 10 45 00    	movsd  0x0(%r13),%xmm0
  40086a:	66 0f ef db          	pxor   %xmm3,%xmm3
  40086e:	f2 0f 5a d0          	cvtsd2ss %xmm0,%xmm2
  400872:	f2 41 0f 10 8d 00 fa 	movsd  0xfa00(%r13),%xmm1
  400879:	00 00 
  40087b:	f2 0f 5a d9          	cvtsd2ss %xmm1,%xmm3
  40087f:	66 0f 7e d6          	movd   %xmm2,%esi
  400883:	41 01 f1             	add    %esi,%r9d
  400886:	66 0f 7e de          	movd   %xmm3,%esi
  40088a:	44 01 ce             	add    %r9d,%esi
  40088d:	89 f0                	mov    %esi,%eax
  40088f:	41 f7 ee             	imul   %r14d
  400892:	89 f0                	mov    %esi,%eax
  400894:	c1 f8 1f             	sar    $0x1f,%eax
  400897:	c1 fa 13             	sar    $0x13,%edx
  40089a:	41 89 d1             	mov    %edx,%r9d
  40089d:	41 29 c1             	sub    %eax,%r9d
  4008a0:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  4008a7:	44 29 ce             	sub    %r9d,%esi
  4008aa:	49 83 fc 09          	cmp    $0x9,%r12
  4008ae:	41 89 f1             	mov    %esi,%r9d
  4008b1:	76 37                	jbe    4008ea <main+0xea>
  4008b3:	49 83 c4 01          	add    $0x1,%r12
  4008b7:	49 83 c5 08          	add    $0x8,%r13
  4008bb:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  4008c2:	75 9c                	jne    400860 <main+0x60>
  4008c4:	48 89 e9             	mov    %rbp,%rcx
  4008c7:	48 89 da             	mov    %rbx,%rdx
  4008ca:	be 90 13 40 00       	mov    $0x401390,%esi
  4008cf:	bf 01 00 00 00       	mov    $0x1,%edi
  4008d4:	31 c0                	xor    %eax,%eax
  4008d6:	e8 85 fe ff ff       	callq  400760 <__printf_chk@plt>
  4008db:	48 83 c4 10          	add    $0x10,%rsp
  4008df:	31 c0                	xor    %eax,%eax
  4008e1:	5b                   	pop    %rbx
  4008e2:	5d                   	pop    %rbp
  4008e3:	41 5c                	pop    %r12
  4008e5:	41 5d                	pop    %r13
  4008e7:	41 5e                	pop    %r14
  4008e9:	c3                   	retq   
  4008ea:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  4008ee:	4c 89 e2             	mov    %r12,%rdx
  4008f1:	be 74 13 40 00       	mov    $0x401374,%esi
  4008f6:	bf 01 00 00 00       	mov    $0x1,%edi
  4008fb:	b8 02 00 00 00       	mov    $0x2,%eax
  400900:	4c 89 04 24          	mov    %r8,(%rsp)
  400904:	e8 57 fe ff ff       	callq  400760 <__printf_chk@plt>
  400909:	4c 8b 04 24          	mov    (%rsp),%r8
  40090d:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  400912:	eb 9f                	jmp    4008b3 <main+0xb3>
  400914:	66 90                	xchg   %ax,%ax
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_GLOBAL__sub_I__Z11r_float2intf>:
  400920:	48 83 ec 08          	sub    $0x8,%rsp
  400924:	bf c8 f6 66 00       	mov    $0x66f6c8,%edi
  400929:	e8 42 fe ff ff       	callq  400770 <_ZNSt8ios_base4InitC1Ev@plt>
  40092e:	ba 70 20 60 00       	mov    $0x602070,%edx
  400933:	be c8 f6 66 00       	mov    $0x66f6c8,%esi
  400938:	bf a0 07 40 00       	mov    $0x4007a0,%edi
  40093d:	48 83 c4 08          	add    $0x8,%rsp
  400941:	e9 4a fe ff ff       	jmpq   400790 <__cxa_atexit@plt>
  400946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094d:	00 00 00 

0000000000400950 <_start>:
  400950:	31 ed                	xor    %ebp,%ebp
  400952:	49 89 d1             	mov    %rdx,%r9
  400955:	5e                   	pop    %rsi
  400956:	48 89 e2             	mov    %rsp,%rdx
  400959:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40095d:	50                   	push   %rax
  40095e:	54                   	push   %rsp
  40095f:	49 c7 c0 60 13 40 00 	mov    $0x401360,%r8
  400966:	48 c7 c1 f0 12 40 00 	mov    $0x4012f0,%rcx
  40096d:	48 c7 c7 00 08 40 00 	mov    $0x400800,%rdi
  400974:	e8 07 fe ff ff       	callq  400780 <__libc_start_main@plt>
  400979:	f4                   	hlt    
  40097a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400980 <deregister_tm_clones>:
  400980:	b8 7f 20 60 00       	mov    $0x60207f,%eax
  400985:	55                   	push   %rbp
  400986:	48 2d 78 20 60 00    	sub    $0x602078,%rax
  40098c:	48 83 f8 0e          	cmp    $0xe,%rax
  400990:	48 89 e5             	mov    %rsp,%rbp
  400993:	76 1b                	jbe    4009b0 <deregister_tm_clones+0x30>
  400995:	b8 00 00 00 00       	mov    $0x0,%eax
  40099a:	48 85 c0             	test   %rax,%rax
  40099d:	74 11                	je     4009b0 <deregister_tm_clones+0x30>
  40099f:	5d                   	pop    %rbp
  4009a0:	bf 78 20 60 00       	mov    $0x602078,%edi
  4009a5:	ff e0                	jmpq   *%rax
  4009a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009ae:	00 00 
  4009b0:	5d                   	pop    %rbp
  4009b1:	c3                   	retq   
  4009b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4009b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009bd:	00 00 00 

00000000004009c0 <register_tm_clones>:
  4009c0:	be 78 20 60 00       	mov    $0x602078,%esi
  4009c5:	55                   	push   %rbp
  4009c6:	48 81 ee 78 20 60 00 	sub    $0x602078,%rsi
  4009cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 89 f0             	mov    %rsi,%rax
  4009d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4009db:	48 01 c6             	add    %rax,%rsi
  4009de:	48 d1 fe             	sar    %rsi
  4009e1:	74 15                	je     4009f8 <register_tm_clones+0x38>
  4009e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4009e8:	48 85 c0             	test   %rax,%rax
  4009eb:	74 0b                	je     4009f8 <register_tm_clones+0x38>
  4009ed:	5d                   	pop    %rbp
  4009ee:	bf 78 20 60 00       	mov    $0x602078,%edi
  4009f3:	ff e0                	jmpq   *%rax
  4009f5:	0f 1f 00             	nopl   (%rax)
  4009f8:	5d                   	pop    %rbp
  4009f9:	c3                   	retq   
  4009fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a00 <__do_global_dtors_aux>:
  400a00:	80 3d 71 16 20 00 00 	cmpb   $0x0,0x201671(%rip)        # 602078 <__TMC_END__>
  400a07:	75 11                	jne    400a1a <__do_global_dtors_aux+0x1a>
  400a09:	55                   	push   %rbp
  400a0a:	48 89 e5             	mov    %rsp,%rbp
  400a0d:	e8 6e ff ff ff       	callq  400980 <deregister_tm_clones>
  400a12:	5d                   	pop    %rbp
  400a13:	c6 05 5e 16 20 00 01 	movb   $0x1,0x20165e(%rip)        # 602078 <__TMC_END__>
  400a1a:	f3 c3                	repz retq 
  400a1c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a20 <frame_dummy>:
  400a20:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400a25:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400a29:	75 05                	jne    400a30 <frame_dummy+0x10>
  400a2b:	eb 93                	jmp    4009c0 <register_tm_clones>
  400a2d:	0f 1f 00             	nopl   (%rax)
  400a30:	b8 00 00 00 00       	mov    $0x0,%eax
  400a35:	48 85 c0             	test   %rax,%rax
  400a38:	74 f1                	je     400a2b <frame_dummy+0xb>
  400a3a:	55                   	push   %rbp
  400a3b:	48 89 e5             	mov    %rsp,%rbp
  400a3e:	ff d0                	callq  *%rax
  400a40:	5d                   	pop    %rbp
  400a41:	e9 7a ff ff ff       	jmpq   4009c0 <register_tm_clones>
  400a46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4d:	00 00 00 

0000000000400a50 <_Z13instantiationv>:
  400a50:	41 56                	push   %r14
  400a52:	41 55                	push   %r13
  400a54:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400a59:	41 54                	push   %r12
  400a5b:	55                   	push   %rbp
  400a5c:	41 bd 40 1f 00 00    	mov    $0x1f40,%r13d
  400a62:	53                   	push   %rbx
  400a63:	48 83 ec 20          	sub    $0x20,%rsp
  400a67:	e8 44 fd ff ff       	callq  4007b0 <srand@plt>
  400a6c:	f2 0f 10 0d 5c 09 00 	movsd  0x95c(%rip),%xmm1        # 4013d0 <_IO_stdin_used+0x60>
  400a73:	00 
  400a74:	f2 0f 10 15 5c 09 00 	movsd  0x95c(%rip),%xmm2        # 4013d8 <_IO_stdin_used+0x68>
  400a7b:	00 
  400a7c:	f2 0f 10 1d 5c 09 00 	movsd  0x95c(%rip),%xmm3        # 4013e0 <_IO_stdin_used+0x70>
  400a83:	00 
  400a84:	0f 1f 40 00          	nopl   0x0(%rax)
  400a88:	f2 0f 11 5c 24 18    	movsd  %xmm3,0x18(%rsp)
  400a8e:	f2 0f 11 54 24 10    	movsd  %xmm2,0x10(%rsp)
  400a94:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
  400a9a:	e8 31 fd ff ff       	callq  4007d0 <rand@plt>
  400a9f:	89 c3                	mov    %eax,%ebx
  400aa1:	e8 2a fd ff ff       	callq  4007d0 <rand@plt>
  400aa6:	41 89 c4             	mov    %eax,%r12d
  400aa9:	e8 22 fd ff ff       	callq  4007d0 <rand@plt>
  400aae:	89 c5                	mov    %eax,%ebp
  400ab0:	e8 1b fd ff ff       	callq  4007d0 <rand@plt>
  400ab5:	41 89 c6             	mov    %eax,%r14d
  400ab8:	e8 13 fd ff ff       	callq  4007d0 <rand@plt>
  400abd:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400ac1:	48 8b 15 b8 15 20 00 	mov    0x2015b8(%rip),%rdx        # 602080 <__Bodydata_buffer>
  400ac8:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  400ace:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  400ad4:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  400ad8:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400adc:	48 83 c2 03          	add    $0x3,%rdx
  400ae0:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  400ae6:	41 83 ed 01          	sub    $0x1,%r13d
  400aea:	48 89 0d 8f 15 20 00 	mov    %rcx,0x20158f(%rip)        # 602080 <__Bodydata_buffer>
  400af1:	48 c7 04 d5 80 02 65 	movq   $0x0,0x650280(,%rdx,8)
  400af8:	00 00 00 00 00 
  400afd:	48 c7 04 d5 80 fc 65 	movq   $0x0,0x65fc80(,%rdx,8)
  400b04:	00 00 00 00 00 
  400b09:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b0d:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400b11:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400b15:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400b19:	f2 0f 11 04 d5 80 20 	movsd  %xmm0,0x602080(,%rdx,8)
  400b20:	60 00 
  400b22:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b26:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400b2b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b2f:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b33:	f2 0f 5c 05 ad 08 00 	subsd  0x8ad(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x78>
  400b3a:	00 
  400b3b:	f2 0f 11 04 d5 80 1a 	movsd  %xmm0,0x611a80(,%rdx,8)
  400b42:	61 00 
  400b44:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b48:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400b4c:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b50:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b54:	f2 0f 5c 05 8c 08 00 	subsd  0x88c(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x78>
  400b5b:	00 
  400b5c:	f2 0f 11 04 d5 80 14 	movsd  %xmm0,0x621480(,%rdx,8)
  400b63:	62 00 
  400b65:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b69:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
  400b6e:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b72:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b76:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b7a:	f2 0f 11 04 d5 80 0e 	movsd  %xmm0,0x630e80(,%rdx,8)
  400b81:	63 00 
  400b83:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b87:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400b8b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b8f:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b93:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b97:	f2 0f 11 04 d5 80 08 	movsd  %xmm0,0x640880(,%rdx,8)
  400b9e:	64 00 
  400ba0:	0f 85 e2 fe ff ff    	jne    400a88 <_Z13instantiationv+0x38>
  400ba6:	48 83 c4 20          	add    $0x20,%rsp
  400baa:	5b                   	pop    %rbx
  400bab:	5d                   	pop    %rbp
  400bac:	41 5c                	pop    %r12
  400bae:	41 5d                	pop    %r13
  400bb0:	41 5e                	pop    %r14
  400bb2:	c3                   	retq   
  400bb3:	0f 1f 00             	nopl   (%rax)
  400bb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bbd:	00 00 00 

0000000000400bc0 <_Z10run_simplev>:
  400bc0:	f2 0f 10 15 10 08 00 	movsd  0x810(%rip),%xmm2        # 4013d8 <_IO_stdin_used+0x68>
  400bc7:	00 
  400bc8:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400bcd:	ba 98 20 60 00       	mov    $0x602098,%edx
  400bd2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400bd8:	f2 0f 10 ba 00 e2 04 	movsd  0x4e200(%rdx),%xmm7
  400bdf:	00 
  400be0:	b8 64 00 00 00       	mov    $0x64,%eax
  400be5:	f2 0f 10 b2 00 dc 05 	movsd  0x5dc00(%rdx),%xmm6
  400bec:	00 
  400bed:	f2 0f 59 fa          	mulsd  %xmm2,%xmm7
  400bf1:	f2 0f 59 f2          	mulsd  %xmm2,%xmm6
  400bf5:	f2 0f 10 0a          	movsd  (%rdx),%xmm1
  400bf9:	f2 0f 10 82 00 ee 02 	movsd  0x2ee00(%rdx),%xmm0
  400c00:	00 
  400c01:	f2 0f 5e f9          	divsd  %xmm1,%xmm7
  400c05:	f2 0f 5e f1          	divsd  %xmm1,%xmm6
  400c09:	f2 0f 10 a2 00 fa 00 	movsd  0xfa00(%rdx),%xmm4
  400c10:	00 
  400c11:	f2 0f 10 8a 00 e8 03 	movsd  0x3e800(%rdx),%xmm1
  400c18:	00 
  400c19:	f2 0f 10 9a 00 f4 01 	movsd  0x1f400(%rdx),%xmm3
  400c20:	00 
  400c21:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400c28:	f2 0f 58 c7          	addsd  %xmm7,%xmm0
  400c2c:	83 e8 01             	sub    $0x1,%eax
  400c2f:	f2 0f 58 ce          	addsd  %xmm6,%xmm1
  400c33:	66 0f 28 e8          	movapd %xmm0,%xmm5
  400c37:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400c3b:	f2 0f 58 e5          	addsd  %xmm5,%xmm4
  400c3f:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400c43:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400c47:	f2 0f 58 dd          	addsd  %xmm5,%xmm3
  400c4b:	75 db                	jne    400c28 <_Z10run_simplev+0x68>
  400c4d:	f2 0f 11 82 00 ee 02 	movsd  %xmm0,0x2ee00(%rdx)
  400c54:	00 
  400c55:	48 83 c2 08          	add    $0x8,%rdx
  400c59:	f2 0f 11 8a f8 e7 03 	movsd  %xmm1,0x3e7f8(%rdx)
  400c60:	00 
  400c61:	f2 0f 11 a2 f8 f9 00 	movsd  %xmm4,0xf9f8(%rdx)
  400c68:	00 
  400c69:	f2 0f 11 9a f8 f3 01 	movsd  %xmm3,0x1f3f8(%rdx)
  400c70:	00 
  400c71:	48 81 fa 98 1a 61 00 	cmp    $0x611a98,%rdx
  400c78:	0f 85 5a ff ff ff    	jne    400bd8 <_Z10run_simplev+0x18>
  400c7e:	83 e9 01             	sub    $0x1,%ecx
  400c81:	0f 85 46 ff ff ff    	jne    400bcd <_Z10run_simplev+0xd>
  400c87:	f3 c3                	repz retq 
  400c89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c90 <_Z14run_simulationv>:
  400c90:	41 56                	push   %r14
  400c92:	41 55                	push   %r13
  400c94:	41 be 05 00 00 00    	mov    $0x5,%r14d
  400c9a:	41 54                	push   %r12
  400c9c:	55                   	push   %rbp
  400c9d:	53                   	push   %rbx
  400c9e:	48 83 ec 30          	sub    $0x30,%rsp
  400ca2:	f2 0f 10 3d 2e 07 00 	movsd  0x72e(%rip),%xmm7        # 4013d8 <_IO_stdin_used+0x68>
  400ca9:	00 
  400caa:	f2 44 0f 10 05 4d 07 	movsd  0x74d(%rip),%xmm8        # 401400 <_IO_stdin_used+0x90>
  400cb1:	00 00 
  400cb3:	f2 0f 10 35 35 07 00 	movsd  0x735(%rip),%xmm6        # 4013f0 <_IO_stdin_used+0x80>
  400cba:	00 
  400cbb:	f2 0f 10 2d 35 07 00 	movsd  0x735(%rip),%xmm5        # 4013f8 <_IO_stdin_used+0x88>
  400cc2:	00 
  400cc3:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400cc8:	31 f6                	xor    %esi,%esi
  400cca:	bf 98 02 65 00       	mov    $0x650298,%edi
  400ccf:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400cd5:	41 bc 98 1a 61 00    	mov    $0x611a98,%r12d
  400cdb:	45 31 ed             	xor    %r13d,%r13d
  400cde:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400ce4:	f2 44 0f 11 44 24 08 	movsd  %xmm8,0x8(%rsp)
  400ceb:	f2 0f 11 3c 24       	movsd  %xmm7,(%rsp)
  400cf0:	e8 5b fa ff ff       	callq  400750 <memset@plt>
  400cf5:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400cfa:	31 f6                	xor    %esi,%esi
  400cfc:	bf 98 fc 65 00       	mov    $0x65fc98,%edi
  400d01:	e8 4a fa ff ff       	callq  400750 <memset@plt>
  400d06:	f2 0f 10 3c 24       	movsd  (%rsp),%xmm7
  400d0b:	f2 44 0f 10 44 24 08 	movsd  0x8(%rsp),%xmm8
  400d12:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400d18:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400d1e:	66 90                	xchg   %ax,%ax
  400d20:	49 83 c5 01          	add    $0x1,%r13
  400d24:	bb 98 02 65 00       	mov    $0x650298,%ebx
  400d29:	31 ed                	xor    %ebp,%ebp
  400d2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400d30:	48 83 c5 01          	add    $0x1,%rbp
  400d34:	49 39 ed             	cmp    %rbp,%r13
  400d37:	0f 84 9c 00 00 00    	je     400dd9 <_Z14run_simulationv+0x149>
  400d3d:	f2 41 0f 10 8c 24 00 	movsd  0xfa00(%r12),%xmm1
  400d44:	fa 00 00 
  400d47:	f2 41 0f 10 1c 24    	movsd  (%r12),%xmm3
  400d4d:	f2 0f 5c 8b 00 12 fd 	subsd  -0x2ee00(%rbx),%xmm1
  400d54:	ff 
  400d55:	f2 0f 5c 9b 00 18 fc 	subsd  -0x3e800(%rbx),%xmm3
  400d5c:	ff 
  400d5d:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400d62:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400d66:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400d6b:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400d6f:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400d74:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400d78:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400d7c:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400d80:	0f 8a 6a 01 00 00    	jp     400ef0 <_Z14run_simulationv+0x260>
  400d86:	f2 0f 10 93 00 1e fb 	movsd  -0x4e200(%rbx),%xmm2
  400d8d:	ff 
  400d8e:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400d92:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400d96:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d9a:	f2 41 0f 59 94 24 00 	mulsd  -0xfa00(%r12),%xmm2
  400da1:	06 ff ff 
  400da4:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400da8:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400dac:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400db0:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400db4:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400db9:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400dbd:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400dc1:	f2 0f 58 1b          	addsd  (%rbx),%xmm3
  400dc5:	f2 0f 11 1b          	movsd  %xmm3,(%rbx)
  400dc9:	f2 0f 58 8b 00 fa 00 	addsd  0xfa00(%rbx),%xmm1
  400dd0:	00 
  400dd1:	f2 0f 11 8b 00 fa 00 	movsd  %xmm1,0xfa00(%rbx)
  400dd8:	00 
  400dd9:	48 83 c3 08          	add    $0x8,%rbx
  400ddd:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400de4:	0f 85 46 ff ff ff    	jne    400d30 <_Z14run_simulationv+0xa0>
  400dea:	49 83 c4 08          	add    $0x8,%r12
  400dee:	49 81 fd 40 1f 00 00 	cmp    $0x1f40,%r13
  400df5:	0f 85 25 ff ff ff    	jne    400d20 <_Z14run_simulationv+0x90>
  400dfb:	f2 0f 10 25 0d 06 00 	movsd  0x60d(%rip),%xmm4        # 401410 <_IO_stdin_used+0xa0>
  400e02:	00 
  400e03:	b8 98 0e 63 00       	mov    $0x630e98,%eax
  400e08:	f2 44 0f 10 0d d7 05 	movsd  0x5d7(%rip),%xmm9        # 4013e8 <_IO_stdin_used+0x78>
  400e0f:	00 00 
  400e11:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400e16:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400e1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400e20:	f2 0f 10 88 00 f4 01 	movsd  0x1f400(%rax),%xmm1
  400e27:	00 
  400e28:	f2 0f 10 90 00 12 fd 	movsd  -0x2ee00(%rax),%xmm2
  400e2f:	ff 
  400e30:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400e34:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  400e3b:	00 
  400e3c:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400e40:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400e44:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400e48:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400e4c:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400e50:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400e54:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400e58:	f2 0f 58 80 00 fa 00 	addsd  0xfa00(%rax),%xmm0
  400e5f:	00 
  400e60:	f2 0f 58 98 00 0c fe 	addsd  -0x1f400(%rax),%xmm3
  400e67:	ff 
  400e68:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400e6c:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400e73:	00 
  400e74:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400e79:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400e7d:	f2 0f 11 98 00 0c fe 	movsd  %xmm3,-0x1f400(%rax)
  400e84:	ff 
  400e85:	f2 0f 58 90 00 06 ff 	addsd  -0xfa00(%rax),%xmm2
  400e8c:	ff 
  400e8d:	f2 0f 11 90 00 06 ff 	movsd  %xmm2,-0xfa00(%rax)
  400e94:	ff 
  400e95:	77 07                	ja     400e9e <_Z14run_simulationv+0x20e>
  400e97:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400e9c:	76 08                	jbe    400ea6 <_Z14run_simulationv+0x216>
  400e9e:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400ea2:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400ea6:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400eab:	77 07                	ja     400eb4 <_Z14run_simulationv+0x224>
  400ead:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400eb2:	76 0d                	jbe    400ec1 <_Z14run_simulationv+0x231>
  400eb4:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400eb9:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400ec0:	00 
  400ec1:	48 83 c0 08          	add    $0x8,%rax
  400ec5:	48 3d 98 08 64 00    	cmp    $0x640898,%rax
  400ecb:	0f 85 4f ff ff ff    	jne    400e20 <_Z14run_simulationv+0x190>
  400ed1:	41 83 ee 01          	sub    $0x1,%r14d
  400ed5:	0f 85 e8 fd ff ff    	jne    400cc3 <_Z14run_simulationv+0x33>
  400edb:	48 83 c4 30          	add    $0x30,%rsp
  400edf:	5b                   	pop    %rbx
  400ee0:	5d                   	pop    %rbp
  400ee1:	41 5c                	pop    %r12
  400ee3:	41 5d                	pop    %r13
  400ee5:	41 5e                	pop    %r14
  400ee7:	c3                   	retq   
  400ee8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400eef:	00 
  400ef0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400ef6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400efc:	f2 44 0f 11 44 24 18 	movsd  %xmm8,0x18(%rsp)
  400f03:	f2 0f 11 7c 24 10    	movsd  %xmm7,0x10(%rsp)
  400f09:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400f10:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400f15:	e8 c6 f8 ff ff       	callq  4007e0 <sqrt@plt>
  400f1a:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400f20:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400f24:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400f2a:	f2 44 0f 10 44 24 18 	movsd  0x18(%rsp),%xmm8
  400f31:	f2 0f 10 7c 24 10    	movsd  0x10(%rsp),%xmm7
  400f37:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400f3e:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400f43:	e9 3e fe ff ff       	jmpq   400d86 <_Z14run_simulationv+0xf6>
  400f48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f4f:	00 

0000000000400f50 <_Z11r_float2intf>:
  400f50:	66 0f 7e c0          	movd   %xmm0,%eax
  400f54:	c3                   	retq   
  400f55:	90                   	nop
  400f56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f5d:	00 00 00 

0000000000400f60 <_ZN4Body7storageEv>:
  400f60:	b8 80 20 60 00       	mov    $0x602080,%eax
  400f65:	c3                   	retq   
  400f66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f6d:	00 00 00 

0000000000400f70 <_Z19codegen_reset_forceP4Body>:
  400f70:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400f74:	f2 0f 11 04 fd 90 02 	movsd  %xmm0,0x650290(,%rdi,8)
  400f7b:	65 00 
  400f7d:	f2 0f 11 04 fd 90 fc 	movsd  %xmm0,0x65fc90(,%rdi,8)
  400f84:	65 00 
  400f86:	c3                   	retq   
  400f87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f8e:	00 00 

0000000000400f90 <_Z17codegen_add_forceP4BodyS0_>:
  400f90:	48 39 f7             	cmp    %rsi,%rdi
  400f93:	0f 84 c7 00 00 00    	je     401060 <_Z17codegen_add_forceP4BodyS0_+0xd0>
  400f99:	41 54                	push   %r12
  400f9b:	55                   	push   %rbp
  400f9c:	49 89 fc             	mov    %rdi,%r12
  400f9f:	53                   	push   %rbx
  400fa0:	48 89 f5             	mov    %rsi,%rbp
  400fa3:	48 8d 1c fd 00 00 00 	lea    0x0(,%rdi,8),%rbx
  400faa:	00 
  400fab:	48 83 ec 10          	sub    $0x10,%rsp
  400faf:	f2 0f 10 0c f5 90 14 	movsd  0x621490(,%rsi,8),%xmm1
  400fb6:	62 00 
  400fb8:	f2 0f 10 1c f5 90 1a 	movsd  0x611a90(,%rsi,8),%xmm3
  400fbf:	61 00 
  400fc1:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400fc5:	f2 0f 5c 1c fd 90 1a 	subsd  0x611a90(,%rdi,8),%xmm3
  400fcc:	61 00 
  400fce:	f2 0f 5c 2c fd 90 14 	subsd  0x621490(,%rdi,8),%xmm5
  400fd5:	62 00 
  400fd7:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400fdb:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400fdf:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400fe3:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400fe7:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400feb:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400fef:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400ff3:	7a 6d                	jp     401062 <_Z17codegen_add_forceP4BodyS0_+0xd2>
  400ff5:	f2 0f 10 15 f3 03 00 	movsd  0x3f3(%rip),%xmm2        # 4013f0 <_IO_stdin_used+0x80>
  400ffc:	00 
  400ffd:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401001:	f2 42 0f 59 14 e5 90 	mulsd  0x602090(,%r12,8),%xmm2
  401008:	20 60 00 
  40100b:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  40100f:	f2 0f 58 05 e1 03 00 	addsd  0x3e1(%rip),%xmm0        # 4013f8 <_IO_stdin_used+0x88>
  401016:	00 
  401017:	f2 0f 59 14 ed 90 20 	mulsd  0x602090(,%rbp,8),%xmm2
  40101e:	60 00 
  401020:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  401024:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401028:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  40102c:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401030:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  401034:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  401038:	f2 0f 58 9b 90 02 65 	addsd  0x650290(%rbx),%xmm3
  40103f:	00 
  401040:	f2 0f 11 9b 90 02 65 	movsd  %xmm3,0x650290(%rbx)
  401047:	00 
  401048:	f2 0f 58 8b 90 fc 65 	addsd  0x65fc90(%rbx),%xmm1
  40104f:	00 
  401050:	f2 0f 11 8b 90 fc 65 	movsd  %xmm1,0x65fc90(%rbx)
  401057:	00 
  401058:	48 83 c4 10          	add    $0x10,%rsp
  40105c:	5b                   	pop    %rbx
  40105d:	5d                   	pop    %rbp
  40105e:	41 5c                	pop    %r12
  401060:	f3 c3                	repz retq 
  401062:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401068:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  40106d:	e8 6e f7 ff ff       	callq  4007e0 <sqrt@plt>
  401072:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  401078:	66 0f 28 e0          	movapd %xmm0,%xmm4
  40107c:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  401081:	e9 6f ff ff ff       	jmpq   400ff5 <_Z17codegen_add_forceP4BodyS0_+0x65>
  401086:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40108d:	00 00 00 

0000000000401090 <_Z14codegen_updateP4Bodyd>:
  401090:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401097:	65 00 
  401099:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  4010a0:	00 
  4010a1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4010a5:	48 8d 88 10 ee 02 00 	lea    0x2ee10(%rax),%rcx
  4010ac:	48 8d 90 10 e8 03 00 	lea    0x3e810(%rax),%rdx
  4010b3:	48 8d b0 10 fa 00 00 	lea    0xfa10(%rax),%rsi
  4010ba:	48 05 10 f4 01 00    	add    $0x1f410,%rax
  4010c0:	f2 0f 5e 88 80 2c 5e 	divsd  0x5e2c80(%rax),%xmm1
  4010c7:	00 
  4010c8:	f2 0f 58 89 80 20 60 	addsd  0x602080(%rcx),%xmm1
  4010cf:	00 
  4010d0:	f2 0f 11 89 80 20 60 	movsd  %xmm1,0x602080(%rcx)
  4010d7:	00 
  4010d8:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  4010df:	65 00 
  4010e1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4010e5:	f2 0f 5e 88 80 2c 5e 	divsd  0x5e2c80(%rax),%xmm1
  4010ec:	00 
  4010ed:	f2 0f 58 8a 80 20 60 	addsd  0x602080(%rdx),%xmm1
  4010f4:	00 
  4010f5:	f2 0f 11 8a 80 20 60 	movsd  %xmm1,0x602080(%rdx)
  4010fc:	00 
  4010fd:	f2 0f 10 89 80 20 60 	movsd  0x602080(%rcx),%xmm1
  401104:	00 
  401105:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401109:	f2 0f 58 88 80 26 5f 	addsd  0x5f2680(%rax),%xmm1
  401110:	00 
  401111:	f2 0f 11 88 80 26 5f 	movsd  %xmm1,0x5f2680(%rax)
  401118:	00 
  401119:	f2 0f 10 0d df 02 00 	movsd  0x2df(%rip),%xmm1        # 401400 <_IO_stdin_used+0x90>
  401120:	00 
  401121:	f2 0f 59 82 80 20 60 	mulsd  0x602080(%rdx),%xmm0
  401128:	00 
  401129:	f2 0f 58 80 80 20 60 	addsd  0x602080(%rax),%xmm0
  401130:	00 
  401131:	f2 0f 11 80 80 20 60 	movsd  %xmm0,0x602080(%rax)
  401138:	00 
  401139:	f2 0f 10 96 80 20 60 	movsd  0x602080(%rsi),%xmm2
  401140:	00 
  401141:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  401145:	77 0a                	ja     401151 <_Z14codegen_updateP4Bodyd+0xc1>
  401147:	66 0f 2e 15 99 02 00 	ucomisd 0x299(%rip),%xmm2        # 4013e8 <_IO_stdin_used+0x78>
  40114e:	00 
  40114f:	76 24                	jbe    401175 <_Z14codegen_updateP4Bodyd+0xe5>
  401151:	f2 0f 10 81 80 20 60 	movsd  0x602080(%rcx),%xmm0
  401158:	00 
  401159:	f2 0f 10 15 af 02 00 	movsd  0x2af(%rip),%xmm2        # 401410 <_IO_stdin_used+0xa0>
  401160:	00 
  401161:	66 0f 57 c2          	xorpd  %xmm2,%xmm0
  401165:	f2 0f 11 81 80 20 60 	movsd  %xmm0,0x602080(%rcx)
  40116c:	00 
  40116d:	f2 0f 10 80 80 20 60 	movsd  0x602080(%rax),%xmm0
  401174:	00 
  401175:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  401179:	77 0a                	ja     401185 <_Z14codegen_updateP4Bodyd+0xf5>
  40117b:	66 0f 2e 05 65 02 00 	ucomisd 0x265(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x78>
  401182:	00 
  401183:	76 23                	jbe    4011a8 <_Z14codegen_updateP4Bodyd+0x118>
  401185:	f2 0f 10 82 80 20 60 	movsd  0x602080(%rdx),%xmm0
  40118c:	00 
  40118d:	f2 0f 10 0d 7b 02 00 	movsd  0x27b(%rip),%xmm1        # 401410 <_IO_stdin_used+0xa0>
  401194:	00 
  401195:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401199:	f2 0f 11 82 80 20 60 	movsd  %xmm0,0x602080(%rdx)
  4011a0:	00 
  4011a1:	c3                   	retq   
  4011a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4011a8:	f3 c3                	repz retq 
  4011aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004011b0 <_Z26codegen_reset_force_manualP4Body>:
  4011b0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4011b4:	f2 0f 11 04 fd 90 02 	movsd  %xmm0,0x650290(,%rdi,8)
  4011bb:	65 00 
  4011bd:	f2 0f 11 04 fd 90 fc 	movsd  %xmm0,0x65fc90(,%rdi,8)
  4011c4:	65 00 
  4011c6:	c3                   	retq   
  4011c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4011ce:	00 00 

00000000004011d0 <_Z24codegen_reset_force_halfP4Body>:
  4011d0:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  4011d7:	00 00 00 00 00 
  4011dc:	c3                   	retq   
  4011dd:	0f 1f 00             	nopl   (%rax)

00000000004011e0 <_Z18codegen_reset_massP4Body>:
  4011e0:	48 c7 04 fd 90 20 60 	movq   $0x0,0x602090(,%rdi,8)
  4011e7:	00 00 00 00 00 
  4011ec:	c3                   	retq   
  4011ed:	0f 1f 00             	nopl   (%rax)

00000000004011f0 <_Z22codengen_simple_updateP4Bodyd>:
  4011f0:	f2 0f 10 3c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm7
  4011f7:	65 00 
  4011f9:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  401200:	00 
  401201:	f2 0f 10 34 fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm6
  401208:	65 00 
  40120a:	f2 0f 59 f8          	mulsd  %xmm0,%xmm7
  40120e:	4c 8d 80 10 ee 02 00 	lea    0x2ee10(%rax),%r8
  401215:	48 8d b0 10 e8 03 00 	lea    0x3e810(%rax),%rsi
  40121c:	f2 0f 59 f0          	mulsd  %xmm0,%xmm6
  401220:	48 8d 88 10 fa 00 00 	lea    0xfa10(%rax),%rcx
  401227:	48 8d 90 10 f4 01 00 	lea    0x1f410(%rax),%rdx
  40122e:	f2 0f 10 0c fd 90 20 	movsd  0x602090(,%rdi,8),%xmm1
  401235:	60 00 
  401237:	b8 64 00 00 00       	mov    $0x64,%eax
  40123c:	f2 41 0f 10 90 80 20 	movsd  0x602080(%r8),%xmm2
  401243:	60 00 
  401245:	f2 0f 5e f9          	divsd  %xmm1,%xmm7
  401249:	f2 0f 5e f1          	divsd  %xmm1,%xmm6
  40124d:	f2 0f 10 a1 80 20 60 	movsd  0x602080(%rcx),%xmm4
  401254:	00 
  401255:	f2 0f 10 8e 80 20 60 	movsd  0x602080(%rsi),%xmm1
  40125c:	00 
  40125d:	f2 0f 10 9a 80 20 60 	movsd  0x602080(%rdx),%xmm3
  401264:	00 
  401265:	0f 1f 00             	nopl   (%rax)
  401268:	f2 0f 58 d7          	addsd  %xmm7,%xmm2
  40126c:	66 0f 28 e8          	movapd %xmm0,%xmm5
  401270:	f2 0f 58 ce          	addsd  %xmm6,%xmm1
  401274:	83 e8 01             	sub    $0x1,%eax
  401277:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  40127b:	f2 0f 58 e5          	addsd  %xmm5,%xmm4
  40127f:	66 0f 28 e8          	movapd %xmm0,%xmm5
  401283:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  401287:	f2 0f 58 dd          	addsd  %xmm5,%xmm3
  40128b:	75 db                	jne    401268 <_Z22codengen_simple_updateP4Bodyd+0x78>
  40128d:	f2 41 0f 11 90 80 20 	movsd  %xmm2,0x602080(%r8)
  401294:	60 00 
  401296:	f2 0f 11 8e 80 20 60 	movsd  %xmm1,0x602080(%rsi)
  40129d:	00 
  40129e:	f2 0f 11 a1 80 20 60 	movsd  %xmm4,0x602080(%rcx)
  4012a5:	00 
  4012a6:	f2 0f 11 9a 80 20 60 	movsd  %xmm3,0x602080(%rdx)
  4012ad:	00 
  4012ae:	c3                   	retq   
  4012af:	90                   	nop

00000000004012b0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  4012b0:	55                   	push   %rbp
  4012b1:	53                   	push   %rbx
  4012b2:	48 89 fd             	mov    %rdi,%rbp
  4012b5:	48 83 ec 08          	sub    $0x8,%rsp
  4012b9:	e8 02 f5 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012be:	48 89 c3             	mov    %rax,%rbx
  4012c1:	ff d5                	callq  *%rbp
  4012c3:	e8 f8 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012c8:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  4012cf:	de 1b 43 
  4012d2:	48 29 d8             	sub    %rbx,%rax
  4012d5:	48 83 c4 08          	add    $0x8,%rsp
  4012d9:	48 89 c1             	mov    %rax,%rcx
  4012dc:	48 f7 ea             	imul   %rdx
  4012df:	48 c1 f9 3f          	sar    $0x3f,%rcx
  4012e3:	5b                   	pop    %rbx
  4012e4:	5d                   	pop    %rbp
  4012e5:	48 c1 fa 12          	sar    $0x12,%rdx
  4012e9:	48 89 d0             	mov    %rdx,%rax
  4012ec:	48 29 c8             	sub    %rcx,%rax
  4012ef:	c3                   	retq   

00000000004012f0 <__libc_csu_init>:
  4012f0:	41 57                	push   %r15
  4012f2:	41 56                	push   %r14
  4012f4:	41 89 ff             	mov    %edi,%r15d
  4012f7:	41 55                	push   %r13
  4012f9:	41 54                	push   %r12
  4012fb:	4c 8d 25 e6 0a 20 00 	lea    0x200ae6(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  401302:	55                   	push   %rbp
  401303:	48 8d 2d ee 0a 20 00 	lea    0x200aee(%rip),%rbp        # 601df8 <__init_array_end>
  40130a:	53                   	push   %rbx
  40130b:	49 89 f6             	mov    %rsi,%r14
  40130e:	49 89 d5             	mov    %rdx,%r13
  401311:	4c 29 e5             	sub    %r12,%rbp
  401314:	48 83 ec 08          	sub    $0x8,%rsp
  401318:	48 c1 fd 03          	sar    $0x3,%rbp
  40131c:	e8 f7 f3 ff ff       	callq  400718 <_init>
  401321:	48 85 ed             	test   %rbp,%rbp
  401324:	74 20                	je     401346 <__libc_csu_init+0x56>
  401326:	31 db                	xor    %ebx,%ebx
  401328:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40132f:	00 
  401330:	4c 89 ea             	mov    %r13,%rdx
  401333:	4c 89 f6             	mov    %r14,%rsi
  401336:	44 89 ff             	mov    %r15d,%edi
  401339:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40133d:	48 83 c3 01          	add    $0x1,%rbx
  401341:	48 39 eb             	cmp    %rbp,%rbx
  401344:	75 ea                	jne    401330 <__libc_csu_init+0x40>
  401346:	48 83 c4 08          	add    $0x8,%rsp
  40134a:	5b                   	pop    %rbx
  40134b:	5d                   	pop    %rbp
  40134c:	41 5c                	pop    %r12
  40134e:	41 5d                	pop    %r13
  401350:	41 5e                	pop    %r14
  401352:	41 5f                	pop    %r15
  401354:	c3                   	retq   
  401355:	90                   	nop
  401356:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40135d:	00 00 00 

0000000000401360 <__libc_csu_fini>:
  401360:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401364 <_fini>:
  401364:	48 83 ec 08          	sub    $0x8,%rsp
  401368:	48 83 c4 08          	add    $0x8,%rsp
  40136c:	c3                   	retq   
