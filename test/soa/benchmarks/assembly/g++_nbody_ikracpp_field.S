
bin/g++_nbody_ikracpp_field:     file format elf64-x86-64


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
  40080c:	41 bd a8 1a 61 00    	mov    $0x611aa8,%r13d
  400812:	53                   	push   %rbx
  400813:	45 31 e4             	xor    %r12d,%r12d
  400816:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  40081c:	48 83 ec 10          	sub    $0x10,%rsp
  400820:	48 c7 05 65 18 20 00 	movq   $0x0,0x201865(%rip)        # 602090 <__Bodydata_buffer>
  400827:	00 00 00 00 
  40082b:	48 c7 05 62 18 20 00 	movq   $0x0,0x201862(%rip)        # 602098 <__Bodydata_buffer+0x8>
  400832:	00 00 00 00 
  400836:	e8 f5 0a 00 00       	callq  401330 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40083b:	bf 40 0d 40 00       	mov    $0x400d40,%edi
  400840:	48 89 c3             	mov    %rax,%rbx
  400843:	e8 e8 0a 00 00       	callq  401330 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400848:	bf c0 0b 40 00       	mov    $0x400bc0,%edi
  40084d:	48 89 c5             	mov    %rax,%rbp
  400850:	e8 db 0a 00 00       	callq  401330 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
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
  4008ca:	be 10 14 40 00       	mov    $0x401410,%esi
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
  4008f1:	be f4 13 40 00       	mov    $0x4013f4,%esi
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
  400924:	bf d8 f6 66 00       	mov    $0x66f6d8,%edi
  400929:	e8 42 fe ff ff       	callq  400770 <_ZNSt8ios_base4InitC1Ev@plt>
  40092e:	ba 70 20 60 00       	mov    $0x602070,%edx
  400933:	be d8 f6 66 00       	mov    $0x66f6d8,%esi
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
  40095f:	49 c7 c0 e0 13 40 00 	mov    $0x4013e0,%r8
  400966:	48 c7 c1 70 13 40 00 	mov    $0x401370,%rcx
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
  400a00:	80 3d 79 16 20 00 00 	cmpb   $0x0,0x201679(%rip)        # 602080 <completed.7585>
  400a07:	75 11                	jne    400a1a <__do_global_dtors_aux+0x1a>
  400a09:	55                   	push   %rbp
  400a0a:	48 89 e5             	mov    %rsp,%rbp
  400a0d:	e8 6e ff ff ff       	callq  400980 <deregister_tm_clones>
  400a12:	5d                   	pop    %rbp
  400a13:	c6 05 66 16 20 00 01 	movb   $0x1,0x201666(%rip)        # 602080 <completed.7585>
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
  400a6c:	f2 0f 10 0d dc 09 00 	movsd  0x9dc(%rip),%xmm1        # 401450 <_IO_stdin_used+0x60>
  400a73:	00 
  400a74:	f2 0f 10 15 dc 09 00 	movsd  0x9dc(%rip),%xmm2        # 401458 <_IO_stdin_used+0x68>
  400a7b:	00 
  400a7c:	f2 0f 10 1d dc 09 00 	movsd  0x9dc(%rip),%xmm3        # 401460 <_IO_stdin_used+0x70>
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
  400ac1:	48 8b 15 c8 15 20 00 	mov    0x2015c8(%rip),%rdx        # 602090 <__Bodydata_buffer>
  400ac8:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  400ace:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  400ad4:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  400ad8:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400adc:	48 83 c2 03          	add    $0x3,%rdx
  400ae0:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  400ae6:	41 83 ed 01          	sub    $0x1,%r13d
  400aea:	48 89 0d 9f 15 20 00 	mov    %rcx,0x20159f(%rip)        # 602090 <__Bodydata_buffer>
  400af1:	48 c7 04 d5 90 02 65 	movq   $0x0,0x650290(,%rdx,8)
  400af8:	00 00 00 00 00 
  400afd:	48 c7 04 d5 90 fc 65 	movq   $0x0,0x65fc90(,%rdx,8)
  400b04:	00 00 00 00 00 
  400b09:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b0d:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400b11:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400b15:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400b19:	f2 0f 11 04 d5 90 20 	movsd  %xmm0,0x602090(,%rdx,8)
  400b20:	60 00 
  400b22:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b26:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400b2b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b2f:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b33:	f2 0f 5c 05 2d 09 00 	subsd  0x92d(%rip),%xmm0        # 401468 <_IO_stdin_used+0x78>
  400b3a:	00 
  400b3b:	f2 0f 11 04 d5 90 1a 	movsd  %xmm0,0x611a90(,%rdx,8)
  400b42:	61 00 
  400b44:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b48:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400b4c:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b50:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b54:	f2 0f 5c 05 0c 09 00 	subsd  0x90c(%rip),%xmm0        # 401468 <_IO_stdin_used+0x78>
  400b5b:	00 
  400b5c:	f2 0f 11 04 d5 90 14 	movsd  %xmm0,0x621490(,%rdx,8)
  400b63:	62 00 
  400b65:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b69:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
  400b6e:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b72:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b76:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b7a:	f2 0f 11 04 d5 90 0e 	movsd  %xmm0,0x630e90(,%rdx,8)
  400b81:	63 00 
  400b83:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b87:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400b8b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b8f:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b93:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b97:	f2 0f 11 04 d5 90 08 	movsd  %xmm0,0x640890(,%rdx,8)
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
  400bc0:	f2 0f 10 25 90 08 00 	movsd  0x890(%rip),%xmm4        # 401458 <_IO_stdin_used+0x68>
  400bc7:	00 
  400bc8:	ba 50 c3 00 00       	mov    $0xc350,%edx
  400bcd:	66 0f 28 15 bb 08 00 	movapd 0x8bb(%rip),%xmm2        # 401490 <_IO_stdin_used+0xa0>
  400bd4:	00 
  400bd5:	0f 1f 00             	nopl   (%rax)
  400bd8:	b8 b0 0e 63 00       	mov    $0x630eb0,%eax
  400bdd:	f2 0f 10 0d c3 f6 24 	movsd  0x24f6c3(%rip),%xmm1        # 6502a8 <__Bodydata_buffer+0x4e218>
  400be4:	00 
  400be5:	f2 0f 10 05 bb f0 25 	movsd  0x25f0bb(%rip),%xmm0        # 65fca8 <__Bodydata_buffer+0x5dc18>
  400bec:	00 
  400bed:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400bf1:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400bf5:	f2 0f 10 1d ab 14 20 	movsd  0x2014ab(%rip),%xmm3        # 6020a8 <__Bodydata_buffer+0x18>
  400bfc:	00 
  400bfd:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400c01:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400c05:	f2 0f 58 0d 9b 02 23 	addsd  0x23029b(%rip),%xmm1        # 630ea8 <__Bodydata_buffer+0x2ee18>
  400c0c:	00 
  400c0d:	f2 0f 11 0d 93 02 23 	movsd  %xmm1,0x230293(%rip)        # 630ea8 <__Bodydata_buffer+0x2ee18>
  400c14:	00 
  400c15:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400c19:	f2 0f 58 05 87 fc 23 	addsd  0x23fc87(%rip),%xmm0        # 6408a8 <__Bodydata_buffer+0x3e818>
  400c20:	00 
  400c21:	f2 0f 58 0d 7f 0e 21 	addsd  0x210e7f(%rip),%xmm1        # 611aa8 <__Bodydata_buffer+0xfa18>
  400c28:	00 
  400c29:	f2 0f 11 05 77 fc 23 	movsd  %xmm0,0x23fc77(%rip)        # 6408a8 <__Bodydata_buffer+0x3e818>
  400c30:	00 
  400c31:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c35:	f2 0f 11 0d 6b 0e 21 	movsd  %xmm1,0x210e6b(%rip)        # 611aa8 <__Bodydata_buffer+0xfa18>
  400c3c:	00 
  400c3d:	f2 0f 58 05 63 08 22 	addsd  0x220863(%rip),%xmm0        # 6214a8 <__Bodydata_buffer+0x1f418>
  400c44:	00 
  400c45:	f2 0f 11 05 5b 08 22 	movsd  %xmm0,0x22085b(%rip)        # 6214a8 <__Bodydata_buffer+0x1f418>
  400c4c:	00 
  400c4d:	0f 1f 00             	nopl   (%rax)
  400c50:	66 0f 28 88 00 f4 01 	movapd 0x1f400(%rax),%xmm1
  400c57:	00 
  400c58:	48 83 c0 10          	add    $0x10,%rax
  400c5c:	66 0f 28 80 f0 ed 02 	movapd 0x2edf0(%rax),%xmm0
  400c63:	00 
  400c64:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400c68:	66 0f 28 98 f0 11 fd 	movapd -0x2ee10(%rax),%xmm3
  400c6f:	ff 
  400c70:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400c74:	66 0f 5e cb          	divpd  %xmm3,%xmm1
  400c78:	66 0f 5e c3          	divpd  %xmm3,%xmm0
  400c7c:	66 0f 58 48 f0       	addpd  -0x10(%rax),%xmm1
  400c81:	0f 29 48 f0          	movaps %xmm1,-0x10(%rax)
  400c85:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400c89:	66 0f 58 88 f0 0b fe 	addpd  -0x1f410(%rax),%xmm1
  400c90:	ff 
  400c91:	66 0f 58 80 f0 f9 00 	addpd  0xf9f0(%rax),%xmm0
  400c98:	00 
  400c99:	0f 29 88 f0 0b fe ff 	movaps %xmm1,-0x1f410(%rax)
  400ca0:	0f 29 80 f0 f9 00 00 	movaps %xmm0,0xf9f0(%rax)
  400ca7:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400cab:	66 0f 58 80 f0 05 ff 	addpd  -0xfa10(%rax),%xmm0
  400cb2:	ff 
  400cb3:	0f 29 80 f0 05 ff ff 	movaps %xmm0,-0xfa10(%rax)
  400cba:	48 3d a0 08 64 00    	cmp    $0x6408a0,%rax
  400cc0:	75 8e                	jne    400c50 <_Z10run_simplev+0x90>
  400cc2:	f2 0f 10 0d d6 ef 25 	movsd  0x25efd6(%rip),%xmm1        # 65fca0 <__Bodydata_buffer+0x5dc10>
  400cc9:	00 
  400cca:	83 ea 01             	sub    $0x1,%edx
  400ccd:	f2 0f 10 05 cb e9 26 	movsd  0x26e9cb(%rip),%xmm0        # 66f6a0 <__Bodydata_buffer+0x6d610>
  400cd4:	00 
  400cd5:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400cd9:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400cdd:	f2 0f 5e 0d bb 0d 21 	divsd  0x210dbb(%rip),%xmm1        # 611aa0 <__Bodydata_buffer+0xfa10>
  400ce4:	00 
  400ce5:	f2 0f 5e 05 b3 0d 21 	divsd  0x210db3(%rip),%xmm0        # 611aa0 <__Bodydata_buffer+0xfa10>
  400cec:	00 
  400ced:	f2 0f 58 0d ab fb 23 	addsd  0x23fbab(%rip),%xmm1        # 6408a0 <__Bodydata_buffer+0x3e810>
  400cf4:	00 
  400cf5:	f2 0f 11 0d a3 fb 23 	movsd  %xmm1,0x23fba3(%rip)        # 6408a0 <__Bodydata_buffer+0x3e810>
  400cfc:	00 
  400cfd:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400d01:	f2 0f 58 0d 97 07 22 	addsd  0x220797(%rip),%xmm1        # 6214a0 <__Bodydata_buffer+0x1f410>
  400d08:	00 
  400d09:	f2 0f 58 05 8f f5 24 	addsd  0x24f58f(%rip),%xmm0        # 6502a0 <__Bodydata_buffer+0x4e210>
  400d10:	00 
  400d11:	f2 0f 11 0d 87 07 22 	movsd  %xmm1,0x220787(%rip)        # 6214a0 <__Bodydata_buffer+0x1f410>
  400d18:	00 
  400d19:	f2 0f 11 05 7f f5 24 	movsd  %xmm0,0x24f57f(%rip)        # 6502a0 <__Bodydata_buffer+0x4e210>
  400d20:	00 
  400d21:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d25:	f2 0f 58 05 73 01 23 	addsd  0x230173(%rip),%xmm0        # 630ea0 <__Bodydata_buffer+0x2ee10>
  400d2c:	00 
  400d2d:	f2 0f 11 05 6b 01 23 	movsd  %xmm0,0x23016b(%rip)        # 630ea0 <__Bodydata_buffer+0x2ee10>
  400d34:	00 
  400d35:	0f 85 9d fe ff ff    	jne    400bd8 <_Z10run_simplev+0x18>
  400d3b:	f3 c3                	repz retq 
  400d3d:	0f 1f 00             	nopl   (%rax)

0000000000400d40 <_Z14run_simulationv>:
  400d40:	41 56                	push   %r14
  400d42:	41 55                	push   %r13
  400d44:	41 be 05 00 00 00    	mov    $0x5,%r14d
  400d4a:	41 54                	push   %r12
  400d4c:	55                   	push   %rbp
  400d4d:	53                   	push   %rbx
  400d4e:	48 83 ec 30          	sub    $0x30,%rsp
  400d52:	f2 0f 10 3d fe 06 00 	movsd  0x6fe(%rip),%xmm7        # 401458 <_IO_stdin_used+0x68>
  400d59:	00 
  400d5a:	f2 44 0f 10 05 1d 07 	movsd  0x71d(%rip),%xmm8        # 401480 <_IO_stdin_used+0x90>
  400d61:	00 00 
  400d63:	f2 0f 10 35 05 07 00 	movsd  0x705(%rip),%xmm6        # 401470 <_IO_stdin_used+0x80>
  400d6a:	00 
  400d6b:	f2 0f 10 2d 05 07 00 	movsd  0x705(%rip),%xmm5        # 401478 <_IO_stdin_used+0x88>
  400d72:	00 
  400d73:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400d78:	31 f6                	xor    %esi,%esi
  400d7a:	bf a8 02 65 00       	mov    $0x6502a8,%edi
  400d7f:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400d85:	41 bc a8 1a 61 00    	mov    $0x611aa8,%r12d
  400d8b:	45 31 ed             	xor    %r13d,%r13d
  400d8e:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400d94:	f2 0f 11 7c 24 08    	movsd  %xmm7,0x8(%rsp)
  400d9a:	f2 44 0f 11 04 24    	movsd  %xmm8,(%rsp)
  400da0:	e8 ab f9 ff ff       	callq  400750 <memset@plt>
  400da5:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400daa:	31 f6                	xor    %esi,%esi
  400dac:	bf a8 fc 65 00       	mov    $0x65fca8,%edi
  400db1:	e8 9a f9 ff ff       	callq  400750 <memset@plt>
  400db6:	f2 44 0f 10 04 24    	movsd  (%rsp),%xmm8
  400dbc:	f2 0f 10 7c 24 08    	movsd  0x8(%rsp),%xmm7
  400dc2:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400dc8:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400dce:	66 90                	xchg   %ax,%ax
  400dd0:	49 83 c5 01          	add    $0x1,%r13
  400dd4:	bb a8 02 65 00       	mov    $0x6502a8,%ebx
  400dd9:	31 ed                	xor    %ebp,%ebp
  400ddb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400de0:	48 83 c5 01          	add    $0x1,%rbp
  400de4:	49 39 ed             	cmp    %rbp,%r13
  400de7:	0f 84 9c 00 00 00    	je     400e89 <_Z14run_simulationv+0x149>
  400ded:	f2 41 0f 10 8c 24 00 	movsd  0xfa00(%r12),%xmm1
  400df4:	fa 00 00 
  400df7:	f2 41 0f 10 1c 24    	movsd  (%r12),%xmm3
  400dfd:	f2 0f 5c 8b 00 12 fd 	subsd  -0x2ee00(%rbx),%xmm1
  400e04:	ff 
  400e05:	f2 0f 5c 9b 00 18 fc 	subsd  -0x3e800(%rbx),%xmm3
  400e0c:	ff 
  400e0d:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400e12:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400e16:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400e1b:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400e1f:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400e24:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400e28:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400e2c:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400e30:	0f 8a 6a 01 00 00    	jp     400fa0 <_Z14run_simulationv+0x260>
  400e36:	f2 0f 10 93 00 1e fb 	movsd  -0x4e200(%rbx),%xmm2
  400e3d:	ff 
  400e3e:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400e42:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400e46:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400e4a:	f2 41 0f 59 94 24 00 	mulsd  -0xfa00(%r12),%xmm2
  400e51:	06 ff ff 
  400e54:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400e58:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400e5c:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400e60:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400e64:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400e69:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400e6d:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400e71:	f2 0f 58 1b          	addsd  (%rbx),%xmm3
  400e75:	f2 0f 11 1b          	movsd  %xmm3,(%rbx)
  400e79:	f2 0f 58 8b 00 fa 00 	addsd  0xfa00(%rbx),%xmm1
  400e80:	00 
  400e81:	f2 0f 11 8b 00 fa 00 	movsd  %xmm1,0xfa00(%rbx)
  400e88:	00 
  400e89:	48 83 c3 08          	add    $0x8,%rbx
  400e8d:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400e94:	0f 85 46 ff ff ff    	jne    400de0 <_Z14run_simulationv+0xa0>
  400e9a:	49 83 c4 08          	add    $0x8,%r12
  400e9e:	49 81 fd 40 1f 00 00 	cmp    $0x1f40,%r13
  400ea5:	0f 85 25 ff ff ff    	jne    400dd0 <_Z14run_simulationv+0x90>
  400eab:	f2 0f 10 25 ed 05 00 	movsd  0x5ed(%rip),%xmm4        # 4014a0 <_IO_stdin_used+0xb0>
  400eb2:	00 
  400eb3:	b8 a8 0e 63 00       	mov    $0x630ea8,%eax
  400eb8:	f2 44 0f 10 0d a7 05 	movsd  0x5a7(%rip),%xmm9        # 401468 <_IO_stdin_used+0x78>
  400ebf:	00 00 
  400ec1:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400ec6:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400ecb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400ed0:	f2 0f 10 88 00 f4 01 	movsd  0x1f400(%rax),%xmm1
  400ed7:	00 
  400ed8:	f2 0f 10 90 00 12 fd 	movsd  -0x2ee00(%rax),%xmm2
  400edf:	ff 
  400ee0:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400ee4:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  400eeb:	00 
  400eec:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400ef0:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400ef4:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400ef8:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400efc:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400f00:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400f04:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400f08:	f2 0f 58 80 00 fa 00 	addsd  0xfa00(%rax),%xmm0
  400f0f:	00 
  400f10:	f2 0f 58 98 00 0c fe 	addsd  -0x1f400(%rax),%xmm3
  400f17:	ff 
  400f18:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400f1c:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400f23:	00 
  400f24:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400f29:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400f2d:	f2 0f 11 98 00 0c fe 	movsd  %xmm3,-0x1f400(%rax)
  400f34:	ff 
  400f35:	f2 0f 58 90 00 06 ff 	addsd  -0xfa00(%rax),%xmm2
  400f3c:	ff 
  400f3d:	f2 0f 11 90 00 06 ff 	movsd  %xmm2,-0xfa00(%rax)
  400f44:	ff 
  400f45:	77 07                	ja     400f4e <_Z14run_simulationv+0x20e>
  400f47:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400f4c:	76 08                	jbe    400f56 <_Z14run_simulationv+0x216>
  400f4e:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400f52:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400f56:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400f5b:	77 07                	ja     400f64 <_Z14run_simulationv+0x224>
  400f5d:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400f62:	76 0d                	jbe    400f71 <_Z14run_simulationv+0x231>
  400f64:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400f69:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400f70:	00 
  400f71:	48 83 c0 08          	add    $0x8,%rax
  400f75:	48 3d a8 08 64 00    	cmp    $0x6408a8,%rax
  400f7b:	0f 85 4f ff ff ff    	jne    400ed0 <_Z14run_simulationv+0x190>
  400f81:	41 83 ee 01          	sub    $0x1,%r14d
  400f85:	0f 85 e8 fd ff ff    	jne    400d73 <_Z14run_simulationv+0x33>
  400f8b:	48 83 c4 30          	add    $0x30,%rsp
  400f8f:	5b                   	pop    %rbx
  400f90:	5d                   	pop    %rbp
  400f91:	41 5c                	pop    %r12
  400f93:	41 5d                	pop    %r13
  400f95:	41 5e                	pop    %r14
  400f97:	c3                   	retq   
  400f98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f9f:	00 
  400fa0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400fa6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400fac:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400fb2:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400fb9:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400fc0:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400fc5:	e8 16 f8 ff ff       	callq  4007e0 <sqrt@plt>
  400fca:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400fd0:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400fd4:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400fda:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400fe0:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400fe7:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400fee:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400ff3:	e9 3e fe ff ff       	jmpq   400e36 <_Z14run_simulationv+0xf6>
  400ff8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400fff:	00 

0000000000401000 <_Z11r_float2intf>:
  401000:	66 0f 7e c0          	movd   %xmm0,%eax
  401004:	c3                   	retq   
  401005:	90                   	nop
  401006:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40100d:	00 00 00 

0000000000401010 <_ZN4Body7storageEv>:
  401010:	b8 90 20 60 00       	mov    $0x602090,%eax
  401015:	c3                   	retq   
  401016:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40101d:	00 00 00 

0000000000401020 <_Z19codegen_reset_forceP4Body>:
  401020:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401024:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  40102b:	f2 0f 11 04 fd 90 20 	movsd  %xmm0,0x602090(,%rdi,8)
  401032:	60 00 
  401034:	f2 0f 11 04 fd 90 1a 	movsd  %xmm0,0x611a90(,%rdi,8)
  40103b:	61 00 
  40103d:	c3                   	retq   
  40103e:	66 90                	xchg   %ax,%ax

0000000000401040 <_Z17codegen_add_forceP4BodyS0_>:
  401040:	48 39 f7             	cmp    %rsi,%rdi
  401043:	0f 84 d1 00 00 00    	je     40111a <_Z17codegen_add_forceP4BodyS0_+0xda>
  401049:	41 54                	push   %r12
  40104b:	55                   	push   %rbp
  40104c:	48 8d ae 42 1f 00 00 	lea    0x1f42(%rsi),%rbp
  401053:	53                   	push   %rbx
  401054:	4c 8d a7 42 1f 00 00 	lea    0x1f42(%rdi),%r12
  40105b:	48 83 ec 10          	sub    $0x10,%rsp
  40105f:	4a 8d 1c e5 00 00 00 	lea    0x0(,%r12,8),%rbx
  401066:	00 
  401067:	f2 0f 10 0c ed 90 1a 	movsd  0x611a90(,%rbp,8),%xmm1
  40106e:	61 00 
  401070:	f2 0f 10 1c ed 90 20 	movsd  0x602090(,%rbp,8),%xmm3
  401077:	60 00 
  401079:	66 0f 28 e9          	movapd %xmm1,%xmm5
  40107d:	f2 42 0f 5c 1c e5 90 	subsd  0x602090(,%r12,8),%xmm3
  401084:	20 60 00 
  401087:	f2 42 0f 5c 2c e5 90 	subsd  0x611a90(,%r12,8),%xmm5
  40108e:	1a 61 00 
  401091:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401095:	66 0f 28 c5          	movapd %xmm5,%xmm0
  401099:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  40109d:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  4010a1:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  4010a5:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  4010a9:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  4010ad:	7a 6d                	jp     40111c <_Z17codegen_add_forceP4BodyS0_+0xdc>
  4010af:	f2 0f 10 15 b9 03 00 	movsd  0x3b9(%rip),%xmm2        # 401470 <_IO_stdin_used+0x80>
  4010b6:	00 
  4010b7:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4010bb:	f2 42 0f 59 14 e5 90 	mulsd  0x5f2690(,%r12,8),%xmm2
  4010c2:	26 5f 00 
  4010c5:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  4010c9:	f2 0f 58 05 a7 03 00 	addsd  0x3a7(%rip),%xmm0        # 401478 <_IO_stdin_used+0x88>
  4010d0:	00 
  4010d1:	f2 0f 59 14 ed 90 26 	mulsd  0x5f2690(,%rbp,8),%xmm2
  4010d8:	5f 00 
  4010da:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  4010de:	66 0f 28 ca          	movapd %xmm2,%xmm1
  4010e2:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  4010e6:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  4010ea:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  4010ee:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  4010f2:	f2 0f 58 9b 90 08 64 	addsd  0x640890(%rbx),%xmm3
  4010f9:	00 
  4010fa:	f2 0f 11 9b 90 08 64 	movsd  %xmm3,0x640890(%rbx)
  401101:	00 
  401102:	f2 0f 58 8b 90 02 65 	addsd  0x650290(%rbx),%xmm1
  401109:	00 
  40110a:	f2 0f 11 8b 90 02 65 	movsd  %xmm1,0x650290(%rbx)
  401111:	00 
  401112:	48 83 c4 10          	add    $0x10,%rsp
  401116:	5b                   	pop    %rbx
  401117:	5d                   	pop    %rbp
  401118:	41 5c                	pop    %r12
  40111a:	f3 c3                	repz retq 
  40111c:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401122:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  401127:	e8 b4 f6 ff ff       	callq  4007e0 <sqrt@plt>
  40112c:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  401132:	66 0f 28 e0          	movapd %xmm0,%xmm4
  401136:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  40113b:	e9 6f ff ff ff       	jmpq   4010af <_Z17codegen_add_forceP4BodyS0_+0x6f>

0000000000401140 <_Z14codegen_updateP4Bodyd>:
  401140:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  401147:	f2 0f 10 15 31 03 00 	movsd  0x331(%rip),%xmm2        # 401480 <_IO_stdin_used+0x90>
  40114e:	00 
  40114f:	f2 0f 10 0c fd 90 20 	movsd  0x602090(,%rdi,8),%xmm1
  401156:	60 00 
  401158:	48 8d 04 fd 00 06 ff 	lea    -0xfa00(,%rdi,8),%rax
  40115f:	ff 
  401160:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401164:	f2 0f 5e 0c fd 90 3e 	divsd  0x5b3e90(,%rdi,8),%xmm1
  40116b:	5b 00 
  40116d:	f2 0f 58 0c fd 90 2c 	addsd  0x5e2c90(,%rdi,8),%xmm1
  401174:	5e 00 
  401176:	f2 0f 11 0c fd 90 2c 	movsd  %xmm1,0x5e2c90(,%rdi,8)
  40117d:	5e 00 
  40117f:	f2 0f 10 0c fd 90 1a 	movsd  0x611a90(,%rdi,8),%xmm1
  401186:	61 00 
  401188:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40118c:	f2 0f 5e 0c fd 90 3e 	divsd  0x5b3e90(,%rdi,8),%xmm1
  401193:	5b 00 
  401195:	f2 0f 58 88 90 20 60 	addsd  0x602090(%rax),%xmm1
  40119c:	00 
  40119d:	f2 0f 11 88 90 20 60 	movsd  %xmm1,0x602090(%rax)
  4011a4:	00 
  4011a5:	f2 0f 10 0c fd 90 2c 	movsd  0x5e2c90(,%rdi,8),%xmm1
  4011ac:	5e 00 
  4011ae:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011b2:	f2 0f 58 0c fd 90 38 	addsd  0x5c3890(,%rdi,8),%xmm1
  4011b9:	5c 00 
  4011bb:	f2 0f 11 0c fd 90 38 	movsd  %xmm1,0x5c3890(,%rdi,8)
  4011c2:	5c 00 
  4011c4:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  4011c8:	f2 0f 59 80 90 20 60 	mulsd  0x602090(%rax),%xmm0
  4011cf:	00 
  4011d0:	f2 0f 58 04 fd 90 32 	addsd  0x5d3290(,%rdi,8),%xmm0
  4011d7:	5d 00 
  4011d9:	f2 0f 11 04 fd 90 32 	movsd  %xmm0,0x5d3290(,%rdi,8)
  4011e0:	5d 00 
  4011e2:	77 0a                	ja     4011ee <_Z14codegen_updateP4Bodyd+0xae>
  4011e4:	66 0f 2e 0d 7c 02 00 	ucomisd 0x27c(%rip),%xmm1        # 401468 <_IO_stdin_used+0x78>
  4011eb:	00 
  4011ec:	76 27                	jbe    401215 <_Z14codegen_updateP4Bodyd+0xd5>
  4011ee:	f2 0f 10 04 fd 90 2c 	movsd  0x5e2c90(,%rdi,8),%xmm0
  4011f5:	5e 00 
  4011f7:	f2 0f 10 0d a1 02 00 	movsd  0x2a1(%rip),%xmm1        # 4014a0 <_IO_stdin_used+0xb0>
  4011fe:	00 
  4011ff:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401203:	f2 0f 11 04 fd 90 2c 	movsd  %xmm0,0x5e2c90(,%rdi,8)
  40120a:	5e 00 
  40120c:	f2 0f 10 04 fd 90 32 	movsd  0x5d3290(,%rdi,8),%xmm0
  401213:	5d 00 
  401215:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  401219:	77 0a                	ja     401225 <_Z14codegen_updateP4Bodyd+0xe5>
  40121b:	66 0f 2e 05 45 02 00 	ucomisd 0x245(%rip),%xmm0        # 401468 <_IO_stdin_used+0x78>
  401222:	00 
  401223:	76 23                	jbe    401248 <_Z14codegen_updateP4Bodyd+0x108>
  401225:	f2 0f 10 80 90 20 60 	movsd  0x602090(%rax),%xmm0
  40122c:	00 
  40122d:	f2 0f 10 0d 6b 02 00 	movsd  0x26b(%rip),%xmm1        # 4014a0 <_IO_stdin_used+0xb0>
  401234:	00 
  401235:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401239:	f2 0f 11 80 90 20 60 	movsd  %xmm0,0x602090(%rax)
  401240:	00 
  401241:	c3                   	retq   
  401242:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401248:	f3 c3                	repz retq 
  40124a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401250 <_Z26codegen_reset_force_manualP4Body>:
  401250:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401254:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  40125b:	f2 0f 11 04 fd 90 20 	movsd  %xmm0,0x602090(,%rdi,8)
  401262:	60 00 
  401264:	f2 0f 11 04 fd 90 1a 	movsd  %xmm0,0x611a90(,%rdi,8)
  40126b:	61 00 
  40126d:	c3                   	retq   
  40126e:	66 90                	xchg   %ax,%ax

0000000000401270 <_Z24codegen_reset_force_halfP4Body>:
  401270:	48 c7 04 fd a0 02 65 	movq   $0x0,0x6502a0(,%rdi,8)
  401277:	00 00 00 00 00 
  40127c:	c3                   	retq   
  40127d:	0f 1f 00             	nopl   (%rax)

0000000000401280 <_Z18codegen_reset_massP4Body>:
  401280:	48 c7 04 fd a0 20 60 	movq   $0x0,0x6020a0(,%rdi,8)
  401287:	00 00 00 00 00 
  40128c:	c3                   	retq   
  40128d:	0f 1f 00             	nopl   (%rax)

0000000000401290 <_Z22codengen_simple_updateP4Bodyd>:
  401290:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  401297:	f2 0f 10 0c fd 90 20 	movsd  0x602090(,%rdi,8),%xmm1
  40129e:	60 00 
  4012a0:	48 8d 04 fd 00 06 ff 	lea    -0xfa00(,%rdi,8),%rax
  4012a7:	ff 
  4012a8:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4012ac:	f2 0f 5e 0c fd 90 3e 	divsd  0x5b3e90(,%rdi,8),%xmm1
  4012b3:	5b 00 
  4012b5:	f2 0f 58 0c fd 90 2c 	addsd  0x5e2c90(,%rdi,8),%xmm1
  4012bc:	5e 00 
  4012be:	f2 0f 11 0c fd 90 2c 	movsd  %xmm1,0x5e2c90(,%rdi,8)
  4012c5:	5e 00 
  4012c7:	f2 0f 10 0c fd 90 1a 	movsd  0x611a90(,%rdi,8),%xmm1
  4012ce:	61 00 
  4012d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4012d4:	f2 0f 5e 0c fd 90 3e 	divsd  0x5b3e90(,%rdi,8),%xmm1
  4012db:	5b 00 
  4012dd:	f2 0f 58 88 90 20 60 	addsd  0x602090(%rax),%xmm1
  4012e4:	00 
  4012e5:	f2 0f 11 88 90 20 60 	movsd  %xmm1,0x602090(%rax)
  4012ec:	00 
  4012ed:	f2 0f 10 0c fd 90 2c 	movsd  0x5e2c90(,%rdi,8),%xmm1
  4012f4:	5e 00 
  4012f6:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4012fa:	f2 0f 58 0c fd 90 38 	addsd  0x5c3890(,%rdi,8),%xmm1
  401301:	5c 00 
  401303:	f2 0f 11 0c fd 90 38 	movsd  %xmm1,0x5c3890(,%rdi,8)
  40130a:	5c 00 
  40130c:	f2 0f 59 80 90 20 60 	mulsd  0x602090(%rax),%xmm0
  401313:	00 
  401314:	f2 0f 58 04 fd 90 32 	addsd  0x5d3290(,%rdi,8),%xmm0
  40131b:	5d 00 
  40131d:	f2 0f 11 04 fd 90 32 	movsd  %xmm0,0x5d3290(,%rdi,8)
  401324:	5d 00 
  401326:	c3                   	retq   
  401327:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40132e:	00 00 

0000000000401330 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401330:	55                   	push   %rbp
  401331:	53                   	push   %rbx
  401332:	48 89 fd             	mov    %rdi,%rbp
  401335:	48 83 ec 08          	sub    $0x8,%rsp
  401339:	e8 82 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40133e:	48 89 c3             	mov    %rax,%rbx
  401341:	ff d5                	callq  *%rbp
  401343:	e8 78 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401348:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40134f:	de 1b 43 
  401352:	48 29 d8             	sub    %rbx,%rax
  401355:	48 83 c4 08          	add    $0x8,%rsp
  401359:	48 89 c1             	mov    %rax,%rcx
  40135c:	48 f7 ea             	imul   %rdx
  40135f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401363:	5b                   	pop    %rbx
  401364:	5d                   	pop    %rbp
  401365:	48 c1 fa 12          	sar    $0x12,%rdx
  401369:	48 89 d0             	mov    %rdx,%rax
  40136c:	48 29 c8             	sub    %rcx,%rax
  40136f:	c3                   	retq   

0000000000401370 <__libc_csu_init>:
  401370:	41 57                	push   %r15
  401372:	41 56                	push   %r14
  401374:	41 89 ff             	mov    %edi,%r15d
  401377:	41 55                	push   %r13
  401379:	41 54                	push   %r12
  40137b:	4c 8d 25 66 0a 20 00 	lea    0x200a66(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  401382:	55                   	push   %rbp
  401383:	48 8d 2d 6e 0a 20 00 	lea    0x200a6e(%rip),%rbp        # 601df8 <__init_array_end>
  40138a:	53                   	push   %rbx
  40138b:	49 89 f6             	mov    %rsi,%r14
  40138e:	49 89 d5             	mov    %rdx,%r13
  401391:	4c 29 e5             	sub    %r12,%rbp
  401394:	48 83 ec 08          	sub    $0x8,%rsp
  401398:	48 c1 fd 03          	sar    $0x3,%rbp
  40139c:	e8 77 f3 ff ff       	callq  400718 <_init>
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
