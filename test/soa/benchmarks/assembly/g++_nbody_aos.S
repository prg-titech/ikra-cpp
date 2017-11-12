
bin/g++_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

00000000004006d8 <_init>:
  4006d8:	48 83 ec 08          	sub    $0x8,%rsp
  4006dc:	48 8b 05 15 19 20 00 	mov    0x201915(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4006e3:	48 85 c0             	test   %rax,%rax
  4006e6:	74 05                	je     4006ed <_init+0x15>
  4006e8:	e8 b3 00 00 00       	callq  4007a0 <sqrt@plt+0x10>
  4006ed:	48 83 c4 08          	add    $0x8,%rsp
  4006f1:	c3                   	retq   

Disassembly of section .plt:

0000000000400700 <__printf_chk@plt-0x10>:
  400700:	ff 35 02 19 20 00    	pushq  0x201902(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400706:	ff 25 04 19 20 00    	jmpq   *0x201904(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <__printf_chk@plt>:
  400710:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400716:	68 00 00 00 00       	pushq  $0x0
  40071b:	e9 e0 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400720 <_ZNSt8ios_base4InitC1Ev@plt>:
  400720:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400726:	68 01 00 00 00       	pushq  $0x1
  40072b:	e9 d0 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400730 <__libc_start_main@plt>:
  400730:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400736:	68 02 00 00 00       	pushq  $0x2
  40073b:	e9 c0 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400740 <__cxa_atexit@plt>:
  400740:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400746:	68 03 00 00 00       	pushq  $0x3
  40074b:	e9 b0 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400750 <_ZNSt8ios_base4InitD1Ev@plt>:
  400750:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400756:	68 04 00 00 00       	pushq  $0x4
  40075b:	e9 a0 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400760 <srand@plt>:
  400760:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400766:	68 05 00 00 00       	pushq  $0x5
  40076b:	e9 90 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400770:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400776:	68 06 00 00 00       	pushq  $0x6
  40077b:	e9 80 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400780 <rand@plt>:
  400780:	ff 25 ca 18 20 00    	jmpq   *0x2018ca(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400786:	68 07 00 00 00       	pushq  $0x7
  40078b:	e9 70 ff ff ff       	jmpq   400700 <_init+0x28>

0000000000400790 <sqrt@plt>:
  400790:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400796:	68 08 00 00 00       	pushq  $0x8
  40079b:	e9 60 ff ff ff       	jmpq   400700 <_init+0x28>

Disassembly of section .plt.got:

00000000004007a0 <.plt.got>:
  4007a0:	ff 25 52 18 20 00    	jmpq   *0x201852(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  4007a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007b0 <main>:
  4007b0:	41 56                	push   %r14
  4007b2:	41 55                	push   %r13
  4007b4:	bf 70 0a 40 00       	mov    $0x400a70,%edi
  4007b9:	41 54                	push   %r12
  4007bb:	55                   	push   %rbp
  4007bc:	41 bd a8 20 60 00    	mov    $0x6020a8,%r13d
  4007c2:	53                   	push   %rbx
  4007c3:	45 31 e4             	xor    %r12d,%r12d
  4007c6:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  4007cc:	48 83 ec 10          	sub    $0x10,%rsp
  4007d0:	e8 5b 06 00 00       	callq  400e30 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007d5:	bf d0 0b 40 00       	mov    $0x400bd0,%edi
  4007da:	48 89 c3             	mov    %rax,%rbx
  4007dd:	e8 4e 06 00 00       	callq  400e30 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007e2:	bf f0 09 40 00       	mov    $0x4009f0,%edi
  4007e7:	48 89 c5             	mov    %rax,%rbp
  4007ea:	e8 41 06 00 00       	callq  400e30 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007ef:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  4007f5:	49 89 c0             	mov    %rax,%r8
  4007f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4007ff:	00 
  400800:	66 0f ef d2          	pxor   %xmm2,%xmm2
  400804:	f2 41 0f 10 45 00    	movsd  0x0(%r13),%xmm0
  40080a:	66 0f ef db          	pxor   %xmm3,%xmm3
  40080e:	f2 0f 5a d0          	cvtsd2ss %xmm0,%xmm2
  400812:	f2 41 0f 10 4d 08    	movsd  0x8(%r13),%xmm1
  400818:	f2 0f 5a d9          	cvtsd2ss %xmm1,%xmm3
  40081c:	66 0f 7e d6          	movd   %xmm2,%esi
  400820:	41 01 f1             	add    %esi,%r9d
  400823:	66 0f 7e de          	movd   %xmm3,%esi
  400827:	44 01 ce             	add    %r9d,%esi
  40082a:	89 f0                	mov    %esi,%eax
  40082c:	41 f7 ee             	imul   %r14d
  40082f:	89 f0                	mov    %esi,%eax
  400831:	c1 f8 1f             	sar    $0x1f,%eax
  400834:	c1 fa 13             	sar    $0x13,%edx
  400837:	41 89 d1             	mov    %edx,%r9d
  40083a:	41 29 c1             	sub    %eax,%r9d
  40083d:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  400844:	44 29 ce             	sub    %r9d,%esi
  400847:	49 83 fc 09          	cmp    $0x9,%r12
  40084b:	41 89 f1             	mov    %esi,%r9d
  40084e:	76 37                	jbe    400887 <main+0xd7>
  400850:	49 83 c4 01          	add    $0x1,%r12
  400854:	49 83 c5 38          	add    $0x38,%r13
  400858:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  40085f:	75 9f                	jne    400800 <main+0x50>
  400861:	48 89 e9             	mov    %rbp,%rcx
  400864:	48 89 da             	mov    %rbx,%rdx
  400867:	be 10 0f 40 00       	mov    $0x400f10,%esi
  40086c:	bf 01 00 00 00       	mov    $0x1,%edi
  400871:	31 c0                	xor    %eax,%eax
  400873:	e8 98 fe ff ff       	callq  400710 <__printf_chk@plt>
  400878:	48 83 c4 10          	add    $0x10,%rsp
  40087c:	31 c0                	xor    %eax,%eax
  40087e:	5b                   	pop    %rbx
  40087f:	5d                   	pop    %rbp
  400880:	41 5c                	pop    %r12
  400882:	41 5d                	pop    %r13
  400884:	41 5e                	pop    %r14
  400886:	c3                   	retq   
  400887:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  40088b:	4c 89 e2             	mov    %r12,%rdx
  40088e:	be f4 0e 40 00       	mov    $0x400ef4,%esi
  400893:	bf 01 00 00 00       	mov    $0x1,%edi
  400898:	b8 02 00 00 00       	mov    $0x2,%eax
  40089d:	4c 89 04 24          	mov    %r8,(%rsp)
  4008a1:	e8 6a fe ff ff       	callq  400710 <__printf_chk@plt>
  4008a6:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  4008ab:	4c 8b 04 24          	mov    (%rsp),%r8
  4008af:	eb 9f                	jmp    400850 <main+0xa0>
  4008b1:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4008b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008bd:	00 00 00 

00000000004008c0 <_GLOBAL__sub_I__Z11r_float2intf>:
  4008c0:	48 83 ec 08          	sub    $0x8,%rsp
  4008c4:	bf a0 f0 67 00       	mov    $0x67f0a0,%edi
  4008c9:	e8 52 fe ff ff       	callq  400720 <_ZNSt8ios_base4InitC1Ev@plt>
  4008ce:	ba 68 20 60 00       	mov    $0x602068,%edx
  4008d3:	be a0 f0 67 00       	mov    $0x67f0a0,%esi
  4008d8:	bf 50 07 40 00       	mov    $0x400750,%edi
  4008dd:	48 83 c4 08          	add    $0x8,%rsp
  4008e1:	e9 5a fe ff ff       	jmpq   400740 <__cxa_atexit@plt>
  4008e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ed:	00 00 00 

00000000004008f0 <_start>:
  4008f0:	31 ed                	xor    %ebp,%ebp
  4008f2:	49 89 d1             	mov    %rdx,%r9
  4008f5:	5e                   	pop    %rsi
  4008f6:	48 89 e2             	mov    %rsp,%rdx
  4008f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008fd:	50                   	push   %rax
  4008fe:	54                   	push   %rsp
  4008ff:	49 c7 c0 e0 0e 40 00 	mov    $0x400ee0,%r8
  400906:	48 c7 c1 70 0e 40 00 	mov    $0x400e70,%rcx
  40090d:	48 c7 c7 b0 07 40 00 	mov    $0x4007b0,%rdi
  400914:	e8 17 fe ff ff       	callq  400730 <__libc_start_main@plt>
  400919:	f4                   	hlt    
  40091a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400920 <deregister_tm_clones>:
  400920:	b8 7f 20 60 00       	mov    $0x60207f,%eax
  400925:	55                   	push   %rbp
  400926:	48 2d 78 20 60 00    	sub    $0x602078,%rax
  40092c:	48 83 f8 0e          	cmp    $0xe,%rax
  400930:	48 89 e5             	mov    %rsp,%rbp
  400933:	76 1b                	jbe    400950 <deregister_tm_clones+0x30>
  400935:	b8 00 00 00 00       	mov    $0x0,%eax
  40093a:	48 85 c0             	test   %rax,%rax
  40093d:	74 11                	je     400950 <deregister_tm_clones+0x30>
  40093f:	5d                   	pop    %rbp
  400940:	bf 78 20 60 00       	mov    $0x602078,%edi
  400945:	ff e0                	jmpq   *%rax
  400947:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40094e:	00 00 
  400950:	5d                   	pop    %rbp
  400951:	c3                   	retq   
  400952:	0f 1f 40 00          	nopl   0x0(%rax)
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <register_tm_clones>:
  400960:	be 78 20 60 00       	mov    $0x602078,%esi
  400965:	55                   	push   %rbp
  400966:	48 81 ee 78 20 60 00 	sub    $0x602078,%rsi
  40096d:	48 c1 fe 03          	sar    $0x3,%rsi
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 89 f0             	mov    %rsi,%rax
  400977:	48 c1 e8 3f          	shr    $0x3f,%rax
  40097b:	48 01 c6             	add    %rax,%rsi
  40097e:	48 d1 fe             	sar    %rsi
  400981:	74 15                	je     400998 <register_tm_clones+0x38>
  400983:	b8 00 00 00 00       	mov    $0x0,%eax
  400988:	48 85 c0             	test   %rax,%rax
  40098b:	74 0b                	je     400998 <register_tm_clones+0x38>
  40098d:	5d                   	pop    %rbp
  40098e:	bf 78 20 60 00       	mov    $0x602078,%edi
  400993:	ff e0                	jmpq   *%rax
  400995:	0f 1f 00             	nopl   (%rax)
  400998:	5d                   	pop    %rbp
  400999:	c3                   	retq   
  40099a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009a0 <__do_global_dtors_aux>:
  4009a0:	80 3d d9 16 20 00 00 	cmpb   $0x0,0x2016d9(%rip)        # 602080 <completed.7585>
  4009a7:	75 11                	jne    4009ba <__do_global_dtors_aux+0x1a>
  4009a9:	55                   	push   %rbp
  4009aa:	48 89 e5             	mov    %rsp,%rbp
  4009ad:	e8 6e ff ff ff       	callq  400920 <deregister_tm_clones>
  4009b2:	5d                   	pop    %rbp
  4009b3:	c6 05 c6 16 20 00 01 	movb   $0x1,0x2016c6(%rip)        # 602080 <completed.7585>
  4009ba:	f3 c3                	repz retq 
  4009bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009c0 <frame_dummy>:
  4009c0:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  4009c5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4009c9:	75 05                	jne    4009d0 <frame_dummy+0x10>
  4009cb:	eb 93                	jmp    400960 <register_tm_clones>
  4009cd:	0f 1f 00             	nopl   (%rax)
  4009d0:	b8 00 00 00 00       	mov    $0x0,%eax
  4009d5:	48 85 c0             	test   %rax,%rax
  4009d8:	74 f1                	je     4009cb <frame_dummy+0xb>
  4009da:	55                   	push   %rbp
  4009db:	48 89 e5             	mov    %rsp,%rbp
  4009de:	ff d0                	callq  *%rax
  4009e0:	5d                   	pop    %rbp
  4009e1:	e9 7a ff ff ff       	jmpq   400960 <register_tm_clones>
  4009e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ed:	00 00 00 

00000000004009f0 <_Z10run_simplev>:
  4009f0:	f2 0f 10 15 58 05 00 	movsd  0x558(%rip),%xmm2        # 400f50 <_IO_stdin_used+0x60>
  4009f7:	00 
  4009f8:	ba 50 c3 00 00       	mov    $0xc350,%edx
  4009fd:	0f 1f 00             	nopl   (%rax)
  400a00:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400a05:	0f 1f 00             	nopl   (%rax)
  400a08:	f2 0f 10 48 10       	movsd  0x10(%rax),%xmm1
  400a0d:	48 83 c0 38          	add    $0x38,%rax
  400a11:	f2 0f 10 40 e0       	movsd  -0x20(%rax),%xmm0
  400a16:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400a1a:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a1e:	f2 0f 10 58 b0       	movsd  -0x50(%rax),%xmm3
  400a23:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400a27:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400a2b:	f2 0f 58 48 c8       	addsd  -0x38(%rax),%xmm1
  400a30:	f2 0f 11 48 c8       	movsd  %xmm1,-0x38(%rax)
  400a35:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400a39:	f2 0f 58 40 d0       	addsd  -0x30(%rax),%xmm0
  400a3e:	f2 0f 58 48 b8       	addsd  -0x48(%rax),%xmm1
  400a43:	f2 0f 11 40 d0       	movsd  %xmm0,-0x30(%rax)
  400a48:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a4c:	f2 0f 11 48 b8       	movsd  %xmm1,-0x48(%rax)
  400a51:	f2 0f 58 40 c0       	addsd  -0x40(%rax),%xmm0
  400a56:	f2 0f 11 40 c0       	movsd  %xmm0,-0x40(%rax)
  400a5b:	48 3d b8 f6 66 00    	cmp    $0x66f6b8,%rax
  400a61:	75 a5                	jne    400a08 <_Z10run_simplev+0x18>
  400a63:	83 ea 01             	sub    $0x1,%edx
  400a66:	75 98                	jne    400a00 <_Z10run_simplev+0x10>
  400a68:	f3 c3                	repz retq 
  400a6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a70 <_Z13instantiationv>:
  400a70:	53                   	push   %rbx
  400a71:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400a76:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  400a7b:	48 83 ec 20          	sub    $0x20,%rsp
  400a7f:	e8 dc fc ff ff       	callq  400760 <srand@plt>
  400a84:	0f 1f 40 00          	nopl   0x0(%rax)
  400a88:	e8 f3 fc ff ff       	callq  400780 <rand@plt>
  400a8d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a91:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a95:	f2 0f 5e 05 bb 04 00 	divsd  0x4bb(%rip),%xmm0        # 400f58 <_IO_stdin_used+0x68>
  400a9c:	00 
  400a9d:	f2 0f 59 05 ab 04 00 	mulsd  0x4ab(%rip),%xmm0        # 400f50 <_IO_stdin_used+0x60>
  400aa4:	00 
  400aa5:	f2 0f 58 05 a3 04 00 	addsd  0x4a3(%rip),%xmm0        # 400f50 <_IO_stdin_used+0x60>
  400aac:	00 
  400aad:	f2 0f 59 05 ab 04 00 	mulsd  0x4ab(%rip),%xmm0        # 400f60 <_IO_stdin_used+0x70>
  400ab4:	00 
  400ab5:	f2 0f 11 04 24       	movsd  %xmm0,(%rsp)
  400aba:	e8 c1 fc ff ff       	callq  400780 <rand@plt>
  400abf:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400ac3:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400ac7:	f2 0f 5e 05 89 04 00 	divsd  0x489(%rip),%xmm0        # 400f58 <_IO_stdin_used+0x68>
  400ace:	00 
  400acf:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400ad3:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400ad7:	f2 0f 5c 15 89 04 00 	subsd  0x489(%rip),%xmm2        # 400f68 <_IO_stdin_used+0x78>
  400ade:	00 
  400adf:	f2 0f 11 54 24 08    	movsd  %xmm2,0x8(%rsp)
  400ae5:	e8 96 fc ff ff       	callq  400780 <rand@plt>
  400aea:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aee:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400af2:	f2 0f 5e 05 5e 04 00 	divsd  0x45e(%rip),%xmm0        # 400f58 <_IO_stdin_used+0x68>
  400af9:	00 
  400afa:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400afe:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400b02:	f2 0f 5c 1d 5e 04 00 	subsd  0x45e(%rip),%xmm3        # 400f68 <_IO_stdin_used+0x78>
  400b09:	00 
  400b0a:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  400b10:	e8 6b fc ff ff       	callq  400780 <rand@plt>
  400b15:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b19:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400b1d:	f2 0f 5e 05 33 04 00 	divsd  0x433(%rip),%xmm0        # 400f58 <_IO_stdin_used+0x68>
  400b24:	00 
  400b25:	f2 0f 5c 05 23 04 00 	subsd  0x423(%rip),%xmm0        # 400f50 <_IO_stdin_used+0x60>
  400b2c:	00 
  400b2d:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400b31:	f2 0f 5e 25 27 04 00 	divsd  0x427(%rip),%xmm4        # 400f60 <_IO_stdin_used+0x70>
  400b38:	00 
  400b39:	f2 0f 11 64 24 18    	movsd  %xmm4,0x18(%rsp)
  400b3f:	e8 3c fc ff ff       	callq  400780 <rand@plt>
  400b44:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b48:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400b4c:	48 8b 05 1d 15 20 00 	mov    0x20151d(%rip),%rax        # 602070 <Body_arena_head>
  400b53:	48 85 c0             	test   %rax,%rax
  400b56:	f2 0f 5e 05 fa 03 00 	divsd  0x3fa(%rip),%xmm0        # 400f58 <_IO_stdin_used+0x68>
  400b5d:	00 
  400b5e:	f2 0f 5c 05 ea 03 00 	subsd  0x3ea(%rip),%xmm0        # 400f50 <_IO_stdin_used+0x60>
  400b65:	00 
  400b66:	f2 0f 5e 05 f2 03 00 	divsd  0x3f2(%rip),%xmm0        # 400f60 <_IO_stdin_used+0x70>
  400b6d:	00 
  400b6e:	74 3f                	je     400baf <_Z13instantiationv+0x13f>
  400b70:	f2 0f 11 40 20       	movsd  %xmm0,0x20(%rax)
  400b75:	48 c7 40 28 00 00 00 	movq   $0x0,0x28(%rax)
  400b7c:	00 
  400b7d:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  400b84:	00 
  400b85:	f2 0f 10 0c 24       	movsd  (%rsp),%xmm1
  400b8a:	f2 0f 10 54 24 08    	movsd  0x8(%rsp),%xmm2
  400b90:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  400b96:	f2 0f 10 64 24 18    	movsd  0x18(%rsp),%xmm4
  400b9c:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400ba0:	f2 0f 11 50 08       	movsd  %xmm2,0x8(%rax)
  400ba5:	f2 0f 11 58 10       	movsd  %xmm3,0x10(%rax)
  400baa:	f2 0f 11 60 18       	movsd  %xmm4,0x18(%rax)
  400baf:	48 83 c0 38          	add    $0x38,%rax
  400bb3:	83 eb 01             	sub    $0x1,%ebx
  400bb6:	48 89 05 b3 14 20 00 	mov    %rax,0x2014b3(%rip)        # 602070 <Body_arena_head>
  400bbd:	0f 85 c5 fe ff ff    	jne    400a88 <_Z13instantiationv+0x18>
  400bc3:	48 83 c4 20          	add    $0x20,%rsp
  400bc7:	5b                   	pop    %rbx
  400bc8:	c3                   	retq   
  400bc9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400bd0 <_Z14run_simulationv>:
  400bd0:	41 54                	push   %r12
  400bd2:	55                   	push   %rbp
  400bd3:	41 bc 05 00 00 00    	mov    $0x5,%r12d
  400bd9:	53                   	push   %rbx
  400bda:	48 83 ec 30          	sub    $0x30,%rsp
  400bde:	f2 0f 10 3d 6a 03 00 	movsd  0x36a(%rip),%xmm7        # 400f50 <_IO_stdin_used+0x60>
  400be5:	00 
  400be6:	f2 44 0f 10 05 91 03 	movsd  0x391(%rip),%xmm8        # 400f80 <_IO_stdin_used+0x90>
  400bed:	00 00 
  400bef:	f2 0f 10 35 79 03 00 	movsd  0x379(%rip),%xmm6        # 400f70 <_IO_stdin_used+0x80>
  400bf6:	00 
  400bf7:	f2 0f 10 2d 79 03 00 	movsd  0x379(%rip),%xmm5        # 400f78 <_IO_stdin_used+0x88>
  400bfe:	00 
  400bff:	b8 c8 20 60 00       	mov    $0x6020c8,%eax
  400c04:	0f 1f 40 00          	nopl   0x0(%rax)
  400c08:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400c0f:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400c16:	00 
  400c17:	48 83 c0 38          	add    $0x38,%rax
  400c1b:	48 3d c8 f6 66 00    	cmp    $0x66f6c8,%rax
  400c21:	75 e5                	jne    400c08 <_Z14run_simulationv+0x38>
  400c23:	bd a0 20 60 00       	mov    $0x6020a0,%ebp
  400c28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c2f:	00 
  400c30:	bb a0 20 60 00       	mov    $0x6020a0,%ebx
  400c35:	0f 1f 00             	nopl   (%rax)
  400c38:	48 39 dd             	cmp    %rbx,%rbp
  400c3b:	0f 84 83 00 00 00    	je     400cc4 <_Z14run_simulationv+0xf4>
  400c41:	f2 0f 10 4d 10       	movsd  0x10(%rbp),%xmm1
  400c46:	f2 0f 10 5d 08       	movsd  0x8(%rbp),%xmm3
  400c4b:	f2 0f 5c 4b 10       	subsd  0x10(%rbx),%xmm1
  400c50:	f2 0f 5c 5b 08       	subsd  0x8(%rbx),%xmm3
  400c55:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400c5a:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400c5e:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400c63:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400c67:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400c6c:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400c70:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400c74:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400c78:	0f 8a 42 01 00 00    	jp     400dc0 <_Z14run_simulationv+0x1f0>
  400c7e:	f2 0f 10 13          	movsd  (%rbx),%xmm2
  400c82:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400c86:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400c8a:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c8e:	f2 0f 59 55 00       	mulsd  0x0(%rbp),%xmm2
  400c93:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400c97:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400c9b:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400c9f:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400ca3:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400ca8:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400cac:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400cb0:	f2 0f 58 5b 28       	addsd  0x28(%rbx),%xmm3
  400cb5:	f2 0f 11 5b 28       	movsd  %xmm3,0x28(%rbx)
  400cba:	f2 0f 58 4b 30       	addsd  0x30(%rbx),%xmm1
  400cbf:	f2 0f 11 4b 30       	movsd  %xmm1,0x30(%rbx)
  400cc4:	48 83 c3 38          	add    $0x38,%rbx
  400cc8:	48 81 fb a0 f6 66 00 	cmp    $0x66f6a0,%rbx
  400ccf:	0f 85 63 ff ff ff    	jne    400c38 <_Z14run_simulationv+0x68>
  400cd5:	48 83 c5 38          	add    $0x38,%rbp
  400cd9:	48 81 fd a0 f6 66 00 	cmp    $0x66f6a0,%rbp
  400ce0:	0f 85 4a ff ff ff    	jne    400c30 <_Z14run_simulationv+0x60>
  400ce6:	f2 0f 10 25 a2 02 00 	movsd  0x2a2(%rip),%xmm4        # 400f90 <_IO_stdin_used+0xa0>
  400ced:	00 
  400cee:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400cf3:	f2 44 0f 10 0d 6c 02 	movsd  0x26c(%rip),%xmm9        # 400f68 <_IO_stdin_used+0x78>
  400cfa:	00 00 
  400cfc:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400d01:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400d06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d0d:	00 00 00 
  400d10:	f2 0f 10 48 10       	movsd  0x10(%rax),%xmm1
  400d15:	f2 0f 10 50 e8       	movsd  -0x18(%rax),%xmm2
  400d1a:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400d1e:	f2 0f 10 40 18       	movsd  0x18(%rax),%xmm0
  400d23:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400d27:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400d2b:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400d2f:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400d33:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400d37:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400d3b:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400d3f:	f2 0f 58 40 08       	addsd  0x8(%rax),%xmm0
  400d44:	f2 0f 58 58 f0       	addsd  -0x10(%rax),%xmm3
  400d49:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400d4d:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400d52:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400d57:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400d5b:	f2 0f 11 58 f0       	movsd  %xmm3,-0x10(%rax)
  400d60:	f2 0f 58 50 f8       	addsd  -0x8(%rax),%xmm2
  400d65:	f2 0f 11 50 f8       	movsd  %xmm2,-0x8(%rax)
  400d6a:	77 07                	ja     400d73 <_Z14run_simulationv+0x1a3>
  400d6c:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400d71:	76 08                	jbe    400d7b <_Z14run_simulationv+0x1ab>
  400d73:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400d77:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400d7b:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400d80:	77 07                	ja     400d89 <_Z14run_simulationv+0x1b9>
  400d82:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400d87:	76 0a                	jbe    400d93 <_Z14run_simulationv+0x1c3>
  400d89:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400d8e:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400d93:	48 83 c0 38          	add    $0x38,%rax
  400d97:	48 3d b8 f6 66 00    	cmp    $0x66f6b8,%rax
  400d9d:	0f 85 6d ff ff ff    	jne    400d10 <_Z14run_simulationv+0x140>
  400da3:	41 83 ec 01          	sub    $0x1,%r12d
  400da7:	0f 85 52 fe ff ff    	jne    400bff <_Z14run_simulationv+0x2f>
  400dad:	48 83 c4 30          	add    $0x30,%rsp
  400db1:	5b                   	pop    %rbx
  400db2:	5d                   	pop    %rbp
  400db3:	41 5c                	pop    %r12
  400db5:	c3                   	retq   
  400db6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dbd:	00 00 00 
  400dc0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400dc6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400dcc:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400dd2:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400dd9:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400de0:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400de5:	e8 a6 f9 ff ff       	callq  400790 <sqrt@plt>
  400dea:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400df0:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400df4:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400dfa:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400e00:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400e07:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400e0e:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400e13:	e9 66 fe ff ff       	jmpq   400c7e <_Z14run_simulationv+0xae>
  400e18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e1f:	00 

0000000000400e20 <_Z11r_float2intf>:
  400e20:	66 0f 7e c0          	movd   %xmm0,%eax
  400e24:	c3                   	retq   
  400e25:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2c:	00 00 00 
  400e2f:	90                   	nop

0000000000400e30 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  400e30:	55                   	push   %rbp
  400e31:	53                   	push   %rbx
  400e32:	48 89 fd             	mov    %rdi,%rbp
  400e35:	48 83 ec 08          	sub    $0x8,%rsp
  400e39:	e8 32 f9 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400e3e:	48 89 c3             	mov    %rax,%rbx
  400e41:	ff d5                	callq  *%rbp
  400e43:	e8 28 f9 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400e48:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  400e4f:	de 1b 43 
  400e52:	48 29 d8             	sub    %rbx,%rax
  400e55:	48 83 c4 08          	add    $0x8,%rsp
  400e59:	48 89 c1             	mov    %rax,%rcx
  400e5c:	48 f7 ea             	imul   %rdx
  400e5f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  400e63:	5b                   	pop    %rbx
  400e64:	5d                   	pop    %rbp
  400e65:	48 c1 fa 12          	sar    $0x12,%rdx
  400e69:	48 89 d0             	mov    %rdx,%rax
  400e6c:	48 29 c8             	sub    %rcx,%rax
  400e6f:	c3                   	retq   

0000000000400e70 <__libc_csu_init>:
  400e70:	41 57                	push   %r15
  400e72:	41 56                	push   %r14
  400e74:	41 89 ff             	mov    %edi,%r15d
  400e77:	41 55                	push   %r13
  400e79:	41 54                	push   %r12
  400e7b:	4c 8d 25 66 0f 20 00 	lea    0x200f66(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  400e82:	55                   	push   %rbp
  400e83:	48 8d 2d 6e 0f 20 00 	lea    0x200f6e(%rip),%rbp        # 601df8 <__init_array_end>
  400e8a:	53                   	push   %rbx
  400e8b:	49 89 f6             	mov    %rsi,%r14
  400e8e:	49 89 d5             	mov    %rdx,%r13
  400e91:	4c 29 e5             	sub    %r12,%rbp
  400e94:	48 83 ec 08          	sub    $0x8,%rsp
  400e98:	48 c1 fd 03          	sar    $0x3,%rbp
  400e9c:	e8 37 f8 ff ff       	callq  4006d8 <_init>
  400ea1:	48 85 ed             	test   %rbp,%rbp
  400ea4:	74 20                	je     400ec6 <__libc_csu_init+0x56>
  400ea6:	31 db                	xor    %ebx,%ebx
  400ea8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400eaf:	00 
  400eb0:	4c 89 ea             	mov    %r13,%rdx
  400eb3:	4c 89 f6             	mov    %r14,%rsi
  400eb6:	44 89 ff             	mov    %r15d,%edi
  400eb9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ebd:	48 83 c3 01          	add    $0x1,%rbx
  400ec1:	48 39 eb             	cmp    %rbp,%rbx
  400ec4:	75 ea                	jne    400eb0 <__libc_csu_init+0x40>
  400ec6:	48 83 c4 08          	add    $0x8,%rsp
  400eca:	5b                   	pop    %rbx
  400ecb:	5d                   	pop    %rbp
  400ecc:	41 5c                	pop    %r12
  400ece:	41 5d                	pop    %r13
  400ed0:	41 5e                	pop    %r14
  400ed2:	41 5f                	pop    %r15
  400ed4:	c3                   	retq   
  400ed5:	90                   	nop
  400ed6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400edd:	00 00 00 

0000000000400ee0 <__libc_csu_fini>:
  400ee0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ee4 <_fini>:
  400ee4:	48 83 ec 08          	sub    $0x8,%rsp
  400ee8:	48 83 c4 08          	add    $0x8,%rsp
  400eec:	c3                   	retq   
