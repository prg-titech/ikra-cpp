
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
  4007b4:	bf f0 09 40 00       	mov    $0x4009f0,%edi
  4007b9:	41 54                	push   %r12
  4007bb:	55                   	push   %rbp
  4007bc:	41 bd a8 20 60 00    	mov    $0x6020a8,%r13d
  4007c2:	53                   	push   %rbx
  4007c3:	45 31 e4             	xor    %r12d,%r12d
  4007c6:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  4007cc:	48 83 ec 10          	sub    $0x10,%rsp
  4007d0:	e8 7b 06 00 00       	callq  400e50 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007d5:	bf f0 0b 40 00       	mov    $0x400bf0,%edi
  4007da:	48 89 c3             	mov    %rax,%rbx
  4007dd:	e8 6e 06 00 00       	callq  400e50 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007e2:	bf 50 0b 40 00       	mov    $0x400b50,%edi
  4007e7:	48 89 c5             	mov    %rax,%rbp
  4007ea:	e8 61 06 00 00       	callq  400e50 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
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
  400867:	be 30 0f 40 00       	mov    $0x400f30,%esi
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
  40088e:	be 14 0f 40 00       	mov    $0x400f14,%esi
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
  4008ff:	49 c7 c0 00 0f 40 00 	mov    $0x400f00,%r8
  400906:	48 c7 c1 90 0e 40 00 	mov    $0x400e90,%rcx
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

00000000004009f0 <_Z13instantiationv>:
  4009f0:	53                   	push   %rbx
  4009f1:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4009f6:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4009fb:	48 83 ec 20          	sub    $0x20,%rsp
  4009ff:	e8 5c fd ff ff       	callq  400760 <srand@plt>
  400a04:	0f 1f 40 00          	nopl   0x0(%rax)
  400a08:	e8 73 fd ff ff       	callq  400780 <rand@plt>
  400a0d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a11:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a15:	f2 0f 5e 05 53 05 00 	divsd  0x553(%rip),%xmm0        # 400f70 <_IO_stdin_used+0x60>
  400a1c:	00 
  400a1d:	f2 0f 59 05 53 05 00 	mulsd  0x553(%rip),%xmm0        # 400f78 <_IO_stdin_used+0x68>
  400a24:	00 
  400a25:	f2 0f 58 05 4b 05 00 	addsd  0x54b(%rip),%xmm0        # 400f78 <_IO_stdin_used+0x68>
  400a2c:	00 
  400a2d:	f2 0f 59 05 4b 05 00 	mulsd  0x54b(%rip),%xmm0        # 400f80 <_IO_stdin_used+0x70>
  400a34:	00 
  400a35:	f2 0f 11 04 24       	movsd  %xmm0,(%rsp)
  400a3a:	e8 41 fd ff ff       	callq  400780 <rand@plt>
  400a3f:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a43:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a47:	f2 0f 5e 05 21 05 00 	divsd  0x521(%rip),%xmm0        # 400f70 <_IO_stdin_used+0x60>
  400a4e:	00 
  400a4f:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a53:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400a57:	f2 0f 5c 15 29 05 00 	subsd  0x529(%rip),%xmm2        # 400f88 <_IO_stdin_used+0x78>
  400a5e:	00 
  400a5f:	f2 0f 11 54 24 08    	movsd  %xmm2,0x8(%rsp)
  400a65:	e8 16 fd ff ff       	callq  400780 <rand@plt>
  400a6a:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a6e:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a72:	f2 0f 5e 05 f6 04 00 	divsd  0x4f6(%rip),%xmm0        # 400f70 <_IO_stdin_used+0x60>
  400a79:	00 
  400a7a:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a7e:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400a82:	f2 0f 5c 1d fe 04 00 	subsd  0x4fe(%rip),%xmm3        # 400f88 <_IO_stdin_used+0x78>
  400a89:	00 
  400a8a:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  400a90:	e8 eb fc ff ff       	callq  400780 <rand@plt>
  400a95:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a99:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a9d:	f2 0f 5e 05 cb 04 00 	divsd  0x4cb(%rip),%xmm0        # 400f70 <_IO_stdin_used+0x60>
  400aa4:	00 
  400aa5:	f2 0f 5c 05 cb 04 00 	subsd  0x4cb(%rip),%xmm0        # 400f78 <_IO_stdin_used+0x68>
  400aac:	00 
  400aad:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400ab1:	f2 0f 5e 25 c7 04 00 	divsd  0x4c7(%rip),%xmm4        # 400f80 <_IO_stdin_used+0x70>
  400ab8:	00 
  400ab9:	f2 0f 11 64 24 18    	movsd  %xmm4,0x18(%rsp)
  400abf:	e8 bc fc ff ff       	callq  400780 <rand@plt>
  400ac4:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400ac8:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400acc:	48 8b 05 9d 15 20 00 	mov    0x20159d(%rip),%rax        # 602070 <Body_arena_head>
  400ad3:	48 85 c0             	test   %rax,%rax
  400ad6:	f2 0f 5e 05 92 04 00 	divsd  0x492(%rip),%xmm0        # 400f70 <_IO_stdin_used+0x60>
  400add:	00 
  400ade:	f2 0f 5c 05 92 04 00 	subsd  0x492(%rip),%xmm0        # 400f78 <_IO_stdin_used+0x68>
  400ae5:	00 
  400ae6:	f2 0f 5e 05 92 04 00 	divsd  0x492(%rip),%xmm0        # 400f80 <_IO_stdin_used+0x70>
  400aed:	00 
  400aee:	74 3f                	je     400b2f <_Z13instantiationv+0x13f>
  400af0:	f2 0f 11 40 20       	movsd  %xmm0,0x20(%rax)
  400af5:	48 c7 40 28 00 00 00 	movq   $0x0,0x28(%rax)
  400afc:	00 
  400afd:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  400b04:	00 
  400b05:	f2 0f 10 0c 24       	movsd  (%rsp),%xmm1
  400b0a:	f2 0f 10 54 24 08    	movsd  0x8(%rsp),%xmm2
  400b10:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  400b16:	f2 0f 10 64 24 18    	movsd  0x18(%rsp),%xmm4
  400b1c:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400b20:	f2 0f 11 50 08       	movsd  %xmm2,0x8(%rax)
  400b25:	f2 0f 11 58 10       	movsd  %xmm3,0x10(%rax)
  400b2a:	f2 0f 11 60 18       	movsd  %xmm4,0x18(%rax)
  400b2f:	48 83 c0 38          	add    $0x38,%rax
  400b33:	83 eb 01             	sub    $0x1,%ebx
  400b36:	48 89 05 33 15 20 00 	mov    %rax,0x201533(%rip)        # 602070 <Body_arena_head>
  400b3d:	0f 85 c5 fe ff ff    	jne    400a08 <_Z13instantiationv+0x18>
  400b43:	48 83 c4 20          	add    $0x20,%rsp
  400b47:	5b                   	pop    %rbx
  400b48:	c3                   	retq   
  400b49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400b50 <_Z10run_simplev>:
  400b50:	f2 0f 10 15 20 04 00 	movsd  0x420(%rip),%xmm2        # 400f78 <_IO_stdin_used+0x68>
  400b57:	00 
  400b58:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400b5d:	ba b8 20 60 00       	mov    $0x6020b8,%edx
  400b62:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b68:	f2 0f 10 7a 10       	movsd  0x10(%rdx),%xmm7
  400b6d:	b8 64 00 00 00       	mov    $0x64,%eax
  400b72:	f2 0f 10 72 18       	movsd  0x18(%rdx),%xmm6
  400b77:	f2 0f 59 fa          	mulsd  %xmm2,%xmm7
  400b7b:	f2 0f 59 f2          	mulsd  %xmm2,%xmm6
  400b7f:	f2 0f 10 42 e8       	movsd  -0x18(%rdx),%xmm0
  400b84:	f2 0f 10 0a          	movsd  (%rdx),%xmm1
  400b88:	f2 0f 5e f8          	divsd  %xmm0,%xmm7
  400b8c:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400b90:	f2 0f 10 62 f0       	movsd  -0x10(%rdx),%xmm4
  400b95:	f2 0f 10 42 08       	movsd  0x8(%rdx),%xmm0
  400b9a:	f2 0f 10 5a f8       	movsd  -0x8(%rdx),%xmm3
  400b9f:	90                   	nop
  400ba0:	f2 0f 58 cf          	addsd  %xmm7,%xmm1
  400ba4:	83 e8 01             	sub    $0x1,%eax
  400ba7:	f2 0f 58 c6          	addsd  %xmm6,%xmm0
  400bab:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400baf:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400bb3:	f2 0f 58 e5          	addsd  %xmm5,%xmm4
  400bb7:	66 0f 28 e8          	movapd %xmm0,%xmm5
  400bbb:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400bbf:	f2 0f 58 dd          	addsd  %xmm5,%xmm3
  400bc3:	75 db                	jne    400ba0 <_Z10run_simplev+0x50>
  400bc5:	f2 0f 11 0a          	movsd  %xmm1,(%rdx)
  400bc9:	48 83 c2 38          	add    $0x38,%rdx
  400bcd:	f2 0f 11 42 d0       	movsd  %xmm0,-0x30(%rdx)
  400bd2:	f2 0f 11 62 b8       	movsd  %xmm4,-0x48(%rdx)
  400bd7:	f2 0f 11 5a c0       	movsd  %xmm3,-0x40(%rdx)
  400bdc:	48 81 fa b8 f6 66 00 	cmp    $0x66f6b8,%rdx
  400be3:	75 83                	jne    400b68 <_Z10run_simplev+0x18>
  400be5:	83 e9 01             	sub    $0x1,%ecx
  400be8:	0f 85 6f ff ff ff    	jne    400b5d <_Z10run_simplev+0xd>
  400bee:	f3 c3                	repz retq 

0000000000400bf0 <_Z14run_simulationv>:
  400bf0:	41 54                	push   %r12
  400bf2:	55                   	push   %rbp
  400bf3:	41 bc 05 00 00 00    	mov    $0x5,%r12d
  400bf9:	53                   	push   %rbx
  400bfa:	48 83 ec 30          	sub    $0x30,%rsp
  400bfe:	f2 0f 10 3d 72 03 00 	movsd  0x372(%rip),%xmm7        # 400f78 <_IO_stdin_used+0x68>
  400c05:	00 
  400c06:	f2 44 0f 10 05 91 03 	movsd  0x391(%rip),%xmm8        # 400fa0 <_IO_stdin_used+0x90>
  400c0d:	00 00 
  400c0f:	f2 0f 10 35 79 03 00 	movsd  0x379(%rip),%xmm6        # 400f90 <_IO_stdin_used+0x80>
  400c16:	00 
  400c17:	f2 0f 10 2d 79 03 00 	movsd  0x379(%rip),%xmm5        # 400f98 <_IO_stdin_used+0x88>
  400c1e:	00 
  400c1f:	b8 c8 20 60 00       	mov    $0x6020c8,%eax
  400c24:	0f 1f 40 00          	nopl   0x0(%rax)
  400c28:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400c2f:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400c36:	00 
  400c37:	48 83 c0 38          	add    $0x38,%rax
  400c3b:	48 3d c8 f6 66 00    	cmp    $0x66f6c8,%rax
  400c41:	75 e5                	jne    400c28 <_Z14run_simulationv+0x38>
  400c43:	bd a0 20 60 00       	mov    $0x6020a0,%ebp
  400c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c4f:	00 
  400c50:	bb a0 20 60 00       	mov    $0x6020a0,%ebx
  400c55:	0f 1f 00             	nopl   (%rax)
  400c58:	48 39 dd             	cmp    %rbx,%rbp
  400c5b:	0f 84 83 00 00 00    	je     400ce4 <_Z14run_simulationv+0xf4>
  400c61:	f2 0f 10 4d 10       	movsd  0x10(%rbp),%xmm1
  400c66:	f2 0f 10 5d 08       	movsd  0x8(%rbp),%xmm3
  400c6b:	f2 0f 5c 4b 10       	subsd  0x10(%rbx),%xmm1
  400c70:	f2 0f 5c 5b 08       	subsd  0x8(%rbx),%xmm3
  400c75:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400c7a:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400c7e:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400c83:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400c87:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400c8c:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400c90:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400c94:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400c98:	0f 8a 42 01 00 00    	jp     400de0 <_Z14run_simulationv+0x1f0>
  400c9e:	f2 0f 10 13          	movsd  (%rbx),%xmm2
  400ca2:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400ca6:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400caa:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400cae:	f2 0f 59 55 00       	mulsd  0x0(%rbp),%xmm2
  400cb3:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400cb7:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400cbb:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400cbf:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400cc3:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400cc8:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400ccc:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400cd0:	f2 0f 58 5b 28       	addsd  0x28(%rbx),%xmm3
  400cd5:	f2 0f 11 5b 28       	movsd  %xmm3,0x28(%rbx)
  400cda:	f2 0f 58 4b 30       	addsd  0x30(%rbx),%xmm1
  400cdf:	f2 0f 11 4b 30       	movsd  %xmm1,0x30(%rbx)
  400ce4:	48 83 c3 38          	add    $0x38,%rbx
  400ce8:	48 81 fb a0 f6 66 00 	cmp    $0x66f6a0,%rbx
  400cef:	0f 85 63 ff ff ff    	jne    400c58 <_Z14run_simulationv+0x68>
  400cf5:	48 83 c5 38          	add    $0x38,%rbp
  400cf9:	48 81 fd a0 f6 66 00 	cmp    $0x66f6a0,%rbp
  400d00:	0f 85 4a ff ff ff    	jne    400c50 <_Z14run_simulationv+0x60>
  400d06:	f2 0f 10 25 a2 02 00 	movsd  0x2a2(%rip),%xmm4        # 400fb0 <_IO_stdin_used+0xa0>
  400d0d:	00 
  400d0e:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400d13:	f2 44 0f 10 0d 6c 02 	movsd  0x26c(%rip),%xmm9        # 400f88 <_IO_stdin_used+0x78>
  400d1a:	00 00 
  400d1c:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400d21:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400d26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d2d:	00 00 00 
  400d30:	f2 0f 10 48 10       	movsd  0x10(%rax),%xmm1
  400d35:	f2 0f 10 50 e8       	movsd  -0x18(%rax),%xmm2
  400d3a:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400d3e:	f2 0f 10 40 18       	movsd  0x18(%rax),%xmm0
  400d43:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400d47:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400d4b:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400d4f:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400d53:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400d57:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400d5b:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400d5f:	f2 0f 58 40 08       	addsd  0x8(%rax),%xmm0
  400d64:	f2 0f 58 58 f0       	addsd  -0x10(%rax),%xmm3
  400d69:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400d6d:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400d72:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400d77:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400d7b:	f2 0f 11 58 f0       	movsd  %xmm3,-0x10(%rax)
  400d80:	f2 0f 58 50 f8       	addsd  -0x8(%rax),%xmm2
  400d85:	f2 0f 11 50 f8       	movsd  %xmm2,-0x8(%rax)
  400d8a:	77 07                	ja     400d93 <_Z14run_simulationv+0x1a3>
  400d8c:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400d91:	76 08                	jbe    400d9b <_Z14run_simulationv+0x1ab>
  400d93:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400d97:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400d9b:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400da0:	77 07                	ja     400da9 <_Z14run_simulationv+0x1b9>
  400da2:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400da7:	76 0a                	jbe    400db3 <_Z14run_simulationv+0x1c3>
  400da9:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400dae:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400db3:	48 83 c0 38          	add    $0x38,%rax
  400db7:	48 3d b8 f6 66 00    	cmp    $0x66f6b8,%rax
  400dbd:	0f 85 6d ff ff ff    	jne    400d30 <_Z14run_simulationv+0x140>
  400dc3:	41 83 ec 01          	sub    $0x1,%r12d
  400dc7:	0f 85 52 fe ff ff    	jne    400c1f <_Z14run_simulationv+0x2f>
  400dcd:	48 83 c4 30          	add    $0x30,%rsp
  400dd1:	5b                   	pop    %rbx
  400dd2:	5d                   	pop    %rbp
  400dd3:	41 5c                	pop    %r12
  400dd5:	c3                   	retq   
  400dd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddd:	00 00 00 
  400de0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400de6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400dec:	f2 44 0f 11 44 24 18 	movsd  %xmm8,0x18(%rsp)
  400df3:	f2 0f 11 7c 24 10    	movsd  %xmm7,0x10(%rsp)
  400df9:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400e00:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400e05:	e8 86 f9 ff ff       	callq  400790 <sqrt@plt>
  400e0a:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400e10:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400e14:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400e1a:	f2 44 0f 10 44 24 18 	movsd  0x18(%rsp),%xmm8
  400e21:	f2 0f 10 7c 24 10    	movsd  0x10(%rsp),%xmm7
  400e27:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400e2e:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400e33:	e9 66 fe ff ff       	jmpq   400c9e <_Z14run_simulationv+0xae>
  400e38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e3f:	00 

0000000000400e40 <_Z11r_float2intf>:
  400e40:	66 0f 7e c0          	movd   %xmm0,%eax
  400e44:	c3                   	retq   
  400e45:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e4c:	00 00 00 
  400e4f:	90                   	nop

0000000000400e50 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  400e50:	55                   	push   %rbp
  400e51:	53                   	push   %rbx
  400e52:	48 89 fd             	mov    %rdi,%rbp
  400e55:	48 83 ec 08          	sub    $0x8,%rsp
  400e59:	e8 12 f9 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400e5e:	48 89 c3             	mov    %rax,%rbx
  400e61:	ff d5                	callq  *%rbp
  400e63:	e8 08 f9 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400e68:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  400e6f:	de 1b 43 
  400e72:	48 29 d8             	sub    %rbx,%rax
  400e75:	48 83 c4 08          	add    $0x8,%rsp
  400e79:	48 89 c1             	mov    %rax,%rcx
  400e7c:	48 f7 ea             	imul   %rdx
  400e7f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  400e83:	5b                   	pop    %rbx
  400e84:	5d                   	pop    %rbp
  400e85:	48 c1 fa 12          	sar    $0x12,%rdx
  400e89:	48 89 d0             	mov    %rdx,%rax
  400e8c:	48 29 c8             	sub    %rcx,%rax
  400e8f:	c3                   	retq   

0000000000400e90 <__libc_csu_init>:
  400e90:	41 57                	push   %r15
  400e92:	41 56                	push   %r14
  400e94:	41 89 ff             	mov    %edi,%r15d
  400e97:	41 55                	push   %r13
  400e99:	41 54                	push   %r12
  400e9b:	4c 8d 25 46 0f 20 00 	lea    0x200f46(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  400ea2:	55                   	push   %rbp
  400ea3:	48 8d 2d 4e 0f 20 00 	lea    0x200f4e(%rip),%rbp        # 601df8 <__init_array_end>
  400eaa:	53                   	push   %rbx
  400eab:	49 89 f6             	mov    %rsi,%r14
  400eae:	49 89 d5             	mov    %rdx,%r13
  400eb1:	4c 29 e5             	sub    %r12,%rbp
  400eb4:	48 83 ec 08          	sub    $0x8,%rsp
  400eb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ebc:	e8 17 f8 ff ff       	callq  4006d8 <_init>
  400ec1:	48 85 ed             	test   %rbp,%rbp
  400ec4:	74 20                	je     400ee6 <__libc_csu_init+0x56>
  400ec6:	31 db                	xor    %ebx,%ebx
  400ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ecf:	00 
  400ed0:	4c 89 ea             	mov    %r13,%rdx
  400ed3:	4c 89 f6             	mov    %r14,%rsi
  400ed6:	44 89 ff             	mov    %r15d,%edi
  400ed9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400edd:	48 83 c3 01          	add    $0x1,%rbx
  400ee1:	48 39 eb             	cmp    %rbp,%rbx
  400ee4:	75 ea                	jne    400ed0 <__libc_csu_init+0x40>
  400ee6:	48 83 c4 08          	add    $0x8,%rsp
  400eea:	5b                   	pop    %rbx
  400eeb:	5d                   	pop    %rbp
  400eec:	41 5c                	pop    %r12
  400eee:	41 5d                	pop    %r13
  400ef0:	41 5e                	pop    %r14
  400ef2:	41 5f                	pop    %r15
  400ef4:	c3                   	retq   
  400ef5:	90                   	nop
  400ef6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400efd:	00 00 00 

0000000000400f00 <__libc_csu_fini>:
  400f00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f04 <_fini>:
  400f04:	48 83 ec 08          	sub    $0x8,%rsp
  400f08:	48 83 c4 08          	add    $0x8,%rsp
  400f0c:	c3                   	retq   
