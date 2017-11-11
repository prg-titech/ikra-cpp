
bin/g++_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004006e0 <_init>:
  4006e0:	48 83 ec 08          	sub    $0x8,%rsp
  4006e4:	48 8b 05 0d 19 20 00 	mov    0x20190d(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4006eb:	48 85 c0             	test   %rax,%rax
  4006ee:	74 05                	je     4006f5 <_init+0x15>
  4006f0:	e8 ab 00 00 00       	callq  4007a0 <sqrt@plt+0x10>
  4006f5:	48 83 c4 08          	add    $0x8,%rsp
  4006f9:	c3                   	retq   

Disassembly of section .plt:

0000000000400700 <memset@plt-0x10>:
  400700:	ff 35 02 19 20 00    	pushq  0x201902(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400706:	ff 25 04 19 20 00    	jmpq   *0x201904(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <memset@plt>:
  400710:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400716:	68 00 00 00 00       	pushq  $0x0
  40071b:	e9 e0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400720 <__printf_chk@plt>:
  400720:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400726:	68 01 00 00 00       	pushq  $0x1
  40072b:	e9 d0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400730 <_ZNSt8ios_base4InitC1Ev@plt>:
  400730:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400736:	68 02 00 00 00       	pushq  $0x2
  40073b:	e9 c0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400740 <__libc_start_main@plt>:
  400740:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400746:	68 03 00 00 00       	pushq  $0x3
  40074b:	e9 b0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400750 <__cxa_atexit@plt>:
  400750:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400756:	68 04 00 00 00       	pushq  $0x4
  40075b:	e9 a0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400760 <_ZNSt8ios_base4InitD1Ev@plt>:
  400760:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400766:	68 05 00 00 00       	pushq  $0x5
  40076b:	e9 90 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400770:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400776:	68 06 00 00 00       	pushq  $0x6
  40077b:	e9 80 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400780 <rand@plt>:
  400780:	ff 25 ca 18 20 00    	jmpq   *0x2018ca(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400786:	68 07 00 00 00       	pushq  $0x7
  40078b:	e9 70 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400790 <sqrt@plt>:
  400790:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400796:	68 08 00 00 00       	pushq  $0x8
  40079b:	e9 60 ff ff ff       	jmpq   400700 <_init+0x20>

Disassembly of section .plt.got:

00000000004007a0 <.plt.got>:
  4007a0:	ff 25 52 18 20 00    	jmpq   *0x201852(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  4007a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007b0 <main>:
  4007b0:	55                   	push   %rbp
  4007b1:	53                   	push   %rbx
  4007b2:	bf a0 09 40 00       	mov    $0x4009a0,%edi
  4007b7:	48 83 ec 08          	sub    $0x8,%rsp
  4007bb:	e8 80 0a 00 00       	callq  401240 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007c0:	bf 90 0b 40 00       	mov    $0x400b90,%edi
  4007c5:	48 89 c3             	mov    %rax,%rbx
  4007c8:	e8 73 0a 00 00       	callq  401240 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007cd:	bf f0 0a 40 00       	mov    $0x400af0,%edi
  4007d2:	48 89 c5             	mov    %rax,%rbp
  4007d5:	e8 66 0a 00 00       	callq  401240 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007da:	31 c9                	xor    %ecx,%ecx
  4007dc:	49 89 c0             	mov    %rax,%r8
  4007df:	ba 0b 00 00 00       	mov    $0xb,%edx
  4007e4:	be 8d 68 b7 6c       	mov    $0x6cb7688d,%esi
  4007e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4007f0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4007f4:	48 83 c1 08          	add    $0x8,%rcx
  4007f8:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4007fc:	f2 0f 5a 81 98 02 65 	cvtsd2ss 0x650298(%rcx),%xmm0
  400803:	00 
  400804:	66 41 0f 7e c1       	movd   %xmm0,%r9d
  400809:	f2 0f 5a 89 98 08 64 	cvtsd2ss 0x640898(%rcx),%xmm1
  400810:	00 
  400811:	44 01 ca             	add    %r9d,%edx
  400814:	66 41 0f 7e c9       	movd   %xmm1,%r9d
  400819:	41 01 d1             	add    %edx,%r9d
  40081c:	44 89 c8             	mov    %r9d,%eax
  40081f:	f7 ee                	imul   %esi
  400821:	44 89 c8             	mov    %r9d,%eax
  400824:	c1 f8 1f             	sar    $0x1f,%eax
  400827:	c1 fa 13             	sar    $0x13,%edx
  40082a:	29 c2                	sub    %eax,%edx
  40082c:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  400832:	41 29 d1             	sub    %edx,%r9d
  400835:	48 81 f9 00 fa 00 00 	cmp    $0xfa00,%rcx
  40083c:	44 89 ca             	mov    %r9d,%edx
  40083f:	75 af                	jne    4007f0 <main+0x40>
  400841:	48 89 e9             	mov    %rbp,%rcx
  400844:	48 89 da             	mov    %rbx,%rdx
  400847:	be 08 13 40 00       	mov    $0x401308,%esi
  40084c:	bf 01 00 00 00       	mov    $0x1,%edi
  400851:	31 c0                	xor    %eax,%eax
  400853:	e8 c8 fe ff ff       	callq  400720 <__printf_chk@plt>
  400858:	48 83 c4 08          	add    $0x8,%rsp
  40085c:	31 c0                	xor    %eax,%eax
  40085e:	5b                   	pop    %rbx
  40085f:	5d                   	pop    %rbp
  400860:	c3                   	retq   
  400861:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400866:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40086d:	00 00 00 

0000000000400870 <_GLOBAL__sub_I__Z12float_as_intf>:
  400870:	48 83 ec 08          	sub    $0x8,%rsp
  400874:	bf a0 f6 66 00       	mov    $0x66f6a0,%edi
  400879:	e8 b2 fe ff ff       	callq  400730 <_ZNSt8ios_base4InitC1Ev@plt>
  40087e:	ba 68 20 60 00       	mov    $0x602068,%edx
  400883:	be a0 f6 66 00       	mov    $0x66f6a0,%esi
  400888:	bf 60 07 40 00       	mov    $0x400760,%edi
  40088d:	48 83 c4 08          	add    $0x8,%rsp
  400891:	e9 ba fe ff ff       	jmpq   400750 <__cxa_atexit@plt>
  400896:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089d:	00 00 00 

00000000004008a0 <_start>:
  4008a0:	31 ed                	xor    %ebp,%ebp
  4008a2:	49 89 d1             	mov    %rdx,%r9
  4008a5:	5e                   	pop    %rsi
  4008a6:	48 89 e2             	mov    %rsp,%rdx
  4008a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008ad:	50                   	push   %rax
  4008ae:	54                   	push   %rsp
  4008af:	49 c7 c0 f0 12 40 00 	mov    $0x4012f0,%r8
  4008b6:	48 c7 c1 80 12 40 00 	mov    $0x401280,%rcx
  4008bd:	48 c7 c7 b0 07 40 00 	mov    $0x4007b0,%rdi
  4008c4:	e8 77 fe ff ff       	callq  400740 <__libc_start_main@plt>
  4008c9:	f4                   	hlt    
  4008ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008d0 <deregister_tm_clones>:
  4008d0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4008d5:	55                   	push   %rbp
  4008d6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4008dc:	48 83 f8 0e          	cmp    $0xe,%rax
  4008e0:	48 89 e5             	mov    %rsp,%rbp
  4008e3:	76 1b                	jbe    400900 <deregister_tm_clones+0x30>
  4008e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008ea:	48 85 c0             	test   %rax,%rax
  4008ed:	74 11                	je     400900 <deregister_tm_clones+0x30>
  4008ef:	5d                   	pop    %rbp
  4008f0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4008f5:	ff e0                	jmpq   *%rax
  4008f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008fe:	00 00 
  400900:	5d                   	pop    %rbp
  400901:	c3                   	retq   
  400902:	0f 1f 40 00          	nopl   0x0(%rax)
  400906:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40090d:	00 00 00 

0000000000400910 <register_tm_clones>:
  400910:	be 70 20 60 00       	mov    $0x602070,%esi
  400915:	55                   	push   %rbp
  400916:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40091d:	48 c1 fe 03          	sar    $0x3,%rsi
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 89 f0             	mov    %rsi,%rax
  400927:	48 c1 e8 3f          	shr    $0x3f,%rax
  40092b:	48 01 c6             	add    %rax,%rsi
  40092e:	48 d1 fe             	sar    %rsi
  400931:	74 15                	je     400948 <register_tm_clones+0x38>
  400933:	b8 00 00 00 00       	mov    $0x0,%eax
  400938:	48 85 c0             	test   %rax,%rax
  40093b:	74 0b                	je     400948 <register_tm_clones+0x38>
  40093d:	5d                   	pop    %rbp
  40093e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400943:	ff e0                	jmpq   *%rax
  400945:	0f 1f 00             	nopl   (%rax)
  400948:	5d                   	pop    %rbp
  400949:	c3                   	retq   
  40094a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400950 <__do_global_dtors_aux>:
  400950:	80 3d 29 17 20 00 00 	cmpb   $0x0,0x201729(%rip)        # 602080 <completed.7585>
  400957:	75 11                	jne    40096a <__do_global_dtors_aux+0x1a>
  400959:	55                   	push   %rbp
  40095a:	48 89 e5             	mov    %rsp,%rbp
  40095d:	e8 6e ff ff ff       	callq  4008d0 <deregister_tm_clones>
  400962:	5d                   	pop    %rbp
  400963:	c6 05 16 17 20 00 01 	movb   $0x1,0x201716(%rip)        # 602080 <completed.7585>
  40096a:	f3 c3                	repz retq 
  40096c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400970 <frame_dummy>:
  400970:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400975:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400979:	75 05                	jne    400980 <frame_dummy+0x10>
  40097b:	eb 93                	jmp    400910 <register_tm_clones>
  40097d:	0f 1f 00             	nopl   (%rax)
  400980:	b8 00 00 00 00       	mov    $0x0,%eax
  400985:	48 85 c0             	test   %rax,%rax
  400988:	74 f1                	je     40097b <frame_dummy+0xb>
  40098a:	55                   	push   %rbp
  40098b:	48 89 e5             	mov    %rsp,%rbp
  40098e:	ff d0                	callq  *%rax
  400990:	5d                   	pop    %rbp
  400991:	e9 7a ff ff ff       	jmpq   400910 <register_tm_clones>
  400996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099d:	00 00 00 

00000000004009a0 <_Z13instantiationv>:
  4009a0:	41 56                	push   %r14
  4009a2:	41 55                	push   %r13
  4009a4:	45 31 ed             	xor    %r13d,%r13d
  4009a7:	41 54                	push   %r12
  4009a9:	55                   	push   %rbp
  4009aa:	53                   	push   %rbx
  4009ab:	48 83 ec 20          	sub    $0x20,%rsp
  4009af:	f2 0f 10 0d 91 09 00 	movsd  0x991(%rip),%xmm1        # 401348 <_IO_stdin_used+0x48>
  4009b6:	00 
  4009b7:	f2 0f 10 15 91 09 00 	movsd  0x991(%rip),%xmm2        # 401350 <_IO_stdin_used+0x50>
  4009be:	00 
  4009bf:	f2 0f 10 1d 91 09 00 	movsd  0x991(%rip),%xmm3        # 401358 <_IO_stdin_used+0x58>
  4009c6:	00 
  4009c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009ce:	00 00 
  4009d0:	f2 0f 11 5c 24 18    	movsd  %xmm3,0x18(%rsp)
  4009d6:	49 83 c5 08          	add    $0x8,%r13
  4009da:	f2 0f 11 54 24 10    	movsd  %xmm2,0x10(%rsp)
  4009e0:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
  4009e6:	e8 95 fd ff ff       	callq  400780 <rand@plt>
  4009eb:	89 c3                	mov    %eax,%ebx
  4009ed:	e8 8e fd ff ff       	callq  400780 <rand@plt>
  4009f2:	41 89 c4             	mov    %eax,%r12d
  4009f5:	e8 86 fd ff ff       	callq  400780 <rand@plt>
  4009fa:	89 c5                	mov    %eax,%ebp
  4009fc:	e8 7f fd ff ff       	callq  400780 <rand@plt>
  400a01:	41 89 c6             	mov    %eax,%r14d
  400a04:	e8 77 fd ff ff       	callq  400780 <rand@plt>
  400a09:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a0d:	49 c7 85 98 1a 61 00 	movq   $0x0,0x611a98(%r13)
  400a14:	00 00 00 00 
  400a18:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  400a1e:	49 c7 85 98 20 60 00 	movq   $0x0,0x602098(%r13)
  400a25:	00 00 00 00 
  400a29:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  400a2f:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  400a33:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  400a39:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a3d:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a41:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400a45:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400a49:	f2 41 0f 11 85 98 fc 	movsd  %xmm0,0x65fc98(%r13)
  400a50:	65 00 
  400a52:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a56:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400a5b:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a5f:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a63:	f2 0f 5c 05 f5 08 00 	subsd  0x8f5(%rip),%xmm0        # 401360 <_IO_stdin_used+0x60>
  400a6a:	00 
  400a6b:	f2 41 0f 11 85 98 02 	movsd  %xmm0,0x650298(%r13)
  400a72:	65 00 
  400a74:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a78:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400a7c:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a80:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a84:	f2 0f 5c 05 d4 08 00 	subsd  0x8d4(%rip),%xmm0        # 401360 <_IO_stdin_used+0x60>
  400a8b:	00 
  400a8c:	f2 41 0f 11 85 98 08 	movsd  %xmm0,0x640898(%r13)
  400a93:	64 00 
  400a95:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a99:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
  400a9e:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400aa2:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400aa6:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400aaa:	f2 41 0f 11 85 98 0e 	movsd  %xmm0,0x630e98(%r13)
  400ab1:	63 00 
  400ab3:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400ab7:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400abb:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400abf:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400ac3:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400ac7:	f2 41 0f 11 85 98 14 	movsd  %xmm0,0x621498(%r13)
  400ace:	62 00 
  400ad0:	49 81 fd 00 fa 00 00 	cmp    $0xfa00,%r13
  400ad7:	0f 85 f3 fe ff ff    	jne    4009d0 <_Z13instantiationv+0x30>
  400add:	48 83 c4 20          	add    $0x20,%rsp
  400ae1:	5b                   	pop    %rbx
  400ae2:	5d                   	pop    %rbp
  400ae3:	41 5c                	pop    %r12
  400ae5:	41 5d                	pop    %r13
  400ae7:	41 5e                	pop    %r14
  400ae9:	c3                   	retq   
  400aea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400af0 <_Z10run_simplev>:
  400af0:	66 0f 28 15 88 08 00 	movapd 0x888(%rip),%xmm2        # 401380 <_IO_stdin_used+0x80>
  400af7:	00 
  400af8:	ba 50 c3 00 00       	mov    $0xc350,%edx
  400afd:	0f 1f 00             	nopl   (%rax)
  400b00:	31 c0                	xor    %eax,%eax
  400b02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b08:	66 0f 28 88 a0 1a 61 	movapd 0x611aa0(%rax),%xmm1
  400b0f:	00 
  400b10:	48 83 c0 10          	add    $0x10,%rax
  400b14:	66 0f 28 80 90 20 60 	movapd 0x602090(%rax),%xmm0
  400b1b:	00 
  400b1c:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400b20:	66 0f 28 98 90 fc 65 	movapd 0x65fc90(%rax),%xmm3
  400b27:	00 
  400b28:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400b2c:	66 0f 5e cb          	divpd  %xmm3,%xmm1
  400b30:	66 0f 5e c3          	divpd  %xmm3,%xmm0
  400b34:	66 0f 58 88 90 0e 63 	addpd  0x630e90(%rax),%xmm1
  400b3b:	00 
  400b3c:	0f 29 88 90 0e 63 00 	movaps %xmm1,0x630e90(%rax)
  400b43:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400b47:	66 0f 58 88 90 02 65 	addpd  0x650290(%rax),%xmm1
  400b4e:	00 
  400b4f:	66 0f 58 80 90 14 62 	addpd  0x621490(%rax),%xmm0
  400b56:	00 
  400b57:	0f 29 88 90 02 65 00 	movaps %xmm1,0x650290(%rax)
  400b5e:	0f 29 80 90 14 62 00 	movaps %xmm0,0x621490(%rax)
  400b65:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400b69:	66 0f 58 80 90 08 64 	addpd  0x640890(%rax),%xmm0
  400b70:	00 
  400b71:	0f 29 80 90 08 64 00 	movaps %xmm0,0x640890(%rax)
  400b78:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400b7e:	75 88                	jne    400b08 <_Z10run_simplev+0x18>
  400b80:	83 ea 01             	sub    $0x1,%edx
  400b83:	0f 85 77 ff ff ff    	jne    400b00 <_Z10run_simplev+0x10>
  400b89:	f3 c3                	repz retq 
  400b8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b90 <_Z14run_simulationv>:
  400b90:	41 55                	push   %r13
  400b92:	41 54                	push   %r12
  400b94:	41 bd 05 00 00 00    	mov    $0x5,%r13d
  400b9a:	55                   	push   %rbp
  400b9b:	53                   	push   %rbx
  400b9c:	48 83 ec 38          	sub    $0x38,%rsp
  400ba0:	f2 0f 10 3d a8 07 00 	movsd  0x7a8(%rip),%xmm7        # 401350 <_IO_stdin_used+0x50>
  400ba7:	00 
  400ba8:	f2 0f 10 35 b8 07 00 	movsd  0x7b8(%rip),%xmm6        # 401368 <_IO_stdin_used+0x68>
  400baf:	00 
  400bb0:	f2 0f 10 2d b8 07 00 	movsd  0x7b8(%rip),%xmm5        # 401370 <_IO_stdin_used+0x70>
  400bb7:	00 
  400bb8:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400bbd:	31 f6                	xor    %esi,%esi
  400bbf:	bf a0 1a 61 00       	mov    $0x611aa0,%edi
  400bc4:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400bca:	31 ed                	xor    %ebp,%ebp
  400bcc:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400bd2:	f2 0f 11 7c 24 08    	movsd  %xmm7,0x8(%rsp)
  400bd8:	e8 33 fb ff ff       	callq  400710 <memset@plt>
  400bdd:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400be2:	31 f6                	xor    %esi,%esi
  400be4:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400be9:	e8 22 fb ff ff       	callq  400710 <memset@plt>
  400bee:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400bf4:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400bfa:	f2 0f 10 7c 24 08    	movsd  0x8(%rsp),%xmm7
  400c00:	31 db                	xor    %ebx,%ebx
  400c02:	45 31 e4             	xor    %r12d,%r12d
  400c05:	0f 1f 00             	nopl   (%rax)
  400c08:	4c 39 e5             	cmp    %r12,%rbp
  400c0b:	0f 84 a5 00 00 00    	je     400cb6 <_Z14run_simulationv+0x126>
  400c11:	f2 0f 10 0c ed a0 08 	movsd  0x6408a0(,%rbp,8),%xmm1
  400c18:	64 00 
  400c1a:	f2 0f 10 1c ed a0 02 	movsd  0x6502a0(,%rbp,8),%xmm3
  400c21:	65 00 
  400c23:	f2 0f 5c 8b a0 08 64 	subsd  0x6408a0(%rbx),%xmm1
  400c2a:	00 
  400c2b:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  400c32:	00 
  400c33:	66 44 0f 28 c1       	movapd %xmm1,%xmm8
  400c38:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400c3c:	66 41 0f 28 c0       	movapd %xmm8,%xmm0
  400c41:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400c45:	f2 41 0f 59 c0       	mulsd  %xmm8,%xmm0
  400c4a:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400c4e:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400c52:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400c56:	0f 8a 94 01 00 00    	jp     400df0 <_Z14run_simulationv+0x260>
  400c5c:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  400c63:	00 
  400c64:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400c68:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400c6c:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c70:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400c77:	65 00 
  400c79:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400c7d:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400c81:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400c85:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400c89:	f2 41 0f 59 c8       	mulsd  %xmm8,%xmm1
  400c8e:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400c92:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400c96:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  400c9d:	00 
  400c9e:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  400ca5:	00 
  400ca6:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  400cad:	00 
  400cae:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  400cb5:	00 
  400cb6:	49 83 c4 01          	add    $0x1,%r12
  400cba:	48 83 c3 08          	add    $0x8,%rbx
  400cbe:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400cc5:	0f 85 3d ff ff ff    	jne    400c08 <_Z14run_simulationv+0x78>
  400ccb:	48 83 c5 01          	add    $0x1,%rbp
  400ccf:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400cd6:	0f 85 24 ff ff ff    	jne    400c00 <_Z14run_simulationv+0x70>
  400cdc:	f2 0f 10 1d ac 06 00 	movsd  0x6ac(%rip),%xmm3        # 401390 <_IO_stdin_used+0x90>
  400ce3:	00 
  400ce4:	31 c0                	xor    %eax,%eax
  400ce6:	f2 0f 10 25 72 06 00 	movsd  0x672(%rip),%xmm4        # 401360 <_IO_stdin_used+0x60>
  400ced:	00 
  400cee:	66 44 0f 28 c3       	movapd %xmm3,%xmm8
  400cf3:	66 44 0f 28 cc       	movapd %xmm4,%xmm9
  400cf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400cff:	00 
  400d00:	f2 0f 10 88 a0 1a 61 	movsd  0x611aa0(%rax),%xmm1
  400d07:	00 
  400d08:	f2 0f 10 90 a0 fc 65 	movsd  0x65fca0(%rax),%xmm2
  400d0f:	00 
  400d10:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400d14:	f2 0f 10 80 a0 20 60 	movsd  0x6020a0(%rax),%xmm0
  400d1b:	00 
  400d1c:	f2 44 0f 10 1d 53 06 	movsd  0x653(%rip),%xmm11        # 401378 <_IO_stdin_used+0x78>
  400d23:	00 00 
  400d25:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400d29:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400d2d:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400d31:	f2 0f 58 88 a0 0e 63 	addsd  0x630ea0(%rax),%xmm1
  400d38:	00 
  400d39:	66 44 0f 28 d1       	movapd %xmm1,%xmm10
  400d3e:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400d45:	00 
  400d46:	f2 44 0f 59 d7       	mulsd  %xmm7,%xmm10
  400d4b:	f2 0f 58 80 a0 14 62 	addsd  0x6214a0(%rax),%xmm0
  400d52:	00 
  400d53:	f2 44 0f 58 90 a0 02 	addsd  0x6502a0(%rax),%xmm10
  400d5a:	65 00 
  400d5c:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400d60:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400d67:	00 
  400d68:	66 45 0f 2e da       	ucomisd %xmm10,%xmm11
  400d6d:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400d71:	f2 44 0f 11 90 a0 02 	movsd  %xmm10,0x6502a0(%rax)
  400d78:	65 00 
  400d7a:	f2 0f 58 90 a0 08 64 	addsd  0x6408a0(%rax),%xmm2
  400d81:	00 
  400d82:	f2 0f 11 90 a0 08 64 	movsd  %xmm2,0x6408a0(%rax)
  400d89:	00 
  400d8a:	77 07                	ja     400d93 <_Z14run_simulationv+0x203>
  400d8c:	66 44 0f 2e d4       	ucomisd %xmm4,%xmm10
  400d91:	76 0c                	jbe    400d9f <_Z14run_simulationv+0x20f>
  400d93:	66 0f 57 cb          	xorpd  %xmm3,%xmm1
  400d97:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400d9e:	00 
  400d9f:	f2 0f 10 0d d1 05 00 	movsd  0x5d1(%rip),%xmm1        # 401378 <_IO_stdin_used+0x78>
  400da6:	00 
  400da7:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  400dab:	77 07                	ja     400db4 <_Z14run_simulationv+0x224>
  400dad:	66 41 0f 2e d1       	ucomisd %xmm9,%xmm2
  400db2:	76 0d                	jbe    400dc1 <_Z14run_simulationv+0x231>
  400db4:	66 41 0f 57 c0       	xorpd  %xmm8,%xmm0
  400db9:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400dc0:	00 
  400dc1:	48 83 c0 08          	add    $0x8,%rax
  400dc5:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400dcb:	0f 85 2f ff ff ff    	jne    400d00 <_Z14run_simulationv+0x170>
  400dd1:	41 83 ed 01          	sub    $0x1,%r13d
  400dd5:	0f 85 dd fd ff ff    	jne    400bb8 <_Z14run_simulationv+0x28>
  400ddb:	48 83 c4 38          	add    $0x38,%rsp
  400ddf:	5b                   	pop    %rbx
  400de0:	5d                   	pop    %rbp
  400de1:	41 5c                	pop    %r12
  400de3:	41 5d                	pop    %r13
  400de5:	c3                   	retq   
  400de6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ded:	00 00 00 
  400df0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400df6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400dfc:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400e02:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400e09:	f2 0f 11 5c 24 08    	movsd  %xmm3,0x8(%rsp)
  400e0f:	e8 7c f9 ff ff       	callq  400790 <sqrt@plt>
  400e14:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400e1a:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400e1e:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400e24:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400e2a:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400e31:	f2 0f 10 5c 24 08    	movsd  0x8(%rsp),%xmm3
  400e37:	e9 20 fe ff ff       	jmpq   400c5c <_Z14run_simulationv+0xcc>
  400e3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400e40 <_Z12float_as_intf>:
  400e40:	66 0f 7e c0          	movd   %xmm0,%eax
  400e44:	c3                   	retq   
  400e45:	90                   	nop
  400e46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e4d:	00 00 00 

0000000000400e50 <_Z15Body_initializemddddd>:
  400e50:	f2 0f 11 04 fd a0 fc 	movsd  %xmm0,0x65fca0(,%rdi,8)
  400e57:	65 00 
  400e59:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400e5d:	f2 0f 11 0c fd a0 02 	movsd  %xmm1,0x6502a0(,%rdi,8)
  400e64:	65 00 
  400e66:	f2 0f 11 14 fd a0 08 	movsd  %xmm2,0x6408a0(,%rdi,8)
  400e6d:	64 00 
  400e6f:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  400e76:	63 00 
  400e78:	f2 0f 11 24 fd a0 14 	movsd  %xmm4,0x6214a0(,%rdi,8)
  400e7f:	62 00 
  400e81:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  400e88:	61 00 
  400e8a:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  400e91:	60 00 
  400e93:	c3                   	retq   
  400e94:	66 90                	xchg   %ax,%ax
  400e96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e9d:	00 00 00 

0000000000400ea0 <_Z14Body_add_forcemm>:
  400ea0:	48 39 f7             	cmp    %rsi,%rdi
  400ea3:	0f 84 be 00 00 00    	je     400f67 <_Z14Body_add_forcemm+0xc7>
  400ea9:	55                   	push   %rbp
  400eaa:	53                   	push   %rbx
  400eab:	48 89 f5             	mov    %rsi,%rbp
  400eae:	48 89 fb             	mov    %rdi,%rbx
  400eb1:	48 83 ec 18          	sub    $0x18,%rsp
  400eb5:	f2 0f 10 0c f5 a0 08 	movsd  0x6408a0(,%rsi,8),%xmm1
  400ebc:	64 00 
  400ebe:	f2 0f 10 1c f5 a0 02 	movsd  0x6502a0(,%rsi,8),%xmm3
  400ec5:	65 00 
  400ec7:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400ecb:	f2 0f 5c 1c fd a0 02 	subsd  0x6502a0(,%rdi,8),%xmm3
  400ed2:	65 00 
  400ed4:	f2 0f 5c 2c fd a0 08 	subsd  0x6408a0(,%rdi,8),%xmm5
  400edb:	64 00 
  400edd:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400ee1:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400ee5:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400ee9:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400eed:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400ef1:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400ef5:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400ef9:	7a 6e                	jp     400f69 <_Z14Body_add_forcemm+0xc9>
  400efb:	f2 0f 10 15 65 04 00 	movsd  0x465(%rip),%xmm2        # 401368 <_IO_stdin_used+0x68>
  400f02:	00 
  400f03:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400f07:	f2 0f 59 14 dd a0 fc 	mulsd  0x65fca0(,%rbx,8),%xmm2
  400f0e:	65 00 
  400f10:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400f14:	f2 0f 58 05 54 04 00 	addsd  0x454(%rip),%xmm0        # 401370 <_IO_stdin_used+0x70>
  400f1b:	00 
  400f1c:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400f23:	65 00 
  400f25:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400f29:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400f2d:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400f31:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  400f35:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400f39:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400f3d:	f2 0f 58 1c dd a0 1a 	addsd  0x611aa0(,%rbx,8),%xmm3
  400f44:	61 00 
  400f46:	f2 0f 11 1c dd a0 1a 	movsd  %xmm3,0x611aa0(,%rbx,8)
  400f4d:	61 00 
  400f4f:	f2 0f 58 0c dd a0 20 	addsd  0x6020a0(,%rbx,8),%xmm1
  400f56:	60 00 
  400f58:	f2 0f 11 0c dd a0 20 	movsd  %xmm1,0x6020a0(,%rbx,8)
  400f5f:	60 00 
  400f61:	48 83 c4 18          	add    $0x18,%rsp
  400f65:	5b                   	pop    %rbx
  400f66:	5d                   	pop    %rbp
  400f67:	f3 c3                	repz retq 
  400f69:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  400f6f:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400f74:	e8 17 f8 ff ff       	callq  400790 <sqrt@plt>
  400f79:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  400f7f:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400f83:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400f88:	e9 6e ff ff ff       	jmpq   400efb <_Z14Body_add_forcemm+0x5b>
  400f8d:	0f 1f 00             	nopl   (%rax)

0000000000400f90 <_Z21Body_add_force_to_allm>:
  400f90:	41 54                	push   %r12
  400f92:	55                   	push   %rbp
  400f93:	49 89 fc             	mov    %rdi,%r12
  400f96:	53                   	push   %rbx
  400f97:	31 ed                	xor    %ebp,%ebp
  400f99:	31 db                	xor    %ebx,%ebx
  400f9b:	48 83 ec 20          	sub    $0x20,%rsp
  400f9f:	f2 0f 10 3d c1 03 00 	movsd  0x3c1(%rip),%xmm7        # 401368 <_IO_stdin_used+0x68>
  400fa6:	00 
  400fa7:	f2 0f 10 35 c1 03 00 	movsd  0x3c1(%rip),%xmm6        # 401370 <_IO_stdin_used+0x70>
  400fae:	00 
  400faf:	90                   	nop
  400fb0:	49 39 ec             	cmp    %rbp,%r12
  400fb3:	0f 84 a0 00 00 00    	je     401059 <_Z21Body_add_force_to_allm+0xc9>
  400fb9:	f2 42 0f 10 0c e5 a0 	movsd  0x6408a0(,%r12,8),%xmm1
  400fc0:	08 64 00 
  400fc3:	f2 42 0f 10 1c e5 a0 	movsd  0x6502a0(,%r12,8),%xmm3
  400fca:	02 65 00 
  400fcd:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400fd1:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  400fd8:	00 
  400fd9:	f2 0f 5c ab a0 08 64 	subsd  0x6408a0(%rbx),%xmm5
  400fe0:	00 
  400fe1:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400fe5:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400fe9:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400fed:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400ff1:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400ff5:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400ff9:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400ffd:	7a 78                	jp     401077 <_Z21Body_add_force_to_allm+0xe7>
  400fff:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  401006:	00 
  401007:	66 0f 28 c4          	movapd %xmm4,%xmm0
  40100b:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  40100f:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  401013:	f2 42 0f 59 14 e5 a0 	mulsd  0x65fca0(,%r12,8),%xmm2
  40101a:	fc 65 00 
  40101d:	f2 0f 58 c6          	addsd  %xmm6,%xmm0
  401021:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  401025:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401029:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  40102d:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401031:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  401035:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  401039:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  401040:	00 
  401041:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  401048:	00 
  401049:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  401050:	00 
  401051:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  401058:	00 
  401059:	48 83 c5 01          	add    $0x1,%rbp
  40105d:	48 83 c3 08          	add    $0x8,%rbx
  401061:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  401068:	0f 85 42 ff ff ff    	jne    400fb0 <_Z21Body_add_force_to_allm+0x20>
  40106e:	48 83 c4 20          	add    $0x20,%rsp
  401072:	5b                   	pop    %rbx
  401073:	5d                   	pop    %rbp
  401074:	41 5c                	pop    %r12
  401076:	c3                   	retq   
  401077:	f2 0f 11 74 24 18    	movsd  %xmm6,0x18(%rsp)
  40107d:	f2 0f 11 7c 24 10    	movsd  %xmm7,0x10(%rsp)
  401083:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401089:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  40108e:	e8 fd f6 ff ff       	callq  400790 <sqrt@plt>
  401093:	f2 0f 10 74 24 18    	movsd  0x18(%rsp),%xmm6
  401099:	66 0f 28 e0          	movapd %xmm0,%xmm4
  40109d:	f2 0f 10 7c 24 10    	movsd  0x10(%rsp),%xmm7
  4010a3:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  4010a9:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  4010ae:	e9 4c ff ff ff       	jmpq   400fff <_Z21Body_add_force_to_allm+0x6f>
  4010b3:	0f 1f 00             	nopl   (%rax)
  4010b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4010bd:	00 00 00 

00000000004010c0 <_Z16Body_reset_forcem>:
  4010c0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4010c4:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  4010cb:	61 00 
  4010cd:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  4010d4:	60 00 
  4010d6:	c3                   	retq   
  4010d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4010de:	00 00 

00000000004010e0 <_Z11Body_updatemd>:
  4010e0:	f2 0f 10 14 fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm2
  4010e7:	61 00 
  4010e9:	f2 0f 10 1c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm3
  4010f0:	65 00 
  4010f2:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  4010f6:	f2 0f 10 0c fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm1
  4010fd:	60 00 
  4010ff:	f2 0f 10 25 71 02 00 	movsd  0x271(%rip),%xmm4        # 401378 <_IO_stdin_used+0x78>
  401106:	00 
  401107:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40110b:	f2 0f 5e d3          	divsd  %xmm3,%xmm2
  40110f:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  401113:	66 0f 28 d8          	movapd %xmm0,%xmm3
  401117:	f2 0f 58 14 fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm2
  40111e:	63 00 
  401120:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  401127:	63 00 
  401129:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  40112d:	f2 0f 58 0c fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm1
  401134:	62 00 
  401136:	f2 0f 58 1c fd a0 02 	addsd  0x6502a0(,%rdi,8),%xmm3
  40113d:	65 00 
  40113f:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  401143:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  40114a:	62 00 
  40114c:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  401150:	f2 0f 11 1c fd a0 02 	movsd  %xmm3,0x6502a0(,%rdi,8)
  401157:	65 00 
  401159:	f2 0f 58 04 fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm0
  401160:	64 00 
  401162:	f2 0f 11 04 fd a0 08 	movsd  %xmm0,0x6408a0(,%rdi,8)
  401169:	64 00 
  40116b:	77 0a                	ja     401177 <_Z11Body_updatemd+0x97>
  40116d:	66 0f 2e 1d eb 01 00 	ucomisd 0x1eb(%rip),%xmm3        # 401360 <_IO_stdin_used+0x60>
  401174:	00 
  401175:	76 15                	jbe    40118c <_Z11Body_updatemd+0xac>
  401177:	f2 0f 10 1d 11 02 00 	movsd  0x211(%rip),%xmm3        # 401390 <_IO_stdin_used+0x90>
  40117e:	00 
  40117f:	66 0f 57 d3          	xorpd  %xmm3,%xmm2
  401183:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  40118a:	63 00 
  40118c:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  401190:	77 0a                	ja     40119c <_Z11Body_updatemd+0xbc>
  401192:	66 0f 2e 05 c6 01 00 	ucomisd 0x1c6(%rip),%xmm0        # 401360 <_IO_stdin_used+0x60>
  401199:	00 
  40119a:	76 15                	jbe    4011b1 <_Z11Body_updatemd+0xd1>
  40119c:	f2 0f 10 05 ec 01 00 	movsd  0x1ec(%rip),%xmm0        # 401390 <_IO_stdin_used+0x90>
  4011a3:	00 
  4011a4:	66 0f 57 c8          	xorpd  %xmm0,%xmm1
  4011a8:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  4011af:	62 00 
  4011b1:	f3 c3                	repz retq 
  4011b3:	0f 1f 00             	nopl   (%rax)
  4011b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011bd:	00 00 00 

00000000004011c0 <_Z26Body_codegen_simple_updatemd>:
  4011c0:	f2 0f 10 14 fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm2
  4011c7:	60 00 
  4011c9:	f2 0f 10 1c fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm3
  4011d0:	61 00 
  4011d2:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  4011d6:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  4011dd:	65 00 
  4011df:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  4011e3:	f2 0f 5e d1          	divsd  %xmm1,%xmm2
  4011e7:	f2 0f 5e d9          	divsd  %xmm1,%xmm3
  4011eb:	f2 0f 58 14 fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm2
  4011f2:	62 00 
  4011f4:	66 0f 28 ca          	movapd %xmm2,%xmm1
  4011f8:	f2 0f 11 14 fd a0 14 	movsd  %xmm2,0x6214a0(,%rdi,8)
  4011ff:	62 00 
  401201:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401205:	f2 0f 58 1c fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm3
  40120c:	63 00 
  40120e:	f2 0f 58 0c fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm1
  401215:	64 00 
  401217:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  40121e:	63 00 
  401220:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  401224:	f2 0f 11 0c fd a0 08 	movsd  %xmm1,0x6408a0(,%rdi,8)
  40122b:	64 00 
  40122d:	f2 0f 58 1c fd a0 02 	addsd  0x6502a0(,%rdi,8),%xmm3
  401234:	65 00 
  401236:	f2 0f 11 1c fd a0 02 	movsd  %xmm3,0x6502a0(,%rdi,8)
  40123d:	65 00 
  40123f:	c3                   	retq   

0000000000401240 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401240:	55                   	push   %rbp
  401241:	53                   	push   %rbx
  401242:	48 89 fd             	mov    %rdi,%rbp
  401245:	48 83 ec 08          	sub    $0x8,%rsp
  401249:	e8 22 f5 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40124e:	48 89 c3             	mov    %rax,%rbx
  401251:	ff d5                	callq  *%rbp
  401253:	e8 18 f5 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401258:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40125f:	de 1b 43 
  401262:	48 29 d8             	sub    %rbx,%rax
  401265:	48 83 c4 08          	add    $0x8,%rsp
  401269:	48 89 c1             	mov    %rax,%rcx
  40126c:	48 f7 ea             	imul   %rdx
  40126f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401273:	5b                   	pop    %rbx
  401274:	5d                   	pop    %rbp
  401275:	48 c1 fa 12          	sar    $0x12,%rdx
  401279:	48 89 d0             	mov    %rdx,%rax
  40127c:	48 29 c8             	sub    %rcx,%rax
  40127f:	c3                   	retq   

0000000000401280 <__libc_csu_init>:
  401280:	41 57                	push   %r15
  401282:	41 56                	push   %r14
  401284:	41 89 ff             	mov    %edi,%r15d
  401287:	41 55                	push   %r13
  401289:	41 54                	push   %r12
  40128b:	4c 8d 25 56 0b 20 00 	lea    0x200b56(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  401292:	55                   	push   %rbp
  401293:	48 8d 2d 5e 0b 20 00 	lea    0x200b5e(%rip),%rbp        # 601df8 <__init_array_end>
  40129a:	53                   	push   %rbx
  40129b:	49 89 f6             	mov    %rsi,%r14
  40129e:	49 89 d5             	mov    %rdx,%r13
  4012a1:	4c 29 e5             	sub    %r12,%rbp
  4012a4:	48 83 ec 08          	sub    $0x8,%rsp
  4012a8:	48 c1 fd 03          	sar    $0x3,%rbp
  4012ac:	e8 2f f4 ff ff       	callq  4006e0 <_init>
  4012b1:	48 85 ed             	test   %rbp,%rbp
  4012b4:	74 20                	je     4012d6 <__libc_csu_init+0x56>
  4012b6:	31 db                	xor    %ebx,%ebx
  4012b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4012bf:	00 
  4012c0:	4c 89 ea             	mov    %r13,%rdx
  4012c3:	4c 89 f6             	mov    %r14,%rsi
  4012c6:	44 89 ff             	mov    %r15d,%edi
  4012c9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4012cd:	48 83 c3 01          	add    $0x1,%rbx
  4012d1:	48 39 eb             	cmp    %rbp,%rbx
  4012d4:	75 ea                	jne    4012c0 <__libc_csu_init+0x40>
  4012d6:	48 83 c4 08          	add    $0x8,%rsp
  4012da:	5b                   	pop    %rbx
  4012db:	5d                   	pop    %rbp
  4012dc:	41 5c                	pop    %r12
  4012de:	41 5d                	pop    %r13
  4012e0:	41 5e                	pop    %r14
  4012e2:	41 5f                	pop    %r15
  4012e4:	c3                   	retq   
  4012e5:	90                   	nop
  4012e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4012ed:	00 00 00 

00000000004012f0 <__libc_csu_fini>:
  4012f0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004012f4 <_fini>:
  4012f4:	48 83 ec 08          	sub    $0x8,%rsp
  4012f8:	48 83 c4 08          	add    $0x8,%rsp
  4012fc:	c3                   	retq   
