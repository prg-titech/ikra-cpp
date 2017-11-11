
bin/g++_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

00000000004006a8 <_init>:
  4006a8:	48 83 ec 08          	sub    $0x8,%rsp
  4006ac:	48 8b 05 45 19 20 00 	mov    0x201945(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4006b3:	48 85 c0             	test   %rax,%rax
  4006b6:	74 05                	je     4006bd <_init+0x15>
  4006b8:	e8 a3 00 00 00       	callq  400760 <sqrt@plt+0x10>
  4006bd:	48 83 c4 08          	add    $0x8,%rsp
  4006c1:	c3                   	retq   

Disassembly of section .plt:

00000000004006d0 <__printf_chk@plt-0x10>:
  4006d0:	ff 35 32 19 20 00    	pushq  0x201932(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006d6:	ff 25 34 19 20 00    	jmpq   *0x201934(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006e0 <__printf_chk@plt>:
  4006e0:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006e6:	68 00 00 00 00       	pushq  $0x0
  4006eb:	e9 e0 ff ff ff       	jmpq   4006d0 <_init+0x28>

00000000004006f0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006f0:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006f6:	68 01 00 00 00       	pushq  $0x1
  4006fb:	e9 d0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400700 <__libc_start_main@plt>:
  400700:	ff 25 22 19 20 00    	jmpq   *0x201922(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400706:	68 02 00 00 00       	pushq  $0x2
  40070b:	e9 c0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400710 <__cxa_atexit@plt>:
  400710:	ff 25 1a 19 20 00    	jmpq   *0x20191a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400716:	68 03 00 00 00       	pushq  $0x3
  40071b:	e9 b0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400720 <_ZNSt8ios_base4InitD1Ev@plt>:
  400720:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400726:	68 04 00 00 00       	pushq  $0x4
  40072b:	e9 a0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400730:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400736:	68 05 00 00 00       	pushq  $0x5
  40073b:	e9 90 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400740 <rand@plt>:
  400740:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400746:	68 06 00 00 00       	pushq  $0x6
  40074b:	e9 80 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400750 <sqrt@plt>:
  400750:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400756:	68 07 00 00 00       	pushq  $0x7
  40075b:	e9 70 ff ff ff       	jmpq   4006d0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400760 <.plt.got>:
  400760:	ff 25 92 18 20 00    	jmpq   *0x201892(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  400766:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400770 <main>:
  400770:	55                   	push   %rbp
  400771:	53                   	push   %rbx
  400772:	bf d0 09 40 00       	mov    $0x4009d0,%edi
  400777:	48 83 ec 08          	sub    $0x8,%rsp
  40077b:	e8 10 06 00 00       	callq  400d90 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400780:	bf 30 0b 40 00       	mov    $0x400b30,%edi
  400785:	48 89 c3             	mov    %rax,%rbx
  400788:	e8 03 06 00 00       	callq  400d90 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40078d:	bf 50 09 40 00       	mov    $0x400950,%edi
  400792:	48 89 c5             	mov    %rax,%rbp
  400795:	e8 f6 05 00 00       	callq  400d90 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40079a:	b9 a8 20 60 00       	mov    $0x6020a8,%ecx
  40079f:	49 89 c0             	mov    %rax,%r8
  4007a2:	bf a8 f6 66 00       	mov    $0x66f6a8,%edi
  4007a7:	ba 0b 00 00 00       	mov    $0xb,%edx
  4007ac:	be 8d 68 b7 6c       	mov    $0x6cb7688d,%esi
  4007b1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4007b8:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4007bc:	48 83 c1 38          	add    $0x38,%rcx
  4007c0:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4007c4:	f2 0f 5a 41 c8       	cvtsd2ss -0x38(%rcx),%xmm0
  4007c9:	66 41 0f 7e c1       	movd   %xmm0,%r9d
  4007ce:	f2 0f 5a 49 d0       	cvtsd2ss -0x30(%rcx),%xmm1
  4007d3:	44 01 ca             	add    %r9d,%edx
  4007d6:	66 41 0f 7e c9       	movd   %xmm1,%r9d
  4007db:	41 01 d1             	add    %edx,%r9d
  4007de:	44 89 c8             	mov    %r9d,%eax
  4007e1:	f7 ee                	imul   %esi
  4007e3:	44 89 c8             	mov    %r9d,%eax
  4007e6:	c1 f8 1f             	sar    $0x1f,%eax
  4007e9:	c1 fa 13             	sar    $0x13,%edx
  4007ec:	29 c2                	sub    %eax,%edx
  4007ee:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  4007f4:	41 29 d1             	sub    %edx,%r9d
  4007f7:	48 39 cf             	cmp    %rcx,%rdi
  4007fa:	44 89 ca             	mov    %r9d,%edx
  4007fd:	75 b9                	jne    4007b8 <main+0x48>
  4007ff:	48 89 e9             	mov    %rbp,%rcx
  400802:	48 89 da             	mov    %rbx,%rdx
  400805:	be 58 0e 40 00       	mov    $0x400e58,%esi
  40080a:	bf 01 00 00 00       	mov    $0x1,%edi
  40080f:	31 c0                	xor    %eax,%eax
  400811:	e8 ca fe ff ff       	callq  4006e0 <__printf_chk@plt>
  400816:	48 83 c4 08          	add    $0x8,%rsp
  40081a:	31 c0                	xor    %eax,%eax
  40081c:	5b                   	pop    %rbx
  40081d:	5d                   	pop    %rbp
  40081e:	c3                   	retq   
  40081f:	90                   	nop

0000000000400820 <_GLOBAL__sub_I__Z12float_as_intf>:
  400820:	48 83 ec 08          	sub    $0x8,%rsp
  400824:	bf a0 f0 67 00       	mov    $0x67f0a0,%edi
  400829:	e8 c2 fe ff ff       	callq  4006f0 <_ZNSt8ios_base4InitC1Ev@plt>
  40082e:	ba 60 20 60 00       	mov    $0x602060,%edx
  400833:	be a0 f0 67 00       	mov    $0x67f0a0,%esi
  400838:	bf 20 07 40 00       	mov    $0x400720,%edi
  40083d:	48 83 c4 08          	add    $0x8,%rsp
  400841:	e9 ca fe ff ff       	jmpq   400710 <__cxa_atexit@plt>
  400846:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40084d:	00 00 00 

0000000000400850 <_start>:
  400850:	31 ed                	xor    %ebp,%ebp
  400852:	49 89 d1             	mov    %rdx,%r9
  400855:	5e                   	pop    %rsi
  400856:	48 89 e2             	mov    %rsp,%rdx
  400859:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40085d:	50                   	push   %rax
  40085e:	54                   	push   %rsp
  40085f:	49 c7 c0 40 0e 40 00 	mov    $0x400e40,%r8
  400866:	48 c7 c1 d0 0d 40 00 	mov    $0x400dd0,%rcx
  40086d:	48 c7 c7 70 07 40 00 	mov    $0x400770,%rdi
  400874:	e8 87 fe ff ff       	callq  400700 <__libc_start_main@plt>
  400879:	f4                   	hlt    
  40087a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400880 <deregister_tm_clones>:
  400880:	b8 77 20 60 00       	mov    $0x602077,%eax
  400885:	55                   	push   %rbp
  400886:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  40088c:	48 83 f8 0e          	cmp    $0xe,%rax
  400890:	48 89 e5             	mov    %rsp,%rbp
  400893:	76 1b                	jbe    4008b0 <deregister_tm_clones+0x30>
  400895:	b8 00 00 00 00       	mov    $0x0,%eax
  40089a:	48 85 c0             	test   %rax,%rax
  40089d:	74 11                	je     4008b0 <deregister_tm_clones+0x30>
  40089f:	5d                   	pop    %rbp
  4008a0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4008a5:	ff e0                	jmpq   *%rax
  4008a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008ae:	00 00 
  4008b0:	5d                   	pop    %rbp
  4008b1:	c3                   	retq   
  4008b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4008b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008bd:	00 00 00 

00000000004008c0 <register_tm_clones>:
  4008c0:	be 70 20 60 00       	mov    $0x602070,%esi
  4008c5:	55                   	push   %rbp
  4008c6:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  4008cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 89 f0             	mov    %rsi,%rax
  4008d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4008db:	48 01 c6             	add    %rax,%rsi
  4008de:	48 d1 fe             	sar    %rsi
  4008e1:	74 15                	je     4008f8 <register_tm_clones+0x38>
  4008e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4008e8:	48 85 c0             	test   %rax,%rax
  4008eb:	74 0b                	je     4008f8 <register_tm_clones+0x38>
  4008ed:	5d                   	pop    %rbp
  4008ee:	bf 70 20 60 00       	mov    $0x602070,%edi
  4008f3:	ff e0                	jmpq   *%rax
  4008f5:	0f 1f 00             	nopl   (%rax)
  4008f8:	5d                   	pop    %rbp
  4008f9:	c3                   	retq   
  4008fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400900 <__do_global_dtors_aux>:
  400900:	80 3d 79 17 20 00 00 	cmpb   $0x0,0x201779(%rip)        # 602080 <completed.7585>
  400907:	75 11                	jne    40091a <__do_global_dtors_aux+0x1a>
  400909:	55                   	push   %rbp
  40090a:	48 89 e5             	mov    %rsp,%rbp
  40090d:	e8 6e ff ff ff       	callq  400880 <deregister_tm_clones>
  400912:	5d                   	pop    %rbp
  400913:	c6 05 66 17 20 00 01 	movb   $0x1,0x201766(%rip)        # 602080 <completed.7585>
  40091a:	f3 c3                	repz retq 
  40091c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400920 <frame_dummy>:
  400920:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400925:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400929:	75 05                	jne    400930 <frame_dummy+0x10>
  40092b:	eb 93                	jmp    4008c0 <register_tm_clones>
  40092d:	0f 1f 00             	nopl   (%rax)
  400930:	b8 00 00 00 00       	mov    $0x0,%eax
  400935:	48 85 c0             	test   %rax,%rax
  400938:	74 f1                	je     40092b <frame_dummy+0xb>
  40093a:	55                   	push   %rbp
  40093b:	48 89 e5             	mov    %rsp,%rbp
  40093e:	ff d0                	callq  *%rax
  400940:	5d                   	pop    %rbp
  400941:	e9 7a ff ff ff       	jmpq   4008c0 <register_tm_clones>
  400946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094d:	00 00 00 

0000000000400950 <_Z10run_simplev>:
  400950:	f2 0f 10 15 40 05 00 	movsd  0x540(%rip),%xmm2        # 400e98 <_IO_stdin_used+0x48>
  400957:	00 
  400958:	ba 50 c3 00 00       	mov    $0xc350,%edx
  40095d:	0f 1f 00             	nopl   (%rax)
  400960:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400965:	0f 1f 00             	nopl   (%rax)
  400968:	f2 0f 10 48 10       	movsd  0x10(%rax),%xmm1
  40096d:	48 83 c0 38          	add    $0x38,%rax
  400971:	f2 0f 10 40 e0       	movsd  -0x20(%rax),%xmm0
  400976:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  40097a:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  40097e:	f2 0f 10 58 b0       	movsd  -0x50(%rax),%xmm3
  400983:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400987:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  40098b:	f2 0f 58 48 c8       	addsd  -0x38(%rax),%xmm1
  400990:	f2 0f 11 48 c8       	movsd  %xmm1,-0x38(%rax)
  400995:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400999:	f2 0f 58 40 d0       	addsd  -0x30(%rax),%xmm0
  40099e:	f2 0f 58 48 b8       	addsd  -0x48(%rax),%xmm1
  4009a3:	f2 0f 11 40 d0       	movsd  %xmm0,-0x30(%rax)
  4009a8:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  4009ac:	f2 0f 11 48 b8       	movsd  %xmm1,-0x48(%rax)
  4009b1:	f2 0f 58 40 c0       	addsd  -0x40(%rax),%xmm0
  4009b6:	f2 0f 11 40 c0       	movsd  %xmm0,-0x40(%rax)
  4009bb:	48 3d b8 f6 66 00    	cmp    $0x66f6b8,%rax
  4009c1:	75 a5                	jne    400968 <_Z10run_simplev+0x18>
  4009c3:	83 ea 01             	sub    $0x1,%edx
  4009c6:	75 98                	jne    400960 <_Z10run_simplev+0x10>
  4009c8:	f3 c3                	repz retq 
  4009ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009d0 <_Z13instantiationv>:
  4009d0:	53                   	push   %rbx
  4009d1:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4009d6:	48 83 ec 20          	sub    $0x20,%rsp
  4009da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4009e0:	e8 5b fd ff ff       	callq  400740 <rand@plt>
  4009e5:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4009e9:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  4009ed:	f2 0f 5e 05 ab 04 00 	divsd  0x4ab(%rip),%xmm0        # 400ea0 <_IO_stdin_used+0x50>
  4009f4:	00 
  4009f5:	f2 0f 59 05 9b 04 00 	mulsd  0x49b(%rip),%xmm0        # 400e98 <_IO_stdin_used+0x48>
  4009fc:	00 
  4009fd:	f2 0f 58 05 93 04 00 	addsd  0x493(%rip),%xmm0        # 400e98 <_IO_stdin_used+0x48>
  400a04:	00 
  400a05:	f2 0f 59 05 9b 04 00 	mulsd  0x49b(%rip),%xmm0        # 400ea8 <_IO_stdin_used+0x58>
  400a0c:	00 
  400a0d:	f2 0f 11 04 24       	movsd  %xmm0,(%rsp)
  400a12:	e8 29 fd ff ff       	callq  400740 <rand@plt>
  400a17:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a1b:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a1f:	f2 0f 5e 05 79 04 00 	divsd  0x479(%rip),%xmm0        # 400ea0 <_IO_stdin_used+0x50>
  400a26:	00 
  400a27:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a2b:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400a2f:	f2 0f 5c 15 79 04 00 	subsd  0x479(%rip),%xmm2        # 400eb0 <_IO_stdin_used+0x60>
  400a36:	00 
  400a37:	f2 0f 11 54 24 08    	movsd  %xmm2,0x8(%rsp)
  400a3d:	e8 fe fc ff ff       	callq  400740 <rand@plt>
  400a42:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a46:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a4a:	f2 0f 5e 05 4e 04 00 	divsd  0x44e(%rip),%xmm0        # 400ea0 <_IO_stdin_used+0x50>
  400a51:	00 
  400a52:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a56:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400a5a:	f2 0f 5c 1d 4e 04 00 	subsd  0x44e(%rip),%xmm3        # 400eb0 <_IO_stdin_used+0x60>
  400a61:	00 
  400a62:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  400a68:	e8 d3 fc ff ff       	callq  400740 <rand@plt>
  400a6d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a71:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400a75:	f2 0f 5e 05 23 04 00 	divsd  0x423(%rip),%xmm0        # 400ea0 <_IO_stdin_used+0x50>
  400a7c:	00 
  400a7d:	f2 0f 5c 05 13 04 00 	subsd  0x413(%rip),%xmm0        # 400e98 <_IO_stdin_used+0x48>
  400a84:	00 
  400a85:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400a89:	f2 0f 5e 25 17 04 00 	divsd  0x417(%rip),%xmm4        # 400ea8 <_IO_stdin_used+0x58>
  400a90:	00 
  400a91:	f2 0f 11 64 24 18    	movsd  %xmm4,0x18(%rsp)
  400a97:	e8 a4 fc ff ff       	callq  400740 <rand@plt>
  400a9c:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aa0:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400aa4:	48 8b 05 bd 15 20 00 	mov    0x2015bd(%rip),%rax        # 602068 <Body_arena_head>
  400aab:	48 85 c0             	test   %rax,%rax
  400aae:	f2 0f 5e 05 ea 03 00 	divsd  0x3ea(%rip),%xmm0        # 400ea0 <_IO_stdin_used+0x50>
  400ab5:	00 
  400ab6:	f2 0f 5c 05 da 03 00 	subsd  0x3da(%rip),%xmm0        # 400e98 <_IO_stdin_used+0x48>
  400abd:	00 
  400abe:	f2 0f 5e 05 e2 03 00 	divsd  0x3e2(%rip),%xmm0        # 400ea8 <_IO_stdin_used+0x58>
  400ac5:	00 
  400ac6:	74 3f                	je     400b07 <_Z13instantiationv+0x137>
  400ac8:	f2 0f 11 40 20       	movsd  %xmm0,0x20(%rax)
  400acd:	48 c7 40 28 00 00 00 	movq   $0x0,0x28(%rax)
  400ad4:	00 
  400ad5:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  400adc:	00 
  400add:	f2 0f 10 0c 24       	movsd  (%rsp),%xmm1
  400ae2:	f2 0f 10 54 24 08    	movsd  0x8(%rsp),%xmm2
  400ae8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  400aee:	f2 0f 10 64 24 18    	movsd  0x18(%rsp),%xmm4
  400af4:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400af8:	f2 0f 11 50 08       	movsd  %xmm2,0x8(%rax)
  400afd:	f2 0f 11 58 10       	movsd  %xmm3,0x10(%rax)
  400b02:	f2 0f 11 60 18       	movsd  %xmm4,0x18(%rax)
  400b07:	48 83 c0 38          	add    $0x38,%rax
  400b0b:	83 eb 01             	sub    $0x1,%ebx
  400b0e:	48 89 05 53 15 20 00 	mov    %rax,0x201553(%rip)        # 602068 <Body_arena_head>
  400b15:	0f 85 c5 fe ff ff    	jne    4009e0 <_Z13instantiationv+0x10>
  400b1b:	48 83 c4 20          	add    $0x20,%rsp
  400b1f:	5b                   	pop    %rbx
  400b20:	c3                   	retq   
  400b21:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <_Z14run_simulationv>:
  400b30:	41 54                	push   %r12
  400b32:	55                   	push   %rbp
  400b33:	41 bc 05 00 00 00    	mov    $0x5,%r12d
  400b39:	53                   	push   %rbx
  400b3a:	48 83 ec 30          	sub    $0x30,%rsp
  400b3e:	f2 0f 10 3d 52 03 00 	movsd  0x352(%rip),%xmm7        # 400e98 <_IO_stdin_used+0x48>
  400b45:	00 
  400b46:	f2 44 0f 10 05 79 03 	movsd  0x379(%rip),%xmm8        # 400ec8 <_IO_stdin_used+0x78>
  400b4d:	00 00 
  400b4f:	f2 0f 10 35 61 03 00 	movsd  0x361(%rip),%xmm6        # 400eb8 <_IO_stdin_used+0x68>
  400b56:	00 
  400b57:	f2 0f 10 2d 61 03 00 	movsd  0x361(%rip),%xmm5        # 400ec0 <_IO_stdin_used+0x70>
  400b5e:	00 
  400b5f:	b8 c8 20 60 00       	mov    $0x6020c8,%eax
  400b64:	0f 1f 40 00          	nopl   0x0(%rax)
  400b68:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400b6f:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b76:	00 
  400b77:	48 83 c0 38          	add    $0x38,%rax
  400b7b:	48 3d c8 f6 66 00    	cmp    $0x66f6c8,%rax
  400b81:	75 e5                	jne    400b68 <_Z14run_simulationv+0x38>
  400b83:	bd a0 20 60 00       	mov    $0x6020a0,%ebp
  400b88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b8f:	00 
  400b90:	bb a0 20 60 00       	mov    $0x6020a0,%ebx
  400b95:	0f 1f 00             	nopl   (%rax)
  400b98:	48 39 dd             	cmp    %rbx,%rbp
  400b9b:	0f 84 83 00 00 00    	je     400c24 <_Z14run_simulationv+0xf4>
  400ba1:	f2 0f 10 4d 10       	movsd  0x10(%rbp),%xmm1
  400ba6:	f2 0f 10 5d 08       	movsd  0x8(%rbp),%xmm3
  400bab:	f2 0f 5c 4b 10       	subsd  0x10(%rbx),%xmm1
  400bb0:	f2 0f 5c 5b 08       	subsd  0x8(%rbx),%xmm3
  400bb5:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400bba:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400bbe:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400bc3:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400bc7:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400bcc:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400bd0:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400bd4:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400bd8:	0f 8a 42 01 00 00    	jp     400d20 <_Z14run_simulationv+0x1f0>
  400bde:	f2 0f 10 13          	movsd  (%rbx),%xmm2
  400be2:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400be6:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400bea:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400bee:	f2 0f 59 55 00       	mulsd  0x0(%rbp),%xmm2
  400bf3:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400bf7:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400bfb:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400bff:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400c03:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400c08:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400c0c:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400c10:	f2 0f 58 5b 28       	addsd  0x28(%rbx),%xmm3
  400c15:	f2 0f 11 5b 28       	movsd  %xmm3,0x28(%rbx)
  400c1a:	f2 0f 58 4b 30       	addsd  0x30(%rbx),%xmm1
  400c1f:	f2 0f 11 4b 30       	movsd  %xmm1,0x30(%rbx)
  400c24:	48 83 c3 38          	add    $0x38,%rbx
  400c28:	48 81 fb a0 f6 66 00 	cmp    $0x66f6a0,%rbx
  400c2f:	0f 85 63 ff ff ff    	jne    400b98 <_Z14run_simulationv+0x68>
  400c35:	48 83 c5 38          	add    $0x38,%rbp
  400c39:	48 81 fd a0 f6 66 00 	cmp    $0x66f6a0,%rbp
  400c40:	0f 85 4a ff ff ff    	jne    400b90 <_Z14run_simulationv+0x60>
  400c46:	f2 0f 10 25 82 02 00 	movsd  0x282(%rip),%xmm4        # 400ed0 <_IO_stdin_used+0x80>
  400c4d:	00 
  400c4e:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400c53:	f2 44 0f 10 0d 54 02 	movsd  0x254(%rip),%xmm9        # 400eb0 <_IO_stdin_used+0x60>
  400c5a:	00 00 
  400c5c:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400c61:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400c66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6d:	00 00 00 
  400c70:	f2 0f 10 48 10       	movsd  0x10(%rax),%xmm1
  400c75:	f2 0f 10 50 e8       	movsd  -0x18(%rax),%xmm2
  400c7a:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400c7e:	f2 0f 10 40 18       	movsd  0x18(%rax),%xmm0
  400c83:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400c87:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400c8b:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400c8f:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400c93:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400c97:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400c9b:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400c9f:	f2 0f 58 40 08       	addsd  0x8(%rax),%xmm0
  400ca4:	f2 0f 58 58 f0       	addsd  -0x10(%rax),%xmm3
  400ca9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400cad:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400cb2:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400cb7:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400cbb:	f2 0f 11 58 f0       	movsd  %xmm3,-0x10(%rax)
  400cc0:	f2 0f 58 50 f8       	addsd  -0x8(%rax),%xmm2
  400cc5:	f2 0f 11 50 f8       	movsd  %xmm2,-0x8(%rax)
  400cca:	77 07                	ja     400cd3 <_Z14run_simulationv+0x1a3>
  400ccc:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400cd1:	76 08                	jbe    400cdb <_Z14run_simulationv+0x1ab>
  400cd3:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400cd7:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400cdb:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400ce0:	77 07                	ja     400ce9 <_Z14run_simulationv+0x1b9>
  400ce2:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400ce7:	76 0a                	jbe    400cf3 <_Z14run_simulationv+0x1c3>
  400ce9:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400cee:	f2 0f 11 40 08       	movsd  %xmm0,0x8(%rax)
  400cf3:	48 83 c0 38          	add    $0x38,%rax
  400cf7:	48 3d b8 f6 66 00    	cmp    $0x66f6b8,%rax
  400cfd:	0f 85 6d ff ff ff    	jne    400c70 <_Z14run_simulationv+0x140>
  400d03:	41 83 ec 01          	sub    $0x1,%r12d
  400d07:	0f 85 52 fe ff ff    	jne    400b5f <_Z14run_simulationv+0x2f>
  400d0d:	48 83 c4 30          	add    $0x30,%rsp
  400d11:	5b                   	pop    %rbx
  400d12:	5d                   	pop    %rbp
  400d13:	41 5c                	pop    %r12
  400d15:	c3                   	retq   
  400d16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1d:	00 00 00 
  400d20:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400d26:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400d2c:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400d32:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400d39:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400d40:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400d45:	e8 06 fa ff ff       	callq  400750 <sqrt@plt>
  400d4a:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400d50:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400d54:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400d5a:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400d60:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400d67:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400d6e:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400d73:	e9 66 fe ff ff       	jmpq   400bde <_Z14run_simulationv+0xae>
  400d78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d7f:	00 

0000000000400d80 <_Z12float_as_intf>:
  400d80:	66 0f 7e c0          	movd   %xmm0,%eax
  400d84:	c3                   	retq   
  400d85:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d8c:	00 00 00 
  400d8f:	90                   	nop

0000000000400d90 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  400d90:	55                   	push   %rbp
  400d91:	53                   	push   %rbx
  400d92:	48 89 fd             	mov    %rdi,%rbp
  400d95:	48 83 ec 08          	sub    $0x8,%rsp
  400d99:	e8 92 f9 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d9e:	48 89 c3             	mov    %rax,%rbx
  400da1:	ff d5                	callq  *%rbp
  400da3:	e8 88 f9 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400da8:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  400daf:	de 1b 43 
  400db2:	48 29 d8             	sub    %rbx,%rax
  400db5:	48 83 c4 08          	add    $0x8,%rsp
  400db9:	48 89 c1             	mov    %rax,%rcx
  400dbc:	48 f7 ea             	imul   %rdx
  400dbf:	48 c1 f9 3f          	sar    $0x3f,%rcx
  400dc3:	5b                   	pop    %rbx
  400dc4:	5d                   	pop    %rbp
  400dc5:	48 c1 fa 12          	sar    $0x12,%rdx
  400dc9:	48 89 d0             	mov    %rdx,%rax
  400dcc:	48 29 c8             	sub    %rcx,%rax
  400dcf:	c3                   	retq   

0000000000400dd0 <__libc_csu_init>:
  400dd0:	41 57                	push   %r15
  400dd2:	41 56                	push   %r14
  400dd4:	41 89 ff             	mov    %edi,%r15d
  400dd7:	41 55                	push   %r13
  400dd9:	41 54                	push   %r12
  400ddb:	4c 8d 25 06 10 20 00 	lea    0x201006(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  400de2:	55                   	push   %rbp
  400de3:	48 8d 2d 0e 10 20 00 	lea    0x20100e(%rip),%rbp        # 601df8 <__init_array_end>
  400dea:	53                   	push   %rbx
  400deb:	49 89 f6             	mov    %rsi,%r14
  400dee:	49 89 d5             	mov    %rdx,%r13
  400df1:	4c 29 e5             	sub    %r12,%rbp
  400df4:	48 83 ec 08          	sub    $0x8,%rsp
  400df8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dfc:	e8 a7 f8 ff ff       	callq  4006a8 <_init>
  400e01:	48 85 ed             	test   %rbp,%rbp
  400e04:	74 20                	je     400e26 <__libc_csu_init+0x56>
  400e06:	31 db                	xor    %ebx,%ebx
  400e08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e0f:	00 
  400e10:	4c 89 ea             	mov    %r13,%rdx
  400e13:	4c 89 f6             	mov    %r14,%rsi
  400e16:	44 89 ff             	mov    %r15d,%edi
  400e19:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e1d:	48 83 c3 01          	add    $0x1,%rbx
  400e21:	48 39 eb             	cmp    %rbp,%rbx
  400e24:	75 ea                	jne    400e10 <__libc_csu_init+0x40>
  400e26:	48 83 c4 08          	add    $0x8,%rsp
  400e2a:	5b                   	pop    %rbx
  400e2b:	5d                   	pop    %rbp
  400e2c:	41 5c                	pop    %r12
  400e2e:	41 5d                	pop    %r13
  400e30:	41 5e                	pop    %r14
  400e32:	41 5f                	pop    %r15
  400e34:	c3                   	retq   
  400e35:	90                   	nop
  400e36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e3d:	00 00 00 

0000000000400e40 <__libc_csu_fini>:
  400e40:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e44 <_fini>:
  400e44:	48 83 ec 08          	sub    $0x8,%rsp
  400e48:	48 83 c4 08          	add    $0x8,%rsp
  400e4c:	c3                   	retq   
