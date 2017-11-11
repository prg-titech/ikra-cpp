
bin/clang++-3.8_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

0000000000400698 <_init>:
  400698:	48 83 ec 08          	sub    $0x8,%rsp
  40069c:	48 8b 05 55 19 20 00 	mov    0x201955(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4006a3:	48 85 c0             	test   %rax,%rax
  4006a6:	74 05                	je     4006ad <_init+0x15>
  4006a8:	e8 a3 00 00 00       	callq  400750 <sqrt@plt+0x10>
  4006ad:	48 83 c4 08          	add    $0x8,%rsp
  4006b1:	c3                   	retq   

Disassembly of section .plt:

00000000004006c0 <printf@plt-0x10>:
  4006c0:	ff 35 42 19 20 00    	pushq  0x201942(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006c6:	ff 25 44 19 20 00    	jmpq   *0x201944(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006d0 <printf@plt>:
  4006d0:	ff 25 42 19 20 00    	jmpq   *0x201942(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006d6:	68 00 00 00 00       	pushq  $0x0
  4006db:	e9 e0 ff ff ff       	jmpq   4006c0 <_init+0x28>

00000000004006e0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006e0:	ff 25 3a 19 20 00    	jmpq   *0x20193a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006e6:	68 01 00 00 00       	pushq  $0x1
  4006eb:	e9 d0 ff ff ff       	jmpq   4006c0 <_init+0x28>

00000000004006f0 <__libc_start_main@plt>:
  4006f0:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006f6:	68 02 00 00 00       	pushq  $0x2
  4006fb:	e9 c0 ff ff ff       	jmpq   4006c0 <_init+0x28>

0000000000400700 <__cxa_atexit@plt>:
  400700:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400706:	68 03 00 00 00       	pushq  $0x3
  40070b:	e9 b0 ff ff ff       	jmpq   4006c0 <_init+0x28>

0000000000400710 <_ZNSt8ios_base4InitD1Ev@plt>:
  400710:	ff 25 22 19 20 00    	jmpq   *0x201922(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400716:	68 04 00 00 00       	pushq  $0x4
  40071b:	e9 a0 ff ff ff       	jmpq   4006c0 <_init+0x28>

0000000000400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400720:	ff 25 1a 19 20 00    	jmpq   *0x20191a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400726:	68 05 00 00 00       	pushq  $0x5
  40072b:	e9 90 ff ff ff       	jmpq   4006c0 <_init+0x28>

0000000000400730 <rand@plt>:
  400730:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400736:	68 06 00 00 00       	pushq  $0x6
  40073b:	e9 80 ff ff ff       	jmpq   4006c0 <_init+0x28>

0000000000400740 <sqrt@plt>:
  400740:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400746:	68 07 00 00 00       	pushq  $0x7
  40074b:	e9 70 ff ff ff       	jmpq   4006c0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400750 <.plt.got>:
  400750:	ff 25 a2 18 20 00    	jmpq   *0x2018a2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400756:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400760 <_GLOBAL__sub_I_aos.cc>:
  400760:	50                   	push   %rax
  400761:	bf 80 20 60 00       	mov    $0x602080,%edi
  400766:	e8 75 ff ff ff       	callq  4006e0 <_ZNSt8ios_base4InitC1Ev@plt>
  40076b:	bf 10 07 40 00       	mov    $0x400710,%edi
  400770:	be 80 20 60 00       	mov    $0x602080,%esi
  400775:	ba 60 20 60 00       	mov    $0x602060,%edx
  40077a:	58                   	pop    %rax
  40077b:	e9 80 ff ff ff       	jmpq   400700 <__cxa_atexit@plt>

0000000000400780 <_start>:
  400780:	31 ed                	xor    %ebp,%ebp
  400782:	49 89 d1             	mov    %rdx,%r9
  400785:	5e                   	pop    %rsi
  400786:	48 89 e2             	mov    %rsp,%rdx
  400789:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40078d:	50                   	push   %rax
  40078e:	54                   	push   %rsp
  40078f:	49 c7 c0 d0 0e 40 00 	mov    $0x400ed0,%r8
  400796:	48 c7 c1 60 0e 40 00 	mov    $0x400e60,%rcx
  40079d:	48 c7 c7 d0 0c 40 00 	mov    $0x400cd0,%rdi
  4007a4:	e8 47 ff ff ff       	callq  4006f0 <__libc_start_main@plt>
  4007a9:	f4                   	hlt    
  4007aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007b0 <deregister_tm_clones>:
  4007b0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4007b5:	55                   	push   %rbp
  4007b6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4007bc:	48 83 f8 0e          	cmp    $0xe,%rax
  4007c0:	48 89 e5             	mov    %rsp,%rbp
  4007c3:	76 1b                	jbe    4007e0 <deregister_tm_clones+0x30>
  4007c5:	b8 00 00 00 00       	mov    $0x0,%eax
  4007ca:	48 85 c0             	test   %rax,%rax
  4007cd:	74 11                	je     4007e0 <deregister_tm_clones+0x30>
  4007cf:	5d                   	pop    %rbp
  4007d0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4007d5:	ff e0                	jmpq   *%rax
  4007d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007de:	00 00 
  4007e0:	5d                   	pop    %rbp
  4007e1:	c3                   	retq   
  4007e2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007ed:	00 00 00 

00000000004007f0 <register_tm_clones>:
  4007f0:	be 70 20 60 00       	mov    $0x602070,%esi
  4007f5:	55                   	push   %rbp
  4007f6:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  4007fd:	48 c1 fe 03          	sar    $0x3,%rsi
  400801:	48 89 e5             	mov    %rsp,%rbp
  400804:	48 89 f0             	mov    %rsi,%rax
  400807:	48 c1 e8 3f          	shr    $0x3f,%rax
  40080b:	48 01 c6             	add    %rax,%rsi
  40080e:	48 d1 fe             	sar    %rsi
  400811:	74 15                	je     400828 <register_tm_clones+0x38>
  400813:	b8 00 00 00 00       	mov    $0x0,%eax
  400818:	48 85 c0             	test   %rax,%rax
  40081b:	74 0b                	je     400828 <register_tm_clones+0x38>
  40081d:	5d                   	pop    %rbp
  40081e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400823:	ff e0                	jmpq   *%rax
  400825:	0f 1f 00             	nopl   (%rax)
  400828:	5d                   	pop    %rbp
  400829:	c3                   	retq   
  40082a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400830 <__do_global_dtors_aux>:
  400830:	80 3d 39 18 20 00 00 	cmpb   $0x0,0x201839(%rip)        # 602070 <__TMC_END__>
  400837:	75 11                	jne    40084a <__do_global_dtors_aux+0x1a>
  400839:	55                   	push   %rbp
  40083a:	48 89 e5             	mov    %rsp,%rbp
  40083d:	e8 6e ff ff ff       	callq  4007b0 <deregister_tm_clones>
  400842:	5d                   	pop    %rbp
  400843:	c6 05 26 18 20 00 01 	movb   $0x1,0x201826(%rip)        # 602070 <__TMC_END__>
  40084a:	f3 c3                	repz retq 
  40084c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400850 <frame_dummy>:
  400850:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400855:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400859:	75 05                	jne    400860 <frame_dummy+0x10>
  40085b:	eb 93                	jmp    4007f0 <register_tm_clones>
  40085d:	0f 1f 00             	nopl   (%rax)
  400860:	b8 00 00 00 00       	mov    $0x0,%eax
  400865:	48 85 c0             	test   %rax,%rax
  400868:	74 f1                	je     40085b <frame_dummy+0xb>
  40086a:	55                   	push   %rbp
  40086b:	48 89 e5             	mov    %rsp,%rbp
  40086e:	ff d0                	callq  *%rax
  400870:	5d                   	pop    %rbp
  400871:	e9 7a ff ff ff       	jmpq   4007f0 <register_tm_clones>
  400876:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40087d:	00 00 00 

0000000000400880 <_Z12float_as_intf>:
  400880:	66 0f 7e c0          	movd   %xmm0,%eax
  400884:	c3                   	retq   
  400885:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40088c:	00 00 00 00 

0000000000400890 <_Z13instantiationv>:
  400890:	53                   	push   %rbx
  400891:	48 83 ec 30          	sub    $0x30,%rsp
  400895:	bb 80 25 00 00       	mov    $0x2580,%ebx
  40089a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4008a0:	e8 8b fe ff ff       	callq  400730 <rand@plt>
  4008a5:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008a8:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008ac:	f2 0f 10 05 34 06 00 	movsd  0x634(%rip),%xmm0        # 400ee8 <_IO_stdin_used+0x8>
  4008b3:	00 
  4008b4:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  4008b8:	f2 0f 10 05 30 06 00 	movsd  0x630(%rip),%xmm0        # 400ef0 <_IO_stdin_used+0x10>
  4008bf:	00 
  4008c0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008c4:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4008c8:	f2 0f 10 05 28 06 00 	movsd  0x628(%rip),%xmm0        # 400ef8 <_IO_stdin_used+0x18>
  4008cf:	00 
  4008d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008d4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  4008da:	e8 51 fe ff ff       	callq  400730 <rand@plt>
  4008df:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008e2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008e6:	f2 0f 5e 0d fa 05 00 	divsd  0x5fa(%rip),%xmm1        # 400ee8 <_IO_stdin_used+0x8>
  4008ed:	00 
  4008ee:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  4008f2:	f2 0f 10 05 06 06 00 	movsd  0x606(%rip),%xmm0        # 400f00 <_IO_stdin_used+0x20>
  4008f9:	00 
  4008fa:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4008fe:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400904:	e8 27 fe ff ff       	callq  400730 <rand@plt>
  400909:	0f 57 c0             	xorps  %xmm0,%xmm0
  40090c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400910:	f2 0f 5e 05 d0 05 00 	divsd  0x5d0(%rip),%xmm0        # 400ee8 <_IO_stdin_used+0x8>
  400917:	00 
  400918:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40091c:	f2 0f 58 05 dc 05 00 	addsd  0x5dc(%rip),%xmm0        # 400f00 <_IO_stdin_used+0x20>
  400923:	00 
  400924:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40092a:	e8 01 fe ff ff       	callq  400730 <rand@plt>
  40092f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400932:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400936:	f2 0f 5e 0d aa 05 00 	divsd  0x5aa(%rip),%xmm1        # 400ee8 <_IO_stdin_used+0x8>
  40093d:	00 
  40093e:	f2 0f 10 05 c2 05 00 	movsd  0x5c2(%rip),%xmm0        # 400f08 <_IO_stdin_used+0x28>
  400945:	00 
  400946:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40094a:	f2 0f 5e 0d a6 05 00 	divsd  0x5a6(%rip),%xmm1        # 400ef8 <_IO_stdin_used+0x18>
  400951:	00 
  400952:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400958:	e8 d3 fd ff ff       	callq  400730 <rand@plt>
  40095d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400960:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400964:	f2 0f 5e 05 7c 05 00 	divsd  0x57c(%rip),%xmm0        # 400ee8 <_IO_stdin_used+0x8>
  40096b:	00 
  40096c:	f2 0f 58 05 94 05 00 	addsd  0x594(%rip),%xmm0        # 400f08 <_IO_stdin_used+0x28>
  400973:	00 
  400974:	f2 0f 5e 05 7c 05 00 	divsd  0x57c(%rip),%xmm0        # 400ef8 <_IO_stdin_used+0x18>
  40097b:	00 
  40097c:	48 8b 05 e5 16 20 00 	mov    0x2016e5(%rip),%rax        # 602068 <Body_arena_head>
  400983:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  400989:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  40098d:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  400993:	f2 0f 11 48 08       	movsd  %xmm1,0x8(%rax)
  400998:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  40099e:	f2 0f 11 48 10       	movsd  %xmm1,0x10(%rax)
  4009a3:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  4009a9:	f2 0f 11 48 18       	movsd  %xmm1,0x18(%rax)
  4009ae:	f2 0f 11 40 20       	movsd  %xmm0,0x20(%rax)
  4009b3:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
  4009b7:	66 0f 11 40 28       	movupd %xmm0,0x28(%rax)
  4009bc:	48 83 c0 38          	add    $0x38,%rax
  4009c0:	48 89 05 a1 16 20 00 	mov    %rax,0x2016a1(%rip)        # 602068 <Body_arena_head>
  4009c7:	ff cb                	dec    %ebx
  4009c9:	0f 85 d1 fe ff ff    	jne    4008a0 <_Z13instantiationv+0x10>
  4009cf:	48 83 c4 30          	add    $0x30,%rsp
  4009d3:	5b                   	pop    %rbx
  4009d4:	c3                   	retq   
  4009d5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4009dc:	00 00 00 00 

00000000004009e0 <_Z14run_simulationv>:
  4009e0:	55                   	push   %rbp
  4009e1:	41 57                	push   %r15
  4009e3:	41 56                	push   %r14
  4009e5:	41 55                	push   %r13
  4009e7:	41 54                	push   %r12
  4009e9:	53                   	push   %rbx
  4009ea:	48 83 ec 28          	sub    $0x28,%rsp
  4009ee:	45 31 f6             	xor    %r14d,%r14d
  4009f1:	f2 0f 10 25 17 05 00 	movsd  0x517(%rip),%xmm4        # 400f10 <_IO_stdin_used+0x30>
  4009f8:	00 
  4009f9:	f2 0f 10 2d 17 05 00 	movsd  0x517(%rip),%xmm5        # 400f18 <_IO_stdin_used+0x38>
  400a00:	00 
  400a01:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400a08:	0f 1f 84 00 00 00 00 
  400a0f:	00 
  400a10:	31 c0                	xor    %eax,%eax
  400a12:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400a19:	1f 84 00 00 00 00 00 
  400a20:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
  400a24:	66 0f 11 80 b8 20 60 	movupd %xmm0,0x6020b8(%rax)
  400a2b:	00 
  400a2c:	66 0f 11 80 f0 20 60 	movupd %xmm0,0x6020f0(%rax)
  400a33:	00 
  400a34:	66 0f 11 80 28 21 60 	movupd %xmm0,0x602128(%rax)
  400a3b:	00 
  400a3c:	66 0f 11 80 60 21 60 	movupd %xmm0,0x602160(%rax)
  400a43:	00 
  400a44:	66 0f 11 80 98 21 60 	movupd %xmm0,0x602198(%rax)
  400a4b:	00 
  400a4c:	66 0f 11 80 d0 21 60 	movupd %xmm0,0x6021d0(%rax)
  400a53:	00 
  400a54:	66 0f 11 80 08 22 60 	movupd %xmm0,0x602208(%rax)
  400a5b:	00 
  400a5c:	66 0f 11 80 40 22 60 	movupd %xmm0,0x602240(%rax)
  400a63:	00 
  400a64:	48 05 c0 01 00 00    	add    $0x1c0,%rax
  400a6a:	48 3d 00 34 08 00    	cmp    $0x83400,%rax
  400a70:	41 bf 00 00 00 00    	mov    $0x0,%r15d
  400a76:	41 bc 00 00 00 00    	mov    $0x0,%r12d
  400a7c:	75 a2                	jne    400a20 <_Z14run_simulationv+0x40>
  400a7e:	66 90                	xchg   %ax,%ax
  400a80:	4d 6b ec 38          	imul   $0x38,%r12,%r13
  400a84:	48 c7 c3 00 cc f7 ff 	mov    $0xfffffffffff7cc00,%rbx
  400a8b:	4c 89 fd             	mov    %r15,%rbp
  400a8e:	66 90                	xchg   %ax,%ax
  400a90:	48 85 ed             	test   %rbp,%rbp
  400a93:	0f 84 a7 00 00 00    	je     400b40 <_Z14run_simulationv+0x160>
  400a99:	66 41 0f 10 9d 98 20 	movupd 0x602098(%r13),%xmm3
  400aa0:	60 00 
  400aa2:	66 0f 10 83 98 54 68 	movupd 0x685498(%rbx),%xmm0
  400aa9:	00 
  400aaa:	66 0f 5c d8          	subpd  %xmm0,%xmm3
  400aae:	66 0f 28 c3          	movapd %xmm3,%xmm0
  400ab2:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400ab6:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400aba:	66 0f c6 c9 01       	shufpd $0x1,%xmm1,%xmm1
  400abf:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400ac3:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400ac7:	0f 57 c0             	xorps  %xmm0,%xmm0
  400aca:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400ace:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400ad2:	7b 25                	jnp    400af9 <_Z14run_simulationv+0x119>
  400ad4:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400ad8:	66 0f 29 5c 24 10    	movapd %xmm3,0x10(%rsp)
  400ade:	e8 5d fc ff ff       	callq  400740 <sqrt@plt>
  400ae3:	66 0f 28 5c 24 10    	movapd 0x10(%rsp),%xmm3
  400ae9:	f2 0f 10 2d 27 04 00 	movsd  0x427(%rip),%xmm5        # 400f18 <_IO_stdin_used+0x38>
  400af0:	00 
  400af1:	f2 0f 10 25 17 04 00 	movsd  0x417(%rip),%xmm4        # 400f10 <_IO_stdin_used+0x30>
  400af8:	00 
  400af9:	f2 0f 10 8b 90 54 68 	movsd  0x685490(%rbx),%xmm1
  400b00:	00 
  400b01:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b05:	f2 41 0f 59 8d 90 20 	mulsd  0x602090(%r13),%xmm1
  400b0c:	60 00 
  400b0e:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400b12:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400b16:	f2 0f 58 d5          	addsd  %xmm5,%xmm2
  400b1a:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400b1e:	0f 16 c9             	movlhps %xmm1,%xmm1
  400b21:	66 0f 59 d9          	mulpd  %xmm1,%xmm3
  400b25:	0f 16 c0             	movlhps %xmm0,%xmm0
  400b28:	66 0f 5e d8          	divpd  %xmm0,%xmm3
  400b2c:	66 0f 10 83 b8 54 68 	movupd 0x6854b8(%rbx),%xmm0
  400b33:	00 
  400b34:	66 0f 58 c3          	addpd  %xmm3,%xmm0
  400b38:	66 0f 11 83 b8 54 68 	movupd %xmm0,0x6854b8(%rbx)
  400b3f:	00 
  400b40:	48 83 c5 c8          	add    $0xffffffffffffffc8,%rbp
  400b44:	48 83 c3 38          	add    $0x38,%rbx
  400b48:	0f 85 42 ff ff ff    	jne    400a90 <_Z14run_simulationv+0xb0>
  400b4e:	49 ff c4             	inc    %r12
  400b51:	49 83 c7 38          	add    $0x38,%r15
  400b55:	48 c7 c0 00 cc f7 ff 	mov    $0xfffffffffff7cc00,%rax
  400b5c:	49 81 fc 80 25 00 00 	cmp    $0x2580,%r12
  400b63:	0f 85 17 ff ff ff    	jne    400a80 <_Z14run_simulationv+0xa0>
  400b69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400b70:	66 0f 10 88 b8 54 68 	movupd 0x6854b8(%rax),%xmm1
  400b77:	00 
  400b78:	66 0f 28 05 c0 03 00 	movapd 0x3c0(%rip),%xmm0        # 400f40 <_IO_stdin_used+0x60>
  400b7f:	00 
  400b80:	66 0f 28 d8          	movapd %xmm0,%xmm3
  400b84:	66 0f 59 cb          	mulpd  %xmm3,%xmm1
  400b88:	f2 0f 10 80 90 54 68 	movsd  0x685490(%rax),%xmm0
  400b8f:	00 
  400b90:	0f 16 c0             	movlhps %xmm0,%xmm0
  400b93:	66 0f 5e c8          	divpd  %xmm0,%xmm1
  400b97:	66 0f 10 80 a8 54 68 	movupd 0x6854a8(%rax),%xmm0
  400b9e:	00 
  400b9f:	66 0f 58 c1          	addpd  %xmm1,%xmm0
  400ba3:	66 0f 11 80 a8 54 68 	movupd %xmm0,0x6854a8(%rax)
  400baa:	00 
  400bab:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400baf:	66 0f 59 d3          	mulpd  %xmm3,%xmm2
  400bb3:	66 0f 10 88 98 54 68 	movupd 0x685498(%rax),%xmm1
  400bba:	00 
  400bbb:	66 0f 58 ca          	addpd  %xmm2,%xmm1
  400bbf:	66 0f 11 88 98 54 68 	movupd %xmm1,0x685498(%rax)
  400bc6:	00 
  400bc7:	f2 0f 10 15 31 03 00 	movsd  0x331(%rip),%xmm2        # 400f00 <_IO_stdin_used+0x20>
  400bce:	00 
  400bcf:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  400bd3:	77 0a                	ja     400bdf <_Z14run_simulationv+0x1ff>
  400bd5:	66 0f 2e 0d 43 03 00 	ucomisd 0x343(%rip),%xmm1        # 400f20 <_IO_stdin_used+0x40>
  400bdc:	00 
  400bdd:	76 14                	jbe    400bf3 <_Z14run_simulationv+0x213>
  400bdf:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400be3:	66 0f 57 15 65 03 00 	xorpd  0x365(%rip),%xmm2        # 400f50 <_IO_stdin_used+0x70>
  400bea:	00 
  400beb:	66 0f 13 90 a8 54 68 	movlpd %xmm2,0x6854a8(%rax)
  400bf2:	00 
  400bf3:	66 0f c6 c9 01       	shufpd $0x1,%xmm1,%xmm1
  400bf8:	f2 0f 10 15 00 03 00 	movsd  0x300(%rip),%xmm2        # 400f00 <_IO_stdin_used+0x20>
  400bff:	00 
  400c00:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  400c04:	77 0a                	ja     400c10 <_Z14run_simulationv+0x230>
  400c06:	66 0f 2e 0d 12 03 00 	ucomisd 0x312(%rip),%xmm1        # 400f20 <_IO_stdin_used+0x40>
  400c0d:	00 
  400c0e:	76 15                	jbe    400c25 <_Z14run_simulationv+0x245>
  400c10:	66 0f c6 c0 01       	shufpd $0x1,%xmm0,%xmm0
  400c15:	66 0f 57 05 33 03 00 	xorpd  0x333(%rip),%xmm0        # 400f50 <_IO_stdin_used+0x70>
  400c1c:	00 
  400c1d:	66 0f 13 80 b0 54 68 	movlpd %xmm0,0x6854b0(%rax)
  400c24:	00 
  400c25:	48 83 c0 38          	add    $0x38,%rax
  400c29:	0f 85 41 ff ff ff    	jne    400b70 <_Z14run_simulationv+0x190>
  400c2f:	41 ff c6             	inc    %r14d
  400c32:	41 83 fe 02          	cmp    $0x2,%r14d
  400c36:	0f 85 d4 fd ff ff    	jne    400a10 <_Z14run_simulationv+0x30>
  400c3c:	48 83 c4 28          	add    $0x28,%rsp
  400c40:	5b                   	pop    %rbx
  400c41:	41 5c                	pop    %r12
  400c43:	41 5d                	pop    %r13
  400c45:	41 5e                	pop    %r14
  400c47:	41 5f                	pop    %r15
  400c49:	5d                   	pop    %rbp
  400c4a:	c3                   	retq   
  400c4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c50 <_Z10run_simplev>:
  400c50:	31 c0                	xor    %eax,%eax
  400c52:	66 0f 28 05 e6 02 00 	movapd 0x2e6(%rip),%xmm0        # 400f40 <_IO_stdin_used+0x60>
  400c59:	00 
  400c5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400c60:	48 c7 c1 00 cc f7 ff 	mov    $0xfffffffffff7cc00,%rcx
  400c67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400c6e:	00 00 
  400c70:	66 0f 10 89 b8 54 68 	movupd 0x6854b8(%rcx),%xmm1
  400c77:	00 
  400c78:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  400c7c:	f2 0f 10 91 90 54 68 	movsd  0x685490(%rcx),%xmm2
  400c83:	00 
  400c84:	0f 16 d2             	movlhps %xmm2,%xmm2
  400c87:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  400c8b:	66 0f 10 91 a8 54 68 	movupd 0x6854a8(%rcx),%xmm2
  400c92:	00 
  400c93:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400c97:	66 0f 11 91 a8 54 68 	movupd %xmm2,0x6854a8(%rcx)
  400c9e:	00 
  400c9f:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  400ca3:	66 0f 10 89 98 54 68 	movupd 0x685498(%rcx),%xmm1
  400caa:	00 
  400cab:	66 0f 58 ca          	addpd  %xmm2,%xmm1
  400caf:	66 0f 11 89 98 54 68 	movupd %xmm1,0x685498(%rcx)
  400cb6:	00 
  400cb7:	48 83 c1 38          	add    $0x38,%rcx
  400cbb:	75 b3                	jne    400c70 <_Z10run_simplev+0x20>
  400cbd:	ff c0                	inc    %eax
  400cbf:	3d 20 4e 00 00       	cmp    $0x4e20,%eax
  400cc4:	75 9a                	jne    400c60 <_Z10run_simplev+0x10>
  400cc6:	c3                   	retq   
  400cc7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400cce:	00 00 

0000000000400cd0 <main>:
  400cd0:	41 57                	push   %r15
  400cd2:	41 56                	push   %r14
  400cd4:	41 55                	push   %r13
  400cd6:	41 54                	push   %r12
  400cd8:	53                   	push   %rbx
  400cd9:	e8 42 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400cde:	48 89 c3             	mov    %rax,%rbx
  400ce1:	e8 aa fb ff ff       	callq  400890 <_Z13instantiationv>
  400ce6:	e8 35 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400ceb:	49 89 c6             	mov    %rax,%r14
  400cee:	49 29 de             	sub    %rbx,%r14
  400cf1:	e8 2a fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400cf6:	48 89 c3             	mov    %rax,%rbx
  400cf9:	e8 e2 fc ff ff       	callq  4009e0 <_Z14run_simulationv>
  400cfe:	e8 1d fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d03:	49 89 c4             	mov    %rax,%r12
  400d06:	49 29 dc             	sub    %rbx,%r12
  400d09:	31 db                	xor    %ebx,%ebx
  400d0b:	e8 10 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d10:	49 89 c7             	mov    %rax,%r15
  400d13:	66 0f 28 05 25 02 00 	movapd 0x225(%rip),%xmm0        # 400f40 <_IO_stdin_used+0x60>
  400d1a:	00 
  400d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400d20:	48 c7 c0 00 cc f7 ff 	mov    $0xfffffffffff7cc00,%rax
  400d27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d2e:	00 00 
  400d30:	66 0f 10 88 b8 54 68 	movupd 0x6854b8(%rax),%xmm1
  400d37:	00 
  400d38:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  400d3c:	f2 0f 10 90 90 54 68 	movsd  0x685490(%rax),%xmm2
  400d43:	00 
  400d44:	0f 16 d2             	movlhps %xmm2,%xmm2
  400d47:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  400d4b:	66 0f 10 90 a8 54 68 	movupd 0x6854a8(%rax),%xmm2
  400d52:	00 
  400d53:	66 0f 58 d1          	addpd  %xmm1,%xmm2
  400d57:	66 0f 11 90 a8 54 68 	movupd %xmm2,0x6854a8(%rax)
  400d5e:	00 
  400d5f:	66 0f 59 d0          	mulpd  %xmm0,%xmm2
  400d63:	66 0f 10 88 98 54 68 	movupd 0x685498(%rax),%xmm1
  400d6a:	00 
  400d6b:	66 0f 58 ca          	addpd  %xmm2,%xmm1
  400d6f:	66 0f 11 88 98 54 68 	movupd %xmm1,0x685498(%rax)
  400d76:	00 
  400d77:	48 83 c0 38          	add    $0x38,%rax
  400d7b:	75 b3                	jne    400d30 <main+0x60>
  400d7d:	ff c3                	inc    %ebx
  400d7f:	81 fb 20 4e 00 00    	cmp    $0x4e20,%ebx
  400d85:	75 99                	jne    400d20 <main+0x50>
  400d87:	49 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%r13
  400d8e:	de 1b 43 
  400d91:	4c 89 f0             	mov    %r14,%rax
  400d94:	49 f7 ed             	imul   %r13
  400d97:	49 89 d6             	mov    %rdx,%r14
  400d9a:	4c 89 f0             	mov    %r14,%rax
  400d9d:	48 c1 e8 3f          	shr    $0x3f,%rax
  400da1:	49 c1 fe 12          	sar    $0x12,%r14
  400da5:	49 01 c6             	add    %rax,%r14
  400da8:	4c 89 e0             	mov    %r12,%rax
  400dab:	49 f7 ed             	imul   %r13
  400dae:	48 89 d3             	mov    %rdx,%rbx
  400db1:	48 89 d8             	mov    %rbx,%rax
  400db4:	48 c1 e8 3f          	shr    $0x3f,%rax
  400db8:	48 c1 fb 12          	sar    $0x12,%rbx
  400dbc:	48 01 c3             	add    %rax,%rbx
  400dbf:	e8 5c f9 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400dc4:	4c 29 f8             	sub    %r15,%rax
  400dc7:	41 b8 0b 00 00 00    	mov    $0xb,%r8d
  400dcd:	48 c7 c1 00 cc f7 ff 	mov    $0xfffffffffff7cc00,%rcx
  400dd4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400ddb:	00 00 00 00 00 
  400de0:	f2 0f 10 81 98 54 68 	movsd  0x685498(%rcx),%xmm0
  400de7:	00 
  400de8:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  400dec:	66 0f 7e c2          	movd   %xmm0,%edx
  400df0:	44 01 c2             	add    %r8d,%edx
  400df3:	f2 0f 10 81 a0 54 68 	movsd  0x6854a0(%rcx),%xmm0
  400dfa:	00 
  400dfb:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  400dff:	66 0f 7e c6          	movd   %xmm0,%esi
  400e03:	01 d6                	add    %edx,%esi
  400e05:	4c 63 c6             	movslq %esi,%r8
  400e08:	49 69 d0 8d 68 b7 6c 	imul   $0x6cb7688d,%r8,%rdx
  400e0f:	48 89 d6             	mov    %rdx,%rsi
  400e12:	48 c1 ee 3f          	shr    $0x3f,%rsi
  400e16:	48 c1 fa 33          	sar    $0x33,%rdx
  400e1a:	01 f2                	add    %esi,%edx
  400e1c:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  400e22:	41 29 d0             	sub    %edx,%r8d
  400e25:	48 83 c1 38          	add    $0x38,%rcx
  400e29:	75 b5                	jne    400de0 <main+0x110>
  400e2b:	49 f7 ed             	imul   %r13
  400e2e:	48 89 d0             	mov    %rdx,%rax
  400e31:	48 c1 e8 3f          	shr    $0x3f,%rax
  400e35:	48 c1 fa 12          	sar    $0x12,%rdx
  400e39:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  400e3d:	bf 60 0f 40 00       	mov    $0x400f60,%edi
  400e42:	31 c0                	xor    %eax,%eax
  400e44:	4c 89 f6             	mov    %r14,%rsi
  400e47:	48 89 da             	mov    %rbx,%rdx
  400e4a:	e8 81 f8 ff ff       	callq  4006d0 <printf@plt>
  400e4f:	31 c0                	xor    %eax,%eax
  400e51:	5b                   	pop    %rbx
  400e52:	41 5c                	pop    %r12
  400e54:	41 5d                	pop    %r13
  400e56:	41 5e                	pop    %r14
  400e58:	41 5f                	pop    %r15
  400e5a:	c3                   	retq   
  400e5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e60 <__libc_csu_init>:
  400e60:	41 57                	push   %r15
  400e62:	41 56                	push   %r14
  400e64:	41 89 ff             	mov    %edi,%r15d
  400e67:	41 55                	push   %r13
  400e69:	41 54                	push   %r12
  400e6b:	4c 8d 25 66 0f 20 00 	lea    0x200f66(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  400e72:	55                   	push   %rbp
  400e73:	48 8d 2d 6e 0f 20 00 	lea    0x200f6e(%rip),%rbp        # 601de8 <__init_array_end>
  400e7a:	53                   	push   %rbx
  400e7b:	49 89 f6             	mov    %rsi,%r14
  400e7e:	49 89 d5             	mov    %rdx,%r13
  400e81:	4c 29 e5             	sub    %r12,%rbp
  400e84:	48 83 ec 08          	sub    $0x8,%rsp
  400e88:	48 c1 fd 03          	sar    $0x3,%rbp
  400e8c:	e8 07 f8 ff ff       	callq  400698 <_init>
  400e91:	48 85 ed             	test   %rbp,%rbp
  400e94:	74 20                	je     400eb6 <__libc_csu_init+0x56>
  400e96:	31 db                	xor    %ebx,%ebx
  400e98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e9f:	00 
  400ea0:	4c 89 ea             	mov    %r13,%rdx
  400ea3:	4c 89 f6             	mov    %r14,%rsi
  400ea6:	44 89 ff             	mov    %r15d,%edi
  400ea9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ead:	48 83 c3 01          	add    $0x1,%rbx
  400eb1:	48 39 eb             	cmp    %rbp,%rbx
  400eb4:	75 ea                	jne    400ea0 <__libc_csu_init+0x40>
  400eb6:	48 83 c4 08          	add    $0x8,%rsp
  400eba:	5b                   	pop    %rbx
  400ebb:	5d                   	pop    %rbp
  400ebc:	41 5c                	pop    %r12
  400ebe:	41 5d                	pop    %r13
  400ec0:	41 5e                	pop    %r14
  400ec2:	41 5f                	pop    %r15
  400ec4:	c3                   	retq   
  400ec5:	90                   	nop
  400ec6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ecd:	00 00 00 

0000000000400ed0 <__libc_csu_fini>:
  400ed0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ed4 <_fini>:
  400ed4:	48 83 ec 08          	sub    $0x8,%rsp
  400ed8:	48 83 c4 08          	add    $0x8,%rsp
  400edc:	c3                   	retq   
