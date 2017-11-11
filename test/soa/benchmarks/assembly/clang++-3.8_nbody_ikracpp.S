
bin/clang++-3.8_nbody_ikracpp:     file format elf64-x86-64


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

0000000000400760 <_GLOBAL__sub_I_ikracpp.cc>:
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
  40078f:	49 c7 c0 30 14 40 00 	mov    $0x401430,%r8
  400796:	48 c7 c1 c0 13 40 00 	mov    $0x4013c0,%rcx
  40079d:	48 c7 c7 00 0d 40 00 	mov    $0x400d00,%rdi
  4007a4:	e8 47 ff ff ff       	callq  4006f0 <__libc_start_main@plt>
  4007a9:	f4                   	hlt    
  4007aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007b0 <deregister_tm_clones>:
  4007b0:	b8 6f 20 60 00       	mov    $0x60206f,%eax
  4007b5:	55                   	push   %rbp
  4007b6:	48 2d 68 20 60 00    	sub    $0x602068,%rax
  4007bc:	48 83 f8 0e          	cmp    $0xe,%rax
  4007c0:	48 89 e5             	mov    %rsp,%rbp
  4007c3:	76 1b                	jbe    4007e0 <deregister_tm_clones+0x30>
  4007c5:	b8 00 00 00 00       	mov    $0x0,%eax
  4007ca:	48 85 c0             	test   %rax,%rax
  4007cd:	74 11                	je     4007e0 <deregister_tm_clones+0x30>
  4007cf:	5d                   	pop    %rbp
  4007d0:	bf 68 20 60 00       	mov    $0x602068,%edi
  4007d5:	ff e0                	jmpq   *%rax
  4007d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007de:	00 00 
  4007e0:	5d                   	pop    %rbp
  4007e1:	c3                   	retq   
  4007e2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007ed:	00 00 00 

00000000004007f0 <register_tm_clones>:
  4007f0:	be 68 20 60 00       	mov    $0x602068,%esi
  4007f5:	55                   	push   %rbp
  4007f6:	48 81 ee 68 20 60 00 	sub    $0x602068,%rsi
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
  40081e:	bf 68 20 60 00       	mov    $0x602068,%edi
  400823:	ff e0                	jmpq   *%rax
  400825:	0f 1f 00             	nopl   (%rax)
  400828:	5d                   	pop    %rbp
  400829:	c3                   	retq   
  40082a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400830 <__do_global_dtors_aux>:
  400830:	80 3d 39 18 20 00 00 	cmpb   $0x0,0x201839(%rip)        # 602070 <completed.7585>
  400837:	75 11                	jne    40084a <__do_global_dtors_aux+0x1a>
  400839:	55                   	push   %rbp
  40083a:	48 89 e5             	mov    %rsp,%rbp
  40083d:	e8 6e ff ff ff       	callq  4007b0 <deregister_tm_clones>
  400842:	5d                   	pop    %rbp
  400843:	c6 05 26 18 20 00 01 	movb   $0x1,0x201826(%rip)        # 602070 <completed.7585>
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

0000000000400890 <_ZN4Body7storageEv>:
  400890:	b8 90 20 60 00       	mov    $0x602090,%eax
  400895:	c3                   	retq   
  400896:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089d:	00 00 00 

00000000004008a0 <_Z13instantiationv>:
  4008a0:	53                   	push   %rbx
  4008a1:	48 83 ec 30          	sub    $0x30,%rsp
  4008a5:	bb 80 25 00 00       	mov    $0x2580,%ebx
  4008aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4008b0:	e8 7b fe ff ff       	callq  400730 <rand@plt>
  4008b5:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008b8:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008bc:	f2 0f 10 05 84 0b 00 	movsd  0xb84(%rip),%xmm0        # 401448 <_IO_stdin_used+0x8>
  4008c3:	00 
  4008c4:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  4008c8:	f2 0f 10 05 80 0b 00 	movsd  0xb80(%rip),%xmm0        # 401450 <_IO_stdin_used+0x10>
  4008cf:	00 
  4008d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008d4:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4008d8:	f2 0f 10 05 78 0b 00 	movsd  0xb78(%rip),%xmm0        # 401458 <_IO_stdin_used+0x18>
  4008df:	00 
  4008e0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008e4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  4008ea:	e8 41 fe ff ff       	callq  400730 <rand@plt>
  4008ef:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008f2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008f6:	f2 0f 5e 0d 4a 0b 00 	divsd  0xb4a(%rip),%xmm1        # 401448 <_IO_stdin_used+0x8>
  4008fd:	00 
  4008fe:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400902:	f2 0f 10 05 56 0b 00 	movsd  0xb56(%rip),%xmm0        # 401460 <_IO_stdin_used+0x20>
  400909:	00 
  40090a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40090e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400914:	e8 17 fe ff ff       	callq  400730 <rand@plt>
  400919:	0f 57 c0             	xorps  %xmm0,%xmm0
  40091c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400920:	f2 0f 5e 05 20 0b 00 	divsd  0xb20(%rip),%xmm0        # 401448 <_IO_stdin_used+0x8>
  400927:	00 
  400928:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40092c:	f2 0f 58 05 2c 0b 00 	addsd  0xb2c(%rip),%xmm0        # 401460 <_IO_stdin_used+0x20>
  400933:	00 
  400934:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40093a:	e8 f1 fd ff ff       	callq  400730 <rand@plt>
  40093f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400942:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400946:	f2 0f 5e 0d fa 0a 00 	divsd  0xafa(%rip),%xmm1        # 401448 <_IO_stdin_used+0x8>
  40094d:	00 
  40094e:	f2 0f 10 05 12 0b 00 	movsd  0xb12(%rip),%xmm0        # 401468 <_IO_stdin_used+0x28>
  400955:	00 
  400956:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40095a:	f2 0f 5e 0d f6 0a 00 	divsd  0xaf6(%rip),%xmm1        # 401458 <_IO_stdin_used+0x18>
  400961:	00 
  400962:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400968:	e8 c3 fd ff ff       	callq  400730 <rand@plt>
  40096d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400970:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400974:	f2 0f 5e 05 cc 0a 00 	divsd  0xacc(%rip),%xmm0        # 401448 <_IO_stdin_used+0x8>
  40097b:	00 
  40097c:	f2 0f 58 05 e4 0a 00 	addsd  0xae4(%rip),%xmm0        # 401468 <_IO_stdin_used+0x28>
  400983:	00 
  400984:	f2 0f 5e 05 cc 0a 00 	divsd  0xacc(%rip),%xmm0        # 401458 <_IO_stdin_used+0x18>
  40098b:	00 
  40098c:	48 8b 05 fd 16 20 00 	mov    0x2016fd(%rip),%rax        # 602090 <__Bodydata_buffer>
  400993:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400997:	48 89 0d f2 16 20 00 	mov    %rcx,0x2016f2(%rip)        # 602090 <__Bodydata_buffer>
  40099e:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  4009a4:	f2 0f 11 0c c5 a8 20 	movsd  %xmm1,0x6020a8(,%rax,8)
  4009ab:	60 00 
  4009ad:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  4009b3:	f2 0f 11 0c c5 a8 4c 	movsd  %xmm1,0x614ca8(,%rax,8)
  4009ba:	61 00 
  4009bc:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  4009c2:	f2 0f 11 0c c5 a8 78 	movsd  %xmm1,0x6278a8(,%rax,8)
  4009c9:	62 00 
  4009cb:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  4009d1:	f2 0f 11 0c c5 a8 a4 	movsd  %xmm1,0x63a4a8(,%rax,8)
  4009d8:	63 00 
  4009da:	f2 0f 11 04 c5 a8 d0 	movsd  %xmm0,0x64d0a8(,%rax,8)
  4009e1:	64 00 
  4009e3:	48 c7 04 c5 a8 fc 65 	movq   $0x0,0x65fca8(,%rax,8)
  4009ea:	00 00 00 00 00 
  4009ef:	48 c7 04 c5 a8 28 67 	movq   $0x0,0x6728a8(,%rax,8)
  4009f6:	00 00 00 00 00 
  4009fb:	ff cb                	dec    %ebx
  4009fd:	0f 85 ad fe ff ff    	jne    4008b0 <_Z13instantiationv+0x10>
  400a03:	48 83 c4 30          	add    $0x30,%rsp
  400a07:	5b                   	pop    %rbx
  400a08:	c3                   	retq   
  400a09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a10 <_Z14run_simulationv>:
  400a10:	41 56                	push   %r14
  400a12:	53                   	push   %rbx
  400a13:	50                   	push   %rax
  400a14:	48 8b 05 75 16 20 00 	mov    0x201675(%rip),%rax        # 602090 <__Bodydata_buffer>
  400a1b:	48 85 c0             	test   %rax,%rax
  400a1e:	0f 84 f8 01 00 00    	je     400c1c <_Z14run_simulationv+0x20c>
  400a24:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
  400a28:	31 c9                	xor    %ecx,%ecx
  400a2a:	a8 03                	test   $0x3,%al
  400a2c:	74 32                	je     400a60 <_Z14run_simulationv+0x50>
  400a2e:	89 c6                	mov    %eax,%esi
  400a30:	83 e6 03             	and    $0x3,%esi
  400a33:	31 c9                	xor    %ecx,%ecx
  400a35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400a3c:	00 00 00 00 
  400a40:	48 c7 04 cd a8 fc 65 	movq   $0x0,0x65fca8(,%rcx,8)
  400a47:	00 00 00 00 00 
  400a4c:	48 c7 04 cd a8 28 67 	movq   $0x0,0x6728a8(,%rcx,8)
  400a53:	00 00 00 00 00 
  400a58:	48 ff c1             	inc    %rcx
  400a5b:	48 39 ce             	cmp    %rcx,%rsi
  400a5e:	75 e0                	jne    400a40 <_Z14run_simulationv+0x30>
  400a60:	45 31 f6             	xor    %r14d,%r14d
  400a63:	48 83 fa 03          	cmp    $0x3,%rdx
  400a67:	72 70                	jb     400ad9 <_Z14run_simulationv+0xc9>
  400a69:	45 31 f6             	xor    %r14d,%r14d
  400a6c:	0f 1f 40 00          	nopl   0x0(%rax)
  400a70:	48 c7 04 cd a8 fc 65 	movq   $0x0,0x65fca8(,%rcx,8)
  400a77:	00 00 00 00 00 
  400a7c:	48 c7 04 cd a8 28 67 	movq   $0x0,0x6728a8(,%rcx,8)
  400a83:	00 00 00 00 00 
  400a88:	48 c7 04 cd b0 fc 65 	movq   $0x0,0x65fcb0(,%rcx,8)
  400a8f:	00 00 00 00 00 
  400a94:	48 c7 04 cd b0 28 67 	movq   $0x0,0x6728b0(,%rcx,8)
  400a9b:	00 00 00 00 00 
  400aa0:	48 c7 04 cd b8 fc 65 	movq   $0x0,0x65fcb8(,%rcx,8)
  400aa7:	00 00 00 00 00 
  400aac:	48 c7 04 cd b8 28 67 	movq   $0x0,0x6728b8(,%rcx,8)
  400ab3:	00 00 00 00 00 
  400ab8:	48 c7 04 cd c0 fc 65 	movq   $0x0,0x65fcc0(,%rcx,8)
  400abf:	00 00 00 00 00 
  400ac4:	48 c7 04 cd c0 28 67 	movq   $0x0,0x6728c0(,%rcx,8)
  400acb:	00 00 00 00 00 
  400ad0:	48 8d 49 04          	lea    0x4(%rcx),%rcx
  400ad4:	48 39 c1             	cmp    %rax,%rcx
  400ad7:	72 97                	jb     400a70 <_Z14run_simulationv+0x60>
  400ad9:	48 89 c1             	mov    %rax,%rcx
  400adc:	0f 1f 40 00          	nopl   0x0(%rax)
  400ae0:	49 ff c6             	inc    %r14
  400ae3:	31 db                	xor    %ebx,%ebx
  400ae5:	48 85 c9             	test   %rcx,%rcx
  400ae8:	74 26                	je     400b10 <_Z14run_simulationv+0x100>
  400aea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400af0:	48 ff c3             	inc    %rbx
  400af3:	48 89 df             	mov    %rbx,%rdi
  400af6:	4c 89 f6             	mov    %r14,%rsi
  400af9:	e8 d2 07 00 00       	callq  4012d0 <_ZN4Body9add_forceEPS_>
  400afe:	48 8b 05 8b 15 20 00 	mov    0x20158b(%rip),%rax        # 602090 <__Bodydata_buffer>
  400b05:	48 39 c3             	cmp    %rax,%rbx
  400b08:	48 89 c1             	mov    %rax,%rcx
  400b0b:	72 e3                	jb     400af0 <_Z14run_simulationv+0xe0>
  400b0d:	eb 03                	jmp    400b12 <_Z14run_simulationv+0x102>
  400b0f:	90                   	nop
  400b10:	31 c9                	xor    %ecx,%ecx
  400b12:	49 39 ce             	cmp    %rcx,%r14
  400b15:	72 c9                	jb     400ae0 <_Z14run_simulationv+0xd0>
  400b17:	31 db                	xor    %ebx,%ebx
  400b19:	48 85 c0             	test   %rax,%rax
  400b1c:	0f 84 fa 00 00 00    	je     400c1c <_Z14run_simulationv+0x20c>
  400b22:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b29:	1f 84 00 00 00 00 00 
  400b30:	48 ff c3             	inc    %rbx
  400b33:	48 89 df             	mov    %rbx,%rdi
  400b36:	f2 0f 10 05 12 09 00 	movsd  0x912(%rip),%xmm0        # 401450 <_IO_stdin_used+0x10>
  400b3d:	00 
  400b3e:	e8 8d 06 00 00       	callq  4011d0 <_ZN4Body6updateEd>
  400b43:	48 8b 05 46 15 20 00 	mov    0x201546(%rip),%rax        # 602090 <__Bodydata_buffer>
  400b4a:	48 39 c3             	cmp    %rax,%rbx
  400b4d:	72 e1                	jb     400b30 <_Z14run_simulationv+0x120>
  400b4f:	48 85 c0             	test   %rax,%rax
  400b52:	0f 84 c4 00 00 00    	je     400c1c <_Z14run_simulationv+0x20c>
  400b58:	48 8b 05 31 15 20 00 	mov    0x201531(%rip),%rax        # 602090 <__Bodydata_buffer>
  400b5f:	31 c9                	xor    %ecx,%ecx
  400b61:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b68:	0f 1f 84 00 00 00 00 
  400b6f:	00 
  400b70:	48 c7 04 cd a8 fc 65 	movq   $0x0,0x65fca8(,%rcx,8)
  400b77:	00 00 00 00 00 
  400b7c:	48 c7 04 cd a8 28 67 	movq   $0x0,0x6728a8(,%rcx,8)
  400b83:	00 00 00 00 00 
  400b88:	48 ff c1             	inc    %rcx
  400b8b:	48 39 c1             	cmp    %rax,%rcx
  400b8e:	72 e0                	jb     400b70 <_Z14run_simulationv+0x160>
  400b90:	48 85 c0             	test   %rax,%rax
  400b93:	0f 84 83 00 00 00    	je     400c1c <_Z14run_simulationv+0x20c>
  400b99:	48 8b 05 f0 14 20 00 	mov    0x2014f0(%rip),%rax        # 602090 <__Bodydata_buffer>
  400ba0:	45 31 f6             	xor    %r14d,%r14d
  400ba3:	48 89 c1             	mov    %rax,%rcx
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 
  400bb0:	49 ff c6             	inc    %r14
  400bb3:	48 85 c9             	test   %rcx,%rcx
  400bb6:	bb 00 00 00 00       	mov    $0x0,%ebx
  400bbb:	b9 00 00 00 00       	mov    $0x0,%ecx
  400bc0:	74 2b                	je     400bed <_Z14run_simulationv+0x1dd>
  400bc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400bc9:	1f 84 00 00 00 00 00 
  400bd0:	48 ff c3             	inc    %rbx
  400bd3:	48 89 df             	mov    %rbx,%rdi
  400bd6:	4c 89 f6             	mov    %r14,%rsi
  400bd9:	e8 f2 06 00 00       	callq  4012d0 <_ZN4Body9add_forceEPS_>
  400bde:	48 8b 05 ab 14 20 00 	mov    0x2014ab(%rip),%rax        # 602090 <__Bodydata_buffer>
  400be5:	48 39 c3             	cmp    %rax,%rbx
  400be8:	48 89 c1             	mov    %rax,%rcx
  400beb:	72 e3                	jb     400bd0 <_Z14run_simulationv+0x1c0>
  400bed:	49 39 ce             	cmp    %rcx,%r14
  400bf0:	72 be                	jb     400bb0 <_Z14run_simulationv+0x1a0>
  400bf2:	31 db                	xor    %ebx,%ebx
  400bf4:	48 85 c0             	test   %rax,%rax
  400bf7:	74 23                	je     400c1c <_Z14run_simulationv+0x20c>
  400bf9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400c00:	48 ff c3             	inc    %rbx
  400c03:	48 89 df             	mov    %rbx,%rdi
  400c06:	f2 0f 10 05 42 08 00 	movsd  0x842(%rip),%xmm0        # 401450 <_IO_stdin_used+0x10>
  400c0d:	00 
  400c0e:	e8 bd 05 00 00       	callq  4011d0 <_ZN4Body6updateEd>
  400c13:	48 3b 1d 76 14 20 00 	cmp    0x201476(%rip),%rbx        # 602090 <__Bodydata_buffer>
  400c1a:	72 e4                	jb     400c00 <_Z14run_simulationv+0x1f0>
  400c1c:	48 83 c4 08          	add    $0x8,%rsp
  400c20:	5b                   	pop    %rbx
  400c21:	41 5e                	pop    %r14
  400c23:	c3                   	retq   
  400c24:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400c2b:	00 00 00 00 00 

0000000000400c30 <_Z10run_simplev>:
  400c30:	31 c0                	xor    %eax,%eax
  400c32:	f2 0f 10 05 16 08 00 	movsd  0x816(%rip),%xmm0        # 401450 <_IO_stdin_used+0x10>
  400c39:	00 
  400c3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400c40:	b9 a8 20 60 00       	mov    $0x6020a8,%ecx
  400c45:	ba 01 00 00 00       	mov    $0x1,%edx
  400c4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400c50:	f2 0f 10 0c d5 a0 fc 	movsd  0x65fca0(,%rdx,8),%xmm1
  400c57:	65 00 
  400c59:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400c5d:	f2 0f 5e 09          	divsd  (%rcx),%xmm1
  400c61:	f2 0f 58 0c d5 a0 a4 	addsd  0x63a4a0(,%rdx,8),%xmm1
  400c68:	63 00 
  400c6a:	f2 0f 11 0c d5 a0 a4 	movsd  %xmm1,0x63a4a0(,%rdx,8)
  400c71:	63 00 
  400c73:	f2 0f 10 0c d5 a0 28 	movsd  0x6728a0(,%rdx,8),%xmm1
  400c7a:	67 00 
  400c7c:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400c80:	f2 0f 5e 09          	divsd  (%rcx),%xmm1
  400c84:	f2 0f 58 0c d5 a0 d0 	addsd  0x64d0a0(,%rdx,8),%xmm1
  400c8b:	64 00 
  400c8d:	f2 0f 11 0c d5 a0 d0 	movsd  %xmm1,0x64d0a0(,%rdx,8)
  400c94:	64 00 
  400c96:	f2 0f 10 0c d5 a0 a4 	movsd  0x63a4a0(,%rdx,8),%xmm1
  400c9d:	63 00 
  400c9f:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400ca3:	f2 0f 58 0c d5 a0 4c 	addsd  0x614ca0(,%rdx,8),%xmm1
  400caa:	61 00 
  400cac:	f2 0f 11 0c d5 a0 4c 	movsd  %xmm1,0x614ca0(,%rdx,8)
  400cb3:	61 00 
  400cb5:	f2 0f 10 0c d5 a0 d0 	movsd  0x64d0a0(,%rdx,8),%xmm1
  400cbc:	64 00 
  400cbe:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400cc2:	f2 0f 58 0c d5 a0 78 	addsd  0x6278a0(,%rdx,8),%xmm1
  400cc9:	62 00 
  400ccb:	f2 0f 11 0c d5 a0 78 	movsd  %xmm1,0x6278a0(,%rdx,8)
  400cd2:	62 00 
  400cd4:	48 ff c2             	inc    %rdx
  400cd7:	48 83 c1 08          	add    $0x8,%rcx
  400cdb:	48 81 fa 81 25 00 00 	cmp    $0x2581,%rdx
  400ce2:	0f 85 68 ff ff ff    	jne    400c50 <_Z10run_simplev+0x20>
  400ce8:	ff c0                	inc    %eax
  400cea:	3d 20 4e 00 00       	cmp    $0x4e20,%eax
  400cef:	0f 85 4b ff ff ff    	jne    400c40 <_Z10run_simplev+0x10>
  400cf5:	c3                   	retq   
  400cf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfd:	00 00 00 

0000000000400d00 <main>:
  400d00:	41 57                	push   %r15
  400d02:	41 56                	push   %r14
  400d04:	41 55                	push   %r13
  400d06:	41 54                	push   %r12
  400d08:	53                   	push   %rbx
  400d09:	0f 57 c0             	xorps  %xmm0,%xmm0
  400d0c:	0f 29 05 7d 13 20 00 	movaps %xmm0,0x20137d(%rip)        # 602090 <__Bodydata_buffer>
  400d13:	e8 08 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d18:	48 89 c3             	mov    %rax,%rbx
  400d1b:	e8 80 fb ff ff       	callq  4008a0 <_Z13instantiationv>
  400d20:	e8 fb f9 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d25:	49 89 c4             	mov    %rax,%r12
  400d28:	49 29 dc             	sub    %rbx,%r12
  400d2b:	e8 f0 f9 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d30:	48 89 c3             	mov    %rax,%rbx
  400d33:	e8 d8 fc ff ff       	callq  400a10 <_Z14run_simulationv>
  400d38:	e8 e3 f9 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d3d:	49 89 c7             	mov    %rax,%r15
  400d40:	49 29 df             	sub    %rbx,%r15
  400d43:	31 db                	xor    %ebx,%ebx
  400d45:	e8 d6 f9 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d4a:	49 89 c6             	mov    %rax,%r14
  400d4d:	f2 0f 10 05 fb 06 00 	movsd  0x6fb(%rip),%xmm0        # 401450 <_IO_stdin_used+0x10>
  400d54:	00 
  400d55:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400d5c:	00 00 00 00 
  400d60:	b8 01 00 00 00       	mov    $0x1,%eax
  400d65:	b9 a8 20 60 00       	mov    $0x6020a8,%ecx
  400d6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400d70:	f2 0f 10 0c c5 a0 fc 	movsd  0x65fca0(,%rax,8),%xmm1
  400d77:	65 00 
  400d79:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400d7d:	f2 0f 5e 09          	divsd  (%rcx),%xmm1
  400d81:	f2 0f 58 0c c5 a0 a4 	addsd  0x63a4a0(,%rax,8),%xmm1
  400d88:	63 00 
  400d8a:	f2 0f 11 0c c5 a0 a4 	movsd  %xmm1,0x63a4a0(,%rax,8)
  400d91:	63 00 
  400d93:	f2 0f 10 0c c5 a0 28 	movsd  0x6728a0(,%rax,8),%xmm1
  400d9a:	67 00 
  400d9c:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400da0:	f2 0f 5e 09          	divsd  (%rcx),%xmm1
  400da4:	f2 0f 58 0c c5 a0 d0 	addsd  0x64d0a0(,%rax,8),%xmm1
  400dab:	64 00 
  400dad:	f2 0f 11 0c c5 a0 d0 	movsd  %xmm1,0x64d0a0(,%rax,8)
  400db4:	64 00 
  400db6:	f2 0f 10 0c c5 a0 a4 	movsd  0x63a4a0(,%rax,8),%xmm1
  400dbd:	63 00 
  400dbf:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400dc3:	f2 0f 58 0c c5 a0 4c 	addsd  0x614ca0(,%rax,8),%xmm1
  400dca:	61 00 
  400dcc:	f2 0f 11 0c c5 a0 4c 	movsd  %xmm1,0x614ca0(,%rax,8)
  400dd3:	61 00 
  400dd5:	f2 0f 10 0c c5 a0 d0 	movsd  0x64d0a0(,%rax,8),%xmm1
  400ddc:	64 00 
  400dde:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400de2:	f2 0f 58 0c c5 a0 78 	addsd  0x6278a0(,%rax,8),%xmm1
  400de9:	62 00 
  400deb:	f2 0f 11 0c c5 a0 78 	movsd  %xmm1,0x6278a0(,%rax,8)
  400df2:	62 00 
  400df4:	48 83 c1 08          	add    $0x8,%rcx
  400df8:	48 ff c0             	inc    %rax
  400dfb:	48 3d 81 25 00 00    	cmp    $0x2581,%rax
  400e01:	0f 85 69 ff ff ff    	jne    400d70 <main+0x70>
  400e07:	ff c3                	inc    %ebx
  400e09:	81 fb 20 4e 00 00    	cmp    $0x4e20,%ebx
  400e0f:	0f 85 4b ff ff ff    	jne    400d60 <main+0x60>
  400e15:	49 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%r13
  400e1c:	de 1b 43 
  400e1f:	4c 89 e0             	mov    %r12,%rax
  400e22:	49 f7 ed             	imul   %r13
  400e25:	49 89 d4             	mov    %rdx,%r12
  400e28:	4c 89 e0             	mov    %r12,%rax
  400e2b:	48 c1 e8 3f          	shr    $0x3f,%rax
  400e2f:	49 c1 fc 12          	sar    $0x12,%r12
  400e33:	49 01 c4             	add    %rax,%r12
  400e36:	4c 89 f8             	mov    %r15,%rax
  400e39:	49 f7 ed             	imul   %r13
  400e3c:	48 89 d3             	mov    %rdx,%rbx
  400e3f:	48 89 d8             	mov    %rbx,%rax
  400e42:	48 c1 e8 3f          	shr    $0x3f,%rax
  400e46:	48 c1 fb 12          	sar    $0x12,%rbx
  400e4a:	48 01 c3             	add    %rax,%rbx
  400e4d:	e8 ce f8 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400e52:	4c 29 f0             	sub    %r14,%rax
  400e55:	41 b8 0b 00 00 00    	mov    $0xb,%r8d
  400e5b:	b9 01 00 00 00       	mov    $0x1,%ecx
  400e60:	f2 0f 10 04 cd a0 4c 	movsd  0x614ca0(,%rcx,8),%xmm0
  400e67:	61 00 
  400e69:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  400e6d:	66 0f 7e c2          	movd   %xmm0,%edx
  400e71:	44 01 c2             	add    %r8d,%edx
  400e74:	f2 0f 10 04 cd a0 78 	movsd  0x6278a0(,%rcx,8),%xmm0
  400e7b:	62 00 
  400e7d:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  400e81:	66 0f 7e c6          	movd   %xmm0,%esi
  400e85:	01 d6                	add    %edx,%esi
  400e87:	4c 63 c6             	movslq %esi,%r8
  400e8a:	49 69 d0 8d 68 b7 6c 	imul   $0x6cb7688d,%r8,%rdx
  400e91:	48 89 d6             	mov    %rdx,%rsi
  400e94:	48 c1 ee 3f          	shr    $0x3f,%rsi
  400e98:	48 c1 fa 33          	sar    $0x33,%rdx
  400e9c:	01 f2                	add    %esi,%edx
  400e9e:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  400ea4:	41 29 d0             	sub    %edx,%r8d
  400ea7:	48 ff c1             	inc    %rcx
  400eaa:	48 81 f9 81 25 00 00 	cmp    $0x2581,%rcx
  400eb1:	75 ad                	jne    400e60 <main+0x160>
  400eb3:	49 f7 ed             	imul   %r13
  400eb6:	48 89 d0             	mov    %rdx,%rax
  400eb9:	48 c1 e8 3f          	shr    $0x3f,%rax
  400ebd:	48 c1 fa 12          	sar    $0x12,%rdx
  400ec1:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  400ec5:	bf a0 14 40 00       	mov    $0x4014a0,%edi
  400eca:	31 c0                	xor    %eax,%eax
  400ecc:	4c 89 e6             	mov    %r12,%rsi
  400ecf:	48 89 da             	mov    %rbx,%rdx
  400ed2:	e8 f9 f7 ff ff       	callq  4006d0 <printf@plt>
  400ed7:	31 c0                	xor    %eax,%eax
  400ed9:	5b                   	pop    %rbx
  400eda:	41 5c                	pop    %r12
  400edc:	41 5d                	pop    %r13
  400ede:	41 5e                	pop    %r14
  400ee0:	41 5f                	pop    %r15
  400ee2:	c3                   	retq   
  400ee3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400eea:	84 00 00 00 00 00 

0000000000400ef0 <_Z19codegen_reset_forceP4Body>:
  400ef0:	48 c7 04 fd a0 fc 65 	movq   $0x0,0x65fca0(,%rdi,8)
  400ef7:	00 00 00 00 00 
  400efc:	48 c7 04 fd a0 28 67 	movq   $0x0,0x6728a0(,%rdi,8)
  400f03:	00 00 00 00 00 
  400f08:	c3                   	retq   
  400f09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400f10 <_Z17codegen_add_forceP4BodyS0_>:
  400f10:	41 56                	push   %r14
  400f12:	53                   	push   %rbx
  400f13:	48 83 ec 18          	sub    $0x18,%rsp
  400f17:	49 89 f6             	mov    %rsi,%r14
  400f1a:	48 89 fb             	mov    %rdi,%rbx
  400f1d:	4c 39 f3             	cmp    %r14,%rbx
  400f20:	0f 84 cb 00 00 00    	je     400ff1 <_Z17codegen_add_forceP4BodyS0_+0xe1>
  400f26:	f2 42 0f 10 24 f5 a0 	movsd  0x614ca0(,%r14,8),%xmm4
  400f2d:	4c 61 00 
  400f30:	f2 0f 5c 24 dd a0 4c 	subsd  0x614ca0(,%rbx,8),%xmm4
  400f37:	61 00 
  400f39:	f2 42 0f 10 1c f5 a0 	movsd  0x6278a0(,%r14,8),%xmm3
  400f40:	78 62 00 
  400f43:	f2 0f 5c 1c dd a0 78 	subsd  0x6278a0(,%rbx,8),%xmm3
  400f4a:	62 00 
  400f4c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400f50:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400f54:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400f58:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400f5c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400f60:	0f 57 c0             	xorps  %xmm0,%xmm0
  400f63:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400f67:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400f6b:	7b 21                	jnp    400f8e <_Z17codegen_add_forceP4BodyS0_+0x7e>
  400f6d:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400f71:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  400f77:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  400f7d:	e8 be f7 ff ff       	callq  400740 <sqrt@plt>
  400f82:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  400f88:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  400f8e:	f2 0f 10 0c dd a0 20 	movsd  0x6020a0(,%rbx,8),%xmm1
  400f95:	60 00 
  400f97:	f2 0f 59 0d d9 04 00 	mulsd  0x4d9(%rip),%xmm1        # 401478 <_IO_stdin_used+0x38>
  400f9e:	00 
  400f9f:	f2 42 0f 59 0c f5 a0 	mulsd  0x6020a0(,%r14,8),%xmm1
  400fa6:	20 60 00 
  400fa9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400fad:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400fb1:	f2 0f 58 15 c7 04 00 	addsd  0x4c7(%rip),%xmm2        # 401480 <_IO_stdin_used+0x40>
  400fb8:	00 
  400fb9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400fbd:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  400fc1:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  400fc5:	f2 0f 58 24 dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm4
  400fcc:	65 00 
  400fce:	f2 0f 11 24 dd a0 fc 	movsd  %xmm4,0x65fca0(,%rbx,8)
  400fd5:	65 00 
  400fd7:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  400fdb:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  400fdf:	f2 0f 58 1c dd a0 28 	addsd  0x6728a0(,%rbx,8),%xmm3
  400fe6:	67 00 
  400fe8:	f2 0f 11 1c dd a0 28 	movsd  %xmm3,0x6728a0(,%rbx,8)
  400fef:	67 00 
  400ff1:	48 83 c4 18          	add    $0x18,%rsp
  400ff5:	5b                   	pop    %rbx
  400ff6:	41 5e                	pop    %r14
  400ff8:	c3                   	retq   
  400ff9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401000 <_Z14codegen_updateP4Bodyd>:
  401000:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  401007:	65 00 
  401009:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40100d:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  401014:	60 00 
  401016:	f2 0f 58 0c fd a0 a4 	addsd  0x63a4a0(,%rdi,8),%xmm1
  40101d:	63 00 
  40101f:	f2 0f 11 0c fd a0 a4 	movsd  %xmm1,0x63a4a0(,%rdi,8)
  401026:	63 00 
  401028:	f2 0f 10 0c fd a0 28 	movsd  0x6728a0(,%rdi,8),%xmm1
  40102f:	67 00 
  401031:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401035:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  40103c:	60 00 
  40103e:	f2 0f 58 0c fd a0 d0 	addsd  0x64d0a0(,%rdi,8),%xmm1
  401045:	64 00 
  401047:	f2 0f 11 0c fd a0 d0 	movsd  %xmm1,0x64d0a0(,%rdi,8)
  40104e:	64 00 
  401050:	f2 0f 10 0c fd a0 a4 	movsd  0x63a4a0(,%rdi,8),%xmm1
  401057:	63 00 
  401059:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40105d:	f2 0f 58 0c fd a0 4c 	addsd  0x614ca0(,%rdi,8),%xmm1
  401064:	61 00 
  401066:	f2 0f 11 0c fd a0 4c 	movsd  %xmm1,0x614ca0(,%rdi,8)
  40106d:	61 00 
  40106f:	f2 0f 59 04 fd a0 d0 	mulsd  0x64d0a0(,%rdi,8),%xmm0
  401076:	64 00 
  401078:	f2 0f 58 04 fd a0 78 	addsd  0x6278a0(,%rdi,8),%xmm0
  40107f:	62 00 
  401081:	f2 0f 11 04 fd a0 78 	movsd  %xmm0,0x6278a0(,%rdi,8)
  401088:	62 00 
  40108a:	f2 0f 10 14 fd a0 4c 	movsd  0x614ca0(,%rdi,8),%xmm2
  401091:	61 00 
  401093:	f2 0f 10 0d c5 03 00 	movsd  0x3c5(%rip),%xmm1        # 401460 <_IO_stdin_used+0x20>
  40109a:	00 
  40109b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40109f:	77 0a                	ja     4010ab <_Z14codegen_updateP4Bodyd+0xab>
  4010a1:	66 0f 2e 15 c7 03 00 	ucomisd 0x3c7(%rip),%xmm2        # 401470 <_IO_stdin_used+0x30>
  4010a8:	00 
  4010a9:	76 23                	jbe    4010ce <_Z14codegen_updateP4Bodyd+0xce>
  4010ab:	f2 0f 10 04 fd a0 a4 	movsd  0x63a4a0(,%rdi,8),%xmm0
  4010b2:	63 00 
  4010b4:	66 0f 57 05 d4 03 00 	xorpd  0x3d4(%rip),%xmm0        # 401490 <_IO_stdin_used+0x50>
  4010bb:	00 
  4010bc:	66 0f 13 04 fd a0 a4 	movlpd %xmm0,0x63a4a0(,%rdi,8)
  4010c3:	63 00 
  4010c5:	f2 0f 10 04 fd a0 78 	movsd  0x6278a0(,%rdi,8),%xmm0
  4010cc:	62 00 
  4010ce:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  4010d2:	77 0a                	ja     4010de <_Z14codegen_updateP4Bodyd+0xde>
  4010d4:	66 0f 2e 05 94 03 00 	ucomisd 0x394(%rip),%xmm0        # 401470 <_IO_stdin_used+0x30>
  4010db:	00 
  4010dc:	76 1a                	jbe    4010f8 <_Z14codegen_updateP4Bodyd+0xf8>
  4010de:	f2 0f 10 04 fd a0 d0 	movsd  0x64d0a0(,%rdi,8),%xmm0
  4010e5:	64 00 
  4010e7:	66 0f 57 05 a1 03 00 	xorpd  0x3a1(%rip),%xmm0        # 401490 <_IO_stdin_used+0x50>
  4010ee:	00 
  4010ef:	66 0f 13 04 fd a0 d0 	movlpd %xmm0,0x64d0a0(,%rdi,8)
  4010f6:	64 00 
  4010f8:	c3                   	retq   
  4010f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401100 <_Z26codegen_reset_force_manualP4Body>:
  401100:	48 c7 04 fd a0 fc 65 	movq   $0x0,0x65fca0(,%rdi,8)
  401107:	00 00 00 00 00 
  40110c:	48 c7 04 fd a0 28 67 	movq   $0x0,0x6728a0(,%rdi,8)
  401113:	00 00 00 00 00 
  401118:	c3                   	retq   
  401119:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401120 <_Z24codegen_reset_force_halfP4Body>:
  401120:	48 c7 04 fd a0 fc 65 	movq   $0x0,0x65fca0(,%rdi,8)
  401127:	00 00 00 00 00 
  40112c:	c3                   	retq   
  40112d:	0f 1f 00             	nopl   (%rax)

0000000000401130 <_Z18codegen_reset_massP4Body>:
  401130:	48 c7 04 fd a0 20 60 	movq   $0x0,0x6020a0(,%rdi,8)
  401137:	00 00 00 00 00 
  40113c:	c3                   	retq   
  40113d:	0f 1f 00             	nopl   (%rax)

0000000000401140 <_Z22codengen_simple_updateP4Bodyd>:
  401140:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  401147:	65 00 
  401149:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40114d:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  401154:	60 00 
  401156:	f2 0f 58 0c fd a0 a4 	addsd  0x63a4a0(,%rdi,8),%xmm1
  40115d:	63 00 
  40115f:	f2 0f 11 0c fd a0 a4 	movsd  %xmm1,0x63a4a0(,%rdi,8)
  401166:	63 00 
  401168:	f2 0f 10 0c fd a0 28 	movsd  0x6728a0(,%rdi,8),%xmm1
  40116f:	67 00 
  401171:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401175:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  40117c:	60 00 
  40117e:	f2 0f 58 0c fd a0 d0 	addsd  0x64d0a0(,%rdi,8),%xmm1
  401185:	64 00 
  401187:	f2 0f 11 0c fd a0 d0 	movsd  %xmm1,0x64d0a0(,%rdi,8)
  40118e:	64 00 
  401190:	f2 0f 10 0c fd a0 a4 	movsd  0x63a4a0(,%rdi,8),%xmm1
  401197:	63 00 
  401199:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40119d:	f2 0f 58 0c fd a0 4c 	addsd  0x614ca0(,%rdi,8),%xmm1
  4011a4:	61 00 
  4011a6:	f2 0f 11 0c fd a0 4c 	movsd  %xmm1,0x614ca0(,%rdi,8)
  4011ad:	61 00 
  4011af:	f2 0f 59 04 fd a0 d0 	mulsd  0x64d0a0(,%rdi,8),%xmm0
  4011b6:	64 00 
  4011b8:	f2 0f 58 04 fd a0 78 	addsd  0x6278a0(,%rdi,8),%xmm0
  4011bf:	62 00 
  4011c1:	f2 0f 11 04 fd a0 78 	movsd  %xmm0,0x6278a0(,%rdi,8)
  4011c8:	62 00 
  4011ca:	c3                   	retq   
  4011cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004011d0 <_ZN4Body6updateEd>:
  4011d0:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  4011d7:	65 00 
  4011d9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011dd:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  4011e4:	60 00 
  4011e6:	f2 0f 58 0c fd a0 a4 	addsd  0x63a4a0(,%rdi,8),%xmm1
  4011ed:	63 00 
  4011ef:	f2 0f 11 0c fd a0 a4 	movsd  %xmm1,0x63a4a0(,%rdi,8)
  4011f6:	63 00 
  4011f8:	f2 0f 10 0c fd a0 28 	movsd  0x6728a0(,%rdi,8),%xmm1
  4011ff:	67 00 
  401201:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401205:	f2 0f 5e 0c fd a0 20 	divsd  0x6020a0(,%rdi,8),%xmm1
  40120c:	60 00 
  40120e:	f2 0f 58 0c fd a0 d0 	addsd  0x64d0a0(,%rdi,8),%xmm1
  401215:	64 00 
  401217:	f2 0f 11 0c fd a0 d0 	movsd  %xmm1,0x64d0a0(,%rdi,8)
  40121e:	64 00 
  401220:	f2 0f 10 0c fd a0 a4 	movsd  0x63a4a0(,%rdi,8),%xmm1
  401227:	63 00 
  401229:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40122d:	f2 0f 58 0c fd a0 4c 	addsd  0x614ca0(,%rdi,8),%xmm1
  401234:	61 00 
  401236:	f2 0f 11 0c fd a0 4c 	movsd  %xmm1,0x614ca0(,%rdi,8)
  40123d:	61 00 
  40123f:	f2 0f 59 04 fd a0 d0 	mulsd  0x64d0a0(,%rdi,8),%xmm0
  401246:	64 00 
  401248:	f2 0f 58 04 fd a0 78 	addsd  0x6278a0(,%rdi,8),%xmm0
  40124f:	62 00 
  401251:	f2 0f 11 04 fd a0 78 	movsd  %xmm0,0x6278a0(,%rdi,8)
  401258:	62 00 
  40125a:	f2 0f 10 14 fd a0 4c 	movsd  0x614ca0(,%rdi,8),%xmm2
  401261:	61 00 
  401263:	f2 0f 10 0d f5 01 00 	movsd  0x1f5(%rip),%xmm1        # 401460 <_IO_stdin_used+0x20>
  40126a:	00 
  40126b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40126f:	77 0a                	ja     40127b <_ZN4Body6updateEd+0xab>
  401271:	66 0f 2e 15 f7 01 00 	ucomisd 0x1f7(%rip),%xmm2        # 401470 <_IO_stdin_used+0x30>
  401278:	00 
  401279:	76 23                	jbe    40129e <_ZN4Body6updateEd+0xce>
  40127b:	f2 0f 10 04 fd a0 a4 	movsd  0x63a4a0(,%rdi,8),%xmm0
  401282:	63 00 
  401284:	66 0f 57 05 04 02 00 	xorpd  0x204(%rip),%xmm0        # 401490 <_IO_stdin_used+0x50>
  40128b:	00 
  40128c:	66 0f 13 04 fd a0 a4 	movlpd %xmm0,0x63a4a0(,%rdi,8)
  401293:	63 00 
  401295:	f2 0f 10 04 fd a0 78 	movsd  0x6278a0(,%rdi,8),%xmm0
  40129c:	62 00 
  40129e:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  4012a2:	77 0a                	ja     4012ae <_ZN4Body6updateEd+0xde>
  4012a4:	66 0f 2e 05 c4 01 00 	ucomisd 0x1c4(%rip),%xmm0        # 401470 <_IO_stdin_used+0x30>
  4012ab:	00 
  4012ac:	76 1a                	jbe    4012c8 <_ZN4Body6updateEd+0xf8>
  4012ae:	f2 0f 10 04 fd a0 d0 	movsd  0x64d0a0(,%rdi,8),%xmm0
  4012b5:	64 00 
  4012b7:	66 0f 57 05 d1 01 00 	xorpd  0x1d1(%rip),%xmm0        # 401490 <_IO_stdin_used+0x50>
  4012be:	00 
  4012bf:	66 0f 13 04 fd a0 d0 	movlpd %xmm0,0x64d0a0(,%rdi,8)
  4012c6:	64 00 
  4012c8:	c3                   	retq   
  4012c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004012d0 <_ZN4Body9add_forceEPS_>:
  4012d0:	41 56                	push   %r14
  4012d2:	53                   	push   %rbx
  4012d3:	48 83 ec 18          	sub    $0x18,%rsp
  4012d7:	49 89 f6             	mov    %rsi,%r14
  4012da:	48 89 fb             	mov    %rdi,%rbx
  4012dd:	4c 39 f3             	cmp    %r14,%rbx
  4012e0:	0f 84 cb 00 00 00    	je     4013b1 <_ZN4Body9add_forceEPS_+0xe1>
  4012e6:	f2 42 0f 10 24 f5 a0 	movsd  0x614ca0(,%r14,8),%xmm4
  4012ed:	4c 61 00 
  4012f0:	f2 0f 5c 24 dd a0 4c 	subsd  0x614ca0(,%rbx,8),%xmm4
  4012f7:	61 00 
  4012f9:	f2 42 0f 10 1c f5 a0 	movsd  0x6278a0(,%r14,8),%xmm3
  401300:	78 62 00 
  401303:	f2 0f 5c 1c dd a0 78 	subsd  0x6278a0(,%rbx,8),%xmm3
  40130a:	62 00 
  40130c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401310:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  401314:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401318:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40131c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  401320:	0f 57 c0             	xorps  %xmm0,%xmm0
  401323:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  401327:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  40132b:	7b 21                	jnp    40134e <_ZN4Body9add_forceEPS_+0x7e>
  40132d:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401331:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  401337:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40133d:	e8 fe f3 ff ff       	callq  400740 <sqrt@plt>
  401342:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  401348:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  40134e:	f2 0f 10 0c dd a0 20 	movsd  0x6020a0(,%rbx,8),%xmm1
  401355:	60 00 
  401357:	f2 0f 59 0d 19 01 00 	mulsd  0x119(%rip),%xmm1        # 401478 <_IO_stdin_used+0x38>
  40135e:	00 
  40135f:	f2 42 0f 59 0c f5 a0 	mulsd  0x6020a0(,%r14,8),%xmm1
  401366:	20 60 00 
  401369:	66 0f 28 d0          	movapd %xmm0,%xmm2
  40136d:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  401371:	f2 0f 58 15 07 01 00 	addsd  0x107(%rip),%xmm2        # 401480 <_IO_stdin_used+0x40>
  401378:	00 
  401379:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  40137d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401381:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401385:	f2 0f 58 24 dd a0 fc 	addsd  0x65fca0(,%rbx,8),%xmm4
  40138c:	65 00 
  40138e:	f2 0f 11 24 dd a0 fc 	movsd  %xmm4,0x65fca0(,%rbx,8)
  401395:	65 00 
  401397:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40139b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40139f:	f2 0f 58 1c dd a0 28 	addsd  0x6728a0(,%rbx,8),%xmm3
  4013a6:	67 00 
  4013a8:	f2 0f 11 1c dd a0 28 	movsd  %xmm3,0x6728a0(,%rbx,8)
  4013af:	67 00 
  4013b1:	48 83 c4 18          	add    $0x18,%rsp
  4013b5:	5b                   	pop    %rbx
  4013b6:	41 5e                	pop    %r14
  4013b8:	c3                   	retq   
  4013b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004013c0 <__libc_csu_init>:
  4013c0:	41 57                	push   %r15
  4013c2:	41 56                	push   %r14
  4013c4:	41 89 ff             	mov    %edi,%r15d
  4013c7:	41 55                	push   %r13
  4013c9:	41 54                	push   %r12
  4013cb:	4c 8d 25 06 0a 20 00 	lea    0x200a06(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  4013d2:	55                   	push   %rbp
  4013d3:	48 8d 2d 0e 0a 20 00 	lea    0x200a0e(%rip),%rbp        # 601de8 <__init_array_end>
  4013da:	53                   	push   %rbx
  4013db:	49 89 f6             	mov    %rsi,%r14
  4013de:	49 89 d5             	mov    %rdx,%r13
  4013e1:	4c 29 e5             	sub    %r12,%rbp
  4013e4:	48 83 ec 08          	sub    $0x8,%rsp
  4013e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4013ec:	e8 a7 f2 ff ff       	callq  400698 <_init>
  4013f1:	48 85 ed             	test   %rbp,%rbp
  4013f4:	74 20                	je     401416 <__libc_csu_init+0x56>
  4013f6:	31 db                	xor    %ebx,%ebx
  4013f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4013ff:	00 
  401400:	4c 89 ea             	mov    %r13,%rdx
  401403:	4c 89 f6             	mov    %r14,%rsi
  401406:	44 89 ff             	mov    %r15d,%edi
  401409:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40140d:	48 83 c3 01          	add    $0x1,%rbx
  401411:	48 39 eb             	cmp    %rbp,%rbx
  401414:	75 ea                	jne    401400 <__libc_csu_init+0x40>
  401416:	48 83 c4 08          	add    $0x8,%rsp
  40141a:	5b                   	pop    %rbx
  40141b:	5d                   	pop    %rbp
  40141c:	41 5c                	pop    %r12
  40141e:	41 5d                	pop    %r13
  401420:	41 5e                	pop    %r14
  401422:	41 5f                	pop    %r15
  401424:	c3                   	retq   
  401425:	90                   	nop
  401426:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40142d:	00 00 00 

0000000000401430 <__libc_csu_fini>:
  401430:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401434 <_fini>:
  401434:	48 83 ec 08          	sub    $0x8,%rsp
  401438:	48 83 c4 08          	add    $0x8,%rsp
  40143c:	c3                   	retq   
