
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
  400761:	bf 70 20 60 00       	mov    $0x602070,%edi
  400766:	e8 75 ff ff ff       	callq  4006e0 <_ZNSt8ios_base4InitC1Ev@plt>
  40076b:	bf 10 07 40 00       	mov    $0x400710,%edi
  400770:	be 70 20 60 00       	mov    $0x602070,%esi
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
  40078f:	49 c7 c0 c0 15 40 00 	mov    $0x4015c0,%r8
  400796:	48 c7 c1 50 15 40 00 	mov    $0x401550,%rcx
  40079d:	48 c7 c7 20 0c 40 00 	mov    $0x400c20,%rdi
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
  400830:	80 3d 31 18 20 00 00 	cmpb   $0x0,0x201831(%rip)        # 602068 <__TMC_END__>
  400837:	75 11                	jne    40084a <__do_global_dtors_aux+0x1a>
  400839:	55                   	push   %rbp
  40083a:	48 89 e5             	mov    %rsp,%rbp
  40083d:	e8 6e ff ff ff       	callq  4007b0 <deregister_tm_clones>
  400842:	5d                   	pop    %rbp
  400843:	c6 05 1e 18 20 00 01 	movb   $0x1,0x20181e(%rip)        # 602068 <__TMC_END__>
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
  400890:	b8 78 20 60 00       	mov    $0x602078,%eax
  400895:	c3                   	retq   
  400896:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089d:	00 00 00 

00000000004008a0 <_Z13instantiationv>:
  4008a0:	53                   	push   %rbx
  4008a1:	48 83 ec 30          	sub    $0x30,%rsp
  4008a5:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4008aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4008b0:	e8 7b fe ff ff       	callq  400730 <rand@plt>
  4008b5:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008b8:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008bc:	f2 0f 10 05 14 0d 00 	movsd  0xd14(%rip),%xmm0        # 4015d8 <_IO_stdin_used+0x8>
  4008c3:	00 
  4008c4:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  4008c8:	f2 0f 10 05 10 0d 00 	movsd  0xd10(%rip),%xmm0        # 4015e0 <_IO_stdin_used+0x10>
  4008cf:	00 
  4008d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008d4:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4008d8:	f2 0f 10 05 08 0d 00 	movsd  0xd08(%rip),%xmm0        # 4015e8 <_IO_stdin_used+0x18>
  4008df:	00 
  4008e0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008e4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  4008ea:	e8 41 fe ff ff       	callq  400730 <rand@plt>
  4008ef:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008f2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008f6:	f2 0f 5e 0d da 0c 00 	divsd  0xcda(%rip),%xmm1        # 4015d8 <_IO_stdin_used+0x8>
  4008fd:	00 
  4008fe:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400902:	f2 0f 10 05 e6 0c 00 	movsd  0xce6(%rip),%xmm0        # 4015f0 <_IO_stdin_used+0x20>
  400909:	00 
  40090a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40090e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400914:	e8 17 fe ff ff       	callq  400730 <rand@plt>
  400919:	0f 57 c0             	xorps  %xmm0,%xmm0
  40091c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400920:	f2 0f 5e 05 b0 0c 00 	divsd  0xcb0(%rip),%xmm0        # 4015d8 <_IO_stdin_used+0x8>
  400927:	00 
  400928:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40092c:	f2 0f 58 05 bc 0c 00 	addsd  0xcbc(%rip),%xmm0        # 4015f0 <_IO_stdin_used+0x20>
  400933:	00 
  400934:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40093a:	e8 f1 fd ff ff       	callq  400730 <rand@plt>
  40093f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400942:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400946:	f2 0f 5e 0d 8a 0c 00 	divsd  0xc8a(%rip),%xmm1        # 4015d8 <_IO_stdin_used+0x8>
  40094d:	00 
  40094e:	f2 0f 10 05 a2 0c 00 	movsd  0xca2(%rip),%xmm0        # 4015f8 <_IO_stdin_used+0x28>
  400955:	00 
  400956:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40095a:	f2 0f 5e 0d 86 0c 00 	divsd  0xc86(%rip),%xmm1        # 4015e8 <_IO_stdin_used+0x18>
  400961:	00 
  400962:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400968:	e8 c3 fd ff ff       	callq  400730 <rand@plt>
  40096d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400970:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400974:	f2 0f 5e 05 5c 0c 00 	divsd  0xc5c(%rip),%xmm0        # 4015d8 <_IO_stdin_used+0x8>
  40097b:	00 
  40097c:	f2 0f 58 05 74 0c 00 	addsd  0xc74(%rip),%xmm0        # 4015f8 <_IO_stdin_used+0x28>
  400983:	00 
  400984:	f2 0f 5e 05 5c 0c 00 	divsd  0xc5c(%rip),%xmm0        # 4015e8 <_IO_stdin_used+0x18>
  40098b:	00 
  40098c:	48 8b 05 e5 16 20 00 	mov    0x2016e5(%rip),%rax        # 602078 <__Bodydata_buffer>
  400993:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400997:	48 89 0d da 16 20 00 	mov    %rcx,0x2016da(%rip)        # 602078 <__Bodydata_buffer>
  40099e:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  4009a4:	f2 0f 11 0c c5 90 20 	movsd  %xmm1,0x602090(,%rax,8)
  4009ab:	60 00 
  4009ad:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  4009b3:	f2 0f 11 0c c5 90 1a 	movsd  %xmm1,0x611a90(,%rax,8)
  4009ba:	61 00 
  4009bc:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  4009c2:	f2 0f 11 0c c5 90 14 	movsd  %xmm1,0x621490(,%rax,8)
  4009c9:	62 00 
  4009cb:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  4009d1:	f2 0f 11 0c c5 90 0e 	movsd  %xmm1,0x630e90(,%rax,8)
  4009d8:	63 00 
  4009da:	f2 0f 11 04 c5 90 08 	movsd  %xmm0,0x640890(,%rax,8)
  4009e1:	64 00 
  4009e3:	48 c7 04 c5 90 02 65 	movq   $0x0,0x650290(,%rax,8)
  4009ea:	00 00 00 00 00 
  4009ef:	48 c7 04 c5 90 fc 65 	movq   $0x0,0x65fc90(,%rax,8)
  4009f6:	00 00 00 00 00 
  4009fb:	ff cb                	dec    %ebx
  4009fd:	0f 85 ad fe ff ff    	jne    4008b0 <_Z13instantiationv+0x10>
  400a03:	48 83 c4 30          	add    $0x30,%rsp
  400a07:	5b                   	pop    %rbx
  400a08:	c3                   	retq   
  400a09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a10 <_Z14run_simulationv>:
  400a10:	41 57                	push   %r15
  400a12:	41 56                	push   %r14
  400a14:	41 55                	push   %r13
  400a16:	41 54                	push   %r12
  400a18:	53                   	push   %rbx
  400a19:	45 31 ff             	xor    %r15d,%r15d
  400a1c:	0f 1f 40 00          	nopl   0x0(%rax)
  400a20:	4c 8b 25 51 16 20 00 	mov    0x201651(%rip),%r12        # 602078 <__Bodydata_buffer>
  400a27:	4d 85 e4             	test   %r12,%r12
  400a2a:	0f 84 18 01 00 00    	je     400b48 <_Z14run_simulationv+0x138>
  400a30:	49 8d 4c 24 ff       	lea    -0x1(%r12),%rcx
  400a35:	41 f6 c4 03          	test   $0x3,%r12b
  400a39:	b8 00 00 00 00       	mov    $0x0,%eax
  400a3e:	74 30                	je     400a70 <_Z14run_simulationv+0x60>
  400a40:	44 89 e2             	mov    %r12d,%edx
  400a43:	83 e2 03             	and    $0x3,%edx
  400a46:	31 c0                	xor    %eax,%eax
  400a48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a4f:	00 
  400a50:	48 c7 04 c5 90 02 65 	movq   $0x0,0x650290(,%rax,8)
  400a57:	00 00 00 00 00 
  400a5c:	48 c7 04 c5 90 fc 65 	movq   $0x0,0x65fc90(,%rax,8)
  400a63:	00 00 00 00 00 
  400a68:	48 ff c0             	inc    %rax
  400a6b:	48 39 c2             	cmp    %rax,%rdx
  400a6e:	75 e0                	jne    400a50 <_Z14run_simulationv+0x40>
  400a70:	48 83 f9 03          	cmp    $0x3,%rcx
  400a74:	4d 89 e5             	mov    %r12,%r13
  400a77:	41 be 00 00 00 00    	mov    $0x0,%r14d
  400a7d:	72 71                	jb     400af0 <_Z14run_simulationv+0xe0>
  400a7f:	90                   	nop
  400a80:	48 c7 04 c5 90 02 65 	movq   $0x0,0x650290(,%rax,8)
  400a87:	00 00 00 00 00 
  400a8c:	48 c7 04 c5 90 fc 65 	movq   $0x0,0x65fc90(,%rax,8)
  400a93:	00 00 00 00 00 
  400a98:	48 c7 04 c5 98 02 65 	movq   $0x0,0x650298(,%rax,8)
  400a9f:	00 00 00 00 00 
  400aa4:	48 c7 04 c5 98 fc 65 	movq   $0x0,0x65fc98(,%rax,8)
  400aab:	00 00 00 00 00 
  400ab0:	48 c7 04 c5 a0 02 65 	movq   $0x0,0x6502a0(,%rax,8)
  400ab7:	00 00 00 00 00 
  400abc:	48 c7 04 c5 a0 fc 65 	movq   $0x0,0x65fca0(,%rax,8)
  400ac3:	00 00 00 00 00 
  400ac8:	48 c7 04 c5 a8 02 65 	movq   $0x0,0x6502a8(,%rax,8)
  400acf:	00 00 00 00 00 
  400ad4:	48 c7 04 c5 a8 fc 65 	movq   $0x0,0x65fca8(,%rax,8)
  400adb:	00 00 00 00 00 
  400ae0:	48 8d 40 04          	lea    0x4(%rax),%rax
  400ae4:	49 39 c4             	cmp    %rax,%r12
  400ae7:	75 97                	jne    400a80 <_Z14run_simulationv+0x70>
  400ae9:	4d 89 e5             	mov    %r12,%r13
  400aec:	45 31 f6             	xor    %r14d,%r14d
  400aef:	90                   	nop
  400af0:	49 ff c6             	inc    %r14
  400af3:	4d 85 ed             	test   %r13,%r13
  400af6:	bb 00 00 00 00       	mov    $0x0,%ebx
  400afb:	74 16                	je     400b13 <_Z14run_simulationv+0x103>
  400afd:	0f 1f 00             	nopl   (%rax)
  400b00:	48 ff c3             	inc    %rbx
  400b03:	48 89 df             	mov    %rbx,%rdi
  400b06:	4c 89 f6             	mov    %r14,%rsi
  400b09:	e8 52 09 00 00       	callq  401460 <_ZN4Body9add_forceEPS_>
  400b0e:	49 39 dd             	cmp    %rbx,%r13
  400b11:	75 ed                	jne    400b00 <_Z14run_simulationv+0xf0>
  400b13:	4c 8b 2d 5e 15 20 00 	mov    0x20155e(%rip),%r13        # 602078 <__Bodydata_buffer>
  400b1a:	4d 39 e6             	cmp    %r12,%r14
  400b1d:	75 d1                	jne    400af0 <_Z14run_simulationv+0xe0>
  400b1f:	4d 85 ed             	test   %r13,%r13
  400b22:	bb 00 00 00 00       	mov    $0x0,%ebx
  400b27:	74 1f                	je     400b48 <_Z14run_simulationv+0x138>
  400b29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400b30:	48 ff c3             	inc    %rbx
  400b33:	48 89 df             	mov    %rbx,%rdi
  400b36:	f2 0f 10 05 a2 0a 00 	movsd  0xaa2(%rip),%xmm0        # 4015e0 <_IO_stdin_used+0x10>
  400b3d:	00 
  400b3e:	e8 1d 08 00 00       	callq  401360 <_ZN4Body6updateEd>
  400b43:	49 39 dd             	cmp    %rbx,%r13
  400b46:	75 e8                	jne    400b30 <_Z14run_simulationv+0x120>
  400b48:	41 ff c7             	inc    %r15d
  400b4b:	41 83 ff 05          	cmp    $0x5,%r15d
  400b4f:	0f 85 cb fe ff ff    	jne    400a20 <_Z14run_simulationv+0x10>
  400b55:	5b                   	pop    %rbx
  400b56:	41 5c                	pop    %r12
  400b58:	41 5d                	pop    %r13
  400b5a:	41 5e                	pop    %r14
  400b5c:	41 5f                	pop    %r15
  400b5e:	c3                   	retq   
  400b5f:	90                   	nop

0000000000400b60 <_Z10run_simplev>:
  400b60:	31 c0                	xor    %eax,%eax
  400b62:	f2 0f 10 05 76 0a 00 	movsd  0xa76(%rip),%xmm0        # 4015e0 <_IO_stdin_used+0x10>
  400b69:	00 
  400b6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b70:	48 c7 c1 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rcx
  400b77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b7e:	00 00 
  400b80:	f2 0f 10 0c cd 90 fc 	movsd  0x65fc90(,%rcx,8),%xmm1
  400b87:	65 00 
  400b89:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400b8d:	f2 0f 5e 0c cd 90 1a 	divsd  0x611a90(,%rcx,8),%xmm1
  400b94:	61 00 
  400b96:	f2 0f 58 0c cd 90 08 	addsd  0x640890(,%rcx,8),%xmm1
  400b9d:	64 00 
  400b9f:	f2 0f 11 0c cd 90 08 	movsd  %xmm1,0x640890(,%rcx,8)
  400ba6:	64 00 
  400ba8:	f2 0f 10 14 cd 90 f6 	movsd  0x66f690(,%rcx,8),%xmm2
  400baf:	66 00 
  400bb1:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  400bb5:	f2 0f 5e 14 cd 90 1a 	divsd  0x611a90(,%rcx,8),%xmm2
  400bbc:	61 00 
  400bbe:	f2 0f 58 14 cd 90 02 	addsd  0x650290(,%rcx,8),%xmm2
  400bc5:	65 00 
  400bc7:	f2 0f 11 14 cd 90 02 	movsd  %xmm2,0x650290(,%rcx,8)
  400bce:	65 00 
  400bd0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400bd4:	f2 0f 58 0c cd 90 14 	addsd  0x621490(,%rcx,8),%xmm1
  400bdb:	62 00 
  400bdd:	f2 0f 11 0c cd 90 14 	movsd  %xmm1,0x621490(,%rcx,8)
  400be4:	62 00 
  400be6:	f2 0f 10 0c cd 90 02 	movsd  0x650290(,%rcx,8),%xmm1
  400bed:	65 00 
  400bef:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400bf3:	f2 0f 58 0c cd 90 0e 	addsd  0x630e90(,%rcx,8),%xmm1
  400bfa:	63 00 
  400bfc:	f2 0f 11 0c cd 90 0e 	movsd  %xmm1,0x630e90(,%rcx,8)
  400c03:	63 00 
  400c05:	48 ff c1             	inc    %rcx
  400c08:	0f 85 72 ff ff ff    	jne    400b80 <_Z10run_simplev+0x20>
  400c0e:	ff c0                	inc    %eax
  400c10:	3d 50 c3 00 00       	cmp    $0xc350,%eax
  400c15:	0f 85 55 ff ff ff    	jne    400b70 <_Z10run_simplev+0x10>
  400c1b:	c3                   	retq   
  400c1c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c20 <main>:
  400c20:	55                   	push   %rbp
  400c21:	41 57                	push   %r15
  400c23:	41 56                	push   %r14
  400c25:	41 55                	push   %r13
  400c27:	41 54                	push   %r12
  400c29:	53                   	push   %rbx
  400c2a:	48 83 ec 28          	sub    $0x28,%rsp
  400c2e:	66 0f 57 c0          	xorpd  %xmm0,%xmm0
  400c32:	66 0f 11 05 3e 14 20 	movupd %xmm0,0x20143e(%rip)        # 602078 <__Bodydata_buffer>
  400c39:	00 
  400c3a:	e8 e1 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c3f:	48 89 c3             	mov    %rax,%rbx
  400c42:	e8 59 fc ff ff       	callq  4008a0 <_Z13instantiationv>
  400c47:	e8 d4 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c4c:	49 89 c7             	mov    %rax,%r15
  400c4f:	49 29 df             	sub    %rbx,%r15
  400c52:	e8 c9 fa ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c57:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400c5c:	45 31 ed             	xor    %r13d,%r13d
  400c5f:	48 8b 05 12 14 20 00 	mov    0x201412(%rip),%rax        # 602078 <__Bodydata_buffer>
  400c66:	48 89 c5             	mov    %rax,%rbp
  400c69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400c70:	48 85 ed             	test   %rbp,%rbp
  400c73:	0f 84 37 02 00 00    	je     400eb0 <main+0x290>
  400c79:	48 8d 55 ff          	lea    -0x1(%rbp),%rdx
  400c7d:	40 f6 c5 03          	test   $0x3,%bpl
  400c81:	b9 00 00 00 00       	mov    $0x0,%ecx
  400c86:	74 28                	je     400cb0 <main+0x90>
  400c88:	89 ee                	mov    %ebp,%esi
  400c8a:	83 e6 03             	and    $0x3,%esi
  400c8d:	31 c9                	xor    %ecx,%ecx
  400c8f:	90                   	nop
  400c90:	48 c7 04 cd 90 02 65 	movq   $0x0,0x650290(,%rcx,8)
  400c97:	00 00 00 00 00 
  400c9c:	48 c7 04 cd 90 fc 65 	movq   $0x0,0x65fc90(,%rcx,8)
  400ca3:	00 00 00 00 00 
  400ca8:	48 ff c1             	inc    %rcx
  400cab:	48 39 ce             	cmp    %rcx,%rsi
  400cae:	75 e0                	jne    400c90 <main+0x70>
  400cb0:	48 83 fa 03          	cmp    $0x3,%rdx
  400cb4:	72 73                	jb     400d29 <main+0x109>
  400cb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cbd:	00 00 00 
  400cc0:	48 c7 04 cd 90 02 65 	movq   $0x0,0x650290(,%rcx,8)
  400cc7:	00 00 00 00 00 
  400ccc:	48 c7 04 cd 90 fc 65 	movq   $0x0,0x65fc90(,%rcx,8)
  400cd3:	00 00 00 00 00 
  400cd8:	48 c7 04 cd 98 02 65 	movq   $0x0,0x650298(,%rcx,8)
  400cdf:	00 00 00 00 00 
  400ce4:	48 c7 04 cd 98 fc 65 	movq   $0x0,0x65fc98(,%rcx,8)
  400ceb:	00 00 00 00 00 
  400cf0:	48 c7 04 cd a0 02 65 	movq   $0x0,0x6502a0(,%rcx,8)
  400cf7:	00 00 00 00 00 
  400cfc:	48 c7 04 cd a0 fc 65 	movq   $0x0,0x65fca0(,%rcx,8)
  400d03:	00 00 00 00 00 
  400d08:	48 c7 04 cd a8 02 65 	movq   $0x0,0x6502a8(,%rcx,8)
  400d0f:	00 00 00 00 00 
  400d14:	48 c7 04 cd a8 fc 65 	movq   $0x0,0x65fca8(,%rcx,8)
  400d1b:	00 00 00 00 00 
  400d20:	48 8d 49 04          	lea    0x4(%rcx),%rcx
  400d24:	48 39 cd             	cmp    %rcx,%rbp
  400d27:	75 97                	jne    400cc0 <main+0xa0>
  400d29:	49 89 ee             	mov    %rbp,%r14
  400d2c:	45 31 e4             	xor    %r12d,%r12d
  400d2f:	90                   	nop
  400d30:	49 ff c4             	inc    %r12
  400d33:	4d 85 f6             	test   %r14,%r14
  400d36:	bb 00 00 00 00       	mov    $0x0,%ebx
  400d3b:	74 1d                	je     400d5a <main+0x13a>
  400d3d:	0f 1f 00             	nopl   (%rax)
  400d40:	48 ff c3             	inc    %rbx
  400d43:	48 89 df             	mov    %rbx,%rdi
  400d46:	4c 89 e6             	mov    %r12,%rsi
  400d49:	e8 12 07 00 00       	callq  401460 <_ZN4Body9add_forceEPS_>
  400d4e:	49 39 de             	cmp    %rbx,%r14
  400d51:	75 ed                	jne    400d40 <main+0x120>
  400d53:	48 8b 05 1e 13 20 00 	mov    0x20131e(%rip),%rax        # 602078 <__Bodydata_buffer>
  400d5a:	49 39 ec             	cmp    %rbp,%r12
  400d5d:	49 89 c6             	mov    %rax,%r14
  400d60:	75 ce                	jne    400d30 <main+0x110>
  400d62:	48 85 c0             	test   %rax,%rax
  400d65:	ba 00 00 00 00       	mov    $0x0,%edx
  400d6a:	b9 00 00 00 00       	mov    $0x0,%ecx
  400d6f:	bd 00 00 00 00       	mov    $0x0,%ebp
  400d74:	f2 0f 10 05 64 08 00 	movsd  0x864(%rip),%xmm0        # 4015e0 <_IO_stdin_used+0x10>
  400d7b:	00 
  400d7c:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400d80:	f2 0f 10 1d 68 08 00 	movsd  0x868(%rip),%xmm3        # 4015f0 <_IO_stdin_used+0x20>
  400d87:	00 
  400d88:	66 0f 28 25 90 08 00 	movapd 0x890(%rip),%xmm4        # 401620 <_IO_stdin_used+0x50>
  400d8f:	00 
  400d90:	0f 84 1f 01 00 00    	je     400eb5 <main+0x295>
  400d96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d9d:	00 00 00 
  400da0:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400da4:	f2 0f 10 04 d5 90 02 	movsd  0x650290(,%rdx,8),%xmm0
  400dab:	65 00 
  400dad:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400db1:	f2 0f 5e 04 d5 90 20 	divsd  0x602090(,%rdx,8),%xmm0
  400db8:	60 00 
  400dba:	f2 0f 58 04 d5 90 0e 	addsd  0x630e90(,%rdx,8),%xmm0
  400dc1:	63 00 
  400dc3:	f2 0f 11 04 d5 90 0e 	movsd  %xmm0,0x630e90(,%rdx,8)
  400dca:	63 00 
  400dcc:	f2 0f 10 04 d5 90 fc 	movsd  0x65fc90(,%rdx,8),%xmm0
  400dd3:	65 00 
  400dd5:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400dd9:	f2 0f 5e 04 d5 90 20 	divsd  0x602090(,%rdx,8),%xmm0
  400de0:	60 00 
  400de2:	f2 0f 58 04 d5 90 08 	addsd  0x640890(,%rdx,8),%xmm0
  400de9:	64 00 
  400deb:	f2 0f 11 04 d5 90 08 	movsd  %xmm0,0x640890(,%rdx,8)
  400df2:	64 00 
  400df4:	f2 0f 10 04 d5 90 0e 	movsd  0x630e90(,%rdx,8),%xmm0
  400dfb:	63 00 
  400dfd:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400e01:	f2 0f 58 04 d5 90 1a 	addsd  0x611a90(,%rdx,8),%xmm0
  400e08:	61 00 
  400e0a:	f2 0f 11 04 d5 90 1a 	movsd  %xmm0,0x611a90(,%rdx,8)
  400e11:	61 00 
  400e13:	f2 0f 10 04 d5 90 08 	movsd  0x640890(,%rdx,8),%xmm0
  400e1a:	64 00 
  400e1c:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400e20:	f2 0f 58 04 d5 90 14 	addsd  0x621490(,%rdx,8),%xmm0
  400e27:	62 00 
  400e29:	f2 0f 11 04 d5 90 14 	movsd  %xmm0,0x621490(,%rdx,8)
  400e30:	62 00 
  400e32:	f2 0f 10 0c d5 90 1a 	movsd  0x611a90(,%rdx,8),%xmm1
  400e39:	61 00 
  400e3b:	66 0f 2e d9          	ucomisd %xmm1,%xmm3
  400e3f:	77 0a                	ja     400e4b <main+0x22b>
  400e41:	66 0f 2e 0d b7 07 00 	ucomisd 0x7b7(%rip),%xmm1        # 401600 <_IO_stdin_used+0x30>
  400e48:	00 
  400e49:	76 1f                	jbe    400e6a <main+0x24a>
  400e4b:	f2 0f 10 04 cd 88 0e 	movsd  0x630e88(,%rcx,8),%xmm0
  400e52:	63 00 
  400e54:	66 0f 57 c4          	xorpd  %xmm4,%xmm0
  400e58:	66 0f 13 04 cd 88 0e 	movlpd %xmm0,0x630e88(,%rcx,8)
  400e5f:	63 00 
  400e61:	f2 0f 10 04 cd 88 14 	movsd  0x621488(,%rcx,8),%xmm0
  400e68:	62 00 
  400e6a:	66 0f 2e d8          	ucomisd %xmm0,%xmm3
  400e6e:	77 0a                	ja     400e7a <main+0x25a>
  400e70:	66 0f 2e 05 88 07 00 	ucomisd 0x788(%rip),%xmm0        # 401600 <_IO_stdin_used+0x30>
  400e77:	00 
  400e78:	76 16                	jbe    400e90 <main+0x270>
  400e7a:	f2 0f 10 04 cd 88 08 	movsd  0x640888(,%rcx,8),%xmm0
  400e81:	64 00 
  400e83:	66 0f 57 c4          	xorpd  %xmm4,%xmm0
  400e87:	66 0f 13 04 cd 88 08 	movlpd %xmm0,0x640888(,%rcx,8)
  400e8e:	64 00 
  400e90:	48 39 c8             	cmp    %rcx,%rax
  400e93:	48 89 ca             	mov    %rcx,%rdx
  400e96:	0f 85 04 ff ff ff    	jne    400da0 <main+0x180>
  400e9c:	48 89 c1             	mov    %rax,%rcx
  400e9f:	48 89 c5             	mov    %rax,%rbp
  400ea2:	eb 11                	jmp    400eb5 <main+0x295>
  400ea4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400eab:	00 00 00 00 00 
  400eb0:	48 89 c1             	mov    %rax,%rcx
  400eb3:	31 ed                	xor    %ebp,%ebp
  400eb5:	41 ff c5             	inc    %r13d
  400eb8:	41 83 fd 05          	cmp    $0x5,%r13d
  400ebc:	48 89 c8             	mov    %rcx,%rax
  400ebf:	0f 85 ab fd ff ff    	jne    400c70 <main+0x50>
  400ec5:	49 bc db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%r12
  400ecc:	de 1b 43 
  400ecf:	4c 89 f8             	mov    %r15,%rax
  400ed2:	49 f7 ec             	imul   %r12
  400ed5:	49 89 d7             	mov    %rdx,%r15
  400ed8:	4c 89 f8             	mov    %r15,%rax
  400edb:	48 c1 e8 3f          	shr    $0x3f,%rax
  400edf:	49 c1 ff 12          	sar    $0x12,%r15
  400ee3:	49 01 c7             	add    %rax,%r15
  400ee6:	e8 35 f8 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400eeb:	48 89 c3             	mov    %rax,%rbx
  400eee:	48 2b 5c 24 20       	sub    0x20(%rsp),%rbx
  400ef3:	31 ed                	xor    %ebp,%ebp
  400ef5:	e8 26 f8 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400efa:	49 89 c6             	mov    %rax,%r14
  400efd:	f2 0f 10 15 db 06 00 	movsd  0x6db(%rip),%xmm2        # 4015e0 <_IO_stdin_used+0x10>
  400f04:	00 
  400f05:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400f0c:	00 00 00 00 
  400f10:	48 c7 c0 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rax
  400f17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f1e:	00 00 
  400f20:	f2 0f 10 04 c5 90 fc 	movsd  0x65fc90(,%rax,8),%xmm0
  400f27:	65 00 
  400f29:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400f2d:	f2 0f 5e 04 c5 90 1a 	divsd  0x611a90(,%rax,8),%xmm0
  400f34:	61 00 
  400f36:	f2 0f 58 04 c5 90 08 	addsd  0x640890(,%rax,8),%xmm0
  400f3d:	64 00 
  400f3f:	f2 0f 11 04 c5 90 08 	movsd  %xmm0,0x640890(,%rax,8)
  400f46:	64 00 
  400f48:	f2 0f 10 0c c5 90 f6 	movsd  0x66f690(,%rax,8),%xmm1
  400f4f:	66 00 
  400f51:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400f55:	f2 0f 5e 0c c5 90 1a 	divsd  0x611a90(,%rax,8),%xmm1
  400f5c:	61 00 
  400f5e:	f2 0f 58 0c c5 90 02 	addsd  0x650290(,%rax,8),%xmm1
  400f65:	65 00 
  400f67:	f2 0f 11 0c c5 90 02 	movsd  %xmm1,0x650290(,%rax,8)
  400f6e:	65 00 
  400f70:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400f74:	f2 0f 58 04 c5 90 14 	addsd  0x621490(,%rax,8),%xmm0
  400f7b:	62 00 
  400f7d:	f2 0f 11 04 c5 90 14 	movsd  %xmm0,0x621490(,%rax,8)
  400f84:	62 00 
  400f86:	f2 0f 10 04 c5 90 02 	movsd  0x650290(,%rax,8),%xmm0
  400f8d:	65 00 
  400f8f:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400f93:	f2 0f 58 04 c5 90 0e 	addsd  0x630e90(,%rax,8),%xmm0
  400f9a:	63 00 
  400f9c:	f2 0f 11 04 c5 90 0e 	movsd  %xmm0,0x630e90(,%rax,8)
  400fa3:	63 00 
  400fa5:	48 ff c0             	inc    %rax
  400fa8:	0f 85 72 ff ff ff    	jne    400f20 <main+0x300>
  400fae:	ff c5                	inc    %ebp
  400fb0:	81 fd 50 c3 00 00    	cmp    $0xc350,%ebp
  400fb6:	0f 85 54 ff ff ff    	jne    400f10 <main+0x2f0>
  400fbc:	48 89 d8             	mov    %rbx,%rax
  400fbf:	49 f7 ec             	imul   %r12
  400fc2:	48 89 d3             	mov    %rdx,%rbx
  400fc5:	48 89 d8             	mov    %rbx,%rax
  400fc8:	48 c1 e8 3f          	shr    $0x3f,%rax
  400fcc:	48 c1 fb 12          	sar    $0x12,%rbx
  400fd0:	48 01 c3             	add    %rax,%rbx
  400fd3:	e8 48 f7 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400fd8:	4c 29 f0             	sub    %r14,%rax
  400fdb:	41 b8 0b 00 00 00    	mov    $0xb,%r8d
  400fe1:	b9 01 00 00 00       	mov    $0x1,%ecx
  400fe6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fed:	00 00 00 
  400ff0:	f2 0f 10 04 cd 88 1a 	movsd  0x611a88(,%rcx,8),%xmm0
  400ff7:	61 00 
  400ff9:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  400ffd:	66 0f 7e c2          	movd   %xmm0,%edx
  401001:	44 01 c2             	add    %r8d,%edx
  401004:	f2 0f 10 04 cd 88 14 	movsd  0x621488(,%rcx,8),%xmm0
  40100b:	62 00 
  40100d:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  401011:	66 0f 7e c6          	movd   %xmm0,%esi
  401015:	01 d6                	add    %edx,%esi
  401017:	4c 63 c6             	movslq %esi,%r8
  40101a:	49 69 d0 8d 68 b7 6c 	imul   $0x6cb7688d,%r8,%rdx
  401021:	48 89 d6             	mov    %rdx,%rsi
  401024:	48 c1 ee 3f          	shr    $0x3f,%rsi
  401028:	48 c1 fa 33          	sar    $0x33,%rdx
  40102c:	01 f2                	add    %esi,%edx
  40102e:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  401034:	41 29 d0             	sub    %edx,%r8d
  401037:	48 ff c1             	inc    %rcx
  40103a:	48 81 f9 41 1f 00 00 	cmp    $0x1f41,%rcx
  401041:	75 ad                	jne    400ff0 <main+0x3d0>
  401043:	49 f7 ec             	imul   %r12
  401046:	48 89 d0             	mov    %rdx,%rax
  401049:	48 c1 e8 3f          	shr    $0x3f,%rax
  40104d:	48 c1 fa 12          	sar    $0x12,%rdx
  401051:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401055:	bf 30 16 40 00       	mov    $0x401630,%edi
  40105a:	31 c0                	xor    %eax,%eax
  40105c:	4c 89 fe             	mov    %r15,%rsi
  40105f:	48 89 da             	mov    %rbx,%rdx
  401062:	e8 69 f6 ff ff       	callq  4006d0 <printf@plt>
  401067:	31 c0                	xor    %eax,%eax
  401069:	48 83 c4 28          	add    $0x28,%rsp
  40106d:	5b                   	pop    %rbx
  40106e:	41 5c                	pop    %r12
  401070:	41 5d                	pop    %r13
  401072:	41 5e                	pop    %r14
  401074:	41 5f                	pop    %r15
  401076:	5d                   	pop    %rbp
  401077:	c3                   	retq   
  401078:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40107f:	00 

0000000000401080 <_Z19codegen_reset_forceP4Body>:
  401080:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  401087:	00 00 00 00 00 
  40108c:	48 c7 04 fd 88 fc 65 	movq   $0x0,0x65fc88(,%rdi,8)
  401093:	00 00 00 00 00 
  401098:	c3                   	retq   
  401099:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004010a0 <_Z17codegen_add_forceP4BodyS0_>:
  4010a0:	41 56                	push   %r14
  4010a2:	53                   	push   %rbx
  4010a3:	48 83 ec 18          	sub    $0x18,%rsp
  4010a7:	49 89 f6             	mov    %rsi,%r14
  4010aa:	48 89 fb             	mov    %rdi,%rbx
  4010ad:	4c 39 f3             	cmp    %r14,%rbx
  4010b0:	0f 84 cb 00 00 00    	je     401181 <_Z17codegen_add_forceP4BodyS0_+0xe1>
  4010b6:	f2 42 0f 10 24 f5 88 	movsd  0x611a88(,%r14,8),%xmm4
  4010bd:	1a 61 00 
  4010c0:	f2 0f 5c 24 dd 88 1a 	subsd  0x611a88(,%rbx,8),%xmm4
  4010c7:	61 00 
  4010c9:	f2 42 0f 10 1c f5 88 	movsd  0x621488(,%r14,8),%xmm3
  4010d0:	14 62 00 
  4010d3:	f2 0f 5c 1c dd 88 14 	subsd  0x621488(,%rbx,8),%xmm3
  4010da:	62 00 
  4010dc:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4010e0:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  4010e4:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4010e8:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  4010ec:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4010f0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4010f3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4010f7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4010fb:	7b 21                	jnp    40111e <_Z17codegen_add_forceP4BodyS0_+0x7e>
  4010fd:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401101:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  401107:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40110d:	e8 2e f6 ff ff       	callq  400740 <sqrt@plt>
  401112:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  401118:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  40111e:	f2 0f 10 0c dd 88 20 	movsd  0x602088(,%rbx,8),%xmm1
  401125:	60 00 
  401127:	f2 0f 59 0d d9 04 00 	mulsd  0x4d9(%rip),%xmm1        # 401608 <_IO_stdin_used+0x38>
  40112e:	00 
  40112f:	f2 42 0f 59 0c f5 88 	mulsd  0x602088(,%r14,8),%xmm1
  401136:	20 60 00 
  401139:	66 0f 28 d0          	movapd %xmm0,%xmm2
  40113d:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  401141:	f2 0f 58 15 c7 04 00 	addsd  0x4c7(%rip),%xmm2        # 401610 <_IO_stdin_used+0x40>
  401148:	00 
  401149:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  40114d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401151:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401155:	f2 0f 58 24 dd 88 02 	addsd  0x650288(,%rbx,8),%xmm4
  40115c:	65 00 
  40115e:	f2 0f 11 24 dd 88 02 	movsd  %xmm4,0x650288(,%rbx,8)
  401165:	65 00 
  401167:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40116b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40116f:	f2 0f 58 1c dd 88 fc 	addsd  0x65fc88(,%rbx,8),%xmm3
  401176:	65 00 
  401178:	f2 0f 11 1c dd 88 fc 	movsd  %xmm3,0x65fc88(,%rbx,8)
  40117f:	65 00 
  401181:	48 83 c4 18          	add    $0x18,%rsp
  401185:	5b                   	pop    %rbx
  401186:	41 5e                	pop    %r14
  401188:	c3                   	retq   
  401189:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401190 <_Z14codegen_updateP4Bodyd>:
  401190:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  401197:	65 00 
  401199:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40119d:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4011a4:	60 00 
  4011a6:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  4011ad:	63 00 
  4011af:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  4011b6:	63 00 
  4011b8:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  4011bf:	65 00 
  4011c1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011c5:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4011cc:	60 00 
  4011ce:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  4011d5:	64 00 
  4011d7:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  4011de:	64 00 
  4011e0:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  4011e7:	63 00 
  4011e9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011ed:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  4011f4:	61 00 
  4011f6:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  4011fd:	61 00 
  4011ff:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  401206:	64 00 
  401208:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  40120f:	62 00 
  401211:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  401218:	62 00 
  40121a:	f2 0f 10 14 fd 88 1a 	movsd  0x611a88(,%rdi,8),%xmm2
  401221:	61 00 
  401223:	f2 0f 10 0d c5 03 00 	movsd  0x3c5(%rip),%xmm1        # 4015f0 <_IO_stdin_used+0x20>
  40122a:	00 
  40122b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40122f:	77 0a                	ja     40123b <_Z14codegen_updateP4Bodyd+0xab>
  401231:	66 0f 2e 15 c7 03 00 	ucomisd 0x3c7(%rip),%xmm2        # 401600 <_IO_stdin_used+0x30>
  401238:	00 
  401239:	76 23                	jbe    40125e <_Z14codegen_updateP4Bodyd+0xce>
  40123b:	f2 0f 10 04 fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm0
  401242:	63 00 
  401244:	66 0f 57 05 d4 03 00 	xorpd  0x3d4(%rip),%xmm0        # 401620 <_IO_stdin_used+0x50>
  40124b:	00 
  40124c:	66 0f 13 04 fd 88 0e 	movlpd %xmm0,0x630e88(,%rdi,8)
  401253:	63 00 
  401255:	f2 0f 10 04 fd 88 14 	movsd  0x621488(,%rdi,8),%xmm0
  40125c:	62 00 
  40125e:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  401262:	77 0a                	ja     40126e <_Z14codegen_updateP4Bodyd+0xde>
  401264:	66 0f 2e 05 94 03 00 	ucomisd 0x394(%rip),%xmm0        # 401600 <_IO_stdin_used+0x30>
  40126b:	00 
  40126c:	76 1a                	jbe    401288 <_Z14codegen_updateP4Bodyd+0xf8>
  40126e:	f2 0f 10 04 fd 88 08 	movsd  0x640888(,%rdi,8),%xmm0
  401275:	64 00 
  401277:	66 0f 57 05 a1 03 00 	xorpd  0x3a1(%rip),%xmm0        # 401620 <_IO_stdin_used+0x50>
  40127e:	00 
  40127f:	66 0f 13 04 fd 88 08 	movlpd %xmm0,0x640888(,%rdi,8)
  401286:	64 00 
  401288:	c3                   	retq   
  401289:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401290 <_Z26codegen_reset_force_manualP4Body>:
  401290:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  401297:	00 00 00 00 00 
  40129c:	48 c7 04 fd 88 fc 65 	movq   $0x0,0x65fc88(,%rdi,8)
  4012a3:	00 00 00 00 00 
  4012a8:	c3                   	retq   
  4012a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004012b0 <_Z24codegen_reset_force_halfP4Body>:
  4012b0:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  4012b7:	00 00 00 00 00 
  4012bc:	c3                   	retq   
  4012bd:	0f 1f 00             	nopl   (%rax)

00000000004012c0 <_Z18codegen_reset_massP4Body>:
  4012c0:	48 c7 04 fd 88 20 60 	movq   $0x0,0x602088(,%rdi,8)
  4012c7:	00 00 00 00 00 
  4012cc:	c3                   	retq   
  4012cd:	0f 1f 00             	nopl   (%rax)

00000000004012d0 <_Z22codengen_simple_updateP4Bodyd>:
  4012d0:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  4012d7:	65 00 
  4012d9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4012dd:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4012e4:	60 00 
  4012e6:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  4012ed:	63 00 
  4012ef:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  4012f6:	63 00 
  4012f8:	f2 0f 10 14 fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm2
  4012ff:	65 00 
  401301:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401305:	f2 0f 5e 14 fd 88 20 	divsd  0x602088(,%rdi,8),%xmm2
  40130c:	60 00 
  40130e:	f2 0f 58 14 fd 88 08 	addsd  0x640888(,%rdi,8),%xmm2
  401315:	64 00 
  401317:	f2 0f 11 14 fd 88 08 	movsd  %xmm2,0x640888(,%rdi,8)
  40131e:	64 00 
  401320:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401324:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  40132b:	61 00 
  40132d:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  401334:	61 00 
  401336:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  40133d:	64 00 
  40133f:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  401346:	62 00 
  401348:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  40134f:	62 00 
  401351:	c3                   	retq   
  401352:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401359:	00 00 00 
  40135c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401360 <_ZN4Body6updateEd>:
  401360:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  401367:	65 00 
  401369:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40136d:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  401374:	60 00 
  401376:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  40137d:	63 00 
  40137f:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  401386:	63 00 
  401388:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  40138f:	65 00 
  401391:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401395:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  40139c:	60 00 
  40139e:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  4013a5:	64 00 
  4013a7:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  4013ae:	64 00 
  4013b0:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  4013b7:	63 00 
  4013b9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4013bd:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  4013c4:	61 00 
  4013c6:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  4013cd:	61 00 
  4013cf:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  4013d6:	64 00 
  4013d8:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  4013df:	62 00 
  4013e1:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  4013e8:	62 00 
  4013ea:	f2 0f 10 14 fd 88 1a 	movsd  0x611a88(,%rdi,8),%xmm2
  4013f1:	61 00 
  4013f3:	f2 0f 10 0d f5 01 00 	movsd  0x1f5(%rip),%xmm1        # 4015f0 <_IO_stdin_used+0x20>
  4013fa:	00 
  4013fb:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  4013ff:	77 0a                	ja     40140b <_ZN4Body6updateEd+0xab>
  401401:	66 0f 2e 15 f7 01 00 	ucomisd 0x1f7(%rip),%xmm2        # 401600 <_IO_stdin_used+0x30>
  401408:	00 
  401409:	76 23                	jbe    40142e <_ZN4Body6updateEd+0xce>
  40140b:	f2 0f 10 04 fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm0
  401412:	63 00 
  401414:	66 0f 57 05 04 02 00 	xorpd  0x204(%rip),%xmm0        # 401620 <_IO_stdin_used+0x50>
  40141b:	00 
  40141c:	66 0f 13 04 fd 88 0e 	movlpd %xmm0,0x630e88(,%rdi,8)
  401423:	63 00 
  401425:	f2 0f 10 04 fd 88 14 	movsd  0x621488(,%rdi,8),%xmm0
  40142c:	62 00 
  40142e:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  401432:	77 0a                	ja     40143e <_ZN4Body6updateEd+0xde>
  401434:	66 0f 2e 05 c4 01 00 	ucomisd 0x1c4(%rip),%xmm0        # 401600 <_IO_stdin_used+0x30>
  40143b:	00 
  40143c:	76 1a                	jbe    401458 <_ZN4Body6updateEd+0xf8>
  40143e:	f2 0f 10 04 fd 88 08 	movsd  0x640888(,%rdi,8),%xmm0
  401445:	64 00 
  401447:	66 0f 57 05 d1 01 00 	xorpd  0x1d1(%rip),%xmm0        # 401620 <_IO_stdin_used+0x50>
  40144e:	00 
  40144f:	66 0f 13 04 fd 88 08 	movlpd %xmm0,0x640888(,%rdi,8)
  401456:	64 00 
  401458:	c3                   	retq   
  401459:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401460 <_ZN4Body9add_forceEPS_>:
  401460:	41 56                	push   %r14
  401462:	53                   	push   %rbx
  401463:	48 83 ec 18          	sub    $0x18,%rsp
  401467:	49 89 f6             	mov    %rsi,%r14
  40146a:	48 89 fb             	mov    %rdi,%rbx
  40146d:	4c 39 f3             	cmp    %r14,%rbx
  401470:	0f 84 cb 00 00 00    	je     401541 <_ZN4Body9add_forceEPS_+0xe1>
  401476:	f2 42 0f 10 24 f5 88 	movsd  0x611a88(,%r14,8),%xmm4
  40147d:	1a 61 00 
  401480:	f2 0f 5c 24 dd 88 1a 	subsd  0x611a88(,%rbx,8),%xmm4
  401487:	61 00 
  401489:	f2 42 0f 10 1c f5 88 	movsd  0x621488(,%r14,8),%xmm3
  401490:	14 62 00 
  401493:	f2 0f 5c 1c dd 88 14 	subsd  0x621488(,%rbx,8),%xmm3
  40149a:	62 00 
  40149c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4014a0:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  4014a4:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4014a8:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  4014ac:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4014b0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4014b3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4014b7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4014bb:	7b 21                	jnp    4014de <_ZN4Body9add_forceEPS_+0x7e>
  4014bd:	66 0f 28 c1          	movapd %xmm1,%xmm0
  4014c1:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  4014c7:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  4014cd:	e8 6e f2 ff ff       	callq  400740 <sqrt@plt>
  4014d2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4014d8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4014de:	f2 0f 10 0c dd 88 20 	movsd  0x602088(,%rbx,8),%xmm1
  4014e5:	60 00 
  4014e7:	f2 0f 59 0d 19 01 00 	mulsd  0x119(%rip),%xmm1        # 401608 <_IO_stdin_used+0x38>
  4014ee:	00 
  4014ef:	f2 42 0f 59 0c f5 88 	mulsd  0x602088(,%r14,8),%xmm1
  4014f6:	20 60 00 
  4014f9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4014fd:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  401501:	f2 0f 58 15 07 01 00 	addsd  0x107(%rip),%xmm2        # 401610 <_IO_stdin_used+0x40>
  401508:	00 
  401509:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  40150d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401511:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401515:	f2 0f 58 24 dd 88 02 	addsd  0x650288(,%rbx,8),%xmm4
  40151c:	65 00 
  40151e:	f2 0f 11 24 dd 88 02 	movsd  %xmm4,0x650288(,%rbx,8)
  401525:	65 00 
  401527:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40152b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40152f:	f2 0f 58 1c dd 88 fc 	addsd  0x65fc88(,%rbx,8),%xmm3
  401536:	65 00 
  401538:	f2 0f 11 1c dd 88 fc 	movsd  %xmm3,0x65fc88(,%rbx,8)
  40153f:	65 00 
  401541:	48 83 c4 18          	add    $0x18,%rsp
  401545:	5b                   	pop    %rbx
  401546:	41 5e                	pop    %r14
  401548:	c3                   	retq   
  401549:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401550 <__libc_csu_init>:
  401550:	41 57                	push   %r15
  401552:	41 56                	push   %r14
  401554:	41 89 ff             	mov    %edi,%r15d
  401557:	41 55                	push   %r13
  401559:	41 54                	push   %r12
  40155b:	4c 8d 25 76 08 20 00 	lea    0x200876(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  401562:	55                   	push   %rbp
  401563:	48 8d 2d 7e 08 20 00 	lea    0x20087e(%rip),%rbp        # 601de8 <__init_array_end>
  40156a:	53                   	push   %rbx
  40156b:	49 89 f6             	mov    %rsi,%r14
  40156e:	49 89 d5             	mov    %rdx,%r13
  401571:	4c 29 e5             	sub    %r12,%rbp
  401574:	48 83 ec 08          	sub    $0x8,%rsp
  401578:	48 c1 fd 03          	sar    $0x3,%rbp
  40157c:	e8 17 f1 ff ff       	callq  400698 <_init>
  401581:	48 85 ed             	test   %rbp,%rbp
  401584:	74 20                	je     4015a6 <__libc_csu_init+0x56>
  401586:	31 db                	xor    %ebx,%ebx
  401588:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40158f:	00 
  401590:	4c 89 ea             	mov    %r13,%rdx
  401593:	4c 89 f6             	mov    %r14,%rsi
  401596:	44 89 ff             	mov    %r15d,%edi
  401599:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40159d:	48 83 c3 01          	add    $0x1,%rbx
  4015a1:	48 39 eb             	cmp    %rbp,%rbx
  4015a4:	75 ea                	jne    401590 <__libc_csu_init+0x40>
  4015a6:	48 83 c4 08          	add    $0x8,%rsp
  4015aa:	5b                   	pop    %rbx
  4015ab:	5d                   	pop    %rbp
  4015ac:	41 5c                	pop    %r12
  4015ae:	41 5d                	pop    %r13
  4015b0:	41 5e                	pop    %r14
  4015b2:	41 5f                	pop    %r15
  4015b4:	c3                   	retq   
  4015b5:	90                   	nop
  4015b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4015bd:	00 00 00 

00000000004015c0 <__libc_csu_fini>:
  4015c0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004015c4 <_fini>:
  4015c4:	48 83 ec 08          	sub    $0x8,%rsp
  4015c8:	48 83 c4 08          	add    $0x8,%rsp
  4015cc:	c3                   	retq   
