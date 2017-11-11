
bin/clang++-3.8_nbody_ikracpp_field:     file format elf64-x86-64


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

0000000000400760 <_GLOBAL__sub_I_ikracpp_field.cc>:
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
  40078f:	49 c7 c0 b0 18 40 00 	mov    $0x4018b0,%r8
  400796:	48 c7 c1 40 18 40 00 	mov    $0x401840,%rcx
  40079d:	48 c7 c7 b0 10 40 00 	mov    $0x4010b0,%rdi
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
  4008bc:	f2 0f 10 05 04 10 00 	movsd  0x1004(%rip),%xmm0        # 4018c8 <_IO_stdin_used+0x8>
  4008c3:	00 
  4008c4:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  4008c8:	f2 0f 10 05 00 10 00 	movsd  0x1000(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  4008cf:	00 
  4008d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008d4:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4008d8:	f2 0f 10 05 f8 0f 00 	movsd  0xff8(%rip),%xmm0        # 4018d8 <_IO_stdin_used+0x18>
  4008df:	00 
  4008e0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4008e4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  4008ea:	e8 41 fe ff ff       	callq  400730 <rand@plt>
  4008ef:	0f 57 c9             	xorps  %xmm1,%xmm1
  4008f2:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  4008f6:	f2 0f 5e 0d ca 0f 00 	divsd  0xfca(%rip),%xmm1        # 4018c8 <_IO_stdin_used+0x8>
  4008fd:	00 
  4008fe:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400902:	f2 0f 10 05 d6 0f 00 	movsd  0xfd6(%rip),%xmm0        # 4018e0 <_IO_stdin_used+0x20>
  400909:	00 
  40090a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40090e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400914:	e8 17 fe ff ff       	callq  400730 <rand@plt>
  400919:	0f 57 c0             	xorps  %xmm0,%xmm0
  40091c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400920:	f2 0f 5e 05 a0 0f 00 	divsd  0xfa0(%rip),%xmm0        # 4018c8 <_IO_stdin_used+0x8>
  400927:	00 
  400928:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40092c:	f2 0f 58 05 ac 0f 00 	addsd  0xfac(%rip),%xmm0        # 4018e0 <_IO_stdin_used+0x20>
  400933:	00 
  400934:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40093a:	e8 f1 fd ff ff       	callq  400730 <rand@plt>
  40093f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400942:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400946:	f2 0f 5e 0d 7a 0f 00 	divsd  0xf7a(%rip),%xmm1        # 4018c8 <_IO_stdin_used+0x8>
  40094d:	00 
  40094e:	f2 0f 10 05 92 0f 00 	movsd  0xf92(%rip),%xmm0        # 4018e8 <_IO_stdin_used+0x28>
  400955:	00 
  400956:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40095a:	f2 0f 5e 0d 76 0f 00 	divsd  0xf76(%rip),%xmm1        # 4018d8 <_IO_stdin_used+0x18>
  400961:	00 
  400962:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400968:	e8 c3 fd ff ff       	callq  400730 <rand@plt>
  40096d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400970:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400974:	f2 0f 5e 05 4c 0f 00 	divsd  0xf4c(%rip),%xmm0        # 4018c8 <_IO_stdin_used+0x8>
  40097b:	00 
  40097c:	f2 0f 58 05 64 0f 00 	addsd  0xf64(%rip),%xmm0        # 4018e8 <_IO_stdin_used+0x28>
  400983:	00 
  400984:	f2 0f 5e 05 4c 0f 00 	divsd  0xf4c(%rip),%xmm0        # 4018d8 <_IO_stdin_used+0x18>
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
  400a14:	53                   	push   %rbx
  400a15:	b8 08 00 00 00       	mov    $0x8,%eax
  400a1a:	45 31 ff             	xor    %r15d,%r15d
  400a1d:	0f 1f 00             	nopl   (%rax)
  400a20:	48 c7 04 c5 50 02 65 	movq   $0x0,0x650250(,%rax,8)
  400a27:	00 00 00 00 00 
  400a2c:	48 c7 04 c5 50 fc 65 	movq   $0x0,0x65fc50(,%rax,8)
  400a33:	00 00 00 00 00 
  400a38:	48 c7 04 c5 58 02 65 	movq   $0x0,0x650258(,%rax,8)
  400a3f:	00 00 00 00 00 
  400a44:	48 c7 04 c5 58 fc 65 	movq   $0x0,0x65fc58(,%rax,8)
  400a4b:	00 00 00 00 00 
  400a50:	48 c7 04 c5 60 02 65 	movq   $0x0,0x650260(,%rax,8)
  400a57:	00 00 00 00 00 
  400a5c:	48 c7 04 c5 60 fc 65 	movq   $0x0,0x65fc60(,%rax,8)
  400a63:	00 00 00 00 00 
  400a68:	48 c7 04 c5 68 02 65 	movq   $0x0,0x650268(,%rax,8)
  400a6f:	00 00 00 00 00 
  400a74:	48 c7 04 c5 68 fc 65 	movq   $0x0,0x65fc68(,%rax,8)
  400a7b:	00 00 00 00 00 
  400a80:	48 c7 04 c5 70 02 65 	movq   $0x0,0x650270(,%rax,8)
  400a87:	00 00 00 00 00 
  400a8c:	48 c7 04 c5 70 fc 65 	movq   $0x0,0x65fc70(,%rax,8)
  400a93:	00 00 00 00 00 
  400a98:	48 c7 04 c5 78 02 65 	movq   $0x0,0x650278(,%rax,8)
  400a9f:	00 00 00 00 00 
  400aa4:	48 c7 04 c5 78 fc 65 	movq   $0x0,0x65fc78(,%rax,8)
  400aab:	00 00 00 00 00 
  400ab0:	48 c7 04 c5 80 02 65 	movq   $0x0,0x650280(,%rax,8)
  400ab7:	00 00 00 00 00 
  400abc:	48 c7 04 c5 80 fc 65 	movq   $0x0,0x65fc80(,%rax,8)
  400ac3:	00 00 00 00 00 
  400ac8:	48 c7 04 c5 88 02 65 	movq   $0x0,0x650288(,%rax,8)
  400acf:	00 00 00 00 00 
  400ad4:	48 c7 04 c5 88 fc 65 	movq   $0x0,0x65fc88(,%rax,8)
  400adb:	00 00 00 00 00 
  400ae0:	48 83 c0 08          	add    $0x8,%rax
  400ae4:	48 3d 48 1f 00 00    	cmp    $0x1f48,%rax
  400aea:	0f 85 30 ff ff ff    	jne    400a20 <_Z14run_simulationv+0x10>
  400af0:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400af6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400afd:	00 00 00 
  400b00:	49 ff c7             	inc    %r15
  400b03:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b0f:	00 
  400b10:	48 89 df             	mov    %rbx,%rdi
  400b13:	4c 89 fe             	mov    %r15,%rsi
  400b16:	e8 35 0c 00 00       	callq  401750 <_ZN4Body9add_forceEPS_>
  400b1b:	48 ff c3             	inc    %rbx
  400b1e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400b25:	75 e9                	jne    400b10 <_Z14run_simulationv+0x100>
  400b27:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400b2e:	75 d0                	jne    400b00 <_Z14run_simulationv+0xf0>
  400b30:	bb 08 00 00 00       	mov    $0x8,%ebx
  400b35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400b3c:	00 00 00 00 
  400b40:	4c 89 f7             	mov    %r14,%rdi
  400b43:	f2 0f 10 05 85 0d 00 	movsd  0xd85(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  400b4a:	00 
  400b4b:	e8 70 0a 00 00       	callq  4015c0 <_ZN4Body6updateEd>
  400b50:	49 ff c6             	inc    %r14
  400b53:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400b5a:	75 e4                	jne    400b40 <_Z14run_simulationv+0x130>
  400b5c:	45 31 ff             	xor    %r15d,%r15d
  400b5f:	90                   	nop
  400b60:	48 c7 04 dd 50 02 65 	movq   $0x0,0x650250(,%rbx,8)
  400b67:	00 00 00 00 00 
  400b6c:	48 c7 04 dd 50 fc 65 	movq   $0x0,0x65fc50(,%rbx,8)
  400b73:	00 00 00 00 00 
  400b78:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400b7f:	00 00 00 00 00 
  400b84:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400b8b:	00 00 00 00 00 
  400b90:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400b97:	00 00 00 00 00 
  400b9c:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400ba3:	00 00 00 00 00 
  400ba8:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400baf:	00 00 00 00 00 
  400bb4:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400bbb:	00 00 00 00 00 
  400bc0:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400bc7:	00 00 00 00 00 
  400bcc:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400bd3:	00 00 00 00 00 
  400bd8:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400bdf:	00 00 00 00 00 
  400be4:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400beb:	00 00 00 00 00 
  400bf0:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400bf7:	00 00 00 00 00 
  400bfc:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400c03:	00 00 00 00 00 
  400c08:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400c0f:	00 00 00 00 00 
  400c14:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400c1b:	00 00 00 00 00 
  400c20:	48 83 c3 08          	add    $0x8,%rbx
  400c24:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400c2b:	0f 85 2f ff ff ff    	jne    400b60 <_Z14run_simulationv+0x150>
  400c31:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400c37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400c3e:	00 00 
  400c40:	49 ff c7             	inc    %r15
  400c43:	bb 01 00 00 00       	mov    $0x1,%ebx
  400c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c4f:	00 
  400c50:	48 89 df             	mov    %rbx,%rdi
  400c53:	4c 89 fe             	mov    %r15,%rsi
  400c56:	e8 f5 0a 00 00       	callq  401750 <_ZN4Body9add_forceEPS_>
  400c5b:	48 ff c3             	inc    %rbx
  400c5e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400c65:	75 e9                	jne    400c50 <_Z14run_simulationv+0x240>
  400c67:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400c6e:	75 d0                	jne    400c40 <_Z14run_simulationv+0x230>
  400c70:	bb 08 00 00 00       	mov    $0x8,%ebx
  400c75:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400c7c:	00 00 00 00 
  400c80:	4c 89 f7             	mov    %r14,%rdi
  400c83:	f2 0f 10 05 45 0c 00 	movsd  0xc45(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  400c8a:	00 
  400c8b:	e8 30 09 00 00       	callq  4015c0 <_ZN4Body6updateEd>
  400c90:	49 ff c6             	inc    %r14
  400c93:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400c9a:	75 e4                	jne    400c80 <_Z14run_simulationv+0x270>
  400c9c:	45 31 ff             	xor    %r15d,%r15d
  400c9f:	90                   	nop
  400ca0:	48 c7 04 dd 50 02 65 	movq   $0x0,0x650250(,%rbx,8)
  400ca7:	00 00 00 00 00 
  400cac:	48 c7 04 dd 50 fc 65 	movq   $0x0,0x65fc50(,%rbx,8)
  400cb3:	00 00 00 00 00 
  400cb8:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400cbf:	00 00 00 00 00 
  400cc4:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400ccb:	00 00 00 00 00 
  400cd0:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400cd7:	00 00 00 00 00 
  400cdc:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400ce3:	00 00 00 00 00 
  400ce8:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400cef:	00 00 00 00 00 
  400cf4:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400cfb:	00 00 00 00 00 
  400d00:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400d07:	00 00 00 00 00 
  400d0c:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400d13:	00 00 00 00 00 
  400d18:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400d1f:	00 00 00 00 00 
  400d24:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400d2b:	00 00 00 00 00 
  400d30:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400d37:	00 00 00 00 00 
  400d3c:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400d43:	00 00 00 00 00 
  400d48:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400d4f:	00 00 00 00 00 
  400d54:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400d5b:	00 00 00 00 00 
  400d60:	48 83 c3 08          	add    $0x8,%rbx
  400d64:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400d6b:	0f 85 2f ff ff ff    	jne    400ca0 <_Z14run_simulationv+0x290>
  400d71:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400d77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d7e:	00 00 
  400d80:	49 ff c7             	inc    %r15
  400d83:	bb 01 00 00 00       	mov    $0x1,%ebx
  400d88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d8f:	00 
  400d90:	48 89 df             	mov    %rbx,%rdi
  400d93:	4c 89 fe             	mov    %r15,%rsi
  400d96:	e8 b5 09 00 00       	callq  401750 <_ZN4Body9add_forceEPS_>
  400d9b:	48 ff c3             	inc    %rbx
  400d9e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400da5:	75 e9                	jne    400d90 <_Z14run_simulationv+0x380>
  400da7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dae:	75 d0                	jne    400d80 <_Z14run_simulationv+0x370>
  400db0:	bb 08 00 00 00       	mov    $0x8,%ebx
  400db5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400dbc:	00 00 00 00 
  400dc0:	4c 89 f7             	mov    %r14,%rdi
  400dc3:	f2 0f 10 05 05 0b 00 	movsd  0xb05(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  400dca:	00 
  400dcb:	e8 f0 07 00 00       	callq  4015c0 <_ZN4Body6updateEd>
  400dd0:	49 ff c6             	inc    %r14
  400dd3:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400dda:	75 e4                	jne    400dc0 <_Z14run_simulationv+0x3b0>
  400ddc:	45 31 ff             	xor    %r15d,%r15d
  400ddf:	90                   	nop
  400de0:	48 c7 04 dd 50 02 65 	movq   $0x0,0x650250(,%rbx,8)
  400de7:	00 00 00 00 00 
  400dec:	48 c7 04 dd 50 fc 65 	movq   $0x0,0x65fc50(,%rbx,8)
  400df3:	00 00 00 00 00 
  400df8:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400dff:	00 00 00 00 00 
  400e04:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400e0b:	00 00 00 00 00 
  400e10:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400e17:	00 00 00 00 00 
  400e1c:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400e23:	00 00 00 00 00 
  400e28:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400e2f:	00 00 00 00 00 
  400e34:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400e3b:	00 00 00 00 00 
  400e40:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400e47:	00 00 00 00 00 
  400e4c:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400e53:	00 00 00 00 00 
  400e58:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400e5f:	00 00 00 00 00 
  400e64:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400e6b:	00 00 00 00 00 
  400e70:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400e77:	00 00 00 00 00 
  400e7c:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400e83:	00 00 00 00 00 
  400e88:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400e8f:	00 00 00 00 00 
  400e94:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400e9b:	00 00 00 00 00 
  400ea0:	48 83 c3 08          	add    $0x8,%rbx
  400ea4:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400eab:	0f 85 2f ff ff ff    	jne    400de0 <_Z14run_simulationv+0x3d0>
  400eb1:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400eb7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400ebe:	00 00 
  400ec0:	49 ff c7             	inc    %r15
  400ec3:	bb 01 00 00 00       	mov    $0x1,%ebx
  400ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ecf:	00 
  400ed0:	48 89 df             	mov    %rbx,%rdi
  400ed3:	4c 89 fe             	mov    %r15,%rsi
  400ed6:	e8 75 08 00 00       	callq  401750 <_ZN4Body9add_forceEPS_>
  400edb:	48 ff c3             	inc    %rbx
  400ede:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400ee5:	75 e9                	jne    400ed0 <_Z14run_simulationv+0x4c0>
  400ee7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400eee:	75 d0                	jne    400ec0 <_Z14run_simulationv+0x4b0>
  400ef0:	bb 08 00 00 00       	mov    $0x8,%ebx
  400ef5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400efc:	00 00 00 00 
  400f00:	4c 89 f7             	mov    %r14,%rdi
  400f03:	f2 0f 10 05 c5 09 00 	movsd  0x9c5(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  400f0a:	00 
  400f0b:	e8 b0 06 00 00       	callq  4015c0 <_ZN4Body6updateEd>
  400f10:	49 ff c6             	inc    %r14
  400f13:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400f1a:	75 e4                	jne    400f00 <_Z14run_simulationv+0x4f0>
  400f1c:	45 31 ff             	xor    %r15d,%r15d
  400f1f:	90                   	nop
  400f20:	48 c7 04 dd 50 02 65 	movq   $0x0,0x650250(,%rbx,8)
  400f27:	00 00 00 00 00 
  400f2c:	48 c7 04 dd 50 fc 65 	movq   $0x0,0x65fc50(,%rbx,8)
  400f33:	00 00 00 00 00 
  400f38:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400f3f:	00 00 00 00 00 
  400f44:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400f4b:	00 00 00 00 00 
  400f50:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400f57:	00 00 00 00 00 
  400f5c:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400f63:	00 00 00 00 00 
  400f68:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400f6f:	00 00 00 00 00 
  400f74:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400f7b:	00 00 00 00 00 
  400f80:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400f87:	00 00 00 00 00 
  400f8c:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400f93:	00 00 00 00 00 
  400f98:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400f9f:	00 00 00 00 00 
  400fa4:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400fab:	00 00 00 00 00 
  400fb0:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400fb7:	00 00 00 00 00 
  400fbc:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400fc3:	00 00 00 00 00 
  400fc8:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400fcf:	00 00 00 00 00 
  400fd4:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400fdb:	00 00 00 00 00 
  400fe0:	48 83 c3 08          	add    $0x8,%rbx
  400fe4:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400feb:	0f 85 2f ff ff ff    	jne    400f20 <_Z14run_simulationv+0x510>
  400ff1:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400ff7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400ffe:	00 00 
  401000:	49 ff c7             	inc    %r15
  401003:	bb 01 00 00 00       	mov    $0x1,%ebx
  401008:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40100f:	00 
  401010:	48 89 df             	mov    %rbx,%rdi
  401013:	4c 89 fe             	mov    %r15,%rsi
  401016:	e8 35 07 00 00       	callq  401750 <_ZN4Body9add_forceEPS_>
  40101b:	48 ff c3             	inc    %rbx
  40101e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  401025:	75 e9                	jne    401010 <_Z14run_simulationv+0x600>
  401027:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  40102e:	75 d0                	jne    401000 <_Z14run_simulationv+0x5f0>
  401030:	4c 89 f7             	mov    %r14,%rdi
  401033:	f2 0f 10 05 95 08 00 	movsd  0x895(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  40103a:	00 
  40103b:	e8 80 05 00 00       	callq  4015c0 <_ZN4Body6updateEd>
  401040:	49 ff c6             	inc    %r14
  401043:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  40104a:	75 e4                	jne    401030 <_Z14run_simulationv+0x620>
  40104c:	5b                   	pop    %rbx
  40104d:	41 5e                	pop    %r14
  40104f:	41 5f                	pop    %r15
  401051:	c3                   	retq   
  401052:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401059:	1f 84 00 00 00 00 00 

0000000000401060 <_Z10run_simplev>:
  401060:	55                   	push   %rbp
  401061:	53                   	push   %rbx
  401062:	50                   	push   %rax
  401063:	31 ed                	xor    %ebp,%ebp
  401065:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40106c:	00 00 00 00 
  401070:	bb 01 00 00 00       	mov    $0x1,%ebx
  401075:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40107c:	00 00 00 00 
  401080:	48 89 df             	mov    %rbx,%rdi
  401083:	f2 0f 10 05 45 08 00 	movsd  0x845(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  40108a:	00 
  40108b:	e8 30 06 00 00       	callq  4016c0 <_ZN4Body22codengen_simple_updateEd>
  401090:	48 ff c3             	inc    %rbx
  401093:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  40109a:	75 e4                	jne    401080 <_Z10run_simplev+0x20>
  40109c:	ff c5                	inc    %ebp
  40109e:	81 fd 50 c3 00 00    	cmp    $0xc350,%ebp
  4010a4:	75 ca                	jne    401070 <_Z10run_simplev+0x10>
  4010a6:	48 83 c4 08          	add    $0x8,%rsp
  4010aa:	5b                   	pop    %rbx
  4010ab:	5d                   	pop    %rbp
  4010ac:	c3                   	retq   
  4010ad:	0f 1f 00             	nopl   (%rax)

00000000004010b0 <main>:
  4010b0:	41 57                	push   %r15
  4010b2:	41 56                	push   %r14
  4010b4:	41 55                	push   %r13
  4010b6:	41 54                	push   %r12
  4010b8:	53                   	push   %rbx
  4010b9:	0f 57 c0             	xorps  %xmm0,%xmm0
  4010bc:	0f 11 05 b5 0f 20 00 	movups %xmm0,0x200fb5(%rip)        # 602078 <__Bodydata_buffer>
  4010c3:	e8 58 f6 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010c8:	48 89 c3             	mov    %rax,%rbx
  4010cb:	e8 d0 f7 ff ff       	callq  4008a0 <_Z13instantiationv>
  4010d0:	e8 4b f6 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010d5:	49 89 c6             	mov    %rax,%r14
  4010d8:	49 29 de             	sub    %rbx,%r14
  4010db:	e8 40 f6 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010e0:	48 89 c3             	mov    %rax,%rbx
  4010e3:	e8 28 f9 ff ff       	callq  400a10 <_Z14run_simulationv>
  4010e8:	e8 33 f6 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010ed:	49 89 c4             	mov    %rax,%r12
  4010f0:	49 29 dc             	sub    %rbx,%r12
  4010f3:	31 db                	xor    %ebx,%ebx
  4010f5:	e8 26 f6 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010fa:	49 89 c7             	mov    %rax,%r15
  4010fd:	f2 0f 10 05 cb 07 00 	movsd  0x7cb(%rip),%xmm0        # 4018d0 <_IO_stdin_used+0x10>
  401104:	00 
  401105:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40110c:	00 00 00 00 
  401110:	48 c7 c0 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rax
  401117:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40111e:	00 00 
  401120:	f2 0f 10 0c c5 90 fc 	movsd  0x65fc90(,%rax,8),%xmm1
  401127:	65 00 
  401129:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40112d:	f2 0f 5e 0c c5 90 1a 	divsd  0x611a90(,%rax,8),%xmm1
  401134:	61 00 
  401136:	f2 0f 58 0c c5 90 08 	addsd  0x640890(,%rax,8),%xmm1
  40113d:	64 00 
  40113f:	f2 0f 11 0c c5 90 08 	movsd  %xmm1,0x640890(,%rax,8)
  401146:	64 00 
  401148:	f2 0f 10 0c c5 90 f6 	movsd  0x66f690(,%rax,8),%xmm1
  40114f:	66 00 
  401151:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401155:	f2 0f 5e 0c c5 90 1a 	divsd  0x611a90(,%rax,8),%xmm1
  40115c:	61 00 
  40115e:	f2 0f 58 0c c5 90 02 	addsd  0x650290(,%rax,8),%xmm1
  401165:	65 00 
  401167:	f2 0f 11 0c c5 90 02 	movsd  %xmm1,0x650290(,%rax,8)
  40116e:	65 00 
  401170:	f2 0f 10 0c c5 90 08 	movsd  0x640890(,%rax,8),%xmm1
  401177:	64 00 
  401179:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40117d:	f2 0f 58 0c c5 90 14 	addsd  0x621490(,%rax,8),%xmm1
  401184:	62 00 
  401186:	f2 0f 11 0c c5 90 14 	movsd  %xmm1,0x621490(,%rax,8)
  40118d:	62 00 
  40118f:	f2 0f 10 0c c5 90 02 	movsd  0x650290(,%rax,8),%xmm1
  401196:	65 00 
  401198:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40119c:	f2 0f 58 0c c5 90 0e 	addsd  0x630e90(,%rax,8),%xmm1
  4011a3:	63 00 
  4011a5:	f2 0f 11 0c c5 90 0e 	movsd  %xmm1,0x630e90(,%rax,8)
  4011ac:	63 00 
  4011ae:	48 ff c0             	inc    %rax
  4011b1:	0f 85 69 ff ff ff    	jne    401120 <main+0x70>
  4011b7:	ff c3                	inc    %ebx
  4011b9:	81 fb 50 c3 00 00    	cmp    $0xc350,%ebx
  4011bf:	0f 85 4b ff ff ff    	jne    401110 <main+0x60>
  4011c5:	49 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%r13
  4011cc:	de 1b 43 
  4011cf:	4c 89 f0             	mov    %r14,%rax
  4011d2:	49 f7 ed             	imul   %r13
  4011d5:	49 89 d6             	mov    %rdx,%r14
  4011d8:	4c 89 f0             	mov    %r14,%rax
  4011db:	48 c1 e8 3f          	shr    $0x3f,%rax
  4011df:	49 c1 fe 12          	sar    $0x12,%r14
  4011e3:	49 01 c6             	add    %rax,%r14
  4011e6:	4c 89 e0             	mov    %r12,%rax
  4011e9:	49 f7 ed             	imul   %r13
  4011ec:	48 89 d3             	mov    %rdx,%rbx
  4011ef:	48 89 d8             	mov    %rbx,%rax
  4011f2:	48 c1 e8 3f          	shr    $0x3f,%rax
  4011f6:	48 c1 fb 12          	sar    $0x12,%rbx
  4011fa:	48 01 c3             	add    %rax,%rbx
  4011fd:	e8 1e f5 ff ff       	callq  400720 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401202:	4c 29 f8             	sub    %r15,%rax
  401205:	41 b8 0b 00 00 00    	mov    $0xb,%r8d
  40120b:	b9 02 00 00 00       	mov    $0x2,%ecx
  401210:	f2 0f 10 04 cd 80 1a 	movsd  0x611a80(,%rcx,8),%xmm0
  401217:	61 00 
  401219:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  40121d:	66 0f 7e c2          	movd   %xmm0,%edx
  401221:	44 01 c2             	add    %r8d,%edx
  401224:	f2 0f 10 04 cd 80 14 	movsd  0x621480(,%rcx,8),%xmm0
  40122b:	62 00 
  40122d:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  401231:	66 0f 7e c6          	movd   %xmm0,%esi
  401235:	01 d6                	add    %edx,%esi
  401237:	48 63 d6             	movslq %esi,%rdx
  40123a:	48 69 f2 8d 68 b7 6c 	imul   $0x6cb7688d,%rdx,%rsi
  401241:	48 89 f7             	mov    %rsi,%rdi
  401244:	48 c1 ef 3f          	shr    $0x3f,%rdi
  401248:	48 c1 fe 33          	sar    $0x33,%rsi
  40124c:	01 fe                	add    %edi,%esi
  40124e:	69 f6 87 d6 12 00    	imul   $0x12d687,%esi,%esi
  401254:	29 f2                	sub    %esi,%edx
  401256:	f2 0f 10 04 cd 88 1a 	movsd  0x611a88(,%rcx,8),%xmm0
  40125d:	61 00 
  40125f:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  401263:	66 0f 7e c6          	movd   %xmm0,%esi
  401267:	01 d6                	add    %edx,%esi
  401269:	f2 0f 10 04 cd 88 14 	movsd  0x621488(,%rcx,8),%xmm0
  401270:	62 00 
  401272:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  401276:	66 0f 7e c2          	movd   %xmm0,%edx
  40127a:	01 f2                	add    %esi,%edx
  40127c:	4c 63 c2             	movslq %edx,%r8
  40127f:	49 69 d0 8d 68 b7 6c 	imul   $0x6cb7688d,%r8,%rdx
  401286:	48 89 d6             	mov    %rdx,%rsi
  401289:	48 c1 ee 3f          	shr    $0x3f,%rsi
  40128d:	48 c1 fa 33          	sar    $0x33,%rdx
  401291:	01 f2                	add    %esi,%edx
  401293:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  401299:	41 29 d0             	sub    %edx,%r8d
  40129c:	48 83 c1 02          	add    $0x2,%rcx
  4012a0:	48 81 f9 42 1f 00 00 	cmp    $0x1f42,%rcx
  4012a7:	0f 85 63 ff ff ff    	jne    401210 <main+0x160>
  4012ad:	49 f7 ed             	imul   %r13
  4012b0:	48 89 d0             	mov    %rdx,%rax
  4012b3:	48 c1 e8 3f          	shr    $0x3f,%rax
  4012b7:	48 c1 fa 12          	sar    $0x12,%rdx
  4012bb:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  4012bf:	bf 20 19 40 00       	mov    $0x401920,%edi
  4012c4:	31 c0                	xor    %eax,%eax
  4012c6:	4c 89 f6             	mov    %r14,%rsi
  4012c9:	48 89 da             	mov    %rbx,%rdx
  4012cc:	e8 ff f3 ff ff       	callq  4006d0 <printf@plt>
  4012d1:	31 c0                	xor    %eax,%eax
  4012d3:	5b                   	pop    %rbx
  4012d4:	41 5c                	pop    %r12
  4012d6:	41 5d                	pop    %r13
  4012d8:	41 5e                	pop    %r14
  4012da:	41 5f                	pop    %r15
  4012dc:	c3                   	retq   
  4012dd:	0f 1f 00             	nopl   (%rax)

00000000004012e0 <_Z19codegen_reset_forceP4Body>:
  4012e0:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  4012e7:	00 00 00 00 00 
  4012ec:	48 c7 04 fd 88 fc 65 	movq   $0x0,0x65fc88(,%rdi,8)
  4012f3:	00 00 00 00 00 
  4012f8:	c3                   	retq   
  4012f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401300 <_Z17codegen_add_forceP4BodyS0_>:
  401300:	41 56                	push   %r14
  401302:	53                   	push   %rbx
  401303:	48 83 ec 18          	sub    $0x18,%rsp
  401307:	49 89 f6             	mov    %rsi,%r14
  40130a:	48 89 fb             	mov    %rdi,%rbx
  40130d:	4c 39 f3             	cmp    %r14,%rbx
  401310:	0f 84 cb 00 00 00    	je     4013e1 <_Z17codegen_add_forceP4BodyS0_+0xe1>
  401316:	f2 42 0f 10 24 f5 88 	movsd  0x611a88(,%r14,8),%xmm4
  40131d:	1a 61 00 
  401320:	f2 0f 5c 24 dd 88 1a 	subsd  0x611a88(,%rbx,8),%xmm4
  401327:	61 00 
  401329:	f2 42 0f 10 1c f5 88 	movsd  0x621488(,%r14,8),%xmm3
  401330:	14 62 00 
  401333:	f2 0f 5c 1c dd 88 14 	subsd  0x621488(,%rbx,8),%xmm3
  40133a:	62 00 
  40133c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401340:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  401344:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401348:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40134c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  401350:	0f 57 c0             	xorps  %xmm0,%xmm0
  401353:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  401357:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  40135b:	7b 21                	jnp    40137e <_Z17codegen_add_forceP4BodyS0_+0x7e>
  40135d:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401361:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  401367:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40136d:	e8 ce f3 ff ff       	callq  400740 <sqrt@plt>
  401372:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  401378:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  40137e:	f2 0f 10 0c dd 88 20 	movsd  0x602088(,%rbx,8),%xmm1
  401385:	60 00 
  401387:	f2 0f 59 0d 69 05 00 	mulsd  0x569(%rip),%xmm1        # 4018f8 <_IO_stdin_used+0x38>
  40138e:	00 
  40138f:	f2 42 0f 59 0c f5 88 	mulsd  0x602088(,%r14,8),%xmm1
  401396:	20 60 00 
  401399:	66 0f 28 d0          	movapd %xmm0,%xmm2
  40139d:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  4013a1:	f2 0f 58 15 57 05 00 	addsd  0x557(%rip),%xmm2        # 401900 <_IO_stdin_used+0x40>
  4013a8:	00 
  4013a9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4013ad:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  4013b1:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  4013b5:	f2 0f 58 24 dd 88 02 	addsd  0x650288(,%rbx,8),%xmm4
  4013bc:	65 00 
  4013be:	f2 0f 11 24 dd 88 02 	movsd  %xmm4,0x650288(,%rbx,8)
  4013c5:	65 00 
  4013c7:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4013cb:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  4013cf:	f2 0f 58 1c dd 88 fc 	addsd  0x65fc88(,%rbx,8),%xmm3
  4013d6:	65 00 
  4013d8:	f2 0f 11 1c dd 88 fc 	movsd  %xmm3,0x65fc88(,%rbx,8)
  4013df:	65 00 
  4013e1:	48 83 c4 18          	add    $0x18,%rsp
  4013e5:	5b                   	pop    %rbx
  4013e6:	41 5e                	pop    %r14
  4013e8:	c3                   	retq   
  4013e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004013f0 <_Z14codegen_updateP4Bodyd>:
  4013f0:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  4013f7:	65 00 
  4013f9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4013fd:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  401404:	60 00 
  401406:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  40140d:	63 00 
  40140f:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  401416:	63 00 
  401418:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  40141f:	65 00 
  401421:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401425:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  40142c:	60 00 
  40142e:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  401435:	64 00 
  401437:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  40143e:	64 00 
  401440:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  401447:	63 00 
  401449:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40144d:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  401454:	61 00 
  401456:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  40145d:	61 00 
  40145f:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  401466:	64 00 
  401468:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  40146f:	62 00 
  401471:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  401478:	62 00 
  40147a:	f2 0f 10 14 fd 88 1a 	movsd  0x611a88(,%rdi,8),%xmm2
  401481:	61 00 
  401483:	f2 0f 10 0d 55 04 00 	movsd  0x455(%rip),%xmm1        # 4018e0 <_IO_stdin_used+0x20>
  40148a:	00 
  40148b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40148f:	77 0a                	ja     40149b <_Z14codegen_updateP4Bodyd+0xab>
  401491:	66 0f 2e 15 57 04 00 	ucomisd 0x457(%rip),%xmm2        # 4018f0 <_IO_stdin_used+0x30>
  401498:	00 
  401499:	76 23                	jbe    4014be <_Z14codegen_updateP4Bodyd+0xce>
  40149b:	f2 0f 10 04 fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm0
  4014a2:	63 00 
  4014a4:	66 0f 57 05 64 04 00 	xorpd  0x464(%rip),%xmm0        # 401910 <_IO_stdin_used+0x50>
  4014ab:	00 
  4014ac:	66 0f 13 04 fd 88 0e 	movlpd %xmm0,0x630e88(,%rdi,8)
  4014b3:	63 00 
  4014b5:	f2 0f 10 04 fd 88 14 	movsd  0x621488(,%rdi,8),%xmm0
  4014bc:	62 00 
  4014be:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  4014c2:	77 0a                	ja     4014ce <_Z14codegen_updateP4Bodyd+0xde>
  4014c4:	66 0f 2e 05 24 04 00 	ucomisd 0x424(%rip),%xmm0        # 4018f0 <_IO_stdin_used+0x30>
  4014cb:	00 
  4014cc:	76 1a                	jbe    4014e8 <_Z14codegen_updateP4Bodyd+0xf8>
  4014ce:	f2 0f 10 04 fd 88 08 	movsd  0x640888(,%rdi,8),%xmm0
  4014d5:	64 00 
  4014d7:	66 0f 57 05 31 04 00 	xorpd  0x431(%rip),%xmm0        # 401910 <_IO_stdin_used+0x50>
  4014de:	00 
  4014df:	66 0f 13 04 fd 88 08 	movlpd %xmm0,0x640888(,%rdi,8)
  4014e6:	64 00 
  4014e8:	c3                   	retq   
  4014e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004014f0 <_Z26codegen_reset_force_manualP4Body>:
  4014f0:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  4014f7:	00 00 00 00 00 
  4014fc:	48 c7 04 fd 88 fc 65 	movq   $0x0,0x65fc88(,%rdi,8)
  401503:	00 00 00 00 00 
  401508:	c3                   	retq   
  401509:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401510 <_Z24codegen_reset_force_halfP4Body>:
  401510:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  401517:	00 00 00 00 00 
  40151c:	c3                   	retq   
  40151d:	0f 1f 00             	nopl   (%rax)

0000000000401520 <_Z18codegen_reset_massP4Body>:
  401520:	48 c7 04 fd 88 20 60 	movq   $0x0,0x602088(,%rdi,8)
  401527:	00 00 00 00 00 
  40152c:	c3                   	retq   
  40152d:	0f 1f 00             	nopl   (%rax)

0000000000401530 <_Z22codengen_simple_updateP4Bodyd>:
  401530:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  401537:	65 00 
  401539:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40153d:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  401544:	60 00 
  401546:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  40154d:	63 00 
  40154f:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  401556:	63 00 
  401558:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  40155f:	65 00 
  401561:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401565:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  40156c:	60 00 
  40156e:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  401575:	64 00 
  401577:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  40157e:	64 00 
  401580:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  401587:	63 00 
  401589:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40158d:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  401594:	61 00 
  401596:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  40159d:	61 00 
  40159f:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  4015a6:	64 00 
  4015a8:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  4015af:	62 00 
  4015b1:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  4015b8:	62 00 
  4015ba:	c3                   	retq   
  4015bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004015c0 <_ZN4Body6updateEd>:
  4015c0:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  4015c7:	65 00 
  4015c9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015cd:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4015d4:	60 00 
  4015d6:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  4015dd:	63 00 
  4015df:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  4015e6:	63 00 
  4015e8:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  4015ef:	65 00 
  4015f1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015f5:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4015fc:	60 00 
  4015fe:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  401605:	64 00 
  401607:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  40160e:	64 00 
  401610:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  401617:	63 00 
  401619:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40161d:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  401624:	61 00 
  401626:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  40162d:	61 00 
  40162f:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  401636:	64 00 
  401638:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  40163f:	62 00 
  401641:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  401648:	62 00 
  40164a:	f2 0f 10 14 fd 88 1a 	movsd  0x611a88(,%rdi,8),%xmm2
  401651:	61 00 
  401653:	f2 0f 10 0d 85 02 00 	movsd  0x285(%rip),%xmm1        # 4018e0 <_IO_stdin_used+0x20>
  40165a:	00 
  40165b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40165f:	77 0a                	ja     40166b <_ZN4Body6updateEd+0xab>
  401661:	66 0f 2e 15 87 02 00 	ucomisd 0x287(%rip),%xmm2        # 4018f0 <_IO_stdin_used+0x30>
  401668:	00 
  401669:	76 23                	jbe    40168e <_ZN4Body6updateEd+0xce>
  40166b:	f2 0f 10 04 fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm0
  401672:	63 00 
  401674:	66 0f 57 05 94 02 00 	xorpd  0x294(%rip),%xmm0        # 401910 <_IO_stdin_used+0x50>
  40167b:	00 
  40167c:	66 0f 13 04 fd 88 0e 	movlpd %xmm0,0x630e88(,%rdi,8)
  401683:	63 00 
  401685:	f2 0f 10 04 fd 88 14 	movsd  0x621488(,%rdi,8),%xmm0
  40168c:	62 00 
  40168e:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  401692:	77 0a                	ja     40169e <_ZN4Body6updateEd+0xde>
  401694:	66 0f 2e 05 54 02 00 	ucomisd 0x254(%rip),%xmm0        # 4018f0 <_IO_stdin_used+0x30>
  40169b:	00 
  40169c:	76 1a                	jbe    4016b8 <_ZN4Body6updateEd+0xf8>
  40169e:	f2 0f 10 04 fd 88 08 	movsd  0x640888(,%rdi,8),%xmm0
  4016a5:	64 00 
  4016a7:	66 0f 57 05 61 02 00 	xorpd  0x261(%rip),%xmm0        # 401910 <_IO_stdin_used+0x50>
  4016ae:	00 
  4016af:	66 0f 13 04 fd 88 08 	movlpd %xmm0,0x640888(,%rdi,8)
  4016b6:	64 00 
  4016b8:	c3                   	retq   
  4016b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004016c0 <_ZN4Body22codengen_simple_updateEd>:
  4016c0:	f2 0f 10 0c fd 88 02 	movsd  0x650288(,%rdi,8),%xmm1
  4016c7:	65 00 
  4016c9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4016cd:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4016d4:	60 00 
  4016d6:	f2 0f 58 0c fd 88 0e 	addsd  0x630e88(,%rdi,8),%xmm1
  4016dd:	63 00 
  4016df:	f2 0f 11 0c fd 88 0e 	movsd  %xmm1,0x630e88(,%rdi,8)
  4016e6:	63 00 
  4016e8:	f2 0f 10 0c fd 88 fc 	movsd  0x65fc88(,%rdi,8),%xmm1
  4016ef:	65 00 
  4016f1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4016f5:	f2 0f 5e 0c fd 88 20 	divsd  0x602088(,%rdi,8),%xmm1
  4016fc:	60 00 
  4016fe:	f2 0f 58 0c fd 88 08 	addsd  0x640888(,%rdi,8),%xmm1
  401705:	64 00 
  401707:	f2 0f 11 0c fd 88 08 	movsd  %xmm1,0x640888(,%rdi,8)
  40170e:	64 00 
  401710:	f2 0f 10 0c fd 88 0e 	movsd  0x630e88(,%rdi,8),%xmm1
  401717:	63 00 
  401719:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40171d:	f2 0f 58 0c fd 88 1a 	addsd  0x611a88(,%rdi,8),%xmm1
  401724:	61 00 
  401726:	f2 0f 11 0c fd 88 1a 	movsd  %xmm1,0x611a88(,%rdi,8)
  40172d:	61 00 
  40172f:	f2 0f 59 04 fd 88 08 	mulsd  0x640888(,%rdi,8),%xmm0
  401736:	64 00 
  401738:	f2 0f 58 04 fd 88 14 	addsd  0x621488(,%rdi,8),%xmm0
  40173f:	62 00 
  401741:	f2 0f 11 04 fd 88 14 	movsd  %xmm0,0x621488(,%rdi,8)
  401748:	62 00 
  40174a:	c3                   	retq   
  40174b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401750 <_ZN4Body9add_forceEPS_>:
  401750:	41 56                	push   %r14
  401752:	53                   	push   %rbx
  401753:	48 83 ec 18          	sub    $0x18,%rsp
  401757:	49 89 f6             	mov    %rsi,%r14
  40175a:	48 89 fb             	mov    %rdi,%rbx
  40175d:	4c 39 f3             	cmp    %r14,%rbx
  401760:	0f 84 cb 00 00 00    	je     401831 <_ZN4Body9add_forceEPS_+0xe1>
  401766:	f2 42 0f 10 24 f5 88 	movsd  0x611a88(,%r14,8),%xmm4
  40176d:	1a 61 00 
  401770:	f2 0f 5c 24 dd 88 1a 	subsd  0x611a88(,%rbx,8),%xmm4
  401777:	61 00 
  401779:	f2 42 0f 10 1c f5 88 	movsd  0x621488(,%r14,8),%xmm3
  401780:	14 62 00 
  401783:	f2 0f 5c 1c dd 88 14 	subsd  0x621488(,%rbx,8),%xmm3
  40178a:	62 00 
  40178c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401790:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  401794:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401798:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40179c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4017a0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4017a3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4017a7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4017ab:	7b 21                	jnp    4017ce <_ZN4Body9add_forceEPS_+0x7e>
  4017ad:	66 0f 28 c1          	movapd %xmm1,%xmm0
  4017b1:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  4017b7:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  4017bd:	e8 7e ef ff ff       	callq  400740 <sqrt@plt>
  4017c2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4017c8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4017ce:	f2 0f 10 0c dd 88 20 	movsd  0x602088(,%rbx,8),%xmm1
  4017d5:	60 00 
  4017d7:	f2 0f 59 0d 19 01 00 	mulsd  0x119(%rip),%xmm1        # 4018f8 <_IO_stdin_used+0x38>
  4017de:	00 
  4017df:	f2 42 0f 59 0c f5 88 	mulsd  0x602088(,%r14,8),%xmm1
  4017e6:	20 60 00 
  4017e9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4017ed:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  4017f1:	f2 0f 58 15 07 01 00 	addsd  0x107(%rip),%xmm2        # 401900 <_IO_stdin_used+0x40>
  4017f8:	00 
  4017f9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4017fd:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401801:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401805:	f2 0f 58 24 dd 88 02 	addsd  0x650288(,%rbx,8),%xmm4
  40180c:	65 00 
  40180e:	f2 0f 11 24 dd 88 02 	movsd  %xmm4,0x650288(,%rbx,8)
  401815:	65 00 
  401817:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40181b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40181f:	f2 0f 58 1c dd 88 fc 	addsd  0x65fc88(,%rbx,8),%xmm3
  401826:	65 00 
  401828:	f2 0f 11 1c dd 88 fc 	movsd  %xmm3,0x65fc88(,%rbx,8)
  40182f:	65 00 
  401831:	48 83 c4 18          	add    $0x18,%rsp
  401835:	5b                   	pop    %rbx
  401836:	41 5e                	pop    %r14
  401838:	c3                   	retq   
  401839:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401840 <__libc_csu_init>:
  401840:	41 57                	push   %r15
  401842:	41 56                	push   %r14
  401844:	41 89 ff             	mov    %edi,%r15d
  401847:	41 55                	push   %r13
  401849:	41 54                	push   %r12
  40184b:	4c 8d 25 86 05 20 00 	lea    0x200586(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  401852:	55                   	push   %rbp
  401853:	48 8d 2d 8e 05 20 00 	lea    0x20058e(%rip),%rbp        # 601de8 <__init_array_end>
  40185a:	53                   	push   %rbx
  40185b:	49 89 f6             	mov    %rsi,%r14
  40185e:	49 89 d5             	mov    %rdx,%r13
  401861:	4c 29 e5             	sub    %r12,%rbp
  401864:	48 83 ec 08          	sub    $0x8,%rsp
  401868:	48 c1 fd 03          	sar    $0x3,%rbp
  40186c:	e8 27 ee ff ff       	callq  400698 <_init>
  401871:	48 85 ed             	test   %rbp,%rbp
  401874:	74 20                	je     401896 <__libc_csu_init+0x56>
  401876:	31 db                	xor    %ebx,%ebx
  401878:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40187f:	00 
  401880:	4c 89 ea             	mov    %r13,%rdx
  401883:	4c 89 f6             	mov    %r14,%rsi
  401886:	44 89 ff             	mov    %r15d,%edi
  401889:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40188d:	48 83 c3 01          	add    $0x1,%rbx
  401891:	48 39 eb             	cmp    %rbp,%rbx
  401894:	75 ea                	jne    401880 <__libc_csu_init+0x40>
  401896:	48 83 c4 08          	add    $0x8,%rsp
  40189a:	5b                   	pop    %rbx
  40189b:	5d                   	pop    %rbp
  40189c:	41 5c                	pop    %r12
  40189e:	41 5d                	pop    %r13
  4018a0:	41 5e                	pop    %r14
  4018a2:	41 5f                	pop    %r15
  4018a4:	c3                   	retq   
  4018a5:	90                   	nop
  4018a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4018ad:	00 00 00 

00000000004018b0 <__libc_csu_fini>:
  4018b0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004018b4 <_fini>:
  4018b4:	48 83 ec 08          	sub    $0x8,%rsp
  4018b8:	48 83 c4 08          	add    $0x8,%rsp
  4018bc:	c3                   	retq   
