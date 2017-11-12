
bin/clang++-3.8_nbody_ikracpp_field:     file format elf64-x86-64


Disassembly of section .init:

00000000004006c8 <_init>:
  4006c8:	48 83 ec 08          	sub    $0x8,%rsp
  4006cc:	48 8b 05 25 19 20 00 	mov    0x201925(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4006d3:	48 85 c0             	test   %rax,%rax
  4006d6:	74 05                	je     4006dd <_init+0x15>
  4006d8:	e8 b3 00 00 00       	callq  400790 <sqrt@plt+0x10>
  4006dd:	48 83 c4 08          	add    $0x8,%rsp
  4006e1:	c3                   	retq   

Disassembly of section .plt:

00000000004006f0 <printf@plt-0x10>:
  4006f0:	ff 35 12 19 20 00    	pushq  0x201912(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006f6:	ff 25 14 19 20 00    	jmpq   *0x201914(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400700 <printf@plt>:
  400700:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400706:	68 00 00 00 00       	pushq  $0x0
  40070b:	e9 e0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400710 <_ZNSt8ios_base4InitC1Ev@plt>:
  400710:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400716:	68 01 00 00 00       	pushq  $0x1
  40071b:	e9 d0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400720 <__libc_start_main@plt>:
  400720:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400726:	68 02 00 00 00       	pushq  $0x2
  40072b:	e9 c0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400730 <__cxa_atexit@plt>:
  400730:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400736:	68 03 00 00 00       	pushq  $0x3
  40073b:	e9 b0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400740 <_ZNSt8ios_base4InitD1Ev@plt>:
  400740:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400746:	68 04 00 00 00       	pushq  $0x4
  40074b:	e9 a0 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400750 <srand@plt>:
  400750:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400756:	68 05 00 00 00       	pushq  $0x5
  40075b:	e9 90 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400760:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400766:	68 06 00 00 00       	pushq  $0x6
  40076b:	e9 80 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400770 <rand@plt>:
  400770:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400776:	68 07 00 00 00       	pushq  $0x7
  40077b:	e9 70 ff ff ff       	jmpq   4006f0 <_init+0x28>

0000000000400780 <sqrt@plt>:
  400780:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400786:	68 08 00 00 00       	pushq  $0x8
  40078b:	e9 60 ff ff ff       	jmpq   4006f0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400790 <.plt.got>:
  400790:	ff 25 62 18 20 00    	jmpq   *0x201862(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400796:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007a0 <_GLOBAL__sub_I_ikracpp_field.cc>:
  4007a0:	50                   	push   %rax
  4007a1:	bf 78 20 60 00       	mov    $0x602078,%edi
  4007a6:	e8 65 ff ff ff       	callq  400710 <_ZNSt8ios_base4InitC1Ev@plt>
  4007ab:	bf 40 07 40 00       	mov    $0x400740,%edi
  4007b0:	be 78 20 60 00       	mov    $0x602078,%esi
  4007b5:	ba 68 20 60 00       	mov    $0x602068,%edx
  4007ba:	58                   	pop    %rax
  4007bb:	e9 70 ff ff ff       	jmpq   400730 <__cxa_atexit@plt>

00000000004007c0 <_start>:
  4007c0:	31 ed                	xor    %ebp,%ebp
  4007c2:	49 89 d1             	mov    %rdx,%r9
  4007c5:	5e                   	pop    %rsi
  4007c6:	48 89 e2             	mov    %rsp,%rdx
  4007c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4007cd:	50                   	push   %rax
  4007ce:	54                   	push   %rsp
  4007cf:	49 c7 c0 e0 18 40 00 	mov    $0x4018e0,%r8
  4007d6:	48 c7 c1 70 18 40 00 	mov    $0x401870,%rcx
  4007dd:	48 c7 c7 00 11 40 00 	mov    $0x401100,%rdi
  4007e4:	e8 37 ff ff ff       	callq  400720 <__libc_start_main@plt>
  4007e9:	f4                   	hlt    
  4007ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007f0 <deregister_tm_clones>:
  4007f0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4007f5:	55                   	push   %rbp
  4007f6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4007fc:	48 83 f8 0e          	cmp    $0xe,%rax
  400800:	48 89 e5             	mov    %rsp,%rbp
  400803:	76 1b                	jbe    400820 <deregister_tm_clones+0x30>
  400805:	b8 00 00 00 00       	mov    $0x0,%eax
  40080a:	48 85 c0             	test   %rax,%rax
  40080d:	74 11                	je     400820 <deregister_tm_clones+0x30>
  40080f:	5d                   	pop    %rbp
  400810:	bf 70 20 60 00       	mov    $0x602070,%edi
  400815:	ff e0                	jmpq   *%rax
  400817:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40081e:	00 00 
  400820:	5d                   	pop    %rbp
  400821:	c3                   	retq   
  400822:	0f 1f 40 00          	nopl   0x0(%rax)
  400826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40082d:	00 00 00 

0000000000400830 <register_tm_clones>:
  400830:	be 70 20 60 00       	mov    $0x602070,%esi
  400835:	55                   	push   %rbp
  400836:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40083d:	48 c1 fe 03          	sar    $0x3,%rsi
  400841:	48 89 e5             	mov    %rsp,%rbp
  400844:	48 89 f0             	mov    %rsi,%rax
  400847:	48 c1 e8 3f          	shr    $0x3f,%rax
  40084b:	48 01 c6             	add    %rax,%rsi
  40084e:	48 d1 fe             	sar    %rsi
  400851:	74 15                	je     400868 <register_tm_clones+0x38>
  400853:	b8 00 00 00 00       	mov    $0x0,%eax
  400858:	48 85 c0             	test   %rax,%rax
  40085b:	74 0b                	je     400868 <register_tm_clones+0x38>
  40085d:	5d                   	pop    %rbp
  40085e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400863:	ff e0                	jmpq   *%rax
  400865:	0f 1f 00             	nopl   (%rax)
  400868:	5d                   	pop    %rbp
  400869:	c3                   	retq   
  40086a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400870 <__do_global_dtors_aux>:
  400870:	80 3d f9 17 20 00 00 	cmpb   $0x0,0x2017f9(%rip)        # 602070 <__TMC_END__>
  400877:	75 11                	jne    40088a <__do_global_dtors_aux+0x1a>
  400879:	55                   	push   %rbp
  40087a:	48 89 e5             	mov    %rsp,%rbp
  40087d:	e8 6e ff ff ff       	callq  4007f0 <deregister_tm_clones>
  400882:	5d                   	pop    %rbp
  400883:	c6 05 e6 17 20 00 01 	movb   $0x1,0x2017e6(%rip)        # 602070 <__TMC_END__>
  40088a:	f3 c3                	repz retq 
  40088c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400890 <frame_dummy>:
  400890:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400895:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400899:	75 05                	jne    4008a0 <frame_dummy+0x10>
  40089b:	eb 93                	jmp    400830 <register_tm_clones>
  40089d:	0f 1f 00             	nopl   (%rax)
  4008a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4008a5:	48 85 c0             	test   %rax,%rax
  4008a8:	74 f1                	je     40089b <frame_dummy+0xb>
  4008aa:	55                   	push   %rbp
  4008ab:	48 89 e5             	mov    %rsp,%rbp
  4008ae:	ff d0                	callq  *%rax
  4008b0:	5d                   	pop    %rbp
  4008b1:	e9 7a ff ff ff       	jmpq   400830 <register_tm_clones>
  4008b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008bd:	00 00 00 

00000000004008c0 <_Z11r_float2intf>:
  4008c0:	66 0f 7e c0          	movd   %xmm0,%eax
  4008c4:	c3                   	retq   
  4008c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4008cc:	00 00 00 00 

00000000004008d0 <_ZN4Body7storageEv>:
  4008d0:	b8 80 20 60 00       	mov    $0x602080,%eax
  4008d5:	c3                   	retq   
  4008d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008dd:	00 00 00 

00000000004008e0 <_Z13instantiationv>:
  4008e0:	53                   	push   %rbx
  4008e1:	48 83 ec 30          	sub    $0x30,%rsp
  4008e5:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4008ea:	e8 61 fe ff ff       	callq  400750 <srand@plt>
  4008ef:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4008f4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008fb:	00 00 00 00 00 
  400900:	e8 6b fe ff ff       	callq  400770 <rand@plt>
  400905:	0f 57 c9             	xorps  %xmm1,%xmm1
  400908:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  40090c:	f2 0f 10 05 e4 0f 00 	movsd  0xfe4(%rip),%xmm0        # 4018f8 <_IO_stdin_used+0x8>
  400913:	00 
  400914:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400918:	f2 0f 10 05 e0 0f 00 	movsd  0xfe0(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  40091f:	00 
  400920:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400924:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400928:	f2 0f 10 05 d8 0f 00 	movsd  0xfd8(%rip),%xmm0        # 401908 <_IO_stdin_used+0x18>
  40092f:	00 
  400930:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400934:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  40093a:	e8 31 fe ff ff       	callq  400770 <rand@plt>
  40093f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400942:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400946:	f2 0f 5e 0d aa 0f 00 	divsd  0xfaa(%rip),%xmm1        # 4018f8 <_IO_stdin_used+0x8>
  40094d:	00 
  40094e:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400952:	f2 0f 10 05 b6 0f 00 	movsd  0xfb6(%rip),%xmm0        # 401910 <_IO_stdin_used+0x20>
  400959:	00 
  40095a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40095e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400964:	e8 07 fe ff ff       	callq  400770 <rand@plt>
  400969:	0f 57 c0             	xorps  %xmm0,%xmm0
  40096c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400970:	f2 0f 5e 05 80 0f 00 	divsd  0xf80(%rip),%xmm0        # 4018f8 <_IO_stdin_used+0x8>
  400977:	00 
  400978:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40097c:	f2 0f 58 05 8c 0f 00 	addsd  0xf8c(%rip),%xmm0        # 401910 <_IO_stdin_used+0x20>
  400983:	00 
  400984:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40098a:	e8 e1 fd ff ff       	callq  400770 <rand@plt>
  40098f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400992:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400996:	f2 0f 5e 0d 5a 0f 00 	divsd  0xf5a(%rip),%xmm1        # 4018f8 <_IO_stdin_used+0x8>
  40099d:	00 
  40099e:	f2 0f 10 05 72 0f 00 	movsd  0xf72(%rip),%xmm0        # 401918 <_IO_stdin_used+0x28>
  4009a5:	00 
  4009a6:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4009aa:	f2 0f 5e 0d 56 0f 00 	divsd  0xf56(%rip),%xmm1        # 401908 <_IO_stdin_used+0x18>
  4009b1:	00 
  4009b2:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  4009b8:	e8 b3 fd ff ff       	callq  400770 <rand@plt>
  4009bd:	0f 57 c0             	xorps  %xmm0,%xmm0
  4009c0:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  4009c4:	f2 0f 5e 05 2c 0f 00 	divsd  0xf2c(%rip),%xmm0        # 4018f8 <_IO_stdin_used+0x8>
  4009cb:	00 
  4009cc:	f2 0f 58 05 44 0f 00 	addsd  0xf44(%rip),%xmm0        # 401918 <_IO_stdin_used+0x28>
  4009d3:	00 
  4009d4:	f2 0f 5e 05 2c 0f 00 	divsd  0xf2c(%rip),%xmm0        # 401908 <_IO_stdin_used+0x18>
  4009db:	00 
  4009dc:	48 8b 05 9d 16 20 00 	mov    0x20169d(%rip),%rax        # 602080 <__Bodydata_buffer>
  4009e3:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4009e7:	48 89 0d 92 16 20 00 	mov    %rcx,0x201692(%rip)        # 602080 <__Bodydata_buffer>
  4009ee:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  4009f4:	f2 0f 11 0c c5 98 20 	movsd  %xmm1,0x602098(,%rax,8)
  4009fb:	60 00 
  4009fd:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  400a03:	f2 0f 11 0c c5 98 1a 	movsd  %xmm1,0x611a98(,%rax,8)
  400a0a:	61 00 
  400a0c:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  400a12:	f2 0f 11 0c c5 98 14 	movsd  %xmm1,0x621498(,%rax,8)
  400a19:	62 00 
  400a1b:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  400a21:	f2 0f 11 0c c5 98 0e 	movsd  %xmm1,0x630e98(,%rax,8)
  400a28:	63 00 
  400a2a:	f2 0f 11 04 c5 98 08 	movsd  %xmm0,0x640898(,%rax,8)
  400a31:	64 00 
  400a33:	48 c7 04 c5 98 02 65 	movq   $0x0,0x650298(,%rax,8)
  400a3a:	00 00 00 00 00 
  400a3f:	48 c7 04 c5 98 fc 65 	movq   $0x0,0x65fc98(,%rax,8)
  400a46:	00 00 00 00 00 
  400a4b:	ff cb                	dec    %ebx
  400a4d:	0f 85 ad fe ff ff    	jne    400900 <_Z13instantiationv+0x20>
  400a53:	48 83 c4 30          	add    $0x30,%rsp
  400a57:	5b                   	pop    %rbx
  400a58:	c3                   	retq   
  400a59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a60 <_Z14run_simulationv>:
  400a60:	41 57                	push   %r15
  400a62:	41 56                	push   %r14
  400a64:	53                   	push   %rbx
  400a65:	b8 08 00 00 00       	mov    $0x8,%eax
  400a6a:	45 31 ff             	xor    %r15d,%r15d
  400a6d:	0f 1f 00             	nopl   (%rax)
  400a70:	48 c7 04 c5 58 02 65 	movq   $0x0,0x650258(,%rax,8)
  400a77:	00 00 00 00 00 
  400a7c:	48 c7 04 c5 58 fc 65 	movq   $0x0,0x65fc58(,%rax,8)
  400a83:	00 00 00 00 00 
  400a88:	48 c7 04 c5 60 02 65 	movq   $0x0,0x650260(,%rax,8)
  400a8f:	00 00 00 00 00 
  400a94:	48 c7 04 c5 60 fc 65 	movq   $0x0,0x65fc60(,%rax,8)
  400a9b:	00 00 00 00 00 
  400aa0:	48 c7 04 c5 68 02 65 	movq   $0x0,0x650268(,%rax,8)
  400aa7:	00 00 00 00 00 
  400aac:	48 c7 04 c5 68 fc 65 	movq   $0x0,0x65fc68(,%rax,8)
  400ab3:	00 00 00 00 00 
  400ab8:	48 c7 04 c5 70 02 65 	movq   $0x0,0x650270(,%rax,8)
  400abf:	00 00 00 00 00 
  400ac4:	48 c7 04 c5 70 fc 65 	movq   $0x0,0x65fc70(,%rax,8)
  400acb:	00 00 00 00 00 
  400ad0:	48 c7 04 c5 78 02 65 	movq   $0x0,0x650278(,%rax,8)
  400ad7:	00 00 00 00 00 
  400adc:	48 c7 04 c5 78 fc 65 	movq   $0x0,0x65fc78(,%rax,8)
  400ae3:	00 00 00 00 00 
  400ae8:	48 c7 04 c5 80 02 65 	movq   $0x0,0x650280(,%rax,8)
  400aef:	00 00 00 00 00 
  400af4:	48 c7 04 c5 80 fc 65 	movq   $0x0,0x65fc80(,%rax,8)
  400afb:	00 00 00 00 00 
  400b00:	48 c7 04 c5 88 02 65 	movq   $0x0,0x650288(,%rax,8)
  400b07:	00 00 00 00 00 
  400b0c:	48 c7 04 c5 88 fc 65 	movq   $0x0,0x65fc88(,%rax,8)
  400b13:	00 00 00 00 00 
  400b18:	48 c7 04 c5 90 02 65 	movq   $0x0,0x650290(,%rax,8)
  400b1f:	00 00 00 00 00 
  400b24:	48 c7 04 c5 90 fc 65 	movq   $0x0,0x65fc90(,%rax,8)
  400b2b:	00 00 00 00 00 
  400b30:	48 83 c0 08          	add    $0x8,%rax
  400b34:	48 3d 48 1f 00 00    	cmp    $0x1f48,%rax
  400b3a:	0f 85 30 ff ff ff    	jne    400a70 <_Z14run_simulationv+0x10>
  400b40:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 
  400b50:	49 ff c7             	inc    %r15
  400b53:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b5f:	00 
  400b60:	48 89 df             	mov    %rbx,%rdi
  400b63:	4c 89 fe             	mov    %r15,%rsi
  400b66:	e8 15 0c 00 00       	callq  401780 <_ZN4Body9add_forceEPS_>
  400b6b:	48 ff c3             	inc    %rbx
  400b6e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400b75:	75 e9                	jne    400b60 <_Z14run_simulationv+0x100>
  400b77:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400b7e:	75 d0                	jne    400b50 <_Z14run_simulationv+0xf0>
  400b80:	bb 08 00 00 00       	mov    $0x8,%ebx
  400b85:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400b8c:	00 00 00 00 
  400b90:	4c 89 f7             	mov    %r14,%rdi
  400b93:	f2 0f 10 05 65 0d 00 	movsd  0xd65(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  400b9a:	00 
  400b9b:	e8 50 0a 00 00       	callq  4015f0 <_ZN4Body6updateEd>
  400ba0:	49 ff c6             	inc    %r14
  400ba3:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400baa:	75 e4                	jne    400b90 <_Z14run_simulationv+0x130>
  400bac:	45 31 ff             	xor    %r15d,%r15d
  400baf:	90                   	nop
  400bb0:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400bb7:	00 00 00 00 00 
  400bbc:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400bc3:	00 00 00 00 00 
  400bc8:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400bcf:	00 00 00 00 00 
  400bd4:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400bdb:	00 00 00 00 00 
  400be0:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400be7:	00 00 00 00 00 
  400bec:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400bf3:	00 00 00 00 00 
  400bf8:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400bff:	00 00 00 00 00 
  400c04:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400c0b:	00 00 00 00 00 
  400c10:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400c17:	00 00 00 00 00 
  400c1c:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400c23:	00 00 00 00 00 
  400c28:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400c2f:	00 00 00 00 00 
  400c34:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400c3b:	00 00 00 00 00 
  400c40:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400c47:	00 00 00 00 00 
  400c4c:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400c53:	00 00 00 00 00 
  400c58:	48 c7 04 dd 90 02 65 	movq   $0x0,0x650290(,%rbx,8)
  400c5f:	00 00 00 00 00 
  400c64:	48 c7 04 dd 90 fc 65 	movq   $0x0,0x65fc90(,%rbx,8)
  400c6b:	00 00 00 00 00 
  400c70:	48 83 c3 08          	add    $0x8,%rbx
  400c74:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400c7b:	0f 85 2f ff ff ff    	jne    400bb0 <_Z14run_simulationv+0x150>
  400c81:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400c87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400c8e:	00 00 
  400c90:	49 ff c7             	inc    %r15
  400c93:	bb 01 00 00 00       	mov    $0x1,%ebx
  400c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c9f:	00 
  400ca0:	48 89 df             	mov    %rbx,%rdi
  400ca3:	4c 89 fe             	mov    %r15,%rsi
  400ca6:	e8 d5 0a 00 00       	callq  401780 <_ZN4Body9add_forceEPS_>
  400cab:	48 ff c3             	inc    %rbx
  400cae:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400cb5:	75 e9                	jne    400ca0 <_Z14run_simulationv+0x240>
  400cb7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400cbe:	75 d0                	jne    400c90 <_Z14run_simulationv+0x230>
  400cc0:	bb 08 00 00 00       	mov    $0x8,%ebx
  400cc5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400ccc:	00 00 00 00 
  400cd0:	4c 89 f7             	mov    %r14,%rdi
  400cd3:	f2 0f 10 05 25 0c 00 	movsd  0xc25(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  400cda:	00 
  400cdb:	e8 10 09 00 00       	callq  4015f0 <_ZN4Body6updateEd>
  400ce0:	49 ff c6             	inc    %r14
  400ce3:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400cea:	75 e4                	jne    400cd0 <_Z14run_simulationv+0x270>
  400cec:	45 31 ff             	xor    %r15d,%r15d
  400cef:	90                   	nop
  400cf0:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400cf7:	00 00 00 00 00 
  400cfc:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400d03:	00 00 00 00 00 
  400d08:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400d0f:	00 00 00 00 00 
  400d14:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400d1b:	00 00 00 00 00 
  400d20:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400d27:	00 00 00 00 00 
  400d2c:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400d33:	00 00 00 00 00 
  400d38:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400d3f:	00 00 00 00 00 
  400d44:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400d4b:	00 00 00 00 00 
  400d50:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400d57:	00 00 00 00 00 
  400d5c:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400d63:	00 00 00 00 00 
  400d68:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400d6f:	00 00 00 00 00 
  400d74:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400d7b:	00 00 00 00 00 
  400d80:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400d87:	00 00 00 00 00 
  400d8c:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400d93:	00 00 00 00 00 
  400d98:	48 c7 04 dd 90 02 65 	movq   $0x0,0x650290(,%rbx,8)
  400d9f:	00 00 00 00 00 
  400da4:	48 c7 04 dd 90 fc 65 	movq   $0x0,0x65fc90(,%rbx,8)
  400dab:	00 00 00 00 00 
  400db0:	48 83 c3 08          	add    $0x8,%rbx
  400db4:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400dbb:	0f 85 2f ff ff ff    	jne    400cf0 <_Z14run_simulationv+0x290>
  400dc1:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400dc7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400dce:	00 00 
  400dd0:	49 ff c7             	inc    %r15
  400dd3:	bb 01 00 00 00       	mov    $0x1,%ebx
  400dd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ddf:	00 
  400de0:	48 89 df             	mov    %rbx,%rdi
  400de3:	4c 89 fe             	mov    %r15,%rsi
  400de6:	e8 95 09 00 00       	callq  401780 <_ZN4Body9add_forceEPS_>
  400deb:	48 ff c3             	inc    %rbx
  400dee:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400df5:	75 e9                	jne    400de0 <_Z14run_simulationv+0x380>
  400df7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dfe:	75 d0                	jne    400dd0 <_Z14run_simulationv+0x370>
  400e00:	bb 08 00 00 00       	mov    $0x8,%ebx
  400e05:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400e0c:	00 00 00 00 
  400e10:	4c 89 f7             	mov    %r14,%rdi
  400e13:	f2 0f 10 05 e5 0a 00 	movsd  0xae5(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  400e1a:	00 
  400e1b:	e8 d0 07 00 00       	callq  4015f0 <_ZN4Body6updateEd>
  400e20:	49 ff c6             	inc    %r14
  400e23:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400e2a:	75 e4                	jne    400e10 <_Z14run_simulationv+0x3b0>
  400e2c:	45 31 ff             	xor    %r15d,%r15d
  400e2f:	90                   	nop
  400e30:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400e37:	00 00 00 00 00 
  400e3c:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400e43:	00 00 00 00 00 
  400e48:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400e4f:	00 00 00 00 00 
  400e54:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400e5b:	00 00 00 00 00 
  400e60:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400e67:	00 00 00 00 00 
  400e6c:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400e73:	00 00 00 00 00 
  400e78:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400e7f:	00 00 00 00 00 
  400e84:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400e8b:	00 00 00 00 00 
  400e90:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400e97:	00 00 00 00 00 
  400e9c:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400ea3:	00 00 00 00 00 
  400ea8:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400eaf:	00 00 00 00 00 
  400eb4:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400ebb:	00 00 00 00 00 
  400ec0:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  400ec7:	00 00 00 00 00 
  400ecc:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  400ed3:	00 00 00 00 00 
  400ed8:	48 c7 04 dd 90 02 65 	movq   $0x0,0x650290(,%rbx,8)
  400edf:	00 00 00 00 00 
  400ee4:	48 c7 04 dd 90 fc 65 	movq   $0x0,0x65fc90(,%rbx,8)
  400eeb:	00 00 00 00 00 
  400ef0:	48 83 c3 08          	add    $0x8,%rbx
  400ef4:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  400efb:	0f 85 2f ff ff ff    	jne    400e30 <_Z14run_simulationv+0x3d0>
  400f01:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400f07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f0e:	00 00 
  400f10:	49 ff c7             	inc    %r15
  400f13:	bb 01 00 00 00       	mov    $0x1,%ebx
  400f18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f1f:	00 
  400f20:	48 89 df             	mov    %rbx,%rdi
  400f23:	4c 89 fe             	mov    %r15,%rsi
  400f26:	e8 55 08 00 00       	callq  401780 <_ZN4Body9add_forceEPS_>
  400f2b:	48 ff c3             	inc    %rbx
  400f2e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400f35:	75 e9                	jne    400f20 <_Z14run_simulationv+0x4c0>
  400f37:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400f3e:	75 d0                	jne    400f10 <_Z14run_simulationv+0x4b0>
  400f40:	bb 08 00 00 00       	mov    $0x8,%ebx
  400f45:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400f4c:	00 00 00 00 
  400f50:	4c 89 f7             	mov    %r14,%rdi
  400f53:	f2 0f 10 05 a5 09 00 	movsd  0x9a5(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  400f5a:	00 
  400f5b:	e8 90 06 00 00       	callq  4015f0 <_ZN4Body6updateEd>
  400f60:	49 ff c6             	inc    %r14
  400f63:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400f6a:	75 e4                	jne    400f50 <_Z14run_simulationv+0x4f0>
  400f6c:	45 31 ff             	xor    %r15d,%r15d
  400f6f:	90                   	nop
  400f70:	48 c7 04 dd 58 02 65 	movq   $0x0,0x650258(,%rbx,8)
  400f77:	00 00 00 00 00 
  400f7c:	48 c7 04 dd 58 fc 65 	movq   $0x0,0x65fc58(,%rbx,8)
  400f83:	00 00 00 00 00 
  400f88:	48 c7 04 dd 60 02 65 	movq   $0x0,0x650260(,%rbx,8)
  400f8f:	00 00 00 00 00 
  400f94:	48 c7 04 dd 60 fc 65 	movq   $0x0,0x65fc60(,%rbx,8)
  400f9b:	00 00 00 00 00 
  400fa0:	48 c7 04 dd 68 02 65 	movq   $0x0,0x650268(,%rbx,8)
  400fa7:	00 00 00 00 00 
  400fac:	48 c7 04 dd 68 fc 65 	movq   $0x0,0x65fc68(,%rbx,8)
  400fb3:	00 00 00 00 00 
  400fb8:	48 c7 04 dd 70 02 65 	movq   $0x0,0x650270(,%rbx,8)
  400fbf:	00 00 00 00 00 
  400fc4:	48 c7 04 dd 70 fc 65 	movq   $0x0,0x65fc70(,%rbx,8)
  400fcb:	00 00 00 00 00 
  400fd0:	48 c7 04 dd 78 02 65 	movq   $0x0,0x650278(,%rbx,8)
  400fd7:	00 00 00 00 00 
  400fdc:	48 c7 04 dd 78 fc 65 	movq   $0x0,0x65fc78(,%rbx,8)
  400fe3:	00 00 00 00 00 
  400fe8:	48 c7 04 dd 80 02 65 	movq   $0x0,0x650280(,%rbx,8)
  400fef:	00 00 00 00 00 
  400ff4:	48 c7 04 dd 80 fc 65 	movq   $0x0,0x65fc80(,%rbx,8)
  400ffb:	00 00 00 00 00 
  401000:	48 c7 04 dd 88 02 65 	movq   $0x0,0x650288(,%rbx,8)
  401007:	00 00 00 00 00 
  40100c:	48 c7 04 dd 88 fc 65 	movq   $0x0,0x65fc88(,%rbx,8)
  401013:	00 00 00 00 00 
  401018:	48 c7 04 dd 90 02 65 	movq   $0x0,0x650290(,%rbx,8)
  40101f:	00 00 00 00 00 
  401024:	48 c7 04 dd 90 fc 65 	movq   $0x0,0x65fc90(,%rbx,8)
  40102b:	00 00 00 00 00 
  401030:	48 83 c3 08          	add    $0x8,%rbx
  401034:	48 81 fb 48 1f 00 00 	cmp    $0x1f48,%rbx
  40103b:	0f 85 2f ff ff ff    	jne    400f70 <_Z14run_simulationv+0x510>
  401041:	41 be 01 00 00 00    	mov    $0x1,%r14d
  401047:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40104e:	00 00 
  401050:	49 ff c7             	inc    %r15
  401053:	bb 01 00 00 00       	mov    $0x1,%ebx
  401058:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40105f:	00 
  401060:	48 89 df             	mov    %rbx,%rdi
  401063:	4c 89 fe             	mov    %r15,%rsi
  401066:	e8 15 07 00 00       	callq  401780 <_ZN4Body9add_forceEPS_>
  40106b:	48 ff c3             	inc    %rbx
  40106e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  401075:	75 e9                	jne    401060 <_Z14run_simulationv+0x600>
  401077:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  40107e:	75 d0                	jne    401050 <_Z14run_simulationv+0x5f0>
  401080:	4c 89 f7             	mov    %r14,%rdi
  401083:	f2 0f 10 05 75 08 00 	movsd  0x875(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  40108a:	00 
  40108b:	e8 60 05 00 00       	callq  4015f0 <_ZN4Body6updateEd>
  401090:	49 ff c6             	inc    %r14
  401093:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  40109a:	75 e4                	jne    401080 <_Z14run_simulationv+0x620>
  40109c:	5b                   	pop    %rbx
  40109d:	41 5e                	pop    %r14
  40109f:	41 5f                	pop    %r15
  4010a1:	c3                   	retq   
  4010a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4010a9:	1f 84 00 00 00 00 00 

00000000004010b0 <_Z10run_simplev>:
  4010b0:	55                   	push   %rbp
  4010b1:	53                   	push   %rbx
  4010b2:	50                   	push   %rax
  4010b3:	31 ed                	xor    %ebp,%ebp
  4010b5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4010bc:	00 00 00 00 
  4010c0:	bb 01 00 00 00       	mov    $0x1,%ebx
  4010c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4010cc:	00 00 00 00 
  4010d0:	48 89 df             	mov    %rbx,%rdi
  4010d3:	f2 0f 10 05 25 08 00 	movsd  0x825(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  4010da:	00 
  4010db:	e8 10 06 00 00       	callq  4016f0 <_ZN4Body22codengen_simple_updateEd>
  4010e0:	48 ff c3             	inc    %rbx
  4010e3:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4010ea:	75 e4                	jne    4010d0 <_Z10run_simplev+0x20>
  4010ec:	ff c5                	inc    %ebp
  4010ee:	81 fd 50 c3 00 00    	cmp    $0xc350,%ebp
  4010f4:	75 ca                	jne    4010c0 <_Z10run_simplev+0x10>
  4010f6:	48 83 c4 08          	add    $0x8,%rsp
  4010fa:	5b                   	pop    %rbx
  4010fb:	5d                   	pop    %rbp
  4010fc:	c3                   	retq   
  4010fd:	0f 1f 00             	nopl   (%rax)

0000000000401100 <main>:
  401100:	55                   	push   %rbp
  401101:	41 57                	push   %r15
  401103:	41 56                	push   %r14
  401105:	41 55                	push   %r13
  401107:	41 54                	push   %r12
  401109:	53                   	push   %rbx
  40110a:	50                   	push   %rax
  40110b:	0f 57 c0             	xorps  %xmm0,%xmm0
  40110e:	0f 11 05 6b 0f 20 00 	movups %xmm0,0x200f6b(%rip)        # 602080 <__Bodydata_buffer>
  401115:	e8 46 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40111a:	48 89 c5             	mov    %rax,%rbp
  40111d:	e8 be f7 ff ff       	callq  4008e0 <_Z13instantiationv>
  401122:	e8 39 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401127:	49 89 c6             	mov    %rax,%r14
  40112a:	49 29 ee             	sub    %rbp,%r14
  40112d:	e8 2e f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401132:	48 89 c3             	mov    %rax,%rbx
  401135:	e8 26 f9 ff ff       	callq  400a60 <_Z14run_simulationv>
  40113a:	e8 21 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40113f:	48 89 c5             	mov    %rax,%rbp
  401142:	48 29 dd             	sub    %rbx,%rbp
  401145:	31 db                	xor    %ebx,%ebx
  401147:	e8 14 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40114c:	49 89 c7             	mov    %rax,%r15
  40114f:	f2 0f 10 05 a9 07 00 	movsd  0x7a9(%rip),%xmm0        # 401900 <_IO_stdin_used+0x10>
  401156:	00 
  401157:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40115e:	00 00 
  401160:	48 c7 c0 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rax
  401167:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40116e:	00 00 
  401170:	f2 0f 10 0c c5 98 fc 	movsd  0x65fc98(,%rax,8),%xmm1
  401177:	65 00 
  401179:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40117d:	f2 0f 5e 0c c5 98 1a 	divsd  0x611a98(,%rax,8),%xmm1
  401184:	61 00 
  401186:	f2 0f 58 0c c5 98 08 	addsd  0x640898(,%rax,8),%xmm1
  40118d:	64 00 
  40118f:	f2 0f 11 0c c5 98 08 	movsd  %xmm1,0x640898(,%rax,8)
  401196:	64 00 
  401198:	f2 0f 10 0c c5 98 f6 	movsd  0x66f698(,%rax,8),%xmm1
  40119f:	66 00 
  4011a1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011a5:	f2 0f 5e 0c c5 98 1a 	divsd  0x611a98(,%rax,8),%xmm1
  4011ac:	61 00 
  4011ae:	f2 0f 58 0c c5 98 02 	addsd  0x650298(,%rax,8),%xmm1
  4011b5:	65 00 
  4011b7:	f2 0f 11 0c c5 98 02 	movsd  %xmm1,0x650298(,%rax,8)
  4011be:	65 00 
  4011c0:	f2 0f 10 0c c5 98 08 	movsd  0x640898(,%rax,8),%xmm1
  4011c7:	64 00 
  4011c9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011cd:	f2 0f 58 0c c5 98 14 	addsd  0x621498(,%rax,8),%xmm1
  4011d4:	62 00 
  4011d6:	f2 0f 11 0c c5 98 14 	movsd  %xmm1,0x621498(,%rax,8)
  4011dd:	62 00 
  4011df:	f2 0f 10 0c c5 98 02 	movsd  0x650298(,%rax,8),%xmm1
  4011e6:	65 00 
  4011e8:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011ec:	f2 0f 58 0c c5 98 0e 	addsd  0x630e98(,%rax,8),%xmm1
  4011f3:	63 00 
  4011f5:	f2 0f 11 0c c5 98 0e 	movsd  %xmm1,0x630e98(,%rax,8)
  4011fc:	63 00 
  4011fe:	48 ff c0             	inc    %rax
  401201:	0f 85 69 ff ff ff    	jne    401170 <main+0x70>
  401207:	ff c3                	inc    %ebx
  401209:	81 fb 50 c3 00 00    	cmp    $0xc350,%ebx
  40120f:	0f 85 4b ff ff ff    	jne    401160 <main+0x60>
  401215:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  40121c:	de 1b 43 
  40121f:	4c 89 f0             	mov    %r14,%rax
  401222:	48 f7 e9             	imul   %rcx
  401225:	49 89 d6             	mov    %rdx,%r14
  401228:	4c 89 f0             	mov    %r14,%rax
  40122b:	48 c1 e8 3f          	shr    $0x3f,%rax
  40122f:	49 c1 fe 12          	sar    $0x12,%r14
  401233:	49 01 c6             	add    %rax,%r14
  401236:	48 89 e8             	mov    %rbp,%rax
  401239:	48 f7 e9             	imul   %rcx
  40123c:	49 89 d5             	mov    %rdx,%r13
  40123f:	4c 89 e8             	mov    %r13,%rax
  401242:	48 c1 e8 3f          	shr    $0x3f,%rax
  401246:	49 c1 fd 12          	sar    $0x12,%r13
  40124a:	49 01 c5             	add    %rax,%r13
  40124d:	e8 0e f5 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401252:	49 89 c4             	mov    %rax,%r12
  401255:	4d 29 fc             	sub    %r15,%r12
  401258:	bd 0b 00 00 00       	mov    $0xb,%ebp
  40125d:	31 f6                	xor    %esi,%esi
  40125f:	90                   	nop
  401260:	4c 8d 7e 01          	lea    0x1(%rsi),%r15
  401264:	f2 0f 10 04 f5 98 1a 	movsd  0x611a98(,%rsi,8),%xmm0
  40126b:	61 00 
  40126d:	f2 0f 5a c8          	cvtsd2ss %xmm0,%xmm1
  401271:	66 0f 7e c8          	movd   %xmm1,%eax
  401275:	01 e8                	add    %ebp,%eax
  401277:	f2 0f 10 0c f5 98 14 	movsd  0x621498(,%rsi,8),%xmm1
  40127e:	62 00 
  401280:	f2 0f 5a d1          	cvtsd2ss %xmm1,%xmm2
  401284:	66 0f 7e d5          	movd   %xmm2,%ebp
  401288:	01 c5                	add    %eax,%ebp
  40128a:	48 63 c5             	movslq %ebp,%rax
  40128d:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  401294:	48 89 c1             	mov    %rax,%rcx
  401297:	48 c1 e9 3f          	shr    $0x3f,%rcx
  40129b:	48 c1 f8 33          	sar    $0x33,%rax
  40129f:	01 c8                	add    %ecx,%eax
  4012a1:	69 d8 87 d6 12 00    	imul   $0x12d687,%eax,%ebx
  4012a7:	48 83 fe 0a          	cmp    $0xa,%rsi
  4012ab:	73 0c                	jae    4012b9 <main+0x1b9>
  4012ad:	bf 50 19 40 00       	mov    $0x401950,%edi
  4012b2:	b0 02                	mov    $0x2,%al
  4012b4:	e8 47 f4 ff ff       	callq  400700 <printf@plt>
  4012b9:	29 dd                	sub    %ebx,%ebp
  4012bb:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4012c2:	4c 89 fe             	mov    %r15,%rsi
  4012c5:	75 99                	jne    401260 <main+0x160>
  4012c7:	4c 89 e0             	mov    %r12,%rax
  4012ca:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  4012d1:	de 1b 43 
  4012d4:	48 f7 e9             	imul   %rcx
  4012d7:	48 89 d0             	mov    %rdx,%rax
  4012da:	48 c1 e8 3f          	shr    $0x3f,%rax
  4012de:	48 c1 fa 12          	sar    $0x12,%rdx
  4012e2:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  4012e6:	bf 65 19 40 00       	mov    $0x401965,%edi
  4012eb:	31 c0                	xor    %eax,%eax
  4012ed:	4c 89 f6             	mov    %r14,%rsi
  4012f0:	4c 89 ea             	mov    %r13,%rdx
  4012f3:	41 89 e8             	mov    %ebp,%r8d
  4012f6:	e8 05 f4 ff ff       	callq  400700 <printf@plt>
  4012fb:	31 c0                	xor    %eax,%eax
  4012fd:	48 83 c4 08          	add    $0x8,%rsp
  401301:	5b                   	pop    %rbx
  401302:	41 5c                	pop    %r12
  401304:	41 5d                	pop    %r13
  401306:	41 5e                	pop    %r14
  401308:	41 5f                	pop    %r15
  40130a:	5d                   	pop    %rbp
  40130b:	c3                   	retq   
  40130c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401310 <_Z19codegen_reset_forceP4Body>:
  401310:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401317:	00 00 00 00 00 
  40131c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  401323:	00 00 00 00 00 
  401328:	c3                   	retq   
  401329:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401330 <_Z17codegen_add_forceP4BodyS0_>:
  401330:	41 56                	push   %r14
  401332:	53                   	push   %rbx
  401333:	48 83 ec 18          	sub    $0x18,%rsp
  401337:	49 89 f6             	mov    %rsi,%r14
  40133a:	48 89 fb             	mov    %rdi,%rbx
  40133d:	4c 39 f3             	cmp    %r14,%rbx
  401340:	0f 84 cb 00 00 00    	je     401411 <_Z17codegen_add_forceP4BodyS0_+0xe1>
  401346:	f2 42 0f 10 24 f5 90 	movsd  0x611a90(,%r14,8),%xmm4
  40134d:	1a 61 00 
  401350:	f2 0f 5c 24 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm4
  401357:	61 00 
  401359:	f2 42 0f 10 1c f5 90 	movsd  0x621490(,%r14,8),%xmm3
  401360:	14 62 00 
  401363:	f2 0f 5c 1c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm3
  40136a:	62 00 
  40136c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401370:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  401374:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401378:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40137c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  401380:	0f 57 c0             	xorps  %xmm0,%xmm0
  401383:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  401387:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  40138b:	7b 21                	jnp    4013ae <_Z17codegen_add_forceP4BodyS0_+0x7e>
  40138d:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401391:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  401397:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40139d:	e8 de f3 ff ff       	callq  400780 <sqrt@plt>
  4013a2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4013a8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4013ae:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  4013b5:	60 00 
  4013b7:	f2 0f 59 0d 69 05 00 	mulsd  0x569(%rip),%xmm1        # 401928 <_IO_stdin_used+0x38>
  4013be:	00 
  4013bf:	f2 42 0f 59 0c f5 90 	mulsd  0x602090(,%r14,8),%xmm1
  4013c6:	20 60 00 
  4013c9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4013cd:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  4013d1:	f2 0f 58 15 57 05 00 	addsd  0x557(%rip),%xmm2        # 401930 <_IO_stdin_used+0x40>
  4013d8:	00 
  4013d9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4013dd:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  4013e1:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  4013e5:	f2 0f 58 24 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm4
  4013ec:	65 00 
  4013ee:	f2 0f 11 24 dd 90 02 	movsd  %xmm4,0x650290(,%rbx,8)
  4013f5:	65 00 
  4013f7:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4013fb:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  4013ff:	f2 0f 58 1c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm3
  401406:	65 00 
  401408:	f2 0f 11 1c dd 90 fc 	movsd  %xmm3,0x65fc90(,%rbx,8)
  40140f:	65 00 
  401411:	48 83 c4 18          	add    $0x18,%rsp
  401415:	5b                   	pop    %rbx
  401416:	41 5e                	pop    %r14
  401418:	c3                   	retq   
  401419:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401420 <_Z14codegen_updateP4Bodyd>:
  401420:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401427:	65 00 
  401429:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40142d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401434:	60 00 
  401436:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40143d:	63 00 
  40143f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401446:	63 00 
  401448:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  40144f:	65 00 
  401451:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401455:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  40145c:	60 00 
  40145e:	f2 0f 58 0c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm1
  401465:	64 00 
  401467:	f2 0f 11 0c fd 90 08 	movsd  %xmm1,0x640890(,%rdi,8)
  40146e:	64 00 
  401470:	f2 0f 10 0c fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm1
  401477:	63 00 
  401479:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40147d:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  401484:	61 00 
  401486:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  40148d:	61 00 
  40148f:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  401496:	64 00 
  401498:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  40149f:	62 00 
  4014a1:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  4014a8:	62 00 
  4014aa:	f2 0f 10 14 fd 90 1a 	movsd  0x611a90(,%rdi,8),%xmm2
  4014b1:	61 00 
  4014b3:	f2 0f 10 0d 55 04 00 	movsd  0x455(%rip),%xmm1        # 401910 <_IO_stdin_used+0x20>
  4014ba:	00 
  4014bb:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  4014bf:	77 0a                	ja     4014cb <_Z14codegen_updateP4Bodyd+0xab>
  4014c1:	66 0f 2e 15 57 04 00 	ucomisd 0x457(%rip),%xmm2        # 401920 <_IO_stdin_used+0x30>
  4014c8:	00 
  4014c9:	76 23                	jbe    4014ee <_Z14codegen_updateP4Bodyd+0xce>
  4014cb:	f2 0f 10 04 fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm0
  4014d2:	63 00 
  4014d4:	66 0f 57 05 64 04 00 	xorpd  0x464(%rip),%xmm0        # 401940 <_IO_stdin_used+0x50>
  4014db:	00 
  4014dc:	66 0f 13 04 fd 90 0e 	movlpd %xmm0,0x630e90(,%rdi,8)
  4014e3:	63 00 
  4014e5:	f2 0f 10 04 fd 90 14 	movsd  0x621490(,%rdi,8),%xmm0
  4014ec:	62 00 
  4014ee:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  4014f2:	77 0a                	ja     4014fe <_Z14codegen_updateP4Bodyd+0xde>
  4014f4:	66 0f 2e 05 24 04 00 	ucomisd 0x424(%rip),%xmm0        # 401920 <_IO_stdin_used+0x30>
  4014fb:	00 
  4014fc:	76 1a                	jbe    401518 <_Z14codegen_updateP4Bodyd+0xf8>
  4014fe:	f2 0f 10 04 fd 90 08 	movsd  0x640890(,%rdi,8),%xmm0
  401505:	64 00 
  401507:	66 0f 57 05 31 04 00 	xorpd  0x431(%rip),%xmm0        # 401940 <_IO_stdin_used+0x50>
  40150e:	00 
  40150f:	66 0f 13 04 fd 90 08 	movlpd %xmm0,0x640890(,%rdi,8)
  401516:	64 00 
  401518:	c3                   	retq   
  401519:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401520 <_Z26codegen_reset_force_manualP4Body>:
  401520:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401527:	00 00 00 00 00 
  40152c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  401533:	00 00 00 00 00 
  401538:	c3                   	retq   
  401539:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401540 <_Z24codegen_reset_force_halfP4Body>:
  401540:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401547:	00 00 00 00 00 
  40154c:	c3                   	retq   
  40154d:	0f 1f 00             	nopl   (%rax)

0000000000401550 <_Z18codegen_reset_massP4Body>:
  401550:	48 c7 04 fd 90 20 60 	movq   $0x0,0x602090(,%rdi,8)
  401557:	00 00 00 00 00 
  40155c:	c3                   	retq   
  40155d:	0f 1f 00             	nopl   (%rax)

0000000000401560 <_Z22codengen_simple_updateP4Bodyd>:
  401560:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401567:	65 00 
  401569:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40156d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401574:	60 00 
  401576:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40157d:	63 00 
  40157f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401586:	63 00 
  401588:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  40158f:	65 00 
  401591:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401595:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  40159c:	60 00 
  40159e:	f2 0f 58 0c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm1
  4015a5:	64 00 
  4015a7:	f2 0f 11 0c fd 90 08 	movsd  %xmm1,0x640890(,%rdi,8)
  4015ae:	64 00 
  4015b0:	f2 0f 10 0c fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm1
  4015b7:	63 00 
  4015b9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015bd:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  4015c4:	61 00 
  4015c6:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  4015cd:	61 00 
  4015cf:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  4015d6:	64 00 
  4015d8:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  4015df:	62 00 
  4015e1:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  4015e8:	62 00 
  4015ea:	c3                   	retq   
  4015eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004015f0 <_ZN4Body6updateEd>:
  4015f0:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  4015f7:	65 00 
  4015f9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015fd:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401604:	60 00 
  401606:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40160d:	63 00 
  40160f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401616:	63 00 
  401618:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  40161f:	65 00 
  401621:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401625:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  40162c:	60 00 
  40162e:	f2 0f 58 0c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm1
  401635:	64 00 
  401637:	f2 0f 11 0c fd 90 08 	movsd  %xmm1,0x640890(,%rdi,8)
  40163e:	64 00 
  401640:	f2 0f 10 0c fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm1
  401647:	63 00 
  401649:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40164d:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  401654:	61 00 
  401656:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  40165d:	61 00 
  40165f:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  401666:	64 00 
  401668:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  40166f:	62 00 
  401671:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  401678:	62 00 
  40167a:	f2 0f 10 14 fd 90 1a 	movsd  0x611a90(,%rdi,8),%xmm2
  401681:	61 00 
  401683:	f2 0f 10 0d 85 02 00 	movsd  0x285(%rip),%xmm1        # 401910 <_IO_stdin_used+0x20>
  40168a:	00 
  40168b:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  40168f:	77 0a                	ja     40169b <_ZN4Body6updateEd+0xab>
  401691:	66 0f 2e 15 87 02 00 	ucomisd 0x287(%rip),%xmm2        # 401920 <_IO_stdin_used+0x30>
  401698:	00 
  401699:	76 23                	jbe    4016be <_ZN4Body6updateEd+0xce>
  40169b:	f2 0f 10 04 fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm0
  4016a2:	63 00 
  4016a4:	66 0f 57 05 94 02 00 	xorpd  0x294(%rip),%xmm0        # 401940 <_IO_stdin_used+0x50>
  4016ab:	00 
  4016ac:	66 0f 13 04 fd 90 0e 	movlpd %xmm0,0x630e90(,%rdi,8)
  4016b3:	63 00 
  4016b5:	f2 0f 10 04 fd 90 14 	movsd  0x621490(,%rdi,8),%xmm0
  4016bc:	62 00 
  4016be:	66 0f 2e c8          	ucomisd %xmm0,%xmm1
  4016c2:	77 0a                	ja     4016ce <_ZN4Body6updateEd+0xde>
  4016c4:	66 0f 2e 05 54 02 00 	ucomisd 0x254(%rip),%xmm0        # 401920 <_IO_stdin_used+0x30>
  4016cb:	00 
  4016cc:	76 1a                	jbe    4016e8 <_ZN4Body6updateEd+0xf8>
  4016ce:	f2 0f 10 04 fd 90 08 	movsd  0x640890(,%rdi,8),%xmm0
  4016d5:	64 00 
  4016d7:	66 0f 57 05 61 02 00 	xorpd  0x261(%rip),%xmm0        # 401940 <_IO_stdin_used+0x50>
  4016de:	00 
  4016df:	66 0f 13 04 fd 90 08 	movlpd %xmm0,0x640890(,%rdi,8)
  4016e6:	64 00 
  4016e8:	c3                   	retq   
  4016e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004016f0 <_ZN4Body22codengen_simple_updateEd>:
  4016f0:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  4016f7:	65 00 
  4016f9:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4016fd:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401704:	60 00 
  401706:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40170d:	63 00 
  40170f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401716:	63 00 
  401718:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  40171f:	65 00 
  401721:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401725:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  40172c:	60 00 
  40172e:	f2 0f 58 0c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm1
  401735:	64 00 
  401737:	f2 0f 11 0c fd 90 08 	movsd  %xmm1,0x640890(,%rdi,8)
  40173e:	64 00 
  401740:	f2 0f 10 0c fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm1
  401747:	63 00 
  401749:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40174d:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  401754:	61 00 
  401756:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  40175d:	61 00 
  40175f:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  401766:	64 00 
  401768:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  40176f:	62 00 
  401771:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  401778:	62 00 
  40177a:	c3                   	retq   
  40177b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401780 <_ZN4Body9add_forceEPS_>:
  401780:	41 56                	push   %r14
  401782:	53                   	push   %rbx
  401783:	48 83 ec 18          	sub    $0x18,%rsp
  401787:	49 89 f6             	mov    %rsi,%r14
  40178a:	48 89 fb             	mov    %rdi,%rbx
  40178d:	4c 39 f3             	cmp    %r14,%rbx
  401790:	0f 84 cb 00 00 00    	je     401861 <_ZN4Body9add_forceEPS_+0xe1>
  401796:	f2 42 0f 10 24 f5 90 	movsd  0x611a90(,%r14,8),%xmm4
  40179d:	1a 61 00 
  4017a0:	f2 0f 5c 24 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm4
  4017a7:	61 00 
  4017a9:	f2 42 0f 10 1c f5 90 	movsd  0x621490(,%r14,8),%xmm3
  4017b0:	14 62 00 
  4017b3:	f2 0f 5c 1c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm3
  4017ba:	62 00 
  4017bc:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4017c0:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  4017c4:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4017c8:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  4017cc:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4017d0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4017d3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4017d7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4017db:	7b 21                	jnp    4017fe <_ZN4Body9add_forceEPS_+0x7e>
  4017dd:	66 0f 28 c1          	movapd %xmm1,%xmm0
  4017e1:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  4017e7:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  4017ed:	e8 8e ef ff ff       	callq  400780 <sqrt@plt>
  4017f2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4017f8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4017fe:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  401805:	60 00 
  401807:	f2 0f 59 0d 19 01 00 	mulsd  0x119(%rip),%xmm1        # 401928 <_IO_stdin_used+0x38>
  40180e:	00 
  40180f:	f2 42 0f 59 0c f5 90 	mulsd  0x602090(,%r14,8),%xmm1
  401816:	20 60 00 
  401819:	66 0f 28 d0          	movapd %xmm0,%xmm2
  40181d:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  401821:	f2 0f 58 15 07 01 00 	addsd  0x107(%rip),%xmm2        # 401930 <_IO_stdin_used+0x40>
  401828:	00 
  401829:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  40182d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401831:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401835:	f2 0f 58 24 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm4
  40183c:	65 00 
  40183e:	f2 0f 11 24 dd 90 02 	movsd  %xmm4,0x650290(,%rbx,8)
  401845:	65 00 
  401847:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40184b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40184f:	f2 0f 58 1c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm3
  401856:	65 00 
  401858:	f2 0f 11 1c dd 90 fc 	movsd  %xmm3,0x65fc90(,%rbx,8)
  40185f:	65 00 
  401861:	48 83 c4 18          	add    $0x18,%rsp
  401865:	5b                   	pop    %rbx
  401866:	41 5e                	pop    %r14
  401868:	c3                   	retq   
  401869:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401870 <__libc_csu_init>:
  401870:	41 57                	push   %r15
  401872:	41 56                	push   %r14
  401874:	41 89 ff             	mov    %edi,%r15d
  401877:	41 55                	push   %r13
  401879:	41 54                	push   %r12
  40187b:	4c 8d 25 56 05 20 00 	lea    0x200556(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  401882:	55                   	push   %rbp
  401883:	48 8d 2d 5e 05 20 00 	lea    0x20055e(%rip),%rbp        # 601de8 <__init_array_end>
  40188a:	53                   	push   %rbx
  40188b:	49 89 f6             	mov    %rsi,%r14
  40188e:	49 89 d5             	mov    %rdx,%r13
  401891:	4c 29 e5             	sub    %r12,%rbp
  401894:	48 83 ec 08          	sub    $0x8,%rsp
  401898:	48 c1 fd 03          	sar    $0x3,%rbp
  40189c:	e8 27 ee ff ff       	callq  4006c8 <_init>
  4018a1:	48 85 ed             	test   %rbp,%rbp
  4018a4:	74 20                	je     4018c6 <__libc_csu_init+0x56>
  4018a6:	31 db                	xor    %ebx,%ebx
  4018a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4018af:	00 
  4018b0:	4c 89 ea             	mov    %r13,%rdx
  4018b3:	4c 89 f6             	mov    %r14,%rsi
  4018b6:	44 89 ff             	mov    %r15d,%edi
  4018b9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4018bd:	48 83 c3 01          	add    $0x1,%rbx
  4018c1:	48 39 eb             	cmp    %rbp,%rbx
  4018c4:	75 ea                	jne    4018b0 <__libc_csu_init+0x40>
  4018c6:	48 83 c4 08          	add    $0x8,%rsp
  4018ca:	5b                   	pop    %rbx
  4018cb:	5d                   	pop    %rbp
  4018cc:	41 5c                	pop    %r12
  4018ce:	41 5d                	pop    %r13
  4018d0:	41 5e                	pop    %r14
  4018d2:	41 5f                	pop    %r15
  4018d4:	c3                   	retq   
  4018d5:	90                   	nop
  4018d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4018dd:	00 00 00 

00000000004018e0 <__libc_csu_fini>:
  4018e0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004018e4 <_fini>:
  4018e4:	48 83 ec 08          	sub    $0x8,%rsp
  4018e8:	48 83 c4 08          	add    $0x8,%rsp
  4018ec:	c3                   	retq   
