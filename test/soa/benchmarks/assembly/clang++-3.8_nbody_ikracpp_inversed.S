
bin/clang++-3.8_nbody_ikracpp_inversed:     file format elf64-x86-64


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

00000000004007a0 <_GLOBAL__sub_I_ikracpp_inversed.cc>:
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
  4007cf:	49 c7 c0 00 19 40 00 	mov    $0x401900,%r8
  4007d6:	48 c7 c1 90 18 40 00 	mov    $0x401890,%rcx
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
  40090c:	f2 0f 10 05 04 10 00 	movsd  0x1004(%rip),%xmm0        # 401918 <_IO_stdin_used+0x8>
  400913:	00 
  400914:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400918:	f2 0f 10 05 00 10 00 	movsd  0x1000(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  40091f:	00 
  400920:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400924:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400928:	f2 0f 10 05 f8 0f 00 	movsd  0xff8(%rip),%xmm0        # 401928 <_IO_stdin_used+0x18>
  40092f:	00 
  400930:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400934:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  40093a:	e8 31 fe ff ff       	callq  400770 <rand@plt>
  40093f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400942:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400946:	f2 0f 5e 0d ca 0f 00 	divsd  0xfca(%rip),%xmm1        # 401918 <_IO_stdin_used+0x8>
  40094d:	00 
  40094e:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400952:	f2 0f 10 05 d6 0f 00 	movsd  0xfd6(%rip),%xmm0        # 401930 <_IO_stdin_used+0x20>
  400959:	00 
  40095a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  40095e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400964:	e8 07 fe ff ff       	callq  400770 <rand@plt>
  400969:	0f 57 c0             	xorps  %xmm0,%xmm0
  40096c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400970:	f2 0f 5e 05 a0 0f 00 	divsd  0xfa0(%rip),%xmm0        # 401918 <_IO_stdin_used+0x8>
  400977:	00 
  400978:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40097c:	f2 0f 58 05 ac 0f 00 	addsd  0xfac(%rip),%xmm0        # 401930 <_IO_stdin_used+0x20>
  400983:	00 
  400984:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  40098a:	e8 e1 fd ff ff       	callq  400770 <rand@plt>
  40098f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400992:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400996:	f2 0f 5e 0d 7a 0f 00 	divsd  0xf7a(%rip),%xmm1        # 401918 <_IO_stdin_used+0x8>
  40099d:	00 
  40099e:	f2 0f 10 05 92 0f 00 	movsd  0xf92(%rip),%xmm0        # 401938 <_IO_stdin_used+0x28>
  4009a5:	00 
  4009a6:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4009aa:	f2 0f 5e 0d 76 0f 00 	divsd  0xf76(%rip),%xmm1        # 401928 <_IO_stdin_used+0x18>
  4009b1:	00 
  4009b2:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  4009b8:	e8 b3 fd ff ff       	callq  400770 <rand@plt>
  4009bd:	0f 57 c0             	xorps  %xmm0,%xmm0
  4009c0:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  4009c4:	f2 0f 5e 05 4c 0f 00 	divsd  0xf4c(%rip),%xmm0        # 401918 <_IO_stdin_used+0x8>
  4009cb:	00 
  4009cc:	f2 0f 58 05 64 0f 00 	addsd  0xf64(%rip),%xmm0        # 401938 <_IO_stdin_used+0x28>
  4009d3:	00 
  4009d4:	f2 0f 5e 05 4c 0f 00 	divsd  0xf4c(%rip),%xmm0        # 401928 <_IO_stdin_used+0x18>
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
  400b60:	4c 89 ff             	mov    %r15,%rdi
  400b63:	48 89 de             	mov    %rbx,%rsi
  400b66:	e8 35 0c 00 00       	callq  4017a0 <_ZN4Body9add_forceEPS_>
  400b6b:	48 ff c3             	inc    %rbx
  400b6e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400b75:	75 e9                	jne    400b60 <_Z14run_simulationv+0x100>
  400b77:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400b7e:	75 d0                	jne    400b50 <_Z14run_simulationv+0xf0>
  400b80:	bb 08 00 00 00       	mov    $0x8,%ebx
  400b85:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400b8c:	00 00 00 00 
  400b90:	4c 89 f7             	mov    %r14,%rdi
  400b93:	f2 0f 10 05 85 0d 00 	movsd  0xd85(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  400b9a:	00 
  400b9b:	e8 70 0a 00 00       	callq  401610 <_ZN4Body6updateEd>
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
  400ca0:	4c 89 ff             	mov    %r15,%rdi
  400ca3:	48 89 de             	mov    %rbx,%rsi
  400ca6:	e8 f5 0a 00 00       	callq  4017a0 <_ZN4Body9add_forceEPS_>
  400cab:	48 ff c3             	inc    %rbx
  400cae:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400cb5:	75 e9                	jne    400ca0 <_Z14run_simulationv+0x240>
  400cb7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400cbe:	75 d0                	jne    400c90 <_Z14run_simulationv+0x230>
  400cc0:	bb 08 00 00 00       	mov    $0x8,%ebx
  400cc5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400ccc:	00 00 00 00 
  400cd0:	4c 89 f7             	mov    %r14,%rdi
  400cd3:	f2 0f 10 05 45 0c 00 	movsd  0xc45(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  400cda:	00 
  400cdb:	e8 30 09 00 00       	callq  401610 <_ZN4Body6updateEd>
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
  400de0:	4c 89 ff             	mov    %r15,%rdi
  400de3:	48 89 de             	mov    %rbx,%rsi
  400de6:	e8 b5 09 00 00       	callq  4017a0 <_ZN4Body9add_forceEPS_>
  400deb:	48 ff c3             	inc    %rbx
  400dee:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400df5:	75 e9                	jne    400de0 <_Z14run_simulationv+0x380>
  400df7:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dfe:	75 d0                	jne    400dd0 <_Z14run_simulationv+0x370>
  400e00:	bb 08 00 00 00       	mov    $0x8,%ebx
  400e05:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400e0c:	00 00 00 00 
  400e10:	4c 89 f7             	mov    %r14,%rdi
  400e13:	f2 0f 10 05 05 0b 00 	movsd  0xb05(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  400e1a:	00 
  400e1b:	e8 f0 07 00 00       	callq  401610 <_ZN4Body6updateEd>
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
  400f20:	4c 89 ff             	mov    %r15,%rdi
  400f23:	48 89 de             	mov    %rbx,%rsi
  400f26:	e8 75 08 00 00       	callq  4017a0 <_ZN4Body9add_forceEPS_>
  400f2b:	48 ff c3             	inc    %rbx
  400f2e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400f35:	75 e9                	jne    400f20 <_Z14run_simulationv+0x4c0>
  400f37:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400f3e:	75 d0                	jne    400f10 <_Z14run_simulationv+0x4b0>
  400f40:	bb 08 00 00 00       	mov    $0x8,%ebx
  400f45:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400f4c:	00 00 00 00 
  400f50:	4c 89 f7             	mov    %r14,%rdi
  400f53:	f2 0f 10 05 c5 09 00 	movsd  0x9c5(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  400f5a:	00 
  400f5b:	e8 b0 06 00 00       	callq  401610 <_ZN4Body6updateEd>
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
  401060:	4c 89 ff             	mov    %r15,%rdi
  401063:	48 89 de             	mov    %rbx,%rsi
  401066:	e8 35 07 00 00       	callq  4017a0 <_ZN4Body9add_forceEPS_>
  40106b:	48 ff c3             	inc    %rbx
  40106e:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  401075:	75 e9                	jne    401060 <_Z14run_simulationv+0x600>
  401077:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  40107e:	75 d0                	jne    401050 <_Z14run_simulationv+0x5f0>
  401080:	4c 89 f7             	mov    %r14,%rdi
  401083:	f2 0f 10 05 95 08 00 	movsd  0x895(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  40108a:	00 
  40108b:	e8 80 05 00 00       	callq  401610 <_ZN4Body6updateEd>
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
  4010d3:	f2 0f 10 05 45 08 00 	movsd  0x845(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  4010da:	00 
  4010db:	e8 20 06 00 00       	callq  401700 <_ZN4Body22codengen_simple_updateEd>
  4010e0:	48 ff c3             	inc    %rbx
  4010e3:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4010ea:	75 e4                	jne    4010d0 <_Z10run_simplev+0x20>
  4010ec:	ff c5                	inc    %ebp
  4010ee:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
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
  40111a:	48 89 c3             	mov    %rax,%rbx
  40111d:	e8 be f7 ff ff       	callq  4008e0 <_Z13instantiationv>
  401122:	e8 39 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401127:	49 89 c6             	mov    %rax,%r14
  40112a:	49 29 de             	sub    %rbx,%r14
  40112d:	e8 2e f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401132:	48 89 c5             	mov    %rax,%rbp
  401135:	e8 26 f9 ff ff       	callq  400a60 <_Z14run_simulationv>
  40113a:	e8 21 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40113f:	48 89 c3             	mov    %rax,%rbx
  401142:	48 29 eb             	sub    %rbp,%rbx
  401145:	31 ed                	xor    %ebp,%ebp
  401147:	e8 14 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40114c:	49 89 c5             	mov    %rax,%r13
  40114f:	f2 0f 10 05 c9 07 00 	movsd  0x7c9(%rip),%xmm0        # 401920 <_IO_stdin_used+0x10>
  401156:	00 
  401157:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40115e:	00 00 
  401160:	31 c0                	xor    %eax,%eax
  401162:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401169:	1f 84 00 00 00 00 00 
  401170:	48 8d 48 01          	lea    0x1(%rax),%rcx
  401174:	ba 64 00 00 00       	mov    $0x64,%edx
  401179:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401180:	f2 0f 10 0c cd 90 02 	movsd  0x650290(,%rcx,8),%xmm1
  401187:	65 00 
  401189:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40118d:	f2 0f 5e 0c c5 98 20 	divsd  0x602098(,%rax,8),%xmm1
  401194:	60 00 
  401196:	f2 0f 58 0c cd 90 0e 	addsd  0x630e90(,%rcx,8),%xmm1
  40119d:	63 00 
  40119f:	f2 0f 11 0c cd 90 0e 	movsd  %xmm1,0x630e90(,%rcx,8)
  4011a6:	63 00 
  4011a8:	f2 0f 10 14 cd 90 fc 	movsd  0x65fc90(,%rcx,8),%xmm2
  4011af:	65 00 
  4011b1:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  4011b5:	f2 0f 5e 14 c5 98 20 	divsd  0x602098(,%rax,8),%xmm2
  4011bc:	60 00 
  4011be:	f2 0f 58 14 cd 90 08 	addsd  0x640890(,%rcx,8),%xmm2
  4011c5:	64 00 
  4011c7:	f2 0f 11 14 cd 90 08 	movsd  %xmm2,0x640890(,%rcx,8)
  4011ce:	64 00 
  4011d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011d4:	f2 0f 58 0c cd 90 1a 	addsd  0x611a90(,%rcx,8),%xmm1
  4011db:	61 00 
  4011dd:	f2 0f 11 0c cd 90 1a 	movsd  %xmm1,0x611a90(,%rcx,8)
  4011e4:	61 00 
  4011e6:	f2 0f 10 0c cd 90 08 	movsd  0x640890(,%rcx,8),%xmm1
  4011ed:	64 00 
  4011ef:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011f3:	f2 0f 58 0c cd 90 14 	addsd  0x621490(,%rcx,8),%xmm1
  4011fa:	62 00 
  4011fc:	ff ca                	dec    %edx
  4011fe:	f2 0f 11 0c cd 90 14 	movsd  %xmm1,0x621490(,%rcx,8)
  401205:	62 00 
  401207:	0f 85 73 ff ff ff    	jne    401180 <main+0x80>
  40120d:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401214:	48 89 c8             	mov    %rcx,%rax
  401217:	0f 85 53 ff ff ff    	jne    401170 <main+0x70>
  40121d:	ff c5                	inc    %ebp
  40121f:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
  401225:	0f 85 35 ff ff ff    	jne    401160 <main+0x60>
  40122b:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401232:	de 1b 43 
  401235:	4c 89 f0             	mov    %r14,%rax
  401238:	48 f7 e9             	imul   %rcx
  40123b:	49 89 d6             	mov    %rdx,%r14
  40123e:	4c 89 f0             	mov    %r14,%rax
  401241:	48 c1 e8 3f          	shr    $0x3f,%rax
  401245:	49 c1 fe 12          	sar    $0x12,%r14
  401249:	49 01 c6             	add    %rax,%r14
  40124c:	48 89 d8             	mov    %rbx,%rax
  40124f:	48 f7 e9             	imul   %rcx
  401252:	49 89 d7             	mov    %rdx,%r15
  401255:	4c 89 f8             	mov    %r15,%rax
  401258:	48 c1 e8 3f          	shr    $0x3f,%rax
  40125c:	49 c1 ff 12          	sar    $0x12,%r15
  401260:	49 01 c7             	add    %rax,%r15
  401263:	e8 f8 f4 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401268:	49 89 c4             	mov    %rax,%r12
  40126b:	4d 29 ec             	sub    %r13,%r12
  40126e:	bd 0b 00 00 00       	mov    $0xb,%ebp
  401273:	31 f6                	xor    %esi,%esi
  401275:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40127c:	00 00 00 00 
  401280:	48 8d 5e 01          	lea    0x1(%rsi),%rbx
  401284:	f2 0f 10 04 f5 98 1a 	movsd  0x611a98(,%rsi,8),%xmm0
  40128b:	61 00 
  40128d:	f2 0f 5a c8          	cvtsd2ss %xmm0,%xmm1
  401291:	66 0f 7e c8          	movd   %xmm1,%eax
  401295:	01 e8                	add    %ebp,%eax
  401297:	f2 0f 10 0c f5 98 14 	movsd  0x621498(,%rsi,8),%xmm1
  40129e:	62 00 
  4012a0:	f2 0f 5a d1          	cvtsd2ss %xmm1,%xmm2
  4012a4:	66 0f 7e d5          	movd   %xmm2,%ebp
  4012a8:	01 c5                	add    %eax,%ebp
  4012aa:	48 63 c5             	movslq %ebp,%rax
  4012ad:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  4012b4:	48 89 c1             	mov    %rax,%rcx
  4012b7:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4012bb:	48 c1 f8 33          	sar    $0x33,%rax
  4012bf:	01 c8                	add    %ecx,%eax
  4012c1:	44 69 e8 87 d6 12 00 	imul   $0x12d687,%eax,%r13d
  4012c8:	48 83 fe 0a          	cmp    $0xa,%rsi
  4012cc:	73 0c                	jae    4012da <main+0x1da>
  4012ce:	bf 70 19 40 00       	mov    $0x401970,%edi
  4012d3:	b0 02                	mov    $0x2,%al
  4012d5:	e8 26 f4 ff ff       	callq  400700 <printf@plt>
  4012da:	44 29 ed             	sub    %r13d,%ebp
  4012dd:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  4012e4:	48 89 de             	mov    %rbx,%rsi
  4012e7:	75 97                	jne    401280 <main+0x180>
  4012e9:	4c 89 e0             	mov    %r12,%rax
  4012ec:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  4012f3:	de 1b 43 
  4012f6:	48 f7 e9             	imul   %rcx
  4012f9:	48 89 d0             	mov    %rdx,%rax
  4012fc:	48 c1 e8 3f          	shr    $0x3f,%rax
  401300:	48 c1 fa 12          	sar    $0x12,%rdx
  401304:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401308:	bf 85 19 40 00       	mov    $0x401985,%edi
  40130d:	31 c0                	xor    %eax,%eax
  40130f:	4c 89 f6             	mov    %r14,%rsi
  401312:	4c 89 fa             	mov    %r15,%rdx
  401315:	41 89 e8             	mov    %ebp,%r8d
  401318:	e8 e3 f3 ff ff       	callq  400700 <printf@plt>
  40131d:	31 c0                	xor    %eax,%eax
  40131f:	48 83 c4 08          	add    $0x8,%rsp
  401323:	5b                   	pop    %rbx
  401324:	41 5c                	pop    %r12
  401326:	41 5d                	pop    %r13
  401328:	41 5e                	pop    %r14
  40132a:	41 5f                	pop    %r15
  40132c:	5d                   	pop    %rbp
  40132d:	c3                   	retq   
  40132e:	66 90                	xchg   %ax,%ax

0000000000401330 <_Z19codegen_reset_forceP4Body>:
  401330:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401337:	00 00 00 00 00 
  40133c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  401343:	00 00 00 00 00 
  401348:	c3                   	retq   
  401349:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401350 <_Z17codegen_add_forceP4BodyS0_>:
  401350:	41 56                	push   %r14
  401352:	53                   	push   %rbx
  401353:	48 83 ec 18          	sub    $0x18,%rsp
  401357:	49 89 f6             	mov    %rsi,%r14
  40135a:	48 89 fb             	mov    %rdi,%rbx
  40135d:	4c 39 f3             	cmp    %r14,%rbx
  401360:	0f 84 cb 00 00 00    	je     401431 <_Z17codegen_add_forceP4BodyS0_+0xe1>
  401366:	f2 42 0f 10 24 f5 90 	movsd  0x611a90(,%r14,8),%xmm4
  40136d:	1a 61 00 
  401370:	f2 0f 5c 24 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm4
  401377:	61 00 
  401379:	f2 42 0f 10 1c f5 90 	movsd  0x621490(,%r14,8),%xmm3
  401380:	14 62 00 
  401383:	f2 0f 5c 1c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm3
  40138a:	62 00 
  40138c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401390:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  401394:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401398:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40139c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4013a0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4013a3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4013a7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4013ab:	7b 21                	jnp    4013ce <_Z17codegen_add_forceP4BodyS0_+0x7e>
  4013ad:	66 0f 28 c1          	movapd %xmm1,%xmm0
  4013b1:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  4013b7:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  4013bd:	e8 be f3 ff ff       	callq  400780 <sqrt@plt>
  4013c2:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  4013c8:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  4013ce:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  4013d5:	60 00 
  4013d7:	f2 0f 59 0d 69 05 00 	mulsd  0x569(%rip),%xmm1        # 401948 <_IO_stdin_used+0x38>
  4013de:	00 
  4013df:	f2 42 0f 59 0c f5 90 	mulsd  0x602090(,%r14,8),%xmm1
  4013e6:	20 60 00 
  4013e9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4013ed:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  4013f1:	f2 0f 58 15 57 05 00 	addsd  0x557(%rip),%xmm2        # 401950 <_IO_stdin_used+0x40>
  4013f8:	00 
  4013f9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4013fd:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401401:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401405:	f2 0f 58 24 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm4
  40140c:	65 00 
  40140e:	f2 0f 11 24 dd 90 02 	movsd  %xmm4,0x650290(,%rbx,8)
  401415:	65 00 
  401417:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40141b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40141f:	f2 0f 58 1c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm3
  401426:	65 00 
  401428:	f2 0f 11 1c dd 90 fc 	movsd  %xmm3,0x65fc90(,%rbx,8)
  40142f:	65 00 
  401431:	48 83 c4 18          	add    $0x18,%rsp
  401435:	5b                   	pop    %rbx
  401436:	41 5e                	pop    %r14
  401438:	c3                   	retq   
  401439:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401440 <_Z14codegen_updateP4Bodyd>:
  401440:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401447:	65 00 
  401449:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40144d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401454:	60 00 
  401456:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40145d:	63 00 
  40145f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401466:	63 00 
  401468:	f2 0f 10 14 fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm2
  40146f:	65 00 
  401471:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401475:	f2 0f 5e 14 fd 90 20 	divsd  0x602090(,%rdi,8),%xmm2
  40147c:	60 00 
  40147e:	f2 0f 58 14 fd 90 08 	addsd  0x640890(,%rdi,8),%xmm2
  401485:	64 00 
  401487:	f2 0f 11 14 fd 90 08 	movsd  %xmm2,0x640890(,%rdi,8)
  40148e:	64 00 
  401490:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401494:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  40149b:	61 00 
  40149d:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  4014a4:	61 00 
  4014a6:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  4014ad:	64 00 
  4014af:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  4014b6:	62 00 
  4014b8:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  4014bf:	62 00 
  4014c1:	f2 0f 10 15 67 04 00 	movsd  0x467(%rip),%xmm2        # 401930 <_IO_stdin_used+0x20>
  4014c8:	00 
  4014c9:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  4014cd:	77 0a                	ja     4014d9 <_Z14codegen_updateP4Bodyd+0x99>
  4014cf:	66 0f 2e 0d 69 04 00 	ucomisd 0x469(%rip),%xmm1        # 401940 <_IO_stdin_used+0x30>
  4014d6:	00 
  4014d7:	76 23                	jbe    4014fc <_Z14codegen_updateP4Bodyd+0xbc>
  4014d9:	f2 0f 10 04 fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm0
  4014e0:	63 00 
  4014e2:	66 0f 57 05 76 04 00 	xorpd  0x476(%rip),%xmm0        # 401960 <_IO_stdin_used+0x50>
  4014e9:	00 
  4014ea:	66 0f 13 04 fd 90 0e 	movlpd %xmm0,0x630e90(,%rdi,8)
  4014f1:	63 00 
  4014f3:	f2 0f 10 04 fd 90 14 	movsd  0x621490(,%rdi,8),%xmm0
  4014fa:	62 00 
  4014fc:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  401500:	77 0a                	ja     40150c <_Z14codegen_updateP4Bodyd+0xcc>
  401502:	66 0f 2e 05 36 04 00 	ucomisd 0x436(%rip),%xmm0        # 401940 <_IO_stdin_used+0x30>
  401509:	00 
  40150a:	76 1a                	jbe    401526 <_Z14codegen_updateP4Bodyd+0xe6>
  40150c:	f2 0f 10 04 fd 90 08 	movsd  0x640890(,%rdi,8),%xmm0
  401513:	64 00 
  401515:	66 0f 57 05 43 04 00 	xorpd  0x443(%rip),%xmm0        # 401960 <_IO_stdin_used+0x50>
  40151c:	00 
  40151d:	66 0f 13 04 fd 90 08 	movlpd %xmm0,0x640890(,%rdi,8)
  401524:	64 00 
  401526:	c3                   	retq   
  401527:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40152e:	00 00 

0000000000401530 <_Z26codegen_reset_force_manualP4Body>:
  401530:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401537:	00 00 00 00 00 
  40153c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  401543:	00 00 00 00 00 
  401548:	c3                   	retq   
  401549:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401550 <_Z24codegen_reset_force_halfP4Body>:
  401550:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  401557:	00 00 00 00 00 
  40155c:	c3                   	retq   
  40155d:	0f 1f 00             	nopl   (%rax)

0000000000401560 <_Z18codegen_reset_massP4Body>:
  401560:	48 c7 04 fd 90 20 60 	movq   $0x0,0x602090(,%rdi,8)
  401567:	00 00 00 00 00 
  40156c:	c3                   	retq   
  40156d:	0f 1f 00             	nopl   (%rax)

0000000000401570 <_Z22codengen_simple_updateP4Bodyd>:
  401570:	b8 64 00 00 00       	mov    $0x64,%eax
  401575:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40157c:	00 00 00 00 
  401580:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401587:	65 00 
  401589:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40158d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401594:	60 00 
  401596:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40159d:	63 00 
  40159f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  4015a6:	63 00 
  4015a8:	f2 0f 10 14 fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm2
  4015af:	65 00 
  4015b1:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  4015b5:	f2 0f 5e 14 fd 90 20 	divsd  0x602090(,%rdi,8),%xmm2
  4015bc:	60 00 
  4015be:	f2 0f 58 14 fd 90 08 	addsd  0x640890(,%rdi,8),%xmm2
  4015c5:	64 00 
  4015c7:	f2 0f 11 14 fd 90 08 	movsd  %xmm2,0x640890(,%rdi,8)
  4015ce:	64 00 
  4015d0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015d4:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  4015db:	61 00 
  4015dd:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  4015e4:	61 00 
  4015e6:	f2 0f 10 0c fd 90 08 	movsd  0x640890(,%rdi,8),%xmm1
  4015ed:	64 00 
  4015ef:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4015f3:	f2 0f 58 0c fd 90 14 	addsd  0x621490(,%rdi,8),%xmm1
  4015fa:	62 00 
  4015fc:	ff c8                	dec    %eax
  4015fe:	f2 0f 11 0c fd 90 14 	movsd  %xmm1,0x621490(,%rdi,8)
  401605:	62 00 
  401607:	0f 85 73 ff ff ff    	jne    401580 <_Z22codengen_simple_updateP4Bodyd+0x10>
  40160d:	c3                   	retq   
  40160e:	66 90                	xchg   %ax,%ax

0000000000401610 <_ZN4Body6updateEd>:
  401610:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401617:	65 00 
  401619:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40161d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401624:	60 00 
  401626:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40162d:	63 00 
  40162f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401636:	63 00 
  401638:	f2 0f 10 14 fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm2
  40163f:	65 00 
  401641:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401645:	f2 0f 5e 14 fd 90 20 	divsd  0x602090(,%rdi,8),%xmm2
  40164c:	60 00 
  40164e:	f2 0f 58 14 fd 90 08 	addsd  0x640890(,%rdi,8),%xmm2
  401655:	64 00 
  401657:	f2 0f 11 14 fd 90 08 	movsd  %xmm2,0x640890(,%rdi,8)
  40165e:	64 00 
  401660:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401664:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  40166b:	61 00 
  40166d:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  401674:	61 00 
  401676:	f2 0f 59 04 fd 90 08 	mulsd  0x640890(,%rdi,8),%xmm0
  40167d:	64 00 
  40167f:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  401686:	62 00 
  401688:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  40168f:	62 00 
  401691:	f2 0f 10 15 97 02 00 	movsd  0x297(%rip),%xmm2        # 401930 <_IO_stdin_used+0x20>
  401698:	00 
  401699:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  40169d:	77 0a                	ja     4016a9 <_ZN4Body6updateEd+0x99>
  40169f:	66 0f 2e 0d 99 02 00 	ucomisd 0x299(%rip),%xmm1        # 401940 <_IO_stdin_used+0x30>
  4016a6:	00 
  4016a7:	76 23                	jbe    4016cc <_ZN4Body6updateEd+0xbc>
  4016a9:	f2 0f 10 04 fd 90 0e 	movsd  0x630e90(,%rdi,8),%xmm0
  4016b0:	63 00 
  4016b2:	66 0f 57 05 a6 02 00 	xorpd  0x2a6(%rip),%xmm0        # 401960 <_IO_stdin_used+0x50>
  4016b9:	00 
  4016ba:	66 0f 13 04 fd 90 0e 	movlpd %xmm0,0x630e90(,%rdi,8)
  4016c1:	63 00 
  4016c3:	f2 0f 10 04 fd 90 14 	movsd  0x621490(,%rdi,8),%xmm0
  4016ca:	62 00 
  4016cc:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  4016d0:	77 0a                	ja     4016dc <_ZN4Body6updateEd+0xcc>
  4016d2:	66 0f 2e 05 66 02 00 	ucomisd 0x266(%rip),%xmm0        # 401940 <_IO_stdin_used+0x30>
  4016d9:	00 
  4016da:	76 1a                	jbe    4016f6 <_ZN4Body6updateEd+0xe6>
  4016dc:	f2 0f 10 04 fd 90 08 	movsd  0x640890(,%rdi,8),%xmm0
  4016e3:	64 00 
  4016e5:	66 0f 57 05 73 02 00 	xorpd  0x273(%rip),%xmm0        # 401960 <_IO_stdin_used+0x50>
  4016ec:	00 
  4016ed:	66 0f 13 04 fd 90 08 	movlpd %xmm0,0x640890(,%rdi,8)
  4016f4:	64 00 
  4016f6:	c3                   	retq   
  4016f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4016fe:	00 00 

0000000000401700 <_ZN4Body22codengen_simple_updateEd>:
  401700:	b8 64 00 00 00       	mov    $0x64,%eax
  401705:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40170c:	00 00 00 00 
  401710:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  401717:	65 00 
  401719:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40171d:	f2 0f 5e 0c fd 90 20 	divsd  0x602090(,%rdi,8),%xmm1
  401724:	60 00 
  401726:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  40172d:	63 00 
  40172f:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  401736:	63 00 
  401738:	f2 0f 10 14 fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm2
  40173f:	65 00 
  401741:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401745:	f2 0f 5e 14 fd 90 20 	divsd  0x602090(,%rdi,8),%xmm2
  40174c:	60 00 
  40174e:	f2 0f 58 14 fd 90 08 	addsd  0x640890(,%rdi,8),%xmm2
  401755:	64 00 
  401757:	f2 0f 11 14 fd 90 08 	movsd  %xmm2,0x640890(,%rdi,8)
  40175e:	64 00 
  401760:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401764:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  40176b:	61 00 
  40176d:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  401774:	61 00 
  401776:	f2 0f 10 0c fd 90 08 	movsd  0x640890(,%rdi,8),%xmm1
  40177d:	64 00 
  40177f:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401783:	f2 0f 58 0c fd 90 14 	addsd  0x621490(,%rdi,8),%xmm1
  40178a:	62 00 
  40178c:	ff c8                	dec    %eax
  40178e:	f2 0f 11 0c fd 90 14 	movsd  %xmm1,0x621490(,%rdi,8)
  401795:	62 00 
  401797:	0f 85 73 ff ff ff    	jne    401710 <_ZN4Body22codengen_simple_updateEd+0x10>
  40179d:	c3                   	retq   
  40179e:	66 90                	xchg   %ax,%ax

00000000004017a0 <_ZN4Body9add_forceEPS_>:
  4017a0:	41 56                	push   %r14
  4017a2:	53                   	push   %rbx
  4017a3:	48 83 ec 18          	sub    $0x18,%rsp
  4017a7:	49 89 f6             	mov    %rsi,%r14
  4017aa:	48 89 fb             	mov    %rdi,%rbx
  4017ad:	4c 39 f3             	cmp    %r14,%rbx
  4017b0:	0f 84 cb 00 00 00    	je     401881 <_ZN4Body9add_forceEPS_+0xe1>
  4017b6:	f2 42 0f 10 24 f5 90 	movsd  0x611a90(,%r14,8),%xmm4
  4017bd:	1a 61 00 
  4017c0:	f2 0f 5c 24 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm4
  4017c7:	61 00 
  4017c9:	f2 42 0f 10 1c f5 90 	movsd  0x621490(,%r14,8),%xmm3
  4017d0:	14 62 00 
  4017d3:	f2 0f 5c 1c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm3
  4017da:	62 00 
  4017dc:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4017e0:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  4017e4:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4017e8:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  4017ec:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  4017f0:	0f 57 c0             	xorps  %xmm0,%xmm0
  4017f3:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  4017f7:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  4017fb:	7b 21                	jnp    40181e <_ZN4Body9add_forceEPS_+0x7e>
  4017fd:	66 0f 28 c1          	movapd %xmm1,%xmm0
  401801:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  401807:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40180d:	e8 6e ef ff ff       	callq  400780 <sqrt@plt>
  401812:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  401818:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  40181e:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  401825:	60 00 
  401827:	f2 0f 59 0d 19 01 00 	mulsd  0x119(%rip),%xmm1        # 401948 <_IO_stdin_used+0x38>
  40182e:	00 
  40182f:	f2 42 0f 59 0c f5 90 	mulsd  0x602090(,%r14,8),%xmm1
  401836:	20 60 00 
  401839:	66 0f 28 d0          	movapd %xmm0,%xmm2
  40183d:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  401841:	f2 0f 58 15 07 01 00 	addsd  0x107(%rip),%xmm2        # 401950 <_IO_stdin_used+0x40>
  401848:	00 
  401849:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  40184d:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  401851:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  401855:	f2 0f 58 24 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm4
  40185c:	65 00 
  40185e:	f2 0f 11 24 dd 90 02 	movsd  %xmm4,0x650290(,%rbx,8)
  401865:	65 00 
  401867:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  40186b:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  40186f:	f2 0f 58 1c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm3
  401876:	65 00 
  401878:	f2 0f 11 1c dd 90 fc 	movsd  %xmm3,0x65fc90(,%rbx,8)
  40187f:	65 00 
  401881:	48 83 c4 18          	add    $0x18,%rsp
  401885:	5b                   	pop    %rbx
  401886:	41 5e                	pop    %r14
  401888:	c3                   	retq   
  401889:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401890 <__libc_csu_init>:
  401890:	41 57                	push   %r15
  401892:	41 56                	push   %r14
  401894:	41 89 ff             	mov    %edi,%r15d
  401897:	41 55                	push   %r13
  401899:	41 54                	push   %r12
  40189b:	4c 8d 25 36 05 20 00 	lea    0x200536(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  4018a2:	55                   	push   %rbp
  4018a3:	48 8d 2d 3e 05 20 00 	lea    0x20053e(%rip),%rbp        # 601de8 <__init_array_end>
  4018aa:	53                   	push   %rbx
  4018ab:	49 89 f6             	mov    %rsi,%r14
  4018ae:	49 89 d5             	mov    %rdx,%r13
  4018b1:	4c 29 e5             	sub    %r12,%rbp
  4018b4:	48 83 ec 08          	sub    $0x8,%rsp
  4018b8:	48 c1 fd 03          	sar    $0x3,%rbp
  4018bc:	e8 07 ee ff ff       	callq  4006c8 <_init>
  4018c1:	48 85 ed             	test   %rbp,%rbp
  4018c4:	74 20                	je     4018e6 <__libc_csu_init+0x56>
  4018c6:	31 db                	xor    %ebx,%ebx
  4018c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4018cf:	00 
  4018d0:	4c 89 ea             	mov    %r13,%rdx
  4018d3:	4c 89 f6             	mov    %r14,%rsi
  4018d6:	44 89 ff             	mov    %r15d,%edi
  4018d9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4018dd:	48 83 c3 01          	add    $0x1,%rbx
  4018e1:	48 39 eb             	cmp    %rbp,%rbx
  4018e4:	75 ea                	jne    4018d0 <__libc_csu_init+0x40>
  4018e6:	48 83 c4 08          	add    $0x8,%rsp
  4018ea:	5b                   	pop    %rbx
  4018eb:	5d                   	pop    %rbp
  4018ec:	41 5c                	pop    %r12
  4018ee:	41 5d                	pop    %r13
  4018f0:	41 5e                	pop    %r14
  4018f2:	41 5f                	pop    %r15
  4018f4:	c3                   	retq   
  4018f5:	90                   	nop
  4018f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4018fd:	00 00 00 

0000000000401900 <__libc_csu_fini>:
  401900:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401904 <_fini>:
  401904:	48 83 ec 08          	sub    $0x8,%rsp
  401908:	48 83 c4 08          	add    $0x8,%rsp
  40190c:	c3                   	retq   
