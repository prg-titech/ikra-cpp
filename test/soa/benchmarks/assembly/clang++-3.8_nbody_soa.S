
bin/clang++-3.8_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004006d0 <_init>:
  4006d0:	48 83 ec 08          	sub    $0x8,%rsp
  4006d4:	48 8b 05 1d 19 20 00 	mov    0x20191d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4006db:	48 85 c0             	test   %rax,%rax
  4006de:	74 05                	je     4006e5 <_init+0x15>
  4006e0:	e8 ab 00 00 00       	callq  400790 <sqrt@plt+0x10>
  4006e5:	48 83 c4 08          	add    $0x8,%rsp
  4006e9:	c3                   	retq   

Disassembly of section .plt:

00000000004006f0 <printf@plt-0x10>:
  4006f0:	ff 35 12 19 20 00    	pushq  0x201912(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006f6:	ff 25 14 19 20 00    	jmpq   *0x201914(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400700 <printf@plt>:
  400700:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400706:	68 00 00 00 00       	pushq  $0x0
  40070b:	e9 e0 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400710 <memset@plt>:
  400710:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400716:	68 01 00 00 00       	pushq  $0x1
  40071b:	e9 d0 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400720 <_ZNSt8ios_base4InitC1Ev@plt>:
  400720:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400726:	68 02 00 00 00       	pushq  $0x2
  40072b:	e9 c0 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400730 <__libc_start_main@plt>:
  400730:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400736:	68 03 00 00 00       	pushq  $0x3
  40073b:	e9 b0 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400740 <__cxa_atexit@plt>:
  400740:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400746:	68 04 00 00 00       	pushq  $0x4
  40074b:	e9 a0 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400750 <_ZNSt8ios_base4InitD1Ev@plt>:
  400750:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400756:	68 05 00 00 00       	pushq  $0x5
  40075b:	e9 90 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400760:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400766:	68 06 00 00 00       	pushq  $0x6
  40076b:	e9 80 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400770 <rand@plt>:
  400770:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400776:	68 07 00 00 00       	pushq  $0x7
  40077b:	e9 70 ff ff ff       	jmpq   4006f0 <_init+0x20>

0000000000400780 <sqrt@plt>:
  400780:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400786:	68 08 00 00 00       	pushq  $0x8
  40078b:	e9 60 ff ff ff       	jmpq   4006f0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400790 <.plt.got>:
  400790:	ff 25 62 18 20 00    	jmpq   *0x201862(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400796:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007a0 <_GLOBAL__sub_I_soa.cc>:
  4007a0:	50                   	push   %rax
  4007a1:	bf 80 20 60 00       	mov    $0x602080,%edi
  4007a6:	e8 75 ff ff ff       	callq  400720 <_ZNSt8ios_base4InitC1Ev@plt>
  4007ab:	bf 50 07 40 00       	mov    $0x400750,%edi
  4007b0:	be 80 20 60 00       	mov    $0x602080,%esi
  4007b5:	ba 68 20 60 00       	mov    $0x602068,%edx
  4007ba:	58                   	pop    %rax
  4007bb:	e9 80 ff ff ff       	jmpq   400740 <__cxa_atexit@plt>

00000000004007c0 <_start>:
  4007c0:	31 ed                	xor    %ebp,%ebp
  4007c2:	49 89 d1             	mov    %rdx,%r9
  4007c5:	5e                   	pop    %rsi
  4007c6:	48 89 e2             	mov    %rsp,%rdx
  4007c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4007cd:	50                   	push   %rax
  4007ce:	54                   	push   %rsp
  4007cf:	49 c7 c0 b0 13 40 00 	mov    $0x4013b0,%r8
  4007d6:	48 c7 c1 40 13 40 00 	mov    $0x401340,%rcx
  4007dd:	48 c7 c7 30 11 40 00 	mov    $0x401130,%rdi
  4007e4:	e8 47 ff ff ff       	callq  400730 <__libc_start_main@plt>
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

00000000004008c0 <_Z12float_as_intf>:
  4008c0:	66 0f 7e c0          	movd   %xmm0,%eax
  4008c4:	c3                   	retq   
  4008c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4008cc:	00 00 00 00 

00000000004008d0 <_Z15Body_initializemddddd>:
  4008d0:	f2 0f 11 04 fd 90 20 	movsd  %xmm0,0x602090(,%rdi,8)
  4008d7:	60 00 
  4008d9:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  4008e0:	61 00 
  4008e2:	f2 0f 11 14 fd 90 14 	movsd  %xmm2,0x621490(,%rdi,8)
  4008e9:	62 00 
  4008eb:	f2 0f 11 1c fd 90 0e 	movsd  %xmm3,0x630e90(,%rdi,8)
  4008f2:	63 00 
  4008f4:	f2 0f 11 24 fd 90 08 	movsd  %xmm4,0x640890(,%rdi,8)
  4008fb:	64 00 
  4008fd:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  400904:	00 00 00 00 00 
  400909:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  400910:	00 00 00 00 00 
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z14Body_add_forcemm>:
  400920:	41 56                	push   %r14
  400922:	53                   	push   %rbx
  400923:	48 83 ec 18          	sub    $0x18,%rsp
  400927:	49 89 f6             	mov    %rsi,%r14
  40092a:	48 89 fb             	mov    %rdi,%rbx
  40092d:	4c 39 f3             	cmp    %r14,%rbx
  400930:	0f 84 cb 00 00 00    	je     400a01 <_Z14Body_add_forcemm+0xe1>
  400936:	f2 42 0f 10 24 f5 90 	movsd  0x611a90(,%r14,8),%xmm4
  40093d:	1a 61 00 
  400940:	f2 0f 5c 24 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm4
  400947:	61 00 
  400949:	f2 42 0f 10 1c f5 90 	movsd  0x621490(,%r14,8),%xmm3
  400950:	14 62 00 
  400953:	f2 0f 5c 1c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm3
  40095a:	62 00 
  40095c:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400960:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400964:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400968:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  40096c:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400970:	0f 57 c0             	xorps  %xmm0,%xmm0
  400973:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400977:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  40097b:	7b 21                	jnp    40099e <_Z14Body_add_forcemm+0x7e>
  40097d:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400981:	f2 0f 11 5c 24 10    	movsd  %xmm3,0x10(%rsp)
  400987:	f2 0f 11 64 24 08    	movsd  %xmm4,0x8(%rsp)
  40098d:	e8 ee fd ff ff       	callq  400780 <sqrt@plt>
  400992:	f2 0f 10 64 24 08    	movsd  0x8(%rsp),%xmm4
  400998:	f2 0f 10 5c 24 10    	movsd  0x10(%rsp),%xmm3
  40099e:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  4009a5:	60 00 
  4009a7:	f2 0f 59 0d 19 0a 00 	mulsd  0xa19(%rip),%xmm1        # 4013c8 <_IO_stdin_used+0x8>
  4009ae:	00 
  4009af:	f2 42 0f 59 0c f5 90 	mulsd  0x602090(,%r14,8),%xmm1
  4009b6:	20 60 00 
  4009b9:	66 0f 28 d0          	movapd %xmm0,%xmm2
  4009bd:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  4009c1:	f2 0f 58 15 07 0a 00 	addsd  0xa07(%rip),%xmm2        # 4013d0 <_IO_stdin_used+0x10>
  4009c8:	00 
  4009c9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  4009cd:	f2 0f 59 e1          	mulsd  %xmm1,%xmm4
  4009d1:	f2 0f 5e e0          	divsd  %xmm0,%xmm4
  4009d5:	f2 0f 58 24 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm4
  4009dc:	65 00 
  4009de:	f2 0f 11 24 dd 90 02 	movsd  %xmm4,0x650290(,%rbx,8)
  4009e5:	65 00 
  4009e7:	f2 0f 59 d9          	mulsd  %xmm1,%xmm3
  4009eb:	f2 0f 5e d8          	divsd  %xmm0,%xmm3
  4009ef:	f2 0f 58 1c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm3
  4009f6:	65 00 
  4009f8:	f2 0f 11 1c dd 90 fc 	movsd  %xmm3,0x65fc90(,%rbx,8)
  4009ff:	65 00 
  400a01:	48 83 c4 18          	add    $0x18,%rsp
  400a05:	5b                   	pop    %rbx
  400a06:	41 5e                	pop    %r14
  400a08:	c3                   	retq   
  400a09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a10 <_Z21Body_add_force_to_allm>:
  400a10:	41 57                	push   %r15
  400a12:	41 56                	push   %r14
  400a14:	53                   	push   %rbx
  400a15:	48 83 ec 10          	sub    $0x10,%rsp
  400a19:	49 89 ff             	mov    %rdi,%r15
  400a1c:	4d 89 fe             	mov    %r15,%r14
  400a1f:	49 f7 de             	neg    %r14
  400a22:	48 c7 c3 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rbx
  400a29:	f2 0f 10 1d 97 09 00 	movsd  0x997(%rip),%xmm3        # 4013c8 <_IO_stdin_used+0x8>
  400a30:	00 
  400a31:	f2 0f 10 25 97 09 00 	movsd  0x997(%rip),%xmm4        # 4013d0 <_IO_stdin_used+0x10>
  400a38:	00 
  400a39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400a40:	49 8d 04 1e          	lea    (%r14,%rbx,1),%rax
  400a44:	48 3d c0 e0 ff ff    	cmp    $0xffffffffffffe0c0,%rax
  400a4a:	0f 84 d1 00 00 00    	je     400b21 <_Z21Body_add_force_to_allm+0x111>
  400a50:	f2 42 0f 10 34 fd 90 	movsd  0x611a90(,%r15,8),%xmm6
  400a57:	1a 61 00 
  400a5a:	f2 0f 5c 34 dd 90 14 	subsd  0x621490(,%rbx,8),%xmm6
  400a61:	62 00 
  400a63:	f2 42 0f 10 2c fd 90 	movsd  0x621490(,%r15,8),%xmm5
  400a6a:	14 62 00 
  400a6d:	f2 0f 5c 2c dd 90 0e 	subsd  0x630e90(,%rbx,8),%xmm5
  400a74:	63 00 
  400a76:	66 0f 28 c6          	movapd %xmm6,%xmm0
  400a7a:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400a7e:	66 0f 28 cd          	movapd %xmm5,%xmm1
  400a82:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400a86:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400a8a:	0f 57 c0             	xorps  %xmm0,%xmm0
  400a8d:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400a91:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400a95:	7b 2f                	jnp    400ac6 <_Z21Body_add_force_to_allm+0xb6>
  400a97:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400a9b:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  400aa1:	f2 0f 11 34 24       	movsd  %xmm6,(%rsp)
  400aa6:	e8 d5 fc ff ff       	callq  400780 <sqrt@plt>
  400aab:	f2 0f 10 34 24       	movsd  (%rsp),%xmm6
  400ab0:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  400ab6:	f2 0f 10 25 12 09 00 	movsd  0x912(%rip),%xmm4        # 4013d0 <_IO_stdin_used+0x10>
  400abd:	00 
  400abe:	f2 0f 10 1d 02 09 00 	movsd  0x902(%rip),%xmm3        # 4013c8 <_IO_stdin_used+0x8>
  400ac5:	00 
  400ac6:	f2 0f 10 0c dd 90 1a 	movsd  0x611a90(,%rbx,8),%xmm1
  400acd:	61 00 
  400acf:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400ad3:	f2 42 0f 59 0c fd 90 	mulsd  0x602090(,%r15,8),%xmm1
  400ada:	20 60 00 
  400add:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400ae1:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400ae5:	f2 0f 58 d4          	addsd  %xmm4,%xmm2
  400ae9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400aed:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  400af1:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400af5:	f2 0f 58 34 dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm6
  400afc:	65 00 
  400afe:	f2 0f 11 34 dd 90 fc 	movsd  %xmm6,0x65fc90(,%rbx,8)
  400b05:	65 00 
  400b07:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  400b0b:	f2 0f 5e e8          	divsd  %xmm0,%xmm5
  400b0f:	f2 0f 58 2c dd 90 f6 	addsd  0x66f690(,%rbx,8),%xmm5
  400b16:	66 00 
  400b18:	f2 0f 11 2c dd 90 f6 	movsd  %xmm5,0x66f690(,%rbx,8)
  400b1f:	66 00 
  400b21:	48 ff c3             	inc    %rbx
  400b24:	0f 85 16 ff ff ff    	jne    400a40 <_Z21Body_add_force_to_allm+0x30>
  400b2a:	48 83 c4 10          	add    $0x10,%rsp
  400b2e:	5b                   	pop    %rbx
  400b2f:	41 5e                	pop    %r14
  400b31:	41 5f                	pop    %r15
  400b33:	c3                   	retq   
  400b34:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b3b:	00 00 00 00 00 

0000000000400b40 <_Z16Body_reset_forcem>:
  400b40:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  400b47:	00 00 00 00 00 
  400b4c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  400b53:	00 00 00 00 00 
  400b58:	c3                   	retq   
  400b59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400b60 <_Z11Body_updatemd>:
  400b60:	f2 0f 10 14 fd 90 02 	movsd  0x650290(,%rdi,8),%xmm2
  400b67:	65 00 
  400b69:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  400b6d:	f2 0f 10 1c fd 90 20 	movsd  0x602090(,%rdi,8),%xmm3
  400b74:	60 00 
  400b76:	f2 0f 5e d3          	divsd  %xmm3,%xmm2
  400b7a:	f2 0f 58 14 fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm2
  400b81:	63 00 
  400b83:	f2 0f 11 14 fd 90 0e 	movsd  %xmm2,0x630e90(,%rdi,8)
  400b8a:	63 00 
  400b8c:	f2 0f 10 0c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm1
  400b93:	65 00 
  400b95:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400b99:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400b9d:	f2 0f 58 0c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm1
  400ba4:	64 00 
  400ba6:	f2 0f 11 0c fd 90 08 	movsd  %xmm1,0x640890(,%rdi,8)
  400bad:	64 00 
  400baf:	66 0f 28 da          	movapd %xmm2,%xmm3
  400bb3:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400bb7:	f2 0f 58 1c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm3
  400bbe:	61 00 
  400bc0:	f2 0f 11 1c fd 90 1a 	movsd  %xmm3,0x611a90(,%rdi,8)
  400bc7:	61 00 
  400bc9:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400bcd:	f2 0f 58 04 fd 90 14 	addsd  0x621490(,%rdi,8),%xmm0
  400bd4:	62 00 
  400bd6:	f2 0f 11 04 fd 90 14 	movsd  %xmm0,0x621490(,%rdi,8)
  400bdd:	62 00 
  400bdf:	f2 0f 10 25 f1 07 00 	movsd  0x7f1(%rip),%xmm4        # 4013d8 <_IO_stdin_used+0x18>
  400be6:	00 
  400be7:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  400beb:	77 0a                	ja     400bf7 <_Z11Body_updatemd+0x97>
  400bed:	66 0f 2e 1d eb 07 00 	ucomisd 0x7eb(%rip),%xmm3        # 4013e0 <_IO_stdin_used+0x20>
  400bf4:	00 
  400bf5:	76 11                	jbe    400c08 <_Z11Body_updatemd+0xa8>
  400bf7:	66 0f 57 15 11 08 00 	xorpd  0x811(%rip),%xmm2        # 401410 <_IO_stdin_used+0x50>
  400bfe:	00 
  400bff:	66 0f 13 14 fd 90 0e 	movlpd %xmm2,0x630e90(,%rdi,8)
  400c06:	63 00 
  400c08:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  400c0c:	77 0a                	ja     400c18 <_Z11Body_updatemd+0xb8>
  400c0e:	66 0f 2e 05 ca 07 00 	ucomisd 0x7ca(%rip),%xmm0        # 4013e0 <_IO_stdin_used+0x20>
  400c15:	00 
  400c16:	76 11                	jbe    400c29 <_Z11Body_updatemd+0xc9>
  400c18:	66 0f 57 0d f0 07 00 	xorpd  0x7f0(%rip),%xmm1        # 401410 <_IO_stdin_used+0x50>
  400c1f:	00 
  400c20:	66 0f 13 0c fd 90 08 	movlpd %xmm1,0x640890(,%rdi,8)
  400c27:	64 00 
  400c29:	c3                   	retq   
  400c2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c30 <_Z26Body_codegen_simple_updatemd>:
  400c30:	f2 0f 10 0c fd 90 02 	movsd  0x650290(,%rdi,8),%xmm1
  400c37:	65 00 
  400c39:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400c3d:	f2 0f 10 14 fd 90 20 	movsd  0x602090(,%rdi,8),%xmm2
  400c44:	60 00 
  400c46:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400c4a:	f2 0f 58 0c fd 90 0e 	addsd  0x630e90(,%rdi,8),%xmm1
  400c51:	63 00 
  400c53:	f2 0f 11 0c fd 90 0e 	movsd  %xmm1,0x630e90(,%rdi,8)
  400c5a:	63 00 
  400c5c:	f2 0f 10 1c fd 90 fc 	movsd  0x65fc90(,%rdi,8),%xmm3
  400c63:	65 00 
  400c65:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400c69:	f2 0f 5e da          	divsd  %xmm2,%xmm3
  400c6d:	f2 0f 58 1c fd 90 08 	addsd  0x640890(,%rdi,8),%xmm3
  400c74:	64 00 
  400c76:	f2 0f 11 1c fd 90 08 	movsd  %xmm3,0x640890(,%rdi,8)
  400c7d:	64 00 
  400c7f:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400c83:	f2 0f 58 0c fd 90 1a 	addsd  0x611a90(,%rdi,8),%xmm1
  400c8a:	61 00 
  400c8c:	f2 0f 11 0c fd 90 1a 	movsd  %xmm1,0x611a90(,%rdi,8)
  400c93:	61 00 
  400c95:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  400c99:	f2 0f 58 1c fd 90 14 	addsd  0x621490(,%rdi,8),%xmm3
  400ca0:	62 00 
  400ca2:	f2 0f 11 1c fd 90 14 	movsd  %xmm3,0x621490(,%rdi,8)
  400ca9:	62 00 
  400cab:	c3                   	retq   
  400cac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cb0 <_Z13instantiationv>:
  400cb0:	53                   	push   %rbx
  400cb1:	48 83 ec 30          	sub    $0x30,%rsp
  400cb5:	48 c7 c3 00 06 ff ff 	mov    $0xffffffffffff0600,%rbx
  400cbc:	0f 1f 40 00          	nopl   0x0(%rax)
  400cc0:	e8 ab fa ff ff       	callq  400770 <rand@plt>
  400cc5:	0f 57 c9             	xorps  %xmm1,%xmm1
  400cc8:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400ccc:	f2 0f 10 05 14 07 00 	movsd  0x714(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x28>
  400cd3:	00 
  400cd4:	f2 0f 5e c8          	divsd  %xmm0,%xmm1
  400cd8:	f2 0f 10 05 10 07 00 	movsd  0x710(%rip),%xmm0        # 4013f0 <_IO_stdin_used+0x30>
  400cdf:	00 
  400ce0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400ce4:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400ce8:	f2 0f 10 05 08 07 00 	movsd  0x708(%rip),%xmm0        # 4013f8 <_IO_stdin_used+0x38>
  400cef:	00 
  400cf0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400cf4:	f2 0f 11 4c 24 28    	movsd  %xmm1,0x28(%rsp)
  400cfa:	e8 71 fa ff ff       	callq  400770 <rand@plt>
  400cff:	0f 57 c9             	xorps  %xmm1,%xmm1
  400d02:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400d06:	f2 0f 5e 0d da 06 00 	divsd  0x6da(%rip),%xmm1        # 4013e8 <_IO_stdin_used+0x28>
  400d0d:	00 
  400d0e:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  400d12:	f2 0f 10 05 be 06 00 	movsd  0x6be(%rip),%xmm0        # 4013d8 <_IO_stdin_used+0x18>
  400d19:	00 
  400d1a:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400d1e:	f2 0f 11 4c 24 20    	movsd  %xmm1,0x20(%rsp)
  400d24:	e8 47 fa ff ff       	callq  400770 <rand@plt>
  400d29:	0f 57 c0             	xorps  %xmm0,%xmm0
  400d2c:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400d30:	f2 0f 5e 05 b0 06 00 	divsd  0x6b0(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x28>
  400d37:	00 
  400d38:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400d3c:	f2 0f 58 05 94 06 00 	addsd  0x694(%rip),%xmm0        # 4013d8 <_IO_stdin_used+0x18>
  400d43:	00 
  400d44:	f2 0f 11 44 24 18    	movsd  %xmm0,0x18(%rsp)
  400d4a:	e8 21 fa ff ff       	callq  400770 <rand@plt>
  400d4f:	0f 57 c9             	xorps  %xmm1,%xmm1
  400d52:	f2 0f 2a c8          	cvtsi2sd %eax,%xmm1
  400d56:	f2 0f 5e 0d 8a 06 00 	divsd  0x68a(%rip),%xmm1        # 4013e8 <_IO_stdin_used+0x28>
  400d5d:	00 
  400d5e:	f2 0f 10 05 9a 06 00 	movsd  0x69a(%rip),%xmm0        # 401400 <_IO_stdin_used+0x40>
  400d65:	00 
  400d66:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400d6a:	f2 0f 5e 0d 86 06 00 	divsd  0x686(%rip),%xmm1        # 4013f8 <_IO_stdin_used+0x38>
  400d71:	00 
  400d72:	f2 0f 11 4c 24 10    	movsd  %xmm1,0x10(%rsp)
  400d78:	e8 f3 f9 ff ff       	callq  400770 <rand@plt>
  400d7d:	0f 57 c0             	xorps  %xmm0,%xmm0
  400d80:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400d84:	f2 0f 5e 05 5c 06 00 	divsd  0x65c(%rip),%xmm0        # 4013e8 <_IO_stdin_used+0x28>
  400d8b:	00 
  400d8c:	f2 0f 58 05 6c 06 00 	addsd  0x66c(%rip),%xmm0        # 401400 <_IO_stdin_used+0x40>
  400d93:	00 
  400d94:	f2 0f 5e 05 5c 06 00 	divsd  0x65c(%rip),%xmm0        # 4013f8 <_IO_stdin_used+0x38>
  400d9b:	00 
  400d9c:	f2 0f 10 4c 24 28    	movsd  0x28(%rsp),%xmm1
  400da2:	f2 0f 11 8b 90 1a 61 	movsd  %xmm1,0x611a90(%rbx)
  400da9:	00 
  400daa:	f2 0f 10 4c 24 20    	movsd  0x20(%rsp),%xmm1
  400db0:	f2 0f 11 8b 90 14 62 	movsd  %xmm1,0x621490(%rbx)
  400db7:	00 
  400db8:	f2 0f 10 4c 24 18    	movsd  0x18(%rsp),%xmm1
  400dbe:	f2 0f 11 8b 90 0e 63 	movsd  %xmm1,0x630e90(%rbx)
  400dc5:	00 
  400dc6:	f2 0f 10 4c 24 10    	movsd  0x10(%rsp),%xmm1
  400dcc:	f2 0f 11 8b 90 08 64 	movsd  %xmm1,0x640890(%rbx)
  400dd3:	00 
  400dd4:	f2 0f 11 83 90 02 65 	movsd  %xmm0,0x650290(%rbx)
  400ddb:	00 
  400ddc:	48 c7 83 90 fc 65 00 	movq   $0x0,0x65fc90(%rbx)
  400de3:	00 00 00 00 
  400de7:	48 c7 83 90 f6 66 00 	movq   $0x0,0x66f690(%rbx)
  400dee:	00 00 00 00 
  400df2:	48 83 c3 08          	add    $0x8,%rbx
  400df6:	0f 85 c4 fe ff ff    	jne    400cc0 <_Z13instantiationv+0x10>
  400dfc:	48 83 c4 30          	add    $0x30,%rsp
  400e00:	5b                   	pop    %rbx
  400e01:	c3                   	retq   
  400e02:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e09:	1f 84 00 00 00 00 00 

0000000000400e10 <_Z14run_simulationv>:
  400e10:	41 57                	push   %r15
  400e12:	41 56                	push   %r14
  400e14:	53                   	push   %rbx
  400e15:	48 83 ec 20          	sub    $0x20,%rsp
  400e19:	45 31 f6             	xor    %r14d,%r14d
  400e1c:	0f 1f 40 00          	nopl   0x0(%rax)
  400e20:	bf 90 02 65 00       	mov    $0x650290,%edi
  400e25:	31 f6                	xor    %esi,%esi
  400e27:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400e2c:	e8 df f8 ff ff       	callq  400710 <memset@plt>
  400e31:	bf 90 fc 65 00       	mov    $0x65fc90,%edi
  400e36:	31 f6                	xor    %esi,%esi
  400e38:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400e3d:	e8 ce f8 ff ff       	callq  400710 <memset@plt>
  400e42:	45 31 ff             	xor    %r15d,%r15d
  400e45:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400e4c:	00 00 00 00 
  400e50:	31 db                	xor    %ebx,%ebx
  400e52:	f2 0f 10 1d 6e 05 00 	movsd  0x56e(%rip),%xmm3        # 4013c8 <_IO_stdin_used+0x8>
  400e59:	00 
  400e5a:	f2 0f 10 25 6e 05 00 	movsd  0x56e(%rip),%xmm4        # 4013d0 <_IO_stdin_used+0x10>
  400e61:	00 
  400e62:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e69:	1f 84 00 00 00 00 00 
  400e70:	49 39 df             	cmp    %rbx,%r15
  400e73:	0f 84 d3 00 00 00    	je     400f4c <_Z14run_simulationv+0x13c>
  400e79:	f2 42 0f 10 34 fd 90 	movsd  0x611a90(,%r15,8),%xmm6
  400e80:	1a 61 00 
  400e83:	f2 0f 5c 34 dd 90 1a 	subsd  0x611a90(,%rbx,8),%xmm6
  400e8a:	61 00 
  400e8c:	f2 42 0f 10 2c fd 90 	movsd  0x621490(,%r15,8),%xmm5
  400e93:	14 62 00 
  400e96:	f2 0f 5c 2c dd 90 14 	subsd  0x621490(,%rbx,8),%xmm5
  400e9d:	62 00 
  400e9f:	66 0f 28 c6          	movapd %xmm6,%xmm0
  400ea3:	f2 0f 59 c0          	mulsd  %xmm0,%xmm0
  400ea7:	66 0f 28 cd          	movapd %xmm5,%xmm1
  400eab:	f2 0f 59 c9          	mulsd  %xmm1,%xmm1
  400eaf:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
  400eb3:	0f 57 c0             	xorps  %xmm0,%xmm0
  400eb6:	f2 0f 51 c1          	sqrtsd %xmm1,%xmm0
  400eba:	66 0f 2e c0          	ucomisd %xmm0,%xmm0
  400ebe:	7b 31                	jnp    400ef1 <_Z14run_simulationv+0xe1>
  400ec0:	66 0f 28 c1          	movapd %xmm1,%xmm0
  400ec4:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400eca:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400ed0:	e8 ab f8 ff ff       	callq  400780 <sqrt@plt>
  400ed5:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400edb:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400ee1:	f2 0f 10 25 e7 04 00 	movsd  0x4e7(%rip),%xmm4        # 4013d0 <_IO_stdin_used+0x10>
  400ee8:	00 
  400ee9:	f2 0f 10 1d d7 04 00 	movsd  0x4d7(%rip),%xmm3        # 4013c8 <_IO_stdin_used+0x8>
  400ef0:	00 
  400ef1:	f2 0f 10 0c dd 90 20 	movsd  0x602090(,%rbx,8),%xmm1
  400ef8:	60 00 
  400efa:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400efe:	f2 42 0f 59 0c fd 90 	mulsd  0x602090(,%r15,8),%xmm1
  400f05:	20 60 00 
  400f08:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400f0c:	f2 0f 59 d2          	mulsd  %xmm2,%xmm2
  400f10:	f2 0f 58 d4          	addsd  %xmm4,%xmm2
  400f14:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400f18:	f2 0f 59 f1          	mulsd  %xmm1,%xmm6
  400f1c:	f2 0f 5e f0          	divsd  %xmm0,%xmm6
  400f20:	f2 0f 58 34 dd 90 02 	addsd  0x650290(,%rbx,8),%xmm6
  400f27:	65 00 
  400f29:	f2 0f 11 34 dd 90 02 	movsd  %xmm6,0x650290(,%rbx,8)
  400f30:	65 00 
  400f32:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  400f36:	f2 0f 5e e8          	divsd  %xmm0,%xmm5
  400f3a:	f2 0f 58 2c dd 90 fc 	addsd  0x65fc90(,%rbx,8),%xmm5
  400f41:	65 00 
  400f43:	f2 0f 11 2c dd 90 fc 	movsd  %xmm5,0x65fc90(,%rbx,8)
  400f4a:	65 00 
  400f4c:	48 ff c3             	inc    %rbx
  400f4f:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  400f56:	0f 85 14 ff ff ff    	jne    400e70 <_Z14run_simulationv+0x60>
  400f5c:	49 ff c7             	inc    %r15
  400f5f:	48 c7 c0 00 06 ff ff 	mov    $0xffffffffffff0600,%rax
  400f66:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400f6d:	0f 85 dd fe ff ff    	jne    400e50 <_Z14run_simulationv+0x40>
  400f73:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400f7a:	84 00 00 00 00 00 
  400f80:	f2 0f 10 88 90 fc 65 	movsd  0x65fc90(%rax),%xmm1
  400f87:	00 
  400f88:	f2 0f 10 05 60 04 00 	movsd  0x460(%rip),%xmm0        # 4013f0 <_IO_stdin_used+0x30>
  400f8f:	00 
  400f90:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400f94:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400f98:	f2 0f 10 90 90 1a 61 	movsd  0x611a90(%rax),%xmm2
  400f9f:	00 
  400fa0:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400fa4:	f2 0f 58 88 90 08 64 	addsd  0x640890(%rax),%xmm1
  400fab:	00 
  400fac:	f2 0f 11 88 90 08 64 	movsd  %xmm1,0x640890(%rax)
  400fb3:	00 
  400fb4:	f2 0f 10 80 90 f6 66 	movsd  0x66f690(%rax),%xmm0
  400fbb:	00 
  400fbc:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400fc0:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400fc4:	f2 0f 58 80 90 02 65 	addsd  0x650290(%rax),%xmm0
  400fcb:	00 
  400fcc:	f2 0f 11 80 90 02 65 	movsd  %xmm0,0x650290(%rax)
  400fd3:	00 
  400fd4:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400fd8:	f2 0f 59 dc          	mulsd  %xmm4,%xmm3
  400fdc:	f2 0f 58 98 90 14 62 	addsd  0x621490(%rax),%xmm3
  400fe3:	00 
  400fe4:	f2 0f 11 98 90 14 62 	movsd  %xmm3,0x621490(%rax)
  400feb:	00 
  400fec:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400ff0:	f2 0f 59 d4          	mulsd  %xmm4,%xmm2
  400ff4:	f2 0f 58 90 90 0e 63 	addsd  0x630e90(%rax),%xmm2
  400ffb:	00 
  400ffc:	f2 0f 11 90 90 0e 63 	movsd  %xmm2,0x630e90(%rax)
  401003:	00 
  401004:	f2 0f 10 25 cc 03 00 	movsd  0x3cc(%rip),%xmm4        # 4013d8 <_IO_stdin_used+0x18>
  40100b:	00 
  40100c:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  401010:	77 0a                	ja     40101c <_Z14run_simulationv+0x20c>
  401012:	66 0f 2e 1d c6 03 00 	ucomisd 0x3c6(%rip),%xmm3        # 4013e0 <_IO_stdin_used+0x20>
  401019:	00 
  40101a:	76 10                	jbe    40102c <_Z14run_simulationv+0x21c>
  40101c:	66 0f 57 0d ec 03 00 	xorpd  0x3ec(%rip),%xmm1        # 401410 <_IO_stdin_used+0x50>
  401023:	00 
  401024:	66 0f 13 88 90 08 64 	movlpd %xmm1,0x640890(%rax)
  40102b:	00 
  40102c:	f2 0f 10 0d a4 03 00 	movsd  0x3a4(%rip),%xmm1        # 4013d8 <_IO_stdin_used+0x18>
  401033:	00 
  401034:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  401038:	77 0a                	ja     401044 <_Z14run_simulationv+0x234>
  40103a:	66 0f 2e 15 9e 03 00 	ucomisd 0x39e(%rip),%xmm2        # 4013e0 <_IO_stdin_used+0x20>
  401041:	00 
  401042:	76 10                	jbe    401054 <_Z14run_simulationv+0x244>
  401044:	66 0f 57 05 c4 03 00 	xorpd  0x3c4(%rip),%xmm0        # 401410 <_IO_stdin_used+0x50>
  40104b:	00 
  40104c:	66 0f 13 80 90 02 65 	movlpd %xmm0,0x650290(%rax)
  401053:	00 
  401054:	48 83 c0 08          	add    $0x8,%rax
  401058:	0f 85 22 ff ff ff    	jne    400f80 <_Z14run_simulationv+0x170>
  40105e:	41 ff c6             	inc    %r14d
  401061:	41 83 fe 05          	cmp    $0x5,%r14d
  401065:	0f 85 b5 fd ff ff    	jne    400e20 <_Z14run_simulationv+0x10>
  40106b:	48 83 c4 20          	add    $0x20,%rsp
  40106f:	5b                   	pop    %rbx
  401070:	41 5e                	pop    %r14
  401072:	41 5f                	pop    %r15
  401074:	c3                   	retq   
  401075:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40107c:	00 00 00 00 

0000000000401080 <_Z10run_simplev>:
  401080:	31 c0                	xor    %eax,%eax
  401082:	66 0f 28 05 96 03 00 	movapd 0x396(%rip),%xmm0        # 401420 <_IO_stdin_used+0x60>
  401089:	00 
  40108a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401090:	31 c9                	xor    %ecx,%ecx
  401092:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401099:	1f 84 00 00 00 00 00 
  4010a0:	66 0f 28 89 90 02 65 	movapd 0x650290(%rcx),%xmm1
  4010a7:	00 
  4010a8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4010ac:	66 0f 28 91 90 20 60 	movapd 0x602090(%rcx),%xmm2
  4010b3:	00 
  4010b4:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  4010b8:	66 0f 58 89 90 0e 63 	addpd  0x630e90(%rcx),%xmm1
  4010bf:	00 
  4010c0:	66 0f 29 89 90 0e 63 	movapd %xmm1,0x630e90(%rcx)
  4010c7:	00 
  4010c8:	66 0f 28 99 90 fc 65 	movapd 0x65fc90(%rcx),%xmm3
  4010cf:	00 
  4010d0:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  4010d4:	66 0f 5e da          	divpd  %xmm2,%xmm3
  4010d8:	66 0f 58 99 90 08 64 	addpd  0x640890(%rcx),%xmm3
  4010df:	00 
  4010e0:	66 0f 29 99 90 08 64 	movapd %xmm3,0x640890(%rcx)
  4010e7:	00 
  4010e8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4010ec:	66 0f 58 89 90 1a 61 	addpd  0x611a90(%rcx),%xmm1
  4010f3:	00 
  4010f4:	66 0f 29 89 90 1a 61 	movapd %xmm1,0x611a90(%rcx)
  4010fb:	00 
  4010fc:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  401100:	66 0f 58 99 90 14 62 	addpd  0x621490(%rcx),%xmm3
  401107:	00 
  401108:	66 0f 29 99 90 14 62 	movapd %xmm3,0x621490(%rcx)
  40110f:	00 
  401110:	48 83 c1 10          	add    $0x10,%rcx
  401114:	48 81 f9 00 fa 00 00 	cmp    $0xfa00,%rcx
  40111b:	75 83                	jne    4010a0 <_Z10run_simplev+0x20>
  40111d:	ff c0                	inc    %eax
  40111f:	3d 50 c3 00 00       	cmp    $0xc350,%eax
  401124:	0f 85 66 ff ff ff    	jne    401090 <_Z10run_simplev+0x10>
  40112a:	c3                   	retq   
  40112b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401130 <main>:
  401130:	41 57                	push   %r15
  401132:	41 56                	push   %r14
  401134:	41 55                	push   %r13
  401136:	41 54                	push   %r12
  401138:	53                   	push   %rbx
  401139:	e8 22 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40113e:	48 89 c3             	mov    %rax,%rbx
  401141:	e8 6a fb ff ff       	callq  400cb0 <_Z13instantiationv>
  401146:	e8 15 f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40114b:	49 89 c6             	mov    %rax,%r14
  40114e:	49 29 de             	sub    %rbx,%r14
  401151:	e8 0a f6 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401156:	48 89 c3             	mov    %rax,%rbx
  401159:	e8 b2 fc ff ff       	callq  400e10 <_Z14run_simulationv>
  40115e:	e8 fd f5 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401163:	49 89 c4             	mov    %rax,%r12
  401166:	49 29 dc             	sub    %rbx,%r12
  401169:	31 db                	xor    %ebx,%ebx
  40116b:	e8 f0 f5 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401170:	49 89 c7             	mov    %rax,%r15
  401173:	66 0f 28 05 a5 02 00 	movapd 0x2a5(%rip),%xmm0        # 401420 <_IO_stdin_used+0x60>
  40117a:	00 
  40117b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  401180:	31 c0                	xor    %eax,%eax
  401182:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401189:	1f 84 00 00 00 00 00 
  401190:	66 0f 28 88 90 02 65 	movapd 0x650290(%rax),%xmm1
  401197:	00 
  401198:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  40119c:	66 0f 28 90 90 20 60 	movapd 0x602090(%rax),%xmm2
  4011a3:	00 
  4011a4:	66 0f 5e ca          	divpd  %xmm2,%xmm1
  4011a8:	66 0f 58 88 90 0e 63 	addpd  0x630e90(%rax),%xmm1
  4011af:	00 
  4011b0:	66 0f 29 88 90 0e 63 	movapd %xmm1,0x630e90(%rax)
  4011b7:	00 
  4011b8:	66 0f 28 98 90 fc 65 	movapd 0x65fc90(%rax),%xmm3
  4011bf:	00 
  4011c0:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  4011c4:	66 0f 5e da          	divpd  %xmm2,%xmm3
  4011c8:	66 0f 58 98 90 08 64 	addpd  0x640890(%rax),%xmm3
  4011cf:	00 
  4011d0:	66 0f 29 98 90 08 64 	movapd %xmm3,0x640890(%rax)
  4011d7:	00 
  4011d8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
  4011dc:	66 0f 58 88 90 1a 61 	addpd  0x611a90(%rax),%xmm1
  4011e3:	00 
  4011e4:	66 0f 29 88 90 1a 61 	movapd %xmm1,0x611a90(%rax)
  4011eb:	00 
  4011ec:	66 0f 59 d8          	mulpd  %xmm0,%xmm3
  4011f0:	66 0f 58 98 90 14 62 	addpd  0x621490(%rax),%xmm3
  4011f7:	00 
  4011f8:	66 0f 29 98 90 14 62 	movapd %xmm3,0x621490(%rax)
  4011ff:	00 
  401200:	48 83 c0 10          	add    $0x10,%rax
  401204:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  40120a:	75 84                	jne    401190 <main+0x60>
  40120c:	ff c3                	inc    %ebx
  40120e:	81 fb 50 c3 00 00    	cmp    $0xc350,%ebx
  401214:	0f 85 66 ff ff ff    	jne    401180 <main+0x50>
  40121a:	49 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%r13
  401221:	de 1b 43 
  401224:	4c 89 f0             	mov    %r14,%rax
  401227:	49 f7 ed             	imul   %r13
  40122a:	49 89 d6             	mov    %rdx,%r14
  40122d:	4c 89 f0             	mov    %r14,%rax
  401230:	48 c1 e8 3f          	shr    $0x3f,%rax
  401234:	49 c1 fe 12          	sar    $0x12,%r14
  401238:	49 01 c6             	add    %rax,%r14
  40123b:	4c 89 e0             	mov    %r12,%rax
  40123e:	49 f7 ed             	imul   %r13
  401241:	48 89 d3             	mov    %rdx,%rbx
  401244:	48 89 d8             	mov    %rbx,%rax
  401247:	48 c1 e8 3f          	shr    $0x3f,%rax
  40124b:	48 c1 fb 12          	sar    $0x12,%rbx
  40124f:	48 01 c3             	add    %rax,%rbx
  401252:	e8 09 f5 ff ff       	callq  400760 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401257:	4c 29 f8             	sub    %r15,%rax
  40125a:	41 b8 0b 00 00 00    	mov    $0xb,%r8d
  401260:	48 c7 c1 00 06 ff ff 	mov    $0xffffffffffff0600,%rcx
  401267:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40126e:	00 00 
  401270:	f2 0f 10 81 90 14 62 	movsd  0x621490(%rcx),%xmm0
  401277:	00 
  401278:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  40127c:	66 0f 7e c2          	movd   %xmm0,%edx
  401280:	44 01 c2             	add    %r8d,%edx
  401283:	f2 0f 10 81 90 0e 63 	movsd  0x630e90(%rcx),%xmm0
  40128a:	00 
  40128b:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  40128f:	66 0f 7e c6          	movd   %xmm0,%esi
  401293:	01 d6                	add    %edx,%esi
  401295:	48 63 d6             	movslq %esi,%rdx
  401298:	48 69 f2 8d 68 b7 6c 	imul   $0x6cb7688d,%rdx,%rsi
  40129f:	48 89 f7             	mov    %rsi,%rdi
  4012a2:	48 c1 ef 3f          	shr    $0x3f,%rdi
  4012a6:	48 c1 fe 33          	sar    $0x33,%rsi
  4012aa:	01 fe                	add    %edi,%esi
  4012ac:	69 f6 87 d6 12 00    	imul   $0x12d687,%esi,%esi
  4012b2:	29 f2                	sub    %esi,%edx
  4012b4:	f2 0f 10 81 98 14 62 	movsd  0x621498(%rcx),%xmm0
  4012bb:	00 
  4012bc:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  4012c0:	66 0f 7e c6          	movd   %xmm0,%esi
  4012c4:	01 d6                	add    %edx,%esi
  4012c6:	f2 0f 10 81 98 0e 63 	movsd  0x630e98(%rcx),%xmm0
  4012cd:	00 
  4012ce:	f2 0f 5a c0          	cvtsd2ss %xmm0,%xmm0
  4012d2:	66 0f 7e c2          	movd   %xmm0,%edx
  4012d6:	01 f2                	add    %esi,%edx
  4012d8:	4c 63 c2             	movslq %edx,%r8
  4012db:	49 69 d0 8d 68 b7 6c 	imul   $0x6cb7688d,%r8,%rdx
  4012e2:	48 89 d6             	mov    %rdx,%rsi
  4012e5:	48 c1 ee 3f          	shr    $0x3f,%rsi
  4012e9:	48 c1 fa 33          	sar    $0x33,%rdx
  4012ed:	01 f2                	add    %esi,%edx
  4012ef:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  4012f5:	41 29 d0             	sub    %edx,%r8d
  4012f8:	48 83 c1 10          	add    $0x10,%rcx
  4012fc:	0f 85 6e ff ff ff    	jne    401270 <main+0x140>
  401302:	49 f7 ed             	imul   %r13
  401305:	48 89 d0             	mov    %rdx,%rax
  401308:	48 c1 e8 3f          	shr    $0x3f,%rax
  40130c:	48 c1 fa 12          	sar    $0x12,%rdx
  401310:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401314:	bf 30 14 40 00       	mov    $0x401430,%edi
  401319:	31 c0                	xor    %eax,%eax
  40131b:	4c 89 f6             	mov    %r14,%rsi
  40131e:	48 89 da             	mov    %rbx,%rdx
  401321:	e8 da f3 ff ff       	callq  400700 <printf@plt>
  401326:	31 c0                	xor    %eax,%eax
  401328:	5b                   	pop    %rbx
  401329:	41 5c                	pop    %r12
  40132b:	41 5d                	pop    %r13
  40132d:	41 5e                	pop    %r14
  40132f:	41 5f                	pop    %r15
  401331:	c3                   	retq   
  401332:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401339:	00 00 00 
  40133c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401340 <__libc_csu_init>:
  401340:	41 57                	push   %r15
  401342:	41 56                	push   %r14
  401344:	41 89 ff             	mov    %edi,%r15d
  401347:	41 55                	push   %r13
  401349:	41 54                	push   %r12
  40134b:	4c 8d 25 86 0a 20 00 	lea    0x200a86(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  401352:	55                   	push   %rbp
  401353:	48 8d 2d 8e 0a 20 00 	lea    0x200a8e(%rip),%rbp        # 601de8 <__init_array_end>
  40135a:	53                   	push   %rbx
  40135b:	49 89 f6             	mov    %rsi,%r14
  40135e:	49 89 d5             	mov    %rdx,%r13
  401361:	4c 29 e5             	sub    %r12,%rbp
  401364:	48 83 ec 08          	sub    $0x8,%rsp
  401368:	48 c1 fd 03          	sar    $0x3,%rbp
  40136c:	e8 5f f3 ff ff       	callq  4006d0 <_init>
  401371:	48 85 ed             	test   %rbp,%rbp
  401374:	74 20                	je     401396 <__libc_csu_init+0x56>
  401376:	31 db                	xor    %ebx,%ebx
  401378:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40137f:	00 
  401380:	4c 89 ea             	mov    %r13,%rdx
  401383:	4c 89 f6             	mov    %r14,%rsi
  401386:	44 89 ff             	mov    %r15d,%edi
  401389:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40138d:	48 83 c3 01          	add    $0x1,%rbx
  401391:	48 39 eb             	cmp    %rbp,%rbx
  401394:	75 ea                	jne    401380 <__libc_csu_init+0x40>
  401396:	48 83 c4 08          	add    $0x8,%rsp
  40139a:	5b                   	pop    %rbx
  40139b:	5d                   	pop    %rbp
  40139c:	41 5c                	pop    %r12
  40139e:	41 5d                	pop    %r13
  4013a0:	41 5e                	pop    %r14
  4013a2:	41 5f                	pop    %r15
  4013a4:	c3                   	retq   
  4013a5:	90                   	nop
  4013a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4013ad:	00 00 00 

00000000004013b0 <__libc_csu_fini>:
  4013b0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004013b4 <_fini>:
  4013b4:	48 83 ec 08          	sub    $0x8,%rsp
  4013b8:	48 83 c4 08          	add    $0x8,%rsp
  4013bc:	c3                   	retq   
