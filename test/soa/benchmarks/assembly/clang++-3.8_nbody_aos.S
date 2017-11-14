
bin/clang++-3.8_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

0000000000400670 <_init>:
  400670:	48 83 ec 08          	sub    $0x8,%rsp
  400674:	48 8b 05 7d 19 20 00 	mov    0x20197d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40067b:	48 85 c0             	test   %rax,%rax
  40067e:	74 05                	je     400685 <_init+0x15>
  400680:	e8 9b 00 00 00       	callq  400720 <rand@plt+0x10>
  400685:	48 83 c4 08          	add    $0x8,%rsp
  400689:	c3                   	retq   

Disassembly of section .plt:

0000000000400690 <printf@plt-0x10>:
  400690:	ff 35 72 19 20 00    	pushq  0x201972(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400696:	ff 25 74 19 20 00    	jmpq   *0x201974(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40069c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006a0 <printf@plt>:
  4006a0:	ff 25 72 19 20 00    	jmpq   *0x201972(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006a6:	68 00 00 00 00       	pushq  $0x0
  4006ab:	e9 e0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006b0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006b0:	ff 25 6a 19 20 00    	jmpq   *0x20196a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006b6:	68 01 00 00 00       	pushq  $0x1
  4006bb:	e9 d0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006c0 <__libc_start_main@plt>:
  4006c0:	ff 25 62 19 20 00    	jmpq   *0x201962(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006c6:	68 02 00 00 00       	pushq  $0x2
  4006cb:	e9 c0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006d0 <__cxa_atexit@plt>:
  4006d0:	ff 25 5a 19 20 00    	jmpq   *0x20195a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006d6:	68 03 00 00 00       	pushq  $0x3
  4006db:	e9 b0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006e0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4006e0:	ff 25 52 19 20 00    	jmpq   *0x201952(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  4006e6:	68 04 00 00 00       	pushq  $0x4
  4006eb:	e9 a0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006f0 <srand@plt>:
  4006f0:	ff 25 4a 19 20 00    	jmpq   *0x20194a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4006f6:	68 05 00 00 00       	pushq  $0x5
  4006fb:	e9 90 ff ff ff       	jmpq   400690 <_init+0x20>

0000000000400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400700:	ff 25 42 19 20 00    	jmpq   *0x201942(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400706:	68 06 00 00 00       	pushq  $0x6
  40070b:	e9 80 ff ff ff       	jmpq   400690 <_init+0x20>

0000000000400710 <rand@plt>:
  400710:	ff 25 3a 19 20 00    	jmpq   *0x20193a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400716:	68 07 00 00 00       	pushq  $0x7
  40071b:	e9 70 ff ff ff       	jmpq   400690 <_init+0x20>

Disassembly of section .plt.got:

0000000000400720 <.plt.got>:
  400720:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400726:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400730 <set_fast_math>:
  400730:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  400735:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  40073c:	00 
  40073d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  400742:	c3                   	retq   
  400743:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40074a:	00 00 00 
  40074d:	0f 1f 00             	nopl   (%rax)

0000000000400750 <_GLOBAL__sub_I_aos.cc>:
  400750:	50                   	push   %rax
  400751:	bf 80 20 60 00       	mov    $0x602080,%edi
  400756:	e8 55 ff ff ff       	callq  4006b0 <_ZNSt8ios_base4InitC1Ev@plt>
  40075b:	bf e0 06 40 00       	mov    $0x4006e0,%edi
  400760:	be 80 20 60 00       	mov    $0x602080,%esi
  400765:	ba 60 20 60 00       	mov    $0x602060,%edx
  40076a:	58                   	pop    %rax
  40076b:	e9 60 ff ff ff       	jmpq   4006d0 <__cxa_atexit@plt>

0000000000400770 <_start>:
  400770:	31 ed                	xor    %ebp,%ebp
  400772:	49 89 d1             	mov    %rdx,%r9
  400775:	5e                   	pop    %rsi
  400776:	48 89 e2             	mov    %rsp,%rdx
  400779:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40077d:	50                   	push   %rax
  40077e:	54                   	push   %rsp
  40077f:	49 c7 c0 10 0f 40 00 	mov    $0x400f10,%r8
  400786:	48 c7 c1 a0 0e 40 00 	mov    $0x400ea0,%rcx
  40078d:	48 c7 c7 70 0c 40 00 	mov    $0x400c70,%rdi
  400794:	e8 27 ff ff ff       	callq  4006c0 <__libc_start_main@plt>
  400799:	f4                   	hlt    
  40079a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007a0 <deregister_tm_clones>:
  4007a0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4007a5:	55                   	push   %rbp
  4007a6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4007ac:	48 83 f8 0e          	cmp    $0xe,%rax
  4007b0:	48 89 e5             	mov    %rsp,%rbp
  4007b3:	76 1b                	jbe    4007d0 <deregister_tm_clones+0x30>
  4007b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4007ba:	48 85 c0             	test   %rax,%rax
  4007bd:	74 11                	je     4007d0 <deregister_tm_clones+0x30>
  4007bf:	5d                   	pop    %rbp
  4007c0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4007c5:	ff e0                	jmpq   *%rax
  4007c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ce:	00 00 
  4007d0:	5d                   	pop    %rbp
  4007d1:	c3                   	retq   
  4007d2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007dd:	00 00 00 

00000000004007e0 <register_tm_clones>:
  4007e0:	be 70 20 60 00       	mov    $0x602070,%esi
  4007e5:	55                   	push   %rbp
  4007e6:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  4007ed:	48 c1 fe 03          	sar    $0x3,%rsi
  4007f1:	48 89 e5             	mov    %rsp,%rbp
  4007f4:	48 89 f0             	mov    %rsi,%rax
  4007f7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4007fb:	48 01 c6             	add    %rax,%rsi
  4007fe:	48 d1 fe             	sar    %rsi
  400801:	74 15                	je     400818 <register_tm_clones+0x38>
  400803:	b8 00 00 00 00       	mov    $0x0,%eax
  400808:	48 85 c0             	test   %rax,%rax
  40080b:	74 0b                	je     400818 <register_tm_clones+0x38>
  40080d:	5d                   	pop    %rbp
  40080e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400813:	ff e0                	jmpq   *%rax
  400815:	0f 1f 00             	nopl   (%rax)
  400818:	5d                   	pop    %rbp
  400819:	c3                   	retq   
  40081a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400820 <__do_global_dtors_aux>:
  400820:	80 3d 49 18 20 00 00 	cmpb   $0x0,0x201849(%rip)        # 602070 <__TMC_END__>
  400827:	75 11                	jne    40083a <__do_global_dtors_aux+0x1a>
  400829:	55                   	push   %rbp
  40082a:	48 89 e5             	mov    %rsp,%rbp
  40082d:	e8 6e ff ff ff       	callq  4007a0 <deregister_tm_clones>
  400832:	5d                   	pop    %rbp
  400833:	c6 05 36 18 20 00 01 	movb   $0x1,0x201836(%rip)        # 602070 <__TMC_END__>
  40083a:	f3 c3                	repz retq 
  40083c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400840 <frame_dummy>:
  400840:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400845:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400849:	75 05                	jne    400850 <frame_dummy+0x10>
  40084b:	eb 93                	jmp    4007e0 <register_tm_clones>
  40084d:	0f 1f 00             	nopl   (%rax)
  400850:	b8 00 00 00 00       	mov    $0x0,%eax
  400855:	48 85 c0             	test   %rax,%rax
  400858:	74 f1                	je     40084b <frame_dummy+0xb>
  40085a:	55                   	push   %rbp
  40085b:	48 89 e5             	mov    %rsp,%rbp
  40085e:	ff d0                	callq  *%rax
  400860:	5d                   	pop    %rbp
  400861:	e9 7a ff ff ff       	jmpq   4007e0 <register_tm_clones>
  400866:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40086d:	00 00 00 

0000000000400870 <_Z11r_float2intf>:
  400870:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  400874:	c3                   	retq   
  400875:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40087c:	00 00 00 00 

0000000000400880 <_Z13instantiationv>:
  400880:	55                   	push   %rbp
  400881:	41 57                	push   %r15
  400883:	41 56                	push   %r14
  400885:	53                   	push   %rbx
  400886:	48 83 ec 38          	sub    $0x38,%rsp
  40088a:	bf 2a 00 00 00       	mov    $0x2a,%edi
  40088f:	e8 5c fe ff ff       	callq  4006f0 <srand@plt>
  400894:	bd 40 1f 00 00       	mov    $0x1f40,%ebp
  400899:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4008a0:	c5 f8 77             	vzeroupper 
  4008a3:	e8 68 fe ff ff       	callq  400710 <rand@plt>
  4008a8:	41 89 c6             	mov    %eax,%r14d
  4008ab:	e8 60 fe ff ff       	callq  400710 <rand@plt>
  4008b0:	41 89 c7             	mov    %eax,%r15d
  4008b3:	e8 58 fe ff ff       	callq  400710 <rand@plt>
  4008b8:	89 c3                	mov    %eax,%ebx
  4008ba:	e8 51 fe ff ff       	callq  400710 <rand@plt>
  4008bf:	c4 c1 79 6e c6       	vmovd  %r14d,%xmm0
  4008c4:	c4 c3 79 22 c7 01    	vpinsrd $0x1,%r15d,%xmm0,%xmm0
  4008ca:	c4 e3 79 22 c3 02    	vpinsrd $0x2,%ebx,%xmm0,%xmm0
  4008d0:	c4 e3 79 22 c0 03    	vpinsrd $0x3,%eax,%xmm0,%xmm0
  4008d6:	c5 fe e6 c8          	vcvtdq2pd %xmm0,%ymm1
  4008da:	c5 fd 28 05 7e 06 00 	vmovapd 0x67e(%rip),%ymm0        # 400f60 <_IO_stdin_used+0x40>
  4008e1:	00 
  4008e2:	c4 e2 fd 98 0d 55 06 	vfmadd132pd 0x655(%rip),%ymm0,%ymm1        # 400f40 <_IO_stdin_used+0x20>
  4008e9:	00 00 
  4008eb:	c5 fd 11 0c 24       	vmovupd %ymm1,(%rsp)
  4008f0:	c5 f8 77             	vzeroupper 
  4008f3:	e8 18 fe ff ff       	callq  400710 <rand@plt>
  4008f8:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008fc:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400900:	c5 fb 10 0d 80 06 00 	vmovsd 0x680(%rip),%xmm1        # 400f88 <_IO_stdin_used+0x68>
  400907:	00 
  400908:	c4 e2 f1 99 05 6f 06 	vfmadd132sd 0x66f(%rip),%xmm1,%xmm0        # 400f80 <_IO_stdin_used+0x60>
  40090f:	00 00 
  400911:	48 8b 05 50 17 20 00 	mov    0x201750(%rip),%rax        # 602068 <Body_arena_head>
  400918:	c5 fc 10 0c 24       	vmovups (%rsp),%ymm1
  40091d:	c5 fc 11 08          	vmovups %ymm1,(%rax)
  400921:	c5 fb 11 40 20       	vmovsd %xmm0,0x20(%rax)
  400926:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  40092a:	c5 f9 11 40 28       	vmovupd %xmm0,0x28(%rax)
  40092f:	48 83 c0 38          	add    $0x38,%rax
  400933:	48 89 05 2e 17 20 00 	mov    %rax,0x20172e(%rip)        # 602068 <Body_arena_head>
  40093a:	83 c5 ff             	add    $0xffffffff,%ebp
  40093d:	0f 85 5d ff ff ff    	jne    4008a0 <_Z13instantiationv+0x20>
  400943:	48 83 c4 38          	add    $0x38,%rsp
  400947:	5b                   	pop    %rbx
  400948:	41 5e                	pop    %r14
  40094a:	41 5f                	pop    %r15
  40094c:	5d                   	pop    %rbp
  40094d:	c5 f8 77             	vzeroupper 
  400950:	c3                   	retq   
  400951:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400958:	0f 1f 84 00 00 00 00 
  40095f:	00 

0000000000400960 <_Z14run_simulationv>:
  400960:	45 31 c0             	xor    %r8d,%r8d
  400963:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400967:	c5 f9 28 0d 51 06 00 	vmovapd 0x651(%rip),%xmm1        # 400fc0 <_IO_stdin_used+0xa0>
  40096e:	00 
  40096f:	c5 79 28 05 59 06 00 	vmovapd 0x659(%rip),%xmm8        # 400fd0 <_IO_stdin_used+0xb0>
  400976:	00 
  400977:	c5 fb 10 1d 11 06 00 	vmovsd 0x611(%rip),%xmm3        # 400f90 <_IO_stdin_used+0x70>
  40097e:	00 
  40097f:	c5 fb 10 25 11 06 00 	vmovsd 0x611(%rip),%xmm4        # 400f98 <_IO_stdin_used+0x78>
  400986:	00 
  400987:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40098e:	00 00 
  400990:	31 c9                	xor    %ecx,%ecx
  400992:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400999:	1f 84 00 00 00 00 00 
  4009a0:	c5 f9 11 81 b8 20 60 	vmovupd %xmm0,0x6020b8(%rcx)
  4009a7:	00 
  4009a8:	c5 f9 11 81 f0 20 60 	vmovupd %xmm0,0x6020f0(%rcx)
  4009af:	00 
  4009b0:	c5 f9 11 81 28 21 60 	vmovupd %xmm0,0x602128(%rcx)
  4009b7:	00 
  4009b8:	c5 f9 11 81 60 21 60 	vmovupd %xmm0,0x602160(%rcx)
  4009bf:	00 
  4009c0:	c5 f9 11 81 98 21 60 	vmovupd %xmm0,0x602198(%rcx)
  4009c7:	00 
  4009c8:	c5 f9 11 81 d0 21 60 	vmovupd %xmm0,0x6021d0(%rcx)
  4009cf:	00 
  4009d0:	c5 f9 11 81 08 22 60 	vmovupd %xmm0,0x602208(%rcx)
  4009d7:	00 
  4009d8:	c5 f9 11 81 40 22 60 	vmovupd %xmm0,0x602240(%rcx)
  4009df:	00 
  4009e0:	c5 f9 11 81 78 22 60 	vmovupd %xmm0,0x602278(%rcx)
  4009e7:	00 
  4009e8:	c5 f9 11 81 b0 22 60 	vmovupd %xmm0,0x6022b0(%rcx)
  4009ef:	00 
  4009f0:	c5 f9 11 81 e8 22 60 	vmovupd %xmm0,0x6022e8(%rcx)
  4009f7:	00 
  4009f8:	c5 f9 11 81 20 23 60 	vmovupd %xmm0,0x602320(%rcx)
  4009ff:	00 
  400a00:	c5 f9 11 81 58 23 60 	vmovupd %xmm0,0x602358(%rcx)
  400a07:	00 
  400a08:	c5 f9 11 81 90 23 60 	vmovupd %xmm0,0x602390(%rcx)
  400a0f:	00 
  400a10:	c5 f9 11 81 c8 23 60 	vmovupd %xmm0,0x6023c8(%rcx)
  400a17:	00 
  400a18:	c5 f9 11 81 00 24 60 	vmovupd %xmm0,0x602400(%rcx)
  400a1f:	00 
  400a20:	48 81 c1 80 03 00 00 	add    $0x380,%rcx
  400a27:	b8 00 00 00 00       	mov    $0x0,%eax
  400a2c:	ba 00 00 00 00       	mov    $0x0,%edx
  400a31:	48 81 f9 00 d6 06 00 	cmp    $0x6d600,%rcx
  400a38:	0f 85 62 ff ff ff    	jne    4009a0 <_Z14run_simulationv+0x40>
  400a3e:	66 90                	xchg   %ax,%ax
  400a40:	48 6b f2 38          	imul   $0x38,%rdx,%rsi
  400a44:	48 c7 c7 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rdi
  400a4b:	48 89 c1             	mov    %rax,%rcx
  400a4e:	66 90                	xchg   %ax,%ax
  400a50:	48 85 c9             	test   %rcx,%rcx
  400a53:	74 5b                	je     400ab0 <_Z14run_simulationv+0x150>
  400a55:	c5 f9 10 ae 98 20 60 	vmovupd 0x602098(%rsi),%xmm5
  400a5c:	00 
  400a5d:	c5 d1 5c af 98 f6 66 	vsubpd 0x66f698(%rdi),%xmm5,%xmm5
  400a64:	00 
  400a65:	c5 d3 59 f5          	vmulsd %xmm5,%xmm5,%xmm6
  400a69:	c4 e3 79 05 fd 01    	vpermilpd $0x1,%xmm5,%xmm7
  400a6f:	c4 e2 c1 a9 fe       	vfmadd213sd %xmm6,%xmm7,%xmm7
  400a74:	c5 fb 51 f7          	vsqrtsd %xmm7,%xmm0,%xmm6
  400a78:	c5 e3 59 97 90 f6 66 	vmulsd 0x66f690(%rdi),%xmm3,%xmm2
  400a7f:	00 
  400a80:	c5 eb 59 96 90 20 60 	vmulsd 0x602090(%rsi),%xmm2,%xmm2
  400a87:	00 
  400a88:	c5 c3 58 fc          	vaddsd %xmm4,%xmm7,%xmm7
  400a8c:	c5 eb 5e d7          	vdivsd %xmm7,%xmm2,%xmm2
  400a90:	c5 fb 12 d2          	vmovddup %xmm2,%xmm2
  400a94:	c5 e9 59 d5          	vmulpd %xmm5,%xmm2,%xmm2
  400a98:	c5 fb 12 ee          	vmovddup %xmm6,%xmm5
  400a9c:	c5 e9 5e d5          	vdivpd %xmm5,%xmm2,%xmm2
  400aa0:	c5 e9 58 97 b8 f6 66 	vaddpd 0x66f6b8(%rdi),%xmm2,%xmm2
  400aa7:	00 
  400aa8:	c5 f9 11 97 b8 f6 66 	vmovupd %xmm2,0x66f6b8(%rdi)
  400aaf:	00 
  400ab0:	48 83 c1 c8          	add    $0xffffffffffffffc8,%rcx
  400ab4:	48 83 c7 38          	add    $0x38,%rdi
  400ab8:	75 96                	jne    400a50 <_Z14run_simulationv+0xf0>
  400aba:	48 83 c2 01          	add    $0x1,%rdx
  400abe:	48 83 c0 38          	add    $0x38,%rax
  400ac2:	48 c7 c6 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rsi
  400ac9:	48 81 fa 40 1f 00 00 	cmp    $0x1f40,%rdx
  400ad0:	0f 85 6a ff ff ff    	jne    400a40 <_Z14run_simulationv+0xe0>
  400ad6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400add:	00 00 00 
  400ae0:	c5 f1 59 96 b8 f6 66 	vmulpd 0x66f6b8(%rsi),%xmm1,%xmm2
  400ae7:	00 
  400ae8:	c5 fb 12 ae 90 f6 66 	vmovddup 0x66f690(%rsi),%xmm5
  400aef:	00 
  400af0:	c5 e9 5e d5          	vdivpd %xmm5,%xmm2,%xmm2
  400af4:	c5 e9 58 ae a8 f6 66 	vaddpd 0x66f6a8(%rsi),%xmm2,%xmm5
  400afb:	00 
  400afc:	c5 f9 11 ae a8 f6 66 	vmovupd %xmm5,0x66f6a8(%rsi)
  400b03:	00 
  400b04:	c5 f9 10 b6 98 f6 66 	vmovupd 0x66f698(%rsi),%xmm6
  400b0b:	00 
  400b0c:	c4 e2 d1 b8 f1       	vfmadd231pd %xmm1,%xmm5,%xmm6
  400b11:	c5 f9 11 b6 98 f6 66 	vmovupd %xmm6,0x66f698(%rsi)
  400b18:	00 
  400b19:	c5 f9 2e 35 7f 04 00 	vucomisd 0x47f(%rip),%xmm6        # 400fa0 <_IO_stdin_used+0x80>
  400b20:	00 
  400b21:	72 0a                	jb     400b2d <_Z14run_simulationv+0x1cd>
  400b23:	c5 f9 2e 35 7d 04 00 	vucomisd 0x47d(%rip),%xmm6        # 400fa8 <_IO_stdin_used+0x88>
  400b2a:	00 
  400b2b:	76 0d                	jbe    400b3a <_Z14run_simulationv+0x1da>
  400b2d:	c4 c1 51 57 d0       	vxorpd %xmm8,%xmm5,%xmm2
  400b32:	c5 f9 13 96 a8 f6 66 	vmovlpd %xmm2,0x66f6a8(%rsi)
  400b39:	00 
  400b3a:	c4 e3 79 05 f6 01    	vpermilpd $0x1,%xmm6,%xmm6
  400b40:	c5 f9 2e 35 58 04 00 	vucomisd 0x458(%rip),%xmm6        # 400fa0 <_IO_stdin_used+0x80>
  400b47:	00 
  400b48:	72 0a                	jb     400b54 <_Z14run_simulationv+0x1f4>
  400b4a:	c5 f9 2e 35 56 04 00 	vucomisd 0x456(%rip),%xmm6        # 400fa8 <_IO_stdin_used+0x88>
  400b51:	00 
  400b52:	76 13                	jbe    400b67 <_Z14run_simulationv+0x207>
  400b54:	c4 e3 79 05 d5 01    	vpermilpd $0x1,%xmm5,%xmm2
  400b5a:	c4 c1 69 57 d0       	vxorpd %xmm8,%xmm2,%xmm2
  400b5f:	c5 f9 13 96 b0 f6 66 	vmovlpd %xmm2,0x66f6b0(%rsi)
  400b66:	00 
  400b67:	48 83 c6 38          	add    $0x38,%rsi
  400b6b:	0f 85 6f ff ff ff    	jne    400ae0 <_Z14run_simulationv+0x180>
  400b71:	41 83 c0 01          	add    $0x1,%r8d
  400b75:	41 83 f8 05          	cmp    $0x5,%r8d
  400b79:	0f 85 11 fe ff ff    	jne    400990 <_Z14run_simulationv+0x30>
  400b7f:	c3                   	retq   

0000000000400b80 <_Z10run_simplev>:
  400b80:	31 c0                	xor    %eax,%eax
  400b82:	c5 f9 28 05 36 04 00 	vmovapd 0x436(%rip),%xmm0        # 400fc0 <_IO_stdin_used+0xa0>
  400b89:	00 
  400b8a:	c5 f9 28 0d 2e 04 00 	vmovapd 0x42e(%rip),%xmm1        # 400fc0 <_IO_stdin_used+0xa0>
  400b91:	00 
  400b92:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b99:	1f 84 00 00 00 00 00 
  400ba0:	31 c9                	xor    %ecx,%ecx
  400ba2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400ba9:	1f 84 00 00 00 00 00 
  400bb0:	48 6b d1 38          	imul   $0x38,%rcx,%rdx
  400bb4:	c5 f9 59 92 b8 20 60 	vmulpd 0x6020b8(%rdx),%xmm0,%xmm2
  400bbb:	00 
  400bbc:	c5 fb 12 9a 90 20 60 	vmovddup 0x602090(%rdx),%xmm3
  400bc3:	00 
  400bc4:	c5 e9 5e db          	vdivpd %xmm3,%xmm2,%xmm3
  400bc8:	c5 f9 10 a2 a8 20 60 	vmovupd 0x6020a8(%rdx),%xmm4
  400bcf:	00 
  400bd0:	c5 f9 10 92 98 20 60 	vmovupd 0x602098(%rdx),%xmm2
  400bd7:	00 
  400bd8:	be 64 00 00 00       	mov    $0x64,%esi
  400bdd:	0f 1f 00             	nopl   (%rax)
  400be0:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400be4:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400be9:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400bed:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400bf2:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400bf6:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400bfb:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400bff:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c04:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c08:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c0d:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c11:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c16:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c1a:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c1f:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c23:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c28:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c2c:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c31:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400c35:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400c3a:	83 c6 f6             	add    $0xfffffff6,%esi
  400c3d:	75 a1                	jne    400be0 <_Z10run_simplev+0x60>
  400c3f:	c5 f9 11 a2 a8 20 60 	vmovupd %xmm4,0x6020a8(%rdx)
  400c46:	00 
  400c47:	c5 f9 11 92 98 20 60 	vmovupd %xmm2,0x602098(%rdx)
  400c4e:	00 
  400c4f:	48 83 c1 01          	add    $0x1,%rcx
  400c53:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  400c5a:	0f 85 50 ff ff ff    	jne    400bb0 <_Z10run_simplev+0x30>
  400c60:	83 c0 01             	add    $0x1,%eax
  400c63:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  400c68:	0f 85 32 ff ff ff    	jne    400ba0 <_Z10run_simplev+0x20>
  400c6e:	c3                   	retq   
  400c6f:	90                   	nop

0000000000400c70 <main>:
  400c70:	55                   	push   %rbp
  400c71:	41 57                	push   %r15
  400c73:	41 56                	push   %r14
  400c75:	41 55                	push   %r13
  400c77:	41 54                	push   %r12
  400c79:	53                   	push   %rbx
  400c7a:	50                   	push   %rax
  400c7b:	e8 80 fa ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c80:	48 89 c5             	mov    %rax,%rbp
  400c83:	e8 f8 fb ff ff       	callq  400880 <_Z13instantiationv>
  400c88:	e8 73 fa ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c8d:	48 89 c3             	mov    %rax,%rbx
  400c90:	48 29 eb             	sub    %rbp,%rbx
  400c93:	e8 68 fa ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400c98:	48 89 c5             	mov    %rax,%rbp
  400c9b:	e8 c0 fc ff ff       	callq  400960 <_Z14run_simulationv>
  400ca0:	e8 5b fa ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400ca5:	49 89 c7             	mov    %rax,%r15
  400ca8:	49 29 ef             	sub    %rbp,%r15
  400cab:	31 ed                	xor    %ebp,%ebp
  400cad:	e8 4e fa ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400cb2:	49 89 c6             	mov    %rax,%r14
  400cb5:	c5 f9 28 05 03 03 00 	vmovapd 0x303(%rip),%xmm0        # 400fc0 <_IO_stdin_used+0xa0>
  400cbc:	00 
  400cbd:	c5 f9 28 0d fb 02 00 	vmovapd 0x2fb(%rip),%xmm1        # 400fc0 <_IO_stdin_used+0xa0>
  400cc4:	00 
  400cc5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400ccc:	00 00 00 00 
  400cd0:	31 c0                	xor    %eax,%eax
  400cd2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400cd9:	1f 84 00 00 00 00 00 
  400ce0:	48 6b c8 38          	imul   $0x38,%rax,%rcx
  400ce4:	c5 f9 59 91 b8 20 60 	vmulpd 0x6020b8(%rcx),%xmm0,%xmm2
  400ceb:	00 
  400cec:	c5 fb 12 99 90 20 60 	vmovddup 0x602090(%rcx),%xmm3
  400cf3:	00 
  400cf4:	c5 e9 5e db          	vdivpd %xmm3,%xmm2,%xmm3
  400cf8:	c5 f9 10 a1 a8 20 60 	vmovupd 0x6020a8(%rcx),%xmm4
  400cff:	00 
  400d00:	c5 f9 10 91 98 20 60 	vmovupd 0x602098(%rcx),%xmm2
  400d07:	00 
  400d08:	ba 64 00 00 00       	mov    $0x64,%edx
  400d0d:	0f 1f 00             	nopl   (%rax)
  400d10:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d14:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d19:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d1d:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d22:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d26:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d2b:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d2f:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d34:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d38:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d3d:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d41:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d46:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d4a:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d4f:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d53:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d58:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d5c:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d61:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  400d65:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  400d6a:	83 c2 f6             	add    $0xfffffff6,%edx
  400d6d:	75 a1                	jne    400d10 <main+0xa0>
  400d6f:	c5 f9 11 a1 a8 20 60 	vmovupd %xmm4,0x6020a8(%rcx)
  400d76:	00 
  400d77:	c5 f9 11 91 98 20 60 	vmovupd %xmm2,0x602098(%rcx)
  400d7e:	00 
  400d7f:	48 83 c0 01          	add    $0x1,%rax
  400d83:	48 3d 40 1f 00 00    	cmp    $0x1f40,%rax
  400d89:	0f 85 51 ff ff ff    	jne    400ce0 <main+0x70>
  400d8f:	83 c5 01             	add    $0x1,%ebp
  400d92:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
  400d98:	0f 85 32 ff ff ff    	jne    400cd0 <main+0x60>
  400d9e:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  400da5:	de 1b 43 
  400da8:	48 89 d8             	mov    %rbx,%rax
  400dab:	48 f7 ee             	imul   %rsi
  400dae:	49 89 d4             	mov    %rdx,%r12
  400db1:	4c 89 e1             	mov    %r12,%rcx
  400db4:	48 c1 e9 3f          	shr    $0x3f,%rcx
  400db8:	49 c1 fc 12          	sar    $0x12,%r12
  400dbc:	4c 89 f8             	mov    %r15,%rax
  400dbf:	48 f7 ee             	imul   %rsi
  400dc2:	49 89 d5             	mov    %rdx,%r13
  400dc5:	49 01 cc             	add    %rcx,%r12
  400dc8:	4c 89 e8             	mov    %r13,%rax
  400dcb:	48 c1 e8 3f          	shr    $0x3f,%rax
  400dcf:	49 c1 fd 12          	sar    $0x12,%r13
  400dd3:	49 01 c5             	add    %rax,%r13
  400dd6:	e8 25 f9 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400ddb:	4c 29 f0             	sub    %r14,%rax
  400dde:	48 89 04 24          	mov    %rax,(%rsp)
  400de2:	bd 0b 00 00 00       	mov    $0xb,%ebp
  400de7:	31 db                	xor    %ebx,%ebx
  400de9:	41 be a0 20 60 00    	mov    $0x6020a0,%r14d
  400def:	90                   	nop
  400df0:	c4 c1 7b 10 46 f8    	vmovsd -0x8(%r14),%xmm0
  400df6:	c4 c1 7b 10 0e       	vmovsd (%r14),%xmm1
  400dfb:	c5 fb 5a d0          	vcvtsd2ss %xmm0,%xmm0,%xmm2
  400dff:	c5 f9 7e d0          	vmovd  %xmm2,%eax
  400e03:	01 e8                	add    %ebp,%eax
  400e05:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  400e09:	c5 f9 7e d5          	vmovd  %xmm2,%ebp
  400e0d:	01 c5                	add    %eax,%ebp
  400e0f:	48 63 c5             	movslq %ebp,%rax
  400e12:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  400e19:	48 89 c1             	mov    %rax,%rcx
  400e1c:	48 c1 e9 3f          	shr    $0x3f,%rcx
  400e20:	48 c1 f8 33          	sar    $0x33,%rax
  400e24:	01 c8                	add    %ecx,%eax
  400e26:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  400e2d:	48 83 fb 09          	cmp    $0x9,%rbx
  400e31:	77 0f                	ja     400e42 <main+0x1d2>
  400e33:	bf e0 0f 40 00       	mov    $0x400fe0,%edi
  400e38:	b0 02                	mov    $0x2,%al
  400e3a:	48 89 de             	mov    %rbx,%rsi
  400e3d:	e8 5e f8 ff ff       	callq  4006a0 <printf@plt>
  400e42:	44 29 fd             	sub    %r15d,%ebp
  400e45:	48 83 c3 01          	add    $0x1,%rbx
  400e49:	49 83 c6 38          	add    $0x38,%r14
  400e4d:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  400e54:	75 9a                	jne    400df0 <main+0x180>
  400e56:	48 8b 04 24          	mov    (%rsp),%rax
  400e5a:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  400e61:	de 1b 43 
  400e64:	48 f7 e9             	imul   %rcx
  400e67:	48 89 d0             	mov    %rdx,%rax
  400e6a:	48 c1 e8 3f          	shr    $0x3f,%rax
  400e6e:	48 c1 fa 12          	sar    $0x12,%rdx
  400e72:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  400e76:	bf f5 0f 40 00       	mov    $0x400ff5,%edi
  400e7b:	31 c0                	xor    %eax,%eax
  400e7d:	4c 89 e6             	mov    %r12,%rsi
  400e80:	4c 89 ea             	mov    %r13,%rdx
  400e83:	41 89 e8             	mov    %ebp,%r8d
  400e86:	e8 15 f8 ff ff       	callq  4006a0 <printf@plt>
  400e8b:	31 c0                	xor    %eax,%eax
  400e8d:	48 83 c4 08          	add    $0x8,%rsp
  400e91:	5b                   	pop    %rbx
  400e92:	41 5c                	pop    %r12
  400e94:	41 5d                	pop    %r13
  400e96:	41 5e                	pop    %r14
  400e98:	41 5f                	pop    %r15
  400e9a:	5d                   	pop    %rbp
  400e9b:	c3                   	retq   
  400e9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ea0 <__libc_csu_init>:
  400ea0:	41 57                	push   %r15
  400ea2:	41 56                	push   %r14
  400ea4:	41 89 ff             	mov    %edi,%r15d
  400ea7:	41 55                	push   %r13
  400ea9:	41 54                	push   %r12
  400eab:	4c 8d 25 1e 0f 20 00 	lea    0x200f1e(%rip),%r12        # 601dd0 <__frame_dummy_init_array_entry>
  400eb2:	55                   	push   %rbp
  400eb3:	48 8d 2d 2e 0f 20 00 	lea    0x200f2e(%rip),%rbp        # 601de8 <__init_array_end>
  400eba:	53                   	push   %rbx
  400ebb:	49 89 f6             	mov    %rsi,%r14
  400ebe:	49 89 d5             	mov    %rdx,%r13
  400ec1:	4c 29 e5             	sub    %r12,%rbp
  400ec4:	48 83 ec 08          	sub    $0x8,%rsp
  400ec8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ecc:	e8 9f f7 ff ff       	callq  400670 <_init>
  400ed1:	48 85 ed             	test   %rbp,%rbp
  400ed4:	74 20                	je     400ef6 <__libc_csu_init+0x56>
  400ed6:	31 db                	xor    %ebx,%ebx
  400ed8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400edf:	00 
  400ee0:	4c 89 ea             	mov    %r13,%rdx
  400ee3:	4c 89 f6             	mov    %r14,%rsi
  400ee6:	44 89 ff             	mov    %r15d,%edi
  400ee9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400eed:	48 83 c3 01          	add    $0x1,%rbx
  400ef1:	48 39 eb             	cmp    %rbp,%rbx
  400ef4:	75 ea                	jne    400ee0 <__libc_csu_init+0x40>
  400ef6:	48 83 c4 08          	add    $0x8,%rsp
  400efa:	5b                   	pop    %rbx
  400efb:	5d                   	pop    %rbp
  400efc:	41 5c                	pop    %r12
  400efe:	41 5d                	pop    %r13
  400f00:	41 5e                	pop    %r14
  400f02:	41 5f                	pop    %r15
  400f04:	c3                   	retq   
  400f05:	90                   	nop
  400f06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f0d:	00 00 00 

0000000000400f10 <__libc_csu_fini>:
  400f10:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f14 <_fini>:
  400f14:	48 83 ec 08          	sub    $0x8,%rsp
  400f18:	48 83 c4 08          	add    $0x8,%rsp
  400f1c:	c3                   	retq   
