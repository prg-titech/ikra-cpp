
bin/clang++-5.0_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

0000000000400650 <_init>:
  400650:	48 83 ec 08          	sub    $0x8,%rsp
  400654:	48 8b 05 9d 19 20 00 	mov    0x20199d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40065b:	48 85 c0             	test   %rax,%rax
  40065e:	74 05                	je     400665 <_init+0x15>
  400660:	e8 9b 00 00 00       	callq  400700 <rand@plt+0x10>
  400665:	48 83 c4 08          	add    $0x8,%rsp
  400669:	c3                   	retq   

Disassembly of section .plt:

0000000000400670 <printf@plt-0x10>:
  400670:	ff 35 92 19 20 00    	pushq  0x201992(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400676:	ff 25 94 19 20 00    	jmpq   *0x201994(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40067c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400680 <printf@plt>:
  400680:	ff 25 92 19 20 00    	jmpq   *0x201992(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400686:	68 00 00 00 00       	pushq  $0x0
  40068b:	e9 e0 ff ff ff       	jmpq   400670 <_init+0x20>

0000000000400690 <_ZNSt8ios_base4InitC1Ev@plt>:
  400690:	ff 25 8a 19 20 00    	jmpq   *0x20198a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400696:	68 01 00 00 00       	pushq  $0x1
  40069b:	e9 d0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006a0 <__libc_start_main@plt>:
  4006a0:	ff 25 82 19 20 00    	jmpq   *0x201982(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006a6:	68 02 00 00 00       	pushq  $0x2
  4006ab:	e9 c0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006b0 <__cxa_atexit@plt>:
  4006b0:	ff 25 7a 19 20 00    	jmpq   *0x20197a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006b6:	68 03 00 00 00       	pushq  $0x3
  4006bb:	e9 b0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006c0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4006c0:	ff 25 72 19 20 00    	jmpq   *0x201972(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  4006c6:	68 04 00 00 00       	pushq  $0x4
  4006cb:	e9 a0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006d0 <srand@plt>:
  4006d0:	ff 25 6a 19 20 00    	jmpq   *0x20196a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4006d6:	68 05 00 00 00       	pushq  $0x5
  4006db:	e9 90 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  4006e0:	ff 25 62 19 20 00    	jmpq   *0x201962(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  4006e6:	68 06 00 00 00       	pushq  $0x6
  4006eb:	e9 80 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006f0 <rand@plt>:
  4006f0:	ff 25 5a 19 20 00    	jmpq   *0x20195a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  4006f6:	68 07 00 00 00       	pushq  $0x7
  4006fb:	e9 70 ff ff ff       	jmpq   400670 <_init+0x20>

Disassembly of section .plt.got:

0000000000400700 <.plt.got>:
  400700:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400706:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400710 <set_fast_math>:
  400710:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  400715:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  40071c:	00 
  40071d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  400722:	c3                   	retq   
  400723:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40072a:	00 00 00 
  40072d:	0f 1f 00             	nopl   (%rax)

0000000000400730 <_GLOBAL__sub_I_aos.cc>:
  400730:	50                   	push   %rax
  400731:	bf 80 20 60 00       	mov    $0x602080,%edi
  400736:	e8 55 ff ff ff       	callq  400690 <_ZNSt8ios_base4InitC1Ev@plt>
  40073b:	bf c0 06 40 00       	mov    $0x4006c0,%edi
  400740:	be 80 20 60 00       	mov    $0x602080,%esi
  400745:	ba 60 20 60 00       	mov    $0x602060,%edx
  40074a:	58                   	pop    %rax
  40074b:	e9 60 ff ff ff       	jmpq   4006b0 <__cxa_atexit@plt>

0000000000400750 <_start>:
  400750:	31 ed                	xor    %ebp,%ebp
  400752:	49 89 d1             	mov    %rdx,%r9
  400755:	5e                   	pop    %rsi
  400756:	48 89 e2             	mov    %rsp,%rdx
  400759:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40075d:	50                   	push   %rax
  40075e:	54                   	push   %rsp
  40075f:	49 c7 c0 c0 14 40 00 	mov    $0x4014c0,%r8
  400766:	48 c7 c1 50 14 40 00 	mov    $0x401450,%rcx
  40076d:	48 c7 c7 40 0f 40 00 	mov    $0x400f40,%rdi
  400774:	e8 27 ff ff ff       	callq  4006a0 <__libc_start_main@plt>
  400779:	f4                   	hlt    
  40077a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400780 <deregister_tm_clones>:
  400780:	b8 77 20 60 00       	mov    $0x602077,%eax
  400785:	55                   	push   %rbp
  400786:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  40078c:	48 83 f8 0e          	cmp    $0xe,%rax
  400790:	48 89 e5             	mov    %rsp,%rbp
  400793:	76 1b                	jbe    4007b0 <deregister_tm_clones+0x30>
  400795:	b8 00 00 00 00       	mov    $0x0,%eax
  40079a:	48 85 c0             	test   %rax,%rax
  40079d:	74 11                	je     4007b0 <deregister_tm_clones+0x30>
  40079f:	5d                   	pop    %rbp
  4007a0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4007a5:	ff e0                	jmpq   *%rax
  4007a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ae:	00 00 
  4007b0:	5d                   	pop    %rbp
  4007b1:	c3                   	retq   
  4007b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007bd:	00 00 00 

00000000004007c0 <register_tm_clones>:
  4007c0:	be 70 20 60 00       	mov    $0x602070,%esi
  4007c5:	55                   	push   %rbp
  4007c6:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  4007cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4007d1:	48 89 e5             	mov    %rsp,%rbp
  4007d4:	48 89 f0             	mov    %rsi,%rax
  4007d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4007db:	48 01 c6             	add    %rax,%rsi
  4007de:	48 d1 fe             	sar    %rsi
  4007e1:	74 15                	je     4007f8 <register_tm_clones+0x38>
  4007e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4007e8:	48 85 c0             	test   %rax,%rax
  4007eb:	74 0b                	je     4007f8 <register_tm_clones+0x38>
  4007ed:	5d                   	pop    %rbp
  4007ee:	bf 70 20 60 00       	mov    $0x602070,%edi
  4007f3:	ff e0                	jmpq   *%rax
  4007f5:	0f 1f 00             	nopl   (%rax)
  4007f8:	5d                   	pop    %rbp
  4007f9:	c3                   	retq   
  4007fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400800 <__do_global_dtors_aux>:
  400800:	80 3d 69 18 20 00 00 	cmpb   $0x0,0x201869(%rip)        # 602070 <__TMC_END__>
  400807:	75 11                	jne    40081a <__do_global_dtors_aux+0x1a>
  400809:	55                   	push   %rbp
  40080a:	48 89 e5             	mov    %rsp,%rbp
  40080d:	e8 6e ff ff ff       	callq  400780 <deregister_tm_clones>
  400812:	5d                   	pop    %rbp
  400813:	c6 05 56 18 20 00 01 	movb   $0x1,0x201856(%rip)        # 602070 <__TMC_END__>
  40081a:	f3 c3                	repz retq 
  40081c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400820 <frame_dummy>:
  400820:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400825:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400829:	75 05                	jne    400830 <frame_dummy+0x10>
  40082b:	eb 93                	jmp    4007c0 <register_tm_clones>
  40082d:	0f 1f 00             	nopl   (%rax)
  400830:	b8 00 00 00 00       	mov    $0x0,%eax
  400835:	48 85 c0             	test   %rax,%rax
  400838:	74 f1                	je     40082b <frame_dummy+0xb>
  40083a:	55                   	push   %rbp
  40083b:	48 89 e5             	mov    %rsp,%rbp
  40083e:	ff d0                	callq  *%rax
  400840:	5d                   	pop    %rbp
  400841:	e9 7a ff ff ff       	jmpq   4007c0 <register_tm_clones>
  400846:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40084d:	00 00 00 

0000000000400850 <_Z11r_float2intf>:
  400850:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  400854:	c3                   	retq   
  400855:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40085c:	00 00 00 00 

0000000000400860 <_Z13instantiationv>:
  400860:	55                   	push   %rbp
  400861:	41 57                	push   %r15
  400863:	41 56                	push   %r14
  400865:	53                   	push   %rbx
  400866:	48 83 ec 38          	sub    $0x38,%rsp
  40086a:	bf 2a 00 00 00       	mov    $0x2a,%edi
  40086f:	e8 5c fe ff ff       	callq  4006d0 <srand@plt>
  400874:	bd 40 1f 00 00       	mov    $0x1f40,%ebp
  400879:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400880:	c5 f8 77             	vzeroupper 
  400883:	e8 68 fe ff ff       	callq  4006f0 <rand@plt>
  400888:	41 89 c6             	mov    %eax,%r14d
  40088b:	e8 60 fe ff ff       	callq  4006f0 <rand@plt>
  400890:	41 89 c7             	mov    %eax,%r15d
  400893:	e8 58 fe ff ff       	callq  4006f0 <rand@plt>
  400898:	89 c3                	mov    %eax,%ebx
  40089a:	e8 51 fe ff ff       	callq  4006f0 <rand@plt>
  40089f:	c4 c1 79 6e c6       	vmovd  %r14d,%xmm0
  4008a4:	c4 c3 79 22 c7 01    	vpinsrd $0x1,%r15d,%xmm0,%xmm0
  4008aa:	c4 e3 79 22 c3 02    	vpinsrd $0x2,%ebx,%xmm0,%xmm0
  4008b0:	c4 e3 79 22 c0 03    	vpinsrd $0x3,%eax,%xmm0,%xmm0
  4008b6:	c5 fe e6 c8          	vcvtdq2pd %xmm0,%ymm1
  4008ba:	c5 fd 28 05 5e 0c 00 	vmovapd 0xc5e(%rip),%ymm0        # 401520 <_IO_stdin_used+0x40>
  4008c1:	00 
  4008c2:	c4 e2 fd 98 0d 35 0c 	vfmadd132pd 0xc35(%rip),%ymm0,%ymm1        # 401500 <_IO_stdin_used+0x20>
  4008c9:	00 00 
  4008cb:	c5 fd 11 0c 24       	vmovupd %ymm1,(%rsp)
  4008d0:	c5 f8 77             	vzeroupper 
  4008d3:	e8 18 fe ff ff       	callq  4006f0 <rand@plt>
  4008d8:	c5 eb 2a c0          	vcvtsi2sd %eax,%xmm2,%xmm0
  4008dc:	c5 fb 10 0d 84 0c 00 	vmovsd 0xc84(%rip),%xmm1        # 401568 <_IO_stdin_used+0x88>
  4008e3:	00 
  4008e4:	c4 e2 f1 99 05 73 0c 	vfmadd132sd 0xc73(%rip),%xmm1,%xmm0        # 401560 <_IO_stdin_used+0x80>
  4008eb:	00 00 
  4008ed:	48 8b 05 74 17 20 00 	mov    0x201774(%rip),%rax        # 602068 <Body_arena_head>
  4008f4:	c5 fc 10 0c 24       	vmovups (%rsp),%ymm1
  4008f9:	c5 fc 11 08          	vmovups %ymm1,(%rax)
  4008fd:	c5 fb 11 40 20       	vmovsd %xmm0,0x20(%rax)
  400902:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400906:	c5 f9 11 40 28       	vmovupd %xmm0,0x28(%rax)
  40090b:	48 83 c0 38          	add    $0x38,%rax
  40090f:	48 89 05 52 17 20 00 	mov    %rax,0x201752(%rip)        # 602068 <Body_arena_head>
  400916:	83 c5 ff             	add    $0xffffffff,%ebp
  400919:	0f 85 61 ff ff ff    	jne    400880 <_Z13instantiationv+0x20>
  40091f:	48 83 c4 38          	add    $0x38,%rsp
  400923:	5b                   	pop    %rbx
  400924:	41 5e                	pop    %r14
  400926:	41 5f                	pop    %r15
  400928:	5d                   	pop    %rbp
  400929:	c5 f8 77             	vzeroupper 
  40092c:	c3                   	retq   
  40092d:	0f 1f 00             	nopl   (%rax)

0000000000400930 <_Z14run_simulationv>:
  400930:	31 c0                	xor    %eax,%eax
  400932:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400936:	c5 79 28 15 72 0c 00 	vmovapd 0xc72(%rip),%xmm10        # 4015b0 <_IO_stdin_used+0xd0>
  40093d:	00 
  40093e:	c5 fb 10 15 3a 0c 00 	vmovsd 0xc3a(%rip),%xmm2        # 401580 <_IO_stdin_used+0xa0>
  400945:	00 
  400946:	c5 79 28 05 72 0c 00 	vmovapd 0xc72(%rip),%xmm8        # 4015c0 <_IO_stdin_used+0xe0>
  40094d:	00 
  40094e:	c5 7b 10 0d 32 0c 00 	vmovsd 0xc32(%rip),%xmm9        # 401588 <_IO_stdin_used+0xa8>
  400955:	00 
  400956:	c5 fb 10 2d 12 0c 00 	vmovsd 0xc12(%rip),%xmm5        # 401570 <_IO_stdin_used+0x90>
  40095d:	00 
  40095e:	c5 fb 10 35 12 0c 00 	vmovsd 0xc12(%rip),%xmm6        # 401578 <_IO_stdin_used+0x98>
  400965:	00 
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 
  400970:	31 c9                	xor    %ecx,%ecx
  400972:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400979:	1f 84 00 00 00 00 00 
  400980:	c5 f9 11 81 b8 20 60 	vmovupd %xmm0,0x6020b8(%rcx)
  400987:	00 
  400988:	c5 f9 11 81 f0 20 60 	vmovupd %xmm0,0x6020f0(%rcx)
  40098f:	00 
  400990:	c5 f9 11 81 28 21 60 	vmovupd %xmm0,0x602128(%rcx)
  400997:	00 
  400998:	c5 f9 11 81 60 21 60 	vmovupd %xmm0,0x602160(%rcx)
  40099f:	00 
  4009a0:	c5 f9 11 81 98 21 60 	vmovupd %xmm0,0x602198(%rcx)
  4009a7:	00 
  4009a8:	c5 f9 11 81 d0 21 60 	vmovupd %xmm0,0x6021d0(%rcx)
  4009af:	00 
  4009b0:	c5 f9 11 81 08 22 60 	vmovupd %xmm0,0x602208(%rcx)
  4009b7:	00 
  4009b8:	c5 f9 11 81 40 22 60 	vmovupd %xmm0,0x602240(%rcx)
  4009bf:	00 
  4009c0:	c5 f9 11 81 78 22 60 	vmovupd %xmm0,0x602278(%rcx)
  4009c7:	00 
  4009c8:	c5 f9 11 81 b0 22 60 	vmovupd %xmm0,0x6022b0(%rcx)
  4009cf:	00 
  4009d0:	c5 f9 11 81 e8 22 60 	vmovupd %xmm0,0x6022e8(%rcx)
  4009d7:	00 
  4009d8:	c5 f9 11 81 20 23 60 	vmovupd %xmm0,0x602320(%rcx)
  4009df:	00 
  4009e0:	c5 f9 11 81 58 23 60 	vmovupd %xmm0,0x602358(%rcx)
  4009e7:	00 
  4009e8:	c5 f9 11 81 90 23 60 	vmovupd %xmm0,0x602390(%rcx)
  4009ef:	00 
  4009f0:	c5 f9 11 81 c8 23 60 	vmovupd %xmm0,0x6023c8(%rcx)
  4009f7:	00 
  4009f8:	c5 f9 11 81 00 24 60 	vmovupd %xmm0,0x602400(%rcx)
  4009ff:	00 
  400a00:	48 81 c1 80 03 00 00 	add    $0x380,%rcx
  400a07:	48 81 f9 00 d6 06 00 	cmp    $0x6d600,%rcx
  400a0e:	0f 85 6c ff ff ff    	jne    400980 <_Z14run_simulationv+0x50>
  400a14:	31 c9                	xor    %ecx,%ecx
  400a16:	31 d2                	xor    %edx,%edx
  400a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a1f:	00 
  400a20:	48 6b f2 38          	imul   $0x38,%rdx,%rsi
  400a24:	31 ff                	xor    %edi,%edi
  400a26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2d:	00 00 00 
  400a30:	48 39 f9             	cmp    %rdi,%rcx
  400a33:	74 5b                	je     400a90 <_Z14run_simulationv+0x160>
  400a35:	c5 f9 10 9e 98 20 60 	vmovupd 0x602098(%rsi),%xmm3
  400a3c:	00 
  400a3d:	c5 e1 5c 9f 98 20 60 	vsubpd 0x602098(%rdi),%xmm3,%xmm3
  400a44:	00 
  400a45:	c5 e3 59 fb          	vmulsd %xmm3,%xmm3,%xmm7
  400a49:	c4 e3 79 05 e3 01    	vpermilpd $0x1,%xmm3,%xmm4
  400a4f:	c4 e2 d9 a9 e7       	vfmadd213sd %xmm7,%xmm4,%xmm4
  400a54:	c5 db 51 fc          	vsqrtsd %xmm4,%xmm4,%xmm7
  400a58:	c5 d3 59 8f 90 20 60 	vmulsd 0x602090(%rdi),%xmm5,%xmm1
  400a5f:	00 
  400a60:	c5 f3 59 8e 90 20 60 	vmulsd 0x602090(%rsi),%xmm1,%xmm1
  400a67:	00 
  400a68:	c5 db 58 e6          	vaddsd %xmm6,%xmm4,%xmm4
  400a6c:	c5 f3 5e cc          	vdivsd %xmm4,%xmm1,%xmm1
  400a70:	c5 fb 12 c9          	vmovddup %xmm1,%xmm1
  400a74:	c5 f1 59 cb          	vmulpd %xmm3,%xmm1,%xmm1
  400a78:	c5 fb 12 df          	vmovddup %xmm7,%xmm3
  400a7c:	c5 f1 5e cb          	vdivpd %xmm3,%xmm1,%xmm1
  400a80:	c5 f1 58 8f b8 20 60 	vaddpd 0x6020b8(%rdi),%xmm1,%xmm1
  400a87:	00 
  400a88:	c5 f9 11 8f b8 20 60 	vmovupd %xmm1,0x6020b8(%rdi)
  400a8f:	00 
  400a90:	48 83 c7 38          	add    $0x38,%rdi
  400a94:	48 81 ff 00 d6 06 00 	cmp    $0x6d600,%rdi
  400a9b:	75 93                	jne    400a30 <_Z14run_simulationv+0x100>
  400a9d:	48 83 c2 01          	add    $0x1,%rdx
  400aa1:	48 83 c1 38          	add    $0x38,%rcx
  400aa5:	48 81 fa 40 1f 00 00 	cmp    $0x1f40,%rdx
  400aac:	0f 85 6e ff ff ff    	jne    400a20 <_Z14run_simulationv+0xf0>
  400ab2:	48 c7 c1 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rcx
  400ab9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400ac0:	c5 a9 59 89 b8 f6 66 	vmulpd 0x66f6b8(%rcx),%xmm10,%xmm1
  400ac7:	00 
  400ac8:	c5 fb 12 99 90 f6 66 	vmovddup 0x66f690(%rcx),%xmm3
  400acf:	00 
  400ad0:	c5 f1 5e cb          	vdivpd %xmm3,%xmm1,%xmm1
  400ad4:	c5 f1 58 b9 a8 f6 66 	vaddpd 0x66f6a8(%rcx),%xmm1,%xmm7
  400adb:	00 
  400adc:	c5 f9 11 b9 a8 f6 66 	vmovupd %xmm7,0x66f6a8(%rcx)
  400ae3:	00 
  400ae4:	c5 f9 10 99 98 f6 66 	vmovupd 0x66f698(%rcx),%xmm3
  400aeb:	00 
  400aec:	c4 c2 c1 b8 da       	vfmadd231pd %xmm10,%xmm7,%xmm3
  400af1:	c5 f9 11 99 98 f6 66 	vmovupd %xmm3,0x66f698(%rcx)
  400af8:	00 
  400af9:	c5 f9 2e da          	vucomisd %xmm2,%xmm3
  400afd:	72 07                	jb     400b06 <_Z14run_simulationv+0x1d6>
  400aff:	c4 c1 79 2e d9       	vucomisd %xmm9,%xmm3
  400b04:	76 0d                	jbe    400b13 <_Z14run_simulationv+0x1e3>
  400b06:	c4 c1 41 57 c8       	vxorpd %xmm8,%xmm7,%xmm1
  400b0b:	c5 f9 13 89 a8 f6 66 	vmovlpd %xmm1,0x66f6a8(%rcx)
  400b12:	00 
  400b13:	c4 e3 79 05 db 01    	vpermilpd $0x1,%xmm3,%xmm3
  400b19:	c5 f9 2e da          	vucomisd %xmm2,%xmm3
  400b1d:	72 11                	jb     400b30 <_Z14run_simulationv+0x200>
  400b1f:	c4 c1 79 2e d9       	vucomisd %xmm9,%xmm3
  400b24:	77 0a                	ja     400b30 <_Z14run_simulationv+0x200>
  400b26:	48 83 c1 38          	add    $0x38,%rcx
  400b2a:	75 94                	jne    400ac0 <_Z14run_simulationv+0x190>
  400b2c:	eb 1f                	jmp    400b4d <_Z14run_simulationv+0x21d>
  400b2e:	66 90                	xchg   %ax,%ax
  400b30:	c4 e3 79 05 cf 01    	vpermilpd $0x1,%xmm7,%xmm1
  400b36:	c4 c1 71 57 c8       	vxorpd %xmm8,%xmm1,%xmm1
  400b3b:	c5 f9 13 89 b0 f6 66 	vmovlpd %xmm1,0x66f6b0(%rcx)
  400b42:	00 
  400b43:	48 83 c1 38          	add    $0x38,%rcx
  400b47:	0f 85 73 ff ff ff    	jne    400ac0 <_Z14run_simulationv+0x190>
  400b4d:	83 c0 01             	add    $0x1,%eax
  400b50:	83 f8 05             	cmp    $0x5,%eax
  400b53:	0f 85 17 fe ff ff    	jne    400970 <_Z14run_simulationv+0x40>
  400b59:	c3                   	retq   
  400b5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b60 <_Z10run_simplev>:
  400b60:	31 c0                	xor    %eax,%eax
  400b62:	c5 7b 10 2d 1e 0a 00 	vmovsd 0xa1e(%rip),%xmm13        # 401588 <_IO_stdin_used+0xa8>
  400b69:	00 
  400b6a:	c5 79 28 35 3e 0a 00 	vmovapd 0xa3e(%rip),%xmm14        # 4015b0 <_IO_stdin_used+0xd0>
  400b71:	00 
  400b72:	c5 fd 28 15 c6 09 00 	vmovapd 0x9c6(%rip),%ymm2        # 401540 <_IO_stdin_used+0x60>
  400b79:	00 
  400b7a:	c5 fb 10 1d 0e 0a 00 	vmovsd 0xa0e(%rip),%xmm3        # 401590 <_IO_stdin_used+0xb0>
  400b81:	00 
  400b82:	c5 f9 28 25 46 0a 00 	vmovapd 0xa46(%rip),%xmm4        # 4015d0 <_IO_stdin_used+0xf0>
  400b89:	00 
  400b8a:	c4 e2 7d 19 2d 05 0a 	vbroadcastsd 0xa05(%rip),%ymm5        # 401598 <_IO_stdin_used+0xb8>
  400b91:	00 00 
  400b93:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b9a:	84 00 00 00 00 00 
  400ba0:	48 c7 c1 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rcx
  400ba7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400bae:	00 00 
  400bb0:	c5 89 59 b1 b8 f6 66 	vmulpd 0x66f6b8(%rcx),%xmm14,%xmm6
  400bb7:	00 
  400bb8:	c5 93 5e b9 90 f6 66 	vdivsd 0x66f690(%rcx),%xmm13,%xmm7
  400bbf:	00 
  400bc0:	c5 79 10 99 a8 f6 66 	vmovupd 0x66f6a8(%rcx),%xmm11
  400bc7:	00 
  400bc8:	c5 fb 12 ff          	vmovddup %xmm7,%xmm7
  400bcc:	c5 c9 59 f7          	vmulpd %xmm7,%xmm6,%xmm6
  400bd0:	c4 43 fd 01 e3 55    	vpermpd $0x55,%ymm11,%ymm12
  400bd6:	c4 e3 79 05 c6 01    	vpermilpd $0x1,%xmm6,%xmm0
  400bdc:	c4 e3 fd 01 fe 55    	vpermpd $0x55,%ymm6,%ymm7
  400be2:	c4 62 c5 b8 e2       	vfmadd231pd %ymm2,%ymm7,%ymm12
  400be7:	c5 fb 59 c3          	vmulsd %xmm3,%xmm0,%xmm0
  400beb:	c4 62 7d 19 c0       	vbroadcastsd %xmm0,%ymm8
  400bf0:	c4 c2 7d 19 c3       	vbroadcastsd %xmm11,%ymm0
  400bf5:	c4 62 7d 19 ce       	vbroadcastsd %xmm6,%ymm9
  400bfa:	c4 e2 b5 b8 c2       	vfmadd231pd %ymm2,%ymm9,%ymm0
  400bff:	c5 cb 59 cb          	vmulsd %xmm3,%xmm6,%xmm1
  400c03:	c4 62 7d 19 d1       	vbroadcastsd %xmm1,%ymm10
  400c08:	c5 fb 10 89 98 f6 66 	vmovsd 0x66f698(%rcx),%xmm1
  400c0f:	00 
  400c10:	c5 7b 10 b9 a0 f6 66 	vmovsd 0x66f6a0(%rcx),%xmm15
  400c17:	00 
  400c18:	c4 c2 d9 a8 f3       	vfmadd213pd %xmm11,%xmm4,%xmm6
  400c1d:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400c22:	c4 62 d5 a8 d9       	vfmadd213pd %ymm1,%ymm5,%ymm11
  400c27:	c5 9d 58 cf          	vaddpd %ymm7,%ymm12,%ymm1
  400c2b:	c4 c2 d5 a8 cf       	vfmadd213pd %ymm15,%ymm5,%ymm1
  400c30:	c4 41 1d 58 e0       	vaddpd %ymm8,%ymm12,%ymm12
  400c35:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400c3a:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400c3f:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400c44:	c5 1d 58 df          	vaddpd %ymm7,%ymm12,%ymm11
  400c48:	c4 62 d5 a8 d9       	vfmadd213pd %ymm1,%ymm5,%ymm11
  400c4d:	c4 c1 1d 58 c8       	vaddpd %ymm8,%ymm12,%ymm1
  400c52:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400c57:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400c5c:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400c61:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400c65:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400c6a:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400c6f:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400c74:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400c79:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400c7e:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400c82:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400c87:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400c8c:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400c91:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400c96:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400c9b:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400c9f:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400ca4:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400ca9:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400cae:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400cb3:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400cb8:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400cbc:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400cc1:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400cc6:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400ccb:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400cd0:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400cd5:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400cd9:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400cde:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400ce3:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400ce8:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400ced:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400cf2:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400cf6:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400cfb:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d00:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d05:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400d0a:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400d0f:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400d13:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400d18:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d1d:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d22:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400d27:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400d2c:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400d30:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400d35:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d3a:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d3f:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400d44:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400d49:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400d4d:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400d52:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d57:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d5c:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400d61:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400d66:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400d6a:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400d6f:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d74:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d79:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400d7e:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400d83:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400d87:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400d8c:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400d91:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400d96:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400d9b:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400da0:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400da4:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400da9:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400dae:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400db3:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400db8:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400dbd:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400dc1:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400dc6:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400dcb:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400dd0:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400dd5:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400dda:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400dde:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400de3:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400de8:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400ded:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400df2:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400df7:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400dfb:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400e00:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e05:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e0a:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400e0f:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400e14:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400e18:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400e1d:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e22:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e27:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400e2c:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400e31:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400e35:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400e3a:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e3f:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e44:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400e49:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400e4e:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400e52:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400e57:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e5c:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e61:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400e66:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400e6b:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400e6f:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400e74:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e79:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e7e:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  400e83:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400e88:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  400e8c:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400e91:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400e96:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400e9b:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  400ea0:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400ea5:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  400ea9:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  400eae:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  400eb3:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  400eb8:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  400ebd:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  400ec2:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  400ec6:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  400ecb:	c4 41 2d 58 c9       	vaddpd %ymm9,%ymm10,%ymm9
  400ed0:	c4 c1 7d 58 c1       	vaddpd %ymm9,%ymm0,%ymm0
  400ed5:	c5 bd 58 ff          	vaddpd %ymm7,%ymm8,%ymm7
  400ed9:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  400edd:	c4 c2 d5 a8 c4       	vfmadd213pd %ymm12,%ymm5,%ymm0
  400ee2:	c4 c2 d5 a8 cf       	vfmadd213pd %ymm15,%ymm5,%ymm1
  400ee7:	c4 e3 7d 19 c7 01    	vextractf128 $0x1,%ymm0,%xmm7
  400eed:	c5 fd 58 c7          	vaddpd %ymm7,%ymm0,%ymm0
  400ef1:	c5 fd 7c c0          	vhaddpd %ymm0,%ymm0,%ymm0
  400ef5:	c4 e3 7d 19 cf 01    	vextractf128 $0x1,%ymm1,%xmm7
  400efb:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  400eff:	c5 f5 7c c9          	vhaddpd %ymm1,%ymm1,%ymm1
  400f03:	c5 f9 11 b1 a8 f6 66 	vmovupd %xmm6,0x66f6a8(%rcx)
  400f0a:	00 
  400f0b:	c5 f9 14 c1          	vunpcklpd %xmm1,%xmm0,%xmm0
  400f0f:	c5 f9 11 81 98 f6 66 	vmovupd %xmm0,0x66f698(%rcx)
  400f16:	00 
  400f17:	48 83 c1 38          	add    $0x38,%rcx
  400f1b:	0f 85 8f fc ff ff    	jne    400bb0 <_Z10run_simplev+0x50>
  400f21:	83 c0 01             	add    $0x1,%eax
  400f24:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  400f29:	0f 85 71 fc ff ff    	jne    400ba0 <_Z10run_simplev+0x40>
  400f2f:	c5 f8 77             	vzeroupper 
  400f32:	c3                   	retq   
  400f33:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400f3a:	84 00 00 00 00 00 

0000000000400f40 <main>:
  400f40:	55                   	push   %rbp
  400f41:	41 57                	push   %r15
  400f43:	41 56                	push   %r14
  400f45:	41 55                	push   %r13
  400f47:	41 54                	push   %r12
  400f49:	53                   	push   %rbx
  400f4a:	50                   	push   %rax
  400f4b:	e8 90 f7 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f50:	48 89 c3             	mov    %rax,%rbx
  400f53:	e8 08 f9 ff ff       	callq  400860 <_Z13instantiationv>
  400f58:	e8 83 f7 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f5d:	49 89 c7             	mov    %rax,%r15
  400f60:	49 29 df             	sub    %rbx,%r15
  400f63:	e8 78 f7 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f68:	48 89 c3             	mov    %rax,%rbx
  400f6b:	e8 c0 f9 ff ff       	callq  400930 <_Z14run_simulationv>
  400f70:	e8 6b f7 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f75:	49 89 c4             	mov    %rax,%r12
  400f78:	49 29 dc             	sub    %rbx,%r12
  400f7b:	31 db                	xor    %ebx,%ebx
  400f7d:	e8 5e f7 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400f82:	49 89 c6             	mov    %rax,%r14
  400f85:	c5 7b 10 2d fb 05 00 	vmovsd 0x5fb(%rip),%xmm13        # 401588 <_IO_stdin_used+0xa8>
  400f8c:	00 
  400f8d:	c5 79 28 35 1b 06 00 	vmovapd 0x61b(%rip),%xmm14        # 4015b0 <_IO_stdin_used+0xd0>
  400f94:	00 
  400f95:	c5 fd 28 15 a3 05 00 	vmovapd 0x5a3(%rip),%ymm2        # 401540 <_IO_stdin_used+0x60>
  400f9c:	00 
  400f9d:	c5 fb 10 1d eb 05 00 	vmovsd 0x5eb(%rip),%xmm3        # 401590 <_IO_stdin_used+0xb0>
  400fa4:	00 
  400fa5:	c5 f9 28 25 23 06 00 	vmovapd 0x623(%rip),%xmm4        # 4015d0 <_IO_stdin_used+0xf0>
  400fac:	00 
  400fad:	c4 e2 7d 19 2d e2 05 	vbroadcastsd 0x5e2(%rip),%ymm5        # 401598 <_IO_stdin_used+0xb8>
  400fb4:	00 00 
  400fb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fbd:	00 00 00 
  400fc0:	48 c7 c0 00 2a f9 ff 	mov    $0xfffffffffff92a00,%rax
  400fc7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400fce:	00 00 
  400fd0:	c5 89 59 b0 b8 f6 66 	vmulpd 0x66f6b8(%rax),%xmm14,%xmm6
  400fd7:	00 
  400fd8:	c5 93 5e b8 90 f6 66 	vdivsd 0x66f690(%rax),%xmm13,%xmm7
  400fdf:	00 
  400fe0:	c5 79 10 98 a8 f6 66 	vmovupd 0x66f6a8(%rax),%xmm11
  400fe7:	00 
  400fe8:	c5 fb 12 ff          	vmovddup %xmm7,%xmm7
  400fec:	c5 c9 59 f7          	vmulpd %xmm7,%xmm6,%xmm6
  400ff0:	c4 43 fd 01 e3 55    	vpermpd $0x55,%ymm11,%ymm12
  400ff6:	c4 e3 79 05 c6 01    	vpermilpd $0x1,%xmm6,%xmm0
  400ffc:	c4 e3 fd 01 fe 55    	vpermpd $0x55,%ymm6,%ymm7
  401002:	c4 62 c5 b8 e2       	vfmadd231pd %ymm2,%ymm7,%ymm12
  401007:	c5 fb 59 c3          	vmulsd %xmm3,%xmm0,%xmm0
  40100b:	c4 62 7d 19 c0       	vbroadcastsd %xmm0,%ymm8
  401010:	c4 c2 7d 19 c3       	vbroadcastsd %xmm11,%ymm0
  401015:	c4 62 7d 19 ce       	vbroadcastsd %xmm6,%ymm9
  40101a:	c4 e2 b5 b8 c2       	vfmadd231pd %ymm2,%ymm9,%ymm0
  40101f:	c5 cb 59 cb          	vmulsd %xmm3,%xmm6,%xmm1
  401023:	c4 62 7d 19 d1       	vbroadcastsd %xmm1,%ymm10
  401028:	c5 fb 10 88 98 f6 66 	vmovsd 0x66f698(%rax),%xmm1
  40102f:	00 
  401030:	c5 7b 10 b8 a0 f6 66 	vmovsd 0x66f6a0(%rax),%xmm15
  401037:	00 
  401038:	c4 c2 d9 a8 f3       	vfmadd213pd %xmm11,%xmm4,%xmm6
  40103d:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  401042:	c4 62 d5 a8 d9       	vfmadd213pd %ymm1,%ymm5,%ymm11
  401047:	c5 9d 58 cf          	vaddpd %ymm7,%ymm12,%ymm1
  40104b:	c4 c2 d5 a8 cf       	vfmadd213pd %ymm15,%ymm5,%ymm1
  401050:	c4 41 1d 58 e0       	vaddpd %ymm8,%ymm12,%ymm12
  401055:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40105a:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40105f:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401064:	c5 1d 58 df          	vaddpd %ymm7,%ymm12,%ymm11
  401068:	c4 62 d5 a8 d9       	vfmadd213pd %ymm1,%ymm5,%ymm11
  40106d:	c4 c1 1d 58 c8       	vaddpd %ymm8,%ymm12,%ymm1
  401072:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401077:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  40107c:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  401081:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401085:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  40108a:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40108f:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401094:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  401099:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  40109e:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  4010a2:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4010a7:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4010ac:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4010b1:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4010b6:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4010bb:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  4010bf:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4010c4:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4010c9:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4010ce:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  4010d3:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4010d8:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4010dc:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4010e1:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4010e6:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4010eb:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  4010f0:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4010f5:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  4010f9:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4010fe:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401103:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401108:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40110d:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401112:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  401116:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  40111b:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401120:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401125:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  40112a:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  40112f:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401133:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401138:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40113d:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401142:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  401147:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  40114c:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  401150:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  401155:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40115a:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40115f:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401164:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401169:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  40116d:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  401172:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401177:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40117c:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  401181:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  401186:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40118a:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  40118f:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401194:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401199:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  40119e:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4011a3:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  4011a7:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4011ac:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4011b1:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4011b6:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4011bb:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4011c0:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  4011c4:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4011c9:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4011ce:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4011d3:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  4011d8:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4011dd:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4011e1:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4011e6:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4011eb:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4011f0:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  4011f5:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4011fa:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  4011fe:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  401203:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401208:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40120d:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401212:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401217:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  40121b:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  401220:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401225:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40122a:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  40122f:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  401234:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401238:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  40123d:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401242:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401247:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  40124c:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  401251:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  401255:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  40125a:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40125f:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401264:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401269:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  40126e:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  401272:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  401277:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40127c:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401281:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  401286:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  40128b:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40128f:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  401294:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401299:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40129e:	c4 41 7d 58 d9       	vaddpd %ymm9,%ymm0,%ymm11
  4012a3:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4012a8:	c5 75 58 e7          	vaddpd %ymm7,%ymm1,%ymm12
  4012ac:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4012b1:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4012b6:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4012bb:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4012c0:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4012c5:	c5 75 58 df          	vaddpd %ymm7,%ymm1,%ymm11
  4012c9:	c4 42 d5 a8 dc       	vfmadd213pd %ymm12,%ymm5,%ymm11
  4012ce:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4012d3:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4012d8:	c4 41 7d 58 e1       	vaddpd %ymm9,%ymm0,%ymm12
  4012dd:	c4 42 d5 a8 e7       	vfmadd213pd %ymm15,%ymm5,%ymm12
  4012e2:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4012e6:	c4 42 d5 a8 fb       	vfmadd213pd %ymm11,%ymm5,%ymm15
  4012eb:	c4 41 2d 58 c9       	vaddpd %ymm9,%ymm10,%ymm9
  4012f0:	c4 c1 7d 58 c1       	vaddpd %ymm9,%ymm0,%ymm0
  4012f5:	c5 bd 58 ff          	vaddpd %ymm7,%ymm8,%ymm7
  4012f9:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  4012fd:	c4 c2 d5 a8 c4       	vfmadd213pd %ymm12,%ymm5,%ymm0
  401302:	c4 c2 d5 a8 cf       	vfmadd213pd %ymm15,%ymm5,%ymm1
  401307:	c4 e3 7d 19 c7 01    	vextractf128 $0x1,%ymm0,%xmm7
  40130d:	c5 fd 58 c7          	vaddpd %ymm7,%ymm0,%ymm0
  401311:	c5 fd 7c c0          	vhaddpd %ymm0,%ymm0,%ymm0
  401315:	c4 e3 7d 19 cf 01    	vextractf128 $0x1,%ymm1,%xmm7
  40131b:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  40131f:	c5 f5 7c c9          	vhaddpd %ymm1,%ymm1,%ymm1
  401323:	c5 f9 11 b0 a8 f6 66 	vmovupd %xmm6,0x66f6a8(%rax)
  40132a:	00 
  40132b:	c5 f9 14 c1          	vunpcklpd %xmm1,%xmm0,%xmm0
  40132f:	c5 f9 11 80 98 f6 66 	vmovupd %xmm0,0x66f698(%rax)
  401336:	00 
  401337:	48 83 c0 38          	add    $0x38,%rax
  40133b:	0f 85 8f fc ff ff    	jne    400fd0 <main+0x90>
  401341:	83 c3 01             	add    $0x1,%ebx
  401344:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  40134a:	0f 85 70 fc ff ff    	jne    400fc0 <main+0x80>
  401350:	48 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rbp
  401357:	de 1b 43 
  40135a:	4c 89 f8             	mov    %r15,%rax
  40135d:	48 f7 ed             	imul   %rbp
  401360:	49 89 d7             	mov    %rdx,%r15
  401363:	4c 89 f9             	mov    %r15,%rcx
  401366:	48 c1 e9 3f          	shr    $0x3f,%rcx
  40136a:	49 c1 ff 12          	sar    $0x12,%r15
  40136e:	4c 89 e0             	mov    %r12,%rax
  401371:	48 f7 ed             	imul   %rbp
  401374:	49 89 d4             	mov    %rdx,%r12
  401377:	49 01 cf             	add    %rcx,%r15
  40137a:	4c 89 e0             	mov    %r12,%rax
  40137d:	48 c1 e8 3f          	shr    $0x3f,%rax
  401381:	49 c1 fc 12          	sar    $0x12,%r12
  401385:	49 01 c4             	add    %rax,%r12
  401388:	c5 f8 77             	vzeroupper 
  40138b:	e8 50 f3 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401390:	49 89 c5             	mov    %rax,%r13
  401393:	4d 29 f5             	sub    %r14,%r13
  401396:	41 be 0b 00 00 00    	mov    $0xb,%r14d
  40139c:	31 db                	xor    %ebx,%ebx
  40139e:	eb 13                	jmp    4013b3 <main+0x473>
  4013a0:	bf e0 15 40 00       	mov    $0x4015e0,%edi
  4013a5:	b0 02                	mov    $0x2,%al
  4013a7:	48 89 de             	mov    %rbx,%rsi
  4013aa:	e8 d1 f2 ff ff       	callq  400680 <printf@plt>
  4013af:	48 83 c3 01          	add    $0x1,%rbx
  4013b3:	48 6b c3 38          	imul   $0x38,%rbx,%rax
  4013b7:	c5 fb 10 80 98 20 60 	vmovsd 0x602098(%rax),%xmm0
  4013be:	00 
  4013bf:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  4013c3:	c5 f9 7e c9          	vmovd  %xmm1,%ecx
  4013c7:	c5 fb 10 88 a0 20 60 	vmovsd 0x6020a0(%rax),%xmm1
  4013ce:	00 
  4013cf:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  4013d3:	44 01 f1             	add    %r14d,%ecx
  4013d6:	c5 f9 7e d0          	vmovd  %xmm2,%eax
  4013da:	01 c8                	add    %ecx,%eax
  4013dc:	4c 63 f0             	movslq %eax,%r14
  4013df:	49 69 c6 8d 68 b7 6c 	imul   $0x6cb7688d,%r14,%rax
  4013e6:	48 89 c1             	mov    %rax,%rcx
  4013e9:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4013ed:	48 c1 f8 33          	sar    $0x33,%rax
  4013f1:	01 c8                	add    %ecx,%eax
  4013f3:	69 c0 87 d6 12 00    	imul   $0x12d687,%eax,%eax
  4013f9:	41 29 c6             	sub    %eax,%r14d
  4013fc:	48 83 fb 09          	cmp    $0x9,%rbx
  401400:	76 9e                	jbe    4013a0 <main+0x460>
  401402:	48 83 c3 01          	add    $0x1,%rbx
  401406:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  40140d:	75 a4                	jne    4013b3 <main+0x473>
  40140f:	4c 89 e8             	mov    %r13,%rax
  401412:	48 f7 ed             	imul   %rbp
  401415:	48 89 d0             	mov    %rdx,%rax
  401418:	48 c1 e8 3f          	shr    $0x3f,%rax
  40141c:	48 c1 fa 12          	sar    $0x12,%rdx
  401420:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401424:	bf f5 15 40 00       	mov    $0x4015f5,%edi
  401429:	31 c0                	xor    %eax,%eax
  40142b:	4c 89 fe             	mov    %r15,%rsi
  40142e:	4c 89 e2             	mov    %r12,%rdx
  401431:	45 89 f0             	mov    %r14d,%r8d
  401434:	e8 47 f2 ff ff       	callq  400680 <printf@plt>
  401439:	31 c0                	xor    %eax,%eax
  40143b:	48 83 c4 08          	add    $0x8,%rsp
  40143f:	5b                   	pop    %rbx
  401440:	41 5c                	pop    %r12
  401442:	41 5d                	pop    %r13
  401444:	41 5e                	pop    %r14
  401446:	41 5f                	pop    %r15
  401448:	5d                   	pop    %rbp
  401449:	c3                   	retq   
  40144a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401450 <__libc_csu_init>:
  401450:	41 57                	push   %r15
  401452:	41 56                	push   %r14
  401454:	41 89 ff             	mov    %edi,%r15d
  401457:	41 55                	push   %r13
  401459:	41 54                	push   %r12
  40145b:	4c 8d 25 6e 09 20 00 	lea    0x20096e(%rip),%r12        # 601dd0 <__frame_dummy_init_array_entry>
  401462:	55                   	push   %rbp
  401463:	48 8d 2d 7e 09 20 00 	lea    0x20097e(%rip),%rbp        # 601de8 <__init_array_end>
  40146a:	53                   	push   %rbx
  40146b:	49 89 f6             	mov    %rsi,%r14
  40146e:	49 89 d5             	mov    %rdx,%r13
  401471:	4c 29 e5             	sub    %r12,%rbp
  401474:	48 83 ec 08          	sub    $0x8,%rsp
  401478:	48 c1 fd 03          	sar    $0x3,%rbp
  40147c:	e8 cf f1 ff ff       	callq  400650 <_init>
  401481:	48 85 ed             	test   %rbp,%rbp
  401484:	74 20                	je     4014a6 <__libc_csu_init+0x56>
  401486:	31 db                	xor    %ebx,%ebx
  401488:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40148f:	00 
  401490:	4c 89 ea             	mov    %r13,%rdx
  401493:	4c 89 f6             	mov    %r14,%rsi
  401496:	44 89 ff             	mov    %r15d,%edi
  401499:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40149d:	48 83 c3 01          	add    $0x1,%rbx
  4014a1:	48 39 eb             	cmp    %rbp,%rbx
  4014a4:	75 ea                	jne    401490 <__libc_csu_init+0x40>
  4014a6:	48 83 c4 08          	add    $0x8,%rsp
  4014aa:	5b                   	pop    %rbx
  4014ab:	5d                   	pop    %rbp
  4014ac:	41 5c                	pop    %r12
  4014ae:	41 5d                	pop    %r13
  4014b0:	41 5e                	pop    %r14
  4014b2:	41 5f                	pop    %r15
  4014b4:	c3                   	retq   
  4014b5:	90                   	nop
  4014b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4014bd:	00 00 00 

00000000004014c0 <__libc_csu_fini>:
  4014c0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004014c4 <_fini>:
  4014c4:	48 83 ec 08          	sub    $0x8,%rsp
  4014c8:	48 83 c4 08          	add    $0x8,%rsp
  4014cc:	c3                   	retq   
