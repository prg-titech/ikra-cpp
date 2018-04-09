
bin/clang++-5.0_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

0000000000400688 <_init>:
  400688:	48 83 ec 08          	sub    $0x8,%rsp
  40068c:	48 8b 05 65 29 20 00 	mov    0x202965(%rip),%rax        # 602ff8 <_DYNAMIC+0x200>
  400693:	48 85 c0             	test   %rax,%rax
  400696:	74 05                	je     40069d <_init+0x15>
  400698:	e8 b3 00 00 00       	callq  400750 <rand@plt+0x10>
  40069d:	48 83 c4 08          	add    $0x8,%rsp
  4006a1:	c3                   	retq   

Disassembly of section .plt:

00000000004006b0 <printf@plt-0x10>:
  4006b0:	ff 35 52 29 20 00    	pushq  0x202952(%rip)        # 603008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006b6:	ff 25 54 29 20 00    	jmpq   *0x202954(%rip)        # 603010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006c0 <printf@plt>:
  4006c0:	ff 25 52 29 20 00    	jmpq   *0x202952(%rip)        # 603018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006c6:	68 00 00 00 00       	pushq  $0x0
  4006cb:	e9 e0 ff ff ff       	jmpq   4006b0 <_init+0x28>

00000000004006d0 <memset@plt>:
  4006d0:	ff 25 4a 29 20 00    	jmpq   *0x20294a(%rip)        # 603020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006d6:	68 01 00 00 00       	pushq  $0x1
  4006db:	e9 d0 ff ff ff       	jmpq   4006b0 <_init+0x28>

00000000004006e0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006e0:	ff 25 42 29 20 00    	jmpq   *0x202942(%rip)        # 603028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006e6:	68 02 00 00 00       	pushq  $0x2
  4006eb:	e9 c0 ff ff ff       	jmpq   4006b0 <_init+0x28>

00000000004006f0 <__libc_start_main@plt>:
  4006f0:	ff 25 3a 29 20 00    	jmpq   *0x20293a(%rip)        # 603030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006f6:	68 03 00 00 00       	pushq  $0x3
  4006fb:	e9 b0 ff ff ff       	jmpq   4006b0 <_init+0x28>

0000000000400700 <__cxa_atexit@plt>:
  400700:	ff 25 32 29 20 00    	jmpq   *0x202932(%rip)        # 603038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400706:	68 04 00 00 00       	pushq  $0x4
  40070b:	e9 a0 ff ff ff       	jmpq   4006b0 <_init+0x28>

0000000000400710 <_ZNSt8ios_base4InitD1Ev@plt>:
  400710:	ff 25 2a 29 20 00    	jmpq   *0x20292a(%rip)        # 603040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400716:	68 05 00 00 00       	pushq  $0x5
  40071b:	e9 90 ff ff ff       	jmpq   4006b0 <_init+0x28>

0000000000400720 <srand@plt>:
  400720:	ff 25 22 29 20 00    	jmpq   *0x202922(%rip)        # 603048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400726:	68 06 00 00 00       	pushq  $0x6
  40072b:	e9 80 ff ff ff       	jmpq   4006b0 <_init+0x28>

0000000000400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400730:	ff 25 1a 29 20 00    	jmpq   *0x20291a(%rip)        # 603050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400736:	68 07 00 00 00       	pushq  $0x7
  40073b:	e9 70 ff ff ff       	jmpq   4006b0 <_init+0x28>

0000000000400740 <rand@plt>:
  400740:	ff 25 12 29 20 00    	jmpq   *0x202912(%rip)        # 603058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400746:	68 08 00 00 00       	pushq  $0x8
  40074b:	e9 60 ff ff ff       	jmpq   4006b0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400750 <.plt.got>:
  400750:	ff 25 a2 28 20 00    	jmpq   *0x2028a2(%rip)        # 602ff8 <_DYNAMIC+0x200>
  400756:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400760 <set_fast_math>:
  400760:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  400765:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  40076c:	00 
  40076d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  400772:	c3                   	retq   
  400773:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40077a:	00 00 00 
  40077d:	0f 1f 00             	nopl   (%rax)

0000000000400780 <_GLOBAL__sub_I_soa.cc>:
  400780:	50                   	push   %rax
  400781:	bf 80 30 60 00       	mov    $0x603080,%edi
  400786:	e8 55 ff ff ff       	callq  4006e0 <_ZNSt8ios_base4InitC1Ev@plt>
  40078b:	bf 10 07 40 00       	mov    $0x400710,%edi
  400790:	be 80 30 60 00       	mov    $0x603080,%esi
  400795:	ba 68 30 60 00       	mov    $0x603068,%edx
  40079a:	58                   	pop    %rax
  40079b:	e9 60 ff ff ff       	jmpq   400700 <__cxa_atexit@plt>

00000000004007a0 <_start>:
  4007a0:	31 ed                	xor    %ebp,%ebp
  4007a2:	49 89 d1             	mov    %rdx,%r9
  4007a5:	5e                   	pop    %rsi
  4007a6:	48 89 e2             	mov    %rsp,%rdx
  4007a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4007ad:	50                   	push   %rax
  4007ae:	54                   	push   %rsp
  4007af:	49 c7 c0 30 1c 40 00 	mov    $0x401c30,%r8
  4007b6:	48 c7 c1 c0 1b 40 00 	mov    $0x401bc0,%rcx
  4007bd:	48 c7 c7 80 16 40 00 	mov    $0x401680,%rdi
  4007c4:	e8 27 ff ff ff       	callq  4006f0 <__libc_start_main@plt>
  4007c9:	f4                   	hlt    
  4007ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007d0 <deregister_tm_clones>:
  4007d0:	b8 77 30 60 00       	mov    $0x603077,%eax
  4007d5:	55                   	push   %rbp
  4007d6:	48 2d 70 30 60 00    	sub    $0x603070,%rax
  4007dc:	48 83 f8 0e          	cmp    $0xe,%rax
  4007e0:	48 89 e5             	mov    %rsp,%rbp
  4007e3:	76 1b                	jbe    400800 <deregister_tm_clones+0x30>
  4007e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4007ea:	48 85 c0             	test   %rax,%rax
  4007ed:	74 11                	je     400800 <deregister_tm_clones+0x30>
  4007ef:	5d                   	pop    %rbp
  4007f0:	bf 70 30 60 00       	mov    $0x603070,%edi
  4007f5:	ff e0                	jmpq   *%rax
  4007f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007fe:	00 00 
  400800:	5d                   	pop    %rbp
  400801:	c3                   	retq   
  400802:	0f 1f 40 00          	nopl   0x0(%rax)
  400806:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40080d:	00 00 00 

0000000000400810 <register_tm_clones>:
  400810:	be 70 30 60 00       	mov    $0x603070,%esi
  400815:	55                   	push   %rbp
  400816:	48 81 ee 70 30 60 00 	sub    $0x603070,%rsi
  40081d:	48 c1 fe 03          	sar    $0x3,%rsi
  400821:	48 89 e5             	mov    %rsp,%rbp
  400824:	48 89 f0             	mov    %rsi,%rax
  400827:	48 c1 e8 3f          	shr    $0x3f,%rax
  40082b:	48 01 c6             	add    %rax,%rsi
  40082e:	48 d1 fe             	sar    %rsi
  400831:	74 15                	je     400848 <register_tm_clones+0x38>
  400833:	b8 00 00 00 00       	mov    $0x0,%eax
  400838:	48 85 c0             	test   %rax,%rax
  40083b:	74 0b                	je     400848 <register_tm_clones+0x38>
  40083d:	5d                   	pop    %rbp
  40083e:	bf 70 30 60 00       	mov    $0x603070,%edi
  400843:	ff e0                	jmpq   *%rax
  400845:	0f 1f 00             	nopl   (%rax)
  400848:	5d                   	pop    %rbp
  400849:	c3                   	retq   
  40084a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400850 <__do_global_dtors_aux>:
  400850:	80 3d 19 28 20 00 00 	cmpb   $0x0,0x202819(%rip)        # 603070 <__TMC_END__>
  400857:	75 11                	jne    40086a <__do_global_dtors_aux+0x1a>
  400859:	55                   	push   %rbp
  40085a:	48 89 e5             	mov    %rsp,%rbp
  40085d:	e8 6e ff ff ff       	callq  4007d0 <deregister_tm_clones>
  400862:	5d                   	pop    %rbp
  400863:	c6 05 06 28 20 00 01 	movb   $0x1,0x202806(%rip)        # 603070 <__TMC_END__>
  40086a:	f3 c3                	repz retq 
  40086c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400870 <frame_dummy>:
  400870:	bf f0 2d 60 00       	mov    $0x602df0,%edi
  400875:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400879:	75 05                	jne    400880 <frame_dummy+0x10>
  40087b:	eb 93                	jmp    400810 <register_tm_clones>
  40087d:	0f 1f 00             	nopl   (%rax)
  400880:	b8 00 00 00 00       	mov    $0x0,%eax
  400885:	48 85 c0             	test   %rax,%rax
  400888:	74 f1                	je     40087b <frame_dummy+0xb>
  40088a:	55                   	push   %rbp
  40088b:	48 89 e5             	mov    %rsp,%rbp
  40088e:	ff d0                	callq  *%rax
  400890:	5d                   	pop    %rbp
  400891:	e9 7a ff ff ff       	jmpq   400810 <register_tm_clones>
  400896:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089d:	00 00 00 

00000000004008a0 <_Z11r_float2intf>:
  4008a0:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  4008a4:	c3                   	retq   
  4008a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4008ac:	00 00 00 00 

00000000004008b0 <_Z15Body_initializemddddd>:
  4008b0:	c5 fb 11 04 fd 90 30 	vmovsd %xmm0,0x603090(,%rdi,8)
  4008b7:	60 00 
  4008b9:	c5 fb 11 0c fd 90 2a 	vmovsd %xmm1,0x612a90(,%rdi,8)
  4008c0:	61 00 
  4008c2:	c5 fb 11 14 fd 90 24 	vmovsd %xmm2,0x622490(,%rdi,8)
  4008c9:	62 00 
  4008cb:	c5 fb 11 1c fd 90 1e 	vmovsd %xmm3,0x631e90(,%rdi,8)
  4008d2:	63 00 
  4008d4:	c5 fb 11 24 fd 90 18 	vmovsd %xmm4,0x641890(,%rdi,8)
  4008db:	64 00 
  4008dd:	48 c7 04 fd 90 12 65 	movq   $0x0,0x651290(,%rdi,8)
  4008e4:	00 00 00 00 00 
  4008e9:	48 c7 04 fd 90 0c 66 	movq   $0x0,0x660c90(,%rdi,8)
  4008f0:	00 00 00 00 00 
  4008f5:	c3                   	retq   
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_Z14Body_add_forcemm>:
  400900:	48 39 f7             	cmp    %rsi,%rdi
  400903:	0f 84 91 00 00 00    	je     40099a <_Z14Body_add_forcemm+0x9a>
  400909:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  400910:	61 00 
  400912:	c5 fb 5c 04 fd 90 2a 	vsubsd 0x612a90(,%rdi,8),%xmm0,%xmm0
  400919:	61 00 
  40091b:	c5 fb 10 0c f5 90 24 	vmovsd 0x622490(,%rsi,8),%xmm1
  400922:	62 00 
  400924:	c5 f3 5c 0c fd 90 24 	vsubsd 0x622490(,%rdi,8),%xmm1,%xmm1
  40092b:	62 00 
  40092d:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  400931:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  400936:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  40093a:	c5 fb 10 24 fd 90 30 	vmovsd 0x603090(,%rdi,8),%xmm4
  400941:	60 00 
  400943:	c5 db 59 25 fd 12 00 	vmulsd 0x12fd(%rip),%xmm4,%xmm4        # 401c48 <_IO_stdin_used+0x8>
  40094a:	00 
  40094b:	c5 db 59 24 f5 90 30 	vmulsd 0x603090(,%rsi,8),%xmm4,%xmm4
  400952:	60 00 
  400954:	c5 eb 58 15 f4 12 00 	vaddsd 0x12f4(%rip),%xmm2,%xmm2        # 401c50 <_IO_stdin_used+0x10>
  40095b:	00 
  40095c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  400960:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  400964:	c5 fb 10 25 ec 12 00 	vmovsd 0x12ec(%rip),%xmm4        # 401c58 <_IO_stdin_used+0x18>
  40096b:	00 
  40096c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  400970:	c4 e2 e1 a9 04 fd 90 	vfmadd213sd 0x651290(,%rdi,8),%xmm3,%xmm0
  400977:	12 65 00 
  40097a:	c5 fb 11 04 fd 90 12 	vmovsd %xmm0,0x651290(,%rdi,8)
  400981:	65 00 
  400983:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  400987:	c4 e2 e1 a9 04 fd 90 	vfmadd213sd 0x660c90(,%rdi,8),%xmm3,%xmm0
  40098e:	0c 66 00 
  400991:	c5 fb 11 04 fd 90 0c 	vmovsd %xmm0,0x660c90(,%rdi,8)
  400998:	66 00 
  40099a:	c3                   	retq   
  40099b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004009a0 <_Z21Body_add_force_to_allm>:
  4009a0:	48 89 f8             	mov    %rdi,%rax
  4009a3:	48 f7 d8             	neg    %rax
  4009a6:	48 c7 c1 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rcx
  4009ad:	c5 fb 10 05 93 12 00 	vmovsd 0x1293(%rip),%xmm0        # 401c48 <_IO_stdin_used+0x8>
  4009b4:	00 
  4009b5:	c5 fb 10 0d 93 12 00 	vmovsd 0x1293(%rip),%xmm1        # 401c50 <_IO_stdin_used+0x10>
  4009bc:	00 
  4009bd:	c5 fb 10 15 93 12 00 	vmovsd 0x1293(%rip),%xmm2        # 401c58 <_IO_stdin_used+0x18>
  4009c4:	00 
  4009c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4009cc:	00 00 00 00 
  4009d0:	48 8d 14 08          	lea    (%rax,%rcx,1),%rdx
  4009d4:	48 81 fa c0 e0 ff ff 	cmp    $0xffffffffffffe0c0,%rdx
  4009db:	74 7d                	je     400a5a <_Z21Body_add_force_to_allm+0xba>
  4009dd:	c5 fb 10 1c fd 90 2a 	vmovsd 0x612a90(,%rdi,8),%xmm3
  4009e4:	61 00 
  4009e6:	c5 e3 5c 1c cd 90 24 	vsubsd 0x622490(,%rcx,8),%xmm3,%xmm3
  4009ed:	62 00 
  4009ef:	c5 fb 10 24 fd 90 24 	vmovsd 0x622490(,%rdi,8),%xmm4
  4009f6:	62 00 
  4009f8:	c5 db 5c 24 cd 90 1e 	vsubsd 0x631e90(,%rcx,8),%xmm4,%xmm4
  4009ff:	63 00 
  400a01:	c5 e3 59 eb          	vmulsd %xmm3,%xmm3,%xmm5
  400a05:	c4 e2 d9 b9 ec       	vfmadd231sd %xmm4,%xmm4,%xmm5
  400a0a:	c5 d3 51 f5          	vsqrtsd %xmm5,%xmm5,%xmm6
  400a0e:	c5 eb 5e f6          	vdivsd %xmm6,%xmm2,%xmm6
  400a12:	c5 fb 59 3c cd 90 2a 	vmulsd 0x612a90(,%rcx,8),%xmm0,%xmm7
  400a19:	61 00 
  400a1b:	c5 c3 59 3c fd 90 30 	vmulsd 0x603090(,%rdi,8),%xmm7,%xmm7
  400a22:	60 00 
  400a24:	c5 d3 58 e9          	vaddsd %xmm1,%xmm5,%xmm5
  400a28:	c5 c3 5e ed          	vdivsd %xmm5,%xmm7,%xmm5
  400a2c:	c5 d3 59 db          	vmulsd %xmm3,%xmm5,%xmm3
  400a30:	c4 e2 c9 a9 1c cd 90 	vfmadd213sd 0x660c90(,%rcx,8),%xmm6,%xmm3
  400a37:	0c 66 00 
  400a3a:	c5 fb 11 1c cd 90 0c 	vmovsd %xmm3,0x660c90(,%rcx,8)
  400a41:	66 00 
  400a43:	c5 d3 59 dc          	vmulsd %xmm4,%xmm5,%xmm3
  400a47:	c4 e2 c9 a9 1c cd 90 	vfmadd213sd 0x670690(,%rcx,8),%xmm6,%xmm3
  400a4e:	06 67 00 
  400a51:	c5 fb 11 1c cd 90 06 	vmovsd %xmm3,0x670690(,%rcx,8)
  400a58:	67 00 
  400a5a:	48 83 c1 01          	add    $0x1,%rcx
  400a5e:	0f 85 6c ff ff ff    	jne    4009d0 <_Z21Body_add_force_to_allm+0x30>
  400a64:	c3                   	retq   
  400a65:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400a6c:	00 00 00 00 

0000000000400a70 <_Z16Body_reset_forcem>:
  400a70:	48 c7 04 fd 90 12 65 	movq   $0x0,0x651290(,%rdi,8)
  400a77:	00 00 00 00 00 
  400a7c:	48 c7 04 fd 90 0c 66 	movq   $0x0,0x660c90(,%rdi,8)
  400a83:	00 00 00 00 00 
  400a88:	c3                   	retq   
  400a89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a90 <_Z11Body_updatemd>:
  400a90:	c5 fb 59 14 fd 90 12 	vmulsd 0x651290(,%rdi,8),%xmm0,%xmm2
  400a97:	65 00 
  400a99:	c5 fb 10 0d b7 11 00 	vmovsd 0x11b7(%rip),%xmm1        # 401c58 <_IO_stdin_used+0x18>
  400aa0:	00 
  400aa1:	c5 f3 5e 1c fd 90 30 	vdivsd 0x603090(,%rdi,8),%xmm1,%xmm3
  400aa8:	60 00 
  400aaa:	c4 e2 e1 a9 14 fd 90 	vfmadd213sd 0x631e90(,%rdi,8),%xmm3,%xmm2
  400ab1:	1e 63 00 
  400ab4:	c5 fb 11 14 fd 90 1e 	vmovsd %xmm2,0x631e90(,%rdi,8)
  400abb:	63 00 
  400abd:	c5 fb 59 0c fd 90 0c 	vmulsd 0x660c90(,%rdi,8),%xmm0,%xmm1
  400ac4:	66 00 
  400ac6:	c4 e2 e1 a9 0c fd 90 	vfmadd213sd 0x641890(,%rdi,8),%xmm3,%xmm1
  400acd:	18 64 00 
  400ad0:	c5 fb 11 0c fd 90 18 	vmovsd %xmm1,0x641890(,%rdi,8)
  400ad7:	64 00 
  400ad9:	c5 fb 10 1c fd 90 2a 	vmovsd 0x612a90(,%rdi,8),%xmm3
  400ae0:	61 00 
  400ae2:	c4 e2 e9 b9 d8       	vfmadd231sd %xmm0,%xmm2,%xmm3
  400ae7:	c5 fb 11 1c fd 90 2a 	vmovsd %xmm3,0x612a90(,%rdi,8)
  400aee:	61 00 
  400af0:	c4 e2 f1 a9 04 fd 90 	vfmadd213sd 0x622490(,%rdi,8),%xmm1,%xmm0
  400af7:	24 62 00 
  400afa:	c5 fb 11 04 fd 90 24 	vmovsd %xmm0,0x622490(,%rdi,8)
  400b01:	62 00 
  400b03:	c5 f9 2e 1d 55 11 00 	vucomisd 0x1155(%rip),%xmm3        # 401c60 <_IO_stdin_used+0x20>
  400b0a:	00 
  400b0b:	72 0a                	jb     400b17 <_Z11Body_updatemd+0x87>
  400b0d:	c5 f9 2e 1d 43 11 00 	vucomisd 0x1143(%rip),%xmm3        # 401c58 <_IO_stdin_used+0x18>
  400b14:	00 
  400b15:	76 11                	jbe    400b28 <_Z11Body_updatemd+0x98>
  400b17:	c5 e9 57 15 91 11 00 	vxorpd 0x1191(%rip),%xmm2,%xmm2        # 401cb0 <_IO_stdin_used+0x70>
  400b1e:	00 
  400b1f:	c5 f9 13 14 fd 90 1e 	vmovlpd %xmm2,0x631e90(,%rdi,8)
  400b26:	63 00 
  400b28:	c5 f9 2e 05 30 11 00 	vucomisd 0x1130(%rip),%xmm0        # 401c60 <_IO_stdin_used+0x20>
  400b2f:	00 
  400b30:	72 0b                	jb     400b3d <_Z11Body_updatemd+0xad>
  400b32:	c5 f9 2e 05 1e 11 00 	vucomisd 0x111e(%rip),%xmm0        # 401c58 <_IO_stdin_used+0x18>
  400b39:	00 
  400b3a:	77 01                	ja     400b3d <_Z11Body_updatemd+0xad>
  400b3c:	c3                   	retq   
  400b3d:	c5 f1 57 05 6b 11 00 	vxorpd 0x116b(%rip),%xmm1,%xmm0        # 401cb0 <_IO_stdin_used+0x70>
  400b44:	00 
  400b45:	c5 f9 13 04 fd 90 18 	vmovlpd %xmm0,0x641890(,%rdi,8)
  400b4c:	64 00 
  400b4e:	c3                   	retq   
  400b4f:	90                   	nop

0000000000400b50 <_Z26Body_codegen_simple_updatemd>:
  400b50:	c5 fb 10 0d 00 11 00 	vmovsd 0x1100(%rip),%xmm1        # 401c58 <_IO_stdin_used+0x18>
  400b57:	00 
  400b58:	c5 f3 5e 0c fd 90 30 	vdivsd 0x603090(,%rdi,8),%xmm1,%xmm1
  400b5f:	60 00 
  400b61:	c5 fb 12 d0          	vmovddup %xmm0,%xmm2
  400b65:	c5 fb 10 1c fd 90 12 	vmovsd 0x651290(,%rdi,8),%xmm3
  400b6c:	65 00 
  400b6e:	c5 e1 16 1c fd 90 0c 	vmovhpd 0x660c90(,%rdi,8),%xmm3,%xmm3
  400b75:	66 00 
  400b77:	c5 e9 59 d3          	vmulpd %xmm3,%xmm2,%xmm2
  400b7b:	c5 fb 10 24 fd 90 1e 	vmovsd 0x631e90(,%rdi,8),%xmm4
  400b82:	63 00 
  400b84:	c5 fb 10 1c fd 90 18 	vmovsd 0x641890(,%rdi,8),%xmm3
  400b8b:	64 00 
  400b8d:	c5 fb 12 c9          	vmovddup %xmm1,%xmm1
  400b91:	c5 e9 59 e9          	vmulpd %xmm1,%xmm2,%xmm5
  400b95:	c5 59 14 c3          	vunpcklpd %xmm3,%xmm4,%xmm8
  400b99:	c4 62 d1 b8 05 1e 11 	vfmadd231pd 0x111e(%rip),%xmm5,%xmm8        # 401cc0 <_IO_stdin_used+0x80>
  400ba0:	00 00 
  400ba2:	c4 e2 7d 19 f3       	vbroadcastsd %xmm3,%ymm6
  400ba7:	c4 e3 79 05 dd 01    	vpermilpd $0x1,%xmm5,%xmm3
  400bad:	c4 e3 fd 01 d5 55    	vpermpd $0x55,%ymm5,%ymm2
  400bb3:	c5 fd 28 3d 25 11 00 	vmovapd 0x1125(%rip),%ymm7        # 401ce0 <_IO_stdin_used+0xa0>
  400bba:	00 
  400bbb:	c4 e2 ed b8 f7       	vfmadd231pd %ymm7,%ymm2,%ymm6
  400bc0:	c5 fb 10 0d a0 10 00 	vmovsd 0x10a0(%rip),%xmm1        # 401c68 <_IO_stdin_used+0x28>
  400bc7:	00 
  400bc8:	c5 e3 59 d9          	vmulsd %xmm1,%xmm3,%xmm3
  400bcc:	c4 e2 7d 19 db       	vbroadcastsd %xmm3,%ymm3
  400bd1:	c4 62 7d 19 cc       	vbroadcastsd %xmm4,%ymm9
  400bd6:	c4 e2 7d 19 e5       	vbroadcastsd %xmm5,%ymm4
  400bdb:	c4 62 dd b8 cf       	vfmadd231pd %ymm7,%ymm4,%ymm9
  400be0:	c5 d3 59 c9          	vmulsd %xmm1,%xmm5,%xmm1
  400be4:	c4 e2 7d 19 e9       	vbroadcastsd %xmm1,%ymm5
  400be9:	c4 e2 7d 19 c0       	vbroadcastsd %xmm0,%ymm0
  400bee:	c5 fb 10 0c fd 90 2a 	vmovsd 0x612a90(,%rdi,8),%xmm1
  400bf5:	61 00 
  400bf7:	c5 fb 10 3c fd 90 24 	vmovsd 0x622490(,%rdi,8),%xmm7
  400bfe:	62 00 
  400c00:	c5 35 58 d4          	vaddpd %ymm4,%ymm9,%ymm10
  400c04:	c5 6d 58 de          	vaddpd %ymm6,%ymm2,%ymm11
  400c08:	c4 62 fd a8 d1       	vfmadd213pd %ymm1,%ymm0,%ymm10
  400c0d:	c4 62 fd a8 df       	vfmadd213pd %ymm7,%ymm0,%ymm11
  400c12:	c5 cd 58 cb          	vaddpd %ymm3,%ymm6,%ymm1
  400c16:	c5 b5 58 f5          	vaddpd %ymm5,%ymm9,%ymm6
  400c1a:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400c1e:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400c22:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400c27:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400c2c:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400c30:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400c34:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400c38:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400c3c:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400c41:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400c46:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400c4a:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400c4e:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400c52:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400c56:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400c5b:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400c60:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400c64:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400c68:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400c6c:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400c70:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400c75:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400c7a:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400c7e:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400c82:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400c86:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400c8a:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400c8f:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400c94:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400c98:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400c9c:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400ca0:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400ca4:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400ca9:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400cae:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400cb2:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400cb6:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400cba:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400cbe:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400cc3:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400cc8:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400ccc:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400cd0:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400cd4:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400cd8:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400cdd:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400ce2:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400ce6:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400cea:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400cee:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400cf2:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400cf7:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400cfc:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d00:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d04:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400d08:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400d0c:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400d11:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400d16:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d1a:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d1e:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400d22:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400d26:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400d2b:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400d30:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d34:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d38:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400d3c:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400d40:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400d45:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400d4a:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d4e:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d52:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400d56:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400d5a:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400d5f:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400d64:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d68:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d6c:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400d70:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400d74:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400d79:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400d7e:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d82:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400d86:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400d8a:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400d8e:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400d93:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400d98:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400d9c:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400da0:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400da4:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400da8:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400dad:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400db2:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400db6:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400dba:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400dbe:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400dc2:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400dc7:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400dcc:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400dd0:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400dd4:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400dd8:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400ddc:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400de1:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400de6:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400dea:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400dee:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400df2:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400df6:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400dfb:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400e00:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400e04:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400e08:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400e0c:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400e10:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400e15:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400e1a:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400e1e:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400e22:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400e26:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400e2a:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400e2f:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400e34:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400e38:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400e3c:	c5 4d 58 d4          	vaddpd %ymm4,%ymm6,%ymm10
  400e40:	c5 6d 58 d9          	vaddpd %ymm1,%ymm2,%ymm11
  400e44:	c4 62 fd a8 d7       	vfmadd213pd %ymm7,%ymm0,%ymm10
  400e49:	c4 42 fd a8 d9       	vfmadd213pd %ymm9,%ymm0,%ymm11
  400e4e:	c5 f5 58 cb          	vaddpd %ymm3,%ymm1,%ymm1
  400e52:	c5 cd 58 f5          	vaddpd %ymm5,%ymm6,%ymm6
  400e56:	c5 cd 58 fc          	vaddpd %ymm4,%ymm6,%ymm7
  400e5a:	c5 6d 58 c9          	vaddpd %ymm1,%ymm2,%ymm9
  400e5e:	c4 c2 fd a8 fa       	vfmadd213pd %ymm10,%ymm0,%ymm7
  400e63:	c4 42 fd a8 cb       	vfmadd213pd %ymm11,%ymm0,%ymm9
  400e68:	c5 d5 58 e4          	vaddpd %ymm4,%ymm5,%ymm4
  400e6c:	c5 cd 58 e4          	vaddpd %ymm4,%ymm6,%ymm4
  400e70:	c5 e5 58 d2          	vaddpd %ymm2,%ymm3,%ymm2
  400e74:	c5 f5 58 ca          	vaddpd %ymm2,%ymm1,%ymm1
  400e78:	c4 e2 fd a8 e7       	vfmadd213pd %ymm7,%ymm0,%ymm4
  400e7d:	c4 c2 fd a8 c9       	vfmadd213pd %ymm9,%ymm0,%ymm1
  400e82:	c4 e3 7d 19 e0 01    	vextractf128 $0x1,%ymm4,%xmm0
  400e88:	c5 dd 58 c0          	vaddpd %ymm0,%ymm4,%ymm0
  400e8c:	c5 fd 7c c0          	vhaddpd %ymm0,%ymm0,%ymm0
  400e90:	c4 e3 7d 19 ca 01    	vextractf128 $0x1,%ymm1,%xmm2
  400e96:	c5 f5 58 ca          	vaddpd %ymm2,%ymm1,%ymm1
  400e9a:	c5 f5 7c c9          	vhaddpd %ymm1,%ymm1,%ymm1
  400e9e:	c5 79 13 04 fd 90 1e 	vmovlpd %xmm8,0x631e90(,%rdi,8)
  400ea5:	63 00 
  400ea7:	c5 79 17 04 fd 90 18 	vmovhpd %xmm8,0x641890(,%rdi,8)
  400eae:	64 00 
  400eb0:	c5 f9 13 04 fd 90 2a 	vmovlpd %xmm0,0x612a90(,%rdi,8)
  400eb7:	61 00 
  400eb9:	c5 f9 13 0c fd 90 24 	vmovlpd %xmm1,0x622490(,%rdi,8)
  400ec0:	62 00 
  400ec2:	c5 f8 77             	vzeroupper 
  400ec5:	c3                   	retq   
  400ec6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ecd:	00 00 00 

0000000000400ed0 <_Z13instantiationv>:
  400ed0:	53                   	push   %rbx
  400ed1:	48 83 ec 20          	sub    $0x20,%rsp
  400ed5:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400eda:	e8 41 f8 ff ff       	callq  400720 <srand@plt>
  400edf:	48 c7 c3 00 06 ff ff 	mov    $0xffffffffffff0600,%rbx
  400ee6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400eed:	00 00 00 
  400ef0:	e8 4b f8 ff ff       	callq  400740 <rand@plt>
  400ef5:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  400ef9:	c5 fb 10 05 77 0d 00 	vmovsd 0xd77(%rip),%xmm0        # 401c78 <_IO_stdin_used+0x38>
  400f00:	00 
  400f01:	c4 e2 f9 99 0d 66 0d 	vfmadd132sd 0xd66(%rip),%xmm0,%xmm1        # 401c70 <_IO_stdin_used+0x30>
  400f08:	00 00 
  400f0a:	c5 fb 11 4c 24 18    	vmovsd %xmm1,0x18(%rsp)
  400f10:	e8 2b f8 ff ff       	callq  400740 <rand@plt>
  400f15:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  400f19:	c5 fb 10 05 3f 0d 00 	vmovsd 0xd3f(%rip),%xmm0        # 401c60 <_IO_stdin_used+0x20>
  400f20:	00 
  400f21:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400f25:	c5 fb 10 05 53 0d 00 	vmovsd 0xd53(%rip),%xmm0        # 401c80 <_IO_stdin_used+0x40>
  400f2c:	00 
  400f2d:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400f32:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  400f38:	e8 03 f8 ff ff       	callq  400740 <rand@plt>
  400f3d:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  400f41:	c5 fb 10 0d 17 0d 00 	vmovsd 0xd17(%rip),%xmm1        # 401c60 <_IO_stdin_used+0x20>
  400f48:	00 
  400f49:	c4 e2 f1 99 05 2e 0d 	vfmadd132sd 0xd2e(%rip),%xmm1,%xmm0        # 401c80 <_IO_stdin_used+0x40>
  400f50:	00 00 
  400f52:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
  400f58:	e8 e3 f7 ff ff       	callq  400740 <rand@plt>
  400f5d:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  400f61:	c5 fb 10 05 1f 0d 00 	vmovsd 0xd1f(%rip),%xmm0        # 401c88 <_IO_stdin_used+0x48>
  400f68:	00 
  400f69:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400f6d:	c5 fb 10 05 1b 0d 00 	vmovsd 0xd1b(%rip),%xmm0        # 401c90 <_IO_stdin_used+0x50>
  400f74:	00 
  400f75:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400f7a:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  400f7f:	e8 bc f7 ff ff       	callq  400740 <rand@plt>
  400f84:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  400f88:	c5 fb 10 0d f8 0c 00 	vmovsd 0xcf8(%rip),%xmm1        # 401c88 <_IO_stdin_used+0x48>
  400f8f:	00 
  400f90:	c4 e2 f1 99 05 f7 0c 	vfmadd132sd 0xcf7(%rip),%xmm1,%xmm0        # 401c90 <_IO_stdin_used+0x50>
  400f97:	00 00 
  400f99:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  400f9f:	c5 fb 11 8b 90 2a 61 	vmovsd %xmm1,0x612a90(%rbx)
  400fa6:	00 
  400fa7:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  400fad:	c5 fb 11 8b 90 24 62 	vmovsd %xmm1,0x622490(%rbx)
  400fb4:	00 
  400fb5:	c5 fb 10 4c 24 08    	vmovsd 0x8(%rsp),%xmm1
  400fbb:	c5 fb 11 8b 90 1e 63 	vmovsd %xmm1,0x631e90(%rbx)
  400fc2:	00 
  400fc3:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  400fc8:	c5 fb 11 8b 90 18 64 	vmovsd %xmm1,0x641890(%rbx)
  400fcf:	00 
  400fd0:	c5 fb 11 83 90 12 65 	vmovsd %xmm0,0x651290(%rbx)
  400fd7:	00 
  400fd8:	48 c7 83 90 0c 66 00 	movq   $0x0,0x660c90(%rbx)
  400fdf:	00 00 00 00 
  400fe3:	48 c7 83 90 06 67 00 	movq   $0x0,0x670690(%rbx)
  400fea:	00 00 00 00 
  400fee:	48 83 c3 08          	add    $0x8,%rbx
  400ff2:	0f 85 f8 fe ff ff    	jne    400ef0 <_Z13instantiationv+0x20>
  400ff8:	48 83 c4 20          	add    $0x20,%rsp
  400ffc:	5b                   	pop    %rbx
  400ffd:	c3                   	retq   
  400ffe:	66 90                	xchg   %ax,%ax

0000000000401000 <_Z14run_simulationv>:
  401000:	53                   	push   %rbx
  401001:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  401008:	31 db                	xor    %ebx,%ebx
  40100a:	c4 e2 7d 19 3d 85 0c 	vbroadcastsd 0xc85(%rip),%ymm7        # 401c98 <_IO_stdin_used+0x58>
  401011:	00 00 
  401013:	c4 62 7d 19 05 3c 0c 	vbroadcastsd 0xc3c(%rip),%ymm8        # 401c58 <_IO_stdin_used+0x18>
  40101a:	00 00 
  40101c:	c4 62 7d 19 0d 3b 0c 	vbroadcastsd 0xc3b(%rip),%ymm9        # 401c60 <_IO_stdin_used+0x20>
  401023:	00 00 
  401025:	c4 62 7d 19 15 72 0c 	vbroadcastsd 0xc72(%rip),%ymm10        # 401ca0 <_IO_stdin_used+0x60>
  40102c:	00 00 
  40102e:	c5 fd 11 7c 24 60    	vmovupd %ymm7,0x60(%rsp)
  401034:	c5 7d 11 44 24 40    	vmovupd %ymm8,0x40(%rsp)
  40103a:	c5 7d 11 4c 24 20    	vmovupd %ymm9,0x20(%rsp)
  401040:	c5 7d 11 14 24       	vmovupd %ymm10,(%rsp)
  401045:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40104c:	00 00 00 00 
  401050:	bf 90 12 65 00       	mov    $0x651290,%edi
  401055:	31 f6                	xor    %esi,%esi
  401057:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  40105c:	c5 f8 77             	vzeroupper 
  40105f:	e8 6c f6 ff ff       	callq  4006d0 <memset@plt>
  401064:	bf 90 0c 66 00       	mov    $0x660c90,%edi
  401069:	31 f6                	xor    %esi,%esi
  40106b:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  401070:	e8 5b f6 ff ff       	callq  4006d0 <memset@plt>
  401075:	31 c0                	xor    %eax,%eax
  401077:	31 c9                	xor    %ecx,%ecx
  401079:	c5 fb 10 2d c7 0b 00 	vmovsd 0xbc7(%rip),%xmm5        # 401c48 <_IO_stdin_used+0x8>
  401080:	00 
  401081:	c5 fb 10 35 c7 0b 00 	vmovsd 0xbc7(%rip),%xmm6        # 401c50 <_IO_stdin_used+0x10>
  401088:	00 
  401089:	c5 fb 10 3d c7 0b 00 	vmovsd 0xbc7(%rip),%xmm7        # 401c58 <_IO_stdin_used+0x18>
  401090:	00 
  401091:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401098:	0f 1f 84 00 00 00 00 
  40109f:	00 
  4010a0:	48 c7 c2 00 06 ff ff 	mov    $0xffffffffffff0600,%rdx
  4010a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4010ae:	00 00 
  4010b0:	48 8d 34 10          	lea    (%rax,%rdx,1),%rsi
  4010b4:	48 81 fe 00 06 ff ff 	cmp    $0xffffffffffff0600,%rsi
  4010bb:	74 76                	je     401133 <_Z14run_simulationv+0x133>
  4010bd:	c5 fb 10 04 cd 90 2a 	vmovsd 0x612a90(,%rcx,8),%xmm0
  4010c4:	61 00 
  4010c6:	c5 fb 5c 82 90 24 62 	vsubsd 0x622490(%rdx),%xmm0,%xmm0
  4010cd:	00 
  4010ce:	c5 fb 10 0c cd 90 24 	vmovsd 0x622490(,%rcx,8),%xmm1
  4010d5:	62 00 
  4010d7:	c5 f3 5c 8a 90 1e 63 	vsubsd 0x631e90(%rdx),%xmm1,%xmm1
  4010de:	00 
  4010df:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  4010e3:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  4010e8:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  4010ec:	c5 c3 5e db          	vdivsd %xmm3,%xmm7,%xmm3
  4010f0:	c5 d3 59 a2 90 2a 61 	vmulsd 0x612a90(%rdx),%xmm5,%xmm4
  4010f7:	00 
  4010f8:	c5 db 59 24 cd 90 30 	vmulsd 0x603090(,%rcx,8),%xmm4,%xmm4
  4010ff:	60 00 
  401101:	c5 eb 58 d6          	vaddsd %xmm6,%xmm2,%xmm2
  401105:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401109:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  40110d:	c4 e2 e1 a9 82 90 0c 	vfmadd213sd 0x660c90(%rdx),%xmm3,%xmm0
  401114:	66 00 
  401116:	c5 fb 11 82 90 0c 66 	vmovsd %xmm0,0x660c90(%rdx)
  40111d:	00 
  40111e:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401122:	c4 e2 e1 a9 82 90 06 	vfmadd213sd 0x670690(%rdx),%xmm3,%xmm0
  401129:	67 00 
  40112b:	c5 fb 11 82 90 06 67 	vmovsd %xmm0,0x670690(%rdx)
  401132:	00 
  401133:	48 83 c2 08          	add    $0x8,%rdx
  401137:	0f 85 73 ff ff ff    	jne    4010b0 <_Z14run_simulationv+0xb0>
  40113d:	48 83 c1 01          	add    $0x1,%rcx
  401141:	48 83 c0 f8          	add    $0xfffffffffffffff8,%rax
  401145:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  40114c:	0f 85 4e ff ff ff    	jne    4010a0 <_Z14run_simulationv+0xa0>
  401152:	31 c0                	xor    %eax,%eax
  401154:	c5 fd 10 7c 24 60    	vmovupd 0x60(%rsp),%ymm7
  40115a:	c5 7d 10 44 24 40    	vmovupd 0x40(%rsp),%ymm8
  401160:	c5 7d 10 4c 24 20    	vmovupd 0x20(%rsp),%ymm9
  401166:	c5 7d 10 14 24       	vmovupd (%rsp),%ymm10
  40116b:	c4 41 21 76 db       	vpcmpeqd %xmm11,%xmm11,%xmm11
  401170:	c5 bd 5e 80 90 30 60 	vdivpd 0x603090(%rax),%ymm8,%ymm0
  401177:	00 
  401178:	c5 c5 59 90 90 12 65 	vmulpd 0x651290(%rax),%ymm7,%ymm2
  40117f:	00 
  401180:	c4 e2 fd a8 90 90 1e 	vfmadd213pd 0x631e90(%rax),%ymm0,%ymm2
  401187:	63 00 
  401189:	c5 fd 11 90 90 1e 63 	vmovupd %ymm2,0x631e90(%rax)
  401190:	00 
  401191:	c5 c5 59 88 90 0c 66 	vmulpd 0x660c90(%rax),%ymm7,%ymm1
  401198:	00 
  401199:	c4 e2 fd a8 88 90 18 	vfmadd213pd 0x641890(%rax),%ymm0,%ymm1
  4011a0:	64 00 
  4011a2:	c5 fd 11 88 90 18 64 	vmovupd %ymm1,0x641890(%rax)
  4011a9:	00 
  4011aa:	c5 fd 10 80 90 2a 61 	vmovupd 0x612a90(%rax),%ymm0
  4011b1:	00 
  4011b2:	c4 e2 ed b8 c7       	vfmadd231pd %ymm7,%ymm2,%ymm0
  4011b7:	c4 c1 7d c2 d9 01    	vcmpltpd %ymm9,%ymm0,%ymm3
  4011bd:	c4 e3 7d 19 dc 01    	vextractf128 $0x1,%ymm3,%xmm4
  4011c3:	c5 e1 63 e4          	vpacksswb %xmm4,%xmm3,%xmm4
  4011c7:	c5 bd c2 e8 01       	vcmpltpd %ymm0,%ymm8,%ymm5
  4011cc:	c4 e3 7d 19 ee 01    	vextractf128 $0x1,%ymm5,%xmm6
  4011d2:	c5 d1 63 f6          	vpacksswb %xmm6,%xmm5,%xmm6
  4011d6:	c5 d9 eb e6          	vpor   %xmm6,%xmm4,%xmm4
  4011da:	c5 fd 10 b0 90 24 62 	vmovupd 0x622490(%rax),%ymm6
  4011e1:	00 
  4011e2:	c4 e2 f5 b8 f7       	vfmadd231pd %ymm7,%ymm1,%ymm6
  4011e7:	c5 e5 56 dd          	vorpd  %ymm5,%ymm3,%ymm3
  4011eb:	c4 c1 6d 57 d2       	vxorpd %ymm10,%ymm2,%ymm2
  4011f0:	c4 e2 65 2f 90 90 1e 	vmaskmovpd %ymm2,%ymm3,0x631e90(%rax)
  4011f7:	63 00 
  4011f9:	c4 c1 4d c2 d1 01    	vcmpltpd %ymm9,%ymm6,%ymm2
  4011ff:	c4 e3 7d 19 d3 01    	vextractf128 $0x1,%ymm2,%xmm3
  401205:	c5 e9 63 d3          	vpacksswb %xmm3,%xmm2,%xmm2
  401209:	c5 bd c2 de 01       	vcmpltpd %ymm6,%ymm8,%ymm3
  40120e:	c4 e3 7d 19 dd 01    	vextractf128 $0x1,%ymm3,%xmm5
  401214:	c5 e1 63 dd          	vpacksswb %xmm5,%xmm3,%xmm3
  401218:	c5 e9 eb d3          	vpor   %xmm3,%xmm2,%xmm2
  40121c:	c4 c1 75 57 ca       	vxorpd %ymm10,%ymm1,%ymm1
  401221:	c4 c1 59 ef db       	vpxor  %xmm11,%xmm4,%xmm3
  401226:	c5 d9 eb db          	vpor   %xmm3,%xmm4,%xmm3
  40122a:	c5 e9 db d3          	vpand  %xmm3,%xmm2,%xmm2
  40122e:	c4 e2 7d 25 d2       	vpmovsxdq %xmm2,%ymm2
  401233:	c4 e2 6d 2f 88 90 18 	vmaskmovpd %ymm1,%ymm2,0x641890(%rax)
  40123a:	64 00 
  40123c:	c5 fd 11 80 90 2a 61 	vmovupd %ymm0,0x612a90(%rax)
  401243:	00 
  401244:	c5 fd 11 b0 90 24 62 	vmovupd %ymm6,0x622490(%rax)
  40124b:	00 
  40124c:	48 83 c0 20          	add    $0x20,%rax
  401250:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  401256:	0f 85 14 ff ff ff    	jne    401170 <_Z14run_simulationv+0x170>
  40125c:	83 c3 01             	add    $0x1,%ebx
  40125f:	83 fb 05             	cmp    $0x5,%ebx
  401262:	0f 85 e8 fd ff ff    	jne    401050 <_Z14run_simulationv+0x50>
  401268:	48 81 c4 90 00 00 00 	add    $0x90,%rsp
  40126f:	5b                   	pop    %rbx
  401270:	c5 f8 77             	vzeroupper 
  401273:	c3                   	retq   
  401274:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40127b:	00 00 00 00 00 

0000000000401280 <_Z10run_simplev>:
  401280:	31 c0                	xor    %eax,%eax
  401282:	c5 7b 10 1d ce 09 00 	vmovsd 0x9ce(%rip),%xmm11        # 401c58 <_IO_stdin_used+0x18>
  401289:	00 
  40128a:	c5 79 28 25 3e 0a 00 	vmovapd 0xa3e(%rip),%xmm12        # 401cd0 <_IO_stdin_used+0x90>
  401291:	00 
  401292:	c5 79 28 2d 26 0a 00 	vmovapd 0xa26(%rip),%xmm13        # 401cc0 <_IO_stdin_used+0x80>
  401299:	00 
  40129a:	c5 fd 28 1d 3e 0a 00 	vmovapd 0xa3e(%rip),%ymm3        # 401ce0 <_IO_stdin_used+0xa0>
  4012a1:	00 
  4012a2:	c5 fb 10 25 be 09 00 	vmovsd 0x9be(%rip),%xmm4        # 401c68 <_IO_stdin_used+0x28>
  4012a9:	00 
  4012aa:	c4 e2 7d 19 2d e5 09 	vbroadcastsd 0x9e5(%rip),%ymm5        # 401c98 <_IO_stdin_used+0x58>
  4012b1:	00 00 
  4012b3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4012ba:	84 00 00 00 00 00 
  4012c0:	48 c7 c1 00 06 ff ff 	mov    $0xffffffffffff0600,%rcx
  4012c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4012ce:	00 00 
  4012d0:	c5 a3 5e b1 90 2a 61 	vdivsd 0x612a90(%rcx),%xmm11,%xmm6
  4012d7:	00 
  4012d8:	c5 fb 10 b9 90 0c 66 	vmovsd 0x660c90(%rcx),%xmm7
  4012df:	00 
  4012e0:	c5 c1 16 b9 90 06 67 	vmovhpd 0x670690(%rcx),%xmm7,%xmm7
  4012e7:	00 
  4012e8:	c4 c1 41 59 fc       	vmulpd %xmm12,%xmm7,%xmm7
  4012ed:	c5 fb 10 81 90 18 64 	vmovsd 0x641890(%rcx),%xmm0
  4012f4:	00 
  4012f5:	c5 fb 10 89 90 12 65 	vmovsd 0x651290(%rcx),%xmm1
  4012fc:	00 
  4012fd:	c5 fb 12 f6          	vmovddup %xmm6,%xmm6
  401301:	c5 41 59 d6          	vmulpd %xmm6,%xmm7,%xmm10
  401305:	c5 f9 14 f1          	vunpcklpd %xmm1,%xmm0,%xmm6
  401309:	c4 c2 a9 b8 f5       	vfmadd231pd %xmm13,%xmm10,%xmm6
  40130e:	c4 e2 7d 19 c9       	vbroadcastsd %xmm1,%ymm1
  401313:	c4 c3 79 05 d2 01    	vpermilpd $0x1,%xmm10,%xmm2
  401319:	c4 c3 fd 01 fa 55    	vpermpd $0x55,%ymm10,%ymm7
  40131f:	c4 e2 c5 b8 cb       	vfmadd231pd %ymm3,%ymm7,%ymm1
  401324:	c5 eb 59 d4          	vmulsd %xmm4,%xmm2,%xmm2
  401328:	c4 62 7d 19 c2       	vbroadcastsd %xmm2,%ymm8
  40132d:	c4 e2 7d 19 c0       	vbroadcastsd %xmm0,%ymm0
  401332:	c4 42 7d 19 ca       	vbroadcastsd %xmm10,%ymm9
  401337:	c4 e2 b5 b8 c3       	vfmadd231pd %ymm3,%ymm9,%ymm0
  40133c:	c5 ab 59 d4          	vmulsd %xmm4,%xmm10,%xmm2
  401340:	c4 62 7d 19 d2       	vbroadcastsd %xmm2,%ymm10
  401345:	c5 fb 10 91 90 24 62 	vmovsd 0x622490(%rcx),%xmm2
  40134c:	00 
  40134d:	c5 7b 10 b1 90 1e 63 	vmovsd 0x631e90(%rcx),%xmm14
  401354:	00 
  401355:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40135a:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40135f:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401363:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401368:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40136d:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401372:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401377:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40137c:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401380:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401385:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40138a:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40138f:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401394:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401399:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  40139d:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4013a2:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4013a7:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4013ac:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4013b1:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4013b6:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4013ba:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4013bf:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4013c4:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4013c9:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4013ce:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4013d3:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4013d7:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4013dc:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4013e1:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4013e6:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4013eb:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4013f0:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4013f4:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4013f9:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4013fe:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401403:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401408:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40140d:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401411:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401416:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40141b:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401420:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401425:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40142a:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40142e:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401433:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401438:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40143d:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401442:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401447:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  40144b:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401450:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401455:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40145a:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40145f:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401464:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401468:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40146d:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401472:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401477:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  40147c:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401481:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401485:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40148a:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40148f:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401494:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401499:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40149e:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4014a2:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4014a7:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4014ac:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4014b1:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4014b6:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4014bb:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4014bf:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4014c4:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4014c9:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4014ce:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4014d3:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4014d8:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4014dc:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4014e1:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4014e6:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4014eb:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4014f0:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4014f5:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4014f9:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4014fe:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401503:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401508:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40150d:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401512:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401516:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40151b:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401520:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401525:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  40152a:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40152f:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401533:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401538:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40153d:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401542:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401547:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40154c:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  401550:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401555:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40155a:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40155f:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401564:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401569:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  40156d:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401572:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401577:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40157c:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401581:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401586:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40158a:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40158f:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401594:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401599:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  40159e:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4015a3:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4015a7:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4015ac:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4015b1:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4015b6:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4015bb:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4015c0:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4015c4:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4015c9:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4015ce:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4015d3:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4015d8:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4015dd:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4015e1:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4015e6:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4015eb:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4015f0:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4015f5:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4015fa:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4015fe:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401603:	c4 41 2d 58 c9       	vaddpd %ymm9,%ymm10,%ymm9
  401608:	c4 c1 7d 58 c1       	vaddpd %ymm9,%ymm0,%ymm0
  40160d:	c5 bd 58 ff          	vaddpd %ymm7,%ymm8,%ymm7
  401611:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  401615:	c4 e2 d5 a8 c2       	vfmadd213pd %ymm2,%ymm5,%ymm0
  40161a:	c4 c2 d5 a8 ce       	vfmadd213pd %ymm14,%ymm5,%ymm1
  40161f:	c4 e3 7d 19 c2 01    	vextractf128 $0x1,%ymm0,%xmm2
  401625:	c5 fd 58 c2          	vaddpd %ymm2,%ymm0,%ymm0
  401629:	c4 e3 7d 19 ca 01    	vextractf128 $0x1,%ymm1,%xmm2
  40162f:	c5 f5 58 ca          	vaddpd %ymm2,%ymm1,%ymm1
  401633:	c5 fd 7c c0          	vhaddpd %ymm0,%ymm0,%ymm0
  401637:	c5 f5 7c c9          	vhaddpd %ymm1,%ymm1,%ymm1
  40163b:	c5 f9 13 b1 90 18 64 	vmovlpd %xmm6,0x641890(%rcx)
  401642:	00 
  401643:	c5 f9 17 b1 90 12 65 	vmovhpd %xmm6,0x651290(%rcx)
  40164a:	00 
  40164b:	c5 f9 13 81 90 24 62 	vmovlpd %xmm0,0x622490(%rcx)
  401652:	00 
  401653:	c5 f9 13 89 90 1e 63 	vmovlpd %xmm1,0x631e90(%rcx)
  40165a:	00 
  40165b:	48 83 c1 08          	add    $0x8,%rcx
  40165f:	0f 85 6b fc ff ff    	jne    4012d0 <_Z10run_simplev+0x50>
  401665:	83 c0 01             	add    $0x1,%eax
  401668:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  40166d:	0f 85 4d fc ff ff    	jne    4012c0 <_Z10run_simplev+0x40>
  401673:	c5 f8 77             	vzeroupper 
  401676:	c3                   	retq   
  401677:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40167e:	00 00 

0000000000401680 <main>:
  401680:	55                   	push   %rbp
  401681:	41 57                	push   %r15
  401683:	41 56                	push   %r14
  401685:	41 55                	push   %r13
  401687:	41 54                	push   %r12
  401689:	53                   	push   %rbx
  40168a:	50                   	push   %rax
  40168b:	e8 a0 f0 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401690:	48 89 c3             	mov    %rax,%rbx
  401693:	e8 38 f8 ff ff       	callq  400ed0 <_Z13instantiationv>
  401698:	e8 93 f0 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40169d:	49 89 c7             	mov    %rax,%r15
  4016a0:	49 29 df             	sub    %rbx,%r15
  4016a3:	e8 88 f0 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4016a8:	48 89 c3             	mov    %rax,%rbx
  4016ab:	e8 50 f9 ff ff       	callq  401000 <_Z14run_simulationv>
  4016b0:	e8 7b f0 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4016b5:	49 89 c4             	mov    %rax,%r12
  4016b8:	49 29 dc             	sub    %rbx,%r12
  4016bb:	31 db                	xor    %ebx,%ebx
  4016bd:	e8 6e f0 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4016c2:	49 89 c6             	mov    %rax,%r14
  4016c5:	c5 7b 10 1d 8b 05 00 	vmovsd 0x58b(%rip),%xmm11        # 401c58 <_IO_stdin_used+0x18>
  4016cc:	00 
  4016cd:	c5 79 28 25 fb 05 00 	vmovapd 0x5fb(%rip),%xmm12        # 401cd0 <_IO_stdin_used+0x90>
  4016d4:	00 
  4016d5:	c5 79 28 2d e3 05 00 	vmovapd 0x5e3(%rip),%xmm13        # 401cc0 <_IO_stdin_used+0x80>
  4016dc:	00 
  4016dd:	c5 fd 28 1d fb 05 00 	vmovapd 0x5fb(%rip),%ymm3        # 401ce0 <_IO_stdin_used+0xa0>
  4016e4:	00 
  4016e5:	c5 fb 10 25 7b 05 00 	vmovsd 0x57b(%rip),%xmm4        # 401c68 <_IO_stdin_used+0x28>
  4016ec:	00 
  4016ed:	c4 e2 7d 19 2d a2 05 	vbroadcastsd 0x5a2(%rip),%ymm5        # 401c98 <_IO_stdin_used+0x58>
  4016f4:	00 00 
  4016f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4016fd:	00 00 00 
  401700:	48 c7 c0 00 06 ff ff 	mov    $0xffffffffffff0600,%rax
  401707:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40170e:	00 00 
  401710:	c5 a3 5e b0 90 2a 61 	vdivsd 0x612a90(%rax),%xmm11,%xmm6
  401717:	00 
  401718:	c5 fb 10 b8 90 0c 66 	vmovsd 0x660c90(%rax),%xmm7
  40171f:	00 
  401720:	c5 c1 16 b8 90 06 67 	vmovhpd 0x670690(%rax),%xmm7,%xmm7
  401727:	00 
  401728:	c4 c1 41 59 fc       	vmulpd %xmm12,%xmm7,%xmm7
  40172d:	c5 fb 10 80 90 18 64 	vmovsd 0x641890(%rax),%xmm0
  401734:	00 
  401735:	c5 fb 10 88 90 12 65 	vmovsd 0x651290(%rax),%xmm1
  40173c:	00 
  40173d:	c5 fb 12 f6          	vmovddup %xmm6,%xmm6
  401741:	c5 41 59 d6          	vmulpd %xmm6,%xmm7,%xmm10
  401745:	c5 f9 14 f1          	vunpcklpd %xmm1,%xmm0,%xmm6
  401749:	c4 c2 a9 b8 f5       	vfmadd231pd %xmm13,%xmm10,%xmm6
  40174e:	c4 e2 7d 19 c9       	vbroadcastsd %xmm1,%ymm1
  401753:	c4 c3 79 05 d2 01    	vpermilpd $0x1,%xmm10,%xmm2
  401759:	c4 c3 fd 01 fa 55    	vpermpd $0x55,%ymm10,%ymm7
  40175f:	c4 e2 c5 b8 cb       	vfmadd231pd %ymm3,%ymm7,%ymm1
  401764:	c5 eb 59 d4          	vmulsd %xmm4,%xmm2,%xmm2
  401768:	c4 62 7d 19 c2       	vbroadcastsd %xmm2,%ymm8
  40176d:	c4 e2 7d 19 c0       	vbroadcastsd %xmm0,%ymm0
  401772:	c4 42 7d 19 ca       	vbroadcastsd %xmm10,%ymm9
  401777:	c4 e2 b5 b8 c3       	vfmadd231pd %ymm3,%ymm9,%ymm0
  40177c:	c5 ab 59 d4          	vmulsd %xmm4,%xmm10,%xmm2
  401780:	c4 62 7d 19 d2       	vbroadcastsd %xmm2,%ymm10
  401785:	c5 fb 10 90 90 24 62 	vmovsd 0x622490(%rax),%xmm2
  40178c:	00 
  40178d:	c5 7b 10 b0 90 1e 63 	vmovsd 0x631e90(%rax),%xmm14
  401794:	00 
  401795:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40179a:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40179f:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4017a3:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4017a8:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4017ad:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4017b2:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4017b7:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4017bc:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4017c0:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4017c5:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4017ca:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4017cf:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4017d4:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4017d9:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4017dd:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4017e2:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4017e7:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4017ec:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4017f1:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4017f6:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4017fa:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4017ff:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401804:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401809:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  40180e:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401813:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401817:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40181c:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401821:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401826:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  40182b:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401830:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  401834:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401839:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40183e:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401843:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  401848:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  40184d:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401851:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401856:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40185b:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401860:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401865:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40186a:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40186e:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401873:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401878:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40187d:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401882:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401887:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  40188b:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401890:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401895:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40189a:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40189f:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4018a4:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4018a8:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4018ad:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4018b2:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4018b7:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4018bc:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4018c1:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4018c5:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4018ca:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4018cf:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4018d4:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4018d9:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4018de:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4018e2:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4018e7:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4018ec:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4018f1:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4018f6:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4018fb:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4018ff:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401904:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401909:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40190e:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401913:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401918:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  40191c:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401921:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401926:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40192b:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401930:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401935:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  401939:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40193e:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401943:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401948:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  40194d:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401952:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401956:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40195b:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401960:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401965:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  40196a:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  40196f:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401973:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401978:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40197d:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401982:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401987:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  40198c:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  401990:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401995:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  40199a:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  40199f:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4019a4:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4019a9:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  4019ad:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4019b2:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4019b7:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4019bc:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  4019c1:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4019c6:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  4019ca:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  4019cf:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4019d4:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4019d9:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  4019de:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  4019e3:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  4019e7:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  4019ec:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  4019f1:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  4019f6:	c4 41 7d 58 f9       	vaddpd %ymm9,%ymm0,%ymm15
  4019fb:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401a00:	c5 f5 58 d7          	vaddpd %ymm7,%ymm1,%ymm2
  401a04:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401a09:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401a0e:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401a13:	c4 41 7d 58 f1       	vaddpd %ymm9,%ymm0,%ymm14
  401a18:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401a1d:	c5 75 58 ff          	vaddpd %ymm7,%ymm1,%ymm15
  401a21:	c4 62 d5 a8 fa       	vfmadd213pd %ymm2,%ymm5,%ymm15
  401a26:	c4 c1 75 58 c8       	vaddpd %ymm8,%ymm1,%ymm1
  401a2b:	c4 c1 7d 58 c2       	vaddpd %ymm10,%ymm0,%ymm0
  401a30:	c4 c1 7d 58 d1       	vaddpd %ymm9,%ymm0,%ymm2
  401a35:	c4 c2 d5 a8 d6       	vfmadd213pd %ymm14,%ymm5,%ymm2
  401a3a:	c5 75 58 f7          	vaddpd %ymm7,%ymm1,%ymm14
  401a3e:	c4 42 d5 a8 f7       	vfmadd213pd %ymm15,%ymm5,%ymm14
  401a43:	c4 41 2d 58 c9       	vaddpd %ymm9,%ymm10,%ymm9
  401a48:	c4 c1 7d 58 c1       	vaddpd %ymm9,%ymm0,%ymm0
  401a4d:	c5 bd 58 ff          	vaddpd %ymm7,%ymm8,%ymm7
  401a51:	c5 f5 58 cf          	vaddpd %ymm7,%ymm1,%ymm1
  401a55:	c4 e2 d5 a8 c2       	vfmadd213pd %ymm2,%ymm5,%ymm0
  401a5a:	c4 c2 d5 a8 ce       	vfmadd213pd %ymm14,%ymm5,%ymm1
  401a5f:	c4 e3 7d 19 c2 01    	vextractf128 $0x1,%ymm0,%xmm2
  401a65:	c5 fd 58 c2          	vaddpd %ymm2,%ymm0,%ymm0
  401a69:	c4 e3 7d 19 ca 01    	vextractf128 $0x1,%ymm1,%xmm2
  401a6f:	c5 f5 58 ca          	vaddpd %ymm2,%ymm1,%ymm1
  401a73:	c5 fd 7c c0          	vhaddpd %ymm0,%ymm0,%ymm0
  401a77:	c5 f5 7c c9          	vhaddpd %ymm1,%ymm1,%ymm1
  401a7b:	c5 f9 13 b0 90 18 64 	vmovlpd %xmm6,0x641890(%rax)
  401a82:	00 
  401a83:	c5 f9 17 b0 90 12 65 	vmovhpd %xmm6,0x651290(%rax)
  401a8a:	00 
  401a8b:	c5 f9 13 80 90 24 62 	vmovlpd %xmm0,0x622490(%rax)
  401a92:	00 
  401a93:	c5 f9 13 88 90 1e 63 	vmovlpd %xmm1,0x631e90(%rax)
  401a9a:	00 
  401a9b:	48 83 c0 08          	add    $0x8,%rax
  401a9f:	0f 85 6b fc ff ff    	jne    401710 <main+0x90>
  401aa5:	83 c3 01             	add    $0x1,%ebx
  401aa8:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  401aae:	0f 85 4c fc ff ff    	jne    401700 <main+0x80>
  401ab4:	48 bd db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rbp
  401abb:	de 1b 43 
  401abe:	4c 89 f8             	mov    %r15,%rax
  401ac1:	48 f7 ed             	imul   %rbp
  401ac4:	49 89 d7             	mov    %rdx,%r15
  401ac7:	4c 89 f9             	mov    %r15,%rcx
  401aca:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401ace:	49 c1 ff 12          	sar    $0x12,%r15
  401ad2:	4c 89 e0             	mov    %r12,%rax
  401ad5:	48 f7 ed             	imul   %rbp
  401ad8:	49 89 d4             	mov    %rdx,%r12
  401adb:	49 01 cf             	add    %rcx,%r15
  401ade:	4c 89 e0             	mov    %r12,%rax
  401ae1:	48 c1 e8 3f          	shr    $0x3f,%rax
  401ae5:	49 c1 fc 12          	sar    $0x12,%r12
  401ae9:	49 01 c4             	add    %rax,%r12
  401aec:	c5 f8 77             	vzeroupper 
  401aef:	e8 3c ec ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401af4:	49 89 c5             	mov    %rax,%r13
  401af7:	4d 29 f5             	sub    %r14,%r13
  401afa:	41 be 0b 00 00 00    	mov    $0xb,%r14d
  401b00:	31 db                	xor    %ebx,%ebx
  401b02:	eb 1f                	jmp    401b23 <main+0x4a3>
  401b04:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401b0b:	00 00 00 00 00 
  401b10:	bf 00 1d 40 00       	mov    $0x401d00,%edi
  401b15:	b0 02                	mov    $0x2,%al
  401b17:	48 89 de             	mov    %rbx,%rsi
  401b1a:	e8 a1 eb ff ff       	callq  4006c0 <printf@plt>
  401b1f:	48 83 c3 01          	add    $0x1,%rbx
  401b23:	c5 fb 10 04 dd 90 2a 	vmovsd 0x612a90(,%rbx,8),%xmm0
  401b2a:	61 00 
  401b2c:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  401b30:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  401b34:	c5 fb 10 0c dd 90 24 	vmovsd 0x622490(,%rbx,8),%xmm1
  401b3b:	62 00 
  401b3d:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  401b41:	44 01 f0             	add    %r14d,%eax
  401b44:	c5 f9 7e d1          	vmovd  %xmm2,%ecx
  401b48:	01 c1                	add    %eax,%ecx
  401b4a:	4c 63 f1             	movslq %ecx,%r14
  401b4d:	49 69 c6 8d 68 b7 6c 	imul   $0x6cb7688d,%r14,%rax
  401b54:	48 89 c1             	mov    %rax,%rcx
  401b57:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401b5b:	48 c1 f8 33          	sar    $0x33,%rax
  401b5f:	01 c8                	add    %ecx,%eax
  401b61:	69 c0 87 d6 12 00    	imul   $0x12d687,%eax,%eax
  401b67:	41 29 c6             	sub    %eax,%r14d
  401b6a:	48 83 fb 09          	cmp    $0x9,%rbx
  401b6e:	76 a0                	jbe    401b10 <main+0x490>
  401b70:	48 83 c3 01          	add    $0x1,%rbx
  401b74:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401b7b:	75 a6                	jne    401b23 <main+0x4a3>
  401b7d:	4c 89 e8             	mov    %r13,%rax
  401b80:	48 f7 ed             	imul   %rbp
  401b83:	48 89 d0             	mov    %rdx,%rax
  401b86:	48 c1 e8 3f          	shr    $0x3f,%rax
  401b8a:	48 c1 fa 12          	sar    $0x12,%rdx
  401b8e:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401b92:	bf 15 1d 40 00       	mov    $0x401d15,%edi
  401b97:	31 c0                	xor    %eax,%eax
  401b99:	4c 89 fe             	mov    %r15,%rsi
  401b9c:	4c 89 e2             	mov    %r12,%rdx
  401b9f:	45 89 f0             	mov    %r14d,%r8d
  401ba2:	e8 19 eb ff ff       	callq  4006c0 <printf@plt>
  401ba7:	31 c0                	xor    %eax,%eax
  401ba9:	48 83 c4 08          	add    $0x8,%rsp
  401bad:	5b                   	pop    %rbx
  401bae:	41 5c                	pop    %r12
  401bb0:	41 5d                	pop    %r13
  401bb2:	41 5e                	pop    %r14
  401bb4:	41 5f                	pop    %r15
  401bb6:	5d                   	pop    %rbp
  401bb7:	c3                   	retq   
  401bb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401bbf:	00 

0000000000401bc0 <__libc_csu_init>:
  401bc0:	41 57                	push   %r15
  401bc2:	41 56                	push   %r14
  401bc4:	41 89 ff             	mov    %edi,%r15d
  401bc7:	41 55                	push   %r13
  401bc9:	41 54                	push   %r12
  401bcb:	4c 8d 25 fe 11 20 00 	lea    0x2011fe(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401bd2:	55                   	push   %rbp
  401bd3:	48 8d 2d 0e 12 20 00 	lea    0x20120e(%rip),%rbp        # 602de8 <__init_array_end>
  401bda:	53                   	push   %rbx
  401bdb:	49 89 f6             	mov    %rsi,%r14
  401bde:	49 89 d5             	mov    %rdx,%r13
  401be1:	4c 29 e5             	sub    %r12,%rbp
  401be4:	48 83 ec 08          	sub    $0x8,%rsp
  401be8:	48 c1 fd 03          	sar    $0x3,%rbp
  401bec:	e8 97 ea ff ff       	callq  400688 <_init>
  401bf1:	48 85 ed             	test   %rbp,%rbp
  401bf4:	74 20                	je     401c16 <__libc_csu_init+0x56>
  401bf6:	31 db                	xor    %ebx,%ebx
  401bf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401bff:	00 
  401c00:	4c 89 ea             	mov    %r13,%rdx
  401c03:	4c 89 f6             	mov    %r14,%rsi
  401c06:	44 89 ff             	mov    %r15d,%edi
  401c09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401c0d:	48 83 c3 01          	add    $0x1,%rbx
  401c11:	48 39 eb             	cmp    %rbp,%rbx
  401c14:	75 ea                	jne    401c00 <__libc_csu_init+0x40>
  401c16:	48 83 c4 08          	add    $0x8,%rsp
  401c1a:	5b                   	pop    %rbx
  401c1b:	5d                   	pop    %rbp
  401c1c:	41 5c                	pop    %r12
  401c1e:	41 5d                	pop    %r13
  401c20:	41 5e                	pop    %r14
  401c22:	41 5f                	pop    %r15
  401c24:	c3                   	retq   
  401c25:	90                   	nop
  401c26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401c2d:	00 00 00 

0000000000401c30 <__libc_csu_fini>:
  401c30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401c34 <_fini>:
  401c34:	48 83 ec 08          	sub    $0x8,%rsp
  401c38:	48 83 c4 08          	add    $0x8,%rsp
  401c3c:	c3                   	retq   
