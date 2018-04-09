
bin/clang++-5.0_nbody_ikracpp_inversed:     file format elf64-x86-64


Disassembly of section .init:

0000000000400650 <_init>:
  400650:	48 83 ec 08          	sub    $0x8,%rsp
  400654:	48 8b 05 9d 29 20 00 	mov    0x20299d(%rip),%rax        # 602ff8 <_DYNAMIC+0x200>
  40065b:	48 85 c0             	test   %rax,%rax
  40065e:	74 05                	je     400665 <_init+0x15>
  400660:	e8 9b 00 00 00       	callq  400700 <rand@plt+0x10>
  400665:	48 83 c4 08          	add    $0x8,%rsp
  400669:	c3                   	retq   

Disassembly of section .plt:

0000000000400670 <printf@plt-0x10>:
  400670:	ff 35 92 29 20 00    	pushq  0x202992(%rip)        # 603008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400676:	ff 25 94 29 20 00    	jmpq   *0x202994(%rip)        # 603010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40067c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400680 <printf@plt>:
  400680:	ff 25 92 29 20 00    	jmpq   *0x202992(%rip)        # 603018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400686:	68 00 00 00 00       	pushq  $0x0
  40068b:	e9 e0 ff ff ff       	jmpq   400670 <_init+0x20>

0000000000400690 <_ZNSt8ios_base4InitC1Ev@plt>:
  400690:	ff 25 8a 29 20 00    	jmpq   *0x20298a(%rip)        # 603020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400696:	68 01 00 00 00       	pushq  $0x1
  40069b:	e9 d0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006a0 <__libc_start_main@plt>:
  4006a0:	ff 25 82 29 20 00    	jmpq   *0x202982(%rip)        # 603028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006a6:	68 02 00 00 00       	pushq  $0x2
  4006ab:	e9 c0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006b0 <__cxa_atexit@plt>:
  4006b0:	ff 25 7a 29 20 00    	jmpq   *0x20297a(%rip)        # 603030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006b6:	68 03 00 00 00       	pushq  $0x3
  4006bb:	e9 b0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006c0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4006c0:	ff 25 72 29 20 00    	jmpq   *0x202972(%rip)        # 603038 <_GLOBAL_OFFSET_TABLE_+0x38>
  4006c6:	68 04 00 00 00       	pushq  $0x4
  4006cb:	e9 a0 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006d0 <srand@plt>:
  4006d0:	ff 25 6a 29 20 00    	jmpq   *0x20296a(%rip)        # 603040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4006d6:	68 05 00 00 00       	pushq  $0x5
  4006db:	e9 90 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  4006e0:	ff 25 62 29 20 00    	jmpq   *0x202962(%rip)        # 603048 <_GLOBAL_OFFSET_TABLE_+0x48>
  4006e6:	68 06 00 00 00       	pushq  $0x6
  4006eb:	e9 80 ff ff ff       	jmpq   400670 <_init+0x20>

00000000004006f0 <rand@plt>:
  4006f0:	ff 25 5a 29 20 00    	jmpq   *0x20295a(%rip)        # 603050 <_GLOBAL_OFFSET_TABLE_+0x50>
  4006f6:	68 07 00 00 00       	pushq  $0x7
  4006fb:	e9 70 ff ff ff       	jmpq   400670 <_init+0x20>

Disassembly of section .plt.got:

0000000000400700 <.plt.got>:
  400700:	ff 25 f2 28 20 00    	jmpq   *0x2028f2(%rip)        # 602ff8 <_DYNAMIC+0x200>
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

0000000000400730 <_GLOBAL__sub_I_ikracpp_inversed.cc>:
  400730:	50                   	push   %rax
  400731:	bf 70 30 60 00       	mov    $0x603070,%edi
  400736:	e8 55 ff ff ff       	callq  400690 <_ZNSt8ios_base4InitC1Ev@plt>
  40073b:	bf c0 06 40 00       	mov    $0x4006c0,%edi
  400740:	be 70 30 60 00       	mov    $0x603070,%esi
  400745:	ba 60 30 60 00       	mov    $0x603060,%edx
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
  40075f:	49 c7 c0 70 1b 40 00 	mov    $0x401b70,%r8
  400766:	48 c7 c1 00 1b 40 00 	mov    $0x401b00,%rcx
  40076d:	48 c7 c7 60 14 40 00 	mov    $0x401460,%rdi
  400774:	e8 27 ff ff ff       	callq  4006a0 <__libc_start_main@plt>
  400779:	f4                   	hlt    
  40077a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400780 <deregister_tm_clones>:
  400780:	b8 6f 30 60 00       	mov    $0x60306f,%eax
  400785:	55                   	push   %rbp
  400786:	48 2d 68 30 60 00    	sub    $0x603068,%rax
  40078c:	48 83 f8 0e          	cmp    $0xe,%rax
  400790:	48 89 e5             	mov    %rsp,%rbp
  400793:	76 1b                	jbe    4007b0 <deregister_tm_clones+0x30>
  400795:	b8 00 00 00 00       	mov    $0x0,%eax
  40079a:	48 85 c0             	test   %rax,%rax
  40079d:	74 11                	je     4007b0 <deregister_tm_clones+0x30>
  40079f:	5d                   	pop    %rbp
  4007a0:	bf 68 30 60 00       	mov    $0x603068,%edi
  4007a5:	ff e0                	jmpq   *%rax
  4007a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ae:	00 00 
  4007b0:	5d                   	pop    %rbp
  4007b1:	c3                   	retq   
  4007b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007bd:	00 00 00 

00000000004007c0 <register_tm_clones>:
  4007c0:	be 68 30 60 00       	mov    $0x603068,%esi
  4007c5:	55                   	push   %rbp
  4007c6:	48 81 ee 68 30 60 00 	sub    $0x603068,%rsi
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
  4007ee:	bf 68 30 60 00       	mov    $0x603068,%edi
  4007f3:	ff e0                	jmpq   *%rax
  4007f5:	0f 1f 00             	nopl   (%rax)
  4007f8:	5d                   	pop    %rbp
  4007f9:	c3                   	retq   
  4007fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400800 <__do_global_dtors_aux>:
  400800:	80 3d 61 28 20 00 00 	cmpb   $0x0,0x202861(%rip)        # 603068 <__TMC_END__>
  400807:	75 11                	jne    40081a <__do_global_dtors_aux+0x1a>
  400809:	55                   	push   %rbp
  40080a:	48 89 e5             	mov    %rsp,%rbp
  40080d:	e8 6e ff ff ff       	callq  400780 <deregister_tm_clones>
  400812:	5d                   	pop    %rbp
  400813:	c6 05 4e 28 20 00 01 	movb   $0x1,0x20284e(%rip)        # 603068 <__TMC_END__>
  40081a:	f3 c3                	repz retq 
  40081c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400820 <frame_dummy>:
  400820:	bf f0 2d 60 00       	mov    $0x602df0,%edi
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
  400860:	53                   	push   %rbx
  400861:	48 83 ec 20          	sub    $0x20,%rsp
  400865:	bf 2a 00 00 00       	mov    $0x2a,%edi
  40086a:	e8 61 fe ff ff       	callq  4006d0 <srand@plt>
  40086f:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  400874:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40087b:	00 00 00 00 00 
  400880:	e8 6b fe ff ff       	callq  4006f0 <rand@plt>
  400885:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  400889:	c5 fb 10 05 ff 12 00 	vmovsd 0x12ff(%rip),%xmm0        # 401b90 <_IO_stdin_used+0x10>
  400890:	00 
  400891:	c4 e2 f9 99 0d ee 12 	vfmadd132sd 0x12ee(%rip),%xmm0,%xmm1        # 401b88 <_IO_stdin_used+0x8>
  400898:	00 00 
  40089a:	c5 fb 11 4c 24 18    	vmovsd %xmm1,0x18(%rsp)
  4008a0:	e8 4b fe ff ff       	callq  4006f0 <rand@plt>
  4008a5:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  4008a9:	c5 fb 10 05 e7 12 00 	vmovsd 0x12e7(%rip),%xmm0        # 401b98 <_IO_stdin_used+0x18>
  4008b0:	00 
  4008b1:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008b5:	c5 fb 10 05 e3 12 00 	vmovsd 0x12e3(%rip),%xmm0        # 401ba0 <_IO_stdin_used+0x20>
  4008bc:	00 
  4008bd:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008c2:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  4008c8:	e8 23 fe ff ff       	callq  4006f0 <rand@plt>
  4008cd:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  4008d1:	c5 fb 10 0d bf 12 00 	vmovsd 0x12bf(%rip),%xmm1        # 401b98 <_IO_stdin_used+0x18>
  4008d8:	00 
  4008d9:	c4 e2 f1 99 05 be 12 	vfmadd132sd 0x12be(%rip),%xmm1,%xmm0        # 401ba0 <_IO_stdin_used+0x20>
  4008e0:	00 00 
  4008e2:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
  4008e8:	e8 03 fe ff ff       	callq  4006f0 <rand@plt>
  4008ed:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  4008f1:	c5 fb 10 05 af 12 00 	vmovsd 0x12af(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x28>
  4008f8:	00 
  4008f9:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008fd:	c5 fb 10 05 ab 12 00 	vmovsd 0x12ab(%rip),%xmm0        # 401bb0 <_IO_stdin_used+0x30>
  400904:	00 
  400905:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  40090a:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  40090f:	e8 dc fd ff ff       	callq  4006f0 <rand@plt>
  400914:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  400918:	c5 fb 10 0d 88 12 00 	vmovsd 0x1288(%rip),%xmm1        # 401ba8 <_IO_stdin_used+0x28>
  40091f:	00 
  400920:	c4 e2 f1 99 05 87 12 	vfmadd132sd 0x1287(%rip),%xmm1,%xmm0        # 401bb0 <_IO_stdin_used+0x30>
  400927:	00 00 
  400929:	48 8b 05 48 27 20 00 	mov    0x202748(%rip),%rax        # 603078 <__Bodydata_buffer>
  400930:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400934:	48 89 0d 3d 27 20 00 	mov    %rcx,0x20273d(%rip)        # 603078 <__Bodydata_buffer>
  40093b:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  400941:	c5 fb 11 0c c5 90 30 	vmovsd %xmm1,0x603090(,%rax,8)
  400948:	60 00 
  40094a:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  400950:	c5 fb 11 0c c5 98 2a 	vmovsd %xmm1,0x612a98(,%rax,8)
  400957:	61 00 
  400959:	c5 fb 10 4c 24 08    	vmovsd 0x8(%rsp),%xmm1
  40095f:	c5 fb 11 0c c5 a0 24 	vmovsd %xmm1,0x6224a0(,%rax,8)
  400966:	62 00 
  400968:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  40096d:	c5 fb 11 0c c5 a8 1e 	vmovsd %xmm1,0x631ea8(,%rax,8)
  400974:	63 00 
  400976:	c5 fb 11 04 c5 b0 18 	vmovsd %xmm0,0x6418b0(,%rax,8)
  40097d:	64 00 
  40097f:	48 c7 04 c5 b8 12 65 	movq   $0x0,0x6512b8(,%rax,8)
  400986:	00 00 00 00 00 
  40098b:	48 c7 04 c5 c0 0c 66 	movq   $0x0,0x660cc0(,%rax,8)
  400992:	00 00 00 00 00 
  400997:	83 c3 ff             	add    $0xffffffff,%ebx
  40099a:	0f 85 e0 fe ff ff    	jne    400880 <_Z13instantiationv+0x20>
  4009a0:	48 83 c4 20          	add    $0x20,%rsp
  4009a4:	5b                   	pop    %rbx
  4009a5:	c3                   	retq   
  4009a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ad:	00 00 00 

00000000004009b0 <_Z14run_simulationv>:
  4009b0:	41 56                	push   %r14
  4009b2:	53                   	push   %rbx
  4009b3:	50                   	push   %rax
  4009b4:	b8 10 00 00 00       	mov    $0x10,%eax
  4009b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4009c0:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  4009c7:	00 00 00 00 00 
  4009cc:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  4009d3:	00 00 00 00 00 
  4009d8:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  4009df:	00 00 00 00 00 
  4009e4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  4009eb:	00 00 00 00 00 
  4009f0:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  4009f7:	00 00 00 00 00 
  4009fc:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400a03:	00 00 00 00 00 
  400a08:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400a0f:	00 00 00 00 00 
  400a14:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400a1b:	00 00 00 00 00 
  400a20:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400a27:	00 00 00 00 00 
  400a2c:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400a33:	00 00 00 00 00 
  400a38:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400a3f:	00 00 00 00 00 
  400a44:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400a4b:	00 00 00 00 00 
  400a50:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400a57:	00 00 00 00 00 
  400a5c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400a63:	00 00 00 00 00 
  400a68:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400a6f:	00 00 00 00 00 
  400a74:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400a7b:	00 00 00 00 00 
  400a80:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400a87:	00 00 00 00 00 
  400a8c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400a93:	00 00 00 00 00 
  400a98:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400a9f:	00 00 00 00 00 
  400aa4:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  400aab:	00 00 00 00 00 
  400ab0:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400ab7:	00 00 00 00 00 
  400abc:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  400ac3:	00 00 00 00 00 
  400ac8:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  400acf:	00 00 00 00 00 
  400ad4:	48 c7 04 c5 98 0c 66 	movq   $0x0,0x660c98(,%rax,8)
  400adb:	00 00 00 00 00 
  400ae0:	48 c7 04 c5 98 12 65 	movq   $0x0,0x651298(,%rax,8)
  400ae7:	00 00 00 00 00 
  400aec:	48 c7 04 c5 a0 0c 66 	movq   $0x0,0x660ca0(,%rax,8)
  400af3:	00 00 00 00 00 
  400af8:	48 c7 04 c5 a0 12 65 	movq   $0x0,0x6512a0(,%rax,8)
  400aff:	00 00 00 00 00 
  400b04:	48 c7 04 c5 a8 0c 66 	movq   $0x0,0x660ca8(,%rax,8)
  400b0b:	00 00 00 00 00 
  400b10:	48 c7 04 c5 a8 12 65 	movq   $0x0,0x6512a8(,%rax,8)
  400b17:	00 00 00 00 00 
  400b1c:	48 c7 04 c5 b0 0c 66 	movq   $0x0,0x660cb0(,%rax,8)
  400b23:	00 00 00 00 00 
  400b28:	48 c7 04 c5 b0 12 65 	movq   $0x0,0x6512b0(,%rax,8)
  400b2f:	00 00 00 00 00 
  400b34:	48 c7 04 c5 b8 0c 66 	movq   $0x0,0x660cb8(,%rax,8)
  400b3b:	00 00 00 00 00 
  400b40:	48 83 c0 10          	add    $0x10,%rax
  400b44:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  400b4a:	0f 85 70 fe ff ff    	jne    4009c0 <_Z14run_simulationv+0x10>
  400b50:	45 31 f6             	xor    %r14d,%r14d
  400b53:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400b5a:	84 00 00 00 00 00 
  400b60:	49 83 c6 01          	add    $0x1,%r14
  400b64:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400b70:	4c 89 f7             	mov    %r14,%rdi
  400b73:	48 89 de             	mov    %rbx,%rsi
  400b76:	e8 e5 0e 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400b7b:	48 83 c3 01          	add    $0x1,%rbx
  400b7f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400b86:	75 e8                	jne    400b70 <_Z14run_simulationv+0x1c0>
  400b88:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400b8f:	75 cf                	jne    400b60 <_Z14run_simulationv+0x1b0>
  400b91:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 
  400ba0:	48 89 df             	mov    %rbx,%rdi
  400ba3:	c5 fb 10 05 0d 10 00 	vmovsd 0x100d(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400baa:	00 
  400bab:	e8 40 0d 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400bb0:	48 83 c3 01          	add    $0x1,%rbx
  400bb4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bbb:	75 e3                	jne    400ba0 <_Z14run_simulationv+0x1f0>
  400bbd:	b8 10 00 00 00       	mov    $0x10,%eax
  400bc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400bc9:	1f 84 00 00 00 00 00 
  400bd0:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400bd7:	00 00 00 00 00 
  400bdc:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400be3:	00 00 00 00 00 
  400be8:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400bef:	00 00 00 00 00 
  400bf4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400bfb:	00 00 00 00 00 
  400c00:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400c07:	00 00 00 00 00 
  400c0c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400c13:	00 00 00 00 00 
  400c18:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400c1f:	00 00 00 00 00 
  400c24:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400c2b:	00 00 00 00 00 
  400c30:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400c37:	00 00 00 00 00 
  400c3c:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400c43:	00 00 00 00 00 
  400c48:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400c4f:	00 00 00 00 00 
  400c54:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400c5b:	00 00 00 00 00 
  400c60:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400c67:	00 00 00 00 00 
  400c6c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400c73:	00 00 00 00 00 
  400c78:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400c7f:	00 00 00 00 00 
  400c84:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400c8b:	00 00 00 00 00 
  400c90:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400c97:	00 00 00 00 00 
  400c9c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400ca3:	00 00 00 00 00 
  400ca8:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400caf:	00 00 00 00 00 
  400cb4:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  400cbb:	00 00 00 00 00 
  400cc0:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400cc7:	00 00 00 00 00 
  400ccc:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  400cd3:	00 00 00 00 00 
  400cd8:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  400cdf:	00 00 00 00 00 
  400ce4:	48 c7 04 c5 98 0c 66 	movq   $0x0,0x660c98(,%rax,8)
  400ceb:	00 00 00 00 00 
  400cf0:	48 c7 04 c5 98 12 65 	movq   $0x0,0x651298(,%rax,8)
  400cf7:	00 00 00 00 00 
  400cfc:	48 c7 04 c5 a0 0c 66 	movq   $0x0,0x660ca0(,%rax,8)
  400d03:	00 00 00 00 00 
  400d08:	48 c7 04 c5 a0 12 65 	movq   $0x0,0x6512a0(,%rax,8)
  400d0f:	00 00 00 00 00 
  400d14:	48 c7 04 c5 a8 0c 66 	movq   $0x0,0x660ca8(,%rax,8)
  400d1b:	00 00 00 00 00 
  400d20:	48 c7 04 c5 a8 12 65 	movq   $0x0,0x6512a8(,%rax,8)
  400d27:	00 00 00 00 00 
  400d2c:	48 c7 04 c5 b0 0c 66 	movq   $0x0,0x660cb0(,%rax,8)
  400d33:	00 00 00 00 00 
  400d38:	48 c7 04 c5 b0 12 65 	movq   $0x0,0x6512b0(,%rax,8)
  400d3f:	00 00 00 00 00 
  400d44:	48 c7 04 c5 b8 0c 66 	movq   $0x0,0x660cb8(,%rax,8)
  400d4b:	00 00 00 00 00 
  400d50:	48 83 c0 10          	add    $0x10,%rax
  400d54:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  400d5a:	0f 85 70 fe ff ff    	jne    400bd0 <_Z14run_simulationv+0x220>
  400d60:	45 31 f6             	xor    %r14d,%r14d
  400d63:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400d6a:	84 00 00 00 00 00 
  400d70:	49 83 c6 01          	add    $0x1,%r14
  400d74:	bb 01 00 00 00       	mov    $0x1,%ebx
  400d79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400d80:	4c 89 f7             	mov    %r14,%rdi
  400d83:	48 89 de             	mov    %rbx,%rsi
  400d86:	e8 d5 0c 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400d8b:	48 83 c3 01          	add    $0x1,%rbx
  400d8f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400d96:	75 e8                	jne    400d80 <_Z14run_simulationv+0x3d0>
  400d98:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400d9f:	75 cf                	jne    400d70 <_Z14run_simulationv+0x3c0>
  400da1:	bb 01 00 00 00       	mov    $0x1,%ebx
  400da6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dad:	00 00 00 
  400db0:	48 89 df             	mov    %rbx,%rdi
  400db3:	c5 fb 10 05 fd 0d 00 	vmovsd 0xdfd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400dba:	00 
  400dbb:	e8 30 0b 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400dc0:	48 83 c3 01          	add    $0x1,%rbx
  400dc4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400dcb:	75 e3                	jne    400db0 <_Z14run_simulationv+0x400>
  400dcd:	b8 10 00 00 00       	mov    $0x10,%eax
  400dd2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400dd9:	1f 84 00 00 00 00 00 
  400de0:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400de7:	00 00 00 00 00 
  400dec:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400df3:	00 00 00 00 00 
  400df8:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400dff:	00 00 00 00 00 
  400e04:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400e0b:	00 00 00 00 00 
  400e10:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400e17:	00 00 00 00 00 
  400e1c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400e23:	00 00 00 00 00 
  400e28:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400e2f:	00 00 00 00 00 
  400e34:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400e3b:	00 00 00 00 00 
  400e40:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400e47:	00 00 00 00 00 
  400e4c:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400e53:	00 00 00 00 00 
  400e58:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400e5f:	00 00 00 00 00 
  400e64:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400e6b:	00 00 00 00 00 
  400e70:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400e77:	00 00 00 00 00 
  400e7c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400e83:	00 00 00 00 00 
  400e88:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400e8f:	00 00 00 00 00 
  400e94:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400e9b:	00 00 00 00 00 
  400ea0:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400ea7:	00 00 00 00 00 
  400eac:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400eb3:	00 00 00 00 00 
  400eb8:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400ebf:	00 00 00 00 00 
  400ec4:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  400ecb:	00 00 00 00 00 
  400ed0:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400ed7:	00 00 00 00 00 
  400edc:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  400ee3:	00 00 00 00 00 
  400ee8:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  400eef:	00 00 00 00 00 
  400ef4:	48 c7 04 c5 98 0c 66 	movq   $0x0,0x660c98(,%rax,8)
  400efb:	00 00 00 00 00 
  400f00:	48 c7 04 c5 98 12 65 	movq   $0x0,0x651298(,%rax,8)
  400f07:	00 00 00 00 00 
  400f0c:	48 c7 04 c5 a0 0c 66 	movq   $0x0,0x660ca0(,%rax,8)
  400f13:	00 00 00 00 00 
  400f18:	48 c7 04 c5 a0 12 65 	movq   $0x0,0x6512a0(,%rax,8)
  400f1f:	00 00 00 00 00 
  400f24:	48 c7 04 c5 a8 0c 66 	movq   $0x0,0x660ca8(,%rax,8)
  400f2b:	00 00 00 00 00 
  400f30:	48 c7 04 c5 a8 12 65 	movq   $0x0,0x6512a8(,%rax,8)
  400f37:	00 00 00 00 00 
  400f3c:	48 c7 04 c5 b0 0c 66 	movq   $0x0,0x660cb0(,%rax,8)
  400f43:	00 00 00 00 00 
  400f48:	48 c7 04 c5 b0 12 65 	movq   $0x0,0x6512b0(,%rax,8)
  400f4f:	00 00 00 00 00 
  400f54:	48 c7 04 c5 b8 0c 66 	movq   $0x0,0x660cb8(,%rax,8)
  400f5b:	00 00 00 00 00 
  400f60:	48 83 c0 10          	add    $0x10,%rax
  400f64:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  400f6a:	0f 85 70 fe ff ff    	jne    400de0 <_Z14run_simulationv+0x430>
  400f70:	45 31 f6             	xor    %r14d,%r14d
  400f73:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400f7a:	84 00 00 00 00 00 
  400f80:	49 83 c6 01          	add    $0x1,%r14
  400f84:	bb 01 00 00 00       	mov    $0x1,%ebx
  400f89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400f90:	4c 89 f7             	mov    %r14,%rdi
  400f93:	48 89 de             	mov    %rbx,%rsi
  400f96:	e8 c5 0a 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400f9b:	48 83 c3 01          	add    $0x1,%rbx
  400f9f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fa6:	75 e8                	jne    400f90 <_Z14run_simulationv+0x5e0>
  400fa8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400faf:	75 cf                	jne    400f80 <_Z14run_simulationv+0x5d0>
  400fb1:	bb 01 00 00 00       	mov    $0x1,%ebx
  400fb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fbd:	00 00 00 
  400fc0:	48 89 df             	mov    %rbx,%rdi
  400fc3:	c5 fb 10 05 ed 0b 00 	vmovsd 0xbed(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400fca:	00 
  400fcb:	e8 20 09 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400fd0:	48 83 c3 01          	add    $0x1,%rbx
  400fd4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fdb:	75 e3                	jne    400fc0 <_Z14run_simulationv+0x610>
  400fdd:	b8 10 00 00 00       	mov    $0x10,%eax
  400fe2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400fe9:	1f 84 00 00 00 00 00 
  400ff0:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400ff7:	00 00 00 00 00 
  400ffc:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  401003:	00 00 00 00 00 
  401008:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  40100f:	00 00 00 00 00 
  401014:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  40101b:	00 00 00 00 00 
  401020:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  401027:	00 00 00 00 00 
  40102c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  401033:	00 00 00 00 00 
  401038:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  40103f:	00 00 00 00 00 
  401044:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  40104b:	00 00 00 00 00 
  401050:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  401057:	00 00 00 00 00 
  40105c:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  401063:	00 00 00 00 00 
  401068:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  40106f:	00 00 00 00 00 
  401074:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  40107b:	00 00 00 00 00 
  401080:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  401087:	00 00 00 00 00 
  40108c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  401093:	00 00 00 00 00 
  401098:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  40109f:	00 00 00 00 00 
  4010a4:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  4010ab:	00 00 00 00 00 
  4010b0:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  4010b7:	00 00 00 00 00 
  4010bc:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  4010c3:	00 00 00 00 00 
  4010c8:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  4010cf:	00 00 00 00 00 
  4010d4:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  4010db:	00 00 00 00 00 
  4010e0:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  4010e7:	00 00 00 00 00 
  4010ec:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  4010f3:	00 00 00 00 00 
  4010f8:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  4010ff:	00 00 00 00 00 
  401104:	48 c7 04 c5 98 0c 66 	movq   $0x0,0x660c98(,%rax,8)
  40110b:	00 00 00 00 00 
  401110:	48 c7 04 c5 98 12 65 	movq   $0x0,0x651298(,%rax,8)
  401117:	00 00 00 00 00 
  40111c:	48 c7 04 c5 a0 0c 66 	movq   $0x0,0x660ca0(,%rax,8)
  401123:	00 00 00 00 00 
  401128:	48 c7 04 c5 a0 12 65 	movq   $0x0,0x6512a0(,%rax,8)
  40112f:	00 00 00 00 00 
  401134:	48 c7 04 c5 a8 0c 66 	movq   $0x0,0x660ca8(,%rax,8)
  40113b:	00 00 00 00 00 
  401140:	48 c7 04 c5 a8 12 65 	movq   $0x0,0x6512a8(,%rax,8)
  401147:	00 00 00 00 00 
  40114c:	48 c7 04 c5 b0 0c 66 	movq   $0x0,0x660cb0(,%rax,8)
  401153:	00 00 00 00 00 
  401158:	48 c7 04 c5 b0 12 65 	movq   $0x0,0x6512b0(,%rax,8)
  40115f:	00 00 00 00 00 
  401164:	48 c7 04 c5 b8 0c 66 	movq   $0x0,0x660cb8(,%rax,8)
  40116b:	00 00 00 00 00 
  401170:	48 83 c0 10          	add    $0x10,%rax
  401174:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  40117a:	0f 85 70 fe ff ff    	jne    400ff0 <_Z14run_simulationv+0x640>
  401180:	45 31 f6             	xor    %r14d,%r14d
  401183:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40118a:	84 00 00 00 00 00 
  401190:	49 83 c6 01          	add    $0x1,%r14
  401194:	bb 01 00 00 00       	mov    $0x1,%ebx
  401199:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011a0:	4c 89 f7             	mov    %r14,%rdi
  4011a3:	48 89 de             	mov    %rbx,%rsi
  4011a6:	e8 b5 08 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  4011ab:	48 83 c3 01          	add    $0x1,%rbx
  4011af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011b6:	75 e8                	jne    4011a0 <_Z14run_simulationv+0x7f0>
  4011b8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  4011bf:	75 cf                	jne    401190 <_Z14run_simulationv+0x7e0>
  4011c1:	bb 01 00 00 00       	mov    $0x1,%ebx
  4011c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011cd:	00 00 00 
  4011d0:	48 89 df             	mov    %rbx,%rdi
  4011d3:	c5 fb 10 05 dd 09 00 	vmovsd 0x9dd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  4011da:	00 
  4011db:	e8 10 07 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  4011e0:	48 83 c3 01          	add    $0x1,%rbx
  4011e4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011eb:	75 e3                	jne    4011d0 <_Z14run_simulationv+0x820>
  4011ed:	b8 10 00 00 00       	mov    $0x10,%eax
  4011f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4011f9:	1f 84 00 00 00 00 00 
  401200:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  401207:	00 00 00 00 00 
  40120c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  401213:	00 00 00 00 00 
  401218:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  40121f:	00 00 00 00 00 
  401224:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  40122b:	00 00 00 00 00 
  401230:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  401237:	00 00 00 00 00 
  40123c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  401243:	00 00 00 00 00 
  401248:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  40124f:	00 00 00 00 00 
  401254:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  40125b:	00 00 00 00 00 
  401260:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  401267:	00 00 00 00 00 
  40126c:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  401273:	00 00 00 00 00 
  401278:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  40127f:	00 00 00 00 00 
  401284:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  40128b:	00 00 00 00 00 
  401290:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  401297:	00 00 00 00 00 
  40129c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  4012a3:	00 00 00 00 00 
  4012a8:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  4012af:	00 00 00 00 00 
  4012b4:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  4012bb:	00 00 00 00 00 
  4012c0:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  4012c7:	00 00 00 00 00 
  4012cc:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  4012d3:	00 00 00 00 00 
  4012d8:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  4012df:	00 00 00 00 00 
  4012e4:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  4012eb:	00 00 00 00 00 
  4012f0:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  4012f7:	00 00 00 00 00 
  4012fc:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  401303:	00 00 00 00 00 
  401308:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  40130f:	00 00 00 00 00 
  401314:	48 c7 04 c5 98 0c 66 	movq   $0x0,0x660c98(,%rax,8)
  40131b:	00 00 00 00 00 
  401320:	48 c7 04 c5 98 12 65 	movq   $0x0,0x651298(,%rax,8)
  401327:	00 00 00 00 00 
  40132c:	48 c7 04 c5 a0 0c 66 	movq   $0x0,0x660ca0(,%rax,8)
  401333:	00 00 00 00 00 
  401338:	48 c7 04 c5 a0 12 65 	movq   $0x0,0x6512a0(,%rax,8)
  40133f:	00 00 00 00 00 
  401344:	48 c7 04 c5 a8 0c 66 	movq   $0x0,0x660ca8(,%rax,8)
  40134b:	00 00 00 00 00 
  401350:	48 c7 04 c5 a8 12 65 	movq   $0x0,0x6512a8(,%rax,8)
  401357:	00 00 00 00 00 
  40135c:	48 c7 04 c5 b0 0c 66 	movq   $0x0,0x660cb0(,%rax,8)
  401363:	00 00 00 00 00 
  401368:	48 c7 04 c5 b0 12 65 	movq   $0x0,0x6512b0(,%rax,8)
  40136f:	00 00 00 00 00 
  401374:	48 c7 04 c5 b8 0c 66 	movq   $0x0,0x660cb8(,%rax,8)
  40137b:	00 00 00 00 00 
  401380:	48 83 c0 10          	add    $0x10,%rax
  401384:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  40138a:	0f 85 70 fe ff ff    	jne    401200 <_Z14run_simulationv+0x850>
  401390:	45 31 f6             	xor    %r14d,%r14d
  401393:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40139a:	84 00 00 00 00 00 
  4013a0:	49 83 c6 01          	add    $0x1,%r14
  4013a4:	bb 01 00 00 00       	mov    $0x1,%ebx
  4013a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4013b0:	4c 89 f7             	mov    %r14,%rdi
  4013b3:	48 89 de             	mov    %rbx,%rsi
  4013b6:	e8 a5 06 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  4013bb:	48 83 c3 01          	add    $0x1,%rbx
  4013bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013c6:	75 e8                	jne    4013b0 <_Z14run_simulationv+0xa00>
  4013c8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  4013cf:	75 cf                	jne    4013a0 <_Z14run_simulationv+0x9f0>
  4013d1:	bb 01 00 00 00       	mov    $0x1,%ebx
  4013d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4013dd:	00 00 00 
  4013e0:	48 89 df             	mov    %rbx,%rdi
  4013e3:	c5 fb 10 05 cd 07 00 	vmovsd 0x7cd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  4013ea:	00 
  4013eb:	e8 00 05 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  4013f0:	48 83 c3 01          	add    $0x1,%rbx
  4013f4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013fb:	75 e3                	jne    4013e0 <_Z14run_simulationv+0xa30>
  4013fd:	48 83 c4 08          	add    $0x8,%rsp
  401401:	5b                   	pop    %rbx
  401402:	41 5e                	pop    %r14
  401404:	c3                   	retq   
  401405:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40140c:	00 00 00 00 

0000000000401410 <_Z10run_simplev>:
  401410:	55                   	push   %rbp
  401411:	53                   	push   %rbx
  401412:	50                   	push   %rax
  401413:	31 ed                	xor    %ebp,%ebp
  401415:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40141c:	00 00 00 00 
  401420:	bb 01 00 00 00       	mov    $0x1,%ebx
  401425:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40142c:	00 00 00 00 
  401430:	48 89 df             	mov    %rbx,%rdi
  401433:	c5 fb 10 05 7d 07 00 	vmovsd 0x77d(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  40143a:	00 
  40143b:	e8 90 05 00 00       	callq  4019d0 <_ZN4Body22codengen_simple_updateEd>
  401440:	48 83 c3 01          	add    $0x1,%rbx
  401444:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  40144b:	75 e3                	jne    401430 <_Z10run_simplev+0x20>
  40144d:	83 c5 01             	add    $0x1,%ebp
  401450:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
  401456:	75 c8                	jne    401420 <_Z10run_simplev+0x10>
  401458:	48 83 c4 08          	add    $0x8,%rsp
  40145c:	5b                   	pop    %rbx
  40145d:	5d                   	pop    %rbp
  40145e:	c3                   	retq   
  40145f:	90                   	nop

0000000000401460 <main>:
  401460:	55                   	push   %rbp
  401461:	41 57                	push   %r15
  401463:	41 56                	push   %r14
  401465:	41 55                	push   %r13
  401467:	41 54                	push   %r12
  401469:	53                   	push   %rbx
  40146a:	50                   	push   %rax
  40146b:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  40146f:	c5 f8 11 05 01 1c 20 	vmovups %xmm0,0x201c01(%rip)        # 603078 <__Bodydata_buffer>
  401476:	00 
  401477:	e8 64 f2 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40147c:	48 89 c5             	mov    %rax,%rbp
  40147f:	e8 dc f3 ff ff       	callq  400860 <_Z13instantiationv>
  401484:	e8 57 f2 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401489:	49 89 c6             	mov    %rax,%r14
  40148c:	49 29 ee             	sub    %rbp,%r14
  40148f:	e8 4c f2 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401494:	48 89 c3             	mov    %rax,%rbx
  401497:	e8 14 f5 ff ff       	callq  4009b0 <_Z14run_simulationv>
  40149c:	e8 3f f2 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4014a1:	48 89 c5             	mov    %rax,%rbp
  4014a4:	48 29 dd             	sub    %rbx,%rbp
  4014a7:	31 db                	xor    %ebx,%ebx
  4014a9:	e8 32 f2 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4014ae:	49 89 c7             	mov    %rax,%r15
  4014b1:	c5 fb 10 05 ff 06 00 	vmovsd 0x6ff(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  4014b8:	00 
  4014b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014c0:	31 c0                	xor    %eax,%eax
  4014c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4014c9:	1f 84 00 00 00 00 00 
  4014d0:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4014d4:	ba 64 00 00 00       	mov    $0x64,%edx
  4014d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014e0:	c5 fb 59 0c cd b0 12 	vmulsd 0x6512b0(,%rcx,8),%xmm0,%xmm1
  4014e7:	65 00 
  4014e9:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  4014f0:	60 00 
  4014f2:	c5 f3 58 0c cd a0 1e 	vaddsd 0x631ea0(,%rcx,8),%xmm1,%xmm1
  4014f9:	63 00 
  4014fb:	c5 fb 11 0c cd a0 1e 	vmovsd %xmm1,0x631ea0(,%rcx,8)
  401502:	63 00 
  401504:	c5 fb 59 14 cd b8 0c 	vmulsd 0x660cb8(,%rcx,8),%xmm0,%xmm2
  40150b:	66 00 
  40150d:	c5 eb 5e 14 c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm2,%xmm2
  401514:	60 00 
  401516:	c5 eb 58 14 cd a8 18 	vaddsd 0x6418a8(,%rcx,8),%xmm2,%xmm2
  40151d:	64 00 
  40151f:	c5 fb 11 14 cd a8 18 	vmovsd %xmm2,0x6418a8(,%rcx,8)
  401526:	64 00 
  401528:	c4 e2 f9 a9 0c cd 90 	vfmadd213sd 0x612a90(,%rcx,8),%xmm0,%xmm1
  40152f:	2a 61 00 
  401532:	c5 fb 11 0c cd 90 2a 	vmovsd %xmm1,0x612a90(,%rcx,8)
  401539:	61 00 
  40153b:	c5 fb 10 0c cd a8 18 	vmovsd 0x6418a8(,%rcx,8),%xmm1
  401542:	64 00 
  401544:	c4 e2 f9 a9 0c cd 98 	vfmadd213sd 0x622498(,%rcx,8),%xmm0,%xmm1
  40154b:	24 62 00 
  40154e:	83 c2 ff             	add    $0xffffffff,%edx
  401551:	c5 fb 11 0c cd 98 24 	vmovsd %xmm1,0x622498(,%rcx,8)
  401558:	62 00 
  40155a:	75 84                	jne    4014e0 <main+0x80>
  40155c:	48 89 c8             	mov    %rcx,%rax
  40155f:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401566:	0f 85 64 ff ff ff    	jne    4014d0 <main+0x70>
  40156c:	83 c3 01             	add    $0x1,%ebx
  40156f:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  401575:	0f 85 45 ff ff ff    	jne    4014c0 <main+0x60>
  40157b:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  401582:	de 1b 43 
  401585:	4c 89 f0             	mov    %r14,%rax
  401588:	48 f7 ee             	imul   %rsi
  40158b:	49 89 d6             	mov    %rdx,%r14
  40158e:	4c 89 f1             	mov    %r14,%rcx
  401591:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401595:	49 c1 fe 12          	sar    $0x12,%r14
  401599:	48 89 e8             	mov    %rbp,%rax
  40159c:	48 f7 ee             	imul   %rsi
  40159f:	49 89 d5             	mov    %rdx,%r13
  4015a2:	49 01 ce             	add    %rcx,%r14
  4015a5:	4c 89 e8             	mov    %r13,%rax
  4015a8:	48 c1 e8 3f          	shr    $0x3f,%rax
  4015ac:	49 c1 fd 12          	sar    $0x12,%r13
  4015b0:	49 01 c5             	add    %rax,%r13
  4015b3:	e8 28 f1 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4015b8:	49 89 c4             	mov    %rax,%r12
  4015bb:	4d 29 fc             	sub    %r15,%r12
  4015be:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4015c3:	31 f6                	xor    %esi,%esi
  4015c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4015cc:	00 00 00 00 
  4015d0:	c5 fb 10 04 f5 98 2a 	vmovsd 0x612a98(,%rsi,8),%xmm0
  4015d7:	61 00 
  4015d9:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  4015dd:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  4015e1:	01 e8                	add    %ebp,%eax
  4015e3:	c5 fb 10 0c f5 a0 24 	vmovsd 0x6224a0(,%rsi,8),%xmm1
  4015ea:	62 00 
  4015ec:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  4015f0:	c5 f9 7e d5          	vmovd  %xmm2,%ebp
  4015f4:	01 c5                	add    %eax,%ebp
  4015f6:	48 63 c5             	movslq %ebp,%rax
  4015f9:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  401600:	48 8d 5e 01          	lea    0x1(%rsi),%rbx
  401604:	48 89 c1             	mov    %rax,%rcx
  401607:	48 c1 e9 3f          	shr    $0x3f,%rcx
  40160b:	48 c1 f8 33          	sar    $0x33,%rax
  40160f:	01 c8                	add    %ecx,%eax
  401611:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  401618:	48 83 fe 09          	cmp    $0x9,%rsi
  40161c:	77 0c                	ja     40162a <main+0x1ca>
  40161e:	bf f0 1b 40 00       	mov    $0x401bf0,%edi
  401623:	b0 02                	mov    $0x2,%al
  401625:	e8 56 f0 ff ff       	callq  400680 <printf@plt>
  40162a:	44 29 fd             	sub    %r15d,%ebp
  40162d:	48 89 de             	mov    %rbx,%rsi
  401630:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401637:	75 97                	jne    4015d0 <main+0x170>
  401639:	4c 89 e0             	mov    %r12,%rax
  40163c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401643:	de 1b 43 
  401646:	48 f7 e9             	imul   %rcx
  401649:	48 89 d0             	mov    %rdx,%rax
  40164c:	48 c1 e8 3f          	shr    $0x3f,%rax
  401650:	48 c1 fa 12          	sar    $0x12,%rdx
  401654:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401658:	bf 05 1c 40 00       	mov    $0x401c05,%edi
  40165d:	31 c0                	xor    %eax,%eax
  40165f:	4c 89 f6             	mov    %r14,%rsi
  401662:	4c 89 ea             	mov    %r13,%rdx
  401665:	41 89 e8             	mov    %ebp,%r8d
  401668:	e8 13 f0 ff ff       	callq  400680 <printf@plt>
  40166d:	31 c0                	xor    %eax,%eax
  40166f:	48 83 c4 08          	add    $0x8,%rsp
  401673:	5b                   	pop    %rbx
  401674:	41 5c                	pop    %r12
  401676:	41 5d                	pop    %r13
  401678:	41 5e                	pop    %r14
  40167a:	41 5f                	pop    %r15
  40167c:	5d                   	pop    %rbp
  40167d:	c3                   	retq   
  40167e:	66 90                	xchg   %ax,%ax

0000000000401680 <_Z19codegen_reset_forceP4Body>:
  401680:	48 c7 04 fd b0 12 65 	movq   $0x0,0x6512b0(,%rdi,8)
  401687:	00 00 00 00 00 
  40168c:	48 c7 04 fd b8 0c 66 	movq   $0x0,0x660cb8(,%rdi,8)
  401693:	00 00 00 00 00 
  401698:	c3                   	retq   
  401699:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004016a0 <_Z17codegen_add_forceP4BodyS0_>:
  4016a0:	48 39 f7             	cmp    %rsi,%rdi
  4016a3:	0f 84 91 00 00 00    	je     40173a <_Z17codegen_add_forceP4BodyS0_+0x9a>
  4016a9:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  4016b0:	61 00 
  4016b2:	c5 fb 5c 04 fd 90 2a 	vsubsd 0x612a90(,%rdi,8),%xmm0,%xmm0
  4016b9:	61 00 
  4016bb:	c5 fb 10 0c f5 98 24 	vmovsd 0x622498(,%rsi,8),%xmm1
  4016c2:	62 00 
  4016c4:	c5 f3 5c 0c fd 98 24 	vsubsd 0x622498(,%rdi,8),%xmm1,%xmm1
  4016cb:	62 00 
  4016cd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  4016d1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  4016d6:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  4016da:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  4016e1:	60 00 
  4016e3:	c5 db 59 25 dd 04 00 	vmulsd 0x4dd(%rip),%xmm4,%xmm4        # 401bc8 <_IO_stdin_used+0x48>
  4016ea:	00 
  4016eb:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  4016f2:	60 00 
  4016f4:	c5 eb 58 15 d4 04 00 	vaddsd 0x4d4(%rip),%xmm2,%xmm2        # 401bd0 <_IO_stdin_used+0x50>
  4016fb:	00 
  4016fc:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401700:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401704:	c5 fb 10 25 b4 04 00 	vmovsd 0x4b4(%rip),%xmm4        # 401bc0 <_IO_stdin_used+0x40>
  40170b:	00 
  40170c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401710:	c4 e2 e1 a9 04 fd b0 	vfmadd213sd 0x6512b0(,%rdi,8),%xmm3,%xmm0
  401717:	12 65 00 
  40171a:	c5 fb 11 04 fd b0 12 	vmovsd %xmm0,0x6512b0(,%rdi,8)
  401721:	65 00 
  401723:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401727:	c4 e2 e1 a9 04 fd b8 	vfmadd213sd 0x660cb8(,%rdi,8),%xmm3,%xmm0
  40172e:	0c 66 00 
  401731:	c5 fb 11 04 fd b8 0c 	vmovsd %xmm0,0x660cb8(,%rdi,8)
  401738:	66 00 
  40173a:	c3                   	retq   
  40173b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401740 <_Z14codegen_updateP4Bodyd>:
  401740:	c5 fb 59 14 fd b0 12 	vmulsd 0x6512b0(,%rdi,8),%xmm0,%xmm2
  401747:	65 00 
  401749:	c5 fb 10 0d 6f 04 00 	vmovsd 0x46f(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  401750:	00 
  401751:	c5 f3 5e 1c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm3
  401758:	60 00 
  40175a:	c4 e2 e1 a9 14 fd a0 	vfmadd213sd 0x631ea0(,%rdi,8),%xmm3,%xmm2
  401761:	1e 63 00 
  401764:	c5 fb 11 14 fd a0 1e 	vmovsd %xmm2,0x631ea0(,%rdi,8)
  40176b:	63 00 
  40176d:	c5 fb 59 0c fd b8 0c 	vmulsd 0x660cb8(,%rdi,8),%xmm0,%xmm1
  401774:	66 00 
  401776:	c4 e2 e1 a9 0c fd a8 	vfmadd213sd 0x6418a8(,%rdi,8),%xmm3,%xmm1
  40177d:	18 64 00 
  401780:	c5 fb 11 0c fd a8 18 	vmovsd %xmm1,0x6418a8(,%rdi,8)
  401787:	64 00 
  401789:	c4 e2 f9 a9 14 fd 90 	vfmadd213sd 0x612a90(,%rdi,8),%xmm0,%xmm2
  401790:	2a 61 00 
  401793:	c5 fb 11 14 fd 90 2a 	vmovsd %xmm2,0x612a90(,%rdi,8)
  40179a:	61 00 
  40179c:	c4 e2 f9 a9 0c fd 98 	vfmadd213sd 0x622498(,%rdi,8),%xmm0,%xmm1
  4017a3:	24 62 00 
  4017a6:	c5 fb 11 0c fd 98 24 	vmovsd %xmm1,0x622498(,%rdi,8)
  4017ad:	62 00 
  4017af:	c5 f9 2e 15 e1 03 00 	vucomisd 0x3e1(%rip),%xmm2        # 401b98 <_IO_stdin_used+0x18>
  4017b6:	00 
  4017b7:	72 0a                	jb     4017c3 <_Z14codegen_updateP4Bodyd+0x83>
  4017b9:	c5 f9 2e 15 ff 03 00 	vucomisd 0x3ff(%rip),%xmm2        # 401bc0 <_IO_stdin_used+0x40>
  4017c0:	00 
  4017c1:	76 23                	jbe    4017e6 <_Z14codegen_updateP4Bodyd+0xa6>
  4017c3:	c5 fb 10 04 fd a0 1e 	vmovsd 0x631ea0(,%rdi,8),%xmm0
  4017ca:	63 00 
  4017cc:	c5 f9 57 05 0c 04 00 	vxorpd 0x40c(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  4017d3:	00 
  4017d4:	c5 f9 13 04 fd a0 1e 	vmovlpd %xmm0,0x631ea0(,%rdi,8)
  4017db:	63 00 
  4017dd:	c5 fb 10 0c fd 98 24 	vmovsd 0x622498(,%rdi,8),%xmm1
  4017e4:	62 00 
  4017e6:	c5 f9 2e 0d aa 03 00 	vucomisd 0x3aa(%rip),%xmm1        # 401b98 <_IO_stdin_used+0x18>
  4017ed:	00 
  4017ee:	72 0b                	jb     4017fb <_Z14codegen_updateP4Bodyd+0xbb>
  4017f0:	c5 f9 2e 0d c8 03 00 	vucomisd 0x3c8(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  4017f7:	00 
  4017f8:	77 01                	ja     4017fb <_Z14codegen_updateP4Bodyd+0xbb>
  4017fa:	c3                   	retq   
  4017fb:	c5 fb 10 04 fd a8 18 	vmovsd 0x6418a8(,%rdi,8),%xmm0
  401802:	64 00 
  401804:	c5 f8 57 05 d4 03 00 	vxorps 0x3d4(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  40180b:	00 
  40180c:	c5 f8 13 04 fd a8 18 	vmovlps %xmm0,0x6418a8(,%rdi,8)
  401813:	64 00 
  401815:	c3                   	retq   
  401816:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40181d:	00 00 00 

0000000000401820 <_Z26codegen_reset_force_manualP4Body>:
  401820:	48 c7 04 fd b0 12 65 	movq   $0x0,0x6512b0(,%rdi,8)
  401827:	00 00 00 00 00 
  40182c:	48 c7 04 fd b8 0c 66 	movq   $0x0,0x660cb8(,%rdi,8)
  401833:	00 00 00 00 00 
  401838:	c3                   	retq   
  401839:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401840 <_Z24codegen_reset_force_halfP4Body>:
  401840:	48 c7 04 fd b0 12 65 	movq   $0x0,0x6512b0(,%rdi,8)
  401847:	00 00 00 00 00 
  40184c:	c3                   	retq   
  40184d:	0f 1f 00             	nopl   (%rax)

0000000000401850 <_Z18codegen_reset_massP4Body>:
  401850:	48 c7 04 fd 88 30 60 	movq   $0x0,0x603088(,%rdi,8)
  401857:	00 00 00 00 00 
  40185c:	c3                   	retq   
  40185d:	0f 1f 00             	nopl   (%rax)

0000000000401860 <_Z22codengen_simple_updateP4Bodyd>:
  401860:	b8 64 00 00 00       	mov    $0x64,%eax
  401865:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40186c:	00 00 00 00 
  401870:	c5 fb 59 0c fd b0 12 	vmulsd 0x6512b0(,%rdi,8),%xmm0,%xmm1
  401877:	65 00 
  401879:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401880:	60 00 
  401882:	c5 f3 58 0c fd a0 1e 	vaddsd 0x631ea0(,%rdi,8),%xmm1,%xmm1
  401889:	63 00 
  40188b:	c5 fb 11 0c fd a0 1e 	vmovsd %xmm1,0x631ea0(,%rdi,8)
  401892:	63 00 
  401894:	c5 fb 59 14 fd b8 0c 	vmulsd 0x660cb8(,%rdi,8),%xmm0,%xmm2
  40189b:	66 00 
  40189d:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  4018a4:	60 00 
  4018a6:	c5 eb 58 14 fd a8 18 	vaddsd 0x6418a8(,%rdi,8),%xmm2,%xmm2
  4018ad:	64 00 
  4018af:	c5 fb 11 14 fd a8 18 	vmovsd %xmm2,0x6418a8(,%rdi,8)
  4018b6:	64 00 
  4018b8:	c4 e2 f9 a9 0c fd 90 	vfmadd213sd 0x612a90(,%rdi,8),%xmm0,%xmm1
  4018bf:	2a 61 00 
  4018c2:	c5 fb 11 0c fd 90 2a 	vmovsd %xmm1,0x612a90(,%rdi,8)
  4018c9:	61 00 
  4018cb:	c5 fb 10 0c fd a8 18 	vmovsd 0x6418a8(,%rdi,8),%xmm1
  4018d2:	64 00 
  4018d4:	c4 e2 f9 a9 0c fd 98 	vfmadd213sd 0x622498(,%rdi,8),%xmm0,%xmm1
  4018db:	24 62 00 
  4018de:	83 c0 ff             	add    $0xffffffff,%eax
  4018e1:	c5 fb 11 0c fd 98 24 	vmovsd %xmm1,0x622498(,%rdi,8)
  4018e8:	62 00 
  4018ea:	75 84                	jne    401870 <_Z22codengen_simple_updateP4Bodyd+0x10>
  4018ec:	c3                   	retq   
  4018ed:	0f 1f 00             	nopl   (%rax)

00000000004018f0 <_ZN4Body6updateEd>:
  4018f0:	c5 fb 59 14 fd b0 12 	vmulsd 0x6512b0(,%rdi,8),%xmm0,%xmm2
  4018f7:	65 00 
  4018f9:	c5 fb 10 0d bf 02 00 	vmovsd 0x2bf(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  401900:	00 
  401901:	c5 f3 5e 1c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm3
  401908:	60 00 
  40190a:	c4 e2 e1 a9 14 fd a0 	vfmadd213sd 0x631ea0(,%rdi,8),%xmm3,%xmm2
  401911:	1e 63 00 
  401914:	c5 fb 11 14 fd a0 1e 	vmovsd %xmm2,0x631ea0(,%rdi,8)
  40191b:	63 00 
  40191d:	c5 fb 59 0c fd b8 0c 	vmulsd 0x660cb8(,%rdi,8),%xmm0,%xmm1
  401924:	66 00 
  401926:	c4 e2 e1 a9 0c fd a8 	vfmadd213sd 0x6418a8(,%rdi,8),%xmm3,%xmm1
  40192d:	18 64 00 
  401930:	c5 fb 11 0c fd a8 18 	vmovsd %xmm1,0x6418a8(,%rdi,8)
  401937:	64 00 
  401939:	c4 e2 f9 a9 14 fd 90 	vfmadd213sd 0x612a90(,%rdi,8),%xmm0,%xmm2
  401940:	2a 61 00 
  401943:	c5 fb 11 14 fd 90 2a 	vmovsd %xmm2,0x612a90(,%rdi,8)
  40194a:	61 00 
  40194c:	c4 e2 f9 a9 0c fd 98 	vfmadd213sd 0x622498(,%rdi,8),%xmm0,%xmm1
  401953:	24 62 00 
  401956:	c5 fb 11 0c fd 98 24 	vmovsd %xmm1,0x622498(,%rdi,8)
  40195d:	62 00 
  40195f:	c5 f9 2e 15 31 02 00 	vucomisd 0x231(%rip),%xmm2        # 401b98 <_IO_stdin_used+0x18>
  401966:	00 
  401967:	72 0a                	jb     401973 <_ZN4Body6updateEd+0x83>
  401969:	c5 f9 2e 15 4f 02 00 	vucomisd 0x24f(%rip),%xmm2        # 401bc0 <_IO_stdin_used+0x40>
  401970:	00 
  401971:	76 23                	jbe    401996 <_ZN4Body6updateEd+0xa6>
  401973:	c5 fb 10 04 fd a0 1e 	vmovsd 0x631ea0(,%rdi,8),%xmm0
  40197a:	63 00 
  40197c:	c5 f9 57 05 5c 02 00 	vxorpd 0x25c(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  401983:	00 
  401984:	c5 f9 13 04 fd a0 1e 	vmovlpd %xmm0,0x631ea0(,%rdi,8)
  40198b:	63 00 
  40198d:	c5 fb 10 0c fd 98 24 	vmovsd 0x622498(,%rdi,8),%xmm1
  401994:	62 00 
  401996:	c5 f9 2e 0d fa 01 00 	vucomisd 0x1fa(%rip),%xmm1        # 401b98 <_IO_stdin_used+0x18>
  40199d:	00 
  40199e:	72 0b                	jb     4019ab <_ZN4Body6updateEd+0xbb>
  4019a0:	c5 f9 2e 0d 18 02 00 	vucomisd 0x218(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  4019a7:	00 
  4019a8:	77 01                	ja     4019ab <_ZN4Body6updateEd+0xbb>
  4019aa:	c3                   	retq   
  4019ab:	c5 fb 10 04 fd a8 18 	vmovsd 0x6418a8(,%rdi,8),%xmm0
  4019b2:	64 00 
  4019b4:	c5 f8 57 05 24 02 00 	vxorps 0x224(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  4019bb:	00 
  4019bc:	c5 f8 13 04 fd a8 18 	vmovlps %xmm0,0x6418a8(,%rdi,8)
  4019c3:	64 00 
  4019c5:	c3                   	retq   
  4019c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4019cd:	00 00 00 

00000000004019d0 <_ZN4Body22codengen_simple_updateEd>:
  4019d0:	b8 64 00 00 00       	mov    $0x64,%eax
  4019d5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4019dc:	00 00 00 00 
  4019e0:	c5 fb 59 0c fd b0 12 	vmulsd 0x6512b0(,%rdi,8),%xmm0,%xmm1
  4019e7:	65 00 
  4019e9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4019f0:	60 00 
  4019f2:	c5 f3 58 0c fd a0 1e 	vaddsd 0x631ea0(,%rdi,8),%xmm1,%xmm1
  4019f9:	63 00 
  4019fb:	c5 fb 11 0c fd a0 1e 	vmovsd %xmm1,0x631ea0(,%rdi,8)
  401a02:	63 00 
  401a04:	c5 fb 59 14 fd b8 0c 	vmulsd 0x660cb8(,%rdi,8),%xmm0,%xmm2
  401a0b:	66 00 
  401a0d:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  401a14:	60 00 
  401a16:	c5 eb 58 14 fd a8 18 	vaddsd 0x6418a8(,%rdi,8),%xmm2,%xmm2
  401a1d:	64 00 
  401a1f:	c5 fb 11 14 fd a8 18 	vmovsd %xmm2,0x6418a8(,%rdi,8)
  401a26:	64 00 
  401a28:	c4 e2 f9 a9 0c fd 90 	vfmadd213sd 0x612a90(,%rdi,8),%xmm0,%xmm1
  401a2f:	2a 61 00 
  401a32:	c5 fb 11 0c fd 90 2a 	vmovsd %xmm1,0x612a90(,%rdi,8)
  401a39:	61 00 
  401a3b:	c5 fb 10 0c fd a8 18 	vmovsd 0x6418a8(,%rdi,8),%xmm1
  401a42:	64 00 
  401a44:	c4 e2 f9 a9 0c fd 98 	vfmadd213sd 0x622498(,%rdi,8),%xmm0,%xmm1
  401a4b:	24 62 00 
  401a4e:	83 c0 ff             	add    $0xffffffff,%eax
  401a51:	c5 fb 11 0c fd 98 24 	vmovsd %xmm1,0x622498(,%rdi,8)
  401a58:	62 00 
  401a5a:	75 84                	jne    4019e0 <_ZN4Body22codengen_simple_updateEd+0x10>
  401a5c:	c3                   	retq   
  401a5d:	0f 1f 00             	nopl   (%rax)

0000000000401a60 <_ZN4Body9add_forceEPS_>:
  401a60:	48 39 f7             	cmp    %rsi,%rdi
  401a63:	0f 84 91 00 00 00    	je     401afa <_ZN4Body9add_forceEPS_+0x9a>
  401a69:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  401a70:	61 00 
  401a72:	c5 fb 5c 04 fd 90 2a 	vsubsd 0x612a90(,%rdi,8),%xmm0,%xmm0
  401a79:	61 00 
  401a7b:	c5 fb 10 0c f5 98 24 	vmovsd 0x622498(,%rsi,8),%xmm1
  401a82:	62 00 
  401a84:	c5 f3 5c 0c fd 98 24 	vsubsd 0x622498(,%rdi,8),%xmm1,%xmm1
  401a8b:	62 00 
  401a8d:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401a91:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401a96:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  401a9a:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  401aa1:	60 00 
  401aa3:	c5 db 59 25 1d 01 00 	vmulsd 0x11d(%rip),%xmm4,%xmm4        # 401bc8 <_IO_stdin_used+0x48>
  401aaa:	00 
  401aab:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401ab2:	60 00 
  401ab4:	c5 eb 58 15 14 01 00 	vaddsd 0x114(%rip),%xmm2,%xmm2        # 401bd0 <_IO_stdin_used+0x50>
  401abb:	00 
  401abc:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401ac0:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401ac4:	c5 fb 10 25 f4 00 00 	vmovsd 0xf4(%rip),%xmm4        # 401bc0 <_IO_stdin_used+0x40>
  401acb:	00 
  401acc:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401ad0:	c4 e2 e1 a9 04 fd b0 	vfmadd213sd 0x6512b0(,%rdi,8),%xmm3,%xmm0
  401ad7:	12 65 00 
  401ada:	c5 fb 11 04 fd b0 12 	vmovsd %xmm0,0x6512b0(,%rdi,8)
  401ae1:	65 00 
  401ae3:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401ae7:	c4 e2 e1 a9 04 fd b8 	vfmadd213sd 0x660cb8(,%rdi,8),%xmm3,%xmm0
  401aee:	0c 66 00 
  401af1:	c5 fb 11 04 fd b8 0c 	vmovsd %xmm0,0x660cb8(,%rdi,8)
  401af8:	66 00 
  401afa:	c3                   	retq   
  401afb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401b00 <__libc_csu_init>:
  401b00:	41 57                	push   %r15
  401b02:	41 56                	push   %r14
  401b04:	41 89 ff             	mov    %edi,%r15d
  401b07:	41 55                	push   %r13
  401b09:	41 54                	push   %r12
  401b0b:	4c 8d 25 be 12 20 00 	lea    0x2012be(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401b12:	55                   	push   %rbp
  401b13:	48 8d 2d ce 12 20 00 	lea    0x2012ce(%rip),%rbp        # 602de8 <__init_array_end>
  401b1a:	53                   	push   %rbx
  401b1b:	49 89 f6             	mov    %rsi,%r14
  401b1e:	49 89 d5             	mov    %rdx,%r13
  401b21:	4c 29 e5             	sub    %r12,%rbp
  401b24:	48 83 ec 08          	sub    $0x8,%rsp
  401b28:	48 c1 fd 03          	sar    $0x3,%rbp
  401b2c:	e8 1f eb ff ff       	callq  400650 <_init>
  401b31:	48 85 ed             	test   %rbp,%rbp
  401b34:	74 20                	je     401b56 <__libc_csu_init+0x56>
  401b36:	31 db                	xor    %ebx,%ebx
  401b38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401b3f:	00 
  401b40:	4c 89 ea             	mov    %r13,%rdx
  401b43:	4c 89 f6             	mov    %r14,%rsi
  401b46:	44 89 ff             	mov    %r15d,%edi
  401b49:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401b4d:	48 83 c3 01          	add    $0x1,%rbx
  401b51:	48 39 eb             	cmp    %rbp,%rbx
  401b54:	75 ea                	jne    401b40 <__libc_csu_init+0x40>
  401b56:	48 83 c4 08          	add    $0x8,%rsp
  401b5a:	5b                   	pop    %rbx
  401b5b:	5d                   	pop    %rbp
  401b5c:	41 5c                	pop    %r12
  401b5e:	41 5d                	pop    %r13
  401b60:	41 5e                	pop    %r14
  401b62:	41 5f                	pop    %r15
  401b64:	c3                   	retq   
  401b65:	90                   	nop
  401b66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401b6d:	00 00 00 

0000000000401b70 <__libc_csu_fini>:
  401b70:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401b74 <_fini>:
  401b74:	48 83 ec 08          	sub    $0x8,%rsp
  401b78:	48 83 c4 08          	add    $0x8,%rsp
  401b7c:	c3                   	retq   
