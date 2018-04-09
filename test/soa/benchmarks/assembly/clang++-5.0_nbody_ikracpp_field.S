
bin/clang++-5.0_nbody_ikracpp_field:     file format elf64-x86-64


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

0000000000400730 <_GLOBAL__sub_I_ikracpp_field.cc>:
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
  40075f:	49 c7 c0 a0 1b 40 00 	mov    $0x401ba0,%r8
  400766:	48 c7 c1 30 1b 40 00 	mov    $0x401b30,%rcx
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
  400889:	c5 fb 10 05 2f 13 00 	vmovsd 0x132f(%rip),%xmm0        # 401bc0 <_IO_stdin_used+0x10>
  400890:	00 
  400891:	c4 e2 f9 99 0d 1e 13 	vfmadd132sd 0x131e(%rip),%xmm0,%xmm1        # 401bb8 <_IO_stdin_used+0x8>
  400898:	00 00 
  40089a:	c5 fb 11 4c 24 18    	vmovsd %xmm1,0x18(%rsp)
  4008a0:	e8 4b fe ff ff       	callq  4006f0 <rand@plt>
  4008a5:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  4008a9:	c5 fb 10 05 17 13 00 	vmovsd 0x1317(%rip),%xmm0        # 401bc8 <_IO_stdin_used+0x18>
  4008b0:	00 
  4008b1:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008b5:	c5 fb 10 05 13 13 00 	vmovsd 0x1313(%rip),%xmm0        # 401bd0 <_IO_stdin_used+0x20>
  4008bc:	00 
  4008bd:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008c2:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  4008c8:	e8 23 fe ff ff       	callq  4006f0 <rand@plt>
  4008cd:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  4008d1:	c5 fb 10 0d ef 12 00 	vmovsd 0x12ef(%rip),%xmm1        # 401bc8 <_IO_stdin_used+0x18>
  4008d8:	00 
  4008d9:	c4 e2 f1 99 05 ee 12 	vfmadd132sd 0x12ee(%rip),%xmm1,%xmm0        # 401bd0 <_IO_stdin_used+0x20>
  4008e0:	00 00 
  4008e2:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
  4008e8:	e8 03 fe ff ff       	callq  4006f0 <rand@plt>
  4008ed:	c5 e3 2a c8          	vcvtsi2sd %eax,%xmm3,%xmm1
  4008f1:	c5 fb 10 05 df 12 00 	vmovsd 0x12df(%rip),%xmm0        # 401bd8 <_IO_stdin_used+0x28>
  4008f8:	00 
  4008f9:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008fd:	c5 fb 10 05 db 12 00 	vmovsd 0x12db(%rip),%xmm0        # 401be0 <_IO_stdin_used+0x30>
  400904:	00 
  400905:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  40090a:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  40090f:	e8 dc fd ff ff       	callq  4006f0 <rand@plt>
  400914:	c5 e3 2a c0          	vcvtsi2sd %eax,%xmm3,%xmm0
  400918:	c5 fb 10 0d b8 12 00 	vmovsd 0x12b8(%rip),%xmm1        # 401bd8 <_IO_stdin_used+0x28>
  40091f:	00 
  400920:	c4 e2 f1 99 05 b7 12 	vfmadd132sd 0x12b7(%rip),%xmm1,%xmm0        # 401be0 <_IO_stdin_used+0x30>
  400927:	00 00 
  400929:	48 8b 05 48 27 20 00 	mov    0x202748(%rip),%rax        # 603078 <__Bodydata_buffer>
  400930:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400934:	48 89 0d 3d 27 20 00 	mov    %rcx,0x20273d(%rip)        # 603078 <__Bodydata_buffer>
  40093b:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  400941:	c5 fb 11 0c c5 90 30 	vmovsd %xmm1,0x603090(,%rax,8)
  400948:	60 00 
  40094a:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  400950:	c5 fb 11 0c c5 90 2a 	vmovsd %xmm1,0x612a90(,%rax,8)
  400957:	61 00 
  400959:	c5 fb 10 4c 24 08    	vmovsd 0x8(%rsp),%xmm1
  40095f:	c5 fb 11 0c c5 90 24 	vmovsd %xmm1,0x622490(,%rax,8)
  400966:	62 00 
  400968:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  40096d:	c5 fb 11 0c c5 90 1e 	vmovsd %xmm1,0x631e90(,%rax,8)
  400974:	63 00 
  400976:	c5 fb 11 04 c5 90 18 	vmovsd %xmm0,0x641890(,%rax,8)
  40097d:	64 00 
  40097f:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  400986:	00 00 00 00 00 
  40098b:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
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
  4009c0:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  4009c7:	00 00 00 00 00 
  4009cc:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  4009d3:	00 00 00 00 00 
  4009d8:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  4009df:	00 00 00 00 00 
  4009e4:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  4009eb:	00 00 00 00 00 
  4009f0:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  4009f7:	00 00 00 00 00 
  4009fc:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  400a03:	00 00 00 00 00 
  400a08:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  400a0f:	00 00 00 00 00 
  400a14:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  400a1b:	00 00 00 00 00 
  400a20:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  400a27:	00 00 00 00 00 
  400a2c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  400a33:	00 00 00 00 00 
  400a38:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400a3f:	00 00 00 00 00 
  400a44:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  400a4b:	00 00 00 00 00 
  400a50:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400a57:	00 00 00 00 00 
  400a5c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400a63:	00 00 00 00 00 
  400a68:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400a6f:	00 00 00 00 00 
  400a74:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400a7b:	00 00 00 00 00 
  400a80:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400a87:	00 00 00 00 00 
  400a8c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400a93:	00 00 00 00 00 
  400a98:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400a9f:	00 00 00 00 00 
  400aa4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400aab:	00 00 00 00 00 
  400ab0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400ab7:	00 00 00 00 00 
  400abc:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400ac3:	00 00 00 00 00 
  400ac8:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400acf:	00 00 00 00 00 
  400ad4:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400adb:	00 00 00 00 00 
  400ae0:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400ae7:	00 00 00 00 00 
  400aec:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400af3:	00 00 00 00 00 
  400af8:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400aff:	00 00 00 00 00 
  400b04:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400b0b:	00 00 00 00 00 
  400b10:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400b17:	00 00 00 00 00 
  400b1c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400b23:	00 00 00 00 00 
  400b28:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400b2f:	00 00 00 00 00 
  400b34:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
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
  400b70:	48 89 df             	mov    %rbx,%rdi
  400b73:	4c 89 f6             	mov    %r14,%rsi
  400b76:	e8 15 0f 00 00       	callq  401a90 <_ZN4Body9add_forceEPS_>
  400b7b:	48 83 c3 01          	add    $0x1,%rbx
  400b7f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400b86:	75 e8                	jne    400b70 <_Z14run_simulationv+0x1c0>
  400b88:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400b8f:	75 cf                	jne    400b60 <_Z14run_simulationv+0x1b0>
  400b91:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 
  400ba0:	48 89 df             	mov    %rbx,%rdi
  400ba3:	c5 fb 10 05 3d 10 00 	vmovsd 0x103d(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  400baa:	00 
  400bab:	e8 60 0d 00 00       	callq  401910 <_ZN4Body6updateEd>
  400bb0:	48 83 c3 01          	add    $0x1,%rbx
  400bb4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bbb:	75 e3                	jne    400ba0 <_Z14run_simulationv+0x1f0>
  400bbd:	b8 10 00 00 00       	mov    $0x10,%eax
  400bc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400bc9:	1f 84 00 00 00 00 00 
  400bd0:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  400bd7:	00 00 00 00 00 
  400bdc:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  400be3:	00 00 00 00 00 
  400be8:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  400bef:	00 00 00 00 00 
  400bf4:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  400bfb:	00 00 00 00 00 
  400c00:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  400c07:	00 00 00 00 00 
  400c0c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  400c13:	00 00 00 00 00 
  400c18:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  400c1f:	00 00 00 00 00 
  400c24:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  400c2b:	00 00 00 00 00 
  400c30:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  400c37:	00 00 00 00 00 
  400c3c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  400c43:	00 00 00 00 00 
  400c48:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400c4f:	00 00 00 00 00 
  400c54:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  400c5b:	00 00 00 00 00 
  400c60:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400c67:	00 00 00 00 00 
  400c6c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400c73:	00 00 00 00 00 
  400c78:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400c7f:	00 00 00 00 00 
  400c84:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400c8b:	00 00 00 00 00 
  400c90:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400c97:	00 00 00 00 00 
  400c9c:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400ca3:	00 00 00 00 00 
  400ca8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400caf:	00 00 00 00 00 
  400cb4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400cbb:	00 00 00 00 00 
  400cc0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400cc7:	00 00 00 00 00 
  400ccc:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400cd3:	00 00 00 00 00 
  400cd8:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400cdf:	00 00 00 00 00 
  400ce4:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400ceb:	00 00 00 00 00 
  400cf0:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400cf7:	00 00 00 00 00 
  400cfc:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400d03:	00 00 00 00 00 
  400d08:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400d0f:	00 00 00 00 00 
  400d14:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400d1b:	00 00 00 00 00 
  400d20:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400d27:	00 00 00 00 00 
  400d2c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400d33:	00 00 00 00 00 
  400d38:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400d3f:	00 00 00 00 00 
  400d44:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
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
  400d80:	48 89 df             	mov    %rbx,%rdi
  400d83:	4c 89 f6             	mov    %r14,%rsi
  400d86:	e8 05 0d 00 00       	callq  401a90 <_ZN4Body9add_forceEPS_>
  400d8b:	48 83 c3 01          	add    $0x1,%rbx
  400d8f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400d96:	75 e8                	jne    400d80 <_Z14run_simulationv+0x3d0>
  400d98:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400d9f:	75 cf                	jne    400d70 <_Z14run_simulationv+0x3c0>
  400da1:	bb 01 00 00 00       	mov    $0x1,%ebx
  400da6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dad:	00 00 00 
  400db0:	48 89 df             	mov    %rbx,%rdi
  400db3:	c5 fb 10 05 2d 0e 00 	vmovsd 0xe2d(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  400dba:	00 
  400dbb:	e8 50 0b 00 00       	callq  401910 <_ZN4Body6updateEd>
  400dc0:	48 83 c3 01          	add    $0x1,%rbx
  400dc4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400dcb:	75 e3                	jne    400db0 <_Z14run_simulationv+0x400>
  400dcd:	b8 10 00 00 00       	mov    $0x10,%eax
  400dd2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400dd9:	1f 84 00 00 00 00 00 
  400de0:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  400de7:	00 00 00 00 00 
  400dec:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  400df3:	00 00 00 00 00 
  400df8:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  400dff:	00 00 00 00 00 
  400e04:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  400e0b:	00 00 00 00 00 
  400e10:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  400e17:	00 00 00 00 00 
  400e1c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  400e23:	00 00 00 00 00 
  400e28:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  400e2f:	00 00 00 00 00 
  400e34:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  400e3b:	00 00 00 00 00 
  400e40:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  400e47:	00 00 00 00 00 
  400e4c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  400e53:	00 00 00 00 00 
  400e58:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400e5f:	00 00 00 00 00 
  400e64:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  400e6b:	00 00 00 00 00 
  400e70:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400e77:	00 00 00 00 00 
  400e7c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400e83:	00 00 00 00 00 
  400e88:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400e8f:	00 00 00 00 00 
  400e94:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400e9b:	00 00 00 00 00 
  400ea0:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400ea7:	00 00 00 00 00 
  400eac:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400eb3:	00 00 00 00 00 
  400eb8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400ebf:	00 00 00 00 00 
  400ec4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400ecb:	00 00 00 00 00 
  400ed0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400ed7:	00 00 00 00 00 
  400edc:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400ee3:	00 00 00 00 00 
  400ee8:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400eef:	00 00 00 00 00 
  400ef4:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400efb:	00 00 00 00 00 
  400f00:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400f07:	00 00 00 00 00 
  400f0c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400f13:	00 00 00 00 00 
  400f18:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400f1f:	00 00 00 00 00 
  400f24:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400f2b:	00 00 00 00 00 
  400f30:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400f37:	00 00 00 00 00 
  400f3c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400f43:	00 00 00 00 00 
  400f48:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400f4f:	00 00 00 00 00 
  400f54:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
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
  400f90:	48 89 df             	mov    %rbx,%rdi
  400f93:	4c 89 f6             	mov    %r14,%rsi
  400f96:	e8 f5 0a 00 00       	callq  401a90 <_ZN4Body9add_forceEPS_>
  400f9b:	48 83 c3 01          	add    $0x1,%rbx
  400f9f:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fa6:	75 e8                	jne    400f90 <_Z14run_simulationv+0x5e0>
  400fa8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  400faf:	75 cf                	jne    400f80 <_Z14run_simulationv+0x5d0>
  400fb1:	bb 01 00 00 00       	mov    $0x1,%ebx
  400fb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fbd:	00 00 00 
  400fc0:	48 89 df             	mov    %rbx,%rdi
  400fc3:	c5 fb 10 05 1d 0c 00 	vmovsd 0xc1d(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  400fca:	00 
  400fcb:	e8 40 09 00 00       	callq  401910 <_ZN4Body6updateEd>
  400fd0:	48 83 c3 01          	add    $0x1,%rbx
  400fd4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fdb:	75 e3                	jne    400fc0 <_Z14run_simulationv+0x610>
  400fdd:	b8 10 00 00 00       	mov    $0x10,%eax
  400fe2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400fe9:	1f 84 00 00 00 00 00 
  400ff0:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  400ff7:	00 00 00 00 00 
  400ffc:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  401003:	00 00 00 00 00 
  401008:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  40100f:	00 00 00 00 00 
  401014:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  40101b:	00 00 00 00 00 
  401020:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  401027:	00 00 00 00 00 
  40102c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  401033:	00 00 00 00 00 
  401038:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  40103f:	00 00 00 00 00 
  401044:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  40104b:	00 00 00 00 00 
  401050:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  401057:	00 00 00 00 00 
  40105c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  401063:	00 00 00 00 00 
  401068:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  40106f:	00 00 00 00 00 
  401074:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  40107b:	00 00 00 00 00 
  401080:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  401087:	00 00 00 00 00 
  40108c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  401093:	00 00 00 00 00 
  401098:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  40109f:	00 00 00 00 00 
  4010a4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  4010ab:	00 00 00 00 00 
  4010b0:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  4010b7:	00 00 00 00 00 
  4010bc:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  4010c3:	00 00 00 00 00 
  4010c8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  4010cf:	00 00 00 00 00 
  4010d4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  4010db:	00 00 00 00 00 
  4010e0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  4010e7:	00 00 00 00 00 
  4010ec:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  4010f3:	00 00 00 00 00 
  4010f8:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  4010ff:	00 00 00 00 00 
  401104:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  40110b:	00 00 00 00 00 
  401110:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  401117:	00 00 00 00 00 
  40111c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  401123:	00 00 00 00 00 
  401128:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  40112f:	00 00 00 00 00 
  401134:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  40113b:	00 00 00 00 00 
  401140:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  401147:	00 00 00 00 00 
  40114c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  401153:	00 00 00 00 00 
  401158:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  40115f:	00 00 00 00 00 
  401164:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
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
  4011a0:	48 89 df             	mov    %rbx,%rdi
  4011a3:	4c 89 f6             	mov    %r14,%rsi
  4011a6:	e8 e5 08 00 00       	callq  401a90 <_ZN4Body9add_forceEPS_>
  4011ab:	48 83 c3 01          	add    $0x1,%rbx
  4011af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011b6:	75 e8                	jne    4011a0 <_Z14run_simulationv+0x7f0>
  4011b8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  4011bf:	75 cf                	jne    401190 <_Z14run_simulationv+0x7e0>
  4011c1:	bb 01 00 00 00       	mov    $0x1,%ebx
  4011c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011cd:	00 00 00 
  4011d0:	48 89 df             	mov    %rbx,%rdi
  4011d3:	c5 fb 10 05 0d 0a 00 	vmovsd 0xa0d(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  4011da:	00 
  4011db:	e8 30 07 00 00       	callq  401910 <_ZN4Body6updateEd>
  4011e0:	48 83 c3 01          	add    $0x1,%rbx
  4011e4:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011eb:	75 e3                	jne    4011d0 <_Z14run_simulationv+0x820>
  4011ed:	b8 10 00 00 00       	mov    $0x10,%eax
  4011f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4011f9:	1f 84 00 00 00 00 00 
  401200:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  401207:	00 00 00 00 00 
  40120c:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  401213:	00 00 00 00 00 
  401218:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  40121f:	00 00 00 00 00 
  401224:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  40122b:	00 00 00 00 00 
  401230:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  401237:	00 00 00 00 00 
  40123c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  401243:	00 00 00 00 00 
  401248:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  40124f:	00 00 00 00 00 
  401254:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  40125b:	00 00 00 00 00 
  401260:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  401267:	00 00 00 00 00 
  40126c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  401273:	00 00 00 00 00 
  401278:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  40127f:	00 00 00 00 00 
  401284:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  40128b:	00 00 00 00 00 
  401290:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  401297:	00 00 00 00 00 
  40129c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  4012a3:	00 00 00 00 00 
  4012a8:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  4012af:	00 00 00 00 00 
  4012b4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  4012bb:	00 00 00 00 00 
  4012c0:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  4012c7:	00 00 00 00 00 
  4012cc:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  4012d3:	00 00 00 00 00 
  4012d8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  4012df:	00 00 00 00 00 
  4012e4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  4012eb:	00 00 00 00 00 
  4012f0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  4012f7:	00 00 00 00 00 
  4012fc:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  401303:	00 00 00 00 00 
  401308:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  40130f:	00 00 00 00 00 
  401314:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  40131b:	00 00 00 00 00 
  401320:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  401327:	00 00 00 00 00 
  40132c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  401333:	00 00 00 00 00 
  401338:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  40133f:	00 00 00 00 00 
  401344:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  40134b:	00 00 00 00 00 
  401350:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  401357:	00 00 00 00 00 
  40135c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  401363:	00 00 00 00 00 
  401368:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  40136f:	00 00 00 00 00 
  401374:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
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
  4013b0:	48 89 df             	mov    %rbx,%rdi
  4013b3:	4c 89 f6             	mov    %r14,%rsi
  4013b6:	e8 d5 06 00 00       	callq  401a90 <_ZN4Body9add_forceEPS_>
  4013bb:	48 83 c3 01          	add    $0x1,%rbx
  4013bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013c6:	75 e8                	jne    4013b0 <_Z14run_simulationv+0xa00>
  4013c8:	49 81 fe 40 1f 00 00 	cmp    $0x1f40,%r14
  4013cf:	75 cf                	jne    4013a0 <_Z14run_simulationv+0x9f0>
  4013d1:	bb 01 00 00 00       	mov    $0x1,%ebx
  4013d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4013dd:	00 00 00 
  4013e0:	48 89 df             	mov    %rbx,%rdi
  4013e3:	c5 fb 10 05 fd 07 00 	vmovsd 0x7fd(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  4013ea:	00 
  4013eb:	e8 20 05 00 00       	callq  401910 <_ZN4Body6updateEd>
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
  401433:	c5 fb 10 05 ad 07 00 	vmovsd 0x7ad(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  40143a:	00 
  40143b:	e8 b0 05 00 00       	callq  4019f0 <_ZN4Body22codengen_simple_updateEd>
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
  4014b1:	c5 fb 10 05 2f 07 00 	vmovsd 0x72f(%rip),%xmm0        # 401be8 <_IO_stdin_used+0x38>
  4014b8:	00 
  4014b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014c0:	31 c0                	xor    %eax,%eax
  4014c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4014c9:	1f 84 00 00 00 00 00 
  4014d0:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4014d4:	ba 64 00 00 00       	mov    $0x64,%edx
  4014d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014e0:	c5 fb 59 0c cd 88 12 	vmulsd 0x651288(,%rcx,8),%xmm0,%xmm1
  4014e7:	65 00 
  4014e9:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  4014f0:	60 00 
  4014f2:	c5 f3 58 0c cd 88 1e 	vaddsd 0x631e88(,%rcx,8),%xmm1,%xmm1
  4014f9:	63 00 
  4014fb:	c5 fb 11 0c cd 88 1e 	vmovsd %xmm1,0x631e88(,%rcx,8)
  401502:	63 00 
  401504:	c5 fb 59 0c cd 88 0c 	vmulsd 0x660c88(,%rcx,8),%xmm0,%xmm1
  40150b:	66 00 
  40150d:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  401514:	60 00 
  401516:	c5 f3 58 0c cd 88 18 	vaddsd 0x641888(,%rcx,8),%xmm1,%xmm1
  40151d:	64 00 
  40151f:	c5 fb 11 0c cd 88 18 	vmovsd %xmm1,0x641888(,%rcx,8)
  401526:	64 00 
  401528:	c5 fb 10 0c cd 88 1e 	vmovsd 0x631e88(,%rcx,8),%xmm1
  40152f:	63 00 
  401531:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x612a88(,%rcx,8),%xmm0,%xmm1
  401538:	2a 61 00 
  40153b:	c5 fb 11 0c cd 88 2a 	vmovsd %xmm1,0x612a88(,%rcx,8)
  401542:	61 00 
  401544:	c5 fb 10 0c cd 88 18 	vmovsd 0x641888(,%rcx,8),%xmm1
  40154b:	64 00 
  40154d:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x622488(,%rcx,8),%xmm0,%xmm1
  401554:	24 62 00 
  401557:	83 c2 ff             	add    $0xffffffff,%edx
  40155a:	c5 fb 11 0c cd 88 24 	vmovsd %xmm1,0x622488(,%rcx,8)
  401561:	62 00 
  401563:	0f 85 77 ff ff ff    	jne    4014e0 <main+0x80>
  401569:	48 89 c8             	mov    %rcx,%rax
  40156c:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401573:	0f 85 57 ff ff ff    	jne    4014d0 <main+0x70>
  401579:	83 c3 01             	add    $0x1,%ebx
  40157c:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  401582:	0f 85 38 ff ff ff    	jne    4014c0 <main+0x60>
  401588:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  40158f:	de 1b 43 
  401592:	4c 89 f0             	mov    %r14,%rax
  401595:	48 f7 ee             	imul   %rsi
  401598:	49 89 d6             	mov    %rdx,%r14
  40159b:	4c 89 f1             	mov    %r14,%rcx
  40159e:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4015a2:	49 c1 fe 12          	sar    $0x12,%r14
  4015a6:	48 89 e8             	mov    %rbp,%rax
  4015a9:	48 f7 ee             	imul   %rsi
  4015ac:	49 89 d5             	mov    %rdx,%r13
  4015af:	49 01 ce             	add    %rcx,%r14
  4015b2:	4c 89 e8             	mov    %r13,%rax
  4015b5:	48 c1 e8 3f          	shr    $0x3f,%rax
  4015b9:	49 c1 fd 12          	sar    $0x12,%r13
  4015bd:	49 01 c5             	add    %rax,%r13
  4015c0:	e8 1b f1 ff ff       	callq  4006e0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4015c5:	49 89 c4             	mov    %rax,%r12
  4015c8:	4d 29 fc             	sub    %r15,%r12
  4015cb:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4015d0:	31 f6                	xor    %esi,%esi
  4015d2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4015d9:	1f 84 00 00 00 00 00 
  4015e0:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  4015e7:	61 00 
  4015e9:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  4015ed:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  4015f1:	01 e8                	add    %ebp,%eax
  4015f3:	c5 fb 10 0c f5 90 24 	vmovsd 0x622490(,%rsi,8),%xmm1
  4015fa:	62 00 
  4015fc:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  401600:	c5 f9 7e d5          	vmovd  %xmm2,%ebp
  401604:	01 c5                	add    %eax,%ebp
  401606:	48 63 c5             	movslq %ebp,%rax
  401609:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  401610:	48 8d 5e 01          	lea    0x1(%rsi),%rbx
  401614:	48 89 c1             	mov    %rax,%rcx
  401617:	48 c1 e9 3f          	shr    $0x3f,%rcx
  40161b:	48 c1 f8 33          	sar    $0x33,%rax
  40161f:	01 c8                	add    %ecx,%eax
  401621:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  401628:	48 83 fe 09          	cmp    $0x9,%rsi
  40162c:	77 0c                	ja     40163a <main+0x1da>
  40162e:	bf 20 1c 40 00       	mov    $0x401c20,%edi
  401633:	b0 02                	mov    $0x2,%al
  401635:	e8 46 f0 ff ff       	callq  400680 <printf@plt>
  40163a:	44 29 fd             	sub    %r15d,%ebp
  40163d:	48 89 de             	mov    %rbx,%rsi
  401640:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401647:	75 97                	jne    4015e0 <main+0x180>
  401649:	4c 89 e0             	mov    %r12,%rax
  40164c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401653:	de 1b 43 
  401656:	48 f7 e9             	imul   %rcx
  401659:	48 89 d0             	mov    %rdx,%rax
  40165c:	48 c1 e8 3f          	shr    $0x3f,%rax
  401660:	48 c1 fa 12          	sar    $0x12,%rdx
  401664:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401668:	bf 35 1c 40 00       	mov    $0x401c35,%edi
  40166d:	31 c0                	xor    %eax,%eax
  40166f:	4c 89 f6             	mov    %r14,%rsi
  401672:	4c 89 ea             	mov    %r13,%rdx
  401675:	41 89 e8             	mov    %ebp,%r8d
  401678:	e8 03 f0 ff ff       	callq  400680 <printf@plt>
  40167d:	31 c0                	xor    %eax,%eax
  40167f:	48 83 c4 08          	add    $0x8,%rsp
  401683:	5b                   	pop    %rbx
  401684:	41 5c                	pop    %r12
  401686:	41 5d                	pop    %r13
  401688:	41 5e                	pop    %r14
  40168a:	41 5f                	pop    %r15
  40168c:	5d                   	pop    %rbp
  40168d:	c3                   	retq   
  40168e:	66 90                	xchg   %ax,%ax

0000000000401690 <_Z19codegen_reset_forceP4Body>:
  401690:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401697:	00 00 00 00 00 
  40169c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  4016a3:	00 00 00 00 00 
  4016a8:	c3                   	retq   
  4016a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004016b0 <_Z17codegen_add_forceP4BodyS0_>:
  4016b0:	48 39 f7             	cmp    %rsi,%rdi
  4016b3:	0f 84 91 00 00 00    	je     40174a <_Z17codegen_add_forceP4BodyS0_+0x9a>
  4016b9:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  4016c0:	61 00 
  4016c2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  4016c9:	61 00 
  4016cb:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  4016d2:	62 00 
  4016d4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  4016db:	62 00 
  4016dd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  4016e1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  4016e6:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  4016ea:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  4016f1:	60 00 
  4016f3:	c5 db 59 25 fd 04 00 	vmulsd 0x4fd(%rip),%xmm4,%xmm4        # 401bf8 <_IO_stdin_used+0x48>
  4016fa:	00 
  4016fb:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401702:	60 00 
  401704:	c5 eb 58 15 f4 04 00 	vaddsd 0x4f4(%rip),%xmm2,%xmm2        # 401c00 <_IO_stdin_used+0x50>
  40170b:	00 
  40170c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401710:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401714:	c5 fb 10 25 d4 04 00 	vmovsd 0x4d4(%rip),%xmm4        # 401bf0 <_IO_stdin_used+0x40>
  40171b:	00 
  40171c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401720:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401727:	12 65 00 
  40172a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401731:	65 00 
  401733:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401737:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  40173e:	0c 66 00 
  401741:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401748:	66 00 
  40174a:	c3                   	retq   
  40174b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401750 <_Z14codegen_updateP4Bodyd>:
  401750:	c5 fb 59 14 fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm2
  401757:	65 00 
  401759:	c5 fb 10 0d 8f 04 00 	vmovsd 0x48f(%rip),%xmm1        # 401bf0 <_IO_stdin_used+0x40>
  401760:	00 
  401761:	c5 f3 5e 1c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm3
  401768:	60 00 
  40176a:	c4 e2 e1 a9 14 fd 88 	vfmadd213sd 0x631e88(,%rdi,8),%xmm3,%xmm2
  401771:	1e 63 00 
  401774:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  40177b:	63 00 
  40177d:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  401784:	66 00 
  401786:	c4 e2 e1 a9 0c fd 88 	vfmadd213sd 0x641888(,%rdi,8),%xmm3,%xmm1
  40178d:	18 64 00 
  401790:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401797:	64 00 
  401799:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  4017a0:	2a 61 00 
  4017a3:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  4017aa:	61 00 
  4017ac:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4017b3:	24 62 00 
  4017b6:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4017bd:	62 00 
  4017bf:	c5 f9 2e 15 01 04 00 	vucomisd 0x401(%rip),%xmm2        # 401bc8 <_IO_stdin_used+0x18>
  4017c6:	00 
  4017c7:	72 0a                	jb     4017d3 <_Z14codegen_updateP4Bodyd+0x83>
  4017c9:	c5 f9 2e 15 1f 04 00 	vucomisd 0x41f(%rip),%xmm2        # 401bf0 <_IO_stdin_used+0x40>
  4017d0:	00 
  4017d1:	76 23                	jbe    4017f6 <_Z14codegen_updateP4Bodyd+0xa6>
  4017d3:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4017da:	63 00 
  4017dc:	c5 f9 57 05 2c 04 00 	vxorpd 0x42c(%rip),%xmm0,%xmm0        # 401c10 <_IO_stdin_used+0x60>
  4017e3:	00 
  4017e4:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4017eb:	63 00 
  4017ed:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4017f4:	62 00 
  4017f6:	c5 f9 2e 0d ca 03 00 	vucomisd 0x3ca(%rip),%xmm1        # 401bc8 <_IO_stdin_used+0x18>
  4017fd:	00 
  4017fe:	72 0b                	jb     40180b <_Z14codegen_updateP4Bodyd+0xbb>
  401800:	c5 f9 2e 0d e8 03 00 	vucomisd 0x3e8(%rip),%xmm1        # 401bf0 <_IO_stdin_used+0x40>
  401807:	00 
  401808:	77 01                	ja     40180b <_Z14codegen_updateP4Bodyd+0xbb>
  40180a:	c3                   	retq   
  40180b:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  401812:	64 00 
  401814:	c5 f8 57 05 f4 03 00 	vxorps 0x3f4(%rip),%xmm0,%xmm0        # 401c10 <_IO_stdin_used+0x60>
  40181b:	00 
  40181c:	c5 f8 13 04 fd 88 18 	vmovlps %xmm0,0x641888(,%rdi,8)
  401823:	64 00 
  401825:	c3                   	retq   
  401826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40182d:	00 00 00 

0000000000401830 <_Z26codegen_reset_force_manualP4Body>:
  401830:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401837:	00 00 00 00 00 
  40183c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401843:	00 00 00 00 00 
  401848:	c3                   	retq   
  401849:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401850 <_Z24codegen_reset_force_halfP4Body>:
  401850:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401857:	00 00 00 00 00 
  40185c:	c3                   	retq   
  40185d:	0f 1f 00             	nopl   (%rax)

0000000000401860 <_Z18codegen_reset_massP4Body>:
  401860:	48 c7 04 fd 88 30 60 	movq   $0x0,0x603088(,%rdi,8)
  401867:	00 00 00 00 00 
  40186c:	c3                   	retq   
  40186d:	0f 1f 00             	nopl   (%rax)

0000000000401870 <_Z22codengen_simple_updateP4Bodyd>:
  401870:	b8 64 00 00 00       	mov    $0x64,%eax
  401875:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40187c:	00 00 00 00 
  401880:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401887:	65 00 
  401889:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401890:	60 00 
  401892:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401899:	63 00 
  40189b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  4018a2:	63 00 
  4018a4:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  4018ab:	66 00 
  4018ad:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018b4:	60 00 
  4018b6:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  4018bd:	64 00 
  4018bf:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  4018c6:	64 00 
  4018c8:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  4018cf:	63 00 
  4018d1:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4018d8:	2a 61 00 
  4018db:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4018e2:	61 00 
  4018e4:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4018eb:	64 00 
  4018ed:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4018f4:	24 62 00 
  4018f7:	83 c0 ff             	add    $0xffffffff,%eax
  4018fa:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401901:	62 00 
  401903:	0f 85 77 ff ff ff    	jne    401880 <_Z22codengen_simple_updateP4Bodyd+0x10>
  401909:	c3                   	retq   
  40190a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401910 <_ZN4Body6updateEd>:
  401910:	c5 fb 59 14 fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm2
  401917:	65 00 
  401919:	c5 fb 10 0d cf 02 00 	vmovsd 0x2cf(%rip),%xmm1        # 401bf0 <_IO_stdin_used+0x40>
  401920:	00 
  401921:	c5 f3 5e 1c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm3
  401928:	60 00 
  40192a:	c4 e2 e1 a9 14 fd 88 	vfmadd213sd 0x631e88(,%rdi,8),%xmm3,%xmm2
  401931:	1e 63 00 
  401934:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  40193b:	63 00 
  40193d:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  401944:	66 00 
  401946:	c4 e2 e1 a9 0c fd 88 	vfmadd213sd 0x641888(,%rdi,8),%xmm3,%xmm1
  40194d:	18 64 00 
  401950:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401957:	64 00 
  401959:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  401960:	2a 61 00 
  401963:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  40196a:	61 00 
  40196c:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401973:	24 62 00 
  401976:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  40197d:	62 00 
  40197f:	c5 f9 2e 15 41 02 00 	vucomisd 0x241(%rip),%xmm2        # 401bc8 <_IO_stdin_used+0x18>
  401986:	00 
  401987:	72 0a                	jb     401993 <_ZN4Body6updateEd+0x83>
  401989:	c5 f9 2e 15 5f 02 00 	vucomisd 0x25f(%rip),%xmm2        # 401bf0 <_IO_stdin_used+0x40>
  401990:	00 
  401991:	76 23                	jbe    4019b6 <_ZN4Body6updateEd+0xa6>
  401993:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  40199a:	63 00 
  40199c:	c5 f9 57 05 6c 02 00 	vxorpd 0x26c(%rip),%xmm0,%xmm0        # 401c10 <_IO_stdin_used+0x60>
  4019a3:	00 
  4019a4:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4019ab:	63 00 
  4019ad:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4019b4:	62 00 
  4019b6:	c5 f9 2e 0d 0a 02 00 	vucomisd 0x20a(%rip),%xmm1        # 401bc8 <_IO_stdin_used+0x18>
  4019bd:	00 
  4019be:	72 0b                	jb     4019cb <_ZN4Body6updateEd+0xbb>
  4019c0:	c5 f9 2e 0d 28 02 00 	vucomisd 0x228(%rip),%xmm1        # 401bf0 <_IO_stdin_used+0x40>
  4019c7:	00 
  4019c8:	77 01                	ja     4019cb <_ZN4Body6updateEd+0xbb>
  4019ca:	c3                   	retq   
  4019cb:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  4019d2:	64 00 
  4019d4:	c5 f8 57 05 34 02 00 	vxorps 0x234(%rip),%xmm0,%xmm0        # 401c10 <_IO_stdin_used+0x60>
  4019db:	00 
  4019dc:	c5 f8 13 04 fd 88 18 	vmovlps %xmm0,0x641888(,%rdi,8)
  4019e3:	64 00 
  4019e5:	c3                   	retq   
  4019e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4019ed:	00 00 00 

00000000004019f0 <_ZN4Body22codengen_simple_updateEd>:
  4019f0:	b8 64 00 00 00       	mov    $0x64,%eax
  4019f5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4019fc:	00 00 00 00 
  401a00:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401a07:	65 00 
  401a09:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a10:	60 00 
  401a12:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401a19:	63 00 
  401a1b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401a22:	63 00 
  401a24:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  401a2b:	66 00 
  401a2d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a34:	60 00 
  401a36:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  401a3d:	64 00 
  401a3f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401a46:	64 00 
  401a48:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  401a4f:	63 00 
  401a51:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401a58:	2a 61 00 
  401a5b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401a62:	61 00 
  401a64:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401a6b:	64 00 
  401a6d:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401a74:	24 62 00 
  401a77:	83 c0 ff             	add    $0xffffffff,%eax
  401a7a:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401a81:	62 00 
  401a83:	0f 85 77 ff ff ff    	jne    401a00 <_ZN4Body22codengen_simple_updateEd+0x10>
  401a89:	c3                   	retq   
  401a8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401a90 <_ZN4Body9add_forceEPS_>:
  401a90:	48 39 f7             	cmp    %rsi,%rdi
  401a93:	0f 84 91 00 00 00    	je     401b2a <_ZN4Body9add_forceEPS_+0x9a>
  401a99:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  401aa0:	61 00 
  401aa2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  401aa9:	61 00 
  401aab:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  401ab2:	62 00 
  401ab4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  401abb:	62 00 
  401abd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401ac1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401ac6:	c5 eb 51 da          	vsqrtsd %xmm2,%xmm2,%xmm3
  401aca:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  401ad1:	60 00 
  401ad3:	c5 db 59 25 1d 01 00 	vmulsd 0x11d(%rip),%xmm4,%xmm4        # 401bf8 <_IO_stdin_used+0x48>
  401ada:	00 
  401adb:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401ae2:	60 00 
  401ae4:	c5 eb 58 15 14 01 00 	vaddsd 0x114(%rip),%xmm2,%xmm2        # 401c00 <_IO_stdin_used+0x50>
  401aeb:	00 
  401aec:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401af0:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401af4:	c5 fb 10 25 f4 00 00 	vmovsd 0xf4(%rip),%xmm4        # 401bf0 <_IO_stdin_used+0x40>
  401afb:	00 
  401afc:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401b00:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401b07:	12 65 00 
  401b0a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401b11:	65 00 
  401b13:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401b17:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  401b1e:	0c 66 00 
  401b21:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401b28:	66 00 
  401b2a:	c3                   	retq   
  401b2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401b30 <__libc_csu_init>:
  401b30:	41 57                	push   %r15
  401b32:	41 56                	push   %r14
  401b34:	41 89 ff             	mov    %edi,%r15d
  401b37:	41 55                	push   %r13
  401b39:	41 54                	push   %r12
  401b3b:	4c 8d 25 8e 12 20 00 	lea    0x20128e(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401b42:	55                   	push   %rbp
  401b43:	48 8d 2d 9e 12 20 00 	lea    0x20129e(%rip),%rbp        # 602de8 <__init_array_end>
  401b4a:	53                   	push   %rbx
  401b4b:	49 89 f6             	mov    %rsi,%r14
  401b4e:	49 89 d5             	mov    %rdx,%r13
  401b51:	4c 29 e5             	sub    %r12,%rbp
  401b54:	48 83 ec 08          	sub    $0x8,%rsp
  401b58:	48 c1 fd 03          	sar    $0x3,%rbp
  401b5c:	e8 ef ea ff ff       	callq  400650 <_init>
  401b61:	48 85 ed             	test   %rbp,%rbp
  401b64:	74 20                	je     401b86 <__libc_csu_init+0x56>
  401b66:	31 db                	xor    %ebx,%ebx
  401b68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401b6f:	00 
  401b70:	4c 89 ea             	mov    %r13,%rdx
  401b73:	4c 89 f6             	mov    %r14,%rsi
  401b76:	44 89 ff             	mov    %r15d,%edi
  401b79:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401b7d:	48 83 c3 01          	add    $0x1,%rbx
  401b81:	48 39 eb             	cmp    %rbp,%rbx
  401b84:	75 ea                	jne    401b70 <__libc_csu_init+0x40>
  401b86:	48 83 c4 08          	add    $0x8,%rsp
  401b8a:	5b                   	pop    %rbx
  401b8b:	5d                   	pop    %rbp
  401b8c:	41 5c                	pop    %r12
  401b8e:	41 5d                	pop    %r13
  401b90:	41 5e                	pop    %r14
  401b92:	41 5f                	pop    %r15
  401b94:	c3                   	retq   
  401b95:	90                   	nop
  401b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401b9d:	00 00 00 

0000000000401ba0 <__libc_csu_fini>:
  401ba0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401ba4 <_fini>:
  401ba4:	48 83 ec 08          	sub    $0x8,%rsp
  401ba8:	48 83 c4 08          	add    $0x8,%rsp
  401bac:	c3                   	retq   
