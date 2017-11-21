
bin/clang++-3.8_nbody_ikracpp_inversed:     file format elf64-x86-64


Disassembly of section .init:

0000000000400670 <_init>:
  400670:	48 83 ec 08          	sub    $0x8,%rsp
  400674:	48 8b 05 7d 29 20 00 	mov    0x20297d(%rip),%rax        # 602ff8 <_DYNAMIC+0x200>
  40067b:	48 85 c0             	test   %rax,%rax
  40067e:	74 05                	je     400685 <_init+0x15>
  400680:	e8 9b 00 00 00       	callq  400720 <rand@plt+0x10>
  400685:	48 83 c4 08          	add    $0x8,%rsp
  400689:	c3                   	retq   

Disassembly of section .plt:

0000000000400690 <printf@plt-0x10>:
  400690:	ff 35 72 29 20 00    	pushq  0x202972(%rip)        # 603008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400696:	ff 25 74 29 20 00    	jmpq   *0x202974(%rip)        # 603010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40069c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006a0 <printf@plt>:
  4006a0:	ff 25 72 29 20 00    	jmpq   *0x202972(%rip)        # 603018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006a6:	68 00 00 00 00       	pushq  $0x0
  4006ab:	e9 e0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006b0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006b0:	ff 25 6a 29 20 00    	jmpq   *0x20296a(%rip)        # 603020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006b6:	68 01 00 00 00       	pushq  $0x1
  4006bb:	e9 d0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006c0 <__libc_start_main@plt>:
  4006c0:	ff 25 62 29 20 00    	jmpq   *0x202962(%rip)        # 603028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006c6:	68 02 00 00 00       	pushq  $0x2
  4006cb:	e9 c0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006d0 <__cxa_atexit@plt>:
  4006d0:	ff 25 5a 29 20 00    	jmpq   *0x20295a(%rip)        # 603030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006d6:	68 03 00 00 00       	pushq  $0x3
  4006db:	e9 b0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006e0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4006e0:	ff 25 52 29 20 00    	jmpq   *0x202952(%rip)        # 603038 <_GLOBAL_OFFSET_TABLE_+0x38>
  4006e6:	68 04 00 00 00       	pushq  $0x4
  4006eb:	e9 a0 ff ff ff       	jmpq   400690 <_init+0x20>

00000000004006f0 <srand@plt>:
  4006f0:	ff 25 4a 29 20 00    	jmpq   *0x20294a(%rip)        # 603040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4006f6:	68 05 00 00 00       	pushq  $0x5
  4006fb:	e9 90 ff ff ff       	jmpq   400690 <_init+0x20>

0000000000400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400700:	ff 25 42 29 20 00    	jmpq   *0x202942(%rip)        # 603048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400706:	68 06 00 00 00       	pushq  $0x6
  40070b:	e9 80 ff ff ff       	jmpq   400690 <_init+0x20>

0000000000400710 <rand@plt>:
  400710:	ff 25 3a 29 20 00    	jmpq   *0x20293a(%rip)        # 603050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400716:	68 07 00 00 00       	pushq  $0x7
  40071b:	e9 70 ff ff ff       	jmpq   400690 <_init+0x20>

Disassembly of section .plt.got:

0000000000400720 <.plt.got>:
  400720:	ff 25 d2 28 20 00    	jmpq   *0x2028d2(%rip)        # 602ff8 <_DYNAMIC+0x200>
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

0000000000400750 <_GLOBAL__sub_I_ikracpp_inversed.cc>:
  400750:	50                   	push   %rax
  400751:	bf 70 30 60 00       	mov    $0x603070,%edi
  400756:	e8 55 ff ff ff       	callq  4006b0 <_ZNSt8ios_base4InitC1Ev@plt>
  40075b:	bf e0 06 40 00       	mov    $0x4006e0,%edi
  400760:	be 70 30 60 00       	mov    $0x603070,%esi
  400765:	ba 60 30 60 00       	mov    $0x603060,%edx
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
  40077f:	49 c7 c0 60 1b 40 00 	mov    $0x401b60,%r8
  400786:	48 c7 c1 f0 1a 40 00 	mov    $0x401af0,%rcx
  40078d:	48 c7 c7 50 14 40 00 	mov    $0x401450,%rdi
  400794:	e8 27 ff ff ff       	callq  4006c0 <__libc_start_main@plt>
  400799:	f4                   	hlt    
  40079a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004007a0 <deregister_tm_clones>:
  4007a0:	b8 6f 30 60 00       	mov    $0x60306f,%eax
  4007a5:	55                   	push   %rbp
  4007a6:	48 2d 68 30 60 00    	sub    $0x603068,%rax
  4007ac:	48 83 f8 0e          	cmp    $0xe,%rax
  4007b0:	48 89 e5             	mov    %rsp,%rbp
  4007b3:	76 1b                	jbe    4007d0 <deregister_tm_clones+0x30>
  4007b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4007ba:	48 85 c0             	test   %rax,%rax
  4007bd:	74 11                	je     4007d0 <deregister_tm_clones+0x30>
  4007bf:	5d                   	pop    %rbp
  4007c0:	bf 68 30 60 00       	mov    $0x603068,%edi
  4007c5:	ff e0                	jmpq   *%rax
  4007c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ce:	00 00 
  4007d0:	5d                   	pop    %rbp
  4007d1:	c3                   	retq   
  4007d2:	0f 1f 40 00          	nopl   0x0(%rax)
  4007d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007dd:	00 00 00 

00000000004007e0 <register_tm_clones>:
  4007e0:	be 68 30 60 00       	mov    $0x603068,%esi
  4007e5:	55                   	push   %rbp
  4007e6:	48 81 ee 68 30 60 00 	sub    $0x603068,%rsi
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
  40080e:	bf 68 30 60 00       	mov    $0x603068,%edi
  400813:	ff e0                	jmpq   *%rax
  400815:	0f 1f 00             	nopl   (%rax)
  400818:	5d                   	pop    %rbp
  400819:	c3                   	retq   
  40081a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400820 <__do_global_dtors_aux>:
  400820:	80 3d 41 28 20 00 00 	cmpb   $0x0,0x202841(%rip)        # 603068 <__TMC_END__>
  400827:	75 11                	jne    40083a <__do_global_dtors_aux+0x1a>
  400829:	55                   	push   %rbp
  40082a:	48 89 e5             	mov    %rsp,%rbp
  40082d:	e8 6e ff ff ff       	callq  4007a0 <deregister_tm_clones>
  400832:	5d                   	pop    %rbp
  400833:	c6 05 2e 28 20 00 01 	movb   $0x1,0x20282e(%rip)        # 603068 <__TMC_END__>
  40083a:	f3 c3                	repz retq 
  40083c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400840 <frame_dummy>:
  400840:	bf f0 2d 60 00       	mov    $0x602df0,%edi
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
  400880:	53                   	push   %rbx
  400881:	48 83 ec 30          	sub    $0x30,%rsp
  400885:	bf 2a 00 00 00       	mov    $0x2a,%edi
  40088a:	e8 61 fe ff ff       	callq  4006f0 <srand@plt>
  40088f:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  400894:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40089b:	00 00 00 00 00 
  4008a0:	e8 6b fe ff ff       	callq  400710 <rand@plt>
  4008a5:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008a9:	c5 fb 10 05 cf 12 00 	vmovsd 0x12cf(%rip),%xmm0        # 401b80 <_IO_stdin_used+0x10>
  4008b0:	00 
  4008b1:	c4 e2 f9 99 0d be 12 	vfmadd132sd 0x12be(%rip),%xmm0,%xmm1        # 401b78 <_IO_stdin_used+0x8>
  4008b8:	00 00 
  4008ba:	c5 fb 11 4c 24 28    	vmovsd %xmm1,0x28(%rsp)
  4008c0:	e8 4b fe ff ff       	callq  400710 <rand@plt>
  4008c5:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008c9:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008cd:	c5 fb 10 05 b3 12 00 	vmovsd 0x12b3(%rip),%xmm0        # 401b88 <_IO_stdin_used+0x18>
  4008d4:	00 
  4008d5:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008d9:	c5 fb 10 05 af 12 00 	vmovsd 0x12af(%rip),%xmm0        # 401b90 <_IO_stdin_used+0x20>
  4008e0:	00 
  4008e1:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008e6:	c5 fb 11 4c 24 20    	vmovsd %xmm1,0x20(%rsp)
  4008ec:	e8 1f fe ff ff       	callq  400710 <rand@plt>
  4008f1:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008f5:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  4008f9:	c5 fb 10 0d 8f 12 00 	vmovsd 0x128f(%rip),%xmm1        # 401b90 <_IO_stdin_used+0x20>
  400900:	00 
  400901:	c4 e2 f1 a9 05 7e 12 	vfmadd213sd 0x127e(%rip),%xmm1,%xmm0        # 401b88 <_IO_stdin_used+0x18>
  400908:	00 00 
  40090a:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
  400910:	e8 fb fd ff ff       	callq  400710 <rand@plt>
  400915:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400919:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  40091d:	c5 fb 10 05 73 12 00 	vmovsd 0x1273(%rip),%xmm0        # 401b98 <_IO_stdin_used+0x28>
  400924:	00 
  400925:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400929:	c5 fb 10 05 6f 12 00 	vmovsd 0x126f(%rip),%xmm0        # 401ba0 <_IO_stdin_used+0x30>
  400930:	00 
  400931:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400936:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  40093c:	e8 cf fd ff ff       	callq  400710 <rand@plt>
  400941:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400945:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400949:	c5 fb 10 0d 4f 12 00 	vmovsd 0x124f(%rip),%xmm1        # 401ba0 <_IO_stdin_used+0x30>
  400950:	00 
  400951:	c4 e2 f1 a9 05 3e 12 	vfmadd213sd 0x123e(%rip),%xmm1,%xmm0        # 401b98 <_IO_stdin_used+0x28>
  400958:	00 00 
  40095a:	48 8b 05 17 27 20 00 	mov    0x202717(%rip),%rax        # 603078 <__Bodydata_buffer>
  400961:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400965:	48 89 0d 0c 27 20 00 	mov    %rcx,0x20270c(%rip)        # 603078 <__Bodydata_buffer>
  40096c:	c5 fb 10 4c 24 28    	vmovsd 0x28(%rsp),%xmm1
  400972:	c5 fb 11 0c c5 90 30 	vmovsd %xmm1,0x603090(,%rax,8)
  400979:	60 00 
  40097b:	c5 fb 10 4c 24 20    	vmovsd 0x20(%rsp),%xmm1
  400981:	c5 fb 11 0c c5 90 2a 	vmovsd %xmm1,0x612a90(,%rax,8)
  400988:	61 00 
  40098a:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  400990:	c5 fb 11 0c c5 90 24 	vmovsd %xmm1,0x622490(,%rax,8)
  400997:	62 00 
  400999:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  40099f:	c5 fb 11 0c c5 90 1e 	vmovsd %xmm1,0x631e90(,%rax,8)
  4009a6:	63 00 
  4009a8:	c5 fb 11 04 c5 90 18 	vmovsd %xmm0,0x641890(,%rax,8)
  4009af:	64 00 
  4009b1:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  4009b8:	00 00 00 00 00 
  4009bd:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  4009c4:	00 00 00 00 00 
  4009c9:	83 c3 ff             	add    $0xffffffff,%ebx
  4009cc:	0f 85 ce fe ff ff    	jne    4008a0 <_Z13instantiationv+0x20>
  4009d2:	48 83 c4 30          	add    $0x30,%rsp
  4009d6:	5b                   	pop    %rbx
  4009d7:	c3                   	retq   
  4009d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009df:	00 

00000000004009e0 <_Z14run_simulationv>:
  4009e0:	41 57                	push   %r15
  4009e2:	41 56                	push   %r14
  4009e4:	53                   	push   %rbx
  4009e5:	b8 10 00 00 00       	mov    $0x10,%eax
  4009ea:	45 31 ff             	xor    %r15d,%r15d
  4009ed:	0f 1f 00             	nopl   (%rax)
  4009f0:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  4009f7:	00 00 00 00 00 
  4009fc:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  400a03:	00 00 00 00 00 
  400a08:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  400a0f:	00 00 00 00 00 
  400a14:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  400a1b:	00 00 00 00 00 
  400a20:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  400a27:	00 00 00 00 00 
  400a2c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  400a33:	00 00 00 00 00 
  400a38:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  400a3f:	00 00 00 00 00 
  400a44:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  400a4b:	00 00 00 00 00 
  400a50:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  400a57:	00 00 00 00 00 
  400a5c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  400a63:	00 00 00 00 00 
  400a68:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400a6f:	00 00 00 00 00 
  400a74:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  400a7b:	00 00 00 00 00 
  400a80:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400a87:	00 00 00 00 00 
  400a8c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400a93:	00 00 00 00 00 
  400a98:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400a9f:	00 00 00 00 00 
  400aa4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400aab:	00 00 00 00 00 
  400ab0:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400ab7:	00 00 00 00 00 
  400abc:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400ac3:	00 00 00 00 00 
  400ac8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400acf:	00 00 00 00 00 
  400ad4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400adb:	00 00 00 00 00 
  400ae0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400ae7:	00 00 00 00 00 
  400aec:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400af3:	00 00 00 00 00 
  400af8:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400aff:	00 00 00 00 00 
  400b04:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400b0b:	00 00 00 00 00 
  400b10:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400b17:	00 00 00 00 00 
  400b1c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400b23:	00 00 00 00 00 
  400b28:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400b2f:	00 00 00 00 00 
  400b34:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400b3b:	00 00 00 00 00 
  400b40:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400b47:	00 00 00 00 00 
  400b4c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400b53:	00 00 00 00 00 
  400b58:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400b5f:	00 00 00 00 00 
  400b64:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  400b6b:	00 00 00 00 00 
  400b70:	48 83 c0 10          	add    $0x10,%rax
  400b74:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  400b7a:	0f 85 70 fe ff ff    	jne    4009f0 <_Z14run_simulationv+0x10>
  400b80:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 
  400b90:	49 83 c7 01          	add    $0x1,%r15
  400b94:	bb 01 00 00 00       	mov    $0x1,%ebx
  400b99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400ba0:	4c 89 ff             	mov    %r15,%rdi
  400ba3:	48 89 de             	mov    %rbx,%rsi
  400ba6:	e8 a5 0e 00 00       	callq  401a50 <_ZN4Body9add_forceEPS_>
  400bab:	48 83 c3 01          	add    $0x1,%rbx
  400baf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bb6:	75 e8                	jne    400ba0 <_Z14run_simulationv+0x1c0>
  400bb8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400bbf:	75 cf                	jne    400b90 <_Z14run_simulationv+0x1b0>
  400bc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bcd:	00 00 00 
  400bd0:	4c 89 f7             	mov    %r14,%rdi
  400bd3:	c5 fb 10 05 cd 0f 00 	vmovsd 0xfcd(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  400bda:	00 
  400bdb:	e8 00 0d 00 00       	callq  4018e0 <_ZN4Body6updateEd>
  400be0:	49 83 c6 01          	add    $0x1,%r14
  400be4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400beb:	75 e3                	jne    400bd0 <_Z14run_simulationv+0x1f0>
  400bed:	45 31 ff             	xor    %r15d,%r15d
  400bf0:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  400bf7:	00 00 00 00 00 
  400bfc:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  400c03:	00 00 00 00 00 
  400c08:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  400c0f:	00 00 00 00 00 
  400c14:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  400c1b:	00 00 00 00 00 
  400c20:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  400c27:	00 00 00 00 00 
  400c2c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  400c33:	00 00 00 00 00 
  400c38:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  400c3f:	00 00 00 00 00 
  400c44:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  400c4b:	00 00 00 00 00 
  400c50:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  400c57:	00 00 00 00 00 
  400c5c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  400c63:	00 00 00 00 00 
  400c68:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  400c6f:	00 00 00 00 00 
  400c74:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  400c7b:	00 00 00 00 00 
  400c80:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  400c87:	00 00 00 00 00 
  400c8c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  400c93:	00 00 00 00 00 
  400c98:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  400c9f:	00 00 00 00 00 
  400ca4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  400cab:	00 00 00 00 00 
  400cb0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  400cb7:	00 00 00 00 00 
  400cbc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  400cc3:	00 00 00 00 00 
  400cc8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  400ccf:	00 00 00 00 00 
  400cd4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  400cdb:	00 00 00 00 00 
  400ce0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  400ce7:	00 00 00 00 00 
  400cec:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  400cf3:	00 00 00 00 00 
  400cf8:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  400cff:	00 00 00 00 00 
  400d04:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  400d0b:	00 00 00 00 00 
  400d10:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  400d17:	00 00 00 00 00 
  400d1c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  400d23:	00 00 00 00 00 
  400d28:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  400d2f:	00 00 00 00 00 
  400d34:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  400d3b:	00 00 00 00 00 
  400d40:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  400d47:	00 00 00 00 00 
  400d4c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  400d53:	00 00 00 00 00 
  400d58:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  400d5f:	00 00 00 00 00 
  400d64:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  400d6b:	00 00 00 00 00 
  400d70:	48 83 c3 10          	add    $0x10,%rbx
  400d74:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  400d7b:	0f 85 6f fe ff ff    	jne    400bf0 <_Z14run_simulationv+0x210>
  400d81:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400d87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d8e:	00 00 
  400d90:	49 83 c7 01          	add    $0x1,%r15
  400d94:	bb 01 00 00 00       	mov    $0x1,%ebx
  400d99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400da0:	4c 89 ff             	mov    %r15,%rdi
  400da3:	48 89 de             	mov    %rbx,%rsi
  400da6:	e8 a5 0c 00 00       	callq  401a50 <_ZN4Body9add_forceEPS_>
  400dab:	48 83 c3 01          	add    $0x1,%rbx
  400daf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400db6:	75 e8                	jne    400da0 <_Z14run_simulationv+0x3c0>
  400db8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dbf:	75 cf                	jne    400d90 <_Z14run_simulationv+0x3b0>
  400dc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400dc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcd:	00 00 00 
  400dd0:	4c 89 f7             	mov    %r14,%rdi
  400dd3:	c5 fb 10 05 cd 0d 00 	vmovsd 0xdcd(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  400dda:	00 
  400ddb:	e8 00 0b 00 00       	callq  4018e0 <_ZN4Body6updateEd>
  400de0:	49 83 c6 01          	add    $0x1,%r14
  400de4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400deb:	75 e3                	jne    400dd0 <_Z14run_simulationv+0x3f0>
  400ded:	45 31 ff             	xor    %r15d,%r15d
  400df0:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  400df7:	00 00 00 00 00 
  400dfc:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  400e03:	00 00 00 00 00 
  400e08:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  400e0f:	00 00 00 00 00 
  400e14:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  400e1b:	00 00 00 00 00 
  400e20:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  400e27:	00 00 00 00 00 
  400e2c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  400e33:	00 00 00 00 00 
  400e38:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  400e3f:	00 00 00 00 00 
  400e44:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  400e4b:	00 00 00 00 00 
  400e50:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  400e57:	00 00 00 00 00 
  400e5c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  400e63:	00 00 00 00 00 
  400e68:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  400e6f:	00 00 00 00 00 
  400e74:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  400e7b:	00 00 00 00 00 
  400e80:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  400e87:	00 00 00 00 00 
  400e8c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  400e93:	00 00 00 00 00 
  400e98:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  400e9f:	00 00 00 00 00 
  400ea4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  400eab:	00 00 00 00 00 
  400eb0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  400eb7:	00 00 00 00 00 
  400ebc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  400ec3:	00 00 00 00 00 
  400ec8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  400ecf:	00 00 00 00 00 
  400ed4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  400edb:	00 00 00 00 00 
  400ee0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  400ee7:	00 00 00 00 00 
  400eec:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  400ef3:	00 00 00 00 00 
  400ef8:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  400eff:	00 00 00 00 00 
  400f04:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  400f0b:	00 00 00 00 00 
  400f10:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  400f17:	00 00 00 00 00 
  400f1c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  400f23:	00 00 00 00 00 
  400f28:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  400f2f:	00 00 00 00 00 
  400f34:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  400f3b:	00 00 00 00 00 
  400f40:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  400f47:	00 00 00 00 00 
  400f4c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  400f53:	00 00 00 00 00 
  400f58:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  400f5f:	00 00 00 00 00 
  400f64:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  400f6b:	00 00 00 00 00 
  400f70:	48 83 c3 10          	add    $0x10,%rbx
  400f74:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  400f7b:	0f 85 6f fe ff ff    	jne    400df0 <_Z14run_simulationv+0x410>
  400f81:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400f87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f8e:	00 00 
  400f90:	49 83 c7 01          	add    $0x1,%r15
  400f94:	bb 01 00 00 00       	mov    $0x1,%ebx
  400f99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400fa0:	4c 89 ff             	mov    %r15,%rdi
  400fa3:	48 89 de             	mov    %rbx,%rsi
  400fa6:	e8 a5 0a 00 00       	callq  401a50 <_ZN4Body9add_forceEPS_>
  400fab:	48 83 c3 01          	add    $0x1,%rbx
  400faf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fb6:	75 e8                	jne    400fa0 <_Z14run_simulationv+0x5c0>
  400fb8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400fbf:	75 cf                	jne    400f90 <_Z14run_simulationv+0x5b0>
  400fc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400fc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fcd:	00 00 00 
  400fd0:	4c 89 f7             	mov    %r14,%rdi
  400fd3:	c5 fb 10 05 cd 0b 00 	vmovsd 0xbcd(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  400fda:	00 
  400fdb:	e8 00 09 00 00       	callq  4018e0 <_ZN4Body6updateEd>
  400fe0:	49 83 c6 01          	add    $0x1,%r14
  400fe4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400feb:	75 e3                	jne    400fd0 <_Z14run_simulationv+0x5f0>
  400fed:	45 31 ff             	xor    %r15d,%r15d
  400ff0:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  400ff7:	00 00 00 00 00 
  400ffc:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  401003:	00 00 00 00 00 
  401008:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  40100f:	00 00 00 00 00 
  401014:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  40101b:	00 00 00 00 00 
  401020:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  401027:	00 00 00 00 00 
  40102c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  401033:	00 00 00 00 00 
  401038:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  40103f:	00 00 00 00 00 
  401044:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  40104b:	00 00 00 00 00 
  401050:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  401057:	00 00 00 00 00 
  40105c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  401063:	00 00 00 00 00 
  401068:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  40106f:	00 00 00 00 00 
  401074:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  40107b:	00 00 00 00 00 
  401080:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  401087:	00 00 00 00 00 
  40108c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  401093:	00 00 00 00 00 
  401098:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  40109f:	00 00 00 00 00 
  4010a4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  4010ab:	00 00 00 00 00 
  4010b0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  4010b7:	00 00 00 00 00 
  4010bc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  4010c3:	00 00 00 00 00 
  4010c8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  4010cf:	00 00 00 00 00 
  4010d4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  4010db:	00 00 00 00 00 
  4010e0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  4010e7:	00 00 00 00 00 
  4010ec:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  4010f3:	00 00 00 00 00 
  4010f8:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  4010ff:	00 00 00 00 00 
  401104:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  40110b:	00 00 00 00 00 
  401110:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  401117:	00 00 00 00 00 
  40111c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  401123:	00 00 00 00 00 
  401128:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  40112f:	00 00 00 00 00 
  401134:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  40113b:	00 00 00 00 00 
  401140:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  401147:	00 00 00 00 00 
  40114c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  401153:	00 00 00 00 00 
  401158:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  40115f:	00 00 00 00 00 
  401164:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  40116b:	00 00 00 00 00 
  401170:	48 83 c3 10          	add    $0x10,%rbx
  401174:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  40117b:	0f 85 6f fe ff ff    	jne    400ff0 <_Z14run_simulationv+0x610>
  401181:	41 be 01 00 00 00    	mov    $0x1,%r14d
  401187:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40118e:	00 00 
  401190:	49 83 c7 01          	add    $0x1,%r15
  401194:	bb 01 00 00 00       	mov    $0x1,%ebx
  401199:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011a0:	4c 89 ff             	mov    %r15,%rdi
  4011a3:	48 89 de             	mov    %rbx,%rsi
  4011a6:	e8 a5 08 00 00       	callq  401a50 <_ZN4Body9add_forceEPS_>
  4011ab:	48 83 c3 01          	add    $0x1,%rbx
  4011af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011b6:	75 e8                	jne    4011a0 <_Z14run_simulationv+0x7c0>
  4011b8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4011bf:	75 cf                	jne    401190 <_Z14run_simulationv+0x7b0>
  4011c1:	bb 10 00 00 00       	mov    $0x10,%ebx
  4011c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011cd:	00 00 00 
  4011d0:	4c 89 f7             	mov    %r14,%rdi
  4011d3:	c5 fb 10 05 cd 09 00 	vmovsd 0x9cd(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  4011da:	00 
  4011db:	e8 00 07 00 00       	callq  4018e0 <_ZN4Body6updateEd>
  4011e0:	49 83 c6 01          	add    $0x1,%r14
  4011e4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  4011eb:	75 e3                	jne    4011d0 <_Z14run_simulationv+0x7f0>
  4011ed:	45 31 ff             	xor    %r15d,%r15d
  4011f0:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  4011f7:	00 00 00 00 00 
  4011fc:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  401203:	00 00 00 00 00 
  401208:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  40120f:	00 00 00 00 00 
  401214:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  40121b:	00 00 00 00 00 
  401220:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  401227:	00 00 00 00 00 
  40122c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  401233:	00 00 00 00 00 
  401238:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  40123f:	00 00 00 00 00 
  401244:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  40124b:	00 00 00 00 00 
  401250:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  401257:	00 00 00 00 00 
  40125c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  401263:	00 00 00 00 00 
  401268:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  40126f:	00 00 00 00 00 
  401274:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  40127b:	00 00 00 00 00 
  401280:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  401287:	00 00 00 00 00 
  40128c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  401293:	00 00 00 00 00 
  401298:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  40129f:	00 00 00 00 00 
  4012a4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  4012ab:	00 00 00 00 00 
  4012b0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  4012b7:	00 00 00 00 00 
  4012bc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  4012c3:	00 00 00 00 00 
  4012c8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  4012cf:	00 00 00 00 00 
  4012d4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  4012db:	00 00 00 00 00 
  4012e0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  4012e7:	00 00 00 00 00 
  4012ec:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  4012f3:	00 00 00 00 00 
  4012f8:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  4012ff:	00 00 00 00 00 
  401304:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  40130b:	00 00 00 00 00 
  401310:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  401317:	00 00 00 00 00 
  40131c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  401323:	00 00 00 00 00 
  401328:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  40132f:	00 00 00 00 00 
  401334:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  40133b:	00 00 00 00 00 
  401340:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  401347:	00 00 00 00 00 
  40134c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  401353:	00 00 00 00 00 
  401358:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  40135f:	00 00 00 00 00 
  401364:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  40136b:	00 00 00 00 00 
  401370:	48 83 c3 10          	add    $0x10,%rbx
  401374:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  40137b:	0f 85 6f fe ff ff    	jne    4011f0 <_Z14run_simulationv+0x810>
  401381:	41 be 01 00 00 00    	mov    $0x1,%r14d
  401387:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40138e:	00 00 
  401390:	49 83 c7 01          	add    $0x1,%r15
  401394:	bb 01 00 00 00       	mov    $0x1,%ebx
  401399:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4013a0:	4c 89 ff             	mov    %r15,%rdi
  4013a3:	48 89 de             	mov    %rbx,%rsi
  4013a6:	e8 a5 06 00 00       	callq  401a50 <_ZN4Body9add_forceEPS_>
  4013ab:	48 83 c3 01          	add    $0x1,%rbx
  4013af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013b6:	75 e8                	jne    4013a0 <_Z14run_simulationv+0x9c0>
  4013b8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4013bf:	75 cf                	jne    401390 <_Z14run_simulationv+0x9b0>
  4013c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4013c8:	0f 1f 84 00 00 00 00 
  4013cf:	00 
  4013d0:	4c 89 f7             	mov    %r14,%rdi
  4013d3:	c5 fb 10 05 cd 07 00 	vmovsd 0x7cd(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  4013da:	00 
  4013db:	e8 00 05 00 00       	callq  4018e0 <_ZN4Body6updateEd>
  4013e0:	49 83 c6 01          	add    $0x1,%r14
  4013e4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  4013eb:	75 e3                	jne    4013d0 <_Z14run_simulationv+0x9f0>
  4013ed:	5b                   	pop    %rbx
  4013ee:	41 5e                	pop    %r14
  4013f0:	41 5f                	pop    %r15
  4013f2:	c3                   	retq   
  4013f3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4013fa:	84 00 00 00 00 00 

0000000000401400 <_Z10run_simplev>:
  401400:	55                   	push   %rbp
  401401:	53                   	push   %rbx
  401402:	50                   	push   %rax
  401403:	31 ed                	xor    %ebp,%ebp
  401405:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40140c:	00 00 00 00 
  401410:	bb 01 00 00 00       	mov    $0x1,%ebx
  401415:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40141c:	00 00 00 00 
  401420:	48 89 df             	mov    %rbx,%rdi
  401423:	c5 fb 10 05 7d 07 00 	vmovsd 0x77d(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  40142a:	00 
  40142b:	e8 90 05 00 00       	callq  4019c0 <_ZN4Body22codengen_simple_updateEd>
  401430:	48 83 c3 01          	add    $0x1,%rbx
  401434:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  40143b:	75 e3                	jne    401420 <_Z10run_simplev+0x20>
  40143d:	83 c5 01             	add    $0x1,%ebp
  401440:	81 fd f4 01 00 00    	cmp    $0x1f4,%ebp
  401446:	75 c8                	jne    401410 <_Z10run_simplev+0x10>
  401448:	48 83 c4 08          	add    $0x8,%rsp
  40144c:	5b                   	pop    %rbx
  40144d:	5d                   	pop    %rbp
  40144e:	c3                   	retq   
  40144f:	90                   	nop

0000000000401450 <main>:
  401450:	55                   	push   %rbp
  401451:	41 57                	push   %r15
  401453:	41 56                	push   %r14
  401455:	41 55                	push   %r13
  401457:	41 54                	push   %r12
  401459:	53                   	push   %rbx
  40145a:	50                   	push   %rax
  40145b:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  40145f:	c5 f8 11 05 11 1c 20 	vmovups %xmm0,0x201c11(%rip)        # 603078 <__Bodydata_buffer>
  401466:	00 
  401467:	e8 94 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40146c:	48 89 c5             	mov    %rax,%rbp
  40146f:	e8 0c f4 ff ff       	callq  400880 <_Z13instantiationv>
  401474:	e8 87 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401479:	49 89 c6             	mov    %rax,%r14
  40147c:	49 29 ee             	sub    %rbp,%r14
  40147f:	e8 7c f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401484:	48 89 c3             	mov    %rax,%rbx
  401487:	e8 54 f5 ff ff       	callq  4009e0 <_Z14run_simulationv>
  40148c:	e8 6f f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401491:	48 89 c5             	mov    %rax,%rbp
  401494:	48 29 dd             	sub    %rbx,%rbp
  401497:	31 db                	xor    %ebx,%ebx
  401499:	e8 62 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40149e:	49 89 c7             	mov    %rax,%r15
  4014a1:	c5 fb 10 05 ff 06 00 	vmovsd 0x6ff(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x38>
  4014a8:	00 
  4014a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014b0:	31 c0                	xor    %eax,%eax
  4014b2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4014b9:	1f 84 00 00 00 00 00 
  4014c0:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4014c4:	ba 64 00 00 00       	mov    $0x64,%edx
  4014c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4014d0:	c5 fb 59 0c cd 88 12 	vmulsd 0x651288(,%rcx,8),%xmm0,%xmm1
  4014d7:	65 00 
  4014d9:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  4014e0:	60 00 
  4014e2:	c5 f3 58 0c cd 88 1e 	vaddsd 0x631e88(,%rcx,8),%xmm1,%xmm1
  4014e9:	63 00 
  4014eb:	c5 fb 11 0c cd 88 1e 	vmovsd %xmm1,0x631e88(,%rcx,8)
  4014f2:	63 00 
  4014f4:	c5 fb 59 14 cd 88 0c 	vmulsd 0x660c88(,%rcx,8),%xmm0,%xmm2
  4014fb:	66 00 
  4014fd:	c5 eb 5e 14 c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm2,%xmm2
  401504:	60 00 
  401506:	c5 eb 58 14 cd 88 18 	vaddsd 0x641888(,%rcx,8),%xmm2,%xmm2
  40150d:	64 00 
  40150f:	c5 fb 11 14 cd 88 18 	vmovsd %xmm2,0x641888(,%rcx,8)
  401516:	64 00 
  401518:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x612a88(,%rcx,8),%xmm0,%xmm1
  40151f:	2a 61 00 
  401522:	c5 fb 11 0c cd 88 2a 	vmovsd %xmm1,0x612a88(,%rcx,8)
  401529:	61 00 
  40152b:	c5 fb 10 0c cd 88 18 	vmovsd 0x641888(,%rcx,8),%xmm1
  401532:	64 00 
  401534:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x622488(,%rcx,8),%xmm0,%xmm1
  40153b:	24 62 00 
  40153e:	83 c2 ff             	add    $0xffffffff,%edx
  401541:	c5 fb 11 0c cd 88 24 	vmovsd %xmm1,0x622488(,%rcx,8)
  401548:	62 00 
  40154a:	75 84                	jne    4014d0 <main+0x80>
  40154c:	48 89 c8             	mov    %rcx,%rax
  40154f:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401556:	0f 85 64 ff ff ff    	jne    4014c0 <main+0x70>
  40155c:	83 c3 01             	add    $0x1,%ebx
  40155f:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  401565:	0f 85 45 ff ff ff    	jne    4014b0 <main+0x60>
  40156b:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  401572:	de 1b 43 
  401575:	4c 89 f0             	mov    %r14,%rax
  401578:	48 f7 ee             	imul   %rsi
  40157b:	49 89 d6             	mov    %rdx,%r14
  40157e:	4c 89 f1             	mov    %r14,%rcx
  401581:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401585:	49 c1 fe 12          	sar    $0x12,%r14
  401589:	48 89 e8             	mov    %rbp,%rax
  40158c:	48 f7 ee             	imul   %rsi
  40158f:	49 89 d5             	mov    %rdx,%r13
  401592:	49 01 ce             	add    %rcx,%r14
  401595:	4c 89 e8             	mov    %r13,%rax
  401598:	48 c1 e8 3f          	shr    $0x3f,%rax
  40159c:	49 c1 fd 12          	sar    $0x12,%r13
  4015a0:	49 01 c5             	add    %rax,%r13
  4015a3:	e8 58 f1 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4015a8:	49 89 c4             	mov    %rax,%r12
  4015ab:	4d 29 fc             	sub    %r15,%r12
  4015ae:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4015b3:	31 f6                	xor    %esi,%esi
  4015b5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4015bc:	00 00 00 00 
  4015c0:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  4015c7:	61 00 
  4015c9:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  4015cd:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  4015d1:	01 e8                	add    %ebp,%eax
  4015d3:	c5 fb 10 0c f5 90 24 	vmovsd 0x622490(,%rsi,8),%xmm1
  4015da:	62 00 
  4015dc:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  4015e0:	c5 f9 7e d5          	vmovd  %xmm2,%ebp
  4015e4:	01 c5                	add    %eax,%ebp
  4015e6:	48 63 c5             	movslq %ebp,%rax
  4015e9:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  4015f0:	48 8d 5e 01          	lea    0x1(%rsi),%rbx
  4015f4:	48 89 c1             	mov    %rax,%rcx
  4015f7:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4015fb:	48 c1 f8 33          	sar    $0x33,%rax
  4015ff:	01 c8                	add    %ecx,%eax
  401601:	44 69 f8 87 d6 12 00 	imul   $0x12d687,%eax,%r15d
  401608:	48 83 fe 0a          	cmp    $0xa,%rsi
  40160c:	73 0c                	jae    40161a <main+0x1ca>
  40160e:	bf e0 1b 40 00       	mov    $0x401be0,%edi
  401613:	b0 02                	mov    $0x2,%al
  401615:	e8 86 f0 ff ff       	callq  4006a0 <printf@plt>
  40161a:	44 29 fd             	sub    %r15d,%ebp
  40161d:	48 89 de             	mov    %rbx,%rsi
  401620:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401627:	75 97                	jne    4015c0 <main+0x170>
  401629:	4c 89 e0             	mov    %r12,%rax
  40162c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401633:	de 1b 43 
  401636:	48 f7 e9             	imul   %rcx
  401639:	48 89 d0             	mov    %rdx,%rax
  40163c:	48 c1 e8 3f          	shr    $0x3f,%rax
  401640:	48 c1 fa 12          	sar    $0x12,%rdx
  401644:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401648:	bf f5 1b 40 00       	mov    $0x401bf5,%edi
  40164d:	31 c0                	xor    %eax,%eax
  40164f:	4c 89 f6             	mov    %r14,%rsi
  401652:	4c 89 ea             	mov    %r13,%rdx
  401655:	41 89 e8             	mov    %ebp,%r8d
  401658:	e8 43 f0 ff ff       	callq  4006a0 <printf@plt>
  40165d:	31 c0                	xor    %eax,%eax
  40165f:	48 83 c4 08          	add    $0x8,%rsp
  401663:	5b                   	pop    %rbx
  401664:	41 5c                	pop    %r12
  401666:	41 5d                	pop    %r13
  401668:	41 5e                	pop    %r14
  40166a:	41 5f                	pop    %r15
  40166c:	5d                   	pop    %rbp
  40166d:	c3                   	retq   
  40166e:	66 90                	xchg   %ax,%ax

0000000000401670 <_Z19codegen_reset_forceP4Body>:
  401670:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401677:	00 00 00 00 00 
  40167c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401683:	00 00 00 00 00 
  401688:	c3                   	retq   
  401689:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401690 <_Z17codegen_add_forceP4BodyS0_>:
  401690:	48 39 f7             	cmp    %rsi,%rdi
  401693:	0f 84 91 00 00 00    	je     40172a <_Z17codegen_add_forceP4BodyS0_+0x9a>
  401699:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  4016a0:	61 00 
  4016a2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  4016a9:	61 00 
  4016ab:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  4016b2:	62 00 
  4016b4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  4016bb:	62 00 
  4016bd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  4016c1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  4016c6:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  4016ca:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  4016d1:	60 00 
  4016d3:	c5 db 59 25 dd 04 00 	vmulsd 0x4dd(%rip),%xmm4,%xmm4        # 401bb8 <_IO_stdin_used+0x48>
  4016da:	00 
  4016db:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  4016e2:	60 00 
  4016e4:	c5 eb 58 15 d4 04 00 	vaddsd 0x4d4(%rip),%xmm2,%xmm2        # 401bc0 <_IO_stdin_used+0x50>
  4016eb:	00 
  4016ec:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  4016f0:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  4016f4:	c5 fb 10 25 b4 04 00 	vmovsd 0x4b4(%rip),%xmm4        # 401bb0 <_IO_stdin_used+0x40>
  4016fb:	00 
  4016fc:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401700:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401707:	12 65 00 
  40170a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401711:	65 00 
  401713:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401717:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  40171e:	0c 66 00 
  401721:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401728:	66 00 
  40172a:	c3                   	retq   
  40172b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401730 <_Z14codegen_updateP4Bodyd>:
  401730:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401737:	65 00 
  401739:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401740:	60 00 
  401742:	c5 f3 58 14 fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm2
  401749:	63 00 
  40174b:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  401752:	63 00 
  401754:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40175b:	66 00 
  40175d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401764:	60 00 
  401766:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40176d:	64 00 
  40176f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401776:	64 00 
  401778:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  40177f:	2a 61 00 
  401782:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  401789:	61 00 
  40178b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401792:	64 00 
  401794:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  40179b:	24 62 00 
  40179e:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4017a5:	62 00 
  4017a7:	c5 f9 2e 15 d9 03 00 	vucomisd 0x3d9(%rip),%xmm2        # 401b88 <_IO_stdin_used+0x18>
  4017ae:	00 
  4017af:	72 0a                	jb     4017bb <_Z14codegen_updateP4Bodyd+0x8b>
  4017b1:	c5 f9 2e 15 f7 03 00 	vucomisd 0x3f7(%rip),%xmm2        # 401bb0 <_IO_stdin_used+0x40>
  4017b8:	00 
  4017b9:	76 23                	jbe    4017de <_Z14codegen_updateP4Bodyd+0xae>
  4017bb:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4017c2:	63 00 
  4017c4:	c5 f9 57 05 04 04 00 	vxorpd 0x404(%rip),%xmm0,%xmm0        # 401bd0 <_IO_stdin_used+0x60>
  4017cb:	00 
  4017cc:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4017d3:	63 00 
  4017d5:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4017dc:	62 00 
  4017de:	c5 f9 2e 0d a2 03 00 	vucomisd 0x3a2(%rip),%xmm1        # 401b88 <_IO_stdin_used+0x18>
  4017e5:	00 
  4017e6:	72 0a                	jb     4017f2 <_Z14codegen_updateP4Bodyd+0xc2>
  4017e8:	c5 f9 2e 0d c0 03 00 	vucomisd 0x3c0(%rip),%xmm1        # 401bb0 <_IO_stdin_used+0x40>
  4017ef:	00 
  4017f0:	76 1a                	jbe    40180c <_Z14codegen_updateP4Bodyd+0xdc>
  4017f2:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  4017f9:	64 00 
  4017fb:	c5 f9 57 05 cd 03 00 	vxorpd 0x3cd(%rip),%xmm0,%xmm0        # 401bd0 <_IO_stdin_used+0x60>
  401802:	00 
  401803:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  40180a:	64 00 
  40180c:	c3                   	retq   
  40180d:	0f 1f 00             	nopl   (%rax)

0000000000401810 <_Z26codegen_reset_force_manualP4Body>:
  401810:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401817:	00 00 00 00 00 
  40181c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401823:	00 00 00 00 00 
  401828:	c3                   	retq   
  401829:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401830 <_Z24codegen_reset_force_halfP4Body>:
  401830:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401837:	00 00 00 00 00 
  40183c:	c3                   	retq   
  40183d:	0f 1f 00             	nopl   (%rax)

0000000000401840 <_Z18codegen_reset_massP4Body>:
  401840:	48 c7 04 fd 88 30 60 	movq   $0x0,0x603088(,%rdi,8)
  401847:	00 00 00 00 00 
  40184c:	c3                   	retq   
  40184d:	0f 1f 00             	nopl   (%rax)

0000000000401850 <_Z22codengen_simple_updateP4Bodyd>:
  401850:	b8 64 00 00 00       	mov    $0x64,%eax
  401855:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40185c:	00 00 00 00 
  401860:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401867:	65 00 
  401869:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401870:	60 00 
  401872:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401879:	63 00 
  40187b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401882:	63 00 
  401884:	c5 fb 59 14 fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm2
  40188b:	66 00 
  40188d:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  401894:	60 00 
  401896:	c5 eb 58 14 fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm2,%xmm2
  40189d:	64 00 
  40189f:	c5 fb 11 14 fd 88 18 	vmovsd %xmm2,0x641888(,%rdi,8)
  4018a6:	64 00 
  4018a8:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4018af:	2a 61 00 
  4018b2:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4018b9:	61 00 
  4018bb:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4018c2:	64 00 
  4018c4:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4018cb:	24 62 00 
  4018ce:	83 c0 ff             	add    $0xffffffff,%eax
  4018d1:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4018d8:	62 00 
  4018da:	75 84                	jne    401860 <_Z22codengen_simple_updateP4Bodyd+0x10>
  4018dc:	c3                   	retq   
  4018dd:	0f 1f 00             	nopl   (%rax)

00000000004018e0 <_ZN4Body6updateEd>:
  4018e0:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  4018e7:	65 00 
  4018e9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018f0:	60 00 
  4018f2:	c5 f3 58 14 fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm2
  4018f9:	63 00 
  4018fb:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  401902:	63 00 
  401904:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40190b:	66 00 
  40190d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401914:	60 00 
  401916:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40191d:	64 00 
  40191f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401926:	64 00 
  401928:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  40192f:	2a 61 00 
  401932:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  401939:	61 00 
  40193b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401942:	64 00 
  401944:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  40194b:	24 62 00 
  40194e:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401955:	62 00 
  401957:	c5 f9 2e 15 29 02 00 	vucomisd 0x229(%rip),%xmm2        # 401b88 <_IO_stdin_used+0x18>
  40195e:	00 
  40195f:	72 0a                	jb     40196b <_ZN4Body6updateEd+0x8b>
  401961:	c5 f9 2e 15 47 02 00 	vucomisd 0x247(%rip),%xmm2        # 401bb0 <_IO_stdin_used+0x40>
  401968:	00 
  401969:	76 23                	jbe    40198e <_ZN4Body6updateEd+0xae>
  40196b:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  401972:	63 00 
  401974:	c5 f9 57 05 54 02 00 	vxorpd 0x254(%rip),%xmm0,%xmm0        # 401bd0 <_IO_stdin_used+0x60>
  40197b:	00 
  40197c:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  401983:	63 00 
  401985:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  40198c:	62 00 
  40198e:	c5 f9 2e 0d f2 01 00 	vucomisd 0x1f2(%rip),%xmm1        # 401b88 <_IO_stdin_used+0x18>
  401995:	00 
  401996:	72 0a                	jb     4019a2 <_ZN4Body6updateEd+0xc2>
  401998:	c5 f9 2e 0d 10 02 00 	vucomisd 0x210(%rip),%xmm1        # 401bb0 <_IO_stdin_used+0x40>
  40199f:	00 
  4019a0:	76 1a                	jbe    4019bc <_ZN4Body6updateEd+0xdc>
  4019a2:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  4019a9:	64 00 
  4019ab:	c5 f9 57 05 1d 02 00 	vxorpd 0x21d(%rip),%xmm0,%xmm0        # 401bd0 <_IO_stdin_used+0x60>
  4019b2:	00 
  4019b3:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  4019ba:	64 00 
  4019bc:	c3                   	retq   
  4019bd:	0f 1f 00             	nopl   (%rax)

00000000004019c0 <_ZN4Body22codengen_simple_updateEd>:
  4019c0:	b8 64 00 00 00       	mov    $0x64,%eax
  4019c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4019cc:	00 00 00 00 
  4019d0:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  4019d7:	65 00 
  4019d9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4019e0:	60 00 
  4019e2:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  4019e9:	63 00 
  4019eb:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  4019f2:	63 00 
  4019f4:	c5 fb 59 14 fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm2
  4019fb:	66 00 
  4019fd:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  401a04:	60 00 
  401a06:	c5 eb 58 14 fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm2,%xmm2
  401a0d:	64 00 
  401a0f:	c5 fb 11 14 fd 88 18 	vmovsd %xmm2,0x641888(,%rdi,8)
  401a16:	64 00 
  401a18:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401a1f:	2a 61 00 
  401a22:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401a29:	61 00 
  401a2b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401a32:	64 00 
  401a34:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401a3b:	24 62 00 
  401a3e:	83 c0 ff             	add    $0xffffffff,%eax
  401a41:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401a48:	62 00 
  401a4a:	75 84                	jne    4019d0 <_ZN4Body22codengen_simple_updateEd+0x10>
  401a4c:	c3                   	retq   
  401a4d:	0f 1f 00             	nopl   (%rax)

0000000000401a50 <_ZN4Body9add_forceEPS_>:
  401a50:	48 39 f7             	cmp    %rsi,%rdi
  401a53:	0f 84 91 00 00 00    	je     401aea <_ZN4Body9add_forceEPS_+0x9a>
  401a59:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  401a60:	61 00 
  401a62:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  401a69:	61 00 
  401a6b:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  401a72:	62 00 
  401a74:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  401a7b:	62 00 
  401a7d:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401a81:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401a86:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  401a8a:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  401a91:	60 00 
  401a93:	c5 db 59 25 1d 01 00 	vmulsd 0x11d(%rip),%xmm4,%xmm4        # 401bb8 <_IO_stdin_used+0x48>
  401a9a:	00 
  401a9b:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401aa2:	60 00 
  401aa4:	c5 eb 58 15 14 01 00 	vaddsd 0x114(%rip),%xmm2,%xmm2        # 401bc0 <_IO_stdin_used+0x50>
  401aab:	00 
  401aac:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401ab0:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401ab4:	c5 fb 10 25 f4 00 00 	vmovsd 0xf4(%rip),%xmm4        # 401bb0 <_IO_stdin_used+0x40>
  401abb:	00 
  401abc:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401ac0:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401ac7:	12 65 00 
  401aca:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401ad1:	65 00 
  401ad3:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401ad7:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  401ade:	0c 66 00 
  401ae1:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401ae8:	66 00 
  401aea:	c3                   	retq   
  401aeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401af0 <__libc_csu_init>:
  401af0:	41 57                	push   %r15
  401af2:	41 56                	push   %r14
  401af4:	41 89 ff             	mov    %edi,%r15d
  401af7:	41 55                	push   %r13
  401af9:	41 54                	push   %r12
  401afb:	4c 8d 25 ce 12 20 00 	lea    0x2012ce(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401b02:	55                   	push   %rbp
  401b03:	48 8d 2d de 12 20 00 	lea    0x2012de(%rip),%rbp        # 602de8 <__init_array_end>
  401b0a:	53                   	push   %rbx
  401b0b:	49 89 f6             	mov    %rsi,%r14
  401b0e:	49 89 d5             	mov    %rdx,%r13
  401b11:	4c 29 e5             	sub    %r12,%rbp
  401b14:	48 83 ec 08          	sub    $0x8,%rsp
  401b18:	48 c1 fd 03          	sar    $0x3,%rbp
  401b1c:	e8 4f eb ff ff       	callq  400670 <_init>
  401b21:	48 85 ed             	test   %rbp,%rbp
  401b24:	74 20                	je     401b46 <__libc_csu_init+0x56>
  401b26:	31 db                	xor    %ebx,%ebx
  401b28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401b2f:	00 
  401b30:	4c 89 ea             	mov    %r13,%rdx
  401b33:	4c 89 f6             	mov    %r14,%rsi
  401b36:	44 89 ff             	mov    %r15d,%edi
  401b39:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401b3d:	48 83 c3 01          	add    $0x1,%rbx
  401b41:	48 39 eb             	cmp    %rbp,%rbx
  401b44:	75 ea                	jne    401b30 <__libc_csu_init+0x40>
  401b46:	48 83 c4 08          	add    $0x8,%rsp
  401b4a:	5b                   	pop    %rbx
  401b4b:	5d                   	pop    %rbp
  401b4c:	41 5c                	pop    %r12
  401b4e:	41 5d                	pop    %r13
  401b50:	41 5e                	pop    %r14
  401b52:	41 5f                	pop    %r15
  401b54:	c3                   	retq   
  401b55:	90                   	nop
  401b56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401b5d:	00 00 00 

0000000000401b60 <__libc_csu_fini>:
  401b60:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401b64 <_fini>:
  401b64:	48 83 ec 08          	sub    $0x8,%rsp
  401b68:	48 83 c4 08          	add    $0x8,%rsp
  401b6c:	c3                   	retq   
