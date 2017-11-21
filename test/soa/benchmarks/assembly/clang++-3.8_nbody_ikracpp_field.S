
bin/clang++-3.8_nbody_ikracpp_field:     file format elf64-x86-64


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

0000000000400750 <_GLOBAL__sub_I_ikracpp_field.cc>:
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
  40077f:	49 c7 c0 d0 1b 40 00 	mov    $0x401bd0,%r8
  400786:	48 c7 c1 60 1b 40 00 	mov    $0x401b60,%rcx
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
  4008a9:	c5 fb 10 05 3f 13 00 	vmovsd 0x133f(%rip),%xmm0        # 401bf0 <_IO_stdin_used+0x10>
  4008b0:	00 
  4008b1:	c4 e2 f9 99 0d 2e 13 	vfmadd132sd 0x132e(%rip),%xmm0,%xmm1        # 401be8 <_IO_stdin_used+0x8>
  4008b8:	00 00 
  4008ba:	c5 fb 11 4c 24 28    	vmovsd %xmm1,0x28(%rsp)
  4008c0:	e8 4b fe ff ff       	callq  400710 <rand@plt>
  4008c5:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008c9:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008cd:	c5 fb 10 05 23 13 00 	vmovsd 0x1323(%rip),%xmm0        # 401bf8 <_IO_stdin_used+0x18>
  4008d4:	00 
  4008d5:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008d9:	c5 fb 10 05 1f 13 00 	vmovsd 0x131f(%rip),%xmm0        # 401c00 <_IO_stdin_used+0x20>
  4008e0:	00 
  4008e1:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008e6:	c5 fb 11 4c 24 20    	vmovsd %xmm1,0x20(%rsp)
  4008ec:	e8 1f fe ff ff       	callq  400710 <rand@plt>
  4008f1:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008f5:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  4008f9:	c5 fb 10 0d ff 12 00 	vmovsd 0x12ff(%rip),%xmm1        # 401c00 <_IO_stdin_used+0x20>
  400900:	00 
  400901:	c4 e2 f1 a9 05 ee 12 	vfmadd213sd 0x12ee(%rip),%xmm1,%xmm0        # 401bf8 <_IO_stdin_used+0x18>
  400908:	00 00 
  40090a:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
  400910:	e8 fb fd ff ff       	callq  400710 <rand@plt>
  400915:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400919:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  40091d:	c5 fb 10 05 e3 12 00 	vmovsd 0x12e3(%rip),%xmm0        # 401c08 <_IO_stdin_used+0x28>
  400924:	00 
  400925:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400929:	c5 fb 10 05 df 12 00 	vmovsd 0x12df(%rip),%xmm0        # 401c10 <_IO_stdin_used+0x30>
  400930:	00 
  400931:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400936:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  40093c:	e8 cf fd ff ff       	callq  400710 <rand@plt>
  400941:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400945:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400949:	c5 fb 10 0d bf 12 00 	vmovsd 0x12bf(%rip),%xmm1        # 401c10 <_IO_stdin_used+0x30>
  400950:	00 
  400951:	c4 e2 f1 a9 05 ae 12 	vfmadd213sd 0x12ae(%rip),%xmm1,%xmm0        # 401c08 <_IO_stdin_used+0x28>
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
  400ba0:	48 89 df             	mov    %rbx,%rdi
  400ba3:	4c 89 fe             	mov    %r15,%rsi
  400ba6:	e8 15 0f 00 00       	callq  401ac0 <_ZN4Body9add_forceEPS_>
  400bab:	48 83 c3 01          	add    $0x1,%rbx
  400baf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bb6:	75 e8                	jne    400ba0 <_Z14run_simulationv+0x1c0>
  400bb8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400bbf:	75 cf                	jne    400b90 <_Z14run_simulationv+0x1b0>
  400bc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bcd:	00 00 00 
  400bd0:	4c 89 f7             	mov    %r14,%rdi
  400bd3:	c5 fb 10 05 3d 10 00 	vmovsd 0x103d(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  400bda:	00 
  400bdb:	e8 40 0d 00 00       	callq  401920 <_ZN4Body6updateEd>
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
  400da0:	48 89 df             	mov    %rbx,%rdi
  400da3:	4c 89 fe             	mov    %r15,%rsi
  400da6:	e8 15 0d 00 00       	callq  401ac0 <_ZN4Body9add_forceEPS_>
  400dab:	48 83 c3 01          	add    $0x1,%rbx
  400daf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400db6:	75 e8                	jne    400da0 <_Z14run_simulationv+0x3c0>
  400db8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dbf:	75 cf                	jne    400d90 <_Z14run_simulationv+0x3b0>
  400dc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400dc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcd:	00 00 00 
  400dd0:	4c 89 f7             	mov    %r14,%rdi
  400dd3:	c5 fb 10 05 3d 0e 00 	vmovsd 0xe3d(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  400dda:	00 
  400ddb:	e8 40 0b 00 00       	callq  401920 <_ZN4Body6updateEd>
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
  400fa0:	48 89 df             	mov    %rbx,%rdi
  400fa3:	4c 89 fe             	mov    %r15,%rsi
  400fa6:	e8 15 0b 00 00       	callq  401ac0 <_ZN4Body9add_forceEPS_>
  400fab:	48 83 c3 01          	add    $0x1,%rbx
  400faf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fb6:	75 e8                	jne    400fa0 <_Z14run_simulationv+0x5c0>
  400fb8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400fbf:	75 cf                	jne    400f90 <_Z14run_simulationv+0x5b0>
  400fc1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400fc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fcd:	00 00 00 
  400fd0:	4c 89 f7             	mov    %r14,%rdi
  400fd3:	c5 fb 10 05 3d 0c 00 	vmovsd 0xc3d(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  400fda:	00 
  400fdb:	e8 40 09 00 00       	callq  401920 <_ZN4Body6updateEd>
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
  4011a0:	48 89 df             	mov    %rbx,%rdi
  4011a3:	4c 89 fe             	mov    %r15,%rsi
  4011a6:	e8 15 09 00 00       	callq  401ac0 <_ZN4Body9add_forceEPS_>
  4011ab:	48 83 c3 01          	add    $0x1,%rbx
  4011af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011b6:	75 e8                	jne    4011a0 <_Z14run_simulationv+0x7c0>
  4011b8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4011bf:	75 cf                	jne    401190 <_Z14run_simulationv+0x7b0>
  4011c1:	bb 10 00 00 00       	mov    $0x10,%ebx
  4011c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011cd:	00 00 00 
  4011d0:	4c 89 f7             	mov    %r14,%rdi
  4011d3:	c5 fb 10 05 3d 0a 00 	vmovsd 0xa3d(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  4011da:	00 
  4011db:	e8 40 07 00 00       	callq  401920 <_ZN4Body6updateEd>
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
  4013a0:	48 89 df             	mov    %rbx,%rdi
  4013a3:	4c 89 fe             	mov    %r15,%rsi
  4013a6:	e8 15 07 00 00       	callq  401ac0 <_ZN4Body9add_forceEPS_>
  4013ab:	48 83 c3 01          	add    $0x1,%rbx
  4013af:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013b6:	75 e8                	jne    4013a0 <_Z14run_simulationv+0x9c0>
  4013b8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4013bf:	75 cf                	jne    401390 <_Z14run_simulationv+0x9b0>
  4013c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4013c8:	0f 1f 84 00 00 00 00 
  4013cf:	00 
  4013d0:	4c 89 f7             	mov    %r14,%rdi
  4013d3:	c5 fb 10 05 3d 08 00 	vmovsd 0x83d(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  4013da:	00 
  4013db:	e8 40 05 00 00       	callq  401920 <_ZN4Body6updateEd>
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
  401423:	c5 fb 10 05 ed 07 00 	vmovsd 0x7ed(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
  40142a:	00 
  40142b:	e8 f0 05 00 00       	callq  401a20 <_ZN4Body22codengen_simple_updateEd>
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
  4014a1:	c5 fb 10 05 6f 07 00 	vmovsd 0x76f(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x38>
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
  4014f4:	c5 fb 59 0c cd 88 0c 	vmulsd 0x660c88(,%rcx,8),%xmm0,%xmm1
  4014fb:	66 00 
  4014fd:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  401504:	60 00 
  401506:	c5 f3 58 0c cd 88 18 	vaddsd 0x641888(,%rcx,8),%xmm1,%xmm1
  40150d:	64 00 
  40150f:	c5 fb 11 0c cd 88 18 	vmovsd %xmm1,0x641888(,%rcx,8)
  401516:	64 00 
  401518:	c5 fb 10 0c cd 88 1e 	vmovsd 0x631e88(,%rcx,8),%xmm1
  40151f:	63 00 
  401521:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x612a88(,%rcx,8),%xmm0,%xmm1
  401528:	2a 61 00 
  40152b:	c5 fb 11 0c cd 88 2a 	vmovsd %xmm1,0x612a88(,%rcx,8)
  401532:	61 00 
  401534:	c5 fb 10 0c cd 88 18 	vmovsd 0x641888(,%rcx,8),%xmm1
  40153b:	64 00 
  40153d:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x622488(,%rcx,8),%xmm0,%xmm1
  401544:	24 62 00 
  401547:	83 c2 ff             	add    $0xffffffff,%edx
  40154a:	c5 fb 11 0c cd 88 24 	vmovsd %xmm1,0x622488(,%rcx,8)
  401551:	62 00 
  401553:	0f 85 77 ff ff ff    	jne    4014d0 <main+0x80>
  401559:	48 89 c8             	mov    %rcx,%rax
  40155c:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  401563:	0f 85 57 ff ff ff    	jne    4014c0 <main+0x70>
  401569:	83 c3 01             	add    $0x1,%ebx
  40156c:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  401572:	0f 85 38 ff ff ff    	jne    4014b0 <main+0x60>
  401578:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  40157f:	de 1b 43 
  401582:	4c 89 f0             	mov    %r14,%rax
  401585:	48 f7 ee             	imul   %rsi
  401588:	49 89 d6             	mov    %rdx,%r14
  40158b:	4c 89 f1             	mov    %r14,%rcx
  40158e:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401592:	49 c1 fe 12          	sar    $0x12,%r14
  401596:	48 89 e8             	mov    %rbp,%rax
  401599:	48 f7 ee             	imul   %rsi
  40159c:	49 89 d5             	mov    %rdx,%r13
  40159f:	49 01 ce             	add    %rcx,%r14
  4015a2:	4c 89 e8             	mov    %r13,%rax
  4015a5:	48 c1 e8 3f          	shr    $0x3f,%rax
  4015a9:	49 c1 fd 12          	sar    $0x12,%r13
  4015ad:	49 01 c5             	add    %rax,%r13
  4015b0:	e8 4b f1 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4015b5:	49 89 c4             	mov    %rax,%r12
  4015b8:	4d 29 fc             	sub    %r15,%r12
  4015bb:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4015c0:	31 f6                	xor    %esi,%esi
  4015c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4015c9:	1f 84 00 00 00 00 00 
  4015d0:	c5 fb 10 04 f5 90 2a 	vmovsd 0x612a90(,%rsi,8),%xmm0
  4015d7:	61 00 
  4015d9:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  4015dd:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  4015e1:	01 e8                	add    %ebp,%eax
  4015e3:	c5 fb 10 0c f5 90 24 	vmovsd 0x622490(,%rsi,8),%xmm1
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
  401618:	48 83 fe 0a          	cmp    $0xa,%rsi
  40161c:	73 0c                	jae    40162a <main+0x1da>
  40161e:	bf 50 1c 40 00       	mov    $0x401c50,%edi
  401623:	b0 02                	mov    $0x2,%al
  401625:	e8 76 f0 ff ff       	callq  4006a0 <printf@plt>
  40162a:	44 29 fd             	sub    %r15d,%ebp
  40162d:	48 89 de             	mov    %rbx,%rsi
  401630:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401637:	75 97                	jne    4015d0 <main+0x180>
  401639:	4c 89 e0             	mov    %r12,%rax
  40163c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401643:	de 1b 43 
  401646:	48 f7 e9             	imul   %rcx
  401649:	48 89 d0             	mov    %rdx,%rax
  40164c:	48 c1 e8 3f          	shr    $0x3f,%rax
  401650:	48 c1 fa 12          	sar    $0x12,%rdx
  401654:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401658:	bf 65 1c 40 00       	mov    $0x401c65,%edi
  40165d:	31 c0                	xor    %eax,%eax
  40165f:	4c 89 f6             	mov    %r14,%rsi
  401662:	4c 89 ea             	mov    %r13,%rdx
  401665:	41 89 e8             	mov    %ebp,%r8d
  401668:	e8 33 f0 ff ff       	callq  4006a0 <printf@plt>
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
  401680:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401687:	00 00 00 00 00 
  40168c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401693:	00 00 00 00 00 
  401698:	c3                   	retq   
  401699:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004016a0 <_Z17codegen_add_forceP4BodyS0_>:
  4016a0:	48 39 f7             	cmp    %rsi,%rdi
  4016a3:	0f 84 91 00 00 00    	je     40173a <_Z17codegen_add_forceP4BodyS0_+0x9a>
  4016a9:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  4016b0:	61 00 
  4016b2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  4016b9:	61 00 
  4016bb:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  4016c2:	62 00 
  4016c4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  4016cb:	62 00 
  4016cd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  4016d1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  4016d6:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  4016da:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  4016e1:	60 00 
  4016e3:	c5 db 59 25 3d 05 00 	vmulsd 0x53d(%rip),%xmm4,%xmm4        # 401c28 <_IO_stdin_used+0x48>
  4016ea:	00 
  4016eb:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  4016f2:	60 00 
  4016f4:	c5 eb 58 15 34 05 00 	vaddsd 0x534(%rip),%xmm2,%xmm2        # 401c30 <_IO_stdin_used+0x50>
  4016fb:	00 
  4016fc:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401700:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401704:	c5 fb 10 25 14 05 00 	vmovsd 0x514(%rip),%xmm4        # 401c20 <_IO_stdin_used+0x40>
  40170b:	00 
  40170c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401710:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401717:	12 65 00 
  40171a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401721:	65 00 
  401723:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401727:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  40172e:	0c 66 00 
  401731:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401738:	66 00 
  40173a:	c3                   	retq   
  40173b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401740 <_Z14codegen_updateP4Bodyd>:
  401740:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401747:	65 00 
  401749:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401750:	60 00 
  401752:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401759:	63 00 
  40175b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401762:	63 00 
  401764:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40176b:	66 00 
  40176d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401774:	60 00 
  401776:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40177d:	64 00 
  40177f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401786:	64 00 
  401788:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  40178f:	63 00 
  401791:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401798:	2a 61 00 
  40179b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4017a2:	61 00 
  4017a4:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4017ab:	64 00 
  4017ad:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4017b4:	24 62 00 
  4017b7:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4017be:	62 00 
  4017c0:	c5 fb 10 04 fd 88 2a 	vmovsd 0x612a88(,%rdi,8),%xmm0
  4017c7:	61 00 
  4017c9:	c5 fb 10 15 27 04 00 	vmovsd 0x427(%rip),%xmm2        # 401bf8 <_IO_stdin_used+0x18>
  4017d0:	00 
  4017d1:	c5 f9 2e d0          	vucomisd %xmm0,%xmm2
  4017d5:	77 0a                	ja     4017e1 <_Z14codegen_updateP4Bodyd+0xa1>
  4017d7:	c5 f9 2e 05 41 04 00 	vucomisd 0x441(%rip),%xmm0        # 401c20 <_IO_stdin_used+0x40>
  4017de:	00 
  4017df:	76 23                	jbe    401804 <_Z14codegen_updateP4Bodyd+0xc4>
  4017e1:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4017e8:	63 00 
  4017ea:	c5 f9 57 05 4e 04 00 	vxorpd 0x44e(%rip),%xmm0,%xmm0        # 401c40 <_IO_stdin_used+0x60>
  4017f1:	00 
  4017f2:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4017f9:	63 00 
  4017fb:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  401802:	62 00 
  401804:	c5 f9 2e 0d ec 03 00 	vucomisd 0x3ec(%rip),%xmm1        # 401bf8 <_IO_stdin_used+0x18>
  40180b:	00 
  40180c:	72 0a                	jb     401818 <_Z14codegen_updateP4Bodyd+0xd8>
  40180e:	c5 f9 2e 0d 0a 04 00 	vucomisd 0x40a(%rip),%xmm1        # 401c20 <_IO_stdin_used+0x40>
  401815:	00 
  401816:	76 1a                	jbe    401832 <_Z14codegen_updateP4Bodyd+0xf2>
  401818:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  40181f:	64 00 
  401821:	c5 f9 57 05 17 04 00 	vxorpd 0x417(%rip),%xmm0,%xmm0        # 401c40 <_IO_stdin_used+0x60>
  401828:	00 
  401829:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  401830:	64 00 
  401832:	c3                   	retq   
  401833:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40183a:	84 00 00 00 00 00 

0000000000401840 <_Z26codegen_reset_force_manualP4Body>:
  401840:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401847:	00 00 00 00 00 
  40184c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401853:	00 00 00 00 00 
  401858:	c3                   	retq   
  401859:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401860 <_Z24codegen_reset_force_halfP4Body>:
  401860:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401867:	00 00 00 00 00 
  40186c:	c3                   	retq   
  40186d:	0f 1f 00             	nopl   (%rax)

0000000000401870 <_Z18codegen_reset_massP4Body>:
  401870:	48 c7 04 fd 88 30 60 	movq   $0x0,0x603088(,%rdi,8)
  401877:	00 00 00 00 00 
  40187c:	c3                   	retq   
  40187d:	0f 1f 00             	nopl   (%rax)

0000000000401880 <_Z22codengen_simple_updateP4Bodyd>:
  401880:	b8 64 00 00 00       	mov    $0x64,%eax
  401885:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40188c:	00 00 00 00 
  401890:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401897:	65 00 
  401899:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018a0:	60 00 
  4018a2:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  4018a9:	63 00 
  4018ab:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  4018b2:	63 00 
  4018b4:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  4018bb:	66 00 
  4018bd:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018c4:	60 00 
  4018c6:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  4018cd:	64 00 
  4018cf:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  4018d6:	64 00 
  4018d8:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  4018df:	63 00 
  4018e1:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4018e8:	2a 61 00 
  4018eb:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4018f2:	61 00 
  4018f4:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4018fb:	64 00 
  4018fd:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401904:	24 62 00 
  401907:	83 c0 ff             	add    $0xffffffff,%eax
  40190a:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401911:	62 00 
  401913:	0f 85 77 ff ff ff    	jne    401890 <_Z22codengen_simple_updateP4Bodyd+0x10>
  401919:	c3                   	retq   
  40191a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401920 <_ZN4Body6updateEd>:
  401920:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401927:	65 00 
  401929:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401930:	60 00 
  401932:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401939:	63 00 
  40193b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401942:	63 00 
  401944:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40194b:	66 00 
  40194d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401954:	60 00 
  401956:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40195d:	64 00 
  40195f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401966:	64 00 
  401968:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  40196f:	63 00 
  401971:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401978:	2a 61 00 
  40197b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401982:	61 00 
  401984:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  40198b:	64 00 
  40198d:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401994:	24 62 00 
  401997:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  40199e:	62 00 
  4019a0:	c5 fb 10 04 fd 88 2a 	vmovsd 0x612a88(,%rdi,8),%xmm0
  4019a7:	61 00 
  4019a9:	c5 fb 10 15 47 02 00 	vmovsd 0x247(%rip),%xmm2        # 401bf8 <_IO_stdin_used+0x18>
  4019b0:	00 
  4019b1:	c5 f9 2e d0          	vucomisd %xmm0,%xmm2
  4019b5:	77 0a                	ja     4019c1 <_ZN4Body6updateEd+0xa1>
  4019b7:	c5 f9 2e 05 61 02 00 	vucomisd 0x261(%rip),%xmm0        # 401c20 <_IO_stdin_used+0x40>
  4019be:	00 
  4019bf:	76 23                	jbe    4019e4 <_ZN4Body6updateEd+0xc4>
  4019c1:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4019c8:	63 00 
  4019ca:	c5 f9 57 05 6e 02 00 	vxorpd 0x26e(%rip),%xmm0,%xmm0        # 401c40 <_IO_stdin_used+0x60>
  4019d1:	00 
  4019d2:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4019d9:	63 00 
  4019db:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4019e2:	62 00 
  4019e4:	c5 f9 2e 0d 0c 02 00 	vucomisd 0x20c(%rip),%xmm1        # 401bf8 <_IO_stdin_used+0x18>
  4019eb:	00 
  4019ec:	72 0a                	jb     4019f8 <_ZN4Body6updateEd+0xd8>
  4019ee:	c5 f9 2e 0d 2a 02 00 	vucomisd 0x22a(%rip),%xmm1        # 401c20 <_IO_stdin_used+0x40>
  4019f5:	00 
  4019f6:	76 1a                	jbe    401a12 <_ZN4Body6updateEd+0xf2>
  4019f8:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  4019ff:	64 00 
  401a01:	c5 f9 57 05 37 02 00 	vxorpd 0x237(%rip),%xmm0,%xmm0        # 401c40 <_IO_stdin_used+0x60>
  401a08:	00 
  401a09:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  401a10:	64 00 
  401a12:	c3                   	retq   
  401a13:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401a1a:	00 00 00 
  401a1d:	0f 1f 00             	nopl   (%rax)

0000000000401a20 <_ZN4Body22codengen_simple_updateEd>:
  401a20:	b8 64 00 00 00       	mov    $0x64,%eax
  401a25:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  401a2c:	00 00 00 00 
  401a30:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401a37:	65 00 
  401a39:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a40:	60 00 
  401a42:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401a49:	63 00 
  401a4b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401a52:	63 00 
  401a54:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  401a5b:	66 00 
  401a5d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a64:	60 00 
  401a66:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  401a6d:	64 00 
  401a6f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401a76:	64 00 
  401a78:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  401a7f:	63 00 
  401a81:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401a88:	2a 61 00 
  401a8b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401a92:	61 00 
  401a94:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401a9b:	64 00 
  401a9d:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401aa4:	24 62 00 
  401aa7:	83 c0 ff             	add    $0xffffffff,%eax
  401aaa:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401ab1:	62 00 
  401ab3:	0f 85 77 ff ff ff    	jne    401a30 <_ZN4Body22codengen_simple_updateEd+0x10>
  401ab9:	c3                   	retq   
  401aba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401ac0 <_ZN4Body9add_forceEPS_>:
  401ac0:	48 39 f7             	cmp    %rsi,%rdi
  401ac3:	0f 84 91 00 00 00    	je     401b5a <_ZN4Body9add_forceEPS_+0x9a>
  401ac9:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  401ad0:	61 00 
  401ad2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  401ad9:	61 00 
  401adb:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  401ae2:	62 00 
  401ae4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  401aeb:	62 00 
  401aed:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401af1:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401af6:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  401afa:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  401b01:	60 00 
  401b03:	c5 db 59 25 1d 01 00 	vmulsd 0x11d(%rip),%xmm4,%xmm4        # 401c28 <_IO_stdin_used+0x48>
  401b0a:	00 
  401b0b:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401b12:	60 00 
  401b14:	c5 eb 58 15 14 01 00 	vaddsd 0x114(%rip),%xmm2,%xmm2        # 401c30 <_IO_stdin_used+0x50>
  401b1b:	00 
  401b1c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401b20:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401b24:	c5 fb 10 25 f4 00 00 	vmovsd 0xf4(%rip),%xmm4        # 401c20 <_IO_stdin_used+0x40>
  401b2b:	00 
  401b2c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401b30:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401b37:	12 65 00 
  401b3a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401b41:	65 00 
  401b43:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401b47:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  401b4e:	0c 66 00 
  401b51:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401b58:	66 00 
  401b5a:	c3                   	retq   
  401b5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401b60 <__libc_csu_init>:
  401b60:	41 57                	push   %r15
  401b62:	41 56                	push   %r14
  401b64:	41 89 ff             	mov    %edi,%r15d
  401b67:	41 55                	push   %r13
  401b69:	41 54                	push   %r12
  401b6b:	4c 8d 25 5e 12 20 00 	lea    0x20125e(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401b72:	55                   	push   %rbp
  401b73:	48 8d 2d 6e 12 20 00 	lea    0x20126e(%rip),%rbp        # 602de8 <__init_array_end>
  401b7a:	53                   	push   %rbx
  401b7b:	49 89 f6             	mov    %rsi,%r14
  401b7e:	49 89 d5             	mov    %rdx,%r13
  401b81:	4c 29 e5             	sub    %r12,%rbp
  401b84:	48 83 ec 08          	sub    $0x8,%rsp
  401b88:	48 c1 fd 03          	sar    $0x3,%rbp
  401b8c:	e8 df ea ff ff       	callq  400670 <_init>
  401b91:	48 85 ed             	test   %rbp,%rbp
  401b94:	74 20                	je     401bb6 <__libc_csu_init+0x56>
  401b96:	31 db                	xor    %ebx,%ebx
  401b98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401b9f:	00 
  401ba0:	4c 89 ea             	mov    %r13,%rdx
  401ba3:	4c 89 f6             	mov    %r14,%rsi
  401ba6:	44 89 ff             	mov    %r15d,%edi
  401ba9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401bad:	48 83 c3 01          	add    $0x1,%rbx
  401bb1:	48 39 eb             	cmp    %rbp,%rbx
  401bb4:	75 ea                	jne    401ba0 <__libc_csu_init+0x40>
  401bb6:	48 83 c4 08          	add    $0x8,%rsp
  401bba:	5b                   	pop    %rbx
  401bbb:	5d                   	pop    %rbp
  401bbc:	41 5c                	pop    %r12
  401bbe:	41 5d                	pop    %r13
  401bc0:	41 5e                	pop    %r14
  401bc2:	41 5f                	pop    %r15
  401bc4:	c3                   	retq   
  401bc5:	90                   	nop
  401bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401bcd:	00 00 00 

0000000000401bd0 <__libc_csu_fini>:
  401bd0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401bd4 <_fini>:
  401bd4:	48 83 ec 08          	sub    $0x8,%rsp
  401bd8:	48 83 c4 08          	add    $0x8,%rsp
  401bdc:	c3                   	retq   
