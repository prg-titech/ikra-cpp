
bin/clang++-3.8_nbody_ikracpp:     file format elf64-x86-64


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

0000000000400750 <_GLOBAL__sub_I_ikracpp.cc>:
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
  40077f:	49 c7 c0 70 1b 40 00 	mov    $0x401b70,%r8
  400786:	48 c7 c1 00 1b 40 00 	mov    $0x401b00,%rcx
  40078d:	48 c7 c7 60 14 40 00 	mov    $0x401460,%rdi
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

0000000000400880 <_ZN4Body7storageEv>:
  400880:	b8 78 30 60 00       	mov    $0x603078,%eax
  400885:	c3                   	retq   
  400886:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40088d:	00 00 00 

0000000000400890 <_Z13instantiationv>:
  400890:	53                   	push   %rbx
  400891:	48 83 ec 30          	sub    $0x30,%rsp
  400895:	bf 2a 00 00 00       	mov    $0x2a,%edi
  40089a:	e8 51 fe ff ff       	callq  4006f0 <srand@plt>
  40089f:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4008a4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008ab:	00 00 00 00 00 
  4008b0:	e8 5b fe ff ff       	callq  400710 <rand@plt>
  4008b5:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008b9:	c5 fb 10 05 cf 12 00 	vmovsd 0x12cf(%rip),%xmm0        # 401b90 <_IO_stdin_used+0x10>
  4008c0:	00 
  4008c1:	c4 e2 f9 99 0d be 12 	vfmadd132sd 0x12be(%rip),%xmm0,%xmm1        # 401b88 <_IO_stdin_used+0x8>
  4008c8:	00 00 
  4008ca:	c5 fb 11 4c 24 28    	vmovsd %xmm1,0x28(%rsp)
  4008d0:	e8 3b fe ff ff       	callq  400710 <rand@plt>
  4008d5:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008d9:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008dd:	c5 fb 10 05 b3 12 00 	vmovsd 0x12b3(%rip),%xmm0        # 401b98 <_IO_stdin_used+0x18>
  4008e4:	00 
  4008e5:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008e9:	c5 fb 10 05 af 12 00 	vmovsd 0x12af(%rip),%xmm0        # 401ba0 <_IO_stdin_used+0x20>
  4008f0:	00 
  4008f1:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008f6:	c5 fb 11 4c 24 20    	vmovsd %xmm1,0x20(%rsp)
  4008fc:	e8 0f fe ff ff       	callq  400710 <rand@plt>
  400901:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400905:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400909:	c5 fb 10 0d 8f 12 00 	vmovsd 0x128f(%rip),%xmm1        # 401ba0 <_IO_stdin_used+0x20>
  400910:	00 
  400911:	c4 e2 f1 a9 05 7e 12 	vfmadd213sd 0x127e(%rip),%xmm1,%xmm0        # 401b98 <_IO_stdin_used+0x18>
  400918:	00 00 
  40091a:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
  400920:	e8 eb fd ff ff       	callq  400710 <rand@plt>
  400925:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400929:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  40092d:	c5 fb 10 05 73 12 00 	vmovsd 0x1273(%rip),%xmm0        # 401ba8 <_IO_stdin_used+0x28>
  400934:	00 
  400935:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400939:	c5 fb 10 05 6f 12 00 	vmovsd 0x126f(%rip),%xmm0        # 401bb0 <_IO_stdin_used+0x30>
  400940:	00 
  400941:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400946:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  40094c:	e8 bf fd ff ff       	callq  400710 <rand@plt>
  400951:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400955:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400959:	c5 fb 10 0d 4f 12 00 	vmovsd 0x124f(%rip),%xmm1        # 401bb0 <_IO_stdin_used+0x30>
  400960:	00 
  400961:	c4 e2 f1 a9 05 3e 12 	vfmadd213sd 0x123e(%rip),%xmm1,%xmm0        # 401ba8 <_IO_stdin_used+0x28>
  400968:	00 00 
  40096a:	48 8b 05 07 27 20 00 	mov    0x202707(%rip),%rax        # 603078 <__Bodydata_buffer>
  400971:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400975:	48 89 0d fc 26 20 00 	mov    %rcx,0x2026fc(%rip)        # 603078 <__Bodydata_buffer>
  40097c:	c5 fb 10 4c 24 28    	vmovsd 0x28(%rsp),%xmm1
  400982:	c5 fb 11 0c c5 90 30 	vmovsd %xmm1,0x603090(,%rax,8)
  400989:	60 00 
  40098b:	c5 fb 10 4c 24 20    	vmovsd 0x20(%rsp),%xmm1
  400991:	c5 fb 11 0c c5 90 2a 	vmovsd %xmm1,0x612a90(,%rax,8)
  400998:	61 00 
  40099a:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  4009a0:	c5 fb 11 0c c5 90 24 	vmovsd %xmm1,0x622490(,%rax,8)
  4009a7:	62 00 
  4009a9:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  4009af:	c5 fb 11 0c c5 90 1e 	vmovsd %xmm1,0x631e90(,%rax,8)
  4009b6:	63 00 
  4009b8:	c5 fb 11 04 c5 90 18 	vmovsd %xmm0,0x641890(,%rax,8)
  4009bf:	64 00 
  4009c1:	48 c7 04 c5 90 12 65 	movq   $0x0,0x651290(,%rax,8)
  4009c8:	00 00 00 00 00 
  4009cd:	48 c7 04 c5 90 0c 66 	movq   $0x0,0x660c90(,%rax,8)
  4009d4:	00 00 00 00 00 
  4009d9:	83 c3 ff             	add    $0xffffffff,%ebx
  4009dc:	0f 85 ce fe ff ff    	jne    4008b0 <_Z13instantiationv+0x20>
  4009e2:	48 83 c4 30          	add    $0x30,%rsp
  4009e6:	5b                   	pop    %rbx
  4009e7:	c3                   	retq   
  4009e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009ef:	00 

00000000004009f0 <_Z14run_simulationv>:
  4009f0:	41 57                	push   %r15
  4009f2:	41 56                	push   %r14
  4009f4:	53                   	push   %rbx
  4009f5:	b8 10 00 00 00       	mov    $0x10,%eax
  4009fa:	45 31 ff             	xor    %r15d,%r15d
  4009fd:	0f 1f 00             	nopl   (%rax)
  400a00:	48 c7 04 c5 10 12 65 	movq   $0x0,0x651210(,%rax,8)
  400a07:	00 00 00 00 00 
  400a0c:	48 c7 04 c5 10 0c 66 	movq   $0x0,0x660c10(,%rax,8)
  400a13:	00 00 00 00 00 
  400a18:	48 c7 04 c5 18 12 65 	movq   $0x0,0x651218(,%rax,8)
  400a1f:	00 00 00 00 00 
  400a24:	48 c7 04 c5 18 0c 66 	movq   $0x0,0x660c18(,%rax,8)
  400a2b:	00 00 00 00 00 
  400a30:	48 c7 04 c5 20 12 65 	movq   $0x0,0x651220(,%rax,8)
  400a37:	00 00 00 00 00 
  400a3c:	48 c7 04 c5 20 0c 66 	movq   $0x0,0x660c20(,%rax,8)
  400a43:	00 00 00 00 00 
  400a48:	48 c7 04 c5 28 12 65 	movq   $0x0,0x651228(,%rax,8)
  400a4f:	00 00 00 00 00 
  400a54:	48 c7 04 c5 28 0c 66 	movq   $0x0,0x660c28(,%rax,8)
  400a5b:	00 00 00 00 00 
  400a60:	48 c7 04 c5 30 12 65 	movq   $0x0,0x651230(,%rax,8)
  400a67:	00 00 00 00 00 
  400a6c:	48 c7 04 c5 30 0c 66 	movq   $0x0,0x660c30(,%rax,8)
  400a73:	00 00 00 00 00 
  400a78:	48 c7 04 c5 38 12 65 	movq   $0x0,0x651238(,%rax,8)
  400a7f:	00 00 00 00 00 
  400a84:	48 c7 04 c5 38 0c 66 	movq   $0x0,0x660c38(,%rax,8)
  400a8b:	00 00 00 00 00 
  400a90:	48 c7 04 c5 40 12 65 	movq   $0x0,0x651240(,%rax,8)
  400a97:	00 00 00 00 00 
  400a9c:	48 c7 04 c5 40 0c 66 	movq   $0x0,0x660c40(,%rax,8)
  400aa3:	00 00 00 00 00 
  400aa8:	48 c7 04 c5 48 12 65 	movq   $0x0,0x651248(,%rax,8)
  400aaf:	00 00 00 00 00 
  400ab4:	48 c7 04 c5 48 0c 66 	movq   $0x0,0x660c48(,%rax,8)
  400abb:	00 00 00 00 00 
  400ac0:	48 c7 04 c5 50 12 65 	movq   $0x0,0x651250(,%rax,8)
  400ac7:	00 00 00 00 00 
  400acc:	48 c7 04 c5 50 0c 66 	movq   $0x0,0x660c50(,%rax,8)
  400ad3:	00 00 00 00 00 
  400ad8:	48 c7 04 c5 58 12 65 	movq   $0x0,0x651258(,%rax,8)
  400adf:	00 00 00 00 00 
  400ae4:	48 c7 04 c5 58 0c 66 	movq   $0x0,0x660c58(,%rax,8)
  400aeb:	00 00 00 00 00 
  400af0:	48 c7 04 c5 60 12 65 	movq   $0x0,0x651260(,%rax,8)
  400af7:	00 00 00 00 00 
  400afc:	48 c7 04 c5 60 0c 66 	movq   $0x0,0x660c60(,%rax,8)
  400b03:	00 00 00 00 00 
  400b08:	48 c7 04 c5 68 12 65 	movq   $0x0,0x651268(,%rax,8)
  400b0f:	00 00 00 00 00 
  400b14:	48 c7 04 c5 68 0c 66 	movq   $0x0,0x660c68(,%rax,8)
  400b1b:	00 00 00 00 00 
  400b20:	48 c7 04 c5 70 12 65 	movq   $0x0,0x651270(,%rax,8)
  400b27:	00 00 00 00 00 
  400b2c:	48 c7 04 c5 70 0c 66 	movq   $0x0,0x660c70(,%rax,8)
  400b33:	00 00 00 00 00 
  400b38:	48 c7 04 c5 78 12 65 	movq   $0x0,0x651278(,%rax,8)
  400b3f:	00 00 00 00 00 
  400b44:	48 c7 04 c5 78 0c 66 	movq   $0x0,0x660c78(,%rax,8)
  400b4b:	00 00 00 00 00 
  400b50:	48 c7 04 c5 80 12 65 	movq   $0x0,0x651280(,%rax,8)
  400b57:	00 00 00 00 00 
  400b5c:	48 c7 04 c5 80 0c 66 	movq   $0x0,0x660c80(,%rax,8)
  400b63:	00 00 00 00 00 
  400b68:	48 c7 04 c5 88 12 65 	movq   $0x0,0x651288(,%rax,8)
  400b6f:	00 00 00 00 00 
  400b74:	48 c7 04 c5 88 0c 66 	movq   $0x0,0x660c88(,%rax,8)
  400b7b:	00 00 00 00 00 
  400b80:	48 83 c0 10          	add    $0x10,%rax
  400b84:	48 3d 50 1f 00 00    	cmp    $0x1f50,%rax
  400b8a:	0f 85 70 fe ff ff    	jne    400a00 <_Z14run_simulationv+0x10>
  400b90:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 
  400ba0:	49 83 c7 01          	add    $0x1,%r15
  400ba4:	bb 01 00 00 00       	mov    $0x1,%ebx
  400ba9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400bb0:	48 89 df             	mov    %rbx,%rdi
  400bb3:	4c 89 fe             	mov    %r15,%rsi
  400bb6:	e8 a5 0e 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400bbb:	48 83 c3 01          	add    $0x1,%rbx
  400bbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bc6:	75 e8                	jne    400bb0 <_Z14run_simulationv+0x1c0>
  400bc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400bcf:	75 cf                	jne    400ba0 <_Z14run_simulationv+0x1b0>
  400bd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 
  400be0:	4c 89 f7             	mov    %r14,%rdi
  400be3:	c5 fb 10 05 cd 0f 00 	vmovsd 0xfcd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400bea:	00 
  400beb:	e8 00 0d 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400bf0:	49 83 c6 01          	add    $0x1,%r14
  400bf4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400bfb:	75 e3                	jne    400be0 <_Z14run_simulationv+0x1f0>
  400bfd:	45 31 ff             	xor    %r15d,%r15d
  400c00:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  400c07:	00 00 00 00 00 
  400c0c:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  400c13:	00 00 00 00 00 
  400c18:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  400c1f:	00 00 00 00 00 
  400c24:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  400c2b:	00 00 00 00 00 
  400c30:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  400c37:	00 00 00 00 00 
  400c3c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  400c43:	00 00 00 00 00 
  400c48:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  400c4f:	00 00 00 00 00 
  400c54:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  400c5b:	00 00 00 00 00 
  400c60:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  400c67:	00 00 00 00 00 
  400c6c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  400c73:	00 00 00 00 00 
  400c78:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  400c7f:	00 00 00 00 00 
  400c84:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  400c8b:	00 00 00 00 00 
  400c90:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  400c97:	00 00 00 00 00 
  400c9c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  400ca3:	00 00 00 00 00 
  400ca8:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  400caf:	00 00 00 00 00 
  400cb4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  400cbb:	00 00 00 00 00 
  400cc0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  400cc7:	00 00 00 00 00 
  400ccc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  400cd3:	00 00 00 00 00 
  400cd8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  400cdf:	00 00 00 00 00 
  400ce4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  400ceb:	00 00 00 00 00 
  400cf0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  400cf7:	00 00 00 00 00 
  400cfc:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  400d03:	00 00 00 00 00 
  400d08:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  400d0f:	00 00 00 00 00 
  400d14:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  400d1b:	00 00 00 00 00 
  400d20:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  400d27:	00 00 00 00 00 
  400d2c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  400d33:	00 00 00 00 00 
  400d38:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  400d3f:	00 00 00 00 00 
  400d44:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  400d4b:	00 00 00 00 00 
  400d50:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  400d57:	00 00 00 00 00 
  400d5c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  400d63:	00 00 00 00 00 
  400d68:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  400d6f:	00 00 00 00 00 
  400d74:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  400d7b:	00 00 00 00 00 
  400d80:	48 83 c3 10          	add    $0x10,%rbx
  400d84:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  400d8b:	0f 85 6f fe ff ff    	jne    400c00 <_Z14run_simulationv+0x210>
  400d91:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400d97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d9e:	00 00 
  400da0:	49 83 c7 01          	add    $0x1,%r15
  400da4:	bb 01 00 00 00       	mov    $0x1,%ebx
  400da9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400db0:	48 89 df             	mov    %rbx,%rdi
  400db3:	4c 89 fe             	mov    %r15,%rsi
  400db6:	e8 a5 0c 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400dbb:	48 83 c3 01          	add    $0x1,%rbx
  400dbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400dc6:	75 e8                	jne    400db0 <_Z14run_simulationv+0x3c0>
  400dc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dcf:	75 cf                	jne    400da0 <_Z14run_simulationv+0x3b0>
  400dd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400dd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddd:	00 00 00 
  400de0:	4c 89 f7             	mov    %r14,%rdi
  400de3:	c5 fb 10 05 cd 0d 00 	vmovsd 0xdcd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400dea:	00 
  400deb:	e8 00 0b 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400df0:	49 83 c6 01          	add    $0x1,%r14
  400df4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400dfb:	75 e3                	jne    400de0 <_Z14run_simulationv+0x3f0>
  400dfd:	45 31 ff             	xor    %r15d,%r15d
  400e00:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  400e07:	00 00 00 00 00 
  400e0c:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  400e13:	00 00 00 00 00 
  400e18:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  400e1f:	00 00 00 00 00 
  400e24:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  400e2b:	00 00 00 00 00 
  400e30:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  400e37:	00 00 00 00 00 
  400e3c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  400e43:	00 00 00 00 00 
  400e48:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  400e4f:	00 00 00 00 00 
  400e54:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  400e5b:	00 00 00 00 00 
  400e60:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  400e67:	00 00 00 00 00 
  400e6c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  400e73:	00 00 00 00 00 
  400e78:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  400e7f:	00 00 00 00 00 
  400e84:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  400e8b:	00 00 00 00 00 
  400e90:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  400e97:	00 00 00 00 00 
  400e9c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  400ea3:	00 00 00 00 00 
  400ea8:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  400eaf:	00 00 00 00 00 
  400eb4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  400ebb:	00 00 00 00 00 
  400ec0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  400ec7:	00 00 00 00 00 
  400ecc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  400ed3:	00 00 00 00 00 
  400ed8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  400edf:	00 00 00 00 00 
  400ee4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  400eeb:	00 00 00 00 00 
  400ef0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  400ef7:	00 00 00 00 00 
  400efc:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  400f03:	00 00 00 00 00 
  400f08:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  400f0f:	00 00 00 00 00 
  400f14:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  400f1b:	00 00 00 00 00 
  400f20:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  400f27:	00 00 00 00 00 
  400f2c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  400f33:	00 00 00 00 00 
  400f38:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  400f3f:	00 00 00 00 00 
  400f44:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  400f4b:	00 00 00 00 00 
  400f50:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  400f57:	00 00 00 00 00 
  400f5c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  400f63:	00 00 00 00 00 
  400f68:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  400f6f:	00 00 00 00 00 
  400f74:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  400f7b:	00 00 00 00 00 
  400f80:	48 83 c3 10          	add    $0x10,%rbx
  400f84:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  400f8b:	0f 85 6f fe ff ff    	jne    400e00 <_Z14run_simulationv+0x410>
  400f91:	41 be 01 00 00 00    	mov    $0x1,%r14d
  400f97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f9e:	00 00 
  400fa0:	49 83 c7 01          	add    $0x1,%r15
  400fa4:	bb 01 00 00 00       	mov    $0x1,%ebx
  400fa9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400fb0:	48 89 df             	mov    %rbx,%rdi
  400fb3:	4c 89 fe             	mov    %r15,%rsi
  400fb6:	e8 a5 0a 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  400fbb:	48 83 c3 01          	add    $0x1,%rbx
  400fbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fc6:	75 e8                	jne    400fb0 <_Z14run_simulationv+0x5c0>
  400fc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400fcf:	75 cf                	jne    400fa0 <_Z14run_simulationv+0x5b0>
  400fd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400fd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fdd:	00 00 00 
  400fe0:	4c 89 f7             	mov    %r14,%rdi
  400fe3:	c5 fb 10 05 cd 0b 00 	vmovsd 0xbcd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  400fea:	00 
  400feb:	e8 00 09 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  400ff0:	49 83 c6 01          	add    $0x1,%r14
  400ff4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  400ffb:	75 e3                	jne    400fe0 <_Z14run_simulationv+0x5f0>
  400ffd:	45 31 ff             	xor    %r15d,%r15d
  401000:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  401007:	00 00 00 00 00 
  40100c:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  401013:	00 00 00 00 00 
  401018:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  40101f:	00 00 00 00 00 
  401024:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  40102b:	00 00 00 00 00 
  401030:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  401037:	00 00 00 00 00 
  40103c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  401043:	00 00 00 00 00 
  401048:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  40104f:	00 00 00 00 00 
  401054:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  40105b:	00 00 00 00 00 
  401060:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  401067:	00 00 00 00 00 
  40106c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  401073:	00 00 00 00 00 
  401078:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  40107f:	00 00 00 00 00 
  401084:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  40108b:	00 00 00 00 00 
  401090:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  401097:	00 00 00 00 00 
  40109c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  4010a3:	00 00 00 00 00 
  4010a8:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  4010af:	00 00 00 00 00 
  4010b4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  4010bb:	00 00 00 00 00 
  4010c0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  4010c7:	00 00 00 00 00 
  4010cc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  4010d3:	00 00 00 00 00 
  4010d8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  4010df:	00 00 00 00 00 
  4010e4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  4010eb:	00 00 00 00 00 
  4010f0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  4010f7:	00 00 00 00 00 
  4010fc:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  401103:	00 00 00 00 00 
  401108:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  40110f:	00 00 00 00 00 
  401114:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  40111b:	00 00 00 00 00 
  401120:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  401127:	00 00 00 00 00 
  40112c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  401133:	00 00 00 00 00 
  401138:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  40113f:	00 00 00 00 00 
  401144:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  40114b:	00 00 00 00 00 
  401150:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  401157:	00 00 00 00 00 
  40115c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  401163:	00 00 00 00 00 
  401168:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  40116f:	00 00 00 00 00 
  401174:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  40117b:	00 00 00 00 00 
  401180:	48 83 c3 10          	add    $0x10,%rbx
  401184:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  40118b:	0f 85 6f fe ff ff    	jne    401000 <_Z14run_simulationv+0x610>
  401191:	41 be 01 00 00 00    	mov    $0x1,%r14d
  401197:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40119e:	00 00 
  4011a0:	49 83 c7 01          	add    $0x1,%r15
  4011a4:	bb 01 00 00 00       	mov    $0x1,%ebx
  4011a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4011b0:	48 89 df             	mov    %rbx,%rdi
  4011b3:	4c 89 fe             	mov    %r15,%rsi
  4011b6:	e8 a5 08 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  4011bb:	48 83 c3 01          	add    $0x1,%rbx
  4011bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011c6:	75 e8                	jne    4011b0 <_Z14run_simulationv+0x7c0>
  4011c8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4011cf:	75 cf                	jne    4011a0 <_Z14run_simulationv+0x7b0>
  4011d1:	bb 10 00 00 00       	mov    $0x10,%ebx
  4011d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011dd:	00 00 00 
  4011e0:	4c 89 f7             	mov    %r14,%rdi
  4011e3:	c5 fb 10 05 cd 09 00 	vmovsd 0x9cd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  4011ea:	00 
  4011eb:	e8 00 07 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  4011f0:	49 83 c6 01          	add    $0x1,%r14
  4011f4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  4011fb:	75 e3                	jne    4011e0 <_Z14run_simulationv+0x7f0>
  4011fd:	45 31 ff             	xor    %r15d,%r15d
  401200:	48 c7 04 dd 10 12 65 	movq   $0x0,0x651210(,%rbx,8)
  401207:	00 00 00 00 00 
  40120c:	48 c7 04 dd 10 0c 66 	movq   $0x0,0x660c10(,%rbx,8)
  401213:	00 00 00 00 00 
  401218:	48 c7 04 dd 18 12 65 	movq   $0x0,0x651218(,%rbx,8)
  40121f:	00 00 00 00 00 
  401224:	48 c7 04 dd 18 0c 66 	movq   $0x0,0x660c18(,%rbx,8)
  40122b:	00 00 00 00 00 
  401230:	48 c7 04 dd 20 12 65 	movq   $0x0,0x651220(,%rbx,8)
  401237:	00 00 00 00 00 
  40123c:	48 c7 04 dd 20 0c 66 	movq   $0x0,0x660c20(,%rbx,8)
  401243:	00 00 00 00 00 
  401248:	48 c7 04 dd 28 12 65 	movq   $0x0,0x651228(,%rbx,8)
  40124f:	00 00 00 00 00 
  401254:	48 c7 04 dd 28 0c 66 	movq   $0x0,0x660c28(,%rbx,8)
  40125b:	00 00 00 00 00 
  401260:	48 c7 04 dd 30 12 65 	movq   $0x0,0x651230(,%rbx,8)
  401267:	00 00 00 00 00 
  40126c:	48 c7 04 dd 30 0c 66 	movq   $0x0,0x660c30(,%rbx,8)
  401273:	00 00 00 00 00 
  401278:	48 c7 04 dd 38 12 65 	movq   $0x0,0x651238(,%rbx,8)
  40127f:	00 00 00 00 00 
  401284:	48 c7 04 dd 38 0c 66 	movq   $0x0,0x660c38(,%rbx,8)
  40128b:	00 00 00 00 00 
  401290:	48 c7 04 dd 40 12 65 	movq   $0x0,0x651240(,%rbx,8)
  401297:	00 00 00 00 00 
  40129c:	48 c7 04 dd 40 0c 66 	movq   $0x0,0x660c40(,%rbx,8)
  4012a3:	00 00 00 00 00 
  4012a8:	48 c7 04 dd 48 12 65 	movq   $0x0,0x651248(,%rbx,8)
  4012af:	00 00 00 00 00 
  4012b4:	48 c7 04 dd 48 0c 66 	movq   $0x0,0x660c48(,%rbx,8)
  4012bb:	00 00 00 00 00 
  4012c0:	48 c7 04 dd 50 12 65 	movq   $0x0,0x651250(,%rbx,8)
  4012c7:	00 00 00 00 00 
  4012cc:	48 c7 04 dd 50 0c 66 	movq   $0x0,0x660c50(,%rbx,8)
  4012d3:	00 00 00 00 00 
  4012d8:	48 c7 04 dd 58 12 65 	movq   $0x0,0x651258(,%rbx,8)
  4012df:	00 00 00 00 00 
  4012e4:	48 c7 04 dd 58 0c 66 	movq   $0x0,0x660c58(,%rbx,8)
  4012eb:	00 00 00 00 00 
  4012f0:	48 c7 04 dd 60 12 65 	movq   $0x0,0x651260(,%rbx,8)
  4012f7:	00 00 00 00 00 
  4012fc:	48 c7 04 dd 60 0c 66 	movq   $0x0,0x660c60(,%rbx,8)
  401303:	00 00 00 00 00 
  401308:	48 c7 04 dd 68 12 65 	movq   $0x0,0x651268(,%rbx,8)
  40130f:	00 00 00 00 00 
  401314:	48 c7 04 dd 68 0c 66 	movq   $0x0,0x660c68(,%rbx,8)
  40131b:	00 00 00 00 00 
  401320:	48 c7 04 dd 70 12 65 	movq   $0x0,0x651270(,%rbx,8)
  401327:	00 00 00 00 00 
  40132c:	48 c7 04 dd 70 0c 66 	movq   $0x0,0x660c70(,%rbx,8)
  401333:	00 00 00 00 00 
  401338:	48 c7 04 dd 78 12 65 	movq   $0x0,0x651278(,%rbx,8)
  40133f:	00 00 00 00 00 
  401344:	48 c7 04 dd 78 0c 66 	movq   $0x0,0x660c78(,%rbx,8)
  40134b:	00 00 00 00 00 
  401350:	48 c7 04 dd 80 12 65 	movq   $0x0,0x651280(,%rbx,8)
  401357:	00 00 00 00 00 
  40135c:	48 c7 04 dd 80 0c 66 	movq   $0x0,0x660c80(,%rbx,8)
  401363:	00 00 00 00 00 
  401368:	48 c7 04 dd 88 12 65 	movq   $0x0,0x651288(,%rbx,8)
  40136f:	00 00 00 00 00 
  401374:	48 c7 04 dd 88 0c 66 	movq   $0x0,0x660c88(,%rbx,8)
  40137b:	00 00 00 00 00 
  401380:	48 83 c3 10          	add    $0x10,%rbx
  401384:	48 81 fb 50 1f 00 00 	cmp    $0x1f50,%rbx
  40138b:	0f 85 6f fe ff ff    	jne    401200 <_Z14run_simulationv+0x810>
  401391:	41 be 01 00 00 00    	mov    $0x1,%r14d
  401397:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40139e:	00 00 
  4013a0:	49 83 c7 01          	add    $0x1,%r15
  4013a4:	bb 01 00 00 00       	mov    $0x1,%ebx
  4013a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4013b0:	48 89 df             	mov    %rbx,%rdi
  4013b3:	4c 89 fe             	mov    %r15,%rsi
  4013b6:	e8 a5 06 00 00       	callq  401a60 <_ZN4Body9add_forceEPS_>
  4013bb:	48 83 c3 01          	add    $0x1,%rbx
  4013bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013c6:	75 e8                	jne    4013b0 <_Z14run_simulationv+0x9c0>
  4013c8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4013cf:	75 cf                	jne    4013a0 <_Z14run_simulationv+0x9b0>
  4013d1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4013d8:	0f 1f 84 00 00 00 00 
  4013df:	00 
  4013e0:	4c 89 f7             	mov    %r14,%rdi
  4013e3:	c5 fb 10 05 cd 07 00 	vmovsd 0x7cd(%rip),%xmm0        # 401bb8 <_IO_stdin_used+0x38>
  4013ea:	00 
  4013eb:	e8 00 05 00 00       	callq  4018f0 <_ZN4Body6updateEd>
  4013f0:	49 83 c6 01          	add    $0x1,%r14
  4013f4:	49 81 fe 41 1f 00 00 	cmp    $0x1f41,%r14
  4013fb:	75 e3                	jne    4013e0 <_Z14run_simulationv+0x9f0>
  4013fd:	5b                   	pop    %rbx
  4013fe:	41 5e                	pop    %r14
  401400:	41 5f                	pop    %r15
  401402:	c3                   	retq   
  401403:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40140a:	84 00 00 00 00 00 

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
  401477:	e8 84 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40147c:	48 89 c5             	mov    %rax,%rbp
  40147f:	e8 0c f4 ff ff       	callq  400890 <_Z13instantiationv>
  401484:	e8 77 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401489:	49 89 c6             	mov    %rax,%r14
  40148c:	49 29 ee             	sub    %rbp,%r14
  40148f:	e8 6c f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401494:	48 89 c3             	mov    %rax,%rbx
  401497:	e8 54 f5 ff ff       	callq  4009f0 <_Z14run_simulationv>
  40149c:	e8 5f f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4014a1:	48 89 c5             	mov    %rax,%rbp
  4014a4:	48 29 dd             	sub    %rbx,%rbp
  4014a7:	31 db                	xor    %ebx,%ebx
  4014a9:	e8 52 f2 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
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
  4014e0:	c5 fb 59 0c cd 88 12 	vmulsd 0x651288(,%rcx,8),%xmm0,%xmm1
  4014e7:	65 00 
  4014e9:	c5 f3 5e 0c c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm1,%xmm1
  4014f0:	60 00 
  4014f2:	c5 f3 58 0c cd 88 1e 	vaddsd 0x631e88(,%rcx,8),%xmm1,%xmm1
  4014f9:	63 00 
  4014fb:	c5 fb 11 0c cd 88 1e 	vmovsd %xmm1,0x631e88(,%rcx,8)
  401502:	63 00 
  401504:	c5 fb 59 14 cd 88 0c 	vmulsd 0x660c88(,%rcx,8),%xmm0,%xmm2
  40150b:	66 00 
  40150d:	c5 eb 5e 14 c5 90 30 	vdivsd 0x603090(,%rax,8),%xmm2,%xmm2
  401514:	60 00 
  401516:	c5 eb 58 14 cd 88 18 	vaddsd 0x641888(,%rcx,8),%xmm2,%xmm2
  40151d:	64 00 
  40151f:	c5 fb 11 14 cd 88 18 	vmovsd %xmm2,0x641888(,%rcx,8)
  401526:	64 00 
  401528:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x612a88(,%rcx,8),%xmm0,%xmm1
  40152f:	2a 61 00 
  401532:	c5 fb 11 0c cd 88 2a 	vmovsd %xmm1,0x612a88(,%rcx,8)
  401539:	61 00 
  40153b:	c5 fb 10 0c cd 88 18 	vmovsd 0x641888(,%rcx,8),%xmm1
  401542:	64 00 
  401544:	c4 e2 f9 a9 0c cd 88 	vfmadd213sd 0x622488(,%rcx,8),%xmm0,%xmm1
  40154b:	24 62 00 
  40154e:	83 c2 ff             	add    $0xffffffff,%edx
  401551:	c5 fb 11 0c cd 88 24 	vmovsd %xmm1,0x622488(,%rcx,8)
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
  4015b3:	e8 48 f1 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4015b8:	49 89 c4             	mov    %rax,%r12
  4015bb:	4d 29 fc             	sub    %r15,%r12
  4015be:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4015c3:	31 f6                	xor    %esi,%esi
  4015c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4015cc:	00 00 00 00 
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
  40161c:	73 0c                	jae    40162a <main+0x1ca>
  40161e:	bf f0 1b 40 00       	mov    $0x401bf0,%edi
  401623:	b0 02                	mov    $0x2,%al
  401625:	e8 76 f0 ff ff       	callq  4006a0 <printf@plt>
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
  401752:	c5 f3 58 14 fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm2
  401759:	63 00 
  40175b:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  401762:	63 00 
  401764:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40176b:	66 00 
  40176d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401774:	60 00 
  401776:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40177d:	64 00 
  40177f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401786:	64 00 
  401788:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  40178f:	2a 61 00 
  401792:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  401799:	61 00 
  40179b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4017a2:	64 00 
  4017a4:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4017ab:	24 62 00 
  4017ae:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4017b5:	62 00 
  4017b7:	c5 f9 2e 15 d9 03 00 	vucomisd 0x3d9(%rip),%xmm2        # 401b98 <_IO_stdin_used+0x18>
  4017be:	00 
  4017bf:	72 0a                	jb     4017cb <_Z14codegen_updateP4Bodyd+0x8b>
  4017c1:	c5 f9 2e 15 f7 03 00 	vucomisd 0x3f7(%rip),%xmm2        # 401bc0 <_IO_stdin_used+0x40>
  4017c8:	00 
  4017c9:	76 23                	jbe    4017ee <_Z14codegen_updateP4Bodyd+0xae>
  4017cb:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4017d2:	63 00 
  4017d4:	c5 f9 57 05 04 04 00 	vxorpd 0x404(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  4017db:	00 
  4017dc:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4017e3:	63 00 
  4017e5:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4017ec:	62 00 
  4017ee:	c5 f9 2e 0d a2 03 00 	vucomisd 0x3a2(%rip),%xmm1        # 401b98 <_IO_stdin_used+0x18>
  4017f5:	00 
  4017f6:	72 0a                	jb     401802 <_Z14codegen_updateP4Bodyd+0xc2>
  4017f8:	c5 f9 2e 0d c0 03 00 	vucomisd 0x3c0(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  4017ff:	00 
  401800:	76 1a                	jbe    40181c <_Z14codegen_updateP4Bodyd+0xdc>
  401802:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  401809:	64 00 
  40180b:	c5 f9 57 05 cd 03 00 	vxorpd 0x3cd(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  401812:	00 
  401813:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  40181a:	64 00 
  40181c:	c3                   	retq   
  40181d:	0f 1f 00             	nopl   (%rax)

0000000000401820 <_Z26codegen_reset_force_manualP4Body>:
  401820:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401827:	00 00 00 00 00 
  40182c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401833:	00 00 00 00 00 
  401838:	c3                   	retq   
  401839:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401840 <_Z24codegen_reset_force_halfP4Body>:
  401840:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
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
  401870:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401877:	65 00 
  401879:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401880:	60 00 
  401882:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401889:	63 00 
  40188b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401892:	63 00 
  401894:	c5 fb 59 14 fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm2
  40189b:	66 00 
  40189d:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  4018a4:	60 00 
  4018a6:	c5 eb 58 14 fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm2,%xmm2
  4018ad:	64 00 
  4018af:	c5 fb 11 14 fd 88 18 	vmovsd %xmm2,0x641888(,%rdi,8)
  4018b6:	64 00 
  4018b8:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4018bf:	2a 61 00 
  4018c2:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4018c9:	61 00 
  4018cb:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4018d2:	64 00 
  4018d4:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4018db:	24 62 00 
  4018de:	83 c0 ff             	add    $0xffffffff,%eax
  4018e1:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4018e8:	62 00 
  4018ea:	75 84                	jne    401870 <_Z22codengen_simple_updateP4Bodyd+0x10>
  4018ec:	c3                   	retq   
  4018ed:	0f 1f 00             	nopl   (%rax)

00000000004018f0 <_ZN4Body6updateEd>:
  4018f0:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  4018f7:	65 00 
  4018f9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401900:	60 00 
  401902:	c5 f3 58 14 fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm2
  401909:	63 00 
  40190b:	c5 fb 11 14 fd 88 1e 	vmovsd %xmm2,0x631e88(,%rdi,8)
  401912:	63 00 
  401914:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40191b:	66 00 
  40191d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401924:	60 00 
  401926:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40192d:	64 00 
  40192f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401936:	64 00 
  401938:	c4 e2 f9 a9 14 fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm2
  40193f:	2a 61 00 
  401942:	c5 fb 11 14 fd 88 2a 	vmovsd %xmm2,0x612a88(,%rdi,8)
  401949:	61 00 
  40194b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401952:	64 00 
  401954:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  40195b:	24 62 00 
  40195e:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401965:	62 00 
  401967:	c5 f9 2e 15 29 02 00 	vucomisd 0x229(%rip),%xmm2        # 401b98 <_IO_stdin_used+0x18>
  40196e:	00 
  40196f:	72 0a                	jb     40197b <_ZN4Body6updateEd+0x8b>
  401971:	c5 f9 2e 15 47 02 00 	vucomisd 0x247(%rip),%xmm2        # 401bc0 <_IO_stdin_used+0x40>
  401978:	00 
  401979:	76 23                	jbe    40199e <_ZN4Body6updateEd+0xae>
  40197b:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  401982:	63 00 
  401984:	c5 f9 57 05 54 02 00 	vxorpd 0x254(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  40198b:	00 
  40198c:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  401993:	63 00 
  401995:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  40199c:	62 00 
  40199e:	c5 f9 2e 0d f2 01 00 	vucomisd 0x1f2(%rip),%xmm1        # 401b98 <_IO_stdin_used+0x18>
  4019a5:	00 
  4019a6:	72 0a                	jb     4019b2 <_ZN4Body6updateEd+0xc2>
  4019a8:	c5 f9 2e 0d 10 02 00 	vucomisd 0x210(%rip),%xmm1        # 401bc0 <_IO_stdin_used+0x40>
  4019af:	00 
  4019b0:	76 1a                	jbe    4019cc <_ZN4Body6updateEd+0xdc>
  4019b2:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  4019b9:	64 00 
  4019bb:	c5 f9 57 05 1d 02 00 	vxorpd 0x21d(%rip),%xmm0,%xmm0        # 401be0 <_IO_stdin_used+0x60>
  4019c2:	00 
  4019c3:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  4019ca:	64 00 
  4019cc:	c3                   	retq   
  4019cd:	0f 1f 00             	nopl   (%rax)

00000000004019d0 <_ZN4Body22codengen_simple_updateEd>:
  4019d0:	b8 64 00 00 00       	mov    $0x64,%eax
  4019d5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4019dc:	00 00 00 00 
  4019e0:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  4019e7:	65 00 
  4019e9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4019f0:	60 00 
  4019f2:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  4019f9:	63 00 
  4019fb:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401a02:	63 00 
  401a04:	c5 fb 59 14 fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm2
  401a0b:	66 00 
  401a0d:	c5 eb 5e 14 fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm2,%xmm2
  401a14:	60 00 
  401a16:	c5 eb 58 14 fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm2,%xmm2
  401a1d:	64 00 
  401a1f:	c5 fb 11 14 fd 88 18 	vmovsd %xmm2,0x641888(,%rdi,8)
  401a26:	64 00 
  401a28:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401a2f:	2a 61 00 
  401a32:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401a39:	61 00 
  401a3b:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401a42:	64 00 
  401a44:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401a4b:	24 62 00 
  401a4e:	83 c0 ff             	add    $0xffffffff,%eax
  401a51:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401a58:	62 00 
  401a5a:	75 84                	jne    4019e0 <_ZN4Body22codengen_simple_updateEd+0x10>
  401a5c:	c3                   	retq   
  401a5d:	0f 1f 00             	nopl   (%rax)

0000000000401a60 <_ZN4Body9add_forceEPS_>:
  401a60:	48 39 f7             	cmp    %rsi,%rdi
  401a63:	0f 84 91 00 00 00    	je     401afa <_ZN4Body9add_forceEPS_+0x9a>
  401a69:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  401a70:	61 00 
  401a72:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  401a79:	61 00 
  401a7b:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  401a82:	62 00 
  401a84:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  401a8b:	62 00 
  401a8d:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401a91:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401a96:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
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
  401ad0:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401ad7:	12 65 00 
  401ada:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401ae1:	65 00 
  401ae3:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401ae7:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  401aee:	0c 66 00 
  401af1:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
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
  401b2c:	e8 3f eb ff ff       	callq  400670 <_init>
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
