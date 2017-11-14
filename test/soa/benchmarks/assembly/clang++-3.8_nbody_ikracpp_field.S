
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
  40077f:	49 c7 c0 e0 1b 40 00 	mov    $0x401be0,%r8
  400786:	48 c7 c1 70 1b 40 00 	mov    $0x401b70,%rcx
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
  4008b9:	c5 fb 10 05 3f 13 00 	vmovsd 0x133f(%rip),%xmm0        # 401c00 <_IO_stdin_used+0x10>
  4008c0:	00 
  4008c1:	c4 e2 f9 99 0d 2e 13 	vfmadd132sd 0x132e(%rip),%xmm0,%xmm1        # 401bf8 <_IO_stdin_used+0x8>
  4008c8:	00 00 
  4008ca:	c5 fb 11 4c 24 28    	vmovsd %xmm1,0x28(%rsp)
  4008d0:	e8 3b fe ff ff       	callq  400710 <rand@plt>
  4008d5:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  4008d9:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  4008dd:	c5 fb 10 05 23 13 00 	vmovsd 0x1323(%rip),%xmm0        # 401c08 <_IO_stdin_used+0x18>
  4008e4:	00 
  4008e5:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  4008e9:	c5 fb 10 05 1f 13 00 	vmovsd 0x131f(%rip),%xmm0        # 401c10 <_IO_stdin_used+0x20>
  4008f0:	00 
  4008f1:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  4008f6:	c5 fb 11 4c 24 20    	vmovsd %xmm1,0x20(%rsp)
  4008fc:	e8 0f fe ff ff       	callq  400710 <rand@plt>
  400901:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400905:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400909:	c5 fb 10 0d ff 12 00 	vmovsd 0x12ff(%rip),%xmm1        # 401c10 <_IO_stdin_used+0x20>
  400910:	00 
  400911:	c4 e2 f1 a9 05 ee 12 	vfmadd213sd 0x12ee(%rip),%xmm1,%xmm0        # 401c08 <_IO_stdin_used+0x18>
  400918:	00 00 
  40091a:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
  400920:	e8 eb fd ff ff       	callq  400710 <rand@plt>
  400925:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400929:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  40092d:	c5 fb 10 05 e3 12 00 	vmovsd 0x12e3(%rip),%xmm0        # 401c18 <_IO_stdin_used+0x28>
  400934:	00 
  400935:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400939:	c5 fb 10 05 df 12 00 	vmovsd 0x12df(%rip),%xmm0        # 401c20 <_IO_stdin_used+0x30>
  400940:	00 
  400941:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400946:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  40094c:	e8 bf fd ff ff       	callq  400710 <rand@plt>
  400951:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400955:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400959:	c5 fb 10 0d bf 12 00 	vmovsd 0x12bf(%rip),%xmm1        # 401c20 <_IO_stdin_used+0x30>
  400960:	00 
  400961:	c4 e2 f1 a9 05 ae 12 	vfmadd213sd 0x12ae(%rip),%xmm1,%xmm0        # 401c18 <_IO_stdin_used+0x28>
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
  400bb6:	e8 15 0f 00 00       	callq  401ad0 <_ZN4Body9add_forceEPS_>
  400bbb:	48 83 c3 01          	add    $0x1,%rbx
  400bbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400bc6:	75 e8                	jne    400bb0 <_Z14run_simulationv+0x1c0>
  400bc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400bcf:	75 cf                	jne    400ba0 <_Z14run_simulationv+0x1b0>
  400bd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 
  400be0:	4c 89 f7             	mov    %r14,%rdi
  400be3:	c5 fb 10 05 3d 10 00 	vmovsd 0x103d(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  400bea:	00 
  400beb:	e8 40 0d 00 00       	callq  401930 <_ZN4Body6updateEd>
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
  400db6:	e8 15 0d 00 00       	callq  401ad0 <_ZN4Body9add_forceEPS_>
  400dbb:	48 83 c3 01          	add    $0x1,%rbx
  400dbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400dc6:	75 e8                	jne    400db0 <_Z14run_simulationv+0x3c0>
  400dc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400dcf:	75 cf                	jne    400da0 <_Z14run_simulationv+0x3b0>
  400dd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400dd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddd:	00 00 00 
  400de0:	4c 89 f7             	mov    %r14,%rdi
  400de3:	c5 fb 10 05 3d 0e 00 	vmovsd 0xe3d(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  400dea:	00 
  400deb:	e8 40 0b 00 00       	callq  401930 <_ZN4Body6updateEd>
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
  400fb6:	e8 15 0b 00 00       	callq  401ad0 <_ZN4Body9add_forceEPS_>
  400fbb:	48 83 c3 01          	add    $0x1,%rbx
  400fbf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  400fc6:	75 e8                	jne    400fb0 <_Z14run_simulationv+0x5c0>
  400fc8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  400fcf:	75 cf                	jne    400fa0 <_Z14run_simulationv+0x5b0>
  400fd1:	bb 10 00 00 00       	mov    $0x10,%ebx
  400fd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400fdd:	00 00 00 
  400fe0:	4c 89 f7             	mov    %r14,%rdi
  400fe3:	c5 fb 10 05 3d 0c 00 	vmovsd 0xc3d(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  400fea:	00 
  400feb:	e8 40 09 00 00       	callq  401930 <_ZN4Body6updateEd>
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
  4011b6:	e8 15 09 00 00       	callq  401ad0 <_ZN4Body9add_forceEPS_>
  4011bb:	48 83 c3 01          	add    $0x1,%rbx
  4011bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4011c6:	75 e8                	jne    4011b0 <_Z14run_simulationv+0x7c0>
  4011c8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4011cf:	75 cf                	jne    4011a0 <_Z14run_simulationv+0x7b0>
  4011d1:	bb 10 00 00 00       	mov    $0x10,%ebx
  4011d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011dd:	00 00 00 
  4011e0:	4c 89 f7             	mov    %r14,%rdi
  4011e3:	c5 fb 10 05 3d 0a 00 	vmovsd 0xa3d(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  4011ea:	00 
  4011eb:	e8 40 07 00 00       	callq  401930 <_ZN4Body6updateEd>
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
  4013b6:	e8 15 07 00 00       	callq  401ad0 <_ZN4Body9add_forceEPS_>
  4013bb:	48 83 c3 01          	add    $0x1,%rbx
  4013bf:	48 81 fb 41 1f 00 00 	cmp    $0x1f41,%rbx
  4013c6:	75 e8                	jne    4013b0 <_Z14run_simulationv+0x9c0>
  4013c8:	49 81 ff 40 1f 00 00 	cmp    $0x1f40,%r15
  4013cf:	75 cf                	jne    4013a0 <_Z14run_simulationv+0x9b0>
  4013d1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4013d8:	0f 1f 84 00 00 00 00 
  4013df:	00 
  4013e0:	4c 89 f7             	mov    %r14,%rdi
  4013e3:	c5 fb 10 05 3d 08 00 	vmovsd 0x83d(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  4013ea:	00 
  4013eb:	e8 40 05 00 00       	callq  401930 <_ZN4Body6updateEd>
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
  401433:	c5 fb 10 05 ed 07 00 	vmovsd 0x7ed(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
  40143a:	00 
  40143b:	e8 f0 05 00 00       	callq  401a30 <_ZN4Body22codengen_simple_updateEd>
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
  4014b1:	c5 fb 10 05 6f 07 00 	vmovsd 0x76f(%rip),%xmm0        # 401c28 <_IO_stdin_used+0x38>
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
  4015c0:	e8 3b f1 ff ff       	callq  400700 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
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
  401628:	48 83 fe 0a          	cmp    $0xa,%rsi
  40162c:	73 0c                	jae    40163a <main+0x1da>
  40162e:	bf 60 1c 40 00       	mov    $0x401c60,%edi
  401633:	b0 02                	mov    $0x2,%al
  401635:	e8 66 f0 ff ff       	callq  4006a0 <printf@plt>
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
  401668:	bf 75 1c 40 00       	mov    $0x401c75,%edi
  40166d:	31 c0                	xor    %eax,%eax
  40166f:	4c 89 f6             	mov    %r14,%rsi
  401672:	4c 89 ea             	mov    %r13,%rdx
  401675:	41 89 e8             	mov    %ebp,%r8d
  401678:	e8 23 f0 ff ff       	callq  4006a0 <printf@plt>
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
  4016e6:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  4016ea:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  4016f1:	60 00 
  4016f3:	c5 db 59 25 3d 05 00 	vmulsd 0x53d(%rip),%xmm4,%xmm4        # 401c38 <_IO_stdin_used+0x48>
  4016fa:	00 
  4016fb:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401702:	60 00 
  401704:	c5 eb 58 15 34 05 00 	vaddsd 0x534(%rip),%xmm2,%xmm2        # 401c40 <_IO_stdin_used+0x50>
  40170b:	00 
  40170c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401710:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401714:	c5 fb 10 25 14 05 00 	vmovsd 0x514(%rip),%xmm4        # 401c30 <_IO_stdin_used+0x40>
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
  401750:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401757:	65 00 
  401759:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401760:	60 00 
  401762:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401769:	63 00 
  40176b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401772:	63 00 
  401774:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40177b:	66 00 
  40177d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401784:	60 00 
  401786:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40178d:	64 00 
  40178f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401796:	64 00 
  401798:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  40179f:	63 00 
  4017a1:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4017a8:	2a 61 00 
  4017ab:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  4017b2:	61 00 
  4017b4:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  4017bb:	64 00 
  4017bd:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4017c4:	24 62 00 
  4017c7:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4017ce:	62 00 
  4017d0:	c5 fb 10 04 fd 88 2a 	vmovsd 0x612a88(,%rdi,8),%xmm0
  4017d7:	61 00 
  4017d9:	c5 fb 10 15 27 04 00 	vmovsd 0x427(%rip),%xmm2        # 401c08 <_IO_stdin_used+0x18>
  4017e0:	00 
  4017e1:	c5 f9 2e d0          	vucomisd %xmm0,%xmm2
  4017e5:	77 0a                	ja     4017f1 <_Z14codegen_updateP4Bodyd+0xa1>
  4017e7:	c5 f9 2e 05 41 04 00 	vucomisd 0x441(%rip),%xmm0        # 401c30 <_IO_stdin_used+0x40>
  4017ee:	00 
  4017ef:	76 23                	jbe    401814 <_Z14codegen_updateP4Bodyd+0xc4>
  4017f1:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4017f8:	63 00 
  4017fa:	c5 f9 57 05 4e 04 00 	vxorpd 0x44e(%rip),%xmm0,%xmm0        # 401c50 <_IO_stdin_used+0x60>
  401801:	00 
  401802:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  401809:	63 00 
  40180b:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  401812:	62 00 
  401814:	c5 f9 2e 0d ec 03 00 	vucomisd 0x3ec(%rip),%xmm1        # 401c08 <_IO_stdin_used+0x18>
  40181b:	00 
  40181c:	72 0a                	jb     401828 <_Z14codegen_updateP4Bodyd+0xd8>
  40181e:	c5 f9 2e 0d 0a 04 00 	vucomisd 0x40a(%rip),%xmm1        # 401c30 <_IO_stdin_used+0x40>
  401825:	00 
  401826:	76 1a                	jbe    401842 <_Z14codegen_updateP4Bodyd+0xf2>
  401828:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  40182f:	64 00 
  401831:	c5 f9 57 05 17 04 00 	vxorpd 0x417(%rip),%xmm0,%xmm0        # 401c50 <_IO_stdin_used+0x60>
  401838:	00 
  401839:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  401840:	64 00 
  401842:	c3                   	retq   
  401843:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40184a:	84 00 00 00 00 00 

0000000000401850 <_Z26codegen_reset_force_manualP4Body>:
  401850:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401857:	00 00 00 00 00 
  40185c:	48 c7 04 fd 88 0c 66 	movq   $0x0,0x660c88(,%rdi,8)
  401863:	00 00 00 00 00 
  401868:	c3                   	retq   
  401869:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401870 <_Z24codegen_reset_force_halfP4Body>:
  401870:	48 c7 04 fd 88 12 65 	movq   $0x0,0x651288(,%rdi,8)
  401877:	00 00 00 00 00 
  40187c:	c3                   	retq   
  40187d:	0f 1f 00             	nopl   (%rax)

0000000000401880 <_Z18codegen_reset_massP4Body>:
  401880:	48 c7 04 fd 88 30 60 	movq   $0x0,0x603088(,%rdi,8)
  401887:	00 00 00 00 00 
  40188c:	c3                   	retq   
  40188d:	0f 1f 00             	nopl   (%rax)

0000000000401890 <_Z22codengen_simple_updateP4Bodyd>:
  401890:	b8 64 00 00 00       	mov    $0x64,%eax
  401895:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40189c:	00 00 00 00 
  4018a0:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  4018a7:	65 00 
  4018a9:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018b0:	60 00 
  4018b2:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  4018b9:	63 00 
  4018bb:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  4018c2:	63 00 
  4018c4:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  4018cb:	66 00 
  4018cd:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  4018d4:	60 00 
  4018d6:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  4018dd:	64 00 
  4018df:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  4018e6:	64 00 
  4018e8:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  4018ef:	63 00 
  4018f1:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  4018f8:	2a 61 00 
  4018fb:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401902:	61 00 
  401904:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  40190b:	64 00 
  40190d:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401914:	24 62 00 
  401917:	83 c0 ff             	add    $0xffffffff,%eax
  40191a:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401921:	62 00 
  401923:	0f 85 77 ff ff ff    	jne    4018a0 <_Z22codengen_simple_updateP4Bodyd+0x10>
  401929:	c3                   	retq   
  40192a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401930 <_ZN4Body6updateEd>:
  401930:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401937:	65 00 
  401939:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401940:	60 00 
  401942:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401949:	63 00 
  40194b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401952:	63 00 
  401954:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  40195b:	66 00 
  40195d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401964:	60 00 
  401966:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  40196d:	64 00 
  40196f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401976:	64 00 
  401978:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  40197f:	63 00 
  401981:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401988:	2a 61 00 
  40198b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401992:	61 00 
  401994:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  40199b:	64 00 
  40199d:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  4019a4:	24 62 00 
  4019a7:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  4019ae:	62 00 
  4019b0:	c5 fb 10 04 fd 88 2a 	vmovsd 0x612a88(,%rdi,8),%xmm0
  4019b7:	61 00 
  4019b9:	c5 fb 10 15 47 02 00 	vmovsd 0x247(%rip),%xmm2        # 401c08 <_IO_stdin_used+0x18>
  4019c0:	00 
  4019c1:	c5 f9 2e d0          	vucomisd %xmm0,%xmm2
  4019c5:	77 0a                	ja     4019d1 <_ZN4Body6updateEd+0xa1>
  4019c7:	c5 f9 2e 05 61 02 00 	vucomisd 0x261(%rip),%xmm0        # 401c30 <_IO_stdin_used+0x40>
  4019ce:	00 
  4019cf:	76 23                	jbe    4019f4 <_ZN4Body6updateEd+0xc4>
  4019d1:	c5 fb 10 04 fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm0
  4019d8:	63 00 
  4019da:	c5 f9 57 05 6e 02 00 	vxorpd 0x26e(%rip),%xmm0,%xmm0        # 401c50 <_IO_stdin_used+0x60>
  4019e1:	00 
  4019e2:	c5 f9 13 04 fd 88 1e 	vmovlpd %xmm0,0x631e88(,%rdi,8)
  4019e9:	63 00 
  4019eb:	c5 fb 10 0c fd 88 24 	vmovsd 0x622488(,%rdi,8),%xmm1
  4019f2:	62 00 
  4019f4:	c5 f9 2e 0d 0c 02 00 	vucomisd 0x20c(%rip),%xmm1        # 401c08 <_IO_stdin_used+0x18>
  4019fb:	00 
  4019fc:	72 0a                	jb     401a08 <_ZN4Body6updateEd+0xd8>
  4019fe:	c5 f9 2e 0d 2a 02 00 	vucomisd 0x22a(%rip),%xmm1        # 401c30 <_IO_stdin_used+0x40>
  401a05:	00 
  401a06:	76 1a                	jbe    401a22 <_ZN4Body6updateEd+0xf2>
  401a08:	c5 fb 10 04 fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm0
  401a0f:	64 00 
  401a11:	c5 f9 57 05 37 02 00 	vxorpd 0x237(%rip),%xmm0,%xmm0        # 401c50 <_IO_stdin_used+0x60>
  401a18:	00 
  401a19:	c5 f9 13 04 fd 88 18 	vmovlpd %xmm0,0x641888(,%rdi,8)
  401a20:	64 00 
  401a22:	c3                   	retq   
  401a23:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401a2a:	00 00 00 
  401a2d:	0f 1f 00             	nopl   (%rax)

0000000000401a30 <_ZN4Body22codengen_simple_updateEd>:
  401a30:	b8 64 00 00 00       	mov    $0x64,%eax
  401a35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  401a3c:	00 00 00 00 
  401a40:	c5 fb 59 0c fd 88 12 	vmulsd 0x651288(,%rdi,8),%xmm0,%xmm1
  401a47:	65 00 
  401a49:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a50:	60 00 
  401a52:	c5 f3 58 0c fd 88 1e 	vaddsd 0x631e88(,%rdi,8),%xmm1,%xmm1
  401a59:	63 00 
  401a5b:	c5 fb 11 0c fd 88 1e 	vmovsd %xmm1,0x631e88(,%rdi,8)
  401a62:	63 00 
  401a64:	c5 fb 59 0c fd 88 0c 	vmulsd 0x660c88(,%rdi,8),%xmm0,%xmm1
  401a6b:	66 00 
  401a6d:	c5 f3 5e 0c fd 88 30 	vdivsd 0x603088(,%rdi,8),%xmm1,%xmm1
  401a74:	60 00 
  401a76:	c5 f3 58 0c fd 88 18 	vaddsd 0x641888(,%rdi,8),%xmm1,%xmm1
  401a7d:	64 00 
  401a7f:	c5 fb 11 0c fd 88 18 	vmovsd %xmm1,0x641888(,%rdi,8)
  401a86:	64 00 
  401a88:	c5 fb 10 0c fd 88 1e 	vmovsd 0x631e88(,%rdi,8),%xmm1
  401a8f:	63 00 
  401a91:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x612a88(,%rdi,8),%xmm0,%xmm1
  401a98:	2a 61 00 
  401a9b:	c5 fb 11 0c fd 88 2a 	vmovsd %xmm1,0x612a88(,%rdi,8)
  401aa2:	61 00 
  401aa4:	c5 fb 10 0c fd 88 18 	vmovsd 0x641888(,%rdi,8),%xmm1
  401aab:	64 00 
  401aad:	c4 e2 f9 a9 0c fd 88 	vfmadd213sd 0x622488(,%rdi,8),%xmm0,%xmm1
  401ab4:	24 62 00 
  401ab7:	83 c0 ff             	add    $0xffffffff,%eax
  401aba:	c5 fb 11 0c fd 88 24 	vmovsd %xmm1,0x622488(,%rdi,8)
  401ac1:	62 00 
  401ac3:	0f 85 77 ff ff ff    	jne    401a40 <_ZN4Body22codengen_simple_updateEd+0x10>
  401ac9:	c3                   	retq   
  401aca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401ad0 <_ZN4Body9add_forceEPS_>:
  401ad0:	48 39 f7             	cmp    %rsi,%rdi
  401ad3:	0f 84 91 00 00 00    	je     401b6a <_ZN4Body9add_forceEPS_+0x9a>
  401ad9:	c5 fb 10 04 f5 88 2a 	vmovsd 0x612a88(,%rsi,8),%xmm0
  401ae0:	61 00 
  401ae2:	c5 fb 5c 04 fd 88 2a 	vsubsd 0x612a88(,%rdi,8),%xmm0,%xmm0
  401ae9:	61 00 
  401aeb:	c5 fb 10 0c f5 88 24 	vmovsd 0x622488(,%rsi,8),%xmm1
  401af2:	62 00 
  401af4:	c5 f3 5c 0c fd 88 24 	vsubsd 0x622488(,%rdi,8),%xmm1,%xmm1
  401afb:	62 00 
  401afd:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  401b01:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  401b06:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  401b0a:	c5 fb 10 24 fd 88 30 	vmovsd 0x603088(,%rdi,8),%xmm4
  401b11:	60 00 
  401b13:	c5 db 59 25 1d 01 00 	vmulsd 0x11d(%rip),%xmm4,%xmm4        # 401c38 <_IO_stdin_used+0x48>
  401b1a:	00 
  401b1b:	c5 db 59 24 f5 88 30 	vmulsd 0x603088(,%rsi,8),%xmm4,%xmm4
  401b22:	60 00 
  401b24:	c5 eb 58 15 14 01 00 	vaddsd 0x114(%rip),%xmm2,%xmm2        # 401c40 <_IO_stdin_used+0x50>
  401b2b:	00 
  401b2c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  401b30:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  401b34:	c5 fb 10 25 f4 00 00 	vmovsd 0xf4(%rip),%xmm4        # 401c30 <_IO_stdin_used+0x40>
  401b3b:	00 
  401b3c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  401b40:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x651288(,%rdi,8),%xmm3,%xmm0
  401b47:	12 65 00 
  401b4a:	c5 fb 11 04 fd 88 12 	vmovsd %xmm0,0x651288(,%rdi,8)
  401b51:	65 00 
  401b53:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  401b57:	c4 e2 e1 a9 04 fd 88 	vfmadd213sd 0x660c88(,%rdi,8),%xmm3,%xmm0
  401b5e:	0c 66 00 
  401b61:	c5 fb 11 04 fd 88 0c 	vmovsd %xmm0,0x660c88(,%rdi,8)
  401b68:	66 00 
  401b6a:	c3                   	retq   
  401b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000401b70 <__libc_csu_init>:
  401b70:	41 57                	push   %r15
  401b72:	41 56                	push   %r14
  401b74:	41 89 ff             	mov    %edi,%r15d
  401b77:	41 55                	push   %r13
  401b79:	41 54                	push   %r12
  401b7b:	4c 8d 25 4e 12 20 00 	lea    0x20124e(%rip),%r12        # 602dd0 <__frame_dummy_init_array_entry>
  401b82:	55                   	push   %rbp
  401b83:	48 8d 2d 5e 12 20 00 	lea    0x20125e(%rip),%rbp        # 602de8 <__init_array_end>
  401b8a:	53                   	push   %rbx
  401b8b:	49 89 f6             	mov    %rsi,%r14
  401b8e:	49 89 d5             	mov    %rdx,%r13
  401b91:	4c 29 e5             	sub    %r12,%rbp
  401b94:	48 83 ec 08          	sub    $0x8,%rsp
  401b98:	48 c1 fd 03          	sar    $0x3,%rbp
  401b9c:	e8 cf ea ff ff       	callq  400670 <_init>
  401ba1:	48 85 ed             	test   %rbp,%rbp
  401ba4:	74 20                	je     401bc6 <__libc_csu_init+0x56>
  401ba6:	31 db                	xor    %ebx,%ebx
  401ba8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  401baf:	00 
  401bb0:	4c 89 ea             	mov    %r13,%rdx
  401bb3:	4c 89 f6             	mov    %r14,%rsi
  401bb6:	44 89 ff             	mov    %r15d,%edi
  401bb9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  401bbd:	48 83 c3 01          	add    $0x1,%rbx
  401bc1:	48 39 eb             	cmp    %rbp,%rbx
  401bc4:	75 ea                	jne    401bb0 <__libc_csu_init+0x40>
  401bc6:	48 83 c4 08          	add    $0x8,%rsp
  401bca:	5b                   	pop    %rbx
  401bcb:	5d                   	pop    %rbp
  401bcc:	41 5c                	pop    %r12
  401bce:	41 5d                	pop    %r13
  401bd0:	41 5e                	pop    %r14
  401bd2:	41 5f                	pop    %r15
  401bd4:	c3                   	retq   
  401bd5:	90                   	nop
  401bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401bdd:	00 00 00 

0000000000401be0 <__libc_csu_fini>:
  401be0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401be4 <_fini>:
  401be4:	48 83 ec 08          	sub    $0x8,%rsp
  401be8:	48 83 c4 08          	add    $0x8,%rsp
  401bec:	c3                   	retq   
