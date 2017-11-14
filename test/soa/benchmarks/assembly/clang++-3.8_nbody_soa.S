
bin/clang++-3.8_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004006a8 <_init>:
  4006a8:	48 83 ec 08          	sub    $0x8,%rsp
  4006ac:	48 8b 05 45 19 20 00 	mov    0x201945(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4006b3:	48 85 c0             	test   %rax,%rax
  4006b6:	74 05                	je     4006bd <_init+0x15>
  4006b8:	e8 b3 00 00 00       	callq  400770 <rand@plt+0x10>
  4006bd:	48 83 c4 08          	add    $0x8,%rsp
  4006c1:	c3                   	retq   

Disassembly of section .plt:

00000000004006d0 <printf@plt-0x10>:
  4006d0:	ff 35 32 19 20 00    	pushq  0x201932(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006d6:	ff 25 34 19 20 00    	jmpq   *0x201934(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006e0 <printf@plt>:
  4006e0:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006e6:	68 00 00 00 00       	pushq  $0x0
  4006eb:	e9 e0 ff ff ff       	jmpq   4006d0 <_init+0x28>

00000000004006f0 <memset@plt>:
  4006f0:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006f6:	68 01 00 00 00       	pushq  $0x1
  4006fb:	e9 d0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400700 <_ZNSt8ios_base4InitC1Ev@plt>:
  400700:	ff 25 22 19 20 00    	jmpq   *0x201922(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400706:	68 02 00 00 00       	pushq  $0x2
  40070b:	e9 c0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400710 <__libc_start_main@plt>:
  400710:	ff 25 1a 19 20 00    	jmpq   *0x20191a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400716:	68 03 00 00 00       	pushq  $0x3
  40071b:	e9 b0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400720 <__cxa_atexit@plt>:
  400720:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400726:	68 04 00 00 00       	pushq  $0x4
  40072b:	e9 a0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400730 <_ZNSt8ios_base4InitD1Ev@plt>:
  400730:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400736:	68 05 00 00 00       	pushq  $0x5
  40073b:	e9 90 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400740 <srand@plt>:
  400740:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400746:	68 06 00 00 00       	pushq  $0x6
  40074b:	e9 80 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400750:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400756:	68 07 00 00 00       	pushq  $0x7
  40075b:	e9 70 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400760 <rand@plt>:
  400760:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400766:	68 08 00 00 00       	pushq  $0x8
  40076b:	e9 60 ff ff ff       	jmpq   4006d0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400770 <.plt.got>:
  400770:	ff 25 82 18 20 00    	jmpq   *0x201882(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400776:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400780 <set_fast_math>:
  400780:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  400785:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  40078c:	00 
  40078d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  400792:	c3                   	retq   
  400793:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40079a:	00 00 00 
  40079d:	0f 1f 00             	nopl   (%rax)

00000000004007a0 <_GLOBAL__sub_I_soa.cc>:
  4007a0:	50                   	push   %rax
  4007a1:	bf 80 20 60 00       	mov    $0x602080,%edi
  4007a6:	e8 55 ff ff ff       	callq  400700 <_ZNSt8ios_base4InitC1Ev@plt>
  4007ab:	bf 30 07 40 00       	mov    $0x400730,%edi
  4007b0:	be 80 20 60 00       	mov    $0x602080,%esi
  4007b5:	ba 68 20 60 00       	mov    $0x602068,%edx
  4007ba:	58                   	pop    %rax
  4007bb:	e9 60 ff ff ff       	jmpq   400720 <__cxa_atexit@plt>

00000000004007c0 <_start>:
  4007c0:	31 ed                	xor    %ebp,%ebp
  4007c2:	49 89 d1             	mov    %rdx,%r9
  4007c5:	5e                   	pop    %rsi
  4007c6:	48 89 e2             	mov    %rsp,%rdx
  4007c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4007cd:	50                   	push   %rax
  4007ce:	54                   	push   %rsp
  4007cf:	49 c7 c0 20 14 40 00 	mov    $0x401420,%r8
  4007d6:	48 c7 c1 b0 13 40 00 	mov    $0x4013b0,%rcx
  4007dd:	48 c7 c7 50 11 40 00 	mov    $0x401150,%rdi
  4007e4:	e8 27 ff ff ff       	callq  400710 <__libc_start_main@plt>
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

00000000004008c0 <_Z11r_float2intf>:
  4008c0:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  4008c4:	c3                   	retq   
  4008c5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4008cc:	00 00 00 00 

00000000004008d0 <_Z15Body_initializemddddd>:
  4008d0:	c5 fb 11 04 fd 90 20 	vmovsd %xmm0,0x602090(,%rdi,8)
  4008d7:	60 00 
  4008d9:	c5 fb 11 0c fd 90 1a 	vmovsd %xmm1,0x611a90(,%rdi,8)
  4008e0:	61 00 
  4008e2:	c5 fb 11 14 fd 90 14 	vmovsd %xmm2,0x621490(,%rdi,8)
  4008e9:	62 00 
  4008eb:	c5 fb 11 1c fd 90 0e 	vmovsd %xmm3,0x630e90(,%rdi,8)
  4008f2:	63 00 
  4008f4:	c5 fb 11 24 fd 90 08 	vmovsd %xmm4,0x640890(,%rdi,8)
  4008fb:	64 00 
  4008fd:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  400904:	00 00 00 00 00 
  400909:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  400910:	00 00 00 00 00 
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z14Body_add_forcemm>:
  400920:	48 39 f7             	cmp    %rsi,%rdi
  400923:	0f 84 91 00 00 00    	je     4009ba <_Z14Body_add_forcemm+0x9a>
  400929:	c5 fb 10 04 f5 90 1a 	vmovsd 0x611a90(,%rsi,8),%xmm0
  400930:	61 00 
  400932:	c5 fb 5c 04 fd 90 1a 	vsubsd 0x611a90(,%rdi,8),%xmm0,%xmm0
  400939:	61 00 
  40093b:	c5 fb 10 0c f5 90 14 	vmovsd 0x621490(,%rsi,8),%xmm1
  400942:	62 00 
  400944:	c5 f3 5c 0c fd 90 14 	vsubsd 0x621490(,%rdi,8),%xmm1,%xmm1
  40094b:	62 00 
  40094d:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  400951:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  400956:	c5 fb 51 da          	vsqrtsd %xmm2,%xmm0,%xmm3
  40095a:	c5 fb 10 24 fd 90 20 	vmovsd 0x602090(,%rdi,8),%xmm4
  400961:	60 00 
  400963:	c5 db 59 25 dd 0a 00 	vmulsd 0xadd(%rip),%xmm4,%xmm4        # 401448 <_IO_stdin_used+0x8>
  40096a:	00 
  40096b:	c5 db 59 24 f5 90 20 	vmulsd 0x602090(,%rsi,8),%xmm4,%xmm4
  400972:	60 00 
  400974:	c5 eb 58 15 d4 0a 00 	vaddsd 0xad4(%rip),%xmm2,%xmm2        # 401450 <_IO_stdin_used+0x10>
  40097b:	00 
  40097c:	c5 db 5e d2          	vdivsd %xmm2,%xmm4,%xmm2
  400980:	c5 eb 59 c0          	vmulsd %xmm0,%xmm2,%xmm0
  400984:	c5 fb 10 25 cc 0a 00 	vmovsd 0xacc(%rip),%xmm4        # 401458 <_IO_stdin_used+0x18>
  40098b:	00 
  40098c:	c5 db 5e db          	vdivsd %xmm3,%xmm4,%xmm3
  400990:	c4 e2 e1 a9 04 fd 90 	vfmadd213sd 0x650290(,%rdi,8),%xmm3,%xmm0
  400997:	02 65 00 
  40099a:	c5 fb 11 04 fd 90 02 	vmovsd %xmm0,0x650290(,%rdi,8)
  4009a1:	65 00 
  4009a3:	c5 eb 59 c1          	vmulsd %xmm1,%xmm2,%xmm0
  4009a7:	c4 e2 e1 a9 04 fd 90 	vfmadd213sd 0x65fc90(,%rdi,8),%xmm3,%xmm0
  4009ae:	fc 65 00 
  4009b1:	c5 fb 11 04 fd 90 fc 	vmovsd %xmm0,0x65fc90(,%rdi,8)
  4009b8:	65 00 
  4009ba:	c3                   	retq   
  4009bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004009c0 <_Z21Body_add_force_to_allm>:
  4009c0:	48 89 f8             	mov    %rdi,%rax
  4009c3:	48 f7 d8             	neg    %rax
  4009c6:	48 c7 c1 c0 e0 ff ff 	mov    $0xffffffffffffe0c0,%rcx
  4009cd:	c5 fb 10 05 73 0a 00 	vmovsd 0xa73(%rip),%xmm0        # 401448 <_IO_stdin_used+0x8>
  4009d4:	00 
  4009d5:	c5 fb 10 0d 73 0a 00 	vmovsd 0xa73(%rip),%xmm1        # 401450 <_IO_stdin_used+0x10>
  4009dc:	00 
  4009dd:	c5 fb 10 15 73 0a 00 	vmovsd 0xa73(%rip),%xmm2        # 401458 <_IO_stdin_used+0x18>
  4009e4:	00 
  4009e5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4009ec:	00 00 00 00 
  4009f0:	48 8d 14 08          	lea    (%rax,%rcx,1),%rdx
  4009f4:	48 81 fa c0 e0 ff ff 	cmp    $0xffffffffffffe0c0,%rdx
  4009fb:	74 7d                	je     400a7a <_Z21Body_add_force_to_allm+0xba>
  4009fd:	c5 fb 10 1c fd 90 1a 	vmovsd 0x611a90(,%rdi,8),%xmm3
  400a04:	61 00 
  400a06:	c5 e3 5c 1c cd 90 14 	vsubsd 0x621490(,%rcx,8),%xmm3,%xmm3
  400a0d:	62 00 
  400a0f:	c5 fb 10 24 fd 90 14 	vmovsd 0x621490(,%rdi,8),%xmm4
  400a16:	62 00 
  400a18:	c5 db 5c 24 cd 90 0e 	vsubsd 0x630e90(,%rcx,8),%xmm4,%xmm4
  400a1f:	63 00 
  400a21:	c5 e3 59 eb          	vmulsd %xmm3,%xmm3,%xmm5
  400a25:	c4 e2 d9 b9 ec       	vfmadd231sd %xmm4,%xmm4,%xmm5
  400a2a:	c5 fb 59 34 cd 90 1a 	vmulsd 0x611a90(,%rcx,8),%xmm0,%xmm6
  400a31:	61 00 
  400a33:	c5 cb 59 34 fd 90 20 	vmulsd 0x602090(,%rdi,8),%xmm6,%xmm6
  400a3a:	60 00 
  400a3c:	c5 fb 51 fd          	vsqrtsd %xmm5,%xmm0,%xmm7
  400a40:	c5 d3 58 e9          	vaddsd %xmm1,%xmm5,%xmm5
  400a44:	c5 cb 5e ed          	vdivsd %xmm5,%xmm6,%xmm5
  400a48:	c5 d3 59 db          	vmulsd %xmm3,%xmm5,%xmm3
  400a4c:	c5 eb 5e f7          	vdivsd %xmm7,%xmm2,%xmm6
  400a50:	c4 e2 c9 a9 1c cd 90 	vfmadd213sd 0x65fc90(,%rcx,8),%xmm6,%xmm3
  400a57:	fc 65 00 
  400a5a:	c5 fb 11 1c cd 90 fc 	vmovsd %xmm3,0x65fc90(,%rcx,8)
  400a61:	65 00 
  400a63:	c5 d3 59 dc          	vmulsd %xmm4,%xmm5,%xmm3
  400a67:	c4 e2 c9 a9 1c cd 90 	vfmadd213sd 0x66f690(,%rcx,8),%xmm6,%xmm3
  400a6e:	f6 66 00 
  400a71:	c5 fb 11 1c cd 90 f6 	vmovsd %xmm3,0x66f690(,%rcx,8)
  400a78:	66 00 
  400a7a:	48 83 c1 01          	add    $0x1,%rcx
  400a7e:	0f 85 6c ff ff ff    	jne    4009f0 <_Z21Body_add_force_to_allm+0x30>
  400a84:	c3                   	retq   
  400a85:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  400a8c:	00 00 00 00 

0000000000400a90 <_Z16Body_reset_forcem>:
  400a90:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  400a97:	00 00 00 00 00 
  400a9c:	48 c7 04 fd 90 fc 65 	movq   $0x0,0x65fc90(,%rdi,8)
  400aa3:	00 00 00 00 00 
  400aa8:	c3                   	retq   
  400aa9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ab0 <_Z11Body_updatemd>:
  400ab0:	c5 fb 59 14 fd 90 02 	vmulsd 0x650290(,%rdi,8),%xmm0,%xmm2
  400ab7:	65 00 
  400ab9:	c5 fb 10 0d 97 09 00 	vmovsd 0x997(%rip),%xmm1        # 401458 <_IO_stdin_used+0x18>
  400ac0:	00 
  400ac1:	c5 f3 5e 1c fd 90 20 	vdivsd 0x602090(,%rdi,8),%xmm1,%xmm3
  400ac8:	60 00 
  400aca:	c4 e2 e1 a9 14 fd 90 	vfmadd213sd 0x630e90(,%rdi,8),%xmm3,%xmm2
  400ad1:	0e 63 00 
  400ad4:	c5 fb 11 14 fd 90 0e 	vmovsd %xmm2,0x630e90(,%rdi,8)
  400adb:	63 00 
  400add:	c5 fb 59 0c fd 90 fc 	vmulsd 0x65fc90(,%rdi,8),%xmm0,%xmm1
  400ae4:	65 00 
  400ae6:	c4 e2 e1 a9 0c fd 90 	vfmadd213sd 0x640890(,%rdi,8),%xmm3,%xmm1
  400aed:	08 64 00 
  400af0:	c5 fb 11 0c fd 90 08 	vmovsd %xmm1,0x640890(,%rdi,8)
  400af7:	64 00 
  400af9:	c5 fb 10 1c fd 90 1a 	vmovsd 0x611a90(,%rdi,8),%xmm3
  400b00:	61 00 
  400b02:	c4 e2 e9 b9 d8       	vfmadd231sd %xmm0,%xmm2,%xmm3
  400b07:	c5 fb 11 1c fd 90 1a 	vmovsd %xmm3,0x611a90(,%rdi,8)
  400b0e:	61 00 
  400b10:	c4 e2 f1 a9 04 fd 90 	vfmadd213sd 0x621490(,%rdi,8),%xmm1,%xmm0
  400b17:	14 62 00 
  400b1a:	c5 fb 11 04 fd 90 14 	vmovsd %xmm0,0x621490(,%rdi,8)
  400b21:	62 00 
  400b23:	c5 f9 2e 1d 35 09 00 	vucomisd 0x935(%rip),%xmm3        # 401460 <_IO_stdin_used+0x20>
  400b2a:	00 
  400b2b:	72 0a                	jb     400b37 <_Z11Body_updatemd+0x87>
  400b2d:	c5 f9 2e 1d 23 09 00 	vucomisd 0x923(%rip),%xmm3        # 401458 <_IO_stdin_used+0x18>
  400b34:	00 
  400b35:	76 11                	jbe    400b48 <_Z11Body_updatemd+0x98>
  400b37:	c5 e9 57 15 61 09 00 	vxorpd 0x961(%rip),%xmm2,%xmm2        # 4014a0 <_IO_stdin_used+0x60>
  400b3e:	00 
  400b3f:	c5 f9 13 14 fd 90 0e 	vmovlpd %xmm2,0x630e90(,%rdi,8)
  400b46:	63 00 
  400b48:	c5 f9 2e 05 10 09 00 	vucomisd 0x910(%rip),%xmm0        # 401460 <_IO_stdin_used+0x20>
  400b4f:	00 
  400b50:	72 0a                	jb     400b5c <_Z11Body_updatemd+0xac>
  400b52:	c5 f9 2e 05 fe 08 00 	vucomisd 0x8fe(%rip),%xmm0        # 401458 <_IO_stdin_used+0x18>
  400b59:	00 
  400b5a:	76 11                	jbe    400b6d <_Z11Body_updatemd+0xbd>
  400b5c:	c5 f1 57 05 3c 09 00 	vxorpd 0x93c(%rip),%xmm1,%xmm0        # 4014a0 <_IO_stdin_used+0x60>
  400b63:	00 
  400b64:	c5 f9 13 04 fd 90 08 	vmovlpd %xmm0,0x640890(,%rdi,8)
  400b6b:	64 00 
  400b6d:	c3                   	retq   
  400b6e:	66 90                	xchg   %ax,%ax

0000000000400b70 <_Z26Body_codegen_simple_updatemd>:
  400b70:	c5 fb 12 c0          	vmovddup %xmm0,%xmm0
  400b74:	c5 fb 10 0c fd 90 fc 	vmovsd 0x65fc90(,%rdi,8),%xmm1
  400b7b:	65 00 
  400b7d:	c5 f1 16 0c fd 90 02 	vmovhpd 0x650290(,%rdi,8),%xmm1,%xmm1
  400b84:	65 00 
  400b86:	c5 f9 59 c9          	vmulpd %xmm1,%xmm0,%xmm1
  400b8a:	c5 fb 12 14 fd 90 20 	vmovddup 0x602090(,%rdi,8),%xmm2
  400b91:	60 00 
  400b93:	c5 f1 5e d2          	vdivpd %xmm2,%xmm1,%xmm2
  400b97:	c5 fb 10 0c fd 90 14 	vmovsd 0x621490(,%rdi,8),%xmm1
  400b9e:	62 00 
  400ba0:	c5 f1 16 0c fd 90 1a 	vmovhpd 0x611a90(,%rdi,8),%xmm1,%xmm1
  400ba7:	61 00 
  400ba9:	c5 fb 10 1c fd 90 08 	vmovsd 0x640890(,%rdi,8),%xmm3
  400bb0:	64 00 
  400bb2:	c5 e1 16 1c fd 90 0e 	vmovhpd 0x630e90(,%rdi,8),%xmm3,%xmm3
  400bb9:	63 00 
  400bbb:	b8 64 00 00 00       	mov    $0x64,%eax
  400bc0:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bc4:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bc9:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bcd:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bd2:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bd6:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bdb:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bdf:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400be4:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400be8:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bed:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bf1:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bf6:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400bfa:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400bff:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400c03:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400c08:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400c0c:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400c11:	c5 e1 58 da          	vaddpd %xmm2,%xmm3,%xmm3
  400c15:	c4 e2 f9 b8 cb       	vfmadd231pd %xmm3,%xmm0,%xmm1
  400c1a:	83 c0 f6             	add    $0xfffffff6,%eax
  400c1d:	75 a1                	jne    400bc0 <_Z26Body_codegen_simple_updatemd+0x50>
  400c1f:	c5 f9 17 1c fd 90 0e 	vmovhpd %xmm3,0x630e90(,%rdi,8)
  400c26:	63 00 
  400c28:	c5 f9 13 1c fd 90 08 	vmovlpd %xmm3,0x640890(,%rdi,8)
  400c2f:	64 00 
  400c31:	c5 f9 17 0c fd 90 1a 	vmovhpd %xmm1,0x611a90(,%rdi,8)
  400c38:	61 00 
  400c3a:	c5 f9 13 0c fd 90 14 	vmovlpd %xmm1,0x621490(,%rdi,8)
  400c41:	62 00 
  400c43:	c3                   	retq   
  400c44:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400c4b:	00 00 00 00 00 

0000000000400c50 <_Z13instantiationv>:
  400c50:	53                   	push   %rbx
  400c51:	48 83 ec 30          	sub    $0x30,%rsp
  400c55:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400c5a:	e8 e1 fa ff ff       	callq  400740 <srand@plt>
  400c5f:	48 c7 c3 00 06 ff ff 	mov    $0xffffffffffff0600,%rbx
  400c66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6d:	00 00 00 
  400c70:	e8 eb fa ff ff       	callq  400760 <rand@plt>
  400c75:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400c79:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  400c7d:	c5 fb 10 05 eb 07 00 	vmovsd 0x7eb(%rip),%xmm0        # 401470 <_IO_stdin_used+0x30>
  400c84:	00 
  400c85:	c4 e2 f9 99 0d da 07 	vfmadd132sd 0x7da(%rip),%xmm0,%xmm1        # 401468 <_IO_stdin_used+0x28>
  400c8c:	00 00 
  400c8e:	c5 fb 11 4c 24 28    	vmovsd %xmm1,0x28(%rsp)
  400c94:	e8 c7 fa ff ff       	callq  400760 <rand@plt>
  400c99:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400c9d:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  400ca1:	c5 fb 10 05 b7 07 00 	vmovsd 0x7b7(%rip),%xmm0        # 401460 <_IO_stdin_used+0x20>
  400ca8:	00 
  400ca9:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400cad:	c5 fb 10 05 c3 07 00 	vmovsd 0x7c3(%rip),%xmm0        # 401478 <_IO_stdin_used+0x38>
  400cb4:	00 
  400cb5:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400cba:	c5 fb 11 4c 24 20    	vmovsd %xmm1,0x20(%rsp)
  400cc0:	e8 9b fa ff ff       	callq  400760 <rand@plt>
  400cc5:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400cc9:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400ccd:	c5 fb 10 0d a3 07 00 	vmovsd 0x7a3(%rip),%xmm1        # 401478 <_IO_stdin_used+0x38>
  400cd4:	00 
  400cd5:	c4 e2 f1 a9 05 82 07 	vfmadd213sd 0x782(%rip),%xmm1,%xmm0        # 401460 <_IO_stdin_used+0x20>
  400cdc:	00 00 
  400cde:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
  400ce4:	e8 77 fa ff ff       	callq  400760 <rand@plt>
  400ce9:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400ced:	c5 fb 2a c8          	vcvtsi2sd %eax,%xmm0,%xmm1
  400cf1:	c5 fb 10 05 87 07 00 	vmovsd 0x787(%rip),%xmm0        # 401480 <_IO_stdin_used+0x40>
  400cf8:	00 
  400cf9:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400cfd:	c5 fb 10 05 83 07 00 	vmovsd 0x783(%rip),%xmm0        # 401488 <_IO_stdin_used+0x48>
  400d04:	00 
  400d05:	c4 e2 f9 a9 ca       	vfmadd213sd %xmm2,%xmm0,%xmm1
  400d0a:	c5 fb 11 4c 24 10    	vmovsd %xmm1,0x10(%rsp)
  400d10:	e8 4b fa ff ff       	callq  400760 <rand@plt>
  400d15:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400d19:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400d1d:	c5 fb 10 0d 63 07 00 	vmovsd 0x763(%rip),%xmm1        # 401488 <_IO_stdin_used+0x48>
  400d24:	00 
  400d25:	c4 e2 f1 a9 05 52 07 	vfmadd213sd 0x752(%rip),%xmm1,%xmm0        # 401480 <_IO_stdin_used+0x40>
  400d2c:	00 00 
  400d2e:	c5 fb 10 4c 24 28    	vmovsd 0x28(%rsp),%xmm1
  400d34:	c5 fb 11 8b 90 1a 61 	vmovsd %xmm1,0x611a90(%rbx)
  400d3b:	00 
  400d3c:	c5 fb 10 4c 24 20    	vmovsd 0x20(%rsp),%xmm1
  400d42:	c5 fb 11 8b 90 14 62 	vmovsd %xmm1,0x621490(%rbx)
  400d49:	00 
  400d4a:	c5 fb 10 4c 24 18    	vmovsd 0x18(%rsp),%xmm1
  400d50:	c5 fb 11 8b 90 0e 63 	vmovsd %xmm1,0x630e90(%rbx)
  400d57:	00 
  400d58:	c5 fb 10 4c 24 10    	vmovsd 0x10(%rsp),%xmm1
  400d5e:	c5 fb 11 8b 90 08 64 	vmovsd %xmm1,0x640890(%rbx)
  400d65:	00 
  400d66:	c5 fb 11 83 90 02 65 	vmovsd %xmm0,0x650290(%rbx)
  400d6d:	00 
  400d6e:	48 c7 83 90 fc 65 00 	movq   $0x0,0x65fc90(%rbx)
  400d75:	00 00 00 00 
  400d79:	48 c7 83 90 f6 66 00 	movq   $0x0,0x66f690(%rbx)
  400d80:	00 00 00 00 
  400d84:	48 83 c3 08          	add    $0x8,%rbx
  400d88:	0f 85 e2 fe ff ff    	jne    400c70 <_Z13instantiationv+0x20>
  400d8e:	48 83 c4 30          	add    $0x30,%rsp
  400d92:	5b                   	pop    %rbx
  400d93:	c3                   	retq   
  400d94:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400d9b:	00 00 00 00 00 

0000000000400da0 <_Z14run_simulationv>:
  400da0:	41 56                	push   %r14
  400da2:	53                   	push   %rbx
  400da3:	48 81 ec 88 00 00 00 	sub    $0x88,%rsp
  400daa:	45 31 f6             	xor    %r14d,%r14d
  400dad:	c4 e2 7d 19 05 92 06 	vbroadcastsd 0x692(%rip),%ymm0        # 401448 <_IO_stdin_used+0x8>
  400db4:	00 00 
  400db6:	c5 fc 11 44 24 60    	vmovups %ymm0,0x60(%rsp)
  400dbc:	c4 e2 7d 19 05 8b 06 	vbroadcastsd 0x68b(%rip),%ymm0        # 401450 <_IO_stdin_used+0x10>
  400dc3:	00 00 
  400dc5:	c5 fc 11 44 24 40    	vmovups %ymm0,0x40(%rsp)
  400dcb:	c4 e2 7d 19 05 84 06 	vbroadcastsd 0x684(%rip),%ymm0        # 401458 <_IO_stdin_used+0x18>
  400dd2:	00 00 
  400dd4:	c5 fd 11 44 24 20    	vmovupd %ymm0,0x20(%rsp)
  400dda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400de0:	bf 90 02 65 00       	mov    $0x650290,%edi
  400de5:	31 f6                	xor    %esi,%esi
  400de7:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400dec:	c5 f8 77             	vzeroupper 
  400def:	e8 fc f8 ff ff       	callq  4006f0 <memset@plt>
  400df4:	bf 90 fc 65 00       	mov    $0x65fc90,%edi
  400df9:	31 f6                	xor    %esi,%esi
  400dfb:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400e00:	e8 eb f8 ff ff       	callq  4006f0 <memset@plt>
  400e05:	c5 7b 10 3d 83 06 00 	vmovsd 0x683(%rip),%xmm15        # 401490 <_IO_stdin_used+0x50>
  400e0c:	00 
  400e0d:	45 31 c0             	xor    %r8d,%r8d
  400e10:	c4 c1 f9 6e c0       	vmovq  %r8,%xmm0
  400e15:	c4 e2 7d 59 c0       	vpbroadcastq %xmm0,%ymm0
  400e1a:	c4 a2 7d 19 0c c5 90 	vbroadcastsd 0x611a90(,%r8,8),%ymm1
  400e21:	1a 61 00 
  400e24:	c4 a2 7d 19 14 c5 90 	vbroadcastsd 0x621490(,%r8,8),%ymm2
  400e2b:	14 62 00 
  400e2e:	c4 a2 7d 19 1c c5 90 	vbroadcastsd 0x602090(,%r8,8),%ymm3
  400e35:	20 60 00 
  400e38:	b9 90 fc 65 00       	mov    $0x65fc90,%ecx
  400e3d:	ba 90 02 65 00       	mov    $0x650290,%edx
  400e42:	be 90 20 60 00       	mov    $0x602090,%esi
  400e47:	bf 90 14 62 00       	mov    $0x621490,%edi
  400e4c:	bb 90 1a 61 00       	mov    $0x611a90,%ebx
  400e51:	31 c0                	xor    %eax,%eax
  400e53:	c5 7d 6f 15 65 06 00 	vmovdqa 0x665(%rip),%ymm10        # 4014c0 <_IO_stdin_used+0x80>
  400e5a:	00 
  400e5b:	c4 41 25 76 db       	vpcmpeqd %ymm11,%ymm11,%ymm11
  400e60:	c5 7d 10 64 24 60    	vmovupd 0x60(%rsp),%ymm12
  400e66:	c5 7d 10 6c 24 40    	vmovupd 0x40(%rsp),%ymm13
  400e6c:	c5 7d 10 74 24 20    	vmovupd 0x20(%rsp),%ymm14
  400e72:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400e79:	1f 84 00 00 00 00 00 
  400e80:	c4 e1 f9 6e e0       	vmovq  %rax,%xmm4
  400e85:	c4 e2 7d 59 e4       	vpbroadcastq %xmm4,%ymm4
  400e8a:	c4 c1 5d d4 e2       	vpaddq %ymm10,%ymm4,%ymm4
  400e8f:	c4 e2 5d 29 e0       	vpcmpeqq %ymm0,%ymm4,%ymm4
  400e94:	c4 c1 5d ef e3       	vpxor  %ymm11,%ymm4,%ymm4
  400e99:	c4 e2 5d 2d 2b       	vmaskmovpd (%rbx),%ymm4,%ymm5
  400e9e:	c4 e2 5d 2d 37       	vmaskmovpd (%rdi),%ymm4,%ymm6
  400ea3:	c5 f5 5c ed          	vsubpd %ymm5,%ymm1,%ymm5
  400ea7:	c5 d5 59 fd          	vmulpd %ymm5,%ymm5,%ymm7
  400eab:	c5 ed 5c f6          	vsubpd %ymm6,%ymm2,%ymm6
  400eaf:	c4 e2 cd b8 fe       	vfmadd231pd %ymm6,%ymm6,%ymm7
  400eb4:	c5 7d 51 c7          	vsqrtpd %ymm7,%ymm8
  400eb8:	c4 62 5d 2d 0e       	vmaskmovpd (%rsi),%ymm4,%ymm9
  400ebd:	c4 41 35 59 cc       	vmulpd %ymm12,%ymm9,%ymm9
  400ec2:	c5 35 59 cb          	vmulpd %ymm3,%ymm9,%ymm9
  400ec6:	c4 c1 45 58 fd       	vaddpd %ymm13,%ymm7,%ymm7
  400ecb:	c5 b5 5e ff          	vdivpd %ymm7,%ymm9,%ymm7
  400ecf:	c4 62 5d 2d 0a       	vmaskmovpd (%rdx),%ymm4,%ymm9
  400ed4:	c5 c5 59 ed          	vmulpd %ymm5,%ymm7,%ymm5
  400ed8:	c5 c5 59 f6          	vmulpd %ymm6,%ymm7,%ymm6
  400edc:	c4 c1 0d 5e f8       	vdivpd %ymm8,%ymm14,%ymm7
  400ee1:	c4 62 5d 2d 01       	vmaskmovpd (%rcx),%ymm4,%ymm8
  400ee6:	c4 c2 c5 a8 e9       	vfmadd213pd %ymm9,%ymm7,%ymm5
  400eeb:	48 83 c0 04          	add    $0x4,%rax
  400eef:	48 83 c3 20          	add    $0x20,%rbx
  400ef3:	48 83 c7 20          	add    $0x20,%rdi
  400ef7:	c4 c2 c5 a8 f0       	vfmadd213pd %ymm8,%ymm7,%ymm6
  400efc:	c4 e2 5d 2f 2a       	vmaskmovpd %ymm5,%ymm4,(%rdx)
  400f01:	c4 e2 5d 2f 31       	vmaskmovpd %ymm6,%ymm4,(%rcx)
  400f06:	48 83 c6 20          	add    $0x20,%rsi
  400f0a:	48 83 c2 20          	add    $0x20,%rdx
  400f0e:	48 83 c1 20          	add    $0x20,%rcx
  400f12:	48 3d 40 1f 00 00    	cmp    $0x1f40,%rax
  400f18:	0f 85 62 ff ff ff    	jne    400e80 <_Z14run_simulationv+0xe0>
  400f1e:	49 83 c0 01          	add    $0x1,%r8
  400f22:	48 c7 c0 00 06 ff ff 	mov    $0xffffffffffff0600,%rax
  400f29:	49 81 f8 40 1f 00 00 	cmp    $0x1f40,%r8
  400f30:	0f 85 da fe ff ff    	jne    400e10 <_Z14run_simulationv+0x70>
  400f36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f3d:	00 00 00 
  400f40:	c5 83 59 88 90 fc 65 	vmulsd 0x65fc90(%rax),%xmm15,%xmm1
  400f47:	00 
  400f48:	c5 fb 10 05 08 05 00 	vmovsd 0x508(%rip),%xmm0        # 401458 <_IO_stdin_used+0x18>
  400f4f:	00 
  400f50:	c5 fb 5e 90 90 1a 61 	vdivsd 0x611a90(%rax),%xmm0,%xmm2
  400f57:	00 
  400f58:	c4 e2 e9 a9 88 90 08 	vfmadd213sd 0x640890(%rax),%xmm2,%xmm1
  400f5f:	64 00 
  400f61:	c5 fb 11 88 90 08 64 	vmovsd %xmm1,0x640890(%rax)
  400f68:	00 
  400f69:	c5 83 59 80 90 f6 66 	vmulsd 0x66f690(%rax),%xmm15,%xmm0
  400f70:	00 
  400f71:	c4 e2 e9 a9 80 90 02 	vfmadd213sd 0x650290(%rax),%xmm2,%xmm0
  400f78:	65 00 
  400f7a:	c5 fb 11 80 90 02 65 	vmovsd %xmm0,0x650290(%rax)
  400f81:	00 
  400f82:	c5 fb 10 98 90 14 62 	vmovsd 0x621490(%rax),%xmm3
  400f89:	00 
  400f8a:	c4 c2 f1 b9 df       	vfmadd231sd %xmm15,%xmm1,%xmm3
  400f8f:	c5 fb 11 98 90 14 62 	vmovsd %xmm3,0x621490(%rax)
  400f96:	00 
  400f97:	c5 fb 10 90 90 0e 63 	vmovsd 0x630e90(%rax),%xmm2
  400f9e:	00 
  400f9f:	c4 c2 f9 b9 d7       	vfmadd231sd %xmm15,%xmm0,%xmm2
  400fa4:	c5 fb 11 90 90 0e 63 	vmovsd %xmm2,0x630e90(%rax)
  400fab:	00 
  400fac:	c5 f9 2e 1d ac 04 00 	vucomisd 0x4ac(%rip),%xmm3        # 401460 <_IO_stdin_used+0x20>
  400fb3:	00 
  400fb4:	72 0a                	jb     400fc0 <_Z14run_simulationv+0x220>
  400fb6:	c5 f9 2e 1d 9a 04 00 	vucomisd 0x49a(%rip),%xmm3        # 401458 <_IO_stdin_used+0x18>
  400fbd:	00 
  400fbe:	76 10                	jbe    400fd0 <_Z14run_simulationv+0x230>
  400fc0:	c5 f1 57 0d d8 04 00 	vxorpd 0x4d8(%rip),%xmm1,%xmm1        # 4014a0 <_IO_stdin_used+0x60>
  400fc7:	00 
  400fc8:	c5 f9 13 88 90 08 64 	vmovlpd %xmm1,0x640890(%rax)
  400fcf:	00 
  400fd0:	c5 f9 2e 15 88 04 00 	vucomisd 0x488(%rip),%xmm2        # 401460 <_IO_stdin_used+0x20>
  400fd7:	00 
  400fd8:	72 0a                	jb     400fe4 <_Z14run_simulationv+0x244>
  400fda:	c5 f9 2e 15 76 04 00 	vucomisd 0x476(%rip),%xmm2        # 401458 <_IO_stdin_used+0x18>
  400fe1:	00 
  400fe2:	76 10                	jbe    400ff4 <_Z14run_simulationv+0x254>
  400fe4:	c5 f9 57 05 b4 04 00 	vxorpd 0x4b4(%rip),%xmm0,%xmm0        # 4014a0 <_IO_stdin_used+0x60>
  400feb:	00 
  400fec:	c5 f9 13 80 90 02 65 	vmovlpd %xmm0,0x650290(%rax)
  400ff3:	00 
  400ff4:	48 83 c0 08          	add    $0x8,%rax
  400ff8:	0f 85 42 ff ff ff    	jne    400f40 <_Z14run_simulationv+0x1a0>
  400ffe:	41 83 c6 01          	add    $0x1,%r14d
  401002:	41 83 fe 05          	cmp    $0x5,%r14d
  401006:	0f 85 d4 fd ff ff    	jne    400de0 <_Z14run_simulationv+0x40>
  40100c:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
  401013:	5b                   	pop    %rbx
  401014:	41 5e                	pop    %r14
  401016:	c5 f8 77             	vzeroupper 
  401019:	c3                   	retq   
  40101a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401020 <_Z10run_simplev>:
  401020:	31 c0                	xor    %eax,%eax
  401022:	c5 f9 28 05 86 04 00 	vmovapd 0x486(%rip),%xmm0        # 4014b0 <_IO_stdin_used+0x70>
  401029:	00 
  40102a:	c5 f9 28 0d 7e 04 00 	vmovapd 0x47e(%rip),%xmm1        # 4014b0 <_IO_stdin_used+0x70>
  401031:	00 
  401032:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401039:	1f 84 00 00 00 00 00 
  401040:	31 c9                	xor    %ecx,%ecx
  401042:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  401049:	1f 84 00 00 00 00 00 
  401050:	c5 fb 10 14 cd 90 fc 	vmovsd 0x65fc90(,%rcx,8),%xmm2
  401057:	65 00 
  401059:	c5 e9 16 14 cd 90 02 	vmovhpd 0x650290(,%rcx,8),%xmm2,%xmm2
  401060:	65 00 
  401062:	c5 e9 59 d0          	vmulpd %xmm0,%xmm2,%xmm2
  401066:	c5 fb 12 1c cd 90 20 	vmovddup 0x602090(,%rcx,8),%xmm3
  40106d:	60 00 
  40106f:	c5 e9 5e db          	vdivpd %xmm3,%xmm2,%xmm3
  401073:	c5 fb 10 14 cd 90 14 	vmovsd 0x621490(,%rcx,8),%xmm2
  40107a:	62 00 
  40107c:	c5 e9 16 14 cd 90 1a 	vmovhpd 0x611a90(,%rcx,8),%xmm2,%xmm2
  401083:	61 00 
  401085:	c5 fb 10 24 cd 90 08 	vmovsd 0x640890(,%rcx,8),%xmm4
  40108c:	64 00 
  40108e:	c5 d9 16 24 cd 90 0e 	vmovhpd 0x630e90(,%rcx,8),%xmm4,%xmm4
  401095:	63 00 
  401097:	ba 64 00 00 00       	mov    $0x64,%edx
  40109c:	0f 1f 40 00          	nopl   0x0(%rax)
  4010a0:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010a4:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010a9:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010ad:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010b2:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010b6:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010bb:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010bf:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010c4:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010c8:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010cd:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010d1:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010d6:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010da:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010df:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010e3:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010e8:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010ec:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010f1:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  4010f5:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  4010fa:	83 c2 f6             	add    $0xfffffff6,%edx
  4010fd:	75 a1                	jne    4010a0 <_Z10run_simplev+0x80>
  4010ff:	c5 f9 17 24 cd 90 0e 	vmovhpd %xmm4,0x630e90(,%rcx,8)
  401106:	63 00 
  401108:	c5 f9 13 24 cd 90 08 	vmovlpd %xmm4,0x640890(,%rcx,8)
  40110f:	64 00 
  401111:	c5 f9 17 14 cd 90 1a 	vmovhpd %xmm2,0x611a90(,%rcx,8)
  401118:	61 00 
  40111a:	c5 f9 13 14 cd 90 14 	vmovlpd %xmm2,0x621490(,%rcx,8)
  401121:	62 00 
  401123:	48 83 c1 01          	add    $0x1,%rcx
  401127:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  40112e:	0f 85 1c ff ff ff    	jne    401050 <_Z10run_simplev+0x30>
  401134:	83 c0 01             	add    $0x1,%eax
  401137:	3d f4 01 00 00       	cmp    $0x1f4,%eax
  40113c:	0f 85 fe fe ff ff    	jne    401040 <_Z10run_simplev+0x20>
  401142:	c3                   	retq   
  401143:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40114a:	84 00 00 00 00 00 

0000000000401150 <main>:
  401150:	55                   	push   %rbp
  401151:	41 57                	push   %r15
  401153:	41 56                	push   %r14
  401155:	41 55                	push   %r13
  401157:	41 54                	push   %r12
  401159:	53                   	push   %rbx
  40115a:	50                   	push   %rax
  40115b:	e8 f0 f5 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401160:	48 89 c3             	mov    %rax,%rbx
  401163:	e8 e8 fa ff ff       	callq  400c50 <_Z13instantiationv>
  401168:	e8 e3 f5 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40116d:	49 89 c7             	mov    %rax,%r15
  401170:	49 29 df             	sub    %rbx,%r15
  401173:	e8 d8 f5 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401178:	48 89 c3             	mov    %rax,%rbx
  40117b:	e8 20 fc ff ff       	callq  400da0 <_Z14run_simulationv>
  401180:	e8 cb f5 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401185:	49 89 c4             	mov    %rax,%r12
  401188:	49 29 dc             	sub    %rbx,%r12
  40118b:	31 db                	xor    %ebx,%ebx
  40118d:	e8 be f5 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401192:	48 89 c5             	mov    %rax,%rbp
  401195:	c5 f9 28 05 13 03 00 	vmovapd 0x313(%rip),%xmm0        # 4014b0 <_IO_stdin_used+0x70>
  40119c:	00 
  40119d:	c5 f9 28 0d 0b 03 00 	vmovapd 0x30b(%rip),%xmm1        # 4014b0 <_IO_stdin_used+0x70>
  4011a4:	00 
  4011a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4011ac:	00 00 00 00 
  4011b0:	31 c0                	xor    %eax,%eax
  4011b2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4011b9:	1f 84 00 00 00 00 00 
  4011c0:	c5 fb 10 14 c5 90 fc 	vmovsd 0x65fc90(,%rax,8),%xmm2
  4011c7:	65 00 
  4011c9:	c5 e9 16 14 c5 90 02 	vmovhpd 0x650290(,%rax,8),%xmm2,%xmm2
  4011d0:	65 00 
  4011d2:	c5 e9 59 d0          	vmulpd %xmm0,%xmm2,%xmm2
  4011d6:	c5 fb 12 1c c5 90 20 	vmovddup 0x602090(,%rax,8),%xmm3
  4011dd:	60 00 
  4011df:	c5 e9 5e db          	vdivpd %xmm3,%xmm2,%xmm3
  4011e3:	c5 fb 10 14 c5 90 14 	vmovsd 0x621490(,%rax,8),%xmm2
  4011ea:	62 00 
  4011ec:	c5 e9 16 14 c5 90 1a 	vmovhpd 0x611a90(,%rax,8),%xmm2,%xmm2
  4011f3:	61 00 
  4011f5:	c5 fb 10 24 c5 90 08 	vmovsd 0x640890(,%rax,8),%xmm4
  4011fc:	64 00 
  4011fe:	c5 d9 16 24 c5 90 0e 	vmovhpd 0x630e90(,%rax,8),%xmm4,%xmm4
  401205:	63 00 
  401207:	b9 64 00 00 00       	mov    $0x64,%ecx
  40120c:	0f 1f 40 00          	nopl   0x0(%rax)
  401210:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401214:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401219:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  40121d:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401222:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401226:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  40122b:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  40122f:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401234:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401238:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  40123d:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401241:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401246:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  40124a:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  40124f:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401253:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401258:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  40125c:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  401261:	c5 d9 58 e3          	vaddpd %xmm3,%xmm4,%xmm4
  401265:	c4 e2 d9 b8 d1       	vfmadd231pd %xmm1,%xmm4,%xmm2
  40126a:	83 c1 f6             	add    $0xfffffff6,%ecx
  40126d:	75 a1                	jne    401210 <main+0xc0>
  40126f:	c5 f9 17 24 c5 90 0e 	vmovhpd %xmm4,0x630e90(,%rax,8)
  401276:	63 00 
  401278:	c5 f9 13 24 c5 90 08 	vmovlpd %xmm4,0x640890(,%rax,8)
  40127f:	64 00 
  401281:	c5 f9 17 14 c5 90 1a 	vmovhpd %xmm2,0x611a90(,%rax,8)
  401288:	61 00 
  40128a:	c5 f9 13 14 c5 90 14 	vmovlpd %xmm2,0x621490(,%rax,8)
  401291:	62 00 
  401293:	48 83 c0 01          	add    $0x1,%rax
  401297:	48 3d 40 1f 00 00    	cmp    $0x1f40,%rax
  40129d:	0f 85 1d ff ff ff    	jne    4011c0 <main+0x70>
  4012a3:	83 c3 01             	add    $0x1,%ebx
  4012a6:	81 fb f4 01 00 00    	cmp    $0x1f4,%ebx
  4012ac:	0f 85 fe fe ff ff    	jne    4011b0 <main+0x60>
  4012b2:	48 be db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rsi
  4012b9:	de 1b 43 
  4012bc:	4c 89 f8             	mov    %r15,%rax
  4012bf:	48 f7 ee             	imul   %rsi
  4012c2:	49 89 d7             	mov    %rdx,%r15
  4012c5:	4c 89 f9             	mov    %r15,%rcx
  4012c8:	48 c1 e9 3f          	shr    $0x3f,%rcx
  4012cc:	49 c1 ff 12          	sar    $0x12,%r15
  4012d0:	4c 89 e0             	mov    %r12,%rax
  4012d3:	48 f7 ee             	imul   %rsi
  4012d6:	49 89 d5             	mov    %rdx,%r13
  4012d9:	49 01 cf             	add    %rcx,%r15
  4012dc:	4c 89 e8             	mov    %r13,%rax
  4012df:	48 c1 e8 3f          	shr    $0x3f,%rax
  4012e3:	49 c1 fd 12          	sar    $0x12,%r13
  4012e7:	49 01 c5             	add    %rax,%r13
  4012ea:	e8 61 f4 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012ef:	49 89 c4             	mov    %rax,%r12
  4012f2:	49 29 ec             	sub    %rbp,%r12
  4012f5:	bd 0b 00 00 00       	mov    $0xb,%ebp
  4012fa:	31 db                	xor    %ebx,%ebx
  4012fc:	0f 1f 40 00          	nopl   0x0(%rax)
  401300:	c5 fb 10 04 dd 90 1a 	vmovsd 0x611a90(,%rbx,8),%xmm0
  401307:	61 00 
  401309:	c5 fb 5a c8          	vcvtsd2ss %xmm0,%xmm0,%xmm1
  40130d:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  401311:	01 e8                	add    %ebp,%eax
  401313:	c5 fb 10 0c dd 90 14 	vmovsd 0x621490(,%rbx,8),%xmm1
  40131a:	62 00 
  40131c:	c5 f3 5a d1          	vcvtsd2ss %xmm1,%xmm1,%xmm2
  401320:	c5 f9 7e d5          	vmovd  %xmm2,%ebp
  401324:	01 c5                	add    %eax,%ebp
  401326:	48 63 c5             	movslq %ebp,%rax
  401329:	48 69 c0 8d 68 b7 6c 	imul   $0x6cb7688d,%rax,%rax
  401330:	48 89 c1             	mov    %rax,%rcx
  401333:	48 c1 e9 3f          	shr    $0x3f,%rcx
  401337:	48 c1 f8 33          	sar    $0x33,%rax
  40133b:	01 c8                	add    %ecx,%eax
  40133d:	44 69 f0 87 d6 12 00 	imul   $0x12d687,%eax,%r14d
  401344:	48 83 fb 09          	cmp    $0x9,%rbx
  401348:	77 0f                	ja     401359 <main+0x209>
  40134a:	bf e0 14 40 00       	mov    $0x4014e0,%edi
  40134f:	b0 02                	mov    $0x2,%al
  401351:	48 89 de             	mov    %rbx,%rsi
  401354:	e8 87 f3 ff ff       	callq  4006e0 <printf@plt>
  401359:	44 29 f5             	sub    %r14d,%ebp
  40135c:	48 83 c3 01          	add    $0x1,%rbx
  401360:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  401367:	75 97                	jne    401300 <main+0x1b0>
  401369:	4c 89 e0             	mov    %r12,%rax
  40136c:	48 b9 db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rcx
  401373:	de 1b 43 
  401376:	48 f7 e9             	imul   %rcx
  401379:	48 89 d0             	mov    %rdx,%rax
  40137c:	48 c1 e8 3f          	shr    $0x3f,%rax
  401380:	48 c1 fa 12          	sar    $0x12,%rdx
  401384:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
  401388:	bf f5 14 40 00       	mov    $0x4014f5,%edi
  40138d:	31 c0                	xor    %eax,%eax
  40138f:	4c 89 fe             	mov    %r15,%rsi
  401392:	4c 89 ea             	mov    %r13,%rdx
  401395:	41 89 e8             	mov    %ebp,%r8d
  401398:	e8 43 f3 ff ff       	callq  4006e0 <printf@plt>
  40139d:	31 c0                	xor    %eax,%eax
  40139f:	48 83 c4 08          	add    $0x8,%rsp
  4013a3:	5b                   	pop    %rbx
  4013a4:	41 5c                	pop    %r12
  4013a6:	41 5d                	pop    %r13
  4013a8:	41 5e                	pop    %r14
  4013aa:	41 5f                	pop    %r15
  4013ac:	5d                   	pop    %rbp
  4013ad:	c3                   	retq   
  4013ae:	66 90                	xchg   %ax,%ax

00000000004013b0 <__libc_csu_init>:
  4013b0:	41 57                	push   %r15
  4013b2:	41 56                	push   %r14
  4013b4:	41 89 ff             	mov    %edi,%r15d
  4013b7:	41 55                	push   %r13
  4013b9:	41 54                	push   %r12
  4013bb:	4c 8d 25 0e 0a 20 00 	lea    0x200a0e(%rip),%r12        # 601dd0 <__frame_dummy_init_array_entry>
  4013c2:	55                   	push   %rbp
  4013c3:	48 8d 2d 1e 0a 20 00 	lea    0x200a1e(%rip),%rbp        # 601de8 <__init_array_end>
  4013ca:	53                   	push   %rbx
  4013cb:	49 89 f6             	mov    %rsi,%r14
  4013ce:	49 89 d5             	mov    %rdx,%r13
  4013d1:	4c 29 e5             	sub    %r12,%rbp
  4013d4:	48 83 ec 08          	sub    $0x8,%rsp
  4013d8:	48 c1 fd 03          	sar    $0x3,%rbp
  4013dc:	e8 c7 f2 ff ff       	callq  4006a8 <_init>
  4013e1:	48 85 ed             	test   %rbp,%rbp
  4013e4:	74 20                	je     401406 <__libc_csu_init+0x56>
  4013e6:	31 db                	xor    %ebx,%ebx
  4013e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4013ef:	00 
  4013f0:	4c 89 ea             	mov    %r13,%rdx
  4013f3:	4c 89 f6             	mov    %r14,%rsi
  4013f6:	44 89 ff             	mov    %r15d,%edi
  4013f9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4013fd:	48 83 c3 01          	add    $0x1,%rbx
  401401:	48 39 eb             	cmp    %rbp,%rbx
  401404:	75 ea                	jne    4013f0 <__libc_csu_init+0x40>
  401406:	48 83 c4 08          	add    $0x8,%rsp
  40140a:	5b                   	pop    %rbx
  40140b:	5d                   	pop    %rbp
  40140c:	41 5c                	pop    %r12
  40140e:	41 5d                	pop    %r13
  401410:	41 5e                	pop    %r14
  401412:	41 5f                	pop    %r15
  401414:	c3                   	retq   
  401415:	90                   	nop
  401416:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40141d:	00 00 00 

0000000000401420 <__libc_csu_fini>:
  401420:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401424 <_fini>:
  401424:	48 83 ec 08          	sub    $0x8,%rsp
  401428:	48 83 c4 08          	add    $0x8,%rsp
  40142c:	c3                   	retq   
