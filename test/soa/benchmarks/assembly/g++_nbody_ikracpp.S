
bin/g++_nbody_ikracpp:     file format elf64-x86-64


Disassembly of section .init:

00000000004006a8 <_init>:
  4006a8:	48 83 ec 08          	sub    $0x8,%rsp
  4006ac:	48 8b 05 45 19 20 00 	mov    0x201945(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4006b3:	48 85 c0             	test   %rax,%rax
  4006b6:	74 05                	je     4006bd <_init+0x15>
  4006b8:	e8 a3 00 00 00       	callq  400760 <sqrt@plt+0x10>
  4006bd:	48 83 c4 08          	add    $0x8,%rsp
  4006c1:	c3                   	retq   

Disassembly of section .plt:

00000000004006d0 <__printf_chk@plt-0x10>:
  4006d0:	ff 35 32 19 20 00    	pushq  0x201932(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006d6:	ff 25 34 19 20 00    	jmpq   *0x201934(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006e0 <__printf_chk@plt>:
  4006e0:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006e6:	68 00 00 00 00       	pushq  $0x0
  4006eb:	e9 e0 ff ff ff       	jmpq   4006d0 <_init+0x28>

00000000004006f0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006f0:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006f6:	68 01 00 00 00       	pushq  $0x1
  4006fb:	e9 d0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400700 <__libc_start_main@plt>:
  400700:	ff 25 22 19 20 00    	jmpq   *0x201922(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400706:	68 02 00 00 00       	pushq  $0x2
  40070b:	e9 c0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400710 <__cxa_atexit@plt>:
  400710:	ff 25 1a 19 20 00    	jmpq   *0x20191a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400716:	68 03 00 00 00       	pushq  $0x3
  40071b:	e9 b0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400720 <_ZNSt8ios_base4InitD1Ev@plt>:
  400720:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400726:	68 04 00 00 00       	pushq  $0x4
  40072b:	e9 a0 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400730:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400736:	68 05 00 00 00       	pushq  $0x5
  40073b:	e9 90 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400740 <rand@plt>:
  400740:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400746:	68 06 00 00 00       	pushq  $0x6
  40074b:	e9 80 ff ff ff       	jmpq   4006d0 <_init+0x28>

0000000000400750 <sqrt@plt>:
  400750:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400756:	68 07 00 00 00       	pushq  $0x7
  40075b:	e9 70 ff ff ff       	jmpq   4006d0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400760 <.plt.got>:
  400760:	ff 25 92 18 20 00    	jmpq   *0x201892(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  400766:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400770 <main>:
  400770:	55                   	push   %rbp
  400771:	53                   	push   %rbx
  400772:	bf 70 09 40 00       	mov    $0x400970,%edi
  400777:	48 83 ec 08          	sub    $0x8,%rsp
  40077b:	48 c7 05 1a 19 20 00 	movq   $0x0,0x20191a(%rip)        # 6020a0 <__Bodydata_buffer>
  400782:	00 00 00 00 
  400786:	48 c7 05 17 19 20 00 	movq   $0x0,0x201917(%rip)        # 6020a8 <__Bodydata_buffer+0x8>
  40078d:	00 00 00 00 
  400791:	e8 ea 09 00 00       	callq  401180 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400796:	bf 70 0b 40 00       	mov    $0x400b70,%edi
  40079b:	48 89 c3             	mov    %rax,%rbx
  40079e:	e8 dd 09 00 00       	callq  401180 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007a3:	bf d0 0a 40 00       	mov    $0x400ad0,%edi
  4007a8:	48 89 c5             	mov    %rax,%rbp
  4007ab:	e8 d0 09 00 00       	callq  401180 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007b0:	b9 b8 4c 61 00       	mov    $0x614cb8,%ecx
  4007b5:	49 89 c0             	mov    %rax,%r8
  4007b8:	bf b8 78 62 00       	mov    $0x6278b8,%edi
  4007bd:	ba 0b 00 00 00       	mov    $0xb,%edx
  4007c2:	be 8d 68 b7 6c       	mov    $0x6cb7688d,%esi
  4007c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ce:	00 00 
  4007d0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4007d4:	48 83 c1 08          	add    $0x8,%rcx
  4007d8:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4007dc:	f2 0f 5a 41 f8       	cvtsd2ss -0x8(%rcx),%xmm0
  4007e1:	66 41 0f 7e c1       	movd   %xmm0,%r9d
  4007e6:	f2 0f 5a 89 f8 2b 01 	cvtsd2ss 0x12bf8(%rcx),%xmm1
  4007ed:	00 
  4007ee:	44 01 ca             	add    %r9d,%edx
  4007f1:	66 41 0f 7e c9       	movd   %xmm1,%r9d
  4007f6:	41 01 d1             	add    %edx,%r9d
  4007f9:	44 89 c8             	mov    %r9d,%eax
  4007fc:	f7 ee                	imul   %esi
  4007fe:	44 89 c8             	mov    %r9d,%eax
  400801:	c1 f8 1f             	sar    $0x1f,%eax
  400804:	c1 fa 13             	sar    $0x13,%edx
  400807:	29 c2                	sub    %eax,%edx
  400809:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  40080f:	41 29 d1             	sub    %edx,%r9d
  400812:	48 39 cf             	cmp    %rcx,%rdi
  400815:	44 89 ca             	mov    %r9d,%edx
  400818:	75 b6                	jne    4007d0 <main+0x60>
  40081a:	48 89 e9             	mov    %rbp,%rcx
  40081d:	48 89 da             	mov    %rbx,%rdx
  400820:	be 48 12 40 00       	mov    $0x401248,%esi
  400825:	bf 01 00 00 00       	mov    $0x1,%edi
  40082a:	31 c0                	xor    %eax,%eax
  40082c:	e8 af fe ff ff       	callq  4006e0 <__printf_chk@plt>
  400831:	48 83 c4 08          	add    $0x8,%rsp
  400835:	31 c0                	xor    %eax,%eax
  400837:	5b                   	pop    %rbx
  400838:	5d                   	pop    %rbp
  400839:	c3                   	retq   
  40083a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400840 <_GLOBAL__sub_I__Z12float_as_intf>:
  400840:	48 83 ec 08          	sub    $0x8,%rsp
  400844:	bf b0 54 68 00       	mov    $0x6854b0,%edi
  400849:	e8 a2 fe ff ff       	callq  4006f0 <_ZNSt8ios_base4InitC1Ev@plt>
  40084e:	ba 60 20 60 00       	mov    $0x602060,%edx
  400853:	be b0 54 68 00       	mov    $0x6854b0,%esi
  400858:	bf 20 07 40 00       	mov    $0x400720,%edi
  40085d:	48 83 c4 08          	add    $0x8,%rsp
  400861:	e9 aa fe ff ff       	jmpq   400710 <__cxa_atexit@plt>
  400866:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40086d:	00 00 00 

0000000000400870 <_start>:
  400870:	31 ed                	xor    %ebp,%ebp
  400872:	49 89 d1             	mov    %rdx,%r9
  400875:	5e                   	pop    %rsi
  400876:	48 89 e2             	mov    %rsp,%rdx
  400879:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40087d:	50                   	push   %rax
  40087e:	54                   	push   %rsp
  40087f:	49 c7 c0 30 12 40 00 	mov    $0x401230,%r8
  400886:	48 c7 c1 c0 11 40 00 	mov    $0x4011c0,%rcx
  40088d:	48 c7 c7 70 07 40 00 	mov    $0x400770,%rdi
  400894:	e8 67 fe ff ff       	callq  400700 <__libc_start_main@plt>
  400899:	f4                   	hlt    
  40089a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008a0 <deregister_tm_clones>:
  4008a0:	b8 6f 20 60 00       	mov    $0x60206f,%eax
  4008a5:	55                   	push   %rbp
  4008a6:	48 2d 68 20 60 00    	sub    $0x602068,%rax
  4008ac:	48 83 f8 0e          	cmp    $0xe,%rax
  4008b0:	48 89 e5             	mov    %rsp,%rbp
  4008b3:	76 1b                	jbe    4008d0 <deregister_tm_clones+0x30>
  4008b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008ba:	48 85 c0             	test   %rax,%rax
  4008bd:	74 11                	je     4008d0 <deregister_tm_clones+0x30>
  4008bf:	5d                   	pop    %rbp
  4008c0:	bf 68 20 60 00       	mov    $0x602068,%edi
  4008c5:	ff e0                	jmpq   *%rax
  4008c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008ce:	00 00 
  4008d0:	5d                   	pop    %rbp
  4008d1:	c3                   	retq   
  4008d2:	0f 1f 40 00          	nopl   0x0(%rax)
  4008d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008dd:	00 00 00 

00000000004008e0 <register_tm_clones>:
  4008e0:	be 68 20 60 00       	mov    $0x602068,%esi
  4008e5:	55                   	push   %rbp
  4008e6:	48 81 ee 68 20 60 00 	sub    $0x602068,%rsi
  4008ed:	48 c1 fe 03          	sar    $0x3,%rsi
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 89 f0             	mov    %rsi,%rax
  4008f7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4008fb:	48 01 c6             	add    %rax,%rsi
  4008fe:	48 d1 fe             	sar    %rsi
  400901:	74 15                	je     400918 <register_tm_clones+0x38>
  400903:	b8 00 00 00 00       	mov    $0x0,%eax
  400908:	48 85 c0             	test   %rax,%rax
  40090b:	74 0b                	je     400918 <register_tm_clones+0x38>
  40090d:	5d                   	pop    %rbp
  40090e:	bf 68 20 60 00       	mov    $0x602068,%edi
  400913:	ff e0                	jmpq   *%rax
  400915:	0f 1f 00             	nopl   (%rax)
  400918:	5d                   	pop    %rbp
  400919:	c3                   	retq   
  40091a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400920 <__do_global_dtors_aux>:
  400920:	80 3d 59 17 20 00 00 	cmpb   $0x0,0x201759(%rip)        # 602080 <completed.7585>
  400927:	75 11                	jne    40093a <__do_global_dtors_aux+0x1a>
  400929:	55                   	push   %rbp
  40092a:	48 89 e5             	mov    %rsp,%rbp
  40092d:	e8 6e ff ff ff       	callq  4008a0 <deregister_tm_clones>
  400932:	5d                   	pop    %rbp
  400933:	c6 05 46 17 20 00 01 	movb   $0x1,0x201746(%rip)        # 602080 <completed.7585>
  40093a:	f3 c3                	repz retq 
  40093c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400940 <frame_dummy>:
  400940:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400945:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400949:	75 05                	jne    400950 <frame_dummy+0x10>
  40094b:	eb 93                	jmp    4008e0 <register_tm_clones>
  40094d:	0f 1f 00             	nopl   (%rax)
  400950:	b8 00 00 00 00       	mov    $0x0,%eax
  400955:	48 85 c0             	test   %rax,%rax
  400958:	74 f1                	je     40094b <frame_dummy+0xb>
  40095a:	55                   	push   %rbp
  40095b:	48 89 e5             	mov    %rsp,%rbp
  40095e:	ff d0                	callq  *%rax
  400960:	5d                   	pop    %rbp
  400961:	e9 7a ff ff ff       	jmpq   4008e0 <register_tm_clones>
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 

0000000000400970 <_Z13instantiationv>:
  400970:	41 56                	push   %r14
  400972:	41 55                	push   %r13
  400974:	41 be 80 25 00 00    	mov    $0x2580,%r14d
  40097a:	41 54                	push   %r12
  40097c:	55                   	push   %rbp
  40097d:	53                   	push   %rbx
  40097e:	48 83 ec 20          	sub    $0x20,%rsp
  400982:	f2 0f 10 0d fe 08 00 	movsd  0x8fe(%rip),%xmm1        # 401288 <_IO_stdin_used+0x48>
  400989:	00 
  40098a:	f2 0f 10 15 fe 08 00 	movsd  0x8fe(%rip),%xmm2        # 401290 <_IO_stdin_used+0x50>
  400991:	00 
  400992:	f2 0f 10 1d fe 08 00 	movsd  0x8fe(%rip),%xmm3        # 401298 <_IO_stdin_used+0x58>
  400999:	00 
  40099a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4009a0:	f2 0f 11 5c 24 18    	movsd  %xmm3,0x18(%rsp)
  4009a6:	f2 0f 11 54 24 10    	movsd  %xmm2,0x10(%rsp)
  4009ac:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
  4009b2:	e8 89 fd ff ff       	callq  400740 <rand@plt>
  4009b7:	89 c3                	mov    %eax,%ebx
  4009b9:	e8 82 fd ff ff       	callq  400740 <rand@plt>
  4009be:	41 89 c4             	mov    %eax,%r12d
  4009c1:	e8 7a fd ff ff       	callq  400740 <rand@plt>
  4009c6:	89 c5                	mov    %eax,%ebp
  4009c8:	e8 73 fd ff ff       	callq  400740 <rand@plt>
  4009cd:	41 89 c5             	mov    %eax,%r13d
  4009d0:	e8 6b fd ff ff       	callq  400740 <rand@plt>
  4009d5:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4009d9:	48 8b 35 c0 16 20 00 	mov    0x2016c0(%rip),%rsi        # 6020a0 <__Bodydata_buffer>
  4009e0:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  4009e6:	41 83 ee 01          	sub    $0x1,%r14d
  4009ea:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  4009f0:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  4009f4:	48 8d 4e 01          	lea    0x1(%rsi),%rcx
  4009f8:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  4009fe:	48 89 0d 9b 16 20 00 	mov    %rcx,0x20169b(%rip)        # 6020a0 <__Bodydata_buffer>
  400a05:	48 8d 14 cd b0 20 60 	lea    0x6020b0(,%rcx,8),%rdx
  400a0c:	00 
  400a0d:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a11:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a15:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400a19:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400a1d:	f2 0f 11 04 cd b0 20 	movsd  %xmm0,0x6020b0(,%rcx,8)
  400a24:	60 00 
  400a26:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a2a:	48 c7 82 00 dc 05 00 	movq   $0x0,0x5dc00(%rdx)
  400a31:	00 00 00 00 
  400a35:	48 c7 82 00 08 07 00 	movq   $0x0,0x70800(%rdx)
  400a3c:	00 00 00 00 
  400a40:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400a45:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a49:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a4d:	f2 0f 5c 05 4b 08 00 	subsd  0x84b(%rip),%xmm0        # 4012a0 <_IO_stdin_used+0x60>
  400a54:	00 
  400a55:	f2 0f 11 82 00 2c 01 	movsd  %xmm0,0x12c00(%rdx)
  400a5c:	00 
  400a5d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a61:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400a65:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a69:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a6d:	f2 0f 5c 05 2b 08 00 	subsd  0x82b(%rip),%xmm0        # 4012a0 <_IO_stdin_used+0x60>
  400a74:	00 
  400a75:	f2 0f 11 82 00 58 02 	movsd  %xmm0,0x25800(%rdx)
  400a7c:	00 
  400a7d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a81:	f2 41 0f 2a c5       	cvtsi2sd %r13d,%xmm0
  400a86:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a8a:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400a8e:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400a92:	f2 0f 11 82 00 84 03 	movsd  %xmm0,0x38400(%rdx)
  400a99:	00 
  400a9a:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a9e:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400aa2:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400aa6:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400aaa:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400aae:	f2 0f 11 82 00 b0 04 	movsd  %xmm0,0x4b000(%rdx)
  400ab5:	00 
  400ab6:	0f 85 e4 fe ff ff    	jne    4009a0 <_Z13instantiationv+0x30>
  400abc:	48 83 c4 20          	add    $0x20,%rsp
  400ac0:	5b                   	pop    %rbx
  400ac1:	5d                   	pop    %rbp
  400ac2:	41 5c                	pop    %r12
  400ac4:	41 5d                	pop    %r13
  400ac6:	41 5e                	pop    %r14
  400ac8:	c3                   	retq   
  400ac9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ad0 <_Z10run_simplev>:
  400ad0:	f2 0f 10 15 b8 07 00 	movsd  0x7b8(%rip),%xmm2        # 401290 <_IO_stdin_used+0x50>
  400ad7:	00 
  400ad8:	ba 20 4e 00 00       	mov    $0x4e20,%edx
  400add:	0f 1f 00             	nopl   (%rax)
  400ae0:	b8 b8 a4 63 00       	mov    $0x63a4b8,%eax
  400ae5:	0f 1f 00             	nopl   (%rax)
  400ae8:	f2 0f 10 88 00 58 02 	movsd  0x25800(%rax),%xmm1
  400aef:	00 
  400af0:	48 83 c0 08          	add    $0x8,%rax
  400af4:	f2 0f 10 80 f8 83 03 	movsd  0x383f8(%rax),%xmm0
  400afb:	00 
  400afc:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400b00:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400b04:	f2 0f 10 98 f8 7b fc 	movsd  -0x38408(%rax),%xmm3
  400b0b:	ff 
  400b0c:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400b10:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b14:	f2 0f 58 48 f8       	addsd  -0x8(%rax),%xmm1
  400b19:	f2 0f 11 48 f8       	movsd  %xmm1,-0x8(%rax)
  400b1e:	f2 0f 59 ca          	mulsd  %xmm2,%xmm1
  400b22:	f2 0f 58 80 f8 2b 01 	addsd  0x12bf8(%rax),%xmm0
  400b29:	00 
  400b2a:	f2 0f 58 88 f8 a7 fd 	addsd  -0x25808(%rax),%xmm1
  400b31:	ff 
  400b32:	f2 0f 11 80 f8 2b 01 	movsd  %xmm0,0x12bf8(%rax)
  400b39:	00 
  400b3a:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400b3e:	f2 0f 11 88 f8 a7 fd 	movsd  %xmm1,-0x25808(%rax)
  400b45:	ff 
  400b46:	f2 0f 58 80 f8 d3 fe 	addsd  -0x12c08(%rax),%xmm0
  400b4d:	ff 
  400b4e:	f2 0f 11 80 f8 d3 fe 	movsd  %xmm0,-0x12c08(%rax)
  400b55:	ff 
  400b56:	48 3d b8 d0 64 00    	cmp    $0x64d0b8,%rax
  400b5c:	75 8a                	jne    400ae8 <_Z10run_simplev+0x18>
  400b5e:	83 ea 01             	sub    $0x1,%edx
  400b61:	0f 85 79 ff ff ff    	jne    400ae0 <_Z10run_simplev+0x10>
  400b67:	f3 c3                	repz retq 
  400b69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400b70 <_Z14run_simulationv>:
  400b70:	41 57                	push   %r15
  400b72:	41 56                	push   %r14
  400b74:	be 02 00 00 00       	mov    $0x2,%esi
  400b79:	41 55                	push   %r13
  400b7b:	41 54                	push   %r12
  400b7d:	55                   	push   %rbp
  400b7e:	53                   	push   %rbx
  400b7f:	48 83 ec 58          	sub    $0x58,%rsp
  400b83:	48 8b 15 16 15 20 00 	mov    0x201516(%rip),%rdx        # 6020a0 <__Bodydata_buffer>
  400b8a:	f2 0f 10 3d 2e 07 00 	movsd  0x72e(%rip),%xmm7        # 4012c0 <_IO_stdin_used+0x80>
  400b91:	00 
  400b92:	f2 0f 10 35 0e 07 00 	movsd  0x70e(%rip),%xmm6        # 4012a8 <_IO_stdin_used+0x68>
  400b99:	00 
  400b9a:	66 44 0f 28 c7       	movapd %xmm7,%xmm8
  400b9f:	48 85 d2             	test   %rdx,%rdx
  400ba2:	0f 84 40 02 00 00    	je     400de8 <_Z14run_simulationv+0x278>
  400ba8:	31 c0                	xor    %eax,%eax
  400baa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400bb0:	48 83 c0 01          	add    $0x1,%rax
  400bb4:	48 8d 0c c5 b0 20 60 	lea    0x6020b0(,%rax,8),%rcx
  400bbb:	00 
  400bbc:	48 39 d0             	cmp    %rdx,%rax
  400bbf:	48 c7 81 00 dc 05 00 	movq   $0x0,0x5dc00(%rcx)
  400bc6:	00 00 00 00 
  400bca:	48 c7 81 00 08 07 00 	movq   $0x0,0x70800(%rcx)
  400bd1:	00 00 00 00 
  400bd5:	75 d9                	jne    400bb0 <_Z14run_simulationv+0x40>
  400bd7:	f2 0f 10 2d d1 06 00 	movsd  0x6d1(%rip),%xmm5        # 4012b0 <_IO_stdin_used+0x70>
  400bde:	00 
  400bdf:	45 31 ed             	xor    %r13d,%r13d
  400be2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400be8:	49 83 c5 01          	add    $0x1,%r13
  400bec:	31 ed                	xor    %ebp,%ebp
  400bee:	4e 8d 34 ed b0 20 60 	lea    0x6020b0(,%r13,8),%r14
  400bf5:	00 
  400bf6:	49 8d 86 00 2c 01 00 	lea    0x12c00(%r14),%rax
  400bfd:	4d 8d be 00 58 02 00 	lea    0x25800(%r14),%r15
  400c04:	48 39 d5             	cmp    %rdx,%rbp
  400c07:	0f 83 bc 00 00 00    	jae    400cc9 <_Z14run_simulationv+0x159>
  400c0d:	48 83 c5 01          	add    $0x1,%rbp
  400c11:	49 39 ed             	cmp    %rbp,%r13
  400c14:	74 ee                	je     400c04 <_Z14run_simulationv+0x94>
  400c16:	4c 8d 24 ed 00 00 00 	lea    0x0(,%rbp,8),%r12
  400c1d:	00 
  400c1e:	f2 41 0f 10 0f       	movsd  (%r15),%xmm1
  400c23:	f2 0f 10 18          	movsd  (%rax),%xmm3
  400c27:	49 8d 9c 24 b0 20 60 	lea    0x6020b0(%r12),%rbx
  400c2e:	00 
  400c2f:	f2 0f 5c 8b 00 58 02 	subsd  0x25800(%rbx),%xmm1
  400c36:	00 
  400c37:	f2 0f 5c 9b 00 2c 01 	subsd  0x12c00(%rbx),%xmm3
  400c3e:	00 
  400c3f:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400c44:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400c48:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400c4d:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400c51:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400c56:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400c5a:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400c5e:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400c62:	0f 8a 98 01 00 00    	jp     400e00 <_Z14run_simulationv+0x290>
  400c68:	f2 41 0f 10 94 24 b0 	movsd  0x6020b0(%r12),%xmm2
  400c6f:	20 60 00 
  400c72:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400c76:	48 39 d5             	cmp    %rdx,%rbp
  400c79:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400c7d:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c81:	f2 41 0f 59 16       	mulsd  (%r14),%xmm2
  400c86:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400c8a:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400c8e:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400c92:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400c96:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400c9b:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400c9f:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400ca3:	f2 0f 58 9b 00 dc 05 	addsd  0x5dc00(%rbx),%xmm3
  400caa:	00 
  400cab:	f2 0f 11 9b 00 dc 05 	movsd  %xmm3,0x5dc00(%rbx)
  400cb2:	00 
  400cb3:	f2 0f 58 8b 00 08 07 	addsd  0x70800(%rbx),%xmm1
  400cba:	00 
  400cbb:	f2 0f 11 8b 00 08 07 	movsd  %xmm1,0x70800(%rbx)
  400cc2:	00 
  400cc3:	0f 82 44 ff ff ff    	jb     400c0d <_Z14run_simulationv+0x9d>
  400cc9:	49 39 d5             	cmp    %rdx,%r13
  400ccc:	0f 82 16 ff ff ff    	jb     400be8 <_Z14run_simulationv+0x78>
  400cd2:	48 85 d2             	test   %rdx,%rdx
  400cd5:	0f 84 0d 01 00 00    	je     400de8 <_Z14run_simulationv+0x278>
  400cdb:	f2 0f 10 1d bd 05 00 	movsd  0x5bd(%rip),%xmm3        # 4012a0 <_IO_stdin_used+0x60>
  400ce2:	00 
  400ce3:	31 c9                	xor    %ecx,%ecx
  400ce5:	f2 0f 10 0d a3 05 00 	movsd  0x5a3(%rip),%xmm1        # 401290 <_IO_stdin_used+0x50>
  400cec:	00 
  400ced:	f2 0f 10 15 c3 05 00 	movsd  0x5c3(%rip),%xmm2        # 4012b8 <_IO_stdin_used+0x78>
  400cf4:	00 
  400cf5:	66 0f 28 e3          	movapd %xmm3,%xmm4
  400cf9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400d00:	48 83 c1 01          	add    $0x1,%rcx
  400d04:	48 8d 04 cd b0 20 60 	lea    0x6020b0(,%rcx,8),%rax
  400d0b:	00 
  400d0c:	f2 0f 10 80 00 dc 05 	movsd  0x5dc00(%rax),%xmm0
  400d13:	00 
  400d14:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400d18:	f2 0f 5e 04 cd b0 20 	divsd  0x6020b0(,%rcx,8),%xmm0
  400d1f:	60 00 
  400d21:	f2 0f 58 80 00 84 03 	addsd  0x38400(%rax),%xmm0
  400d28:	00 
  400d29:	f2 0f 11 80 00 84 03 	movsd  %xmm0,0x38400(%rax)
  400d30:	00 
  400d31:	f2 0f 10 80 00 08 07 	movsd  0x70800(%rax),%xmm0
  400d38:	00 
  400d39:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400d3d:	f2 0f 5e 04 cd b0 20 	divsd  0x6020b0(,%rcx,8),%xmm0
  400d44:	60 00 
  400d46:	f2 0f 58 80 00 b0 04 	addsd  0x4b000(%rax),%xmm0
  400d4d:	00 
  400d4e:	f2 0f 11 80 00 b0 04 	movsd  %xmm0,0x4b000(%rax)
  400d55:	00 
  400d56:	f2 0f 10 80 00 84 03 	movsd  0x38400(%rax),%xmm0
  400d5d:	00 
  400d5e:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400d62:	f2 0f 58 80 00 2c 01 	addsd  0x12c00(%rax),%xmm0
  400d69:	00 
  400d6a:	f2 0f 11 80 00 2c 01 	movsd  %xmm0,0x12c00(%rax)
  400d71:	00 
  400d72:	f2 0f 10 80 00 b0 04 	movsd  0x4b000(%rax),%xmm0
  400d79:	00 
  400d7a:	f2 0f 10 a8 00 2c 01 	movsd  0x12c00(%rax),%xmm5
  400d81:	00 
  400d82:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400d86:	66 0f 2e d5          	ucomisd %xmm5,%xmm2
  400d8a:	f2 0f 58 80 00 58 02 	addsd  0x25800(%rax),%xmm0
  400d91:	00 
  400d92:	f2 0f 11 80 00 58 02 	movsd  %xmm0,0x25800(%rax)
  400d99:	00 
  400d9a:	77 06                	ja     400da2 <_Z14run_simulationv+0x232>
  400d9c:	66 0f 2e eb          	ucomisd %xmm3,%xmm5
  400da0:	76 1c                	jbe    400dbe <_Z14run_simulationv+0x24e>
  400da2:	f2 0f 10 80 00 84 03 	movsd  0x38400(%rax),%xmm0
  400da9:	00 
  400daa:	66 0f 57 c7          	xorpd  %xmm7,%xmm0
  400dae:	f2 0f 11 80 00 84 03 	movsd  %xmm0,0x38400(%rax)
  400db5:	00 
  400db6:	f2 0f 10 80 00 58 02 	movsd  0x25800(%rax),%xmm0
  400dbd:	00 
  400dbe:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  400dc2:	77 06                	ja     400dca <_Z14run_simulationv+0x25a>
  400dc4:	66 0f 2e c4          	ucomisd %xmm4,%xmm0
  400dc8:	76 15                	jbe    400ddf <_Z14run_simulationv+0x26f>
  400dca:	f2 0f 10 80 00 b0 04 	movsd  0x4b000(%rax),%xmm0
  400dd1:	00 
  400dd2:	66 41 0f 57 c0       	xorpd  %xmm8,%xmm0
  400dd7:	f2 0f 11 80 00 b0 04 	movsd  %xmm0,0x4b000(%rax)
  400dde:	00 
  400ddf:	48 39 d1             	cmp    %rdx,%rcx
  400de2:	0f 85 18 ff ff ff    	jne    400d00 <_Z14run_simulationv+0x190>
  400de8:	83 ee 01             	sub    $0x1,%esi
  400deb:	0f 85 ae fd ff ff    	jne    400b9f <_Z14run_simulationv+0x2f>
  400df1:	48 83 c4 58          	add    $0x58,%rsp
  400df5:	5b                   	pop    %rbx
  400df6:	5d                   	pop    %rbp
  400df7:	41 5c                	pop    %r12
  400df9:	41 5d                	pop    %r13
  400dfb:	41 5e                	pop    %r14
  400dfd:	41 5f                	pop    %r15
  400dff:	c3                   	retq   
  400e00:	44 0f 29 44 24 40    	movaps %xmm8,0x40(%rsp)
  400e06:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400e0a:	48 89 04 24          	mov    %rax,(%rsp)
  400e0e:	0f 29 7c 24 30       	movaps %xmm7,0x30(%rsp)
  400e13:	f2 0f 11 74 24 28    	movsd  %xmm6,0x28(%rsp)
  400e19:	f2 0f 11 6c 24 20    	movsd  %xmm5,0x20(%rsp)
  400e1f:	f2 44 0f 11 4c 24 10 	movsd  %xmm9,0x10(%rsp)
  400e26:	f2 0f 11 5c 24 08    	movsd  %xmm3,0x8(%rsp)
  400e2c:	e8 1f f9 ff ff       	callq  400750 <sqrt@plt>
  400e31:	48 8b 15 68 12 20 00 	mov    0x201268(%rip),%rdx        # 6020a0 <__Bodydata_buffer>
  400e38:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400e3c:	66 44 0f 28 44 24 40 	movapd 0x40(%rsp),%xmm8
  400e43:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400e47:	48 8b 04 24          	mov    (%rsp),%rax
  400e4b:	66 0f 28 7c 24 30    	movapd 0x30(%rsp),%xmm7
  400e51:	f2 0f 10 74 24 28    	movsd  0x28(%rsp),%xmm6
  400e57:	f2 0f 10 6c 24 20    	movsd  0x20(%rsp),%xmm5
  400e5d:	f2 44 0f 10 4c 24 10 	movsd  0x10(%rsp),%xmm9
  400e64:	f2 0f 10 5c 24 08    	movsd  0x8(%rsp),%xmm3
  400e6a:	e9 f9 fd ff ff       	jmpq   400c68 <_Z14run_simulationv+0xf8>
  400e6f:	90                   	nop

0000000000400e70 <_Z12float_as_intf>:
  400e70:	66 0f 7e c0          	movd   %xmm0,%eax
  400e74:	c3                   	retq   
  400e75:	90                   	nop
  400e76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e7d:	00 00 00 

0000000000400e80 <_ZN4Body7storageEv>:
  400e80:	b8 a0 20 60 00       	mov    $0x6020a0,%eax
  400e85:	c3                   	retq   
  400e86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e8d:	00 00 00 

0000000000400e90 <_Z19codegen_reset_forceP4Body>:
  400e90:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400e94:	48 8d 04 fd b0 20 60 	lea    0x6020b0(,%rdi,8),%rax
  400e9b:	00 
  400e9c:	f2 0f 11 80 00 dc 05 	movsd  %xmm0,0x5dc00(%rax)
  400ea3:	00 
  400ea4:	f2 0f 11 80 00 08 07 	movsd  %xmm0,0x70800(%rax)
  400eab:	00 
  400eac:	c3                   	retq   
  400ead:	0f 1f 00             	nopl   (%rax)

0000000000400eb0 <_Z17codegen_add_forceP4BodyS0_>:
  400eb0:	48 39 f7             	cmp    %rsi,%rdi
  400eb3:	0f 84 cb 00 00 00    	je     400f84 <_Z17codegen_add_forceP4BodyS0_+0xd4>
  400eb9:	41 54                	push   %r12
  400ebb:	55                   	push   %rbp
  400ebc:	48 8d 04 f5 b0 20 60 	lea    0x6020b0(,%rsi,8),%rax
  400ec3:	00 
  400ec4:	53                   	push   %rbx
  400ec5:	48 8d 1c fd b0 20 60 	lea    0x6020b0(,%rdi,8),%rbx
  400ecc:	00 
  400ecd:	48 89 f5             	mov    %rsi,%rbp
  400ed0:	49 89 fc             	mov    %rdi,%r12
  400ed3:	48 83 ec 10          	sub    $0x10,%rsp
  400ed7:	f2 0f 10 88 00 58 02 	movsd  0x25800(%rax),%xmm1
  400ede:	00 
  400edf:	f2 0f 10 98 00 2c 01 	movsd  0x12c00(%rax),%xmm3
  400ee6:	00 
  400ee7:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400eeb:	f2 0f 5c 9b 00 2c 01 	subsd  0x12c00(%rbx),%xmm3
  400ef2:	00 
  400ef3:	f2 0f 5c ab 00 58 02 	subsd  0x25800(%rbx),%xmm5
  400efa:	00 
  400efb:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400eff:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400f03:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400f07:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400f0b:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400f0f:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400f13:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400f17:	7a 6d                	jp     400f86 <_Z17codegen_add_forceP4BodyS0_+0xd6>
  400f19:	f2 0f 10 15 87 03 00 	movsd  0x387(%rip),%xmm2        # 4012a8 <_IO_stdin_used+0x68>
  400f20:	00 
  400f21:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400f25:	f2 42 0f 59 14 e5 b0 	mulsd  0x6020b0(,%r12,8),%xmm2
  400f2c:	20 60 00 
  400f2f:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400f33:	f2 0f 58 05 75 03 00 	addsd  0x375(%rip),%xmm0        # 4012b0 <_IO_stdin_used+0x70>
  400f3a:	00 
  400f3b:	f2 0f 59 14 ed b0 20 	mulsd  0x6020b0(,%rbp,8),%xmm2
  400f42:	60 00 
  400f44:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400f48:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400f4c:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400f50:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  400f54:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400f58:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400f5c:	f2 0f 58 9b 00 dc 05 	addsd  0x5dc00(%rbx),%xmm3
  400f63:	00 
  400f64:	f2 0f 11 9b 00 dc 05 	movsd  %xmm3,0x5dc00(%rbx)
  400f6b:	00 
  400f6c:	f2 0f 58 8b 00 08 07 	addsd  0x70800(%rbx),%xmm1
  400f73:	00 
  400f74:	f2 0f 11 8b 00 08 07 	movsd  %xmm1,0x70800(%rbx)
  400f7b:	00 
  400f7c:	48 83 c4 10          	add    $0x10,%rsp
  400f80:	5b                   	pop    %rbx
  400f81:	5d                   	pop    %rbp
  400f82:	41 5c                	pop    %r12
  400f84:	f3 c3                	repz retq 
  400f86:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  400f8c:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400f91:	e8 ba f7 ff ff       	callq  400750 <sqrt@plt>
  400f96:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  400f9c:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400fa0:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400fa5:	e9 6f ff ff ff       	jmpq   400f19 <_Z17codegen_add_forceP4BodyS0_+0x69>
  400faa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400fb0 <_Z14codegen_updateP4Bodyd>:
  400fb0:	48 8d 04 fd b0 20 60 	lea    0x6020b0(,%rdi,8),%rax
  400fb7:	00 
  400fb8:	f2 0f 10 15 f8 02 00 	movsd  0x2f8(%rip),%xmm2        # 4012b8 <_IO_stdin_used+0x78>
  400fbf:	00 
  400fc0:	f2 0f 10 88 00 dc 05 	movsd  0x5dc00(%rax),%xmm1
  400fc7:	00 
  400fc8:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400fcc:	f2 0f 5e 0c fd b0 20 	divsd  0x6020b0(,%rdi,8),%xmm1
  400fd3:	60 00 
  400fd5:	f2 0f 58 88 00 84 03 	addsd  0x38400(%rax),%xmm1
  400fdc:	00 
  400fdd:	f2 0f 11 88 00 84 03 	movsd  %xmm1,0x38400(%rax)
  400fe4:	00 
  400fe5:	f2 0f 10 88 00 08 07 	movsd  0x70800(%rax),%xmm1
  400fec:	00 
  400fed:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  400ff1:	f2 0f 5e 0c fd b0 20 	divsd  0x6020b0(,%rdi,8),%xmm1
  400ff8:	60 00 
  400ffa:	f2 0f 58 88 00 b0 04 	addsd  0x4b000(%rax),%xmm1
  401001:	00 
  401002:	f2 0f 11 88 00 b0 04 	movsd  %xmm1,0x4b000(%rax)
  401009:	00 
  40100a:	f2 0f 10 88 00 84 03 	movsd  0x38400(%rax),%xmm1
  401011:	00 
  401012:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401016:	f2 0f 59 80 00 b0 04 	mulsd  0x4b000(%rax),%xmm0
  40101d:	00 
  40101e:	f2 0f 58 88 00 2c 01 	addsd  0x12c00(%rax),%xmm1
  401025:	00 
  401026:	f2 0f 58 80 00 58 02 	addsd  0x25800(%rax),%xmm0
  40102d:	00 
  40102e:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  401032:	f2 0f 11 88 00 2c 01 	movsd  %xmm1,0x12c00(%rax)
  401039:	00 
  40103a:	f2 0f 11 80 00 58 02 	movsd  %xmm0,0x25800(%rax)
  401041:	00 
  401042:	77 0a                	ja     40104e <_Z14codegen_updateP4Bodyd+0x9e>
  401044:	66 0f 2e 0d 54 02 00 	ucomisd 0x254(%rip),%xmm1        # 4012a0 <_IO_stdin_used+0x60>
  40104b:	00 
  40104c:	76 24                	jbe    401072 <_Z14codegen_updateP4Bodyd+0xc2>
  40104e:	f2 0f 10 80 00 84 03 	movsd  0x38400(%rax),%xmm0
  401055:	00 
  401056:	f2 0f 10 0d 62 02 00 	movsd  0x262(%rip),%xmm1        # 4012c0 <_IO_stdin_used+0x80>
  40105d:	00 
  40105e:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401062:	f2 0f 11 80 00 84 03 	movsd  %xmm0,0x38400(%rax)
  401069:	00 
  40106a:	f2 0f 10 80 00 58 02 	movsd  0x25800(%rax),%xmm0
  401071:	00 
  401072:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  401076:	77 0a                	ja     401082 <_Z14codegen_updateP4Bodyd+0xd2>
  401078:	66 0f 2e 05 20 02 00 	ucomisd 0x220(%rip),%xmm0        # 4012a0 <_IO_stdin_used+0x60>
  40107f:	00 
  401080:	76 1e                	jbe    4010a0 <_Z14codegen_updateP4Bodyd+0xf0>
  401082:	f2 0f 10 80 00 b0 04 	movsd  0x4b000(%rax),%xmm0
  401089:	00 
  40108a:	f2 0f 10 0d 2e 02 00 	movsd  0x22e(%rip),%xmm1        # 4012c0 <_IO_stdin_used+0x80>
  401091:	00 
  401092:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401096:	f2 0f 11 80 00 b0 04 	movsd  %xmm0,0x4b000(%rax)
  40109d:	00 
  40109e:	c3                   	retq   
  40109f:	90                   	nop
  4010a0:	f3 c3                	repz retq 
  4010a2:	0f 1f 40 00          	nopl   0x0(%rax)
  4010a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4010ad:	00 00 00 

00000000004010b0 <_Z26codegen_reset_force_manualP4Body>:
  4010b0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4010b4:	48 8d 04 fd b0 20 60 	lea    0x6020b0(,%rdi,8),%rax
  4010bb:	00 
  4010bc:	f2 0f 11 80 00 dc 05 	movsd  %xmm0,0x5dc00(%rax)
  4010c3:	00 
  4010c4:	f2 0f 11 80 00 08 07 	movsd  %xmm0,0x70800(%rax)
  4010cb:	00 
  4010cc:	c3                   	retq   
  4010cd:	0f 1f 00             	nopl   (%rax)

00000000004010d0 <_Z24codegen_reset_force_halfP4Body>:
  4010d0:	48 c7 04 fd b0 fc 65 	movq   $0x0,0x65fcb0(,%rdi,8)
  4010d7:	00 00 00 00 00 
  4010dc:	c3                   	retq   
  4010dd:	0f 1f 00             	nopl   (%rax)

00000000004010e0 <_Z18codegen_reset_massP4Body>:
  4010e0:	48 c7 04 fd b0 20 60 	movq   $0x0,0x6020b0(,%rdi,8)
  4010e7:	00 00 00 00 00 
  4010ec:	c3                   	retq   
  4010ed:	0f 1f 00             	nopl   (%rax)

00000000004010f0 <_Z22codengen_simple_updateP4Bodyd>:
  4010f0:	48 8d 04 fd b0 20 60 	lea    0x6020b0(,%rdi,8),%rax
  4010f7:	00 
  4010f8:	f2 0f 10 88 00 dc 05 	movsd  0x5dc00(%rax),%xmm1
  4010ff:	00 
  401100:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401104:	f2 0f 5e 0c fd b0 20 	divsd  0x6020b0(,%rdi,8),%xmm1
  40110b:	60 00 
  40110d:	f2 0f 58 88 00 84 03 	addsd  0x38400(%rax),%xmm1
  401114:	00 
  401115:	f2 0f 11 88 00 84 03 	movsd  %xmm1,0x38400(%rax)
  40111c:	00 
  40111d:	f2 0f 10 88 00 08 07 	movsd  0x70800(%rax),%xmm1
  401124:	00 
  401125:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401129:	f2 0f 5e 0c fd b0 20 	divsd  0x6020b0(,%rdi,8),%xmm1
  401130:	60 00 
  401132:	f2 0f 58 88 00 b0 04 	addsd  0x4b000(%rax),%xmm1
  401139:	00 
  40113a:	f2 0f 11 88 00 b0 04 	movsd  %xmm1,0x4b000(%rax)
  401141:	00 
  401142:	f2 0f 10 88 00 84 03 	movsd  0x38400(%rax),%xmm1
  401149:	00 
  40114a:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40114e:	f2 0f 59 80 00 b0 04 	mulsd  0x4b000(%rax),%xmm0
  401155:	00 
  401156:	f2 0f 58 88 00 2c 01 	addsd  0x12c00(%rax),%xmm1
  40115d:	00 
  40115e:	f2 0f 58 80 00 58 02 	addsd  0x25800(%rax),%xmm0
  401165:	00 
  401166:	f2 0f 11 88 00 2c 01 	movsd  %xmm1,0x12c00(%rax)
  40116d:	00 
  40116e:	f2 0f 11 80 00 58 02 	movsd  %xmm0,0x25800(%rax)
  401175:	00 
  401176:	c3                   	retq   
  401177:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40117e:	00 00 

0000000000401180 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401180:	55                   	push   %rbp
  401181:	53                   	push   %rbx
  401182:	48 89 fd             	mov    %rdi,%rbp
  401185:	48 83 ec 08          	sub    $0x8,%rsp
  401189:	e8 a2 f5 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40118e:	48 89 c3             	mov    %rax,%rbx
  401191:	ff d5                	callq  *%rbp
  401193:	e8 98 f5 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401198:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40119f:	de 1b 43 
  4011a2:	48 29 d8             	sub    %rbx,%rax
  4011a5:	48 83 c4 08          	add    $0x8,%rsp
  4011a9:	48 89 c1             	mov    %rax,%rcx
  4011ac:	48 f7 ea             	imul   %rdx
  4011af:	48 c1 f9 3f          	sar    $0x3f,%rcx
  4011b3:	5b                   	pop    %rbx
  4011b4:	5d                   	pop    %rbp
  4011b5:	48 c1 fa 12          	sar    $0x12,%rdx
  4011b9:	48 89 d0             	mov    %rdx,%rax
  4011bc:	48 29 c8             	sub    %rcx,%rax
  4011bf:	c3                   	retq   

00000000004011c0 <__libc_csu_init>:
  4011c0:	41 57                	push   %r15
  4011c2:	41 56                	push   %r14
  4011c4:	41 89 ff             	mov    %edi,%r15d
  4011c7:	41 55                	push   %r13
  4011c9:	41 54                	push   %r12
  4011cb:	4c 8d 25 16 0c 20 00 	lea    0x200c16(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  4011d2:	55                   	push   %rbp
  4011d3:	48 8d 2d 1e 0c 20 00 	lea    0x200c1e(%rip),%rbp        # 601df8 <__init_array_end>
  4011da:	53                   	push   %rbx
  4011db:	49 89 f6             	mov    %rsi,%r14
  4011de:	49 89 d5             	mov    %rdx,%r13
  4011e1:	4c 29 e5             	sub    %r12,%rbp
  4011e4:	48 83 ec 08          	sub    $0x8,%rsp
  4011e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4011ec:	e8 b7 f4 ff ff       	callq  4006a8 <_init>
  4011f1:	48 85 ed             	test   %rbp,%rbp
  4011f4:	74 20                	je     401216 <__libc_csu_init+0x56>
  4011f6:	31 db                	xor    %ebx,%ebx
  4011f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4011ff:	00 
  401200:	4c 89 ea             	mov    %r13,%rdx
  401203:	4c 89 f6             	mov    %r14,%rsi
  401206:	44 89 ff             	mov    %r15d,%edi
  401209:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40120d:	48 83 c3 01          	add    $0x1,%rbx
  401211:	48 39 eb             	cmp    %rbp,%rbx
  401214:	75 ea                	jne    401200 <__libc_csu_init+0x40>
  401216:	48 83 c4 08          	add    $0x8,%rsp
  40121a:	5b                   	pop    %rbx
  40121b:	5d                   	pop    %rbp
  40121c:	41 5c                	pop    %r12
  40121e:	41 5d                	pop    %r13
  401220:	41 5e                	pop    %r14
  401222:	41 5f                	pop    %r15
  401224:	c3                   	retq   
  401225:	90                   	nop
  401226:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40122d:	00 00 00 

0000000000401230 <__libc_csu_fini>:
  401230:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401234 <_fini>:
  401234:	48 83 ec 08          	sub    $0x8,%rsp
  401238:	48 83 c4 08          	add    $0x8,%rsp
  40123c:	c3                   	retq   
