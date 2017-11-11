
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
  40077b:	48 c7 05 fa 18 20 00 	movq   $0x0,0x2018fa(%rip)        # 602080 <__Bodydata_buffer>
  400782:	00 00 00 00 
  400786:	48 c7 05 f7 18 20 00 	movq   $0x0,0x2018f7(%rip)        # 602088 <__Bodydata_buffer+0x8>
  40078d:	00 00 00 00 
  400791:	e8 0a 0b 00 00       	callq  4012a0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400796:	bf 50 0c 40 00       	mov    $0x400c50,%edi
  40079b:	48 89 c3             	mov    %rax,%rbx
  40079e:	e8 fd 0a 00 00       	callq  4012a0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007a3:	bf d0 0a 40 00       	mov    $0x400ad0,%edi
  4007a8:	48 89 c5             	mov    %rax,%rbp
  4007ab:	e8 f0 0a 00 00       	callq  4012a0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007b0:	b9 98 1a 61 00       	mov    $0x611a98,%ecx
  4007b5:	49 89 c0             	mov    %rax,%r8
  4007b8:	bf 98 14 62 00       	mov    $0x621498,%edi
  4007bd:	ba 0b 00 00 00       	mov    $0xb,%edx
  4007c2:	be 8d 68 b7 6c       	mov    $0x6cb7688d,%esi
  4007c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4007ce:	00 00 
  4007d0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4007d4:	48 83 c1 08          	add    $0x8,%rcx
  4007d8:	66 0f ef c9          	pxor   %xmm1,%xmm1
  4007dc:	f2 0f 5a 41 f8       	cvtsd2ss -0x8(%rcx),%xmm0
  4007e1:	66 41 0f 7e c1       	movd   %xmm0,%r9d
  4007e6:	f2 0f 5a 89 f8 f9 00 	cvtsd2ss 0xf9f8(%rcx),%xmm1
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
  400820:	be 68 13 40 00       	mov    $0x401368,%esi
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
  400844:	bf 90 f6 66 00       	mov    $0x66f690,%edi
  400849:	e8 a2 fe ff ff       	callq  4006f0 <_ZNSt8ios_base4InitC1Ev@plt>
  40084e:	ba 60 20 60 00       	mov    $0x602060,%edx
  400853:	be 90 f6 66 00       	mov    $0x66f690,%esi
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
  40087f:	49 c7 c0 50 13 40 00 	mov    $0x401350,%r8
  400886:	48 c7 c1 e0 12 40 00 	mov    $0x4012e0,%rcx
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
  400920:	80 3d 49 17 20 00 00 	cmpb   $0x0,0x201749(%rip)        # 602070 <completed.7585>
  400927:	75 11                	jne    40093a <__do_global_dtors_aux+0x1a>
  400929:	55                   	push   %rbp
  40092a:	48 89 e5             	mov    %rsp,%rbp
  40092d:	e8 6e ff ff ff       	callq  4008a0 <deregister_tm_clones>
  400932:	5d                   	pop    %rbp
  400933:	c6 05 36 17 20 00 01 	movb   $0x1,0x201736(%rip)        # 602070 <completed.7585>
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
  400974:	41 be 40 1f 00 00    	mov    $0x1f40,%r14d
  40097a:	41 54                	push   %r12
  40097c:	55                   	push   %rbp
  40097d:	53                   	push   %rbx
  40097e:	48 83 ec 20          	sub    $0x20,%rsp
  400982:	f2 0f 10 0d 1e 0a 00 	movsd  0xa1e(%rip),%xmm1        # 4013a8 <_IO_stdin_used+0x48>
  400989:	00 
  40098a:	f2 0f 10 15 1e 0a 00 	movsd  0xa1e(%rip),%xmm2        # 4013b0 <_IO_stdin_used+0x50>
  400991:	00 
  400992:	f2 0f 10 1d 1e 0a 00 	movsd  0xa1e(%rip),%xmm3        # 4013b8 <_IO_stdin_used+0x58>
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
  4009d9:	48 8b 15 a0 16 20 00 	mov    0x2016a0(%rip),%rdx        # 602080 <__Bodydata_buffer>
  4009e0:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  4009e6:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  4009ec:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  4009f0:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  4009f4:	48 83 c2 03          	add    $0x3,%rdx
  4009f8:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  4009fe:	41 83 ee 01          	sub    $0x1,%r14d
  400a02:	48 89 0d 77 16 20 00 	mov    %rcx,0x201677(%rip)        # 602080 <__Bodydata_buffer>
  400a09:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a0d:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a11:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400a15:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400a19:	f2 0f 11 04 d5 80 20 	movsd  %xmm0,0x602080(,%rdx,8)
  400a20:	60 00 
  400a22:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a26:	48 8d 14 d5 80 20 60 	lea    0x602080(,%rdx,8),%rdx
  400a2d:	00 
  400a2e:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400a33:	48 c7 82 00 e2 04 00 	movq   $0x0,0x4e200(%rdx)
  400a3a:	00 00 00 00 
  400a3e:	48 c7 82 00 dc 05 00 	movq   $0x0,0x5dc00(%rdx)
  400a45:	00 00 00 00 
  400a49:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a4d:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a51:	f2 0f 5c 05 67 09 00 	subsd  0x967(%rip),%xmm0        # 4013c0 <_IO_stdin_used+0x60>
  400a58:	00 
  400a59:	f2 0f 11 82 00 fa 00 	movsd  %xmm0,0xfa00(%rdx)
  400a60:	00 
  400a61:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a65:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400a69:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a6d:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a71:	f2 0f 5c 05 47 09 00 	subsd  0x947(%rip),%xmm0        # 4013c0 <_IO_stdin_used+0x60>
  400a78:	00 
  400a79:	f2 0f 11 82 00 f4 01 	movsd  %xmm0,0x1f400(%rdx)
  400a80:	00 
  400a81:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a85:	f2 41 0f 2a c5       	cvtsi2sd %r13d,%xmm0
  400a8a:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a8e:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400a92:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400a96:	f2 0f 11 82 00 ee 02 	movsd  %xmm0,0x2ee00(%rdx)
  400a9d:	00 
  400a9e:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aa2:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400aa6:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400aaa:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400aae:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400ab2:	f2 0f 11 82 00 e8 03 	movsd  %xmm0,0x3e800(%rdx)
  400ab9:	00 
  400aba:	0f 85 e0 fe ff ff    	jne    4009a0 <_Z13instantiationv+0x30>
  400ac0:	48 83 c4 20          	add    $0x20,%rsp
  400ac4:	5b                   	pop    %rbx
  400ac5:	5d                   	pop    %rbp
  400ac6:	41 5c                	pop    %r12
  400ac8:	41 5d                	pop    %r13
  400aca:	41 5e                	pop    %r14
  400acc:	c3                   	retq   
  400acd:	0f 1f 00             	nopl   (%rax)

0000000000400ad0 <_Z10run_simplev>:
  400ad0:	f2 0f 10 25 d8 08 00 	movsd  0x8d8(%rip),%xmm4        # 4013b0 <_IO_stdin_used+0x50>
  400ad7:	00 
  400ad8:	ba 50 c3 00 00       	mov    $0xc350,%edx
  400add:	66 0f 28 15 fb 08 00 	movapd 0x8fb(%rip),%xmm2        # 4013e0 <_IO_stdin_used+0x80>
  400ae4:	00 
  400ae5:	0f 1f 00             	nopl   (%rax)
  400ae8:	b8 a0 0e 63 00       	mov    $0x630ea0,%eax
  400aed:	f2 0f 10 0d a3 f7 24 	movsd  0x24f7a3(%rip),%xmm1        # 650298 <__Bodydata_buffer+0x4e218>
  400af4:	00 
  400af5:	f2 0f 10 05 9b f1 25 	movsd  0x25f19b(%rip),%xmm0        # 65fc98 <__Bodydata_buffer+0x5dc18>
  400afc:	00 
  400afd:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b01:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400b05:	f2 0f 10 1d 8b 15 20 	movsd  0x20158b(%rip),%xmm3        # 602098 <__Bodydata_buffer+0x18>
  400b0c:	00 
  400b0d:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400b11:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b15:	f2 0f 58 0d 7b 03 23 	addsd  0x23037b(%rip),%xmm1        # 630e98 <__Bodydata_buffer+0x2ee18>
  400b1c:	00 
  400b1d:	f2 0f 11 0d 73 03 23 	movsd  %xmm1,0x230373(%rip)        # 630e98 <__Bodydata_buffer+0x2ee18>
  400b24:	00 
  400b25:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b29:	f2 0f 58 05 67 fd 23 	addsd  0x23fd67(%rip),%xmm0        # 640898 <__Bodydata_buffer+0x3e818>
  400b30:	00 
  400b31:	f2 0f 58 0d 5f 0f 21 	addsd  0x210f5f(%rip),%xmm1        # 611a98 <__Bodydata_buffer+0xfa18>
  400b38:	00 
  400b39:	f2 0f 11 05 57 fd 23 	movsd  %xmm0,0x23fd57(%rip)        # 640898 <__Bodydata_buffer+0x3e818>
  400b40:	00 
  400b41:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400b45:	f2 0f 11 0d 4b 0f 21 	movsd  %xmm1,0x210f4b(%rip)        # 611a98 <__Bodydata_buffer+0xfa18>
  400b4c:	00 
  400b4d:	f2 0f 58 05 43 09 22 	addsd  0x220943(%rip),%xmm0        # 621498 <__Bodydata_buffer+0x1f418>
  400b54:	00 
  400b55:	f2 0f 11 05 3b 09 22 	movsd  %xmm0,0x22093b(%rip)        # 621498 <__Bodydata_buffer+0x1f418>
  400b5c:	00 
  400b5d:	0f 1f 00             	nopl   (%rax)
  400b60:	66 0f 28 88 00 f4 01 	movapd 0x1f400(%rax),%xmm1
  400b67:	00 
  400b68:	48 83 c0 10          	add    $0x10,%rax
  400b6c:	66 0f 28 80 f0 ed 02 	movapd 0x2edf0(%rax),%xmm0
  400b73:	00 
  400b74:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400b78:	66 0f 28 98 f0 11 fd 	movapd -0x2ee10(%rax),%xmm3
  400b7f:	ff 
  400b80:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400b84:	66 0f 5e cb          	divpd  %xmm3,%xmm1
  400b88:	66 0f 5e c3          	divpd  %xmm3,%xmm0
  400b8c:	66 0f 58 48 f0       	addpd  -0x10(%rax),%xmm1
  400b91:	0f 29 48 f0          	movaps %xmm1,-0x10(%rax)
  400b95:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400b99:	66 0f 58 88 f0 0b fe 	addpd  -0x1f410(%rax),%xmm1
  400ba0:	ff 
  400ba1:	66 0f 58 80 f0 f9 00 	addpd  0xf9f0(%rax),%xmm0
  400ba8:	00 
  400ba9:	0f 29 88 f0 0b fe ff 	movaps %xmm1,-0x1f410(%rax)
  400bb0:	0f 29 80 f0 f9 00 00 	movaps %xmm0,0xf9f0(%rax)
  400bb7:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400bbb:	66 0f 58 80 f0 05 ff 	addpd  -0xfa10(%rax),%xmm0
  400bc2:	ff 
  400bc3:	0f 29 80 f0 05 ff ff 	movaps %xmm0,-0xfa10(%rax)
  400bca:	48 3d 90 08 64 00    	cmp    $0x640890,%rax
  400bd0:	75 8e                	jne    400b60 <_Z10run_simplev+0x90>
  400bd2:	f2 0f 10 0d b6 f0 25 	movsd  0x25f0b6(%rip),%xmm1        # 65fc90 <__Bodydata_buffer+0x5dc10>
  400bd9:	00 
  400bda:	83 ea 01             	sub    $0x1,%edx
  400bdd:	f2 0f 10 05 ab ea 26 	movsd  0x26eaab(%rip),%xmm0        # 66f690 <_ZStL8__ioinit>
  400be4:	00 
  400be5:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400be9:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400bed:	f2 0f 5e 0d 9b 0e 21 	divsd  0x210e9b(%rip),%xmm1        # 611a90 <__Bodydata_buffer+0xfa10>
  400bf4:	00 
  400bf5:	f2 0f 5e 05 93 0e 21 	divsd  0x210e93(%rip),%xmm0        # 611a90 <__Bodydata_buffer+0xfa10>
  400bfc:	00 
  400bfd:	f2 0f 58 0d 8b fc 23 	addsd  0x23fc8b(%rip),%xmm1        # 640890 <__Bodydata_buffer+0x3e810>
  400c04:	00 
  400c05:	f2 0f 11 0d 83 fc 23 	movsd  %xmm1,0x23fc83(%rip)        # 640890 <__Bodydata_buffer+0x3e810>
  400c0c:	00 
  400c0d:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400c11:	f2 0f 58 0d 77 08 22 	addsd  0x220877(%rip),%xmm1        # 621490 <__Bodydata_buffer+0x1f410>
  400c18:	00 
  400c19:	f2 0f 58 05 6f f6 24 	addsd  0x24f66f(%rip),%xmm0        # 650290 <__Bodydata_buffer+0x4e210>
  400c20:	00 
  400c21:	f2 0f 11 0d 67 08 22 	movsd  %xmm1,0x220867(%rip)        # 621490 <__Bodydata_buffer+0x1f410>
  400c28:	00 
  400c29:	f2 0f 11 05 5f f6 24 	movsd  %xmm0,0x24f65f(%rip)        # 650290 <__Bodydata_buffer+0x4e210>
  400c30:	00 
  400c31:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c35:	f2 0f 58 05 53 02 23 	addsd  0x230253(%rip),%xmm0        # 630e90 <__Bodydata_buffer+0x2ee10>
  400c3c:	00 
  400c3d:	f2 0f 11 05 4b 02 23 	movsd  %xmm0,0x23024b(%rip)        # 630e90 <__Bodydata_buffer+0x2ee10>
  400c44:	00 
  400c45:	0f 85 9d fe ff ff    	jne    400ae8 <_Z10run_simplev+0x18>
  400c4b:	f3 c3                	repz retq 
  400c4d:	0f 1f 00             	nopl   (%rax)

0000000000400c50 <_Z14run_simulationv>:
  400c50:	41 57                	push   %r15
  400c52:	41 56                	push   %r14
  400c54:	41 55                	push   %r13
  400c56:	41 54                	push   %r12
  400c58:	55                   	push   %rbp
  400c59:	53                   	push   %rbx
  400c5a:	48 83 ec 68          	sub    $0x68,%rsp
  400c5e:	4c 8b 35 1b 14 20 00 	mov    0x20141b(%rip),%r14        # 602080 <__Bodydata_buffer>
  400c65:	f2 0f 10 3d 83 07 00 	movsd  0x783(%rip),%xmm7        # 4013f0 <_IO_stdin_used+0x90>
  400c6c:	00 
  400c6d:	c7 44 24 5c 05 00 00 	movl   $0x5,0x5c(%rsp)
  400c74:	00 
  400c75:	f2 0f 10 35 4b 07 00 	movsd  0x74b(%rip),%xmm6        # 4013c8 <_IO_stdin_used+0x68>
  400c7c:	00 
  400c7d:	4d 85 f6             	test   %r14,%r14
  400c80:	66 44 0f 28 c7       	movapd %xmm7,%xmm8
  400c85:	0f 84 65 02 00 00    	je     400ef0 <_Z14run_simulationv+0x2a0>
  400c8b:	31 c0                	xor    %eax,%eax
  400c8d:	0f 1f 00             	nopl   (%rax)
  400c90:	48 83 c0 01          	add    $0x1,%rax
  400c94:	48 8d 14 c5 90 20 60 	lea    0x602090(,%rax,8),%rdx
  400c9b:	00 
  400c9c:	4c 39 f0             	cmp    %r14,%rax
  400c9f:	48 c7 82 00 e2 04 00 	movq   $0x0,0x4e200(%rdx)
  400ca6:	00 00 00 00 
  400caa:	48 c7 82 00 dc 05 00 	movq   $0x0,0x5dc00(%rdx)
  400cb1:	00 00 00 00 
  400cb5:	75 d9                	jne    400c90 <_Z14run_simulationv+0x40>
  400cb7:	f2 0f 10 2d 11 07 00 	movsd  0x711(%rip),%xmm5        # 4013d0 <_IO_stdin_used+0x70>
  400cbe:	00 
  400cbf:	4d 89 f7             	mov    %r14,%r15
  400cc2:	45 31 ed             	xor    %r13d,%r13d
  400cc5:	0f 1f 00             	nopl   (%rax)
  400cc8:	49 83 c5 01          	add    $0x1,%r13
  400ccc:	4d 85 ff             	test   %r15,%r15
  400ccf:	0f 84 e8 00 00 00    	je     400dbd <_Z14run_simulationv+0x16d>
  400cd5:	4a 8d 04 ed 00 00 00 	lea    0x0(,%r13,8),%rax
  400cdc:	00 
  400cdd:	31 ed                	xor    %ebp,%ebp
  400cdf:	48 8d 88 90 1a 61 00 	lea    0x611a90(%rax),%rcx
  400ce6:	48 8d 90 90 14 62 00 	lea    0x621490(%rax),%rdx
  400ced:	48 05 90 20 60 00    	add    $0x602090,%rax
  400cf3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400cf8:	48 83 c5 01          	add    $0x1,%rbp
  400cfc:	49 39 ed             	cmp    %rbp,%r13
  400cff:	0f 84 a8 00 00 00    	je     400dad <_Z14run_simulationv+0x15d>
  400d05:	4c 8d 24 ed 00 00 00 	lea    0x0(,%rbp,8),%r12
  400d0c:	00 
  400d0d:	f2 0f 10 0a          	movsd  (%rdx),%xmm1
  400d11:	f2 0f 10 19          	movsd  (%rcx),%xmm3
  400d15:	49 8d 9c 24 90 20 60 	lea    0x602090(%r12),%rbx
  400d1c:	00 
  400d1d:	f2 0f 5c 8b 00 f4 01 	subsd  0x1f400(%rbx),%xmm1
  400d24:	00 
  400d25:	f2 0f 5c 9b 00 fa 00 	subsd  0xfa00(%rbx),%xmm3
  400d2c:	00 
  400d2d:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400d32:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400d36:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400d3b:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400d3f:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400d44:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400d48:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400d4c:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400d50:	0f 8a b3 01 00 00    	jp     400f09 <_Z14run_simulationv+0x2b9>
  400d56:	f2 41 0f 10 94 24 90 	movsd  0x602090(%r12),%xmm2
  400d5d:	20 60 00 
  400d60:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400d64:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400d68:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d6c:	f2 0f 59 10          	mulsd  (%rax),%xmm2
  400d70:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400d74:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400d78:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400d7c:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400d80:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400d85:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400d89:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400d8d:	f2 0f 58 9b 00 e2 04 	addsd  0x4e200(%rbx),%xmm3
  400d94:	00 
  400d95:	f2 0f 11 9b 00 e2 04 	movsd  %xmm3,0x4e200(%rbx)
  400d9c:	00 
  400d9d:	f2 0f 58 8b 00 dc 05 	addsd  0x5dc00(%rbx),%xmm1
  400da4:	00 
  400da5:	f2 0f 11 8b 00 dc 05 	movsd  %xmm1,0x5dc00(%rbx)
  400dac:	00 
  400dad:	4c 39 fd             	cmp    %r15,%rbp
  400db0:	0f 85 42 ff ff ff    	jne    400cf8 <_Z14run_simulationv+0xa8>
  400db6:	4c 8b 3d c3 12 20 00 	mov    0x2012c3(%rip),%r15        # 602080 <__Bodydata_buffer>
  400dbd:	4d 39 f5             	cmp    %r14,%r13
  400dc0:	0f 85 02 ff ff ff    	jne    400cc8 <_Z14run_simulationv+0x78>
  400dc6:	4d 85 ff             	test   %r15,%r15
  400dc9:	0f 84 0e 01 00 00    	je     400edd <_Z14run_simulationv+0x28d>
  400dcf:	f2 0f 10 1d e9 05 00 	movsd  0x5e9(%rip),%xmm3        # 4013c0 <_IO_stdin_used+0x60>
  400dd6:	00 
  400dd7:	31 c9                	xor    %ecx,%ecx
  400dd9:	f2 0f 10 0d cf 05 00 	movsd  0x5cf(%rip),%xmm1        # 4013b0 <_IO_stdin_used+0x50>
  400de0:	00 
  400de1:	f2 0f 10 15 ef 05 00 	movsd  0x5ef(%rip),%xmm2        # 4013d8 <_IO_stdin_used+0x78>
  400de8:	00 
  400de9:	66 0f 28 e3          	movapd %xmm3,%xmm4
  400ded:	0f 1f 00             	nopl   (%rax)
  400df0:	48 83 c1 01          	add    $0x1,%rcx
  400df4:	48 8d 14 cd 00 00 00 	lea    0x0(,%rcx,8),%rdx
  400dfb:	00 
  400dfc:	48 8d 82 90 20 60 00 	lea    0x602090(%rdx),%rax
  400e03:	f2 0f 10 80 00 e2 04 	movsd  0x4e200(%rax),%xmm0
  400e0a:	00 
  400e0b:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400e0f:	f2 0f 5e 82 90 20 60 	divsd  0x602090(%rdx),%xmm0
  400e16:	00 
  400e17:	f2 0f 58 80 00 ee 02 	addsd  0x2ee00(%rax),%xmm0
  400e1e:	00 
  400e1f:	f2 0f 11 80 00 ee 02 	movsd  %xmm0,0x2ee00(%rax)
  400e26:	00 
  400e27:	f2 0f 10 80 00 dc 05 	movsd  0x5dc00(%rax),%xmm0
  400e2e:	00 
  400e2f:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400e33:	f2 0f 5e 82 90 20 60 	divsd  0x602090(%rdx),%xmm0
  400e3a:	00 
  400e3b:	f2 0f 58 80 00 e8 03 	addsd  0x3e800(%rax),%xmm0
  400e42:	00 
  400e43:	f2 0f 11 80 00 e8 03 	movsd  %xmm0,0x3e800(%rax)
  400e4a:	00 
  400e4b:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  400e52:	00 
  400e53:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400e57:	f2 0f 58 80 00 fa 00 	addsd  0xfa00(%rax),%xmm0
  400e5e:	00 
  400e5f:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400e66:	00 
  400e67:	f2 0f 10 80 00 e8 03 	movsd  0x3e800(%rax),%xmm0
  400e6e:	00 
  400e6f:	f2 0f 10 a8 00 fa 00 	movsd  0xfa00(%rax),%xmm5
  400e76:	00 
  400e77:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  400e7b:	66 0f 2e d5          	ucomisd %xmm5,%xmm2
  400e7f:	f2 0f 58 80 00 f4 01 	addsd  0x1f400(%rax),%xmm0
  400e86:	00 
  400e87:	f2 0f 11 80 00 f4 01 	movsd  %xmm0,0x1f400(%rax)
  400e8e:	00 
  400e8f:	77 06                	ja     400e97 <_Z14run_simulationv+0x247>
  400e91:	66 0f 2e eb          	ucomisd %xmm3,%xmm5
  400e95:	76 1c                	jbe    400eb3 <_Z14run_simulationv+0x263>
  400e97:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  400e9e:	00 
  400e9f:	66 0f 57 c7          	xorpd  %xmm7,%xmm0
  400ea3:	f2 0f 11 80 00 ee 02 	movsd  %xmm0,0x2ee00(%rax)
  400eaa:	00 
  400eab:	f2 0f 10 80 00 f4 01 	movsd  0x1f400(%rax),%xmm0
  400eb2:	00 
  400eb3:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  400eb7:	77 06                	ja     400ebf <_Z14run_simulationv+0x26f>
  400eb9:	66 0f 2e c4          	ucomisd %xmm4,%xmm0
  400ebd:	76 15                	jbe    400ed4 <_Z14run_simulationv+0x284>
  400ebf:	f2 0f 10 80 00 e8 03 	movsd  0x3e800(%rax),%xmm0
  400ec6:	00 
  400ec7:	66 41 0f 57 c0       	xorpd  %xmm8,%xmm0
  400ecc:	f2 0f 11 80 00 e8 03 	movsd  %xmm0,0x3e800(%rax)
  400ed3:	00 
  400ed4:	4c 39 f9             	cmp    %r15,%rcx
  400ed7:	0f 85 13 ff ff ff    	jne    400df0 <_Z14run_simulationv+0x1a0>
  400edd:	83 6c 24 5c 01       	subl   $0x1,0x5c(%rsp)
  400ee2:	74 16                	je     400efa <_Z14run_simulationv+0x2aa>
  400ee4:	4d 89 fe             	mov    %r15,%r14
  400ee7:	4d 85 f6             	test   %r14,%r14
  400eea:	0f 85 9b fd ff ff    	jne    400c8b <_Z14run_simulationv+0x3b>
  400ef0:	45 31 ff             	xor    %r15d,%r15d
  400ef3:	83 6c 24 5c 01       	subl   $0x1,0x5c(%rsp)
  400ef8:	75 ea                	jne    400ee4 <_Z14run_simulationv+0x294>
  400efa:	48 83 c4 68          	add    $0x68,%rsp
  400efe:	5b                   	pop    %rbx
  400eff:	5d                   	pop    %rbp
  400f00:	41 5c                	pop    %r12
  400f02:	41 5d                	pop    %r13
  400f04:	41 5e                	pop    %r14
  400f06:	41 5f                	pop    %r15
  400f08:	c3                   	retq   
  400f09:	44 0f 29 44 24 40    	movaps %xmm8,0x40(%rsp)
  400f0f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400f14:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
  400f19:	0f 29 7c 24 30       	movaps %xmm7,0x30(%rsp)
  400f1e:	f2 0f 11 74 24 50    	movsd  %xmm6,0x50(%rsp)
  400f24:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400f2a:	f2 44 0f 11 4c 24 18 	movsd  %xmm9,0x18(%rsp)
  400f31:	48 89 0c 24          	mov    %rcx,(%rsp)
  400f35:	f2 0f 11 5c 24 08    	movsd  %xmm3,0x8(%rsp)
  400f3b:	e8 10 f8 ff ff       	callq  400750 <sqrt@plt>
  400f40:	f2 0f 10 74 24 50    	movsd  0x50(%rsp),%xmm6
  400f46:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400f4a:	66 44 0f 28 44 24 40 	movapd 0x40(%rsp),%xmm8
  400f51:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400f56:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400f5b:	66 0f 28 7c 24 30    	movapd 0x30(%rsp),%xmm7
  400f61:	48 8b 0c 24          	mov    (%rsp),%rcx
  400f65:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400f6b:	f2 44 0f 10 4c 24 18 	movsd  0x18(%rsp),%xmm9
  400f72:	f2 0f 10 5c 24 08    	movsd  0x8(%rsp),%xmm3
  400f78:	e9 d9 fd ff ff       	jmpq   400d56 <_Z14run_simulationv+0x106>
  400f7d:	0f 1f 00             	nopl   (%rax)

0000000000400f80 <_Z12float_as_intf>:
  400f80:	66 0f 7e c0          	movd   %xmm0,%eax
  400f84:	c3                   	retq   
  400f85:	90                   	nop
  400f86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f8d:	00 00 00 

0000000000400f90 <_ZN4Body7storageEv>:
  400f90:	b8 80 20 60 00       	mov    $0x602080,%eax
  400f95:	c3                   	retq   
  400f96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f9d:	00 00 00 

0000000000400fa0 <_Z19codegen_reset_forceP4Body>:
  400fa0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400fa4:	48 8d 04 fd 90 20 60 	lea    0x602090(,%rdi,8),%rax
  400fab:	00 
  400fac:	f2 0f 11 80 00 e2 04 	movsd  %xmm0,0x4e200(%rax)
  400fb3:	00 
  400fb4:	f2 0f 11 80 00 dc 05 	movsd  %xmm0,0x5dc00(%rax)
  400fbb:	00 
  400fbc:	c3                   	retq   
  400fbd:	0f 1f 00             	nopl   (%rax)

0000000000400fc0 <_Z17codegen_add_forceP4BodyS0_>:
  400fc0:	48 39 f7             	cmp    %rsi,%rdi
  400fc3:	0f 84 cb 00 00 00    	je     401094 <_Z17codegen_add_forceP4BodyS0_+0xd4>
  400fc9:	41 54                	push   %r12
  400fcb:	55                   	push   %rbp
  400fcc:	48 8d 04 f5 90 20 60 	lea    0x602090(,%rsi,8),%rax
  400fd3:	00 
  400fd4:	53                   	push   %rbx
  400fd5:	48 8d 1c fd 90 20 60 	lea    0x602090(,%rdi,8),%rbx
  400fdc:	00 
  400fdd:	48 89 f5             	mov    %rsi,%rbp
  400fe0:	49 89 fc             	mov    %rdi,%r12
  400fe3:	48 83 ec 10          	sub    $0x10,%rsp
  400fe7:	f2 0f 10 88 00 f4 01 	movsd  0x1f400(%rax),%xmm1
  400fee:	00 
  400fef:	f2 0f 10 98 00 fa 00 	movsd  0xfa00(%rax),%xmm3
  400ff6:	00 
  400ff7:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400ffb:	f2 0f 5c 9b 00 fa 00 	subsd  0xfa00(%rbx),%xmm3
  401002:	00 
  401003:	f2 0f 5c ab 00 f4 01 	subsd  0x1f400(%rbx),%xmm5
  40100a:	00 
  40100b:	66 0f 28 cb          	movapd %xmm3,%xmm1
  40100f:	66 0f 28 c5          	movapd %xmm5,%xmm0
  401013:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  401017:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  40101b:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  40101f:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  401023:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  401027:	7a 6d                	jp     401096 <_Z17codegen_add_forceP4BodyS0_+0xd6>
  401029:	f2 0f 10 15 97 03 00 	movsd  0x397(%rip),%xmm2        # 4013c8 <_IO_stdin_used+0x68>
  401030:	00 
  401031:	66 0f 28 c4          	movapd %xmm4,%xmm0
  401035:	f2 42 0f 59 14 e5 90 	mulsd  0x602090(,%r12,8),%xmm2
  40103c:	20 60 00 
  40103f:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  401043:	f2 0f 58 05 85 03 00 	addsd  0x385(%rip),%xmm0        # 4013d0 <_IO_stdin_used+0x70>
  40104a:	00 
  40104b:	f2 0f 59 14 ed 90 20 	mulsd  0x602090(,%rbp,8),%xmm2
  401052:	60 00 
  401054:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  401058:	66 0f 28 ca          	movapd %xmm2,%xmm1
  40105c:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  401060:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401064:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  401068:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  40106c:	f2 0f 58 9b 00 e2 04 	addsd  0x4e200(%rbx),%xmm3
  401073:	00 
  401074:	f2 0f 11 9b 00 e2 04 	movsd  %xmm3,0x4e200(%rbx)
  40107b:	00 
  40107c:	f2 0f 58 8b 00 dc 05 	addsd  0x5dc00(%rbx),%xmm1
  401083:	00 
  401084:	f2 0f 11 8b 00 dc 05 	movsd  %xmm1,0x5dc00(%rbx)
  40108b:	00 
  40108c:	48 83 c4 10          	add    $0x10,%rsp
  401090:	5b                   	pop    %rbx
  401091:	5d                   	pop    %rbp
  401092:	41 5c                	pop    %r12
  401094:	f3 c3                	repz retq 
  401096:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  40109c:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  4010a1:	e8 aa f6 ff ff       	callq  400750 <sqrt@plt>
  4010a6:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  4010ac:	66 0f 28 e0          	movapd %xmm0,%xmm4
  4010b0:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  4010b5:	e9 6f ff ff ff       	jmpq   401029 <_Z17codegen_add_forceP4BodyS0_+0x69>
  4010ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004010c0 <_Z14codegen_updateP4Bodyd>:
  4010c0:	48 c1 e7 03          	shl    $0x3,%rdi
  4010c4:	f2 0f 10 15 0c 03 00 	movsd  0x30c(%rip),%xmm2        # 4013d8 <_IO_stdin_used+0x78>
  4010cb:	00 
  4010cc:	48 8d 87 90 20 60 00 	lea    0x602090(%rdi),%rax
  4010d3:	f2 0f 10 88 00 e2 04 	movsd  0x4e200(%rax),%xmm1
  4010da:	00 
  4010db:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4010df:	f2 0f 5e 8f 90 20 60 	divsd  0x602090(%rdi),%xmm1
  4010e6:	00 
  4010e7:	f2 0f 58 88 00 ee 02 	addsd  0x2ee00(%rax),%xmm1
  4010ee:	00 
  4010ef:	f2 0f 11 88 00 ee 02 	movsd  %xmm1,0x2ee00(%rax)
  4010f6:	00 
  4010f7:	f2 0f 10 88 00 dc 05 	movsd  0x5dc00(%rax),%xmm1
  4010fe:	00 
  4010ff:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401103:	f2 0f 5e 8f 90 20 60 	divsd  0x602090(%rdi),%xmm1
  40110a:	00 
  40110b:	f2 0f 58 88 00 e8 03 	addsd  0x3e800(%rax),%xmm1
  401112:	00 
  401113:	f2 0f 11 88 00 e8 03 	movsd  %xmm1,0x3e800(%rax)
  40111a:	00 
  40111b:	f2 0f 10 88 00 ee 02 	movsd  0x2ee00(%rax),%xmm1
  401122:	00 
  401123:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401127:	f2 0f 59 80 00 e8 03 	mulsd  0x3e800(%rax),%xmm0
  40112e:	00 
  40112f:	f2 0f 58 88 00 fa 00 	addsd  0xfa00(%rax),%xmm1
  401136:	00 
  401137:	f2 0f 58 80 00 f4 01 	addsd  0x1f400(%rax),%xmm0
  40113e:	00 
  40113f:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  401143:	f2 0f 11 88 00 fa 00 	movsd  %xmm1,0xfa00(%rax)
  40114a:	00 
  40114b:	f2 0f 11 80 00 f4 01 	movsd  %xmm0,0x1f400(%rax)
  401152:	00 
  401153:	77 0a                	ja     40115f <_Z14codegen_updateP4Bodyd+0x9f>
  401155:	66 0f 2e 0d 63 02 00 	ucomisd 0x263(%rip),%xmm1        # 4013c0 <_IO_stdin_used+0x60>
  40115c:	00 
  40115d:	76 24                	jbe    401183 <_Z14codegen_updateP4Bodyd+0xc3>
  40115f:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  401166:	00 
  401167:	f2 0f 10 0d 81 02 00 	movsd  0x281(%rip),%xmm1        # 4013f0 <_IO_stdin_used+0x90>
  40116e:	00 
  40116f:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401173:	f2 0f 11 80 00 ee 02 	movsd  %xmm0,0x2ee00(%rax)
  40117a:	00 
  40117b:	f2 0f 10 80 00 f4 01 	movsd  0x1f400(%rax),%xmm0
  401182:	00 
  401183:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  401187:	77 0a                	ja     401193 <_Z14codegen_updateP4Bodyd+0xd3>
  401189:	66 0f 2e 05 2f 02 00 	ucomisd 0x22f(%rip),%xmm0        # 4013c0 <_IO_stdin_used+0x60>
  401190:	00 
  401191:	76 1d                	jbe    4011b0 <_Z14codegen_updateP4Bodyd+0xf0>
  401193:	f2 0f 10 80 00 e8 03 	movsd  0x3e800(%rax),%xmm0
  40119a:	00 
  40119b:	f2 0f 10 0d 4d 02 00 	movsd  0x24d(%rip),%xmm1        # 4013f0 <_IO_stdin_used+0x90>
  4011a2:	00 
  4011a3:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  4011a7:	f2 0f 11 80 00 e8 03 	movsd  %xmm0,0x3e800(%rax)
  4011ae:	00 
  4011af:	c3                   	retq   
  4011b0:	f3 c3                	repz retq 
  4011b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4011b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011bd:	00 00 00 

00000000004011c0 <_Z26codegen_reset_force_manualP4Body>:
  4011c0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4011c4:	48 8d 04 fd 90 20 60 	lea    0x602090(,%rdi,8),%rax
  4011cb:	00 
  4011cc:	f2 0f 11 80 00 e2 04 	movsd  %xmm0,0x4e200(%rax)
  4011d3:	00 
  4011d4:	f2 0f 11 80 00 dc 05 	movsd  %xmm0,0x5dc00(%rax)
  4011db:	00 
  4011dc:	c3                   	retq   
  4011dd:	0f 1f 00             	nopl   (%rax)

00000000004011e0 <_Z24codegen_reset_force_halfP4Body>:
  4011e0:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  4011e7:	00 00 00 00 00 
  4011ec:	c3                   	retq   
  4011ed:	0f 1f 00             	nopl   (%rax)

00000000004011f0 <_Z18codegen_reset_massP4Body>:
  4011f0:	48 c7 04 fd 90 20 60 	movq   $0x0,0x602090(,%rdi,8)
  4011f7:	00 00 00 00 00 
  4011fc:	c3                   	retq   
  4011fd:	0f 1f 00             	nopl   (%rax)

0000000000401200 <_Z22codengen_simple_updateP4Bodyd>:
  401200:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  401207:	f2 0f 10 0c fd 80 20 	movsd  0x602080(,%rdi,8),%xmm1
  40120e:	60 00 
  401210:	48 8d 04 fd 00 06 ff 	lea    -0xfa00(,%rdi,8),%rax
  401217:	ff 
  401218:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40121c:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  401223:	5b 00 
  401225:	f2 0f 58 0c fd 80 2c 	addsd  0x5e2c80(,%rdi,8),%xmm1
  40122c:	5e 00 
  40122e:	f2 0f 11 0c fd 80 2c 	movsd  %xmm1,0x5e2c80(,%rdi,8)
  401235:	5e 00 
  401237:	f2 0f 10 0c fd 80 1a 	movsd  0x611a80(,%rdi,8),%xmm1
  40123e:	61 00 
  401240:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401244:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  40124b:	5b 00 
  40124d:	f2 0f 58 88 80 20 60 	addsd  0x602080(%rax),%xmm1
  401254:	00 
  401255:	f2 0f 11 88 80 20 60 	movsd  %xmm1,0x602080(%rax)
  40125c:	00 
  40125d:	f2 0f 10 0c fd 80 2c 	movsd  0x5e2c80(,%rdi,8),%xmm1
  401264:	5e 00 
  401266:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40126a:	f2 0f 58 0c fd 80 38 	addsd  0x5c3880(,%rdi,8),%xmm1
  401271:	5c 00 
  401273:	f2 0f 11 0c fd 80 38 	movsd  %xmm1,0x5c3880(,%rdi,8)
  40127a:	5c 00 
  40127c:	f2 0f 59 80 80 20 60 	mulsd  0x602080(%rax),%xmm0
  401283:	00 
  401284:	f2 0f 58 04 fd 80 32 	addsd  0x5d3280(,%rdi,8),%xmm0
  40128b:	5d 00 
  40128d:	f2 0f 11 04 fd 80 32 	movsd  %xmm0,0x5d3280(,%rdi,8)
  401294:	5d 00 
  401296:	c3                   	retq   
  401297:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40129e:	00 00 

00000000004012a0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  4012a0:	55                   	push   %rbp
  4012a1:	53                   	push   %rbx
  4012a2:	48 89 fd             	mov    %rdi,%rbp
  4012a5:	48 83 ec 08          	sub    $0x8,%rsp
  4012a9:	e8 82 f4 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012ae:	48 89 c3             	mov    %rax,%rbx
  4012b1:	ff d5                	callq  *%rbp
  4012b3:	e8 78 f4 ff ff       	callq  400730 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012b8:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  4012bf:	de 1b 43 
  4012c2:	48 29 d8             	sub    %rbx,%rax
  4012c5:	48 83 c4 08          	add    $0x8,%rsp
  4012c9:	48 89 c1             	mov    %rax,%rcx
  4012cc:	48 f7 ea             	imul   %rdx
  4012cf:	48 c1 f9 3f          	sar    $0x3f,%rcx
  4012d3:	5b                   	pop    %rbx
  4012d4:	5d                   	pop    %rbp
  4012d5:	48 c1 fa 12          	sar    $0x12,%rdx
  4012d9:	48 89 d0             	mov    %rdx,%rax
  4012dc:	48 29 c8             	sub    %rcx,%rax
  4012df:	c3                   	retq   

00000000004012e0 <__libc_csu_init>:
  4012e0:	41 57                	push   %r15
  4012e2:	41 56                	push   %r14
  4012e4:	41 89 ff             	mov    %edi,%r15d
  4012e7:	41 55                	push   %r13
  4012e9:	41 54                	push   %r12
  4012eb:	4c 8d 25 f6 0a 20 00 	lea    0x200af6(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  4012f2:	55                   	push   %rbp
  4012f3:	48 8d 2d fe 0a 20 00 	lea    0x200afe(%rip),%rbp        # 601df8 <__init_array_end>
  4012fa:	53                   	push   %rbx
  4012fb:	49 89 f6             	mov    %rsi,%r14
  4012fe:	49 89 d5             	mov    %rdx,%r13
  401301:	4c 29 e5             	sub    %r12,%rbp
  401304:	48 83 ec 08          	sub    $0x8,%rsp
  401308:	48 c1 fd 03          	sar    $0x3,%rbp
  40130c:	e8 97 f3 ff ff       	callq  4006a8 <_init>
  401311:	48 85 ed             	test   %rbp,%rbp
  401314:	74 20                	je     401336 <__libc_csu_init+0x56>
  401316:	31 db                	xor    %ebx,%ebx
  401318:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40131f:	00 
  401320:	4c 89 ea             	mov    %r13,%rdx
  401323:	4c 89 f6             	mov    %r14,%rsi
  401326:	44 89 ff             	mov    %r15d,%edi
  401329:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40132d:	48 83 c3 01          	add    $0x1,%rbx
  401331:	48 39 eb             	cmp    %rbp,%rbx
  401334:	75 ea                	jne    401320 <__libc_csu_init+0x40>
  401336:	48 83 c4 08          	add    $0x8,%rsp
  40133a:	5b                   	pop    %rbx
  40133b:	5d                   	pop    %rbp
  40133c:	41 5c                	pop    %r12
  40133e:	41 5d                	pop    %r13
  401340:	41 5e                	pop    %r14
  401342:	41 5f                	pop    %r15
  401344:	c3                   	retq   
  401345:	90                   	nop
  401346:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40134d:	00 00 00 

0000000000401350 <__libc_csu_fini>:
  401350:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401354 <_fini>:
  401354:	48 83 ec 08          	sub    $0x8,%rsp
  401358:	48 83 c4 08          	add    $0x8,%rsp
  40135c:	c3                   	retq   
