
bin/g++_nbody_ikracpp_field:     file format elf64-x86-64


Disassembly of section .init:

00000000004006e0 <_init>:
  4006e0:	48 83 ec 08          	sub    $0x8,%rsp
  4006e4:	48 8b 05 0d 19 20 00 	mov    0x20190d(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4006eb:	48 85 c0             	test   %rax,%rax
  4006ee:	74 05                	je     4006f5 <_init+0x15>
  4006f0:	e8 ab 00 00 00       	callq  4007a0 <sqrt@plt+0x10>
  4006f5:	48 83 c4 08          	add    $0x8,%rsp
  4006f9:	c3                   	retq   

Disassembly of section .plt:

0000000000400700 <memset@plt-0x10>:
  400700:	ff 35 02 19 20 00    	pushq  0x201902(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400706:	ff 25 04 19 20 00    	jmpq   *0x201904(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <memset@plt>:
  400710:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400716:	68 00 00 00 00       	pushq  $0x0
  40071b:	e9 e0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400720 <__printf_chk@plt>:
  400720:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400726:	68 01 00 00 00       	pushq  $0x1
  40072b:	e9 d0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400730 <_ZNSt8ios_base4InitC1Ev@plt>:
  400730:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400736:	68 02 00 00 00       	pushq  $0x2
  40073b:	e9 c0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400740 <__libc_start_main@plt>:
  400740:	ff 25 ea 18 20 00    	jmpq   *0x2018ea(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400746:	68 03 00 00 00       	pushq  $0x3
  40074b:	e9 b0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400750 <__cxa_atexit@plt>:
  400750:	ff 25 e2 18 20 00    	jmpq   *0x2018e2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400756:	68 04 00 00 00       	pushq  $0x4
  40075b:	e9 a0 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400760 <_ZNSt8ios_base4InitD1Ev@plt>:
  400760:	ff 25 da 18 20 00    	jmpq   *0x2018da(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400766:	68 05 00 00 00       	pushq  $0x5
  40076b:	e9 90 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400770:	ff 25 d2 18 20 00    	jmpq   *0x2018d2(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400776:	68 06 00 00 00       	pushq  $0x6
  40077b:	e9 80 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400780 <rand@plt>:
  400780:	ff 25 ca 18 20 00    	jmpq   *0x2018ca(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400786:	68 07 00 00 00       	pushq  $0x7
  40078b:	e9 70 ff ff ff       	jmpq   400700 <_init+0x20>

0000000000400790 <sqrt@plt>:
  400790:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400796:	68 08 00 00 00       	pushq  $0x8
  40079b:	e9 60 ff ff ff       	jmpq   400700 <_init+0x20>

Disassembly of section .plt.got:

00000000004007a0 <.plt.got>:
  4007a0:	ff 25 52 18 20 00    	jmpq   *0x201852(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  4007a6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004007b0 <main>:
  4007b0:	55                   	push   %rbp
  4007b1:	53                   	push   %rbx
  4007b2:	bf b0 09 40 00       	mov    $0x4009b0,%edi
  4007b7:	48 83 ec 08          	sub    $0x8,%rsp
  4007bb:	48 c7 05 ba 18 20 00 	movq   $0x0,0x2018ba(%rip)        # 602080 <__Bodydata_buffer>
  4007c2:	00 00 00 00 
  4007c6:	48 c7 05 b7 18 20 00 	movq   $0x0,0x2018b7(%rip)        # 602088 <__Bodydata_buffer+0x8>
  4007cd:	00 00 00 00 
  4007d1:	e8 aa 0a 00 00       	callq  401280 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007d6:	bf 90 0c 40 00       	mov    $0x400c90,%edi
  4007db:	48 89 c3             	mov    %rax,%rbx
  4007de:	e8 9d 0a 00 00       	callq  401280 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007e3:	bf 10 0b 40 00       	mov    $0x400b10,%edi
  4007e8:	48 89 c5             	mov    %rax,%rbp
  4007eb:	e8 90 0a 00 00       	callq  401280 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007f0:	b9 98 1a 61 00       	mov    $0x611a98,%ecx
  4007f5:	49 89 c0             	mov    %rax,%r8
  4007f8:	bf 98 14 62 00       	mov    $0x621498,%edi
  4007fd:	ba 0b 00 00 00       	mov    $0xb,%edx
  400802:	be 8d 68 b7 6c       	mov    $0x6cb7688d,%esi
  400807:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40080e:	00 00 
  400810:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400814:	48 83 c1 08          	add    $0x8,%rcx
  400818:	66 0f ef c9          	pxor   %xmm1,%xmm1
  40081c:	f2 0f 5a 41 f8       	cvtsd2ss -0x8(%rcx),%xmm0
  400821:	66 41 0f 7e c1       	movd   %xmm0,%r9d
  400826:	f2 0f 5a 89 f8 f9 00 	cvtsd2ss 0xf9f8(%rcx),%xmm1
  40082d:	00 
  40082e:	44 01 ca             	add    %r9d,%edx
  400831:	66 41 0f 7e c9       	movd   %xmm1,%r9d
  400836:	41 01 d1             	add    %edx,%r9d
  400839:	44 89 c8             	mov    %r9d,%eax
  40083c:	f7 ee                	imul   %esi
  40083e:	44 89 c8             	mov    %r9d,%eax
  400841:	c1 f8 1f             	sar    $0x1f,%eax
  400844:	c1 fa 13             	sar    $0x13,%edx
  400847:	29 c2                	sub    %eax,%edx
  400849:	69 d2 87 d6 12 00    	imul   $0x12d687,%edx,%edx
  40084f:	41 29 d1             	sub    %edx,%r9d
  400852:	48 39 cf             	cmp    %rcx,%rdi
  400855:	44 89 ca             	mov    %r9d,%edx
  400858:	75 b6                	jne    400810 <main+0x60>
  40085a:	48 89 e9             	mov    %rbp,%rcx
  40085d:	48 89 da             	mov    %rbx,%rdx
  400860:	be 48 13 40 00       	mov    $0x401348,%esi
  400865:	bf 01 00 00 00       	mov    $0x1,%edi
  40086a:	31 c0                	xor    %eax,%eax
  40086c:	e8 af fe ff ff       	callq  400720 <__printf_chk@plt>
  400871:	48 83 c4 08          	add    $0x8,%rsp
  400875:	31 c0                	xor    %eax,%eax
  400877:	5b                   	pop    %rbx
  400878:	5d                   	pop    %rbp
  400879:	c3                   	retq   
  40087a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400880 <_GLOBAL__sub_I__Z12float_as_intf>:
  400880:	48 83 ec 08          	sub    $0x8,%rsp
  400884:	bf 90 f6 66 00       	mov    $0x66f690,%edi
  400889:	e8 a2 fe ff ff       	callq  400730 <_ZNSt8ios_base4InitC1Ev@plt>
  40088e:	ba 68 20 60 00       	mov    $0x602068,%edx
  400893:	be 90 f6 66 00       	mov    $0x66f690,%esi
  400898:	bf 60 07 40 00       	mov    $0x400760,%edi
  40089d:	48 83 c4 08          	add    $0x8,%rsp
  4008a1:	e9 aa fe ff ff       	jmpq   400750 <__cxa_atexit@plt>
  4008a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ad:	00 00 00 

00000000004008b0 <_start>:
  4008b0:	31 ed                	xor    %ebp,%ebp
  4008b2:	49 89 d1             	mov    %rdx,%r9
  4008b5:	5e                   	pop    %rsi
  4008b6:	48 89 e2             	mov    %rsp,%rdx
  4008b9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008bd:	50                   	push   %rax
  4008be:	54                   	push   %rsp
  4008bf:	49 c7 c0 30 13 40 00 	mov    $0x401330,%r8
  4008c6:	48 c7 c1 c0 12 40 00 	mov    $0x4012c0,%rcx
  4008cd:	48 c7 c7 b0 07 40 00 	mov    $0x4007b0,%rdi
  4008d4:	e8 67 fe ff ff       	callq  400740 <__libc_start_main@plt>
  4008d9:	f4                   	hlt    
  4008da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008e0 <deregister_tm_clones>:
  4008e0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4008e5:	55                   	push   %rbp
  4008e6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4008ec:	48 83 f8 0e          	cmp    $0xe,%rax
  4008f0:	48 89 e5             	mov    %rsp,%rbp
  4008f3:	76 1b                	jbe    400910 <deregister_tm_clones+0x30>
  4008f5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008fa:	48 85 c0             	test   %rax,%rax
  4008fd:	74 11                	je     400910 <deregister_tm_clones+0x30>
  4008ff:	5d                   	pop    %rbp
  400900:	bf 70 20 60 00       	mov    $0x602070,%edi
  400905:	ff e0                	jmpq   *%rax
  400907:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40090e:	00 00 
  400910:	5d                   	pop    %rbp
  400911:	c3                   	retq   
  400912:	0f 1f 40 00          	nopl   0x0(%rax)
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <register_tm_clones>:
  400920:	be 70 20 60 00       	mov    $0x602070,%esi
  400925:	55                   	push   %rbp
  400926:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40092d:	48 c1 fe 03          	sar    $0x3,%rsi
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 89 f0             	mov    %rsi,%rax
  400937:	48 c1 e8 3f          	shr    $0x3f,%rax
  40093b:	48 01 c6             	add    %rax,%rsi
  40093e:	48 d1 fe             	sar    %rsi
  400941:	74 15                	je     400958 <register_tm_clones+0x38>
  400943:	b8 00 00 00 00       	mov    $0x0,%eax
  400948:	48 85 c0             	test   %rax,%rax
  40094b:	74 0b                	je     400958 <register_tm_clones+0x38>
  40094d:	5d                   	pop    %rbp
  40094e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400953:	ff e0                	jmpq   *%rax
  400955:	0f 1f 00             	nopl   (%rax)
  400958:	5d                   	pop    %rbp
  400959:	c3                   	retq   
  40095a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400960 <__do_global_dtors_aux>:
  400960:	80 3d 09 17 20 00 00 	cmpb   $0x0,0x201709(%rip)        # 602070 <__TMC_END__>
  400967:	75 11                	jne    40097a <__do_global_dtors_aux+0x1a>
  400969:	55                   	push   %rbp
  40096a:	48 89 e5             	mov    %rsp,%rbp
  40096d:	e8 6e ff ff ff       	callq  4008e0 <deregister_tm_clones>
  400972:	5d                   	pop    %rbp
  400973:	c6 05 f6 16 20 00 01 	movb   $0x1,0x2016f6(%rip)        # 602070 <__TMC_END__>
  40097a:	f3 c3                	repz retq 
  40097c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400980 <frame_dummy>:
  400980:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400985:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400989:	75 05                	jne    400990 <frame_dummy+0x10>
  40098b:	eb 93                	jmp    400920 <register_tm_clones>
  40098d:	0f 1f 00             	nopl   (%rax)
  400990:	b8 00 00 00 00       	mov    $0x0,%eax
  400995:	48 85 c0             	test   %rax,%rax
  400998:	74 f1                	je     40098b <frame_dummy+0xb>
  40099a:	55                   	push   %rbp
  40099b:	48 89 e5             	mov    %rsp,%rbp
  40099e:	ff d0                	callq  *%rax
  4009a0:	5d                   	pop    %rbp
  4009a1:	e9 7a ff ff ff       	jmpq   400920 <register_tm_clones>
  4009a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ad:	00 00 00 

00000000004009b0 <_Z13instantiationv>:
  4009b0:	41 56                	push   %r14
  4009b2:	41 55                	push   %r13
  4009b4:	41 bd 40 1f 00 00    	mov    $0x1f40,%r13d
  4009ba:	41 54                	push   %r12
  4009bc:	55                   	push   %rbp
  4009bd:	53                   	push   %rbx
  4009be:	48 83 ec 20          	sub    $0x20,%rsp
  4009c2:	f2 0f 10 0d be 09 00 	movsd  0x9be(%rip),%xmm1        # 401388 <_IO_stdin_used+0x48>
  4009c9:	00 
  4009ca:	f2 0f 10 15 be 09 00 	movsd  0x9be(%rip),%xmm2        # 401390 <_IO_stdin_used+0x50>
  4009d1:	00 
  4009d2:	f2 0f 10 1d be 09 00 	movsd  0x9be(%rip),%xmm3        # 401398 <_IO_stdin_used+0x58>
  4009d9:	00 
  4009da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4009e0:	f2 0f 11 5c 24 18    	movsd  %xmm3,0x18(%rsp)
  4009e6:	f2 0f 11 54 24 10    	movsd  %xmm2,0x10(%rsp)
  4009ec:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
  4009f2:	e8 89 fd ff ff       	callq  400780 <rand@plt>
  4009f7:	89 c3                	mov    %eax,%ebx
  4009f9:	e8 82 fd ff ff       	callq  400780 <rand@plt>
  4009fe:	41 89 c4             	mov    %eax,%r12d
  400a01:	e8 7a fd ff ff       	callq  400780 <rand@plt>
  400a06:	89 c5                	mov    %eax,%ebp
  400a08:	e8 73 fd ff ff       	callq  400780 <rand@plt>
  400a0d:	41 89 c6             	mov    %eax,%r14d
  400a10:	e8 6b fd ff ff       	callq  400780 <rand@plt>
  400a15:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a19:	48 8b 15 60 16 20 00 	mov    0x201660(%rip),%rdx        # 602080 <__Bodydata_buffer>
  400a20:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  400a26:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  400a2c:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  400a30:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400a34:	48 83 c2 03          	add    $0x3,%rdx
  400a38:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  400a3e:	41 83 ed 01          	sub    $0x1,%r13d
  400a42:	48 89 0d 37 16 20 00 	mov    %rcx,0x201637(%rip)        # 602080 <__Bodydata_buffer>
  400a49:	48 c7 04 d5 80 02 65 	movq   $0x0,0x650280(,%rdx,8)
  400a50:	00 00 00 00 00 
  400a55:	48 c7 04 d5 80 fc 65 	movq   $0x0,0x65fc80(,%rdx,8)
  400a5c:	00 00 00 00 00 
  400a61:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a65:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400a69:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400a6d:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400a71:	f2 0f 11 04 d5 80 20 	movsd  %xmm0,0x602080(,%rdx,8)
  400a78:	60 00 
  400a7a:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400a7e:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400a83:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400a87:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400a8b:	f2 0f 5c 05 0d 09 00 	subsd  0x90d(%rip),%xmm0        # 4013a0 <_IO_stdin_used+0x60>
  400a92:	00 
  400a93:	f2 0f 11 04 d5 80 1a 	movsd  %xmm0,0x611a80(,%rdx,8)
  400a9a:	61 00 
  400a9c:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aa0:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400aa4:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400aa8:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400aac:	f2 0f 5c 05 ec 08 00 	subsd  0x8ec(%rip),%xmm0        # 4013a0 <_IO_stdin_used+0x60>
  400ab3:	00 
  400ab4:	f2 0f 11 04 d5 80 14 	movsd  %xmm0,0x621480(,%rdx,8)
  400abb:	62 00 
  400abd:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400ac1:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
  400ac6:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400aca:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400ace:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400ad2:	f2 0f 11 04 d5 80 0e 	movsd  %xmm0,0x630e80(,%rdx,8)
  400ad9:	63 00 
  400adb:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400adf:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400ae3:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400ae7:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400aeb:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400aef:	f2 0f 11 04 d5 80 08 	movsd  %xmm0,0x640880(,%rdx,8)
  400af6:	64 00 
  400af8:	0f 85 e2 fe ff ff    	jne    4009e0 <_Z13instantiationv+0x30>
  400afe:	48 83 c4 20          	add    $0x20,%rsp
  400b02:	5b                   	pop    %rbx
  400b03:	5d                   	pop    %rbp
  400b04:	41 5c                	pop    %r12
  400b06:	41 5d                	pop    %r13
  400b08:	41 5e                	pop    %r14
  400b0a:	c3                   	retq   
  400b0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b10 <_Z10run_simplev>:
  400b10:	f2 0f 10 25 78 08 00 	movsd  0x878(%rip),%xmm4        # 401390 <_IO_stdin_used+0x50>
  400b17:	00 
  400b18:	ba 50 c3 00 00       	mov    $0xc350,%edx
  400b1d:	66 0f 28 15 9b 08 00 	movapd 0x89b(%rip),%xmm2        # 4013c0 <_IO_stdin_used+0x80>
  400b24:	00 
  400b25:	0f 1f 00             	nopl   (%rax)
  400b28:	b8 a0 0e 63 00       	mov    $0x630ea0,%eax
  400b2d:	f2 0f 10 0d 63 f7 24 	movsd  0x24f763(%rip),%xmm1        # 650298 <__Bodydata_buffer+0x4e218>
  400b34:	00 
  400b35:	f2 0f 10 05 5b f1 25 	movsd  0x25f15b(%rip),%xmm0        # 65fc98 <__Bodydata_buffer+0x5dc18>
  400b3c:	00 
  400b3d:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b41:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400b45:	f2 0f 10 1d 4b 15 20 	movsd  0x20154b(%rip),%xmm3        # 602098 <__Bodydata_buffer+0x18>
  400b4c:	00 
  400b4d:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  400b51:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b55:	f2 0f 58 0d 3b 03 23 	addsd  0x23033b(%rip),%xmm1        # 630e98 <__Bodydata_buffer+0x2ee18>
  400b5c:	00 
  400b5d:	f2 0f 11 0d 33 03 23 	movsd  %xmm1,0x230333(%rip)        # 630e98 <__Bodydata_buffer+0x2ee18>
  400b64:	00 
  400b65:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400b69:	f2 0f 58 05 27 fd 23 	addsd  0x23fd27(%rip),%xmm0        # 640898 <__Bodydata_buffer+0x3e818>
  400b70:	00 
  400b71:	f2 0f 58 0d 1f 0f 21 	addsd  0x210f1f(%rip),%xmm1        # 611a98 <__Bodydata_buffer+0xfa18>
  400b78:	00 
  400b79:	f2 0f 11 05 17 fd 23 	movsd  %xmm0,0x23fd17(%rip)        # 640898 <__Bodydata_buffer+0x3e818>
  400b80:	00 
  400b81:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400b85:	f2 0f 11 0d 0b 0f 21 	movsd  %xmm1,0x210f0b(%rip)        # 611a98 <__Bodydata_buffer+0xfa18>
  400b8c:	00 
  400b8d:	f2 0f 58 05 03 09 22 	addsd  0x220903(%rip),%xmm0        # 621498 <__Bodydata_buffer+0x1f418>
  400b94:	00 
  400b95:	f2 0f 11 05 fb 08 22 	movsd  %xmm0,0x2208fb(%rip)        # 621498 <__Bodydata_buffer+0x1f418>
  400b9c:	00 
  400b9d:	0f 1f 00             	nopl   (%rax)
  400ba0:	66 0f 28 88 00 f4 01 	movapd 0x1f400(%rax),%xmm1
  400ba7:	00 
  400ba8:	48 83 c0 10          	add    $0x10,%rax
  400bac:	66 0f 28 80 f0 ed 02 	movapd 0x2edf0(%rax),%xmm0
  400bb3:	00 
  400bb4:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400bb8:	66 0f 28 98 f0 11 fd 	movapd -0x2ee10(%rax),%xmm3
  400bbf:	ff 
  400bc0:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400bc4:	66 0f 5e cb          	divpd  %xmm3,%xmm1
  400bc8:	66 0f 5e c3          	divpd  %xmm3,%xmm0
  400bcc:	66 0f 58 48 f0       	addpd  -0x10(%rax),%xmm1
  400bd1:	0f 29 48 f0          	movaps %xmm1,-0x10(%rax)
  400bd5:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400bd9:	66 0f 58 88 f0 0b fe 	addpd  -0x1f410(%rax),%xmm1
  400be0:	ff 
  400be1:	66 0f 58 80 f0 f9 00 	addpd  0xf9f0(%rax),%xmm0
  400be8:	00 
  400be9:	0f 29 88 f0 0b fe ff 	movaps %xmm1,-0x1f410(%rax)
  400bf0:	0f 29 80 f0 f9 00 00 	movaps %xmm0,0xf9f0(%rax)
  400bf7:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400bfb:	66 0f 58 80 f0 05 ff 	addpd  -0xfa10(%rax),%xmm0
  400c02:	ff 
  400c03:	0f 29 80 f0 05 ff ff 	movaps %xmm0,-0xfa10(%rax)
  400c0a:	48 3d 90 08 64 00    	cmp    $0x640890,%rax
  400c10:	75 8e                	jne    400ba0 <_Z10run_simplev+0x90>
  400c12:	f2 0f 10 0d 76 f0 25 	movsd  0x25f076(%rip),%xmm1        # 65fc90 <__Bodydata_buffer+0x5dc10>
  400c19:	00 
  400c1a:	83 ea 01             	sub    $0x1,%edx
  400c1d:	f2 0f 10 05 6b ea 26 	movsd  0x26ea6b(%rip),%xmm0        # 66f690 <_ZStL8__ioinit>
  400c24:	00 
  400c25:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400c29:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c2d:	f2 0f 5e 0d 5b 0e 21 	divsd  0x210e5b(%rip),%xmm1        # 611a90 <__Bodydata_buffer+0xfa10>
  400c34:	00 
  400c35:	f2 0f 5e 05 53 0e 21 	divsd  0x210e53(%rip),%xmm0        # 611a90 <__Bodydata_buffer+0xfa10>
  400c3c:	00 
  400c3d:	f2 0f 58 0d 4b fc 23 	addsd  0x23fc4b(%rip),%xmm1        # 640890 <__Bodydata_buffer+0x3e810>
  400c44:	00 
  400c45:	f2 0f 11 0d 43 fc 23 	movsd  %xmm1,0x23fc43(%rip)        # 640890 <__Bodydata_buffer+0x3e810>
  400c4c:	00 
  400c4d:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
  400c51:	f2 0f 58 0d 37 08 22 	addsd  0x220837(%rip),%xmm1        # 621490 <__Bodydata_buffer+0x1f410>
  400c58:	00 
  400c59:	f2 0f 58 05 2f f6 24 	addsd  0x24f62f(%rip),%xmm0        # 650290 <__Bodydata_buffer+0x4e210>
  400c60:	00 
  400c61:	f2 0f 11 0d 27 08 22 	movsd  %xmm1,0x220827(%rip)        # 621490 <__Bodydata_buffer+0x1f410>
  400c68:	00 
  400c69:	f2 0f 11 05 1f f6 24 	movsd  %xmm0,0x24f61f(%rip)        # 650290 <__Bodydata_buffer+0x4e210>
  400c70:	00 
  400c71:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400c75:	f2 0f 58 05 13 02 23 	addsd  0x230213(%rip),%xmm0        # 630e90 <__Bodydata_buffer+0x2ee10>
  400c7c:	00 
  400c7d:	f2 0f 11 05 0b 02 23 	movsd  %xmm0,0x23020b(%rip)        # 630e90 <__Bodydata_buffer+0x2ee10>
  400c84:	00 
  400c85:	0f 85 9d fe ff ff    	jne    400b28 <_Z10run_simplev+0x18>
  400c8b:	f3 c3                	repz retq 
  400c8d:	0f 1f 00             	nopl   (%rax)

0000000000400c90 <_Z14run_simulationv>:
  400c90:	41 56                	push   %r14
  400c92:	41 55                	push   %r13
  400c94:	41 be 05 00 00 00    	mov    $0x5,%r14d
  400c9a:	41 54                	push   %r12
  400c9c:	55                   	push   %rbp
  400c9d:	53                   	push   %rbx
  400c9e:	48 83 ec 30          	sub    $0x30,%rsp
  400ca2:	f2 0f 10 3d e6 06 00 	movsd  0x6e6(%rip),%xmm7        # 401390 <_IO_stdin_used+0x50>
  400ca9:	00 
  400caa:	f2 44 0f 10 05 05 07 	movsd  0x705(%rip),%xmm8        # 4013b8 <_IO_stdin_used+0x78>
  400cb1:	00 00 
  400cb3:	f2 0f 10 35 ed 06 00 	movsd  0x6ed(%rip),%xmm6        # 4013a8 <_IO_stdin_used+0x68>
  400cba:	00 
  400cbb:	f2 0f 10 2d ed 06 00 	movsd  0x6ed(%rip),%xmm5        # 4013b0 <_IO_stdin_used+0x70>
  400cc2:	00 
  400cc3:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400cc8:	31 f6                	xor    %esi,%esi
  400cca:	bf 98 02 65 00       	mov    $0x650298,%edi
  400ccf:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400cd5:	41 bc 98 1a 61 00    	mov    $0x611a98,%r12d
  400cdb:	45 31 ed             	xor    %r13d,%r13d
  400cde:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400ce4:	f2 0f 11 7c 24 08    	movsd  %xmm7,0x8(%rsp)
  400cea:	f2 44 0f 11 04 24    	movsd  %xmm8,(%rsp)
  400cf0:	e8 1b fa ff ff       	callq  400710 <memset@plt>
  400cf5:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400cfa:	31 f6                	xor    %esi,%esi
  400cfc:	bf 98 fc 65 00       	mov    $0x65fc98,%edi
  400d01:	e8 0a fa ff ff       	callq  400710 <memset@plt>
  400d06:	f2 44 0f 10 04 24    	movsd  (%rsp),%xmm8
  400d0c:	f2 0f 10 7c 24 08    	movsd  0x8(%rsp),%xmm7
  400d12:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400d18:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400d1e:	66 90                	xchg   %ax,%ax
  400d20:	49 83 c5 01          	add    $0x1,%r13
  400d24:	bb 98 02 65 00       	mov    $0x650298,%ebx
  400d29:	31 ed                	xor    %ebp,%ebp
  400d2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400d30:	48 83 c5 01          	add    $0x1,%rbp
  400d34:	49 39 ed             	cmp    %rbp,%r13
  400d37:	0f 84 9c 00 00 00    	je     400dd9 <_Z14run_simulationv+0x149>
  400d3d:	f2 41 0f 10 8c 24 00 	movsd  0xfa00(%r12),%xmm1
  400d44:	fa 00 00 
  400d47:	f2 41 0f 10 1c 24    	movsd  (%r12),%xmm3
  400d4d:	f2 0f 5c 8b 00 12 fd 	subsd  -0x2ee00(%rbx),%xmm1
  400d54:	ff 
  400d55:	f2 0f 5c 9b 00 18 fc 	subsd  -0x3e800(%rbx),%xmm3
  400d5c:	ff 
  400d5d:	66 44 0f 28 c9       	movapd %xmm1,%xmm9
  400d62:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400d66:	66 41 0f 28 c1       	movapd %xmm9,%xmm0
  400d6b:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400d6f:	f2 41 0f 59 c1       	mulsd  %xmm9,%xmm0
  400d74:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400d78:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400d7c:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400d80:	0f 8a 6a 01 00 00    	jp     400ef0 <_Z14run_simulationv+0x260>
  400d86:	f2 0f 10 93 00 1e fb 	movsd  -0x4e200(%rbx),%xmm2
  400d8d:	ff 
  400d8e:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400d92:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400d96:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d9a:	f2 41 0f 59 94 24 00 	mulsd  -0xfa00(%r12),%xmm2
  400da1:	06 ff ff 
  400da4:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400da8:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400dac:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400db0:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400db4:	f2 41 0f 59 c9       	mulsd  %xmm9,%xmm1
  400db9:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400dbd:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400dc1:	f2 0f 58 1b          	addsd  (%rbx),%xmm3
  400dc5:	f2 0f 11 1b          	movsd  %xmm3,(%rbx)
  400dc9:	f2 0f 58 8b 00 fa 00 	addsd  0xfa00(%rbx),%xmm1
  400dd0:	00 
  400dd1:	f2 0f 11 8b 00 fa 00 	movsd  %xmm1,0xfa00(%rbx)
  400dd8:	00 
  400dd9:	48 83 c3 08          	add    $0x8,%rbx
  400ddd:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400de4:	0f 85 46 ff ff ff    	jne    400d30 <_Z14run_simulationv+0xa0>
  400dea:	49 83 c4 08          	add    $0x8,%r12
  400dee:	49 81 fd 40 1f 00 00 	cmp    $0x1f40,%r13
  400df5:	0f 85 25 ff ff ff    	jne    400d20 <_Z14run_simulationv+0x90>
  400dfb:	f2 0f 10 25 cd 05 00 	movsd  0x5cd(%rip),%xmm4        # 4013d0 <_IO_stdin_used+0x90>
  400e02:	00 
  400e03:	b8 98 0e 63 00       	mov    $0x630e98,%eax
  400e08:	f2 44 0f 10 0d 8f 05 	movsd  0x58f(%rip),%xmm9        # 4013a0 <_IO_stdin_used+0x60>
  400e0f:	00 00 
  400e11:	66 44 0f 28 d4       	movapd %xmm4,%xmm10
  400e16:	66 45 0f 28 d9       	movapd %xmm9,%xmm11
  400e1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400e20:	f2 0f 10 88 00 f4 01 	movsd  0x1f400(%rax),%xmm1
  400e27:	00 
  400e28:	f2 0f 10 90 00 12 fd 	movsd  -0x2ee00(%rax),%xmm2
  400e2f:	ff 
  400e30:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400e34:	f2 0f 10 80 00 ee 02 	movsd  0x2ee00(%rax),%xmm0
  400e3b:	00 
  400e3c:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400e40:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400e44:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400e48:	f2 0f 58 08          	addsd  (%rax),%xmm1
  400e4c:	66 0f 28 d9          	movapd %xmm1,%xmm3
  400e50:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400e54:	f2 0f 59 df          	mulsd  %xmm7,%xmm3
  400e58:	f2 0f 58 80 00 fa 00 	addsd  0xfa00(%rax),%xmm0
  400e5f:	00 
  400e60:	f2 0f 58 98 00 0c fe 	addsd  -0x1f400(%rax),%xmm3
  400e67:	ff 
  400e68:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400e6c:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400e73:	00 
  400e74:	66 44 0f 2e c3       	ucomisd %xmm3,%xmm8
  400e79:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400e7d:	f2 0f 11 98 00 0c fe 	movsd  %xmm3,-0x1f400(%rax)
  400e84:	ff 
  400e85:	f2 0f 58 90 00 06 ff 	addsd  -0xfa00(%rax),%xmm2
  400e8c:	ff 
  400e8d:	f2 0f 11 90 00 06 ff 	movsd  %xmm2,-0xfa00(%rax)
  400e94:	ff 
  400e95:	77 07                	ja     400e9e <_Z14run_simulationv+0x20e>
  400e97:	66 41 0f 2e d9       	ucomisd %xmm9,%xmm3
  400e9c:	76 08                	jbe    400ea6 <_Z14run_simulationv+0x216>
  400e9e:	66 0f 57 cc          	xorpd  %xmm4,%xmm1
  400ea2:	f2 0f 11 08          	movsd  %xmm1,(%rax)
  400ea6:	66 44 0f 2e c2       	ucomisd %xmm2,%xmm8
  400eab:	77 07                	ja     400eb4 <_Z14run_simulationv+0x224>
  400ead:	66 41 0f 2e d3       	ucomisd %xmm11,%xmm2
  400eb2:	76 0d                	jbe    400ec1 <_Z14run_simulationv+0x231>
  400eb4:	66 41 0f 57 c2       	xorpd  %xmm10,%xmm0
  400eb9:	f2 0f 11 80 00 fa 00 	movsd  %xmm0,0xfa00(%rax)
  400ec0:	00 
  400ec1:	48 83 c0 08          	add    $0x8,%rax
  400ec5:	48 3d 98 08 64 00    	cmp    $0x640898,%rax
  400ecb:	0f 85 4f ff ff ff    	jne    400e20 <_Z14run_simulationv+0x190>
  400ed1:	41 83 ee 01          	sub    $0x1,%r14d
  400ed5:	0f 85 e8 fd ff ff    	jne    400cc3 <_Z14run_simulationv+0x33>
  400edb:	48 83 c4 30          	add    $0x30,%rsp
  400edf:	5b                   	pop    %rbx
  400ee0:	5d                   	pop    %rbp
  400ee1:	41 5c                	pop    %r12
  400ee3:	41 5d                	pop    %r13
  400ee5:	41 5e                	pop    %r14
  400ee7:	c3                   	retq   
  400ee8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400eef:	00 
  400ef0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400ef6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400efc:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400f02:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400f09:	f2 44 0f 11 4c 24 08 	movsd  %xmm9,0x8(%rsp)
  400f10:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  400f15:	e8 76 f8 ff ff       	callq  400790 <sqrt@plt>
  400f1a:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400f20:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400f24:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400f2a:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400f30:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400f37:	f2 44 0f 10 4c 24 08 	movsd  0x8(%rsp),%xmm9
  400f3e:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  400f43:	e9 3e fe ff ff       	jmpq   400d86 <_Z14run_simulationv+0xf6>
  400f48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f4f:	00 

0000000000400f50 <_Z12float_as_intf>:
  400f50:	66 0f 7e c0          	movd   %xmm0,%eax
  400f54:	c3                   	retq   
  400f55:	90                   	nop
  400f56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f5d:	00 00 00 

0000000000400f60 <_ZN4Body7storageEv>:
  400f60:	b8 80 20 60 00       	mov    $0x602080,%eax
  400f65:	c3                   	retq   
  400f66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f6d:	00 00 00 

0000000000400f70 <_Z19codegen_reset_forceP4Body>:
  400f70:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400f74:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  400f7b:	f2 0f 11 04 fd 80 20 	movsd  %xmm0,0x602080(,%rdi,8)
  400f82:	60 00 
  400f84:	f2 0f 11 04 fd 80 1a 	movsd  %xmm0,0x611a80(,%rdi,8)
  400f8b:	61 00 
  400f8d:	c3                   	retq   
  400f8e:	66 90                	xchg   %ax,%ax

0000000000400f90 <_Z17codegen_add_forceP4BodyS0_>:
  400f90:	48 39 f7             	cmp    %rsi,%rdi
  400f93:	0f 84 d1 00 00 00    	je     40106a <_Z17codegen_add_forceP4BodyS0_+0xda>
  400f99:	41 54                	push   %r12
  400f9b:	55                   	push   %rbp
  400f9c:	48 8d ae 42 1f 00 00 	lea    0x1f42(%rsi),%rbp
  400fa3:	53                   	push   %rbx
  400fa4:	4c 8d a7 42 1f 00 00 	lea    0x1f42(%rdi),%r12
  400fab:	48 83 ec 10          	sub    $0x10,%rsp
  400faf:	4a 8d 1c e5 00 00 00 	lea    0x0(,%r12,8),%rbx
  400fb6:	00 
  400fb7:	f2 0f 10 0c ed 80 1a 	movsd  0x611a80(,%rbp,8),%xmm1
  400fbe:	61 00 
  400fc0:	f2 0f 10 1c ed 80 20 	movsd  0x602080(,%rbp,8),%xmm3
  400fc7:	60 00 
  400fc9:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400fcd:	f2 42 0f 5c 1c e5 80 	subsd  0x602080(,%r12,8),%xmm3
  400fd4:	20 60 00 
  400fd7:	f2 42 0f 5c 2c e5 80 	subsd  0x611a80(,%r12,8),%xmm5
  400fde:	1a 61 00 
  400fe1:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400fe5:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400fe9:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400fed:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400ff1:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400ff5:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400ff9:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400ffd:	7a 6d                	jp     40106c <_Z17codegen_add_forceP4BodyS0_+0xdc>
  400fff:	f2 0f 10 15 a1 03 00 	movsd  0x3a1(%rip),%xmm2        # 4013a8 <_IO_stdin_used+0x68>
  401006:	00 
  401007:	66 0f 28 c4          	movapd %xmm4,%xmm0
  40100b:	f2 42 0f 59 14 e5 80 	mulsd  0x5f2680(,%r12,8),%xmm2
  401012:	26 5f 00 
  401015:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  401019:	f2 0f 58 05 8f 03 00 	addsd  0x38f(%rip),%xmm0        # 4013b0 <_IO_stdin_used+0x70>
  401020:	00 
  401021:	f2 0f 59 14 ed 80 26 	mulsd  0x5f2680(,%rbp,8),%xmm2
  401028:	5f 00 
  40102a:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  40102e:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401032:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  401036:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  40103a:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  40103e:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  401042:	f2 0f 58 9b 80 08 64 	addsd  0x640880(%rbx),%xmm3
  401049:	00 
  40104a:	f2 0f 11 9b 80 08 64 	movsd  %xmm3,0x640880(%rbx)
  401051:	00 
  401052:	f2 0f 58 8b 80 02 65 	addsd  0x650280(%rbx),%xmm1
  401059:	00 
  40105a:	f2 0f 11 8b 80 02 65 	movsd  %xmm1,0x650280(%rbx)
  401061:	00 
  401062:	48 83 c4 10          	add    $0x10,%rsp
  401066:	5b                   	pop    %rbx
  401067:	5d                   	pop    %rbp
  401068:	41 5c                	pop    %r12
  40106a:	f3 c3                	repz retq 
  40106c:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401072:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  401077:	e8 14 f7 ff ff       	callq  400790 <sqrt@plt>
  40107c:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  401082:	66 0f 28 e0          	movapd %xmm0,%xmm4
  401086:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  40108b:	e9 6f ff ff ff       	jmpq   400fff <_Z17codegen_add_forceP4BodyS0_+0x6f>

0000000000401090 <_Z14codegen_updateP4Bodyd>:
  401090:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  401097:	f2 0f 10 15 19 03 00 	movsd  0x319(%rip),%xmm2        # 4013b8 <_IO_stdin_used+0x78>
  40109e:	00 
  40109f:	f2 0f 10 0c fd 80 20 	movsd  0x602080(,%rdi,8),%xmm1
  4010a6:	60 00 
  4010a8:	48 8d 04 fd 00 06 ff 	lea    -0xfa00(,%rdi,8),%rax
  4010af:	ff 
  4010b0:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4010b4:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  4010bb:	5b 00 
  4010bd:	f2 0f 58 0c fd 80 2c 	addsd  0x5e2c80(,%rdi,8),%xmm1
  4010c4:	5e 00 
  4010c6:	f2 0f 11 0c fd 80 2c 	movsd  %xmm1,0x5e2c80(,%rdi,8)
  4010cd:	5e 00 
  4010cf:	f2 0f 10 0c fd 80 1a 	movsd  0x611a80(,%rdi,8),%xmm1
  4010d6:	61 00 
  4010d8:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4010dc:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  4010e3:	5b 00 
  4010e5:	f2 0f 58 88 80 20 60 	addsd  0x602080(%rax),%xmm1
  4010ec:	00 
  4010ed:	f2 0f 11 88 80 20 60 	movsd  %xmm1,0x602080(%rax)
  4010f4:	00 
  4010f5:	f2 0f 10 0c fd 80 2c 	movsd  0x5e2c80(,%rdi,8),%xmm1
  4010fc:	5e 00 
  4010fe:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401102:	f2 0f 58 0c fd 80 38 	addsd  0x5c3880(,%rdi,8),%xmm1
  401109:	5c 00 
  40110b:	f2 0f 11 0c fd 80 38 	movsd  %xmm1,0x5c3880(,%rdi,8)
  401112:	5c 00 
  401114:	66 0f 2e d1          	ucomisd %xmm1,%xmm2
  401118:	f2 0f 59 80 80 20 60 	mulsd  0x602080(%rax),%xmm0
  40111f:	00 
  401120:	f2 0f 58 04 fd 80 32 	addsd  0x5d3280(,%rdi,8),%xmm0
  401127:	5d 00 
  401129:	f2 0f 11 04 fd 80 32 	movsd  %xmm0,0x5d3280(,%rdi,8)
  401130:	5d 00 
  401132:	77 0a                	ja     40113e <_Z14codegen_updateP4Bodyd+0xae>
  401134:	66 0f 2e 0d 64 02 00 	ucomisd 0x264(%rip),%xmm1        # 4013a0 <_IO_stdin_used+0x60>
  40113b:	00 
  40113c:	76 27                	jbe    401165 <_Z14codegen_updateP4Bodyd+0xd5>
  40113e:	f2 0f 10 04 fd 80 2c 	movsd  0x5e2c80(,%rdi,8),%xmm0
  401145:	5e 00 
  401147:	f2 0f 10 0d 81 02 00 	movsd  0x281(%rip),%xmm1        # 4013d0 <_IO_stdin_used+0x90>
  40114e:	00 
  40114f:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401153:	f2 0f 11 04 fd 80 2c 	movsd  %xmm0,0x5e2c80(,%rdi,8)
  40115a:	5e 00 
  40115c:	f2 0f 10 04 fd 80 32 	movsd  0x5d3280(,%rdi,8),%xmm0
  401163:	5d 00 
  401165:	66 0f 2e d0          	ucomisd %xmm0,%xmm2
  401169:	77 0a                	ja     401175 <_Z14codegen_updateP4Bodyd+0xe5>
  40116b:	66 0f 2e 05 2d 02 00 	ucomisd 0x22d(%rip),%xmm0        # 4013a0 <_IO_stdin_used+0x60>
  401172:	00 
  401173:	76 23                	jbe    401198 <_Z14codegen_updateP4Bodyd+0x108>
  401175:	f2 0f 10 80 80 20 60 	movsd  0x602080(%rax),%xmm0
  40117c:	00 
  40117d:	f2 0f 10 0d 4b 02 00 	movsd  0x24b(%rip),%xmm1        # 4013d0 <_IO_stdin_used+0x90>
  401184:	00 
  401185:	66 0f 57 c1          	xorpd  %xmm1,%xmm0
  401189:	f2 0f 11 80 80 20 60 	movsd  %xmm0,0x602080(%rax)
  401190:	00 
  401191:	c3                   	retq   
  401192:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  401198:	f3 c3                	repz retq 
  40119a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004011a0 <_Z26codegen_reset_force_manualP4Body>:
  4011a0:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4011a4:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  4011ab:	f2 0f 11 04 fd 80 20 	movsd  %xmm0,0x602080(,%rdi,8)
  4011b2:	60 00 
  4011b4:	f2 0f 11 04 fd 80 1a 	movsd  %xmm0,0x611a80(,%rdi,8)
  4011bb:	61 00 
  4011bd:	c3                   	retq   
  4011be:	66 90                	xchg   %ax,%ax

00000000004011c0 <_Z24codegen_reset_force_halfP4Body>:
  4011c0:	48 c7 04 fd 90 02 65 	movq   $0x0,0x650290(,%rdi,8)
  4011c7:	00 00 00 00 00 
  4011cc:	c3                   	retq   
  4011cd:	0f 1f 00             	nopl   (%rax)

00000000004011d0 <_Z18codegen_reset_massP4Body>:
  4011d0:	48 c7 04 fd 90 20 60 	movq   $0x0,0x602090(,%rdi,8)
  4011d7:	00 00 00 00 00 
  4011dc:	c3                   	retq   
  4011dd:	0f 1f 00             	nopl   (%rax)

00000000004011e0 <_Z22codengen_simple_updateP4Bodyd>:
  4011e0:	48 81 c7 42 9c 00 00 	add    $0x9c42,%rdi
  4011e7:	f2 0f 10 0c fd 80 20 	movsd  0x602080(,%rdi,8),%xmm1
  4011ee:	60 00 
  4011f0:	48 8d 04 fd 00 06 ff 	lea    -0xfa00(,%rdi,8),%rax
  4011f7:	ff 
  4011f8:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011fc:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  401203:	5b 00 
  401205:	f2 0f 58 0c fd 80 2c 	addsd  0x5e2c80(,%rdi,8),%xmm1
  40120c:	5e 00 
  40120e:	f2 0f 11 0c fd 80 2c 	movsd  %xmm1,0x5e2c80(,%rdi,8)
  401215:	5e 00 
  401217:	f2 0f 10 0c fd 80 1a 	movsd  0x611a80(,%rdi,8),%xmm1
  40121e:	61 00 
  401220:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  401224:	f2 0f 5e 0c fd 80 3e 	divsd  0x5b3e80(,%rdi,8),%xmm1
  40122b:	5b 00 
  40122d:	f2 0f 58 88 80 20 60 	addsd  0x602080(%rax),%xmm1
  401234:	00 
  401235:	f2 0f 11 88 80 20 60 	movsd  %xmm1,0x602080(%rax)
  40123c:	00 
  40123d:	f2 0f 10 0c fd 80 2c 	movsd  0x5e2c80(,%rdi,8),%xmm1
  401244:	5e 00 
  401246:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  40124a:	f2 0f 58 0c fd 80 38 	addsd  0x5c3880(,%rdi,8),%xmm1
  401251:	5c 00 
  401253:	f2 0f 11 0c fd 80 38 	movsd  %xmm1,0x5c3880(,%rdi,8)
  40125a:	5c 00 
  40125c:	f2 0f 59 80 80 20 60 	mulsd  0x602080(%rax),%xmm0
  401263:	00 
  401264:	f2 0f 58 04 fd 80 32 	addsd  0x5d3280(,%rdi,8),%xmm0
  40126b:	5d 00 
  40126d:	f2 0f 11 04 fd 80 32 	movsd  %xmm0,0x5d3280(,%rdi,8)
  401274:	5d 00 
  401276:	c3                   	retq   
  401277:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40127e:	00 00 

0000000000401280 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401280:	55                   	push   %rbp
  401281:	53                   	push   %rbx
  401282:	48 89 fd             	mov    %rdi,%rbp
  401285:	48 83 ec 08          	sub    $0x8,%rsp
  401289:	e8 e2 f4 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40128e:	48 89 c3             	mov    %rax,%rbx
  401291:	ff d5                	callq  *%rbp
  401293:	e8 d8 f4 ff ff       	callq  400770 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401298:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40129f:	de 1b 43 
  4012a2:	48 29 d8             	sub    %rbx,%rax
  4012a5:	48 83 c4 08          	add    $0x8,%rsp
  4012a9:	48 89 c1             	mov    %rax,%rcx
  4012ac:	48 f7 ea             	imul   %rdx
  4012af:	48 c1 f9 3f          	sar    $0x3f,%rcx
  4012b3:	5b                   	pop    %rbx
  4012b4:	5d                   	pop    %rbp
  4012b5:	48 c1 fa 12          	sar    $0x12,%rdx
  4012b9:	48 89 d0             	mov    %rdx,%rax
  4012bc:	48 29 c8             	sub    %rcx,%rax
  4012bf:	c3                   	retq   

00000000004012c0 <__libc_csu_init>:
  4012c0:	41 57                	push   %r15
  4012c2:	41 56                	push   %r14
  4012c4:	41 89 ff             	mov    %edi,%r15d
  4012c7:	41 55                	push   %r13
  4012c9:	41 54                	push   %r12
  4012cb:	4c 8d 25 16 0b 20 00 	lea    0x200b16(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  4012d2:	55                   	push   %rbp
  4012d3:	48 8d 2d 1e 0b 20 00 	lea    0x200b1e(%rip),%rbp        # 601df8 <__init_array_end>
  4012da:	53                   	push   %rbx
  4012db:	49 89 f6             	mov    %rsi,%r14
  4012de:	49 89 d5             	mov    %rdx,%r13
  4012e1:	4c 29 e5             	sub    %r12,%rbp
  4012e4:	48 83 ec 08          	sub    $0x8,%rsp
  4012e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4012ec:	e8 ef f3 ff ff       	callq  4006e0 <_init>
  4012f1:	48 85 ed             	test   %rbp,%rbp
  4012f4:	74 20                	je     401316 <__libc_csu_init+0x56>
  4012f6:	31 db                	xor    %ebx,%ebx
  4012f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4012ff:	00 
  401300:	4c 89 ea             	mov    %r13,%rdx
  401303:	4c 89 f6             	mov    %r14,%rsi
  401306:	44 89 ff             	mov    %r15d,%edi
  401309:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40130d:	48 83 c3 01          	add    $0x1,%rbx
  401311:	48 39 eb             	cmp    %rbp,%rbx
  401314:	75 ea                	jne    401300 <__libc_csu_init+0x40>
  401316:	48 83 c4 08          	add    $0x8,%rsp
  40131a:	5b                   	pop    %rbx
  40131b:	5d                   	pop    %rbp
  40131c:	41 5c                	pop    %r12
  40131e:	41 5d                	pop    %r13
  401320:	41 5e                	pop    %r14
  401322:	41 5f                	pop    %r15
  401324:	c3                   	retq   
  401325:	90                   	nop
  401326:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40132d:	00 00 00 

0000000000401330 <__libc_csu_fini>:
  401330:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401334 <_fini>:
  401334:	48 83 ec 08          	sub    $0x8,%rsp
  401338:	48 83 c4 08          	add    $0x8,%rsp
  40133c:	c3                   	retq   
