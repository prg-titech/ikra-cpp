
bin/g++_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

0000000000400718 <_init>:
  400718:	48 83 ec 08          	sub    $0x8,%rsp
  40071c:	48 8b 05 d5 18 20 00 	mov    0x2018d5(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  400723:	48 85 c0             	test   %rax,%rax
  400726:	74 05                	je     40072d <_init+0x15>
  400728:	e8 c3 00 00 00       	callq  4007f0 <sqrt@plt+0x10>
  40072d:	48 83 c4 08          	add    $0x8,%rsp
  400731:	c3                   	retq   

Disassembly of section .plt:

0000000000400740 <memset@plt-0x10>:
  400740:	ff 35 c2 18 20 00    	pushq  0x2018c2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400746:	ff 25 c4 18 20 00    	jmpq   *0x2018c4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40074c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400750 <memset@plt>:
  400750:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400756:	68 00 00 00 00       	pushq  $0x0
  40075b:	e9 e0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400760 <__printf_chk@plt>:
  400760:	ff 25 ba 18 20 00    	jmpq   *0x2018ba(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400766:	68 01 00 00 00       	pushq  $0x1
  40076b:	e9 d0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400770 <_ZNSt8ios_base4InitC1Ev@plt>:
  400770:	ff 25 b2 18 20 00    	jmpq   *0x2018b2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400776:	68 02 00 00 00       	pushq  $0x2
  40077b:	e9 c0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400780 <__libc_start_main@plt>:
  400780:	ff 25 aa 18 20 00    	jmpq   *0x2018aa(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400786:	68 03 00 00 00       	pushq  $0x3
  40078b:	e9 b0 ff ff ff       	jmpq   400740 <_init+0x28>

0000000000400790 <__cxa_atexit@plt>:
  400790:	ff 25 a2 18 20 00    	jmpq   *0x2018a2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400796:	68 04 00 00 00       	pushq  $0x4
  40079b:	e9 a0 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007a0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4007a0:	ff 25 9a 18 20 00    	jmpq   *0x20189a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  4007a6:	68 05 00 00 00       	pushq  $0x5
  4007ab:	e9 90 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007b0 <srand@plt>:
  4007b0:	ff 25 92 18 20 00    	jmpq   *0x201892(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  4007b6:	68 06 00 00 00       	pushq  $0x6
  4007bb:	e9 80 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  4007c0:	ff 25 8a 18 20 00    	jmpq   *0x20188a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  4007c6:	68 07 00 00 00       	pushq  $0x7
  4007cb:	e9 70 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007d0 <rand@plt>:
  4007d0:	ff 25 82 18 20 00    	jmpq   *0x201882(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  4007d6:	68 08 00 00 00       	pushq  $0x8
  4007db:	e9 60 ff ff ff       	jmpq   400740 <_init+0x28>

00000000004007e0 <sqrt@plt>:
  4007e0:	ff 25 7a 18 20 00    	jmpq   *0x20187a(%rip)        # 602060 <_GLOBAL_OFFSET_TABLE_+0x60>
  4007e6:	68 09 00 00 00       	pushq  $0x9
  4007eb:	e9 50 ff ff ff       	jmpq   400740 <_init+0x28>

Disassembly of section .plt.got:

00000000004007f0 <.plt.got>:
  4007f0:	ff 25 02 18 20 00    	jmpq   *0x201802(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  4007f6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400800 <main>:
  400800:	41 55                	push   %r13
  400802:	41 54                	push   %r12
  400804:	bf 30 0a 40 00       	mov    $0x400a30,%edi
  400809:	55                   	push   %rbp
  40080a:	53                   	push   %rbx
  40080b:	41 bd 8d 68 b7 6c    	mov    $0x6cb7688d,%r13d
  400811:	31 db                	xor    %ebx,%ebx
  400813:	48 83 ec 18          	sub    $0x18,%rsp
  400817:	e8 24 0b 00 00       	callq  401340 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40081c:	bf 60 0c 40 00       	mov    $0x400c60,%edi
  400821:	48 89 c5             	mov    %rax,%rbp
  400824:	e8 17 0b 00 00       	callq  401340 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400829:	bf 90 0b 40 00       	mov    $0x400b90,%edi
  40082e:	49 89 c4             	mov    %rax,%r12
  400831:	e8 0a 0b 00 00       	callq  401340 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400836:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  40083c:	49 89 c0             	mov    %rax,%r8
  40083f:	90                   	nop
  400840:	66 0f ef d2          	pxor   %xmm2,%xmm2
  400844:	f2 0f 10 04 dd a0 02 	movsd  0x6502a0(,%rbx,8),%xmm0
  40084b:	65 00 
  40084d:	66 0f ef db          	pxor   %xmm3,%xmm3
  400851:	f2 0f 5a d0          	cvtsd2ss %xmm0,%xmm2
  400855:	f2 0f 10 0c dd a0 08 	movsd  0x6408a0(,%rbx,8),%xmm1
  40085c:	64 00 
  40085e:	f2 0f 5a d9          	cvtsd2ss %xmm1,%xmm3
  400862:	66 0f 7e d1          	movd   %xmm2,%ecx
  400866:	41 01 c9             	add    %ecx,%r9d
  400869:	66 0f 7e d9          	movd   %xmm3,%ecx
  40086d:	44 01 c9             	add    %r9d,%ecx
  400870:	89 c8                	mov    %ecx,%eax
  400872:	41 f7 ed             	imul   %r13d
  400875:	89 c8                	mov    %ecx,%eax
  400877:	c1 f8 1f             	sar    $0x1f,%eax
  40087a:	c1 fa 13             	sar    $0x13,%edx
  40087d:	41 89 d1             	mov    %edx,%r9d
  400880:	41 29 c1             	sub    %eax,%r9d
  400883:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  40088a:	44 29 c9             	sub    %r9d,%ecx
  40088d:	48 83 fb 09          	cmp    $0x9,%rbx
  400891:	41 89 c9             	mov    %ecx,%r9d
  400894:	76 31                	jbe    4008c7 <main+0xc7>
  400896:	48 83 c3 01          	add    $0x1,%rbx
  40089a:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  4008a1:	75 9d                	jne    400840 <main+0x40>
  4008a3:	4c 89 e1             	mov    %r12,%rcx
  4008a6:	48 89 ea             	mov    %rbp,%rdx
  4008a9:	be 20 14 40 00       	mov    $0x401420,%esi
  4008ae:	bf 01 00 00 00       	mov    $0x1,%edi
  4008b3:	31 c0                	xor    %eax,%eax
  4008b5:	e8 a6 fe ff ff       	callq  400760 <__printf_chk@plt>
  4008ba:	48 83 c4 18          	add    $0x18,%rsp
  4008be:	31 c0                	xor    %eax,%eax
  4008c0:	5b                   	pop    %rbx
  4008c1:	5d                   	pop    %rbp
  4008c2:	41 5c                	pop    %r12
  4008c4:	41 5d                	pop    %r13
  4008c6:	c3                   	retq   
  4008c7:	48 89 da             	mov    %rbx,%rdx
  4008ca:	be 04 14 40 00       	mov    $0x401404,%esi
  4008cf:	bf 01 00 00 00       	mov    $0x1,%edi
  4008d4:	b8 02 00 00 00       	mov    $0x2,%eax
  4008d9:	4c 89 04 24          	mov    %r8,(%rsp)
  4008dd:	89 4c 24 0c          	mov    %ecx,0xc(%rsp)
  4008e1:	e8 7a fe ff ff       	callq  400760 <__printf_chk@plt>
  4008e6:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  4008eb:	4c 8b 04 24          	mov    (%rsp),%r8
  4008ef:	eb a5                	jmp    400896 <main+0x96>
  4008f1:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_GLOBAL__sub_I__Z11r_float2intf>:
  400900:	48 83 ec 08          	sub    $0x8,%rsp
  400904:	bf a0 f6 66 00       	mov    $0x66f6a0,%edi
  400909:	e8 62 fe ff ff       	callq  400770 <_ZNSt8ios_base4InitC1Ev@plt>
  40090e:	ba 70 20 60 00       	mov    $0x602070,%edx
  400913:	be a0 f6 66 00       	mov    $0x66f6a0,%esi
  400918:	bf a0 07 40 00       	mov    $0x4007a0,%edi
  40091d:	48 83 c4 08          	add    $0x8,%rsp
  400921:	e9 6a fe ff ff       	jmpq   400790 <__cxa_atexit@plt>
  400926:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40092d:	00 00 00 

0000000000400930 <_start>:
  400930:	31 ed                	xor    %ebp,%ebp
  400932:	49 89 d1             	mov    %rdx,%r9
  400935:	5e                   	pop    %rsi
  400936:	48 89 e2             	mov    %rsp,%rdx
  400939:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40093d:	50                   	push   %rax
  40093e:	54                   	push   %rsp
  40093f:	49 c7 c0 f0 13 40 00 	mov    $0x4013f0,%r8
  400946:	48 c7 c1 80 13 40 00 	mov    $0x401380,%rcx
  40094d:	48 c7 c7 00 08 40 00 	mov    $0x400800,%rdi
  400954:	e8 27 fe ff ff       	callq  400780 <__libc_start_main@plt>
  400959:	f4                   	hlt    
  40095a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400960 <deregister_tm_clones>:
  400960:	b8 7f 20 60 00       	mov    $0x60207f,%eax
  400965:	55                   	push   %rbp
  400966:	48 2d 78 20 60 00    	sub    $0x602078,%rax
  40096c:	48 83 f8 0e          	cmp    $0xe,%rax
  400970:	48 89 e5             	mov    %rsp,%rbp
  400973:	76 1b                	jbe    400990 <deregister_tm_clones+0x30>
  400975:	b8 00 00 00 00       	mov    $0x0,%eax
  40097a:	48 85 c0             	test   %rax,%rax
  40097d:	74 11                	je     400990 <deregister_tm_clones+0x30>
  40097f:	5d                   	pop    %rbp
  400980:	bf 78 20 60 00       	mov    $0x602078,%edi
  400985:	ff e0                	jmpq   *%rax
  400987:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40098e:	00 00 
  400990:	5d                   	pop    %rbp
  400991:	c3                   	retq   
  400992:	0f 1f 40 00          	nopl   0x0(%rax)
  400996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099d:	00 00 00 

00000000004009a0 <register_tm_clones>:
  4009a0:	be 78 20 60 00       	mov    $0x602078,%esi
  4009a5:	55                   	push   %rbp
  4009a6:	48 81 ee 78 20 60 00 	sub    $0x602078,%rsi
  4009ad:	48 c1 fe 03          	sar    $0x3,%rsi
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 89 f0             	mov    %rsi,%rax
  4009b7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4009bb:	48 01 c6             	add    %rax,%rsi
  4009be:	48 d1 fe             	sar    %rsi
  4009c1:	74 15                	je     4009d8 <register_tm_clones+0x38>
  4009c3:	b8 00 00 00 00       	mov    $0x0,%eax
  4009c8:	48 85 c0             	test   %rax,%rax
  4009cb:	74 0b                	je     4009d8 <register_tm_clones+0x38>
  4009cd:	5d                   	pop    %rbp
  4009ce:	bf 78 20 60 00       	mov    $0x602078,%edi
  4009d3:	ff e0                	jmpq   *%rax
  4009d5:	0f 1f 00             	nopl   (%rax)
  4009d8:	5d                   	pop    %rbp
  4009d9:	c3                   	retq   
  4009da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009e0 <__do_global_dtors_aux>:
  4009e0:	80 3d 99 16 20 00 00 	cmpb   $0x0,0x201699(%rip)        # 602080 <completed.7585>
  4009e7:	75 11                	jne    4009fa <__do_global_dtors_aux+0x1a>
  4009e9:	55                   	push   %rbp
  4009ea:	48 89 e5             	mov    %rsp,%rbp
  4009ed:	e8 6e ff ff ff       	callq  400960 <deregister_tm_clones>
  4009f2:	5d                   	pop    %rbp
  4009f3:	c6 05 86 16 20 00 01 	movb   $0x1,0x201686(%rip)        # 602080 <completed.7585>
  4009fa:	f3 c3                	repz retq 
  4009fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a00 <frame_dummy>:
  400a00:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400a05:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400a09:	75 05                	jne    400a10 <frame_dummy+0x10>
  400a0b:	eb 93                	jmp    4009a0 <register_tm_clones>
  400a0d:	0f 1f 00             	nopl   (%rax)
  400a10:	b8 00 00 00 00       	mov    $0x0,%eax
  400a15:	48 85 c0             	test   %rax,%rax
  400a18:	74 f1                	je     400a0b <frame_dummy+0xb>
  400a1a:	55                   	push   %rbp
  400a1b:	48 89 e5             	mov    %rsp,%rbp
  400a1e:	ff d0                	callq  *%rax
  400a20:	5d                   	pop    %rbp
  400a21:	e9 7a ff ff ff       	jmpq   4009a0 <register_tm_clones>
  400a26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2d:	00 00 00 

0000000000400a30 <_Z13instantiationv>:
  400a30:	41 56                	push   %r14
  400a32:	41 55                	push   %r13
  400a34:	bf 2a 00 00 00       	mov    $0x2a,%edi
  400a39:	41 54                	push   %r12
  400a3b:	55                   	push   %rbp
  400a3c:	45 31 ed             	xor    %r13d,%r13d
  400a3f:	53                   	push   %rbx
  400a40:	48 83 ec 20          	sub    $0x20,%rsp
  400a44:	e8 67 fd ff ff       	callq  4007b0 <srand@plt>
  400a49:	f2 0f 10 0d 0f 0a 00 	movsd  0xa0f(%rip),%xmm1        # 401460 <_IO_stdin_used+0x60>
  400a50:	00 
  400a51:	f2 0f 10 15 0f 0a 00 	movsd  0xa0f(%rip),%xmm2        # 401468 <_IO_stdin_used+0x68>
  400a58:	00 
  400a59:	f2 0f 10 1d 0f 0a 00 	movsd  0xa0f(%rip),%xmm3        # 401470 <_IO_stdin_used+0x70>
  400a60:	00 
  400a61:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400a68:	f2 0f 11 5c 24 18    	movsd  %xmm3,0x18(%rsp)
  400a6e:	49 83 c5 08          	add    $0x8,%r13
  400a72:	f2 0f 11 54 24 10    	movsd  %xmm2,0x10(%rsp)
  400a78:	f2 0f 11 4c 24 08    	movsd  %xmm1,0x8(%rsp)
  400a7e:	e8 4d fd ff ff       	callq  4007d0 <rand@plt>
  400a83:	89 c3                	mov    %eax,%ebx
  400a85:	e8 46 fd ff ff       	callq  4007d0 <rand@plt>
  400a8a:	41 89 c4             	mov    %eax,%r12d
  400a8d:	e8 3e fd ff ff       	callq  4007d0 <rand@plt>
  400a92:	89 c5                	mov    %eax,%ebp
  400a94:	e8 37 fd ff ff       	callq  4007d0 <rand@plt>
  400a99:	41 89 c6             	mov    %eax,%r14d
  400a9c:	e8 2f fd ff ff       	callq  4007d0 <rand@plt>
  400aa1:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aa5:	49 c7 85 98 1a 61 00 	movq   $0x0,0x611a98(%r13)
  400aac:	00 00 00 00 
  400ab0:	f2 0f 10 4c 24 08    	movsd  0x8(%rsp),%xmm1
  400ab6:	49 c7 85 98 20 60 00 	movq   $0x0,0x602098(%r13)
  400abd:	00 00 00 00 
  400ac1:	f2 0f 10 54 24 10    	movsd  0x10(%rsp),%xmm2
  400ac7:	f2 0f 2a c3          	cvtsi2sd %ebx,%xmm0
  400acb:	f2 0f 10 5c 24 18    	movsd  0x18(%rsp),%xmm3
  400ad1:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400ad5:	f2 0f 59 c2          	mulsd  %xmm2,%xmm0
  400ad9:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
  400add:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
  400ae1:	f2 41 0f 11 85 98 fc 	movsd  %xmm0,0x65fc98(%r13)
  400ae8:	65 00 
  400aea:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400aee:	f2 41 0f 2a c4       	cvtsi2sd %r12d,%xmm0
  400af3:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400af7:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400afb:	f2 0f 5c 05 75 09 00 	subsd  0x975(%rip),%xmm0        # 401478 <_IO_stdin_used+0x78>
  400b02:	00 
  400b03:	f2 41 0f 11 85 98 02 	movsd  %xmm0,0x650298(%r13)
  400b0a:	65 00 
  400b0c:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b10:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400b14:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b18:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b1c:	f2 0f 5c 05 54 09 00 	subsd  0x954(%rip),%xmm0        # 401478 <_IO_stdin_used+0x78>
  400b23:	00 
  400b24:	f2 41 0f 11 85 98 08 	movsd  %xmm0,0x640898(%r13)
  400b2b:	64 00 
  400b2d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b31:	f2 41 0f 2a c6       	cvtsi2sd %r14d,%xmm0
  400b36:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b3a:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b3e:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b42:	f2 41 0f 11 85 98 0e 	movsd  %xmm0,0x630e98(%r13)
  400b49:	63 00 
  400b4b:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b4f:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
  400b53:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b57:	f2 0f 5c c2          	subsd  %xmm2,%xmm0
  400b5b:	f2 0f 5e c3          	divsd  %xmm3,%xmm0
  400b5f:	f2 41 0f 11 85 98 14 	movsd  %xmm0,0x621498(%r13)
  400b66:	62 00 
  400b68:	49 81 fd 00 fa 00 00 	cmp    $0xfa00,%r13
  400b6f:	0f 85 f3 fe ff ff    	jne    400a68 <_Z13instantiationv+0x38>
  400b75:	48 83 c4 20          	add    $0x20,%rsp
  400b79:	5b                   	pop    %rbx
  400b7a:	5d                   	pop    %rbp
  400b7b:	41 5c                	pop    %r12
  400b7d:	41 5d                	pop    %r13
  400b7f:	41 5e                	pop    %r14
  400b81:	c3                   	retq   
  400b82:	0f 1f 40 00          	nopl   0x0(%rax)
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 

0000000000400b90 <_Z10run_simplev>:
  400b90:	f2 0f 10 15 d0 08 00 	movsd  0x8d0(%rip),%xmm2        # 401468 <_IO_stdin_used+0x68>
  400b97:	00 
  400b98:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400b9d:	31 d2                	xor    %edx,%edx
  400b9f:	90                   	nop
  400ba0:	f2 0f 10 ba a0 1a 61 	movsd  0x611aa0(%rdx),%xmm7
  400ba7:	00 
  400ba8:	b8 64 00 00 00       	mov    $0x64,%eax
  400bad:	f2 0f 10 b2 a0 20 60 	movsd  0x6020a0(%rdx),%xmm6
  400bb4:	00 
  400bb5:	f2 0f 59 fa          	mulsd  %xmm2,%xmm7
  400bb9:	f2 0f 59 f2          	mulsd  %xmm2,%xmm6
  400bbd:	f2 0f 10 9a a0 fc 65 	movsd  0x65fca0(%rdx),%xmm3
  400bc4:	00 
  400bc5:	f2 0f 10 8a a0 0e 63 	movsd  0x630ea0(%rdx),%xmm1
  400bcc:	00 
  400bcd:	f2 0f 5e fb          	divsd  %xmm3,%xmm7
  400bd1:	f2 0f 5e f3          	divsd  %xmm3,%xmm6
  400bd5:	f2 0f 10 82 a0 14 62 	movsd  0x6214a0(%rdx),%xmm0
  400bdc:	00 
  400bdd:	f2 0f 10 a2 a0 02 65 	movsd  0x6502a0(%rdx),%xmm4
  400be4:	00 
  400be5:	f2 0f 10 9a a0 08 64 	movsd  0x6408a0(%rdx),%xmm3
  400bec:	00 
  400bed:	0f 1f 00             	nopl   (%rax)
  400bf0:	f2 0f 58 cf          	addsd  %xmm7,%xmm1
  400bf4:	83 e8 01             	sub    $0x1,%eax
  400bf7:	f2 0f 58 c6          	addsd  %xmm6,%xmm0
  400bfb:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400bff:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400c03:	f2 0f 58 e5          	addsd  %xmm5,%xmm4
  400c07:	66 0f 28 e8          	movapd %xmm0,%xmm5
  400c0b:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  400c0f:	f2 0f 58 dd          	addsd  %xmm5,%xmm3
  400c13:	75 db                	jne    400bf0 <_Z10run_simplev+0x60>
  400c15:	f2 0f 11 8a a0 0e 63 	movsd  %xmm1,0x630ea0(%rdx)
  400c1c:	00 
  400c1d:	48 83 c2 08          	add    $0x8,%rdx
  400c21:	f2 0f 11 82 98 14 62 	movsd  %xmm0,0x621498(%rdx)
  400c28:	00 
  400c29:	f2 0f 11 a2 98 02 65 	movsd  %xmm4,0x650298(%rdx)
  400c30:	00 
  400c31:	f2 0f 11 9a 98 08 64 	movsd  %xmm3,0x640898(%rdx)
  400c38:	00 
  400c39:	48 81 fa 00 fa 00 00 	cmp    $0xfa00,%rdx
  400c40:	0f 85 5a ff ff ff    	jne    400ba0 <_Z10run_simplev+0x10>
  400c46:	83 e9 01             	sub    $0x1,%ecx
  400c49:	0f 85 4e ff ff ff    	jne    400b9d <_Z10run_simplev+0xd>
  400c4f:	f3 c3                	repz retq 
  400c51:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400c56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c5d:	00 00 00 

0000000000400c60 <_Z14run_simulationv>:
  400c60:	41 55                	push   %r13
  400c62:	41 54                	push   %r12
  400c64:	41 bd 05 00 00 00    	mov    $0x5,%r13d
  400c6a:	55                   	push   %rbp
  400c6b:	53                   	push   %rbx
  400c6c:	48 83 ec 38          	sub    $0x38,%rsp
  400c70:	f2 0f 10 3d f0 07 00 	movsd  0x7f0(%rip),%xmm7        # 401468 <_IO_stdin_used+0x68>
  400c77:	00 
  400c78:	f2 0f 10 35 00 08 00 	movsd  0x800(%rip),%xmm6        # 401480 <_IO_stdin_used+0x80>
  400c7f:	00 
  400c80:	f2 0f 10 2d 00 08 00 	movsd  0x800(%rip),%xmm5        # 401488 <_IO_stdin_used+0x88>
  400c87:	00 
  400c88:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c8d:	31 f6                	xor    %esi,%esi
  400c8f:	bf a0 1a 61 00       	mov    $0x611aa0,%edi
  400c94:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400c9a:	31 ed                	xor    %ebp,%ebp
  400c9c:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400ca2:	f2 0f 11 7c 24 08    	movsd  %xmm7,0x8(%rsp)
  400ca8:	e8 a3 fa ff ff       	callq  400750 <memset@plt>
  400cad:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400cb2:	31 f6                	xor    %esi,%esi
  400cb4:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400cb9:	e8 92 fa ff ff       	callq  400750 <memset@plt>
  400cbe:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400cc4:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400cca:	f2 0f 10 7c 24 08    	movsd  0x8(%rsp),%xmm7
  400cd0:	31 db                	xor    %ebx,%ebx
  400cd2:	45 31 e4             	xor    %r12d,%r12d
  400cd5:	0f 1f 00             	nopl   (%rax)
  400cd8:	4c 39 e5             	cmp    %r12,%rbp
  400cdb:	0f 84 a5 00 00 00    	je     400d86 <_Z14run_simulationv+0x126>
  400ce1:	f2 0f 10 0c ed a0 08 	movsd  0x6408a0(,%rbp,8),%xmm1
  400ce8:	64 00 
  400cea:	f2 0f 10 1c ed a0 02 	movsd  0x6502a0(,%rbp,8),%xmm3
  400cf1:	65 00 
  400cf3:	f2 0f 5c 8b a0 08 64 	subsd  0x6408a0(%rbx),%xmm1
  400cfa:	00 
  400cfb:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  400d02:	00 
  400d03:	66 44 0f 28 c1       	movapd %xmm1,%xmm8
  400d08:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400d0c:	66 41 0f 28 c0       	movapd %xmm8,%xmm0
  400d11:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400d15:	f2 41 0f 59 c0       	mulsd  %xmm8,%xmm0
  400d1a:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400d1e:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400d22:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400d26:	0f 8a 94 01 00 00    	jp     400ec0 <_Z14run_simulationv+0x260>
  400d2c:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  400d33:	00 
  400d34:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400d38:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400d3c:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d40:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400d47:	65 00 
  400d49:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400d4d:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400d51:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400d55:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400d59:	f2 41 0f 59 c8       	mulsd  %xmm8,%xmm1
  400d5e:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400d62:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400d66:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  400d6d:	00 
  400d6e:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  400d75:	00 
  400d76:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  400d7d:	00 
  400d7e:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  400d85:	00 
  400d86:	49 83 c4 01          	add    $0x1,%r12
  400d8a:	48 83 c3 08          	add    $0x8,%rbx
  400d8e:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400d95:	0f 85 3d ff ff ff    	jne    400cd8 <_Z14run_simulationv+0x78>
  400d9b:	48 83 c5 01          	add    $0x1,%rbp
  400d9f:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400da6:	0f 85 24 ff ff ff    	jne    400cd0 <_Z14run_simulationv+0x70>
  400dac:	f2 0f 10 1d ec 06 00 	movsd  0x6ec(%rip),%xmm3        # 4014a0 <_IO_stdin_used+0xa0>
  400db3:	00 
  400db4:	31 c0                	xor    %eax,%eax
  400db6:	f2 0f 10 25 ba 06 00 	movsd  0x6ba(%rip),%xmm4        # 401478 <_IO_stdin_used+0x78>
  400dbd:	00 
  400dbe:	66 44 0f 28 c3       	movapd %xmm3,%xmm8
  400dc3:	66 44 0f 28 cc       	movapd %xmm4,%xmm9
  400dc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dcf:	00 
  400dd0:	f2 0f 10 88 a0 1a 61 	movsd  0x611aa0(%rax),%xmm1
  400dd7:	00 
  400dd8:	f2 0f 10 90 a0 fc 65 	movsd  0x65fca0(%rax),%xmm2
  400ddf:	00 
  400de0:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400de4:	f2 0f 10 80 a0 20 60 	movsd  0x6020a0(%rax),%xmm0
  400deb:	00 
  400dec:	f2 44 0f 10 1d 9b 06 	movsd  0x69b(%rip),%xmm11        # 401490 <_IO_stdin_used+0x90>
  400df3:	00 00 
  400df5:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400df9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400dfd:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400e01:	f2 0f 58 88 a0 0e 63 	addsd  0x630ea0(%rax),%xmm1
  400e08:	00 
  400e09:	66 44 0f 28 d1       	movapd %xmm1,%xmm10
  400e0e:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400e15:	00 
  400e16:	f2 44 0f 59 d7       	mulsd  %xmm7,%xmm10
  400e1b:	f2 0f 58 80 a0 14 62 	addsd  0x6214a0(%rax),%xmm0
  400e22:	00 
  400e23:	f2 44 0f 58 90 a0 02 	addsd  0x6502a0(%rax),%xmm10
  400e2a:	65 00 
  400e2c:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400e30:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400e37:	00 
  400e38:	66 45 0f 2e da       	ucomisd %xmm10,%xmm11
  400e3d:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400e41:	f2 44 0f 11 90 a0 02 	movsd  %xmm10,0x6502a0(%rax)
  400e48:	65 00 
  400e4a:	f2 0f 58 90 a0 08 64 	addsd  0x6408a0(%rax),%xmm2
  400e51:	00 
  400e52:	f2 0f 11 90 a0 08 64 	movsd  %xmm2,0x6408a0(%rax)
  400e59:	00 
  400e5a:	77 07                	ja     400e63 <_Z14run_simulationv+0x203>
  400e5c:	66 44 0f 2e d4       	ucomisd %xmm4,%xmm10
  400e61:	76 0c                	jbe    400e6f <_Z14run_simulationv+0x20f>
  400e63:	66 0f 57 cb          	xorpd  %xmm3,%xmm1
  400e67:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400e6e:	00 
  400e6f:	f2 0f 10 0d 19 06 00 	movsd  0x619(%rip),%xmm1        # 401490 <_IO_stdin_used+0x90>
  400e76:	00 
  400e77:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  400e7b:	77 07                	ja     400e84 <_Z14run_simulationv+0x224>
  400e7d:	66 41 0f 2e d1       	ucomisd %xmm9,%xmm2
  400e82:	76 0d                	jbe    400e91 <_Z14run_simulationv+0x231>
  400e84:	66 41 0f 57 c0       	xorpd  %xmm8,%xmm0
  400e89:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400e90:	00 
  400e91:	48 83 c0 08          	add    $0x8,%rax
  400e95:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400e9b:	0f 85 2f ff ff ff    	jne    400dd0 <_Z14run_simulationv+0x170>
  400ea1:	41 83 ed 01          	sub    $0x1,%r13d
  400ea5:	0f 85 dd fd ff ff    	jne    400c88 <_Z14run_simulationv+0x28>
  400eab:	48 83 c4 38          	add    $0x38,%rsp
  400eaf:	5b                   	pop    %rbx
  400eb0:	5d                   	pop    %rbp
  400eb1:	41 5c                	pop    %r12
  400eb3:	41 5d                	pop    %r13
  400eb5:	c3                   	retq   
  400eb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ebd:	00 00 00 
  400ec0:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400ec6:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400ecc:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400ed2:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400ed9:	f2 0f 11 5c 24 08    	movsd  %xmm3,0x8(%rsp)
  400edf:	e8 fc f8 ff ff       	callq  4007e0 <sqrt@plt>
  400ee4:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400eea:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400eee:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400ef4:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400efa:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400f01:	f2 0f 10 5c 24 08    	movsd  0x8(%rsp),%xmm3
  400f07:	e9 20 fe ff ff       	jmpq   400d2c <_Z14run_simulationv+0xcc>
  400f0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400f10 <_Z11r_float2intf>:
  400f10:	66 0f 7e c0          	movd   %xmm0,%eax
  400f14:	c3                   	retq   
  400f15:	90                   	nop
  400f16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f1d:	00 00 00 

0000000000400f20 <_Z15Body_initializemddddd>:
  400f20:	f2 0f 11 04 fd a0 fc 	movsd  %xmm0,0x65fca0(,%rdi,8)
  400f27:	65 00 
  400f29:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400f2d:	f2 0f 11 0c fd a0 02 	movsd  %xmm1,0x6502a0(,%rdi,8)
  400f34:	65 00 
  400f36:	f2 0f 11 14 fd a0 08 	movsd  %xmm2,0x6408a0(,%rdi,8)
  400f3d:	64 00 
  400f3f:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  400f46:	63 00 
  400f48:	f2 0f 11 24 fd a0 14 	movsd  %xmm4,0x6214a0(,%rdi,8)
  400f4f:	62 00 
  400f51:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  400f58:	61 00 
  400f5a:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  400f61:	60 00 
  400f63:	c3                   	retq   
  400f64:	66 90                	xchg   %ax,%ax
  400f66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f6d:	00 00 00 

0000000000400f70 <_Z14Body_add_forcemm>:
  400f70:	48 39 f7             	cmp    %rsi,%rdi
  400f73:	0f 84 be 00 00 00    	je     401037 <_Z14Body_add_forcemm+0xc7>
  400f79:	55                   	push   %rbp
  400f7a:	53                   	push   %rbx
  400f7b:	48 89 f5             	mov    %rsi,%rbp
  400f7e:	48 89 fb             	mov    %rdi,%rbx
  400f81:	48 83 ec 18          	sub    $0x18,%rsp
  400f85:	f2 0f 10 0c f5 a0 08 	movsd  0x6408a0(,%rsi,8),%xmm1
  400f8c:	64 00 
  400f8e:	f2 0f 10 1c f5 a0 02 	movsd  0x6502a0(,%rsi,8),%xmm3
  400f95:	65 00 
  400f97:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400f9b:	f2 0f 5c 1c fd a0 02 	subsd  0x6502a0(,%rdi,8),%xmm3
  400fa2:	65 00 
  400fa4:	f2 0f 5c 2c fd a0 08 	subsd  0x6408a0(,%rdi,8),%xmm5
  400fab:	64 00 
  400fad:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400fb1:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400fb5:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400fb9:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400fbd:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400fc1:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400fc5:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400fc9:	7a 6e                	jp     401039 <_Z14Body_add_forcemm+0xc9>
  400fcb:	f2 0f 10 15 ad 04 00 	movsd  0x4ad(%rip),%xmm2        # 401480 <_IO_stdin_used+0x80>
  400fd2:	00 
  400fd3:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400fd7:	f2 0f 59 14 dd a0 fc 	mulsd  0x65fca0(,%rbx,8),%xmm2
  400fde:	65 00 
  400fe0:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400fe4:	f2 0f 58 05 9c 04 00 	addsd  0x49c(%rip),%xmm0        # 401488 <_IO_stdin_used+0x88>
  400feb:	00 
  400fec:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400ff3:	65 00 
  400ff5:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400ff9:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400ffd:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  401001:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401005:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  401009:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  40100d:	f2 0f 58 1c dd a0 1a 	addsd  0x611aa0(,%rbx,8),%xmm3
  401014:	61 00 
  401016:	f2 0f 11 1c dd a0 1a 	movsd  %xmm3,0x611aa0(,%rbx,8)
  40101d:	61 00 
  40101f:	f2 0f 58 0c dd a0 20 	addsd  0x6020a0(,%rbx,8),%xmm1
  401026:	60 00 
  401028:	f2 0f 11 0c dd a0 20 	movsd  %xmm1,0x6020a0(,%rbx,8)
  40102f:	60 00 
  401031:	48 83 c4 18          	add    $0x18,%rsp
  401035:	5b                   	pop    %rbx
  401036:	5d                   	pop    %rbp
  401037:	f3 c3                	repz retq 
  401039:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  40103f:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  401044:	e8 97 f7 ff ff       	callq  4007e0 <sqrt@plt>
  401049:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  40104f:	66 0f 28 e0          	movapd %xmm0,%xmm4
  401053:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  401058:	e9 6e ff ff ff       	jmpq   400fcb <_Z14Body_add_forcemm+0x5b>
  40105d:	0f 1f 00             	nopl   (%rax)

0000000000401060 <_Z21Body_add_force_to_allm>:
  401060:	41 54                	push   %r12
  401062:	55                   	push   %rbp
  401063:	49 89 fc             	mov    %rdi,%r12
  401066:	53                   	push   %rbx
  401067:	31 ed                	xor    %ebp,%ebp
  401069:	31 db                	xor    %ebx,%ebx
  40106b:	48 83 ec 20          	sub    $0x20,%rsp
  40106f:	f2 0f 10 3d 09 04 00 	movsd  0x409(%rip),%xmm7        # 401480 <_IO_stdin_used+0x80>
  401076:	00 
  401077:	f2 0f 10 35 09 04 00 	movsd  0x409(%rip),%xmm6        # 401488 <_IO_stdin_used+0x88>
  40107e:	00 
  40107f:	90                   	nop
  401080:	49 39 ec             	cmp    %rbp,%r12
  401083:	0f 84 a0 00 00 00    	je     401129 <_Z21Body_add_force_to_allm+0xc9>
  401089:	f2 42 0f 10 0c e5 a0 	movsd  0x6408a0(,%r12,8),%xmm1
  401090:	08 64 00 
  401093:	f2 42 0f 10 1c e5 a0 	movsd  0x6502a0(,%r12,8),%xmm3
  40109a:	02 65 00 
  40109d:	66 0f 28 e9          	movapd %xmm1,%xmm5
  4010a1:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  4010a8:	00 
  4010a9:	f2 0f 5c ab a0 08 64 	subsd  0x6408a0(%rbx),%xmm5
  4010b0:	00 
  4010b1:	66 0f 28 cb          	movapd %xmm3,%xmm1
  4010b5:	66 0f 28 c5          	movapd %xmm5,%xmm0
  4010b9:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  4010bd:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  4010c1:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  4010c5:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  4010c9:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  4010cd:	7a 78                	jp     401147 <_Z21Body_add_force_to_allm+0xe7>
  4010cf:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  4010d6:	00 
  4010d7:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4010db:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  4010df:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  4010e3:	f2 42 0f 59 14 e5 a0 	mulsd  0x65fca0(,%r12,8),%xmm2
  4010ea:	fc 65 00 
  4010ed:	f2 0f 58 c6          	addsd  %xmm6,%xmm0
  4010f1:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  4010f5:	66 0f 28 ca          	movapd %xmm2,%xmm1
  4010f9:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  4010fd:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  401101:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  401105:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  401109:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  401110:	00 
  401111:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  401118:	00 
  401119:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  401120:	00 
  401121:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  401128:	00 
  401129:	48 83 c5 01          	add    $0x1,%rbp
  40112d:	48 83 c3 08          	add    $0x8,%rbx
  401131:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  401138:	0f 85 42 ff ff ff    	jne    401080 <_Z21Body_add_force_to_allm+0x20>
  40113e:	48 83 c4 20          	add    $0x20,%rsp
  401142:	5b                   	pop    %rbx
  401143:	5d                   	pop    %rbp
  401144:	41 5c                	pop    %r12
  401146:	c3                   	retq   
  401147:	f2 0f 11 74 24 18    	movsd  %xmm6,0x18(%rsp)
  40114d:	f2 0f 11 7c 24 10    	movsd  %xmm7,0x10(%rsp)
  401153:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401159:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  40115e:	e8 7d f6 ff ff       	callq  4007e0 <sqrt@plt>
  401163:	f2 0f 10 74 24 18    	movsd  0x18(%rsp),%xmm6
  401169:	66 0f 28 e0          	movapd %xmm0,%xmm4
  40116d:	f2 0f 10 7c 24 10    	movsd  0x10(%rsp),%xmm7
  401173:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  401179:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  40117e:	e9 4c ff ff ff       	jmpq   4010cf <_Z21Body_add_force_to_allm+0x6f>
  401183:	0f 1f 00             	nopl   (%rax)
  401186:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40118d:	00 00 00 

0000000000401190 <_Z16Body_reset_forcem>:
  401190:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401194:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  40119b:	61 00 
  40119d:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  4011a4:	60 00 
  4011a6:	c3                   	retq   
  4011a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4011ae:	00 00 

00000000004011b0 <_Z11Body_updatemd>:
  4011b0:	f2 0f 10 14 fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm2
  4011b7:	61 00 
  4011b9:	f2 0f 10 1c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm3
  4011c0:	65 00 
  4011c2:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  4011c6:	f2 0f 10 0c fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm1
  4011cd:	60 00 
  4011cf:	f2 0f 10 25 b9 02 00 	movsd  0x2b9(%rip),%xmm4        # 401490 <_IO_stdin_used+0x90>
  4011d6:	00 
  4011d7:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011db:	f2 0f 5e d3          	divsd  %xmm3,%xmm2
  4011df:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  4011e3:	66 0f 28 d8          	movapd %xmm0,%xmm3
  4011e7:	f2 0f 58 14 fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm2
  4011ee:	63 00 
  4011f0:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  4011f7:	63 00 
  4011f9:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  4011fd:	f2 0f 58 0c fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm1
  401204:	62 00 
  401206:	f2 0f 58 1c fd a0 02 	addsd  0x6502a0(,%rdi,8),%xmm3
  40120d:	65 00 
  40120f:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  401213:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  40121a:	62 00 
  40121c:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  401220:	f2 0f 11 1c fd a0 02 	movsd  %xmm3,0x6502a0(,%rdi,8)
  401227:	65 00 
  401229:	f2 0f 58 04 fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm0
  401230:	64 00 
  401232:	f2 0f 11 04 fd a0 08 	movsd  %xmm0,0x6408a0(,%rdi,8)
  401239:	64 00 
  40123b:	77 0a                	ja     401247 <_Z11Body_updatemd+0x97>
  40123d:	66 0f 2e 1d 33 02 00 	ucomisd 0x233(%rip),%xmm3        # 401478 <_IO_stdin_used+0x78>
  401244:	00 
  401245:	76 15                	jbe    40125c <_Z11Body_updatemd+0xac>
  401247:	f2 0f 10 1d 51 02 00 	movsd  0x251(%rip),%xmm3        # 4014a0 <_IO_stdin_used+0xa0>
  40124e:	00 
  40124f:	66 0f 57 d3          	xorpd  %xmm3,%xmm2
  401253:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  40125a:	63 00 
  40125c:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  401260:	77 0a                	ja     40126c <_Z11Body_updatemd+0xbc>
  401262:	66 0f 2e 05 0e 02 00 	ucomisd 0x20e(%rip),%xmm0        # 401478 <_IO_stdin_used+0x78>
  401269:	00 
  40126a:	76 15                	jbe    401281 <_Z11Body_updatemd+0xd1>
  40126c:	f2 0f 10 05 2c 02 00 	movsd  0x22c(%rip),%xmm0        # 4014a0 <_IO_stdin_used+0xa0>
  401273:	00 
  401274:	66 0f 57 c8          	xorpd  %xmm0,%xmm1
  401278:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  40127f:	62 00 
  401281:	f3 c3                	repz retq 
  401283:	0f 1f 00             	nopl   (%rax)
  401286:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40128d:	00 00 00 

0000000000401290 <_Z26Body_codegen_simple_updatemd>:
  401290:	f2 0f 10 3c fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm7
  401297:	61 00 
  401299:	b8 64 00 00 00       	mov    $0x64,%eax
  40129e:	f2 0f 10 34 fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm6
  4012a5:	60 00 
  4012a7:	f2 0f 59 f8          	mulsd  %xmm0,%xmm7
  4012ab:	f2 0f 59 f0          	mulsd  %xmm0,%xmm6
  4012af:	f2 0f 10 1c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm3
  4012b6:	65 00 
  4012b8:	f2 0f 10 14 fd a0 0e 	movsd  0x630ea0(,%rdi,8),%xmm2
  4012bf:	63 00 
  4012c1:	f2 0f 5e fb          	divsd  %xmm3,%xmm7
  4012c5:	f2 0f 5e f3          	divsd  %xmm3,%xmm6
  4012c9:	f2 0f 10 0c fd a0 14 	movsd  0x6214a0(,%rdi,8),%xmm1
  4012d0:	62 00 
  4012d2:	f2 0f 10 24 fd a0 02 	movsd  0x6502a0(,%rdi,8),%xmm4
  4012d9:	65 00 
  4012db:	f2 0f 10 1c fd a0 08 	movsd  0x6408a0(,%rdi,8),%xmm3
  4012e2:	64 00 
  4012e4:	0f 1f 40 00          	nopl   0x0(%rax)
  4012e8:	f2 0f 58 d7          	addsd  %xmm7,%xmm2
  4012ec:	66 0f 28 e8          	movapd %xmm0,%xmm5
  4012f0:	f2 0f 58 ce          	addsd  %xmm6,%xmm1
  4012f4:	83 e8 01             	sub    $0x1,%eax
  4012f7:	f2 0f 59 ea          	mulsd  %xmm2,%xmm5
  4012fb:	f2 0f 58 e5          	addsd  %xmm5,%xmm4
  4012ff:	66 0f 28 e8          	movapd %xmm0,%xmm5
  401303:	f2 0f 59 e9          	mulsd  %xmm1,%xmm5
  401307:	f2 0f 58 dd          	addsd  %xmm5,%xmm3
  40130b:	75 db                	jne    4012e8 <_Z26Body_codegen_simple_updatemd+0x58>
  40130d:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  401314:	63 00 
  401316:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  40131d:	62 00 
  40131f:	f2 0f 11 24 fd a0 02 	movsd  %xmm4,0x6502a0(,%rdi,8)
  401326:	65 00 
  401328:	f2 0f 11 1c fd a0 08 	movsd  %xmm3,0x6408a0(,%rdi,8)
  40132f:	64 00 
  401331:	c3                   	retq   
  401332:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  401339:	00 00 00 
  40133c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401340 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401340:	55                   	push   %rbp
  401341:	53                   	push   %rbx
  401342:	48 89 fd             	mov    %rdi,%rbp
  401345:	48 83 ec 08          	sub    $0x8,%rsp
  401349:	e8 72 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40134e:	48 89 c3             	mov    %rax,%rbx
  401351:	ff d5                	callq  *%rbp
  401353:	e8 68 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401358:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40135f:	de 1b 43 
  401362:	48 29 d8             	sub    %rbx,%rax
  401365:	48 83 c4 08          	add    $0x8,%rsp
  401369:	48 89 c1             	mov    %rax,%rcx
  40136c:	48 f7 ea             	imul   %rdx
  40136f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401373:	5b                   	pop    %rbx
  401374:	5d                   	pop    %rbp
  401375:	48 c1 fa 12          	sar    $0x12,%rdx
  401379:	48 89 d0             	mov    %rdx,%rax
  40137c:	48 29 c8             	sub    %rcx,%rax
  40137f:	c3                   	retq   

0000000000401380 <__libc_csu_init>:
  401380:	41 57                	push   %r15
  401382:	41 56                	push   %r14
  401384:	41 89 ff             	mov    %edi,%r15d
  401387:	41 55                	push   %r13
  401389:	41 54                	push   %r12
  40138b:	4c 8d 25 56 0a 20 00 	lea    0x200a56(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  401392:	55                   	push   %rbp
  401393:	48 8d 2d 5e 0a 20 00 	lea    0x200a5e(%rip),%rbp        # 601df8 <__init_array_end>
  40139a:	53                   	push   %rbx
  40139b:	49 89 f6             	mov    %rsi,%r14
  40139e:	49 89 d5             	mov    %rdx,%r13
  4013a1:	4c 29 e5             	sub    %r12,%rbp
  4013a4:	48 83 ec 08          	sub    $0x8,%rsp
  4013a8:	48 c1 fd 03          	sar    $0x3,%rbp
  4013ac:	e8 67 f3 ff ff       	callq  400718 <_init>
  4013b1:	48 85 ed             	test   %rbp,%rbp
  4013b4:	74 20                	je     4013d6 <__libc_csu_init+0x56>
  4013b6:	31 db                	xor    %ebx,%ebx
  4013b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4013bf:	00 
  4013c0:	4c 89 ea             	mov    %r13,%rdx
  4013c3:	4c 89 f6             	mov    %r14,%rsi
  4013c6:	44 89 ff             	mov    %r15d,%edi
  4013c9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4013cd:	48 83 c3 01          	add    $0x1,%rbx
  4013d1:	48 39 eb             	cmp    %rbp,%rbx
  4013d4:	75 ea                	jne    4013c0 <__libc_csu_init+0x40>
  4013d6:	48 83 c4 08          	add    $0x8,%rsp
  4013da:	5b                   	pop    %rbx
  4013db:	5d                   	pop    %rbp
  4013dc:	41 5c                	pop    %r12
  4013de:	41 5d                	pop    %r13
  4013e0:	41 5e                	pop    %r14
  4013e2:	41 5f                	pop    %r15
  4013e4:	c3                   	retq   
  4013e5:	90                   	nop
  4013e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4013ed:	00 00 00 

00000000004013f0 <__libc_csu_fini>:
  4013f0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004013f4 <_fini>:
  4013f4:	48 83 ec 08          	sub    $0x8,%rsp
  4013f8:	48 83 c4 08          	add    $0x8,%rsp
  4013fc:	c3                   	retq   
