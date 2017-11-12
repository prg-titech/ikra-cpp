
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
  400817:	e8 c4 0a 00 00       	callq  4012e0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40081c:	bf 30 0c 40 00       	mov    $0x400c30,%edi
  400821:	48 89 c5             	mov    %rax,%rbp
  400824:	e8 b7 0a 00 00       	callq  4012e0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400829:	bf 90 0b 40 00       	mov    $0x400b90,%edi
  40082e:	49 89 c4             	mov    %rax,%r12
  400831:	e8 aa 0a 00 00       	callq  4012e0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
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
  4008a9:	be c0 13 40 00       	mov    $0x4013c0,%esi
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
  4008ca:	be a4 13 40 00       	mov    $0x4013a4,%esi
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
  40093f:	49 c7 c0 90 13 40 00 	mov    $0x401390,%r8
  400946:	48 c7 c1 20 13 40 00 	mov    $0x401320,%rcx
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
  400a49:	f2 0f 10 0d af 09 00 	movsd  0x9af(%rip),%xmm1        # 401400 <_IO_stdin_used+0x60>
  400a50:	00 
  400a51:	f2 0f 10 15 af 09 00 	movsd  0x9af(%rip),%xmm2        # 401408 <_IO_stdin_used+0x68>
  400a58:	00 
  400a59:	f2 0f 10 1d af 09 00 	movsd  0x9af(%rip),%xmm3        # 401410 <_IO_stdin_used+0x70>
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
  400afb:	f2 0f 5c 05 15 09 00 	subsd  0x915(%rip),%xmm0        # 401418 <_IO_stdin_used+0x78>
  400b02:	00 
  400b03:	f2 41 0f 11 85 98 02 	movsd  %xmm0,0x650298(%r13)
  400b0a:	65 00 
  400b0c:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400b10:	f2 0f 2a c5          	cvtsi2sd %ebp,%xmm0
  400b14:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  400b18:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  400b1c:	f2 0f 5c 05 f4 08 00 	subsd  0x8f4(%rip),%xmm0        # 401418 <_IO_stdin_used+0x78>
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
  400b90:	66 0f 28 15 a8 08 00 	movapd 0x8a8(%rip),%xmm2        # 401440 <_IO_stdin_used+0xa0>
  400b97:	00 
  400b98:	ba 50 c3 00 00       	mov    $0xc350,%edx
  400b9d:	0f 1f 00             	nopl   (%rax)
  400ba0:	31 c0                	xor    %eax,%eax
  400ba2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400ba8:	66 0f 28 88 a0 1a 61 	movapd 0x611aa0(%rax),%xmm1
  400baf:	00 
  400bb0:	48 83 c0 10          	add    $0x10,%rax
  400bb4:	66 0f 28 80 90 20 60 	movapd 0x602090(%rax),%xmm0
  400bbb:	00 
  400bbc:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400bc0:	66 0f 28 98 90 fc 65 	movapd 0x65fc90(%rax),%xmm3
  400bc7:	00 
  400bc8:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400bcc:	66 0f 5e cb          	divpd  %xmm3,%xmm1
  400bd0:	66 0f 5e c3          	divpd  %xmm3,%xmm0
  400bd4:	66 0f 58 88 90 0e 63 	addpd  0x630e90(%rax),%xmm1
  400bdb:	00 
  400bdc:	0f 29 88 90 0e 63 00 	movaps %xmm1,0x630e90(%rax)
  400be3:	66 0f 59 ca          	mulpd  %xmm2,%xmm1
  400be7:	66 0f 58 88 90 02 65 	addpd  0x650290(%rax),%xmm1
  400bee:	00 
  400bef:	66 0f 58 80 90 14 62 	addpd  0x621490(%rax),%xmm0
  400bf6:	00 
  400bf7:	0f 29 88 90 02 65 00 	movaps %xmm1,0x650290(%rax)
  400bfe:	0f 29 80 90 14 62 00 	movaps %xmm0,0x621490(%rax)
  400c05:	66 0f 59 c2          	mulpd  %xmm2,%xmm0
  400c09:	66 0f 58 80 90 08 64 	addpd  0x640890(%rax),%xmm0
  400c10:	00 
  400c11:	0f 29 80 90 08 64 00 	movaps %xmm0,0x640890(%rax)
  400c18:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400c1e:	75 88                	jne    400ba8 <_Z10run_simplev+0x18>
  400c20:	83 ea 01             	sub    $0x1,%edx
  400c23:	0f 85 77 ff ff ff    	jne    400ba0 <_Z10run_simplev+0x10>
  400c29:	f3 c3                	repz retq 
  400c2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c30 <_Z14run_simulationv>:
  400c30:	41 55                	push   %r13
  400c32:	41 54                	push   %r12
  400c34:	41 bd 05 00 00 00    	mov    $0x5,%r13d
  400c3a:	55                   	push   %rbp
  400c3b:	53                   	push   %rbx
  400c3c:	48 83 ec 38          	sub    $0x38,%rsp
  400c40:	f2 0f 10 3d c0 07 00 	movsd  0x7c0(%rip),%xmm7        # 401408 <_IO_stdin_used+0x68>
  400c47:	00 
  400c48:	f2 0f 10 35 d0 07 00 	movsd  0x7d0(%rip),%xmm6        # 401420 <_IO_stdin_used+0x80>
  400c4f:	00 
  400c50:	f2 0f 10 2d d0 07 00 	movsd  0x7d0(%rip),%xmm5        # 401428 <_IO_stdin_used+0x88>
  400c57:	00 
  400c58:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c5d:	31 f6                	xor    %esi,%esi
  400c5f:	bf a0 1a 61 00       	mov    $0x611aa0,%edi
  400c64:	f2 0f 11 6c 24 18    	movsd  %xmm5,0x18(%rsp)
  400c6a:	31 ed                	xor    %ebp,%ebp
  400c6c:	f2 0f 11 74 24 10    	movsd  %xmm6,0x10(%rsp)
  400c72:	f2 0f 11 7c 24 08    	movsd  %xmm7,0x8(%rsp)
  400c78:	e8 d3 fa ff ff       	callq  400750 <memset@plt>
  400c7d:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c82:	31 f6                	xor    %esi,%esi
  400c84:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400c89:	e8 c2 fa ff ff       	callq  400750 <memset@plt>
  400c8e:	f2 0f 10 6c 24 18    	movsd  0x18(%rsp),%xmm5
  400c94:	f2 0f 10 74 24 10    	movsd  0x10(%rsp),%xmm6
  400c9a:	f2 0f 10 7c 24 08    	movsd  0x8(%rsp),%xmm7
  400ca0:	31 db                	xor    %ebx,%ebx
  400ca2:	45 31 e4             	xor    %r12d,%r12d
  400ca5:	0f 1f 00             	nopl   (%rax)
  400ca8:	4c 39 e5             	cmp    %r12,%rbp
  400cab:	0f 84 a5 00 00 00    	je     400d56 <_Z14run_simulationv+0x126>
  400cb1:	f2 0f 10 0c ed a0 08 	movsd  0x6408a0(,%rbp,8),%xmm1
  400cb8:	64 00 
  400cba:	f2 0f 10 1c ed a0 02 	movsd  0x6502a0(,%rbp,8),%xmm3
  400cc1:	65 00 
  400cc3:	f2 0f 5c 8b a0 08 64 	subsd  0x6408a0(%rbx),%xmm1
  400cca:	00 
  400ccb:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  400cd2:	00 
  400cd3:	66 44 0f 28 c1       	movapd %xmm1,%xmm8
  400cd8:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400cdc:	66 41 0f 28 c0       	movapd %xmm8,%xmm0
  400ce1:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400ce5:	f2 41 0f 59 c0       	mulsd  %xmm8,%xmm0
  400cea:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400cee:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400cf2:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400cf6:	0f 8a 94 01 00 00    	jp     400e90 <_Z14run_simulationv+0x260>
  400cfc:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  400d03:	00 
  400d04:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400d08:	f2 0f 59 d6          	mulsd  %xmm6,%xmm2
  400d0c:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400d10:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400d17:	65 00 
  400d19:	f2 0f 58 c5          	addsd  %xmm5,%xmm0
  400d1d:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400d21:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400d25:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400d29:	f2 41 0f 59 c8       	mulsd  %xmm8,%xmm1
  400d2e:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400d32:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400d36:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  400d3d:	00 
  400d3e:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  400d45:	00 
  400d46:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  400d4d:	00 
  400d4e:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  400d55:	00 
  400d56:	49 83 c4 01          	add    $0x1,%r12
  400d5a:	48 83 c3 08          	add    $0x8,%rbx
  400d5e:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400d65:	0f 85 3d ff ff ff    	jne    400ca8 <_Z14run_simulationv+0x78>
  400d6b:	48 83 c5 01          	add    $0x1,%rbp
  400d6f:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  400d76:	0f 85 24 ff ff ff    	jne    400ca0 <_Z14run_simulationv+0x70>
  400d7c:	f2 0f 10 1d cc 06 00 	movsd  0x6cc(%rip),%xmm3        # 401450 <_IO_stdin_used+0xb0>
  400d83:	00 
  400d84:	31 c0                	xor    %eax,%eax
  400d86:	f2 0f 10 25 8a 06 00 	movsd  0x68a(%rip),%xmm4        # 401418 <_IO_stdin_used+0x78>
  400d8d:	00 
  400d8e:	66 44 0f 28 c3       	movapd %xmm3,%xmm8
  400d93:	66 44 0f 28 cc       	movapd %xmm4,%xmm9
  400d98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d9f:	00 
  400da0:	f2 0f 10 88 a0 1a 61 	movsd  0x611aa0(%rax),%xmm1
  400da7:	00 
  400da8:	f2 0f 10 90 a0 fc 65 	movsd  0x65fca0(%rax),%xmm2
  400daf:	00 
  400db0:	f2 0f 59 cf          	mulsd  %xmm7,%xmm1
  400db4:	f2 0f 10 80 a0 20 60 	movsd  0x6020a0(%rax),%xmm0
  400dbb:	00 
  400dbc:	f2 44 0f 10 1d 6b 06 	movsd  0x66b(%rip),%xmm11        # 401430 <_IO_stdin_used+0x90>
  400dc3:	00 00 
  400dc5:	f2 0f 59 c7          	mulsd  %xmm7,%xmm0
  400dc9:	f2 0f 5e ca          	divsd  %xmm2,%xmm1
  400dcd:	f2 0f 5e c2          	divsd  %xmm2,%xmm0
  400dd1:	f2 0f 58 88 a0 0e 63 	addsd  0x630ea0(%rax),%xmm1
  400dd8:	00 
  400dd9:	66 44 0f 28 d1       	movapd %xmm1,%xmm10
  400dde:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400de5:	00 
  400de6:	f2 44 0f 59 d7       	mulsd  %xmm7,%xmm10
  400deb:	f2 0f 58 80 a0 14 62 	addsd  0x6214a0(%rax),%xmm0
  400df2:	00 
  400df3:	f2 44 0f 58 90 a0 02 	addsd  0x6502a0(%rax),%xmm10
  400dfa:	65 00 
  400dfc:	66 0f 28 d0          	movapd %xmm0,%xmm2
  400e00:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400e07:	00 
  400e08:	66 45 0f 2e da       	ucomisd %xmm10,%xmm11
  400e0d:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  400e11:	f2 44 0f 11 90 a0 02 	movsd  %xmm10,0x6502a0(%rax)
  400e18:	65 00 
  400e1a:	f2 0f 58 90 a0 08 64 	addsd  0x6408a0(%rax),%xmm2
  400e21:	00 
  400e22:	f2 0f 11 90 a0 08 64 	movsd  %xmm2,0x6408a0(%rax)
  400e29:	00 
  400e2a:	77 07                	ja     400e33 <_Z14run_simulationv+0x203>
  400e2c:	66 44 0f 2e d4       	ucomisd %xmm4,%xmm10
  400e31:	76 0c                	jbe    400e3f <_Z14run_simulationv+0x20f>
  400e33:	66 0f 57 cb          	xorpd  %xmm3,%xmm1
  400e37:	f2 0f 11 88 a0 0e 63 	movsd  %xmm1,0x630ea0(%rax)
  400e3e:	00 
  400e3f:	f2 0f 10 0d e9 05 00 	movsd  0x5e9(%rip),%xmm1        # 401430 <_IO_stdin_used+0x90>
  400e46:	00 
  400e47:	66 0f 2e ca          	ucomisd %xmm2,%xmm1
  400e4b:	77 07                	ja     400e54 <_Z14run_simulationv+0x224>
  400e4d:	66 41 0f 2e d1       	ucomisd %xmm9,%xmm2
  400e52:	76 0d                	jbe    400e61 <_Z14run_simulationv+0x231>
  400e54:	66 41 0f 57 c0       	xorpd  %xmm8,%xmm0
  400e59:	f2 0f 11 80 a0 14 62 	movsd  %xmm0,0x6214a0(%rax)
  400e60:	00 
  400e61:	48 83 c0 08          	add    $0x8,%rax
  400e65:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400e6b:	0f 85 2f ff ff ff    	jne    400da0 <_Z14run_simulationv+0x170>
  400e71:	41 83 ed 01          	sub    $0x1,%r13d
  400e75:	0f 85 dd fd ff ff    	jne    400c58 <_Z14run_simulationv+0x28>
  400e7b:	48 83 c4 38          	add    $0x38,%rsp
  400e7f:	5b                   	pop    %rbx
  400e80:	5d                   	pop    %rbp
  400e81:	41 5c                	pop    %r12
  400e83:	41 5d                	pop    %r13
  400e85:	c3                   	retq   
  400e86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e8d:	00 00 00 
  400e90:	f2 0f 11 6c 24 28    	movsd  %xmm5,0x28(%rsp)
  400e96:	f2 0f 11 74 24 20    	movsd  %xmm6,0x20(%rsp)
  400e9c:	f2 0f 11 7c 24 18    	movsd  %xmm7,0x18(%rsp)
  400ea2:	f2 44 0f 11 44 24 10 	movsd  %xmm8,0x10(%rsp)
  400ea9:	f2 0f 11 5c 24 08    	movsd  %xmm3,0x8(%rsp)
  400eaf:	e8 2c f9 ff ff       	callq  4007e0 <sqrt@plt>
  400eb4:	f2 0f 10 6c 24 28    	movsd  0x28(%rsp),%xmm5
  400eba:	66 0f 28 e0          	movapd %xmm0,%xmm4
  400ebe:	f2 0f 10 74 24 20    	movsd  0x20(%rsp),%xmm6
  400ec4:	f2 0f 10 7c 24 18    	movsd  0x18(%rsp),%xmm7
  400eca:	f2 44 0f 10 44 24 10 	movsd  0x10(%rsp),%xmm8
  400ed1:	f2 0f 10 5c 24 08    	movsd  0x8(%rsp),%xmm3
  400ed7:	e9 20 fe ff ff       	jmpq   400cfc <_Z14run_simulationv+0xcc>
  400edc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ee0 <_Z11r_float2intf>:
  400ee0:	66 0f 7e c0          	movd   %xmm0,%eax
  400ee4:	c3                   	retq   
  400ee5:	90                   	nop
  400ee6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400eed:	00 00 00 

0000000000400ef0 <_Z15Body_initializemddddd>:
  400ef0:	f2 0f 11 04 fd a0 fc 	movsd  %xmm0,0x65fca0(,%rdi,8)
  400ef7:	65 00 
  400ef9:	66 0f ef c0          	pxor   %xmm0,%xmm0
  400efd:	f2 0f 11 0c fd a0 02 	movsd  %xmm1,0x6502a0(,%rdi,8)
  400f04:	65 00 
  400f06:	f2 0f 11 14 fd a0 08 	movsd  %xmm2,0x6408a0(,%rdi,8)
  400f0d:	64 00 
  400f0f:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  400f16:	63 00 
  400f18:	f2 0f 11 24 fd a0 14 	movsd  %xmm4,0x6214a0(,%rdi,8)
  400f1f:	62 00 
  400f21:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  400f28:	61 00 
  400f2a:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  400f31:	60 00 
  400f33:	c3                   	retq   
  400f34:	66 90                	xchg   %ax,%ax
  400f36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f3d:	00 00 00 

0000000000400f40 <_Z14Body_add_forcemm>:
  400f40:	48 39 f7             	cmp    %rsi,%rdi
  400f43:	0f 84 be 00 00 00    	je     401007 <_Z14Body_add_forcemm+0xc7>
  400f49:	55                   	push   %rbp
  400f4a:	53                   	push   %rbx
  400f4b:	48 89 f5             	mov    %rsi,%rbp
  400f4e:	48 89 fb             	mov    %rdi,%rbx
  400f51:	48 83 ec 18          	sub    $0x18,%rsp
  400f55:	f2 0f 10 0c f5 a0 08 	movsd  0x6408a0(,%rsi,8),%xmm1
  400f5c:	64 00 
  400f5e:	f2 0f 10 1c f5 a0 02 	movsd  0x6502a0(,%rsi,8),%xmm3
  400f65:	65 00 
  400f67:	66 0f 28 e9          	movapd %xmm1,%xmm5
  400f6b:	f2 0f 5c 1c fd a0 02 	subsd  0x6502a0(,%rdi,8),%xmm3
  400f72:	65 00 
  400f74:	f2 0f 5c 2c fd a0 08 	subsd  0x6408a0(,%rdi,8),%xmm5
  400f7b:	64 00 
  400f7d:	66 0f 28 cb          	movapd %xmm3,%xmm1
  400f81:	66 0f 28 c5          	movapd %xmm5,%xmm0
  400f85:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  400f89:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  400f8d:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  400f91:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  400f95:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  400f99:	7a 6e                	jp     401009 <_Z14Body_add_forcemm+0xc9>
  400f9b:	f2 0f 10 15 7d 04 00 	movsd  0x47d(%rip),%xmm2        # 401420 <_IO_stdin_used+0x80>
  400fa2:	00 
  400fa3:	66 0f 28 c4          	movapd %xmm4,%xmm0
  400fa7:	f2 0f 59 14 dd a0 fc 	mulsd  0x65fca0(,%rbx,8),%xmm2
  400fae:	65 00 
  400fb0:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  400fb4:	f2 0f 58 05 6c 04 00 	addsd  0x46c(%rip),%xmm0        # 401428 <_IO_stdin_used+0x88>
  400fbb:	00 
  400fbc:	f2 0f 59 14 ed a0 fc 	mulsd  0x65fca0(,%rbp,8),%xmm2
  400fc3:	65 00 
  400fc5:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  400fc9:	66 0f 28 ca          	movapd %xmm2,%xmm1
  400fcd:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  400fd1:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  400fd5:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  400fd9:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  400fdd:	f2 0f 58 1c dd a0 1a 	addsd  0x611aa0(,%rbx,8),%xmm3
  400fe4:	61 00 
  400fe6:	f2 0f 11 1c dd a0 1a 	movsd  %xmm3,0x611aa0(,%rbx,8)
  400fed:	61 00 
  400fef:	f2 0f 58 0c dd a0 20 	addsd  0x6020a0(,%rbx,8),%xmm1
  400ff6:	60 00 
  400ff8:	f2 0f 11 0c dd a0 20 	movsd  %xmm1,0x6020a0(,%rbx,8)
  400fff:	60 00 
  401001:	48 83 c4 18          	add    $0x18,%rsp
  401005:	5b                   	pop    %rbx
  401006:	5d                   	pop    %rbp
  401007:	f3 c3                	repz retq 
  401009:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  40100f:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  401014:	e8 c7 f7 ff ff       	callq  4007e0 <sqrt@plt>
  401019:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  40101f:	66 0f 28 e0          	movapd %xmm0,%xmm4
  401023:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  401028:	e9 6e ff ff ff       	jmpq   400f9b <_Z14Body_add_forcemm+0x5b>
  40102d:	0f 1f 00             	nopl   (%rax)

0000000000401030 <_Z21Body_add_force_to_allm>:
  401030:	41 54                	push   %r12
  401032:	55                   	push   %rbp
  401033:	49 89 fc             	mov    %rdi,%r12
  401036:	53                   	push   %rbx
  401037:	31 ed                	xor    %ebp,%ebp
  401039:	31 db                	xor    %ebx,%ebx
  40103b:	48 83 ec 20          	sub    $0x20,%rsp
  40103f:	f2 0f 10 3d d9 03 00 	movsd  0x3d9(%rip),%xmm7        # 401420 <_IO_stdin_used+0x80>
  401046:	00 
  401047:	f2 0f 10 35 d9 03 00 	movsd  0x3d9(%rip),%xmm6        # 401428 <_IO_stdin_used+0x88>
  40104e:	00 
  40104f:	90                   	nop
  401050:	49 39 ec             	cmp    %rbp,%r12
  401053:	0f 84 a0 00 00 00    	je     4010f9 <_Z21Body_add_force_to_allm+0xc9>
  401059:	f2 42 0f 10 0c e5 a0 	movsd  0x6408a0(,%r12,8),%xmm1
  401060:	08 64 00 
  401063:	f2 42 0f 10 1c e5 a0 	movsd  0x6502a0(,%r12,8),%xmm3
  40106a:	02 65 00 
  40106d:	66 0f 28 e9          	movapd %xmm1,%xmm5
  401071:	f2 0f 5c 9b a0 02 65 	subsd  0x6502a0(%rbx),%xmm3
  401078:	00 
  401079:	f2 0f 5c ab a0 08 64 	subsd  0x6408a0(%rbx),%xmm5
  401080:	00 
  401081:	66 0f 28 cb          	movapd %xmm3,%xmm1
  401085:	66 0f 28 c5          	movapd %xmm5,%xmm0
  401089:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
  40108d:	f2 0f 59 c5          	mulsd  %xmm5,%xmm0
  401091:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
  401095:	f2 0f 51 e0          	sqrtsd %xmm0,%xmm4
  401099:	66 0f 2e e4          	ucomisd %xmm4,%xmm4
  40109d:	7a 78                	jp     401117 <_Z21Body_add_force_to_allm+0xe7>
  40109f:	f2 0f 10 93 a0 fc 65 	movsd  0x65fca0(%rbx),%xmm2
  4010a6:	00 
  4010a7:	66 0f 28 c4          	movapd %xmm4,%xmm0
  4010ab:	f2 0f 59 d7          	mulsd  %xmm7,%xmm2
  4010af:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
  4010b3:	f2 42 0f 59 14 e5 a0 	mulsd  0x65fca0(,%r12,8),%xmm2
  4010ba:	fc 65 00 
  4010bd:	f2 0f 58 c6          	addsd  %xmm6,%xmm0
  4010c1:	f2 0f 5e d0          	divsd  %xmm0,%xmm2
  4010c5:	66 0f 28 ca          	movapd %xmm2,%xmm1
  4010c9:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  4010cd:	f2 0f 59 cd          	mulsd  %xmm5,%xmm1
  4010d1:	f2 0f 5e dc          	divsd  %xmm4,%xmm3
  4010d5:	f2 0f 5e cc          	divsd  %xmm4,%xmm1
  4010d9:	f2 0f 58 9b a0 1a 61 	addsd  0x611aa0(%rbx),%xmm3
  4010e0:	00 
  4010e1:	f2 0f 11 9b a0 1a 61 	movsd  %xmm3,0x611aa0(%rbx)
  4010e8:	00 
  4010e9:	f2 0f 58 8b a0 20 60 	addsd  0x6020a0(%rbx),%xmm1
  4010f0:	00 
  4010f1:	f2 0f 11 8b a0 20 60 	movsd  %xmm1,0x6020a0(%rbx)
  4010f8:	00 
  4010f9:	48 83 c5 01          	add    $0x1,%rbp
  4010fd:	48 83 c3 08          	add    $0x8,%rbx
  401101:	48 81 fd 40 1f 00 00 	cmp    $0x1f40,%rbp
  401108:	0f 85 42 ff ff ff    	jne    401050 <_Z21Body_add_force_to_allm+0x20>
  40110e:	48 83 c4 20          	add    $0x20,%rsp
  401112:	5b                   	pop    %rbx
  401113:	5d                   	pop    %rbp
  401114:	41 5c                	pop    %r12
  401116:	c3                   	retq   
  401117:	f2 0f 11 74 24 18    	movsd  %xmm6,0x18(%rsp)
  40111d:	f2 0f 11 7c 24 10    	movsd  %xmm7,0x10(%rsp)
  401123:	f2 0f 11 6c 24 08    	movsd  %xmm5,0x8(%rsp)
  401129:	f2 0f 11 1c 24       	movsd  %xmm3,(%rsp)
  40112e:	e8 ad f6 ff ff       	callq  4007e0 <sqrt@plt>
  401133:	f2 0f 10 74 24 18    	movsd  0x18(%rsp),%xmm6
  401139:	66 0f 28 e0          	movapd %xmm0,%xmm4
  40113d:	f2 0f 10 7c 24 10    	movsd  0x10(%rsp),%xmm7
  401143:	f2 0f 10 6c 24 08    	movsd  0x8(%rsp),%xmm5
  401149:	f2 0f 10 1c 24       	movsd  (%rsp),%xmm3
  40114e:	e9 4c ff ff ff       	jmpq   40109f <_Z21Body_add_force_to_allm+0x6f>
  401153:	0f 1f 00             	nopl   (%rax)
  401156:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40115d:	00 00 00 

0000000000401160 <_Z16Body_reset_forcem>:
  401160:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401164:	f2 0f 11 04 fd a0 1a 	movsd  %xmm0,0x611aa0(,%rdi,8)
  40116b:	61 00 
  40116d:	f2 0f 11 04 fd a0 20 	movsd  %xmm0,0x6020a0(,%rdi,8)
  401174:	60 00 
  401176:	c3                   	retq   
  401177:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40117e:	00 00 

0000000000401180 <_Z11Body_updatemd>:
  401180:	f2 0f 10 14 fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm2
  401187:	61 00 
  401189:	f2 0f 10 1c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm3
  401190:	65 00 
  401192:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401196:	f2 0f 10 0c fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm1
  40119d:	60 00 
  40119f:	f2 0f 10 25 89 02 00 	movsd  0x289(%rip),%xmm4        # 401430 <_IO_stdin_used+0x90>
  4011a6:	00 
  4011a7:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4011ab:	f2 0f 5e d3          	divsd  %xmm3,%xmm2
  4011af:	f2 0f 5e cb          	divsd  %xmm3,%xmm1
  4011b3:	66 0f 28 d8          	movapd %xmm0,%xmm3
  4011b7:	f2 0f 58 14 fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm2
  4011be:	63 00 
  4011c0:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  4011c7:	63 00 
  4011c9:	f2 0f 59 da          	mulsd  %xmm2,%xmm3
  4011cd:	f2 0f 58 0c fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm1
  4011d4:	62 00 
  4011d6:	f2 0f 58 1c fd a0 02 	addsd  0x6502a0(,%rdi,8),%xmm3
  4011dd:	65 00 
  4011df:	f2 0f 59 c1          	mulsd  %xmm1,%xmm0
  4011e3:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  4011ea:	62 00 
  4011ec:	66 0f 2e e3          	ucomisd %xmm3,%xmm4
  4011f0:	f2 0f 11 1c fd a0 02 	movsd  %xmm3,0x6502a0(,%rdi,8)
  4011f7:	65 00 
  4011f9:	f2 0f 58 04 fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm0
  401200:	64 00 
  401202:	f2 0f 11 04 fd a0 08 	movsd  %xmm0,0x6408a0(,%rdi,8)
  401209:	64 00 
  40120b:	77 0a                	ja     401217 <_Z11Body_updatemd+0x97>
  40120d:	66 0f 2e 1d 03 02 00 	ucomisd 0x203(%rip),%xmm3        # 401418 <_IO_stdin_used+0x78>
  401214:	00 
  401215:	76 15                	jbe    40122c <_Z11Body_updatemd+0xac>
  401217:	f2 0f 10 1d 31 02 00 	movsd  0x231(%rip),%xmm3        # 401450 <_IO_stdin_used+0xb0>
  40121e:	00 
  40121f:	66 0f 57 d3          	xorpd  %xmm3,%xmm2
  401223:	f2 0f 11 14 fd a0 0e 	movsd  %xmm2,0x630ea0(,%rdi,8)
  40122a:	63 00 
  40122c:	66 0f 2e e0          	ucomisd %xmm0,%xmm4
  401230:	77 0a                	ja     40123c <_Z11Body_updatemd+0xbc>
  401232:	66 0f 2e 05 de 01 00 	ucomisd 0x1de(%rip),%xmm0        # 401418 <_IO_stdin_used+0x78>
  401239:	00 
  40123a:	76 15                	jbe    401251 <_Z11Body_updatemd+0xd1>
  40123c:	f2 0f 10 05 0c 02 00 	movsd  0x20c(%rip),%xmm0        # 401450 <_IO_stdin_used+0xb0>
  401243:	00 
  401244:	66 0f 57 c8          	xorpd  %xmm0,%xmm1
  401248:	f2 0f 11 0c fd a0 14 	movsd  %xmm1,0x6214a0(,%rdi,8)
  40124f:	62 00 
  401251:	f3 c3                	repz retq 
  401253:	0f 1f 00             	nopl   (%rax)
  401256:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40125d:	00 00 00 

0000000000401260 <_Z26Body_codegen_simple_updatemd>:
  401260:	f2 0f 10 14 fd a0 20 	movsd  0x6020a0(,%rdi,8),%xmm2
  401267:	60 00 
  401269:	f2 0f 10 1c fd a0 1a 	movsd  0x611aa0(,%rdi,8),%xmm3
  401270:	61 00 
  401272:	f2 0f 59 d0          	mulsd  %xmm0,%xmm2
  401276:	f2 0f 10 0c fd a0 fc 	movsd  0x65fca0(,%rdi,8),%xmm1
  40127d:	65 00 
  40127f:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  401283:	f2 0f 5e d1          	divsd  %xmm1,%xmm2
  401287:	f2 0f 5e d9          	divsd  %xmm1,%xmm3
  40128b:	f2 0f 58 14 fd a0 14 	addsd  0x6214a0(,%rdi,8),%xmm2
  401292:	62 00 
  401294:	66 0f 28 ca          	movapd %xmm2,%xmm1
  401298:	f2 0f 11 14 fd a0 14 	movsd  %xmm2,0x6214a0(,%rdi,8)
  40129f:	62 00 
  4012a1:	f2 0f 59 c8          	mulsd  %xmm0,%xmm1
  4012a5:	f2 0f 58 1c fd a0 0e 	addsd  0x630ea0(,%rdi,8),%xmm3
  4012ac:	63 00 
  4012ae:	f2 0f 58 0c fd a0 08 	addsd  0x6408a0(,%rdi,8),%xmm1
  4012b5:	64 00 
  4012b7:	f2 0f 11 1c fd a0 0e 	movsd  %xmm3,0x630ea0(,%rdi,8)
  4012be:	63 00 
  4012c0:	f2 0f 59 d8          	mulsd  %xmm0,%xmm3
  4012c4:	f2 0f 11 0c fd a0 08 	movsd  %xmm1,0x6408a0(,%rdi,8)
  4012cb:	64 00 
  4012cd:	f2 0f 58 1c fd a0 02 	addsd  0x6502a0(,%rdi,8),%xmm3
  4012d4:	65 00 
  4012d6:	f2 0f 11 1c fd a0 02 	movsd  %xmm3,0x6502a0(,%rdi,8)
  4012dd:	65 00 
  4012df:	c3                   	retq   

00000000004012e0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  4012e0:	55                   	push   %rbp
  4012e1:	53                   	push   %rbx
  4012e2:	48 89 fd             	mov    %rdi,%rbp
  4012e5:	48 83 ec 08          	sub    $0x8,%rsp
  4012e9:	e8 d2 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012ee:	48 89 c3             	mov    %rax,%rbx
  4012f1:	ff d5                	callq  *%rbp
  4012f3:	e8 c8 f4 ff ff       	callq  4007c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4012f8:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  4012ff:	de 1b 43 
  401302:	48 29 d8             	sub    %rbx,%rax
  401305:	48 83 c4 08          	add    $0x8,%rsp
  401309:	48 89 c1             	mov    %rax,%rcx
  40130c:	48 f7 ea             	imul   %rdx
  40130f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401313:	5b                   	pop    %rbx
  401314:	5d                   	pop    %rbp
  401315:	48 c1 fa 12          	sar    $0x12,%rdx
  401319:	48 89 d0             	mov    %rdx,%rax
  40131c:	48 29 c8             	sub    %rcx,%rax
  40131f:	c3                   	retq   

0000000000401320 <__libc_csu_init>:
  401320:	41 57                	push   %r15
  401322:	41 56                	push   %r14
  401324:	41 89 ff             	mov    %edi,%r15d
  401327:	41 55                	push   %r13
  401329:	41 54                	push   %r12
  40132b:	4c 8d 25 b6 0a 20 00 	lea    0x200ab6(%rip),%r12        # 601de8 <__frame_dummy_init_array_entry>
  401332:	55                   	push   %rbp
  401333:	48 8d 2d be 0a 20 00 	lea    0x200abe(%rip),%rbp        # 601df8 <__init_array_end>
  40133a:	53                   	push   %rbx
  40133b:	49 89 f6             	mov    %rsi,%r14
  40133e:	49 89 d5             	mov    %rdx,%r13
  401341:	4c 29 e5             	sub    %r12,%rbp
  401344:	48 83 ec 08          	sub    $0x8,%rsp
  401348:	48 c1 fd 03          	sar    $0x3,%rbp
  40134c:	e8 c7 f3 ff ff       	callq  400718 <_init>
  401351:	48 85 ed             	test   %rbp,%rbp
  401354:	74 20                	je     401376 <__libc_csu_init+0x56>
  401356:	31 db                	xor    %ebx,%ebx
  401358:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40135f:	00 
  401360:	4c 89 ea             	mov    %r13,%rdx
  401363:	4c 89 f6             	mov    %r14,%rsi
  401366:	44 89 ff             	mov    %r15d,%edi
  401369:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40136d:	48 83 c3 01          	add    $0x1,%rbx
  401371:	48 39 eb             	cmp    %rbp,%rbx
  401374:	75 ea                	jne    401360 <__libc_csu_init+0x40>
  401376:	48 83 c4 08          	add    $0x8,%rsp
  40137a:	5b                   	pop    %rbx
  40137b:	5d                   	pop    %rbp
  40137c:	41 5c                	pop    %r12
  40137e:	41 5d                	pop    %r13
  401380:	41 5e                	pop    %r14
  401382:	41 5f                	pop    %r15
  401384:	c3                   	retq   
  401385:	90                   	nop
  401386:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40138d:	00 00 00 

0000000000401390 <__libc_csu_fini>:
  401390:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000401394 <_fini>:
  401394:	48 83 ec 08          	sub    $0x8,%rsp
  401398:	48 83 c4 08          	add    $0x8,%rsp
  40139c:	c3                   	retq   
