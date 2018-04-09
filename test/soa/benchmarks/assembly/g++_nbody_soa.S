
bin/g++_nbody_soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004006b0 <_init>:
  4006b0:	48 83 ec 08          	sub    $0x8,%rsp
  4006b4:	48 8b 05 3d 19 20 00 	mov    0x20193d(%rip),%rax        # 601ff8 <_DYNAMIC+0x1e0>
  4006bb:	48 85 c0             	test   %rax,%rax
  4006be:	74 05                	je     4006c5 <_init+0x15>
  4006c0:	e8 ab 00 00 00       	callq  400770 <rand@plt+0x10>
  4006c5:	48 83 c4 08          	add    $0x8,%rsp
  4006c9:	c3                   	retq   

Disassembly of section .plt:

00000000004006d0 <memset@plt-0x10>:
  4006d0:	ff 35 32 19 20 00    	pushq  0x201932(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006d6:	ff 25 34 19 20 00    	jmpq   *0x201934(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006e0 <memset@plt>:
  4006e0:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006e6:	68 00 00 00 00       	pushq  $0x0
  4006eb:	e9 e0 ff ff ff       	jmpq   4006d0 <_init+0x20>

00000000004006f0 <__printf_chk@plt>:
  4006f0:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006f6:	68 01 00 00 00       	pushq  $0x1
  4006fb:	e9 d0 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400700 <_ZNSt8ios_base4InitC1Ev@plt>:
  400700:	ff 25 22 19 20 00    	jmpq   *0x201922(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400706:	68 02 00 00 00       	pushq  $0x2
  40070b:	e9 c0 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400710 <__libc_start_main@plt>:
  400710:	ff 25 1a 19 20 00    	jmpq   *0x20191a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400716:	68 03 00 00 00       	pushq  $0x3
  40071b:	e9 b0 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400720 <__cxa_atexit@plt>:
  400720:	ff 25 12 19 20 00    	jmpq   *0x201912(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400726:	68 04 00 00 00       	pushq  $0x4
  40072b:	e9 a0 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400730 <_ZNSt8ios_base4InitD1Ev@plt>:
  400730:	ff 25 0a 19 20 00    	jmpq   *0x20190a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400736:	68 05 00 00 00       	pushq  $0x5
  40073b:	e9 90 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400740 <srand@plt>:
  400740:	ff 25 02 19 20 00    	jmpq   *0x201902(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400746:	68 06 00 00 00       	pushq  $0x6
  40074b:	e9 80 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400750:	ff 25 fa 18 20 00    	jmpq   *0x2018fa(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400756:	68 07 00 00 00       	pushq  $0x7
  40075b:	e9 70 ff ff ff       	jmpq   4006d0 <_init+0x20>

0000000000400760 <rand@plt>:
  400760:	ff 25 f2 18 20 00    	jmpq   *0x2018f2(%rip)        # 602058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400766:	68 08 00 00 00       	pushq  $0x8
  40076b:	e9 60 ff ff ff       	jmpq   4006d0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400770 <.plt.got>:
  400770:	ff 25 82 18 20 00    	jmpq   *0x201882(%rip)        # 601ff8 <_DYNAMIC+0x1e0>
  400776:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400780 <main>:
  400780:	41 55                	push   %r13
  400782:	bf d0 09 40 00       	mov    $0x4009d0,%edi
  400787:	41 bd 8d 68 b7 6c    	mov    $0x6cb7688d,%r13d
  40078d:	41 54                	push   %r12
  40078f:	55                   	push   %rbp
  400790:	53                   	push   %rbx
  400791:	31 db                	xor    %ebx,%ebx
  400793:	48 83 ec 18          	sub    $0x18,%rsp
  400797:	e8 64 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40079c:	bf c0 0b 40 00       	mov    $0x400bc0,%edi
  4007a1:	48 89 c5             	mov    %rax,%rbp
  4007a4:	e8 57 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007a9:	bf 10 0b 40 00       	mov    $0x400b10,%edi
  4007ae:	49 89 c4             	mov    %rax,%r12
  4007b1:	e8 4a 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007b6:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  4007bc:	49 89 c0             	mov    %rax,%r8
  4007bf:	90                   	nop
  4007c0:	c5 fb 10 04 dd a0 02 	vmovsd 0x6502a0(,%rbx,8),%xmm0
  4007c7:	65 00 
  4007c9:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  4007cd:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  4007d1:	c5 fb 10 0c dd a0 08 	vmovsd 0x6408a0(,%rbx,8),%xmm1
  4007d8:	64 00 
  4007da:	c5 eb 5a d0          	vcvtsd2ss %xmm0,%xmm2,%xmm2
  4007de:	c5 f9 7e d1          	vmovd  %xmm2,%ecx
  4007e2:	41 01 c9             	add    %ecx,%r9d
  4007e5:	c5 e3 5a d9          	vcvtsd2ss %xmm1,%xmm3,%xmm3
  4007e9:	c5 f9 7e d9          	vmovd  %xmm3,%ecx
  4007ed:	44 01 c9             	add    %r9d,%ecx
  4007f0:	89 c8                	mov    %ecx,%eax
  4007f2:	41 f7 ed             	imul   %r13d
  4007f5:	89 c8                	mov    %ecx,%eax
  4007f7:	c1 f8 1f             	sar    $0x1f,%eax
  4007fa:	c1 fa 13             	sar    $0x13,%edx
  4007fd:	41 89 d1             	mov    %edx,%r9d
  400800:	41 29 c1             	sub    %eax,%r9d
  400803:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  40080a:	44 29 c9             	sub    %r9d,%ecx
  40080d:	41 89 c9             	mov    %ecx,%r9d
  400810:	48 83 fb 09          	cmp    $0x9,%rbx
  400814:	76 31                	jbe    400847 <main+0xc7>
  400816:	48 83 c3 01          	add    $0x1,%rbx
  40081a:	48 81 fb 40 1f 00 00 	cmp    $0x1f40,%rbx
  400821:	75 9d                	jne    4007c0 <main+0x40>
  400823:	4c 89 e1             	mov    %r12,%rcx
  400826:	48 89 ea             	mov    %rbp,%rdx
  400829:	be e0 11 40 00       	mov    $0x4011e0,%esi
  40082e:	bf 01 00 00 00       	mov    $0x1,%edi
  400833:	31 c0                	xor    %eax,%eax
  400835:	e8 b6 fe ff ff       	callq  4006f0 <__printf_chk@plt>
  40083a:	48 83 c4 18          	add    $0x18,%rsp
  40083e:	31 c0                	xor    %eax,%eax
  400840:	5b                   	pop    %rbx
  400841:	5d                   	pop    %rbp
  400842:	41 5c                	pop    %r12
  400844:	41 5d                	pop    %r13
  400846:	c3                   	retq   
  400847:	48 89 da             	mov    %rbx,%rdx
  40084a:	be c4 11 40 00       	mov    $0x4011c4,%esi
  40084f:	bf 01 00 00 00       	mov    $0x1,%edi
  400854:	4c 89 04 24          	mov    %r8,(%rsp)
  400858:	b8 02 00 00 00       	mov    $0x2,%eax
  40085d:	89 4c 24 0c          	mov    %ecx,0xc(%rsp)
  400861:	e8 8a fe ff ff       	callq  4006f0 <__printf_chk@plt>
  400866:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  40086b:	4c 8b 04 24          	mov    (%rsp),%r8
  40086f:	eb a5                	jmp    400816 <main+0x96>
  400871:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400876:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40087d:	00 00 00 

0000000000400880 <_GLOBAL__sub_I__Z11r_float2intf>:
  400880:	48 83 ec 08          	sub    $0x8,%rsp
  400884:	bf a0 f6 66 00       	mov    $0x66f6a0,%edi
  400889:	e8 72 fe ff ff       	callq  400700 <_ZNSt8ios_base4InitC1Ev@plt>
  40088e:	ba 68 20 60 00       	mov    $0x602068,%edx
  400893:	be a0 f6 66 00       	mov    $0x66f6a0,%esi
  400898:	bf 30 07 40 00       	mov    $0x400730,%edi
  40089d:	48 83 c4 08          	add    $0x8,%rsp
  4008a1:	e9 7a fe ff ff       	jmpq   400720 <__cxa_atexit@plt>
  4008a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ad:	00 00 00 

00000000004008b0 <set_fast_math>:
  4008b0:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  4008b5:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  4008bc:	00 
  4008bd:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  4008c2:	c3                   	retq   
  4008c3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ca:	00 00 00 
  4008cd:	0f 1f 00             	nopl   (%rax)

00000000004008d0 <_start>:
  4008d0:	31 ed                	xor    %ebp,%ebp
  4008d2:	49 89 d1             	mov    %rdx,%r9
  4008d5:	5e                   	pop    %rsi
  4008d6:	48 89 e2             	mov    %rsp,%rdx
  4008d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008dd:	50                   	push   %rax
  4008de:	54                   	push   %rsp
  4008df:	49 c7 c0 b0 11 40 00 	mov    $0x4011b0,%r8
  4008e6:	48 c7 c1 40 11 40 00 	mov    $0x401140,%rcx
  4008ed:	48 c7 c7 80 07 40 00 	mov    $0x400780,%rdi
  4008f4:	e8 17 fe ff ff       	callq  400710 <__libc_start_main@plt>
  4008f9:	f4                   	hlt    
  4008fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400900 <deregister_tm_clones>:
  400900:	b8 77 20 60 00       	mov    $0x602077,%eax
  400905:	55                   	push   %rbp
  400906:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  40090c:	48 83 f8 0e          	cmp    $0xe,%rax
  400910:	48 89 e5             	mov    %rsp,%rbp
  400913:	76 1b                	jbe    400930 <deregister_tm_clones+0x30>
  400915:	b8 00 00 00 00       	mov    $0x0,%eax
  40091a:	48 85 c0             	test   %rax,%rax
  40091d:	74 11                	je     400930 <deregister_tm_clones+0x30>
  40091f:	5d                   	pop    %rbp
  400920:	bf 70 20 60 00       	mov    $0x602070,%edi
  400925:	ff e0                	jmpq   *%rax
  400927:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40092e:	00 00 
  400930:	5d                   	pop    %rbp
  400931:	c3                   	retq   
  400932:	0f 1f 40 00          	nopl   0x0(%rax)
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <register_tm_clones>:
  400940:	be 70 20 60 00       	mov    $0x602070,%esi
  400945:	55                   	push   %rbp
  400946:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40094d:	48 c1 fe 03          	sar    $0x3,%rsi
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 89 f0             	mov    %rsi,%rax
  400957:	48 c1 e8 3f          	shr    $0x3f,%rax
  40095b:	48 01 c6             	add    %rax,%rsi
  40095e:	48 d1 fe             	sar    %rsi
  400961:	74 15                	je     400978 <register_tm_clones+0x38>
  400963:	b8 00 00 00 00       	mov    $0x0,%eax
  400968:	48 85 c0             	test   %rax,%rax
  40096b:	74 0b                	je     400978 <register_tm_clones+0x38>
  40096d:	5d                   	pop    %rbp
  40096e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400973:	ff e0                	jmpq   *%rax
  400975:	0f 1f 00             	nopl   (%rax)
  400978:	5d                   	pop    %rbp
  400979:	c3                   	retq   
  40097a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400980 <__do_global_dtors_aux>:
  400980:	80 3d f9 16 20 00 00 	cmpb   $0x0,0x2016f9(%rip)        # 602080 <completed.7594>
  400987:	75 11                	jne    40099a <__do_global_dtors_aux+0x1a>
  400989:	55                   	push   %rbp
  40098a:	48 89 e5             	mov    %rsp,%rbp
  40098d:	e8 6e ff ff ff       	callq  400900 <deregister_tm_clones>
  400992:	5d                   	pop    %rbp
  400993:	c6 05 e6 16 20 00 01 	movb   $0x1,0x2016e6(%rip)        # 602080 <completed.7594>
  40099a:	f3 c3                	repz retq 
  40099c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009a0 <frame_dummy>:
  4009a0:	bf 10 1e 60 00       	mov    $0x601e10,%edi
  4009a5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4009a9:	75 05                	jne    4009b0 <frame_dummy+0x10>
  4009ab:	eb 93                	jmp    400940 <register_tm_clones>
  4009ad:	0f 1f 00             	nopl   (%rax)
  4009b0:	b8 00 00 00 00       	mov    $0x0,%eax
  4009b5:	48 85 c0             	test   %rax,%rax
  4009b8:	74 f1                	je     4009ab <frame_dummy+0xb>
  4009ba:	55                   	push   %rbp
  4009bb:	48 89 e5             	mov    %rsp,%rbp
  4009be:	ff d0                	callq  *%rax
  4009c0:	5d                   	pop    %rbp
  4009c1:	e9 7a ff ff ff       	jmpq   400940 <register_tm_clones>
  4009c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009cd:	00 00 00 

00000000004009d0 <_Z13instantiationv>:
  4009d0:	41 54                	push   %r12
  4009d2:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4009d7:	55                   	push   %rbp
  4009d8:	53                   	push   %rbx
  4009d9:	31 db                	xor    %ebx,%ebx
  4009db:	48 83 ec 10          	sub    $0x10,%rsp
  4009df:	e8 5c fd ff ff       	callq  400740 <srand@plt>
  4009e4:	0f 1f 40 00          	nopl   0x0(%rax)
  4009e8:	e8 73 fd ff ff       	callq  400760 <rand@plt>
  4009ed:	48 83 c3 08          	add    $0x8,%rbx
  4009f1:	41 89 c4             	mov    %eax,%r12d
  4009f4:	e8 67 fd ff ff       	callq  400760 <rand@plt>
  4009f9:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
  4009fd:	c5 fb 10 1d 23 08 00 	vmovsd 0x823(%rip),%xmm3        # 401228 <_IO_stdin_used+0x68>
  400a04:	00 
  400a05:	c5 eb 2a d0          	vcvtsi2sd %eax,%xmm2,%xmm2
  400a09:	c4 e2 e1 99 15 0e 08 	vfmadd132sd 0x80e(%rip),%xmm3,%xmm2        # 401220 <_IO_stdin_used+0x60>
  400a10:	00 00 
  400a12:	c5 fb 11 54 24 08    	vmovsd %xmm2,0x8(%rsp)
  400a18:	e8 43 fd ff ff       	callq  400760 <rand@plt>
  400a1d:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  400a21:	c5 fb 10 25 ff 07 00 	vmovsd 0x7ff(%rip),%xmm4        # 401228 <_IO_stdin_used+0x68>
  400a28:	00 
  400a29:	c5 f3 2a c8          	vcvtsi2sd %eax,%xmm1,%xmm1
  400a2d:	c4 e2 d9 99 0d ea 07 	vfmadd132sd 0x7ea(%rip),%xmm4,%xmm1        # 401220 <_IO_stdin_used+0x60>
  400a34:	00 00 
  400a36:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  400a3b:	e8 20 fd ff ff       	callq  400760 <rand@plt>
  400a40:	89 c5                	mov    %eax,%ebp
  400a42:	e8 19 fd ff ff       	callq  400760 <rand@plt>
  400a47:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a4b:	c5 fb 10 54 24 08    	vmovsd 0x8(%rsp),%xmm2
  400a51:	48 c7 83 98 1a 61 00 	movq   $0x0,0x611a98(%rbx)
  400a58:	00 00 00 00 
  400a5c:	c4 c1 7b 2a c4       	vcvtsi2sd %r12d,%xmm0,%xmm0
  400a61:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  400a66:	48 c7 83 98 20 60 00 	movq   $0x0,0x602098(%rbx)
  400a6d:	00 00 00 00 
  400a71:	c5 fb 10 2d bf 07 00 	vmovsd 0x7bf(%rip),%xmm5        # 401238 <_IO_stdin_used+0x78>
  400a78:	00 
  400a79:	c5 fb 11 93 98 02 65 	vmovsd %xmm2,0x650298(%rbx)
  400a80:	00 
  400a81:	c4 e2 d1 99 05 a6 07 	vfmadd132sd 0x7a6(%rip),%xmm5,%xmm0        # 401230 <_IO_stdin_used+0x70>
  400a88:	00 00 
  400a8a:	c5 fb 59 05 ae 07 00 	vmulsd 0x7ae(%rip),%xmm0,%xmm0        # 401240 <_IO_stdin_used+0x80>
  400a91:	00 
  400a92:	c5 fb 11 8b 98 08 64 	vmovsd %xmm1,0x640898(%rbx)
  400a99:	00 
  400a9a:	c5 fb 10 35 ae 07 00 	vmovsd 0x7ae(%rip),%xmm6        # 401250 <_IO_stdin_used+0x90>
  400aa1:	00 
  400aa2:	c5 fb 11 83 98 fc 65 	vmovsd %xmm0,0x65fc98(%rbx)
  400aa9:	00 
  400aaa:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400aae:	c5 fb 2a c5          	vcvtsi2sd %ebp,%xmm0,%xmm0
  400ab2:	c4 e2 c9 99 05 8d 07 	vfmadd132sd 0x78d(%rip),%xmm6,%xmm0        # 401248 <_IO_stdin_used+0x88>
  400ab9:	00 00 
  400abb:	c5 fb 59 05 95 07 00 	vmulsd 0x795(%rip),%xmm0,%xmm0        # 401258 <_IO_stdin_used+0x98>
  400ac2:	00 
  400ac3:	c5 fb 11 83 98 0e 63 	vmovsd %xmm0,0x630e98(%rbx)
  400aca:	00 
  400acb:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400acf:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400ad3:	c4 e2 c9 99 05 6c 07 	vfmadd132sd 0x76c(%rip),%xmm6,%xmm0        # 401248 <_IO_stdin_used+0x88>
  400ada:	00 00 
  400adc:	c5 fb 59 05 74 07 00 	vmulsd 0x774(%rip),%xmm0,%xmm0        # 401258 <_IO_stdin_used+0x98>
  400ae3:	00 
  400ae4:	c5 fb 11 83 98 14 62 	vmovsd %xmm0,0x621498(%rbx)
  400aeb:	00 
  400aec:	48 81 fb 00 fa 00 00 	cmp    $0xfa00,%rbx
  400af3:	0f 85 ef fe ff ff    	jne    4009e8 <_Z13instantiationv+0x18>
  400af9:	48 83 c4 10          	add    $0x10,%rsp
  400afd:	5b                   	pop    %rbx
  400afe:	5d                   	pop    %rbp
  400aff:	41 5c                	pop    %r12
  400b01:	c3                   	retq   
  400b02:	0f 1f 40 00          	nopl   0x0(%rax)
  400b06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0d:	00 00 00 

0000000000400b10 <_Z10run_simplev>:
  400b10:	c5 fb 10 15 20 07 00 	vmovsd 0x720(%rip),%xmm2        # 401238 <_IO_stdin_used+0x78>
  400b17:	00 
  400b18:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400b1d:	31 d2                	xor    %edx,%edx
  400b1f:	90                   	nop
  400b20:	c5 eb 59 b2 a0 1a 61 	vmulsd 0x611aa0(%rdx),%xmm2,%xmm6
  400b27:	00 
  400b28:	b8 64 00 00 00       	mov    $0x64,%eax
  400b2d:	c5 eb 59 aa a0 20 60 	vmulsd 0x6020a0(%rdx),%xmm2,%xmm5
  400b34:	00 
  400b35:	c5 fb 10 9a a0 fc 65 	vmovsd 0x65fca0(%rdx),%xmm3
  400b3c:	00 
  400b3d:	c5 fb 10 8a a0 0e 63 	vmovsd 0x630ea0(%rdx),%xmm1
  400b44:	00 
  400b45:	c5 fb 10 82 a0 14 62 	vmovsd 0x6214a0(%rdx),%xmm0
  400b4c:	00 
  400b4d:	c5 cb 5e f3          	vdivsd %xmm3,%xmm6,%xmm6
  400b51:	c5 fb 10 a2 a0 02 65 	vmovsd 0x6502a0(%rdx),%xmm4
  400b58:	00 
  400b59:	c5 d3 5e eb          	vdivsd %xmm3,%xmm5,%xmm5
  400b5d:	c5 fb 10 9a a0 08 64 	vmovsd 0x6408a0(%rdx),%xmm3
  400b64:	00 
  400b65:	0f 1f 00             	nopl   (%rax)
  400b68:	c5 f3 58 ce          	vaddsd %xmm6,%xmm1,%xmm1
  400b6c:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
  400b70:	c4 e2 f1 b9 e2       	vfmadd231sd %xmm2,%xmm1,%xmm4
  400b75:	c4 e2 f9 b9 da       	vfmadd231sd %xmm2,%xmm0,%xmm3
  400b7a:	83 e8 01             	sub    $0x1,%eax
  400b7d:	75 e9                	jne    400b68 <_Z10run_simplev+0x58>
  400b7f:	c5 fb 11 8a a0 0e 63 	vmovsd %xmm1,0x630ea0(%rdx)
  400b86:	00 
  400b87:	48 83 c2 08          	add    $0x8,%rdx
  400b8b:	c5 fb 11 82 98 14 62 	vmovsd %xmm0,0x621498(%rdx)
  400b92:	00 
  400b93:	c5 fb 11 a2 98 02 65 	vmovsd %xmm4,0x650298(%rdx)
  400b9a:	00 
  400b9b:	c5 fb 11 9a 98 08 64 	vmovsd %xmm3,0x640898(%rdx)
  400ba2:	00 
  400ba3:	48 81 fa 00 fa 00 00 	cmp    $0xfa00,%rdx
  400baa:	0f 85 70 ff ff ff    	jne    400b20 <_Z10run_simplev+0x10>
  400bb0:	83 e9 01             	sub    $0x1,%ecx
  400bb3:	0f 85 64 ff ff ff    	jne    400b1d <_Z10run_simplev+0xd>
  400bb9:	c3                   	retq   
  400bba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bc0 <_Z14run_simulationv>:
  400bc0:	53                   	push   %rbx
  400bc1:	bb 05 00 00 00       	mov    $0x5,%ebx
  400bc6:	48 83 ec 20          	sub    $0x20,%rsp
  400bca:	c5 fb 10 35 66 06 00 	vmovsd 0x666(%rip),%xmm6        # 401238 <_IO_stdin_used+0x78>
  400bd1:	00 
  400bd2:	c5 fb 10 3d 96 06 00 	vmovsd 0x696(%rip),%xmm7        # 401270 <_IO_stdin_used+0xb0>
  400bd9:	00 
  400bda:	c5 fb 10 2d 7e 06 00 	vmovsd 0x67e(%rip),%xmm5        # 401260 <_IO_stdin_used+0xa0>
  400be1:	00 
  400be2:	c5 fb 10 25 7e 06 00 	vmovsd 0x67e(%rip),%xmm4        # 401268 <_IO_stdin_used+0xa8>
  400be9:	00 
  400bea:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400bef:	31 f6                	xor    %esi,%esi
  400bf1:	bf a0 1a 61 00       	mov    $0x611aa0,%edi
  400bf6:	c5 fb 11 64 24 18    	vmovsd %xmm4,0x18(%rsp)
  400bfc:	c5 fb 11 6c 24 10    	vmovsd %xmm5,0x10(%rsp)
  400c02:	c5 fb 11 74 24 08    	vmovsd %xmm6,0x8(%rsp)
  400c08:	c5 fb 11 3c 24       	vmovsd %xmm7,(%rsp)
  400c0d:	e8 ce fa ff ff       	callq  4006e0 <memset@plt>
  400c12:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c17:	31 f6                	xor    %esi,%esi
  400c19:	bf a0 20 60 00       	mov    $0x6020a0,%edi
  400c1e:	e8 bd fa ff ff       	callq  4006e0 <memset@plt>
  400c23:	c5 fb 10 64 24 18    	vmovsd 0x18(%rsp),%xmm4
  400c29:	31 c9                	xor    %ecx,%ecx
  400c2b:	c5 fb 10 6c 24 10    	vmovsd 0x10(%rsp),%xmm5
  400c31:	c5 fb 10 74 24 08    	vmovsd 0x8(%rsp),%xmm6
  400c37:	c5 fb 10 3c 24       	vmovsd (%rsp),%xmm7
  400c3c:	0f 1f 40 00          	nopl   0x0(%rax)
  400c40:	31 c0                	xor    %eax,%eax
  400c42:	31 d2                	xor    %edx,%edx
  400c44:	0f 1f 40 00          	nopl   0x0(%rax)
  400c48:	48 39 d1             	cmp    %rdx,%rcx
  400c4b:	74 7f                	je     400ccc <_Z14run_simulationv+0x10c>
  400c4d:	c5 fb 10 04 cd a0 08 	vmovsd 0x6408a0(,%rcx,8),%xmm0
  400c54:	64 00 
  400c56:	c5 7b 5c 80 a0 08 64 	vsubsd 0x6408a0(%rax),%xmm0,%xmm8
  400c5d:	00 
  400c5e:	c5 fb 10 14 cd a0 02 	vmovsd 0x6502a0(,%rcx,8),%xmm2
  400c65:	65 00 
  400c67:	c5 eb 5c 90 a0 02 65 	vsubsd 0x6502a0(%rax),%xmm2,%xmm2
  400c6e:	00 
  400c6f:	c4 c1 3b 59 d8       	vmulsd %xmm8,%xmm8,%xmm3
  400c74:	c5 d3 59 0c cd a0 fc 	vmulsd 0x65fca0(,%rcx,8),%xmm5,%xmm1
  400c7b:	65 00 
  400c7d:	c5 f3 59 88 a0 fc 65 	vmulsd 0x65fca0(%rax),%xmm1,%xmm1
  400c84:	00 
  400c85:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
  400c8a:	c5 e3 51 db          	vsqrtsd %xmm3,%xmm3,%xmm3
  400c8e:	c5 f9 28 c3          	vmovapd %xmm3,%xmm0
  400c92:	c4 e2 d9 99 c3       	vfmadd132sd %xmm3,%xmm4,%xmm0
  400c97:	c5 f3 5e c0          	vdivsd %xmm0,%xmm1,%xmm0
  400c9b:	c5 fb 59 d2          	vmulsd %xmm2,%xmm0,%xmm2
  400c9f:	c4 c1 7b 59 c0       	vmulsd %xmm8,%xmm0,%xmm0
  400ca4:	c5 eb 5e d3          	vdivsd %xmm3,%xmm2,%xmm2
  400ca8:	c5 eb 58 90 a0 1a 61 	vaddsd 0x611aa0(%rax),%xmm2,%xmm2
  400caf:	00 
  400cb0:	c5 fb 5e c3          	vdivsd %xmm3,%xmm0,%xmm0
  400cb4:	c5 fb 11 90 a0 1a 61 	vmovsd %xmm2,0x611aa0(%rax)
  400cbb:	00 
  400cbc:	c5 fb 58 80 a0 20 60 	vaddsd 0x6020a0(%rax),%xmm0,%xmm0
  400cc3:	00 
  400cc4:	c5 fb 11 80 a0 20 60 	vmovsd %xmm0,0x6020a0(%rax)
  400ccb:	00 
  400ccc:	48 83 c2 01          	add    $0x1,%rdx
  400cd0:	48 83 c0 08          	add    $0x8,%rax
  400cd4:	48 81 fa 40 1f 00 00 	cmp    $0x1f40,%rdx
  400cdb:	0f 85 67 ff ff ff    	jne    400c48 <_Z14run_simulationv+0x88>
  400ce1:	48 83 c1 01          	add    $0x1,%rcx
  400ce5:	48 81 f9 40 1f 00 00 	cmp    $0x1f40,%rcx
  400cec:	0f 85 4e ff ff ff    	jne    400c40 <_Z14run_simulationv+0x80>
  400cf2:	c5 7b 10 05 86 05 00 	vmovsd 0x586(%rip),%xmm8        # 401280 <_IO_stdin_used+0xc0>
  400cf9:	00 
  400cfa:	31 c0                	xor    %eax,%eax
  400cfc:	c5 fb 10 1d 24 05 00 	vmovsd 0x524(%rip),%xmm3        # 401228 <_IO_stdin_used+0x68>
  400d03:	00 
  400d04:	c4 41 79 28 c8       	vmovapd %xmm8,%xmm9
  400d09:	c5 79 28 db          	vmovapd %xmm3,%xmm11
  400d0d:	c5 cb 59 88 a0 1a 61 	vmulsd 0x611aa0(%rax),%xmm6,%xmm1
  400d14:	00 
  400d15:	c5 fb 10 90 a0 fc 65 	vmovsd 0x65fca0(%rax),%xmm2
  400d1c:	00 
  400d1d:	c5 cb 59 80 a0 20 60 	vmulsd 0x6020a0(%rax),%xmm6,%xmm0
  400d24:	00 
  400d25:	c5 f3 5e ca          	vdivsd %xmm2,%xmm1,%xmm1
  400d29:	c5 f3 58 88 a0 0e 63 	vaddsd 0x630ea0(%rax),%xmm1,%xmm1
  400d30:	00 
  400d31:	c5 fb 5e c2          	vdivsd %xmm2,%xmm0,%xmm0
  400d35:	c5 79 28 d1          	vmovapd %xmm1,%xmm10
  400d39:	c5 fb 11 88 a0 0e 63 	vmovsd %xmm1,0x630ea0(%rax)
  400d40:	00 
  400d41:	c4 62 c9 a9 90 a0 02 	vfmadd213sd 0x6502a0(%rax),%xmm6,%xmm10
  400d48:	65 00 
  400d4a:	c5 79 2f d7          	vcomisd %xmm7,%xmm10
  400d4e:	c5 7b 11 90 a0 02 65 	vmovsd %xmm10,0x6502a0(%rax)
  400d55:	00 
  400d56:	c5 fb 58 80 a0 14 62 	vaddsd 0x6214a0(%rax),%xmm0,%xmm0
  400d5d:	00 
  400d5e:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400d62:	c5 fb 11 80 a0 14 62 	vmovsd %xmm0,0x6214a0(%rax)
  400d69:	00 
  400d6a:	c4 e2 c9 a9 90 a0 08 	vfmadd213sd 0x6408a0(%rax),%xmm6,%xmm2
  400d71:	64 00 
  400d73:	c5 fb 11 90 a0 08 64 	vmovsd %xmm2,0x6408a0(%rax)
  400d7a:	00 
  400d7b:	77 07                	ja     400d84 <_Z14run_simulationv+0x1c4>
  400d7d:	c4 c1 79 2f da       	vcomisd %xmm10,%xmm3
  400d82:	76 0d                	jbe    400d91 <_Z14run_simulationv+0x1d1>
  400d84:	c4 c1 71 57 c8       	vxorpd %xmm8,%xmm1,%xmm1
  400d89:	c5 fb 11 88 a0 0e 63 	vmovsd %xmm1,0x630ea0(%rax)
  400d90:	00 
  400d91:	c5 f9 2f d7          	vcomisd %xmm7,%xmm2
  400d95:	77 06                	ja     400d9d <_Z14run_simulationv+0x1dd>
  400d97:	c5 79 2f da          	vcomisd %xmm2,%xmm11
  400d9b:	76 0d                	jbe    400daa <_Z14run_simulationv+0x1ea>
  400d9d:	c4 c1 79 57 c1       	vxorpd %xmm9,%xmm0,%xmm0
  400da2:	c5 fb 11 80 a0 14 62 	vmovsd %xmm0,0x6214a0(%rax)
  400da9:	00 
  400daa:	48 83 c0 08          	add    $0x8,%rax
  400dae:	48 3d 00 fa 00 00    	cmp    $0xfa00,%rax
  400db4:	0f 85 53 ff ff ff    	jne    400d0d <_Z14run_simulationv+0x14d>
  400dba:	83 eb 01             	sub    $0x1,%ebx
  400dbd:	0f 85 27 fe ff ff    	jne    400bea <_Z14run_simulationv+0x2a>
  400dc3:	48 83 c4 20          	add    $0x20,%rsp
  400dc7:	5b                   	pop    %rbx
  400dc8:	c3                   	retq   
  400dc9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400dd0 <_Z11r_float2intf>:
  400dd0:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  400dd4:	c3                   	retq   
  400dd5:	90                   	nop
  400dd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddd:	00 00 00 

0000000000400de0 <_Z15Body_initializemddddd>:
  400de0:	c5 fb 11 04 fd a0 fc 	vmovsd %xmm0,0x65fca0(,%rdi,8)
  400de7:	65 00 
  400de9:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400ded:	c5 fb 11 0c fd a0 02 	vmovsd %xmm1,0x6502a0(,%rdi,8)
  400df4:	65 00 
  400df6:	c5 fb 11 14 fd a0 08 	vmovsd %xmm2,0x6408a0(,%rdi,8)
  400dfd:	64 00 
  400dff:	c5 fb 11 1c fd a0 0e 	vmovsd %xmm3,0x630ea0(,%rdi,8)
  400e06:	63 00 
  400e08:	c5 fb 11 24 fd a0 14 	vmovsd %xmm4,0x6214a0(,%rdi,8)
  400e0f:	62 00 
  400e11:	c5 fb 11 04 fd a0 1a 	vmovsd %xmm0,0x611aa0(,%rdi,8)
  400e18:	61 00 
  400e1a:	c5 fb 11 04 fd a0 20 	vmovsd %xmm0,0x6020a0(,%rdi,8)
  400e21:	60 00 
  400e23:	c3                   	retq   
  400e24:	66 90                	xchg   %ax,%ax
  400e26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2d:	00 00 00 

0000000000400e30 <_Z14Body_add_forcemm>:
  400e30:	48 39 f7             	cmp    %rsi,%rdi
  400e33:	0f 84 90 00 00 00    	je     400ec9 <_Z14Body_add_forcemm+0x99>
  400e39:	c5 fb 10 04 f5 a0 08 	vmovsd 0x6408a0(,%rsi,8),%xmm0
  400e40:	64 00 
  400e42:	c5 fb 5c 04 fd a0 08 	vsubsd 0x6408a0(,%rdi,8),%xmm0,%xmm0
  400e49:	64 00 
  400e4b:	c5 fb 10 0c f5 a0 02 	vmovsd 0x6502a0(,%rsi,8),%xmm1
  400e52:	65 00 
  400e54:	c5 fb 10 1d 04 04 00 	vmovsd 0x404(%rip),%xmm3        # 401260 <_IO_stdin_used+0xa0>
  400e5b:	00 
  400e5c:	c5 f3 5c 0c fd a0 02 	vsubsd 0x6502a0(,%rdi,8),%xmm1,%xmm1
  400e63:	65 00 
  400e65:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  400e69:	c5 e3 59 1c f5 a0 fc 	vmulsd 0x65fca0(,%rsi,8),%xmm3,%xmm3
  400e70:	65 00 
  400e72:	c5 e3 59 1c fd a0 fc 	vmulsd 0x65fca0(,%rdi,8),%xmm3,%xmm3
  400e79:	65 00 
  400e7b:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  400e80:	c5 eb 51 d2          	vsqrtsd %xmm2,%xmm2,%xmm2
  400e84:	c5 f9 28 e2          	vmovapd %xmm2,%xmm4
  400e88:	c4 e2 e9 a9 25 d7 03 	vfmadd213sd 0x3d7(%rip),%xmm2,%xmm4        # 401268 <_IO_stdin_used+0xa8>
  400e8f:	00 00 
  400e91:	c5 e3 5e dc          	vdivsd %xmm4,%xmm3,%xmm3
  400e95:	c5 f3 59 cb          	vmulsd %xmm3,%xmm1,%xmm1
  400e99:	c5 fb 59 c3          	vmulsd %xmm3,%xmm0,%xmm0
  400e9d:	c5 f3 5e ca          	vdivsd %xmm2,%xmm1,%xmm1
  400ea1:	c5 f3 58 0c fd a0 1a 	vaddsd 0x611aa0(,%rdi,8),%xmm1,%xmm1
  400ea8:	61 00 
  400eaa:	c5 fb 11 0c fd a0 1a 	vmovsd %xmm1,0x611aa0(,%rdi,8)
  400eb1:	61 00 
  400eb3:	c5 fb 5e c2          	vdivsd %xmm2,%xmm0,%xmm0
  400eb7:	c5 fb 58 04 fd a0 20 	vaddsd 0x6020a0(,%rdi,8),%xmm0,%xmm0
  400ebe:	60 00 
  400ec0:	c5 fb 11 04 fd a0 20 	vmovsd %xmm0,0x6020a0(,%rdi,8)
  400ec7:	60 00 
  400ec9:	c3                   	retq   
  400eca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400ed0 <_Z21Body_add_force_to_allm>:
  400ed0:	c5 fb 10 2d 88 03 00 	vmovsd 0x388(%rip),%xmm5        # 401260 <_IO_stdin_used+0xa0>
  400ed7:	00 
  400ed8:	31 c0                	xor    %eax,%eax
  400eda:	31 d2                	xor    %edx,%edx
  400edc:	c5 fb 10 25 84 03 00 	vmovsd 0x384(%rip),%xmm4        # 401268 <_IO_stdin_used+0xa8>
  400ee3:	00 
  400ee4:	0f 1f 40 00          	nopl   0x0(%rax)
  400ee8:	48 39 d7             	cmp    %rdx,%rdi
  400eeb:	74 7d                	je     400f6a <_Z21Body_add_force_to_allm+0x9a>
  400eed:	c5 fb 10 04 fd a0 08 	vmovsd 0x6408a0(,%rdi,8),%xmm0
  400ef4:	64 00 
  400ef6:	c5 fb 5c b0 a0 08 64 	vsubsd 0x6408a0(%rax),%xmm0,%xmm6
  400efd:	00 
  400efe:	c5 fb 10 14 fd a0 02 	vmovsd 0x6502a0(,%rdi,8),%xmm2
  400f05:	65 00 
  400f07:	c5 eb 5c 90 a0 02 65 	vsubsd 0x6502a0(%rax),%xmm2,%xmm2
  400f0e:	00 
  400f0f:	c5 cb 59 de          	vmulsd %xmm6,%xmm6,%xmm3
  400f13:	c5 d3 59 0c fd a0 fc 	vmulsd 0x65fca0(,%rdi,8),%xmm5,%xmm1
  400f1a:	65 00 
  400f1c:	c5 f3 59 88 a0 fc 65 	vmulsd 0x65fca0(%rax),%xmm1,%xmm1
  400f23:	00 
  400f24:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
  400f29:	c5 e3 51 db          	vsqrtsd %xmm3,%xmm3,%xmm3
  400f2d:	c5 f9 28 c3          	vmovapd %xmm3,%xmm0
  400f31:	c4 e2 d9 99 c3       	vfmadd132sd %xmm3,%xmm4,%xmm0
  400f36:	c5 f3 5e c0          	vdivsd %xmm0,%xmm1,%xmm0
  400f3a:	c5 fb 59 d2          	vmulsd %xmm2,%xmm0,%xmm2
  400f3e:	c5 fb 59 c6          	vmulsd %xmm6,%xmm0,%xmm0
  400f42:	c5 eb 5e d3          	vdivsd %xmm3,%xmm2,%xmm2
  400f46:	c5 eb 58 90 a0 1a 61 	vaddsd 0x611aa0(%rax),%xmm2,%xmm2
  400f4d:	00 
  400f4e:	c5 fb 5e c3          	vdivsd %xmm3,%xmm0,%xmm0
  400f52:	c5 fb 11 90 a0 1a 61 	vmovsd %xmm2,0x611aa0(%rax)
  400f59:	00 
  400f5a:	c5 fb 58 80 a0 20 60 	vaddsd 0x6020a0(%rax),%xmm0,%xmm0
  400f61:	00 
  400f62:	c5 fb 11 80 a0 20 60 	vmovsd %xmm0,0x6020a0(%rax)
  400f69:	00 
  400f6a:	48 83 c2 01          	add    $0x1,%rdx
  400f6e:	48 83 c0 08          	add    $0x8,%rax
  400f72:	48 81 fa 40 1f 00 00 	cmp    $0x1f40,%rdx
  400f79:	0f 85 69 ff ff ff    	jne    400ee8 <_Z21Body_add_force_to_allm+0x18>
  400f7f:	c3                   	retq   

0000000000400f80 <_Z16Body_reset_forcem>:
  400f80:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400f84:	c5 fb 11 04 fd a0 1a 	vmovsd %xmm0,0x611aa0(,%rdi,8)
  400f8b:	61 00 
  400f8d:	c5 fb 11 04 fd a0 20 	vmovsd %xmm0,0x6020a0(,%rdi,8)
  400f94:	60 00 
  400f96:	c3                   	retq   
  400f97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400f9e:	00 00 

0000000000400fa0 <_Z11Body_updatemd>:
  400fa0:	c5 fb 59 14 fd a0 1a 	vmulsd 0x611aa0(,%rdi,8),%xmm0,%xmm2
  400fa7:	61 00 
  400fa9:	c5 fb 59 0c fd a0 20 	vmulsd 0x6020a0(,%rdi,8),%xmm0,%xmm1
  400fb0:	60 00 
  400fb2:	c5 fb 10 1c fd a0 fc 	vmovsd 0x65fca0(,%rdi,8),%xmm3
  400fb9:	65 00 
  400fbb:	c5 fb 10 25 ad 02 00 	vmovsd 0x2ad(%rip),%xmm4        # 401270 <_IO_stdin_used+0xb0>
  400fc2:	00 
  400fc3:	c5 eb 5e d3          	vdivsd %xmm3,%xmm2,%xmm2
  400fc7:	c5 eb 58 14 fd a0 0e 	vaddsd 0x630ea0(,%rdi,8),%xmm2,%xmm2
  400fce:	63 00 
  400fd0:	c5 fb 11 14 fd a0 0e 	vmovsd %xmm2,0x630ea0(,%rdi,8)
  400fd7:	63 00 
  400fd9:	c5 f3 5e cb          	vdivsd %xmm3,%xmm1,%xmm1
  400fdd:	c5 f9 28 d8          	vmovapd %xmm0,%xmm3
  400fe1:	c4 e2 e9 a9 1c fd a0 	vfmadd213sd 0x6502a0(,%rdi,8),%xmm2,%xmm3
  400fe8:	02 65 00 
  400feb:	c5 fb 11 1c fd a0 02 	vmovsd %xmm3,0x6502a0(,%rdi,8)
  400ff2:	65 00 
  400ff4:	c5 f9 2f dc          	vcomisd %xmm4,%xmm3
  400ff8:	c5 f3 58 0c fd a0 14 	vaddsd 0x6214a0(,%rdi,8),%xmm1,%xmm1
  400fff:	62 00 
  401001:	c4 e2 f1 a9 04 fd a0 	vfmadd213sd 0x6408a0(,%rdi,8),%xmm1,%xmm0
  401008:	08 64 00 
  40100b:	c5 fb 11 04 fd a0 08 	vmovsd %xmm0,0x6408a0(,%rdi,8)
  401012:	64 00 
  401014:	c5 fb 11 0c fd a0 14 	vmovsd %xmm1,0x6214a0(,%rdi,8)
  40101b:	62 00 
  40101d:	77 0e                	ja     40102d <_Z11Body_updatemd+0x8d>
  40101f:	c5 fb 10 2d 01 02 00 	vmovsd 0x201(%rip),%xmm5        # 401228 <_IO_stdin_used+0x68>
  401026:	00 
  401027:	c5 f9 2f eb          	vcomisd %xmm3,%xmm5
  40102b:	76 15                	jbe    401042 <_Z11Body_updatemd+0xa2>
  40102d:	c5 fb 10 1d 4b 02 00 	vmovsd 0x24b(%rip),%xmm3        # 401280 <_IO_stdin_used+0xc0>
  401034:	00 
  401035:	c5 e9 57 d3          	vxorpd %xmm3,%xmm2,%xmm2
  401039:	c5 fb 11 14 fd a0 0e 	vmovsd %xmm2,0x630ea0(,%rdi,8)
  401040:	63 00 
  401042:	c5 f9 2f c4          	vcomisd %xmm4,%xmm0
  401046:	77 0e                	ja     401056 <_Z11Body_updatemd+0xb6>
  401048:	c5 fb 10 15 d8 01 00 	vmovsd 0x1d8(%rip),%xmm2        # 401228 <_IO_stdin_used+0x68>
  40104f:	00 
  401050:	c5 f9 2f d0          	vcomisd %xmm0,%xmm2
  401054:	76 15                	jbe    40106b <_Z11Body_updatemd+0xcb>
  401056:	c5 fb 10 05 22 02 00 	vmovsd 0x222(%rip),%xmm0        # 401280 <_IO_stdin_used+0xc0>
  40105d:	00 
  40105e:	c5 f1 57 c8          	vxorpd %xmm0,%xmm1,%xmm1
  401062:	c5 fb 11 0c fd a0 14 	vmovsd %xmm1,0x6214a0(,%rdi,8)
  401069:	62 00 
  40106b:	c3                   	retq   
  40106c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401070 <_Z26Body_codegen_simple_updatemd>:
  401070:	c5 fb 59 34 fd a0 1a 	vmulsd 0x611aa0(,%rdi,8),%xmm0,%xmm6
  401077:	61 00 
  401079:	b8 64 00 00 00       	mov    $0x64,%eax
  40107e:	c5 fb 59 2c fd a0 20 	vmulsd 0x6020a0(,%rdi,8),%xmm0,%xmm5
  401085:	60 00 
  401087:	c5 fb 10 1c fd a0 fc 	vmovsd 0x65fca0(,%rdi,8),%xmm3
  40108e:	65 00 
  401090:	c5 fb 10 14 fd a0 0e 	vmovsd 0x630ea0(,%rdi,8),%xmm2
  401097:	63 00 
  401099:	c5 fb 10 0c fd a0 14 	vmovsd 0x6214a0(,%rdi,8),%xmm1
  4010a0:	62 00 
  4010a2:	c5 fb 10 24 fd a0 02 	vmovsd 0x6502a0(,%rdi,8),%xmm4
  4010a9:	65 00 
  4010ab:	c5 cb 5e f3          	vdivsd %xmm3,%xmm6,%xmm6
  4010af:	c5 d3 5e eb          	vdivsd %xmm3,%xmm5,%xmm5
  4010b3:	c5 fb 10 1c fd a0 08 	vmovsd 0x6408a0(,%rdi,8),%xmm3
  4010ba:	64 00 
  4010bc:	0f 1f 40 00          	nopl   0x0(%rax)
  4010c0:	c5 eb 58 d6          	vaddsd %xmm6,%xmm2,%xmm2
  4010c4:	c5 f3 58 cd          	vaddsd %xmm5,%xmm1,%xmm1
  4010c8:	c4 e2 f9 b9 e2       	vfmadd231sd %xmm2,%xmm0,%xmm4
  4010cd:	c4 e2 f9 b9 d9       	vfmadd231sd %xmm1,%xmm0,%xmm3
  4010d2:	83 e8 01             	sub    $0x1,%eax
  4010d5:	75 e9                	jne    4010c0 <_Z26Body_codegen_simple_updatemd+0x50>
  4010d7:	c5 fb 11 14 fd a0 0e 	vmovsd %xmm2,0x630ea0(,%rdi,8)
  4010de:	63 00 
  4010e0:	c5 fb 11 0c fd a0 14 	vmovsd %xmm1,0x6214a0(,%rdi,8)
  4010e7:	62 00 
  4010e9:	c5 fb 11 24 fd a0 02 	vmovsd %xmm4,0x6502a0(,%rdi,8)
  4010f0:	65 00 
  4010f2:	c5 fb 11 1c fd a0 08 	vmovsd %xmm3,0x6408a0(,%rdi,8)
  4010f9:	64 00 
  4010fb:	c3                   	retq   
  4010fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401100:	55                   	push   %rbp
  401101:	48 89 fd             	mov    %rdi,%rbp
  401104:	53                   	push   %rbx
  401105:	48 83 ec 08          	sub    $0x8,%rsp
  401109:	e8 42 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40110e:	48 89 c3             	mov    %rax,%rbx
  401111:	ff d5                	callq  *%rbp
  401113:	e8 38 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401118:	48 83 c4 08          	add    $0x8,%rsp
  40111c:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  401123:	de 1b 43 
  401126:	48 29 d8             	sub    %rbx,%rax
  401129:	5b                   	pop    %rbx
  40112a:	48 89 c1             	mov    %rax,%rcx
  40112d:	5d                   	pop    %rbp
  40112e:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401132:	48 f7 ea             	imul   %rdx
  401135:	48 89 d0             	mov    %rdx,%rax
  401138:	48 c1 f8 12          	sar    $0x12,%rax
  40113c:	48 29 c8             	sub    %rcx,%rax
  40113f:	c3                   	retq   

0000000000401140 <__libc_csu_init>:
  401140:	41 57                	push   %r15
  401142:	41 56                	push   %r14
  401144:	41 89 ff             	mov    %edi,%r15d
  401147:	41 55                	push   %r13
  401149:	41 54                	push   %r12
  40114b:	4c 8d 25 9e 0c 20 00 	lea    0x200c9e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  401152:	55                   	push   %rbp
  401153:	48 8d 2d ae 0c 20 00 	lea    0x200cae(%rip),%rbp        # 601e08 <__init_array_end>
  40115a:	53                   	push   %rbx
  40115b:	49 89 f6             	mov    %rsi,%r14
  40115e:	49 89 d5             	mov    %rdx,%r13
  401161:	4c 29 e5             	sub    %r12,%rbp
  401164:	48 83 ec 08          	sub    $0x8,%rsp
  401168:	48 c1 fd 03          	sar    $0x3,%rbp
  40116c:	e8 3f f5 ff ff       	callq  4006b0 <_init>
  401171:	48 85 ed             	test   %rbp,%rbp
  401174:	74 20                	je     401196 <__libc_csu_init+0x56>
  401176:	31 db                	xor    %ebx,%ebx
  401178:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40117f:	00 
  401180:	4c 89 ea             	mov    %r13,%rdx
  401183:	4c 89 f6             	mov    %r14,%rsi
  401186:	44 89 ff             	mov    %r15d,%edi
  401189:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40118d:	48 83 c3 01          	add    $0x1,%rbx
  401191:	48 39 eb             	cmp    %rbp,%rbx
  401194:	75 ea                	jne    401180 <__libc_csu_init+0x40>
  401196:	48 83 c4 08          	add    $0x8,%rsp
  40119a:	5b                   	pop    %rbx
  40119b:	5d                   	pop    %rbp
  40119c:	41 5c                	pop    %r12
  40119e:	41 5d                	pop    %r13
  4011a0:	41 5e                	pop    %r14
  4011a2:	41 5f                	pop    %r15
  4011a4:	c3                   	retq   
  4011a5:	90                   	nop
  4011a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011ad:	00 00 00 

00000000004011b0 <__libc_csu_fini>:
  4011b0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004011b4 <_fini>:
  4011b4:	48 83 ec 08          	sub    $0x8,%rsp
  4011b8:	48 83 c4 08          	add    $0x8,%rsp
  4011bc:	c3                   	retq   
