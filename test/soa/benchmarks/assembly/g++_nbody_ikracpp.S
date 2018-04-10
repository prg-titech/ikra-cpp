
bin/g++_nbody_ikracpp:     file format elf64-x86-64


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
  400780:	41 56                	push   %r14
  400782:	bf f0 09 40 00       	mov    $0x4009f0,%edi
  400787:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  40078d:	41 55                	push   %r13
  40078f:	41 bd 90 1a 61 00    	mov    $0x611a90,%r13d
  400795:	41 54                	push   %r12
  400797:	45 31 e4             	xor    %r12d,%r12d
  40079a:	55                   	push   %rbp
  40079b:	53                   	push   %rbx
  40079c:	48 83 ec 10          	sub    $0x10,%rsp
  4007a0:	48 c7 05 cd 18 20 00 	movq   $0x0,0x2018cd(%rip)        # 602078 <__Bodydata_buffer>
  4007a7:	00 00 00 00 
  4007ab:	c7 05 cb 18 20 00 00 	movl   $0x0,0x2018cb(%rip)        # 602080 <__Bodydata_buffer+0x8>
  4007b2:	00 00 00 
  4007b5:	e8 36 09 00 00       	callq  4010f0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007ba:	bf 00 0c 40 00       	mov    $0x400c00,%edi
  4007bf:	48 89 c3             	mov    %rax,%rbx
  4007c2:	e8 29 09 00 00       	callq  4010f0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007c7:	bf 40 0b 40 00       	mov    $0x400b40,%edi
  4007cc:	48 89 c5             	mov    %rax,%rbp
  4007cf:	e8 1c 09 00 00       	callq  4010f0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007d4:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  4007da:	49 89 c0             	mov    %rax,%r8
  4007dd:	0f 1f 00             	nopl   (%rax)
  4007e0:	c4 c1 7b 10 45 00    	vmovsd 0x0(%r13),%xmm0
  4007e6:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  4007ea:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  4007ee:	c4 c1 7b 10 8d 00 fa 	vmovsd 0xfa00(%r13),%xmm1
  4007f5:	00 00 
  4007f7:	c5 eb 5a d0          	vcvtsd2ss %xmm0,%xmm2,%xmm2
  4007fb:	c5 f9 7e d6          	vmovd  %xmm2,%esi
  4007ff:	41 01 f1             	add    %esi,%r9d
  400802:	c5 e3 5a d9          	vcvtsd2ss %xmm1,%xmm3,%xmm3
  400806:	c5 f9 7e de          	vmovd  %xmm3,%esi
  40080a:	44 01 ce             	add    %r9d,%esi
  40080d:	89 f0                	mov    %esi,%eax
  40080f:	41 f7 ee             	imul   %r14d
  400812:	89 f0                	mov    %esi,%eax
  400814:	c1 f8 1f             	sar    $0x1f,%eax
  400817:	c1 fa 13             	sar    $0x13,%edx
  40081a:	41 89 d1             	mov    %edx,%r9d
  40081d:	41 29 c1             	sub    %eax,%r9d
  400820:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  400827:	44 29 ce             	sub    %r9d,%esi
  40082a:	41 89 f1             	mov    %esi,%r9d
  40082d:	49 83 fc 09          	cmp    $0x9,%r12
  400831:	76 37                	jbe    40086a <main+0xea>
  400833:	49 83 c4 01          	add    $0x1,%r12
  400837:	49 83 c5 08          	add    $0x8,%r13
  40083b:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400842:	75 9c                	jne    4007e0 <main+0x60>
  400844:	48 89 e9             	mov    %rbp,%rcx
  400847:	48 89 da             	mov    %rbx,%rdx
  40084a:	be d0 11 40 00       	mov    $0x4011d0,%esi
  40084f:	bf 01 00 00 00       	mov    $0x1,%edi
  400854:	31 c0                	xor    %eax,%eax
  400856:	e8 95 fe ff ff       	callq  4006f0 <__printf_chk@plt>
  40085b:	48 83 c4 10          	add    $0x10,%rsp
  40085f:	31 c0                	xor    %eax,%eax
  400861:	5b                   	pop    %rbx
  400862:	5d                   	pop    %rbp
  400863:	41 5c                	pop    %r12
  400865:	41 5d                	pop    %r13
  400867:	41 5e                	pop    %r14
  400869:	c3                   	retq   
  40086a:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  40086e:	4c 89 e2             	mov    %r12,%rdx
  400871:	be b4 11 40 00       	mov    $0x4011b4,%esi
  400876:	bf 01 00 00 00       	mov    $0x1,%edi
  40087b:	b8 02 00 00 00       	mov    $0x2,%eax
  400880:	4c 89 04 24          	mov    %r8,(%rsp)
  400884:	e8 67 fe ff ff       	callq  4006f0 <__printf_chk@plt>
  400889:	4c 8b 04 24          	mov    (%rsp),%r8
  40088d:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  400892:	eb 9f                	jmp    400833 <main+0xb3>
  400894:	66 90                	xchg   %ax,%ax
  400896:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089d:	00 00 00 

00000000004008a0 <_GLOBAL__sub_I__Z11r_float2intf>:
  4008a0:	48 83 ec 08          	sub    $0x8,%rsp
  4008a4:	bf 88 f6 66 00       	mov    $0x66f688,%edi
  4008a9:	e8 52 fe ff ff       	callq  400700 <_ZNSt8ios_base4InitC1Ev@plt>
  4008ae:	ba 68 20 60 00       	mov    $0x602068,%edx
  4008b3:	be 88 f6 66 00       	mov    $0x66f688,%esi
  4008b8:	bf 30 07 40 00       	mov    $0x400730,%edi
  4008bd:	48 83 c4 08          	add    $0x8,%rsp
  4008c1:	e9 5a fe ff ff       	jmpq   400720 <__cxa_atexit@plt>
  4008c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008cd:	00 00 00 

00000000004008d0 <set_fast_math>:
  4008d0:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  4008d5:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  4008dc:	00 
  4008dd:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  4008e2:	c3                   	retq   
  4008e3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ea:	00 00 00 
  4008ed:	0f 1f 00             	nopl   (%rax)

00000000004008f0 <_start>:
  4008f0:	31 ed                	xor    %ebp,%ebp
  4008f2:	49 89 d1             	mov    %rdx,%r9
  4008f5:	5e                   	pop    %rsi
  4008f6:	48 89 e2             	mov    %rsp,%rdx
  4008f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008fd:	50                   	push   %rax
  4008fe:	54                   	push   %rsp
  4008ff:	49 c7 c0 a0 11 40 00 	mov    $0x4011a0,%r8
  400906:	48 c7 c1 30 11 40 00 	mov    $0x401130,%rcx
  40090d:	48 c7 c7 80 07 40 00 	mov    $0x400780,%rdi
  400914:	e8 f7 fd ff ff       	callq  400710 <__libc_start_main@plt>
  400919:	f4                   	hlt    
  40091a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400920 <deregister_tm_clones>:
  400920:	b8 77 20 60 00       	mov    $0x602077,%eax
  400925:	55                   	push   %rbp
  400926:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  40092c:	48 83 f8 0e          	cmp    $0xe,%rax
  400930:	48 89 e5             	mov    %rsp,%rbp
  400933:	76 1b                	jbe    400950 <deregister_tm_clones+0x30>
  400935:	b8 00 00 00 00       	mov    $0x0,%eax
  40093a:	48 85 c0             	test   %rax,%rax
  40093d:	74 11                	je     400950 <deregister_tm_clones+0x30>
  40093f:	5d                   	pop    %rbp
  400940:	bf 70 20 60 00       	mov    $0x602070,%edi
  400945:	ff e0                	jmpq   *%rax
  400947:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40094e:	00 00 
  400950:	5d                   	pop    %rbp
  400951:	c3                   	retq   
  400952:	0f 1f 40 00          	nopl   0x0(%rax)
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <register_tm_clones>:
  400960:	be 70 20 60 00       	mov    $0x602070,%esi
  400965:	55                   	push   %rbp
  400966:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40096d:	48 c1 fe 03          	sar    $0x3,%rsi
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 89 f0             	mov    %rsi,%rax
  400977:	48 c1 e8 3f          	shr    $0x3f,%rax
  40097b:	48 01 c6             	add    %rax,%rsi
  40097e:	48 d1 fe             	sar    %rsi
  400981:	74 15                	je     400998 <register_tm_clones+0x38>
  400983:	b8 00 00 00 00       	mov    $0x0,%eax
  400988:	48 85 c0             	test   %rax,%rax
  40098b:	74 0b                	je     400998 <register_tm_clones+0x38>
  40098d:	5d                   	pop    %rbp
  40098e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400993:	ff e0                	jmpq   *%rax
  400995:	0f 1f 00             	nopl   (%rax)
  400998:	5d                   	pop    %rbp
  400999:	c3                   	retq   
  40099a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009a0 <__do_global_dtors_aux>:
  4009a0:	80 3d c9 16 20 00 00 	cmpb   $0x0,0x2016c9(%rip)        # 602070 <__TMC_END__>
  4009a7:	75 11                	jne    4009ba <__do_global_dtors_aux+0x1a>
  4009a9:	55                   	push   %rbp
  4009aa:	48 89 e5             	mov    %rsp,%rbp
  4009ad:	e8 6e ff ff ff       	callq  400920 <deregister_tm_clones>
  4009b2:	5d                   	pop    %rbp
  4009b3:	c6 05 b6 16 20 00 01 	movb   $0x1,0x2016b6(%rip)        # 602070 <__TMC_END__>
  4009ba:	f3 c3                	repz retq 
  4009bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009c0 <frame_dummy>:
  4009c0:	bf 10 1e 60 00       	mov    $0x601e10,%edi
  4009c5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4009c9:	75 05                	jne    4009d0 <frame_dummy+0x10>
  4009cb:	eb 93                	jmp    400960 <register_tm_clones>
  4009cd:	0f 1f 00             	nopl   (%rax)
  4009d0:	b8 00 00 00 00       	mov    $0x0,%eax
  4009d5:	48 85 c0             	test   %rax,%rax
  4009d8:	74 f1                	je     4009cb <frame_dummy+0xb>
  4009da:	55                   	push   %rbp
  4009db:	48 89 e5             	mov    %rsp,%rbp
  4009de:	ff d0                	callq  *%rax
  4009e0:	5d                   	pop    %rbp
  4009e1:	e9 7a ff ff ff       	jmpq   400960 <register_tm_clones>
  4009e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ed:	00 00 00 

00000000004009f0 <_Z13instantiationv>:
  4009f0:	41 54                	push   %r12
  4009f2:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4009f7:	55                   	push   %rbp
  4009f8:	53                   	push   %rbx
  4009f9:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4009fe:	48 83 ec 10          	sub    $0x10,%rsp
  400a02:	e8 39 fd ff ff       	callq  400740 <srand@plt>
  400a07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a0e:	00 00 
  400a10:	e8 4b fd ff ff       	callq  400760 <rand@plt>
  400a15:	41 89 c4             	mov    %eax,%r12d
  400a18:	e8 43 fd ff ff       	callq  400760 <rand@plt>
  400a1d:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
  400a21:	c5 fb 10 1d ef 07 00 	vmovsd 0x7ef(%rip),%xmm3        # 401218 <_IO_stdin_used+0x68>
  400a28:	00 
  400a29:	c5 eb 2a d0          	vcvtsi2sd %eax,%xmm2,%xmm2
  400a2d:	c4 e2 e1 99 15 da 07 	vfmadd132sd 0x7da(%rip),%xmm3,%xmm2        # 401210 <_IO_stdin_used+0x60>
  400a34:	00 00 
  400a36:	c5 fb 11 54 24 08    	vmovsd %xmm2,0x8(%rsp)
  400a3c:	e8 1f fd ff ff       	callq  400760 <rand@plt>
  400a41:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  400a45:	c5 fb 10 25 cb 07 00 	vmovsd 0x7cb(%rip),%xmm4        # 401218 <_IO_stdin_used+0x68>
  400a4c:	00 
  400a4d:	c5 f3 2a c8          	vcvtsi2sd %eax,%xmm1,%xmm1
  400a51:	c4 e2 d9 99 0d b6 07 	vfmadd132sd 0x7b6(%rip),%xmm4,%xmm1        # 401210 <_IO_stdin_used+0x60>
  400a58:	00 00 
  400a5a:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  400a5f:	e8 fc fc ff ff       	callq  400760 <rand@plt>
  400a64:	89 c5                	mov    %eax,%ebp
  400a66:	e8 f5 fc ff ff       	callq  400760 <rand@plt>
  400a6b:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a6f:	c5 fb 10 54 24 08    	vmovsd 0x8(%rsp),%xmm2
  400a75:	c4 c1 7b 2a c4       	vcvtsi2sd %r12d,%xmm0,%xmm0
  400a7a:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  400a7f:	48 8b 15 f2 15 20 00 	mov    0x2015f2(%rip),%rdx        # 602078 <__Bodydata_buffer>
  400a86:	c5 fb 10 2d 9a 07 00 	vmovsd 0x79a(%rip),%xmm5        # 401228 <_IO_stdin_used+0x78>
  400a8d:	00 
  400a8e:	c4 e2 d1 99 05 89 07 	vfmadd132sd 0x789(%rip),%xmm5,%xmm0        # 401220 <_IO_stdin_used+0x70>
  400a95:	00 00 
  400a97:	c5 fb 59 05 91 07 00 	vmulsd 0x791(%rip),%xmm0,%xmm0        # 401230 <_IO_stdin_used+0x80>
  400a9e:	00 
  400a9f:	c5 fb 10 35 99 07 00 	vmovsd 0x799(%rip),%xmm6        # 401240 <_IO_stdin_used+0x90>
  400aa6:	00 
  400aa7:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400aab:	48 83 c2 03          	add    $0x3,%rdx
  400aaf:	48 89 0d c2 15 20 00 	mov    %rcx,0x2015c2(%rip)        # 602078 <__Bodydata_buffer>
  400ab6:	c5 fb 11 14 d5 78 1a 	vmovsd %xmm2,0x611a78(,%rdx,8)
  400abd:	61 00 
  400abf:	c5 fb 11 04 d5 78 20 	vmovsd %xmm0,0x602078(,%rdx,8)
  400ac6:	60 00 
  400ac8:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400acc:	c5 fb 2a c5          	vcvtsi2sd %ebp,%xmm0,%xmm0
  400ad0:	c4 e2 c9 99 05 5f 07 	vfmadd132sd 0x75f(%rip),%xmm6,%xmm0        # 401238 <_IO_stdin_used+0x88>
  400ad7:	00 00 
  400ad9:	c5 fb 11 0c d5 78 14 	vmovsd %xmm1,0x621478(,%rdx,8)
  400ae0:	62 00 
  400ae2:	c5 fb 59 05 5e 07 00 	vmulsd 0x75e(%rip),%xmm0,%xmm0        # 401248 <_IO_stdin_used+0x98>
  400ae9:	00 
  400aea:	48 c7 04 d5 78 02 65 	movq   $0x0,0x650278(,%rdx,8)
  400af1:	00 00 00 00 00 
  400af6:	48 c7 04 d5 78 fc 65 	movq   $0x0,0x65fc78(,%rdx,8)
  400afd:	00 00 00 00 00 
  400b02:	c5 fb 11 04 d5 78 0e 	vmovsd %xmm0,0x630e78(,%rdx,8)
  400b09:	63 00 
  400b0b:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400b0f:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400b13:	c4 e2 c9 99 05 1c 07 	vfmadd132sd 0x71c(%rip),%xmm6,%xmm0        # 401238 <_IO_stdin_used+0x88>
  400b1a:	00 00 
  400b1c:	c5 fb 59 05 24 07 00 	vmulsd 0x724(%rip),%xmm0,%xmm0        # 401248 <_IO_stdin_used+0x98>
  400b23:	00 
  400b24:	c5 fb 11 04 d5 78 08 	vmovsd %xmm0,0x640878(,%rdx,8)
  400b2b:	64 00 
  400b2d:	83 eb 01             	sub    $0x1,%ebx
  400b30:	0f 85 da fe ff ff    	jne    400a10 <_Z13instantiationv+0x20>
  400b36:	48 83 c4 10          	add    $0x10,%rsp
  400b3a:	5b                   	pop    %rbx
  400b3b:	5d                   	pop    %rbp
  400b3c:	41 5c                	pop    %r12
  400b3e:	c3                   	retq   
  400b3f:	90                   	nop

0000000000400b40 <_Z10run_simplev>:
  400b40:	c5 fb 10 15 e0 06 00 	vmovsd 0x6e0(%rip),%xmm2        # 401228 <_IO_stdin_used+0x78>
  400b47:	00 
  400b48:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400b4d:	ba 90 20 60 00       	mov    $0x602090,%edx
  400b52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b58:	c5 eb 59 b2 00 e2 04 	vmulsd 0x4e200(%rdx),%xmm2,%xmm6
  400b5f:	00 
  400b60:	b8 64 00 00 00       	mov    $0x64,%eax
  400b65:	c5 eb 59 aa 00 dc 05 	vmulsd 0x5dc00(%rdx),%xmm2,%xmm5
  400b6c:	00 
  400b6d:	c5 fb 10 02          	vmovsd (%rdx),%xmm0
  400b71:	c5 fb 10 8a 00 ee 02 	vmovsd 0x2ee00(%rdx),%xmm1
  400b78:	00 
  400b79:	c5 fb 10 9a 00 fa 00 	vmovsd 0xfa00(%rdx),%xmm3
  400b80:	00 
  400b81:	c5 cb 5e f0          	vdivsd %xmm0,%xmm6,%xmm6
  400b85:	c5 fb 10 a2 00 f4 01 	vmovsd 0x1f400(%rdx),%xmm4
  400b8c:	00 
  400b8d:	c5 d3 5e e8          	vdivsd %xmm0,%xmm5,%xmm5
  400b91:	c5 fb 10 82 00 e8 03 	vmovsd 0x3e800(%rdx),%xmm0
  400b98:	00 
  400b99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400ba0:	c5 f3 58 ce          	vaddsd %xmm6,%xmm1,%xmm1
  400ba4:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
  400ba8:	c4 e2 f1 b9 da       	vfmadd231sd %xmm2,%xmm1,%xmm3
  400bad:	c4 e2 f9 b9 e2       	vfmadd231sd %xmm2,%xmm0,%xmm4
  400bb2:	83 e8 01             	sub    $0x1,%eax
  400bb5:	75 e9                	jne    400ba0 <_Z10run_simplev+0x60>
  400bb7:	c5 fb 11 8a 00 ee 02 	vmovsd %xmm1,0x2ee00(%rdx)
  400bbe:	00 
  400bbf:	48 83 c2 08          	add    $0x8,%rdx
  400bc3:	c5 fb 11 82 f8 e7 03 	vmovsd %xmm0,0x3e7f8(%rdx)
  400bca:	00 
  400bcb:	c5 fb 11 9a f8 f9 00 	vmovsd %xmm3,0xf9f8(%rdx)
  400bd2:	00 
  400bd3:	c5 fb 11 a2 f8 f3 01 	vmovsd %xmm4,0x1f3f8(%rdx)
  400bda:	00 
  400bdb:	48 81 fa 90 1a 61 00 	cmp    $0x611a90,%rdx
  400be2:	0f 85 70 ff ff ff    	jne    400b58 <_Z10run_simplev+0x18>
  400be8:	83 e9 01             	sub    $0x1,%ecx
  400beb:	0f 85 5c ff ff ff    	jne    400b4d <_Z10run_simplev+0xd>
  400bf1:	c3                   	retq   
  400bf2:	0f 1f 40 00          	nopl   0x0(%rax)
  400bf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bfd:	00 00 00 

0000000000400c00 <_Z14run_simulationv>:
  400c00:	53                   	push   %rbx
  400c01:	bb 05 00 00 00       	mov    $0x5,%ebx
  400c06:	48 83 ec 20          	sub    $0x20,%rsp
  400c0a:	c5 fb 10 35 16 06 00 	vmovsd 0x616(%rip),%xmm6        # 401228 <_IO_stdin_used+0x78>
  400c11:	00 
  400c12:	c5 fb 10 3d fe 05 00 	vmovsd 0x5fe(%rip),%xmm7        # 401218 <_IO_stdin_used+0x68>
  400c19:	00 
  400c1a:	c5 fb 10 2d 2e 06 00 	vmovsd 0x62e(%rip),%xmm5        # 401250 <_IO_stdin_used+0xa0>
  400c21:	00 
  400c22:	c5 fb 10 25 2e 06 00 	vmovsd 0x62e(%rip),%xmm4        # 401258 <_IO_stdin_used+0xa8>
  400c29:	00 
  400c2a:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c2f:	31 f6                	xor    %esi,%esi
  400c31:	bf 90 02 65 00       	mov    $0x650290,%edi
  400c36:	c5 fb 11 64 24 18    	vmovsd %xmm4,0x18(%rsp)
  400c3c:	c5 fb 11 6c 24 10    	vmovsd %xmm5,0x10(%rsp)
  400c42:	c5 fb 11 74 24 08    	vmovsd %xmm6,0x8(%rsp)
  400c48:	c5 fb 11 3c 24       	vmovsd %xmm7,(%rsp)
  400c4d:	e8 8e fa ff ff       	callq  4006e0 <memset@plt>
  400c52:	31 f6                	xor    %esi,%esi
  400c54:	ba 00 fa 00 00       	mov    $0xfa00,%edx
  400c59:	bf 90 fc 65 00       	mov    $0x65fc90,%edi
  400c5e:	e8 7d fa ff ff       	callq  4006e0 <memset@plt>
  400c63:	c5 fb 10 3c 24       	vmovsd (%rsp),%xmm7
  400c68:	b9 90 1a 61 00       	mov    $0x611a90,%ecx
  400c6d:	31 f6                	xor    %esi,%esi
  400c6f:	c5 fb 10 74 24 08    	vmovsd 0x8(%rsp),%xmm6
  400c75:	c5 fb 10 6c 24 10    	vmovsd 0x10(%rsp),%xmm5
  400c7b:	c5 fb 10 64 24 18    	vmovsd 0x18(%rsp),%xmm4
  400c81:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400c88:	48 83 c6 01          	add    $0x1,%rsi
  400c8c:	b8 90 02 65 00       	mov    $0x650290,%eax
  400c91:	31 d2                	xor    %edx,%edx
  400c93:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400c98:	48 83 c2 01          	add    $0x1,%rdx
  400c9c:	48 39 d6             	cmp    %rdx,%rsi
  400c9f:	74 70                	je     400d11 <_Z14run_simulationv+0x111>
  400ca1:	c5 fb 10 81 00 fa 00 	vmovsd 0xfa00(%rcx),%xmm0
  400ca8:	00 
  400ca9:	c5 7b 5c 80 00 12 fd 	vsubsd -0x2ee00(%rax),%xmm0,%xmm8
  400cb0:	ff 
  400cb1:	c5 fb 10 11          	vmovsd (%rcx),%xmm2
  400cb5:	c5 eb 5c 90 00 18 fc 	vsubsd -0x3e800(%rax),%xmm2,%xmm2
  400cbc:	ff 
  400cbd:	c4 c1 3b 59 d8       	vmulsd %xmm8,%xmm8,%xmm3
  400cc2:	c5 d3 59 89 00 06 ff 	vmulsd -0xfa00(%rcx),%xmm5,%xmm1
  400cc9:	ff 
  400cca:	c5 f3 59 88 00 1e fb 	vmulsd -0x4e200(%rax),%xmm1,%xmm1
  400cd1:	ff 
  400cd2:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
  400cd7:	c5 e3 51 db          	vsqrtsd %xmm3,%xmm3,%xmm3
  400cdb:	c5 f9 28 c3          	vmovapd %xmm3,%xmm0
  400cdf:	c4 e2 d9 99 c3       	vfmadd132sd %xmm3,%xmm4,%xmm0
  400ce4:	c5 f3 5e c0          	vdivsd %xmm0,%xmm1,%xmm0
  400ce8:	c5 fb 59 d2          	vmulsd %xmm2,%xmm0,%xmm2
  400cec:	c4 c1 7b 59 c0       	vmulsd %xmm8,%xmm0,%xmm0
  400cf1:	c5 eb 5e d3          	vdivsd %xmm3,%xmm2,%xmm2
  400cf5:	c5 eb 58 10          	vaddsd (%rax),%xmm2,%xmm2
  400cf9:	c5 fb 5e c3          	vdivsd %xmm3,%xmm0,%xmm0
  400cfd:	c5 fb 11 10          	vmovsd %xmm2,(%rax)
  400d01:	c5 fb 58 80 00 fa 00 	vaddsd 0xfa00(%rax),%xmm0,%xmm0
  400d08:	00 
  400d09:	c5 fb 11 80 00 fa 00 	vmovsd %xmm0,0xfa00(%rax)
  400d10:	00 
  400d11:	48 83 c0 08          	add    $0x8,%rax
  400d15:	48 81 fa 40 1f 00 00 	cmp    $0x1f40,%rdx
  400d1c:	0f 85 76 ff ff ff    	jne    400c98 <_Z14run_simulationv+0x98>
  400d22:	48 83 c1 08          	add    $0x8,%rcx
  400d26:	48 81 fe 40 1f 00 00 	cmp    $0x1f40,%rsi
  400d2d:	0f 85 55 ff ff ff    	jne    400c88 <_Z14run_simulationv+0x88>
  400d33:	c5 fb 10 15 35 05 00 	vmovsd 0x535(%rip),%xmm2        # 401270 <_IO_stdin_used+0xc0>
  400d3a:	00 
  400d3b:	b8 90 0e 63 00       	mov    $0x630e90,%eax
  400d40:	c5 7b 10 05 18 05 00 	vmovsd 0x518(%rip),%xmm8        # 401260 <_IO_stdin_used+0xb0>
  400d47:	00 
  400d48:	c5 f9 28 da          	vmovapd %xmm2,%xmm3
  400d4c:	c4 41 79 28 c8       	vmovapd %xmm8,%xmm9
  400d51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400d58:	c5 cb 59 88 00 f4 01 	vmulsd 0x1f400(%rax),%xmm6,%xmm1
  400d5f:	00 
  400d60:	c5 7b 10 90 00 12 fd 	vmovsd -0x2ee00(%rax),%xmm10
  400d67:	ff 
  400d68:	c5 cb 59 80 00 ee 02 	vmulsd 0x2ee00(%rax),%xmm6,%xmm0
  400d6f:	00 
  400d70:	c4 c1 73 5e ca       	vdivsd %xmm10,%xmm1,%xmm1
  400d75:	c5 f3 58 08          	vaddsd (%rax),%xmm1,%xmm1
  400d79:	c4 c1 7b 5e c2       	vdivsd %xmm10,%xmm0,%xmm0
  400d7e:	c5 79 28 d9          	vmovapd %xmm1,%xmm11
  400d82:	c5 fb 11 08          	vmovsd %xmm1,(%rax)
  400d86:	c4 62 c9 a9 98 00 0c 	vfmadd213sd -0x1f400(%rax),%xmm6,%xmm11
  400d8d:	fe ff 
  400d8f:	c4 c1 79 2f fb       	vcomisd %xmm11,%xmm7
  400d94:	c5 7b 11 98 00 0c fe 	vmovsd %xmm11,-0x1f400(%rax)
  400d9b:	ff 
  400d9c:	c5 fb 58 80 00 fa 00 	vaddsd 0xfa00(%rax),%xmm0,%xmm0
  400da3:	00 
  400da4:	c5 79 28 d0          	vmovapd %xmm0,%xmm10
  400da8:	c5 fb 11 80 00 fa 00 	vmovsd %xmm0,0xfa00(%rax)
  400daf:	00 
  400db0:	c4 62 c9 a9 90 00 06 	vfmadd213sd -0xfa00(%rax),%xmm6,%xmm10
  400db7:	ff ff 
  400db9:	c5 7b 11 90 00 06 ff 	vmovsd %xmm10,-0xfa00(%rax)
  400dc0:	ff 
  400dc1:	77 07                	ja     400dca <_Z14run_simulationv+0x1ca>
  400dc3:	c4 41 79 2f d8       	vcomisd %xmm8,%xmm11
  400dc8:	76 08                	jbe    400dd2 <_Z14run_simulationv+0x1d2>
  400dca:	c5 f1 57 ca          	vxorpd %xmm2,%xmm1,%xmm1
  400dce:	c5 fb 11 08          	vmovsd %xmm1,(%rax)
  400dd2:	c4 c1 79 2f fa       	vcomisd %xmm10,%xmm7
  400dd7:	77 07                	ja     400de0 <_Z14run_simulationv+0x1e0>
  400dd9:	c4 41 79 2f d1       	vcomisd %xmm9,%xmm10
  400dde:	76 0c                	jbe    400dec <_Z14run_simulationv+0x1ec>
  400de0:	c5 f9 57 c3          	vxorpd %xmm3,%xmm0,%xmm0
  400de4:	c5 fb 11 80 00 fa 00 	vmovsd %xmm0,0xfa00(%rax)
  400deb:	00 
  400dec:	48 83 c0 08          	add    $0x8,%rax
  400df0:	48 3d 90 08 64 00    	cmp    $0x640890,%rax
  400df6:	0f 85 5c ff ff ff    	jne    400d58 <_Z14run_simulationv+0x158>
  400dfc:	83 eb 01             	sub    $0x1,%ebx
  400dff:	0f 85 25 fe ff ff    	jne    400c2a <_Z14run_simulationv+0x2a>
  400e05:	48 83 c4 20          	add    $0x20,%rsp
  400e09:	5b                   	pop    %rbx
  400e0a:	c3                   	retq   
  400e0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e10 <_Z11r_float2intf>:
  400e10:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  400e14:	c3                   	retq   
  400e15:	90                   	nop
  400e16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e1d:	00 00 00 

0000000000400e20 <_Z19codegen_reset_forceP4Body>:
  400e20:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400e24:	c5 fb 11 04 fd 88 02 	vmovsd %xmm0,0x650288(,%rdi,8)
  400e2b:	65 00 
  400e2d:	c5 fb 11 04 fd 88 fc 	vmovsd %xmm0,0x65fc88(,%rdi,8)
  400e34:	65 00 
  400e36:	c3                   	retq   
  400e37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e3e:	00 00 

0000000000400e40 <_Z17codegen_add_forceP4BodyS0_>:
  400e40:	48 39 f7             	cmp    %rsi,%rdi
  400e43:	0f 84 94 00 00 00    	je     400edd <_Z17codegen_add_forceP4BodyS0_+0x9d>
  400e49:	c5 fb 10 04 f5 88 14 	vmovsd 0x621488(,%rsi,8),%xmm0
  400e50:	62 00 
  400e52:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  400e59:	00 
  400e5a:	c5 fb 5c 04 fd 88 14 	vsubsd 0x621488(,%rdi,8),%xmm0,%xmm0
  400e61:	62 00 
  400e63:	c5 fb 10 0c f5 88 1a 	vmovsd 0x611a88(,%rsi,8),%xmm1
  400e6a:	61 00 
  400e6c:	c5 fb 10 1c fd 88 20 	vmovsd 0x602088(,%rdi,8),%xmm3
  400e73:	60 00 
  400e75:	c5 f3 5c 0c fd 88 1a 	vsubsd 0x611a88(,%rdi,8),%xmm1,%xmm1
  400e7c:	61 00 
  400e7e:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  400e82:	c5 e3 59 1c f5 88 20 	vmulsd 0x602088(,%rsi,8),%xmm3,%xmm3
  400e89:	60 00 
  400e8b:	c5 e3 59 1d bd 03 00 	vmulsd 0x3bd(%rip),%xmm3,%xmm3        # 401250 <_IO_stdin_used+0xa0>
  400e92:	00 
  400e93:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  400e98:	c5 eb 51 d2          	vsqrtsd %xmm2,%xmm2,%xmm2
  400e9c:	c5 f9 28 e2          	vmovapd %xmm2,%xmm4
  400ea0:	c4 e2 e9 a9 25 af 03 	vfmadd213sd 0x3af(%rip),%xmm2,%xmm4        # 401258 <_IO_stdin_used+0xa8>
  400ea7:	00 00 
  400ea9:	c5 e3 5e dc          	vdivsd %xmm4,%xmm3,%xmm3
  400ead:	c5 f3 59 cb          	vmulsd %xmm3,%xmm1,%xmm1
  400eb1:	c5 fb 59 c3          	vmulsd %xmm3,%xmm0,%xmm0
  400eb5:	c5 f3 5e ca          	vdivsd %xmm2,%xmm1,%xmm1
  400eb9:	c5 f3 58 88 88 02 65 	vaddsd 0x650288(%rax),%xmm1,%xmm1
  400ec0:	00 
  400ec1:	c5 fb 5e c2          	vdivsd %xmm2,%xmm0,%xmm0
  400ec5:	c5 fb 11 88 88 02 65 	vmovsd %xmm1,0x650288(%rax)
  400ecc:	00 
  400ecd:	c5 fb 58 80 88 fc 65 	vaddsd 0x65fc88(%rax),%xmm0,%xmm0
  400ed4:	00 
  400ed5:	c5 fb 11 80 88 fc 65 	vmovsd %xmm0,0x65fc88(%rax)
  400edc:	00 
  400edd:	c3                   	retq   
  400ede:	66 90                	xchg   %ax,%ax

0000000000400ee0 <_Z14codegen_updateP4Bodyd>:
  400ee0:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  400ee7:	00 
  400ee8:	48 8d 88 10 ee 02 00 	lea    0x2ee10(%rax),%rcx
  400eef:	c5 fb 59 0c fd 88 02 	vmulsd 0x650288(,%rdi,8),%xmm0,%xmm1
  400ef6:	65 00 
  400ef8:	c5 f3 5e 88 88 20 60 	vdivsd 0x602088(%rax),%xmm1,%xmm1
  400eff:	00 
  400f00:	c5 f3 58 89 78 20 60 	vaddsd 0x602078(%rcx),%xmm1,%xmm1
  400f07:	00 
  400f08:	48 8d 90 10 e8 03 00 	lea    0x3e810(%rax),%rdx
  400f0f:	48 8d b0 10 fa 00 00 	lea    0xfa10(%rax),%rsi
  400f16:	48 05 10 f4 01 00    	add    $0x1f410,%rax
  400f1c:	c5 fb 11 89 78 20 60 	vmovsd %xmm1,0x602078(%rcx)
  400f23:	00 
  400f24:	c5 fb 59 0c fd 88 fc 	vmulsd 0x65fc88(,%rdi,8),%xmm0,%xmm1
  400f2b:	65 00 
  400f2d:	c5 f3 5e 88 78 2c 5e 	vdivsd 0x5e2c78(%rax),%xmm1,%xmm1
  400f34:	00 
  400f35:	c5 f3 58 8a 78 20 60 	vaddsd 0x602078(%rdx),%xmm1,%xmm1
  400f3c:	00 
  400f3d:	c5 fb 11 8a 78 20 60 	vmovsd %xmm1,0x602078(%rdx)
  400f44:	00 
  400f45:	c5 fb 10 89 78 20 60 	vmovsd 0x602078(%rcx),%xmm1
  400f4c:	00 
  400f4d:	c4 e2 f9 a9 88 78 26 	vfmadd213sd 0x5f2678(%rax),%xmm0,%xmm1
  400f54:	5f 00 
  400f56:	c5 fb 11 88 78 26 5f 	vmovsd %xmm1,0x5f2678(%rax)
  400f5d:	00 
  400f5e:	c5 fb 10 a0 78 20 60 	vmovsd 0x602078(%rax),%xmm4
  400f65:	00 
  400f66:	c4 e2 d9 99 82 78 20 	vfmadd132sd 0x602078(%rdx),%xmm4,%xmm0
  400f6d:	60 00 
  400f6f:	c5 fb 10 0d e9 02 00 	vmovsd 0x2e9(%rip),%xmm1        # 401260 <_IO_stdin_used+0xb0>
  400f76:	00 
  400f77:	c5 fb 11 80 78 20 60 	vmovsd %xmm0,0x602078(%rax)
  400f7e:	00 
  400f7f:	c5 fb 10 96 78 20 60 	vmovsd 0x602078(%rsi),%xmm2
  400f86:	00 
  400f87:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
  400f8b:	77 0e                	ja     400f9b <_Z14codegen_updateP4Bodyd+0xbb>
  400f8d:	c5 fb 10 1d 83 02 00 	vmovsd 0x283(%rip),%xmm3        # 401218 <_IO_stdin_used+0x68>
  400f94:	00 
  400f95:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
  400f99:	76 24                	jbe    400fbf <_Z14codegen_updateP4Bodyd+0xdf>
  400f9b:	c5 fb 10 81 78 20 60 	vmovsd 0x602078(%rcx),%xmm0
  400fa2:	00 
  400fa3:	c5 fb 10 15 c5 02 00 	vmovsd 0x2c5(%rip),%xmm2        # 401270 <_IO_stdin_used+0xc0>
  400faa:	00 
  400fab:	c5 f9 57 c2          	vxorpd %xmm2,%xmm0,%xmm0
  400faf:	c5 fb 11 81 78 20 60 	vmovsd %xmm0,0x602078(%rcx)
  400fb6:	00 
  400fb7:	c5 fb 10 80 78 20 60 	vmovsd 0x602078(%rax),%xmm0
  400fbe:	00 
  400fbf:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
  400fc3:	77 0e                	ja     400fd3 <_Z14codegen_updateP4Bodyd+0xf3>
  400fc5:	c5 fb 10 0d 4b 02 00 	vmovsd 0x24b(%rip),%xmm1        # 401218 <_IO_stdin_used+0x68>
  400fcc:	00 
  400fcd:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
  400fd1:	76 1d                	jbe    400ff0 <_Z14codegen_updateP4Bodyd+0x110>
  400fd3:	c5 fb 10 82 78 20 60 	vmovsd 0x602078(%rdx),%xmm0
  400fda:	00 
  400fdb:	c5 fb 10 0d 8d 02 00 	vmovsd 0x28d(%rip),%xmm1        # 401270 <_IO_stdin_used+0xc0>
  400fe2:	00 
  400fe3:	c5 f9 57 c1          	vxorpd %xmm1,%xmm0,%xmm0
  400fe7:	c5 fb 11 82 78 20 60 	vmovsd %xmm0,0x602078(%rdx)
  400fee:	00 
  400fef:	c3                   	retq   
  400ff0:	c3                   	retq   
  400ff1:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400ff6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ffd:	00 00 00 

0000000000401000 <_Z26codegen_reset_force_manualP4Body>:
  401000:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  401004:	c5 fb 11 04 fd 88 02 	vmovsd %xmm0,0x650288(,%rdi,8)
  40100b:	65 00 
  40100d:	c5 fb 11 04 fd 88 fc 	vmovsd %xmm0,0x65fc88(,%rdi,8)
  401014:	65 00 
  401016:	c3                   	retq   
  401017:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40101e:	00 00 

0000000000401020 <_Z24codegen_reset_force_halfP4Body>:
  401020:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  401027:	00 00 00 00 00 
  40102c:	c3                   	retq   
  40102d:	0f 1f 00             	nopl   (%rax)

0000000000401030 <_Z18codegen_reset_massP4Body>:
  401030:	48 c7 04 fd 88 20 60 	movq   $0x0,0x602088(,%rdi,8)
  401037:	00 00 00 00 00 
  40103c:	c3                   	retq   
  40103d:	0f 1f 00             	nopl   (%rax)

0000000000401040 <_Z22codengen_simple_updateP4Bodyd>:
  401040:	c5 fb 59 34 fd 88 02 	vmulsd 0x650288(,%rdi,8),%xmm0,%xmm6
  401047:	65 00 
  401049:	c5 fb 59 2c fd 88 fc 	vmulsd 0x65fc88(,%rdi,8),%xmm0,%xmm5
  401050:	65 00 
  401052:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  401059:	00 
  40105a:	c5 fb 10 0c fd 88 20 	vmovsd 0x602088(,%rdi,8),%xmm1
  401061:	60 00 
  401063:	4c 8d 80 10 ee 02 00 	lea    0x2ee10(%rax),%r8
  40106a:	c5 fb 10 a0 88 1a 61 	vmovsd 0x611a88(%rax),%xmm4
  401071:	00 
  401072:	48 8d b0 10 e8 03 00 	lea    0x3e810(%rax),%rsi
  401079:	c4 c1 7b 10 90 78 20 	vmovsd 0x602078(%r8),%xmm2
  401080:	60 00 
  401082:	48 8d 88 10 fa 00 00 	lea    0xfa10(%rax),%rcx
  401089:	c5 fb 10 98 88 14 62 	vmovsd 0x621488(%rax),%xmm3
  401090:	00 
  401091:	48 8d 90 10 f4 01 00 	lea    0x1f410(%rax),%rdx
  401098:	b8 64 00 00 00       	mov    $0x64,%eax
  40109d:	c5 cb 5e f1          	vdivsd %xmm1,%xmm6,%xmm6
  4010a1:	c5 d3 5e e9          	vdivsd %xmm1,%xmm5,%xmm5
  4010a5:	c5 fb 10 8e 78 20 60 	vmovsd 0x602078(%rsi),%xmm1
  4010ac:	00 
  4010ad:	0f 1f 00             	nopl   (%rax)
  4010b0:	c5 eb 58 d6          	vaddsd %xmm6,%xmm2,%xmm2
  4010b4:	c5 f3 58 cd          	vaddsd %xmm5,%xmm1,%xmm1
  4010b8:	c4 e2 f9 b9 e2       	vfmadd231sd %xmm2,%xmm0,%xmm4
  4010bd:	c4 e2 f9 b9 d9       	vfmadd231sd %xmm1,%xmm0,%xmm3
  4010c2:	83 e8 01             	sub    $0x1,%eax
  4010c5:	75 e9                	jne    4010b0 <_Z22codengen_simple_updateP4Bodyd+0x70>
  4010c7:	c4 c1 7b 11 90 78 20 	vmovsd %xmm2,0x602078(%r8)
  4010ce:	60 00 
  4010d0:	c5 fb 11 8e 78 20 60 	vmovsd %xmm1,0x602078(%rsi)
  4010d7:	00 
  4010d8:	c5 fb 11 a1 78 20 60 	vmovsd %xmm4,0x602078(%rcx)
  4010df:	00 
  4010e0:	c5 fb 11 9a 78 20 60 	vmovsd %xmm3,0x602078(%rdx)
  4010e7:	00 
  4010e8:	c3                   	retq   
  4010e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004010f0 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  4010f0:	55                   	push   %rbp
  4010f1:	48 89 fd             	mov    %rdi,%rbp
  4010f4:	53                   	push   %rbx
  4010f5:	48 83 ec 08          	sub    $0x8,%rsp
  4010f9:	e8 52 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4010fe:	48 89 c3             	mov    %rax,%rbx
  401101:	ff d5                	callq  *%rbp
  401103:	e8 48 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401108:	48 83 c4 08          	add    $0x8,%rsp
  40110c:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  401113:	de 1b 43 
  401116:	48 29 d8             	sub    %rbx,%rax
  401119:	5b                   	pop    %rbx
  40111a:	48 89 c1             	mov    %rax,%rcx
  40111d:	5d                   	pop    %rbp
  40111e:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401122:	48 f7 ea             	imul   %rdx
  401125:	48 89 d0             	mov    %rdx,%rax
  401128:	48 c1 f8 12          	sar    $0x12,%rax
  40112c:	48 29 c8             	sub    %rcx,%rax
  40112f:	c3                   	retq   

0000000000401130 <__libc_csu_init>:
  401130:	41 57                	push   %r15
  401132:	41 56                	push   %r14
  401134:	41 89 ff             	mov    %edi,%r15d
  401137:	41 55                	push   %r13
  401139:	41 54                	push   %r12
  40113b:	4c 8d 25 ae 0c 20 00 	lea    0x200cae(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  401142:	55                   	push   %rbp
  401143:	48 8d 2d be 0c 20 00 	lea    0x200cbe(%rip),%rbp        # 601e08 <__init_array_end>
  40114a:	53                   	push   %rbx
  40114b:	49 89 f6             	mov    %rsi,%r14
  40114e:	49 89 d5             	mov    %rdx,%r13
  401151:	4c 29 e5             	sub    %r12,%rbp
  401154:	48 83 ec 08          	sub    $0x8,%rsp
  401158:	48 c1 fd 03          	sar    $0x3,%rbp
  40115c:	e8 4f f5 ff ff       	callq  4006b0 <_init>
  401161:	48 85 ed             	test   %rbp,%rbp
  401164:	74 20                	je     401186 <__libc_csu_init+0x56>
  401166:	31 db                	xor    %ebx,%ebx
  401168:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40116f:	00 
  401170:	4c 89 ea             	mov    %r13,%rdx
  401173:	4c 89 f6             	mov    %r14,%rsi
  401176:	44 89 ff             	mov    %r15d,%edi
  401179:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40117d:	48 83 c3 01          	add    $0x1,%rbx
  401181:	48 39 eb             	cmp    %rbp,%rbx
  401184:	75 ea                	jne    401170 <__libc_csu_init+0x40>
  401186:	48 83 c4 08          	add    $0x8,%rsp
  40118a:	5b                   	pop    %rbx
  40118b:	5d                   	pop    %rbp
  40118c:	41 5c                	pop    %r12
  40118e:	41 5d                	pop    %r13
  401190:	41 5e                	pop    %r14
  401192:	41 5f                	pop    %r15
  401194:	c3                   	retq   
  401195:	90                   	nop
  401196:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40119d:	00 00 00 

00000000004011a0 <__libc_csu_fini>:
  4011a0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004011a4 <_fini>:
  4011a4:	48 83 ec 08          	sub    $0x8,%rsp
  4011a8:	48 83 c4 08          	add    $0x8,%rsp
  4011ac:	c3                   	retq   
