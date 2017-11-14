
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
  400782:	41 55                	push   %r13
  400784:	bf f0 09 40 00       	mov    $0x4009f0,%edi
  400789:	41 54                	push   %r12
  40078b:	55                   	push   %rbp
  40078c:	41 bd 90 1a 61 00    	mov    $0x611a90,%r13d
  400792:	53                   	push   %rbx
  400793:	45 31 e4             	xor    %r12d,%r12d
  400796:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  40079c:	48 83 ec 10          	sub    $0x10,%rsp
  4007a0:	48 c7 05 cd 18 20 00 	movq   $0x0,0x2018cd(%rip)        # 602078 <__Bodydata_buffer>
  4007a7:	00 00 00 00 
  4007ab:	48 c7 05 ca 18 20 00 	movq   $0x0,0x2018ca(%rip)        # 602080 <__Bodydata_buffer+0x8>
  4007b2:	00 00 00 00 
  4007b6:	e8 45 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007bb:	bf 00 0c 40 00       	mov    $0x400c00,%edi
  4007c0:	48 89 c3             	mov    %rax,%rbx
  4007c3:	e8 38 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007c8:	bf 40 0b 40 00       	mov    $0x400b40,%edi
  4007cd:	48 89 c5             	mov    %rax,%rbp
  4007d0:	e8 2b 09 00 00       	callq  401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  4007d5:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  4007db:	49 89 c0             	mov    %rax,%r8
  4007de:	66 90                	xchg   %ax,%ax
  4007e0:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  4007e4:	c4 c1 7b 10 45 00    	vmovsd 0x0(%r13),%xmm0
  4007ea:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  4007ee:	c5 eb 5a d0          	vcvtsd2ss %xmm0,%xmm2,%xmm2
  4007f2:	c4 c1 7b 10 8d 00 fa 	vmovsd 0xfa00(%r13),%xmm1
  4007f9:	00 00 
  4007fb:	c5 f9 7e d6          	vmovd  %xmm2,%esi
  4007ff:	c5 e3 5a d9          	vcvtsd2ss %xmm1,%xmm3,%xmm3
  400803:	41 01 f1             	add    %esi,%r9d
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
  40082a:	49 83 fc 09          	cmp    $0x9,%r12
  40082e:	41 89 f1             	mov    %esi,%r9d
  400831:	76 37                	jbe    40086a <main+0xea>
  400833:	49 83 c4 01          	add    $0x1,%r12
  400837:	49 83 c5 08          	add    $0x8,%r13
  40083b:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  400842:	75 9c                	jne    4007e0 <main+0x60>
  400844:	48 89 e9             	mov    %rbp,%rcx
  400847:	48 89 da             	mov    %rbx,%rdx
  40084a:	be e0 11 40 00       	mov    $0x4011e0,%esi
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
  400871:	be c4 11 40 00       	mov    $0x4011c4,%esi
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
  4008a4:	bf c0 f6 66 00       	mov    $0x66f6c0,%edi
  4008a9:	e8 52 fe ff ff       	callq  400700 <_ZNSt8ios_base4InitC1Ev@plt>
  4008ae:	ba 68 20 60 00       	mov    $0x602068,%edx
  4008b3:	be c0 f6 66 00       	mov    $0x66f6c0,%esi
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
  4008ff:	49 c7 c0 b0 11 40 00 	mov    $0x4011b0,%r8
  400906:	48 c7 c1 40 11 40 00 	mov    $0x401140,%rcx
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
  4009f2:	55                   	push   %rbp
  4009f3:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4009f8:	53                   	push   %rbx
  4009f9:	41 bc 40 1f 00 00    	mov    $0x1f40,%r12d
  4009ff:	48 83 ec 10          	sub    $0x10,%rsp
  400a03:	e8 38 fd ff ff       	callq  400740 <srand@plt>
  400a08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a0f:	00 
  400a10:	e8 4b fd ff ff       	callq  400760 <rand@plt>
  400a15:	89 c5                	mov    %eax,%ebp
  400a17:	e8 44 fd ff ff       	callq  400760 <rand@plt>
  400a1c:	c5 e9 57 d2          	vxorpd %xmm2,%xmm2,%xmm2
  400a20:	c5 fb 10 1d 00 08 00 	vmovsd 0x800(%rip),%xmm3        # 401228 <_IO_stdin_used+0x68>
  400a27:	00 
  400a28:	c5 eb 2a d0          	vcvtsi2sd %eax,%xmm2,%xmm2
  400a2c:	c4 e2 e1 99 15 eb 07 	vfmadd132sd 0x7eb(%rip),%xmm3,%xmm2        # 401220 <_IO_stdin_used+0x60>
  400a33:	00 00 
  400a35:	c5 fb 11 54 24 08    	vmovsd %xmm2,0x8(%rsp)
  400a3b:	e8 20 fd ff ff       	callq  400760 <rand@plt>
  400a40:	c5 f1 57 c9          	vxorpd %xmm1,%xmm1,%xmm1
  400a44:	c5 fb 10 25 dc 07 00 	vmovsd 0x7dc(%rip),%xmm4        # 401228 <_IO_stdin_used+0x68>
  400a4b:	00 
  400a4c:	c5 f3 2a c8          	vcvtsi2sd %eax,%xmm1,%xmm1
  400a50:	c4 e2 d9 99 0d c7 07 	vfmadd132sd 0x7c7(%rip),%xmm4,%xmm1        # 401220 <_IO_stdin_used+0x60>
  400a57:	00 00 
  400a59:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  400a5e:	e8 fd fc ff ff       	callq  400760 <rand@plt>
  400a63:	89 c3                	mov    %eax,%ebx
  400a65:	e8 f6 fc ff ff       	callq  400760 <rand@plt>
  400a6a:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a6e:	48 8b 15 03 16 20 00 	mov    0x201603(%rip),%rdx        # 602078 <__Bodydata_buffer>
  400a75:	c5 fb 10 2d bb 07 00 	vmovsd 0x7bb(%rip),%xmm5        # 401238 <_IO_stdin_used+0x78>
  400a7c:	00 
  400a7d:	c5 fb 10 35 cb 07 00 	vmovsd 0x7cb(%rip),%xmm6        # 401250 <_IO_stdin_used+0x90>
  400a84:	00 
  400a85:	c5 fb 2a c5          	vcvtsi2sd %ebp,%xmm0,%xmm0
  400a89:	c4 e2 d1 99 05 9e 07 	vfmadd132sd 0x79e(%rip),%xmm5,%xmm0        # 401230 <_IO_stdin_used+0x70>
  400a90:	00 00 
  400a92:	c5 fb 59 05 a6 07 00 	vmulsd 0x7a6(%rip),%xmm0,%xmm0        # 401240 <_IO_stdin_used+0x80>
  400a99:	00 
  400a9a:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  400a9e:	48 83 c2 03          	add    $0x3,%rdx
  400aa2:	41 83 ec 01          	sub    $0x1,%r12d
  400aa6:	c5 fb 10 54 24 08    	vmovsd 0x8(%rsp),%xmm2
  400aac:	48 89 0d c5 15 20 00 	mov    %rcx,0x2015c5(%rip)        # 602078 <__Bodydata_buffer>
  400ab3:	c5 fb 10 0c 24       	vmovsd (%rsp),%xmm1
  400ab8:	c5 fb 11 14 d5 78 1a 	vmovsd %xmm2,0x611a78(,%rdx,8)
  400abf:	61 00 
  400ac1:	48 c7 04 d5 78 02 65 	movq   $0x0,0x650278(,%rdx,8)
  400ac8:	00 00 00 00 00 
  400acd:	48 c7 04 d5 78 fc 65 	movq   $0x0,0x65fc78(,%rdx,8)
  400ad4:	00 00 00 00 00 
  400ad9:	c5 fb 11 0c d5 78 14 	vmovsd %xmm1,0x621478(,%rdx,8)
  400ae0:	62 00 
  400ae2:	c5 fb 11 04 d5 78 20 	vmovsd %xmm0,0x602078(,%rdx,8)
  400ae9:	60 00 
  400aeb:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400aef:	c5 fb 2a c3          	vcvtsi2sd %ebx,%xmm0,%xmm0
  400af3:	c4 e2 c9 99 05 4c 07 	vfmadd132sd 0x74c(%rip),%xmm6,%xmm0        # 401248 <_IO_stdin_used+0x88>
  400afa:	00 00 
  400afc:	c5 fb 59 05 54 07 00 	vmulsd 0x754(%rip),%xmm0,%xmm0        # 401258 <_IO_stdin_used+0x98>
  400b03:	00 
  400b04:	c5 fb 11 04 d5 78 0e 	vmovsd %xmm0,0x630e78(,%rdx,8)
  400b0b:	63 00 
  400b0d:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400b11:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400b15:	c4 e2 c9 99 05 2a 07 	vfmadd132sd 0x72a(%rip),%xmm6,%xmm0        # 401248 <_IO_stdin_used+0x88>
  400b1c:	00 00 
  400b1e:	c5 fb 59 05 32 07 00 	vmulsd 0x732(%rip),%xmm0,%xmm0        # 401258 <_IO_stdin_used+0x98>
  400b25:	00 
  400b26:	c5 fb 11 04 d5 78 08 	vmovsd %xmm0,0x640878(,%rdx,8)
  400b2d:	64 00 
  400b2f:	0f 85 db fe ff ff    	jne    400a10 <_Z13instantiationv+0x20>
  400b35:	48 83 c4 10          	add    $0x10,%rsp
  400b39:	5b                   	pop    %rbx
  400b3a:	5d                   	pop    %rbp
  400b3b:	41 5c                	pop    %r12
  400b3d:	c3                   	retq   
  400b3e:	66 90                	xchg   %ax,%ax

0000000000400b40 <_Z10run_simplev>:
  400b40:	c5 fb 10 15 f0 06 00 	vmovsd 0x6f0(%rip),%xmm2        # 401238 <_IO_stdin_used+0x78>
  400b47:	00 
  400b48:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400b4d:	ba 90 20 60 00       	mov    $0x602090,%edx
  400b52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b58:	c5 eb 59 82 00 dc 05 	vmulsd 0x5dc00(%rdx),%xmm2,%xmm0
  400b5f:	00 
  400b60:	b8 64 00 00 00       	mov    $0x64,%eax
  400b65:	c5 eb 59 b2 00 e2 04 	vmulsd 0x4e200(%rdx),%xmm2,%xmm6
  400b6c:	00 
  400b6d:	c5 fb 10 2a          	vmovsd (%rdx),%xmm5
  400b71:	c5 fb 10 8a 00 ee 02 	vmovsd 0x2ee00(%rdx),%xmm1
  400b78:	00 
  400b79:	c5 fb 10 9a 00 fa 00 	vmovsd 0xfa00(%rdx),%xmm3
  400b80:	00 
  400b81:	c5 fb 10 a2 00 f4 01 	vmovsd 0x1f400(%rdx),%xmm4
  400b88:	00 
  400b89:	c5 cb 5e f5          	vdivsd %xmm5,%xmm6,%xmm6
  400b8d:	c5 fb 5e ed          	vdivsd %xmm5,%xmm0,%xmm5
  400b91:	c5 fb 10 82 00 e8 03 	vmovsd 0x3e800(%rdx),%xmm0
  400b98:	00 
  400b99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  400ba0:	c5 f3 58 ce          	vaddsd %xmm6,%xmm1,%xmm1
  400ba4:	83 e8 01             	sub    $0x1,%eax
  400ba7:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
  400bab:	c4 e2 f1 b9 da       	vfmadd231sd %xmm2,%xmm1,%xmm3
  400bb0:	c4 e2 f9 b9 e2       	vfmadd231sd %xmm2,%xmm0,%xmm4
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
  400bf1:	f3 c3                	repz retq 
  400bf3:	0f 1f 00             	nopl   (%rax)
  400bf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bfd:	00 00 00 

0000000000400c00 <_Z14run_simulationv>:
  400c00:	53                   	push   %rbx
  400c01:	bb 05 00 00 00       	mov    $0x5,%ebx
  400c06:	48 83 ec 20          	sub    $0x20,%rsp
  400c0a:	c5 fb 10 35 26 06 00 	vmovsd 0x626(%rip),%xmm6        # 401238 <_IO_stdin_used+0x78>
  400c11:	00 
  400c12:	c5 fb 10 3d 0e 06 00 	vmovsd 0x60e(%rip),%xmm7        # 401228 <_IO_stdin_used+0x68>
  400c19:	00 
  400c1a:	c5 fb 10 2d 3e 06 00 	vmovsd 0x63e(%rip),%xmm5        # 401260 <_IO_stdin_used+0xa0>
  400c21:	00 
  400c22:	c5 fb 10 25 3e 06 00 	vmovsd 0x63e(%rip),%xmm4        # 401268 <_IO_stdin_used+0xa8>
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
  400c6d:	c5 fb 10 74 24 08    	vmovsd 0x8(%rsp),%xmm6
  400c73:	31 f6                	xor    %esi,%esi
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
  400ca9:	c5 d3 59 89 00 06 ff 	vmulsd -0xfa00(%rcx),%xmm5,%xmm1
  400cb0:	ff 
  400cb1:	c5 7b 5c 80 00 12 fd 	vsubsd -0x2ee00(%rax),%xmm0,%xmm8
  400cb8:	ff 
  400cb9:	c5 fb 10 11          	vmovsd (%rcx),%xmm2
  400cbd:	c5 eb 5c 90 00 18 fc 	vsubsd -0x3e800(%rax),%xmm2,%xmm2
  400cc4:	ff 
  400cc5:	c5 f3 59 88 00 1e fb 	vmulsd -0x4e200(%rax),%xmm1,%xmm1
  400ccc:	ff 
  400ccd:	c4 c1 3b 59 d8       	vmulsd %xmm8,%xmm8,%xmm3
  400cd2:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
  400cd7:	c5 e3 51 db          	vsqrtsd %xmm3,%xmm3,%xmm3
  400cdb:	c5 f9 28 c3          	vmovapd %xmm3,%xmm0
  400cdf:	c4 e2 d9 99 c3       	vfmadd132sd %xmm3,%xmm4,%xmm0
  400ce4:	c5 f3 5e c0          	vdivsd %xmm0,%xmm1,%xmm0
  400ce8:	c5 fb 59 d2          	vmulsd %xmm2,%xmm0,%xmm2
  400cec:	c4 c1 7b 59 c0       	vmulsd %xmm8,%xmm0,%xmm0
  400cf1:	c5 eb 5e d3          	vdivsd %xmm3,%xmm2,%xmm2
  400cf5:	c5 fb 5e c3          	vdivsd %xmm3,%xmm0,%xmm0
  400cf9:	c5 eb 58 10          	vaddsd (%rax),%xmm2,%xmm2
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
  400d33:	c5 fb 10 15 45 05 00 	vmovsd 0x545(%rip),%xmm2        # 401280 <_IO_stdin_used+0xc0>
  400d3a:	00 
  400d3b:	b8 90 0e 63 00       	mov    $0x630e90,%eax
  400d40:	c5 7b 10 05 28 05 00 	vmovsd 0x528(%rip),%xmm8        # 401270 <_IO_stdin_used+0xb0>
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
  400d75:	c4 c1 7b 5e c2       	vdivsd %xmm10,%xmm0,%xmm0
  400d7a:	c5 f3 58 08          	vaddsd (%rax),%xmm1,%xmm1
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

0000000000400e20 <_ZN4Body7storageEv>:
  400e20:	b8 78 20 60 00       	mov    $0x602078,%eax
  400e25:	c3                   	retq   
  400e26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2d:	00 00 00 

0000000000400e30 <_Z19codegen_reset_forceP4Body>:
  400e30:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400e34:	c5 fb 11 04 fd 88 02 	vmovsd %xmm0,0x650288(,%rdi,8)
  400e3b:	65 00 
  400e3d:	c5 fb 11 04 fd 88 fc 	vmovsd %xmm0,0x65fc88(,%rdi,8)
  400e44:	65 00 
  400e46:	c3                   	retq   
  400e47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e4e:	00 00 

0000000000400e50 <_Z17codegen_add_forceP4BodyS0_>:
  400e50:	48 39 f7             	cmp    %rsi,%rdi
  400e53:	0f 84 94 00 00 00    	je     400eed <_Z17codegen_add_forceP4BodyS0_+0x9d>
  400e59:	c5 fb 10 04 f5 88 14 	vmovsd 0x621488(,%rsi,8),%xmm0
  400e60:	62 00 
  400e62:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  400e69:	00 
  400e6a:	c5 fb 10 1c fd 88 20 	vmovsd 0x602088(,%rdi,8),%xmm3
  400e71:	60 00 
  400e73:	c5 fb 5c 04 fd 88 14 	vsubsd 0x621488(,%rdi,8),%xmm0,%xmm0
  400e7a:	62 00 
  400e7c:	c5 fb 10 0c f5 88 1a 	vmovsd 0x611a88(,%rsi,8),%xmm1
  400e83:	61 00 
  400e85:	c5 e3 59 1c f5 88 20 	vmulsd 0x602088(,%rsi,8),%xmm3,%xmm3
  400e8c:	60 00 
  400e8e:	c5 f3 5c 0c fd 88 1a 	vsubsd 0x611a88(,%rdi,8),%xmm1,%xmm1
  400e95:	61 00 
  400e97:	c5 fb 59 d0          	vmulsd %xmm0,%xmm0,%xmm2
  400e9b:	c5 e3 59 1d bd 03 00 	vmulsd 0x3bd(%rip),%xmm3,%xmm3        # 401260 <_IO_stdin_used+0xa0>
  400ea2:	00 
  400ea3:	c4 e2 f1 b9 d1       	vfmadd231sd %xmm1,%xmm1,%xmm2
  400ea8:	c5 eb 51 d2          	vsqrtsd %xmm2,%xmm2,%xmm2
  400eac:	c5 f9 28 e2          	vmovapd %xmm2,%xmm4
  400eb0:	c4 e2 e9 a9 25 af 03 	vfmadd213sd 0x3af(%rip),%xmm2,%xmm4        # 401268 <_IO_stdin_used+0xa8>
  400eb7:	00 00 
  400eb9:	c5 e3 5e dc          	vdivsd %xmm4,%xmm3,%xmm3
  400ebd:	c5 f3 59 cb          	vmulsd %xmm3,%xmm1,%xmm1
  400ec1:	c5 fb 59 c3          	vmulsd %xmm3,%xmm0,%xmm0
  400ec5:	c5 f3 5e ca          	vdivsd %xmm2,%xmm1,%xmm1
  400ec9:	c5 fb 5e c2          	vdivsd %xmm2,%xmm0,%xmm0
  400ecd:	c5 f3 58 88 88 02 65 	vaddsd 0x650288(%rax),%xmm1,%xmm1
  400ed4:	00 
  400ed5:	c5 fb 11 88 88 02 65 	vmovsd %xmm1,0x650288(%rax)
  400edc:	00 
  400edd:	c5 fb 58 80 88 fc 65 	vaddsd 0x65fc88(%rax),%xmm0,%xmm0
  400ee4:	00 
  400ee5:	c5 fb 11 80 88 fc 65 	vmovsd %xmm0,0x65fc88(%rax)
  400eec:	00 
  400eed:	f3 c3                	repz retq 
  400eef:	90                   	nop

0000000000400ef0 <_Z14codegen_updateP4Bodyd>:
  400ef0:	c5 fb 59 0c fd 88 02 	vmulsd 0x650288(,%rdi,8),%xmm0,%xmm1
  400ef7:	65 00 
  400ef9:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  400f00:	00 
  400f01:	48 8d 88 10 ee 02 00 	lea    0x2ee10(%rax),%rcx
  400f08:	48 8d 90 10 e8 03 00 	lea    0x3e810(%rax),%rdx
  400f0f:	48 8d b0 10 fa 00 00 	lea    0xfa10(%rax),%rsi
  400f16:	48 05 10 f4 01 00    	add    $0x1f410,%rax
  400f1c:	c5 f3 5e 88 78 2c 5e 	vdivsd 0x5e2c78(%rax),%xmm1,%xmm1
  400f23:	00 
  400f24:	c5 f3 58 89 78 20 60 	vaddsd 0x602078(%rcx),%xmm1,%xmm1
  400f2b:	00 
  400f2c:	c5 fb 11 89 78 20 60 	vmovsd %xmm1,0x602078(%rcx)
  400f33:	00 
  400f34:	c5 fb 59 0c fd 88 fc 	vmulsd 0x65fc88(,%rdi,8),%xmm0,%xmm1
  400f3b:	65 00 
  400f3d:	c5 f3 5e 88 78 2c 5e 	vdivsd 0x5e2c78(%rax),%xmm1,%xmm1
  400f44:	00 
  400f45:	c5 f3 58 8a 78 20 60 	vaddsd 0x602078(%rdx),%xmm1,%xmm1
  400f4c:	00 
  400f4d:	c5 fb 11 8a 78 20 60 	vmovsd %xmm1,0x602078(%rdx)
  400f54:	00 
  400f55:	c5 fb 10 89 78 20 60 	vmovsd 0x602078(%rcx),%xmm1
  400f5c:	00 
  400f5d:	c4 e2 f9 a9 88 78 26 	vfmadd213sd 0x5f2678(%rax),%xmm0,%xmm1
  400f64:	5f 00 
  400f66:	c5 fb 11 88 78 26 5f 	vmovsd %xmm1,0x5f2678(%rax)
  400f6d:	00 
  400f6e:	c5 fb 10 0d fa 02 00 	vmovsd 0x2fa(%rip),%xmm1        # 401270 <_IO_stdin_used+0xb0>
  400f75:	00 
  400f76:	c5 fb 10 a0 78 20 60 	vmovsd 0x602078(%rax),%xmm4
  400f7d:	00 
  400f7e:	c4 e2 d9 99 82 78 20 	vfmadd132sd 0x602078(%rdx),%xmm4,%xmm0
  400f85:	60 00 
  400f87:	c5 fb 11 80 78 20 60 	vmovsd %xmm0,0x602078(%rax)
  400f8e:	00 
  400f8f:	c5 fb 10 96 78 20 60 	vmovsd 0x602078(%rsi),%xmm2
  400f96:	00 
  400f97:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
  400f9b:	77 0e                	ja     400fab <_Z14codegen_updateP4Bodyd+0xbb>
  400f9d:	c5 fb 10 1d 83 02 00 	vmovsd 0x283(%rip),%xmm3        # 401228 <_IO_stdin_used+0x68>
  400fa4:	00 
  400fa5:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
  400fa9:	76 24                	jbe    400fcf <_Z14codegen_updateP4Bodyd+0xdf>
  400fab:	c5 fb 10 81 78 20 60 	vmovsd 0x602078(%rcx),%xmm0
  400fb2:	00 
  400fb3:	c5 fb 10 15 c5 02 00 	vmovsd 0x2c5(%rip),%xmm2        # 401280 <_IO_stdin_used+0xc0>
  400fba:	00 
  400fbb:	c5 f9 57 c2          	vxorpd %xmm2,%xmm0,%xmm0
  400fbf:	c5 fb 11 81 78 20 60 	vmovsd %xmm0,0x602078(%rcx)
  400fc6:	00 
  400fc7:	c5 fb 10 80 78 20 60 	vmovsd 0x602078(%rax),%xmm0
  400fce:	00 
  400fcf:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
  400fd3:	77 0e                	ja     400fe3 <_Z14codegen_updateP4Bodyd+0xf3>
  400fd5:	c5 fb 10 0d 4b 02 00 	vmovsd 0x24b(%rip),%xmm1        # 401228 <_IO_stdin_used+0x68>
  400fdc:	00 
  400fdd:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
  400fe1:	76 1d                	jbe    401000 <_Z14codegen_updateP4Bodyd+0x110>
  400fe3:	c5 fb 10 82 78 20 60 	vmovsd 0x602078(%rdx),%xmm0
  400fea:	00 
  400feb:	c5 fb 10 0d 8d 02 00 	vmovsd 0x28d(%rip),%xmm1        # 401280 <_IO_stdin_used+0xc0>
  400ff2:	00 
  400ff3:	c5 f9 57 c1          	vxorpd %xmm1,%xmm0,%xmm0
  400ff7:	c5 fb 11 82 78 20 60 	vmovsd %xmm0,0x602078(%rdx)
  400ffe:	00 
  400fff:	c3                   	retq   
  401000:	f3 c3                	repz retq 
  401002:	0f 1f 40 00          	nopl   0x0(%rax)
  401006:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40100d:	00 00 00 

0000000000401010 <_Z26codegen_reset_force_manualP4Body>:
  401010:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  401014:	c5 fb 11 04 fd 88 02 	vmovsd %xmm0,0x650288(,%rdi,8)
  40101b:	65 00 
  40101d:	c5 fb 11 04 fd 88 fc 	vmovsd %xmm0,0x65fc88(,%rdi,8)
  401024:	65 00 
  401026:	c3                   	retq   
  401027:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40102e:	00 00 

0000000000401030 <_Z24codegen_reset_force_halfP4Body>:
  401030:	48 c7 04 fd 88 02 65 	movq   $0x0,0x650288(,%rdi,8)
  401037:	00 00 00 00 00 
  40103c:	c3                   	retq   
  40103d:	0f 1f 00             	nopl   (%rax)

0000000000401040 <_Z18codegen_reset_massP4Body>:
  401040:	48 c7 04 fd 88 20 60 	movq   $0x0,0x602088(,%rdi,8)
  401047:	00 00 00 00 00 
  40104c:	c3                   	retq   
  40104d:	0f 1f 00             	nopl   (%rax)

0000000000401050 <_Z22codengen_simple_updateP4Bodyd>:
  401050:	c5 fb 59 0c fd 88 fc 	vmulsd 0x65fc88(,%rdi,8),%xmm0,%xmm1
  401057:	65 00 
  401059:	48 8d 04 fd 00 00 00 	lea    0x0(,%rdi,8),%rax
  401060:	00 
  401061:	c5 fb 59 34 fd 88 02 	vmulsd 0x650288(,%rdi,8),%xmm0,%xmm6
  401068:	65 00 
  40106a:	4c 8d 80 10 ee 02 00 	lea    0x2ee10(%rax),%r8
  401071:	48 8d b0 10 e8 03 00 	lea    0x3e810(%rax),%rsi
  401078:	48 8d 88 10 fa 00 00 	lea    0xfa10(%rax),%rcx
  40107f:	48 8d 90 10 f4 01 00 	lea    0x1f410(%rax),%rdx
  401086:	c5 fb 10 2c fd 88 20 	vmovsd 0x602088(,%rdi,8),%xmm5
  40108d:	60 00 
  40108f:	c4 c1 7b 10 90 78 20 	vmovsd 0x602078(%r8),%xmm2
  401096:	60 00 
  401098:	b8 64 00 00 00       	mov    $0x64,%eax
  40109d:	c5 fb 10 a1 78 20 60 	vmovsd 0x602078(%rcx),%xmm4
  4010a4:	00 
  4010a5:	c5 fb 10 9a 78 20 60 	vmovsd 0x602078(%rdx),%xmm3
  4010ac:	00 
  4010ad:	c5 cb 5e f5          	vdivsd %xmm5,%xmm6,%xmm6
  4010b1:	c5 f3 5e ed          	vdivsd %xmm5,%xmm1,%xmm5
  4010b5:	c5 fb 10 8e 78 20 60 	vmovsd 0x602078(%rsi),%xmm1
  4010bc:	00 
  4010bd:	0f 1f 00             	nopl   (%rax)
  4010c0:	c5 eb 58 d6          	vaddsd %xmm6,%xmm2,%xmm2
  4010c4:	83 e8 01             	sub    $0x1,%eax
  4010c7:	c5 f3 58 cd          	vaddsd %xmm5,%xmm1,%xmm1
  4010cb:	c4 e2 f9 b9 e2       	vfmadd231sd %xmm2,%xmm0,%xmm4
  4010d0:	c4 e2 f9 b9 d9       	vfmadd231sd %xmm1,%xmm0,%xmm3
  4010d5:	75 e9                	jne    4010c0 <_Z22codengen_simple_updateP4Bodyd+0x70>
  4010d7:	c4 c1 7b 11 90 78 20 	vmovsd %xmm2,0x602078(%r8)
  4010de:	60 00 
  4010e0:	c5 fb 11 8e 78 20 60 	vmovsd %xmm1,0x602078(%rsi)
  4010e7:	00 
  4010e8:	c5 fb 11 a1 78 20 60 	vmovsd %xmm4,0x602078(%rcx)
  4010ef:	00 
  4010f0:	c5 fb 11 9a 78 20 60 	vmovsd %xmm3,0x602078(%rdx)
  4010f7:	00 
  4010f8:	c3                   	retq   
  4010f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000401100 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  401100:	55                   	push   %rbp
  401101:	53                   	push   %rbx
  401102:	48 89 fd             	mov    %rdi,%rbp
  401105:	48 83 ec 08          	sub    $0x8,%rsp
  401109:	e8 42 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  40110e:	48 89 c3             	mov    %rax,%rbx
  401111:	ff d5                	callq  *%rbp
  401113:	e8 38 f6 ff ff       	callq  400750 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  401118:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  40111f:	de 1b 43 
  401122:	48 29 d8             	sub    %rbx,%rax
  401125:	48 83 c4 08          	add    $0x8,%rsp
  401129:	48 89 c1             	mov    %rax,%rcx
  40112c:	48 f7 ea             	imul   %rdx
  40112f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  401133:	5b                   	pop    %rbx
  401134:	5d                   	pop    %rbp
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
