
bin/g++_nbody_aos:     file format elf64-x86-64


Disassembly of section .init:

0000000000400678 <_init>:
  400678:	48 83 ec 08          	sub    $0x8,%rsp
  40067c:	48 8b 05 75 19 20 00 	mov    0x201975(%rip),%rax        # 601ff8 <_DYNAMIC+0x1e0>
  400683:	48 85 c0             	test   %rax,%rax
  400686:	74 05                	je     40068d <_init+0x15>
  400688:	e8 a3 00 00 00       	callq  400730 <rand@plt+0x10>
  40068d:	48 83 c4 08          	add    $0x8,%rsp
  400691:	c3                   	retq   

Disassembly of section .plt:

00000000004006a0 <__printf_chk@plt-0x10>:
  4006a0:	ff 35 62 19 20 00    	pushq  0x201962(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4006a6:	ff 25 64 19 20 00    	jmpq   *0x201964(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4006ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006b0 <__printf_chk@plt>:
  4006b0:	ff 25 62 19 20 00    	jmpq   *0x201962(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4006b6:	68 00 00 00 00       	pushq  $0x0
  4006bb:	e9 e0 ff ff ff       	jmpq   4006a0 <_init+0x28>

00000000004006c0 <_ZNSt8ios_base4InitC1Ev@plt>:
  4006c0:	ff 25 5a 19 20 00    	jmpq   *0x20195a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4006c6:	68 01 00 00 00       	pushq  $0x1
  4006cb:	e9 d0 ff ff ff       	jmpq   4006a0 <_init+0x28>

00000000004006d0 <__libc_start_main@plt>:
  4006d0:	ff 25 52 19 20 00    	jmpq   *0x201952(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4006d6:	68 02 00 00 00       	pushq  $0x2
  4006db:	e9 c0 ff ff ff       	jmpq   4006a0 <_init+0x28>

00000000004006e0 <__cxa_atexit@plt>:
  4006e0:	ff 25 4a 19 20 00    	jmpq   *0x20194a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4006e6:	68 03 00 00 00       	pushq  $0x3
  4006eb:	e9 b0 ff ff ff       	jmpq   4006a0 <_init+0x28>

00000000004006f0 <_ZNSt8ios_base4InitD1Ev@plt>:
  4006f0:	ff 25 42 19 20 00    	jmpq   *0x201942(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  4006f6:	68 04 00 00 00       	pushq  $0x4
  4006fb:	e9 a0 ff ff ff       	jmpq   4006a0 <_init+0x28>

0000000000400700 <srand@plt>:
  400700:	ff 25 3a 19 20 00    	jmpq   *0x20193a(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400706:	68 05 00 00 00       	pushq  $0x5
  40070b:	e9 90 ff ff ff       	jmpq   4006a0 <_init+0x28>

0000000000400710 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
  400710:	ff 25 32 19 20 00    	jmpq   *0x201932(%rip)        # 602048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400716:	68 06 00 00 00       	pushq  $0x6
  40071b:	e9 80 ff ff ff       	jmpq   4006a0 <_init+0x28>

0000000000400720 <rand@plt>:
  400720:	ff 25 2a 19 20 00    	jmpq   *0x20192a(%rip)        # 602050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400726:	68 07 00 00 00       	pushq  $0x7
  40072b:	e9 70 ff ff ff       	jmpq   4006a0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400730 <.plt.got>:
  400730:	ff 25 c2 18 20 00    	jmpq   *0x2018c2(%rip)        # 601ff8 <_DYNAMIC+0x1e0>
  400736:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400740 <main>:
  400740:	41 56                	push   %r14
  400742:	41 55                	push   %r13
  400744:	bf a0 09 40 00       	mov    $0x4009a0,%edi
  400749:	41 54                	push   %r12
  40074b:	55                   	push   %rbp
  40074c:	41 bd a8 20 60 00    	mov    $0x6020a8,%r13d
  400752:	53                   	push   %rbx
  400753:	45 31 e4             	xor    %r12d,%r12d
  400756:	41 be 8d 68 b7 6c    	mov    $0x6cb7688d,%r14d
  40075c:	48 83 ec 10          	sub    $0x10,%rsp
  400760:	e8 bb 05 00 00       	callq  400d20 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400765:	bf 80 0b 40 00       	mov    $0x400b80,%edi
  40076a:	48 89 c3             	mov    %rax,%rbx
  40076d:	e8 ae 05 00 00       	callq  400d20 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  400772:	bf f0 0a 40 00       	mov    $0x400af0,%edi
  400777:	48 89 c5             	mov    %rax,%rbp
  40077a:	e8 a1 05 00 00       	callq  400d20 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>
  40077f:	41 b9 0b 00 00 00    	mov    $0xb,%r9d
  400785:	49 89 c0             	mov    %rax,%r8
  400788:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40078f:	00 
  400790:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  400794:	c4 c1 7b 10 45 00    	vmovsd 0x0(%r13),%xmm0
  40079a:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  40079e:	c5 eb 5a d0          	vcvtsd2ss %xmm0,%xmm2,%xmm2
  4007a2:	c4 c1 7b 10 4d 08    	vmovsd 0x8(%r13),%xmm1
  4007a8:	c5 f9 7e d6          	vmovd  %xmm2,%esi
  4007ac:	c5 e3 5a d9          	vcvtsd2ss %xmm1,%xmm3,%xmm3
  4007b0:	41 01 f1             	add    %esi,%r9d
  4007b3:	c5 f9 7e de          	vmovd  %xmm3,%esi
  4007b7:	44 01 ce             	add    %r9d,%esi
  4007ba:	89 f0                	mov    %esi,%eax
  4007bc:	41 f7 ee             	imul   %r14d
  4007bf:	89 f0                	mov    %esi,%eax
  4007c1:	c1 f8 1f             	sar    $0x1f,%eax
  4007c4:	c1 fa 13             	sar    $0x13,%edx
  4007c7:	41 89 d1             	mov    %edx,%r9d
  4007ca:	41 29 c1             	sub    %eax,%r9d
  4007cd:	45 69 c9 87 d6 12 00 	imul   $0x12d687,%r9d,%r9d
  4007d4:	44 29 ce             	sub    %r9d,%esi
  4007d7:	49 83 fc 09          	cmp    $0x9,%r12
  4007db:	41 89 f1             	mov    %esi,%r9d
  4007de:	76 37                	jbe    400817 <main+0xd7>
  4007e0:	49 83 c4 01          	add    $0x1,%r12
  4007e4:	49 83 c5 38          	add    $0x38,%r13
  4007e8:	49 81 fc 40 1f 00 00 	cmp    $0x1f40,%r12
  4007ef:	75 9f                	jne    400790 <main+0x50>
  4007f1:	48 89 e9             	mov    %rbp,%rcx
  4007f4:	48 89 da             	mov    %rbx,%rdx
  4007f7:	be 00 0e 40 00       	mov    $0x400e00,%esi
  4007fc:	bf 01 00 00 00       	mov    $0x1,%edi
  400801:	31 c0                	xor    %eax,%eax
  400803:	e8 a8 fe ff ff       	callq  4006b0 <__printf_chk@plt>
  400808:	48 83 c4 10          	add    $0x10,%rsp
  40080c:	31 c0                	xor    %eax,%eax
  40080e:	5b                   	pop    %rbx
  40080f:	5d                   	pop    %rbp
  400810:	41 5c                	pop    %r12
  400812:	41 5d                	pop    %r13
  400814:	41 5e                	pop    %r14
  400816:	c3                   	retq   
  400817:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  40081b:	4c 89 e2             	mov    %r12,%rdx
  40081e:	be e4 0d 40 00       	mov    $0x400de4,%esi
  400823:	bf 01 00 00 00       	mov    $0x1,%edi
  400828:	b8 02 00 00 00       	mov    $0x2,%eax
  40082d:	4c 89 04 24          	mov    %r8,(%rsp)
  400831:	e8 7a fe ff ff       	callq  4006b0 <__printf_chk@plt>
  400836:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  40083b:	4c 8b 04 24          	mov    (%rsp),%r8
  40083f:	eb 9f                	jmp    4007e0 <main+0xa0>
  400841:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  400846:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40084d:	00 00 00 

0000000000400850 <_GLOBAL__sub_I__Z11r_float2intf>:
  400850:	48 83 ec 08          	sub    $0x8,%rsp
  400854:	bf a0 f0 67 00       	mov    $0x67f0a0,%edi
  400859:	e8 62 fe ff ff       	callq  4006c0 <_ZNSt8ios_base4InitC1Ev@plt>
  40085e:	ba 60 20 60 00       	mov    $0x602060,%edx
  400863:	be a0 f0 67 00       	mov    $0x67f0a0,%esi
  400868:	bf f0 06 40 00       	mov    $0x4006f0,%edi
  40086d:	48 83 c4 08          	add    $0x8,%rsp
  400871:	e9 6a fe ff ff       	jmpq   4006e0 <__cxa_atexit@plt>
  400876:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40087d:	00 00 00 

0000000000400880 <set_fast_math>:
  400880:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
  400885:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
  40088c:	00 
  40088d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
  400892:	c3                   	retq   
  400893:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40089a:	00 00 00 
  40089d:	0f 1f 00             	nopl   (%rax)

00000000004008a0 <_start>:
  4008a0:	31 ed                	xor    %ebp,%ebp
  4008a2:	49 89 d1             	mov    %rdx,%r9
  4008a5:	5e                   	pop    %rsi
  4008a6:	48 89 e2             	mov    %rsp,%rdx
  4008a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4008ad:	50                   	push   %rax
  4008ae:	54                   	push   %rsp
  4008af:	49 c7 c0 d0 0d 40 00 	mov    $0x400dd0,%r8
  4008b6:	48 c7 c1 60 0d 40 00 	mov    $0x400d60,%rcx
  4008bd:	48 c7 c7 40 07 40 00 	mov    $0x400740,%rdi
  4008c4:	e8 07 fe ff ff       	callq  4006d0 <__libc_start_main@plt>
  4008c9:	f4                   	hlt    
  4008ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008d0 <deregister_tm_clones>:
  4008d0:	b8 77 20 60 00       	mov    $0x602077,%eax
  4008d5:	55                   	push   %rbp
  4008d6:	48 2d 70 20 60 00    	sub    $0x602070,%rax
  4008dc:	48 83 f8 0e          	cmp    $0xe,%rax
  4008e0:	48 89 e5             	mov    %rsp,%rbp
  4008e3:	76 1b                	jbe    400900 <deregister_tm_clones+0x30>
  4008e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008ea:	48 85 c0             	test   %rax,%rax
  4008ed:	74 11                	je     400900 <deregister_tm_clones+0x30>
  4008ef:	5d                   	pop    %rbp
  4008f0:	bf 70 20 60 00       	mov    $0x602070,%edi
  4008f5:	ff e0                	jmpq   *%rax
  4008f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008fe:	00 00 
  400900:	5d                   	pop    %rbp
  400901:	c3                   	retq   
  400902:	0f 1f 40 00          	nopl   0x0(%rax)
  400906:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40090d:	00 00 00 

0000000000400910 <register_tm_clones>:
  400910:	be 70 20 60 00       	mov    $0x602070,%esi
  400915:	55                   	push   %rbp
  400916:	48 81 ee 70 20 60 00 	sub    $0x602070,%rsi
  40091d:	48 c1 fe 03          	sar    $0x3,%rsi
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 89 f0             	mov    %rsi,%rax
  400927:	48 c1 e8 3f          	shr    $0x3f,%rax
  40092b:	48 01 c6             	add    %rax,%rsi
  40092e:	48 d1 fe             	sar    %rsi
  400931:	74 15                	je     400948 <register_tm_clones+0x38>
  400933:	b8 00 00 00 00       	mov    $0x0,%eax
  400938:	48 85 c0             	test   %rax,%rax
  40093b:	74 0b                	je     400948 <register_tm_clones+0x38>
  40093d:	5d                   	pop    %rbp
  40093e:	bf 70 20 60 00       	mov    $0x602070,%edi
  400943:	ff e0                	jmpq   *%rax
  400945:	0f 1f 00             	nopl   (%rax)
  400948:	5d                   	pop    %rbp
  400949:	c3                   	retq   
  40094a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400950 <__do_global_dtors_aux>:
  400950:	80 3d 29 17 20 00 00 	cmpb   $0x0,0x201729(%rip)        # 602080 <completed.7585>
  400957:	75 11                	jne    40096a <__do_global_dtors_aux+0x1a>
  400959:	55                   	push   %rbp
  40095a:	48 89 e5             	mov    %rsp,%rbp
  40095d:	e8 6e ff ff ff       	callq  4008d0 <deregister_tm_clones>
  400962:	5d                   	pop    %rbp
  400963:	c6 05 16 17 20 00 01 	movb   $0x1,0x201716(%rip)        # 602080 <completed.7585>
  40096a:	f3 c3                	repz retq 
  40096c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400970 <frame_dummy>:
  400970:	bf 10 1e 60 00       	mov    $0x601e10,%edi
  400975:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400979:	75 05                	jne    400980 <frame_dummy+0x10>
  40097b:	eb 93                	jmp    400910 <register_tm_clones>
  40097d:	0f 1f 00             	nopl   (%rax)
  400980:	b8 00 00 00 00       	mov    $0x0,%eax
  400985:	48 85 c0             	test   %rax,%rax
  400988:	74 f1                	je     40097b <frame_dummy+0xb>
  40098a:	55                   	push   %rbp
  40098b:	48 89 e5             	mov    %rsp,%rbp
  40098e:	ff d0                	callq  *%rax
  400990:	5d                   	pop    %rbp
  400991:	e9 7a ff ff ff       	jmpq   400910 <register_tm_clones>
  400996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099d:	00 00 00 

00000000004009a0 <_Z13instantiationv>:
  4009a0:	53                   	push   %rbx
  4009a1:	bf 2a 00 00 00       	mov    $0x2a,%edi
  4009a6:	bb 40 1f 00 00       	mov    $0x1f40,%ebx
  4009ab:	48 83 ec 20          	sub    $0x20,%rsp
  4009af:	e8 4c fd ff ff       	callq  400700 <srand@plt>
  4009b4:	0f 1f 40 00          	nopl   0x0(%rax)
  4009b8:	e8 63 fd ff ff       	callq  400720 <rand@plt>
  4009bd:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  4009c1:	c5 fb 10 1d 7f 04 00 	vmovsd 0x47f(%rip),%xmm3        # 400e48 <_IO_stdin_used+0x68>
  4009c8:	00 
  4009c9:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  4009cd:	c4 e2 e1 99 05 6a 04 	vfmadd132sd 0x46a(%rip),%xmm3,%xmm0        # 400e40 <_IO_stdin_used+0x60>
  4009d4:	00 00 
  4009d6:	c5 fb 59 0d 72 04 00 	vmulsd 0x472(%rip),%xmm0,%xmm1        # 400e50 <_IO_stdin_used+0x70>
  4009dd:	00 
  4009de:	c5 fb 11 0c 24       	vmovsd %xmm1,(%rsp)
  4009e3:	e8 38 fd ff ff       	callq  400720 <rand@plt>
  4009e8:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  4009ec:	c5 fb 10 25 6c 04 00 	vmovsd 0x46c(%rip),%xmm4        # 400e60 <_IO_stdin_used+0x80>
  4009f3:	00 
  4009f4:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  4009f8:	c4 e2 d9 99 05 57 04 	vfmadd132sd 0x457(%rip),%xmm4,%xmm0        # 400e58 <_IO_stdin_used+0x78>
  4009ff:	00 00 
  400a01:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
  400a07:	e8 14 fd ff ff       	callq  400720 <rand@plt>
  400a0c:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a10:	c5 fb 10 2d 48 04 00 	vmovsd 0x448(%rip),%xmm5        # 400e60 <_IO_stdin_used+0x80>
  400a17:	00 
  400a18:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400a1c:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
  400a20:	c4 e2 d1 99 15 2f 04 	vfmadd132sd 0x42f(%rip),%xmm5,%xmm2        # 400e58 <_IO_stdin_used+0x78>
  400a27:	00 00 
  400a29:	c5 fb 11 54 24 10    	vmovsd %xmm2,0x10(%rsp)
  400a2f:	e8 ec fc ff ff       	callq  400720 <rand@plt>
  400a34:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a38:	c5 fb 10 35 30 04 00 	vmovsd 0x430(%rip),%xmm6        # 400e70 <_IO_stdin_used+0x90>
  400a3f:	00 
  400a40:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400a44:	c4 e2 c9 99 05 1b 04 	vfmadd132sd 0x41b(%rip),%xmm6,%xmm0        # 400e68 <_IO_stdin_used+0x88>
  400a4b:	00 00 
  400a4d:	c5 fb 59 25 23 04 00 	vmulsd 0x423(%rip),%xmm0,%xmm4        # 400e78 <_IO_stdin_used+0x98>
  400a54:	00 
  400a55:	c5 fb 11 64 24 18    	vmovsd %xmm4,0x18(%rsp)
  400a5b:	e8 c0 fc ff ff       	callq  400720 <rand@plt>
  400a60:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
  400a64:	c5 fb 10 3d 04 04 00 	vmovsd 0x404(%rip),%xmm7        # 400e70 <_IO_stdin_used+0x90>
  400a6b:	00 
  400a6c:	c5 fb 2a c0          	vcvtsi2sd %eax,%xmm0,%xmm0
  400a70:	48 8b 05 f1 15 20 00 	mov    0x2015f1(%rip),%rax        # 602068 <Body_arena_head>
  400a77:	c4 e2 c1 99 05 e8 03 	vfmadd132sd 0x3e8(%rip),%xmm7,%xmm0        # 400e68 <_IO_stdin_used+0x88>
  400a7e:	00 00 
  400a80:	c5 fb 59 05 f0 03 00 	vmulsd 0x3f0(%rip),%xmm0,%xmm0        # 400e78 <_IO_stdin_used+0x98>
  400a87:	00 
  400a88:	48 85 c0             	test   %rax,%rax
  400a8b:	74 3f                	je     400acc <_Z13instantiationv+0x12c>
  400a8d:	c5 fb 10 1c 24       	vmovsd (%rsp),%xmm3
  400a92:	48 c7 40 28 00 00 00 	movq   $0x0,0x28(%rax)
  400a99:	00 
  400a9a:	c5 fb 10 4c 24 08    	vmovsd 0x8(%rsp),%xmm1
  400aa0:	48 c7 40 30 00 00 00 	movq   $0x0,0x30(%rax)
  400aa7:	00 
  400aa8:	c5 fb 10 54 24 10    	vmovsd 0x10(%rsp),%xmm2
  400aae:	c5 fb 10 64 24 18    	vmovsd 0x18(%rsp),%xmm4
  400ab4:	c5 fb 11 18          	vmovsd %xmm3,(%rax)
  400ab8:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
  400abd:	c5 fb 11 50 10       	vmovsd %xmm2,0x10(%rax)
  400ac2:	c5 fb 11 60 18       	vmovsd %xmm4,0x18(%rax)
  400ac7:	c5 fb 11 40 20       	vmovsd %xmm0,0x20(%rax)
  400acc:	48 83 c0 38          	add    $0x38,%rax
  400ad0:	83 eb 01             	sub    $0x1,%ebx
  400ad3:	48 89 05 8e 15 20 00 	mov    %rax,0x20158e(%rip)        # 602068 <Body_arena_head>
  400ada:	0f 85 d8 fe ff ff    	jne    4009b8 <_Z13instantiationv+0x18>
  400ae0:	48 83 c4 20          	add    $0x20,%rsp
  400ae4:	5b                   	pop    %rbx
  400ae5:	c3                   	retq   
  400ae6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aed:	00 00 00 

0000000000400af0 <_Z10run_simplev>:
  400af0:	c5 fb 10 15 50 03 00 	vmovsd 0x350(%rip),%xmm2        # 400e48 <_IO_stdin_used+0x68>
  400af7:	00 
  400af8:	b9 f4 01 00 00       	mov    $0x1f4,%ecx
  400afd:	ba b8 20 60 00       	mov    $0x6020b8,%edx
  400b02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  400b08:	c5 eb 59 42 18       	vmulsd 0x18(%rdx),%xmm2,%xmm0
  400b0d:	b8 64 00 00 00       	mov    $0x64,%eax
  400b12:	c5 eb 59 72 10       	vmulsd 0x10(%rdx),%xmm2,%xmm6
  400b17:	c5 fb 10 6a e8       	vmovsd -0x18(%rdx),%xmm5
  400b1c:	c5 fb 10 0a          	vmovsd (%rdx),%xmm1
  400b20:	c5 fb 10 62 f0       	vmovsd -0x10(%rdx),%xmm4
  400b25:	c5 fb 10 5a f8       	vmovsd -0x8(%rdx),%xmm3
  400b2a:	c5 cb 5e f5          	vdivsd %xmm5,%xmm6,%xmm6
  400b2e:	c5 fb 5e ed          	vdivsd %xmm5,%xmm0,%xmm5
  400b32:	c5 fb 10 42 08       	vmovsd 0x8(%rdx),%xmm0
  400b37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b3e:	00 00 
  400b40:	c5 f3 58 ce          	vaddsd %xmm6,%xmm1,%xmm1
  400b44:	83 e8 01             	sub    $0x1,%eax
  400b47:	c5 fb 58 c5          	vaddsd %xmm5,%xmm0,%xmm0
  400b4b:	c4 e2 f1 b9 e2       	vfmadd231sd %xmm2,%xmm1,%xmm4
  400b50:	c4 e2 f9 b9 da       	vfmadd231sd %xmm2,%xmm0,%xmm3
  400b55:	75 e9                	jne    400b40 <_Z10run_simplev+0x50>
  400b57:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
  400b5b:	48 83 c2 38          	add    $0x38,%rdx
  400b5f:	c5 fb 11 42 d0       	vmovsd %xmm0,-0x30(%rdx)
  400b64:	c5 fb 11 62 b8       	vmovsd %xmm4,-0x48(%rdx)
  400b69:	c5 fb 11 5a c0       	vmovsd %xmm3,-0x40(%rdx)
  400b6e:	48 81 fa b8 f6 66 00 	cmp    $0x66f6b8,%rdx
  400b75:	75 91                	jne    400b08 <_Z10run_simplev+0x18>
  400b77:	83 e9 01             	sub    $0x1,%ecx
  400b7a:	75 81                	jne    400afd <_Z10run_simplev+0xd>
  400b7c:	f3 c3                	repz retq 
  400b7e:	66 90                	xchg   %ax,%ax

0000000000400b80 <_Z14run_simulationv>:
  400b80:	c5 fb 10 3d 18 03 00 	vmovsd 0x318(%rip),%xmm7        # 400ea0 <_IO_stdin_used+0xc0>
  400b87:	00 
  400b88:	be 05 00 00 00       	mov    $0x5,%esi
  400b8d:	c5 fb 10 2d b3 02 00 	vmovsd 0x2b3(%rip),%xmm5        # 400e48 <_IO_stdin_used+0x68>
  400b94:	00 
  400b95:	c5 fb 10 35 c3 02 00 	vmovsd 0x2c3(%rip),%xmm6        # 400e60 <_IO_stdin_used+0x80>
  400b9c:	00 
  400b9d:	c5 79 28 c7          	vmovapd %xmm7,%xmm8
  400ba1:	c5 fb 10 25 d7 02 00 	vmovsd 0x2d7(%rip),%xmm4        # 400e80 <_IO_stdin_used+0xa0>
  400ba8:	00 
  400ba9:	c5 fb 10 1d d7 02 00 	vmovsd 0x2d7(%rip),%xmm3        # 400e88 <_IO_stdin_used+0xa8>
  400bb0:	00 
  400bb1:	c5 7b 10 0d d7 02 00 	vmovsd 0x2d7(%rip),%xmm9        # 400e90 <_IO_stdin_used+0xb0>
  400bb8:	00 
  400bb9:	b8 c8 20 60 00       	mov    $0x6020c8,%eax
  400bbe:	66 90                	xchg   %ax,%ax
  400bc0:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400bc7:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400bce:	00 
  400bcf:	48 83 c0 38          	add    $0x38,%rax
  400bd3:	48 3d c8 f6 66 00    	cmp    $0x66f6c8,%rax
  400bd9:	75 e5                	jne    400bc0 <_Z14run_simulationv+0x40>
  400bdb:	b9 a0 f6 66 00       	mov    $0x66f6a0,%ecx
  400be0:	ba a0 20 60 00       	mov    $0x6020a0,%edx
  400be5:	0f 1f 00             	nopl   (%rax)
  400be8:	b8 a0 20 60 00       	mov    $0x6020a0,%eax
  400bed:	0f 1f 00             	nopl   (%rax)
  400bf0:	48 39 c2             	cmp    %rax,%rdx
  400bf3:	74 5e                	je     400c53 <_Z14run_simulationv+0xd3>
  400bf5:	c5 fb 10 42 10       	vmovsd 0x10(%rdx),%xmm0
  400bfa:	c5 db 59 0a          	vmulsd (%rdx),%xmm4,%xmm1
  400bfe:	c5 7b 5c 58 10       	vsubsd 0x10(%rax),%xmm0,%xmm11
  400c03:	c5 fb 10 52 08       	vmovsd 0x8(%rdx),%xmm2
  400c08:	c5 eb 5c 50 08       	vsubsd 0x8(%rax),%xmm2,%xmm2
  400c0d:	c5 f3 59 08          	vmulsd (%rax),%xmm1,%xmm1
  400c11:	c4 c1 23 59 c3       	vmulsd %xmm11,%xmm11,%xmm0
  400c16:	c4 e2 e9 b9 c2       	vfmadd231sd %xmm2,%xmm2,%xmm0
  400c1b:	c5 2b 51 d0          	vsqrtsd %xmm0,%xmm10,%xmm10
  400c1f:	c5 79 29 d0          	vmovapd %xmm10,%xmm0
  400c23:	c4 c2 e1 99 c2       	vfmadd132sd %xmm10,%xmm3,%xmm0
  400c28:	c5 f3 5e c0          	vdivsd %xmm0,%xmm1,%xmm0
  400c2c:	c5 fb 59 d2          	vmulsd %xmm2,%xmm0,%xmm2
  400c30:	c4 c1 7b 59 c3       	vmulsd %xmm11,%xmm0,%xmm0
  400c35:	c4 c1 6b 5e d2       	vdivsd %xmm10,%xmm2,%xmm2
  400c3a:	c4 c1 7b 5e c2       	vdivsd %xmm10,%xmm0,%xmm0
  400c3f:	c5 eb 58 50 28       	vaddsd 0x28(%rax),%xmm2,%xmm2
  400c44:	c5 fb 11 50 28       	vmovsd %xmm2,0x28(%rax)
  400c49:	c5 fb 58 40 30       	vaddsd 0x30(%rax),%xmm0,%xmm0
  400c4e:	c5 fb 11 40 30       	vmovsd %xmm0,0x30(%rax)
  400c53:	48 83 c0 38          	add    $0x38,%rax
  400c57:	48 39 c1             	cmp    %rax,%rcx
  400c5a:	75 94                	jne    400bf0 <_Z14run_simulationv+0x70>
  400c5c:	48 83 c2 38          	add    $0x38,%rdx
  400c60:	48 39 d1             	cmp    %rdx,%rcx
  400c63:	75 83                	jne    400be8 <_Z14run_simulationv+0x68>
  400c65:	c5 fb 10 15 23 02 00 	vmovsd 0x223(%rip),%xmm2        # 400e90 <_IO_stdin_used+0xb0>
  400c6c:	00 
  400c6d:	b8 b8 20 60 00       	mov    $0x6020b8,%eax
  400c72:	ba b8 f6 66 00       	mov    $0x66f6b8,%edx
  400c77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400c7e:	00 00 
  400c80:	c5 d3 59 48 10       	vmulsd 0x10(%rax),%xmm5,%xmm1
  400c85:	c5 7b 10 50 e8       	vmovsd -0x18(%rax),%xmm10
  400c8a:	c5 d3 59 40 18       	vmulsd 0x18(%rax),%xmm5,%xmm0
  400c8f:	c4 c1 73 5e ca       	vdivsd %xmm10,%xmm1,%xmm1
  400c94:	c4 c1 7b 5e c2       	vdivsd %xmm10,%xmm0,%xmm0
  400c99:	c5 f3 58 08          	vaddsd (%rax),%xmm1,%xmm1
  400c9d:	c5 79 28 d9          	vmovapd %xmm1,%xmm11
  400ca1:	c5 fb 11 08          	vmovsd %xmm1,(%rax)
  400ca5:	c4 62 d1 a9 58 f0    	vfmadd213sd -0x10(%rax),%xmm5,%xmm11
  400cab:	c4 c1 79 2f f3       	vcomisd %xmm11,%xmm6
  400cb0:	c5 7b 11 58 f0       	vmovsd %xmm11,-0x10(%rax)
  400cb5:	c5 fb 58 40 08       	vaddsd 0x8(%rax),%xmm0,%xmm0
  400cba:	c5 79 28 d0          	vmovapd %xmm0,%xmm10
  400cbe:	c5 fb 11 40 08       	vmovsd %xmm0,0x8(%rax)
  400cc3:	c4 62 d1 a9 50 f8    	vfmadd213sd -0x8(%rax),%xmm5,%xmm10
  400cc9:	c5 7b 11 50 f8       	vmovsd %xmm10,-0x8(%rax)
  400cce:	77 07                	ja     400cd7 <_Z14run_simulationv+0x157>
  400cd0:	c4 41 79 2f d9       	vcomisd %xmm9,%xmm11
  400cd5:	76 08                	jbe    400cdf <_Z14run_simulationv+0x15f>
  400cd7:	c5 f1 57 cf          	vxorpd %xmm7,%xmm1,%xmm1
  400cdb:	c5 fb 11 08          	vmovsd %xmm1,(%rax)
  400cdf:	c4 c1 79 2f f2       	vcomisd %xmm10,%xmm6
  400ce4:	77 06                	ja     400cec <_Z14run_simulationv+0x16c>
  400ce6:	c5 79 2f d2          	vcomisd %xmm2,%xmm10
  400cea:	76 0a                	jbe    400cf6 <_Z14run_simulationv+0x176>
  400cec:	c4 c1 79 57 c0       	vxorpd %xmm8,%xmm0,%xmm0
  400cf1:	c5 fb 11 40 08       	vmovsd %xmm0,0x8(%rax)
  400cf6:	48 83 c0 38          	add    $0x38,%rax
  400cfa:	48 39 c2             	cmp    %rax,%rdx
  400cfd:	75 81                	jne    400c80 <_Z14run_simulationv+0x100>
  400cff:	83 ee 01             	sub    $0x1,%esi
  400d02:	0f 85 b1 fe ff ff    	jne    400bb9 <_Z14run_simulationv+0x39>
  400d08:	f3 c3                	repz retq 
  400d0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400d10 <_Z11r_float2intf>:
  400d10:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  400d14:	c3                   	retq   
  400d15:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1c:	00 00 00 
  400d1f:	90                   	nop

0000000000400d20 <_ZN7measureINSt6chrono8durationIlSt5ratioILl1ELl1000EEEEE9executionIRFvvEIEEElOT_DpOT0_>:
  400d20:	55                   	push   %rbp
  400d21:	53                   	push   %rbx
  400d22:	48 89 fd             	mov    %rdi,%rbp
  400d25:	48 83 ec 08          	sub    $0x8,%rsp
  400d29:	e8 e2 f9 ff ff       	callq  400710 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d2e:	48 89 c3             	mov    %rax,%rbx
  400d31:	ff d5                	callq  *%rbp
  400d33:	e8 d8 f9 ff ff       	callq  400710 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  400d38:	48 ba db 34 b6 d7 82 	movabs $0x431bde82d7b634db,%rdx
  400d3f:	de 1b 43 
  400d42:	48 29 d8             	sub    %rbx,%rax
  400d45:	48 83 c4 08          	add    $0x8,%rsp
  400d49:	48 89 c1             	mov    %rax,%rcx
  400d4c:	48 f7 ea             	imul   %rdx
  400d4f:	48 c1 f9 3f          	sar    $0x3f,%rcx
  400d53:	5b                   	pop    %rbx
  400d54:	5d                   	pop    %rbp
  400d55:	48 89 d0             	mov    %rdx,%rax
  400d58:	48 c1 f8 12          	sar    $0x12,%rax
  400d5c:	48 29 c8             	sub    %rcx,%rax
  400d5f:	c3                   	retq   

0000000000400d60 <__libc_csu_init>:
  400d60:	41 57                	push   %r15
  400d62:	41 56                	push   %r14
  400d64:	41 89 ff             	mov    %edi,%r15d
  400d67:	41 55                	push   %r13
  400d69:	41 54                	push   %r12
  400d6b:	4c 8d 25 7e 10 20 00 	lea    0x20107e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400d72:	55                   	push   %rbp
  400d73:	48 8d 2d 8e 10 20 00 	lea    0x20108e(%rip),%rbp        # 601e08 <__init_array_end>
  400d7a:	53                   	push   %rbx
  400d7b:	49 89 f6             	mov    %rsi,%r14
  400d7e:	49 89 d5             	mov    %rdx,%r13
  400d81:	4c 29 e5             	sub    %r12,%rbp
  400d84:	48 83 ec 08          	sub    $0x8,%rsp
  400d88:	48 c1 fd 03          	sar    $0x3,%rbp
  400d8c:	e8 e7 f8 ff ff       	callq  400678 <_init>
  400d91:	48 85 ed             	test   %rbp,%rbp
  400d94:	74 20                	je     400db6 <__libc_csu_init+0x56>
  400d96:	31 db                	xor    %ebx,%ebx
  400d98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d9f:	00 
  400da0:	4c 89 ea             	mov    %r13,%rdx
  400da3:	4c 89 f6             	mov    %r14,%rsi
  400da6:	44 89 ff             	mov    %r15d,%edi
  400da9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dad:	48 83 c3 01          	add    $0x1,%rbx
  400db1:	48 39 eb             	cmp    %rbp,%rbx
  400db4:	75 ea                	jne    400da0 <__libc_csu_init+0x40>
  400db6:	48 83 c4 08          	add    $0x8,%rsp
  400dba:	5b                   	pop    %rbx
  400dbb:	5d                   	pop    %rbp
  400dbc:	41 5c                	pop    %r12
  400dbe:	41 5d                	pop    %r13
  400dc0:	41 5e                	pop    %r14
  400dc2:	41 5f                	pop    %r15
  400dc4:	c3                   	retq   
  400dc5:	90                   	nop
  400dc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcd:	00 00 00 

0000000000400dd0 <__libc_csu_fini>:
  400dd0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400dd4 <_fini>:
  400dd4:	48 83 ec 08          	sub    $0x8,%rsp
  400dd8:	48 83 c4 08          	add    $0x8,%rsp
  400ddc:	c3                   	retq   
