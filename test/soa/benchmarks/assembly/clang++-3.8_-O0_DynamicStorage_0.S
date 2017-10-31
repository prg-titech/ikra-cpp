
bin/clang++-3.8_-O0_DynamicStorage_0:     file format elf64-x86-64


Disassembly of section .init:

00000000004005a8 <_init>:
  4005a8:	48 83 ec 08          	sub    $0x8,%rsp
  4005ac:	48 8b 05 45 1a 20 00 	mov    0x201a45(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4005b3:	48 85 c0             	test   %rax,%rax
  4005b6:	74 05                	je     4005bd <_init+0x15>
  4005b8:	e8 73 00 00 00       	callq  400630 <_Unwind_Resume@plt+0x10>
  4005bd:	48 83 c4 08          	add    $0x8,%rsp
  4005c1:	c3                   	retq   

Disassembly of section .plt:

00000000004005d0 <printf@plt-0x10>:
  4005d0:	ff 35 32 1a 20 00    	pushq  0x201a32(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4005d6:	ff 25 34 1a 20 00    	jmpq   *0x201a34(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4005dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005e0 <printf@plt>:
  4005e0:	ff 25 32 1a 20 00    	jmpq   *0x201a32(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005e6:	68 00 00 00 00       	pushq  $0x0
  4005eb:	e9 e0 ff ff ff       	jmpq   4005d0 <_init+0x28>

00000000004005f0 <malloc@plt>:
  4005f0:	ff 25 2a 1a 20 00    	jmpq   *0x201a2a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005f6:	68 01 00 00 00       	pushq  $0x1
  4005fb:	e9 d0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400600 <__libc_start_main@plt>:
  400600:	ff 25 22 1a 20 00    	jmpq   *0x201a22(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400606:	68 02 00 00 00       	pushq  $0x2
  40060b:	e9 c0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400610 <__gxx_personality_v0@plt>:
  400610:	ff 25 1a 1a 20 00    	jmpq   *0x201a1a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400616:	68 03 00 00 00       	pushq  $0x3
  40061b:	e9 b0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400620 <_Unwind_Resume@plt>:
  400620:	ff 25 12 1a 20 00    	jmpq   *0x201a12(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400626:	68 04 00 00 00       	pushq  $0x4
  40062b:	e9 a0 ff ff ff       	jmpq   4005d0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400630 <.plt.got>:
  400630:	ff 25 c2 19 20 00    	jmpq   *0x2019c2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400636:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400640 <_start>:
  400640:	31 ed                	xor    %ebp,%ebp
  400642:	49 89 d1             	mov    %rdx,%r9
  400645:	5e                   	pop    %rsi
  400646:	48 89 e2             	mov    %rsp,%rdx
  400649:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40064d:	50                   	push   %rax
  40064e:	54                   	push   %rsp
  40064f:	49 c7 c0 c0 0d 40 00 	mov    $0x400dc0,%r8
  400656:	48 c7 c1 50 0d 40 00 	mov    $0x400d50,%rcx
  40065d:	48 c7 c7 40 07 40 00 	mov    $0x400740,%rdi
  400664:	e8 97 ff ff ff       	callq  400600 <__libc_start_main@plt>
  400669:	f4                   	hlt    
  40066a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400670 <deregister_tm_clones>:
  400670:	b8 57 20 60 00       	mov    $0x602057,%eax
  400675:	55                   	push   %rbp
  400676:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  40067c:	48 83 f8 0e          	cmp    $0xe,%rax
  400680:	48 89 e5             	mov    %rsp,%rbp
  400683:	76 1b                	jbe    4006a0 <deregister_tm_clones+0x30>
  400685:	b8 00 00 00 00       	mov    $0x0,%eax
  40068a:	48 85 c0             	test   %rax,%rax
  40068d:	74 11                	je     4006a0 <deregister_tm_clones+0x30>
  40068f:	5d                   	pop    %rbp
  400690:	bf 50 20 60 00       	mov    $0x602050,%edi
  400695:	ff e0                	jmpq   *%rax
  400697:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40069e:	00 00 
  4006a0:	5d                   	pop    %rbp
  4006a1:	c3                   	retq   
  4006a2:	0f 1f 40 00          	nopl   0x0(%rax)
  4006a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ad:	00 00 00 

00000000004006b0 <register_tm_clones>:
  4006b0:	be 50 20 60 00       	mov    $0x602050,%esi
  4006b5:	55                   	push   %rbp
  4006b6:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  4006bd:	48 c1 fe 03          	sar    $0x3,%rsi
  4006c1:	48 89 e5             	mov    %rsp,%rbp
  4006c4:	48 89 f0             	mov    %rsi,%rax
  4006c7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006cb:	48 01 c6             	add    %rax,%rsi
  4006ce:	48 d1 fe             	sar    %rsi
  4006d1:	74 15                	je     4006e8 <register_tm_clones+0x38>
  4006d3:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d8:	48 85 c0             	test   %rax,%rax
  4006db:	74 0b                	je     4006e8 <register_tm_clones+0x38>
  4006dd:	5d                   	pop    %rbp
  4006de:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006e3:	ff e0                	jmpq   *%rax
  4006e5:	0f 1f 00             	nopl   (%rax)
  4006e8:	5d                   	pop    %rbp
  4006e9:	c3                   	retq   
  4006ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006f0 <__do_global_dtors_aux>:
  4006f0:	80 3d 59 19 20 00 00 	cmpb   $0x0,0x201959(%rip)        # 602050 <__TMC_END__>
  4006f7:	75 11                	jne    40070a <__do_global_dtors_aux+0x1a>
  4006f9:	55                   	push   %rbp
  4006fa:	48 89 e5             	mov    %rsp,%rbp
  4006fd:	e8 6e ff ff ff       	callq  400670 <deregister_tm_clones>
  400702:	5d                   	pop    %rbp
  400703:	c6 05 46 19 20 00 01 	movb   $0x1,0x201946(%rip)        # 602050 <__TMC_END__>
  40070a:	f3 c3                	repz retq 
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <frame_dummy>:
  400710:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400715:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400719:	75 05                	jne    400720 <frame_dummy+0x10>
  40071b:	eb 93                	jmp    4006b0 <register_tm_clones>
  40071d:	0f 1f 00             	nopl   (%rax)
  400720:	b8 00 00 00 00       	mov    $0x0,%eax
  400725:	48 85 c0             	test   %rax,%rax
  400728:	74 f1                	je     40071b <frame_dummy+0xb>
  40072a:	55                   	push   %rbp
  40072b:	48 89 e5             	mov    %rsp,%rbp
  40072e:	ff d0                	callq  *%rax
  400730:	5d                   	pop    %rbp
  400731:	e9 7a ff ff ff       	jmpq   4006b0 <register_tm_clones>
  400736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073d:	00 00 00 

0000000000400740 <main>:
  400740:	55                   	push   %rbp
  400741:	48 89 e5             	mov    %rsp,%rbp
  400744:	48 83 ec 50          	sub    $0x50,%rsp
  400748:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  40074f:	e8 3c 02 00 00       	callq  400990 <_ZN9TestClass18initialize_storageEv>
  400754:	31 c0                	xor    %eax,%eax
  400756:	89 c7                	mov    %eax,%edi
  400758:	e8 53 02 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40075d:	48 89 c7             	mov    %rax,%rdi
  400760:	48 89 c1             	mov    %rax,%rcx
  400763:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400767:	48 89 c7             	mov    %rax,%rdi
  40076a:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  40076e:	e8 6d 02 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  400773:	e9 00 00 00 00       	jmpq   400778 <main+0x38>
  400778:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  40077c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400781:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400785:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400789:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40078d:	e8 8e 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400792:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400796:	ba 88 88 00 00       	mov    $0x8888,%edx
  40079b:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40079f:	e8 cc 02 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007a4:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007a8:	e8 13 03 00 00       	callq  400ac0 <_ZN9TestClass15increase_field0Ev>
  4007ad:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b1:	e8 3a 03 00 00       	callq  400af0 <_ZN9TestClass15increase_field1Ev>
  4007b6:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007ba:	e8 61 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007bf:	8b 30                	mov    (%rax),%esi
  4007c1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c5:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007c8:	e8 73 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007cd:	48 bf d4 0d 40 00 00 	movabs $0x400dd4,%rdi
  4007d4:	00 00 00 
  4007d7:	8b 10                	mov    (%rax),%edx
  4007d9:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007dc:	b0 00                	mov    $0x0,%al
  4007de:	e8 fd fd ff ff       	callq  4005e0 <printf@plt>
  4007e3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007e7:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007ea:	e8 31 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007ef:	31 d2                	xor    %edx,%edx
  4007f1:	88 d1                	mov    %dl,%cl
  4007f3:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007f9:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007fc:	0f 85 15 00 00 00    	jne    400817 <main+0xd7>
  400802:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400806:	e8 35 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40080b:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400811:	0f 94 c1             	sete   %cl
  400814:	88 4d bf             	mov    %cl,-0x41(%rbp)
  400817:	8a 45 bf             	mov    -0x41(%rbp),%al
  40081a:	34 ff                	xor    $0xff,%al
  40081c:	24 01                	and    $0x1,%al
  40081e:	0f b6 c0             	movzbl %al,%eax
  400821:	48 83 c4 50          	add    $0x50,%rsp
  400825:	5d                   	pop    %rbp
  400826:	c3                   	retq   
  400827:	89 d1                	mov    %edx,%ecx
  400829:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40082d:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400830:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400834:	e8 d7 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  400839:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40083d:	e8 de fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400842:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400849:	1f 84 00 00 00 00 00 

0000000000400850 <_Z12new_instancev>:
  400850:	55                   	push   %rbp
  400851:	48 89 e5             	mov    %rsp,%rbp
  400854:	48 83 ec 20          	sub    $0x20,%rsp
  400858:	31 c0                	xor    %eax,%eax
  40085a:	89 c7                	mov    %eax,%edi
  40085c:	e8 4f 01 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  400861:	48 89 c7             	mov    %rax,%rdi
  400864:	48 89 c1             	mov    %rax,%rcx
  400867:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40086b:	48 89 c7             	mov    %rax,%rdi
  40086e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400872:	e8 69 01 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  400877:	e9 00 00 00 00       	jmpq   40087c <_Z12new_instancev+0x2c>
  40087c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400880:	48 83 c4 20          	add    $0x20,%rsp
  400884:	5d                   	pop    %rbp
  400885:	c3                   	retq   
  400886:	89 d1                	mov    %edx,%ecx
  400888:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40088c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40088f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400893:	e8 78 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  400898:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40089c:	e8 7f fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008a1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008a8:	0f 1f 84 00 00 00 00 
  4008af:	00 

00000000004008b0 <_Z12write_field0P9TestClass>:
  4008b0:	55                   	push   %rbp
  4008b1:	48 89 e5             	mov    %rsp,%rbp
  4008b4:	48 83 ec 10          	sub    $0x10,%rsp
  4008b8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008bc:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008c1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008c9:	48 89 c7             	mov    %rax,%rdi
  4008cc:	e8 4f 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4008d1:	48 83 c4 10          	add    $0x10,%rsp
  4008d5:	5d                   	pop    %rbp
  4008d6:	c3                   	retq   
  4008d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008de:	00 00 

00000000004008e0 <_Z11read_field0P9TestClass>:
  4008e0:	55                   	push   %rbp
  4008e1:	48 89 e5             	mov    %rsp,%rbp
  4008e4:	48 83 ec 10          	sub    $0x10,%rsp
  4008e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008f0:	e8 2b 02 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4008f5:	8b 00                	mov    (%rax),%eax
  4008f7:	48 83 c4 10          	add    $0x10,%rsp
  4008fb:	5d                   	pop    %rbp
  4008fc:	c3                   	retq   
  4008fd:	0f 1f 00             	nopl   (%rax)

0000000000400900 <_Z12write_field1P9TestClass>:
  400900:	55                   	push   %rbp
  400901:	48 89 e5             	mov    %rsp,%rbp
  400904:	48 83 ec 10          	sub    $0x10,%rsp
  400908:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40090c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400911:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400915:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400919:	48 89 c7             	mov    %rax,%rdi
  40091c:	e8 4f 01 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400921:	48 83 c4 10          	add    $0x10,%rsp
  400925:	5d                   	pop    %rbp
  400926:	c3                   	retq   
  400927:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40092e:	00 00 

0000000000400930 <_Z11read_field1P9TestClass>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400940:	e8 fb 01 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400945:	8b 00                	mov    (%rax),%eax
  400947:	48 83 c4 10          	add    $0x10,%rsp
  40094b:	5d                   	pop    %rbp
  40094c:	c3                   	retq   
  40094d:	0f 1f 00             	nopl   (%rax)

0000000000400950 <_Z21explicit_write_field0m>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400958:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40095c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  400963:	00 
  400964:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40096b:	5d                   	pop    %rbp
  40096c:	c3                   	retq   
  40096d:	0f 1f 00             	nopl   (%rax)

0000000000400970 <_Z20explicit_read_field0m>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400978:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40097c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  400983:	00 
  400984:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  400987:	5d                   	pop    %rbp
  400988:	c3                   	retq   
  400989:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400990 <_ZN9TestClass18initialize_storageEv>:
  400990:	55                   	push   %rbp
  400991:	48 89 e5             	mov    %rsp,%rbp
  400994:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  40099b:	00 00 00 
  40099e:	48 89 c7             	mov    %rax,%rdi
  4009a1:	e8 ba 01 00 00       	callq  400b60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009a6:	5d                   	pop    %rbp
  4009a7:	c3                   	retq   
  4009a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009af:	00 

00000000004009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 83 ec 10          	sub    $0x10,%rsp
  4009b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bc:	e8 4f 03 00 00       	callq  400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009c1:	e8 9a 02 00 00       	callq  400c60 <_ZN9TestClass7storageEv>
  4009c6:	48 8b 38             	mov    (%rax),%rdi
  4009c9:	48 89 f9             	mov    %rdi,%rcx
  4009cc:	48 83 c1 01          	add    $0x1,%rcx
  4009d0:	48 89 08             	mov    %rcx,(%rax)
  4009d3:	e8 48 03 00 00       	callq  400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009d8:	48 83 c4 10          	add    $0x10,%rsp
  4009dc:	5d                   	pop    %rbp
  4009dd:	c3                   	retq   
  4009de:	66 90                	xchg   %ax,%ax

00000000004009e0 <_ZN9TestClassC2Ev>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 83 ec 10          	sub    $0x10,%rsp
  4009e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009f0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009f4:	e8 b7 01 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009f9:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4009fd:	e8 be 01 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400a02:	48 83 c4 10          	add    $0x10,%rsp
  400a06:	5d                   	pop    %rbp
  400a07:	c3                   	retq   
  400a08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a0f:	00 

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a18:	5d                   	pop    %rbp
  400a19:	c3                   	retq   
  400a1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 83 ec 30          	sub    $0x30,%rsp
  400a28:	48 89 f8             	mov    %rdi,%rax
  400a2b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a2f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a32:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a36:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a39:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a3d:	48 89 f7             	mov    %rsi,%rdi
  400a40:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a44:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a47:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a4b:	e8 d0 01 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a50:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a53:	89 10                	mov    %edx,(%rax)
  400a55:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a59:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a5d:	e8 ee 01 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400a62:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a66:	48 83 c4 30          	add    $0x30,%rsp
  400a6a:	5d                   	pop    %rbp
  400a6b:	c3                   	retq   
  400a6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a70:	55                   	push   %rbp
  400a71:	48 89 e5             	mov    %rsp,%rbp
  400a74:	48 83 ec 30          	sub    $0x30,%rsp
  400a78:	48 89 f8             	mov    %rdi,%rax
  400a7b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a7f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a82:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a86:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a89:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a8d:	48 89 f7             	mov    %rsi,%rdi
  400a90:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a94:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a97:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a9b:	e8 20 02 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400aa3:	89 10                	mov    %edx,(%rax)
  400aa5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400aa9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400aad:	e8 4e 02 00 00       	callq  400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400ab2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ab6:	48 83 c4 30          	add    $0x30,%rsp
  400aba:	5d                   	pop    %rbp
  400abb:	c3                   	retq   
  400abc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ac0 <_ZN9TestClass15increase_field0Ev>:
  400ac0:	55                   	push   %rbp
  400ac1:	48 89 e5             	mov    %rsp,%rbp
  400ac4:	48 83 ec 10          	sub    $0x10,%rsp
  400ac8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400acc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ad1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ad9:	48 89 c7             	mov    %rax,%rdi
  400adc:	e8 ef 00 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400ae1:	48 83 c4 10          	add    $0x10,%rsp
  400ae5:	5d                   	pop    %rbp
  400ae6:	c3                   	retq   
  400ae7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400aee:	00 00 

0000000000400af0 <_ZN9TestClass15increase_field1Ev>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 83 ec 10          	sub    $0x10,%rsp
  400af8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400afc:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b01:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b05:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b09:	48 89 c7             	mov    %rax,%rdi
  400b0c:	e8 5f 01 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400b11:	48 83 c4 10          	add    $0x10,%rsp
  400b15:	5d                   	pop    %rbp
  400b16:	c3                   	retq   
  400b17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b1e:	00 00 

0000000000400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 83 ec 10          	sub    $0x10,%rsp
  400b28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b30:	e8 eb 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b35:	48 83 c4 10          	add    $0x10,%rsp
  400b39:	5d                   	pop    %rbp
  400b3a:	c3                   	retq   
  400b3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 10          	sub    $0x10,%rsp
  400b48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b50:	e8 6b 01 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b55:	48 83 c4 10          	add    $0x10,%rsp
  400b59:	5d                   	pop    %rbp
  400b5a:	c3                   	retq   
  400b5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 83 ec 10          	sub    $0x10,%rsp
  400b68:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  400b6d:	89 c1                	mov    %eax,%ecx
  400b6f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b73:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b77:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400b7b:	48 89 cf             	mov    %rcx,%rdi
  400b7e:	e8 6d fa ff ff       	callq  4005f0 <malloc@plt>
  400b83:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400b87:	48 89 41 08          	mov    %rax,0x8(%rcx)
  400b8b:	48 c7 41 18 00 00 00 	movq   $0x0,0x18(%rcx)
  400b92:	00 
  400b93:	48 c7 41 10 00 00 00 	movq   $0x0,0x10(%rcx)
  400b9a:	00 
  400b9b:	48 83 c4 10          	add    $0x10,%rsp
  400b9f:	5d                   	pop    %rbp
  400ba0:	c3                   	retq   
  400ba1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ba8:	00 00 00 
  400bab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb8:	5d                   	pop    %rbp
  400bb9:	c3                   	retq   
  400bba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bc8:	5d                   	pop    %rbp
  400bc9:	c3                   	retq   
  400bca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 83 ec 30          	sub    $0x30,%rsp
  400bd8:	48 89 f8             	mov    %rdi,%rax
  400bdb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bdf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400be2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400be6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400be9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bed:	48 89 f7             	mov    %rsi,%rdi
  400bf0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400bf4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400bf7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400bfb:	e8 20 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c00:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c03:	0f af 10             	imul   (%rax),%edx
  400c06:	89 10                	mov    %edx,(%rax)
  400c08:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c0c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c10:	e8 3b 00 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400c15:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c19:	48 83 c4 30          	add    $0x30,%rsp
  400c1d:	5d                   	pop    %rbp
  400c1e:	c3                   	retq   
  400c1f:	90                   	nop

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c20:	55                   	push   %rbp
  400c21:	48 89 e5             	mov    %rsp,%rbp
  400c24:	48 83 ec 10          	sub    $0x10,%rsp
  400c28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c30:	48 c1 e7 02          	shl    $0x2,%rdi
  400c34:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c38:	e8 23 00 00 00       	callq  400c60 <_ZN9TestClass7storageEv>
  400c3d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400c41:	48 03 78 08          	add    0x8(%rax),%rdi
  400c45:	48 89 f8             	mov    %rdi,%rax
  400c48:	48 83 c4 10          	add    $0x10,%rsp
  400c4c:	5d                   	pop    %rbp
  400c4d:	c3                   	retq   
  400c4e:	66 90                	xchg   %ax,%ax

0000000000400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400c50:	55                   	push   %rbp
  400c51:	48 89 e5             	mov    %rsp,%rbp
  400c54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c58:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c5c:	5d                   	pop    %rbp
  400c5d:	c3                   	retq   
  400c5e:	66 90                	xchg   %ax,%ax

0000000000400c60 <_ZN9TestClass7storageEv>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c6b:	00 00 00 
  400c6e:	5d                   	pop    %rbp
  400c6f:	c3                   	retq   

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400c70:	55                   	push   %rbp
  400c71:	48 89 e5             	mov    %rsp,%rbp
  400c74:	48 83 ec 30          	sub    $0x30,%rsp
  400c78:	48 89 f8             	mov    %rdi,%rax
  400c7b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c7f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c82:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c86:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c89:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c8d:	48 89 f7             	mov    %rsi,%rdi
  400c90:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c94:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c97:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c9b:	e8 20 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ca3:	0f af 10             	imul   (%rax),%edx
  400ca6:	89 10                	mov    %edx,(%rax)
  400ca8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cac:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cb0:	e8 4b 00 00 00       	callq  400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400cb5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cb9:	48 83 c4 30          	add    $0x30,%rsp
  400cbd:	5d                   	pop    %rbp
  400cbe:	c3                   	retq   
  400cbf:	90                   	nop

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 83 ec 10          	sub    $0x10,%rsp
  400cc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ccc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cd0:	48 c1 e7 02          	shl    $0x2,%rdi
  400cd4:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400cd8:	e8 83 ff ff ff       	callq  400c60 <_ZN9TestClass7storageEv>
  400cdd:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400ce1:	48 03 78 08          	add    0x8(%rax),%rdi
  400ce5:	48 81 c7 d0 48 00 00 	add    $0x48d0,%rdi
  400cec:	48 89 f8             	mov    %rdi,%rax
  400cef:	48 83 c4 10          	add    $0x10,%rsp
  400cf3:	5d                   	pop    %rbp
  400cf4:	c3                   	retq   
  400cf5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfc:	00 00 00 
  400cff:	90                   	nop

0000000000400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d08:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d0c:	5d                   	pop    %rbp
  400d0d:	c3                   	retq   
  400d0e:	66 90                	xchg   %ax,%ax

0000000000400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d10:	55                   	push   %rbp
  400d11:	48 89 e5             	mov    %rsp,%rbp
  400d14:	5d                   	pop    %rbp
  400d15:	c3                   	retq   
  400d16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1d:	00 00 00 

0000000000400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400d20:	55                   	push   %rbp
  400d21:	48 89 e5             	mov    %rsp,%rbp
  400d24:	48 83 ec 10          	sub    $0x10,%rsp
  400d28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d30:	e8 0b 00 00 00       	callq  400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d35:	48 83 c4 10          	add    $0x10,%rsp
  400d39:	5d                   	pop    %rbp
  400d3a:	c3                   	retq   
  400d3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d40:	55                   	push   %rbp
  400d41:	48 89 e5             	mov    %rsp,%rbp
  400d44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d48:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400d4c:	5d                   	pop    %rbp
  400d4d:	c3                   	retq   
  400d4e:	66 90                	xchg   %ax,%ax

0000000000400d50 <__libc_csu_init>:
  400d50:	41 57                	push   %r15
  400d52:	41 56                	push   %r14
  400d54:	41 89 ff             	mov    %edi,%r15d
  400d57:	41 55                	push   %r13
  400d59:	41 54                	push   %r12
  400d5b:	4c 8d 25 7e 10 20 00 	lea    0x20107e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d62:	55                   	push   %rbp
  400d63:	48 8d 2d 7e 10 20 00 	lea    0x20107e(%rip),%rbp        # 601de8 <__init_array_end>
  400d6a:	53                   	push   %rbx
  400d6b:	49 89 f6             	mov    %rsi,%r14
  400d6e:	49 89 d5             	mov    %rdx,%r13
  400d71:	4c 29 e5             	sub    %r12,%rbp
  400d74:	48 83 ec 08          	sub    $0x8,%rsp
  400d78:	48 c1 fd 03          	sar    $0x3,%rbp
  400d7c:	e8 27 f8 ff ff       	callq  4005a8 <_init>
  400d81:	48 85 ed             	test   %rbp,%rbp
  400d84:	74 20                	je     400da6 <__libc_csu_init+0x56>
  400d86:	31 db                	xor    %ebx,%ebx
  400d88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d8f:	00 
  400d90:	4c 89 ea             	mov    %r13,%rdx
  400d93:	4c 89 f6             	mov    %r14,%rsi
  400d96:	44 89 ff             	mov    %r15d,%edi
  400d99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d9d:	48 83 c3 01          	add    $0x1,%rbx
  400da1:	48 39 eb             	cmp    %rbp,%rbx
  400da4:	75 ea                	jne    400d90 <__libc_csu_init+0x40>
  400da6:	48 83 c4 08          	add    $0x8,%rsp
  400daa:	5b                   	pop    %rbx
  400dab:	5d                   	pop    %rbp
  400dac:	41 5c                	pop    %r12
  400dae:	41 5d                	pop    %r13
  400db0:	41 5e                	pop    %r14
  400db2:	41 5f                	pop    %r15
  400db4:	c3                   	retq   
  400db5:	90                   	nop
  400db6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dbd:	00 00 00 

0000000000400dc0 <__libc_csu_fini>:
  400dc0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400dc4 <_fini>:
  400dc4:	48 83 ec 08          	sub    $0x8,%rsp
  400dc8:	48 83 c4 08          	add    $0x8,%rsp
  400dcc:	c3                   	retq   
