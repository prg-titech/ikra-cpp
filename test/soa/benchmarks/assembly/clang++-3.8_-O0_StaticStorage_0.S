
bin/clang++-3.8_-O0_StaticStorage_0:     file format elf64-x86-64


Disassembly of section .init:

0000000000400570 <_init>:
  400570:	48 83 ec 08          	sub    $0x8,%rsp
  400574:	48 8b 05 7d 1a 20 00 	mov    0x201a7d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40057b:	48 85 c0             	test   %rax,%rax
  40057e:	74 05                	je     400585 <_init+0x15>
  400580:	e8 5b 00 00 00       	callq  4005e0 <_Unwind_Resume@plt+0x10>
  400585:	48 83 c4 08          	add    $0x8,%rsp
  400589:	c3                   	retq   

Disassembly of section .plt:

0000000000400590 <printf@plt-0x10>:
  400590:	ff 35 72 1a 20 00    	pushq  0x201a72(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400596:	ff 25 74 1a 20 00    	jmpq   *0x201a74(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40059c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005a0 <printf@plt>:
  4005a0:	ff 25 72 1a 20 00    	jmpq   *0x201a72(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005a6:	68 00 00 00 00       	pushq  $0x0
  4005ab:	e9 e0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005b0 <__libc_start_main@plt>:
  4005b0:	ff 25 6a 1a 20 00    	jmpq   *0x201a6a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005b6:	68 01 00 00 00       	pushq  $0x1
  4005bb:	e9 d0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005c0 <__gxx_personality_v0@plt>:
  4005c0:	ff 25 62 1a 20 00    	jmpq   *0x201a62(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4005c6:	68 02 00 00 00       	pushq  $0x2
  4005cb:	e9 c0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005d0 <_Unwind_Resume@plt>:
  4005d0:	ff 25 5a 1a 20 00    	jmpq   *0x201a5a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4005d6:	68 03 00 00 00       	pushq  $0x3
  4005db:	e9 b0 ff ff ff       	jmpq   400590 <_init+0x20>

Disassembly of section .plt.got:

00000000004005e0 <.plt.got>:
  4005e0:	ff 25 12 1a 20 00    	jmpq   *0x201a12(%rip)        # 601ff8 <_DYNAMIC+0x200>
  4005e6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004005f0 <_start>:
  4005f0:	31 ed                	xor    %ebp,%ebp
  4005f2:	49 89 d1             	mov    %rdx,%r9
  4005f5:	5e                   	pop    %rsi
  4005f6:	48 89 e2             	mov    %rsp,%rdx
  4005f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4005fd:	50                   	push   %rax
  4005fe:	54                   	push   %rsp
  4005ff:	49 c7 c0 e0 0c 40 00 	mov    $0x400ce0,%r8
  400606:	48 c7 c1 70 0c 40 00 	mov    $0x400c70,%rcx
  40060d:	48 c7 c7 f0 06 40 00 	mov    $0x4006f0,%rdi
  400614:	e8 97 ff ff ff       	callq  4005b0 <__libc_start_main@plt>
  400619:	f4                   	hlt    
  40061a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400620 <deregister_tm_clones>:
  400620:	b8 4f 20 60 00       	mov    $0x60204f,%eax
  400625:	55                   	push   %rbp
  400626:	48 2d 48 20 60 00    	sub    $0x602048,%rax
  40062c:	48 83 f8 0e          	cmp    $0xe,%rax
  400630:	48 89 e5             	mov    %rsp,%rbp
  400633:	76 1b                	jbe    400650 <deregister_tm_clones+0x30>
  400635:	b8 00 00 00 00       	mov    $0x0,%eax
  40063a:	48 85 c0             	test   %rax,%rax
  40063d:	74 11                	je     400650 <deregister_tm_clones+0x30>
  40063f:	5d                   	pop    %rbp
  400640:	bf 48 20 60 00       	mov    $0x602048,%edi
  400645:	ff e0                	jmpq   *%rax
  400647:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40064e:	00 00 
  400650:	5d                   	pop    %rbp
  400651:	c3                   	retq   
  400652:	0f 1f 40 00          	nopl   0x0(%rax)
  400656:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40065d:	00 00 00 

0000000000400660 <register_tm_clones>:
  400660:	be 48 20 60 00       	mov    $0x602048,%esi
  400665:	55                   	push   %rbp
  400666:	48 81 ee 48 20 60 00 	sub    $0x602048,%rsi
  40066d:	48 c1 fe 03          	sar    $0x3,%rsi
  400671:	48 89 e5             	mov    %rsp,%rbp
  400674:	48 89 f0             	mov    %rsi,%rax
  400677:	48 c1 e8 3f          	shr    $0x3f,%rax
  40067b:	48 01 c6             	add    %rax,%rsi
  40067e:	48 d1 fe             	sar    %rsi
  400681:	74 15                	je     400698 <register_tm_clones+0x38>
  400683:	b8 00 00 00 00       	mov    $0x0,%eax
  400688:	48 85 c0             	test   %rax,%rax
  40068b:	74 0b                	je     400698 <register_tm_clones+0x38>
  40068d:	5d                   	pop    %rbp
  40068e:	bf 48 20 60 00       	mov    $0x602048,%edi
  400693:	ff e0                	jmpq   *%rax
  400695:	0f 1f 00             	nopl   (%rax)
  400698:	5d                   	pop    %rbp
  400699:	c3                   	retq   
  40069a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006a0 <__do_global_dtors_aux>:
  4006a0:	80 3d a1 19 20 00 00 	cmpb   $0x0,0x2019a1(%rip)        # 602048 <__TMC_END__>
  4006a7:	75 11                	jne    4006ba <__do_global_dtors_aux+0x1a>
  4006a9:	55                   	push   %rbp
  4006aa:	48 89 e5             	mov    %rsp,%rbp
  4006ad:	e8 6e ff ff ff       	callq  400620 <deregister_tm_clones>
  4006b2:	5d                   	pop    %rbp
  4006b3:	c6 05 8e 19 20 00 01 	movb   $0x1,0x20198e(%rip)        # 602048 <__TMC_END__>
  4006ba:	f3 c3                	repz retq 
  4006bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006c0 <frame_dummy>:
  4006c0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  4006c5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4006c9:	75 05                	jne    4006d0 <frame_dummy+0x10>
  4006cb:	eb 93                	jmp    400660 <register_tm_clones>
  4006cd:	0f 1f 00             	nopl   (%rax)
  4006d0:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d5:	48 85 c0             	test   %rax,%rax
  4006d8:	74 f1                	je     4006cb <frame_dummy+0xb>
  4006da:	55                   	push   %rbp
  4006db:	48 89 e5             	mov    %rsp,%rbp
  4006de:	ff d0                	callq  *%rax
  4006e0:	5d                   	pop    %rbp
  4006e1:	e9 7a ff ff ff       	jmpq   400660 <register_tm_clones>
  4006e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ed:	00 00 00 

00000000004006f0 <main>:
  4006f0:	55                   	push   %rbp
  4006f1:	48 89 e5             	mov    %rsp,%rbp
  4006f4:	48 83 ec 50          	sub    $0x50,%rsp
  4006f8:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4006ff:	31 c0                	xor    %eax,%eax
  400701:	89 c7                	mov    %eax,%edi
  400703:	e8 28 02 00 00       	callq  400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400708:	48 89 c7             	mov    %rax,%rdi
  40070b:	48 89 c1             	mov    %rax,%rcx
  40070e:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400712:	48 89 c7             	mov    %rax,%rdi
  400715:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400719:	e8 52 02 00 00       	callq  400970 <_ZN9TestClassC2Ev>
  40071e:	e9 00 00 00 00       	jmpq   400723 <main+0x33>
  400723:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400727:	ba 77 77 00 00       	mov    $0x7777,%edx
  40072c:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400730:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400734:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  400738:	e8 73 02 00 00       	callq  4009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40073d:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400741:	ba 88 88 00 00       	mov    $0x8888,%edx
  400746:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40074a:	e8 b1 02 00 00       	callq  400a00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40074f:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400753:	e8 f8 02 00 00       	callq  400a50 <_ZN9TestClass15increase_field0Ev>
  400758:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40075c:	e8 1f 03 00 00       	callq  400a80 <_ZN9TestClass15increase_field1Ev>
  400761:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400765:	e8 46 03 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40076a:	8b 30                	mov    (%rax),%esi
  40076c:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400770:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400773:	e8 58 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400778:	48 bf f4 0c 40 00 00 	movabs $0x400cf4,%rdi
  40077f:	00 00 00 
  400782:	8b 10                	mov    (%rax),%edx
  400784:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  400787:	b0 00                	mov    $0x0,%al
  400789:	e8 12 fe ff ff       	callq  4005a0 <printf@plt>
  40078e:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400792:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400795:	e8 16 03 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40079a:	31 d2                	xor    %edx,%edx
  40079c:	88 d1                	mov    %dl,%cl
  40079e:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007a4:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007a7:	0f 85 15 00 00 00    	jne    4007c2 <main+0xd2>
  4007ad:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b1:	e8 1a 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007b6:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007bc:	0f 94 c1             	sete   %cl
  4007bf:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007c2:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007c5:	34 ff                	xor    $0xff,%al
  4007c7:	24 01                	and    $0x1,%al
  4007c9:	0f b6 c0             	movzbl %al,%eax
  4007cc:	48 83 c4 50          	add    $0x50,%rsp
  4007d0:	5d                   	pop    %rbp
  4007d1:	c3                   	retq   
  4007d2:	89 d1                	mov    %edx,%ecx
  4007d4:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007d8:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4007db:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  4007df:	e8 bc 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  4007e4:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4007e8:	e8 e3 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  4007ed:	0f 1f 00             	nopl   (%rax)

00000000004007f0 <_Z12new_instancev>:
  4007f0:	55                   	push   %rbp
  4007f1:	48 89 e5             	mov    %rsp,%rbp
  4007f4:	48 83 ec 20          	sub    $0x20,%rsp
  4007f8:	31 c0                	xor    %eax,%eax
  4007fa:	89 c7                	mov    %eax,%edi
  4007fc:	e8 2f 01 00 00       	callq  400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400801:	48 89 c7             	mov    %rax,%rdi
  400804:	48 89 c1             	mov    %rax,%rcx
  400807:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40080b:	48 89 c7             	mov    %rax,%rdi
  40080e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400812:	e8 59 01 00 00       	callq  400970 <_ZN9TestClassC2Ev>
  400817:	e9 00 00 00 00       	jmpq   40081c <_Z12new_instancev+0x2c>
  40081c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400820:	48 83 c4 20          	add    $0x20,%rsp
  400824:	5d                   	pop    %rbp
  400825:	c3                   	retq   
  400826:	89 d1                	mov    %edx,%ecx
  400828:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40082c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40082f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400833:	e8 68 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400838:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40083c:	e8 8f fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400841:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400848:	0f 1f 84 00 00 00 00 
  40084f:	00 

0000000000400850 <_Z12write_field0P9TestClass>:
  400850:	55                   	push   %rbp
  400851:	48 89 e5             	mov    %rsp,%rbp
  400854:	48 83 ec 10          	sub    $0x10,%rsp
  400858:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40085c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400861:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400865:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400869:	48 89 c7             	mov    %rax,%rdi
  40086c:	e8 3f 01 00 00       	callq  4009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400871:	48 83 c4 10          	add    $0x10,%rsp
  400875:	5d                   	pop    %rbp
  400876:	c3                   	retq   
  400877:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40087e:	00 00 

0000000000400880 <_Z11read_field0P9TestClass>:
  400880:	55                   	push   %rbp
  400881:	48 89 e5             	mov    %rsp,%rbp
  400884:	48 83 ec 10          	sub    $0x10,%rsp
  400888:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40088c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400890:	e8 1b 02 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400895:	8b 00                	mov    (%rax),%eax
  400897:	48 83 c4 10          	add    $0x10,%rsp
  40089b:	5d                   	pop    %rbp
  40089c:	c3                   	retq   
  40089d:	0f 1f 00             	nopl   (%rax)

00000000004008a0 <_Z12write_field1P9TestClass>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 83 ec 10          	sub    $0x10,%rsp
  4008a8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008ac:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008b1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008b9:	48 89 c7             	mov    %rax,%rdi
  4008bc:	e8 3f 01 00 00       	callq  400a00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4008c1:	48 83 c4 10          	add    $0x10,%rsp
  4008c5:	5d                   	pop    %rbp
  4008c6:	c3                   	retq   
  4008c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008ce:	00 00 

00000000004008d0 <_Z11read_field1P9TestClass>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 10          	sub    $0x10,%rsp
  4008d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008dc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e0:	e8 eb 01 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4008e5:	8b 00                	mov    (%rax),%eax
  4008e7:	48 83 c4 10          	add    $0x10,%rsp
  4008eb:	5d                   	pop    %rbp
  4008ec:	c3                   	retq   
  4008ed:	0f 1f 00             	nopl   (%rax)

00000000004008f0 <_Z21explicit_write_field0m>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008fc:	48 8b 04 25 f8 b1 60 	mov    0x60b1f8,%rax
  400903:	00 
  400904:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40090b:	5d                   	pop    %rbp
  40090c:	c3                   	retq   
  40090d:	0f 1f 00             	nopl   (%rax)

0000000000400910 <_Z20explicit_read_field0m>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400918:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40091c:	48 8b 04 25 f8 b1 60 	mov    0x60b1f8,%rax
  400923:	00 
  400924:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  400927:	5d                   	pop    %rbp
  400928:	c3                   	retq   
  400929:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	e8 ef 02 00 00       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400941:	48 8b 3c 25 50 20 60 	mov    0x602050,%rdi
  400948:	00 
  400949:	48 89 f8             	mov    %rdi,%rax
  40094c:	48 83 c0 01          	add    $0x1,%rax
  400950:	48 89 04 25 50 20 60 	mov    %rax,0x602050
  400957:	00 
  400958:	e8 e3 02 00 00       	callq  400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  40095d:	48 83 c4 10          	add    $0x10,%rsp
  400961:	5d                   	pop    %rbp
  400962:	c3                   	retq   
  400963:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096a:	00 00 00 
  40096d:	0f 1f 00             	nopl   (%rax)

0000000000400970 <_ZN9TestClassC2Ev>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 83 ec 10          	sub    $0x10,%rsp
  400978:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400980:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400984:	e8 67 01 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  400989:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40098d:	e8 6e 01 00 00       	callq  400b00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400992:	48 83 c4 10          	add    $0x10,%rsp
  400996:	5d                   	pop    %rbp
  400997:	c3                   	retq   
  400998:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40099f:	00 

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a8:	5d                   	pop    %rbp
  4009a9:	c3                   	retq   
  4009aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 83 ec 30          	sub    $0x30,%rsp
  4009b8:	48 89 f8             	mov    %rdi,%rax
  4009bb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  4009bf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  4009c2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4009c6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4009c9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4009cd:	48 89 f7             	mov    %rsi,%rdi
  4009d0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4009d4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  4009d7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4009db:	e8 80 01 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009e0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  4009e3:	89 10                	mov    %edx,(%rax)
  4009e5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4009e9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  4009ed:	e8 9e 01 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  4009f2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4009f6:	48 83 c4 30          	add    $0x30,%rsp
  4009fa:	5d                   	pop    %rbp
  4009fb:	c3                   	retq   
  4009fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 30          	sub    $0x30,%rsp
  400a08:	48 89 f8             	mov    %rdi,%rax
  400a0b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a0f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a12:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a16:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a19:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a1d:	48 89 f7             	mov    %rsi,%rdi
  400a20:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a24:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a27:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a2b:	e8 c0 01 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a33:	89 10                	mov    %edx,(%rax)
  400a35:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a39:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a3d:	e8 de 01 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400a42:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a46:	48 83 c4 30          	add    $0x30,%rsp
  400a4a:	5d                   	pop    %rbp
  400a4b:	c3                   	retq   
  400a4c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a50 <_ZN9TestClass15increase_field0Ev>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 83 ec 10          	sub    $0x10,%rsp
  400a58:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400a5c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a61:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a65:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a69:	48 89 c7             	mov    %rax,%rdi
  400a6c:	e8 9f 00 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400a71:	48 83 c4 10          	add    $0x10,%rsp
  400a75:	5d                   	pop    %rbp
  400a76:	c3                   	retq   
  400a77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a7e:	00 00 

0000000000400a80 <_ZN9TestClass15increase_field1Ev>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 83 ec 10          	sub    $0x10,%rsp
  400a88:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400a8c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a91:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a95:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a99:	48 89 c7             	mov    %rax,%rdi
  400a9c:	e8 ff 00 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400aa1:	48 83 c4 10          	add    $0x10,%rsp
  400aa5:	5d                   	pop    %rbp
  400aa6:	c3                   	retq   
  400aa7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400aae:	00 00 

0000000000400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400ab0:	55                   	push   %rbp
  400ab1:	48 89 e5             	mov    %rsp,%rbp
  400ab4:	48 83 ec 10          	sub    $0x10,%rsp
  400ab8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400abc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ac0:	e8 9b 00 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac5:	48 83 c4 10          	add    $0x10,%rsp
  400ac9:	5d                   	pop    %rbp
  400aca:	c3                   	retq   
  400acb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 83 ec 10          	sub    $0x10,%rsp
  400ad8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400adc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ae0:	e8 0b 01 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae5:	48 83 c4 10          	add    $0x10,%rsp
  400ae9:	5d                   	pop    %rbp
  400aea:	c3                   	retq   
  400aeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400af0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af8:	5d                   	pop    %rbp
  400af9:	c3                   	retq   
  400afa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b08:	5d                   	pop    %rbp
  400b09:	c3                   	retq   
  400b0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b10:	55                   	push   %rbp
  400b11:	48 89 e5             	mov    %rsp,%rbp
  400b14:	48 83 ec 30          	sub    $0x30,%rsp
  400b18:	48 89 f8             	mov    %rdi,%rax
  400b1b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b1f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b22:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b26:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b29:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b2d:	48 89 f7             	mov    %rsi,%rdi
  400b30:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b34:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b37:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b3b:	e8 20 00 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b40:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b43:	0f af 10             	imul   (%rax),%edx
  400b46:	89 10                	mov    %edx,(%rax)
  400b48:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b4c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b50:	e8 3b 00 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b55:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b59:	48 83 c4 30          	add    $0x30,%rsp
  400b5d:	5d                   	pop    %rbp
  400b5e:	c3                   	retq   
  400b5f:	90                   	nop

0000000000400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 b8 50 20 60 00 00 	movabs $0x602050,%rax
  400b6b:	00 00 00 
  400b6e:	48 83 c0 08          	add    $0x8,%rax
  400b72:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b76:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b7a:	48 c1 e7 02          	shl    $0x2,%rdi
  400b7e:	48 01 f8             	add    %rdi,%rax
  400b81:	5d                   	pop    %rbp
  400b82:	c3                   	retq   
  400b83:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8a:	00 00 00 
  400b8d:	0f 1f 00             	nopl   (%rax)

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b98:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b9c:	5d                   	pop    %rbp
  400b9d:	c3                   	retq   
  400b9e:	66 90                	xchg   %ax,%ax

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 83 ec 30          	sub    $0x30,%rsp
  400ba8:	48 89 f8             	mov    %rdi,%rax
  400bab:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400baf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400bb2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400bb6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400bb9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bbd:	48 89 f7             	mov    %rsi,%rdi
  400bc0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400bc4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400bc7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400bcb:	e8 20 00 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400bd3:	0f af 10             	imul   (%rax),%edx
  400bd6:	89 10                	mov    %edx,(%rax)
  400bd8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400bdc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400be0:	e8 3b 00 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400be5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400be9:	48 83 c4 30          	add    $0x30,%rsp
  400bed:	5d                   	pop    %rbp
  400bee:	c3                   	retq   
  400bef:	90                   	nop

0000000000400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 b8 50 20 60 00 00 	movabs $0x602050,%rax
  400bfb:	00 00 00 
  400bfe:	48 83 c0 08          	add    $0x8,%rax
  400c02:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c06:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c0a:	48 c1 e7 02          	shl    $0x2,%rdi
  400c0e:	48 01 f8             	add    %rdi,%rax
  400c11:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c17:	5d                   	pop    %rbp
  400c18:	c3                   	retq   
  400c19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400c20:	55                   	push   %rbp
  400c21:	48 89 e5             	mov    %rsp,%rbp
  400c24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c2c:	5d                   	pop    %rbp
  400c2d:	c3                   	retq   
  400c2e:	66 90                	xchg   %ax,%ax

0000000000400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c30:	55                   	push   %rbp
  400c31:	48 89 e5             	mov    %rsp,%rbp
  400c34:	5d                   	pop    %rbp
  400c35:	c3                   	retq   
  400c36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c3d:	00 00 00 

0000000000400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400c40:	55                   	push   %rbp
  400c41:	48 89 e5             	mov    %rsp,%rbp
  400c44:	48 83 ec 10          	sub    $0x10,%rsp
  400c48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c50:	e8 0b 00 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400c55:	48 83 c4 10          	add    $0x10,%rsp
  400c59:	5d                   	pop    %rbp
  400c5a:	c3                   	retq   
  400c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c6c:	5d                   	pop    %rbp
  400c6d:	c3                   	retq   
  400c6e:	66 90                	xchg   %ax,%ax

0000000000400c70 <__libc_csu_init>:
  400c70:	41 57                	push   %r15
  400c72:	41 56                	push   %r14
  400c74:	41 89 ff             	mov    %edi,%r15d
  400c77:	41 55                	push   %r13
  400c79:	41 54                	push   %r12
  400c7b:	4c 8d 25 5e 11 20 00 	lea    0x20115e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400c82:	55                   	push   %rbp
  400c83:	48 8d 2d 5e 11 20 00 	lea    0x20115e(%rip),%rbp        # 601de8 <__init_array_end>
  400c8a:	53                   	push   %rbx
  400c8b:	49 89 f6             	mov    %rsi,%r14
  400c8e:	49 89 d5             	mov    %rdx,%r13
  400c91:	4c 29 e5             	sub    %r12,%rbp
  400c94:	48 83 ec 08          	sub    $0x8,%rsp
  400c98:	48 c1 fd 03          	sar    $0x3,%rbp
  400c9c:	e8 cf f8 ff ff       	callq  400570 <_init>
  400ca1:	48 85 ed             	test   %rbp,%rbp
  400ca4:	74 20                	je     400cc6 <__libc_csu_init+0x56>
  400ca6:	31 db                	xor    %ebx,%ebx
  400ca8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400caf:	00 
  400cb0:	4c 89 ea             	mov    %r13,%rdx
  400cb3:	4c 89 f6             	mov    %r14,%rsi
  400cb6:	44 89 ff             	mov    %r15d,%edi
  400cb9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400cbd:	48 83 c3 01          	add    $0x1,%rbx
  400cc1:	48 39 eb             	cmp    %rbp,%rbx
  400cc4:	75 ea                	jne    400cb0 <__libc_csu_init+0x40>
  400cc6:	48 83 c4 08          	add    $0x8,%rsp
  400cca:	5b                   	pop    %rbx
  400ccb:	5d                   	pop    %rbp
  400ccc:	41 5c                	pop    %r12
  400cce:	41 5d                	pop    %r13
  400cd0:	41 5e                	pop    %r14
  400cd2:	41 5f                	pop    %r15
  400cd4:	c3                   	retq   
  400cd5:	90                   	nop
  400cd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cdd:	00 00 00 

0000000000400ce0 <__libc_csu_fini>:
  400ce0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ce4 <_fini>:
  400ce4:	48 83 ec 08          	sub    $0x8,%rsp
  400ce8:	48 83 c4 08          	add    $0x8,%rsp
  400cec:	c3                   	retq   
