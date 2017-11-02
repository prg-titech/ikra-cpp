
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
  4005ff:	49 c7 c0 80 0d 40 00 	mov    $0x400d80,%r8
  400606:	48 c7 c1 10 0d 40 00 	mov    $0x400d10,%rcx
  40060d:	48 c7 c7 00 07 40 00 	mov    $0x400700,%rdi
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
  4006a0:	80 3d a9 19 20 00 00 	cmpb   $0x0,0x2019a9(%rip)        # 602050 <completed.7585>
  4006a7:	75 11                	jne    4006ba <__do_global_dtors_aux+0x1a>
  4006a9:	55                   	push   %rbp
  4006aa:	48 89 e5             	mov    %rsp,%rbp
  4006ad:	e8 6e ff ff ff       	callq  400620 <deregister_tm_clones>
  4006b2:	5d                   	pop    %rbp
  4006b3:	c6 05 96 19 20 00 01 	movb   $0x1,0x201996(%rip)        # 602050 <completed.7585>
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

00000000004006f0 <_ZN9TestClass7storageEv>:
  4006f0:	55                   	push   %rbp
  4006f1:	48 89 e5             	mov    %rsp,%rbp
  4006f4:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  4006fb:	00 00 00 
  4006fe:	5d                   	pop    %rbp
  4006ff:	c3                   	retq   

0000000000400700 <main>:
  400700:	55                   	push   %rbp
  400701:	48 89 e5             	mov    %rsp,%rbp
  400704:	48 83 ec 50          	sub    $0x50,%rsp
  400708:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  40070f:	e8 3c 02 00 00       	callq  400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageEv>
  400714:	31 c0                	xor    %eax,%eax
  400716:	89 c7                	mov    %eax,%edi
  400718:	e8 53 02 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40071d:	48 89 c7             	mov    %rax,%rdi
  400720:	48 89 c1             	mov    %rax,%rcx
  400723:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400727:	48 89 c7             	mov    %rax,%rdi
  40072a:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  40072e:	e8 6d 02 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  400733:	e9 00 00 00 00       	jmpq   400738 <main+0x38>
  400738:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  40073c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400741:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400745:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400749:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40074d:	e8 8e 02 00 00       	callq  4009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400752:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400756:	ba 88 88 00 00       	mov    $0x8888,%edx
  40075b:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40075f:	e8 cc 02 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400764:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400768:	e8 13 03 00 00       	callq  400a80 <_ZN9TestClass15increase_field0Ev>
  40076d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400771:	e8 3a 03 00 00       	callq  400ab0 <_ZN9TestClass15increase_field1Ev>
  400776:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40077a:	e8 61 03 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40077f:	8b 30                	mov    (%rax),%esi
  400781:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400785:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400788:	e8 73 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40078d:	48 bf 94 0d 40 00 00 	movabs $0x400d94,%rdi
  400794:	00 00 00 
  400797:	8b 10                	mov    (%rax),%edx
  400799:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  40079c:	b0 00                	mov    $0x0,%al
  40079e:	e8 fd fd ff ff       	callq  4005a0 <printf@plt>
  4007a3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007a7:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007aa:	e8 31 03 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007af:	31 d2                	xor    %edx,%edx
  4007b1:	88 d1                	mov    %dl,%cl
  4007b3:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007b9:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007bc:	0f 85 15 00 00 00    	jne    4007d7 <main+0xd7>
  4007c2:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c6:	e8 35 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007cb:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007d1:	0f 94 c1             	sete   %cl
  4007d4:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007d7:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007da:	34 ff                	xor    $0xff,%al
  4007dc:	24 01                	and    $0x1,%al
  4007de:	0f b6 c0             	movzbl %al,%eax
  4007e1:	48 83 c4 50          	add    $0x50,%rsp
  4007e5:	5d                   	pop    %rbp
  4007e6:	c3                   	retq   
  4007e7:	89 d1                	mov    %edx,%ecx
  4007e9:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007ed:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4007f0:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  4007f4:	e8 d7 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  4007f9:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4007fd:	e8 ce fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400802:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400809:	1f 84 00 00 00 00 00 

0000000000400810 <_Z12new_instancev>:
  400810:	55                   	push   %rbp
  400811:	48 89 e5             	mov    %rsp,%rbp
  400814:	48 83 ec 20          	sub    $0x20,%rsp
  400818:	31 c0                	xor    %eax,%eax
  40081a:	89 c7                	mov    %eax,%edi
  40081c:	e8 4f 01 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400821:	48 89 c7             	mov    %rax,%rdi
  400824:	48 89 c1             	mov    %rax,%rcx
  400827:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40082b:	48 89 c7             	mov    %rax,%rdi
  40082e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400832:	e8 69 01 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  400837:	e9 00 00 00 00       	jmpq   40083c <_Z12new_instancev+0x2c>
  40083c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400840:	48 83 c4 20          	add    $0x20,%rsp
  400844:	5d                   	pop    %rbp
  400845:	c3                   	retq   
  400846:	89 d1                	mov    %edx,%ecx
  400848:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40084c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40084f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400853:	e8 78 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400858:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40085c:	e8 6f fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400861:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400868:	0f 1f 84 00 00 00 00 
  40086f:	00 

0000000000400870 <_Z12write_field0P9TestClass>:
  400870:	55                   	push   %rbp
  400871:	48 89 e5             	mov    %rsp,%rbp
  400874:	48 83 ec 10          	sub    $0x10,%rsp
  400878:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40087c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400881:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400885:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400889:	48 89 c7             	mov    %rax,%rdi
  40088c:	e8 4f 01 00 00       	callq  4009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400891:	48 83 c4 10          	add    $0x10,%rsp
  400895:	5d                   	pop    %rbp
  400896:	c3                   	retq   
  400897:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40089e:	00 00 

00000000004008a0 <_Z11read_field0P9TestClass>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 83 ec 10          	sub    $0x10,%rsp
  4008a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008b0:	e8 2b 02 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4008b5:	8b 00                	mov    (%rax),%eax
  4008b7:	48 83 c4 10          	add    $0x10,%rsp
  4008bb:	5d                   	pop    %rbp
  4008bc:	c3                   	retq   
  4008bd:	0f 1f 00             	nopl   (%rax)

00000000004008c0 <_Z12write_field1P9TestClass>:
  4008c0:	55                   	push   %rbp
  4008c1:	48 89 e5             	mov    %rsp,%rbp
  4008c4:	48 83 ec 10          	sub    $0x10,%rsp
  4008c8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008cc:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008d1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008d9:	48 89 c7             	mov    %rax,%rdi
  4008dc:	e8 4f 01 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4008e1:	48 83 c4 10          	add    $0x10,%rsp
  4008e5:	5d                   	pop    %rbp
  4008e6:	c3                   	retq   
  4008e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008ee:	00 00 

00000000004008f0 <_Z11read_field1P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400900:	e8 fb 01 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400905:	8b 00                	mov    (%rax),%eax
  400907:	48 83 c4 10          	add    $0x10,%rsp
  40090b:	5d                   	pop    %rbp
  40090c:	c3                   	retq   
  40090d:	0f 1f 00             	nopl   (%rax)

0000000000400910 <_Z21explicit_write_field0m>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400918:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40091c:	c7 04 bd 10 b2 60 00 	movl   $0x7777,0x60b210(,%rdi,4)
  400923:	77 77 00 00 
  400927:	5d                   	pop    %rbp
  400928:	c3                   	retq   
  400929:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400930 <_Z20explicit_read_field0m>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400938:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40093c:	8b 04 bd 10 b2 60 00 	mov    0x60b210(,%rdi,4),%eax
  400943:	5d                   	pop    %rbp
  400944:	c3                   	retq   
  400945:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094c:	00 00 00 
  40094f:	90                   	nop

0000000000400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageEv>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	e8 97 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400959:	48 89 c7             	mov    %rax,%rdi
  40095c:	e8 4f 03 00 00       	callq  400cb0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  400961:	5d                   	pop    %rbp
  400962:	c3                   	retq   
  400963:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096a:	00 00 00 
  40096d:	0f 1f 00             	nopl   (%rax)

0000000000400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 83 ec 10          	sub    $0x10,%rsp
  400978:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097c:	e8 4f 03 00 00       	callq  400cd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400981:	e8 6a fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400986:	48 8b 38             	mov    (%rax),%rdi
  400989:	48 89 f9             	mov    %rdi,%rcx
  40098c:	48 83 c1 01          	add    $0x1,%rcx
  400990:	48 89 08             	mov    %rcx,(%rax)
  400993:	e8 48 03 00 00       	callq  400ce0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400998:	48 83 c4 10          	add    $0x10,%rsp
  40099c:	5d                   	pop    %rbp
  40099d:	c3                   	retq   
  40099e:	66 90                	xchg   %ax,%ax

00000000004009a0 <_ZN9TestClassC2Ev>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 83 ec 10          	sub    $0x10,%rsp
  4009a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009b0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009b4:	e8 67 01 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009b9:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4009bd:	e8 6e 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  4009c2:	48 83 c4 10          	add    $0x10,%rsp
  4009c6:	5d                   	pop    %rbp
  4009c7:	c3                   	retq   
  4009c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009cf:	00 

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d8:	5d                   	pop    %rbp
  4009d9:	c3                   	retq   
  4009da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 83 ec 30          	sub    $0x30,%rsp
  4009e8:	48 89 f8             	mov    %rdi,%rax
  4009eb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  4009ef:	89 55 f4             	mov    %edx,-0xc(%rbp)
  4009f2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4009f6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4009f9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4009fd:	48 89 f7             	mov    %rsi,%rdi
  400a00:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a04:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a07:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a0b:	e8 80 01 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a10:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a13:	89 10                	mov    %edx,(%rax)
  400a15:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a19:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a1d:	e8 8e 01 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400a22:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a26:	48 83 c4 30          	add    $0x30,%rsp
  400a2a:	5d                   	pop    %rbp
  400a2b:	c3                   	retq   
  400a2c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 83 ec 30          	sub    $0x30,%rsp
  400a38:	48 89 f8             	mov    %rdi,%rax
  400a3b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a3f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a42:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a46:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a49:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a4d:	48 89 f7             	mov    %rsi,%rdi
  400a50:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a54:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a57:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a5b:	e8 e0 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a60:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a63:	89 10                	mov    %edx,(%rax)
  400a65:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a69:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a6d:	e8 ee 01 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400a72:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a76:	48 83 c4 30          	add    $0x30,%rsp
  400a7a:	5d                   	pop    %rbp
  400a7b:	c3                   	retq   
  400a7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a80 <_ZN9TestClass15increase_field0Ev>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 83 ec 10          	sub    $0x10,%rsp
  400a88:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400a8c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a91:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a95:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a99:	48 89 c7             	mov    %rax,%rdi
  400a9c:	e8 9f 00 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400aa1:	48 83 c4 10          	add    $0x10,%rsp
  400aa5:	5d                   	pop    %rbp
  400aa6:	c3                   	retq   
  400aa7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400aae:	00 00 

0000000000400ab0 <_ZN9TestClass15increase_field1Ev>:
  400ab0:	55                   	push   %rbp
  400ab1:	48 89 e5             	mov    %rsp,%rbp
  400ab4:	48 83 ec 10          	sub    $0x10,%rsp
  400ab8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400abc:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ac1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ac9:	48 89 c7             	mov    %rax,%rdi
  400acc:	e8 1f 01 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400ad1:	48 83 c4 10          	add    $0x10,%rsp
  400ad5:	5d                   	pop    %rbp
  400ad6:	c3                   	retq   
  400ad7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400ade:	00 00 

0000000000400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	48 83 ec 10          	sub    $0x10,%rsp
  400ae8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400af0:	e8 9b 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400af5:	48 83 c4 10          	add    $0x10,%rsp
  400af9:	5d                   	pop    %rbp
  400afa:	c3                   	retq   
  400afb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 10          	sub    $0x10,%rsp
  400b08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b10:	e8 2b 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b15:	48 83 c4 10          	add    $0x10,%rsp
  400b19:	5d                   	pop    %rbp
  400b1a:	c3                   	retq   
  400b1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b28:	5d                   	pop    %rbp
  400b29:	c3                   	retq   
  400b2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b38:	5d                   	pop    %rbp
  400b39:	c3                   	retq   
  400b3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 30          	sub    $0x30,%rsp
  400b48:	48 89 f8             	mov    %rdi,%rax
  400b4b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b4f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b52:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b56:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b59:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b5d:	48 89 f7             	mov    %rsi,%rdi
  400b60:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b64:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b67:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b6b:	e8 20 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b70:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b73:	0f af 10             	imul   (%rax),%edx
  400b76:	89 10                	mov    %edx,(%rax)
  400b78:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b7c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b80:	e8 2b 00 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b85:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b89:	48 83 c4 30          	add    $0x30,%rsp
  400b8d:	5d                   	pop    %rbp
  400b8e:	c3                   	retq   
  400b8f:	90                   	nop

0000000000400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 83 ec 10          	sub    $0x10,%rsp
  400b98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b9c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ba0:	e8 1b 00 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ba5:	48 83 c4 10          	add    $0x10,%rsp
  400ba9:	5d                   	pop    %rbp
  400baa:	c3                   	retq   
  400bab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bbc:	5d                   	pop    %rbp
  400bbd:	c3                   	retq   
  400bbe:	66 90                	xchg   %ax,%ax

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 83 ec 10          	sub    $0x10,%rsp
  400bc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bcc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bd0:	48 c1 e7 02          	shl    $0x2,%rdi
  400bd4:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400bd8:	e8 13 fb ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400bdd:	48 83 c0 08          	add    $0x8,%rax
  400be1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400be5:	48 01 f8             	add    %rdi,%rax
  400be8:	48 83 c4 10          	add    $0x10,%rsp
  400bec:	5d                   	pop    %rbp
  400bed:	c3                   	retq   
  400bee:	66 90                	xchg   %ax,%ax

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 83 ec 30          	sub    $0x30,%rsp
  400bf8:	48 89 f8             	mov    %rdi,%rax
  400bfb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bff:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c02:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c06:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c09:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c0d:	48 89 f7             	mov    %rsi,%rdi
  400c10:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c14:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c17:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c1b:	e8 20 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c20:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c23:	0f af 10             	imul   (%rax),%edx
  400c26:	89 10                	mov    %edx,(%rax)
  400c28:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c2c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c30:	e8 2b 00 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400c35:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c39:	48 83 c4 30          	add    $0x30,%rsp
  400c3d:	5d                   	pop    %rbp
  400c3e:	c3                   	retq   
  400c3f:	90                   	nop

0000000000400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c40:	55                   	push   %rbp
  400c41:	48 89 e5             	mov    %rsp,%rbp
  400c44:	48 83 ec 10          	sub    $0x10,%rsp
  400c48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c50:	e8 1b 00 00 00       	callq  400c70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c55:	48 83 c4 10          	add    $0x10,%rsp
  400c59:	5d                   	pop    %rbp
  400c5a:	c3                   	retq   
  400c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c6c:	5d                   	pop    %rbp
  400c6d:	c3                   	retq   
  400c6e:	66 90                	xchg   %ax,%ax

0000000000400c70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c70:	55                   	push   %rbp
  400c71:	48 89 e5             	mov    %rsp,%rbp
  400c74:	48 83 ec 10          	sub    $0x10,%rsp
  400c78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c80:	48 c1 e7 02          	shl    $0x2,%rdi
  400c84:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c88:	e8 63 fa ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400c8d:	48 83 c0 08          	add    $0x8,%rax
  400c91:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400c95:	48 01 f8             	add    %rdi,%rax
  400c98:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c9e:	48 83 c4 10          	add    $0x10,%rsp
  400ca2:	5d                   	pop    %rbp
  400ca3:	c3                   	retq   
  400ca4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cab:	00 00 00 
  400cae:	66 90                	xchg   %ax,%ax

0000000000400cb0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cb8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cbc:	48 c7 87 a8 91 00 00 	movq   $0x0,0x91a8(%rdi)
  400cc3:	00 00 00 00 
  400cc7:	5d                   	pop    %rbp
  400cc8:	c3                   	retq   
  400cc9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400cd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cd0:	55                   	push   %rbp
  400cd1:	48 89 e5             	mov    %rsp,%rbp
  400cd4:	5d                   	pop    %rbp
  400cd5:	c3                   	retq   
  400cd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cdd:	00 00 00 

0000000000400ce0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ce0:	55                   	push   %rbp
  400ce1:	48 89 e5             	mov    %rsp,%rbp
  400ce4:	48 83 ec 10          	sub    $0x10,%rsp
  400ce8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cf0:	48 83 c7 01          	add    $0x1,%rdi
  400cf4:	e8 07 00 00 00       	callq  400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400cf9:	48 83 c4 10          	add    $0x10,%rsp
  400cfd:	5d                   	pop    %rbp
  400cfe:	c3                   	retq   
  400cff:	90                   	nop

0000000000400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d08:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400d0c:	5d                   	pop    %rbp
  400d0d:	c3                   	retq   
  400d0e:	66 90                	xchg   %ax,%ax

0000000000400d10 <__libc_csu_init>:
  400d10:	41 57                	push   %r15
  400d12:	41 56                	push   %r14
  400d14:	41 89 ff             	mov    %edi,%r15d
  400d17:	41 55                	push   %r13
  400d19:	41 54                	push   %r12
  400d1b:	4c 8d 25 be 10 20 00 	lea    0x2010be(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d22:	55                   	push   %rbp
  400d23:	48 8d 2d be 10 20 00 	lea    0x2010be(%rip),%rbp        # 601de8 <__init_array_end>
  400d2a:	53                   	push   %rbx
  400d2b:	49 89 f6             	mov    %rsi,%r14
  400d2e:	49 89 d5             	mov    %rdx,%r13
  400d31:	4c 29 e5             	sub    %r12,%rbp
  400d34:	48 83 ec 08          	sub    $0x8,%rsp
  400d38:	48 c1 fd 03          	sar    $0x3,%rbp
  400d3c:	e8 2f f8 ff ff       	callq  400570 <_init>
  400d41:	48 85 ed             	test   %rbp,%rbp
  400d44:	74 20                	je     400d66 <__libc_csu_init+0x56>
  400d46:	31 db                	xor    %ebx,%ebx
  400d48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d4f:	00 
  400d50:	4c 89 ea             	mov    %r13,%rdx
  400d53:	4c 89 f6             	mov    %r14,%rsi
  400d56:	44 89 ff             	mov    %r15d,%edi
  400d59:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d5d:	48 83 c3 01          	add    $0x1,%rbx
  400d61:	48 39 eb             	cmp    %rbp,%rbx
  400d64:	75 ea                	jne    400d50 <__libc_csu_init+0x40>
  400d66:	48 83 c4 08          	add    $0x8,%rsp
  400d6a:	5b                   	pop    %rbx
  400d6b:	5d                   	pop    %rbp
  400d6c:	41 5c                	pop    %r12
  400d6e:	41 5d                	pop    %r13
  400d70:	41 5e                	pop    %r14
  400d72:	41 5f                	pop    %r15
  400d74:	c3                   	retq   
  400d75:	90                   	nop
  400d76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d7d:	00 00 00 

0000000000400d80 <__libc_csu_fini>:
  400d80:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d84 <_fini>:
  400d84:	48 83 ec 08          	sub    $0x8,%rsp
  400d88:	48 83 c4 08          	add    $0x8,%rsp
  400d8c:	c3                   	retq   
