
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
  4005ff:	49 c7 c0 20 0e 40 00 	mov    $0x400e20,%r8
  400606:	48 c7 c1 b0 0d 40 00 	mov    $0x400db0,%rcx
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
  40070f:	e8 3c 02 00 00       	callq  400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400714:	31 c0                	xor    %eax,%eax
  400716:	89 c7                	mov    %eax,%edi
  400718:	e8 53 02 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40071d:	48 89 c7             	mov    %rax,%rdi
  400720:	48 89 c1             	mov    %rax,%rcx
  400723:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400727:	48 89 c7             	mov    %rax,%rdi
  40072a:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
  40072e:	e8 7d 02 00 00       	callq  4009b0 <_ZN9TestClassC2Ev>
  400733:	e9 00 00 00 00       	jmpq   400738 <main+0x38>
  400738:	be 77 77 00 00       	mov    $0x7777,%esi
  40073d:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  400741:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400745:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400749:	e8 a2 02 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40074e:	be 88 88 00 00       	mov    $0x8888,%esi
  400753:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400757:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  40075b:	e8 d0 02 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400760:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400764:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  400768:	e8 03 03 00 00       	callq  400a70 <_ZN9TestClass15increase_field0Ev>
  40076d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400771:	e8 2a 03 00 00       	callq  400aa0 <_ZN9TestClass15increase_field1Ev>
  400776:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40077a:	e8 51 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40077f:	8b 30                	mov    (%rax),%esi
  400781:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400785:	89 75 bc             	mov    %esi,-0x44(%rbp)
  400788:	e8 63 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40078d:	48 bf 34 0e 40 00 00 	movabs $0x400e34,%rdi
  400794:	00 00 00 
  400797:	8b 10                	mov    (%rax),%edx
  400799:	8b 75 bc             	mov    -0x44(%rbp),%esi
  40079c:	b0 00                	mov    $0x0,%al
  40079e:	e8 fd fd ff ff       	callq  4005a0 <printf@plt>
  4007a3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007a7:	89 45 b8             	mov    %eax,-0x48(%rbp)
  4007aa:	e8 21 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007af:	31 d2                	xor    %edx,%edx
  4007b1:	88 d1                	mov    %dl,%cl
  4007b3:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007b9:	88 4d b7             	mov    %cl,-0x49(%rbp)
  4007bc:	0f 85 15 00 00 00    	jne    4007d7 <main+0xd7>
  4007c2:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c6:	e8 25 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007cb:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007d1:	0f 94 c1             	sete   %cl
  4007d4:	88 4d b7             	mov    %cl,-0x49(%rbp)
  4007d7:	8a 45 b7             	mov    -0x49(%rbp),%al
  4007da:	34 ff                	xor    $0xff,%al
  4007dc:	24 01                	and    $0x1,%al
  4007de:	0f b6 c0             	movzbl %al,%eax
  4007e1:	48 83 c4 50          	add    $0x50,%rsp
  4007e5:	5d                   	pop    %rbp
  4007e6:	c3                   	retq   
  4007e7:	89 d1                	mov    %edx,%ecx
  4007e9:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007ed:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4007f0:	48 8b 7d d8          	mov    -0x28(%rbp),%rdi
  4007f4:	e8 e7 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
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
  400832:	e8 79 01 00 00       	callq  4009b0 <_ZN9TestClassC2Ev>
  400837:	e9 00 00 00 00       	jmpq   40083c <_Z12new_instancev+0x2c>
  40083c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400840:	48 83 c4 20          	add    $0x20,%rsp
  400844:	5d                   	pop    %rbp
  400845:	c3                   	retq   
  400846:	89 d1                	mov    %edx,%ecx
  400848:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40084c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40084f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400853:	e8 88 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400858:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40085c:	e8 6f fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400861:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400868:	0f 1f 84 00 00 00 00 
  40086f:	00 

0000000000400870 <_Z12write_field0P9TestClass>:
  400870:	55                   	push   %rbp
  400871:	48 89 e5             	mov    %rsp,%rbp
  400874:	48 83 ec 10          	sub    $0x10,%rsp
  400878:	be 77 77 00 00       	mov    $0x7777,%esi
  40087d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400881:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400885:	e8 66 01 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40088a:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40088e:	48 83 c4 10          	add    $0x10,%rsp
  400892:	5d                   	pop    %rbp
  400893:	c3                   	retq   
  400894:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40089b:	00 00 00 00 00 

00000000004008a0 <_Z11read_field0P9TestClass>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 83 ec 10          	sub    $0x10,%rsp
  4008a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008b0:	e8 1b 02 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4008b5:	8b 00                	mov    (%rax),%eax
  4008b7:	48 83 c4 10          	add    $0x10,%rsp
  4008bb:	5d                   	pop    %rbp
  4008bc:	c3                   	retq   
  4008bd:	0f 1f 00             	nopl   (%rax)

00000000004008c0 <_Z12write_field1P9TestClass>:
  4008c0:	55                   	push   %rbp
  4008c1:	48 89 e5             	mov    %rsp,%rbp
  4008c4:	48 83 ec 10          	sub    $0x10,%rsp
  4008c8:	be 77 77 00 00       	mov    $0x7777,%esi
  4008cd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008d5:	e8 56 01 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4008da:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4008de:	48 83 c4 10          	add    $0x10,%rsp
  4008e2:	5d                   	pop    %rbp
  4008e3:	c3                   	retq   
  4008e4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008eb:	00 00 00 00 00 

00000000004008f0 <_Z11read_field1P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400900:	e8 eb 01 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
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
  40091c:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400923:	77 77 00 00 
  400927:	5d                   	pop    %rbp
  400928:	c3                   	retq   
  400929:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400930 <_Z20explicit_read_field0m>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400938:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40093c:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400943:	5d                   	pop    %rbp
  400944:	c3                   	retq   
  400945:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094c:	00 00 00 
  40094f:	90                   	nop

0000000000400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	e8 97 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400959:	48 89 c7             	mov    %rax,%rdi
  40095c:	e8 5f 03 00 00       	callq  400cc0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
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
  40097c:	e8 bf 03 00 00       	callq  400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400981:	e8 6a fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400986:	b9 01 00 00 00       	mov    $0x1,%ecx
  40098b:	89 ce                	mov    %ecx,%esi
  40098d:	48 89 c7             	mov    %rax,%rdi
  400990:	e8 db 03 00 00       	callq  400d70 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400995:	48 89 c7             	mov    %rax,%rdi
  400998:	e8 b3 03 00 00       	callq  400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  40099d:	48 83 c4 10          	add    $0x10,%rsp
  4009a1:	5d                   	pop    %rbp
  4009a2:	c3                   	retq   
  4009a3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009aa:	00 00 00 
  4009ad:	0f 1f 00             	nopl   (%rax)

00000000004009b0 <_ZN9TestClassC2Ev>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 83 ec 10          	sub    $0x10,%rsp
  4009b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009c0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009c4:	e8 47 01 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009c9:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4009cd:	e8 4e 01 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  4009d2:	48 83 c4 10          	add    $0x10,%rsp
  4009d6:	5d                   	pop    %rbp
  4009d7:	c3                   	retq   
  4009d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009df:	00 

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e8:	5d                   	pop    %rbp
  4009e9:	c3                   	retq   
  4009ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 83 ec 20          	sub    $0x20,%rsp
  4009f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fc:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4009ff:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a03:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400a06:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a0a:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400a0d:	e8 6e 01 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a12:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400a15:	89 30                	mov    %esi,(%rax)
  400a17:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a1b:	48 83 c4 20          	add    $0x20,%rsp
  400a1f:	5d                   	pop    %rbp
  400a20:	c3                   	retq   
  400a21:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a28:	00 00 00 
  400a2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 83 ec 20          	sub    $0x20,%rsp
  400a38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a3f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a43:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400a46:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a4a:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400a4d:	e8 ee 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a52:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400a55:	89 30                	mov    %esi,(%rax)
  400a57:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a5b:	48 83 c4 20          	add    $0x20,%rsp
  400a5f:	5d                   	pop    %rbp
  400a60:	c3                   	retq   
  400a61:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a68:	00 00 00 
  400a6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400a70 <_ZN9TestClass15increase_field0Ev>:
  400a70:	55                   	push   %rbp
  400a71:	48 89 e5             	mov    %rsp,%rbp
  400a74:	48 83 ec 10          	sub    $0x10,%rsp
  400a78:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400a7c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a81:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a85:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a89:	48 89 c7             	mov    %rax,%rdi
  400a8c:	e8 9f 00 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400a91:	48 83 c4 10          	add    $0x10,%rsp
  400a95:	5d                   	pop    %rbp
  400a96:	c3                   	retq   
  400a97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a9e:	00 00 

0000000000400aa0 <_ZN9TestClass15increase_field1Ev>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 83 ec 10          	sub    $0x10,%rsp
  400aa8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400aac:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ab1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ab9:	48 89 c7             	mov    %rax,%rdi
  400abc:	e8 2f 01 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400ac1:	48 83 c4 10          	add    $0x10,%rsp
  400ac5:	5d                   	pop    %rbp
  400ac6:	c3                   	retq   
  400ac7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400ace:	00 00 

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 83 ec 10          	sub    $0x10,%rsp
  400ad8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400adc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ae0:	e8 9b 00 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae5:	48 83 c4 10          	add    $0x10,%rsp
  400ae9:	5d                   	pop    %rbp
  400aea:	c3                   	retq   
  400aeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 83 ec 10          	sub    $0x10,%rsp
  400af8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b00:	e8 3b 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b05:	48 83 c4 10          	add    $0x10,%rsp
  400b09:	5d                   	pop    %rbp
  400b0a:	c3                   	retq   
  400b0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b10:	55                   	push   %rbp
  400b11:	48 89 e5             	mov    %rsp,%rbp
  400b14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b18:	5d                   	pop    %rbp
  400b19:	c3                   	retq   
  400b1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b28:	5d                   	pop    %rbp
  400b29:	c3                   	retq   
  400b2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 83 ec 30          	sub    $0x30,%rsp
  400b38:	48 89 f8             	mov    %rdi,%rax
  400b3b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b3f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b42:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b46:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b49:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b4d:	48 89 f7             	mov    %rsi,%rdi
  400b50:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b54:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b57:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b5b:	e8 20 00 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b60:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b63:	0f af 10             	imul   (%rax),%edx
  400b66:	89 10                	mov    %edx,(%rax)
  400b68:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b6c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b70:	e8 2b 00 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b75:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b79:	48 83 c4 30          	add    $0x30,%rsp
  400b7d:	5d                   	pop    %rbp
  400b7e:	c3                   	retq   
  400b7f:	90                   	nop

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 83 ec 10          	sub    $0x10,%rsp
  400b88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b90:	e8 1b 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b95:	48 83 c4 10          	add    $0x10,%rsp
  400b99:	5d                   	pop    %rbp
  400b9a:	c3                   	retq   
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ba8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bac:	5d                   	pop    %rbp
  400bad:	c3                   	retq   
  400bae:	66 90                	xchg   %ax,%ax

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400bbb:	00 00 00 
  400bbe:	48 89 c1             	mov    %rax,%rcx
  400bc1:	48 83 c1 10          	add    $0x10,%rcx
  400bc5:	48 89 ca             	mov    %rcx,%rdx
  400bc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bcc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bd0:	48 c7 45 f0 10 00 00 	movq   $0x10,-0x10(%rbp)
  400bd7:	00 
  400bd8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bdc:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400be0:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  400be4:	48 c1 e7 02          	shl    $0x2,%rdi
  400be8:	48 01 fa             	add    %rdi,%rdx
  400beb:	48 89 d0             	mov    %rdx,%rax
  400bee:	5d                   	pop    %rbp
  400bef:	c3                   	retq   

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
  400c74:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c7b:	00 00 00 
  400c7e:	48 89 c1             	mov    %rax,%rcx
  400c81:	48 81 c1 e0 48 00 00 	add    $0x48e0,%rcx
  400c88:	48 89 ca             	mov    %rcx,%rdx
  400c8b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c8f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c93:	48 c7 45 f0 10 00 00 	movq   $0x10,-0x10(%rbp)
  400c9a:	00 
  400c9b:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c9f:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400ca3:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
  400ca7:	48 c1 e7 02          	shl    $0x2,%rdi
  400cab:	48 01 fa             	add    %rdi,%rdx
  400cae:	48 89 d0             	mov    %rdx,%rax
  400cb1:	5d                   	pop    %rbp
  400cb2:	c3                   	retq   
  400cb3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cba:	00 00 00 
  400cbd:	0f 1f 00             	nopl   (%rax)

0000000000400cc0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 83 ec 10          	sub    $0x10,%rsp
  400cc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ccc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cd0:	48 89 f8             	mov    %rdi,%rax
  400cd3:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400cd7:	48 89 c7             	mov    %rax,%rdi
  400cda:	e8 21 00 00 00       	callq  400d00 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400cdf:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ce3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400cea:	00 
  400ceb:	48 83 c4 10          	add    $0x10,%rsp
  400cef:	5d                   	pop    %rbp
  400cf0:	c3                   	retq   
  400cf1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cf8:	00 00 00 
  400cfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d00 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 10          	sub    $0x10,%rsp
  400d08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d10:	e8 0b 00 00 00       	callq  400d20 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d15:	48 83 c4 10          	add    $0x10,%rsp
  400d19:	5d                   	pop    %rbp
  400d1a:	c3                   	retq   
  400d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d20 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d20:	55                   	push   %rbp
  400d21:	48 89 e5             	mov    %rsp,%rbp
  400d24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d28:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d2c:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d33:	5d                   	pop    %rbp
  400d34:	c3                   	retq   
  400d35:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3c:	00 00 00 
  400d3f:	90                   	nop

0000000000400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d40:	55                   	push   %rbp
  400d41:	48 89 e5             	mov    %rsp,%rbp
  400d44:	5d                   	pop    %rbp
  400d45:	c3                   	retq   
  400d46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d4d:	00 00 00 

0000000000400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	48 83 ec 10          	sub    $0x10,%rsp
  400d58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d60:	48 83 c7 01          	add    $0x1,%rdi
  400d64:	e8 37 00 00 00       	callq  400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d69:	48 83 c4 10          	add    $0x10,%rsp
  400d6d:	5d                   	pop    %rbp
  400d6e:	c3                   	retq   
  400d6f:	90                   	nop

0000000000400d70 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400d70:	55                   	push   %rbp
  400d71:	48 89 e5             	mov    %rsp,%rbp
  400d74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d78:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d7c:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400d80:	48 8b 3e             	mov    (%rsi),%rdi
  400d83:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400d87:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400d8b:	48 03 3e             	add    (%rsi),%rdi
  400d8e:	48 89 3e             	mov    %rdi,(%rsi)
  400d91:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400d95:	5d                   	pop    %rbp
  400d96:	c3                   	retq   
  400d97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d9e:	00 00 

0000000000400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400da0:	55                   	push   %rbp
  400da1:	48 89 e5             	mov    %rsp,%rbp
  400da4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400da8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400dac:	5d                   	pop    %rbp
  400dad:	c3                   	retq   
  400dae:	66 90                	xchg   %ax,%ax

0000000000400db0 <__libc_csu_init>:
  400db0:	41 57                	push   %r15
  400db2:	41 56                	push   %r14
  400db4:	41 89 ff             	mov    %edi,%r15d
  400db7:	41 55                	push   %r13
  400db9:	41 54                	push   %r12
  400dbb:	4c 8d 25 1e 10 20 00 	lea    0x20101e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400dc2:	55                   	push   %rbp
  400dc3:	48 8d 2d 1e 10 20 00 	lea    0x20101e(%rip),%rbp        # 601de8 <__init_array_end>
  400dca:	53                   	push   %rbx
  400dcb:	49 89 f6             	mov    %rsi,%r14
  400dce:	49 89 d5             	mov    %rdx,%r13
  400dd1:	4c 29 e5             	sub    %r12,%rbp
  400dd4:	48 83 ec 08          	sub    $0x8,%rsp
  400dd8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ddc:	e8 8f f7 ff ff       	callq  400570 <_init>
  400de1:	48 85 ed             	test   %rbp,%rbp
  400de4:	74 20                	je     400e06 <__libc_csu_init+0x56>
  400de6:	31 db                	xor    %ebx,%ebx
  400de8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400def:	00 
  400df0:	4c 89 ea             	mov    %r13,%rdx
  400df3:	4c 89 f6             	mov    %r14,%rsi
  400df6:	44 89 ff             	mov    %r15d,%edi
  400df9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dfd:	48 83 c3 01          	add    $0x1,%rbx
  400e01:	48 39 eb             	cmp    %rbp,%rbx
  400e04:	75 ea                	jne    400df0 <__libc_csu_init+0x40>
  400e06:	48 83 c4 08          	add    $0x8,%rsp
  400e0a:	5b                   	pop    %rbx
  400e0b:	5d                   	pop    %rbp
  400e0c:	41 5c                	pop    %r12
  400e0e:	41 5d                	pop    %r13
  400e10:	41 5e                	pop    %r14
  400e12:	41 5f                	pop    %r15
  400e14:	c3                   	retq   
  400e15:	90                   	nop
  400e16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e1d:	00 00 00 

0000000000400e20 <__libc_csu_fini>:
  400e20:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e24 <_fini>:
  400e24:	48 83 ec 08          	sub    $0x8,%rsp
  400e28:	48 83 c4 08          	add    $0x8,%rsp
  400e2c:	c3                   	retq   
