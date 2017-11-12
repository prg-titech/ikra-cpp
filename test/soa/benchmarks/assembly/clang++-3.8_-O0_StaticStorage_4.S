
bin/clang++-3.8_-O0_StaticStorage_4:     file format elf64-x86-64


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
  4005ff:	49 c7 c0 10 0f 40 00 	mov    $0x400f10,%r8
  400606:	48 c7 c1 a0 0e 40 00 	mov    $0x400ea0,%rcx
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
  40070f:	e8 8c 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400714:	b8 04 00 00 00       	mov    $0x4,%eax
  400719:	89 c7                	mov    %eax,%edi
  40071b:	e8 a0 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400720:	48 89 c7             	mov    %rax,%rdi
  400723:	48 89 c1             	mov    %rax,%rcx
  400726:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40072c:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400730:	48 89 c7             	mov    %rax,%rdi
  400733:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
  400737:	e8 c4 02 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  40073c:	e9 00 00 00 00       	jmpq   400741 <main+0x41>
  400741:	be 77 77 00 00       	mov    $0x7777,%esi
  400746:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  40074a:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40074e:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400752:	48 83 c1 04          	add    $0x4,%rcx
  400756:	48 89 cf             	mov    %rcx,%rdi
  400759:	e8 f2 02 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40075e:	be 88 88 00 00       	mov    $0x8888,%esi
  400763:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400767:	48 83 c1 04          	add    $0x4,%rcx
  40076b:	48 89 cf             	mov    %rcx,%rdi
  40076e:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  400772:	e8 19 03 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400777:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40077b:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  40077f:	e8 4c 03 00 00       	callq  400ad0 <_ZN9TestClass15increase_field0Ev>
  400784:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400788:	e8 83 03 00 00       	callq  400b10 <_ZN9TestClass15increase_field1Ev>
  40078d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400791:	48 83 c0 04          	add    $0x4,%rax
  400795:	48 89 c7             	mov    %rax,%rdi
  400798:	e8 b3 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40079d:	8b 30                	mov    (%rax),%esi
  40079f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007a3:	48 83 c0 04          	add    $0x4,%rax
  4007a7:	48 89 c7             	mov    %rax,%rdi
  4007aa:	89 75 bc             	mov    %esi,-0x44(%rbp)
  4007ad:	e8 be 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007b2:	48 bf 24 0f 40 00 00 	movabs $0x400f24,%rdi
  4007b9:	00 00 00 
  4007bc:	8b 10                	mov    (%rax),%edx
  4007be:	8b 75 bc             	mov    -0x44(%rbp),%esi
  4007c1:	b0 00                	mov    $0x0,%al
  4007c3:	e8 d8 fd ff ff       	callq  4005a0 <printf@plt>
  4007c8:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007cc:	48 83 c1 04          	add    $0x4,%rcx
  4007d0:	48 89 cf             	mov    %rcx,%rdi
  4007d3:	89 45 b8             	mov    %eax,-0x48(%rbp)
  4007d6:	e8 75 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007db:	31 d2                	xor    %edx,%edx
  4007dd:	41 88 d0             	mov    %dl,%r8b
  4007e0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007e6:	44 88 45 b7          	mov    %r8b,-0x49(%rbp)
  4007ea:	0f 85 1c 00 00 00    	jne    40080c <main+0x10c>
  4007f0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007f4:	48 83 c0 04          	add    $0x4,%rax
  4007f8:	48 89 c7             	mov    %rax,%rdi
  4007fb:	e8 70 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400800:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400806:	0f 94 c1             	sete   %cl
  400809:	88 4d b7             	mov    %cl,-0x49(%rbp)
  40080c:	8a 45 b7             	mov    -0x49(%rbp),%al
  40080f:	34 ff                	xor    $0xff,%al
  400811:	24 01                	and    $0x1,%al
  400813:	0f b6 c0             	movzbl %al,%eax
  400816:	48 83 c4 50          	add    $0x50,%rsp
  40081a:	5d                   	pop    %rbp
  40081b:	c3                   	retq   
  40081c:	89 d1                	mov    %edx,%ecx
  40081e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400822:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400825:	48 8b 7d d8          	mov    -0x28(%rbp),%rdi
  400829:	e8 12 02 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  40082e:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400832:	e8 99 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400837:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40083e:	00 00 

0000000000400840 <_Z12new_instancev>:
  400840:	55                   	push   %rbp
  400841:	48 89 e5             	mov    %rsp,%rbp
  400844:	48 83 ec 20          	sub    $0x20,%rsp
  400848:	b8 04 00 00 00       	mov    $0x4,%eax
  40084d:	89 c7                	mov    %eax,%edi
  40084f:	e8 6c 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	48 89 c1             	mov    %rax,%rcx
  40085a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400860:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400864:	48 89 c7             	mov    %rax,%rdi
  400867:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40086b:	e8 90 01 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  400870:	e9 00 00 00 00       	jmpq   400875 <_Z12new_instancev+0x35>
  400875:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400879:	48 83 c4 20          	add    $0x20,%rsp
  40087d:	5d                   	pop    %rbp
  40087e:	c3                   	retq   
  40087f:	89 d1                	mov    %edx,%ecx
  400881:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400885:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  400888:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40088c:	e8 af 01 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400891:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400895:	e8 36 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40089a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008a0 <_Z12write_field0P9TestClass>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 83 ec 10          	sub    $0x10,%rsp
  4008a8:	be 77 77 00 00       	mov    $0x7777,%esi
  4008ad:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008b5:	48 83 c7 04          	add    $0x4,%rdi
  4008b9:	e8 92 01 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4008be:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4008c2:	48 83 c4 10          	add    $0x10,%rsp
  4008c6:	5d                   	pop    %rbp
  4008c7:	c3                   	retq   
  4008c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4008cf:	00 

00000000004008d0 <_Z11read_field0P9TestClass>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 10          	sub    $0x10,%rsp
  4008d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008dc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e0:	48 83 c7 04          	add    $0x4,%rdi
  4008e4:	e8 67 02 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4008e9:	8b 00                	mov    (%rax),%eax
  4008eb:	48 83 c4 10          	add    $0x10,%rsp
  4008ef:	5d                   	pop    %rbp
  4008f0:	c3                   	retq   
  4008f1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008f8:	0f 1f 84 00 00 00 00 
  4008ff:	00 

0000000000400900 <_Z12write_field1P9TestClass>:
  400900:	55                   	push   %rbp
  400901:	48 89 e5             	mov    %rsp,%rbp
  400904:	48 83 ec 10          	sub    $0x10,%rsp
  400908:	be 77 77 00 00       	mov    $0x7777,%esi
  40090d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400911:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400915:	48 83 c7 04          	add    $0x4,%rdi
  400919:	e8 72 01 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40091e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400922:	48 83 c4 10          	add    $0x10,%rsp
  400926:	5d                   	pop    %rbp
  400927:	c3                   	retq   
  400928:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40092f:	00 

0000000000400930 <_Z11read_field1P9TestClass>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400940:	48 83 c7 04          	add    $0x4,%rdi
  400944:	e8 27 02 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400949:	8b 00                	mov    (%rax),%eax
  40094b:	48 83 c4 10          	add    $0x10,%rsp
  40094f:	5d                   	pop    %rbp
  400950:	c3                   	retq   
  400951:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400958:	0f 1f 84 00 00 00 00 
  40095f:	00 

0000000000400960 <_Z21explicit_write_field0m>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400968:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40096c:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400973:	77 77 00 00 
  400977:	5d                   	pop    %rbp
  400978:	c3                   	retq   
  400979:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400980 <_Z20explicit_read_field0m>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40098c:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400993:	5d                   	pop    %rbp
  400994:	c3                   	retq   
  400995:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099c:	00 00 00 
  40099f:	90                   	nop

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	e8 47 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  4009a9:	48 89 c7             	mov    %rax,%rdi
  4009ac:	e8 cf 03 00 00       	callq  400d80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  4009b1:	5d                   	pop    %rbp
  4009b2:	c3                   	retq   
  4009b3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ba:	00 00 00 
  4009bd:	0f 1f 00             	nopl   (%rax)

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 83 ec 10          	sub    $0x10,%rsp
  4009c8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009cc:	e8 2f 04 00 00       	callq  400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009d1:	e8 1a fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  4009d6:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009db:	89 ce                	mov    %ecx,%esi
  4009dd:	48 89 c7             	mov    %rax,%rdi
  4009e0:	e8 4b 04 00 00       	callq  400e30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009e5:	48 89 c7             	mov    %rax,%rdi
  4009e8:	e8 23 04 00 00       	callq  400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009ed:	48 83 c4 10          	add    $0x10,%rsp
  4009f1:	5d                   	pop    %rbp
  4009f2:	c3                   	retq   
  4009f3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009fa:	00 00 00 
  4009fd:	0f 1f 00             	nopl   (%rax)

0000000000400a00 <_ZN9TestClassC2Ev>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 10          	sub    $0x10,%rsp
  400a08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a10:	48 89 f8             	mov    %rdi,%rax
  400a13:	48 83 c0 04          	add    $0x4,%rax
  400a17:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a1b:	48 89 c7             	mov    %rax,%rdi
  400a1e:	e8 6d 01 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a23:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a27:	48 83 c0 04          	add    $0x4,%rax
  400a2b:	48 89 c7             	mov    %rax,%rdi
  400a2e:	e8 6d 01 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a33:	48 83 c4 10          	add    $0x10,%rsp
  400a37:	5d                   	pop    %rbp
  400a38:	c3                   	retq   
  400a39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a40:	55                   	push   %rbp
  400a41:	48 89 e5             	mov    %rsp,%rbp
  400a44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a48:	5d                   	pop    %rbp
  400a49:	c3                   	retq   
  400a4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 83 ec 20          	sub    $0x20,%rsp
  400a58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a5c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a5f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a63:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400a66:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a6a:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400a6d:	e8 8e 01 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a72:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400a75:	89 30                	mov    %esi,(%rax)
  400a77:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a7b:	48 83 c4 20          	add    $0x20,%rsp
  400a7f:	5d                   	pop    %rbp
  400a80:	c3                   	retq   
  400a81:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a88:	00 00 00 
  400a8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a90:	55                   	push   %rbp
  400a91:	48 89 e5             	mov    %rsp,%rbp
  400a94:	48 83 ec 20          	sub    $0x20,%rsp
  400a98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a9c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a9f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400aa3:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400aa6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400aaa:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400aad:	e8 3e 02 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab2:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400ab5:	89 30                	mov    %esi,(%rax)
  400ab7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400abb:	48 83 c4 20          	add    $0x20,%rsp
  400abf:	5d                   	pop    %rbp
  400ac0:	c3                   	retq   
  400ac1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ac8:	00 00 00 
  400acb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ad0 <_ZN9TestClass15increase_field0Ev>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 83 ec 20          	sub    $0x20,%rsp
  400ad8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400adc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ae1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ae9:	48 83 c7 04          	add    $0x4,%rdi
  400aed:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400af1:	48 89 c7             	mov    %rax,%rdi
  400af4:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400af8:	e8 b3 00 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400afd:	48 83 c4 20          	add    $0x20,%rsp
  400b01:	5d                   	pop    %rbp
  400b02:	c3                   	retq   
  400b03:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0a:	00 00 00 
  400b0d:	0f 1f 00             	nopl   (%rax)

0000000000400b10 <_ZN9TestClass15increase_field1Ev>:
  400b10:	55                   	push   %rbp
  400b11:	48 89 e5             	mov    %rsp,%rbp
  400b14:	48 83 ec 20          	sub    $0x20,%rsp
  400b18:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b1c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b21:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b25:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b29:	48 83 c7 04          	add    $0x4,%rdi
  400b2d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b31:	48 89 c7             	mov    %rax,%rdi
  400b34:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b38:	e8 63 01 00 00       	callq  400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b3d:	48 83 c4 20          	add    $0x20,%rsp
  400b41:	5d                   	pop    %rbp
  400b42:	c3                   	retq   
  400b43:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4a:	00 00 00 
  400b4d:	0f 1f 00             	nopl   (%rax)

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 83 ec 10          	sub    $0x10,%rsp
  400b58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b60:	e8 9b 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b65:	48 83 c4 10          	add    $0x10,%rsp
  400b69:	5d                   	pop    %rbp
  400b6a:	c3                   	retq   
  400b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 83 ec 10          	sub    $0x10,%rsp
  400b78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b80:	e8 6b 01 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b85:	48 83 c4 10          	add    $0x10,%rsp
  400b89:	5d                   	pop    %rbp
  400b8a:	c3                   	retq   
  400b8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b98:	5d                   	pop    %rbp
  400b99:	c3                   	retq   
  400b9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ba8:	5d                   	pop    %rbp
  400ba9:	c3                   	retq   
  400baa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 83 ec 30          	sub    $0x30,%rsp
  400bb8:	48 89 f8             	mov    %rdi,%rax
  400bbb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bbf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400bc2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400bc6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400bc9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bcd:	48 89 f7             	mov    %rsi,%rdi
  400bd0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400bd4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400bd7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400bdb:	e8 20 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400be3:	0f af 10             	imul   (%rax),%edx
  400be6:	89 10                	mov    %edx,(%rax)
  400be8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400bec:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400bf0:	e8 2b 00 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400bf5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400bf9:	48 83 c4 30          	add    $0x30,%rsp
  400bfd:	5d                   	pop    %rbp
  400bfe:	c3                   	retq   
  400bff:	90                   	nop

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 83 ec 10          	sub    $0x10,%rsp
  400c08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c10:	e8 1b 00 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c15:	48 83 c4 10          	add    $0x10,%rsp
  400c19:	5d                   	pop    %rbp
  400c1a:	c3                   	retq   
  400c1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400c20:	55                   	push   %rbp
  400c21:	48 89 e5             	mov    %rsp,%rbp
  400c24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c2c:	5d                   	pop    %rbp
  400c2d:	c3                   	retq   
  400c2e:	66 90                	xchg   %ax,%ax

0000000000400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c30:	55                   	push   %rbp
  400c31:	48 89 e5             	mov    %rsp,%rbp
  400c34:	48 83 ec 20          	sub    $0x20,%rsp
  400c38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c40:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c44:	e8 a7 fa ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400c49:	48 89 c7             	mov    %rax,%rdi
  400c4c:	e8 2f 00 00 00       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400c51:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c55:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c59:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400c5d:	48 83 e8 04          	sub    $0x4,%rax
  400c61:	48 c1 e8 02          	shr    $0x2,%rax
  400c65:	48 c1 e0 02          	shl    $0x2,%rax
  400c69:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400c6d:	48 83 c0 00          	add    $0x0,%rax
  400c71:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c75:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c79:	48 83 c4 20          	add    $0x20,%rsp
  400c7d:	5d                   	pop    %rbp
  400c7e:	c3                   	retq   
  400c7f:	90                   	nop

0000000000400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c88:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c8c:	48 83 c7 10          	add    $0x10,%rdi
  400c90:	48 89 f8             	mov    %rdi,%rax
  400c93:	5d                   	pop    %rbp
  400c94:	c3                   	retq   
  400c95:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9c:	00 00 00 
  400c9f:	90                   	nop

0000000000400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 83 ec 30          	sub    $0x30,%rsp
  400ca8:	48 89 f8             	mov    %rdi,%rax
  400cab:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400caf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400cb2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400cb6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400cb9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400cbd:	48 89 f7             	mov    %rsi,%rdi
  400cc0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400cc4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400cc7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400ccb:	e8 20 00 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cd0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400cd3:	0f af 10             	imul   (%rax),%edx
  400cd6:	89 10                	mov    %edx,(%rax)
  400cd8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cdc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400ce0:	e8 2b 00 00 00       	callq  400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400ce5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ce9:	48 83 c4 30          	add    $0x30,%rsp
  400ced:	5d                   	pop    %rbp
  400cee:	c3                   	retq   
  400cef:	90                   	nop

0000000000400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cf0:	55                   	push   %rbp
  400cf1:	48 89 e5             	mov    %rsp,%rbp
  400cf4:	48 83 ec 10          	sub    $0x10,%rsp
  400cf8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cfc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d00:	e8 1b 00 00 00       	callq  400d20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d05:	48 83 c4 10          	add    $0x10,%rsp
  400d09:	5d                   	pop    %rbp
  400d0a:	c3                   	retq   
  400d0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d10:	55                   	push   %rbp
  400d11:	48 89 e5             	mov    %rsp,%rbp
  400d14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d18:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d1c:	5d                   	pop    %rbp
  400d1d:	c3                   	retq   
  400d1e:	66 90                	xchg   %ax,%ax

0000000000400d20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d20:	55                   	push   %rbp
  400d21:	48 89 e5             	mov    %rsp,%rbp
  400d24:	48 83 ec 20          	sub    $0x20,%rsp
  400d28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d30:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d34:	e8 b7 f9 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400d39:	48 89 c7             	mov    %rax,%rdi
  400d3c:	e8 3f ff ff ff       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d41:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d45:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d49:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400d4d:	48 83 e8 04          	sub    $0x4,%rax
  400d51:	48 c1 e8 02          	shr    $0x2,%rax
  400d55:	48 c1 e0 02          	shl    $0x2,%rax
  400d59:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400d5d:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d63:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d67:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d6b:	48 83 c4 20          	add    $0x20,%rsp
  400d6f:	5d                   	pop    %rbp
  400d70:	c3                   	retq   
  400d71:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d78:	00 00 00 
  400d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d80:	55                   	push   %rbp
  400d81:	48 89 e5             	mov    %rsp,%rbp
  400d84:	48 83 ec 10          	sub    $0x10,%rsp
  400d88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d90:	48 89 f8             	mov    %rdi,%rax
  400d93:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d97:	48 89 c7             	mov    %rax,%rdi
  400d9a:	e8 21 00 00 00       	callq  400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400d9f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400da3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400daa:	00 
  400dab:	48 83 c4 10          	add    $0x10,%rsp
  400daf:	5d                   	pop    %rbp
  400db0:	c3                   	retq   
  400db1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400db8:	00 00 00 
  400dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400dc0:	55                   	push   %rbp
  400dc1:	48 89 e5             	mov    %rsp,%rbp
  400dc4:	48 83 ec 10          	sub    $0x10,%rsp
  400dc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dcc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400dd0:	e8 0b 00 00 00       	callq  400de0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400dd5:	48 83 c4 10          	add    $0x10,%rsp
  400dd9:	5d                   	pop    %rbp
  400dda:	c3                   	retq   
  400ddb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400de0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400de0:	55                   	push   %rbp
  400de1:	48 89 e5             	mov    %rsp,%rbp
  400de4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400de8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400dec:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400df3:	5d                   	pop    %rbp
  400df4:	c3                   	retq   
  400df5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dfc:	00 00 00 
  400dff:	90                   	nop

0000000000400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e00:	55                   	push   %rbp
  400e01:	48 89 e5             	mov    %rsp,%rbp
  400e04:	5d                   	pop    %rbp
  400e05:	c3                   	retq   
  400e06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e0d:	00 00 00 

0000000000400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400e10:	55                   	push   %rbp
  400e11:	48 89 e5             	mov    %rsp,%rbp
  400e14:	48 83 ec 10          	sub    $0x10,%rsp
  400e18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e1c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e20:	48 83 c7 01          	add    $0x1,%rdi
  400e24:	e8 37 00 00 00       	callq  400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e29:	48 83 c4 10          	add    $0x10,%rsp
  400e2d:	5d                   	pop    %rbp
  400e2e:	c3                   	retq   
  400e2f:	90                   	nop

0000000000400e30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e30:	55                   	push   %rbp
  400e31:	48 89 e5             	mov    %rsp,%rbp
  400e34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e38:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400e3c:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400e40:	48 8b 3e             	mov    (%rsi),%rdi
  400e43:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400e47:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400e4b:	48 03 3e             	add    (%rsi),%rdi
  400e4e:	48 89 3e             	mov    %rdi,(%rsi)
  400e51:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400e55:	5d                   	pop    %rbp
  400e56:	c3                   	retq   
  400e57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e5e:	00 00 

0000000000400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e60:	55                   	push   %rbp
  400e61:	48 89 e5             	mov    %rsp,%rbp
  400e64:	48 83 ec 10          	sub    $0x10,%rsp
  400e68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e6c:	e8 7f f8 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400e71:	48 89 c7             	mov    %rax,%rdi
  400e74:	e8 07 fe ff ff       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e79:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e7d:	48 c1 e7 02          	shl    $0x2,%rdi
  400e81:	48 01 f8             	add    %rdi,%rax
  400e84:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400e88:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400e8c:	48 83 c4 10          	add    $0x10,%rsp
  400e90:	5d                   	pop    %rbp
  400e91:	c3                   	retq   
  400e92:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e99:	00 00 00 
  400e9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ea0 <__libc_csu_init>:
  400ea0:	41 57                	push   %r15
  400ea2:	41 56                	push   %r14
  400ea4:	41 89 ff             	mov    %edi,%r15d
  400ea7:	41 55                	push   %r13
  400ea9:	41 54                	push   %r12
  400eab:	4c 8d 25 2e 0f 20 00 	lea    0x200f2e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400eb2:	55                   	push   %rbp
  400eb3:	48 8d 2d 2e 0f 20 00 	lea    0x200f2e(%rip),%rbp        # 601de8 <__init_array_end>
  400eba:	53                   	push   %rbx
  400ebb:	49 89 f6             	mov    %rsi,%r14
  400ebe:	49 89 d5             	mov    %rdx,%r13
  400ec1:	4c 29 e5             	sub    %r12,%rbp
  400ec4:	48 83 ec 08          	sub    $0x8,%rsp
  400ec8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ecc:	e8 9f f6 ff ff       	callq  400570 <_init>
  400ed1:	48 85 ed             	test   %rbp,%rbp
  400ed4:	74 20                	je     400ef6 <__libc_csu_init+0x56>
  400ed6:	31 db                	xor    %ebx,%ebx
  400ed8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400edf:	00 
  400ee0:	4c 89 ea             	mov    %r13,%rdx
  400ee3:	4c 89 f6             	mov    %r14,%rsi
  400ee6:	44 89 ff             	mov    %r15d,%edi
  400ee9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400eed:	48 83 c3 01          	add    $0x1,%rbx
  400ef1:	48 39 eb             	cmp    %rbp,%rbx
  400ef4:	75 ea                	jne    400ee0 <__libc_csu_init+0x40>
  400ef6:	48 83 c4 08          	add    $0x8,%rsp
  400efa:	5b                   	pop    %rbx
  400efb:	5d                   	pop    %rbp
  400efc:	41 5c                	pop    %r12
  400efe:	41 5d                	pop    %r13
  400f00:	41 5e                	pop    %r14
  400f02:	41 5f                	pop    %r15
  400f04:	c3                   	retq   
  400f05:	90                   	nop
  400f06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f0d:	00 00 00 

0000000000400f10 <__libc_csu_fini>:
  400f10:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f14 <_fini>:
  400f14:	48 83 ec 08          	sub    $0x8,%rsp
  400f18:	48 83 c4 08          	add    $0x8,%rsp
  400f1c:	c3                   	retq   
