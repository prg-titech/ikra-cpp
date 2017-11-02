
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
  4005ff:	49 c7 c0 70 0e 40 00 	mov    $0x400e70,%r8
  400606:	48 c7 c1 00 0e 40 00 	mov    $0x400e00,%rcx
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
  40070f:	e8 ac 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageEv>
  400714:	b8 04 00 00 00       	mov    $0x4,%eax
  400719:	89 c7                	mov    %eax,%edi
  40071b:	e8 c0 02 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400720:	48 89 c7             	mov    %rax,%rdi
  400723:	48 89 c1             	mov    %rax,%rcx
  400726:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40072c:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400730:	48 89 c7             	mov    %rax,%rdi
  400733:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400737:	e8 d4 02 00 00       	callq  400a10 <_ZN9TestClassC2Ev>
  40073c:	e9 00 00 00 00       	jmpq   400741 <main+0x41>
  400741:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400745:	ba 77 77 00 00       	mov    $0x7777,%edx
  40074a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40074e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400752:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400756:	48 83 c1 04          	add    $0x4,%rcx
  40075a:	48 89 ce             	mov    %rcx,%rsi
  40075d:	e8 fe 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400762:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400766:	ba 88 88 00 00       	mov    $0x8888,%edx
  40076b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40076f:	48 83 c0 04          	add    $0x4,%rax
  400773:	48 89 c6             	mov    %rax,%rsi
  400776:	e8 35 03 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40077b:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40077f:	e8 7c 03 00 00       	callq  400b00 <_ZN9TestClass15increase_field0Ev>
  400784:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400788:	e8 b3 03 00 00       	callq  400b40 <_ZN9TestClass15increase_field1Ev>
  40078d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400791:	48 83 c0 04          	add    $0x4,%rax
  400795:	48 89 c7             	mov    %rax,%rdi
  400798:	e8 e3 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40079d:	8b 30                	mov    (%rax),%esi
  40079f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007a3:	48 83 c0 04          	add    $0x4,%rax
  4007a7:	48 89 c7             	mov    %rax,%rdi
  4007aa:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007ad:	e8 ee 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007b2:	48 bf 84 0e 40 00 00 	movabs $0x400e84,%rdi
  4007b9:	00 00 00 
  4007bc:	8b 10                	mov    (%rax),%edx
  4007be:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007c1:	b0 00                	mov    $0x0,%al
  4007c3:	e8 d8 fd ff ff       	callq  4005a0 <printf@plt>
  4007c8:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007cc:	48 83 c1 04          	add    $0x4,%rcx
  4007d0:	48 89 cf             	mov    %rcx,%rdi
  4007d3:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007d6:	e8 a5 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007db:	31 d2                	xor    %edx,%edx
  4007dd:	41 88 d0             	mov    %dl,%r8b
  4007e0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007e6:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  4007ea:	0f 85 1c 00 00 00    	jne    40080c <main+0x10c>
  4007f0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007f4:	48 83 c0 04          	add    $0x4,%rax
  4007f8:	48 89 c7             	mov    %rax,%rdi
  4007fb:	e8 a0 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400800:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400806:	0f 94 c1             	sete   %cl
  400809:	88 4d bf             	mov    %cl,-0x41(%rbp)
  40080c:	8a 45 bf             	mov    -0x41(%rbp),%al
  40080f:	34 ff                	xor    $0xff,%al
  400811:	24 01                	and    $0x1,%al
  400813:	0f b6 c0             	movzbl %al,%eax
  400816:	48 83 c4 50          	add    $0x50,%rsp
  40081a:	5d                   	pop    %rbp
  40081b:	c3                   	retq   
  40081c:	89 d1                	mov    %edx,%ecx
  40081e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400822:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400825:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400829:	e8 22 02 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
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
  40084f:	e8 8c 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	48 89 c1             	mov    %rax,%rcx
  40085a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400860:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400864:	48 89 c7             	mov    %rax,%rdi
  400867:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40086b:	e8 a0 01 00 00       	callq  400a10 <_ZN9TestClassC2Ev>
  400870:	e9 00 00 00 00       	jmpq   400875 <_Z12new_instancev+0x35>
  400875:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400879:	48 83 c4 20          	add    $0x20,%rsp
  40087d:	5d                   	pop    %rbp
  40087e:	c3                   	retq   
  40087f:	89 d1                	mov    %edx,%ecx
  400881:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400885:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  400888:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40088c:	e8 bf 01 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400891:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400895:	e8 36 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40089a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008a0 <_Z12write_field0P9TestClass>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 83 ec 20          	sub    $0x20,%rsp
  4008a8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008ac:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008b1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008b9:	48 83 c7 04          	add    $0x4,%rdi
  4008bd:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008c1:	48 89 c7             	mov    %rax,%rdi
  4008c4:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  4008c8:	e8 93 01 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4008cd:	48 83 c4 20          	add    $0x20,%rsp
  4008d1:	5d                   	pop    %rbp
  4008d2:	c3                   	retq   
  4008d3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008da:	84 00 00 00 00 00 

00000000004008e0 <_Z11read_field0P9TestClass>:
  4008e0:	55                   	push   %rbp
  4008e1:	48 89 e5             	mov    %rsp,%rbp
  4008e4:	48 83 ec 10          	sub    $0x10,%rsp
  4008e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008f0:	48 83 c7 04          	add    $0x4,%rdi
  4008f4:	e8 87 02 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4008f9:	8b 00                	mov    (%rax),%eax
  4008fb:	48 83 c4 10          	add    $0x10,%rsp
  4008ff:	5d                   	pop    %rbp
  400900:	c3                   	retq   
  400901:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400908:	0f 1f 84 00 00 00 00 
  40090f:	00 

0000000000400910 <_Z12write_field1P9TestClass>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 20          	sub    $0x20,%rsp
  400918:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40091c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400921:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400925:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400929:	48 83 c7 04          	add    $0x4,%rdi
  40092d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400931:	48 89 c7             	mov    %rax,%rdi
  400934:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400938:	e8 73 01 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40093d:	48 83 c4 20          	add    $0x20,%rsp
  400941:	5d                   	pop    %rbp
  400942:	c3                   	retq   
  400943:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40094a:	84 00 00 00 00 00 

0000000000400950 <_Z11read_field1P9TestClass>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 83 ec 10          	sub    $0x10,%rsp
  400958:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40095c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400960:	48 83 c7 04          	add    $0x4,%rdi
  400964:	e8 37 02 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400969:	8b 00                	mov    (%rax),%eax
  40096b:	48 83 c4 10          	add    $0x10,%rsp
  40096f:	5d                   	pop    %rbp
  400970:	c3                   	retq   
  400971:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400978:	0f 1f 84 00 00 00 00 
  40097f:	00 

0000000000400980 <_Z21explicit_write_field0m>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40098c:	c7 04 bd 10 b2 60 00 	movl   $0x7777,0x60b210(,%rdi,4)
  400993:	77 77 00 00 
  400997:	5d                   	pop    %rbp
  400998:	c3                   	retq   
  400999:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009a0 <_Z20explicit_read_field0m>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009ac:	8b 04 bd 10 b2 60 00 	mov    0x60b210(,%rdi,4),%eax
  4009b3:	5d                   	pop    %rbp
  4009b4:	c3                   	retq   
  4009b5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009bc:	00 00 00 
  4009bf:	90                   	nop

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageEv>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	e8 27 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  4009c9:	48 89 c7             	mov    %rax,%rdi
  4009cc:	e8 af 03 00 00       	callq  400d80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  4009d1:	5d                   	pop    %rbp
  4009d2:	c3                   	retq   
  4009d3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009da:	00 00 00 
  4009dd:	0f 1f 00             	nopl   (%rax)

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 83 ec 10          	sub    $0x10,%rsp
  4009e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ec:	e8 af 03 00 00       	callq  400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009f1:	e8 fa fc ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  4009f6:	48 8b 38             	mov    (%rax),%rdi
  4009f9:	48 89 f9             	mov    %rdi,%rcx
  4009fc:	48 83 c1 01          	add    $0x1,%rcx
  400a00:	48 89 08             	mov    %rcx,(%rax)
  400a03:	e8 a8 03 00 00       	callq  400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400a08:	48 83 c4 10          	add    $0x10,%rsp
  400a0c:	5d                   	pop    %rbp
  400a0d:	c3                   	retq   
  400a0e:	66 90                	xchg   %ax,%ax

0000000000400a10 <_ZN9TestClassC2Ev>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	48 83 ec 10          	sub    $0x10,%rsp
  400a18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a20:	48 89 f8             	mov    %rdi,%rax
  400a23:	48 83 c0 04          	add    $0x4,%rax
  400a27:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a2b:	48 89 c7             	mov    %rax,%rdi
  400a2e:	e8 8d 01 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a33:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a37:	48 83 c0 04          	add    $0x4,%rax
  400a3b:	48 89 c7             	mov    %rax,%rdi
  400a3e:	e8 8d 01 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a43:	48 83 c4 10          	add    $0x10,%rsp
  400a47:	5d                   	pop    %rbp
  400a48:	c3                   	retq   
  400a49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a58:	5d                   	pop    %rbp
  400a59:	c3                   	retq   
  400a5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a60:	55                   	push   %rbp
  400a61:	48 89 e5             	mov    %rsp,%rbp
  400a64:	48 83 ec 30          	sub    $0x30,%rsp
  400a68:	48 89 f8             	mov    %rdi,%rax
  400a6b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a6f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a72:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a76:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a79:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a7d:	48 89 f7             	mov    %rsi,%rdi
  400a80:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a84:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a87:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a8b:	e8 a0 01 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a90:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a93:	89 10                	mov    %edx,(%rax)
  400a95:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a99:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a9d:	e8 ae 01 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400aa2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400aa6:	48 83 c4 30          	add    $0x30,%rsp
  400aaa:	5d                   	pop    %rbp
  400aab:	c3                   	retq   
  400aac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400ab0:	55                   	push   %rbp
  400ab1:	48 89 e5             	mov    %rsp,%rbp
  400ab4:	48 83 ec 30          	sub    $0x30,%rsp
  400ab8:	48 89 f8             	mov    %rdi,%rax
  400abb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400abf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400ac2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ac6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400ac9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400acd:	48 89 f7             	mov    %rsi,%rdi
  400ad0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ad4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ad7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400adb:	e8 20 02 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ae3:	89 10                	mov    %edx,(%rax)
  400ae5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ae9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400aed:	e8 2e 02 00 00       	callq  400d20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400af2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400af6:	48 83 c4 30          	add    $0x30,%rsp
  400afa:	5d                   	pop    %rbp
  400afb:	c3                   	retq   
  400afc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b00 <_ZN9TestClass15increase_field0Ev>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 20          	sub    $0x20,%rsp
  400b08:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b0c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b11:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b15:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b19:	48 83 c7 04          	add    $0x4,%rdi
  400b1d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b21:	48 89 c7             	mov    %rax,%rdi
  400b24:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b28:	e8 b3 00 00 00       	callq  400be0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b2d:	48 83 c4 20          	add    $0x20,%rsp
  400b31:	5d                   	pop    %rbp
  400b32:	c3                   	retq   
  400b33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b3a:	00 00 00 
  400b3d:	0f 1f 00             	nopl   (%rax)

0000000000400b40 <_ZN9TestClass15increase_field1Ev>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 20          	sub    $0x20,%rsp
  400b48:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b4c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b51:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b55:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b59:	48 83 c7 04          	add    $0x4,%rdi
  400b5d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b61:	48 89 c7             	mov    %rax,%rdi
  400b64:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b68:	e8 43 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b6d:	48 83 c4 20          	add    $0x20,%rsp
  400b71:	5d                   	pop    %rbp
  400b72:	c3                   	retq   
  400b73:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7a:	00 00 00 
  400b7d:	0f 1f 00             	nopl   (%rax)

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 83 ec 10          	sub    $0x10,%rsp
  400b88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b90:	e8 9b 00 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b95:	48 83 c4 10          	add    $0x10,%rsp
  400b99:	5d                   	pop    %rbp
  400b9a:	c3                   	retq   
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 83 ec 10          	sub    $0x10,%rsp
  400ba8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bb0:	e8 4b 01 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bb5:	48 83 c4 10          	add    $0x10,%rsp
  400bb9:	5d                   	pop    %rbp
  400bba:	c3                   	retq   
  400bbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bc8:	5d                   	pop    %rbp
  400bc9:	c3                   	retq   
  400bca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bd8:	5d                   	pop    %rbp
  400bd9:	c3                   	retq   
  400bda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400be0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 83 ec 30          	sub    $0x30,%rsp
  400be8:	48 89 f8             	mov    %rdi,%rax
  400beb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bef:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400bf2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400bf6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400bf9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bfd:	48 89 f7             	mov    %rsi,%rdi
  400c00:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c04:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c07:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c0b:	e8 20 00 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c10:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c13:	0f af 10             	imul   (%rax),%edx
  400c16:	89 10                	mov    %edx,(%rax)
  400c18:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c1c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c20:	e8 2b 00 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c25:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c29:	48 83 c4 30          	add    $0x30,%rsp
  400c2d:	5d                   	pop    %rbp
  400c2e:	c3                   	retq   
  400c2f:	90                   	nop

0000000000400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c30:	55                   	push   %rbp
  400c31:	48 89 e5             	mov    %rsp,%rbp
  400c34:	48 83 ec 10          	sub    $0x10,%rsp
  400c38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c40:	e8 1b 00 00 00       	callq  400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c45:	48 83 c4 10          	add    $0x10,%rsp
  400c49:	5d                   	pop    %rbp
  400c4a:	c3                   	retq   
  400c4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400c50:	55                   	push   %rbp
  400c51:	48 89 e5             	mov    %rsp,%rbp
  400c54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c58:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c5c:	5d                   	pop    %rbp
  400c5d:	c3                   	retq   
  400c5e:	66 90                	xchg   %ax,%ax

0000000000400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 83 ec 20          	sub    $0x20,%rsp
  400c68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c6c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c70:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c74:	e8 77 fa ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400c79:	48 83 c0 08          	add    $0x8,%rax
  400c7d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c81:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c85:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400c89:	48 83 e8 04          	sub    $0x4,%rax
  400c8d:	48 c1 e8 02          	shr    $0x2,%rax
  400c91:	48 c1 e0 02          	shl    $0x2,%rax
  400c95:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400c99:	48 83 c0 00          	add    $0x0,%rax
  400c9d:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ca1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ca5:	48 83 c4 20          	add    $0x20,%rsp
  400ca9:	5d                   	pop    %rbp
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 83 ec 30          	sub    $0x30,%rsp
  400cb8:	48 89 f8             	mov    %rdi,%rax
  400cbb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400cbf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400cc2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400cc6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400cc9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400ccd:	48 89 f7             	mov    %rsi,%rdi
  400cd0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400cd4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400cd7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400cdb:	e8 20 00 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ce0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ce3:	0f af 10             	imul   (%rax),%edx
  400ce6:	89 10                	mov    %edx,(%rax)
  400ce8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cec:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cf0:	e8 2b 00 00 00       	callq  400d20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400cf5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cf9:	48 83 c4 30          	add    $0x30,%rsp
  400cfd:	5d                   	pop    %rbp
  400cfe:	c3                   	retq   
  400cff:	90                   	nop

0000000000400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 10          	sub    $0x10,%rsp
  400d08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d10:	e8 1b 00 00 00       	callq  400d30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d15:	48 83 c4 10          	add    $0x10,%rsp
  400d19:	5d                   	pop    %rbp
  400d1a:	c3                   	retq   
  400d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d20:	55                   	push   %rbp
  400d21:	48 89 e5             	mov    %rsp,%rbp
  400d24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d2c:	5d                   	pop    %rbp
  400d2d:	c3                   	retq   
  400d2e:	66 90                	xchg   %ax,%ax

0000000000400d30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d30:	55                   	push   %rbp
  400d31:	48 89 e5             	mov    %rsp,%rbp
  400d34:	48 83 ec 20          	sub    $0x20,%rsp
  400d38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d40:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d44:	e8 a7 f9 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400d49:	48 83 c0 08          	add    $0x8,%rax
  400d4d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d51:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d55:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400d59:	48 83 e8 04          	sub    $0x4,%rax
  400d5d:	48 c1 e8 02          	shr    $0x2,%rax
  400d61:	48 c1 e0 02          	shl    $0x2,%rax
  400d65:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400d69:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d6f:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d73:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d77:	48 83 c4 20          	add    $0x20,%rsp
  400d7b:	5d                   	pop    %rbp
  400d7c:	c3                   	retq   
  400d7d:	0f 1f 00             	nopl   (%rax)

0000000000400d80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d80:	55                   	push   %rbp
  400d81:	48 89 e5             	mov    %rsp,%rbp
  400d84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d88:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d8c:	48 c7 87 a8 91 00 00 	movq   $0x0,0x91a8(%rdi)
  400d93:	00 00 00 00 
  400d97:	5d                   	pop    %rbp
  400d98:	c3                   	retq   
  400d99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400da0:	55                   	push   %rbp
  400da1:	48 89 e5             	mov    %rsp,%rbp
  400da4:	5d                   	pop    %rbp
  400da5:	c3                   	retq   
  400da6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dad:	00 00 00 

0000000000400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400db0:	55                   	push   %rbp
  400db1:	48 89 e5             	mov    %rsp,%rbp
  400db4:	48 83 ec 10          	sub    $0x10,%rsp
  400db8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dbc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400dc0:	48 83 c7 01          	add    $0x1,%rdi
  400dc4:	e8 07 00 00 00       	callq  400dd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400dc9:	48 83 c4 10          	add    $0x10,%rsp
  400dcd:	5d                   	pop    %rbp
  400dce:	c3                   	retq   
  400dcf:	90                   	nop

0000000000400dd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400dd0:	55                   	push   %rbp
  400dd1:	48 89 e5             	mov    %rsp,%rbp
  400dd4:	48 83 ec 10          	sub    $0x10,%rsp
  400dd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ddc:	e8 0f f9 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400de1:	48 83 c0 08          	add    $0x8,%rax
  400de5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400de9:	48 c1 e7 02          	shl    $0x2,%rdi
  400ded:	48 01 f8             	add    %rdi,%rax
  400df0:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400df4:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400df8:	48 83 c4 10          	add    $0x10,%rsp
  400dfc:	5d                   	pop    %rbp
  400dfd:	c3                   	retq   
  400dfe:	66 90                	xchg   %ax,%ax

0000000000400e00 <__libc_csu_init>:
  400e00:	41 57                	push   %r15
  400e02:	41 56                	push   %r14
  400e04:	41 89 ff             	mov    %edi,%r15d
  400e07:	41 55                	push   %r13
  400e09:	41 54                	push   %r12
  400e0b:	4c 8d 25 ce 0f 20 00 	lea    0x200fce(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e12:	55                   	push   %rbp
  400e13:	48 8d 2d ce 0f 20 00 	lea    0x200fce(%rip),%rbp        # 601de8 <__init_array_end>
  400e1a:	53                   	push   %rbx
  400e1b:	49 89 f6             	mov    %rsi,%r14
  400e1e:	49 89 d5             	mov    %rdx,%r13
  400e21:	4c 29 e5             	sub    %r12,%rbp
  400e24:	48 83 ec 08          	sub    $0x8,%rsp
  400e28:	48 c1 fd 03          	sar    $0x3,%rbp
  400e2c:	e8 3f f7 ff ff       	callq  400570 <_init>
  400e31:	48 85 ed             	test   %rbp,%rbp
  400e34:	74 20                	je     400e56 <__libc_csu_init+0x56>
  400e36:	31 db                	xor    %ebx,%ebx
  400e38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e3f:	00 
  400e40:	4c 89 ea             	mov    %r13,%rdx
  400e43:	4c 89 f6             	mov    %r14,%rsi
  400e46:	44 89 ff             	mov    %r15d,%edi
  400e49:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e4d:	48 83 c3 01          	add    $0x1,%rbx
  400e51:	48 39 eb             	cmp    %rbp,%rbx
  400e54:	75 ea                	jne    400e40 <__libc_csu_init+0x40>
  400e56:	48 83 c4 08          	add    $0x8,%rsp
  400e5a:	5b                   	pop    %rbx
  400e5b:	5d                   	pop    %rbp
  400e5c:	41 5c                	pop    %r12
  400e5e:	41 5d                	pop    %r13
  400e60:	41 5e                	pop    %r14
  400e62:	41 5f                	pop    %r15
  400e64:	c3                   	retq   
  400e65:	90                   	nop
  400e66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e6d:	00 00 00 

0000000000400e70 <__libc_csu_fini>:
  400e70:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e74 <_fini>:
  400e74:	48 83 ec 08          	sub    $0x8,%rsp
  400e78:	48 83 c4 08          	add    $0x8,%rsp
  400e7c:	c3                   	retq   
