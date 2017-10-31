
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
  4005ff:	49 c7 c0 40 0d 40 00 	mov    $0x400d40,%r8
  400606:	48 c7 c1 d0 0c 40 00 	mov    $0x400cd0,%rcx
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

00000000004006f0 <main>:
  4006f0:	55                   	push   %rbp
  4006f1:	48 89 e5             	mov    %rsp,%rbp
  4006f4:	48 83 ec 50          	sub    $0x50,%rsp
  4006f8:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4006ff:	e8 3c 02 00 00       	callq  400940 <_ZN9TestClass18initialize_storageEv>
  400704:	31 c0                	xor    %eax,%eax
  400706:	89 c7                	mov    %eax,%edi
  400708:	e8 53 02 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40070d:	48 89 c7             	mov    %rax,%rdi
  400710:	48 89 c1             	mov    %rax,%rcx
  400713:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400717:	48 89 c7             	mov    %rax,%rdi
  40071a:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  40071e:	e8 6d 02 00 00       	callq  400990 <_ZN9TestClassC2Ev>
  400723:	e9 00 00 00 00       	jmpq   400728 <main+0x38>
  400728:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  40072c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400731:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400735:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400739:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40073d:	e8 8e 02 00 00       	callq  4009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400742:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400746:	ba 88 88 00 00       	mov    $0x8888,%edx
  40074b:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40074f:	e8 cc 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400754:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400758:	e8 13 03 00 00       	callq  400a70 <_ZN9TestClass15increase_field0Ev>
  40075d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400761:	e8 3a 03 00 00       	callq  400aa0 <_ZN9TestClass15increase_field1Ev>
  400766:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40076a:	e8 61 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40076f:	8b 30                	mov    (%rax),%esi
  400771:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400775:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400778:	e8 73 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40077d:	48 bf 54 0d 40 00 00 	movabs $0x400d54,%rdi
  400784:	00 00 00 
  400787:	8b 10                	mov    (%rax),%edx
  400789:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  40078c:	b0 00                	mov    $0x0,%al
  40078e:	e8 0d fe ff ff       	callq  4005a0 <printf@plt>
  400793:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400797:	89 45 c0             	mov    %eax,-0x40(%rbp)
  40079a:	e8 31 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40079f:	31 d2                	xor    %edx,%edx
  4007a1:	88 d1                	mov    %dl,%cl
  4007a3:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007a9:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007ac:	0f 85 15 00 00 00    	jne    4007c7 <main+0xd7>
  4007b2:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b6:	e8 35 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007bb:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007c1:	0f 94 c1             	sete   %cl
  4007c4:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007c7:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007ca:	34 ff                	xor    $0xff,%al
  4007cc:	24 01                	and    $0x1,%al
  4007ce:	0f b6 c0             	movzbl %al,%eax
  4007d1:	48 83 c4 50          	add    $0x50,%rsp
  4007d5:	5d                   	pop    %rbp
  4007d6:	c3                   	retq   
  4007d7:	89 d1                	mov    %edx,%ecx
  4007d9:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007dd:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4007e0:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  4007e4:	e8 d7 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  4007e9:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4007ed:	e8 de fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  4007f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4007f9:	1f 84 00 00 00 00 00 

0000000000400800 <_Z12new_instancev>:
  400800:	55                   	push   %rbp
  400801:	48 89 e5             	mov    %rsp,%rbp
  400804:	48 83 ec 20          	sub    $0x20,%rsp
  400808:	31 c0                	xor    %eax,%eax
  40080a:	89 c7                	mov    %eax,%edi
  40080c:	e8 4f 01 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400811:	48 89 c7             	mov    %rax,%rdi
  400814:	48 89 c1             	mov    %rax,%rcx
  400817:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40081b:	48 89 c7             	mov    %rax,%rdi
  40081e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400822:	e8 69 01 00 00       	callq  400990 <_ZN9TestClassC2Ev>
  400827:	e9 00 00 00 00       	jmpq   40082c <_Z12new_instancev+0x2c>
  40082c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400830:	48 83 c4 20          	add    $0x20,%rsp
  400834:	5d                   	pop    %rbp
  400835:	c3                   	retq   
  400836:	89 d1                	mov    %edx,%ecx
  400838:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40083c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40083f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400843:	e8 78 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400848:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40084c:	e8 7f fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400851:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400858:	0f 1f 84 00 00 00 00 
  40085f:	00 

0000000000400860 <_Z12write_field0P9TestClass>:
  400860:	55                   	push   %rbp
  400861:	48 89 e5             	mov    %rsp,%rbp
  400864:	48 83 ec 10          	sub    $0x10,%rsp
  400868:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40086c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400871:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400875:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400879:	48 89 c7             	mov    %rax,%rdi
  40087c:	e8 4f 01 00 00       	callq  4009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400881:	48 83 c4 10          	add    $0x10,%rsp
  400885:	5d                   	pop    %rbp
  400886:	c3                   	retq   
  400887:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40088e:	00 00 

0000000000400890 <_Z11read_field0P9TestClass>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 83 ec 10          	sub    $0x10,%rsp
  400898:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40089c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008a0:	e8 2b 02 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4008a5:	8b 00                	mov    (%rax),%eax
  4008a7:	48 83 c4 10          	add    $0x10,%rsp
  4008ab:	5d                   	pop    %rbp
  4008ac:	c3                   	retq   
  4008ad:	0f 1f 00             	nopl   (%rax)

00000000004008b0 <_Z12write_field1P9TestClass>:
  4008b0:	55                   	push   %rbp
  4008b1:	48 89 e5             	mov    %rsp,%rbp
  4008b4:	48 83 ec 10          	sub    $0x10,%rsp
  4008b8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008bc:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008c1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008c9:	48 89 c7             	mov    %rax,%rdi
  4008cc:	e8 4f 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4008d1:	48 83 c4 10          	add    $0x10,%rsp
  4008d5:	5d                   	pop    %rbp
  4008d6:	c3                   	retq   
  4008d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008de:	00 00 

00000000004008e0 <_Z11read_field1P9TestClass>:
  4008e0:	55                   	push   %rbp
  4008e1:	48 89 e5             	mov    %rsp,%rbp
  4008e4:	48 83 ec 10          	sub    $0x10,%rsp
  4008e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008f0:	e8 fb 01 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4008f5:	8b 00                	mov    (%rax),%eax
  4008f7:	48 83 c4 10          	add    $0x10,%rsp
  4008fb:	5d                   	pop    %rbp
  4008fc:	c3                   	retq   
  4008fd:	0f 1f 00             	nopl   (%rax)

0000000000400900 <_Z21explicit_write_field0m>:
  400900:	55                   	push   %rbp
  400901:	48 89 e5             	mov    %rsp,%rbp
  400904:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400908:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40090c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  400913:	00 
  400914:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40091b:	5d                   	pop    %rbp
  40091c:	c3                   	retq   
  40091d:	0f 1f 00             	nopl   (%rax)

0000000000400920 <_Z20explicit_read_field0m>:
  400920:	55                   	push   %rbp
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400928:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40092c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  400933:	00 
  400934:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  400937:	5d                   	pop    %rbp
  400938:	c3                   	retq   
  400939:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400940 <_ZN9TestClass18initialize_storageEv>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  40094b:	00 00 00 
  40094e:	48 89 c7             	mov    %rax,%rdi
  400951:	e8 ba 01 00 00       	callq  400b10 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  400956:	5d                   	pop    %rbp
  400957:	c3                   	retq   
  400958:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40095f:	00 

0000000000400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 83 ec 10          	sub    $0x10,%rsp
  400968:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40096c:	e8 1f 03 00 00       	callq  400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400971:	e8 6a 02 00 00       	callq  400be0 <_ZN9TestClass7storageEv>
  400976:	48 8b 38             	mov    (%rax),%rdi
  400979:	48 89 f9             	mov    %rdi,%rcx
  40097c:	48 83 c1 01          	add    $0x1,%rcx
  400980:	48 89 08             	mov    %rcx,(%rax)
  400983:	e8 18 03 00 00       	callq  400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400988:	48 83 c4 10          	add    $0x10,%rsp
  40098c:	5d                   	pop    %rbp
  40098d:	c3                   	retq   
  40098e:	66 90                	xchg   %ax,%ax

0000000000400990 <_ZN9TestClassC2Ev>:
  400990:	55                   	push   %rbp
  400991:	48 89 e5             	mov    %rsp,%rbp
  400994:	48 83 ec 10          	sub    $0x10,%rsp
  400998:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40099c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009a0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009a4:	e8 87 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009a9:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4009ad:	e8 8e 01 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  4009b2:	48 83 c4 10          	add    $0x10,%rsp
  4009b6:	5d                   	pop    %rbp
  4009b7:	c3                   	retq   
  4009b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009bf:	00 

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c8:	5d                   	pop    %rbp
  4009c9:	c3                   	retq   
  4009ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 83 ec 30          	sub    $0x30,%rsp
  4009d8:	48 89 f8             	mov    %rdi,%rax
  4009db:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  4009df:	89 55 f4             	mov    %edx,-0xc(%rbp)
  4009e2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4009e6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4009e9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4009ed:	48 89 f7             	mov    %rsi,%rdi
  4009f0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4009f4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  4009f7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4009fb:	e8 a0 01 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a00:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a03:	89 10                	mov    %edx,(%rax)
  400a05:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a09:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a0d:	e8 be 01 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400a12:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a16:	48 83 c4 30          	add    $0x30,%rsp
  400a1a:	5d                   	pop    %rbp
  400a1b:	c3                   	retq   
  400a1c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
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
  400a4b:	e8 f0 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a50:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a53:	89 10                	mov    %edx,(%rax)
  400a55:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a59:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a5d:	e8 1e 02 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400a62:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a66:	48 83 c4 30          	add    $0x30,%rsp
  400a6a:	5d                   	pop    %rbp
  400a6b:	c3                   	retq   
  400a6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a70 <_ZN9TestClass15increase_field0Ev>:
  400a70:	55                   	push   %rbp
  400a71:	48 89 e5             	mov    %rsp,%rbp
  400a74:	48 83 ec 10          	sub    $0x10,%rsp
  400a78:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400a7c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a81:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a85:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a89:	48 89 c7             	mov    %rax,%rdi
  400a8c:	e8 bf 00 00 00       	callq  400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
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
  400ae0:	e8 bb 00 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
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

0000000000400b10 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400b10:	55                   	push   %rbp
  400b11:	48 89 e5             	mov    %rsp,%rbp
  400b14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b18:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b1c:	48 c7 87 a8 91 00 00 	movq   $0x0,0x91a8(%rdi)
  400b23:	00 00 00 00 
  400b27:	5d                   	pop    %rbp
  400b28:	c3                   	retq   
  400b29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b38:	5d                   	pop    %rbp
  400b39:	c3                   	retq   
  400b3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b48:	5d                   	pop    %rbp
  400b49:	c3                   	retq   
  400b4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 83 ec 30          	sub    $0x30,%rsp
  400b58:	48 89 f8             	mov    %rdi,%rax
  400b5b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b5f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b62:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b66:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b69:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b6d:	48 89 f7             	mov    %rsi,%rdi
  400b70:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b74:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b77:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b7b:	e8 20 00 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b80:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b83:	0f af 10             	imul   (%rax),%edx
  400b86:	89 10                	mov    %edx,(%rax)
  400b88:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b8c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b90:	e8 3b 00 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b95:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b99:	48 83 c4 30          	add    $0x30,%rsp
  400b9d:	5d                   	pop    %rbp
  400b9e:	c3                   	retq   
  400b9f:	90                   	nop

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 83 ec 10          	sub    $0x10,%rsp
  400ba8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bb0:	48 c1 e7 02          	shl    $0x2,%rdi
  400bb4:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400bb8:	e8 23 00 00 00       	callq  400be0 <_ZN9TestClass7storageEv>
  400bbd:	48 83 c0 08          	add    $0x8,%rax
  400bc1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400bc5:	48 01 f8             	add    %rdi,%rax
  400bc8:	48 83 c4 10          	add    $0x10,%rsp
  400bcc:	5d                   	pop    %rbp
  400bcd:	c3                   	retq   
  400bce:	66 90                	xchg   %ax,%ax

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bd8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bdc:	5d                   	pop    %rbp
  400bdd:	c3                   	retq   
  400bde:	66 90                	xchg   %ax,%ax

0000000000400be0 <_ZN9TestClass7storageEv>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400beb:	00 00 00 
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
  400c30:	e8 4b 00 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
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
  400c50:	48 c1 e7 02          	shl    $0x2,%rdi
  400c54:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c58:	e8 83 ff ff ff       	callq  400be0 <_ZN9TestClass7storageEv>
  400c5d:	48 83 c0 08          	add    $0x8,%rax
  400c61:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400c65:	48 01 f8             	add    %rdi,%rax
  400c68:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c6e:	48 83 c4 10          	add    $0x10,%rsp
  400c72:	5d                   	pop    %rbp
  400c73:	c3                   	retq   
  400c74:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7b:	00 00 00 
  400c7e:	66 90                	xchg   %ax,%ax

0000000000400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c88:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c8c:	5d                   	pop    %rbp
  400c8d:	c3                   	retq   
  400c8e:	66 90                	xchg   %ax,%ax

0000000000400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c90:	55                   	push   %rbp
  400c91:	48 89 e5             	mov    %rsp,%rbp
  400c94:	5d                   	pop    %rbp
  400c95:	c3                   	retq   
  400c96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9d:	00 00 00 

0000000000400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 83 ec 10          	sub    $0x10,%rsp
  400ca8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cb0:	e8 0b 00 00 00       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400cb5:	48 83 c4 10          	add    $0x10,%rsp
  400cb9:	5d                   	pop    %rbp
  400cba:	c3                   	retq   
  400cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cc8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ccc:	5d                   	pop    %rbp
  400ccd:	c3                   	retq   
  400cce:	66 90                	xchg   %ax,%ax

0000000000400cd0 <__libc_csu_init>:
  400cd0:	41 57                	push   %r15
  400cd2:	41 56                	push   %r14
  400cd4:	41 89 ff             	mov    %edi,%r15d
  400cd7:	41 55                	push   %r13
  400cd9:	41 54                	push   %r12
  400cdb:	4c 8d 25 fe 10 20 00 	lea    0x2010fe(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400ce2:	55                   	push   %rbp
  400ce3:	48 8d 2d fe 10 20 00 	lea    0x2010fe(%rip),%rbp        # 601de8 <__init_array_end>
  400cea:	53                   	push   %rbx
  400ceb:	49 89 f6             	mov    %rsi,%r14
  400cee:	49 89 d5             	mov    %rdx,%r13
  400cf1:	4c 29 e5             	sub    %r12,%rbp
  400cf4:	48 83 ec 08          	sub    $0x8,%rsp
  400cf8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cfc:	e8 6f f8 ff ff       	callq  400570 <_init>
  400d01:	48 85 ed             	test   %rbp,%rbp
  400d04:	74 20                	je     400d26 <__libc_csu_init+0x56>
  400d06:	31 db                	xor    %ebx,%ebx
  400d08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d0f:	00 
  400d10:	4c 89 ea             	mov    %r13,%rdx
  400d13:	4c 89 f6             	mov    %r14,%rsi
  400d16:	44 89 ff             	mov    %r15d,%edi
  400d19:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d1d:	48 83 c3 01          	add    $0x1,%rbx
  400d21:	48 39 eb             	cmp    %rbp,%rbx
  400d24:	75 ea                	jne    400d10 <__libc_csu_init+0x40>
  400d26:	48 83 c4 08          	add    $0x8,%rsp
  400d2a:	5b                   	pop    %rbx
  400d2b:	5d                   	pop    %rbp
  400d2c:	41 5c                	pop    %r12
  400d2e:	41 5d                	pop    %r13
  400d30:	41 5e                	pop    %r14
  400d32:	41 5f                	pop    %r15
  400d34:	c3                   	retq   
  400d35:	90                   	nop
  400d36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3d:	00 00 00 

0000000000400d40 <__libc_csu_fini>:
  400d40:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d44 <_fini>:
  400d44:	48 83 ec 08          	sub    $0x8,%rsp
  400d48:	48 83 c4 08          	add    $0x8,%rsp
  400d4c:	c3                   	retq   
