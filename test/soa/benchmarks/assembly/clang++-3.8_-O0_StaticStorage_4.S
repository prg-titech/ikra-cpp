
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
  4005ff:	49 c7 c0 b0 0e 40 00 	mov    $0x400eb0,%r8
  400606:	48 c7 c1 40 0e 40 00 	mov    $0x400e40,%rcx
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
  4006f0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  4006f7:	00 00 00 
  4006fa:	c3                   	retq   
  4006fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400700 <main>:
  400700:	48 83 ec 58          	sub    $0x58,%rsp
  400704:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40070b:	00 
  40070c:	e8 7f 02 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400711:	b8 04 00 00 00       	mov    $0x4,%eax
  400716:	89 c7                	mov    %eax,%edi
  400718:	e8 83 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  40071d:	48 89 c7             	mov    %rax,%rdi
  400720:	48 89 c1             	mov    %rax,%rcx
  400723:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400729:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40072e:	48 89 c7             	mov    %rax,%rdi
  400731:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400736:	e8 95 02 00 00       	callq  4009d0 <_ZN9TestClassC2Ev>
  40073b:	e9 00 00 00 00       	jmpq   400740 <main+0x40>
  400740:	be 77 77 00 00       	mov    $0x7777,%esi
  400745:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40074a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40074f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400754:	48 83 c1 04          	add    $0x4,%rcx
  400758:	48 89 cf             	mov    %rcx,%rdi
  40075b:	e8 c0 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400760:	be 88 88 00 00       	mov    $0x8888,%esi
  400765:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40076a:	48 83 c1 04          	add    $0x4,%rcx
  40076e:	48 89 cf             	mov    %rcx,%rdi
  400771:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400776:	e8 e5 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40077b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400780:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400785:	e8 16 03 00 00       	callq  400aa0 <_ZN9TestClass15increase_field0Ev>
  40078a:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40078f:	e8 4c 03 00 00       	callq  400ae0 <_ZN9TestClass15increase_field1Ev>
  400794:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400799:	48 83 c0 04          	add    $0x4,%rax
  40079d:	48 89 c7             	mov    %rax,%rdi
  4007a0:	e8 7b 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007a5:	8b 30                	mov    (%rax),%esi
  4007a7:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007ac:	48 83 c0 04          	add    $0x4,%rax
  4007b0:	48 89 c7             	mov    %rax,%rdi
  4007b3:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007b7:	e8 74 03 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007bc:	48 bf c4 0e 40 00 00 	movabs $0x400ec4,%rdi
  4007c3:	00 00 00 
  4007c6:	8b 10                	mov    (%rax),%edx
  4007c8:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007cc:	b0 00                	mov    $0x0,%al
  4007ce:	e8 cd fd ff ff       	callq  4005a0 <printf@plt>
  4007d3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007d8:	48 83 c1 04          	add    $0x4,%rcx
  4007dc:	48 89 cf             	mov    %rcx,%rdi
  4007df:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007e3:	e8 38 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007e8:	31 d2                	xor    %edx,%edx
  4007ea:	41 88 d0             	mov    %dl,%r8b
  4007ed:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007f3:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  4007f8:	0f 85 1e 00 00 00    	jne    40081c <main+0x11c>
  4007fe:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400803:	48 83 c0 04          	add    $0x4,%rax
  400807:	48 89 c7             	mov    %rax,%rdi
  40080a:	e8 21 03 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40080f:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400815:	0f 94 c1             	sete   %cl
  400818:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40081c:	8a 44 24 0f          	mov    0xf(%rsp),%al
  400820:	34 ff                	xor    $0xff,%al
  400822:	24 01                	and    $0x1,%al
  400824:	0f b6 c0             	movzbl %al,%eax
  400827:	48 83 c4 58          	add    $0x58,%rsp
  40082b:	c3                   	retq   
  40082c:	89 d1                	mov    %edx,%ecx
  40082e:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400833:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400837:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40083c:	e8 cf 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400841:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400846:	e8 85 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40084b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400850 <_Z12new_instancev>:
  400850:	48 83 ec 28          	sub    $0x28,%rsp
  400854:	b8 04 00 00 00       	mov    $0x4,%eax
  400859:	89 c7                	mov    %eax,%edi
  40085b:	e8 40 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400860:	48 89 c7             	mov    %rax,%rdi
  400863:	48 89 c1             	mov    %rax,%rcx
  400866:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40086c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400871:	48 89 c7             	mov    %rax,%rdi
  400874:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400879:	e8 52 01 00 00       	callq  4009d0 <_ZN9TestClassC2Ev>
  40087e:	e9 00 00 00 00       	jmpq   400883 <_Z12new_instancev+0x33>
  400883:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400888:	48 83 c4 28          	add    $0x28,%rsp
  40088c:	c3                   	retq   
  40088d:	89 d1                	mov    %edx,%ecx
  40088f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400894:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  400898:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40089d:	e8 6e 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  4008a2:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008a7:	e8 24 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  4008ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008b0 <_Z12write_field0P9TestClass>:
  4008b0:	48 83 ec 18          	sub    $0x18,%rsp
  4008b4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008b9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008be:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008c3:	48 83 c7 04          	add    $0x4,%rdi
  4008c7:	e8 54 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4008cc:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008d1:	48 83 c4 18          	add    $0x18,%rsp
  4008d5:	c3                   	retq   
  4008d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008dd:	00 00 00 

00000000004008e0 <_Z11read_field0P9TestClass>:
  4008e0:	50                   	push   %rax
  4008e1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008e5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008e9:	48 83 c7 04          	add    $0x4,%rdi
  4008ed:	e8 2e 02 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4008f2:	8b 00                	mov    (%rax),%eax
  4008f4:	59                   	pop    %rcx
  4008f5:	c3                   	retq   
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_Z12write_field1P9TestClass>:
  400900:	48 83 ec 18          	sub    $0x18,%rsp
  400904:	be 77 77 00 00       	mov    $0x7777,%esi
  400909:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40090e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400913:	48 83 c7 04          	add    $0x4,%rdi
  400917:	e8 44 01 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40091c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400921:	48 83 c4 18          	add    $0x18,%rsp
  400925:	c3                   	retq   
  400926:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40092d:	00 00 00 

0000000000400930 <_Z11read_field1P9TestClass>:
  400930:	50                   	push   %rax
  400931:	48 89 3c 24          	mov    %rdi,(%rsp)
  400935:	48 8b 3c 24          	mov    (%rsp),%rdi
  400939:	48 83 c7 04          	add    $0x4,%rdi
  40093d:	e8 ee 01 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400942:	8b 00                	mov    (%rax),%eax
  400944:	59                   	pop    %rcx
  400945:	c3                   	retq   
  400946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094d:	00 00 00 

0000000000400950 <_Z21explicit_write_field0m>:
  400950:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400955:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40095a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400961:	77 77 00 00 
  400965:	c3                   	retq   
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 

0000000000400970 <_Z20explicit_read_field0m>:
  400970:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400975:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40097a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400981:	c3                   	retq   
  400982:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400989:	00 00 00 
  40098c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400990:	50                   	push   %rax
  400991:	e8 5a fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400996:	48 89 c7             	mov    %rax,%rdi
  400999:	e8 a2 03 00 00       	callq  400d40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40099e:	58                   	pop    %rax
  40099f:	c3                   	retq   

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009a0:	50                   	push   %rax
  4009a1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009a5:	e8 f6 03 00 00       	callq  400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009aa:	e8 41 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  4009af:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009b4:	89 ce                	mov    %ecx,%esi
  4009b6:	48 89 c7             	mov    %rax,%rdi
  4009b9:	e8 12 04 00 00       	callq  400dd0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009be:	48 89 c7             	mov    %rax,%rdi
  4009c1:	e8 ea 03 00 00       	callq  400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009c6:	59                   	pop    %rcx
  4009c7:	c3                   	retq   
  4009c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009cf:	00 

00000000004009d0 <_ZN9TestClassC2Ev>:
  4009d0:	48 83 ec 18          	sub    $0x18,%rsp
  4009d4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009d9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009de:	48 89 f8             	mov    %rdi,%rax
  4009e1:	48 83 c0 04          	add    $0x4,%rax
  4009e5:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009ea:	48 89 c7             	mov    %rax,%rdi
  4009ed:	e8 4e 01 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  4009f2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009f7:	48 83 c0 04          	add    $0x4,%rax
  4009fb:	48 89 c7             	mov    %rax,%rdi
  4009fe:	e8 4d 01 00 00       	callq  400b50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a03:	48 83 c4 18          	add    $0x18,%rsp
  400a07:	c3                   	retq   
  400a08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a0f:	00 

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a15:	c3                   	retq   
  400a16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a1d:	00 00 00 

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a20:	48 83 ec 28          	sub    $0x28,%rsp
  400a24:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a29:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a2d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a32:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a36:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a3b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a3f:	e8 7c 01 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a44:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a48:	89 30                	mov    %esi,(%rax)
  400a4a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a4f:	48 83 c4 28          	add    $0x28,%rsp
  400a53:	c3                   	retq   
  400a54:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5b:	00 00 00 
  400a5e:	66 90                	xchg   %ax,%ax

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a60:	48 83 ec 28          	sub    $0x28,%rsp
  400a64:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a69:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a72:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a76:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a7b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a7f:	e8 3c 02 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a84:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a88:	89 30                	mov    %esi,(%rax)
  400a8a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a8f:	48 83 c4 28          	add    $0x28,%rsp
  400a93:	c3                   	retq   
  400a94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a9b:	00 00 00 
  400a9e:	66 90                	xchg   %ax,%ax

0000000000400aa0 <_ZN9TestClass15increase_field0Ev>:
  400aa0:	48 83 ec 18          	sub    $0x18,%rsp
  400aa4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400aa9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400aae:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ab3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ab8:	48 83 c7 04          	add    $0x4,%rdi
  400abc:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ac0:	48 89 c7             	mov    %rax,%rdi
  400ac3:	48 8b 34 24          	mov    (%rsp),%rsi
  400ac7:	e8 94 00 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400acc:	48 83 c4 18          	add    $0x18,%rsp
  400ad0:	c3                   	retq   
  400ad1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ad8:	00 00 00 
  400adb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ae0 <_ZN9TestClass15increase_field1Ev>:
  400ae0:	48 83 ec 18          	sub    $0x18,%rsp
  400ae4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ae9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400aee:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400af3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400af8:	48 83 c7 04          	add    $0x4,%rdi
  400afc:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b00:	48 89 c7             	mov    %rax,%rdi
  400b03:	48 8b 34 24          	mov    (%rsp),%rsi
  400b07:	e8 54 01 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b0c:	48 83 c4 18          	add    $0x18,%rsp
  400b10:	c3                   	retq   
  400b11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b18:	00 00 00 
  400b1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b20:	50                   	push   %rax
  400b21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b29:	e8 92 00 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2e:	59                   	pop    %rcx
  400b2f:	c3                   	retq   

0000000000400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b30:	50                   	push   %rax
  400b31:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b35:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b39:	e8 82 01 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b3e:	59                   	pop    %rcx
  400b3f:	c3                   	retq   

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400b40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b45:	c3                   	retq   
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 

0000000000400b50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400b50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b55:	c3                   	retq   
  400b56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b5d:	00 00 00 

0000000000400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400b60:	48 83 ec 38          	sub    $0x38,%rsp
  400b64:	48 89 f8             	mov    %rdi,%rax
  400b67:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b6c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b70:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b75:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b79:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b7e:	48 89 f7             	mov    %rsi,%rdi
  400b81:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b86:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b8a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b8f:	e8 2c 00 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b94:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b98:	0f af 10             	imul   (%rax),%edx
  400b9b:	89 10                	mov    %edx,(%rax)
  400b9d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ba2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400ba7:	e8 24 00 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400bac:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bb1:	48 83 c4 38          	add    $0x38,%rsp
  400bb5:	c3                   	retq   
  400bb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bbd:	00 00 00 

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bc0:	50                   	push   %rax
  400bc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bc9:	e8 12 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bce:	59                   	pop    %rcx
  400bcf:	c3                   	retq   

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400bd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bd5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bda:	c3                   	retq   
  400bdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400be0:	48 83 ec 28          	sub    $0x28,%rsp
  400be4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400be9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bee:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400bf3:	e8 f8 fa ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400bf8:	48 89 c7             	mov    %rax,%rdi
  400bfb:	e8 40 00 00 00       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400c00:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c05:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c0a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c0f:	48 83 e8 04          	sub    $0x4,%rax
  400c13:	48 c1 e8 02          	shr    $0x2,%rax
  400c17:	48 c1 e0 02          	shl    $0x2,%rax
  400c1b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c20:	48 83 c0 00          	add    $0x0,%rax
  400c24:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c29:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c2e:	48 83 c4 28          	add    $0x28,%rsp
  400c32:	c3                   	retq   
  400c33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c3a:	00 00 00 
  400c3d:	0f 1f 00             	nopl   (%rax)

0000000000400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c45:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c4a:	48 83 c7 10          	add    $0x10,%rdi
  400c4e:	48 89 f8             	mov    %rdi,%rax
  400c51:	c3                   	retq   
  400c52:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c59:	00 00 00 
  400c5c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400c60:	48 83 ec 38          	sub    $0x38,%rsp
  400c64:	48 89 f8             	mov    %rdi,%rax
  400c67:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c6c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c70:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c75:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c79:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c7e:	48 89 f7             	mov    %rsi,%rdi
  400c81:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c86:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c8a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c8f:	e8 2c 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c94:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c98:	0f af 10             	imul   (%rax),%edx
  400c9b:	89 10                	mov    %edx,(%rax)
  400c9d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ca2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400ca7:	e8 24 00 00 00       	callq  400cd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400cac:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cb1:	48 83 c4 38          	add    $0x38,%rsp
  400cb5:	c3                   	retq   
  400cb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cbd:	00 00 00 

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cc0:	50                   	push   %rax
  400cc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cc9:	e8 12 00 00 00       	callq  400ce0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cce:	59                   	pop    %rcx
  400ccf:	c3                   	retq   

0000000000400cd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400cd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cd5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cda:	c3                   	retq   
  400cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ce0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ce0:	48 83 ec 28          	sub    $0x28,%rsp
  400ce4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400ce9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cee:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400cf3:	e8 f8 f9 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400cf8:	48 89 c7             	mov    %rax,%rdi
  400cfb:	e8 40 ff ff ff       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d00:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d05:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d0a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d0f:	48 83 e8 04          	sub    $0x4,%rax
  400d13:	48 c1 e8 02          	shr    $0x2,%rax
  400d17:	48 c1 e0 02          	shl    $0x2,%rax
  400d1b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d20:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d26:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d2b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d30:	48 83 c4 28          	add    $0x28,%rsp
  400d34:	c3                   	retq   
  400d35:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3c:	00 00 00 
  400d3f:	90                   	nop

0000000000400d40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d40:	48 83 ec 18          	sub    $0x18,%rsp
  400d44:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d49:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d4e:	48 89 f8             	mov    %rdi,%rax
  400d51:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d56:	48 89 c7             	mov    %rax,%rdi
  400d59:	e8 12 00 00 00       	callq  400d70 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400d5e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d63:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d6a:	00 
  400d6b:	48 83 c4 18          	add    $0x18,%rsp
  400d6f:	c3                   	retq   

0000000000400d70 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400d70:	50                   	push   %rax
  400d71:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d75:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d79:	e8 02 00 00 00       	callq  400d80 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d7e:	58                   	pop    %rax
  400d7f:	c3                   	retq   

0000000000400d80 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d85:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d8a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d91:	c3                   	retq   
  400d92:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d99:	00 00 00 
  400d9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400da0:	c3                   	retq   
  400da1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400da8:	00 00 00 
  400dab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400db0:	50                   	push   %rax
  400db1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400db5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400db9:	48 83 c7 01          	add    $0x1,%rdi
  400dbd:	e8 3e 00 00 00       	callq  400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400dc2:	59                   	pop    %rcx
  400dc3:	c3                   	retq   
  400dc4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcb:	00 00 00 
  400dce:	66 90                	xchg   %ax,%ax

0000000000400dd0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400dd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dd5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400dda:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400ddf:	48 8b 3e             	mov    (%rsi),%rdi
  400de2:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400de7:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400dec:	48 03 3e             	add    (%rsi),%rdi
  400def:	48 89 3e             	mov    %rdi,(%rsi)
  400df2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400df7:	c3                   	retq   
  400df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dff:	00 

0000000000400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e00:	48 83 ec 18          	sub    $0x18,%rsp
  400e04:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e09:	e8 e2 f8 ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400e0e:	48 89 c7             	mov    %rax,%rdi
  400e11:	e8 2a fe ff ff       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e16:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e1b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e1f:	48 01 f8             	add    %rdi,%rax
  400e22:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e27:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e2c:	48 83 c4 18          	add    $0x18,%rsp
  400e30:	c3                   	retq   
  400e31:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e38:	00 00 00 
  400e3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e40 <__libc_csu_init>:
  400e40:	41 57                	push   %r15
  400e42:	41 56                	push   %r14
  400e44:	41 89 ff             	mov    %edi,%r15d
  400e47:	41 55                	push   %r13
  400e49:	41 54                	push   %r12
  400e4b:	4c 8d 25 8e 0f 20 00 	lea    0x200f8e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e52:	55                   	push   %rbp
  400e53:	48 8d 2d 8e 0f 20 00 	lea    0x200f8e(%rip),%rbp        # 601de8 <__init_array_end>
  400e5a:	53                   	push   %rbx
  400e5b:	49 89 f6             	mov    %rsi,%r14
  400e5e:	49 89 d5             	mov    %rdx,%r13
  400e61:	4c 29 e5             	sub    %r12,%rbp
  400e64:	48 83 ec 08          	sub    $0x8,%rsp
  400e68:	48 c1 fd 03          	sar    $0x3,%rbp
  400e6c:	e8 ff f6 ff ff       	callq  400570 <_init>
  400e71:	48 85 ed             	test   %rbp,%rbp
  400e74:	74 20                	je     400e96 <__libc_csu_init+0x56>
  400e76:	31 db                	xor    %ebx,%ebx
  400e78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e7f:	00 
  400e80:	4c 89 ea             	mov    %r13,%rdx
  400e83:	4c 89 f6             	mov    %r14,%rsi
  400e86:	44 89 ff             	mov    %r15d,%edi
  400e89:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e8d:	48 83 c3 01          	add    $0x1,%rbx
  400e91:	48 39 eb             	cmp    %rbp,%rbx
  400e94:	75 ea                	jne    400e80 <__libc_csu_init+0x40>
  400e96:	48 83 c4 08          	add    $0x8,%rsp
  400e9a:	5b                   	pop    %rbx
  400e9b:	5d                   	pop    %rbp
  400e9c:	41 5c                	pop    %r12
  400e9e:	41 5d                	pop    %r13
  400ea0:	41 5e                	pop    %r14
  400ea2:	41 5f                	pop    %r15
  400ea4:	c3                   	retq   
  400ea5:	90                   	nop
  400ea6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ead:	00 00 00 

0000000000400eb0 <__libc_csu_fini>:
  400eb0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400eb4 <_fini>:
  400eb4:	48 83 ec 08          	sub    $0x8,%rsp
  400eb8:	48 83 c4 08          	add    $0x8,%rsp
  400ebc:	c3                   	retq   
