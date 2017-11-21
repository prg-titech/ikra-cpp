
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
  4005ff:	49 c7 c0 c0 0e 40 00 	mov    $0x400ec0,%r8
  400606:	48 c7 c1 50 0e 40 00 	mov    $0x400e50,%rcx
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
  4006f0:	48 83 ec 58          	sub    $0x58,%rsp
  4006f4:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  4006fb:	00 
  4006fc:	e8 7f 02 00 00       	callq  400980 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400701:	b8 04 00 00 00       	mov    $0x4,%eax
  400706:	89 c7                	mov    %eax,%edi
  400708:	e8 83 02 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  40070d:	48 89 c7             	mov    %rax,%rdi
  400710:	48 89 c1             	mov    %rax,%rcx
  400713:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400719:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40071e:	48 89 c7             	mov    %rax,%rdi
  400721:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400726:	e8 95 02 00 00       	callq  4009c0 <_ZN9TestClassC2Ev>
  40072b:	e9 00 00 00 00       	jmpq   400730 <main+0x40>
  400730:	be 77 77 00 00       	mov    $0x7777,%esi
  400735:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40073a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40073f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400744:	48 83 c1 04          	add    $0x4,%rcx
  400748:	48 89 cf             	mov    %rcx,%rdi
  40074b:	e8 c0 02 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  400750:	be 88 88 00 00       	mov    $0x8888,%esi
  400755:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40075a:	48 83 c1 04          	add    $0x4,%rcx
  40075e:	48 89 cf             	mov    %rcx,%rdi
  400761:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400766:	e8 e5 02 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  40076b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400770:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400775:	e8 16 03 00 00       	callq  400a90 <_ZN9TestClass15increase_field0Ev>
  40077a:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40077f:	e8 4c 03 00 00       	callq  400ad0 <_ZN9TestClass15increase_field1Ev>
  400784:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400789:	48 83 c0 04          	add    $0x4,%rax
  40078d:	48 89 c7             	mov    %rax,%rdi
  400790:	e8 7b 03 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  400795:	8b 30                	mov    (%rax),%esi
  400797:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40079c:	48 83 c0 04          	add    $0x4,%rax
  4007a0:	48 89 c7             	mov    %rax,%rdi
  4007a3:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007a7:	e8 74 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  4007ac:	48 bf d4 0e 40 00 00 	movabs $0x400ed4,%rdi
  4007b3:	00 00 00 
  4007b6:	8b 10                	mov    (%rax),%edx
  4007b8:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007bc:	b0 00                	mov    $0x0,%al
  4007be:	e8 dd fd ff ff       	callq  4005a0 <printf@plt>
  4007c3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007c8:	48 83 c1 04          	add    $0x4,%rcx
  4007cc:	48 89 cf             	mov    %rcx,%rdi
  4007cf:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007d3:	e8 38 03 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  4007d8:	31 d2                	xor    %edx,%edx
  4007da:	41 88 d0             	mov    %dl,%r8b
  4007dd:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007e3:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  4007e8:	0f 85 1e 00 00 00    	jne    40080c <main+0x11c>
  4007ee:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007f3:	48 83 c0 04          	add    $0x4,%rax
  4007f7:	48 89 c7             	mov    %rax,%rdi
  4007fa:	e8 21 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  4007ff:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400805:	0f 94 c1             	sete   %cl
  400808:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40080c:	8a 44 24 0f          	mov    0xf(%rsp),%al
  400810:	34 ff                	xor    $0xff,%al
  400812:	24 01                	and    $0x1,%al
  400814:	0f b6 c0             	movzbl %al,%eax
  400817:	48 83 c4 58          	add    $0x58,%rsp
  40081b:	c3                   	retq   
  40081c:	89 d1                	mov    %edx,%ecx
  40081e:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400823:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400827:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40082c:	e8 cf 01 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400831:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400836:	e8 95 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40083b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400840 <_Z12new_instancev>:
  400840:	48 83 ec 28          	sub    $0x28,%rsp
  400844:	b8 04 00 00 00       	mov    $0x4,%eax
  400849:	89 c7                	mov    %eax,%edi
  40084b:	e8 40 01 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400850:	48 89 c7             	mov    %rax,%rdi
  400853:	48 89 c1             	mov    %rax,%rcx
  400856:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40085c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400861:	48 89 c7             	mov    %rax,%rdi
  400864:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400869:	e8 52 01 00 00       	callq  4009c0 <_ZN9TestClassC2Ev>
  40086e:	e9 00 00 00 00       	jmpq   400873 <_Z12new_instancev+0x33>
  400873:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400878:	48 83 c4 28          	add    $0x28,%rsp
  40087c:	c3                   	retq   
  40087d:	89 d1                	mov    %edx,%ecx
  40087f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400884:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  400888:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40088d:	e8 6e 01 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400892:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400897:	e8 34 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40089c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008a0 <_Z12write_field0P9TestClass>:
  4008a0:	48 83 ec 18          	sub    $0x18,%rsp
  4008a4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008a9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ae:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008b3:	48 83 c7 04          	add    $0x4,%rdi
  4008b7:	e8 54 01 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  4008bc:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008c1:	48 83 c4 18          	add    $0x18,%rsp
  4008c5:	c3                   	retq   
  4008c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008cd:	00 00 00 

00000000004008d0 <_Z11read_field0P9TestClass>:
  4008d0:	50                   	push   %rax
  4008d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008d5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008d9:	48 83 c7 04          	add    $0x4,%rdi
  4008dd:	e8 2e 02 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  4008e2:	8b 00                	mov    (%rax),%eax
  4008e4:	59                   	pop    %rcx
  4008e5:	c3                   	retq   
  4008e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ed:	00 00 00 

00000000004008f0 <_Z12write_field1P9TestClass>:
  4008f0:	48 83 ec 18          	sub    $0x18,%rsp
  4008f4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008f9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008fe:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400903:	48 83 c7 04          	add    $0x4,%rdi
  400907:	e8 44 01 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  40090c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400911:	48 83 c4 18          	add    $0x18,%rsp
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z11read_field1P9TestClass>:
  400920:	50                   	push   %rax
  400921:	48 89 3c 24          	mov    %rdi,(%rsp)
  400925:	48 8b 3c 24          	mov    (%rsp),%rdi
  400929:	48 83 c7 04          	add    $0x4,%rdi
  40092d:	e8 ee 01 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  400932:	8b 00                	mov    (%rax),%eax
  400934:	59                   	pop    %rcx
  400935:	c3                   	retq   
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <_Z21explicit_write_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400951:	77 77 00 00 
  400955:	c3                   	retq   
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <_Z20explicit_read_field0m>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400971:	c3                   	retq   
  400972:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400979:	00 00 00 
  40097c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400980 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400980:	50                   	push   %rax
  400981:	e8 aa 02 00 00       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 c2 03 00 00       	callq  400d50 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40098e:	58                   	pop    %rax
  40098f:	c3                   	retq   

0000000000400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  400990:	50                   	push   %rax
  400991:	48 89 3c 24          	mov    %rdi,(%rsp)
  400995:	e8 16 04 00 00       	callq  400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40099a:	e8 91 02 00 00       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  40099f:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009a4:	89 ce                	mov    %ecx,%esi
  4009a6:	48 89 c7             	mov    %rax,%rdi
  4009a9:	e8 32 04 00 00       	callq  400de0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009ae:	48 89 c7             	mov    %rax,%rdi
  4009b1:	e8 0a 04 00 00       	callq  400dc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009b6:	59                   	pop    %rcx
  4009b7:	c3                   	retq   
  4009b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009bf:	00 

00000000004009c0 <_ZN9TestClassC2Ev>:
  4009c0:	48 83 ec 18          	sub    $0x18,%rsp
  4009c4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009c9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009ce:	48 89 f8             	mov    %rdi,%rax
  4009d1:	48 83 c0 04          	add    $0x4,%rax
  4009d5:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009da:	48 89 c7             	mov    %rax,%rdi
  4009dd:	e8 4e 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2Ev>
  4009e2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009e7:	48 83 c0 04          	add    $0x4,%rax
  4009eb:	48 89 c7             	mov    %rax,%rdi
  4009ee:	e8 4d 01 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2Ev>
  4009f3:	48 83 c4 18          	add    $0x18,%rsp
  4009f7:	c3                   	retq   
  4009f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009ff:	00 

0000000000400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a05:	c3                   	retq   
  400a06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a0d:	00 00 00 

0000000000400a10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>:
  400a10:	48 83 ec 28          	sub    $0x28,%rsp
  400a14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a19:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a1d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a22:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a26:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a2b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a2f:	e8 7c 01 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a34:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a38:	89 30                	mov    %esi,(%rax)
  400a3a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a3f:	48 83 c4 28          	add    $0x28,%rsp
  400a43:	c3                   	retq   
  400a44:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4b:	00 00 00 
  400a4e:	66 90                	xchg   %ax,%ax

0000000000400a50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>:
  400a50:	48 83 ec 28          	sub    $0x28,%rsp
  400a54:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a59:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a5d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a62:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a66:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a6b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a6f:	e8 5c 02 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a74:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a78:	89 30                	mov    %esi,(%rax)
  400a7a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a7f:	48 83 c4 28          	add    $0x28,%rsp
  400a83:	c3                   	retq   
  400a84:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8b:	00 00 00 
  400a8e:	66 90                	xchg   %ax,%ax

0000000000400a90 <_ZN9TestClass15increase_field0Ev>:
  400a90:	48 83 ec 18          	sub    $0x18,%rsp
  400a94:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a99:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a9e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400aa3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400aa8:	48 83 c7 04          	add    $0x4,%rdi
  400aac:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ab0:	48 89 c7             	mov    %rax,%rdi
  400ab3:	48 8b 34 24          	mov    (%rsp),%rsi
  400ab7:	e8 94 00 00 00       	callq  400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>
  400abc:	48 83 c4 18          	add    $0x18,%rsp
  400ac0:	c3                   	retq   
  400ac1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ac8:	00 00 00 
  400acb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ad0 <_ZN9TestClass15increase_field1Ev>:
  400ad0:	48 83 ec 18          	sub    $0x18,%rsp
  400ad4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ad9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ade:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ae3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ae8:	48 83 c7 04          	add    $0x4,%rdi
  400aec:	48 89 3c 24          	mov    %rdi,(%rsp)
  400af0:	48 89 c7             	mov    %rax,%rdi
  400af3:	48 8b 34 24          	mov    (%rsp),%rsi
  400af7:	e8 74 01 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>
  400afc:	48 83 c4 18          	add    $0x18,%rsp
  400b00:	c3                   	retq   
  400b01:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b08:	00 00 00 
  400b0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>:
  400b10:	50                   	push   %rax
  400b11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b19:	e8 92 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b1e:	59                   	pop    %rcx
  400b1f:	c3                   	retq   

0000000000400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>:
  400b20:	50                   	push   %rax
  400b21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b29:	e8 a2 01 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2e:	59                   	pop    %rcx
  400b2f:	c3                   	retq   

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2Ev>:
  400b30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b35:	c3                   	retq   
  400b36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b3d:	00 00 00 

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2Ev>:
  400b40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b45:	c3                   	retq   
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 

0000000000400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>:
  400b50:	48 83 ec 38          	sub    $0x38,%rsp
  400b54:	48 89 f8             	mov    %rdi,%rax
  400b57:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b5c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b60:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b65:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b69:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b6e:	48 89 f7             	mov    %rsi,%rdi
  400b71:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b76:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b7a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b7f:	e8 2c 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b84:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b88:	0f af 10             	imul   (%rax),%edx
  400b8b:	89 10                	mov    %edx,(%rax)
  400b8d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b92:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b97:	e8 24 00 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2ERKS3_>
  400b9c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ba1:	48 83 c4 38          	add    $0x38,%rsp
  400ba5:	c3                   	retq   
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb0:	50                   	push   %rax
  400bb1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bb5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bb9:	e8 12 00 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bbe:	59                   	pop    %rcx
  400bbf:	c3                   	retq   

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2ERKS3_>:
  400bc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bc5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bca:	c3                   	retq   
  400bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bd0:	48 83 ec 28          	sub    $0x28,%rsp
  400bd4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bd9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bde:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400be3:	e8 48 00 00 00       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400be8:	48 89 c7             	mov    %rax,%rdi
  400beb:	e8 50 00 00 00       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400bf0:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400bf5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bfa:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400bff:	48 83 e8 04          	sub    $0x4,%rax
  400c03:	48 c1 e8 02          	shr    $0x2,%rax
  400c07:	48 c1 e0 02          	shl    $0x2,%rax
  400c0b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c10:	48 83 c0 00          	add    $0x0,%rax
  400c14:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c19:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c1e:	48 83 c4 28          	add    $0x28,%rsp
  400c22:	c3                   	retq   
  400c23:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c2a:	00 00 00 
  400c2d:	0f 1f 00             	nopl   (%rax)

0000000000400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>:
  400c30:	50                   	push   %rax
  400c31:	e8 2a 00 00 00       	callq  400c60 <_ZN9TestClass14storage_bufferEv>
  400c36:	59                   	pop    %rcx
  400c37:	c3                   	retq   
  400c38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c3f:	00 

0000000000400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c45:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c4a:	48 83 c7 10          	add    $0x10,%rdi
  400c4e:	48 89 f8             	mov    %rdi,%rax
  400c51:	c3                   	retq   
  400c52:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c59:	00 00 00 
  400c5c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c60 <_ZN9TestClass14storage_bufferEv>:
  400c60:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c67:	00 00 00 
  400c6a:	c3                   	retq   
  400c6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>:
  400c70:	48 83 ec 38          	sub    $0x38,%rsp
  400c74:	48 89 f8             	mov    %rdi,%rax
  400c77:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c7c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c80:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c85:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c89:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c8e:	48 89 f7             	mov    %rsi,%rdi
  400c91:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c96:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c9a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c9f:	e8 2c 00 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400ca8:	0f af 10             	imul   (%rax),%edx
  400cab:	89 10                	mov    %edx,(%rax)
  400cad:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cb2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400cb7:	e8 24 00 00 00       	callq  400ce0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2ERKS3_>
  400cbc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cc1:	48 83 c4 38          	add    $0x38,%rsp
  400cc5:	c3                   	retq   
  400cc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ccd:	00 00 00 

0000000000400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cd0:	50                   	push   %rax
  400cd1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cd5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cd9:	e8 12 00 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cde:	59                   	pop    %rcx
  400cdf:	c3                   	retq   

0000000000400ce0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2ERKS3_>:
  400ce0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ce5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cea:	c3                   	retq   
  400ceb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cf0:	48 83 ec 28          	sub    $0x28,%rsp
  400cf4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400cf9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cfe:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d03:	e8 28 ff ff ff       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400d08:	48 89 c7             	mov    %rax,%rdi
  400d0b:	e8 30 ff ff ff       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d10:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d15:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d1a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d1f:	48 83 e8 04          	sub    $0x4,%rax
  400d23:	48 c1 e8 02          	shr    $0x2,%rax
  400d27:	48 c1 e0 02          	shl    $0x2,%rax
  400d2b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d30:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d36:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d3b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d40:	48 83 c4 28          	add    $0x28,%rsp
  400d44:	c3                   	retq   
  400d45:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d4c:	00 00 00 
  400d4f:	90                   	nop

0000000000400d50 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d50:	48 83 ec 18          	sub    $0x18,%rsp
  400d54:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d59:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d5e:	48 89 f8             	mov    %rdi,%rax
  400d61:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d66:	48 89 c7             	mov    %rax,%rdi
  400d69:	e8 12 00 00 00       	callq  400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400d6e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d73:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d7a:	00 
  400d7b:	48 83 c4 18          	add    $0x18,%rsp
  400d7f:	c3                   	retq   

0000000000400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400d80:	50                   	push   %rax
  400d81:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d85:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d89:	e8 02 00 00 00       	callq  400d90 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d8e:	58                   	pop    %rax
  400d8f:	c3                   	retq   

0000000000400d90 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d95:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d9a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400da1:	c3                   	retq   
  400da2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400da9:	00 00 00 
  400dac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400db0:	c3                   	retq   
  400db1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400db8:	00 00 00 
  400dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400dc0:	50                   	push   %rax
  400dc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400dc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400dc9:	48 83 c7 01          	add    $0x1,%rdi
  400dcd:	e8 3e 00 00 00       	callq  400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400dd2:	59                   	pop    %rcx
  400dd3:	c3                   	retq   
  400dd4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddb:	00 00 00 
  400dde:	66 90                	xchg   %ax,%ax

0000000000400de0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400de0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400de5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400dea:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400def:	48 8b 3e             	mov    (%rsi),%rdi
  400df2:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400df7:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400dfc:	48 03 3e             	add    (%rsi),%rdi
  400dff:	48 89 3e             	mov    %rdi,(%rsi)
  400e02:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e07:	c3                   	retq   
  400e08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e0f:	00 

0000000000400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e10:	48 83 ec 18          	sub    $0x18,%rsp
  400e14:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e19:	e8 12 fe ff ff       	callq  400c30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400e1e:	48 89 c7             	mov    %rax,%rdi
  400e21:	e8 1a fe ff ff       	callq  400c40 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e26:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e2b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e2f:	48 01 f8             	add    %rdi,%rax
  400e32:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e37:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e3c:	48 83 c4 18          	add    $0x18,%rsp
  400e40:	c3                   	retq   
  400e41:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e48:	00 00 00 
  400e4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e50 <__libc_csu_init>:
  400e50:	41 57                	push   %r15
  400e52:	41 56                	push   %r14
  400e54:	41 89 ff             	mov    %edi,%r15d
  400e57:	41 55                	push   %r13
  400e59:	41 54                	push   %r12
  400e5b:	4c 8d 25 7e 0f 20 00 	lea    0x200f7e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e62:	55                   	push   %rbp
  400e63:	48 8d 2d 7e 0f 20 00 	lea    0x200f7e(%rip),%rbp        # 601de8 <__init_array_end>
  400e6a:	53                   	push   %rbx
  400e6b:	49 89 f6             	mov    %rsi,%r14
  400e6e:	49 89 d5             	mov    %rdx,%r13
  400e71:	4c 29 e5             	sub    %r12,%rbp
  400e74:	48 83 ec 08          	sub    $0x8,%rsp
  400e78:	48 c1 fd 03          	sar    $0x3,%rbp
  400e7c:	e8 ef f6 ff ff       	callq  400570 <_init>
  400e81:	48 85 ed             	test   %rbp,%rbp
  400e84:	74 20                	je     400ea6 <__libc_csu_init+0x56>
  400e86:	31 db                	xor    %ebx,%ebx
  400e88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e8f:	00 
  400e90:	4c 89 ea             	mov    %r13,%rdx
  400e93:	4c 89 f6             	mov    %r14,%rsi
  400e96:	44 89 ff             	mov    %r15d,%edi
  400e99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e9d:	48 83 c3 01          	add    $0x1,%rbx
  400ea1:	48 39 eb             	cmp    %rbp,%rbx
  400ea4:	75 ea                	jne    400e90 <__libc_csu_init+0x40>
  400ea6:	48 83 c4 08          	add    $0x8,%rsp
  400eaa:	5b                   	pop    %rbx
  400eab:	5d                   	pop    %rbp
  400eac:	41 5c                	pop    %r12
  400eae:	41 5d                	pop    %r13
  400eb0:	41 5e                	pop    %r14
  400eb2:	41 5f                	pop    %r15
  400eb4:	c3                   	retq   
  400eb5:	90                   	nop
  400eb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ebd:	00 00 00 

0000000000400ec0 <__libc_csu_fini>:
  400ec0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ec4 <_fini>:
  400ec4:	48 83 ec 08          	sub    $0x8,%rsp
  400ec8:	48 83 c4 08          	add    $0x8,%rsp
  400ecc:	c3                   	retq   
