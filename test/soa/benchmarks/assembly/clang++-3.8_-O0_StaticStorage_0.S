
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
  4005ff:	49 c7 c0 f0 0d 40 00 	mov    $0x400df0,%r8
  400606:	48 c7 c1 80 0d 40 00 	mov    $0x400d80,%rcx
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
  4006fc:	e8 4f 02 00 00       	callq  400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400701:	31 c0                	xor    %eax,%eax
  400703:	89 c7                	mov    %eax,%edi
  400705:	e8 56 02 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40070a:	48 89 c7             	mov    %rax,%rdi
  40070d:	48 89 c1             	mov    %rax,%rcx
  400710:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400715:	48 89 c7             	mov    %rax,%rdi
  400718:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40071d:	e8 6e 02 00 00       	callq  400990 <_ZN9TestClassC2Ev>
  400722:	e9 00 00 00 00       	jmpq   400727 <main+0x37>
  400727:	be 77 77 00 00       	mov    $0x7777,%esi
  40072c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400731:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400736:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40073b:	e8 90 02 00 00       	callq  4009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  400740:	be 88 88 00 00       	mov    $0x8888,%esi
  400745:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40074a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40074f:	e8 bc 02 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  400754:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400759:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40075e:	e8 ed 02 00 00       	callq  400a50 <_ZN9TestClass15increase_field0Ev>
  400763:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400768:	e8 13 03 00 00       	callq  400a80 <_ZN9TestClass15increase_field1Ev>
  40076d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400772:	e8 39 03 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  400777:	8b 30                	mov    (%rax),%esi
  400779:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40077e:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400782:	e8 39 03 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  400787:	48 bf 04 0e 40 00 00 	movabs $0x400e04,%rdi
  40078e:	00 00 00 
  400791:	8b 10                	mov    (%rax),%edx
  400793:	8b 74 24 14          	mov    0x14(%rsp),%esi
  400797:	b0 00                	mov    $0x0,%al
  400799:	e8 02 fe ff ff       	callq  4005a0 <printf@plt>
  40079e:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a3:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007a7:	e8 04 03 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  4007ac:	31 d2                	xor    %edx,%edx
  4007ae:	88 d1                	mov    %dl,%cl
  4007b0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007b6:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007ba:	0f 85 17 00 00 00    	jne    4007d7 <main+0xe7>
  4007c0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007c5:	e8 f6 02 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  4007ca:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007d0:	0f 94 c1             	sete   %cl
  4007d3:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007d7:	8a 44 24 0f          	mov    0xf(%rsp),%al
  4007db:	34 ff                	xor    $0xff,%al
  4007dd:	24 01                	and    $0x1,%al
  4007df:	0f b6 c0             	movzbl %al,%eax
  4007e2:	48 83 c4 58          	add    $0x58,%rsp
  4007e6:	c3                   	retq   
  4007e7:	89 d1                	mov    %edx,%ecx
  4007e9:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  4007ee:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  4007f2:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  4007f7:	e8 c4 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  4007fc:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400801:	e8 ca fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400806:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40080d:	00 00 00 

0000000000400810 <_Z12new_instancev>:
  400810:	48 83 ec 28          	sub    $0x28,%rsp
  400814:	31 c0                	xor    %eax,%eax
  400816:	89 c7                	mov    %eax,%edi
  400818:	e8 43 01 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40081d:	48 89 c7             	mov    %rax,%rdi
  400820:	48 89 c1             	mov    %rax,%rcx
  400823:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400828:	48 89 c7             	mov    %rax,%rdi
  40082b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400830:	e8 5b 01 00 00       	callq  400990 <_ZN9TestClassC2Ev>
  400835:	e9 00 00 00 00       	jmpq   40083a <_Z12new_instancev+0x2a>
  40083a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40083f:	48 83 c4 28          	add    $0x28,%rsp
  400843:	c3                   	retq   
  400844:	89 d1                	mov    %edx,%ecx
  400846:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40084b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40084f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400854:	e8 67 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400859:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40085e:	e8 6d fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400863:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40086a:	84 00 00 00 00 00 

0000000000400870 <_Z12write_field0P9TestClass>:
  400870:	48 83 ec 18          	sub    $0x18,%rsp
  400874:	be 77 77 00 00       	mov    $0x7777,%esi
  400879:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40087e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400883:	e8 48 01 00 00       	callq  4009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  400888:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40088d:	48 83 c4 18          	add    $0x18,%rsp
  400891:	c3                   	retq   
  400892:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400899:	1f 84 00 00 00 00 00 

00000000004008a0 <_Z11read_field0P9TestClass>:
  4008a0:	50                   	push   %rax
  4008a1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008a5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008a9:	e8 02 02 00 00       	callq  400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  4008ae:	8b 00                	mov    (%rax),%eax
  4008b0:	59                   	pop    %rcx
  4008b1:	c3                   	retq   
  4008b2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008b9:	1f 84 00 00 00 00 00 

00000000004008c0 <_Z12write_field1P9TestClass>:
  4008c0:	48 83 ec 18          	sub    $0x18,%rsp
  4008c4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008c9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ce:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008d3:	e8 38 01 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  4008d8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008dd:	48 83 c4 18          	add    $0x18,%rsp
  4008e1:	c3                   	retq   
  4008e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008e9:	1f 84 00 00 00 00 00 

00000000004008f0 <_Z11read_field1P9TestClass>:
  4008f0:	50                   	push   %rax
  4008f1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008f5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008f9:	e8 c2 01 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  4008fe:	8b 00                	mov    (%rax),%eax
  400900:	59                   	pop    %rcx
  400901:	c3                   	retq   
  400902:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400909:	1f 84 00 00 00 00 00 

0000000000400910 <_Z21explicit_write_field0m>:
  400910:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400915:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40091a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400921:	77 77 00 00 
  400925:	c3                   	retq   
  400926:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40092d:	00 00 00 

0000000000400930 <_Z20explicit_read_field0m>:
  400930:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400935:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40093a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400941:	c3                   	retq   
  400942:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400949:	00 00 00 
  40094c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400950:	50                   	push   %rax
  400951:	e8 3a 03 00 00       	callq  400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400956:	48 89 c7             	mov    %rax,%rdi
  400959:	e8 42 03 00 00       	callq  400ca0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40095e:	58                   	pop    %rax
  40095f:	c3                   	retq   

0000000000400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400960:	50                   	push   %rax
  400961:	48 89 3c 24          	mov    %rdi,(%rsp)
  400965:	e8 a6 03 00 00       	callq  400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40096a:	e8 21 03 00 00       	callq  400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  40096f:	b9 01 00 00 00       	mov    $0x1,%ecx
  400974:	89 ce                	mov    %ecx,%esi
  400976:	48 89 c7             	mov    %rax,%rdi
  400979:	e8 c2 03 00 00       	callq  400d40 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  40097e:	48 89 c7             	mov    %rax,%rdi
  400981:	e8 9a 03 00 00       	callq  400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400986:	59                   	pop    %rcx
  400987:	c3                   	retq   
  400988:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40098f:	00 

0000000000400990 <_ZN9TestClassC2Ev>:
  400990:	48 83 ec 18          	sub    $0x18,%rsp
  400994:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400999:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40099e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009a3:	e8 28 01 00 00       	callq  400ad0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2Ev>
  4009a8:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4009ad:	e8 2e 01 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2Ev>
  4009b2:	48 83 c4 18          	add    $0x18,%rsp
  4009b6:	c3                   	retq   
  4009b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009be:	00 00 

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009c0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009c5:	c3                   	retq   
  4009c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009cd:	00 00 00 

00000000004009d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>:
  4009d0:	48 83 ec 28          	sub    $0x28,%rsp
  4009d4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  4009d9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  4009dd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4009e2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  4009e6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009eb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  4009ef:	e8 5c 01 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  4009f4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  4009f8:	89 30                	mov    %esi,(%rax)
  4009fa:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4009ff:	48 83 c4 28          	add    $0x28,%rsp
  400a03:	c3                   	retq   
  400a04:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a0b:	00 00 00 
  400a0e:	66 90                	xchg   %ax,%ax

0000000000400a10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>:
  400a10:	48 83 ec 28          	sub    $0x28,%rsp
  400a14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a19:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a1d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a22:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a26:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a2b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a2f:	e8 ec 01 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  400a34:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a38:	89 30                	mov    %esi,(%rax)
  400a3a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a3f:	48 83 c4 28          	add    $0x28,%rsp
  400a43:	c3                   	retq   
  400a44:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4b:	00 00 00 
  400a4e:	66 90                	xchg   %ax,%ax

0000000000400a50 <_ZN9TestClass15increase_field0Ev>:
  400a50:	48 83 ec 18          	sub    $0x18,%rsp
  400a54:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a59:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a5e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a63:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a68:	48 89 c7             	mov    %rax,%rdi
  400a6b:	e8 80 00 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>
  400a70:	48 83 c4 18          	add    $0x18,%rsp
  400a74:	c3                   	retq   
  400a75:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a7c:	00 00 00 
  400a7f:	90                   	nop

0000000000400a80 <_ZN9TestClass15increase_field1Ev>:
  400a80:	48 83 ec 18          	sub    $0x18,%rsp
  400a84:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a89:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a8e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a93:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a98:	48 89 c7             	mov    %rax,%rdi
  400a9b:	e8 20 01 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>
  400aa0:	48 83 c4 18          	add    $0x18,%rsp
  400aa4:	c3                   	retq   
  400aa5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aac:	00 00 00 
  400aaf:	90                   	nop

0000000000400ab0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>:
  400ab0:	50                   	push   %rax
  400ab1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ab5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ab9:	e8 92 00 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  400abe:	59                   	pop    %rcx
  400abf:	c3                   	retq   

0000000000400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>:
  400ac0:	50                   	push   %rax
  400ac1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ac5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ac9:	e8 52 01 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  400ace:	59                   	pop    %rcx
  400acf:	c3                   	retq   

0000000000400ad0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2Ev>:
  400ad0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ad5:	c3                   	retq   
  400ad6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400add:	00 00 00 

0000000000400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2Ev>:
  400ae0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ae5:	c3                   	retq   
  400ae6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aed:	00 00 00 

0000000000400af0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>:
  400af0:	48 83 ec 38          	sub    $0x38,%rsp
  400af4:	48 89 f8             	mov    %rdi,%rax
  400af7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400afc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b00:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b05:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b09:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b0e:	48 89 f7             	mov    %rsi,%rdi
  400b11:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b16:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b1a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b1f:	e8 2c 00 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  400b24:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b28:	0f af 10             	imul   (%rax),%edx
  400b2b:	89 10                	mov    %edx,(%rax)
  400b2d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b32:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b37:	e8 24 00 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2ERKS3_>
  400b3c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b41:	48 83 c4 38          	add    $0x38,%rsp
  400b45:	c3                   	retq   
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>:
  400b50:	50                   	push   %rax
  400b51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b59:	e8 12 00 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400b5e:	59                   	pop    %rcx
  400b5f:	c3                   	retq   

0000000000400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2ERKS3_>:
  400b60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b65:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b6a:	c3                   	retq   
  400b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400b70:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400b77:	00 00 00 
  400b7a:	48 89 c1             	mov    %rax,%rcx
  400b7d:	48 83 c1 10          	add    $0x10,%rcx
  400b81:	48 89 ca             	mov    %rcx,%rdx
  400b84:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b89:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400b8e:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400b95:	00 00 
  400b97:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400b9c:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400ba1:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400ba6:	48 c1 e7 02          	shl    $0x2,%rdi
  400baa:	48 01 fa             	add    %rdi,%rdx
  400bad:	48 89 d0             	mov    %rdx,%rax
  400bb0:	c3                   	retq   
  400bb1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bb8:	00 00 00 
  400bbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>:
  400bc0:	48 83 ec 38          	sub    $0x38,%rsp
  400bc4:	48 89 f8             	mov    %rdi,%rax
  400bc7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bcc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400bd0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400bd5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400bd9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bde:	48 89 f7             	mov    %rsi,%rdi
  400be1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400be6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bea:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bef:	e8 2c 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>
  400bf4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bf8:	0f af 10             	imul   (%rax),%edx
  400bfb:	89 10                	mov    %edx,(%rax)
  400bfd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c02:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c07:	e8 24 00 00 00       	callq  400c30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2ERKS3_>
  400c0c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c11:	48 83 c4 38          	add    $0x38,%rsp
  400c15:	c3                   	retq   
  400c16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c1d:	00 00 00 

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrIJEEEPiDpT_>:
  400c20:	50                   	push   %rax
  400c21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c29:	e8 12 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400c2e:	59                   	pop    %rcx
  400c2f:	c3                   	retq   

0000000000400c30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2ERKS3_>:
  400c30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c3a:	c3                   	retq   
  400c3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400c40:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c47:	00 00 00 
  400c4a:	48 89 c1             	mov    %rax,%rcx
  400c4d:	48 81 c1 e0 48 00 00 	add    $0x48e0,%rcx
  400c54:	48 89 ca             	mov    %rcx,%rdx
  400c57:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c5c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c61:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400c68:	00 00 
  400c6a:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400c6f:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400c74:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400c79:	48 c1 e7 02          	shl    $0x2,%rdi
  400c7d:	48 01 fa             	add    %rdi,%rdx
  400c80:	48 89 d0             	mov    %rdx,%rax
  400c83:	c3                   	retq   
  400c84:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c8b:	00 00 00 
  400c8e:	66 90                	xchg   %ax,%ax

0000000000400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>:
  400c90:	50                   	push   %rax
  400c91:	e8 3a 00 00 00       	callq  400cd0 <_ZN9TestClass14storage_bufferEv>
  400c96:	59                   	pop    %rcx
  400c97:	c3                   	retq   
  400c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c9f:	00 

0000000000400ca0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400ca0:	48 83 ec 18          	sub    $0x18,%rsp
  400ca4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ca9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cae:	48 89 f8             	mov    %rdi,%rax
  400cb1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cb6:	48 89 c7             	mov    %rax,%rdi
  400cb9:	e8 22 00 00 00       	callq  400ce0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400cbe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400cca:	00 
  400ccb:	48 83 c4 18          	add    $0x18,%rsp
  400ccf:	c3                   	retq   

0000000000400cd0 <_ZN9TestClass14storage_bufferEv>:
  400cd0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400cd7:	00 00 00 
  400cda:	c3                   	retq   
  400cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ce0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400ce0:	50                   	push   %rax
  400ce1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ce5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ce9:	e8 02 00 00 00       	callq  400cf0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400cee:	58                   	pop    %rax
  400cef:	c3                   	retq   

0000000000400cf0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400cf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cf5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400cfa:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d01:	c3                   	retq   
  400d02:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d09:	00 00 00 
  400d0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d10:	c3                   	retq   
  400d11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d18:	00 00 00 
  400d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400d20:	50                   	push   %rax
  400d21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d29:	48 83 c7 01          	add    $0x1,%rdi
  400d2d:	e8 3e 00 00 00       	callq  400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d32:	59                   	pop    %rcx
  400d33:	c3                   	retq   
  400d34:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3b:	00 00 00 
  400d3e:	66 90                	xchg   %ax,%ax

0000000000400d40 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400d40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d45:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d4a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400d4f:	48 8b 3e             	mov    (%rsi),%rdi
  400d52:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d57:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400d5c:	48 03 3e             	add    (%rsi),%rdi
  400d5f:	48 89 3e             	mov    %rdi,(%rsi)
  400d62:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d67:	c3                   	retq   
  400d68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d6f:	00 

0000000000400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d75:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d7a:	c3                   	retq   
  400d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d80 <__libc_csu_init>:
  400d80:	41 57                	push   %r15
  400d82:	41 56                	push   %r14
  400d84:	41 89 ff             	mov    %edi,%r15d
  400d87:	41 55                	push   %r13
  400d89:	41 54                	push   %r12
  400d8b:	4c 8d 25 4e 10 20 00 	lea    0x20104e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d92:	55                   	push   %rbp
  400d93:	48 8d 2d 4e 10 20 00 	lea    0x20104e(%rip),%rbp        # 601de8 <__init_array_end>
  400d9a:	53                   	push   %rbx
  400d9b:	49 89 f6             	mov    %rsi,%r14
  400d9e:	49 89 d5             	mov    %rdx,%r13
  400da1:	4c 29 e5             	sub    %r12,%rbp
  400da4:	48 83 ec 08          	sub    $0x8,%rsp
  400da8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dac:	e8 bf f7 ff ff       	callq  400570 <_init>
  400db1:	48 85 ed             	test   %rbp,%rbp
  400db4:	74 20                	je     400dd6 <__libc_csu_init+0x56>
  400db6:	31 db                	xor    %ebx,%ebx
  400db8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dbf:	00 
  400dc0:	4c 89 ea             	mov    %r13,%rdx
  400dc3:	4c 89 f6             	mov    %r14,%rsi
  400dc6:	44 89 ff             	mov    %r15d,%edi
  400dc9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dcd:	48 83 c3 01          	add    $0x1,%rbx
  400dd1:	48 39 eb             	cmp    %rbp,%rbx
  400dd4:	75 ea                	jne    400dc0 <__libc_csu_init+0x40>
  400dd6:	48 83 c4 08          	add    $0x8,%rsp
  400dda:	5b                   	pop    %rbx
  400ddb:	5d                   	pop    %rbp
  400ddc:	41 5c                	pop    %r12
  400dde:	41 5d                	pop    %r13
  400de0:	41 5e                	pop    %r14
  400de2:	41 5f                	pop    %r15
  400de4:	c3                   	retq   
  400de5:	90                   	nop
  400de6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ded:	00 00 00 

0000000000400df0 <__libc_csu_fini>:
  400df0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400df4 <_fini>:
  400df4:	48 83 ec 08          	sub    $0x8,%rsp
  400df8:	48 83 c4 08          	add    $0x8,%rsp
  400dfc:	c3                   	retq   
