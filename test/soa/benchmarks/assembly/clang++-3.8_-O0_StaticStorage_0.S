
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
  4005ff:	49 c7 c0 e0 0d 40 00 	mov    $0x400de0,%r8
  400606:	48 c7 c1 70 0d 40 00 	mov    $0x400d70,%rcx
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
  40070c:	e8 4f 02 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  400711:	31 c0                	xor    %eax,%eax
  400713:	89 c7                	mov    %eax,%edi
  400715:	e8 56 02 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40071a:	48 89 c7             	mov    %rax,%rdi
  40071d:	48 89 c1             	mov    %rax,%rcx
  400720:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400725:	48 89 c7             	mov    %rax,%rdi
  400728:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40072d:	e8 6e 02 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  400732:	e9 00 00 00 00       	jmpq   400737 <main+0x37>
  400737:	be 77 77 00 00       	mov    $0x7777,%esi
  40073c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400741:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400746:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40074b:	e8 90 02 00 00       	callq  4009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400750:	be 88 88 00 00       	mov    $0x8888,%esi
  400755:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40075a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40075f:	e8 bc 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400764:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400769:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40076e:	e8 ed 02 00 00       	callq  400a60 <_ZN9TestClass15increase_field0Ev>
  400773:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400778:	e8 13 03 00 00       	callq  400a90 <_ZN9TestClass15increase_field1Ev>
  40077d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400782:	e8 39 03 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400787:	8b 30                	mov    (%rax),%esi
  400789:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40078e:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400792:	e8 39 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400797:	48 bf f4 0d 40 00 00 	movabs $0x400df4,%rdi
  40079e:	00 00 00 
  4007a1:	8b 10                	mov    (%rax),%edx
  4007a3:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007a7:	b0 00                	mov    $0x0,%al
  4007a9:	e8 f2 fd ff ff       	callq  4005a0 <printf@plt>
  4007ae:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007b3:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007b7:	e8 04 03 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007bc:	31 d2                	xor    %edx,%edx
  4007be:	88 d1                	mov    %dl,%cl
  4007c0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007c6:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007ca:	0f 85 17 00 00 00    	jne    4007e7 <main+0xe7>
  4007d0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007d5:	e8 f6 02 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007da:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007e0:	0f 94 c1             	sete   %cl
  4007e3:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007e7:	8a 44 24 0f          	mov    0xf(%rsp),%al
  4007eb:	34 ff                	xor    $0xff,%al
  4007ed:	24 01                	and    $0x1,%al
  4007ef:	0f b6 c0             	movzbl %al,%eax
  4007f2:	48 83 c4 58          	add    $0x58,%rsp
  4007f6:	c3                   	retq   
  4007f7:	89 d1                	mov    %edx,%ecx
  4007f9:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  4007fe:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400802:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  400807:	e8 c4 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  40080c:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400811:	e8 ba fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400816:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40081d:	00 00 00 

0000000000400820 <_Z12new_instancev>:
  400820:	48 83 ec 28          	sub    $0x28,%rsp
  400824:	31 c0                	xor    %eax,%eax
  400826:	89 c7                	mov    %eax,%edi
  400828:	e8 43 01 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40082d:	48 89 c7             	mov    %rax,%rdi
  400830:	48 89 c1             	mov    %rax,%rcx
  400833:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400838:	48 89 c7             	mov    %rax,%rdi
  40083b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400840:	e8 5b 01 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  400845:	e9 00 00 00 00       	jmpq   40084a <_Z12new_instancev+0x2a>
  40084a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40084f:	48 83 c4 28          	add    $0x28,%rsp
  400853:	c3                   	retq   
  400854:	89 d1                	mov    %edx,%ecx
  400856:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40085b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40085f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400864:	e8 67 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400869:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40086e:	e8 5d fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400873:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40087a:	84 00 00 00 00 00 

0000000000400880 <_Z12write_field0P9TestClass>:
  400880:	48 83 ec 18          	sub    $0x18,%rsp
  400884:	be 77 77 00 00       	mov    $0x7777,%esi
  400889:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40088e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400893:	e8 48 01 00 00       	callq  4009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400898:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40089d:	48 83 c4 18          	add    $0x18,%rsp
  4008a1:	c3                   	retq   
  4008a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008a9:	1f 84 00 00 00 00 00 

00000000004008b0 <_Z11read_field0P9TestClass>:
  4008b0:	50                   	push   %rax
  4008b1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008b5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008b9:	e8 02 02 00 00       	callq  400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4008be:	8b 00                	mov    (%rax),%eax
  4008c0:	59                   	pop    %rcx
  4008c1:	c3                   	retq   
  4008c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c9:	1f 84 00 00 00 00 00 

00000000004008d0 <_Z12write_field1P9TestClass>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008de:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008e3:	e8 38 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4008e8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008ed:	48 83 c4 18          	add    $0x18,%rsp
  4008f1:	c3                   	retq   
  4008f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008f9:	1f 84 00 00 00 00 00 

0000000000400900 <_Z11read_field1P9TestClass>:
  400900:	50                   	push   %rax
  400901:	48 89 3c 24          	mov    %rdi,(%rsp)
  400905:	48 8b 3c 24          	mov    (%rsp),%rdi
  400909:	e8 c2 01 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40090e:	8b 00                	mov    (%rax),%eax
  400910:	59                   	pop    %rcx
  400911:	c3                   	retq   
  400912:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400919:	1f 84 00 00 00 00 00 

0000000000400920 <_Z21explicit_write_field0m>:
  400920:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400925:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40092a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400931:	77 77 00 00 
  400935:	c3                   	retq   
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <_Z20explicit_read_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400951:	c3                   	retq   
  400952:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400959:	00 00 00 
  40095c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400960:	50                   	push   %rax
  400961:	e8 8a fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  400966:	48 89 c7             	mov    %rax,%rdi
  400969:	e8 32 03 00 00       	callq  400ca0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40096e:	58                   	pop    %rax
  40096f:	c3                   	retq   

0000000000400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400970:	50                   	push   %rax
  400971:	48 89 3c 24          	mov    %rdi,(%rsp)
  400975:	e8 86 03 00 00       	callq  400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40097a:	e8 71 fd ff ff       	callq  4006f0 <_ZN9TestClass7storageEv>
  40097f:	b9 01 00 00 00       	mov    $0x1,%ecx
  400984:	89 ce                	mov    %ecx,%esi
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 a2 03 00 00       	callq  400d30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  40098e:	48 89 c7             	mov    %rax,%rdi
  400991:	e8 7a 03 00 00       	callq  400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400996:	59                   	pop    %rcx
  400997:	c3                   	retq   
  400998:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40099f:	00 

00000000004009a0 <_ZN9TestClassC2Ev>:
  4009a0:	48 83 ec 18          	sub    $0x18,%rsp
  4009a4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009a9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009ae:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009b3:	e8 28 01 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009b8:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4009bd:	e8 2e 01 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  4009c2:	48 83 c4 18          	add    $0x18,%rsp
  4009c6:	c3                   	retq   
  4009c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009ce:	00 00 

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009d0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009d5:	c3                   	retq   
  4009d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009dd:	00 00 00 

00000000004009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009e0:	48 83 ec 28          	sub    $0x28,%rsp
  4009e4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  4009e9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  4009ed:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4009f2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  4009f6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009fb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  4009ff:	e8 5c 01 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a04:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a08:	89 30                	mov    %esi,(%rax)
  400a0a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a0f:	48 83 c4 28          	add    $0x28,%rsp
  400a13:	c3                   	retq   
  400a14:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a1b:	00 00 00 
  400a1e:	66 90                	xchg   %ax,%ax

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a20:	48 83 ec 28          	sub    $0x28,%rsp
  400a24:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a29:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a2d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a32:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a36:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a3b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a3f:	e8 ec 01 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a44:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a48:	89 30                	mov    %esi,(%rax)
  400a4a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a4f:	48 83 c4 28          	add    $0x28,%rsp
  400a53:	c3                   	retq   
  400a54:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5b:	00 00 00 
  400a5e:	66 90                	xchg   %ax,%ax

0000000000400a60 <_ZN9TestClass15increase_field0Ev>:
  400a60:	48 83 ec 18          	sub    $0x18,%rsp
  400a64:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a69:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a6e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a73:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a78:	48 89 c7             	mov    %rax,%rdi
  400a7b:	e8 80 00 00 00       	callq  400b00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400a80:	48 83 c4 18          	add    $0x18,%rsp
  400a84:	c3                   	retq   
  400a85:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8c:	00 00 00 
  400a8f:	90                   	nop

0000000000400a90 <_ZN9TestClass15increase_field1Ev>:
  400a90:	48 83 ec 18          	sub    $0x18,%rsp
  400a94:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a99:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a9e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400aa3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 20 01 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400ab0:	48 83 c4 18          	add    $0x18,%rsp
  400ab4:	c3                   	retq   
  400ab5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400abc:	00 00 00 
  400abf:	90                   	nop

0000000000400ac0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400ac0:	50                   	push   %rax
  400ac1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ac5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ac9:	e8 92 00 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ace:	59                   	pop    %rcx
  400acf:	c3                   	retq   

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400ad0:	50                   	push   %rax
  400ad1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ad5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ad9:	e8 52 01 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ade:	59                   	pop    %rcx
  400adf:	c3                   	retq   

0000000000400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400ae0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ae5:	c3                   	retq   
  400ae6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aed:	00 00 00 

0000000000400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400af0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400af5:	c3                   	retq   
  400af6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400afd:	00 00 00 

0000000000400b00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b00:	48 83 ec 38          	sub    $0x38,%rsp
  400b04:	48 89 f8             	mov    %rdi,%rax
  400b07:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b0c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b10:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b15:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b19:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b1e:	48 89 f7             	mov    %rsi,%rdi
  400b21:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b26:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b2a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b2f:	e8 2c 00 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b34:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b38:	0f af 10             	imul   (%rax),%edx
  400b3b:	89 10                	mov    %edx,(%rax)
  400b3d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b42:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b47:	e8 24 00 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b4c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b51:	48 83 c4 38          	add    $0x38,%rsp
  400b55:	c3                   	retq   
  400b56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b5d:	00 00 00 

0000000000400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b60:	50                   	push   %rax
  400b61:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b65:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b69:	e8 12 00 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b6e:	59                   	pop    %rcx
  400b6f:	c3                   	retq   

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400b70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b75:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b7a:	c3                   	retq   
  400b7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b80:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400b87:	00 00 00 
  400b8a:	48 89 c1             	mov    %rax,%rcx
  400b8d:	48 83 c1 10          	add    $0x10,%rcx
  400b91:	48 89 ca             	mov    %rcx,%rdx
  400b94:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b99:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400b9e:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400ba5:	00 00 
  400ba7:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400bac:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400bb1:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400bb6:	48 c1 e7 02          	shl    $0x2,%rdi
  400bba:	48 01 fa             	add    %rdi,%rdx
  400bbd:	48 89 d0             	mov    %rdx,%rax
  400bc0:	c3                   	retq   
  400bc1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bc8:	00 00 00 
  400bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400bd0:	48 83 ec 38          	sub    $0x38,%rsp
  400bd4:	48 89 f8             	mov    %rdi,%rax
  400bd7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bdc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400be0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400be5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400be9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bee:	48 89 f7             	mov    %rsi,%rdi
  400bf1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bf6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bfa:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bff:	e8 2c 00 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c04:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c08:	0f af 10             	imul   (%rax),%edx
  400c0b:	89 10                	mov    %edx,(%rax)
  400c0d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c12:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c17:	e8 24 00 00 00       	callq  400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400c1c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c21:	48 83 c4 38          	add    $0x38,%rsp
  400c25:	c3                   	retq   
  400c26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c2d:	00 00 00 

0000000000400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c30:	50                   	push   %rax
  400c31:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c35:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c39:	e8 12 00 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c3e:	59                   	pop    %rcx
  400c3f:	c3                   	retq   

0000000000400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400c40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c45:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c4a:	c3                   	retq   
  400c4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c50:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c57:	00 00 00 
  400c5a:	48 89 c1             	mov    %rax,%rcx
  400c5d:	48 81 c1 e0 48 00 00 	add    $0x48e0,%rcx
  400c64:	48 89 ca             	mov    %rcx,%rdx
  400c67:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c6c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c71:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400c78:	00 00 
  400c7a:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400c7f:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400c84:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400c89:	48 c1 e7 02          	shl    $0x2,%rdi
  400c8d:	48 01 fa             	add    %rdi,%rdx
  400c90:	48 89 d0             	mov    %rdx,%rax
  400c93:	c3                   	retq   
  400c94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9b:	00 00 00 
  400c9e:	66 90                	xchg   %ax,%ax

0000000000400ca0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400ca0:	48 83 ec 18          	sub    $0x18,%rsp
  400ca4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ca9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cae:	48 89 f8             	mov    %rdi,%rax
  400cb1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cb6:	48 89 c7             	mov    %rax,%rdi
  400cb9:	e8 12 00 00 00       	callq  400cd0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400cbe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400cca:	00 
  400ccb:	48 83 c4 18          	add    $0x18,%rsp
  400ccf:	c3                   	retq   

0000000000400cd0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400cd0:	50                   	push   %rax
  400cd1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cd5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cd9:	e8 02 00 00 00       	callq  400ce0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400cde:	58                   	pop    %rax
  400cdf:	c3                   	retq   

0000000000400ce0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400ce0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ce5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400cea:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400cf1:	c3                   	retq   
  400cf2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cf9:	00 00 00 
  400cfc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d00:	c3                   	retq   
  400d01:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d08:	00 00 00 
  400d0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400d10:	50                   	push   %rax
  400d11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d19:	48 83 c7 01          	add    $0x1,%rdi
  400d1d:	e8 3e 00 00 00       	callq  400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d22:	59                   	pop    %rcx
  400d23:	c3                   	retq   
  400d24:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d2b:	00 00 00 
  400d2e:	66 90                	xchg   %ax,%ax

0000000000400d30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400d30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d3a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400d3f:	48 8b 3e             	mov    (%rsi),%rdi
  400d42:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d47:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400d4c:	48 03 3e             	add    (%rsi),%rdi
  400d4f:	48 89 3e             	mov    %rdi,(%rsi)
  400d52:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d57:	c3                   	retq   
  400d58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d5f:	00 

0000000000400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d65:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d6a:	c3                   	retq   
  400d6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d70 <__libc_csu_init>:
  400d70:	41 57                	push   %r15
  400d72:	41 56                	push   %r14
  400d74:	41 89 ff             	mov    %edi,%r15d
  400d77:	41 55                	push   %r13
  400d79:	41 54                	push   %r12
  400d7b:	4c 8d 25 5e 10 20 00 	lea    0x20105e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d82:	55                   	push   %rbp
  400d83:	48 8d 2d 5e 10 20 00 	lea    0x20105e(%rip),%rbp        # 601de8 <__init_array_end>
  400d8a:	53                   	push   %rbx
  400d8b:	49 89 f6             	mov    %rsi,%r14
  400d8e:	49 89 d5             	mov    %rdx,%r13
  400d91:	4c 29 e5             	sub    %r12,%rbp
  400d94:	48 83 ec 08          	sub    $0x8,%rsp
  400d98:	48 c1 fd 03          	sar    $0x3,%rbp
  400d9c:	e8 cf f7 ff ff       	callq  400570 <_init>
  400da1:	48 85 ed             	test   %rbp,%rbp
  400da4:	74 20                	je     400dc6 <__libc_csu_init+0x56>
  400da6:	31 db                	xor    %ebx,%ebx
  400da8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400daf:	00 
  400db0:	4c 89 ea             	mov    %r13,%rdx
  400db3:	4c 89 f6             	mov    %r14,%rsi
  400db6:	44 89 ff             	mov    %r15d,%edi
  400db9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dbd:	48 83 c3 01          	add    $0x1,%rbx
  400dc1:	48 39 eb             	cmp    %rbp,%rbx
  400dc4:	75 ea                	jne    400db0 <__libc_csu_init+0x40>
  400dc6:	48 83 c4 08          	add    $0x8,%rsp
  400dca:	5b                   	pop    %rbx
  400dcb:	5d                   	pop    %rbp
  400dcc:	41 5c                	pop    %r12
  400dce:	41 5d                	pop    %r13
  400dd0:	41 5e                	pop    %r14
  400dd2:	41 5f                	pop    %r15
  400dd4:	c3                   	retq   
  400dd5:	90                   	nop
  400dd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddd:	00 00 00 

0000000000400de0 <__libc_csu_fini>:
  400de0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400de4 <_fini>:
  400de4:	48 83 ec 08          	sub    $0x8,%rsp
  400de8:	48 83 c4 08          	add    $0x8,%rsp
  400dec:	c3                   	retq   
