
bin/clang++-5.0_-O0_DynamicStorage_4:     file format elf64-x86-64


Disassembly of section .init:

0000000000400588 <_init>:
  400588:	48 83 ec 08          	sub    $0x8,%rsp
  40058c:	48 8b 05 65 1a 20 00 	mov    0x201a65(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  400593:	48 85 c0             	test   %rax,%rax
  400596:	74 05                	je     40059d <_init+0x15>
  400598:	e8 73 00 00 00       	callq  400610 <_Unwind_Resume@plt+0x10>
  40059d:	48 83 c4 08          	add    $0x8,%rsp
  4005a1:	c3                   	retq   

Disassembly of section .plt:

00000000004005b0 <printf@plt-0x10>:
  4005b0:	ff 35 52 1a 20 00    	pushq  0x201a52(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4005b6:	ff 25 54 1a 20 00    	jmpq   *0x201a54(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4005bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005c0 <printf@plt>:
  4005c0:	ff 25 52 1a 20 00    	jmpq   *0x201a52(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005c6:	68 00 00 00 00       	pushq  $0x0
  4005cb:	e9 e0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005d0 <malloc@plt>:
  4005d0:	ff 25 4a 1a 20 00    	jmpq   *0x201a4a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005d6:	68 01 00 00 00       	pushq  $0x1
  4005db:	e9 d0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005e0 <__libc_start_main@plt>:
  4005e0:	ff 25 42 1a 20 00    	jmpq   *0x201a42(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4005e6:	68 02 00 00 00       	pushq  $0x2
  4005eb:	e9 c0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005f0 <__gxx_personality_v0@plt>:
  4005f0:	ff 25 3a 1a 20 00    	jmpq   *0x201a3a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4005f6:	68 03 00 00 00       	pushq  $0x3
  4005fb:	e9 b0 ff ff ff       	jmpq   4005b0 <_init+0x28>

0000000000400600 <_Unwind_Resume@plt>:
  400600:	ff 25 32 1a 20 00    	jmpq   *0x201a32(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400606:	68 04 00 00 00       	pushq  $0x4
  40060b:	e9 a0 ff ff ff       	jmpq   4005b0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400610 <.plt.got>:
  400610:	ff 25 e2 19 20 00    	jmpq   *0x2019e2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400616:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400620 <_start>:
  400620:	31 ed                	xor    %ebp,%ebp
  400622:	49 89 d1             	mov    %rdx,%r9
  400625:	5e                   	pop    %rsi
  400626:	48 89 e2             	mov    %rsp,%rdx
  400629:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40062d:	50                   	push   %rax
  40062e:	54                   	push   %rsp
  40062f:	49 c7 c0 00 0f 40 00 	mov    $0x400f00,%r8
  400636:	48 c7 c1 90 0e 40 00 	mov    $0x400e90,%rcx
  40063d:	48 c7 c7 20 07 40 00 	mov    $0x400720,%rdi
  400644:	e8 97 ff ff ff       	callq  4005e0 <__libc_start_main@plt>
  400649:	f4                   	hlt    
  40064a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400650 <deregister_tm_clones>:
  400650:	b8 57 20 60 00       	mov    $0x602057,%eax
  400655:	55                   	push   %rbp
  400656:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  40065c:	48 83 f8 0e          	cmp    $0xe,%rax
  400660:	48 89 e5             	mov    %rsp,%rbp
  400663:	76 1b                	jbe    400680 <deregister_tm_clones+0x30>
  400665:	b8 00 00 00 00       	mov    $0x0,%eax
  40066a:	48 85 c0             	test   %rax,%rax
  40066d:	74 11                	je     400680 <deregister_tm_clones+0x30>
  40066f:	5d                   	pop    %rbp
  400670:	bf 50 20 60 00       	mov    $0x602050,%edi
  400675:	ff e0                	jmpq   *%rax
  400677:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40067e:	00 00 
  400680:	5d                   	pop    %rbp
  400681:	c3                   	retq   
  400682:	0f 1f 40 00          	nopl   0x0(%rax)
  400686:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40068d:	00 00 00 

0000000000400690 <register_tm_clones>:
  400690:	be 50 20 60 00       	mov    $0x602050,%esi
  400695:	55                   	push   %rbp
  400696:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  40069d:	48 c1 fe 03          	sar    $0x3,%rsi
  4006a1:	48 89 e5             	mov    %rsp,%rbp
  4006a4:	48 89 f0             	mov    %rsi,%rax
  4006a7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006ab:	48 01 c6             	add    %rax,%rsi
  4006ae:	48 d1 fe             	sar    %rsi
  4006b1:	74 15                	je     4006c8 <register_tm_clones+0x38>
  4006b3:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b8:	48 85 c0             	test   %rax,%rax
  4006bb:	74 0b                	je     4006c8 <register_tm_clones+0x38>
  4006bd:	5d                   	pop    %rbp
  4006be:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006c3:	ff e0                	jmpq   *%rax
  4006c5:	0f 1f 00             	nopl   (%rax)
  4006c8:	5d                   	pop    %rbp
  4006c9:	c3                   	retq   
  4006ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006d0 <__do_global_dtors_aux>:
  4006d0:	80 3d 79 19 20 00 00 	cmpb   $0x0,0x201979(%rip)        # 602050 <__TMC_END__>
  4006d7:	75 11                	jne    4006ea <__do_global_dtors_aux+0x1a>
  4006d9:	55                   	push   %rbp
  4006da:	48 89 e5             	mov    %rsp,%rbp
  4006dd:	e8 6e ff ff ff       	callq  400650 <deregister_tm_clones>
  4006e2:	5d                   	pop    %rbp
  4006e3:	c6 05 66 19 20 00 01 	movb   $0x1,0x201966(%rip)        # 602050 <__TMC_END__>
  4006ea:	f3 c3                	repz retq 
  4006ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006f0 <frame_dummy>:
  4006f0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  4006f5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4006f9:	75 05                	jne    400700 <frame_dummy+0x10>
  4006fb:	eb 93                	jmp    400690 <register_tm_clones>
  4006fd:	0f 1f 00             	nopl   (%rax)
  400700:	b8 00 00 00 00       	mov    $0x0,%eax
  400705:	48 85 c0             	test   %rax,%rax
  400708:	74 f1                	je     4006fb <frame_dummy+0xb>
  40070a:	55                   	push   %rbp
  40070b:	48 89 e5             	mov    %rsp,%rbp
  40070e:	ff d0                	callq  *%rax
  400710:	5d                   	pop    %rbp
  400711:	e9 7a ff ff ff       	jmpq   400690 <register_tm_clones>
  400716:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40071d:	00 00 00 

0000000000400720 <main>:
  400720:	48 83 ec 58          	sub    $0x58,%rsp
  400724:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40072b:	00 
  40072c:	e8 7f 02 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400731:	b8 04 00 00 00       	mov    $0x4,%eax
  400736:	89 c7                	mov    %eax,%edi
  400738:	e8 83 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  40073d:	48 89 c7             	mov    %rax,%rdi
  400740:	48 89 c1             	mov    %rax,%rcx
  400743:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400749:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40074e:	48 89 c7             	mov    %rax,%rdi
  400751:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400756:	e8 95 02 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  40075b:	e9 00 00 00 00       	jmpq   400760 <main+0x40>
  400760:	be 77 77 00 00       	mov    $0x7777,%esi
  400765:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40076a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40076f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400774:	48 83 c1 04          	add    $0x4,%rcx
  400778:	48 89 cf             	mov    %rcx,%rdi
  40077b:	e8 c0 02 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  400780:	be 88 88 00 00       	mov    $0x8888,%esi
  400785:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40078a:	48 83 c1 04          	add    $0x4,%rcx
  40078e:	48 89 cf             	mov    %rcx,%rdi
  400791:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400796:	e8 e5 02 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40079b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a0:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007a5:	e8 16 03 00 00       	callq  400ac0 <_ZN9TestClass15increase_field0Ev>
  4007aa:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007af:	e8 4c 03 00 00       	callq  400b00 <_ZN9TestClass15increase_field1Ev>
  4007b4:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007b9:	48 83 c0 04          	add    $0x4,%rax
  4007bd:	48 89 c7             	mov    %rax,%rdi
  4007c0:	e8 7b 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  4007c5:	8b 30                	mov    (%rax),%esi
  4007c7:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007cc:	48 83 c0 04          	add    $0x4,%rax
  4007d0:	48 89 c7             	mov    %rax,%rdi
  4007d3:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007d7:	e8 74 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  4007dc:	48 bf 14 0f 40 00 00 	movabs $0x400f14,%rdi
  4007e3:	00 00 00 
  4007e6:	8b 10                	mov    (%rax),%edx
  4007e8:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007ec:	b0 00                	mov    $0x0,%al
  4007ee:	e8 cd fd ff ff       	callq  4005c0 <printf@plt>
  4007f3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007f8:	48 83 c1 04          	add    $0x4,%rcx
  4007fc:	48 89 cf             	mov    %rcx,%rdi
  4007ff:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400803:	e8 38 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400808:	31 d2                	xor    %edx,%edx
  40080a:	41 88 d0             	mov    %dl,%r8b
  40080d:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400813:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  400818:	0f 85 1e 00 00 00    	jne    40083c <main+0x11c>
  40081e:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400823:	48 83 c0 04          	add    $0x4,%rax
  400827:	48 89 c7             	mov    %rax,%rdi
  40082a:	e8 21 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  40082f:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400835:	0f 94 c1             	sete   %cl
  400838:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40083c:	8a 44 24 0f          	mov    0xf(%rsp),%al
  400840:	34 ff                	xor    $0xff,%al
  400842:	24 01                	and    $0x1,%al
  400844:	0f b6 c0             	movzbl %al,%eax
  400847:	48 83 c4 58          	add    $0x58,%rsp
  40084b:	c3                   	retq   
  40084c:	89 d1                	mov    %edx,%ecx
  40084e:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400853:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400857:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40085c:	e8 cf 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  400861:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400866:	e8 95 fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  40086b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400870 <_Z12new_instancev>:
  400870:	48 83 ec 28          	sub    $0x28,%rsp
  400874:	b8 04 00 00 00       	mov    $0x4,%eax
  400879:	89 c7                	mov    %eax,%edi
  40087b:	e8 40 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400880:	48 89 c7             	mov    %rax,%rdi
  400883:	48 89 c1             	mov    %rax,%rcx
  400886:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40088c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400891:	48 89 c7             	mov    %rax,%rdi
  400894:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400899:	e8 52 01 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  40089e:	e9 00 00 00 00       	jmpq   4008a3 <_Z12new_instancev+0x33>
  4008a3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008a8:	48 83 c4 28          	add    $0x28,%rsp
  4008ac:	c3                   	retq   
  4008ad:	89 d1                	mov    %edx,%ecx
  4008af:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4008b4:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  4008b8:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008bd:	e8 6e 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008c2:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008c7:	e8 34 fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  4008cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008d0 <_Z12write_field0P9TestClass>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008de:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008e3:	48 83 c7 04          	add    $0x4,%rdi
  4008e7:	e8 54 01 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  4008ec:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008f1:	48 83 c4 18          	add    $0x18,%rsp
  4008f5:	c3                   	retq   
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_Z11read_field0P9TestClass>:
  400900:	50                   	push   %rax
  400901:	48 89 3c 24          	mov    %rdi,(%rsp)
  400905:	48 8b 3c 24          	mov    (%rsp),%rdi
  400909:	48 83 c7 04          	add    $0x4,%rdi
  40090d:	e8 2e 02 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400912:	8b 00                	mov    (%rax),%eax
  400914:	59                   	pop    %rcx
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z12write_field1P9TestClass>:
  400920:	48 83 ec 18          	sub    $0x18,%rsp
  400924:	be 77 77 00 00       	mov    $0x7777,%esi
  400929:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40092e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400933:	48 83 c7 04          	add    $0x4,%rdi
  400937:	e8 44 01 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40093c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400941:	48 83 c4 18          	add    $0x18,%rsp
  400945:	c3                   	retq   
  400946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094d:	00 00 00 

0000000000400950 <_Z11read_field1P9TestClass>:
  400950:	50                   	push   %rax
  400951:	48 89 3c 24          	mov    %rdi,(%rsp)
  400955:	48 8b 3c 24          	mov    (%rsp),%rdi
  400959:	48 83 c7 04          	add    $0x4,%rdi
  40095d:	e8 ee 01 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  400962:	8b 00                	mov    (%rax),%eax
  400964:	59                   	pop    %rcx
  400965:	c3                   	retq   
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 

0000000000400970 <_Z21explicit_write_field0m>:
  400970:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400975:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40097a:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  400981:	77 77 00 00 
  400985:	c3                   	retq   
  400986:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40098d:	00 00 00 

0000000000400990 <_Z20explicit_read_field0m>:
  400990:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400995:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40099a:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  4009a1:	c3                   	retq   
  4009a2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009a9:	00 00 00 
  4009ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009b0:	50                   	push   %rax
  4009b1:	e8 aa 02 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  4009b6:	48 89 c7             	mov    %rax,%rdi
  4009b9:	e8 b2 03 00 00       	callq  400d70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009be:	58                   	pop    %rax
  4009bf:	c3                   	retq   

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  4009c0:	50                   	push   %rax
  4009c1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009c5:	e8 26 04 00 00       	callq  400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009ca:	e8 91 02 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  4009cf:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009d4:	89 ce                	mov    %ecx,%esi
  4009d6:	48 89 c7             	mov    %rax,%rdi
  4009d9:	e8 42 04 00 00       	callq  400e20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009de:	48 89 c7             	mov    %rax,%rdi
  4009e1:	e8 1a 04 00 00       	callq  400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  4009e6:	59                   	pop    %rcx
  4009e7:	c3                   	retq   
  4009e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009ef:	00 

00000000004009f0 <_ZN9TestClassC2Ev>:
  4009f0:	48 83 ec 18          	sub    $0x18,%rsp
  4009f4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009f9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009fe:	48 89 f8             	mov    %rdi,%rax
  400a01:	48 83 c0 04          	add    $0x4,%rax
  400a05:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a0a:	48 89 c7             	mov    %rax,%rdi
  400a0d:	e8 4e 01 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>
  400a12:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a17:	48 83 c0 04          	add    $0x4,%rax
  400a1b:	48 89 c7             	mov    %rax,%rdi
  400a1e:	e8 4d 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>
  400a23:	48 83 c4 18          	add    $0x18,%rsp
  400a27:	c3                   	retq   
  400a28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a2f:	00 

0000000000400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a35:	c3                   	retq   
  400a36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a3d:	00 00 00 

0000000000400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>:
  400a40:	48 83 ec 28          	sub    $0x28,%rsp
  400a44:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a49:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a4d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a52:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a56:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a5b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a5f:	e8 7c 01 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a64:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a68:	89 30                	mov    %esi,(%rax)
  400a6a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a6f:	48 83 c4 28          	add    $0x28,%rsp
  400a73:	c3                   	retq   
  400a74:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a7b:	00 00 00 
  400a7e:	66 90                	xchg   %ax,%ax

0000000000400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>:
  400a80:	48 83 ec 28          	sub    $0x28,%rsp
  400a84:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a89:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a8d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a92:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a96:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a9b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a9f:	e8 4c 02 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400aa8:	89 30                	mov    %esi,(%rax)
  400aaa:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400aaf:	48 83 c4 28          	add    $0x28,%rsp
  400ab3:	c3                   	retq   
  400ab4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400abb:	00 00 00 
  400abe:	66 90                	xchg   %ax,%ax

0000000000400ac0 <_ZN9TestClass15increase_field0Ev>:
  400ac0:	48 83 ec 18          	sub    $0x18,%rsp
  400ac4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ac9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ace:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ad3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ad8:	48 83 c7 04          	add    $0x4,%rdi
  400adc:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ae0:	48 89 c7             	mov    %rax,%rdi
  400ae3:	48 8b 34 24          	mov    (%rsp),%rsi
  400ae7:	e8 94 00 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>
  400aec:	48 83 c4 18          	add    $0x18,%rsp
  400af0:	c3                   	retq   
  400af1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400af8:	00 00 00 
  400afb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b00 <_ZN9TestClass15increase_field1Ev>:
  400b00:	48 83 ec 18          	sub    $0x18,%rsp
  400b04:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b09:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b0e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b13:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b18:	48 83 c7 04          	add    $0x4,%rdi
  400b1c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b20:	48 89 c7             	mov    %rax,%rdi
  400b23:	48 8b 34 24          	mov    (%rsp),%rsi
  400b27:	e8 64 01 00 00       	callq  400c90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>
  400b2c:	48 83 c4 18          	add    $0x18,%rsp
  400b30:	c3                   	retq   
  400b31:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b38:	00 00 00 
  400b3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>:
  400b40:	50                   	push   %rax
  400b41:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b45:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b49:	e8 92 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b4e:	59                   	pop    %rcx
  400b4f:	c3                   	retq   

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>:
  400b50:	50                   	push   %rax
  400b51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b59:	e8 92 01 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b5e:	59                   	pop    %rcx
  400b5f:	c3                   	retq   

0000000000400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>:
  400b60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b65:	c3                   	retq   
  400b66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b6d:	00 00 00 

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>:
  400b70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b75:	c3                   	retq   
  400b76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7d:	00 00 00 

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>:
  400b80:	48 83 ec 38          	sub    $0x38,%rsp
  400b84:	48 89 f8             	mov    %rdi,%rax
  400b87:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b8c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b90:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b95:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b99:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b9e:	48 89 f7             	mov    %rsi,%rdi
  400ba1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400ba6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400baa:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400baf:	e8 2c 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bb4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bb8:	0f af 10             	imul   (%rax),%edx
  400bbb:	89 10                	mov    %edx,(%rax)
  400bbd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bc2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400bc7:	e8 24 00 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>
  400bcc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bd1:	48 83 c4 38          	add    $0x38,%rsp
  400bd5:	c3                   	retq   
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 

0000000000400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400be0:	50                   	push   %rax
  400be1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400be5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400be9:	e8 12 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bee:	59                   	pop    %rcx
  400bef:	c3                   	retq   

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>:
  400bf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bf5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bfa:	c3                   	retq   
  400bfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c00:	48 83 ec 28          	sub    $0x28,%rsp
  400c04:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c09:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c0e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400c13:	e8 48 00 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400c18:	48 89 c7             	mov    %rax,%rdi
  400c1b:	e8 50 00 00 00       	callq  400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c20:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c25:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c2a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c2f:	48 83 e8 04          	sub    $0x4,%rax
  400c33:	48 c1 e8 02          	shr    $0x2,%rax
  400c37:	48 c1 e0 02          	shl    $0x2,%rax
  400c3b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c40:	48 83 c0 00          	add    $0x0,%rax
  400c44:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c49:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c4e:	48 83 c4 28          	add    $0x28,%rsp
  400c52:	c3                   	retq   
  400c53:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c5a:	00 00 00 
  400c5d:	0f 1f 00             	nopl   (%rax)

0000000000400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>:
  400c60:	50                   	push   %rax
  400c61:	e8 1a 00 00 00       	callq  400c80 <_ZN9TestClass14storage_bufferEv>
  400c66:	59                   	pop    %rcx
  400c67:	c3                   	retq   
  400c68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c6f:	00 

0000000000400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c75:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c7a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c7e:	c3                   	retq   
  400c7f:	90                   	nop

0000000000400c80 <_ZN9TestClass14storage_bufferEv>:
  400c80:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c87:	00 00 00 
  400c8a:	c3                   	retq   
  400c8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>:
  400c90:	48 83 ec 38          	sub    $0x38,%rsp
  400c94:	48 89 f8             	mov    %rdi,%rax
  400c97:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c9c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400ca0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400ca5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400ca9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400cae:	48 89 f7             	mov    %rsi,%rdi
  400cb1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400cb6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400cba:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400cbf:	e8 2c 00 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cc4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400cc8:	0f af 10             	imul   (%rax),%edx
  400ccb:	89 10                	mov    %edx,(%rax)
  400ccd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cd2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400cd7:	e8 24 00 00 00       	callq  400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>
  400cdc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ce1:	48 83 c4 38          	add    $0x38,%rsp
  400ce5:	c3                   	retq   
  400ce6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ced:	00 00 00 

0000000000400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cf0:	50                   	push   %rax
  400cf1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cf5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cf9:	e8 12 00 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cfe:	59                   	pop    %rcx
  400cff:	c3                   	retq   

0000000000400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>:
  400d00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d05:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d0a:	c3                   	retq   
  400d0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d10:	48 83 ec 28          	sub    $0x28,%rsp
  400d14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d19:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d1e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d23:	e8 38 ff ff ff       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400d28:	48 89 c7             	mov    %rax,%rdi
  400d2b:	e8 40 ff ff ff       	callq  400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d30:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d35:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d3a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d3f:	48 83 e8 04          	sub    $0x4,%rax
  400d43:	48 c1 e8 02          	shr    $0x2,%rax
  400d47:	48 c1 e0 02          	shl    $0x2,%rax
  400d4b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d50:	48 05 d4 48 00 00    	add    $0x48d4,%rax
  400d56:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d5b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d60:	48 83 c4 28          	add    $0x28,%rsp
  400d64:	c3                   	retq   
  400d65:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d6c:	00 00 00 
  400d6f:	90                   	nop

0000000000400d70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d70:	48 83 ec 18          	sub    $0x18,%rsp
  400d74:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d79:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d7e:	48 89 f8             	mov    %rdi,%rax
  400d81:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d86:	48 89 c7             	mov    %rax,%rdi
  400d89:	e8 32 00 00 00       	callq  400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d8e:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400d93:	89 cf                	mov    %ecx,%edi
  400d95:	e8 36 f8 ff ff       	callq  4005d0 <malloc@plt>
  400d9a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400d9f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400da3:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400daa:	00 
  400dab:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400db2:	00 
  400db3:	48 83 c4 18          	add    $0x18,%rsp
  400db7:	c3                   	retq   
  400db8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dbf:	00 

0000000000400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400dc0:	50                   	push   %rax
  400dc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400dc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400dc9:	e8 02 00 00 00       	callq  400dd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400dce:	58                   	pop    %rax
  400dcf:	c3                   	retq   

0000000000400dd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400dd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dd5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400dda:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400de1:	c3                   	retq   
  400de2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400de9:	00 00 00 
  400dec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400df0:	c3                   	retq   
  400df1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400df8:	00 00 00 
  400dfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400e00:	50                   	push   %rax
  400e01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400e05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400e09:	48 83 c7 01          	add    $0x1,%rdi
  400e0d:	e8 3e 00 00 00       	callq  400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e12:	59                   	pop    %rcx
  400e13:	c3                   	retq   
  400e14:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e1b:	00 00 00 
  400e1e:	66 90                	xchg   %ax,%ax

0000000000400e20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400e2a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400e2f:	48 8b 3e             	mov    (%rsi),%rdi
  400e32:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400e37:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400e3c:	48 03 3e             	add    (%rsi),%rdi
  400e3f:	48 89 3e             	mov    %rdi,(%rsi)
  400e42:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e47:	c3                   	retq   
  400e48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e4f:	00 

0000000000400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e50:	48 83 ec 18          	sub    $0x18,%rsp
  400e54:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e59:	e8 02 fe ff ff       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400e5e:	48 89 c7             	mov    %rax,%rdi
  400e61:	e8 0a fe ff ff       	callq  400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e66:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e6b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e6f:	48 01 f8             	add    %rdi,%rax
  400e72:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e77:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e7c:	48 83 c4 18          	add    $0x18,%rsp
  400e80:	c3                   	retq   
  400e81:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e88:	00 00 00 
  400e8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e90 <__libc_csu_init>:
  400e90:	41 57                	push   %r15
  400e92:	41 56                	push   %r14
  400e94:	41 89 ff             	mov    %edi,%r15d
  400e97:	41 55                	push   %r13
  400e99:	41 54                	push   %r12
  400e9b:	4c 8d 25 3e 0f 20 00 	lea    0x200f3e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400ea2:	55                   	push   %rbp
  400ea3:	48 8d 2d 3e 0f 20 00 	lea    0x200f3e(%rip),%rbp        # 601de8 <__init_array_end>
  400eaa:	53                   	push   %rbx
  400eab:	49 89 f6             	mov    %rsi,%r14
  400eae:	49 89 d5             	mov    %rdx,%r13
  400eb1:	4c 29 e5             	sub    %r12,%rbp
  400eb4:	48 83 ec 08          	sub    $0x8,%rsp
  400eb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ebc:	e8 c7 f6 ff ff       	callq  400588 <_init>
  400ec1:	48 85 ed             	test   %rbp,%rbp
  400ec4:	74 20                	je     400ee6 <__libc_csu_init+0x56>
  400ec6:	31 db                	xor    %ebx,%ebx
  400ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ecf:	00 
  400ed0:	4c 89 ea             	mov    %r13,%rdx
  400ed3:	4c 89 f6             	mov    %r14,%rsi
  400ed6:	44 89 ff             	mov    %r15d,%edi
  400ed9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400edd:	48 83 c3 01          	add    $0x1,%rbx
  400ee1:	48 39 eb             	cmp    %rbp,%rbx
  400ee4:	75 ea                	jne    400ed0 <__libc_csu_init+0x40>
  400ee6:	48 83 c4 08          	add    $0x8,%rsp
  400eea:	5b                   	pop    %rbx
  400eeb:	5d                   	pop    %rbp
  400eec:	41 5c                	pop    %r12
  400eee:	41 5d                	pop    %r13
  400ef0:	41 5e                	pop    %r14
  400ef2:	41 5f                	pop    %r15
  400ef4:	c3                   	retq   
  400ef5:	90                   	nop
  400ef6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400efd:	00 00 00 

0000000000400f00 <__libc_csu_fini>:
  400f00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f04 <_fini>:
  400f04:	48 83 ec 08          	sub    $0x8,%rsp
  400f08:	48 83 c4 08          	add    $0x8,%rsp
  400f0c:	c3                   	retq   
