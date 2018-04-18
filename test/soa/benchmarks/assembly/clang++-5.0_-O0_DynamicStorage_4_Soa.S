
bin/clang++-5.0_-O0_DynamicStorage_4_Soa:     file format elf64-x86-64


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
  40062f:	49 c7 c0 90 0f 40 00 	mov    $0x400f90,%r8
  400636:	48 c7 c1 20 0f 40 00 	mov    $0x400f20,%rcx
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
  40072c:	e8 df 02 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIJEEEvDpT_>
  400731:	b8 04 00 00 00       	mov    $0x4,%eax
  400736:	89 c7                	mov    %eax,%edi
  400738:	e8 e3 02 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>
  40073d:	48 89 c7             	mov    %rax,%rdi
  400740:	48 89 c1             	mov    %rax,%rcx
  400743:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400749:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40074e:	48 89 c7             	mov    %rax,%rdi
  400751:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400756:	e8 f5 02 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  40075b:	e9 00 00 00 00       	jmpq   400760 <main+0x40>
  400760:	be 77 77 00 00       	mov    $0x7777,%esi
  400765:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40076a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40076f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400774:	48 83 c1 04          	add    $0x4,%rcx
  400778:	48 89 cf             	mov    %rcx,%rdi
  40077b:	e8 20 03 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>
  400780:	be 88 88 00 00       	mov    $0x8888,%esi
  400785:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40078a:	48 83 c1 04          	add    $0x4,%rcx
  40078e:	48 89 cf             	mov    %rcx,%rdi
  400791:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400796:	e8 45 03 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>
  40079b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a0:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007a5:	e8 76 03 00 00       	callq  400b20 <_ZN9TestClass15increase_field0Ev>
  4007aa:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007af:	e8 ac 03 00 00       	callq  400b60 <_ZN9TestClass15increase_field1Ev>
  4007b4:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007b9:	48 83 c0 04          	add    $0x4,%rax
  4007bd:	48 89 c7             	mov    %rax,%rdi
  4007c0:	e8 db 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
  4007c5:	8b 30                	mov    (%rax),%esi
  4007c7:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007cc:	48 83 c0 04          	add    $0x4,%rax
  4007d0:	48 89 c7             	mov    %rax,%rdi
  4007d3:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007d7:	e8 d4 03 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
  4007dc:	48 bf a4 0f 40 00 00 	movabs $0x400fa4,%rdi
  4007e3:	00 00 00 
  4007e6:	8b 10                	mov    (%rax),%edx
  4007e8:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007ec:	b0 00                	mov    $0x0,%al
  4007ee:	e8 cd fd ff ff       	callq  4005c0 <printf@plt>
  4007f3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007f8:	48 83 c1 04          	add    $0x4,%rcx
  4007fc:	48 89 cf             	mov    %rcx,%rdi
  4007ff:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400803:	e8 98 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
  400808:	31 d2                	xor    %edx,%edx
  40080a:	41 88 d0             	mov    %dl,%r8b
  40080d:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400813:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  400818:	0f 85 1e 00 00 00    	jne    40083c <main+0x11c>
  40081e:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400823:	48 83 c0 04          	add    $0x4,%rax
  400827:	48 89 c7             	mov    %rax,%rdi
  40082a:	e8 81 03 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
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
  40085c:	e8 2f 02 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEdlEPv>
  400861:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400866:	e8 95 fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  40086b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400870 <_Z12new_instancev>:
  400870:	48 83 ec 28          	sub    $0x28,%rsp
  400874:	b8 04 00 00 00       	mov    $0x4,%eax
  400879:	89 c7                	mov    %eax,%edi
  40087b:	e8 a0 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>
  400880:	48 89 c7             	mov    %rax,%rdi
  400883:	48 89 c1             	mov    %rax,%rcx
  400886:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40088c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400891:	48 89 c7             	mov    %rax,%rdi
  400894:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400899:	e8 b2 01 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  40089e:	e9 00 00 00 00       	jmpq   4008a3 <_Z12new_instancev+0x33>
  4008a3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008a8:	48 83 c4 28          	add    $0x28,%rsp
  4008ac:	c3                   	retq   
  4008ad:	89 d1                	mov    %edx,%ecx
  4008af:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4008b4:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  4008b8:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008bd:	e8 ce 01 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEdlEPv>
  4008c2:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008c7:	e8 34 fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  4008cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008d0 <_Z12write_field0P9TestClass>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008de:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008e3:	48 83 c7 04          	add    $0x4,%rdi
  4008e7:	e8 b4 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>
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
  40090d:	e8 8e 02 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
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
  400937:	e8 a4 01 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>
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
  40095d:	e8 4e 02 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
  400962:	8b 00                	mov    (%rax),%eax
  400964:	59                   	pop    %rcx
  400965:	c3                   	retq   
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 

0000000000400970 <_Z21explicit_write_field0m>:
  400970:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400975:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40097a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400981:	77 77 00 00 
  400985:	c3                   	retq   
  400986:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40098d:	00 00 00 

0000000000400990 <_Z20explicit_read_field0m>:
  400990:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400995:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40099a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  4009a1:	c3                   	retq   
  4009a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009a9:	1f 84 00 00 00 00 00 

00000000004009b0 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  4009b0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ba:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  4009c0:	c3                   	retq   
  4009c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009c8:	0f 1f 84 00 00 00 00 
  4009cf:	00 

00000000004009d0 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  4009d0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009d5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009da:	8b 07                	mov    (%rdi),%eax
  4009dc:	c3                   	retq   
  4009dd:	0f 1f 00             	nopl   (%rax)

00000000004009e0 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  4009e0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009e5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ea:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  4009f1:	c3                   	retq   
  4009f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009f9:	1f 84 00 00 00 00 00 

0000000000400a00 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  400a00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a05:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400a0a:	8b 47 04             	mov    0x4(%rdi),%eax
  400a0d:	c3                   	retq   
  400a0e:	66 90                	xchg   %ax,%ax

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIJEEEvDpT_>:
  400a10:	50                   	push   %rax
  400a11:	e8 aa 02 00 00       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400a16:	48 89 c7             	mov    %rax,%rdi
  400a19:	e8 b2 03 00 00       	callq  400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  400a1e:	58                   	pop    %rax
  400a1f:	c3                   	retq   

0000000000400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>:
  400a20:	50                   	push   %rax
  400a21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400a25:	e8 16 04 00 00       	callq  400e40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a2a:	e8 91 02 00 00       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400a2f:	b9 01 00 00 00       	mov    $0x1,%ecx
  400a34:	89 ce                	mov    %ecx,%esi
  400a36:	48 89 c7             	mov    %rax,%rdi
  400a39:	e8 32 04 00 00       	callq  400e70 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400a3e:	48 89 c7             	mov    %rax,%rdi
  400a41:	e8 0a 04 00 00       	callq  400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE3getEm>
  400a46:	59                   	pop    %rcx
  400a47:	c3                   	retq   
  400a48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a4f:	00 

0000000000400a50 <_ZN9TestClassC2Ev>:
  400a50:	48 83 ec 18          	sub    $0x18,%rsp
  400a54:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a59:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a5e:	48 89 f8             	mov    %rdi,%rax
  400a61:	48 83 c0 04          	add    $0x4,%rax
  400a65:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a6a:	48 89 c7             	mov    %rax,%rdi
  400a6d:	e8 4e 01 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC2Ev>
  400a72:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a77:	48 83 c0 04          	add    $0x4,%rax
  400a7b:	48 89 c7             	mov    %rax,%rdi
  400a7e:	e8 4d 01 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC2Ev>
  400a83:	48 83 c4 18          	add    $0x18,%rsp
  400a87:	c3                   	retq   
  400a88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a8f:	00 

0000000000400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEdlEPv>:
  400a90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a95:	c3                   	retq   
  400a96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a9d:	00 00 00 

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>:
  400aa0:	48 83 ec 28          	sub    $0x28,%rsp
  400aa4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400aa9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400aad:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ab2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400ab6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400abb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400abf:	e8 7c 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400ac4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400ac8:	89 30                	mov    %esi,(%rax)
  400aca:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400acf:	48 83 c4 28          	add    $0x28,%rsp
  400ad3:	c3                   	retq   
  400ad4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400adb:	00 00 00 
  400ade:	66 90                	xchg   %ax,%ax

0000000000400ae0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>:
  400ae0:	48 83 ec 28          	sub    $0x28,%rsp
  400ae4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400ae9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400aed:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400af2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400af6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400afb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400aff:	e8 4c 02 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400b04:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400b08:	89 30                	mov    %esi,(%rax)
  400b0a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b0f:	48 83 c4 28          	add    $0x28,%rsp
  400b13:	c3                   	retq   
  400b14:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b1b:	00 00 00 
  400b1e:	66 90                	xchg   %ax,%ax

0000000000400b20 <_ZN9TestClass15increase_field0Ev>:
  400b20:	48 83 ec 18          	sub    $0x18,%rsp
  400b24:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b29:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b2e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b33:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b38:	48 83 c7 04          	add    $0x4,%rdi
  400b3c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b40:	48 89 c7             	mov    %rax,%rdi
  400b43:	48 8b 34 24          	mov    (%rsp),%rsi
  400b47:	e8 94 00 00 00       	callq  400be0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEmLEi>
  400b4c:	48 83 c4 18          	add    $0x18,%rsp
  400b50:	c3                   	retq   
  400b51:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b58:	00 00 00 
  400b5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b60 <_ZN9TestClass15increase_field1Ev>:
  400b60:	48 83 ec 18          	sub    $0x18,%rsp
  400b64:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b69:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b6e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b73:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b78:	48 83 c7 04          	add    $0x4,%rdi
  400b7c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b80:	48 89 c7             	mov    %rax,%rdi
  400b83:	48 8b 34 24          	mov    (%rsp),%rsi
  400b87:	e8 64 01 00 00       	callq  400cf0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEmLEi>
  400b8c:	48 83 c4 18          	add    $0x18,%rsp
  400b90:	c3                   	retq   
  400b91:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b98:	00 00 00 
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>:
  400ba0:	50                   	push   %rax
  400ba1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ba5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ba9:	e8 92 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400bae:	59                   	pop    %rcx
  400baf:	c3                   	retq   

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>:
  400bb0:	50                   	push   %rax
  400bb1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bb5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bb9:	e8 92 01 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400bbe:	59                   	pop    %rcx
  400bbf:	c3                   	retq   

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC2Ev>:
  400bc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bc5:	c3                   	retq   
  400bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bcd:	00 00 00 

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC2Ev>:
  400bd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bd5:	c3                   	retq   
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 

0000000000400be0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEmLEi>:
  400be0:	48 83 ec 38          	sub    $0x38,%rsp
  400be4:	48 89 f8             	mov    %rdi,%rax
  400be7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bec:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400bf0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400bf5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400bf9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bfe:	48 89 f7             	mov    %rsi,%rdi
  400c01:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c06:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c0a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c0f:	e8 2c 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400c14:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c18:	0f af 10             	imul   (%rax),%edx
  400c1b:	89 10                	mov    %edx,(%rax)
  400c1d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c22:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c27:	e8 24 00 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC2ERKS3_>
  400c2c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c31:	48 83 c4 38          	add    $0x38,%rsp
  400c35:	c3                   	retq   
  400c36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c3d:	00 00 00 

0000000000400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>:
  400c40:	50                   	push   %rax
  400c41:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c45:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c49:	e8 12 00 00 00       	callq  400c60 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400c4e:	59                   	pop    %rcx
  400c4f:	c3                   	retq   

0000000000400c50 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC2ERKS3_>:
  400c50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c55:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c5a:	c3                   	retq   
  400c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c60 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400c60:	48 83 ec 28          	sub    $0x28,%rsp
  400c64:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c69:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c6e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400c73:	e8 48 00 00 00       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400c78:	48 89 c7             	mov    %rax,%rdi
  400c7b:	e8 50 00 00 00       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c80:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c85:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c8a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c8f:	48 83 e8 04          	sub    $0x4,%rax
  400c93:	48 c1 e8 02          	shr    $0x2,%rax
  400c97:	48 c1 e0 02          	shl    $0x2,%rax
  400c9b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400ca0:	48 83 c0 00          	add    $0x0,%rax
  400ca4:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400ca9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cae:	48 83 c4 28          	add    $0x28,%rsp
  400cb2:	c3                   	retq   
  400cb3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cba:	00 00 00 
  400cbd:	0f 1f 00             	nopl   (%rax)

0000000000400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>:
  400cc0:	50                   	push   %rax
  400cc1:	e8 1a 00 00 00       	callq  400ce0 <_ZN9TestClass14storage_bufferEv>
  400cc6:	59                   	pop    %rcx
  400cc7:	c3                   	retq   
  400cc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ccf:	00 

0000000000400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400cd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cd5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400cda:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400cde:	c3                   	retq   
  400cdf:	90                   	nop

0000000000400ce0 <_ZN9TestClass14storage_bufferEv>:
  400ce0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400ce7:	00 00 00 
  400cea:	c3                   	retq   
  400ceb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cf0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEmLEi>:
  400cf0:	48 83 ec 38          	sub    $0x38,%rsp
  400cf4:	48 89 f8             	mov    %rdi,%rax
  400cf7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400cfc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400d00:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400d05:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400d09:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d0e:	48 89 f7             	mov    %rsi,%rdi
  400d11:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d16:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400d1a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400d1f:	e8 2c 00 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400d24:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400d28:	0f af 10             	imul   (%rax),%edx
  400d2b:	89 10                	mov    %edx,(%rax)
  400d2d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d32:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400d37:	e8 24 00 00 00       	callq  400d60 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC2ERKS3_>
  400d3c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d41:	48 83 c4 38          	add    $0x38,%rsp
  400d45:	c3                   	retq   
  400d46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d4d:	00 00 00 

0000000000400d50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>:
  400d50:	50                   	push   %rax
  400d51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d59:	e8 12 00 00 00       	callq  400d70 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400d5e:	59                   	pop    %rcx
  400d5f:	c3                   	retq   

0000000000400d60 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC2ERKS3_>:
  400d60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d65:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d6a:	c3                   	retq   
  400d6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d70 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d70:	48 83 ec 28          	sub    $0x28,%rsp
  400d74:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d79:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d7e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d83:	e8 38 ff ff ff       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400d88:	48 89 c7             	mov    %rax,%rdi
  400d8b:	e8 40 ff ff ff       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d90:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d95:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d9a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d9f:	48 83 e8 04          	sub    $0x4,%rax
  400da3:	48 c1 e8 02          	shr    $0x2,%rax
  400da7:	48 c1 e0 02          	shl    $0x2,%rax
  400dab:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400db0:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400db6:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400dbb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400dc0:	48 83 c4 28          	add    $0x28,%rsp
  400dc4:	c3                   	retq   
  400dc5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcc:	00 00 00 
  400dcf:	90                   	nop

0000000000400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400dd0:	48 83 ec 18          	sub    $0x18,%rsp
  400dd4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400dd9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400dde:	48 89 f8             	mov    %rdi,%rax
  400de1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400de6:	48 89 c7             	mov    %rax,%rdi
  400de9:	e8 32 00 00 00       	callq  400e20 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400dee:	b9 c0 91 00 00       	mov    $0x91c0,%ecx
  400df3:	89 cf                	mov    %ecx,%edi
  400df5:	e8 d6 f7 ff ff       	callq  4005d0 <malloc@plt>
  400dfa:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400dff:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400e03:	c7 47 10 00 00 00 00 	movl   $0x0,0x10(%rdi)
  400e0a:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400e11:	00 
  400e12:	48 83 c4 18          	add    $0x18,%rsp
  400e16:	c3                   	retq   
  400e17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e1e:	00 00 

0000000000400e20 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400e20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e25:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400e2a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400e31:	c3                   	retq   
  400e32:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e39:	00 00 00 
  400e3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400e40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e40:	c3                   	retq   
  400e41:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e48:	00 00 00 
  400e4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE3getEm>:
  400e50:	50                   	push   %rax
  400e51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400e55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400e59:	48 83 c7 01          	add    $0x1,%rdi
  400e5d:	e8 4e 00 00 00       	callq  400eb0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e62:	59                   	pop    %rcx
  400e63:	c3                   	retq   
  400e64:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e6b:	00 00 00 
  400e6e:	66 90                	xchg   %ax,%ax

0000000000400e70 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400e70:	48 83 ec 28          	sub    $0x28,%rsp
  400e74:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400e79:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  400e7e:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  400e83:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400e88:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  400e8d:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400e92:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e97:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400e9c:	e8 4f 00 00 00       	callq  400ef0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400ea1:	48 83 c4 28          	add    $0x28,%rsp
  400ea5:	c3                   	retq   
  400ea6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ead:	00 00 00 

0000000000400eb0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400eb0:	48 83 ec 18          	sub    $0x18,%rsp
  400eb4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400eb9:	e8 02 fe ff ff       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400ebe:	48 89 c7             	mov    %rax,%rdi
  400ec1:	e8 0a fe ff ff       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400ec6:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ecb:	48 c1 e7 02          	shl    $0x2,%rdi
  400ecf:	48 01 f8             	add    %rdi,%rax
  400ed2:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400ed7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400edc:	48 83 c4 18          	add    $0x18,%rsp
  400ee0:	c3                   	retq   
  400ee1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ee8:	00 00 00 
  400eeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ef0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400ef0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ef5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400efa:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400eff:	48 8b 36             	mov    (%rsi),%rsi
  400f02:	48 89 74 24 e8       	mov    %rsi,-0x18(%rsp)
  400f07:	48 8b 74 24 f0       	mov    -0x10(%rsp),%rsi
  400f0c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400f11:	48 03 37             	add    (%rdi),%rsi
  400f14:	48 89 37             	mov    %rsi,(%rdi)
  400f17:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400f1c:	c3                   	retq   
  400f1d:	0f 1f 00             	nopl   (%rax)

0000000000400f20 <__libc_csu_init>:
  400f20:	41 57                	push   %r15
  400f22:	41 56                	push   %r14
  400f24:	41 89 ff             	mov    %edi,%r15d
  400f27:	41 55                	push   %r13
  400f29:	41 54                	push   %r12
  400f2b:	4c 8d 25 ae 0e 20 00 	lea    0x200eae(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400f32:	55                   	push   %rbp
  400f33:	48 8d 2d ae 0e 20 00 	lea    0x200eae(%rip),%rbp        # 601de8 <__init_array_end>
  400f3a:	53                   	push   %rbx
  400f3b:	49 89 f6             	mov    %rsi,%r14
  400f3e:	49 89 d5             	mov    %rdx,%r13
  400f41:	4c 29 e5             	sub    %r12,%rbp
  400f44:	48 83 ec 08          	sub    $0x8,%rsp
  400f48:	48 c1 fd 03          	sar    $0x3,%rbp
  400f4c:	e8 37 f6 ff ff       	callq  400588 <_init>
  400f51:	48 85 ed             	test   %rbp,%rbp
  400f54:	74 20                	je     400f76 <__libc_csu_init+0x56>
  400f56:	31 db                	xor    %ebx,%ebx
  400f58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f5f:	00 
  400f60:	4c 89 ea             	mov    %r13,%rdx
  400f63:	4c 89 f6             	mov    %r14,%rsi
  400f66:	44 89 ff             	mov    %r15d,%edi
  400f69:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400f6d:	48 83 c3 01          	add    $0x1,%rbx
  400f71:	48 39 eb             	cmp    %rbp,%rbx
  400f74:	75 ea                	jne    400f60 <__libc_csu_init+0x40>
  400f76:	48 83 c4 08          	add    $0x8,%rsp
  400f7a:	5b                   	pop    %rbx
  400f7b:	5d                   	pop    %rbp
  400f7c:	41 5c                	pop    %r12
  400f7e:	41 5d                	pop    %r13
  400f80:	41 5e                	pop    %r14
  400f82:	41 5f                	pop    %r15
  400f84:	c3                   	retq   
  400f85:	90                   	nop
  400f86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f8d:	00 00 00 

0000000000400f90 <__libc_csu_fini>:
  400f90:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f94 <_fini>:
  400f94:	48 83 ec 08          	sub    $0x8,%rsp
  400f98:	48 83 c4 08          	add    $0x8,%rsp
  400f9c:	c3                   	retq   
