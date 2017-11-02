
bin/g++_-O0_DynamicStorage_4:     file format elf64-x86-64


Disassembly of section .init:

0000000000400460 <_init>:
  400460:	48 83 ec 08          	sub    $0x8,%rsp
  400464:	48 8b 05 8d 1b 20 00 	mov    0x201b8d(%rip),%rax        # 601ff8 <_DYNAMIC+0x1d0>
  40046b:	48 85 c0             	test   %rax,%rax
  40046e:	74 05                	je     400475 <_init+0x15>
  400470:	e8 5b 00 00 00       	callq  4004d0 <malloc@plt+0x10>
  400475:	48 83 c4 08          	add    $0x8,%rsp
  400479:	c3                   	retq   

Disassembly of section .plt:

0000000000400480 <__stack_chk_fail@plt-0x10>:
  400480:	ff 35 82 1b 20 00    	pushq  0x201b82(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400486:	ff 25 84 1b 20 00    	jmpq   *0x201b84(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40048c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400490 <__stack_chk_fail@plt>:
  400490:	ff 25 82 1b 20 00    	jmpq   *0x201b82(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400496:	68 00 00 00 00       	pushq  $0x0
  40049b:	e9 e0 ff ff ff       	jmpq   400480 <_init+0x20>

00000000004004a0 <printf@plt>:
  4004a0:	ff 25 7a 1b 20 00    	jmpq   *0x201b7a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4004a6:	68 01 00 00 00       	pushq  $0x1
  4004ab:	e9 d0 ff ff ff       	jmpq   400480 <_init+0x20>

00000000004004b0 <__libc_start_main@plt>:
  4004b0:	ff 25 72 1b 20 00    	jmpq   *0x201b72(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4004b6:	68 02 00 00 00       	pushq  $0x2
  4004bb:	e9 c0 ff ff ff       	jmpq   400480 <_init+0x20>

00000000004004c0 <malloc@plt>:
  4004c0:	ff 25 6a 1b 20 00    	jmpq   *0x201b6a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4004c6:	68 03 00 00 00       	pushq  $0x3
  4004cb:	e9 b0 ff ff ff       	jmpq   400480 <_init+0x20>

Disassembly of section .plt.got:

00000000004004d0 <.plt.got>:
  4004d0:	ff 25 22 1b 20 00    	jmpq   *0x201b22(%rip)        # 601ff8 <_DYNAMIC+0x1d0>
  4004d6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004e0 <_start>:
  4004e0:	31 ed                	xor    %ebp,%ebp
  4004e2:	49 89 d1             	mov    %rdx,%r9
  4004e5:	5e                   	pop    %rsi
  4004e6:	48 89 e2             	mov    %rsp,%rdx
  4004e9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004ed:	50                   	push   %rax
  4004ee:	54                   	push   %rsp
  4004ef:	49 c7 c0 10 0d 40 00 	mov    $0x400d10,%r8
  4004f6:	48 c7 c1 a0 0c 40 00 	mov    $0x400ca0,%rcx
  4004fd:	48 c7 c7 e1 05 40 00 	mov    $0x4005e1,%rdi
  400504:	e8 a7 ff ff ff       	callq  4004b0 <__libc_start_main@plt>
  400509:	f4                   	hlt    
  40050a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400510 <deregister_tm_clones>:
  400510:	b8 4f 20 60 00       	mov    $0x60204f,%eax
  400515:	55                   	push   %rbp
  400516:	48 2d 48 20 60 00    	sub    $0x602048,%rax
  40051c:	48 83 f8 0e          	cmp    $0xe,%rax
  400520:	48 89 e5             	mov    %rsp,%rbp
  400523:	76 1b                	jbe    400540 <deregister_tm_clones+0x30>
  400525:	b8 00 00 00 00       	mov    $0x0,%eax
  40052a:	48 85 c0             	test   %rax,%rax
  40052d:	74 11                	je     400540 <deregister_tm_clones+0x30>
  40052f:	5d                   	pop    %rbp
  400530:	bf 48 20 60 00       	mov    $0x602048,%edi
  400535:	ff e0                	jmpq   *%rax
  400537:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40053e:	00 00 
  400540:	5d                   	pop    %rbp
  400541:	c3                   	retq   
  400542:	0f 1f 40 00          	nopl   0x0(%rax)
  400546:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40054d:	00 00 00 

0000000000400550 <register_tm_clones>:
  400550:	be 48 20 60 00       	mov    $0x602048,%esi
  400555:	55                   	push   %rbp
  400556:	48 81 ee 48 20 60 00 	sub    $0x602048,%rsi
  40055d:	48 c1 fe 03          	sar    $0x3,%rsi
  400561:	48 89 e5             	mov    %rsp,%rbp
  400564:	48 89 f0             	mov    %rsi,%rax
  400567:	48 c1 e8 3f          	shr    $0x3f,%rax
  40056b:	48 01 c6             	add    %rax,%rsi
  40056e:	48 d1 fe             	sar    %rsi
  400571:	74 15                	je     400588 <register_tm_clones+0x38>
  400573:	b8 00 00 00 00       	mov    $0x0,%eax
  400578:	48 85 c0             	test   %rax,%rax
  40057b:	74 0b                	je     400588 <register_tm_clones+0x38>
  40057d:	5d                   	pop    %rbp
  40057e:	bf 48 20 60 00       	mov    $0x602048,%edi
  400583:	ff e0                	jmpq   *%rax
  400585:	0f 1f 00             	nopl   (%rax)
  400588:	5d                   	pop    %rbp
  400589:	c3                   	retq   
  40058a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400590 <__do_global_dtors_aux>:
  400590:	80 3d c9 1a 20 00 00 	cmpb   $0x0,0x201ac9(%rip)        # 602060 <completed.7585>
  400597:	75 11                	jne    4005aa <__do_global_dtors_aux+0x1a>
  400599:	55                   	push   %rbp
  40059a:	48 89 e5             	mov    %rsp,%rbp
  40059d:	e8 6e ff ff ff       	callq  400510 <deregister_tm_clones>
  4005a2:	5d                   	pop    %rbp
  4005a3:	c6 05 b6 1a 20 00 01 	movb   $0x1,0x201ab6(%rip)        # 602060 <completed.7585>
  4005aa:	f3 c3                	repz retq 
  4005ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005b0 <frame_dummy>:
  4005b0:	bf 20 1e 60 00       	mov    $0x601e20,%edi
  4005b5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005b9:	75 05                	jne    4005c0 <frame_dummy+0x10>
  4005bb:	eb 93                	jmp    400550 <register_tm_clones>
  4005bd:	0f 1f 00             	nopl   (%rax)
  4005c0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c5:	48 85 c0             	test   %rax,%rax
  4005c8:	74 f1                	je     4005bb <frame_dummy+0xb>
  4005ca:	55                   	push   %rbp
  4005cb:	48 89 e5             	mov    %rsp,%rbp
  4005ce:	ff d0                	callq  *%rax
  4005d0:	5d                   	pop    %rbp
  4005d1:	e9 7a ff ff ff       	jmpq   400550 <register_tm_clones>

00000000004005d6 <_ZN9TestClass7storageEv>:
  4005d6:	55                   	push   %rbp
  4005d7:	48 89 e5             	mov    %rsp,%rbp
  4005da:	b8 80 20 60 00       	mov    $0x602080,%eax
  4005df:	5d                   	pop    %rbp
  4005e0:	c3                   	retq   

00000000004005e1 <main>:
  4005e1:	55                   	push   %rbp
  4005e2:	48 89 e5             	mov    %rsp,%rbp
  4005e5:	53                   	push   %rbx
  4005e6:	48 83 ec 28          	sub    $0x28,%rsp
  4005ea:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005f1:	00 00 
  4005f3:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005f7:	31 c0                	xor    %eax,%eax
  4005f9:	e8 ec 03 00 00       	callq  4009ea <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageEv>
  4005fe:	bf 04 00 00 00       	mov    $0x4,%edi
  400603:	e8 08 04 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400608:	48 89 c3             	mov    %rax,%rbx
  40060b:	b8 00 00 00 00       	mov    $0x0,%eax
  400610:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400614:	48 83 f8 03          	cmp    $0x3,%rax
  400618:	74 06                	je     400620 <main+0x3f>
  40061a:	48 83 c0 01          	add    $0x1,%rax
  40061e:	eb f0                	jmp    400610 <main+0x2f>
  400620:	48 89 df             	mov    %rbx,%rdi
  400623:	e8 06 03 00 00       	callq  40092e <_ZN9TestClassC1Ev>
  400628:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  40062c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400630:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400634:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400638:	ba 77 77 00 00       	mov    $0x7777,%edx
  40063d:	48 89 ce             	mov    %rcx,%rsi
  400640:	48 89 c7             	mov    %rax,%rdi
  400643:	e8 18 04 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400648:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064c:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400650:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400654:	ba 88 88 00 00       	mov    $0x8888,%edx
  400659:	48 89 ce             	mov    %rcx,%rsi
  40065c:	48 89 c7             	mov    %rax,%rdi
  40065f:	e8 3c 04 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400664:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400668:	48 89 c7             	mov    %rax,%rdi
  40066b:	e8 fa 01 00 00       	callq  40086a <_ZN9TestClass15increase_field0Ev>
  400670:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400674:	48 89 c7             	mov    %rax,%rdi
  400677:	e8 3c 02 00 00       	callq  4008b8 <_ZN9TestClass15increase_field1Ev>
  40067c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400680:	48 83 c0 04          	add    $0x4,%rax
  400684:	48 89 c7             	mov    %rax,%rdi
  400687:	e8 6e 04 00 00       	callq  400afa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40068c:	8b 18                	mov    (%rax),%ebx
  40068e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400692:	48 83 c0 04          	add    $0x4,%rax
  400696:	48 89 c7             	mov    %rax,%rdi
  400699:	e8 42 04 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40069e:	8b 00                	mov    (%rax),%eax
  4006a0:	89 da                	mov    %ebx,%edx
  4006a2:	89 c6                	mov    %eax,%esi
  4006a4:	bf 34 0d 40 00       	mov    $0x400d34,%edi
  4006a9:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ae:	e8 ed fd ff ff       	callq  4004a0 <printf@plt>
  4006b3:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006b7:	48 83 c0 04          	add    $0x4,%rax
  4006bb:	48 89 c7             	mov    %rax,%rdi
  4006be:	e8 1d 04 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006c3:	8b 00                	mov    (%rax),%eax
  4006c5:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006ca:	75 19                	jne    4006e5 <main+0x104>
  4006cc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006d0:	48 83 c0 04          	add    $0x4,%rax
  4006d4:	48 89 c7             	mov    %rax,%rdi
  4006d7:	e8 1e 04 00 00       	callq  400afa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4006dc:	8b 00                	mov    (%rax),%eax
  4006de:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006e3:	74 07                	je     4006ec <main+0x10b>
  4006e5:	b8 01 00 00 00       	mov    $0x1,%eax
  4006ea:	eb 05                	jmp    4006f1 <main+0x110>
  4006ec:	b8 00 00 00 00       	mov    $0x0,%eax
  4006f1:	0f b6 c0             	movzbl %al,%eax
  4006f4:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006f8:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006ff:	00 00 
  400701:	74 05                	je     400708 <main+0x127>
  400703:	e8 88 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  400708:	48 83 c4 28          	add    $0x28,%rsp
  40070c:	5b                   	pop    %rbx
  40070d:	5d                   	pop    %rbp
  40070e:	c3                   	retq   

000000000040070f <_Z12new_instancev>:
  40070f:	55                   	push   %rbp
  400710:	48 89 e5             	mov    %rsp,%rbp
  400713:	53                   	push   %rbx
  400714:	48 83 ec 08          	sub    $0x8,%rsp
  400718:	bf 04 00 00 00       	mov    $0x4,%edi
  40071d:	e8 ee 02 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400722:	48 89 c3             	mov    %rax,%rbx
  400725:	b8 00 00 00 00       	mov    $0x0,%eax
  40072a:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  40072e:	48 83 f8 03          	cmp    $0x3,%rax
  400732:	74 06                	je     40073a <_Z12new_instancev+0x2b>
  400734:	48 83 c0 01          	add    $0x1,%rax
  400738:	eb f0                	jmp    40072a <_Z12new_instancev+0x1b>
  40073a:	48 89 df             	mov    %rbx,%rdi
  40073d:	e8 ec 01 00 00       	callq  40092e <_ZN9TestClassC1Ev>
  400742:	48 89 d8             	mov    %rbx,%rax
  400745:	48 83 c4 08          	add    $0x8,%rsp
  400749:	5b                   	pop    %rbx
  40074a:	5d                   	pop    %rbp
  40074b:	c3                   	retq   

000000000040074c <_Z12write_field0P9TestClass>:
  40074c:	55                   	push   %rbp
  40074d:	48 89 e5             	mov    %rsp,%rbp
  400750:	48 83 ec 30          	sub    $0x30,%rsp
  400754:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400758:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40075f:	00 00 
  400761:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400765:	31 c0                	xor    %eax,%eax
  400767:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40076b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40076f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400773:	ba 77 77 00 00       	mov    $0x7777,%edx
  400778:	48 89 ce             	mov    %rcx,%rsi
  40077b:	48 89 c7             	mov    %rax,%rdi
  40077e:	e8 dd 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400783:	90                   	nop
  400784:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400788:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40078f:	00 00 
  400791:	74 05                	je     400798 <_Z12write_field0P9TestClass+0x4c>
  400793:	e8 f8 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400798:	c9                   	leaveq 
  400799:	c3                   	retq   

000000000040079a <_Z11read_field0P9TestClass>:
  40079a:	55                   	push   %rbp
  40079b:	48 89 e5             	mov    %rsp,%rbp
  40079e:	48 83 ec 10          	sub    $0x10,%rsp
  4007a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007aa:	48 83 c0 04          	add    $0x4,%rax
  4007ae:	48 89 c7             	mov    %rax,%rdi
  4007b1:	e8 2a 03 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007b6:	8b 00                	mov    (%rax),%eax
  4007b8:	c9                   	leaveq 
  4007b9:	c3                   	retq   

00000000004007ba <_Z12write_field1P9TestClass>:
  4007ba:	55                   	push   %rbp
  4007bb:	48 89 e5             	mov    %rsp,%rbp
  4007be:	48 83 ec 30          	sub    $0x30,%rsp
  4007c2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007c6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007cd:	00 00 
  4007cf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007d3:	31 c0                	xor    %eax,%eax
  4007d5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007d9:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4007dd:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007e1:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007e6:	48 89 ce             	mov    %rcx,%rsi
  4007e9:	48 89 c7             	mov    %rax,%rdi
  4007ec:	e8 af 02 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007f1:	90                   	nop
  4007f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007fd:	00 00 
  4007ff:	74 05                	je     400806 <_Z12write_field1P9TestClass+0x4c>
  400801:	e8 8a fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400806:	c9                   	leaveq 
  400807:	c3                   	retq   

0000000000400808 <_Z11read_field1P9TestClass>:
  400808:	55                   	push   %rbp
  400809:	48 89 e5             	mov    %rsp,%rbp
  40080c:	48 83 ec 10          	sub    $0x10,%rsp
  400810:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400814:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400818:	48 83 c0 04          	add    $0x4,%rax
  40081c:	48 89 c7             	mov    %rax,%rdi
  40081f:	e8 d6 02 00 00       	callq  400afa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400824:	8b 00                	mov    (%rax),%eax
  400826:	c9                   	leaveq 
  400827:	c3                   	retq   

0000000000400828 <_Z21explicit_write_field0m>:
  400828:	55                   	push   %rbp
  400829:	48 89 e5             	mov    %rsp,%rbp
  40082c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400830:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400834:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  40083b:	77 77 00 00 
  40083f:	90                   	nop
  400840:	5d                   	pop    %rbp
  400841:	c3                   	retq   

0000000000400842 <_Z20explicit_read_field0m>:
  400842:	55                   	push   %rbp
  400843:	48 89 e5             	mov    %rsp,%rbp
  400846:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40084a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40084e:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  400855:	5d                   	pop    %rbp
  400856:	c3                   	retq   

0000000000400857 <_ZnwmPv>:
  400857:	55                   	push   %rbp
  400858:	48 89 e5             	mov    %rsp,%rbp
  40085b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40085f:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400863:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400867:	5d                   	pop    %rbp
  400868:	c3                   	retq   
  400869:	90                   	nop

000000000040086a <_ZN9TestClass15increase_field0Ev>:
  40086a:	55                   	push   %rbp
  40086b:	48 89 e5             	mov    %rsp,%rbp
  40086e:	48 83 ec 30          	sub    $0x30,%rsp
  400872:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400876:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40087d:	00 00 
  40087f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400883:	31 c0                	xor    %eax,%eax
  400885:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400889:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40088d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400891:	ba 55 55 00 00       	mov    $0x5555,%edx
  400896:	48 89 ce             	mov    %rcx,%rsi
  400899:	48 89 c7             	mov    %rax,%rdi
  40089c:	e8 c9 00 00 00       	callq  40096a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  4008a1:	90                   	nop
  4008a2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008ad:	00 00 
  4008af:	74 05                	je     4008b6 <_ZN9TestClass15increase_field0Ev+0x4c>
  4008b1:	e8 da fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008b6:	c9                   	leaveq 
  4008b7:	c3                   	retq   

00000000004008b8 <_ZN9TestClass15increase_field1Ev>:
  4008b8:	55                   	push   %rbp
  4008b9:	48 89 e5             	mov    %rsp,%rbp
  4008bc:	48 83 ec 30          	sub    $0x30,%rsp
  4008c0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008c4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008cb:	00 00 
  4008cd:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008d1:	31 c0                	xor    %eax,%eax
  4008d3:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008d7:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008db:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008df:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008e4:	48 89 ce             	mov    %rcx,%rsi
  4008e7:	48 89 c7             	mov    %rax,%rdi
  4008ea:	e8 bb 00 00 00       	callq  4009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008ef:	90                   	nop
  4008f0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008fb:	00 00 
  4008fd:	74 05                	je     400904 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008ff:	e8 8c fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  400904:	c9                   	leaveq 
  400905:	c3                   	retq   

0000000000400906 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400906:	55                   	push   %rbp
  400907:	48 89 e5             	mov    %rsp,%rbp
  40090a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40090e:	90                   	nop
  40090f:	5d                   	pop    %rbp
  400910:	c3                   	retq   
  400911:	90                   	nop

0000000000400912 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  400912:	55                   	push   %rbp
  400913:	48 89 e5             	mov    %rsp,%rbp
  400916:	48 83 ec 10          	sub    $0x10,%rsp
  40091a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40091e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400922:	48 89 c7             	mov    %rax,%rdi
  400925:	e8 dc ff ff ff       	callq  400906 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40092a:	90                   	nop
  40092b:	c9                   	leaveq 
  40092c:	c3                   	retq   
  40092d:	90                   	nop

000000000040092e <_ZN9TestClassC1Ev>:
  40092e:	55                   	push   %rbp
  40092f:	48 89 e5             	mov    %rsp,%rbp
  400932:	48 83 ec 10          	sub    $0x10,%rsp
  400936:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40093e:	48 89 c7             	mov    %rax,%rdi
  400941:	e8 cc ff ff ff       	callq  400912 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400946:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40094a:	48 83 c0 04          	add    $0x4,%rax
  40094e:	48 89 c7             	mov    %rax,%rdi
  400951:	e8 e8 00 00 00       	callq  400a3e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400956:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40095a:	48 83 c0 04          	add    $0x4,%rax
  40095e:	48 89 c7             	mov    %rax,%rdi
  400961:	e8 e4 00 00 00       	callq  400a4a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400966:	90                   	nop
  400967:	c9                   	leaveq 
  400968:	c3                   	retq   
  400969:	90                   	nop

000000000040096a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40096a:	55                   	push   %rbp
  40096b:	48 89 e5             	mov    %rsp,%rbp
  40096e:	48 83 ec 20          	sub    $0x20,%rsp
  400972:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400976:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40097a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40097d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400981:	48 89 c7             	mov    %rax,%rdi
  400984:	e8 8b 01 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400989:	8b 10                	mov    (%rax),%edx
  40098b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40098f:	89 10                	mov    %edx,(%rax)
  400991:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400995:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400999:	48 89 d6             	mov    %rdx,%rsi
  40099c:	48 89 c7             	mov    %rax,%rdi
  40099f:	e8 8a 01 00 00       	callq  400b2e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  4009a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a8:	c9                   	leaveq 
  4009a9:	c3                   	retq   

00000000004009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	48 83 ec 20          	sub    $0x20,%rsp
  4009b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009ba:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009bd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009c1:	48 89 c7             	mov    %rax,%rdi
  4009c4:	e8 75 01 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009c9:	8b 10                	mov    (%rax),%edx
  4009cb:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009cf:	89 10                	mov    %edx,(%rax)
  4009d1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009d5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d9:	48 89 d6             	mov    %rdx,%rsi
  4009dc:	48 89 c7             	mov    %rax,%rdi
  4009df:	e8 74 01 00 00       	callq  400b58 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009e4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e8:	c9                   	leaveq 
  4009e9:	c3                   	retq   

00000000004009ea <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageEv>:
  4009ea:	55                   	push   %rbp
  4009eb:	48 89 e5             	mov    %rsp,%rbp
  4009ee:	e8 e3 fb ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009f3:	48 89 c6             	mov    %rax,%rsi
  4009f6:	bf 20 00 00 00       	mov    $0x20,%edi
  4009fb:	e8 57 fe ff ff       	callq  400857 <_ZnwmPv>
  400a00:	48 85 c0             	test   %rax,%rax
  400a03:	74 08                	je     400a0d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageEv+0x23>
  400a05:	48 89 c7             	mov    %rax,%rdi
  400a08:	e8 5b 01 00 00       	callq  400b68 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400a0d:	90                   	nop
  400a0e:	5d                   	pop    %rbp
  400a0f:	c3                   	retq   

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	48 83 ec 10          	sub    $0x10,%rsp
  400a18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1c:	e8 87 01 00 00       	callq  400ba8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a21:	e8 b0 fb ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400a26:	48 89 c2             	mov    %rax,%rdx
  400a29:	48 8b 02             	mov    (%rdx),%rax
  400a2c:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a30:	48 89 0a             	mov    %rcx,(%rdx)
  400a33:	48 89 c7             	mov    %rax,%rdi
  400a36:	e8 74 01 00 00       	callq  400baf <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a3b:	c9                   	leaveq 
  400a3c:	c3                   	retq   
  400a3d:	90                   	nop

0000000000400a3e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a3e:	55                   	push   %rbp
  400a3f:	48 89 e5             	mov    %rsp,%rbp
  400a42:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a46:	90                   	nop
  400a47:	5d                   	pop    %rbp
  400a48:	c3                   	retq   
  400a49:	90                   	nop

0000000000400a4a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a4a:	55                   	push   %rbp
  400a4b:	48 89 e5             	mov    %rsp,%rbp
  400a4e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a52:	90                   	nop
  400a53:	5d                   	pop    %rbp
  400a54:	c3                   	retq   

0000000000400a55 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a55:	55                   	push   %rbp
  400a56:	48 89 e5             	mov    %rsp,%rbp
  400a59:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a5d:	90                   	nop
  400a5e:	5d                   	pop    %rbp
  400a5f:	c3                   	retq   

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a60:	55                   	push   %rbp
  400a61:	48 89 e5             	mov    %rsp,%rbp
  400a64:	48 83 ec 20          	sub    $0x20,%rsp
  400a68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a70:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a73:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a77:	48 89 c7             	mov    %rax,%rdi
  400a7a:	e8 95 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a7f:	48 89 c2             	mov    %rax,%rdx
  400a82:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a85:	89 02                	mov    %eax,(%rdx)
  400a87:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a8b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a8f:	48 89 d6             	mov    %rdx,%rsi
  400a92:	48 89 c7             	mov    %rax,%rdi
  400a95:	e8 94 00 00 00       	callq  400b2e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a9a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a9e:	c9                   	leaveq 
  400a9f:	c3                   	retq   

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 83 ec 20          	sub    $0x20,%rsp
  400aa8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aac:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ab0:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400ab3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ab7:	48 89 c7             	mov    %rax,%rdi
  400aba:	e8 7f 00 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400abf:	48 89 c2             	mov    %rax,%rdx
  400ac2:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400ac5:	89 02                	mov    %eax,(%rdx)
  400ac7:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400acb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400acf:	48 89 d6             	mov    %rdx,%rsi
  400ad2:	48 89 c7             	mov    %rax,%rdi
  400ad5:	e8 7e 00 00 00       	callq  400b58 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400ada:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ade:	c9                   	leaveq 
  400adf:	c3                   	retq   

0000000000400ae0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	48 83 ec 10          	sub    $0x10,%rsp
  400ae8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af0:	48 89 c7             	mov    %rax,%rdi
  400af3:	e8 1c 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400af8:	c9                   	leaveq 
  400af9:	c3                   	retq   

0000000000400afa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400afa:	55                   	push   %rbp
  400afb:	48 89 e5             	mov    %rsp,%rbp
  400afe:	48 83 ec 10          	sub    $0x10,%rsp
  400b02:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b06:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b0a:	48 89 c7             	mov    %rax,%rdi
  400b0d:	e8 2c 00 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b12:	c9                   	leaveq 
  400b13:	c3                   	retq   

0000000000400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b14:	55                   	push   %rbp
  400b15:	48 89 e5             	mov    %rsp,%rbp
  400b18:	48 83 ec 10          	sub    $0x10,%rsp
  400b1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b20:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b24:	48 89 c7             	mov    %rax,%rdi
  400b27:	e8 a2 00 00 00       	callq  400bce <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2c:	c9                   	leaveq 
  400b2d:	c3                   	retq   

0000000000400b2e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b2e:	55                   	push   %rbp
  400b2f:	48 89 e5             	mov    %rsp,%rbp
  400b32:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b36:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b3a:	90                   	nop
  400b3b:	5d                   	pop    %rbp
  400b3c:	c3                   	retq   
  400b3d:	90                   	nop

0000000000400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b3e:	55                   	push   %rbp
  400b3f:	48 89 e5             	mov    %rsp,%rbp
  400b42:	48 83 ec 10          	sub    $0x10,%rsp
  400b46:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b4a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b4e:	48 89 c7             	mov    %rax,%rdi
  400b51:	e8 c2 00 00 00       	callq  400c18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b56:	c9                   	leaveq 
  400b57:	c3                   	retq   

0000000000400b58 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b58:	55                   	push   %rbp
  400b59:	48 89 e5             	mov    %rsp,%rbp
  400b5c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b60:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b64:	90                   	nop
  400b65:	5d                   	pop    %rbp
  400b66:	c3                   	retq   
  400b67:	90                   	nop

0000000000400b68 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400b68:	55                   	push   %rbp
  400b69:	48 89 e5             	mov    %rsp,%rbp
  400b6c:	48 83 ec 10          	sub    $0x10,%rsp
  400b70:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b74:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400b79:	e8 42 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b7e:	48 89 c2             	mov    %rax,%rdx
  400b81:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b85:	48 89 50 08          	mov    %rdx,0x8(%rax)
  400b89:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b8d:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  400b94:	00 
  400b95:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b99:	48 8b 50 18          	mov    0x18(%rax),%rdx
  400b9d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ba1:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400ba5:	90                   	nop
  400ba6:	c9                   	leaveq 
  400ba7:	c3                   	retq   

0000000000400ba8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ba8:	55                   	push   %rbp
  400ba9:	48 89 e5             	mov    %rsp,%rbp
  400bac:	90                   	nop
  400bad:	5d                   	pop    %rbp
  400bae:	c3                   	retq   

0000000000400baf <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400baf:	55                   	push   %rbp
  400bb0:	48 89 e5             	mov    %rsp,%rbp
  400bb3:	48 83 ec 10          	sub    $0x10,%rsp
  400bb7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bbb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bbf:	48 83 c0 01          	add    $0x1,%rax
  400bc3:	48 89 c7             	mov    %rax,%rdi
  400bc6:	e8 9d 00 00 00       	callq  400c68 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bcb:	c9                   	leaveq 
  400bcc:	c3                   	retq   
  400bcd:	90                   	nop

0000000000400bce <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bce:	55                   	push   %rbp
  400bcf:	48 89 e5             	mov    %rsp,%rbp
  400bd2:	48 83 ec 28          	sub    $0x28,%rsp
  400bd6:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400bda:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bde:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400be2:	e8 ef f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400be7:	48 8b 40 08          	mov    0x8(%rax),%rax
  400beb:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bef:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bf3:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bf7:	48 83 e8 04          	sub    $0x4,%rax
  400bfb:	48 c1 e8 02          	shr    $0x2,%rax
  400bff:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c06:	00 
  400c07:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c0b:	48 01 d0             	add    %rdx,%rax
  400c0e:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c12:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c16:	c9                   	leaveq 
  400c17:	c3                   	retq   

0000000000400c18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c18:	55                   	push   %rbp
  400c19:	48 89 e5             	mov    %rsp,%rbp
  400c1c:	48 83 ec 28          	sub    $0x28,%rsp
  400c20:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400c24:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400c28:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c2c:	e8 a5 f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400c31:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c35:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400c39:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c3d:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400c41:	48 83 e8 04          	sub    $0x4,%rax
  400c45:	48 c1 e8 02          	shr    $0x2,%rax
  400c49:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c50:	00 
  400c51:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c55:	48 01 d0             	add    %rdx,%rax
  400c58:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c5e:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c62:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c66:	c9                   	leaveq 
  400c67:	c3                   	retq   

0000000000400c68 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c68:	55                   	push   %rbp
  400c69:	48 89 e5             	mov    %rsp,%rbp
  400c6c:	48 83 ec 18          	sub    $0x18,%rsp
  400c70:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c74:	e8 5d f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400c79:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c7d:	48 89 c2             	mov    %rax,%rdx
  400c80:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c84:	48 c1 e0 02          	shl    $0x2,%rax
  400c88:	48 01 d0             	add    %rdx,%rax
  400c8b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c8f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c93:	c9                   	leaveq 
  400c94:	c3                   	retq   
  400c95:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9c:	00 00 00 
  400c9f:	90                   	nop

0000000000400ca0 <__libc_csu_init>:
  400ca0:	41 57                	push   %r15
  400ca2:	41 56                	push   %r14
  400ca4:	41 89 ff             	mov    %edi,%r15d
  400ca7:	41 55                	push   %r13
  400ca9:	41 54                	push   %r12
  400cab:	4c 8d 25 5e 11 20 00 	lea    0x20115e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400cb2:	55                   	push   %rbp
  400cb3:	48 8d 2d 5e 11 20 00 	lea    0x20115e(%rip),%rbp        # 601e18 <__init_array_end>
  400cba:	53                   	push   %rbx
  400cbb:	49 89 f6             	mov    %rsi,%r14
  400cbe:	49 89 d5             	mov    %rdx,%r13
  400cc1:	4c 29 e5             	sub    %r12,%rbp
  400cc4:	48 83 ec 08          	sub    $0x8,%rsp
  400cc8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ccc:	e8 8f f7 ff ff       	callq  400460 <_init>
  400cd1:	48 85 ed             	test   %rbp,%rbp
  400cd4:	74 20                	je     400cf6 <__libc_csu_init+0x56>
  400cd6:	31 db                	xor    %ebx,%ebx
  400cd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400cdf:	00 
  400ce0:	4c 89 ea             	mov    %r13,%rdx
  400ce3:	4c 89 f6             	mov    %r14,%rsi
  400ce6:	44 89 ff             	mov    %r15d,%edi
  400ce9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ced:	48 83 c3 01          	add    $0x1,%rbx
  400cf1:	48 39 eb             	cmp    %rbp,%rbx
  400cf4:	75 ea                	jne    400ce0 <__libc_csu_init+0x40>
  400cf6:	48 83 c4 08          	add    $0x8,%rsp
  400cfa:	5b                   	pop    %rbx
  400cfb:	5d                   	pop    %rbp
  400cfc:	41 5c                	pop    %r12
  400cfe:	41 5d                	pop    %r13
  400d00:	41 5e                	pop    %r14
  400d02:	41 5f                	pop    %r15
  400d04:	c3                   	retq   
  400d05:	90                   	nop
  400d06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d0d:	00 00 00 

0000000000400d10 <__libc_csu_fini>:
  400d10:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d14 <_fini>:
  400d14:	48 83 ec 08          	sub    $0x8,%rsp
  400d18:	48 83 c4 08          	add    $0x8,%rsp
  400d1c:	c3                   	retq   
