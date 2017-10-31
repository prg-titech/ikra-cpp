
bin/g++_-O0_DynamicStorage_0:     file format elf64-x86-64


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
  4004ef:	49 c7 c0 10 0c 40 00 	mov    $0x400c10,%r8
  4004f6:	48 c7 c1 a0 0b 40 00 	mov    $0x400ba0,%rcx
  4004fd:	48 c7 c7 d6 05 40 00 	mov    $0x4005d6,%rdi
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

00000000004005d6 <main>:
  4005d6:	55                   	push   %rbp
  4005d7:	48 89 e5             	mov    %rsp,%rbp
  4005da:	53                   	push   %rbx
  4005db:	48 83 ec 28          	sub    $0x28,%rsp
  4005df:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005e6:	00 00 
  4005e8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005ec:	31 c0                	xor    %eax,%eax
  4005ee:	e8 36 02 00 00       	callq  400829 <_ZN9TestClass18initialize_storageEv>
  4005f3:	bf 00 00 00 00       	mov    $0x0,%edi
  4005f8:	e8 ff 03 00 00       	callq  4009fc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4005fd:	48 89 c3             	mov    %rax,%rbx
  400600:	48 89 df             	mov    %rbx,%rdi
  400603:	e8 00 03 00 00       	callq  400908 <_ZN9TestClassC1Ev>
  400608:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  40060c:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400610:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400614:	ba 77 77 00 00       	mov    $0x7777,%edx
  400619:	48 89 ce             	mov    %rcx,%rsi
  40061c:	48 89 c7             	mov    %rax,%rdi
  40061f:	e8 28 04 00 00       	callq  400a4c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400624:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400628:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40062c:	ba 88 88 00 00       	mov    $0x8888,%edx
  400631:	48 89 ce             	mov    %rcx,%rsi
  400634:	48 89 c7             	mov    %rax,%rdi
  400637:	e8 50 04 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40063c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400640:	48 89 c7             	mov    %rax,%rdi
  400643:	e8 04 02 00 00       	callq  40084c <_ZN9TestClass15increase_field0Ev>
  400648:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064c:	48 89 c7             	mov    %rax,%rdi
  40064f:	e8 42 02 00 00       	callq  400896 <_ZN9TestClass15increase_field1Ev>
  400654:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400658:	48 89 c7             	mov    %rax,%rdi
  40065b:	e8 86 04 00 00       	callq  400ae6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400660:	8b 18                	mov    (%rax),%ebx
  400662:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400666:	48 89 c7             	mov    %rax,%rdi
  400669:	e8 5e 04 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40066e:	8b 00                	mov    (%rax),%eax
  400670:	89 da                	mov    %ebx,%edx
  400672:	89 c6                	mov    %eax,%esi
  400674:	bf 34 0c 40 00       	mov    $0x400c34,%edi
  400679:	b8 00 00 00 00       	mov    $0x0,%eax
  40067e:	e8 1d fe ff ff       	callq  4004a0 <printf@plt>
  400683:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400687:	48 89 c7             	mov    %rax,%rdi
  40068a:	e8 3d 04 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40068f:	8b 00                	mov    (%rax),%eax
  400691:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400696:	75 15                	jne    4006ad <main+0xd7>
  400698:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40069c:	48 89 c7             	mov    %rax,%rdi
  40069f:	e8 42 04 00 00       	callq  400ae6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4006a4:	8b 00                	mov    (%rax),%eax
  4006a6:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006ab:	74 07                	je     4006b4 <main+0xde>
  4006ad:	b8 01 00 00 00       	mov    $0x1,%eax
  4006b2:	eb 05                	jmp    4006b9 <main+0xe3>
  4006b4:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b9:	0f b6 c0             	movzbl %al,%eax
  4006bc:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006c0:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006c7:	00 00 
  4006c9:	74 05                	je     4006d0 <main+0xfa>
  4006cb:	e8 c0 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  4006d0:	48 83 c4 28          	add    $0x28,%rsp
  4006d4:	5b                   	pop    %rbx
  4006d5:	5d                   	pop    %rbp
  4006d6:	c3                   	retq   

00000000004006d7 <_Z12new_instancev>:
  4006d7:	55                   	push   %rbp
  4006d8:	48 89 e5             	mov    %rsp,%rbp
  4006db:	53                   	push   %rbx
  4006dc:	48 83 ec 08          	sub    $0x8,%rsp
  4006e0:	bf 00 00 00 00       	mov    $0x0,%edi
  4006e5:	e8 12 03 00 00       	callq  4009fc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006ea:	48 89 c3             	mov    %rax,%rbx
  4006ed:	48 89 df             	mov    %rbx,%rdi
  4006f0:	e8 13 02 00 00       	callq  400908 <_ZN9TestClassC1Ev>
  4006f5:	48 89 d8             	mov    %rbx,%rax
  4006f8:	48 83 c4 08          	add    $0x8,%rsp
  4006fc:	5b                   	pop    %rbx
  4006fd:	5d                   	pop    %rbp
  4006fe:	c3                   	retq   

00000000004006ff <_Z12write_field0P9TestClass>:
  4006ff:	55                   	push   %rbp
  400700:	48 89 e5             	mov    %rsp,%rbp
  400703:	48 83 ec 30          	sub    $0x30,%rsp
  400707:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40070b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400712:	00 00 
  400714:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400718:	31 c0                	xor    %eax,%eax
  40071a:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40071e:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400722:	ba 77 77 00 00       	mov    $0x7777,%edx
  400727:	48 89 ce             	mov    %rcx,%rsi
  40072a:	48 89 c7             	mov    %rax,%rdi
  40072d:	e8 1a 03 00 00       	callq  400a4c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400732:	90                   	nop
  400733:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400737:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40073e:	00 00 
  400740:	74 05                	je     400747 <_Z12write_field0P9TestClass+0x48>
  400742:	e8 49 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  400747:	c9                   	leaveq 
  400748:	c3                   	retq   

0000000000400749 <_Z11read_field0P9TestClass>:
  400749:	55                   	push   %rbp
  40074a:	48 89 e5             	mov    %rsp,%rbp
  40074d:	48 83 ec 10          	sub    $0x10,%rsp
  400751:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400755:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400759:	48 89 c7             	mov    %rax,%rdi
  40075c:	e8 6b 03 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400761:	8b 00                	mov    (%rax),%eax
  400763:	c9                   	leaveq 
  400764:	c3                   	retq   

0000000000400765 <_Z12write_field1P9TestClass>:
  400765:	55                   	push   %rbp
  400766:	48 89 e5             	mov    %rsp,%rbp
  400769:	48 83 ec 30          	sub    $0x30,%rsp
  40076d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400771:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400778:	00 00 
  40077a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40077e:	31 c0                	xor    %eax,%eax
  400780:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400784:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400788:	ba 77 77 00 00       	mov    $0x7777,%edx
  40078d:	48 89 ce             	mov    %rcx,%rsi
  400790:	48 89 c7             	mov    %rax,%rdi
  400793:	e8 f4 02 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400798:	90                   	nop
  400799:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40079d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007a4:	00 00 
  4007a6:	74 05                	je     4007ad <_Z12write_field1P9TestClass+0x48>
  4007a8:	e8 e3 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  4007ad:	c9                   	leaveq 
  4007ae:	c3                   	retq   

00000000004007af <_Z11read_field1P9TestClass>:
  4007af:	55                   	push   %rbp
  4007b0:	48 89 e5             	mov    %rsp,%rbp
  4007b3:	48 83 ec 10          	sub    $0x10,%rsp
  4007b7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007bb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007bf:	48 89 c7             	mov    %rax,%rdi
  4007c2:	e8 1f 03 00 00       	callq  400ae6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007c7:	8b 00                	mov    (%rax),%eax
  4007c9:	c9                   	leaveq 
  4007ca:	c3                   	retq   

00000000004007cb <_Z21explicit_write_field0m>:
  4007cb:	55                   	push   %rbp
  4007cc:	48 89 e5             	mov    %rsp,%rbp
  4007cf:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d3:	48 8b 05 46 9f 21 00 	mov    0x219f46(%rip),%rax        # 61a720 <explicit_field0>
  4007da:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007de:	48 c1 e2 02          	shl    $0x2,%rdx
  4007e2:	48 01 d0             	add    %rdx,%rax
  4007e5:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007eb:	90                   	nop
  4007ec:	5d                   	pop    %rbp
  4007ed:	c3                   	retq   

00000000004007ee <_Z20explicit_read_field0m>:
  4007ee:	55                   	push   %rbp
  4007ef:	48 89 e5             	mov    %rsp,%rbp
  4007f2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007f6:	48 8b 05 23 9f 21 00 	mov    0x219f23(%rip),%rax        # 61a720 <explicit_field0>
  4007fd:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400801:	48 c1 e2 02          	shl    $0x2,%rdx
  400805:	48 01 d0             	add    %rdx,%rax
  400808:	8b 00                	mov    (%rax),%eax
  40080a:	5d                   	pop    %rbp
  40080b:	c3                   	retq   

000000000040080c <_ZnwmPv>:
  40080c:	55                   	push   %rbp
  40080d:	48 89 e5             	mov    %rsp,%rbp
  400810:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400814:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400818:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40081c:	5d                   	pop    %rbp
  40081d:	c3                   	retq   

000000000040081e <_ZN9TestClass7storageEv>:
  40081e:	55                   	push   %rbp
  40081f:	48 89 e5             	mov    %rsp,%rbp
  400822:	b8 80 20 60 00       	mov    $0x602080,%eax
  400827:	5d                   	pop    %rbp
  400828:	c3                   	retq   

0000000000400829 <_ZN9TestClass18initialize_storageEv>:
  400829:	55                   	push   %rbp
  40082a:	48 89 e5             	mov    %rsp,%rbp
  40082d:	be 80 20 60 00       	mov    $0x602080,%esi
  400832:	bf 20 00 00 00       	mov    $0x20,%edi
  400837:	e8 d0 ff ff ff       	callq  40080c <_ZnwmPv>
  40083c:	48 85 c0             	test   %rax,%rax
  40083f:	74 08                	je     400849 <_ZN9TestClass18initialize_storageEv+0x20>
  400841:	48 89 c7             	mov    %rax,%rdi
  400844:	e8 f3 00 00 00       	callq  40093c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400849:	90                   	nop
  40084a:	5d                   	pop    %rbp
  40084b:	c3                   	retq   

000000000040084c <_ZN9TestClass15increase_field0Ev>:
  40084c:	55                   	push   %rbp
  40084d:	48 89 e5             	mov    %rsp,%rbp
  400850:	48 83 ec 30          	sub    $0x30,%rsp
  400854:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400858:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40085f:	00 00 
  400861:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400865:	31 c0                	xor    %eax,%eax
  400867:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40086b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40086f:	ba 55 55 00 00       	mov    $0x5555,%edx
  400874:	48 89 ce             	mov    %rcx,%rsi
  400877:	48 89 c7             	mov    %rax,%rdi
  40087a:	e8 fd 00 00 00       	callq  40097c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40087f:	90                   	nop
  400880:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400884:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40088b:	00 00 
  40088d:	74 05                	je     400894 <_ZN9TestClass15increase_field0Ev+0x48>
  40088f:	e8 fc fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  400894:	c9                   	leaveq 
  400895:	c3                   	retq   

0000000000400896 <_ZN9TestClass15increase_field1Ev>:
  400896:	55                   	push   %rbp
  400897:	48 89 e5             	mov    %rsp,%rbp
  40089a:	48 83 ec 30          	sub    $0x30,%rsp
  40089e:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008a2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008a9:	00 00 
  4008ab:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008af:	31 c0                	xor    %eax,%eax
  4008b1:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4008b5:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008b9:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008be:	48 89 ce             	mov    %rcx,%rsi
  4008c1:	48 89 c7             	mov    %rax,%rdi
  4008c4:	e8 f3 00 00 00       	callq  4009bc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  4008c9:	90                   	nop
  4008ca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ce:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008d5:	00 00 
  4008d7:	74 05                	je     4008de <_ZN9TestClass15increase_field1Ev+0x48>
  4008d9:	e8 b2 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008de:	c9                   	leaveq 
  4008df:	c3                   	retq   

00000000004008e0 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4008e0:	55                   	push   %rbp
  4008e1:	48 89 e5             	mov    %rsp,%rbp
  4008e4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e8:	90                   	nop
  4008e9:	5d                   	pop    %rbp
  4008ea:	c3                   	retq   
  4008eb:	90                   	nop

00000000004008ec <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  4008ec:	55                   	push   %rbp
  4008ed:	48 89 e5             	mov    %rsp,%rbp
  4008f0:	48 83 ec 10          	sub    $0x10,%rsp
  4008f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008fc:	48 89 c7             	mov    %rax,%rdi
  4008ff:	e8 dc ff ff ff       	callq  4008e0 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400904:	90                   	nop
  400905:	c9                   	leaveq 
  400906:	c3                   	retq   
  400907:	90                   	nop

0000000000400908 <_ZN9TestClassC1Ev>:
  400908:	55                   	push   %rbp
  400909:	48 89 e5             	mov    %rsp,%rbp
  40090c:	48 83 ec 10          	sub    $0x10,%rsp
  400910:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400914:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400918:	48 89 c7             	mov    %rax,%rdi
  40091b:	e8 cc ff ff ff       	callq  4008ec <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  400920:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400924:	48 89 c7             	mov    %rax,%rdi
  400927:	e8 fe 00 00 00       	callq  400a2a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  40092c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400930:	48 89 c7             	mov    %rax,%rdi
  400933:	e8 fe 00 00 00       	callq  400a36 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  400938:	90                   	nop
  400939:	c9                   	leaveq 
  40093a:	c3                   	retq   
  40093b:	90                   	nop

000000000040093c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  40093c:	55                   	push   %rbp
  40093d:	48 89 e5             	mov    %rsp,%rbp
  400940:	48 83 ec 10          	sub    $0x10,%rsp
  400944:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400948:	bf a0 91 00 00       	mov    $0x91a0,%edi
  40094d:	e8 6e fb ff ff       	callq  4004c0 <malloc@plt>
  400952:	48 89 c2             	mov    %rax,%rdx
  400955:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400959:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40095d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400961:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  400968:	00 
  400969:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096d:	48 8b 50 18          	mov    0x18(%rax),%rdx
  400971:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400975:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400979:	90                   	nop
  40097a:	c9                   	leaveq 
  40097b:	c3                   	retq   

000000000040097c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  40097c:	55                   	push   %rbp
  40097d:	48 89 e5             	mov    %rsp,%rbp
  400980:	48 83 ec 20          	sub    $0x20,%rsp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40098c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40098f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400993:	48 89 c7             	mov    %rax,%rdi
  400996:	e8 65 01 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40099b:	8b 10                	mov    (%rax),%edx
  40099d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009a1:	89 10                	mov    %edx,(%rax)
  4009a3:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009a7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009ab:	48 89 d6             	mov    %rdx,%rsi
  4009ae:	48 89 c7             	mov    %rax,%rdi
  4009b1:	e8 6c 01 00 00       	callq  400b22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4009b6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009ba:	c9                   	leaveq 
  4009bb:	c3                   	retq   

00000000004009bc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  4009bc:	55                   	push   %rbp
  4009bd:	48 89 e5             	mov    %rsp,%rbp
  4009c0:	48 83 ec 20          	sub    $0x20,%rsp
  4009c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009cc:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009cf:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009d3:	48 89 c7             	mov    %rax,%rdi
  4009d6:	e8 57 01 00 00       	callq  400b32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009db:	8b 10                	mov    (%rax),%edx
  4009dd:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009e1:	89 10                	mov    %edx,(%rax)
  4009e3:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009e7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009eb:	48 89 d6             	mov    %rdx,%rsi
  4009ee:	48 89 c7             	mov    %rax,%rdi
  4009f1:	e8 66 01 00 00       	callq  400b5c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  4009f6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009fa:	c9                   	leaveq 
  4009fb:	c3                   	retq   

00000000004009fc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009fc:	55                   	push   %rbp
  4009fd:	48 89 e5             	mov    %rsp,%rbp
  400a00:	48 83 ec 10          	sub    $0x10,%rsp
  400a04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a08:	e8 5e 01 00 00       	callq  400b6b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a0d:	e8 0c fe ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400a12:	48 89 c2             	mov    %rax,%rdx
  400a15:	48 8b 02             	mov    (%rdx),%rax
  400a18:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a1c:	48 89 0a             	mov    %rcx,(%rdx)
  400a1f:	48 89 c7             	mov    %rax,%rdi
  400a22:	e8 4b 01 00 00       	callq  400b72 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  400a27:	c9                   	leaveq 
  400a28:	c3                   	retq   
  400a29:	90                   	nop

0000000000400a2a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  400a2a:	55                   	push   %rbp
  400a2b:	48 89 e5             	mov    %rsp,%rbp
  400a2e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a32:	90                   	nop
  400a33:	5d                   	pop    %rbp
  400a34:	c3                   	retq   
  400a35:	90                   	nop

0000000000400a36 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  400a36:	55                   	push   %rbp
  400a37:	48 89 e5             	mov    %rsp,%rbp
  400a3a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3e:	90                   	nop
  400a3f:	5d                   	pop    %rbp
  400a40:	c3                   	retq   

0000000000400a41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a41:	55                   	push   %rbp
  400a42:	48 89 e5             	mov    %rsp,%rbp
  400a45:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a49:	90                   	nop
  400a4a:	5d                   	pop    %rbp
  400a4b:	c3                   	retq   

0000000000400a4c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a4c:	55                   	push   %rbp
  400a4d:	48 89 e5             	mov    %rsp,%rbp
  400a50:	48 83 ec 20          	sub    $0x20,%rsp
  400a54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a58:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a5c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a5f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a63:	48 89 c7             	mov    %rax,%rdi
  400a66:	e8 95 00 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a6b:	48 89 c2             	mov    %rax,%rdx
  400a6e:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a71:	89 02                	mov    %eax,(%rdx)
  400a73:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a77:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a7b:	48 89 d6             	mov    %rdx,%rsi
  400a7e:	48 89 c7             	mov    %rax,%rdi
  400a81:	e8 9c 00 00 00       	callq  400b22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a86:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a8a:	c9                   	leaveq 
  400a8b:	c3                   	retq   

0000000000400a8c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a8c:	55                   	push   %rbp
  400a8d:	48 89 e5             	mov    %rsp,%rbp
  400a90:	48 83 ec 20          	sub    $0x20,%rsp
  400a94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a98:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a9c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a9f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400aa3:	48 89 c7             	mov    %rax,%rdi
  400aa6:	e8 87 00 00 00       	callq  400b32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aab:	48 89 c2             	mov    %rax,%rdx
  400aae:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400ab1:	89 02                	mov    %eax,(%rdx)
  400ab3:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400ab7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400abb:	48 89 d6             	mov    %rdx,%rsi
  400abe:	48 89 c7             	mov    %rax,%rdi
  400ac1:	e8 96 00 00 00       	callq  400b5c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400ac6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aca:	c9                   	leaveq 
  400acb:	c3                   	retq   

0000000000400acc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400acc:	55                   	push   %rbp
  400acd:	48 89 e5             	mov    %rsp,%rbp
  400ad0:	48 83 ec 10          	sub    $0x10,%rsp
  400ad4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400adc:	48 89 c7             	mov    %rax,%rdi
  400adf:	e8 1c 00 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae4:	c9                   	leaveq 
  400ae5:	c3                   	retq   

0000000000400ae6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400ae6:	55                   	push   %rbp
  400ae7:	48 89 e5             	mov    %rsp,%rbp
  400aea:	48 83 ec 10          	sub    $0x10,%rsp
  400aee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af6:	48 89 c7             	mov    %rax,%rdi
  400af9:	e8 34 00 00 00       	callq  400b32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400afe:	c9                   	leaveq 
  400aff:	c3                   	retq   

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 10          	sub    $0x10,%rsp
  400b08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0c:	e8 0d fd ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400b11:	48 8b 40 08          	mov    0x8(%rax),%rax
  400b15:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400b19:	48 c1 e2 02          	shl    $0x2,%rdx
  400b1d:	48 01 d0             	add    %rdx,%rax
  400b20:	c9                   	leaveq 
  400b21:	c3                   	retq   

0000000000400b22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400b22:	55                   	push   %rbp
  400b23:	48 89 e5             	mov    %rsp,%rbp
  400b26:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b2a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b2e:	90                   	nop
  400b2f:	5d                   	pop    %rbp
  400b30:	c3                   	retq   
  400b31:	90                   	nop

0000000000400b32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b32:	55                   	push   %rbp
  400b33:	48 89 e5             	mov    %rsp,%rbp
  400b36:	48 83 ec 10          	sub    $0x10,%rsp
  400b3a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b3e:	e8 db fc ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400b43:	48 8b 40 08          	mov    0x8(%rax),%rax
  400b47:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400b4b:	48 c1 e2 02          	shl    $0x2,%rdx
  400b4f:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400b56:	48 01 d0             	add    %rdx,%rax
  400b59:	c9                   	leaveq 
  400b5a:	c3                   	retq   
  400b5b:	90                   	nop

0000000000400b5c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400b5c:	55                   	push   %rbp
  400b5d:	48 89 e5             	mov    %rsp,%rbp
  400b60:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b64:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b68:	90                   	nop
  400b69:	5d                   	pop    %rbp
  400b6a:	c3                   	retq   

0000000000400b6b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b6b:	55                   	push   %rbp
  400b6c:	48 89 e5             	mov    %rsp,%rbp
  400b6f:	90                   	nop
  400b70:	5d                   	pop    %rbp
  400b71:	c3                   	retq   

0000000000400b72 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b72:	55                   	push   %rbp
  400b73:	48 89 e5             	mov    %rsp,%rbp
  400b76:	48 83 ec 10          	sub    $0x10,%rsp
  400b7a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b7e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b82:	48 89 c7             	mov    %rax,%rdi
  400b85:	e8 02 00 00 00       	callq  400b8c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b8a:	c9                   	leaveq 
  400b8b:	c3                   	retq   

0000000000400b8c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b8c:	55                   	push   %rbp
  400b8d:	48 89 e5             	mov    %rsp,%rbp
  400b90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b94:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b98:	5d                   	pop    %rbp
  400b99:	c3                   	retq   
  400b9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400ba0 <__libc_csu_init>:
  400ba0:	41 57                	push   %r15
  400ba2:	41 56                	push   %r14
  400ba4:	41 89 ff             	mov    %edi,%r15d
  400ba7:	41 55                	push   %r13
  400ba9:	41 54                	push   %r12
  400bab:	4c 8d 25 5e 12 20 00 	lea    0x20125e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400bb2:	55                   	push   %rbp
  400bb3:	48 8d 2d 5e 12 20 00 	lea    0x20125e(%rip),%rbp        # 601e18 <__init_array_end>
  400bba:	53                   	push   %rbx
  400bbb:	49 89 f6             	mov    %rsi,%r14
  400bbe:	49 89 d5             	mov    %rdx,%r13
  400bc1:	4c 29 e5             	sub    %r12,%rbp
  400bc4:	48 83 ec 08          	sub    $0x8,%rsp
  400bc8:	48 c1 fd 03          	sar    $0x3,%rbp
  400bcc:	e8 8f f8 ff ff       	callq  400460 <_init>
  400bd1:	48 85 ed             	test   %rbp,%rbp
  400bd4:	74 20                	je     400bf6 <__libc_csu_init+0x56>
  400bd6:	31 db                	xor    %ebx,%ebx
  400bd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400bdf:	00 
  400be0:	4c 89 ea             	mov    %r13,%rdx
  400be3:	4c 89 f6             	mov    %r14,%rsi
  400be6:	44 89 ff             	mov    %r15d,%edi
  400be9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400bed:	48 83 c3 01          	add    $0x1,%rbx
  400bf1:	48 39 eb             	cmp    %rbp,%rbx
  400bf4:	75 ea                	jne    400be0 <__libc_csu_init+0x40>
  400bf6:	48 83 c4 08          	add    $0x8,%rsp
  400bfa:	5b                   	pop    %rbx
  400bfb:	5d                   	pop    %rbp
  400bfc:	41 5c                	pop    %r12
  400bfe:	41 5d                	pop    %r13
  400c00:	41 5e                	pop    %r14
  400c02:	41 5f                	pop    %r15
  400c04:	c3                   	retq   
  400c05:	90                   	nop
  400c06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c0d:	00 00 00 

0000000000400c10 <__libc_csu_fini>:
  400c10:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c14 <_fini>:
  400c14:	48 83 ec 08          	sub    $0x8,%rsp
  400c18:	48 83 c4 08          	add    $0x8,%rsp
  400c1c:	c3                   	retq   
