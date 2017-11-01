
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
  4004ef:	49 c7 c0 30 0c 40 00 	mov    $0x400c30,%r8
  4004f6:	48 c7 c1 c0 0b 40 00 	mov    $0x400bc0,%rcx
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
  4005ee:	e8 24 02 00 00       	callq  400817 <_ZN9TestClass18initialize_storageEv>
  4005f3:	bf 00 00 00 00       	mov    $0x0,%edi
  4005f8:	e8 ed 03 00 00       	callq  4009ea <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4005fd:	48 89 c3             	mov    %rax,%rbx
  400600:	48 89 df             	mov    %rbx,%rdi
  400603:	e8 ee 02 00 00       	callq  4008f6 <_ZN9TestClassC1Ev>
  400608:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  40060c:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400610:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400614:	ba 77 77 00 00       	mov    $0x7777,%edx
  400619:	48 89 ce             	mov    %rcx,%rsi
  40061c:	48 89 c7             	mov    %rax,%rdi
  40061f:	e8 16 04 00 00       	callq  400a3a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400624:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400628:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40062c:	ba 88 88 00 00       	mov    $0x8888,%edx
  400631:	48 89 ce             	mov    %rcx,%rsi
  400634:	48 89 c7             	mov    %rax,%rdi
  400637:	e8 3e 04 00 00       	callq  400a7a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40063c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400640:	48 89 c7             	mov    %rax,%rdi
  400643:	e8 f2 01 00 00       	callq  40083a <_ZN9TestClass15increase_field0Ev>
  400648:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064c:	48 89 c7             	mov    %rax,%rdi
  40064f:	e8 30 02 00 00       	callq  400884 <_ZN9TestClass15increase_field1Ev>
  400654:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400658:	48 89 c7             	mov    %rax,%rdi
  40065b:	e8 74 04 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400660:	8b 18                	mov    (%rax),%ebx
  400662:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400666:	48 89 c7             	mov    %rax,%rdi
  400669:	e8 4c 04 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40066e:	8b 00                	mov    (%rax),%eax
  400670:	89 da                	mov    %ebx,%edx
  400672:	89 c6                	mov    %eax,%esi
  400674:	bf 54 0c 40 00       	mov    $0x400c54,%edi
  400679:	b8 00 00 00 00       	mov    $0x0,%eax
  40067e:	e8 1d fe ff ff       	callq  4004a0 <printf@plt>
  400683:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400687:	48 89 c7             	mov    %rax,%rdi
  40068a:	e8 2b 04 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40068f:	8b 00                	mov    (%rax),%eax
  400691:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400696:	75 15                	jne    4006ad <main+0xd7>
  400698:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40069c:	48 89 c7             	mov    %rax,%rdi
  40069f:	e8 30 04 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
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
  4006e5:	e8 00 03 00 00       	callq  4009ea <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006ea:	48 89 c3             	mov    %rax,%rbx
  4006ed:	48 89 df             	mov    %rbx,%rdi
  4006f0:	e8 01 02 00 00       	callq  4008f6 <_ZN9TestClassC1Ev>
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
  40072d:	e8 08 03 00 00       	callq  400a3a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
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
  40075c:	e8 59 03 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
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
  400793:	e8 e2 02 00 00       	callq  400a7a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
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
  4007c2:	e8 0d 03 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007c7:	8b 00                	mov    (%rax),%eax
  4007c9:	c9                   	leaveq 
  4007ca:	c3                   	retq   

00000000004007cb <_Z21explicit_write_field0m>:
  4007cb:	55                   	push   %rbp
  4007cc:	48 89 e5             	mov    %rsp,%rbp
  4007cf:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007d7:	c7 04 85 20 a7 61 00 	movl   $0x7777,0x61a720(,%rax,4)
  4007de:	77 77 00 00 
  4007e2:	90                   	nop
  4007e3:	5d                   	pop    %rbp
  4007e4:	c3                   	retq   

00000000004007e5 <_Z20explicit_read_field0m>:
  4007e5:	55                   	push   %rbp
  4007e6:	48 89 e5             	mov    %rsp,%rbp
  4007e9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ed:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f1:	8b 04 85 20 a7 61 00 	mov    0x61a720(,%rax,4),%eax
  4007f8:	5d                   	pop    %rbp
  4007f9:	c3                   	retq   

00000000004007fa <_ZnwmPv>:
  4007fa:	55                   	push   %rbp
  4007fb:	48 89 e5             	mov    %rsp,%rbp
  4007fe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400802:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400806:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40080a:	5d                   	pop    %rbp
  40080b:	c3                   	retq   

000000000040080c <_ZN9TestClass7storageEv>:
  40080c:	55                   	push   %rbp
  40080d:	48 89 e5             	mov    %rsp,%rbp
  400810:	b8 80 20 60 00       	mov    $0x602080,%eax
  400815:	5d                   	pop    %rbp
  400816:	c3                   	retq   

0000000000400817 <_ZN9TestClass18initialize_storageEv>:
  400817:	55                   	push   %rbp
  400818:	48 89 e5             	mov    %rsp,%rbp
  40081b:	be 80 20 60 00       	mov    $0x602080,%esi
  400820:	bf 20 00 00 00       	mov    $0x20,%edi
  400825:	e8 d0 ff ff ff       	callq  4007fa <_ZnwmPv>
  40082a:	48 85 c0             	test   %rax,%rax
  40082d:	74 08                	je     400837 <_ZN9TestClass18initialize_storageEv+0x20>
  40082f:	48 89 c7             	mov    %rax,%rdi
  400832:	e8 f3 00 00 00       	callq  40092a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400837:	90                   	nop
  400838:	5d                   	pop    %rbp
  400839:	c3                   	retq   

000000000040083a <_ZN9TestClass15increase_field0Ev>:
  40083a:	55                   	push   %rbp
  40083b:	48 89 e5             	mov    %rsp,%rbp
  40083e:	48 83 ec 30          	sub    $0x30,%rsp
  400842:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400846:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40084d:	00 00 
  40084f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400853:	31 c0                	xor    %eax,%eax
  400855:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400859:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40085d:	ba 55 55 00 00       	mov    $0x5555,%edx
  400862:	48 89 ce             	mov    %rcx,%rsi
  400865:	48 89 c7             	mov    %rax,%rdi
  400868:	e8 fd 00 00 00       	callq  40096a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40086d:	90                   	nop
  40086e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400872:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400879:	00 00 
  40087b:	74 05                	je     400882 <_ZN9TestClass15increase_field0Ev+0x48>
  40087d:	e8 0e fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400882:	c9                   	leaveq 
  400883:	c3                   	retq   

0000000000400884 <_ZN9TestClass15increase_field1Ev>:
  400884:	55                   	push   %rbp
  400885:	48 89 e5             	mov    %rsp,%rbp
  400888:	48 83 ec 30          	sub    $0x30,%rsp
  40088c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400890:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400897:	00 00 
  400899:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40089d:	31 c0                	xor    %eax,%eax
  40089f:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4008a3:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008a7:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008ac:	48 89 ce             	mov    %rcx,%rsi
  4008af:	48 89 c7             	mov    %rax,%rdi
  4008b2:	e8 f3 00 00 00       	callq  4009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  4008b7:	90                   	nop
  4008b8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008bc:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008c3:	00 00 
  4008c5:	74 05                	je     4008cc <_ZN9TestClass15increase_field1Ev+0x48>
  4008c7:	e8 c4 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008cc:	c9                   	leaveq 
  4008cd:	c3                   	retq   

00000000004008ce <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4008ce:	55                   	push   %rbp
  4008cf:	48 89 e5             	mov    %rsp,%rbp
  4008d2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d6:	90                   	nop
  4008d7:	5d                   	pop    %rbp
  4008d8:	c3                   	retq   
  4008d9:	90                   	nop

00000000004008da <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  4008da:	55                   	push   %rbp
  4008db:	48 89 e5             	mov    %rsp,%rbp
  4008de:	48 83 ec 10          	sub    $0x10,%rsp
  4008e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ea:	48 89 c7             	mov    %rax,%rdi
  4008ed:	e8 dc ff ff ff       	callq  4008ce <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008f2:	90                   	nop
  4008f3:	c9                   	leaveq 
  4008f4:	c3                   	retq   
  4008f5:	90                   	nop

00000000004008f6 <_ZN9TestClassC1Ev>:
  4008f6:	55                   	push   %rbp
  4008f7:	48 89 e5             	mov    %rsp,%rbp
  4008fa:	48 83 ec 10          	sub    $0x10,%rsp
  4008fe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400902:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400906:	48 89 c7             	mov    %rax,%rdi
  400909:	e8 cc ff ff ff       	callq  4008da <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  40090e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400912:	48 89 c7             	mov    %rax,%rdi
  400915:	e8 fe 00 00 00       	callq  400a18 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  40091a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 fe 00 00 00       	callq  400a24 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  400926:	90                   	nop
  400927:	c9                   	leaveq 
  400928:	c3                   	retq   
  400929:	90                   	nop

000000000040092a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  40092a:	55                   	push   %rbp
  40092b:	48 89 e5             	mov    %rsp,%rbp
  40092e:	48 83 ec 10          	sub    $0x10,%rsp
  400932:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400936:	bf a0 91 00 00       	mov    $0x91a0,%edi
  40093b:	e8 80 fb ff ff       	callq  4004c0 <malloc@plt>
  400940:	48 89 c2             	mov    %rax,%rdx
  400943:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400947:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40094b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40094f:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  400956:	00 
  400957:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40095b:	48 8b 50 18          	mov    0x18(%rax),%rdx
  40095f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400963:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400967:	90                   	nop
  400968:	c9                   	leaveq 
  400969:	c3                   	retq   

000000000040096a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  40096a:	55                   	push   %rbp
  40096b:	48 89 e5             	mov    %rsp,%rbp
  40096e:	48 83 ec 20          	sub    $0x20,%rsp
  400972:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400976:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40097a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40097d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400981:	48 89 c7             	mov    %rax,%rdi
  400984:	e8 65 01 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400989:	8b 10                	mov    (%rax),%edx
  40098b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40098f:	89 10                	mov    %edx,(%rax)
  400991:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400995:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400999:	48 89 d6             	mov    %rdx,%rsi
  40099c:	48 89 c7             	mov    %rax,%rdi
  40099f:	e8 64 01 00 00       	callq  400b08 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4009a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a8:	c9                   	leaveq 
  4009a9:	c3                   	retq   

00000000004009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	48 83 ec 20          	sub    $0x20,%rsp
  4009b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009ba:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009bd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009c1:	48 89 c7             	mov    %rax,%rdi
  4009c4:	e8 4f 01 00 00       	callq  400b18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009c9:	8b 10                	mov    (%rax),%edx
  4009cb:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009cf:	89 10                	mov    %edx,(%rax)
  4009d1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009d5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d9:	48 89 d6             	mov    %rdx,%rsi
  4009dc:	48 89 c7             	mov    %rax,%rdi
  4009df:	e8 4e 01 00 00       	callq  400b32 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  4009e4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e8:	c9                   	leaveq 
  4009e9:	c3                   	retq   

00000000004009ea <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009ea:	55                   	push   %rbp
  4009eb:	48 89 e5             	mov    %rsp,%rbp
  4009ee:	48 83 ec 10          	sub    $0x10,%rsp
  4009f2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f6:	e8 46 01 00 00       	callq  400b41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009fb:	e8 0c fe ff ff       	callq  40080c <_ZN9TestClass7storageEv>
  400a00:	48 89 c2             	mov    %rax,%rdx
  400a03:	48 8b 02             	mov    (%rdx),%rax
  400a06:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a0a:	48 89 0a             	mov    %rcx,(%rdx)
  400a0d:	48 89 c7             	mov    %rax,%rdi
  400a10:	e8 33 01 00 00       	callq  400b48 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  400a15:	c9                   	leaveq 
  400a16:	c3                   	retq   
  400a17:	90                   	nop

0000000000400a18 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  400a18:	55                   	push   %rbp
  400a19:	48 89 e5             	mov    %rsp,%rbp
  400a1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a20:	90                   	nop
  400a21:	5d                   	pop    %rbp
  400a22:	c3                   	retq   
  400a23:	90                   	nop

0000000000400a24 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  400a24:	55                   	push   %rbp
  400a25:	48 89 e5             	mov    %rsp,%rbp
  400a28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2c:	90                   	nop
  400a2d:	5d                   	pop    %rbp
  400a2e:	c3                   	retq   

0000000000400a2f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a2f:	55                   	push   %rbp
  400a30:	48 89 e5             	mov    %rsp,%rbp
  400a33:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a37:	90                   	nop
  400a38:	5d                   	pop    %rbp
  400a39:	c3                   	retq   

0000000000400a3a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a3a:	55                   	push   %rbp
  400a3b:	48 89 e5             	mov    %rsp,%rbp
  400a3e:	48 83 ec 20          	sub    $0x20,%rsp
  400a42:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a46:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a4a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a4d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a51:	48 89 c7             	mov    %rax,%rdi
  400a54:	e8 95 00 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a59:	48 89 c2             	mov    %rax,%rdx
  400a5c:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a5f:	89 02                	mov    %eax,(%rdx)
  400a61:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a65:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a69:	48 89 d6             	mov    %rdx,%rsi
  400a6c:	48 89 c7             	mov    %rax,%rdi
  400a6f:	e8 94 00 00 00       	callq  400b08 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a74:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a78:	c9                   	leaveq 
  400a79:	c3                   	retq   

0000000000400a7a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a7a:	55                   	push   %rbp
  400a7b:	48 89 e5             	mov    %rsp,%rbp
  400a7e:	48 83 ec 20          	sub    $0x20,%rsp
  400a82:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a86:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a8a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a8d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a91:	48 89 c7             	mov    %rax,%rdi
  400a94:	e8 7f 00 00 00       	callq  400b18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a99:	48 89 c2             	mov    %rax,%rdx
  400a9c:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a9f:	89 02                	mov    %eax,(%rdx)
  400aa1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400aa5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa9:	48 89 d6             	mov    %rdx,%rsi
  400aac:	48 89 c7             	mov    %rax,%rdi
  400aaf:	e8 7e 00 00 00       	callq  400b32 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400ab4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab8:	c9                   	leaveq 
  400ab9:	c3                   	retq   

0000000000400aba <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400aba:	55                   	push   %rbp
  400abb:	48 89 e5             	mov    %rsp,%rbp
  400abe:	48 83 ec 10          	sub    $0x10,%rsp
  400ac2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aca:	48 89 c7             	mov    %rax,%rdi
  400acd:	e8 1c 00 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad2:	c9                   	leaveq 
  400ad3:	c3                   	retq   

0000000000400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400ad4:	55                   	push   %rbp
  400ad5:	48 89 e5             	mov    %rsp,%rbp
  400ad8:	48 83 ec 10          	sub    $0x10,%rsp
  400adc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae4:	48 89 c7             	mov    %rax,%rdi
  400ae7:	e8 2c 00 00 00       	callq  400b18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aec:	c9                   	leaveq 
  400aed:	c3                   	retq   

0000000000400aee <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aee:	55                   	push   %rbp
  400aef:	48 89 e5             	mov    %rsp,%rbp
  400af2:	48 83 ec 10          	sub    $0x10,%rsp
  400af6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400afe:	48 89 c7             	mov    %rax,%rdi
  400b01:	e8 60 00 00 00       	callq  400b66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b06:	c9                   	leaveq 
  400b07:	c3                   	retq   

0000000000400b08 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400b08:	55                   	push   %rbp
  400b09:	48 89 e5             	mov    %rsp,%rbp
  400b0c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b10:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b14:	90                   	nop
  400b15:	5d                   	pop    %rbp
  400b16:	c3                   	retq   
  400b17:	90                   	nop

0000000000400b18 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b18:	55                   	push   %rbp
  400b19:	48 89 e5             	mov    %rsp,%rbp
  400b1c:	48 83 ec 10          	sub    $0x10,%rsp
  400b20:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b24:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b28:	48 89 c7             	mov    %rax,%rdi
  400b2b:	e8 58 00 00 00       	callq  400b88 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b30:	c9                   	leaveq 
  400b31:	c3                   	retq   

0000000000400b32 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400b32:	55                   	push   %rbp
  400b33:	48 89 e5             	mov    %rsp,%rbp
  400b36:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b3a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b3e:	90                   	nop
  400b3f:	5d                   	pop    %rbp
  400b40:	c3                   	retq   

0000000000400b41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b41:	55                   	push   %rbp
  400b42:	48 89 e5             	mov    %rsp,%rbp
  400b45:	90                   	nop
  400b46:	5d                   	pop    %rbp
  400b47:	c3                   	retq   

0000000000400b48 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b48:	55                   	push   %rbp
  400b49:	48 89 e5             	mov    %rsp,%rbp
  400b4c:	48 83 ec 10          	sub    $0x10,%rsp
  400b50:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b54:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b58:	48 83 c0 01          	add    $0x1,%rax
  400b5c:	48 89 c7             	mov    %rax,%rdi
  400b5f:	e8 4d 00 00 00       	callq  400bb1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b64:	c9                   	leaveq 
  400b65:	c3                   	retq   

0000000000400b66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b66:	55                   	push   %rbp
  400b67:	48 89 e5             	mov    %rsp,%rbp
  400b6a:	48 83 ec 10          	sub    $0x10,%rsp
  400b6e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b72:	e8 95 fc ff ff       	callq  40080c <_ZN9TestClass7storageEv>
  400b77:	48 8b 40 08          	mov    0x8(%rax),%rax
  400b7b:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400b7f:	48 c1 e2 02          	shl    $0x2,%rdx
  400b83:	48 01 d0             	add    %rdx,%rax
  400b86:	c9                   	leaveq 
  400b87:	c3                   	retq   

0000000000400b88 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b88:	55                   	push   %rbp
  400b89:	48 89 e5             	mov    %rsp,%rbp
  400b8c:	48 83 ec 10          	sub    $0x10,%rsp
  400b90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b94:	e8 73 fc ff ff       	callq  40080c <_ZN9TestClass7storageEv>
  400b99:	48 8b 40 08          	mov    0x8(%rax),%rax
  400b9d:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400ba1:	48 c1 e2 02          	shl    $0x2,%rdx
  400ba5:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400bac:	48 01 d0             	add    %rdx,%rax
  400baf:	c9                   	leaveq 
  400bb0:	c3                   	retq   

0000000000400bb1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bb1:	55                   	push   %rbp
  400bb2:	48 89 e5             	mov    %rsp,%rbp
  400bb5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bbd:	5d                   	pop    %rbp
  400bbe:	c3                   	retq   
  400bbf:	90                   	nop

0000000000400bc0 <__libc_csu_init>:
  400bc0:	41 57                	push   %r15
  400bc2:	41 56                	push   %r14
  400bc4:	41 89 ff             	mov    %edi,%r15d
  400bc7:	41 55                	push   %r13
  400bc9:	41 54                	push   %r12
  400bcb:	4c 8d 25 3e 12 20 00 	lea    0x20123e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400bd2:	55                   	push   %rbp
  400bd3:	48 8d 2d 3e 12 20 00 	lea    0x20123e(%rip),%rbp        # 601e18 <__init_array_end>
  400bda:	53                   	push   %rbx
  400bdb:	49 89 f6             	mov    %rsi,%r14
  400bde:	49 89 d5             	mov    %rdx,%r13
  400be1:	4c 29 e5             	sub    %r12,%rbp
  400be4:	48 83 ec 08          	sub    $0x8,%rsp
  400be8:	48 c1 fd 03          	sar    $0x3,%rbp
  400bec:	e8 6f f8 ff ff       	callq  400460 <_init>
  400bf1:	48 85 ed             	test   %rbp,%rbp
  400bf4:	74 20                	je     400c16 <__libc_csu_init+0x56>
  400bf6:	31 db                	xor    %ebx,%ebx
  400bf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400bff:	00 
  400c00:	4c 89 ea             	mov    %r13,%rdx
  400c03:	4c 89 f6             	mov    %r14,%rsi
  400c06:	44 89 ff             	mov    %r15d,%edi
  400c09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c0d:	48 83 c3 01          	add    $0x1,%rbx
  400c11:	48 39 eb             	cmp    %rbp,%rbx
  400c14:	75 ea                	jne    400c00 <__libc_csu_init+0x40>
  400c16:	48 83 c4 08          	add    $0x8,%rsp
  400c1a:	5b                   	pop    %rbx
  400c1b:	5d                   	pop    %rbp
  400c1c:	41 5c                	pop    %r12
  400c1e:	41 5d                	pop    %r13
  400c20:	41 5e                	pop    %r14
  400c22:	41 5f                	pop    %r15
  400c24:	c3                   	retq   
  400c25:	90                   	nop
  400c26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c2d:	00 00 00 

0000000000400c30 <__libc_csu_fini>:
  400c30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c34 <_fini>:
  400c34:	48 83 ec 08          	sub    $0x8,%rsp
  400c38:	48 83 c4 08          	add    $0x8,%rsp
  400c3c:	c3                   	retq   
