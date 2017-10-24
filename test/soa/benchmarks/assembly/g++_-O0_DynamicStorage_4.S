
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
  4004ef:	49 c7 c0 b0 0c 40 00 	mov    $0x400cb0,%r8
  4004f6:	48 c7 c1 40 0c 40 00 	mov    $0x400c40,%rcx
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
  400590:	80 3d b9 1a 20 00 00 	cmpb   $0x0,0x201ab9(%rip)        # 602050 <completed.7585>
  400597:	75 11                	jne    4005aa <__do_global_dtors_aux+0x1a>
  400599:	55                   	push   %rbp
  40059a:	48 89 e5             	mov    %rsp,%rbp
  40059d:	e8 6e ff ff ff       	callq  400510 <deregister_tm_clones>
  4005a2:	5d                   	pop    %rbp
  4005a3:	c6 05 a6 1a 20 00 01 	movb   $0x1,0x201aa6(%rip)        # 602050 <completed.7585>
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
  4005ee:	bf 04 00 00 00       	mov    $0x4,%edi
  4005f3:	e8 44 04 00 00       	callq  400a3c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005f8:	48 89 c3             	mov    %rax,%rbx
  4005fb:	b8 00 00 00 00       	mov    $0x0,%eax
  400600:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400604:	48 83 f8 03          	cmp    $0x3,%rax
  400608:	74 06                	je     400610 <main+0x3a>
  40060a:	48 83 c0 01          	add    $0x1,%rax
  40060e:	eb f0                	jmp    400600 <main+0x2a>
  400610:	48 89 df             	mov    %rbx,%rdi
  400613:	e8 44 03 00 00       	callq  40095c <_ZN9TestClassC1Ev>
  400618:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  40061c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400620:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400624:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400628:	ba 77 77 00 00       	mov    $0x7777,%edx
  40062d:	48 89 ce             	mov    %rcx,%rsi
  400630:	48 89 c7             	mov    %rax,%rdi
  400633:	e8 52 04 00 00       	callq  400a8a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400638:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40063c:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400640:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400644:	ba 88 88 00 00       	mov    $0x8888,%edx
  400649:	48 89 ce             	mov    %rcx,%rsi
  40064c:	48 89 c7             	mov    %rax,%rdi
  40064f:	e8 76 04 00 00       	callq  400aca <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400654:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400658:	48 89 c7             	mov    %rax,%rdi
  40065b:	e8 38 02 00 00       	callq  400898 <_ZN9TestClass15increase_field0Ev>
  400660:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400664:	48 89 c7             	mov    %rax,%rdi
  400667:	e8 7a 02 00 00       	callq  4008e6 <_ZN9TestClass15increase_field1Ev>
  40066c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400670:	48 83 c0 04          	add    $0x4,%rax
  400674:	48 89 c7             	mov    %rax,%rdi
  400677:	e8 a8 04 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40067c:	8b 18                	mov    (%rax),%ebx
  40067e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400682:	48 83 c0 04          	add    $0x4,%rax
  400686:	48 89 c7             	mov    %rax,%rdi
  400689:	e8 7c 04 00 00       	callq  400b0a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40068e:	8b 00                	mov    (%rax),%eax
  400690:	89 da                	mov    %ebx,%edx
  400692:	89 c6                	mov    %eax,%esi
  400694:	bf d0 0c 40 00       	mov    $0x400cd0,%edi
  400699:	b8 00 00 00 00       	mov    $0x0,%eax
  40069e:	e8 fd fd ff ff       	callq  4004a0 <printf@plt>
  4006a3:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006a7:	48 83 c0 04          	add    $0x4,%rax
  4006ab:	48 89 c7             	mov    %rax,%rdi
  4006ae:	e8 57 04 00 00       	callq  400b0a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006b3:	8b 00                	mov    (%rax),%eax
  4006b5:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006ba:	75 19                	jne    4006d5 <main+0xff>
  4006bc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006c0:	48 83 c0 04          	add    $0x4,%rax
  4006c4:	48 89 c7             	mov    %rax,%rdi
  4006c7:	e8 58 04 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4006cc:	8b 00                	mov    (%rax),%eax
  4006ce:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006d3:	74 07                	je     4006dc <main+0x106>
  4006d5:	b8 01 00 00 00       	mov    $0x1,%eax
  4006da:	eb 05                	jmp    4006e1 <main+0x10b>
  4006dc:	b8 00 00 00 00       	mov    $0x0,%eax
  4006e1:	0f b6 c0             	movzbl %al,%eax
  4006e4:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006e8:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006ef:	00 00 
  4006f1:	74 05                	je     4006f8 <main+0x122>
  4006f3:	e8 98 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  4006f8:	48 83 c4 28          	add    $0x28,%rsp
  4006fc:	5b                   	pop    %rbx
  4006fd:	5d                   	pop    %rbp
  4006fe:	c3                   	retq   

00000000004006ff <_Z12new_instancev>:
  4006ff:	55                   	push   %rbp
  400700:	48 89 e5             	mov    %rsp,%rbp
  400703:	53                   	push   %rbx
  400704:	48 83 ec 08          	sub    $0x8,%rsp
  400708:	bf 04 00 00 00       	mov    $0x4,%edi
  40070d:	e8 2a 03 00 00       	callq  400a3c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400712:	48 89 c3             	mov    %rax,%rbx
  400715:	b8 00 00 00 00       	mov    $0x0,%eax
  40071a:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  40071e:	48 83 f8 03          	cmp    $0x3,%rax
  400722:	74 06                	je     40072a <_Z12new_instancev+0x2b>
  400724:	48 83 c0 01          	add    $0x1,%rax
  400728:	eb f0                	jmp    40071a <_Z12new_instancev+0x1b>
  40072a:	48 89 df             	mov    %rbx,%rdi
  40072d:	e8 2a 02 00 00       	callq  40095c <_ZN9TestClassC1Ev>
  400732:	48 89 d8             	mov    %rbx,%rax
  400735:	48 83 c4 08          	add    $0x8,%rsp
  400739:	5b                   	pop    %rbx
  40073a:	5d                   	pop    %rbp
  40073b:	c3                   	retq   

000000000040073c <_Z12write_field0P9TestClass>:
  40073c:	55                   	push   %rbp
  40073d:	48 89 e5             	mov    %rsp,%rbp
  400740:	48 83 ec 30          	sub    $0x30,%rsp
  400744:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400748:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40074f:	00 00 
  400751:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400755:	31 c0                	xor    %eax,%eax
  400757:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40075b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40075f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400763:	ba 77 77 00 00       	mov    $0x7777,%edx
  400768:	48 89 ce             	mov    %rcx,%rsi
  40076b:	48 89 c7             	mov    %rax,%rdi
  40076e:	e8 17 03 00 00       	callq  400a8a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400773:	90                   	nop
  400774:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400778:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40077f:	00 00 
  400781:	74 05                	je     400788 <_Z12write_field0P9TestClass+0x4c>
  400783:	e8 08 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  400788:	c9                   	leaveq 
  400789:	c3                   	retq   

000000000040078a <_Z11read_field0P9TestClass>:
  40078a:	55                   	push   %rbp
  40078b:	48 89 e5             	mov    %rsp,%rbp
  40078e:	48 83 ec 10          	sub    $0x10,%rsp
  400792:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400796:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40079a:	48 83 c0 04          	add    $0x4,%rax
  40079e:	48 89 c7             	mov    %rax,%rdi
  4007a1:	e8 64 03 00 00       	callq  400b0a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007a6:	8b 00                	mov    (%rax),%eax
  4007a8:	c9                   	leaveq 
  4007a9:	c3                   	retq   

00000000004007aa <_Z12write_field1P9TestClass>:
  4007aa:	55                   	push   %rbp
  4007ab:	48 89 e5             	mov    %rsp,%rbp
  4007ae:	48 83 ec 30          	sub    $0x30,%rsp
  4007b2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007b6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007bd:	00 00 
  4007bf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007c3:	31 c0                	xor    %eax,%eax
  4007c5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007c9:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4007cd:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007d1:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007d6:	48 89 ce             	mov    %rcx,%rsi
  4007d9:	48 89 c7             	mov    %rax,%rdi
  4007dc:	e8 e9 02 00 00       	callq  400aca <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007e1:	90                   	nop
  4007e2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007e6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007ed:	00 00 
  4007ef:	74 05                	je     4007f6 <_Z12write_field1P9TestClass+0x4c>
  4007f1:	e8 9a fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  4007f6:	c9                   	leaveq 
  4007f7:	c3                   	retq   

00000000004007f8 <_Z11read_field1P9TestClass>:
  4007f8:	55                   	push   %rbp
  4007f9:	48 89 e5             	mov    %rsp,%rbp
  4007fc:	48 83 ec 10          	sub    $0x10,%rsp
  400800:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400804:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400808:	48 83 c0 04          	add    $0x4,%rax
  40080c:	48 89 c7             	mov    %rax,%rdi
  40080f:	e8 10 03 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400814:	8b 00                	mov    (%rax),%eax
  400816:	c9                   	leaveq 
  400817:	c3                   	retq   

0000000000400818 <_Z21explicit_write_field0m>:
  400818:	55                   	push   %rbp
  400819:	48 89 e5             	mov    %rsp,%rbp
  40081c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400820:	48 8b 05 49 18 20 00 	mov    0x201849(%rip),%rax        # 602070 <explicit_field0>
  400827:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  40082b:	48 c1 e2 02          	shl    $0x2,%rdx
  40082f:	48 01 d0             	add    %rdx,%rax
  400832:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  400838:	90                   	nop
  400839:	5d                   	pop    %rbp
  40083a:	c3                   	retq   

000000000040083b <_Z20explicit_read_field0m>:
  40083b:	55                   	push   %rbp
  40083c:	48 89 e5             	mov    %rsp,%rbp
  40083f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400843:	48 8b 05 26 18 20 00 	mov    0x201826(%rip),%rax        # 602070 <explicit_field0>
  40084a:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  40084e:	48 c1 e2 02          	shl    $0x2,%rdx
  400852:	48 01 d0             	add    %rdx,%rax
  400855:	8b 00                	mov    (%rax),%eax
  400857:	5d                   	pop    %rbp
  400858:	c3                   	retq   

0000000000400859 <_Z41__static_initialization_and_destruction_0ii>:
  400859:	55                   	push   %rbp
  40085a:	48 89 e5             	mov    %rsp,%rbp
  40085d:	48 83 ec 10          	sub    $0x10,%rsp
  400861:	89 7d fc             	mov    %edi,-0x4(%rbp)
  400864:	89 75 f8             	mov    %esi,-0x8(%rbp)
  400867:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
  40086b:	75 13                	jne    400880 <_Z41__static_initialization_and_destruction_0ii+0x27>
  40086d:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
  400874:	75 0a                	jne    400880 <_Z41__static_initialization_and_destruction_0ii+0x27>
  400876:	bf 60 20 60 00       	mov    $0x602060,%edi
  40087b:	e8 98 01 00 00       	callq  400a18 <_ZN4ikra3soa14DynamicStorageI9TestClassEC1Ev>
  400880:	90                   	nop
  400881:	c9                   	leaveq 
  400882:	c3                   	retq   

0000000000400883 <_GLOBAL__sub_I__ZN9TestClass7storageE>:
  400883:	55                   	push   %rbp
  400884:	48 89 e5             	mov    %rsp,%rbp
  400887:	be ff ff 00 00       	mov    $0xffff,%esi
  40088c:	bf 01 00 00 00       	mov    $0x1,%edi
  400891:	e8 c3 ff ff ff       	callq  400859 <_Z41__static_initialization_and_destruction_0ii>
  400896:	5d                   	pop    %rbp
  400897:	c3                   	retq   

0000000000400898 <_ZN9TestClass15increase_field0Ev>:
  400898:	55                   	push   %rbp
  400899:	48 89 e5             	mov    %rsp,%rbp
  40089c:	48 83 ec 30          	sub    $0x30,%rsp
  4008a0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008a4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008ab:	00 00 
  4008ad:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008b1:	31 c0                	xor    %eax,%eax
  4008b3:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008b7:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008bb:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008bf:	ba 55 55 00 00       	mov    $0x5555,%edx
  4008c4:	48 89 ce             	mov    %rcx,%rsi
  4008c7:	48 89 c7             	mov    %rax,%rdi
  4008ca:	e8 c9 00 00 00       	callq  400998 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  4008cf:	90                   	nop
  4008d0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008d4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008db:	00 00 
  4008dd:	74 05                	je     4008e4 <_ZN9TestClass15increase_field0Ev+0x4c>
  4008df:	e8 ac fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008e4:	c9                   	leaveq 
  4008e5:	c3                   	retq   

00000000004008e6 <_ZN9TestClass15increase_field1Ev>:
  4008e6:	55                   	push   %rbp
  4008e7:	48 89 e5             	mov    %rsp,%rbp
  4008ea:	48 83 ec 30          	sub    $0x30,%rsp
  4008ee:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008f2:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008f9:	00 00 
  4008fb:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008ff:	31 c0                	xor    %eax,%eax
  400901:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400905:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400909:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40090d:	ba 44 44 00 00       	mov    $0x4444,%edx
  400912:	48 89 ce             	mov    %rcx,%rsi
  400915:	48 89 c7             	mov    %rax,%rdi
  400918:	e8 bb 00 00 00       	callq  4009d8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  40091d:	90                   	nop
  40091e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400922:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400929:	00 00 
  40092b:	74 05                	je     400932 <_ZN9TestClass15increase_field1Ev+0x4c>
  40092d:	e8 5e fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  400932:	c9                   	leaveq 
  400933:	c3                   	retq   

0000000000400934 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400934:	55                   	push   %rbp
  400935:	48 89 e5             	mov    %rsp,%rbp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	90                   	nop
  40093d:	5d                   	pop    %rbp
  40093e:	c3                   	retq   
  40093f:	90                   	nop

0000000000400940 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 83 ec 10          	sub    $0x10,%rsp
  400948:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40094c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400950:	48 89 c7             	mov    %rax,%rdi
  400953:	e8 dc ff ff ff       	callq  400934 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  400958:	90                   	nop
  400959:	c9                   	leaveq 
  40095a:	c3                   	retq   
  40095b:	90                   	nop

000000000040095c <_ZN9TestClassC1Ev>:
  40095c:	55                   	push   %rbp
  40095d:	48 89 e5             	mov    %rsp,%rbp
  400960:	48 83 ec 10          	sub    $0x10,%rsp
  400964:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400968:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096c:	48 89 c7             	mov    %rax,%rdi
  40096f:	e8 cc ff ff ff       	callq  400940 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400974:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400978:	48 83 c0 04          	add    $0x4,%rax
  40097c:	48 89 c7             	mov    %rax,%rdi
  40097f:	e8 e4 00 00 00       	callq  400a68 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400984:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400988:	48 83 c0 04          	add    $0x4,%rax
  40098c:	48 89 c7             	mov    %rax,%rdi
  40098f:	e8 e0 00 00 00       	callq  400a74 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400994:	90                   	nop
  400995:	c9                   	leaveq 
  400996:	c3                   	retq   
  400997:	90                   	nop

0000000000400998 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400998:	55                   	push   %rbp
  400999:	48 89 e5             	mov    %rsp,%rbp
  40099c:	48 83 ec 20          	sub    $0x20,%rsp
  4009a0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009a8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009ab:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009af:	48 89 c7             	mov    %rax,%rdi
  4009b2:	e8 87 01 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009b7:	8b 10                	mov    (%rax),%edx
  4009b9:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009bd:	89 10                	mov    %edx,(%rax)
  4009bf:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009c3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009c7:	48 89 d6             	mov    %rdx,%rsi
  4009ca:	48 89 c7             	mov    %rax,%rdi
  4009cd:	e8 b0 01 00 00       	callq  400b82 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  4009d2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d6:	c9                   	leaveq 
  4009d7:	c3                   	retq   

00000000004009d8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  4009d8:	55                   	push   %rbp
  4009d9:	48 89 e5             	mov    %rsp,%rbp
  4009dc:	48 83 ec 20          	sub    $0x20,%rsp
  4009e0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009e8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009eb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009ef:	48 89 c7             	mov    %rax,%rdi
  4009f2:	e8 9b 01 00 00       	callq  400b92 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009f7:	8b 10                	mov    (%rax),%edx
  4009f9:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009fd:	89 10                	mov    %edx,(%rax)
  4009ff:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a03:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a07:	48 89 d6             	mov    %rdx,%rsi
  400a0a:	48 89 c7             	mov    %rax,%rdi
  400a0d:	e8 ca 01 00 00       	callq  400bdc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a12:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a16:	c9                   	leaveq 
  400a17:	c3                   	retq   

0000000000400a18 <_ZN4ikra3soa14DynamicStorageI9TestClassEC1Ev>:
  400a18:	55                   	push   %rbp
  400a19:	48 89 e5             	mov    %rsp,%rbp
  400a1c:	48 83 ec 10          	sub    $0x10,%rsp
  400a20:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a24:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400a29:	e8 92 fa ff ff       	callq  4004c0 <malloc@plt>
  400a2e:	48 89 c2             	mov    %rax,%rdx
  400a31:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a35:	48 89 50 08          	mov    %rdx,0x8(%rax)
  400a39:	90                   	nop
  400a3a:	c9                   	leaveq 
  400a3b:	c3                   	retq   

0000000000400a3c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a3c:	55                   	push   %rbp
  400a3d:	48 89 e5             	mov    %rsp,%rbp
  400a40:	48 83 ec 10          	sub    $0x10,%rsp
  400a44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a48:	e8 9e 01 00 00       	callq  400beb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a4d:	8b 05 0d 16 20 00    	mov    0x20160d(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  400a53:	8d 50 01             	lea    0x1(%rax),%edx
  400a56:	89 15 04 16 20 00    	mov    %edx,0x201604(%rip)        # 602060 <_ZN9TestClass7storageE>
  400a5c:	89 c0                	mov    %eax,%eax
  400a5e:	48 89 c7             	mov    %rax,%rdi
  400a61:	e8 8c 01 00 00       	callq  400bf2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a66:	c9                   	leaveq 
  400a67:	c3                   	retq   

0000000000400a68 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a68:	55                   	push   %rbp
  400a69:	48 89 e5             	mov    %rsp,%rbp
  400a6c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a70:	90                   	nop
  400a71:	5d                   	pop    %rbp
  400a72:	c3                   	retq   
  400a73:	90                   	nop

0000000000400a74 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a74:	55                   	push   %rbp
  400a75:	48 89 e5             	mov    %rsp,%rbp
  400a78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a7c:	90                   	nop
  400a7d:	5d                   	pop    %rbp
  400a7e:	c3                   	retq   

0000000000400a7f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a7f:	55                   	push   %rbp
  400a80:	48 89 e5             	mov    %rsp,%rbp
  400a83:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a87:	90                   	nop
  400a88:	5d                   	pop    %rbp
  400a89:	c3                   	retq   

0000000000400a8a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a8a:	55                   	push   %rbp
  400a8b:	48 89 e5             	mov    %rsp,%rbp
  400a8e:	48 83 ec 20          	sub    $0x20,%rsp
  400a92:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a96:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a9a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a9d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400aa1:	48 89 c7             	mov    %rax,%rdi
  400aa4:	e8 95 00 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa9:	48 89 c2             	mov    %rax,%rdx
  400aac:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400aaf:	89 02                	mov    %eax,(%rdx)
  400ab1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400ab5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab9:	48 89 d6             	mov    %rdx,%rsi
  400abc:	48 89 c7             	mov    %rax,%rdi
  400abf:	e8 be 00 00 00       	callq  400b82 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400ac4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac8:	c9                   	leaveq 
  400ac9:	c3                   	retq   

0000000000400aca <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400aca:	55                   	push   %rbp
  400acb:	48 89 e5             	mov    %rsp,%rbp
  400ace:	48 83 ec 20          	sub    $0x20,%rsp
  400ad2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ada:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400add:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ae1:	48 89 c7             	mov    %rax,%rdi
  400ae4:	e8 a9 00 00 00       	callq  400b92 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae9:	48 89 c2             	mov    %rax,%rdx
  400aec:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400aef:	89 02                	mov    %eax,(%rdx)
  400af1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400af5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af9:	48 89 d6             	mov    %rdx,%rsi
  400afc:	48 89 c7             	mov    %rax,%rdi
  400aff:	e8 d8 00 00 00       	callq  400bdc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400b04:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b08:	c9                   	leaveq 
  400b09:	c3                   	retq   

0000000000400b0a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b0a:	55                   	push   %rbp
  400b0b:	48 89 e5             	mov    %rsp,%rbp
  400b0e:	48 83 ec 10          	sub    $0x10,%rsp
  400b12:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b16:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1a:	48 89 c7             	mov    %rax,%rdi
  400b1d:	e8 1c 00 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b22:	c9                   	leaveq 
  400b23:	c3                   	retq   

0000000000400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b24:	55                   	push   %rbp
  400b25:	48 89 e5             	mov    %rsp,%rbp
  400b28:	48 83 ec 10          	sub    $0x10,%rsp
  400b2c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b30:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b34:	48 89 c7             	mov    %rax,%rdi
  400b37:	e8 56 00 00 00       	callq  400b92 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b3c:	c9                   	leaveq 
  400b3d:	c3                   	retq   

0000000000400b3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b3e:	55                   	push   %rbp
  400b3f:	48 89 e5             	mov    %rsp,%rbp
  400b42:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b46:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b4a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b4e:	48 8b 05 13 15 20 00 	mov    0x201513(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400b55:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b59:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b5d:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b61:	48 83 e8 04          	sub    $0x4,%rax
  400b65:	48 c1 e8 02          	shr    $0x2,%rax
  400b69:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b70:	00 
  400b71:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b75:	48 01 d0             	add    %rdx,%rax
  400b78:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b7c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b80:	5d                   	pop    %rbp
  400b81:	c3                   	retq   

0000000000400b82 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b82:	55                   	push   %rbp
  400b83:	48 89 e5             	mov    %rsp,%rbp
  400b86:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b8a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b8e:	90                   	nop
  400b8f:	5d                   	pop    %rbp
  400b90:	c3                   	retq   
  400b91:	90                   	nop

0000000000400b92 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b92:	55                   	push   %rbp
  400b93:	48 89 e5             	mov    %rsp,%rbp
  400b96:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b9a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b9e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400ba2:	48 8b 05 bf 14 20 00 	mov    0x2014bf(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400ba9:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bad:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bb1:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bb5:	48 83 e8 04          	sub    $0x4,%rax
  400bb9:	48 c1 e8 02          	shr    $0x2,%rax
  400bbd:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bc4:	00 
  400bc5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bc9:	48 01 d0             	add    %rdx,%rax
  400bcc:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bd2:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bd6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bda:	5d                   	pop    %rbp
  400bdb:	c3                   	retq   

0000000000400bdc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400bdc:	55                   	push   %rbp
  400bdd:	48 89 e5             	mov    %rsp,%rbp
  400be0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400be4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400be8:	90                   	nop
  400be9:	5d                   	pop    %rbp
  400bea:	c3                   	retq   

0000000000400beb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400beb:	55                   	push   %rbp
  400bec:	48 89 e5             	mov    %rsp,%rbp
  400bef:	90                   	nop
  400bf0:	5d                   	pop    %rbp
  400bf1:	c3                   	retq   

0000000000400bf2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400bf2:	55                   	push   %rbp
  400bf3:	48 89 e5             	mov    %rsp,%rbp
  400bf6:	48 83 ec 10          	sub    $0x10,%rsp
  400bfa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bfe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c02:	48 89 c7             	mov    %rax,%rdi
  400c05:	e8 02 00 00 00       	callq  400c0c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400c0a:	c9                   	leaveq 
  400c0b:	c3                   	retq   

0000000000400c0c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c0c:	55                   	push   %rbp
  400c0d:	48 89 e5             	mov    %rsp,%rbp
  400c10:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c14:	48 8b 05 4d 14 20 00 	mov    0x20144d(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400c1b:	48 89 c2             	mov    %rax,%rdx
  400c1e:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c22:	48 c1 e0 02          	shl    $0x2,%rax
  400c26:	48 01 d0             	add    %rdx,%rax
  400c29:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c2d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c31:	5d                   	pop    %rbp
  400c32:	c3                   	retq   
  400c33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c3a:	00 00 00 
  400c3d:	0f 1f 00             	nopl   (%rax)

0000000000400c40 <__libc_csu_init>:
  400c40:	41 57                	push   %r15
  400c42:	41 56                	push   %r14
  400c44:	41 89 ff             	mov    %edi,%r15d
  400c47:	41 55                	push   %r13
  400c49:	41 54                	push   %r12
  400c4b:	4c 8d 25 b6 11 20 00 	lea    0x2011b6(%rip),%r12        # 601e08 <__frame_dummy_init_array_entry>
  400c52:	55                   	push   %rbp
  400c53:	48 8d 2d be 11 20 00 	lea    0x2011be(%rip),%rbp        # 601e18 <__init_array_end>
  400c5a:	53                   	push   %rbx
  400c5b:	49 89 f6             	mov    %rsi,%r14
  400c5e:	49 89 d5             	mov    %rdx,%r13
  400c61:	4c 29 e5             	sub    %r12,%rbp
  400c64:	48 83 ec 08          	sub    $0x8,%rsp
  400c68:	48 c1 fd 03          	sar    $0x3,%rbp
  400c6c:	e8 ef f7 ff ff       	callq  400460 <_init>
  400c71:	48 85 ed             	test   %rbp,%rbp
  400c74:	74 20                	je     400c96 <__libc_csu_init+0x56>
  400c76:	31 db                	xor    %ebx,%ebx
  400c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c7f:	00 
  400c80:	4c 89 ea             	mov    %r13,%rdx
  400c83:	4c 89 f6             	mov    %r14,%rsi
  400c86:	44 89 ff             	mov    %r15d,%edi
  400c89:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c8d:	48 83 c3 01          	add    $0x1,%rbx
  400c91:	48 39 eb             	cmp    %rbp,%rbx
  400c94:	75 ea                	jne    400c80 <__libc_csu_init+0x40>
  400c96:	48 83 c4 08          	add    $0x8,%rsp
  400c9a:	5b                   	pop    %rbx
  400c9b:	5d                   	pop    %rbp
  400c9c:	41 5c                	pop    %r12
  400c9e:	41 5d                	pop    %r13
  400ca0:	41 5e                	pop    %r14
  400ca2:	41 5f                	pop    %r15
  400ca4:	c3                   	retq   
  400ca5:	90                   	nop
  400ca6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cad:	00 00 00 

0000000000400cb0 <__libc_csu_fini>:
  400cb0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400cb4 <_fini>:
  400cb4:	48 83 ec 08          	sub    $0x8,%rsp
  400cb8:	48 83 c4 08          	add    $0x8,%rsp
  400cbc:	c3                   	retq   
