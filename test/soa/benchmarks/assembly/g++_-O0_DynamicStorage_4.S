
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
  4004ef:	49 c7 c0 20 0d 40 00 	mov    $0x400d20,%r8
  4004f6:	48 c7 c1 b0 0c 40 00 	mov    $0x400cb0,%rcx
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
  4005ee:	e8 88 02 00 00       	callq  40087b <_ZN9TestClass18initialize_storageEv>
  4005f3:	bf 04 00 00 00       	mov    $0x4,%edi
  4005f8:	e8 61 04 00 00       	callq  400a5e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005fd:	48 89 c3             	mov    %rax,%rbx
  400600:	b8 00 00 00 00       	mov    $0x0,%eax
  400605:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400609:	48 83 f8 03          	cmp    $0x3,%rax
  40060d:	74 06                	je     400615 <main+0x3f>
  40060f:	48 83 c0 01          	add    $0x1,%rax
  400613:	eb f0                	jmp    400605 <main+0x2f>
  400615:	48 89 df             	mov    %rbx,%rdi
  400618:	e8 45 03 00 00       	callq  400962 <_ZN9TestClassC1Ev>
  40061d:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  400621:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400625:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400629:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40062d:	ba 77 77 00 00       	mov    $0x7777,%edx
  400632:	48 89 ce             	mov    %rcx,%rsi
  400635:	48 89 c7             	mov    %rax,%rdi
  400638:	e8 71 04 00 00       	callq  400aae <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40063d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400641:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400645:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400649:	ba 88 88 00 00       	mov    $0x8888,%edx
  40064e:	48 89 ce             	mov    %rcx,%rsi
  400651:	48 89 c7             	mov    %rax,%rdi
  400654:	e8 95 04 00 00       	callq  400aee <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400659:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40065d:	48 89 c7             	mov    %rax,%rdi
  400660:	e8 39 02 00 00       	callq  40089e <_ZN9TestClass15increase_field0Ev>
  400665:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400669:	48 89 c7             	mov    %rax,%rdi
  40066c:	e8 7b 02 00 00       	callq  4008ec <_ZN9TestClass15increase_field1Ev>
  400671:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400675:	48 83 c0 04          	add    $0x4,%rax
  400679:	48 89 c7             	mov    %rax,%rdi
  40067c:	e8 c7 04 00 00       	callq  400b48 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400681:	8b 18                	mov    (%rax),%ebx
  400683:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400687:	48 83 c0 04          	add    $0x4,%rax
  40068b:	48 89 c7             	mov    %rax,%rdi
  40068e:	e8 9b 04 00 00       	callq  400b2e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400693:	8b 00                	mov    (%rax),%eax
  400695:	89 da                	mov    %ebx,%edx
  400697:	89 c6                	mov    %eax,%esi
  400699:	bf 44 0d 40 00       	mov    $0x400d44,%edi
  40069e:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a3:	e8 f8 fd ff ff       	callq  4004a0 <printf@plt>
  4006a8:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006ac:	48 83 c0 04          	add    $0x4,%rax
  4006b0:	48 89 c7             	mov    %rax,%rdi
  4006b3:	e8 76 04 00 00       	callq  400b2e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006b8:	8b 00                	mov    (%rax),%eax
  4006ba:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006bf:	75 19                	jne    4006da <main+0x104>
  4006c1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006c5:	48 83 c0 04          	add    $0x4,%rax
  4006c9:	48 89 c7             	mov    %rax,%rdi
  4006cc:	e8 77 04 00 00       	callq  400b48 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4006d1:	8b 00                	mov    (%rax),%eax
  4006d3:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006d8:	74 07                	je     4006e1 <main+0x10b>
  4006da:	b8 01 00 00 00       	mov    $0x1,%eax
  4006df:	eb 05                	jmp    4006e6 <main+0x110>
  4006e1:	b8 00 00 00 00       	mov    $0x0,%eax
  4006e6:	0f b6 c0             	movzbl %al,%eax
  4006e9:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006ed:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006f4:	00 00 
  4006f6:	74 05                	je     4006fd <main+0x127>
  4006f8:	e8 93 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  4006fd:	48 83 c4 28          	add    $0x28,%rsp
  400701:	5b                   	pop    %rbx
  400702:	5d                   	pop    %rbp
  400703:	c3                   	retq   

0000000000400704 <_Z12new_instancev>:
  400704:	55                   	push   %rbp
  400705:	48 89 e5             	mov    %rsp,%rbp
  400708:	53                   	push   %rbx
  400709:	48 83 ec 08          	sub    $0x8,%rsp
  40070d:	bf 04 00 00 00       	mov    $0x4,%edi
  400712:	e8 47 03 00 00       	callq  400a5e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400717:	48 89 c3             	mov    %rax,%rbx
  40071a:	b8 00 00 00 00       	mov    $0x0,%eax
  40071f:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400723:	48 83 f8 03          	cmp    $0x3,%rax
  400727:	74 06                	je     40072f <_Z12new_instancev+0x2b>
  400729:	48 83 c0 01          	add    $0x1,%rax
  40072d:	eb f0                	jmp    40071f <_Z12new_instancev+0x1b>
  40072f:	48 89 df             	mov    %rbx,%rdi
  400732:	e8 2b 02 00 00       	callq  400962 <_ZN9TestClassC1Ev>
  400737:	48 89 d8             	mov    %rbx,%rax
  40073a:	48 83 c4 08          	add    $0x8,%rsp
  40073e:	5b                   	pop    %rbx
  40073f:	5d                   	pop    %rbp
  400740:	c3                   	retq   

0000000000400741 <_Z12write_field0P9TestClass>:
  400741:	55                   	push   %rbp
  400742:	48 89 e5             	mov    %rsp,%rbp
  400745:	48 83 ec 30          	sub    $0x30,%rsp
  400749:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40074d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400754:	00 00 
  400756:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40075a:	31 c0                	xor    %eax,%eax
  40075c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400760:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400764:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400768:	ba 77 77 00 00       	mov    $0x7777,%edx
  40076d:	48 89 ce             	mov    %rcx,%rsi
  400770:	48 89 c7             	mov    %rax,%rdi
  400773:	e8 36 03 00 00       	callq  400aae <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400778:	90                   	nop
  400779:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40077d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400784:	00 00 
  400786:	74 05                	je     40078d <_Z12write_field0P9TestClass+0x4c>
  400788:	e8 03 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  40078d:	c9                   	leaveq 
  40078e:	c3                   	retq   

000000000040078f <_Z11read_field0P9TestClass>:
  40078f:	55                   	push   %rbp
  400790:	48 89 e5             	mov    %rsp,%rbp
  400793:	48 83 ec 10          	sub    $0x10,%rsp
  400797:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40079b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40079f:	48 83 c0 04          	add    $0x4,%rax
  4007a3:	48 89 c7             	mov    %rax,%rdi
  4007a6:	e8 83 03 00 00       	callq  400b2e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007ab:	8b 00                	mov    (%rax),%eax
  4007ad:	c9                   	leaveq 
  4007ae:	c3                   	retq   

00000000004007af <_Z12write_field1P9TestClass>:
  4007af:	55                   	push   %rbp
  4007b0:	48 89 e5             	mov    %rsp,%rbp
  4007b3:	48 83 ec 30          	sub    $0x30,%rsp
  4007b7:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007bb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007c2:	00 00 
  4007c4:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007c8:	31 c0                	xor    %eax,%eax
  4007ca:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007ce:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4007d2:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007d6:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007db:	48 89 ce             	mov    %rcx,%rsi
  4007de:	48 89 c7             	mov    %rax,%rdi
  4007e1:	e8 08 03 00 00       	callq  400aee <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007e6:	90                   	nop
  4007e7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007eb:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007f2:	00 00 
  4007f4:	74 05                	je     4007fb <_Z12write_field1P9TestClass+0x4c>
  4007f6:	e8 95 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  4007fb:	c9                   	leaveq 
  4007fc:	c3                   	retq   

00000000004007fd <_Z11read_field1P9TestClass>:
  4007fd:	55                   	push   %rbp
  4007fe:	48 89 e5             	mov    %rsp,%rbp
  400801:	48 83 ec 10          	sub    $0x10,%rsp
  400805:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400809:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40080d:	48 83 c0 04          	add    $0x4,%rax
  400811:	48 89 c7             	mov    %rax,%rdi
  400814:	e8 2f 03 00 00       	callq  400b48 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400819:	8b 00                	mov    (%rax),%eax
  40081b:	c9                   	leaveq 
  40081c:	c3                   	retq   

000000000040081d <_Z21explicit_write_field0m>:
  40081d:	55                   	push   %rbp
  40081e:	48 89 e5             	mov    %rsp,%rbp
  400821:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400825:	48 8b 05 f4 9e 21 00 	mov    0x219ef4(%rip),%rax        # 61a720 <explicit_field0>
  40082c:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400830:	48 c1 e2 02          	shl    $0x2,%rdx
  400834:	48 01 d0             	add    %rdx,%rax
  400837:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  40083d:	90                   	nop
  40083e:	5d                   	pop    %rbp
  40083f:	c3                   	retq   

0000000000400840 <_Z20explicit_read_field0m>:
  400840:	55                   	push   %rbp
  400841:	48 89 e5             	mov    %rsp,%rbp
  400844:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400848:	48 8b 05 d1 9e 21 00 	mov    0x219ed1(%rip),%rax        # 61a720 <explicit_field0>
  40084f:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400853:	48 c1 e2 02          	shl    $0x2,%rdx
  400857:	48 01 d0             	add    %rdx,%rax
  40085a:	8b 00                	mov    (%rax),%eax
  40085c:	5d                   	pop    %rbp
  40085d:	c3                   	retq   

000000000040085e <_ZnwmPv>:
  40085e:	55                   	push   %rbp
  40085f:	48 89 e5             	mov    %rsp,%rbp
  400862:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400866:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40086a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40086e:	5d                   	pop    %rbp
  40086f:	c3                   	retq   

0000000000400870 <_ZN9TestClass7storageEv>:
  400870:	55                   	push   %rbp
  400871:	48 89 e5             	mov    %rsp,%rbp
  400874:	b8 80 20 60 00       	mov    $0x602080,%eax
  400879:	5d                   	pop    %rbp
  40087a:	c3                   	retq   

000000000040087b <_ZN9TestClass18initialize_storageEv>:
  40087b:	55                   	push   %rbp
  40087c:	48 89 e5             	mov    %rsp,%rbp
  40087f:	be 80 20 60 00       	mov    $0x602080,%esi
  400884:	bf 20 00 00 00       	mov    $0x20,%edi
  400889:	e8 d0 ff ff ff       	callq  40085e <_ZnwmPv>
  40088e:	48 85 c0             	test   %rax,%rax
  400891:	74 08                	je     40089b <_ZN9TestClass18initialize_storageEv+0x20>
  400893:	48 89 c7             	mov    %rax,%rdi
  400896:	e8 03 01 00 00       	callq  40099e <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  40089b:	90                   	nop
  40089c:	5d                   	pop    %rbp
  40089d:	c3                   	retq   

000000000040089e <_ZN9TestClass15increase_field0Ev>:
  40089e:	55                   	push   %rbp
  40089f:	48 89 e5             	mov    %rsp,%rbp
  4008a2:	48 83 ec 30          	sub    $0x30,%rsp
  4008a6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008aa:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008b1:	00 00 
  4008b3:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008b7:	31 c0                	xor    %eax,%eax
  4008b9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008bd:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008c1:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008c5:	ba 55 55 00 00       	mov    $0x5555,%edx
  4008ca:	48 89 ce             	mov    %rcx,%rsi
  4008cd:	48 89 c7             	mov    %rax,%rdi
  4008d0:	e8 09 01 00 00       	callq  4009de <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  4008d5:	90                   	nop
  4008d6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008da:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008e1:	00 00 
  4008e3:	74 05                	je     4008ea <_ZN9TestClass15increase_field0Ev+0x4c>
  4008e5:	e8 a6 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008ea:	c9                   	leaveq 
  4008eb:	c3                   	retq   

00000000004008ec <_ZN9TestClass15increase_field1Ev>:
  4008ec:	55                   	push   %rbp
  4008ed:	48 89 e5             	mov    %rsp,%rbp
  4008f0:	48 83 ec 30          	sub    $0x30,%rsp
  4008f4:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008f8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008ff:	00 00 
  400901:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400905:	31 c0                	xor    %eax,%eax
  400907:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40090b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40090f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400913:	ba 44 44 00 00       	mov    $0x4444,%edx
  400918:	48 89 ce             	mov    %rcx,%rsi
  40091b:	48 89 c7             	mov    %rax,%rdi
  40091e:	e8 fb 00 00 00       	callq  400a1e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400923:	90                   	nop
  400924:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400928:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40092f:	00 00 
  400931:	74 05                	je     400938 <_ZN9TestClass15increase_field1Ev+0x4c>
  400933:	e8 58 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  400938:	c9                   	leaveq 
  400939:	c3                   	retq   

000000000040093a <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  40093a:	55                   	push   %rbp
  40093b:	48 89 e5             	mov    %rsp,%rbp
  40093e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400942:	90                   	nop
  400943:	5d                   	pop    %rbp
  400944:	c3                   	retq   
  400945:	90                   	nop

0000000000400946 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  400946:	55                   	push   %rbp
  400947:	48 89 e5             	mov    %rsp,%rbp
  40094a:	48 83 ec 10          	sub    $0x10,%rsp
  40094e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400952:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400956:	48 89 c7             	mov    %rax,%rdi
  400959:	e8 dc ff ff ff       	callq  40093a <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40095e:	90                   	nop
  40095f:	c9                   	leaveq 
  400960:	c3                   	retq   
  400961:	90                   	nop

0000000000400962 <_ZN9TestClassC1Ev>:
  400962:	55                   	push   %rbp
  400963:	48 89 e5             	mov    %rsp,%rbp
  400966:	48 83 ec 10          	sub    $0x10,%rsp
  40096a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40096e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400972:	48 89 c7             	mov    %rax,%rdi
  400975:	e8 cc ff ff ff       	callq  400946 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  40097a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40097e:	48 83 c0 04          	add    $0x4,%rax
  400982:	48 89 c7             	mov    %rax,%rdi
  400985:	e8 02 01 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  40098a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40098e:	48 83 c0 04          	add    $0x4,%rax
  400992:	48 89 c7             	mov    %rax,%rdi
  400995:	e8 fe 00 00 00       	callq  400a98 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  40099a:	90                   	nop
  40099b:	c9                   	leaveq 
  40099c:	c3                   	retq   
  40099d:	90                   	nop

000000000040099e <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  40099e:	55                   	push   %rbp
  40099f:	48 89 e5             	mov    %rsp,%rbp
  4009a2:	48 83 ec 10          	sub    $0x10,%rsp
  4009a6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009aa:	bf a0 91 00 00       	mov    $0x91a0,%edi
  4009af:	e8 0c fb ff ff       	callq  4004c0 <malloc@plt>
  4009b4:	48 89 c2             	mov    %rax,%rdx
  4009b7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009bb:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4009bf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009c3:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  4009ca:	00 
  4009cb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009cf:	48 8b 50 18          	mov    0x18(%rax),%rdx
  4009d3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d7:	48 89 50 10          	mov    %rdx,0x10(%rax)
  4009db:	90                   	nop
  4009dc:	c9                   	leaveq 
  4009dd:	c3                   	retq   

00000000004009de <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  4009de:	55                   	push   %rbp
  4009df:	48 89 e5             	mov    %rsp,%rbp
  4009e2:	48 83 ec 20          	sub    $0x20,%rsp
  4009e6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ea:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009ee:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009f1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009f5:	48 89 c7             	mov    %rax,%rdi
  4009f8:	e8 65 01 00 00       	callq  400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009fd:	8b 10                	mov    (%rax),%edx
  4009ff:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400a03:	89 10                	mov    %edx,(%rax)
  400a05:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a09:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a0d:	48 89 d6             	mov    %rdx,%rsi
  400a10:	48 89 c7             	mov    %rax,%rdi
  400a13:	e8 64 01 00 00       	callq  400b7c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a18:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a1c:	c9                   	leaveq 
  400a1d:	c3                   	retq   

0000000000400a1e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400a1e:	55                   	push   %rbp
  400a1f:	48 89 e5             	mov    %rsp,%rbp
  400a22:	48 83 ec 20          	sub    $0x20,%rsp
  400a26:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a2e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a31:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a35:	48 89 c7             	mov    %rax,%rdi
  400a38:	e8 4f 01 00 00       	callq  400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3d:	8b 10                	mov    (%rax),%edx
  400a3f:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400a43:	89 10                	mov    %edx,(%rax)
  400a45:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a49:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a4d:	48 89 d6             	mov    %rdx,%rsi
  400a50:	48 89 c7             	mov    %rax,%rdi
  400a53:	e8 4e 01 00 00       	callq  400ba6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a58:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a5c:	c9                   	leaveq 
  400a5d:	c3                   	retq   

0000000000400a5e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a5e:	55                   	push   %rbp
  400a5f:	48 89 e5             	mov    %rsp,%rbp
  400a62:	48 83 ec 10          	sub    $0x10,%rsp
  400a66:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6a:	e8 46 01 00 00       	callq  400bb5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a6f:	e8 fc fd ff ff       	callq  400870 <_ZN9TestClass7storageEv>
  400a74:	48 89 c2             	mov    %rax,%rdx
  400a77:	48 8b 02             	mov    (%rdx),%rax
  400a7a:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a7e:	48 89 0a             	mov    %rcx,(%rdx)
  400a81:	48 89 c7             	mov    %rax,%rdi
  400a84:	e8 33 01 00 00       	callq  400bbc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a89:	c9                   	leaveq 
  400a8a:	c3                   	retq   
  400a8b:	90                   	nop

0000000000400a8c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a8c:	55                   	push   %rbp
  400a8d:	48 89 e5             	mov    %rsp,%rbp
  400a90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a94:	90                   	nop
  400a95:	5d                   	pop    %rbp
  400a96:	c3                   	retq   
  400a97:	90                   	nop

0000000000400a98 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a98:	55                   	push   %rbp
  400a99:	48 89 e5             	mov    %rsp,%rbp
  400a9c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa0:	90                   	nop
  400aa1:	5d                   	pop    %rbp
  400aa2:	c3                   	retq   

0000000000400aa3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400aa3:	55                   	push   %rbp
  400aa4:	48 89 e5             	mov    %rsp,%rbp
  400aa7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aab:	90                   	nop
  400aac:	5d                   	pop    %rbp
  400aad:	c3                   	retq   

0000000000400aae <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400aae:	55                   	push   %rbp
  400aaf:	48 89 e5             	mov    %rsp,%rbp
  400ab2:	48 83 ec 20          	sub    $0x20,%rsp
  400ab6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aba:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400abe:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400ac1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ac5:	48 89 c7             	mov    %rax,%rdi
  400ac8:	e8 95 00 00 00       	callq  400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400acd:	48 89 c2             	mov    %rax,%rdx
  400ad0:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400ad3:	89 02                	mov    %eax,(%rdx)
  400ad5:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400ad9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400add:	48 89 d6             	mov    %rdx,%rsi
  400ae0:	48 89 c7             	mov    %rax,%rdi
  400ae3:	e8 94 00 00 00       	callq  400b7c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400ae8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aec:	c9                   	leaveq 
  400aed:	c3                   	retq   

0000000000400aee <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400aee:	55                   	push   %rbp
  400aef:	48 89 e5             	mov    %rsp,%rbp
  400af2:	48 83 ec 20          	sub    $0x20,%rsp
  400af6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afa:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400afe:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400b01:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b05:	48 89 c7             	mov    %rax,%rdi
  400b08:	e8 7f 00 00 00       	callq  400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b0d:	48 89 c2             	mov    %rax,%rdx
  400b10:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400b13:	89 02                	mov    %eax,(%rdx)
  400b15:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400b19:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1d:	48 89 d6             	mov    %rdx,%rsi
  400b20:	48 89 c7             	mov    %rax,%rdi
  400b23:	e8 7e 00 00 00       	callq  400ba6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400b28:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b2c:	c9                   	leaveq 
  400b2d:	c3                   	retq   

0000000000400b2e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b2e:	55                   	push   %rbp
  400b2f:	48 89 e5             	mov    %rsp,%rbp
  400b32:	48 83 ec 10          	sub    $0x10,%rsp
  400b36:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b3a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b3e:	48 89 c7             	mov    %rax,%rdi
  400b41:	e8 1c 00 00 00       	callq  400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b46:	c9                   	leaveq 
  400b47:	c3                   	retq   

0000000000400b48 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b48:	55                   	push   %rbp
  400b49:	48 89 e5             	mov    %rsp,%rbp
  400b4c:	48 83 ec 10          	sub    $0x10,%rsp
  400b50:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b54:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b58:	48 89 c7             	mov    %rax,%rdi
  400b5b:	e8 2c 00 00 00       	callq  400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b60:	c9                   	leaveq 
  400b61:	c3                   	retq   

0000000000400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b62:	55                   	push   %rbp
  400b63:	48 89 e5             	mov    %rsp,%rbp
  400b66:	48 83 ec 10          	sub    $0x10,%rsp
  400b6a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b6e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b72:	48 89 c7             	mov    %rax,%rdi
  400b75:	e8 60 00 00 00       	callq  400bda <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b7a:	c9                   	leaveq 
  400b7b:	c3                   	retq   

0000000000400b7c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b7c:	55                   	push   %rbp
  400b7d:	48 89 e5             	mov    %rsp,%rbp
  400b80:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b84:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b88:	90                   	nop
  400b89:	5d                   	pop    %rbp
  400b8a:	c3                   	retq   
  400b8b:	90                   	nop

0000000000400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b8c:	55                   	push   %rbp
  400b8d:	48 89 e5             	mov    %rsp,%rbp
  400b90:	48 83 ec 10          	sub    $0x10,%rsp
  400b94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b98:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b9c:	48 89 c7             	mov    %rax,%rdi
  400b9f:	e8 80 00 00 00       	callq  400c24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ba4:	c9                   	leaveq 
  400ba5:	c3                   	retq   

0000000000400ba6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400ba6:	55                   	push   %rbp
  400ba7:	48 89 e5             	mov    %rsp,%rbp
  400baa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bae:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bb2:	90                   	nop
  400bb3:	5d                   	pop    %rbp
  400bb4:	c3                   	retq   

0000000000400bb5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400bb5:	55                   	push   %rbp
  400bb6:	48 89 e5             	mov    %rsp,%rbp
  400bb9:	90                   	nop
  400bba:	5d                   	pop    %rbp
  400bbb:	c3                   	retq   

0000000000400bbc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400bbc:	55                   	push   %rbp
  400bbd:	48 89 e5             	mov    %rsp,%rbp
  400bc0:	48 83 ec 10          	sub    $0x10,%rsp
  400bc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bc8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bcc:	48 83 c0 01          	add    $0x1,%rax
  400bd0:	48 89 c7             	mov    %rax,%rdi
  400bd3:	e8 9c 00 00 00       	callq  400c74 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bd8:	c9                   	leaveq 
  400bd9:	c3                   	retq   

0000000000400bda <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bda:	55                   	push   %rbp
  400bdb:	48 89 e5             	mov    %rsp,%rbp
  400bde:	48 83 ec 30          	sub    $0x30,%rsp
  400be2:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400be6:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bea:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bee:	e8 7d fc ff ff       	callq  400870 <_ZN9TestClass7storageEv>
  400bf3:	48 8b 40 08          	mov    0x8(%rax),%rax
  400bf7:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bfb:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bff:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400c03:	48 83 e8 04          	sub    $0x4,%rax
  400c07:	48 c1 e8 02          	shr    $0x2,%rax
  400c0b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c12:	00 
  400c13:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c17:	48 01 d0             	add    %rdx,%rax
  400c1a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c1e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c22:	c9                   	leaveq 
  400c23:	c3                   	retq   

0000000000400c24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c24:	55                   	push   %rbp
  400c25:	48 89 e5             	mov    %rsp,%rbp
  400c28:	48 83 ec 30          	sub    $0x30,%rsp
  400c2c:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400c30:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400c34:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c38:	e8 33 fc ff ff       	callq  400870 <_ZN9TestClass7storageEv>
  400c3d:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c41:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400c45:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c49:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400c4d:	48 83 e8 04          	sub    $0x4,%rax
  400c51:	48 c1 e8 02          	shr    $0x2,%rax
  400c55:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c5c:	00 
  400c5d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c61:	48 01 d0             	add    %rdx,%rax
  400c64:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c6a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c6e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c72:	c9                   	leaveq 
  400c73:	c3                   	retq   

0000000000400c74 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c74:	55                   	push   %rbp
  400c75:	48 89 e5             	mov    %rsp,%rbp
  400c78:	48 83 ec 20          	sub    $0x20,%rsp
  400c7c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c80:	e8 eb fb ff ff       	callq  400870 <_ZN9TestClass7storageEv>
  400c85:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c89:	48 89 c2             	mov    %rax,%rdx
  400c8c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c90:	48 c1 e0 02          	shl    $0x2,%rax
  400c94:	48 01 d0             	add    %rdx,%rax
  400c97:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c9b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c9f:	c9                   	leaveq 
  400ca0:	c3                   	retq   
  400ca1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ca8:	00 00 00 
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <__libc_csu_init>:
  400cb0:	41 57                	push   %r15
  400cb2:	41 56                	push   %r14
  400cb4:	41 89 ff             	mov    %edi,%r15d
  400cb7:	41 55                	push   %r13
  400cb9:	41 54                	push   %r12
  400cbb:	4c 8d 25 4e 11 20 00 	lea    0x20114e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400cc2:	55                   	push   %rbp
  400cc3:	48 8d 2d 4e 11 20 00 	lea    0x20114e(%rip),%rbp        # 601e18 <__init_array_end>
  400cca:	53                   	push   %rbx
  400ccb:	49 89 f6             	mov    %rsi,%r14
  400cce:	49 89 d5             	mov    %rdx,%r13
  400cd1:	4c 29 e5             	sub    %r12,%rbp
  400cd4:	48 83 ec 08          	sub    $0x8,%rsp
  400cd8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cdc:	e8 7f f7 ff ff       	callq  400460 <_init>
  400ce1:	48 85 ed             	test   %rbp,%rbp
  400ce4:	74 20                	je     400d06 <__libc_csu_init+0x56>
  400ce6:	31 db                	xor    %ebx,%ebx
  400ce8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400cef:	00 
  400cf0:	4c 89 ea             	mov    %r13,%rdx
  400cf3:	4c 89 f6             	mov    %r14,%rsi
  400cf6:	44 89 ff             	mov    %r15d,%edi
  400cf9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400cfd:	48 83 c3 01          	add    $0x1,%rbx
  400d01:	48 39 eb             	cmp    %rbp,%rbx
  400d04:	75 ea                	jne    400cf0 <__libc_csu_init+0x40>
  400d06:	48 83 c4 08          	add    $0x8,%rsp
  400d0a:	5b                   	pop    %rbx
  400d0b:	5d                   	pop    %rbp
  400d0c:	41 5c                	pop    %r12
  400d0e:	41 5d                	pop    %r13
  400d10:	41 5e                	pop    %r14
  400d12:	41 5f                	pop    %r15
  400d14:	c3                   	retq   
  400d15:	90                   	nop
  400d16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1d:	00 00 00 

0000000000400d20 <__libc_csu_fini>:
  400d20:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d24 <_fini>:
  400d24:	48 83 ec 08          	sub    $0x8,%rsp
  400d28:	48 83 c4 08          	add    $0x8,%rsp
  400d2c:	c3                   	retq   
