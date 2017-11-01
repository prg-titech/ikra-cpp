
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
  4004ef:	49 c7 c0 00 0d 40 00 	mov    $0x400d00,%r8
  4004f6:	48 c7 c1 90 0c 40 00 	mov    $0x400c90,%rcx
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
  4005ee:	e8 76 02 00 00       	callq  400869 <_ZN9TestClass18initialize_storageEv>
  4005f3:	bf 04 00 00 00       	mov    $0x4,%edi
  4005f8:	e8 4f 04 00 00       	callq  400a4c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005fd:	48 89 c3             	mov    %rax,%rbx
  400600:	b8 00 00 00 00       	mov    $0x0,%eax
  400605:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400609:	48 83 f8 03          	cmp    $0x3,%rax
  40060d:	74 06                	je     400615 <main+0x3f>
  40060f:	48 83 c0 01          	add    $0x1,%rax
  400613:	eb f0                	jmp    400605 <main+0x2f>
  400615:	48 89 df             	mov    %rbx,%rdi
  400618:	e8 33 03 00 00       	callq  400950 <_ZN9TestClassC1Ev>
  40061d:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  400621:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400625:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400629:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40062d:	ba 77 77 00 00       	mov    $0x7777,%edx
  400632:	48 89 ce             	mov    %rcx,%rsi
  400635:	48 89 c7             	mov    %rax,%rdi
  400638:	e8 5f 04 00 00       	callq  400a9c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40063d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400641:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400645:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400649:	ba 88 88 00 00       	mov    $0x8888,%edx
  40064e:	48 89 ce             	mov    %rcx,%rsi
  400651:	48 89 c7             	mov    %rax,%rdi
  400654:	e8 83 04 00 00       	callq  400adc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400659:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40065d:	48 89 c7             	mov    %rax,%rdi
  400660:	e8 27 02 00 00       	callq  40088c <_ZN9TestClass15increase_field0Ev>
  400665:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400669:	48 89 c7             	mov    %rax,%rdi
  40066c:	e8 69 02 00 00       	callq  4008da <_ZN9TestClass15increase_field1Ev>
  400671:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400675:	48 83 c0 04          	add    $0x4,%rax
  400679:	48 89 c7             	mov    %rax,%rdi
  40067c:	e8 b5 04 00 00       	callq  400b36 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400681:	8b 18                	mov    (%rax),%ebx
  400683:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400687:	48 83 c0 04          	add    $0x4,%rax
  40068b:	48 89 c7             	mov    %rax,%rdi
  40068e:	e8 89 04 00 00       	callq  400b1c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400693:	8b 00                	mov    (%rax),%eax
  400695:	89 da                	mov    %ebx,%edx
  400697:	89 c6                	mov    %eax,%esi
  400699:	bf 24 0d 40 00       	mov    $0x400d24,%edi
  40069e:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a3:	e8 f8 fd ff ff       	callq  4004a0 <printf@plt>
  4006a8:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006ac:	48 83 c0 04          	add    $0x4,%rax
  4006b0:	48 89 c7             	mov    %rax,%rdi
  4006b3:	e8 64 04 00 00       	callq  400b1c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006b8:	8b 00                	mov    (%rax),%eax
  4006ba:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006bf:	75 19                	jne    4006da <main+0x104>
  4006c1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006c5:	48 83 c0 04          	add    $0x4,%rax
  4006c9:	48 89 c7             	mov    %rax,%rdi
  4006cc:	e8 65 04 00 00       	callq  400b36 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
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
  400712:	e8 35 03 00 00       	callq  400a4c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400717:	48 89 c3             	mov    %rax,%rbx
  40071a:	b8 00 00 00 00       	mov    $0x0,%eax
  40071f:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400723:	48 83 f8 03          	cmp    $0x3,%rax
  400727:	74 06                	je     40072f <_Z12new_instancev+0x2b>
  400729:	48 83 c0 01          	add    $0x1,%rax
  40072d:	eb f0                	jmp    40071f <_Z12new_instancev+0x1b>
  40072f:	48 89 df             	mov    %rbx,%rdi
  400732:	e8 19 02 00 00       	callq  400950 <_ZN9TestClassC1Ev>
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
  400773:	e8 24 03 00 00       	callq  400a9c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
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
  4007a6:	e8 71 03 00 00       	callq  400b1c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
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
  4007e1:	e8 f6 02 00 00       	callq  400adc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
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
  400814:	e8 1d 03 00 00       	callq  400b36 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400819:	8b 00                	mov    (%rax),%eax
  40081b:	c9                   	leaveq 
  40081c:	c3                   	retq   

000000000040081d <_Z21explicit_write_field0m>:
  40081d:	55                   	push   %rbp
  40081e:	48 89 e5             	mov    %rsp,%rbp
  400821:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400825:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400829:	c7 04 85 20 a7 61 00 	movl   $0x7777,0x61a720(,%rax,4)
  400830:	77 77 00 00 
  400834:	90                   	nop
  400835:	5d                   	pop    %rbp
  400836:	c3                   	retq   

0000000000400837 <_Z20explicit_read_field0m>:
  400837:	55                   	push   %rbp
  400838:	48 89 e5             	mov    %rsp,%rbp
  40083b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40083f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400843:	8b 04 85 20 a7 61 00 	mov    0x61a720(,%rax,4),%eax
  40084a:	5d                   	pop    %rbp
  40084b:	c3                   	retq   

000000000040084c <_ZnwmPv>:
  40084c:	55                   	push   %rbp
  40084d:	48 89 e5             	mov    %rsp,%rbp
  400850:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400854:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400858:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40085c:	5d                   	pop    %rbp
  40085d:	c3                   	retq   

000000000040085e <_ZN9TestClass7storageEv>:
  40085e:	55                   	push   %rbp
  40085f:	48 89 e5             	mov    %rsp,%rbp
  400862:	b8 80 20 60 00       	mov    $0x602080,%eax
  400867:	5d                   	pop    %rbp
  400868:	c3                   	retq   

0000000000400869 <_ZN9TestClass18initialize_storageEv>:
  400869:	55                   	push   %rbp
  40086a:	48 89 e5             	mov    %rsp,%rbp
  40086d:	be 80 20 60 00       	mov    $0x602080,%esi
  400872:	bf 20 00 00 00       	mov    $0x20,%edi
  400877:	e8 d0 ff ff ff       	callq  40084c <_ZnwmPv>
  40087c:	48 85 c0             	test   %rax,%rax
  40087f:	74 08                	je     400889 <_ZN9TestClass18initialize_storageEv+0x20>
  400881:	48 89 c7             	mov    %rax,%rdi
  400884:	e8 03 01 00 00       	callq  40098c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400889:	90                   	nop
  40088a:	5d                   	pop    %rbp
  40088b:	c3                   	retq   

000000000040088c <_ZN9TestClass15increase_field0Ev>:
  40088c:	55                   	push   %rbp
  40088d:	48 89 e5             	mov    %rsp,%rbp
  400890:	48 83 ec 30          	sub    $0x30,%rsp
  400894:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400898:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40089f:	00 00 
  4008a1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008a5:	31 c0                	xor    %eax,%eax
  4008a7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008ab:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008af:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008b3:	ba 55 55 00 00       	mov    $0x5555,%edx
  4008b8:	48 89 ce             	mov    %rcx,%rsi
  4008bb:	48 89 c7             	mov    %rax,%rdi
  4008be:	e8 09 01 00 00       	callq  4009cc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  4008c3:	90                   	nop
  4008c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008cf:	00 00 
  4008d1:	74 05                	je     4008d8 <_ZN9TestClass15increase_field0Ev+0x4c>
  4008d3:	e8 b8 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008d8:	c9                   	leaveq 
  4008d9:	c3                   	retq   

00000000004008da <_ZN9TestClass15increase_field1Ev>:
  4008da:	55                   	push   %rbp
  4008db:	48 89 e5             	mov    %rsp,%rbp
  4008de:	48 83 ec 30          	sub    $0x30,%rsp
  4008e2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008e6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008ed:	00 00 
  4008ef:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008f3:	31 c0                	xor    %eax,%eax
  4008f5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008f9:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008fd:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400901:	ba 44 44 00 00       	mov    $0x4444,%edx
  400906:	48 89 ce             	mov    %rcx,%rsi
  400909:	48 89 c7             	mov    %rax,%rdi
  40090c:	e8 fb 00 00 00       	callq  400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400911:	90                   	nop
  400912:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400916:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40091d:	00 00 
  40091f:	74 05                	je     400926 <_ZN9TestClass15increase_field1Ev+0x4c>
  400921:	e8 6a fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  400926:	c9                   	leaveq 
  400927:	c3                   	retq   

0000000000400928 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400928:	55                   	push   %rbp
  400929:	48 89 e5             	mov    %rsp,%rbp
  40092c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400930:	90                   	nop
  400931:	5d                   	pop    %rbp
  400932:	c3                   	retq   
  400933:	90                   	nop

0000000000400934 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  400934:	55                   	push   %rbp
  400935:	48 89 e5             	mov    %rsp,%rbp
  400938:	48 83 ec 10          	sub    $0x10,%rsp
  40093c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400940:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400944:	48 89 c7             	mov    %rax,%rdi
  400947:	e8 dc ff ff ff       	callq  400928 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40094c:	90                   	nop
  40094d:	c9                   	leaveq 
  40094e:	c3                   	retq   
  40094f:	90                   	nop

0000000000400950 <_ZN9TestClassC1Ev>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 83 ec 10          	sub    $0x10,%rsp
  400958:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40095c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400960:	48 89 c7             	mov    %rax,%rdi
  400963:	e8 cc ff ff ff       	callq  400934 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400968:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096c:	48 83 c0 04          	add    $0x4,%rax
  400970:	48 89 c7             	mov    %rax,%rdi
  400973:	e8 02 01 00 00       	callq  400a7a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400978:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40097c:	48 83 c0 04          	add    $0x4,%rax
  400980:	48 89 c7             	mov    %rax,%rdi
  400983:	e8 fe 00 00 00       	callq  400a86 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400988:	90                   	nop
  400989:	c9                   	leaveq 
  40098a:	c3                   	retq   
  40098b:	90                   	nop

000000000040098c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  40098c:	55                   	push   %rbp
  40098d:	48 89 e5             	mov    %rsp,%rbp
  400990:	48 83 ec 10          	sub    $0x10,%rsp
  400994:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400998:	bf a0 91 00 00       	mov    $0x91a0,%edi
  40099d:	e8 1e fb ff ff       	callq  4004c0 <malloc@plt>
  4009a2:	48 89 c2             	mov    %rax,%rdx
  4009a5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a9:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4009ad:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009b1:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  4009b8:	00 
  4009b9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009bd:	48 8b 50 18          	mov    0x18(%rax),%rdx
  4009c1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009c5:	48 89 50 10          	mov    %rdx,0x10(%rax)
  4009c9:	90                   	nop
  4009ca:	c9                   	leaveq 
  4009cb:	c3                   	retq   

00000000004009cc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  4009cc:	55                   	push   %rbp
  4009cd:	48 89 e5             	mov    %rsp,%rbp
  4009d0:	48 83 ec 20          	sub    $0x20,%rsp
  4009d4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009dc:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009df:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009e3:	48 89 c7             	mov    %rax,%rdi
  4009e6:	e8 65 01 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009eb:	8b 10                	mov    (%rax),%edx
  4009ed:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009f1:	89 10                	mov    %edx,(%rax)
  4009f3:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009f7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009fb:	48 89 d6             	mov    %rdx,%rsi
  4009fe:	48 89 c7             	mov    %rax,%rdi
  400a01:	e8 64 01 00 00       	callq  400b6a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a06:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a0a:	c9                   	leaveq 
  400a0b:	c3                   	retq   

0000000000400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400a0c:	55                   	push   %rbp
  400a0d:	48 89 e5             	mov    %rsp,%rbp
  400a10:	48 83 ec 20          	sub    $0x20,%rsp
  400a14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a18:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a1c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a1f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a23:	48 89 c7             	mov    %rax,%rdi
  400a26:	e8 4f 01 00 00       	callq  400b7a <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a2b:	8b 10                	mov    (%rax),%edx
  400a2d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400a31:	89 10                	mov    %edx,(%rax)
  400a33:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a37:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a3b:	48 89 d6             	mov    %rdx,%rsi
  400a3e:	48 89 c7             	mov    %rax,%rdi
  400a41:	e8 4e 01 00 00       	callq  400b94 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a46:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a4a:	c9                   	leaveq 
  400a4b:	c3                   	retq   

0000000000400a4c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a4c:	55                   	push   %rbp
  400a4d:	48 89 e5             	mov    %rsp,%rbp
  400a50:	48 83 ec 10          	sub    $0x10,%rsp
  400a54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a58:	e8 46 01 00 00       	callq  400ba3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a5d:	e8 fc fd ff ff       	callq  40085e <_ZN9TestClass7storageEv>
  400a62:	48 89 c2             	mov    %rax,%rdx
  400a65:	48 8b 02             	mov    (%rdx),%rax
  400a68:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a6c:	48 89 0a             	mov    %rcx,(%rdx)
  400a6f:	48 89 c7             	mov    %rax,%rdi
  400a72:	e8 33 01 00 00       	callq  400baa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a77:	c9                   	leaveq 
  400a78:	c3                   	retq   
  400a79:	90                   	nop

0000000000400a7a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a7a:	55                   	push   %rbp
  400a7b:	48 89 e5             	mov    %rsp,%rbp
  400a7e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a82:	90                   	nop
  400a83:	5d                   	pop    %rbp
  400a84:	c3                   	retq   
  400a85:	90                   	nop

0000000000400a86 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a86:	55                   	push   %rbp
  400a87:	48 89 e5             	mov    %rsp,%rbp
  400a8a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8e:	90                   	nop
  400a8f:	5d                   	pop    %rbp
  400a90:	c3                   	retq   

0000000000400a91 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a91:	55                   	push   %rbp
  400a92:	48 89 e5             	mov    %rsp,%rbp
  400a95:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a99:	90                   	nop
  400a9a:	5d                   	pop    %rbp
  400a9b:	c3                   	retq   

0000000000400a9c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a9c:	55                   	push   %rbp
  400a9d:	48 89 e5             	mov    %rsp,%rbp
  400aa0:	48 83 ec 20          	sub    $0x20,%rsp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aac:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400aaf:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ab3:	48 89 c7             	mov    %rax,%rdi
  400ab6:	e8 95 00 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400abb:	48 89 c2             	mov    %rax,%rdx
  400abe:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400ac1:	89 02                	mov    %eax,(%rdx)
  400ac3:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400ac7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400acb:	48 89 d6             	mov    %rdx,%rsi
  400ace:	48 89 c7             	mov    %rax,%rdi
  400ad1:	e8 94 00 00 00       	callq  400b6a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400ad6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ada:	c9                   	leaveq 
  400adb:	c3                   	retq   

0000000000400adc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400adc:	55                   	push   %rbp
  400add:	48 89 e5             	mov    %rsp,%rbp
  400ae0:	48 83 ec 20          	sub    $0x20,%rsp
  400ae4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aec:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400aef:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400af3:	48 89 c7             	mov    %rax,%rdi
  400af6:	e8 7f 00 00 00       	callq  400b7a <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400afb:	48 89 c2             	mov    %rax,%rdx
  400afe:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400b01:	89 02                	mov    %eax,(%rdx)
  400b03:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400b07:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b0b:	48 89 d6             	mov    %rdx,%rsi
  400b0e:	48 89 c7             	mov    %rax,%rdi
  400b11:	e8 7e 00 00 00       	callq  400b94 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400b16:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1a:	c9                   	leaveq 
  400b1b:	c3                   	retq   

0000000000400b1c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b1c:	55                   	push   %rbp
  400b1d:	48 89 e5             	mov    %rsp,%rbp
  400b20:	48 83 ec 10          	sub    $0x10,%rsp
  400b24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b28:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b2c:	48 89 c7             	mov    %rax,%rdi
  400b2f:	e8 1c 00 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b34:	c9                   	leaveq 
  400b35:	c3                   	retq   

0000000000400b36 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b36:	55                   	push   %rbp
  400b37:	48 89 e5             	mov    %rsp,%rbp
  400b3a:	48 83 ec 10          	sub    $0x10,%rsp
  400b3e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b42:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b46:	48 89 c7             	mov    %rax,%rdi
  400b49:	e8 2c 00 00 00       	callq  400b7a <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b4e:	c9                   	leaveq 
  400b4f:	c3                   	retq   

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 83 ec 10          	sub    $0x10,%rsp
  400b58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b5c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b60:	48 89 c7             	mov    %rax,%rdi
  400b63:	e8 60 00 00 00       	callq  400bc8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b68:	c9                   	leaveq 
  400b69:	c3                   	retq   

0000000000400b6a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b6a:	55                   	push   %rbp
  400b6b:	48 89 e5             	mov    %rsp,%rbp
  400b6e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b72:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b76:	90                   	nop
  400b77:	5d                   	pop    %rbp
  400b78:	c3                   	retq   
  400b79:	90                   	nop

0000000000400b7a <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b7a:	55                   	push   %rbp
  400b7b:	48 89 e5             	mov    %rsp,%rbp
  400b7e:	48 83 ec 10          	sub    $0x10,%rsp
  400b82:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b86:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b8a:	48 89 c7             	mov    %rax,%rdi
  400b8d:	e8 80 00 00 00       	callq  400c12 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b92:	c9                   	leaveq 
  400b93:	c3                   	retq   

0000000000400b94 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b94:	55                   	push   %rbp
  400b95:	48 89 e5             	mov    %rsp,%rbp
  400b98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b9c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ba0:	90                   	nop
  400ba1:	5d                   	pop    %rbp
  400ba2:	c3                   	retq   

0000000000400ba3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ba3:	55                   	push   %rbp
  400ba4:	48 89 e5             	mov    %rsp,%rbp
  400ba7:	90                   	nop
  400ba8:	5d                   	pop    %rbp
  400ba9:	c3                   	retq   

0000000000400baa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400baa:	55                   	push   %rbp
  400bab:	48 89 e5             	mov    %rsp,%rbp
  400bae:	48 83 ec 10          	sub    $0x10,%rsp
  400bb2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bba:	48 83 c0 01          	add    $0x1,%rax
  400bbe:	48 89 c7             	mov    %rax,%rdi
  400bc1:	e8 9c 00 00 00       	callq  400c62 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bc6:	c9                   	leaveq 
  400bc7:	c3                   	retq   

0000000000400bc8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bc8:	55                   	push   %rbp
  400bc9:	48 89 e5             	mov    %rsp,%rbp
  400bcc:	48 83 ec 30          	sub    $0x30,%rsp
  400bd0:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400bd4:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bd8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bdc:	e8 7d fc ff ff       	callq  40085e <_ZN9TestClass7storageEv>
  400be1:	48 8b 40 08          	mov    0x8(%rax),%rax
  400be5:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400be9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bed:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bf1:	48 83 e8 04          	sub    $0x4,%rax
  400bf5:	48 c1 e8 02          	shr    $0x2,%rax
  400bf9:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c00:	00 
  400c01:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c05:	48 01 d0             	add    %rdx,%rax
  400c08:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c0c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c10:	c9                   	leaveq 
  400c11:	c3                   	retq   

0000000000400c12 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c12:	55                   	push   %rbp
  400c13:	48 89 e5             	mov    %rsp,%rbp
  400c16:	48 83 ec 30          	sub    $0x30,%rsp
  400c1a:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400c1e:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400c22:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c26:	e8 33 fc ff ff       	callq  40085e <_ZN9TestClass7storageEv>
  400c2b:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c2f:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400c33:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c37:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400c3b:	48 83 e8 04          	sub    $0x4,%rax
  400c3f:	48 c1 e8 02          	shr    $0x2,%rax
  400c43:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c4a:	00 
  400c4b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c4f:	48 01 d0             	add    %rdx,%rax
  400c52:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c58:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c5c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c60:	c9                   	leaveq 
  400c61:	c3                   	retq   

0000000000400c62 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c62:	55                   	push   %rbp
  400c63:	48 89 e5             	mov    %rsp,%rbp
  400c66:	48 83 ec 20          	sub    $0x20,%rsp
  400c6a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c6e:	e8 eb fb ff ff       	callq  40085e <_ZN9TestClass7storageEv>
  400c73:	48 8b 40 08          	mov    0x8(%rax),%rax
  400c77:	48 89 c2             	mov    %rax,%rdx
  400c7a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c7e:	48 c1 e0 02          	shl    $0x2,%rax
  400c82:	48 01 d0             	add    %rdx,%rax
  400c85:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c89:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c8d:	c9                   	leaveq 
  400c8e:	c3                   	retq   
  400c8f:	90                   	nop

0000000000400c90 <__libc_csu_init>:
  400c90:	41 57                	push   %r15
  400c92:	41 56                	push   %r14
  400c94:	41 89 ff             	mov    %edi,%r15d
  400c97:	41 55                	push   %r13
  400c99:	41 54                	push   %r12
  400c9b:	4c 8d 25 6e 11 20 00 	lea    0x20116e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400ca2:	55                   	push   %rbp
  400ca3:	48 8d 2d 6e 11 20 00 	lea    0x20116e(%rip),%rbp        # 601e18 <__init_array_end>
  400caa:	53                   	push   %rbx
  400cab:	49 89 f6             	mov    %rsi,%r14
  400cae:	49 89 d5             	mov    %rdx,%r13
  400cb1:	4c 29 e5             	sub    %r12,%rbp
  400cb4:	48 83 ec 08          	sub    $0x8,%rsp
  400cb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cbc:	e8 9f f7 ff ff       	callq  400460 <_init>
  400cc1:	48 85 ed             	test   %rbp,%rbp
  400cc4:	74 20                	je     400ce6 <__libc_csu_init+0x56>
  400cc6:	31 db                	xor    %ebx,%ebx
  400cc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ccf:	00 
  400cd0:	4c 89 ea             	mov    %r13,%rdx
  400cd3:	4c 89 f6             	mov    %r14,%rsi
  400cd6:	44 89 ff             	mov    %r15d,%edi
  400cd9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400cdd:	48 83 c3 01          	add    $0x1,%rbx
  400ce1:	48 39 eb             	cmp    %rbp,%rbx
  400ce4:	75 ea                	jne    400cd0 <__libc_csu_init+0x40>
  400ce6:	48 83 c4 08          	add    $0x8,%rsp
  400cea:	5b                   	pop    %rbx
  400ceb:	5d                   	pop    %rbp
  400cec:	41 5c                	pop    %r12
  400cee:	41 5d                	pop    %r13
  400cf0:	41 5e                	pop    %r14
  400cf2:	41 5f                	pop    %r15
  400cf4:	c3                   	retq   
  400cf5:	90                   	nop
  400cf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfd:	00 00 00 

0000000000400d00 <__libc_csu_fini>:
  400d00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d04 <_fini>:
  400d04:	48 83 ec 08          	sub    $0x8,%rsp
  400d08:	48 83 c4 08          	add    $0x8,%rsp
  400d0c:	c3                   	retq   
