
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
  4004ef:	49 c7 c0 30 0d 40 00 	mov    $0x400d30,%r8
  4004f6:	48 c7 c1 c0 0c 40 00 	mov    $0x400cc0,%rcx
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
  400590:	80 3d c9 1a 20 00 00 	cmpb   $0x0,0x201ac9(%rip)        # 602060 <completed.7594>
  400597:	75 11                	jne    4005aa <__do_global_dtors_aux+0x1a>
  400599:	55                   	push   %rbp
  40059a:	48 89 e5             	mov    %rsp,%rbp
  40059d:	e8 6e ff ff ff       	callq  400510 <deregister_tm_clones>
  4005a2:	5d                   	pop    %rbp
  4005a3:	c6 05 b6 1a 20 00 01 	movb   $0x1,0x201ab6(%rip)        # 602060 <completed.7594>
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
  4005d6:	53                   	push   %rbx
  4005d7:	48 83 ec 10          	sub    $0x10,%rsp
  4005db:	e8 0b 04 00 00       	callq  4009eb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005e0:	bf 04 00 00 00       	mov    $0x4,%edi
  4005e5:	e8 2a 04 00 00       	callq  400a14 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005ea:	48 89 c3             	mov    %rax,%rbx
  4005ed:	b8 00 00 00 00       	mov    $0x0,%eax
  4005f2:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005f6:	48 83 f8 03          	cmp    $0x3,%rax
  4005fa:	74 06                	je     400602 <main+0x2c>
  4005fc:	48 83 c0 01          	add    $0x1,%rax
  400600:	eb f0                	jmp    4005f2 <main+0x1c>
  400602:	48 89 df             	mov    %rbx,%rdi
  400605:	e8 06 03 00 00       	callq  400910 <_ZN9TestClassC1Ev>
  40060a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40060f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400614:	48 83 c0 04          	add    $0x4,%rax
  400618:	be 77 77 00 00       	mov    $0x7777,%esi
  40061d:	48 89 c7             	mov    %rax,%rdi
  400620:	e8 33 04 00 00       	callq  400a58 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  400625:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40062a:	48 83 c0 04          	add    $0x4,%rax
  40062e:	be 88 88 00 00       	mov    $0x8888,%esi
  400633:	48 89 c7             	mov    %rax,%rdi
  400636:	e8 4b 04 00 00       	callq  400a86 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40063b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400640:	48 89 c7             	mov    %rax,%rdi
  400643:	e8 ce 01 00 00       	callq  400816 <_ZN9TestClass15increase_field0Ev>
  400648:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40064d:	48 89 c7             	mov    %rax,%rdi
  400650:	e8 29 02 00 00       	callq  40087e <_ZN9TestClass15increase_field1Ev>
  400655:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40065a:	48 83 c0 04          	add    $0x4,%rax
  40065e:	48 89 c7             	mov    %rax,%rdi
  400661:	e8 6a 04 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  400666:	8b 18                	mov    (%rax),%ebx
  400668:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40066d:	48 83 c0 04          	add    $0x4,%rax
  400671:	48 89 c7             	mov    %rax,%rdi
  400674:	e8 3b 04 00 00       	callq  400ab4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400679:	8b 00                	mov    (%rax),%eax
  40067b:	89 da                	mov    %ebx,%edx
  40067d:	89 c6                	mov    %eax,%esi
  40067f:	bf 5c 0d 40 00       	mov    $0x400d5c,%edi
  400684:	b8 00 00 00 00       	mov    $0x0,%eax
  400689:	e8 12 fe ff ff       	callq  4004a0 <printf@plt>
  40068e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400693:	48 83 c0 04          	add    $0x4,%rax
  400697:	48 89 c7             	mov    %rax,%rdi
  40069a:	e8 15 04 00 00       	callq  400ab4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  40069f:	8b 00                	mov    (%rax),%eax
  4006a1:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006a6:	75 1a                	jne    4006c2 <main+0xec>
  4006a8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006ad:	48 83 c0 04          	add    $0x4,%rax
  4006b1:	48 89 c7             	mov    %rax,%rdi
  4006b4:	e8 17 04 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  4006b9:	8b 00                	mov    (%rax),%eax
  4006bb:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006c0:	74 07                	je     4006c9 <main+0xf3>
  4006c2:	b8 01 00 00 00       	mov    $0x1,%eax
  4006c7:	eb 05                	jmp    4006ce <main+0xf8>
  4006c9:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ce:	0f b6 c0             	movzbl %al,%eax
  4006d1:	48 83 c4 10          	add    $0x10,%rsp
  4006d5:	5b                   	pop    %rbx
  4006d6:	c3                   	retq   

00000000004006d7 <_Z12new_instancev>:
  4006d7:	53                   	push   %rbx
  4006d8:	bf 04 00 00 00       	mov    $0x4,%edi
  4006dd:	e8 32 03 00 00       	callq  400a14 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4006e2:	48 89 c3             	mov    %rax,%rbx
  4006e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ea:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006ee:	48 83 f8 03          	cmp    $0x3,%rax
  4006f2:	74 06                	je     4006fa <_Z12new_instancev+0x23>
  4006f4:	48 83 c0 01          	add    $0x1,%rax
  4006f8:	eb f0                	jmp    4006ea <_Z12new_instancev+0x13>
  4006fa:	48 89 df             	mov    %rbx,%rdi
  4006fd:	e8 0e 02 00 00       	callq  400910 <_ZN9TestClassC1Ev>
  400702:	48 89 d8             	mov    %rbx,%rax
  400705:	5b                   	pop    %rbx
  400706:	c3                   	retq   

0000000000400707 <_Z12write_field0P9TestClass>:
  400707:	48 83 ec 18          	sub    $0x18,%rsp
  40070b:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400710:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400715:	48 83 c0 04          	add    $0x4,%rax
  400719:	be 77 77 00 00       	mov    $0x7777,%esi
  40071e:	48 89 c7             	mov    %rax,%rdi
  400721:	e8 32 03 00 00       	callq  400a58 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  400726:	90                   	nop
  400727:	48 83 c4 18          	add    $0x18,%rsp
  40072b:	c3                   	retq   

000000000040072c <_Z11read_field0P9TestClass>:
  40072c:	48 83 ec 18          	sub    $0x18,%rsp
  400730:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400735:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40073a:	48 83 c0 04          	add    $0x4,%rax
  40073e:	48 89 c7             	mov    %rax,%rdi
  400741:	e8 6e 03 00 00       	callq  400ab4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400746:	8b 00                	mov    (%rax),%eax
  400748:	48 83 c4 18          	add    $0x18,%rsp
  40074c:	c3                   	retq   

000000000040074d <_Z12write_field1P9TestClass>:
  40074d:	48 83 ec 18          	sub    $0x18,%rsp
  400751:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400756:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40075b:	48 83 c0 04          	add    $0x4,%rax
  40075f:	be 77 77 00 00       	mov    $0x7777,%esi
  400764:	48 89 c7             	mov    %rax,%rdi
  400767:	e8 1a 03 00 00       	callq  400a86 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40076c:	90                   	nop
  40076d:	48 83 c4 18          	add    $0x18,%rsp
  400771:	c3                   	retq   

0000000000400772 <_Z11read_field1P9TestClass>:
  400772:	48 83 ec 18          	sub    $0x18,%rsp
  400776:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40077b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400780:	48 83 c0 04          	add    $0x4,%rax
  400784:	48 89 c7             	mov    %rax,%rdi
  400787:	e8 44 03 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  40078c:	8b 00                	mov    (%rax),%eax
  40078e:	48 83 c4 18          	add    $0x18,%rsp
  400792:	c3                   	retq   

0000000000400793 <_Z21explicit_write_field0m>:
  400793:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400798:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40079d:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  4007a4:	77 77 00 00 
  4007a8:	90                   	nop
  4007a9:	c3                   	retq   

00000000004007aa <_Z20explicit_read_field0m>:
  4007aa:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007af:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007b4:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  4007bb:	c3                   	retq   

00000000004007bc <_ZnwmPv>:
  4007bc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007c1:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4007c6:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  4007cb:	c3                   	retq   

00000000004007cc <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  4007cc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007d1:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007d6:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007dd:	90                   	nop
  4007de:	c3                   	retq   
  4007df:	90                   	nop

00000000004007e0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007e0:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4007e5:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  4007ea:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007ef:	48 8b 00             	mov    (%rax),%rax
  4007f2:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007f7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007fc:	48 8b 10             	mov    (%rax),%rdx
  4007ff:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400804:	48 01 c2             	add    %rax,%rdx
  400807:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  40080c:	48 89 10             	mov    %rdx,(%rax)
  40080f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400814:	c3                   	retq   
  400815:	90                   	nop

0000000000400816 <_ZN9TestClass15increase_field0Ev>:
  400816:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40081b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40081f:	41 ff 72 f8          	pushq  -0x8(%r10)
  400823:	55                   	push   %rbp
  400824:	48 89 e5             	mov    %rsp,%rbp
  400827:	41 52                	push   %r10
  400829:	48 83 ec 48          	sub    $0x48,%rsp
  40082d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400831:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400838:	00 00 
  40083a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40083e:	31 c0                	xor    %eax,%eax
  400840:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400844:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400848:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40084c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400851:	48 89 ce             	mov    %rcx,%rsi
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	e8 f2 00 00 00       	callq  40094e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>
  40085c:	90                   	nop
  40085d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400861:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400868:	00 00 
  40086a:	74 05                	je     400871 <_ZN9TestClass15increase_field0Ev+0x5b>
  40086c:	e8 1f fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400871:	48 83 c4 48          	add    $0x48,%rsp
  400875:	41 5a                	pop    %r10
  400877:	5d                   	pop    %rbp
  400878:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  40087c:	c3                   	retq   
  40087d:	90                   	nop

000000000040087e <_ZN9TestClass15increase_field1Ev>:
  40087e:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400883:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400887:	41 ff 72 f8          	pushq  -0x8(%r10)
  40088b:	55                   	push   %rbp
  40088c:	48 89 e5             	mov    %rsp,%rbp
  40088f:	41 52                	push   %r10
  400891:	48 83 ec 48          	sub    $0x48,%rsp
  400895:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400899:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008a0:	00 00 
  4008a2:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4008a6:	31 c0                	xor    %eax,%eax
  4008a8:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4008ac:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008b0:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4008b4:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008b9:	48 89 ce             	mov    %rcx,%rsi
  4008bc:	48 89 c7             	mov    %rax,%rdi
  4008bf:	e8 d2 00 00 00       	callq  400996 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>
  4008c4:	90                   	nop
  4008c5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008c9:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008d0:	00 00 
  4008d2:	74 05                	je     4008d9 <_ZN9TestClass15increase_field1Ev+0x5b>
  4008d4:	e8 b7 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008d9:	48 83 c4 48          	add    $0x48,%rsp
  4008dd:	41 5a                	pop    %r10
  4008df:	5d                   	pop    %rbp
  4008e0:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4008e4:	c3                   	retq   
  4008e5:	90                   	nop

00000000004008e6 <_ZN9TestClass14storage_bufferEv>:
  4008e6:	b8 80 20 60 00       	mov    $0x602080,%eax
  4008eb:	c3                   	retq   

00000000004008ec <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008ec:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008f1:	90                   	nop
  4008f2:	c3                   	retq   
  4008f3:	90                   	nop

00000000004008f4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  4008f4:	48 83 ec 18          	sub    $0x18,%rsp
  4008f8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008fd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400902:	48 89 c7             	mov    %rax,%rdi
  400905:	e8 e2 ff ff ff       	callq  4008ec <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40090a:	90                   	nop
  40090b:	48 83 c4 18          	add    $0x18,%rsp
  40090f:	c3                   	retq   

0000000000400910 <_ZN9TestClassC1Ev>:
  400910:	48 83 ec 18          	sub    $0x18,%rsp
  400914:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400919:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 ce ff ff ff       	callq  4008f4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400926:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40092b:	48 83 c0 04          	add    $0x4,%rax
  40092f:	48 89 c7             	mov    %rax,%rdi
  400932:	e8 0b 01 00 00       	callq  400a42 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC1Ev>
  400937:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40093c:	48 83 c0 04          	add    $0x4,%rax
  400940:	48 89 c7             	mov    %rax,%rdi
  400943:	e8 02 01 00 00       	callq  400a4a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC1Ev>
  400948:	90                   	nop
  400949:	48 83 c4 18          	add    $0x18,%rsp
  40094d:	c3                   	retq   

000000000040094e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>:
  40094e:	48 83 ec 28          	sub    $0x28,%rsp
  400952:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400957:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  40095c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400960:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400965:	48 89 c7             	mov    %rax,%rdi
  400968:	e8 7f 01 00 00       	callq  400aec <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40096d:	8b 10                	mov    (%rax),%edx
  40096f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400974:	89 10                	mov    %edx,(%rax)
  400976:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  40097b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400980:	48 89 d6             	mov    %rdx,%rsi
  400983:	48 89 c7             	mov    %rax,%rdi
  400986:	e8 7d 01 00 00       	callq  400b08 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC1ERKS3_>
  40098b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400990:	48 83 c4 28          	add    $0x28,%rsp
  400994:	c3                   	retq   
  400995:	90                   	nop

0000000000400996 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>:
  400996:	48 83 ec 28          	sub    $0x28,%rsp
  40099a:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40099f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4009a4:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4009a8:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4009ad:	48 89 c7             	mov    %rax,%rdi
  4009b0:	e8 5f 01 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009b5:	8b 10                	mov    (%rax),%edx
  4009b7:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  4009bc:	89 10                	mov    %edx,(%rax)
  4009be:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4009c3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009c8:	48 89 d6             	mov    %rdx,%rsi
  4009cb:	48 89 c7             	mov    %rax,%rdi
  4009ce:	e8 5d 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC1ERKS3_>
  4009d3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009d8:	48 83 c4 28          	add    $0x28,%rsp
  4009dc:	c3                   	retq   

00000000004009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>:
  4009dd:	48 83 ec 08          	sub    $0x8,%rsp
  4009e1:	e8 00 ff ff ff       	callq  4008e6 <_ZN9TestClass14storage_bufferEv>
  4009e6:	48 83 c4 08          	add    $0x8,%rsp
  4009ea:	c3                   	retq   

00000000004009eb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  4009eb:	48 83 ec 08          	sub    $0x8,%rsp
  4009ef:	e8 e9 ff ff ff       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  4009f4:	48 89 c6             	mov    %rax,%rsi
  4009f7:	bf 20 00 00 00       	mov    $0x20,%edi
  4009fc:	e8 bb fd ff ff       	callq  4007bc <_ZnwmPv>
  400a01:	48 85 c0             	test   %rax,%rax
  400a04:	74 08                	je     400a0e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400a06:	48 89 c7             	mov    %rax,%rdi
  400a09:	e8 4a 01 00 00       	callq  400b58 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400a0e:	90                   	nop
  400a0f:	48 83 c4 08          	add    $0x8,%rsp
  400a13:	c3                   	retq   

0000000000400a14 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a14:	48 83 ec 18          	sub    $0x18,%rsp
  400a18:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a1d:	e8 87 01 00 00       	callq  400ba9 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a22:	e8 b6 ff ff ff       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400a27:	be 01 00 00 00       	mov    $0x1,%esi
  400a2c:	48 89 c7             	mov    %rax,%rdi
  400a2f:	e8 ac fd ff ff       	callq  4007e0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400a34:	48 89 c7             	mov    %rax,%rdi
  400a37:	e8 6f 01 00 00       	callq  400bab <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a3c:	48 83 c4 18          	add    $0x18,%rsp
  400a40:	c3                   	retq   
  400a41:	90                   	nop

0000000000400a42 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC1Ev>:
  400a42:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a47:	90                   	nop
  400a48:	c3                   	retq   
  400a49:	90                   	nop

0000000000400a4a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC1Ev>:
  400a4a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a4f:	90                   	nop
  400a50:	c3                   	retq   

0000000000400a51 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a51:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a56:	90                   	nop
  400a57:	c3                   	retq   

0000000000400a58 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>:
  400a58:	48 83 ec 18          	sub    $0x18,%rsp
  400a5c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a61:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a65:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a6a:	48 89 c7             	mov    %rax,%rdi
  400a6d:	e8 7a 00 00 00       	callq  400aec <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a72:	48 89 c2             	mov    %rax,%rdx
  400a75:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a79:	89 02                	mov    %eax,(%rdx)
  400a7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a80:	48 83 c4 18          	add    $0x18,%rsp
  400a84:	c3                   	retq   
  400a85:	90                   	nop

0000000000400a86 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>:
  400a86:	48 83 ec 18          	sub    $0x18,%rsp
  400a8a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a8f:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a93:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a98:	48 89 c7             	mov    %rax,%rdi
  400a9b:	e8 74 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa0:	48 89 c2             	mov    %rax,%rdx
  400aa3:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400aa7:	89 02                	mov    %eax,(%rdx)
  400aa9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aae:	48 83 c4 18          	add    $0x18,%rsp
  400ab2:	c3                   	retq   
  400ab3:	90                   	nop

0000000000400ab4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>:
  400ab4:	48 83 ec 18          	sub    $0x18,%rsp
  400ab8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400abd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ac2:	48 89 c7             	mov    %rax,%rdi
  400ac5:	e8 22 00 00 00       	callq  400aec <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aca:	48 83 c4 18          	add    $0x18,%rsp
  400ace:	c3                   	retq   
  400acf:	90                   	nop

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>:
  400ad0:	48 83 ec 18          	sub    $0x18,%rsp
  400ad4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ad9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ade:	48 89 c7             	mov    %rax,%rdi
  400ae1:	e8 2e 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae6:	48 83 c4 18          	add    $0x18,%rsp
  400aea:	c3                   	retq   
  400aeb:	90                   	nop

0000000000400aec <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aec:	48 83 ec 18          	sub    $0x18,%rsp
  400af0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400af5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400afa:	48 89 c7             	mov    %rax,%rdi
  400afd:	e8 c8 00 00 00       	callq  400bca <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b02:	48 83 c4 18          	add    $0x18,%rsp
  400b06:	c3                   	retq   
  400b07:	90                   	nop

0000000000400b08 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC1ERKS3_>:
  400b08:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b0d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b12:	90                   	nop
  400b13:	c3                   	retq   

0000000000400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b14:	48 83 ec 18          	sub    $0x18,%rsp
  400b18:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b1d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b22:	48 89 c7             	mov    %rax,%rdi
  400b25:	e8 f6 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2a:	48 83 c4 18          	add    $0x18,%rsp
  400b2e:	c3                   	retq   
  400b2f:	90                   	nop

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC1ERKS3_>:
  400b30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b3a:	90                   	nop
  400b3b:	c3                   	retq   

0000000000400b3c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400b3c:	48 83 ec 18          	sub    $0x18,%rsp
  400b40:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b45:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b4a:	48 89 c7             	mov    %rax,%rdi
  400b4d:	e8 7a fc ff ff       	callq  4007cc <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400b52:	90                   	nop
  400b53:	48 83 c4 18          	add    $0x18,%rsp
  400b57:	c3                   	retq   

0000000000400b58 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400b58:	48 83 ec 18          	sub    $0x18,%rsp
  400b5c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b61:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b66:	48 89 c7             	mov    %rax,%rdi
  400b69:	e8 ce ff ff ff       	callq  400b3c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400b6e:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400b73:	e8 48 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b78:	48 89 c2             	mov    %rax,%rdx
  400b7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b80:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400b84:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b89:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b90:	00 
  400b91:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b96:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400b9a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b9f:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400ba3:	90                   	nop
  400ba4:	48 83 c4 18          	add    $0x18,%rsp
  400ba8:	c3                   	retq   

0000000000400ba9 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ba9:	90                   	nop
  400baa:	c3                   	retq   

0000000000400bab <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400bab:	48 83 ec 18          	sub    $0x18,%rsp
  400baf:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bb4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bb9:	48 83 c0 01          	add    $0x1,%rax
  400bbd:	48 89 c7             	mov    %rax,%rdi
  400bc0:	e8 b7 00 00 00       	callq  400c7c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bc5:	48 83 c4 18          	add    $0x18,%rsp
  400bc9:	c3                   	retq   

0000000000400bca <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bca:	48 83 ec 38          	sub    $0x38,%rsp
  400bce:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bd3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bd8:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bdd:	e8 fb fd ff ff       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400be2:	48 89 c7             	mov    %rax,%rdi
  400be5:	e8 c6 00 00 00       	callq  400cb0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bea:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400bef:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bf4:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400bf9:	48 83 e8 04          	sub    $0x4,%rax
  400bfd:	48 c1 e8 02          	shr    $0x2,%rax
  400c01:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c08:	00 
  400c09:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400c0e:	48 01 d0             	add    %rdx,%rax
  400c11:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400c16:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c1b:	48 83 c4 38          	add    $0x38,%rsp
  400c1f:	c3                   	retq   

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c20:	48 83 ec 38          	sub    $0x38,%rsp
  400c24:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c29:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c2e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c33:	e8 a5 fd ff ff       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400c38:	48 89 c7             	mov    %rax,%rdi
  400c3b:	e8 70 00 00 00       	callq  400cb0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c40:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400c45:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c4a:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400c4f:	48 83 e8 04          	sub    $0x4,%rax
  400c53:	48 c1 e8 02          	shr    $0x2,%rax
  400c57:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c5e:	00 
  400c5f:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400c64:	48 01 d0             	add    %rdx,%rax
  400c67:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c6d:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400c72:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c77:	48 83 c4 38          	add    $0x38,%rsp
  400c7b:	c3                   	retq   

0000000000400c7c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c7c:	48 83 ec 28          	sub    $0x28,%rsp
  400c80:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c85:	e8 53 fd ff ff       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400c8a:	48 89 c7             	mov    %rax,%rdi
  400c8d:	e8 1e 00 00 00       	callq  400cb0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c92:	48 89 c2             	mov    %rax,%rdx
  400c95:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c9a:	48 c1 e0 02          	shl    $0x2,%rax
  400c9e:	48 01 d0             	add    %rdx,%rax
  400ca1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400ca6:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cab:	48 83 c4 28          	add    $0x28,%rsp
  400caf:	c3                   	retq   

0000000000400cb0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400cb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cb5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400cba:	48 8b 40 18          	mov    0x18(%rax),%rax
  400cbe:	c3                   	retq   
  400cbf:	90                   	nop

0000000000400cc0 <__libc_csu_init>:
  400cc0:	41 57                	push   %r15
  400cc2:	41 56                	push   %r14
  400cc4:	41 89 ff             	mov    %edi,%r15d
  400cc7:	41 55                	push   %r13
  400cc9:	41 54                	push   %r12
  400ccb:	4c 8d 25 3e 11 20 00 	lea    0x20113e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400cd2:	55                   	push   %rbp
  400cd3:	48 8d 2d 3e 11 20 00 	lea    0x20113e(%rip),%rbp        # 601e18 <__init_array_end>
  400cda:	53                   	push   %rbx
  400cdb:	49 89 f6             	mov    %rsi,%r14
  400cde:	49 89 d5             	mov    %rdx,%r13
  400ce1:	4c 29 e5             	sub    %r12,%rbp
  400ce4:	48 83 ec 08          	sub    $0x8,%rsp
  400ce8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cec:	e8 6f f7 ff ff       	callq  400460 <_init>
  400cf1:	48 85 ed             	test   %rbp,%rbp
  400cf4:	74 20                	je     400d16 <__libc_csu_init+0x56>
  400cf6:	31 db                	xor    %ebx,%ebx
  400cf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400cff:	00 
  400d00:	4c 89 ea             	mov    %r13,%rdx
  400d03:	4c 89 f6             	mov    %r14,%rsi
  400d06:	44 89 ff             	mov    %r15d,%edi
  400d09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d0d:	48 83 c3 01          	add    $0x1,%rbx
  400d11:	48 39 eb             	cmp    %rbp,%rbx
  400d14:	75 ea                	jne    400d00 <__libc_csu_init+0x40>
  400d16:	48 83 c4 08          	add    $0x8,%rsp
  400d1a:	5b                   	pop    %rbx
  400d1b:	5d                   	pop    %rbp
  400d1c:	41 5c                	pop    %r12
  400d1e:	41 5d                	pop    %r13
  400d20:	41 5e                	pop    %r14
  400d22:	41 5f                	pop    %r15
  400d24:	c3                   	retq   
  400d25:	90                   	nop
  400d26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d2d:	00 00 00 

0000000000400d30 <__libc_csu_fini>:
  400d30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d34 <_fini>:
  400d34:	48 83 ec 08          	sub    $0x8,%rsp
  400d38:	48 83 c4 08          	add    $0x8,%rsp
  400d3c:	c3                   	retq   
