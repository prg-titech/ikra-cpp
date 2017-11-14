
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
  4004fd:	48 c7 c7 dc 05 40 00 	mov    $0x4005dc,%rdi
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
  4005d6:	b8 80 20 60 00       	mov    $0x602080,%eax
  4005db:	c3                   	retq   

00000000004005dc <main>:
  4005dc:	53                   	push   %rbx
  4005dd:	48 83 ec 10          	sub    $0x10,%rsp
  4005e1:	e8 f7 03 00 00       	callq  4009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005e6:	bf 04 00 00 00       	mov    $0x4,%edi
  4005eb:	e8 16 04 00 00       	callq  400a06 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005f0:	48 89 c3             	mov    %rax,%rbx
  4005f3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005f8:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005fc:	48 83 f8 03          	cmp    $0x3,%rax
  400600:	74 06                	je     400608 <main+0x2c>
  400602:	48 83 c0 01          	add    $0x1,%rax
  400606:	eb f0                	jmp    4005f8 <main+0x1c>
  400608:	48 89 df             	mov    %rbx,%rdi
  40060b:	e8 00 03 00 00       	callq  400910 <_ZN9TestClassC1Ev>
  400610:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  400615:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40061a:	48 83 c0 04          	add    $0x4,%rax
  40061e:	be 77 77 00 00       	mov    $0x7777,%esi
  400623:	48 89 c7             	mov    %rax,%rdi
  400626:	e8 1f 04 00 00       	callq  400a4a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40062b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400630:	48 83 c0 04          	add    $0x4,%rax
  400634:	be 88 88 00 00       	mov    $0x8888,%esi
  400639:	48 89 c7             	mov    %rax,%rdi
  40063c:	e8 37 04 00 00       	callq  400a78 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400641:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400646:	48 89 c7             	mov    %rax,%rdi
  400649:	e8 ce 01 00 00       	callq  40081c <_ZN9TestClass15increase_field0Ev>
  40064e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400653:	48 89 c7             	mov    %rax,%rdi
  400656:	e8 29 02 00 00       	callq  400884 <_ZN9TestClass15increase_field1Ev>
  40065b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400660:	48 83 c0 04          	add    $0x4,%rax
  400664:	48 89 c7             	mov    %rax,%rdi
  400667:	e8 56 04 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40066c:	8b 18                	mov    (%rax),%ebx
  40066e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400673:	48 83 c0 04          	add    $0x4,%rax
  400677:	48 89 c7             	mov    %rax,%rdi
  40067a:	e8 27 04 00 00       	callq  400aa6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40067f:	8b 00                	mov    (%rax),%eax
  400681:	89 da                	mov    %ebx,%edx
  400683:	89 c6                	mov    %eax,%esi
  400685:	bf 54 0d 40 00       	mov    $0x400d54,%edi
  40068a:	b8 00 00 00 00       	mov    $0x0,%eax
  40068f:	e8 0c fe ff ff       	callq  4004a0 <printf@plt>
  400694:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400699:	48 83 c0 04          	add    $0x4,%rax
  40069d:	48 89 c7             	mov    %rax,%rdi
  4006a0:	e8 01 04 00 00       	callq  400aa6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006a5:	8b 00                	mov    (%rax),%eax
  4006a7:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006ac:	75 1a                	jne    4006c8 <main+0xec>
  4006ae:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006b3:	48 83 c0 04          	add    $0x4,%rax
  4006b7:	48 89 c7             	mov    %rax,%rdi
  4006ba:	e8 03 04 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4006bf:	8b 00                	mov    (%rax),%eax
  4006c1:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006c6:	74 07                	je     4006cf <main+0xf3>
  4006c8:	b8 01 00 00 00       	mov    $0x1,%eax
  4006cd:	eb 05                	jmp    4006d4 <main+0xf8>
  4006cf:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d4:	0f b6 c0             	movzbl %al,%eax
  4006d7:	48 83 c4 10          	add    $0x10,%rsp
  4006db:	5b                   	pop    %rbx
  4006dc:	c3                   	retq   

00000000004006dd <_Z12new_instancev>:
  4006dd:	53                   	push   %rbx
  4006de:	bf 04 00 00 00       	mov    $0x4,%edi
  4006e3:	e8 1e 03 00 00       	callq  400a06 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4006e8:	48 89 c3             	mov    %rax,%rbx
  4006eb:	b8 00 00 00 00       	mov    $0x0,%eax
  4006f0:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006f4:	48 83 f8 03          	cmp    $0x3,%rax
  4006f8:	74 06                	je     400700 <_Z12new_instancev+0x23>
  4006fa:	48 83 c0 01          	add    $0x1,%rax
  4006fe:	eb f0                	jmp    4006f0 <_Z12new_instancev+0x13>
  400700:	48 89 df             	mov    %rbx,%rdi
  400703:	e8 08 02 00 00       	callq  400910 <_ZN9TestClassC1Ev>
  400708:	48 89 d8             	mov    %rbx,%rax
  40070b:	5b                   	pop    %rbx
  40070c:	c3                   	retq   

000000000040070d <_Z12write_field0P9TestClass>:
  40070d:	48 83 ec 18          	sub    $0x18,%rsp
  400711:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400716:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40071b:	48 83 c0 04          	add    $0x4,%rax
  40071f:	be 77 77 00 00       	mov    $0x7777,%esi
  400724:	48 89 c7             	mov    %rax,%rdi
  400727:	e8 1e 03 00 00       	callq  400a4a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40072c:	90                   	nop
  40072d:	48 83 c4 18          	add    $0x18,%rsp
  400731:	c3                   	retq   

0000000000400732 <_Z11read_field0P9TestClass>:
  400732:	48 83 ec 18          	sub    $0x18,%rsp
  400736:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40073b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400740:	48 83 c0 04          	add    $0x4,%rax
  400744:	48 89 c7             	mov    %rax,%rdi
  400747:	e8 5a 03 00 00       	callq  400aa6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40074c:	8b 00                	mov    (%rax),%eax
  40074e:	48 83 c4 18          	add    $0x18,%rsp
  400752:	c3                   	retq   

0000000000400753 <_Z12write_field1P9TestClass>:
  400753:	48 83 ec 18          	sub    $0x18,%rsp
  400757:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40075c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400761:	48 83 c0 04          	add    $0x4,%rax
  400765:	be 77 77 00 00       	mov    $0x7777,%esi
  40076a:	48 89 c7             	mov    %rax,%rdi
  40076d:	e8 06 03 00 00       	callq  400a78 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400772:	90                   	nop
  400773:	48 83 c4 18          	add    $0x18,%rsp
  400777:	c3                   	retq   

0000000000400778 <_Z11read_field1P9TestClass>:
  400778:	48 83 ec 18          	sub    $0x18,%rsp
  40077c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400781:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400786:	48 83 c0 04          	add    $0x4,%rax
  40078a:	48 89 c7             	mov    %rax,%rdi
  40078d:	e8 30 03 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400792:	8b 00                	mov    (%rax),%eax
  400794:	48 83 c4 18          	add    $0x18,%rsp
  400798:	c3                   	retq   

0000000000400799 <_Z21explicit_write_field0m>:
  400799:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40079e:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007a3:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  4007aa:	77 77 00 00 
  4007ae:	90                   	nop
  4007af:	c3                   	retq   

00000000004007b0 <_Z20explicit_read_field0m>:
  4007b0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007b5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007ba:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  4007c1:	c3                   	retq   

00000000004007c2 <_ZnwmPv>:
  4007c2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007c7:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4007cc:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  4007d1:	c3                   	retq   

00000000004007d2 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  4007d2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4007d7:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007dc:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007e3:	90                   	nop
  4007e4:	c3                   	retq   
  4007e5:	90                   	nop

00000000004007e6 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007e6:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4007eb:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  4007f0:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007f5:	48 8b 00             	mov    (%rax),%rax
  4007f8:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007fd:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400802:	48 8b 10             	mov    (%rax),%rdx
  400805:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  40080a:	48 01 c2             	add    %rax,%rdx
  40080d:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400812:	48 89 10             	mov    %rdx,(%rax)
  400815:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40081a:	c3                   	retq   
  40081b:	90                   	nop

000000000040081c <_ZN9TestClass15increase_field0Ev>:
  40081c:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400821:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400825:	41 ff 72 f8          	pushq  -0x8(%r10)
  400829:	55                   	push   %rbp
  40082a:	48 89 e5             	mov    %rsp,%rbp
  40082d:	41 52                	push   %r10
  40082f:	48 83 ec 48          	sub    $0x48,%rsp
  400833:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400837:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40083e:	00 00 
  400840:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400844:	31 c0                	xor    %eax,%eax
  400846:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40084a:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40084e:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400852:	ba 55 55 00 00       	mov    $0x5555,%edx
  400857:	48 89 ce             	mov    %rcx,%rsi
  40085a:	48 89 c7             	mov    %rax,%rdi
  40085d:	e8 ec 00 00 00       	callq  40094e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400862:	90                   	nop
  400863:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400867:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40086e:	00 00 
  400870:	74 05                	je     400877 <_ZN9TestClass15increase_field0Ev+0x5b>
  400872:	e8 19 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400877:	48 83 c4 48          	add    $0x48,%rsp
  40087b:	41 5a                	pop    %r10
  40087d:	5d                   	pop    %rbp
  40087e:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400882:	c3                   	retq   
  400883:	90                   	nop

0000000000400884 <_ZN9TestClass15increase_field1Ev>:
  400884:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400889:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40088d:	41 ff 72 f8          	pushq  -0x8(%r10)
  400891:	55                   	push   %rbp
  400892:	48 89 e5             	mov    %rsp,%rbp
  400895:	41 52                	push   %r10
  400897:	48 83 ec 48          	sub    $0x48,%rsp
  40089b:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  40089f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008a6:	00 00 
  4008a8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4008ac:	31 c0                	xor    %eax,%eax
  4008ae:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4008b2:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008b6:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4008ba:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008bf:	48 89 ce             	mov    %rcx,%rsi
  4008c2:	48 89 c7             	mov    %rax,%rdi
  4008c5:	e8 cc 00 00 00       	callq  400996 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008ca:	90                   	nop
  4008cb:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008cf:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008d6:	00 00 
  4008d8:	74 05                	je     4008df <_ZN9TestClass15increase_field1Ev+0x5b>
  4008da:	e8 b1 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008df:	48 83 c4 48          	add    $0x48,%rsp
  4008e3:	41 5a                	pop    %r10
  4008e5:	5d                   	pop    %rbp
  4008e6:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4008ea:	c3                   	retq   
  4008eb:	90                   	nop

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
  400932:	e8 fd 00 00 00       	callq  400a34 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400937:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40093c:	48 83 c0 04          	add    $0x4,%rax
  400940:	48 89 c7             	mov    %rax,%rdi
  400943:	e8 f4 00 00 00       	callq  400a3c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400948:	90                   	nop
  400949:	48 83 c4 18          	add    $0x18,%rsp
  40094d:	c3                   	retq   

000000000040094e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40094e:	48 83 ec 28          	sub    $0x28,%rsp
  400952:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400957:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  40095c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400960:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400965:	48 89 c7             	mov    %rax,%rdi
  400968:	e8 71 01 00 00       	callq  400ade <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40096d:	8b 10                	mov    (%rax),%edx
  40096f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400974:	89 10                	mov    %edx,(%rax)
  400976:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  40097b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400980:	48 89 d6             	mov    %rdx,%rsi
  400983:	48 89 c7             	mov    %rax,%rdi
  400986:	e8 6f 01 00 00       	callq  400afa <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  40098b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400990:	48 83 c4 28          	add    $0x28,%rsp
  400994:	c3                   	retq   
  400995:	90                   	nop

0000000000400996 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400996:	48 83 ec 28          	sub    $0x28,%rsp
  40099a:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40099f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4009a4:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4009a8:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4009ad:	48 89 c7             	mov    %rax,%rdi
  4009b0:	e8 51 01 00 00       	callq  400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009b5:	8b 10                	mov    (%rax),%edx
  4009b7:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  4009bc:	89 10                	mov    %edx,(%rax)
  4009be:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4009c3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009c8:	48 89 d6             	mov    %rdx,%rsi
  4009cb:	48 89 c7             	mov    %rax,%rdi
  4009ce:	e8 4f 01 00 00       	callq  400b22 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009d3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009d8:	48 83 c4 28          	add    $0x28,%rsp
  4009dc:	c3                   	retq   

00000000004009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  4009dd:	48 83 ec 08          	sub    $0x8,%rsp
  4009e1:	e8 f0 fb ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009e6:	48 89 c6             	mov    %rax,%rsi
  4009e9:	bf 20 00 00 00       	mov    $0x20,%edi
  4009ee:	e8 cf fd ff ff       	callq  4007c2 <_ZnwmPv>
  4009f3:	48 85 c0             	test   %rax,%rax
  4009f6:	74 08                	je     400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  4009f8:	48 89 c7             	mov    %rax,%rdi
  4009fb:	e8 4a 01 00 00       	callq  400b4a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400a00:	90                   	nop
  400a01:	48 83 c4 08          	add    $0x8,%rsp
  400a05:	c3                   	retq   

0000000000400a06 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a06:	48 83 ec 18          	sub    $0x18,%rsp
  400a0a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a0f:	e8 87 01 00 00       	callq  400b9b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a14:	e8 bd fb ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400a19:	be 01 00 00 00       	mov    $0x1,%esi
  400a1e:	48 89 c7             	mov    %rax,%rdi
  400a21:	e8 c0 fd ff ff       	callq  4007e6 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400a26:	48 89 c7             	mov    %rax,%rdi
  400a29:	e8 6f 01 00 00       	callq  400b9d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a2e:	48 83 c4 18          	add    $0x18,%rsp
  400a32:	c3                   	retq   
  400a33:	90                   	nop

0000000000400a34 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a34:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a39:	90                   	nop
  400a3a:	c3                   	retq   
  400a3b:	90                   	nop

0000000000400a3c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a3c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a41:	90                   	nop
  400a42:	c3                   	retq   

0000000000400a43 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a43:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a48:	90                   	nop
  400a49:	c3                   	retq   

0000000000400a4a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a4a:	48 83 ec 18          	sub    $0x18,%rsp
  400a4e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a53:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a57:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a5c:	48 89 c7             	mov    %rax,%rdi
  400a5f:	e8 7a 00 00 00       	callq  400ade <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a64:	48 89 c2             	mov    %rax,%rdx
  400a67:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a6b:	89 02                	mov    %eax,(%rdx)
  400a6d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a72:	48 83 c4 18          	add    $0x18,%rsp
  400a76:	c3                   	retq   
  400a77:	90                   	nop

0000000000400a78 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a78:	48 83 ec 18          	sub    $0x18,%rsp
  400a7c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a81:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a85:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a8a:	48 89 c7             	mov    %rax,%rdi
  400a8d:	e8 74 00 00 00       	callq  400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a92:	48 89 c2             	mov    %rax,%rdx
  400a95:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a99:	89 02                	mov    %eax,(%rdx)
  400a9b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aa0:	48 83 c4 18          	add    $0x18,%rsp
  400aa4:	c3                   	retq   
  400aa5:	90                   	nop

0000000000400aa6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400aa6:	48 83 ec 18          	sub    $0x18,%rsp
  400aaa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400aaf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ab4:	48 89 c7             	mov    %rax,%rdi
  400ab7:	e8 22 00 00 00       	callq  400ade <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400abc:	48 83 c4 18          	add    $0x18,%rsp
  400ac0:	c3                   	retq   
  400ac1:	90                   	nop

0000000000400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400ac2:	48 83 ec 18          	sub    $0x18,%rsp
  400ac6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400acb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ad0:	48 89 c7             	mov    %rax,%rdi
  400ad3:	e8 2e 00 00 00       	callq  400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad8:	48 83 c4 18          	add    $0x18,%rsp
  400adc:	c3                   	retq   
  400add:	90                   	nop

0000000000400ade <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ade:	48 83 ec 18          	sub    $0x18,%rsp
  400ae2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ae7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aec:	48 89 c7             	mov    %rax,%rdi
  400aef:	e8 c8 00 00 00       	callq  400bbc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400af4:	48 83 c4 18          	add    $0x18,%rsp
  400af8:	c3                   	retq   
  400af9:	90                   	nop

0000000000400afa <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400afa:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400aff:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b04:	90                   	nop
  400b05:	c3                   	retq   

0000000000400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b06:	48 83 ec 18          	sub    $0x18,%rsp
  400b0a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b0f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b14:	48 89 c7             	mov    %rax,%rdi
  400b17:	e8 f6 00 00 00       	callq  400c12 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b1c:	48 83 c4 18          	add    $0x18,%rsp
  400b20:	c3                   	retq   
  400b21:	90                   	nop

0000000000400b22 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b22:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b27:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b2c:	90                   	nop
  400b2d:	c3                   	retq   

0000000000400b2e <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400b2e:	48 83 ec 18          	sub    $0x18,%rsp
  400b32:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b37:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b3c:	48 89 c7             	mov    %rax,%rdi
  400b3f:	e8 8e fc ff ff       	callq  4007d2 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400b44:	90                   	nop
  400b45:	48 83 c4 18          	add    $0x18,%rsp
  400b49:	c3                   	retq   

0000000000400b4a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400b4a:	48 83 ec 18          	sub    $0x18,%rsp
  400b4e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b53:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b58:	48 89 c7             	mov    %rax,%rdi
  400b5b:	e8 ce ff ff ff       	callq  400b2e <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400b60:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400b65:	e8 56 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b6a:	48 89 c2             	mov    %rax,%rdx
  400b6d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b72:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400b76:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b7b:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b82:	00 
  400b83:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b88:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400b8c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b91:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b95:	90                   	nop
  400b96:	48 83 c4 18          	add    $0x18,%rsp
  400b9a:	c3                   	retq   

0000000000400b9b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b9b:	90                   	nop
  400b9c:	c3                   	retq   

0000000000400b9d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400b9d:	48 83 ec 18          	sub    $0x18,%rsp
  400ba1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ba6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bab:	48 83 c0 01          	add    $0x1,%rax
  400baf:	48 89 c7             	mov    %rax,%rdi
  400bb2:	e8 b7 00 00 00       	callq  400c6e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bb7:	48 83 c4 18          	add    $0x18,%rsp
  400bbb:	c3                   	retq   

0000000000400bbc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bbc:	48 83 ec 38          	sub    $0x38,%rsp
  400bc0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bc5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bca:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bcf:	e8 02 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400bd4:	48 89 c7             	mov    %rax,%rdi
  400bd7:	e8 c6 00 00 00       	callq  400ca2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bdc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400be1:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400be6:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400beb:	48 83 e8 04          	sub    $0x4,%rax
  400bef:	48 c1 e8 02          	shr    $0x2,%rax
  400bf3:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bfa:	00 
  400bfb:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400c00:	48 01 d0             	add    %rdx,%rax
  400c03:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400c08:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c0d:	48 83 c4 38          	add    $0x38,%rsp
  400c11:	c3                   	retq   

0000000000400c12 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c12:	48 83 ec 38          	sub    $0x38,%rsp
  400c16:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c1b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c20:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c25:	e8 ac f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400c2a:	48 89 c7             	mov    %rax,%rdi
  400c2d:	e8 70 00 00 00       	callq  400ca2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c32:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400c37:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c3c:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400c41:	48 83 e8 04          	sub    $0x4,%rax
  400c45:	48 c1 e8 02          	shr    $0x2,%rax
  400c49:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c50:	00 
  400c51:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400c56:	48 01 d0             	add    %rdx,%rax
  400c59:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c5f:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400c64:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c69:	48 83 c4 38          	add    $0x38,%rsp
  400c6d:	c3                   	retq   

0000000000400c6e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c6e:	48 83 ec 28          	sub    $0x28,%rsp
  400c72:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c77:	e8 5a f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400c7c:	48 89 c7             	mov    %rax,%rdi
  400c7f:	e8 1e 00 00 00       	callq  400ca2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c84:	48 89 c2             	mov    %rax,%rdx
  400c87:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c8c:	48 c1 e0 02          	shl    $0x2,%rax
  400c90:	48 01 d0             	add    %rdx,%rax
  400c93:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c98:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c9d:	48 83 c4 28          	add    $0x28,%rsp
  400ca1:	c3                   	retq   

0000000000400ca2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400ca2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ca7:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400cac:	48 8b 40 18          	mov    0x18(%rax),%rax
  400cb0:	c3                   	retq   
  400cb1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cb8:	00 00 00 
  400cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

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
