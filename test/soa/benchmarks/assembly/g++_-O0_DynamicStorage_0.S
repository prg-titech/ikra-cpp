
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
  4004ef:	49 c7 c0 80 0c 40 00 	mov    $0x400c80,%r8
  4004f6:	48 c7 c1 10 0c 40 00 	mov    $0x400c10,%rcx
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
  4005d6:	53                   	push   %rbx
  4005d7:	48 83 ec 10          	sub    $0x10,%rsp
  4005db:	e8 a9 03 00 00       	callq  400989 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005e0:	bf 00 00 00 00       	mov    $0x0,%edi
  4005e5:	e8 c8 03 00 00       	callq  4009b2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4005ea:	48 89 c3             	mov    %rax,%rbx
  4005ed:	48 89 df             	mov    %rbx,%rdi
  4005f0:	e8 c1 02 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
  4005f5:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4005fa:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005ff:	be 77 77 00 00       	mov    $0x7777,%esi
  400604:	48 89 c7             	mov    %rax,%rdi
  400607:	e8 ea 03 00 00       	callq  4009f6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>
  40060c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400611:	be 88 88 00 00       	mov    $0x8888,%esi
  400616:	48 89 c7             	mov    %rax,%rdi
  400619:	e8 06 04 00 00       	callq  400a24 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>
  40061e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400623:	48 89 c7             	mov    %rax,%rdi
  400626:	e8 99 01 00 00       	callq  4007c4 <_ZN9TestClass15increase_field0Ev>
  40062b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400630:	48 89 c7             	mov    %rax,%rdi
  400633:	e8 f0 01 00 00       	callq  400828 <_ZN9TestClass15increase_field1Ev>
  400638:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40063d:	48 89 c7             	mov    %rax,%rdi
  400640:	e8 29 04 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  400645:	8b 18                	mov    (%rax),%ebx
  400647:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40064c:	48 89 c7             	mov    %rax,%rdi
  40064f:	e8 fe 03 00 00       	callq  400a52 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  400654:	8b 00                	mov    (%rax),%eax
  400656:	89 da                	mov    %ebx,%edx
  400658:	89 c6                	mov    %eax,%esi
  40065a:	bf ac 0c 40 00       	mov    $0x400cac,%edi
  40065f:	b8 00 00 00 00       	mov    $0x0,%eax
  400664:	e8 37 fe ff ff       	callq  4004a0 <printf@plt>
  400669:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40066e:	48 89 c7             	mov    %rax,%rdi
  400671:	e8 dc 03 00 00       	callq  400a52 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  400676:	8b 00                	mov    (%rax),%eax
  400678:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40067d:	75 16                	jne    400695 <main+0xbf>
  40067f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400684:	48 89 c7             	mov    %rax,%rdi
  400687:	e8 e2 03 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  40068c:	8b 00                	mov    (%rax),%eax
  40068e:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400693:	74 07                	je     40069c <main+0xc6>
  400695:	b8 01 00 00 00       	mov    $0x1,%eax
  40069a:	eb 05                	jmp    4006a1 <main+0xcb>
  40069c:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a1:	0f b6 c0             	movzbl %al,%eax
  4006a4:	48 83 c4 10          	add    $0x10,%rsp
  4006a8:	5b                   	pop    %rbx
  4006a9:	c3                   	retq   

00000000004006aa <_Z12new_instancev>:
  4006aa:	53                   	push   %rbx
  4006ab:	bf 00 00 00 00       	mov    $0x0,%edi
  4006b0:	e8 fd 02 00 00       	callq  4009b2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006b5:	48 89 c3             	mov    %rax,%rbx
  4006b8:	48 89 df             	mov    %rbx,%rdi
  4006bb:	e8 f6 01 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
  4006c0:	48 89 d8             	mov    %rbx,%rax
  4006c3:	5b                   	pop    %rbx
  4006c4:	c3                   	retq   

00000000004006c5 <_Z12write_field0P9TestClass>:
  4006c5:	48 83 ec 18          	sub    $0x18,%rsp
  4006c9:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006ce:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006d3:	be 77 77 00 00       	mov    $0x7777,%esi
  4006d8:	48 89 c7             	mov    %rax,%rdi
  4006db:	e8 16 03 00 00       	callq  4009f6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>
  4006e0:	90                   	nop
  4006e1:	48 83 c4 18          	add    $0x18,%rsp
  4006e5:	c3                   	retq   

00000000004006e6 <_Z11read_field0P9TestClass>:
  4006e6:	48 83 ec 18          	sub    $0x18,%rsp
  4006ea:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006ef:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006f4:	48 89 c7             	mov    %rax,%rdi
  4006f7:	e8 56 03 00 00       	callq  400a52 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  4006fc:	8b 00                	mov    (%rax),%eax
  4006fe:	48 83 c4 18          	add    $0x18,%rsp
  400702:	c3                   	retq   

0000000000400703 <_Z12write_field1P9TestClass>:
  400703:	48 83 ec 18          	sub    $0x18,%rsp
  400707:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40070c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400711:	be 77 77 00 00       	mov    $0x7777,%esi
  400716:	48 89 c7             	mov    %rax,%rdi
  400719:	e8 06 03 00 00       	callq  400a24 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>
  40071e:	90                   	nop
  40071f:	48 83 c4 18          	add    $0x18,%rsp
  400723:	c3                   	retq   

0000000000400724 <_Z11read_field1P9TestClass>:
  400724:	48 83 ec 18          	sub    $0x18,%rsp
  400728:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40072d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400732:	48 89 c7             	mov    %rax,%rdi
  400735:	e8 34 03 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  40073a:	8b 00                	mov    (%rax),%eax
  40073c:	48 83 c4 18          	add    $0x18,%rsp
  400740:	c3                   	retq   

0000000000400741 <_Z21explicit_write_field0m>:
  400741:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400746:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40074b:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  400752:	77 77 00 00 
  400756:	90                   	nop
  400757:	c3                   	retq   

0000000000400758 <_Z20explicit_read_field0m>:
  400758:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40075d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400762:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  400769:	c3                   	retq   

000000000040076a <_ZnwmPv>:
  40076a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40076f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400774:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400779:	c3                   	retq   

000000000040077a <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  40077a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40077f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400784:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  40078b:	90                   	nop
  40078c:	c3                   	retq   
  40078d:	90                   	nop

000000000040078e <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  40078e:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400793:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400798:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  40079d:	48 8b 00             	mov    (%rax),%rax
  4007a0:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007a5:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007aa:	48 8b 10             	mov    (%rax),%rdx
  4007ad:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  4007b2:	48 01 c2             	add    %rax,%rdx
  4007b5:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007ba:	48 89 10             	mov    %rdx,(%rax)
  4007bd:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007c2:	c3                   	retq   
  4007c3:	90                   	nop

00000000004007c4 <_ZN9TestClass15increase_field0Ev>:
  4007c4:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007c9:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007cd:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007d1:	55                   	push   %rbp
  4007d2:	48 89 e5             	mov    %rsp,%rbp
  4007d5:	41 52                	push   %r10
  4007d7:	48 83 ec 48          	sub    $0x48,%rsp
  4007db:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4007df:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007e6:	00 00 
  4007e8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007ec:	31 c0                	xor    %eax,%eax
  4007ee:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4007f2:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4007f6:	ba 55 55 00 00       	mov    $0x5555,%edx
  4007fb:	48 89 ce             	mov    %rcx,%rsi
  4007fe:	48 89 c7             	mov    %rax,%rdi
  400801:	e8 e6 00 00 00       	callq  4008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEmLEi>
  400806:	90                   	nop
  400807:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40080b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400812:	00 00 
  400814:	74 05                	je     40081b <_ZN9TestClass15increase_field0Ev+0x57>
  400816:	e8 75 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  40081b:	48 83 c4 48          	add    $0x48,%rsp
  40081f:	41 5a                	pop    %r10
  400821:	5d                   	pop    %rbp
  400822:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400826:	c3                   	retq   
  400827:	90                   	nop

0000000000400828 <_ZN9TestClass15increase_field1Ev>:
  400828:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40082d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400831:	41 ff 72 f8          	pushq  -0x8(%r10)
  400835:	55                   	push   %rbp
  400836:	48 89 e5             	mov    %rsp,%rbp
  400839:	41 52                	push   %r10
  40083b:	48 83 ec 48          	sub    $0x48,%rsp
  40083f:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400843:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40084a:	00 00 
  40084c:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400850:	31 c0                	xor    %eax,%eax
  400852:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400856:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40085a:	ba 44 44 00 00       	mov    $0x4444,%edx
  40085f:	48 89 ce             	mov    %rcx,%rsi
  400862:	48 89 c7             	mov    %rax,%rdi
  400865:	e8 ca 00 00 00       	callq  400934 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEmLEi>
  40086a:	90                   	nop
  40086b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40086f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400876:	00 00 
  400878:	74 05                	je     40087f <_ZN9TestClass15increase_field1Ev+0x57>
  40087a:	e8 11 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  40087f:	48 83 c4 48          	add    $0x48,%rsp
  400883:	41 5a                	pop    %r10
  400885:	5d                   	pop    %rbp
  400886:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  40088a:	c3                   	retq   
  40088b:	90                   	nop

000000000040088c <_ZN9TestClass14storage_bufferEv>:
  40088c:	b8 80 20 60 00       	mov    $0x602080,%eax
  400891:	c3                   	retq   

0000000000400892 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400892:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400897:	90                   	nop
  400898:	c3                   	retq   
  400899:	90                   	nop

000000000040089a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  40089a:	48 83 ec 18          	sub    $0x18,%rsp
  40089e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008a3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008a8:	48 89 c7             	mov    %rax,%rdi
  4008ab:	e8 e2 ff ff ff       	callq  400892 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008b0:	90                   	nop
  4008b1:	48 83 c4 18          	add    $0x18,%rsp
  4008b5:	c3                   	retq   

00000000004008b6 <_ZN9TestClassC1Ev>:
  4008b6:	48 83 ec 18          	sub    $0x18,%rsp
  4008ba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008bf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008c4:	48 89 c7             	mov    %rax,%rdi
  4008c7:	e8 ce ff ff ff       	callq  40089a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  4008cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008d1:	48 89 c7             	mov    %rax,%rdi
  4008d4:	e8 07 01 00 00       	callq  4009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC1Ev>
  4008d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 02 01 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC1Ev>
  4008e6:	90                   	nop
  4008e7:	48 83 c4 18          	add    $0x18,%rsp
  4008eb:	c3                   	retq   

00000000004008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEmLEi>:
  4008ec:	48 83 ec 28          	sub    $0x28,%rsp
  4008f0:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008f5:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4008fa:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4008fe:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400903:	48 89 c7             	mov    %rax,%rdi
  400906:	e8 7f 01 00 00       	callq  400a8a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40090b:	8b 10                	mov    (%rax),%edx
  40090d:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400912:	89 10                	mov    %edx,(%rax)
  400914:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400919:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40091e:	48 89 d6             	mov    %rdx,%rsi
  400921:	48 89 c7             	mov    %rax,%rdi
  400924:	e8 7d 01 00 00       	callq  400aa6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC1ERKS3_>
  400929:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40092e:	48 83 c4 28          	add    $0x28,%rsp
  400932:	c3                   	retq   
  400933:	90                   	nop

0000000000400934 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEmLEi>:
  400934:	48 83 ec 28          	sub    $0x28,%rsp
  400938:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40093d:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400942:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400946:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40094b:	48 89 c7             	mov    %rax,%rdi
  40094e:	e8 5f 01 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400953:	8b 10                	mov    (%rax),%edx
  400955:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  40095a:	89 10                	mov    %edx,(%rax)
  40095c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400961:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400966:	48 89 d6             	mov    %rdx,%rsi
  400969:	48 89 c7             	mov    %rax,%rdi
  40096c:	e8 5d 01 00 00       	callq  400ace <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC1ERKS3_>
  400971:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400976:	48 83 c4 28          	add    $0x28,%rsp
  40097a:	c3                   	retq   

000000000040097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>:
  40097b:	48 83 ec 08          	sub    $0x8,%rsp
  40097f:	e8 08 ff ff ff       	callq  40088c <_ZN9TestClass14storage_bufferEv>
  400984:	48 83 c4 08          	add    $0x8,%rsp
  400988:	c3                   	retq   

0000000000400989 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  400989:	48 83 ec 08          	sub    $0x8,%rsp
  40098d:	e8 e9 ff ff ff       	callq  40097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400992:	48 89 c6             	mov    %rax,%rsi
  400995:	bf 20 00 00 00       	mov    $0x20,%edi
  40099a:	e8 cb fd ff ff       	callq  40076a <_ZnwmPv>
  40099f:	48 85 c0             	test   %rax,%rax
  4009a2:	74 08                	je     4009ac <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  4009a4:	48 89 c7             	mov    %rax,%rdi
  4009a7:	e8 4a 01 00 00       	callq  400af6 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  4009ac:	90                   	nop
  4009ad:	48 83 c4 08          	add    $0x8,%rsp
  4009b1:	c3                   	retq   

00000000004009b2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009b2:	48 83 ec 18          	sub    $0x18,%rsp
  4009b6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009bb:	e8 87 01 00 00       	callq  400b47 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009c0:	e8 b6 ff ff ff       	callq  40097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  4009c5:	be 01 00 00 00       	mov    $0x1,%esi
  4009ca:	48 89 c7             	mov    %rax,%rdi
  4009cd:	e8 bc fd ff ff       	callq  40078e <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009d2:	48 89 c7             	mov    %rax,%rdi
  4009d5:	e8 6f 01 00 00       	callq  400b49 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009da:	48 83 c4 18          	add    $0x18,%rsp
  4009de:	c3                   	retq   
  4009df:	90                   	nop

00000000004009e0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC1Ev>:
  4009e0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009e5:	90                   	nop
  4009e6:	c3                   	retq   
  4009e7:	90                   	nop

00000000004009e8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC1Ev>:
  4009e8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009ed:	90                   	nop
  4009ee:	c3                   	retq   

00000000004009ef <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  4009ef:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009f4:	90                   	nop
  4009f5:	c3                   	retq   

00000000004009f6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>:
  4009f6:	48 83 ec 18          	sub    $0x18,%rsp
  4009fa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009ff:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a03:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a08:	48 89 c7             	mov    %rax,%rdi
  400a0b:	e8 7a 00 00 00       	callq  400a8a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a10:	48 89 c2             	mov    %rax,%rdx
  400a13:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a17:	89 02                	mov    %eax,(%rdx)
  400a19:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a1e:	48 83 c4 18          	add    $0x18,%rsp
  400a22:	c3                   	retq   
  400a23:	90                   	nop

0000000000400a24 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>:
  400a24:	48 83 ec 18          	sub    $0x18,%rsp
  400a28:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a2d:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a31:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a36:	48 89 c7             	mov    %rax,%rdi
  400a39:	e8 74 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3e:	48 89 c2             	mov    %rax,%rdx
  400a41:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a45:	89 02                	mov    %eax,(%rdx)
  400a47:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a4c:	48 83 c4 18          	add    $0x18,%rsp
  400a50:	c3                   	retq   
  400a51:	90                   	nop

0000000000400a52 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>:
  400a52:	48 83 ec 18          	sub    $0x18,%rsp
  400a56:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a5b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a60:	48 89 c7             	mov    %rax,%rdi
  400a63:	e8 22 00 00 00       	callq  400a8a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a68:	48 83 c4 18          	add    $0x18,%rsp
  400a6c:	c3                   	retq   
  400a6d:	90                   	nop

0000000000400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>:
  400a6e:	48 83 ec 18          	sub    $0x18,%rsp
  400a72:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a77:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a7c:	48 89 c7             	mov    %rax,%rdi
  400a7f:	e8 2e 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a84:	48 83 c4 18          	add    $0x18,%rsp
  400a88:	c3                   	retq   
  400a89:	90                   	nop

0000000000400a8a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a8a:	48 83 ec 18          	sub    $0x18,%rsp
  400a8e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a93:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a98:	48 89 c7             	mov    %rax,%rdi
  400a9b:	e8 c8 00 00 00       	callq  400b68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400aa0:	48 83 c4 18          	add    $0x18,%rsp
  400aa4:	c3                   	retq   
  400aa5:	90                   	nop

0000000000400aa6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC1ERKS3_>:
  400aa6:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400aab:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400ab0:	90                   	nop
  400ab1:	c3                   	retq   

0000000000400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ab2:	48 83 ec 18          	sub    $0x18,%rsp
  400ab6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400abb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ac0:	48 89 c7             	mov    %rax,%rdi
  400ac3:	e8 d6 00 00 00       	callq  400b9e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400ac8:	48 83 c4 18          	add    $0x18,%rsp
  400acc:	c3                   	retq   
  400acd:	90                   	nop

0000000000400ace <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC1ERKS3_>:
  400ace:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ad3:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400ad8:	90                   	nop
  400ad9:	c3                   	retq   

0000000000400ada <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400ada:	48 83 ec 18          	sub    $0x18,%rsp
  400ade:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ae3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ae8:	48 89 c7             	mov    %rax,%rdi
  400aeb:	e8 8a fc ff ff       	callq  40077a <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400af0:	90                   	nop
  400af1:	48 83 c4 18          	add    $0x18,%rsp
  400af5:	c3                   	retq   

0000000000400af6 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400af6:	48 83 ec 18          	sub    $0x18,%rsp
  400afa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400aff:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b04:	48 89 c7             	mov    %rax,%rdi
  400b07:	e8 ce ff ff ff       	callq  400ada <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400b0c:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400b11:	e8 aa f9 ff ff       	callq  4004c0 <malloc@plt>
  400b16:	48 89 c2             	mov    %rax,%rdx
  400b19:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b1e:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400b22:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b27:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b2e:	00 
  400b2f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b34:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400b38:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b3d:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b41:	90                   	nop
  400b42:	48 83 c4 18          	add    $0x18,%rsp
  400b46:	c3                   	retq   

0000000000400b47 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b47:	90                   	nop
  400b48:	c3                   	retq   

0000000000400b49 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b49:	48 83 ec 18          	sub    $0x18,%rsp
  400b4d:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b52:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b57:	48 83 c0 01          	add    $0x1,%rax
  400b5b:	48 89 c7             	mov    %rax,%rdi
  400b5e:	e8 73 00 00 00       	callq  400bd6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b63:	48 83 c4 18          	add    $0x18,%rsp
  400b67:	c3                   	retq   

0000000000400b68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400b68:	48 83 ec 28          	sub    $0x28,%rsp
  400b6c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b71:	e8 05 fe ff ff       	callq  40097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400b76:	48 89 c7             	mov    %rax,%rdi
  400b79:	e8 64 00 00 00       	callq  400be2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400b7e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b83:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b88:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b8f:	00 
  400b90:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b95:	48 01 d0             	add    %rdx,%rax
  400b98:	48 83 c4 28          	add    $0x28,%rsp
  400b9c:	c3                   	retq   
  400b9d:	90                   	nop

0000000000400b9e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400b9e:	48 83 ec 28          	sub    $0x28,%rsp
  400ba2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ba7:	e8 cf fd ff ff       	callq  40097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400bac:	48 89 c7             	mov    %rax,%rdi
  400baf:	e8 2e 00 00 00       	callq  400be2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400bb4:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bb9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bbe:	48 c1 e0 02          	shl    $0x2,%rax
  400bc2:	48 8d 90 d4 48 00 00 	lea    0x48d4(%rax),%rdx
  400bc9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bce:	48 01 d0             	add    %rdx,%rax
  400bd1:	48 83 c4 28          	add    $0x28,%rsp
  400bd5:	c3                   	retq   

0000000000400bd6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bd6:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bdb:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400be0:	c3                   	retq   
  400be1:	90                   	nop

0000000000400be2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400be2:	48 83 ec 18          	sub    $0x18,%rsp
  400be6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400beb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bf0:	48 89 c7             	mov    %rax,%rdi
  400bf3:	e8 06 00 00 00       	callq  400bfe <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bf8:	48 83 c4 18          	add    $0x18,%rsp
  400bfc:	c3                   	retq   
  400bfd:	90                   	nop

0000000000400bfe <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400bfe:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c03:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400c08:	48 8b 40 18          	mov    0x18(%rax),%rax
  400c0c:	c3                   	retq   
  400c0d:	0f 1f 00             	nopl   (%rax)

0000000000400c10 <__libc_csu_init>:
  400c10:	41 57                	push   %r15
  400c12:	41 56                	push   %r14
  400c14:	41 89 ff             	mov    %edi,%r15d
  400c17:	41 55                	push   %r13
  400c19:	41 54                	push   %r12
  400c1b:	4c 8d 25 ee 11 20 00 	lea    0x2011ee(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c22:	55                   	push   %rbp
  400c23:	48 8d 2d ee 11 20 00 	lea    0x2011ee(%rip),%rbp        # 601e18 <__init_array_end>
  400c2a:	53                   	push   %rbx
  400c2b:	49 89 f6             	mov    %rsi,%r14
  400c2e:	49 89 d5             	mov    %rdx,%r13
  400c31:	4c 29 e5             	sub    %r12,%rbp
  400c34:	48 83 ec 08          	sub    $0x8,%rsp
  400c38:	48 c1 fd 03          	sar    $0x3,%rbp
  400c3c:	e8 1f f8 ff ff       	callq  400460 <_init>
  400c41:	48 85 ed             	test   %rbp,%rbp
  400c44:	74 20                	je     400c66 <__libc_csu_init+0x56>
  400c46:	31 db                	xor    %ebx,%ebx
  400c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c4f:	00 
  400c50:	4c 89 ea             	mov    %r13,%rdx
  400c53:	4c 89 f6             	mov    %r14,%rsi
  400c56:	44 89 ff             	mov    %r15d,%edi
  400c59:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c5d:	48 83 c3 01          	add    $0x1,%rbx
  400c61:	48 39 eb             	cmp    %rbp,%rbx
  400c64:	75 ea                	jne    400c50 <__libc_csu_init+0x40>
  400c66:	48 83 c4 08          	add    $0x8,%rsp
  400c6a:	5b                   	pop    %rbx
  400c6b:	5d                   	pop    %rbp
  400c6c:	41 5c                	pop    %r12
  400c6e:	41 5d                	pop    %r13
  400c70:	41 5e                	pop    %r14
  400c72:	41 5f                	pop    %r15
  400c74:	c3                   	retq   
  400c75:	90                   	nop
  400c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7d:	00 00 00 

0000000000400c80 <__libc_csu_fini>:
  400c80:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c84 <_fini>:
  400c84:	48 83 ec 08          	sub    $0x8,%rsp
  400c88:	48 83 c4 08          	add    $0x8,%rsp
  400c8c:	c3                   	retq   
