
bin/valid_gcc_O0:     file format elf64-x86-64


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
  4004ef:	49 c7 c0 40 0c 40 00 	mov    $0x400c40,%r8
  4004f6:	48 c7 c1 d0 0b 40 00 	mov    $0x400bd0,%rcx
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
  4005f3:	e8 d6 03 00 00       	callq  4009ce <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>
  4005f8:	48 89 c3             	mov    %rax,%rbx
  4005fb:	b8 00 00 00 00       	mov    $0x0,%eax
  400600:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400604:	48 83 f8 03          	cmp    $0x3,%rax
  400608:	74 06                	je     400610 <main+0x3a>
  40060a:	48 83 c0 01          	add    $0x1,%rax
  40060e:	eb f0                	jmp    400600 <main+0x2a>
  400610:	48 89 df             	mov    %rbx,%rdi
  400613:	e8 d6 02 00 00       	callq  4008ee <_ZN9TestClassC1Ev>
  400618:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  40061c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400620:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400624:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400628:	ba 77 77 00 00       	mov    $0x7777,%edx
  40062d:	48 89 ce             	mov    %rcx,%rsi
  400630:	48 89 c7             	mov    %rax,%rdi
  400633:	e8 e4 03 00 00       	callq  400a1c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400638:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40063c:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400640:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400644:	ba 88 88 00 00       	mov    $0x8888,%edx
  400649:	48 89 ce             	mov    %rcx,%rsi
  40064c:	48 89 c7             	mov    %rax,%rdi
  40064f:	e8 08 04 00 00       	callq  400a5c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400654:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400658:	48 89 c7             	mov    %rax,%rdi
  40065b:	e8 ca 01 00 00       	callq  40082a <_ZN9TestClass15increase_field0Ev>
  400660:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400664:	48 89 c7             	mov    %rax,%rdi
  400667:	e8 0c 02 00 00       	callq  400878 <_ZN9TestClass15increase_field1Ev>
  40066c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400670:	48 83 c0 04          	add    $0x4,%rax
  400674:	48 89 c7             	mov    %rax,%rdi
  400677:	e8 3a 04 00 00       	callq  400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40067c:	8b 18                	mov    (%rax),%ebx
  40067e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400682:	48 83 c0 04          	add    $0x4,%rax
  400686:	48 89 c7             	mov    %rax,%rdi
  400689:	e8 0e 04 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40068e:	8b 00                	mov    (%rax),%eax
  400690:	89 da                	mov    %ebx,%edx
  400692:	89 c6                	mov    %eax,%esi
  400694:	bf 60 0c 40 00       	mov    $0x400c60,%edi
  400699:	b8 00 00 00 00       	mov    $0x0,%eax
  40069e:	e8 fd fd ff ff       	callq  4004a0 <printf@plt>
  4006a3:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006a7:	48 83 c0 04          	add    $0x4,%rax
  4006ab:	48 89 c7             	mov    %rax,%rdi
  4006ae:	e8 e9 03 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006b3:	8b 00                	mov    (%rax),%eax
  4006b5:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006ba:	75 19                	jne    4006d5 <main+0xff>
  4006bc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006c0:	48 83 c0 04          	add    $0x4,%rax
  4006c4:	48 89 c7             	mov    %rax,%rdi
  4006c7:	e8 ea 03 00 00       	callq  400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
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
  40070d:	e8 bc 02 00 00       	callq  4009ce <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>
  400712:	48 89 c3             	mov    %rax,%rbx
  400715:	b8 00 00 00 00       	mov    $0x0,%eax
  40071a:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  40071e:	48 83 f8 03          	cmp    $0x3,%rax
  400722:	74 06                	je     40072a <_Z12new_instancev+0x2b>
  400724:	48 83 c0 01          	add    $0x1,%rax
  400728:	eb f0                	jmp    40071a <_Z12new_instancev+0x1b>
  40072a:	48 89 df             	mov    %rbx,%rdi
  40072d:	e8 bc 01 00 00       	callq  4008ee <_ZN9TestClassC1Ev>
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
  40076e:	e8 a9 02 00 00       	callq  400a1c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
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
  4007a1:	e8 f6 02 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007a6:	8b 00                	mov    (%rax),%eax
  4007a8:	c9                   	leaveq 
  4007a9:	c3                   	retq   

00000000004007aa <_Z21explicit_write_field0m>:
  4007aa:	55                   	push   %rbp
  4007ab:	48 89 e5             	mov    %rsp,%rbp
  4007ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b2:	48 8b 05 b7 18 20 00 	mov    0x2018b7(%rip),%rax        # 602070 <explicit_field0>
  4007b9:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007bd:	48 c1 e2 02          	shl    $0x2,%rdx
  4007c1:	48 01 d0             	add    %rdx,%rax
  4007c4:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007ca:	90                   	nop
  4007cb:	5d                   	pop    %rbp
  4007cc:	c3                   	retq   

00000000004007cd <_Z20explicit_read_field0m>:
  4007cd:	55                   	push   %rbp
  4007ce:	48 89 e5             	mov    %rsp,%rbp
  4007d1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d5:	48 8b 05 94 18 20 00 	mov    0x201894(%rip),%rax        # 602070 <explicit_field0>
  4007dc:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007e0:	48 c1 e2 02          	shl    $0x2,%rdx
  4007e4:	48 01 d0             	add    %rdx,%rax
  4007e7:	8b 00                	mov    (%rax),%eax
  4007e9:	5d                   	pop    %rbp
  4007ea:	c3                   	retq   

00000000004007eb <_Z41__static_initialization_and_destruction_0ii>:
  4007eb:	55                   	push   %rbp
  4007ec:	48 89 e5             	mov    %rsp,%rbp
  4007ef:	48 83 ec 10          	sub    $0x10,%rsp
  4007f3:	89 7d fc             	mov    %edi,-0x4(%rbp)
  4007f6:	89 75 f8             	mov    %esi,-0x8(%rbp)
  4007f9:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
  4007fd:	75 13                	jne    400812 <_Z41__static_initialization_and_destruction_0ii+0x27>
  4007ff:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
  400806:	75 0a                	jne    400812 <_Z41__static_initialization_and_destruction_0ii+0x27>
  400808:	bf 60 20 60 00       	mov    $0x602060,%edi
  40080d:	e8 98 01 00 00       	callq  4009aa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE15InternalStorageC1Ev>
  400812:	90                   	nop
  400813:	c9                   	leaveq 
  400814:	c3                   	retq   

0000000000400815 <_GLOBAL__sub_I__ZN9TestClass7storageE>:
  400815:	55                   	push   %rbp
  400816:	48 89 e5             	mov    %rsp,%rbp
  400819:	be ff ff 00 00       	mov    $0xffff,%esi
  40081e:	bf 01 00 00 00       	mov    $0x1,%edi
  400823:	e8 c3 ff ff ff       	callq  4007eb <_Z41__static_initialization_and_destruction_0ii>
  400828:	5d                   	pop    %rbp
  400829:	c3                   	retq   

000000000040082a <_ZN9TestClass15increase_field0Ev>:
  40082a:	55                   	push   %rbp
  40082b:	48 89 e5             	mov    %rsp,%rbp
  40082e:	48 83 ec 30          	sub    $0x30,%rsp
  400832:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400836:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40083d:	00 00 
  40083f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400843:	31 c0                	xor    %eax,%eax
  400845:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400849:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40084d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400851:	ba 55 55 00 00       	mov    $0x5555,%edx
  400856:	48 89 ce             	mov    %rcx,%rsi
  400859:	48 89 c7             	mov    %rax,%rdi
  40085c:	e8 c9 00 00 00       	callq  40092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400861:	90                   	nop
  400862:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400866:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40086d:	00 00 
  40086f:	74 05                	je     400876 <_ZN9TestClass15increase_field0Ev+0x4c>
  400871:	e8 1a fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400876:	c9                   	leaveq 
  400877:	c3                   	retq   

0000000000400878 <_ZN9TestClass15increase_field1Ev>:
  400878:	55                   	push   %rbp
  400879:	48 89 e5             	mov    %rsp,%rbp
  40087c:	48 83 ec 30          	sub    $0x30,%rsp
  400880:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400884:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40088b:	00 00 
  40088d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400891:	31 c0                	xor    %eax,%eax
  400893:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400897:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40089b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40089f:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008a4:	48 89 ce             	mov    %rcx,%rsi
  4008a7:	48 89 c7             	mov    %rax,%rdi
  4008aa:	e8 bb 00 00 00       	callq  40096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008af:	90                   	nop
  4008b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008bb:	00 00 
  4008bd:	74 05                	je     4008c4 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008bf:	e8 cc fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008c4:	c9                   	leaveq 
  4008c5:	c3                   	retq   

00000000004008c6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008c6:	55                   	push   %rbp
  4008c7:	48 89 e5             	mov    %rsp,%rbp
  4008ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ce:	90                   	nop
  4008cf:	5d                   	pop    %rbp
  4008d0:	c3                   	retq   
  4008d1:	90                   	nop

00000000004008d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEC1Ev>:
  4008d2:	55                   	push   %rbp
  4008d3:	48 89 e5             	mov    %rsp,%rbp
  4008d6:	48 83 ec 10          	sub    $0x10,%rsp
  4008da:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008de:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e2:	48 89 c7             	mov    %rax,%rdi
  4008e5:	e8 dc ff ff ff       	callq  4008c6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ea:	90                   	nop
  4008eb:	c9                   	leaveq 
  4008ec:	c3                   	retq   
  4008ed:	90                   	nop

00000000004008ee <_ZN9TestClassC1Ev>:
  4008ee:	55                   	push   %rbp
  4008ef:	48 89 e5             	mov    %rsp,%rbp
  4008f2:	48 83 ec 10          	sub    $0x10,%rsp
  4008f6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008fe:	48 89 c7             	mov    %rax,%rdi
  400901:	e8 cc ff ff ff       	callq  4008d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEC1Ev>
  400906:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40090a:	48 83 c0 04          	add    $0x4,%rax
  40090e:	48 89 c7             	mov    %rax,%rdi
  400911:	e8 e4 00 00 00       	callq  4009fa <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400916:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091a:	48 83 c0 04          	add    $0x4,%rax
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 e0 00 00 00       	callq  400a06 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400926:	90                   	nop
  400927:	c9                   	leaveq 
  400928:	c3                   	retq   
  400929:	90                   	nop

000000000040092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40092a:	55                   	push   %rbp
  40092b:	48 89 e5             	mov    %rsp,%rbp
  40092e:	48 83 ec 20          	sub    $0x20,%rsp
  400932:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400936:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40093a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40093d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400941:	48 89 c7             	mov    %rax,%rdi
  400944:	e8 87 01 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400949:	8b 10                	mov    (%rax),%edx
  40094b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40094f:	89 10                	mov    %edx,(%rax)
  400951:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400955:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400959:	48 89 d6             	mov    %rdx,%rsi
  40095c:	48 89 c7             	mov    %rax,%rdi
  40095f:	e8 b0 01 00 00       	callq  400b14 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400964:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400968:	c9                   	leaveq 
  400969:	c3                   	retq   

000000000040096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  40096a:	55                   	push   %rbp
  40096b:	48 89 e5             	mov    %rsp,%rbp
  40096e:	48 83 ec 20          	sub    $0x20,%rsp
  400972:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400976:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40097a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40097d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400981:	48 89 c7             	mov    %rax,%rdi
  400984:	e8 9b 01 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400989:	8b 10                	mov    (%rax),%edx
  40098b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40098f:	89 10                	mov    %edx,(%rax)
  400991:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400995:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400999:	48 89 d6             	mov    %rdx,%rsi
  40099c:	48 89 c7             	mov    %rax,%rdi
  40099f:	e8 ca 01 00 00       	callq  400b6e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a8:	c9                   	leaveq 
  4009a9:	c3                   	retq   

00000000004009aa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE15InternalStorageC1Ev>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	48 83 ec 10          	sub    $0x10,%rsp
  4009b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b6:	bf a0 91 00 00       	mov    $0x91a0,%edi
  4009bb:	e8 00 fb ff ff       	callq  4004c0 <malloc@plt>
  4009c0:	48 89 c2             	mov    %rax,%rdx
  4009c3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009c7:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4009cb:	90                   	nop
  4009cc:	c9                   	leaveq 
  4009cd:	c3                   	retq   

00000000004009ce <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>:
  4009ce:	55                   	push   %rbp
  4009cf:	48 89 e5             	mov    %rsp,%rbp
  4009d2:	48 83 ec 10          	sub    $0x10,%rsp
  4009d6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009da:	e8 9e 01 00 00       	callq  400b7d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009df:	8b 05 7b 16 20 00    	mov    0x20167b(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  4009e5:	8d 50 01             	lea    0x1(%rax),%edx
  4009e8:	89 15 72 16 20 00    	mov    %edx,0x201672(%rip)        # 602060 <_ZN9TestClass7storageE>
  4009ee:	89 c0                	mov    %eax,%eax
  4009f0:	48 89 c7             	mov    %rax,%rdi
  4009f3:	e8 8c 01 00 00       	callq  400b84 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE3getEm>
  4009f8:	c9                   	leaveq 
  4009f9:	c3                   	retq   

00000000004009fa <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4009fa:	55                   	push   %rbp
  4009fb:	48 89 e5             	mov    %rsp,%rbp
  4009fe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a02:	90                   	nop
  400a03:	5d                   	pop    %rbp
  400a04:	c3                   	retq   
  400a05:	90                   	nop

0000000000400a06 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a06:	55                   	push   %rbp
  400a07:	48 89 e5             	mov    %rsp,%rbp
  400a0a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0e:	90                   	nop
  400a0f:	5d                   	pop    %rbp
  400a10:	c3                   	retq   

0000000000400a11 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEdlEPv>:
  400a11:	55                   	push   %rbp
  400a12:	48 89 e5             	mov    %rsp,%rbp
  400a15:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a19:	90                   	nop
  400a1a:	5d                   	pop    %rbp
  400a1b:	c3                   	retq   

0000000000400a1c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a1c:	55                   	push   %rbp
  400a1d:	48 89 e5             	mov    %rsp,%rbp
  400a20:	48 83 ec 20          	sub    $0x20,%rsp
  400a24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a2c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a2f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a33:	48 89 c7             	mov    %rax,%rdi
  400a36:	e8 95 00 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3b:	48 89 c2             	mov    %rax,%rdx
  400a3e:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a41:	89 02                	mov    %eax,(%rdx)
  400a43:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a47:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a4b:	48 89 d6             	mov    %rdx,%rsi
  400a4e:	48 89 c7             	mov    %rax,%rdi
  400a51:	e8 be 00 00 00       	callq  400b14 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a56:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a5a:	c9                   	leaveq 
  400a5b:	c3                   	retq   

0000000000400a5c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a5c:	55                   	push   %rbp
  400a5d:	48 89 e5             	mov    %rsp,%rbp
  400a60:	48 83 ec 20          	sub    $0x20,%rsp
  400a64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a68:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a6c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a6f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a73:	48 89 c7             	mov    %rax,%rdi
  400a76:	e8 a9 00 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a7b:	48 89 c2             	mov    %rax,%rdx
  400a7e:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a81:	89 02                	mov    %eax,(%rdx)
  400a83:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a87:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a8b:	48 89 d6             	mov    %rdx,%rsi
  400a8e:	48 89 c7             	mov    %rax,%rdi
  400a91:	e8 d8 00 00 00       	callq  400b6e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a96:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a9a:	c9                   	leaveq 
  400a9b:	c3                   	retq   

0000000000400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400a9c:	55                   	push   %rbp
  400a9d:	48 89 e5             	mov    %rsp,%rbp
  400aa0:	48 83 ec 10          	sub    $0x10,%rsp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aac:	48 89 c7             	mov    %rax,%rdi
  400aaf:	e8 1c 00 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab4:	c9                   	leaveq 
  400ab5:	c3                   	retq   

0000000000400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400ab6:	55                   	push   %rbp
  400ab7:	48 89 e5             	mov    %rsp,%rbp
  400aba:	48 83 ec 10          	sub    $0x10,%rsp
  400abe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac6:	48 89 c7             	mov    %rax,%rdi
  400ac9:	e8 56 00 00 00       	callq  400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ace:	c9                   	leaveq 
  400acf:	c3                   	retq   

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400ad8:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400adc:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400ae0:	48 8b 05 81 15 20 00 	mov    0x201581(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400ae7:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400aeb:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400aef:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400af3:	48 83 e8 04          	sub    $0x4,%rax
  400af7:	48 c1 e8 02          	shr    $0x2,%rax
  400afb:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b02:	00 
  400b03:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b07:	48 01 d0             	add    %rdx,%rax
  400b0a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b0e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b12:	5d                   	pop    %rbp
  400b13:	c3                   	retq   

0000000000400b14 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b14:	55                   	push   %rbp
  400b15:	48 89 e5             	mov    %rsp,%rbp
  400b18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b1c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b20:	90                   	nop
  400b21:	5d                   	pop    %rbp
  400b22:	c3                   	retq   
  400b23:	90                   	nop

0000000000400b24 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b24:	55                   	push   %rbp
  400b25:	48 89 e5             	mov    %rsp,%rbp
  400b28:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b2c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b30:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b34:	48 8b 05 2d 15 20 00 	mov    0x20152d(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400b3b:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b3f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b43:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b47:	48 83 e8 04          	sub    $0x4,%rax
  400b4b:	48 c1 e8 02          	shr    $0x2,%rax
  400b4f:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b56:	00 
  400b57:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b5b:	48 01 d0             	add    %rdx,%rax
  400b5e:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400b64:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b68:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b6c:	5d                   	pop    %rbp
  400b6d:	c3                   	retq   

0000000000400b6e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b6e:	55                   	push   %rbp
  400b6f:	48 89 e5             	mov    %rsp,%rbp
  400b72:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b76:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b7a:	90                   	nop
  400b7b:	5d                   	pop    %rbp
  400b7c:	c3                   	retq   

0000000000400b7d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b7d:	55                   	push   %rbp
  400b7e:	48 89 e5             	mov    %rsp,%rbp
  400b81:	90                   	nop
  400b82:	5d                   	pop    %rbp
  400b83:	c3                   	retq   

0000000000400b84 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE3getEm>:
  400b84:	55                   	push   %rbp
  400b85:	48 89 e5             	mov    %rsp,%rbp
  400b88:	48 83 ec 10          	sub    $0x10,%rsp
  400b8c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b90:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b94:	48 89 c7             	mov    %rax,%rdi
  400b97:	e8 02 00 00 00       	callq  400b9e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b9c:	c9                   	leaveq 
  400b9d:	c3                   	retq   

0000000000400b9e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b9e:	55                   	push   %rbp
  400b9f:	48 89 e5             	mov    %rsp,%rbp
  400ba2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400ba6:	48 8b 05 bb 14 20 00 	mov    0x2014bb(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400bad:	48 89 c2             	mov    %rax,%rdx
  400bb0:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bb4:	48 c1 e0 02          	shl    $0x2,%rax
  400bb8:	48 01 d0             	add    %rdx,%rax
  400bbb:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bbf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bc3:	5d                   	pop    %rbp
  400bc4:	c3                   	retq   
  400bc5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bcc:	00 00 00 
  400bcf:	90                   	nop

0000000000400bd0 <__libc_csu_init>:
  400bd0:	41 57                	push   %r15
  400bd2:	41 56                	push   %r14
  400bd4:	41 89 ff             	mov    %edi,%r15d
  400bd7:	41 55                	push   %r13
  400bd9:	41 54                	push   %r12
  400bdb:	4c 8d 25 26 12 20 00 	lea    0x201226(%rip),%r12        # 601e08 <__frame_dummy_init_array_entry>
  400be2:	55                   	push   %rbp
  400be3:	48 8d 2d 2e 12 20 00 	lea    0x20122e(%rip),%rbp        # 601e18 <__init_array_end>
  400bea:	53                   	push   %rbx
  400beb:	49 89 f6             	mov    %rsi,%r14
  400bee:	49 89 d5             	mov    %rdx,%r13
  400bf1:	4c 29 e5             	sub    %r12,%rbp
  400bf4:	48 83 ec 08          	sub    $0x8,%rsp
  400bf8:	48 c1 fd 03          	sar    $0x3,%rbp
  400bfc:	e8 5f f8 ff ff       	callq  400460 <_init>
  400c01:	48 85 ed             	test   %rbp,%rbp
  400c04:	74 20                	je     400c26 <__libc_csu_init+0x56>
  400c06:	31 db                	xor    %ebx,%ebx
  400c08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c0f:	00 
  400c10:	4c 89 ea             	mov    %r13,%rdx
  400c13:	4c 89 f6             	mov    %r14,%rsi
  400c16:	44 89 ff             	mov    %r15d,%edi
  400c19:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c1d:	48 83 c3 01          	add    $0x1,%rbx
  400c21:	48 39 eb             	cmp    %rbp,%rbx
  400c24:	75 ea                	jne    400c10 <__libc_csu_init+0x40>
  400c26:	48 83 c4 08          	add    $0x8,%rsp
  400c2a:	5b                   	pop    %rbx
  400c2b:	5d                   	pop    %rbp
  400c2c:	41 5c                	pop    %r12
  400c2e:	41 5d                	pop    %r13
  400c30:	41 5e                	pop    %r14
  400c32:	41 5f                	pop    %r15
  400c34:	c3                   	retq   
  400c35:	90                   	nop
  400c36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c3d:	00 00 00 

0000000000400c40 <__libc_csu_fini>:
  400c40:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c44 <_fini>:
  400c44:	48 83 ec 08          	sub    $0x8,%rsp
  400c48:	48 83 c4 08          	add    $0x8,%rsp
  400c4c:	c3                   	retq   
