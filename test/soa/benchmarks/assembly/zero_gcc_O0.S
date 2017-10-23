
bin/zero_gcc_O0:     file format elf64-x86-64


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
  4004ef:	49 c7 c0 e0 0b 40 00 	mov    $0x400be0,%r8
  4004f6:	48 c7 c1 70 0b 40 00 	mov    $0x400b70,%rcx
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
  4005ee:	bf 00 00 00 00       	mov    $0x0,%edi
  4005f3:	e8 e2 03 00 00       	callq  4009da <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  4005f8:	48 89 c3             	mov    %rax,%rbx
  4005fb:	48 89 df             	mov    %rbx,%rdi
  4005fe:	e8 ff 02 00 00       	callq  400902 <_ZN9TestClassC1Ev>
  400603:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  400607:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  40060b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40060f:	ba 77 77 00 00       	mov    $0x7777,%edx
  400614:	48 89 ce             	mov    %rcx,%rsi
  400617:	48 89 c7             	mov    %rax,%rdi
  40061a:	e8 09 04 00 00       	callq  400a28 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40061f:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400623:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400627:	ba 88 88 00 00       	mov    $0x8888,%edx
  40062c:	48 89 ce             	mov    %rcx,%rsi
  40062f:	48 89 c7             	mov    %rax,%rdi
  400632:	e8 31 04 00 00       	callq  400a68 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400637:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40063b:	48 89 c7             	mov    %rax,%rdi
  40063e:	e8 03 02 00 00       	callq  400846 <_ZN9TestClass15increase_field0Ev>
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 89 c7             	mov    %rax,%rdi
  40064a:	e8 41 02 00 00       	callq  400890 <_ZN9TestClass15increase_field1Ev>
  40064f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400653:	48 89 c7             	mov    %rax,%rdi
  400656:	e8 67 04 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40065b:	8b 18                	mov    (%rax),%ebx
  40065d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400661:	48 89 c7             	mov    %rax,%rdi
  400664:	e8 3f 04 00 00       	callq  400aa8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400669:	8b 00                	mov    (%rax),%eax
  40066b:	89 da                	mov    %ebx,%edx
  40066d:	89 c6                	mov    %eax,%esi
  40066f:	bf 00 0c 40 00       	mov    $0x400c00,%edi
  400674:	b8 00 00 00 00       	mov    $0x0,%eax
  400679:	e8 22 fe ff ff       	callq  4004a0 <printf@plt>
  40067e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400682:	48 89 c7             	mov    %rax,%rdi
  400685:	e8 1e 04 00 00       	callq  400aa8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40068a:	8b 00                	mov    (%rax),%eax
  40068c:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400691:	75 15                	jne    4006a8 <main+0xd2>
  400693:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400697:	48 89 c7             	mov    %rax,%rdi
  40069a:	e8 23 04 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40069f:	8b 00                	mov    (%rax),%eax
  4006a1:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006a6:	74 07                	je     4006af <main+0xd9>
  4006a8:	b8 01 00 00 00       	mov    $0x1,%eax
  4006ad:	eb 05                	jmp    4006b4 <main+0xde>
  4006af:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b4:	0f b6 c0             	movzbl %al,%eax
  4006b7:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006bb:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006c2:	00 00 
  4006c4:	74 05                	je     4006cb <main+0xf5>
  4006c6:	e8 c5 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  4006cb:	48 83 c4 28          	add    $0x28,%rsp
  4006cf:	5b                   	pop    %rbx
  4006d0:	5d                   	pop    %rbp
  4006d1:	c3                   	retq   

00000000004006d2 <_Z12new_instancev>:
  4006d2:	55                   	push   %rbp
  4006d3:	48 89 e5             	mov    %rsp,%rbp
  4006d6:	53                   	push   %rbx
  4006d7:	48 83 ec 08          	sub    $0x8,%rsp
  4006db:	bf 00 00 00 00       	mov    $0x0,%edi
  4006e0:	e8 f5 02 00 00       	callq  4009da <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  4006e5:	48 89 c3             	mov    %rax,%rbx
  4006e8:	48 89 df             	mov    %rbx,%rdi
  4006eb:	e8 12 02 00 00       	callq  400902 <_ZN9TestClassC1Ev>
  4006f0:	48 89 d8             	mov    %rbx,%rax
  4006f3:	48 83 c4 08          	add    $0x8,%rsp
  4006f7:	5b                   	pop    %rbx
  4006f8:	5d                   	pop    %rbp
  4006f9:	c3                   	retq   

00000000004006fa <_Z12write_field0P9TestClass>:
  4006fa:	55                   	push   %rbp
  4006fb:	48 89 e5             	mov    %rsp,%rbp
  4006fe:	48 83 ec 30          	sub    $0x30,%rsp
  400702:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400706:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40070d:	00 00 
  40070f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400713:	31 c0                	xor    %eax,%eax
  400715:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400719:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40071d:	ba 77 77 00 00       	mov    $0x7777,%edx
  400722:	48 89 ce             	mov    %rcx,%rsi
  400725:	48 89 c7             	mov    %rax,%rdi
  400728:	e8 fb 02 00 00       	callq  400a28 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40072d:	90                   	nop
  40072e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400732:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400739:	00 00 
  40073b:	74 05                	je     400742 <_Z12write_field0P9TestClass+0x48>
  40073d:	e8 4e fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  400742:	c9                   	leaveq 
  400743:	c3                   	retq   

0000000000400744 <_Z11read_field0P9TestClass>:
  400744:	55                   	push   %rbp
  400745:	48 89 e5             	mov    %rsp,%rbp
  400748:	48 83 ec 10          	sub    $0x10,%rsp
  40074c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400750:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400754:	48 89 c7             	mov    %rax,%rdi
  400757:	e8 4c 03 00 00       	callq  400aa8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40075c:	8b 00                	mov    (%rax),%eax
  40075e:	c9                   	leaveq 
  40075f:	c3                   	retq   

0000000000400760 <_Z12write_field1P9TestClass>:
  400760:	55                   	push   %rbp
  400761:	48 89 e5             	mov    %rsp,%rbp
  400764:	48 83 ec 30          	sub    $0x30,%rsp
  400768:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40076c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400773:	00 00 
  400775:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400779:	31 c0                	xor    %eax,%eax
  40077b:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40077f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400783:	ba 77 77 00 00       	mov    $0x7777,%edx
  400788:	48 89 ce             	mov    %rcx,%rsi
  40078b:	48 89 c7             	mov    %rax,%rdi
  40078e:	e8 d5 02 00 00       	callq  400a68 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400793:	90                   	nop
  400794:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400798:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40079f:	00 00 
  4007a1:	74 05                	je     4007a8 <_Z12write_field1P9TestClass+0x48>
  4007a3:	e8 e8 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  4007a8:	c9                   	leaveq 
  4007a9:	c3                   	retq   

00000000004007aa <_Z11read_field1P9TestClass>:
  4007aa:	55                   	push   %rbp
  4007ab:	48 89 e5             	mov    %rsp,%rbp
  4007ae:	48 83 ec 10          	sub    $0x10,%rsp
  4007b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007ba:	48 89 c7             	mov    %rax,%rdi
  4007bd:	e8 00 03 00 00       	callq  400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007c2:	8b 00                	mov    (%rax),%eax
  4007c4:	c9                   	leaveq 
  4007c5:	c3                   	retq   

00000000004007c6 <_Z21explicit_write_field0m>:
  4007c6:	55                   	push   %rbp
  4007c7:	48 89 e5             	mov    %rsp,%rbp
  4007ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ce:	48 8b 05 9b 18 20 00 	mov    0x20189b(%rip),%rax        # 602070 <explicit_field0>
  4007d5:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007d9:	48 c1 e2 02          	shl    $0x2,%rdx
  4007dd:	48 01 d0             	add    %rdx,%rax
  4007e0:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007e6:	90                   	nop
  4007e7:	5d                   	pop    %rbp
  4007e8:	c3                   	retq   

00000000004007e9 <_Z20explicit_read_field0m>:
  4007e9:	55                   	push   %rbp
  4007ea:	48 89 e5             	mov    %rsp,%rbp
  4007ed:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007f1:	48 8b 05 78 18 20 00 	mov    0x201878(%rip),%rax        # 602070 <explicit_field0>
  4007f8:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007fc:	48 c1 e2 02          	shl    $0x2,%rdx
  400800:	48 01 d0             	add    %rdx,%rax
  400803:	8b 00                	mov    (%rax),%eax
  400805:	5d                   	pop    %rbp
  400806:	c3                   	retq   

0000000000400807 <_Z41__static_initialization_and_destruction_0ii>:
  400807:	55                   	push   %rbp
  400808:	48 89 e5             	mov    %rsp,%rbp
  40080b:	48 83 ec 10          	sub    $0x10,%rsp
  40080f:	89 7d fc             	mov    %edi,-0x4(%rbp)
  400812:	89 75 f8             	mov    %esi,-0x8(%rbp)
  400815:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
  400819:	75 13                	jne    40082e <_Z41__static_initialization_and_destruction_0ii+0x27>
  40081b:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
  400822:	75 0a                	jne    40082e <_Z41__static_initialization_and_destruction_0ii+0x27>
  400824:	bf 60 20 60 00       	mov    $0x602060,%edi
  400829:	e8 88 01 00 00       	callq  4009b6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC1Ev>
  40082e:	90                   	nop
  40082f:	c9                   	leaveq 
  400830:	c3                   	retq   

0000000000400831 <_GLOBAL__sub_I__ZN9TestClass7storageE>:
  400831:	55                   	push   %rbp
  400832:	48 89 e5             	mov    %rsp,%rbp
  400835:	be ff ff 00 00       	mov    $0xffff,%esi
  40083a:	bf 01 00 00 00       	mov    $0x1,%edi
  40083f:	e8 c3 ff ff ff       	callq  400807 <_Z41__static_initialization_and_destruction_0ii>
  400844:	5d                   	pop    %rbp
  400845:	c3                   	retq   

0000000000400846 <_ZN9TestClass15increase_field0Ev>:
  400846:	55                   	push   %rbp
  400847:	48 89 e5             	mov    %rsp,%rbp
  40084a:	48 83 ec 30          	sub    $0x30,%rsp
  40084e:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400852:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400859:	00 00 
  40085b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40085f:	31 c0                	xor    %eax,%eax
  400861:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400865:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400869:	ba 55 55 00 00       	mov    $0x5555,%edx
  40086e:	48 89 ce             	mov    %rcx,%rsi
  400871:	48 89 c7             	mov    %rax,%rdi
  400874:	e8 bd 00 00 00       	callq  400936 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400879:	90                   	nop
  40087a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40087e:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400885:	00 00 
  400887:	74 05                	je     40088e <_ZN9TestClass15increase_field0Ev+0x48>
  400889:	e8 02 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  40088e:	c9                   	leaveq 
  40088f:	c3                   	retq   

0000000000400890 <_ZN9TestClass15increase_field1Ev>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 83 ec 30          	sub    $0x30,%rsp
  400898:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40089c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008a3:	00 00 
  4008a5:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008a9:	31 c0                	xor    %eax,%eax
  4008ab:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4008af:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008b3:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008b8:	48 89 ce             	mov    %rcx,%rsi
  4008bb:	48 89 c7             	mov    %rax,%rdi
  4008be:	e8 b3 00 00 00       	callq  400976 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  4008c3:	90                   	nop
  4008c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008cf:	00 00 
  4008d1:	74 05                	je     4008d8 <_ZN9TestClass15increase_field1Ev+0x48>
  4008d3:	e8 b8 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008d8:	c9                   	leaveq 
  4008d9:	c3                   	retq   

00000000004008da <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4008da:	55                   	push   %rbp
  4008db:	48 89 e5             	mov    %rsp,%rbp
  4008de:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e2:	90                   	nop
  4008e3:	5d                   	pop    %rbp
  4008e4:	c3                   	retq   
  4008e5:	90                   	nop

00000000004008e6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEC1Ev>:
  4008e6:	55                   	push   %rbp
  4008e7:	48 89 e5             	mov    %rsp,%rbp
  4008ea:	48 83 ec 10          	sub    $0x10,%rsp
  4008ee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f6:	48 89 c7             	mov    %rax,%rdi
  4008f9:	e8 dc ff ff ff       	callq  4008da <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008fe:	90                   	nop
  4008ff:	c9                   	leaveq 
  400900:	c3                   	retq   
  400901:	90                   	nop

0000000000400902 <_ZN9TestClassC1Ev>:
  400902:	55                   	push   %rbp
  400903:	48 89 e5             	mov    %rsp,%rbp
  400906:	48 83 ec 10          	sub    $0x10,%rsp
  40090a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40090e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400912:	48 89 c7             	mov    %rax,%rdi
  400915:	e8 cc ff ff ff       	callq  4008e6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEC1Ev>
  40091a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 e0 00 00 00       	callq  400a06 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  400926:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40092a:	48 89 c7             	mov    %rax,%rdi
  40092d:	e8 e0 00 00 00       	callq  400a12 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  400932:	90                   	nop
  400933:	c9                   	leaveq 
  400934:	c3                   	retq   
  400935:	90                   	nop

0000000000400936 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400936:	55                   	push   %rbp
  400937:	48 89 e5             	mov    %rsp,%rbp
  40093a:	48 83 ec 20          	sub    $0x20,%rsp
  40093e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400942:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400946:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400949:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40094d:	48 89 c7             	mov    %rax,%rdi
  400950:	e8 87 01 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400955:	8b 10                	mov    (%rax),%edx
  400957:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40095b:	89 10                	mov    %edx,(%rax)
  40095d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400961:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400965:	48 89 d6             	mov    %rdx,%rsi
  400968:	48 89 c7             	mov    %rax,%rdi
  40096b:	e8 88 01 00 00       	callq  400af8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400970:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400974:	c9                   	leaveq 
  400975:	c3                   	retq   

0000000000400976 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400976:	55                   	push   %rbp
  400977:	48 89 e5             	mov    %rsp,%rbp
  40097a:	48 83 ec 20          	sub    $0x20,%rsp
  40097e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400982:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400986:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400989:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40098d:	48 89 c7             	mov    %rax,%rdi
  400990:	e8 73 01 00 00       	callq  400b08 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400995:	8b 10                	mov    (%rax),%edx
  400997:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40099b:	89 10                	mov    %edx,(%rax)
  40099d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009a1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a5:	48 89 d6             	mov    %rdx,%rsi
  4009a8:	48 89 c7             	mov    %rax,%rdi
  4009ab:	e8 7c 01 00 00       	callq  400b2c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  4009b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009b4:	c9                   	leaveq 
  4009b5:	c3                   	retq   

00000000004009b6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC1Ev>:
  4009b6:	55                   	push   %rbp
  4009b7:	48 89 e5             	mov    %rsp,%rbp
  4009ba:	48 83 ec 10          	sub    $0x10,%rsp
  4009be:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c2:	bf a0 91 00 00       	mov    $0x91a0,%edi
  4009c7:	e8 f4 fa ff ff       	callq  4004c0 <malloc@plt>
  4009cc:	48 89 c2             	mov    %rax,%rdx
  4009cf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d3:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4009d7:	90                   	nop
  4009d8:	c9                   	leaveq 
  4009d9:	c3                   	retq   

00000000004009da <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>:
  4009da:	55                   	push   %rbp
  4009db:	48 89 e5             	mov    %rsp,%rbp
  4009de:	48 83 ec 10          	sub    $0x10,%rsp
  4009e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e6:	e8 50 01 00 00       	callq  400b3b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009eb:	8b 05 6f 16 20 00    	mov    0x20166f(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  4009f1:	8d 50 01             	lea    0x1(%rax),%edx
  4009f4:	89 15 66 16 20 00    	mov    %edx,0x201666(%rip)        # 602060 <_ZN9TestClass7storageE>
  4009fa:	89 c0                	mov    %eax,%eax
  4009fc:	48 89 c7             	mov    %rax,%rdi
  4009ff:	e8 3e 01 00 00       	callq  400b42 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>
  400a04:	c9                   	leaveq 
  400a05:	c3                   	retq   

0000000000400a06 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  400a06:	55                   	push   %rbp
  400a07:	48 89 e5             	mov    %rsp,%rbp
  400a0a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0e:	90                   	nop
  400a0f:	5d                   	pop    %rbp
  400a10:	c3                   	retq   
  400a11:	90                   	nop

0000000000400a12 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  400a12:	55                   	push   %rbp
  400a13:	48 89 e5             	mov    %rsp,%rbp
  400a16:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1a:	90                   	nop
  400a1b:	5d                   	pop    %rbp
  400a1c:	c3                   	retq   

0000000000400a1d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEdlEPv>:
  400a1d:	55                   	push   %rbp
  400a1e:	48 89 e5             	mov    %rsp,%rbp
  400a21:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a25:	90                   	nop
  400a26:	5d                   	pop    %rbp
  400a27:	c3                   	retq   

0000000000400a28 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a28:	55                   	push   %rbp
  400a29:	48 89 e5             	mov    %rsp,%rbp
  400a2c:	48 83 ec 20          	sub    $0x20,%rsp
  400a30:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a34:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a38:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a3b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a3f:	48 89 c7             	mov    %rax,%rdi
  400a42:	e8 95 00 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a47:	48 89 c2             	mov    %rax,%rdx
  400a4a:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a4d:	89 02                	mov    %eax,(%rdx)
  400a4f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a53:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a57:	48 89 d6             	mov    %rdx,%rsi
  400a5a:	48 89 c7             	mov    %rax,%rdi
  400a5d:	e8 96 00 00 00       	callq  400af8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a62:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a66:	c9                   	leaveq 
  400a67:	c3                   	retq   

0000000000400a68 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a68:	55                   	push   %rbp
  400a69:	48 89 e5             	mov    %rsp,%rbp
  400a6c:	48 83 ec 20          	sub    $0x20,%rsp
  400a70:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a74:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a78:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a7b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a7f:	48 89 c7             	mov    %rax,%rdi
  400a82:	e8 81 00 00 00       	callq  400b08 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a87:	48 89 c2             	mov    %rax,%rdx
  400a8a:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a8d:	89 02                	mov    %eax,(%rdx)
  400a8f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a93:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a97:	48 89 d6             	mov    %rdx,%rsi
  400a9a:	48 89 c7             	mov    %rax,%rdi
  400a9d:	e8 8a 00 00 00       	callq  400b2c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400aa2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa6:	c9                   	leaveq 
  400aa7:	c3                   	retq   

0000000000400aa8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400aa8:	55                   	push   %rbp
  400aa9:	48 89 e5             	mov    %rsp,%rbp
  400aac:	48 83 ec 10          	sub    $0x10,%rsp
  400ab0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab8:	48 89 c7             	mov    %rax,%rdi
  400abb:	e8 1c 00 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac0:	c9                   	leaveq 
  400ac1:	c3                   	retq   

0000000000400ac2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400ac2:	55                   	push   %rbp
  400ac3:	48 89 e5             	mov    %rsp,%rbp
  400ac6:	48 83 ec 10          	sub    $0x10,%rsp
  400aca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ace:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ad2:	48 89 c7             	mov    %rax,%rdi
  400ad5:	e8 2e 00 00 00       	callq  400b08 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ada:	c9                   	leaveq 
  400adb:	c3                   	retq   

0000000000400adc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400adc:	55                   	push   %rbp
  400add:	48 89 e5             	mov    %rsp,%rbp
  400ae0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae4:	48 8b 05 7d 15 20 00 	mov    0x20157d(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400aeb:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400aef:	48 c1 e2 02          	shl    $0x2,%rdx
  400af3:	48 01 d0             	add    %rdx,%rax
  400af6:	5d                   	pop    %rbp
  400af7:	c3                   	retq   

0000000000400af8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400af8:	55                   	push   %rbp
  400af9:	48 89 e5             	mov    %rsp,%rbp
  400afc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b00:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b04:	90                   	nop
  400b05:	5d                   	pop    %rbp
  400b06:	c3                   	retq   
  400b07:	90                   	nop

0000000000400b08 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b08:	55                   	push   %rbp
  400b09:	48 89 e5             	mov    %rsp,%rbp
  400b0c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b10:	48 8b 05 51 15 20 00 	mov    0x201551(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400b17:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400b1b:	48 c1 e2 02          	shl    $0x2,%rdx
  400b1f:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400b26:	48 01 d0             	add    %rdx,%rax
  400b29:	5d                   	pop    %rbp
  400b2a:	c3                   	retq   
  400b2b:	90                   	nop

0000000000400b2c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400b2c:	55                   	push   %rbp
  400b2d:	48 89 e5             	mov    %rsp,%rbp
  400b30:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b34:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b38:	90                   	nop
  400b39:	5d                   	pop    %rbp
  400b3a:	c3                   	retq   

0000000000400b3b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b3b:	55                   	push   %rbp
  400b3c:	48 89 e5             	mov    %rsp,%rbp
  400b3f:	90                   	nop
  400b40:	5d                   	pop    %rbp
  400b41:	c3                   	retq   

0000000000400b42 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>:
  400b42:	55                   	push   %rbp
  400b43:	48 89 e5             	mov    %rsp,%rbp
  400b46:	48 83 ec 10          	sub    $0x10,%rsp
  400b4a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b4e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b52:	48 89 c7             	mov    %rax,%rdi
  400b55:	e8 02 00 00 00       	callq  400b5c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b5a:	c9                   	leaveq 
  400b5b:	c3                   	retq   

0000000000400b5c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b5c:	55                   	push   %rbp
  400b5d:	48 89 e5             	mov    %rsp,%rbp
  400b60:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b64:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b68:	5d                   	pop    %rbp
  400b69:	c3                   	retq   
  400b6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b70 <__libc_csu_init>:
  400b70:	41 57                	push   %r15
  400b72:	41 56                	push   %r14
  400b74:	41 89 ff             	mov    %edi,%r15d
  400b77:	41 55                	push   %r13
  400b79:	41 54                	push   %r12
  400b7b:	4c 8d 25 86 12 20 00 	lea    0x201286(%rip),%r12        # 601e08 <__frame_dummy_init_array_entry>
  400b82:	55                   	push   %rbp
  400b83:	48 8d 2d 8e 12 20 00 	lea    0x20128e(%rip),%rbp        # 601e18 <__init_array_end>
  400b8a:	53                   	push   %rbx
  400b8b:	49 89 f6             	mov    %rsi,%r14
  400b8e:	49 89 d5             	mov    %rdx,%r13
  400b91:	4c 29 e5             	sub    %r12,%rbp
  400b94:	48 83 ec 08          	sub    $0x8,%rsp
  400b98:	48 c1 fd 03          	sar    $0x3,%rbp
  400b9c:	e8 bf f8 ff ff       	callq  400460 <_init>
  400ba1:	48 85 ed             	test   %rbp,%rbp
  400ba4:	74 20                	je     400bc6 <__libc_csu_init+0x56>
  400ba6:	31 db                	xor    %ebx,%ebx
  400ba8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400baf:	00 
  400bb0:	4c 89 ea             	mov    %r13,%rdx
  400bb3:	4c 89 f6             	mov    %r14,%rsi
  400bb6:	44 89 ff             	mov    %r15d,%edi
  400bb9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400bbd:	48 83 c3 01          	add    $0x1,%rbx
  400bc1:	48 39 eb             	cmp    %rbp,%rbx
  400bc4:	75 ea                	jne    400bb0 <__libc_csu_init+0x40>
  400bc6:	48 83 c4 08          	add    $0x8,%rsp
  400bca:	5b                   	pop    %rbx
  400bcb:	5d                   	pop    %rbp
  400bcc:	41 5c                	pop    %r12
  400bce:	41 5d                	pop    %r13
  400bd0:	41 5e                	pop    %r14
  400bd2:	41 5f                	pop    %r15
  400bd4:	c3                   	retq   
  400bd5:	90                   	nop
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 

0000000000400be0 <__libc_csu_fini>:
  400be0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400be4 <_fini>:
  400be4:	48 83 ec 08          	sub    $0x8,%rsp
  400be8:	48 83 c4 08          	add    $0x8,%rsp
  400bec:	c3                   	retq   
