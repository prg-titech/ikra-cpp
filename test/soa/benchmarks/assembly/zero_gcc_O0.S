
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
  4004ef:	49 c7 c0 80 0b 40 00 	mov    $0x400b80,%r8
  4004f6:	48 c7 c1 10 0b 40 00 	mov    $0x400b10,%rcx
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
  4005f3:	e8 7c 03 00 00       	callq  400974 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  4005f8:	48 89 c3             	mov    %rax,%rbx
  4005fb:	48 89 df             	mov    %rbx,%rdi
  4005fe:	e8 99 02 00 00       	callq  40089c <_ZN9TestClassC1Ev>
  400603:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  400607:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  40060b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40060f:	ba 77 77 00 00       	mov    $0x7777,%edx
  400614:	48 89 ce             	mov    %rcx,%rsi
  400617:	48 89 c7             	mov    %rax,%rdi
  40061a:	e8 a3 03 00 00       	callq  4009c2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40061f:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400623:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400627:	ba 88 88 00 00       	mov    $0x8888,%edx
  40062c:	48 89 ce             	mov    %rcx,%rsi
  40062f:	48 89 c7             	mov    %rax,%rdi
  400632:	e8 cb 03 00 00       	callq  400a02 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400637:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40063b:	48 89 c7             	mov    %rax,%rdi
  40063e:	e8 9d 01 00 00       	callq  4007e0 <_ZN9TestClass15increase_field0Ev>
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 89 c7             	mov    %rax,%rdi
  40064a:	e8 db 01 00 00       	callq  40082a <_ZN9TestClass15increase_field1Ev>
  40064f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400653:	48 89 c7             	mov    %rax,%rdi
  400656:	e8 01 04 00 00       	callq  400a5c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40065b:	8b 18                	mov    (%rax),%ebx
  40065d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400661:	48 89 c7             	mov    %rax,%rdi
  400664:	e8 d9 03 00 00       	callq  400a42 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400669:	8b 00                	mov    (%rax),%eax
  40066b:	89 da                	mov    %ebx,%edx
  40066d:	89 c6                	mov    %eax,%esi
  40066f:	bf a0 0b 40 00       	mov    $0x400ba0,%edi
  400674:	b8 00 00 00 00       	mov    $0x0,%eax
  400679:	e8 22 fe ff ff       	callq  4004a0 <printf@plt>
  40067e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400682:	48 89 c7             	mov    %rax,%rdi
  400685:	e8 b8 03 00 00       	callq  400a42 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40068a:	8b 00                	mov    (%rax),%eax
  40068c:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400691:	75 15                	jne    4006a8 <main+0xd2>
  400693:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400697:	48 89 c7             	mov    %rax,%rdi
  40069a:	e8 bd 03 00 00       	callq  400a5c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
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
  4006e0:	e8 8f 02 00 00       	callq  400974 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  4006e5:	48 89 c3             	mov    %rax,%rbx
  4006e8:	48 89 df             	mov    %rbx,%rdi
  4006eb:	e8 ac 01 00 00       	callq  40089c <_ZN9TestClassC1Ev>
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
  400728:	e8 95 02 00 00       	callq  4009c2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
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
  400757:	e8 e6 02 00 00       	callq  400a42 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40075c:	8b 00                	mov    (%rax),%eax
  40075e:	c9                   	leaveq 
  40075f:	c3                   	retq   

0000000000400760 <_Z21explicit_write_field0m>:
  400760:	55                   	push   %rbp
  400761:	48 89 e5             	mov    %rsp,%rbp
  400764:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400768:	48 8b 05 01 19 20 00 	mov    0x201901(%rip),%rax        # 602070 <explicit_field0>
  40076f:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400773:	48 c1 e2 02          	shl    $0x2,%rdx
  400777:	48 01 d0             	add    %rdx,%rax
  40077a:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  400780:	90                   	nop
  400781:	5d                   	pop    %rbp
  400782:	c3                   	retq   

0000000000400783 <_Z20explicit_read_field0m>:
  400783:	55                   	push   %rbp
  400784:	48 89 e5             	mov    %rsp,%rbp
  400787:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40078b:	48 8b 05 de 18 20 00 	mov    0x2018de(%rip),%rax        # 602070 <explicit_field0>
  400792:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400796:	48 c1 e2 02          	shl    $0x2,%rdx
  40079a:	48 01 d0             	add    %rdx,%rax
  40079d:	8b 00                	mov    (%rax),%eax
  40079f:	5d                   	pop    %rbp
  4007a0:	c3                   	retq   

00000000004007a1 <_Z41__static_initialization_and_destruction_0ii>:
  4007a1:	55                   	push   %rbp
  4007a2:	48 89 e5             	mov    %rsp,%rbp
  4007a5:	48 83 ec 10          	sub    $0x10,%rsp
  4007a9:	89 7d fc             	mov    %edi,-0x4(%rbp)
  4007ac:	89 75 f8             	mov    %esi,-0x8(%rbp)
  4007af:	83 7d fc 01          	cmpl   $0x1,-0x4(%rbp)
  4007b3:	75 13                	jne    4007c8 <_Z41__static_initialization_and_destruction_0ii+0x27>
  4007b5:	81 7d f8 ff ff 00 00 	cmpl   $0xffff,-0x8(%rbp)
  4007bc:	75 0a                	jne    4007c8 <_Z41__static_initialization_and_destruction_0ii+0x27>
  4007be:	bf 60 20 60 00       	mov    $0x602060,%edi
  4007c3:	e8 88 01 00 00       	callq  400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC1Ev>
  4007c8:	90                   	nop
  4007c9:	c9                   	leaveq 
  4007ca:	c3                   	retq   

00000000004007cb <_GLOBAL__sub_I__ZN9TestClass7storageE>:
  4007cb:	55                   	push   %rbp
  4007cc:	48 89 e5             	mov    %rsp,%rbp
  4007cf:	be ff ff 00 00       	mov    $0xffff,%esi
  4007d4:	bf 01 00 00 00       	mov    $0x1,%edi
  4007d9:	e8 c3 ff ff ff       	callq  4007a1 <_Z41__static_initialization_and_destruction_0ii>
  4007de:	5d                   	pop    %rbp
  4007df:	c3                   	retq   

00000000004007e0 <_ZN9TestClass15increase_field0Ev>:
  4007e0:	55                   	push   %rbp
  4007e1:	48 89 e5             	mov    %rsp,%rbp
  4007e4:	48 83 ec 30          	sub    $0x30,%rsp
  4007e8:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007ec:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007f3:	00 00 
  4007f5:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007f9:	31 c0                	xor    %eax,%eax
  4007fb:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4007ff:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400803:	ba 55 55 00 00       	mov    $0x5555,%edx
  400808:	48 89 ce             	mov    %rcx,%rsi
  40080b:	48 89 c7             	mov    %rax,%rdi
  40080e:	e8 bd 00 00 00       	callq  4008d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400813:	90                   	nop
  400814:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400818:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40081f:	00 00 
  400821:	74 05                	je     400828 <_ZN9TestClass15increase_field0Ev+0x48>
  400823:	e8 68 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400828:	c9                   	leaveq 
  400829:	c3                   	retq   

000000000040082a <_ZN9TestClass15increase_field1Ev>:
  40082a:	55                   	push   %rbp
  40082b:	48 89 e5             	mov    %rsp,%rbp
  40082e:	48 83 ec 30          	sub    $0x30,%rsp
  400832:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400836:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40083d:	00 00 
  40083f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400843:	31 c0                	xor    %eax,%eax
  400845:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400849:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40084d:	ba 44 44 00 00       	mov    $0x4444,%edx
  400852:	48 89 ce             	mov    %rcx,%rsi
  400855:	48 89 c7             	mov    %rax,%rdi
  400858:	e8 b3 00 00 00       	callq  400910 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  40085d:	90                   	nop
  40085e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400862:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400869:	00 00 
  40086b:	74 05                	je     400872 <_ZN9TestClass15increase_field1Ev+0x48>
  40086d:	e8 1e fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400872:	c9                   	leaveq 
  400873:	c3                   	retq   

0000000000400874 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400874:	55                   	push   %rbp
  400875:	48 89 e5             	mov    %rsp,%rbp
  400878:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40087c:	90                   	nop
  40087d:	5d                   	pop    %rbp
  40087e:	c3                   	retq   
  40087f:	90                   	nop

0000000000400880 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEC1Ev>:
  400880:	55                   	push   %rbp
  400881:	48 89 e5             	mov    %rsp,%rbp
  400884:	48 83 ec 10          	sub    $0x10,%rsp
  400888:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40088c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400890:	48 89 c7             	mov    %rax,%rdi
  400893:	e8 dc ff ff ff       	callq  400874 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400898:	90                   	nop
  400899:	c9                   	leaveq 
  40089a:	c3                   	retq   
  40089b:	90                   	nop

000000000040089c <_ZN9TestClassC1Ev>:
  40089c:	55                   	push   %rbp
  40089d:	48 89 e5             	mov    %rsp,%rbp
  4008a0:	48 83 ec 10          	sub    $0x10,%rsp
  4008a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ac:	48 89 c7             	mov    %rax,%rdi
  4008af:	e8 cc ff ff ff       	callq  400880 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEC1Ev>
  4008b4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b8:	48 89 c7             	mov    %rax,%rdi
  4008bb:	e8 e0 00 00 00       	callq  4009a0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008c0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c4:	48 89 c7             	mov    %rax,%rdi
  4008c7:	e8 e0 00 00 00       	callq  4009ac <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008cc:	90                   	nop
  4008cd:	c9                   	leaveq 
  4008ce:	c3                   	retq   
  4008cf:	90                   	nop

00000000004008d0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 20          	sub    $0x20,%rsp
  4008d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008dc:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008e0:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008e3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008e7:	48 89 c7             	mov    %rax,%rdi
  4008ea:	e8 87 01 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008ef:	8b 10                	mov    (%rax),%edx
  4008f1:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008f5:	89 10                	mov    %edx,(%rax)
  4008f7:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008fb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ff:	48 89 d6             	mov    %rdx,%rsi
  400902:	48 89 c7             	mov    %rax,%rdi
  400905:	e8 88 01 00 00       	callq  400a92 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  40090a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40090e:	c9                   	leaveq 
  40090f:	c3                   	retq   

0000000000400910 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 20          	sub    $0x20,%rsp
  400918:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40091c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400920:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400923:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400927:	48 89 c7             	mov    %rax,%rdi
  40092a:	e8 73 01 00 00       	callq  400aa2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40092f:	8b 10                	mov    (%rax),%edx
  400931:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400935:	89 10                	mov    %edx,(%rax)
  400937:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40093b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40093f:	48 89 d6             	mov    %rdx,%rsi
  400942:	48 89 c7             	mov    %rax,%rdi
  400945:	e8 7c 01 00 00       	callq  400ac6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  40094a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40094e:	c9                   	leaveq 
  40094f:	c3                   	retq   

0000000000400950 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC1Ev>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 83 ec 10          	sub    $0x10,%rsp
  400958:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40095c:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400961:	e8 5a fb ff ff       	callq  4004c0 <malloc@plt>
  400966:	48 89 c2             	mov    %rax,%rdx
  400969:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096d:	48 89 50 08          	mov    %rdx,0x8(%rax)
  400971:	90                   	nop
  400972:	c9                   	leaveq 
  400973:	c3                   	retq   

0000000000400974 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>:
  400974:	55                   	push   %rbp
  400975:	48 89 e5             	mov    %rsp,%rbp
  400978:	48 83 ec 10          	sub    $0x10,%rsp
  40097c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400980:	e8 50 01 00 00       	callq  400ad5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400985:	8b 05 d5 16 20 00    	mov    0x2016d5(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  40098b:	8d 50 01             	lea    0x1(%rax),%edx
  40098e:	89 15 cc 16 20 00    	mov    %edx,0x2016cc(%rip)        # 602060 <_ZN9TestClass7storageE>
  400994:	89 c0                	mov    %eax,%eax
  400996:	48 89 c7             	mov    %rax,%rdi
  400999:	e8 3e 01 00 00       	callq  400adc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>
  40099e:	c9                   	leaveq 
  40099f:	c3                   	retq   

00000000004009a0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a8:	90                   	nop
  4009a9:	5d                   	pop    %rbp
  4009aa:	c3                   	retq   
  4009ab:	90                   	nop

00000000004009ac <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009ac:	55                   	push   %rbp
  4009ad:	48 89 e5             	mov    %rsp,%rbp
  4009b0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b4:	90                   	nop
  4009b5:	5d                   	pop    %rbp
  4009b6:	c3                   	retq   

00000000004009b7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEdlEPv>:
  4009b7:	55                   	push   %rbp
  4009b8:	48 89 e5             	mov    %rsp,%rbp
  4009bb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bf:	90                   	nop
  4009c0:	5d                   	pop    %rbp
  4009c1:	c3                   	retq   

00000000004009c2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009c2:	55                   	push   %rbp
  4009c3:	48 89 e5             	mov    %rsp,%rbp
  4009c6:	48 83 ec 20          	sub    $0x20,%rsp
  4009ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ce:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009d2:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009d5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009d9:	48 89 c7             	mov    %rax,%rdi
  4009dc:	e8 95 00 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009e1:	48 89 c2             	mov    %rax,%rdx
  4009e4:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4009e7:	89 02                	mov    %eax,(%rdx)
  4009e9:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009ed:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009f1:	48 89 d6             	mov    %rdx,%rsi
  4009f4:	48 89 c7             	mov    %rax,%rdi
  4009f7:	e8 96 00 00 00       	callq  400a92 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4009fc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a00:	c9                   	leaveq 
  400a01:	c3                   	retq   

0000000000400a02 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a02:	55                   	push   %rbp
  400a03:	48 89 e5             	mov    %rsp,%rbp
  400a06:	48 83 ec 20          	sub    $0x20,%rsp
  400a0a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a12:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a15:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a19:	48 89 c7             	mov    %rax,%rdi
  400a1c:	e8 81 00 00 00       	callq  400aa2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a21:	48 89 c2             	mov    %rax,%rdx
  400a24:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a27:	89 02                	mov    %eax,(%rdx)
  400a29:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a2d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a31:	48 89 d6             	mov    %rdx,%rsi
  400a34:	48 89 c7             	mov    %rax,%rdi
  400a37:	e8 8a 00 00 00       	callq  400ac6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a3c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a40:	c9                   	leaveq 
  400a41:	c3                   	retq   

0000000000400a42 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a42:	55                   	push   %rbp
  400a43:	48 89 e5             	mov    %rsp,%rbp
  400a46:	48 83 ec 10          	sub    $0x10,%rsp
  400a4a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a4e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a52:	48 89 c7             	mov    %rax,%rdi
  400a55:	e8 1c 00 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a5a:	c9                   	leaveq 
  400a5b:	c3                   	retq   

0000000000400a5c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a5c:	55                   	push   %rbp
  400a5d:	48 89 e5             	mov    %rsp,%rbp
  400a60:	48 83 ec 10          	sub    $0x10,%rsp
  400a64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a68:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a6c:	48 89 c7             	mov    %rax,%rdi
  400a6f:	e8 2e 00 00 00       	callq  400aa2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a74:	c9                   	leaveq 
  400a75:	c3                   	retq   

0000000000400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a76:	55                   	push   %rbp
  400a77:	48 89 e5             	mov    %rsp,%rbp
  400a7a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a7e:	48 8b 05 e3 15 20 00 	mov    0x2015e3(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400a85:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400a89:	48 c1 e2 02          	shl    $0x2,%rdx
  400a8d:	48 01 d0             	add    %rdx,%rax
  400a90:	5d                   	pop    %rbp
  400a91:	c3                   	retq   

0000000000400a92 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a92:	55                   	push   %rbp
  400a93:	48 89 e5             	mov    %rsp,%rbp
  400a96:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a9a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a9e:	90                   	nop
  400a9f:	5d                   	pop    %rbp
  400aa0:	c3                   	retq   
  400aa1:	90                   	nop

0000000000400aa2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aa2:	55                   	push   %rbp
  400aa3:	48 89 e5             	mov    %rsp,%rbp
  400aa6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aaa:	48 8b 05 b7 15 20 00 	mov    0x2015b7(%rip),%rax        # 602068 <_ZN9TestClass7storageE+0x8>
  400ab1:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400ab5:	48 c1 e2 02          	shl    $0x2,%rdx
  400ab9:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400ac0:	48 01 d0             	add    %rdx,%rax
  400ac3:	5d                   	pop    %rbp
  400ac4:	c3                   	retq   
  400ac5:	90                   	nop

0000000000400ac6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400ac6:	55                   	push   %rbp
  400ac7:	48 89 e5             	mov    %rsp,%rbp
  400aca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ace:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ad2:	90                   	nop
  400ad3:	5d                   	pop    %rbp
  400ad4:	c3                   	retq   

0000000000400ad5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ad5:	55                   	push   %rbp
  400ad6:	48 89 e5             	mov    %rsp,%rbp
  400ad9:	90                   	nop
  400ada:	5d                   	pop    %rbp
  400adb:	c3                   	retq   

0000000000400adc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>:
  400adc:	55                   	push   %rbp
  400add:	48 89 e5             	mov    %rsp,%rbp
  400ae0:	48 83 ec 10          	sub    $0x10,%rsp
  400ae4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aec:	48 89 c7             	mov    %rax,%rdi
  400aef:	e8 02 00 00 00       	callq  400af6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400af4:	c9                   	leaveq 
  400af5:	c3                   	retq   

0000000000400af6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400af6:	55                   	push   %rbp
  400af7:	48 89 e5             	mov    %rsp,%rbp
  400afa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b02:	5d                   	pop    %rbp
  400b03:	c3                   	retq   
  400b04:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0b:	00 00 00 
  400b0e:	66 90                	xchg   %ax,%ax

0000000000400b10 <__libc_csu_init>:
  400b10:	41 57                	push   %r15
  400b12:	41 56                	push   %r14
  400b14:	41 89 ff             	mov    %edi,%r15d
  400b17:	41 55                	push   %r13
  400b19:	41 54                	push   %r12
  400b1b:	4c 8d 25 e6 12 20 00 	lea    0x2012e6(%rip),%r12        # 601e08 <__frame_dummy_init_array_entry>
  400b22:	55                   	push   %rbp
  400b23:	48 8d 2d ee 12 20 00 	lea    0x2012ee(%rip),%rbp        # 601e18 <__init_array_end>
  400b2a:	53                   	push   %rbx
  400b2b:	49 89 f6             	mov    %rsi,%r14
  400b2e:	49 89 d5             	mov    %rdx,%r13
  400b31:	4c 29 e5             	sub    %r12,%rbp
  400b34:	48 83 ec 08          	sub    $0x8,%rsp
  400b38:	48 c1 fd 03          	sar    $0x3,%rbp
  400b3c:	e8 1f f9 ff ff       	callq  400460 <_init>
  400b41:	48 85 ed             	test   %rbp,%rbp
  400b44:	74 20                	je     400b66 <__libc_csu_init+0x56>
  400b46:	31 db                	xor    %ebx,%ebx
  400b48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b4f:	00 
  400b50:	4c 89 ea             	mov    %r13,%rdx
  400b53:	4c 89 f6             	mov    %r14,%rsi
  400b56:	44 89 ff             	mov    %r15d,%edi
  400b59:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400b5d:	48 83 c3 01          	add    $0x1,%rbx
  400b61:	48 39 eb             	cmp    %rbp,%rbx
  400b64:	75 ea                	jne    400b50 <__libc_csu_init+0x40>
  400b66:	48 83 c4 08          	add    $0x8,%rsp
  400b6a:	5b                   	pop    %rbx
  400b6b:	5d                   	pop    %rbp
  400b6c:	41 5c                	pop    %r12
  400b6e:	41 5d                	pop    %r13
  400b70:	41 5e                	pop    %r14
  400b72:	41 5f                	pop    %r15
  400b74:	c3                   	retq   
  400b75:	90                   	nop
  400b76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7d:	00 00 00 

0000000000400b80 <__libc_csu_fini>:
  400b80:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400b84 <_fini>:
  400b84:	48 83 ec 08          	sub    $0x8,%rsp
  400b88:	48 83 c4 08          	add    $0x8,%rsp
  400b8c:	c3                   	retq   
