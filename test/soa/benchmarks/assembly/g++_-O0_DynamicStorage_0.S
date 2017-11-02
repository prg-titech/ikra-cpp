
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
  4004ef:	49 c7 c0 40 0c 40 00 	mov    $0x400c40,%r8
  4004f6:	48 c7 c1 d0 0b 40 00 	mov    $0x400bd0,%rcx
  4004fd:	48 c7 c7 e1 05 40 00 	mov    $0x4005e1,%rdi
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
  4005d6:	55                   	push   %rbp
  4005d7:	48 89 e5             	mov    %rsp,%rbp
  4005da:	b8 80 20 60 00       	mov    $0x602080,%eax
  4005df:	5d                   	pop    %rbp
  4005e0:	c3                   	retq   

00000000004005e1 <main>:
  4005e1:	55                   	push   %rbp
  4005e2:	48 89 e5             	mov    %rsp,%rbp
  4005e5:	53                   	push   %rbx
  4005e6:	48 83 ec 28          	sub    $0x28,%rsp
  4005ea:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005f1:	00 00 
  4005f3:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005f7:	31 c0                	xor    %eax,%eax
  4005f9:	e8 8a 03 00 00       	callq  400988 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageEv>
  4005fe:	bf 00 00 00 00       	mov    $0x0,%edi
  400603:	e8 a6 03 00 00       	callq  4009ae <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  400608:	48 89 c3             	mov    %rax,%rbx
  40060b:	48 89 df             	mov    %rbx,%rdi
  40060e:	e8 c1 02 00 00       	callq  4008d4 <_ZN9TestClassC1Ev>
  400613:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  400617:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  40061b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40061f:	ba 77 77 00 00       	mov    $0x7777,%edx
  400624:	48 89 ce             	mov    %rcx,%rsi
  400627:	48 89 c7             	mov    %rax,%rdi
  40062a:	e8 cf 03 00 00       	callq  4009fe <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40062f:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  400633:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400637:	ba 88 88 00 00       	mov    $0x8888,%edx
  40063c:	48 89 ce             	mov    %rcx,%rsi
  40063f:	48 89 c7             	mov    %rax,%rdi
  400642:	e8 f7 03 00 00       	callq  400a3e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400647:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064b:	48 89 c7             	mov    %rax,%rdi
  40064e:	e8 c5 01 00 00       	callq  400818 <_ZN9TestClass15increase_field0Ev>
  400653:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400657:	48 89 c7             	mov    %rax,%rdi
  40065a:	e8 03 02 00 00       	callq  400862 <_ZN9TestClass15increase_field1Ev>
  40065f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400663:	48 89 c7             	mov    %rax,%rdi
  400666:	e8 2d 04 00 00       	callq  400a98 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40066b:	8b 18                	mov    (%rax),%ebx
  40066d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400671:	48 89 c7             	mov    %rax,%rdi
  400674:	e8 05 04 00 00       	callq  400a7e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400679:	8b 00                	mov    (%rax),%eax
  40067b:	89 da                	mov    %ebx,%edx
  40067d:	89 c6                	mov    %eax,%esi
  40067f:	bf 64 0c 40 00       	mov    $0x400c64,%edi
  400684:	b8 00 00 00 00       	mov    $0x0,%eax
  400689:	e8 12 fe ff ff       	callq  4004a0 <printf@plt>
  40068e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400692:	48 89 c7             	mov    %rax,%rdi
  400695:	e8 e4 03 00 00       	callq  400a7e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40069a:	8b 00                	mov    (%rax),%eax
  40069c:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006a1:	75 15                	jne    4006b8 <main+0xd7>
  4006a3:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4006a7:	48 89 c7             	mov    %rax,%rdi
  4006aa:	e8 e9 03 00 00       	callq  400a98 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4006af:	8b 00                	mov    (%rax),%eax
  4006b1:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006b6:	74 07                	je     4006bf <main+0xde>
  4006b8:	b8 01 00 00 00       	mov    $0x1,%eax
  4006bd:	eb 05                	jmp    4006c4 <main+0xe3>
  4006bf:	b8 00 00 00 00       	mov    $0x0,%eax
  4006c4:	0f b6 c0             	movzbl %al,%eax
  4006c7:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006cb:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006d2:	00 00 
  4006d4:	74 05                	je     4006db <main+0xfa>
  4006d6:	e8 b5 fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  4006db:	48 83 c4 28          	add    $0x28,%rsp
  4006df:	5b                   	pop    %rbx
  4006e0:	5d                   	pop    %rbp
  4006e1:	c3                   	retq   

00000000004006e2 <_Z12new_instancev>:
  4006e2:	55                   	push   %rbp
  4006e3:	48 89 e5             	mov    %rsp,%rbp
  4006e6:	53                   	push   %rbx
  4006e7:	48 83 ec 08          	sub    $0x8,%rsp
  4006eb:	bf 00 00 00 00       	mov    $0x0,%edi
  4006f0:	e8 b9 02 00 00       	callq  4009ae <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006f5:	48 89 c3             	mov    %rax,%rbx
  4006f8:	48 89 df             	mov    %rbx,%rdi
  4006fb:	e8 d4 01 00 00       	callq  4008d4 <_ZN9TestClassC1Ev>
  400700:	48 89 d8             	mov    %rbx,%rax
  400703:	48 83 c4 08          	add    $0x8,%rsp
  400707:	5b                   	pop    %rbx
  400708:	5d                   	pop    %rbp
  400709:	c3                   	retq   

000000000040070a <_Z12write_field0P9TestClass>:
  40070a:	55                   	push   %rbp
  40070b:	48 89 e5             	mov    %rsp,%rbp
  40070e:	48 83 ec 30          	sub    $0x30,%rsp
  400712:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400716:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40071d:	00 00 
  40071f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400723:	31 c0                	xor    %eax,%eax
  400725:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400729:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40072d:	ba 77 77 00 00       	mov    $0x7777,%edx
  400732:	48 89 ce             	mov    %rcx,%rsi
  400735:	48 89 c7             	mov    %rax,%rdi
  400738:	e8 c1 02 00 00       	callq  4009fe <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40073d:	90                   	nop
  40073e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400742:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400749:	00 00 
  40074b:	74 05                	je     400752 <_Z12write_field0P9TestClass+0x48>
  40074d:	e8 3e fd ff ff       	callq  400490 <__stack_chk_fail@plt>
  400752:	c9                   	leaveq 
  400753:	c3                   	retq   

0000000000400754 <_Z11read_field0P9TestClass>:
  400754:	55                   	push   %rbp
  400755:	48 89 e5             	mov    %rsp,%rbp
  400758:	48 83 ec 10          	sub    $0x10,%rsp
  40075c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400760:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400764:	48 89 c7             	mov    %rax,%rdi
  400767:	e8 12 03 00 00       	callq  400a7e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40076c:	8b 00                	mov    (%rax),%eax
  40076e:	c9                   	leaveq 
  40076f:	c3                   	retq   

0000000000400770 <_Z12write_field1P9TestClass>:
  400770:	55                   	push   %rbp
  400771:	48 89 e5             	mov    %rsp,%rbp
  400774:	48 83 ec 30          	sub    $0x30,%rsp
  400778:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40077c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400783:	00 00 
  400785:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400789:	31 c0                	xor    %eax,%eax
  40078b:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40078f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400793:	ba 77 77 00 00       	mov    $0x7777,%edx
  400798:	48 89 ce             	mov    %rcx,%rsi
  40079b:	48 89 c7             	mov    %rax,%rdi
  40079e:	e8 9b 02 00 00       	callq  400a3e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007a3:	90                   	nop
  4007a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007a8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007af:	00 00 
  4007b1:	74 05                	je     4007b8 <_Z12write_field1P9TestClass+0x48>
  4007b3:	e8 d8 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  4007b8:	c9                   	leaveq 
  4007b9:	c3                   	retq   

00000000004007ba <_Z11read_field1P9TestClass>:
  4007ba:	55                   	push   %rbp
  4007bb:	48 89 e5             	mov    %rsp,%rbp
  4007be:	48 83 ec 10          	sub    $0x10,%rsp
  4007c2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007ca:	48 89 c7             	mov    %rax,%rdi
  4007cd:	e8 c6 02 00 00       	callq  400a98 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007d2:	8b 00                	mov    (%rax),%eax
  4007d4:	c9                   	leaveq 
  4007d5:	c3                   	retq   

00000000004007d6 <_Z21explicit_write_field0m>:
  4007d6:	55                   	push   %rbp
  4007d7:	48 89 e5             	mov    %rsp,%rbp
  4007da:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007de:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007e2:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  4007e9:	77 77 00 00 
  4007ed:	90                   	nop
  4007ee:	5d                   	pop    %rbp
  4007ef:	c3                   	retq   

00000000004007f0 <_Z20explicit_read_field0m>:
  4007f0:	55                   	push   %rbp
  4007f1:	48 89 e5             	mov    %rsp,%rbp
  4007f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007fc:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  400803:	5d                   	pop    %rbp
  400804:	c3                   	retq   

0000000000400805 <_ZnwmPv>:
  400805:	55                   	push   %rbp
  400806:	48 89 e5             	mov    %rsp,%rbp
  400809:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40080d:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400811:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400815:	5d                   	pop    %rbp
  400816:	c3                   	retq   
  400817:	90                   	nop

0000000000400818 <_ZN9TestClass15increase_field0Ev>:
  400818:	55                   	push   %rbp
  400819:	48 89 e5             	mov    %rsp,%rbp
  40081c:	48 83 ec 30          	sub    $0x30,%rsp
  400820:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400824:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40082b:	00 00 
  40082d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400831:	31 c0                	xor    %eax,%eax
  400833:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400837:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40083b:	ba 55 55 00 00       	mov    $0x5555,%edx
  400840:	48 89 ce             	mov    %rcx,%rsi
  400843:	48 89 c7             	mov    %rax,%rdi
  400846:	e8 bd 00 00 00       	callq  400908 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40084b:	90                   	nop
  40084c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400850:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400857:	00 00 
  400859:	74 05                	je     400860 <_ZN9TestClass15increase_field0Ev+0x48>
  40085b:	e8 30 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400860:	c9                   	leaveq 
  400861:	c3                   	retq   

0000000000400862 <_ZN9TestClass15increase_field1Ev>:
  400862:	55                   	push   %rbp
  400863:	48 89 e5             	mov    %rsp,%rbp
  400866:	48 83 ec 30          	sub    $0x30,%rsp
  40086a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40086e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400875:	00 00 
  400877:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40087b:	31 c0                	xor    %eax,%eax
  40087d:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400881:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400885:	ba 44 44 00 00       	mov    $0x4444,%edx
  40088a:	48 89 ce             	mov    %rcx,%rsi
  40088d:	48 89 c7             	mov    %rax,%rdi
  400890:	e8 b3 00 00 00       	callq  400948 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400895:	90                   	nop
  400896:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40089a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008a1:	00 00 
  4008a3:	74 05                	je     4008aa <_ZN9TestClass15increase_field1Ev+0x48>
  4008a5:	e8 e6 fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008aa:	c9                   	leaveq 
  4008ab:	c3                   	retq   

00000000004008ac <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4008ac:	55                   	push   %rbp
  4008ad:	48 89 e5             	mov    %rsp,%rbp
  4008b0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b4:	90                   	nop
  4008b5:	5d                   	pop    %rbp
  4008b6:	c3                   	retq   
  4008b7:	90                   	nop

00000000004008b8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  4008b8:	55                   	push   %rbp
  4008b9:	48 89 e5             	mov    %rsp,%rbp
  4008bc:	48 83 ec 10          	sub    $0x10,%rsp
  4008c0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c8:	48 89 c7             	mov    %rax,%rdi
  4008cb:	e8 dc ff ff ff       	callq  4008ac <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008d0:	90                   	nop
  4008d1:	c9                   	leaveq 
  4008d2:	c3                   	retq   
  4008d3:	90                   	nop

00000000004008d4 <_ZN9TestClassC1Ev>:
  4008d4:	55                   	push   %rbp
  4008d5:	48 89 e5             	mov    %rsp,%rbp
  4008d8:	48 83 ec 10          	sub    $0x10,%rsp
  4008dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e4:	48 89 c7             	mov    %rax,%rdi
  4008e7:	e8 cc ff ff ff       	callq  4008b8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  4008ec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f0:	48 89 c7             	mov    %rax,%rdi
  4008f3:	e8 e4 00 00 00       	callq  4009dc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008fc:	48 89 c7             	mov    %rax,%rdi
  4008ff:	e8 e4 00 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  400904:	90                   	nop
  400905:	c9                   	leaveq 
  400906:	c3                   	retq   
  400907:	90                   	nop

0000000000400908 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400908:	55                   	push   %rbp
  400909:	48 89 e5             	mov    %rsp,%rbp
  40090c:	48 83 ec 20          	sub    $0x20,%rsp
  400910:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400914:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400918:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40091b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40091f:	48 89 c7             	mov    %rax,%rdi
  400922:	e8 8b 01 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400927:	8b 10                	mov    (%rax),%edx
  400929:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40092d:	89 10                	mov    %edx,(%rax)
  40092f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400933:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400937:	48 89 d6             	mov    %rdx,%rsi
  40093a:	48 89 c7             	mov    %rax,%rdi
  40093d:	e8 8a 01 00 00       	callq  400acc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400942:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400946:	c9                   	leaveq 
  400947:	c3                   	retq   

0000000000400948 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400948:	55                   	push   %rbp
  400949:	48 89 e5             	mov    %rsp,%rbp
  40094c:	48 83 ec 20          	sub    $0x20,%rsp
  400950:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400954:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400958:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40095b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40095f:	48 89 c7             	mov    %rax,%rdi
  400962:	e8 75 01 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400967:	8b 10                	mov    (%rax),%edx
  400969:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40096d:	89 10                	mov    %edx,(%rax)
  40096f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400973:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400977:	48 89 d6             	mov    %rdx,%rsi
  40097a:	48 89 c7             	mov    %rax,%rdi
  40097d:	e8 74 01 00 00       	callq  400af6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400982:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400986:	c9                   	leaveq 
  400987:	c3                   	retq   

0000000000400988 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageEv>:
  400988:	55                   	push   %rbp
  400989:	48 89 e5             	mov    %rsp,%rbp
  40098c:	e8 45 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400991:	48 89 c6             	mov    %rax,%rsi
  400994:	bf 20 00 00 00       	mov    $0x20,%edi
  400999:	e8 67 fe ff ff       	callq  400805 <_ZnwmPv>
  40099e:	48 85 c0             	test   %rax,%rax
  4009a1:	74 08                	je     4009ab <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageEv+0x23>
  4009a3:	48 89 c7             	mov    %rax,%rdi
  4009a6:	e8 5b 01 00 00       	callq  400b06 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  4009ab:	90                   	nop
  4009ac:	5d                   	pop    %rbp
  4009ad:	c3                   	retq   

00000000004009ae <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009ae:	55                   	push   %rbp
  4009af:	48 89 e5             	mov    %rsp,%rbp
  4009b2:	48 83 ec 10          	sub    $0x10,%rsp
  4009b6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ba:	e8 87 01 00 00       	callq  400b46 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009bf:	e8 12 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009c4:	48 89 c2             	mov    %rax,%rdx
  4009c7:	48 8b 02             	mov    (%rdx),%rax
  4009ca:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4009ce:	48 89 0a             	mov    %rcx,(%rdx)
  4009d1:	48 89 c7             	mov    %rax,%rdi
  4009d4:	e8 74 01 00 00       	callq  400b4d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009d9:	c9                   	leaveq 
  4009da:	c3                   	retq   
  4009db:	90                   	nop

00000000004009dc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4009dc:	55                   	push   %rbp
  4009dd:	48 89 e5             	mov    %rsp,%rbp
  4009e0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e4:	90                   	nop
  4009e5:	5d                   	pop    %rbp
  4009e6:	c3                   	retq   
  4009e7:	90                   	nop

00000000004009e8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009e8:	55                   	push   %rbp
  4009e9:	48 89 e5             	mov    %rsp,%rbp
  4009ec:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f0:	90                   	nop
  4009f1:	5d                   	pop    %rbp
  4009f2:	c3                   	retq   

00000000004009f3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  4009f3:	55                   	push   %rbp
  4009f4:	48 89 e5             	mov    %rsp,%rbp
  4009f7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fb:	90                   	nop
  4009fc:	5d                   	pop    %rbp
  4009fd:	c3                   	retq   

00000000004009fe <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009fe:	55                   	push   %rbp
  4009ff:	48 89 e5             	mov    %rsp,%rbp
  400a02:	48 83 ec 20          	sub    $0x20,%rsp
  400a06:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a0e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a11:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a15:	48 89 c7             	mov    %rax,%rdi
  400a18:	e8 95 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a1d:	48 89 c2             	mov    %rax,%rdx
  400a20:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a23:	89 02                	mov    %eax,(%rdx)
  400a25:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a29:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a2d:	48 89 d6             	mov    %rdx,%rsi
  400a30:	48 89 c7             	mov    %rax,%rdi
  400a33:	e8 94 00 00 00       	callq  400acc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a38:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a3c:	c9                   	leaveq 
  400a3d:	c3                   	retq   

0000000000400a3e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a3e:	55                   	push   %rbp
  400a3f:	48 89 e5             	mov    %rsp,%rbp
  400a42:	48 83 ec 20          	sub    $0x20,%rsp
  400a46:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a4a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a4e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a51:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a55:	48 89 c7             	mov    %rax,%rdi
  400a58:	e8 7f 00 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a5d:	48 89 c2             	mov    %rax,%rdx
  400a60:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a63:	89 02                	mov    %eax,(%rdx)
  400a65:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a69:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a6d:	48 89 d6             	mov    %rdx,%rsi
  400a70:	48 89 c7             	mov    %rax,%rdi
  400a73:	e8 7e 00 00 00       	callq  400af6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a78:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a7c:	c9                   	leaveq 
  400a7d:	c3                   	retq   

0000000000400a7e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a7e:	55                   	push   %rbp
  400a7f:	48 89 e5             	mov    %rsp,%rbp
  400a82:	48 83 ec 10          	sub    $0x10,%rsp
  400a86:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a8e:	48 89 c7             	mov    %rax,%rdi
  400a91:	e8 1c 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a96:	c9                   	leaveq 
  400a97:	c3                   	retq   

0000000000400a98 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a98:	55                   	push   %rbp
  400a99:	48 89 e5             	mov    %rsp,%rbp
  400a9c:	48 83 ec 10          	sub    $0x10,%rsp
  400aa0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 2c 00 00 00       	callq  400adc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab0:	c9                   	leaveq 
  400ab1:	c3                   	retq   

0000000000400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ab2:	55                   	push   %rbp
  400ab3:	48 89 e5             	mov    %rsp,%rbp
  400ab6:	48 83 ec 10          	sub    $0x10,%rsp
  400aba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400abe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac2:	48 89 c7             	mov    %rax,%rdi
  400ac5:	e8 a2 00 00 00       	callq  400b6c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aca:	c9                   	leaveq 
  400acb:	c3                   	retq   

0000000000400acc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400acc:	55                   	push   %rbp
  400acd:	48 89 e5             	mov    %rsp,%rbp
  400ad0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ad8:	90                   	nop
  400ad9:	5d                   	pop    %rbp
  400ada:	c3                   	retq   
  400adb:	90                   	nop

0000000000400adc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400adc:	55                   	push   %rbp
  400add:	48 89 e5             	mov    %rsp,%rbp
  400ae0:	48 83 ec 10          	sub    $0x10,%rsp
  400ae4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aec:	48 89 c7             	mov    %rax,%rdi
  400aef:	e8 9a 00 00 00       	callq  400b8e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400af4:	c9                   	leaveq 
  400af5:	c3                   	retq   

0000000000400af6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400af6:	55                   	push   %rbp
  400af7:	48 89 e5             	mov    %rsp,%rbp
  400afa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afe:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b02:	90                   	nop
  400b03:	5d                   	pop    %rbp
  400b04:	c3                   	retq   
  400b05:	90                   	nop

0000000000400b06 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400b06:	55                   	push   %rbp
  400b07:	48 89 e5             	mov    %rsp,%rbp
  400b0a:	48 83 ec 10          	sub    $0x10,%rsp
  400b0e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b12:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400b17:	e8 a4 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b1c:	48 89 c2             	mov    %rax,%rdx
  400b1f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b23:	48 89 50 08          	mov    %rdx,0x8(%rax)
  400b27:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b2b:	48 c7 40 18 00 00 00 	movq   $0x0,0x18(%rax)
  400b32:	00 
  400b33:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b37:	48 8b 50 18          	mov    0x18(%rax),%rdx
  400b3b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b3f:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b43:	90                   	nop
  400b44:	c9                   	leaveq 
  400b45:	c3                   	retq   

0000000000400b46 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b46:	55                   	push   %rbp
  400b47:	48 89 e5             	mov    %rsp,%rbp
  400b4a:	90                   	nop
  400b4b:	5d                   	pop    %rbp
  400b4c:	c3                   	retq   

0000000000400b4d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b4d:	55                   	push   %rbp
  400b4e:	48 89 e5             	mov    %rsp,%rbp
  400b51:	48 83 ec 10          	sub    $0x10,%rsp
  400b55:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b59:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b5d:	48 83 c0 01          	add    $0x1,%rax
  400b61:	48 89 c7             	mov    %rax,%rdi
  400b64:	e8 4e 00 00 00       	callq  400bb7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b69:	c9                   	leaveq 
  400b6a:	c3                   	retq   
  400b6b:	90                   	nop

0000000000400b6c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b6c:	55                   	push   %rbp
  400b6d:	48 89 e5             	mov    %rsp,%rbp
  400b70:	48 83 ec 08          	sub    $0x8,%rsp
  400b74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b78:	e8 59 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b7d:	48 8b 40 08          	mov    0x8(%rax),%rax
  400b81:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400b85:	48 c1 e2 02          	shl    $0x2,%rdx
  400b89:	48 01 d0             	add    %rdx,%rax
  400b8c:	c9                   	leaveq 
  400b8d:	c3                   	retq   

0000000000400b8e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b8e:	55                   	push   %rbp
  400b8f:	48 89 e5             	mov    %rsp,%rbp
  400b92:	48 83 ec 08          	sub    $0x8,%rsp
  400b96:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b9a:	e8 37 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b9f:	48 8b 40 08          	mov    0x8(%rax),%rax
  400ba3:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400ba7:	48 c1 e2 02          	shl    $0x2,%rdx
  400bab:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400bb2:	48 01 d0             	add    %rdx,%rax
  400bb5:	c9                   	leaveq 
  400bb6:	c3                   	retq   

0000000000400bb7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bb7:	55                   	push   %rbp
  400bb8:	48 89 e5             	mov    %rsp,%rbp
  400bbb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
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
  400bdb:	4c 8d 25 2e 12 20 00 	lea    0x20122e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
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
