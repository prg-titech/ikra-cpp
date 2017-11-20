
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
  4004ef:	49 c7 c0 70 0c 40 00 	mov    $0x400c70,%r8
  4004f6:	48 c7 c1 00 0c 40 00 	mov    $0x400c00,%rcx
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
  4005e1:	e8 95 03 00 00       	callq  40097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005e6:	bf 00 00 00 00       	mov    $0x0,%edi
  4005eb:	e8 b4 03 00 00       	callq  4009a4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4005f0:	48 89 c3             	mov    %rax,%rbx
  4005f3:	48 89 df             	mov    %rbx,%rdi
  4005f6:	e8 bb 02 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
  4005fb:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  400600:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400605:	be 77 77 00 00       	mov    $0x7777,%esi
  40060a:	48 89 c7             	mov    %rax,%rdi
  40060d:	e8 d6 03 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400612:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400617:	be 88 88 00 00       	mov    $0x8888,%esi
  40061c:	48 89 c7             	mov    %rax,%rdi
  40061f:	e8 f2 03 00 00       	callq  400a16 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400624:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400629:	48 89 c7             	mov    %rax,%rdi
  40062c:	e8 99 01 00 00       	callq  4007ca <_ZN9TestClass15increase_field0Ev>
  400631:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400636:	48 89 c7             	mov    %rax,%rdi
  400639:	e8 f0 01 00 00       	callq  40082e <_ZN9TestClass15increase_field1Ev>
  40063e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400643:	48 89 c7             	mov    %rax,%rdi
  400646:	e8 15 04 00 00       	callq  400a60 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40064b:	8b 18                	mov    (%rax),%ebx
  40064d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400652:	48 89 c7             	mov    %rax,%rdi
  400655:	e8 ea 03 00 00       	callq  400a44 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40065a:	8b 00                	mov    (%rax),%eax
  40065c:	89 da                	mov    %ebx,%edx
  40065e:	89 c6                	mov    %eax,%esi
  400660:	bf 94 0c 40 00       	mov    $0x400c94,%edi
  400665:	b8 00 00 00 00       	mov    $0x0,%eax
  40066a:	e8 31 fe ff ff       	callq  4004a0 <printf@plt>
  40066f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400674:	48 89 c7             	mov    %rax,%rdi
  400677:	e8 c8 03 00 00       	callq  400a44 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40067c:	8b 00                	mov    (%rax),%eax
  40067e:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400683:	75 16                	jne    40069b <main+0xbf>
  400685:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40068a:	48 89 c7             	mov    %rax,%rdi
  40068d:	e8 ce 03 00 00       	callq  400a60 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400692:	8b 00                	mov    (%rax),%eax
  400694:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400699:	74 07                	je     4006a2 <main+0xc6>
  40069b:	b8 01 00 00 00       	mov    $0x1,%eax
  4006a0:	eb 05                	jmp    4006a7 <main+0xcb>
  4006a2:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a7:	0f b6 c0             	movzbl %al,%eax
  4006aa:	48 83 c4 10          	add    $0x10,%rsp
  4006ae:	5b                   	pop    %rbx
  4006af:	c3                   	retq   

00000000004006b0 <_Z12new_instancev>:
  4006b0:	53                   	push   %rbx
  4006b1:	bf 00 00 00 00       	mov    $0x0,%edi
  4006b6:	e8 e9 02 00 00       	callq  4009a4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006bb:	48 89 c3             	mov    %rax,%rbx
  4006be:	48 89 df             	mov    %rbx,%rdi
  4006c1:	e8 f0 01 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
  4006c6:	48 89 d8             	mov    %rbx,%rax
  4006c9:	5b                   	pop    %rbx
  4006ca:	c3                   	retq   

00000000004006cb <_Z12write_field0P9TestClass>:
  4006cb:	48 83 ec 18          	sub    $0x18,%rsp
  4006cf:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006d4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006d9:	be 77 77 00 00       	mov    $0x7777,%esi
  4006de:	48 89 c7             	mov    %rax,%rdi
  4006e1:	e8 02 03 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006e6:	90                   	nop
  4006e7:	48 83 c4 18          	add    $0x18,%rsp
  4006eb:	c3                   	retq   

00000000004006ec <_Z11read_field0P9TestClass>:
  4006ec:	48 83 ec 18          	sub    $0x18,%rsp
  4006f0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006f5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006fa:	48 89 c7             	mov    %rax,%rdi
  4006fd:	e8 42 03 00 00       	callq  400a44 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400702:	8b 00                	mov    (%rax),%eax
  400704:	48 83 c4 18          	add    $0x18,%rsp
  400708:	c3                   	retq   

0000000000400709 <_Z12write_field1P9TestClass>:
  400709:	48 83 ec 18          	sub    $0x18,%rsp
  40070d:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400712:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400717:	be 77 77 00 00       	mov    $0x7777,%esi
  40071c:	48 89 c7             	mov    %rax,%rdi
  40071f:	e8 f2 02 00 00       	callq  400a16 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400724:	90                   	nop
  400725:	48 83 c4 18          	add    $0x18,%rsp
  400729:	c3                   	retq   

000000000040072a <_Z11read_field1P9TestClass>:
  40072a:	48 83 ec 18          	sub    $0x18,%rsp
  40072e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400733:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400738:	48 89 c7             	mov    %rax,%rdi
  40073b:	e8 20 03 00 00       	callq  400a60 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400740:	8b 00                	mov    (%rax),%eax
  400742:	48 83 c4 18          	add    $0x18,%rsp
  400746:	c3                   	retq   

0000000000400747 <_Z21explicit_write_field0m>:
  400747:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40074c:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400751:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  400758:	77 77 00 00 
  40075c:	90                   	nop
  40075d:	c3                   	retq   

000000000040075e <_Z20explicit_read_field0m>:
  40075e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400763:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400768:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  40076f:	c3                   	retq   

0000000000400770 <_ZnwmPv>:
  400770:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400775:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  40077a:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40077f:	c3                   	retq   

0000000000400780 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  400780:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400785:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40078a:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400791:	90                   	nop
  400792:	c3                   	retq   
  400793:	90                   	nop

0000000000400794 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400794:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400799:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  40079e:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007a3:	48 8b 00             	mov    (%rax),%rax
  4007a6:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007ab:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007b0:	48 8b 10             	mov    (%rax),%rdx
  4007b3:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  4007b8:	48 01 c2             	add    %rax,%rdx
  4007bb:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007c0:	48 89 10             	mov    %rdx,(%rax)
  4007c3:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007c8:	c3                   	retq   
  4007c9:	90                   	nop

00000000004007ca <_ZN9TestClass15increase_field0Ev>:
  4007ca:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007cf:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007d3:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007d7:	55                   	push   %rbp
  4007d8:	48 89 e5             	mov    %rsp,%rbp
  4007db:	41 52                	push   %r10
  4007dd:	48 83 ec 48          	sub    $0x48,%rsp
  4007e1:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4007e5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007ec:	00 00 
  4007ee:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007f2:	31 c0                	xor    %eax,%eax
  4007f4:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4007f8:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4007fc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400801:	48 89 ce             	mov    %rcx,%rsi
  400804:	48 89 c7             	mov    %rax,%rdi
  400807:	e8 e0 00 00 00       	callq  4008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40080c:	90                   	nop
  40080d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400811:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400818:	00 00 
  40081a:	74 05                	je     400821 <_ZN9TestClass15increase_field0Ev+0x57>
  40081c:	e8 6f fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400821:	48 83 c4 48          	add    $0x48,%rsp
  400825:	41 5a                	pop    %r10
  400827:	5d                   	pop    %rbp
  400828:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  40082c:	c3                   	retq   
  40082d:	90                   	nop

000000000040082e <_ZN9TestClass15increase_field1Ev>:
  40082e:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400833:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400837:	41 ff 72 f8          	pushq  -0x8(%r10)
  40083b:	55                   	push   %rbp
  40083c:	48 89 e5             	mov    %rsp,%rbp
  40083f:	41 52                	push   %r10
  400841:	48 83 ec 48          	sub    $0x48,%rsp
  400845:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400849:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400850:	00 00 
  400852:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400856:	31 c0                	xor    %eax,%eax
  400858:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  40085c:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400860:	ba 44 44 00 00       	mov    $0x4444,%edx
  400865:	48 89 ce             	mov    %rcx,%rsi
  400868:	48 89 c7             	mov    %rax,%rdi
  40086b:	e8 c4 00 00 00       	callq  400934 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400870:	90                   	nop
  400871:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400875:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40087c:	00 00 
  40087e:	74 05                	je     400885 <_ZN9TestClass15increase_field1Ev+0x57>
  400880:	e8 0b fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400885:	48 83 c4 48          	add    $0x48,%rsp
  400889:	41 5a                	pop    %r10
  40088b:	5d                   	pop    %rbp
  40088c:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400890:	c3                   	retq   
  400891:	90                   	nop

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
  4008d4:	e8 f9 00 00 00       	callq  4009d2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 f4 00 00 00       	callq  4009da <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008e6:	90                   	nop
  4008e7:	48 83 c4 18          	add    $0x18,%rsp
  4008eb:	c3                   	retq   

00000000004008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4008ec:	48 83 ec 28          	sub    $0x28,%rsp
  4008f0:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008f5:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4008fa:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4008fe:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400903:	48 89 c7             	mov    %rax,%rdi
  400906:	e8 71 01 00 00       	callq  400a7c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40090b:	8b 10                	mov    (%rax),%edx
  40090d:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400912:	89 10                	mov    %edx,(%rax)
  400914:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400919:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40091e:	48 89 d6             	mov    %rdx,%rsi
  400921:	48 89 c7             	mov    %rax,%rdi
  400924:	e8 6f 01 00 00       	callq  400a98 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400929:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40092e:	48 83 c4 28          	add    $0x28,%rsp
  400932:	c3                   	retq   
  400933:	90                   	nop

0000000000400934 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400934:	48 83 ec 28          	sub    $0x28,%rsp
  400938:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40093d:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400942:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400946:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40094b:	48 89 c7             	mov    %rax,%rdi
  40094e:	e8 51 01 00 00       	callq  400aa4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400953:	8b 10                	mov    (%rax),%edx
  400955:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  40095a:	89 10                	mov    %edx,(%rax)
  40095c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400961:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400966:	48 89 d6             	mov    %rdx,%rsi
  400969:	48 89 c7             	mov    %rax,%rdi
  40096c:	e8 4f 01 00 00       	callq  400ac0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400971:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400976:	48 83 c4 28          	add    $0x28,%rsp
  40097a:	c3                   	retq   

000000000040097b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  40097b:	48 83 ec 08          	sub    $0x8,%rsp
  40097f:	e8 52 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400984:	48 89 c6             	mov    %rax,%rsi
  400987:	bf 20 00 00 00       	mov    $0x20,%edi
  40098c:	e8 df fd ff ff       	callq  400770 <_ZnwmPv>
  400991:	48 85 c0             	test   %rax,%rax
  400994:	74 08                	je     40099e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400996:	48 89 c7             	mov    %rax,%rdi
  400999:	e8 4a 01 00 00       	callq  400ae8 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  40099e:	90                   	nop
  40099f:	48 83 c4 08          	add    $0x8,%rsp
  4009a3:	c3                   	retq   

00000000004009a4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009a4:	48 83 ec 18          	sub    $0x18,%rsp
  4009a8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009ad:	e8 87 01 00 00       	callq  400b39 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009b2:	e8 1f fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009b7:	be 01 00 00 00       	mov    $0x1,%esi
  4009bc:	48 89 c7             	mov    %rax,%rdi
  4009bf:	e8 d0 fd ff ff       	callq  400794 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009c4:	48 89 c7             	mov    %rax,%rdi
  4009c7:	e8 6f 01 00 00       	callq  400b3b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009cc:	48 83 c4 18          	add    $0x18,%rsp
  4009d0:	c3                   	retq   
  4009d1:	90                   	nop

00000000004009d2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4009d2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009d7:	90                   	nop
  4009d8:	c3                   	retq   
  4009d9:	90                   	nop

00000000004009da <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009da:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009df:	90                   	nop
  4009e0:	c3                   	retq   

00000000004009e1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  4009e1:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009e6:	90                   	nop
  4009e7:	c3                   	retq   

00000000004009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009e8:	48 83 ec 18          	sub    $0x18,%rsp
  4009ec:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009f1:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4009f5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009fa:	48 89 c7             	mov    %rax,%rdi
  4009fd:	e8 7a 00 00 00       	callq  400a7c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a02:	48 89 c2             	mov    %rax,%rdx
  400a05:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a09:	89 02                	mov    %eax,(%rdx)
  400a0b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a10:	48 83 c4 18          	add    $0x18,%rsp
  400a14:	c3                   	retq   
  400a15:	90                   	nop

0000000000400a16 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a16:	48 83 ec 18          	sub    $0x18,%rsp
  400a1a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a1f:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a23:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a28:	48 89 c7             	mov    %rax,%rdi
  400a2b:	e8 74 00 00 00       	callq  400aa4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a30:	48 89 c2             	mov    %rax,%rdx
  400a33:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a37:	89 02                	mov    %eax,(%rdx)
  400a39:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a3e:	48 83 c4 18          	add    $0x18,%rsp
  400a42:	c3                   	retq   
  400a43:	90                   	nop

0000000000400a44 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a44:	48 83 ec 18          	sub    $0x18,%rsp
  400a48:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a4d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a52:	48 89 c7             	mov    %rax,%rdi
  400a55:	e8 22 00 00 00       	callq  400a7c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a5a:	48 83 c4 18          	add    $0x18,%rsp
  400a5e:	c3                   	retq   
  400a5f:	90                   	nop

0000000000400a60 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a60:	48 83 ec 18          	sub    $0x18,%rsp
  400a64:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a69:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a6e:	48 89 c7             	mov    %rax,%rdi
  400a71:	e8 2e 00 00 00       	callq  400aa4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a76:	48 83 c4 18          	add    $0x18,%rsp
  400a7a:	c3                   	retq   
  400a7b:	90                   	nop

0000000000400a7c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a7c:	48 83 ec 18          	sub    $0x18,%rsp
  400a80:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a85:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a8a:	48 89 c7             	mov    %rax,%rdi
  400a8d:	e8 c8 00 00 00       	callq  400b5a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a92:	48 83 c4 18          	add    $0x18,%rsp
  400a96:	c3                   	retq   
  400a97:	90                   	nop

0000000000400a98 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a98:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a9d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400aa2:	90                   	nop
  400aa3:	c3                   	retq   

0000000000400aa4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aa4:	48 83 ec 18          	sub    $0x18,%rsp
  400aa8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400aad:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ab2:	48 89 c7             	mov    %rax,%rdi
  400ab5:	e8 d6 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aba:	48 83 c4 18          	add    $0x18,%rsp
  400abe:	c3                   	retq   
  400abf:	90                   	nop

0000000000400ac0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400ac0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ac5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400aca:	90                   	nop
  400acb:	c3                   	retq   

0000000000400acc <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400acc:	48 83 ec 18          	sub    $0x18,%rsp
  400ad0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ad5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ada:	48 89 c7             	mov    %rax,%rdi
  400add:	e8 9e fc ff ff       	callq  400780 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400ae2:	90                   	nop
  400ae3:	48 83 c4 18          	add    $0x18,%rsp
  400ae7:	c3                   	retq   

0000000000400ae8 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400ae8:	48 83 ec 18          	sub    $0x18,%rsp
  400aec:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400af1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400af6:	48 89 c7             	mov    %rax,%rdi
  400af9:	e8 ce ff ff ff       	callq  400acc <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400afe:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400b03:	e8 b8 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b08:	48 89 c2             	mov    %rax,%rdx
  400b0b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b10:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400b14:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b19:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b20:	00 
  400b21:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b26:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400b2a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b2f:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b33:	90                   	nop
  400b34:	48 83 c4 18          	add    $0x18,%rsp
  400b38:	c3                   	retq   

0000000000400b39 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b39:	90                   	nop
  400b3a:	c3                   	retq   

0000000000400b3b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b3b:	48 83 ec 18          	sub    $0x18,%rsp
  400b3f:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b44:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b49:	48 83 c0 01          	add    $0x1,%rax
  400b4d:	48 89 c7             	mov    %rax,%rdi
  400b50:	e8 73 00 00 00       	callq  400bc8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b55:	48 83 c4 18          	add    $0x18,%rsp
  400b59:	c3                   	retq   

0000000000400b5a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b5a:	48 83 ec 28          	sub    $0x28,%rsp
  400b5e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b63:	e8 6e fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b68:	48 89 c7             	mov    %rax,%rdi
  400b6b:	e8 64 00 00 00       	callq  400bd4 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400b70:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b75:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b7a:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b81:	00 
  400b82:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b87:	48 01 d0             	add    %rdx,%rax
  400b8a:	48 83 c4 28          	add    $0x28,%rsp
  400b8e:	c3                   	retq   
  400b8f:	90                   	nop

0000000000400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b90:	48 83 ec 28          	sub    $0x28,%rsp
  400b94:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b99:	e8 38 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b9e:	48 89 c7             	mov    %rax,%rdi
  400ba1:	e8 2e 00 00 00       	callq  400bd4 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400ba6:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bab:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bb0:	48 c1 e0 02          	shl    $0x2,%rax
  400bb4:	48 8d 90 d0 48 00 00 	lea    0x48d0(%rax),%rdx
  400bbb:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bc0:	48 01 d0             	add    %rdx,%rax
  400bc3:	48 83 c4 28          	add    $0x28,%rsp
  400bc7:	c3                   	retq   

0000000000400bc8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bc8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bcd:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400bd2:	c3                   	retq   
  400bd3:	90                   	nop

0000000000400bd4 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400bd4:	48 83 ec 18          	sub    $0x18,%rsp
  400bd8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bdd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400be2:	48 89 c7             	mov    %rax,%rdi
  400be5:	e8 06 00 00 00       	callq  400bf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bea:	48 83 c4 18          	add    $0x18,%rsp
  400bee:	c3                   	retq   
  400bef:	90                   	nop

0000000000400bf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400bf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bf5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400bfa:	48 8b 40 18          	mov    0x18(%rax),%rax
  400bfe:	c3                   	retq   
  400bff:	90                   	nop

0000000000400c00 <__libc_csu_init>:
  400c00:	41 57                	push   %r15
  400c02:	41 56                	push   %r14
  400c04:	41 89 ff             	mov    %edi,%r15d
  400c07:	41 55                	push   %r13
  400c09:	41 54                	push   %r12
  400c0b:	4c 8d 25 fe 11 20 00 	lea    0x2011fe(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c12:	55                   	push   %rbp
  400c13:	48 8d 2d fe 11 20 00 	lea    0x2011fe(%rip),%rbp        # 601e18 <__init_array_end>
  400c1a:	53                   	push   %rbx
  400c1b:	49 89 f6             	mov    %rsi,%r14
  400c1e:	49 89 d5             	mov    %rdx,%r13
  400c21:	4c 29 e5             	sub    %r12,%rbp
  400c24:	48 83 ec 08          	sub    $0x8,%rsp
  400c28:	48 c1 fd 03          	sar    $0x3,%rbp
  400c2c:	e8 2f f8 ff ff       	callq  400460 <_init>
  400c31:	48 85 ed             	test   %rbp,%rbp
  400c34:	74 20                	je     400c56 <__libc_csu_init+0x56>
  400c36:	31 db                	xor    %ebx,%ebx
  400c38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c3f:	00 
  400c40:	4c 89 ea             	mov    %r13,%rdx
  400c43:	4c 89 f6             	mov    %r14,%rsi
  400c46:	44 89 ff             	mov    %r15d,%edi
  400c49:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c4d:	48 83 c3 01          	add    $0x1,%rbx
  400c51:	48 39 eb             	cmp    %rbp,%rbx
  400c54:	75 ea                	jne    400c40 <__libc_csu_init+0x40>
  400c56:	48 83 c4 08          	add    $0x8,%rsp
  400c5a:	5b                   	pop    %rbx
  400c5b:	5d                   	pop    %rbp
  400c5c:	41 5c                	pop    %r12
  400c5e:	41 5d                	pop    %r13
  400c60:	41 5e                	pop    %r14
  400c62:	41 5f                	pop    %r15
  400c64:	c3                   	retq   
  400c65:	90                   	nop
  400c66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6d:	00 00 00 

0000000000400c70 <__libc_csu_fini>:
  400c70:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c74 <_fini>:
  400c74:	48 83 ec 08          	sub    $0x8,%rsp
  400c78:	48 83 c4 08          	add    $0x8,%rsp
  400c7c:	c3                   	retq   
