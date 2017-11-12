
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
  4004ef:	49 c7 c0 f0 0c 40 00 	mov    $0x400cf0,%r8
  4004f6:	48 c7 c1 80 0c 40 00 	mov    $0x400c80,%rcx
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
  4005e6:	48 83 ec 18          	sub    $0x18,%rsp
  4005ea:	e8 bd 03 00 00       	callq  4009ac <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005ef:	bf 04 00 00 00       	mov    $0x4,%edi
  4005f4:	e8 d9 03 00 00       	callq  4009d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4005f9:	48 89 c3             	mov    %rax,%rbx
  4005fc:	b8 00 00 00 00       	mov    $0x0,%eax
  400601:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400605:	48 83 f8 03          	cmp    $0x3,%rax
  400609:	74 06                	je     400611 <main+0x30>
  40060b:	48 83 c0 01          	add    $0x1,%rax
  40060f:	eb f0                	jmp    400601 <main+0x20>
  400611:	48 89 df             	mov    %rbx,%rdi
  400614:	e8 d7 02 00 00       	callq  4008f0 <_ZN9TestClassC1Ev>
  400619:	48 89 5d e8          	mov    %rbx,-0x18(%rbp)
  40061d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400621:	48 83 c0 04          	add    $0x4,%rax
  400625:	be 77 77 00 00       	mov    $0x7777,%esi
  40062a:	48 89 c7             	mov    %rax,%rdi
  40062d:	e8 f0 03 00 00       	callq  400a22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400632:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400636:	48 83 c0 04          	add    $0x4,%rax
  40063a:	be 88 88 00 00       	mov    $0x8888,%esi
  40063f:	48 89 c7             	mov    %rax,%rdi
  400642:	e8 05 04 00 00       	callq  400a4c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400647:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40064b:	48 89 c7             	mov    %rax,%rdi
  40064e:	e8 d9 01 00 00       	callq  40082c <_ZN9TestClass15increase_field0Ev>
  400653:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400657:	48 89 c7             	mov    %rax,%rdi
  40065a:	e8 1b 02 00 00       	callq  40087a <_ZN9TestClass15increase_field1Ev>
  40065f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400663:	48 83 c0 04          	add    $0x4,%rax
  400667:	48 89 c7             	mov    %rax,%rdi
  40066a:	e8 21 04 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40066f:	8b 18                	mov    (%rax),%ebx
  400671:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400675:	48 83 c0 04          	add    $0x4,%rax
  400679:	48 89 c7             	mov    %rax,%rdi
  40067c:	e8 f5 03 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400681:	8b 00                	mov    (%rax),%eax
  400683:	89 da                	mov    %ebx,%edx
  400685:	89 c6                	mov    %eax,%esi
  400687:	bf 14 0d 40 00       	mov    $0x400d14,%edi
  40068c:	b8 00 00 00 00       	mov    $0x0,%eax
  400691:	e8 0a fe ff ff       	callq  4004a0 <printf@plt>
  400696:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40069a:	48 83 c0 04          	add    $0x4,%rax
  40069e:	48 89 c7             	mov    %rax,%rdi
  4006a1:	e8 d0 03 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4006a6:	8b 00                	mov    (%rax),%eax
  4006a8:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4006ad:	75 19                	jne    4006c8 <main+0xe7>
  4006af:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4006b3:	48 83 c0 04          	add    $0x4,%rax
  4006b7:	48 89 c7             	mov    %rax,%rdi
  4006ba:	e8 d1 03 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4006bf:	8b 00                	mov    (%rax),%eax
  4006c1:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006c6:	74 07                	je     4006cf <main+0xee>
  4006c8:	b8 01 00 00 00       	mov    $0x1,%eax
  4006cd:	eb 05                	jmp    4006d4 <main+0xf3>
  4006cf:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d4:	0f b6 c0             	movzbl %al,%eax
  4006d7:	48 83 c4 18          	add    $0x18,%rsp
  4006db:	5b                   	pop    %rbx
  4006dc:	5d                   	pop    %rbp
  4006dd:	c3                   	retq   

00000000004006de <_Z12new_instancev>:
  4006de:	55                   	push   %rbp
  4006df:	48 89 e5             	mov    %rsp,%rbp
  4006e2:	53                   	push   %rbx
  4006e3:	48 83 ec 08          	sub    $0x8,%rsp
  4006e7:	bf 04 00 00 00       	mov    $0x4,%edi
  4006ec:	e8 e1 02 00 00       	callq  4009d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4006f1:	48 89 c3             	mov    %rax,%rbx
  4006f4:	b8 00 00 00 00       	mov    $0x0,%eax
  4006f9:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006fd:	48 83 f8 03          	cmp    $0x3,%rax
  400701:	74 06                	je     400709 <_Z12new_instancev+0x2b>
  400703:	48 83 c0 01          	add    $0x1,%rax
  400707:	eb f0                	jmp    4006f9 <_Z12new_instancev+0x1b>
  400709:	48 89 df             	mov    %rbx,%rdi
  40070c:	e8 df 01 00 00       	callq  4008f0 <_ZN9TestClassC1Ev>
  400711:	48 89 d8             	mov    %rbx,%rax
  400714:	48 83 c4 08          	add    $0x8,%rsp
  400718:	5b                   	pop    %rbx
  400719:	5d                   	pop    %rbp
  40071a:	c3                   	retq   

000000000040071b <_Z12write_field0P9TestClass>:
  40071b:	55                   	push   %rbp
  40071c:	48 89 e5             	mov    %rsp,%rbp
  40071f:	48 83 ec 10          	sub    $0x10,%rsp
  400723:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400727:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40072b:	48 83 c0 04          	add    $0x4,%rax
  40072f:	be 77 77 00 00       	mov    $0x7777,%esi
  400734:	48 89 c7             	mov    %rax,%rdi
  400737:	e8 e6 02 00 00       	callq  400a22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40073c:	90                   	nop
  40073d:	c9                   	leaveq 
  40073e:	c3                   	retq   

000000000040073f <_Z11read_field0P9TestClass>:
  40073f:	55                   	push   %rbp
  400740:	48 89 e5             	mov    %rsp,%rbp
  400743:	48 83 ec 10          	sub    $0x10,%rsp
  400747:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40074b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40074f:	48 83 c0 04          	add    $0x4,%rax
  400753:	48 89 c7             	mov    %rax,%rdi
  400756:	e8 1b 03 00 00       	callq  400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40075b:	8b 00                	mov    (%rax),%eax
  40075d:	c9                   	leaveq 
  40075e:	c3                   	retq   

000000000040075f <_Z12write_field1P9TestClass>:
  40075f:	55                   	push   %rbp
  400760:	48 89 e5             	mov    %rsp,%rbp
  400763:	48 83 ec 10          	sub    $0x10,%rsp
  400767:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40076b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40076f:	48 83 c0 04          	add    $0x4,%rax
  400773:	be 77 77 00 00       	mov    $0x7777,%esi
  400778:	48 89 c7             	mov    %rax,%rdi
  40077b:	e8 cc 02 00 00       	callq  400a4c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400780:	90                   	nop
  400781:	c9                   	leaveq 
  400782:	c3                   	retq   

0000000000400783 <_Z11read_field1P9TestClass>:
  400783:	55                   	push   %rbp
  400784:	48 89 e5             	mov    %rsp,%rbp
  400787:	48 83 ec 10          	sub    $0x10,%rsp
  40078b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40078f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400793:	48 83 c0 04          	add    $0x4,%rax
  400797:	48 89 c7             	mov    %rax,%rdi
  40079a:	e8 f1 02 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40079f:	8b 00                	mov    (%rax),%eax
  4007a1:	c9                   	leaveq 
  4007a2:	c3                   	retq   

00000000004007a3 <_Z21explicit_write_field0m>:
  4007a3:	55                   	push   %rbp
  4007a4:	48 89 e5             	mov    %rsp,%rbp
  4007a7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ab:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007af:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  4007b6:	77 77 00 00 
  4007ba:	90                   	nop
  4007bb:	5d                   	pop    %rbp
  4007bc:	c3                   	retq   

00000000004007bd <_Z20explicit_read_field0m>:
  4007bd:	55                   	push   %rbp
  4007be:	48 89 e5             	mov    %rsp,%rbp
  4007c1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007c9:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  4007d0:	5d                   	pop    %rbp
  4007d1:	c3                   	retq   

00000000004007d2 <_ZnwmPv>:
  4007d2:	55                   	push   %rbp
  4007d3:	48 89 e5             	mov    %rsp,%rbp
  4007d6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007da:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007de:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007e2:	5d                   	pop    %rbp
  4007e3:	c3                   	retq   

00000000004007e4 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  4007e4:	55                   	push   %rbp
  4007e5:	48 89 e5             	mov    %rsp,%rbp
  4007e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f0:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007f7:	90                   	nop
  4007f8:	5d                   	pop    %rbp
  4007f9:	c3                   	retq   

00000000004007fa <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007fa:	55                   	push   %rbp
  4007fb:	48 89 e5             	mov    %rsp,%rbp
  4007fe:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400802:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  400806:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40080a:	48 8b 00             	mov    (%rax),%rax
  40080d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400811:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400815:	48 8b 10             	mov    (%rax),%rdx
  400818:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40081c:	48 01 c2             	add    %rax,%rdx
  40081f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400823:	48 89 10             	mov    %rdx,(%rax)
  400826:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40082a:	5d                   	pop    %rbp
  40082b:	c3                   	retq   

000000000040082c <_ZN9TestClass15increase_field0Ev>:
  40082c:	55                   	push   %rbp
  40082d:	48 89 e5             	mov    %rsp,%rbp
  400830:	48 83 ec 30          	sub    $0x30,%rsp
  400834:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400838:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40083f:	00 00 
  400841:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400845:	31 c0                	xor    %eax,%eax
  400847:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40084b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40084f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400853:	ba 55 55 00 00       	mov    $0x5555,%edx
  400858:	48 89 ce             	mov    %rcx,%rsi
  40085b:	48 89 c7             	mov    %rax,%rdi
  40085e:	e8 c9 00 00 00       	callq  40092c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400863:	90                   	nop
  400864:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400868:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40086f:	00 00 
  400871:	74 05                	je     400878 <_ZN9TestClass15increase_field0Ev+0x4c>
  400873:	e8 18 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400878:	c9                   	leaveq 
  400879:	c3                   	retq   

000000000040087a <_ZN9TestClass15increase_field1Ev>:
  40087a:	55                   	push   %rbp
  40087b:	48 89 e5             	mov    %rsp,%rbp
  40087e:	48 83 ec 30          	sub    $0x30,%rsp
  400882:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400886:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40088d:	00 00 
  40088f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400893:	31 c0                	xor    %eax,%eax
  400895:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400899:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40089d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008a1:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008a6:	48 89 ce             	mov    %rcx,%rsi
  4008a9:	48 89 c7             	mov    %rax,%rdi
  4008ac:	e8 bb 00 00 00       	callq  40096c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008b1:	90                   	nop
  4008b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008bd:	00 00 
  4008bf:	74 05                	je     4008c6 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008c1:	e8 ca fb ff ff       	callq  400490 <__stack_chk_fail@plt>
  4008c6:	c9                   	leaveq 
  4008c7:	c3                   	retq   

00000000004008c8 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008c8:	55                   	push   %rbp
  4008c9:	48 89 e5             	mov    %rsp,%rbp
  4008cc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d0:	90                   	nop
  4008d1:	5d                   	pop    %rbp
  4008d2:	c3                   	retq   
  4008d3:	90                   	nop

00000000004008d4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  4008d4:	55                   	push   %rbp
  4008d5:	48 89 e5             	mov    %rsp,%rbp
  4008d8:	48 83 ec 10          	sub    $0x10,%rsp
  4008dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e4:	48 89 c7             	mov    %rax,%rdi
  4008e7:	e8 dc ff ff ff       	callq  4008c8 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ec:	90                   	nop
  4008ed:	c9                   	leaveq 
  4008ee:	c3                   	retq   
  4008ef:	90                   	nop

00000000004008f0 <_ZN9TestClassC1Ev>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400900:	48 89 c7             	mov    %rax,%rdi
  400903:	e8 cc ff ff ff       	callq  4008d4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400908:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40090c:	48 83 c0 04          	add    $0x4,%rax
  400910:	48 89 c7             	mov    %rax,%rdi
  400913:	e8 e8 00 00 00       	callq  400a00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400918:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091c:	48 83 c0 04          	add    $0x4,%rax
  400920:	48 89 c7             	mov    %rax,%rdi
  400923:	e8 e4 00 00 00       	callq  400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400928:	90                   	nop
  400929:	c9                   	leaveq 
  40092a:	c3                   	retq   
  40092b:	90                   	nop

000000000040092c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40092c:	55                   	push   %rbp
  40092d:	48 89 e5             	mov    %rsp,%rbp
  400930:	48 83 ec 20          	sub    $0x20,%rsp
  400934:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400938:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40093c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40093f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400943:	48 89 c7             	mov    %rax,%rdi
  400946:	e8 5f 01 00 00       	callq  400aaa <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40094b:	8b 10                	mov    (%rax),%edx
  40094d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400951:	89 10                	mov    %edx,(%rax)
  400953:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400957:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40095b:	48 89 d6             	mov    %rdx,%rsi
  40095e:	48 89 c7             	mov    %rax,%rdi
  400961:	e8 5e 01 00 00       	callq  400ac4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400966:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096a:	c9                   	leaveq 
  40096b:	c3                   	retq   

000000000040096c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  40096c:	55                   	push   %rbp
  40096d:	48 89 e5             	mov    %rsp,%rbp
  400970:	48 83 ec 20          	sub    $0x20,%rsp
  400974:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400978:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40097c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40097f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400983:	48 89 c7             	mov    %rax,%rdi
  400986:	e8 49 01 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40098b:	8b 10                	mov    (%rax),%edx
  40098d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400991:	89 10                	mov    %edx,(%rax)
  400993:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400997:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40099b:	48 89 d6             	mov    %rdx,%rsi
  40099e:	48 89 c7             	mov    %rax,%rdi
  4009a1:	e8 48 01 00 00       	callq  400aee <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009aa:	c9                   	leaveq 
  4009ab:	c3                   	retq   

00000000004009ac <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  4009ac:	55                   	push   %rbp
  4009ad:	48 89 e5             	mov    %rsp,%rbp
  4009b0:	e8 21 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009b5:	48 89 c6             	mov    %rax,%rsi
  4009b8:	bf 20 00 00 00       	mov    $0x20,%edi
  4009bd:	e8 10 fe ff ff       	callq  4007d2 <_ZnwmPv>
  4009c2:	48 85 c0             	test   %rax,%rax
  4009c5:	74 08                	je     4009cf <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  4009c7:	48 89 c7             	mov    %rax,%rdi
  4009ca:	e8 4b 01 00 00       	callq  400b1a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  4009cf:	90                   	nop
  4009d0:	5d                   	pop    %rbp
  4009d1:	c3                   	retq   

00000000004009d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  4009d2:	55                   	push   %rbp
  4009d3:	48 89 e5             	mov    %rsp,%rbp
  4009d6:	48 83 ec 10          	sub    $0x10,%rsp
  4009da:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009de:	e8 83 01 00 00       	callq  400b66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009e3:	e8 ee fb ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  4009e8:	be 01 00 00 00       	mov    $0x1,%esi
  4009ed:	48 89 c7             	mov    %rax,%rdi
  4009f0:	e8 05 fe ff ff       	callq  4007fa <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009f5:	48 89 c7             	mov    %rax,%rdi
  4009f8:	e8 70 01 00 00       	callq  400b6d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  4009fd:	c9                   	leaveq 
  4009fe:	c3                   	retq   
  4009ff:	90                   	nop

0000000000400a00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a08:	90                   	nop
  400a09:	5d                   	pop    %rbp
  400a0a:	c3                   	retq   
  400a0b:	90                   	nop

0000000000400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a0c:	55                   	push   %rbp
  400a0d:	48 89 e5             	mov    %rsp,%rbp
  400a10:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a14:	90                   	nop
  400a15:	5d                   	pop    %rbp
  400a16:	c3                   	retq   

0000000000400a17 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a17:	55                   	push   %rbp
  400a18:	48 89 e5             	mov    %rsp,%rbp
  400a1b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1f:	90                   	nop
  400a20:	5d                   	pop    %rbp
  400a21:	c3                   	retq   

0000000000400a22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a22:	55                   	push   %rbp
  400a23:	48 89 e5             	mov    %rsp,%rbp
  400a26:	48 83 ec 10          	sub    $0x10,%rsp
  400a2a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2e:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a31:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a35:	48 89 c7             	mov    %rax,%rdi
  400a38:	e8 6d 00 00 00       	callq  400aaa <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3d:	48 89 c2             	mov    %rax,%rdx
  400a40:	8b 45 f4             	mov    -0xc(%rbp),%eax
  400a43:	89 02                	mov    %eax,(%rdx)
  400a45:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a49:	c9                   	leaveq 
  400a4a:	c3                   	retq   
  400a4b:	90                   	nop

0000000000400a4c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a4c:	55                   	push   %rbp
  400a4d:	48 89 e5             	mov    %rsp,%rbp
  400a50:	48 83 ec 10          	sub    $0x10,%rsp
  400a54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a58:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a5b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a5f:	48 89 c7             	mov    %rax,%rdi
  400a62:	e8 6d 00 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a67:	48 89 c2             	mov    %rax,%rdx
  400a6a:	8b 45 f4             	mov    -0xc(%rbp),%eax
  400a6d:	89 02                	mov    %eax,(%rdx)
  400a6f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a73:	c9                   	leaveq 
  400a74:	c3                   	retq   
  400a75:	90                   	nop

0000000000400a76 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400a76:	55                   	push   %rbp
  400a77:	48 89 e5             	mov    %rsp,%rbp
  400a7a:	48 83 ec 10          	sub    $0x10,%rsp
  400a7e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a82:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a86:	48 89 c7             	mov    %rax,%rdi
  400a89:	e8 1c 00 00 00       	callq  400aaa <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a8e:	c9                   	leaveq 
  400a8f:	c3                   	retq   

0000000000400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400a90:	55                   	push   %rbp
  400a91:	48 89 e5             	mov    %rsp,%rbp
  400a94:	48 83 ec 10          	sub    $0x10,%rsp
  400a98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a9c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa0:	48 89 c7             	mov    %rax,%rdi
  400aa3:	e8 2c 00 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa8:	c9                   	leaveq 
  400aa9:	c3                   	retq   

0000000000400aaa <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aaa:	55                   	push   %rbp
  400aab:	48 89 e5             	mov    %rsp,%rbp
  400aae:	48 83 ec 10          	sub    $0x10,%rsp
  400ab2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aba:	48 89 c7             	mov    %rax,%rdi
  400abd:	e8 ca 00 00 00       	callq  400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac2:	c9                   	leaveq 
  400ac3:	c3                   	retq   

0000000000400ac4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400ac4:	55                   	push   %rbp
  400ac5:	48 89 e5             	mov    %rsp,%rbp
  400ac8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400acc:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ad0:	90                   	nop
  400ad1:	5d                   	pop    %rbp
  400ad2:	c3                   	retq   
  400ad3:	90                   	nop

0000000000400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ad4:	55                   	push   %rbp
  400ad5:	48 89 e5             	mov    %rsp,%rbp
  400ad8:	48 83 ec 10          	sub    $0x10,%rsp
  400adc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae4:	48 89 c7             	mov    %rax,%rdi
  400ae7:	e8 ee 00 00 00       	callq  400bda <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aec:	c9                   	leaveq 
  400aed:	c3                   	retq   

0000000000400aee <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400aee:	55                   	push   %rbp
  400aef:	48 89 e5             	mov    %rsp,%rbp
  400af2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400afa:	90                   	nop
  400afb:	5d                   	pop    %rbp
  400afc:	c3                   	retq   
  400afd:	90                   	nop

0000000000400afe <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400afe:	55                   	push   %rbp
  400aff:	48 89 e5             	mov    %rsp,%rbp
  400b02:	48 83 ec 10          	sub    $0x10,%rsp
  400b06:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b0e:	48 89 c7             	mov    %rax,%rdi
  400b11:	e8 ce fc ff ff       	callq  4007e4 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400b16:	90                   	nop
  400b17:	c9                   	leaveq 
  400b18:	c3                   	retq   
  400b19:	90                   	nop

0000000000400b1a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400b1a:	55                   	push   %rbp
  400b1b:	48 89 e5             	mov    %rsp,%rbp
  400b1e:	48 83 ec 10          	sub    $0x10,%rsp
  400b22:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b26:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b2a:	48 89 c7             	mov    %rax,%rdi
  400b2d:	e8 cc ff ff ff       	callq  400afe <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400b32:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400b37:	e8 84 f9 ff ff       	callq  4004c0 <malloc@plt>
  400b3c:	48 89 c2             	mov    %rax,%rdx
  400b3f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b43:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400b47:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b4b:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b52:	00 
  400b53:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b57:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400b5b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b5f:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b63:	90                   	nop
  400b64:	c9                   	leaveq 
  400b65:	c3                   	retq   

0000000000400b66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b66:	55                   	push   %rbp
  400b67:	48 89 e5             	mov    %rsp,%rbp
  400b6a:	90                   	nop
  400b6b:	5d                   	pop    %rbp
  400b6c:	c3                   	retq   

0000000000400b6d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400b6d:	55                   	push   %rbp
  400b6e:	48 89 e5             	mov    %rsp,%rbp
  400b71:	48 83 ec 10          	sub    $0x10,%rsp
  400b75:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b79:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b7d:	48 83 c0 01          	add    $0x1,%rax
  400b81:	48 89 c7             	mov    %rax,%rdi
  400b84:	e8 a5 00 00 00       	callq  400c2e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b89:	c9                   	leaveq 
  400b8a:	c3                   	retq   
  400b8b:	90                   	nop

0000000000400b8c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b8c:	55                   	push   %rbp
  400b8d:	48 89 e5             	mov    %rsp,%rbp
  400b90:	48 83 ec 30          	sub    $0x30,%rsp
  400b94:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b98:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b9c:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400ba0:	e8 31 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400ba5:	48 89 c7             	mov    %rax,%rdi
  400ba8:	e8 b3 00 00 00       	callq  400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bad:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bb1:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bb5:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bb9:	48 83 e8 04          	sub    $0x4,%rax
  400bbd:	48 c1 e8 02          	shr    $0x2,%rax
  400bc1:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bc8:	00 
  400bc9:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bcd:	48 01 d0             	add    %rdx,%rax
  400bd0:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bd4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bd8:	c9                   	leaveq 
  400bd9:	c3                   	retq   

0000000000400bda <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bda:	55                   	push   %rbp
  400bdb:	48 89 e5             	mov    %rsp,%rbp
  400bde:	48 83 ec 30          	sub    $0x30,%rsp
  400be2:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400be6:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bea:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bee:	e8 e3 f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400bf3:	48 89 c7             	mov    %rax,%rdi
  400bf6:	e8 65 00 00 00       	callq  400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bfb:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bff:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c03:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400c07:	48 83 e8 04          	sub    $0x4,%rax
  400c0b:	48 c1 e8 02          	shr    $0x2,%rax
  400c0f:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400c16:	00 
  400c17:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c1b:	48 01 d0             	add    %rdx,%rax
  400c1e:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c24:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c28:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c2c:	c9                   	leaveq 
  400c2d:	c3                   	retq   

0000000000400c2e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c2e:	55                   	push   %rbp
  400c2f:	48 89 e5             	mov    %rsp,%rbp
  400c32:	48 83 ec 20          	sub    $0x20,%rsp
  400c36:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c3a:	e8 97 f9 ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400c3f:	48 89 c7             	mov    %rax,%rdi
  400c42:	e8 19 00 00 00       	callq  400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c47:	48 89 c2             	mov    %rax,%rdx
  400c4a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c4e:	48 c1 e0 02          	shl    $0x2,%rax
  400c52:	48 01 d0             	add    %rdx,%rax
  400c55:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c59:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c5d:	c9                   	leaveq 
  400c5e:	c3                   	retq   
  400c5f:	90                   	nop

0000000000400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c6c:	48 8b 40 18          	mov    0x18(%rax),%rax
  400c70:	5d                   	pop    %rbp
  400c71:	c3                   	retq   
  400c72:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c79:	00 00 00 
  400c7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c80 <__libc_csu_init>:
  400c80:	41 57                	push   %r15
  400c82:	41 56                	push   %r14
  400c84:	41 89 ff             	mov    %edi,%r15d
  400c87:	41 55                	push   %r13
  400c89:	41 54                	push   %r12
  400c8b:	4c 8d 25 7e 11 20 00 	lea    0x20117e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c92:	55                   	push   %rbp
  400c93:	48 8d 2d 7e 11 20 00 	lea    0x20117e(%rip),%rbp        # 601e18 <__init_array_end>
  400c9a:	53                   	push   %rbx
  400c9b:	49 89 f6             	mov    %rsi,%r14
  400c9e:	49 89 d5             	mov    %rdx,%r13
  400ca1:	4c 29 e5             	sub    %r12,%rbp
  400ca4:	48 83 ec 08          	sub    $0x8,%rsp
  400ca8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cac:	e8 af f7 ff ff       	callq  400460 <_init>
  400cb1:	48 85 ed             	test   %rbp,%rbp
  400cb4:	74 20                	je     400cd6 <__libc_csu_init+0x56>
  400cb6:	31 db                	xor    %ebx,%ebx
  400cb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400cbf:	00 
  400cc0:	4c 89 ea             	mov    %r13,%rdx
  400cc3:	4c 89 f6             	mov    %r14,%rsi
  400cc6:	44 89 ff             	mov    %r15d,%edi
  400cc9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ccd:	48 83 c3 01          	add    $0x1,%rbx
  400cd1:	48 39 eb             	cmp    %rbp,%rbx
  400cd4:	75 ea                	jne    400cc0 <__libc_csu_init+0x40>
  400cd6:	48 83 c4 08          	add    $0x8,%rsp
  400cda:	5b                   	pop    %rbx
  400cdb:	5d                   	pop    %rbp
  400cdc:	41 5c                	pop    %r12
  400cde:	41 5d                	pop    %r13
  400ce0:	41 5e                	pop    %r14
  400ce2:	41 5f                	pop    %r15
  400ce4:	c3                   	retq   
  400ce5:	90                   	nop
  400ce6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ced:	00 00 00 

0000000000400cf0 <__libc_csu_fini>:
  400cf0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400cf4 <_fini>:
  400cf4:	48 83 ec 08          	sub    $0x8,%rsp
  400cf8:	48 83 c4 08          	add    $0x8,%rsp
  400cfc:	c3                   	retq   
