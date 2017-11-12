
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
  4004ef:	49 c7 c0 50 0c 40 00 	mov    $0x400c50,%r8
  4004f6:	48 c7 c1 e0 0b 40 00 	mov    $0x400be0,%rcx
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
  4005ea:	e8 5b 03 00 00       	callq  40094a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4005ef:	bf 00 00 00 00       	mov    $0x0,%edi
  4005f4:	e8 77 03 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4005f9:	48 89 c3             	mov    %rax,%rbx
  4005fc:	48 89 df             	mov    %rbx,%rdi
  4005ff:	e8 92 02 00 00       	callq  400896 <_ZN9TestClassC1Ev>
  400604:	48 89 5d e8          	mov    %rbx,-0x18(%rbp)
  400608:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40060c:	be 77 77 00 00       	mov    $0x7777,%esi
  400611:	48 89 c7             	mov    %rax,%rdi
  400614:	e8 a7 03 00 00       	callq  4009c0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400619:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40061d:	be 88 88 00 00       	mov    $0x8888,%esi
  400622:	48 89 c7             	mov    %rax,%rdi
  400625:	e8 c0 03 00 00       	callq  4009ea <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40062a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40062e:	48 89 c7             	mov    %rax,%rdi
  400631:	e8 a4 01 00 00       	callq  4007da <_ZN9TestClass15increase_field0Ev>
  400636:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40063a:	48 89 c7             	mov    %rax,%rdi
  40063d:	e8 e2 01 00 00       	callq  400824 <_ZN9TestClass15increase_field1Ev>
  400642:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400646:	48 89 c7             	mov    %rax,%rdi
  400649:	e8 e0 03 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40064e:	8b 18                	mov    (%rax),%ebx
  400650:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400654:	48 89 c7             	mov    %rax,%rdi
  400657:	e8 b8 03 00 00       	callq  400a14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40065c:	8b 00                	mov    (%rax),%eax
  40065e:	89 da                	mov    %ebx,%edx
  400660:	89 c6                	mov    %eax,%esi
  400662:	bf 74 0c 40 00       	mov    $0x400c74,%edi
  400667:	b8 00 00 00 00       	mov    $0x0,%eax
  40066c:	e8 2f fe ff ff       	callq  4004a0 <printf@plt>
  400671:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400675:	48 89 c7             	mov    %rax,%rdi
  400678:	e8 97 03 00 00       	callq  400a14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40067d:	8b 00                	mov    (%rax),%eax
  40067f:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400684:	75 15                	jne    40069b <main+0xba>
  400686:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40068a:	48 89 c7             	mov    %rax,%rdi
  40068d:	e8 9c 03 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400692:	8b 00                	mov    (%rax),%eax
  400694:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400699:	74 07                	je     4006a2 <main+0xc1>
  40069b:	b8 01 00 00 00       	mov    $0x1,%eax
  4006a0:	eb 05                	jmp    4006a7 <main+0xc6>
  4006a2:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a7:	0f b6 c0             	movzbl %al,%eax
  4006aa:	48 83 c4 18          	add    $0x18,%rsp
  4006ae:	5b                   	pop    %rbx
  4006af:	5d                   	pop    %rbp
  4006b0:	c3                   	retq   

00000000004006b1 <_Z12new_instancev>:
  4006b1:	55                   	push   %rbp
  4006b2:	48 89 e5             	mov    %rsp,%rbp
  4006b5:	53                   	push   %rbx
  4006b6:	48 83 ec 08          	sub    $0x8,%rsp
  4006ba:	bf 00 00 00 00       	mov    $0x0,%edi
  4006bf:	e8 ac 02 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4006c4:	48 89 c3             	mov    %rax,%rbx
  4006c7:	48 89 df             	mov    %rbx,%rdi
  4006ca:	e8 c7 01 00 00       	callq  400896 <_ZN9TestClassC1Ev>
  4006cf:	48 89 d8             	mov    %rbx,%rax
  4006d2:	48 83 c4 08          	add    $0x8,%rsp
  4006d6:	5b                   	pop    %rbx
  4006d7:	5d                   	pop    %rbp
  4006d8:	c3                   	retq   

00000000004006d9 <_Z12write_field0P9TestClass>:
  4006d9:	55                   	push   %rbp
  4006da:	48 89 e5             	mov    %rsp,%rbp
  4006dd:	48 83 ec 10          	sub    $0x10,%rsp
  4006e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4006e5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006e9:	be 77 77 00 00       	mov    $0x7777,%esi
  4006ee:	48 89 c7             	mov    %rax,%rdi
  4006f1:	e8 ca 02 00 00       	callq  4009c0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006f6:	90                   	nop
  4006f7:	c9                   	leaveq 
  4006f8:	c3                   	retq   

00000000004006f9 <_Z11read_field0P9TestClass>:
  4006f9:	55                   	push   %rbp
  4006fa:	48 89 e5             	mov    %rsp,%rbp
  4006fd:	48 83 ec 10          	sub    $0x10,%rsp
  400701:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400705:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400709:	48 89 c7             	mov    %rax,%rdi
  40070c:	e8 03 03 00 00       	callq  400a14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400711:	8b 00                	mov    (%rax),%eax
  400713:	c9                   	leaveq 
  400714:	c3                   	retq   

0000000000400715 <_Z12write_field1P9TestClass>:
  400715:	55                   	push   %rbp
  400716:	48 89 e5             	mov    %rsp,%rbp
  400719:	48 83 ec 10          	sub    $0x10,%rsp
  40071d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400721:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400725:	be 77 77 00 00       	mov    $0x7777,%esi
  40072a:	48 89 c7             	mov    %rax,%rdi
  40072d:	e8 b8 02 00 00       	callq  4009ea <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400732:	90                   	nop
  400733:	c9                   	leaveq 
  400734:	c3                   	retq   

0000000000400735 <_Z11read_field1P9TestClass>:
  400735:	55                   	push   %rbp
  400736:	48 89 e5             	mov    %rsp,%rbp
  400739:	48 83 ec 10          	sub    $0x10,%rsp
  40073d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400741:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400745:	48 89 c7             	mov    %rax,%rdi
  400748:	e8 e1 02 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40074d:	8b 00                	mov    (%rax),%eax
  40074f:	c9                   	leaveq 
  400750:	c3                   	retq   

0000000000400751 <_Z21explicit_write_field0m>:
  400751:	55                   	push   %rbp
  400752:	48 89 e5             	mov    %rsp,%rbp
  400755:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400759:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075d:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  400764:	77 77 00 00 
  400768:	90                   	nop
  400769:	5d                   	pop    %rbp
  40076a:	c3                   	retq   

000000000040076b <_Z20explicit_read_field0m>:
  40076b:	55                   	push   %rbp
  40076c:	48 89 e5             	mov    %rsp,%rbp
  40076f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400773:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400777:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  40077e:	5d                   	pop    %rbp
  40077f:	c3                   	retq   

0000000000400780 <_ZnwmPv>:
  400780:	55                   	push   %rbp
  400781:	48 89 e5             	mov    %rsp,%rbp
  400784:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400788:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40078c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400790:	5d                   	pop    %rbp
  400791:	c3                   	retq   

0000000000400792 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  400792:	55                   	push   %rbp
  400793:	48 89 e5             	mov    %rsp,%rbp
  400796:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40079a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40079e:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007a5:	90                   	nop
  4007a6:	5d                   	pop    %rbp
  4007a7:	c3                   	retq   

00000000004007a8 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007a8:	55                   	push   %rbp
  4007a9:	48 89 e5             	mov    %rsp,%rbp
  4007ac:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007b0:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  4007b4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007b8:	48 8b 00             	mov    (%rax),%rax
  4007bb:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007bf:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007c3:	48 8b 10             	mov    (%rax),%rdx
  4007c6:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4007ca:	48 01 c2             	add    %rax,%rdx
  4007cd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007d1:	48 89 10             	mov    %rdx,(%rax)
  4007d4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007d8:	5d                   	pop    %rbp
  4007d9:	c3                   	retq   

00000000004007da <_ZN9TestClass15increase_field0Ev>:
  4007da:	55                   	push   %rbp
  4007db:	48 89 e5             	mov    %rsp,%rbp
  4007de:	48 83 ec 30          	sub    $0x30,%rsp
  4007e2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007e6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007ed:	00 00 
  4007ef:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007f3:	31 c0                	xor    %eax,%eax
  4007f5:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4007f9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007fd:	ba 55 55 00 00       	mov    $0x5555,%edx
  400802:	48 89 ce             	mov    %rcx,%rsi
  400805:	48 89 c7             	mov    %rax,%rdi
  400808:	e8 bd 00 00 00       	callq  4008ca <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40080d:	90                   	nop
  40080e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400812:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400819:	00 00 
  40081b:	74 05                	je     400822 <_ZN9TestClass15increase_field0Ev+0x48>
  40081d:	e8 6e fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  400822:	c9                   	leaveq 
  400823:	c3                   	retq   

0000000000400824 <_ZN9TestClass15increase_field1Ev>:
  400824:	55                   	push   %rbp
  400825:	48 89 e5             	mov    %rsp,%rbp
  400828:	48 83 ec 30          	sub    $0x30,%rsp
  40082c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400830:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400837:	00 00 
  400839:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40083d:	31 c0                	xor    %eax,%eax
  40083f:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400843:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400847:	ba 44 44 00 00       	mov    $0x4444,%edx
  40084c:	48 89 ce             	mov    %rcx,%rsi
  40084f:	48 89 c7             	mov    %rax,%rdi
  400852:	e8 b3 00 00 00       	callq  40090a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400857:	90                   	nop
  400858:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40085c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400863:	00 00 
  400865:	74 05                	je     40086c <_ZN9TestClass15increase_field1Ev+0x48>
  400867:	e8 24 fc ff ff       	callq  400490 <__stack_chk_fail@plt>
  40086c:	c9                   	leaveq 
  40086d:	c3                   	retq   

000000000040086e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  40086e:	55                   	push   %rbp
  40086f:	48 89 e5             	mov    %rsp,%rbp
  400872:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400876:	90                   	nop
  400877:	5d                   	pop    %rbp
  400878:	c3                   	retq   
  400879:	90                   	nop

000000000040087a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  40087a:	55                   	push   %rbp
  40087b:	48 89 e5             	mov    %rsp,%rbp
  40087e:	48 83 ec 10          	sub    $0x10,%rsp
  400882:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400886:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40088a:	48 89 c7             	mov    %rax,%rdi
  40088d:	e8 dc ff ff ff       	callq  40086e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400892:	90                   	nop
  400893:	c9                   	leaveq 
  400894:	c3                   	retq   
  400895:	90                   	nop

0000000000400896 <_ZN9TestClassC1Ev>:
  400896:	55                   	push   %rbp
  400897:	48 89 e5             	mov    %rsp,%rbp
  40089a:	48 83 ec 10          	sub    $0x10,%rsp
  40089e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a6:	48 89 c7             	mov    %rax,%rdi
  4008a9:	e8 cc ff ff ff       	callq  40087a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  4008ae:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b2:	48 89 c7             	mov    %rax,%rdi
  4008b5:	e8 e4 00 00 00       	callq  40099e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008ba:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008be:	48 89 c7             	mov    %rax,%rdi
  4008c1:	e8 e4 00 00 00       	callq  4009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008c6:	90                   	nop
  4008c7:	c9                   	leaveq 
  4008c8:	c3                   	retq   
  4008c9:	90                   	nop

00000000004008ca <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4008ca:	55                   	push   %rbp
  4008cb:	48 89 e5             	mov    %rsp,%rbp
  4008ce:	48 83 ec 20          	sub    $0x20,%rsp
  4008d2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008da:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008dd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008e1:	48 89 c7             	mov    %rax,%rdi
  4008e4:	e8 5f 01 00 00       	callq  400a48 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008e9:	8b 10                	mov    (%rax),%edx
  4008eb:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008ef:	89 10                	mov    %edx,(%rax)
  4008f1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008f5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f9:	48 89 d6             	mov    %rdx,%rsi
  4008fc:	48 89 c7             	mov    %rax,%rdi
  4008ff:	e8 5e 01 00 00       	callq  400a62 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400904:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400908:	c9                   	leaveq 
  400909:	c3                   	retq   

000000000040090a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  40090a:	55                   	push   %rbp
  40090b:	48 89 e5             	mov    %rsp,%rbp
  40090e:	48 83 ec 20          	sub    $0x20,%rsp
  400912:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400916:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40091a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40091d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400921:	48 89 c7             	mov    %rax,%rdi
  400924:	e8 49 01 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400929:	8b 10                	mov    (%rax),%edx
  40092b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40092f:	89 10                	mov    %edx,(%rax)
  400931:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400935:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400939:	48 89 d6             	mov    %rdx,%rsi
  40093c:	48 89 c7             	mov    %rax,%rdi
  40093f:	e8 48 01 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400944:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400948:	c9                   	leaveq 
  400949:	c3                   	retq   

000000000040094a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  40094a:	55                   	push   %rbp
  40094b:	48 89 e5             	mov    %rsp,%rbp
  40094e:	e8 83 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400953:	48 89 c6             	mov    %rax,%rsi
  400956:	bf 20 00 00 00       	mov    $0x20,%edi
  40095b:	e8 20 fe ff ff       	callq  400780 <_ZnwmPv>
  400960:	48 85 c0             	test   %rax,%rax
  400963:	74 08                	je     40096d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400965:	48 89 c7             	mov    %rax,%rdi
  400968:	e8 4b 01 00 00       	callq  400ab8 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  40096d:	90                   	nop
  40096e:	5d                   	pop    %rbp
  40096f:	c3                   	retq   

0000000000400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 83 ec 10          	sub    $0x10,%rsp
  400978:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097c:	e8 83 01 00 00       	callq  400b04 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400981:	e8 50 fc ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400986:	be 01 00 00 00       	mov    $0x1,%esi
  40098b:	48 89 c7             	mov    %rax,%rdi
  40098e:	e8 15 fe ff ff       	callq  4007a8 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400993:	48 89 c7             	mov    %rax,%rdi
  400996:	e8 70 01 00 00       	callq  400b0b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  40099b:	c9                   	leaveq 
  40099c:	c3                   	retq   
  40099d:	90                   	nop

000000000040099e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  40099e:	55                   	push   %rbp
  40099f:	48 89 e5             	mov    %rsp,%rbp
  4009a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a6:	90                   	nop
  4009a7:	5d                   	pop    %rbp
  4009a8:	c3                   	retq   
  4009a9:	90                   	nop

00000000004009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b2:	90                   	nop
  4009b3:	5d                   	pop    %rbp
  4009b4:	c3                   	retq   

00000000004009b5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  4009b5:	55                   	push   %rbp
  4009b6:	48 89 e5             	mov    %rsp,%rbp
  4009b9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bd:	90                   	nop
  4009be:	5d                   	pop    %rbp
  4009bf:	c3                   	retq   

00000000004009c0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 83 ec 10          	sub    $0x10,%rsp
  4009c8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009cc:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4009cf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d3:	48 89 c7             	mov    %rax,%rdi
  4009d6:	e8 6d 00 00 00       	callq  400a48 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009db:	48 89 c2             	mov    %rax,%rdx
  4009de:	8b 45 f4             	mov    -0xc(%rbp),%eax
  4009e1:	89 02                	mov    %eax,(%rdx)
  4009e3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e7:	c9                   	leaveq 
  4009e8:	c3                   	retq   
  4009e9:	90                   	nop

00000000004009ea <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  4009ea:	55                   	push   %rbp
  4009eb:	48 89 e5             	mov    %rsp,%rbp
  4009ee:	48 83 ec 10          	sub    $0x10,%rsp
  4009f2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f6:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4009f9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009fd:	48 89 c7             	mov    %rax,%rdi
  400a00:	e8 6d 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a05:	48 89 c2             	mov    %rax,%rdx
  400a08:	8b 45 f4             	mov    -0xc(%rbp),%eax
  400a0b:	89 02                	mov    %eax,(%rdx)
  400a0d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a11:	c9                   	leaveq 
  400a12:	c3                   	retq   
  400a13:	90                   	nop

0000000000400a14 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a14:	55                   	push   %rbp
  400a15:	48 89 e5             	mov    %rsp,%rbp
  400a18:	48 83 ec 10          	sub    $0x10,%rsp
  400a1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a20:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a24:	48 89 c7             	mov    %rax,%rdi
  400a27:	e8 1c 00 00 00       	callq  400a48 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a2c:	c9                   	leaveq 
  400a2d:	c3                   	retq   

0000000000400a2e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a2e:	55                   	push   %rbp
  400a2f:	48 89 e5             	mov    %rsp,%rbp
  400a32:	48 83 ec 10          	sub    $0x10,%rsp
  400a36:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a3e:	48 89 c7             	mov    %rax,%rdi
  400a41:	e8 2c 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a46:	c9                   	leaveq 
  400a47:	c3                   	retq   

0000000000400a48 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a48:	55                   	push   %rbp
  400a49:	48 89 e5             	mov    %rsp,%rbp
  400a4c:	48 83 ec 10          	sub    $0x10,%rsp
  400a50:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a54:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a58:	48 89 c7             	mov    %rax,%rdi
  400a5b:	e8 ca 00 00 00       	callq  400b2a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a60:	c9                   	leaveq 
  400a61:	c3                   	retq   

0000000000400a62 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a62:	55                   	push   %rbp
  400a63:	48 89 e5             	mov    %rsp,%rbp
  400a66:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a6e:	90                   	nop
  400a6f:	5d                   	pop    %rbp
  400a70:	c3                   	retq   
  400a71:	90                   	nop

0000000000400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a72:	55                   	push   %rbp
  400a73:	48 89 e5             	mov    %rsp,%rbp
  400a76:	48 83 ec 10          	sub    $0x10,%rsp
  400a7a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a7e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a82:	48 89 c7             	mov    %rax,%rdi
  400a85:	e8 e2 00 00 00       	callq  400b6c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a8a:	c9                   	leaveq 
  400a8b:	c3                   	retq   

0000000000400a8c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400a8c:	55                   	push   %rbp
  400a8d:	48 89 e5             	mov    %rsp,%rbp
  400a90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a94:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a98:	90                   	nop
  400a99:	5d                   	pop    %rbp
  400a9a:	c3                   	retq   
  400a9b:	90                   	nop

0000000000400a9c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400a9c:	55                   	push   %rbp
  400a9d:	48 89 e5             	mov    %rsp,%rbp
  400aa0:	48 83 ec 10          	sub    $0x10,%rsp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aac:	48 89 c7             	mov    %rax,%rdi
  400aaf:	e8 de fc ff ff       	callq  400792 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400ab4:	90                   	nop
  400ab5:	c9                   	leaveq 
  400ab6:	c3                   	retq   
  400ab7:	90                   	nop

0000000000400ab8 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400ab8:	55                   	push   %rbp
  400ab9:	48 89 e5             	mov    %rsp,%rbp
  400abc:	48 83 ec 10          	sub    $0x10,%rsp
  400ac0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac8:	48 89 c7             	mov    %rax,%rdi
  400acb:	e8 cc ff ff ff       	callq  400a9c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400ad0:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400ad5:	e8 e6 f9 ff ff       	callq  4004c0 <malloc@plt>
  400ada:	48 89 c2             	mov    %rax,%rdx
  400add:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae1:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400ae5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae9:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400af0:	00 
  400af1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af5:	48 8b 50 08          	mov    0x8(%rax),%rdx
  400af9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400afd:	48 89 50 10          	mov    %rdx,0x10(%rax)
  400b01:	90                   	nop
  400b02:	c9                   	leaveq 
  400b03:	c3                   	retq   

0000000000400b04 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b04:	55                   	push   %rbp
  400b05:	48 89 e5             	mov    %rsp,%rbp
  400b08:	90                   	nop
  400b09:	5d                   	pop    %rbp
  400b0a:	c3                   	retq   

0000000000400b0b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400b0b:	55                   	push   %rbp
  400b0c:	48 89 e5             	mov    %rsp,%rbp
  400b0f:	48 83 ec 10          	sub    $0x10,%rsp
  400b13:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b17:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1b:	48 83 c0 01          	add    $0x1,%rax
  400b1f:	48 89 c7             	mov    %rax,%rdi
  400b22:	e8 8d 00 00 00       	callq  400bb4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b27:	c9                   	leaveq 
  400b28:	c3                   	retq   
  400b29:	90                   	nop

0000000000400b2a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b2a:	55                   	push   %rbp
  400b2b:	48 89 e5             	mov    %rsp,%rbp
  400b2e:	48 83 ec 30          	sub    $0x30,%rsp
  400b32:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b36:	48 c7 45 e8 18 00 00 	movq   $0x18,-0x18(%rbp)
  400b3d:	00 
  400b3e:	48 c7 45 f0 80 20 60 	movq   $0x602080,-0x10(%rbp)
  400b45:	00 
  400b46:	e8 8b fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b4b:	48 89 c7             	mov    %rax,%rdi
  400b4e:	e8 6f 00 00 00       	callq  400bc2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400b53:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b57:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b5b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b62:	00 
  400b63:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b67:	48 01 d0             	add    %rdx,%rax
  400b6a:	c9                   	leaveq 
  400b6b:	c3                   	retq   

0000000000400b6c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b6c:	55                   	push   %rbp
  400b6d:	48 89 e5             	mov    %rsp,%rbp
  400b70:	48 83 ec 30          	sub    $0x30,%rsp
  400b74:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b78:	48 c7 45 e8 18 00 00 	movq   $0x18,-0x18(%rbp)
  400b7f:	00 
  400b80:	48 c7 45 f0 80 20 60 	movq   $0x602080,-0x10(%rbp)
  400b87:	00 
  400b88:	e8 49 fa ff ff       	callq  4005d6 <_ZN9TestClass7storageEv>
  400b8d:	48 89 c7             	mov    %rax,%rdi
  400b90:	e8 2d 00 00 00       	callq  400bc2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400b95:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b99:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b9d:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400ba4:	00 
  400ba5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ba9:	48 01 d0             	add    %rdx,%rax
  400bac:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bb2:	c9                   	leaveq 
  400bb3:	c3                   	retq   

0000000000400bb4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bb4:	55                   	push   %rbp
  400bb5:	48 89 e5             	mov    %rsp,%rbp
  400bb8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bbc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bc0:	5d                   	pop    %rbp
  400bc1:	c3                   	retq   

0000000000400bc2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400bc2:	55                   	push   %rbp
  400bc3:	48 89 e5             	mov    %rsp,%rbp
  400bc6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bce:	48 8b 40 18          	mov    0x18(%rax),%rax
  400bd2:	5d                   	pop    %rbp
  400bd3:	c3                   	retq   
  400bd4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdb:	00 00 00 
  400bde:	66 90                	xchg   %ax,%ax

0000000000400be0 <__libc_csu_init>:
  400be0:	41 57                	push   %r15
  400be2:	41 56                	push   %r14
  400be4:	41 89 ff             	mov    %edi,%r15d
  400be7:	41 55                	push   %r13
  400be9:	41 54                	push   %r12
  400beb:	4c 8d 25 1e 12 20 00 	lea    0x20121e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400bf2:	55                   	push   %rbp
  400bf3:	48 8d 2d 1e 12 20 00 	lea    0x20121e(%rip),%rbp        # 601e18 <__init_array_end>
  400bfa:	53                   	push   %rbx
  400bfb:	49 89 f6             	mov    %rsi,%r14
  400bfe:	49 89 d5             	mov    %rdx,%r13
  400c01:	4c 29 e5             	sub    %r12,%rbp
  400c04:	48 83 ec 08          	sub    $0x8,%rsp
  400c08:	48 c1 fd 03          	sar    $0x3,%rbp
  400c0c:	e8 4f f8 ff ff       	callq  400460 <_init>
  400c11:	48 85 ed             	test   %rbp,%rbp
  400c14:	74 20                	je     400c36 <__libc_csu_init+0x56>
  400c16:	31 db                	xor    %ebx,%ebx
  400c18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c1f:	00 
  400c20:	4c 89 ea             	mov    %r13,%rdx
  400c23:	4c 89 f6             	mov    %r14,%rsi
  400c26:	44 89 ff             	mov    %r15d,%edi
  400c29:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c2d:	48 83 c3 01          	add    $0x1,%rbx
  400c31:	48 39 eb             	cmp    %rbp,%rbx
  400c34:	75 ea                	jne    400c20 <__libc_csu_init+0x40>
  400c36:	48 83 c4 08          	add    $0x8,%rsp
  400c3a:	5b                   	pop    %rbx
  400c3b:	5d                   	pop    %rbp
  400c3c:	41 5c                	pop    %r12
  400c3e:	41 5d                	pop    %r13
  400c40:	41 5e                	pop    %r14
  400c42:	41 5f                	pop    %r15
  400c44:	c3                   	retq   
  400c45:	90                   	nop
  400c46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c4d:	00 00 00 

0000000000400c50 <__libc_csu_fini>:
  400c50:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c54 <_fini>:
  400c54:	48 83 ec 08          	sub    $0x8,%rsp
  400c58:	48 83 c4 08          	add    $0x8,%rsp
  400c5c:	c3                   	retq   
