
bin/g++_-O0_StaticStorage_0:     file format elf64-x86-64


Disassembly of section .init:

0000000000400428 <_init>:
  400428:	48 83 ec 08          	sub    $0x8,%rsp
  40042c:	48 8b 05 c5 1b 20 00 	mov    0x201bc5(%rip),%rax        # 601ff8 <_DYNAMIC+0x1d0>
  400433:	48 85 c0             	test   %rax,%rax
  400436:	74 05                	je     40043d <_init+0x15>
  400438:	e8 53 00 00 00       	callq  400490 <__libc_start_main@plt+0x10>
  40043d:	48 83 c4 08          	add    $0x8,%rsp
  400441:	c3                   	retq   

Disassembly of section .plt:

0000000000400450 <__stack_chk_fail@plt-0x10>:
  400450:	ff 35 b2 1b 20 00    	pushq  0x201bb2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400456:	ff 25 b4 1b 20 00    	jmpq   *0x201bb4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40045c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400460 <__stack_chk_fail@plt>:
  400460:	ff 25 b2 1b 20 00    	jmpq   *0x201bb2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400466:	68 00 00 00 00       	pushq  $0x0
  40046b:	e9 e0 ff ff ff       	jmpq   400450 <_init+0x28>

0000000000400470 <printf@plt>:
  400470:	ff 25 aa 1b 20 00    	jmpq   *0x201baa(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400476:	68 01 00 00 00       	pushq  $0x1
  40047b:	e9 d0 ff ff ff       	jmpq   400450 <_init+0x28>

0000000000400480 <__libc_start_main@plt>:
  400480:	ff 25 a2 1b 20 00    	jmpq   *0x201ba2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400486:	68 02 00 00 00       	pushq  $0x2
  40048b:	e9 c0 ff ff ff       	jmpq   400450 <_init+0x28>

Disassembly of section .plt.got:

0000000000400490 <.plt.got>:
  400490:	ff 25 62 1b 20 00    	jmpq   *0x201b62(%rip)        # 601ff8 <_DYNAMIC+0x1d0>
  400496:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004a0 <_start>:
  4004a0:	31 ed                	xor    %ebp,%ebp
  4004a2:	49 89 d1             	mov    %rdx,%r9
  4004a5:	5e                   	pop    %rsi
  4004a6:	48 89 e2             	mov    %rsp,%rdx
  4004a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004ad:	50                   	push   %rax
  4004ae:	54                   	push   %rsp
  4004af:	49 c7 c0 c0 0b 40 00 	mov    $0x400bc0,%r8
  4004b6:	48 c7 c1 50 0b 40 00 	mov    $0x400b50,%rcx
  4004bd:	48 c7 c7 a1 05 40 00 	mov    $0x4005a1,%rdi
  4004c4:	e8 b7 ff ff ff       	callq  400480 <__libc_start_main@plt>
  4004c9:	f4                   	hlt    
  4004ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004004d0 <deregister_tm_clones>:
  4004d0:	b8 47 20 60 00       	mov    $0x602047,%eax
  4004d5:	55                   	push   %rbp
  4004d6:	48 2d 40 20 60 00    	sub    $0x602040,%rax
  4004dc:	48 83 f8 0e          	cmp    $0xe,%rax
  4004e0:	48 89 e5             	mov    %rsp,%rbp
  4004e3:	76 1b                	jbe    400500 <deregister_tm_clones+0x30>
  4004e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4004ea:	48 85 c0             	test   %rax,%rax
  4004ed:	74 11                	je     400500 <deregister_tm_clones+0x30>
  4004ef:	5d                   	pop    %rbp
  4004f0:	bf 40 20 60 00       	mov    $0x602040,%edi
  4004f5:	ff e0                	jmpq   *%rax
  4004f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4004fe:	00 00 
  400500:	5d                   	pop    %rbp
  400501:	c3                   	retq   
  400502:	0f 1f 40 00          	nopl   0x0(%rax)
  400506:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40050d:	00 00 00 

0000000000400510 <register_tm_clones>:
  400510:	be 40 20 60 00       	mov    $0x602040,%esi
  400515:	55                   	push   %rbp
  400516:	48 81 ee 40 20 60 00 	sub    $0x602040,%rsi
  40051d:	48 c1 fe 03          	sar    $0x3,%rsi
  400521:	48 89 e5             	mov    %rsp,%rbp
  400524:	48 89 f0             	mov    %rsi,%rax
  400527:	48 c1 e8 3f          	shr    $0x3f,%rax
  40052b:	48 01 c6             	add    %rax,%rsi
  40052e:	48 d1 fe             	sar    %rsi
  400531:	74 15                	je     400548 <register_tm_clones+0x38>
  400533:	b8 00 00 00 00       	mov    $0x0,%eax
  400538:	48 85 c0             	test   %rax,%rax
  40053b:	74 0b                	je     400548 <register_tm_clones+0x38>
  40053d:	5d                   	pop    %rbp
  40053e:	bf 40 20 60 00       	mov    $0x602040,%edi
  400543:	ff e0                	jmpq   *%rax
  400545:	0f 1f 00             	nopl   (%rax)
  400548:	5d                   	pop    %rbp
  400549:	c3                   	retq   
  40054a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400550 <__do_global_dtors_aux>:
  400550:	80 3d e9 1a 20 00 00 	cmpb   $0x0,0x201ae9(%rip)        # 602040 <__TMC_END__>
  400557:	75 11                	jne    40056a <__do_global_dtors_aux+0x1a>
  400559:	55                   	push   %rbp
  40055a:	48 89 e5             	mov    %rsp,%rbp
  40055d:	e8 6e ff ff ff       	callq  4004d0 <deregister_tm_clones>
  400562:	5d                   	pop    %rbp
  400563:	c6 05 d6 1a 20 00 01 	movb   $0x1,0x201ad6(%rip)        # 602040 <__TMC_END__>
  40056a:	f3 c3                	repz retq 
  40056c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400570 <frame_dummy>:
  400570:	bf 20 1e 60 00       	mov    $0x601e20,%edi
  400575:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400579:	75 05                	jne    400580 <frame_dummy+0x10>
  40057b:	eb 93                	jmp    400510 <register_tm_clones>
  40057d:	0f 1f 00             	nopl   (%rax)
  400580:	b8 00 00 00 00       	mov    $0x0,%eax
  400585:	48 85 c0             	test   %rax,%rax
  400588:	74 f1                	je     40057b <frame_dummy+0xb>
  40058a:	55                   	push   %rbp
  40058b:	48 89 e5             	mov    %rsp,%rbp
  40058e:	ff d0                	callq  *%rax
  400590:	5d                   	pop    %rbp
  400591:	e9 7a ff ff ff       	jmpq   400510 <register_tm_clones>

0000000000400596 <_ZN9TestClass7storageEv>:
  400596:	55                   	push   %rbp
  400597:	48 89 e5             	mov    %rsp,%rbp
  40059a:	b8 60 20 60 00       	mov    $0x602060,%eax
  40059f:	5d                   	pop    %rbp
  4005a0:	c3                   	retq   

00000000004005a1 <main>:
  4005a1:	55                   	push   %rbp
  4005a2:	48 89 e5             	mov    %rsp,%rbp
  4005a5:	53                   	push   %rbx
  4005a6:	48 83 ec 18          	sub    $0x18,%rsp
  4005aa:	e8 5b 03 00 00       	callq  40090a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005af:	bf 00 00 00 00       	mov    $0x0,%edi
  4005b4:	e8 77 03 00 00       	callq  400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005b9:	48 89 c3             	mov    %rax,%rbx
  4005bc:	48 89 df             	mov    %rbx,%rdi
  4005bf:	e8 92 02 00 00       	callq  400856 <_ZN9TestClassC1Ev>
  4005c4:	48 89 5d e8          	mov    %rbx,-0x18(%rbp)
  4005c8:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005cc:	be 77 77 00 00       	mov    $0x7777,%esi
  4005d1:	48 89 c7             	mov    %rax,%rdi
  4005d4:	e8 a7 03 00 00       	callq  400980 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005d9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005dd:	be 88 88 00 00       	mov    $0x8888,%esi
  4005e2:	48 89 c7             	mov    %rax,%rdi
  4005e5:	e8 c0 03 00 00       	callq  4009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4005ea:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005ee:	48 89 c7             	mov    %rax,%rdi
  4005f1:	e8 a4 01 00 00       	callq  40079a <_ZN9TestClass15increase_field0Ev>
  4005f6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005fa:	48 89 c7             	mov    %rax,%rdi
  4005fd:	e8 e2 01 00 00       	callq  4007e4 <_ZN9TestClass15increase_field1Ev>
  400602:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400606:	48 89 c7             	mov    %rax,%rdi
  400609:	e8 e0 03 00 00       	callq  4009ee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40060e:	8b 18                	mov    (%rax),%ebx
  400610:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400614:	48 89 c7             	mov    %rax,%rdi
  400617:	e8 b8 03 00 00       	callq  4009d4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40061c:	8b 00                	mov    (%rax),%eax
  40061e:	89 da                	mov    %ebx,%edx
  400620:	89 c6                	mov    %eax,%esi
  400622:	bf e4 0b 40 00       	mov    $0x400be4,%edi
  400627:	b8 00 00 00 00       	mov    $0x0,%eax
  40062c:	e8 3f fe ff ff       	callq  400470 <printf@plt>
  400631:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400635:	48 89 c7             	mov    %rax,%rdi
  400638:	e8 97 03 00 00       	callq  4009d4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40063d:	8b 00                	mov    (%rax),%eax
  40063f:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400644:	75 15                	jne    40065b <main+0xba>
  400646:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40064a:	48 89 c7             	mov    %rax,%rdi
  40064d:	e8 9c 03 00 00       	callq  4009ee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400652:	8b 00                	mov    (%rax),%eax
  400654:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400659:	74 07                	je     400662 <main+0xc1>
  40065b:	b8 01 00 00 00       	mov    $0x1,%eax
  400660:	eb 05                	jmp    400667 <main+0xc6>
  400662:	b8 00 00 00 00       	mov    $0x0,%eax
  400667:	0f b6 c0             	movzbl %al,%eax
  40066a:	48 83 c4 18          	add    $0x18,%rsp
  40066e:	5b                   	pop    %rbx
  40066f:	5d                   	pop    %rbp
  400670:	c3                   	retq   

0000000000400671 <_Z12new_instancev>:
  400671:	55                   	push   %rbp
  400672:	48 89 e5             	mov    %rsp,%rbp
  400675:	53                   	push   %rbx
  400676:	48 83 ec 08          	sub    $0x8,%rsp
  40067a:	bf 00 00 00 00       	mov    $0x0,%edi
  40067f:	e8 ac 02 00 00       	callq  400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400684:	48 89 c3             	mov    %rax,%rbx
  400687:	48 89 df             	mov    %rbx,%rdi
  40068a:	e8 c7 01 00 00       	callq  400856 <_ZN9TestClassC1Ev>
  40068f:	48 89 d8             	mov    %rbx,%rax
  400692:	48 83 c4 08          	add    $0x8,%rsp
  400696:	5b                   	pop    %rbx
  400697:	5d                   	pop    %rbp
  400698:	c3                   	retq   

0000000000400699 <_Z12write_field0P9TestClass>:
  400699:	55                   	push   %rbp
  40069a:	48 89 e5             	mov    %rsp,%rbp
  40069d:	48 83 ec 10          	sub    $0x10,%rsp
  4006a1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4006a5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006a9:	be 77 77 00 00       	mov    $0x7777,%esi
  4006ae:	48 89 c7             	mov    %rax,%rdi
  4006b1:	e8 ca 02 00 00       	callq  400980 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006b6:	90                   	nop
  4006b7:	c9                   	leaveq 
  4006b8:	c3                   	retq   

00000000004006b9 <_Z11read_field0P9TestClass>:
  4006b9:	55                   	push   %rbp
  4006ba:	48 89 e5             	mov    %rsp,%rbp
  4006bd:	48 83 ec 10          	sub    $0x10,%rsp
  4006c1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4006c5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006c9:	48 89 c7             	mov    %rax,%rdi
  4006cc:	e8 03 03 00 00       	callq  4009d4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4006d1:	8b 00                	mov    (%rax),%eax
  4006d3:	c9                   	leaveq 
  4006d4:	c3                   	retq   

00000000004006d5 <_Z12write_field1P9TestClass>:
  4006d5:	55                   	push   %rbp
  4006d6:	48 89 e5             	mov    %rsp,%rbp
  4006d9:	48 83 ec 10          	sub    $0x10,%rsp
  4006dd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4006e1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006e5:	be 77 77 00 00       	mov    $0x7777,%esi
  4006ea:	48 89 c7             	mov    %rax,%rdi
  4006ed:	e8 b8 02 00 00       	callq  4009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4006f2:	90                   	nop
  4006f3:	c9                   	leaveq 
  4006f4:	c3                   	retq   

00000000004006f5 <_Z11read_field1P9TestClass>:
  4006f5:	55                   	push   %rbp
  4006f6:	48 89 e5             	mov    %rsp,%rbp
  4006f9:	48 83 ec 10          	sub    $0x10,%rsp
  4006fd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400701:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400705:	48 89 c7             	mov    %rax,%rdi
  400708:	e8 e1 02 00 00       	callq  4009ee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40070d:	8b 00                	mov    (%rax),%eax
  40070f:	c9                   	leaveq 
  400710:	c3                   	retq   

0000000000400711 <_Z21explicit_write_field0m>:
  400711:	55                   	push   %rbp
  400712:	48 89 e5             	mov    %rsp,%rbp
  400715:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400719:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40071d:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  400724:	77 77 00 00 
  400728:	90                   	nop
  400729:	5d                   	pop    %rbp
  40072a:	c3                   	retq   

000000000040072b <_Z20explicit_read_field0m>:
  40072b:	55                   	push   %rbp
  40072c:	48 89 e5             	mov    %rsp,%rbp
  40072f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400733:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400737:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  40073e:	5d                   	pop    %rbp
  40073f:	c3                   	retq   

0000000000400740 <_ZnwmPv>:
  400740:	55                   	push   %rbp
  400741:	48 89 e5             	mov    %rsp,%rbp
  400744:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400748:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40074c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400750:	5d                   	pop    %rbp
  400751:	c3                   	retq   

0000000000400752 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  400752:	55                   	push   %rbp
  400753:	48 89 e5             	mov    %rsp,%rbp
  400756:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40075a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075e:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400765:	90                   	nop
  400766:	5d                   	pop    %rbp
  400767:	c3                   	retq   

0000000000400768 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400768:	55                   	push   %rbp
  400769:	48 89 e5             	mov    %rsp,%rbp
  40076c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400770:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  400774:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400778:	48 8b 00             	mov    (%rax),%rax
  40077b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40077f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400783:	48 8b 10             	mov    (%rax),%rdx
  400786:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40078a:	48 01 c2             	add    %rax,%rdx
  40078d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400791:	48 89 10             	mov    %rdx,(%rax)
  400794:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400798:	5d                   	pop    %rbp
  400799:	c3                   	retq   

000000000040079a <_ZN9TestClass15increase_field0Ev>:
  40079a:	55                   	push   %rbp
  40079b:	48 89 e5             	mov    %rsp,%rbp
  40079e:	48 83 ec 30          	sub    $0x30,%rsp
  4007a2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007a6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007ad:	00 00 
  4007af:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007b3:	31 c0                	xor    %eax,%eax
  4007b5:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4007b9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007bd:	ba 55 55 00 00       	mov    $0x5555,%edx
  4007c2:	48 89 ce             	mov    %rcx,%rsi
  4007c5:	48 89 c7             	mov    %rax,%rdi
  4007c8:	e8 bd 00 00 00       	callq  40088a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  4007cd:	90                   	nop
  4007ce:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007d2:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007d9:	00 00 
  4007db:	74 05                	je     4007e2 <_ZN9TestClass15increase_field0Ev+0x48>
  4007dd:	e8 7e fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007e2:	c9                   	leaveq 
  4007e3:	c3                   	retq   

00000000004007e4 <_ZN9TestClass15increase_field1Ev>:
  4007e4:	55                   	push   %rbp
  4007e5:	48 89 e5             	mov    %rsp,%rbp
  4007e8:	48 83 ec 30          	sub    $0x30,%rsp
  4007ec:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007f0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007f7:	00 00 
  4007f9:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007fd:	31 c0                	xor    %eax,%eax
  4007ff:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400803:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400807:	ba 44 44 00 00       	mov    $0x4444,%edx
  40080c:	48 89 ce             	mov    %rcx,%rsi
  40080f:	48 89 c7             	mov    %rax,%rdi
  400812:	e8 b3 00 00 00       	callq  4008ca <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400817:	90                   	nop
  400818:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40081c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400823:	00 00 
  400825:	74 05                	je     40082c <_ZN9TestClass15increase_field1Ev+0x48>
  400827:	e8 34 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  40082c:	c9                   	leaveq 
  40082d:	c3                   	retq   

000000000040082e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  40082e:	55                   	push   %rbp
  40082f:	48 89 e5             	mov    %rsp,%rbp
  400832:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400836:	90                   	nop
  400837:	5d                   	pop    %rbp
  400838:	c3                   	retq   
  400839:	90                   	nop

000000000040083a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  40083a:	55                   	push   %rbp
  40083b:	48 89 e5             	mov    %rsp,%rbp
  40083e:	48 83 ec 10          	sub    $0x10,%rsp
  400842:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400846:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40084a:	48 89 c7             	mov    %rax,%rdi
  40084d:	e8 dc ff ff ff       	callq  40082e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400852:	90                   	nop
  400853:	c9                   	leaveq 
  400854:	c3                   	retq   
  400855:	90                   	nop

0000000000400856 <_ZN9TestClassC1Ev>:
  400856:	55                   	push   %rbp
  400857:	48 89 e5             	mov    %rsp,%rbp
  40085a:	48 83 ec 10          	sub    $0x10,%rsp
  40085e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400862:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400866:	48 89 c7             	mov    %rax,%rdi
  400869:	e8 cc ff ff ff       	callq  40083a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  40086e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400872:	48 89 c7             	mov    %rax,%rdi
  400875:	e8 e4 00 00 00       	callq  40095e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  40087a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40087e:	48 89 c7             	mov    %rax,%rdi
  400881:	e8 e4 00 00 00       	callq  40096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  400886:	90                   	nop
  400887:	c9                   	leaveq 
  400888:	c3                   	retq   
  400889:	90                   	nop

000000000040088a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  40088a:	55                   	push   %rbp
  40088b:	48 89 e5             	mov    %rsp,%rbp
  40088e:	48 83 ec 20          	sub    $0x20,%rsp
  400892:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400896:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40089a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40089d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008a1:	48 89 c7             	mov    %rax,%rdi
  4008a4:	e8 5f 01 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008a9:	8b 10                	mov    (%rax),%edx
  4008ab:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008af:	89 10                	mov    %edx,(%rax)
  4008b1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008b5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b9:	48 89 d6             	mov    %rdx,%rsi
  4008bc:	48 89 c7             	mov    %rax,%rdi
  4008bf:	e8 5e 01 00 00       	callq  400a22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4008c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c8:	c9                   	leaveq 
  4008c9:	c3                   	retq   

00000000004008ca <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  4008ca:	55                   	push   %rbp
  4008cb:	48 89 e5             	mov    %rsp,%rbp
  4008ce:	48 83 ec 20          	sub    $0x20,%rsp
  4008d2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008da:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008dd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008e1:	48 89 c7             	mov    %rax,%rdi
  4008e4:	e8 49 01 00 00       	callq  400a32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008e9:	8b 10                	mov    (%rax),%edx
  4008eb:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008ef:	89 10                	mov    %edx,(%rax)
  4008f1:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008f5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f9:	48 89 d6             	mov    %rdx,%rsi
  4008fc:	48 89 c7             	mov    %rax,%rdi
  4008ff:	e8 48 01 00 00       	callq  400a4c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400904:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400908:	c9                   	leaveq 
  400909:	c3                   	retq   

000000000040090a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  40090a:	55                   	push   %rbp
  40090b:	48 89 e5             	mov    %rsp,%rbp
  40090e:	e8 83 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400913:	48 89 c6             	mov    %rax,%rsi
  400916:	bf b0 91 00 00       	mov    $0x91b0,%edi
  40091b:	e8 20 fe ff ff       	callq  400740 <_ZnwmPv>
  400920:	48 85 c0             	test   %rax,%rax
  400923:	74 08                	je     40092d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400925:	48 89 c7             	mov    %rax,%rdi
  400928:	e8 4b 01 00 00       	callq  400a78 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40092d:	90                   	nop
  40092e:	5d                   	pop    %rbp
  40092f:	c3                   	retq   

0000000000400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	e8 5e 01 00 00       	callq  400a9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400941:	e8 50 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400946:	be 01 00 00 00       	mov    $0x1,%esi
  40094b:	48 89 c7             	mov    %rax,%rdi
  40094e:	e8 15 fe ff ff       	callq  400768 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400953:	48 89 c7             	mov    %rax,%rdi
  400956:	e8 4b 01 00 00       	callq  400aa6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  40095b:	c9                   	leaveq 
  40095c:	c3                   	retq   
  40095d:	90                   	nop

000000000040095e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  40095e:	55                   	push   %rbp
  40095f:	48 89 e5             	mov    %rsp,%rbp
  400962:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400966:	90                   	nop
  400967:	5d                   	pop    %rbp
  400968:	c3                   	retq   
  400969:	90                   	nop

000000000040096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  40096a:	55                   	push   %rbp
  40096b:	48 89 e5             	mov    %rsp,%rbp
  40096e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400972:	90                   	nop
  400973:	5d                   	pop    %rbp
  400974:	c3                   	retq   

0000000000400975 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  400975:	55                   	push   %rbp
  400976:	48 89 e5             	mov    %rsp,%rbp
  400979:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097d:	90                   	nop
  40097e:	5d                   	pop    %rbp
  40097f:	c3                   	retq   

0000000000400980 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 83 ec 10          	sub    $0x10,%rsp
  400988:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40098c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  40098f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400993:	48 89 c7             	mov    %rax,%rdi
  400996:	e8 6d 00 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40099b:	48 89 c2             	mov    %rax,%rdx
  40099e:	8b 45 f4             	mov    -0xc(%rbp),%eax
  4009a1:	89 02                	mov    %eax,(%rdx)
  4009a3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a7:	c9                   	leaveq 
  4009a8:	c3                   	retq   
  4009a9:	90                   	nop

00000000004009aa <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	48 83 ec 10          	sub    $0x10,%rsp
  4009b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b6:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4009b9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009bd:	48 89 c7             	mov    %rax,%rdi
  4009c0:	e8 6d 00 00 00       	callq  400a32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009c5:	48 89 c2             	mov    %rax,%rdx
  4009c8:	8b 45 f4             	mov    -0xc(%rbp),%eax
  4009cb:	89 02                	mov    %eax,(%rdx)
  4009cd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d1:	c9                   	leaveq 
  4009d2:	c3                   	retq   
  4009d3:	90                   	nop

00000000004009d4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  4009d4:	55                   	push   %rbp
  4009d5:	48 89 e5             	mov    %rsp,%rbp
  4009d8:	48 83 ec 10          	sub    $0x10,%rsp
  4009dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e4:	48 89 c7             	mov    %rax,%rdi
  4009e7:	e8 1c 00 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009ec:	c9                   	leaveq 
  4009ed:	c3                   	retq   

00000000004009ee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  4009ee:	55                   	push   %rbp
  4009ef:	48 89 e5             	mov    %rsp,%rbp
  4009f2:	48 83 ec 10          	sub    $0x10,%rsp
  4009f6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009fe:	48 89 c7             	mov    %rax,%rdi
  400a01:	e8 2c 00 00 00       	callq  400a32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a06:	c9                   	leaveq 
  400a07:	c3                   	retq   

0000000000400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a08:	55                   	push   %rbp
  400a09:	48 89 e5             	mov    %rsp,%rbp
  400a0c:	48 83 ec 10          	sub    $0x10,%rsp
  400a10:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a18:	48 89 c7             	mov    %rax,%rdi
  400a1b:	e8 a4 00 00 00       	callq  400ac4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a20:	c9                   	leaveq 
  400a21:	c3                   	retq   

0000000000400a22 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a22:	55                   	push   %rbp
  400a23:	48 89 e5             	mov    %rsp,%rbp
  400a26:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a2e:	90                   	nop
  400a2f:	5d                   	pop    %rbp
  400a30:	c3                   	retq   
  400a31:	90                   	nop

0000000000400a32 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a32:	55                   	push   %rbp
  400a33:	48 89 e5             	mov    %rsp,%rbp
  400a36:	48 83 ec 10          	sub    $0x10,%rsp
  400a3a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a42:	48 89 c7             	mov    %rax,%rdi
  400a45:	e8 b6 00 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a4a:	c9                   	leaveq 
  400a4b:	c3                   	retq   

0000000000400a4c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400a4c:	55                   	push   %rbp
  400a4d:	48 89 e5             	mov    %rsp,%rbp
  400a50:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a54:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a58:	90                   	nop
  400a59:	5d                   	pop    %rbp
  400a5a:	c3                   	retq   
  400a5b:	90                   	nop

0000000000400a5c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400a5c:	55                   	push   %rbp
  400a5d:	48 89 e5             	mov    %rsp,%rbp
  400a60:	48 83 ec 10          	sub    $0x10,%rsp
  400a64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a68:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a6c:	48 89 c7             	mov    %rax,%rdi
  400a6f:	e8 de fc ff ff       	callq  400752 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400a74:	90                   	nop
  400a75:	c9                   	leaveq 
  400a76:	c3                   	retq   
  400a77:	90                   	nop

0000000000400a78 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400a78:	55                   	push   %rbp
  400a79:	48 89 e5             	mov    %rsp,%rbp
  400a7c:	48 83 ec 10          	sub    $0x10,%rsp
  400a80:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a84:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a88:	48 89 c7             	mov    %rax,%rdi
  400a8b:	e8 cc ff ff ff       	callq  400a5c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400a90:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a94:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400a9b:	00 
  400a9c:	90                   	nop
  400a9d:	c9                   	leaveq 
  400a9e:	c3                   	retq   

0000000000400a9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400a9f:	55                   	push   %rbp
  400aa0:	48 89 e5             	mov    %rsp,%rbp
  400aa3:	90                   	nop
  400aa4:	5d                   	pop    %rbp
  400aa5:	c3                   	retq   

0000000000400aa6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400aa6:	55                   	push   %rbp
  400aa7:	48 89 e5             	mov    %rsp,%rbp
  400aaa:	48 83 ec 10          	sub    $0x10,%rsp
  400aae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab6:	48 83 c0 01          	add    $0x1,%rax
  400aba:	48 89 c7             	mov    %rax,%rdi
  400abd:	e8 7c 00 00 00       	callq  400b3e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400ac2:	c9                   	leaveq 
  400ac3:	c3                   	retq   

0000000000400ac4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ac4:	55                   	push   %rbp
  400ac5:	48 89 e5             	mov    %rsp,%rbp
  400ac8:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400acc:	48 c7 45 e0 10 00 00 	movq   $0x10,-0x20(%rbp)
  400ad3:	00 
  400ad4:	48 c7 45 e8 60 20 60 	movq   $0x602060,-0x18(%rbp)
  400adb:	00 
  400adc:	48 c7 45 f0 70 20 60 	movq   $0x602070,-0x10(%rbp)
  400ae3:	00 
  400ae4:	48 c7 45 f8 70 20 60 	movq   $0x602070,-0x8(%rbp)
  400aeb:	00 
  400aec:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400af0:	48 83 c0 04          	add    $0x4,%rax
  400af4:	48 c1 e0 02          	shl    $0x2,%rax
  400af8:	48 05 60 20 60 00    	add    $0x602060,%rax
  400afe:	5d                   	pop    %rbp
  400aff:	c3                   	retq   

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b08:	48 c7 45 e0 10 00 00 	movq   $0x10,-0x20(%rbp)
  400b0f:	00 
  400b10:	48 c7 45 e8 60 20 60 	movq   $0x602060,-0x18(%rbp)
  400b17:	00 
  400b18:	48 c7 45 f0 40 69 60 	movq   $0x606940,-0x10(%rbp)
  400b1f:	00 
  400b20:	48 c7 45 f8 40 69 60 	movq   $0x606940,-0x8(%rbp)
  400b27:	00 
  400b28:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b2c:	48 05 38 12 00 00    	add    $0x1238,%rax
  400b32:	48 c1 e0 02          	shl    $0x2,%rax
  400b36:	48 05 60 20 60 00    	add    $0x602060,%rax
  400b3c:	5d                   	pop    %rbp
  400b3d:	c3                   	retq   

0000000000400b3e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b3e:	55                   	push   %rbp
  400b3f:	48 89 e5             	mov    %rsp,%rbp
  400b42:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b46:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b4a:	5d                   	pop    %rbp
  400b4b:	c3                   	retq   
  400b4c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b50 <__libc_csu_init>:
  400b50:	41 57                	push   %r15
  400b52:	41 56                	push   %r14
  400b54:	41 89 ff             	mov    %edi,%r15d
  400b57:	41 55                	push   %r13
  400b59:	41 54                	push   %r12
  400b5b:	4c 8d 25 ae 12 20 00 	lea    0x2012ae(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400b62:	55                   	push   %rbp
  400b63:	48 8d 2d ae 12 20 00 	lea    0x2012ae(%rip),%rbp        # 601e18 <__init_array_end>
  400b6a:	53                   	push   %rbx
  400b6b:	49 89 f6             	mov    %rsi,%r14
  400b6e:	49 89 d5             	mov    %rdx,%r13
  400b71:	4c 29 e5             	sub    %r12,%rbp
  400b74:	48 83 ec 08          	sub    $0x8,%rsp
  400b78:	48 c1 fd 03          	sar    $0x3,%rbp
  400b7c:	e8 a7 f8 ff ff       	callq  400428 <_init>
  400b81:	48 85 ed             	test   %rbp,%rbp
  400b84:	74 20                	je     400ba6 <__libc_csu_init+0x56>
  400b86:	31 db                	xor    %ebx,%ebx
  400b88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b8f:	00 
  400b90:	4c 89 ea             	mov    %r13,%rdx
  400b93:	4c 89 f6             	mov    %r14,%rsi
  400b96:	44 89 ff             	mov    %r15d,%edi
  400b99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400b9d:	48 83 c3 01          	add    $0x1,%rbx
  400ba1:	48 39 eb             	cmp    %rbp,%rbx
  400ba4:	75 ea                	jne    400b90 <__libc_csu_init+0x40>
  400ba6:	48 83 c4 08          	add    $0x8,%rsp
  400baa:	5b                   	pop    %rbx
  400bab:	5d                   	pop    %rbp
  400bac:	41 5c                	pop    %r12
  400bae:	41 5d                	pop    %r13
  400bb0:	41 5e                	pop    %r14
  400bb2:	41 5f                	pop    %r15
  400bb4:	c3                   	retq   
  400bb5:	90                   	nop
  400bb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bbd:	00 00 00 

0000000000400bc0 <__libc_csu_fini>:
  400bc0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400bc4 <_fini>:
  400bc4:	48 83 ec 08          	sub    $0x8,%rsp
  400bc8:	48 83 c4 08          	add    $0x8,%rsp
  400bcc:	c3                   	retq   
