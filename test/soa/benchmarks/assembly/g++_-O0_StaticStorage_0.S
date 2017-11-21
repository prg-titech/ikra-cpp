
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
  4004af:	49 c7 c0 f0 0b 40 00 	mov    $0x400bf0,%r8
  4004b6:	48 c7 c1 80 0b 40 00 	mov    $0x400b80,%rcx
  4004bd:	48 c7 c7 9c 05 40 00 	mov    $0x40059c,%rdi
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
  400596:	b8 60 20 60 00       	mov    $0x602060,%eax
  40059b:	c3                   	retq   

000000000040059c <main>:
  40059c:	53                   	push   %rbx
  40059d:	48 83 ec 10          	sub    $0x10,%rsp
  4005a1:	e8 95 03 00 00       	callq  40093b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005a6:	bf 00 00 00 00       	mov    $0x0,%edi
  4005ab:	e8 b4 03 00 00       	callq  400964 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005b0:	48 89 c3             	mov    %rax,%rbx
  4005b3:	48 89 df             	mov    %rbx,%rdi
  4005b6:	e8 bb 02 00 00       	callq  400876 <_ZN9TestClassC1Ev>
  4005bb:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4005c0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005c5:	be 77 77 00 00       	mov    $0x7777,%esi
  4005ca:	48 89 c7             	mov    %rax,%rdi
  4005cd:	e8 d6 03 00 00       	callq  4009a8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  4005d2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005d7:	be 88 88 00 00       	mov    $0x8888,%esi
  4005dc:	48 89 c7             	mov    %rax,%rdi
  4005df:	e8 f2 03 00 00       	callq  4009d6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  4005e4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005e9:	48 89 c7             	mov    %rax,%rdi
  4005ec:	e8 99 01 00 00       	callq  40078a <_ZN9TestClass15increase_field0Ev>
  4005f1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005f6:	48 89 c7             	mov    %rax,%rdi
  4005f9:	e8 f0 01 00 00       	callq  4007ee <_ZN9TestClass15increase_field1Ev>
  4005fe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400603:	48 89 c7             	mov    %rax,%rdi
  400606:	e8 15 04 00 00       	callq  400a20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  40060b:	8b 18                	mov    (%rax),%ebx
  40060d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400612:	48 89 c7             	mov    %rax,%rdi
  400615:	e8 ea 03 00 00       	callq  400a04 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  40061a:	8b 00                	mov    (%rax),%eax
  40061c:	89 da                	mov    %ebx,%edx
  40061e:	89 c6                	mov    %eax,%esi
  400620:	bf 1c 0c 40 00       	mov    $0x400c1c,%edi
  400625:	b8 00 00 00 00       	mov    $0x0,%eax
  40062a:	e8 41 fe ff ff       	callq  400470 <printf@plt>
  40062f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400634:	48 89 c7             	mov    %rax,%rdi
  400637:	e8 c8 03 00 00       	callq  400a04 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  40063c:	8b 00                	mov    (%rax),%eax
  40063e:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400643:	75 16                	jne    40065b <main+0xbf>
  400645:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40064a:	48 89 c7             	mov    %rax,%rdi
  40064d:	e8 ce 03 00 00       	callq  400a20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  400652:	8b 00                	mov    (%rax),%eax
  400654:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400659:	74 07                	je     400662 <main+0xc6>
  40065b:	b8 01 00 00 00       	mov    $0x1,%eax
  400660:	eb 05                	jmp    400667 <main+0xcb>
  400662:	b8 00 00 00 00       	mov    $0x0,%eax
  400667:	0f b6 c0             	movzbl %al,%eax
  40066a:	48 83 c4 10          	add    $0x10,%rsp
  40066e:	5b                   	pop    %rbx
  40066f:	c3                   	retq   

0000000000400670 <_Z12new_instancev>:
  400670:	53                   	push   %rbx
  400671:	bf 00 00 00 00       	mov    $0x0,%edi
  400676:	e8 e9 02 00 00       	callq  400964 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40067b:	48 89 c3             	mov    %rax,%rbx
  40067e:	48 89 df             	mov    %rbx,%rdi
  400681:	e8 f0 01 00 00       	callq  400876 <_ZN9TestClassC1Ev>
  400686:	48 89 d8             	mov    %rbx,%rax
  400689:	5b                   	pop    %rbx
  40068a:	c3                   	retq   

000000000040068b <_Z12write_field0P9TestClass>:
  40068b:	48 83 ec 18          	sub    $0x18,%rsp
  40068f:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400694:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400699:	be 77 77 00 00       	mov    $0x7777,%esi
  40069e:	48 89 c7             	mov    %rax,%rdi
  4006a1:	e8 02 03 00 00       	callq  4009a8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  4006a6:	90                   	nop
  4006a7:	48 83 c4 18          	add    $0x18,%rsp
  4006ab:	c3                   	retq   

00000000004006ac <_Z11read_field0P9TestClass>:
  4006ac:	48 83 ec 18          	sub    $0x18,%rsp
  4006b0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006b5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006ba:	48 89 c7             	mov    %rax,%rdi
  4006bd:	e8 42 03 00 00       	callq  400a04 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  4006c2:	8b 00                	mov    (%rax),%eax
  4006c4:	48 83 c4 18          	add    $0x18,%rsp
  4006c8:	c3                   	retq   

00000000004006c9 <_Z12write_field1P9TestClass>:
  4006c9:	48 83 ec 18          	sub    $0x18,%rsp
  4006cd:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006d2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006d7:	be 77 77 00 00       	mov    $0x7777,%esi
  4006dc:	48 89 c7             	mov    %rax,%rdi
  4006df:	e8 f2 02 00 00       	callq  4009d6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  4006e4:	90                   	nop
  4006e5:	48 83 c4 18          	add    $0x18,%rsp
  4006e9:	c3                   	retq   

00000000004006ea <_Z11read_field1P9TestClass>:
  4006ea:	48 83 ec 18          	sub    $0x18,%rsp
  4006ee:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006f3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006f8:	48 89 c7             	mov    %rax,%rdi
  4006fb:	e8 20 03 00 00       	callq  400a20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  400700:	8b 00                	mov    (%rax),%eax
  400702:	48 83 c4 18          	add    $0x18,%rsp
  400706:	c3                   	retq   

0000000000400707 <_Z21explicit_write_field0m>:
  400707:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40070c:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400711:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  400718:	77 77 00 00 
  40071c:	90                   	nop
  40071d:	c3                   	retq   

000000000040071e <_Z20explicit_read_field0m>:
  40071e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400723:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400728:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  40072f:	c3                   	retq   

0000000000400730 <_ZnwmPv>:
  400730:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400735:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  40073a:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40073f:	c3                   	retq   

0000000000400740 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  400740:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400745:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40074a:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400751:	90                   	nop
  400752:	c3                   	retq   
  400753:	90                   	nop

0000000000400754 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400754:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400759:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  40075e:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400763:	48 8b 00             	mov    (%rax),%rax
  400766:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  40076b:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400770:	48 8b 10             	mov    (%rax),%rdx
  400773:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400778:	48 01 c2             	add    %rax,%rdx
  40077b:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400780:	48 89 10             	mov    %rdx,(%rax)
  400783:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400788:	c3                   	retq   
  400789:	90                   	nop

000000000040078a <_ZN9TestClass15increase_field0Ev>:
  40078a:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40078f:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400793:	41 ff 72 f8          	pushq  -0x8(%r10)
  400797:	55                   	push   %rbp
  400798:	48 89 e5             	mov    %rsp,%rbp
  40079b:	41 52                	push   %r10
  40079d:	48 83 ec 48          	sub    $0x48,%rsp
  4007a1:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4007a5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007ac:	00 00 
  4007ae:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007b2:	31 c0                	xor    %eax,%eax
  4007b4:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4007b8:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4007bc:	ba 55 55 00 00       	mov    $0x5555,%edx
  4007c1:	48 89 ce             	mov    %rcx,%rsi
  4007c4:	48 89 c7             	mov    %rax,%rdi
  4007c7:	e8 e0 00 00 00       	callq  4008ac <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>
  4007cc:	90                   	nop
  4007cd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007d1:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007d8:	00 00 
  4007da:	74 05                	je     4007e1 <_ZN9TestClass15increase_field0Ev+0x57>
  4007dc:	e8 7f fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007e1:	48 83 c4 48          	add    $0x48,%rsp
  4007e5:	41 5a                	pop    %r10
  4007e7:	5d                   	pop    %rbp
  4007e8:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4007ec:	c3                   	retq   
  4007ed:	90                   	nop

00000000004007ee <_ZN9TestClass15increase_field1Ev>:
  4007ee:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007f3:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007f7:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007fb:	55                   	push   %rbp
  4007fc:	48 89 e5             	mov    %rsp,%rbp
  4007ff:	41 52                	push   %r10
  400801:	48 83 ec 48          	sub    $0x48,%rsp
  400805:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400809:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400810:	00 00 
  400812:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400816:	31 c0                	xor    %eax,%eax
  400818:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  40081c:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400820:	ba 44 44 00 00       	mov    $0x4444,%edx
  400825:	48 89 ce             	mov    %rcx,%rsi
  400828:	48 89 c7             	mov    %rax,%rdi
  40082b:	e8 c4 00 00 00       	callq  4008f4 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>
  400830:	90                   	nop
  400831:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400835:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40083c:	00 00 
  40083e:	74 05                	je     400845 <_ZN9TestClass15increase_field1Ev+0x57>
  400840:	e8 1b fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400845:	48 83 c4 48          	add    $0x48,%rsp
  400849:	41 5a                	pop    %r10
  40084b:	5d                   	pop    %rbp
  40084c:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400850:	c3                   	retq   
  400851:	90                   	nop

0000000000400852 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400852:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400857:	90                   	nop
  400858:	c3                   	retq   
  400859:	90                   	nop

000000000040085a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  40085a:	48 83 ec 18          	sub    $0x18,%rsp
  40085e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400863:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400868:	48 89 c7             	mov    %rax,%rdi
  40086b:	e8 e2 ff ff ff       	callq  400852 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400870:	90                   	nop
  400871:	48 83 c4 18          	add    $0x18,%rsp
  400875:	c3                   	retq   

0000000000400876 <_ZN9TestClassC1Ev>:
  400876:	48 83 ec 18          	sub    $0x18,%rsp
  40087a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40087f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400884:	48 89 c7             	mov    %rax,%rdi
  400887:	e8 ce ff ff ff       	callq  40085a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  40088c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400891:	48 89 c7             	mov    %rax,%rdi
  400894:	e8 f9 00 00 00       	callq  400992 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC1Ev>
  400899:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40089e:	48 89 c7             	mov    %rax,%rdi
  4008a1:	e8 f4 00 00 00       	callq  40099a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC1Ev>
  4008a6:	90                   	nop
  4008a7:	48 83 c4 18          	add    $0x18,%rsp
  4008ab:	c3                   	retq   

00000000004008ac <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>:
  4008ac:	48 83 ec 28          	sub    $0x28,%rsp
  4008b0:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008b5:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4008ba:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4008be:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4008c3:	48 89 c7             	mov    %rax,%rdi
  4008c6:	e8 71 01 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008cb:	8b 10                	mov    (%rax),%edx
  4008cd:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  4008d2:	89 10                	mov    %edx,(%rax)
  4008d4:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4008d9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4008de:	48 89 d6             	mov    %rdx,%rsi
  4008e1:	48 89 c7             	mov    %rax,%rdi
  4008e4:	e8 6f 01 00 00       	callq  400a58 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC1ERKS3_>
  4008e9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4008ee:	48 83 c4 28          	add    $0x28,%rsp
  4008f2:	c3                   	retq   
  4008f3:	90                   	nop

00000000004008f4 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>:
  4008f4:	48 83 ec 28          	sub    $0x28,%rsp
  4008f8:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008fd:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400902:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400906:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40090b:	48 89 c7             	mov    %rax,%rdi
  40090e:	e8 51 01 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400913:	8b 10                	mov    (%rax),%edx
  400915:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  40091a:	89 10                	mov    %edx,(%rax)
  40091c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400921:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400926:	48 89 d6             	mov    %rdx,%rsi
  400929:	48 89 c7             	mov    %rax,%rdi
  40092c:	e8 4f 01 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC1ERKS3_>
  400931:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400936:	48 83 c4 28          	add    $0x28,%rsp
  40093a:	c3                   	retq   

000000000040093b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  40093b:	48 83 ec 08          	sub    $0x8,%rsp
  40093f:	e8 52 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400944:	48 89 c6             	mov    %rax,%rsi
  400947:	bf b8 91 00 00       	mov    $0x91b8,%edi
  40094c:	e8 df fd ff ff       	callq  400730 <_ZnwmPv>
  400951:	48 85 c0             	test   %rax,%rax
  400954:	74 08                	je     40095e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400956:	48 89 c7             	mov    %rax,%rdi
  400959:	e8 4a 01 00 00       	callq  400aa8 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40095e:	90                   	nop
  40095f:	48 83 c4 08          	add    $0x8,%rsp
  400963:	c3                   	retq   

0000000000400964 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400964:	48 83 ec 18          	sub    $0x18,%rsp
  400968:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40096d:	e8 5f 01 00 00       	callq  400ad1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400972:	e8 1f fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400977:	be 01 00 00 00       	mov    $0x1,%esi
  40097c:	48 89 c7             	mov    %rax,%rdi
  40097f:	e8 d0 fd ff ff       	callq  400754 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400984:	48 89 c7             	mov    %rax,%rdi
  400987:	e8 47 01 00 00       	callq  400ad3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  40098c:	48 83 c4 18          	add    $0x18,%rsp
  400990:	c3                   	retq   
  400991:	90                   	nop

0000000000400992 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC1Ev>:
  400992:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400997:	90                   	nop
  400998:	c3                   	retq   
  400999:	90                   	nop

000000000040099a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC1Ev>:
  40099a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40099f:	90                   	nop
  4009a0:	c3                   	retq   

00000000004009a1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009a1:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a6:	90                   	nop
  4009a7:	c3                   	retq   

00000000004009a8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>:
  4009a8:	48 83 ec 18          	sub    $0x18,%rsp
  4009ac:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009b1:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4009b5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009ba:	48 89 c7             	mov    %rax,%rdi
  4009bd:	e8 7a 00 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009c2:	48 89 c2             	mov    %rax,%rdx
  4009c5:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4009c9:	89 02                	mov    %eax,(%rdx)
  4009cb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009d0:	48 83 c4 18          	add    $0x18,%rsp
  4009d4:	c3                   	retq   
  4009d5:	90                   	nop

00000000004009d6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>:
  4009d6:	48 83 ec 18          	sub    $0x18,%rsp
  4009da:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009df:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4009e3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009e8:	48 89 c7             	mov    %rax,%rdi
  4009eb:	e8 74 00 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009f0:	48 89 c2             	mov    %rax,%rdx
  4009f3:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4009f7:	89 02                	mov    %eax,(%rdx)
  4009f9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009fe:	48 83 c4 18          	add    $0x18,%rsp
  400a02:	c3                   	retq   
  400a03:	90                   	nop

0000000000400a04 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>:
  400a04:	48 83 ec 18          	sub    $0x18,%rsp
  400a08:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a0d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a12:	48 89 c7             	mov    %rax,%rdi
  400a15:	e8 22 00 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a1a:	48 83 c4 18          	add    $0x18,%rsp
  400a1e:	c3                   	retq   
  400a1f:	90                   	nop

0000000000400a20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>:
  400a20:	48 83 ec 18          	sub    $0x18,%rsp
  400a24:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a29:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a2e:	48 89 c7             	mov    %rax,%rdi
  400a31:	e8 2e 00 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a36:	48 83 c4 18          	add    $0x18,%rsp
  400a3a:	c3                   	retq   
  400a3b:	90                   	nop

0000000000400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a3c:	48 83 ec 18          	sub    $0x18,%rsp
  400a40:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a45:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a4a:	48 89 c7             	mov    %rax,%rdi
  400a4d:	e8 a0 00 00 00       	callq  400af2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400a52:	48 83 c4 18          	add    $0x18,%rsp
  400a56:	c3                   	retq   
  400a57:	90                   	nop

0000000000400a58 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC1ERKS3_>:
  400a58:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a5d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400a62:	90                   	nop
  400a63:	c3                   	retq   

0000000000400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a64:	48 83 ec 18          	sub    $0x18,%rsp
  400a68:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a6d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a72:	48 89 c7             	mov    %rax,%rdi
  400a75:	e8 b6 00 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400a7a:	48 83 c4 18          	add    $0x18,%rsp
  400a7e:	c3                   	retq   
  400a7f:	90                   	nop

0000000000400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC1ERKS3_>:
  400a80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a85:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400a8a:	90                   	nop
  400a8b:	c3                   	retq   

0000000000400a8c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400a8c:	48 83 ec 18          	sub    $0x18,%rsp
  400a90:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a95:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a9a:	48 89 c7             	mov    %rax,%rdi
  400a9d:	e8 9e fc ff ff       	callq  400740 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400aa2:	90                   	nop
  400aa3:	48 83 c4 18          	add    $0x18,%rsp
  400aa7:	c3                   	retq   

0000000000400aa8 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400aa8:	48 83 ec 18          	sub    $0x18,%rsp
  400aac:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ab1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ab6:	48 89 c7             	mov    %rax,%rdi
  400ab9:	e8 ce ff ff ff       	callq  400a8c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400abe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ac3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400aca:	00 
  400acb:	90                   	nop
  400acc:	48 83 c4 18          	add    $0x18,%rsp
  400ad0:	c3                   	retq   

0000000000400ad1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ad1:	90                   	nop
  400ad2:	c3                   	retq   

0000000000400ad3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ad3:	48 83 ec 18          	sub    $0x18,%rsp
  400ad7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400adc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ae1:	48 83 c0 01          	add    $0x1,%rax
  400ae5:	48 89 c7             	mov    %rax,%rdi
  400ae8:	e8 82 00 00 00       	callq  400b6f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400aed:	48 83 c4 18          	add    $0x18,%rsp
  400af1:	c3                   	retq   

0000000000400af2 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400af2:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400af7:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400afe:	00 00 
  400b00:	48 c7 44 24 e8 60 20 	movq   $0x602060,-0x18(%rsp)
  400b07:	60 00 
  400b09:	48 c7 44 24 f0 70 20 	movq   $0x602070,-0x10(%rsp)
  400b10:	60 00 
  400b12:	48 c7 44 24 f8 70 20 	movq   $0x602070,-0x8(%rsp)
  400b19:	60 00 
  400b1b:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400b20:	48 83 c0 04          	add    $0x4,%rax
  400b24:	48 c1 e0 02          	shl    $0x2,%rax
  400b28:	48 05 60 20 60 00    	add    $0x602060,%rax
  400b2e:	c3                   	retq   
  400b2f:	90                   	nop

0000000000400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400b30:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400b35:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400b3c:	00 00 
  400b3e:	48 c7 44 24 e8 60 20 	movq   $0x602060,-0x18(%rsp)
  400b45:	60 00 
  400b47:	48 c7 44 24 f0 40 69 	movq   $0x606940,-0x10(%rsp)
  400b4e:	60 00 
  400b50:	48 c7 44 24 f8 40 69 	movq   $0x606940,-0x8(%rsp)
  400b57:	60 00 
  400b59:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400b5e:	48 05 38 12 00 00    	add    $0x1238,%rax
  400b64:	48 c1 e0 02          	shl    $0x2,%rax
  400b68:	48 05 60 20 60 00    	add    $0x602060,%rax
  400b6e:	c3                   	retq   

0000000000400b6f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b6f:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b74:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400b79:	c3                   	retq   
  400b7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b80 <__libc_csu_init>:
  400b80:	41 57                	push   %r15
  400b82:	41 56                	push   %r14
  400b84:	41 89 ff             	mov    %edi,%r15d
  400b87:	41 55                	push   %r13
  400b89:	41 54                	push   %r12
  400b8b:	4c 8d 25 7e 12 20 00 	lea    0x20127e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400b92:	55                   	push   %rbp
  400b93:	48 8d 2d 7e 12 20 00 	lea    0x20127e(%rip),%rbp        # 601e18 <__init_array_end>
  400b9a:	53                   	push   %rbx
  400b9b:	49 89 f6             	mov    %rsi,%r14
  400b9e:	49 89 d5             	mov    %rdx,%r13
  400ba1:	4c 29 e5             	sub    %r12,%rbp
  400ba4:	48 83 ec 08          	sub    $0x8,%rsp
  400ba8:	48 c1 fd 03          	sar    $0x3,%rbp
  400bac:	e8 77 f8 ff ff       	callq  400428 <_init>
  400bb1:	48 85 ed             	test   %rbp,%rbp
  400bb4:	74 20                	je     400bd6 <__libc_csu_init+0x56>
  400bb6:	31 db                	xor    %ebx,%ebx
  400bb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400bbf:	00 
  400bc0:	4c 89 ea             	mov    %r13,%rdx
  400bc3:	4c 89 f6             	mov    %r14,%rsi
  400bc6:	44 89 ff             	mov    %r15d,%edi
  400bc9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400bcd:	48 83 c3 01          	add    $0x1,%rbx
  400bd1:	48 39 eb             	cmp    %rbp,%rbx
  400bd4:	75 ea                	jne    400bc0 <__libc_csu_init+0x40>
  400bd6:	48 83 c4 08          	add    $0x8,%rsp
  400bda:	5b                   	pop    %rbx
  400bdb:	5d                   	pop    %rbp
  400bdc:	41 5c                	pop    %r12
  400bde:	41 5d                	pop    %r13
  400be0:	41 5e                	pop    %r14
  400be2:	41 5f                	pop    %r15
  400be4:	c3                   	retq   
  400be5:	90                   	nop
  400be6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bed:	00 00 00 

0000000000400bf0 <__libc_csu_fini>:
  400bf0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400bf4 <_fini>:
  400bf4:	48 83 ec 08          	sub    $0x8,%rsp
  400bf8:	48 83 c4 08          	add    $0x8,%rsp
  400bfc:	c3                   	retq   
