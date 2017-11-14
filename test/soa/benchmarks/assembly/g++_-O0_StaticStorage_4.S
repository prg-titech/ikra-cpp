
bin/g++_-O0_StaticStorage_4:     file format elf64-x86-64


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
  4004af:	49 c7 c0 c0 0c 40 00 	mov    $0x400cc0,%r8
  4004b6:	48 c7 c1 50 0c 40 00 	mov    $0x400c50,%rcx
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
  4005a1:	e8 f7 03 00 00       	callq  40099d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005a6:	bf 04 00 00 00       	mov    $0x4,%edi
  4005ab:	e8 16 04 00 00       	callq  4009c6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005b0:	48 89 c3             	mov    %rax,%rbx
  4005b3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005b8:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005bc:	48 83 f8 03          	cmp    $0x3,%rax
  4005c0:	74 06                	je     4005c8 <main+0x2c>
  4005c2:	48 83 c0 01          	add    $0x1,%rax
  4005c6:	eb f0                	jmp    4005b8 <main+0x1c>
  4005c8:	48 89 df             	mov    %rbx,%rdi
  4005cb:	e8 00 03 00 00       	callq  4008d0 <_ZN9TestClassC1Ev>
  4005d0:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4005d5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005da:	48 83 c0 04          	add    $0x4,%rax
  4005de:	be 77 77 00 00       	mov    $0x7777,%esi
  4005e3:	48 89 c7             	mov    %rax,%rdi
  4005e6:	e8 1f 04 00 00       	callq  400a0a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4005eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005f0:	48 83 c0 04          	add    $0x4,%rax
  4005f4:	be 88 88 00 00       	mov    $0x8888,%esi
  4005f9:	48 89 c7             	mov    %rax,%rdi
  4005fc:	e8 37 04 00 00       	callq  400a38 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400601:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400606:	48 89 c7             	mov    %rax,%rdi
  400609:	e8 ce 01 00 00       	callq  4007dc <_ZN9TestClass15increase_field0Ev>
  40060e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400613:	48 89 c7             	mov    %rax,%rdi
  400616:	e8 29 02 00 00       	callq  400844 <_ZN9TestClass15increase_field1Ev>
  40061b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400620:	48 83 c0 04          	add    $0x4,%rax
  400624:	48 89 c7             	mov    %rax,%rdi
  400627:	e8 56 04 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40062c:	8b 18                	mov    (%rax),%ebx
  40062e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400633:	48 83 c0 04          	add    $0x4,%rax
  400637:	48 89 c7             	mov    %rax,%rdi
  40063a:	e8 27 04 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40063f:	8b 00                	mov    (%rax),%eax
  400641:	89 da                	mov    %ebx,%edx
  400643:	89 c6                	mov    %eax,%esi
  400645:	bf e4 0c 40 00       	mov    $0x400ce4,%edi
  40064a:	b8 00 00 00 00       	mov    $0x0,%eax
  40064f:	e8 1c fe ff ff       	callq  400470 <printf@plt>
  400654:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400659:	48 83 c0 04          	add    $0x4,%rax
  40065d:	48 89 c7             	mov    %rax,%rdi
  400660:	e8 01 04 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400665:	8b 00                	mov    (%rax),%eax
  400667:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40066c:	75 1a                	jne    400688 <main+0xec>
  40066e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400673:	48 83 c0 04          	add    $0x4,%rax
  400677:	48 89 c7             	mov    %rax,%rdi
  40067a:	e8 03 04 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40067f:	8b 00                	mov    (%rax),%eax
  400681:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400686:	74 07                	je     40068f <main+0xf3>
  400688:	b8 01 00 00 00       	mov    $0x1,%eax
  40068d:	eb 05                	jmp    400694 <main+0xf8>
  40068f:	b8 00 00 00 00       	mov    $0x0,%eax
  400694:	0f b6 c0             	movzbl %al,%eax
  400697:	48 83 c4 10          	add    $0x10,%rsp
  40069b:	5b                   	pop    %rbx
  40069c:	c3                   	retq   

000000000040069d <_Z12new_instancev>:
  40069d:	53                   	push   %rbx
  40069e:	bf 04 00 00 00       	mov    $0x4,%edi
  4006a3:	e8 1e 03 00 00       	callq  4009c6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006a8:	48 89 c3             	mov    %rax,%rbx
  4006ab:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b0:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006b4:	48 83 f8 03          	cmp    $0x3,%rax
  4006b8:	74 06                	je     4006c0 <_Z12new_instancev+0x23>
  4006ba:	48 83 c0 01          	add    $0x1,%rax
  4006be:	eb f0                	jmp    4006b0 <_Z12new_instancev+0x13>
  4006c0:	48 89 df             	mov    %rbx,%rdi
  4006c3:	e8 08 02 00 00       	callq  4008d0 <_ZN9TestClassC1Ev>
  4006c8:	48 89 d8             	mov    %rbx,%rax
  4006cb:	5b                   	pop    %rbx
  4006cc:	c3                   	retq   

00000000004006cd <_Z12write_field0P9TestClass>:
  4006cd:	48 83 ec 18          	sub    $0x18,%rsp
  4006d1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006d6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006db:	48 83 c0 04          	add    $0x4,%rax
  4006df:	be 77 77 00 00       	mov    $0x7777,%esi
  4006e4:	48 89 c7             	mov    %rax,%rdi
  4006e7:	e8 1e 03 00 00       	callq  400a0a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4006ec:	90                   	nop
  4006ed:	48 83 c4 18          	add    $0x18,%rsp
  4006f1:	c3                   	retq   

00000000004006f2 <_Z11read_field0P9TestClass>:
  4006f2:	48 83 ec 18          	sub    $0x18,%rsp
  4006f6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006fb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400700:	48 83 c0 04          	add    $0x4,%rax
  400704:	48 89 c7             	mov    %rax,%rdi
  400707:	e8 5a 03 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40070c:	8b 00                	mov    (%rax),%eax
  40070e:	48 83 c4 18          	add    $0x18,%rsp
  400712:	c3                   	retq   

0000000000400713 <_Z12write_field1P9TestClass>:
  400713:	48 83 ec 18          	sub    $0x18,%rsp
  400717:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40071c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400721:	48 83 c0 04          	add    $0x4,%rax
  400725:	be 77 77 00 00       	mov    $0x7777,%esi
  40072a:	48 89 c7             	mov    %rax,%rdi
  40072d:	e8 06 03 00 00       	callq  400a38 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400732:	90                   	nop
  400733:	48 83 c4 18          	add    $0x18,%rsp
  400737:	c3                   	retq   

0000000000400738 <_Z11read_field1P9TestClass>:
  400738:	48 83 ec 18          	sub    $0x18,%rsp
  40073c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400741:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400746:	48 83 c0 04          	add    $0x4,%rax
  40074a:	48 89 c7             	mov    %rax,%rdi
  40074d:	e8 30 03 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400752:	8b 00                	mov    (%rax),%eax
  400754:	48 83 c4 18          	add    $0x18,%rsp
  400758:	c3                   	retq   

0000000000400759 <_Z21explicit_write_field0m>:
  400759:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40075e:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400763:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  40076a:	77 77 00 00 
  40076e:	90                   	nop
  40076f:	c3                   	retq   

0000000000400770 <_Z20explicit_read_field0m>:
  400770:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400775:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40077a:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  400781:	c3                   	retq   

0000000000400782 <_ZnwmPv>:
  400782:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400787:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  40078c:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400791:	c3                   	retq   

0000000000400792 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  400792:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400797:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40079c:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007a3:	90                   	nop
  4007a4:	c3                   	retq   
  4007a5:	90                   	nop

00000000004007a6 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007a6:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4007ab:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  4007b0:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007b5:	48 8b 00             	mov    (%rax),%rax
  4007b8:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007bd:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007c2:	48 8b 10             	mov    (%rax),%rdx
  4007c5:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  4007ca:	48 01 c2             	add    %rax,%rdx
  4007cd:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007d2:	48 89 10             	mov    %rdx,(%rax)
  4007d5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007da:	c3                   	retq   
  4007db:	90                   	nop

00000000004007dc <_ZN9TestClass15increase_field0Ev>:
  4007dc:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007e1:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007e5:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007e9:	55                   	push   %rbp
  4007ea:	48 89 e5             	mov    %rsp,%rbp
  4007ed:	41 52                	push   %r10
  4007ef:	48 83 ec 48          	sub    $0x48,%rsp
  4007f3:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4007f7:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007fe:	00 00 
  400800:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400804:	31 c0                	xor    %eax,%eax
  400806:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40080a:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40080e:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400812:	ba 55 55 00 00       	mov    $0x5555,%edx
  400817:	48 89 ce             	mov    %rcx,%rsi
  40081a:	48 89 c7             	mov    %rax,%rdi
  40081d:	e8 ec 00 00 00       	callq  40090e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400822:	90                   	nop
  400823:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400827:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40082e:	00 00 
  400830:	74 05                	je     400837 <_ZN9TestClass15increase_field0Ev+0x5b>
  400832:	e8 29 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400837:	48 83 c4 48          	add    $0x48,%rsp
  40083b:	41 5a                	pop    %r10
  40083d:	5d                   	pop    %rbp
  40083e:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400842:	c3                   	retq   
  400843:	90                   	nop

0000000000400844 <_ZN9TestClass15increase_field1Ev>:
  400844:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400849:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40084d:	41 ff 72 f8          	pushq  -0x8(%r10)
  400851:	55                   	push   %rbp
  400852:	48 89 e5             	mov    %rsp,%rbp
  400855:	41 52                	push   %r10
  400857:	48 83 ec 48          	sub    $0x48,%rsp
  40085b:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  40085f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400866:	00 00 
  400868:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40086c:	31 c0                	xor    %eax,%eax
  40086e:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400872:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400876:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40087a:	ba 44 44 00 00       	mov    $0x4444,%edx
  40087f:	48 89 ce             	mov    %rcx,%rsi
  400882:	48 89 c7             	mov    %rax,%rdi
  400885:	e8 cc 00 00 00       	callq  400956 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  40088a:	90                   	nop
  40088b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40088f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400896:	00 00 
  400898:	74 05                	je     40089f <_ZN9TestClass15increase_field1Ev+0x5b>
  40089a:	e8 c1 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  40089f:	48 83 c4 48          	add    $0x48,%rsp
  4008a3:	41 5a                	pop    %r10
  4008a5:	5d                   	pop    %rbp
  4008a6:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4008aa:	c3                   	retq   
  4008ab:	90                   	nop

00000000004008ac <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008ac:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008b1:	90                   	nop
  4008b2:	c3                   	retq   
  4008b3:	90                   	nop

00000000004008b4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  4008b4:	48 83 ec 18          	sub    $0x18,%rsp
  4008b8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008bd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008c2:	48 89 c7             	mov    %rax,%rdi
  4008c5:	e8 e2 ff ff ff       	callq  4008ac <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ca:	90                   	nop
  4008cb:	48 83 c4 18          	add    $0x18,%rsp
  4008cf:	c3                   	retq   

00000000004008d0 <_ZN9TestClassC1Ev>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 ce ff ff ff       	callq  4008b4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  4008e6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008eb:	48 83 c0 04          	add    $0x4,%rax
  4008ef:	48 89 c7             	mov    %rax,%rdi
  4008f2:	e8 fd 00 00 00       	callq  4009f4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  4008f7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008fc:	48 83 c0 04          	add    $0x4,%rax
  400900:	48 89 c7             	mov    %rax,%rdi
  400903:	e8 f4 00 00 00       	callq  4009fc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400908:	90                   	nop
  400909:	48 83 c4 18          	add    $0x18,%rsp
  40090d:	c3                   	retq   

000000000040090e <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40090e:	48 83 ec 28          	sub    $0x28,%rsp
  400912:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400917:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  40091c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400920:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400925:	48 89 c7             	mov    %rax,%rdi
  400928:	e8 71 01 00 00       	callq  400a9e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40092d:	8b 10                	mov    (%rax),%edx
  40092f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400934:	89 10                	mov    %edx,(%rax)
  400936:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  40093b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400940:	48 89 d6             	mov    %rdx,%rsi
  400943:	48 89 c7             	mov    %rax,%rdi
  400946:	e8 6f 01 00 00       	callq  400aba <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  40094b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400950:	48 83 c4 28          	add    $0x28,%rsp
  400954:	c3                   	retq   
  400955:	90                   	nop

0000000000400956 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400956:	48 83 ec 28          	sub    $0x28,%rsp
  40095a:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40095f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400964:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400968:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40096d:	48 89 c7             	mov    %rax,%rdi
  400970:	e8 51 01 00 00       	callq  400ac6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400975:	8b 10                	mov    (%rax),%edx
  400977:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  40097c:	89 10                	mov    %edx,(%rax)
  40097e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400983:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400988:	48 89 d6             	mov    %rdx,%rsi
  40098b:	48 89 c7             	mov    %rax,%rdi
  40098e:	e8 4f 01 00 00       	callq  400ae2 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400993:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400998:	48 83 c4 28          	add    $0x28,%rsp
  40099c:	c3                   	retq   

000000000040099d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  40099d:	48 83 ec 08          	sub    $0x8,%rsp
  4009a1:	e8 f0 fb ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  4009a6:	48 89 c6             	mov    %rax,%rsi
  4009a9:	bf b8 91 00 00       	mov    $0x91b8,%edi
  4009ae:	e8 cf fd ff ff       	callq  400782 <_ZnwmPv>
  4009b3:	48 85 c0             	test   %rax,%rax
  4009b6:	74 08                	je     4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  4009b8:	48 89 c7             	mov    %rax,%rdi
  4009bb:	e8 4a 01 00 00       	callq  400b0a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  4009c0:	90                   	nop
  4009c1:	48 83 c4 08          	add    $0x8,%rsp
  4009c5:	c3                   	retq   

00000000004009c6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009c6:	48 83 ec 18          	sub    $0x18,%rsp
  4009ca:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009cf:	e8 5f 01 00 00       	callq  400b33 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009d4:	e8 bd fb ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  4009d9:	be 01 00 00 00       	mov    $0x1,%esi
  4009de:	48 89 c7             	mov    %rax,%rdi
  4009e1:	e8 c0 fd ff ff       	callq  4007a6 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009e6:	48 89 c7             	mov    %rax,%rdi
  4009e9:	e8 47 01 00 00       	callq  400b35 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009ee:	48 83 c4 18          	add    $0x18,%rsp
  4009f2:	c3                   	retq   
  4009f3:	90                   	nop

00000000004009f4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4009f4:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009f9:	90                   	nop
  4009fa:	c3                   	retq   
  4009fb:	90                   	nop

00000000004009fc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  4009fc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a01:	90                   	nop
  400a02:	c3                   	retq   

0000000000400a03 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a03:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a08:	90                   	nop
  400a09:	c3                   	retq   

0000000000400a0a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a0a:	48 83 ec 18          	sub    $0x18,%rsp
  400a0e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a13:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a17:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a1c:	48 89 c7             	mov    %rax,%rdi
  400a1f:	e8 7a 00 00 00       	callq  400a9e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a24:	48 89 c2             	mov    %rax,%rdx
  400a27:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a2b:	89 02                	mov    %eax,(%rdx)
  400a2d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a32:	48 83 c4 18          	add    $0x18,%rsp
  400a36:	c3                   	retq   
  400a37:	90                   	nop

0000000000400a38 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a38:	48 83 ec 18          	sub    $0x18,%rsp
  400a3c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a41:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400a45:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a4a:	48 89 c7             	mov    %rax,%rdi
  400a4d:	e8 74 00 00 00       	callq  400ac6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a52:	48 89 c2             	mov    %rax,%rdx
  400a55:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a59:	89 02                	mov    %eax,(%rdx)
  400a5b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a60:	48 83 c4 18          	add    $0x18,%rsp
  400a64:	c3                   	retq   
  400a65:	90                   	nop

0000000000400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400a66:	48 83 ec 18          	sub    $0x18,%rsp
  400a6a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a6f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a74:	48 89 c7             	mov    %rax,%rdi
  400a77:	e8 22 00 00 00       	callq  400a9e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a7c:	48 83 c4 18          	add    $0x18,%rsp
  400a80:	c3                   	retq   
  400a81:	90                   	nop

0000000000400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400a82:	48 83 ec 18          	sub    $0x18,%rsp
  400a86:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a8b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a90:	48 89 c7             	mov    %rax,%rdi
  400a93:	e8 2e 00 00 00       	callq  400ac6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a98:	48 83 c4 18          	add    $0x18,%rsp
  400a9c:	c3                   	retq   
  400a9d:	90                   	nop

0000000000400a9e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a9e:	48 83 ec 18          	sub    $0x18,%rsp
  400aa2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400aa7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aac:	48 89 c7             	mov    %rax,%rdi
  400aaf:	e8 a0 00 00 00       	callq  400b54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab4:	48 83 c4 18          	add    $0x18,%rsp
  400ab8:	c3                   	retq   
  400ab9:	90                   	nop

0000000000400aba <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400aba:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400abf:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400ac4:	90                   	nop
  400ac5:	c3                   	retq   

0000000000400ac6 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ac6:	48 83 ec 18          	sub    $0x18,%rsp
  400aca:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400acf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ad4:	48 89 c7             	mov    %rax,%rdi
  400ad7:	e8 ce 00 00 00       	callq  400baa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400adc:	48 83 c4 18          	add    $0x18,%rsp
  400ae0:	c3                   	retq   
  400ae1:	90                   	nop

0000000000400ae2 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400ae2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ae7:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400aec:	90                   	nop
  400aed:	c3                   	retq   

0000000000400aee <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400aee:	48 83 ec 18          	sub    $0x18,%rsp
  400af2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400af7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400afc:	48 89 c7             	mov    %rax,%rdi
  400aff:	e8 8e fc ff ff       	callq  400792 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400b04:	90                   	nop
  400b05:	48 83 c4 18          	add    $0x18,%rsp
  400b09:	c3                   	retq   

0000000000400b0a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400b0a:	48 83 ec 18          	sub    $0x18,%rsp
  400b0e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b13:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b18:	48 89 c7             	mov    %rax,%rdi
  400b1b:	e8 ce ff ff ff       	callq  400aee <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400b20:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b25:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b2c:	00 
  400b2d:	90                   	nop
  400b2e:	48 83 c4 18          	add    $0x18,%rsp
  400b32:	c3                   	retq   

0000000000400b33 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b33:	90                   	nop
  400b34:	c3                   	retq   

0000000000400b35 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b35:	48 83 ec 18          	sub    $0x18,%rsp
  400b39:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b3e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b43:	48 83 c0 01          	add    $0x1,%rax
  400b47:	48 89 c7             	mov    %rax,%rdi
  400b4a:	e8 b7 00 00 00       	callq  400c06 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b4f:	48 83 c4 18          	add    $0x18,%rsp
  400b53:	c3                   	retq   

0000000000400b54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b54:	48 83 ec 38          	sub    $0x38,%rsp
  400b58:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b5d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b62:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b67:	e8 2a fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b6c:	48 89 c7             	mov    %rax,%rdi
  400b6f:	e8 c6 00 00 00       	callq  400c3a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400b74:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400b79:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b7e:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400b83:	48 83 e8 04          	sub    $0x4,%rax
  400b87:	48 c1 e8 02          	shr    $0x2,%rax
  400b8b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b92:	00 
  400b93:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400b98:	48 01 d0             	add    %rdx,%rax
  400b9b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400ba0:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400ba5:	48 83 c4 38          	add    $0x38,%rsp
  400ba9:	c3                   	retq   

0000000000400baa <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400baa:	48 83 ec 38          	sub    $0x38,%rsp
  400bae:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bb3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bb8:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bbd:	e8 d4 f9 ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400bc2:	48 89 c7             	mov    %rax,%rdi
  400bc5:	e8 70 00 00 00       	callq  400c3a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400bca:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400bcf:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bd4:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400bd9:	48 83 e8 04          	sub    $0x4,%rax
  400bdd:	48 c1 e8 02          	shr    $0x2,%rax
  400be1:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400be8:	00 
  400be9:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400bee:	48 01 d0             	add    %rdx,%rax
  400bf1:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bf7:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400bfc:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c01:	48 83 c4 38          	add    $0x38,%rsp
  400c05:	c3                   	retq   

0000000000400c06 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c06:	48 83 ec 28          	sub    $0x28,%rsp
  400c0a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c0f:	e8 82 f9 ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400c14:	48 89 c7             	mov    %rax,%rdi
  400c17:	e8 1e 00 00 00       	callq  400c3a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400c1c:	48 89 c2             	mov    %rax,%rdx
  400c1f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c24:	48 c1 e0 02          	shl    $0x2,%rax
  400c28:	48 01 d0             	add    %rdx,%rax
  400c2b:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c30:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c35:	48 83 c4 28          	add    $0x28,%rsp
  400c39:	c3                   	retq   

0000000000400c3a <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c3a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c3f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400c44:	48 83 c0 10          	add    $0x10,%rax
  400c48:	c3                   	retq   
  400c49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c50 <__libc_csu_init>:
  400c50:	41 57                	push   %r15
  400c52:	41 56                	push   %r14
  400c54:	41 89 ff             	mov    %edi,%r15d
  400c57:	41 55                	push   %r13
  400c59:	41 54                	push   %r12
  400c5b:	4c 8d 25 ae 11 20 00 	lea    0x2011ae(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c62:	55                   	push   %rbp
  400c63:	48 8d 2d ae 11 20 00 	lea    0x2011ae(%rip),%rbp        # 601e18 <__init_array_end>
  400c6a:	53                   	push   %rbx
  400c6b:	49 89 f6             	mov    %rsi,%r14
  400c6e:	49 89 d5             	mov    %rdx,%r13
  400c71:	4c 29 e5             	sub    %r12,%rbp
  400c74:	48 83 ec 08          	sub    $0x8,%rsp
  400c78:	48 c1 fd 03          	sar    $0x3,%rbp
  400c7c:	e8 a7 f7 ff ff       	callq  400428 <_init>
  400c81:	48 85 ed             	test   %rbp,%rbp
  400c84:	74 20                	je     400ca6 <__libc_csu_init+0x56>
  400c86:	31 db                	xor    %ebx,%ebx
  400c88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c8f:	00 
  400c90:	4c 89 ea             	mov    %r13,%rdx
  400c93:	4c 89 f6             	mov    %r14,%rsi
  400c96:	44 89 ff             	mov    %r15d,%edi
  400c99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c9d:	48 83 c3 01          	add    $0x1,%rbx
  400ca1:	48 39 eb             	cmp    %rbp,%rbx
  400ca4:	75 ea                	jne    400c90 <__libc_csu_init+0x40>
  400ca6:	48 83 c4 08          	add    $0x8,%rsp
  400caa:	5b                   	pop    %rbx
  400cab:	5d                   	pop    %rbp
  400cac:	41 5c                	pop    %r12
  400cae:	41 5d                	pop    %r13
  400cb0:	41 5e                	pop    %r14
  400cb2:	41 5f                	pop    %r15
  400cb4:	c3                   	retq   
  400cb5:	90                   	nop
  400cb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cbd:	00 00 00 

0000000000400cc0 <__libc_csu_fini>:
  400cc0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400cc4 <_fini>:
  400cc4:	48 83 ec 08          	sub    $0x8,%rsp
  400cc8:	48 83 c4 08          	add    $0x8,%rsp
  400ccc:	c3                   	retq   
