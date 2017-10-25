
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
  4004af:	49 c7 c0 30 0b 40 00 	mov    $0x400b30,%r8
  4004b6:	48 c7 c1 c0 0a 40 00 	mov    $0x400ac0,%rcx
  4004bd:	48 c7 c7 96 05 40 00 	mov    $0x400596,%rdi
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

0000000000400596 <main>:
  400596:	55                   	push   %rbp
  400597:	48 89 e5             	mov    %rsp,%rbp
  40059a:	53                   	push   %rbx
  40059b:	48 83 ec 28          	sub    $0x28,%rsp
  40059f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005a6:	00 00 
  4005a8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005ac:	31 c0                	xor    %eax,%eax
  4005ae:	bf 00 00 00 00       	mov    $0x0,%edi
  4005b3:	e8 80 03 00 00       	callq  400938 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005b8:	48 89 c3             	mov    %rax,%rbx
  4005bb:	48 89 df             	mov    %rbx,%rdi
  4005be:	e8 c1 02 00 00       	callq  400884 <_ZN9TestClassC1Ev>
  4005c3:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005c7:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005cb:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005cf:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005d4:	48 89 ce             	mov    %rcx,%rsi
  4005d7:	48 89 c7             	mov    %rax,%rdi
  4005da:	e8 a9 03 00 00       	callq  400988 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005df:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005e3:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005e7:	ba 88 88 00 00       	mov    $0x8888,%edx
  4005ec:	48 89 ce             	mov    %rcx,%rsi
  4005ef:	48 89 c7             	mov    %rax,%rdi
  4005f2:	e8 d1 03 00 00       	callq  4009c8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4005f7:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005fb:	48 89 c7             	mov    %rax,%rdi
  4005fe:	e8 c5 01 00 00       	callq  4007c8 <_ZN9TestClass15increase_field0Ev>
  400603:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400607:	48 89 c7             	mov    %rax,%rdi
  40060a:	e8 03 02 00 00       	callq  400812 <_ZN9TestClass15increase_field1Ev>
  40060f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400613:	48 89 c7             	mov    %rax,%rdi
  400616:	e8 07 04 00 00       	callq  400a22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40061b:	8b 18                	mov    (%rax),%ebx
  40061d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400621:	48 89 c7             	mov    %rax,%rdi
  400624:	e8 df 03 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400629:	8b 00                	mov    (%rax),%eax
  40062b:	89 da                	mov    %ebx,%edx
  40062d:	89 c6                	mov    %eax,%esi
  40062f:	bf 54 0b 40 00       	mov    $0x400b54,%edi
  400634:	b8 00 00 00 00       	mov    $0x0,%eax
  400639:	e8 32 fe ff ff       	callq  400470 <printf@plt>
  40063e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400642:	48 89 c7             	mov    %rax,%rdi
  400645:	e8 be 03 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40064a:	8b 00                	mov    (%rax),%eax
  40064c:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400651:	75 15                	jne    400668 <main+0xd2>
  400653:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400657:	48 89 c7             	mov    %rax,%rdi
  40065a:	e8 c3 03 00 00       	callq  400a22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40065f:	8b 00                	mov    (%rax),%eax
  400661:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400666:	74 07                	je     40066f <main+0xd9>
  400668:	b8 01 00 00 00       	mov    $0x1,%eax
  40066d:	eb 05                	jmp    400674 <main+0xde>
  40066f:	b8 00 00 00 00       	mov    $0x0,%eax
  400674:	0f b6 c0             	movzbl %al,%eax
  400677:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40067b:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  400682:	00 00 
  400684:	74 05                	je     40068b <main+0xf5>
  400686:	e8 d5 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  40068b:	48 83 c4 28          	add    $0x28,%rsp
  40068f:	5b                   	pop    %rbx
  400690:	5d                   	pop    %rbp
  400691:	c3                   	retq   

0000000000400692 <_Z12new_instancev>:
  400692:	55                   	push   %rbp
  400693:	48 89 e5             	mov    %rsp,%rbp
  400696:	53                   	push   %rbx
  400697:	48 83 ec 08          	sub    $0x8,%rsp
  40069b:	bf 00 00 00 00       	mov    $0x0,%edi
  4006a0:	e8 93 02 00 00       	callq  400938 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4006a5:	48 89 c3             	mov    %rax,%rbx
  4006a8:	48 89 df             	mov    %rbx,%rdi
  4006ab:	e8 d4 01 00 00       	callq  400884 <_ZN9TestClassC1Ev>
  4006b0:	48 89 d8             	mov    %rbx,%rax
  4006b3:	48 83 c4 08          	add    $0x8,%rsp
  4006b7:	5b                   	pop    %rbx
  4006b8:	5d                   	pop    %rbp
  4006b9:	c3                   	retq   

00000000004006ba <_Z12write_field0P9TestClass>:
  4006ba:	55                   	push   %rbp
  4006bb:	48 89 e5             	mov    %rsp,%rbp
  4006be:	48 83 ec 30          	sub    $0x30,%rsp
  4006c2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4006c6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4006cd:	00 00 
  4006cf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4006d3:	31 c0                	xor    %eax,%eax
  4006d5:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4006d9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4006dd:	ba 77 77 00 00       	mov    $0x7777,%edx
  4006e2:	48 89 ce             	mov    %rcx,%rsi
  4006e5:	48 89 c7             	mov    %rax,%rdi
  4006e8:	e8 9b 02 00 00       	callq  400988 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006ed:	90                   	nop
  4006ee:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006f2:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4006f9:	00 00 
  4006fb:	74 05                	je     400702 <_Z12write_field0P9TestClass+0x48>
  4006fd:	e8 5e fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400702:	c9                   	leaveq 
  400703:	c3                   	retq   

0000000000400704 <_Z11read_field0P9TestClass>:
  400704:	55                   	push   %rbp
  400705:	48 89 e5             	mov    %rsp,%rbp
  400708:	48 83 ec 10          	sub    $0x10,%rsp
  40070c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400710:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400714:	48 89 c7             	mov    %rax,%rdi
  400717:	e8 ec 02 00 00       	callq  400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40071c:	8b 00                	mov    (%rax),%eax
  40071e:	c9                   	leaveq 
  40071f:	c3                   	retq   

0000000000400720 <_Z12write_field1P9TestClass>:
  400720:	55                   	push   %rbp
  400721:	48 89 e5             	mov    %rsp,%rbp
  400724:	48 83 ec 30          	sub    $0x30,%rsp
  400728:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40072c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400733:	00 00 
  400735:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400739:	31 c0                	xor    %eax,%eax
  40073b:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40073f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400743:	ba 77 77 00 00       	mov    $0x7777,%edx
  400748:	48 89 ce             	mov    %rcx,%rsi
  40074b:	48 89 c7             	mov    %rax,%rdi
  40074e:	e8 75 02 00 00       	callq  4009c8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400753:	90                   	nop
  400754:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400758:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40075f:	00 00 
  400761:	74 05                	je     400768 <_Z12write_field1P9TestClass+0x48>
  400763:	e8 f8 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400768:	c9                   	leaveq 
  400769:	c3                   	retq   

000000000040076a <_Z11read_field1P9TestClass>:
  40076a:	55                   	push   %rbp
  40076b:	48 89 e5             	mov    %rsp,%rbp
  40076e:	48 83 ec 10          	sub    $0x10,%rsp
  400772:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400776:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40077a:	48 89 c7             	mov    %rax,%rdi
  40077d:	e8 a0 02 00 00       	callq  400a22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400782:	8b 00                	mov    (%rax),%eax
  400784:	c9                   	leaveq 
  400785:	c3                   	retq   

0000000000400786 <_Z21explicit_write_field0m>:
  400786:	55                   	push   %rbp
  400787:	48 89 e5             	mov    %rsp,%rbp
  40078a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40078e:	48 8b 05 73 aa 20 00 	mov    0x20aa73(%rip),%rax        # 60b208 <explicit_field0>
  400795:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400799:	48 c1 e2 02          	shl    $0x2,%rdx
  40079d:	48 01 d0             	add    %rdx,%rax
  4007a0:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007a6:	90                   	nop
  4007a7:	5d                   	pop    %rbp
  4007a8:	c3                   	retq   

00000000004007a9 <_Z20explicit_read_field0m>:
  4007a9:	55                   	push   %rbp
  4007aa:	48 89 e5             	mov    %rsp,%rbp
  4007ad:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b1:	48 8b 05 50 aa 20 00 	mov    0x20aa50(%rip),%rax        # 60b208 <explicit_field0>
  4007b8:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007bc:	48 c1 e2 02          	shl    $0x2,%rdx
  4007c0:	48 01 d0             	add    %rdx,%rax
  4007c3:	8b 00                	mov    (%rax),%eax
  4007c5:	5d                   	pop    %rbp
  4007c6:	c3                   	retq   
  4007c7:	90                   	nop

00000000004007c8 <_ZN9TestClass15increase_field0Ev>:
  4007c8:	55                   	push   %rbp
  4007c9:	48 89 e5             	mov    %rsp,%rbp
  4007cc:	48 83 ec 30          	sub    $0x30,%rsp
  4007d0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007d4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007db:	00 00 
  4007dd:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007e1:	31 c0                	xor    %eax,%eax
  4007e3:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4007e7:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007eb:	ba 55 55 00 00       	mov    $0x5555,%edx
  4007f0:	48 89 ce             	mov    %rcx,%rsi
  4007f3:	48 89 c7             	mov    %rax,%rdi
  4007f6:	e8 bd 00 00 00       	callq  4008b8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  4007fb:	90                   	nop
  4007fc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400800:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400807:	00 00 
  400809:	74 05                	je     400810 <_ZN9TestClass15increase_field0Ev+0x48>
  40080b:	e8 50 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400810:	c9                   	leaveq 
  400811:	c3                   	retq   

0000000000400812 <_ZN9TestClass15increase_field1Ev>:
  400812:	55                   	push   %rbp
  400813:	48 89 e5             	mov    %rsp,%rbp
  400816:	48 83 ec 30          	sub    $0x30,%rsp
  40081a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40081e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400825:	00 00 
  400827:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40082b:	31 c0                	xor    %eax,%eax
  40082d:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400831:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400835:	ba 44 44 00 00       	mov    $0x4444,%edx
  40083a:	48 89 ce             	mov    %rcx,%rsi
  40083d:	48 89 c7             	mov    %rax,%rdi
  400840:	e8 b3 00 00 00       	callq  4008f8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400845:	90                   	nop
  400846:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40084a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400851:	00 00 
  400853:	74 05                	je     40085a <_ZN9TestClass15increase_field1Ev+0x48>
  400855:	e8 06 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  40085a:	c9                   	leaveq 
  40085b:	c3                   	retq   

000000000040085c <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  40085c:	55                   	push   %rbp
  40085d:	48 89 e5             	mov    %rsp,%rbp
  400860:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400864:	90                   	nop
  400865:	5d                   	pop    %rbp
  400866:	c3                   	retq   
  400867:	90                   	nop

0000000000400868 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  400868:	55                   	push   %rbp
  400869:	48 89 e5             	mov    %rsp,%rbp
  40086c:	48 83 ec 10          	sub    $0x10,%rsp
  400870:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400874:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400878:	48 89 c7             	mov    %rax,%rdi
  40087b:	e8 dc ff ff ff       	callq  40085c <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400880:	90                   	nop
  400881:	c9                   	leaveq 
  400882:	c3                   	retq   
  400883:	90                   	nop

0000000000400884 <_ZN9TestClassC1Ev>:
  400884:	55                   	push   %rbp
  400885:	48 89 e5             	mov    %rsp,%rbp
  400888:	48 83 ec 10          	sub    $0x10,%rsp
  40088c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400890:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400894:	48 89 c7             	mov    %rax,%rdi
  400897:	e8 cc ff ff ff       	callq  400868 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  40089c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a0:	48 89 c7             	mov    %rax,%rdi
  4008a3:	e8 be 00 00 00       	callq  400966 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008a8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ac:	48 89 c7             	mov    %rax,%rdi
  4008af:	e8 be 00 00 00       	callq  400972 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008b4:	90                   	nop
  4008b5:	c9                   	leaveq 
  4008b6:	c3                   	retq   
  4008b7:	90                   	nop

00000000004008b8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4008b8:	55                   	push   %rbp
  4008b9:	48 89 e5             	mov    %rsp,%rbp
  4008bc:	48 83 ec 20          	sub    $0x20,%rsp
  4008c0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008c8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008cb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008cf:	48 89 c7             	mov    %rax,%rdi
  4008d2:	e8 65 01 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008d7:	8b 10                	mov    (%rax),%edx
  4008d9:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008dd:	89 10                	mov    %edx,(%rax)
  4008df:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008e3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e7:	48 89 d6             	mov    %rdx,%rsi
  4008ea:	48 89 c7             	mov    %rax,%rdi
  4008ed:	e8 62 01 00 00       	callq  400a54 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4008f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f6:	c9                   	leaveq 
  4008f7:	c3                   	retq   

00000000004008f8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  4008f8:	55                   	push   %rbp
  4008f9:	48 89 e5             	mov    %rsp,%rbp
  4008fc:	48 83 ec 20          	sub    $0x20,%rsp
  400900:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400904:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400908:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40090b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40090f:	48 89 c7             	mov    %rax,%rdi
  400912:	e8 4d 01 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400917:	8b 10                	mov    (%rax),%edx
  400919:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40091d:	89 10                	mov    %edx,(%rax)
  40091f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400923:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400927:	48 89 d6             	mov    %rdx,%rsi
  40092a:	48 89 c7             	mov    %rax,%rdi
  40092d:	e8 50 01 00 00       	callq  400a82 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400932:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400936:	c9                   	leaveq 
  400937:	c3                   	retq   

0000000000400938 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400938:	55                   	push   %rbp
  400939:	48 89 e5             	mov    %rsp,%rbp
  40093c:	48 83 ec 10          	sub    $0x10,%rsp
  400940:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400944:	e8 48 01 00 00       	callq  400a91 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400949:	48 8b 05 10 17 20 00 	mov    0x201710(%rip),%rax        # 602060 <_ZN9TestClass7storageE>
  400950:	48 8d 50 01          	lea    0x1(%rax),%rdx
  400954:	48 89 15 05 17 20 00 	mov    %rdx,0x201705(%rip)        # 602060 <_ZN9TestClass7storageE>
  40095b:	48 89 c7             	mov    %rax,%rdi
  40095e:	e8 35 01 00 00       	callq  400a98 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400963:	c9                   	leaveq 
  400964:	c3                   	retq   
  400965:	90                   	nop

0000000000400966 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  400966:	55                   	push   %rbp
  400967:	48 89 e5             	mov    %rsp,%rbp
  40096a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40096e:	90                   	nop
  40096f:	5d                   	pop    %rbp
  400970:	c3                   	retq   
  400971:	90                   	nop

0000000000400972 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  400972:	55                   	push   %rbp
  400973:	48 89 e5             	mov    %rsp,%rbp
  400976:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097a:	90                   	nop
  40097b:	5d                   	pop    %rbp
  40097c:	c3                   	retq   

000000000040097d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  40097d:	55                   	push   %rbp
  40097e:	48 89 e5             	mov    %rsp,%rbp
  400981:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400985:	90                   	nop
  400986:	5d                   	pop    %rbp
  400987:	c3                   	retq   

0000000000400988 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400988:	55                   	push   %rbp
  400989:	48 89 e5             	mov    %rsp,%rbp
  40098c:	48 83 ec 20          	sub    $0x20,%rsp
  400990:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400994:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400998:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40099b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40099f:	48 89 c7             	mov    %rax,%rdi
  4009a2:	e8 95 00 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009a7:	48 89 c2             	mov    %rax,%rdx
  4009aa:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4009ad:	89 02                	mov    %eax,(%rdx)
  4009af:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009b3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009b7:	48 89 d6             	mov    %rdx,%rsi
  4009ba:	48 89 c7             	mov    %rax,%rdi
  4009bd:	e8 92 00 00 00       	callq  400a54 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4009c2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009c6:	c9                   	leaveq 
  4009c7:	c3                   	retq   

00000000004009c8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  4009c8:	55                   	push   %rbp
  4009c9:	48 89 e5             	mov    %rsp,%rbp
  4009cc:	48 83 ec 20          	sub    $0x20,%rsp
  4009d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009d8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009db:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009df:	48 89 c7             	mov    %rax,%rdi
  4009e2:	e8 7d 00 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009e7:	48 89 c2             	mov    %rax,%rdx
  4009ea:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4009ed:	89 02                	mov    %eax,(%rdx)
  4009ef:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009f3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009f7:	48 89 d6             	mov    %rdx,%rsi
  4009fa:	48 89 c7             	mov    %rax,%rdi
  4009fd:	e8 80 00 00 00       	callq  400a82 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a02:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a06:	c9                   	leaveq 
  400a07:	c3                   	retq   

0000000000400a08 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a08:	55                   	push   %rbp
  400a09:	48 89 e5             	mov    %rsp,%rbp
  400a0c:	48 83 ec 10          	sub    $0x10,%rsp
  400a10:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a18:	48 89 c7             	mov    %rax,%rdi
  400a1b:	e8 1c 00 00 00       	callq  400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a20:	c9                   	leaveq 
  400a21:	c3                   	retq   

0000000000400a22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a22:	55                   	push   %rbp
  400a23:	48 89 e5             	mov    %rsp,%rbp
  400a26:	48 83 ec 10          	sub    $0x10,%rsp
  400a2a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a32:	48 89 c7             	mov    %rax,%rdi
  400a35:	e8 2a 00 00 00       	callq  400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3a:	c9                   	leaveq 
  400a3b:	c3                   	retq   

0000000000400a3c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a3c:	55                   	push   %rbp
  400a3d:	48 89 e5             	mov    %rsp,%rbp
  400a40:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a44:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a48:	48 c1 e0 02          	shl    $0x2,%rax
  400a4c:	48 05 68 20 60 00    	add    $0x602068,%rax
  400a52:	5d                   	pop    %rbp
  400a53:	c3                   	retq   

0000000000400a54 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a54:	55                   	push   %rbp
  400a55:	48 89 e5             	mov    %rsp,%rbp
  400a58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a5c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a60:	90                   	nop
  400a61:	5d                   	pop    %rbp
  400a62:	c3                   	retq   
  400a63:	90                   	nop

0000000000400a64 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a64:	55                   	push   %rbp
  400a65:	48 89 e5             	mov    %rsp,%rbp
  400a68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a70:	48 c1 e0 02          	shl    $0x2,%rax
  400a74:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400a7a:	48 05 68 20 60 00    	add    $0x602068,%rax
  400a80:	5d                   	pop    %rbp
  400a81:	c3                   	retq   

0000000000400a82 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400a82:	55                   	push   %rbp
  400a83:	48 89 e5             	mov    %rsp,%rbp
  400a86:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a8e:	90                   	nop
  400a8f:	5d                   	pop    %rbp
  400a90:	c3                   	retq   

0000000000400a91 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400a91:	55                   	push   %rbp
  400a92:	48 89 e5             	mov    %rsp,%rbp
  400a95:	90                   	nop
  400a96:	5d                   	pop    %rbp
  400a97:	c3                   	retq   

0000000000400a98 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400a98:	55                   	push   %rbp
  400a99:	48 89 e5             	mov    %rsp,%rbp
  400a9c:	48 83 ec 10          	sub    $0x10,%rsp
  400aa0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 02 00 00 00       	callq  400ab2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400ab0:	c9                   	leaveq 
  400ab1:	c3                   	retq   

0000000000400ab2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400ab2:	55                   	push   %rbp
  400ab3:	48 89 e5             	mov    %rsp,%rbp
  400ab6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aba:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400abe:	5d                   	pop    %rbp
  400abf:	c3                   	retq   

0000000000400ac0 <__libc_csu_init>:
  400ac0:	41 57                	push   %r15
  400ac2:	41 56                	push   %r14
  400ac4:	41 89 ff             	mov    %edi,%r15d
  400ac7:	41 55                	push   %r13
  400ac9:	41 54                	push   %r12
  400acb:	4c 8d 25 3e 13 20 00 	lea    0x20133e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400ad2:	55                   	push   %rbp
  400ad3:	48 8d 2d 3e 13 20 00 	lea    0x20133e(%rip),%rbp        # 601e18 <__init_array_end>
  400ada:	53                   	push   %rbx
  400adb:	49 89 f6             	mov    %rsi,%r14
  400ade:	49 89 d5             	mov    %rdx,%r13
  400ae1:	4c 29 e5             	sub    %r12,%rbp
  400ae4:	48 83 ec 08          	sub    $0x8,%rsp
  400ae8:	48 c1 fd 03          	sar    $0x3,%rbp
  400aec:	e8 37 f9 ff ff       	callq  400428 <_init>
  400af1:	48 85 ed             	test   %rbp,%rbp
  400af4:	74 20                	je     400b16 <__libc_csu_init+0x56>
  400af6:	31 db                	xor    %ebx,%ebx
  400af8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400aff:	00 
  400b00:	4c 89 ea             	mov    %r13,%rdx
  400b03:	4c 89 f6             	mov    %r14,%rsi
  400b06:	44 89 ff             	mov    %r15d,%edi
  400b09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400b0d:	48 83 c3 01          	add    $0x1,%rbx
  400b11:	48 39 eb             	cmp    %rbp,%rbx
  400b14:	75 ea                	jne    400b00 <__libc_csu_init+0x40>
  400b16:	48 83 c4 08          	add    $0x8,%rsp
  400b1a:	5b                   	pop    %rbx
  400b1b:	5d                   	pop    %rbp
  400b1c:	41 5c                	pop    %r12
  400b1e:	41 5d                	pop    %r13
  400b20:	41 5e                	pop    %r14
  400b22:	41 5f                	pop    %r15
  400b24:	c3                   	retq   
  400b25:	90                   	nop
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <__libc_csu_fini>:
  400b30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400b34 <_fini>:
  400b34:	48 83 ec 08          	sub    $0x8,%rsp
  400b38:	48 83 c4 08          	add    $0x8,%rsp
  400b3c:	c3                   	retq   
