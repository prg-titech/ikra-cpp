
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
  4004af:	49 c7 c0 b0 0b 40 00 	mov    $0x400bb0,%r8
  4004b6:	48 c7 c1 40 0b 40 00 	mov    $0x400b40,%rcx
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
  4005ae:	e8 36 02 00 00       	callq  4007e9 <_ZN9TestClass18initialize_storageEv>
  4005b3:	bf 00 00 00 00       	mov    $0x0,%edi
  4005b8:	e8 d9 03 00 00       	callq  400996 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005bd:	48 89 c3             	mov    %rax,%rbx
  4005c0:	48 89 df             	mov    %rbx,%rdi
  4005c3:	e8 00 03 00 00       	callq  4008c8 <_ZN9TestClassC1Ev>
  4005c8:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005cc:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005d0:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005d4:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005d9:	48 89 ce             	mov    %rcx,%rsi
  4005dc:	48 89 c7             	mov    %rax,%rdi
  4005df:	e8 02 04 00 00       	callq  4009e6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005e4:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005e8:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005ec:	ba 88 88 00 00       	mov    $0x8888,%edx
  4005f1:	48 89 ce             	mov    %rcx,%rsi
  4005f4:	48 89 c7             	mov    %rax,%rdi
  4005f7:	e8 2a 04 00 00       	callq  400a26 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4005fc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400600:	48 89 c7             	mov    %rax,%rdi
  400603:	e8 04 02 00 00       	callq  40080c <_ZN9TestClass15increase_field0Ev>
  400608:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40060c:	48 89 c7             	mov    %rax,%rdi
  40060f:	e8 42 02 00 00       	callq  400856 <_ZN9TestClass15increase_field1Ev>
  400614:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400618:	48 89 c7             	mov    %rax,%rdi
  40061b:	e8 60 04 00 00       	callq  400a80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400620:	8b 18                	mov    (%rax),%ebx
  400622:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400626:	48 89 c7             	mov    %rax,%rdi
  400629:	e8 38 04 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40062e:	8b 00                	mov    (%rax),%eax
  400630:	89 da                	mov    %ebx,%edx
  400632:	89 c6                	mov    %eax,%esi
  400634:	bf d4 0b 40 00       	mov    $0x400bd4,%edi
  400639:	b8 00 00 00 00       	mov    $0x0,%eax
  40063e:	e8 2d fe ff ff       	callq  400470 <printf@plt>
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 89 c7             	mov    %rax,%rdi
  40064a:	e8 17 04 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40064f:	8b 00                	mov    (%rax),%eax
  400651:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400656:	75 15                	jne    40066d <main+0xd7>
  400658:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40065c:	48 89 c7             	mov    %rax,%rdi
  40065f:	e8 1c 04 00 00       	callq  400a80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400664:	8b 00                	mov    (%rax),%eax
  400666:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  40066b:	74 07                	je     400674 <main+0xde>
  40066d:	b8 01 00 00 00       	mov    $0x1,%eax
  400672:	eb 05                	jmp    400679 <main+0xe3>
  400674:	b8 00 00 00 00       	mov    $0x0,%eax
  400679:	0f b6 c0             	movzbl %al,%eax
  40067c:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  400680:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  400687:	00 00 
  400689:	74 05                	je     400690 <main+0xfa>
  40068b:	e8 d0 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400690:	48 83 c4 28          	add    $0x28,%rsp
  400694:	5b                   	pop    %rbx
  400695:	5d                   	pop    %rbp
  400696:	c3                   	retq   

0000000000400697 <_Z12new_instancev>:
  400697:	55                   	push   %rbp
  400698:	48 89 e5             	mov    %rsp,%rbp
  40069b:	53                   	push   %rbx
  40069c:	48 83 ec 08          	sub    $0x8,%rsp
  4006a0:	bf 00 00 00 00       	mov    $0x0,%edi
  4006a5:	e8 ec 02 00 00       	callq  400996 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4006aa:	48 89 c3             	mov    %rax,%rbx
  4006ad:	48 89 df             	mov    %rbx,%rdi
  4006b0:	e8 13 02 00 00       	callq  4008c8 <_ZN9TestClassC1Ev>
  4006b5:	48 89 d8             	mov    %rbx,%rax
  4006b8:	48 83 c4 08          	add    $0x8,%rsp
  4006bc:	5b                   	pop    %rbx
  4006bd:	5d                   	pop    %rbp
  4006be:	c3                   	retq   

00000000004006bf <_Z12write_field0P9TestClass>:
  4006bf:	55                   	push   %rbp
  4006c0:	48 89 e5             	mov    %rsp,%rbp
  4006c3:	48 83 ec 30          	sub    $0x30,%rsp
  4006c7:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4006cb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4006d2:	00 00 
  4006d4:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4006d8:	31 c0                	xor    %eax,%eax
  4006da:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4006de:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4006e2:	ba 77 77 00 00       	mov    $0x7777,%edx
  4006e7:	48 89 ce             	mov    %rcx,%rsi
  4006ea:	48 89 c7             	mov    %rax,%rdi
  4006ed:	e8 f4 02 00 00       	callq  4009e6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006f2:	90                   	nop
  4006f3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006f7:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4006fe:	00 00 
  400700:	74 05                	je     400707 <_Z12write_field0P9TestClass+0x48>
  400702:	e8 59 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400707:	c9                   	leaveq 
  400708:	c3                   	retq   

0000000000400709 <_Z11read_field0P9TestClass>:
  400709:	55                   	push   %rbp
  40070a:	48 89 e5             	mov    %rsp,%rbp
  40070d:	48 83 ec 10          	sub    $0x10,%rsp
  400711:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400715:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400719:	48 89 c7             	mov    %rax,%rdi
  40071c:	e8 45 03 00 00       	callq  400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400721:	8b 00                	mov    (%rax),%eax
  400723:	c9                   	leaveq 
  400724:	c3                   	retq   

0000000000400725 <_Z12write_field1P9TestClass>:
  400725:	55                   	push   %rbp
  400726:	48 89 e5             	mov    %rsp,%rbp
  400729:	48 83 ec 30          	sub    $0x30,%rsp
  40072d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400731:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400738:	00 00 
  40073a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40073e:	31 c0                	xor    %eax,%eax
  400740:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400744:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400748:	ba 77 77 00 00       	mov    $0x7777,%edx
  40074d:	48 89 ce             	mov    %rcx,%rsi
  400750:	48 89 c7             	mov    %rax,%rdi
  400753:	e8 ce 02 00 00       	callq  400a26 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400758:	90                   	nop
  400759:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400764:	00 00 
  400766:	74 05                	je     40076d <_Z12write_field1P9TestClass+0x48>
  400768:	e8 f3 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  40076d:	c9                   	leaveq 
  40076e:	c3                   	retq   

000000000040076f <_Z11read_field1P9TestClass>:
  40076f:	55                   	push   %rbp
  400770:	48 89 e5             	mov    %rsp,%rbp
  400773:	48 83 ec 10          	sub    $0x10,%rsp
  400777:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40077b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40077f:	48 89 c7             	mov    %rax,%rdi
  400782:	e8 f9 02 00 00       	callq  400a80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400787:	8b 00                	mov    (%rax),%eax
  400789:	c9                   	leaveq 
  40078a:	c3                   	retq   

000000000040078b <_Z21explicit_write_field0m>:
  40078b:	55                   	push   %rbp
  40078c:	48 89 e5             	mov    %rsp,%rbp
  40078f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400793:	48 8b 05 66 9f 21 00 	mov    0x219f66(%rip),%rax        # 61a700 <explicit_field0>
  40079a:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  40079e:	48 c1 e2 02          	shl    $0x2,%rdx
  4007a2:	48 01 d0             	add    %rdx,%rax
  4007a5:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007ab:	90                   	nop
  4007ac:	5d                   	pop    %rbp
  4007ad:	c3                   	retq   

00000000004007ae <_Z20explicit_read_field0m>:
  4007ae:	55                   	push   %rbp
  4007af:	48 89 e5             	mov    %rsp,%rbp
  4007b2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b6:	48 8b 05 43 9f 21 00 	mov    0x219f43(%rip),%rax        # 61a700 <explicit_field0>
  4007bd:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007c1:	48 c1 e2 02          	shl    $0x2,%rdx
  4007c5:	48 01 d0             	add    %rdx,%rax
  4007c8:	8b 00                	mov    (%rax),%eax
  4007ca:	5d                   	pop    %rbp
  4007cb:	c3                   	retq   

00000000004007cc <_ZnwmPv>:
  4007cc:	55                   	push   %rbp
  4007cd:	48 89 e5             	mov    %rsp,%rbp
  4007d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007d8:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007dc:	5d                   	pop    %rbp
  4007dd:	c3                   	retq   

00000000004007de <_ZN9TestClass7storageEv>:
  4007de:	55                   	push   %rbp
  4007df:	48 89 e5             	mov    %rsp,%rbp
  4007e2:	b8 60 20 60 00       	mov    $0x602060,%eax
  4007e7:	5d                   	pop    %rbp
  4007e8:	c3                   	retq   

00000000004007e9 <_ZN9TestClass18initialize_storageEv>:
  4007e9:	55                   	push   %rbp
  4007ea:	48 89 e5             	mov    %rsp,%rbp
  4007ed:	be 60 20 60 00       	mov    $0x602060,%esi
  4007f2:	bf b0 91 00 00       	mov    $0x91b0,%edi
  4007f7:	e8 d0 ff ff ff       	callq  4007cc <_ZnwmPv>
  4007fc:	48 85 c0             	test   %rax,%rax
  4007ff:	74 08                	je     400809 <_ZN9TestClass18initialize_storageEv+0x20>
  400801:	48 89 c7             	mov    %rax,%rdi
  400804:	e8 f3 00 00 00       	callq  4008fc <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400809:	90                   	nop
  40080a:	5d                   	pop    %rbp
  40080b:	c3                   	retq   

000000000040080c <_ZN9TestClass15increase_field0Ev>:
  40080c:	55                   	push   %rbp
  40080d:	48 89 e5             	mov    %rsp,%rbp
  400810:	48 83 ec 30          	sub    $0x30,%rsp
  400814:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400818:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40081f:	00 00 
  400821:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400825:	31 c0                	xor    %eax,%eax
  400827:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40082b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40082f:	ba 55 55 00 00       	mov    $0x5555,%edx
  400834:	48 89 ce             	mov    %rcx,%rsi
  400837:	48 89 c7             	mov    %rax,%rdi
  40083a:	e8 d7 00 00 00       	callq  400916 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40083f:	90                   	nop
  400840:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400844:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40084b:	00 00 
  40084d:	74 05                	je     400854 <_ZN9TestClass15increase_field0Ev+0x48>
  40084f:	e8 0c fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400854:	c9                   	leaveq 
  400855:	c3                   	retq   

0000000000400856 <_ZN9TestClass15increase_field1Ev>:
  400856:	55                   	push   %rbp
  400857:	48 89 e5             	mov    %rsp,%rbp
  40085a:	48 83 ec 30          	sub    $0x30,%rsp
  40085e:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400862:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400869:	00 00 
  40086b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40086f:	31 c0                	xor    %eax,%eax
  400871:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400875:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400879:	ba 44 44 00 00       	mov    $0x4444,%edx
  40087e:	48 89 ce             	mov    %rcx,%rsi
  400881:	48 89 c7             	mov    %rax,%rdi
  400884:	e8 cd 00 00 00       	callq  400956 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400889:	90                   	nop
  40088a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40088e:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400895:	00 00 
  400897:	74 05                	je     40089e <_ZN9TestClass15increase_field1Ev+0x48>
  400899:	e8 c2 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  40089e:	c9                   	leaveq 
  40089f:	c3                   	retq   

00000000004008a0 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a8:	90                   	nop
  4008a9:	5d                   	pop    %rbp
  4008aa:	c3                   	retq   
  4008ab:	90                   	nop

00000000004008ac <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  4008ac:	55                   	push   %rbp
  4008ad:	48 89 e5             	mov    %rsp,%rbp
  4008b0:	48 83 ec 10          	sub    $0x10,%rsp
  4008b4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008bc:	48 89 c7             	mov    %rax,%rdi
  4008bf:	e8 dc ff ff ff       	callq  4008a0 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008c4:	90                   	nop
  4008c5:	c9                   	leaveq 
  4008c6:	c3                   	retq   
  4008c7:	90                   	nop

00000000004008c8 <_ZN9TestClassC1Ev>:
  4008c8:	55                   	push   %rbp
  4008c9:	48 89 e5             	mov    %rsp,%rbp
  4008cc:	48 83 ec 10          	sub    $0x10,%rsp
  4008d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008d8:	48 89 c7             	mov    %rax,%rdi
  4008db:	e8 cc ff ff ff       	callq  4008ac <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  4008e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e4:	48 89 c7             	mov    %rax,%rdi
  4008e7:	e8 d8 00 00 00       	callq  4009c4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008ec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f0:	48 89 c7             	mov    %rax,%rdi
  4008f3:	e8 d8 00 00 00       	callq  4009d0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008f8:	90                   	nop
  4008f9:	c9                   	leaveq 
  4008fa:	c3                   	retq   
  4008fb:	90                   	nop

00000000004008fc <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  4008fc:	55                   	push   %rbp
  4008fd:	48 89 e5             	mov    %rsp,%rbp
  400900:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400904:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400908:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  40090f:	00 00 00 00 
  400913:	90                   	nop
  400914:	5d                   	pop    %rbp
  400915:	c3                   	retq   

0000000000400916 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400916:	55                   	push   %rbp
  400917:	48 89 e5             	mov    %rsp,%rbp
  40091a:	48 83 ec 20          	sub    $0x20,%rsp
  40091e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400922:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400926:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400929:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40092d:	48 89 c7             	mov    %rax,%rdi
  400930:	e8 65 01 00 00       	callq  400a9a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400935:	8b 10                	mov    (%rax),%edx
  400937:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40093b:	89 10                	mov    %edx,(%rax)
  40093d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400941:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400945:	48 89 d6             	mov    %rdx,%rsi
  400948:	48 89 c7             	mov    %rax,%rdi
  40094b:	e8 6c 01 00 00       	callq  400abc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400950:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400954:	c9                   	leaveq 
  400955:	c3                   	retq   

0000000000400956 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400956:	55                   	push   %rbp
  400957:	48 89 e5             	mov    %rsp,%rbp
  40095a:	48 83 ec 20          	sub    $0x20,%rsp
  40095e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400962:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400966:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400969:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40096d:	48 89 c7             	mov    %rax,%rdi
  400970:	e8 57 01 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400975:	8b 10                	mov    (%rax),%edx
  400977:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40097b:	89 10                	mov    %edx,(%rax)
  40097d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400981:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400985:	48 89 d6             	mov    %rdx,%rsi
  400988:	48 89 c7             	mov    %rax,%rdi
  40098b:	e8 64 01 00 00       	callq  400af4 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400990:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400994:	c9                   	leaveq 
  400995:	c3                   	retq   

0000000000400996 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400996:	55                   	push   %rbp
  400997:	48 89 e5             	mov    %rsp,%rbp
  40099a:	48 83 ec 10          	sub    $0x10,%rsp
  40099e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a2:	e8 5c 01 00 00       	callq  400b03 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009a7:	e8 32 fe ff ff       	callq  4007de <_ZN9TestClass7storageEv>
  4009ac:	48 89 c2             	mov    %rax,%rdx
  4009af:	48 8b 02             	mov    (%rdx),%rax
  4009b2:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4009b6:	48 89 0a             	mov    %rcx,(%rdx)
  4009b9:	48 89 c7             	mov    %rax,%rdi
  4009bc:	e8 49 01 00 00       	callq  400b0a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  4009c1:	c9                   	leaveq 
  4009c2:	c3                   	retq   
  4009c3:	90                   	nop

00000000004009c4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4009c4:	55                   	push   %rbp
  4009c5:	48 89 e5             	mov    %rsp,%rbp
  4009c8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009cc:	90                   	nop
  4009cd:	5d                   	pop    %rbp
  4009ce:	c3                   	retq   
  4009cf:	90                   	nop

00000000004009d0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d8:	90                   	nop
  4009d9:	5d                   	pop    %rbp
  4009da:	c3                   	retq   

00000000004009db <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009db:	55                   	push   %rbp
  4009dc:	48 89 e5             	mov    %rsp,%rbp
  4009df:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e3:	90                   	nop
  4009e4:	5d                   	pop    %rbp
  4009e5:	c3                   	retq   

00000000004009e6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009e6:	55                   	push   %rbp
  4009e7:	48 89 e5             	mov    %rsp,%rbp
  4009ea:	48 83 ec 20          	sub    $0x20,%rsp
  4009ee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009f6:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009f9:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009fd:	48 89 c7             	mov    %rax,%rdi
  400a00:	e8 95 00 00 00       	callq  400a9a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a05:	48 89 c2             	mov    %rax,%rdx
  400a08:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a0b:	89 02                	mov    %eax,(%rdx)
  400a0d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a11:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a15:	48 89 d6             	mov    %rdx,%rsi
  400a18:	48 89 c7             	mov    %rax,%rdi
  400a1b:	e8 9c 00 00 00       	callq  400abc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a20:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a24:	c9                   	leaveq 
  400a25:	c3                   	retq   

0000000000400a26 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a26:	55                   	push   %rbp
  400a27:	48 89 e5             	mov    %rsp,%rbp
  400a2a:	48 83 ec 20          	sub    $0x20,%rsp
  400a2e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a32:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a36:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a39:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a3d:	48 89 c7             	mov    %rax,%rdi
  400a40:	e8 87 00 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a45:	48 89 c2             	mov    %rax,%rdx
  400a48:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a4b:	89 02                	mov    %eax,(%rdx)
  400a4d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a51:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a55:	48 89 d6             	mov    %rdx,%rsi
  400a58:	48 89 c7             	mov    %rax,%rdi
  400a5b:	e8 94 00 00 00       	callq  400af4 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a60:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a64:	c9                   	leaveq 
  400a65:	c3                   	retq   

0000000000400a66 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a66:	55                   	push   %rbp
  400a67:	48 89 e5             	mov    %rsp,%rbp
  400a6a:	48 83 ec 10          	sub    $0x10,%rsp
  400a6e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a72:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a76:	48 89 c7             	mov    %rax,%rdi
  400a79:	e8 1c 00 00 00       	callq  400a9a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a7e:	c9                   	leaveq 
  400a7f:	c3                   	retq   

0000000000400a80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 83 ec 10          	sub    $0x10,%rsp
  400a88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a90:	48 89 c7             	mov    %rax,%rdi
  400a93:	e8 34 00 00 00       	callq  400acc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a98:	c9                   	leaveq 
  400a99:	c3                   	retq   

0000000000400a9a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a9a:	55                   	push   %rbp
  400a9b:	48 89 e5             	mov    %rsp,%rbp
  400a9e:	48 83 ec 10          	sub    $0x10,%rsp
  400aa2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa6:	e8 33 fd ff ff       	callq  4007de <_ZN9TestClass7storageEv>
  400aab:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400aaf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab3:	48 c1 e0 02          	shl    $0x2,%rax
  400ab7:	48 01 d0             	add    %rdx,%rax
  400aba:	c9                   	leaveq 
  400abb:	c3                   	retq   

0000000000400abc <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400abc:	55                   	push   %rbp
  400abd:	48 89 e5             	mov    %rsp,%rbp
  400ac0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ac8:	90                   	nop
  400ac9:	5d                   	pop    %rbp
  400aca:	c3                   	retq   
  400acb:	90                   	nop

0000000000400acc <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400acc:	55                   	push   %rbp
  400acd:	48 89 e5             	mov    %rsp,%rbp
  400ad0:	48 83 ec 10          	sub    $0x10,%rsp
  400ad4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad8:	e8 01 fd ff ff       	callq  4007de <_ZN9TestClass7storageEv>
  400add:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400ae1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae5:	48 c1 e0 02          	shl    $0x2,%rax
  400ae9:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400aef:	48 01 d0             	add    %rdx,%rax
  400af2:	c9                   	leaveq 
  400af3:	c3                   	retq   

0000000000400af4 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400af4:	55                   	push   %rbp
  400af5:	48 89 e5             	mov    %rsp,%rbp
  400af8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400afc:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b00:	90                   	nop
  400b01:	5d                   	pop    %rbp
  400b02:	c3                   	retq   

0000000000400b03 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b03:	55                   	push   %rbp
  400b04:	48 89 e5             	mov    %rsp,%rbp
  400b07:	90                   	nop
  400b08:	5d                   	pop    %rbp
  400b09:	c3                   	retq   

0000000000400b0a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400b0a:	55                   	push   %rbp
  400b0b:	48 89 e5             	mov    %rsp,%rbp
  400b0e:	48 83 ec 10          	sub    $0x10,%rsp
  400b12:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b16:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1a:	48 89 c7             	mov    %rax,%rdi
  400b1d:	e8 02 00 00 00       	callq  400b24 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b22:	c9                   	leaveq 
  400b23:	c3                   	retq   

0000000000400b24 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b24:	55                   	push   %rbp
  400b25:	48 89 e5             	mov    %rsp,%rbp
  400b28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b2c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b30:	5d                   	pop    %rbp
  400b31:	c3                   	retq   
  400b32:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b39:	00 00 00 
  400b3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b40 <__libc_csu_init>:
  400b40:	41 57                	push   %r15
  400b42:	41 56                	push   %r14
  400b44:	41 89 ff             	mov    %edi,%r15d
  400b47:	41 55                	push   %r13
  400b49:	41 54                	push   %r12
  400b4b:	4c 8d 25 be 12 20 00 	lea    0x2012be(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400b52:	55                   	push   %rbp
  400b53:	48 8d 2d be 12 20 00 	lea    0x2012be(%rip),%rbp        # 601e18 <__init_array_end>
  400b5a:	53                   	push   %rbx
  400b5b:	49 89 f6             	mov    %rsi,%r14
  400b5e:	49 89 d5             	mov    %rdx,%r13
  400b61:	4c 29 e5             	sub    %r12,%rbp
  400b64:	48 83 ec 08          	sub    $0x8,%rsp
  400b68:	48 c1 fd 03          	sar    $0x3,%rbp
  400b6c:	e8 b7 f8 ff ff       	callq  400428 <_init>
  400b71:	48 85 ed             	test   %rbp,%rbp
  400b74:	74 20                	je     400b96 <__libc_csu_init+0x56>
  400b76:	31 db                	xor    %ebx,%ebx
  400b78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b7f:	00 
  400b80:	4c 89 ea             	mov    %r13,%rdx
  400b83:	4c 89 f6             	mov    %r14,%rsi
  400b86:	44 89 ff             	mov    %r15d,%edi
  400b89:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400b8d:	48 83 c3 01          	add    $0x1,%rbx
  400b91:	48 39 eb             	cmp    %rbp,%rbx
  400b94:	75 ea                	jne    400b80 <__libc_csu_init+0x40>
  400b96:	48 83 c4 08          	add    $0x8,%rsp
  400b9a:	5b                   	pop    %rbx
  400b9b:	5d                   	pop    %rbp
  400b9c:	41 5c                	pop    %r12
  400b9e:	41 5d                	pop    %r13
  400ba0:	41 5e                	pop    %r14
  400ba2:	41 5f                	pop    %r15
  400ba4:	c3                   	retq   
  400ba5:	90                   	nop
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 

0000000000400bb0 <__libc_csu_fini>:
  400bb0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400bb4 <_fini>:
  400bb4:	48 83 ec 08          	sub    $0x8,%rsp
  400bb8:	48 83 c4 08          	add    $0x8,%rsp
  400bbc:	c3                   	retq   
