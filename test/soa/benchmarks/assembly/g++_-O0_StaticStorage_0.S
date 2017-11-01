
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
  4004af:	49 c7 c0 d0 0b 40 00 	mov    $0x400bd0,%r8
  4004b6:	48 c7 c1 60 0b 40 00 	mov    $0x400b60,%rcx
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
  4005ae:	e8 24 02 00 00       	callq  4007d7 <_ZN9TestClass18initialize_storageEv>
  4005b3:	bf 00 00 00 00       	mov    $0x0,%edi
  4005b8:	e8 c7 03 00 00       	callq  400984 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005bd:	48 89 c3             	mov    %rax,%rbx
  4005c0:	48 89 df             	mov    %rbx,%rdi
  4005c3:	e8 ee 02 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
  4005c8:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005cc:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005d0:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005d4:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005d9:	48 89 ce             	mov    %rcx,%rsi
  4005dc:	48 89 c7             	mov    %rax,%rdi
  4005df:	e8 f0 03 00 00       	callq  4009d4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005e4:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005e8:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005ec:	ba 88 88 00 00       	mov    $0x8888,%edx
  4005f1:	48 89 ce             	mov    %rcx,%rsi
  4005f4:	48 89 c7             	mov    %rax,%rdi
  4005f7:	e8 18 04 00 00       	callq  400a14 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4005fc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400600:	48 89 c7             	mov    %rax,%rdi
  400603:	e8 f2 01 00 00       	callq  4007fa <_ZN9TestClass15increase_field0Ev>
  400608:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40060c:	48 89 c7             	mov    %rax,%rdi
  40060f:	e8 30 02 00 00       	callq  400844 <_ZN9TestClass15increase_field1Ev>
  400614:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400618:	48 89 c7             	mov    %rax,%rdi
  40061b:	e8 4e 04 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400620:	8b 18                	mov    (%rax),%ebx
  400622:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400626:	48 89 c7             	mov    %rax,%rdi
  400629:	e8 26 04 00 00       	callq  400a54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40062e:	8b 00                	mov    (%rax),%eax
  400630:	89 da                	mov    %ebx,%edx
  400632:	89 c6                	mov    %eax,%esi
  400634:	bf f4 0b 40 00       	mov    $0x400bf4,%edi
  400639:	b8 00 00 00 00       	mov    $0x0,%eax
  40063e:	e8 2d fe ff ff       	callq  400470 <printf@plt>
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 89 c7             	mov    %rax,%rdi
  40064a:	e8 05 04 00 00       	callq  400a54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40064f:	8b 00                	mov    (%rax),%eax
  400651:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400656:	75 15                	jne    40066d <main+0xd7>
  400658:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40065c:	48 89 c7             	mov    %rax,%rdi
  40065f:	e8 0a 04 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
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
  4006a5:	e8 da 02 00 00       	callq  400984 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4006aa:	48 89 c3             	mov    %rax,%rbx
  4006ad:	48 89 df             	mov    %rbx,%rdi
  4006b0:	e8 01 02 00 00       	callq  4008b6 <_ZN9TestClassC1Ev>
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
  4006ed:	e8 e2 02 00 00       	callq  4009d4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
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
  40071c:	e8 33 03 00 00       	callq  400a54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
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
  400753:	e8 bc 02 00 00       	callq  400a14 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
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
  400782:	e8 e7 02 00 00       	callq  400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400787:	8b 00                	mov    (%rax),%eax
  400789:	c9                   	leaveq 
  40078a:	c3                   	retq   

000000000040078b <_Z21explicit_write_field0m>:
  40078b:	55                   	push   %rbp
  40078c:	48 89 e5             	mov    %rsp,%rbp
  40078f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400793:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400797:	c7 04 85 00 a7 61 00 	movl   $0x7777,0x61a700(,%rax,4)
  40079e:	77 77 00 00 
  4007a2:	90                   	nop
  4007a3:	5d                   	pop    %rbp
  4007a4:	c3                   	retq   

00000000004007a5 <_Z20explicit_read_field0m>:
  4007a5:	55                   	push   %rbp
  4007a6:	48 89 e5             	mov    %rsp,%rbp
  4007a9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ad:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007b1:	8b 04 85 00 a7 61 00 	mov    0x61a700(,%rax,4),%eax
  4007b8:	5d                   	pop    %rbp
  4007b9:	c3                   	retq   

00000000004007ba <_ZnwmPv>:
  4007ba:	55                   	push   %rbp
  4007bb:	48 89 e5             	mov    %rsp,%rbp
  4007be:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007c6:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007ca:	5d                   	pop    %rbp
  4007cb:	c3                   	retq   

00000000004007cc <_ZN9TestClass7storageEv>:
  4007cc:	55                   	push   %rbp
  4007cd:	48 89 e5             	mov    %rsp,%rbp
  4007d0:	b8 60 20 60 00       	mov    $0x602060,%eax
  4007d5:	5d                   	pop    %rbp
  4007d6:	c3                   	retq   

00000000004007d7 <_ZN9TestClass18initialize_storageEv>:
  4007d7:	55                   	push   %rbp
  4007d8:	48 89 e5             	mov    %rsp,%rbp
  4007db:	be 60 20 60 00       	mov    $0x602060,%esi
  4007e0:	bf b0 91 00 00       	mov    $0x91b0,%edi
  4007e5:	e8 d0 ff ff ff       	callq  4007ba <_ZnwmPv>
  4007ea:	48 85 c0             	test   %rax,%rax
  4007ed:	74 08                	je     4007f7 <_ZN9TestClass18initialize_storageEv+0x20>
  4007ef:	48 89 c7             	mov    %rax,%rdi
  4007f2:	e8 f3 00 00 00       	callq  4008ea <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  4007f7:	90                   	nop
  4007f8:	5d                   	pop    %rbp
  4007f9:	c3                   	retq   

00000000004007fa <_ZN9TestClass15increase_field0Ev>:
  4007fa:	55                   	push   %rbp
  4007fb:	48 89 e5             	mov    %rsp,%rbp
  4007fe:	48 83 ec 30          	sub    $0x30,%rsp
  400802:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400806:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40080d:	00 00 
  40080f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400813:	31 c0                	xor    %eax,%eax
  400815:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400819:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40081d:	ba 55 55 00 00       	mov    $0x5555,%edx
  400822:	48 89 ce             	mov    %rcx,%rsi
  400825:	48 89 c7             	mov    %rax,%rdi
  400828:	e8 d7 00 00 00       	callq  400904 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40082d:	90                   	nop
  40082e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400832:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400839:	00 00 
  40083b:	74 05                	je     400842 <_ZN9TestClass15increase_field0Ev+0x48>
  40083d:	e8 1e fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400842:	c9                   	leaveq 
  400843:	c3                   	retq   

0000000000400844 <_ZN9TestClass15increase_field1Ev>:
  400844:	55                   	push   %rbp
  400845:	48 89 e5             	mov    %rsp,%rbp
  400848:	48 83 ec 30          	sub    $0x30,%rsp
  40084c:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400850:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400857:	00 00 
  400859:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40085d:	31 c0                	xor    %eax,%eax
  40085f:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400863:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400867:	ba 44 44 00 00       	mov    $0x4444,%edx
  40086c:	48 89 ce             	mov    %rcx,%rsi
  40086f:	48 89 c7             	mov    %rax,%rdi
  400872:	e8 cd 00 00 00       	callq  400944 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400877:	90                   	nop
  400878:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40087c:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400883:	00 00 
  400885:	74 05                	je     40088c <_ZN9TestClass15increase_field1Ev+0x48>
  400887:	e8 d4 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  40088c:	c9                   	leaveq 
  40088d:	c3                   	retq   

000000000040088e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  40088e:	55                   	push   %rbp
  40088f:	48 89 e5             	mov    %rsp,%rbp
  400892:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400896:	90                   	nop
  400897:	5d                   	pop    %rbp
  400898:	c3                   	retq   
  400899:	90                   	nop

000000000040089a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  40089a:	55                   	push   %rbp
  40089b:	48 89 e5             	mov    %rsp,%rbp
  40089e:	48 83 ec 10          	sub    $0x10,%rsp
  4008a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008aa:	48 89 c7             	mov    %rax,%rdi
  4008ad:	e8 dc ff ff ff       	callq  40088e <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4008b2:	90                   	nop
  4008b3:	c9                   	leaveq 
  4008b4:	c3                   	retq   
  4008b5:	90                   	nop

00000000004008b6 <_ZN9TestClassC1Ev>:
  4008b6:	55                   	push   %rbp
  4008b7:	48 89 e5             	mov    %rsp,%rbp
  4008ba:	48 83 ec 10          	sub    $0x10,%rsp
  4008be:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c6:	48 89 c7             	mov    %rax,%rdi
  4008c9:	e8 cc ff ff ff       	callq  40089a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  4008ce:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008d2:	48 89 c7             	mov    %rax,%rdi
  4008d5:	e8 d8 00 00 00       	callq  4009b2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008da:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 d8 00 00 00       	callq  4009be <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008e6:	90                   	nop
  4008e7:	c9                   	leaveq 
  4008e8:	c3                   	retq   
  4008e9:	90                   	nop

00000000004008ea <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  4008ea:	55                   	push   %rbp
  4008eb:	48 89 e5             	mov    %rsp,%rbp
  4008ee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f6:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  4008fd:	00 00 00 00 
  400901:	90                   	nop
  400902:	5d                   	pop    %rbp
  400903:	c3                   	retq   

0000000000400904 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400904:	55                   	push   %rbp
  400905:	48 89 e5             	mov    %rsp,%rbp
  400908:	48 83 ec 20          	sub    $0x20,%rsp
  40090c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400910:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400914:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400917:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40091b:	48 89 c7             	mov    %rax,%rdi
  40091e:	e8 65 01 00 00       	callq  400a88 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400923:	8b 10                	mov    (%rax),%edx
  400925:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400929:	89 10                	mov    %edx,(%rax)
  40092b:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40092f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400933:	48 89 d6             	mov    %rdx,%rsi
  400936:	48 89 c7             	mov    %rax,%rdi
  400939:	e8 64 01 00 00       	callq  400aa2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  40093e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400942:	c9                   	leaveq 
  400943:	c3                   	retq   

0000000000400944 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400944:	55                   	push   %rbp
  400945:	48 89 e5             	mov    %rsp,%rbp
  400948:	48 83 ec 20          	sub    $0x20,%rsp
  40094c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400950:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400954:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400957:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40095b:	48 89 c7             	mov    %rax,%rdi
  40095e:	e8 4f 01 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400963:	8b 10                	mov    (%rax),%edx
  400965:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400969:	89 10                	mov    %edx,(%rax)
  40096b:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40096f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400973:	48 89 d6             	mov    %rdx,%rsi
  400976:	48 89 c7             	mov    %rax,%rdi
  400979:	e8 4e 01 00 00       	callq  400acc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  40097e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400982:	c9                   	leaveq 
  400983:	c3                   	retq   

0000000000400984 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400984:	55                   	push   %rbp
  400985:	48 89 e5             	mov    %rsp,%rbp
  400988:	48 83 ec 10          	sub    $0x10,%rsp
  40098c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400990:	e8 46 01 00 00       	callq  400adb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400995:	e8 32 fe ff ff       	callq  4007cc <_ZN9TestClass7storageEv>
  40099a:	48 89 c2             	mov    %rax,%rdx
  40099d:	48 8b 02             	mov    (%rdx),%rax
  4009a0:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4009a4:	48 89 0a             	mov    %rcx,(%rdx)
  4009a7:	48 89 c7             	mov    %rax,%rdi
  4009aa:	e8 33 01 00 00       	callq  400ae2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  4009af:	c9                   	leaveq 
  4009b0:	c3                   	retq   
  4009b1:	90                   	nop

00000000004009b2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4009b2:	55                   	push   %rbp
  4009b3:	48 89 e5             	mov    %rsp,%rbp
  4009b6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ba:	90                   	nop
  4009bb:	5d                   	pop    %rbp
  4009bc:	c3                   	retq   
  4009bd:	90                   	nop

00000000004009be <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009be:	55                   	push   %rbp
  4009bf:	48 89 e5             	mov    %rsp,%rbp
  4009c2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c6:	90                   	nop
  4009c7:	5d                   	pop    %rbp
  4009c8:	c3                   	retq   

00000000004009c9 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009c9:	55                   	push   %rbp
  4009ca:	48 89 e5             	mov    %rsp,%rbp
  4009cd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d1:	90                   	nop
  4009d2:	5d                   	pop    %rbp
  4009d3:	c3                   	retq   

00000000004009d4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009d4:	55                   	push   %rbp
  4009d5:	48 89 e5             	mov    %rsp,%rbp
  4009d8:	48 83 ec 20          	sub    $0x20,%rsp
  4009dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e0:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009e4:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009e7:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009eb:	48 89 c7             	mov    %rax,%rdi
  4009ee:	e8 95 00 00 00       	callq  400a88 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009f3:	48 89 c2             	mov    %rax,%rdx
  4009f6:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4009f9:	89 02                	mov    %eax,(%rdx)
  4009fb:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009ff:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a03:	48 89 d6             	mov    %rdx,%rsi
  400a06:	48 89 c7             	mov    %rax,%rdi
  400a09:	e8 94 00 00 00       	callq  400aa2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400a0e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a12:	c9                   	leaveq 
  400a13:	c3                   	retq   

0000000000400a14 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a14:	55                   	push   %rbp
  400a15:	48 89 e5             	mov    %rsp,%rbp
  400a18:	48 83 ec 20          	sub    $0x20,%rsp
  400a1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a20:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a24:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a27:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a2b:	48 89 c7             	mov    %rax,%rdi
  400a2e:	e8 7f 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a33:	48 89 c2             	mov    %rax,%rdx
  400a36:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a39:	89 02                	mov    %eax,(%rdx)
  400a3b:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a3f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a43:	48 89 d6             	mov    %rdx,%rsi
  400a46:	48 89 c7             	mov    %rax,%rdi
  400a49:	e8 7e 00 00 00       	callq  400acc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a4e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a52:	c9                   	leaveq 
  400a53:	c3                   	retq   

0000000000400a54 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a54:	55                   	push   %rbp
  400a55:	48 89 e5             	mov    %rsp,%rbp
  400a58:	48 83 ec 10          	sub    $0x10,%rsp
  400a5c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a60:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a64:	48 89 c7             	mov    %rax,%rdi
  400a67:	e8 1c 00 00 00       	callq  400a88 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a6c:	c9                   	leaveq 
  400a6d:	c3                   	retq   

0000000000400a6e <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a6e:	55                   	push   %rbp
  400a6f:	48 89 e5             	mov    %rsp,%rbp
  400a72:	48 83 ec 10          	sub    $0x10,%rsp
  400a76:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a7a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a7e:	48 89 c7             	mov    %rax,%rdi
  400a81:	e8 2c 00 00 00       	callq  400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a86:	c9                   	leaveq 
  400a87:	c3                   	retq   

0000000000400a88 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a88:	55                   	push   %rbp
  400a89:	48 89 e5             	mov    %rsp,%rbp
  400a8c:	48 83 ec 10          	sub    $0x10,%rsp
  400a90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a94:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a98:	48 89 c7             	mov    %rax,%rdi
  400a9b:	e8 60 00 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa0:	c9                   	leaveq 
  400aa1:	c3                   	retq   

0000000000400aa2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400aa2:	55                   	push   %rbp
  400aa3:	48 89 e5             	mov    %rsp,%rbp
  400aa6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aaa:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aae:	90                   	nop
  400aaf:	5d                   	pop    %rbp
  400ab0:	c3                   	retq   
  400ab1:	90                   	nop

0000000000400ab2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ab2:	55                   	push   %rbp
  400ab3:	48 89 e5             	mov    %rsp,%rbp
  400ab6:	48 83 ec 10          	sub    $0x10,%rsp
  400aba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400abe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac2:	48 89 c7             	mov    %rax,%rdi
  400ac5:	e8 58 00 00 00       	callq  400b22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aca:	c9                   	leaveq 
  400acb:	c3                   	retq   

0000000000400acc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400acc:	55                   	push   %rbp
  400acd:	48 89 e5             	mov    %rsp,%rbp
  400ad0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ad8:	90                   	nop
  400ad9:	5d                   	pop    %rbp
  400ada:	c3                   	retq   

0000000000400adb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400adb:	55                   	push   %rbp
  400adc:	48 89 e5             	mov    %rsp,%rbp
  400adf:	90                   	nop
  400ae0:	5d                   	pop    %rbp
  400ae1:	c3                   	retq   

0000000000400ae2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ae2:	55                   	push   %rbp
  400ae3:	48 89 e5             	mov    %rsp,%rbp
  400ae6:	48 83 ec 10          	sub    $0x10,%rsp
  400aea:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aee:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af2:	48 83 c0 01          	add    $0x1,%rax
  400af6:	48 89 c7             	mov    %rax,%rdi
  400af9:	e8 4c 00 00 00       	callq  400b4a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400afe:	c9                   	leaveq 
  400aff:	c3                   	retq   

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 10          	sub    $0x10,%rsp
  400b08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0c:	e8 bb fc ff ff       	callq  4007cc <_ZN9TestClass7storageEv>
  400b11:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400b15:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b19:	48 c1 e0 02          	shl    $0x2,%rax
  400b1d:	48 01 d0             	add    %rdx,%rax
  400b20:	c9                   	leaveq 
  400b21:	c3                   	retq   

0000000000400b22 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b22:	55                   	push   %rbp
  400b23:	48 89 e5             	mov    %rsp,%rbp
  400b26:	48 83 ec 10          	sub    $0x10,%rsp
  400b2a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b2e:	e8 99 fc ff ff       	callq  4007cc <_ZN9TestClass7storageEv>
  400b33:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400b37:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b3b:	48 c1 e0 02          	shl    $0x2,%rax
  400b3f:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400b45:	48 01 d0             	add    %rdx,%rax
  400b48:	c9                   	leaveq 
  400b49:	c3                   	retq   

0000000000400b4a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b4a:	55                   	push   %rbp
  400b4b:	48 89 e5             	mov    %rsp,%rbp
  400b4e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b52:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b56:	5d                   	pop    %rbp
  400b57:	c3                   	retq   
  400b58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b5f:	00 

0000000000400b60 <__libc_csu_init>:
  400b60:	41 57                	push   %r15
  400b62:	41 56                	push   %r14
  400b64:	41 89 ff             	mov    %edi,%r15d
  400b67:	41 55                	push   %r13
  400b69:	41 54                	push   %r12
  400b6b:	4c 8d 25 9e 12 20 00 	lea    0x20129e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400b72:	55                   	push   %rbp
  400b73:	48 8d 2d 9e 12 20 00 	lea    0x20129e(%rip),%rbp        # 601e18 <__init_array_end>
  400b7a:	53                   	push   %rbx
  400b7b:	49 89 f6             	mov    %rsi,%r14
  400b7e:	49 89 d5             	mov    %rdx,%r13
  400b81:	4c 29 e5             	sub    %r12,%rbp
  400b84:	48 83 ec 08          	sub    $0x8,%rsp
  400b88:	48 c1 fd 03          	sar    $0x3,%rbp
  400b8c:	e8 97 f8 ff ff       	callq  400428 <_init>
  400b91:	48 85 ed             	test   %rbp,%rbp
  400b94:	74 20                	je     400bb6 <__libc_csu_init+0x56>
  400b96:	31 db                	xor    %ebx,%ebx
  400b98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b9f:	00 
  400ba0:	4c 89 ea             	mov    %r13,%rdx
  400ba3:	4c 89 f6             	mov    %r14,%rsi
  400ba6:	44 89 ff             	mov    %r15d,%edi
  400ba9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400bad:	48 83 c3 01          	add    $0x1,%rbx
  400bb1:	48 39 eb             	cmp    %rbp,%rbx
  400bb4:	75 ea                	jne    400ba0 <__libc_csu_init+0x40>
  400bb6:	48 83 c4 08          	add    $0x8,%rsp
  400bba:	5b                   	pop    %rbx
  400bbb:	5d                   	pop    %rbp
  400bbc:	41 5c                	pop    %r12
  400bbe:	41 5d                	pop    %r13
  400bc0:	41 5e                	pop    %r14
  400bc2:	41 5f                	pop    %r15
  400bc4:	c3                   	retq   
  400bc5:	90                   	nop
  400bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bcd:	00 00 00 

0000000000400bd0 <__libc_csu_fini>:
  400bd0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400bd4 <_fini>:
  400bd4:	48 83 ec 08          	sub    $0x8,%rsp
  400bd8:	48 83 c4 08          	add    $0x8,%rsp
  400bdc:	c3                   	retq   
