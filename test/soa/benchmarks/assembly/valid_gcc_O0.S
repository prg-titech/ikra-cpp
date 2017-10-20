
bin/valid_gcc_O0:     file format elf64-x86-64


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
  4004af:	49 c7 c0 50 0b 40 00 	mov    $0x400b50,%r8
  4004b6:	48 c7 c1 e0 0a 40 00 	mov    $0x400ae0,%rcx
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

0000000000400596 <_ZN4ikra3soa12_GLOBAL__N_110SizeNDummyILm4EEC2Ev>:
  400596:	55                   	push   %rbp
  400597:	48 89 e5             	mov    %rsp,%rbp
  40059a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40059e:	90                   	nop
  40059f:	5d                   	pop    %rbp
  4005a0:	c3                   	retq   

00000000004005a1 <main>:
  4005a1:	55                   	push   %rbp
  4005a2:	48 89 e5             	mov    %rsp,%rbp
  4005a5:	53                   	push   %rbx
  4005a6:	48 83 ec 28          	sub    $0x28,%rsp
  4005aa:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005b1:	00 00 
  4005b3:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005b7:	31 c0                	xor    %eax,%eax
  4005b9:	bf 04 00 00 00       	mov    $0x4,%edi
  4005be:	e8 9c 04 00 00       	callq  400a5f <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>
  4005c3:	48 89 c3             	mov    %rax,%rbx
  4005c6:	b8 00 00 00 00       	mov    $0x0,%eax
  4005cb:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005cf:	48 83 f8 03          	cmp    $0x3,%rax
  4005d3:	74 06                	je     4005db <main+0x3a>
  4005d5:	48 83 c0 01          	add    $0x1,%rax
  4005d9:	eb f0                	jmp    4005cb <main+0x2a>
  4005db:	48 89 df             	mov    %rbx,%rdi
  4005de:	e8 41 04 00 00       	callq  400a24 <_ZN9TestClassC1Ev>
  4005e3:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005e7:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005eb:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4005ef:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005f3:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005f8:	48 89 ce             	mov    %rcx,%rsi
  4005fb:	48 89 c7             	mov    %rax,%rdi
  4005fe:	e8 0b 02 00 00       	callq  40080e <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400603:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400607:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40060b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40060f:	ba 88 88 00 00       	mov    $0x8888,%edx
  400614:	48 89 ce             	mov    %rcx,%rsi
  400617:	48 89 c7             	mov    %rax,%rdi
  40061a:	e8 2f 02 00 00       	callq  40084e <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40061f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400623:	48 89 c7             	mov    %rax,%rdi
  400626:	e8 41 03 00 00       	callq  40096c <_ZN9TestClass15increase_field0Ev>
  40062b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40062f:	48 89 c7             	mov    %rax,%rdi
  400632:	e8 83 03 00 00       	callq  4009ba <_ZN9TestClass15increase_field1Ev>
  400637:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40063b:	48 83 c0 04          	add    $0x4,%rax
  40063f:	48 89 c7             	mov    %rax,%rdi
  400642:	e8 61 02 00 00       	callq  4008a8 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400647:	8b 18                	mov    (%rax),%ebx
  400649:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064d:	48 83 c0 04          	add    $0x4,%rax
  400651:	48 89 c7             	mov    %rax,%rdi
  400654:	e8 35 02 00 00       	callq  40088e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400659:	8b 00                	mov    (%rax),%eax
  40065b:	89 da                	mov    %ebx,%edx
  40065d:	89 c6                	mov    %eax,%esi
  40065f:	bf 70 0b 40 00       	mov    $0x400b70,%edi
  400664:	b8 00 00 00 00       	mov    $0x0,%eax
  400669:	e8 02 fe ff ff       	callq  400470 <printf@plt>
  40066e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400672:	48 83 c0 04          	add    $0x4,%rax
  400676:	48 89 c7             	mov    %rax,%rdi
  400679:	e8 10 02 00 00       	callq  40088e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40067e:	8b 00                	mov    (%rax),%eax
  400680:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400685:	75 19                	jne    4006a0 <main+0xff>
  400687:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40068b:	48 83 c0 04          	add    $0x4,%rax
  40068f:	48 89 c7             	mov    %rax,%rdi
  400692:	e8 11 02 00 00       	callq  4008a8 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400697:	8b 00                	mov    (%rax),%eax
  400699:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  40069e:	74 07                	je     4006a7 <main+0x106>
  4006a0:	b8 01 00 00 00       	mov    $0x1,%eax
  4006a5:	eb 05                	jmp    4006ac <main+0x10b>
  4006a7:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ac:	0f b6 c0             	movzbl %al,%eax
  4006af:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006b3:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006ba:	00 00 
  4006bc:	74 05                	je     4006c3 <main+0x122>
  4006be:	e8 9d fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  4006c3:	48 83 c4 28          	add    $0x28,%rsp
  4006c7:	5b                   	pop    %rbx
  4006c8:	5d                   	pop    %rbp
  4006c9:	c3                   	retq   

00000000004006ca <_Z12new_instancev>:
  4006ca:	55                   	push   %rbp
  4006cb:	48 89 e5             	mov    %rsp,%rbp
  4006ce:	53                   	push   %rbx
  4006cf:	48 83 ec 08          	sub    $0x8,%rsp
  4006d3:	bf 04 00 00 00       	mov    $0x4,%edi
  4006d8:	e8 82 03 00 00       	callq  400a5f <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>
  4006dd:	48 89 c3             	mov    %rax,%rbx
  4006e0:	b8 00 00 00 00       	mov    $0x0,%eax
  4006e5:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006e9:	48 83 f8 03          	cmp    $0x3,%rax
  4006ed:	74 06                	je     4006f5 <_Z12new_instancev+0x2b>
  4006ef:	48 83 c0 01          	add    $0x1,%rax
  4006f3:	eb f0                	jmp    4006e5 <_Z12new_instancev+0x1b>
  4006f5:	48 89 df             	mov    %rbx,%rdi
  4006f8:	e8 27 03 00 00       	callq  400a24 <_ZN9TestClassC1Ev>
  4006fd:	48 89 d8             	mov    %rbx,%rax
  400700:	48 83 c4 08          	add    $0x8,%rsp
  400704:	5b                   	pop    %rbx
  400705:	5d                   	pop    %rbp
  400706:	c3                   	retq   

0000000000400707 <_Z12write_field0P9TestClass>:
  400707:	55                   	push   %rbp
  400708:	48 89 e5             	mov    %rsp,%rbp
  40070b:	48 83 ec 30          	sub    $0x30,%rsp
  40070f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400713:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40071a:	00 00 
  40071c:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400720:	31 c0                	xor    %eax,%eax
  400722:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400726:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40072a:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40072e:	ba 77 77 00 00       	mov    $0x7777,%edx
  400733:	48 89 ce             	mov    %rcx,%rsi
  400736:	48 89 c7             	mov    %rax,%rdi
  400739:	e8 d0 00 00 00       	callq  40080e <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40073e:	90                   	nop
  40073f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400743:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40074a:	00 00 
  40074c:	74 05                	je     400753 <_Z12write_field0P9TestClass+0x4c>
  40074e:	e8 0d fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400753:	c9                   	leaveq 
  400754:	c3                   	retq   

0000000000400755 <_Z11read_field0P9TestClass>:
  400755:	55                   	push   %rbp
  400756:	48 89 e5             	mov    %rsp,%rbp
  400759:	48 83 ec 10          	sub    $0x10,%rsp
  40075d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400761:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400765:	48 83 c0 04          	add    $0x4,%rax
  400769:	48 89 c7             	mov    %rax,%rdi
  40076c:	e8 1d 01 00 00       	callq  40088e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400771:	8b 00                	mov    (%rax),%eax
  400773:	c9                   	leaveq 
  400774:	c3                   	retq   
  400775:	90                   	nop

0000000000400776 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400776:	55                   	push   %rbp
  400777:	48 89 e5             	mov    %rsp,%rbp
  40077a:	48 83 ec 20          	sub    $0x20,%rsp
  40077e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400782:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400786:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400789:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40078d:	48 89 c7             	mov    %rax,%rdi
  400790:	e8 2d 01 00 00       	callq  4008c2 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400795:	8b 10                	mov    (%rax),%edx
  400797:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40079b:	89 10                	mov    %edx,(%rax)
  40079d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4007a1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007a5:	48 89 d6             	mov    %rdx,%rsi
  4007a8:	48 89 c7             	mov    %rax,%rdi
  4007ab:	e8 54 01 00 00       	callq  400904 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS4_>
  4007b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007b4:	c9                   	leaveq 
  4007b5:	c3                   	retq   

00000000004007b6 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  4007b6:	55                   	push   %rbp
  4007b7:	48 89 e5             	mov    %rsp,%rbp
  4007ba:	48 83 ec 20          	sub    $0x20,%rsp
  4007be:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007c6:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4007c9:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007cd:	48 89 c7             	mov    %rax,%rdi
  4007d0:	e8 3f 01 00 00       	callq  400914 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4007d5:	8b 10                	mov    (%rax),%edx
  4007d7:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4007db:	89 10                	mov    %edx,(%rax)
  4007dd:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4007e1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007e5:	48 89 d6             	mov    %rdx,%rsi
  4007e8:	48 89 c7             	mov    %rax,%rdi
  4007eb:	e8 6c 01 00 00       	callq  40095c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS4_>
  4007f0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f4:	c9                   	leaveq 
  4007f5:	c3                   	retq   

00000000004007f6 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4007f6:	55                   	push   %rbp
  4007f7:	48 89 e5             	mov    %rsp,%rbp
  4007fa:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007fe:	90                   	nop
  4007ff:	5d                   	pop    %rbp
  400800:	c3                   	retq   
  400801:	90                   	nop

0000000000400802 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400802:	55                   	push   %rbp
  400803:	48 89 e5             	mov    %rsp,%rbp
  400806:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40080a:	90                   	nop
  40080b:	5d                   	pop    %rbp
  40080c:	c3                   	retq   
  40080d:	90                   	nop

000000000040080e <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  40080e:	55                   	push   %rbp
  40080f:	48 89 e5             	mov    %rsp,%rbp
  400812:	48 83 ec 20          	sub    $0x20,%rsp
  400816:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40081a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40081e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400821:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400825:	48 89 c7             	mov    %rax,%rdi
  400828:	e8 95 00 00 00       	callq  4008c2 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40082d:	48 89 c2             	mov    %rax,%rdx
  400830:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400833:	89 02                	mov    %eax,(%rdx)
  400835:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400839:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40083d:	48 89 d6             	mov    %rdx,%rsi
  400840:	48 89 c7             	mov    %rax,%rdi
  400843:	e8 bc 00 00 00       	callq  400904 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS4_>
  400848:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40084c:	c9                   	leaveq 
  40084d:	c3                   	retq   

000000000040084e <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  40084e:	55                   	push   %rbp
  40084f:	48 89 e5             	mov    %rsp,%rbp
  400852:	48 83 ec 20          	sub    $0x20,%rsp
  400856:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40085a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40085e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400861:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400865:	48 89 c7             	mov    %rax,%rdi
  400868:	e8 a7 00 00 00       	callq  400914 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40086d:	48 89 c2             	mov    %rax,%rdx
  400870:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400873:	89 02                	mov    %eax,(%rdx)
  400875:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400879:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40087d:	48 89 d6             	mov    %rdx,%rsi
  400880:	48 89 c7             	mov    %rax,%rdi
  400883:	e8 d4 00 00 00       	callq  40095c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS4_>
  400888:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40088c:	c9                   	leaveq 
  40088d:	c3                   	retq   

000000000040088e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  40088e:	55                   	push   %rbp
  40088f:	48 89 e5             	mov    %rsp,%rbp
  400892:	48 83 ec 10          	sub    $0x10,%rsp
  400896:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40089a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40089e:	48 89 c7             	mov    %rax,%rdi
  4008a1:	e8 1c 00 00 00       	callq  4008c2 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008a6:	c9                   	leaveq 
  4008a7:	c3                   	retq   

00000000004008a8 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  4008a8:	55                   	push   %rbp
  4008a9:	48 89 e5             	mov    %rsp,%rbp
  4008ac:	48 83 ec 10          	sub    $0x10,%rsp
  4008b0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008b4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b8:	48 89 c7             	mov    %rax,%rdi
  4008bb:	e8 54 00 00 00       	callq  400914 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008c0:	c9                   	leaveq 
  4008c1:	c3                   	retq   

00000000004008c2 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  4008c2:	55                   	push   %rbp
  4008c3:	48 89 e5             	mov    %rsp,%rbp
  4008c6:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4008ca:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4008ce:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4008d2:	48 c7 45 f0 64 20 60 	movq   $0x602064,-0x10(%rbp)
  4008d9:	00 
  4008da:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008de:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  4008e2:	48 83 e8 04          	sub    $0x4,%rax
  4008e6:	48 c1 e8 02          	shr    $0x2,%rax
  4008ea:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4008f1:	00 
  4008f2:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008f6:	48 01 d0             	add    %rdx,%rax
  4008f9:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008fd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400901:	5d                   	pop    %rbp
  400902:	c3                   	retq   
  400903:	90                   	nop

0000000000400904 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS4_>:
  400904:	55                   	push   %rbp
  400905:	48 89 e5             	mov    %rsp,%rbp
  400908:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40090c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400910:	90                   	nop
  400911:	5d                   	pop    %rbp
  400912:	c3                   	retq   
  400913:	90                   	nop

0000000000400914 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400914:	55                   	push   %rbp
  400915:	48 89 e5             	mov    %rsp,%rbp
  400918:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  40091c:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400920:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400924:	48 c7 45 f0 64 20 60 	movq   $0x602064,-0x10(%rbp)
  40092b:	00 
  40092c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400930:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400934:	48 83 e8 04          	sub    $0x4,%rax
  400938:	48 c1 e8 02          	shr    $0x2,%rax
  40093c:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400943:	00 
  400944:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400948:	48 01 d0             	add    %rdx,%rax
  40094b:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400951:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400955:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400959:	5d                   	pop    %rbp
  40095a:	c3                   	retq   
  40095b:	90                   	nop

000000000040095c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS4_>:
  40095c:	55                   	push   %rbp
  40095d:	48 89 e5             	mov    %rsp,%rbp
  400960:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400964:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400968:	90                   	nop
  400969:	5d                   	pop    %rbp
  40096a:	c3                   	retq   
  40096b:	90                   	nop

000000000040096c <_ZN9TestClass15increase_field0Ev>:
  40096c:	55                   	push   %rbp
  40096d:	48 89 e5             	mov    %rsp,%rbp
  400970:	48 83 ec 30          	sub    $0x30,%rsp
  400974:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400978:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40097f:	00 00 
  400981:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400985:	31 c0                	xor    %eax,%eax
  400987:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40098b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40098f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400993:	ba 55 55 00 00       	mov    $0x5555,%edx
  400998:	48 89 ce             	mov    %rcx,%rsi
  40099b:	48 89 c7             	mov    %rax,%rdi
  40099e:	e8 d3 fd ff ff       	callq  400776 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  4009a3:	90                   	nop
  4009a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009af:	00 00 
  4009b1:	74 05                	je     4009b8 <_ZN9TestClass15increase_field0Ev+0x4c>
  4009b3:	e8 a8 fa ff ff       	callq  400460 <__stack_chk_fail@plt>
  4009b8:	c9                   	leaveq 
  4009b9:	c3                   	retq   

00000000004009ba <_ZN9TestClass15increase_field1Ev>:
  4009ba:	55                   	push   %rbp
  4009bb:	48 89 e5             	mov    %rsp,%rbp
  4009be:	48 83 ec 30          	sub    $0x30,%rsp
  4009c2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4009c6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009cd:	00 00 
  4009cf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4009d3:	31 c0                	xor    %eax,%eax
  4009d5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009d9:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4009dd:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4009e1:	ba 44 44 00 00       	mov    $0x4444,%edx
  4009e6:	48 89 ce             	mov    %rcx,%rsi
  4009e9:	48 89 c7             	mov    %rax,%rdi
  4009ec:	e8 c5 fd ff ff       	callq  4007b6 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4009f1:	90                   	nop
  4009f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009f6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009fd:	00 00 
  4009ff:	74 05                	je     400a06 <_ZN9TestClass15increase_field1Ev+0x4c>
  400a01:	e8 5a fa ff ff       	callq  400460 <__stack_chk_fail@plt>
  400a06:	c9                   	leaveq 
  400a07:	c3                   	retq   

0000000000400a08 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEC1Ev>:
  400a08:	55                   	push   %rbp
  400a09:	48 89 e5             	mov    %rsp,%rbp
  400a0c:	48 83 ec 08          	sub    $0x8,%rsp
  400a10:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a18:	48 89 c7             	mov    %rax,%rdi
  400a1b:	e8 76 fb ff ff       	callq  400596 <_ZN4ikra3soa12_GLOBAL__N_110SizeNDummyILm4EEC2Ev>
  400a20:	90                   	nop
  400a21:	c9                   	leaveq 
  400a22:	c3                   	retq   
  400a23:	90                   	nop

0000000000400a24 <_ZN9TestClassC1Ev>:
  400a24:	55                   	push   %rbp
  400a25:	48 89 e5             	mov    %rsp,%rbp
  400a28:	48 83 ec 10          	sub    $0x10,%rsp
  400a2c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a30:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a34:	48 89 c7             	mov    %rax,%rdi
  400a37:	e8 cc ff ff ff       	callq  400a08 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEC1Ev>
  400a3c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a40:	48 83 c0 04          	add    $0x4,%rax
  400a44:	48 89 c7             	mov    %rax,%rdi
  400a47:	e8 aa fd ff ff       	callq  4007f6 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400a4c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a50:	48 83 c0 04          	add    $0x4,%rax
  400a54:	48 89 c7             	mov    %rax,%rdi
  400a57:	e8 a6 fd ff ff       	callq  400802 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400a5c:	90                   	nop
  400a5d:	c9                   	leaveq 
  400a5e:	c3                   	retq   

0000000000400a5f <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>:
  400a5f:	55                   	push   %rbp
  400a60:	48 89 e5             	mov    %rsp,%rbp
  400a63:	48 83 ec 10          	sub    $0x10,%rsp
  400a67:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6b:	e8 26 00 00 00       	callq  400a96 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a70:	8b 05 ea 15 20 00    	mov    0x2015ea(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  400a76:	8d 50 01             	lea    0x1(%rax),%edx
  400a79:	89 15 e1 15 20 00    	mov    %edx,0x2015e1(%rip)        # 602060 <_ZN9TestClass7storageE>
  400a7f:	89 c0                	mov    %eax,%eax
  400a81:	48 89 c7             	mov    %rax,%rdi
  400a84:	e8 14 00 00 00       	callq  400a9d <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE3getEm>
  400a89:	c9                   	leaveq 
  400a8a:	c3                   	retq   

0000000000400a8b <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEdlEPv>:
  400a8b:	55                   	push   %rbp
  400a8c:	48 89 e5             	mov    %rsp,%rbp
  400a8f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a93:	90                   	nop
  400a94:	5d                   	pop    %rbp
  400a95:	c3                   	retq   

0000000000400a96 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400a96:	55                   	push   %rbp
  400a97:	48 89 e5             	mov    %rsp,%rbp
  400a9a:	90                   	nop
  400a9b:	5d                   	pop    %rbp
  400a9c:	c3                   	retq   

0000000000400a9d <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE3getEm>:
  400a9d:	55                   	push   %rbp
  400a9e:	48 89 e5             	mov    %rsp,%rbp
  400aa1:	48 83 ec 10          	sub    $0x10,%rsp
  400aa5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aad:	48 89 c7             	mov    %rax,%rdi
  400ab0:	e8 02 00 00 00       	callq  400ab7 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400ab5:	c9                   	leaveq 
  400ab6:	c3                   	retq   

0000000000400ab7 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400ab7:	55                   	push   %rbp
  400ab8:	48 89 e5             	mov    %rsp,%rbp
  400abb:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400abf:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400ac3:	48 c1 e0 02          	shl    $0x2,%rax
  400ac7:	ba 64 20 60 00       	mov    $0x602064,%edx
  400acc:	48 01 d0             	add    %rdx,%rax
  400acf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400ad3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ad7:	5d                   	pop    %rbp
  400ad8:	c3                   	retq   
  400ad9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ae0 <__libc_csu_init>:
  400ae0:	41 57                	push   %r15
  400ae2:	41 56                	push   %r14
  400ae4:	41 89 ff             	mov    %edi,%r15d
  400ae7:	41 55                	push   %r13
  400ae9:	41 54                	push   %r12
  400aeb:	4c 8d 25 1e 13 20 00 	lea    0x20131e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400af2:	55                   	push   %rbp
  400af3:	48 8d 2d 1e 13 20 00 	lea    0x20131e(%rip),%rbp        # 601e18 <__init_array_end>
  400afa:	53                   	push   %rbx
  400afb:	49 89 f6             	mov    %rsi,%r14
  400afe:	49 89 d5             	mov    %rdx,%r13
  400b01:	4c 29 e5             	sub    %r12,%rbp
  400b04:	48 83 ec 08          	sub    $0x8,%rsp
  400b08:	48 c1 fd 03          	sar    $0x3,%rbp
  400b0c:	e8 17 f9 ff ff       	callq  400428 <_init>
  400b11:	48 85 ed             	test   %rbp,%rbp
  400b14:	74 20                	je     400b36 <__libc_csu_init+0x56>
  400b16:	31 db                	xor    %ebx,%ebx
  400b18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b1f:	00 
  400b20:	4c 89 ea             	mov    %r13,%rdx
  400b23:	4c 89 f6             	mov    %r14,%rsi
  400b26:	44 89 ff             	mov    %r15d,%edi
  400b29:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400b2d:	48 83 c3 01          	add    $0x1,%rbx
  400b31:	48 39 eb             	cmp    %rbp,%rbx
  400b34:	75 ea                	jne    400b20 <__libc_csu_init+0x40>
  400b36:	48 83 c4 08          	add    $0x8,%rsp
  400b3a:	5b                   	pop    %rbx
  400b3b:	5d                   	pop    %rbp
  400b3c:	41 5c                	pop    %r12
  400b3e:	41 5d                	pop    %r13
  400b40:	41 5e                	pop    %r14
  400b42:	41 5f                	pop    %r15
  400b44:	c3                   	retq   
  400b45:	90                   	nop
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 

0000000000400b50 <__libc_csu_fini>:
  400b50:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400b54 <_fini>:
  400b54:	48 83 ec 08          	sub    $0x8,%rsp
  400b58:	48 83 c4 08          	add    $0x8,%rsp
  400b5c:	c3                   	retq   
