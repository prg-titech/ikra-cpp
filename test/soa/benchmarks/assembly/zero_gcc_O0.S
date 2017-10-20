
bin/zero_gcc_O0:     file format elf64-x86-64


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
  4004af:	49 c7 c0 90 0a 40 00 	mov    $0x400a90,%r8
  4004b6:	48 c7 c1 20 0a 40 00 	mov    $0x400a20,%rcx
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

0000000000400596 <_ZN4ikra3soa12_GLOBAL__N_110SizeNDummyILm0EEC2Ev>:
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
  4005b9:	bf 00 00 00 00       	mov    $0x0,%edi
  4005be:	e8 ee 03 00 00       	callq  4009b1 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>
  4005c3:	48 89 c3             	mov    %rax,%rbx
  4005c6:	48 89 df             	mov    %rbx,%rdi
  4005c9:	e8 b0 03 00 00       	callq  40097e <_ZN9TestClassC1Ev>
  4005ce:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005d2:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005d6:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005da:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005df:	48 89 ce             	mov    %rcx,%rsi
  4005e2:	48 89 c7             	mov    %rax,%rdi
  4005e5:	e8 da 01 00 00       	callq  4007c4 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005ea:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005ee:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005f2:	ba 88 88 00 00       	mov    $0x8888,%edx
  4005f7:	48 89 ce             	mov    %rcx,%rsi
  4005fa:	48 89 c7             	mov    %rax,%rdi
  4005fd:	e8 02 02 00 00       	callq  400804 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400602:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400606:	48 89 c7             	mov    %rax,%rdi
  400609:	e8 c0 02 00 00       	callq  4008ce <_ZN9TestClass15increase_field0Ev>
  40060e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400612:	48 89 c7             	mov    %rax,%rdi
  400615:	e8 fe 02 00 00       	callq  400918 <_ZN9TestClass15increase_field1Ev>
  40061a:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40061e:	48 89 c7             	mov    %rax,%rdi
  400621:	e8 38 02 00 00       	callq  40085e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400626:	8b 18                	mov    (%rax),%ebx
  400628:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40062c:	48 89 c7             	mov    %rax,%rdi
  40062f:	e8 10 02 00 00       	callq  400844 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400634:	8b 00                	mov    (%rax),%eax
  400636:	89 da                	mov    %ebx,%edx
  400638:	89 c6                	mov    %eax,%esi
  40063a:	bf b0 0a 40 00       	mov    $0x400ab0,%edi
  40063f:	b8 00 00 00 00       	mov    $0x0,%eax
  400644:	e8 27 fe ff ff       	callq  400470 <printf@plt>
  400649:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40064d:	48 89 c7             	mov    %rax,%rdi
  400650:	e8 ef 01 00 00       	callq  400844 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400655:	8b 00                	mov    (%rax),%eax
  400657:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40065c:	75 15                	jne    400673 <main+0xd2>
  40065e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400662:	48 89 c7             	mov    %rax,%rdi
  400665:	e8 f4 01 00 00       	callq  40085e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40066a:	8b 00                	mov    (%rax),%eax
  40066c:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400671:	74 07                	je     40067a <main+0xd9>
  400673:	b8 01 00 00 00       	mov    $0x1,%eax
  400678:	eb 05                	jmp    40067f <main+0xde>
  40067a:	b8 00 00 00 00       	mov    $0x0,%eax
  40067f:	0f b6 c0             	movzbl %al,%eax
  400682:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  400686:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  40068d:	00 00 
  40068f:	74 05                	je     400696 <main+0xf5>
  400691:	e8 ca fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400696:	48 83 c4 28          	add    $0x28,%rsp
  40069a:	5b                   	pop    %rbx
  40069b:	5d                   	pop    %rbp
  40069c:	c3                   	retq   

000000000040069d <_Z12new_instancev>:
  40069d:	55                   	push   %rbp
  40069e:	48 89 e5             	mov    %rsp,%rbp
  4006a1:	53                   	push   %rbx
  4006a2:	48 83 ec 08          	sub    $0x8,%rsp
  4006a6:	bf 00 00 00 00       	mov    $0x0,%edi
  4006ab:	e8 01 03 00 00       	callq  4009b1 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>
  4006b0:	48 89 c3             	mov    %rax,%rbx
  4006b3:	48 89 df             	mov    %rbx,%rdi
  4006b6:	e8 c3 02 00 00       	callq  40097e <_ZN9TestClassC1Ev>
  4006bb:	48 89 d8             	mov    %rbx,%rax
  4006be:	48 83 c4 08          	add    $0x8,%rsp
  4006c2:	5b                   	pop    %rbx
  4006c3:	5d                   	pop    %rbp
  4006c4:	c3                   	retq   

00000000004006c5 <_Z12write_field0P9TestClass>:
  4006c5:	55                   	push   %rbp
  4006c6:	48 89 e5             	mov    %rsp,%rbp
  4006c9:	48 83 ec 30          	sub    $0x30,%rsp
  4006cd:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4006d1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4006d8:	00 00 
  4006da:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4006de:	31 c0                	xor    %eax,%eax
  4006e0:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4006e4:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4006e8:	ba 77 77 00 00       	mov    $0x7777,%edx
  4006ed:	48 89 ce             	mov    %rcx,%rsi
  4006f0:	48 89 c7             	mov    %rax,%rdi
  4006f3:	e8 cc 00 00 00       	callq  4007c4 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006f8:	90                   	nop
  4006f9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006fd:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400704:	00 00 
  400706:	74 05                	je     40070d <_Z12write_field0P9TestClass+0x48>
  400708:	e8 53 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  40070d:	c9                   	leaveq 
  40070e:	c3                   	retq   

000000000040070f <_Z11read_field0P9TestClass>:
  40070f:	55                   	push   %rbp
  400710:	48 89 e5             	mov    %rsp,%rbp
  400713:	48 83 ec 10          	sub    $0x10,%rsp
  400717:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40071b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40071f:	48 89 c7             	mov    %rax,%rdi
  400722:	e8 1d 01 00 00       	callq  400844 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400727:	8b 00                	mov    (%rax),%eax
  400729:	c9                   	leaveq 
  40072a:	c3                   	retq   
  40072b:	90                   	nop

000000000040072c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  40072c:	55                   	push   %rbp
  40072d:	48 89 e5             	mov    %rsp,%rbp
  400730:	48 83 ec 20          	sub    $0x20,%rsp
  400734:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400738:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40073c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40073f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400743:	48 89 c7             	mov    %rax,%rdi
  400746:	e8 2d 01 00 00       	callq  400878 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40074b:	8b 10                	mov    (%rax),%edx
  40074d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400751:	89 10                	mov    %edx,(%rax)
  400753:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400757:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075b:	48 89 d6             	mov    %rdx,%rsi
  40075e:	48 89 c7             	mov    %rax,%rdi
  400761:	e8 2a 01 00 00       	callq  400890 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS4_>
  400766:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40076a:	c9                   	leaveq 
  40076b:	c3                   	retq   

000000000040076c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  40076c:	55                   	push   %rbp
  40076d:	48 89 e5             	mov    %rsp,%rbp
  400770:	48 83 ec 20          	sub    $0x20,%rsp
  400774:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400778:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40077c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40077f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400783:	48 89 c7             	mov    %rax,%rdi
  400786:	e8 15 01 00 00       	callq  4008a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40078b:	8b 10                	mov    (%rax),%edx
  40078d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400791:	89 10                	mov    %edx,(%rax)
  400793:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400797:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40079b:	48 89 d6             	mov    %rdx,%rsi
  40079e:	48 89 c7             	mov    %rax,%rdi
  4007a1:	e8 18 01 00 00       	callq  4008be <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS4_>
  4007a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007aa:	c9                   	leaveq 
  4007ab:	c3                   	retq   

00000000004007ac <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  4007ac:	55                   	push   %rbp
  4007ad:	48 89 e5             	mov    %rsp,%rbp
  4007b0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b4:	90                   	nop
  4007b5:	5d                   	pop    %rbp
  4007b6:	c3                   	retq   
  4007b7:	90                   	nop

00000000004007b8 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4007b8:	55                   	push   %rbp
  4007b9:	48 89 e5             	mov    %rsp,%rbp
  4007bc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c0:	90                   	nop
  4007c1:	5d                   	pop    %rbp
  4007c2:	c3                   	retq   
  4007c3:	90                   	nop

00000000004007c4 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4007c4:	55                   	push   %rbp
  4007c5:	48 89 e5             	mov    %rsp,%rbp
  4007c8:	48 83 ec 20          	sub    $0x20,%rsp
  4007cc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d0:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007d4:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4007d7:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007db:	48 89 c7             	mov    %rax,%rdi
  4007de:	e8 95 00 00 00       	callq  400878 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4007e3:	48 89 c2             	mov    %rax,%rdx
  4007e6:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4007e9:	89 02                	mov    %eax,(%rdx)
  4007eb:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4007ef:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f3:	48 89 d6             	mov    %rdx,%rsi
  4007f6:	48 89 c7             	mov    %rax,%rdi
  4007f9:	e8 92 00 00 00       	callq  400890 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS4_>
  4007fe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400802:	c9                   	leaveq 
  400803:	c3                   	retq   

0000000000400804 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400804:	55                   	push   %rbp
  400805:	48 89 e5             	mov    %rsp,%rbp
  400808:	48 83 ec 20          	sub    $0x20,%rsp
  40080c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400810:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400814:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400817:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40081b:	48 89 c7             	mov    %rax,%rdi
  40081e:	e8 7d 00 00 00       	callq  4008a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400823:	48 89 c2             	mov    %rax,%rdx
  400826:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400829:	89 02                	mov    %eax,(%rdx)
  40082b:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40082f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400833:	48 89 d6             	mov    %rdx,%rsi
  400836:	48 89 c7             	mov    %rax,%rdi
  400839:	e8 80 00 00 00       	callq  4008be <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS4_>
  40083e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400842:	c9                   	leaveq 
  400843:	c3                   	retq   

0000000000400844 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400844:	55                   	push   %rbp
  400845:	48 89 e5             	mov    %rsp,%rbp
  400848:	48 83 ec 10          	sub    $0x10,%rsp
  40084c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400850:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	e8 1c 00 00 00       	callq  400878 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40085c:	c9                   	leaveq 
  40085d:	c3                   	retq   

000000000040085e <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  40085e:	55                   	push   %rbp
  40085f:	48 89 e5             	mov    %rsp,%rbp
  400862:	48 83 ec 10          	sub    $0x10,%rsp
  400866:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40086a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40086e:	48 89 c7             	mov    %rax,%rdi
  400871:	e8 2a 00 00 00       	callq  4008a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400876:	c9                   	leaveq 
  400877:	c3                   	retq   

0000000000400878 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400878:	55                   	push   %rbp
  400879:	48 89 e5             	mov    %rsp,%rbp
  40087c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400880:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400884:	48 c1 e0 02          	shl    $0x2,%rax
  400888:	48 05 64 20 60 00    	add    $0x602064,%rax
  40088e:	5d                   	pop    %rbp
  40088f:	c3                   	retq   

0000000000400890 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS4_>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400898:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40089c:	90                   	nop
  40089d:	5d                   	pop    %rbp
  40089e:	c3                   	retq   
  40089f:	90                   	nop

00000000004008a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  4008a0:	55                   	push   %rbp
  4008a1:	48 89 e5             	mov    %rsp,%rbp
  4008a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ac:	48 c1 e0 02          	shl    $0x2,%rax
  4008b0:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  4008b6:	48 05 64 20 60 00    	add    $0x602064,%rax
  4008bc:	5d                   	pop    %rbp
  4008bd:	c3                   	retq   

00000000004008be <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS4_>:
  4008be:	55                   	push   %rbp
  4008bf:	48 89 e5             	mov    %rsp,%rbp
  4008c2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008c6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008ca:	90                   	nop
  4008cb:	5d                   	pop    %rbp
  4008cc:	c3                   	retq   
  4008cd:	90                   	nop

00000000004008ce <_ZN9TestClass15increase_field0Ev>:
  4008ce:	55                   	push   %rbp
  4008cf:	48 89 e5             	mov    %rsp,%rbp
  4008d2:	48 83 ec 30          	sub    $0x30,%rsp
  4008d6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008da:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008e1:	00 00 
  4008e3:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008e7:	31 c0                	xor    %eax,%eax
  4008e9:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4008ed:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008f1:	ba 55 55 00 00       	mov    $0x5555,%edx
  4008f6:	48 89 ce             	mov    %rcx,%rsi
  4008f9:	48 89 c7             	mov    %rax,%rdi
  4008fc:	e8 2b fe ff ff       	callq  40072c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400901:	90                   	nop
  400902:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400906:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40090d:	00 00 
  40090f:	74 05                	je     400916 <_ZN9TestClass15increase_field0Ev+0x48>
  400911:	e8 4a fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400916:	c9                   	leaveq 
  400917:	c3                   	retq   

0000000000400918 <_ZN9TestClass15increase_field1Ev>:
  400918:	55                   	push   %rbp
  400919:	48 89 e5             	mov    %rsp,%rbp
  40091c:	48 83 ec 30          	sub    $0x30,%rsp
  400920:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400924:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40092b:	00 00 
  40092d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400931:	31 c0                	xor    %eax,%eax
  400933:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400937:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40093b:	ba 44 44 00 00       	mov    $0x4444,%edx
  400940:	48 89 ce             	mov    %rcx,%rsi
  400943:	48 89 c7             	mov    %rax,%rdi
  400946:	e8 21 fe ff ff       	callq  40076c <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  40094b:	90                   	nop
  40094c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400950:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400957:	00 00 
  400959:	74 05                	je     400960 <_ZN9TestClass15increase_field1Ev+0x48>
  40095b:	e8 00 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400960:	c9                   	leaveq 
  400961:	c3                   	retq   

0000000000400962 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEC1Ev>:
  400962:	55                   	push   %rbp
  400963:	48 89 e5             	mov    %rsp,%rbp
  400966:	48 83 ec 08          	sub    $0x8,%rsp
  40096a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40096e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400972:	48 89 c7             	mov    %rax,%rdi
  400975:	e8 1c fc ff ff       	callq  400596 <_ZN4ikra3soa12_GLOBAL__N_110SizeNDummyILm0EEC2Ev>
  40097a:	90                   	nop
  40097b:	c9                   	leaveq 
  40097c:	c3                   	retq   
  40097d:	90                   	nop

000000000040097e <_ZN9TestClassC1Ev>:
  40097e:	55                   	push   %rbp
  40097f:	48 89 e5             	mov    %rsp,%rbp
  400982:	48 83 ec 10          	sub    $0x10,%rsp
  400986:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40098a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40098e:	48 89 c7             	mov    %rax,%rdi
  400991:	e8 cc ff ff ff       	callq  400962 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEC1Ev>
  400996:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40099a:	48 89 c7             	mov    %rax,%rdi
  40099d:	e8 0a fe ff ff       	callq  4007ac <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4009a2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a6:	48 89 c7             	mov    %rax,%rdi
  4009a9:	e8 0a fe ff ff       	callq  4007b8 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4009ae:	90                   	nop
  4009af:	c9                   	leaveq 
  4009b0:	c3                   	retq   

00000000004009b1 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>:
  4009b1:	55                   	push   %rbp
  4009b2:	48 89 e5             	mov    %rsp,%rbp
  4009b5:	48 83 ec 10          	sub    $0x10,%rsp
  4009b9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bd:	e8 26 00 00 00       	callq  4009e8 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009c2:	8b 05 98 16 20 00    	mov    0x201698(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  4009c8:	8d 50 01             	lea    0x1(%rax),%edx
  4009cb:	89 15 8f 16 20 00    	mov    %edx,0x20168f(%rip)        # 602060 <_ZN9TestClass7storageE>
  4009d1:	89 c0                	mov    %eax,%eax
  4009d3:	48 89 c7             	mov    %rax,%rdi
  4009d6:	e8 14 00 00 00       	callq  4009ef <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE3getEm>
  4009db:	c9                   	leaveq 
  4009dc:	c3                   	retq   

00000000004009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEdlEPv>:
  4009dd:	55                   	push   %rbp
  4009de:	48 89 e5             	mov    %rsp,%rbp
  4009e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e5:	90                   	nop
  4009e6:	5d                   	pop    %rbp
  4009e7:	c3                   	retq   

00000000004009e8 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  4009e8:	55                   	push   %rbp
  4009e9:	48 89 e5             	mov    %rsp,%rbp
  4009ec:	90                   	nop
  4009ed:	5d                   	pop    %rbp
  4009ee:	c3                   	retq   

00000000004009ef <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE3getEm>:
  4009ef:	55                   	push   %rbp
  4009f0:	48 89 e5             	mov    %rsp,%rbp
  4009f3:	48 83 ec 10          	sub    $0x10,%rsp
  4009f7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009ff:	48 89 c7             	mov    %rax,%rdi
  400a02:	e8 02 00 00 00       	callq  400a09 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400a07:	c9                   	leaveq 
  400a08:	c3                   	retq   

0000000000400a09 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400a09:	55                   	push   %rbp
  400a0a:	48 89 e5             	mov    %rsp,%rbp
  400a0d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a11:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a15:	5d                   	pop    %rbp
  400a16:	c3                   	retq   
  400a17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a1e:	00 00 

0000000000400a20 <__libc_csu_init>:
  400a20:	41 57                	push   %r15
  400a22:	41 56                	push   %r14
  400a24:	41 89 ff             	mov    %edi,%r15d
  400a27:	41 55                	push   %r13
  400a29:	41 54                	push   %r12
  400a2b:	4c 8d 25 de 13 20 00 	lea    0x2013de(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400a32:	55                   	push   %rbp
  400a33:	48 8d 2d de 13 20 00 	lea    0x2013de(%rip),%rbp        # 601e18 <__init_array_end>
  400a3a:	53                   	push   %rbx
  400a3b:	49 89 f6             	mov    %rsi,%r14
  400a3e:	49 89 d5             	mov    %rdx,%r13
  400a41:	4c 29 e5             	sub    %r12,%rbp
  400a44:	48 83 ec 08          	sub    $0x8,%rsp
  400a48:	48 c1 fd 03          	sar    $0x3,%rbp
  400a4c:	e8 d7 f9 ff ff       	callq  400428 <_init>
  400a51:	48 85 ed             	test   %rbp,%rbp
  400a54:	74 20                	je     400a76 <__libc_csu_init+0x56>
  400a56:	31 db                	xor    %ebx,%ebx
  400a58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a5f:	00 
  400a60:	4c 89 ea             	mov    %r13,%rdx
  400a63:	4c 89 f6             	mov    %r14,%rsi
  400a66:	44 89 ff             	mov    %r15d,%edi
  400a69:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400a6d:	48 83 c3 01          	add    $0x1,%rbx
  400a71:	48 39 eb             	cmp    %rbp,%rbx
  400a74:	75 ea                	jne    400a60 <__libc_csu_init+0x40>
  400a76:	48 83 c4 08          	add    $0x8,%rsp
  400a7a:	5b                   	pop    %rbx
  400a7b:	5d                   	pop    %rbp
  400a7c:	41 5c                	pop    %r12
  400a7e:	41 5d                	pop    %r13
  400a80:	41 5e                	pop    %r14
  400a82:	41 5f                	pop    %r15
  400a84:	c3                   	retq   
  400a85:	90                   	nop
  400a86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8d:	00 00 00 

0000000000400a90 <__libc_csu_fini>:
  400a90:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400a94 <_fini>:
  400a94:	48 83 ec 08          	sub    $0x8,%rsp
  400a98:	48 83 c4 08          	add    $0x8,%rsp
  400a9c:	c3                   	retq   
