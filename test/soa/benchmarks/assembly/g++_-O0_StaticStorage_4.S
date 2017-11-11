
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
  4004af:	49 c7 c0 80 0c 40 00 	mov    $0x400c80,%r8
  4004b6:	48 c7 c1 10 0c 40 00 	mov    $0x400c10,%rcx
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
  4005aa:	e8 bd 03 00 00       	callq  40096c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005af:	bf 04 00 00 00       	mov    $0x4,%edi
  4005b4:	e8 d9 03 00 00       	callq  400992 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005b9:	48 89 c3             	mov    %rax,%rbx
  4005bc:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c1:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005c5:	48 83 f8 03          	cmp    $0x3,%rax
  4005c9:	74 06                	je     4005d1 <main+0x30>
  4005cb:	48 83 c0 01          	add    $0x1,%rax
  4005cf:	eb f0                	jmp    4005c1 <main+0x20>
  4005d1:	48 89 df             	mov    %rbx,%rdi
  4005d4:	e8 d7 02 00 00       	callq  4008b0 <_ZN9TestClassC1Ev>
  4005d9:	48 89 5d e8          	mov    %rbx,-0x18(%rbp)
  4005dd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005e1:	48 83 c0 04          	add    $0x4,%rax
  4005e5:	be 77 77 00 00       	mov    $0x7777,%esi
  4005ea:	48 89 c7             	mov    %rax,%rdi
  4005ed:	e8 f0 03 00 00       	callq  4009e2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4005f2:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4005f6:	48 83 c0 04          	add    $0x4,%rax
  4005fa:	be 88 88 00 00       	mov    $0x8888,%esi
  4005ff:	48 89 c7             	mov    %rax,%rdi
  400602:	e8 05 04 00 00       	callq  400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400607:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40060b:	48 89 c7             	mov    %rax,%rdi
  40060e:	e8 d9 01 00 00       	callq  4007ec <_ZN9TestClass15increase_field0Ev>
  400613:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400617:	48 89 c7             	mov    %rax,%rdi
  40061a:	e8 1b 02 00 00       	callq  40083a <_ZN9TestClass15increase_field1Ev>
  40061f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400623:	48 83 c0 04          	add    $0x4,%rax
  400627:	48 89 c7             	mov    %rax,%rdi
  40062a:	e8 21 04 00 00       	callq  400a50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40062f:	8b 18                	mov    (%rax),%ebx
  400631:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400635:	48 83 c0 04          	add    $0x4,%rax
  400639:	48 89 c7             	mov    %rax,%rdi
  40063c:	e8 f5 03 00 00       	callq  400a36 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400641:	8b 00                	mov    (%rax),%eax
  400643:	89 da                	mov    %ebx,%edx
  400645:	89 c6                	mov    %eax,%esi
  400647:	bf a4 0c 40 00       	mov    $0x400ca4,%edi
  40064c:	b8 00 00 00 00       	mov    $0x0,%eax
  400651:	e8 1a fe ff ff       	callq  400470 <printf@plt>
  400656:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40065a:	48 83 c0 04          	add    $0x4,%rax
  40065e:	48 89 c7             	mov    %rax,%rdi
  400661:	e8 d0 03 00 00       	callq  400a36 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400666:	8b 00                	mov    (%rax),%eax
  400668:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40066d:	75 19                	jne    400688 <main+0xe7>
  40066f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400673:	48 83 c0 04          	add    $0x4,%rax
  400677:	48 89 c7             	mov    %rax,%rdi
  40067a:	e8 d1 03 00 00       	callq  400a50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40067f:	8b 00                	mov    (%rax),%eax
  400681:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400686:	74 07                	je     40068f <main+0xee>
  400688:	b8 01 00 00 00       	mov    $0x1,%eax
  40068d:	eb 05                	jmp    400694 <main+0xf3>
  40068f:	b8 00 00 00 00       	mov    $0x0,%eax
  400694:	0f b6 c0             	movzbl %al,%eax
  400697:	48 83 c4 18          	add    $0x18,%rsp
  40069b:	5b                   	pop    %rbx
  40069c:	5d                   	pop    %rbp
  40069d:	c3                   	retq   

000000000040069e <_Z12new_instancev>:
  40069e:	55                   	push   %rbp
  40069f:	48 89 e5             	mov    %rsp,%rbp
  4006a2:	53                   	push   %rbx
  4006a3:	48 83 ec 08          	sub    $0x8,%rsp
  4006a7:	bf 04 00 00 00       	mov    $0x4,%edi
  4006ac:	e8 e1 02 00 00       	callq  400992 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006b1:	48 89 c3             	mov    %rax,%rbx
  4006b4:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b9:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006bd:	48 83 f8 03          	cmp    $0x3,%rax
  4006c1:	74 06                	je     4006c9 <_Z12new_instancev+0x2b>
  4006c3:	48 83 c0 01          	add    $0x1,%rax
  4006c7:	eb f0                	jmp    4006b9 <_Z12new_instancev+0x1b>
  4006c9:	48 89 df             	mov    %rbx,%rdi
  4006cc:	e8 df 01 00 00       	callq  4008b0 <_ZN9TestClassC1Ev>
  4006d1:	48 89 d8             	mov    %rbx,%rax
  4006d4:	48 83 c4 08          	add    $0x8,%rsp
  4006d8:	5b                   	pop    %rbx
  4006d9:	5d                   	pop    %rbp
  4006da:	c3                   	retq   

00000000004006db <_Z12write_field0P9TestClass>:
  4006db:	55                   	push   %rbp
  4006dc:	48 89 e5             	mov    %rsp,%rbp
  4006df:	48 83 ec 10          	sub    $0x10,%rsp
  4006e3:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4006e7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4006eb:	48 83 c0 04          	add    $0x4,%rax
  4006ef:	be 77 77 00 00       	mov    $0x7777,%esi
  4006f4:	48 89 c7             	mov    %rax,%rdi
  4006f7:	e8 e6 02 00 00       	callq  4009e2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4006fc:	90                   	nop
  4006fd:	c9                   	leaveq 
  4006fe:	c3                   	retq   

00000000004006ff <_Z11read_field0P9TestClass>:
  4006ff:	55                   	push   %rbp
  400700:	48 89 e5             	mov    %rsp,%rbp
  400703:	48 83 ec 10          	sub    $0x10,%rsp
  400707:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40070b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40070f:	48 83 c0 04          	add    $0x4,%rax
  400713:	48 89 c7             	mov    %rax,%rdi
  400716:	e8 1b 03 00 00       	callq  400a36 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40071b:	8b 00                	mov    (%rax),%eax
  40071d:	c9                   	leaveq 
  40071e:	c3                   	retq   

000000000040071f <_Z12write_field1P9TestClass>:
  40071f:	55                   	push   %rbp
  400720:	48 89 e5             	mov    %rsp,%rbp
  400723:	48 83 ec 10          	sub    $0x10,%rsp
  400727:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40072b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40072f:	48 83 c0 04          	add    $0x4,%rax
  400733:	be 77 77 00 00       	mov    $0x7777,%esi
  400738:	48 89 c7             	mov    %rax,%rdi
  40073b:	e8 cc 02 00 00       	callq  400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400740:	90                   	nop
  400741:	c9                   	leaveq 
  400742:	c3                   	retq   

0000000000400743 <_Z11read_field1P9TestClass>:
  400743:	55                   	push   %rbp
  400744:	48 89 e5             	mov    %rsp,%rbp
  400747:	48 83 ec 10          	sub    $0x10,%rsp
  40074b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40074f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400753:	48 83 c0 04          	add    $0x4,%rax
  400757:	48 89 c7             	mov    %rax,%rdi
  40075a:	e8 f1 02 00 00       	callq  400a50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40075f:	8b 00                	mov    (%rax),%eax
  400761:	c9                   	leaveq 
  400762:	c3                   	retq   

0000000000400763 <_Z21explicit_write_field0m>:
  400763:	55                   	push   %rbp
  400764:	48 89 e5             	mov    %rsp,%rbp
  400767:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40076b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40076f:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  400776:	77 77 00 00 
  40077a:	90                   	nop
  40077b:	5d                   	pop    %rbp
  40077c:	c3                   	retq   

000000000040077d <_Z20explicit_read_field0m>:
  40077d:	55                   	push   %rbp
  40077e:	48 89 e5             	mov    %rsp,%rbp
  400781:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400785:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400789:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  400790:	5d                   	pop    %rbp
  400791:	c3                   	retq   

0000000000400792 <_ZnwmPv>:
  400792:	55                   	push   %rbp
  400793:	48 89 e5             	mov    %rsp,%rbp
  400796:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40079a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40079e:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007a2:	5d                   	pop    %rbp
  4007a3:	c3                   	retq   

00000000004007a4 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  4007a4:	55                   	push   %rbp
  4007a5:	48 89 e5             	mov    %rsp,%rbp
  4007a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ac:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007b0:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  4007b7:	90                   	nop
  4007b8:	5d                   	pop    %rbp
  4007b9:	c3                   	retq   

00000000004007ba <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007ba:	55                   	push   %rbp
  4007bb:	48 89 e5             	mov    %rsp,%rbp
  4007be:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007c2:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  4007c6:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007ca:	48 8b 00             	mov    (%rax),%rax
  4007cd:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007d1:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007d5:	48 8b 10             	mov    (%rax),%rdx
  4007d8:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4007dc:	48 01 c2             	add    %rax,%rdx
  4007df:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007e3:	48 89 10             	mov    %rdx,(%rax)
  4007e6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007ea:	5d                   	pop    %rbp
  4007eb:	c3                   	retq   

00000000004007ec <_ZN9TestClass15increase_field0Ev>:
  4007ec:	55                   	push   %rbp
  4007ed:	48 89 e5             	mov    %rsp,%rbp
  4007f0:	48 83 ec 30          	sub    $0x30,%rsp
  4007f4:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007f8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007ff:	00 00 
  400801:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400805:	31 c0                	xor    %eax,%eax
  400807:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40080b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40080f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400813:	ba 55 55 00 00       	mov    $0x5555,%edx
  400818:	48 89 ce             	mov    %rcx,%rsi
  40081b:	48 89 c7             	mov    %rax,%rdi
  40081e:	e8 c9 00 00 00       	callq  4008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400823:	90                   	nop
  400824:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400828:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40082f:	00 00 
  400831:	74 05                	je     400838 <_ZN9TestClass15increase_field0Ev+0x4c>
  400833:	e8 28 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400838:	c9                   	leaveq 
  400839:	c3                   	retq   

000000000040083a <_ZN9TestClass15increase_field1Ev>:
  40083a:	55                   	push   %rbp
  40083b:	48 89 e5             	mov    %rsp,%rbp
  40083e:	48 83 ec 30          	sub    $0x30,%rsp
  400842:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400846:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40084d:	00 00 
  40084f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400853:	31 c0                	xor    %eax,%eax
  400855:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400859:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40085d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400861:	ba 44 44 00 00       	mov    $0x4444,%edx
  400866:	48 89 ce             	mov    %rcx,%rsi
  400869:	48 89 c7             	mov    %rax,%rdi
  40086c:	e8 bb 00 00 00       	callq  40092c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400871:	90                   	nop
  400872:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400876:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40087d:	00 00 
  40087f:	74 05                	je     400886 <_ZN9TestClass15increase_field1Ev+0x4c>
  400881:	e8 da fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400886:	c9                   	leaveq 
  400887:	c3                   	retq   

0000000000400888 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400888:	55                   	push   %rbp
  400889:	48 89 e5             	mov    %rsp,%rbp
  40088c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400890:	90                   	nop
  400891:	5d                   	pop    %rbp
  400892:	c3                   	retq   
  400893:	90                   	nop

0000000000400894 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  400894:	55                   	push   %rbp
  400895:	48 89 e5             	mov    %rsp,%rbp
  400898:	48 83 ec 10          	sub    $0x10,%rsp
  40089c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a4:	48 89 c7             	mov    %rax,%rdi
  4008a7:	e8 dc ff ff ff       	callq  400888 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ac:	90                   	nop
  4008ad:	c9                   	leaveq 
  4008ae:	c3                   	retq   
  4008af:	90                   	nop

00000000004008b0 <_ZN9TestClassC1Ev>:
  4008b0:	55                   	push   %rbp
  4008b1:	48 89 e5             	mov    %rsp,%rbp
  4008b4:	48 83 ec 10          	sub    $0x10,%rsp
  4008b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008bc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008c0:	48 89 c7             	mov    %rax,%rdi
  4008c3:	e8 cc ff ff ff       	callq  400894 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  4008c8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008cc:	48 83 c0 04          	add    $0x4,%rax
  4008d0:	48 89 c7             	mov    %rax,%rdi
  4008d3:	e8 e8 00 00 00       	callq  4009c0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  4008d8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008dc:	48 83 c0 04          	add    $0x4,%rax
  4008e0:	48 89 c7             	mov    %rax,%rdi
  4008e3:	e8 e4 00 00 00       	callq  4009cc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  4008e8:	90                   	nop
  4008e9:	c9                   	leaveq 
  4008ea:	c3                   	retq   
  4008eb:	90                   	nop

00000000004008ec <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  4008ec:	55                   	push   %rbp
  4008ed:	48 89 e5             	mov    %rsp,%rbp
  4008f0:	48 83 ec 20          	sub    $0x20,%rsp
  4008f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008fc:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008ff:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400903:	48 89 c7             	mov    %rax,%rdi
  400906:	e8 5f 01 00 00       	callq  400a6a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40090b:	8b 10                	mov    (%rax),%edx
  40090d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400911:	89 10                	mov    %edx,(%rax)
  400913:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400917:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091b:	48 89 d6             	mov    %rdx,%rsi
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 5e 01 00 00       	callq  400a84 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400926:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40092a:	c9                   	leaveq 
  40092b:	c3                   	retq   

000000000040092c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  40092c:	55                   	push   %rbp
  40092d:	48 89 e5             	mov    %rsp,%rbp
  400930:	48 83 ec 20          	sub    $0x20,%rsp
  400934:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400938:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40093c:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40093f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400943:	48 89 c7             	mov    %rax,%rdi
  400946:	e8 49 01 00 00       	callq  400a94 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  40094b:	8b 10                	mov    (%rax),%edx
  40094d:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  400951:	89 10                	mov    %edx,(%rax)
  400953:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400957:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40095b:	48 89 d6             	mov    %rdx,%rsi
  40095e:	48 89 c7             	mov    %rax,%rdi
  400961:	e8 48 01 00 00       	callq  400aae <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400966:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096a:	c9                   	leaveq 
  40096b:	c3                   	retq   

000000000040096c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  40096c:	55                   	push   %rbp
  40096d:	48 89 e5             	mov    %rsp,%rbp
  400970:	e8 21 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400975:	48 89 c6             	mov    %rax,%rsi
  400978:	bf b0 91 00 00       	mov    $0x91b0,%edi
  40097d:	e8 10 fe ff ff       	callq  400792 <_ZnwmPv>
  400982:	48 85 c0             	test   %rax,%rax
  400985:	74 08                	je     40098f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400987:	48 89 c7             	mov    %rax,%rdi
  40098a:	e8 4b 01 00 00       	callq  400ada <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40098f:	90                   	nop
  400990:	5d                   	pop    %rbp
  400991:	c3                   	retq   

0000000000400992 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  400992:	55                   	push   %rbp
  400993:	48 89 e5             	mov    %rsp,%rbp
  400996:	48 83 ec 10          	sub    $0x10,%rsp
  40099a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40099e:	e8 5e 01 00 00       	callq  400b01 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009a3:	e8 ee fb ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  4009a8:	be 01 00 00 00       	mov    $0x1,%esi
  4009ad:	48 89 c7             	mov    %rax,%rdi
  4009b0:	e8 05 fe ff ff       	callq  4007ba <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009b5:	48 89 c7             	mov    %rax,%rdi
  4009b8:	e8 4b 01 00 00       	callq  400b08 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009bd:	c9                   	leaveq 
  4009be:	c3                   	retq   
  4009bf:	90                   	nop

00000000004009c0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c8:	90                   	nop
  4009c9:	5d                   	pop    %rbp
  4009ca:	c3                   	retq   
  4009cb:	90                   	nop

00000000004009cc <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  4009cc:	55                   	push   %rbp
  4009cd:	48 89 e5             	mov    %rsp,%rbp
  4009d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d4:	90                   	nop
  4009d5:	5d                   	pop    %rbp
  4009d6:	c3                   	retq   

00000000004009d7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  4009d7:	55                   	push   %rbp
  4009d8:	48 89 e5             	mov    %rsp,%rbp
  4009db:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009df:	90                   	nop
  4009e0:	5d                   	pop    %rbp
  4009e1:	c3                   	retq   

00000000004009e2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  4009e2:	55                   	push   %rbp
  4009e3:	48 89 e5             	mov    %rsp,%rbp
  4009e6:	48 83 ec 10          	sub    $0x10,%rsp
  4009ea:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ee:	89 75 f4             	mov    %esi,-0xc(%rbp)
  4009f1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009f5:	48 89 c7             	mov    %rax,%rdi
  4009f8:	e8 6d 00 00 00       	callq  400a6a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009fd:	48 89 c2             	mov    %rax,%rdx
  400a00:	8b 45 f4             	mov    -0xc(%rbp),%eax
  400a03:	89 02                	mov    %eax,(%rdx)
  400a05:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a09:	c9                   	leaveq 
  400a0a:	c3                   	retq   
  400a0b:	90                   	nop

0000000000400a0c <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a0c:	55                   	push   %rbp
  400a0d:	48 89 e5             	mov    %rsp,%rbp
  400a10:	48 83 ec 10          	sub    $0x10,%rsp
  400a14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a18:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a1b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a1f:	48 89 c7             	mov    %rax,%rdi
  400a22:	e8 6d 00 00 00       	callq  400a94 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a27:	48 89 c2             	mov    %rax,%rdx
  400a2a:	8b 45 f4             	mov    -0xc(%rbp),%eax
  400a2d:	89 02                	mov    %eax,(%rdx)
  400a2f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a33:	c9                   	leaveq 
  400a34:	c3                   	retq   
  400a35:	90                   	nop

0000000000400a36 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400a36:	55                   	push   %rbp
  400a37:	48 89 e5             	mov    %rsp,%rbp
  400a3a:	48 83 ec 10          	sub    $0x10,%rsp
  400a3e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a42:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a46:	48 89 c7             	mov    %rax,%rdi
  400a49:	e8 1c 00 00 00       	callq  400a6a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a4e:	c9                   	leaveq 
  400a4f:	c3                   	retq   

0000000000400a50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 83 ec 10          	sub    $0x10,%rsp
  400a58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a5c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a60:	48 89 c7             	mov    %rax,%rdi
  400a63:	e8 2c 00 00 00       	callq  400a94 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a68:	c9                   	leaveq 
  400a69:	c3                   	retq   

0000000000400a6a <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a6a:	55                   	push   %rbp
  400a6b:	48 89 e5             	mov    %rsp,%rbp
  400a6e:	48 83 ec 10          	sub    $0x10,%rsp
  400a72:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a76:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a7a:	48 89 c7             	mov    %rax,%rdi
  400a7d:	e8 a4 00 00 00       	callq  400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a82:	c9                   	leaveq 
  400a83:	c3                   	retq   

0000000000400a84 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400a84:	55                   	push   %rbp
  400a85:	48 89 e5             	mov    %rsp,%rbp
  400a88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a90:	90                   	nop
  400a91:	5d                   	pop    %rbp
  400a92:	c3                   	retq   
  400a93:	90                   	nop

0000000000400a94 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a94:	55                   	push   %rbp
  400a95:	48 89 e5             	mov    %rsp,%rbp
  400a98:	48 83 ec 10          	sub    $0x10,%rsp
  400a9c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa4:	48 89 c7             	mov    %rax,%rdi
  400aa7:	e8 c8 00 00 00       	callq  400b74 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aac:	c9                   	leaveq 
  400aad:	c3                   	retq   

0000000000400aae <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400aae:	55                   	push   %rbp
  400aaf:	48 89 e5             	mov    %rsp,%rbp
  400ab2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aba:	90                   	nop
  400abb:	5d                   	pop    %rbp
  400abc:	c3                   	retq   
  400abd:	90                   	nop

0000000000400abe <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400abe:	55                   	push   %rbp
  400abf:	48 89 e5             	mov    %rsp,%rbp
  400ac2:	48 83 ec 10          	sub    $0x10,%rsp
  400ac6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aca:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ace:	48 89 c7             	mov    %rax,%rdi
  400ad1:	e8 ce fc ff ff       	callq  4007a4 <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400ad6:	90                   	nop
  400ad7:	c9                   	leaveq 
  400ad8:	c3                   	retq   
  400ad9:	90                   	nop

0000000000400ada <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400ada:	55                   	push   %rbp
  400adb:	48 89 e5             	mov    %rsp,%rbp
  400ade:	48 83 ec 10          	sub    $0x10,%rsp
  400ae2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aea:	48 89 c7             	mov    %rax,%rdi
  400aed:	e8 cc ff ff ff       	callq  400abe <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400af2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af6:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400afd:	00 
  400afe:	90                   	nop
  400aff:	c9                   	leaveq 
  400b00:	c3                   	retq   

0000000000400b01 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b01:	55                   	push   %rbp
  400b02:	48 89 e5             	mov    %rsp,%rbp
  400b05:	90                   	nop
  400b06:	5d                   	pop    %rbp
  400b07:	c3                   	retq   

0000000000400b08 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b08:	55                   	push   %rbp
  400b09:	48 89 e5             	mov    %rsp,%rbp
  400b0c:	48 83 ec 10          	sub    $0x10,%rsp
  400b10:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b18:	48 83 c0 01          	add    $0x1,%rax
  400b1c:	48 89 c7             	mov    %rax,%rdi
  400b1f:	e8 a4 00 00 00       	callq  400bc8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b24:	c9                   	leaveq 
  400b25:	c3                   	retq   

0000000000400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b26:	55                   	push   %rbp
  400b27:	48 89 e5             	mov    %rsp,%rbp
  400b2a:	48 83 ec 30          	sub    $0x30,%rsp
  400b2e:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b32:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b36:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b3a:	e8 57 fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b3f:	48 89 c7             	mov    %rax,%rdi
  400b42:	e8 b3 00 00 00       	callq  400bfa <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400b47:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b4b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b4f:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b53:	48 83 e8 04          	sub    $0x4,%rax
  400b57:	48 c1 e8 02          	shr    $0x2,%rax
  400b5b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b62:	00 
  400b63:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b67:	48 01 d0             	add    %rdx,%rax
  400b6a:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b6e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b72:	c9                   	leaveq 
  400b73:	c3                   	retq   

0000000000400b74 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b74:	55                   	push   %rbp
  400b75:	48 89 e5             	mov    %rsp,%rbp
  400b78:	48 83 ec 30          	sub    $0x30,%rsp
  400b7c:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b80:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b84:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b88:	e8 09 fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b8d:	48 89 c7             	mov    %rax,%rdi
  400b90:	e8 65 00 00 00       	callq  400bfa <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400b95:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b99:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b9d:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400ba1:	48 83 e8 04          	sub    $0x4,%rax
  400ba5:	48 c1 e8 02          	shr    $0x2,%rax
  400ba9:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bb0:	00 
  400bb1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bb5:	48 01 d0             	add    %rdx,%rax
  400bb8:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bbe:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bc2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bc6:	c9                   	leaveq 
  400bc7:	c3                   	retq   

0000000000400bc8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bc8:	55                   	push   %rbp
  400bc9:	48 89 e5             	mov    %rsp,%rbp
  400bcc:	48 83 ec 20          	sub    $0x20,%rsp
  400bd0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bd4:	e8 bd f9 ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400bd9:	48 89 c7             	mov    %rax,%rdi
  400bdc:	e8 19 00 00 00       	callq  400bfa <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400be1:	48 89 c2             	mov    %rax,%rdx
  400be4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400be8:	48 c1 e0 02          	shl    $0x2,%rax
  400bec:	48 01 d0             	add    %rdx,%rax
  400bef:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bf3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bf7:	c9                   	leaveq 
  400bf8:	c3                   	retq   
  400bf9:	90                   	nop

0000000000400bfa <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400bfa:	55                   	push   %rbp
  400bfb:	48 89 e5             	mov    %rsp,%rbp
  400bfe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c02:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c06:	48 83 c0 10          	add    $0x10,%rax
  400c0a:	5d                   	pop    %rbp
  400c0b:	c3                   	retq   
  400c0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c10 <__libc_csu_init>:
  400c10:	41 57                	push   %r15
  400c12:	41 56                	push   %r14
  400c14:	41 89 ff             	mov    %edi,%r15d
  400c17:	41 55                	push   %r13
  400c19:	41 54                	push   %r12
  400c1b:	4c 8d 25 ee 11 20 00 	lea    0x2011ee(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c22:	55                   	push   %rbp
  400c23:	48 8d 2d ee 11 20 00 	lea    0x2011ee(%rip),%rbp        # 601e18 <__init_array_end>
  400c2a:	53                   	push   %rbx
  400c2b:	49 89 f6             	mov    %rsi,%r14
  400c2e:	49 89 d5             	mov    %rdx,%r13
  400c31:	4c 29 e5             	sub    %r12,%rbp
  400c34:	48 83 ec 08          	sub    $0x8,%rsp
  400c38:	48 c1 fd 03          	sar    $0x3,%rbp
  400c3c:	e8 e7 f7 ff ff       	callq  400428 <_init>
  400c41:	48 85 ed             	test   %rbp,%rbp
  400c44:	74 20                	je     400c66 <__libc_csu_init+0x56>
  400c46:	31 db                	xor    %ebx,%ebx
  400c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c4f:	00 
  400c50:	4c 89 ea             	mov    %r13,%rdx
  400c53:	4c 89 f6             	mov    %r14,%rsi
  400c56:	44 89 ff             	mov    %r15d,%edi
  400c59:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c5d:	48 83 c3 01          	add    $0x1,%rbx
  400c61:	48 39 eb             	cmp    %rbp,%rbx
  400c64:	75 ea                	jne    400c50 <__libc_csu_init+0x40>
  400c66:	48 83 c4 08          	add    $0x8,%rsp
  400c6a:	5b                   	pop    %rbx
  400c6b:	5d                   	pop    %rbp
  400c6c:	41 5c                	pop    %r12
  400c6e:	41 5d                	pop    %r13
  400c70:	41 5e                	pop    %r14
  400c72:	41 5f                	pop    %r15
  400c74:	c3                   	retq   
  400c75:	90                   	nop
  400c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7d:	00 00 00 

0000000000400c80 <__libc_csu_fini>:
  400c80:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c84 <_fini>:
  400c84:	48 83 ec 08          	sub    $0x8,%rsp
  400c88:	48 83 c4 08          	add    $0x8,%rsp
  400c8c:	c3                   	retq   
