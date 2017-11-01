
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
  4004af:	49 c7 c0 b0 0c 40 00 	mov    $0x400cb0,%r8
  4004b6:	48 c7 c1 40 0c 40 00 	mov    $0x400c40,%rcx
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
  4005ae:	e8 88 02 00 00       	callq  40083b <_ZN9TestClass18initialize_storageEv>
  4005b3:	bf 04 00 00 00       	mov    $0x4,%edi
  4005b8:	e8 3b 04 00 00       	callq  4009f8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005bd:	48 89 c3             	mov    %rax,%rbx
  4005c0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c5:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005c9:	48 83 f8 03          	cmp    $0x3,%rax
  4005cd:	74 06                	je     4005d5 <main+0x3f>
  4005cf:	48 83 c0 01          	add    $0x1,%rax
  4005d3:	eb f0                	jmp    4005c5 <main+0x2f>
  4005d5:	48 89 df             	mov    %rbx,%rdi
  4005d8:	e8 45 03 00 00       	callq  400922 <_ZN9TestClassC1Ev>
  4005dd:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005e1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005e5:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4005e9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005ed:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005f2:	48 89 ce             	mov    %rcx,%rsi
  4005f5:	48 89 c7             	mov    %rax,%rdi
  4005f8:	e8 4b 04 00 00       	callq  400a48 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4005fd:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400601:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400605:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400609:	ba 88 88 00 00       	mov    $0x8888,%edx
  40060e:	48 89 ce             	mov    %rcx,%rsi
  400611:	48 89 c7             	mov    %rax,%rdi
  400614:	e8 6f 04 00 00       	callq  400a88 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400619:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40061d:	48 89 c7             	mov    %rax,%rdi
  400620:	e8 39 02 00 00       	callq  40085e <_ZN9TestClass15increase_field0Ev>
  400625:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400629:	48 89 c7             	mov    %rax,%rdi
  40062c:	e8 7b 02 00 00       	callq  4008ac <_ZN9TestClass15increase_field1Ev>
  400631:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400635:	48 83 c0 04          	add    $0x4,%rax
  400639:	48 89 c7             	mov    %rax,%rdi
  40063c:	e8 a1 04 00 00       	callq  400ae2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400641:	8b 18                	mov    (%rax),%ebx
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 83 c0 04          	add    $0x4,%rax
  40064b:	48 89 c7             	mov    %rax,%rdi
  40064e:	e8 75 04 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400653:	8b 00                	mov    (%rax),%eax
  400655:	89 da                	mov    %ebx,%edx
  400657:	89 c6                	mov    %eax,%esi
  400659:	bf d4 0c 40 00       	mov    $0x400cd4,%edi
  40065e:	b8 00 00 00 00       	mov    $0x0,%eax
  400663:	e8 08 fe ff ff       	callq  400470 <printf@plt>
  400668:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40066c:	48 83 c0 04          	add    $0x4,%rax
  400670:	48 89 c7             	mov    %rax,%rdi
  400673:	e8 50 04 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400678:	8b 00                	mov    (%rax),%eax
  40067a:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40067f:	75 19                	jne    40069a <main+0x104>
  400681:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400685:	48 83 c0 04          	add    $0x4,%rax
  400689:	48 89 c7             	mov    %rax,%rdi
  40068c:	e8 51 04 00 00       	callq  400ae2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400691:	8b 00                	mov    (%rax),%eax
  400693:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400698:	74 07                	je     4006a1 <main+0x10b>
  40069a:	b8 01 00 00 00       	mov    $0x1,%eax
  40069f:	eb 05                	jmp    4006a6 <main+0x110>
  4006a1:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a6:	0f b6 c0             	movzbl %al,%eax
  4006a9:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006ad:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006b4:	00 00 
  4006b6:	74 05                	je     4006bd <main+0x127>
  4006b8:	e8 a3 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  4006bd:	48 83 c4 28          	add    $0x28,%rsp
  4006c1:	5b                   	pop    %rbx
  4006c2:	5d                   	pop    %rbp
  4006c3:	c3                   	retq   

00000000004006c4 <_Z12new_instancev>:
  4006c4:	55                   	push   %rbp
  4006c5:	48 89 e5             	mov    %rsp,%rbp
  4006c8:	53                   	push   %rbx
  4006c9:	48 83 ec 08          	sub    $0x8,%rsp
  4006cd:	bf 04 00 00 00       	mov    $0x4,%edi
  4006d2:	e8 21 03 00 00       	callq  4009f8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006d7:	48 89 c3             	mov    %rax,%rbx
  4006da:	b8 00 00 00 00       	mov    $0x0,%eax
  4006df:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006e3:	48 83 f8 03          	cmp    $0x3,%rax
  4006e7:	74 06                	je     4006ef <_Z12new_instancev+0x2b>
  4006e9:	48 83 c0 01          	add    $0x1,%rax
  4006ed:	eb f0                	jmp    4006df <_Z12new_instancev+0x1b>
  4006ef:	48 89 df             	mov    %rbx,%rdi
  4006f2:	e8 2b 02 00 00       	callq  400922 <_ZN9TestClassC1Ev>
  4006f7:	48 89 d8             	mov    %rbx,%rax
  4006fa:	48 83 c4 08          	add    $0x8,%rsp
  4006fe:	5b                   	pop    %rbx
  4006ff:	5d                   	pop    %rbp
  400700:	c3                   	retq   

0000000000400701 <_Z12write_field0P9TestClass>:
  400701:	55                   	push   %rbp
  400702:	48 89 e5             	mov    %rsp,%rbp
  400705:	48 83 ec 30          	sub    $0x30,%rsp
  400709:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40070d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400714:	00 00 
  400716:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40071a:	31 c0                	xor    %eax,%eax
  40071c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400720:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400724:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400728:	ba 77 77 00 00       	mov    $0x7777,%edx
  40072d:	48 89 ce             	mov    %rcx,%rsi
  400730:	48 89 c7             	mov    %rax,%rdi
  400733:	e8 10 03 00 00       	callq  400a48 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400738:	90                   	nop
  400739:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40073d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400744:	00 00 
  400746:	74 05                	je     40074d <_Z12write_field0P9TestClass+0x4c>
  400748:	e8 13 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  40074d:	c9                   	leaveq 
  40074e:	c3                   	retq   

000000000040074f <_Z11read_field0P9TestClass>:
  40074f:	55                   	push   %rbp
  400750:	48 89 e5             	mov    %rsp,%rbp
  400753:	48 83 ec 10          	sub    $0x10,%rsp
  400757:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40075b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075f:	48 83 c0 04          	add    $0x4,%rax
  400763:	48 89 c7             	mov    %rax,%rdi
  400766:	e8 5d 03 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40076b:	8b 00                	mov    (%rax),%eax
  40076d:	c9                   	leaveq 
  40076e:	c3                   	retq   

000000000040076f <_Z12write_field1P9TestClass>:
  40076f:	55                   	push   %rbp
  400770:	48 89 e5             	mov    %rsp,%rbp
  400773:	48 83 ec 30          	sub    $0x30,%rsp
  400777:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40077b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400782:	00 00 
  400784:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400788:	31 c0                	xor    %eax,%eax
  40078a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40078e:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400792:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400796:	ba 77 77 00 00       	mov    $0x7777,%edx
  40079b:	48 89 ce             	mov    %rcx,%rsi
  40079e:	48 89 c7             	mov    %rax,%rdi
  4007a1:	e8 e2 02 00 00       	callq  400a88 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007a6:	90                   	nop
  4007a7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007ab:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007b2:	00 00 
  4007b4:	74 05                	je     4007bb <_Z12write_field1P9TestClass+0x4c>
  4007b6:	e8 a5 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007bb:	c9                   	leaveq 
  4007bc:	c3                   	retq   

00000000004007bd <_Z11read_field1P9TestClass>:
  4007bd:	55                   	push   %rbp
  4007be:	48 89 e5             	mov    %rsp,%rbp
  4007c1:	48 83 ec 10          	sub    $0x10,%rsp
  4007c5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007cd:	48 83 c0 04          	add    $0x4,%rax
  4007d1:	48 89 c7             	mov    %rax,%rdi
  4007d4:	e8 09 03 00 00       	callq  400ae2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007d9:	8b 00                	mov    (%rax),%eax
  4007db:	c9                   	leaveq 
  4007dc:	c3                   	retq   

00000000004007dd <_Z21explicit_write_field0m>:
  4007dd:	55                   	push   %rbp
  4007de:	48 89 e5             	mov    %rsp,%rbp
  4007e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007e5:	48 8b 05 14 9f 21 00 	mov    0x219f14(%rip),%rax        # 61a700 <explicit_field0>
  4007ec:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007f0:	48 c1 e2 02          	shl    $0x2,%rdx
  4007f4:	48 01 d0             	add    %rdx,%rax
  4007f7:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007fd:	90                   	nop
  4007fe:	5d                   	pop    %rbp
  4007ff:	c3                   	retq   

0000000000400800 <_Z20explicit_read_field0m>:
  400800:	55                   	push   %rbp
  400801:	48 89 e5             	mov    %rsp,%rbp
  400804:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400808:	48 8b 05 f1 9e 21 00 	mov    0x219ef1(%rip),%rax        # 61a700 <explicit_field0>
  40080f:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  400813:	48 c1 e2 02          	shl    $0x2,%rdx
  400817:	48 01 d0             	add    %rdx,%rax
  40081a:	8b 00                	mov    (%rax),%eax
  40081c:	5d                   	pop    %rbp
  40081d:	c3                   	retq   

000000000040081e <_ZnwmPv>:
  40081e:	55                   	push   %rbp
  40081f:	48 89 e5             	mov    %rsp,%rbp
  400822:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400826:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40082a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40082e:	5d                   	pop    %rbp
  40082f:	c3                   	retq   

0000000000400830 <_ZN9TestClass7storageEv>:
  400830:	55                   	push   %rbp
  400831:	48 89 e5             	mov    %rsp,%rbp
  400834:	b8 60 20 60 00       	mov    $0x602060,%eax
  400839:	5d                   	pop    %rbp
  40083a:	c3                   	retq   

000000000040083b <_ZN9TestClass18initialize_storageEv>:
  40083b:	55                   	push   %rbp
  40083c:	48 89 e5             	mov    %rsp,%rbp
  40083f:	be 60 20 60 00       	mov    $0x602060,%esi
  400844:	bf b0 91 00 00       	mov    $0x91b0,%edi
  400849:	e8 d0 ff ff ff       	callq  40081e <_ZnwmPv>
  40084e:	48 85 c0             	test   %rax,%rax
  400851:	74 08                	je     40085b <_ZN9TestClass18initialize_storageEv+0x20>
  400853:	48 89 c7             	mov    %rax,%rdi
  400856:	e8 03 01 00 00       	callq  40095e <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40085b:	90                   	nop
  40085c:	5d                   	pop    %rbp
  40085d:	c3                   	retq   

000000000040085e <_ZN9TestClass15increase_field0Ev>:
  40085e:	55                   	push   %rbp
  40085f:	48 89 e5             	mov    %rsp,%rbp
  400862:	48 83 ec 30          	sub    $0x30,%rsp
  400866:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40086a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400871:	00 00 
  400873:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400877:	31 c0                	xor    %eax,%eax
  400879:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40087d:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400881:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400885:	ba 55 55 00 00       	mov    $0x5555,%edx
  40088a:	48 89 ce             	mov    %rcx,%rsi
  40088d:	48 89 c7             	mov    %rax,%rdi
  400890:	e8 e3 00 00 00       	callq  400978 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400895:	90                   	nop
  400896:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40089a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008a1:	00 00 
  4008a3:	74 05                	je     4008aa <_ZN9TestClass15increase_field0Ev+0x4c>
  4008a5:	e8 b6 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  4008aa:	c9                   	leaveq 
  4008ab:	c3                   	retq   

00000000004008ac <_ZN9TestClass15increase_field1Ev>:
  4008ac:	55                   	push   %rbp
  4008ad:	48 89 e5             	mov    %rsp,%rbp
  4008b0:	48 83 ec 30          	sub    $0x30,%rsp
  4008b4:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008b8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008bf:	00 00 
  4008c1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008c5:	31 c0                	xor    %eax,%eax
  4008c7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008cb:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008cf:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008d3:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008d8:	48 89 ce             	mov    %rcx,%rsi
  4008db:	48 89 c7             	mov    %rax,%rdi
  4008de:	e8 d5 00 00 00       	callq  4009b8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008e3:	90                   	nop
  4008e4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008ef:	00 00 
  4008f1:	74 05                	je     4008f8 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008f3:	e8 68 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  4008f8:	c9                   	leaveq 
  4008f9:	c3                   	retq   

00000000004008fa <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008fa:	55                   	push   %rbp
  4008fb:	48 89 e5             	mov    %rsp,%rbp
  4008fe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400902:	90                   	nop
  400903:	5d                   	pop    %rbp
  400904:	c3                   	retq   
  400905:	90                   	nop

0000000000400906 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  400906:	55                   	push   %rbp
  400907:	48 89 e5             	mov    %rsp,%rbp
  40090a:	48 83 ec 10          	sub    $0x10,%rsp
  40090e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400912:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400916:	48 89 c7             	mov    %rax,%rdi
  400919:	e8 dc ff ff ff       	callq  4008fa <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40091e:	90                   	nop
  40091f:	c9                   	leaveq 
  400920:	c3                   	retq   
  400921:	90                   	nop

0000000000400922 <_ZN9TestClassC1Ev>:
  400922:	55                   	push   %rbp
  400923:	48 89 e5             	mov    %rsp,%rbp
  400926:	48 83 ec 10          	sub    $0x10,%rsp
  40092a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40092e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400932:	48 89 c7             	mov    %rax,%rdi
  400935:	e8 cc ff ff ff       	callq  400906 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  40093a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40093e:	48 83 c0 04          	add    $0x4,%rax
  400942:	48 89 c7             	mov    %rax,%rdi
  400945:	e8 dc 00 00 00       	callq  400a26 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  40094a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40094e:	48 83 c0 04          	add    $0x4,%rax
  400952:	48 89 c7             	mov    %rax,%rdi
  400955:	e8 d8 00 00 00       	callq  400a32 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  40095a:	90                   	nop
  40095b:	c9                   	leaveq 
  40095c:	c3                   	retq   
  40095d:	90                   	nop

000000000040095e <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  40095e:	55                   	push   %rbp
  40095f:	48 89 e5             	mov    %rsp,%rbp
  400962:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400966:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40096a:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  400971:	00 00 00 00 
  400975:	90                   	nop
  400976:	5d                   	pop    %rbp
  400977:	c3                   	retq   

0000000000400978 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400978:	55                   	push   %rbp
  400979:	48 89 e5             	mov    %rsp,%rbp
  40097c:	48 83 ec 20          	sub    $0x20,%rsp
  400980:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400984:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400988:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40098b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40098f:	48 89 c7             	mov    %rax,%rdi
  400992:	e8 65 01 00 00       	callq  400afc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400997:	8b 10                	mov    (%rax),%edx
  400999:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40099d:	89 10                	mov    %edx,(%rax)
  40099f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009a3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a7:	48 89 d6             	mov    %rdx,%rsi
  4009aa:	48 89 c7             	mov    %rax,%rdi
  4009ad:	e8 64 01 00 00       	callq  400b16 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  4009b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009b6:	c9                   	leaveq 
  4009b7:	c3                   	retq   

00000000004009b8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  4009b8:	55                   	push   %rbp
  4009b9:	48 89 e5             	mov    %rsp,%rbp
  4009bc:	48 83 ec 20          	sub    $0x20,%rsp
  4009c0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009c8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009cb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009cf:	48 89 c7             	mov    %rax,%rdi
  4009d2:	e8 4f 01 00 00       	callq  400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009d7:	8b 10                	mov    (%rax),%edx
  4009d9:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009dd:	89 10                	mov    %edx,(%rax)
  4009df:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009e3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e7:	48 89 d6             	mov    %rdx,%rsi
  4009ea:	48 89 c7             	mov    %rax,%rdi
  4009ed:	e8 4e 01 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009f2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009f6:	c9                   	leaveq 
  4009f7:	c3                   	retq   

00000000004009f8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009f8:	55                   	push   %rbp
  4009f9:	48 89 e5             	mov    %rsp,%rbp
  4009fc:	48 83 ec 10          	sub    $0x10,%rsp
  400a00:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a04:	e8 46 01 00 00       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a09:	e8 22 fe ff ff       	callq  400830 <_ZN9TestClass7storageEv>
  400a0e:	48 89 c2             	mov    %rax,%rdx
  400a11:	48 8b 02             	mov    (%rdx),%rax
  400a14:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a18:	48 89 0a             	mov    %rcx,(%rdx)
  400a1b:	48 89 c7             	mov    %rax,%rdi
  400a1e:	e8 33 01 00 00       	callq  400b56 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400a23:	c9                   	leaveq 
  400a24:	c3                   	retq   
  400a25:	90                   	nop

0000000000400a26 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a26:	55                   	push   %rbp
  400a27:	48 89 e5             	mov    %rsp,%rbp
  400a2a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2e:	90                   	nop
  400a2f:	5d                   	pop    %rbp
  400a30:	c3                   	retq   
  400a31:	90                   	nop

0000000000400a32 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a32:	55                   	push   %rbp
  400a33:	48 89 e5             	mov    %rsp,%rbp
  400a36:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3a:	90                   	nop
  400a3b:	5d                   	pop    %rbp
  400a3c:	c3                   	retq   

0000000000400a3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a3d:	55                   	push   %rbp
  400a3e:	48 89 e5             	mov    %rsp,%rbp
  400a41:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a45:	90                   	nop
  400a46:	5d                   	pop    %rbp
  400a47:	c3                   	retq   

0000000000400a48 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a48:	55                   	push   %rbp
  400a49:	48 89 e5             	mov    %rsp,%rbp
  400a4c:	48 83 ec 20          	sub    $0x20,%rsp
  400a50:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a54:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a58:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a5b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a5f:	48 89 c7             	mov    %rax,%rdi
  400a62:	e8 95 00 00 00       	callq  400afc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a67:	48 89 c2             	mov    %rax,%rdx
  400a6a:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a6d:	89 02                	mov    %eax,(%rdx)
  400a6f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a73:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a77:	48 89 d6             	mov    %rdx,%rsi
  400a7a:	48 89 c7             	mov    %rax,%rdi
  400a7d:	e8 94 00 00 00       	callq  400b16 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a82:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a86:	c9                   	leaveq 
  400a87:	c3                   	retq   

0000000000400a88 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a88:	55                   	push   %rbp
  400a89:	48 89 e5             	mov    %rsp,%rbp
  400a8c:	48 83 ec 20          	sub    $0x20,%rsp
  400a90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a94:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a98:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a9b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a9f:	48 89 c7             	mov    %rax,%rdi
  400aa2:	e8 7f 00 00 00       	callq  400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa7:	48 89 c2             	mov    %rax,%rdx
  400aaa:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400aad:	89 02                	mov    %eax,(%rdx)
  400aaf:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400ab3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab7:	48 89 d6             	mov    %rdx,%rsi
  400aba:	48 89 c7             	mov    %rax,%rdi
  400abd:	e8 7e 00 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400ac2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac6:	c9                   	leaveq 
  400ac7:	c3                   	retq   

0000000000400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400ac8:	55                   	push   %rbp
  400ac9:	48 89 e5             	mov    %rsp,%rbp
  400acc:	48 83 ec 10          	sub    $0x10,%rsp
  400ad0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ad8:	48 89 c7             	mov    %rax,%rdi
  400adb:	e8 1c 00 00 00       	callq  400afc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae0:	c9                   	leaveq 
  400ae1:	c3                   	retq   

0000000000400ae2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400ae2:	55                   	push   %rbp
  400ae3:	48 89 e5             	mov    %rsp,%rbp
  400ae6:	48 83 ec 10          	sub    $0x10,%rsp
  400aea:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aee:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af2:	48 89 c7             	mov    %rax,%rdi
  400af5:	e8 2c 00 00 00       	callq  400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400afa:	c9                   	leaveq 
  400afb:	c3                   	retq   

0000000000400afc <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400afc:	55                   	push   %rbp
  400afd:	48 89 e5             	mov    %rsp,%rbp
  400b00:	48 83 ec 10          	sub    $0x10,%rsp
  400b04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b08:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b0c:	48 89 c7             	mov    %rax,%rdi
  400b0f:	e8 60 00 00 00       	callq  400b74 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b14:	c9                   	leaveq 
  400b15:	c3                   	retq   

0000000000400b16 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b16:	55                   	push   %rbp
  400b17:	48 89 e5             	mov    %rsp,%rbp
  400b1a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b1e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b22:	90                   	nop
  400b23:	5d                   	pop    %rbp
  400b24:	c3                   	retq   
  400b25:	90                   	nop

0000000000400b26 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b26:	55                   	push   %rbp
  400b27:	48 89 e5             	mov    %rsp,%rbp
  400b2a:	48 83 ec 10          	sub    $0x10,%rsp
  400b2e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b32:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b36:	48 89 c7             	mov    %rax,%rdi
  400b39:	e8 80 00 00 00       	callq  400bbe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b3e:	c9                   	leaveq 
  400b3f:	c3                   	retq   

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b48:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b4c:	90                   	nop
  400b4d:	5d                   	pop    %rbp
  400b4e:	c3                   	retq   

0000000000400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b4f:	55                   	push   %rbp
  400b50:	48 89 e5             	mov    %rsp,%rbp
  400b53:	90                   	nop
  400b54:	5d                   	pop    %rbp
  400b55:	c3                   	retq   

0000000000400b56 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b56:	55                   	push   %rbp
  400b57:	48 89 e5             	mov    %rsp,%rbp
  400b5a:	48 83 ec 10          	sub    $0x10,%rsp
  400b5e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b62:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b66:	48 83 c0 01          	add    $0x1,%rax
  400b6a:	48 89 c7             	mov    %rax,%rdi
  400b6d:	e8 9c 00 00 00       	callq  400c0e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b72:	c9                   	leaveq 
  400b73:	c3                   	retq   

0000000000400b74 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b74:	55                   	push   %rbp
  400b75:	48 89 e5             	mov    %rsp,%rbp
  400b78:	48 83 ec 30          	sub    $0x30,%rsp
  400b7c:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b80:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b84:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b88:	e8 a3 fc ff ff       	callq  400830 <_ZN9TestClass7storageEv>
  400b8d:	48 83 c0 08          	add    $0x8,%rax
  400b91:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b95:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b99:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b9d:	48 83 e8 04          	sub    $0x4,%rax
  400ba1:	48 c1 e8 02          	shr    $0x2,%rax
  400ba5:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bac:	00 
  400bad:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bb1:	48 01 d0             	add    %rdx,%rax
  400bb4:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bb8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bbc:	c9                   	leaveq 
  400bbd:	c3                   	retq   

0000000000400bbe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bbe:	55                   	push   %rbp
  400bbf:	48 89 e5             	mov    %rsp,%rbp
  400bc2:	48 83 ec 30          	sub    $0x30,%rsp
  400bc6:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400bca:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bce:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bd2:	e8 59 fc ff ff       	callq  400830 <_ZN9TestClass7storageEv>
  400bd7:	48 83 c0 08          	add    $0x8,%rax
  400bdb:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bdf:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400be3:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400be7:	48 83 e8 04          	sub    $0x4,%rax
  400beb:	48 c1 e8 02          	shr    $0x2,%rax
  400bef:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bf6:	00 
  400bf7:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bfb:	48 01 d0             	add    %rdx,%rax
  400bfe:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c04:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c08:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c0c:	c9                   	leaveq 
  400c0d:	c3                   	retq   

0000000000400c0e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c0e:	55                   	push   %rbp
  400c0f:	48 89 e5             	mov    %rsp,%rbp
  400c12:	48 83 ec 20          	sub    $0x20,%rsp
  400c16:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c1a:	e8 11 fc ff ff       	callq  400830 <_ZN9TestClass7storageEv>
  400c1f:	48 83 c0 08          	add    $0x8,%rax
  400c23:	48 89 c2             	mov    %rax,%rdx
  400c26:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c2a:	48 c1 e0 02          	shl    $0x2,%rax
  400c2e:	48 01 d0             	add    %rdx,%rax
  400c31:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c35:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c39:	c9                   	leaveq 
  400c3a:	c3                   	retq   
  400c3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c40 <__libc_csu_init>:
  400c40:	41 57                	push   %r15
  400c42:	41 56                	push   %r14
  400c44:	41 89 ff             	mov    %edi,%r15d
  400c47:	41 55                	push   %r13
  400c49:	41 54                	push   %r12
  400c4b:	4c 8d 25 be 11 20 00 	lea    0x2011be(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c52:	55                   	push   %rbp
  400c53:	48 8d 2d be 11 20 00 	lea    0x2011be(%rip),%rbp        # 601e18 <__init_array_end>
  400c5a:	53                   	push   %rbx
  400c5b:	49 89 f6             	mov    %rsi,%r14
  400c5e:	49 89 d5             	mov    %rdx,%r13
  400c61:	4c 29 e5             	sub    %r12,%rbp
  400c64:	48 83 ec 08          	sub    $0x8,%rsp
  400c68:	48 c1 fd 03          	sar    $0x3,%rbp
  400c6c:	e8 b7 f7 ff ff       	callq  400428 <_init>
  400c71:	48 85 ed             	test   %rbp,%rbp
  400c74:	74 20                	je     400c96 <__libc_csu_init+0x56>
  400c76:	31 db                	xor    %ebx,%ebx
  400c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c7f:	00 
  400c80:	4c 89 ea             	mov    %r13,%rdx
  400c83:	4c 89 f6             	mov    %r14,%rsi
  400c86:	44 89 ff             	mov    %r15d,%edi
  400c89:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c8d:	48 83 c3 01          	add    $0x1,%rbx
  400c91:	48 39 eb             	cmp    %rbp,%rbx
  400c94:	75 ea                	jne    400c80 <__libc_csu_init+0x40>
  400c96:	48 83 c4 08          	add    $0x8,%rsp
  400c9a:	5b                   	pop    %rbx
  400c9b:	5d                   	pop    %rbp
  400c9c:	41 5c                	pop    %r12
  400c9e:	41 5d                	pop    %r13
  400ca0:	41 5e                	pop    %r14
  400ca2:	41 5f                	pop    %r15
  400ca4:	c3                   	retq   
  400ca5:	90                   	nop
  400ca6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cad:	00 00 00 

0000000000400cb0 <__libc_csu_fini>:
  400cb0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400cb4 <_fini>:
  400cb4:	48 83 ec 08          	sub    $0x8,%rsp
  400cb8:	48 83 c4 08          	add    $0x8,%rsp
  400cbc:	c3                   	retq   
