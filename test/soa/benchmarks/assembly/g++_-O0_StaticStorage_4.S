
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
  4004af:	49 c7 c0 a0 0c 40 00 	mov    $0x400ca0,%r8
  4004b6:	48 c7 c1 30 0c 40 00 	mov    $0x400c30,%rcx
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
  4005ae:	e8 76 02 00 00       	callq  400829 <_ZN9TestClass18initialize_storageEv>
  4005b3:	bf 04 00 00 00       	mov    $0x4,%edi
  4005b8:	e8 29 04 00 00       	callq  4009e6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005bd:	48 89 c3             	mov    %rax,%rbx
  4005c0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c5:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005c9:	48 83 f8 03          	cmp    $0x3,%rax
  4005cd:	74 06                	je     4005d5 <main+0x3f>
  4005cf:	48 83 c0 01          	add    $0x1,%rax
  4005d3:	eb f0                	jmp    4005c5 <main+0x2f>
  4005d5:	48 89 df             	mov    %rbx,%rdi
  4005d8:	e8 33 03 00 00       	callq  400910 <_ZN9TestClassC1Ev>
  4005dd:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005e1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005e5:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4005e9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005ed:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005f2:	48 89 ce             	mov    %rcx,%rsi
  4005f5:	48 89 c7             	mov    %rax,%rdi
  4005f8:	e8 39 04 00 00       	callq  400a36 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4005fd:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400601:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400605:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400609:	ba 88 88 00 00       	mov    $0x8888,%edx
  40060e:	48 89 ce             	mov    %rcx,%rsi
  400611:	48 89 c7             	mov    %rax,%rdi
  400614:	e8 5d 04 00 00       	callq  400a76 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400619:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40061d:	48 89 c7             	mov    %rax,%rdi
  400620:	e8 27 02 00 00       	callq  40084c <_ZN9TestClass15increase_field0Ev>
  400625:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400629:	48 89 c7             	mov    %rax,%rdi
  40062c:	e8 69 02 00 00       	callq  40089a <_ZN9TestClass15increase_field1Ev>
  400631:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400635:	48 83 c0 04          	add    $0x4,%rax
  400639:	48 89 c7             	mov    %rax,%rdi
  40063c:	e8 8f 04 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400641:	8b 18                	mov    (%rax),%ebx
  400643:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400647:	48 83 c0 04          	add    $0x4,%rax
  40064b:	48 89 c7             	mov    %rax,%rdi
  40064e:	e8 63 04 00 00       	callq  400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400653:	8b 00                	mov    (%rax),%eax
  400655:	89 da                	mov    %ebx,%edx
  400657:	89 c6                	mov    %eax,%esi
  400659:	bf c4 0c 40 00       	mov    $0x400cc4,%edi
  40065e:	b8 00 00 00 00       	mov    $0x0,%eax
  400663:	e8 08 fe ff ff       	callq  400470 <printf@plt>
  400668:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40066c:	48 83 c0 04          	add    $0x4,%rax
  400670:	48 89 c7             	mov    %rax,%rdi
  400673:	e8 3e 04 00 00       	callq  400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400678:	8b 00                	mov    (%rax),%eax
  40067a:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40067f:	75 19                	jne    40069a <main+0x104>
  400681:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400685:	48 83 c0 04          	add    $0x4,%rax
  400689:	48 89 c7             	mov    %rax,%rdi
  40068c:	e8 3f 04 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
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
  4006d2:	e8 0f 03 00 00       	callq  4009e6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006d7:	48 89 c3             	mov    %rax,%rbx
  4006da:	b8 00 00 00 00       	mov    $0x0,%eax
  4006df:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006e3:	48 83 f8 03          	cmp    $0x3,%rax
  4006e7:	74 06                	je     4006ef <_Z12new_instancev+0x2b>
  4006e9:	48 83 c0 01          	add    $0x1,%rax
  4006ed:	eb f0                	jmp    4006df <_Z12new_instancev+0x1b>
  4006ef:	48 89 df             	mov    %rbx,%rdi
  4006f2:	e8 19 02 00 00       	callq  400910 <_ZN9TestClassC1Ev>
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
  400733:	e8 fe 02 00 00       	callq  400a36 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
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
  400766:	e8 4b 03 00 00       	callq  400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
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
  4007a1:	e8 d0 02 00 00       	callq  400a76 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
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
  4007d4:	e8 f7 02 00 00       	callq  400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007d9:	8b 00                	mov    (%rax),%eax
  4007db:	c9                   	leaveq 
  4007dc:	c3                   	retq   

00000000004007dd <_Z21explicit_write_field0m>:
  4007dd:	55                   	push   %rbp
  4007de:	48 89 e5             	mov    %rsp,%rbp
  4007e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007e5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007e9:	c7 04 85 00 a7 61 00 	movl   $0x7777,0x61a700(,%rax,4)
  4007f0:	77 77 00 00 
  4007f4:	90                   	nop
  4007f5:	5d                   	pop    %rbp
  4007f6:	c3                   	retq   

00000000004007f7 <_Z20explicit_read_field0m>:
  4007f7:	55                   	push   %rbp
  4007f8:	48 89 e5             	mov    %rsp,%rbp
  4007fb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007ff:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400803:	8b 04 85 00 a7 61 00 	mov    0x61a700(,%rax,4),%eax
  40080a:	5d                   	pop    %rbp
  40080b:	c3                   	retq   

000000000040080c <_ZnwmPv>:
  40080c:	55                   	push   %rbp
  40080d:	48 89 e5             	mov    %rsp,%rbp
  400810:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400814:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400818:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40081c:	5d                   	pop    %rbp
  40081d:	c3                   	retq   

000000000040081e <_ZN9TestClass7storageEv>:
  40081e:	55                   	push   %rbp
  40081f:	48 89 e5             	mov    %rsp,%rbp
  400822:	b8 60 20 60 00       	mov    $0x602060,%eax
  400827:	5d                   	pop    %rbp
  400828:	c3                   	retq   

0000000000400829 <_ZN9TestClass18initialize_storageEv>:
  400829:	55                   	push   %rbp
  40082a:	48 89 e5             	mov    %rsp,%rbp
  40082d:	be 60 20 60 00       	mov    $0x602060,%esi
  400832:	bf b0 91 00 00       	mov    $0x91b0,%edi
  400837:	e8 d0 ff ff ff       	callq  40080c <_ZnwmPv>
  40083c:	48 85 c0             	test   %rax,%rax
  40083f:	74 08                	je     400849 <_ZN9TestClass18initialize_storageEv+0x20>
  400841:	48 89 c7             	mov    %rax,%rdi
  400844:	e8 03 01 00 00       	callq  40094c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400849:	90                   	nop
  40084a:	5d                   	pop    %rbp
  40084b:	c3                   	retq   

000000000040084c <_ZN9TestClass15increase_field0Ev>:
  40084c:	55                   	push   %rbp
  40084d:	48 89 e5             	mov    %rsp,%rbp
  400850:	48 83 ec 30          	sub    $0x30,%rsp
  400854:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400858:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40085f:	00 00 
  400861:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400865:	31 c0                	xor    %eax,%eax
  400867:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40086b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40086f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400873:	ba 55 55 00 00       	mov    $0x5555,%edx
  400878:	48 89 ce             	mov    %rcx,%rsi
  40087b:	48 89 c7             	mov    %rax,%rdi
  40087e:	e8 e3 00 00 00       	callq  400966 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400883:	90                   	nop
  400884:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400888:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40088f:	00 00 
  400891:	74 05                	je     400898 <_ZN9TestClass15increase_field0Ev+0x4c>
  400893:	e8 c8 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400898:	c9                   	leaveq 
  400899:	c3                   	retq   

000000000040089a <_ZN9TestClass15increase_field1Ev>:
  40089a:	55                   	push   %rbp
  40089b:	48 89 e5             	mov    %rsp,%rbp
  40089e:	48 83 ec 30          	sub    $0x30,%rsp
  4008a2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008a6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4008ad:	00 00 
  4008af:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008b3:	31 c0                	xor    %eax,%eax
  4008b5:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4008b9:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4008bd:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4008c1:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008c6:	48 89 ce             	mov    %rcx,%rsi
  4008c9:	48 89 c7             	mov    %rax,%rdi
  4008cc:	e8 d5 00 00 00       	callq  4009a6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008d1:	90                   	nop
  4008d2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008d6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008dd:	00 00 
  4008df:	74 05                	je     4008e6 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008e1:	e8 7a fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  4008e6:	c9                   	leaveq 
  4008e7:	c3                   	retq   

00000000004008e8 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008e8:	55                   	push   %rbp
  4008e9:	48 89 e5             	mov    %rsp,%rbp
  4008ec:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f0:	90                   	nop
  4008f1:	5d                   	pop    %rbp
  4008f2:	c3                   	retq   
  4008f3:	90                   	nop

00000000004008f4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  4008f4:	55                   	push   %rbp
  4008f5:	48 89 e5             	mov    %rsp,%rbp
  4008f8:	48 83 ec 10          	sub    $0x10,%rsp
  4008fc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400900:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400904:	48 89 c7             	mov    %rax,%rdi
  400907:	e8 dc ff ff ff       	callq  4008e8 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  40090c:	90                   	nop
  40090d:	c9                   	leaveq 
  40090e:	c3                   	retq   
  40090f:	90                   	nop

0000000000400910 <_ZN9TestClassC1Ev>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 10          	sub    $0x10,%rsp
  400918:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40091c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400920:	48 89 c7             	mov    %rax,%rdi
  400923:	e8 cc ff ff ff       	callq  4008f4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  400928:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40092c:	48 83 c0 04          	add    $0x4,%rax
  400930:	48 89 c7             	mov    %rax,%rdi
  400933:	e8 dc 00 00 00       	callq  400a14 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400938:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40093c:	48 83 c0 04          	add    $0x4,%rax
  400940:	48 89 c7             	mov    %rax,%rdi
  400943:	e8 d8 00 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400948:	90                   	nop
  400949:	c9                   	leaveq 
  40094a:	c3                   	retq   
  40094b:	90                   	nop

000000000040094c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  40094c:	55                   	push   %rbp
  40094d:	48 89 e5             	mov    %rsp,%rbp
  400950:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400954:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400958:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  40095f:	00 00 00 00 
  400963:	90                   	nop
  400964:	5d                   	pop    %rbp
  400965:	c3                   	retq   

0000000000400966 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400966:	55                   	push   %rbp
  400967:	48 89 e5             	mov    %rsp,%rbp
  40096a:	48 83 ec 20          	sub    $0x20,%rsp
  40096e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400972:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400976:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400979:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40097d:	48 89 c7             	mov    %rax,%rdi
  400980:	e8 65 01 00 00       	callq  400aea <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400985:	8b 10                	mov    (%rax),%edx
  400987:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40098b:	89 10                	mov    %edx,(%rax)
  40098d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400991:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400995:	48 89 d6             	mov    %rdx,%rsi
  400998:	48 89 c7             	mov    %rax,%rdi
  40099b:	e8 64 01 00 00       	callq  400b04 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  4009a0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a4:	c9                   	leaveq 
  4009a5:	c3                   	retq   

00000000004009a6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  4009a6:	55                   	push   %rbp
  4009a7:	48 89 e5             	mov    %rsp,%rbp
  4009aa:	48 83 ec 20          	sub    $0x20,%rsp
  4009ae:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b2:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009b6:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009b9:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009bd:	48 89 c7             	mov    %rax,%rdi
  4009c0:	e8 4f 01 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009c5:	8b 10                	mov    (%rax),%edx
  4009c7:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4009cb:	89 10                	mov    %edx,(%rax)
  4009cd:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009d1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009d5:	48 89 d6             	mov    %rdx,%rsi
  4009d8:	48 89 c7             	mov    %rax,%rdi
  4009db:	e8 4e 01 00 00       	callq  400b2e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009e0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009e4:	c9                   	leaveq 
  4009e5:	c3                   	retq   

00000000004009e6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009e6:	55                   	push   %rbp
  4009e7:	48 89 e5             	mov    %rsp,%rbp
  4009ea:	48 83 ec 10          	sub    $0x10,%rsp
  4009ee:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f2:	e8 46 01 00 00       	callq  400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009f7:	e8 22 fe ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  4009fc:	48 89 c2             	mov    %rax,%rdx
  4009ff:	48 8b 02             	mov    (%rdx),%rax
  400a02:	48 8d 48 01          	lea    0x1(%rax),%rcx
  400a06:	48 89 0a             	mov    %rcx,(%rdx)
  400a09:	48 89 c7             	mov    %rax,%rdi
  400a0c:	e8 33 01 00 00       	callq  400b44 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400a11:	c9                   	leaveq 
  400a12:	c3                   	retq   
  400a13:	90                   	nop

0000000000400a14 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  400a14:	55                   	push   %rbp
  400a15:	48 89 e5             	mov    %rsp,%rbp
  400a18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1c:	90                   	nop
  400a1d:	5d                   	pop    %rbp
  400a1e:	c3                   	retq   
  400a1f:	90                   	nop

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a28:	90                   	nop
  400a29:	5d                   	pop    %rbp
  400a2a:	c3                   	retq   

0000000000400a2b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a2b:	55                   	push   %rbp
  400a2c:	48 89 e5             	mov    %rsp,%rbp
  400a2f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a33:	90                   	nop
  400a34:	5d                   	pop    %rbp
  400a35:	c3                   	retq   

0000000000400a36 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a36:	55                   	push   %rbp
  400a37:	48 89 e5             	mov    %rsp,%rbp
  400a3a:	48 83 ec 20          	sub    $0x20,%rsp
  400a3e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a42:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a46:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a49:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a4d:	48 89 c7             	mov    %rax,%rdi
  400a50:	e8 95 00 00 00       	callq  400aea <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a55:	48 89 c2             	mov    %rax,%rdx
  400a58:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a5b:	89 02                	mov    %eax,(%rdx)
  400a5d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a61:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a65:	48 89 d6             	mov    %rdx,%rsi
  400a68:	48 89 c7             	mov    %rax,%rdi
  400a6b:	e8 94 00 00 00       	callq  400b04 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a70:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a74:	c9                   	leaveq 
  400a75:	c3                   	retq   

0000000000400a76 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a76:	55                   	push   %rbp
  400a77:	48 89 e5             	mov    %rsp,%rbp
  400a7a:	48 83 ec 20          	sub    $0x20,%rsp
  400a7e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a82:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a86:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a89:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a8d:	48 89 c7             	mov    %rax,%rdi
  400a90:	e8 7f 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a95:	48 89 c2             	mov    %rax,%rdx
  400a98:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a9b:	89 02                	mov    %eax,(%rdx)
  400a9d:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400aa1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aa5:	48 89 d6             	mov    %rdx,%rsi
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 7e 00 00 00       	callq  400b2e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400ab0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab4:	c9                   	leaveq 
  400ab5:	c3                   	retq   

0000000000400ab6 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400ab6:	55                   	push   %rbp
  400ab7:	48 89 e5             	mov    %rsp,%rbp
  400aba:	48 83 ec 10          	sub    $0x10,%rsp
  400abe:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ac6:	48 89 c7             	mov    %rax,%rdi
  400ac9:	e8 1c 00 00 00       	callq  400aea <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ace:	c9                   	leaveq 
  400acf:	c3                   	retq   

0000000000400ad0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 83 ec 10          	sub    $0x10,%rsp
  400ad8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400adc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae0:	48 89 c7             	mov    %rax,%rdi
  400ae3:	e8 2c 00 00 00       	callq  400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae8:	c9                   	leaveq 
  400ae9:	c3                   	retq   

0000000000400aea <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aea:	55                   	push   %rbp
  400aeb:	48 89 e5             	mov    %rsp,%rbp
  400aee:	48 83 ec 10          	sub    $0x10,%rsp
  400af2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400afa:	48 89 c7             	mov    %rax,%rdi
  400afd:	e8 60 00 00 00       	callq  400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b02:	c9                   	leaveq 
  400b03:	c3                   	retq   

0000000000400b04 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400b04:	55                   	push   %rbp
  400b05:	48 89 e5             	mov    %rsp,%rbp
  400b08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b10:	90                   	nop
  400b11:	5d                   	pop    %rbp
  400b12:	c3                   	retq   
  400b13:	90                   	nop

0000000000400b14 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b14:	55                   	push   %rbp
  400b15:	48 89 e5             	mov    %rsp,%rbp
  400b18:	48 83 ec 10          	sub    $0x10,%rsp
  400b1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b20:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b24:	48 89 c7             	mov    %rax,%rdi
  400b27:	e8 80 00 00 00       	callq  400bac <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2c:	c9                   	leaveq 
  400b2d:	c3                   	retq   

0000000000400b2e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b2e:	55                   	push   %rbp
  400b2f:	48 89 e5             	mov    %rsp,%rbp
  400b32:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b36:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b3a:	90                   	nop
  400b3b:	5d                   	pop    %rbp
  400b3c:	c3                   	retq   

0000000000400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b3d:	55                   	push   %rbp
  400b3e:	48 89 e5             	mov    %rsp,%rbp
  400b41:	90                   	nop
  400b42:	5d                   	pop    %rbp
  400b43:	c3                   	retq   

0000000000400b44 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b44:	55                   	push   %rbp
  400b45:	48 89 e5             	mov    %rsp,%rbp
  400b48:	48 83 ec 10          	sub    $0x10,%rsp
  400b4c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b50:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b54:	48 83 c0 01          	add    $0x1,%rax
  400b58:	48 89 c7             	mov    %rax,%rdi
  400b5b:	e8 9c 00 00 00       	callq  400bfc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b60:	c9                   	leaveq 
  400b61:	c3                   	retq   

0000000000400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b62:	55                   	push   %rbp
  400b63:	48 89 e5             	mov    %rsp,%rbp
  400b66:	48 83 ec 30          	sub    $0x30,%rsp
  400b6a:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b6e:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b72:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b76:	e8 a3 fc ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400b7b:	48 83 c0 08          	add    $0x8,%rax
  400b7f:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b83:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b87:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b8b:	48 83 e8 04          	sub    $0x4,%rax
  400b8f:	48 c1 e8 02          	shr    $0x2,%rax
  400b93:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b9a:	00 
  400b9b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b9f:	48 01 d0             	add    %rdx,%rax
  400ba2:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400ba6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400baa:	c9                   	leaveq 
  400bab:	c3                   	retq   

0000000000400bac <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bac:	55                   	push   %rbp
  400bad:	48 89 e5             	mov    %rsp,%rbp
  400bb0:	48 83 ec 30          	sub    $0x30,%rsp
  400bb4:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400bb8:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bbc:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bc0:	e8 59 fc ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400bc5:	48 83 c0 08          	add    $0x8,%rax
  400bc9:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bcd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bd1:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bd5:	48 83 e8 04          	sub    $0x4,%rax
  400bd9:	48 c1 e8 02          	shr    $0x2,%rax
  400bdd:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400be4:	00 
  400be5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400be9:	48 01 d0             	add    %rdx,%rax
  400bec:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bf2:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bf6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bfa:	c9                   	leaveq 
  400bfb:	c3                   	retq   

0000000000400bfc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bfc:	55                   	push   %rbp
  400bfd:	48 89 e5             	mov    %rsp,%rbp
  400c00:	48 83 ec 20          	sub    $0x20,%rsp
  400c04:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c08:	e8 11 fc ff ff       	callq  40081e <_ZN9TestClass7storageEv>
  400c0d:	48 83 c0 08          	add    $0x8,%rax
  400c11:	48 89 c2             	mov    %rax,%rdx
  400c14:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c18:	48 c1 e0 02          	shl    $0x2,%rax
  400c1c:	48 01 d0             	add    %rdx,%rax
  400c1f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c23:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c27:	c9                   	leaveq 
  400c28:	c3                   	retq   
  400c29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c30 <__libc_csu_init>:
  400c30:	41 57                	push   %r15
  400c32:	41 56                	push   %r14
  400c34:	41 89 ff             	mov    %edi,%r15d
  400c37:	41 55                	push   %r13
  400c39:	41 54                	push   %r12
  400c3b:	4c 8d 25 ce 11 20 00 	lea    0x2011ce(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c42:	55                   	push   %rbp
  400c43:	48 8d 2d ce 11 20 00 	lea    0x2011ce(%rip),%rbp        # 601e18 <__init_array_end>
  400c4a:	53                   	push   %rbx
  400c4b:	49 89 f6             	mov    %rsi,%r14
  400c4e:	49 89 d5             	mov    %rdx,%r13
  400c51:	4c 29 e5             	sub    %r12,%rbp
  400c54:	48 83 ec 08          	sub    $0x8,%rsp
  400c58:	48 c1 fd 03          	sar    $0x3,%rbp
  400c5c:	e8 c7 f7 ff ff       	callq  400428 <_init>
  400c61:	48 85 ed             	test   %rbp,%rbp
  400c64:	74 20                	je     400c86 <__libc_csu_init+0x56>
  400c66:	31 db                	xor    %ebx,%ebx
  400c68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c6f:	00 
  400c70:	4c 89 ea             	mov    %r13,%rdx
  400c73:	4c 89 f6             	mov    %r14,%rsi
  400c76:	44 89 ff             	mov    %r15d,%edi
  400c79:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c7d:	48 83 c3 01          	add    $0x1,%rbx
  400c81:	48 39 eb             	cmp    %rbp,%rbx
  400c84:	75 ea                	jne    400c70 <__libc_csu_init+0x40>
  400c86:	48 83 c4 08          	add    $0x8,%rsp
  400c8a:	5b                   	pop    %rbx
  400c8b:	5d                   	pop    %rbp
  400c8c:	41 5c                	pop    %r12
  400c8e:	41 5d                	pop    %r13
  400c90:	41 5e                	pop    %r14
  400c92:	41 5f                	pop    %r15
  400c94:	c3                   	retq   
  400c95:	90                   	nop
  400c96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9d:	00 00 00 

0000000000400ca0 <__libc_csu_fini>:
  400ca0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ca4 <_fini>:
  400ca4:	48 83 ec 08          	sub    $0x8,%rsp
  400ca8:	48 83 c4 08          	add    $0x8,%rsp
  400cac:	c3                   	retq   
