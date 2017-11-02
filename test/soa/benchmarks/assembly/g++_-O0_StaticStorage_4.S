
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
  4005a6:	48 83 ec 28          	sub    $0x28,%rsp
  4005aa:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4005b1:	00 00 
  4005b3:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4005b7:	31 c0                	xor    %eax,%eax
  4005b9:	e8 ec 03 00 00       	callq  4009aa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageEv>
  4005be:	bf 04 00 00 00       	mov    $0x4,%edi
  4005c3:	e8 08 04 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005c8:	48 89 c3             	mov    %rax,%rbx
  4005cb:	b8 00 00 00 00       	mov    $0x0,%eax
  4005d0:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005d4:	48 83 f8 03          	cmp    $0x3,%rax
  4005d8:	74 06                	je     4005e0 <main+0x3f>
  4005da:	48 83 c0 01          	add    $0x1,%rax
  4005de:	eb f0                	jmp    4005d0 <main+0x2f>
  4005e0:	48 89 df             	mov    %rbx,%rdi
  4005e3:	e8 06 03 00 00       	callq  4008ee <_ZN9TestClassC1Ev>
  4005e8:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005ec:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005f0:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4005f4:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005f8:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005fd:	48 89 ce             	mov    %rcx,%rsi
  400600:	48 89 c7             	mov    %rax,%rdi
  400603:	e8 18 04 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400608:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40060c:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400610:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400614:	ba 88 88 00 00       	mov    $0x8888,%edx
  400619:	48 89 ce             	mov    %rcx,%rsi
  40061c:	48 89 c7             	mov    %rax,%rdi
  40061f:	e8 3c 04 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400624:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400628:	48 89 c7             	mov    %rax,%rdi
  40062b:	e8 fa 01 00 00       	callq  40082a <_ZN9TestClass15increase_field0Ev>
  400630:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400634:	48 89 c7             	mov    %rax,%rdi
  400637:	e8 3c 02 00 00       	callq  400878 <_ZN9TestClass15increase_field1Ev>
  40063c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400640:	48 83 c0 04          	add    $0x4,%rax
  400644:	48 89 c7             	mov    %rax,%rdi
  400647:	e8 6e 04 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40064c:	8b 18                	mov    (%rax),%ebx
  40064e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400652:	48 83 c0 04          	add    $0x4,%rax
  400656:	48 89 c7             	mov    %rax,%rdi
  400659:	e8 42 04 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40065e:	8b 00                	mov    (%rax),%eax
  400660:	89 da                	mov    %ebx,%edx
  400662:	89 c6                	mov    %eax,%esi
  400664:	bf c4 0c 40 00       	mov    $0x400cc4,%edi
  400669:	b8 00 00 00 00       	mov    $0x0,%eax
  40066e:	e8 fd fd ff ff       	callq  400470 <printf@plt>
  400673:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400677:	48 83 c0 04          	add    $0x4,%rax
  40067b:	48 89 c7             	mov    %rax,%rdi
  40067e:	e8 1d 04 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400683:	8b 00                	mov    (%rax),%eax
  400685:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40068a:	75 19                	jne    4006a5 <main+0x104>
  40068c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400690:	48 83 c0 04          	add    $0x4,%rax
  400694:	48 89 c7             	mov    %rax,%rdi
  400697:	e8 1e 04 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40069c:	8b 00                	mov    (%rax),%eax
  40069e:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  4006a3:	74 07                	je     4006ac <main+0x10b>
  4006a5:	b8 01 00 00 00       	mov    $0x1,%eax
  4006aa:	eb 05                	jmp    4006b1 <main+0x110>
  4006ac:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b1:	0f b6 c0             	movzbl %al,%eax
  4006b4:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006b8:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006bf:	00 00 
  4006c1:	74 05                	je     4006c8 <main+0x127>
  4006c3:	e8 98 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  4006c8:	48 83 c4 28          	add    $0x28,%rsp
  4006cc:	5b                   	pop    %rbx
  4006cd:	5d                   	pop    %rbp
  4006ce:	c3                   	retq   

00000000004006cf <_Z12new_instancev>:
  4006cf:	55                   	push   %rbp
  4006d0:	48 89 e5             	mov    %rsp,%rbp
  4006d3:	53                   	push   %rbx
  4006d4:	48 83 ec 08          	sub    $0x8,%rsp
  4006d8:	bf 04 00 00 00       	mov    $0x4,%edi
  4006dd:	e8 ee 02 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006e2:	48 89 c3             	mov    %rax,%rbx
  4006e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ea:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006ee:	48 83 f8 03          	cmp    $0x3,%rax
  4006f2:	74 06                	je     4006fa <_Z12new_instancev+0x2b>
  4006f4:	48 83 c0 01          	add    $0x1,%rax
  4006f8:	eb f0                	jmp    4006ea <_Z12new_instancev+0x1b>
  4006fa:	48 89 df             	mov    %rbx,%rdi
  4006fd:	e8 ec 01 00 00       	callq  4008ee <_ZN9TestClassC1Ev>
  400702:	48 89 d8             	mov    %rbx,%rax
  400705:	48 83 c4 08          	add    $0x8,%rsp
  400709:	5b                   	pop    %rbx
  40070a:	5d                   	pop    %rbp
  40070b:	c3                   	retq   

000000000040070c <_Z12write_field0P9TestClass>:
  40070c:	55                   	push   %rbp
  40070d:	48 89 e5             	mov    %rsp,%rbp
  400710:	48 83 ec 30          	sub    $0x30,%rsp
  400714:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400718:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40071f:	00 00 
  400721:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400725:	31 c0                	xor    %eax,%eax
  400727:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40072b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40072f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400733:	ba 77 77 00 00       	mov    $0x7777,%edx
  400738:	48 89 ce             	mov    %rcx,%rsi
  40073b:	48 89 c7             	mov    %rax,%rdi
  40073e:	e8 dd 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400743:	90                   	nop
  400744:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400748:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40074f:	00 00 
  400751:	74 05                	je     400758 <_Z12write_field0P9TestClass+0x4c>
  400753:	e8 08 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400758:	c9                   	leaveq 
  400759:	c3                   	retq   

000000000040075a <_Z11read_field0P9TestClass>:
  40075a:	55                   	push   %rbp
  40075b:	48 89 e5             	mov    %rsp,%rbp
  40075e:	48 83 ec 10          	sub    $0x10,%rsp
  400762:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400766:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40076a:	48 83 c0 04          	add    $0x4,%rax
  40076e:	48 89 c7             	mov    %rax,%rdi
  400771:	e8 2a 03 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400776:	8b 00                	mov    (%rax),%eax
  400778:	c9                   	leaveq 
  400779:	c3                   	retq   

000000000040077a <_Z12write_field1P9TestClass>:
  40077a:	55                   	push   %rbp
  40077b:	48 89 e5             	mov    %rsp,%rbp
  40077e:	48 83 ec 30          	sub    $0x30,%rsp
  400782:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400786:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40078d:	00 00 
  40078f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400793:	31 c0                	xor    %eax,%eax
  400795:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400799:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40079d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007a1:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007a6:	48 89 ce             	mov    %rcx,%rsi
  4007a9:	48 89 c7             	mov    %rax,%rdi
  4007ac:	e8 af 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007b1:	90                   	nop
  4007b2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007b6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007bd:	00 00 
  4007bf:	74 05                	je     4007c6 <_Z12write_field1P9TestClass+0x4c>
  4007c1:	e8 9a fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007c6:	c9                   	leaveq 
  4007c7:	c3                   	retq   

00000000004007c8 <_Z11read_field1P9TestClass>:
  4007c8:	55                   	push   %rbp
  4007c9:	48 89 e5             	mov    %rsp,%rbp
  4007cc:	48 83 ec 10          	sub    $0x10,%rsp
  4007d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007d4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007d8:	48 83 c0 04          	add    $0x4,%rax
  4007dc:	48 89 c7             	mov    %rax,%rdi
  4007df:	e8 d6 02 00 00       	callq  400aba <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007e4:	8b 00                	mov    (%rax),%eax
  4007e6:	c9                   	leaveq 
  4007e7:	c3                   	retq   

00000000004007e8 <_Z21explicit_write_field0m>:
  4007e8:	55                   	push   %rbp
  4007e9:	48 89 e5             	mov    %rsp,%rbp
  4007ec:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007f0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007f4:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  4007fb:	77 77 00 00 
  4007ff:	90                   	nop
  400800:	5d                   	pop    %rbp
  400801:	c3                   	retq   

0000000000400802 <_Z20explicit_read_field0m>:
  400802:	55                   	push   %rbp
  400803:	48 89 e5             	mov    %rsp,%rbp
  400806:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40080a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40080e:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  400815:	5d                   	pop    %rbp
  400816:	c3                   	retq   

0000000000400817 <_ZnwmPv>:
  400817:	55                   	push   %rbp
  400818:	48 89 e5             	mov    %rsp,%rbp
  40081b:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40081f:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400823:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400827:	5d                   	pop    %rbp
  400828:	c3                   	retq   
  400829:	90                   	nop

000000000040082a <_ZN9TestClass15increase_field0Ev>:
  40082a:	55                   	push   %rbp
  40082b:	48 89 e5             	mov    %rsp,%rbp
  40082e:	48 83 ec 30          	sub    $0x30,%rsp
  400832:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400836:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40083d:	00 00 
  40083f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400843:	31 c0                	xor    %eax,%eax
  400845:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400849:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40084d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400851:	ba 55 55 00 00       	mov    $0x5555,%edx
  400856:	48 89 ce             	mov    %rcx,%rsi
  400859:	48 89 c7             	mov    %rax,%rdi
  40085c:	e8 c9 00 00 00       	callq  40092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400861:	90                   	nop
  400862:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400866:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40086d:	00 00 
  40086f:	74 05                	je     400876 <_ZN9TestClass15increase_field0Ev+0x4c>
  400871:	e8 ea fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400876:	c9                   	leaveq 
  400877:	c3                   	retq   

0000000000400878 <_ZN9TestClass15increase_field1Ev>:
  400878:	55                   	push   %rbp
  400879:	48 89 e5             	mov    %rsp,%rbp
  40087c:	48 83 ec 30          	sub    $0x30,%rsp
  400880:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400884:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40088b:	00 00 
  40088d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400891:	31 c0                	xor    %eax,%eax
  400893:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400897:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40089b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40089f:	ba 44 44 00 00       	mov    $0x4444,%edx
  4008a4:	48 89 ce             	mov    %rcx,%rsi
  4008a7:	48 89 c7             	mov    %rax,%rdi
  4008aa:	e8 bb 00 00 00       	callq  40096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  4008af:	90                   	nop
  4008b0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008bb:	00 00 
  4008bd:	74 05                	je     4008c4 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008bf:	e8 9c fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  4008c4:	c9                   	leaveq 
  4008c5:	c3                   	retq   

00000000004008c6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008c6:	55                   	push   %rbp
  4008c7:	48 89 e5             	mov    %rsp,%rbp
  4008ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ce:	90                   	nop
  4008cf:	5d                   	pop    %rbp
  4008d0:	c3                   	retq   
  4008d1:	90                   	nop

00000000004008d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  4008d2:	55                   	push   %rbp
  4008d3:	48 89 e5             	mov    %rsp,%rbp
  4008d6:	48 83 ec 10          	sub    $0x10,%rsp
  4008da:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008de:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008e2:	48 89 c7             	mov    %rax,%rdi
  4008e5:	e8 dc ff ff ff       	callq  4008c6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ea:	90                   	nop
  4008eb:	c9                   	leaveq 
  4008ec:	c3                   	retq   
  4008ed:	90                   	nop

00000000004008ee <_ZN9TestClassC1Ev>:
  4008ee:	55                   	push   %rbp
  4008ef:	48 89 e5             	mov    %rsp,%rbp
  4008f2:	48 83 ec 10          	sub    $0x10,%rsp
  4008f6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fa:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008fe:	48 89 c7             	mov    %rax,%rdi
  400901:	e8 cc ff ff ff       	callq  4008d2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  400906:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40090a:	48 83 c0 04          	add    $0x4,%rax
  40090e:	48 89 c7             	mov    %rax,%rdi
  400911:	e8 e8 00 00 00       	callq  4009fe <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400916:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40091a:	48 83 c0 04          	add    $0x4,%rax
  40091e:	48 89 c7             	mov    %rax,%rdi
  400921:	e8 e4 00 00 00       	callq  400a0a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400926:	90                   	nop
  400927:	c9                   	leaveq 
  400928:	c3                   	retq   
  400929:	90                   	nop

000000000040092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40092a:	55                   	push   %rbp
  40092b:	48 89 e5             	mov    %rsp,%rbp
  40092e:	48 83 ec 20          	sub    $0x20,%rsp
  400932:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400936:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40093a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40093d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400941:	48 89 c7             	mov    %rax,%rdi
  400944:	e8 8b 01 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400949:	8b 10                	mov    (%rax),%edx
  40094b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40094f:	89 10                	mov    %edx,(%rax)
  400951:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400955:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400959:	48 89 d6             	mov    %rdx,%rsi
  40095c:	48 89 c7             	mov    %rax,%rdi
  40095f:	e8 8a 01 00 00       	callq  400aee <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400964:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400968:	c9                   	leaveq 
  400969:	c3                   	retq   

000000000040096a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  40096a:	55                   	push   %rbp
  40096b:	48 89 e5             	mov    %rsp,%rbp
  40096e:	48 83 ec 20          	sub    $0x20,%rsp
  400972:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400976:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40097a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40097d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400981:	48 89 c7             	mov    %rax,%rdi
  400984:	e8 75 01 00 00       	callq  400afe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400989:	8b 10                	mov    (%rax),%edx
  40098b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40098f:	89 10                	mov    %edx,(%rax)
  400991:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400995:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400999:	48 89 d6             	mov    %rdx,%rsi
  40099c:	48 89 c7             	mov    %rax,%rdi
  40099f:	e8 74 01 00 00       	callq  400b18 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  4009a4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009a8:	c9                   	leaveq 
  4009a9:	c3                   	retq   

00000000004009aa <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageEv>:
  4009aa:	55                   	push   %rbp
  4009ab:	48 89 e5             	mov    %rsp,%rbp
  4009ae:	e8 e3 fb ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  4009b3:	48 89 c6             	mov    %rax,%rsi
  4009b6:	bf b0 91 00 00       	mov    $0x91b0,%edi
  4009bb:	e8 57 fe ff ff       	callq  400817 <_ZnwmPv>
  4009c0:	48 85 c0             	test   %rax,%rax
  4009c3:	74 08                	je     4009cd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageEv+0x23>
  4009c5:	48 89 c7             	mov    %rax,%rdi
  4009c8:	e8 5b 01 00 00       	callq  400b28 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  4009cd:	90                   	nop
  4009ce:	5d                   	pop    %rbp
  4009cf:	c3                   	retq   

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 83 ec 10          	sub    $0x10,%rsp
  4009d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009dc:	e8 61 01 00 00       	callq  400b42 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009e1:	e8 b0 fb ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  4009e6:	48 89 c2             	mov    %rax,%rdx
  4009e9:	48 8b 02             	mov    (%rdx),%rax
  4009ec:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4009f0:	48 89 0a             	mov    %rcx,(%rdx)
  4009f3:	48 89 c7             	mov    %rax,%rdi
  4009f6:	e8 4e 01 00 00       	callq  400b49 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009fb:	c9                   	leaveq 
  4009fc:	c3                   	retq   
  4009fd:	90                   	nop

00000000004009fe <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4009fe:	55                   	push   %rbp
  4009ff:	48 89 e5             	mov    %rsp,%rbp
  400a02:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a06:	90                   	nop
  400a07:	5d                   	pop    %rbp
  400a08:	c3                   	retq   
  400a09:	90                   	nop

0000000000400a0a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  400a0a:	55                   	push   %rbp
  400a0b:	48 89 e5             	mov    %rsp,%rbp
  400a0e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a12:	90                   	nop
  400a13:	5d                   	pop    %rbp
  400a14:	c3                   	retq   

0000000000400a15 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a15:	55                   	push   %rbp
  400a16:	48 89 e5             	mov    %rsp,%rbp
  400a19:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1d:	90                   	nop
  400a1e:	5d                   	pop    %rbp
  400a1f:	c3                   	retq   

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 83 ec 20          	sub    $0x20,%rsp
  400a28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a30:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a33:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a37:	48 89 c7             	mov    %rax,%rdi
  400a3a:	e8 95 00 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3f:	48 89 c2             	mov    %rax,%rdx
  400a42:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a45:	89 02                	mov    %eax,(%rdx)
  400a47:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a4b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a4f:	48 89 d6             	mov    %rdx,%rsi
  400a52:	48 89 c7             	mov    %rax,%rdi
  400a55:	e8 94 00 00 00       	callq  400aee <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a5a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a5e:	c9                   	leaveq 
  400a5f:	c3                   	retq   

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a60:	55                   	push   %rbp
  400a61:	48 89 e5             	mov    %rsp,%rbp
  400a64:	48 83 ec 20          	sub    $0x20,%rsp
  400a68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a70:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a73:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a77:	48 89 c7             	mov    %rax,%rdi
  400a7a:	e8 7f 00 00 00       	callq  400afe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a7f:	48 89 c2             	mov    %rax,%rdx
  400a82:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a85:	89 02                	mov    %eax,(%rdx)
  400a87:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a8b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a8f:	48 89 d6             	mov    %rdx,%rsi
  400a92:	48 89 c7             	mov    %rax,%rdi
  400a95:	e8 7e 00 00 00       	callq  400b18 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a9a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a9e:	c9                   	leaveq 
  400a9f:	c3                   	retq   

0000000000400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 83 ec 10          	sub    $0x10,%rsp
  400aa8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aac:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ab0:	48 89 c7             	mov    %rax,%rdi
  400ab3:	e8 1c 00 00 00       	callq  400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab8:	c9                   	leaveq 
  400ab9:	c3                   	retq   

0000000000400aba <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400aba:	55                   	push   %rbp
  400abb:	48 89 e5             	mov    %rsp,%rbp
  400abe:	48 83 ec 10          	sub    $0x10,%rsp
  400ac2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ac6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aca:	48 89 c7             	mov    %rax,%rdi
  400acd:	e8 2c 00 00 00       	callq  400afe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad2:	c9                   	leaveq 
  400ad3:	c3                   	retq   

0000000000400ad4 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ad4:	55                   	push   %rbp
  400ad5:	48 89 e5             	mov    %rsp,%rbp
  400ad8:	48 83 ec 10          	sub    $0x10,%rsp
  400adc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ae4:	48 89 c7             	mov    %rax,%rdi
  400ae7:	e8 7c 00 00 00       	callq  400b68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aec:	c9                   	leaveq 
  400aed:	c3                   	retq   

0000000000400aee <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400aee:	55                   	push   %rbp
  400aef:	48 89 e5             	mov    %rsp,%rbp
  400af2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400afa:	90                   	nop
  400afb:	5d                   	pop    %rbp
  400afc:	c3                   	retq   
  400afd:	90                   	nop

0000000000400afe <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400afe:	55                   	push   %rbp
  400aff:	48 89 e5             	mov    %rsp,%rbp
  400b02:	48 83 ec 10          	sub    $0x10,%rsp
  400b06:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b0e:	48 89 c7             	mov    %rax,%rdi
  400b11:	e8 9c 00 00 00       	callq  400bb2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b16:	c9                   	leaveq 
  400b17:	c3                   	retq   

0000000000400b18 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b18:	55                   	push   %rbp
  400b19:	48 89 e5             	mov    %rsp,%rbp
  400b1c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b20:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b24:	90                   	nop
  400b25:	5d                   	pop    %rbp
  400b26:	c3                   	retq   
  400b27:	90                   	nop

0000000000400b28 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400b28:	55                   	push   %rbp
  400b29:	48 89 e5             	mov    %rsp,%rbp
  400b2c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b30:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b34:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  400b3b:	00 00 00 00 
  400b3f:	90                   	nop
  400b40:	5d                   	pop    %rbp
  400b41:	c3                   	retq   

0000000000400b42 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b42:	55                   	push   %rbp
  400b43:	48 89 e5             	mov    %rsp,%rbp
  400b46:	90                   	nop
  400b47:	5d                   	pop    %rbp
  400b48:	c3                   	retq   

0000000000400b49 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b49:	55                   	push   %rbp
  400b4a:	48 89 e5             	mov    %rsp,%rbp
  400b4d:	48 83 ec 10          	sub    $0x10,%rsp
  400b51:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b55:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b59:	48 83 c0 01          	add    $0x1,%rax
  400b5d:	48 89 c7             	mov    %rax,%rdi
  400b60:	e8 9d 00 00 00       	callq  400c02 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b65:	c9                   	leaveq 
  400b66:	c3                   	retq   
  400b67:	90                   	nop

0000000000400b68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b68:	55                   	push   %rbp
  400b69:	48 89 e5             	mov    %rsp,%rbp
  400b6c:	48 83 ec 28          	sub    $0x28,%rsp
  400b70:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400b74:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400b78:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b7c:	e8 15 fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b81:	48 83 c0 08          	add    $0x8,%rax
  400b85:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400b89:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b8d:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b91:	48 83 e8 04          	sub    $0x4,%rax
  400b95:	48 c1 e8 02          	shr    $0x2,%rax
  400b99:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400ba0:	00 
  400ba1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ba5:	48 01 d0             	add    %rdx,%rax
  400ba8:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bac:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bb0:	c9                   	leaveq 
  400bb1:	c3                   	retq   

0000000000400bb2 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb2:	55                   	push   %rbp
  400bb3:	48 89 e5             	mov    %rsp,%rbp
  400bb6:	48 83 ec 28          	sub    $0x28,%rsp
  400bba:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400bbe:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400bc2:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400bc6:	e8 cb f9 ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400bcb:	48 83 c0 08          	add    $0x8,%rax
  400bcf:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400bd3:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400bd7:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400bdb:	48 83 e8 04          	sub    $0x4,%rax
  400bdf:	48 c1 e8 02          	shr    $0x2,%rax
  400be3:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bea:	00 
  400beb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bef:	48 01 d0             	add    %rdx,%rax
  400bf2:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400bf8:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400bfc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c00:	c9                   	leaveq 
  400c01:	c3                   	retq   

0000000000400c02 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c02:	55                   	push   %rbp
  400c03:	48 89 e5             	mov    %rsp,%rbp
  400c06:	48 83 ec 18          	sub    $0x18,%rsp
  400c0a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c0e:	e8 83 f9 ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400c13:	48 83 c0 08          	add    $0x8,%rax
  400c17:	48 89 c2             	mov    %rax,%rdx
  400c1a:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400c1e:	48 c1 e0 02          	shl    $0x2,%rax
  400c22:	48 01 d0             	add    %rdx,%rax
  400c25:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400c29:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400c2d:	c9                   	leaveq 
  400c2e:	c3                   	retq   
  400c2f:	90                   	nop

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
