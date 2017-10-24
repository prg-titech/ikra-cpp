
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
  4004af:	49 c7 c0 00 0c 40 00 	mov    $0x400c00,%r8
  4004b6:	48 c7 c1 90 0b 40 00 	mov    $0x400b90,%rcx
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
  4005ae:	bf 04 00 00 00       	mov    $0x4,%edi
  4005b3:	e8 e2 03 00 00       	callq  40099a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005b8:	48 89 c3             	mov    %rax,%rbx
  4005bb:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c0:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005c4:	48 83 f8 03          	cmp    $0x3,%rax
  4005c8:	74 06                	je     4005d0 <main+0x3a>
  4005ca:	48 83 c0 01          	add    $0x1,%rax
  4005ce:	eb f0                	jmp    4005c0 <main+0x2a>
  4005d0:	48 89 df             	mov    %rbx,%rdi
  4005d3:	e8 06 03 00 00       	callq  4008de <_ZN9TestClassC1Ev>
  4005d8:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005dc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005e0:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4005e4:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005e8:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005ed:	48 89 ce             	mov    %rcx,%rsi
  4005f0:	48 89 c7             	mov    %rax,%rdi
  4005f3:	e8 f0 03 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4005f8:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4005fc:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400600:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400604:	ba 88 88 00 00       	mov    $0x8888,%edx
  400609:	48 89 ce             	mov    %rcx,%rsi
  40060c:	48 89 c7             	mov    %rax,%rdi
  40060f:	e8 14 04 00 00       	callq  400a28 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400614:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400618:	48 89 c7             	mov    %rax,%rdi
  40061b:	e8 fa 01 00 00       	callq  40081a <_ZN9TestClass15increase_field0Ev>
  400620:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400624:	48 89 c7             	mov    %rax,%rdi
  400627:	e8 3c 02 00 00       	callq  400868 <_ZN9TestClass15increase_field1Ev>
  40062c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400630:	48 83 c0 04          	add    $0x4,%rax
  400634:	48 89 c7             	mov    %rax,%rdi
  400637:	e8 46 04 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40063c:	8b 18                	mov    (%rax),%ebx
  40063e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400642:	48 83 c0 04          	add    $0x4,%rax
  400646:	48 89 c7             	mov    %rax,%rdi
  400649:	e8 1a 04 00 00       	callq  400a68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40064e:	8b 00                	mov    (%rax),%eax
  400650:	89 da                	mov    %ebx,%edx
  400652:	89 c6                	mov    %eax,%esi
  400654:	bf 20 0c 40 00       	mov    $0x400c20,%edi
  400659:	b8 00 00 00 00       	mov    $0x0,%eax
  40065e:	e8 0d fe ff ff       	callq  400470 <printf@plt>
  400663:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400667:	48 83 c0 04          	add    $0x4,%rax
  40066b:	48 89 c7             	mov    %rax,%rdi
  40066e:	e8 f5 03 00 00       	callq  400a68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400673:	8b 00                	mov    (%rax),%eax
  400675:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40067a:	75 19                	jne    400695 <main+0xff>
  40067c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400680:	48 83 c0 04          	add    $0x4,%rax
  400684:	48 89 c7             	mov    %rax,%rdi
  400687:	e8 f6 03 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40068c:	8b 00                	mov    (%rax),%eax
  40068e:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400693:	74 07                	je     40069c <main+0x106>
  400695:	b8 01 00 00 00       	mov    $0x1,%eax
  40069a:	eb 05                	jmp    4006a1 <main+0x10b>
  40069c:	b8 00 00 00 00       	mov    $0x0,%eax
  4006a1:	0f b6 c0             	movzbl %al,%eax
  4006a4:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4006a8:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  4006af:	00 00 
  4006b1:	74 05                	je     4006b8 <main+0x122>
  4006b3:	e8 a8 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  4006b8:	48 83 c4 28          	add    $0x28,%rsp
  4006bc:	5b                   	pop    %rbx
  4006bd:	5d                   	pop    %rbp
  4006be:	c3                   	retq   

00000000004006bf <_Z12new_instancev>:
  4006bf:	55                   	push   %rbp
  4006c0:	48 89 e5             	mov    %rsp,%rbp
  4006c3:	53                   	push   %rbx
  4006c4:	48 83 ec 08          	sub    $0x8,%rsp
  4006c8:	bf 04 00 00 00       	mov    $0x4,%edi
  4006cd:	e8 c8 02 00 00       	callq  40099a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006d2:	48 89 c3             	mov    %rax,%rbx
  4006d5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006da:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006de:	48 83 f8 03          	cmp    $0x3,%rax
  4006e2:	74 06                	je     4006ea <_Z12new_instancev+0x2b>
  4006e4:	48 83 c0 01          	add    $0x1,%rax
  4006e8:	eb f0                	jmp    4006da <_Z12new_instancev+0x1b>
  4006ea:	48 89 df             	mov    %rbx,%rdi
  4006ed:	e8 ec 01 00 00       	callq  4008de <_ZN9TestClassC1Ev>
  4006f2:	48 89 d8             	mov    %rbx,%rax
  4006f5:	48 83 c4 08          	add    $0x8,%rsp
  4006f9:	5b                   	pop    %rbx
  4006fa:	5d                   	pop    %rbp
  4006fb:	c3                   	retq   

00000000004006fc <_Z12write_field0P9TestClass>:
  4006fc:	55                   	push   %rbp
  4006fd:	48 89 e5             	mov    %rsp,%rbp
  400700:	48 83 ec 30          	sub    $0x30,%rsp
  400704:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400708:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40070f:	00 00 
  400711:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400715:	31 c0                	xor    %eax,%eax
  400717:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40071b:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40071f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400723:	ba 77 77 00 00       	mov    $0x7777,%edx
  400728:	48 89 ce             	mov    %rcx,%rsi
  40072b:	48 89 c7             	mov    %rax,%rdi
  40072e:	e8 b5 02 00 00       	callq  4009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400733:	90                   	nop
  400734:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400738:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40073f:	00 00 
  400741:	74 05                	je     400748 <_Z12write_field0P9TestClass+0x4c>
  400743:	e8 18 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400748:	c9                   	leaveq 
  400749:	c3                   	retq   

000000000040074a <_Z11read_field0P9TestClass>:
  40074a:	55                   	push   %rbp
  40074b:	48 89 e5             	mov    %rsp,%rbp
  40074e:	48 83 ec 10          	sub    $0x10,%rsp
  400752:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400756:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40075a:	48 83 c0 04          	add    $0x4,%rax
  40075e:	48 89 c7             	mov    %rax,%rdi
  400761:	e8 02 03 00 00       	callq  400a68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400766:	8b 00                	mov    (%rax),%eax
  400768:	c9                   	leaveq 
  400769:	c3                   	retq   

000000000040076a <_Z12write_field1P9TestClass>:
  40076a:	55                   	push   %rbp
  40076b:	48 89 e5             	mov    %rsp,%rbp
  40076e:	48 83 ec 30          	sub    $0x30,%rsp
  400772:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400776:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40077d:	00 00 
  40077f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400783:	31 c0                	xor    %eax,%eax
  400785:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400789:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40078d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400791:	ba 77 77 00 00       	mov    $0x7777,%edx
  400796:	48 89 ce             	mov    %rcx,%rsi
  400799:	48 89 c7             	mov    %rax,%rdi
  40079c:	e8 87 02 00 00       	callq  400a28 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007a1:	90                   	nop
  4007a2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007a6:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007ad:	00 00 
  4007af:	74 05                	je     4007b6 <_Z12write_field1P9TestClass+0x4c>
  4007b1:	e8 aa fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007b6:	c9                   	leaveq 
  4007b7:	c3                   	retq   

00000000004007b8 <_Z11read_field1P9TestClass>:
  4007b8:	55                   	push   %rbp
  4007b9:	48 89 e5             	mov    %rsp,%rbp
  4007bc:	48 83 ec 10          	sub    $0x10,%rsp
  4007c0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007c8:	48 83 c0 04          	add    $0x4,%rax
  4007cc:	48 89 c7             	mov    %rax,%rdi
  4007cf:	e8 ae 02 00 00       	callq  400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007d4:	8b 00                	mov    (%rax),%eax
  4007d6:	c9                   	leaveq 
  4007d7:	c3                   	retq   

00000000004007d8 <_Z21explicit_write_field0m>:
  4007d8:	55                   	push   %rbp
  4007d9:	48 89 e5             	mov    %rsp,%rbp
  4007dc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007e0:	48 8b 05 21 aa 20 00 	mov    0x20aa21(%rip),%rax        # 60b208 <explicit_field0>
  4007e7:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  4007eb:	48 c1 e2 02          	shl    $0x2,%rdx
  4007ef:	48 01 d0             	add    %rdx,%rax
  4007f2:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4007f8:	90                   	nop
  4007f9:	5d                   	pop    %rbp
  4007fa:	c3                   	retq   

00000000004007fb <_Z20explicit_read_field0m>:
  4007fb:	55                   	push   %rbp
  4007fc:	48 89 e5             	mov    %rsp,%rbp
  4007ff:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400803:	48 8b 05 fe a9 20 00 	mov    0x20a9fe(%rip),%rax        # 60b208 <explicit_field0>
  40080a:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  40080e:	48 c1 e2 02          	shl    $0x2,%rdx
  400812:	48 01 d0             	add    %rdx,%rax
  400815:	8b 00                	mov    (%rax),%eax
  400817:	5d                   	pop    %rbp
  400818:	c3                   	retq   
  400819:	90                   	nop

000000000040081a <_ZN9TestClass15increase_field0Ev>:
  40081a:	55                   	push   %rbp
  40081b:	48 89 e5             	mov    %rsp,%rbp
  40081e:	48 83 ec 30          	sub    $0x30,%rsp
  400822:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400826:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40082d:	00 00 
  40082f:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400833:	31 c0                	xor    %eax,%eax
  400835:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400839:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40083d:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400841:	ba 55 55 00 00       	mov    $0x5555,%edx
  400846:	48 89 ce             	mov    %rcx,%rsi
  400849:	48 89 c7             	mov    %rax,%rdi
  40084c:	e8 c9 00 00 00       	callq  40091a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400851:	90                   	nop
  400852:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400856:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40085d:	00 00 
  40085f:	74 05                	je     400866 <_ZN9TestClass15increase_field0Ev+0x4c>
  400861:	e8 fa fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400866:	c9                   	leaveq 
  400867:	c3                   	retq   

0000000000400868 <_ZN9TestClass15increase_field1Ev>:
  400868:	55                   	push   %rbp
  400869:	48 89 e5             	mov    %rsp,%rbp
  40086c:	48 83 ec 30          	sub    $0x30,%rsp
  400870:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400874:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40087b:	00 00 
  40087d:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400881:	31 c0                	xor    %eax,%eax
  400883:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400887:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40088b:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40088f:	ba 44 44 00 00       	mov    $0x4444,%edx
  400894:	48 89 ce             	mov    %rcx,%rsi
  400897:	48 89 c7             	mov    %rax,%rdi
  40089a:	e8 bb 00 00 00       	callq  40095a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  40089f:	90                   	nop
  4008a0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a4:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4008ab:	00 00 
  4008ad:	74 05                	je     4008b4 <_ZN9TestClass15increase_field1Ev+0x4c>
  4008af:	e8 ac fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  4008b4:	c9                   	leaveq 
  4008b5:	c3                   	retq   

00000000004008b6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008b6:	55                   	push   %rbp
  4008b7:	48 89 e5             	mov    %rsp,%rbp
  4008ba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008be:	90                   	nop
  4008bf:	5d                   	pop    %rbp
  4008c0:	c3                   	retq   
  4008c1:	90                   	nop

00000000004008c2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  4008c2:	55                   	push   %rbp
  4008c3:	48 89 e5             	mov    %rsp,%rbp
  4008c6:	48 83 ec 10          	sub    $0x10,%rsp
  4008ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ce:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008d2:	48 89 c7             	mov    %rax,%rdi
  4008d5:	e8 dc ff ff ff       	callq  4008b6 <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008da:	90                   	nop
  4008db:	c9                   	leaveq 
  4008dc:	c3                   	retq   
  4008dd:	90                   	nop

00000000004008de <_ZN9TestClassC1Ev>:
  4008de:	55                   	push   %rbp
  4008df:	48 89 e5             	mov    %rsp,%rbp
  4008e2:	48 83 ec 10          	sub    $0x10,%rsp
  4008e6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008ea:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008ee:	48 89 c7             	mov    %rax,%rdi
  4008f1:	e8 cc ff ff ff       	callq  4008c2 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  4008f6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008fa:	48 83 c0 04          	add    $0x4,%rax
  4008fe:	48 89 c7             	mov    %rax,%rdi
  400901:	e8 c0 00 00 00       	callq  4009c6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>
  400906:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40090a:	48 83 c0 04          	add    $0x4,%rax
  40090e:	48 89 c7             	mov    %rax,%rdi
  400911:	e8 bc 00 00 00       	callq  4009d2 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>
  400916:	90                   	nop
  400917:	c9                   	leaveq 
  400918:	c3                   	retq   
  400919:	90                   	nop

000000000040091a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  40091a:	55                   	push   %rbp
  40091b:	48 89 e5             	mov    %rsp,%rbp
  40091e:	48 83 ec 20          	sub    $0x20,%rsp
  400922:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400926:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40092a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40092d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400931:	48 89 c7             	mov    %rax,%rdi
  400934:	e8 63 01 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400939:	8b 10                	mov    (%rax),%edx
  40093b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40093f:	89 10                	mov    %edx,(%rax)
  400941:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400945:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400949:	48 89 d6             	mov    %rdx,%rsi
  40094c:	48 89 c7             	mov    %rax,%rdi
  40094f:	e8 8a 01 00 00       	callq  400ade <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400954:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400958:	c9                   	leaveq 
  400959:	c3                   	retq   

000000000040095a <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  40095a:	55                   	push   %rbp
  40095b:	48 89 e5             	mov    %rsp,%rbp
  40095e:	48 83 ec 20          	sub    $0x20,%rsp
  400962:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400966:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  40096a:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40096d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400971:	48 89 c7             	mov    %rax,%rdi
  400974:	e8 75 01 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400979:	8b 10                	mov    (%rax),%edx
  40097b:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40097f:	89 10                	mov    %edx,(%rax)
  400981:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400985:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400989:	48 89 d6             	mov    %rdx,%rsi
  40098c:	48 89 c7             	mov    %rax,%rdi
  40098f:	e8 a2 01 00 00       	callq  400b36 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400994:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400998:	c9                   	leaveq 
  400999:	c3                   	retq   

000000000040099a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  40099a:	55                   	push   %rbp
  40099b:	48 89 e5             	mov    %rsp,%rbp
  40099e:	48 83 ec 10          	sub    $0x10,%rsp
  4009a2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a6:	e8 9a 01 00 00       	callq  400b45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009ab:	8b 05 af 16 20 00    	mov    0x2016af(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  4009b1:	8d 50 01             	lea    0x1(%rax),%edx
  4009b4:	89 15 a6 16 20 00    	mov    %edx,0x2016a6(%rip)        # 602060 <_ZN9TestClass7storageE>
  4009ba:	89 c0                	mov    %eax,%eax
  4009bc:	48 89 c7             	mov    %rax,%rdi
  4009bf:	e8 88 01 00 00       	callq  400b4c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009c4:	c9                   	leaveq 
  4009c5:	c3                   	retq   

00000000004009c6 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1Ev>:
  4009c6:	55                   	push   %rbp
  4009c7:	48 89 e5             	mov    %rsp,%rbp
  4009ca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ce:	90                   	nop
  4009cf:	5d                   	pop    %rbp
  4009d0:	c3                   	retq   
  4009d1:	90                   	nop

00000000004009d2 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1Ev>:
  4009d2:	55                   	push   %rbp
  4009d3:	48 89 e5             	mov    %rsp,%rbp
  4009d6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009da:	90                   	nop
  4009db:	5d                   	pop    %rbp
  4009dc:	c3                   	retq   

00000000004009dd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  4009dd:	55                   	push   %rbp
  4009de:	48 89 e5             	mov    %rsp,%rbp
  4009e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e5:	90                   	nop
  4009e6:	5d                   	pop    %rbp
  4009e7:	c3                   	retq   

00000000004009e8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  4009e8:	55                   	push   %rbp
  4009e9:	48 89 e5             	mov    %rsp,%rbp
  4009ec:	48 83 ec 20          	sub    $0x20,%rsp
  4009f0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009f8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009fb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009ff:	48 89 c7             	mov    %rax,%rdi
  400a02:	e8 95 00 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a07:	48 89 c2             	mov    %rax,%rdx
  400a0a:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a0d:	89 02                	mov    %eax,(%rdx)
  400a0f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a13:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a17:	48 89 d6             	mov    %rdx,%rsi
  400a1a:	48 89 c7             	mov    %rax,%rdi
  400a1d:	e8 bc 00 00 00       	callq  400ade <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>
  400a22:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a26:	c9                   	leaveq 
  400a27:	c3                   	retq   

0000000000400a28 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400a28:	55                   	push   %rbp
  400a29:	48 89 e5             	mov    %rsp,%rbp
  400a2c:	48 83 ec 20          	sub    $0x20,%rsp
  400a30:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a34:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a38:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a3b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a3f:	48 89 c7             	mov    %rax,%rdi
  400a42:	e8 a7 00 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a47:	48 89 c2             	mov    %rax,%rdx
  400a4a:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a4d:	89 02                	mov    %eax,(%rdx)
  400a4f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a53:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a57:	48 89 d6             	mov    %rdx,%rsi
  400a5a:	48 89 c7             	mov    %rax,%rdi
  400a5d:	e8 d4 00 00 00       	callq  400b36 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>
  400a62:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a66:	c9                   	leaveq 
  400a67:	c3                   	retq   

0000000000400a68 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400a68:	55                   	push   %rbp
  400a69:	48 89 e5             	mov    %rsp,%rbp
  400a6c:	48 83 ec 10          	sub    $0x10,%rsp
  400a70:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a74:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a78:	48 89 c7             	mov    %rax,%rdi
  400a7b:	e8 1c 00 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a80:	c9                   	leaveq 
  400a81:	c3                   	retq   

0000000000400a82 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400a82:	55                   	push   %rbp
  400a83:	48 89 e5             	mov    %rsp,%rbp
  400a86:	48 83 ec 10          	sub    $0x10,%rsp
  400a8a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a92:	48 89 c7             	mov    %rax,%rdi
  400a95:	e8 54 00 00 00       	callq  400aee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a9a:	c9                   	leaveq 
  400a9b:	c3                   	retq   

0000000000400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a9c:	55                   	push   %rbp
  400a9d:	48 89 e5             	mov    %rsp,%rbp
  400aa0:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400aa4:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400aa8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400aac:	48 c7 45 f0 64 20 60 	movq   $0x602064,-0x10(%rbp)
  400ab3:	00 
  400ab4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400ab8:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400abc:	48 83 e8 04          	sub    $0x4,%rax
  400ac0:	48 c1 e8 02          	shr    $0x2,%rax
  400ac4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400acb:	00 
  400acc:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ad0:	48 01 d0             	add    %rdx,%rax
  400ad3:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400ad7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400adb:	5d                   	pop    %rbp
  400adc:	c3                   	retq   
  400add:	90                   	nop

0000000000400ade <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC1ERKS3_>:
  400ade:	55                   	push   %rbp
  400adf:	48 89 e5             	mov    %rsp,%rbp
  400ae2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae6:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aea:	90                   	nop
  400aeb:	5d                   	pop    %rbp
  400aec:	c3                   	retq   
  400aed:	90                   	nop

0000000000400aee <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400aee:	55                   	push   %rbp
  400aef:	48 89 e5             	mov    %rsp,%rbp
  400af2:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400af6:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400afa:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400afe:	48 c7 45 f0 64 20 60 	movq   $0x602064,-0x10(%rbp)
  400b05:	00 
  400b06:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b0a:	48 2b 45 f0          	sub    -0x10(%rbp),%rax
  400b0e:	48 83 e8 04          	sub    $0x4,%rax
  400b12:	48 c1 e8 02          	shr    $0x2,%rax
  400b16:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400b1d:	00 
  400b1e:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b22:	48 01 d0             	add    %rdx,%rax
  400b25:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400b2b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b2f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b33:	5d                   	pop    %rbp
  400b34:	c3                   	retq   
  400b35:	90                   	nop

0000000000400b36 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC1ERKS3_>:
  400b36:	55                   	push   %rbp
  400b37:	48 89 e5             	mov    %rsp,%rbp
  400b3a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b3e:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b42:	90                   	nop
  400b43:	5d                   	pop    %rbp
  400b44:	c3                   	retq   

0000000000400b45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b45:	55                   	push   %rbp
  400b46:	48 89 e5             	mov    %rsp,%rbp
  400b49:	90                   	nop
  400b4a:	5d                   	pop    %rbp
  400b4b:	c3                   	retq   

0000000000400b4c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b4c:	55                   	push   %rbp
  400b4d:	48 89 e5             	mov    %rsp,%rbp
  400b50:	48 83 ec 10          	sub    $0x10,%rsp
  400b54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b58:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b5c:	48 89 c7             	mov    %rax,%rdi
  400b5f:	e8 02 00 00 00       	callq  400b66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b64:	c9                   	leaveq 
  400b65:	c3                   	retq   

0000000000400b66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b66:	55                   	push   %rbp
  400b67:	48 89 e5             	mov    %rsp,%rbp
  400b6a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b6e:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b72:	48 c1 e0 02          	shl    $0x2,%rax
  400b76:	ba 64 20 60 00       	mov    $0x602064,%edx
  400b7b:	48 01 d0             	add    %rdx,%rax
  400b7e:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400b82:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b86:	5d                   	pop    %rbp
  400b87:	c3                   	retq   
  400b88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b8f:	00 

0000000000400b90 <__libc_csu_init>:
  400b90:	41 57                	push   %r15
  400b92:	41 56                	push   %r14
  400b94:	41 89 ff             	mov    %edi,%r15d
  400b97:	41 55                	push   %r13
  400b99:	41 54                	push   %r12
  400b9b:	4c 8d 25 6e 12 20 00 	lea    0x20126e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400ba2:	55                   	push   %rbp
  400ba3:	48 8d 2d 6e 12 20 00 	lea    0x20126e(%rip),%rbp        # 601e18 <__init_array_end>
  400baa:	53                   	push   %rbx
  400bab:	49 89 f6             	mov    %rsi,%r14
  400bae:	49 89 d5             	mov    %rdx,%r13
  400bb1:	4c 29 e5             	sub    %r12,%rbp
  400bb4:	48 83 ec 08          	sub    $0x8,%rsp
  400bb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400bbc:	e8 67 f8 ff ff       	callq  400428 <_init>
  400bc1:	48 85 ed             	test   %rbp,%rbp
  400bc4:	74 20                	je     400be6 <__libc_csu_init+0x56>
  400bc6:	31 db                	xor    %ebx,%ebx
  400bc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400bcf:	00 
  400bd0:	4c 89 ea             	mov    %r13,%rdx
  400bd3:	4c 89 f6             	mov    %r14,%rsi
  400bd6:	44 89 ff             	mov    %r15d,%edi
  400bd9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400bdd:	48 83 c3 01          	add    $0x1,%rbx
  400be1:	48 39 eb             	cmp    %rbp,%rbx
  400be4:	75 ea                	jne    400bd0 <__libc_csu_init+0x40>
  400be6:	48 83 c4 08          	add    $0x8,%rsp
  400bea:	5b                   	pop    %rbx
  400beb:	5d                   	pop    %rbp
  400bec:	41 5c                	pop    %r12
  400bee:	41 5d                	pop    %r13
  400bf0:	41 5e                	pop    %r14
  400bf2:	41 5f                	pop    %r15
  400bf4:	c3                   	retq   
  400bf5:	90                   	nop
  400bf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bfd:	00 00 00 

0000000000400c00 <__libc_csu_fini>:
  400c00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c04 <_fini>:
  400c04:	48 83 ec 08          	sub    $0x8,%rsp
  400c08:	48 83 c4 08          	add    $0x8,%rsp
  400c0c:	c3                   	retq   
