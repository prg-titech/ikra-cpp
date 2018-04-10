
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
  400596:	53                   	push   %rbx
  400597:	48 83 ec 10          	sub    $0x10,%rsp
  40059b:	e8 a9 03 00 00       	callq  400949 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005a0:	bf 00 00 00 00       	mov    $0x0,%edi
  4005a5:	e8 c8 03 00 00       	callq  400972 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005aa:	48 89 c3             	mov    %rax,%rbx
  4005ad:	48 89 df             	mov    %rbx,%rdi
  4005b0:	e8 c1 02 00 00       	callq  400876 <_ZN9TestClassC1Ev>
  4005b5:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4005ba:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005bf:	be 77 77 00 00       	mov    $0x7777,%esi
  4005c4:	48 89 c7             	mov    %rax,%rdi
  4005c7:	e8 ea 03 00 00       	callq  4009b6 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>
  4005cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005d1:	be 88 88 00 00       	mov    $0x8888,%esi
  4005d6:	48 89 c7             	mov    %rax,%rdi
  4005d9:	e8 06 04 00 00       	callq  4009e4 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>
  4005de:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005e3:	48 89 c7             	mov    %rax,%rdi
  4005e6:	e8 99 01 00 00       	callq  400784 <_ZN9TestClass15increase_field0Ev>
  4005eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005f0:	48 89 c7             	mov    %rax,%rdi
  4005f3:	e8 f0 01 00 00       	callq  4007e8 <_ZN9TestClass15increase_field1Ev>
  4005f8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005fd:	48 89 c7             	mov    %rax,%rdi
  400600:	e8 29 04 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
  400605:	8b 18                	mov    (%rax),%ebx
  400607:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40060c:	48 89 c7             	mov    %rax,%rdi
  40060f:	e8 fe 03 00 00       	callq  400a12 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  400614:	8b 00                	mov    (%rax),%eax
  400616:	89 da                	mov    %ebx,%edx
  400618:	89 c6                	mov    %eax,%esi
  40061a:	bf 2c 0c 40 00       	mov    $0x400c2c,%edi
  40061f:	b8 00 00 00 00       	mov    $0x0,%eax
  400624:	e8 47 fe ff ff       	callq  400470 <printf@plt>
  400629:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40062e:	48 89 c7             	mov    %rax,%rdi
  400631:	e8 dc 03 00 00       	callq  400a12 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  400636:	8b 00                	mov    (%rax),%eax
  400638:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40063d:	75 16                	jne    400655 <main+0xbf>
  40063f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400644:	48 89 c7             	mov    %rax,%rdi
  400647:	e8 e2 03 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
  40064c:	8b 00                	mov    (%rax),%eax
  40064e:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400653:	74 07                	je     40065c <main+0xc6>
  400655:	b8 01 00 00 00       	mov    $0x1,%eax
  40065a:	eb 05                	jmp    400661 <main+0xcb>
  40065c:	b8 00 00 00 00       	mov    $0x0,%eax
  400661:	0f b6 c0             	movzbl %al,%eax
  400664:	48 83 c4 10          	add    $0x10,%rsp
  400668:	5b                   	pop    %rbx
  400669:	c3                   	retq   

000000000040066a <_Z12new_instancev>:
  40066a:	53                   	push   %rbx
  40066b:	bf 00 00 00 00       	mov    $0x0,%edi
  400670:	e8 fd 02 00 00       	callq  400972 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400675:	48 89 c3             	mov    %rax,%rbx
  400678:	48 89 df             	mov    %rbx,%rdi
  40067b:	e8 f6 01 00 00       	callq  400876 <_ZN9TestClassC1Ev>
  400680:	48 89 d8             	mov    %rbx,%rax
  400683:	5b                   	pop    %rbx
  400684:	c3                   	retq   

0000000000400685 <_Z12write_field0P9TestClass>:
  400685:	48 83 ec 18          	sub    $0x18,%rsp
  400689:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40068e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400693:	be 77 77 00 00       	mov    $0x7777,%esi
  400698:	48 89 c7             	mov    %rax,%rdi
  40069b:	e8 16 03 00 00       	callq  4009b6 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>
  4006a0:	90                   	nop
  4006a1:	48 83 c4 18          	add    $0x18,%rsp
  4006a5:	c3                   	retq   

00000000004006a6 <_Z11read_field0P9TestClass>:
  4006a6:	48 83 ec 18          	sub    $0x18,%rsp
  4006aa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006af:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006b4:	48 89 c7             	mov    %rax,%rdi
  4006b7:	e8 56 03 00 00       	callq  400a12 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  4006bc:	8b 00                	mov    (%rax),%eax
  4006be:	48 83 c4 18          	add    $0x18,%rsp
  4006c2:	c3                   	retq   

00000000004006c3 <_Z12write_field1P9TestClass>:
  4006c3:	48 83 ec 18          	sub    $0x18,%rsp
  4006c7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006d1:	be 77 77 00 00       	mov    $0x7777,%esi
  4006d6:	48 89 c7             	mov    %rax,%rdi
  4006d9:	e8 06 03 00 00       	callq  4009e4 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>
  4006de:	90                   	nop
  4006df:	48 83 c4 18          	add    $0x18,%rsp
  4006e3:	c3                   	retq   

00000000004006e4 <_Z11read_field1P9TestClass>:
  4006e4:	48 83 ec 18          	sub    $0x18,%rsp
  4006e8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006ed:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006f2:	48 89 c7             	mov    %rax,%rdi
  4006f5:	e8 34 03 00 00       	callq  400a2e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
  4006fa:	8b 00                	mov    (%rax),%eax
  4006fc:	48 83 c4 18          	add    $0x18,%rsp
  400700:	c3                   	retq   

0000000000400701 <_Z21explicit_write_field0m>:
  400701:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400706:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40070b:	c7 04 85 40 b2 60 00 	movl   $0x7777,0x60b240(,%rax,4)
  400712:	77 77 00 00 
  400716:	90                   	nop
  400717:	c3                   	retq   

0000000000400718 <_Z20explicit_read_field0m>:
  400718:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40071d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400722:	8b 04 85 40 b2 60 00 	mov    0x60b240(,%rax,4),%eax
  400729:	c3                   	retq   

000000000040072a <_ZnwmPv>:
  40072a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40072f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400734:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400739:	c3                   	retq   

000000000040073a <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  40073a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40073f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400744:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  40074b:	90                   	nop
  40074c:	c3                   	retq   
  40074d:	90                   	nop

000000000040074e <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  40074e:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400753:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400758:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  40075d:	48 8b 00             	mov    (%rax),%rax
  400760:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400765:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  40076a:	48 8b 10             	mov    (%rax),%rdx
  40076d:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400772:	48 01 c2             	add    %rax,%rdx
  400775:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  40077a:	48 89 10             	mov    %rdx,(%rax)
  40077d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400782:	c3                   	retq   
  400783:	90                   	nop

0000000000400784 <_ZN9TestClass15increase_field0Ev>:
  400784:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400789:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40078d:	41 ff 72 f8          	pushq  -0x8(%r10)
  400791:	55                   	push   %rbp
  400792:	48 89 e5             	mov    %rsp,%rbp
  400795:	41 52                	push   %r10
  400797:	48 83 ec 48          	sub    $0x48,%rsp
  40079b:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  40079f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007a6:	00 00 
  4007a8:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007ac:	31 c0                	xor    %eax,%eax
  4007ae:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4007b2:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4007b6:	ba 55 55 00 00       	mov    $0x5555,%edx
  4007bb:	48 89 ce             	mov    %rcx,%rsi
  4007be:	48 89 c7             	mov    %rax,%rdi
  4007c1:	e8 e6 00 00 00       	callq  4008ac <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEmLEi>
  4007c6:	90                   	nop
  4007c7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4007cb:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4007d2:	00 00 
  4007d4:	74 05                	je     4007db <_ZN9TestClass15increase_field0Ev+0x57>
  4007d6:	e8 85 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  4007db:	48 83 c4 48          	add    $0x48,%rsp
  4007df:	41 5a                	pop    %r10
  4007e1:	5d                   	pop    %rbp
  4007e2:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4007e6:	c3                   	retq   
  4007e7:	90                   	nop

00000000004007e8 <_ZN9TestClass15increase_field1Ev>:
  4007e8:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007ed:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007f1:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007f5:	55                   	push   %rbp
  4007f6:	48 89 e5             	mov    %rsp,%rbp
  4007f9:	41 52                	push   %r10
  4007fb:	48 83 ec 48          	sub    $0x48,%rsp
  4007ff:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400803:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40080a:	00 00 
  40080c:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400810:	31 c0                	xor    %eax,%eax
  400812:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400816:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40081a:	ba 44 44 00 00       	mov    $0x4444,%edx
  40081f:	48 89 ce             	mov    %rcx,%rsi
  400822:	48 89 c7             	mov    %rax,%rdi
  400825:	e8 ca 00 00 00       	callq  4008f4 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEmLEi>
  40082a:	90                   	nop
  40082b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40082f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400836:	00 00 
  400838:	74 05                	je     40083f <_ZN9TestClass15increase_field1Ev+0x57>
  40083a:	e8 21 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  40083f:	48 83 c4 48          	add    $0x48,%rsp
  400843:	41 5a                	pop    %r10
  400845:	5d                   	pop    %rbp
  400846:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  40084a:	c3                   	retq   
  40084b:	90                   	nop

000000000040084c <_ZN9TestClass14storage_bufferEv>:
  40084c:	b8 60 20 60 00       	mov    $0x602060,%eax
  400851:	c3                   	retq   

0000000000400852 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400852:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400857:	90                   	nop
  400858:	c3                   	retq   
  400859:	90                   	nop

000000000040085a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  40085a:	48 83 ec 18          	sub    $0x18,%rsp
  40085e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400863:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400868:	48 89 c7             	mov    %rax,%rdi
  40086b:	e8 e2 ff ff ff       	callq  400852 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400870:	90                   	nop
  400871:	48 83 c4 18          	add    $0x18,%rsp
  400875:	c3                   	retq   

0000000000400876 <_ZN9TestClassC1Ev>:
  400876:	48 83 ec 18          	sub    $0x18,%rsp
  40087a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40087f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400884:	48 89 c7             	mov    %rax,%rdi
  400887:	e8 ce ff ff ff       	callq  40085a <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  40088c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400891:	48 89 c7             	mov    %rax,%rdi
  400894:	e8 07 01 00 00       	callq  4009a0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1Ev>
  400899:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40089e:	48 89 c7             	mov    %rax,%rdi
  4008a1:	e8 02 01 00 00       	callq  4009a8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1Ev>
  4008a6:	90                   	nop
  4008a7:	48 83 c4 18          	add    $0x18,%rsp
  4008ab:	c3                   	retq   

00000000004008ac <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEmLEi>:
  4008ac:	48 83 ec 28          	sub    $0x28,%rsp
  4008b0:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008b5:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4008ba:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4008be:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4008c3:	48 89 c7             	mov    %rax,%rdi
  4008c6:	e8 7f 01 00 00       	callq  400a4a <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008cb:	8b 10                	mov    (%rax),%edx
  4008cd:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  4008d2:	89 10                	mov    %edx,(%rax)
  4008d4:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4008d9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4008de:	48 89 d6             	mov    %rdx,%rsi
  4008e1:	48 89 c7             	mov    %rax,%rdi
  4008e4:	e8 7d 01 00 00       	callq  400a66 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1ERKS3_>
  4008e9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4008ee:	48 83 c4 28          	add    $0x28,%rsp
  4008f2:	c3                   	retq   
  4008f3:	90                   	nop

00000000004008f4 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEmLEi>:
  4008f4:	48 83 ec 28          	sub    $0x28,%rsp
  4008f8:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4008fd:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400902:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400906:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40090b:	48 89 c7             	mov    %rax,%rdi
  40090e:	e8 5f 01 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400913:	8b 10                	mov    (%rax),%edx
  400915:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  40091a:	89 10                	mov    %edx,(%rax)
  40091c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400921:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400926:	48 89 d6             	mov    %rdx,%rsi
  400929:	48 89 c7             	mov    %rax,%rdi
  40092c:	e8 5d 01 00 00       	callq  400a8e <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1ERKS3_>
  400931:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400936:	48 83 c4 28          	add    $0x28,%rsp
  40093a:	c3                   	retq   

000000000040093b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>:
  40093b:	48 83 ec 08          	sub    $0x8,%rsp
  40093f:	e8 08 ff ff ff       	callq  40084c <_ZN9TestClass14storage_bufferEv>
  400944:	48 83 c4 08          	add    $0x8,%rsp
  400948:	c3                   	retq   

0000000000400949 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  400949:	48 83 ec 08          	sub    $0x8,%rsp
  40094d:	e8 e9 ff ff ff       	callq  40093b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400952:	48 89 c6             	mov    %rax,%rsi
  400955:	bf d0 91 00 00       	mov    $0x91d0,%edi
  40095a:	e8 cb fd ff ff       	callq  40072a <_ZnwmPv>
  40095f:	48 85 c0             	test   %rax,%rax
  400962:	74 08                	je     40096c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400964:	48 89 c7             	mov    %rax,%rdi
  400967:	e8 4a 01 00 00       	callq  400ab6 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40096c:	90                   	nop
  40096d:	48 83 c4 08          	add    $0x8,%rsp
  400971:	c3                   	retq   

0000000000400972 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400972:	48 83 ec 18          	sub    $0x18,%rsp
  400976:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40097b:	e8 5e 01 00 00       	callq  400ade <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400980:	e8 b6 ff ff ff       	callq  40093b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400985:	be 01 00 00 00       	mov    $0x1,%esi
  40098a:	48 89 c7             	mov    %rax,%rdi
  40098d:	e8 bc fd ff ff       	callq  40074e <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400992:	48 89 c7             	mov    %rax,%rdi
  400995:	e8 46 01 00 00       	callq  400ae0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  40099a:	48 83 c4 18          	add    $0x18,%rsp
  40099e:	c3                   	retq   
  40099f:	90                   	nop

00000000004009a0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1Ev>:
  4009a0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a5:	90                   	nop
  4009a6:	c3                   	retq   
  4009a7:	90                   	nop

00000000004009a8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1Ev>:
  4009a8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009ad:	90                   	nop
  4009ae:	c3                   	retq   

00000000004009af <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009af:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b4:	90                   	nop
  4009b5:	c3                   	retq   

00000000004009b6 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>:
  4009b6:	48 83 ec 18          	sub    $0x18,%rsp
  4009ba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009bf:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4009c3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009c8:	48 89 c7             	mov    %rax,%rdi
  4009cb:	e8 7a 00 00 00       	callq  400a4a <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009d0:	48 89 c2             	mov    %rax,%rdx
  4009d3:	8b 44 24 04          	mov    0x4(%rsp),%eax
  4009d7:	89 02                	mov    %eax,(%rdx)
  4009d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009de:	48 83 c4 18          	add    $0x18,%rsp
  4009e2:	c3                   	retq   
  4009e3:	90                   	nop

00000000004009e4 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>:
  4009e4:	48 83 ec 18          	sub    $0x18,%rsp
  4009e8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009ed:	89 74 24 04          	mov    %esi,0x4(%rsp)
  4009f1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009f6:	48 89 c7             	mov    %rax,%rdi
  4009f9:	e8 74 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009fe:	48 89 c2             	mov    %rax,%rdx
  400a01:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400a05:	89 02                	mov    %eax,(%rdx)
  400a07:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a0c:	48 83 c4 18          	add    $0x18,%rsp
  400a10:	c3                   	retq   
  400a11:	90                   	nop

0000000000400a12 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>:
  400a12:	48 83 ec 18          	sub    $0x18,%rsp
  400a16:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a1b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a20:	48 89 c7             	mov    %rax,%rdi
  400a23:	e8 22 00 00 00       	callq  400a4a <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a28:	48 83 c4 18          	add    $0x18,%rsp
  400a2c:	c3                   	retq   
  400a2d:	90                   	nop

0000000000400a2e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>:
  400a2e:	48 83 ec 18          	sub    $0x18,%rsp
  400a32:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a37:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a3c:	48 89 c7             	mov    %rax,%rdi
  400a3f:	e8 2e 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a44:	48 83 c4 18          	add    $0x18,%rsp
  400a48:	c3                   	retq   
  400a49:	90                   	nop

0000000000400a4a <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a4a:	48 83 ec 18          	sub    $0x18,%rsp
  400a4e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a53:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a58:	48 89 c7             	mov    %rax,%rdi
  400a5b:	e8 a0 00 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400a60:	48 83 c4 18          	add    $0x18,%rsp
  400a64:	c3                   	retq   
  400a65:	90                   	nop

0000000000400a66 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1ERKS3_>:
  400a66:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a6b:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400a70:	90                   	nop
  400a71:	c3                   	retq   

0000000000400a72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a72:	48 83 ec 18          	sub    $0x18,%rsp
  400a76:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a80:	48 89 c7             	mov    %rax,%rdi
  400a83:	e8 b6 00 00 00       	callq  400b3e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400a88:	48 83 c4 18          	add    $0x18,%rsp
  400a8c:	c3                   	retq   
  400a8d:	90                   	nop

0000000000400a8e <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1ERKS3_>:
  400a8e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a93:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400a98:	90                   	nop
  400a99:	c3                   	retq   

0000000000400a9a <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400a9a:	48 83 ec 18          	sub    $0x18,%rsp
  400a9e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400aa3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 8a fc ff ff       	callq  40073a <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400ab0:	90                   	nop
  400ab1:	48 83 c4 18          	add    $0x18,%rsp
  400ab5:	c3                   	retq   

0000000000400ab6 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400ab6:	48 83 ec 18          	sub    $0x18,%rsp
  400aba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400abf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ac4:	48 89 c7             	mov    %rax,%rdi
  400ac7:	e8 ce ff ff ff       	callq  400a9a <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400acc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ad1:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400ad8:	90                   	nop
  400ad9:	48 83 c4 18          	add    $0x18,%rsp
  400add:	c3                   	retq   

0000000000400ade <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ade:	90                   	nop
  400adf:	c3                   	retq   

0000000000400ae0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ae0:	48 83 ec 18          	sub    $0x18,%rsp
  400ae4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ae9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aee:	48 83 c0 01          	add    $0x1,%rax
  400af2:	48 89 c7             	mov    %rax,%rdi
  400af5:	e8 83 00 00 00       	callq  400b7d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400afa:	48 83 c4 18          	add    $0x18,%rsp
  400afe:	c3                   	retq   
  400aff:	90                   	nop

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400b00:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400b05:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400b0c:	00 00 
  400b0e:	48 c7 44 24 e8 60 20 	movq   $0x602060,-0x18(%rsp)
  400b15:	60 00 
  400b17:	48 c7 44 24 f0 70 20 	movq   $0x602070,-0x10(%rsp)
  400b1e:	60 00 
  400b20:	48 c7 44 24 f8 70 20 	movq   $0x602070,-0x8(%rsp)
  400b27:	60 00 
  400b29:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400b2e:	48 83 c0 04          	add    $0x4,%rax
  400b32:	48 c1 e0 02          	shl    $0x2,%rax
  400b36:	48 05 60 20 60 00    	add    $0x602060,%rax
  400b3c:	c3                   	retq   
  400b3d:	90                   	nop

0000000000400b3e <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400b3e:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400b43:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400b4a:	00 00 
  400b4c:	48 c7 44 24 e8 60 20 	movq   $0x602060,-0x18(%rsp)
  400b53:	60 00 
  400b55:	48 c7 44 24 f0 50 69 	movq   $0x606950,-0x10(%rsp)
  400b5c:	60 00 
  400b5e:	48 c7 44 24 f8 50 69 	movq   $0x606950,-0x8(%rsp)
  400b65:	60 00 
  400b67:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400b6c:	48 05 3c 12 00 00    	add    $0x123c,%rax
  400b72:	48 c1 e0 02          	shl    $0x2,%rax
  400b76:	48 05 60 20 60 00    	add    $0x602060,%rax
  400b7c:	c3                   	retq   

0000000000400b7d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b7d:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b82:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
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
