
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
  4004af:	49 c7 c0 d0 0c 40 00 	mov    $0x400cd0,%r8
  4004b6:	48 c7 c1 60 0c 40 00 	mov    $0x400c60,%rcx
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
  40059b:	e8 9f 04 00 00       	callq  400a3f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  4005a0:	bf 04 00 00 00       	mov    $0x4,%edi
  4005a5:	e8 be 04 00 00       	callq  400a68 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4005aa:	48 89 c3             	mov    %rax,%rbx
  4005ad:	b8 00 00 00 00       	mov    $0x0,%eax
  4005b2:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4005b6:	48 83 f8 03          	cmp    $0x3,%rax
  4005ba:	74 06                	je     4005c2 <main+0x2c>
  4005bc:	48 83 c0 01          	add    $0x1,%rax
  4005c0:	eb f0                	jmp    4005b2 <main+0x1c>
  4005c2:	48 89 df             	mov    %rbx,%rdi
  4005c5:	e8 06 03 00 00       	callq  4008d0 <_ZN9TestClassC1Ev>
  4005ca:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4005cf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005d4:	48 83 c0 04          	add    $0x4,%rax
  4005d8:	be 77 77 00 00       	mov    $0x7777,%esi
  4005dd:	48 89 c7             	mov    %rax,%rdi
  4005e0:	e8 45 03 00 00       	callq  40092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  4005e5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4005ea:	48 83 c0 04          	add    $0x4,%rax
  4005ee:	be 88 88 00 00       	mov    $0x8888,%esi
  4005f3:	48 89 c7             	mov    %rax,%rdi
  4005f6:	e8 79 03 00 00       	callq  400974 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  4005fb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400600:	48 89 c7             	mov    %rax,%rdi
  400603:	e8 ce 01 00 00       	callq  4007d6 <_ZN9TestClass15increase_field0Ev>
  400608:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40060d:	48 89 c7             	mov    %rax,%rdi
  400610:	e8 29 02 00 00       	callq  40083e <_ZN9TestClass15increase_field1Ev>
  400615:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40061a:	48 83 c0 04          	add    $0x4,%rax
  40061e:	48 89 c7             	mov    %rax,%rdi
  400621:	e8 32 03 00 00       	callq  400958 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  400626:	8b 18                	mov    (%rax),%ebx
  400628:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40062d:	48 83 c0 04          	add    $0x4,%rax
  400631:	48 89 c7             	mov    %rax,%rdi
  400634:	e8 d5 02 00 00       	callq  40090e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  400639:	8b 00                	mov    (%rax),%eax
  40063b:	89 da                	mov    %ebx,%edx
  40063d:	89 c6                	mov    %eax,%esi
  40063f:	bf fc 0c 40 00       	mov    $0x400cfc,%edi
  400644:	b8 00 00 00 00       	mov    $0x0,%eax
  400649:	e8 22 fe ff ff       	callq  400470 <printf@plt>
  40064e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400653:	48 83 c0 04          	add    $0x4,%rax
  400657:	48 89 c7             	mov    %rax,%rdi
  40065a:	e8 af 02 00 00       	callq  40090e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  40065f:	8b 00                	mov    (%rax),%eax
  400661:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400666:	75 1a                	jne    400682 <main+0xec>
  400668:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40066d:	48 83 c0 04          	add    $0x4,%rax
  400671:	48 89 c7             	mov    %rax,%rdi
  400674:	e8 df 02 00 00       	callq  400958 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  400679:	8b 00                	mov    (%rax),%eax
  40067b:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400680:	74 07                	je     400689 <main+0xf3>
  400682:	b8 01 00 00 00       	mov    $0x1,%eax
  400687:	eb 05                	jmp    40068e <main+0xf8>
  400689:	b8 00 00 00 00       	mov    $0x0,%eax
  40068e:	0f b6 c0             	movzbl %al,%eax
  400691:	48 83 c4 10          	add    $0x10,%rsp
  400695:	5b                   	pop    %rbx
  400696:	c3                   	retq   

0000000000400697 <_Z12new_instancev>:
  400697:	53                   	push   %rbx
  400698:	bf 04 00 00 00       	mov    $0x4,%edi
  40069d:	e8 c6 03 00 00       	callq  400a68 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006a2:	48 89 c3             	mov    %rax,%rbx
  4006a5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006aa:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4006ae:	48 83 f8 03          	cmp    $0x3,%rax
  4006b2:	74 06                	je     4006ba <_Z12new_instancev+0x23>
  4006b4:	48 83 c0 01          	add    $0x1,%rax
  4006b8:	eb f0                	jmp    4006aa <_Z12new_instancev+0x13>
  4006ba:	48 89 df             	mov    %rbx,%rdi
  4006bd:	e8 0e 02 00 00       	callq  4008d0 <_ZN9TestClassC1Ev>
  4006c2:	48 89 d8             	mov    %rbx,%rax
  4006c5:	5b                   	pop    %rbx
  4006c6:	c3                   	retq   

00000000004006c7 <_Z12write_field0P9TestClass>:
  4006c7:	48 83 ec 18          	sub    $0x18,%rsp
  4006cb:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006d0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006d5:	48 83 c0 04          	add    $0x4,%rax
  4006d9:	be 77 77 00 00       	mov    $0x7777,%esi
  4006de:	48 89 c7             	mov    %rax,%rdi
  4006e1:	e8 44 02 00 00       	callq  40092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  4006e6:	90                   	nop
  4006e7:	48 83 c4 18          	add    $0x18,%rsp
  4006eb:	c3                   	retq   

00000000004006ec <_Z11read_field0P9TestClass>:
  4006ec:	48 83 ec 18          	sub    $0x18,%rsp
  4006f0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4006f5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4006fa:	48 83 c0 04          	add    $0x4,%rax
  4006fe:	48 89 c7             	mov    %rax,%rdi
  400701:	e8 08 02 00 00       	callq  40090e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  400706:	8b 00                	mov    (%rax),%eax
  400708:	48 83 c4 18          	add    $0x18,%rsp
  40070c:	c3                   	retq   

000000000040070d <_Z12write_field1P9TestClass>:
  40070d:	48 83 ec 18          	sub    $0x18,%rsp
  400711:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400716:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40071b:	48 83 c0 04          	add    $0x4,%rax
  40071f:	be 77 77 00 00       	mov    $0x7777,%esi
  400724:	48 89 c7             	mov    %rax,%rdi
  400727:	e8 48 02 00 00       	callq  400974 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  40072c:	90                   	nop
  40072d:	48 83 c4 18          	add    $0x18,%rsp
  400731:	c3                   	retq   

0000000000400732 <_Z11read_field1P9TestClass>:
  400732:	48 83 ec 18          	sub    $0x18,%rsp
  400736:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40073b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400740:	48 83 c0 04          	add    $0x4,%rax
  400744:	48 89 c7             	mov    %rax,%rdi
  400747:	e8 0c 02 00 00       	callq  400958 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  40074c:	8b 00                	mov    (%rax),%eax
  40074e:	48 83 c4 18          	add    $0x18,%rsp
  400752:	c3                   	retq   

0000000000400753 <_Z21explicit_write_field0m>:
  400753:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400758:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40075d:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  400764:	77 77 00 00 
  400768:	90                   	nop
  400769:	c3                   	retq   

000000000040076a <_Z20explicit_read_field0m>:
  40076a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40076f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400774:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  40077b:	c3                   	retq   

000000000040077c <_ZnwmPv>:
  40077c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400781:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400786:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40078b:	c3                   	retq   

000000000040078c <_ZN4ikra3soa19StorageStrategyBaseC1Ev>:
  40078c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400791:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400796:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  40079d:	90                   	nop
  40079e:	c3                   	retq   
  40079f:	90                   	nop

00000000004007a0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  4007a0:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  4007a5:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  4007aa:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007af:	48 8b 00             	mov    (%rax),%rax
  4007b2:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  4007b7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007bc:	48 8b 10             	mov    (%rax),%rdx
  4007bf:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  4007c4:	48 01 c2             	add    %rax,%rdx
  4007c7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  4007cc:	48 89 10             	mov    %rdx,(%rax)
  4007cf:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4007d4:	c3                   	retq   
  4007d5:	90                   	nop

00000000004007d6 <_ZN9TestClass15increase_field0Ev>:
  4007d6:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4007db:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4007df:	41 ff 72 f8          	pushq  -0x8(%r10)
  4007e3:	55                   	push   %rbp
  4007e4:	48 89 e5             	mov    %rsp,%rbp
  4007e7:	41 52                	push   %r10
  4007e9:	48 83 ec 48          	sub    $0x48,%rsp
  4007ed:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4007f1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007f8:	00 00 
  4007fa:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007fe:	31 c0                	xor    %eax,%eax
  400800:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400804:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400808:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40080c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400811:	48 89 ce             	mov    %rcx,%rsi
  400814:	48 89 c7             	mov    %rax,%rdi
  400817:	e8 86 01 00 00       	callq  4009a2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>
  40081c:	90                   	nop
  40081d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400821:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400828:	00 00 
  40082a:	74 05                	je     400831 <_ZN9TestClass15increase_field0Ev+0x5b>
  40082c:	e8 2f fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400831:	48 83 c4 48          	add    $0x48,%rsp
  400835:	41 5a                	pop    %r10
  400837:	5d                   	pop    %rbp
  400838:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  40083c:	c3                   	retq   
  40083d:	90                   	nop

000000000040083e <_ZN9TestClass15increase_field1Ev>:
  40083e:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400843:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400847:	41 ff 72 f8          	pushq  -0x8(%r10)
  40084b:	55                   	push   %rbp
  40084c:	48 89 e5             	mov    %rsp,%rbp
  40084f:	41 52                	push   %r10
  400851:	48 83 ec 48          	sub    $0x48,%rsp
  400855:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400859:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400860:	00 00 
  400862:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400866:	31 c0                	xor    %eax,%eax
  400868:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  40086c:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400870:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400874:	ba 44 44 00 00       	mov    $0x4444,%edx
  400879:	48 89 ce             	mov    %rcx,%rsi
  40087c:	48 89 c7             	mov    %rax,%rdi
  40087f:	e8 66 01 00 00       	callq  4009ea <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>
  400884:	90                   	nop
  400885:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400889:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400890:	00 00 
  400892:	74 05                	je     400899 <_ZN9TestClass15increase_field1Ev+0x5b>
  400894:	e8 c7 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  400899:	48 83 c4 48          	add    $0x48,%rsp
  40089d:	41 5a                	pop    %r10
  40089f:	5d                   	pop    %rbp
  4008a0:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4008a4:	c3                   	retq   
  4008a5:	90                   	nop

00000000004008a6 <_ZN9TestClass14storage_bufferEv>:
  4008a6:	b8 60 20 60 00       	mov    $0x602060,%eax
  4008ab:	c3                   	retq   

00000000004008ac <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  4008ac:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008b1:	90                   	nop
  4008b2:	c3                   	retq   
  4008b3:	90                   	nop

00000000004008b4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  4008b4:	48 83 ec 18          	sub    $0x18,%rsp
  4008b8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008bd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008c2:	48 89 c7             	mov    %rax,%rdi
  4008c5:	e8 e2 ff ff ff       	callq  4008ac <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  4008ca:	90                   	nop
  4008cb:	48 83 c4 18          	add    $0x18,%rsp
  4008cf:	c3                   	retq   

00000000004008d0 <_ZN9TestClassC1Ev>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 ce ff ff ff       	callq  4008b4 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  4008e6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008eb:	48 83 c0 04          	add    $0x4,%rax
  4008ef:	48 89 c7             	mov    %rax,%rdi
  4008f2:	e8 9f 01 00 00       	callq  400a96 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC1Ev>
  4008f7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008fc:	48 83 c0 04          	add    $0x4,%rax
  400900:	48 89 c7             	mov    %rax,%rdi
  400903:	e8 96 01 00 00       	callq  400a9e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC1Ev>
  400908:	90                   	nop
  400909:	48 83 c4 18          	add    $0x18,%rsp
  40090d:	c3                   	retq   

000000000040090e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>:
  40090e:	48 83 ec 18          	sub    $0x18,%rsp
  400912:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400917:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40091c:	48 89 c7             	mov    %rax,%rdi
  40091f:	e8 88 01 00 00       	callq  400aac <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  400924:	48 83 c4 18          	add    $0x18,%rsp
  400928:	c3                   	retq   
  400929:	90                   	nop

000000000040092a <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>:
  40092a:	48 83 ec 18          	sub    $0x18,%rsp
  40092e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400933:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400937:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40093c:	48 89 c7             	mov    %rax,%rdi
  40093f:	e8 68 01 00 00       	callq  400aac <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  400944:	48 89 c2             	mov    %rax,%rdx
  400947:	8b 44 24 04          	mov    0x4(%rsp),%eax
  40094b:	89 02                	mov    %eax,(%rdx)
  40094d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400952:	48 83 c4 18          	add    $0x18,%rsp
  400956:	c3                   	retq   
  400957:	90                   	nop

0000000000400958 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>:
  400958:	48 83 ec 18          	sub    $0x18,%rsp
  40095c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400961:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400966:	48 89 c7             	mov    %rax,%rdi
  400969:	e8 5a 01 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  40096e:	48 83 c4 18          	add    $0x18,%rsp
  400972:	c3                   	retq   
  400973:	90                   	nop

0000000000400974 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>:
  400974:	48 83 ec 18          	sub    $0x18,%rsp
  400978:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40097d:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400981:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 3a 01 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  40098e:	48 89 c2             	mov    %rax,%rdx
  400991:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400995:	89 02                	mov    %eax,(%rdx)
  400997:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40099c:	48 83 c4 18          	add    $0x18,%rsp
  4009a0:	c3                   	retq   
  4009a1:	90                   	nop

00000000004009a2 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>:
  4009a2:	48 83 ec 28          	sub    $0x28,%rsp
  4009a6:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4009ab:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4009b0:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4009b4:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4009b9:	48 89 c7             	mov    %rax,%rdi
  4009bc:	e8 eb 00 00 00       	callq  400aac <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  4009c1:	8b 10                	mov    (%rax),%edx
  4009c3:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  4009c8:	89 10                	mov    %edx,(%rax)
  4009ca:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4009cf:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009d4:	48 89 d6             	mov    %rdx,%rsi
  4009d7:	48 89 c7             	mov    %rax,%rdi
  4009da:	e8 05 01 00 00       	callq  400ae4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC1ERKS3_>
  4009df:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4009e4:	48 83 c4 28          	add    $0x28,%rsp
  4009e8:	c3                   	retq   
  4009e9:	90                   	nop

00000000004009ea <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>:
  4009ea:	48 83 ec 28          	sub    $0x28,%rsp
  4009ee:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4009f3:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  4009f8:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  4009fc:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a01:	48 89 c7             	mov    %rax,%rdi
  400a04:	e8 bf 00 00 00       	callq  400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>
  400a09:	8b 10                	mov    (%rax),%edx
  400a0b:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400a10:	89 10                	mov    %edx,(%rax)
  400a12:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400a17:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a1c:	48 89 d6             	mov    %rdx,%rsi
  400a1f:	48 89 c7             	mov    %rax,%rdi
  400a22:	e8 c9 00 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC1ERKS3_>
  400a27:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a2c:	48 83 c4 28          	add    $0x28,%rsp
  400a30:	c3                   	retq   

0000000000400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>:
  400a31:	48 83 ec 08          	sub    $0x8,%rsp
  400a35:	e8 6c fe ff ff       	callq  4008a6 <_ZN9TestClass14storage_bufferEv>
  400a3a:	48 83 c4 08          	add    $0x8,%rsp
  400a3e:	c3                   	retq   

0000000000400a3f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  400a3f:	48 83 ec 08          	sub    $0x8,%rsp
  400a43:	e8 e9 ff ff ff       	callq  400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400a48:	48 89 c6             	mov    %rax,%rsi
  400a4b:	bf b8 91 00 00       	mov    $0x91b8,%edi
  400a50:	e8 27 fd ff ff       	callq  40077c <_ZnwmPv>
  400a55:	48 85 c0             	test   %rax,%rax
  400a58:	74 08                	je     400a62 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x23>
  400a5a:	48 89 c7             	mov    %rax,%rdi
  400a5d:	e8 b6 00 00 00       	callq  400b18 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400a62:	90                   	nop
  400a63:	48 83 c4 08          	add    $0x8,%rsp
  400a67:	c3                   	retq   

0000000000400a68 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  400a68:	48 83 ec 18          	sub    $0x18,%rsp
  400a6c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a71:	e8 cb 00 00 00       	callq  400b41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a76:	e8 b6 ff ff ff       	callq  400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400a7b:	be 01 00 00 00       	mov    $0x1,%esi
  400a80:	48 89 c7             	mov    %rax,%rdi
  400a83:	e8 18 fd ff ff       	callq  4007a0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400a88:	48 89 c7             	mov    %rax,%rdi
  400a8b:	e8 b3 00 00 00       	callq  400b43 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400a90:	48 83 c4 18          	add    $0x18,%rsp
  400a94:	c3                   	retq   
  400a95:	90                   	nop

0000000000400a96 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC1Ev>:
  400a96:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a9b:	90                   	nop
  400a9c:	c3                   	retq   
  400a9d:	90                   	nop

0000000000400a9e <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC1Ev>:
  400a9e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400aa3:	90                   	nop
  400aa4:	c3                   	retq   

0000000000400aa5 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400aa5:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400aaa:	90                   	nop
  400aab:	c3                   	retq   

0000000000400aac <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>:
  400aac:	48 83 ec 18          	sub    $0x18,%rsp
  400ab0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ab5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aba:	48 89 c7             	mov    %rax,%rdi
  400abd:	e8 a0 00 00 00       	callq  400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac2:	48 83 c4 18          	add    $0x18,%rsp
  400ac6:	c3                   	retq   
  400ac7:	90                   	nop

0000000000400ac8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrIIEEEPiDpT_>:
  400ac8:	48 83 ec 18          	sub    $0x18,%rsp
  400acc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ad1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ad6:	48 89 c7             	mov    %rax,%rdi
  400ad9:	e8 da 00 00 00       	callq  400bb8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ade:	48 83 c4 18          	add    $0x18,%rsp
  400ae2:	c3                   	retq   
  400ae3:	90                   	nop

0000000000400ae4 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC1ERKS3_>:
  400ae4:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ae9:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400aee:	90                   	nop
  400aef:	c3                   	retq   

0000000000400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC1ERKS3_>:
  400af0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400af5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400afa:	90                   	nop
  400afb:	c3                   	retq   

0000000000400afc <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400afc:	48 83 ec 18          	sub    $0x18,%rsp
  400b00:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b05:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b0a:	48 89 c7             	mov    %rax,%rdi
  400b0d:	e8 7a fc ff ff       	callq  40078c <_ZN4ikra3soa19StorageStrategyBaseC1Ev>
  400b12:	90                   	nop
  400b13:	48 83 c4 18          	add    $0x18,%rsp
  400b17:	c3                   	retq   

0000000000400b18 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400b18:	48 83 ec 18          	sub    $0x18,%rsp
  400b1c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b21:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b26:	48 89 c7             	mov    %rax,%rdi
  400b29:	e8 ce ff ff ff       	callq  400afc <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400b2e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b33:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400b3a:	00 
  400b3b:	90                   	nop
  400b3c:	48 83 c4 18          	add    $0x18,%rsp
  400b40:	c3                   	retq   

0000000000400b41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400b41:	90                   	nop
  400b42:	c3                   	retq   

0000000000400b43 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400b43:	48 83 ec 18          	sub    $0x18,%rsp
  400b47:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b4c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b51:	48 83 c0 01          	add    $0x1,%rax
  400b55:	48 89 c7             	mov    %rax,%rdi
  400b58:	e8 b7 00 00 00       	callq  400c14 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b5d:	48 83 c4 18          	add    $0x18,%rsp
  400b61:	c3                   	retq   

0000000000400b62 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b62:	48 83 ec 38          	sub    $0x38,%rsp
  400b66:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b6b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b70:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b75:	e8 b7 fe ff ff       	callq  400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400b7a:	48 89 c7             	mov    %rax,%rdi
  400b7d:	e8 c6 00 00 00       	callq  400c48 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400b82:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400b87:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b8c:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400b91:	48 83 e8 04          	sub    $0x4,%rax
  400b95:	48 c1 e8 02          	shr    $0x2,%rax
  400b99:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400ba0:	00 
  400ba1:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400ba6:	48 01 d0             	add    %rdx,%rax
  400ba9:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400bae:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400bb3:	48 83 c4 38          	add    $0x38,%rsp
  400bb7:	c3                   	retq   

0000000000400bb8 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb8:	48 83 ec 38          	sub    $0x38,%rsp
  400bbc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bc1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bc6:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bcb:	e8 61 fe ff ff       	callq  400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400bd0:	48 89 c7             	mov    %rax,%rdi
  400bd3:	e8 70 00 00 00       	callq  400c48 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400bd8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400bdd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400be2:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400be7:	48 83 e8 04          	sub    $0x4,%rax
  400beb:	48 c1 e8 02          	shr    $0x2,%rax
  400bef:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400bf6:	00 
  400bf7:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400bfc:	48 01 d0             	add    %rdx,%rax
  400bff:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400c05:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400c0a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400c0f:	48 83 c4 38          	add    $0x38,%rsp
  400c13:	c3                   	retq   

0000000000400c14 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400c14:	48 83 ec 28          	sub    $0x28,%rsp
  400c18:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c1d:	e8 0f fe ff ff       	callq  400a31 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400c22:	48 89 c7             	mov    %rax,%rdi
  400c25:	e8 1e 00 00 00       	callq  400c48 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400c2a:	48 89 c2             	mov    %rax,%rdx
  400c2d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c32:	48 c1 e0 02          	shl    $0x2,%rax
  400c36:	48 01 d0             	add    %rdx,%rax
  400c39:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c3e:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c43:	48 83 c4 28          	add    $0x28,%rsp
  400c47:	c3                   	retq   

0000000000400c48 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c48:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c4d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400c52:	48 83 c0 10          	add    $0x10,%rax
  400c56:	c3                   	retq   
  400c57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400c5e:	00 00 

0000000000400c60 <__libc_csu_init>:
  400c60:	41 57                	push   %r15
  400c62:	41 56                	push   %r14
  400c64:	41 89 ff             	mov    %edi,%r15d
  400c67:	41 55                	push   %r13
  400c69:	41 54                	push   %r12
  400c6b:	4c 8d 25 9e 11 20 00 	lea    0x20119e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  400c72:	55                   	push   %rbp
  400c73:	48 8d 2d 9e 11 20 00 	lea    0x20119e(%rip),%rbp        # 601e18 <__init_array_end>
  400c7a:	53                   	push   %rbx
  400c7b:	49 89 f6             	mov    %rsi,%r14
  400c7e:	49 89 d5             	mov    %rdx,%r13
  400c81:	4c 29 e5             	sub    %r12,%rbp
  400c84:	48 83 ec 08          	sub    $0x8,%rsp
  400c88:	48 c1 fd 03          	sar    $0x3,%rbp
  400c8c:	e8 97 f7 ff ff       	callq  400428 <_init>
  400c91:	48 85 ed             	test   %rbp,%rbp
  400c94:	74 20                	je     400cb6 <__libc_csu_init+0x56>
  400c96:	31 db                	xor    %ebx,%ebx
  400c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c9f:	00 
  400ca0:	4c 89 ea             	mov    %r13,%rdx
  400ca3:	4c 89 f6             	mov    %r14,%rsi
  400ca6:	44 89 ff             	mov    %r15d,%edi
  400ca9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400cad:	48 83 c3 01          	add    $0x1,%rbx
  400cb1:	48 39 eb             	cmp    %rbp,%rbx
  400cb4:	75 ea                	jne    400ca0 <__libc_csu_init+0x40>
  400cb6:	48 83 c4 08          	add    $0x8,%rsp
  400cba:	5b                   	pop    %rbx
  400cbb:	5d                   	pop    %rbp
  400cbc:	41 5c                	pop    %r12
  400cbe:	41 5d                	pop    %r13
  400cc0:	41 5e                	pop    %r14
  400cc2:	41 5f                	pop    %r15
  400cc4:	c3                   	retq   
  400cc5:	90                   	nop
  400cc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ccd:	00 00 00 

0000000000400cd0 <__libc_csu_fini>:
  400cd0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400cd4 <_fini>:
  400cd4:	48 83 ec 08          	sub    $0x8,%rsp
  400cd8:	48 83 c4 08          	add    $0x8,%rsp
  400cdc:	c3                   	retq   
