
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
  4005b9:	e8 8a 03 00 00       	callq  400948 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageEv>
  4005be:	bf 00 00 00 00       	mov    $0x0,%edi
  4005c3:	e8 a6 03 00 00       	callq  40096e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4005c8:	48 89 c3             	mov    %rax,%rbx
  4005cb:	48 89 df             	mov    %rbx,%rdi
  4005ce:	e8 c1 02 00 00       	callq  400894 <_ZN9TestClassC1Ev>
  4005d3:	48 89 5d e0          	mov    %rbx,-0x20(%rbp)
  4005d7:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005db:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005df:	ba 77 77 00 00       	mov    $0x7777,%edx
  4005e4:	48 89 ce             	mov    %rcx,%rsi
  4005e7:	48 89 c7             	mov    %rax,%rdi
  4005ea:	e8 cf 03 00 00       	callq  4009be <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4005ef:	48 8b 4d e0          	mov    -0x20(%rbp),%rcx
  4005f3:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4005f7:	ba 88 88 00 00       	mov    $0x8888,%edx
  4005fc:	48 89 ce             	mov    %rcx,%rsi
  4005ff:	48 89 c7             	mov    %rax,%rdi
  400602:	e8 f7 03 00 00       	callq  4009fe <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400607:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40060b:	48 89 c7             	mov    %rax,%rdi
  40060e:	e8 c5 01 00 00       	callq  4007d8 <_ZN9TestClass15increase_field0Ev>
  400613:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400617:	48 89 c7             	mov    %rax,%rdi
  40061a:	e8 03 02 00 00       	callq  400822 <_ZN9TestClass15increase_field1Ev>
  40061f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400623:	48 89 c7             	mov    %rax,%rdi
  400626:	e8 2d 04 00 00       	callq  400a58 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40062b:	8b 18                	mov    (%rax),%ebx
  40062d:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400631:	48 89 c7             	mov    %rax,%rdi
  400634:	e8 05 04 00 00       	callq  400a3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400639:	8b 00                	mov    (%rax),%eax
  40063b:	89 da                	mov    %ebx,%edx
  40063d:	89 c6                	mov    %eax,%esi
  40063f:	bf f4 0b 40 00       	mov    $0x400bf4,%edi
  400644:	b8 00 00 00 00       	mov    $0x0,%eax
  400649:	e8 22 fe ff ff       	callq  400470 <printf@plt>
  40064e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400652:	48 89 c7             	mov    %rax,%rdi
  400655:	e8 e4 03 00 00       	callq  400a3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40065a:	8b 00                	mov    (%rax),%eax
  40065c:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400661:	75 15                	jne    400678 <main+0xd7>
  400663:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400667:	48 89 c7             	mov    %rax,%rdi
  40066a:	e8 e9 03 00 00       	callq  400a58 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40066f:	8b 00                	mov    (%rax),%eax
  400671:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400676:	74 07                	je     40067f <main+0xde>
  400678:	b8 01 00 00 00       	mov    $0x1,%eax
  40067d:	eb 05                	jmp    400684 <main+0xe3>
  40067f:	b8 00 00 00 00       	mov    $0x0,%eax
  400684:	0f b6 c0             	movzbl %al,%eax
  400687:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40068b:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  400692:	00 00 
  400694:	74 05                	je     40069b <main+0xfa>
  400696:	e8 c5 fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  40069b:	48 83 c4 28          	add    $0x28,%rsp
  40069f:	5b                   	pop    %rbx
  4006a0:	5d                   	pop    %rbp
  4006a1:	c3                   	retq   

00000000004006a2 <_Z12new_instancev>:
  4006a2:	55                   	push   %rbp
  4006a3:	48 89 e5             	mov    %rsp,%rbp
  4006a6:	53                   	push   %rbx
  4006a7:	48 83 ec 08          	sub    $0x8,%rsp
  4006ab:	bf 00 00 00 00       	mov    $0x0,%edi
  4006b0:	e8 b9 02 00 00       	callq  40096e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4006b5:	48 89 c3             	mov    %rax,%rbx
  4006b8:	48 89 df             	mov    %rbx,%rdi
  4006bb:	e8 d4 01 00 00       	callq  400894 <_ZN9TestClassC1Ev>
  4006c0:	48 89 d8             	mov    %rbx,%rax
  4006c3:	48 83 c4 08          	add    $0x8,%rsp
  4006c7:	5b                   	pop    %rbx
  4006c8:	5d                   	pop    %rbp
  4006c9:	c3                   	retq   

00000000004006ca <_Z12write_field0P9TestClass>:
  4006ca:	55                   	push   %rbp
  4006cb:	48 89 e5             	mov    %rsp,%rbp
  4006ce:	48 83 ec 30          	sub    $0x30,%rsp
  4006d2:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4006d6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4006dd:	00 00 
  4006df:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4006e3:	31 c0                	xor    %eax,%eax
  4006e5:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4006e9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4006ed:	ba 77 77 00 00       	mov    $0x7777,%edx
  4006f2:	48 89 ce             	mov    %rcx,%rsi
  4006f5:	48 89 c7             	mov    %rax,%rdi
  4006f8:	e8 c1 02 00 00       	callq  4009be <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4006fd:	90                   	nop
  4006fe:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400702:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400709:	00 00 
  40070b:	74 05                	je     400712 <_Z12write_field0P9TestClass+0x48>
  40070d:	e8 4e fd ff ff       	callq  400460 <__stack_chk_fail@plt>
  400712:	c9                   	leaveq 
  400713:	c3                   	retq   

0000000000400714 <_Z11read_field0P9TestClass>:
  400714:	55                   	push   %rbp
  400715:	48 89 e5             	mov    %rsp,%rbp
  400718:	48 83 ec 10          	sub    $0x10,%rsp
  40071c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400720:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400724:	48 89 c7             	mov    %rax,%rdi
  400727:	e8 12 03 00 00       	callq  400a3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40072c:	8b 00                	mov    (%rax),%eax
  40072e:	c9                   	leaveq 
  40072f:	c3                   	retq   

0000000000400730 <_Z12write_field1P9TestClass>:
  400730:	55                   	push   %rbp
  400731:	48 89 e5             	mov    %rsp,%rbp
  400734:	48 83 ec 30          	sub    $0x30,%rsp
  400738:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40073c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400743:	00 00 
  400745:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400749:	31 c0                	xor    %eax,%eax
  40074b:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  40074f:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400753:	ba 77 77 00 00       	mov    $0x7777,%edx
  400758:	48 89 ce             	mov    %rcx,%rsi
  40075b:	48 89 c7             	mov    %rax,%rdi
  40075e:	e8 9b 02 00 00       	callq  4009fe <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400763:	90                   	nop
  400764:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400768:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40076f:	00 00 
  400771:	74 05                	je     400778 <_Z12write_field1P9TestClass+0x48>
  400773:	e8 e8 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400778:	c9                   	leaveq 
  400779:	c3                   	retq   

000000000040077a <_Z11read_field1P9TestClass>:
  40077a:	55                   	push   %rbp
  40077b:	48 89 e5             	mov    %rsp,%rbp
  40077e:	48 83 ec 10          	sub    $0x10,%rsp
  400782:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400786:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40078a:	48 89 c7             	mov    %rax,%rdi
  40078d:	e8 c6 02 00 00       	callq  400a58 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400792:	8b 00                	mov    (%rax),%eax
  400794:	c9                   	leaveq 
  400795:	c3                   	retq   

0000000000400796 <_Z21explicit_write_field0m>:
  400796:	55                   	push   %rbp
  400797:	48 89 e5             	mov    %rsp,%rbp
  40079a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40079e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007a2:	c7 04 85 20 b2 60 00 	movl   $0x7777,0x60b220(,%rax,4)
  4007a9:	77 77 00 00 
  4007ad:	90                   	nop
  4007ae:	5d                   	pop    %rbp
  4007af:	c3                   	retq   

00000000004007b0 <_Z20explicit_read_field0m>:
  4007b0:	55                   	push   %rbp
  4007b1:	48 89 e5             	mov    %rsp,%rbp
  4007b4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007b8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4007bc:	8b 04 85 20 b2 60 00 	mov    0x60b220(,%rax,4),%eax
  4007c3:	5d                   	pop    %rbp
  4007c4:	c3                   	retq   

00000000004007c5 <_ZnwmPv>:
  4007c5:	55                   	push   %rbp
  4007c6:	48 89 e5             	mov    %rsp,%rbp
  4007c9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4007cd:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4007d1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007d5:	5d                   	pop    %rbp
  4007d6:	c3                   	retq   
  4007d7:	90                   	nop

00000000004007d8 <_ZN9TestClass15increase_field0Ev>:
  4007d8:	55                   	push   %rbp
  4007d9:	48 89 e5             	mov    %rsp,%rbp
  4007dc:	48 83 ec 30          	sub    $0x30,%rsp
  4007e0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4007e4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4007eb:	00 00 
  4007ed:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4007f1:	31 c0                	xor    %eax,%eax
  4007f3:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  4007f7:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4007fb:	ba 55 55 00 00       	mov    $0x5555,%edx
  400800:	48 89 ce             	mov    %rcx,%rsi
  400803:	48 89 c7             	mov    %rax,%rdi
  400806:	e8 bd 00 00 00       	callq  4008c8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  40080b:	90                   	nop
  40080c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400810:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400817:	00 00 
  400819:	74 05                	je     400820 <_ZN9TestClass15increase_field0Ev+0x48>
  40081b:	e8 40 fc ff ff       	callq  400460 <__stack_chk_fail@plt>
  400820:	c9                   	leaveq 
  400821:	c3                   	retq   

0000000000400822 <_ZN9TestClass15increase_field1Ev>:
  400822:	55                   	push   %rbp
  400823:	48 89 e5             	mov    %rsp,%rbp
  400826:	48 83 ec 30          	sub    $0x30,%rsp
  40082a:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40082e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400835:	00 00 
  400837:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40083b:	31 c0                	xor    %eax,%eax
  40083d:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
  400841:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  400845:	ba 44 44 00 00       	mov    $0x4444,%edx
  40084a:	48 89 ce             	mov    %rcx,%rsi
  40084d:	48 89 c7             	mov    %rax,%rdi
  400850:	e8 b3 00 00 00       	callq  400908 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400855:	90                   	nop
  400856:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40085a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400861:	00 00 
  400863:	74 05                	je     40086a <_ZN9TestClass15increase_field1Ev+0x48>
  400865:	e8 f6 fb ff ff       	callq  400460 <__stack_chk_fail@plt>
  40086a:	c9                   	leaveq 
  40086b:	c3                   	retq   

000000000040086c <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  40086c:	55                   	push   %rbp
  40086d:	48 89 e5             	mov    %rsp,%rbp
  400870:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400874:	90                   	nop
  400875:	5d                   	pop    %rbp
  400876:	c3                   	retq   
  400877:	90                   	nop

0000000000400878 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  400878:	55                   	push   %rbp
  400879:	48 89 e5             	mov    %rsp,%rbp
  40087c:	48 83 ec 10          	sub    $0x10,%rsp
  400880:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400884:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400888:	48 89 c7             	mov    %rax,%rdi
  40088b:	e8 dc ff ff ff       	callq  40086c <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400890:	90                   	nop
  400891:	c9                   	leaveq 
  400892:	c3                   	retq   
  400893:	90                   	nop

0000000000400894 <_ZN9TestClassC1Ev>:
  400894:	55                   	push   %rbp
  400895:	48 89 e5             	mov    %rsp,%rbp
  400898:	48 83 ec 10          	sub    $0x10,%rsp
  40089c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008a4:	48 89 c7             	mov    %rax,%rdi
  4008a7:	e8 cc ff ff ff       	callq  400878 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  4008ac:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008b0:	48 89 c7             	mov    %rax,%rdi
  4008b3:	e8 e4 00 00 00       	callq  40099c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>
  4008b8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008bc:	48 89 c7             	mov    %rax,%rdi
  4008bf:	e8 e4 00 00 00       	callq  4009a8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>
  4008c4:	90                   	nop
  4008c5:	c9                   	leaveq 
  4008c6:	c3                   	retq   
  4008c7:	90                   	nop

00000000004008c8 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4008c8:	55                   	push   %rbp
  4008c9:	48 89 e5             	mov    %rsp,%rbp
  4008cc:	48 83 ec 20          	sub    $0x20,%rsp
  4008d0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d4:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4008d8:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4008db:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4008df:	48 89 c7             	mov    %rax,%rdi
  4008e2:	e8 8b 01 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008e7:	8b 10                	mov    (%rax),%edx
  4008e9:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  4008ed:	89 10                	mov    %edx,(%rax)
  4008ef:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4008f3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4008f7:	48 89 d6             	mov    %rdx,%rsi
  4008fa:	48 89 c7             	mov    %rax,%rdi
  4008fd:	e8 8a 01 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  400902:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400906:	c9                   	leaveq 
  400907:	c3                   	retq   

0000000000400908 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400908:	55                   	push   %rbp
  400909:	48 89 e5             	mov    %rsp,%rbp
  40090c:	48 83 ec 20          	sub    $0x20,%rsp
  400910:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400914:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400918:	89 55 ec             	mov    %edx,-0x14(%rbp)
  40091b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40091f:	48 89 c7             	mov    %rax,%rdi
  400922:	e8 75 01 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400927:	8b 10                	mov    (%rax),%edx
  400929:	0f af 55 ec          	imul   -0x14(%rbp),%edx
  40092d:	89 10                	mov    %edx,(%rax)
  40092f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400933:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400937:	48 89 d6             	mov    %rdx,%rsi
  40093a:	48 89 c7             	mov    %rax,%rdi
  40093d:	e8 74 01 00 00       	callq  400ab6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400942:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400946:	c9                   	leaveq 
  400947:	c3                   	retq   

0000000000400948 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageEv>:
  400948:	55                   	push   %rbp
  400949:	48 89 e5             	mov    %rsp,%rbp
  40094c:	e8 45 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400951:	48 89 c6             	mov    %rax,%rsi
  400954:	bf b0 91 00 00       	mov    $0x91b0,%edi
  400959:	e8 67 fe ff ff       	callq  4007c5 <_ZnwmPv>
  40095e:	48 85 c0             	test   %rax,%rax
  400961:	74 08                	je     40096b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageEv+0x23>
  400963:	48 89 c7             	mov    %rax,%rdi
  400966:	e8 5b 01 00 00       	callq  400ac6 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  40096b:	90                   	nop
  40096c:	5d                   	pop    %rbp
  40096d:	c3                   	retq   

000000000040096e <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  40096e:	55                   	push   %rbp
  40096f:	48 89 e5             	mov    %rsp,%rbp
  400972:	48 83 ec 10          	sub    $0x10,%rsp
  400976:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40097a:	e8 61 01 00 00       	callq  400ae0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40097f:	e8 12 fc ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400984:	48 89 c2             	mov    %rax,%rdx
  400987:	48 8b 02             	mov    (%rdx),%rax
  40098a:	48 8d 48 01          	lea    0x1(%rax),%rcx
  40098e:	48 89 0a             	mov    %rcx,(%rdx)
  400991:	48 89 c7             	mov    %rax,%rdi
  400994:	e8 4e 01 00 00       	callq  400ae7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400999:	c9                   	leaveq 
  40099a:	c3                   	retq   
  40099b:	90                   	nop

000000000040099c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1Ev>:
  40099c:	55                   	push   %rbp
  40099d:	48 89 e5             	mov    %rsp,%rbp
  4009a0:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a4:	90                   	nop
  4009a5:	5d                   	pop    %rbp
  4009a6:	c3                   	retq   
  4009a7:	90                   	nop

00000000004009a8 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1Ev>:
  4009a8:	55                   	push   %rbp
  4009a9:	48 89 e5             	mov    %rsp,%rbp
  4009ac:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b0:	90                   	nop
  4009b1:	5d                   	pop    %rbp
  4009b2:	c3                   	retq   

00000000004009b3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009b3:	55                   	push   %rbp
  4009b4:	48 89 e5             	mov    %rsp,%rbp
  4009b7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bb:	90                   	nop
  4009bc:	5d                   	pop    %rbp
  4009bd:	c3                   	retq   

00000000004009be <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4009be:	55                   	push   %rbp
  4009bf:	48 89 e5             	mov    %rsp,%rbp
  4009c2:	48 83 ec 20          	sub    $0x20,%rsp
  4009c6:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ca:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4009ce:	89 55 ec             	mov    %edx,-0x14(%rbp)
  4009d1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4009d5:	48 89 c7             	mov    %rax,%rdi
  4009d8:	e8 95 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009dd:	48 89 c2             	mov    %rax,%rdx
  4009e0:	8b 45 ec             	mov    -0x14(%rbp),%eax
  4009e3:	89 02                	mov    %eax,(%rdx)
  4009e5:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  4009e9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009ed:	48 89 d6             	mov    %rdx,%rsi
  4009f0:	48 89 c7             	mov    %rax,%rdi
  4009f3:	e8 94 00 00 00       	callq  400a8c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>
  4009f8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4009fc:	c9                   	leaveq 
  4009fd:	c3                   	retq   

00000000004009fe <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  4009fe:	55                   	push   %rbp
  4009ff:	48 89 e5             	mov    %rsp,%rbp
  400a02:	48 83 ec 20          	sub    $0x20,%rsp
  400a06:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0a:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a0e:	89 55 ec             	mov    %edx,-0x14(%rbp)
  400a11:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a15:	48 89 c7             	mov    %rax,%rdi
  400a18:	e8 7f 00 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a1d:	48 89 c2             	mov    %rax,%rdx
  400a20:	8b 45 ec             	mov    -0x14(%rbp),%eax
  400a23:	89 02                	mov    %eax,(%rdx)
  400a25:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  400a29:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a2d:	48 89 d6             	mov    %rdx,%rsi
  400a30:	48 89 c7             	mov    %rax,%rdi
  400a33:	e8 7e 00 00 00       	callq  400ab6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>
  400a38:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a3c:	c9                   	leaveq 
  400a3d:	c3                   	retq   

0000000000400a3e <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400a3e:	55                   	push   %rbp
  400a3f:	48 89 e5             	mov    %rsp,%rbp
  400a42:	48 83 ec 10          	sub    $0x10,%rsp
  400a46:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a4a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a4e:	48 89 c7             	mov    %rax,%rdi
  400a51:	e8 1c 00 00 00       	callq  400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a56:	c9                   	leaveq 
  400a57:	c3                   	retq   

0000000000400a58 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400a58:	55                   	push   %rbp
  400a59:	48 89 e5             	mov    %rsp,%rbp
  400a5c:	48 83 ec 10          	sub    $0x10,%rsp
  400a60:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a64:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a68:	48 89 c7             	mov    %rax,%rdi
  400a6b:	e8 2c 00 00 00       	callq  400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a70:	c9                   	leaveq 
  400a71:	c3                   	retq   

0000000000400a72 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a72:	55                   	push   %rbp
  400a73:	48 89 e5             	mov    %rsp,%rbp
  400a76:	48 83 ec 10          	sub    $0x10,%rsp
  400a7a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a7e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400a82:	48 89 c7             	mov    %rax,%rdi
  400a85:	e8 7c 00 00 00       	callq  400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a8a:	c9                   	leaveq 
  400a8b:	c3                   	retq   

0000000000400a8c <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC1ERKS3_>:
  400a8c:	55                   	push   %rbp
  400a8d:	48 89 e5             	mov    %rsp,%rbp
  400a90:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a94:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a98:	90                   	nop
  400a99:	5d                   	pop    %rbp
  400a9a:	c3                   	retq   
  400a9b:	90                   	nop

0000000000400a9c <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a9c:	55                   	push   %rbp
  400a9d:	48 89 e5             	mov    %rsp,%rbp
  400aa0:	48 83 ec 10          	sub    $0x10,%rsp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400aac:	48 89 c7             	mov    %rax,%rdi
  400aaf:	e8 74 00 00 00       	callq  400b28 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab4:	c9                   	leaveq 
  400ab5:	c3                   	retq   

0000000000400ab6 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC1ERKS3_>:
  400ab6:	55                   	push   %rbp
  400ab7:	48 89 e5             	mov    %rsp,%rbp
  400aba:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400abe:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400ac2:	90                   	nop
  400ac3:	5d                   	pop    %rbp
  400ac4:	c3                   	retq   
  400ac5:	90                   	nop

0000000000400ac6 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400ac6:	55                   	push   %rbp
  400ac7:	48 89 e5             	mov    %rsp,%rbp
  400aca:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ace:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ad2:	48 c7 80 a8 91 00 00 	movq   $0x0,0x91a8(%rax)
  400ad9:	00 00 00 00 
  400add:	90                   	nop
  400ade:	5d                   	pop    %rbp
  400adf:	c3                   	retq   

0000000000400ae0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	90                   	nop
  400ae5:	5d                   	pop    %rbp
  400ae6:	c3                   	retq   

0000000000400ae7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400ae7:	55                   	push   %rbp
  400ae8:	48 89 e5             	mov    %rsp,%rbp
  400aeb:	48 83 ec 10          	sub    $0x10,%rsp
  400aef:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400af7:	48 83 c0 01          	add    $0x1,%rax
  400afb:	48 89 c7             	mov    %rax,%rdi
  400afe:	e8 4d 00 00 00       	callq  400b50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400b03:	c9                   	leaveq 
  400b04:	c3                   	retq   
  400b05:	90                   	nop

0000000000400b06 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b06:	55                   	push   %rbp
  400b07:	48 89 e5             	mov    %rsp,%rbp
  400b0a:	48 83 ec 08          	sub    $0x8,%rsp
  400b0e:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b12:	e8 7f fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b17:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400b1b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b1f:	48 c1 e0 02          	shl    $0x2,%rax
  400b23:	48 01 d0             	add    %rdx,%rax
  400b26:	c9                   	leaveq 
  400b27:	c3                   	retq   

0000000000400b28 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b28:	55                   	push   %rbp
  400b29:	48 89 e5             	mov    %rsp,%rbp
  400b2c:	48 83 ec 08          	sub    $0x8,%rsp
  400b30:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b34:	e8 5d fa ff ff       	callq  400596 <_ZN9TestClass7storageEv>
  400b39:	48 8d 50 08          	lea    0x8(%rax),%rdx
  400b3d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b41:	48 c1 e0 02          	shl    $0x2,%rax
  400b45:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400b4b:	48 01 d0             	add    %rdx,%rax
  400b4e:	c9                   	leaveq 
  400b4f:	c3                   	retq   

0000000000400b50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b58:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400b5c:	5d                   	pop    %rbp
  400b5d:	c3                   	retq   
  400b5e:	66 90                	xchg   %ax,%ax

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
