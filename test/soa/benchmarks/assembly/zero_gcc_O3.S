
bin/zero_gcc_O3:     file format elf64-x86-64


Disassembly of section .init:

0000000000400428 <_init>:
  400428:	48 83 ec 08          	sub    $0x8,%rsp
  40042c:	48 8b 05 c5 0b 20 00 	mov    0x200bc5(%rip),%rax        # 600ff8 <_DYNAMIC+0x1d0>
  400433:	48 85 c0             	test   %rax,%rax
  400436:	74 05                	je     40043d <_init+0x15>
  400438:	e8 53 00 00 00       	callq  400490 <__printf_chk@plt+0x10>
  40043d:	48 83 c4 08          	add    $0x8,%rsp
  400441:	c3                   	retq   

Disassembly of section .plt:

0000000000400450 <__libc_start_main@plt-0x10>:
  400450:	ff 35 b2 0b 20 00    	pushq  0x200bb2(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400456:	ff 25 b4 0b 20 00    	jmpq   *0x200bb4(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40045c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400460 <__libc_start_main@plt>:
  400460:	ff 25 b2 0b 20 00    	jmpq   *0x200bb2(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400466:	68 00 00 00 00       	pushq  $0x0
  40046b:	e9 e0 ff ff ff       	jmpq   400450 <_init+0x28>

0000000000400470 <malloc@plt>:
  400470:	ff 25 aa 0b 20 00    	jmpq   *0x200baa(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400476:	68 01 00 00 00       	pushq  $0x1
  40047b:	e9 d0 ff ff ff       	jmpq   400450 <_init+0x28>

0000000000400480 <__printf_chk@plt>:
  400480:	ff 25 a2 0b 20 00    	jmpq   *0x200ba2(%rip)        # 601028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400486:	68 02 00 00 00       	pushq  $0x2
  40048b:	e9 c0 ff ff ff       	jmpq   400450 <_init+0x28>

Disassembly of section .plt.got:

0000000000400490 <.plt.got>:
  400490:	ff 25 62 0b 20 00    	jmpq   *0x200b62(%rip)        # 600ff8 <_DYNAMIC+0x1d0>
  400496:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004a0 <main>:
  4004a0:	55                   	push   %rbp
  4004a1:	53                   	push   %rbx
  4004a2:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  4004a7:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  4004ac:	be 14 07 40 00       	mov    $0x400714,%esi
  4004b1:	bf 01 00 00 00       	mov    $0x1,%edi
  4004b6:	48 83 ec 08          	sub    $0x8,%rsp
  4004ba:	8b 1d a0 0b 20 00    	mov    0x200ba0(%rip),%ebx        # 601060 <_ZN9TestClass7storageE>
  4004c0:	8d 43 01             	lea    0x1(%rbx),%eax
  4004c3:	48 8d 2c 9d 00 00 00 	lea    0x0(,%rbx,4),%rbp
  4004ca:	00 
  4004cb:	89 05 8f 0b 20 00    	mov    %eax,0x200b8f(%rip)        # 601060 <_ZN9TestClass7storageE>
  4004d1:	48 8b 05 90 0b 20 00 	mov    0x200b90(%rip),%rax        # 601068 <_ZN9TestClass7storageE+0x8>
  4004d8:	89 14 28             	mov    %edx,(%rax,%rbp,1)
  4004db:	89 8c 28 d0 48 00 00 	mov    %ecx,0x48d0(%rax,%rbp,1)
  4004e2:	31 c0                	xor    %eax,%eax
  4004e4:	e8 97 ff ff ff       	callq  400480 <__printf_chk@plt>
  4004e9:	48 8b 15 78 0b 20 00 	mov    0x200b78(%rip),%rdx        # 601068 <_ZN9TestClass7storageE+0x8>
  4004f0:	b8 01 00 00 00       	mov    $0x1,%eax
  4004f5:	81 3c 9a 83 2d d2 27 	cmpl   $0x27d22d83,(%rdx,%rbx,4)
  4004fc:	75 0e                	jne    40050c <main+0x6c>
  4004fe:	81 bc 2a d0 48 00 00 	cmpl   $0x24686420,0x48d0(%rdx,%rbp,1)
  400505:	20 64 68 24 
  400509:	0f 95 c0             	setne  %al
  40050c:	48 83 c4 08          	add    $0x8,%rsp
  400510:	0f b6 c0             	movzbl %al,%eax
  400513:	5b                   	pop    %rbx
  400514:	5d                   	pop    %rbp
  400515:	c3                   	retq   
  400516:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40051d:	00 00 00 

0000000000400520 <_GLOBAL__sub_I__ZN9TestClass7storageE>:
  400520:	48 83 ec 08          	sub    $0x8,%rsp
  400524:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400529:	e8 42 ff ff ff       	callq  400470 <malloc@plt>
  40052e:	48 89 05 33 0b 20 00 	mov    %rax,0x200b33(%rip)        # 601068 <_ZN9TestClass7storageE+0x8>
  400535:	48 83 c4 08          	add    $0x8,%rsp
  400539:	c3                   	retq   
  40053a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400540 <_start>:
  400540:	31 ed                	xor    %ebp,%ebp
  400542:	49 89 d1             	mov    %rdx,%r9
  400545:	5e                   	pop    %rsi
  400546:	48 89 e2             	mov    %rsp,%rdx
  400549:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40054d:	50                   	push   %rax
  40054e:	54                   	push   %rsp
  40054f:	49 c7 c0 00 07 40 00 	mov    $0x400700,%r8
  400556:	48 c7 c1 90 06 40 00 	mov    $0x400690,%rcx
  40055d:	48 c7 c7 a0 04 40 00 	mov    $0x4004a0,%rdi
  400564:	e8 f7 fe ff ff       	callq  400460 <__libc_start_main@plt>
  400569:	f4                   	hlt    
  40056a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400570 <deregister_tm_clones>:
  400570:	b8 47 10 60 00       	mov    $0x601047,%eax
  400575:	55                   	push   %rbp
  400576:	48 2d 40 10 60 00    	sub    $0x601040,%rax
  40057c:	48 83 f8 0e          	cmp    $0xe,%rax
  400580:	48 89 e5             	mov    %rsp,%rbp
  400583:	76 1b                	jbe    4005a0 <deregister_tm_clones+0x30>
  400585:	b8 00 00 00 00       	mov    $0x0,%eax
  40058a:	48 85 c0             	test   %rax,%rax
  40058d:	74 11                	je     4005a0 <deregister_tm_clones+0x30>
  40058f:	5d                   	pop    %rbp
  400590:	bf 40 10 60 00       	mov    $0x601040,%edi
  400595:	ff e0                	jmpq   *%rax
  400597:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40059e:	00 00 
  4005a0:	5d                   	pop    %rbp
  4005a1:	c3                   	retq   
  4005a2:	0f 1f 40 00          	nopl   0x0(%rax)
  4005a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005ad:	00 00 00 

00000000004005b0 <register_tm_clones>:
  4005b0:	be 40 10 60 00       	mov    $0x601040,%esi
  4005b5:	55                   	push   %rbp
  4005b6:	48 81 ee 40 10 60 00 	sub    $0x601040,%rsi
  4005bd:	48 c1 fe 03          	sar    $0x3,%rsi
  4005c1:	48 89 e5             	mov    %rsp,%rbp
  4005c4:	48 89 f0             	mov    %rsi,%rax
  4005c7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4005cb:	48 01 c6             	add    %rax,%rsi
  4005ce:	48 d1 fe             	sar    %rsi
  4005d1:	74 15                	je     4005e8 <register_tm_clones+0x38>
  4005d3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005d8:	48 85 c0             	test   %rax,%rax
  4005db:	74 0b                	je     4005e8 <register_tm_clones+0x38>
  4005dd:	5d                   	pop    %rbp
  4005de:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005e3:	ff e0                	jmpq   *%rax
  4005e5:	0f 1f 00             	nopl   (%rax)
  4005e8:	5d                   	pop    %rbp
  4005e9:	c3                   	retq   
  4005ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005f0 <__do_global_dtors_aux>:
  4005f0:	80 3d 49 0a 20 00 00 	cmpb   $0x0,0x200a49(%rip)        # 601040 <__TMC_END__>
  4005f7:	75 11                	jne    40060a <__do_global_dtors_aux+0x1a>
  4005f9:	55                   	push   %rbp
  4005fa:	48 89 e5             	mov    %rsp,%rbp
  4005fd:	e8 6e ff ff ff       	callq  400570 <deregister_tm_clones>
  400602:	5d                   	pop    %rbp
  400603:	c6 05 36 0a 20 00 01 	movb   $0x1,0x200a36(%rip)        # 601040 <__TMC_END__>
  40060a:	f3 c3                	repz retq 
  40060c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400610 <frame_dummy>:
  400610:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  400615:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400619:	75 05                	jne    400620 <frame_dummy+0x10>
  40061b:	eb 93                	jmp    4005b0 <register_tm_clones>
  40061d:	0f 1f 00             	nopl   (%rax)
  400620:	b8 00 00 00 00       	mov    $0x0,%eax
  400625:	48 85 c0             	test   %rax,%rax
  400628:	74 f1                	je     40061b <frame_dummy+0xb>
  40062a:	55                   	push   %rbp
  40062b:	48 89 e5             	mov    %rsp,%rbp
  40062e:	ff d0                	callq  *%rax
  400630:	5d                   	pop    %rbp
  400631:	e9 7a ff ff ff       	jmpq   4005b0 <register_tm_clones>
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <_Z12new_instancev>:
  400640:	8b 05 1a 0a 20 00    	mov    0x200a1a(%rip),%eax        # 601060 <_ZN9TestClass7storageE>
  400646:	8d 50 01             	lea    0x1(%rax),%edx
  400649:	89 15 11 0a 20 00    	mov    %edx,0x200a11(%rip)        # 601060 <_ZN9TestClass7storageE>
  40064f:	c3                   	retq   

0000000000400650 <_Z12write_field0P9TestClass>:
  400650:	48 8b 05 11 0a 20 00 	mov    0x200a11(%rip),%rax        # 601068 <_ZN9TestClass7storageE+0x8>
  400657:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40065e:	c3                   	retq   
  40065f:	90                   	nop

0000000000400660 <_Z11read_field0P9TestClass>:
  400660:	48 8b 05 01 0a 20 00 	mov    0x200a01(%rip),%rax        # 601068 <_ZN9TestClass7storageE+0x8>
  400667:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40066a:	c3                   	retq   
  40066b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400670 <_Z21explicit_write_field0m>:
  400670:	48 8b 05 d9 09 20 00 	mov    0x2009d9(%rip),%rax        # 601050 <explicit_field0>
  400677:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40067e:	c3                   	retq   
  40067f:	90                   	nop

0000000000400680 <_Z20explicit_read_field0m>:
  400680:	48 8b 05 c9 09 20 00 	mov    0x2009c9(%rip),%rax        # 601050 <explicit_field0>
  400687:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40068a:	c3                   	retq   
  40068b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400690 <__libc_csu_init>:
  400690:	41 57                	push   %r15
  400692:	41 56                	push   %r14
  400694:	41 89 ff             	mov    %edi,%r15d
  400697:	41 55                	push   %r13
  400699:	41 54                	push   %r12
  40069b:	4c 8d 25 66 07 20 00 	lea    0x200766(%rip),%r12        # 600e08 <__frame_dummy_init_array_entry>
  4006a2:	55                   	push   %rbp
  4006a3:	48 8d 2d 6e 07 20 00 	lea    0x20076e(%rip),%rbp        # 600e18 <__init_array_end>
  4006aa:	53                   	push   %rbx
  4006ab:	49 89 f6             	mov    %rsi,%r14
  4006ae:	49 89 d5             	mov    %rdx,%r13
  4006b1:	4c 29 e5             	sub    %r12,%rbp
  4006b4:	48 83 ec 08          	sub    $0x8,%rsp
  4006b8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006bc:	e8 67 fd ff ff       	callq  400428 <_init>
  4006c1:	48 85 ed             	test   %rbp,%rbp
  4006c4:	74 20                	je     4006e6 <__libc_csu_init+0x56>
  4006c6:	31 db                	xor    %ebx,%ebx
  4006c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006cf:	00 
  4006d0:	4c 89 ea             	mov    %r13,%rdx
  4006d3:	4c 89 f6             	mov    %r14,%rsi
  4006d6:	44 89 ff             	mov    %r15d,%edi
  4006d9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4006dd:	48 83 c3 01          	add    $0x1,%rbx
  4006e1:	48 39 eb             	cmp    %rbp,%rbx
  4006e4:	75 ea                	jne    4006d0 <__libc_csu_init+0x40>
  4006e6:	48 83 c4 08          	add    $0x8,%rsp
  4006ea:	5b                   	pop    %rbx
  4006eb:	5d                   	pop    %rbp
  4006ec:	41 5c                	pop    %r12
  4006ee:	41 5d                	pop    %r13
  4006f0:	41 5e                	pop    %r14
  4006f2:	41 5f                	pop    %r15
  4006f4:	c3                   	retq   
  4006f5:	90                   	nop
  4006f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006fd:	00 00 00 

0000000000400700 <__libc_csu_fini>:
  400700:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400704 <_fini>:
  400704:	48 83 ec 08          	sub    $0x8,%rsp
  400708:	48 83 c4 08          	add    $0x8,%rsp
  40070c:	c3                   	retq   
