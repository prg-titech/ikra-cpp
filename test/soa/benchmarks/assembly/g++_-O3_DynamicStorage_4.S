
bin/g++_-O3_DynamicStorage_4:     file format elf64-x86-64


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
  4004a0:	53                   	push   %rbx
  4004a1:	bf a0 91 00 00       	mov    $0x91a0,%edi
  4004a6:	e8 c5 ff ff ff       	callq  400470 <malloc@plt>
  4004ab:	48 8b 15 ce 0b 20 00 	mov    0x200bce(%rip),%rdx        # 601080 <storage_buffer>
  4004b2:	48 89 05 cf 0b 20 00 	mov    %rax,0x200bcf(%rip)        # 601088 <storage_buffer+0x8>
  4004b9:	be b4 07 40 00       	mov    $0x4007b4,%esi
  4004be:	48 c7 05 cf 0b 20 00 	movq   $0x0,0x200bcf(%rip)        # 601098 <storage_buffer+0x18>
  4004c5:	00 00 00 00 
  4004c9:	48 c7 05 bc 0b 20 00 	movq   $0x0,0x200bbc(%rip)        # 601090 <storage_buffer+0x10>
  4004d0:	00 00 00 00 
  4004d4:	bf 01 00 00 00       	mov    $0x1,%edi
  4004d9:	48 8d 1c 90          	lea    (%rax,%rdx,4),%rbx
  4004dd:	48 8d 4a 01          	lea    0x1(%rdx),%rcx
  4004e1:	48 89 da             	mov    %rbx,%rdx
  4004e4:	48 89 0d 95 0b 20 00 	mov    %rcx,0x200b95(%rip)        # 601080 <storage_buffer>
  4004eb:	c6 03 00             	movb   $0x0,(%rbx)
  4004ee:	48 29 c2             	sub    %rax,%rdx
  4004f1:	c6 43 01 00          	movb   $0x0,0x1(%rbx)
  4004f5:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  4004f9:	48 83 e2 fc          	and    $0xfffffffffffffffc,%rdx
  4004fd:	c6 43 03 00          	movb   $0x0,0x3(%rbx)
  400501:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  400506:	48 01 d0             	add    %rdx,%rax
  400509:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  40050e:	c7 00 83 2d d2 27    	movl   $0x27d22d83,(%rax)
  400514:	c7 80 d0 48 00 00 20 	movl   $0x24686420,0x48d0(%rax)
  40051b:	64 68 24 
  40051e:	31 c0                	xor    %eax,%eax
  400520:	e8 5b ff ff ff       	callq  400480 <__printf_chk@plt>
  400525:	48 8b 05 5c 0b 20 00 	mov    0x200b5c(%rip),%rax        # 601088 <storage_buffer+0x8>
  40052c:	48 29 c3             	sub    %rax,%rbx
  40052f:	48 83 e3 fc          	and    $0xfffffffffffffffc,%rbx
  400533:	48 01 c3             	add    %rax,%rbx
  400536:	b8 01 00 00 00       	mov    $0x1,%eax
  40053b:	81 3b 83 2d d2 27    	cmpl   $0x27d22d83,(%rbx)
  400541:	75 0d                	jne    400550 <main+0xb0>
  400543:	81 bb d0 48 00 00 20 	cmpl   $0x24686420,0x48d0(%rbx)
  40054a:	64 68 24 
  40054d:	0f 95 c0             	setne  %al
  400550:	0f b6 c0             	movzbl %al,%eax
  400553:	5b                   	pop    %rbx
  400554:	c3                   	retq   
  400555:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40055c:	00 00 00 
  40055f:	90                   	nop

0000000000400560 <_start>:
  400560:	31 ed                	xor    %ebp,%ebp
  400562:	49 89 d1             	mov    %rdx,%r9
  400565:	5e                   	pop    %rsi
  400566:	48 89 e2             	mov    %rsp,%rdx
  400569:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40056d:	50                   	push   %rax
  40056e:	54                   	push   %rsp
  40056f:	49 c7 c0 a0 07 40 00 	mov    $0x4007a0,%r8
  400576:	48 c7 c1 30 07 40 00 	mov    $0x400730,%rcx
  40057d:	48 c7 c7 a0 04 40 00 	mov    $0x4004a0,%rdi
  400584:	e8 d7 fe ff ff       	callq  400460 <__libc_start_main@plt>
  400589:	f4                   	hlt    
  40058a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400590 <deregister_tm_clones>:
  400590:	b8 47 10 60 00       	mov    $0x601047,%eax
  400595:	55                   	push   %rbp
  400596:	48 2d 40 10 60 00    	sub    $0x601040,%rax
  40059c:	48 83 f8 0e          	cmp    $0xe,%rax
  4005a0:	48 89 e5             	mov    %rsp,%rbp
  4005a3:	76 1b                	jbe    4005c0 <deregister_tm_clones+0x30>
  4005a5:	b8 00 00 00 00       	mov    $0x0,%eax
  4005aa:	48 85 c0             	test   %rax,%rax
  4005ad:	74 11                	je     4005c0 <deregister_tm_clones+0x30>
  4005af:	5d                   	pop    %rbp
  4005b0:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005b5:	ff e0                	jmpq   *%rax
  4005b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4005be:	00 00 
  4005c0:	5d                   	pop    %rbp
  4005c1:	c3                   	retq   
  4005c2:	0f 1f 40 00          	nopl   0x0(%rax)
  4005c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005cd:	00 00 00 

00000000004005d0 <register_tm_clones>:
  4005d0:	be 40 10 60 00       	mov    $0x601040,%esi
  4005d5:	55                   	push   %rbp
  4005d6:	48 81 ee 40 10 60 00 	sub    $0x601040,%rsi
  4005dd:	48 c1 fe 03          	sar    $0x3,%rsi
  4005e1:	48 89 e5             	mov    %rsp,%rbp
  4005e4:	48 89 f0             	mov    %rsi,%rax
  4005e7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4005eb:	48 01 c6             	add    %rax,%rsi
  4005ee:	48 d1 fe             	sar    %rsi
  4005f1:	74 15                	je     400608 <register_tm_clones+0x38>
  4005f3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005f8:	48 85 c0             	test   %rax,%rax
  4005fb:	74 0b                	je     400608 <register_tm_clones+0x38>
  4005fd:	5d                   	pop    %rbp
  4005fe:	bf 40 10 60 00       	mov    $0x601040,%edi
  400603:	ff e0                	jmpq   *%rax
  400605:	0f 1f 00             	nopl   (%rax)
  400608:	5d                   	pop    %rbp
  400609:	c3                   	retq   
  40060a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400610 <__do_global_dtors_aux>:
  400610:	80 3d 29 0a 20 00 00 	cmpb   $0x0,0x200a29(%rip)        # 601040 <__TMC_END__>
  400617:	75 11                	jne    40062a <__do_global_dtors_aux+0x1a>
  400619:	55                   	push   %rbp
  40061a:	48 89 e5             	mov    %rsp,%rbp
  40061d:	e8 6e ff ff ff       	callq  400590 <deregister_tm_clones>
  400622:	5d                   	pop    %rbp
  400623:	c6 05 16 0a 20 00 01 	movb   $0x1,0x200a16(%rip)        # 601040 <__TMC_END__>
  40062a:	f3 c3                	repz retq 
  40062c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400630 <frame_dummy>:
  400630:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  400635:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400639:	75 05                	jne    400640 <frame_dummy+0x10>
  40063b:	eb 93                	jmp    4005d0 <register_tm_clones>
  40063d:	0f 1f 00             	nopl   (%rax)
  400640:	b8 00 00 00 00       	mov    $0x0,%eax
  400645:	48 85 c0             	test   %rax,%rax
  400648:	74 f1                	je     40063b <frame_dummy+0xb>
  40064a:	55                   	push   %rbp
  40064b:	48 89 e5             	mov    %rsp,%rbp
  40064e:	ff d0                	callq  *%rax
  400650:	5d                   	pop    %rbp
  400651:	e9 7a ff ff ff       	jmpq   4005d0 <register_tm_clones>
  400656:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40065d:	00 00 00 

0000000000400660 <_Z12new_instancev>:
  400660:	48 8b 05 19 0a 20 00 	mov    0x200a19(%rip),%rax        # 601080 <storage_buffer>
  400667:	48 8d 50 01          	lea    0x1(%rax),%rdx
  40066b:	48 89 15 0e 0a 20 00 	mov    %rdx,0x200a0e(%rip)        # 601080 <storage_buffer>
  400672:	48 8b 15 0f 0a 20 00 	mov    0x200a0f(%rip),%rdx        # 601088 <storage_buffer+0x8>
  400679:	48 8d 04 82          	lea    (%rdx,%rax,4),%rax
  40067d:	c6 00 00             	movb   $0x0,(%rax)
  400680:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  400684:	c6 40 02 00          	movb   $0x0,0x2(%rax)
  400688:	c6 40 03 00          	movb   $0x0,0x3(%rax)
  40068c:	c3                   	retq   
  40068d:	0f 1f 00             	nopl   (%rax)

0000000000400690 <_Z12write_field0P9TestClass>:
  400690:	48 8b 05 f1 09 20 00 	mov    0x2009f1(%rip),%rax        # 601088 <storage_buffer+0x8>
  400697:	48 29 c7             	sub    %rax,%rdi
  40069a:	48 c1 ef 02          	shr    $0x2,%rdi
  40069e:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006a5:	c3                   	retq   
  4006a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ad:	00 00 00 

00000000004006b0 <_Z11read_field0P9TestClass>:
  4006b0:	48 8b 05 d1 09 20 00 	mov    0x2009d1(%rip),%rax        # 601088 <storage_buffer+0x8>
  4006b7:	48 29 c7             	sub    %rax,%rdi
  4006ba:	48 c1 ef 02          	shr    $0x2,%rdi
  4006be:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006c1:	c3                   	retq   
  4006c2:	0f 1f 40 00          	nopl   0x0(%rax)
  4006c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006cd:	00 00 00 

00000000004006d0 <_Z12write_field1P9TestClass>:
  4006d0:	48 8b 05 b1 09 20 00 	mov    0x2009b1(%rip),%rax        # 601088 <storage_buffer+0x8>
  4006d7:	48 29 c7             	sub    %rax,%rdi
  4006da:	48 c1 ef 02          	shr    $0x2,%rdi
  4006de:	c7 84 b8 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,4)
  4006e5:	77 77 00 00 
  4006e9:	c3                   	retq   
  4006ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006f0 <_Z11read_field1P9TestClass>:
  4006f0:	48 8b 05 91 09 20 00 	mov    0x200991(%rip),%rax        # 601088 <storage_buffer+0x8>
  4006f7:	48 29 c7             	sub    %rax,%rdi
  4006fa:	48 c1 ef 02          	shr    $0x2,%rdi
  4006fe:	8b 84 b8 d0 48 00 00 	mov    0x48d0(%rax,%rdi,4),%eax
  400705:	c3                   	retq   
  400706:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40070d:	00 00 00 

0000000000400710 <_Z21explicit_write_field0m>:
  400710:	48 8b 05 49 09 20 00 	mov    0x200949(%rip),%rax        # 601060 <explicit_field0>
  400717:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40071e:	c3                   	retq   
  40071f:	90                   	nop

0000000000400720 <_Z20explicit_read_field0m>:
  400720:	48 8b 05 39 09 20 00 	mov    0x200939(%rip),%rax        # 601060 <explicit_field0>
  400727:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40072a:	c3                   	retq   
  40072b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400730 <__libc_csu_init>:
  400730:	41 57                	push   %r15
  400732:	41 56                	push   %r14
  400734:	41 89 ff             	mov    %edi,%r15d
  400737:	41 55                	push   %r13
  400739:	41 54                	push   %r12
  40073b:	4c 8d 25 ce 06 20 00 	lea    0x2006ce(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  400742:	55                   	push   %rbp
  400743:	48 8d 2d ce 06 20 00 	lea    0x2006ce(%rip),%rbp        # 600e18 <__init_array_end>
  40074a:	53                   	push   %rbx
  40074b:	49 89 f6             	mov    %rsi,%r14
  40074e:	49 89 d5             	mov    %rdx,%r13
  400751:	4c 29 e5             	sub    %r12,%rbp
  400754:	48 83 ec 08          	sub    $0x8,%rsp
  400758:	48 c1 fd 03          	sar    $0x3,%rbp
  40075c:	e8 c7 fc ff ff       	callq  400428 <_init>
  400761:	48 85 ed             	test   %rbp,%rbp
  400764:	74 20                	je     400786 <__libc_csu_init+0x56>
  400766:	31 db                	xor    %ebx,%ebx
  400768:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40076f:	00 
  400770:	4c 89 ea             	mov    %r13,%rdx
  400773:	4c 89 f6             	mov    %r14,%rsi
  400776:	44 89 ff             	mov    %r15d,%edi
  400779:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40077d:	48 83 c3 01          	add    $0x1,%rbx
  400781:	48 39 eb             	cmp    %rbp,%rbx
  400784:	75 ea                	jne    400770 <__libc_csu_init+0x40>
  400786:	48 83 c4 08          	add    $0x8,%rsp
  40078a:	5b                   	pop    %rbx
  40078b:	5d                   	pop    %rbp
  40078c:	41 5c                	pop    %r12
  40078e:	41 5d                	pop    %r13
  400790:	41 5e                	pop    %r14
  400792:	41 5f                	pop    %r15
  400794:	c3                   	retq   
  400795:	90                   	nop
  400796:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40079d:	00 00 00 

00000000004007a0 <__libc_csu_fini>:
  4007a0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004007a4 <_fini>:
  4007a4:	48 83 ec 08          	sub    $0x8,%rsp
  4007a8:	48 83 c4 08          	add    $0x8,%rsp
  4007ac:	c3                   	retq   
