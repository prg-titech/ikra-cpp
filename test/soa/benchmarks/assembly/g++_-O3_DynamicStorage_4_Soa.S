
bin/g++_-O3_DynamicStorage_4_Soa:     file format elf64-x86-64


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
  4004a1:	bf c0 91 00 00       	mov    $0x91c0,%edi
  4004a6:	e8 c5 ff ff ff       	callq  400470 <malloc@plt>
  4004ab:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  4004b0:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  4004b5:	48 89 05 fc 9e 20 00 	mov    %rax,0x209efc(%rip)        # 60a3b8 <__TestClassdata_buffer+0x18>
  4004bc:	c7 05 ea 9e 20 00 00 	movl   $0x0,0x209eea(%rip)        # 60a3b0 <__TestClassdata_buffer+0x10>
  4004c3:	00 00 00 
  4004c6:	48 c7 05 d7 9e 20 00 	movq   $0x0,0x209ed7(%rip)        # 60a3a8 <__TestClassdata_buffer+0x8>
  4004cd:	00 00 00 00 
  4004d1:	48 8d 58 04          	lea    0x4(%rax),%rbx
  4004d5:	48 c7 05 c0 9e 20 00 	movq   $0x1,0x209ec0(%rip)        # 60a3a0 <__TestClassdata_buffer>
  4004dc:	01 00 00 00 
  4004e0:	be c4 07 40 00       	mov    $0x4007c4,%esi
  4004e5:	89 50 04             	mov    %edx,0x4(%rax)
  4004e8:	89 88 e4 48 00 00    	mov    %ecx,0x48e4(%rax)
  4004ee:	bf 01 00 00 00       	mov    $0x1,%edi
  4004f3:	31 c0                	xor    %eax,%eax
  4004f5:	e8 86 ff ff ff       	callq  400480 <__printf_chk@plt>
  4004fa:	48 8b 05 b7 9e 20 00 	mov    0x209eb7(%rip),%rax        # 60a3b8 <__TestClassdata_buffer+0x18>
  400501:	48 29 c3             	sub    %rax,%rbx
  400504:	48 83 e3 fc          	and    $0xfffffffffffffffc,%rbx
  400508:	48 01 c3             	add    %rax,%rbx
  40050b:	b8 01 00 00 00       	mov    $0x1,%eax
  400510:	81 3b 83 2d d2 27    	cmpl   $0x27d22d83,(%rbx)
  400516:	75 0d                	jne    400525 <main+0x85>
  400518:	81 bb e0 48 00 00 20 	cmpl   $0x24686420,0x48e0(%rbx)
  40051f:	64 68 24 
  400522:	0f 95 c0             	setne  %al
  400525:	0f b6 c0             	movzbl %al,%eax
  400528:	5b                   	pop    %rbx
  400529:	c3                   	retq   
  40052a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400530 <_start>:
  400530:	31 ed                	xor    %ebp,%ebp
  400532:	49 89 d1             	mov    %rdx,%r9
  400535:	5e                   	pop    %rsi
  400536:	48 89 e2             	mov    %rsp,%rdx
  400539:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40053d:	50                   	push   %rax
  40053e:	54                   	push   %rsp
  40053f:	49 c7 c0 b0 07 40 00 	mov    $0x4007b0,%r8
  400546:	48 c7 c1 40 07 40 00 	mov    $0x400740,%rcx
  40054d:	48 c7 c7 a0 04 40 00 	mov    $0x4004a0,%rdi
  400554:	e8 07 ff ff ff       	callq  400460 <__libc_start_main@plt>
  400559:	f4                   	hlt    
  40055a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400560 <deregister_tm_clones>:
  400560:	b8 47 10 60 00       	mov    $0x601047,%eax
  400565:	55                   	push   %rbp
  400566:	48 2d 40 10 60 00    	sub    $0x601040,%rax
  40056c:	48 83 f8 0e          	cmp    $0xe,%rax
  400570:	48 89 e5             	mov    %rsp,%rbp
  400573:	76 1b                	jbe    400590 <deregister_tm_clones+0x30>
  400575:	b8 00 00 00 00       	mov    $0x0,%eax
  40057a:	48 85 c0             	test   %rax,%rax
  40057d:	74 11                	je     400590 <deregister_tm_clones+0x30>
  40057f:	5d                   	pop    %rbp
  400580:	bf 40 10 60 00       	mov    $0x601040,%edi
  400585:	ff e0                	jmpq   *%rax
  400587:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40058e:	00 00 
  400590:	5d                   	pop    %rbp
  400591:	c3                   	retq   
  400592:	0f 1f 40 00          	nopl   0x0(%rax)
  400596:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40059d:	00 00 00 

00000000004005a0 <register_tm_clones>:
  4005a0:	be 40 10 60 00       	mov    $0x601040,%esi
  4005a5:	55                   	push   %rbp
  4005a6:	48 81 ee 40 10 60 00 	sub    $0x601040,%rsi
  4005ad:	48 c1 fe 03          	sar    $0x3,%rsi
  4005b1:	48 89 e5             	mov    %rsp,%rbp
  4005b4:	48 89 f0             	mov    %rsi,%rax
  4005b7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4005bb:	48 01 c6             	add    %rax,%rsi
  4005be:	48 d1 fe             	sar    %rsi
  4005c1:	74 15                	je     4005d8 <register_tm_clones+0x38>
  4005c3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005c8:	48 85 c0             	test   %rax,%rax
  4005cb:	74 0b                	je     4005d8 <register_tm_clones+0x38>
  4005cd:	5d                   	pop    %rbp
  4005ce:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005d3:	ff e0                	jmpq   *%rax
  4005d5:	0f 1f 00             	nopl   (%rax)
  4005d8:	5d                   	pop    %rbp
  4005d9:	c3                   	retq   
  4005da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005e0 <__do_global_dtors_aux>:
  4005e0:	80 3d 59 0a 20 00 00 	cmpb   $0x0,0x200a59(%rip)        # 601040 <__TMC_END__>
  4005e7:	75 11                	jne    4005fa <__do_global_dtors_aux+0x1a>
  4005e9:	55                   	push   %rbp
  4005ea:	48 89 e5             	mov    %rsp,%rbp
  4005ed:	e8 6e ff ff ff       	callq  400560 <deregister_tm_clones>
  4005f2:	5d                   	pop    %rbp
  4005f3:	c6 05 46 0a 20 00 01 	movb   $0x1,0x200a46(%rip)        # 601040 <__TMC_END__>
  4005fa:	f3 c3                	repz retq 
  4005fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400600 <frame_dummy>:
  400600:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  400605:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400609:	75 05                	jne    400610 <frame_dummy+0x10>
  40060b:	eb 93                	jmp    4005a0 <register_tm_clones>
  40060d:	0f 1f 00             	nopl   (%rax)
  400610:	b8 00 00 00 00       	mov    $0x0,%eax
  400615:	48 85 c0             	test   %rax,%rax
  400618:	74 f1                	je     40060b <frame_dummy+0xb>
  40061a:	55                   	push   %rbp
  40061b:	48 89 e5             	mov    %rsp,%rbp
  40061e:	ff d0                	callq  *%rax
  400620:	5d                   	pop    %rbp
  400621:	e9 7a ff ff ff       	jmpq   4005a0 <register_tm_clones>
  400626:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40062d:	00 00 00 

0000000000400630 <_Z12new_instancev>:
  400630:	48 8b 05 69 9d 20 00 	mov    0x209d69(%rip),%rax        # 60a3a0 <__TestClassdata_buffer>
  400637:	48 8b 15 7a 9d 20 00 	mov    0x209d7a(%rip),%rdx        # 60a3b8 <__TestClassdata_buffer+0x18>
  40063e:	48 83 c0 01          	add    $0x1,%rax
  400642:	48 89 05 57 9d 20 00 	mov    %rax,0x209d57(%rip)        # 60a3a0 <__TestClassdata_buffer>
  400649:	48 8d 04 82          	lea    (%rdx,%rax,4),%rax
  40064d:	c6 00 00             	movb   $0x0,(%rax)
  400650:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  400654:	c6 40 02 00          	movb   $0x0,0x2(%rax)
  400658:	c6 40 03 00          	movb   $0x0,0x3(%rax)
  40065c:	c3                   	retq   
  40065d:	0f 1f 00             	nopl   (%rax)

0000000000400660 <_Z12write_field0P9TestClass>:
  400660:	48 8b 05 51 9d 20 00 	mov    0x209d51(%rip),%rax        # 60a3b8 <__TestClassdata_buffer+0x18>
  400667:	48 29 c7             	sub    %rax,%rdi
  40066a:	48 c1 ef 02          	shr    $0x2,%rdi
  40066e:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  400675:	c3                   	retq   
  400676:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40067d:	00 00 00 

0000000000400680 <_Z11read_field0P9TestClass>:
  400680:	48 8b 05 31 9d 20 00 	mov    0x209d31(%rip),%rax        # 60a3b8 <__TestClassdata_buffer+0x18>
  400687:	48 29 c7             	sub    %rax,%rdi
  40068a:	48 c1 ef 02          	shr    $0x2,%rdi
  40068e:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  400691:	c3                   	retq   
  400692:	0f 1f 40 00          	nopl   0x0(%rax)
  400696:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40069d:	00 00 00 

00000000004006a0 <_Z12write_field1P9TestClass>:
  4006a0:	48 8b 05 11 9d 20 00 	mov    0x209d11(%rip),%rax        # 60a3b8 <__TestClassdata_buffer+0x18>
  4006a7:	48 29 c7             	sub    %rax,%rdi
  4006aa:	48 c1 ef 02          	shr    $0x2,%rdi
  4006ae:	c7 84 b8 e0 48 00 00 	movl   $0x7777,0x48e0(%rax,%rdi,4)
  4006b5:	77 77 00 00 
  4006b9:	c3                   	retq   
  4006ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006c0 <_Z11read_field1P9TestClass>:
  4006c0:	48 8b 05 f1 9c 20 00 	mov    0x209cf1(%rip),%rax        # 60a3b8 <__TestClassdata_buffer+0x18>
  4006c7:	48 29 c7             	sub    %rax,%rdi
  4006ca:	48 c1 ef 02          	shr    $0x2,%rdi
  4006ce:	8b 84 b8 e0 48 00 00 	mov    0x48e0(%rax,%rdi,4),%eax
  4006d5:	c3                   	retq   
  4006d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006dd:	00 00 00 

00000000004006e0 <_Z21explicit_write_field0m>:
  4006e0:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  4006e7:	77 77 00 00 
  4006eb:	c3                   	retq   
  4006ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006f0 <_Z20explicit_read_field0m>:
  4006f0:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  4006f7:	c3                   	retq   
  4006f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006ff:	00 

0000000000400700 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400700:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400706:	c3                   	retq   
  400707:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40070e:	00 00 

0000000000400710 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  400710:	8b 07                	mov    (%rdi),%eax
  400712:	c3                   	retq   
  400713:	0f 1f 00             	nopl   (%rax)
  400716:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40071d:	00 00 00 

0000000000400720 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400720:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400727:	c3                   	retq   
  400728:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40072f:	00 

0000000000400730 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  400730:	8b 47 04             	mov    0x4(%rdi),%eax
  400733:	c3                   	retq   
  400734:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073b:	00 00 00 
  40073e:	66 90                	xchg   %ax,%ax

0000000000400740 <__libc_csu_init>:
  400740:	41 57                	push   %r15
  400742:	41 56                	push   %r14
  400744:	41 89 ff             	mov    %edi,%r15d
  400747:	41 55                	push   %r13
  400749:	41 54                	push   %r12
  40074b:	4c 8d 25 be 06 20 00 	lea    0x2006be(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  400752:	55                   	push   %rbp
  400753:	48 8d 2d be 06 20 00 	lea    0x2006be(%rip),%rbp        # 600e18 <__init_array_end>
  40075a:	53                   	push   %rbx
  40075b:	49 89 f6             	mov    %rsi,%r14
  40075e:	49 89 d5             	mov    %rdx,%r13
  400761:	4c 29 e5             	sub    %r12,%rbp
  400764:	48 83 ec 08          	sub    $0x8,%rsp
  400768:	48 c1 fd 03          	sar    $0x3,%rbp
  40076c:	e8 b7 fc ff ff       	callq  400428 <_init>
  400771:	48 85 ed             	test   %rbp,%rbp
  400774:	74 20                	je     400796 <__libc_csu_init+0x56>
  400776:	31 db                	xor    %ebx,%ebx
  400778:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40077f:	00 
  400780:	4c 89 ea             	mov    %r13,%rdx
  400783:	4c 89 f6             	mov    %r14,%rsi
  400786:	44 89 ff             	mov    %r15d,%edi
  400789:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40078d:	48 83 c3 01          	add    $0x1,%rbx
  400791:	48 39 eb             	cmp    %rbp,%rbx
  400794:	75 ea                	jne    400780 <__libc_csu_init+0x40>
  400796:	48 83 c4 08          	add    $0x8,%rsp
  40079a:	5b                   	pop    %rbx
  40079b:	5d                   	pop    %rbp
  40079c:	41 5c                	pop    %r12
  40079e:	41 5d                	pop    %r13
  4007a0:	41 5e                	pop    %r14
  4007a2:	41 5f                	pop    %r15
  4007a4:	c3                   	retq   
  4007a5:	90                   	nop
  4007a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007ad:	00 00 00 

00000000004007b0 <__libc_csu_fini>:
  4007b0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004007b4 <_fini>:
  4007b4:	48 83 ec 08          	sub    $0x8,%rsp
  4007b8:	48 83 c4 08          	add    $0x8,%rsp
  4007bc:	c3                   	retq   
