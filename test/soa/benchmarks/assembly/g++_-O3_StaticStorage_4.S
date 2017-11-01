
bin/g++_-O3_StaticStorage_4:     file format elf64-x86-64


Disassembly of section .init:

00000000004003f0 <_init>:
  4003f0:	48 83 ec 08          	sub    $0x8,%rsp
  4003f4:	48 8b 05 fd 0b 20 00 	mov    0x200bfd(%rip),%rax        # 600ff8 <_DYNAMIC+0x1d0>
  4003fb:	48 85 c0             	test   %rax,%rax
  4003fe:	74 05                	je     400405 <_init+0x15>
  400400:	e8 3b 00 00 00       	callq  400440 <__printf_chk@plt+0x10>
  400405:	48 83 c4 08          	add    $0x8,%rsp
  400409:	c3                   	retq   

Disassembly of section .plt:

0000000000400410 <__libc_start_main@plt-0x10>:
  400410:	ff 35 f2 0b 20 00    	pushq  0x200bf2(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400416:	ff 25 f4 0b 20 00    	jmpq   *0x200bf4(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40041c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400420 <__libc_start_main@plt>:
  400420:	ff 25 f2 0b 20 00    	jmpq   *0x200bf2(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400426:	68 00 00 00 00       	pushq  $0x0
  40042b:	e9 e0 ff ff ff       	jmpq   400410 <_init+0x20>

0000000000400430 <__printf_chk@plt>:
  400430:	ff 25 ea 0b 20 00    	jmpq   *0x200bea(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400436:	68 01 00 00 00       	pushq  $0x1
  40043b:	e9 d0 ff ff ff       	jmpq   400410 <_init+0x20>

Disassembly of section .plt.got:

0000000000400440 <.plt.got>:
  400440:	ff 25 b2 0b 20 00    	jmpq   *0x200bb2(%rip)        # 600ff8 <_DYNAMIC+0x1d0>
  400446:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400450 <main>:
  400450:	55                   	push   %rbp
  400451:	53                   	push   %rbx
  400452:	be 74 07 40 00       	mov    $0x400774,%esi
  400457:	bf 01 00 00 00       	mov    $0x1,%edi
  40045c:	48 83 ec 08          	sub    $0x8,%rsp
  400460:	48 8b 05 99 0d 20 00 	mov    0x200d99(%rip),%rax        # 601200 <storage_buffer>
  400467:	48 c7 05 36 9f 20 00 	movq   $0x0,0x209f36(%rip)        # 60a3a8 <storage_buffer+0x91a8>
  40046e:	00 00 00 00 
  400472:	48 83 c0 01          	add    $0x1,%rax
  400476:	48 8d 1c 85 08 12 60 	lea    0x601208(,%rax,4),%rbx
  40047d:	00 
  40047e:	48 89 05 7b 0d 20 00 	mov    %rax,0x200d7b(%rip)        # 601200 <storage_buffer>
  400485:	c6 04 85 08 12 60 00 	movb   $0x0,0x601208(,%rax,4)
  40048c:	00 
  40048d:	c6 43 01 00          	movb   $0x0,0x1(%rbx)
  400491:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  400495:	c6 43 03 00          	movb   $0x0,0x3(%rbx)
  400499:	48 81 eb 08 12 60 00 	sub    $0x601208,%rbx
  4004a0:	48 c1 eb 02          	shr    $0x2,%rbx
  4004a4:	48 8d 2c 9d 08 12 60 	lea    0x601208(,%rbx,4),%rbp
  4004ab:	00 
  4004ac:	c7 04 9d 08 12 60 00 	movl   $0x7777,0x601208(,%rbx,4)
  4004b3:	77 77 00 00 
  4004b7:	c7 85 d0 48 00 00 88 	movl   $0x8888,0x48d0(%rbp)
  4004be:	88 00 00 
  4004c1:	69 04 9d 08 12 60 00 	imul   $0x5555,0x601208(,%rbx,4),%eax
  4004c8:	55 55 00 00 
  4004cc:	89 04 9d 08 12 60 00 	mov    %eax,0x601208(,%rbx,4)
  4004d3:	31 c0                	xor    %eax,%eax
  4004d5:	69 8d d0 48 00 00 44 	imul   $0x4444,0x48d0(%rbp),%ecx
  4004dc:	44 00 00 
  4004df:	89 8d d0 48 00 00    	mov    %ecx,0x48d0(%rbp)
  4004e5:	8b 14 9d 08 12 60 00 	mov    0x601208(,%rbx,4),%edx
  4004ec:	e8 3f ff ff ff       	callq  400430 <__printf_chk@plt>
  4004f1:	81 3c 9d 08 12 60 00 	cmpl   $0x27d22d83,0x601208(,%rbx,4)
  4004f8:	83 2d d2 27 
  4004fc:	b8 01 00 00 00       	mov    $0x1,%eax
  400501:	75 0d                	jne    400510 <main+0xc0>
  400503:	81 bd d0 48 00 00 20 	cmpl   $0x24686420,0x48d0(%rbp)
  40050a:	64 68 24 
  40050d:	0f 95 c0             	setne  %al
  400510:	48 83 c4 08          	add    $0x8,%rsp
  400514:	0f b6 c0             	movzbl %al,%eax
  400517:	5b                   	pop    %rbx
  400518:	5d                   	pop    %rbp
  400519:	c3                   	retq   
  40051a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400520 <_start>:
  400520:	31 ed                	xor    %ebp,%ebp
  400522:	49 89 d1             	mov    %rdx,%r9
  400525:	5e                   	pop    %rsi
  400526:	48 89 e2             	mov    %rsp,%rdx
  400529:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40052d:	50                   	push   %rax
  40052e:	54                   	push   %rsp
  40052f:	49 c7 c0 60 07 40 00 	mov    $0x400760,%r8
  400536:	48 c7 c1 f0 06 40 00 	mov    $0x4006f0,%rcx
  40053d:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  400544:	e8 d7 fe ff ff       	callq  400420 <__libc_start_main@plt>
  400549:	f4                   	hlt    
  40054a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400550 <deregister_tm_clones>:
  400550:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  400555:	55                   	push   %rbp
  400556:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  40055c:	48 83 f8 0e          	cmp    $0xe,%rax
  400560:	48 89 e5             	mov    %rsp,%rbp
  400563:	76 1b                	jbe    400580 <deregister_tm_clones+0x30>
  400565:	b8 00 00 00 00       	mov    $0x0,%eax
  40056a:	48 85 c0             	test   %rax,%rax
  40056d:	74 11                	je     400580 <deregister_tm_clones+0x30>
  40056f:	5d                   	pop    %rbp
  400570:	bf 38 10 60 00       	mov    $0x601038,%edi
  400575:	ff e0                	jmpq   *%rax
  400577:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40057e:	00 00 
  400580:	5d                   	pop    %rbp
  400581:	c3                   	retq   
  400582:	0f 1f 40 00          	nopl   0x0(%rax)
  400586:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40058d:	00 00 00 

0000000000400590 <register_tm_clones>:
  400590:	be 38 10 60 00       	mov    $0x601038,%esi
  400595:	55                   	push   %rbp
  400596:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40059d:	48 c1 fe 03          	sar    $0x3,%rsi
  4005a1:	48 89 e5             	mov    %rsp,%rbp
  4005a4:	48 89 f0             	mov    %rsi,%rax
  4005a7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4005ab:	48 01 c6             	add    %rax,%rsi
  4005ae:	48 d1 fe             	sar    %rsi
  4005b1:	74 15                	je     4005c8 <register_tm_clones+0x38>
  4005b3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005b8:	48 85 c0             	test   %rax,%rax
  4005bb:	74 0b                	je     4005c8 <register_tm_clones+0x38>
  4005bd:	5d                   	pop    %rbp
  4005be:	bf 38 10 60 00       	mov    $0x601038,%edi
  4005c3:	ff e0                	jmpq   *%rax
  4005c5:	0f 1f 00             	nopl   (%rax)
  4005c8:	5d                   	pop    %rbp
  4005c9:	c3                   	retq   
  4005ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005d0 <__do_global_dtors_aux>:
  4005d0:	80 3d 69 0a 20 00 00 	cmpb   $0x0,0x200a69(%rip)        # 601040 <completed.7585>
  4005d7:	75 11                	jne    4005ea <__do_global_dtors_aux+0x1a>
  4005d9:	55                   	push   %rbp
  4005da:	48 89 e5             	mov    %rsp,%rbp
  4005dd:	e8 6e ff ff ff       	callq  400550 <deregister_tm_clones>
  4005e2:	5d                   	pop    %rbp
  4005e3:	c6 05 56 0a 20 00 01 	movb   $0x1,0x200a56(%rip)        # 601040 <completed.7585>
  4005ea:	f3 c3                	repz retq 
  4005ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005f0 <frame_dummy>:
  4005f0:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  4005f5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005f9:	75 05                	jne    400600 <frame_dummy+0x10>
  4005fb:	eb 93                	jmp    400590 <register_tm_clones>
  4005fd:	0f 1f 00             	nopl   (%rax)
  400600:	b8 00 00 00 00       	mov    $0x0,%eax
  400605:	48 85 c0             	test   %rax,%rax
  400608:	74 f1                	je     4005fb <frame_dummy+0xb>
  40060a:	55                   	push   %rbp
  40060b:	48 89 e5             	mov    %rsp,%rbp
  40060e:	ff d0                	callq  *%rax
  400610:	5d                   	pop    %rbp
  400611:	e9 7a ff ff ff       	jmpq   400590 <register_tm_clones>
  400616:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40061d:	00 00 00 

0000000000400620 <_Z12new_instancev>:
  400620:	48 8b 05 d9 0b 20 00 	mov    0x200bd9(%rip),%rax        # 601200 <storage_buffer>
  400627:	48 8d 50 01          	lea    0x1(%rax),%rdx
  40062b:	48 8d 04 95 08 12 60 	lea    0x601208(,%rdx,4),%rax
  400632:	00 
  400633:	48 89 15 c6 0b 20 00 	mov    %rdx,0x200bc6(%rip)        # 601200 <storage_buffer>
  40063a:	c6 04 95 08 12 60 00 	movb   $0x0,0x601208(,%rdx,4)
  400641:	00 
  400642:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  400646:	c6 40 02 00          	movb   $0x0,0x2(%rax)
  40064a:	c6 40 03 00          	movb   $0x0,0x3(%rax)
  40064e:	c3                   	retq   
  40064f:	90                   	nop

0000000000400650 <_Z12write_field0P9TestClass>:
  400650:	48 81 ef 08 12 60 00 	sub    $0x601208,%rdi
  400657:	48 c1 ef 02          	shr    $0x2,%rdi
  40065b:	c7 04 bd 08 12 60 00 	movl   $0x7777,0x601208(,%rdi,4)
  400662:	77 77 00 00 
  400666:	c3                   	retq   
  400667:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40066e:	00 00 

0000000000400670 <_Z11read_field0P9TestClass>:
  400670:	48 81 ef 08 12 60 00 	sub    $0x601208,%rdi
  400677:	48 c1 ef 02          	shr    $0x2,%rdi
  40067b:	8b 04 bd 08 12 60 00 	mov    0x601208(,%rdi,4),%eax
  400682:	c3                   	retq   
  400683:	0f 1f 00             	nopl   (%rax)
  400686:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40068d:	00 00 00 

0000000000400690 <_Z12write_field1P9TestClass>:
  400690:	48 81 ef 08 12 60 00 	sub    $0x601208,%rdi
  400697:	48 c1 ef 02          	shr    $0x2,%rdi
  40069b:	c7 04 bd d8 5a 60 00 	movl   $0x7777,0x605ad8(,%rdi,4)
  4006a2:	77 77 00 00 
  4006a6:	c3                   	retq   
  4006a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006ae:	00 00 

00000000004006b0 <_Z11read_field1P9TestClass>:
  4006b0:	48 81 ef 08 12 60 00 	sub    $0x601208,%rdi
  4006b7:	48 c1 ef 02          	shr    $0x2,%rdi
  4006bb:	8b 04 bd d8 5a 60 00 	mov    0x605ad8(,%rdi,4),%eax
  4006c2:	c3                   	retq   
  4006c3:	0f 1f 00             	nopl   (%rax)
  4006c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006cd:	00 00 00 

00000000004006d0 <_Z21explicit_write_field0m>:
  4006d0:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  4006d7:	77 77 00 00 
  4006db:	c3                   	retq   
  4006dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006e0 <_Z20explicit_read_field0m>:
  4006e0:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  4006e7:	c3                   	retq   
  4006e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006ef:	00 

00000000004006f0 <__libc_csu_init>:
  4006f0:	41 57                	push   %r15
  4006f2:	41 56                	push   %r14
  4006f4:	41 89 ff             	mov    %edi,%r15d
  4006f7:	41 55                	push   %r13
  4006f9:	41 54                	push   %r12
  4006fb:	4c 8d 25 0e 07 20 00 	lea    0x20070e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  400702:	55                   	push   %rbp
  400703:	48 8d 2d 0e 07 20 00 	lea    0x20070e(%rip),%rbp        # 600e18 <__init_array_end>
  40070a:	53                   	push   %rbx
  40070b:	49 89 f6             	mov    %rsi,%r14
  40070e:	49 89 d5             	mov    %rdx,%r13
  400711:	4c 29 e5             	sub    %r12,%rbp
  400714:	48 83 ec 08          	sub    $0x8,%rsp
  400718:	48 c1 fd 03          	sar    $0x3,%rbp
  40071c:	e8 cf fc ff ff       	callq  4003f0 <_init>
  400721:	48 85 ed             	test   %rbp,%rbp
  400724:	74 20                	je     400746 <__libc_csu_init+0x56>
  400726:	31 db                	xor    %ebx,%ebx
  400728:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40072f:	00 
  400730:	4c 89 ea             	mov    %r13,%rdx
  400733:	4c 89 f6             	mov    %r14,%rsi
  400736:	44 89 ff             	mov    %r15d,%edi
  400739:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40073d:	48 83 c3 01          	add    $0x1,%rbx
  400741:	48 39 eb             	cmp    %rbp,%rbx
  400744:	75 ea                	jne    400730 <__libc_csu_init+0x40>
  400746:	48 83 c4 08          	add    $0x8,%rsp
  40074a:	5b                   	pop    %rbx
  40074b:	5d                   	pop    %rbp
  40074c:	41 5c                	pop    %r12
  40074e:	41 5d                	pop    %r13
  400750:	41 5e                	pop    %r14
  400752:	41 5f                	pop    %r15
  400754:	c3                   	retq   
  400755:	90                   	nop
  400756:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40075d:	00 00 00 

0000000000400760 <__libc_csu_fini>:
  400760:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400764 <_fini>:
  400764:	48 83 ec 08          	sub    $0x8,%rsp
  400768:	48 83 c4 08          	add    $0x8,%rsp
  40076c:	c3                   	retq   
