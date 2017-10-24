
bin/g++_-O3_StaticStorage_0:     file format elf64-x86-64


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
  400452:	be e4 06 40 00       	mov    $0x4006e4,%esi
  400457:	bf 01 00 00 00       	mov    $0x1,%edi
  40045c:	48 83 ec 08          	sub    $0x8,%rsp
  400460:	8b 1d 1a 0c 20 00    	mov    0x200c1a(%rip),%ebx        # 601080 <_ZN9TestClass7storageE>
  400466:	48 8d 2c 9d d0 48 00 	lea    0x48d0(,%rbx,4),%rbp
  40046d:	00 
  40046e:	8d 43 01             	lea    0x1(%rbx),%eax
  400471:	89 05 09 0c 20 00    	mov    %eax,0x200c09(%rip)        # 601080 <_ZN9TestClass7storageE>
  400477:	c7 04 9d 84 10 60 00 	movl   $0x7777,0x601084(,%rbx,4)
  40047e:	77 77 00 00 
  400482:	c7 85 84 10 60 00 88 	movl   $0x8888,0x601084(%rbp)
  400489:	88 00 00 
  40048c:	69 04 9d 84 10 60 00 	imul   $0x5555,0x601084(,%rbx,4),%eax
  400493:	55 55 00 00 
  400497:	89 04 9d 84 10 60 00 	mov    %eax,0x601084(,%rbx,4)
  40049e:	31 c0                	xor    %eax,%eax
  4004a0:	69 8d 84 10 60 00 44 	imul   $0x4444,0x601084(%rbp),%ecx
  4004a7:	44 00 00 
  4004aa:	89 8d 84 10 60 00    	mov    %ecx,0x601084(%rbp)
  4004b0:	8b 14 9d 84 10 60 00 	mov    0x601084(,%rbx,4),%edx
  4004b7:	e8 74 ff ff ff       	callq  400430 <__printf_chk@plt>
  4004bc:	81 3c 9d 84 10 60 00 	cmpl   $0x27d22d83,0x601084(,%rbx,4)
  4004c3:	83 2d d2 27 
  4004c7:	b8 01 00 00 00       	mov    $0x1,%eax
  4004cc:	75 0d                	jne    4004db <main+0x8b>
  4004ce:	81 bd 84 10 60 00 20 	cmpl   $0x24686420,0x601084(%rbp)
  4004d5:	64 68 24 
  4004d8:	0f 95 c0             	setne  %al
  4004db:	48 83 c4 08          	add    $0x8,%rsp
  4004df:	0f b6 c0             	movzbl %al,%eax
  4004e2:	5b                   	pop    %rbx
  4004e3:	5d                   	pop    %rbp
  4004e4:	c3                   	retq   
  4004e5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4004ec:	00 00 00 
  4004ef:	90                   	nop

00000000004004f0 <_start>:
  4004f0:	31 ed                	xor    %ebp,%ebp
  4004f2:	49 89 d1             	mov    %rdx,%r9
  4004f5:	5e                   	pop    %rsi
  4004f6:	48 89 e2             	mov    %rsp,%rdx
  4004f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004fd:	50                   	push   %rax
  4004fe:	54                   	push   %rsp
  4004ff:	49 c7 c0 d0 06 40 00 	mov    $0x4006d0,%r8
  400506:	48 c7 c1 60 06 40 00 	mov    $0x400660,%rcx
  40050d:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  400514:	e8 07 ff ff ff       	callq  400420 <__libc_start_main@plt>
  400519:	f4                   	hlt    
  40051a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400520 <deregister_tm_clones>:
  400520:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  400525:	55                   	push   %rbp
  400526:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  40052c:	48 83 f8 0e          	cmp    $0xe,%rax
  400530:	48 89 e5             	mov    %rsp,%rbp
  400533:	76 1b                	jbe    400550 <deregister_tm_clones+0x30>
  400535:	b8 00 00 00 00       	mov    $0x0,%eax
  40053a:	48 85 c0             	test   %rax,%rax
  40053d:	74 11                	je     400550 <deregister_tm_clones+0x30>
  40053f:	5d                   	pop    %rbp
  400540:	bf 38 10 60 00       	mov    $0x601038,%edi
  400545:	ff e0                	jmpq   *%rax
  400547:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40054e:	00 00 
  400550:	5d                   	pop    %rbp
  400551:	c3                   	retq   
  400552:	0f 1f 40 00          	nopl   0x0(%rax)
  400556:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40055d:	00 00 00 

0000000000400560 <register_tm_clones>:
  400560:	be 38 10 60 00       	mov    $0x601038,%esi
  400565:	55                   	push   %rbp
  400566:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40056d:	48 c1 fe 03          	sar    $0x3,%rsi
  400571:	48 89 e5             	mov    %rsp,%rbp
  400574:	48 89 f0             	mov    %rsi,%rax
  400577:	48 c1 e8 3f          	shr    $0x3f,%rax
  40057b:	48 01 c6             	add    %rax,%rsi
  40057e:	48 d1 fe             	sar    %rsi
  400581:	74 15                	je     400598 <register_tm_clones+0x38>
  400583:	b8 00 00 00 00       	mov    $0x0,%eax
  400588:	48 85 c0             	test   %rax,%rax
  40058b:	74 0b                	je     400598 <register_tm_clones+0x38>
  40058d:	5d                   	pop    %rbp
  40058e:	bf 38 10 60 00       	mov    $0x601038,%edi
  400593:	ff e0                	jmpq   *%rax
  400595:	0f 1f 00             	nopl   (%rax)
  400598:	5d                   	pop    %rbp
  400599:	c3                   	retq   
  40059a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005a0 <__do_global_dtors_aux>:
  4005a0:	80 3d 99 0a 20 00 00 	cmpb   $0x0,0x200a99(%rip)        # 601040 <completed.7585>
  4005a7:	75 11                	jne    4005ba <__do_global_dtors_aux+0x1a>
  4005a9:	55                   	push   %rbp
  4005aa:	48 89 e5             	mov    %rsp,%rbp
  4005ad:	e8 6e ff ff ff       	callq  400520 <deregister_tm_clones>
  4005b2:	5d                   	pop    %rbp
  4005b3:	c6 05 86 0a 20 00 01 	movb   $0x1,0x200a86(%rip)        # 601040 <completed.7585>
  4005ba:	f3 c3                	repz retq 
  4005bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005c0 <frame_dummy>:
  4005c0:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  4005c5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005c9:	75 05                	jne    4005d0 <frame_dummy+0x10>
  4005cb:	eb 93                	jmp    400560 <register_tm_clones>
  4005cd:	0f 1f 00             	nopl   (%rax)
  4005d0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005d5:	48 85 c0             	test   %rax,%rax
  4005d8:	74 f1                	je     4005cb <frame_dummy+0xb>
  4005da:	55                   	push   %rbp
  4005db:	48 89 e5             	mov    %rsp,%rbp
  4005de:	ff d0                	callq  *%rax
  4005e0:	5d                   	pop    %rbp
  4005e1:	e9 7a ff ff ff       	jmpq   400560 <register_tm_clones>
  4005e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005ed:	00 00 00 

00000000004005f0 <_Z12new_instancev>:
  4005f0:	8b 05 8a 0a 20 00    	mov    0x200a8a(%rip),%eax        # 601080 <_ZN9TestClass7storageE>
  4005f6:	8d 50 01             	lea    0x1(%rax),%edx
  4005f9:	89 15 81 0a 20 00    	mov    %edx,0x200a81(%rip)        # 601080 <_ZN9TestClass7storageE>
  4005ff:	c3                   	retq   

0000000000400600 <_Z12write_field0P9TestClass>:
  400600:	c7 04 bd 84 10 60 00 	movl   $0x7777,0x601084(,%rdi,4)
  400607:	77 77 00 00 
  40060b:	c3                   	retq   
  40060c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400610 <_Z11read_field0P9TestClass>:
  400610:	8b 04 bd 84 10 60 00 	mov    0x601084(,%rdi,4),%eax
  400617:	c3                   	retq   
  400618:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40061f:	00 

0000000000400620 <_Z12write_field1P9TestClass>:
  400620:	c7 04 bd 54 59 60 00 	movl   $0x7777,0x605954(,%rdi,4)
  400627:	77 77 00 00 
  40062b:	c3                   	retq   
  40062c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400630 <_Z11read_field1P9TestClass>:
  400630:	8b 04 bd 54 59 60 00 	mov    0x605954(,%rdi,4),%eax
  400637:	c3                   	retq   
  400638:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40063f:	00 

0000000000400640 <_Z21explicit_write_field0m>:
  400640:	48 8b 05 19 0a 20 00 	mov    0x200a19(%rip),%rax        # 601060 <explicit_field0>
  400647:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40064e:	c3                   	retq   
  40064f:	90                   	nop

0000000000400650 <_Z20explicit_read_field0m>:
  400650:	48 8b 05 09 0a 20 00 	mov    0x200a09(%rip),%rax        # 601060 <explicit_field0>
  400657:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40065a:	c3                   	retq   
  40065b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400660 <__libc_csu_init>:
  400660:	41 57                	push   %r15
  400662:	41 56                	push   %r14
  400664:	41 89 ff             	mov    %edi,%r15d
  400667:	41 55                	push   %r13
  400669:	41 54                	push   %r12
  40066b:	4c 8d 25 9e 07 20 00 	lea    0x20079e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  400672:	55                   	push   %rbp
  400673:	48 8d 2d 9e 07 20 00 	lea    0x20079e(%rip),%rbp        # 600e18 <__init_array_end>
  40067a:	53                   	push   %rbx
  40067b:	49 89 f6             	mov    %rsi,%r14
  40067e:	49 89 d5             	mov    %rdx,%r13
  400681:	4c 29 e5             	sub    %r12,%rbp
  400684:	48 83 ec 08          	sub    $0x8,%rsp
  400688:	48 c1 fd 03          	sar    $0x3,%rbp
  40068c:	e8 5f fd ff ff       	callq  4003f0 <_init>
  400691:	48 85 ed             	test   %rbp,%rbp
  400694:	74 20                	je     4006b6 <__libc_csu_init+0x56>
  400696:	31 db                	xor    %ebx,%ebx
  400698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40069f:	00 
  4006a0:	4c 89 ea             	mov    %r13,%rdx
  4006a3:	4c 89 f6             	mov    %r14,%rsi
  4006a6:	44 89 ff             	mov    %r15d,%edi
  4006a9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4006ad:	48 83 c3 01          	add    $0x1,%rbx
  4006b1:	48 39 eb             	cmp    %rbp,%rbx
  4006b4:	75 ea                	jne    4006a0 <__libc_csu_init+0x40>
  4006b6:	48 83 c4 08          	add    $0x8,%rsp
  4006ba:	5b                   	pop    %rbx
  4006bb:	5d                   	pop    %rbp
  4006bc:	41 5c                	pop    %r12
  4006be:	41 5d                	pop    %r13
  4006c0:	41 5e                	pop    %r14
  4006c2:	41 5f                	pop    %r15
  4006c4:	c3                   	retq   
  4006c5:	90                   	nop
  4006c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006cd:	00 00 00 

00000000004006d0 <__libc_csu_fini>:
  4006d0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004006d4 <_fini>:
  4006d4:	48 83 ec 08          	sub    $0x8,%rsp
  4006d8:	48 83 c4 08          	add    $0x8,%rsp
  4006dc:	c3                   	retq   
