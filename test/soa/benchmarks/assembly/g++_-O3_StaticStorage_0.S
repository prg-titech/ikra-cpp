
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
  400452:	be 14 07 40 00       	mov    $0x400714,%esi
  400457:	bf 01 00 00 00       	mov    $0x1,%edi
  40045c:	48 83 ec 08          	sub    $0x8,%rsp
  400460:	48 8b 05 99 0d 20 00 	mov    0x200d99(%rip),%rax        # 601200 <__TestClassdata_buffer>
  400467:	48 c7 05 36 9f 20 00 	movq   $0x0,0x209f36(%rip)        # 60a3a8 <__TestClassdata_buffer+0x91a8>
  40046e:	00 00 00 00 
  400472:	48 8d 58 01          	lea    0x1(%rax),%rbx
  400476:	48 8d 2c 9d d0 48 00 	lea    0x48d0(,%rbx,4),%rbp
  40047d:	00 
  40047e:	48 89 1d 7b 0d 20 00 	mov    %rbx,0x200d7b(%rip)        # 601200 <__TestClassdata_buffer>
  400485:	c7 04 9d 08 12 60 00 	movl   $0x7777,0x601208(,%rbx,4)
  40048c:	77 77 00 00 
  400490:	c7 85 08 12 60 00 88 	movl   $0x8888,0x601208(%rbp)
  400497:	88 00 00 
  40049a:	69 04 9d 08 12 60 00 	imul   $0x5555,0x601208(,%rbx,4),%eax
  4004a1:	55 55 00 00 
  4004a5:	89 04 9d 08 12 60 00 	mov    %eax,0x601208(,%rbx,4)
  4004ac:	31 c0                	xor    %eax,%eax
  4004ae:	69 8d 08 12 60 00 44 	imul   $0x4444,0x601208(%rbp),%ecx
  4004b5:	44 00 00 
  4004b8:	89 8d 08 12 60 00    	mov    %ecx,0x601208(%rbp)
  4004be:	8b 14 9d 08 12 60 00 	mov    0x601208(,%rbx,4),%edx
  4004c5:	e8 66 ff ff ff       	callq  400430 <__printf_chk@plt>
  4004ca:	81 3c 9d 08 12 60 00 	cmpl   $0x27d22d83,0x601208(,%rbx,4)
  4004d1:	83 2d d2 27 
  4004d5:	b8 01 00 00 00       	mov    $0x1,%eax
  4004da:	75 0d                	jne    4004e9 <main+0x99>
  4004dc:	81 bd 08 12 60 00 20 	cmpl   $0x24686420,0x601208(%rbp)
  4004e3:	64 68 24 
  4004e6:	0f 95 c0             	setne  %al
  4004e9:	48 83 c4 08          	add    $0x8,%rsp
  4004ed:	0f b6 c0             	movzbl %al,%eax
  4004f0:	5b                   	pop    %rbx
  4004f1:	5d                   	pop    %rbp
  4004f2:	c3                   	retq   
  4004f3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4004fa:	00 00 00 
  4004fd:	0f 1f 00             	nopl   (%rax)

0000000000400500 <_start>:
  400500:	31 ed                	xor    %ebp,%ebp
  400502:	49 89 d1             	mov    %rdx,%r9
  400505:	5e                   	pop    %rsi
  400506:	48 89 e2             	mov    %rsp,%rdx
  400509:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40050d:	50                   	push   %rax
  40050e:	54                   	push   %rsp
  40050f:	49 c7 c0 00 07 40 00 	mov    $0x400700,%r8
  400516:	48 c7 c1 90 06 40 00 	mov    $0x400690,%rcx
  40051d:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  400524:	e8 f7 fe ff ff       	callq  400420 <__libc_start_main@plt>
  400529:	f4                   	hlt    
  40052a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400530 <deregister_tm_clones>:
  400530:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  400535:	55                   	push   %rbp
  400536:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  40053c:	48 83 f8 0e          	cmp    $0xe,%rax
  400540:	48 89 e5             	mov    %rsp,%rbp
  400543:	76 1b                	jbe    400560 <deregister_tm_clones+0x30>
  400545:	b8 00 00 00 00       	mov    $0x0,%eax
  40054a:	48 85 c0             	test   %rax,%rax
  40054d:	74 11                	je     400560 <deregister_tm_clones+0x30>
  40054f:	5d                   	pop    %rbp
  400550:	bf 38 10 60 00       	mov    $0x601038,%edi
  400555:	ff e0                	jmpq   *%rax
  400557:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40055e:	00 00 
  400560:	5d                   	pop    %rbp
  400561:	c3                   	retq   
  400562:	0f 1f 40 00          	nopl   0x0(%rax)
  400566:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40056d:	00 00 00 

0000000000400570 <register_tm_clones>:
  400570:	be 38 10 60 00       	mov    $0x601038,%esi
  400575:	55                   	push   %rbp
  400576:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40057d:	48 c1 fe 03          	sar    $0x3,%rsi
  400581:	48 89 e5             	mov    %rsp,%rbp
  400584:	48 89 f0             	mov    %rsi,%rax
  400587:	48 c1 e8 3f          	shr    $0x3f,%rax
  40058b:	48 01 c6             	add    %rax,%rsi
  40058e:	48 d1 fe             	sar    %rsi
  400591:	74 15                	je     4005a8 <register_tm_clones+0x38>
  400593:	b8 00 00 00 00       	mov    $0x0,%eax
  400598:	48 85 c0             	test   %rax,%rax
  40059b:	74 0b                	je     4005a8 <register_tm_clones+0x38>
  40059d:	5d                   	pop    %rbp
  40059e:	bf 38 10 60 00       	mov    $0x601038,%edi
  4005a3:	ff e0                	jmpq   *%rax
  4005a5:	0f 1f 00             	nopl   (%rax)
  4005a8:	5d                   	pop    %rbp
  4005a9:	c3                   	retq   
  4005aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005b0 <__do_global_dtors_aux>:
  4005b0:	80 3d 89 0a 20 00 00 	cmpb   $0x0,0x200a89(%rip)        # 601040 <completed.7585>
  4005b7:	75 11                	jne    4005ca <__do_global_dtors_aux+0x1a>
  4005b9:	55                   	push   %rbp
  4005ba:	48 89 e5             	mov    %rsp,%rbp
  4005bd:	e8 6e ff ff ff       	callq  400530 <deregister_tm_clones>
  4005c2:	5d                   	pop    %rbp
  4005c3:	c6 05 76 0a 20 00 01 	movb   $0x1,0x200a76(%rip)        # 601040 <completed.7585>
  4005ca:	f3 c3                	repz retq 
  4005cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005d0 <frame_dummy>:
  4005d0:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  4005d5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005d9:	75 05                	jne    4005e0 <frame_dummy+0x10>
  4005db:	eb 93                	jmp    400570 <register_tm_clones>
  4005dd:	0f 1f 00             	nopl   (%rax)
  4005e0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005e5:	48 85 c0             	test   %rax,%rax
  4005e8:	74 f1                	je     4005db <frame_dummy+0xb>
  4005ea:	55                   	push   %rbp
  4005eb:	48 89 e5             	mov    %rsp,%rbp
  4005ee:	ff d0                	callq  *%rax
  4005f0:	5d                   	pop    %rbp
  4005f1:	e9 7a ff ff ff       	jmpq   400570 <register_tm_clones>
  4005f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005fd:	00 00 00 

0000000000400600 <_ZN9TestClass7storageEv>:
  400600:	b8 00 12 60 00       	mov    $0x601200,%eax
  400605:	c3                   	retq   
  400606:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40060d:	00 00 00 

0000000000400610 <_Z12new_instancev>:
  400610:	48 8b 05 e9 0b 20 00 	mov    0x200be9(%rip),%rax        # 601200 <__TestClassdata_buffer>
  400617:	48 83 c0 01          	add    $0x1,%rax
  40061b:	48 89 05 de 0b 20 00 	mov    %rax,0x200bde(%rip)        # 601200 <__TestClassdata_buffer>
  400622:	c3                   	retq   
  400623:	0f 1f 00             	nopl   (%rax)
  400626:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40062d:	00 00 00 

0000000000400630 <_Z12write_field0P9TestClass>:
  400630:	c7 04 bd 08 12 60 00 	movl   $0x7777,0x601208(,%rdi,4)
  400637:	77 77 00 00 
  40063b:	c3                   	retq   
  40063c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400640 <_Z11read_field0P9TestClass>:
  400640:	8b 04 bd 08 12 60 00 	mov    0x601208(,%rdi,4),%eax
  400647:	c3                   	retq   
  400648:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40064f:	00 

0000000000400650 <_Z12write_field1P9TestClass>:
  400650:	c7 04 bd d8 5a 60 00 	movl   $0x7777,0x605ad8(,%rdi,4)
  400657:	77 77 00 00 
  40065b:	c3                   	retq   
  40065c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400660 <_Z11read_field1P9TestClass>:
  400660:	8b 04 bd d8 5a 60 00 	mov    0x605ad8(,%rdi,4),%eax
  400667:	c3                   	retq   
  400668:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40066f:	00 

0000000000400670 <_Z21explicit_write_field0m>:
  400670:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  400677:	77 77 00 00 
  40067b:	c3                   	retq   
  40067c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400680 <_Z20explicit_read_field0m>:
  400680:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  400687:	c3                   	retq   
  400688:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40068f:	00 

0000000000400690 <__libc_csu_init>:
  400690:	41 57                	push   %r15
  400692:	41 56                	push   %r14
  400694:	41 89 ff             	mov    %edi,%r15d
  400697:	41 55                	push   %r13
  400699:	41 54                	push   %r12
  40069b:	4c 8d 25 6e 07 20 00 	lea    0x20076e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  4006a2:	55                   	push   %rbp
  4006a3:	48 8d 2d 6e 07 20 00 	lea    0x20076e(%rip),%rbp        # 600e18 <__init_array_end>
  4006aa:	53                   	push   %rbx
  4006ab:	49 89 f6             	mov    %rsi,%r14
  4006ae:	49 89 d5             	mov    %rdx,%r13
  4006b1:	4c 29 e5             	sub    %r12,%rbp
  4006b4:	48 83 ec 08          	sub    $0x8,%rsp
  4006b8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006bc:	e8 2f fd ff ff       	callq  4003f0 <_init>
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
