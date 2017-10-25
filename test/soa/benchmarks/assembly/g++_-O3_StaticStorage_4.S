
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
  400452:	be 64 07 40 00       	mov    $0x400764,%esi
  400457:	bf 01 00 00 00       	mov    $0x1,%edi
  40045c:	48 83 ec 08          	sub    $0x8,%rsp
  400460:	48 8b 05 19 0c 20 00 	mov    0x200c19(%rip),%rax        # 601080 <_ZN9TestClass7storageE>
  400467:	48 8d 1c 85 88 10 60 	lea    0x601088(,%rax,4),%rbx
  40046e:	00 
  40046f:	48 8d 50 01          	lea    0x1(%rax),%rdx
  400473:	48 89 15 06 0c 20 00 	mov    %rdx,0x200c06(%rip)        # 601080 <_ZN9TestClass7storageE>
  40047a:	c6 04 85 88 10 60 00 	movb   $0x0,0x601088(,%rax,4)
  400481:	00 
  400482:	c6 43 01 00          	movb   $0x0,0x1(%rbx)
  400486:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  40048a:	c6 43 03 00          	movb   $0x0,0x3(%rbx)
  40048e:	48 81 eb 88 10 60 00 	sub    $0x601088,%rbx
  400495:	48 c1 eb 02          	shr    $0x2,%rbx
  400499:	48 8d 2c 9d 88 10 60 	lea    0x601088(,%rbx,4),%rbp
  4004a0:	00 
  4004a1:	c7 04 9d 88 10 60 00 	movl   $0x7777,0x601088(,%rbx,4)
  4004a8:	77 77 00 00 
  4004ac:	c7 85 d0 48 00 00 88 	movl   $0x8888,0x48d0(%rbp)
  4004b3:	88 00 00 
  4004b6:	69 04 9d 88 10 60 00 	imul   $0x5555,0x601088(,%rbx,4),%eax
  4004bd:	55 55 00 00 
  4004c1:	89 04 9d 88 10 60 00 	mov    %eax,0x601088(,%rbx,4)
  4004c8:	31 c0                	xor    %eax,%eax
  4004ca:	69 8d d0 48 00 00 44 	imul   $0x4444,0x48d0(%rbp),%ecx
  4004d1:	44 00 00 
  4004d4:	89 8d d0 48 00 00    	mov    %ecx,0x48d0(%rbp)
  4004da:	8b 14 9d 88 10 60 00 	mov    0x601088(,%rbx,4),%edx
  4004e1:	e8 4a ff ff ff       	callq  400430 <__printf_chk@plt>
  4004e6:	81 3c 9d 88 10 60 00 	cmpl   $0x27d22d83,0x601088(,%rbx,4)
  4004ed:	83 2d d2 27 
  4004f1:	b8 01 00 00 00       	mov    $0x1,%eax
  4004f6:	75 0d                	jne    400505 <main+0xb5>
  4004f8:	81 bd d0 48 00 00 20 	cmpl   $0x24686420,0x48d0(%rbp)
  4004ff:	64 68 24 
  400502:	0f 95 c0             	setne  %al
  400505:	48 83 c4 08          	add    $0x8,%rsp
  400509:	0f b6 c0             	movzbl %al,%eax
  40050c:	5b                   	pop    %rbx
  40050d:	5d                   	pop    %rbp
  40050e:	c3                   	retq   
  40050f:	90                   	nop

0000000000400510 <_start>:
  400510:	31 ed                	xor    %ebp,%ebp
  400512:	49 89 d1             	mov    %rdx,%r9
  400515:	5e                   	pop    %rsi
  400516:	48 89 e2             	mov    %rsp,%rdx
  400519:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40051d:	50                   	push   %rax
  40051e:	54                   	push   %rsp
  40051f:	49 c7 c0 50 07 40 00 	mov    $0x400750,%r8
  400526:	48 c7 c1 e0 06 40 00 	mov    $0x4006e0,%rcx
  40052d:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  400534:	e8 e7 fe ff ff       	callq  400420 <__libc_start_main@plt>
  400539:	f4                   	hlt    
  40053a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400540 <deregister_tm_clones>:
  400540:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  400545:	55                   	push   %rbp
  400546:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  40054c:	48 83 f8 0e          	cmp    $0xe,%rax
  400550:	48 89 e5             	mov    %rsp,%rbp
  400553:	76 1b                	jbe    400570 <deregister_tm_clones+0x30>
  400555:	b8 00 00 00 00       	mov    $0x0,%eax
  40055a:	48 85 c0             	test   %rax,%rax
  40055d:	74 11                	je     400570 <deregister_tm_clones+0x30>
  40055f:	5d                   	pop    %rbp
  400560:	bf 38 10 60 00       	mov    $0x601038,%edi
  400565:	ff e0                	jmpq   *%rax
  400567:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40056e:	00 00 
  400570:	5d                   	pop    %rbp
  400571:	c3                   	retq   
  400572:	0f 1f 40 00          	nopl   0x0(%rax)
  400576:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40057d:	00 00 00 

0000000000400580 <register_tm_clones>:
  400580:	be 38 10 60 00       	mov    $0x601038,%esi
  400585:	55                   	push   %rbp
  400586:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40058d:	48 c1 fe 03          	sar    $0x3,%rsi
  400591:	48 89 e5             	mov    %rsp,%rbp
  400594:	48 89 f0             	mov    %rsi,%rax
  400597:	48 c1 e8 3f          	shr    $0x3f,%rax
  40059b:	48 01 c6             	add    %rax,%rsi
  40059e:	48 d1 fe             	sar    %rsi
  4005a1:	74 15                	je     4005b8 <register_tm_clones+0x38>
  4005a3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005a8:	48 85 c0             	test   %rax,%rax
  4005ab:	74 0b                	je     4005b8 <register_tm_clones+0x38>
  4005ad:	5d                   	pop    %rbp
  4005ae:	bf 38 10 60 00       	mov    $0x601038,%edi
  4005b3:	ff e0                	jmpq   *%rax
  4005b5:	0f 1f 00             	nopl   (%rax)
  4005b8:	5d                   	pop    %rbp
  4005b9:	c3                   	retq   
  4005ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005c0 <__do_global_dtors_aux>:
  4005c0:	80 3d 79 0a 20 00 00 	cmpb   $0x0,0x200a79(%rip)        # 601040 <completed.7585>
  4005c7:	75 11                	jne    4005da <__do_global_dtors_aux+0x1a>
  4005c9:	55                   	push   %rbp
  4005ca:	48 89 e5             	mov    %rsp,%rbp
  4005cd:	e8 6e ff ff ff       	callq  400540 <deregister_tm_clones>
  4005d2:	5d                   	pop    %rbp
  4005d3:	c6 05 66 0a 20 00 01 	movb   $0x1,0x200a66(%rip)        # 601040 <completed.7585>
  4005da:	f3 c3                	repz retq 
  4005dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005e0 <frame_dummy>:
  4005e0:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  4005e5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005e9:	75 05                	jne    4005f0 <frame_dummy+0x10>
  4005eb:	eb 93                	jmp    400580 <register_tm_clones>
  4005ed:	0f 1f 00             	nopl   (%rax)
  4005f0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005f5:	48 85 c0             	test   %rax,%rax
  4005f8:	74 f1                	je     4005eb <frame_dummy+0xb>
  4005fa:	55                   	push   %rbp
  4005fb:	48 89 e5             	mov    %rsp,%rbp
  4005fe:	ff d0                	callq  *%rax
  400600:	5d                   	pop    %rbp
  400601:	e9 7a ff ff ff       	jmpq   400580 <register_tm_clones>
  400606:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40060d:	00 00 00 

0000000000400610 <_Z12new_instancev>:
  400610:	48 8b 15 69 0a 20 00 	mov    0x200a69(%rip),%rdx        # 601080 <_ZN9TestClass7storageE>
  400617:	48 8d 42 01          	lea    0x1(%rdx),%rax
  40061b:	48 89 05 5e 0a 20 00 	mov    %rax,0x200a5e(%rip)        # 601080 <_ZN9TestClass7storageE>
  400622:	48 8d 04 95 88 10 60 	lea    0x601088(,%rdx,4),%rax
  400629:	00 
  40062a:	c6 04 95 88 10 60 00 	movb   $0x0,0x601088(,%rdx,4)
  400631:	00 
  400632:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  400636:	c6 40 02 00          	movb   $0x0,0x2(%rax)
  40063a:	c6 40 03 00          	movb   $0x0,0x3(%rax)
  40063e:	c3                   	retq   
  40063f:	90                   	nop

0000000000400640 <_Z12write_field0P9TestClass>:
  400640:	48 81 ef 88 10 60 00 	sub    $0x601088,%rdi
  400647:	48 c1 ef 02          	shr    $0x2,%rdi
  40064b:	c7 04 bd 88 10 60 00 	movl   $0x7777,0x601088(,%rdi,4)
  400652:	77 77 00 00 
  400656:	c3                   	retq   
  400657:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40065e:	00 00 

0000000000400660 <_Z11read_field0P9TestClass>:
  400660:	48 81 ef 88 10 60 00 	sub    $0x601088,%rdi
  400667:	48 c1 ef 02          	shr    $0x2,%rdi
  40066b:	8b 04 bd 88 10 60 00 	mov    0x601088(,%rdi,4),%eax
  400672:	c3                   	retq   
  400673:	0f 1f 00             	nopl   (%rax)
  400676:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40067d:	00 00 00 

0000000000400680 <_Z12write_field1P9TestClass>:
  400680:	48 81 ef 88 10 60 00 	sub    $0x601088,%rdi
  400687:	48 c1 ef 02          	shr    $0x2,%rdi
  40068b:	c7 04 bd 58 59 60 00 	movl   $0x7777,0x605958(,%rdi,4)
  400692:	77 77 00 00 
  400696:	c3                   	retq   
  400697:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40069e:	00 00 

00000000004006a0 <_Z11read_field1P9TestClass>:
  4006a0:	48 81 ef 88 10 60 00 	sub    $0x601088,%rdi
  4006a7:	48 c1 ef 02          	shr    $0x2,%rdi
  4006ab:	8b 04 bd 58 59 60 00 	mov    0x605958(,%rdi,4),%eax
  4006b2:	c3                   	retq   
  4006b3:	0f 1f 00             	nopl   (%rax)
  4006b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006bd:	00 00 00 

00000000004006c0 <_Z21explicit_write_field0m>:
  4006c0:	48 8b 05 99 09 20 00 	mov    0x200999(%rip),%rax        # 601060 <explicit_field0>
  4006c7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006ce:	c3                   	retq   
  4006cf:	90                   	nop

00000000004006d0 <_Z20explicit_read_field0m>:
  4006d0:	48 8b 05 89 09 20 00 	mov    0x200989(%rip),%rax        # 601060 <explicit_field0>
  4006d7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006da:	c3                   	retq   
  4006db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004006e0 <__libc_csu_init>:
  4006e0:	41 57                	push   %r15
  4006e2:	41 56                	push   %r14
  4006e4:	41 89 ff             	mov    %edi,%r15d
  4006e7:	41 55                	push   %r13
  4006e9:	41 54                	push   %r12
  4006eb:	4c 8d 25 1e 07 20 00 	lea    0x20071e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  4006f2:	55                   	push   %rbp
  4006f3:	48 8d 2d 1e 07 20 00 	lea    0x20071e(%rip),%rbp        # 600e18 <__init_array_end>
  4006fa:	53                   	push   %rbx
  4006fb:	49 89 f6             	mov    %rsi,%r14
  4006fe:	49 89 d5             	mov    %rdx,%r13
  400701:	4c 29 e5             	sub    %r12,%rbp
  400704:	48 83 ec 08          	sub    $0x8,%rsp
  400708:	48 c1 fd 03          	sar    $0x3,%rbp
  40070c:	e8 df fc ff ff       	callq  4003f0 <_init>
  400711:	48 85 ed             	test   %rbp,%rbp
  400714:	74 20                	je     400736 <__libc_csu_init+0x56>
  400716:	31 db                	xor    %ebx,%ebx
  400718:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40071f:	00 
  400720:	4c 89 ea             	mov    %r13,%rdx
  400723:	4c 89 f6             	mov    %r14,%rsi
  400726:	44 89 ff             	mov    %r15d,%edi
  400729:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40072d:	48 83 c3 01          	add    $0x1,%rbx
  400731:	48 39 eb             	cmp    %rbp,%rbx
  400734:	75 ea                	jne    400720 <__libc_csu_init+0x40>
  400736:	48 83 c4 08          	add    $0x8,%rsp
  40073a:	5b                   	pop    %rbx
  40073b:	5d                   	pop    %rbp
  40073c:	41 5c                	pop    %r12
  40073e:	41 5d                	pop    %r13
  400740:	41 5e                	pop    %r14
  400742:	41 5f                	pop    %r15
  400744:	c3                   	retq   
  400745:	90                   	nop
  400746:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40074d:	00 00 00 

0000000000400750 <__libc_csu_fini>:
  400750:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400754 <_fini>:
  400754:	48 83 ec 08          	sub    $0x8,%rsp
  400758:	48 83 c4 08          	add    $0x8,%rsp
  40075c:	c3                   	retq   
