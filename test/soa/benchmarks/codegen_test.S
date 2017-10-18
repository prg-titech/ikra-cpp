
a.out:     file format elf64-x86-64


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
  400450:	48 83 ec 08          	sub    $0x8,%rsp
  400454:	8b 05 06 0c 20 00    	mov    0x200c06(%rip),%eax        # 601060 <_ZN9TestClass7storageE>
  40045a:	be 54 06 40 00       	mov    $0x400654,%esi
  40045f:	bf 01 00 00 00       	mov    $0x1,%edi
  400464:	8d 50 01             	lea    0x1(%rax),%edx
  400467:	89 15 f3 0b 20 00    	mov    %edx,0x200bf3(%rip)        # 601060 <_ZN9TestClass7storageE>
  40046d:	48 8d 14 85 d0 48 00 	lea    0x48d0(,%rax,4),%rdx
  400474:	00 
  400475:	c7 04 85 64 10 60 00 	movl   $0x7777,0x601064(,%rax,4)
  40047c:	77 77 00 00 
  400480:	c7 82 64 10 60 00 88 	movl   $0x8888,0x601064(%rdx)
  400487:	88 00 00 
  40048a:	69 0c 85 64 10 60 00 	imul   $0x5555,0x601064(,%rax,4),%ecx
  400491:	55 55 00 00 
  400495:	89 0c 85 64 10 60 00 	mov    %ecx,0x601064(,%rax,4)
  40049c:	69 8a 64 10 60 00 44 	imul   $0x4444,0x601064(%rdx),%ecx
  4004a3:	44 00 00 
  4004a6:	89 8a 64 10 60 00    	mov    %ecx,0x601064(%rdx)
  4004ac:	8b 14 85 64 10 60 00 	mov    0x601064(,%rax,4),%edx
  4004b3:	31 c0                	xor    %eax,%eax
  4004b5:	e8 76 ff ff ff       	callq  400430 <__printf_chk@plt>
  4004ba:	31 c0                	xor    %eax,%eax
  4004bc:	48 83 c4 08          	add    $0x8,%rsp
  4004c0:	c3                   	retq   
  4004c1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4004c8:	00 00 00 
  4004cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004004d0 <_start>:
  4004d0:	31 ed                	xor    %ebp,%ebp
  4004d2:	49 89 d1             	mov    %rdx,%r9
  4004d5:	5e                   	pop    %rsi
  4004d6:	48 89 e2             	mov    %rsp,%rdx
  4004d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004dd:	50                   	push   %rax
  4004de:	54                   	push   %rsp
  4004df:	49 c7 c0 40 06 40 00 	mov    $0x400640,%r8
  4004e6:	48 c7 c1 d0 05 40 00 	mov    $0x4005d0,%rcx
  4004ed:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  4004f4:	e8 27 ff ff ff       	callq  400420 <__libc_start_main@plt>
  4004f9:	f4                   	hlt    
  4004fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400500 <deregister_tm_clones>:
  400500:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  400505:	55                   	push   %rbp
  400506:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  40050c:	48 83 f8 0e          	cmp    $0xe,%rax
  400510:	48 89 e5             	mov    %rsp,%rbp
  400513:	76 1b                	jbe    400530 <deregister_tm_clones+0x30>
  400515:	b8 00 00 00 00       	mov    $0x0,%eax
  40051a:	48 85 c0             	test   %rax,%rax
  40051d:	74 11                	je     400530 <deregister_tm_clones+0x30>
  40051f:	5d                   	pop    %rbp
  400520:	bf 38 10 60 00       	mov    $0x601038,%edi
  400525:	ff e0                	jmpq   *%rax
  400527:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40052e:	00 00 
  400530:	5d                   	pop    %rbp
  400531:	c3                   	retq   
  400532:	0f 1f 40 00          	nopl   0x0(%rax)
  400536:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40053d:	00 00 00 

0000000000400540 <register_tm_clones>:
  400540:	be 38 10 60 00       	mov    $0x601038,%esi
  400545:	55                   	push   %rbp
  400546:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40054d:	48 c1 fe 03          	sar    $0x3,%rsi
  400551:	48 89 e5             	mov    %rsp,%rbp
  400554:	48 89 f0             	mov    %rsi,%rax
  400557:	48 c1 e8 3f          	shr    $0x3f,%rax
  40055b:	48 01 c6             	add    %rax,%rsi
  40055e:	48 d1 fe             	sar    %rsi
  400561:	74 15                	je     400578 <register_tm_clones+0x38>
  400563:	b8 00 00 00 00       	mov    $0x0,%eax
  400568:	48 85 c0             	test   %rax,%rax
  40056b:	74 0b                	je     400578 <register_tm_clones+0x38>
  40056d:	5d                   	pop    %rbp
  40056e:	bf 38 10 60 00       	mov    $0x601038,%edi
  400573:	ff e0                	jmpq   *%rax
  400575:	0f 1f 00             	nopl   (%rax)
  400578:	5d                   	pop    %rbp
  400579:	c3                   	retq   
  40057a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400580 <__do_global_dtors_aux>:
  400580:	80 3d b9 0a 20 00 00 	cmpb   $0x0,0x200ab9(%rip)        # 601040 <completed.7585>
  400587:	75 11                	jne    40059a <__do_global_dtors_aux+0x1a>
  400589:	55                   	push   %rbp
  40058a:	48 89 e5             	mov    %rsp,%rbp
  40058d:	e8 6e ff ff ff       	callq  400500 <deregister_tm_clones>
  400592:	5d                   	pop    %rbp
  400593:	c6 05 a6 0a 20 00 01 	movb   $0x1,0x200aa6(%rip)        # 601040 <completed.7585>
  40059a:	f3 c3                	repz retq 
  40059c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005a0 <frame_dummy>:
  4005a0:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  4005a5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005a9:	75 05                	jne    4005b0 <frame_dummy+0x10>
  4005ab:	eb 93                	jmp    400540 <register_tm_clones>
  4005ad:	0f 1f 00             	nopl   (%rax)
  4005b0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005b5:	48 85 c0             	test   %rax,%rax
  4005b8:	74 f1                	je     4005ab <frame_dummy+0xb>
  4005ba:	55                   	push   %rbp
  4005bb:	48 89 e5             	mov    %rsp,%rbp
  4005be:	ff d0                	callq  *%rax
  4005c0:	5d                   	pop    %rbp
  4005c1:	e9 7a ff ff ff       	jmpq   400540 <register_tm_clones>
  4005c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005cd:	00 00 00 

00000000004005d0 <__libc_csu_init>:
  4005d0:	41 57                	push   %r15
  4005d2:	41 56                	push   %r14
  4005d4:	41 89 ff             	mov    %edi,%r15d
  4005d7:	41 55                	push   %r13
  4005d9:	41 54                	push   %r12
  4005db:	4c 8d 25 2e 08 20 00 	lea    0x20082e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  4005e2:	55                   	push   %rbp
  4005e3:	48 8d 2d 2e 08 20 00 	lea    0x20082e(%rip),%rbp        # 600e18 <__init_array_end>
  4005ea:	53                   	push   %rbx
  4005eb:	49 89 f6             	mov    %rsi,%r14
  4005ee:	49 89 d5             	mov    %rdx,%r13
  4005f1:	4c 29 e5             	sub    %r12,%rbp
  4005f4:	48 83 ec 08          	sub    $0x8,%rsp
  4005f8:	48 c1 fd 03          	sar    $0x3,%rbp
  4005fc:	e8 ef fd ff ff       	callq  4003f0 <_init>
  400601:	48 85 ed             	test   %rbp,%rbp
  400604:	74 20                	je     400626 <__libc_csu_init+0x56>
  400606:	31 db                	xor    %ebx,%ebx
  400608:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40060f:	00 
  400610:	4c 89 ea             	mov    %r13,%rdx
  400613:	4c 89 f6             	mov    %r14,%rsi
  400616:	44 89 ff             	mov    %r15d,%edi
  400619:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40061d:	48 83 c3 01          	add    $0x1,%rbx
  400621:	48 39 eb             	cmp    %rbp,%rbx
  400624:	75 ea                	jne    400610 <__libc_csu_init+0x40>
  400626:	48 83 c4 08          	add    $0x8,%rsp
  40062a:	5b                   	pop    %rbx
  40062b:	5d                   	pop    %rbp
  40062c:	41 5c                	pop    %r12
  40062e:	41 5d                	pop    %r13
  400630:	41 5e                	pop    %r14
  400632:	41 5f                	pop    %r15
  400634:	c3                   	retq   
  400635:	90                   	nop
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <__libc_csu_fini>:
  400640:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400644 <_fini>:
  400644:	48 83 ec 08          	sub    $0x8,%rsp
  400648:	48 83 c4 08          	add    $0x8,%rsp
  40064c:	c3                   	retq   
