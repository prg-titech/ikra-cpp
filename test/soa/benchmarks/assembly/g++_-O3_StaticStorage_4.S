
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
  400450:	48 83 ec 08          	sub    $0x8,%rsp
  400454:	31 c0                	xor    %eax,%eax
  400456:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  40045b:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  400460:	be 24 07 40 00       	mov    $0x400724,%esi
  400465:	bf 01 00 00 00       	mov    $0x1,%edi
  40046a:	48 c7 05 83 0d 20 00 	movq   $0x0,0x200d83(%rip)        # 6011f8 <__TestClassdata_buffer+0x8>
  400471:	00 00 00 00 
  400475:	48 c7 05 70 0d 20 00 	movq   $0x1,0x200d70(%rip)        # 6011f0 <__TestClassdata_buffer>
  40047c:	01 00 00 00 
  400480:	c7 05 7a 0d 20 00 83 	movl   $0x27d22d83,0x200d7a(%rip)        # 601204 <__TestClassdata_buffer+0x14>
  400487:	2d d2 27 
  40048a:	c7 05 40 56 20 00 20 	movl   $0x24686420,0x205640(%rip)        # 605ad4 <__TestClassdata_buffer+0x48e4>
  400491:	64 68 24 
  400494:	e8 97 ff ff ff       	callq  400430 <__printf_chk@plt>
  400499:	81 3d 61 0d 20 00 83 	cmpl   $0x27d22d83,0x200d61(%rip)        # 601204 <__TestClassdata_buffer+0x14>
  4004a0:	2d d2 27 
  4004a3:	b8 01 00 00 00       	mov    $0x1,%eax
  4004a8:	75 0d                	jne    4004b7 <main+0x67>
  4004aa:	81 3d 20 56 20 00 20 	cmpl   $0x24686420,0x205620(%rip)        # 605ad4 <__TestClassdata_buffer+0x48e4>
  4004b1:	64 68 24 
  4004b4:	0f 95 c0             	setne  %al
  4004b7:	0f b6 c0             	movzbl %al,%eax
  4004ba:	48 83 c4 08          	add    $0x8,%rsp
  4004be:	c3                   	retq   
  4004bf:	90                   	nop

00000000004004c0 <_start>:
  4004c0:	31 ed                	xor    %ebp,%ebp
  4004c2:	49 89 d1             	mov    %rdx,%r9
  4004c5:	5e                   	pop    %rsi
  4004c6:	48 89 e2             	mov    %rsp,%rdx
  4004c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004cd:	50                   	push   %rax
  4004ce:	54                   	push   %rsp
  4004cf:	49 c7 c0 10 07 40 00 	mov    $0x400710,%r8
  4004d6:	48 c7 c1 a0 06 40 00 	mov    $0x4006a0,%rcx
  4004dd:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  4004e4:	e8 37 ff ff ff       	callq  400420 <__libc_start_main@plt>
  4004e9:	f4                   	hlt    
  4004ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004004f0 <deregister_tm_clones>:
  4004f0:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  4004f5:	55                   	push   %rbp
  4004f6:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  4004fc:	48 83 f8 0e          	cmp    $0xe,%rax
  400500:	48 89 e5             	mov    %rsp,%rbp
  400503:	76 1b                	jbe    400520 <deregister_tm_clones+0x30>
  400505:	b8 00 00 00 00       	mov    $0x0,%eax
  40050a:	48 85 c0             	test   %rax,%rax
  40050d:	74 11                	je     400520 <deregister_tm_clones+0x30>
  40050f:	5d                   	pop    %rbp
  400510:	bf 38 10 60 00       	mov    $0x601038,%edi
  400515:	ff e0                	jmpq   *%rax
  400517:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40051e:	00 00 
  400520:	5d                   	pop    %rbp
  400521:	c3                   	retq   
  400522:	0f 1f 40 00          	nopl   0x0(%rax)
  400526:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40052d:	00 00 00 

0000000000400530 <register_tm_clones>:
  400530:	be 38 10 60 00       	mov    $0x601038,%esi
  400535:	55                   	push   %rbp
  400536:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40053d:	48 c1 fe 03          	sar    $0x3,%rsi
  400541:	48 89 e5             	mov    %rsp,%rbp
  400544:	48 89 f0             	mov    %rsi,%rax
  400547:	48 c1 e8 3f          	shr    $0x3f,%rax
  40054b:	48 01 c6             	add    %rax,%rsi
  40054e:	48 d1 fe             	sar    %rsi
  400551:	74 15                	je     400568 <register_tm_clones+0x38>
  400553:	b8 00 00 00 00       	mov    $0x0,%eax
  400558:	48 85 c0             	test   %rax,%rax
  40055b:	74 0b                	je     400568 <register_tm_clones+0x38>
  40055d:	5d                   	pop    %rbp
  40055e:	bf 38 10 60 00       	mov    $0x601038,%edi
  400563:	ff e0                	jmpq   *%rax
  400565:	0f 1f 00             	nopl   (%rax)
  400568:	5d                   	pop    %rbp
  400569:	c3                   	retq   
  40056a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400570 <__do_global_dtors_aux>:
  400570:	80 3d c9 0a 20 00 00 	cmpb   $0x0,0x200ac9(%rip)        # 601040 <completed.7585>
  400577:	75 11                	jne    40058a <__do_global_dtors_aux+0x1a>
  400579:	55                   	push   %rbp
  40057a:	48 89 e5             	mov    %rsp,%rbp
  40057d:	e8 6e ff ff ff       	callq  4004f0 <deregister_tm_clones>
  400582:	5d                   	pop    %rbp
  400583:	c6 05 b6 0a 20 00 01 	movb   $0x1,0x200ab6(%rip)        # 601040 <completed.7585>
  40058a:	f3 c3                	repz retq 
  40058c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400590 <frame_dummy>:
  400590:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  400595:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400599:	75 05                	jne    4005a0 <frame_dummy+0x10>
  40059b:	eb 93                	jmp    400530 <register_tm_clones>
  40059d:	0f 1f 00             	nopl   (%rax)
  4005a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005a5:	48 85 c0             	test   %rax,%rax
  4005a8:	74 f1                	je     40059b <frame_dummy+0xb>
  4005aa:	55                   	push   %rbp
  4005ab:	48 89 e5             	mov    %rsp,%rbp
  4005ae:	ff d0                	callq  *%rax
  4005b0:	5d                   	pop    %rbp
  4005b1:	e9 7a ff ff ff       	jmpq   400530 <register_tm_clones>
  4005b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005bd:	00 00 00 

00000000004005c0 <_ZN9TestClass7storageEv>:
  4005c0:	b8 f0 11 60 00       	mov    $0x6011f0,%eax
  4005c5:	c3                   	retq   
  4005c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005cd:	00 00 00 

00000000004005d0 <_Z12new_instancev>:
  4005d0:	48 8b 05 19 0c 20 00 	mov    0x200c19(%rip),%rax        # 6011f0 <__TestClassdata_buffer>
  4005d7:	48 8d 50 01          	lea    0x1(%rax),%rdx
  4005db:	48 8d 04 95 00 12 60 	lea    0x601200(,%rdx,4),%rax
  4005e2:	00 
  4005e3:	48 89 15 06 0c 20 00 	mov    %rdx,0x200c06(%rip)        # 6011f0 <__TestClassdata_buffer>
  4005ea:	c6 04 95 00 12 60 00 	movb   $0x0,0x601200(,%rdx,4)
  4005f1:	00 
  4005f2:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  4005f6:	c6 40 02 00          	movb   $0x0,0x2(%rax)
  4005fa:	c6 40 03 00          	movb   $0x0,0x3(%rax)
  4005fe:	c3                   	retq   
  4005ff:	90                   	nop

0000000000400600 <_Z12write_field0P9TestClass>:
  400600:	48 81 ef 00 12 60 00 	sub    $0x601200,%rdi
  400607:	48 c1 ef 02          	shr    $0x2,%rdi
  40060b:	c7 04 bd 00 12 60 00 	movl   $0x7777,0x601200(,%rdi,4)
  400612:	77 77 00 00 
  400616:	c3                   	retq   
  400617:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40061e:	00 00 

0000000000400620 <_Z11read_field0P9TestClass>:
  400620:	48 81 ef 00 12 60 00 	sub    $0x601200,%rdi
  400627:	48 c1 ef 02          	shr    $0x2,%rdi
  40062b:	8b 04 bd 00 12 60 00 	mov    0x601200(,%rdi,4),%eax
  400632:	c3                   	retq   
  400633:	0f 1f 00             	nopl   (%rax)
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <_Z12write_field1P9TestClass>:
  400640:	48 81 ef 00 12 60 00 	sub    $0x601200,%rdi
  400647:	48 c1 ef 02          	shr    $0x2,%rdi
  40064b:	c7 04 bd d0 5a 60 00 	movl   $0x7777,0x605ad0(,%rdi,4)
  400652:	77 77 00 00 
  400656:	c3                   	retq   
  400657:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40065e:	00 00 

0000000000400660 <_Z11read_field1P9TestClass>:
  400660:	48 81 ef 00 12 60 00 	sub    $0x601200,%rdi
  400667:	48 c1 ef 02          	shr    $0x2,%rdi
  40066b:	8b 04 bd d0 5a 60 00 	mov    0x605ad0(,%rdi,4),%eax
  400672:	c3                   	retq   
  400673:	0f 1f 00             	nopl   (%rax)
  400676:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40067d:	00 00 00 

0000000000400680 <_Z21explicit_write_field0m>:
  400680:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  400687:	77 77 00 00 
  40068b:	c3                   	retq   
  40068c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400690 <_Z20explicit_read_field0m>:
  400690:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  400697:	c3                   	retq   
  400698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40069f:	00 

00000000004006a0 <__libc_csu_init>:
  4006a0:	41 57                	push   %r15
  4006a2:	41 56                	push   %r14
  4006a4:	41 89 ff             	mov    %edi,%r15d
  4006a7:	41 55                	push   %r13
  4006a9:	41 54                	push   %r12
  4006ab:	4c 8d 25 5e 07 20 00 	lea    0x20075e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  4006b2:	55                   	push   %rbp
  4006b3:	48 8d 2d 5e 07 20 00 	lea    0x20075e(%rip),%rbp        # 600e18 <__init_array_end>
  4006ba:	53                   	push   %rbx
  4006bb:	49 89 f6             	mov    %rsi,%r14
  4006be:	49 89 d5             	mov    %rdx,%r13
  4006c1:	4c 29 e5             	sub    %r12,%rbp
  4006c4:	48 83 ec 08          	sub    $0x8,%rsp
  4006c8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006cc:	e8 1f fd ff ff       	callq  4003f0 <_init>
  4006d1:	48 85 ed             	test   %rbp,%rbp
  4006d4:	74 20                	je     4006f6 <__libc_csu_init+0x56>
  4006d6:	31 db                	xor    %ebx,%ebx
  4006d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006df:	00 
  4006e0:	4c 89 ea             	mov    %r13,%rdx
  4006e3:	4c 89 f6             	mov    %r14,%rsi
  4006e6:	44 89 ff             	mov    %r15d,%edi
  4006e9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4006ed:	48 83 c3 01          	add    $0x1,%rbx
  4006f1:	48 39 eb             	cmp    %rbp,%rbx
  4006f4:	75 ea                	jne    4006e0 <__libc_csu_init+0x40>
  4006f6:	48 83 c4 08          	add    $0x8,%rsp
  4006fa:	5b                   	pop    %rbx
  4006fb:	5d                   	pop    %rbp
  4006fc:	41 5c                	pop    %r12
  4006fe:	41 5d                	pop    %r13
  400700:	41 5e                	pop    %r14
  400702:	41 5f                	pop    %r15
  400704:	c3                   	retq   
  400705:	90                   	nop
  400706:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40070d:	00 00 00 

0000000000400710 <__libc_csu_fini>:
  400710:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400714 <_fini>:
  400714:	48 83 ec 08          	sub    $0x8,%rsp
  400718:	48 83 c4 08          	add    $0x8,%rsp
  40071c:	c3                   	retq   
