
bin/clang++-3.8_-O3_DynamicStorage_4:     file format elf64-x86-64


Disassembly of section .init:

00000000004004c0 <_init>:
  4004c0:	48 83 ec 08          	sub    $0x8,%rsp
  4004c4:	48 8b 05 2d 0b 20 00 	mov    0x200b2d(%rip),%rax        # 600ff8 <_DYNAMIC+0x200>
  4004cb:	48 85 c0             	test   %rax,%rax
  4004ce:	74 05                	je     4004d5 <_init+0x15>
  4004d0:	e8 4b 00 00 00       	callq  400520 <__libc_start_main@plt+0x10>
  4004d5:	48 83 c4 08          	add    $0x8,%rsp
  4004d9:	c3                   	retq   

Disassembly of section .plt:

00000000004004e0 <printf@plt-0x10>:
  4004e0:	ff 35 22 0b 20 00    	pushq  0x200b22(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4004e6:	ff 25 24 0b 20 00    	jmpq   *0x200b24(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4004ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004004f0 <printf@plt>:
  4004f0:	ff 25 22 0b 20 00    	jmpq   *0x200b22(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4004f6:	68 00 00 00 00       	pushq  $0x0
  4004fb:	e9 e0 ff ff ff       	jmpq   4004e0 <_init+0x20>

0000000000400500 <malloc@plt>:
  400500:	ff 25 1a 0b 20 00    	jmpq   *0x200b1a(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400506:	68 01 00 00 00       	pushq  $0x1
  40050b:	e9 d0 ff ff ff       	jmpq   4004e0 <_init+0x20>

0000000000400510 <__libc_start_main@plt>:
  400510:	ff 25 12 0b 20 00    	jmpq   *0x200b12(%rip)        # 601028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400516:	68 02 00 00 00       	pushq  $0x2
  40051b:	e9 c0 ff ff ff       	jmpq   4004e0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400520 <.plt.got>:
  400520:	ff 25 d2 0a 20 00    	jmpq   *0x200ad2(%rip)        # 600ff8 <_DYNAMIC+0x200>
  400526:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400530 <_start>:
  400530:	31 ed                	xor    %ebp,%ebp
  400532:	49 89 d1             	mov    %rdx,%r9
  400535:	5e                   	pop    %rsi
  400536:	48 89 e2             	mov    %rsp,%rdx
  400539:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40053d:	50                   	push   %rax
  40053e:	54                   	push   %rsp
  40053f:	49 c7 c0 30 08 40 00 	mov    $0x400830,%r8
  400546:	48 c7 c1 c0 07 40 00 	mov    $0x4007c0,%rcx
  40054d:	48 c7 c7 40 06 40 00 	mov    $0x400640,%rdi
  400554:	e8 b7 ff ff ff       	callq  400510 <__libc_start_main@plt>
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
  400600:	bf f0 0d 60 00       	mov    $0x600df0,%edi
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

0000000000400630 <_ZN9TestClass7storageEv>:
  400630:	b8 50 10 60 00       	mov    $0x601050,%eax
  400635:	c3                   	retq   
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <main>:
  400640:	53                   	push   %rbx
  400641:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400646:	e8 b5 fe ff ff       	callq  400500 <malloc@plt>
  40064b:	48 89 c3             	mov    %rax,%rbx
  40064e:	48 89 1d 13 0a 20 00 	mov    %rbx,0x200a13(%rip)        # 601068 <__TestClassdata_buffer+0x18>
  400655:	0f 57 c0             	xorps  %xmm0,%xmm0
  400658:	0f 11 05 f9 09 20 00 	movups %xmm0,0x2009f9(%rip)        # 601058 <__TestClassdata_buffer+0x8>
  40065f:	48 c7 05 e6 09 20 00 	movq   $0x1,0x2009e6(%rip)        # 601050 <__TestClassdata_buffer>
  400666:	01 00 00 00 
  40066a:	c7 43 04 00 00 00 00 	movl   $0x0,0x4(%rbx)
  400671:	48 83 c3 04          	add    $0x4,%rbx
  400675:	48 8b 05 ec 09 20 00 	mov    0x2009ec(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  40067c:	48 89 d9             	mov    %rbx,%rcx
  40067f:	48 29 c1             	sub    %rax,%rcx
  400682:	48 83 e1 fc          	and    $0xfffffffffffffffc,%rcx
  400686:	c7 04 01 77 77 00 00 	movl   $0x7777,(%rcx,%rax,1)
  40068d:	c7 84 01 d0 48 00 00 	movl   $0x8888,0x48d0(%rcx,%rax,1)
  400694:	88 88 00 00 
  400698:	69 14 01 55 55 00 00 	imul   $0x5555,(%rcx,%rax,1),%edx
  40069f:	89 14 01             	mov    %edx,(%rcx,%rax,1)
  4006a2:	69 94 01 d0 48 00 00 	imul   $0x4444,0x48d0(%rcx,%rax,1),%edx
  4006a9:	44 44 00 00 
  4006ad:	89 94 01 d0 48 00 00 	mov    %edx,0x48d0(%rcx,%rax,1)
  4006b4:	8b 34 01             	mov    (%rcx,%rax,1),%esi
  4006b7:	bf 44 08 40 00       	mov    $0x400844,%edi
  4006bc:	31 c0                	xor    %eax,%eax
  4006be:	e8 2d fe ff ff       	callq  4004f0 <printf@plt>
  4006c3:	48 8b 05 9e 09 20 00 	mov    0x20099e(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006ca:	48 29 c3             	sub    %rax,%rbx
  4006cd:	48 83 e3 fc          	and    $0xfffffffffffffffc,%rbx
  4006d1:	b1 01                	mov    $0x1,%cl
  4006d3:	81 3c 03 83 2d d2 27 	cmpl   $0x27d22d83,(%rbx,%rax,1)
  4006da:	75 0e                	jne    4006ea <main+0xaa>
  4006dc:	81 bc 18 d0 48 00 00 	cmpl   $0x24686420,0x48d0(%rax,%rbx,1)
  4006e3:	20 64 68 24 
  4006e7:	0f 95 c1             	setne  %cl
  4006ea:	0f b6 c1             	movzbl %cl,%eax
  4006ed:	5b                   	pop    %rbx
  4006ee:	c3                   	retq   
  4006ef:	90                   	nop

00000000004006f0 <_Z12new_instancev>:
  4006f0:	48 8b 0d 59 09 20 00 	mov    0x200959(%rip),%rcx        # 601050 <__TestClassdata_buffer>
  4006f7:	48 8d 41 01          	lea    0x1(%rcx),%rax
  4006fb:	48 89 05 4e 09 20 00 	mov    %rax,0x20094e(%rip)        # 601050 <__TestClassdata_buffer>
  400702:	48 8b 15 5f 09 20 00 	mov    0x20095f(%rip),%rdx        # 601068 <__TestClassdata_buffer+0x18>
  400709:	48 8d 44 8a 04       	lea    0x4(%rdx,%rcx,4),%rax
  40070e:	c7 44 8a 04 00 00 00 	movl   $0x0,0x4(%rdx,%rcx,4)
  400715:	00 
  400716:	c3                   	retq   
  400717:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40071e:	00 00 

0000000000400720 <_Z12write_field0P9TestClass>:
  400720:	48 8b 05 41 09 20 00 	mov    0x200941(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400727:	48 29 c7             	sub    %rax,%rdi
  40072a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40072e:	c7 04 07 77 77 00 00 	movl   $0x7777,(%rdi,%rax,1)
  400735:	c3                   	retq   
  400736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073d:	00 00 00 

0000000000400740 <_Z11read_field0P9TestClass>:
  400740:	48 8b 05 21 09 20 00 	mov    0x200921(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400747:	48 29 c7             	sub    %rax,%rdi
  40074a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40074e:	8b 04 07             	mov    (%rdi,%rax,1),%eax
  400751:	c3                   	retq   
  400752:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400759:	1f 84 00 00 00 00 00 

0000000000400760 <_Z12write_field1P9TestClass>:
  400760:	48 8b 05 01 09 20 00 	mov    0x200901(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400767:	48 29 c7             	sub    %rax,%rdi
  40076a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40076e:	c7 84 38 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,1)
  400775:	77 77 00 00 
  400779:	c3                   	retq   
  40077a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400780 <_Z11read_field1P9TestClass>:
  400780:	48 8b 05 e1 08 20 00 	mov    0x2008e1(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400787:	48 29 c7             	sub    %rax,%rdi
  40078a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40078e:	8b 84 38 d0 48 00 00 	mov    0x48d0(%rax,%rdi,1),%eax
  400795:	c3                   	retq   
  400796:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40079d:	00 00 00 

00000000004007a0 <_Z21explicit_write_field0m>:
  4007a0:	c7 04 bd 70 10 60 00 	movl   $0x7777,0x601070(,%rdi,4)
  4007a7:	77 77 00 00 
  4007ab:	c3                   	retq   
  4007ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004007b0 <_Z20explicit_read_field0m>:
  4007b0:	8b 04 bd 70 10 60 00 	mov    0x601070(,%rdi,4),%eax
  4007b7:	c3                   	retq   
  4007b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4007bf:	00 

00000000004007c0 <__libc_csu_init>:
  4007c0:	41 57                	push   %r15
  4007c2:	41 56                	push   %r14
  4007c4:	41 89 ff             	mov    %edi,%r15d
  4007c7:	41 55                	push   %r13
  4007c9:	41 54                	push   %r12
  4007cb:	4c 8d 25 0e 06 20 00 	lea    0x20060e(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  4007d2:	55                   	push   %rbp
  4007d3:	48 8d 2d 0e 06 20 00 	lea    0x20060e(%rip),%rbp        # 600de8 <__init_array_end>
  4007da:	53                   	push   %rbx
  4007db:	49 89 f6             	mov    %rsi,%r14
  4007de:	49 89 d5             	mov    %rdx,%r13
  4007e1:	4c 29 e5             	sub    %r12,%rbp
  4007e4:	48 83 ec 08          	sub    $0x8,%rsp
  4007e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4007ec:	e8 cf fc ff ff       	callq  4004c0 <_init>
  4007f1:	48 85 ed             	test   %rbp,%rbp
  4007f4:	74 20                	je     400816 <__libc_csu_init+0x56>
  4007f6:	31 db                	xor    %ebx,%ebx
  4007f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4007ff:	00 
  400800:	4c 89 ea             	mov    %r13,%rdx
  400803:	4c 89 f6             	mov    %r14,%rsi
  400806:	44 89 ff             	mov    %r15d,%edi
  400809:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40080d:	48 83 c3 01          	add    $0x1,%rbx
  400811:	48 39 eb             	cmp    %rbp,%rbx
  400814:	75 ea                	jne    400800 <__libc_csu_init+0x40>
  400816:	48 83 c4 08          	add    $0x8,%rsp
  40081a:	5b                   	pop    %rbx
  40081b:	5d                   	pop    %rbp
  40081c:	41 5c                	pop    %r12
  40081e:	41 5d                	pop    %r13
  400820:	41 5e                	pop    %r14
  400822:	41 5f                	pop    %r15
  400824:	c3                   	retq   
  400825:	90                   	nop
  400826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40082d:	00 00 00 

0000000000400830 <__libc_csu_fini>:
  400830:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400834 <_fini>:
  400834:	48 83 ec 08          	sub    $0x8,%rsp
  400838:	48 83 c4 08          	add    $0x8,%rsp
  40083c:	c3                   	retq   
