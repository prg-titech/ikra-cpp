
bin/g++_-O3_DynamicStorage_0:     file format elf64-x86-64


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
  4004a0:	48 83 ec 08          	sub    $0x8,%rsp
  4004a4:	bf a0 91 00 00       	mov    $0x91a0,%edi
  4004a9:	e8 c2 ff ff ff       	callq  400470 <malloc@plt>
  4004ae:	48 c7 05 4f 0d 20 00 	movq   $0x0,0x200d4f(%rip)        # 601208 <__TestClassdata_buffer+0x8>
  4004b5:	00 00 00 00 
  4004b9:	48 89 05 58 0d 20 00 	mov    %rax,0x200d58(%rip)        # 601218 <__TestClassdata_buffer+0x18>
  4004c0:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  4004c5:	48 c7 05 40 0d 20 00 	movq   $0x0,0x200d40(%rip)        # 601210 <__TestClassdata_buffer+0x10>
  4004cc:	00 00 00 00 
  4004d0:	48 c7 05 25 0d 20 00 	movq   $0x1,0x200d25(%rip)        # 601200 <__TestClassdata_buffer>
  4004d7:	01 00 00 00 
  4004db:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  4004e0:	c7 40 04 83 2d d2 27 	movl   $0x27d22d83,0x4(%rax)
  4004e7:	c7 80 d4 48 00 00 20 	movl   $0x24686420,0x48d4(%rax)
  4004ee:	64 68 24 
  4004f1:	be 54 07 40 00       	mov    $0x400754,%esi
  4004f6:	31 c0                	xor    %eax,%eax
  4004f8:	bf 01 00 00 00       	mov    $0x1,%edi
  4004fd:	e8 7e ff ff ff       	callq  400480 <__printf_chk@plt>
  400502:	48 8b 15 0f 0d 20 00 	mov    0x200d0f(%rip),%rdx        # 601218 <__TestClassdata_buffer+0x18>
  400509:	b8 01 00 00 00       	mov    $0x1,%eax
  40050e:	81 7a 04 83 2d d2 27 	cmpl   $0x27d22d83,0x4(%rdx)
  400515:	75 0d                	jne    400524 <main+0x84>
  400517:	81 ba d4 48 00 00 20 	cmpl   $0x24686420,0x48d4(%rdx)
  40051e:	64 68 24 
  400521:	0f 95 c0             	setne  %al
  400524:	0f b6 c0             	movzbl %al,%eax
  400527:	48 83 c4 08          	add    $0x8,%rsp
  40052b:	c3                   	retq   
  40052c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400530 <_start>:
  400530:	31 ed                	xor    %ebp,%ebp
  400532:	49 89 d1             	mov    %rdx,%r9
  400535:	5e                   	pop    %rsi
  400536:	48 89 e2             	mov    %rsp,%rdx
  400539:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40053d:	50                   	push   %rax
  40053e:	54                   	push   %rsp
  40053f:	49 c7 c0 40 07 40 00 	mov    $0x400740,%r8
  400546:	48 c7 c1 d0 06 40 00 	mov    $0x4006d0,%rcx
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

0000000000400630 <_ZN9TestClass7storageEv>:
  400630:	b8 00 12 60 00       	mov    $0x601200,%eax
  400635:	c3                   	retq   
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <_Z12new_instancev>:
  400640:	48 8b 05 b9 0b 20 00 	mov    0x200bb9(%rip),%rax        # 601200 <__TestClassdata_buffer>
  400647:	48 83 c0 01          	add    $0x1,%rax
  40064b:	48 89 05 ae 0b 20 00 	mov    %rax,0x200bae(%rip)        # 601200 <__TestClassdata_buffer>
  400652:	c3                   	retq   
  400653:	0f 1f 00             	nopl   (%rax)
  400656:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40065d:	00 00 00 

0000000000400660 <_Z12write_field0P9TestClass>:
  400660:	48 8b 05 b1 0b 20 00 	mov    0x200bb1(%rip),%rax        # 601218 <__TestClassdata_buffer+0x18>
  400667:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40066e:	c3                   	retq   
  40066f:	90                   	nop

0000000000400670 <_Z11read_field0P9TestClass>:
  400670:	48 8b 05 a1 0b 20 00 	mov    0x200ba1(%rip),%rax        # 601218 <__TestClassdata_buffer+0x18>
  400677:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40067a:	c3                   	retq   
  40067b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400680 <_Z12write_field1P9TestClass>:
  400680:	48 8b 05 91 0b 20 00 	mov    0x200b91(%rip),%rax        # 601218 <__TestClassdata_buffer+0x18>
  400687:	c7 84 b8 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,4)
  40068e:	77 77 00 00 
  400692:	c3                   	retq   
  400693:	0f 1f 00             	nopl   (%rax)
  400696:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40069d:	00 00 00 

00000000004006a0 <_Z11read_field1P9TestClass>:
  4006a0:	48 8b 05 71 0b 20 00 	mov    0x200b71(%rip),%rax        # 601218 <__TestClassdata_buffer+0x18>
  4006a7:	8b 84 b8 d0 48 00 00 	mov    0x48d0(%rax,%rdi,4),%eax
  4006ae:	c3                   	retq   
  4006af:	90                   	nop

00000000004006b0 <_Z21explicit_write_field0m>:
  4006b0:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  4006b7:	77 77 00 00 
  4006bb:	c3                   	retq   
  4006bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006c0 <_Z20explicit_read_field0m>:
  4006c0:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  4006c7:	c3                   	retq   
  4006c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006cf:	00 

00000000004006d0 <__libc_csu_init>:
  4006d0:	41 57                	push   %r15
  4006d2:	41 56                	push   %r14
  4006d4:	41 89 ff             	mov    %edi,%r15d
  4006d7:	41 55                	push   %r13
  4006d9:	41 54                	push   %r12
  4006db:	4c 8d 25 2e 07 20 00 	lea    0x20072e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  4006e2:	55                   	push   %rbp
  4006e3:	48 8d 2d 2e 07 20 00 	lea    0x20072e(%rip),%rbp        # 600e18 <__init_array_end>
  4006ea:	53                   	push   %rbx
  4006eb:	49 89 f6             	mov    %rsi,%r14
  4006ee:	49 89 d5             	mov    %rdx,%r13
  4006f1:	4c 29 e5             	sub    %r12,%rbp
  4006f4:	48 83 ec 08          	sub    $0x8,%rsp
  4006f8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006fc:	e8 27 fd ff ff       	callq  400428 <_init>
  400701:	48 85 ed             	test   %rbp,%rbp
  400704:	74 20                	je     400726 <__libc_csu_init+0x56>
  400706:	31 db                	xor    %ebx,%ebx
  400708:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40070f:	00 
  400710:	4c 89 ea             	mov    %r13,%rdx
  400713:	4c 89 f6             	mov    %r14,%rsi
  400716:	44 89 ff             	mov    %r15d,%edi
  400719:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40071d:	48 83 c3 01          	add    $0x1,%rbx
  400721:	48 39 eb             	cmp    %rbp,%rbx
  400724:	75 ea                	jne    400710 <__libc_csu_init+0x40>
  400726:	48 83 c4 08          	add    $0x8,%rsp
  40072a:	5b                   	pop    %rbx
  40072b:	5d                   	pop    %rbp
  40072c:	41 5c                	pop    %r12
  40072e:	41 5d                	pop    %r13
  400730:	41 5e                	pop    %r14
  400732:	41 5f                	pop    %r15
  400734:	c3                   	retq   
  400735:	90                   	nop
  400736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073d:	00 00 00 

0000000000400740 <__libc_csu_fini>:
  400740:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400744 <_fini>:
  400744:	48 83 ec 08          	sub    $0x8,%rsp
  400748:	48 83 c4 08          	add    $0x8,%rsp
  40074c:	c3                   	retq   
