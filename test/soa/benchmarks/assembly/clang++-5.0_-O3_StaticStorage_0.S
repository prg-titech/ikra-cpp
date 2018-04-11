
bin/clang++-5.0_-O3_StaticStorage_0:     file format elf64-x86-64


Disassembly of section .init:

0000000000400468 <_init>:
  400468:	48 83 ec 08          	sub    $0x8,%rsp
  40046c:	48 8b 05 85 0b 20 00 	mov    0x200b85(%rip),%rax        # 600ff8 <_DYNAMIC+0x200>
  400473:	48 85 c0             	test   %rax,%rax
  400476:	74 05                	je     40047d <_init+0x15>
  400478:	e8 43 00 00 00       	callq  4004c0 <__libc_start_main@plt+0x10>
  40047d:	48 83 c4 08          	add    $0x8,%rsp
  400481:	c3                   	retq   

Disassembly of section .plt:

0000000000400490 <printf@plt-0x10>:
  400490:	ff 35 72 0b 20 00    	pushq  0x200b72(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400496:	ff 25 74 0b 20 00    	jmpq   *0x200b74(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40049c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004004a0 <printf@plt>:
  4004a0:	ff 25 72 0b 20 00    	jmpq   *0x200b72(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4004a6:	68 00 00 00 00       	pushq  $0x0
  4004ab:	e9 e0 ff ff ff       	jmpq   400490 <_init+0x28>

00000000004004b0 <__libc_start_main@plt>:
  4004b0:	ff 25 6a 0b 20 00    	jmpq   *0x200b6a(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4004b6:	68 01 00 00 00       	pushq  $0x1
  4004bb:	e9 d0 ff ff ff       	jmpq   400490 <_init+0x28>

Disassembly of section .plt.got:

00000000004004c0 <.plt.got>:
  4004c0:	ff 25 32 0b 20 00    	jmpq   *0x200b32(%rip)        # 600ff8 <_DYNAMIC+0x200>
  4004c6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004d0 <_start>:
  4004d0:	31 ed                	xor    %ebp,%ebp
  4004d2:	49 89 d1             	mov    %rdx,%r9
  4004d5:	5e                   	pop    %rsi
  4004d6:	48 89 e2             	mov    %rsp,%rdx
  4004d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004dd:	50                   	push   %rax
  4004de:	54                   	push   %rsp
  4004df:	49 c7 c0 30 07 40 00 	mov    $0x400730,%r8
  4004e6:	48 c7 c1 c0 06 40 00 	mov    $0x4006c0,%rcx
  4004ed:	48 c7 c7 d0 05 40 00 	mov    $0x4005d0,%rdi
  4004f4:	e8 b7 ff ff ff       	callq  4004b0 <__libc_start_main@plt>
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
  4005a0:	bf f0 0d 60 00       	mov    $0x600df0,%edi
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

00000000004005d0 <main>:
  4005d0:	55                   	push   %rbp
  4005d1:	53                   	push   %rbx
  4005d2:	50                   	push   %rax
  4005d3:	c7 05 7b 0a 20 00 00 	movl   $0x0,0x200a7b(%rip)        # 601058 <__TestClassdata_buffer+0x8>
  4005da:	00 00 00 
  4005dd:	48 c7 05 68 0a 20 00 	movq   $0x1,0x200a68(%rip)        # 601050 <__TestClassdata_buffer>
  4005e4:	01 00 00 00 
  4005e8:	c7 05 72 0a 20 00 83 	movl   $0x27d22d83,0x200a72(%rip)        # 601064 <__TestClassdata_buffer+0x14>
  4005ef:	2d d2 27 
  4005f2:	c7 05 48 53 20 00 20 	movl   $0x24686420,0x205348(%rip)        # 605944 <__TestClassdata_buffer+0x48f4>
  4005f9:	64 68 24 
  4005fc:	bb 83 2d d2 27       	mov    $0x27d22d83,%ebx
  400601:	bd 20 64 68 24       	mov    $0x24686420,%ebp
  400606:	bf 44 07 40 00       	mov    $0x400744,%edi
  40060b:	be 83 2d d2 27       	mov    $0x27d22d83,%esi
  400610:	ba 20 64 68 24       	mov    $0x24686420,%edx
  400615:	31 c0                	xor    %eax,%eax
  400617:	e8 84 fe ff ff       	callq  4004a0 <printf@plt>
  40061c:	33 1d 42 0a 20 00    	xor    0x200a42(%rip),%ebx        # 601064 <__TestClassdata_buffer+0x14>
  400622:	33 2d 1c 53 20 00    	xor    0x20531c(%rip),%ebp        # 605944 <__TestClassdata_buffer+0x48f4>
  400628:	31 c0                	xor    %eax,%eax
  40062a:	09 dd                	or     %ebx,%ebp
  40062c:	0f 95 c0             	setne  %al
  40062f:	48 83 c4 08          	add    $0x8,%rsp
  400633:	5b                   	pop    %rbx
  400634:	5d                   	pop    %rbp
  400635:	c3                   	retq   
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <_Z12new_instancev>:
  400640:	48 8b 05 09 0a 20 00 	mov    0x200a09(%rip),%rax        # 601050 <__TestClassdata_buffer>
  400647:	48 83 c0 01          	add    $0x1,%rax
  40064b:	48 89 05 fe 09 20 00 	mov    %rax,0x2009fe(%rip)        # 601050 <__TestClassdata_buffer>
  400652:	c3                   	retq   
  400653:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40065a:	84 00 00 00 00 00 

0000000000400660 <_Z12write_field0P9TestClass>:
  400660:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  400667:	77 77 00 00 
  40066b:	c3                   	retq   
  40066c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400670 <_Z11read_field0P9TestClass>:
  400670:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  400677:	c3                   	retq   
  400678:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40067f:	00 

0000000000400680 <_Z12write_field1P9TestClass>:
  400680:	c7 04 bd 40 59 60 00 	movl   $0x7777,0x605940(,%rdi,4)
  400687:	77 77 00 00 
  40068b:	c3                   	retq   
  40068c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400690 <_Z11read_field1P9TestClass>:
  400690:	8b 04 bd 40 59 60 00 	mov    0x605940(,%rdi,4),%eax
  400697:	c3                   	retq   
  400698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40069f:	00 

00000000004006a0 <_Z21explicit_write_field0m>:
  4006a0:	c7 04 bd 20 a2 60 00 	movl   $0x7777,0x60a220(,%rdi,4)
  4006a7:	77 77 00 00 
  4006ab:	c3                   	retq   
  4006ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006b0 <_Z20explicit_read_field0m>:
  4006b0:	8b 04 bd 20 a2 60 00 	mov    0x60a220(,%rdi,4),%eax
  4006b7:	c3                   	retq   
  4006b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006bf:	00 

00000000004006c0 <__libc_csu_init>:
  4006c0:	41 57                	push   %r15
  4006c2:	41 56                	push   %r14
  4006c4:	41 89 ff             	mov    %edi,%r15d
  4006c7:	41 55                	push   %r13
  4006c9:	41 54                	push   %r12
  4006cb:	4c 8d 25 0e 07 20 00 	lea    0x20070e(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  4006d2:	55                   	push   %rbp
  4006d3:	48 8d 2d 0e 07 20 00 	lea    0x20070e(%rip),%rbp        # 600de8 <__init_array_end>
  4006da:	53                   	push   %rbx
  4006db:	49 89 f6             	mov    %rsi,%r14
  4006de:	49 89 d5             	mov    %rdx,%r13
  4006e1:	4c 29 e5             	sub    %r12,%rbp
  4006e4:	48 83 ec 08          	sub    $0x8,%rsp
  4006e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006ec:	e8 77 fd ff ff       	callq  400468 <_init>
  4006f1:	48 85 ed             	test   %rbp,%rbp
  4006f4:	74 20                	je     400716 <__libc_csu_init+0x56>
  4006f6:	31 db                	xor    %ebx,%ebx
  4006f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006ff:	00 
  400700:	4c 89 ea             	mov    %r13,%rdx
  400703:	4c 89 f6             	mov    %r14,%rsi
  400706:	44 89 ff             	mov    %r15d,%edi
  400709:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40070d:	48 83 c3 01          	add    $0x1,%rbx
  400711:	48 39 eb             	cmp    %rbp,%rbx
  400714:	75 ea                	jne    400700 <__libc_csu_init+0x40>
  400716:	48 83 c4 08          	add    $0x8,%rsp
  40071a:	5b                   	pop    %rbx
  40071b:	5d                   	pop    %rbp
  40071c:	41 5c                	pop    %r12
  40071e:	41 5d                	pop    %r13
  400720:	41 5e                	pop    %r14
  400722:	41 5f                	pop    %r15
  400724:	c3                   	retq   
  400725:	90                   	nop
  400726:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40072d:	00 00 00 

0000000000400730 <__libc_csu_fini>:
  400730:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400734 <_fini>:
  400734:	48 83 ec 08          	sub    $0x8,%rsp
  400738:	48 83 c4 08          	add    $0x8,%rsp
  40073c:	c3                   	retq   
