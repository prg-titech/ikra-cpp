
bin/clang++-5.0_-O3_StaticStorage_4:     file format elf64-x86-64


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
  4004df:	49 c7 c0 c0 07 40 00 	mov    $0x4007c0,%r8
  4004e6:	48 c7 c1 50 07 40 00 	mov    $0x400750,%rcx
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
  4005d0:	53                   	push   %rbx
  4005d1:	c7 05 7d 0a 20 00 00 	movl   $0x0,0x200a7d(%rip)        # 601058 <__TestClassdata_buffer+0x8>
  4005d8:	00 00 00 
  4005db:	48 c7 05 6a 0a 20 00 	movq   $0x1,0x200a6a(%rip)        # 601050 <__TestClassdata_buffer>
  4005e2:	01 00 00 00 
  4005e6:	c7 05 74 0a 20 00 00 	movl   $0x0,0x200a74(%rip)        # 601064 <__TestClassdata_buffer+0x14>
  4005ed:	00 00 00 
  4005f0:	b8 60 10 60 00       	mov    $0x601060,%eax
  4005f5:	48 f7 d8             	neg    %rax
  4005f8:	48 8d 1c 05 64 10 60 	lea    0x601064(,%rax,1),%rbx
  4005ff:	00 
  400600:	48 83 e3 fc          	and    $0xfffffffffffffffc,%rbx
  400604:	c7 83 60 10 60 00 77 	movl   $0x7777,0x601060(%rbx)
  40060b:	77 00 00 
  40060e:	c7 83 40 59 60 00 88 	movl   $0x8888,0x605940(%rbx)
  400615:	88 00 00 
  400618:	69 83 60 10 60 00 55 	imul   $0x5555,0x601060(%rbx),%eax
  40061f:	55 00 00 
  400622:	89 83 60 10 60 00    	mov    %eax,0x601060(%rbx)
  400628:	69 93 40 59 60 00 44 	imul   $0x4444,0x605940(%rbx),%edx
  40062f:	44 00 00 
  400632:	89 93 40 59 60 00    	mov    %edx,0x605940(%rbx)
  400638:	8b b3 60 10 60 00    	mov    0x601060(%rbx),%esi
  40063e:	bf d4 07 40 00       	mov    $0x4007d4,%edi
  400643:	31 c0                	xor    %eax,%eax
  400645:	e8 56 fe ff ff       	callq  4004a0 <printf@plt>
  40064a:	b8 01 00 00 00       	mov    $0x1,%eax
  40064f:	81 bb 60 10 60 00 83 	cmpl   $0x27d22d83,0x601060(%rbx)
  400656:	2d d2 27 
  400659:	75 23                	jne    40067e <main+0xae>
  40065b:	b8 60 10 60 00       	mov    $0x601060,%eax
  400660:	48 f7 d8             	neg    %rax
  400663:	48 8d 0c 05 64 10 60 	lea    0x601064(,%rax,1),%rcx
  40066a:	00 
  40066b:	48 83 e1 fc          	and    $0xfffffffffffffffc,%rcx
  40066f:	31 c0                	xor    %eax,%eax
  400671:	81 b9 40 59 60 00 20 	cmpl   $0x24686420,0x605940(%rcx)
  400678:	64 68 24 
  40067b:	0f 95 c0             	setne  %al
  40067e:	5b                   	pop    %rbx
  40067f:	c3                   	retq   

0000000000400680 <_Z12new_instancev>:
  400680:	48 8b 0d c9 09 20 00 	mov    0x2009c9(%rip),%rcx        # 601050 <__TestClassdata_buffer>
  400687:	48 8d 41 01          	lea    0x1(%rcx),%rax
  40068b:	48 89 05 be 09 20 00 	mov    %rax,0x2009be(%rip)        # 601050 <__TestClassdata_buffer>
  400692:	48 8d 04 8d 64 10 60 	lea    0x601064(,%rcx,4),%rax
  400699:	00 
  40069a:	c7 04 8d 64 10 60 00 	movl   $0x0,0x601064(,%rcx,4)
  4006a1:	00 00 00 00 
  4006a5:	c3                   	retq   
  4006a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ad:	00 00 00 

00000000004006b0 <_Z12write_field0P9TestClass>:
  4006b0:	b8 60 10 60 00       	mov    $0x601060,%eax
  4006b5:	48 29 c7             	sub    %rax,%rdi
  4006b8:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  4006bc:	c7 87 60 10 60 00 77 	movl   $0x7777,0x601060(%rdi)
  4006c3:	77 00 00 
  4006c6:	c3                   	retq   
  4006c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006ce:	00 00 

00000000004006d0 <_Z11read_field0P9TestClass>:
  4006d0:	b8 60 10 60 00       	mov    $0x601060,%eax
  4006d5:	48 29 c7             	sub    %rax,%rdi
  4006d8:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  4006dc:	8b 87 60 10 60 00    	mov    0x601060(%rdi),%eax
  4006e2:	c3                   	retq   
  4006e3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006ea:	84 00 00 00 00 00 

00000000004006f0 <_Z12write_field1P9TestClass>:
  4006f0:	b8 60 10 60 00       	mov    $0x601060,%eax
  4006f5:	48 29 c7             	sub    %rax,%rdi
  4006f8:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  4006fc:	c7 87 40 59 60 00 77 	movl   $0x7777,0x605940(%rdi)
  400703:	77 00 00 
  400706:	c3                   	retq   
  400707:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40070e:	00 00 

0000000000400710 <_Z11read_field1P9TestClass>:
  400710:	b8 60 10 60 00       	mov    $0x601060,%eax
  400715:	48 29 c7             	sub    %rax,%rdi
  400718:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40071c:	8b 87 40 59 60 00    	mov    0x605940(%rdi),%eax
  400722:	c3                   	retq   
  400723:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40072a:	84 00 00 00 00 00 

0000000000400730 <_Z21explicit_write_field0m>:
  400730:	c7 04 bd 20 a2 60 00 	movl   $0x7777,0x60a220(,%rdi,4)
  400737:	77 77 00 00 
  40073b:	c3                   	retq   
  40073c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400740 <_Z20explicit_read_field0m>:
  400740:	8b 04 bd 20 a2 60 00 	mov    0x60a220(,%rdi,4),%eax
  400747:	c3                   	retq   
  400748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40074f:	00 

0000000000400750 <__libc_csu_init>:
  400750:	41 57                	push   %r15
  400752:	41 56                	push   %r14
  400754:	41 89 ff             	mov    %edi,%r15d
  400757:	41 55                	push   %r13
  400759:	41 54                	push   %r12
  40075b:	4c 8d 25 7e 06 20 00 	lea    0x20067e(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  400762:	55                   	push   %rbp
  400763:	48 8d 2d 7e 06 20 00 	lea    0x20067e(%rip),%rbp        # 600de8 <__init_array_end>
  40076a:	53                   	push   %rbx
  40076b:	49 89 f6             	mov    %rsi,%r14
  40076e:	49 89 d5             	mov    %rdx,%r13
  400771:	4c 29 e5             	sub    %r12,%rbp
  400774:	48 83 ec 08          	sub    $0x8,%rsp
  400778:	48 c1 fd 03          	sar    $0x3,%rbp
  40077c:	e8 e7 fc ff ff       	callq  400468 <_init>
  400781:	48 85 ed             	test   %rbp,%rbp
  400784:	74 20                	je     4007a6 <__libc_csu_init+0x56>
  400786:	31 db                	xor    %ebx,%ebx
  400788:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40078f:	00 
  400790:	4c 89 ea             	mov    %r13,%rdx
  400793:	4c 89 f6             	mov    %r14,%rsi
  400796:	44 89 ff             	mov    %r15d,%edi
  400799:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40079d:	48 83 c3 01          	add    $0x1,%rbx
  4007a1:	48 39 eb             	cmp    %rbp,%rbx
  4007a4:	75 ea                	jne    400790 <__libc_csu_init+0x40>
  4007a6:	48 83 c4 08          	add    $0x8,%rsp
  4007aa:	5b                   	pop    %rbx
  4007ab:	5d                   	pop    %rbp
  4007ac:	41 5c                	pop    %r12
  4007ae:	41 5d                	pop    %r13
  4007b0:	41 5e                	pop    %r14
  4007b2:	41 5f                	pop    %r15
  4007b4:	c3                   	retq   
  4007b5:	90                   	nop
  4007b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007bd:	00 00 00 

00000000004007c0 <__libc_csu_fini>:
  4007c0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004007c4 <_fini>:
  4007c4:	48 83 ec 08          	sub    $0x8,%rsp
  4007c8:	48 83 c4 08          	add    $0x8,%rsp
  4007cc:	c3                   	retq   
