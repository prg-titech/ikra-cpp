
bin/clang++-5.0_-O3_StaticStorage_0_Aos:     file format elf64-x86-64


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
  4004df:	49 c7 c0 70 07 40 00 	mov    $0x400770,%r8
  4004e6:	48 c7 c1 00 07 40 00 	mov    $0x400700,%rcx
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
  4005e8:	48 b8 83 2d d2 27 20 	movabs $0x2468642027d22d83,%rax
  4005ef:	64 68 24 
  4005f2:	48 89 05 6f 0a 20 00 	mov    %rax,0x200a6f(%rip)        # 601068 <__TestClassdata_buffer+0x18>
  4005f9:	bb 83 2d d2 27       	mov    $0x27d22d83,%ebx
  4005fe:	bd 20 64 68 24       	mov    $0x24686420,%ebp
  400603:	bf 84 07 40 00       	mov    $0x400784,%edi
  400608:	be 83 2d d2 27       	mov    $0x27d22d83,%esi
  40060d:	ba 20 64 68 24       	mov    $0x24686420,%edx
  400612:	31 c0                	xor    %eax,%eax
  400614:	e8 87 fe ff ff       	callq  4004a0 <printf@plt>
  400619:	33 1d 49 0a 20 00    	xor    0x200a49(%rip),%ebx        # 601068 <__TestClassdata_buffer+0x18>
  40061f:	33 2d 47 0a 20 00    	xor    0x200a47(%rip),%ebp        # 60106c <__TestClassdata_buffer+0x1c>
  400625:	31 c0                	xor    %eax,%eax
  400627:	09 dd                	or     %ebx,%ebp
  400629:	0f 95 c0             	setne  %al
  40062c:	48 83 c4 08          	add    $0x8,%rsp
  400630:	5b                   	pop    %rbx
  400631:	5d                   	pop    %rbp
  400632:	c3                   	retq   
  400633:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40063a:	84 00 00 00 00 00 

0000000000400640 <_Z12new_instancev>:
  400640:	48 8b 05 09 0a 20 00 	mov    0x200a09(%rip),%rax        # 601050 <__TestClassdata_buffer>
  400647:	48 8d 48 01          	lea    0x1(%rax),%rcx
  40064b:	48 89 0d fe 09 20 00 	mov    %rcx,0x2009fe(%rip)        # 601050 <__TestClassdata_buffer>
  400652:	48 8d 04 c5 68 10 60 	lea    0x601068(,%rax,8),%rax
  400659:	00 
  40065a:	c3                   	retq   
  40065b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400660 <_Z12write_field0P9TestClass>:
  400660:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400666:	c3                   	retq   
  400667:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40066e:	00 00 

0000000000400670 <_Z11read_field0P9TestClass>:
  400670:	8b 07                	mov    (%rdi),%eax
  400672:	c3                   	retq   
  400673:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40067a:	84 00 00 00 00 00 

0000000000400680 <_Z12write_field1P9TestClass>:
  400680:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400687:	c3                   	retq   
  400688:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40068f:	00 

0000000000400690 <_Z11read_field1P9TestClass>:
  400690:	8b 47 04             	mov    0x4(%rdi),%eax
  400693:	c3                   	retq   
  400694:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40069b:	00 00 00 00 00 

00000000004006a0 <_Z21explicit_write_field0m>:
  4006a0:	c7 04 bd c0 33 61 00 	movl   $0x7777,0x6133c0(,%rdi,4)
  4006a7:	77 77 00 00 
  4006ab:	c3                   	retq   
  4006ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006b0 <_Z20explicit_read_field0m>:
  4006b0:	8b 04 bd c0 33 61 00 	mov    0x6133c0(,%rdi,4),%eax
  4006b7:	c3                   	retq   
  4006b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006bf:	00 

00000000004006c0 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  4006c0:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  4006c6:	c3                   	retq   
  4006c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006ce:	00 00 

00000000004006d0 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  4006d0:	8b 07                	mov    (%rdi),%eax
  4006d2:	c3                   	retq   
  4006d3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006da:	84 00 00 00 00 00 

00000000004006e0 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  4006e0:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  4006e7:	c3                   	retq   
  4006e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006ef:	00 

00000000004006f0 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  4006f0:	8b 47 04             	mov    0x4(%rdi),%eax
  4006f3:	c3                   	retq   
  4006f4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006fb:	00 00 00 
  4006fe:	66 90                	xchg   %ax,%ax

0000000000400700 <__libc_csu_init>:
  400700:	41 57                	push   %r15
  400702:	41 56                	push   %r14
  400704:	41 89 ff             	mov    %edi,%r15d
  400707:	41 55                	push   %r13
  400709:	41 54                	push   %r12
  40070b:	4c 8d 25 ce 06 20 00 	lea    0x2006ce(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  400712:	55                   	push   %rbp
  400713:	48 8d 2d ce 06 20 00 	lea    0x2006ce(%rip),%rbp        # 600de8 <__init_array_end>
  40071a:	53                   	push   %rbx
  40071b:	49 89 f6             	mov    %rsi,%r14
  40071e:	49 89 d5             	mov    %rdx,%r13
  400721:	4c 29 e5             	sub    %r12,%rbp
  400724:	48 83 ec 08          	sub    $0x8,%rsp
  400728:	48 c1 fd 03          	sar    $0x3,%rbp
  40072c:	e8 37 fd ff ff       	callq  400468 <_init>
  400731:	48 85 ed             	test   %rbp,%rbp
  400734:	74 20                	je     400756 <__libc_csu_init+0x56>
  400736:	31 db                	xor    %ebx,%ebx
  400738:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40073f:	00 
  400740:	4c 89 ea             	mov    %r13,%rdx
  400743:	4c 89 f6             	mov    %r14,%rsi
  400746:	44 89 ff             	mov    %r15d,%edi
  400749:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40074d:	48 83 c3 01          	add    $0x1,%rbx
  400751:	48 39 eb             	cmp    %rbp,%rbx
  400754:	75 ea                	jne    400740 <__libc_csu_init+0x40>
  400756:	48 83 c4 08          	add    $0x8,%rsp
  40075a:	5b                   	pop    %rbx
  40075b:	5d                   	pop    %rbp
  40075c:	41 5c                	pop    %r12
  40075e:	41 5d                	pop    %r13
  400760:	41 5e                	pop    %r14
  400762:	41 5f                	pop    %r15
  400764:	c3                   	retq   
  400765:	90                   	nop
  400766:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40076d:	00 00 00 

0000000000400770 <__libc_csu_fini>:
  400770:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400774 <_fini>:
  400774:	48 83 ec 08          	sub    $0x8,%rsp
  400778:	48 83 c4 08          	add    $0x8,%rsp
  40077c:	c3                   	retq   
