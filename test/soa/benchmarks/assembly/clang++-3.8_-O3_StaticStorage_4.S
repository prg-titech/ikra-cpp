
bin/clang++-3.8_-O3_StaticStorage_4:     file format elf64-x86-64


Disassembly of section .init:

0000000000400488 <_init>:
  400488:	48 83 ec 08          	sub    $0x8,%rsp
  40048c:	48 8b 05 65 0b 20 00 	mov    0x200b65(%rip),%rax        # 600ff8 <_DYNAMIC+0x200>
  400493:	48 85 c0             	test   %rax,%rax
  400496:	74 05                	je     40049d <_init+0x15>
  400498:	e8 43 00 00 00       	callq  4004e0 <__libc_start_main@plt+0x10>
  40049d:	48 83 c4 08          	add    $0x8,%rsp
  4004a1:	c3                   	retq   

Disassembly of section .plt:

00000000004004b0 <printf@plt-0x10>:
  4004b0:	ff 35 52 0b 20 00    	pushq  0x200b52(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4004b6:	ff 25 54 0b 20 00    	jmpq   *0x200b54(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4004bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004004c0 <printf@plt>:
  4004c0:	ff 25 52 0b 20 00    	jmpq   *0x200b52(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4004c6:	68 00 00 00 00       	pushq  $0x0
  4004cb:	e9 e0 ff ff ff       	jmpq   4004b0 <_init+0x28>

00000000004004d0 <__libc_start_main@plt>:
  4004d0:	ff 25 4a 0b 20 00    	jmpq   *0x200b4a(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4004d6:	68 01 00 00 00       	pushq  $0x1
  4004db:	e9 d0 ff ff ff       	jmpq   4004b0 <_init+0x28>

Disassembly of section .plt.got:

00000000004004e0 <.plt.got>:
  4004e0:	ff 25 12 0b 20 00    	jmpq   *0x200b12(%rip)        # 600ff8 <_DYNAMIC+0x200>
  4004e6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004f0 <_start>:
  4004f0:	31 ed                	xor    %ebp,%ebp
  4004f2:	49 89 d1             	mov    %rdx,%r9
  4004f5:	5e                   	pop    %rsi
  4004f6:	48 89 e2             	mov    %rsp,%rdx
  4004f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004fd:	50                   	push   %rax
  4004fe:	54                   	push   %rsp
  4004ff:	49 c7 c0 d0 07 40 00 	mov    $0x4007d0,%r8
  400506:	48 c7 c1 60 07 40 00 	mov    $0x400760,%rcx
  40050d:	48 c7 c7 f0 05 40 00 	mov    $0x4005f0,%rdi
  400514:	e8 b7 ff ff ff       	callq  4004d0 <__libc_start_main@plt>
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
  4005c0:	bf f0 0d 60 00       	mov    $0x600df0,%edi
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

00000000004005f0 <main>:
  4005f0:	53                   	push   %rbx
  4005f1:	48 c7 05 fc 9b 20 00 	movq   $0x0,0x209bfc(%rip)        # 60a1f8 <storage_buffer+0x91a8>
  4005f8:	00 00 00 00 
  4005fc:	48 8b 1d 4d 0a 20 00 	mov    0x200a4d(%rip),%rbx        # 601050 <storage_buffer>
  400603:	48 8d 43 01          	lea    0x1(%rbx),%rax
  400607:	48 89 05 42 0a 20 00 	mov    %rax,0x200a42(%rip)        # 601050 <storage_buffer>
  40060e:	c7 04 9d 58 10 60 00 	movl   $0x0,0x601058(,%rbx,4)
  400615:	00 00 00 00 
  400619:	c7 04 9d 58 10 60 00 	movl   $0x7777,0x601058(,%rbx,4)
  400620:	77 77 00 00 
  400624:	c7 04 9d 28 59 60 00 	movl   $0x8888,0x605928(,%rbx,4)
  40062b:	88 88 00 00 
  40062f:	69 04 9d 58 10 60 00 	imul   $0x5555,0x601058(,%rbx,4),%eax
  400636:	55 55 00 00 
  40063a:	89 04 9d 58 10 60 00 	mov    %eax,0x601058(,%rbx,4)
  400641:	69 14 9d 28 59 60 00 	imul   $0x4444,0x605928(,%rbx,4),%edx
  400648:	44 44 00 00 
  40064c:	89 14 9d 28 59 60 00 	mov    %edx,0x605928(,%rbx,4)
  400653:	8b 34 9d 58 10 60 00 	mov    0x601058(,%rbx,4),%esi
  40065a:	bf e4 07 40 00       	mov    $0x4007e4,%edi
  40065f:	31 c0                	xor    %eax,%eax
  400661:	e8 5a fe ff ff       	callq  4004c0 <printf@plt>
  400666:	b0 01                	mov    $0x1,%al
  400668:	81 3c 9d 58 10 60 00 	cmpl   $0x27d22d83,0x601058(,%rbx,4)
  40066f:	83 2d d2 27 
  400673:	75 11                	jne    400686 <main+0x96>
  400675:	48 c1 e3 02          	shl    $0x2,%rbx
  400679:	81 bb 28 59 60 00 20 	cmpl   $0x24686420,0x605928(%rbx)
  400680:	64 68 24 
  400683:	0f 95 c0             	setne  %al
  400686:	0f b6 c0             	movzbl %al,%eax
  400689:	5b                   	pop    %rbx
  40068a:	c3                   	retq   
  40068b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400690 <_Z12new_instancev>:
  400690:	48 8b 0d b9 09 20 00 	mov    0x2009b9(%rip),%rcx        # 601050 <storage_buffer>
  400697:	48 8d 41 01          	lea    0x1(%rcx),%rax
  40069b:	48 89 05 ae 09 20 00 	mov    %rax,0x2009ae(%rip)        # 601050 <storage_buffer>
  4006a2:	48 8d 04 8d 58 10 60 	lea    0x601058(,%rcx,4),%rax
  4006a9:	00 
  4006aa:	c7 04 8d 58 10 60 00 	movl   $0x0,0x601058(,%rcx,4)
  4006b1:	00 00 00 00 
  4006b5:	c3                   	retq   
  4006b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006bd:	00 00 00 

00000000004006c0 <_Z12write_field0P9TestClass>:
  4006c0:	b8 58 10 60 00       	mov    $0x601058,%eax
  4006c5:	48 29 c7             	sub    %rax,%rdi
  4006c8:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  4006cc:	c7 87 58 10 60 00 77 	movl   $0x7777,0x601058(%rdi)
  4006d3:	77 00 00 
  4006d6:	c3                   	retq   
  4006d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006de:	00 00 

00000000004006e0 <_Z11read_field0P9TestClass>:
  4006e0:	b8 58 10 60 00       	mov    $0x601058,%eax
  4006e5:	48 29 c7             	sub    %rax,%rdi
  4006e8:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  4006ec:	8b 87 58 10 60 00    	mov    0x601058(%rdi),%eax
  4006f2:	c3                   	retq   
  4006f3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006fa:	84 00 00 00 00 00 

0000000000400700 <_Z12write_field1P9TestClass>:
  400700:	b8 58 10 60 00       	mov    $0x601058,%eax
  400705:	48 29 c7             	sub    %rax,%rdi
  400708:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40070c:	c7 87 28 59 60 00 77 	movl   $0x7777,0x605928(%rdi)
  400713:	77 00 00 
  400716:	c3                   	retq   
  400717:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40071e:	00 00 

0000000000400720 <_Z11read_field1P9TestClass>:
  400720:	b8 58 10 60 00       	mov    $0x601058,%eax
  400725:	48 29 c7             	sub    %rax,%rdi
  400728:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40072c:	8b 87 28 59 60 00    	mov    0x605928(%rdi),%eax
  400732:	c3                   	retq   
  400733:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40073a:	84 00 00 00 00 00 

0000000000400740 <_Z21explicit_write_field0m>:
  400740:	48 8b 05 a9 8f 21 00 	mov    0x218fa9(%rip),%rax        # 6196f0 <explicit_field0>
  400747:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40074e:	c3                   	retq   
  40074f:	90                   	nop

0000000000400750 <_Z20explicit_read_field0m>:
  400750:	48 8b 05 99 8f 21 00 	mov    0x218f99(%rip),%rax        # 6196f0 <explicit_field0>
  400757:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40075a:	c3                   	retq   
  40075b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400760 <__libc_csu_init>:
  400760:	41 57                	push   %r15
  400762:	41 56                	push   %r14
  400764:	41 89 ff             	mov    %edi,%r15d
  400767:	41 55                	push   %r13
  400769:	41 54                	push   %r12
  40076b:	4c 8d 25 6e 06 20 00 	lea    0x20066e(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  400772:	55                   	push   %rbp
  400773:	48 8d 2d 6e 06 20 00 	lea    0x20066e(%rip),%rbp        # 600de8 <__init_array_end>
  40077a:	53                   	push   %rbx
  40077b:	49 89 f6             	mov    %rsi,%r14
  40077e:	49 89 d5             	mov    %rdx,%r13
  400781:	4c 29 e5             	sub    %r12,%rbp
  400784:	48 83 ec 08          	sub    $0x8,%rsp
  400788:	48 c1 fd 03          	sar    $0x3,%rbp
  40078c:	e8 f7 fc ff ff       	callq  400488 <_init>
  400791:	48 85 ed             	test   %rbp,%rbp
  400794:	74 20                	je     4007b6 <__libc_csu_init+0x56>
  400796:	31 db                	xor    %ebx,%ebx
  400798:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40079f:	00 
  4007a0:	4c 89 ea             	mov    %r13,%rdx
  4007a3:	4c 89 f6             	mov    %r14,%rsi
  4007a6:	44 89 ff             	mov    %r15d,%edi
  4007a9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4007ad:	48 83 c3 01          	add    $0x1,%rbx
  4007b1:	48 39 eb             	cmp    %rbp,%rbx
  4007b4:	75 ea                	jne    4007a0 <__libc_csu_init+0x40>
  4007b6:	48 83 c4 08          	add    $0x8,%rsp
  4007ba:	5b                   	pop    %rbx
  4007bb:	5d                   	pop    %rbp
  4007bc:	41 5c                	pop    %r12
  4007be:	41 5d                	pop    %r13
  4007c0:	41 5e                	pop    %r14
  4007c2:	41 5f                	pop    %r15
  4007c4:	c3                   	retq   
  4007c5:	90                   	nop
  4007c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007cd:	00 00 00 

00000000004007d0 <__libc_csu_fini>:
  4007d0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004007d4 <_fini>:
  4007d4:	48 83 ec 08          	sub    $0x8,%rsp
  4007d8:	48 83 c4 08          	add    $0x8,%rsp
  4007dc:	c3                   	retq   
