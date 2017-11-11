
bin/clang++-3.8_-O3_StaticStorage_0:     file format elf64-x86-64


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
  4004ff:	49 c7 c0 50 07 40 00 	mov    $0x400750,%r8
  400506:	48 c7 c1 e0 06 40 00 	mov    $0x4006e0,%rcx
  40050d:	48 c7 c7 00 06 40 00 	mov    $0x400600,%rdi
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

00000000004005f0 <_ZN9TestClass7storageEv>:
  4005f0:	b8 50 10 60 00       	mov    $0x601050,%eax
  4005f5:	c3                   	retq   
  4005f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005fd:	00 00 00 

0000000000400600 <main>:
  400600:	50                   	push   %rax
  400601:	0f 57 c0             	xorps  %xmm0,%xmm0
  400604:	0f 11 05 45 0a 20 00 	movups %xmm0,0x200a45(%rip)        # 601050 <__TestClassdata_buffer>
  40060b:	48 c7 05 3a 0a 20 00 	movq   $0x1,0x200a3a(%rip)        # 601050 <__TestClassdata_buffer>
  400612:	01 00 00 00 
  400616:	c7 05 44 0a 20 00 83 	movl   $0x27d22d83,0x200a44(%rip)        # 601064 <__TestClassdata_buffer+0x14>
  40061d:	2d d2 27 
  400620:	c7 05 0a 53 20 00 20 	movl   $0x24686420,0x20530a(%rip)        # 605934 <__TestClassdata_buffer+0x48e4>
  400627:	64 68 24 
  40062a:	bf 64 07 40 00       	mov    $0x400764,%edi
  40062f:	be 83 2d d2 27       	mov    $0x27d22d83,%esi
  400634:	ba 20 64 68 24       	mov    $0x24686420,%edx
  400639:	31 c0                	xor    %eax,%eax
  40063b:	e8 80 fe ff ff       	callq  4004c0 <printf@plt>
  400640:	b0 01                	mov    $0x1,%al
  400642:	81 3d 18 0a 20 00 83 	cmpl   $0x27d22d83,0x200a18(%rip)        # 601064 <__TestClassdata_buffer+0x14>
  400649:	2d d2 27 
  40064c:	75 0d                	jne    40065b <main+0x5b>
  40064e:	81 3d dc 52 20 00 20 	cmpl   $0x24686420,0x2052dc(%rip)        # 605934 <__TestClassdata_buffer+0x48e4>
  400655:	64 68 24 
  400658:	0f 95 c0             	setne  %al
  40065b:	0f b6 c0             	movzbl %al,%eax
  40065e:	59                   	pop    %rcx
  40065f:	c3                   	retq   

0000000000400660 <_Z12new_instancev>:
  400660:	48 8b 05 e9 09 20 00 	mov    0x2009e9(%rip),%rax        # 601050 <__TestClassdata_buffer>
  400667:	48 ff c0             	inc    %rax
  40066a:	48 89 05 df 09 20 00 	mov    %rax,0x2009df(%rip)        # 601050 <__TestClassdata_buffer>
  400671:	c3                   	retq   
  400672:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400679:	1f 84 00 00 00 00 00 

0000000000400680 <_Z12write_field0P9TestClass>:
  400680:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  400687:	77 77 00 00 
  40068b:	c3                   	retq   
  40068c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400690 <_Z11read_field0P9TestClass>:
  400690:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  400697:	c3                   	retq   
  400698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40069f:	00 

00000000004006a0 <_Z12write_field1P9TestClass>:
  4006a0:	c7 04 bd 30 59 60 00 	movl   $0x7777,0x605930(,%rdi,4)
  4006a7:	77 77 00 00 
  4006ab:	c3                   	retq   
  4006ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006b0 <_Z11read_field1P9TestClass>:
  4006b0:	8b 04 bd 30 59 60 00 	mov    0x605930(,%rdi,4),%eax
  4006b7:	c3                   	retq   
  4006b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006bf:	00 

00000000004006c0 <_Z21explicit_write_field0m>:
  4006c0:	c7 04 bd 00 a2 60 00 	movl   $0x7777,0x60a200(,%rdi,4)
  4006c7:	77 77 00 00 
  4006cb:	c3                   	retq   
  4006cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006d0 <_Z20explicit_read_field0m>:
  4006d0:	8b 04 bd 00 a2 60 00 	mov    0x60a200(,%rdi,4),%eax
  4006d7:	c3                   	retq   
  4006d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006df:	00 

00000000004006e0 <__libc_csu_init>:
  4006e0:	41 57                	push   %r15
  4006e2:	41 56                	push   %r14
  4006e4:	41 89 ff             	mov    %edi,%r15d
  4006e7:	41 55                	push   %r13
  4006e9:	41 54                	push   %r12
  4006eb:	4c 8d 25 ee 06 20 00 	lea    0x2006ee(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  4006f2:	55                   	push   %rbp
  4006f3:	48 8d 2d ee 06 20 00 	lea    0x2006ee(%rip),%rbp        # 600de8 <__init_array_end>
  4006fa:	53                   	push   %rbx
  4006fb:	49 89 f6             	mov    %rsi,%r14
  4006fe:	49 89 d5             	mov    %rdx,%r13
  400701:	4c 29 e5             	sub    %r12,%rbp
  400704:	48 83 ec 08          	sub    $0x8,%rsp
  400708:	48 c1 fd 03          	sar    $0x3,%rbp
  40070c:	e8 77 fd ff ff       	callq  400488 <_init>
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
