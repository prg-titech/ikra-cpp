
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
  4004ff:	49 c7 c0 40 07 40 00 	mov    $0x400740,%r8
  400506:	48 c7 c1 d0 06 40 00 	mov    $0x4006d0,%rcx
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
  4005a0:	80 3d 91 0a 20 00 00 	cmpb   $0x0,0x200a91(%rip)        # 601038 <__TMC_END__>
  4005a7:	75 11                	jne    4005ba <__do_global_dtors_aux+0x1a>
  4005a9:	55                   	push   %rbp
  4005aa:	48 89 e5             	mov    %rsp,%rbp
  4005ad:	e8 6e ff ff ff       	callq  400520 <deregister_tm_clones>
  4005b2:	5d                   	pop    %rbp
  4005b3:	c6 05 7e 0a 20 00 01 	movb   $0x1,0x200a7e(%rip)        # 601038 <__TMC_END__>
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
  4005f1:	8b 1d 49 0a 20 00    	mov    0x200a49(%rip),%ebx        # 601040 <_ZN9TestClass7storageE>
  4005f7:	8d 43 01             	lea    0x1(%rbx),%eax
  4005fa:	89 05 40 0a 20 00    	mov    %eax,0x200a40(%rip)        # 601040 <_ZN9TestClass7storageE>
  400600:	c7 04 9d 14 59 60 00 	movl   $0x8888,0x605914(,%rbx,4)
  400607:	88 88 00 00 
  40060b:	c7 04 9d 44 10 60 00 	movl   $0x27d22d83,0x601044(,%rbx,4)
  400612:	83 2d d2 27 
  400616:	69 14 9d 14 59 60 00 	imul   $0x4444,0x605914(,%rbx,4),%edx
  40061d:	44 44 00 00 
  400621:	89 14 9d 14 59 60 00 	mov    %edx,0x605914(,%rbx,4)
  400628:	8b 34 9d 44 10 60 00 	mov    0x601044(,%rbx,4),%esi
  40062f:	bf 54 07 40 00       	mov    $0x400754,%edi
  400634:	31 c0                	xor    %eax,%eax
  400636:	e8 85 fe ff ff       	callq  4004c0 <printf@plt>
  40063b:	b0 01                	mov    $0x1,%al
  40063d:	81 3c 9d 44 10 60 00 	cmpl   $0x27d22d83,0x601044(,%rbx,4)
  400644:	83 2d d2 27 
  400648:	75 0e                	jne    400658 <main+0x68>
  40064a:	81 3c 9d 14 59 60 00 	cmpl   $0x24686420,0x605914(,%rbx,4)
  400651:	20 64 68 24 
  400655:	0f 95 c0             	setne  %al
  400658:	0f b6 c0             	movzbl %al,%eax
  40065b:	5b                   	pop    %rbx
  40065c:	c3                   	retq   
  40065d:	0f 1f 00             	nopl   (%rax)

0000000000400660 <_Z12new_instancev>:
  400660:	8b 05 da 09 20 00    	mov    0x2009da(%rip),%eax        # 601040 <_ZN9TestClass7storageE>
  400666:	8d 48 01             	lea    0x1(%rax),%ecx
  400669:	89 0d d1 09 20 00    	mov    %ecx,0x2009d1(%rip)        # 601040 <_ZN9TestClass7storageE>
  40066f:	c3                   	retq   

0000000000400670 <_Z12write_field0P9TestClass>:
  400670:	c7 04 bd 44 10 60 00 	movl   $0x7777,0x601044(,%rdi,4)
  400677:	77 77 00 00 
  40067b:	c3                   	retq   
  40067c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400680 <_Z11read_field0P9TestClass>:
  400680:	8b 04 bd 44 10 60 00 	mov    0x601044(,%rdi,4),%eax
  400687:	c3                   	retq   
  400688:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40068f:	00 

0000000000400690 <_Z12write_field1P9TestClass>:
  400690:	c7 04 bd 14 59 60 00 	movl   $0x7777,0x605914(,%rdi,4)
  400697:	77 77 00 00 
  40069b:	c3                   	retq   
  40069c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006a0 <_Z11read_field1P9TestClass>:
  4006a0:	8b 04 bd 14 59 60 00 	mov    0x605914(,%rdi,4),%eax
  4006a7:	c3                   	retq   
  4006a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006af:	00 

00000000004006b0 <_Z21explicit_write_field0m>:
  4006b0:	48 8b 05 31 9b 20 00 	mov    0x209b31(%rip),%rax        # 60a1e8 <explicit_field0>
  4006b7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006be:	c3                   	retq   
  4006bf:	90                   	nop

00000000004006c0 <_Z20explicit_read_field0m>:
  4006c0:	48 8b 05 21 9b 20 00 	mov    0x209b21(%rip),%rax        # 60a1e8 <explicit_field0>
  4006c7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006ca:	c3                   	retq   
  4006cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004006d0 <__libc_csu_init>:
  4006d0:	41 57                	push   %r15
  4006d2:	41 56                	push   %r14
  4006d4:	41 89 ff             	mov    %edi,%r15d
  4006d7:	41 55                	push   %r13
  4006d9:	41 54                	push   %r12
  4006db:	4c 8d 25 fe 06 20 00 	lea    0x2006fe(%rip),%r12        # 600de0 <__frame_dummy_init_array_entry>
  4006e2:	55                   	push   %rbp
  4006e3:	48 8d 2d fe 06 20 00 	lea    0x2006fe(%rip),%rbp        # 600de8 <__init_array_end>
  4006ea:	53                   	push   %rbx
  4006eb:	49 89 f6             	mov    %rsi,%r14
  4006ee:	49 89 d5             	mov    %rdx,%r13
  4006f1:	4c 29 e5             	sub    %r12,%rbp
  4006f4:	48 83 ec 08          	sub    $0x8,%rsp
  4006f8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006fc:	e8 87 fd ff ff       	callq  400488 <_init>
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
