
bin/clang++-3.8_-O3_DynamicStorage_0:     file format elf64-x86-64


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
  40053f:	49 c7 c0 c0 07 40 00 	mov    $0x4007c0,%r8
  400546:	48 c7 c1 50 07 40 00 	mov    $0x400750,%rcx
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
  400640:	50                   	push   %rax
  400641:	bf a8 91 00 00       	mov    $0x91a8,%edi
  400646:	e8 b5 fe ff ff       	callq  400500 <malloc@plt>
  40064b:	48 89 05 16 0a 20 00 	mov    %rax,0x200a16(%rip)        # 601068 <__TestClassdata_buffer+0x18>
  400652:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  400656:	c5 f8 11 05 fa 09 20 	vmovups %xmm0,0x2009fa(%rip)        # 601058 <__TestClassdata_buffer+0x8>
  40065d:	00 
  40065e:	48 c7 05 e7 09 20 00 	movq   $0x1,0x2009e7(%rip)        # 601050 <__TestClassdata_buffer>
  400665:	01 00 00 00 
  400669:	c7 40 04 83 2d d2 27 	movl   $0x27d22d83,0x4(%rax)
  400670:	c7 80 d4 48 00 00 20 	movl   $0x24686420,0x48d4(%rax)
  400677:	64 68 24 
  40067a:	bf d4 07 40 00       	mov    $0x4007d4,%edi
  40067f:	be 83 2d d2 27       	mov    $0x27d22d83,%esi
  400684:	ba 20 64 68 24       	mov    $0x24686420,%edx
  400689:	31 c0                	xor    %eax,%eax
  40068b:	e8 60 fe ff ff       	callq  4004f0 <printf@plt>
  400690:	48 8b 0d d1 09 20 00 	mov    0x2009d1(%rip),%rcx        # 601068 <__TestClassdata_buffer+0x18>
  400697:	b0 01                	mov    $0x1,%al
  400699:	81 79 04 83 2d d2 27 	cmpl   $0x27d22d83,0x4(%rcx)
  4006a0:	75 0d                	jne    4006af <main+0x6f>
  4006a2:	81 b9 d4 48 00 00 20 	cmpl   $0x24686420,0x48d4(%rcx)
  4006a9:	64 68 24 
  4006ac:	0f 95 c0             	setne  %al
  4006af:	0f b6 c0             	movzbl %al,%eax
  4006b2:	59                   	pop    %rcx
  4006b3:	c3                   	retq   
  4006b4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006bb:	00 00 00 00 00 

00000000004006c0 <_Z12new_instancev>:
  4006c0:	48 8b 05 89 09 20 00 	mov    0x200989(%rip),%rax        # 601050 <__TestClassdata_buffer>
  4006c7:	48 83 c0 01          	add    $0x1,%rax
  4006cb:	48 89 05 7e 09 20 00 	mov    %rax,0x20097e(%rip)        # 601050 <__TestClassdata_buffer>
  4006d2:	c3                   	retq   
  4006d3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006da:	84 00 00 00 00 00 

00000000004006e0 <_Z12write_field0P9TestClass>:
  4006e0:	48 8b 05 81 09 20 00 	mov    0x200981(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006e7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006ee:	c3                   	retq   
  4006ef:	90                   	nop

00000000004006f0 <_Z11read_field0P9TestClass>:
  4006f0:	48 8b 05 71 09 20 00 	mov    0x200971(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006f7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006fa:	c3                   	retq   
  4006fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400700 <_Z12write_field1P9TestClass>:
  400700:	48 8b 05 61 09 20 00 	mov    0x200961(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400707:	c7 84 b8 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,4)
  40070e:	77 77 00 00 
  400712:	c3                   	retq   
  400713:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40071a:	84 00 00 00 00 00 

0000000000400720 <_Z11read_field1P9TestClass>:
  400720:	48 8b 05 41 09 20 00 	mov    0x200941(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  400727:	8b 84 b8 d0 48 00 00 	mov    0x48d0(%rax,%rdi,4),%eax
  40072e:	c3                   	retq   
  40072f:	90                   	nop

0000000000400730 <_Z21explicit_write_field0m>:
  400730:	c7 04 bd 70 10 60 00 	movl   $0x7777,0x601070(,%rdi,4)
  400737:	77 77 00 00 
  40073b:	c3                   	retq   
  40073c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400740 <_Z20explicit_read_field0m>:
  400740:	8b 04 bd 70 10 60 00 	mov    0x601070(,%rdi,4),%eax
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
  40077c:	e8 3f fd ff ff       	callq  4004c0 <_init>
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
