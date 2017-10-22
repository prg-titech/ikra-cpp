
bin/zero_clang38_O3:     file format elf64-x86-64


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

0000000000400530 <_GLOBAL__sub_I_codegen_test.cc>:
  400530:	50                   	push   %rax
  400531:	bf a0 91 00 00       	mov    $0x91a0,%edi
  400536:	e8 c5 ff ff ff       	callq  400500 <malloc@plt>
  40053b:	48 89 05 0e 0b 20 00 	mov    %rax,0x200b0e(%rip)        # 601050 <_ZN9TestClass7storageE+0x8>
  400542:	58                   	pop    %rax
  400543:	c3                   	retq   
  400544:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40054b:	00 00 00 
  40054e:	66 90                	xchg   %ax,%ax

0000000000400550 <_start>:
  400550:	31 ed                	xor    %ebp,%ebp
  400552:	49 89 d1             	mov    %rdx,%r9
  400555:	5e                   	pop    %rsi
  400556:	48 89 e2             	mov    %rsp,%rdx
  400559:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40055d:	50                   	push   %rax
  40055e:	54                   	push   %rsp
  40055f:	49 c7 c0 80 07 40 00 	mov    $0x400780,%r8
  400566:	48 c7 c1 10 07 40 00 	mov    $0x400710,%rcx
  40056d:	48 c7 c7 50 06 40 00 	mov    $0x400650,%rdi
  400574:	e8 97 ff ff ff       	callq  400510 <__libc_start_main@plt>
  400579:	f4                   	hlt    
  40057a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400580 <deregister_tm_clones>:
  400580:	b8 47 10 60 00       	mov    $0x601047,%eax
  400585:	55                   	push   %rbp
  400586:	48 2d 40 10 60 00    	sub    $0x601040,%rax
  40058c:	48 83 f8 0e          	cmp    $0xe,%rax
  400590:	48 89 e5             	mov    %rsp,%rbp
  400593:	76 1b                	jbe    4005b0 <deregister_tm_clones+0x30>
  400595:	b8 00 00 00 00       	mov    $0x0,%eax
  40059a:	48 85 c0             	test   %rax,%rax
  40059d:	74 11                	je     4005b0 <deregister_tm_clones+0x30>
  40059f:	5d                   	pop    %rbp
  4005a0:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005a5:	ff e0                	jmpq   *%rax
  4005a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4005ae:	00 00 
  4005b0:	5d                   	pop    %rbp
  4005b1:	c3                   	retq   
  4005b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4005b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005bd:	00 00 00 

00000000004005c0 <register_tm_clones>:
  4005c0:	be 40 10 60 00       	mov    $0x601040,%esi
  4005c5:	55                   	push   %rbp
  4005c6:	48 81 ee 40 10 60 00 	sub    $0x601040,%rsi
  4005cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4005d1:	48 89 e5             	mov    %rsp,%rbp
  4005d4:	48 89 f0             	mov    %rsi,%rax
  4005d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4005db:	48 01 c6             	add    %rax,%rsi
  4005de:	48 d1 fe             	sar    %rsi
  4005e1:	74 15                	je     4005f8 <register_tm_clones+0x38>
  4005e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005e8:	48 85 c0             	test   %rax,%rax
  4005eb:	74 0b                	je     4005f8 <register_tm_clones+0x38>
  4005ed:	5d                   	pop    %rbp
  4005ee:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005f3:	ff e0                	jmpq   *%rax
  4005f5:	0f 1f 00             	nopl   (%rax)
  4005f8:	5d                   	pop    %rbp
  4005f9:	c3                   	retq   
  4005fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400600 <__do_global_dtors_aux>:
  400600:	80 3d 39 0a 20 00 00 	cmpb   $0x0,0x200a39(%rip)        # 601040 <__TMC_END__>
  400607:	75 11                	jne    40061a <__do_global_dtors_aux+0x1a>
  400609:	55                   	push   %rbp
  40060a:	48 89 e5             	mov    %rsp,%rbp
  40060d:	e8 6e ff ff ff       	callq  400580 <deregister_tm_clones>
  400612:	5d                   	pop    %rbp
  400613:	c6 05 26 0a 20 00 01 	movb   $0x1,0x200a26(%rip)        # 601040 <__TMC_END__>
  40061a:	f3 c3                	repz retq 
  40061c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400620 <frame_dummy>:
  400620:	bf f0 0d 60 00       	mov    $0x600df0,%edi
  400625:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400629:	75 05                	jne    400630 <frame_dummy+0x10>
  40062b:	eb 93                	jmp    4005c0 <register_tm_clones>
  40062d:	0f 1f 00             	nopl   (%rax)
  400630:	b8 00 00 00 00       	mov    $0x0,%eax
  400635:	48 85 c0             	test   %rax,%rax
  400638:	74 f1                	je     40062b <frame_dummy+0xb>
  40063a:	55                   	push   %rbp
  40063b:	48 89 e5             	mov    %rsp,%rbp
  40063e:	ff d0                	callq  *%rax
  400640:	5d                   	pop    %rbp
  400641:	e9 7a ff ff ff       	jmpq   4005c0 <register_tm_clones>
  400646:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40064d:	00 00 00 

0000000000400650 <main>:
  400650:	53                   	push   %rbx
  400651:	8b 1d f1 09 20 00    	mov    0x2009f1(%rip),%ebx        # 601048 <_ZN9TestClass7storageE>
  400657:	8d 43 01             	lea    0x1(%rbx),%eax
  40065a:	89 05 e8 09 20 00    	mov    %eax,0x2009e8(%rip)        # 601048 <_ZN9TestClass7storageE>
  400660:	48 8b 05 e9 09 20 00 	mov    0x2009e9(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400667:	c7 84 98 d0 48 00 00 	movl   $0x8888,0x48d0(%rax,%rbx,4)
  40066e:	88 88 00 00 
  400672:	c7 04 98 83 2d d2 27 	movl   $0x27d22d83,(%rax,%rbx,4)
  400679:	69 94 98 d0 48 00 00 	imul   $0x4444,0x48d0(%rax,%rbx,4),%edx
  400680:	44 44 00 00 
  400684:	89 94 98 d0 48 00 00 	mov    %edx,0x48d0(%rax,%rbx,4)
  40068b:	8b 34 98             	mov    (%rax,%rbx,4),%esi
  40068e:	bf 94 07 40 00       	mov    $0x400794,%edi
  400693:	31 c0                	xor    %eax,%eax
  400695:	e8 56 fe ff ff       	callq  4004f0 <printf@plt>
  40069a:	48 8b 0d af 09 20 00 	mov    0x2009af(%rip),%rcx        # 601050 <_ZN9TestClass7storageE+0x8>
  4006a1:	b0 01                	mov    $0x1,%al
  4006a3:	81 3c 99 83 2d d2 27 	cmpl   $0x27d22d83,(%rcx,%rbx,4)
  4006aa:	75 0e                	jne    4006ba <main+0x6a>
  4006ac:	81 bc 99 d0 48 00 00 	cmpl   $0x24686420,0x48d0(%rcx,%rbx,4)
  4006b3:	20 64 68 24 
  4006b7:	0f 95 c0             	setne  %al
  4006ba:	0f b6 c0             	movzbl %al,%eax
  4006bd:	5b                   	pop    %rbx
  4006be:	c3                   	retq   
  4006bf:	90                   	nop

00000000004006c0 <_Z12new_instancev>:
  4006c0:	8b 05 82 09 20 00    	mov    0x200982(%rip),%eax        # 601048 <_ZN9TestClass7storageE>
  4006c6:	8d 48 01             	lea    0x1(%rax),%ecx
  4006c9:	89 0d 79 09 20 00    	mov    %ecx,0x200979(%rip)        # 601048 <_ZN9TestClass7storageE>
  4006cf:	c3                   	retq   

00000000004006d0 <_Z12write_field0P9TestClass>:
  4006d0:	48 8b 05 79 09 20 00 	mov    0x200979(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  4006d7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006de:	c3                   	retq   
  4006df:	90                   	nop

00000000004006e0 <_Z11read_field0P9TestClass>:
  4006e0:	48 8b 05 69 09 20 00 	mov    0x200969(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  4006e7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006ea:	c3                   	retq   
  4006eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004006f0 <_Z21explicit_write_field0m>:
  4006f0:	48 8b 05 61 09 20 00 	mov    0x200961(%rip),%rax        # 601058 <explicit_field0>
  4006f7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006fe:	c3                   	retq   
  4006ff:	90                   	nop

0000000000400700 <_Z20explicit_read_field0m>:
  400700:	48 8b 05 51 09 20 00 	mov    0x200951(%rip),%rax        # 601058 <explicit_field0>
  400707:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40070a:	c3                   	retq   
  40070b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400710 <__libc_csu_init>:
  400710:	41 57                	push   %r15
  400712:	41 56                	push   %r14
  400714:	41 89 ff             	mov    %edi,%r15d
  400717:	41 55                	push   %r13
  400719:	41 54                	push   %r12
  40071b:	4c 8d 25 b6 06 20 00 	lea    0x2006b6(%rip),%r12        # 600dd8 <__frame_dummy_init_array_entry>
  400722:	55                   	push   %rbp
  400723:	48 8d 2d be 06 20 00 	lea    0x2006be(%rip),%rbp        # 600de8 <__init_array_end>
  40072a:	53                   	push   %rbx
  40072b:	49 89 f6             	mov    %rsi,%r14
  40072e:	49 89 d5             	mov    %rdx,%r13
  400731:	4c 29 e5             	sub    %r12,%rbp
  400734:	48 83 ec 08          	sub    $0x8,%rsp
  400738:	48 c1 fd 03          	sar    $0x3,%rbp
  40073c:	e8 7f fd ff ff       	callq  4004c0 <_init>
  400741:	48 85 ed             	test   %rbp,%rbp
  400744:	74 20                	je     400766 <__libc_csu_init+0x56>
  400746:	31 db                	xor    %ebx,%ebx
  400748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40074f:	00 
  400750:	4c 89 ea             	mov    %r13,%rdx
  400753:	4c 89 f6             	mov    %r14,%rsi
  400756:	44 89 ff             	mov    %r15d,%edi
  400759:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40075d:	48 83 c3 01          	add    $0x1,%rbx
  400761:	48 39 eb             	cmp    %rbp,%rbx
  400764:	75 ea                	jne    400750 <__libc_csu_init+0x40>
  400766:	48 83 c4 08          	add    $0x8,%rsp
  40076a:	5b                   	pop    %rbx
  40076b:	5d                   	pop    %rbp
  40076c:	41 5c                	pop    %r12
  40076e:	41 5d                	pop    %r13
  400770:	41 5e                	pop    %r14
  400772:	41 5f                	pop    %r15
  400774:	c3                   	retq   
  400775:	90                   	nop
  400776:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40077d:	00 00 00 

0000000000400780 <__libc_csu_fini>:
  400780:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400784 <_fini>:
  400784:	48 83 ec 08          	sub    $0x8,%rsp
  400788:	48 83 c4 08          	add    $0x8,%rsp
  40078c:	c3                   	retq   
