
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
  40055f:	49 c7 c0 d0 07 40 00 	mov    $0x4007d0,%r8
  400566:	48 c7 c1 60 07 40 00 	mov    $0x400760,%rcx
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
  400651:	48 8b 1d f0 09 20 00 	mov    0x2009f0(%rip),%rbx        # 601048 <_ZN9TestClass7storageE>
  400658:	48 8d 43 01          	lea    0x1(%rbx),%rax
  40065c:	48 89 05 e5 09 20 00 	mov    %rax,0x2009e5(%rip)        # 601048 <_ZN9TestClass7storageE>
  400663:	48 8b 05 e6 09 20 00 	mov    0x2009e6(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  40066a:	c7 84 98 d0 48 00 00 	movl   $0x8888,0x48d0(%rax,%rbx,4)
  400671:	88 88 00 00 
  400675:	c7 04 98 83 2d d2 27 	movl   $0x27d22d83,(%rax,%rbx,4)
  40067c:	69 94 98 d0 48 00 00 	imul   $0x4444,0x48d0(%rax,%rbx,4),%edx
  400683:	44 44 00 00 
  400687:	89 94 98 d0 48 00 00 	mov    %edx,0x48d0(%rax,%rbx,4)
  40068e:	8b 34 98             	mov    (%rax,%rbx,4),%esi
  400691:	bf e4 07 40 00       	mov    $0x4007e4,%edi
  400696:	31 c0                	xor    %eax,%eax
  400698:	e8 53 fe ff ff       	callq  4004f0 <printf@plt>
  40069d:	48 8b 0d ac 09 20 00 	mov    0x2009ac(%rip),%rcx        # 601050 <_ZN9TestClass7storageE+0x8>
  4006a4:	b0 01                	mov    $0x1,%al
  4006a6:	81 3c 99 83 2d d2 27 	cmpl   $0x27d22d83,(%rcx,%rbx,4)
  4006ad:	75 0e                	jne    4006bd <main+0x6d>
  4006af:	81 bc 99 d0 48 00 00 	cmpl   $0x24686420,0x48d0(%rcx,%rbx,4)
  4006b6:	20 64 68 24 
  4006ba:	0f 95 c0             	setne  %al
  4006bd:	0f b6 c0             	movzbl %al,%eax
  4006c0:	5b                   	pop    %rbx
  4006c1:	c3                   	retq   
  4006c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006c9:	1f 84 00 00 00 00 00 

00000000004006d0 <_Z12new_instancev>:
  4006d0:	48 8b 05 71 09 20 00 	mov    0x200971(%rip),%rax        # 601048 <_ZN9TestClass7storageE>
  4006d7:	48 8d 48 01          	lea    0x1(%rax),%rcx
  4006db:	48 89 0d 66 09 20 00 	mov    %rcx,0x200966(%rip)        # 601048 <_ZN9TestClass7storageE>
  4006e2:	c3                   	retq   
  4006e3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006ea:	84 00 00 00 00 00 

00000000004006f0 <_Z12write_field0P9TestClass>:
  4006f0:	48 8b 05 59 09 20 00 	mov    0x200959(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  4006f7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006fe:	c3                   	retq   
  4006ff:	90                   	nop

0000000000400700 <_Z11read_field0P9TestClass>:
  400700:	48 8b 05 49 09 20 00 	mov    0x200949(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400707:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40070a:	c3                   	retq   
  40070b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400710 <_Z12write_field1P9TestClass>:
  400710:	48 8b 05 39 09 20 00 	mov    0x200939(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400717:	c7 84 b8 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,4)
  40071e:	77 77 00 00 
  400722:	c3                   	retq   
  400723:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40072a:	84 00 00 00 00 00 

0000000000400730 <_Z11read_field1P9TestClass>:
  400730:	48 8b 05 19 09 20 00 	mov    0x200919(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400737:	8b 84 b8 d0 48 00 00 	mov    0x48d0(%rax,%rdi,4),%eax
  40073e:	c3                   	retq   
  40073f:	90                   	nop

0000000000400740 <_Z21explicit_write_field0m>:
  400740:	48 8b 05 11 09 20 00 	mov    0x200911(%rip),%rax        # 601058 <explicit_field0>
  400747:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40074e:	c3                   	retq   
  40074f:	90                   	nop

0000000000400750 <_Z20explicit_read_field0m>:
  400750:	48 8b 05 01 09 20 00 	mov    0x200901(%rip),%rax        # 601058 <explicit_field0>
  400757:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  40075a:	c3                   	retq   
  40075b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400760 <__libc_csu_init>:
  400760:	41 57                	push   %r15
  400762:	41 56                	push   %r14
  400764:	41 89 ff             	mov    %edi,%r15d
  400767:	41 55                	push   %r13
  400769:	41 54                	push   %r12
  40076b:	4c 8d 25 66 06 20 00 	lea    0x200666(%rip),%r12        # 600dd8 <__frame_dummy_init_array_entry>
  400772:	55                   	push   %rbp
  400773:	48 8d 2d 6e 06 20 00 	lea    0x20066e(%rip),%rbp        # 600de8 <__init_array_end>
  40077a:	53                   	push   %rbx
  40077b:	49 89 f6             	mov    %rsi,%r14
  40077e:	49 89 d5             	mov    %rdx,%r13
  400781:	4c 29 e5             	sub    %r12,%rbp
  400784:	48 83 ec 08          	sub    $0x8,%rsp
  400788:	48 c1 fd 03          	sar    $0x3,%rbp
  40078c:	e8 2f fd ff ff       	callq  4004c0 <_init>
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
