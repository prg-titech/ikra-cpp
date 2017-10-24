
bin/clang++-3.8_-O3_DynamicStorage_4:     file format elf64-x86-64


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
  40055f:	49 c7 c0 30 08 40 00 	mov    $0x400830,%r8
  400566:	48 c7 c1 c0 07 40 00 	mov    $0x4007c0,%rcx
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
  400651:	8b 05 f1 09 20 00    	mov    0x2009f1(%rip),%eax        # 601048 <_ZN9TestClass7storageE>
  400657:	8d 48 01             	lea    0x1(%rax),%ecx
  40065a:	89 0d e8 09 20 00    	mov    %ecx,0x2009e8(%rip)        # 601048 <_ZN9TestClass7storageE>
  400660:	48 8b 0d e9 09 20 00 	mov    0x2009e9(%rip),%rcx        # 601050 <_ZN9TestClass7storageE+0x8>
  400667:	48 8d 1c 81          	lea    (%rcx,%rax,4),%rbx
  40066b:	c7 04 81 00 00 00 00 	movl   $0x0,(%rcx,%rax,4)
  400672:	48 8b 05 d7 09 20 00 	mov    0x2009d7(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400679:	48 89 d9             	mov    %rbx,%rcx
  40067c:	48 29 c1             	sub    %rax,%rcx
  40067f:	48 83 e1 fc          	and    $0xfffffffffffffffc,%rcx
  400683:	c7 04 01 77 77 00 00 	movl   $0x7777,(%rcx,%rax,1)
  40068a:	c7 84 01 d0 48 00 00 	movl   $0x8888,0x48d0(%rcx,%rax,1)
  400691:	88 88 00 00 
  400695:	69 14 01 55 55 00 00 	imul   $0x5555,(%rcx,%rax,1),%edx
  40069c:	89 14 01             	mov    %edx,(%rcx,%rax,1)
  40069f:	69 94 01 d0 48 00 00 	imul   $0x4444,0x48d0(%rcx,%rax,1),%edx
  4006a6:	44 44 00 00 
  4006aa:	89 94 01 d0 48 00 00 	mov    %edx,0x48d0(%rcx,%rax,1)
  4006b1:	8b 34 01             	mov    (%rcx,%rax,1),%esi
  4006b4:	bf 44 08 40 00       	mov    $0x400844,%edi
  4006b9:	31 c0                	xor    %eax,%eax
  4006bb:	e8 30 fe ff ff       	callq  4004f0 <printf@plt>
  4006c0:	48 8b 05 89 09 20 00 	mov    0x200989(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  4006c7:	48 29 c3             	sub    %rax,%rbx
  4006ca:	48 83 e3 fc          	and    $0xfffffffffffffffc,%rbx
  4006ce:	b1 01                	mov    $0x1,%cl
  4006d0:	81 3c 03 83 2d d2 27 	cmpl   $0x27d22d83,(%rbx,%rax,1)
  4006d7:	75 0e                	jne    4006e7 <main+0x97>
  4006d9:	81 bc 18 d0 48 00 00 	cmpl   $0x24686420,0x48d0(%rax,%rbx,1)
  4006e0:	20 64 68 24 
  4006e4:	0f 95 c1             	setne  %cl
  4006e7:	0f b6 c1             	movzbl %cl,%eax
  4006ea:	5b                   	pop    %rbx
  4006eb:	c3                   	retq   
  4006ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006f0 <_Z12new_instancev>:
  4006f0:	8b 0d 52 09 20 00    	mov    0x200952(%rip),%ecx        # 601048 <_ZN9TestClass7storageE>
  4006f6:	8d 41 01             	lea    0x1(%rcx),%eax
  4006f9:	89 05 49 09 20 00    	mov    %eax,0x200949(%rip)        # 601048 <_ZN9TestClass7storageE>
  4006ff:	48 8b 15 4a 09 20 00 	mov    0x20094a(%rip),%rdx        # 601050 <_ZN9TestClass7storageE+0x8>
  400706:	48 8d 04 8a          	lea    (%rdx,%rcx,4),%rax
  40070a:	c7 04 8a 00 00 00 00 	movl   $0x0,(%rdx,%rcx,4)
  400711:	c3                   	retq   
  400712:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400719:	1f 84 00 00 00 00 00 

0000000000400720 <_Z12write_field0P9TestClass>:
  400720:	48 8b 05 29 09 20 00 	mov    0x200929(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400727:	48 29 c7             	sub    %rax,%rdi
  40072a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40072e:	c7 04 07 77 77 00 00 	movl   $0x7777,(%rdi,%rax,1)
  400735:	c3                   	retq   
  400736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073d:	00 00 00 

0000000000400740 <_Z11read_field0P9TestClass>:
  400740:	48 8b 05 09 09 20 00 	mov    0x200909(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400747:	48 29 c7             	sub    %rax,%rdi
  40074a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40074e:	8b 04 07             	mov    (%rdi,%rax,1),%eax
  400751:	c3                   	retq   
  400752:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400759:	1f 84 00 00 00 00 00 

0000000000400760 <_Z12write_field1P9TestClass>:
  400760:	48 8b 05 e9 08 20 00 	mov    0x2008e9(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400767:	48 29 c7             	sub    %rax,%rdi
  40076a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40076e:	c7 84 38 d0 48 00 00 	movl   $0x7777,0x48d0(%rax,%rdi,1)
  400775:	77 77 00 00 
  400779:	c3                   	retq   
  40077a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400780 <_Z11read_field1P9TestClass>:
  400780:	48 8b 05 c9 08 20 00 	mov    0x2008c9(%rip),%rax        # 601050 <_ZN9TestClass7storageE+0x8>
  400787:	48 29 c7             	sub    %rax,%rdi
  40078a:	48 83 e7 fc          	and    $0xfffffffffffffffc,%rdi
  40078e:	8b 84 38 d0 48 00 00 	mov    0x48d0(%rax,%rdi,1),%eax
  400795:	c3                   	retq   
  400796:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40079d:	00 00 00 

00000000004007a0 <_Z21explicit_write_field0m>:
  4007a0:	48 8b 05 b1 08 20 00 	mov    0x2008b1(%rip),%rax        # 601058 <explicit_field0>
  4007a7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4007ae:	c3                   	retq   
  4007af:	90                   	nop

00000000004007b0 <_Z20explicit_read_field0m>:
  4007b0:	48 8b 05 a1 08 20 00 	mov    0x2008a1(%rip),%rax        # 601058 <explicit_field0>
  4007b7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4007ba:	c3                   	retq   
  4007bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004007c0 <__libc_csu_init>:
  4007c0:	41 57                	push   %r15
  4007c2:	41 56                	push   %r14
  4007c4:	41 89 ff             	mov    %edi,%r15d
  4007c7:	41 55                	push   %r13
  4007c9:	41 54                	push   %r12
  4007cb:	4c 8d 25 06 06 20 00 	lea    0x200606(%rip),%r12        # 600dd8 <__frame_dummy_init_array_entry>
  4007d2:	55                   	push   %rbp
  4007d3:	48 8d 2d 0e 06 20 00 	lea    0x20060e(%rip),%rbp        # 600de8 <__init_array_end>
  4007da:	53                   	push   %rbx
  4007db:	49 89 f6             	mov    %rsi,%r14
  4007de:	49 89 d5             	mov    %rdx,%r13
  4007e1:	4c 29 e5             	sub    %r12,%rbp
  4007e4:	48 83 ec 08          	sub    $0x8,%rsp
  4007e8:	48 c1 fd 03          	sar    $0x3,%rbp
  4007ec:	e8 cf fc ff ff       	callq  4004c0 <_init>
  4007f1:	48 85 ed             	test   %rbp,%rbp
  4007f4:	74 20                	je     400816 <__libc_csu_init+0x56>
  4007f6:	31 db                	xor    %ebx,%ebx
  4007f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4007ff:	00 
  400800:	4c 89 ea             	mov    %r13,%rdx
  400803:	4c 89 f6             	mov    %r14,%rsi
  400806:	44 89 ff             	mov    %r15d,%edi
  400809:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40080d:	48 83 c3 01          	add    $0x1,%rbx
  400811:	48 39 eb             	cmp    %rbp,%rbx
  400814:	75 ea                	jne    400800 <__libc_csu_init+0x40>
  400816:	48 83 c4 08          	add    $0x8,%rsp
  40081a:	5b                   	pop    %rbx
  40081b:	5d                   	pop    %rbp
  40081c:	41 5c                	pop    %r12
  40081e:	41 5d                	pop    %r13
  400820:	41 5e                	pop    %r14
  400822:	41 5f                	pop    %r15
  400824:	c3                   	retq   
  400825:	90                   	nop
  400826:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40082d:	00 00 00 

0000000000400830 <__libc_csu_fini>:
  400830:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400834 <_fini>:
  400834:	48 83 ec 08          	sub    $0x8,%rsp
  400838:	48 83 c4 08          	add    $0x8,%rsp
  40083c:	c3                   	retq   
