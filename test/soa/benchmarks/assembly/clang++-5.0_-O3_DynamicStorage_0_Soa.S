
bin/clang++-5.0_-O3_DynamicStorage_0_Soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004004a0 <_init>:
  4004a0:	48 83 ec 08          	sub    $0x8,%rsp
  4004a4:	48 8b 05 4d 0b 20 00 	mov    0x200b4d(%rip),%rax        # 600ff8 <_DYNAMIC+0x200>
  4004ab:	48 85 c0             	test   %rax,%rax
  4004ae:	74 05                	je     4004b5 <_init+0x15>
  4004b0:	e8 4b 00 00 00       	callq  400500 <__libc_start_main@plt+0x10>
  4004b5:	48 83 c4 08          	add    $0x8,%rsp
  4004b9:	c3                   	retq   

Disassembly of section .plt:

00000000004004c0 <printf@plt-0x10>:
  4004c0:	ff 35 42 0b 20 00    	pushq  0x200b42(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4004c6:	ff 25 44 0b 20 00    	jmpq   *0x200b44(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4004cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004004d0 <printf@plt>:
  4004d0:	ff 25 42 0b 20 00    	jmpq   *0x200b42(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4004d6:	68 00 00 00 00       	pushq  $0x0
  4004db:	e9 e0 ff ff ff       	jmpq   4004c0 <_init+0x20>

00000000004004e0 <malloc@plt>:
  4004e0:	ff 25 3a 0b 20 00    	jmpq   *0x200b3a(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4004e6:	68 01 00 00 00       	pushq  $0x1
  4004eb:	e9 d0 ff ff ff       	jmpq   4004c0 <_init+0x20>

00000000004004f0 <__libc_start_main@plt>:
  4004f0:	ff 25 32 0b 20 00    	jmpq   *0x200b32(%rip)        # 601028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4004f6:	68 02 00 00 00       	pushq  $0x2
  4004fb:	e9 c0 ff ff ff       	jmpq   4004c0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400500 <.plt.got>:
  400500:	ff 25 f2 0a 20 00    	jmpq   *0x200af2(%rip)        # 600ff8 <_DYNAMIC+0x200>
  400506:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400510 <_start>:
  400510:	31 ed                	xor    %ebp,%ebp
  400512:	49 89 d1             	mov    %rdx,%r9
  400515:	5e                   	pop    %rsi
  400516:	48 89 e2             	mov    %rsp,%rdx
  400519:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40051d:	50                   	push   %rax
  40051e:	54                   	push   %rsp
  40051f:	49 c7 c0 d0 07 40 00 	mov    $0x4007d0,%r8
  400526:	48 c7 c1 60 07 40 00 	mov    $0x400760,%rcx
  40052d:	48 c7 c7 10 06 40 00 	mov    $0x400610,%rdi
  400534:	e8 b7 ff ff ff       	callq  4004f0 <__libc_start_main@plt>
  400539:	f4                   	hlt    
  40053a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400540 <deregister_tm_clones>:
  400540:	b8 47 10 60 00       	mov    $0x601047,%eax
  400545:	55                   	push   %rbp
  400546:	48 2d 40 10 60 00    	sub    $0x601040,%rax
  40054c:	48 83 f8 0e          	cmp    $0xe,%rax
  400550:	48 89 e5             	mov    %rsp,%rbp
  400553:	76 1b                	jbe    400570 <deregister_tm_clones+0x30>
  400555:	b8 00 00 00 00       	mov    $0x0,%eax
  40055a:	48 85 c0             	test   %rax,%rax
  40055d:	74 11                	je     400570 <deregister_tm_clones+0x30>
  40055f:	5d                   	pop    %rbp
  400560:	bf 40 10 60 00       	mov    $0x601040,%edi
  400565:	ff e0                	jmpq   *%rax
  400567:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40056e:	00 00 
  400570:	5d                   	pop    %rbp
  400571:	c3                   	retq   
  400572:	0f 1f 40 00          	nopl   0x0(%rax)
  400576:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40057d:	00 00 00 

0000000000400580 <register_tm_clones>:
  400580:	be 40 10 60 00       	mov    $0x601040,%esi
  400585:	55                   	push   %rbp
  400586:	48 81 ee 40 10 60 00 	sub    $0x601040,%rsi
  40058d:	48 c1 fe 03          	sar    $0x3,%rsi
  400591:	48 89 e5             	mov    %rsp,%rbp
  400594:	48 89 f0             	mov    %rsi,%rax
  400597:	48 c1 e8 3f          	shr    $0x3f,%rax
  40059b:	48 01 c6             	add    %rax,%rsi
  40059e:	48 d1 fe             	sar    %rsi
  4005a1:	74 15                	je     4005b8 <register_tm_clones+0x38>
  4005a3:	b8 00 00 00 00       	mov    $0x0,%eax
  4005a8:	48 85 c0             	test   %rax,%rax
  4005ab:	74 0b                	je     4005b8 <register_tm_clones+0x38>
  4005ad:	5d                   	pop    %rbp
  4005ae:	bf 40 10 60 00       	mov    $0x601040,%edi
  4005b3:	ff e0                	jmpq   *%rax
  4005b5:	0f 1f 00             	nopl   (%rax)
  4005b8:	5d                   	pop    %rbp
  4005b9:	c3                   	retq   
  4005ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004005c0 <__do_global_dtors_aux>:
  4005c0:	80 3d 79 0a 20 00 00 	cmpb   $0x0,0x200a79(%rip)        # 601040 <__TMC_END__>
  4005c7:	75 11                	jne    4005da <__do_global_dtors_aux+0x1a>
  4005c9:	55                   	push   %rbp
  4005ca:	48 89 e5             	mov    %rsp,%rbp
  4005cd:	e8 6e ff ff ff       	callq  400540 <deregister_tm_clones>
  4005d2:	5d                   	pop    %rbp
  4005d3:	c6 05 66 0a 20 00 01 	movb   $0x1,0x200a66(%rip)        # 601040 <__TMC_END__>
  4005da:	f3 c3                	repz retq 
  4005dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005e0 <frame_dummy>:
  4005e0:	bf f0 0d 60 00       	mov    $0x600df0,%edi
  4005e5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4005e9:	75 05                	jne    4005f0 <frame_dummy+0x10>
  4005eb:	eb 93                	jmp    400580 <register_tm_clones>
  4005ed:	0f 1f 00             	nopl   (%rax)
  4005f0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005f5:	48 85 c0             	test   %rax,%rax
  4005f8:	74 f1                	je     4005eb <frame_dummy+0xb>
  4005fa:	55                   	push   %rbp
  4005fb:	48 89 e5             	mov    %rsp,%rbp
  4005fe:	ff d0                	callq  *%rax
  400600:	5d                   	pop    %rbp
  400601:	e9 7a ff ff ff       	jmpq   400580 <register_tm_clones>
  400606:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40060d:	00 00 00 

0000000000400610 <main>:
  400610:	50                   	push   %rax
  400611:	bf c0 91 00 00       	mov    $0x91c0,%edi
  400616:	e8 c5 fe ff ff       	callq  4004e0 <malloc@plt>
  40061b:	48 89 05 46 0a 20 00 	mov    %rax,0x200a46(%rip)        # 601068 <__TestClassdata_buffer+0x18>
  400622:	c7 05 34 0a 20 00 00 	movl   $0x0,0x200a34(%rip)        # 601060 <__TestClassdata_buffer+0x10>
  400629:	00 00 00 
  40062c:	48 c7 05 21 0a 20 00 	movq   $0x0,0x200a21(%rip)        # 601058 <__TestClassdata_buffer+0x8>
  400633:	00 00 00 00 
  400637:	48 c7 05 0e 0a 20 00 	movq   $0x1,0x200a0e(%rip)        # 601050 <__TestClassdata_buffer>
  40063e:	01 00 00 00 
  400642:	c7 40 04 83 2d d2 27 	movl   $0x27d22d83,0x4(%rax)
  400649:	c7 80 e4 48 00 00 20 	movl   $0x24686420,0x48e4(%rax)
  400650:	64 68 24 
  400653:	bf e4 07 40 00       	mov    $0x4007e4,%edi
  400658:	be 83 2d d2 27       	mov    $0x27d22d83,%esi
  40065d:	ba 20 64 68 24       	mov    $0x24686420,%edx
  400662:	31 c0                	xor    %eax,%eax
  400664:	e8 67 fe ff ff       	callq  4004d0 <printf@plt>
  400669:	48 8b 0d f8 09 20 00 	mov    0x2009f8(%rip),%rcx        # 601068 <__TestClassdata_buffer+0x18>
  400670:	b8 01 00 00 00       	mov    $0x1,%eax
  400675:	81 79 04 83 2d d2 27 	cmpl   $0x27d22d83,0x4(%rcx)
  40067c:	75 0f                	jne    40068d <main+0x7d>
  40067e:	31 c0                	xor    %eax,%eax
  400680:	81 b9 e4 48 00 00 20 	cmpl   $0x24686420,0x48e4(%rcx)
  400687:	64 68 24 
  40068a:	0f 95 c0             	setne  %al
  40068d:	59                   	pop    %rcx
  40068e:	c3                   	retq   
  40068f:	90                   	nop

0000000000400690 <_Z12new_instancev>:
  400690:	48 8b 05 b9 09 20 00 	mov    0x2009b9(%rip),%rax        # 601050 <__TestClassdata_buffer>
  400697:	48 83 c0 01          	add    $0x1,%rax
  40069b:	48 89 05 ae 09 20 00 	mov    %rax,0x2009ae(%rip)        # 601050 <__TestClassdata_buffer>
  4006a2:	c3                   	retq   
  4006a3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006aa:	84 00 00 00 00 00 

00000000004006b0 <_Z12write_field0P9TestClass>:
  4006b0:	48 8b 05 b1 09 20 00 	mov    0x2009b1(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006b7:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4006be:	c3                   	retq   
  4006bf:	90                   	nop

00000000004006c0 <_Z11read_field0P9TestClass>:
  4006c0:	48 8b 05 a1 09 20 00 	mov    0x2009a1(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006c7:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4006ca:	c3                   	retq   
  4006cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004006d0 <_Z12write_field1P9TestClass>:
  4006d0:	48 8b 05 91 09 20 00 	mov    0x200991(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006d7:	c7 84 b8 e0 48 00 00 	movl   $0x7777,0x48e0(%rax,%rdi,4)
  4006de:	77 77 00 00 
  4006e2:	c3                   	retq   
  4006e3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4006ea:	84 00 00 00 00 00 

00000000004006f0 <_Z11read_field1P9TestClass>:
  4006f0:	48 8b 05 71 09 20 00 	mov    0x200971(%rip),%rax        # 601068 <__TestClassdata_buffer+0x18>
  4006f7:	8b 84 b8 e0 48 00 00 	mov    0x48e0(%rax,%rdi,4),%eax
  4006fe:	c3                   	retq   
  4006ff:	90                   	nop

0000000000400700 <_Z21explicit_write_field0m>:
  400700:	c7 04 bd 10 a2 60 00 	movl   $0x7777,0x60a210(,%rdi,4)
  400707:	77 77 00 00 
  40070b:	c3                   	retq   
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <_Z20explicit_read_field0m>:
  400710:	8b 04 bd 10 a2 60 00 	mov    0x60a210(,%rdi,4),%eax
  400717:	c3                   	retq   
  400718:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40071f:	00 

0000000000400720 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400720:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400726:	c3                   	retq   
  400727:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40072e:	00 00 

0000000000400730 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  400730:	8b 07                	mov    (%rdi),%eax
  400732:	c3                   	retq   
  400733:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40073a:	84 00 00 00 00 00 

0000000000400740 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400740:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400747:	c3                   	retq   
  400748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40074f:	00 

0000000000400750 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  400750:	8b 47 04             	mov    0x4(%rdi),%eax
  400753:	c3                   	retq   
  400754:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40075b:	00 00 00 
  40075e:	66 90                	xchg   %ax,%ax

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
  40078c:	e8 0f fd ff ff       	callq  4004a0 <_init>
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
