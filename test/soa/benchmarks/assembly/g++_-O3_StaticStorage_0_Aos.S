
bin/g++_-O3_StaticStorage_0_Aos:     file format elf64-x86-64


Disassembly of section .init:

00000000004003f0 <_init>:
  4003f0:	48 83 ec 08          	sub    $0x8,%rsp
  4003f4:	48 8b 05 fd 0b 20 00 	mov    0x200bfd(%rip),%rax        # 600ff8 <_DYNAMIC+0x1d0>
  4003fb:	48 85 c0             	test   %rax,%rax
  4003fe:	74 05                	je     400405 <_init+0x15>
  400400:	e8 3b 00 00 00       	callq  400440 <__printf_chk@plt+0x10>
  400405:	48 83 c4 08          	add    $0x8,%rsp
  400409:	c3                   	retq   

Disassembly of section .plt:

0000000000400410 <__libc_start_main@plt-0x10>:
  400410:	ff 35 f2 0b 20 00    	pushq  0x200bf2(%rip)        # 601008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400416:	ff 25 f4 0b 20 00    	jmpq   *0x200bf4(%rip)        # 601010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40041c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400420 <__libc_start_main@plt>:
  400420:	ff 25 f2 0b 20 00    	jmpq   *0x200bf2(%rip)        # 601018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400426:	68 00 00 00 00       	pushq  $0x0
  40042b:	e9 e0 ff ff ff       	jmpq   400410 <_init+0x20>

0000000000400430 <__printf_chk@plt>:
  400430:	ff 25 ea 0b 20 00    	jmpq   *0x200bea(%rip)        # 601020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400436:	68 01 00 00 00       	pushq  $0x1
  40043b:	e9 d0 ff ff ff       	jmpq   400410 <_init+0x20>

Disassembly of section .plt.got:

0000000000400440 <.plt.got>:
  400440:	ff 25 b2 0b 20 00    	jmpq   *0x200bb2(%rip)        # 600ff8 <_DYNAMIC+0x1d0>
  400446:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400450 <main>:
  400450:	48 83 ec 08          	sub    $0x8,%rsp
  400454:	31 c0                	xor    %eax,%eax
  400456:	b9 20 64 68 24       	mov    $0x24686420,%ecx
  40045b:	c7 05 43 9f 20 00 00 	movl   $0x0,0x209f43(%rip)        # 60a3a8 <__TestClassdata_buffer+0x8>
  400462:	00 00 00 
  400465:	ba 83 2d d2 27       	mov    $0x27d22d83,%edx
  40046a:	be 04 07 40 00       	mov    $0x400704,%esi
  40046f:	bf 01 00 00 00       	mov    $0x1,%edi
  400474:	48 c7 05 21 9f 20 00 	movq   $0x1,0x209f21(%rip)        # 60a3a0 <__TestClassdata_buffer>
  40047b:	01 00 00 00 
  40047f:	c7 05 2f 9f 20 00 83 	movl   $0x27d22d83,0x209f2f(%rip)        # 60a3b8 <__TestClassdata_buffer+0x18>
  400486:	2d d2 27 
  400489:	c7 05 29 9f 20 00 20 	movl   $0x24686420,0x209f29(%rip)        # 60a3bc <__TestClassdata_buffer+0x1c>
  400490:	64 68 24 
  400493:	e8 98 ff ff ff       	callq  400430 <__printf_chk@plt>
  400498:	b8 01 00 00 00       	mov    $0x1,%eax
  40049d:	81 3d 11 9f 20 00 83 	cmpl   $0x27d22d83,0x209f11(%rip)        # 60a3b8 <__TestClassdata_buffer+0x18>
  4004a4:	2d d2 27 
  4004a7:	75 0d                	jne    4004b6 <main+0x66>
  4004a9:	81 3d 09 9f 20 00 20 	cmpl   $0x24686420,0x209f09(%rip)        # 60a3bc <__TestClassdata_buffer+0x1c>
  4004b0:	64 68 24 
  4004b3:	0f 95 c0             	setne  %al
  4004b6:	0f b6 c0             	movzbl %al,%eax
  4004b9:	48 83 c4 08          	add    $0x8,%rsp
  4004bd:	c3                   	retq   
  4004be:	66 90                	xchg   %ax,%ax

00000000004004c0 <_start>:
  4004c0:	31 ed                	xor    %ebp,%ebp
  4004c2:	49 89 d1             	mov    %rdx,%r9
  4004c5:	5e                   	pop    %rsi
  4004c6:	48 89 e2             	mov    %rsp,%rdx
  4004c9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4004cd:	50                   	push   %rax
  4004ce:	54                   	push   %rsp
  4004cf:	49 c7 c0 f0 06 40 00 	mov    $0x4006f0,%r8
  4004d6:	48 c7 c1 80 06 40 00 	mov    $0x400680,%rcx
  4004dd:	48 c7 c7 50 04 40 00 	mov    $0x400450,%rdi
  4004e4:	e8 37 ff ff ff       	callq  400420 <__libc_start_main@plt>
  4004e9:	f4                   	hlt    
  4004ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004004f0 <deregister_tm_clones>:
  4004f0:	b8 3f 10 60 00       	mov    $0x60103f,%eax
  4004f5:	55                   	push   %rbp
  4004f6:	48 2d 38 10 60 00    	sub    $0x601038,%rax
  4004fc:	48 83 f8 0e          	cmp    $0xe,%rax
  400500:	48 89 e5             	mov    %rsp,%rbp
  400503:	76 1b                	jbe    400520 <deregister_tm_clones+0x30>
  400505:	b8 00 00 00 00       	mov    $0x0,%eax
  40050a:	48 85 c0             	test   %rax,%rax
  40050d:	74 11                	je     400520 <deregister_tm_clones+0x30>
  40050f:	5d                   	pop    %rbp
  400510:	bf 38 10 60 00       	mov    $0x601038,%edi
  400515:	ff e0                	jmpq   *%rax
  400517:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40051e:	00 00 
  400520:	5d                   	pop    %rbp
  400521:	c3                   	retq   
  400522:	0f 1f 40 00          	nopl   0x0(%rax)
  400526:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40052d:	00 00 00 

0000000000400530 <register_tm_clones>:
  400530:	be 38 10 60 00       	mov    $0x601038,%esi
  400535:	55                   	push   %rbp
  400536:	48 81 ee 38 10 60 00 	sub    $0x601038,%rsi
  40053d:	48 c1 fe 03          	sar    $0x3,%rsi
  400541:	48 89 e5             	mov    %rsp,%rbp
  400544:	48 89 f0             	mov    %rsi,%rax
  400547:	48 c1 e8 3f          	shr    $0x3f,%rax
  40054b:	48 01 c6             	add    %rax,%rsi
  40054e:	48 d1 fe             	sar    %rsi
  400551:	74 15                	je     400568 <register_tm_clones+0x38>
  400553:	b8 00 00 00 00       	mov    $0x0,%eax
  400558:	48 85 c0             	test   %rax,%rax
  40055b:	74 0b                	je     400568 <register_tm_clones+0x38>
  40055d:	5d                   	pop    %rbp
  40055e:	bf 38 10 60 00       	mov    $0x601038,%edi
  400563:	ff e0                	jmpq   *%rax
  400565:	0f 1f 00             	nopl   (%rax)
  400568:	5d                   	pop    %rbp
  400569:	c3                   	retq   
  40056a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400570 <__do_global_dtors_aux>:
  400570:	80 3d c9 0a 20 00 00 	cmpb   $0x0,0x200ac9(%rip)        # 601040 <completed.7594>
  400577:	75 11                	jne    40058a <__do_global_dtors_aux+0x1a>
  400579:	55                   	push   %rbp
  40057a:	48 89 e5             	mov    %rsp,%rbp
  40057d:	e8 6e ff ff ff       	callq  4004f0 <deregister_tm_clones>
  400582:	5d                   	pop    %rbp
  400583:	c6 05 b6 0a 20 00 01 	movb   $0x1,0x200ab6(%rip)        # 601040 <completed.7594>
  40058a:	f3 c3                	repz retq 
  40058c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400590 <frame_dummy>:
  400590:	bf 20 0e 60 00       	mov    $0x600e20,%edi
  400595:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400599:	75 05                	jne    4005a0 <frame_dummy+0x10>
  40059b:	eb 93                	jmp    400530 <register_tm_clones>
  40059d:	0f 1f 00             	nopl   (%rax)
  4005a0:	b8 00 00 00 00       	mov    $0x0,%eax
  4005a5:	48 85 c0             	test   %rax,%rax
  4005a8:	74 f1                	je     40059b <frame_dummy+0xb>
  4005aa:	55                   	push   %rbp
  4005ab:	48 89 e5             	mov    %rsp,%rbp
  4005ae:	ff d0                	callq  *%rax
  4005b0:	5d                   	pop    %rbp
  4005b1:	e9 7a ff ff ff       	jmpq   400530 <register_tm_clones>
  4005b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005bd:	00 00 00 

00000000004005c0 <_Z12new_instancev>:
  4005c0:	48 8b 05 d9 9d 20 00 	mov    0x209dd9(%rip),%rax        # 60a3a0 <__TestClassdata_buffer>
  4005c7:	48 83 c0 01          	add    $0x1,%rax
  4005cb:	48 89 05 ce 9d 20 00 	mov    %rax,0x209dce(%rip)        # 60a3a0 <__TestClassdata_buffer>
  4005d2:	48 8d 04 c5 b0 a3 60 	lea    0x60a3b0(,%rax,8),%rax
  4005d9:	00 
  4005da:	c3                   	retq   
  4005db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004005e0 <_Z12write_field0P9TestClass>:
  4005e0:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  4005e6:	c3                   	retq   
  4005e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4005ee:	00 00 

00000000004005f0 <_Z11read_field0P9TestClass>:
  4005f0:	8b 07                	mov    (%rdi),%eax
  4005f2:	c3                   	retq   
  4005f3:	0f 1f 00             	nopl   (%rax)
  4005f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4005fd:	00 00 00 

0000000000400600 <_Z12write_field1P9TestClass>:
  400600:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400607:	c3                   	retq   
  400608:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40060f:	00 

0000000000400610 <_Z11read_field1P9TestClass>:
  400610:	8b 47 04             	mov    0x4(%rdi),%eax
  400613:	c3                   	retq   
  400614:	66 90                	xchg   %ax,%ax
  400616:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40061d:	00 00 00 

0000000000400620 <_Z21explicit_write_field0m>:
  400620:	c7 04 bd 60 10 60 00 	movl   $0x7777,0x601060(,%rdi,4)
  400627:	77 77 00 00 
  40062b:	c3                   	retq   
  40062c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400630 <_Z20explicit_read_field0m>:
  400630:	8b 04 bd 60 10 60 00 	mov    0x601060(,%rdi,4),%eax
  400637:	c3                   	retq   
  400638:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40063f:	00 

0000000000400640 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400640:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400646:	c3                   	retq   
  400647:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40064e:	00 00 

0000000000400650 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  400650:	8b 07                	mov    (%rdi),%eax
  400652:	c3                   	retq   
  400653:	0f 1f 00             	nopl   (%rax)
  400656:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40065d:	00 00 00 

0000000000400660 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400660:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400667:	c3                   	retq   
  400668:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40066f:	00 

0000000000400670 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  400670:	8b 47 04             	mov    0x4(%rdi),%eax
  400673:	c3                   	retq   
  400674:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40067b:	00 00 00 
  40067e:	66 90                	xchg   %ax,%ax

0000000000400680 <__libc_csu_init>:
  400680:	41 57                	push   %r15
  400682:	41 56                	push   %r14
  400684:	41 89 ff             	mov    %edi,%r15d
  400687:	41 55                	push   %r13
  400689:	41 54                	push   %r12
  40068b:	4c 8d 25 7e 07 20 00 	lea    0x20077e(%rip),%r12        # 600e10 <__frame_dummy_init_array_entry>
  400692:	55                   	push   %rbp
  400693:	48 8d 2d 7e 07 20 00 	lea    0x20077e(%rip),%rbp        # 600e18 <__init_array_end>
  40069a:	53                   	push   %rbx
  40069b:	49 89 f6             	mov    %rsi,%r14
  40069e:	49 89 d5             	mov    %rdx,%r13
  4006a1:	4c 29 e5             	sub    %r12,%rbp
  4006a4:	48 83 ec 08          	sub    $0x8,%rsp
  4006a8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006ac:	e8 3f fd ff ff       	callq  4003f0 <_init>
  4006b1:	48 85 ed             	test   %rbp,%rbp
  4006b4:	74 20                	je     4006d6 <__libc_csu_init+0x56>
  4006b6:	31 db                	xor    %ebx,%ebx
  4006b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4006bf:	00 
  4006c0:	4c 89 ea             	mov    %r13,%rdx
  4006c3:	4c 89 f6             	mov    %r14,%rsi
  4006c6:	44 89 ff             	mov    %r15d,%edi
  4006c9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  4006cd:	48 83 c3 01          	add    $0x1,%rbx
  4006d1:	48 39 eb             	cmp    %rbp,%rbx
  4006d4:	75 ea                	jne    4006c0 <__libc_csu_init+0x40>
  4006d6:	48 83 c4 08          	add    $0x8,%rsp
  4006da:	5b                   	pop    %rbx
  4006db:	5d                   	pop    %rbp
  4006dc:	41 5c                	pop    %r12
  4006de:	41 5d                	pop    %r13
  4006e0:	41 5e                	pop    %r14
  4006e2:	41 5f                	pop    %r15
  4006e4:	c3                   	retq   
  4006e5:	90                   	nop
  4006e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ed:	00 00 00 

00000000004006f0 <__libc_csu_fini>:
  4006f0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000004006f4 <_fini>:
  4006f4:	48 83 ec 08          	sub    $0x8,%rsp
  4006f8:	48 83 c4 08          	add    $0x8,%rsp
  4006fc:	c3                   	retq   
