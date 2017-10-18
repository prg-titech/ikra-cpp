
a.out:     file format elf64-x86-64


Disassembly of section .init:

0000000000400430 <_init>:
  400430:	48 83 ec 08          	sub    $0x8,%rsp
  400434:	48 8b 05 bd 1b 20 00 	mov    0x201bbd(%rip),%rax        # 601ff8 <_DYNAMIC+0x1d0>
  40043b:	48 85 c0             	test   %rax,%rax
  40043e:	74 05                	je     400445 <_init+0x15>
  400440:	e8 4b 00 00 00       	callq  400490 <__printf_chk@plt+0x10>
  400445:	48 83 c4 08          	add    $0x8,%rsp
  400449:	c3                   	retq   

Disassembly of section .plt:

0000000000400450 <__assert_fail@plt-0x10>:
  400450:	ff 35 b2 1b 20 00    	pushq  0x201bb2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400456:	ff 25 b4 1b 20 00    	jmpq   *0x201bb4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40045c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400460 <__assert_fail@plt>:
  400460:	ff 25 b2 1b 20 00    	jmpq   *0x201bb2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400466:	68 00 00 00 00       	pushq  $0x0
  40046b:	e9 e0 ff ff ff       	jmpq   400450 <_init+0x20>

0000000000400470 <__libc_start_main@plt>:
  400470:	ff 25 aa 1b 20 00    	jmpq   *0x201baa(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400476:	68 01 00 00 00       	pushq  $0x1
  40047b:	e9 d0 ff ff ff       	jmpq   400450 <_init+0x20>

0000000000400480 <__printf_chk@plt>:
  400480:	ff 25 a2 1b 20 00    	jmpq   *0x201ba2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400486:	68 02 00 00 00       	pushq  $0x2
  40048b:	e9 c0 ff ff ff       	jmpq   400450 <_init+0x20>

Disassembly of section .plt.got:

0000000000400490 <.plt.got>:
  400490:	ff 25 62 1b 20 00    	jmpq   *0x201b62(%rip)        # 601ff8 <_DYNAMIC+0x1d0>
  400496:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004004a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv.part.5>:
  4004a0:	50                   	push   %rax
  4004a1:	b9 80 09 40 00       	mov    $0x400980,%ecx
  4004a6:	ba 85 00 00 00       	mov    $0x85,%edx
  4004ab:	be 64 07 40 00       	mov    $0x400764,%esi
  4004b0:	bf 77 07 40 00       	mov    $0x400777,%edi
  4004b5:	e8 a6 ff ff ff       	callq  400460 <__assert_fail@plt>

00000000004004ba <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv.part.6>:
  4004ba:	50                   	push   %rax
  4004bb:	b9 00 08 40 00       	mov    $0x400800,%ecx
  4004c0:	ba 85 00 00 00       	mov    $0x85,%edx
  4004c5:	be 64 07 40 00       	mov    $0x400764,%esi
  4004ca:	bf 77 07 40 00       	mov    $0x400777,%edi
  4004cf:	e8 8c ff ff ff       	callq  400460 <__assert_fail@plt>
  4004d4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4004db:	00 00 00 
  4004de:	66 90                	xchg   %ax,%ax

00000000004004e0 <main>:
  4004e0:	48 83 ec 08          	sub    $0x8,%rsp
  4004e4:	8b 05 76 1b 20 00    	mov    0x201b76(%rip),%eax        # 602060 <_ZN9TestClass7storageE>
  4004ea:	3d 34 12 00 00       	cmp    $0x1234,%eax
  4004ef:	0f 87 c2 00 00 00    	ja     4005b7 <main+0xd7>
  4004f5:	8d 50 01             	lea    0x1(%rax),%edx
  4004f8:	48 39 d0             	cmp    %rdx,%rax
  4004fb:	89 15 5f 1b 20 00    	mov    %edx,0x201b5f(%rip)        # 602060 <_ZN9TestClass7storageE>
  400501:	0f 87 97 00 00 00    	ja     40059e <main+0xbe>
  400507:	0f 83 8c 00 00 00    	jae    400599 <main+0xb9>
  40050d:	c7 04 85 64 20 60 00 	movl   $0x7777,0x602064(,%rax,4)
  400514:	77 77 00 00 
  400518:	8b 35 42 1b 20 00    	mov    0x201b42(%rip),%esi        # 602060 <_ZN9TestClass7storageE>
  40051e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400525:	00 
  400526:	48 8d 8a 64 20 60 00 	lea    0x602064(%rdx),%rcx
  40052d:	48 39 f0             	cmp    %rsi,%rax
  400530:	73 62                	jae    400594 <main+0xb4>
  400532:	c7 82 34 69 60 00 88 	movl   $0x8888,0x606934(%rdx)
  400539:	88 00 00 
  40053c:	8b 35 1e 1b 20 00    	mov    0x201b1e(%rip),%esi        # 602060 <_ZN9TestClass7storageE>
  400542:	48 81 c2 d0 48 00 00 	add    $0x48d0,%rdx
  400549:	48 39 f0             	cmp    %rsi,%rax
  40054c:	73 4b                	jae    400599 <main+0xb9>
  40054e:	d1 21                	shll   (%rcx)
  400550:	8b 0d 0a 1b 20 00    	mov    0x201b0a(%rip),%ecx        # 602060 <_ZN9TestClass7storageE>
  400556:	48 39 c8             	cmp    %rcx,%rax
  400559:	73 39                	jae    400594 <main+0xb4>
  40055b:	8b ba 64 20 60 00    	mov    0x602064(%rdx),%edi
  400561:	8d 0c 3f             	lea    (%rdi,%rdi,1),%ecx
  400564:	89 8a 64 20 60 00    	mov    %ecx,0x602064(%rdx)
  40056a:	8b 15 f0 1a 20 00    	mov    0x201af0(%rip),%edx        # 602060 <_ZN9TestClass7storageE>
  400570:	48 39 d0             	cmp    %rdx,%rax
  400573:	73 1f                	jae    400594 <main+0xb4>
  400575:	8b 14 85 64 20 60 00 	mov    0x602064(,%rax,4),%edx
  40057c:	be ab 07 40 00       	mov    $0x4007ab,%esi
  400581:	bf 01 00 00 00       	mov    $0x1,%edi
  400586:	31 c0                	xor    %eax,%eax
  400588:	e8 f3 fe ff ff       	callq  400480 <__printf_chk@plt>
  40058d:	31 c0                	xor    %eax,%eax
  40058f:	48 83 c4 08          	add    $0x8,%rsp
  400593:	c3                   	retq   
  400594:	e8 21 ff ff ff       	callq  4004ba <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv.part.6>
  400599:	e8 02 ff ff ff       	callq  4004a0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv.part.5>
  40059e:	b9 00 0b 40 00       	mov    $0x400b00,%ecx
  4005a3:	ba cd 00 00 00       	mov    $0xcd,%edx
  4005a8:	be 64 07 40 00       	mov    $0x400764,%esi
  4005ad:	bf 92 07 40 00       	mov    $0x400792,%edi
  4005b2:	e8 a9 fe ff ff       	callq  400460 <__assert_fail@plt>
  4005b7:	b9 00 0c 40 00       	mov    $0x400c00,%ecx
  4005bc:	ba a6 00 00 00       	mov    $0xa6,%edx
  4005c1:	be 64 07 40 00       	mov    $0x400764,%esi
  4005c6:	bf c8 07 40 00       	mov    $0x4007c8,%edi
  4005cb:	e8 90 fe ff ff       	callq  400460 <__assert_fail@plt>

00000000004005d0 <_start>:
  4005d0:	31 ed                	xor    %ebp,%ebp
  4005d2:	49 89 d1             	mov    %rdx,%r9
  4005d5:	5e                   	pop    %rsi
  4005d6:	48 89 e2             	mov    %rsp,%rdx
  4005d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4005dd:	50                   	push   %rax
  4005de:	54                   	push   %rsp
  4005df:	49 c7 c0 40 07 40 00 	mov    $0x400740,%r8
  4005e6:	48 c7 c1 d0 06 40 00 	mov    $0x4006d0,%rcx
  4005ed:	48 c7 c7 e0 04 40 00 	mov    $0x4004e0,%rdi
  4005f4:	e8 77 fe ff ff       	callq  400470 <__libc_start_main@plt>
  4005f9:	f4                   	hlt    
  4005fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400600 <deregister_tm_clones>:
  400600:	b8 47 20 60 00       	mov    $0x602047,%eax
  400605:	55                   	push   %rbp
  400606:	48 2d 40 20 60 00    	sub    $0x602040,%rax
  40060c:	48 83 f8 0e          	cmp    $0xe,%rax
  400610:	48 89 e5             	mov    %rsp,%rbp
  400613:	76 1b                	jbe    400630 <deregister_tm_clones+0x30>
  400615:	b8 00 00 00 00       	mov    $0x0,%eax
  40061a:	48 85 c0             	test   %rax,%rax
  40061d:	74 11                	je     400630 <deregister_tm_clones+0x30>
  40061f:	5d                   	pop    %rbp
  400620:	bf 40 20 60 00       	mov    $0x602040,%edi
  400625:	ff e0                	jmpq   *%rax
  400627:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40062e:	00 00 
  400630:	5d                   	pop    %rbp
  400631:	c3                   	retq   
  400632:	0f 1f 40 00          	nopl   0x0(%rax)
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <register_tm_clones>:
  400640:	be 40 20 60 00       	mov    $0x602040,%esi
  400645:	55                   	push   %rbp
  400646:	48 81 ee 40 20 60 00 	sub    $0x602040,%rsi
  40064d:	48 c1 fe 03          	sar    $0x3,%rsi
  400651:	48 89 e5             	mov    %rsp,%rbp
  400654:	48 89 f0             	mov    %rsi,%rax
  400657:	48 c1 e8 3f          	shr    $0x3f,%rax
  40065b:	48 01 c6             	add    %rax,%rsi
  40065e:	48 d1 fe             	sar    %rsi
  400661:	74 15                	je     400678 <register_tm_clones+0x38>
  400663:	b8 00 00 00 00       	mov    $0x0,%eax
  400668:	48 85 c0             	test   %rax,%rax
  40066b:	74 0b                	je     400678 <register_tm_clones+0x38>
  40066d:	5d                   	pop    %rbp
  40066e:	bf 40 20 60 00       	mov    $0x602040,%edi
  400673:	ff e0                	jmpq   *%rax
  400675:	0f 1f 00             	nopl   (%rax)
  400678:	5d                   	pop    %rbp
  400679:	c3                   	retq   
  40067a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400680 <__do_global_dtors_aux>:
  400680:	80 3d b9 19 20 00 00 	cmpb   $0x0,0x2019b9(%rip)        # 602040 <__TMC_END__>
  400687:	75 11                	jne    40069a <__do_global_dtors_aux+0x1a>
  400689:	55                   	push   %rbp
  40068a:	48 89 e5             	mov    %rsp,%rbp
  40068d:	e8 6e ff ff ff       	callq  400600 <deregister_tm_clones>
  400692:	5d                   	pop    %rbp
  400693:	c6 05 a6 19 20 00 01 	movb   $0x1,0x2019a6(%rip)        # 602040 <__TMC_END__>
  40069a:	f3 c3                	repz retq 
  40069c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006a0 <frame_dummy>:
  4006a0:	bf 20 1e 60 00       	mov    $0x601e20,%edi
  4006a5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4006a9:	75 05                	jne    4006b0 <frame_dummy+0x10>
  4006ab:	eb 93                	jmp    400640 <register_tm_clones>
  4006ad:	0f 1f 00             	nopl   (%rax)
  4006b0:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b5:	48 85 c0             	test   %rax,%rax
  4006b8:	74 f1                	je     4006ab <frame_dummy+0xb>
  4006ba:	55                   	push   %rbp
  4006bb:	48 89 e5             	mov    %rsp,%rbp
  4006be:	ff d0                	callq  *%rax
  4006c0:	5d                   	pop    %rbp
  4006c1:	e9 7a ff ff ff       	jmpq   400640 <register_tm_clones>
  4006c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006cd:	00 00 00 

00000000004006d0 <__libc_csu_init>:
  4006d0:	41 57                	push   %r15
  4006d2:	41 56                	push   %r14
  4006d4:	41 89 ff             	mov    %edi,%r15d
  4006d7:	41 55                	push   %r13
  4006d9:	41 54                	push   %r12
  4006db:	4c 8d 25 2e 17 20 00 	lea    0x20172e(%rip),%r12        # 601e10 <__frame_dummy_init_array_entry>
  4006e2:	55                   	push   %rbp
  4006e3:	48 8d 2d 2e 17 20 00 	lea    0x20172e(%rip),%rbp        # 601e18 <__init_array_end>
  4006ea:	53                   	push   %rbx
  4006eb:	49 89 f6             	mov    %rsi,%r14
  4006ee:	49 89 d5             	mov    %rdx,%r13
  4006f1:	4c 29 e5             	sub    %r12,%rbp
  4006f4:	48 83 ec 08          	sub    $0x8,%rsp
  4006f8:	48 c1 fd 03          	sar    $0x3,%rbp
  4006fc:	e8 2f fd ff ff       	callq  400430 <_init>
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
