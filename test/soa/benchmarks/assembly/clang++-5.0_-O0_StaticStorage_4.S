
bin/clang++-5.0_-O0_StaticStorage_4:     file format elf64-x86-64


Disassembly of section .init:

0000000000400550 <_init>:
  400550:	48 83 ec 08          	sub    $0x8,%rsp
  400554:	48 8b 05 9d 1a 20 00 	mov    0x201a9d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40055b:	48 85 c0             	test   %rax,%rax
  40055e:	74 05                	je     400565 <_init+0x15>
  400560:	e8 5b 00 00 00       	callq  4005c0 <_Unwind_Resume@plt+0x10>
  400565:	48 83 c4 08          	add    $0x8,%rsp
  400569:	c3                   	retq   

Disassembly of section .plt:

0000000000400570 <printf@plt-0x10>:
  400570:	ff 35 92 1a 20 00    	pushq  0x201a92(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400576:	ff 25 94 1a 20 00    	jmpq   *0x201a94(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40057c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400580 <printf@plt>:
  400580:	ff 25 92 1a 20 00    	jmpq   *0x201a92(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400586:	68 00 00 00 00       	pushq  $0x0
  40058b:	e9 e0 ff ff ff       	jmpq   400570 <_init+0x20>

0000000000400590 <__libc_start_main@plt>:
  400590:	ff 25 8a 1a 20 00    	jmpq   *0x201a8a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400596:	68 01 00 00 00       	pushq  $0x1
  40059b:	e9 d0 ff ff ff       	jmpq   400570 <_init+0x20>

00000000004005a0 <__gxx_personality_v0@plt>:
  4005a0:	ff 25 82 1a 20 00    	jmpq   *0x201a82(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4005a6:	68 02 00 00 00       	pushq  $0x2
  4005ab:	e9 c0 ff ff ff       	jmpq   400570 <_init+0x20>

00000000004005b0 <_Unwind_Resume@plt>:
  4005b0:	ff 25 7a 1a 20 00    	jmpq   *0x201a7a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4005b6:	68 03 00 00 00       	pushq  $0x3
  4005bb:	e9 b0 ff ff ff       	jmpq   400570 <_init+0x20>

Disassembly of section .plt.got:

00000000004005c0 <.plt.got>:
  4005c0:	ff 25 32 1a 20 00    	jmpq   *0x201a32(%rip)        # 601ff8 <_DYNAMIC+0x200>
  4005c6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004005d0 <_start>:
  4005d0:	31 ed                	xor    %ebp,%ebp
  4005d2:	49 89 d1             	mov    %rdx,%r9
  4005d5:	5e                   	pop    %rsi
  4005d6:	48 89 e2             	mov    %rsp,%rdx
  4005d9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4005dd:	50                   	push   %rax
  4005de:	54                   	push   %rsp
  4005df:	49 c7 c0 a0 0e 40 00 	mov    $0x400ea0,%r8
  4005e6:	48 c7 c1 30 0e 40 00 	mov    $0x400e30,%rcx
  4005ed:	48 c7 c7 d0 06 40 00 	mov    $0x4006d0,%rdi
  4005f4:	e8 97 ff ff ff       	callq  400590 <__libc_start_main@plt>
  4005f9:	f4                   	hlt    
  4005fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400600 <deregister_tm_clones>:
  400600:	b8 4f 20 60 00       	mov    $0x60204f,%eax
  400605:	55                   	push   %rbp
  400606:	48 2d 48 20 60 00    	sub    $0x602048,%rax
  40060c:	48 83 f8 0e          	cmp    $0xe,%rax
  400610:	48 89 e5             	mov    %rsp,%rbp
  400613:	76 1b                	jbe    400630 <deregister_tm_clones+0x30>
  400615:	b8 00 00 00 00       	mov    $0x0,%eax
  40061a:	48 85 c0             	test   %rax,%rax
  40061d:	74 11                	je     400630 <deregister_tm_clones+0x30>
  40061f:	5d                   	pop    %rbp
  400620:	bf 48 20 60 00       	mov    $0x602048,%edi
  400625:	ff e0                	jmpq   *%rax
  400627:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40062e:	00 00 
  400630:	5d                   	pop    %rbp
  400631:	c3                   	retq   
  400632:	0f 1f 40 00          	nopl   0x0(%rax)
  400636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40063d:	00 00 00 

0000000000400640 <register_tm_clones>:
  400640:	be 48 20 60 00       	mov    $0x602048,%esi
  400645:	55                   	push   %rbp
  400646:	48 81 ee 48 20 60 00 	sub    $0x602048,%rsi
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
  40066e:	bf 48 20 60 00       	mov    $0x602048,%edi
  400673:	ff e0                	jmpq   *%rax
  400675:	0f 1f 00             	nopl   (%rax)
  400678:	5d                   	pop    %rbp
  400679:	c3                   	retq   
  40067a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400680 <__do_global_dtors_aux>:
  400680:	80 3d c9 19 20 00 00 	cmpb   $0x0,0x2019c9(%rip)        # 602050 <completed.7585>
  400687:	75 11                	jne    40069a <__do_global_dtors_aux+0x1a>
  400689:	55                   	push   %rbp
  40068a:	48 89 e5             	mov    %rsp,%rbp
  40068d:	e8 6e ff ff ff       	callq  400600 <deregister_tm_clones>
  400692:	5d                   	pop    %rbp
  400693:	c6 05 b6 19 20 00 01 	movb   $0x1,0x2019b6(%rip)        # 602050 <completed.7585>
  40069a:	f3 c3                	repz retq 
  40069c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006a0 <frame_dummy>:
  4006a0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
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

00000000004006d0 <main>:
  4006d0:	48 83 ec 58          	sub    $0x58,%rsp
  4006d4:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  4006db:	00 
  4006dc:	e8 7f 02 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  4006e1:	b8 04 00 00 00       	mov    $0x4,%eax
  4006e6:	89 c7                	mov    %eax,%edi
  4006e8:	e8 83 02 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  4006ed:	48 89 c7             	mov    %rax,%rdi
  4006f0:	48 89 c1             	mov    %rax,%rcx
  4006f3:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4006f9:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  4006fe:	48 89 c7             	mov    %rax,%rdi
  400701:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400706:	e8 95 02 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  40070b:	e9 00 00 00 00       	jmpq   400710 <main+0x40>
  400710:	be 77 77 00 00       	mov    $0x7777,%esi
  400715:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40071a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40071f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400724:	48 83 c1 04          	add    $0x4,%rcx
  400728:	48 89 cf             	mov    %rcx,%rdi
  40072b:	e8 c0 02 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  400730:	be 88 88 00 00       	mov    $0x8888,%esi
  400735:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40073a:	48 83 c1 04          	add    $0x4,%rcx
  40073e:	48 89 cf             	mov    %rcx,%rdi
  400741:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400746:	e8 e5 02 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  40074b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400750:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400755:	e8 16 03 00 00       	callq  400a70 <_ZN9TestClass15increase_field0Ev>
  40075a:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40075f:	e8 4c 03 00 00       	callq  400ab0 <_ZN9TestClass15increase_field1Ev>
  400764:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400769:	48 83 c0 04          	add    $0x4,%rax
  40076d:	48 89 c7             	mov    %rax,%rdi
  400770:	e8 7b 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  400775:	8b 30                	mov    (%rax),%esi
  400777:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40077c:	48 83 c0 04          	add    $0x4,%rax
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400787:	e8 74 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  40078c:	48 bf b4 0e 40 00 00 	movabs $0x400eb4,%rdi
  400793:	00 00 00 
  400796:	8b 10                	mov    (%rax),%edx
  400798:	8b 74 24 14          	mov    0x14(%rsp),%esi
  40079c:	b0 00                	mov    $0x0,%al
  40079e:	e8 dd fd ff ff       	callq  400580 <printf@plt>
  4007a3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007a8:	48 83 c1 04          	add    $0x4,%rcx
  4007ac:	48 89 cf             	mov    %rcx,%rdi
  4007af:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007b3:	e8 38 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  4007b8:	31 d2                	xor    %edx,%edx
  4007ba:	41 88 d0             	mov    %dl,%r8b
  4007bd:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007c3:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  4007c8:	0f 85 1e 00 00 00    	jne    4007ec <main+0x11c>
  4007ce:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007d3:	48 83 c0 04          	add    $0x4,%rax
  4007d7:	48 89 c7             	mov    %rax,%rdi
  4007da:	e8 21 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  4007df:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007e5:	0f 94 c1             	sete   %cl
  4007e8:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007ec:	8a 44 24 0f          	mov    0xf(%rsp),%al
  4007f0:	34 ff                	xor    $0xff,%al
  4007f2:	24 01                	and    $0x1,%al
  4007f4:	0f b6 c0             	movzbl %al,%eax
  4007f7:	48 83 c4 58          	add    $0x58,%rsp
  4007fb:	c3                   	retq   
  4007fc:	89 d1                	mov    %edx,%ecx
  4007fe:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400803:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400807:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40080c:	e8 cf 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400811:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400816:	e8 95 fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  40081b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400820 <_Z12new_instancev>:
  400820:	48 83 ec 28          	sub    $0x28,%rsp
  400824:	b8 04 00 00 00       	mov    $0x4,%eax
  400829:	89 c7                	mov    %eax,%edi
  40082b:	e8 40 01 00 00       	callq  400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400830:	48 89 c7             	mov    %rax,%rdi
  400833:	48 89 c1             	mov    %rax,%rcx
  400836:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40083c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400841:	48 89 c7             	mov    %rax,%rdi
  400844:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400849:	e8 52 01 00 00       	callq  4009a0 <_ZN9TestClassC2Ev>
  40084e:	e9 00 00 00 00       	jmpq   400853 <_Z12new_instancev+0x33>
  400853:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400858:	48 83 c4 28          	add    $0x28,%rsp
  40085c:	c3                   	retq   
  40085d:	89 d1                	mov    %edx,%ecx
  40085f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400864:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  400868:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40086d:	e8 6e 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400872:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400877:	e8 34 fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  40087c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400880 <_Z12write_field0P9TestClass>:
  400880:	48 83 ec 18          	sub    $0x18,%rsp
  400884:	be 77 77 00 00       	mov    $0x7777,%esi
  400889:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40088e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400893:	48 83 c7 04          	add    $0x4,%rdi
  400897:	e8 54 01 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>
  40089c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008a1:	48 83 c4 18          	add    $0x18,%rsp
  4008a5:	c3                   	retq   
  4008a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008ad:	00 00 00 

00000000004008b0 <_Z11read_field0P9TestClass>:
  4008b0:	50                   	push   %rax
  4008b1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008b5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008b9:	48 83 c7 04          	add    $0x4,%rdi
  4008bd:	e8 2e 02 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>
  4008c2:	8b 00                	mov    (%rax),%eax
  4008c4:	59                   	pop    %rcx
  4008c5:	c3                   	retq   
  4008c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008cd:	00 00 00 

00000000004008d0 <_Z12write_field1P9TestClass>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008de:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008e3:	48 83 c7 04          	add    $0x4,%rdi
  4008e7:	e8 44 01 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>
  4008ec:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008f1:	48 83 c4 18          	add    $0x18,%rsp
  4008f5:	c3                   	retq   
  4008f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4008fd:	00 00 00 

0000000000400900 <_Z11read_field1P9TestClass>:
  400900:	50                   	push   %rax
  400901:	48 89 3c 24          	mov    %rdi,(%rsp)
  400905:	48 8b 3c 24          	mov    (%rsp),%rdi
  400909:	48 83 c7 04          	add    $0x4,%rdi
  40090d:	e8 ee 01 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>
  400912:	8b 00                	mov    (%rax),%eax
  400914:	59                   	pop    %rcx
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z21explicit_write_field0m>:
  400920:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400925:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40092a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400931:	77 77 00 00 
  400935:	c3                   	retq   
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <_Z20explicit_read_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400951:	c3                   	retq   
  400952:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400959:	00 00 00 
  40095c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400960:	50                   	push   %rax
  400961:	e8 aa 02 00 00       	callq  400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400966:	48 89 c7             	mov    %rax,%rdi
  400969:	e8 c2 03 00 00       	callq  400d30 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40096e:	58                   	pop    %rax
  40096f:	c3                   	retq   

0000000000400970 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  400970:	50                   	push   %rax
  400971:	48 89 3c 24          	mov    %rdi,(%rsp)
  400975:	e8 16 04 00 00       	callq  400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40097a:	e8 91 02 00 00       	callq  400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  40097f:	b9 01 00 00 00       	mov    $0x1,%ecx
  400984:	89 ce                	mov    %ecx,%esi
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 32 04 00 00       	callq  400dc0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  40098e:	48 89 c7             	mov    %rax,%rdi
  400991:	e8 0a 04 00 00       	callq  400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400996:	59                   	pop    %rcx
  400997:	c3                   	retq   
  400998:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40099f:	00 

00000000004009a0 <_ZN9TestClassC2Ev>:
  4009a0:	48 83 ec 18          	sub    $0x18,%rsp
  4009a4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009a9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009ae:	48 89 f8             	mov    %rdi,%rax
  4009b1:	48 83 c0 04          	add    $0x4,%rax
  4009b5:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009ba:	48 89 c7             	mov    %rax,%rdi
  4009bd:	e8 4e 01 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2Ev>
  4009c2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009c7:	48 83 c0 04          	add    $0x4,%rax
  4009cb:	48 89 c7             	mov    %rax,%rdi
  4009ce:	e8 4d 01 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2Ev>
  4009d3:	48 83 c4 18          	add    $0x18,%rsp
  4009d7:	c3                   	retq   
  4009d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009df:	00 

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  4009e0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009e5:	c3                   	retq   
  4009e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ed:	00 00 00 

00000000004009f0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEaSEi>:
  4009f0:	48 83 ec 28          	sub    $0x28,%rsp
  4009f4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  4009f9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  4009fd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a02:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a06:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a0b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a0f:	e8 7c 01 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a14:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a18:	89 30                	mov    %esi,(%rax)
  400a1a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a1f:	48 83 c4 28          	add    $0x28,%rsp
  400a23:	c3                   	retq   
  400a24:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2b:	00 00 00 
  400a2e:	66 90                	xchg   %ax,%ax

0000000000400a30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEaSEi>:
  400a30:	48 83 ec 28          	sub    $0x28,%rsp
  400a34:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a39:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a3d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a42:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a46:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a4b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a4f:	e8 5c 02 00 00       	callq  400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a54:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a58:	89 30                	mov    %esi,(%rax)
  400a5a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a5f:	48 83 c4 28          	add    $0x28,%rsp
  400a63:	c3                   	retq   
  400a64:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a6b:	00 00 00 
  400a6e:	66 90                	xchg   %ax,%ax

0000000000400a70 <_ZN9TestClass15increase_field0Ev>:
  400a70:	48 83 ec 18          	sub    $0x18,%rsp
  400a74:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a79:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a7e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a83:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a88:	48 83 c7 04          	add    $0x4,%rdi
  400a8c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400a90:	48 89 c7             	mov    %rax,%rdi
  400a93:	48 8b 34 24          	mov    (%rsp),%rsi
  400a97:	e8 94 00 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>
  400a9c:	48 83 c4 18          	add    $0x18,%rsp
  400aa0:	c3                   	retq   
  400aa1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aa8:	00 00 00 
  400aab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ab0 <_ZN9TestClass15increase_field1Ev>:
  400ab0:	48 83 ec 18          	sub    $0x18,%rsp
  400ab4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ab9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400abe:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ac3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ac8:	48 83 c7 04          	add    $0x4,%rdi
  400acc:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ad0:	48 89 c7             	mov    %rax,%rdi
  400ad3:	48 8b 34 24          	mov    (%rsp),%rsi
  400ad7:	e8 74 01 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>
  400adc:	48 83 c4 18          	add    $0x18,%rsp
  400ae0:	c3                   	retq   
  400ae1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ae8:	00 00 00 
  400aeb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400af0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEcvRiEv>:
  400af0:	50                   	push   %rax
  400af1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400af5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400af9:	e8 92 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400afe:	59                   	pop    %rcx
  400aff:	c3                   	retq   

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEcvRiEv>:
  400b00:	50                   	push   %rax
  400b01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b09:	e8 a2 01 00 00       	callq  400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b0e:	59                   	pop    %rcx
  400b0f:	c3                   	retq   

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2Ev>:
  400b10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b15:	c3                   	retq   
  400b16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b1d:	00 00 00 

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2Ev>:
  400b20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b25:	c3                   	retq   
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEmLEi>:
  400b30:	48 83 ec 38          	sub    $0x38,%rsp
  400b34:	48 89 f8             	mov    %rdi,%rax
  400b37:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b3c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b40:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b45:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b49:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b4e:	48 89 f7             	mov    %rsi,%rdi
  400b51:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b56:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b5a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b5f:	e8 2c 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b64:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b68:	0f af 10             	imul   (%rax),%edx
  400b6b:	89 10                	mov    %edx,(%rax)
  400b6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b72:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b77:	e8 24 00 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2ERKS3_>
  400b7c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b81:	48 83 c4 38          	add    $0x38,%rsp
  400b85:	c3                   	retq   
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 

0000000000400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b90:	50                   	push   %rax
  400b91:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b95:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b99:	e8 12 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b9e:	59                   	pop    %rcx
  400b9f:	c3                   	retq   

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassEC2ERKS3_>:
  400ba0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ba5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400baa:	c3                   	retq   
  400bab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb0:	48 83 ec 28          	sub    $0x28,%rsp
  400bb4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bb9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bbe:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400bc3:	e8 48 00 00 00       	callq  400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400bc8:	48 89 c7             	mov    %rax,%rdi
  400bcb:	e8 50 00 00 00       	callq  400c20 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400bd0:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400bd5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bda:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400bdf:	48 83 e8 04          	sub    $0x4,%rax
  400be3:	48 c1 e8 02          	shr    $0x2,%rax
  400be7:	48 c1 e0 02          	shl    $0x2,%rax
  400beb:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400bf0:	48 83 c0 00          	add    $0x0,%rax
  400bf4:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400bf9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bfe:	48 83 c4 28          	add    $0x28,%rsp
  400c02:	c3                   	retq   
  400c03:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c0a:	00 00 00 
  400c0d:	0f 1f 00             	nopl   (%rax)

0000000000400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>:
  400c10:	50                   	push   %rax
  400c11:	e8 2a 00 00 00       	callq  400c40 <_ZN9TestClass14storage_bufferEv>
  400c16:	59                   	pop    %rcx
  400c17:	c3                   	retq   
  400c18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c1f:	00 

0000000000400c20 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c25:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c2a:	48 83 c7 10          	add    $0x10,%rdi
  400c2e:	48 89 f8             	mov    %rdi,%rax
  400c31:	c3                   	retq   
  400c32:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c39:	00 00 00 
  400c3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c40 <_ZN9TestClass14storage_bufferEv>:
  400c40:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c47:	00 00 00 
  400c4a:	c3                   	retq   
  400c4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEmLEi>:
  400c50:	48 83 ec 38          	sub    $0x38,%rsp
  400c54:	48 89 f8             	mov    %rdi,%rax
  400c57:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c5c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c60:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c65:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c69:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c6e:	48 89 f7             	mov    %rsi,%rdi
  400c71:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c76:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c7a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c7f:	e8 2c 00 00 00       	callq  400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c84:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c88:	0f af 10             	imul   (%rax),%edx
  400c8b:	89 10                	mov    %edx,(%rax)
  400c8d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c92:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c97:	e8 24 00 00 00       	callq  400cc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2ERKS3_>
  400c9c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ca1:	48 83 c4 38          	add    $0x38,%rsp
  400ca5:	c3                   	retq   
  400ca6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cad:	00 00 00 

0000000000400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cb0:	50                   	push   %rax
  400cb1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cb5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cb9:	e8 12 00 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cbe:	59                   	pop    %rcx
  400cbf:	c3                   	retq   

0000000000400cc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassEC2ERKS3_>:
  400cc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cc5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cca:	c3                   	retq   
  400ccb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cd0:	48 83 ec 28          	sub    $0x28,%rsp
  400cd4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400cd9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cde:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400ce3:	e8 28 ff ff ff       	callq  400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400ce8:	48 89 c7             	mov    %rax,%rdi
  400ceb:	e8 30 ff ff ff       	callq  400c20 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400cf0:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400cf5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cfa:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400cff:	48 83 e8 04          	sub    $0x4,%rax
  400d03:	48 c1 e8 02          	shr    $0x2,%rax
  400d07:	48 c1 e0 02          	shl    $0x2,%rax
  400d0b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d10:	48 05 d4 48 00 00    	add    $0x48d4,%rax
  400d16:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d1b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d20:	48 83 c4 28          	add    $0x28,%rsp
  400d24:	c3                   	retq   
  400d25:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d2c:	00 00 00 
  400d2f:	90                   	nop

0000000000400d30 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d30:	48 83 ec 18          	sub    $0x18,%rsp
  400d34:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d39:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d3e:	48 89 f8             	mov    %rdi,%rax
  400d41:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d46:	48 89 c7             	mov    %rax,%rdi
  400d49:	e8 12 00 00 00       	callq  400d60 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400d4e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d53:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d5a:	00 
  400d5b:	48 83 c4 18          	add    $0x18,%rsp
  400d5f:	c3                   	retq   

0000000000400d60 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400d60:	50                   	push   %rax
  400d61:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d65:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d69:	e8 02 00 00 00       	callq  400d70 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d6e:	58                   	pop    %rax
  400d6f:	c3                   	retq   

0000000000400d70 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d75:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d7a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d81:	c3                   	retq   
  400d82:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d89:	00 00 00 
  400d8c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d90:	c3                   	retq   
  400d91:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d98:	00 00 00 
  400d9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400da0:	50                   	push   %rax
  400da1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400da5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400da9:	48 83 c7 01          	add    $0x1,%rdi
  400dad:	e8 3e 00 00 00       	callq  400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400db2:	59                   	pop    %rcx
  400db3:	c3                   	retq   
  400db4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dbb:	00 00 00 
  400dbe:	66 90                	xchg   %ax,%ax

0000000000400dc0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400dc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dc5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400dca:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400dcf:	48 8b 3e             	mov    (%rsi),%rdi
  400dd2:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400dd7:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400ddc:	48 03 3e             	add    (%rsi),%rdi
  400ddf:	48 89 3e             	mov    %rdi,(%rsi)
  400de2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400de7:	c3                   	retq   
  400de8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400def:	00 

0000000000400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400df0:	48 83 ec 18          	sub    $0x18,%rsp
  400df4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400df9:	e8 12 fe ff ff       	callq  400c10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400dfe:	48 89 c7             	mov    %rax,%rdi
  400e01:	e8 1a fe ff ff       	callq  400c20 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e06:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e0b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e0f:	48 01 f8             	add    %rdi,%rax
  400e12:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e17:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e1c:	48 83 c4 18          	add    $0x18,%rsp
  400e20:	c3                   	retq   
  400e21:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e28:	00 00 00 
  400e2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e30 <__libc_csu_init>:
  400e30:	41 57                	push   %r15
  400e32:	41 56                	push   %r14
  400e34:	41 89 ff             	mov    %edi,%r15d
  400e37:	41 55                	push   %r13
  400e39:	41 54                	push   %r12
  400e3b:	4c 8d 25 9e 0f 20 00 	lea    0x200f9e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e42:	55                   	push   %rbp
  400e43:	48 8d 2d 9e 0f 20 00 	lea    0x200f9e(%rip),%rbp        # 601de8 <__init_array_end>
  400e4a:	53                   	push   %rbx
  400e4b:	49 89 f6             	mov    %rsi,%r14
  400e4e:	49 89 d5             	mov    %rdx,%r13
  400e51:	4c 29 e5             	sub    %r12,%rbp
  400e54:	48 83 ec 08          	sub    $0x8,%rsp
  400e58:	48 c1 fd 03          	sar    $0x3,%rbp
  400e5c:	e8 ef f6 ff ff       	callq  400550 <_init>
  400e61:	48 85 ed             	test   %rbp,%rbp
  400e64:	74 20                	je     400e86 <__libc_csu_init+0x56>
  400e66:	31 db                	xor    %ebx,%ebx
  400e68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e6f:	00 
  400e70:	4c 89 ea             	mov    %r13,%rdx
  400e73:	4c 89 f6             	mov    %r14,%rsi
  400e76:	44 89 ff             	mov    %r15d,%edi
  400e79:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e7d:	48 83 c3 01          	add    $0x1,%rbx
  400e81:	48 39 eb             	cmp    %rbp,%rbx
  400e84:	75 ea                	jne    400e70 <__libc_csu_init+0x40>
  400e86:	48 83 c4 08          	add    $0x8,%rsp
  400e8a:	5b                   	pop    %rbx
  400e8b:	5d                   	pop    %rbp
  400e8c:	41 5c                	pop    %r12
  400e8e:	41 5d                	pop    %r13
  400e90:	41 5e                	pop    %r14
  400e92:	41 5f                	pop    %r15
  400e94:	c3                   	retq   
  400e95:	90                   	nop
  400e96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e9d:	00 00 00 

0000000000400ea0 <__libc_csu_fini>:
  400ea0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ea4 <_fini>:
  400ea4:	48 83 ec 08          	sub    $0x8,%rsp
  400ea8:	48 83 c4 08          	add    $0x8,%rsp
  400eac:	c3                   	retq   
