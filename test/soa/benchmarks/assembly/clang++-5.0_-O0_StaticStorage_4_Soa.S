
bin/clang++-5.0_-O0_StaticStorage_4_Soa:     file format elf64-x86-64


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
  4005df:	49 c7 c0 30 0f 40 00 	mov    $0x400f30,%r8
  4005e6:	48 c7 c1 c0 0e 40 00 	mov    $0x400ec0,%rcx
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
  400680:	80 3d c9 19 20 00 00 	cmpb   $0x0,0x2019c9(%rip)        # 602050 <completed.7594>
  400687:	75 11                	jne    40069a <__do_global_dtors_aux+0x1a>
  400689:	55                   	push   %rbp
  40068a:	48 89 e5             	mov    %rsp,%rbp
  40068d:	e8 6e ff ff ff       	callq  400600 <deregister_tm_clones>
  400692:	5d                   	pop    %rbp
  400693:	c6 05 b6 19 20 00 01 	movb   $0x1,0x2019b6(%rip)        # 602050 <completed.7594>
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
  4006dc:	e8 df 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIJEEEvDpT_>
  4006e1:	b8 04 00 00 00       	mov    $0x4,%eax
  4006e6:	89 c7                	mov    %eax,%edi
  4006e8:	e8 e3 02 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>
  4006ed:	48 89 c7             	mov    %rax,%rdi
  4006f0:	48 89 c1             	mov    %rax,%rcx
  4006f3:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4006f9:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  4006fe:	48 89 c7             	mov    %rax,%rdi
  400701:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400706:	e8 f5 02 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  40070b:	e9 00 00 00 00       	jmpq   400710 <main+0x40>
  400710:	be 77 77 00 00       	mov    $0x7777,%esi
  400715:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40071a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40071f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400724:	48 83 c1 04          	add    $0x4,%rcx
  400728:	48 89 cf             	mov    %rcx,%rdi
  40072b:	e8 20 03 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>
  400730:	be 88 88 00 00       	mov    $0x8888,%esi
  400735:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40073a:	48 83 c1 04          	add    $0x4,%rcx
  40073e:	48 89 cf             	mov    %rcx,%rdi
  400741:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400746:	e8 45 03 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>
  40074b:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400750:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400755:	e8 76 03 00 00       	callq  400ad0 <_ZN9TestClass15increase_field0Ev>
  40075a:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40075f:	e8 ac 03 00 00       	callq  400b10 <_ZN9TestClass15increase_field1Ev>
  400764:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400769:	48 83 c0 04          	add    $0x4,%rax
  40076d:	48 89 c7             	mov    %rax,%rdi
  400770:	e8 db 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
  400775:	8b 30                	mov    (%rax),%esi
  400777:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40077c:	48 83 c0 04          	add    $0x4,%rax
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400787:	e8 d4 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
  40078c:	48 bf 44 0f 40 00 00 	movabs $0x400f44,%rdi
  400793:	00 00 00 
  400796:	8b 10                	mov    (%rax),%edx
  400798:	8b 74 24 14          	mov    0x14(%rsp),%esi
  40079c:	b0 00                	mov    $0x0,%al
  40079e:	e8 dd fd ff ff       	callq  400580 <printf@plt>
  4007a3:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007a8:	48 83 c1 04          	add    $0x4,%rcx
  4007ac:	48 89 cf             	mov    %rcx,%rdi
  4007af:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007b3:	e8 98 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
  4007b8:	31 d2                	xor    %edx,%edx
  4007ba:	41 88 d0             	mov    %dl,%r8b
  4007bd:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007c3:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  4007c8:	0f 85 1e 00 00 00    	jne    4007ec <main+0x11c>
  4007ce:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007d3:	48 83 c0 04          	add    $0x4,%rax
  4007d7:	48 89 c7             	mov    %rax,%rdi
  4007da:	e8 81 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
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
  40080c:	e8 2f 02 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEdlEPv>
  400811:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400816:	e8 95 fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  40081b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400820 <_Z12new_instancev>:
  400820:	48 83 ec 28          	sub    $0x28,%rsp
  400824:	b8 04 00 00 00       	mov    $0x4,%eax
  400829:	89 c7                	mov    %eax,%edi
  40082b:	e8 a0 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>
  400830:	48 89 c7             	mov    %rax,%rdi
  400833:	48 89 c1             	mov    %rax,%rcx
  400836:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40083c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400841:	48 89 c7             	mov    %rax,%rdi
  400844:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400849:	e8 b2 01 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  40084e:	e9 00 00 00 00       	jmpq   400853 <_Z12new_instancev+0x33>
  400853:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400858:	48 83 c4 28          	add    $0x28,%rsp
  40085c:	c3                   	retq   
  40085d:	89 d1                	mov    %edx,%ecx
  40085f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400864:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  400868:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40086d:	e8 ce 01 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEdlEPv>
  400872:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400877:	e8 34 fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  40087c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400880 <_Z12write_field0P9TestClass>:
  400880:	48 83 ec 18          	sub    $0x18,%rsp
  400884:	be 77 77 00 00       	mov    $0x7777,%esi
  400889:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40088e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400893:	48 83 c7 04          	add    $0x4,%rdi
  400897:	e8 b4 01 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>
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
  4008bd:	e8 8e 02 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
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
  4008e7:	e8 a4 01 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>
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
  40090d:	e8 4e 02 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
  400912:	8b 00                	mov    (%rax),%eax
  400914:	59                   	pop    %rcx
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z21explicit_write_field0m>:
  400920:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400925:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40092a:	c7 04 bd d0 43 61 00 	movl   $0x7777,0x6143d0(,%rdi,4)
  400931:	77 77 00 00 
  400935:	c3                   	retq   
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <_Z20explicit_read_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	8b 04 bd d0 43 61 00 	mov    0x6143d0(,%rdi,4),%eax
  400951:	c3                   	retq   
  400952:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400959:	1f 84 00 00 00 00 00 

0000000000400960 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400970:	c3                   	retq   
  400971:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400978:	0f 1f 84 00 00 00 00 
  40097f:	00 

0000000000400980 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  400980:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400985:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40098a:	8b 07                	mov    (%rdi),%eax
  40098c:	c3                   	retq   
  40098d:	0f 1f 00             	nopl   (%rax)

0000000000400990 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400990:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400995:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40099a:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  4009a1:	c3                   	retq   
  4009a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009a9:	1f 84 00 00 00 00 00 

00000000004009b0 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  4009b0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ba:	8b 47 04             	mov    0x4(%rdi),%eax
  4009bd:	c3                   	retq   
  4009be:	66 90                	xchg   %ax,%ax

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIJEEEvDpT_>:
  4009c0:	50                   	push   %rax
  4009c1:	e8 aa 02 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  4009c6:	48 89 c7             	mov    %rax,%rdi
  4009c9:	e8 c2 03 00 00       	callq  400d90 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  4009ce:	58                   	pop    %rax
  4009cf:	c3                   	retq   

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>:
  4009d0:	50                   	push   %rax
  4009d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009d5:	e8 06 04 00 00       	callq  400de0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009da:	e8 91 02 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  4009df:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009e4:	89 ce                	mov    %ecx,%esi
  4009e6:	48 89 c7             	mov    %rax,%rdi
  4009e9:	e8 22 04 00 00       	callq  400e10 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  4009ee:	48 89 c7             	mov    %rax,%rdi
  4009f1:	e8 fa 03 00 00       	callq  400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE3getEm>
  4009f6:	59                   	pop    %rcx
  4009f7:	c3                   	retq   
  4009f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009ff:	00 

0000000000400a00 <_ZN9TestClassC2Ev>:
  400a00:	48 83 ec 18          	sub    $0x18,%rsp
  400a04:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a09:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a0e:	48 89 f8             	mov    %rdi,%rax
  400a11:	48 83 c0 04          	add    $0x4,%rax
  400a15:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a1a:	48 89 c7             	mov    %rax,%rdi
  400a1d:	e8 4e 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC2Ev>
  400a22:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a27:	48 83 c0 04          	add    $0x4,%rax
  400a2b:	48 89 c7             	mov    %rax,%rdi
  400a2e:	e8 4d 01 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC2Ev>
  400a33:	48 83 c4 18          	add    $0x18,%rsp
  400a37:	c3                   	retq   
  400a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a3f:	00 

0000000000400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEdlEPv>:
  400a40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a45:	c3                   	retq   
  400a46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4d:	00 00 00 

0000000000400a50 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>:
  400a50:	48 83 ec 28          	sub    $0x28,%rsp
  400a54:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a59:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a5d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a62:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a66:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a6b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a6f:	e8 7c 01 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400a74:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a78:	89 30                	mov    %esi,(%rax)
  400a7a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a7f:	48 83 c4 28          	add    $0x28,%rsp
  400a83:	c3                   	retq   
  400a84:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8b:	00 00 00 
  400a8e:	66 90                	xchg   %ax,%ax

0000000000400a90 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>:
  400a90:	48 83 ec 28          	sub    $0x28,%rsp
  400a94:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a99:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a9d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400aa2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400aa6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400aab:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400aaf:	e8 5c 02 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400ab4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400ab8:	89 30                	mov    %esi,(%rax)
  400aba:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400abf:	48 83 c4 28          	add    $0x28,%rsp
  400ac3:	c3                   	retq   
  400ac4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400acb:	00 00 00 
  400ace:	66 90                	xchg   %ax,%ax

0000000000400ad0 <_ZN9TestClass15increase_field0Ev>:
  400ad0:	48 83 ec 18          	sub    $0x18,%rsp
  400ad4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ad9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ade:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ae3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400ae8:	48 83 c7 04          	add    $0x4,%rdi
  400aec:	48 89 3c 24          	mov    %rdi,(%rsp)
  400af0:	48 89 c7             	mov    %rax,%rdi
  400af3:	48 8b 34 24          	mov    (%rsp),%rsi
  400af7:	e8 94 00 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEmLEi>
  400afc:	48 83 c4 18          	add    $0x18,%rsp
  400b00:	c3                   	retq   
  400b01:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b08:	00 00 00 
  400b0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b10 <_ZN9TestClass15increase_field1Ev>:
  400b10:	48 83 ec 18          	sub    $0x18,%rsp
  400b14:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b19:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b1e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b23:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b28:	48 83 c7 04          	add    $0x4,%rdi
  400b2c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b30:	48 89 c7             	mov    %rax,%rdi
  400b33:	48 8b 34 24          	mov    (%rsp),%rsi
  400b37:	e8 74 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEmLEi>
  400b3c:	48 83 c4 18          	add    $0x18,%rsp
  400b40:	c3                   	retq   
  400b41:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b48:	00 00 00 
  400b4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>:
  400b50:	50                   	push   %rax
  400b51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b59:	e8 92 00 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400b5e:	59                   	pop    %rcx
  400b5f:	c3                   	retq   

0000000000400b60 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>:
  400b60:	50                   	push   %rax
  400b61:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b65:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b69:	e8 a2 01 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400b6e:	59                   	pop    %rcx
  400b6f:	c3                   	retq   

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC2Ev>:
  400b70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b75:	c3                   	retq   
  400b76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7d:	00 00 00 

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC2Ev>:
  400b80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b85:	c3                   	retq   
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEmLEi>:
  400b90:	48 83 ec 38          	sub    $0x38,%rsp
  400b94:	48 89 f8             	mov    %rdi,%rax
  400b97:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b9c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400ba0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400ba5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400ba9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bae:	48 89 f7             	mov    %rsi,%rdi
  400bb1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bb6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bba:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bbf:	e8 2c 00 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400bc4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bc8:	0f af 10             	imul   (%rax),%edx
  400bcb:	89 10                	mov    %edx,(%rax)
  400bcd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bd2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400bd7:	e8 24 00 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC2ERKS3_>
  400bdc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400be1:	48 83 c4 38          	add    $0x38,%rsp
  400be5:	c3                   	retq   
  400be6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bed:	00 00 00 

0000000000400bf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>:
  400bf0:	50                   	push   %rax
  400bf1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bf5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bf9:	e8 12 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400bfe:	59                   	pop    %rcx
  400bff:	c3                   	retq   

0000000000400c00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC2ERKS3_>:
  400c00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c05:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c0a:	c3                   	retq   
  400c0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400c10:	48 83 ec 28          	sub    $0x28,%rsp
  400c14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c19:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c1e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400c23:	e8 48 00 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400c28:	48 89 c7             	mov    %rax,%rdi
  400c2b:	e8 50 00 00 00       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400c30:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c35:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c3a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c3f:	48 83 e8 04          	sub    $0x4,%rax
  400c43:	48 c1 e8 02          	shr    $0x2,%rax
  400c47:	48 c1 e0 02          	shl    $0x2,%rax
  400c4b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c50:	48 83 c0 00          	add    $0x0,%rax
  400c54:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c59:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c5e:	48 83 c4 28          	add    $0x28,%rsp
  400c62:	c3                   	retq   
  400c63:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6a:	00 00 00 
  400c6d:	0f 1f 00             	nopl   (%rax)

0000000000400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>:
  400c70:	50                   	push   %rax
  400c71:	e8 2a 00 00 00       	callq  400ca0 <_ZN9TestClass14storage_bufferEv>
  400c76:	59                   	pop    %rcx
  400c77:	c3                   	retq   
  400c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c7f:	00 

0000000000400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400c80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c85:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c8a:	48 83 c7 10          	add    $0x10,%rdi
  400c8e:	48 89 f8             	mov    %rdi,%rax
  400c91:	c3                   	retq   
  400c92:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c99:	00 00 00 
  400c9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ca0 <_ZN9TestClass14storage_bufferEv>:
  400ca0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400ca7:	00 00 00 
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEmLEi>:
  400cb0:	48 83 ec 38          	sub    $0x38,%rsp
  400cb4:	48 89 f8             	mov    %rdi,%rax
  400cb7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400cbc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400cc0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400cc5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400cc9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400cce:	48 89 f7             	mov    %rsi,%rdi
  400cd1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400cd6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400cda:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400cdf:	e8 2c 00 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400ce4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400ce8:	0f af 10             	imul   (%rax),%edx
  400ceb:	89 10                	mov    %edx,(%rax)
  400ced:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cf2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400cf7:	e8 24 00 00 00       	callq  400d20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC2ERKS3_>
  400cfc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d01:	48 83 c4 38          	add    $0x38,%rsp
  400d05:	c3                   	retq   
  400d06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d0d:	00 00 00 

0000000000400d10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>:
  400d10:	50                   	push   %rax
  400d11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d19:	e8 12 00 00 00       	callq  400d30 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400d1e:	59                   	pop    %rcx
  400d1f:	c3                   	retq   

0000000000400d20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC2ERKS3_>:
  400d20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d2a:	c3                   	retq   
  400d2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d30 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d30:	48 83 ec 28          	sub    $0x28,%rsp
  400d34:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d39:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d3e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d43:	e8 28 ff ff ff       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400d48:	48 89 c7             	mov    %rax,%rdi
  400d4b:	e8 30 ff ff ff       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d50:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d55:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d5a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d5f:	48 83 e8 04          	sub    $0x4,%rax
  400d63:	48 c1 e8 02          	shr    $0x2,%rax
  400d67:	48 c1 e0 02          	shl    $0x2,%rax
  400d6b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d70:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400d76:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d80:	48 83 c4 28          	add    $0x28,%rsp
  400d84:	c3                   	retq   
  400d85:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d8c:	00 00 00 
  400d8f:	90                   	nop

0000000000400d90 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400d90:	48 83 ec 18          	sub    $0x18,%rsp
  400d94:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d99:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d9e:	48 89 f8             	mov    %rdi,%rax
  400da1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400da6:	48 89 c7             	mov    %rax,%rdi
  400da9:	e8 12 00 00 00       	callq  400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400dae:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400db3:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400dba:	48 83 c4 18          	add    $0x18,%rsp
  400dbe:	c3                   	retq   
  400dbf:	90                   	nop

0000000000400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400dc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dc5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400dca:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400dd1:	c3                   	retq   
  400dd2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dd9:	00 00 00 
  400ddc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400de0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400de0:	c3                   	retq   
  400de1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400de8:	00 00 00 
  400deb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE3getEm>:
  400df0:	50                   	push   %rax
  400df1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400df5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400df9:	48 83 c7 01          	add    $0x1,%rdi
  400dfd:	e8 4e 00 00 00       	callq  400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e02:	59                   	pop    %rcx
  400e03:	c3                   	retq   
  400e04:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e0b:	00 00 00 
  400e0e:	66 90                	xchg   %ax,%ax

0000000000400e10 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400e10:	48 83 ec 28          	sub    $0x28,%rsp
  400e14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400e19:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  400e1e:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  400e23:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400e28:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  400e2d:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400e32:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e37:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400e3c:	e8 4f 00 00 00       	callq  400e90 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400e41:	48 83 c4 28          	add    $0x28,%rsp
  400e45:	c3                   	retq   
  400e46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e4d:	00 00 00 

0000000000400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e50:	48 83 ec 18          	sub    $0x18,%rsp
  400e54:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e59:	e8 12 fe ff ff       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400e5e:	48 89 c7             	mov    %rax,%rdi
  400e61:	e8 1a fe ff ff       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e66:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e6b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e6f:	48 01 f8             	add    %rdi,%rax
  400e72:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e77:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e7c:	48 83 c4 18          	add    $0x18,%rsp
  400e80:	c3                   	retq   
  400e81:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e88:	00 00 00 
  400e8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e90 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400e90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e95:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400e9a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400e9f:	48 8b 36             	mov    (%rsi),%rsi
  400ea2:	48 89 74 24 e8       	mov    %rsi,-0x18(%rsp)
  400ea7:	48 8b 74 24 f0       	mov    -0x10(%rsp),%rsi
  400eac:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400eb1:	48 03 37             	add    (%rdi),%rsi
  400eb4:	48 89 37             	mov    %rsi,(%rdi)
  400eb7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ebc:	c3                   	retq   
  400ebd:	0f 1f 00             	nopl   (%rax)

0000000000400ec0 <__libc_csu_init>:
  400ec0:	41 57                	push   %r15
  400ec2:	41 56                	push   %r14
  400ec4:	41 89 ff             	mov    %edi,%r15d
  400ec7:	41 55                	push   %r13
  400ec9:	41 54                	push   %r12
  400ecb:	4c 8d 25 0e 0f 20 00 	lea    0x200f0e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400ed2:	55                   	push   %rbp
  400ed3:	48 8d 2d 0e 0f 20 00 	lea    0x200f0e(%rip),%rbp        # 601de8 <__init_array_end>
  400eda:	53                   	push   %rbx
  400edb:	49 89 f6             	mov    %rsi,%r14
  400ede:	49 89 d5             	mov    %rdx,%r13
  400ee1:	4c 29 e5             	sub    %r12,%rbp
  400ee4:	48 83 ec 08          	sub    $0x8,%rsp
  400ee8:	48 c1 fd 03          	sar    $0x3,%rbp
  400eec:	e8 5f f6 ff ff       	callq  400550 <_init>
  400ef1:	48 85 ed             	test   %rbp,%rbp
  400ef4:	74 20                	je     400f16 <__libc_csu_init+0x56>
  400ef6:	31 db                	xor    %ebx,%ebx
  400ef8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400eff:	00 
  400f00:	4c 89 ea             	mov    %r13,%rdx
  400f03:	4c 89 f6             	mov    %r14,%rsi
  400f06:	44 89 ff             	mov    %r15d,%edi
  400f09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400f0d:	48 83 c3 01          	add    $0x1,%rbx
  400f11:	48 39 eb             	cmp    %rbp,%rbx
  400f14:	75 ea                	jne    400f00 <__libc_csu_init+0x40>
  400f16:	48 83 c4 08          	add    $0x8,%rsp
  400f1a:	5b                   	pop    %rbx
  400f1b:	5d                   	pop    %rbp
  400f1c:	41 5c                	pop    %r12
  400f1e:	41 5d                	pop    %r13
  400f20:	41 5e                	pop    %r14
  400f22:	41 5f                	pop    %r15
  400f24:	c3                   	retq   
  400f25:	90                   	nop
  400f26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f2d:	00 00 00 

0000000000400f30 <__libc_csu_fini>:
  400f30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f34 <_fini>:
  400f34:	48 83 ec 08          	sub    $0x8,%rsp
  400f38:	48 83 c4 08          	add    $0x8,%rsp
  400f3c:	c3                   	retq   
