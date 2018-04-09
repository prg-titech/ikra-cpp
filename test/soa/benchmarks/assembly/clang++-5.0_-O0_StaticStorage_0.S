
bin/clang++-5.0_-O0_StaticStorage_0:     file format elf64-x86-64


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
  4005df:	49 c7 c0 d0 0d 40 00 	mov    $0x400dd0,%r8
  4005e6:	48 c7 c1 60 0d 40 00 	mov    $0x400d60,%rcx
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
  4006dc:	e8 4f 02 00 00       	callq  400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>
  4006e1:	31 c0                	xor    %eax,%eax
  4006e3:	89 c7                	mov    %eax,%edi
  4006e5:	e8 56 02 00 00       	callq  400940 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4006ea:	48 89 c7             	mov    %rax,%rdi
  4006ed:	48 89 c1             	mov    %rax,%rcx
  4006f0:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  4006f5:	48 89 c7             	mov    %rax,%rdi
  4006f8:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  4006fd:	e8 6e 02 00 00       	callq  400970 <_ZN9TestClassC2Ev>
  400702:	e9 00 00 00 00       	jmpq   400707 <main+0x37>
  400707:	be 77 77 00 00       	mov    $0x7777,%esi
  40070c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400711:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400716:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40071b:	e8 90 02 00 00       	callq  4009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  400720:	be 88 88 00 00       	mov    $0x8888,%esi
  400725:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40072a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40072f:	e8 bc 02 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  400734:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400739:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40073e:	e8 ed 02 00 00       	callq  400a30 <_ZN9TestClass15increase_field0Ev>
  400743:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400748:	e8 13 03 00 00       	callq  400a60 <_ZN9TestClass15increase_field1Ev>
  40074d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400752:	e8 39 03 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  400757:	8b 30                	mov    (%rax),%esi
  400759:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40075e:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400762:	e8 39 03 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  400767:	48 bf e4 0d 40 00 00 	movabs $0x400de4,%rdi
  40076e:	00 00 00 
  400771:	8b 10                	mov    (%rax),%edx
  400773:	8b 74 24 14          	mov    0x14(%rsp),%esi
  400777:	b0 00                	mov    $0x0,%al
  400779:	e8 02 fe ff ff       	callq  400580 <printf@plt>
  40077e:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400783:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400787:	e8 04 03 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  40078c:	31 d2                	xor    %edx,%edx
  40078e:	88 d1                	mov    %dl,%cl
  400790:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400796:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40079a:	0f 85 17 00 00 00    	jne    4007b7 <main+0xe7>
  4007a0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a5:	e8 f6 02 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  4007aa:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007b0:	0f 94 c1             	sete   %cl
  4007b3:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007b7:	8a 44 24 0f          	mov    0xf(%rsp),%al
  4007bb:	34 ff                	xor    $0xff,%al
  4007bd:	24 01                	and    $0x1,%al
  4007bf:	0f b6 c0             	movzbl %al,%eax
  4007c2:	48 83 c4 58          	add    $0x58,%rsp
  4007c6:	c3                   	retq   
  4007c7:	89 d1                	mov    %edx,%ecx
  4007c9:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  4007ce:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  4007d2:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  4007d7:	e8 c4 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  4007dc:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  4007e1:	e8 ca fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  4007e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007ed:	00 00 00 

00000000004007f0 <_Z12new_instancev>:
  4007f0:	48 83 ec 28          	sub    $0x28,%rsp
  4007f4:	31 c0                	xor    %eax,%eax
  4007f6:	89 c7                	mov    %eax,%edi
  4007f8:	e8 43 01 00 00       	callq  400940 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  4007fd:	48 89 c7             	mov    %rax,%rdi
  400800:	48 89 c1             	mov    %rax,%rcx
  400803:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400808:	48 89 c7             	mov    %rax,%rdi
  40080b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400810:	e8 5b 01 00 00       	callq  400970 <_ZN9TestClassC2Ev>
  400815:	e9 00 00 00 00       	jmpq   40081a <_Z12new_instancev+0x2a>
  40081a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081f:	48 83 c4 28          	add    $0x28,%rsp
  400823:	c3                   	retq   
  400824:	89 d1                	mov    %edx,%ecx
  400826:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40082b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40082f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400834:	e8 67 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>
  400839:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40083e:	e8 6d fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  400843:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40084a:	84 00 00 00 00 00 

0000000000400850 <_Z12write_field0P9TestClass>:
  400850:	48 83 ec 18          	sub    $0x18,%rsp
  400854:	be 77 77 00 00       	mov    $0x7777,%esi
  400859:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40085e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400863:	e8 48 01 00 00       	callq  4009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>
  400868:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40086d:	48 83 c4 18          	add    $0x18,%rsp
  400871:	c3                   	retq   
  400872:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400879:	1f 84 00 00 00 00 00 

0000000000400880 <_Z11read_field0P9TestClass>:
  400880:	50                   	push   %rax
  400881:	48 89 3c 24          	mov    %rdi,(%rsp)
  400885:	48 8b 3c 24          	mov    (%rsp),%rdi
  400889:	e8 02 02 00 00       	callq  400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>
  40088e:	8b 00                	mov    (%rax),%eax
  400890:	59                   	pop    %rcx
  400891:	c3                   	retq   
  400892:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400899:	1f 84 00 00 00 00 00 

00000000004008a0 <_Z12write_field1P9TestClass>:
  4008a0:	48 83 ec 18          	sub    $0x18,%rsp
  4008a4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008a9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ae:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008b3:	e8 38 01 00 00       	callq  4009f0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>
  4008b8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008bd:	48 83 c4 18          	add    $0x18,%rsp
  4008c1:	c3                   	retq   
  4008c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c9:	1f 84 00 00 00 00 00 

00000000004008d0 <_Z11read_field1P9TestClass>:
  4008d0:	50                   	push   %rax
  4008d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008d5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008d9:	e8 c2 01 00 00       	callq  400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>
  4008de:	8b 00                	mov    (%rax),%eax
  4008e0:	59                   	pop    %rcx
  4008e1:	c3                   	retq   
  4008e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008e9:	1f 84 00 00 00 00 00 

00000000004008f0 <_Z21explicit_write_field0m>:
  4008f0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008f5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4008fa:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400901:	77 77 00 00 
  400905:	c3                   	retq   
  400906:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40090d:	00 00 00 

0000000000400910 <_Z20explicit_read_field0m>:
  400910:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400915:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40091a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400921:	c3                   	retq   
  400922:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400929:	00 00 00 
  40092c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400930 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIJEEEvDpT_>:
  400930:	50                   	push   %rax
  400931:	e8 3a 03 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400936:	48 89 c7             	mov    %rax,%rdi
  400939:	e8 42 03 00 00       	callq  400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40093e:	58                   	pop    %rax
  40093f:	c3                   	retq   

0000000000400940 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400940:	50                   	push   %rax
  400941:	48 89 3c 24          	mov    %rdi,(%rsp)
  400945:	e8 a6 03 00 00       	callq  400cf0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40094a:	e8 21 03 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  40094f:	b9 01 00 00 00       	mov    $0x1,%ecx
  400954:	89 ce                	mov    %ecx,%esi
  400956:	48 89 c7             	mov    %rax,%rdi
  400959:	e8 c2 03 00 00       	callq  400d20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  40095e:	48 89 c7             	mov    %rax,%rdi
  400961:	e8 9a 03 00 00       	callq  400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400966:	59                   	pop    %rcx
  400967:	c3                   	retq   
  400968:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40096f:	00 

0000000000400970 <_ZN9TestClassC2Ev>:
  400970:	48 83 ec 18          	sub    $0x18,%rsp
  400974:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400979:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40097e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400983:	e8 28 01 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2Ev>
  400988:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  40098d:	e8 2e 01 00 00       	callq  400ac0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2Ev>
  400992:	48 83 c4 18          	add    $0x18,%rsp
  400996:	c3                   	retq   
  400997:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40099e:	00 00 

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  4009a0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a5:	c3                   	retq   
  4009a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ad:	00 00 00 

00000000004009b0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEaSEi>:
  4009b0:	48 83 ec 28          	sub    $0x28,%rsp
  4009b4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  4009b9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  4009bd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4009c2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  4009c6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009cb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  4009cf:	e8 5c 01 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009d4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  4009d8:	89 30                	mov    %esi,(%rax)
  4009da:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4009df:	48 83 c4 28          	add    $0x28,%rsp
  4009e3:	c3                   	retq   
  4009e4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009eb:	00 00 00 
  4009ee:	66 90                	xchg   %ax,%ax

00000000004009f0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEaSEi>:
  4009f0:	48 83 ec 28          	sub    $0x28,%rsp
  4009f4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  4009f9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  4009fd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a02:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a06:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a0b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a0f:	e8 ec 01 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a14:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a18:	89 30                	mov    %esi,(%rax)
  400a1a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a1f:	48 83 c4 28          	add    $0x28,%rsp
  400a23:	c3                   	retq   
  400a24:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2b:	00 00 00 
  400a2e:	66 90                	xchg   %ax,%ax

0000000000400a30 <_ZN9TestClass15increase_field0Ev>:
  400a30:	48 83 ec 18          	sub    $0x18,%rsp
  400a34:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a39:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a3e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a43:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a48:	48 89 c7             	mov    %rax,%rdi
  400a4b:	e8 80 00 00 00       	callq  400ad0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>
  400a50:	48 83 c4 18          	add    $0x18,%rsp
  400a54:	c3                   	retq   
  400a55:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5c:	00 00 00 
  400a5f:	90                   	nop

0000000000400a60 <_ZN9TestClass15increase_field1Ev>:
  400a60:	48 83 ec 18          	sub    $0x18,%rsp
  400a64:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a69:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a6e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a73:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a78:	48 89 c7             	mov    %rax,%rdi
  400a7b:	e8 20 01 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>
  400a80:	48 83 c4 18          	add    $0x18,%rsp
  400a84:	c3                   	retq   
  400a85:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8c:	00 00 00 
  400a8f:	90                   	nop

0000000000400a90 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEcvRiEv>:
  400a90:	50                   	push   %rax
  400a91:	48 89 3c 24          	mov    %rdi,(%rsp)
  400a95:	48 8b 3c 24          	mov    (%rsp),%rdi
  400a99:	e8 92 00 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a9e:	59                   	pop    %rcx
  400a9f:	c3                   	retq   

0000000000400aa0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEcvRiEv>:
  400aa0:	50                   	push   %rax
  400aa1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400aa5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400aa9:	e8 52 01 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aae:	59                   	pop    %rcx
  400aaf:	c3                   	retq   

0000000000400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2Ev>:
  400ab0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ab5:	c3                   	retq   
  400ab6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400abd:	00 00 00 

0000000000400ac0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2Ev>:
  400ac0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ac5:	c3                   	retq   
  400ac6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400acd:	00 00 00 

0000000000400ad0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEmLEi>:
  400ad0:	48 83 ec 38          	sub    $0x38,%rsp
  400ad4:	48 89 f8             	mov    %rdi,%rax
  400ad7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400adc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400ae0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400ae5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400ae9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400aee:	48 89 f7             	mov    %rsi,%rdi
  400af1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400af6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400afa:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400aff:	e8 2c 00 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b04:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b08:	0f af 10             	imul   (%rax),%edx
  400b0b:	89 10                	mov    %edx,(%rax)
  400b0d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b12:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b17:	e8 24 00 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2ERKS3_>
  400b1c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b21:	48 83 c4 38          	add    $0x38,%rsp
  400b25:	c3                   	retq   
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b30:	50                   	push   %rax
  400b31:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b35:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b39:	e8 12 00 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400b3e:	59                   	pop    %rcx
  400b3f:	c3                   	retq   

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassEC2ERKS3_>:
  400b40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b45:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b4a:	c3                   	retq   
  400b4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400b50:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400b57:	00 00 00 
  400b5a:	48 89 c1             	mov    %rax,%rcx
  400b5d:	48 83 c1 10          	add    $0x10,%rcx
  400b61:	48 89 ca             	mov    %rcx,%rdx
  400b64:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b69:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400b6e:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400b75:	00 00 
  400b77:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400b7c:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400b81:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400b86:	48 c1 e7 02          	shl    $0x2,%rdi
  400b8a:	48 01 fa             	add    %rdi,%rdx
  400b8d:	48 89 d0             	mov    %rdx,%rax
  400b90:	c3                   	retq   
  400b91:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b98:	00 00 00 
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEmLEi>:
  400ba0:	48 83 ec 38          	sub    $0x38,%rsp
  400ba4:	48 89 f8             	mov    %rdi,%rax
  400ba7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bac:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400bb0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400bb5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400bb9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bbe:	48 89 f7             	mov    %rsi,%rdi
  400bc1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bc6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bca:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bcf:	e8 2c 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bd8:	0f af 10             	imul   (%rax),%edx
  400bdb:	89 10                	mov    %edx,(%rax)
  400bdd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400be2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400be7:	e8 24 00 00 00       	callq  400c10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2ERKS3_>
  400bec:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bf1:	48 83 c4 38          	add    $0x38,%rsp
  400bf5:	c3                   	retq   
  400bf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bfd:	00 00 00 

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c00:	50                   	push   %rax
  400c01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c09:	e8 12 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400c0e:	59                   	pop    %rcx
  400c0f:	c3                   	retq   

0000000000400c10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassEC2ERKS3_>:
  400c10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c15:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c1a:	c3                   	retq   
  400c1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400c20:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c27:	00 00 00 
  400c2a:	48 89 c1             	mov    %rax,%rcx
  400c2d:	48 81 c1 e0 48 00 00 	add    $0x48e0,%rcx
  400c34:	48 89 ca             	mov    %rcx,%rdx
  400c37:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c3c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c41:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400c48:	00 00 
  400c4a:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400c4f:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400c54:	48 89 54 24 d8       	mov    %rdx,-0x28(%rsp)
  400c59:	48 c1 e7 02          	shl    $0x2,%rdi
  400c5d:	48 01 fa             	add    %rdi,%rdx
  400c60:	48 89 d0             	mov    %rdx,%rax
  400c63:	c3                   	retq   
  400c64:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6b:	00 00 00 
  400c6e:	66 90                	xchg   %ax,%ax

0000000000400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>:
  400c70:	50                   	push   %rax
  400c71:	e8 3a 00 00 00       	callq  400cb0 <_ZN9TestClass14storage_bufferEv>
  400c76:	59                   	pop    %rcx
  400c77:	c3                   	retq   
  400c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c7f:	00 

0000000000400c80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400c80:	48 83 ec 18          	sub    $0x18,%rsp
  400c84:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400c89:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400c8e:	48 89 f8             	mov    %rdi,%rax
  400c91:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c96:	48 89 c7             	mov    %rax,%rdi
  400c99:	e8 22 00 00 00       	callq  400cc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400c9e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca3:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400caa:	00 
  400cab:	48 83 c4 18          	add    $0x18,%rsp
  400caf:	c3                   	retq   

0000000000400cb0 <_ZN9TestClass14storage_bufferEv>:
  400cb0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400cb7:	00 00 00 
  400cba:	c3                   	retq   
  400cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400cc0:	50                   	push   %rax
  400cc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cc9:	e8 02 00 00 00       	callq  400cd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400cce:	58                   	pop    %rax
  400ccf:	c3                   	retq   

0000000000400cd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400cd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cd5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400cda:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400ce1:	c3                   	retq   
  400ce2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ce9:	00 00 00 
  400cec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cf0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cf0:	c3                   	retq   
  400cf1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cf8:	00 00 00 
  400cfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400d00:	50                   	push   %rax
  400d01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d09:	48 83 c7 01          	add    $0x1,%rdi
  400d0d:	e8 3e 00 00 00       	callq  400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d12:	59                   	pop    %rcx
  400d13:	c3                   	retq   
  400d14:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1b:	00 00 00 
  400d1e:	66 90                	xchg   %ax,%ax

0000000000400d20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400d20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d2a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400d2f:	48 8b 3e             	mov    (%rsi),%rdi
  400d32:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d37:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400d3c:	48 03 3e             	add    (%rsi),%rdi
  400d3f:	48 89 3e             	mov    %rdi,(%rsi)
  400d42:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d47:	c3                   	retq   
  400d48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d4f:	00 

0000000000400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d55:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d5a:	c3                   	retq   
  400d5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d60 <__libc_csu_init>:
  400d60:	41 57                	push   %r15
  400d62:	41 56                	push   %r14
  400d64:	41 89 ff             	mov    %edi,%r15d
  400d67:	41 55                	push   %r13
  400d69:	41 54                	push   %r12
  400d6b:	4c 8d 25 6e 10 20 00 	lea    0x20106e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d72:	55                   	push   %rbp
  400d73:	48 8d 2d 6e 10 20 00 	lea    0x20106e(%rip),%rbp        # 601de8 <__init_array_end>
  400d7a:	53                   	push   %rbx
  400d7b:	49 89 f6             	mov    %rsi,%r14
  400d7e:	49 89 d5             	mov    %rdx,%r13
  400d81:	4c 29 e5             	sub    %r12,%rbp
  400d84:	48 83 ec 08          	sub    $0x8,%rsp
  400d88:	48 c1 fd 03          	sar    $0x3,%rbp
  400d8c:	e8 bf f7 ff ff       	callq  400550 <_init>
  400d91:	48 85 ed             	test   %rbp,%rbp
  400d94:	74 20                	je     400db6 <__libc_csu_init+0x56>
  400d96:	31 db                	xor    %ebx,%ebx
  400d98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d9f:	00 
  400da0:	4c 89 ea             	mov    %r13,%rdx
  400da3:	4c 89 f6             	mov    %r14,%rsi
  400da6:	44 89 ff             	mov    %r15d,%edi
  400da9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dad:	48 83 c3 01          	add    $0x1,%rbx
  400db1:	48 39 eb             	cmp    %rbp,%rbx
  400db4:	75 ea                	jne    400da0 <__libc_csu_init+0x40>
  400db6:	48 83 c4 08          	add    $0x8,%rsp
  400dba:	5b                   	pop    %rbx
  400dbb:	5d                   	pop    %rbp
  400dbc:	41 5c                	pop    %r12
  400dbe:	41 5d                	pop    %r13
  400dc0:	41 5e                	pop    %r14
  400dc2:	41 5f                	pop    %r15
  400dc4:	c3                   	retq   
  400dc5:	90                   	nop
  400dc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcd:	00 00 00 

0000000000400dd0 <__libc_csu_fini>:
  400dd0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400dd4 <_fini>:
  400dd4:	48 83 ec 08          	sub    $0x8,%rsp
  400dd8:	48 83 c4 08          	add    $0x8,%rsp
  400ddc:	c3                   	retq   
