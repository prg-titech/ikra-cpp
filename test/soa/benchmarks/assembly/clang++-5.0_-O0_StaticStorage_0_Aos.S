
bin/clang++-5.0_-O0_StaticStorage_0_Aos:     file format elf64-x86-64


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
  4005df:	49 c7 c0 20 0e 40 00 	mov    $0x400e20,%r8
  4005e6:	48 c7 c1 b0 0d 40 00 	mov    $0x400db0,%rcx
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
  4006dc:	e8 af 02 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIJEEEvDpT_>
  4006e1:	31 c0                	xor    %eax,%eax
  4006e3:	89 c7                	mov    %eax,%edi
  4006e5:	e8 b6 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>
  4006ea:	48 89 c7             	mov    %rax,%rdi
  4006ed:	48 89 c1             	mov    %rax,%rcx
  4006f0:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  4006f5:	48 89 c7             	mov    %rax,%rdi
  4006f8:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  4006fd:	e8 ce 02 00 00       	callq  4009d0 <_ZN9TestClassC2Ev>
  400702:	e9 00 00 00 00       	jmpq   400707 <main+0x37>
  400707:	be 77 77 00 00       	mov    $0x7777,%esi
  40070c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400711:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400716:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40071b:	e8 f0 02 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>
  400720:	be 88 88 00 00       	mov    $0x8888,%esi
  400725:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40072a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40072f:	e8 1c 03 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>
  400734:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400739:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40073e:	e8 4d 03 00 00       	callq  400a90 <_ZN9TestClass15increase_field0Ev>
  400743:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400748:	e8 73 03 00 00       	callq  400ac0 <_ZN9TestClass15increase_field1Ev>
  40074d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400752:	e8 99 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
  400757:	8b 30                	mov    (%rax),%esi
  400759:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40075e:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400762:	e8 99 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
  400767:	48 bf 34 0e 40 00 00 	movabs $0x400e34,%rdi
  40076e:	00 00 00 
  400771:	8b 10                	mov    (%rax),%edx
  400773:	8b 74 24 14          	mov    0x14(%rsp),%esi
  400777:	b0 00                	mov    $0x0,%al
  400779:	e8 02 fe ff ff       	callq  400580 <printf@plt>
  40077e:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400783:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400787:	e8 64 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
  40078c:	31 d2                	xor    %edx,%edx
  40078e:	88 d1                	mov    %dl,%cl
  400790:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400796:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40079a:	0f 85 17 00 00 00    	jne    4007b7 <main+0xe7>
  4007a0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a5:	e8 56 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
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
  4007d7:	e8 24 02 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEdlEPv>
  4007dc:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  4007e1:	e8 ca fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  4007e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4007ed:	00 00 00 

00000000004007f0 <_Z12new_instancev>:
  4007f0:	48 83 ec 28          	sub    $0x28,%rsp
  4007f4:	31 c0                	xor    %eax,%eax
  4007f6:	89 c7                	mov    %eax,%edi
  4007f8:	e8 a3 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>
  4007fd:	48 89 c7             	mov    %rax,%rdi
  400800:	48 89 c1             	mov    %rax,%rcx
  400803:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400808:	48 89 c7             	mov    %rax,%rdi
  40080b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400810:	e8 bb 01 00 00       	callq  4009d0 <_ZN9TestClassC2Ev>
  400815:	e9 00 00 00 00       	jmpq   40081a <_Z12new_instancev+0x2a>
  40081a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081f:	48 83 c4 28          	add    $0x28,%rsp
  400823:	c3                   	retq   
  400824:	89 d1                	mov    %edx,%ecx
  400826:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40082b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40082f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400834:	e8 c7 01 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEdlEPv>
  400839:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40083e:	e8 6d fd ff ff       	callq  4005b0 <_Unwind_Resume@plt>
  400843:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40084a:	84 00 00 00 00 00 

0000000000400850 <_Z12write_field0P9TestClass>:
  400850:	48 83 ec 18          	sub    $0x18,%rsp
  400854:	be 77 77 00 00       	mov    $0x7777,%esi
  400859:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40085e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400863:	e8 a8 01 00 00       	callq  400a10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>
  400868:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40086d:	48 83 c4 18          	add    $0x18,%rsp
  400871:	c3                   	retq   
  400872:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400879:	1f 84 00 00 00 00 00 

0000000000400880 <_Z11read_field0P9TestClass>:
  400880:	50                   	push   %rax
  400881:	48 89 3c 24          	mov    %rdi,(%rsp)
  400885:	48 8b 3c 24          	mov    (%rsp),%rdi
  400889:	e8 62 02 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
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
  4008b3:	e8 98 01 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>
  4008b8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008bd:	48 83 c4 18          	add    $0x18,%rsp
  4008c1:	c3                   	retq   
  4008c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c9:	1f 84 00 00 00 00 00 

00000000004008d0 <_Z11read_field1P9TestClass>:
  4008d0:	50                   	push   %rax
  4008d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008d5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008d9:	e8 22 02 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
  4008de:	8b 00                	mov    (%rax),%eax
  4008e0:	59                   	pop    %rcx
  4008e1:	c3                   	retq   
  4008e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008e9:	1f 84 00 00 00 00 00 

00000000004008f0 <_Z21explicit_write_field0m>:
  4008f0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008f5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4008fa:	c7 04 bd d0 43 61 00 	movl   $0x7777,0x6143d0(,%rdi,4)
  400901:	77 77 00 00 
  400905:	c3                   	retq   
  400906:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40090d:	00 00 00 

0000000000400910 <_Z20explicit_read_field0m>:
  400910:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400915:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40091a:	8b 04 bd d0 43 61 00 	mov    0x6143d0(,%rdi,4),%eax
  400921:	c3                   	retq   
  400922:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400929:	1f 84 00 00 00 00 00 

0000000000400930 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400930:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400935:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40093a:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400940:	c3                   	retq   
  400941:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400948:	0f 1f 84 00 00 00 00 
  40094f:	00 

0000000000400950 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  400950:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400955:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40095a:	8b 07                	mov    (%rdi),%eax
  40095c:	c3                   	retq   
  40095d:	0f 1f 00             	nopl   (%rax)

0000000000400960 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  400971:	c3                   	retq   
  400972:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400979:	1f 84 00 00 00 00 00 

0000000000400980 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  400980:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400985:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40098a:	8b 47 04             	mov    0x4(%rdi),%eax
  40098d:	c3                   	retq   
  40098e:	66 90                	xchg   %ax,%ax

0000000000400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIJEEEvDpT_>:
  400990:	50                   	push   %rax
  400991:	e8 ca 02 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>
  400996:	48 89 c7             	mov    %rax,%rdi
  400999:	e8 d2 02 00 00       	callq  400c70 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  40099e:	58                   	pop    %rax
  40099f:	c3                   	retq   

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>:
  4009a0:	50                   	push   %rax
  4009a1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009a5:	e8 26 03 00 00       	callq  400cd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009aa:	e8 b1 02 00 00       	callq  400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>
  4009af:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009b4:	89 ce                	mov    %ecx,%esi
  4009b6:	48 89 c7             	mov    %rax,%rdi
  4009b9:	e8 42 03 00 00       	callq  400d00 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  4009be:	48 89 c7             	mov    %rax,%rdi
  4009c1:	e8 1a 03 00 00       	callq  400ce0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE3getEm>
  4009c6:	59                   	pop    %rcx
  4009c7:	c3                   	retq   
  4009c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009cf:	00 

00000000004009d0 <_ZN9TestClassC2Ev>:
  4009d0:	48 83 ec 18          	sub    $0x18,%rsp
  4009d4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009d9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009de:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009e3:	e8 28 01 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC2Ev>
  4009e8:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4009ed:	e8 2e 01 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC2Ev>
  4009f2:	48 83 c4 18          	add    $0x18,%rsp
  4009f6:	c3                   	retq   
  4009f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009fe:	00 00 

0000000000400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEdlEPv>:
  400a00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a05:	c3                   	retq   
  400a06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a0d:	00 00 00 

0000000000400a10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>:
  400a10:	48 83 ec 28          	sub    $0x28,%rsp
  400a14:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a19:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a1d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a22:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a26:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a2b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a2f:	e8 5c 01 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400a34:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a38:	89 30                	mov    %esi,(%rax)
  400a3a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a3f:	48 83 c4 28          	add    $0x28,%rsp
  400a43:	c3                   	retq   
  400a44:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4b:	00 00 00 
  400a4e:	66 90                	xchg   %ax,%ax

0000000000400a50 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>:
  400a50:	48 83 ec 28          	sub    $0x28,%rsp
  400a54:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a59:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a5d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a62:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a66:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a6b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a6f:	e8 ac 01 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400a74:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a78:	89 30                	mov    %esi,(%rax)
  400a7a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a7f:	48 83 c4 28          	add    $0x28,%rsp
  400a83:	c3                   	retq   
  400a84:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a8b:	00 00 00 
  400a8e:	66 90                	xchg   %ax,%ax

0000000000400a90 <_ZN9TestClass15increase_field0Ev>:
  400a90:	48 83 ec 18          	sub    $0x18,%rsp
  400a94:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a99:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a9e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400aa3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400aa8:	48 89 c7             	mov    %rax,%rdi
  400aab:	e8 80 00 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEmLEi>
  400ab0:	48 83 c4 18          	add    $0x18,%rsp
  400ab4:	c3                   	retq   
  400ab5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400abc:	00 00 00 
  400abf:	90                   	nop

0000000000400ac0 <_ZN9TestClass15increase_field1Ev>:
  400ac0:	48 83 ec 18          	sub    $0x18,%rsp
  400ac4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ac9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ace:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ad3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400ad8:	48 89 c7             	mov    %rax,%rdi
  400adb:	e8 e0 00 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEmLEi>
  400ae0:	48 83 c4 18          	add    $0x18,%rsp
  400ae4:	c3                   	retq   
  400ae5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aec:	00 00 00 
  400aef:	90                   	nop

0000000000400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>:
  400af0:	50                   	push   %rax
  400af1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400af5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400af9:	e8 92 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400afe:	59                   	pop    %rcx
  400aff:	c3                   	retq   

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>:
  400b00:	50                   	push   %rax
  400b01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b09:	e8 12 01 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400b0e:	59                   	pop    %rcx
  400b0f:	c3                   	retq   

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC2Ev>:
  400b10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b15:	c3                   	retq   
  400b16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b1d:	00 00 00 

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC2Ev>:
  400b20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b25:	c3                   	retq   
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEmLEi>:
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
  400b5f:	e8 2c 00 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400b64:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b68:	0f af 10             	imul   (%rax),%edx
  400b6b:	89 10                	mov    %edx,(%rax)
  400b6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b72:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b77:	e8 24 00 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC2ERKS3_>
  400b7c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b81:	48 83 c4 38          	add    $0x38,%rsp
  400b85:	c3                   	retq   
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 

0000000000400b90 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>:
  400b90:	50                   	push   %rax
  400b91:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b95:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b99:	e8 12 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>
  400b9e:	59                   	pop    %rcx
  400b9f:	c3                   	retq   

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC2ERKS3_>:
  400ba0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ba5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400baa:	c3                   	retq   
  400bab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>:
  400bb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bb5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400bba:	48 89 f8             	mov    %rdi,%rax
  400bbd:	c3                   	retq   
  400bbe:	66 90                	xchg   %ax,%ax

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEmLEi>:
  400bc0:	48 83 ec 38          	sub    $0x38,%rsp
  400bc4:	48 89 f8             	mov    %rdi,%rax
  400bc7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bcc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400bd0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400bd5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400bd9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bde:	48 89 f7             	mov    %rsi,%rdi
  400be1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400be6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bea:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bef:	e8 2c 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400bf4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bf8:	0f af 10             	imul   (%rax),%edx
  400bfb:	89 10                	mov    %edx,(%rax)
  400bfd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c02:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c07:	e8 24 00 00 00       	callq  400c30 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC2ERKS3_>
  400c0c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c11:	48 83 c4 38          	add    $0x38,%rsp
  400c15:	c3                   	retq   
  400c16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c1d:	00 00 00 

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>:
  400c20:	50                   	push   %rax
  400c21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c29:	e8 12 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>
  400c2e:	59                   	pop    %rcx
  400c2f:	c3                   	retq   

0000000000400c30 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC2ERKS3_>:
  400c30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c3a:	c3                   	retq   
  400c3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c40 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>:
  400c40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c45:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c4a:	48 83 c7 04          	add    $0x4,%rdi
  400c4e:	48 89 f8             	mov    %rdi,%rax
  400c51:	c3                   	retq   
  400c52:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c59:	00 00 00 
  400c5c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>:
  400c60:	50                   	push   %rax
  400c61:	e8 3a 00 00 00       	callq  400ca0 <_ZN9TestClass14storage_bufferEv>
  400c66:	59                   	pop    %rcx
  400c67:	c3                   	retq   
  400c68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c6f:	00 

0000000000400c70 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400c70:	48 83 ec 18          	sub    $0x18,%rsp
  400c74:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400c79:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400c7e:	48 89 f8             	mov    %rdi,%rax
  400c81:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c86:	48 89 c7             	mov    %rax,%rdi
  400c89:	e8 22 00 00 00       	callq  400cb0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>
  400c8e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c93:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400c9a:	48 83 c4 18          	add    $0x18,%rsp
  400c9e:	c3                   	retq   
  400c9f:	90                   	nop

0000000000400ca0 <_ZN9TestClass14storage_bufferEv>:
  400ca0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400ca7:	00 00 00 
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC2Ev>:
  400cb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cb5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400cba:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400cc1:	c3                   	retq   
  400cc2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cc9:	00 00 00 
  400ccc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cd0:	c3                   	retq   
  400cd1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cd8:	00 00 00 
  400cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ce0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE3getEm>:
  400ce0:	50                   	push   %rax
  400ce1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ce5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ce9:	48 83 c7 01          	add    $0x1,%rdi
  400ced:	e8 4e 00 00 00       	callq  400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE4get_ILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeAosEEEPS2_E4typeEm>
  400cf2:	59                   	pop    %rcx
  400cf3:	c3                   	retq   
  400cf4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfb:	00 00 00 
  400cfe:	66 90                	xchg   %ax,%ax

0000000000400d00 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d00:	48 83 ec 28          	sub    $0x28,%rsp
  400d04:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d09:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  400d0e:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  400d13:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400d18:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  400d1d:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400d22:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d27:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400d2c:	e8 4f 00 00 00       	callq  400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400d31:	48 83 c4 28          	add    $0x28,%rsp
  400d35:	c3                   	retq   
  400d36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3d:	00 00 00 

0000000000400d40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE4get_ILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeAosEEEPS2_E4typeEm>:
  400d40:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400d47:	00 00 00 
  400d4a:	48 89 c1             	mov    %rax,%rcx
  400d4d:	48 83 c1 10          	add    $0x10,%rcx
  400d51:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d56:	48 c7 44 24 f0 10 00 	movq   $0x10,-0x10(%rsp)
  400d5d:	00 00 
  400d5f:	48 89 44 24 e8       	mov    %rax,-0x18(%rsp)
  400d64:	48 89 4c 24 e0       	mov    %rcx,-0x20(%rsp)
  400d69:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400d6e:	48 8b 4c 24 f8       	mov    -0x8(%rsp),%rcx
  400d73:	48 c1 e1 03          	shl    $0x3,%rcx
  400d77:	48 01 c8             	add    %rcx,%rax
  400d7a:	c3                   	retq   
  400d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400d80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d85:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d8a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400d8f:	48 8b 36             	mov    (%rsi),%rsi
  400d92:	48 89 74 24 e8       	mov    %rsi,-0x18(%rsp)
  400d97:	48 8b 74 24 f0       	mov    -0x10(%rsp),%rsi
  400d9c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400da1:	48 03 37             	add    (%rdi),%rsi
  400da4:	48 89 37             	mov    %rsi,(%rdi)
  400da7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400dac:	c3                   	retq   
  400dad:	0f 1f 00             	nopl   (%rax)

0000000000400db0 <__libc_csu_init>:
  400db0:	41 57                	push   %r15
  400db2:	41 56                	push   %r14
  400db4:	41 89 ff             	mov    %edi,%r15d
  400db7:	41 55                	push   %r13
  400db9:	41 54                	push   %r12
  400dbb:	4c 8d 25 1e 10 20 00 	lea    0x20101e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400dc2:	55                   	push   %rbp
  400dc3:	48 8d 2d 1e 10 20 00 	lea    0x20101e(%rip),%rbp        # 601de8 <__init_array_end>
  400dca:	53                   	push   %rbx
  400dcb:	49 89 f6             	mov    %rsi,%r14
  400dce:	49 89 d5             	mov    %rdx,%r13
  400dd1:	4c 29 e5             	sub    %r12,%rbp
  400dd4:	48 83 ec 08          	sub    $0x8,%rsp
  400dd8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ddc:	e8 6f f7 ff ff       	callq  400550 <_init>
  400de1:	48 85 ed             	test   %rbp,%rbp
  400de4:	74 20                	je     400e06 <__libc_csu_init+0x56>
  400de6:	31 db                	xor    %ebx,%ebx
  400de8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400def:	00 
  400df0:	4c 89 ea             	mov    %r13,%rdx
  400df3:	4c 89 f6             	mov    %r14,%rsi
  400df6:	44 89 ff             	mov    %r15d,%edi
  400df9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dfd:	48 83 c3 01          	add    $0x1,%rbx
  400e01:	48 39 eb             	cmp    %rbp,%rbx
  400e04:	75 ea                	jne    400df0 <__libc_csu_init+0x40>
  400e06:	48 83 c4 08          	add    $0x8,%rsp
  400e0a:	5b                   	pop    %rbx
  400e0b:	5d                   	pop    %rbp
  400e0c:	41 5c                	pop    %r12
  400e0e:	41 5d                	pop    %r13
  400e10:	41 5e                	pop    %r14
  400e12:	41 5f                	pop    %r15
  400e14:	c3                   	retq   
  400e15:	90                   	nop
  400e16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e1d:	00 00 00 

0000000000400e20 <__libc_csu_fini>:
  400e20:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e24 <_fini>:
  400e24:	48 83 ec 08          	sub    $0x8,%rsp
  400e28:	48 83 c4 08          	add    $0x8,%rsp
  400e2c:	c3                   	retq   
