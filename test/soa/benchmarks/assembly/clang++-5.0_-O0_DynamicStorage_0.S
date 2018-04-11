
bin/clang++-5.0_-O0_DynamicStorage_0:     file format elf64-x86-64


Disassembly of section .init:

0000000000400588 <_init>:
  400588:	48 83 ec 08          	sub    $0x8,%rsp
  40058c:	48 8b 05 65 1a 20 00 	mov    0x201a65(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  400593:	48 85 c0             	test   %rax,%rax
  400596:	74 05                	je     40059d <_init+0x15>
  400598:	e8 73 00 00 00       	callq  400610 <_Unwind_Resume@plt+0x10>
  40059d:	48 83 c4 08          	add    $0x8,%rsp
  4005a1:	c3                   	retq   

Disassembly of section .plt:

00000000004005b0 <printf@plt-0x10>:
  4005b0:	ff 35 52 1a 20 00    	pushq  0x201a52(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4005b6:	ff 25 54 1a 20 00    	jmpq   *0x201a54(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4005bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005c0 <printf@plt>:
  4005c0:	ff 25 52 1a 20 00    	jmpq   *0x201a52(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005c6:	68 00 00 00 00       	pushq  $0x0
  4005cb:	e9 e0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005d0 <malloc@plt>:
  4005d0:	ff 25 4a 1a 20 00    	jmpq   *0x201a4a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005d6:	68 01 00 00 00       	pushq  $0x1
  4005db:	e9 d0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005e0 <__libc_start_main@plt>:
  4005e0:	ff 25 42 1a 20 00    	jmpq   *0x201a42(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4005e6:	68 02 00 00 00       	pushq  $0x2
  4005eb:	e9 c0 ff ff ff       	jmpq   4005b0 <_init+0x28>

00000000004005f0 <__gxx_personality_v0@plt>:
  4005f0:	ff 25 3a 1a 20 00    	jmpq   *0x201a3a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4005f6:	68 03 00 00 00       	pushq  $0x3
  4005fb:	e9 b0 ff ff ff       	jmpq   4005b0 <_init+0x28>

0000000000400600 <_Unwind_Resume@plt>:
  400600:	ff 25 32 1a 20 00    	jmpq   *0x201a32(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400606:	68 04 00 00 00       	pushq  $0x4
  40060b:	e9 a0 ff ff ff       	jmpq   4005b0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400610 <.plt.got>:
  400610:	ff 25 e2 19 20 00    	jmpq   *0x2019e2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400616:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400620 <_start>:
  400620:	31 ed                	xor    %ebp,%ebp
  400622:	49 89 d1             	mov    %rdx,%r9
  400625:	5e                   	pop    %rsi
  400626:	48 89 e2             	mov    %rsp,%rdx
  400629:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40062d:	50                   	push   %rax
  40062e:	54                   	push   %rsp
  40062f:	49 c7 c0 30 0e 40 00 	mov    $0x400e30,%r8
  400636:	48 c7 c1 c0 0d 40 00 	mov    $0x400dc0,%rcx
  40063d:	48 c7 c7 20 07 40 00 	mov    $0x400720,%rdi
  400644:	e8 97 ff ff ff       	callq  4005e0 <__libc_start_main@plt>
  400649:	f4                   	hlt    
  40064a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400650 <deregister_tm_clones>:
  400650:	b8 57 20 60 00       	mov    $0x602057,%eax
  400655:	55                   	push   %rbp
  400656:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  40065c:	48 83 f8 0e          	cmp    $0xe,%rax
  400660:	48 89 e5             	mov    %rsp,%rbp
  400663:	76 1b                	jbe    400680 <deregister_tm_clones+0x30>
  400665:	b8 00 00 00 00       	mov    $0x0,%eax
  40066a:	48 85 c0             	test   %rax,%rax
  40066d:	74 11                	je     400680 <deregister_tm_clones+0x30>
  40066f:	5d                   	pop    %rbp
  400670:	bf 50 20 60 00       	mov    $0x602050,%edi
  400675:	ff e0                	jmpq   *%rax
  400677:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40067e:	00 00 
  400680:	5d                   	pop    %rbp
  400681:	c3                   	retq   
  400682:	0f 1f 40 00          	nopl   0x0(%rax)
  400686:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40068d:	00 00 00 

0000000000400690 <register_tm_clones>:
  400690:	be 50 20 60 00       	mov    $0x602050,%esi
  400695:	55                   	push   %rbp
  400696:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  40069d:	48 c1 fe 03          	sar    $0x3,%rsi
  4006a1:	48 89 e5             	mov    %rsp,%rbp
  4006a4:	48 89 f0             	mov    %rsi,%rax
  4006a7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006ab:	48 01 c6             	add    %rax,%rsi
  4006ae:	48 d1 fe             	sar    %rsi
  4006b1:	74 15                	je     4006c8 <register_tm_clones+0x38>
  4006b3:	b8 00 00 00 00       	mov    $0x0,%eax
  4006b8:	48 85 c0             	test   %rax,%rax
  4006bb:	74 0b                	je     4006c8 <register_tm_clones+0x38>
  4006bd:	5d                   	pop    %rbp
  4006be:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006c3:	ff e0                	jmpq   *%rax
  4006c5:	0f 1f 00             	nopl   (%rax)
  4006c8:	5d                   	pop    %rbp
  4006c9:	c3                   	retq   
  4006ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006d0 <__do_global_dtors_aux>:
  4006d0:	80 3d 79 19 20 00 00 	cmpb   $0x0,0x201979(%rip)        # 602050 <__TMC_END__>
  4006d7:	75 11                	jne    4006ea <__do_global_dtors_aux+0x1a>
  4006d9:	55                   	push   %rbp
  4006da:	48 89 e5             	mov    %rsp,%rbp
  4006dd:	e8 6e ff ff ff       	callq  400650 <deregister_tm_clones>
  4006e2:	5d                   	pop    %rbp
  4006e3:	c6 05 66 19 20 00 01 	movb   $0x1,0x201966(%rip)        # 602050 <__TMC_END__>
  4006ea:	f3 c3                	repz retq 
  4006ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006f0 <frame_dummy>:
  4006f0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  4006f5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4006f9:	75 05                	jne    400700 <frame_dummy+0x10>
  4006fb:	eb 93                	jmp    400690 <register_tm_clones>
  4006fd:	0f 1f 00             	nopl   (%rax)
  400700:	b8 00 00 00 00       	mov    $0x0,%eax
  400705:	48 85 c0             	test   %rax,%rax
  400708:	74 f1                	je     4006fb <frame_dummy+0xb>
  40070a:	55                   	push   %rbp
  40070b:	48 89 e5             	mov    %rsp,%rbp
  40070e:	ff d0                	callq  *%rax
  400710:	5d                   	pop    %rbp
  400711:	e9 7a ff ff ff       	jmpq   400690 <register_tm_clones>
  400716:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40071d:	00 00 00 

0000000000400720 <main>:
  400720:	48 83 ec 58          	sub    $0x58,%rsp
  400724:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40072b:	00 
  40072c:	e8 4f 02 00 00       	callq  400980 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400731:	31 c0                	xor    %eax,%eax
  400733:	89 c7                	mov    %eax,%edi
  400735:	e8 56 02 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40073a:	48 89 c7             	mov    %rax,%rdi
  40073d:	48 89 c1             	mov    %rax,%rcx
  400740:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400745:	48 89 c7             	mov    %rax,%rdi
  400748:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40074d:	e8 6e 02 00 00       	callq  4009c0 <_ZN9TestClassC2Ev>
  400752:	e9 00 00 00 00       	jmpq   400757 <main+0x37>
  400757:	be 77 77 00 00       	mov    $0x7777,%esi
  40075c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400761:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400766:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40076b:	e8 90 02 00 00       	callq  400a00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>
  400770:	be 88 88 00 00       	mov    $0x8888,%esi
  400775:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40077a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40077f:	e8 bc 02 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>
  400784:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400789:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40078e:	e8 ed 02 00 00       	callq  400a80 <_ZN9TestClass15increase_field0Ev>
  400793:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400798:	e8 13 03 00 00       	callq  400ab0 <_ZN9TestClass15increase_field1Ev>
  40079d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a2:	e8 39 03 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  4007a7:	8b 30                	mov    (%rax),%esi
  4007a9:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007ae:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007b2:	e8 39 03 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  4007b7:	48 bf 44 0e 40 00 00 	movabs $0x400e44,%rdi
  4007be:	00 00 00 
  4007c1:	8b 10                	mov    (%rax),%edx
  4007c3:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007c7:	b0 00                	mov    $0x0,%al
  4007c9:	e8 f2 fd ff ff       	callq  4005c0 <printf@plt>
  4007ce:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007d3:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007d7:	e8 04 03 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  4007dc:	31 d2                	xor    %edx,%edx
  4007de:	88 d1                	mov    %dl,%cl
  4007e0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007e6:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007ea:	0f 85 17 00 00 00    	jne    400807 <main+0xe7>
  4007f0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007f5:	e8 f6 02 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  4007fa:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400800:	0f 94 c1             	sete   %cl
  400803:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  400807:	8a 44 24 0f          	mov    0xf(%rsp),%al
  40080b:	34 ff                	xor    $0xff,%al
  40080d:	24 01                	and    $0x1,%al
  40080f:	0f b6 c0             	movzbl %al,%eax
  400812:	48 83 c4 58          	add    $0x58,%rsp
  400816:	c3                   	retq   
  400817:	89 d1                	mov    %edx,%ecx
  400819:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40081e:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400822:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  400827:	e8 c4 01 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  40082c:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400831:	e8 ca fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  400836:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40083d:	00 00 00 

0000000000400840 <_Z12new_instancev>:
  400840:	48 83 ec 28          	sub    $0x28,%rsp
  400844:	31 c0                	xor    %eax,%eax
  400846:	89 c7                	mov    %eax,%edi
  400848:	e8 43 01 00 00       	callq  400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40084d:	48 89 c7             	mov    %rax,%rdi
  400850:	48 89 c1             	mov    %rax,%rcx
  400853:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400858:	48 89 c7             	mov    %rax,%rdi
  40085b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400860:	e8 5b 01 00 00       	callq  4009c0 <_ZN9TestClassC2Ev>
  400865:	e9 00 00 00 00       	jmpq   40086a <_Z12new_instancev+0x2a>
  40086a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40086f:	48 83 c4 28          	add    $0x28,%rsp
  400873:	c3                   	retq   
  400874:	89 d1                	mov    %edx,%ecx
  400876:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40087b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40087f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400884:	e8 67 01 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  400889:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40088e:	e8 6d fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  400893:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40089a:	84 00 00 00 00 00 

00000000004008a0 <_Z12write_field0P9TestClass>:
  4008a0:	48 83 ec 18          	sub    $0x18,%rsp
  4008a4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008a9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ae:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008b3:	e8 48 01 00 00       	callq  400a00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>
  4008b8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008bd:	48 83 c4 18          	add    $0x18,%rsp
  4008c1:	c3                   	retq   
  4008c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c9:	1f 84 00 00 00 00 00 

00000000004008d0 <_Z11read_field0P9TestClass>:
  4008d0:	50                   	push   %rax
  4008d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008d5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008d9:	e8 02 02 00 00       	callq  400ae0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  4008de:	8b 00                	mov    (%rax),%eax
  4008e0:	59                   	pop    %rcx
  4008e1:	c3                   	retq   
  4008e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008e9:	1f 84 00 00 00 00 00 

00000000004008f0 <_Z12write_field1P9TestClass>:
  4008f0:	48 83 ec 18          	sub    $0x18,%rsp
  4008f4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008f9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008fe:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400903:	e8 38 01 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>
  400908:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40090d:	48 83 c4 18          	add    $0x18,%rsp
  400911:	c3                   	retq   
  400912:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400919:	1f 84 00 00 00 00 00 

0000000000400920 <_Z11read_field1P9TestClass>:
  400920:	50                   	push   %rax
  400921:	48 89 3c 24          	mov    %rdi,(%rsp)
  400925:	48 8b 3c 24          	mov    (%rsp),%rdi
  400929:	e8 c2 01 00 00       	callq  400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  40092e:	8b 00                	mov    (%rax),%eax
  400930:	59                   	pop    %rcx
  400931:	c3                   	retq   
  400932:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400939:	1f 84 00 00 00 00 00 

0000000000400940 <_Z21explicit_write_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  400951:	77 77 00 00 
  400955:	c3                   	retq   
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <_Z20explicit_read_field0m>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  400971:	c3                   	retq   
  400972:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400979:	00 00 00 
  40097c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400980 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  400980:	50                   	push   %rax
  400981:	e8 5a 02 00 00       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 52 03 00 00       	callq  400ce0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  40098e:	58                   	pop    %rax
  40098f:	c3                   	retq   

0000000000400990 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  400990:	50                   	push   %rax
  400991:	48 89 3c 24          	mov    %rdi,(%rsp)
  400995:	e8 b6 03 00 00       	callq  400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  40099a:	e8 41 02 00 00       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  40099f:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009a4:	89 ce                	mov    %ecx,%esi
  4009a6:	48 89 c7             	mov    %rax,%rdi
  4009a9:	e8 d2 03 00 00       	callq  400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  4009ae:	48 89 c7             	mov    %rax,%rdi
  4009b1:	e8 aa 03 00 00       	callq  400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009b6:	59                   	pop    %rcx
  4009b7:	c3                   	retq   
  4009b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009bf:	00 

00000000004009c0 <_ZN9TestClassC2Ev>:
  4009c0:	48 83 ec 18          	sub    $0x18,%rsp
  4009c4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009c9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009ce:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009d3:	e8 28 01 00 00       	callq  400b00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC2Ev>
  4009d8:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4009dd:	e8 2e 01 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC2Ev>
  4009e2:	48 83 c4 18          	add    $0x18,%rsp
  4009e6:	c3                   	retq   
  4009e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4009ee:	00 00 

00000000004009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  4009f0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009f5:	c3                   	retq   
  4009f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009fd:	00 00 00 

0000000000400a00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>:
  400a00:	48 83 ec 28          	sub    $0x28,%rsp
  400a04:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a09:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a0d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a12:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a16:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a1b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a1f:	e8 5c 01 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a24:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a28:	89 30                	mov    %esi,(%rax)
  400a2a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a2f:	48 83 c4 28          	add    $0x28,%rsp
  400a33:	c3                   	retq   
  400a34:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a3b:	00 00 00 
  400a3e:	66 90                	xchg   %ax,%ax

0000000000400a40 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>:
  400a40:	48 83 ec 28          	sub    $0x28,%rsp
  400a44:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a49:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a4d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a52:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a56:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a5b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a5f:	e8 1c 02 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a64:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a68:	89 30                	mov    %esi,(%rax)
  400a6a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a6f:	48 83 c4 28          	add    $0x28,%rsp
  400a73:	c3                   	retq   
  400a74:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a7b:	00 00 00 
  400a7e:	66 90                	xchg   %ax,%ax

0000000000400a80 <_ZN9TestClass15increase_field0Ev>:
  400a80:	48 83 ec 18          	sub    $0x18,%rsp
  400a84:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400a89:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a8e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a93:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400a98:	48 89 c7             	mov    %rax,%rdi
  400a9b:	e8 80 00 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEmLEi>
  400aa0:	48 83 c4 18          	add    $0x18,%rsp
  400aa4:	c3                   	retq   
  400aa5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aac:	00 00 00 
  400aaf:	90                   	nop

0000000000400ab0 <_ZN9TestClass15increase_field1Ev>:
  400ab0:	48 83 ec 18          	sub    $0x18,%rsp
  400ab4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ab9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400abe:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ac3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400ac8:	48 89 c7             	mov    %rax,%rdi
  400acb:	e8 50 01 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEmLEi>
  400ad0:	48 83 c4 18          	add    $0x18,%rsp
  400ad4:	c3                   	retq   
  400ad5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400adc:	00 00 00 
  400adf:	90                   	nop

0000000000400ae0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>:
  400ae0:	50                   	push   %rax
  400ae1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ae5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ae9:	e8 92 00 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aee:	59                   	pop    %rcx
  400aef:	c3                   	retq   

0000000000400af0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>:
  400af0:	50                   	push   %rax
  400af1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400af5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400af9:	e8 82 01 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400afe:	59                   	pop    %rcx
  400aff:	c3                   	retq   

0000000000400b00 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC2Ev>:
  400b00:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b05:	c3                   	retq   
  400b06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0d:	00 00 00 

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC2Ev>:
  400b10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b15:	c3                   	retq   
  400b16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b1d:	00 00 00 

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEmLEi>:
  400b20:	48 83 ec 38          	sub    $0x38,%rsp
  400b24:	48 89 f8             	mov    %rdi,%rax
  400b27:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b2c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b30:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b35:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b39:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b3e:	48 89 f7             	mov    %rsi,%rdi
  400b41:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b46:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b4a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b4f:	e8 2c 00 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b54:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b58:	0f af 10             	imul   (%rax),%edx
  400b5b:	89 10                	mov    %edx,(%rax)
  400b5d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b62:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b67:	e8 24 00 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC2ERKS3_>
  400b6c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b71:	48 83 c4 38          	add    $0x38,%rsp
  400b75:	c3                   	retq   
  400b76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7d:	00 00 00 

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b80:	50                   	push   %rax
  400b81:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b85:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b89:	e8 12 00 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400b8e:	59                   	pop    %rcx
  400b8f:	c3                   	retq   

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC2ERKS3_>:
  400b90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b95:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400b9a:	c3                   	retq   
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400ba0:	48 83 ec 18          	sub    $0x18,%rsp
  400ba4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ba9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400bae:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bb2:	e8 29 00 00 00       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400bb7:	48 89 c7             	mov    %rax,%rdi
  400bba:	e8 31 00 00 00       	callq  400bf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400bbf:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400bc4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bc9:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bcd:	48 c1 e7 02          	shl    $0x2,%rdi
  400bd1:	48 01 f8             	add    %rdi,%rax
  400bd4:	48 83 c4 18          	add    $0x18,%rsp
  400bd8:	c3                   	retq   
  400bd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>:
  400be0:	50                   	push   %rax
  400be1:	e8 1a 00 00 00       	callq  400c00 <_ZN9TestClass14storage_bufferEv>
  400be6:	59                   	pop    %rcx
  400be7:	c3                   	retq   
  400be8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400bef:	00 

0000000000400bf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400bf0:	50                   	push   %rax
  400bf1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bf5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bf9:	e8 12 00 00 00       	callq  400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400bfe:	59                   	pop    %rcx
  400bff:	c3                   	retq   

0000000000400c00 <_ZN9TestClass14storage_bufferEv>:
  400c00:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c07:	00 00 00 
  400c0a:	c3                   	retq   
  400c0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c15:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c1a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c1e:	c3                   	retq   
  400c1f:	90                   	nop

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEmLEi>:
  400c20:	48 83 ec 38          	sub    $0x38,%rsp
  400c24:	48 89 f8             	mov    %rdi,%rax
  400c27:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c2c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c30:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c35:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c39:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c3e:	48 89 f7             	mov    %rsi,%rdi
  400c41:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c46:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c4a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c4f:	e8 2c 00 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c54:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c58:	0f af 10             	imul   (%rax),%edx
  400c5b:	89 10                	mov    %edx,(%rax)
  400c5d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c62:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c67:	e8 24 00 00 00       	callq  400c90 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC2ERKS3_>
  400c6c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c71:	48 83 c4 38          	add    $0x38,%rsp
  400c75:	c3                   	retq   
  400c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7d:	00 00 00 

0000000000400c80 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c80:	50                   	push   %rax
  400c81:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c85:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c89:	e8 12 00 00 00       	callq  400ca0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400c8e:	59                   	pop    %rcx
  400c8f:	c3                   	retq   

0000000000400c90 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC2ERKS3_>:
  400c90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c95:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c9a:	c3                   	retq   
  400c9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ca0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400ca0:	48 83 ec 18          	sub    $0x18,%rsp
  400ca4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ca9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cae:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cb2:	e8 29 ff ff ff       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400cb7:	48 89 c7             	mov    %rax,%rdi
  400cba:	e8 31 ff ff ff       	callq  400bf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400cbf:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400cc4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc9:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400ccf:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cd3:	48 c1 e7 02          	shl    $0x2,%rdi
  400cd7:	48 01 f8             	add    %rdi,%rax
  400cda:	48 83 c4 18          	add    $0x18,%rsp
  400cde:	c3                   	retq   
  400cdf:	90                   	nop

0000000000400ce0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400ce0:	48 83 ec 18          	sub    $0x18,%rsp
  400ce4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ce9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cee:	48 89 f8             	mov    %rdi,%rax
  400cf1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cf6:	48 89 c7             	mov    %rax,%rdi
  400cf9:	e8 32 00 00 00       	callq  400d30 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400cfe:	b9 c0 91 00 00       	mov    $0x91c0,%ecx
  400d03:	89 cf                	mov    %ecx,%edi
  400d05:	e8 c6 f8 ff ff       	callq  4005d0 <malloc@plt>
  400d0a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400d0f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400d13:	c7 47 10 00 00 00 00 	movl   $0x0,0x10(%rdi)
  400d1a:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400d21:	00 
  400d22:	48 83 c4 18          	add    $0x18,%rsp
  400d26:	c3                   	retq   
  400d27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d2e:	00 00 

0000000000400d30 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400d30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d35:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d3a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d41:	c3                   	retq   
  400d42:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d49:	00 00 00 
  400d4c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d50:	c3                   	retq   
  400d51:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d58:	00 00 00 
  400d5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400d60:	50                   	push   %rax
  400d61:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d65:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d69:	48 83 c7 01          	add    $0x1,%rdi
  400d6d:	e8 3e 00 00 00       	callq  400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d72:	59                   	pop    %rcx
  400d73:	c3                   	retq   
  400d74:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d7b:	00 00 00 
  400d7e:	66 90                	xchg   %ax,%ax

0000000000400d80 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d85:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d8a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400d8f:	48 8b 3e             	mov    (%rsi),%rdi
  400d92:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d97:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400d9c:	48 03 3e             	add    (%rsi),%rdi
  400d9f:	48 89 3e             	mov    %rdi,(%rsi)
  400da2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400da7:	c3                   	retq   
  400da8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400daf:	00 

0000000000400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400db0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400db5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dba:	c3                   	retq   
  400dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dc0 <__libc_csu_init>:
  400dc0:	41 57                	push   %r15
  400dc2:	41 56                	push   %r14
  400dc4:	41 89 ff             	mov    %edi,%r15d
  400dc7:	41 55                	push   %r13
  400dc9:	41 54                	push   %r12
  400dcb:	4c 8d 25 0e 10 20 00 	lea    0x20100e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400dd2:	55                   	push   %rbp
  400dd3:	48 8d 2d 0e 10 20 00 	lea    0x20100e(%rip),%rbp        # 601de8 <__init_array_end>
  400dda:	53                   	push   %rbx
  400ddb:	49 89 f6             	mov    %rsi,%r14
  400dde:	49 89 d5             	mov    %rdx,%r13
  400de1:	4c 29 e5             	sub    %r12,%rbp
  400de4:	48 83 ec 08          	sub    $0x8,%rsp
  400de8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dec:	e8 97 f7 ff ff       	callq  400588 <_init>
  400df1:	48 85 ed             	test   %rbp,%rbp
  400df4:	74 20                	je     400e16 <__libc_csu_init+0x56>
  400df6:	31 db                	xor    %ebx,%ebx
  400df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dff:	00 
  400e00:	4c 89 ea             	mov    %r13,%rdx
  400e03:	4c 89 f6             	mov    %r14,%rsi
  400e06:	44 89 ff             	mov    %r15d,%edi
  400e09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e0d:	48 83 c3 01          	add    $0x1,%rbx
  400e11:	48 39 eb             	cmp    %rbp,%rbx
  400e14:	75 ea                	jne    400e00 <__libc_csu_init+0x40>
  400e16:	48 83 c4 08          	add    $0x8,%rsp
  400e1a:	5b                   	pop    %rbx
  400e1b:	5d                   	pop    %rbp
  400e1c:	41 5c                	pop    %r12
  400e1e:	41 5d                	pop    %r13
  400e20:	41 5e                	pop    %r14
  400e22:	41 5f                	pop    %r15
  400e24:	c3                   	retq   
  400e25:	90                   	nop
  400e26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2d:	00 00 00 

0000000000400e30 <__libc_csu_fini>:
  400e30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e34 <_fini>:
  400e34:	48 83 ec 08          	sub    $0x8,%rsp
  400e38:	48 83 c4 08          	add    $0x8,%rsp
  400e3c:	c3                   	retq   
