
bin/clang++-5.0_-O0_DynamicStorage_0_Soa:     file format elf64-x86-64


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
  40062f:	49 c7 c0 d0 0e 40 00 	mov    $0x400ed0,%r8
  400636:	48 c7 c1 60 0e 40 00 	mov    $0x400e60,%rcx
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
  40072c:	e8 af 02 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIJEEEvDpT_>
  400731:	31 c0                	xor    %eax,%eax
  400733:	89 c7                	mov    %eax,%edi
  400735:	e8 b6 02 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>
  40073a:	48 89 c7             	mov    %rax,%rdi
  40073d:	48 89 c1             	mov    %rax,%rcx
  400740:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400745:	48 89 c7             	mov    %rax,%rdi
  400748:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40074d:	e8 ce 02 00 00       	callq  400a20 <_ZN9TestClassC2Ev>
  400752:	e9 00 00 00 00       	jmpq   400757 <main+0x37>
  400757:	be 77 77 00 00       	mov    $0x7777,%esi
  40075c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400761:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400766:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40076b:	e8 f0 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>
  400770:	be 88 88 00 00       	mov    $0x8888,%esi
  400775:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40077a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40077f:	e8 1c 03 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>
  400784:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400789:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40078e:	e8 4d 03 00 00       	callq  400ae0 <_ZN9TestClass15increase_field0Ev>
  400793:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400798:	e8 73 03 00 00       	callq  400b10 <_ZN9TestClass15increase_field1Ev>
  40079d:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a2:	e8 99 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
  4007a7:	8b 30                	mov    (%rax),%esi
  4007a9:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007ae:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007b2:	e8 99 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
  4007b7:	48 bf e4 0e 40 00 00 	movabs $0x400ee4,%rdi
  4007be:	00 00 00 
  4007c1:	8b 10                	mov    (%rax),%edx
  4007c3:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007c7:	b0 00                	mov    $0x0,%al
  4007c9:	e8 f2 fd ff ff       	callq  4005c0 <printf@plt>
  4007ce:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007d3:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007d7:	e8 64 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
  4007dc:	31 d2                	xor    %edx,%edx
  4007de:	88 d1                	mov    %dl,%cl
  4007e0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007e6:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  4007ea:	0f 85 17 00 00 00    	jne    400807 <main+0xe7>
  4007f0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007f5:	e8 56 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
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
  400827:	e8 24 02 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEdlEPv>
  40082c:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400831:	e8 ca fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  400836:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40083d:	00 00 00 

0000000000400840 <_Z12new_instancev>:
  400840:	48 83 ec 28          	sub    $0x28,%rsp
  400844:	31 c0                	xor    %eax,%eax
  400846:	89 c7                	mov    %eax,%edi
  400848:	e8 a3 01 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>
  40084d:	48 89 c7             	mov    %rax,%rdi
  400850:	48 89 c1             	mov    %rax,%rcx
  400853:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400858:	48 89 c7             	mov    %rax,%rdi
  40085b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400860:	e8 bb 01 00 00       	callq  400a20 <_ZN9TestClassC2Ev>
  400865:	e9 00 00 00 00       	jmpq   40086a <_Z12new_instancev+0x2a>
  40086a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40086f:	48 83 c4 28          	add    $0x28,%rsp
  400873:	c3                   	retq   
  400874:	89 d1                	mov    %edx,%ecx
  400876:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40087b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40087f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400884:	e8 c7 01 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEdlEPv>
  400889:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40088e:	e8 6d fd ff ff       	callq  400600 <_Unwind_Resume@plt>
  400893:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40089a:	84 00 00 00 00 00 

00000000004008a0 <_Z12write_field0P9TestClass>:
  4008a0:	48 83 ec 18          	sub    $0x18,%rsp
  4008a4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008a9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ae:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008b3:	e8 a8 01 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>
  4008b8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008bd:	48 83 c4 18          	add    $0x18,%rsp
  4008c1:	c3                   	retq   
  4008c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c9:	1f 84 00 00 00 00 00 

00000000004008d0 <_Z11read_field0P9TestClass>:
  4008d0:	50                   	push   %rax
  4008d1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008d5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008d9:	e8 62 02 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
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
  400903:	e8 98 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>
  400908:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40090d:	48 83 c4 18          	add    $0x18,%rsp
  400911:	c3                   	retq   
  400912:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400919:	1f 84 00 00 00 00 00 

0000000000400920 <_Z11read_field1P9TestClass>:
  400920:	50                   	push   %rax
  400921:	48 89 3c 24          	mov    %rdi,(%rsp)
  400925:	48 8b 3c 24          	mov    (%rsp),%rdi
  400929:	e8 22 02 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
  40092e:	8b 00                	mov    (%rax),%eax
  400930:	59                   	pop    %rcx
  400931:	c3                   	retq   
  400932:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400939:	1f 84 00 00 00 00 00 

0000000000400940 <_Z21explicit_write_field0m>:
  400940:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400945:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40094a:	c7 04 bd 20 b2 60 00 	movl   $0x7777,0x60b220(,%rdi,4)
  400951:	77 77 00 00 
  400955:	c3                   	retq   
  400956:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40095d:	00 00 00 

0000000000400960 <_Z20explicit_read_field0m>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	8b 04 bd 20 b2 60 00 	mov    0x60b220(,%rdi,4),%eax
  400971:	c3                   	retq   
  400972:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400979:	1f 84 00 00 00 00 00 

0000000000400980 <_Z25explicit_write_field0_aosP16TestClassCompare>:
  400980:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400985:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40098a:	c7 07 77 77 00 00    	movl   $0x7777,(%rdi)
  400990:	c3                   	retq   
  400991:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400998:	0f 1f 84 00 00 00 00 
  40099f:	00 

00000000004009a0 <_Z24explicit_read_field0_aosP16TestClassCompare>:
  4009a0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009aa:	8b 07                	mov    (%rdi),%eax
  4009ac:	c3                   	retq   
  4009ad:	0f 1f 00             	nopl   (%rax)

00000000004009b0 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  4009b0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ba:	c7 47 04 77 77 00 00 	movl   $0x7777,0x4(%rdi)
  4009c1:	c3                   	retq   
  4009c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009c9:	1f 84 00 00 00 00 00 

00000000004009d0 <_Z24explicit_read_field1_aosP16TestClassCompare>:
  4009d0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009d5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009da:	8b 47 04             	mov    0x4(%rdi),%eax
  4009dd:	c3                   	retq   
  4009de:	66 90                	xchg   %ax,%ax

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIJEEEvDpT_>:
  4009e0:	50                   	push   %rax
  4009e1:	e8 5a 02 00 00       	callq  400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  4009e6:	48 89 c7             	mov    %rax,%rdi
  4009e9:	e8 52 03 00 00       	callq  400d40 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009ee:	58                   	pop    %rax
  4009ef:	c3                   	retq   

00000000004009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>:
  4009f0:	50                   	push   %rax
  4009f1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009f5:	e8 b6 03 00 00       	callq  400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009fa:	e8 41 02 00 00       	callq  400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  4009ff:	b9 01 00 00 00       	mov    $0x1,%ecx
  400a04:	89 ce                	mov    %ecx,%esi
  400a06:	48 89 c7             	mov    %rax,%rdi
  400a09:	e8 d2 03 00 00       	callq  400de0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400a0e:	48 89 c7             	mov    %rax,%rdi
  400a11:	e8 aa 03 00 00       	callq  400dc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE3getEm>
  400a16:	59                   	pop    %rcx
  400a17:	c3                   	retq   
  400a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a1f:	00 

0000000000400a20 <_ZN9TestClassC2Ev>:
  400a20:	48 83 ec 18          	sub    $0x18,%rsp
  400a24:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a29:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a2e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a33:	e8 28 01 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC2Ev>
  400a38:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400a3d:	e8 2e 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC2Ev>
  400a42:	48 83 c4 18          	add    $0x18,%rsp
  400a46:	c3                   	retq   
  400a47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a4e:	00 00 

0000000000400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEdlEPv>:
  400a50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a55:	c3                   	retq   
  400a56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5d:	00 00 00 

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>:
  400a60:	48 83 ec 28          	sub    $0x28,%rsp
  400a64:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a69:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a72:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a76:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a7b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a7f:	e8 5c 01 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400a84:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a88:	89 30                	mov    %esi,(%rax)
  400a8a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a8f:	48 83 c4 28          	add    $0x28,%rsp
  400a93:	c3                   	retq   
  400a94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a9b:	00 00 00 
  400a9e:	66 90                	xchg   %ax,%ax

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>:
  400aa0:	48 83 ec 28          	sub    $0x28,%rsp
  400aa4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400aa9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400aad:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ab2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400ab6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400abb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400abf:	e8 1c 02 00 00       	callq  400ce0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400ac4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400ac8:	89 30                	mov    %esi,(%rax)
  400aca:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400acf:	48 83 c4 28          	add    $0x28,%rsp
  400ad3:	c3                   	retq   
  400ad4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400adb:	00 00 00 
  400ade:	66 90                	xchg   %ax,%ax

0000000000400ae0 <_ZN9TestClass15increase_field0Ev>:
  400ae0:	48 83 ec 18          	sub    $0x18,%rsp
  400ae4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ae9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400aee:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400af3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400af8:	48 89 c7             	mov    %rax,%rdi
  400afb:	e8 80 00 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEmLEi>
  400b00:	48 83 c4 18          	add    $0x18,%rsp
  400b04:	c3                   	retq   
  400b05:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0c:	00 00 00 
  400b0f:	90                   	nop

0000000000400b10 <_ZN9TestClass15increase_field1Ev>:
  400b10:	48 83 ec 18          	sub    $0x18,%rsp
  400b14:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b19:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b1e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b23:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400b28:	48 89 c7             	mov    %rax,%rdi
  400b2b:	e8 50 01 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEmLEi>
  400b30:	48 83 c4 18          	add    $0x18,%rsp
  400b34:	c3                   	retq   
  400b35:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b3c:	00 00 00 
  400b3f:	90                   	nop

0000000000400b40 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>:
  400b40:	50                   	push   %rax
  400b41:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b45:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b49:	e8 92 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400b4e:	59                   	pop    %rcx
  400b4f:	c3                   	retq   

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>:
  400b50:	50                   	push   %rax
  400b51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b59:	e8 82 01 00 00       	callq  400ce0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400b5e:	59                   	pop    %rcx
  400b5f:	c3                   	retq   

0000000000400b60 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC2Ev>:
  400b60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b65:	c3                   	retq   
  400b66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b6d:	00 00 00 

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC2Ev>:
  400b70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b75:	c3                   	retq   
  400b76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7d:	00 00 00 

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEmLEi>:
  400b80:	48 83 ec 38          	sub    $0x38,%rsp
  400b84:	48 89 f8             	mov    %rdi,%rax
  400b87:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b8c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b90:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b95:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b99:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b9e:	48 89 f7             	mov    %rsi,%rdi
  400ba1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400ba6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400baa:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400baf:	e8 2c 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400bb4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bb8:	0f af 10             	imul   (%rax),%edx
  400bbb:	89 10                	mov    %edx,(%rax)
  400bbd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bc2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400bc7:	e8 24 00 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC2ERKS3_>
  400bcc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bd1:	48 83 c4 38          	add    $0x38,%rsp
  400bd5:	c3                   	retq   
  400bd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bdd:	00 00 00 

0000000000400be0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>:
  400be0:	50                   	push   %rax
  400be1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400be5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400be9:	e8 12 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400bee:	59                   	pop    %rcx
  400bef:	c3                   	retq   

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC2ERKS3_>:
  400bf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bf5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bfa:	c3                   	retq   
  400bfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400c00:	48 83 ec 18          	sub    $0x18,%rsp
  400c04:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400c09:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400c0e:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c12:	e8 29 00 00 00       	callq  400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400c17:	48 89 c7             	mov    %rax,%rdi
  400c1a:	e8 31 00 00 00       	callq  400c50 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400c1f:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c24:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c29:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c2d:	48 c1 e7 02          	shl    $0x2,%rdi
  400c31:	48 01 f8             	add    %rdi,%rax
  400c34:	48 83 c4 18          	add    $0x18,%rsp
  400c38:	c3                   	retq   
  400c39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>:
  400c40:	50                   	push   %rax
  400c41:	e8 1a 00 00 00       	callq  400c60 <_ZN9TestClass14storage_bufferEv>
  400c46:	59                   	pop    %rcx
  400c47:	c3                   	retq   
  400c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c4f:	00 

0000000000400c50 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400c50:	50                   	push   %rax
  400c51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c59:	e8 12 00 00 00       	callq  400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c5e:	59                   	pop    %rcx
  400c5f:	c3                   	retq   

0000000000400c60 <_ZN9TestClass14storage_bufferEv>:
  400c60:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c67:	00 00 00 
  400c6a:	c3                   	retq   
  400c6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c70 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c75:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c7a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c7e:	c3                   	retq   
  400c7f:	90                   	nop

0000000000400c80 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEmLEi>:
  400c80:	48 83 ec 38          	sub    $0x38,%rsp
  400c84:	48 89 f8             	mov    %rdi,%rax
  400c87:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c8c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c90:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c95:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c99:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c9e:	48 89 f7             	mov    %rsi,%rdi
  400ca1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400ca6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400caa:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400caf:	e8 2c 00 00 00       	callq  400ce0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400cb4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400cb8:	0f af 10             	imul   (%rax),%edx
  400cbb:	89 10                	mov    %edx,(%rax)
  400cbd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cc2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400cc7:	e8 24 00 00 00       	callq  400cf0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC2ERKS3_>
  400ccc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cd1:	48 83 c4 38          	add    $0x38,%rsp
  400cd5:	c3                   	retq   
  400cd6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cdd:	00 00 00 

0000000000400ce0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>:
  400ce0:	50                   	push   %rax
  400ce1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ce5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ce9:	e8 12 00 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400cee:	59                   	pop    %rcx
  400cef:	c3                   	retq   

0000000000400cf0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC2ERKS3_>:
  400cf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cf5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cfa:	c3                   	retq   
  400cfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d00:	48 83 ec 18          	sub    $0x18,%rsp
  400d04:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d09:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d0e:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d12:	e8 29 ff ff ff       	callq  400c40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400d17:	48 89 c7             	mov    %rax,%rdi
  400d1a:	e8 31 ff ff ff       	callq  400c50 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400d1f:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d24:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d29:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400d2f:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d33:	48 c1 e7 02          	shl    $0x2,%rdi
  400d37:	48 01 f8             	add    %rdi,%rax
  400d3a:	48 83 c4 18          	add    $0x18,%rsp
  400d3e:	c3                   	retq   
  400d3f:	90                   	nop

0000000000400d40 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d40:	48 83 ec 18          	sub    $0x18,%rsp
  400d44:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d49:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d4e:	48 89 f8             	mov    %rdi,%rax
  400d51:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d56:	48 89 c7             	mov    %rax,%rdi
  400d59:	e8 32 00 00 00       	callq  400d90 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d5e:	b9 c0 91 00 00       	mov    $0x91c0,%ecx
  400d63:	89 cf                	mov    %ecx,%edi
  400d65:	e8 66 f8 ff ff       	callq  4005d0 <malloc@plt>
  400d6a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400d6f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400d73:	c7 47 10 00 00 00 00 	movl   $0x0,0x10(%rdi)
  400d7a:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400d81:	00 
  400d82:	48 83 c4 18          	add    $0x18,%rsp
  400d86:	c3                   	retq   
  400d87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d8e:	00 00 

0000000000400d90 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400d90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d95:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d9a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400da1:	c3                   	retq   
  400da2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400da9:	00 00 00 
  400dac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400db0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400db0:	c3                   	retq   
  400db1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400db8:	00 00 00 
  400dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE3getEm>:
  400dc0:	50                   	push   %rax
  400dc1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400dc5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400dc9:	48 83 c7 01          	add    $0x1,%rdi
  400dcd:	e8 4e 00 00 00       	callq  400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE4get_ILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPS2_E4typeEm>
  400dd2:	59                   	pop    %rcx
  400dd3:	c3                   	retq   
  400dd4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddb:	00 00 00 
  400dde:	66 90                	xchg   %ax,%ax

0000000000400de0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400de0:	48 83 ec 28          	sub    $0x28,%rsp
  400de4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400de9:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  400dee:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  400df3:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400df8:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  400dfd:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400e02:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e07:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400e0c:	e8 1f 00 00 00       	callq  400e30 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400e11:	48 83 c4 28          	add    $0x28,%rsp
  400e15:	c3                   	retq   
  400e16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e1d:	00 00 00 

0000000000400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE4get_ILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPS2_E4typeEm>:
  400e20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e25:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e2a:	c3                   	retq   
  400e2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e30 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400e30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400e3a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400e3f:	48 8b 36             	mov    (%rsi),%rsi
  400e42:	48 89 74 24 e8       	mov    %rsi,-0x18(%rsp)
  400e47:	48 8b 74 24 f0       	mov    -0x10(%rsp),%rsi
  400e4c:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400e51:	48 03 37             	add    (%rdi),%rsi
  400e54:	48 89 37             	mov    %rsi,(%rdi)
  400e57:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e5c:	c3                   	retq   
  400e5d:	0f 1f 00             	nopl   (%rax)

0000000000400e60 <__libc_csu_init>:
  400e60:	41 57                	push   %r15
  400e62:	41 56                	push   %r14
  400e64:	41 89 ff             	mov    %edi,%r15d
  400e67:	41 55                	push   %r13
  400e69:	41 54                	push   %r12
  400e6b:	4c 8d 25 6e 0f 20 00 	lea    0x200f6e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e72:	55                   	push   %rbp
  400e73:	48 8d 2d 6e 0f 20 00 	lea    0x200f6e(%rip),%rbp        # 601de8 <__init_array_end>
  400e7a:	53                   	push   %rbx
  400e7b:	49 89 f6             	mov    %rsi,%r14
  400e7e:	49 89 d5             	mov    %rdx,%r13
  400e81:	4c 29 e5             	sub    %r12,%rbp
  400e84:	48 83 ec 08          	sub    $0x8,%rsp
  400e88:	48 c1 fd 03          	sar    $0x3,%rbp
  400e8c:	e8 f7 f6 ff ff       	callq  400588 <_init>
  400e91:	48 85 ed             	test   %rbp,%rbp
  400e94:	74 20                	je     400eb6 <__libc_csu_init+0x56>
  400e96:	31 db                	xor    %ebx,%ebx
  400e98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e9f:	00 
  400ea0:	4c 89 ea             	mov    %r13,%rdx
  400ea3:	4c 89 f6             	mov    %r14,%rsi
  400ea6:	44 89 ff             	mov    %r15d,%edi
  400ea9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ead:	48 83 c3 01          	add    $0x1,%rbx
  400eb1:	48 39 eb             	cmp    %rbp,%rbx
  400eb4:	75 ea                	jne    400ea0 <__libc_csu_init+0x40>
  400eb6:	48 83 c4 08          	add    $0x8,%rsp
  400eba:	5b                   	pop    %rbx
  400ebb:	5d                   	pop    %rbp
  400ebc:	41 5c                	pop    %r12
  400ebe:	41 5d                	pop    %r13
  400ec0:	41 5e                	pop    %r14
  400ec2:	41 5f                	pop    %r15
  400ec4:	c3                   	retq   
  400ec5:	90                   	nop
  400ec6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ecd:	00 00 00 

0000000000400ed0 <__libc_csu_fini>:
  400ed0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ed4 <_fini>:
  400ed4:	48 83 ec 08          	sub    $0x8,%rsp
  400ed8:	48 83 c4 08          	add    $0x8,%rsp
  400edc:	c3                   	retq   
