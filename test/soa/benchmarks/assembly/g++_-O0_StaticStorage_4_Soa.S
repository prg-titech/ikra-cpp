
bin/g++_-O0_StaticStorage_4_Soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004005c0 <_init>:
  4005c0:	48 83 ec 08          	sub    $0x8,%rsp
  4005c4:	48 8b 05 2d 1a 20 00 	mov    0x201a2d(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  4005cb:	48 85 c0             	test   %rax,%rax
  4005ce:	74 05                	je     4005d5 <_init+0x15>
  4005d0:	e8 6b 00 00 00       	callq  400640 <_Unwind_Resume@plt+0x10>
  4005d5:	48 83 c4 08          	add    $0x8,%rsp
  4005d9:	c3                   	retq   

Disassembly of section .plt:

00000000004005e0 <printf@plt-0x10>:
  4005e0:	ff 35 22 1a 20 00    	pushq  0x201a22(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4005e6:	ff 25 24 1a 20 00    	jmpq   *0x201a24(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4005ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005f0 <printf@plt>:
  4005f0:	ff 25 22 1a 20 00    	jmpq   *0x201a22(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005f6:	68 00 00 00 00       	pushq  $0x0
  4005fb:	e9 e0 ff ff ff       	jmpq   4005e0 <_init+0x20>

0000000000400600 <__libc_start_main@plt>:
  400600:	ff 25 1a 1a 20 00    	jmpq   *0x201a1a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400606:	68 01 00 00 00       	pushq  $0x1
  40060b:	e9 d0 ff ff ff       	jmpq   4005e0 <_init+0x20>

0000000000400610 <__stack_chk_fail@plt>:
  400610:	ff 25 12 1a 20 00    	jmpq   *0x201a12(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400616:	68 02 00 00 00       	pushq  $0x2
  40061b:	e9 c0 ff ff ff       	jmpq   4005e0 <_init+0x20>

0000000000400620 <__gxx_personality_v0@plt>:
  400620:	ff 25 0a 1a 20 00    	jmpq   *0x201a0a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400626:	68 03 00 00 00       	pushq  $0x3
  40062b:	e9 b0 ff ff ff       	jmpq   4005e0 <_init+0x20>

0000000000400630 <_Unwind_Resume@plt>:
  400630:	ff 25 02 1a 20 00    	jmpq   *0x201a02(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400636:	68 04 00 00 00       	pushq  $0x4
  40063b:	e9 a0 ff ff ff       	jmpq   4005e0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400640 <.plt.got>:
  400640:	ff 25 b2 19 20 00    	jmpq   *0x2019b2(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  400646:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400650 <_start>:
  400650:	31 ed                	xor    %ebp,%ebp
  400652:	49 89 d1             	mov    %rdx,%r9
  400655:	5e                   	pop    %rsi
  400656:	48 89 e2             	mov    %rsp,%rdx
  400659:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40065d:	50                   	push   %rax
  40065e:	54                   	push   %rsp
  40065f:	49 c7 c0 00 0f 40 00 	mov    $0x400f00,%r8
  400666:	48 c7 c1 90 0e 40 00 	mov    $0x400e90,%rcx
  40066d:	48 c7 c7 46 07 40 00 	mov    $0x400746,%rdi
  400674:	e8 87 ff ff ff       	callq  400600 <__libc_start_main@plt>
  400679:	f4                   	hlt    
  40067a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400680 <deregister_tm_clones>:
  400680:	b8 57 20 60 00       	mov    $0x602057,%eax
  400685:	55                   	push   %rbp
  400686:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  40068c:	48 83 f8 0e          	cmp    $0xe,%rax
  400690:	48 89 e5             	mov    %rsp,%rbp
  400693:	76 1b                	jbe    4006b0 <deregister_tm_clones+0x30>
  400695:	b8 00 00 00 00       	mov    $0x0,%eax
  40069a:	48 85 c0             	test   %rax,%rax
  40069d:	74 11                	je     4006b0 <deregister_tm_clones+0x30>
  40069f:	5d                   	pop    %rbp
  4006a0:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006a5:	ff e0                	jmpq   *%rax
  4006a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006ae:	00 00 
  4006b0:	5d                   	pop    %rbp
  4006b1:	c3                   	retq   
  4006b2:	0f 1f 40 00          	nopl   0x0(%rax)
  4006b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006bd:	00 00 00 

00000000004006c0 <register_tm_clones>:
  4006c0:	be 50 20 60 00       	mov    $0x602050,%esi
  4006c5:	55                   	push   %rbp
  4006c6:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  4006cd:	48 c1 fe 03          	sar    $0x3,%rsi
  4006d1:	48 89 e5             	mov    %rsp,%rbp
  4006d4:	48 89 f0             	mov    %rsi,%rax
  4006d7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006db:	48 01 c6             	add    %rax,%rsi
  4006de:	48 d1 fe             	sar    %rsi
  4006e1:	74 15                	je     4006f8 <register_tm_clones+0x38>
  4006e3:	b8 00 00 00 00       	mov    $0x0,%eax
  4006e8:	48 85 c0             	test   %rax,%rax
  4006eb:	74 0b                	je     4006f8 <register_tm_clones+0x38>
  4006ed:	5d                   	pop    %rbp
  4006ee:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006f3:	ff e0                	jmpq   *%rax
  4006f5:	0f 1f 00             	nopl   (%rax)
  4006f8:	5d                   	pop    %rbp
  4006f9:	c3                   	retq   
  4006fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400700 <__do_global_dtors_aux>:
  400700:	80 3d 59 19 20 00 00 	cmpb   $0x0,0x201959(%rip)        # 602060 <completed.7594>
  400707:	75 11                	jne    40071a <__do_global_dtors_aux+0x1a>
  400709:	55                   	push   %rbp
  40070a:	48 89 e5             	mov    %rsp,%rbp
  40070d:	e8 6e ff ff ff       	callq  400680 <deregister_tm_clones>
  400712:	5d                   	pop    %rbp
  400713:	c6 05 46 19 20 00 01 	movb   $0x1,0x201946(%rip)        # 602060 <completed.7594>
  40071a:	f3 c3                	repz retq 
  40071c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400720 <frame_dummy>:
  400720:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400725:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400729:	75 05                	jne    400730 <frame_dummy+0x10>
  40072b:	eb 93                	jmp    4006c0 <register_tm_clones>
  40072d:	0f 1f 00             	nopl   (%rax)
  400730:	b8 00 00 00 00       	mov    $0x0,%eax
  400735:	48 85 c0             	test   %rax,%rax
  400738:	74 f1                	je     40072b <frame_dummy+0xb>
  40073a:	55                   	push   %rbp
  40073b:	48 89 e5             	mov    %rsp,%rbp
  40073e:	ff d0                	callq  *%rax
  400740:	5d                   	pop    %rbp
  400741:	e9 7a ff ff ff       	jmpq   4006c0 <register_tm_clones>

0000000000400746 <main>:
  400746:	53                   	push   %rbx
  400747:	48 83 ec 10          	sub    $0x10,%rsp
  40074b:	e8 0d 04 00 00       	callq  400b5d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIIEEEvDpT_>
  400750:	bf 04 00 00 00       	mov    $0x4,%edi
  400755:	e8 4d 04 00 00       	callq  400ba7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>
  40075a:	48 89 c3             	mov    %rax,%rbx
  40075d:	b8 00 00 00 00       	mov    $0x0,%eax
  400762:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400766:	48 83 f8 03          	cmp    $0x3,%rax
  40076a:	74 06                	je     400772 <main+0x2c>
  40076c:	48 83 c0 01          	add    $0x1,%rax
  400770:	eb f0                	jmp    400762 <main+0x1c>
  400772:	48 89 df             	mov    %rbx,%rdi
  400775:	e8 08 03 00 00       	callq  400a82 <_ZN9TestClassC1Ev>
  40077a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40077f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400784:	48 83 c0 04          	add    $0x4,%rax
  400788:	be 77 77 00 00       	mov    $0x7777,%esi
  40078d:	48 89 c7             	mov    %rax,%rdi
  400790:	e8 55 04 00 00       	callq  400bea <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>
  400795:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40079a:	48 83 c0 04          	add    $0x4,%rax
  40079e:	be 88 88 00 00       	mov    $0x8888,%esi
  4007a3:	48 89 c7             	mov    %rax,%rdi
  4007a6:	e8 6d 04 00 00       	callq  400c18 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>
  4007ab:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007b0:	48 89 c7             	mov    %rax,%rdi
  4007b3:	e8 d0 01 00 00       	callq  400988 <_ZN9TestClass15increase_field0Ev>
  4007b8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bd:	48 89 c7             	mov    %rax,%rdi
  4007c0:	e8 2b 02 00 00       	callq  4009f0 <_ZN9TestClass15increase_field1Ev>
  4007c5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ca:	48 83 c0 04          	add    $0x4,%rax
  4007ce:	48 89 c7             	mov    %rax,%rdi
  4007d1:	e8 8c 04 00 00       	callq  400c62 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
  4007d6:	8b 18                	mov    (%rax),%ebx
  4007d8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007dd:	48 83 c0 04          	add    $0x4,%rax
  4007e1:	48 89 c7             	mov    %rax,%rdi
  4007e4:	e8 5d 04 00 00       	callq  400c46 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
  4007e9:	8b 00                	mov    (%rax),%eax
  4007eb:	89 da                	mov    %ebx,%edx
  4007ed:	89 c6                	mov    %eax,%esi
  4007ef:	bf 34 0f 40 00       	mov    $0x400f34,%edi
  4007f4:	b8 00 00 00 00       	mov    $0x0,%eax
  4007f9:	e8 f2 fd ff ff       	callq  4005f0 <printf@plt>
  4007fe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400803:	48 83 c0 04          	add    $0x4,%rax
  400807:	48 89 c7             	mov    %rax,%rdi
  40080a:	e8 37 04 00 00       	callq  400c46 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
  40080f:	8b 00                	mov    (%rax),%eax
  400811:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400816:	75 1a                	jne    400832 <main+0xec>
  400818:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081d:	48 83 c0 04          	add    $0x4,%rax
  400821:	48 89 c7             	mov    %rax,%rdi
  400824:	e8 39 04 00 00       	callq  400c62 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
  400829:	8b 00                	mov    (%rax),%eax
  40082b:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400830:	74 07                	je     400839 <main+0xf3>
  400832:	b8 01 00 00 00       	mov    $0x1,%eax
  400837:	eb 05                	jmp    40083e <main+0xf8>
  400839:	b8 00 00 00 00       	mov    $0x0,%eax
  40083e:	0f b6 c0             	movzbl %al,%eax
  400841:	48 83 c4 10          	add    $0x10,%rsp
  400845:	5b                   	pop    %rbx
  400846:	c3                   	retq   

0000000000400847 <_Z12new_instancev>:
  400847:	53                   	push   %rbx
  400848:	bf 04 00 00 00       	mov    $0x4,%edi
  40084d:	e8 55 03 00 00       	callq  400ba7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>
  400852:	48 89 c3             	mov    %rax,%rbx
  400855:	b8 00 00 00 00       	mov    $0x0,%eax
  40085a:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  40085e:	48 83 f8 03          	cmp    $0x3,%rax
  400862:	74 06                	je     40086a <_Z12new_instancev+0x23>
  400864:	48 83 c0 01          	add    $0x1,%rax
  400868:	eb f0                	jmp    40085a <_Z12new_instancev+0x13>
  40086a:	48 89 df             	mov    %rbx,%rdi
  40086d:	e8 10 02 00 00       	callq  400a82 <_ZN9TestClassC1Ev>
  400872:	48 89 d8             	mov    %rbx,%rax
  400875:	5b                   	pop    %rbx
  400876:	c3                   	retq   

0000000000400877 <_Z12write_field0P9TestClass>:
  400877:	48 83 ec 18          	sub    $0x18,%rsp
  40087b:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400880:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400885:	48 83 c0 04          	add    $0x4,%rax
  400889:	be 77 77 00 00       	mov    $0x7777,%esi
  40088e:	48 89 c7             	mov    %rax,%rdi
  400891:	e8 54 03 00 00       	callq  400bea <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>
  400896:	90                   	nop
  400897:	48 83 c4 18          	add    $0x18,%rsp
  40089b:	c3                   	retq   

000000000040089c <_Z11read_field0P9TestClass>:
  40089c:	48 83 ec 18          	sub    $0x18,%rsp
  4008a0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008a5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008aa:	48 83 c0 04          	add    $0x4,%rax
  4008ae:	48 89 c7             	mov    %rax,%rdi
  4008b1:	e8 90 03 00 00       	callq  400c46 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>
  4008b6:	8b 00                	mov    (%rax),%eax
  4008b8:	48 83 c4 18          	add    $0x18,%rsp
  4008bc:	c3                   	retq   

00000000004008bd <_Z12write_field1P9TestClass>:
  4008bd:	48 83 ec 18          	sub    $0x18,%rsp
  4008c1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008c6:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008cb:	48 83 c0 04          	add    $0x4,%rax
  4008cf:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d4:	48 89 c7             	mov    %rax,%rdi
  4008d7:	e8 3c 03 00 00       	callq  400c18 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>
  4008dc:	90                   	nop
  4008dd:	48 83 c4 18          	add    $0x18,%rsp
  4008e1:	c3                   	retq   

00000000004008e2 <_Z11read_field1P9TestClass>:
  4008e2:	48 83 ec 18          	sub    $0x18,%rsp
  4008e6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008f0:	48 83 c0 04          	add    $0x4,%rax
  4008f4:	48 89 c7             	mov    %rax,%rdi
  4008f7:	e8 66 03 00 00       	callq  400c62 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>
  4008fc:	8b 00                	mov    (%rax),%eax
  4008fe:	48 83 c4 18          	add    $0x18,%rsp
  400902:	c3                   	retq   

0000000000400903 <_Z21explicit_write_field0m>:
  400903:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400908:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40090d:	c7 04 85 00 44 61 00 	movl   $0x7777,0x614400(,%rax,4)
  400914:	77 77 00 00 
  400918:	90                   	nop
  400919:	c3                   	retq   

000000000040091a <_Z20explicit_read_field0m>:
  40091a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40091f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400924:	8b 04 85 00 44 61 00 	mov    0x614400(,%rax,4),%eax
  40092b:	c3                   	retq   

000000000040092c <_Z25explicit_write_field0_aosP16TestClassCompare>:
  40092c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400931:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400936:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  40093c:	90                   	nop
  40093d:	c3                   	retq   

000000000040093e <_Z24explicit_read_field0_aosP16TestClassCompare>:
  40093e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400943:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400948:	8b 00                	mov    (%rax),%eax
  40094a:	c3                   	retq   

000000000040094b <_Z25explicit_write_field1_aosP16TestClassCompare>:
  40094b:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400950:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400955:	c7 40 04 77 77 00 00 	movl   $0x7777,0x4(%rax)
  40095c:	90                   	nop
  40095d:	c3                   	retq   

000000000040095e <_Z24explicit_read_field1_aosP16TestClassCompare>:
  40095e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400963:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400968:	8b 40 04             	mov    0x4(%rax),%eax
  40096b:	c3                   	retq   

000000000040096c <_ZnwmPv>:
  40096c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400971:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400976:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40097b:	c3                   	retq   

000000000040097c <_ZdlPvS_>:
  40097c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400981:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400986:	90                   	nop
  400987:	c3                   	retq   

0000000000400988 <_ZN9TestClass15increase_field0Ev>:
  400988:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40098d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400991:	41 ff 72 f8          	pushq  -0x8(%r10)
  400995:	55                   	push   %rbp
  400996:	48 89 e5             	mov    %rsp,%rbp
  400999:	41 52                	push   %r10
  40099b:	48 83 ec 48          	sub    $0x48,%rsp
  40099f:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009a3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009aa:	00 00 
  4009ac:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009b0:	31 c0                	xor    %eax,%eax
  4009b2:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4009b6:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4009ba:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009be:	ba 55 55 00 00       	mov    $0x5555,%edx
  4009c3:	48 89 ce             	mov    %rcx,%rsi
  4009c6:	48 89 c7             	mov    %rax,%rdi
  4009c9:	e8 f2 00 00 00       	callq  400ac0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEmLEi>
  4009ce:	90                   	nop
  4009cf:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009d3:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009da:	00 00 
  4009dc:	74 05                	je     4009e3 <_ZN9TestClass15increase_field0Ev+0x5b>
  4009de:	e8 2d fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  4009e3:	48 83 c4 48          	add    $0x48,%rsp
  4009e7:	41 5a                	pop    %r10
  4009e9:	5d                   	pop    %rbp
  4009ea:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009ee:	c3                   	retq   
  4009ef:	90                   	nop

00000000004009f0 <_ZN9TestClass15increase_field1Ev>:
  4009f0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4009f5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009f9:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009fd:	55                   	push   %rbp
  4009fe:	48 89 e5             	mov    %rsp,%rbp
  400a01:	41 52                	push   %r10
  400a03:	48 83 ec 48          	sub    $0x48,%rsp
  400a07:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400a0b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400a12:	00 00 
  400a14:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a18:	31 c0                	xor    %eax,%eax
  400a1a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400a1e:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400a22:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400a26:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a2b:	48 89 ce             	mov    %rcx,%rsi
  400a2e:	48 89 c7             	mov    %rax,%rdi
  400a31:	e8 d2 00 00 00       	callq  400b08 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEmLEi>
  400a36:	90                   	nop
  400a37:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a3b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a42:	00 00 
  400a44:	74 05                	je     400a4b <_ZN9TestClass15increase_field1Ev+0x5b>
  400a46:	e8 c5 fb ff ff       	callq  400610 <__stack_chk_fail@plt>
  400a4b:	48 83 c4 48          	add    $0x48,%rsp
  400a4f:	41 5a                	pop    %r10
  400a51:	5d                   	pop    %rbp
  400a52:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a56:	c3                   	retq   
  400a57:	90                   	nop

0000000000400a58 <_ZN9TestClass14storage_bufferEv>:
  400a58:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a5d:	c3                   	retq   

0000000000400a5e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400a5e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a63:	90                   	nop
  400a64:	c3                   	retq   
  400a65:	90                   	nop

0000000000400a66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEC1Ev>:
  400a66:	48 83 ec 18          	sub    $0x18,%rsp
  400a6a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a6f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a74:	48 89 c7             	mov    %rax,%rdi
  400a77:	e8 e2 ff ff ff       	callq  400a5e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  400a7c:	90                   	nop
  400a7d:	48 83 c4 18          	add    $0x18,%rsp
  400a81:	c3                   	retq   

0000000000400a82 <_ZN9TestClassC1Ev>:
  400a82:	48 83 ec 18          	sub    $0x18,%rsp
  400a86:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a8b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a90:	48 89 c7             	mov    %rax,%rdi
  400a93:	e8 ce ff ff ff       	callq  400a66 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEC1Ev>
  400a98:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a9d:	48 83 c0 04          	add    $0x4,%rax
  400aa1:	48 89 c7             	mov    %rax,%rdi
  400aa4:	e8 2b 01 00 00       	callq  400bd4 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC1Ev>
  400aa9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aae:	48 83 c0 04          	add    $0x4,%rax
  400ab2:	48 89 c7             	mov    %rax,%rdi
  400ab5:	e8 22 01 00 00       	callq  400bdc <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC1Ev>
  400aba:	90                   	nop
  400abb:	48 83 c4 18          	add    $0x18,%rsp
  400abf:	c3                   	retq   

0000000000400ac0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEmLEi>:
  400ac0:	48 83 ec 28          	sub    $0x28,%rsp
  400ac4:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400ac9:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400ace:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ad2:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400ad7:	48 89 c7             	mov    %rax,%rdi
  400ada:	e8 9f 01 00 00       	callq  400c7e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400adf:	8b 10                	mov    (%rax),%edx
  400ae1:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400ae6:	89 10                	mov    %edx,(%rax)
  400ae8:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400aed:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400af2:	48 89 d6             	mov    %rdx,%rsi
  400af5:	48 89 c7             	mov    %rax,%rdi
  400af8:	e8 9d 01 00 00       	callq  400c9a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC1ERKS3_>
  400afd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b02:	48 83 c4 28          	add    $0x28,%rsp
  400b06:	c3                   	retq   
  400b07:	90                   	nop

0000000000400b08 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEmLEi>:
  400b08:	48 83 ec 28          	sub    $0x28,%rsp
  400b0c:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400b11:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400b16:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400b1a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b1f:	48 89 c7             	mov    %rax,%rdi
  400b22:	e8 7f 01 00 00       	callq  400ca6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400b27:	8b 10                	mov    (%rax),%edx
  400b29:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400b2e:	89 10                	mov    %edx,(%rax)
  400b30:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400b35:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b3a:	48 89 d6             	mov    %rdx,%rsi
  400b3d:	48 89 c7             	mov    %rax,%rdi
  400b40:	e8 7d 01 00 00       	callq  400cc2 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC1ERKS3_>
  400b45:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b4a:	48 83 c4 28          	add    $0x28,%rsp
  400b4e:	c3                   	retq   

0000000000400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>:
  400b4f:	48 83 ec 08          	sub    $0x8,%rsp
  400b53:	e8 00 ff ff ff       	callq  400a58 <_ZN9TestClass14storage_bufferEv>
  400b58:	48 83 c4 08          	add    $0x8,%rsp
  400b5c:	c3                   	retq   

0000000000400b5d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIIEEEvDpT_>:
  400b5d:	41 54                	push   %r12
  400b5f:	55                   	push   %rbp
  400b60:	53                   	push   %rbx
  400b61:	e8 e9 ff ff ff       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400b66:	48 89 c5             	mov    %rax,%rbp
  400b69:	48 89 ee             	mov    %rbp,%rsi
  400b6c:	bf d0 91 00 00       	mov    $0x91d0,%edi
  400b71:	e8 f6 fd ff ff       	callq  40096c <_ZnwmPv>
  400b76:	48 89 c3             	mov    %rax,%rbx
  400b79:	48 85 db             	test   %rbx,%rbx
  400b7c:	74 23                	je     400ba1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400b7e:	48 89 df             	mov    %rbx,%rdi
  400b81:	e8 48 01 00 00       	callq  400cce <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400b86:	eb 19                	jmp    400ba1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400b88:	49 89 c4             	mov    %rax,%r12
  400b8b:	48 89 ee             	mov    %rbp,%rsi
  400b8e:	48 89 df             	mov    %rbx,%rdi
  400b91:	e8 e6 fd ff ff       	callq  40097c <_ZdlPvS_>
  400b96:	4c 89 e0             	mov    %r12,%rax
  400b99:	48 89 c7             	mov    %rax,%rdi
  400b9c:	e8 8f fa ff ff       	callq  400630 <_Unwind_Resume@plt>
  400ba1:	90                   	nop
  400ba2:	5b                   	pop    %rbx
  400ba3:	5d                   	pop    %rbp
  400ba4:	41 5c                	pop    %r12
  400ba6:	c3                   	retq   

0000000000400ba7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEnwEm>:
  400ba7:	48 83 ec 18          	sub    $0x18,%rsp
  400bab:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bb0:	e8 41 01 00 00       	callq  400cf6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400bb5:	e8 95 ff ff ff       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400bba:	be 01 00 00 00       	mov    $0x1,%esi
  400bbf:	48 89 c7             	mov    %rax,%rdi
  400bc2:	e8 51 01 00 00       	callq  400d18 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400bc7:	48 89 c7             	mov    %rax,%rdi
  400bca:	e8 29 01 00 00       	callq  400cf8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE3getEm>
  400bcf:	48 83 c4 18          	add    $0x18,%rsp
  400bd3:	c3                   	retq   

0000000000400bd4 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC1Ev>:
  400bd4:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bd9:	90                   	nop
  400bda:	c3                   	retq   
  400bdb:	90                   	nop

0000000000400bdc <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC1Ev>:
  400bdc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400be1:	90                   	nop
  400be2:	c3                   	retq   

0000000000400be3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EEdlEPv>:
  400be3:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400be8:	90                   	nop
  400be9:	c3                   	retq   

0000000000400bea <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEaSEi>:
  400bea:	48 83 ec 18          	sub    $0x18,%rsp
  400bee:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bf3:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400bf7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bfc:	48 89 c7             	mov    %rax,%rdi
  400bff:	e8 7a 00 00 00       	callq  400c7e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400c04:	48 89 c2             	mov    %rax,%rdx
  400c07:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c0b:	89 02                	mov    %eax,(%rdx)
  400c0d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c12:	48 83 c4 18          	add    $0x18,%rsp
  400c16:	c3                   	retq   
  400c17:	90                   	nop

0000000000400c18 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEaSEi>:
  400c18:	48 83 ec 18          	sub    $0x18,%rsp
  400c1c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c21:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c25:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c2a:	48 89 c7             	mov    %rax,%rdi
  400c2d:	e8 74 00 00 00       	callq  400ca6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400c32:	48 89 c2             	mov    %rax,%rdx
  400c35:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c39:	89 02                	mov    %eax,(%rdx)
  400c3b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c40:	48 83 c4 18          	add    $0x18,%rsp
  400c44:	c3                   	retq   
  400c45:	90                   	nop

0000000000400c46 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEcvRiEv>:
  400c46:	48 83 ec 18          	sub    $0x18,%rsp
  400c4a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c4f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c54:	48 89 c7             	mov    %rax,%rdi
  400c57:	e8 22 00 00 00       	callq  400c7e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400c5c:	48 83 c4 18          	add    $0x18,%rsp
  400c60:	c3                   	retq   
  400c61:	90                   	nop

0000000000400c62 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEcvRiEv>:
  400c62:	48 83 ec 18          	sub    $0x18,%rsp
  400c66:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c6b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c70:	48 89 c7             	mov    %rax,%rdi
  400c73:	e8 2e 00 00 00       	callq  400ca6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>
  400c78:	48 83 c4 18          	add    $0x18,%rsp
  400c7c:	c3                   	retq   
  400c7d:	90                   	nop

0000000000400c7e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE8data_ptrEv>:
  400c7e:	48 83 ec 18          	sub    $0x18,%rsp
  400c82:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c87:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c8c:	48 89 c7             	mov    %rax,%rdi
  400c8f:	e8 be 00 00 00       	callq  400d52 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400c94:	48 83 c4 18          	add    $0x18,%rsp
  400c98:	c3                   	retq   
  400c99:	90                   	nop

0000000000400c9a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassEC1ERKS3_>:
  400c9a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c9f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400ca4:	90                   	nop
  400ca5:	c3                   	retq   

0000000000400ca6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE8data_ptrEv>:
  400ca6:	48 83 ec 18          	sub    $0x18,%rsp
  400caa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400caf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cb4:	48 89 c7             	mov    %rax,%rdi
  400cb7:	e8 ec 00 00 00       	callq  400da8 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400cbc:	48 83 c4 18          	add    $0x18,%rsp
  400cc0:	c3                   	retq   
  400cc1:	90                   	nop

0000000000400cc2 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassEC1ERKS3_>:
  400cc2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cc7:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400ccc:	90                   	nop
  400ccd:	c3                   	retq   

0000000000400cce <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400cce:	48 83 ec 18          	sub    $0x18,%rsp
  400cd2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cd7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cdc:	48 89 c7             	mov    %rax,%rdi
  400cdf:	e8 20 01 00 00       	callq  400e04 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400ce4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ce9:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400cf0:	90                   	nop
  400cf1:	48 83 c4 18          	add    $0x18,%rsp
  400cf5:	c3                   	retq   

0000000000400cf6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cf6:	90                   	nop
  400cf7:	c3                   	retq   

0000000000400cf8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE3getEm>:
  400cf8:	48 83 ec 18          	sub    $0x18,%rsp
  400cfc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d01:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d06:	48 83 c0 01          	add    $0x1,%rax
  400d0a:	48 89 c7             	mov    %rax,%rdi
  400d0d:	e8 05 01 00 00       	callq  400e17 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d12:	48 83 c4 18          	add    $0x18,%rsp
  400d16:	c3                   	retq   
  400d17:	90                   	nop

0000000000400d18 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d18:	48 83 ec 28          	sub    $0x28,%rsp
  400d1c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d21:	48 89 34 24          	mov    %rsi,(%rsp)
  400d25:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d2a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d2f:	48 8b 04 24          	mov    (%rsp),%rax
  400d33:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d38:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  400d3d:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400d42:	48 89 d6             	mov    %rdx,%rsi
  400d45:	48 89 c7             	mov    %rax,%rdi
  400d48:	e8 fe 00 00 00       	callq  400e4b <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400d4d:	48 83 c4 28          	add    $0x28,%rsp
  400d51:	c3                   	retq   

0000000000400d52 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d52:	48 83 ec 38          	sub    $0x38,%rsp
  400d56:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d5b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d60:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d65:	e8 e5 fd ff ff       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400d6a:	48 89 c7             	mov    %rax,%rdi
  400d6d:	e8 0e 01 00 00       	callq  400e80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d72:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400d77:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d7c:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400d81:	48 83 e8 04          	sub    $0x4,%rax
  400d85:	48 c1 e8 02          	shr    $0x2,%rax
  400d89:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400d90:	00 
  400d91:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400d96:	48 01 d0             	add    %rdx,%rax
  400d99:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400d9e:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400da3:	48 83 c4 38          	add    $0x38,%rsp
  400da7:	c3                   	retq   

0000000000400da8 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400da8:	48 83 ec 38          	sub    $0x38,%rsp
  400dac:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400db1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400db6:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400dbb:	e8 8f fd ff ff       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400dc0:	48 89 c7             	mov    %rax,%rdi
  400dc3:	e8 b8 00 00 00       	callq  400e80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400dc8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400dcd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400dd2:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400dd7:	48 83 e8 04          	sub    $0x4,%rax
  400ddb:	48 c1 e8 02          	shr    $0x2,%rax
  400ddf:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400de6:	00 
  400de7:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400dec:	48 01 d0             	add    %rdx,%rax
  400def:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400df5:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400dfa:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400dff:	48 83 c4 38          	add    $0x38,%rsp
  400e03:	c3                   	retq   

0000000000400e04 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400e04:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e09:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e0e:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400e15:	90                   	nop
  400e16:	c3                   	retq   

0000000000400e17 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e17:	48 83 ec 28          	sub    $0x28,%rsp
  400e1b:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400e20:	e8 2a fd ff ff       	callq  400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageELi1EE7storageEv>
  400e25:	48 89 c7             	mov    %rax,%rdi
  400e28:	e8 53 00 00 00       	callq  400e80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400e2d:	48 89 c2             	mov    %rax,%rdx
  400e30:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e35:	48 c1 e0 02          	shl    $0x2,%rax
  400e39:	48 01 d0             	add    %rdx,%rax
  400e3c:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400e41:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400e46:	48 83 c4 28          	add    $0x28,%rsp
  400e4a:	c3                   	retq   

0000000000400e4b <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400e4b:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400e50:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400e55:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e5a:	48 8b 00             	mov    (%rax),%rax
  400e5d:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400e62:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e67:	48 8b 10             	mov    (%rax),%rdx
  400e6a:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400e6f:	48 01 c2             	add    %rax,%rdx
  400e72:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e77:	48 89 10             	mov    %rdx,(%rax)
  400e7a:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e7f:	c3                   	retq   

0000000000400e80 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400e80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e85:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e8a:	48 83 c0 10          	add    $0x10,%rax
  400e8e:	c3                   	retq   
  400e8f:	90                   	nop

0000000000400e90 <__libc_csu_init>:
  400e90:	41 57                	push   %r15
  400e92:	41 56                	push   %r14
  400e94:	41 89 ff             	mov    %edi,%r15d
  400e97:	41 55                	push   %r13
  400e99:	41 54                	push   %r12
  400e9b:	4c 8d 25 4e 0f 20 00 	lea    0x200f4e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400ea2:	55                   	push   %rbp
  400ea3:	48 8d 2d 4e 0f 20 00 	lea    0x200f4e(%rip),%rbp        # 601df8 <__init_array_end>
  400eaa:	53                   	push   %rbx
  400eab:	49 89 f6             	mov    %rsi,%r14
  400eae:	49 89 d5             	mov    %rdx,%r13
  400eb1:	4c 29 e5             	sub    %r12,%rbp
  400eb4:	48 83 ec 08          	sub    $0x8,%rsp
  400eb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ebc:	e8 ff f6 ff ff       	callq  4005c0 <_init>
  400ec1:	48 85 ed             	test   %rbp,%rbp
  400ec4:	74 20                	je     400ee6 <__libc_csu_init+0x56>
  400ec6:	31 db                	xor    %ebx,%ebx
  400ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ecf:	00 
  400ed0:	4c 89 ea             	mov    %r13,%rdx
  400ed3:	4c 89 f6             	mov    %r14,%rsi
  400ed6:	44 89 ff             	mov    %r15d,%edi
  400ed9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400edd:	48 83 c3 01          	add    $0x1,%rbx
  400ee1:	48 39 eb             	cmp    %rbp,%rbx
  400ee4:	75 ea                	jne    400ed0 <__libc_csu_init+0x40>
  400ee6:	48 83 c4 08          	add    $0x8,%rsp
  400eea:	5b                   	pop    %rbx
  400eeb:	5d                   	pop    %rbp
  400eec:	41 5c                	pop    %r12
  400eee:	41 5d                	pop    %r13
  400ef0:	41 5e                	pop    %r14
  400ef2:	41 5f                	pop    %r15
  400ef4:	c3                   	retq   
  400ef5:	90                   	nop
  400ef6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400efd:	00 00 00 

0000000000400f00 <__libc_csu_fini>:
  400f00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f04 <_fini>:
  400f04:	48 83 ec 08          	sub    $0x8,%rsp
  400f08:	48 83 c4 08          	add    $0x8,%rsp
  400f0c:	c3                   	retq   
