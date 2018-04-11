
bin/g++_-O0_StaticStorage_4:     file format elf64-x86-64


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
  40065f:	49 c7 c0 90 0e 40 00 	mov    $0x400e90,%r8
  400666:	48 c7 c1 20 0e 40 00 	mov    $0x400e20,%rcx
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
  400700:	80 3d 59 19 20 00 00 	cmpb   $0x0,0x201959(%rip)        # 602060 <completed.7585>
  400707:	75 11                	jne    40071a <__do_global_dtors_aux+0x1a>
  400709:	55                   	push   %rbp
  40070a:	48 89 e5             	mov    %rsp,%rbp
  40070d:	e8 6e ff ff ff       	callq  400680 <deregister_tm_clones>
  400712:	5d                   	pop    %rbp
  400713:	c6 05 46 19 20 00 01 	movb   $0x1,0x201946(%rip)        # 602060 <completed.7585>
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
  40074b:	e8 cd 03 00 00       	callq  400b1d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  400750:	bf 04 00 00 00       	mov    $0x4,%edi
  400755:	e8 0d 04 00 00       	callq  400b67 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  40075a:	48 89 c3             	mov    %rax,%rbx
  40075d:	b8 00 00 00 00       	mov    $0x0,%eax
  400762:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  400766:	48 83 f8 03          	cmp    $0x3,%rax
  40076a:	74 06                	je     400772 <main+0x2c>
  40076c:	48 83 c0 01          	add    $0x1,%rax
  400770:	eb f0                	jmp    400762 <main+0x1c>
  400772:	48 89 df             	mov    %rbx,%rdi
  400775:	e8 c8 02 00 00       	callq  400a42 <_ZN9TestClassC1Ev>
  40077a:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40077f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400784:	48 83 c0 04          	add    $0x4,%rax
  400788:	be 77 77 00 00       	mov    $0x7777,%esi
  40078d:	48 89 c7             	mov    %rax,%rdi
  400790:	e8 15 04 00 00       	callq  400baa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEaSEi>
  400795:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40079a:	48 83 c0 04          	add    $0x4,%rax
  40079e:	be 88 88 00 00       	mov    $0x8888,%esi
  4007a3:	48 89 c7             	mov    %rax,%rdi
  4007a6:	e8 2d 04 00 00       	callq  400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEaSEi>
  4007ab:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007b0:	48 89 c7             	mov    %rax,%rdi
  4007b3:	e8 90 01 00 00       	callq  400948 <_ZN9TestClass15increase_field0Ev>
  4007b8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bd:	48 89 c7             	mov    %rax,%rdi
  4007c0:	e8 eb 01 00 00       	callq  4009b0 <_ZN9TestClass15increase_field1Ev>
  4007c5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ca:	48 83 c0 04          	add    $0x4,%rax
  4007ce:	48 89 c7             	mov    %rax,%rdi
  4007d1:	e8 4c 04 00 00       	callq  400c22 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEcvRiEv>
  4007d6:	8b 18                	mov    (%rax),%ebx
  4007d8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007dd:	48 83 c0 04          	add    $0x4,%rax
  4007e1:	48 89 c7             	mov    %rax,%rdi
  4007e4:	e8 1d 04 00 00       	callq  400c06 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEcvRiEv>
  4007e9:	8b 00                	mov    (%rax),%eax
  4007eb:	89 da                	mov    %ebx,%edx
  4007ed:	89 c6                	mov    %eax,%esi
  4007ef:	bf bc 0e 40 00       	mov    $0x400ebc,%edi
  4007f4:	b8 00 00 00 00       	mov    $0x0,%eax
  4007f9:	e8 f2 fd ff ff       	callq  4005f0 <printf@plt>
  4007fe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400803:	48 83 c0 04          	add    $0x4,%rax
  400807:	48 89 c7             	mov    %rax,%rdi
  40080a:	e8 f7 03 00 00       	callq  400c06 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEcvRiEv>
  40080f:	8b 00                	mov    (%rax),%eax
  400811:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400816:	75 1a                	jne    400832 <main+0xec>
  400818:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081d:	48 83 c0 04          	add    $0x4,%rax
  400821:	48 89 c7             	mov    %rax,%rdi
  400824:	e8 f9 03 00 00       	callq  400c22 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEcvRiEv>
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
  40084d:	e8 15 03 00 00       	callq  400b67 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400852:	48 89 c3             	mov    %rax,%rbx
  400855:	b8 00 00 00 00       	mov    $0x0,%eax
  40085a:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  40085e:	48 83 f8 03          	cmp    $0x3,%rax
  400862:	74 06                	je     40086a <_Z12new_instancev+0x23>
  400864:	48 83 c0 01          	add    $0x1,%rax
  400868:	eb f0                	jmp    40085a <_Z12new_instancev+0x13>
  40086a:	48 89 df             	mov    %rbx,%rdi
  40086d:	e8 d0 01 00 00       	callq  400a42 <_ZN9TestClassC1Ev>
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
  400891:	e8 14 03 00 00       	callq  400baa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEaSEi>
  400896:	90                   	nop
  400897:	48 83 c4 18          	add    $0x18,%rsp
  40089b:	c3                   	retq   

000000000040089c <_Z11read_field0P9TestClass>:
  40089c:	48 83 ec 18          	sub    $0x18,%rsp
  4008a0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008a5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008aa:	48 83 c0 04          	add    $0x4,%rax
  4008ae:	48 89 c7             	mov    %rax,%rdi
  4008b1:	e8 50 03 00 00       	callq  400c06 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEcvRiEv>
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
  4008d7:	e8 fc 02 00 00       	callq  400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEaSEi>
  4008dc:	90                   	nop
  4008dd:	48 83 c4 18          	add    $0x18,%rsp
  4008e1:	c3                   	retq   

00000000004008e2 <_Z11read_field1P9TestClass>:
  4008e2:	48 83 ec 18          	sub    $0x18,%rsp
  4008e6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008f0:	48 83 c0 04          	add    $0x4,%rax
  4008f4:	48 89 c7             	mov    %rax,%rdi
  4008f7:	e8 26 03 00 00       	callq  400c22 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEcvRiEv>
  4008fc:	8b 00                	mov    (%rax),%eax
  4008fe:	48 83 c4 18          	add    $0x18,%rsp
  400902:	c3                   	retq   

0000000000400903 <_Z21explicit_write_field0m>:
  400903:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400908:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40090d:	c7 04 85 60 b2 60 00 	movl   $0x7777,0x60b260(,%rax,4)
  400914:	77 77 00 00 
  400918:	90                   	nop
  400919:	c3                   	retq   

000000000040091a <_Z20explicit_read_field0m>:
  40091a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40091f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400924:	8b 04 85 60 b2 60 00 	mov    0x60b260(,%rax,4),%eax
  40092b:	c3                   	retq   

000000000040092c <_ZnwmPv>:
  40092c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400931:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400936:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40093b:	c3                   	retq   

000000000040093c <_ZdlPvS_>:
  40093c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400941:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400946:	90                   	nop
  400947:	c3                   	retq   

0000000000400948 <_ZN9TestClass15increase_field0Ev>:
  400948:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40094d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400951:	41 ff 72 f8          	pushq  -0x8(%r10)
  400955:	55                   	push   %rbp
  400956:	48 89 e5             	mov    %rsp,%rbp
  400959:	41 52                	push   %r10
  40095b:	48 83 ec 48          	sub    $0x48,%rsp
  40095f:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400963:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40096a:	00 00 
  40096c:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400970:	31 c0                	xor    %eax,%eax
  400972:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400976:	48 8d 48 04          	lea    0x4(%rax),%rcx
  40097a:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40097e:	ba 55 55 00 00       	mov    $0x5555,%edx
  400983:	48 89 ce             	mov    %rcx,%rsi
  400986:	48 89 c7             	mov    %rax,%rdi
  400989:	e8 f2 00 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEmLEi>
  40098e:	90                   	nop
  40098f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400993:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40099a:	00 00 
  40099c:	74 05                	je     4009a3 <_ZN9TestClass15increase_field0Ev+0x5b>
  40099e:	e8 6d fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  4009a3:	48 83 c4 48          	add    $0x48,%rsp
  4009a7:	41 5a                	pop    %r10
  4009a9:	5d                   	pop    %rbp
  4009aa:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009ae:	c3                   	retq   
  4009af:	90                   	nop

00000000004009b0 <_ZN9TestClass15increase_field1Ev>:
  4009b0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4009b5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009b9:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009bd:	55                   	push   %rbp
  4009be:	48 89 e5             	mov    %rsp,%rbp
  4009c1:	41 52                	push   %r10
  4009c3:	48 83 ec 48          	sub    $0x48,%rsp
  4009c7:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009cb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009d2:	00 00 
  4009d4:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009d8:	31 c0                	xor    %eax,%eax
  4009da:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4009de:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4009e2:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009e6:	ba 44 44 00 00       	mov    $0x4444,%edx
  4009eb:	48 89 ce             	mov    %rcx,%rsi
  4009ee:	48 89 c7             	mov    %rax,%rdi
  4009f1:	e8 d2 00 00 00       	callq  400ac8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEmLEi>
  4009f6:	90                   	nop
  4009f7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009fb:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a02:	00 00 
  400a04:	74 05                	je     400a0b <_ZN9TestClass15increase_field1Ev+0x5b>
  400a06:	e8 05 fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  400a0b:	48 83 c4 48          	add    $0x48,%rsp
  400a0f:	41 5a                	pop    %r10
  400a11:	5d                   	pop    %rbp
  400a12:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a16:	c3                   	retq   
  400a17:	90                   	nop

0000000000400a18 <_ZN9TestClass14storage_bufferEv>:
  400a18:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a1d:	c3                   	retq   

0000000000400a1e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400a1e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a23:	90                   	nop
  400a24:	c3                   	retq   
  400a25:	90                   	nop

0000000000400a26 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>:
  400a26:	48 83 ec 18          	sub    $0x18,%rsp
  400a2a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a2f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a34:	48 89 c7             	mov    %rax,%rdi
  400a37:	e8 e2 ff ff ff       	callq  400a1e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  400a3c:	90                   	nop
  400a3d:	48 83 c4 18          	add    $0x18,%rsp
  400a41:	c3                   	retq   

0000000000400a42 <_ZN9TestClassC1Ev>:
  400a42:	48 83 ec 18          	sub    $0x18,%rsp
  400a46:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a4b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a50:	48 89 c7             	mov    %rax,%rdi
  400a53:	e8 ce ff ff ff       	callq  400a26 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEC1Ev>
  400a58:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a5d:	48 83 c0 04          	add    $0x4,%rax
  400a61:	48 89 c7             	mov    %rax,%rdi
  400a64:	e8 2b 01 00 00       	callq  400b94 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEC1Ev>
  400a69:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a6e:	48 83 c0 04          	add    $0x4,%rax
  400a72:	48 89 c7             	mov    %rax,%rdi
  400a75:	e8 22 01 00 00       	callq  400b9c <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEC1Ev>
  400a7a:	90                   	nop
  400a7b:	48 83 c4 18          	add    $0x18,%rsp
  400a7f:	c3                   	retq   

0000000000400a80 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEmLEi>:
  400a80:	48 83 ec 28          	sub    $0x28,%rsp
  400a84:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400a89:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400a8e:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400a92:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a97:	48 89 c7             	mov    %rax,%rdi
  400a9a:	e8 9f 01 00 00       	callq  400c3e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a9f:	8b 10                	mov    (%rax),%edx
  400aa1:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400aa6:	89 10                	mov    %edx,(%rax)
  400aa8:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400aad:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ab2:	48 89 d6             	mov    %rdx,%rsi
  400ab5:	48 89 c7             	mov    %rax,%rdi
  400ab8:	e8 9d 01 00 00       	callq  400c5a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEC1ERKS3_>
  400abd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ac2:	48 83 c4 28          	add    $0x28,%rsp
  400ac6:	c3                   	retq   
  400ac7:	90                   	nop

0000000000400ac8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEmLEi>:
  400ac8:	48 83 ec 28          	sub    $0x28,%rsp
  400acc:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400ad1:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400ad6:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ada:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400adf:	48 89 c7             	mov    %rax,%rdi
  400ae2:	e8 7f 01 00 00       	callq  400c66 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae7:	8b 10                	mov    (%rax),%edx
  400ae9:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400aee:	89 10                	mov    %edx,(%rax)
  400af0:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400af5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400afa:	48 89 d6             	mov    %rdx,%rsi
  400afd:	48 89 c7             	mov    %rax,%rdi
  400b00:	e8 7d 01 00 00       	callq  400c82 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEC1ERKS3_>
  400b05:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b0a:	48 83 c4 28          	add    $0x28,%rsp
  400b0e:	c3                   	retq   

0000000000400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>:
  400b0f:	48 83 ec 08          	sub    $0x8,%rsp
  400b13:	e8 00 ff ff ff       	callq  400a18 <_ZN9TestClass14storage_bufferEv>
  400b18:	48 83 c4 08          	add    $0x8,%rsp
  400b1c:	c3                   	retq   

0000000000400b1d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  400b1d:	41 54                	push   %r12
  400b1f:	55                   	push   %rbp
  400b20:	53                   	push   %rbx
  400b21:	e8 e9 ff ff ff       	callq  400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400b26:	48 89 c5             	mov    %rax,%rbp
  400b29:	48 89 ee             	mov    %rbp,%rsi
  400b2c:	bf d0 91 00 00       	mov    $0x91d0,%edi
  400b31:	e8 f6 fd ff ff       	callq  40092c <_ZnwmPv>
  400b36:	48 89 c3             	mov    %rax,%rbx
  400b39:	48 85 db             	test   %rbx,%rbx
  400b3c:	74 23                	je     400b61 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b3e:	48 89 df             	mov    %rbx,%rdi
  400b41:	e8 48 01 00 00       	callq  400c8e <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400b46:	eb 19                	jmp    400b61 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b48:	49 89 c4             	mov    %rax,%r12
  400b4b:	48 89 ee             	mov    %rbp,%rsi
  400b4e:	48 89 df             	mov    %rbx,%rdi
  400b51:	e8 e6 fd ff ff       	callq  40093c <_ZdlPvS_>
  400b56:	4c 89 e0             	mov    %r12,%rax
  400b59:	48 89 c7             	mov    %rax,%rdi
  400b5c:	e8 cf fa ff ff       	callq  400630 <_Unwind_Resume@plt>
  400b61:	90                   	nop
  400b62:	5b                   	pop    %rbx
  400b63:	5d                   	pop    %rbp
  400b64:	41 5c                	pop    %r12
  400b66:	c3                   	retq   

0000000000400b67 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  400b67:	48 83 ec 18          	sub    $0x18,%rsp
  400b6b:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b70:	e8 41 01 00 00       	callq  400cb6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400b75:	e8 95 ff ff ff       	callq  400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400b7a:	be 01 00 00 00       	mov    $0x1,%esi
  400b7f:	48 89 c7             	mov    %rax,%rdi
  400b82:	e8 51 01 00 00       	callq  400cd8 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400b87:	48 89 c7             	mov    %rax,%rdi
  400b8a:	e8 29 01 00 00       	callq  400cb8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  400b8f:	48 83 c4 18          	add    $0x18,%rsp
  400b93:	c3                   	retq   

0000000000400b94 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEC1Ev>:
  400b94:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b99:	90                   	nop
  400b9a:	c3                   	retq   
  400b9b:	90                   	nop

0000000000400b9c <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEC1Ev>:
  400b9c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ba1:	90                   	nop
  400ba2:	c3                   	retq   

0000000000400ba3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400ba3:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ba8:	90                   	nop
  400ba9:	c3                   	retq   

0000000000400baa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEaSEi>:
  400baa:	48 83 ec 18          	sub    $0x18,%rsp
  400bae:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bb3:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400bb7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bbc:	48 89 c7             	mov    %rax,%rdi
  400bbf:	e8 7a 00 00 00       	callq  400c3e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bc4:	48 89 c2             	mov    %rax,%rdx
  400bc7:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400bcb:	89 02                	mov    %eax,(%rdx)
  400bcd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bd2:	48 83 c4 18          	add    $0x18,%rsp
  400bd6:	c3                   	retq   
  400bd7:	90                   	nop

0000000000400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEaSEi>:
  400bd8:	48 83 ec 18          	sub    $0x18,%rsp
  400bdc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400be1:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400be5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bea:	48 89 c7             	mov    %rax,%rdi
  400bed:	e8 74 00 00 00       	callq  400c66 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bf2:	48 89 c2             	mov    %rax,%rdx
  400bf5:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400bf9:	89 02                	mov    %eax,(%rdx)
  400bfb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c00:	48 83 c4 18          	add    $0x18,%rsp
  400c04:	c3                   	retq   
  400c05:	90                   	nop

0000000000400c06 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEcvRiEv>:
  400c06:	48 83 ec 18          	sub    $0x18,%rsp
  400c0a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c0f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c14:	48 89 c7             	mov    %rax,%rdi
  400c17:	e8 22 00 00 00       	callq  400c3e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c1c:	48 83 c4 18          	add    $0x18,%rsp
  400c20:	c3                   	retq   
  400c21:	90                   	nop

0000000000400c22 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEcvRiEv>:
  400c22:	48 83 ec 18          	sub    $0x18,%rsp
  400c26:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c2b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c30:	48 89 c7             	mov    %rax,%rdi
  400c33:	e8 2e 00 00 00       	callq  400c66 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c38:	48 83 c4 18          	add    $0x18,%rsp
  400c3c:	c3                   	retq   
  400c3d:	90                   	nop

0000000000400c3e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c3e:	48 83 ec 18          	sub    $0x18,%rsp
  400c42:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c47:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c4c:	48 89 c7             	mov    %rax,%rdi
  400c4f:	e8 ba 00 00 00       	callq  400d0e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c54:	48 83 c4 18          	add    $0x18,%rsp
  400c58:	c3                   	retq   
  400c59:	90                   	nop

0000000000400c5a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassEC1ERKS3_>:
  400c5a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c5f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c64:	90                   	nop
  400c65:	c3                   	retq   

0000000000400c66 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c66:	48 83 ec 18          	sub    $0x18,%rsp
  400c6a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c6f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c74:	48 89 c7             	mov    %rax,%rdi
  400c77:	e8 e8 00 00 00       	callq  400d64 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c7c:	48 83 c4 18          	add    $0x18,%rsp
  400c80:	c3                   	retq   
  400c81:	90                   	nop

0000000000400c82 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassEC1ERKS3_>:
  400c82:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c87:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c8c:	90                   	nop
  400c8d:	c3                   	retq   

0000000000400c8e <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400c8e:	48 83 ec 18          	sub    $0x18,%rsp
  400c92:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c97:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c9c:	48 89 c7             	mov    %rax,%rdi
  400c9f:	e8 1c 01 00 00       	callq  400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400ca4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca9:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400cb0:	90                   	nop
  400cb1:	48 83 c4 18          	add    $0x18,%rsp
  400cb5:	c3                   	retq   

0000000000400cb6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cb6:	90                   	nop
  400cb7:	c3                   	retq   

0000000000400cb8 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400cb8:	48 83 ec 18          	sub    $0x18,%rsp
  400cbc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cc1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc6:	48 83 c0 01          	add    $0x1,%rax
  400cca:	48 89 c7             	mov    %rax,%rdi
  400ccd:	e8 01 01 00 00       	callq  400dd3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400cd2:	48 83 c4 18          	add    $0x18,%rsp
  400cd6:	c3                   	retq   
  400cd7:	90                   	nop

0000000000400cd8 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400cd8:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400cdd:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400ce2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ce7:	48 8b 00             	mov    (%rax),%rax
  400cea:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400cef:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400cf4:	48 8b 10             	mov    (%rax),%rdx
  400cf7:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400cfc:	48 01 c2             	add    %rax,%rdx
  400cff:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d04:	48 89 10             	mov    %rdx,(%rax)
  400d07:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d0c:	c3                   	retq   
  400d0d:	90                   	nop

0000000000400d0e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d0e:	48 83 ec 38          	sub    $0x38,%rsp
  400d12:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d17:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d1c:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d21:	e8 e9 fd ff ff       	callq  400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400d26:	48 89 c7             	mov    %rax,%rdi
  400d29:	e8 da 00 00 00       	callq  400e08 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d2e:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400d33:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d38:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400d3d:	48 83 e8 04          	sub    $0x4,%rax
  400d41:	48 c1 e8 02          	shr    $0x2,%rax
  400d45:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400d4c:	00 
  400d4d:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400d52:	48 01 d0             	add    %rdx,%rax
  400d55:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400d5a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400d5f:	48 83 c4 38          	add    $0x38,%rsp
  400d63:	c3                   	retq   

0000000000400d64 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi0E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d64:	48 83 ec 38          	sub    $0x38,%rsp
  400d68:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d6d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d72:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d77:	e8 93 fd ff ff       	callq  400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400d7c:	48 89 c7             	mov    %rax,%rdi
  400d7f:	e8 84 00 00 00       	callq  400e08 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400d84:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400d89:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d8e:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400d93:	48 83 e8 04          	sub    $0x4,%rax
  400d97:	48 c1 e8 02          	shr    $0x2,%rax
  400d9b:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400da2:	00 
  400da3:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400da8:	48 01 d0             	add    %rdx,%rax
  400dab:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400db1:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400db6:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400dbb:	48 83 c4 38          	add    $0x38,%rsp
  400dbf:	c3                   	retq   

0000000000400dc0 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400dc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dc5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dca:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400dd1:	90                   	nop
  400dd2:	c3                   	retq   

0000000000400dd3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400dd3:	48 83 ec 28          	sub    $0x28,%rsp
  400dd7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ddc:	e8 2e fd ff ff       	callq  400b0f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE7storageEv>
  400de1:	48 89 c7             	mov    %rax,%rdi
  400de4:	e8 1f 00 00 00       	callq  400e08 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>
  400de9:	48 89 c2             	mov    %rax,%rdx
  400dec:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400df1:	48 c1 e0 02          	shl    $0x2,%rax
  400df5:	48 01 d0             	add    %rdx,%rax
  400df8:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400dfd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400e02:	48 83 c4 28          	add    $0x28,%rsp
  400e06:	c3                   	retq   
  400e07:	90                   	nop

0000000000400e08 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EE8data_ptrEv>:
  400e08:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e0d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e12:	48 83 c0 10          	add    $0x10,%rax
  400e16:	c3                   	retq   
  400e17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e1e:	00 00 

0000000000400e20 <__libc_csu_init>:
  400e20:	41 57                	push   %r15
  400e22:	41 56                	push   %r14
  400e24:	41 89 ff             	mov    %edi,%r15d
  400e27:	41 55                	push   %r13
  400e29:	41 54                	push   %r12
  400e2b:	4c 8d 25 be 0f 20 00 	lea    0x200fbe(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400e32:	55                   	push   %rbp
  400e33:	48 8d 2d be 0f 20 00 	lea    0x200fbe(%rip),%rbp        # 601df8 <__init_array_end>
  400e3a:	53                   	push   %rbx
  400e3b:	49 89 f6             	mov    %rsi,%r14
  400e3e:	49 89 d5             	mov    %rdx,%r13
  400e41:	4c 29 e5             	sub    %r12,%rbp
  400e44:	48 83 ec 08          	sub    $0x8,%rsp
  400e48:	48 c1 fd 03          	sar    $0x3,%rbp
  400e4c:	e8 6f f7 ff ff       	callq  4005c0 <_init>
  400e51:	48 85 ed             	test   %rbp,%rbp
  400e54:	74 20                	je     400e76 <__libc_csu_init+0x56>
  400e56:	31 db                	xor    %ebx,%ebx
  400e58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e5f:	00 
  400e60:	4c 89 ea             	mov    %r13,%rdx
  400e63:	4c 89 f6             	mov    %r14,%rsi
  400e66:	44 89 ff             	mov    %r15d,%edi
  400e69:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e6d:	48 83 c3 01          	add    $0x1,%rbx
  400e71:	48 39 eb             	cmp    %rbp,%rbx
  400e74:	75 ea                	jne    400e60 <__libc_csu_init+0x40>
  400e76:	48 83 c4 08          	add    $0x8,%rsp
  400e7a:	5b                   	pop    %rbx
  400e7b:	5d                   	pop    %rbp
  400e7c:	41 5c                	pop    %r12
  400e7e:	41 5d                	pop    %r13
  400e80:	41 5e                	pop    %r14
  400e82:	41 5f                	pop    %r15
  400e84:	c3                   	retq   
  400e85:	90                   	nop
  400e86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e8d:	00 00 00 

0000000000400e90 <__libc_csu_fini>:
  400e90:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e94 <_fini>:
  400e94:	48 83 ec 08          	sub    $0x8,%rsp
  400e98:	48 83 c4 08          	add    $0x8,%rsp
  400e9c:	c3                   	retq   
