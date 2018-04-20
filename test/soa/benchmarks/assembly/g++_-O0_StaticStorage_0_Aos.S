
bin/g++_-O0_StaticStorage_0_Aos:     file format elf64-x86-64


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
  40065f:	49 c7 c0 00 0e 40 00 	mov    $0x400e00,%r8
  400666:	48 c7 c1 90 0d 40 00 	mov    $0x400d90,%rcx
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
  40074b:	e8 ab 03 00 00       	callq  400afb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIIEEEvDpT_>
  400750:	bf 00 00 00 00       	mov    $0x0,%edi
  400755:	e8 eb 03 00 00       	callq  400b45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>
  40075a:	48 89 c3             	mov    %rax,%rbx
  40075d:	48 89 df             	mov    %rbx,%rdi
  400760:	e8 c3 02 00 00       	callq  400a28 <_ZN9TestClassC1Ev>
  400765:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40076a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40076f:	be 77 77 00 00       	mov    $0x7777,%esi
  400774:	48 89 c7             	mov    %rax,%rdi
  400777:	e8 0c 04 00 00       	callq  400b88 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>
  40077c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400781:	be 88 88 00 00       	mov    $0x8888,%esi
  400786:	48 89 c7             	mov    %rax,%rdi
  400789:	e8 28 04 00 00       	callq  400bb6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>
  40078e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400793:	48 89 c7             	mov    %rax,%rdi
  400796:	e8 9b 01 00 00       	callq  400936 <_ZN9TestClass15increase_field0Ev>
  40079b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007a0:	48 89 c7             	mov    %rax,%rdi
  4007a3:	e8 f2 01 00 00       	callq  40099a <_ZN9TestClass15increase_field1Ev>
  4007a8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ad:	48 89 c7             	mov    %rax,%rdi
  4007b0:	e8 4b 04 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
  4007b5:	8b 18                	mov    (%rax),%ebx
  4007b7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bc:	48 89 c7             	mov    %rax,%rdi
  4007bf:	e8 20 04 00 00       	callq  400be4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
  4007c4:	8b 00                	mov    (%rax),%eax
  4007c6:	89 da                	mov    %ebx,%edx
  4007c8:	89 c6                	mov    %eax,%esi
  4007ca:	bf 34 0e 40 00       	mov    $0x400e34,%edi
  4007cf:	b8 00 00 00 00       	mov    $0x0,%eax
  4007d4:	e8 17 fe ff ff       	callq  4005f0 <printf@plt>
  4007d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007de:	48 89 c7             	mov    %rax,%rdi
  4007e1:	e8 fe 03 00 00       	callq  400be4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
  4007e6:	8b 00                	mov    (%rax),%eax
  4007e8:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4007ed:	75 16                	jne    400805 <main+0xbf>
  4007ef:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007f4:	48 89 c7             	mov    %rax,%rdi
  4007f7:	e8 04 04 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
  4007fc:	8b 00                	mov    (%rax),%eax
  4007fe:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400803:	74 07                	je     40080c <main+0xc6>
  400805:	b8 01 00 00 00       	mov    $0x1,%eax
  40080a:	eb 05                	jmp    400811 <main+0xcb>
  40080c:	b8 00 00 00 00       	mov    $0x0,%eax
  400811:	0f b6 c0             	movzbl %al,%eax
  400814:	48 83 c4 10          	add    $0x10,%rsp
  400818:	5b                   	pop    %rbx
  400819:	c3                   	retq   

000000000040081a <_Z12new_instancev>:
  40081a:	53                   	push   %rbx
  40081b:	bf 00 00 00 00       	mov    $0x0,%edi
  400820:	e8 20 03 00 00       	callq  400b45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>
  400825:	48 89 c3             	mov    %rax,%rbx
  400828:	48 89 df             	mov    %rbx,%rdi
  40082b:	e8 f8 01 00 00       	callq  400a28 <_ZN9TestClassC1Ev>
  400830:	48 89 d8             	mov    %rbx,%rax
  400833:	5b                   	pop    %rbx
  400834:	c3                   	retq   

0000000000400835 <_Z12write_field0P9TestClass>:
  400835:	48 83 ec 18          	sub    $0x18,%rsp
  400839:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40083e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400843:	be 77 77 00 00       	mov    $0x7777,%esi
  400848:	48 89 c7             	mov    %rax,%rdi
  40084b:	e8 38 03 00 00       	callq  400b88 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>
  400850:	90                   	nop
  400851:	48 83 c4 18          	add    $0x18,%rsp
  400855:	c3                   	retq   

0000000000400856 <_Z11read_field0P9TestClass>:
  400856:	48 83 ec 18          	sub    $0x18,%rsp
  40085a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40085f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400864:	48 89 c7             	mov    %rax,%rdi
  400867:	e8 78 03 00 00       	callq  400be4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>
  40086c:	8b 00                	mov    (%rax),%eax
  40086e:	48 83 c4 18          	add    $0x18,%rsp
  400872:	c3                   	retq   

0000000000400873 <_Z12write_field1P9TestClass>:
  400873:	48 83 ec 18          	sub    $0x18,%rsp
  400877:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40087c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400881:	be 77 77 00 00       	mov    $0x7777,%esi
  400886:	48 89 c7             	mov    %rax,%rdi
  400889:	e8 28 03 00 00       	callq  400bb6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>
  40088e:	90                   	nop
  40088f:	48 83 c4 18          	add    $0x18,%rsp
  400893:	c3                   	retq   

0000000000400894 <_Z11read_field1P9TestClass>:
  400894:	48 83 ec 18          	sub    $0x18,%rsp
  400898:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40089d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008a2:	48 89 c7             	mov    %rax,%rdi
  4008a5:	e8 56 03 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>
  4008aa:	8b 00                	mov    (%rax),%eax
  4008ac:	48 83 c4 18          	add    $0x18,%rsp
  4008b0:	c3                   	retq   

00000000004008b1 <_Z21explicit_write_field0m>:
  4008b1:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008b6:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008bb:	c7 04 85 00 44 61 00 	movl   $0x7777,0x614400(,%rax,4)
  4008c2:	77 77 00 00 
  4008c6:	90                   	nop
  4008c7:	c3                   	retq   

00000000004008c8 <_Z20explicit_read_field0m>:
  4008c8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008cd:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008d2:	8b 04 85 00 44 61 00 	mov    0x614400(,%rax,4),%eax
  4008d9:	c3                   	retq   

00000000004008da <_Z25explicit_write_field0_aosP16TestClassCompare>:
  4008da:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008df:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008e4:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  4008ea:	90                   	nop
  4008eb:	c3                   	retq   

00000000004008ec <_Z24explicit_read_field0_aosP16TestClassCompare>:
  4008ec:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008f1:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008f6:	8b 00                	mov    (%rax),%eax
  4008f8:	c3                   	retq   

00000000004008f9 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  4008f9:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008fe:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400903:	c7 40 04 77 77 00 00 	movl   $0x7777,0x4(%rax)
  40090a:	90                   	nop
  40090b:	c3                   	retq   

000000000040090c <_Z24explicit_read_field1_aosP16TestClassCompare>:
  40090c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400911:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400916:	8b 40 04             	mov    0x4(%rax),%eax
  400919:	c3                   	retq   

000000000040091a <_ZnwmPv>:
  40091a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40091f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400924:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400929:	c3                   	retq   

000000000040092a <_ZdlPvS_>:
  40092a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40092f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400934:	90                   	nop
  400935:	c3                   	retq   

0000000000400936 <_ZN9TestClass15increase_field0Ev>:
  400936:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40093b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40093f:	41 ff 72 f8          	pushq  -0x8(%r10)
  400943:	55                   	push   %rbp
  400944:	48 89 e5             	mov    %rsp,%rbp
  400947:	41 52                	push   %r10
  400949:	48 83 ec 48          	sub    $0x48,%rsp
  40094d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400951:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400958:	00 00 
  40095a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40095e:	31 c0                	xor    %eax,%eax
  400960:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400964:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400968:	ba 55 55 00 00       	mov    $0x5555,%edx
  40096d:	48 89 ce             	mov    %rcx,%rsi
  400970:	48 89 c7             	mov    %rax,%rdi
  400973:	e8 e6 00 00 00       	callq  400a5e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEmLEi>
  400978:	90                   	nop
  400979:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40097d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400984:	00 00 
  400986:	74 05                	je     40098d <_ZN9TestClass15increase_field0Ev+0x57>
  400988:	e8 83 fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  40098d:	48 83 c4 48          	add    $0x48,%rsp
  400991:	41 5a                	pop    %r10
  400993:	5d                   	pop    %rbp
  400994:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400998:	c3                   	retq   
  400999:	90                   	nop

000000000040099a <_ZN9TestClass15increase_field1Ev>:
  40099a:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40099f:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009a3:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009a7:	55                   	push   %rbp
  4009a8:	48 89 e5             	mov    %rsp,%rbp
  4009ab:	41 52                	push   %r10
  4009ad:	48 83 ec 48          	sub    $0x48,%rsp
  4009b1:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009b5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009bc:	00 00 
  4009be:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009c2:	31 c0                	xor    %eax,%eax
  4009c4:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4009c8:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009cc:	ba 44 44 00 00       	mov    $0x4444,%edx
  4009d1:	48 89 ce             	mov    %rcx,%rsi
  4009d4:	48 89 c7             	mov    %rax,%rdi
  4009d7:	e8 ca 00 00 00       	callq  400aa6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEmLEi>
  4009dc:	90                   	nop
  4009dd:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009e1:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009e8:	00 00 
  4009ea:	74 05                	je     4009f1 <_ZN9TestClass15increase_field1Ev+0x57>
  4009ec:	e8 1f fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  4009f1:	48 83 c4 48          	add    $0x48,%rsp
  4009f5:	41 5a                	pop    %r10
  4009f7:	5d                   	pop    %rbp
  4009f8:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009fc:	c3                   	retq   
  4009fd:	90                   	nop

00000000004009fe <_ZN9TestClass14storage_bufferEv>:
  4009fe:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a03:	c3                   	retq   

0000000000400a04 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400a04:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a09:	90                   	nop
  400a0a:	c3                   	retq   
  400a0b:	90                   	nop

0000000000400a0c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEC1Ev>:
  400a0c:	48 83 ec 18          	sub    $0x18,%rsp
  400a10:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a15:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a1a:	48 89 c7             	mov    %rax,%rdi
  400a1d:	e8 e2 ff ff ff       	callq  400a04 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400a22:	90                   	nop
  400a23:	48 83 c4 18          	add    $0x18,%rsp
  400a27:	c3                   	retq   

0000000000400a28 <_ZN9TestClassC1Ev>:
  400a28:	48 83 ec 18          	sub    $0x18,%rsp
  400a2c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a31:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a36:	48 89 c7             	mov    %rax,%rdi
  400a39:	e8 ce ff ff ff       	callq  400a0c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEC1Ev>
  400a3e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a43:	48 89 c7             	mov    %rax,%rdi
  400a46:	e8 27 01 00 00       	callq  400b72 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC1Ev>
  400a4b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a50:	48 89 c7             	mov    %rax,%rdi
  400a53:	e8 22 01 00 00       	callq  400b7a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC1Ev>
  400a58:	90                   	nop
  400a59:	48 83 c4 18          	add    $0x18,%rsp
  400a5d:	c3                   	retq   

0000000000400a5e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEmLEi>:
  400a5e:	48 83 ec 28          	sub    $0x28,%rsp
  400a62:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400a67:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400a6c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400a70:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a75:	48 89 c7             	mov    %rax,%rdi
  400a78:	e8 9f 01 00 00       	callq  400c1c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400a7d:	8b 10                	mov    (%rax),%edx
  400a7f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400a84:	89 10                	mov    %edx,(%rax)
  400a86:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400a8b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a90:	48 89 d6             	mov    %rdx,%rsi
  400a93:	48 89 c7             	mov    %rax,%rdi
  400a96:	e8 9d 01 00 00       	callq  400c38 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC1ERKS3_>
  400a9b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400aa0:	48 83 c4 28          	add    $0x28,%rsp
  400aa4:	c3                   	retq   
  400aa5:	90                   	nop

0000000000400aa6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEmLEi>:
  400aa6:	48 83 ec 28          	sub    $0x28,%rsp
  400aaa:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400aaf:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400ab4:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ab8:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400abd:	48 89 c7             	mov    %rax,%rdi
  400ac0:	e8 7f 01 00 00       	callq  400c44 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400ac5:	8b 10                	mov    (%rax),%edx
  400ac7:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400acc:	89 10                	mov    %edx,(%rax)
  400ace:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400ad3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ad8:	48 89 d6             	mov    %rdx,%rsi
  400adb:	48 89 c7             	mov    %rax,%rdi
  400ade:	e8 7d 01 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC1ERKS3_>
  400ae3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ae8:	48 83 c4 28          	add    $0x28,%rsp
  400aec:	c3                   	retq   

0000000000400aed <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>:
  400aed:	48 83 ec 08          	sub    $0x8,%rsp
  400af1:	e8 08 ff ff ff       	callq  4009fe <_ZN9TestClass14storage_bufferEv>
  400af6:	48 83 c4 08          	add    $0x8,%rsp
  400afa:	c3                   	retq   

0000000000400afb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIIEEEvDpT_>:
  400afb:	41 54                	push   %r12
  400afd:	55                   	push   %rbp
  400afe:	53                   	push   %rbx
  400aff:	e8 e9 ff ff ff       	callq  400aed <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>
  400b04:	48 89 c5             	mov    %rax,%rbp
  400b07:	48 89 ee             	mov    %rbp,%rsi
  400b0a:	bf d0 91 00 00       	mov    $0x91d0,%edi
  400b0f:	e8 06 fe ff ff       	callq  40091a <_ZnwmPv>
  400b14:	48 89 c3             	mov    %rax,%rbx
  400b17:	48 85 db             	test   %rbx,%rbx
  400b1a:	74 23                	je     400b3f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIIEEEvDpT_+0x44>
  400b1c:	48 89 df             	mov    %rbx,%rdi
  400b1f:	e8 48 01 00 00       	callq  400c6c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400b24:	eb 19                	jmp    400b3f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18initialize_storageIIEEEvDpT_+0x44>
  400b26:	49 89 c4             	mov    %rax,%r12
  400b29:	48 89 ee             	mov    %rbp,%rsi
  400b2c:	48 89 df             	mov    %rbx,%rdi
  400b2f:	e8 f6 fd ff ff       	callq  40092a <_ZdlPvS_>
  400b34:	4c 89 e0             	mov    %r12,%rax
  400b37:	48 89 c7             	mov    %rax,%rdi
  400b3a:	e8 f1 fa ff ff       	callq  400630 <_Unwind_Resume@plt>
  400b3f:	90                   	nop
  400b40:	5b                   	pop    %rbx
  400b41:	5d                   	pop    %rbp
  400b42:	41 5c                	pop    %r12
  400b44:	c3                   	retq   

0000000000400b45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEnwEm>:
  400b45:	48 83 ec 18          	sub    $0x18,%rsp
  400b49:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b4e:	e8 41 01 00 00       	callq  400c94 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400b53:	e8 95 ff ff ff       	callq  400aed <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE7storageEv>
  400b58:	be 01 00 00 00       	mov    $0x1,%esi
  400b5d:	48 89 c7             	mov    %rax,%rdi
  400b60:	e8 51 01 00 00       	callq  400cb6 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400b65:	48 89 c7             	mov    %rax,%rdi
  400b68:	e8 29 01 00 00       	callq  400c96 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE3getEm>
  400b6d:	48 83 c4 18          	add    $0x18,%rsp
  400b71:	c3                   	retq   

0000000000400b72 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC1Ev>:
  400b72:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b77:	90                   	nop
  400b78:	c3                   	retq   
  400b79:	90                   	nop

0000000000400b7a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC1Ev>:
  400b7a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b7f:	90                   	nop
  400b80:	c3                   	retq   

0000000000400b81 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EEdlEPv>:
  400b81:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b86:	90                   	nop
  400b87:	c3                   	retq   

0000000000400b88 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEaSEi>:
  400b88:	48 83 ec 18          	sub    $0x18,%rsp
  400b8c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b91:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400b95:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b9a:	48 89 c7             	mov    %rax,%rdi
  400b9d:	e8 7a 00 00 00       	callq  400c1c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400ba2:	48 89 c2             	mov    %rax,%rdx
  400ba5:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400ba9:	89 02                	mov    %eax,(%rdx)
  400bab:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bb0:	48 83 c4 18          	add    $0x18,%rsp
  400bb4:	c3                   	retq   
  400bb5:	90                   	nop

0000000000400bb6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEaSEi>:
  400bb6:	48 83 ec 18          	sub    $0x18,%rsp
  400bba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bbf:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400bc3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bc8:	48 89 c7             	mov    %rax,%rdi
  400bcb:	e8 74 00 00 00       	callq  400c44 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400bd0:	48 89 c2             	mov    %rax,%rdx
  400bd3:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400bd7:	89 02                	mov    %eax,(%rdx)
  400bd9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bde:	48 83 c4 18          	add    $0x18,%rsp
  400be2:	c3                   	retq   
  400be3:	90                   	nop

0000000000400be4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEcvRiEv>:
  400be4:	48 83 ec 18          	sub    $0x18,%rsp
  400be8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bed:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bf2:	48 89 c7             	mov    %rax,%rdi
  400bf5:	e8 22 00 00 00       	callq  400c1c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400bfa:	48 83 c4 18          	add    $0x18,%rsp
  400bfe:	c3                   	retq   
  400bff:	90                   	nop

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEcvRiEv>:
  400c00:	48 83 ec 18          	sub    $0x18,%rsp
  400c04:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c09:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c0e:	48 89 c7             	mov    %rax,%rdi
  400c11:	e8 2e 00 00 00       	callq  400c44 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>
  400c16:	48 83 c4 18          	add    $0x18,%rsp
  400c1a:	c3                   	retq   
  400c1b:	90                   	nop

0000000000400c1c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE8data_ptrEv>:
  400c1c:	48 83 ec 18          	sub    $0x18,%rsp
  400c20:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c25:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c2a:	48 89 c7             	mov    %rax,%rdi
  400c2d:	e8 be 00 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>
  400c32:	48 83 c4 18          	add    $0x18,%rsp
  400c36:	c3                   	retq   
  400c37:	90                   	nop

0000000000400c38 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassEC1ERKS3_>:
  400c38:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c3d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c42:	90                   	nop
  400c43:	c3                   	retq   

0000000000400c44 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE8data_ptrEv>:
  400c44:	48 83 ec 18          	sub    $0x18,%rsp
  400c48:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c4d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c52:	48 89 c7             	mov    %rax,%rdi
  400c55:	e8 a2 00 00 00       	callq  400cfc <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>
  400c5a:	48 83 c4 18          	add    $0x18,%rsp
  400c5e:	c3                   	retq   
  400c5f:	90                   	nop

0000000000400c60 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassEC1ERKS3_>:
  400c60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c65:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c6a:	90                   	nop
  400c6b:	c3                   	retq   

0000000000400c6c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400c6c:	48 83 ec 18          	sub    $0x18,%rsp
  400c70:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c75:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c7a:	48 89 c7             	mov    %rax,%rdi
  400c7d:	e8 8a 00 00 00       	callq  400d0c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400c82:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c87:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400c8e:	90                   	nop
  400c8f:	48 83 c4 18          	add    $0x18,%rsp
  400c93:	c3                   	retq   

0000000000400c94 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c94:	90                   	nop
  400c95:	c3                   	retq   

0000000000400c96 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE3getEm>:
  400c96:	48 83 ec 18          	sub    $0x18,%rsp
  400c9a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c9f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca4:	48 83 c0 01          	add    $0x1,%rax
  400ca8:	48 89 c7             	mov    %rax,%rdi
  400cab:	e8 6f 00 00 00       	callq  400d1f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE4get_ILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeAosEEEPS2_E4typeEm>
  400cb0:	48 83 c4 18          	add    $0x18,%rsp
  400cb4:	c3                   	retq   
  400cb5:	90                   	nop

0000000000400cb6 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400cb6:	48 83 ec 28          	sub    $0x28,%rsp
  400cba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cbf:	48 89 34 24          	mov    %rsi,(%rsp)
  400cc3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc8:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400ccd:	48 8b 04 24          	mov    (%rsp),%rax
  400cd1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400cd6:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  400cdb:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400ce0:	48 89 d6             	mov    %rdx,%rsi
  400ce3:	48 89 c7             	mov    %rax,%rdi
  400ce6:	e8 6a 00 00 00       	callq  400d55 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400ceb:	48 83 c4 28          	add    $0x28,%rsp
  400cef:	c3                   	retq   

0000000000400cf0 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>:
  400cf0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cf5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400cfa:	c3                   	retq   
  400cfb:	90                   	nop

0000000000400cfc <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0ELi0EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEeqT1_L_ZNS0_L14kLayoutModeAosEEEPiE4typeEv>:
  400cfc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d01:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d06:	48 83 c0 04          	add    $0x4,%rax
  400d0a:	c3                   	retq   
  400d0b:	90                   	nop

0000000000400d0c <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400d0c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d11:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d16:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400d1d:	90                   	nop
  400d1e:	c3                   	retq   

0000000000400d1f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageELi0EE4get_ILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeAosEEEPS2_E4typeEm>:
  400d1f:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400d24:	48 c7 44 24 e8 10 00 	movq   $0x10,-0x18(%rsp)
  400d2b:	00 00 
  400d2d:	48 c7 44 24 f0 80 20 	movq   $0x602080,-0x10(%rsp)
  400d34:	60 00 
  400d36:	48 c7 44 24 f8 90 20 	movq   $0x602090,-0x8(%rsp)
  400d3d:	60 00 
  400d3f:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400d44:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
  400d4b:	00 
  400d4c:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d51:	48 01 d0             	add    %rdx,%rax
  400d54:	c3                   	retq   

0000000000400d55 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400d55:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d5a:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400d5f:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d64:	48 8b 00             	mov    (%rax),%rax
  400d67:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400d6c:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d71:	48 8b 10             	mov    (%rax),%rdx
  400d74:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400d79:	48 01 c2             	add    %rax,%rdx
  400d7c:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d81:	48 89 10             	mov    %rdx,(%rax)
  400d84:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d89:	c3                   	retq   
  400d8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400d90 <__libc_csu_init>:
  400d90:	41 57                	push   %r15
  400d92:	41 56                	push   %r14
  400d94:	41 89 ff             	mov    %edi,%r15d
  400d97:	41 55                	push   %r13
  400d99:	41 54                	push   %r12
  400d9b:	4c 8d 25 4e 10 20 00 	lea    0x20104e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400da2:	55                   	push   %rbp
  400da3:	48 8d 2d 4e 10 20 00 	lea    0x20104e(%rip),%rbp        # 601df8 <__init_array_end>
  400daa:	53                   	push   %rbx
  400dab:	49 89 f6             	mov    %rsi,%r14
  400dae:	49 89 d5             	mov    %rdx,%r13
  400db1:	4c 29 e5             	sub    %r12,%rbp
  400db4:	48 83 ec 08          	sub    $0x8,%rsp
  400db8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dbc:	e8 ff f7 ff ff       	callq  4005c0 <_init>
  400dc1:	48 85 ed             	test   %rbp,%rbp
  400dc4:	74 20                	je     400de6 <__libc_csu_init+0x56>
  400dc6:	31 db                	xor    %ebx,%ebx
  400dc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dcf:	00 
  400dd0:	4c 89 ea             	mov    %r13,%rdx
  400dd3:	4c 89 f6             	mov    %r14,%rsi
  400dd6:	44 89 ff             	mov    %r15d,%edi
  400dd9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ddd:	48 83 c3 01          	add    $0x1,%rbx
  400de1:	48 39 eb             	cmp    %rbp,%rbx
  400de4:	75 ea                	jne    400dd0 <__libc_csu_init+0x40>
  400de6:	48 83 c4 08          	add    $0x8,%rsp
  400dea:	5b                   	pop    %rbx
  400deb:	5d                   	pop    %rbp
  400dec:	41 5c                	pop    %r12
  400dee:	41 5d                	pop    %r13
  400df0:	41 5e                	pop    %r14
  400df2:	41 5f                	pop    %r15
  400df4:	c3                   	retq   
  400df5:	90                   	nop
  400df6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dfd:	00 00 00 

0000000000400e00 <__libc_csu_fini>:
  400e00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e04 <_fini>:
  400e04:	48 83 ec 08          	sub    $0x8,%rsp
  400e08:	48 83 c4 08          	add    $0x8,%rsp
  400e0c:	c3                   	retq   
