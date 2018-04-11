
bin/g++_-O0_StaticStorage_0:     file format elf64-x86-64


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
  40065f:	49 c7 c0 c0 0d 40 00 	mov    $0x400dc0,%r8
  400666:	48 c7 c1 50 0d 40 00 	mov    $0x400d50,%rcx
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
  40074b:	e8 6b 03 00 00       	callq  400abb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>
  400750:	bf 00 00 00 00       	mov    $0x0,%edi
  400755:	e8 ab 03 00 00       	callq  400b05 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  40075a:	48 89 c3             	mov    %rax,%rbx
  40075d:	48 89 df             	mov    %rbx,%rdi
  400760:	e8 83 02 00 00       	callq  4009e8 <_ZN9TestClassC1Ev>
  400765:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  40076a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40076f:	be 77 77 00 00       	mov    $0x7777,%esi
  400774:	48 89 c7             	mov    %rax,%rdi
  400777:	e8 cc 03 00 00       	callq  400b48 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>
  40077c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400781:	be 88 88 00 00       	mov    $0x8888,%esi
  400786:	48 89 c7             	mov    %rax,%rdi
  400789:	e8 e8 03 00 00       	callq  400b76 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>
  40078e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400793:	48 89 c7             	mov    %rax,%rdi
  400796:	e8 5b 01 00 00       	callq  4008f6 <_ZN9TestClass15increase_field0Ev>
  40079b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007a0:	48 89 c7             	mov    %rax,%rdi
  4007a3:	e8 b2 01 00 00       	callq  40095a <_ZN9TestClass15increase_field1Ev>
  4007a8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ad:	48 89 c7             	mov    %rax,%rdi
  4007b0:	e8 0b 04 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
  4007b5:	8b 18                	mov    (%rax),%ebx
  4007b7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bc:	48 89 c7             	mov    %rax,%rdi
  4007bf:	e8 e0 03 00 00       	callq  400ba4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  4007c4:	8b 00                	mov    (%rax),%eax
  4007c6:	89 da                	mov    %ebx,%edx
  4007c8:	89 c6                	mov    %eax,%esi
  4007ca:	bf ec 0d 40 00       	mov    $0x400dec,%edi
  4007cf:	b8 00 00 00 00       	mov    $0x0,%eax
  4007d4:	e8 17 fe ff ff       	callq  4005f0 <printf@plt>
  4007d9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007de:	48 89 c7             	mov    %rax,%rdi
  4007e1:	e8 be 03 00 00       	callq  400ba4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  4007e6:	8b 00                	mov    (%rax),%eax
  4007e8:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  4007ed:	75 16                	jne    400805 <main+0xbf>
  4007ef:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007f4:	48 89 c7             	mov    %rax,%rdi
  4007f7:	e8 c4 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
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
  400820:	e8 e0 02 00 00       	callq  400b05 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>
  400825:	48 89 c3             	mov    %rax,%rbx
  400828:	48 89 df             	mov    %rbx,%rdi
  40082b:	e8 b8 01 00 00       	callq  4009e8 <_ZN9TestClassC1Ev>
  400830:	48 89 d8             	mov    %rbx,%rax
  400833:	5b                   	pop    %rbx
  400834:	c3                   	retq   

0000000000400835 <_Z12write_field0P9TestClass>:
  400835:	48 83 ec 18          	sub    $0x18,%rsp
  400839:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40083e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400843:	be 77 77 00 00       	mov    $0x7777,%esi
  400848:	48 89 c7             	mov    %rax,%rdi
  40084b:	e8 f8 02 00 00       	callq  400b48 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>
  400850:	90                   	nop
  400851:	48 83 c4 18          	add    $0x18,%rsp
  400855:	c3                   	retq   

0000000000400856 <_Z11read_field0P9TestClass>:
  400856:	48 83 ec 18          	sub    $0x18,%rsp
  40085a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40085f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400864:	48 89 c7             	mov    %rax,%rdi
  400867:	e8 38 03 00 00       	callq  400ba4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>
  40086c:	8b 00                	mov    (%rax),%eax
  40086e:	48 83 c4 18          	add    $0x18,%rsp
  400872:	c3                   	retq   

0000000000400873 <_Z12write_field1P9TestClass>:
  400873:	48 83 ec 18          	sub    $0x18,%rsp
  400877:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40087c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400881:	be 77 77 00 00       	mov    $0x7777,%esi
  400886:	48 89 c7             	mov    %rax,%rdi
  400889:	e8 e8 02 00 00       	callq  400b76 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>
  40088e:	90                   	nop
  40088f:	48 83 c4 18          	add    $0x18,%rsp
  400893:	c3                   	retq   

0000000000400894 <_Z11read_field1P9TestClass>:
  400894:	48 83 ec 18          	sub    $0x18,%rsp
  400898:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40089d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008a2:	48 89 c7             	mov    %rax,%rdi
  4008a5:	e8 16 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>
  4008aa:	8b 00                	mov    (%rax),%eax
  4008ac:	48 83 c4 18          	add    $0x18,%rsp
  4008b0:	c3                   	retq   

00000000004008b1 <_Z21explicit_write_field0m>:
  4008b1:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008b6:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008bb:	c7 04 85 60 b2 60 00 	movl   $0x7777,0x60b260(,%rax,4)
  4008c2:	77 77 00 00 
  4008c6:	90                   	nop
  4008c7:	c3                   	retq   

00000000004008c8 <_Z20explicit_read_field0m>:
  4008c8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008cd:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4008d2:	8b 04 85 60 b2 60 00 	mov    0x60b260(,%rax,4),%eax
  4008d9:	c3                   	retq   

00000000004008da <_ZnwmPv>:
  4008da:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008df:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4008e4:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  4008e9:	c3                   	retq   

00000000004008ea <_ZdlPvS_>:
  4008ea:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4008ef:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4008f4:	90                   	nop
  4008f5:	c3                   	retq   

00000000004008f6 <_ZN9TestClass15increase_field0Ev>:
  4008f6:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4008fb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4008ff:	41 ff 72 f8          	pushq  -0x8(%r10)
  400903:	55                   	push   %rbp
  400904:	48 89 e5             	mov    %rsp,%rbp
  400907:	41 52                	push   %r10
  400909:	48 83 ec 48          	sub    $0x48,%rsp
  40090d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400911:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400918:	00 00 
  40091a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40091e:	31 c0                	xor    %eax,%eax
  400920:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400924:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400928:	ba 55 55 00 00       	mov    $0x5555,%edx
  40092d:	48 89 ce             	mov    %rcx,%rsi
  400930:	48 89 c7             	mov    %rax,%rdi
  400933:	e8 e6 00 00 00       	callq  400a1e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEmLEi>
  400938:	90                   	nop
  400939:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40093d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400944:	00 00 
  400946:	74 05                	je     40094d <_ZN9TestClass15increase_field0Ev+0x57>
  400948:	e8 c3 fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  40094d:	48 83 c4 48          	add    $0x48,%rsp
  400951:	41 5a                	pop    %r10
  400953:	5d                   	pop    %rbp
  400954:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400958:	c3                   	retq   
  400959:	90                   	nop

000000000040095a <_ZN9TestClass15increase_field1Ev>:
  40095a:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40095f:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400963:	41 ff 72 f8          	pushq  -0x8(%r10)
  400967:	55                   	push   %rbp
  400968:	48 89 e5             	mov    %rsp,%rbp
  40096b:	41 52                	push   %r10
  40096d:	48 83 ec 48          	sub    $0x48,%rsp
  400971:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400975:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40097c:	00 00 
  40097e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400982:	31 c0                	xor    %eax,%eax
  400984:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400988:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  40098c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400991:	48 89 ce             	mov    %rcx,%rsi
  400994:	48 89 c7             	mov    %rax,%rdi
  400997:	e8 ca 00 00 00       	callq  400a66 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEmLEi>
  40099c:	90                   	nop
  40099d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009a1:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009a8:	00 00 
  4009aa:	74 05                	je     4009b1 <_ZN9TestClass15increase_field1Ev+0x57>
  4009ac:	e8 5f fc ff ff       	callq  400610 <__stack_chk_fail@plt>
  4009b1:	48 83 c4 48          	add    $0x48,%rsp
  4009b5:	41 5a                	pop    %r10
  4009b7:	5d                   	pop    %rbp
  4009b8:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009bc:	c3                   	retq   
  4009bd:	90                   	nop

00000000004009be <_ZN9TestClass14storage_bufferEv>:
  4009be:	b8 80 20 60 00       	mov    $0x602080,%eax
  4009c3:	c3                   	retq   

00000000004009c4 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  4009c4:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009c9:	90                   	nop
  4009ca:	c3                   	retq   
  4009cb:	90                   	nop

00000000004009cc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>:
  4009cc:	48 83 ec 18          	sub    $0x18,%rsp
  4009d0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009d5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009da:	48 89 c7             	mov    %rax,%rdi
  4009dd:	e8 e2 ff ff ff       	callq  4009c4 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  4009e2:	90                   	nop
  4009e3:	48 83 c4 18          	add    $0x18,%rsp
  4009e7:	c3                   	retq   

00000000004009e8 <_ZN9TestClassC1Ev>:
  4009e8:	48 83 ec 18          	sub    $0x18,%rsp
  4009ec:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009f1:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4009f6:	48 89 c7             	mov    %rax,%rdi
  4009f9:	e8 ce ff ff ff       	callq  4009cc <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEC1Ev>
  4009fe:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a03:	48 89 c7             	mov    %rax,%rdi
  400a06:	e8 27 01 00 00       	callq  400b32 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1Ev>
  400a0b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a10:	48 89 c7             	mov    %rax,%rdi
  400a13:	e8 22 01 00 00       	callq  400b3a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1Ev>
  400a18:	90                   	nop
  400a19:	48 83 c4 18          	add    $0x18,%rsp
  400a1d:	c3                   	retq   

0000000000400a1e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEmLEi>:
  400a1e:	48 83 ec 28          	sub    $0x28,%rsp
  400a22:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400a27:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400a2c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400a30:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a35:	48 89 c7             	mov    %rax,%rdi
  400a38:	e8 9f 01 00 00       	callq  400bdc <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a3d:	8b 10                	mov    (%rax),%edx
  400a3f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400a44:	89 10                	mov    %edx,(%rax)
  400a46:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400a4b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a50:	48 89 d6             	mov    %rdx,%rsi
  400a53:	48 89 c7             	mov    %rax,%rdi
  400a56:	e8 9d 01 00 00       	callq  400bf8 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1ERKS3_>
  400a5b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a60:	48 83 c4 28          	add    $0x28,%rsp
  400a64:	c3                   	retq   
  400a65:	90                   	nop

0000000000400a66 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEmLEi>:
  400a66:	48 83 ec 28          	sub    $0x28,%rsp
  400a6a:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400a6f:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400a74:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400a78:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a7d:	48 89 c7             	mov    %rax,%rdi
  400a80:	e8 7f 01 00 00       	callq  400c04 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a85:	8b 10                	mov    (%rax),%edx
  400a87:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400a8c:	89 10                	mov    %edx,(%rax)
  400a8e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400a93:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400a98:	48 89 d6             	mov    %rdx,%rsi
  400a9b:	48 89 c7             	mov    %rax,%rdi
  400a9e:	e8 7d 01 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1ERKS3_>
  400aa3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400aa8:	48 83 c4 28          	add    $0x28,%rsp
  400aac:	c3                   	retq   

0000000000400aad <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>:
  400aad:	48 83 ec 08          	sub    $0x8,%rsp
  400ab1:	e8 08 ff ff ff       	callq  4009be <_ZN9TestClass14storage_bufferEv>
  400ab6:	48 83 c4 08          	add    $0x8,%rsp
  400aba:	c3                   	retq   

0000000000400abb <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_>:
  400abb:	41 54                	push   %r12
  400abd:	55                   	push   %rbp
  400abe:	53                   	push   %rbx
  400abf:	e8 e9 ff ff ff       	callq  400aad <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400ac4:	48 89 c5             	mov    %rax,%rbp
  400ac7:	48 89 ee             	mov    %rbp,%rsi
  400aca:	bf d0 91 00 00       	mov    $0x91d0,%edi
  400acf:	e8 06 fe ff ff       	callq  4008da <_ZnwmPv>
  400ad4:	48 89 c3             	mov    %rax,%rbx
  400ad7:	48 85 db             	test   %rbx,%rbx
  400ada:	74 23                	je     400aff <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400adc:	48 89 df             	mov    %rbx,%rdi
  400adf:	e8 48 01 00 00       	callq  400c2c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>
  400ae4:	eb 19                	jmp    400aff <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400ae6:	49 89 c4             	mov    %rax,%r12
  400ae9:	48 89 ee             	mov    %rbp,%rsi
  400aec:	48 89 df             	mov    %rbx,%rdi
  400aef:	e8 f6 fd ff ff       	callq  4008ea <_ZdlPvS_>
  400af4:	4c 89 e0             	mov    %r12,%rax
  400af7:	48 89 c7             	mov    %rax,%rdi
  400afa:	e8 31 fb ff ff       	callq  400630 <_Unwind_Resume@plt>
  400aff:	90                   	nop
  400b00:	5b                   	pop    %rbx
  400b01:	5d                   	pop    %rbp
  400b02:	41 5c                	pop    %r12
  400b04:	c3                   	retq   

0000000000400b05 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEnwEm>:
  400b05:	48 83 ec 18          	sub    $0x18,%rsp
  400b09:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b0e:	e8 41 01 00 00       	callq  400c54 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400b13:	e8 95 ff ff ff       	callq  400aad <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE7storageEv>
  400b18:	be 01 00 00 00       	mov    $0x1,%esi
  400b1d:	48 89 c7             	mov    %rax,%rdi
  400b20:	e8 51 01 00 00       	callq  400c76 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400b25:	48 89 c7             	mov    %rax,%rdi
  400b28:	e8 29 01 00 00       	callq  400c56 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>
  400b2d:	48 83 c4 18          	add    $0x18,%rsp
  400b31:	c3                   	retq   

0000000000400b32 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1Ev>:
  400b32:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b37:	90                   	nop
  400b38:	c3                   	retq   
  400b39:	90                   	nop

0000000000400b3a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1Ev>:
  400b3a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b3f:	90                   	nop
  400b40:	c3                   	retq   

0000000000400b41 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEEdlEPv>:
  400b41:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b46:	90                   	nop
  400b47:	c3                   	retq   

0000000000400b48 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEaSEi>:
  400b48:	48 83 ec 18          	sub    $0x18,%rsp
  400b4c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b51:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400b55:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b5a:	48 89 c7             	mov    %rax,%rdi
  400b5d:	e8 7a 00 00 00       	callq  400bdc <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b62:	48 89 c2             	mov    %rax,%rdx
  400b65:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400b69:	89 02                	mov    %eax,(%rdx)
  400b6b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b70:	48 83 c4 18          	add    $0x18,%rsp
  400b74:	c3                   	retq   
  400b75:	90                   	nop

0000000000400b76 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEaSEi>:
  400b76:	48 83 ec 18          	sub    $0x18,%rsp
  400b7a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b7f:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400b83:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b88:	48 89 c7             	mov    %rax,%rdi
  400b8b:	e8 74 00 00 00       	callq  400c04 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b90:	48 89 c2             	mov    %rax,%rdx
  400b93:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400b97:	89 02                	mov    %eax,(%rdx)
  400b99:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400b9e:	48 83 c4 18          	add    $0x18,%rsp
  400ba2:	c3                   	retq   
  400ba3:	90                   	nop

0000000000400ba4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEcvRiEv>:
  400ba4:	48 83 ec 18          	sub    $0x18,%rsp
  400ba8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bad:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bb2:	48 89 c7             	mov    %rax,%rdi
  400bb5:	e8 22 00 00 00       	callq  400bdc <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bba:	48 83 c4 18          	add    $0x18,%rsp
  400bbe:	c3                   	retq   
  400bbf:	90                   	nop

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEcvRiEv>:
  400bc0:	48 83 ec 18          	sub    $0x18,%rsp
  400bc4:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bc9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bce:	48 89 c7             	mov    %rax,%rdi
  400bd1:	e8 2e 00 00 00       	callq  400c04 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd6:	48 83 c4 18          	add    $0x18,%rsp
  400bda:	c3                   	retq   
  400bdb:	90                   	nop

0000000000400bdc <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bdc:	48 83 ec 18          	sub    $0x18,%rsp
  400be0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400be5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bea:	48 89 c7             	mov    %rax,%rdi
  400bed:	e8 ba 00 00 00       	callq  400cac <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400bf2:	48 83 c4 18          	add    $0x18,%rsp
  400bf6:	c3                   	retq   
  400bf7:	90                   	nop

0000000000400bf8 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassEC1ERKS3_>:
  400bf8:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bfd:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c02:	90                   	nop
  400c03:	c3                   	retq   

0000000000400c04 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c04:	48 83 ec 18          	sub    $0x18,%rsp
  400c08:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c0d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c12:	48 89 c7             	mov    %rax,%rdi
  400c15:	e8 d0 00 00 00       	callq  400cea <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>
  400c1a:	48 83 c4 18          	add    $0x18,%rsp
  400c1e:	c3                   	retq   
  400c1f:	90                   	nop

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassEC1ERKS3_>:
  400c20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c2a:	90                   	nop
  400c2b:	c3                   	retq   

0000000000400c2c <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC1Ev>:
  400c2c:	48 83 ec 18          	sub    $0x18,%rsp
  400c30:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c35:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c3a:	48 89 c7             	mov    %rax,%rdi
  400c3d:	e8 e8 00 00 00       	callq  400d2a <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>
  400c42:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c47:	c7 40 08 00 00 00 00 	movl   $0x0,0x8(%rax)
  400c4e:	90                   	nop
  400c4f:	48 83 c4 18          	add    $0x18,%rsp
  400c53:	c3                   	retq   

0000000000400c54 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c54:	90                   	nop
  400c55:	c3                   	retq   

0000000000400c56 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE3getEm>:
  400c56:	48 83 ec 18          	sub    $0x18,%rsp
  400c5a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c5f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c64:	48 83 c0 01          	add    $0x1,%rax
  400c68:	48 89 c7             	mov    %rax,%rdi
  400c6b:	e8 cd 00 00 00       	callq  400d3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400c70:	48 83 c4 18          	add    $0x18,%rsp
  400c74:	c3                   	retq   
  400c75:	90                   	nop

0000000000400c76 <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400c76:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400c7b:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400c80:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400c85:	48 8b 00             	mov    (%rax),%rax
  400c88:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400c8d:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400c92:	48 8b 10             	mov    (%rax),%rdx
  400c95:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400c9a:	48 01 c2             	add    %rax,%rdx
  400c9d:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ca2:	48 89 10             	mov    %rdx,(%rax)
  400ca5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400caa:	c3                   	retq   
  400cab:	90                   	nop

0000000000400cac <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400cac:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400cb1:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400cb8:	00 00 
  400cba:	48 c7 44 24 e8 80 20 	movq   $0x602080,-0x18(%rsp)
  400cc1:	60 00 
  400cc3:	48 c7 44 24 f0 90 20 	movq   $0x602090,-0x10(%rsp)
  400cca:	60 00 
  400ccc:	48 c7 44 24 f8 90 20 	movq   $0x602090,-0x8(%rsp)
  400cd3:	60 00 
  400cd5:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400cda:	48 83 c0 04          	add    $0x4,%rax
  400cde:	48 c1 e0 02          	shl    $0x2,%rax
  400ce2:	48 05 80 20 60 00    	add    $0x602080,%rax
  400ce8:	c3                   	retq   
  400ce9:	90                   	nop

0000000000400cea <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi0E9TestClassE22data_ptr_uninitializedILi0ELi0EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L18kStorageModeStaticEEEPiE4typeEv>:
  400cea:	48 89 7c 24 d8       	mov    %rdi,-0x28(%rsp)
  400cef:	48 c7 44 24 e0 10 00 	movq   $0x10,-0x20(%rsp)
  400cf6:	00 00 
  400cf8:	48 c7 44 24 e8 80 20 	movq   $0x602080,-0x18(%rsp)
  400cff:	60 00 
  400d01:	48 c7 44 24 f0 70 69 	movq   $0x606970,-0x10(%rsp)
  400d08:	60 00 
  400d0a:	48 c7 44 24 f8 70 69 	movq   $0x606970,-0x8(%rsp)
  400d11:	60 00 
  400d13:	48 8b 44 24 d8       	mov    -0x28(%rsp),%rax
  400d18:	48 05 3c 12 00 00    	add    $0x123c,%rax
  400d1e:	48 c1 e0 02          	shl    $0x2,%rax
  400d22:	48 05 80 20 60 00    	add    $0x602080,%rax
  400d28:	c3                   	retq   
  400d29:	90                   	nop

0000000000400d2a <_ZN4ikra3soa19StorageStrategySelfINS0_14StaticStorage_I9TestClassLm0EEEEC1Ev>:
  400d2a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d2f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d34:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400d3b:	90                   	nop
  400d3c:	c3                   	retq   

0000000000400d3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_13StaticStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d3d:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d42:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d47:	c3                   	retq   
  400d48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d4f:	00 

0000000000400d50 <__libc_csu_init>:
  400d50:	41 57                	push   %r15
  400d52:	41 56                	push   %r14
  400d54:	41 89 ff             	mov    %edi,%r15d
  400d57:	41 55                	push   %r13
  400d59:	41 54                	push   %r12
  400d5b:	4c 8d 25 8e 10 20 00 	lea    0x20108e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400d62:	55                   	push   %rbp
  400d63:	48 8d 2d 8e 10 20 00 	lea    0x20108e(%rip),%rbp        # 601df8 <__init_array_end>
  400d6a:	53                   	push   %rbx
  400d6b:	49 89 f6             	mov    %rsi,%r14
  400d6e:	49 89 d5             	mov    %rdx,%r13
  400d71:	4c 29 e5             	sub    %r12,%rbp
  400d74:	48 83 ec 08          	sub    $0x8,%rsp
  400d78:	48 c1 fd 03          	sar    $0x3,%rbp
  400d7c:	e8 3f f8 ff ff       	callq  4005c0 <_init>
  400d81:	48 85 ed             	test   %rbp,%rbp
  400d84:	74 20                	je     400da6 <__libc_csu_init+0x56>
  400d86:	31 db                	xor    %ebx,%ebx
  400d88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d8f:	00 
  400d90:	4c 89 ea             	mov    %r13,%rdx
  400d93:	4c 89 f6             	mov    %r14,%rsi
  400d96:	44 89 ff             	mov    %r15d,%edi
  400d99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d9d:	48 83 c3 01          	add    $0x1,%rbx
  400da1:	48 39 eb             	cmp    %rbp,%rbx
  400da4:	75 ea                	jne    400d90 <__libc_csu_init+0x40>
  400da6:	48 83 c4 08          	add    $0x8,%rsp
  400daa:	5b                   	pop    %rbx
  400dab:	5d                   	pop    %rbp
  400dac:	41 5c                	pop    %r12
  400dae:	41 5d                	pop    %r13
  400db0:	41 5e                	pop    %r14
  400db2:	41 5f                	pop    %r15
  400db4:	c3                   	retq   
  400db5:	90                   	nop
  400db6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dbd:	00 00 00 

0000000000400dc0 <__libc_csu_fini>:
  400dc0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400dc4 <_fini>:
  400dc4:	48 83 ec 08          	sub    $0x8,%rsp
  400dc8:	48 83 c4 08          	add    $0x8,%rsp
  400dcc:	c3                   	retq   
