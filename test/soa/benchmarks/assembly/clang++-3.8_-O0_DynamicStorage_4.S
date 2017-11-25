
bin/clang++-3.8_-O0_DynamicStorage_4:     file format elf64-x86-64


Disassembly of section .init:

00000000004005a8 <_init>:
  4005a8:	48 83 ec 08          	sub    $0x8,%rsp
  4005ac:	48 8b 05 45 1a 20 00 	mov    0x201a45(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  4005b3:	48 85 c0             	test   %rax,%rax
  4005b6:	74 05                	je     4005bd <_init+0x15>
  4005b8:	e8 73 00 00 00       	callq  400630 <_Unwind_Resume@plt+0x10>
  4005bd:	48 83 c4 08          	add    $0x8,%rsp
  4005c1:	c3                   	retq   

Disassembly of section .plt:

00000000004005d0 <printf@plt-0x10>:
  4005d0:	ff 35 32 1a 20 00    	pushq  0x201a32(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  4005d6:	ff 25 34 1a 20 00    	jmpq   *0x201a34(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  4005dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005e0 <printf@plt>:
  4005e0:	ff 25 32 1a 20 00    	jmpq   *0x201a32(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005e6:	68 00 00 00 00       	pushq  $0x0
  4005eb:	e9 e0 ff ff ff       	jmpq   4005d0 <_init+0x28>

00000000004005f0 <malloc@plt>:
  4005f0:	ff 25 2a 1a 20 00    	jmpq   *0x201a2a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005f6:	68 01 00 00 00       	pushq  $0x1
  4005fb:	e9 d0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400600 <__libc_start_main@plt>:
  400600:	ff 25 22 1a 20 00    	jmpq   *0x201a22(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400606:	68 02 00 00 00       	pushq  $0x2
  40060b:	e9 c0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400610 <__gxx_personality_v0@plt>:
  400610:	ff 25 1a 1a 20 00    	jmpq   *0x201a1a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400616:	68 03 00 00 00       	pushq  $0x3
  40061b:	e9 b0 ff ff ff       	jmpq   4005d0 <_init+0x28>

0000000000400620 <_Unwind_Resume@plt>:
  400620:	ff 25 12 1a 20 00    	jmpq   *0x201a12(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400626:	68 04 00 00 00       	pushq  $0x4
  40062b:	e9 a0 ff ff ff       	jmpq   4005d0 <_init+0x28>

Disassembly of section .plt.got:

0000000000400630 <.plt.got>:
  400630:	ff 25 c2 19 20 00    	jmpq   *0x2019c2(%rip)        # 601ff8 <_DYNAMIC+0x200>
  400636:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400640 <_start>:
  400640:	31 ed                	xor    %ebp,%ebp
  400642:	49 89 d1             	mov    %rdx,%r9
  400645:	5e                   	pop    %rsi
  400646:	48 89 e2             	mov    %rsp,%rdx
  400649:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40064d:	50                   	push   %rax
  40064e:	54                   	push   %rsp
  40064f:	49 c7 c0 20 0f 40 00 	mov    $0x400f20,%r8
  400656:	48 c7 c1 b0 0e 40 00 	mov    $0x400eb0,%rcx
  40065d:	48 c7 c7 40 07 40 00 	mov    $0x400740,%rdi
  400664:	e8 97 ff ff ff       	callq  400600 <__libc_start_main@plt>
  400669:	f4                   	hlt    
  40066a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400670 <deregister_tm_clones>:
  400670:	b8 57 20 60 00       	mov    $0x602057,%eax
  400675:	55                   	push   %rbp
  400676:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  40067c:	48 83 f8 0e          	cmp    $0xe,%rax
  400680:	48 89 e5             	mov    %rsp,%rbp
  400683:	76 1b                	jbe    4006a0 <deregister_tm_clones+0x30>
  400685:	b8 00 00 00 00       	mov    $0x0,%eax
  40068a:	48 85 c0             	test   %rax,%rax
  40068d:	74 11                	je     4006a0 <deregister_tm_clones+0x30>
  40068f:	5d                   	pop    %rbp
  400690:	bf 50 20 60 00       	mov    $0x602050,%edi
  400695:	ff e0                	jmpq   *%rax
  400697:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40069e:	00 00 
  4006a0:	5d                   	pop    %rbp
  4006a1:	c3                   	retq   
  4006a2:	0f 1f 40 00          	nopl   0x0(%rax)
  4006a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ad:	00 00 00 

00000000004006b0 <register_tm_clones>:
  4006b0:	be 50 20 60 00       	mov    $0x602050,%esi
  4006b5:	55                   	push   %rbp
  4006b6:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  4006bd:	48 c1 fe 03          	sar    $0x3,%rsi
  4006c1:	48 89 e5             	mov    %rsp,%rbp
  4006c4:	48 89 f0             	mov    %rsi,%rax
  4006c7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006cb:	48 01 c6             	add    %rax,%rsi
  4006ce:	48 d1 fe             	sar    %rsi
  4006d1:	74 15                	je     4006e8 <register_tm_clones+0x38>
  4006d3:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d8:	48 85 c0             	test   %rax,%rax
  4006db:	74 0b                	je     4006e8 <register_tm_clones+0x38>
  4006dd:	5d                   	pop    %rbp
  4006de:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006e3:	ff e0                	jmpq   *%rax
  4006e5:	0f 1f 00             	nopl   (%rax)
  4006e8:	5d                   	pop    %rbp
  4006e9:	c3                   	retq   
  4006ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006f0 <__do_global_dtors_aux>:
  4006f0:	80 3d 59 19 20 00 00 	cmpb   $0x0,0x201959(%rip)        # 602050 <__TMC_END__>
  4006f7:	75 11                	jne    40070a <__do_global_dtors_aux+0x1a>
  4006f9:	55                   	push   %rbp
  4006fa:	48 89 e5             	mov    %rsp,%rbp
  4006fd:	e8 6e ff ff ff       	callq  400670 <deregister_tm_clones>
  400702:	5d                   	pop    %rbp
  400703:	c6 05 46 19 20 00 01 	movb   $0x1,0x201946(%rip)        # 602050 <__TMC_END__>
  40070a:	f3 c3                	repz retq 
  40070c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400710 <frame_dummy>:
  400710:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400715:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400719:	75 05                	jne    400720 <frame_dummy+0x10>
  40071b:	eb 93                	jmp    4006b0 <register_tm_clones>
  40071d:	0f 1f 00             	nopl   (%rax)
  400720:	b8 00 00 00 00       	mov    $0x0,%eax
  400725:	48 85 c0             	test   %rax,%rax
  400728:	74 f1                	je     40071b <frame_dummy+0xb>
  40072a:	55                   	push   %rbp
  40072b:	48 89 e5             	mov    %rsp,%rbp
  40072e:	ff d0                	callq  *%rax
  400730:	5d                   	pop    %rbp
  400731:	e9 7a ff ff ff       	jmpq   4006b0 <register_tm_clones>
  400736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40073d:	00 00 00 

0000000000400740 <main>:
  400740:	48 83 ec 58          	sub    $0x58,%rsp
  400744:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40074b:	00 
  40074c:	e8 7f 02 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400751:	b8 04 00 00 00       	mov    $0x4,%eax
  400756:	89 c7                	mov    %eax,%edi
  400758:	e8 83 02 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  40075d:	48 89 c7             	mov    %rax,%rdi
  400760:	48 89 c1             	mov    %rax,%rcx
  400763:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400769:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40076e:	48 89 c7             	mov    %rax,%rdi
  400771:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400776:	e8 95 02 00 00       	callq  400a10 <_ZN9TestClassC2Ev>
  40077b:	e9 00 00 00 00       	jmpq   400780 <main+0x40>
  400780:	be 77 77 00 00       	mov    $0x7777,%esi
  400785:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40078a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40078f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400794:	48 83 c1 04          	add    $0x4,%rcx
  400798:	48 89 cf             	mov    %rcx,%rdi
  40079b:	e8 c0 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  4007a0:	be 88 88 00 00       	mov    $0x8888,%esi
  4007a5:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007aa:	48 83 c1 04          	add    $0x4,%rcx
  4007ae:	48 89 cf             	mov    %rcx,%rdi
  4007b1:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4007b6:	e8 e5 02 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  4007bb:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007c0:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007c5:	e8 16 03 00 00       	callq  400ae0 <_ZN9TestClass15increase_field0Ev>
  4007ca:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007cf:	e8 4c 03 00 00       	callq  400b20 <_ZN9TestClass15increase_field1Ev>
  4007d4:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007d9:	48 83 c0 04          	add    $0x4,%rax
  4007dd:	48 89 c7             	mov    %rax,%rdi
  4007e0:	e8 7b 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  4007e5:	8b 30                	mov    (%rax),%esi
  4007e7:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007ec:	48 83 c0 04          	add    $0x4,%rax
  4007f0:	48 89 c7             	mov    %rax,%rdi
  4007f3:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007f7:	e8 74 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  4007fc:	48 bf 34 0f 40 00 00 	movabs $0x400f34,%rdi
  400803:	00 00 00 
  400806:	8b 10                	mov    (%rax),%edx
  400808:	8b 74 24 14          	mov    0x14(%rsp),%esi
  40080c:	b0 00                	mov    $0x0,%al
  40080e:	e8 cd fd ff ff       	callq  4005e0 <printf@plt>
  400813:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400818:	48 83 c1 04          	add    $0x4,%rcx
  40081c:	48 89 cf             	mov    %rcx,%rdi
  40081f:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400823:	e8 38 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400828:	31 d2                	xor    %edx,%edx
  40082a:	41 88 d0             	mov    %dl,%r8b
  40082d:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400833:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  400838:	0f 85 1e 00 00 00    	jne    40085c <main+0x11c>
  40083e:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400843:	48 83 c0 04          	add    $0x4,%rax
  400847:	48 89 c7             	mov    %rax,%rdi
  40084a:	e8 21 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  40084f:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400855:	0f 94 c1             	sete   %cl
  400858:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40085c:	8a 44 24 0f          	mov    0xf(%rsp),%al
  400860:	34 ff                	xor    $0xff,%al
  400862:	24 01                	and    $0x1,%al
  400864:	0f b6 c0             	movzbl %al,%eax
  400867:	48 83 c4 58          	add    $0x58,%rsp
  40086b:	c3                   	retq   
  40086c:	89 d1                	mov    %edx,%ecx
  40086e:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400873:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400877:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40087c:	e8 cf 01 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  400881:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400886:	e8 95 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  40088b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400890 <_Z12new_instancev>:
  400890:	48 83 ec 28          	sub    $0x28,%rsp
  400894:	b8 04 00 00 00       	mov    $0x4,%eax
  400899:	89 c7                	mov    %eax,%edi
  40089b:	e8 40 01 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4008a0:	48 89 c7             	mov    %rax,%rdi
  4008a3:	48 89 c1             	mov    %rax,%rcx
  4008a6:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008ac:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008b1:	48 89 c7             	mov    %rax,%rdi
  4008b4:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  4008b9:	e8 52 01 00 00       	callq  400a10 <_ZN9TestClassC2Ev>
  4008be:	e9 00 00 00 00       	jmpq   4008c3 <_Z12new_instancev+0x33>
  4008c3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008c8:	48 83 c4 28          	add    $0x28,%rsp
  4008cc:	c3                   	retq   
  4008cd:	89 d1                	mov    %edx,%ecx
  4008cf:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4008d4:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  4008d8:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008dd:	e8 6e 01 00 00       	callq  400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008e2:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008e7:	e8 34 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008f0 <_Z12write_field0P9TestClass>:
  4008f0:	48 83 ec 18          	sub    $0x18,%rsp
  4008f4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008f9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008fe:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400903:	48 83 c7 04          	add    $0x4,%rdi
  400907:	e8 54 01 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  40090c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400911:	48 83 c4 18          	add    $0x18,%rsp
  400915:	c3                   	retq   
  400916:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40091d:	00 00 00 

0000000000400920 <_Z11read_field0P9TestClass>:
  400920:	50                   	push   %rax
  400921:	48 89 3c 24          	mov    %rdi,(%rsp)
  400925:	48 8b 3c 24          	mov    (%rsp),%rdi
  400929:	48 83 c7 04          	add    $0x4,%rdi
  40092d:	e8 2e 02 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400932:	8b 00                	mov    (%rax),%eax
  400934:	59                   	pop    %rcx
  400935:	c3                   	retq   
  400936:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40093d:	00 00 00 

0000000000400940 <_Z12write_field1P9TestClass>:
  400940:	48 83 ec 18          	sub    $0x18,%rsp
  400944:	be 77 77 00 00       	mov    $0x7777,%esi
  400949:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40094e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400953:	48 83 c7 04          	add    $0x4,%rdi
  400957:	e8 44 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40095c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400961:	48 83 c4 18          	add    $0x18,%rsp
  400965:	c3                   	retq   
  400966:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40096d:	00 00 00 

0000000000400970 <_Z11read_field1P9TestClass>:
  400970:	50                   	push   %rax
  400971:	48 89 3c 24          	mov    %rdi,(%rsp)
  400975:	48 8b 3c 24          	mov    (%rsp),%rdi
  400979:	48 83 c7 04          	add    $0x4,%rdi
  40097d:	e8 ee 01 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  400982:	8b 00                	mov    (%rax),%eax
  400984:	59                   	pop    %rcx
  400985:	c3                   	retq   
  400986:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40098d:	00 00 00 

0000000000400990 <_Z21explicit_write_field0m>:
  400990:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400995:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40099a:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  4009a1:	77 77 00 00 
  4009a5:	c3                   	retq   
  4009a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ad:	00 00 00 

00000000004009b0 <_Z20explicit_read_field0m>:
  4009b0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ba:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  4009c1:	c3                   	retq   
  4009c2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009c9:	00 00 00 
  4009cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009d0:	50                   	push   %rax
  4009d1:	e8 aa 02 00 00       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  4009d6:	48 89 c7             	mov    %rax,%rdi
  4009d9:	e8 b2 03 00 00       	callq  400d90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009de:	58                   	pop    %rax
  4009df:	c3                   	retq   

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  4009e0:	50                   	push   %rax
  4009e1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009e5:	e8 26 04 00 00       	callq  400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009ea:	e8 91 02 00 00       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  4009ef:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009f4:	89 ce                	mov    %ecx,%esi
  4009f6:	48 89 c7             	mov    %rax,%rdi
  4009f9:	e8 42 04 00 00       	callq  400e40 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009fe:	48 89 c7             	mov    %rax,%rdi
  400a01:	e8 1a 04 00 00       	callq  400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a06:	59                   	pop    %rcx
  400a07:	c3                   	retq   
  400a08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a0f:	00 

0000000000400a10 <_ZN9TestClassC2Ev>:
  400a10:	48 83 ec 18          	sub    $0x18,%rsp
  400a14:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a19:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a1e:	48 89 f8             	mov    %rdi,%rax
  400a21:	48 83 c0 04          	add    $0x4,%rax
  400a25:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a2a:	48 89 c7             	mov    %rax,%rdi
  400a2d:	e8 4e 01 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>
  400a32:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a37:	48 83 c0 04          	add    $0x4,%rax
  400a3b:	48 89 c7             	mov    %rax,%rdi
  400a3e:	e8 4d 01 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>
  400a43:	48 83 c4 18          	add    $0x18,%rsp
  400a47:	c3                   	retq   
  400a48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a4f:	00 

0000000000400a50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a55:	c3                   	retq   
  400a56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5d:	00 00 00 

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>:
  400a60:	48 83 ec 28          	sub    $0x28,%rsp
  400a64:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a69:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a72:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a76:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a7b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a7f:	e8 7c 01 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a84:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a88:	89 30                	mov    %esi,(%rax)
  400a8a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a8f:	48 83 c4 28          	add    $0x28,%rsp
  400a93:	c3                   	retq   
  400a94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a9b:	00 00 00 
  400a9e:	66 90                	xchg   %ax,%ax

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>:
  400aa0:	48 83 ec 28          	sub    $0x28,%rsp
  400aa4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400aa9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400aad:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ab2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400ab6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400abb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400abf:	e8 4c 02 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
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
  400af3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400af8:	48 83 c7 04          	add    $0x4,%rdi
  400afc:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b00:	48 89 c7             	mov    %rax,%rdi
  400b03:	48 8b 34 24          	mov    (%rsp),%rsi
  400b07:	e8 94 00 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>
  400b0c:	48 83 c4 18          	add    $0x18,%rsp
  400b10:	c3                   	retq   
  400b11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b18:	00 00 00 
  400b1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b20 <_ZN9TestClass15increase_field1Ev>:
  400b20:	48 83 ec 18          	sub    $0x18,%rsp
  400b24:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b29:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b2e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b33:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b38:	48 83 c7 04          	add    $0x4,%rdi
  400b3c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b40:	48 89 c7             	mov    %rax,%rdi
  400b43:	48 8b 34 24          	mov    (%rsp),%rsi
  400b47:	e8 64 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>
  400b4c:	48 83 c4 18          	add    $0x18,%rsp
  400b50:	c3                   	retq   
  400b51:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b58:	00 00 00 
  400b5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>:
  400b60:	50                   	push   %rax
  400b61:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b65:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b69:	e8 92 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b6e:	59                   	pop    %rcx
  400b6f:	c3                   	retq   

0000000000400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>:
  400b70:	50                   	push   %rax
  400b71:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b75:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b79:	e8 92 01 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b7e:	59                   	pop    %rcx
  400b7f:	c3                   	retq   

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>:
  400b80:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b85:	c3                   	retq   
  400b86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8d:	00 00 00 

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>:
  400b90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b95:	c3                   	retq   
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>:
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
  400bcf:	e8 2c 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400bd8:	0f af 10             	imul   (%rax),%edx
  400bdb:	89 10                	mov    %edx,(%rax)
  400bdd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400be2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400be7:	e8 24 00 00 00       	callq  400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>
  400bec:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400bf1:	48 83 c4 38          	add    $0x38,%rsp
  400bf5:	c3                   	retq   
  400bf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bfd:	00 00 00 

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c00:	50                   	push   %rax
  400c01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c09:	e8 12 00 00 00       	callq  400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c0e:	59                   	pop    %rcx
  400c0f:	c3                   	retq   

0000000000400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>:
  400c10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c15:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c1a:	c3                   	retq   
  400c1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c20:	48 83 ec 28          	sub    $0x28,%rsp
  400c24:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c29:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c2e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400c33:	e8 48 00 00 00       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400c38:	48 89 c7             	mov    %rax,%rdi
  400c3b:	e8 50 00 00 00       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c40:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c45:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c4a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c4f:	48 83 e8 04          	sub    $0x4,%rax
  400c53:	48 c1 e8 02          	shr    $0x2,%rax
  400c57:	48 c1 e0 02          	shl    $0x2,%rax
  400c5b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c60:	48 83 c0 00          	add    $0x0,%rax
  400c64:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c69:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c6e:	48 83 c4 28          	add    $0x28,%rsp
  400c72:	c3                   	retq   
  400c73:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7a:	00 00 00 
  400c7d:	0f 1f 00             	nopl   (%rax)

0000000000400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>:
  400c80:	50                   	push   %rax
  400c81:	e8 1a 00 00 00       	callq  400ca0 <_ZN9TestClass14storage_bufferEv>
  400c86:	59                   	pop    %rcx
  400c87:	c3                   	retq   
  400c88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c8f:	00 

0000000000400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c95:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c9a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c9e:	c3                   	retq   
  400c9f:	90                   	nop

0000000000400ca0 <_ZN9TestClass14storage_bufferEv>:
  400ca0:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400ca7:	00 00 00 
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>:
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
  400cdf:	e8 2c 00 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ce4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400ce8:	0f af 10             	imul   (%rax),%edx
  400ceb:	89 10                	mov    %edx,(%rax)
  400ced:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400cf2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400cf7:	e8 24 00 00 00       	callq  400d20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>
  400cfc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d01:	48 83 c4 38          	add    $0x38,%rsp
  400d05:	c3                   	retq   
  400d06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d0d:	00 00 00 

0000000000400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d10:	50                   	push   %rax
  400d11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d19:	e8 12 00 00 00       	callq  400d30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d1e:	59                   	pop    %rcx
  400d1f:	c3                   	retq   

0000000000400d20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>:
  400d20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d2a:	c3                   	retq   
  400d2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d30 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d30:	48 83 ec 28          	sub    $0x28,%rsp
  400d34:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d39:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d3e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d43:	e8 38 ff ff ff       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400d48:	48 89 c7             	mov    %rax,%rdi
  400d4b:	e8 40 ff ff ff       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d50:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d55:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d5a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d5f:	48 83 e8 04          	sub    $0x4,%rax
  400d63:	48 c1 e8 02          	shr    $0x2,%rax
  400d67:	48 c1 e0 02          	shl    $0x2,%rax
  400d6b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d70:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d76:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d80:	48 83 c4 28          	add    $0x28,%rsp
  400d84:	c3                   	retq   
  400d85:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d8c:	00 00 00 
  400d8f:	90                   	nop

0000000000400d90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d90:	48 83 ec 18          	sub    $0x18,%rsp
  400d94:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d99:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d9e:	48 89 f8             	mov    %rdi,%rax
  400da1:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400da6:	48 89 c7             	mov    %rax,%rdi
  400da9:	e8 32 00 00 00       	callq  400de0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400dae:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400db3:	89 cf                	mov    %ecx,%edi
  400db5:	e8 36 f8 ff ff       	callq  4005f0 <malloc@plt>
  400dba:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400dbf:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400dc3:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400dca:	00 
  400dcb:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400dd2:	00 
  400dd3:	48 83 c4 18          	add    $0x18,%rsp
  400dd7:	c3                   	retq   
  400dd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ddf:	00 

0000000000400de0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400de0:	50                   	push   %rax
  400de1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400de5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400de9:	e8 02 00 00 00       	callq  400df0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400dee:	58                   	pop    %rax
  400def:	c3                   	retq   

0000000000400df0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400df0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400df5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400dfa:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400e01:	c3                   	retq   
  400e02:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e09:	00 00 00 
  400e0c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e10:	c3                   	retq   
  400e11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e18:	00 00 00 
  400e1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400e20:	50                   	push   %rax
  400e21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400e25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400e29:	48 83 c7 01          	add    $0x1,%rdi
  400e2d:	e8 3e 00 00 00       	callq  400e70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e32:	59                   	pop    %rcx
  400e33:	c3                   	retq   
  400e34:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e3b:	00 00 00 
  400e3e:	66 90                	xchg   %ax,%ax

0000000000400e40 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e45:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400e4a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400e4f:	48 8b 3e             	mov    (%rsi),%rdi
  400e52:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400e57:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400e5c:	48 03 3e             	add    (%rsi),%rdi
  400e5f:	48 89 3e             	mov    %rdi,(%rsi)
  400e62:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e67:	c3                   	retq   
  400e68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e6f:	00 

0000000000400e70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e70:	48 83 ec 18          	sub    $0x18,%rsp
  400e74:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e79:	e8 02 fe ff ff       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400e7e:	48 89 c7             	mov    %rax,%rdi
  400e81:	e8 0a fe ff ff       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e86:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e8b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e8f:	48 01 f8             	add    %rdi,%rax
  400e92:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e97:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e9c:	48 83 c4 18          	add    $0x18,%rsp
  400ea0:	c3                   	retq   
  400ea1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ea8:	00 00 00 
  400eab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400eb0 <__libc_csu_init>:
  400eb0:	41 57                	push   %r15
  400eb2:	41 56                	push   %r14
  400eb4:	41 89 ff             	mov    %edi,%r15d
  400eb7:	41 55                	push   %r13
  400eb9:	41 54                	push   %r12
  400ebb:	4c 8d 25 1e 0f 20 00 	lea    0x200f1e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400ec2:	55                   	push   %rbp
  400ec3:	48 8d 2d 1e 0f 20 00 	lea    0x200f1e(%rip),%rbp        # 601de8 <__init_array_end>
  400eca:	53                   	push   %rbx
  400ecb:	49 89 f6             	mov    %rsi,%r14
  400ece:	49 89 d5             	mov    %rdx,%r13
  400ed1:	4c 29 e5             	sub    %r12,%rbp
  400ed4:	48 83 ec 08          	sub    $0x8,%rsp
  400ed8:	48 c1 fd 03          	sar    $0x3,%rbp
  400edc:	e8 c7 f6 ff ff       	callq  4005a8 <_init>
  400ee1:	48 85 ed             	test   %rbp,%rbp
  400ee4:	74 20                	je     400f06 <__libc_csu_init+0x56>
  400ee6:	31 db                	xor    %ebx,%ebx
  400ee8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400eef:	00 
  400ef0:	4c 89 ea             	mov    %r13,%rdx
  400ef3:	4c 89 f6             	mov    %r14,%rsi
  400ef6:	44 89 ff             	mov    %r15d,%edi
  400ef9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400efd:	48 83 c3 01          	add    $0x1,%rbx
  400f01:	48 39 eb             	cmp    %rbp,%rbx
  400f04:	75 ea                	jne    400ef0 <__libc_csu_init+0x40>
  400f06:	48 83 c4 08          	add    $0x8,%rsp
  400f0a:	5b                   	pop    %rbx
  400f0b:	5d                   	pop    %rbp
  400f0c:	41 5c                	pop    %r12
  400f0e:	41 5d                	pop    %r13
  400f10:	41 5e                	pop    %r14
  400f12:	41 5f                	pop    %r15
  400f14:	c3                   	retq   
  400f15:	90                   	nop
  400f16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f1d:	00 00 00 

0000000000400f20 <__libc_csu_fini>:
  400f20:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f24 <_fini>:
  400f24:	48 83 ec 08          	sub    $0x8,%rsp
  400f28:	48 83 c4 08          	add    $0x8,%rsp
  400f2c:	c3                   	retq   
