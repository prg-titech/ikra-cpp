
bin/clang++-3.8_-O0_DynamicStorage_0:     file format elf64-x86-64


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
  40064f:	49 c7 c0 00 0e 40 00 	mov    $0x400e00,%r8
  400656:	48 c7 c1 90 0d 40 00 	mov    $0x400d90,%rcx
  40065d:	48 c7 c7 50 07 40 00 	mov    $0x400750,%rdi
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

0000000000400740 <_ZN9TestClass7storageEv>:
  400740:	55                   	push   %rbp
  400741:	48 89 e5             	mov    %rsp,%rbp
  400744:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  40074b:	00 00 00 
  40074e:	5d                   	pop    %rbp
  40074f:	c3                   	retq   

0000000000400750 <main>:
  400750:	55                   	push   %rbp
  400751:	48 89 e5             	mov    %rsp,%rbp
  400754:	48 83 ec 50          	sub    $0x50,%rsp
  400758:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  40075f:	e8 3c 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageEv>
  400764:	31 c0                	xor    %eax,%eax
  400766:	89 c7                	mov    %eax,%edi
  400768:	e8 53 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40076d:	48 89 c7             	mov    %rax,%rdi
  400770:	48 89 c1             	mov    %rax,%rcx
  400773:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400777:	48 89 c7             	mov    %rax,%rdi
  40077a:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  40077e:	e8 6d 02 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400783:	e9 00 00 00 00       	jmpq   400788 <main+0x38>
  400788:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  40078c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400791:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400795:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400799:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40079d:	e8 8e 02 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4007a2:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  4007a6:	ba 88 88 00 00       	mov    $0x8888,%edx
  4007ab:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  4007af:	e8 cc 02 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007b4:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b8:	e8 13 03 00 00       	callq  400ad0 <_ZN9TestClass15increase_field0Ev>
  4007bd:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c1:	e8 3a 03 00 00       	callq  400b00 <_ZN9TestClass15increase_field1Ev>
  4007c6:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007ca:	e8 61 03 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007cf:	8b 30                	mov    (%rax),%esi
  4007d1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007d5:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007d8:	e8 73 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007dd:	48 bf 14 0e 40 00 00 	movabs $0x400e14,%rdi
  4007e4:	00 00 00 
  4007e7:	8b 10                	mov    (%rax),%edx
  4007e9:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007ec:	b0 00                	mov    $0x0,%al
  4007ee:	e8 ed fd ff ff       	callq  4005e0 <printf@plt>
  4007f3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007f7:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007fa:	e8 31 03 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007ff:	31 d2                	xor    %edx,%edx
  400801:	88 d1                	mov    %dl,%cl
  400803:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400809:	88 4d bf             	mov    %cl,-0x41(%rbp)
  40080c:	0f 85 15 00 00 00    	jne    400827 <main+0xd7>
  400812:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400816:	e8 35 03 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40081b:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400821:	0f 94 c1             	sete   %cl
  400824:	88 4d bf             	mov    %cl,-0x41(%rbp)
  400827:	8a 45 bf             	mov    -0x41(%rbp),%al
  40082a:	34 ff                	xor    $0xff,%al
  40082c:	24 01                	and    $0x1,%al
  40082e:	0f b6 c0             	movzbl %al,%eax
  400831:	48 83 c4 50          	add    $0x50,%rsp
  400835:	5d                   	pop    %rbp
  400836:	c3                   	retq   
  400837:	89 d1                	mov    %edx,%ecx
  400839:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40083d:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400840:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400844:	e8 d7 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  400849:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40084d:	e8 ce fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400852:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400859:	1f 84 00 00 00 00 00 

0000000000400860 <_Z12new_instancev>:
  400860:	55                   	push   %rbp
  400861:	48 89 e5             	mov    %rsp,%rbp
  400864:	48 83 ec 20          	sub    $0x20,%rsp
  400868:	31 c0                	xor    %eax,%eax
  40086a:	89 c7                	mov    %eax,%edi
  40086c:	e8 4f 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  400871:	48 89 c7             	mov    %rax,%rdi
  400874:	48 89 c1             	mov    %rax,%rcx
  400877:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40087b:	48 89 c7             	mov    %rax,%rdi
  40087e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400882:	e8 69 01 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400887:	e9 00 00 00 00       	jmpq   40088c <_Z12new_instancev+0x2c>
  40088c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400890:	48 83 c4 20          	add    $0x20,%rsp
  400894:	5d                   	pop    %rbp
  400895:	c3                   	retq   
  400896:	89 d1                	mov    %edx,%ecx
  400898:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40089c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40089f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008a3:	e8 78 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  4008a8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008ac:	e8 6f fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008b1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008b8:	0f 1f 84 00 00 00 00 
  4008bf:	00 

00000000004008c0 <_Z12write_field0P9TestClass>:
  4008c0:	55                   	push   %rbp
  4008c1:	48 89 e5             	mov    %rsp,%rbp
  4008c4:	48 83 ec 10          	sub    $0x10,%rsp
  4008c8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008cc:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008d1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008d9:	48 89 c7             	mov    %rax,%rdi
  4008dc:	e8 4f 01 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4008e1:	48 83 c4 10          	add    $0x10,%rsp
  4008e5:	5d                   	pop    %rbp
  4008e6:	c3                   	retq   
  4008e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008ee:	00 00 

00000000004008f0 <_Z11read_field0P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400900:	e8 2b 02 00 00       	callq  400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400905:	8b 00                	mov    (%rax),%eax
  400907:	48 83 c4 10          	add    $0x10,%rsp
  40090b:	5d                   	pop    %rbp
  40090c:	c3                   	retq   
  40090d:	0f 1f 00             	nopl   (%rax)

0000000000400910 <_Z12write_field1P9TestClass>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 10          	sub    $0x10,%rsp
  400918:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40091c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400921:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400925:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400929:	48 89 c7             	mov    %rax,%rdi
  40092c:	e8 4f 01 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400931:	48 83 c4 10          	add    $0x10,%rsp
  400935:	5d                   	pop    %rbp
  400936:	c3                   	retq   
  400937:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40093e:	00 00 

0000000000400940 <_Z11read_field1P9TestClass>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 83 ec 10          	sub    $0x10,%rsp
  400948:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40094c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400950:	e8 fb 01 00 00       	callq  400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400955:	8b 00                	mov    (%rax),%eax
  400957:	48 83 c4 10          	add    $0x10,%rsp
  40095b:	5d                   	pop    %rbp
  40095c:	c3                   	retq   
  40095d:	0f 1f 00             	nopl   (%rax)

0000000000400960 <_Z21explicit_write_field0m>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400968:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40096c:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  400973:	77 77 00 00 
  400977:	5d                   	pop    %rbp
  400978:	c3                   	retq   
  400979:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400980 <_Z20explicit_read_field0m>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40098c:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  400993:	5d                   	pop    %rbp
  400994:	c3                   	retq   
  400995:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099c:	00 00 00 
  40099f:	90                   	nop

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageEv>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	e8 97 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009a9:	48 89 c7             	mov    %rax,%rdi
  4009ac:	e8 4f 03 00 00       	callq  400d00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009b1:	5d                   	pop    %rbp
  4009b2:	c3                   	retq   
  4009b3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ba:	00 00 00 
  4009bd:	0f 1f 00             	nopl   (%rax)

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 83 ec 10          	sub    $0x10,%rsp
  4009c8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009cc:	e8 7f 03 00 00       	callq  400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009d1:	e8 6a fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009d6:	48 8b 38             	mov    (%rax),%rdi
  4009d9:	48 89 f9             	mov    %rdi,%rcx
  4009dc:	48 83 c1 01          	add    $0x1,%rcx
  4009e0:	48 89 08             	mov    %rcx,(%rax)
  4009e3:	e8 78 03 00 00       	callq  400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009e8:	48 83 c4 10          	add    $0x10,%rsp
  4009ec:	5d                   	pop    %rbp
  4009ed:	c3                   	retq   
  4009ee:	66 90                	xchg   %ax,%ax

00000000004009f0 <_ZN9TestClassC2Ev>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 83 ec 10          	sub    $0x10,%rsp
  4009f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a00:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a04:	e8 67 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  400a09:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400a0d:	e8 6e 01 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400a12:	48 83 c4 10          	add    $0x10,%rsp
  400a16:	5d                   	pop    %rbp
  400a17:	c3                   	retq   
  400a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a1f:	00 

0000000000400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a28:	5d                   	pop    %rbp
  400a29:	c3                   	retq   
  400a2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 83 ec 30          	sub    $0x30,%rsp
  400a38:	48 89 f8             	mov    %rdi,%rax
  400a3b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a3f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a42:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a46:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a49:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a4d:	48 89 f7             	mov    %rsi,%rdi
  400a50:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a54:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a57:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a5b:	e8 80 01 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a60:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a63:	89 10                	mov    %edx,(%rax)
  400a65:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a69:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a6d:	e8 8e 01 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400a72:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a76:	48 83 c4 30          	add    $0x30,%rsp
  400a7a:	5d                   	pop    %rbp
  400a7b:	c3                   	retq   
  400a7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 83 ec 30          	sub    $0x30,%rsp
  400a88:	48 89 f8             	mov    %rdi,%rax
  400a8b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a8f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a92:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a96:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a99:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a9d:	48 89 f7             	mov    %rsi,%rdi
  400aa0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400aa4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400aa7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400aab:	e8 e0 01 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ab0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ab3:	89 10                	mov    %edx,(%rax)
  400ab5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ab9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400abd:	e8 ee 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400ac2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ac6:	48 83 c4 30          	add    $0x30,%rsp
  400aca:	5d                   	pop    %rbp
  400acb:	c3                   	retq   
  400acc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ad0 <_ZN9TestClass15increase_field0Ev>:
  400ad0:	55                   	push   %rbp
  400ad1:	48 89 e5             	mov    %rsp,%rbp
  400ad4:	48 83 ec 10          	sub    $0x10,%rsp
  400ad8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400adc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ae1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ae5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ae9:	48 89 c7             	mov    %rax,%rdi
  400aec:	e8 9f 00 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400af1:	48 83 c4 10          	add    $0x10,%rsp
  400af5:	5d                   	pop    %rbp
  400af6:	c3                   	retq   
  400af7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400afe:	00 00 

0000000000400b00 <_ZN9TestClass15increase_field1Ev>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 10          	sub    $0x10,%rsp
  400b08:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b0c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b11:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b15:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b19:	48 89 c7             	mov    %rax,%rdi
  400b1c:	e8 1f 01 00 00       	callq  400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400b21:	48 83 c4 10          	add    $0x10,%rsp
  400b25:	5d                   	pop    %rbp
  400b26:	c3                   	retq   
  400b27:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b2e:	00 00 

0000000000400b30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 83 ec 10          	sub    $0x10,%rsp
  400b38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b40:	e8 9b 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b45:	48 83 c4 10          	add    $0x10,%rsp
  400b49:	5d                   	pop    %rbp
  400b4a:	c3                   	retq   
  400b4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 83 ec 10          	sub    $0x10,%rsp
  400b58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b60:	e8 2b 01 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b65:	48 83 c4 10          	add    $0x10,%rsp
  400b69:	5d                   	pop    %rbp
  400b6a:	c3                   	retq   
  400b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b78:	5d                   	pop    %rbp
  400b79:	c3                   	retq   
  400b7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b88:	5d                   	pop    %rbp
  400b89:	c3                   	retq   
  400b8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 83 ec 30          	sub    $0x30,%rsp
  400b98:	48 89 f8             	mov    %rdi,%rax
  400b9b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b9f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400ba2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ba6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400ba9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bad:	48 89 f7             	mov    %rsi,%rdi
  400bb0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400bb4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400bb7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400bbb:	e8 20 00 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bc0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400bc3:	0f af 10             	imul   (%rax),%edx
  400bc6:	89 10                	mov    %edx,(%rax)
  400bc8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400bcc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400bd0:	e8 2b 00 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400bd5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400bd9:	48 83 c4 30          	add    $0x30,%rsp
  400bdd:	5d                   	pop    %rbp
  400bde:	c3                   	retq   
  400bdf:	90                   	nop

0000000000400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 83 ec 10          	sub    $0x10,%rsp
  400be8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bf0:	e8 1b 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bf5:	48 83 c4 10          	add    $0x10,%rsp
  400bf9:	5d                   	pop    %rbp
  400bfa:	c3                   	retq   
  400bfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c08:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c0c:	5d                   	pop    %rbp
  400c0d:	c3                   	retq   
  400c0e:	66 90                	xchg   %ax,%ax

0000000000400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c10:	55                   	push   %rbp
  400c11:	48 89 e5             	mov    %rsp,%rbp
  400c14:	48 83 ec 10          	sub    $0x10,%rsp
  400c18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c1c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c20:	48 c1 e7 02          	shl    $0x2,%rdi
  400c24:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c28:	e8 13 fb ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400c2d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400c31:	48 03 78 08          	add    0x8(%rax),%rdi
  400c35:	48 89 f8             	mov    %rdi,%rax
  400c38:	48 83 c4 10          	add    $0x10,%rsp
  400c3c:	5d                   	pop    %rbp
  400c3d:	c3                   	retq   
  400c3e:	66 90                	xchg   %ax,%ax

0000000000400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400c40:	55                   	push   %rbp
  400c41:	48 89 e5             	mov    %rsp,%rbp
  400c44:	48 83 ec 30          	sub    $0x30,%rsp
  400c48:	48 89 f8             	mov    %rdi,%rax
  400c4b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c4f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c52:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c56:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c59:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c5d:	48 89 f7             	mov    %rsi,%rdi
  400c60:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c64:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c67:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c6b:	e8 20 00 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c70:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c73:	0f af 10             	imul   (%rax),%edx
  400c76:	89 10                	mov    %edx,(%rax)
  400c78:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c7c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c80:	e8 2b 00 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400c85:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c89:	48 83 c4 30          	add    $0x30,%rsp
  400c8d:	5d                   	pop    %rbp
  400c8e:	c3                   	retq   
  400c8f:	90                   	nop

0000000000400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c90:	55                   	push   %rbp
  400c91:	48 89 e5             	mov    %rsp,%rbp
  400c94:	48 83 ec 10          	sub    $0x10,%rsp
  400c98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c9c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ca0:	e8 1b 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca5:	48 83 c4 10          	add    $0x10,%rsp
  400ca9:	5d                   	pop    %rbp
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cb8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400cbc:	5d                   	pop    %rbp
  400cbd:	c3                   	retq   
  400cbe:	66 90                	xchg   %ax,%ax

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 83 ec 10          	sub    $0x10,%rsp
  400cc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ccc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cd0:	48 c1 e7 02          	shl    $0x2,%rdi
  400cd4:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400cd8:	e8 63 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400cdd:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400ce1:	48 03 78 08          	add    0x8(%rax),%rdi
  400ce5:	48 81 c7 d0 48 00 00 	add    $0x48d0,%rdi
  400cec:	48 89 f8             	mov    %rdi,%rax
  400cef:	48 83 c4 10          	add    $0x10,%rsp
  400cf3:	5d                   	pop    %rbp
  400cf4:	c3                   	retq   
  400cf5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfc:	00 00 00 
  400cff:	90                   	nop

0000000000400d00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 10          	sub    $0x10,%rsp
  400d08:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  400d0d:	89 c1                	mov    %eax,%ecx
  400d0f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d13:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d17:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d1b:	48 89 cf             	mov    %rcx,%rdi
  400d1e:	e8 cd f8 ff ff       	callq  4005f0 <malloc@plt>
  400d23:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400d27:	48 89 41 08          	mov    %rax,0x8(%rcx)
  400d2b:	48 c7 41 18 00 00 00 	movq   $0x0,0x18(%rcx)
  400d32:	00 
  400d33:	48 c7 41 10 00 00 00 	movq   $0x0,0x10(%rcx)
  400d3a:	00 
  400d3b:	48 83 c4 10          	add    $0x10,%rsp
  400d3f:	5d                   	pop    %rbp
  400d40:	c3                   	retq   
  400d41:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d48:	00 00 00 
  400d4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	5d                   	pop    %rbp
  400d55:	c3                   	retq   
  400d56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d5d:	00 00 00 

0000000000400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400d60:	55                   	push   %rbp
  400d61:	48 89 e5             	mov    %rsp,%rbp
  400d64:	48 83 ec 10          	sub    $0x10,%rsp
  400d68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d6c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d70:	48 83 c7 01          	add    $0x1,%rdi
  400d74:	e8 07 00 00 00       	callq  400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d79:	48 83 c4 10          	add    $0x10,%rsp
  400d7d:	5d                   	pop    %rbp
  400d7e:	c3                   	retq   
  400d7f:	90                   	nop

0000000000400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d80:	55                   	push   %rbp
  400d81:	48 89 e5             	mov    %rsp,%rbp
  400d84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d88:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400d8c:	5d                   	pop    %rbp
  400d8d:	c3                   	retq   
  400d8e:	66 90                	xchg   %ax,%ax

0000000000400d90 <__libc_csu_init>:
  400d90:	41 57                	push   %r15
  400d92:	41 56                	push   %r14
  400d94:	41 89 ff             	mov    %edi,%r15d
  400d97:	41 55                	push   %r13
  400d99:	41 54                	push   %r12
  400d9b:	4c 8d 25 3e 10 20 00 	lea    0x20103e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400da2:	55                   	push   %rbp
  400da3:	48 8d 2d 3e 10 20 00 	lea    0x20103e(%rip),%rbp        # 601de8 <__init_array_end>
  400daa:	53                   	push   %rbx
  400dab:	49 89 f6             	mov    %rsi,%r14
  400dae:	49 89 d5             	mov    %rdx,%r13
  400db1:	4c 29 e5             	sub    %r12,%rbp
  400db4:	48 83 ec 08          	sub    $0x8,%rsp
  400db8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dbc:	e8 e7 f7 ff ff       	callq  4005a8 <_init>
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
