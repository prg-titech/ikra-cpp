
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
  40064f:	49 c7 c0 d0 0e 40 00 	mov    $0x400ed0,%r8
  400656:	48 c7 c1 60 0e 40 00 	mov    $0x400e60,%rcx
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
  40075f:	e8 3c 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400764:	31 c0                	xor    %eax,%eax
  400766:	89 c7                	mov    %eax,%edi
  400768:	e8 53 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40076d:	48 89 c7             	mov    %rax,%rdi
  400770:	48 89 c1             	mov    %rax,%rcx
  400773:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400777:	48 89 c7             	mov    %rax,%rdi
  40077a:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
  40077e:	e8 7d 02 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  400783:	e9 00 00 00 00       	jmpq   400788 <main+0x38>
  400788:	be 77 77 00 00       	mov    $0x7777,%esi
  40078d:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  400791:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400795:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400799:	e8 a2 02 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40079e:	be 88 88 00 00       	mov    $0x8888,%esi
  4007a3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007a7:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  4007ab:	e8 d0 02 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007b0:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b4:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  4007b8:	e8 03 03 00 00       	callq  400ac0 <_ZN9TestClass15increase_field0Ev>
  4007bd:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c1:	e8 2a 03 00 00       	callq  400af0 <_ZN9TestClass15increase_field1Ev>
  4007c6:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007ca:	e8 51 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007cf:	8b 30                	mov    (%rax),%esi
  4007d1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007d5:	89 75 bc             	mov    %esi,-0x44(%rbp)
  4007d8:	e8 63 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007dd:	48 bf e4 0e 40 00 00 	movabs $0x400ee4,%rdi
  4007e4:	00 00 00 
  4007e7:	8b 10                	mov    (%rax),%edx
  4007e9:	8b 75 bc             	mov    -0x44(%rbp),%esi
  4007ec:	b0 00                	mov    $0x0,%al
  4007ee:	e8 ed fd ff ff       	callq  4005e0 <printf@plt>
  4007f3:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007f7:	89 45 b8             	mov    %eax,-0x48(%rbp)
  4007fa:	e8 21 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007ff:	31 d2                	xor    %edx,%edx
  400801:	88 d1                	mov    %dl,%cl
  400803:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400809:	88 4d b7             	mov    %cl,-0x49(%rbp)
  40080c:	0f 85 15 00 00 00    	jne    400827 <main+0xd7>
  400812:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400816:	e8 25 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40081b:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400821:	0f 94 c1             	sete   %cl
  400824:	88 4d b7             	mov    %cl,-0x49(%rbp)
  400827:	8a 45 b7             	mov    -0x49(%rbp),%al
  40082a:	34 ff                	xor    $0xff,%al
  40082c:	24 01                	and    $0x1,%al
  40082e:	0f b6 c0             	movzbl %al,%eax
  400831:	48 83 c4 50          	add    $0x50,%rsp
  400835:	5d                   	pop    %rbp
  400836:	c3                   	retq   
  400837:	89 d1                	mov    %edx,%ecx
  400839:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40083d:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400840:	48 8b 7d d8          	mov    -0x28(%rbp),%rdi
  400844:	e8 e7 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
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
  400882:	e8 79 01 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  400887:	e9 00 00 00 00       	jmpq   40088c <_Z12new_instancev+0x2c>
  40088c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400890:	48 83 c4 20          	add    $0x20,%rsp
  400894:	5d                   	pop    %rbp
  400895:	c3                   	retq   
  400896:	89 d1                	mov    %edx,%ecx
  400898:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40089c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40089f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008a3:	e8 88 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  4008a8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008ac:	e8 6f fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008b1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008b8:	0f 1f 84 00 00 00 00 
  4008bf:	00 

00000000004008c0 <_Z12write_field0P9TestClass>:
  4008c0:	55                   	push   %rbp
  4008c1:	48 89 e5             	mov    %rsp,%rbp
  4008c4:	48 83 ec 10          	sub    $0x10,%rsp
  4008c8:	be 77 77 00 00       	mov    $0x7777,%esi
  4008cd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008d1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008d5:	e8 66 01 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4008da:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4008de:	48 83 c4 10          	add    $0x10,%rsp
  4008e2:	5d                   	pop    %rbp
  4008e3:	c3                   	retq   
  4008e4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008eb:	00 00 00 00 00 

00000000004008f0 <_Z11read_field0P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400900:	e8 1b 02 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400905:	8b 00                	mov    (%rax),%eax
  400907:	48 83 c4 10          	add    $0x10,%rsp
  40090b:	5d                   	pop    %rbp
  40090c:	c3                   	retq   
  40090d:	0f 1f 00             	nopl   (%rax)

0000000000400910 <_Z12write_field1P9TestClass>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 10          	sub    $0x10,%rsp
  400918:	be 77 77 00 00       	mov    $0x7777,%esi
  40091d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400921:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400925:	e8 56 01 00 00       	callq  400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40092a:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40092e:	48 83 c4 10          	add    $0x10,%rsp
  400932:	5d                   	pop    %rbp
  400933:	c3                   	retq   
  400934:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40093b:	00 00 00 00 00 

0000000000400940 <_Z11read_field1P9TestClass>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 83 ec 10          	sub    $0x10,%rsp
  400948:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40094c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400950:	e8 eb 01 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
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

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	e8 97 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009a9:	48 89 c7             	mov    %rax,%rdi
  4009ac:	e8 af 03 00 00       	callq  400d60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
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
  4009cc:	e8 1f 04 00 00       	callq  400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009d1:	e8 6a fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009d6:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009db:	89 ce                	mov    %ecx,%esi
  4009dd:	48 89 c7             	mov    %rax,%rdi
  4009e0:	e8 3b 04 00 00       	callq  400e20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009e5:	48 89 c7             	mov    %rax,%rdi
  4009e8:	e8 13 04 00 00       	callq  400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009ed:	48 83 c4 10          	add    $0x10,%rsp
  4009f1:	5d                   	pop    %rbp
  4009f2:	c3                   	retq   
  4009f3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009fa:	00 00 00 
  4009fd:	0f 1f 00             	nopl   (%rax)

0000000000400a00 <_ZN9TestClassC2Ev>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 10          	sub    $0x10,%rsp
  400a08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a10:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a14:	e8 47 01 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  400a19:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400a1d:	e8 4e 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400a22:	48 83 c4 10          	add    $0x10,%rsp
  400a26:	5d                   	pop    %rbp
  400a27:	c3                   	retq   
  400a28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a2f:	00 

0000000000400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a38:	5d                   	pop    %rbp
  400a39:	c3                   	retq   
  400a3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a40:	55                   	push   %rbp
  400a41:	48 89 e5             	mov    %rsp,%rbp
  400a44:	48 83 ec 20          	sub    $0x20,%rsp
  400a48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a4c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a4f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a53:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400a56:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a5a:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400a5d:	e8 6e 01 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a62:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400a65:	89 30                	mov    %esi,(%rax)
  400a67:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a6b:	48 83 c4 20          	add    $0x20,%rsp
  400a6f:	5d                   	pop    %rbp
  400a70:	c3                   	retq   
  400a71:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a78:	00 00 00 
  400a7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400a80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 83 ec 20          	sub    $0x20,%rsp
  400a88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a8c:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400a8f:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a93:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400a96:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a9a:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400a9d:	e8 2e 02 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa2:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400aa5:	89 30                	mov    %esi,(%rax)
  400aa7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400aab:	48 83 c4 20          	add    $0x20,%rsp
  400aaf:	5d                   	pop    %rbp
  400ab0:	c3                   	retq   
  400ab1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ab8:	00 00 00 
  400abb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ac0 <_ZN9TestClass15increase_field0Ev>:
  400ac0:	55                   	push   %rbp
  400ac1:	48 89 e5             	mov    %rsp,%rbp
  400ac4:	48 83 ec 10          	sub    $0x10,%rsp
  400ac8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400acc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400ad1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ad5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ad9:	48 89 c7             	mov    %rax,%rdi
  400adc:	e8 9f 00 00 00       	callq  400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400ae1:	48 83 c4 10          	add    $0x10,%rsp
  400ae5:	5d                   	pop    %rbp
  400ae6:	c3                   	retq   
  400ae7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400aee:	00 00 

0000000000400af0 <_ZN9TestClass15increase_field1Ev>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 83 ec 10          	sub    $0x10,%rsp
  400af8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400afc:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b01:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b05:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b09:	48 89 c7             	mov    %rax,%rdi
  400b0c:	e8 6f 01 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400b11:	48 83 c4 10          	add    $0x10,%rsp
  400b15:	5d                   	pop    %rbp
  400b16:	c3                   	retq   
  400b17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b1e:	00 00 

0000000000400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 83 ec 10          	sub    $0x10,%rsp
  400b28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b30:	e8 9b 00 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b35:	48 83 c4 10          	add    $0x10,%rsp
  400b39:	5d                   	pop    %rbp
  400b3a:	c3                   	retq   
  400b3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 10          	sub    $0x10,%rsp
  400b48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b50:	e8 7b 01 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b55:	48 83 c4 10          	add    $0x10,%rsp
  400b59:	5d                   	pop    %rbp
  400b5a:	c3                   	retq   
  400b5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b68:	5d                   	pop    %rbp
  400b69:	c3                   	retq   
  400b6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b78:	5d                   	pop    %rbp
  400b79:	c3                   	retq   
  400b7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b80 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 83 ec 30          	sub    $0x30,%rsp
  400b88:	48 89 f8             	mov    %rdi,%rax
  400b8b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b8f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b92:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b96:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b99:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b9d:	48 89 f7             	mov    %rsi,%rdi
  400ba0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ba4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ba7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400bab:	e8 20 00 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bb0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400bb3:	0f af 10             	imul   (%rax),%edx
  400bb6:	89 10                	mov    %edx,(%rax)
  400bb8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400bbc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400bc0:	e8 2b 00 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400bc5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400bc9:	48 83 c4 30          	add    $0x30,%rsp
  400bcd:	5d                   	pop    %rbp
  400bce:	c3                   	retq   
  400bcf:	90                   	nop

0000000000400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 83 ec 10          	sub    $0x10,%rsp
  400bd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bdc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400be0:	e8 1b 00 00 00       	callq  400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be5:	48 83 c4 10          	add    $0x10,%rsp
  400be9:	5d                   	pop    %rbp
  400bea:	c3                   	retq   
  400beb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bf8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bfc:	5d                   	pop    %rbp
  400bfd:	c3                   	retq   
  400bfe:	66 90                	xchg   %ax,%ax

0000000000400c00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 83 ec 30          	sub    $0x30,%rsp
  400c08:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c0f:	00 00 00 
  400c12:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c16:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c1a:	48 c7 45 f0 18 00 00 	movq   $0x18,-0x10(%rbp)
  400c21:	00 
  400c22:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c26:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400c2a:	e8 11 fb ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400c2f:	48 89 c7             	mov    %rax,%rdi
  400c32:	e8 29 00 00 00       	callq  400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c37:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c3b:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400c3f:	48 c1 e0 02          	shl    $0x2,%rax
  400c43:	48 03 45 e0          	add    -0x20(%rbp),%rax
  400c47:	48 83 c0 00          	add    $0x0,%rax
  400c4b:	48 83 c4 30          	add    $0x30,%rsp
  400c4f:	5d                   	pop    %rbp
  400c50:	c3                   	retq   
  400c51:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c58:	00 00 00 
  400c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c6c:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c70:	5d                   	pop    %rbp
  400c71:	c3                   	retq   
  400c72:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c79:	00 00 00 
  400c7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 83 ec 30          	sub    $0x30,%rsp
  400c88:	48 89 f8             	mov    %rdi,%rax
  400c8b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c8f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c92:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c96:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c99:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c9d:	48 89 f7             	mov    %rsi,%rdi
  400ca0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ca4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ca7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400cab:	e8 20 00 00 00       	callq  400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cb0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400cb3:	0f af 10             	imul   (%rax),%edx
  400cb6:	89 10                	mov    %edx,(%rax)
  400cb8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cbc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cc0:	e8 2b 00 00 00       	callq  400cf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400cc5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cc9:	48 83 c4 30          	add    $0x30,%rsp
  400ccd:	5d                   	pop    %rbp
  400cce:	c3                   	retq   
  400ccf:	90                   	nop

0000000000400cd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cd0:	55                   	push   %rbp
  400cd1:	48 89 e5             	mov    %rsp,%rbp
  400cd4:	48 83 ec 10          	sub    $0x10,%rsp
  400cd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cdc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ce0:	e8 1b 00 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ce5:	48 83 c4 10          	add    $0x10,%rsp
  400ce9:	5d                   	pop    %rbp
  400cea:	c3                   	retq   
  400ceb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400cf0:	55                   	push   %rbp
  400cf1:	48 89 e5             	mov    %rsp,%rbp
  400cf4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cf8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400cfc:	5d                   	pop    %rbp
  400cfd:	c3                   	retq   
  400cfe:	66 90                	xchg   %ax,%ax

0000000000400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 30          	sub    $0x30,%rsp
  400d08:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400d0f:	00 00 00 
  400d12:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d16:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d1a:	48 c7 45 f0 18 00 00 	movq   $0x18,-0x10(%rbp)
  400d21:	00 
  400d22:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d26:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400d2a:	e8 11 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400d2f:	48 89 c7             	mov    %rax,%rdi
  400d32:	e8 29 ff ff ff       	callq  400c60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d37:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d3b:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400d3f:	48 c1 e0 02          	shl    $0x2,%rax
  400d43:	48 03 45 e0          	add    -0x20(%rbp),%rax
  400d47:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d4d:	48 83 c4 30          	add    $0x30,%rsp
  400d51:	5d                   	pop    %rbp
  400d52:	c3                   	retq   
  400d53:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d5a:	00 00 00 
  400d5d:	0f 1f 00             	nopl   (%rax)

0000000000400d60 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d60:	55                   	push   %rbp
  400d61:	48 89 e5             	mov    %rsp,%rbp
  400d64:	48 83 ec 10          	sub    $0x10,%rsp
  400d68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d6c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d70:	48 89 f8             	mov    %rdi,%rax
  400d73:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d77:	48 89 c7             	mov    %rax,%rdi
  400d7a:	e8 31 00 00 00       	callq  400db0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d7f:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400d84:	89 cf                	mov    %ecx,%edi
  400d86:	e8 65 f8 ff ff       	callq  4005f0 <malloc@plt>
  400d8b:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400d8f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400d93:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400d9a:	00 
  400d9b:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400da2:	00 
  400da3:	48 83 c4 10          	add    $0x10,%rsp
  400da7:	5d                   	pop    %rbp
  400da8:	c3                   	retq   
  400da9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400db0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400db0:	55                   	push   %rbp
  400db1:	48 89 e5             	mov    %rsp,%rbp
  400db4:	48 83 ec 10          	sub    $0x10,%rsp
  400db8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dbc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400dc0:	e8 0b 00 00 00       	callq  400dd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400dc5:	48 83 c4 10          	add    $0x10,%rsp
  400dc9:	5d                   	pop    %rbp
  400dca:	c3                   	retq   
  400dcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dd0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400dd0:	55                   	push   %rbp
  400dd1:	48 89 e5             	mov    %rsp,%rbp
  400dd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dd8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ddc:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400de3:	5d                   	pop    %rbp
  400de4:	c3                   	retq   
  400de5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dec:	00 00 00 
  400def:	90                   	nop

0000000000400df0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400df0:	55                   	push   %rbp
  400df1:	48 89 e5             	mov    %rsp,%rbp
  400df4:	5d                   	pop    %rbp
  400df5:	c3                   	retq   
  400df6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dfd:	00 00 00 

0000000000400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400e00:	55                   	push   %rbp
  400e01:	48 89 e5             	mov    %rsp,%rbp
  400e04:	48 83 ec 10          	sub    $0x10,%rsp
  400e08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e10:	48 83 c7 01          	add    $0x1,%rdi
  400e14:	e8 37 00 00 00       	callq  400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e19:	48 83 c4 10          	add    $0x10,%rsp
  400e1d:	5d                   	pop    %rbp
  400e1e:	c3                   	retq   
  400e1f:	90                   	nop

0000000000400e20 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e20:	55                   	push   %rbp
  400e21:	48 89 e5             	mov    %rsp,%rbp
  400e24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400e2c:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400e30:	48 8b 3e             	mov    (%rsi),%rdi
  400e33:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400e37:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400e3b:	48 03 3e             	add    (%rsi),%rdi
  400e3e:	48 89 3e             	mov    %rdi,(%rsi)
  400e41:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400e45:	5d                   	pop    %rbp
  400e46:	c3                   	retq   
  400e47:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400e4e:	00 00 

0000000000400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e50:	55                   	push   %rbp
  400e51:	48 89 e5             	mov    %rsp,%rbp
  400e54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e58:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400e5c:	5d                   	pop    %rbp
  400e5d:	c3                   	retq   
  400e5e:	66 90                	xchg   %ax,%ax

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
  400e8c:	e8 17 f7 ff ff       	callq  4005a8 <_init>
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
