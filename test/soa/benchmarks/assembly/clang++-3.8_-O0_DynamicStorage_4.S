
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
  40064f:	49 c7 c0 f0 0e 40 00 	mov    $0x400ef0,%r8
  400656:	48 c7 c1 80 0e 40 00 	mov    $0x400e80,%rcx
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
  40075f:	e8 ac 02 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageEv>
  400764:	b8 04 00 00 00       	mov    $0x4,%eax
  400769:	89 c7                	mov    %eax,%edi
  40076b:	e8 c0 02 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400770:	48 89 c7             	mov    %rax,%rdi
  400773:	48 89 c1             	mov    %rax,%rcx
  400776:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40077c:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400787:	e8 d4 02 00 00       	callq  400a60 <_ZN9TestClassC2Ev>
  40078c:	e9 00 00 00 00       	jmpq   400791 <main+0x41>
  400791:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400795:	ba 77 77 00 00       	mov    $0x7777,%edx
  40079a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40079e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4007a2:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007a6:	48 83 c1 04          	add    $0x4,%rcx
  4007aa:	48 89 ce             	mov    %rcx,%rsi
  4007ad:	e8 fe 02 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4007b2:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  4007b6:	ba 88 88 00 00       	mov    $0x8888,%edx
  4007bb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007bf:	48 83 c0 04          	add    $0x4,%rax
  4007c3:	48 89 c6             	mov    %rax,%rsi
  4007c6:	e8 35 03 00 00       	callq  400b00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007cb:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007cf:	e8 7c 03 00 00       	callq  400b50 <_ZN9TestClass15increase_field0Ev>
  4007d4:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007d8:	e8 b3 03 00 00       	callq  400b90 <_ZN9TestClass15increase_field1Ev>
  4007dd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007e1:	48 83 c0 04          	add    $0x4,%rax
  4007e5:	48 89 c7             	mov    %rax,%rdi
  4007e8:	e8 e3 03 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007ed:	8b 30                	mov    (%rax),%esi
  4007ef:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007f3:	48 83 c0 04          	add    $0x4,%rax
  4007f7:	48 89 c7             	mov    %rax,%rdi
  4007fa:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007fd:	e8 ee 03 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400802:	48 bf 04 0f 40 00 00 	movabs $0x400f04,%rdi
  400809:	00 00 00 
  40080c:	8b 10                	mov    (%rax),%edx
  40080e:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  400811:	b0 00                	mov    $0x0,%al
  400813:	e8 c8 fd ff ff       	callq  4005e0 <printf@plt>
  400818:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  40081c:	48 83 c1 04          	add    $0x4,%rcx
  400820:	48 89 cf             	mov    %rcx,%rdi
  400823:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400826:	e8 a5 03 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40082b:	31 d2                	xor    %edx,%edx
  40082d:	41 88 d0             	mov    %dl,%r8b
  400830:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400836:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  40083a:	0f 85 1c 00 00 00    	jne    40085c <main+0x10c>
  400840:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400844:	48 83 c0 04          	add    $0x4,%rax
  400848:	48 89 c7             	mov    %rax,%rdi
  40084b:	e8 a0 03 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400850:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400856:	0f 94 c1             	sete   %cl
  400859:	88 4d bf             	mov    %cl,-0x41(%rbp)
  40085c:	8a 45 bf             	mov    -0x41(%rbp),%al
  40085f:	34 ff                	xor    $0xff,%al
  400861:	24 01                	and    $0x1,%al
  400863:	0f b6 c0             	movzbl %al,%eax
  400866:	48 83 c4 50          	add    $0x50,%rsp
  40086a:	5d                   	pop    %rbp
  40086b:	c3                   	retq   
  40086c:	89 d1                	mov    %edx,%ecx
  40086e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400872:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400875:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400879:	e8 22 02 00 00       	callq  400aa0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  40087e:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400882:	e8 99 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400887:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40088e:	00 00 

0000000000400890 <_Z12new_instancev>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 83 ec 20          	sub    $0x20,%rsp
  400898:	b8 04 00 00 00       	mov    $0x4,%eax
  40089d:	89 c7                	mov    %eax,%edi
  40089f:	e8 8c 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4008a4:	48 89 c7             	mov    %rax,%rdi
  4008a7:	48 89 c1             	mov    %rax,%rcx
  4008aa:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008b0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008b4:	48 89 c7             	mov    %rax,%rdi
  4008b7:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4008bb:	e8 a0 01 00 00       	callq  400a60 <_ZN9TestClassC2Ev>
  4008c0:	e9 00 00 00 00       	jmpq   4008c5 <_Z12new_instancev+0x35>
  4008c5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008c9:	48 83 c4 20          	add    $0x20,%rsp
  4008cd:	5d                   	pop    %rbp
  4008ce:	c3                   	retq   
  4008cf:	89 d1                	mov    %edx,%ecx
  4008d1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008d5:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  4008d8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008dc:	e8 bf 01 00 00       	callq  400aa0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008e1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e5:	e8 36 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008f0 <_Z12write_field0P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 20          	sub    $0x20,%rsp
  4008f8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008fc:	ba 77 77 00 00       	mov    $0x7777,%edx
  400901:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400905:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400909:	48 83 c7 04          	add    $0x4,%rdi
  40090d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400911:	48 89 c7             	mov    %rax,%rdi
  400914:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400918:	e8 93 01 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40091d:	48 83 c4 20          	add    $0x20,%rsp
  400921:	5d                   	pop    %rbp
  400922:	c3                   	retq   
  400923:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40092a:	84 00 00 00 00 00 

0000000000400930 <_Z11read_field0P9TestClass>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40093c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400940:	48 83 c7 04          	add    $0x4,%rdi
  400944:	e8 87 02 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400949:	8b 00                	mov    (%rax),%eax
  40094b:	48 83 c4 10          	add    $0x10,%rsp
  40094f:	5d                   	pop    %rbp
  400950:	c3                   	retq   
  400951:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400958:	0f 1f 84 00 00 00 00 
  40095f:	00 

0000000000400960 <_Z12write_field1P9TestClass>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 83 ec 20          	sub    $0x20,%rsp
  400968:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40096c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400971:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400975:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400979:	48 83 c7 04          	add    $0x4,%rdi
  40097d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400981:	48 89 c7             	mov    %rax,%rdi
  400984:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400988:	e8 73 01 00 00       	callq  400b00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40098d:	48 83 c4 20          	add    $0x20,%rsp
  400991:	5d                   	pop    %rbp
  400992:	c3                   	retq   
  400993:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40099a:	84 00 00 00 00 00 

00000000004009a0 <_Z11read_field1P9TestClass>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 83 ec 10          	sub    $0x10,%rsp
  4009a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009b0:	48 83 c7 04          	add    $0x4,%rdi
  4009b4:	e8 37 02 00 00       	callq  400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4009b9:	8b 00                	mov    (%rax),%eax
  4009bb:	48 83 c4 10          	add    $0x10,%rsp
  4009bf:	5d                   	pop    %rbp
  4009c0:	c3                   	retq   
  4009c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009c8:	0f 1f 84 00 00 00 00 
  4009cf:	00 

00000000004009d0 <_Z21explicit_write_field0m>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009dc:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  4009e3:	77 77 00 00 
  4009e7:	5d                   	pop    %rbp
  4009e8:	c3                   	retq   
  4009e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009f0 <_Z20explicit_read_field0m>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009fc:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  400a03:	5d                   	pop    %rbp
  400a04:	c3                   	retq   
  400a05:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a0c:	00 00 00 
  400a0f:	90                   	nop

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageEv>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	e8 27 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400a19:	48 89 c7             	mov    %rax,%rdi
  400a1c:	e8 af 03 00 00       	callq  400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  400a21:	5d                   	pop    %rbp
  400a22:	c3                   	retq   
  400a23:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2a:	00 00 00 
  400a2d:	0f 1f 00             	nopl   (%rax)

0000000000400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 83 ec 10          	sub    $0x10,%rsp
  400a38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a3c:	e8 df 03 00 00       	callq  400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a41:	e8 fa fc ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400a46:	48 8b 38             	mov    (%rax),%rdi
  400a49:	48 89 f9             	mov    %rdi,%rcx
  400a4c:	48 83 c1 01          	add    $0x1,%rcx
  400a50:	48 89 08             	mov    %rcx,(%rax)
  400a53:	e8 d8 03 00 00       	callq  400e30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a58:	48 83 c4 10          	add    $0x10,%rsp
  400a5c:	5d                   	pop    %rbp
  400a5d:	c3                   	retq   
  400a5e:	66 90                	xchg   %ax,%ax

0000000000400a60 <_ZN9TestClassC2Ev>:
  400a60:	55                   	push   %rbp
  400a61:	48 89 e5             	mov    %rsp,%rbp
  400a64:	48 83 ec 10          	sub    $0x10,%rsp
  400a68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a6c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a70:	48 89 f8             	mov    %rdi,%rax
  400a73:	48 83 c0 04          	add    $0x4,%rax
  400a77:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a7b:	48 89 c7             	mov    %rax,%rdi
  400a7e:	e8 8d 01 00 00       	callq  400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a83:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a87:	48 83 c0 04          	add    $0x4,%rax
  400a8b:	48 89 c7             	mov    %rax,%rdi
  400a8e:	e8 8d 01 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a93:	48 83 c4 10          	add    $0x10,%rsp
  400a97:	5d                   	pop    %rbp
  400a98:	c3                   	retq   
  400a99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400aa0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	5d                   	pop    %rbp
  400aa9:	c3                   	retq   
  400aaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400ab0:	55                   	push   %rbp
  400ab1:	48 89 e5             	mov    %rsp,%rbp
  400ab4:	48 83 ec 30          	sub    $0x30,%rsp
  400ab8:	48 89 f8             	mov    %rdi,%rax
  400abb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400abf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400ac2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ac6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400ac9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400acd:	48 89 f7             	mov    %rsi,%rdi
  400ad0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ad4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ad7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400adb:	e8 a0 01 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ae3:	89 10                	mov    %edx,(%rax)
  400ae5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ae9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400aed:	e8 ae 01 00 00       	callq  400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400af2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400af6:	48 83 c4 30          	add    $0x30,%rsp
  400afa:	5d                   	pop    %rbp
  400afb:	c3                   	retq   
  400afc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 30          	sub    $0x30,%rsp
  400b08:	48 89 f8             	mov    %rdi,%rax
  400b0b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400b0f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b12:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b16:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b19:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b1d:	48 89 f7             	mov    %rsi,%rdi
  400b20:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b24:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b27:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b2b:	e8 20 02 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b33:	89 10                	mov    %edx,(%rax)
  400b35:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b39:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b3d:	e8 2e 02 00 00       	callq  400d70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400b42:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b46:	48 83 c4 30          	add    $0x30,%rsp
  400b4a:	5d                   	pop    %rbp
  400b4b:	c3                   	retq   
  400b4c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b50 <_ZN9TestClass15increase_field0Ev>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 83 ec 20          	sub    $0x20,%rsp
  400b58:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b5c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b61:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b65:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b69:	48 83 c7 04          	add    $0x4,%rdi
  400b6d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b71:	48 89 c7             	mov    %rax,%rdi
  400b74:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b78:	e8 b3 00 00 00       	callq  400c30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b7d:	48 83 c4 20          	add    $0x20,%rsp
  400b81:	5d                   	pop    %rbp
  400b82:	c3                   	retq   
  400b83:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b8a:	00 00 00 
  400b8d:	0f 1f 00             	nopl   (%rax)

0000000000400b90 <_ZN9TestClass15increase_field1Ev>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 83 ec 20          	sub    $0x20,%rsp
  400b98:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b9c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ba1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ba5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ba9:	48 83 c7 04          	add    $0x4,%rdi
  400bad:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bb1:	48 89 c7             	mov    %rax,%rdi
  400bb4:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400bb8:	e8 43 01 00 00       	callq  400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400bbd:	48 83 c4 20          	add    $0x20,%rsp
  400bc1:	5d                   	pop    %rbp
  400bc2:	c3                   	retq   
  400bc3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bca:	00 00 00 
  400bcd:	0f 1f 00             	nopl   (%rax)

0000000000400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 83 ec 10          	sub    $0x10,%rsp
  400bd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bdc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400be0:	e8 9b 00 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be5:	48 83 c4 10          	add    $0x10,%rsp
  400be9:	5d                   	pop    %rbp
  400bea:	c3                   	retq   
  400beb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bf0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 83 ec 10          	sub    $0x10,%rsp
  400bf8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bfc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c00:	e8 4b 01 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c05:	48 83 c4 10          	add    $0x10,%rsp
  400c09:	5d                   	pop    %rbp
  400c0a:	c3                   	retq   
  400c0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400c10:	55                   	push   %rbp
  400c11:	48 89 e5             	mov    %rsp,%rbp
  400c14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c18:	5d                   	pop    %rbp
  400c19:	c3                   	retq   
  400c1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400c20:	55                   	push   %rbp
  400c21:	48 89 e5             	mov    %rsp,%rbp
  400c24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c28:	5d                   	pop    %rbp
  400c29:	c3                   	retq   
  400c2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400c30:	55                   	push   %rbp
  400c31:	48 89 e5             	mov    %rsp,%rbp
  400c34:	48 83 ec 30          	sub    $0x30,%rsp
  400c38:	48 89 f8             	mov    %rdi,%rax
  400c3b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c3f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c42:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c46:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c49:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c4d:	48 89 f7             	mov    %rsi,%rdi
  400c50:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c54:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c57:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c5b:	e8 20 00 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c60:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c63:	0f af 10             	imul   (%rax),%edx
  400c66:	89 10                	mov    %edx,(%rax)
  400c68:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c6c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c70:	e8 2b 00 00 00       	callq  400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c75:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c79:	48 83 c4 30          	add    $0x30,%rsp
  400c7d:	5d                   	pop    %rbp
  400c7e:	c3                   	retq   
  400c7f:	90                   	nop

0000000000400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 83 ec 10          	sub    $0x10,%rsp
  400c88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c90:	e8 1b 00 00 00       	callq  400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c95:	48 83 c4 10          	add    $0x10,%rsp
  400c99:	5d                   	pop    %rbp
  400c9a:	c3                   	retq   
  400c9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ca8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400cac:	5d                   	pop    %rbp
  400cad:	c3                   	retq   
  400cae:	66 90                	xchg   %ax,%ax

0000000000400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 83 ec 20          	sub    $0x20,%rsp
  400cb8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cbc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cc0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400cc4:	e8 77 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400cc9:	48 8b 40 08          	mov    0x8(%rax),%rax
  400ccd:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400cd1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400cd5:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400cd9:	48 83 e8 04          	sub    $0x4,%rax
  400cdd:	48 c1 e8 02          	shr    $0x2,%rax
  400ce1:	48 c1 e0 02          	shl    $0x2,%rax
  400ce5:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400ce9:	48 83 c0 00          	add    $0x0,%rax
  400ced:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400cf1:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cf5:	48 83 c4 20          	add    $0x20,%rsp
  400cf9:	5d                   	pop    %rbp
  400cfa:	c3                   	retq   
  400cfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 30          	sub    $0x30,%rsp
  400d08:	48 89 f8             	mov    %rdi,%rax
  400d0b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400d0f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400d12:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400d16:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400d19:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400d1d:	48 89 f7             	mov    %rsi,%rdi
  400d20:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d24:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400d27:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400d2b:	e8 20 00 00 00       	callq  400d50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400d33:	0f af 10             	imul   (%rax),%edx
  400d36:	89 10                	mov    %edx,(%rax)
  400d38:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400d3c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400d40:	e8 2b 00 00 00       	callq  400d70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400d45:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d49:	48 83 c4 30          	add    $0x30,%rsp
  400d4d:	5d                   	pop    %rbp
  400d4e:	c3                   	retq   
  400d4f:	90                   	nop

0000000000400d50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	48 83 ec 10          	sub    $0x10,%rsp
  400d58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d60:	e8 1b 00 00 00       	callq  400d80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d65:	48 83 c4 10          	add    $0x10,%rsp
  400d69:	5d                   	pop    %rbp
  400d6a:	c3                   	retq   
  400d6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d70:	55                   	push   %rbp
  400d71:	48 89 e5             	mov    %rsp,%rbp
  400d74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d78:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d7c:	5d                   	pop    %rbp
  400d7d:	c3                   	retq   
  400d7e:	66 90                	xchg   %ax,%ax

0000000000400d80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d80:	55                   	push   %rbp
  400d81:	48 89 e5             	mov    %rsp,%rbp
  400d84:	48 83 ec 20          	sub    $0x20,%rsp
  400d88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d90:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d94:	e8 a7 f9 ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400d99:	48 8b 40 08          	mov    0x8(%rax),%rax
  400d9d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400da1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400da5:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400da9:	48 83 e8 04          	sub    $0x4,%rax
  400dad:	48 c1 e8 02          	shr    $0x2,%rax
  400db1:	48 c1 e0 02          	shl    $0x2,%rax
  400db5:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400db9:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400dbf:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400dc3:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400dc7:	48 83 c4 20          	add    $0x20,%rsp
  400dcb:	5d                   	pop    %rbp
  400dcc:	c3                   	retq   
  400dcd:	0f 1f 00             	nopl   (%rax)

0000000000400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400dd0:	55                   	push   %rbp
  400dd1:	48 89 e5             	mov    %rsp,%rbp
  400dd4:	48 83 ec 10          	sub    $0x10,%rsp
  400dd8:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  400ddd:	89 c1                	mov    %eax,%ecx
  400ddf:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400de3:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400de7:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400deb:	48 89 cf             	mov    %rcx,%rdi
  400dee:	e8 fd f7 ff ff       	callq  4005f0 <malloc@plt>
  400df3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400df7:	48 89 41 08          	mov    %rax,0x8(%rcx)
  400dfb:	48 c7 41 18 00 00 00 	movq   $0x0,0x18(%rcx)
  400e02:	00 
  400e03:	48 c7 41 10 00 00 00 	movq   $0x0,0x10(%rcx)
  400e0a:	00 
  400e0b:	48 83 c4 10          	add    $0x10,%rsp
  400e0f:	5d                   	pop    %rbp
  400e10:	c3                   	retq   
  400e11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e18:	00 00 00 
  400e1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e20:	55                   	push   %rbp
  400e21:	48 89 e5             	mov    %rsp,%rbp
  400e24:	5d                   	pop    %rbp
  400e25:	c3                   	retq   
  400e26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2d:	00 00 00 

0000000000400e30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400e30:	55                   	push   %rbp
  400e31:	48 89 e5             	mov    %rsp,%rbp
  400e34:	48 83 ec 10          	sub    $0x10,%rsp
  400e38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e40:	48 83 c7 01          	add    $0x1,%rdi
  400e44:	e8 07 00 00 00       	callq  400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e49:	48 83 c4 10          	add    $0x10,%rsp
  400e4d:	5d                   	pop    %rbp
  400e4e:	c3                   	retq   
  400e4f:	90                   	nop

0000000000400e50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e50:	55                   	push   %rbp
  400e51:	48 89 e5             	mov    %rsp,%rbp
  400e54:	48 83 ec 10          	sub    $0x10,%rsp
  400e58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e5c:	e8 df f8 ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400e61:	48 8b 40 08          	mov    0x8(%rax),%rax
  400e65:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e69:	48 c1 e7 02          	shl    $0x2,%rdi
  400e6d:	48 01 f8             	add    %rdi,%rax
  400e70:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400e74:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400e78:	48 83 c4 10          	add    $0x10,%rsp
  400e7c:	5d                   	pop    %rbp
  400e7d:	c3                   	retq   
  400e7e:	66 90                	xchg   %ax,%ax

0000000000400e80 <__libc_csu_init>:
  400e80:	41 57                	push   %r15
  400e82:	41 56                	push   %r14
  400e84:	41 89 ff             	mov    %edi,%r15d
  400e87:	41 55                	push   %r13
  400e89:	41 54                	push   %r12
  400e8b:	4c 8d 25 4e 0f 20 00 	lea    0x200f4e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e92:	55                   	push   %rbp
  400e93:	48 8d 2d 4e 0f 20 00 	lea    0x200f4e(%rip),%rbp        # 601de8 <__init_array_end>
  400e9a:	53                   	push   %rbx
  400e9b:	49 89 f6             	mov    %rsi,%r14
  400e9e:	49 89 d5             	mov    %rdx,%r13
  400ea1:	4c 29 e5             	sub    %r12,%rbp
  400ea4:	48 83 ec 08          	sub    $0x8,%rsp
  400ea8:	48 c1 fd 03          	sar    $0x3,%rbp
  400eac:	e8 f7 f6 ff ff       	callq  4005a8 <_init>
  400eb1:	48 85 ed             	test   %rbp,%rbp
  400eb4:	74 20                	je     400ed6 <__libc_csu_init+0x56>
  400eb6:	31 db                	xor    %ebx,%ebx
  400eb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ebf:	00 
  400ec0:	4c 89 ea             	mov    %r13,%rdx
  400ec3:	4c 89 f6             	mov    %r14,%rsi
  400ec6:	44 89 ff             	mov    %r15d,%edi
  400ec9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400ecd:	48 83 c3 01          	add    $0x1,%rbx
  400ed1:	48 39 eb             	cmp    %rbp,%rbx
  400ed4:	75 ea                	jne    400ec0 <__libc_csu_init+0x40>
  400ed6:	48 83 c4 08          	add    $0x8,%rsp
  400eda:	5b                   	pop    %rbx
  400edb:	5d                   	pop    %rbp
  400edc:	41 5c                	pop    %r12
  400ede:	41 5d                	pop    %r13
  400ee0:	41 5e                	pop    %r14
  400ee2:	41 5f                	pop    %r15
  400ee4:	c3                   	retq   
  400ee5:	90                   	nop
  400ee6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400eed:	00 00 00 

0000000000400ef0 <__libc_csu_fini>:
  400ef0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ef4 <_fini>:
  400ef4:	48 83 ec 08          	sub    $0x8,%rsp
  400ef8:	48 83 c4 08          	add    $0x8,%rsp
  400efc:	c3                   	retq   
