
bin/valid_clang38_O0:     file format elf64-x86-64


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

0000000000400640 <__cxx_global_var_init>:
  400640:	55                   	push   %rbp
  400641:	48 89 e5             	mov    %rsp,%rbp
  400644:	48 bf 58 20 60 00 00 	movabs $0x602058,%rdi
  40064b:	00 00 00 
  40064e:	e8 6d 03 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE15InternalStorageC2Ev>
  400653:	5d                   	pop    %rbp
  400654:	c3                   	retq   
  400655:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40065c:	00 00 00 00 

0000000000400660 <_GLOBAL__sub_I_codegen_test.cc>:
  400660:	55                   	push   %rbp
  400661:	48 89 e5             	mov    %rsp,%rbp
  400664:	e8 d7 ff ff ff       	callq  400640 <__cxx_global_var_init>
  400669:	5d                   	pop    %rbp
  40066a:	c3                   	retq   
  40066b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400670 <_start>:
  400670:	31 ed                	xor    %ebp,%ebp
  400672:	49 89 d1             	mov    %rdx,%r9
  400675:	5e                   	pop    %rsi
  400676:	48 89 e2             	mov    %rsp,%rdx
  400679:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40067d:	50                   	push   %rax
  40067e:	54                   	push   %rsp
  40067f:	49 c7 c0 40 0e 40 00 	mov    $0x400e40,%r8
  400686:	48 c7 c1 d0 0d 40 00 	mov    $0x400dd0,%rcx
  40068d:	48 c7 c7 70 07 40 00 	mov    $0x400770,%rdi
  400694:	e8 67 ff ff ff       	callq  400600 <__libc_start_main@plt>
  400699:	f4                   	hlt    
  40069a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006a0 <deregister_tm_clones>:
  4006a0:	b8 57 20 60 00       	mov    $0x602057,%eax
  4006a5:	55                   	push   %rbp
  4006a6:	48 2d 50 20 60 00    	sub    $0x602050,%rax
  4006ac:	48 83 f8 0e          	cmp    $0xe,%rax
  4006b0:	48 89 e5             	mov    %rsp,%rbp
  4006b3:	76 1b                	jbe    4006d0 <deregister_tm_clones+0x30>
  4006b5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ba:	48 85 c0             	test   %rax,%rax
  4006bd:	74 11                	je     4006d0 <deregister_tm_clones+0x30>
  4006bf:	5d                   	pop    %rbp
  4006c0:	bf 50 20 60 00       	mov    $0x602050,%edi
  4006c5:	ff e0                	jmpq   *%rax
  4006c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006ce:	00 00 
  4006d0:	5d                   	pop    %rbp
  4006d1:	c3                   	retq   
  4006d2:	0f 1f 40 00          	nopl   0x0(%rax)
  4006d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006dd:	00 00 00 

00000000004006e0 <register_tm_clones>:
  4006e0:	be 50 20 60 00       	mov    $0x602050,%esi
  4006e5:	55                   	push   %rbp
  4006e6:	48 81 ee 50 20 60 00 	sub    $0x602050,%rsi
  4006ed:	48 c1 fe 03          	sar    $0x3,%rsi
  4006f1:	48 89 e5             	mov    %rsp,%rbp
  4006f4:	48 89 f0             	mov    %rsi,%rax
  4006f7:	48 c1 e8 3f          	shr    $0x3f,%rax
  4006fb:	48 01 c6             	add    %rax,%rsi
  4006fe:	48 d1 fe             	sar    %rsi
  400701:	74 15                	je     400718 <register_tm_clones+0x38>
  400703:	b8 00 00 00 00       	mov    $0x0,%eax
  400708:	48 85 c0             	test   %rax,%rax
  40070b:	74 0b                	je     400718 <register_tm_clones+0x38>
  40070d:	5d                   	pop    %rbp
  40070e:	bf 50 20 60 00       	mov    $0x602050,%edi
  400713:	ff e0                	jmpq   *%rax
  400715:	0f 1f 00             	nopl   (%rax)
  400718:	5d                   	pop    %rbp
  400719:	c3                   	retq   
  40071a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400720 <__do_global_dtors_aux>:
  400720:	80 3d 29 19 20 00 00 	cmpb   $0x0,0x201929(%rip)        # 602050 <__TMC_END__>
  400727:	75 11                	jne    40073a <__do_global_dtors_aux+0x1a>
  400729:	55                   	push   %rbp
  40072a:	48 89 e5             	mov    %rsp,%rbp
  40072d:	e8 6e ff ff ff       	callq  4006a0 <deregister_tm_clones>
  400732:	5d                   	pop    %rbp
  400733:	c6 05 16 19 20 00 01 	movb   $0x1,0x201916(%rip)        # 602050 <__TMC_END__>
  40073a:	f3 c3                	repz retq 
  40073c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400740 <frame_dummy>:
  400740:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  400745:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400749:	75 05                	jne    400750 <frame_dummy+0x10>
  40074b:	eb 93                	jmp    4006e0 <register_tm_clones>
  40074d:	0f 1f 00             	nopl   (%rax)
  400750:	b8 00 00 00 00       	mov    $0x0,%eax
  400755:	48 85 c0             	test   %rax,%rax
  400758:	74 f1                	je     40074b <frame_dummy+0xb>
  40075a:	55                   	push   %rbp
  40075b:	48 89 e5             	mov    %rsp,%rbp
  40075e:	ff d0                	callq  *%rax
  400760:	5d                   	pop    %rbp
  400761:	e9 7a ff ff ff       	jmpq   4006e0 <register_tm_clones>
  400766:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40076d:	00 00 00 

0000000000400770 <main>:
  400770:	55                   	push   %rbp
  400771:	48 89 e5             	mov    %rsp,%rbp
  400774:	48 83 ec 50          	sub    $0x50,%rsp
  400778:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  40077f:	b8 04 00 00 00       	mov    $0x4,%eax
  400784:	89 c7                	mov    %eax,%edi
  400786:	e8 75 02 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>
  40078b:	48 89 c7             	mov    %rax,%rdi
  40078e:	48 89 c1             	mov    %rax,%rcx
  400791:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400797:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  40079b:	48 89 c7             	mov    %rax,%rdi
  40079e:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  4007a2:	e8 99 02 00 00       	callq  400a40 <_ZN9TestClassC2Ev>
  4007a7:	e9 00 00 00 00       	jmpq   4007ac <main+0x3c>
  4007ac:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  4007b0:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007b5:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  4007b9:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4007bd:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007c1:	48 83 c1 04          	add    $0x4,%rcx
  4007c5:	48 89 ce             	mov    %rcx,%rsi
  4007c8:	e8 c3 02 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4007cd:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  4007d1:	ba 88 88 00 00       	mov    $0x8888,%edx
  4007d6:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007da:	48 83 c0 04          	add    $0x4,%rax
  4007de:	48 89 c6             	mov    %rax,%rsi
  4007e1:	e8 fa 02 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007e6:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007ea:	e8 41 03 00 00       	callq  400b30 <_ZN9TestClass15increase_field0Ev>
  4007ef:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007f3:	e8 78 03 00 00       	callq  400b70 <_ZN9TestClass15increase_field1Ev>
  4007f8:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007fc:	48 83 c0 04          	add    $0x4,%rax
  400800:	48 89 c7             	mov    %rax,%rdi
  400803:	e8 a8 03 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400808:	8b 30                	mov    (%rax),%esi
  40080a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40080e:	48 83 c0 04          	add    $0x4,%rax
  400812:	48 89 c7             	mov    %rax,%rdi
  400815:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400818:	e8 b3 03 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40081d:	48 bf 54 0e 40 00 00 	movabs $0x400e54,%rdi
  400824:	00 00 00 
  400827:	8b 10                	mov    (%rax),%edx
  400829:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  40082c:	b0 00                	mov    $0x0,%al
  40082e:	e8 ad fd ff ff       	callq  4005e0 <printf@plt>
  400833:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400837:	48 83 c1 04          	add    $0x4,%rcx
  40083b:	48 89 cf             	mov    %rcx,%rdi
  40083e:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400841:	e8 6a 03 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400846:	31 d2                	xor    %edx,%edx
  400848:	41 88 d0             	mov    %dl,%r8b
  40084b:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400851:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  400855:	0f 85 1c 00 00 00    	jne    400877 <main+0x107>
  40085b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40085f:	48 83 c0 04          	add    $0x4,%rax
  400863:	48 89 c7             	mov    %rax,%rdi
  400866:	e8 65 03 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40086b:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400871:	0f 94 c1             	sete   %cl
  400874:	88 4d bf             	mov    %cl,-0x41(%rbp)
  400877:	8a 45 bf             	mov    -0x41(%rbp),%al
  40087a:	34 ff                	xor    $0xff,%al
  40087c:	24 01                	and    $0x1,%al
  40087e:	0f b6 c0             	movzbl %al,%eax
  400881:	48 83 c4 50          	add    $0x50,%rsp
  400885:	5d                   	pop    %rbp
  400886:	c3                   	retq   
  400887:	89 d1                	mov    %edx,%ecx
  400889:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40088d:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400890:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400894:	e8 e7 01 00 00       	callq  400a80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEdlEPv>
  400899:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40089d:	e8 7e fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008a9:	1f 84 00 00 00 00 00 

00000000004008b0 <_Z12new_instancev>:
  4008b0:	55                   	push   %rbp
  4008b1:	48 89 e5             	mov    %rsp,%rbp
  4008b4:	48 83 ec 20          	sub    $0x20,%rsp
  4008b8:	b8 04 00 00 00       	mov    $0x4,%eax
  4008bd:	89 c7                	mov    %eax,%edi
  4008bf:	e8 3c 01 00 00       	callq  400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>
  4008c4:	48 89 c7             	mov    %rax,%rdi
  4008c7:	48 89 c1             	mov    %rax,%rcx
  4008ca:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008d0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008d4:	48 89 c7             	mov    %rax,%rdi
  4008d7:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4008db:	e8 60 01 00 00       	callq  400a40 <_ZN9TestClassC2Ev>
  4008e0:	e9 00 00 00 00       	jmpq   4008e5 <_Z12new_instancev+0x35>
  4008e5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008e9:	48 83 c4 20          	add    $0x20,%rsp
  4008ed:	5d                   	pop    %rbp
  4008ee:	c3                   	retq   
  4008ef:	89 d1                	mov    %edx,%ecx
  4008f1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008f5:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  4008f8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008fc:	e8 7f 01 00 00       	callq  400a80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEdlEPv>
  400901:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400905:	e8 16 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  40090a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400910 <_Z12write_field0P9TestClass>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 20          	sub    $0x20,%rsp
  400918:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40091c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400921:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400925:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400929:	48 83 c7 04          	add    $0x4,%rdi
  40092d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400931:	48 89 c7             	mov    %rax,%rdi
  400934:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400938:	e8 53 01 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40093d:	48 83 c4 20          	add    $0x20,%rsp
  400941:	5d                   	pop    %rbp
  400942:	c3                   	retq   
  400943:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40094a:	84 00 00 00 00 00 

0000000000400950 <_Z11read_field0P9TestClass>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 83 ec 10          	sub    $0x10,%rsp
  400958:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40095c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400960:	48 83 c7 04          	add    $0x4,%rdi
  400964:	e8 47 02 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400969:	8b 00                	mov    (%rax),%eax
  40096b:	48 83 c4 10          	add    $0x10,%rsp
  40096f:	5d                   	pop    %rbp
  400970:	c3                   	retq   
  400971:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400978:	0f 1f 84 00 00 00 00 
  40097f:	00 

0000000000400980 <_Z21explicit_write_field0m>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40098c:	48 8b 04 25 68 20 60 	mov    0x602068,%rax
  400993:	00 
  400994:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40099b:	5d                   	pop    %rbp
  40099c:	c3                   	retq   
  40099d:	0f 1f 00             	nopl   (%rax)

00000000004009a0 <_Z20explicit_read_field0m>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009a8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009ac:	48 8b 04 25 68 20 60 	mov    0x602068,%rax
  4009b3:	00 
  4009b4:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4009b7:	5d                   	pop    %rbp
  4009b8:	c3                   	retq   
  4009b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE15InternalStorageC2Ev>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 83 ec 10          	sub    $0x10,%rsp
  4009c8:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  4009cd:	89 c1                	mov    %eax,%ecx
  4009cf:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d3:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009d7:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009db:	48 89 cf             	mov    %rcx,%rdi
  4009de:	e8 0d fc ff ff       	callq  4005f0 <malloc@plt>
  4009e3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4009e7:	48 89 41 08          	mov    %rax,0x8(%rcx)
  4009eb:	48 83 c4 10          	add    $0x10,%rsp
  4009ef:	5d                   	pop    %rbp
  4009f0:	c3                   	retq   
  4009f1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009f8:	00 00 00 
  4009fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400a00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEnwEm>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 10          	sub    $0x10,%rsp
  400a08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0c:	e8 5f 03 00 00       	callq  400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a11:	8b 04 25 58 20 60 00 	mov    0x602058,%eax
  400a18:	89 c1                	mov    %eax,%ecx
  400a1a:	83 c1 01             	add    $0x1,%ecx
  400a1d:	89 0c 25 58 20 60 00 	mov    %ecx,0x602058
  400a24:	89 c0                	mov    %eax,%eax
  400a26:	89 c7                	mov    %eax,%edi
  400a28:	e8 53 03 00 00       	callq  400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE3getEm>
  400a2d:	48 83 c4 10          	add    $0x10,%rsp
  400a31:	5d                   	pop    %rbp
  400a32:	c3                   	retq   
  400a33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a3a:	00 00 00 
  400a3d:	0f 1f 00             	nopl   (%rax)

0000000000400a40 <_ZN9TestClassC2Ev>:
  400a40:	55                   	push   %rbp
  400a41:	48 89 e5             	mov    %rsp,%rbp
  400a44:	48 83 ec 10          	sub    $0x10,%rsp
  400a48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a50:	48 89 f8             	mov    %rdi,%rax
  400a53:	48 83 c0 04          	add    $0x4,%rax
  400a57:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a5b:	48 89 c7             	mov    %rax,%rdi
  400a5e:	e8 8d 01 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a63:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a67:	48 83 c0 04          	add    $0x4,%rax
  400a6b:	48 89 c7             	mov    %rax,%rdi
  400a6e:	e8 8d 01 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a73:	48 83 c4 10          	add    $0x10,%rsp
  400a77:	5d                   	pop    %rbp
  400a78:	c3                   	retq   
  400a79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EEdlEPv>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a88:	5d                   	pop    %rbp
  400a89:	c3                   	retq   
  400a8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a90:	55                   	push   %rbp
  400a91:	48 89 e5             	mov    %rsp,%rbp
  400a94:	48 83 ec 30          	sub    $0x30,%rsp
  400a98:	48 89 f8             	mov    %rdi,%rax
  400a9b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a9f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400aa2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400aa6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400aa9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400aad:	48 89 f7             	mov    %rsi,%rdi
  400ab0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ab4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ab7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400abb:	e8 a0 01 00 00       	callq  400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ac3:	89 10                	mov    %edx,(%rax)
  400ac5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ac9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400acd:	e8 de 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400ad2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ad6:	48 83 c4 30          	add    $0x30,%rsp
  400ada:	5d                   	pop    %rbp
  400adb:	c3                   	retq   
  400adc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	48 83 ec 30          	sub    $0x30,%rsp
  400ae8:	48 89 f8             	mov    %rdi,%rax
  400aeb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400aef:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400af2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400af6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400af9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400afd:	48 89 f7             	mov    %rsi,%rdi
  400b00:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b04:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b07:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b0b:	e8 00 02 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b10:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b13:	89 10                	mov    %edx,(%rax)
  400b15:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b19:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b1d:	e8 3e 02 00 00       	callq  400d60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400b22:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b26:	48 83 c4 30          	add    $0x30,%rsp
  400b2a:	5d                   	pop    %rbp
  400b2b:	c3                   	retq   
  400b2c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b30 <_ZN9TestClass15increase_field0Ev>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 83 ec 20          	sub    $0x20,%rsp
  400b38:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b3c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b41:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b45:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b49:	48 83 c7 04          	add    $0x4,%rdi
  400b4d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b51:	48 89 c7             	mov    %rax,%rdi
  400b54:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b58:	e8 b3 00 00 00       	callq  400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b5d:	48 83 c4 20          	add    $0x20,%rsp
  400b61:	5d                   	pop    %rbp
  400b62:	c3                   	retq   
  400b63:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b6a:	00 00 00 
  400b6d:	0f 1f 00             	nopl   (%rax)

0000000000400b70 <_ZN9TestClass15increase_field1Ev>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 83 ec 20          	sub    $0x20,%rsp
  400b78:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b7c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b81:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b85:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b89:	48 83 c7 04          	add    $0x4,%rdi
  400b8d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b91:	48 89 c7             	mov    %rax,%rdi
  400b94:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b98:	e8 23 01 00 00       	callq  400cc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b9d:	48 83 c4 20          	add    $0x20,%rsp
  400ba1:	5d                   	pop    %rbp
  400ba2:	c3                   	retq   
  400ba3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400baa:	00 00 00 
  400bad:	0f 1f 00             	nopl   (%rax)

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 83 ec 10          	sub    $0x10,%rsp
  400bb8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bbc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bc0:	e8 9b 00 00 00       	callq  400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bc5:	48 83 c4 10          	add    $0x10,%rsp
  400bc9:	5d                   	pop    %rbp
  400bca:	c3                   	retq   
  400bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 83 ec 10          	sub    $0x10,%rsp
  400bd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bdc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400be0:	e8 2b 01 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be5:	48 83 c4 10          	add    $0x10,%rsp
  400be9:	5d                   	pop    %rbp
  400bea:	c3                   	retq   
  400beb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bf8:	5d                   	pop    %rbp
  400bf9:	c3                   	retq   
  400bfa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c08:	5d                   	pop    %rbp
  400c09:	c3                   	retq   
  400c0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c10 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400c10:	55                   	push   %rbp
  400c11:	48 89 e5             	mov    %rsp,%rbp
  400c14:	48 83 ec 30          	sub    $0x30,%rsp
  400c18:	48 89 f8             	mov    %rdi,%rax
  400c1b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c1f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c22:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c26:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c29:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c2d:	48 89 f7             	mov    %rsi,%rdi
  400c30:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c34:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c37:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c3b:	e8 20 00 00 00       	callq  400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c40:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c43:	0f af 10             	imul   (%rax),%edx
  400c46:	89 10                	mov    %edx,(%rax)
  400c48:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c4c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c50:	e8 5b 00 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c55:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c59:	48 83 c4 30          	add    $0x30,%rsp
  400c5d:	5d                   	pop    %rbp
  400c5e:	c3                   	retq   
  400c5f:	90                   	nop

0000000000400c60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c6c:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c70:	48 8b 3c 25 60 20 60 	mov    0x602060,%rdi
  400c77:	00 
  400c78:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c7c:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400c80:	48 2b 7d e8          	sub    -0x18(%rbp),%rdi
  400c84:	48 83 ef 04          	sub    $0x4,%rdi
  400c88:	48 c1 ef 02          	shr    $0x2,%rdi
  400c8c:	48 c1 e7 02          	shl    $0x2,%rdi
  400c90:	48 03 7d e8          	add    -0x18(%rbp),%rdi
  400c94:	48 83 c7 00          	add    $0x0,%rdi
  400c98:	48 89 7d e0          	mov    %rdi,-0x20(%rbp)
  400c9c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ca0:	5d                   	pop    %rbp
  400ca1:	c3                   	retq   
  400ca2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ca9:	00 00 00 
  400cac:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cb8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400cbc:	5d                   	pop    %rbp
  400cbd:	c3                   	retq   
  400cbe:	66 90                	xchg   %ax,%ax

0000000000400cc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 83 ec 30          	sub    $0x30,%rsp
  400cc8:	48 89 f8             	mov    %rdi,%rax
  400ccb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400ccf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400cd2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400cd6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400cd9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400cdd:	48 89 f7             	mov    %rsi,%rdi
  400ce0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ce4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ce7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400ceb:	e8 20 00 00 00       	callq  400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cf0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400cf3:	0f af 10             	imul   (%rax),%edx
  400cf6:	89 10                	mov    %edx,(%rax)
  400cf8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cfc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400d00:	e8 5b 00 00 00       	callq  400d60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400d05:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d09:	48 83 c4 30          	add    $0x30,%rsp
  400d0d:	5d                   	pop    %rbp
  400d0e:	c3                   	retq   
  400d0f:	90                   	nop

0000000000400d10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d10:	55                   	push   %rbp
  400d11:	48 89 e5             	mov    %rsp,%rbp
  400d14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d18:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d1c:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d20:	48 8b 3c 25 60 20 60 	mov    0x602060,%rdi
  400d27:	00 
  400d28:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400d2c:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400d30:	48 2b 7d e8          	sub    -0x18(%rbp),%rdi
  400d34:	48 83 ef 04          	sub    $0x4,%rdi
  400d38:	48 c1 ef 02          	shr    $0x2,%rdi
  400d3c:	48 c1 e7 02          	shl    $0x2,%rdi
  400d40:	48 03 7d e8          	add    -0x18(%rbp),%rdi
  400d44:	48 81 c7 d0 48 00 00 	add    $0x48d0,%rdi
  400d4b:	48 89 7d e0          	mov    %rdi,-0x20(%rbp)
  400d4f:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d53:	5d                   	pop    %rbp
  400d54:	c3                   	retq   
  400d55:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d5c:	00 00 00 
  400d5f:	90                   	nop

0000000000400d60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d60:	55                   	push   %rbp
  400d61:	48 89 e5             	mov    %rsp,%rbp
  400d64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d68:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d6c:	5d                   	pop    %rbp
  400d6d:	c3                   	retq   
  400d6e:	66 90                	xchg   %ax,%ax

0000000000400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d70:	55                   	push   %rbp
  400d71:	48 89 e5             	mov    %rsp,%rbp
  400d74:	5d                   	pop    %rbp
  400d75:	c3                   	retq   
  400d76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d7d:	00 00 00 

0000000000400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE3getEm>:
  400d80:	55                   	push   %rbp
  400d81:	48 89 e5             	mov    %rsp,%rbp
  400d84:	48 83 ec 10          	sub    $0x10,%rsp
  400d88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d90:	e8 0b 00 00 00       	callq  400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d95:	48 83 c4 10          	add    $0x10,%rsp
  400d99:	5d                   	pop    %rbp
  400d9a:	c3                   	retq   
  400d9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400da0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400da0:	55                   	push   %rbp
  400da1:	48 89 e5             	mov    %rsp,%rbp
  400da4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400da8:	48 8b 3c 25 60 20 60 	mov    0x602060,%rdi
  400daf:	00 
  400db0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400db4:	48 c1 e0 02          	shl    $0x2,%rax
  400db8:	48 01 c7             	add    %rax,%rdi
  400dbb:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400dbf:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400dc3:	5d                   	pop    %rbp
  400dc4:	c3                   	retq   
  400dc5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dcc:	00 00 00 
  400dcf:	90                   	nop

0000000000400dd0 <__libc_csu_init>:
  400dd0:	41 57                	push   %r15
  400dd2:	41 56                	push   %r14
  400dd4:	41 89 ff             	mov    %edi,%r15d
  400dd7:	41 55                	push   %r13
  400dd9:	41 54                	push   %r12
  400ddb:	4c 8d 25 f6 0f 20 00 	lea    0x200ff6(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  400de2:	55                   	push   %rbp
  400de3:	48 8d 2d fe 0f 20 00 	lea    0x200ffe(%rip),%rbp        # 601de8 <__init_array_end>
  400dea:	53                   	push   %rbx
  400deb:	49 89 f6             	mov    %rsi,%r14
  400dee:	49 89 d5             	mov    %rdx,%r13
  400df1:	4c 29 e5             	sub    %r12,%rbp
  400df4:	48 83 ec 08          	sub    $0x8,%rsp
  400df8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dfc:	e8 a7 f7 ff ff       	callq  4005a8 <_init>
  400e01:	48 85 ed             	test   %rbp,%rbp
  400e04:	74 20                	je     400e26 <__libc_csu_init+0x56>
  400e06:	31 db                	xor    %ebx,%ebx
  400e08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e0f:	00 
  400e10:	4c 89 ea             	mov    %r13,%rdx
  400e13:	4c 89 f6             	mov    %r14,%rsi
  400e16:	44 89 ff             	mov    %r15d,%edi
  400e19:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e1d:	48 83 c3 01          	add    $0x1,%rbx
  400e21:	48 39 eb             	cmp    %rbp,%rbx
  400e24:	75 ea                	jne    400e10 <__libc_csu_init+0x40>
  400e26:	48 83 c4 08          	add    $0x8,%rsp
  400e2a:	5b                   	pop    %rbx
  400e2b:	5d                   	pop    %rbp
  400e2c:	41 5c                	pop    %r12
  400e2e:	41 5d                	pop    %r13
  400e30:	41 5e                	pop    %r14
  400e32:	41 5f                	pop    %r15
  400e34:	c3                   	retq   
  400e35:	90                   	nop
  400e36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e3d:	00 00 00 

0000000000400e40 <__libc_csu_fini>:
  400e40:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e44 <_fini>:
  400e44:	48 83 ec 08          	sub    $0x8,%rsp
  400e48:	48 83 c4 08          	add    $0x8,%rsp
  400e4c:	c3                   	retq   
