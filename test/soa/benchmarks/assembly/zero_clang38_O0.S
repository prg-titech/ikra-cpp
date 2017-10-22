
bin/zero_clang38_O0:     file format elf64-x86-64


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
  40064e:	e8 0d 03 00 00       	callq  400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC2Ev>
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
  40067f:	49 c7 c0 40 0d 40 00 	mov    $0x400d40,%r8
  400686:	48 c7 c1 d0 0c 40 00 	mov    $0x400cd0,%rcx
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
  40077f:	31 c0                	xor    %eax,%eax
  400781:	89 c7                	mov    %eax,%edi
  400783:	e8 18 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  400788:	48 89 c7             	mov    %rax,%rdi
  40078b:	48 89 c1             	mov    %rax,%rcx
  40078e:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400792:	48 89 c7             	mov    %rax,%rdi
  400795:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400799:	e8 42 02 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  40079e:	e9 00 00 00 00       	jmpq   4007a3 <main+0x33>
  4007a3:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  4007a7:	ba 77 77 00 00       	mov    $0x7777,%edx
  4007ac:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  4007b0:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4007b4:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  4007b8:	e8 63 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4007bd:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  4007c1:	ba 88 88 00 00       	mov    $0x8888,%edx
  4007c6:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  4007ca:	e8 a1 02 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007cf:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007d3:	e8 e8 02 00 00       	callq  400ac0 <_ZN9TestClass15increase_field0Ev>
  4007d8:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007dc:	e8 0f 03 00 00       	callq  400af0 <_ZN9TestClass15increase_field1Ev>
  4007e1:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007e5:	e8 36 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007ea:	8b 30                	mov    (%rax),%esi
  4007ec:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007f0:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007f3:	e8 48 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007f8:	48 bf 54 0d 40 00 00 	movabs $0x400d54,%rdi
  4007ff:	00 00 00 
  400802:	8b 10                	mov    (%rax),%edx
  400804:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  400807:	b0 00                	mov    $0x0,%al
  400809:	e8 d2 fd ff ff       	callq  4005e0 <printf@plt>
  40080e:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400812:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400815:	e8 06 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40081a:	31 d2                	xor    %edx,%edx
  40081c:	88 d1                	mov    %dl,%cl
  40081e:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400824:	88 4d bf             	mov    %cl,-0x41(%rbp)
  400827:	0f 85 15 00 00 00    	jne    400842 <main+0xd2>
  40082d:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400831:	e8 0a 03 00 00       	callq  400b40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400836:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  40083c:	0f 94 c1             	sete   %cl
  40083f:	88 4d bf             	mov    %cl,-0x41(%rbp)
  400842:	8a 45 bf             	mov    -0x41(%rbp),%al
  400845:	34 ff                	xor    $0xff,%al
  400847:	24 01                	and    $0x1,%al
  400849:	0f b6 c0             	movzbl %al,%eax
  40084c:	48 83 c4 50          	add    $0x50,%rsp
  400850:	5d                   	pop    %rbp
  400851:	c3                   	retq   
  400852:	89 d1                	mov    %edx,%ecx
  400854:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400858:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  40085b:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  40085f:	e8 ac 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEdlEPv>
  400864:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400868:	e8 b3 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  40086d:	0f 1f 00             	nopl   (%rax)

0000000000400870 <_Z12new_instancev>:
  400870:	55                   	push   %rbp
  400871:	48 89 e5             	mov    %rsp,%rbp
  400874:	48 83 ec 20          	sub    $0x20,%rsp
  400878:	31 c0                	xor    %eax,%eax
  40087a:	89 c7                	mov    %eax,%edi
  40087c:	e8 1f 01 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>
  400881:	48 89 c7             	mov    %rax,%rdi
  400884:	48 89 c1             	mov    %rax,%rcx
  400887:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40088b:	48 89 c7             	mov    %rax,%rdi
  40088e:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  400892:	e8 49 01 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  400897:	e9 00 00 00 00       	jmpq   40089c <_Z12new_instancev+0x2c>
  40089c:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008a0:	48 83 c4 20          	add    $0x20,%rsp
  4008a4:	5d                   	pop    %rbp
  4008a5:	c3                   	retq   
  4008a6:	89 d1                	mov    %edx,%ecx
  4008a8:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008ac:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  4008af:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008b3:	e8 58 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEdlEPv>
  4008b8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008bc:	e8 5f fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008c8:	0f 1f 84 00 00 00 00 
  4008cf:	00 

00000000004008d0 <_Z12write_field0P9TestClass>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 10          	sub    $0x10,%rsp
  4008d8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008dc:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008e1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008e5:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4008e9:	48 89 c7             	mov    %rax,%rdi
  4008ec:	e8 2f 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4008f1:	48 83 c4 10          	add    $0x10,%rsp
  4008f5:	5d                   	pop    %rbp
  4008f6:	c3                   	retq   
  4008f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4008fe:	00 00 

0000000000400900 <_Z11read_field0P9TestClass>:
  400900:	55                   	push   %rbp
  400901:	48 89 e5             	mov    %rsp,%rbp
  400904:	48 83 ec 10          	sub    $0x10,%rsp
  400908:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40090c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400910:	e8 0b 02 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400915:	8b 00                	mov    (%rax),%eax
  400917:	48 83 c4 10          	add    $0x10,%rsp
  40091b:	5d                   	pop    %rbp
  40091c:	c3                   	retq   
  40091d:	0f 1f 00             	nopl   (%rax)

0000000000400920 <_Z21explicit_write_field0m>:
  400920:	55                   	push   %rbp
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400928:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40092c:	48 8b 04 25 68 20 60 	mov    0x602068,%rax
  400933:	00 
  400934:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40093b:	5d                   	pop    %rbp
  40093c:	c3                   	retq   
  40093d:	0f 1f 00             	nopl   (%rax)

0000000000400940 <_Z20explicit_read_field0m>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400948:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40094c:	48 8b 04 25 68 20 60 	mov    0x602068,%rax
  400953:	00 
  400954:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  400957:	5d                   	pop    %rbp
  400958:	c3                   	retq   
  400959:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400960 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE15InternalStorageC2Ev>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 83 ec 10          	sub    $0x10,%rsp
  400968:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  40096d:	89 c1                	mov    %eax,%ecx
  40096f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400973:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400977:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  40097b:	48 89 cf             	mov    %rcx,%rdi
  40097e:	e8 6d fc ff ff       	callq  4005f0 <malloc@plt>
  400983:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400987:	48 89 41 08          	mov    %rax,0x8(%rcx)
  40098b:	48 83 c4 10          	add    $0x10,%rsp
  40098f:	5d                   	pop    %rbp
  400990:	c3                   	retq   
  400991:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400998:	00 00 00 
  40099b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEnwEm>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 83 ec 10          	sub    $0x10,%rsp
  4009a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ac:	e8 df 02 00 00       	callq  400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009b1:	8b 04 25 58 20 60 00 	mov    0x602058,%eax
  4009b8:	89 c1                	mov    %eax,%ecx
  4009ba:	83 c1 01             	add    $0x1,%ecx
  4009bd:	89 0c 25 58 20 60 00 	mov    %ecx,0x602058
  4009c4:	89 c0                	mov    %eax,%eax
  4009c6:	89 c7                	mov    %eax,%edi
  4009c8:	e8 d3 02 00 00       	callq  400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>
  4009cd:	48 83 c4 10          	add    $0x10,%rsp
  4009d1:	5d                   	pop    %rbp
  4009d2:	c3                   	retq   
  4009d3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009da:	00 00 00 
  4009dd:	0f 1f 00             	nopl   (%rax)

00000000004009e0 <_ZN9TestClassC2Ev>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 83 ec 10          	sub    $0x10,%rsp
  4009e8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009ec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009f0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  4009f4:	e8 67 01 00 00       	callq  400b60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  4009f9:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4009fd:	e8 6e 01 00 00       	callq  400b70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400a02:	48 83 c4 10          	add    $0x10,%rsp
  400a06:	5d                   	pop    %rbp
  400a07:	c3                   	retq   
  400a08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a0f:	00 

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EEdlEPv>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a18:	5d                   	pop    %rbp
  400a19:	c3                   	retq   
  400a1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 83 ec 30          	sub    $0x30,%rsp
  400a28:	48 89 f8             	mov    %rdi,%rax
  400a2b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a2f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a32:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a36:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a39:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a3d:	48 89 f7             	mov    %rsi,%rdi
  400a40:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a44:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a47:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a4b:	e8 80 01 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a50:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a53:	89 10                	mov    %edx,(%rax)
  400a55:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a59:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a5d:	e8 8e 01 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400a62:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a66:	48 83 c4 30          	add    $0x30,%rsp
  400a6a:	5d                   	pop    %rbp
  400a6b:	c3                   	retq   
  400a6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a70:	55                   	push   %rbp
  400a71:	48 89 e5             	mov    %rsp,%rbp
  400a74:	48 83 ec 30          	sub    $0x30,%rsp
  400a78:	48 89 f8             	mov    %rdi,%rax
  400a7b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a7f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a82:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a86:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a89:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a8d:	48 89 f7             	mov    %rsi,%rdi
  400a90:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a94:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a97:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a9b:	e8 b0 01 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aa0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400aa3:	89 10                	mov    %edx,(%rax)
  400aa5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400aa9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400aad:	e8 ce 01 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400ab2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ab6:	48 83 c4 30          	add    $0x30,%rsp
  400aba:	5d                   	pop    %rbp
  400abb:	c3                   	retq   
  400abc:	0f 1f 40 00          	nopl   0x0(%rax)

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
  400b0c:	e8 ef 00 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
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
  400b50:	e8 fb 00 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
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
  400bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bd8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bdc:	48 c1 e7 02          	shl    $0x2,%rdi
  400be0:	48 03 3c 25 60 20 60 	add    0x602060,%rdi
  400be7:	00 
  400be8:	48 89 f8             	mov    %rdi,%rax
  400beb:	5d                   	pop    %rbp
  400bec:	c3                   	retq   
  400bed:	0f 1f 00             	nopl   (%rax)

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bf8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400bfc:	5d                   	pop    %rbp
  400bfd:	c3                   	retq   
  400bfe:	66 90                	xchg   %ax,%ax

0000000000400c00 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 83 ec 30          	sub    $0x30,%rsp
  400c08:	48 89 f8             	mov    %rdi,%rax
  400c0b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c0f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c12:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c16:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c19:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c1d:	48 89 f7             	mov    %rsi,%rdi
  400c20:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c24:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c27:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c2b:	e8 20 00 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c33:	0f af 10             	imul   (%rax),%edx
  400c36:	89 10                	mov    %edx,(%rax)
  400c38:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c3c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c40:	e8 3b 00 00 00       	callq  400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400c45:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c49:	48 83 c4 30          	add    $0x30,%rsp
  400c4d:	5d                   	pop    %rbp
  400c4e:	c3                   	retq   
  400c4f:	90                   	nop

0000000000400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c50:	55                   	push   %rbp
  400c51:	48 89 e5             	mov    %rsp,%rbp
  400c54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c58:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c5c:	48 c1 e7 02          	shl    $0x2,%rdi
  400c60:	48 03 3c 25 60 20 60 	add    0x602060,%rdi
  400c67:	00 
  400c68:	48 81 c7 d0 48 00 00 	add    $0x48d0,%rdi
  400c6f:	48 89 f8             	mov    %rdi,%rax
  400c72:	5d                   	pop    %rbp
  400c73:	c3                   	retq   
  400c74:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7b:	00 00 00 
  400c7e:	66 90                	xchg   %ax,%ax

0000000000400c80 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c88:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c8c:	5d                   	pop    %rbp
  400c8d:	c3                   	retq   
  400c8e:	66 90                	xchg   %ax,%ax

0000000000400c90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c90:	55                   	push   %rbp
  400c91:	48 89 e5             	mov    %rsp,%rbp
  400c94:	5d                   	pop    %rbp
  400c95:	c3                   	retq   
  400c96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9d:	00 00 00 

0000000000400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE3getEm>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 83 ec 10          	sub    $0x10,%rsp
  400ca8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cb0:	e8 0b 00 00 00       	callq  400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400cb5:	48 83 c4 10          	add    $0x10,%rsp
  400cb9:	5d                   	pop    %rbp
  400cba:	c3                   	retq   
  400cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cc0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cc8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400ccc:	5d                   	pop    %rbp
  400ccd:	c3                   	retq   
  400cce:	66 90                	xchg   %ax,%ax

0000000000400cd0 <__libc_csu_init>:
  400cd0:	41 57                	push   %r15
  400cd2:	41 56                	push   %r14
  400cd4:	41 89 ff             	mov    %edi,%r15d
  400cd7:	41 55                	push   %r13
  400cd9:	41 54                	push   %r12
  400cdb:	4c 8d 25 f6 10 20 00 	lea    0x2010f6(%rip),%r12        # 601dd8 <__frame_dummy_init_array_entry>
  400ce2:	55                   	push   %rbp
  400ce3:	48 8d 2d fe 10 20 00 	lea    0x2010fe(%rip),%rbp        # 601de8 <__init_array_end>
  400cea:	53                   	push   %rbx
  400ceb:	49 89 f6             	mov    %rsi,%r14
  400cee:	49 89 d5             	mov    %rdx,%r13
  400cf1:	4c 29 e5             	sub    %r12,%rbp
  400cf4:	48 83 ec 08          	sub    $0x8,%rsp
  400cf8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cfc:	e8 a7 f8 ff ff       	callq  4005a8 <_init>
  400d01:	48 85 ed             	test   %rbp,%rbp
  400d04:	74 20                	je     400d26 <__libc_csu_init+0x56>
  400d06:	31 db                	xor    %ebx,%ebx
  400d08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d0f:	00 
  400d10:	4c 89 ea             	mov    %r13,%rdx
  400d13:	4c 89 f6             	mov    %r14,%rsi
  400d16:	44 89 ff             	mov    %r15d,%edi
  400d19:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400d1d:	48 83 c3 01          	add    $0x1,%rbx
  400d21:	48 39 eb             	cmp    %rbp,%rbx
  400d24:	75 ea                	jne    400d10 <__libc_csu_init+0x40>
  400d26:	48 83 c4 08          	add    $0x8,%rsp
  400d2a:	5b                   	pop    %rbx
  400d2b:	5d                   	pop    %rbp
  400d2c:	41 5c                	pop    %r12
  400d2e:	41 5d                	pop    %r13
  400d30:	41 5e                	pop    %r14
  400d32:	41 5f                	pop    %r15
  400d34:	c3                   	retq   
  400d35:	90                   	nop
  400d36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d3d:	00 00 00 

0000000000400d40 <__libc_csu_fini>:
  400d40:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400d44 <_fini>:
  400d44:	48 83 ec 08          	sub    $0x8,%rsp
  400d48:	48 83 c4 08          	add    $0x8,%rsp
  400d4c:	c3                   	retq   
