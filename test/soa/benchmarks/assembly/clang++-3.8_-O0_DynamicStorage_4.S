
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
  400740:	55                   	push   %rbp
  400741:	48 89 e5             	mov    %rsp,%rbp
  400744:	48 83 ec 50          	sub    $0x50,%rsp
  400748:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  40074f:	e8 ac 02 00 00       	callq  400a00 <_ZN9TestClass18initialize_storageEv>
  400754:	b8 04 00 00 00       	mov    $0x4,%eax
  400759:	89 c7                	mov    %eax,%edi
  40075b:	e8 c0 02 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400760:	48 89 c7             	mov    %rax,%rdi
  400763:	48 89 c1             	mov    %rax,%rcx
  400766:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40076c:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400770:	48 89 c7             	mov    %rax,%rdi
  400773:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400777:	e8 d4 02 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  40077c:	e9 00 00 00 00       	jmpq   400781 <main+0x41>
  400781:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400785:	ba 77 77 00 00       	mov    $0x7777,%edx
  40078a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40078e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400792:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400796:	48 83 c1 04          	add    $0x4,%rcx
  40079a:	48 89 ce             	mov    %rcx,%rsi
  40079d:	e8 fe 02 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4007a2:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  4007a6:	ba 88 88 00 00       	mov    $0x8888,%edx
  4007ab:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007af:	48 83 c0 04          	add    $0x4,%rax
  4007b3:	48 89 c6             	mov    %rax,%rsi
  4007b6:	e8 35 03 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007bb:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007bf:	e8 7c 03 00 00       	callq  400b40 <_ZN9TestClass15increase_field0Ev>
  4007c4:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007c8:	e8 b3 03 00 00       	callq  400b80 <_ZN9TestClass15increase_field1Ev>
  4007cd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007d1:	48 83 c0 04          	add    $0x4,%rax
  4007d5:	48 89 c7             	mov    %rax,%rdi
  4007d8:	e8 e3 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007dd:	8b 30                	mov    (%rax),%esi
  4007df:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007e3:	48 83 c0 04          	add    $0x4,%rax
  4007e7:	48 89 c7             	mov    %rax,%rdi
  4007ea:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  4007ed:	e8 ee 03 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007f2:	48 bf 04 0f 40 00 00 	movabs $0x400f04,%rdi
  4007f9:	00 00 00 
  4007fc:	8b 10                	mov    (%rax),%edx
  4007fe:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  400801:	b0 00                	mov    $0x0,%al
  400803:	e8 d8 fd ff ff       	callq  4005e0 <printf@plt>
  400808:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  40080c:	48 83 c1 04          	add    $0x4,%rcx
  400810:	48 89 cf             	mov    %rcx,%rdi
  400813:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400816:	e8 a5 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40081b:	31 d2                	xor    %edx,%edx
  40081d:	41 88 d0             	mov    %dl,%r8b
  400820:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400826:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  40082a:	0f 85 1c 00 00 00    	jne    40084c <main+0x10c>
  400830:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400834:	48 83 c0 04          	add    $0x4,%rax
  400838:	48 89 c7             	mov    %rax,%rdi
  40083b:	e8 a0 03 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400840:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400846:	0f 94 c1             	sete   %cl
  400849:	88 4d bf             	mov    %cl,-0x41(%rbp)
  40084c:	8a 45 bf             	mov    -0x41(%rbp),%al
  40084f:	34 ff                	xor    $0xff,%al
  400851:	24 01                	and    $0x1,%al
  400853:	0f b6 c0             	movzbl %al,%eax
  400856:	48 83 c4 50          	add    $0x50,%rsp
  40085a:	5d                   	pop    %rbp
  40085b:	c3                   	retq   
  40085c:	89 d1                	mov    %edx,%ecx
  40085e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400862:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400865:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400869:	e8 22 02 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  40086e:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400872:	e8 a9 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400877:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40087e:	00 00 

0000000000400880 <_Z12new_instancev>:
  400880:	55                   	push   %rbp
  400881:	48 89 e5             	mov    %rsp,%rbp
  400884:	48 83 ec 20          	sub    $0x20,%rsp
  400888:	b8 04 00 00 00       	mov    $0x4,%eax
  40088d:	89 c7                	mov    %eax,%edi
  40088f:	e8 8c 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400894:	48 89 c7             	mov    %rax,%rdi
  400897:	48 89 c1             	mov    %rax,%rcx
  40089a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008a0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008a4:	48 89 c7             	mov    %rax,%rdi
  4008a7:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4008ab:	e8 a0 01 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  4008b0:	e9 00 00 00 00       	jmpq   4008b5 <_Z12new_instancev+0x35>
  4008b5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008b9:	48 83 c4 20          	add    $0x20,%rsp
  4008bd:	5d                   	pop    %rbp
  4008be:	c3                   	retq   
  4008bf:	89 d1                	mov    %edx,%ecx
  4008c1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008c5:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  4008c8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008cc:	e8 bf 01 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008d1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008d5:	e8 46 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008e0 <_Z12write_field0P9TestClass>:
  4008e0:	55                   	push   %rbp
  4008e1:	48 89 e5             	mov    %rsp,%rbp
  4008e4:	48 83 ec 20          	sub    $0x20,%rsp
  4008e8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  4008ec:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008f1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008f5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008f9:	48 83 c7 04          	add    $0x4,%rdi
  4008fd:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400901:	48 89 c7             	mov    %rax,%rdi
  400904:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400908:	e8 93 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40090d:	48 83 c4 20          	add    $0x20,%rsp
  400911:	5d                   	pop    %rbp
  400912:	c3                   	retq   
  400913:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40091a:	84 00 00 00 00 00 

0000000000400920 <_Z11read_field0P9TestClass>:
  400920:	55                   	push   %rbp
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 83 ec 10          	sub    $0x10,%rsp
  400928:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40092c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400930:	48 83 c7 04          	add    $0x4,%rdi
  400934:	e8 87 02 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400939:	8b 00                	mov    (%rax),%eax
  40093b:	48 83 c4 10          	add    $0x10,%rsp
  40093f:	5d                   	pop    %rbp
  400940:	c3                   	retq   
  400941:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400948:	0f 1f 84 00 00 00 00 
  40094f:	00 

0000000000400950 <_Z12write_field1P9TestClass>:
  400950:	55                   	push   %rbp
  400951:	48 89 e5             	mov    %rsp,%rbp
  400954:	48 83 ec 20          	sub    $0x20,%rsp
  400958:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40095c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400961:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400965:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400969:	48 83 c7 04          	add    $0x4,%rdi
  40096d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400971:	48 89 c7             	mov    %rax,%rdi
  400974:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400978:	e8 73 01 00 00       	callq  400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40097d:	48 83 c4 20          	add    $0x20,%rsp
  400981:	5d                   	pop    %rbp
  400982:	c3                   	retq   
  400983:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40098a:	84 00 00 00 00 00 

0000000000400990 <_Z11read_field1P9TestClass>:
  400990:	55                   	push   %rbp
  400991:	48 89 e5             	mov    %rsp,%rbp
  400994:	48 83 ec 10          	sub    $0x10,%rsp
  400998:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40099c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009a0:	48 83 c7 04          	add    $0x4,%rdi
  4009a4:	e8 37 02 00 00       	callq  400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4009a9:	8b 00                	mov    (%rax),%eax
  4009ab:	48 83 c4 10          	add    $0x10,%rsp
  4009af:	5d                   	pop    %rbp
  4009b0:	c3                   	retq   
  4009b1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009b8:	0f 1f 84 00 00 00 00 
  4009bf:	00 

00000000004009c0 <_Z21explicit_write_field0m>:
  4009c0:	55                   	push   %rbp
  4009c1:	48 89 e5             	mov    %rsp,%rbp
  4009c4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009c8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009cc:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  4009d3:	00 
  4009d4:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  4009db:	5d                   	pop    %rbp
  4009dc:	c3                   	retq   
  4009dd:	0f 1f 00             	nopl   (%rax)

00000000004009e0 <_Z20explicit_read_field0m>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009ec:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  4009f3:	00 
  4009f4:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4009f7:	5d                   	pop    %rbp
  4009f8:	c3                   	retq   
  4009f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a00 <_ZN9TestClass18initialize_storageEv>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400a0b:	00 00 00 
  400a0e:	48 89 c7             	mov    %rax,%rdi
  400a11:	e8 ea 01 00 00       	callq  400c00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  400a16:	5d                   	pop    %rbp
  400a17:	c3                   	retq   
  400a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a1f:	00 

0000000000400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 83 ec 10          	sub    $0x10,%rsp
  400a28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a2c:	e8 ef 03 00 00       	callq  400e20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a31:	e8 0a 03 00 00       	callq  400d40 <_ZN9TestClass7storageEv>
  400a36:	48 8b 38             	mov    (%rax),%rdi
  400a39:	48 89 f9             	mov    %rdi,%rcx
  400a3c:	48 83 c1 01          	add    $0x1,%rcx
  400a40:	48 89 08             	mov    %rcx,(%rax)
  400a43:	e8 e8 03 00 00       	callq  400e30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a48:	48 83 c4 10          	add    $0x10,%rsp
  400a4c:	5d                   	pop    %rbp
  400a4d:	c3                   	retq   
  400a4e:	66 90                	xchg   %ax,%ax

0000000000400a50 <_ZN9TestClassC2Ev>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 83 ec 10          	sub    $0x10,%rsp
  400a58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a60:	48 89 f8             	mov    %rdi,%rax
  400a63:	48 83 c0 04          	add    $0x4,%rax
  400a67:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a6b:	48 89 c7             	mov    %rax,%rdi
  400a6e:	e8 dd 01 00 00       	callq  400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a73:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a77:	48 83 c0 04          	add    $0x4,%rax
  400a7b:	48 89 c7             	mov    %rax,%rdi
  400a7e:	e8 dd 01 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a83:	48 83 c4 10          	add    $0x10,%rsp
  400a87:	5d                   	pop    %rbp
  400a88:	c3                   	retq   
  400a89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a90:	55                   	push   %rbp
  400a91:	48 89 e5             	mov    %rsp,%rbp
  400a94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a98:	5d                   	pop    %rbp
  400a99:	c3                   	retq   
  400a9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 83 ec 30          	sub    $0x30,%rsp
  400aa8:	48 89 f8             	mov    %rdi,%rax
  400aab:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400aaf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400ab2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ab6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400ab9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400abd:	48 89 f7             	mov    %rsi,%rdi
  400ac0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400ac4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400ac7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400acb:	e8 f0 01 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ad3:	89 10                	mov    %edx,(%rax)
  400ad5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ad9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400add:	e8 fe 01 00 00       	callq  400ce0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400ae2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ae6:	48 83 c4 30          	add    $0x30,%rsp
  400aea:	5d                   	pop    %rbp
  400aeb:	c3                   	retq   
  400aec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400af0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 83 ec 30          	sub    $0x30,%rsp
  400af8:	48 89 f8             	mov    %rdi,%rax
  400afb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400aff:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400b02:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b06:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400b09:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b0d:	48 89 f7             	mov    %rsi,%rdi
  400b10:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b14:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400b17:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400b1b:	e8 80 02 00 00       	callq  400da0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b20:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400b23:	89 10                	mov    %edx,(%rax)
  400b25:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400b29:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400b2d:	e8 8e 02 00 00       	callq  400dc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400b32:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b36:	48 83 c4 30          	add    $0x30,%rsp
  400b3a:	5d                   	pop    %rbp
  400b3b:	c3                   	retq   
  400b3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400b40 <_ZN9TestClass15increase_field0Ev>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 20          	sub    $0x20,%rsp
  400b48:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b4c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b51:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b55:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b59:	48 83 c7 04          	add    $0x4,%rdi
  400b5d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b61:	48 89 c7             	mov    %rax,%rdi
  400b64:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b68:	e8 03 01 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b6d:	48 83 c4 20          	add    $0x20,%rsp
  400b71:	5d                   	pop    %rbp
  400b72:	c3                   	retq   
  400b73:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b7a:	00 00 00 
  400b7d:	0f 1f 00             	nopl   (%rax)

0000000000400b80 <_ZN9TestClass15increase_field1Ev>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 83 ec 20          	sub    $0x20,%rsp
  400b88:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b8c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b91:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b95:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b99:	48 83 c7 04          	add    $0x4,%rdi
  400b9d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400ba1:	48 89 c7             	mov    %rax,%rdi
  400ba4:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400ba8:	e8 a3 01 00 00       	callq  400d50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400bad:	48 83 c4 20          	add    $0x20,%rsp
  400bb1:	5d                   	pop    %rbp
  400bb2:	c3                   	retq   
  400bb3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bba:	00 00 00 
  400bbd:	0f 1f 00             	nopl   (%rax)

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 83 ec 10          	sub    $0x10,%rsp
  400bc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bcc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bd0:	e8 eb 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd5:	48 83 c4 10          	add    $0x10,%rsp
  400bd9:	5d                   	pop    %rbp
  400bda:	c3                   	retq   
  400bdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400be0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 83 ec 10          	sub    $0x10,%rsp
  400be8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bec:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bf0:	e8 ab 01 00 00       	callq  400da0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bf5:	48 83 c4 10          	add    $0x10,%rsp
  400bf9:	5d                   	pop    %rbp
  400bfa:	c3                   	retq   
  400bfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400c00:	55                   	push   %rbp
  400c01:	48 89 e5             	mov    %rsp,%rbp
  400c04:	48 83 ec 10          	sub    $0x10,%rsp
  400c08:	b8 a0 91 00 00       	mov    $0x91a0,%eax
  400c0d:	89 c1                	mov    %eax,%ecx
  400c0f:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c13:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c17:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c1b:	48 89 cf             	mov    %rcx,%rdi
  400c1e:	e8 cd f9 ff ff       	callq  4005f0 <malloc@plt>
  400c23:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400c27:	48 89 41 08          	mov    %rax,0x8(%rcx)
  400c2b:	48 c7 41 18 00 00 00 	movq   $0x0,0x18(%rcx)
  400c32:	00 
  400c33:	48 c7 41 10 00 00 00 	movq   $0x0,0x10(%rcx)
  400c3a:	00 
  400c3b:	48 83 c4 10          	add    $0x10,%rsp
  400c3f:	5d                   	pop    %rbp
  400c40:	c3                   	retq   
  400c41:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c48:	00 00 00 
  400c4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400c50:	55                   	push   %rbp
  400c51:	48 89 e5             	mov    %rsp,%rbp
  400c54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c58:	5d                   	pop    %rbp
  400c59:	c3                   	retq   
  400c5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	5d                   	pop    %rbp
  400c69:	c3                   	retq   
  400c6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400c70:	55                   	push   %rbp
  400c71:	48 89 e5             	mov    %rsp,%rbp
  400c74:	48 83 ec 30          	sub    $0x30,%rsp
  400c78:	48 89 f8             	mov    %rdi,%rax
  400c7b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400c7f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c82:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c86:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c89:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c8d:	48 89 f7             	mov    %rsi,%rdi
  400c90:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c94:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c97:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c9b:	e8 20 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ca3:	0f af 10             	imul   (%rax),%edx
  400ca6:	89 10                	mov    %edx,(%rax)
  400ca8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cac:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cb0:	e8 2b 00 00 00       	callq  400ce0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400cb5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cb9:	48 83 c4 30          	add    $0x30,%rsp
  400cbd:	5d                   	pop    %rbp
  400cbe:	c3                   	retq   
  400cbf:	90                   	nop

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 83 ec 10          	sub    $0x10,%rsp
  400cc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ccc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cd0:	e8 1b 00 00 00       	callq  400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cd5:	48 83 c4 10          	add    $0x10,%rsp
  400cd9:	5d                   	pop    %rbp
  400cda:	c3                   	retq   
  400cdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ce0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400ce0:	55                   	push   %rbp
  400ce1:	48 89 e5             	mov    %rsp,%rbp
  400ce4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ce8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400cec:	5d                   	pop    %rbp
  400ced:	c3                   	retq   
  400cee:	66 90                	xchg   %ax,%ax

0000000000400cf0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cf0:	55                   	push   %rbp
  400cf1:	48 89 e5             	mov    %rsp,%rbp
  400cf4:	48 83 ec 20          	sub    $0x20,%rsp
  400cf8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cfc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d00:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d04:	e8 37 00 00 00       	callq  400d40 <_ZN9TestClass7storageEv>
  400d09:	48 8b 40 08          	mov    0x8(%rax),%rax
  400d0d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d11:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d15:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400d19:	48 83 e8 04          	sub    $0x4,%rax
  400d1d:	48 c1 e8 02          	shr    $0x2,%rax
  400d21:	48 c1 e0 02          	shl    $0x2,%rax
  400d25:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400d29:	48 83 c0 00          	add    $0x0,%rax
  400d2d:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d31:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d35:	48 83 c4 20          	add    $0x20,%rsp
  400d39:	5d                   	pop    %rbp
  400d3a:	c3                   	retq   
  400d3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d40 <_ZN9TestClass7storageEv>:
  400d40:	55                   	push   %rbp
  400d41:	48 89 e5             	mov    %rsp,%rbp
  400d44:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400d4b:	00 00 00 
  400d4e:	5d                   	pop    %rbp
  400d4f:	c3                   	retq   

0000000000400d50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	48 83 ec 30          	sub    $0x30,%rsp
  400d58:	48 89 f8             	mov    %rdi,%rax
  400d5b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400d5f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400d62:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400d66:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400d69:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400d6d:	48 89 f7             	mov    %rsi,%rdi
  400d70:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d74:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400d77:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400d7b:	e8 20 00 00 00       	callq  400da0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d80:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400d83:	0f af 10             	imul   (%rax),%edx
  400d86:	89 10                	mov    %edx,(%rax)
  400d88:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400d8c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400d90:	e8 2b 00 00 00       	callq  400dc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400d95:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d99:	48 83 c4 30          	add    $0x30,%rsp
  400d9d:	5d                   	pop    %rbp
  400d9e:	c3                   	retq   
  400d9f:	90                   	nop

0000000000400da0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400da0:	55                   	push   %rbp
  400da1:	48 89 e5             	mov    %rsp,%rbp
  400da4:	48 83 ec 10          	sub    $0x10,%rsp
  400da8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400db0:	e8 1b 00 00 00       	callq  400dd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400db5:	48 83 c4 10          	add    $0x10,%rsp
  400db9:	5d                   	pop    %rbp
  400dba:	c3                   	retq   
  400dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dc0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400dc0:	55                   	push   %rbp
  400dc1:	48 89 e5             	mov    %rsp,%rbp
  400dc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400dc8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400dcc:	5d                   	pop    %rbp
  400dcd:	c3                   	retq   
  400dce:	66 90                	xchg   %ax,%ax

0000000000400dd0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400dd0:	55                   	push   %rbp
  400dd1:	48 89 e5             	mov    %rsp,%rbp
  400dd4:	48 83 ec 20          	sub    $0x20,%rsp
  400dd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ddc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400de0:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400de4:	e8 57 ff ff ff       	callq  400d40 <_ZN9TestClass7storageEv>
  400de9:	48 8b 40 08          	mov    0x8(%rax),%rax
  400ded:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400df1:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400df5:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400df9:	48 83 e8 04          	sub    $0x4,%rax
  400dfd:	48 c1 e8 02          	shr    $0x2,%rax
  400e01:	48 c1 e0 02          	shl    $0x2,%rax
  400e05:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400e09:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400e0f:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400e13:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400e17:	48 83 c4 20          	add    $0x20,%rsp
  400e1b:	5d                   	pop    %rbp
  400e1c:	c3                   	retq   
  400e1d:	0f 1f 00             	nopl   (%rax)

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
  400e5c:	e8 df fe ff ff       	callq  400d40 <_ZN9TestClass7storageEv>
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
