
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
  40064f:	49 c7 c0 70 0f 40 00 	mov    $0x400f70,%r8
  400656:	48 c7 c1 00 0f 40 00 	mov    $0x400f00,%rcx
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
  40075f:	e8 8c 02 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400764:	b8 04 00 00 00       	mov    $0x4,%eax
  400769:	89 c7                	mov    %eax,%edi
  40076b:	e8 a0 02 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  400770:	48 89 c7             	mov    %rax,%rdi
  400773:	48 89 c1             	mov    %rax,%rcx
  400776:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40077c:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
  400787:	e8 c4 02 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  40078c:	e9 00 00 00 00       	jmpq   400791 <main+0x41>
  400791:	be 77 77 00 00       	mov    $0x7777,%esi
  400796:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
  40079a:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40079e:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007a2:	48 83 c1 04          	add    $0x4,%rcx
  4007a6:	48 89 cf             	mov    %rcx,%rdi
  4007a9:	e8 f2 02 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4007ae:	be 88 88 00 00       	mov    $0x8888,%esi
  4007b3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007b7:	48 83 c1 04          	add    $0x4,%rcx
  4007bb:	48 89 cf             	mov    %rcx,%rdi
  4007be:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  4007c2:	e8 19 03 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  4007c7:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007cb:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
  4007cf:	e8 4c 03 00 00       	callq  400b20 <_ZN9TestClass15increase_field0Ev>
  4007d4:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007d8:	e8 83 03 00 00       	callq  400b60 <_ZN9TestClass15increase_field1Ev>
  4007dd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007e1:	48 83 c0 04          	add    $0x4,%rax
  4007e5:	48 89 c7             	mov    %rax,%rdi
  4007e8:	e8 b3 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007ed:	8b 30                	mov    (%rax),%esi
  4007ef:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007f3:	48 83 c0 04          	add    $0x4,%rax
  4007f7:	48 89 c7             	mov    %rax,%rdi
  4007fa:	89 75 bc             	mov    %esi,-0x44(%rbp)
  4007fd:	e8 be 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400802:	48 bf 84 0f 40 00 00 	movabs $0x400f84,%rdi
  400809:	00 00 00 
  40080c:	8b 10                	mov    (%rax),%edx
  40080e:	8b 75 bc             	mov    -0x44(%rbp),%esi
  400811:	b0 00                	mov    $0x0,%al
  400813:	e8 c8 fd ff ff       	callq  4005e0 <printf@plt>
  400818:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  40081c:	48 83 c1 04          	add    $0x4,%rcx
  400820:	48 89 cf             	mov    %rcx,%rdi
  400823:	89 45 b8             	mov    %eax,-0x48(%rbp)
  400826:	e8 75 03 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40082b:	31 d2                	xor    %edx,%edx
  40082d:	41 88 d0             	mov    %dl,%r8b
  400830:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400836:	44 88 45 b7          	mov    %r8b,-0x49(%rbp)
  40083a:	0f 85 1c 00 00 00    	jne    40085c <main+0x10c>
  400840:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400844:	48 83 c0 04          	add    $0x4,%rax
  400848:	48 89 c7             	mov    %rax,%rdi
  40084b:	e8 70 03 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400850:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400856:	0f 94 c1             	sete   %cl
  400859:	88 4d b7             	mov    %cl,-0x49(%rbp)
  40085c:	8a 45 b7             	mov    -0x49(%rbp),%al
  40085f:	34 ff                	xor    $0xff,%al
  400861:	24 01                	and    $0x1,%al
  400863:	0f b6 c0             	movzbl %al,%eax
  400866:	48 83 c4 50          	add    $0x50,%rsp
  40086a:	5d                   	pop    %rbp
  40086b:	c3                   	retq   
  40086c:	89 d1                	mov    %edx,%ecx
  40086e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400872:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400875:	48 8b 7d d8          	mov    -0x28(%rbp),%rdi
  400879:	e8 12 02 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
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
  40089f:	e8 6c 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4008a4:	48 89 c7             	mov    %rax,%rdi
  4008a7:	48 89 c1             	mov    %rax,%rcx
  4008aa:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008b0:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008b4:	48 89 c7             	mov    %rax,%rdi
  4008b7:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4008bb:	e8 90 01 00 00       	callq  400a50 <_ZN9TestClassC2Ev>
  4008c0:	e9 00 00 00 00       	jmpq   4008c5 <_Z12new_instancev+0x35>
  4008c5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008c9:	48 83 c4 20          	add    $0x20,%rsp
  4008cd:	5d                   	pop    %rbp
  4008ce:	c3                   	retq   
  4008cf:	89 d1                	mov    %edx,%ecx
  4008d1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4008d5:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  4008d8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008dc:	e8 af 01 00 00       	callq  400a90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008e1:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e5:	e8 36 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004008f0 <_Z12write_field0P9TestClass>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	be 77 77 00 00       	mov    $0x7777,%esi
  4008fd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400901:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400905:	48 83 c7 04          	add    $0x4,%rdi
  400909:	e8 92 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40090e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400912:	48 83 c4 10          	add    $0x10,%rsp
  400916:	5d                   	pop    %rbp
  400917:	c3                   	retq   
  400918:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40091f:	00 

0000000000400920 <_Z11read_field0P9TestClass>:
  400920:	55                   	push   %rbp
  400921:	48 89 e5             	mov    %rsp,%rbp
  400924:	48 83 ec 10          	sub    $0x10,%rsp
  400928:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40092c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400930:	48 83 c7 04          	add    $0x4,%rdi
  400934:	e8 67 02 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
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
  400954:	48 83 ec 10          	sub    $0x10,%rsp
  400958:	be 77 77 00 00       	mov    $0x7777,%esi
  40095d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400961:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400965:	48 83 c7 04          	add    $0x4,%rdi
  400969:	e8 72 01 00 00       	callq  400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40096e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400972:	48 83 c4 10          	add    $0x10,%rsp
  400976:	5d                   	pop    %rbp
  400977:	c3                   	retq   
  400978:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40097f:	00 

0000000000400980 <_Z11read_field1P9TestClass>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 83 ec 10          	sub    $0x10,%rsp
  400988:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40098c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400990:	48 83 c7 04          	add    $0x4,%rdi
  400994:	e8 27 02 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400999:	8b 00                	mov    (%rax),%eax
  40099b:	48 83 c4 10          	add    $0x10,%rsp
  40099f:	5d                   	pop    %rbp
  4009a0:	c3                   	retq   
  4009a1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009a8:	0f 1f 84 00 00 00 00 
  4009af:	00 

00000000004009b0 <_Z21explicit_write_field0m>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009b8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009bc:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  4009c3:	77 77 00 00 
  4009c7:	5d                   	pop    %rbp
  4009c8:	c3                   	retq   
  4009c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009d0 <_Z20explicit_read_field0m>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009d8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009dc:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  4009e3:	5d                   	pop    %rbp
  4009e4:	c3                   	retq   
  4009e5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ec:	00 00 00 
  4009ef:	90                   	nop

00000000004009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	e8 47 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009f9:	48 89 c7             	mov    %rax,%rdi
  4009fc:	e8 cf 03 00 00       	callq  400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  400a01:	5d                   	pop    %rbp
  400a02:	c3                   	retq   
  400a03:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a0a:	00 00 00 
  400a0d:	0f 1f 00             	nopl   (%rax)

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400a10:	55                   	push   %rbp
  400a11:	48 89 e5             	mov    %rsp,%rbp
  400a14:	48 83 ec 10          	sub    $0x10,%rsp
  400a18:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a1c:	e8 3f 04 00 00       	callq  400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400a21:	e8 1a fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400a26:	b9 01 00 00 00       	mov    $0x1,%ecx
  400a2b:	89 ce                	mov    %ecx,%esi
  400a2d:	48 89 c7             	mov    %rax,%rdi
  400a30:	e8 5b 04 00 00       	callq  400e90 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400a35:	48 89 c7             	mov    %rax,%rdi
  400a38:	e8 33 04 00 00       	callq  400e70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a3d:	48 83 c4 10          	add    $0x10,%rsp
  400a41:	5d                   	pop    %rbp
  400a42:	c3                   	retq   
  400a43:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a4a:	00 00 00 
  400a4d:	0f 1f 00             	nopl   (%rax)

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
  400a6e:	e8 6d 01 00 00       	callq  400be0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a73:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a77:	48 83 c0 04          	add    $0x4,%rax
  400a7b:	48 89 c7             	mov    %rax,%rdi
  400a7e:	e8 6d 01 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
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
  400aa4:	48 83 ec 20          	sub    $0x20,%rsp
  400aa8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aac:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400aaf:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ab3:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400ab6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400aba:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400abd:	e8 8e 01 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac2:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400ac5:	89 30                	mov    %esi,(%rax)
  400ac7:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400acb:	48 83 c4 20          	add    $0x20,%rsp
  400acf:	5d                   	pop    %rbp
  400ad0:	c3                   	retq   
  400ad1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ad8:	00 00 00 
  400adb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ae0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	48 83 ec 20          	sub    $0x20,%rsp
  400ae8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aec:	89 75 f4             	mov    %esi,-0xc(%rbp)
  400aef:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400af3:	8b 75 f4             	mov    -0xc(%rbp),%esi
  400af6:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400afa:	89 75 e4             	mov    %esi,-0x1c(%rbp)
  400afd:	e8 3e 02 00 00       	callq  400d40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b02:	8b 75 e4             	mov    -0x1c(%rbp),%esi
  400b05:	89 30                	mov    %esi,(%rax)
  400b07:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400b0b:	48 83 c4 20          	add    $0x20,%rsp
  400b0f:	5d                   	pop    %rbp
  400b10:	c3                   	retq   
  400b11:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b18:	00 00 00 
  400b1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b20 <_ZN9TestClass15increase_field0Ev>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 83 ec 20          	sub    $0x20,%rsp
  400b28:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b2c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b31:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b35:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b39:	48 83 c7 04          	add    $0x4,%rdi
  400b3d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b41:	48 89 c7             	mov    %rax,%rdi
  400b44:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b48:	e8 b3 00 00 00       	callq  400c00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b4d:	48 83 c4 20          	add    $0x20,%rsp
  400b51:	5d                   	pop    %rbp
  400b52:	c3                   	retq   
  400b53:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b5a:	00 00 00 
  400b5d:	0f 1f 00             	nopl   (%rax)

0000000000400b60 <_ZN9TestClass15increase_field1Ev>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 83 ec 20          	sub    $0x20,%rsp
  400b68:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b6c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b71:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b75:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b79:	48 83 c7 04          	add    $0x4,%rdi
  400b7d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b81:	48 89 c7             	mov    %rax,%rdi
  400b84:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b88:	e8 63 01 00 00       	callq  400cf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b8d:	48 83 c4 20          	add    $0x20,%rsp
  400b91:	5d                   	pop    %rbp
  400b92:	c3                   	retq   
  400b93:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9a:	00 00 00 
  400b9d:	0f 1f 00             	nopl   (%rax)

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 83 ec 10          	sub    $0x10,%rsp
  400ba8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bb0:	e8 9b 00 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bb5:	48 83 c4 10          	add    $0x10,%rsp
  400bb9:	5d                   	pop    %rbp
  400bba:	c3                   	retq   
  400bbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 83 ec 10          	sub    $0x10,%rsp
  400bc8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bcc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bd0:	e8 6b 01 00 00       	callq  400d40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bd5:	48 83 c4 10          	add    $0x10,%rsp
  400bd9:	5d                   	pop    %rbp
  400bda:	c3                   	retq   
  400bdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400be0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400be8:	5d                   	pop    %rbp
  400be9:	c3                   	retq   
  400bea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bf8:	5d                   	pop    %rbp
  400bf9:	c3                   	retq   
  400bfa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400c00 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
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
  400c2b:	e8 20 00 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c33:	0f af 10             	imul   (%rax),%edx
  400c36:	89 10                	mov    %edx,(%rax)
  400c38:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c3c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c40:	e8 2b 00 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c45:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c49:	48 83 c4 30          	add    $0x30,%rsp
  400c4d:	5d                   	pop    %rbp
  400c4e:	c3                   	retq   
  400c4f:	90                   	nop

0000000000400c50 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c50:	55                   	push   %rbp
  400c51:	48 89 e5             	mov    %rsp,%rbp
  400c54:	48 83 ec 10          	sub    $0x10,%rsp
  400c58:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c5c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c60:	e8 1b 00 00 00       	callq  400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c65:	48 83 c4 10          	add    $0x10,%rsp
  400c69:	5d                   	pop    %rbp
  400c6a:	c3                   	retq   
  400c6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400c70:	55                   	push   %rbp
  400c71:	48 89 e5             	mov    %rsp,%rbp
  400c74:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c78:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c7c:	5d                   	pop    %rbp
  400c7d:	c3                   	retq   
  400c7e:	66 90                	xchg   %ax,%ax

0000000000400c80 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 83 ec 20          	sub    $0x20,%rsp
  400c88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c90:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c94:	e8 a7 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400c99:	48 89 c7             	mov    %rax,%rdi
  400c9c:	e8 2f 00 00 00       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400ca1:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400ca5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ca9:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400cad:	48 83 e8 04          	sub    $0x4,%rax
  400cb1:	48 c1 e8 02          	shr    $0x2,%rax
  400cb5:	48 c1 e0 02          	shl    $0x2,%rax
  400cb9:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400cbd:	48 83 c0 00          	add    $0x0,%rax
  400cc1:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400cc5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cc9:	48 83 c4 20          	add    $0x20,%rsp
  400ccd:	5d                   	pop    %rbp
  400cce:	c3                   	retq   
  400ccf:	90                   	nop

0000000000400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400cd0:	55                   	push   %rbp
  400cd1:	48 89 e5             	mov    %rsp,%rbp
  400cd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cd8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cdc:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400ce0:	5d                   	pop    %rbp
  400ce1:	c3                   	retq   
  400ce2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ce9:	00 00 00 
  400cec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400cf0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400cf0:	55                   	push   %rbp
  400cf1:	48 89 e5             	mov    %rsp,%rbp
  400cf4:	48 83 ec 30          	sub    $0x30,%rsp
  400cf8:	48 89 f8             	mov    %rdi,%rax
  400cfb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400cff:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400d02:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400d06:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400d09:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400d0d:	48 89 f7             	mov    %rsi,%rdi
  400d10:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d14:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400d17:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400d1b:	e8 20 00 00 00       	callq  400d40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d20:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400d23:	0f af 10             	imul   (%rax),%edx
  400d26:	89 10                	mov    %edx,(%rax)
  400d28:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400d2c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400d30:	e8 2b 00 00 00       	callq  400d60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400d35:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d39:	48 83 c4 30          	add    $0x30,%rsp
  400d3d:	5d                   	pop    %rbp
  400d3e:	c3                   	retq   
  400d3f:	90                   	nop

0000000000400d40 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d40:	55                   	push   %rbp
  400d41:	48 89 e5             	mov    %rsp,%rbp
  400d44:	48 83 ec 10          	sub    $0x10,%rsp
  400d48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d50:	e8 1b 00 00 00       	callq  400d70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d55:	48 83 c4 10          	add    $0x10,%rsp
  400d59:	5d                   	pop    %rbp
  400d5a:	c3                   	retq   
  400d5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d60:	55                   	push   %rbp
  400d61:	48 89 e5             	mov    %rsp,%rbp
  400d64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d68:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d6c:	5d                   	pop    %rbp
  400d6d:	c3                   	retq   
  400d6e:	66 90                	xchg   %ax,%ax

0000000000400d70 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d70:	55                   	push   %rbp
  400d71:	48 89 e5             	mov    %rsp,%rbp
  400d74:	48 83 ec 20          	sub    $0x20,%rsp
  400d78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d80:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d84:	e8 b7 f9 ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400d89:	48 89 c7             	mov    %rax,%rdi
  400d8c:	e8 3f ff ff ff       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d91:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d95:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d99:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400d9d:	48 83 e8 04          	sub    $0x4,%rax
  400da1:	48 c1 e8 02          	shr    $0x2,%rax
  400da5:	48 c1 e0 02          	shl    $0x2,%rax
  400da9:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400dad:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400db3:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400db7:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400dbb:	48 83 c4 20          	add    $0x20,%rsp
  400dbf:	5d                   	pop    %rbp
  400dc0:	c3                   	retq   
  400dc1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dc8:	00 00 00 
  400dcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400dd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400dd0:	55                   	push   %rbp
  400dd1:	48 89 e5             	mov    %rsp,%rbp
  400dd4:	48 83 ec 10          	sub    $0x10,%rsp
  400dd8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ddc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400de0:	48 89 f8             	mov    %rdi,%rax
  400de3:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400de7:	48 89 c7             	mov    %rax,%rdi
  400dea:	e8 31 00 00 00       	callq  400e20 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400def:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400df4:	89 cf                	mov    %ecx,%edi
  400df6:	e8 f5 f7 ff ff       	callq  4005f0 <malloc@plt>
  400dfb:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400dff:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400e03:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400e0a:	00 
  400e0b:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400e12:	00 
  400e13:	48 83 c4 10          	add    $0x10,%rsp
  400e17:	5d                   	pop    %rbp
  400e18:	c3                   	retq   
  400e19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400e20 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400e20:	55                   	push   %rbp
  400e21:	48 89 e5             	mov    %rsp,%rbp
  400e24:	48 83 ec 10          	sub    $0x10,%rsp
  400e28:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e2c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e30:	e8 0b 00 00 00       	callq  400e40 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400e35:	48 83 c4 10          	add    $0x10,%rsp
  400e39:	5d                   	pop    %rbp
  400e3a:	c3                   	retq   
  400e3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e40 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400e40:	55                   	push   %rbp
  400e41:	48 89 e5             	mov    %rsp,%rbp
  400e44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e48:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e4c:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400e53:	5d                   	pop    %rbp
  400e54:	c3                   	retq   
  400e55:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e5c:	00 00 00 
  400e5f:	90                   	nop

0000000000400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e60:	55                   	push   %rbp
  400e61:	48 89 e5             	mov    %rsp,%rbp
  400e64:	5d                   	pop    %rbp
  400e65:	c3                   	retq   
  400e66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e6d:	00 00 00 

0000000000400e70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400e70:	55                   	push   %rbp
  400e71:	48 89 e5             	mov    %rsp,%rbp
  400e74:	48 83 ec 10          	sub    $0x10,%rsp
  400e78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400e80:	48 83 c7 01          	add    $0x1,%rdi
  400e84:	e8 37 00 00 00       	callq  400ec0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e89:	48 83 c4 10          	add    $0x10,%rsp
  400e8d:	5d                   	pop    %rbp
  400e8e:	c3                   	retq   
  400e8f:	90                   	nop

0000000000400e90 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e90:	55                   	push   %rbp
  400e91:	48 89 e5             	mov    %rsp,%rbp
  400e94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400e98:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400e9c:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400ea0:	48 8b 3e             	mov    (%rsi),%rdi
  400ea3:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400ea7:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400eab:	48 03 3e             	add    (%rsi),%rdi
  400eae:	48 89 3e             	mov    %rdi,(%rsi)
  400eb1:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400eb5:	5d                   	pop    %rbp
  400eb6:	c3                   	retq   
  400eb7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400ebe:	00 00 

0000000000400ec0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400ec0:	55                   	push   %rbp
  400ec1:	48 89 e5             	mov    %rsp,%rbp
  400ec4:	48 83 ec 10          	sub    $0x10,%rsp
  400ec8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ecc:	e8 6f f8 ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400ed1:	48 89 c7             	mov    %rax,%rdi
  400ed4:	e8 f7 fd ff ff       	callq  400cd0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400ed9:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400edd:	48 c1 e7 02          	shl    $0x2,%rdi
  400ee1:	48 01 f8             	add    %rdi,%rax
  400ee4:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400ee8:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400eec:	48 83 c4 10          	add    $0x10,%rsp
  400ef0:	5d                   	pop    %rbp
  400ef1:	c3                   	retq   
  400ef2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ef9:	00 00 00 
  400efc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400f00 <__libc_csu_init>:
  400f00:	41 57                	push   %r15
  400f02:	41 56                	push   %r14
  400f04:	41 89 ff             	mov    %edi,%r15d
  400f07:	41 55                	push   %r13
  400f09:	41 54                	push   %r12
  400f0b:	4c 8d 25 ce 0e 20 00 	lea    0x200ece(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400f12:	55                   	push   %rbp
  400f13:	48 8d 2d ce 0e 20 00 	lea    0x200ece(%rip),%rbp        # 601de8 <__init_array_end>
  400f1a:	53                   	push   %rbx
  400f1b:	49 89 f6             	mov    %rsi,%r14
  400f1e:	49 89 d5             	mov    %rdx,%r13
  400f21:	4c 29 e5             	sub    %r12,%rbp
  400f24:	48 83 ec 08          	sub    $0x8,%rsp
  400f28:	48 c1 fd 03          	sar    $0x3,%rbp
  400f2c:	e8 77 f6 ff ff       	callq  4005a8 <_init>
  400f31:	48 85 ed             	test   %rbp,%rbp
  400f34:	74 20                	je     400f56 <__libc_csu_init+0x56>
  400f36:	31 db                	xor    %ebx,%ebx
  400f38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f3f:	00 
  400f40:	4c 89 ea             	mov    %r13,%rdx
  400f43:	4c 89 f6             	mov    %r14,%rsi
  400f46:	44 89 ff             	mov    %r15d,%edi
  400f49:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400f4d:	48 83 c3 01          	add    $0x1,%rbx
  400f51:	48 39 eb             	cmp    %rbp,%rbx
  400f54:	75 ea                	jne    400f40 <__libc_csu_init+0x40>
  400f56:	48 83 c4 08          	add    $0x8,%rsp
  400f5a:	5b                   	pop    %rbx
  400f5b:	5d                   	pop    %rbp
  400f5c:	41 5c                	pop    %r12
  400f5e:	41 5d                	pop    %r13
  400f60:	41 5e                	pop    %r14
  400f62:	41 5f                	pop    %r15
  400f64:	c3                   	retq   
  400f65:	90                   	nop
  400f66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f6d:	00 00 00 

0000000000400f70 <__libc_csu_fini>:
  400f70:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f74 <_fini>:
  400f74:	48 83 ec 08          	sub    $0x8,%rsp
  400f78:	48 83 c4 08          	add    $0x8,%rsp
  400f7c:	c3                   	retq   
