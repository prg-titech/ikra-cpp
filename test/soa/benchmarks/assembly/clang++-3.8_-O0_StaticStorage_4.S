
bin/clang++-3.8_-O0_StaticStorage_4:     file format elf64-x86-64


Disassembly of section .init:

0000000000400570 <_init>:
  400570:	48 83 ec 08          	sub    $0x8,%rsp
  400574:	48 8b 05 7d 1a 20 00 	mov    0x201a7d(%rip),%rax        # 601ff8 <_DYNAMIC+0x200>
  40057b:	48 85 c0             	test   %rax,%rax
  40057e:	74 05                	je     400585 <_init+0x15>
  400580:	e8 5b 00 00 00       	callq  4005e0 <_Unwind_Resume@plt+0x10>
  400585:	48 83 c4 08          	add    $0x8,%rsp
  400589:	c3                   	retq   

Disassembly of section .plt:

0000000000400590 <printf@plt-0x10>:
  400590:	ff 35 72 1a 20 00    	pushq  0x201a72(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400596:	ff 25 74 1a 20 00    	jmpq   *0x201a74(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40059c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004005a0 <printf@plt>:
  4005a0:	ff 25 72 1a 20 00    	jmpq   *0x201a72(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  4005a6:	68 00 00 00 00       	pushq  $0x0
  4005ab:	e9 e0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005b0 <__libc_start_main@plt>:
  4005b0:	ff 25 6a 1a 20 00    	jmpq   *0x201a6a(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  4005b6:	68 01 00 00 00       	pushq  $0x1
  4005bb:	e9 d0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005c0 <__gxx_personality_v0@plt>:
  4005c0:	ff 25 62 1a 20 00    	jmpq   *0x201a62(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  4005c6:	68 02 00 00 00       	pushq  $0x2
  4005cb:	e9 c0 ff ff ff       	jmpq   400590 <_init+0x20>

00000000004005d0 <_Unwind_Resume@plt>:
  4005d0:	ff 25 5a 1a 20 00    	jmpq   *0x201a5a(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  4005d6:	68 03 00 00 00       	pushq  $0x3
  4005db:	e9 b0 ff ff ff       	jmpq   400590 <_init+0x20>

Disassembly of section .plt.got:

00000000004005e0 <.plt.got>:
  4005e0:	ff 25 12 1a 20 00    	jmpq   *0x201a12(%rip)        # 601ff8 <_DYNAMIC+0x200>
  4005e6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004005f0 <_start>:
  4005f0:	31 ed                	xor    %ebp,%ebp
  4005f2:	49 89 d1             	mov    %rdx,%r9
  4005f5:	5e                   	pop    %rsi
  4005f6:	48 89 e2             	mov    %rsp,%rdx
  4005f9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4005fd:	50                   	push   %rax
  4005fe:	54                   	push   %rsp
  4005ff:	49 c7 c0 30 0e 40 00 	mov    $0x400e30,%r8
  400606:	48 c7 c1 c0 0d 40 00 	mov    $0x400dc0,%rcx
  40060d:	48 c7 c7 f0 06 40 00 	mov    $0x4006f0,%rdi
  400614:	e8 97 ff ff ff       	callq  4005b0 <__libc_start_main@plt>
  400619:	f4                   	hlt    
  40061a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400620 <deregister_tm_clones>:
  400620:	b8 4f 20 60 00       	mov    $0x60204f,%eax
  400625:	55                   	push   %rbp
  400626:	48 2d 48 20 60 00    	sub    $0x602048,%rax
  40062c:	48 83 f8 0e          	cmp    $0xe,%rax
  400630:	48 89 e5             	mov    %rsp,%rbp
  400633:	76 1b                	jbe    400650 <deregister_tm_clones+0x30>
  400635:	b8 00 00 00 00       	mov    $0x0,%eax
  40063a:	48 85 c0             	test   %rax,%rax
  40063d:	74 11                	je     400650 <deregister_tm_clones+0x30>
  40063f:	5d                   	pop    %rbp
  400640:	bf 48 20 60 00       	mov    $0x602048,%edi
  400645:	ff e0                	jmpq   *%rax
  400647:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40064e:	00 00 
  400650:	5d                   	pop    %rbp
  400651:	c3                   	retq   
  400652:	0f 1f 40 00          	nopl   0x0(%rax)
  400656:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40065d:	00 00 00 

0000000000400660 <register_tm_clones>:
  400660:	be 48 20 60 00       	mov    $0x602048,%esi
  400665:	55                   	push   %rbp
  400666:	48 81 ee 48 20 60 00 	sub    $0x602048,%rsi
  40066d:	48 c1 fe 03          	sar    $0x3,%rsi
  400671:	48 89 e5             	mov    %rsp,%rbp
  400674:	48 89 f0             	mov    %rsi,%rax
  400677:	48 c1 e8 3f          	shr    $0x3f,%rax
  40067b:	48 01 c6             	add    %rax,%rsi
  40067e:	48 d1 fe             	sar    %rsi
  400681:	74 15                	je     400698 <register_tm_clones+0x38>
  400683:	b8 00 00 00 00       	mov    $0x0,%eax
  400688:	48 85 c0             	test   %rax,%rax
  40068b:	74 0b                	je     400698 <register_tm_clones+0x38>
  40068d:	5d                   	pop    %rbp
  40068e:	bf 48 20 60 00       	mov    $0x602048,%edi
  400693:	ff e0                	jmpq   *%rax
  400695:	0f 1f 00             	nopl   (%rax)
  400698:	5d                   	pop    %rbp
  400699:	c3                   	retq   
  40069a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006a0 <__do_global_dtors_aux>:
  4006a0:	80 3d a9 19 20 00 00 	cmpb   $0x0,0x2019a9(%rip)        # 602050 <completed.7585>
  4006a7:	75 11                	jne    4006ba <__do_global_dtors_aux+0x1a>
  4006a9:	55                   	push   %rbp
  4006aa:	48 89 e5             	mov    %rsp,%rbp
  4006ad:	e8 6e ff ff ff       	callq  400620 <deregister_tm_clones>
  4006b2:	5d                   	pop    %rbp
  4006b3:	c6 05 96 19 20 00 01 	movb   $0x1,0x201996(%rip)        # 602050 <completed.7585>
  4006ba:	f3 c3                	repz retq 
  4006bc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004006c0 <frame_dummy>:
  4006c0:	bf f0 1d 60 00       	mov    $0x601df0,%edi
  4006c5:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  4006c9:	75 05                	jne    4006d0 <frame_dummy+0x10>
  4006cb:	eb 93                	jmp    400660 <register_tm_clones>
  4006cd:	0f 1f 00             	nopl   (%rax)
  4006d0:	b8 00 00 00 00       	mov    $0x0,%eax
  4006d5:	48 85 c0             	test   %rax,%rax
  4006d8:	74 f1                	je     4006cb <frame_dummy+0xb>
  4006da:	55                   	push   %rbp
  4006db:	48 89 e5             	mov    %rsp,%rbp
  4006de:	ff d0                	callq  *%rax
  4006e0:	5d                   	pop    %rbp
  4006e1:	e9 7a ff ff ff       	jmpq   400660 <register_tm_clones>
  4006e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4006ed:	00 00 00 

00000000004006f0 <main>:
  4006f0:	55                   	push   %rbp
  4006f1:	48 89 e5             	mov    %rsp,%rbp
  4006f4:	48 83 ec 50          	sub    $0x50,%rsp
  4006f8:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  4006ff:	e8 ac 02 00 00       	callq  4009b0 <_ZN9TestClass18initialize_storageEv>
  400704:	b8 04 00 00 00       	mov    $0x4,%eax
  400709:	89 c7                	mov    %eax,%edi
  40070b:	e8 c0 02 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400710:	48 89 c7             	mov    %rax,%rdi
  400713:	48 89 c1             	mov    %rax,%rcx
  400716:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  40071c:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400720:	48 89 c7             	mov    %rax,%rdi
  400723:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400727:	e8 d4 02 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  40072c:	e9 00 00 00 00       	jmpq   400731 <main+0x41>
  400731:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400735:	ba 77 77 00 00       	mov    $0x7777,%edx
  40073a:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  40073e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400742:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400746:	48 83 c1 04          	add    $0x4,%rcx
  40074a:	48 89 ce             	mov    %rcx,%rsi
  40074d:	e8 fe 02 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  400752:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400756:	ba 88 88 00 00       	mov    $0x8888,%edx
  40075b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40075f:	48 83 c0 04          	add    $0x4,%rax
  400763:	48 89 c6             	mov    %rax,%rsi
  400766:	e8 35 03 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40076b:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40076f:	e8 7c 03 00 00       	callq  400af0 <_ZN9TestClass15increase_field0Ev>
  400774:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400778:	e8 b3 03 00 00       	callq  400b30 <_ZN9TestClass15increase_field1Ev>
  40077d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400781:	48 83 c0 04          	add    $0x4,%rax
  400785:	48 89 c7             	mov    %rax,%rdi
  400788:	e8 e3 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  40078d:	8b 30                	mov    (%rax),%esi
  40078f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400793:	48 83 c0 04          	add    $0x4,%rax
  400797:	48 89 c7             	mov    %rax,%rdi
  40079a:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  40079d:	e8 ee 03 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007a2:	48 bf 44 0e 40 00 00 	movabs $0x400e44,%rdi
  4007a9:	00 00 00 
  4007ac:	8b 10                	mov    (%rax),%edx
  4007ae:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007b1:	b0 00                	mov    $0x0,%al
  4007b3:	e8 e8 fd ff ff       	callq  4005a0 <printf@plt>
  4007b8:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007bc:	48 83 c1 04          	add    $0x4,%rcx
  4007c0:	48 89 cf             	mov    %rcx,%rdi
  4007c3:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007c6:	e8 a5 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007cb:	31 d2                	xor    %edx,%edx
  4007cd:	41 88 d0             	mov    %dl,%r8b
  4007d0:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007d6:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  4007da:	0f 85 1c 00 00 00    	jne    4007fc <main+0x10c>
  4007e0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007e4:	48 83 c0 04          	add    $0x4,%rax
  4007e8:	48 89 c7             	mov    %rax,%rdi
  4007eb:	e8 a0 03 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007f0:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007f6:	0f 94 c1             	sete   %cl
  4007f9:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007fc:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007ff:	34 ff                	xor    $0xff,%al
  400801:	24 01                	and    $0x1,%al
  400803:	0f b6 c0             	movzbl %al,%eax
  400806:	48 83 c4 50          	add    $0x50,%rsp
  40080a:	5d                   	pop    %rbp
  40080b:	c3                   	retq   
  40080c:	89 d1                	mov    %edx,%ecx
  40080e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400812:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400815:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400819:	e8 22 02 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  40081e:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400822:	e8 a9 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400827:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40082e:	00 00 

0000000000400830 <_Z12new_instancev>:
  400830:	55                   	push   %rbp
  400831:	48 89 e5             	mov    %rsp,%rbp
  400834:	48 83 ec 20          	sub    $0x20,%rsp
  400838:	b8 04 00 00 00       	mov    $0x4,%eax
  40083d:	89 c7                	mov    %eax,%edi
  40083f:	e8 8c 01 00 00       	callq  4009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400844:	48 89 c7             	mov    %rax,%rdi
  400847:	48 89 c1             	mov    %rax,%rcx
  40084a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400850:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40085b:	e8 a0 01 00 00       	callq  400a00 <_ZN9TestClassC2Ev>
  400860:	e9 00 00 00 00       	jmpq   400865 <_Z12new_instancev+0x35>
  400865:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400869:	48 83 c4 20          	add    $0x20,%rsp
  40086d:	5d                   	pop    %rbp
  40086e:	c3                   	retq   
  40086f:	89 d1                	mov    %edx,%ecx
  400871:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400875:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  400878:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40087c:	e8 bf 01 00 00       	callq  400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400881:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400885:	e8 46 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40088a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400890 <_Z12write_field0P9TestClass>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 83 ec 20          	sub    $0x20,%rsp
  400898:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40089c:	ba 77 77 00 00       	mov    $0x7777,%edx
  4008a1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008a5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008a9:	48 83 c7 04          	add    $0x4,%rdi
  4008ad:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008b1:	48 89 c7             	mov    %rax,%rdi
  4008b4:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  4008b8:	e8 93 01 00 00       	callq  400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  4008bd:	48 83 c4 20          	add    $0x20,%rsp
  4008c1:	5d                   	pop    %rbp
  4008c2:	c3                   	retq   
  4008c3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008ca:	84 00 00 00 00 00 

00000000004008d0 <_Z11read_field0P9TestClass>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 10          	sub    $0x10,%rsp
  4008d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008dc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e0:	48 83 c7 04          	add    $0x4,%rdi
  4008e4:	e8 87 02 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4008e9:	8b 00                	mov    (%rax),%eax
  4008eb:	48 83 c4 10          	add    $0x10,%rsp
  4008ef:	5d                   	pop    %rbp
  4008f0:	c3                   	retq   
  4008f1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008f8:	0f 1f 84 00 00 00 00 
  4008ff:	00 

0000000000400900 <_Z12write_field1P9TestClass>:
  400900:	55                   	push   %rbp
  400901:	48 89 e5             	mov    %rsp,%rbp
  400904:	48 83 ec 20          	sub    $0x20,%rsp
  400908:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40090c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400911:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400915:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400919:	48 83 c7 04          	add    $0x4,%rdi
  40091d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400921:	48 89 c7             	mov    %rax,%rdi
  400924:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400928:	e8 73 01 00 00       	callq  400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  40092d:	48 83 c4 20          	add    $0x20,%rsp
  400931:	5d                   	pop    %rbp
  400932:	c3                   	retq   
  400933:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40093a:	84 00 00 00 00 00 

0000000000400940 <_Z11read_field1P9TestClass>:
  400940:	55                   	push   %rbp
  400941:	48 89 e5             	mov    %rsp,%rbp
  400944:	48 83 ec 10          	sub    $0x10,%rsp
  400948:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40094c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400950:	48 83 c7 04          	add    $0x4,%rdi
  400954:	e8 37 02 00 00       	callq  400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  400959:	8b 00                	mov    (%rax),%eax
  40095b:	48 83 c4 10          	add    $0x10,%rsp
  40095f:	5d                   	pop    %rbp
  400960:	c3                   	retq   
  400961:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400968:	0f 1f 84 00 00 00 00 
  40096f:	00 

0000000000400970 <_Z21explicit_write_field0m>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400978:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40097c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  400983:	00 
  400984:	c7 04 b8 77 77 00 00 	movl   $0x7777,(%rax,%rdi,4)
  40098b:	5d                   	pop    %rbp
  40098c:	c3                   	retq   
  40098d:	0f 1f 00             	nopl   (%rax)

0000000000400990 <_Z20explicit_read_field0m>:
  400990:	55                   	push   %rbp
  400991:	48 89 e5             	mov    %rsp,%rbp
  400994:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400998:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40099c:	48 8b 04 25 00 a7 61 	mov    0x61a700,%rax
  4009a3:	00 
  4009a4:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4009a7:	5d                   	pop    %rbp
  4009a8:	c3                   	retq   
  4009a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009b0 <_ZN9TestClass18initialize_storageEv>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  4009bb:	00 00 00 
  4009be:	48 89 c7             	mov    %rax,%rdi
  4009c1:	e8 ea 01 00 00       	callq  400bb0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>
  4009c6:	5d                   	pop    %rbp
  4009c7:	c3                   	retq   
  4009c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009cf:	00 

00000000004009d0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009d0:	55                   	push   %rbp
  4009d1:	48 89 e5             	mov    %rsp,%rbp
  4009d4:	48 83 ec 10          	sub    $0x10,%rsp
  4009d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009dc:	e8 7f 03 00 00       	callq  400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009e1:	e8 ba 02 00 00       	callq  400ca0 <_ZN9TestClass7storageEv>
  4009e6:	48 8b 38             	mov    (%rax),%rdi
  4009e9:	48 89 f9             	mov    %rdi,%rcx
  4009ec:	48 83 c1 01          	add    $0x1,%rcx
  4009f0:	48 89 08             	mov    %rcx,(%rax)
  4009f3:	e8 78 03 00 00       	callq  400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009f8:	48 83 c4 10          	add    $0x10,%rsp
  4009fc:	5d                   	pop    %rbp
  4009fd:	c3                   	retq   
  4009fe:	66 90                	xchg   %ax,%ax

0000000000400a00 <_ZN9TestClassC2Ev>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 10          	sub    $0x10,%rsp
  400a08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a10:	48 89 f8             	mov    %rdi,%rax
  400a13:	48 83 c0 04          	add    $0x4,%rax
  400a17:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a1b:	48 89 c7             	mov    %rax,%rdi
  400a1e:	e8 ad 01 00 00       	callq  400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a23:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a27:	48 83 c0 04          	add    $0x4,%rax
  400a2b:	48 89 c7             	mov    %rax,%rdi
  400a2e:	e8 ad 01 00 00       	callq  400be0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a33:	48 83 c4 10          	add    $0x10,%rsp
  400a37:	5d                   	pop    %rbp
  400a38:	c3                   	retq   
  400a39:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a40 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a40:	55                   	push   %rbp
  400a41:	48 89 e5             	mov    %rsp,%rbp
  400a44:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a48:	5d                   	pop    %rbp
  400a49:	c3                   	retq   
  400a4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 83 ec 30          	sub    $0x30,%rsp
  400a58:	48 89 f8             	mov    %rdi,%rax
  400a5b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a5f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a62:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a66:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a69:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a6d:	48 89 f7             	mov    %rsi,%rdi
  400a70:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a74:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a77:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a7b:	e8 c0 01 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a80:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a83:	89 10                	mov    %edx,(%rax)
  400a85:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a89:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a8d:	e8 fe 01 00 00       	callq  400c90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400a92:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a96:	48 83 c4 30          	add    $0x30,%rsp
  400a9a:	5d                   	pop    %rbp
  400a9b:	c3                   	retq   
  400a9c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400aa0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
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
  400acb:	e8 30 02 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ad3:	89 10                	mov    %edx,(%rax)
  400ad5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ad9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400add:	e8 6e 02 00 00       	callq  400d50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400ae2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ae6:	48 83 c4 30          	add    $0x30,%rsp
  400aea:	5d                   	pop    %rbp
  400aeb:	c3                   	retq   
  400aec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400af0 <_ZN9TestClass15increase_field0Ev>:
  400af0:	55                   	push   %rbp
  400af1:	48 89 e5             	mov    %rsp,%rbp
  400af4:	48 83 ec 20          	sub    $0x20,%rsp
  400af8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400afc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b01:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b05:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b09:	48 83 c7 04          	add    $0x4,%rdi
  400b0d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b11:	48 89 c7             	mov    %rax,%rdi
  400b14:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b18:	e8 d3 00 00 00       	callq  400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b1d:	48 83 c4 20          	add    $0x20,%rsp
  400b21:	5d                   	pop    %rbp
  400b22:	c3                   	retq   
  400b23:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2a:	00 00 00 
  400b2d:	0f 1f 00             	nopl   (%rax)

0000000000400b30 <_ZN9TestClass15increase_field1Ev>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 83 ec 20          	sub    $0x20,%rsp
  400b38:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b3c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b41:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b45:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b49:	48 83 c7 04          	add    $0x4,%rdi
  400b4d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b51:	48 89 c7             	mov    %rax,%rdi
  400b54:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b58:	e8 53 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b5d:	48 83 c4 20          	add    $0x20,%rsp
  400b61:	5d                   	pop    %rbp
  400b62:	c3                   	retq   
  400b63:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b6a:	00 00 00 
  400b6d:	0f 1f 00             	nopl   (%rax)

0000000000400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 83 ec 10          	sub    $0x10,%rsp
  400b78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b80:	e8 bb 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b85:	48 83 c4 10          	add    $0x10,%rsp
  400b89:	5d                   	pop    %rbp
  400b8a:	c3                   	retq   
  400b8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b90:	55                   	push   %rbp
  400b91:	48 89 e5             	mov    %rsp,%rbp
  400b94:	48 83 ec 10          	sub    $0x10,%rsp
  400b98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b9c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400ba0:	e8 5b 01 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ba5:	48 83 c4 10          	add    $0x10,%rsp
  400ba9:	5d                   	pop    %rbp
  400baa:	c3                   	retq   
  400bab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bb0 <_ZN4ikra3soa14StaticStorage_I9TestClassLm0EEC2Ev>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb8:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bbc:	48 c7 87 a8 91 00 00 	movq   $0x0,0x91a8(%rdi)
  400bc3:	00 00 00 00 
  400bc7:	5d                   	pop    %rbp
  400bc8:	c3                   	retq   
  400bc9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400bd0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bd8:	5d                   	pop    %rbp
  400bd9:	c3                   	retq   
  400bda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400be0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400be8:	5d                   	pop    %rbp
  400be9:	c3                   	retq   
  400bea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bf0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 83 ec 30          	sub    $0x30,%rsp
  400bf8:	48 89 f8             	mov    %rdi,%rax
  400bfb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bff:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400c02:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400c06:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400c09:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c0d:	48 89 f7             	mov    %rsi,%rdi
  400c10:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c14:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400c17:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400c1b:	e8 20 00 00 00       	callq  400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c20:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400c23:	0f af 10             	imul   (%rax),%edx
  400c26:	89 10                	mov    %edx,(%rax)
  400c28:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400c2c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c30:	e8 5b 00 00 00       	callq  400c90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c35:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c39:	48 83 c4 30          	add    $0x30,%rsp
  400c3d:	5d                   	pop    %rbp
  400c3e:	c3                   	retq   
  400c3f:	90                   	nop

0000000000400c40 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c40:	55                   	push   %rbp
  400c41:	48 89 e5             	mov    %rsp,%rbp
  400c44:	48 83 ec 20          	sub    $0x20,%rsp
  400c48:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c4c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c50:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c54:	e8 47 00 00 00       	callq  400ca0 <_ZN9TestClass7storageEv>
  400c59:	48 83 c0 08          	add    $0x8,%rax
  400c5d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c61:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c65:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400c69:	48 83 e8 04          	sub    $0x4,%rax
  400c6d:	48 c1 e8 02          	shr    $0x2,%rax
  400c71:	48 c1 e0 02          	shl    $0x2,%rax
  400c75:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400c79:	48 83 c0 00          	add    $0x0,%rax
  400c7d:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c81:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c85:	48 83 c4 20          	add    $0x20,%rsp
  400c89:	5d                   	pop    %rbp
  400c8a:	c3                   	retq   
  400c8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400c90:	55                   	push   %rbp
  400c91:	48 89 e5             	mov    %rsp,%rbp
  400c94:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c98:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c9c:	5d                   	pop    %rbp
  400c9d:	c3                   	retq   
  400c9e:	66 90                	xchg   %ax,%ax

0000000000400ca0 <_ZN9TestClass7storageEv>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400cab:	00 00 00 
  400cae:	5d                   	pop    %rbp
  400caf:	c3                   	retq   

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
  400cb0:	55                   	push   %rbp
  400cb1:	48 89 e5             	mov    %rsp,%rbp
  400cb4:	48 83 ec 30          	sub    $0x30,%rsp
  400cb8:	48 89 f8             	mov    %rdi,%rax
  400cbb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400cbf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400cc2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400cc6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400cc9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400ccd:	48 89 f7             	mov    %rsi,%rdi
  400cd0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400cd4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400cd7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400cdb:	e8 20 00 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ce0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ce3:	0f af 10             	imul   (%rax),%edx
  400ce6:	89 10                	mov    %edx,(%rax)
  400ce8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cec:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cf0:	e8 5b 00 00 00       	callq  400d50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400cf5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cf9:	48 83 c4 30          	add    $0x30,%rsp
  400cfd:	5d                   	pop    %rbp
  400cfe:	c3                   	retq   
  400cff:	90                   	nop

0000000000400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d00:	55                   	push   %rbp
  400d01:	48 89 e5             	mov    %rsp,%rbp
  400d04:	48 83 ec 20          	sub    $0x20,%rsp
  400d08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d10:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400d14:	e8 87 ff ff ff       	callq  400ca0 <_ZN9TestClass7storageEv>
  400d19:	48 83 c0 08          	add    $0x8,%rax
  400d1d:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400d21:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d25:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400d29:	48 83 e8 04          	sub    $0x4,%rax
  400d2d:	48 c1 e8 02          	shr    $0x2,%rax
  400d31:	48 c1 e0 02          	shl    $0x2,%rax
  400d35:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400d39:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d3f:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d43:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d47:	48 83 c4 20          	add    $0x20,%rsp
  400d4b:	5d                   	pop    %rbp
  400d4c:	c3                   	retq   
  400d4d:	0f 1f 00             	nopl   (%rax)

0000000000400d50 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d58:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d5c:	5d                   	pop    %rbp
  400d5d:	c3                   	retq   
  400d5e:	66 90                	xchg   %ax,%ax

0000000000400d60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d60:	55                   	push   %rbp
  400d61:	48 89 e5             	mov    %rsp,%rbp
  400d64:	5d                   	pop    %rbp
  400d65:	c3                   	retq   
  400d66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d6d:	00 00 00 

0000000000400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400d70:	55                   	push   %rbp
  400d71:	48 89 e5             	mov    %rsp,%rbp
  400d74:	48 83 ec 10          	sub    $0x10,%rsp
  400d78:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d7c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d80:	e8 0b 00 00 00       	callq  400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d85:	48 83 c4 10          	add    $0x10,%rsp
  400d89:	5d                   	pop    %rbp
  400d8a:	c3                   	retq   
  400d8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d90:	55                   	push   %rbp
  400d91:	48 89 e5             	mov    %rsp,%rbp
  400d94:	48 83 ec 10          	sub    $0x10,%rsp
  400d98:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d9c:	e8 ff fe ff ff       	callq  400ca0 <_ZN9TestClass7storageEv>
  400da1:	48 83 c0 08          	add    $0x8,%rax
  400da5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400da9:	48 c1 e7 02          	shl    $0x2,%rdi
  400dad:	48 01 f8             	add    %rdi,%rax
  400db0:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400db4:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400db8:	48 83 c4 10          	add    $0x10,%rsp
  400dbc:	5d                   	pop    %rbp
  400dbd:	c3                   	retq   
  400dbe:	66 90                	xchg   %ax,%ax

0000000000400dc0 <__libc_csu_init>:
  400dc0:	41 57                	push   %r15
  400dc2:	41 56                	push   %r14
  400dc4:	41 89 ff             	mov    %edi,%r15d
  400dc7:	41 55                	push   %r13
  400dc9:	41 54                	push   %r12
  400dcb:	4c 8d 25 0e 10 20 00 	lea    0x20100e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400dd2:	55                   	push   %rbp
  400dd3:	48 8d 2d 0e 10 20 00 	lea    0x20100e(%rip),%rbp        # 601de8 <__init_array_end>
  400dda:	53                   	push   %rbx
  400ddb:	49 89 f6             	mov    %rsi,%r14
  400dde:	49 89 d5             	mov    %rdx,%r13
  400de1:	4c 29 e5             	sub    %r12,%rbp
  400de4:	48 83 ec 08          	sub    $0x8,%rsp
  400de8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dec:	e8 7f f7 ff ff       	callq  400570 <_init>
  400df1:	48 85 ed             	test   %rbp,%rbp
  400df4:	74 20                	je     400e16 <__libc_csu_init+0x56>
  400df6:	31 db                	xor    %ebx,%ebx
  400df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dff:	00 
  400e00:	4c 89 ea             	mov    %r13,%rdx
  400e03:	4c 89 f6             	mov    %r14,%rsi
  400e06:	44 89 ff             	mov    %r15d,%edi
  400e09:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e0d:	48 83 c3 01          	add    $0x1,%rbx
  400e11:	48 39 eb             	cmp    %rbp,%rbx
  400e14:	75 ea                	jne    400e00 <__libc_csu_init+0x40>
  400e16:	48 83 c4 08          	add    $0x8,%rsp
  400e1a:	5b                   	pop    %rbx
  400e1b:	5d                   	pop    %rbp
  400e1c:	41 5c                	pop    %r12
  400e1e:	41 5d                	pop    %r13
  400e20:	41 5e                	pop    %r14
  400e22:	41 5f                	pop    %r15
  400e24:	c3                   	retq   
  400e25:	90                   	nop
  400e26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2d:	00 00 00 

0000000000400e30 <__libc_csu_fini>:
  400e30:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e34 <_fini>:
  400e34:	48 83 ec 08          	sub    $0x8,%rsp
  400e38:	48 83 c4 08          	add    $0x8,%rsp
  400e3c:	c3                   	retq   
