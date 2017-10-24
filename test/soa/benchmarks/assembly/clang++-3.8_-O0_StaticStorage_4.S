
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
  4005ff:	49 c7 c0 f0 0d 40 00 	mov    $0x400df0,%r8
  400606:	48 c7 c1 80 0d 40 00 	mov    $0x400d80,%rcx
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
  4006a0:	80 3d a1 19 20 00 00 	cmpb   $0x0,0x2019a1(%rip)        # 602048 <__TMC_END__>
  4006a7:	75 11                	jne    4006ba <__do_global_dtors_aux+0x1a>
  4006a9:	55                   	push   %rbp
  4006aa:	48 89 e5             	mov    %rsp,%rbp
  4006ad:	e8 6e ff ff ff       	callq  400620 <deregister_tm_clones>
  4006b2:	5d                   	pop    %rbp
  4006b3:	c6 05 8e 19 20 00 01 	movb   $0x1,0x20198e(%rip)        # 602048 <__TMC_END__>
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
  4006ff:	b8 04 00 00 00       	mov    $0x4,%eax
  400704:	89 c7                	mov    %eax,%edi
  400706:	e8 a5 02 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  40070b:	48 89 c7             	mov    %rax,%rdi
  40070e:	48 89 c1             	mov    %rax,%rcx
  400711:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400717:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  40071b:	48 89 c7             	mov    %rax,%rdi
  40071e:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400722:	e8 c9 02 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400727:	e9 00 00 00 00       	jmpq   40072c <main+0x3c>
  40072c:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400730:	ba 77 77 00 00       	mov    $0x7777,%edx
  400735:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400739:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40073d:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400741:	48 83 c1 04          	add    $0x4,%rcx
  400745:	48 89 ce             	mov    %rcx,%rsi
  400748:	e8 f3 02 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40074d:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400751:	ba 88 88 00 00       	mov    $0x8888,%edx
  400756:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40075a:	48 83 c0 04          	add    $0x4,%rax
  40075e:	48 89 c6             	mov    %rax,%rsi
  400761:	e8 2a 03 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400766:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40076a:	e8 71 03 00 00       	callq  400ae0 <_ZN9TestClass15increase_field0Ev>
  40076f:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400773:	e8 a8 03 00 00       	callq  400b20 <_ZN9TestClass15increase_field1Ev>
  400778:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40077c:	48 83 c0 04          	add    $0x4,%rax
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	e8 d8 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400788:	8b 30                	mov    (%rax),%esi
  40078a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40078e:	48 83 c0 04          	add    $0x4,%rax
  400792:	48 89 c7             	mov    %rax,%rdi
  400795:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400798:	e8 e3 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40079d:	48 bf 04 0e 40 00 00 	movabs $0x400e04,%rdi
  4007a4:	00 00 00 
  4007a7:	8b 10                	mov    (%rax),%edx
  4007a9:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007ac:	b0 00                	mov    $0x0,%al
  4007ae:	e8 ed fd ff ff       	callq  4005a0 <printf@plt>
  4007b3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007b7:	48 83 c1 04          	add    $0x4,%rcx
  4007bb:	48 89 cf             	mov    %rcx,%rdi
  4007be:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007c1:	e8 9a 03 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007c6:	31 d2                	xor    %edx,%edx
  4007c8:	41 88 d0             	mov    %dl,%r8b
  4007cb:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007d1:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  4007d5:	0f 85 1c 00 00 00    	jne    4007f7 <main+0x107>
  4007db:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007df:	48 83 c0 04          	add    $0x4,%rax
  4007e3:	48 89 c7             	mov    %rax,%rdi
  4007e6:	e8 95 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  4007eb:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007f1:	0f 94 c1             	sete   %cl
  4007f4:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007f7:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007fa:	34 ff                	xor    $0xff,%al
  4007fc:	24 01                	and    $0x1,%al
  4007fe:	0f b6 c0             	movzbl %al,%eax
  400801:	48 83 c4 50          	add    $0x50,%rsp
  400805:	5d                   	pop    %rbp
  400806:	c3                   	retq   
  400807:	89 d1                	mov    %edx,%ecx
  400809:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40080d:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  400810:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  400814:	e8 17 02 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
  400819:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40081d:	e8 ae fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400822:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400829:	1f 84 00 00 00 00 00 

0000000000400830 <_Z12new_instancev>:
  400830:	55                   	push   %rbp
  400831:	48 89 e5             	mov    %rsp,%rbp
  400834:	48 83 ec 20          	sub    $0x20,%rsp
  400838:	b8 04 00 00 00       	mov    $0x4,%eax
  40083d:	89 c7                	mov    %eax,%edi
  40083f:	e8 6c 01 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>
  400844:	48 89 c7             	mov    %rax,%rdi
  400847:	48 89 c1             	mov    %rax,%rcx
  40084a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400850:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400854:	48 89 c7             	mov    %rax,%rdi
  400857:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40085b:	e8 90 01 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400860:	e9 00 00 00 00       	jmpq   400865 <_Z12new_instancev+0x35>
  400865:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400869:	48 83 c4 20          	add    $0x20,%rsp
  40086d:	5d                   	pop    %rbp
  40086e:	c3                   	retq   
  40086f:	89 d1                	mov    %edx,%ecx
  400871:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400875:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  400878:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40087c:	e8 af 01 00 00       	callq  400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>
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
  4008b8:	e8 83 01 00 00       	callq  400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
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
  4008e4:	e8 77 02 00 00       	callq  400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
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
  400928:	e8 63 01 00 00       	callq  400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
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
  400954:	e8 27 02 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
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
  40097c:	48 8b 04 25 f8 b1 60 	mov    0x60b1f8,%rax
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
  40099c:	48 8b 04 25 f8 b1 60 	mov    0x60b1f8,%rax
  4009a3:	00 
  4009a4:	8b 04 b8             	mov    (%rax,%rdi,4),%eax
  4009a7:	5d                   	pop    %rbp
  4009a8:	c3                   	retq   
  4009a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000004009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEnwEm>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 83 ec 10          	sub    $0x10,%rsp
  4009b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bc:	e8 5f 03 00 00       	callq  400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009c1:	8b 04 25 50 20 60 00 	mov    0x602050,%eax
  4009c8:	89 c1                	mov    %eax,%ecx
  4009ca:	83 c1 01             	add    $0x1,%ecx
  4009cd:	89 0c 25 50 20 60 00 	mov    %ecx,0x602050
  4009d4:	89 c0                	mov    %eax,%eax
  4009d6:	89 c7                	mov    %eax,%edi
  4009d8:	e8 53 03 00 00       	callq  400d30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>
  4009dd:	48 83 c4 10          	add    $0x10,%rsp
  4009e1:	5d                   	pop    %rbp
  4009e2:	c3                   	retq   
  4009e3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009ea:	00 00 00 
  4009ed:	0f 1f 00             	nopl   (%rax)

00000000004009f0 <_ZN9TestClassC2Ev>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 83 ec 10          	sub    $0x10,%rsp
  4009f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a00:	48 89 f8             	mov    %rdi,%rax
  400a03:	48 83 c0 04          	add    $0x4,%rax
  400a07:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a0b:	48 89 c7             	mov    %rax,%rdi
  400a0e:	e8 8d 01 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400a13:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a17:	48 83 c0 04          	add    $0x4,%rax
  400a1b:	48 89 c7             	mov    %rax,%rdi
  400a1e:	e8 8d 01 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400a23:	48 83 c4 10          	add    $0x10,%rsp
  400a27:	5d                   	pop    %rbp
  400a28:	c3                   	retq   
  400a29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400a30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEEdlEPv>:
  400a30:	55                   	push   %rbp
  400a31:	48 89 e5             	mov    %rsp,%rbp
  400a34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a38:	5d                   	pop    %rbp
  400a39:	c3                   	retq   
  400a3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400a40:	55                   	push   %rbp
  400a41:	48 89 e5             	mov    %rsp,%rbp
  400a44:	48 83 ec 30          	sub    $0x30,%rsp
  400a48:	48 89 f8             	mov    %rdi,%rax
  400a4b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a4f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a52:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a56:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a59:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a5d:	48 89 f7             	mov    %rsi,%rdi
  400a60:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a64:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a67:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a6b:	e8 a0 01 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a70:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a73:	89 10                	mov    %edx,(%rax)
  400a75:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a79:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a7d:	e8 de 01 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400a82:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a86:	48 83 c4 30          	add    $0x30,%rsp
  400a8a:	5d                   	pop    %rbp
  400a8b:	c3                   	retq   
  400a8c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400a90 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
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
  400abb:	e8 00 02 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ac0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ac3:	89 10                	mov    %edx,(%rax)
  400ac5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400ac9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400acd:	e8 3e 02 00 00       	callq  400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400ad2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400ad6:	48 83 c4 30          	add    $0x30,%rsp
  400ada:	5d                   	pop    %rbp
  400adb:	c3                   	retq   
  400adc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400ae0 <_ZN9TestClass15increase_field0Ev>:
  400ae0:	55                   	push   %rbp
  400ae1:	48 89 e5             	mov    %rsp,%rbp
  400ae4:	48 83 ec 20          	sub    $0x20,%rsp
  400ae8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400aec:	ba 55 55 00 00       	mov    $0x5555,%edx
  400af1:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400af5:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400af9:	48 83 c7 04          	add    $0x4,%rdi
  400afd:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b01:	48 89 c7             	mov    %rax,%rdi
  400b04:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b08:	e8 b3 00 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400b0d:	48 83 c4 20          	add    $0x20,%rsp
  400b11:	5d                   	pop    %rbp
  400b12:	c3                   	retq   
  400b13:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b1a:	00 00 00 
  400b1d:	0f 1f 00             	nopl   (%rax)

0000000000400b20 <_ZN9TestClass15increase_field1Ev>:
  400b20:	55                   	push   %rbp
  400b21:	48 89 e5             	mov    %rsp,%rbp
  400b24:	48 83 ec 20          	sub    $0x20,%rsp
  400b28:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b2c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b31:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b35:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b39:	48 83 c7 04          	add    $0x4,%rdi
  400b3d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400b41:	48 89 c7             	mov    %rax,%rdi
  400b44:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400b48:	e8 23 01 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400b4d:	48 83 c4 20          	add    $0x20,%rsp
  400b51:	5d                   	pop    %rbp
  400b52:	c3                   	retq   
  400b53:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b5a:	00 00 00 
  400b5d:	0f 1f 00             	nopl   (%rax)

0000000000400b60 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 83 ec 10          	sub    $0x10,%rsp
  400b68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b6c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b70:	e8 9b 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b75:	48 83 c4 10          	add    $0x10,%rsp
  400b79:	5d                   	pop    %rbp
  400b7a:	c3                   	retq   
  400b7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  400b80:	55                   	push   %rbp
  400b81:	48 89 e5             	mov    %rsp,%rbp
  400b84:	48 83 ec 10          	sub    $0x10,%rsp
  400b88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b90:	e8 2b 01 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b95:	48 83 c4 10          	add    $0x10,%rsp
  400b99:	5d                   	pop    %rbp
  400b9a:	c3                   	retq   
  400b9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ba8:	5d                   	pop    %rbp
  400ba9:	c3                   	retq   
  400baa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bb8:	5d                   	pop    %rbp
  400bb9:	c3                   	retq   
  400bba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400bc0:	55                   	push   %rbp
  400bc1:	48 89 e5             	mov    %rsp,%rbp
  400bc4:	48 83 ec 30          	sub    $0x30,%rsp
  400bc8:	48 89 f8             	mov    %rdi,%rax
  400bcb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400bcf:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400bd2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400bd6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400bd9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400bdd:	48 89 f7             	mov    %rsi,%rdi
  400be0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400be4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400be7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400beb:	e8 20 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bf0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400bf3:	0f af 10             	imul   (%rax),%edx
  400bf6:	89 10                	mov    %edx,(%rax)
  400bf8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400bfc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400c00:	e8 5b 00 00 00       	callq  400c60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>
  400c05:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c09:	48 83 c4 30          	add    $0x30,%rsp
  400c0d:	5d                   	pop    %rbp
  400c0e:	c3                   	retq   
  400c0f:	90                   	nop

0000000000400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c10:	55                   	push   %rbp
  400c11:	48 89 e5             	mov    %rsp,%rbp
  400c14:	48 b8 50 20 60 00 00 	movabs $0x602050,%rax
  400c1b:	00 00 00 
  400c1e:	48 83 c0 04          	add    $0x4,%rax
  400c22:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c26:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c2a:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400c2e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400c32:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400c36:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400c3a:	48 83 e8 04          	sub    $0x4,%rax
  400c3e:	48 c1 e8 02          	shr    $0x2,%rax
  400c42:	48 c1 e0 02          	shl    $0x2,%rax
  400c46:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400c4a:	48 83 c0 00          	add    $0x0,%rax
  400c4e:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400c52:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400c56:	5d                   	pop    %rbp
  400c57:	c3                   	retq   
  400c58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c5f:	00 

0000000000400c60 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS3_>:
  400c60:	55                   	push   %rbp
  400c61:	48 89 e5             	mov    %rsp,%rbp
  400c64:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c68:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400c6c:	5d                   	pop    %rbp
  400c6d:	c3                   	retq   
  400c6e:	66 90                	xchg   %ax,%ax

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
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
  400c9b:	e8 20 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ca3:	0f af 10             	imul   (%rax),%edx
  400ca6:	89 10                	mov    %edx,(%rax)
  400ca8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400cac:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400cb0:	e8 5b 00 00 00       	callq  400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>
  400cb5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400cb9:	48 83 c4 30          	add    $0x30,%rsp
  400cbd:	5d                   	pop    %rbp
  400cbe:	c3                   	retq   
  400cbf:	90                   	nop

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cc0:	55                   	push   %rbp
  400cc1:	48 89 e5             	mov    %rsp,%rbp
  400cc4:	48 b8 50 20 60 00 00 	movabs $0x602050,%rax
  400ccb:	00 00 00 
  400cce:	48 83 c0 04          	add    $0x4,%rax
  400cd2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cd6:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cda:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400cde:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400ce2:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400ce6:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400cea:	48 83 e8 04          	sub    $0x4,%rax
  400cee:	48 c1 e8 02          	shr    $0x2,%rax
  400cf2:	48 c1 e0 02          	shl    $0x2,%rax
  400cf6:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400cfa:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d00:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400d04:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400d08:	5d                   	pop    %rbp
  400d09:	c3                   	retq   
  400d0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS3_>:
  400d10:	55                   	push   %rbp
  400d11:	48 89 e5             	mov    %rsp,%rbp
  400d14:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d18:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400d1c:	5d                   	pop    %rbp
  400d1d:	c3                   	retq   
  400d1e:	66 90                	xchg   %ax,%ax

0000000000400d20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d20:	55                   	push   %rbp
  400d21:	48 89 e5             	mov    %rsp,%rbp
  400d24:	5d                   	pop    %rbp
  400d25:	c3                   	retq   
  400d26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d2d:	00 00 00 

0000000000400d30 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE3getEm>:
  400d30:	55                   	push   %rbp
  400d31:	48 89 e5             	mov    %rsp,%rbp
  400d34:	48 83 ec 10          	sub    $0x10,%rsp
  400d38:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d3c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d40:	e8 0b 00 00 00       	callq  400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d45:	48 83 c4 10          	add    $0x10,%rsp
  400d49:	5d                   	pop    %rbp
  400d4a:	c3                   	retq   
  400d4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d50 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_13StaticStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d50:	55                   	push   %rbp
  400d51:	48 89 e5             	mov    %rsp,%rbp
  400d54:	48 b8 50 20 60 00 00 	movabs $0x602050,%rax
  400d5b:	00 00 00 
  400d5e:	48 83 c0 04          	add    $0x4,%rax
  400d62:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400d66:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400d6a:	48 c1 e7 02          	shl    $0x2,%rdi
  400d6e:	48 01 f8             	add    %rdi,%rax
  400d71:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400d75:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400d79:	5d                   	pop    %rbp
  400d7a:	c3                   	retq   
  400d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d80 <__libc_csu_init>:
  400d80:	41 57                	push   %r15
  400d82:	41 56                	push   %r14
  400d84:	41 89 ff             	mov    %edi,%r15d
  400d87:	41 55                	push   %r13
  400d89:	41 54                	push   %r12
  400d8b:	4c 8d 25 4e 10 20 00 	lea    0x20104e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400d92:	55                   	push   %rbp
  400d93:	48 8d 2d 4e 10 20 00 	lea    0x20104e(%rip),%rbp        # 601de8 <__init_array_end>
  400d9a:	53                   	push   %rbx
  400d9b:	49 89 f6             	mov    %rsi,%r14
  400d9e:	49 89 d5             	mov    %rdx,%r13
  400da1:	4c 29 e5             	sub    %r12,%rbp
  400da4:	48 83 ec 08          	sub    $0x8,%rsp
  400da8:	48 c1 fd 03          	sar    $0x3,%rbp
  400dac:	e8 bf f7 ff ff       	callq  400570 <_init>
  400db1:	48 85 ed             	test   %rbp,%rbp
  400db4:	74 20                	je     400dd6 <__libc_csu_init+0x56>
  400db6:	31 db                	xor    %ebx,%ebx
  400db8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dbf:	00 
  400dc0:	4c 89 ea             	mov    %r13,%rdx
  400dc3:	4c 89 f6             	mov    %r14,%rsi
  400dc6:	44 89 ff             	mov    %r15d,%edi
  400dc9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400dcd:	48 83 c3 01          	add    $0x1,%rbx
  400dd1:	48 39 eb             	cmp    %rbp,%rbx
  400dd4:	75 ea                	jne    400dc0 <__libc_csu_init+0x40>
  400dd6:	48 83 c4 08          	add    $0x8,%rsp
  400dda:	5b                   	pop    %rbx
  400ddb:	5d                   	pop    %rbp
  400ddc:	41 5c                	pop    %r12
  400dde:	41 5d                	pop    %r13
  400de0:	41 5e                	pop    %r14
  400de2:	41 5f                	pop    %r15
  400de4:	c3                   	retq   
  400de5:	90                   	nop
  400de6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ded:	00 00 00 

0000000000400df0 <__libc_csu_fini>:
  400df0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400df4 <_fini>:
  400df4:	48 83 ec 08          	sub    $0x8,%rsp
  400df8:	48 83 c4 08          	add    $0x8,%rsp
  400dfc:	c3                   	retq   
