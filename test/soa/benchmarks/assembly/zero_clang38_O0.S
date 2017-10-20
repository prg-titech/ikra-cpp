
bin/zero_clang38_O0:     file format elf64-x86-64


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
  4005ff:	49 c7 c0 50 0c 40 00 	mov    $0x400c50,%r8
  400606:	48 c7 c1 e0 0b 40 00 	mov    $0x400be0,%rcx
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
  4006ff:	31 c0                	xor    %eax,%eax
  400701:	89 c7                	mov    %eax,%edi
  400703:	e8 b8 03 00 00       	callq  400ac0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>
  400708:	48 89 c7             	mov    %rax,%rdi
  40070b:	48 89 c1             	mov    %rax,%rcx
  40070e:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  400712:	48 89 c7             	mov    %rax,%rdi
  400715:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400719:	e8 e2 03 00 00       	callq  400b00 <_ZN9TestClassC2Ev>
  40071e:	e9 00 00 00 00       	jmpq   400723 <main+0x33>
  400723:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400727:	ba 77 77 00 00       	mov    $0x7777,%edx
  40072c:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400730:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400734:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  400738:	e8 b3 00 00 00       	callq  4007f0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  40073d:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400741:	ba 88 88 00 00       	mov    $0x8888,%edx
  400746:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40074a:	e8 f1 00 00 00       	callq  400840 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  40074f:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400753:	e8 e8 03 00 00       	callq  400b40 <_ZN9TestClass15increase_field0Ev>
  400758:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40075c:	e8 0f 04 00 00       	callq  400b70 <_ZN9TestClass15increase_field1Ev>
  400761:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400765:	e8 26 01 00 00       	callq  400890 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40076a:	8b 30                	mov    (%rax),%esi
  40076c:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400770:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400773:	e8 38 01 00 00       	callq  4008b0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  400778:	48 bf 64 0c 40 00 00 	movabs $0x400c64,%rdi
  40077f:	00 00 00 
  400782:	8b 10                	mov    (%rax),%edx
  400784:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  400787:	b0 00                	mov    $0x0,%al
  400789:	e8 12 fe ff ff       	callq  4005a0 <printf@plt>
  40078e:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400792:	89 45 c0             	mov    %eax,-0x40(%rbp)
  400795:	e8 f6 00 00 00       	callq  400890 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40079a:	31 d2                	xor    %edx,%edx
  40079c:	88 d1                	mov    %dl,%cl
  40079e:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007a4:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007a7:	0f 85 15 00 00 00    	jne    4007c2 <main+0xd2>
  4007ad:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  4007b1:	e8 fa 00 00 00       	callq  4008b0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007b6:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  4007bc:	0f 94 c1             	sete   %cl
  4007bf:	88 4d bf             	mov    %cl,-0x41(%rbp)
  4007c2:	8a 45 bf             	mov    -0x41(%rbp),%al
  4007c5:	34 ff                	xor    $0xff,%al
  4007c7:	24 01                	and    $0x1,%al
  4007c9:	0f b6 c0             	movzbl %al,%eax
  4007cc:	48 83 c4 50          	add    $0x50,%rsp
  4007d0:	5d                   	pop    %rbp
  4007d1:	c3                   	retq   
  4007d2:	89 d1                	mov    %edx,%ecx
  4007d4:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4007d8:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4007db:	48 8b 7d d0          	mov    -0x30(%rbp),%rdi
  4007df:	e8 4c 03 00 00       	callq  400b30 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEdlEPv>
  4007e4:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4007e8:	e8 e3 fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  4007ed:	0f 1f 00             	nopl   (%rax)

00000000004007f0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  4007f0:	55                   	push   %rbp
  4007f1:	48 89 e5             	mov    %rsp,%rbp
  4007f4:	48 83 ec 30          	sub    $0x30,%rsp
  4007f8:	48 89 f8             	mov    %rdi,%rax
  4007fb:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  4007ff:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400802:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400806:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400809:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40080d:	48 89 f7             	mov    %rsi,%rdi
  400810:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400814:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400817:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  40081b:	e8 d0 01 00 00       	callq  4009f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400820:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400823:	89 10                	mov    %edx,(%rax)
  400825:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400829:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  40082d:	e8 ee 01 00 00       	callq  400a20 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS4_>
  400832:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400836:	48 83 c4 30          	add    $0x30,%rsp
  40083a:	5d                   	pop    %rbp
  40083b:	c3                   	retq   
  40083c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400840 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400840:	55                   	push   %rbp
  400841:	48 89 e5             	mov    %rsp,%rbp
  400844:	48 83 ec 30          	sub    $0x30,%rsp
  400848:	48 89 f8             	mov    %rdi,%rax
  40084b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  40084f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400852:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400856:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400859:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40085d:	48 89 f7             	mov    %rsi,%rdi
  400860:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400864:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400867:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  40086b:	e8 10 02 00 00       	callq  400a80 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400870:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400873:	89 10                	mov    %edx,(%rax)
  400875:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400879:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  40087d:	e8 2e 02 00 00       	callq  400ab0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS4_>
  400882:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400886:	48 83 c4 30          	add    $0x30,%rsp
  40088a:	5d                   	pop    %rbp
  40088b:	c3                   	retq   
  40088c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400890 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400890:	55                   	push   %rbp
  400891:	48 89 e5             	mov    %rsp,%rbp
  400894:	48 83 ec 10          	sub    $0x10,%rsp
  400898:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40089c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008a0:	e8 4b 01 00 00       	callq  4009f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008a5:	48 83 c4 10          	add    $0x10,%rsp
  4008a9:	5d                   	pop    %rbp
  4008aa:	c3                   	retq   
  4008ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004008b0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  4008b0:	55                   	push   %rbp
  4008b1:	48 89 e5             	mov    %rsp,%rbp
  4008b4:	48 83 ec 10          	sub    $0x10,%rsp
  4008b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008bc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008c0:	e8 bb 01 00 00       	callq  400a80 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008c5:	48 83 c4 10          	add    $0x10,%rsp
  4008c9:	5d                   	pop    %rbp
  4008ca:	c3                   	retq   
  4008cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004008d0 <_Z12new_instancev>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 20          	sub    $0x20,%rsp
  4008d8:	31 c0                	xor    %eax,%eax
  4008da:	89 c7                	mov    %eax,%edi
  4008dc:	e8 df 01 00 00       	callq  400ac0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>
  4008e1:	48 89 c7             	mov    %rax,%rdi
  4008e4:	48 89 c1             	mov    %rax,%rcx
  4008e7:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4008eb:	48 89 c7             	mov    %rax,%rdi
  4008ee:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  4008f2:	e8 09 02 00 00       	callq  400b00 <_ZN9TestClassC2Ev>
  4008f7:	e9 00 00 00 00       	jmpq   4008fc <_Z12new_instancev+0x2c>
  4008fc:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400900:	48 83 c4 20          	add    $0x20,%rsp
  400904:	5d                   	pop    %rbp
  400905:	c3                   	retq   
  400906:	89 d1                	mov    %edx,%ecx
  400908:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40090c:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  40090f:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400913:	e8 18 02 00 00       	callq  400b30 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEdlEPv>
  400918:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  40091c:	e8 af fc ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400921:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400928:	0f 1f 84 00 00 00 00 
  40092f:	00 

0000000000400930 <_Z12write_field0P9TestClass>:
  400930:	55                   	push   %rbp
  400931:	48 89 e5             	mov    %rsp,%rbp
  400934:	48 83 ec 10          	sub    $0x10,%rsp
  400938:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40093c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400941:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400945:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400949:	48 89 c7             	mov    %rax,%rdi
  40094c:	e8 9f fe ff ff       	callq  4007f0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  400951:	48 83 c4 10          	add    $0x10,%rsp
  400955:	5d                   	pop    %rbp
  400956:	c3                   	retq   
  400957:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40095e:	00 00 

0000000000400960 <_Z11read_field0P9TestClass>:
  400960:	55                   	push   %rbp
  400961:	48 89 e5             	mov    %rsp,%rbp
  400964:	48 83 ec 10          	sub    $0x10,%rsp
  400968:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40096c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400970:	e8 1b ff ff ff       	callq  400890 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  400975:	8b 00                	mov    (%rax),%eax
  400977:	48 83 c4 10          	add    $0x10,%rsp
  40097b:	5d                   	pop    %rbp
  40097c:	c3                   	retq   
  40097d:	0f 1f 00             	nopl   (%rax)

0000000000400980 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400980:	55                   	push   %rbp
  400981:	48 89 e5             	mov    %rsp,%rbp
  400984:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400988:	5d                   	pop    %rbp
  400989:	c3                   	retq   
  40098a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400990 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400990:	55                   	push   %rbp
  400991:	48 89 e5             	mov    %rsp,%rbp
  400994:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400998:	5d                   	pop    %rbp
  400999:	c3                   	retq   
  40099a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009a0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  4009a0:	55                   	push   %rbp
  4009a1:	48 89 e5             	mov    %rsp,%rbp
  4009a4:	48 83 ec 30          	sub    $0x30,%rsp
  4009a8:	48 89 f8             	mov    %rdi,%rax
  4009ab:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  4009af:	89 55 f4             	mov    %edx,-0xc(%rbp)
  4009b2:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  4009b6:	8b 55 f4             	mov    -0xc(%rbp),%edx
  4009b9:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  4009bd:	48 89 f7             	mov    %rsi,%rdi
  4009c0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4009c4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  4009c7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4009cb:	e8 20 00 00 00       	callq  4009f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4009d0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  4009d3:	0f af 10             	imul   (%rax),%edx
  4009d6:	89 10                	mov    %edx,(%rax)
  4009d8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4009dc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  4009e0:	e8 3b 00 00 00       	callq  400a20 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS4_>
  4009e5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4009e9:	48 83 c4 30          	add    $0x30,%rsp
  4009ed:	5d                   	pop    %rbp
  4009ee:	c3                   	retq   
  4009ef:	90                   	nop

00000000004009f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 b8 4c 20 60 00 00 	movabs $0x60204c,%rax
  4009fb:	00 00 00 
  4009fe:	48 83 c0 04          	add    $0x4,%rax
  400a02:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a06:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a0a:	48 c1 e7 02          	shl    $0x2,%rdi
  400a0e:	48 01 f8             	add    %rdi,%rax
  400a11:	5d                   	pop    %rbp
  400a12:	c3                   	retq   
  400a13:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400a1a:	84 00 00 00 00 00 

0000000000400a20 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS4_>:
  400a20:	55                   	push   %rbp
  400a21:	48 89 e5             	mov    %rsp,%rbp
  400a24:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a28:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400a2c:	5d                   	pop    %rbp
  400a2d:	c3                   	retq   
  400a2e:	66 90                	xchg   %ax,%ax

0000000000400a30 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
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
  400a5b:	e8 20 00 00 00       	callq  400a80 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a60:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a63:	0f af 10             	imul   (%rax),%edx
  400a66:	89 10                	mov    %edx,(%rax)
  400a68:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a6c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a70:	e8 3b 00 00 00       	callq  400ab0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS4_>
  400a75:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a79:	48 83 c4 30          	add    $0x30,%rsp
  400a7d:	5d                   	pop    %rbp
  400a7e:	c3                   	retq   
  400a7f:	90                   	nop

0000000000400a80 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a80:	55                   	push   %rbp
  400a81:	48 89 e5             	mov    %rsp,%rbp
  400a84:	48 b8 4c 20 60 00 00 	movabs $0x60204c,%rax
  400a8b:	00 00 00 
  400a8e:	48 83 c0 04          	add    $0x4,%rax
  400a92:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a96:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a9a:	48 c1 e7 02          	shl    $0x2,%rdi
  400a9e:	48 01 f8             	add    %rdi,%rax
  400aa1:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400aa7:	5d                   	pop    %rbp
  400aa8:	c3                   	retq   
  400aa9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400ab0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS4_>:
  400ab0:	55                   	push   %rbp
  400ab1:	48 89 e5             	mov    %rsp,%rbp
  400ab4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400ab8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400abc:	5d                   	pop    %rbp
  400abd:	c3                   	retq   
  400abe:	66 90                	xchg   %ax,%ax

0000000000400ac0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEnwEm>:
  400ac0:	55                   	push   %rbp
  400ac1:	48 89 e5             	mov    %rsp,%rbp
  400ac4:	48 83 ec 10          	sub    $0x10,%rsp
  400ac8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400acc:	e8 cf 00 00 00       	callq  400ba0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400ad1:	8b 04 25 4c 20 60 00 	mov    0x60204c,%eax
  400ad8:	89 c1                	mov    %eax,%ecx
  400ada:	83 c1 01             	add    $0x1,%ecx
  400add:	89 0c 25 4c 20 60 00 	mov    %ecx,0x60204c
  400ae4:	89 c0                	mov    %eax,%eax
  400ae6:	89 c7                	mov    %eax,%edi
  400ae8:	e8 c3 00 00 00       	callq  400bb0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE3getEm>
  400aed:	48 83 c4 10          	add    $0x10,%rsp
  400af1:	5d                   	pop    %rbp
  400af2:	c3                   	retq   
  400af3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400afa:	00 00 00 
  400afd:	0f 1f 00             	nopl   (%rax)

0000000000400b00 <_ZN9TestClassC2Ev>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 83 ec 10          	sub    $0x10,%rsp
  400b08:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b0c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b10:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400b14:	e8 67 fe ff ff       	callq  400980 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  400b19:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400b1d:	e8 6e fe ff ff       	callq  400990 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400b22:	48 83 c4 10          	add    $0x10,%rsp
  400b26:	5d                   	pop    %rbp
  400b27:	c3                   	retq   
  400b28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400b2f:	00 

0000000000400b30 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EEdlEPv>:
  400b30:	55                   	push   %rbp
  400b31:	48 89 e5             	mov    %rsp,%rbp
  400b34:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b38:	5d                   	pop    %rbp
  400b39:	c3                   	retq   
  400b3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b40 <_ZN9TestClass15increase_field0Ev>:
  400b40:	55                   	push   %rbp
  400b41:	48 89 e5             	mov    %rsp,%rbp
  400b44:	48 83 ec 10          	sub    $0x10,%rsp
  400b48:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b4c:	ba 55 55 00 00       	mov    $0x5555,%edx
  400b51:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b55:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b59:	48 89 c7             	mov    %rax,%rdi
  400b5c:	e8 3f fe ff ff       	callq  4009a0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400b61:	48 83 c4 10          	add    $0x10,%rsp
  400b65:	5d                   	pop    %rbp
  400b66:	c3                   	retq   
  400b67:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b6e:	00 00 

0000000000400b70 <_ZN9TestClass15increase_field1Ev>:
  400b70:	55                   	push   %rbp
  400b71:	48 89 e5             	mov    %rsp,%rbp
  400b74:	48 83 ec 10          	sub    $0x10,%rsp
  400b78:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400b7c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b81:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b85:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400b89:	48 89 c7             	mov    %rax,%rdi
  400b8c:	e8 9f fe ff ff       	callq  400a30 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400b91:	48 83 c4 10          	add    $0x10,%rsp
  400b95:	5d                   	pop    %rbp
  400b96:	c3                   	retq   
  400b97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400b9e:	00 00 

0000000000400ba0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	5d                   	pop    %rbp
  400ba5:	c3                   	retq   
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 

0000000000400bb0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE3getEm>:
  400bb0:	55                   	push   %rbp
  400bb1:	48 89 e5             	mov    %rsp,%rbp
  400bb4:	48 83 ec 10          	sub    $0x10,%rsp
  400bb8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bbc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bc0:	e8 0b 00 00 00       	callq  400bd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400bc5:	48 83 c4 10          	add    $0x10,%rsp
  400bc9:	5d                   	pop    %rbp
  400bca:	c3                   	retq   
  400bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi0EE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400bd0:	55                   	push   %rbp
  400bd1:	48 89 e5             	mov    %rsp,%rbp
  400bd4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bd8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  400bdc:	5d                   	pop    %rbp
  400bdd:	c3                   	retq   
  400bde:	66 90                	xchg   %ax,%ax

0000000000400be0 <__libc_csu_init>:
  400be0:	41 57                	push   %r15
  400be2:	41 56                	push   %r14
  400be4:	41 89 ff             	mov    %edi,%r15d
  400be7:	41 55                	push   %r13
  400be9:	41 54                	push   %r12
  400beb:	4c 8d 25 ee 11 20 00 	lea    0x2011ee(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400bf2:	55                   	push   %rbp
  400bf3:	48 8d 2d ee 11 20 00 	lea    0x2011ee(%rip),%rbp        # 601de8 <__init_array_end>
  400bfa:	53                   	push   %rbx
  400bfb:	49 89 f6             	mov    %rsi,%r14
  400bfe:	49 89 d5             	mov    %rdx,%r13
  400c01:	4c 29 e5             	sub    %r12,%rbp
  400c04:	48 83 ec 08          	sub    $0x8,%rsp
  400c08:	48 c1 fd 03          	sar    $0x3,%rbp
  400c0c:	e8 5f f9 ff ff       	callq  400570 <_init>
  400c11:	48 85 ed             	test   %rbp,%rbp
  400c14:	74 20                	je     400c36 <__libc_csu_init+0x56>
  400c16:	31 db                	xor    %ebx,%ebx
  400c18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c1f:	00 
  400c20:	4c 89 ea             	mov    %r13,%rdx
  400c23:	4c 89 f6             	mov    %r14,%rsi
  400c26:	44 89 ff             	mov    %r15d,%edi
  400c29:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400c2d:	48 83 c3 01          	add    $0x1,%rbx
  400c31:	48 39 eb             	cmp    %rbp,%rbx
  400c34:	75 ea                	jne    400c20 <__libc_csu_init+0x40>
  400c36:	48 83 c4 08          	add    $0x8,%rsp
  400c3a:	5b                   	pop    %rbx
  400c3b:	5d                   	pop    %rbp
  400c3c:	41 5c                	pop    %r12
  400c3e:	41 5d                	pop    %r13
  400c40:	41 5e                	pop    %r14
  400c42:	41 5f                	pop    %r15
  400c44:	c3                   	retq   
  400c45:	90                   	nop
  400c46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c4d:	00 00 00 

0000000000400c50 <__libc_csu_fini>:
  400c50:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400c54 <_fini>:
  400c54:	48 83 ec 08          	sub    $0x8,%rsp
  400c58:	48 83 c4 08          	add    $0x8,%rsp
  400c5c:	c3                   	retq   
