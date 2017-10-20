
bin/valid_clang38_O0:     file format elf64-x86-64


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
  4005ff:	49 c7 c0 40 0d 40 00 	mov    $0x400d40,%r8
  400606:	48 c7 c1 d0 0c 40 00 	mov    $0x400cd0,%rcx
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
  400706:	e8 55 04 00 00       	callq  400b60 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>
  40070b:	48 89 c7             	mov    %rax,%rdi
  40070e:	48 89 c1             	mov    %rax,%rcx
  400711:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400717:	48 89 7d d0          	mov    %rdi,-0x30(%rbp)
  40071b:	48 89 c7             	mov    %rax,%rdi
  40071e:	48 89 4d c8          	mov    %rcx,-0x38(%rbp)
  400722:	e8 79 04 00 00       	callq  400ba0 <_ZN9TestClassC2Ev>
  400727:	e9 00 00 00 00       	jmpq   40072c <main+0x3c>
  40072c:	48 8d 7d e0          	lea    -0x20(%rbp),%rdi
  400730:	ba 77 77 00 00       	mov    $0x7777,%edx
  400735:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  400739:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40073d:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  400741:	48 83 c1 04          	add    $0x4,%rcx
  400745:	48 89 ce             	mov    %rcx,%rsi
  400748:	e8 e3 00 00 00       	callq  400830 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40074d:	48 8d 7d d8          	lea    -0x28(%rbp),%rdi
  400751:	ba 88 88 00 00       	mov    $0x8888,%edx
  400756:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40075a:	48 83 c0 04          	add    $0x4,%rax
  40075e:	48 89 c6             	mov    %rax,%rsi
  400761:	e8 1a 01 00 00       	callq  400880 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEaSEi>
  400766:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  40076a:	e8 81 04 00 00       	callq  400bf0 <_ZN9TestClass15increase_field0Ev>
  40076f:	48 8b 7d f0          	mov    -0x10(%rbp),%rdi
  400773:	e8 b8 04 00 00       	callq  400c30 <_ZN9TestClass15increase_field1Ev>
  400778:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40077c:	48 83 c0 04          	add    $0x4,%rax
  400780:	48 89 c7             	mov    %rax,%rdi
  400783:	e8 48 01 00 00       	callq  4008d0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  400788:	8b 30                	mov    (%rax),%esi
  40078a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40078e:	48 83 c0 04          	add    $0x4,%rax
  400792:	48 89 c7             	mov    %rax,%rdi
  400795:	89 75 c4             	mov    %esi,-0x3c(%rbp)
  400798:	e8 53 01 00 00       	callq  4008f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
  40079d:	48 bf 54 0d 40 00 00 	movabs $0x400d54,%rdi
  4007a4:	00 00 00 
  4007a7:	8b 10                	mov    (%rax),%edx
  4007a9:	8b 75 c4             	mov    -0x3c(%rbp),%esi
  4007ac:	b0 00                	mov    $0x0,%al
  4007ae:	e8 ed fd ff ff       	callq  4005a0 <printf@plt>
  4007b3:	48 8b 4d f0          	mov    -0x10(%rbp),%rcx
  4007b7:	48 83 c1 04          	add    $0x4,%rcx
  4007bb:	48 89 cf             	mov    %rcx,%rdi
  4007be:	89 45 c0             	mov    %eax,-0x40(%rbp)
  4007c1:	e8 0a 01 00 00       	callq  4008d0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4007c6:	31 d2                	xor    %edx,%edx
  4007c8:	41 88 d0             	mov    %dl,%r8b
  4007cb:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  4007d1:	44 88 45 bf          	mov    %r8b,-0x41(%rbp)
  4007d5:	0f 85 1c 00 00 00    	jne    4007f7 <main+0x107>
  4007db:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4007df:	48 83 c0 04          	add    $0x4,%rax
  4007e3:	48 89 c7             	mov    %rax,%rdi
  4007e6:	e8 05 01 00 00       	callq  4008f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>
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
  400814:	e8 c7 03 00 00       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEdlEPv>
  400819:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40081d:	e8 ae fd ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  400822:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400829:	1f 84 00 00 00 00 00 

0000000000400830 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>:
  400830:	55                   	push   %rbp
  400831:	48 89 e5             	mov    %rsp,%rbp
  400834:	48 83 ec 30          	sub    $0x30,%rsp
  400838:	48 89 f8             	mov    %rdi,%rax
  40083b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  40083f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400842:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400846:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400849:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40084d:	48 89 f7             	mov    %rsi,%rdi
  400850:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400854:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400857:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  40085b:	e8 f0 01 00 00       	callq  400a50 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400860:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400863:	89 10                	mov    %edx,(%rax)
  400865:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400869:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  40086d:	e8 2e 02 00 00       	callq  400aa0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS4_>
  400872:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400876:	48 83 c4 30          	add    $0x30,%rsp
  40087a:	5d                   	pop    %rbp
  40087b:	c3                   	retq   
  40087c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400880 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEaSEi>:
  400880:	55                   	push   %rbp
  400881:	48 89 e5             	mov    %rsp,%rbp
  400884:	48 83 ec 30          	sub    $0x30,%rsp
  400888:	48 89 f8             	mov    %rdi,%rax
  40088b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  40088f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400892:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400896:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400899:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  40089d:	48 89 f7             	mov    %rsi,%rdi
  4008a0:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4008a4:	89 55 dc             	mov    %edx,-0x24(%rbp)
  4008a7:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  4008ab:	e8 50 02 00 00       	callq  400b00 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008b0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  4008b3:	89 10                	mov    %edx,(%rax)
  4008b5:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  4008b9:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  4008bd:	e8 8e 02 00 00       	callq  400b50 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS4_>
  4008c2:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4008c6:	48 83 c4 30          	add    $0x30,%rsp
  4008ca:	5d                   	pop    %rbp
  4008cb:	c3                   	retq   
  4008cc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004008d0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>:
  4008d0:	55                   	push   %rbp
  4008d1:	48 89 e5             	mov    %rsp,%rbp
  4008d4:	48 83 ec 10          	sub    $0x10,%rsp
  4008d8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008dc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4008e0:	e8 6b 01 00 00       	callq  400a50 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  4008e5:	48 83 c4 10          	add    $0x10,%rsp
  4008e9:	5d                   	pop    %rbp
  4008ea:	c3                   	retq   
  4008eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004008f0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEcvRiEv>:
  4008f0:	55                   	push   %rbp
  4008f1:	48 89 e5             	mov    %rsp,%rbp
  4008f4:	48 83 ec 10          	sub    $0x10,%rsp
  4008f8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4008fc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400900:	e8 fb 01 00 00       	callq  400b00 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400905:	48 83 c4 10          	add    $0x10,%rsp
  400909:	5d                   	pop    %rbp
  40090a:	c3                   	retq   
  40090b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400910 <_Z12new_instancev>:
  400910:	55                   	push   %rbp
  400911:	48 89 e5             	mov    %rsp,%rbp
  400914:	48 83 ec 20          	sub    $0x20,%rsp
  400918:	b8 04 00 00 00       	mov    $0x4,%eax
  40091d:	89 c7                	mov    %eax,%edi
  40091f:	e8 3c 02 00 00       	callq  400b60 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>
  400924:	48 89 c7             	mov    %rax,%rdi
  400927:	48 89 c1             	mov    %rax,%rcx
  40092a:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400930:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400934:	48 89 c7             	mov    %rax,%rdi
  400937:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
  40093b:	e8 60 02 00 00       	callq  400ba0 <_ZN9TestClassC2Ev>
  400940:	e9 00 00 00 00       	jmpq   400945 <_Z12new_instancev+0x35>
  400945:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400949:	48 83 c4 20          	add    $0x20,%rsp
  40094d:	5d                   	pop    %rbp
  40094e:	c3                   	retq   
  40094f:	89 d1                	mov    %edx,%ecx
  400951:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  400955:	89 4d f4             	mov    %ecx,-0xc(%rbp)
  400958:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  40095c:	e8 7f 02 00 00       	callq  400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEdlEPv>
  400961:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400965:	e8 66 fc ff ff       	callq  4005d0 <_Unwind_Resume@plt>
  40096a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400970 <_Z12write_field0P9TestClass>:
  400970:	55                   	push   %rbp
  400971:	48 89 e5             	mov    %rsp,%rbp
  400974:	48 83 ec 20          	sub    $0x20,%rsp
  400978:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  40097c:	ba 77 77 00 00       	mov    $0x7777,%edx
  400981:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400985:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400989:	48 83 c7 04          	add    $0x4,%rdi
  40098d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400991:	48 89 c7             	mov    %rax,%rdi
  400994:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400998:	e8 93 fe ff ff       	callq  400830 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEaSEi>
  40099d:	48 83 c4 20          	add    $0x20,%rsp
  4009a1:	5d                   	pop    %rbp
  4009a2:	c3                   	retq   
  4009a3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009aa:	84 00 00 00 00 00 

00000000004009b0 <_Z11read_field0P9TestClass>:
  4009b0:	55                   	push   %rbp
  4009b1:	48 89 e5             	mov    %rsp,%rbp
  4009b4:	48 83 ec 10          	sub    $0x10,%rsp
  4009b8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009bc:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  4009c0:	48 83 c7 04          	add    $0x4,%rdi
  4009c4:	e8 07 ff ff ff       	callq  4008d0 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEcvRiEv>
  4009c9:	8b 00                	mov    (%rax),%eax
  4009cb:	48 83 c4 10          	add    $0x10,%rsp
  4009cf:	5d                   	pop    %rbp
  4009d0:	c3                   	retq   
  4009d1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4009d8:	0f 1f 84 00 00 00 00 
  4009df:	00 

00000000004009e0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>:
  4009e0:	55                   	push   %rbp
  4009e1:	48 89 e5             	mov    %rsp,%rbp
  4009e4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009e8:	5d                   	pop    %rbp
  4009e9:	c3                   	retq   
  4009ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004009f0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>:
  4009f0:	55                   	push   %rbp
  4009f1:	48 89 e5             	mov    %rsp,%rbp
  4009f4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4009f8:	5d                   	pop    %rbp
  4009f9:	c3                   	retq   
  4009fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400a00 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEmLEi>:
  400a00:	55                   	push   %rbp
  400a01:	48 89 e5             	mov    %rsp,%rbp
  400a04:	48 83 ec 30          	sub    $0x30,%rsp
  400a08:	48 89 f8             	mov    %rdi,%rax
  400a0b:	48 89 75 f8          	mov    %rsi,-0x8(%rbp)
  400a0f:	89 55 f4             	mov    %edx,-0xc(%rbp)
  400a12:	48 8b 75 f8          	mov    -0x8(%rbp),%rsi
  400a16:	8b 55 f4             	mov    -0xc(%rbp),%edx
  400a19:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400a1d:	48 89 f7             	mov    %rsi,%rdi
  400a20:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a24:	89 55 dc             	mov    %edx,-0x24(%rbp)
  400a27:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
  400a2b:	e8 20 00 00 00       	callq  400a50 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a30:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400a33:	0f af 10             	imul   (%rax),%edx
  400a36:	89 10                	mov    %edx,(%rax)
  400a38:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400a3c:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400a40:	e8 5b 00 00 00       	callq  400aa0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS4_>
  400a45:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a49:	48 83 c4 30          	add    $0x30,%rsp
  400a4d:	5d                   	pop    %rbp
  400a4e:	c3                   	retq   
  400a4f:	90                   	nop

0000000000400a50 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400a50:	55                   	push   %rbp
  400a51:	48 89 e5             	mov    %rsp,%rbp
  400a54:	48 b8 4c 20 60 00 00 	movabs $0x60204c,%rax
  400a5b:	00 00 00 
  400a5e:	48 83 c0 04          	add    $0x4,%rax
  400a62:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400a66:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400a6a:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400a6e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a72:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400a76:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400a7a:	48 83 e8 04          	sub    $0x4,%rax
  400a7e:	48 c1 e8 02          	shr    $0x2,%rax
  400a82:	48 c1 e0 02          	shl    $0x2,%rax
  400a86:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400a8a:	48 83 c0 00          	add    $0x0,%rax
  400a8e:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400a92:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400a96:	5d                   	pop    %rbp
  400a97:	c3                   	retq   
  400a98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a9f:	00 

0000000000400aa0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC2ERKS4_>:
  400aa0:	55                   	push   %rbp
  400aa1:	48 89 e5             	mov    %rsp,%rbp
  400aa4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400aa8:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400aac:	5d                   	pop    %rbp
  400aad:	c3                   	retq   
  400aae:	66 90                	xchg   %ax,%ax

0000000000400ab0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEmLEi>:
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
  400adb:	e8 20 00 00 00       	callq  400b00 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ae0:	8b 55 dc             	mov    -0x24(%rbp),%edx
  400ae3:	0f af 10             	imul   (%rax),%edx
  400ae6:	89 10                	mov    %edx,(%rax)
  400ae8:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
  400aec:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  400af0:	e8 5b 00 00 00       	callq  400b50 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS4_>
  400af5:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400af9:	48 83 c4 30          	add    $0x30,%rsp
  400afd:	5d                   	pop    %rbp
  400afe:	c3                   	retq   
  400aff:	90                   	nop

0000000000400b00 <_ZNK4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400b00:	55                   	push   %rbp
  400b01:	48 89 e5             	mov    %rsp,%rbp
  400b04:	48 b8 4c 20 60 00 00 	movabs $0x60204c,%rax
  400b0b:	00 00 00 
  400b0e:	48 83 c0 04          	add    $0x4,%rax
  400b12:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b16:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400b1a:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400b1e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400b22:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400b26:	48 2b 45 e8          	sub    -0x18(%rbp),%rax
  400b2a:	48 83 e8 04          	sub    $0x4,%rax
  400b2e:	48 c1 e8 02          	shr    $0x2,%rax
  400b32:	48 c1 e0 02          	shl    $0x2,%rax
  400b36:	48 03 45 e8          	add    -0x18(%rbp),%rax
  400b3a:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400b40:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  400b44:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  400b48:	5d                   	pop    %rbp
  400b49:	c3                   	retq   
  400b4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400b50 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC2ERKS4_>:
  400b50:	55                   	push   %rbp
  400b51:	48 89 e5             	mov    %rsp,%rbp
  400b54:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b58:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  400b5c:	5d                   	pop    %rbp
  400b5d:	c3                   	retq   
  400b5e:	66 90                	xchg   %ax,%ax

0000000000400b60 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEnwEm>:
  400b60:	55                   	push   %rbp
  400b61:	48 89 e5             	mov    %rsp,%rbp
  400b64:	48 83 ec 10          	sub    $0x10,%rsp
  400b68:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400b6c:	e8 ff 00 00 00       	callq  400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400b71:	8b 04 25 4c 20 60 00 	mov    0x60204c,%eax
  400b78:	89 c1                	mov    %eax,%ecx
  400b7a:	83 c1 01             	add    $0x1,%ecx
  400b7d:	89 0c 25 4c 20 60 00 	mov    %ecx,0x60204c
  400b84:	89 c0                	mov    %eax,%eax
  400b86:	89 c7                	mov    %eax,%edi
  400b88:	e8 f3 00 00 00       	callq  400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE3getEm>
  400b8d:	48 83 c4 10          	add    $0x10,%rsp
  400b91:	5d                   	pop    %rbp
  400b92:	c3                   	retq   
  400b93:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9a:	00 00 00 
  400b9d:	0f 1f 00             	nopl   (%rax)

0000000000400ba0 <_ZN9TestClassC2Ev>:
  400ba0:	55                   	push   %rbp
  400ba1:	48 89 e5             	mov    %rsp,%rbp
  400ba4:	48 83 ec 10          	sub    $0x10,%rsp
  400ba8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400bac:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400bb0:	48 89 f8             	mov    %rdi,%rax
  400bb3:	48 83 c0 04          	add    $0x4,%rax
  400bb7:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
  400bbb:	48 89 c7             	mov    %rax,%rdi
  400bbe:	e8 1d fe ff ff       	callq  4009e0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEC2Ev>
  400bc3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400bc7:	48 83 c0 04          	add    $0x4,%rax
  400bcb:	48 89 c7             	mov    %rax,%rdi
  400bce:	e8 1d fe ff ff       	callq  4009f0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEC2Ev>
  400bd3:	48 83 c4 10          	add    $0x10,%rsp
  400bd7:	5d                   	pop    %rbp
  400bd8:	c3                   	retq   
  400bd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400be0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EEdlEPv>:
  400be0:	55                   	push   %rbp
  400be1:	48 89 e5             	mov    %rsp,%rbp
  400be4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400be8:	5d                   	pop    %rbp
  400be9:	c3                   	retq   
  400bea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400bf0 <_ZN9TestClass15increase_field0Ev>:
  400bf0:	55                   	push   %rbp
  400bf1:	48 89 e5             	mov    %rsp,%rbp
  400bf4:	48 83 ec 20          	sub    $0x20,%rsp
  400bf8:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400bfc:	ba 55 55 00 00       	mov    $0x5555,%edx
  400c01:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c05:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c09:	48 83 c7 04          	add    $0x4,%rdi
  400c0d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c11:	48 89 c7             	mov    %rax,%rdi
  400c14:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400c18:	e8 e3 fd ff ff       	callq  400a00 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj0ELi4E9TestClassEmLEi>
  400c1d:	48 83 c4 20          	add    $0x20,%rsp
  400c21:	5d                   	pop    %rbp
  400c22:	c3                   	retq   
  400c23:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c2a:	00 00 00 
  400c2d:	0f 1f 00             	nopl   (%rax)

0000000000400c30 <_ZN9TestClass15increase_field1Ev>:
  400c30:	55                   	push   %rbp
  400c31:	48 89 e5             	mov    %rsp,%rbp
  400c34:	48 83 ec 20          	sub    $0x20,%rsp
  400c38:	48 8d 45 f0          	lea    -0x10(%rbp),%rax
  400c3c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400c41:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c45:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c49:	48 83 c7 04          	add    $0x4,%rdi
  400c4d:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
  400c51:	48 89 c7             	mov    %rax,%rdi
  400c54:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  400c58:	e8 53 fe ff ff       	callq  400ab0 <_ZN4ikra3soa12_GLOBAL__N_16Field_IiLm4660ELj4ELi4E9TestClassEmLEi>
  400c5d:	48 83 c4 20          	add    $0x20,%rsp
  400c61:	5d                   	pop    %rbp
  400c62:	c3                   	retq   
  400c63:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c6a:	00 00 00 
  400c6d:	0f 1f 00             	nopl   (%rax)

0000000000400c70 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400c70:	55                   	push   %rbp
  400c71:	48 89 e5             	mov    %rsp,%rbp
  400c74:	5d                   	pop    %rbp
  400c75:	c3                   	retq   
  400c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c7d:	00 00 00 

0000000000400c80 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE3getEm>:
  400c80:	55                   	push   %rbp
  400c81:	48 89 e5             	mov    %rsp,%rbp
  400c84:	48 83 ec 10          	sub    $0x10,%rsp
  400c88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400c8c:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400c90:	e8 0b 00 00 00       	callq  400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400c95:	48 83 c4 10          	add    $0x10,%rsp
  400c99:	5d                   	pop    %rbp
  400c9a:	c3                   	retq   
  400c9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ca0 <_ZN4ikra3soa9SoaLayoutI9TestClassLj8ELm4660ELi4EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400ca0:	55                   	push   %rbp
  400ca1:	48 89 e5             	mov    %rsp,%rbp
  400ca4:	48 b8 4c 20 60 00 00 	movabs $0x60204c,%rax
  400cab:	00 00 00 
  400cae:	48 83 c0 04          	add    $0x4,%rax
  400cb2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  400cb6:	48 8b 7d f8          	mov    -0x8(%rbp),%rdi
  400cba:	48 c1 e7 02          	shl    $0x2,%rdi
  400cbe:	48 01 f8             	add    %rdi,%rax
  400cc1:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  400cc5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  400cc9:	5d                   	pop    %rbp
  400cca:	c3                   	retq   
  400ccb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cd0 <__libc_csu_init>:
  400cd0:	41 57                	push   %r15
  400cd2:	41 56                	push   %r14
  400cd4:	41 89 ff             	mov    %edi,%r15d
  400cd7:	41 55                	push   %r13
  400cd9:	41 54                	push   %r12
  400cdb:	4c 8d 25 fe 10 20 00 	lea    0x2010fe(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400ce2:	55                   	push   %rbp
  400ce3:	48 8d 2d fe 10 20 00 	lea    0x2010fe(%rip),%rbp        # 601de8 <__init_array_end>
  400cea:	53                   	push   %rbx
  400ceb:	49 89 f6             	mov    %rsi,%r14
  400cee:	49 89 d5             	mov    %rdx,%r13
  400cf1:	4c 29 e5             	sub    %r12,%rbp
  400cf4:	48 83 ec 08          	sub    $0x8,%rsp
  400cf8:	48 c1 fd 03          	sar    $0x3,%rbp
  400cfc:	e8 6f f8 ff ff       	callq  400570 <_init>
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
