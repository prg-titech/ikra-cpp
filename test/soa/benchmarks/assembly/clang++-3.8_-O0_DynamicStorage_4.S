
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
  40064f:	49 c7 c0 10 0f 40 00 	mov    $0x400f10,%r8
  400656:	48 c7 c1 a0 0e 40 00 	mov    $0x400ea0,%rcx
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
  400740:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400747:	00 00 00 
  40074a:	c3                   	retq   
  40074b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400750 <main>:
  400750:	48 83 ec 58          	sub    $0x58,%rsp
  400754:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40075b:	00 
  40075c:	e8 7f 02 00 00       	callq  4009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400761:	b8 04 00 00 00       	mov    $0x4,%eax
  400766:	89 c7                	mov    %eax,%edi
  400768:	e8 83 02 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  40076d:	48 89 c7             	mov    %rax,%rdi
  400770:	48 89 c1             	mov    %rax,%rcx
  400773:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  400779:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  40077e:	48 89 c7             	mov    %rax,%rdi
  400781:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  400786:	e8 95 02 00 00       	callq  400a20 <_ZN9TestClassC2Ev>
  40078b:	e9 00 00 00 00       	jmpq   400790 <main+0x40>
  400790:	be 77 77 00 00       	mov    $0x7777,%esi
  400795:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40079a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40079f:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007a4:	48 83 c1 04          	add    $0x4,%rcx
  4007a8:	48 89 cf             	mov    %rcx,%rdi
  4007ab:	e8 c0 02 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  4007b0:	be 88 88 00 00       	mov    $0x8888,%esi
  4007b5:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  4007ba:	48 83 c1 04          	add    $0x4,%rcx
  4007be:	48 89 cf             	mov    %rcx,%rdi
  4007c1:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4007c6:	e8 e5 02 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  4007cb:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007d0:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007d5:	e8 16 03 00 00       	callq  400af0 <_ZN9TestClass15increase_field0Ev>
  4007da:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007df:	e8 4c 03 00 00       	callq  400b30 <_ZN9TestClass15increase_field1Ev>
  4007e4:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007e9:	48 83 c0 04          	add    $0x4,%rax
  4007ed:	48 89 c7             	mov    %rax,%rdi
  4007f0:	e8 7b 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  4007f5:	8b 30                	mov    (%rax),%esi
  4007f7:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  4007fc:	48 83 c0 04          	add    $0x4,%rax
  400800:	48 89 c7             	mov    %rax,%rdi
  400803:	89 74 24 14          	mov    %esi,0x14(%rsp)
  400807:	e8 74 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  40080c:	48 bf 24 0f 40 00 00 	movabs $0x400f24,%rdi
  400813:	00 00 00 
  400816:	8b 10                	mov    (%rax),%edx
  400818:	8b 74 24 14          	mov    0x14(%rsp),%esi
  40081c:	b0 00                	mov    $0x0,%al
  40081e:	e8 bd fd ff ff       	callq  4005e0 <printf@plt>
  400823:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  400828:	48 83 c1 04          	add    $0x4,%rcx
  40082c:	48 89 cf             	mov    %rcx,%rdi
  40082f:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400833:	e8 38 03 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400838:	31 d2                	xor    %edx,%edx
  40083a:	41 88 d0             	mov    %dl,%r8b
  40083d:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400843:	44 88 44 24 0f       	mov    %r8b,0xf(%rsp)
  400848:	0f 85 1e 00 00 00    	jne    40086c <main+0x11c>
  40084e:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  400853:	48 83 c0 04          	add    $0x4,%rax
  400857:	48 89 c7             	mov    %rax,%rdi
  40085a:	e8 21 03 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  40085f:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400865:	0f 94 c1             	sete   %cl
  400868:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40086c:	8a 44 24 0f          	mov    0xf(%rsp),%al
  400870:	34 ff                	xor    $0xff,%al
  400872:	24 01                	and    $0x1,%al
  400874:	0f b6 c0             	movzbl %al,%eax
  400877:	48 83 c4 58          	add    $0x58,%rsp
  40087b:	c3                   	retq   
  40087c:	89 d1                	mov    %edx,%ecx
  40087e:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  400883:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400887:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  40088c:	e8 cf 01 00 00       	callq  400a60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  400891:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400896:	e8 85 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  40089b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004008a0 <_Z12new_instancev>:
  4008a0:	48 83 ec 28          	sub    $0x28,%rsp
  4008a4:	b8 04 00 00 00       	mov    $0x4,%eax
  4008a9:	89 c7                	mov    %eax,%edi
  4008ab:	e8 40 01 00 00       	callq  4009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4008b0:	48 89 c7             	mov    %rax,%rdi
  4008b3:	48 89 c1             	mov    %rax,%rcx
  4008b6:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4008bc:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008c1:	48 89 c7             	mov    %rax,%rdi
  4008c4:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  4008c9:	e8 52 01 00 00       	callq  400a20 <_ZN9TestClassC2Ev>
  4008ce:	e9 00 00 00 00       	jmpq   4008d3 <_Z12new_instancev+0x33>
  4008d3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008d8:	48 83 c4 28          	add    $0x28,%rsp
  4008dc:	c3                   	retq   
  4008dd:	89 d1                	mov    %edx,%ecx
  4008df:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4008e4:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  4008e8:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008ed:	e8 6e 01 00 00       	callq  400a60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>
  4008f2:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008f7:	e8 24 fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008fc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400900 <_Z12write_field0P9TestClass>:
  400900:	48 83 ec 18          	sub    $0x18,%rsp
  400904:	be 77 77 00 00       	mov    $0x7777,%esi
  400909:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40090e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400913:	48 83 c7 04          	add    $0x4,%rdi
  400917:	e8 54 01 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>
  40091c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400921:	48 83 c4 18          	add    $0x18,%rsp
  400925:	c3                   	retq   
  400926:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40092d:	00 00 00 

0000000000400930 <_Z11read_field0P9TestClass>:
  400930:	50                   	push   %rax
  400931:	48 89 3c 24          	mov    %rdi,(%rsp)
  400935:	48 8b 3c 24          	mov    (%rsp),%rdi
  400939:	48 83 c7 04          	add    $0x4,%rdi
  40093d:	e8 2e 02 00 00       	callq  400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>
  400942:	8b 00                	mov    (%rax),%eax
  400944:	59                   	pop    %rcx
  400945:	c3                   	retq   
  400946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40094d:	00 00 00 

0000000000400950 <_Z12write_field1P9TestClass>:
  400950:	48 83 ec 18          	sub    $0x18,%rsp
  400954:	be 77 77 00 00       	mov    $0x7777,%esi
  400959:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40095e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400963:	48 83 c7 04          	add    $0x4,%rdi
  400967:	e8 44 01 00 00       	callq  400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>
  40096c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400971:	48 83 c4 18          	add    $0x18,%rsp
  400975:	c3                   	retq   
  400976:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40097d:	00 00 00 

0000000000400980 <_Z11read_field1P9TestClass>:
  400980:	50                   	push   %rax
  400981:	48 89 3c 24          	mov    %rdi,(%rsp)
  400985:	48 8b 3c 24          	mov    (%rsp),%rdi
  400989:	48 83 c7 04          	add    $0x4,%rdi
  40098d:	e8 ee 01 00 00       	callq  400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>
  400992:	8b 00                	mov    (%rax),%eax
  400994:	59                   	pop    %rcx
  400995:	c3                   	retq   
  400996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40099d:	00 00 00 

00000000004009a0 <_Z21explicit_write_field0m>:
  4009a0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009aa:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  4009b1:	77 77 00 00 
  4009b5:	c3                   	retq   
  4009b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009bd:	00 00 00 

00000000004009c0 <_Z20explicit_read_field0m>:
  4009c0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009c5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  4009ca:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  4009d1:	c3                   	retq   
  4009d2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009d9:	00 00 00 
  4009dc:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009e0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009e0:	50                   	push   %rax
  4009e1:	e8 5a fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009e6:	48 89 c7             	mov    %rax,%rdi
  4009e9:	e8 92 03 00 00       	callq  400d80 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009ee:	58                   	pop    %rax
  4009ef:	c3                   	retq   

00000000004009f0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  4009f0:	50                   	push   %rax
  4009f1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009f5:	e8 06 04 00 00       	callq  400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009fa:	e8 41 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009ff:	b9 01 00 00 00       	mov    $0x1,%ecx
  400a04:	89 ce                	mov    %ecx,%esi
  400a06:	48 89 c7             	mov    %rax,%rdi
  400a09:	e8 22 04 00 00       	callq  400e30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  400a0e:	48 89 c7             	mov    %rax,%rdi
  400a11:	e8 fa 03 00 00       	callq  400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400a16:	59                   	pop    %rcx
  400a17:	c3                   	retq   
  400a18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a1f:	00 

0000000000400a20 <_ZN9TestClassC2Ev>:
  400a20:	48 83 ec 18          	sub    $0x18,%rsp
  400a24:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a29:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400a2e:	48 89 f8             	mov    %rdi,%rax
  400a31:	48 83 c0 04          	add    $0x4,%rax
  400a35:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a3a:	48 89 c7             	mov    %rax,%rdi
  400a3d:	e8 4e 01 00 00       	callq  400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>
  400a42:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a47:	48 83 c0 04          	add    $0x4,%rax
  400a4b:	48 89 c7             	mov    %rax,%rdi
  400a4e:	e8 4d 01 00 00       	callq  400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>
  400a53:	48 83 c4 18          	add    $0x18,%rsp
  400a57:	c3                   	retq   
  400a58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400a5f:	00 

0000000000400a60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400a60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a65:	c3                   	retq   
  400a66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a6d:	00 00 00 

0000000000400a70 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEaSEi>:
  400a70:	48 83 ec 28          	sub    $0x28,%rsp
  400a74:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a79:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a7d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a82:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a86:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a8b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a8f:	e8 7c 01 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a94:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a98:	89 30                	mov    %esi,(%rax)
  400a9a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a9f:	48 83 c4 28          	add    $0x28,%rsp
  400aa3:	c3                   	retq   
  400aa4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aab:	00 00 00 
  400aae:	66 90                	xchg   %ax,%ax

0000000000400ab0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEaSEi>:
  400ab0:	48 83 ec 28          	sub    $0x28,%rsp
  400ab4:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400ab9:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400abd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ac2:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400ac6:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400acb:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400acf:	e8 2c 02 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad4:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400ad8:	89 30                	mov    %esi,(%rax)
  400ada:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400adf:	48 83 c4 28          	add    $0x28,%rsp
  400ae3:	c3                   	retq   
  400ae4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aeb:	00 00 00 
  400aee:	66 90                	xchg   %ax,%ax

0000000000400af0 <_ZN9TestClass15increase_field0Ev>:
  400af0:	48 83 ec 18          	sub    $0x18,%rsp
  400af4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400af9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400afe:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b03:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b08:	48 83 c7 04          	add    $0x4,%rdi
  400b0c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b10:	48 89 c7             	mov    %rax,%rdi
  400b13:	48 8b 34 24          	mov    (%rsp),%rsi
  400b17:	e8 94 00 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>
  400b1c:	48 83 c4 18          	add    $0x18,%rsp
  400b20:	c3                   	retq   
  400b21:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b28:	00 00 00 
  400b2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b30 <_ZN9TestClass15increase_field1Ev>:
  400b30:	48 83 ec 18          	sub    $0x18,%rsp
  400b34:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400b39:	ba 44 44 00 00       	mov    $0x4444,%edx
  400b3e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400b43:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400b48:	48 83 c7 04          	add    $0x4,%rdi
  400b4c:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b50:	48 89 c7             	mov    %rax,%rdi
  400b53:	48 8b 34 24          	mov    (%rsp),%rsi
  400b57:	e8 44 01 00 00       	callq  400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>
  400b5c:	48 83 c4 18          	add    $0x18,%rsp
  400b60:	c3                   	retq   
  400b61:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b68:	00 00 00 
  400b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400b70 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEcvRiEv>:
  400b70:	50                   	push   %rax
  400b71:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b75:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b79:	e8 92 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b7e:	59                   	pop    %rcx
  400b7f:	c3                   	retq   

0000000000400b80 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEcvRiEv>:
  400b80:	50                   	push   %rax
  400b81:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b85:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b89:	e8 72 01 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b8e:	59                   	pop    %rcx
  400b8f:	c3                   	retq   

0000000000400b90 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2Ev>:
  400b90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b95:	c3                   	retq   
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 

0000000000400ba0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2Ev>:
  400ba0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ba5:	c3                   	retq   
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEmLEi>:
  400bb0:	48 83 ec 38          	sub    $0x38,%rsp
  400bb4:	48 89 f8             	mov    %rdi,%rax
  400bb7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400bbc:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400bc0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400bc5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400bc9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400bce:	48 89 f7             	mov    %rsi,%rdi
  400bd1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400bd6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400bda:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400bdf:	e8 2c 00 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400be8:	0f af 10             	imul   (%rax),%edx
  400beb:	89 10                	mov    %edx,(%rax)
  400bed:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400bf2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400bf7:	e8 24 00 00 00       	callq  400c20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>
  400bfc:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c01:	48 83 c4 38          	add    $0x38,%rsp
  400c05:	c3                   	retq   
  400c06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c0d:	00 00 00 

0000000000400c10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c10:	50                   	push   %rax
  400c11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c19:	e8 12 00 00 00       	callq  400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c1e:	59                   	pop    %rcx
  400c1f:	c3                   	retq   

0000000000400c20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassEC2ERKS3_>:
  400c20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c25:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c2a:	c3                   	retq   
  400c2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c30 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c30:	48 83 ec 28          	sub    $0x28,%rsp
  400c34:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c39:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c3e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400c43:	e8 f8 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400c48:	48 89 c7             	mov    %rax,%rdi
  400c4b:	e8 40 00 00 00       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c50:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400c55:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c5a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400c5f:	48 83 e8 04          	sub    $0x4,%rax
  400c63:	48 c1 e8 02          	shr    $0x2,%rax
  400c67:	48 c1 e0 02          	shl    $0x2,%rax
  400c6b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400c70:	48 83 c0 00          	add    $0x0,%rax
  400c74:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400c79:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c7e:	48 83 c4 28          	add    $0x28,%rsp
  400c82:	c3                   	retq   
  400c83:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c8a:	00 00 00 
  400c8d:	0f 1f 00             	nopl   (%rax)

0000000000400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c90:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c95:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c9a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c9e:	c3                   	retq   
  400c9f:	90                   	nop

0000000000400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEmLEi>:
  400ca0:	48 83 ec 38          	sub    $0x38,%rsp
  400ca4:	48 89 f8             	mov    %rdi,%rax
  400ca7:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400cac:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400cb0:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400cb5:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400cb9:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400cbe:	48 89 f7             	mov    %rsi,%rdi
  400cc1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400cc6:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400cca:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400ccf:	e8 2c 00 00 00       	callq  400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400cd4:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400cd8:	0f af 10             	imul   (%rax),%edx
  400cdb:	89 10                	mov    %edx,(%rax)
  400cdd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400ce2:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400ce7:	e8 24 00 00 00       	callq  400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>
  400cec:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400cf1:	48 83 c4 38          	add    $0x38,%rsp
  400cf5:	c3                   	retq   
  400cf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400cfd:	00 00 00 

0000000000400d00 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d00:	50                   	push   %rax
  400d01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d09:	e8 12 00 00 00       	callq  400d20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400d0e:	59                   	pop    %rcx
  400d0f:	c3                   	retq   

0000000000400d10 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassEC2ERKS3_>:
  400d10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d15:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d1a:	c3                   	retq   
  400d1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d20:	48 83 ec 28          	sub    $0x28,%rsp
  400d24:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400d29:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400d2e:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400d33:	e8 08 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400d38:	48 89 c7             	mov    %rax,%rdi
  400d3b:	e8 50 ff ff ff       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400d40:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d45:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d4a:	48 2b 44 24 10       	sub    0x10(%rsp),%rax
  400d4f:	48 83 e8 04          	sub    $0x4,%rax
  400d53:	48 c1 e8 02          	shr    $0x2,%rax
  400d57:	48 c1 e0 02          	shl    $0x2,%rax
  400d5b:	48 03 44 24 10       	add    0x10(%rsp),%rax
  400d60:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400d66:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400d6b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d70:	48 83 c4 28          	add    $0x28,%rsp
  400d74:	c3                   	retq   
  400d75:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d7c:	00 00 00 
  400d7f:	90                   	nop

0000000000400d80 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d80:	48 83 ec 18          	sub    $0x18,%rsp
  400d84:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d89:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d8e:	48 89 f8             	mov    %rdi,%rax
  400d91:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d96:	48 89 c7             	mov    %rax,%rdi
  400d99:	e8 32 00 00 00       	callq  400dd0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d9e:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400da3:	89 cf                	mov    %ecx,%edi
  400da5:	e8 46 f8 ff ff       	callq  4005f0 <malloc@plt>
  400daa:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400daf:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400db3:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400dba:	00 
  400dbb:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400dc2:	00 
  400dc3:	48 83 c4 18          	add    $0x18,%rsp
  400dc7:	c3                   	retq   
  400dc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dcf:	00 

0000000000400dd0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400dd0:	50                   	push   %rax
  400dd1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400dd5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400dd9:	e8 02 00 00 00       	callq  400de0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400dde:	58                   	pop    %rax
  400ddf:	c3                   	retq   

0000000000400de0 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400de0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400de5:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400dea:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400df1:	c3                   	retq   
  400df2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400df9:	00 00 00 
  400dfc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400e00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400e00:	c3                   	retq   
  400e01:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e08:	00 00 00 
  400e0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400e10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400e10:	50                   	push   %rax
  400e11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400e15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400e19:	48 83 c7 01          	add    $0x1,%rdi
  400e1d:	e8 3e 00 00 00       	callq  400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400e22:	59                   	pop    %rcx
  400e23:	c3                   	retq   
  400e24:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e2b:	00 00 00 
  400e2e:	66 90                	xchg   %ax,%ax

0000000000400e30 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400e30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e35:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400e3a:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400e3f:	48 8b 3e             	mov    (%rsi),%rdi
  400e42:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400e47:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400e4c:	48 03 3e             	add    (%rsi),%rdi
  400e4f:	48 89 3e             	mov    %rdi,(%rsi)
  400e52:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e57:	c3                   	retq   
  400e58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e5f:	00 

0000000000400e60 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e60:	48 83 ec 18          	sub    $0x18,%rsp
  400e64:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400e69:	e8 d2 f8 ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400e6e:	48 89 c7             	mov    %rax,%rdi
  400e71:	e8 1a fe ff ff       	callq  400c90 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e76:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400e7b:	48 c1 e7 02          	shl    $0x2,%rdi
  400e7f:	48 01 f8             	add    %rdi,%rax
  400e82:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400e87:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e8c:	48 83 c4 18          	add    $0x18,%rsp
  400e90:	c3                   	retq   
  400e91:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e98:	00 00 00 
  400e9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400ea0 <__libc_csu_init>:
  400ea0:	41 57                	push   %r15
  400ea2:	41 56                	push   %r14
  400ea4:	41 89 ff             	mov    %edi,%r15d
  400ea7:	41 55                	push   %r13
  400ea9:	41 54                	push   %r12
  400eab:	4c 8d 25 2e 0f 20 00 	lea    0x200f2e(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400eb2:	55                   	push   %rbp
  400eb3:	48 8d 2d 2e 0f 20 00 	lea    0x200f2e(%rip),%rbp        # 601de8 <__init_array_end>
  400eba:	53                   	push   %rbx
  400ebb:	49 89 f6             	mov    %rsi,%r14
  400ebe:	49 89 d5             	mov    %rdx,%r13
  400ec1:	4c 29 e5             	sub    %r12,%rbp
  400ec4:	48 83 ec 08          	sub    $0x8,%rsp
  400ec8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ecc:	e8 d7 f6 ff ff       	callq  4005a8 <_init>
  400ed1:	48 85 ed             	test   %rbp,%rbp
  400ed4:	74 20                	je     400ef6 <__libc_csu_init+0x56>
  400ed6:	31 db                	xor    %ebx,%ebx
  400ed8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400edf:	00 
  400ee0:	4c 89 ea             	mov    %r13,%rdx
  400ee3:	4c 89 f6             	mov    %r14,%rsi
  400ee6:	44 89 ff             	mov    %r15d,%edi
  400ee9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400eed:	48 83 c3 01          	add    $0x1,%rbx
  400ef1:	48 39 eb             	cmp    %rbp,%rbx
  400ef4:	75 ea                	jne    400ee0 <__libc_csu_init+0x40>
  400ef6:	48 83 c4 08          	add    $0x8,%rsp
  400efa:	5b                   	pop    %rbx
  400efb:	5d                   	pop    %rbp
  400efc:	41 5c                	pop    %r12
  400efe:	41 5d                	pop    %r13
  400f00:	41 5e                	pop    %r14
  400f02:	41 5f                	pop    %r15
  400f04:	c3                   	retq   
  400f05:	90                   	nop
  400f06:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f0d:	00 00 00 

0000000000400f10 <__libc_csu_fini>:
  400f10:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f14 <_fini>:
  400f14:	48 83 ec 08          	sub    $0x8,%rsp
  400f18:	48 83 c4 08          	add    $0x8,%rsp
  400f1c:	c3                   	retq   
