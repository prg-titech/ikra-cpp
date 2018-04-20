
bin/g++_-O0_DynamicStorage_4_Soa:     file format elf64-x86-64


Disassembly of section .init:

00000000004005f8 <_init>:
  4005f8:	48 83 ec 08          	sub    $0x8,%rsp
  4005fc:	48 8b 05 f5 19 20 00 	mov    0x2019f5(%rip),%rax        # 601ff8 <_DYNAMIC+0x1f0>
  400603:	48 85 c0             	test   %rax,%rax
  400606:	74 05                	je     40060d <_init+0x15>
  400608:	e8 83 00 00 00       	callq  400690 <_Unwind_Resume@plt+0x10>
  40060d:	48 83 c4 08          	add    $0x8,%rsp
  400611:	c3                   	retq   

Disassembly of section .plt:

0000000000400620 <printf@plt-0x10>:
  400620:	ff 35 e2 19 20 00    	pushq  0x2019e2(%rip)        # 602008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400626:	ff 25 e4 19 20 00    	jmpq   *0x2019e4(%rip)        # 602010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40062c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400630 <printf@plt>:
  400630:	ff 25 e2 19 20 00    	jmpq   *0x2019e2(%rip)        # 602018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400636:	68 00 00 00 00       	pushq  $0x0
  40063b:	e9 e0 ff ff ff       	jmpq   400620 <_init+0x28>

0000000000400640 <malloc@plt>:
  400640:	ff 25 da 19 20 00    	jmpq   *0x2019da(%rip)        # 602020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400646:	68 01 00 00 00       	pushq  $0x1
  40064b:	e9 d0 ff ff ff       	jmpq   400620 <_init+0x28>

0000000000400650 <__libc_start_main@plt>:
  400650:	ff 25 d2 19 20 00    	jmpq   *0x2019d2(%rip)        # 602028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400656:	68 02 00 00 00       	pushq  $0x2
  40065b:	e9 c0 ff ff ff       	jmpq   400620 <_init+0x28>

0000000000400660 <__stack_chk_fail@plt>:
  400660:	ff 25 ca 19 20 00    	jmpq   *0x2019ca(%rip)        # 602030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400666:	68 03 00 00 00       	pushq  $0x3
  40066b:	e9 b0 ff ff ff       	jmpq   400620 <_init+0x28>

0000000000400670 <__gxx_personality_v0@plt>:
  400670:	ff 25 c2 19 20 00    	jmpq   *0x2019c2(%rip)        # 602038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400676:	68 04 00 00 00       	pushq  $0x4
  40067b:	e9 a0 ff ff ff       	jmpq   400620 <_init+0x28>

0000000000400680 <_Unwind_Resume@plt>:
  400680:	ff 25 ba 19 20 00    	jmpq   *0x2019ba(%rip)        # 602040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400686:	68 05 00 00 00       	pushq  $0x5
  40068b:	e9 90 ff ff ff       	jmpq   400620 <_init+0x28>

Disassembly of section .plt.got:

0000000000400690 <.plt.got>:
  400690:	ff 25 62 19 20 00    	jmpq   *0x201962(%rip)        # 601ff8 <_DYNAMIC+0x1f0>
  400696:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000004006a0 <_start>:
  4006a0:	31 ed                	xor    %ebp,%ebp
  4006a2:	49 89 d1             	mov    %rdx,%r9
  4006a5:	5e                   	pop    %rsi
  4006a6:	48 89 e2             	mov    %rsp,%rdx
  4006a9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4006ad:	50                   	push   %rax
  4006ae:	54                   	push   %rsp
  4006af:	49 c7 c0 80 0f 40 00 	mov    $0x400f80,%r8
  4006b6:	48 c7 c1 10 0f 40 00 	mov    $0x400f10,%rcx
  4006bd:	48 c7 c7 96 07 40 00 	mov    $0x400796,%rdi
  4006c4:	e8 87 ff ff ff       	callq  400650 <__libc_start_main@plt>
  4006c9:	f4                   	hlt    
  4006ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004006d0 <deregister_tm_clones>:
  4006d0:	b8 5f 20 60 00       	mov    $0x60205f,%eax
  4006d5:	55                   	push   %rbp
  4006d6:	48 2d 58 20 60 00    	sub    $0x602058,%rax
  4006dc:	48 83 f8 0e          	cmp    $0xe,%rax
  4006e0:	48 89 e5             	mov    %rsp,%rbp
  4006e3:	76 1b                	jbe    400700 <deregister_tm_clones+0x30>
  4006e5:	b8 00 00 00 00       	mov    $0x0,%eax
  4006ea:	48 85 c0             	test   %rax,%rax
  4006ed:	74 11                	je     400700 <deregister_tm_clones+0x30>
  4006ef:	5d                   	pop    %rbp
  4006f0:	bf 58 20 60 00       	mov    $0x602058,%edi
  4006f5:	ff e0                	jmpq   *%rax
  4006f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4006fe:	00 00 
  400700:	5d                   	pop    %rbp
  400701:	c3                   	retq   
  400702:	0f 1f 40 00          	nopl   0x0(%rax)
  400706:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40070d:	00 00 00 

0000000000400710 <register_tm_clones>:
  400710:	be 58 20 60 00       	mov    $0x602058,%esi
  400715:	55                   	push   %rbp
  400716:	48 81 ee 58 20 60 00 	sub    $0x602058,%rsi
  40071d:	48 c1 fe 03          	sar    $0x3,%rsi
  400721:	48 89 e5             	mov    %rsp,%rbp
  400724:	48 89 f0             	mov    %rsi,%rax
  400727:	48 c1 e8 3f          	shr    $0x3f,%rax
  40072b:	48 01 c6             	add    %rax,%rsi
  40072e:	48 d1 fe             	sar    %rsi
  400731:	74 15                	je     400748 <register_tm_clones+0x38>
  400733:	b8 00 00 00 00       	mov    $0x0,%eax
  400738:	48 85 c0             	test   %rax,%rax
  40073b:	74 0b                	je     400748 <register_tm_clones+0x38>
  40073d:	5d                   	pop    %rbp
  40073e:	bf 58 20 60 00       	mov    $0x602058,%edi
  400743:	ff e0                	jmpq   *%rax
  400745:	0f 1f 00             	nopl   (%rax)
  400748:	5d                   	pop    %rbp
  400749:	c3                   	retq   
  40074a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400750 <__do_global_dtors_aux>:
  400750:	80 3d 09 19 20 00 00 	cmpb   $0x0,0x201909(%rip)        # 602060 <completed.7594>
  400757:	75 11                	jne    40076a <__do_global_dtors_aux+0x1a>
  400759:	55                   	push   %rbp
  40075a:	48 89 e5             	mov    %rsp,%rbp
  40075d:	e8 6e ff ff ff       	callq  4006d0 <deregister_tm_clones>
  400762:	5d                   	pop    %rbp
  400763:	c6 05 f6 18 20 00 01 	movb   $0x1,0x2018f6(%rip)        # 602060 <completed.7594>
  40076a:	f3 c3                	repz retq 
  40076c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400770 <frame_dummy>:
  400770:	bf 00 1e 60 00       	mov    $0x601e00,%edi
  400775:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400779:	75 05                	jne    400780 <frame_dummy+0x10>
  40077b:	eb 93                	jmp    400710 <register_tm_clones>
  40077d:	0f 1f 00             	nopl   (%rax)
  400780:	b8 00 00 00 00       	mov    $0x0,%eax
  400785:	48 85 c0             	test   %rax,%rax
  400788:	74 f1                	je     40077b <frame_dummy+0xb>
  40078a:	55                   	push   %rbp
  40078b:	48 89 e5             	mov    %rsp,%rbp
  40078e:	ff d0                	callq  *%rax
  400790:	5d                   	pop    %rbp
  400791:	e9 7a ff ff ff       	jmpq   400710 <register_tm_clones>

0000000000400796 <main>:
  400796:	53                   	push   %rbx
  400797:	48 83 ec 10          	sub    $0x10,%rsp
  40079b:	e8 0d 04 00 00       	callq  400bad <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_>
  4007a0:	bf 04 00 00 00       	mov    $0x4,%edi
  4007a5:	e8 4d 04 00 00       	callq  400bf7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>
  4007aa:	48 89 c3             	mov    %rax,%rbx
  4007ad:	b8 00 00 00 00       	mov    $0x0,%eax
  4007b2:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4007b6:	48 83 f8 03          	cmp    $0x3,%rax
  4007ba:	74 06                	je     4007c2 <main+0x2c>
  4007bc:	48 83 c0 01          	add    $0x1,%rax
  4007c0:	eb f0                	jmp    4007b2 <main+0x1c>
  4007c2:	48 89 df             	mov    %rbx,%rdi
  4007c5:	e8 08 03 00 00       	callq  400ad2 <_ZN9TestClassC1Ev>
  4007ca:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4007cf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007d4:	48 83 c0 04          	add    $0x4,%rax
  4007d8:	be 77 77 00 00       	mov    $0x7777,%esi
  4007dd:	48 89 c7             	mov    %rax,%rdi
  4007e0:	e8 55 04 00 00       	callq  400c3a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>
  4007e5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ea:	48 83 c0 04          	add    $0x4,%rax
  4007ee:	be 88 88 00 00       	mov    $0x8888,%esi
  4007f3:	48 89 c7             	mov    %rax,%rdi
  4007f6:	e8 6d 04 00 00       	callq  400c68 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>
  4007fb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400800:	48 89 c7             	mov    %rax,%rdi
  400803:	e8 d0 01 00 00       	callq  4009d8 <_ZN9TestClass15increase_field0Ev>
  400808:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40080d:	48 89 c7             	mov    %rax,%rdi
  400810:	e8 2b 02 00 00       	callq  400a40 <_ZN9TestClass15increase_field1Ev>
  400815:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081a:	48 83 c0 04          	add    $0x4,%rax
  40081e:	48 89 c7             	mov    %rax,%rdi
  400821:	e8 8c 04 00 00       	callq  400cb2 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
  400826:	8b 18                	mov    (%rax),%ebx
  400828:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40082d:	48 83 c0 04          	add    $0x4,%rax
  400831:	48 89 c7             	mov    %rax,%rdi
  400834:	e8 5d 04 00 00       	callq  400c96 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
  400839:	8b 00                	mov    (%rax),%eax
  40083b:	89 da                	mov    %ebx,%edx
  40083d:	89 c6                	mov    %eax,%esi
  40083f:	bf b4 0f 40 00       	mov    $0x400fb4,%edi
  400844:	b8 00 00 00 00       	mov    $0x0,%eax
  400849:	e8 e2 fd ff ff       	callq  400630 <printf@plt>
  40084e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400853:	48 83 c0 04          	add    $0x4,%rax
  400857:	48 89 c7             	mov    %rax,%rdi
  40085a:	e8 37 04 00 00       	callq  400c96 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
  40085f:	8b 00                	mov    (%rax),%eax
  400861:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400866:	75 1a                	jne    400882 <main+0xec>
  400868:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40086d:	48 83 c0 04          	add    $0x4,%rax
  400871:	48 89 c7             	mov    %rax,%rdi
  400874:	e8 39 04 00 00       	callq  400cb2 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
  400879:	8b 00                	mov    (%rax),%eax
  40087b:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400880:	74 07                	je     400889 <main+0xf3>
  400882:	b8 01 00 00 00       	mov    $0x1,%eax
  400887:	eb 05                	jmp    40088e <main+0xf8>
  400889:	b8 00 00 00 00       	mov    $0x0,%eax
  40088e:	0f b6 c0             	movzbl %al,%eax
  400891:	48 83 c4 10          	add    $0x10,%rsp
  400895:	5b                   	pop    %rbx
  400896:	c3                   	retq   

0000000000400897 <_Z12new_instancev>:
  400897:	53                   	push   %rbx
  400898:	bf 04 00 00 00       	mov    $0x4,%edi
  40089d:	e8 55 03 00 00       	callq  400bf7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>
  4008a2:	48 89 c3             	mov    %rax,%rbx
  4008a5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008aa:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4008ae:	48 83 f8 03          	cmp    $0x3,%rax
  4008b2:	74 06                	je     4008ba <_Z12new_instancev+0x23>
  4008b4:	48 83 c0 01          	add    $0x1,%rax
  4008b8:	eb f0                	jmp    4008aa <_Z12new_instancev+0x13>
  4008ba:	48 89 df             	mov    %rbx,%rdi
  4008bd:	e8 10 02 00 00       	callq  400ad2 <_ZN9TestClassC1Ev>
  4008c2:	48 89 d8             	mov    %rbx,%rax
  4008c5:	5b                   	pop    %rbx
  4008c6:	c3                   	retq   

00000000004008c7 <_Z12write_field0P9TestClass>:
  4008c7:	48 83 ec 18          	sub    $0x18,%rsp
  4008cb:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008d0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008d5:	48 83 c0 04          	add    $0x4,%rax
  4008d9:	be 77 77 00 00       	mov    $0x7777,%esi
  4008de:	48 89 c7             	mov    %rax,%rdi
  4008e1:	e8 54 03 00 00       	callq  400c3a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>
  4008e6:	90                   	nop
  4008e7:	48 83 c4 18          	add    $0x18,%rsp
  4008eb:	c3                   	retq   

00000000004008ec <_Z11read_field0P9TestClass>:
  4008ec:	48 83 ec 18          	sub    $0x18,%rsp
  4008f0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008f5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008fa:	48 83 c0 04          	add    $0x4,%rax
  4008fe:	48 89 c7             	mov    %rax,%rdi
  400901:	e8 90 03 00 00       	callq  400c96 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>
  400906:	8b 00                	mov    (%rax),%eax
  400908:	48 83 c4 18          	add    $0x18,%rsp
  40090c:	c3                   	retq   

000000000040090d <_Z12write_field1P9TestClass>:
  40090d:	48 83 ec 18          	sub    $0x18,%rsp
  400911:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400916:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40091b:	48 83 c0 04          	add    $0x4,%rax
  40091f:	be 77 77 00 00       	mov    $0x7777,%esi
  400924:	48 89 c7             	mov    %rax,%rdi
  400927:	e8 3c 03 00 00       	callq  400c68 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>
  40092c:	90                   	nop
  40092d:	48 83 c4 18          	add    $0x18,%rsp
  400931:	c3                   	retq   

0000000000400932 <_Z11read_field1P9TestClass>:
  400932:	48 83 ec 18          	sub    $0x18,%rsp
  400936:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40093b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400940:	48 83 c0 04          	add    $0x4,%rax
  400944:	48 89 c7             	mov    %rax,%rdi
  400947:	e8 66 03 00 00       	callq  400cb2 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>
  40094c:	8b 00                	mov    (%rax),%eax
  40094e:	48 83 c4 18          	add    $0x18,%rsp
  400952:	c3                   	retq   

0000000000400953 <_Z21explicit_write_field0m>:
  400953:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400958:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40095d:	c7 04 85 40 b2 60 00 	movl   $0x7777,0x60b240(,%rax,4)
  400964:	77 77 00 00 
  400968:	90                   	nop
  400969:	c3                   	retq   

000000000040096a <_Z20explicit_read_field0m>:
  40096a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40096f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400974:	8b 04 85 40 b2 60 00 	mov    0x60b240(,%rax,4),%eax
  40097b:	c3                   	retq   

000000000040097c <_Z25explicit_write_field0_aosP16TestClassCompare>:
  40097c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400981:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400986:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  40098c:	90                   	nop
  40098d:	c3                   	retq   

000000000040098e <_Z24explicit_read_field0_aosP16TestClassCompare>:
  40098e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400993:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400998:	8b 00                	mov    (%rax),%eax
  40099a:	c3                   	retq   

000000000040099b <_Z25explicit_write_field1_aosP16TestClassCompare>:
  40099b:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009a0:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4009a5:	c7 40 04 77 77 00 00 	movl   $0x7777,0x4(%rax)
  4009ac:	90                   	nop
  4009ad:	c3                   	retq   

00000000004009ae <_Z24explicit_read_field1_aosP16TestClassCompare>:
  4009ae:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009b3:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4009b8:	8b 40 04             	mov    0x4(%rax),%eax
  4009bb:	c3                   	retq   

00000000004009bc <_ZnwmPv>:
  4009bc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009c1:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4009c6:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  4009cb:	c3                   	retq   

00000000004009cc <_ZdlPvS_>:
  4009cc:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  4009d1:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  4009d6:	90                   	nop
  4009d7:	c3                   	retq   

00000000004009d8 <_ZN9TestClass15increase_field0Ev>:
  4009d8:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4009dd:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009e1:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009e5:	55                   	push   %rbp
  4009e6:	48 89 e5             	mov    %rsp,%rbp
  4009e9:	41 52                	push   %r10
  4009eb:	48 83 ec 48          	sub    $0x48,%rsp
  4009ef:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009f3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009fa:	00 00 
  4009fc:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a00:	31 c0                	xor    %eax,%eax
  400a02:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400a06:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400a0a:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400a0e:	ba 55 55 00 00       	mov    $0x5555,%edx
  400a13:	48 89 ce             	mov    %rcx,%rsi
  400a16:	48 89 c7             	mov    %rax,%rdi
  400a19:	e8 f2 00 00 00       	callq  400b10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEmLEi>
  400a1e:	90                   	nop
  400a1f:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a23:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a2a:	00 00 
  400a2c:	74 05                	je     400a33 <_ZN9TestClass15increase_field0Ev+0x5b>
  400a2e:	e8 2d fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  400a33:	48 83 c4 48          	add    $0x48,%rsp
  400a37:	41 5a                	pop    %r10
  400a39:	5d                   	pop    %rbp
  400a3a:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a3e:	c3                   	retq   
  400a3f:	90                   	nop

0000000000400a40 <_ZN9TestClass15increase_field1Ev>:
  400a40:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400a45:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400a49:	41 ff 72 f8          	pushq  -0x8(%r10)
  400a4d:	55                   	push   %rbp
  400a4e:	48 89 e5             	mov    %rsp,%rbp
  400a51:	41 52                	push   %r10
  400a53:	48 83 ec 48          	sub    $0x48,%rsp
  400a57:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400a5b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400a62:	00 00 
  400a64:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a68:	31 c0                	xor    %eax,%eax
  400a6a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400a6e:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400a72:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400a76:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a7b:	48 89 ce             	mov    %rcx,%rsi
  400a7e:	48 89 c7             	mov    %rax,%rdi
  400a81:	e8 d2 00 00 00       	callq  400b58 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEmLEi>
  400a86:	90                   	nop
  400a87:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a8b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a92:	00 00 
  400a94:	74 05                	je     400a9b <_ZN9TestClass15increase_field1Ev+0x5b>
  400a96:	e8 c5 fb ff ff       	callq  400660 <__stack_chk_fail@plt>
  400a9b:	48 83 c4 48          	add    $0x48,%rsp
  400a9f:	41 5a                	pop    %r10
  400aa1:	5d                   	pop    %rbp
  400aa2:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400aa6:	c3                   	retq   
  400aa7:	90                   	nop

0000000000400aa8 <_ZN9TestClass14storage_bufferEv>:
  400aa8:	b8 80 20 60 00       	mov    $0x602080,%eax
  400aad:	c3                   	retq   

0000000000400aae <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400aae:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ab3:	90                   	nop
  400ab4:	c3                   	retq   
  400ab5:	90                   	nop

0000000000400ab6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEC1Ev>:
  400ab6:	48 83 ec 18          	sub    $0x18,%rsp
  400aba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400abf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ac4:	48 89 c7             	mov    %rax,%rdi
  400ac7:	e8 e2 ff ff ff       	callq  400aae <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  400acc:	90                   	nop
  400acd:	48 83 c4 18          	add    $0x18,%rsp
  400ad1:	c3                   	retq   

0000000000400ad2 <_ZN9TestClassC1Ev>:
  400ad2:	48 83 ec 18          	sub    $0x18,%rsp
  400ad6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400adb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ae0:	48 89 c7             	mov    %rax,%rdi
  400ae3:	e8 ce ff ff ff       	callq  400ab6 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEC1Ev>
  400ae8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aed:	48 83 c0 04          	add    $0x4,%rax
  400af1:	48 89 c7             	mov    %rax,%rdi
  400af4:	e8 2b 01 00 00       	callq  400c24 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC1Ev>
  400af9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400afe:	48 83 c0 04          	add    $0x4,%rax
  400b02:	48 89 c7             	mov    %rax,%rdi
  400b05:	e8 22 01 00 00       	callq  400c2c <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC1Ev>
  400b0a:	90                   	nop
  400b0b:	48 83 c4 18          	add    $0x18,%rsp
  400b0f:	c3                   	retq   

0000000000400b10 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEmLEi>:
  400b10:	48 83 ec 28          	sub    $0x28,%rsp
  400b14:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400b19:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400b1e:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400b22:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b27:	48 89 c7             	mov    %rax,%rdi
  400b2a:	e8 9f 01 00 00       	callq  400cce <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400b2f:	8b 10                	mov    (%rax),%edx
  400b31:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400b36:	89 10                	mov    %edx,(%rax)
  400b38:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400b3d:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b42:	48 89 d6             	mov    %rdx,%rsi
  400b45:	48 89 c7             	mov    %rax,%rdi
  400b48:	e8 9d 01 00 00       	callq  400cea <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC1ERKS3_>
  400b4d:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b52:	48 83 c4 28          	add    $0x28,%rsp
  400b56:	c3                   	retq   
  400b57:	90                   	nop

0000000000400b58 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEmLEi>:
  400b58:	48 83 ec 28          	sub    $0x28,%rsp
  400b5c:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400b61:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400b66:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400b6a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b6f:	48 89 c7             	mov    %rax,%rdi
  400b72:	e8 7f 01 00 00       	callq  400cf6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400b77:	8b 10                	mov    (%rax),%edx
  400b79:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400b7e:	89 10                	mov    %edx,(%rax)
  400b80:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400b85:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b8a:	48 89 d6             	mov    %rdx,%rsi
  400b8d:	48 89 c7             	mov    %rax,%rdi
  400b90:	e8 7d 01 00 00       	callq  400d12 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC1ERKS3_>
  400b95:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b9a:	48 83 c4 28          	add    $0x28,%rsp
  400b9e:	c3                   	retq   

0000000000400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>:
  400b9f:	48 83 ec 08          	sub    $0x8,%rsp
  400ba3:	e8 00 ff ff ff       	callq  400aa8 <_ZN9TestClass14storage_bufferEv>
  400ba8:	48 83 c4 08          	add    $0x8,%rsp
  400bac:	c3                   	retq   

0000000000400bad <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_>:
  400bad:	41 54                	push   %r12
  400baf:	55                   	push   %rbp
  400bb0:	53                   	push   %rbx
  400bb1:	e8 e9 ff ff ff       	callq  400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400bb6:	48 89 c5             	mov    %rax,%rbp
  400bb9:	48 89 ee             	mov    %rbp,%rsi
  400bbc:	bf 20 00 00 00       	mov    $0x20,%edi
  400bc1:	e8 f6 fd ff ff       	callq  4009bc <_ZnwmPv>
  400bc6:	48 89 c3             	mov    %rax,%rbx
  400bc9:	48 85 db             	test   %rbx,%rbx
  400bcc:	74 23                	je     400bf1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400bce:	48 89 df             	mov    %rbx,%rdi
  400bd1:	e8 48 01 00 00       	callq  400d1e <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400bd6:	eb 19                	jmp    400bf1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400bd8:	49 89 c4             	mov    %rax,%r12
  400bdb:	48 89 ee             	mov    %rbp,%rsi
  400bde:	48 89 df             	mov    %rbx,%rdi
  400be1:	e8 e6 fd ff ff       	callq  4009cc <_ZdlPvS_>
  400be6:	4c 89 e0             	mov    %r12,%rax
  400be9:	48 89 c7             	mov    %rax,%rdi
  400bec:	e8 8f fa ff ff       	callq  400680 <_Unwind_Resume@plt>
  400bf1:	90                   	nop
  400bf2:	5b                   	pop    %rbx
  400bf3:	5d                   	pop    %rbp
  400bf4:	41 5c                	pop    %r12
  400bf6:	c3                   	retq   

0000000000400bf7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEnwEm>:
  400bf7:	48 83 ec 18          	sub    $0x18,%rsp
  400bfb:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c00:	e8 64 01 00 00       	callq  400d69 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400c05:	e8 95 ff ff ff       	callq  400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400c0a:	be 01 00 00 00       	mov    $0x1,%esi
  400c0f:	48 89 c7             	mov    %rax,%rdi
  400c12:	e8 73 01 00 00       	callq  400d8a <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400c17:	48 89 c7             	mov    %rax,%rdi
  400c1a:	e8 4c 01 00 00       	callq  400d6b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE3getEm>
  400c1f:	48 83 c4 18          	add    $0x18,%rsp
  400c23:	c3                   	retq   

0000000000400c24 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC1Ev>:
  400c24:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c29:	90                   	nop
  400c2a:	c3                   	retq   
  400c2b:	90                   	nop

0000000000400c2c <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC1Ev>:
  400c2c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c31:	90                   	nop
  400c32:	c3                   	retq   

0000000000400c33 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EEdlEPv>:
  400c33:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c38:	90                   	nop
  400c39:	c3                   	retq   

0000000000400c3a <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEaSEi>:
  400c3a:	48 83 ec 18          	sub    $0x18,%rsp
  400c3e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c43:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c47:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c4c:	48 89 c7             	mov    %rax,%rdi
  400c4f:	e8 7a 00 00 00       	callq  400cce <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400c54:	48 89 c2             	mov    %rax,%rdx
  400c57:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c5b:	89 02                	mov    %eax,(%rdx)
  400c5d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c62:	48 83 c4 18          	add    $0x18,%rsp
  400c66:	c3                   	retq   
  400c67:	90                   	nop

0000000000400c68 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEaSEi>:
  400c68:	48 83 ec 18          	sub    $0x18,%rsp
  400c6c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c71:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c75:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c7a:	48 89 c7             	mov    %rax,%rdi
  400c7d:	e8 74 00 00 00       	callq  400cf6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400c82:	48 89 c2             	mov    %rax,%rdx
  400c85:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c89:	89 02                	mov    %eax,(%rdx)
  400c8b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c90:	48 83 c4 18          	add    $0x18,%rsp
  400c94:	c3                   	retq   
  400c95:	90                   	nop

0000000000400c96 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEcvRiEv>:
  400c96:	48 83 ec 18          	sub    $0x18,%rsp
  400c9a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c9f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca4:	48 89 c7             	mov    %rax,%rdi
  400ca7:	e8 22 00 00 00       	callq  400cce <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400cac:	48 83 c4 18          	add    $0x18,%rsp
  400cb0:	c3                   	retq   
  400cb1:	90                   	nop

0000000000400cb2 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEcvRiEv>:
  400cb2:	48 83 ec 18          	sub    $0x18,%rsp
  400cb6:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cbb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc0:	48 89 c7             	mov    %rax,%rdi
  400cc3:	e8 2e 00 00 00       	callq  400cf6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>
  400cc8:	48 83 c4 18          	add    $0x18,%rsp
  400ccc:	c3                   	retq   
  400ccd:	90                   	nop

0000000000400cce <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE8data_ptrEv>:
  400cce:	48 83 ec 18          	sub    $0x18,%rsp
  400cd2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cd7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cdc:	48 89 c7             	mov    %rax,%rdi
  400cdf:	e8 e0 00 00 00       	callq  400dc4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400ce4:	48 83 c4 18          	add    $0x18,%rsp
  400ce8:	c3                   	retq   
  400ce9:	90                   	nop

0000000000400cea <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassEC1ERKS3_>:
  400cea:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cef:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cf4:	90                   	nop
  400cf5:	c3                   	retq   

0000000000400cf6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE8data_ptrEv>:
  400cf6:	48 83 ec 18          	sub    $0x18,%rsp
  400cfa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cff:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d04:	48 89 c7             	mov    %rax,%rdi
  400d07:	e8 0e 01 00 00       	callq  400e1a <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400d0c:	48 83 c4 18          	add    $0x18,%rsp
  400d10:	c3                   	retq   
  400d11:	90                   	nop

0000000000400d12 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassEC1ERKS3_>:
  400d12:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d17:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400d1c:	90                   	nop
  400d1d:	c3                   	retq   

0000000000400d1e <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400d1e:	48 83 ec 18          	sub    $0x18,%rsp
  400d22:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d27:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d2c:	48 89 c7             	mov    %rax,%rdi
  400d2f:	e8 42 01 00 00       	callq  400e76 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400d34:	bf c0 91 00 00       	mov    $0x91c0,%edi
  400d39:	e8 02 f9 ff ff       	callq  400640 <malloc@plt>
  400d3e:	48 89 c2             	mov    %rax,%rdx
  400d41:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d46:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400d4a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d4f:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
  400d56:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d5b:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d62:	00 
  400d63:	90                   	nop
  400d64:	48 83 c4 18          	add    $0x18,%rsp
  400d68:	c3                   	retq   

0000000000400d69 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d69:	90                   	nop
  400d6a:	c3                   	retq   

0000000000400d6b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE3getEm>:
  400d6b:	48 83 ec 18          	sub    $0x18,%rsp
  400d6f:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d74:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d79:	48 83 c0 01          	add    $0x1,%rax
  400d7d:	48 89 c7             	mov    %rax,%rdi
  400d80:	e8 04 01 00 00       	callq  400e89 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d85:	48 83 c4 18          	add    $0x18,%rsp
  400d89:	c3                   	retq   

0000000000400d8a <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d8a:	48 83 ec 28          	sub    $0x28,%rsp
  400d8e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d93:	48 89 34 24          	mov    %rsi,(%rsp)
  400d97:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d9c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400da1:	48 8b 04 24          	mov    (%rsp),%rax
  400da5:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400daa:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  400daf:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400db4:	48 89 d6             	mov    %rdx,%rsi
  400db7:	48 89 c7             	mov    %rax,%rdi
  400dba:	e8 fe 00 00 00       	callq  400ebd <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400dbf:	48 83 c4 28          	add    $0x28,%rsp
  400dc3:	c3                   	retq   

0000000000400dc4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400dc4:	48 83 ec 38          	sub    $0x38,%rsp
  400dc8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400dcd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400dd2:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400dd7:	e8 c3 fd ff ff       	callq  400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400ddc:	48 89 c7             	mov    %rax,%rdi
  400ddf:	e8 0e 01 00 00       	callq  400ef2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400de4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400de9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400dee:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400df3:	48 83 e8 04          	sub    $0x4,%rax
  400df7:	48 c1 e8 02          	shr    $0x2,%rax
  400dfb:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400e02:	00 
  400e03:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400e08:	48 01 d0             	add    %rdx,%rax
  400e0b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400e10:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400e15:	48 83 c4 38          	add    $0x38,%rsp
  400e19:	c3                   	retq   

0000000000400e1a <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1ELi1E9TestClassE22data_ptr_uninitializedILi4ELi1EEENSt9enable_ifIXaaneT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400e1a:	48 83 ec 38          	sub    $0x38,%rsp
  400e1e:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400e23:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e28:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400e2d:	e8 6d fd ff ff       	callq  400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400e32:	48 89 c7             	mov    %rax,%rdi
  400e35:	e8 b8 00 00 00       	callq  400ef2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e3a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400e3f:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400e44:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400e49:	48 83 e8 04          	sub    $0x4,%rax
  400e4d:	48 c1 e8 02          	shr    $0x2,%rax
  400e51:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400e58:	00 
  400e59:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400e5e:	48 01 d0             	add    %rdx,%rax
  400e61:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400e67:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400e6c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400e71:	48 83 c4 38          	add    $0x38,%rsp
  400e75:	c3                   	retq   

0000000000400e76 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400e76:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e7b:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e80:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400e87:	90                   	nop
  400e88:	c3                   	retq   

0000000000400e89 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e89:	48 83 ec 28          	sub    $0x28,%rsp
  400e8d:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400e92:	e8 08 fd ff ff       	callq  400b9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageELi1EE7storageEv>
  400e97:	48 89 c7             	mov    %rax,%rdi
  400e9a:	e8 53 00 00 00       	callq  400ef2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e9f:	48 89 c2             	mov    %rax,%rdx
  400ea2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ea7:	48 c1 e0 02          	shl    $0x2,%rax
  400eab:	48 01 d0             	add    %rdx,%rax
  400eae:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400eb3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400eb8:	48 83 c4 28          	add    $0x28,%rsp
  400ebc:	c3                   	retq   

0000000000400ebd <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400ebd:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400ec2:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400ec7:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ecc:	48 8b 00             	mov    (%rax),%rax
  400ecf:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400ed4:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ed9:	48 8b 10             	mov    (%rax),%rdx
  400edc:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400ee1:	48 01 c2             	add    %rax,%rdx
  400ee4:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400ee9:	48 89 10             	mov    %rdx,(%rax)
  400eec:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400ef1:	c3                   	retq   

0000000000400ef2 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400ef2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ef7:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400efc:	48 8b 40 18          	mov    0x18(%rax),%rax
  400f00:	c3                   	retq   
  400f01:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f08:	00 00 00 
  400f0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400f10 <__libc_csu_init>:
  400f10:	41 57                	push   %r15
  400f12:	41 56                	push   %r14
  400f14:	41 89 ff             	mov    %edi,%r15d
  400f17:	41 55                	push   %r13
  400f19:	41 54                	push   %r12
  400f1b:	4c 8d 25 ce 0e 20 00 	lea    0x200ece(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400f22:	55                   	push   %rbp
  400f23:	48 8d 2d ce 0e 20 00 	lea    0x200ece(%rip),%rbp        # 601df8 <__init_array_end>
  400f2a:	53                   	push   %rbx
  400f2b:	49 89 f6             	mov    %rsi,%r14
  400f2e:	49 89 d5             	mov    %rdx,%r13
  400f31:	4c 29 e5             	sub    %r12,%rbp
  400f34:	48 83 ec 08          	sub    $0x8,%rsp
  400f38:	48 c1 fd 03          	sar    $0x3,%rbp
  400f3c:	e8 b7 f6 ff ff       	callq  4005f8 <_init>
  400f41:	48 85 ed             	test   %rbp,%rbp
  400f44:	74 20                	je     400f66 <__libc_csu_init+0x56>
  400f46:	31 db                	xor    %ebx,%ebx
  400f48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400f4f:	00 
  400f50:	4c 89 ea             	mov    %r13,%rdx
  400f53:	4c 89 f6             	mov    %r14,%rsi
  400f56:	44 89 ff             	mov    %r15d,%edi
  400f59:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400f5d:	48 83 c3 01          	add    $0x1,%rbx
  400f61:	48 39 eb             	cmp    %rbp,%rbx
  400f64:	75 ea                	jne    400f50 <__libc_csu_init+0x40>
  400f66:	48 83 c4 08          	add    $0x8,%rsp
  400f6a:	5b                   	pop    %rbx
  400f6b:	5d                   	pop    %rbp
  400f6c:	41 5c                	pop    %r12
  400f6e:	41 5d                	pop    %r13
  400f70:	41 5e                	pop    %r14
  400f72:	41 5f                	pop    %r15
  400f74:	c3                   	retq   
  400f75:	90                   	nop
  400f76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400f7d:	00 00 00 

0000000000400f80 <__libc_csu_fini>:
  400f80:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f84 <_fini>:
  400f84:	48 83 ec 08          	sub    $0x8,%rsp
  400f88:	48 83 c4 08          	add    $0x8,%rsp
  400f8c:	c3                   	retq   
