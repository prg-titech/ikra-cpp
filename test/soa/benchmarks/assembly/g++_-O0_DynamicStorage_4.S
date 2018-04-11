
bin/g++_-O0_DynamicStorage_4:     file format elf64-x86-64


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
  4006af:	49 c7 c0 00 0f 40 00 	mov    $0x400f00,%r8
  4006b6:	48 c7 c1 90 0e 40 00 	mov    $0x400e90,%rcx
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
  400750:	80 3d 09 19 20 00 00 	cmpb   $0x0,0x201909(%rip)        # 602060 <completed.7585>
  400757:	75 11                	jne    40076a <__do_global_dtors_aux+0x1a>
  400759:	55                   	push   %rbp
  40075a:	48 89 e5             	mov    %rsp,%rbp
  40075d:	e8 6e ff ff ff       	callq  4006d0 <deregister_tm_clones>
  400762:	5d                   	pop    %rbp
  400763:	c6 05 f6 18 20 00 01 	movb   $0x1,0x2018f6(%rip)        # 602060 <completed.7585>
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
  40079b:	e8 cd 03 00 00       	callq  400b6d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4007a0:	bf 04 00 00 00       	mov    $0x4,%edi
  4007a5:	e8 0d 04 00 00       	callq  400bb7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4007aa:	48 89 c3             	mov    %rax,%rbx
  4007ad:	b8 00 00 00 00       	mov    $0x0,%eax
  4007b2:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4007b6:	48 83 f8 03          	cmp    $0x3,%rax
  4007ba:	74 06                	je     4007c2 <main+0x2c>
  4007bc:	48 83 c0 01          	add    $0x1,%rax
  4007c0:	eb f0                	jmp    4007b2 <main+0x1c>
  4007c2:	48 89 df             	mov    %rbx,%rdi
  4007c5:	e8 c8 02 00 00       	callq  400a92 <_ZN9TestClassC1Ev>
  4007ca:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4007cf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007d4:	48 83 c0 04          	add    $0x4,%rax
  4007d8:	be 77 77 00 00       	mov    $0x7777,%esi
  4007dd:	48 89 c7             	mov    %rax,%rdi
  4007e0:	e8 15 04 00 00       	callq  400bfa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEaSEi>
  4007e5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007ea:	48 83 c0 04          	add    $0x4,%rax
  4007ee:	be 88 88 00 00       	mov    $0x8888,%esi
  4007f3:	48 89 c7             	mov    %rax,%rdi
  4007f6:	e8 2d 04 00 00       	callq  400c28 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEaSEi>
  4007fb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400800:	48 89 c7             	mov    %rax,%rdi
  400803:	e8 90 01 00 00       	callq  400998 <_ZN9TestClass15increase_field0Ev>
  400808:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40080d:	48 89 c7             	mov    %rax,%rdi
  400810:	e8 eb 01 00 00       	callq  400a00 <_ZN9TestClass15increase_field1Ev>
  400815:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40081a:	48 83 c0 04          	add    $0x4,%rax
  40081e:	48 89 c7             	mov    %rax,%rdi
  400821:	e8 4c 04 00 00       	callq  400c72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEcvRiEv>
  400826:	8b 18                	mov    (%rax),%ebx
  400828:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40082d:	48 83 c0 04          	add    $0x4,%rax
  400831:	48 89 c7             	mov    %rax,%rdi
  400834:	e8 1d 04 00 00       	callq  400c56 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEcvRiEv>
  400839:	8b 00                	mov    (%rax),%eax
  40083b:	89 da                	mov    %ebx,%edx
  40083d:	89 c6                	mov    %eax,%esi
  40083f:	bf 2c 0f 40 00       	mov    $0x400f2c,%edi
  400844:	b8 00 00 00 00       	mov    $0x0,%eax
  400849:	e8 e2 fd ff ff       	callq  400630 <printf@plt>
  40084e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400853:	48 83 c0 04          	add    $0x4,%rax
  400857:	48 89 c7             	mov    %rax,%rdi
  40085a:	e8 f7 03 00 00       	callq  400c56 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEcvRiEv>
  40085f:	8b 00                	mov    (%rax),%eax
  400861:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  400866:	75 1a                	jne    400882 <main+0xec>
  400868:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40086d:	48 83 c0 04          	add    $0x4,%rax
  400871:	48 89 c7             	mov    %rax,%rdi
  400874:	e8 f9 03 00 00       	callq  400c72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEcvRiEv>
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
  40089d:	e8 15 03 00 00       	callq  400bb7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>
  4008a2:	48 89 c3             	mov    %rax,%rbx
  4008a5:	b8 00 00 00 00       	mov    $0x0,%eax
  4008aa:	c6 04 03 00          	movb   $0x0,(%rbx,%rax,1)
  4008ae:	48 83 f8 03          	cmp    $0x3,%rax
  4008b2:	74 06                	je     4008ba <_Z12new_instancev+0x23>
  4008b4:	48 83 c0 01          	add    $0x1,%rax
  4008b8:	eb f0                	jmp    4008aa <_Z12new_instancev+0x13>
  4008ba:	48 89 df             	mov    %rbx,%rdi
  4008bd:	e8 d0 01 00 00       	callq  400a92 <_ZN9TestClassC1Ev>
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
  4008e1:	e8 14 03 00 00       	callq  400bfa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEaSEi>
  4008e6:	90                   	nop
  4008e7:	48 83 c4 18          	add    $0x18,%rsp
  4008eb:	c3                   	retq   

00000000004008ec <_Z11read_field0P9TestClass>:
  4008ec:	48 83 ec 18          	sub    $0x18,%rsp
  4008f0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008f5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008fa:	48 83 c0 04          	add    $0x4,%rax
  4008fe:	48 89 c7             	mov    %rax,%rdi
  400901:	e8 50 03 00 00       	callq  400c56 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEcvRiEv>
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
  400927:	e8 fc 02 00 00       	callq  400c28 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEaSEi>
  40092c:	90                   	nop
  40092d:	48 83 c4 18          	add    $0x18,%rsp
  400931:	c3                   	retq   

0000000000400932 <_Z11read_field1P9TestClass>:
  400932:	48 83 ec 18          	sub    $0x18,%rsp
  400936:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40093b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400940:	48 83 c0 04          	add    $0x4,%rax
  400944:	48 89 c7             	mov    %rax,%rdi
  400947:	e8 26 03 00 00       	callq  400c72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEcvRiEv>
  40094c:	8b 00                	mov    (%rax),%eax
  40094e:	48 83 c4 18          	add    $0x18,%rsp
  400952:	c3                   	retq   

0000000000400953 <_Z21explicit_write_field0m>:
  400953:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400958:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40095d:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  400964:	77 77 00 00 
  400968:	90                   	nop
  400969:	c3                   	retq   

000000000040096a <_Z20explicit_read_field0m>:
  40096a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40096f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400974:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  40097b:	c3                   	retq   

000000000040097c <_ZnwmPv>:
  40097c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400981:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400986:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  40098b:	c3                   	retq   

000000000040098c <_ZdlPvS_>:
  40098c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400991:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400996:	90                   	nop
  400997:	c3                   	retq   

0000000000400998 <_ZN9TestClass15increase_field0Ev>:
  400998:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40099d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009a1:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009a5:	55                   	push   %rbp
  4009a6:	48 89 e5             	mov    %rsp,%rbp
  4009a9:	41 52                	push   %r10
  4009ab:	48 83 ec 48          	sub    $0x48,%rsp
  4009af:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009b3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009ba:	00 00 
  4009bc:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009c0:	31 c0                	xor    %eax,%eax
  4009c2:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  4009c6:	48 8d 48 04          	lea    0x4(%rax),%rcx
  4009ca:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009ce:	ba 55 55 00 00       	mov    $0x5555,%edx
  4009d3:	48 89 ce             	mov    %rcx,%rsi
  4009d6:	48 89 c7             	mov    %rax,%rdi
  4009d9:	e8 f2 00 00 00       	callq  400ad0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEmLEi>
  4009de:	90                   	nop
  4009df:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009e3:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009ea:	00 00 
  4009ec:	74 05                	je     4009f3 <_ZN9TestClass15increase_field0Ev+0x5b>
  4009ee:	e8 6d fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  4009f3:	48 83 c4 48          	add    $0x48,%rsp
  4009f7:	41 5a                	pop    %r10
  4009f9:	5d                   	pop    %rbp
  4009fa:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009fe:	c3                   	retq   
  4009ff:	90                   	nop

0000000000400a00 <_ZN9TestClass15increase_field1Ev>:
  400a00:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  400a05:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  400a09:	41 ff 72 f8          	pushq  -0x8(%r10)
  400a0d:	55                   	push   %rbp
  400a0e:	48 89 e5             	mov    %rsp,%rbp
  400a11:	41 52                	push   %r10
  400a13:	48 83 ec 48          	sub    $0x48,%rsp
  400a17:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400a1b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400a22:	00 00 
  400a24:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a28:	31 c0                	xor    %eax,%eax
  400a2a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
  400a2e:	48 8d 48 04          	lea    0x4(%rax),%rcx
  400a32:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400a36:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a3b:	48 89 ce             	mov    %rcx,%rsi
  400a3e:	48 89 c7             	mov    %rax,%rdi
  400a41:	e8 d2 00 00 00       	callq  400b18 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEmLEi>
  400a46:	90                   	nop
  400a47:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a4b:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a52:	00 00 
  400a54:	74 05                	je     400a5b <_ZN9TestClass15increase_field1Ev+0x5b>
  400a56:	e8 05 fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  400a5b:	48 83 c4 48          	add    $0x48,%rsp
  400a5f:	41 5a                	pop    %r10
  400a61:	5d                   	pop    %rbp
  400a62:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a66:	c3                   	retq   
  400a67:	90                   	nop

0000000000400a68 <_ZN9TestClass14storage_bufferEv>:
  400a68:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a6d:	c3                   	retq   

0000000000400a6e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>:
  400a6e:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a73:	90                   	nop
  400a74:	c3                   	retq   
  400a75:	90                   	nop

0000000000400a76 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>:
  400a76:	48 83 ec 18          	sub    $0x18,%rsp
  400a7a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a7f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a84:	48 89 c7             	mov    %rax,%rdi
  400a87:	e8 e2 ff ff ff       	callq  400a6e <_ZN4ikra3soa10SizeNDummyILm4EEC1Ev>
  400a8c:	90                   	nop
  400a8d:	48 83 c4 18          	add    $0x18,%rsp
  400a91:	c3                   	retq   

0000000000400a92 <_ZN9TestClassC1Ev>:
  400a92:	48 83 ec 18          	sub    $0x18,%rsp
  400a96:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a9b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aa0:	48 89 c7             	mov    %rax,%rdi
  400aa3:	e8 ce ff ff ff       	callq  400a76 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEC1Ev>
  400aa8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aad:	48 83 c0 04          	add    $0x4,%rax
  400ab1:	48 89 c7             	mov    %rax,%rdi
  400ab4:	e8 2b 01 00 00       	callq  400be4 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEC1Ev>
  400ab9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400abe:	48 83 c0 04          	add    $0x4,%rax
  400ac2:	48 89 c7             	mov    %rax,%rdi
  400ac5:	e8 22 01 00 00       	callq  400bec <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEC1Ev>
  400aca:	90                   	nop
  400acb:	48 83 c4 18          	add    $0x18,%rsp
  400acf:	c3                   	retq   

0000000000400ad0 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEmLEi>:
  400ad0:	48 83 ec 28          	sub    $0x28,%rsp
  400ad4:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400ad9:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400ade:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ae2:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400ae7:	48 89 c7             	mov    %rax,%rdi
  400aea:	e8 9f 01 00 00       	callq  400c8e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400aef:	8b 10                	mov    (%rax),%edx
  400af1:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400af6:	89 10                	mov    %edx,(%rax)
  400af8:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400afd:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b02:	48 89 d6             	mov    %rdx,%rsi
  400b05:	48 89 c7             	mov    %rax,%rdi
  400b08:	e8 9d 01 00 00       	callq  400caa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEC1ERKS3_>
  400b0d:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b12:	48 83 c4 28          	add    $0x28,%rsp
  400b16:	c3                   	retq   
  400b17:	90                   	nop

0000000000400b18 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEmLEi>:
  400b18:	48 83 ec 28          	sub    $0x28,%rsp
  400b1c:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400b21:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400b26:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400b2a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b2f:	48 89 c7             	mov    %rax,%rdi
  400b32:	e8 7f 01 00 00       	callq  400cb6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b37:	8b 10                	mov    (%rax),%edx
  400b39:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400b3e:	89 10                	mov    %edx,(%rax)
  400b40:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400b45:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b4a:	48 89 d6             	mov    %rdx,%rsi
  400b4d:	48 89 c7             	mov    %rax,%rdi
  400b50:	e8 7d 01 00 00       	callq  400cd2 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEC1ERKS3_>
  400b55:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b5a:	48 83 c4 28          	add    $0x28,%rsp
  400b5e:	c3                   	retq   

0000000000400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>:
  400b5f:	48 83 ec 08          	sub    $0x8,%rsp
  400b63:	e8 00 ff ff ff       	callq  400a68 <_ZN9TestClass14storage_bufferEv>
  400b68:	48 83 c4 08          	add    $0x8,%rsp
  400b6c:	c3                   	retq   

0000000000400b6d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  400b6d:	41 54                	push   %r12
  400b6f:	55                   	push   %rbp
  400b70:	53                   	push   %rbx
  400b71:	e8 e9 ff ff ff       	callq  400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400b76:	48 89 c5             	mov    %rax,%rbp
  400b79:	48 89 ee             	mov    %rbp,%rsi
  400b7c:	bf 20 00 00 00       	mov    $0x20,%edi
  400b81:	e8 f6 fd ff ff       	callq  40097c <_ZnwmPv>
  400b86:	48 89 c3             	mov    %rax,%rbx
  400b89:	48 85 db             	test   %rbx,%rbx
  400b8c:	74 23                	je     400bb1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b8e:	48 89 df             	mov    %rbx,%rdi
  400b91:	e8 48 01 00 00       	callq  400cde <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400b96:	eb 19                	jmp    400bb1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b98:	49 89 c4             	mov    %rax,%r12
  400b9b:	48 89 ee             	mov    %rbp,%rsi
  400b9e:	48 89 df             	mov    %rbx,%rdi
  400ba1:	e8 e6 fd ff ff       	callq  40098c <_ZdlPvS_>
  400ba6:	4c 89 e0             	mov    %r12,%rax
  400ba9:	48 89 c7             	mov    %rax,%rdi
  400bac:	e8 cf fa ff ff       	callq  400680 <_Unwind_Resume@plt>
  400bb1:	90                   	nop
  400bb2:	5b                   	pop    %rbx
  400bb3:	5d                   	pop    %rbp
  400bb4:	41 5c                	pop    %r12
  400bb6:	c3                   	retq   

0000000000400bb7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEnwEm>:
  400bb7:	48 83 ec 18          	sub    $0x18,%rsp
  400bbb:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bc0:	e8 64 01 00 00       	callq  400d29 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400bc5:	e8 95 ff ff ff       	callq  400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400bca:	be 01 00 00 00       	mov    $0x1,%esi
  400bcf:	48 89 c7             	mov    %rax,%rdi
  400bd2:	e8 73 01 00 00       	callq  400d4a <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400bd7:	48 89 c7             	mov    %rax,%rdi
  400bda:	e8 4c 01 00 00       	callq  400d2b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>
  400bdf:	48 83 c4 18          	add    $0x18,%rsp
  400be3:	c3                   	retq   

0000000000400be4 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEC1Ev>:
  400be4:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400be9:	90                   	nop
  400bea:	c3                   	retq   
  400beb:	90                   	nop

0000000000400bec <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEC1Ev>:
  400bec:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bf1:	90                   	nop
  400bf2:	c3                   	retq   

0000000000400bf3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEEdlEPv>:
  400bf3:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bf8:	90                   	nop
  400bf9:	c3                   	retq   

0000000000400bfa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEaSEi>:
  400bfa:	48 83 ec 18          	sub    $0x18,%rsp
  400bfe:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c03:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c07:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c0c:	48 89 c7             	mov    %rax,%rdi
  400c0f:	e8 7a 00 00 00       	callq  400c8e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c14:	48 89 c2             	mov    %rax,%rdx
  400c17:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c1b:	89 02                	mov    %eax,(%rdx)
  400c1d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c22:	48 83 c4 18          	add    $0x18,%rsp
  400c26:	c3                   	retq   
  400c27:	90                   	nop

0000000000400c28 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEaSEi>:
  400c28:	48 83 ec 18          	sub    $0x18,%rsp
  400c2c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c31:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c35:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c3a:	48 89 c7             	mov    %rax,%rdi
  400c3d:	e8 74 00 00 00       	callq  400cb6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c42:	48 89 c2             	mov    %rax,%rdx
  400c45:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c49:	89 02                	mov    %eax,(%rdx)
  400c4b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c50:	48 83 c4 18          	add    $0x18,%rsp
  400c54:	c3                   	retq   
  400c55:	90                   	nop

0000000000400c56 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEcvRiEv>:
  400c56:	48 83 ec 18          	sub    $0x18,%rsp
  400c5a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c5f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c64:	48 89 c7             	mov    %rax,%rdi
  400c67:	e8 22 00 00 00       	callq  400c8e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c6c:	48 83 c4 18          	add    $0x18,%rsp
  400c70:	c3                   	retq   
  400c71:	90                   	nop

0000000000400c72 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEcvRiEv>:
  400c72:	48 83 ec 18          	sub    $0x18,%rsp
  400c76:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c7b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c80:	48 89 c7             	mov    %rax,%rdi
  400c83:	e8 2e 00 00 00       	callq  400cb6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c88:	48 83 c4 18          	add    $0x18,%rsp
  400c8c:	c3                   	retq   
  400c8d:	90                   	nop

0000000000400c8e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c8e:	48 83 ec 18          	sub    $0x18,%rsp
  400c92:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c97:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c9c:	48 89 c7             	mov    %rax,%rdi
  400c9f:	e8 dc 00 00 00       	callq  400d80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ca4:	48 83 c4 18          	add    $0x18,%rsp
  400ca8:	c3                   	retq   
  400ca9:	90                   	nop

0000000000400caa <_ZN4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassEC1ERKS3_>:
  400caa:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400caf:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cb4:	90                   	nop
  400cb5:	c3                   	retq   

0000000000400cb6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE8data_ptrILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cb6:	48 83 ec 18          	sub    $0x18,%rsp
  400cba:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cbf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cc4:	48 89 c7             	mov    %rax,%rdi
  400cc7:	e8 0a 01 00 00       	callq  400dd6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ccc:	48 83 c4 18          	add    $0x18,%rsp
  400cd0:	c3                   	retq   
  400cd1:	90                   	nop

0000000000400cd2 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassEC1ERKS3_>:
  400cd2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cd7:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cdc:	90                   	nop
  400cdd:	c3                   	retq   

0000000000400cde <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400cde:	48 83 ec 18          	sub    $0x18,%rsp
  400ce2:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ce7:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cec:	48 89 c7             	mov    %rax,%rdi
  400cef:	e8 3e 01 00 00       	callq  400e32 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400cf4:	bf c0 91 00 00       	mov    $0x91c0,%edi
  400cf9:	e8 42 f9 ff ff       	callq  400640 <malloc@plt>
  400cfe:	48 89 c2             	mov    %rax,%rdx
  400d01:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d06:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400d0a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d0f:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
  400d16:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d1b:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d22:	00 
  400d23:	90                   	nop
  400d24:	48 83 c4 18          	add    $0x18,%rsp
  400d28:	c3                   	retq   

0000000000400d29 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE18check_sizeof_classILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d29:	90                   	nop
  400d2a:	c3                   	retq   

0000000000400d2b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE3getEm>:
  400d2b:	48 83 ec 18          	sub    $0x18,%rsp
  400d2f:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d34:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d39:	48 83 c0 01          	add    $0x1,%rax
  400d3d:	48 89 c7             	mov    %rax,%rdi
  400d40:	e8 00 01 00 00       	callq  400e45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d45:	48 83 c4 18          	add    $0x18,%rsp
  400d49:	c3                   	retq   

0000000000400d4a <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d4a:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400d4f:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400d54:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d59:	48 8b 00             	mov    (%rax),%rax
  400d5c:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400d61:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d66:	48 8b 10             	mov    (%rax),%rdx
  400d69:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400d6e:	48 01 c2             	add    %rax,%rdx
  400d71:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d76:	48 89 10             	mov    %rdx,(%rax)
  400d79:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d7e:	c3                   	retq   
  400d7f:	90                   	nop

0000000000400d80 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400d80:	48 83 ec 38          	sub    $0x38,%rsp
  400d84:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d89:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d8e:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d93:	e8 c7 fd ff ff       	callq  400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400d98:	48 89 c7             	mov    %rax,%rdi
  400d9b:	e8 da 00 00 00       	callq  400e7a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400da0:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400da5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400daa:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400daf:	48 83 e8 04          	sub    $0x4,%rax
  400db3:	48 c1 e8 02          	shr    $0x2,%rax
  400db7:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400dbe:	00 
  400dbf:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400dc4:	48 01 d0             	add    %rdx,%rax
  400dc7:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400dcc:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400dd1:	48 83 c4 38          	add    $0x38,%rsp
  400dd5:	c3                   	retq   

0000000000400dd6 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi4ELi1E9TestClassE22data_ptr_uninitializedILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400dd6:	48 83 ec 38          	sub    $0x38,%rsp
  400dda:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ddf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400de4:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400de9:	e8 71 fd ff ff       	callq  400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400dee:	48 89 c7             	mov    %rax,%rdi
  400df1:	e8 84 00 00 00       	callq  400e7a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400df6:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  400dfb:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400e00:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
  400e05:	48 83 e8 04          	sub    $0x4,%rax
  400e09:	48 c1 e8 02          	shr    $0x2,%rax
  400e0d:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400e14:	00 
  400e15:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  400e1a:	48 01 d0             	add    %rdx,%rax
  400e1d:	48 05 e0 48 00 00    	add    $0x48e0,%rax
  400e23:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  400e28:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400e2d:	48 83 c4 38          	add    $0x38,%rsp
  400e31:	c3                   	retq   

0000000000400e32 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400e32:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e37:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e3c:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400e43:	90                   	nop
  400e44:	c3                   	retq   

0000000000400e45 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE4get_ILi4EEENSt9enable_ifIXneT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400e45:	48 83 ec 28          	sub    $0x28,%rsp
  400e49:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400e4e:	e8 0c fd ff ff       	callq  400b5f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi4ENS0_14DynamicStorageEE7storageEv>
  400e53:	48 89 c7             	mov    %rax,%rdi
  400e56:	e8 1f 00 00 00       	callq  400e7a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e5b:	48 89 c2             	mov    %rax,%rdx
  400e5e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e63:	48 c1 e0 02          	shl    $0x2,%rax
  400e67:	48 01 d0             	add    %rdx,%rax
  400e6a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400e6f:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400e74:	48 83 c4 28          	add    $0x28,%rsp
  400e78:	c3                   	retq   
  400e79:	90                   	nop

0000000000400e7a <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400e7a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e7f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e84:	48 8b 40 18          	mov    0x18(%rax),%rax
  400e88:	c3                   	retq   
  400e89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400e90 <__libc_csu_init>:
  400e90:	41 57                	push   %r15
  400e92:	41 56                	push   %r14
  400e94:	41 89 ff             	mov    %edi,%r15d
  400e97:	41 55                	push   %r13
  400e99:	41 54                	push   %r12
  400e9b:	4c 8d 25 4e 0f 20 00 	lea    0x200f4e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400ea2:	55                   	push   %rbp
  400ea3:	48 8d 2d 4e 0f 20 00 	lea    0x200f4e(%rip),%rbp        # 601df8 <__init_array_end>
  400eaa:	53                   	push   %rbx
  400eab:	49 89 f6             	mov    %rsi,%r14
  400eae:	49 89 d5             	mov    %rdx,%r13
  400eb1:	4c 29 e5             	sub    %r12,%rbp
  400eb4:	48 83 ec 08          	sub    $0x8,%rsp
  400eb8:	48 c1 fd 03          	sar    $0x3,%rbp
  400ebc:	e8 37 f7 ff ff       	callq  4005f8 <_init>
  400ec1:	48 85 ed             	test   %rbp,%rbp
  400ec4:	74 20                	je     400ee6 <__libc_csu_init+0x56>
  400ec6:	31 db                	xor    %ebx,%ebx
  400ec8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ecf:	00 
  400ed0:	4c 89 ea             	mov    %r13,%rdx
  400ed3:	4c 89 f6             	mov    %r14,%rsi
  400ed6:	44 89 ff             	mov    %r15d,%edi
  400ed9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400edd:	48 83 c3 01          	add    $0x1,%rbx
  400ee1:	48 39 eb             	cmp    %rbp,%rbx
  400ee4:	75 ea                	jne    400ed0 <__libc_csu_init+0x40>
  400ee6:	48 83 c4 08          	add    $0x8,%rsp
  400eea:	5b                   	pop    %rbx
  400eeb:	5d                   	pop    %rbp
  400eec:	41 5c                	pop    %r12
  400eee:	41 5d                	pop    %r13
  400ef0:	41 5e                	pop    %r14
  400ef2:	41 5f                	pop    %r15
  400ef4:	c3                   	retq   
  400ef5:	90                   	nop
  400ef6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400efd:	00 00 00 

0000000000400f00 <__libc_csu_fini>:
  400f00:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400f04 <_fini>:
  400f04:	48 83 ec 08          	sub    $0x8,%rsp
  400f08:	48 83 c4 08          	add    $0x8,%rsp
  400f0c:	c3                   	retq   
