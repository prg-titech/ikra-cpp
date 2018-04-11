
bin/g++_-O0_DynamicStorage_0:     file format elf64-x86-64


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
  4006af:	49 c7 c0 50 0e 40 00 	mov    $0x400e50,%r8
  4006b6:	48 c7 c1 e0 0d 40 00 	mov    $0x400de0,%rcx
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
  40079b:	e8 6b 03 00 00       	callq  400b0b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>
  4007a0:	bf 00 00 00 00       	mov    $0x0,%edi
  4007a5:	e8 ab 03 00 00       	callq  400b55 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  4007aa:	48 89 c3             	mov    %rax,%rbx
  4007ad:	48 89 df             	mov    %rbx,%rdi
  4007b0:	e8 83 02 00 00       	callq  400a38 <_ZN9TestClassC1Ev>
  4007b5:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4007ba:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bf:	be 77 77 00 00       	mov    $0x7777,%esi
  4007c4:	48 89 c7             	mov    %rax,%rdi
  4007c7:	e8 cc 03 00 00       	callq  400b98 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>
  4007cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007d1:	be 88 88 00 00       	mov    $0x8888,%esi
  4007d6:	48 89 c7             	mov    %rax,%rdi
  4007d9:	e8 e8 03 00 00       	callq  400bc6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>
  4007de:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007e3:	48 89 c7             	mov    %rax,%rdi
  4007e6:	e8 5b 01 00 00       	callq  400946 <_ZN9TestClass15increase_field0Ev>
  4007eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007f0:	48 89 c7             	mov    %rax,%rdi
  4007f3:	e8 b2 01 00 00       	callq  4009aa <_ZN9TestClass15increase_field1Ev>
  4007f8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007fd:	48 89 c7             	mov    %rax,%rdi
  400800:	e8 0b 04 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  400805:	8b 18                	mov    (%rax),%ebx
  400807:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40080c:	48 89 c7             	mov    %rax,%rdi
  40080f:	e8 e0 03 00 00       	callq  400bf4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  400814:	8b 00                	mov    (%rax),%eax
  400816:	89 da                	mov    %ebx,%edx
  400818:	89 c6                	mov    %eax,%esi
  40081a:	bf 7c 0e 40 00       	mov    $0x400e7c,%edi
  40081f:	b8 00 00 00 00       	mov    $0x0,%eax
  400824:	e8 07 fe ff ff       	callq  400630 <printf@plt>
  400829:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40082e:	48 89 c7             	mov    %rax,%rdi
  400831:	e8 be 03 00 00       	callq  400bf4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  400836:	8b 00                	mov    (%rax),%eax
  400838:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40083d:	75 16                	jne    400855 <main+0xbf>
  40083f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400844:	48 89 c7             	mov    %rax,%rdi
  400847:	e8 c4 03 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  40084c:	8b 00                	mov    (%rax),%eax
  40084e:	3d 20 64 68 24       	cmp    $0x24686420,%eax
  400853:	74 07                	je     40085c <main+0xc6>
  400855:	b8 01 00 00 00       	mov    $0x1,%eax
  40085a:	eb 05                	jmp    400861 <main+0xcb>
  40085c:	b8 00 00 00 00       	mov    $0x0,%eax
  400861:	0f b6 c0             	movzbl %al,%eax
  400864:	48 83 c4 10          	add    $0x10,%rsp
  400868:	5b                   	pop    %rbx
  400869:	c3                   	retq   

000000000040086a <_Z12new_instancev>:
  40086a:	53                   	push   %rbx
  40086b:	bf 00 00 00 00       	mov    $0x0,%edi
  400870:	e8 e0 02 00 00       	callq  400b55 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  400875:	48 89 c3             	mov    %rax,%rbx
  400878:	48 89 df             	mov    %rbx,%rdi
  40087b:	e8 b8 01 00 00       	callq  400a38 <_ZN9TestClassC1Ev>
  400880:	48 89 d8             	mov    %rbx,%rax
  400883:	5b                   	pop    %rbx
  400884:	c3                   	retq   

0000000000400885 <_Z12write_field0P9TestClass>:
  400885:	48 83 ec 18          	sub    $0x18,%rsp
  400889:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40088e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400893:	be 77 77 00 00       	mov    $0x7777,%esi
  400898:	48 89 c7             	mov    %rax,%rdi
  40089b:	e8 f8 02 00 00       	callq  400b98 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>
  4008a0:	90                   	nop
  4008a1:	48 83 c4 18          	add    $0x18,%rsp
  4008a5:	c3                   	retq   

00000000004008a6 <_Z11read_field0P9TestClass>:
  4008a6:	48 83 ec 18          	sub    $0x18,%rsp
  4008aa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008af:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008b4:	48 89 c7             	mov    %rax,%rdi
  4008b7:	e8 38 03 00 00       	callq  400bf4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>
  4008bc:	8b 00                	mov    (%rax),%eax
  4008be:	48 83 c4 18          	add    $0x18,%rsp
  4008c2:	c3                   	retq   

00000000004008c3 <_Z12write_field1P9TestClass>:
  4008c3:	48 83 ec 18          	sub    $0x18,%rsp
  4008c7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008d1:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d6:	48 89 c7             	mov    %rax,%rdi
  4008d9:	e8 e8 02 00 00       	callq  400bc6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>
  4008de:	90                   	nop
  4008df:	48 83 c4 18          	add    $0x18,%rsp
  4008e3:	c3                   	retq   

00000000004008e4 <_Z11read_field1P9TestClass>:
  4008e4:	48 83 ec 18          	sub    $0x18,%rsp
  4008e8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008ed:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008f2:	48 89 c7             	mov    %rax,%rdi
  4008f5:	e8 16 03 00 00       	callq  400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>
  4008fa:	8b 00                	mov    (%rax),%eax
  4008fc:	48 83 c4 18          	add    $0x18,%rsp
  400900:	c3                   	retq   

0000000000400901 <_Z21explicit_write_field0m>:
  400901:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400906:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40090b:	c7 04 85 a0 20 60 00 	movl   $0x7777,0x6020a0(,%rax,4)
  400912:	77 77 00 00 
  400916:	90                   	nop
  400917:	c3                   	retq   

0000000000400918 <_Z20explicit_read_field0m>:
  400918:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40091d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400922:	8b 04 85 a0 20 60 00 	mov    0x6020a0(,%rax,4),%eax
  400929:	c3                   	retq   

000000000040092a <_ZnwmPv>:
  40092a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40092f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400934:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400939:	c3                   	retq   

000000000040093a <_ZdlPvS_>:
  40093a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40093f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400944:	90                   	nop
  400945:	c3                   	retq   

0000000000400946 <_ZN9TestClass15increase_field0Ev>:
  400946:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40094b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40094f:	41 ff 72 f8          	pushq  -0x8(%r10)
  400953:	55                   	push   %rbp
  400954:	48 89 e5             	mov    %rsp,%rbp
  400957:	41 52                	push   %r10
  400959:	48 83 ec 48          	sub    $0x48,%rsp
  40095d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400961:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400968:	00 00 
  40096a:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  40096e:	31 c0                	xor    %eax,%eax
  400970:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400974:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400978:	ba 55 55 00 00       	mov    $0x5555,%edx
  40097d:	48 89 ce             	mov    %rcx,%rsi
  400980:	48 89 c7             	mov    %rax,%rdi
  400983:	e8 e6 00 00 00       	callq  400a6e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEmLEi>
  400988:	90                   	nop
  400989:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  40098d:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400994:	00 00 
  400996:	74 05                	je     40099d <_ZN9TestClass15increase_field0Ev+0x57>
  400998:	e8 c3 fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  40099d:	48 83 c4 48          	add    $0x48,%rsp
  4009a1:	41 5a                	pop    %r10
  4009a3:	5d                   	pop    %rbp
  4009a4:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009a8:	c3                   	retq   
  4009a9:	90                   	nop

00000000004009aa <_ZN9TestClass15increase_field1Ev>:
  4009aa:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4009af:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009b3:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009b7:	55                   	push   %rbp
  4009b8:	48 89 e5             	mov    %rsp,%rbp
  4009bb:	41 52                	push   %r10
  4009bd:	48 83 ec 48          	sub    $0x48,%rsp
  4009c1:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009c5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009cc:	00 00 
  4009ce:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009d2:	31 c0                	xor    %eax,%eax
  4009d4:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4009d8:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009dc:	ba 44 44 00 00       	mov    $0x4444,%edx
  4009e1:	48 89 ce             	mov    %rcx,%rsi
  4009e4:	48 89 c7             	mov    %rax,%rdi
  4009e7:	e8 ca 00 00 00       	callq  400ab6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEmLEi>
  4009ec:	90                   	nop
  4009ed:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009f1:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009f8:	00 00 
  4009fa:	74 05                	je     400a01 <_ZN9TestClass15increase_field1Ev+0x57>
  4009fc:	e8 5f fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  400a01:	48 83 c4 48          	add    $0x48,%rsp
  400a05:	41 5a                	pop    %r10
  400a07:	5d                   	pop    %rbp
  400a08:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a0c:	c3                   	retq   
  400a0d:	90                   	nop

0000000000400a0e <_ZN9TestClass14storage_bufferEv>:
  400a0e:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a13:	c3                   	retq   

0000000000400a14 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400a14:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a19:	90                   	nop
  400a1a:	c3                   	retq   
  400a1b:	90                   	nop

0000000000400a1c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>:
  400a1c:	48 83 ec 18          	sub    $0x18,%rsp
  400a20:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a25:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a2a:	48 89 c7             	mov    %rax,%rdi
  400a2d:	e8 e2 ff ff ff       	callq  400a14 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400a32:	90                   	nop
  400a33:	48 83 c4 18          	add    $0x18,%rsp
  400a37:	c3                   	retq   

0000000000400a38 <_ZN9TestClassC1Ev>:
  400a38:	48 83 ec 18          	sub    $0x18,%rsp
  400a3c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a41:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a46:	48 89 c7             	mov    %rax,%rdi
  400a49:	e8 ce ff ff ff       	callq  400a1c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEC1Ev>
  400a4e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a53:	48 89 c7             	mov    %rax,%rdi
  400a56:	e8 27 01 00 00       	callq  400b82 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC1Ev>
  400a5b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a60:	48 89 c7             	mov    %rax,%rdi
  400a63:	e8 22 01 00 00       	callq  400b8a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC1Ev>
  400a68:	90                   	nop
  400a69:	48 83 c4 18          	add    $0x18,%rsp
  400a6d:	c3                   	retq   

0000000000400a6e <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEmLEi>:
  400a6e:	48 83 ec 28          	sub    $0x28,%rsp
  400a72:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400a77:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400a7c:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400a80:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a85:	48 89 c7             	mov    %rax,%rdi
  400a88:	e8 9f 01 00 00       	callq  400c2c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a8d:	8b 10                	mov    (%rax),%edx
  400a8f:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400a94:	89 10                	mov    %edx,(%rax)
  400a96:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400a9b:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400aa0:	48 89 d6             	mov    %rdx,%rsi
  400aa3:	48 89 c7             	mov    %rax,%rdi
  400aa6:	e8 9d 01 00 00       	callq  400c48 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC1ERKS3_>
  400aab:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ab0:	48 83 c4 28          	add    $0x28,%rsp
  400ab4:	c3                   	retq   
  400ab5:	90                   	nop

0000000000400ab6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEmLEi>:
  400ab6:	48 83 ec 28          	sub    $0x28,%rsp
  400aba:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400abf:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400ac4:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ac8:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400acd:	48 89 c7             	mov    %rax,%rdi
  400ad0:	e8 7f 01 00 00       	callq  400c54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400ad5:	8b 10                	mov    (%rax),%edx
  400ad7:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400adc:	89 10                	mov    %edx,(%rax)
  400ade:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400ae3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ae8:	48 89 d6             	mov    %rdx,%rsi
  400aeb:	48 89 c7             	mov    %rax,%rdi
  400aee:	e8 7d 01 00 00       	callq  400c70 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC1ERKS3_>
  400af3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400af8:	48 83 c4 28          	add    $0x28,%rsp
  400afc:	c3                   	retq   

0000000000400afd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>:
  400afd:	48 83 ec 08          	sub    $0x8,%rsp
  400b01:	e8 08 ff ff ff       	callq  400a0e <_ZN9TestClass14storage_bufferEv>
  400b06:	48 83 c4 08          	add    $0x8,%rsp
  400b0a:	c3                   	retq   

0000000000400b0b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_>:
  400b0b:	41 54                	push   %r12
  400b0d:	55                   	push   %rbp
  400b0e:	53                   	push   %rbx
  400b0f:	e8 e9 ff ff ff       	callq  400afd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400b14:	48 89 c5             	mov    %rax,%rbp
  400b17:	48 89 ee             	mov    %rbp,%rsi
  400b1a:	bf 20 00 00 00       	mov    $0x20,%edi
  400b1f:	e8 06 fe ff ff       	callq  40092a <_ZnwmPv>
  400b24:	48 89 c3             	mov    %rax,%rbx
  400b27:	48 85 db             	test   %rbx,%rbx
  400b2a:	74 23                	je     400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b2c:	48 89 df             	mov    %rbx,%rdi
  400b2f:	e8 48 01 00 00       	callq  400c7c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400b34:	eb 19                	jmp    400b4f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIIEEEvDpT_+0x44>
  400b36:	49 89 c4             	mov    %rax,%r12
  400b39:	48 89 ee             	mov    %rbp,%rsi
  400b3c:	48 89 df             	mov    %rbx,%rdi
  400b3f:	e8 f6 fd ff ff       	callq  40093a <_ZdlPvS_>
  400b44:	4c 89 e0             	mov    %r12,%rax
  400b47:	48 89 c7             	mov    %rax,%rdi
  400b4a:	e8 31 fb ff ff       	callq  400680 <_Unwind_Resume@plt>
  400b4f:	90                   	nop
  400b50:	5b                   	pop    %rbx
  400b51:	5d                   	pop    %rbp
  400b52:	41 5c                	pop    %r12
  400b54:	c3                   	retq   

0000000000400b55 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  400b55:	48 83 ec 18          	sub    $0x18,%rsp
  400b59:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b5e:	e8 64 01 00 00       	callq  400cc7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400b63:	e8 95 ff ff ff       	callq  400afd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400b68:	be 01 00 00 00       	mov    $0x1,%esi
  400b6d:	48 89 c7             	mov    %rax,%rdi
  400b70:	e8 73 01 00 00       	callq  400ce8 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400b75:	48 89 c7             	mov    %rax,%rdi
  400b78:	e8 4c 01 00 00       	callq  400cc9 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  400b7d:	48 83 c4 18          	add    $0x18,%rsp
  400b81:	c3                   	retq   

0000000000400b82 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC1Ev>:
  400b82:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b87:	90                   	nop
  400b88:	c3                   	retq   
  400b89:	90                   	nop

0000000000400b8a <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC1Ev>:
  400b8a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b8f:	90                   	nop
  400b90:	c3                   	retq   

0000000000400b91 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400b91:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b96:	90                   	nop
  400b97:	c3                   	retq   

0000000000400b98 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEaSEi>:
  400b98:	48 83 ec 18          	sub    $0x18,%rsp
  400b9c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400ba1:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400ba5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400baa:	48 89 c7             	mov    %rax,%rdi
  400bad:	e8 7a 00 00 00       	callq  400c2c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bb2:	48 89 c2             	mov    %rax,%rdx
  400bb5:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400bb9:	89 02                	mov    %eax,(%rdx)
  400bbb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bc0:	48 83 c4 18          	add    $0x18,%rsp
  400bc4:	c3                   	retq   
  400bc5:	90                   	nop

0000000000400bc6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEaSEi>:
  400bc6:	48 83 ec 18          	sub    $0x18,%rsp
  400bca:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bcf:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400bd3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bd8:	48 89 c7             	mov    %rax,%rdi
  400bdb:	e8 74 00 00 00       	callq  400c54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400be0:	48 89 c2             	mov    %rax,%rdx
  400be3:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400be7:	89 02                	mov    %eax,(%rdx)
  400be9:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bee:	48 83 c4 18          	add    $0x18,%rsp
  400bf2:	c3                   	retq   
  400bf3:	90                   	nop

0000000000400bf4 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEcvRiEv>:
  400bf4:	48 83 ec 18          	sub    $0x18,%rsp
  400bf8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400bfd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c02:	48 89 c7             	mov    %rax,%rdi
  400c05:	e8 22 00 00 00       	callq  400c2c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c0a:	48 83 c4 18          	add    $0x18,%rsp
  400c0e:	c3                   	retq   
  400c0f:	90                   	nop

0000000000400c10 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEcvRiEv>:
  400c10:	48 83 ec 18          	sub    $0x18,%rsp
  400c14:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c19:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c1e:	48 89 c7             	mov    %rax,%rdi
  400c21:	e8 2e 00 00 00       	callq  400c54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c26:	48 83 c4 18          	add    $0x18,%rsp
  400c2a:	c3                   	retq   
  400c2b:	90                   	nop

0000000000400c2c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c2c:	48 83 ec 18          	sub    $0x18,%rsp
  400c30:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c35:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c3a:	48 89 c7             	mov    %rax,%rdi
  400c3d:	e8 dc 00 00 00       	callq  400d1e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400c42:	48 83 c4 18          	add    $0x18,%rsp
  400c46:	c3                   	retq   
  400c47:	90                   	nop

0000000000400c48 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassEC1ERKS3_>:
  400c48:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c4d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c52:	90                   	nop
  400c53:	c3                   	retq   

0000000000400c54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c54:	48 83 ec 18          	sub    $0x18,%rsp
  400c58:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c5d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c62:	48 89 c7             	mov    %rax,%rdi
  400c65:	e8 ea 00 00 00       	callq  400d54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400c6a:	48 83 c4 18          	add    $0x18,%rsp
  400c6e:	c3                   	retq   
  400c6f:	90                   	nop

0000000000400c70 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassEC1ERKS3_>:
  400c70:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c75:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c7a:	90                   	nop
  400c7b:	c3                   	retq   

0000000000400c7c <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400c7c:	48 83 ec 18          	sub    $0x18,%rsp
  400c80:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c85:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c8a:	48 89 c7             	mov    %rax,%rdi
  400c8d:	e8 fa 00 00 00       	callq  400d8c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400c92:	bf c0 91 00 00       	mov    $0x91c0,%edi
  400c97:	e8 a4 f9 ff ff       	callq  400640 <malloc@plt>
  400c9c:	48 89 c2             	mov    %rax,%rdx
  400c9f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca4:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400ca8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cad:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
  400cb4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cb9:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400cc0:	00 
  400cc1:	90                   	nop
  400cc2:	48 83 c4 18          	add    $0x18,%rsp
  400cc6:	c3                   	retq   

0000000000400cc7 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400cc7:	90                   	nop
  400cc8:	c3                   	retq   

0000000000400cc9 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400cc9:	48 83 ec 18          	sub    $0x18,%rsp
  400ccd:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cd2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cd7:	48 83 c0 01          	add    $0x1,%rax
  400cdb:	48 89 c7             	mov    %rax,%rdi
  400cde:	e8 bc 00 00 00       	callq  400d9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400ce3:	48 83 c4 18          	add    $0x18,%rsp
  400ce7:	c3                   	retq   

0000000000400ce8 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400ce8:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400ced:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400cf2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400cf7:	48 8b 00             	mov    (%rax),%rax
  400cfa:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400cff:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d04:	48 8b 10             	mov    (%rax),%rdx
  400d07:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400d0c:	48 01 c2             	add    %rax,%rdx
  400d0f:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400d14:	48 89 10             	mov    %rdx,(%rax)
  400d17:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d1c:	c3                   	retq   
  400d1d:	90                   	nop

0000000000400d1e <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400d1e:	48 83 ec 28          	sub    $0x28,%rsp
  400d22:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d27:	e8 d1 fd ff ff       	callq  400afd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400d2c:	48 89 c7             	mov    %rax,%rdi
  400d2f:	e8 76 00 00 00       	callq  400daa <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400d34:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d39:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d3e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400d45:	00 
  400d46:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d4b:	48 01 d0             	add    %rdx,%rax
  400d4e:	48 83 c4 28          	add    $0x28,%rsp
  400d52:	c3                   	retq   
  400d53:	90                   	nop

0000000000400d54 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400d54:	48 83 ec 28          	sub    $0x28,%rsp
  400d58:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d5d:	e8 9b fd ff ff       	callq  400afd <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400d62:	48 89 c7             	mov    %rax,%rdi
  400d65:	e8 40 00 00 00       	callq  400daa <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400d6a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d6f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d74:	48 c1 e0 02          	shl    $0x2,%rax
  400d78:	48 8d 90 e0 48 00 00 	lea    0x48e0(%rax),%rdx
  400d7f:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d84:	48 01 d0             	add    %rdx,%rax
  400d87:	48 83 c4 28          	add    $0x28,%rsp
  400d8b:	c3                   	retq   

0000000000400d8c <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400d8c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d91:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400d96:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400d9d:	90                   	nop
  400d9e:	c3                   	retq   

0000000000400d9f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400d9f:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400da4:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400da9:	c3                   	retq   

0000000000400daa <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400daa:	48 83 ec 18          	sub    $0x18,%rsp
  400dae:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400db3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400db8:	48 89 c7             	mov    %rax,%rdi
  400dbb:	e8 06 00 00 00       	callq  400dc6 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400dc0:	48 83 c4 18          	add    $0x18,%rsp
  400dc4:	c3                   	retq   
  400dc5:	90                   	nop

0000000000400dc6 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400dc6:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dcb:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dd0:	48 8b 40 18          	mov    0x18(%rax),%rax
  400dd4:	c3                   	retq   
  400dd5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ddc:	00 00 00 
  400ddf:	90                   	nop

0000000000400de0 <__libc_csu_init>:
  400de0:	41 57                	push   %r15
  400de2:	41 56                	push   %r14
  400de4:	41 89 ff             	mov    %edi,%r15d
  400de7:	41 55                	push   %r13
  400de9:	41 54                	push   %r12
  400deb:	4c 8d 25 fe 0f 20 00 	lea    0x200ffe(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400df2:	55                   	push   %rbp
  400df3:	48 8d 2d fe 0f 20 00 	lea    0x200ffe(%rip),%rbp        # 601df8 <__init_array_end>
  400dfa:	53                   	push   %rbx
  400dfb:	49 89 f6             	mov    %rsi,%r14
  400dfe:	49 89 d5             	mov    %rdx,%r13
  400e01:	4c 29 e5             	sub    %r12,%rbp
  400e04:	48 83 ec 08          	sub    $0x8,%rsp
  400e08:	48 c1 fd 03          	sar    $0x3,%rbp
  400e0c:	e8 e7 f7 ff ff       	callq  4005f8 <_init>
  400e11:	48 85 ed             	test   %rbp,%rbp
  400e14:	74 20                	je     400e36 <__libc_csu_init+0x56>
  400e16:	31 db                	xor    %ebx,%ebx
  400e18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e1f:	00 
  400e20:	4c 89 ea             	mov    %r13,%rdx
  400e23:	4c 89 f6             	mov    %r14,%rsi
  400e26:	44 89 ff             	mov    %r15d,%edi
  400e29:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e2d:	48 83 c3 01          	add    $0x1,%rbx
  400e31:	48 39 eb             	cmp    %rbp,%rbx
  400e34:	75 ea                	jne    400e20 <__libc_csu_init+0x40>
  400e36:	48 83 c4 08          	add    $0x8,%rsp
  400e3a:	5b                   	pop    %rbx
  400e3b:	5d                   	pop    %rbp
  400e3c:	41 5c                	pop    %r12
  400e3e:	41 5d                	pop    %r13
  400e40:	41 5e                	pop    %r14
  400e42:	41 5f                	pop    %r15
  400e44:	c3                   	retq   
  400e45:	90                   	nop
  400e46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e4d:	00 00 00 

0000000000400e50 <__libc_csu_fini>:
  400e50:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e54 <_fini>:
  400e54:	48 83 ec 08          	sub    $0x8,%rsp
  400e58:	48 83 c4 08          	add    $0x8,%rsp
  400e5c:	c3                   	retq   
