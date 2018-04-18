
bin/g++_-O0_DynamicStorage_0_Soa:     file format elf64-x86-64


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
  4006af:	49 c7 c0 c0 0e 40 00 	mov    $0x400ec0,%r8
  4006b6:	48 c7 c1 50 0e 40 00 	mov    $0x400e50,%rcx
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
  40079b:	e8 ab 03 00 00       	callq  400b4b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_>
  4007a0:	bf 00 00 00 00       	mov    $0x0,%edi
  4007a5:	e8 eb 03 00 00       	callq  400b95 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>
  4007aa:	48 89 c3             	mov    %rax,%rbx
  4007ad:	48 89 df             	mov    %rbx,%rdi
  4007b0:	e8 c3 02 00 00       	callq  400a78 <_ZN9TestClassC1Ev>
  4007b5:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  4007ba:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007bf:	be 77 77 00 00       	mov    $0x7777,%esi
  4007c4:	48 89 c7             	mov    %rax,%rdi
  4007c7:	e8 0c 04 00 00       	callq  400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>
  4007cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007d1:	be 88 88 00 00       	mov    $0x8888,%esi
  4007d6:	48 89 c7             	mov    %rax,%rdi
  4007d9:	e8 28 04 00 00       	callq  400c06 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>
  4007de:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007e3:	48 89 c7             	mov    %rax,%rdi
  4007e6:	e8 9b 01 00 00       	callq  400986 <_ZN9TestClass15increase_field0Ev>
  4007eb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007f0:	48 89 c7             	mov    %rax,%rdi
  4007f3:	e8 f2 01 00 00       	callq  4009ea <_ZN9TestClass15increase_field1Ev>
  4007f8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4007fd:	48 89 c7             	mov    %rax,%rdi
  400800:	e8 4b 04 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
  400805:	8b 18                	mov    (%rax),%ebx
  400807:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40080c:	48 89 c7             	mov    %rax,%rdi
  40080f:	e8 20 04 00 00       	callq  400c34 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
  400814:	8b 00                	mov    (%rax),%eax
  400816:	89 da                	mov    %ebx,%edx
  400818:	89 c6                	mov    %eax,%esi
  40081a:	bf f4 0e 40 00       	mov    $0x400ef4,%edi
  40081f:	b8 00 00 00 00       	mov    $0x0,%eax
  400824:	e8 07 fe ff ff       	callq  400630 <printf@plt>
  400829:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40082e:	48 89 c7             	mov    %rax,%rdi
  400831:	e8 fe 03 00 00       	callq  400c34 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
  400836:	8b 00                	mov    (%rax),%eax
  400838:	3d 83 2d d2 27       	cmp    $0x27d22d83,%eax
  40083d:	75 16                	jne    400855 <main+0xbf>
  40083f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400844:	48 89 c7             	mov    %rax,%rdi
  400847:	e8 04 04 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
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
  400870:	e8 20 03 00 00       	callq  400b95 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>
  400875:	48 89 c3             	mov    %rax,%rbx
  400878:	48 89 df             	mov    %rbx,%rdi
  40087b:	e8 f8 01 00 00       	callq  400a78 <_ZN9TestClassC1Ev>
  400880:	48 89 d8             	mov    %rbx,%rax
  400883:	5b                   	pop    %rbx
  400884:	c3                   	retq   

0000000000400885 <_Z12write_field0P9TestClass>:
  400885:	48 83 ec 18          	sub    $0x18,%rsp
  400889:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40088e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400893:	be 77 77 00 00       	mov    $0x7777,%esi
  400898:	48 89 c7             	mov    %rax,%rdi
  40089b:	e8 38 03 00 00       	callq  400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>
  4008a0:	90                   	nop
  4008a1:	48 83 c4 18          	add    $0x18,%rsp
  4008a5:	c3                   	retq   

00000000004008a6 <_Z11read_field0P9TestClass>:
  4008a6:	48 83 ec 18          	sub    $0x18,%rsp
  4008aa:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008af:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008b4:	48 89 c7             	mov    %rax,%rdi
  4008b7:	e8 78 03 00 00       	callq  400c34 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>
  4008bc:	8b 00                	mov    (%rax),%eax
  4008be:	48 83 c4 18          	add    $0x18,%rsp
  4008c2:	c3                   	retq   

00000000004008c3 <_Z12write_field1P9TestClass>:
  4008c3:	48 83 ec 18          	sub    $0x18,%rsp
  4008c7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008d1:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d6:	48 89 c7             	mov    %rax,%rdi
  4008d9:	e8 28 03 00 00       	callq  400c06 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>
  4008de:	90                   	nop
  4008df:	48 83 c4 18          	add    $0x18,%rsp
  4008e3:	c3                   	retq   

00000000004008e4 <_Z11read_field1P9TestClass>:
  4008e4:	48 83 ec 18          	sub    $0x18,%rsp
  4008e8:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4008ed:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4008f2:	48 89 c7             	mov    %rax,%rdi
  4008f5:	e8 56 03 00 00       	callq  400c50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>
  4008fa:	8b 00                	mov    (%rax),%eax
  4008fc:	48 83 c4 18          	add    $0x18,%rsp
  400900:	c3                   	retq   

0000000000400901 <_Z21explicit_write_field0m>:
  400901:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400906:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  40090b:	c7 04 85 40 b2 60 00 	movl   $0x7777,0x60b240(,%rax,4)
  400912:	77 77 00 00 
  400916:	90                   	nop
  400917:	c3                   	retq   

0000000000400918 <_Z20explicit_read_field0m>:
  400918:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40091d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400922:	8b 04 85 40 b2 60 00 	mov    0x60b240(,%rax,4),%eax
  400929:	c3                   	retq   

000000000040092a <_Z25explicit_write_field0_aosP16TestClassCompare>:
  40092a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40092f:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400934:	c7 00 77 77 00 00    	movl   $0x7777,(%rax)
  40093a:	90                   	nop
  40093b:	c3                   	retq   

000000000040093c <_Z24explicit_read_field0_aosP16TestClassCompare>:
  40093c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400941:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400946:	8b 00                	mov    (%rax),%eax
  400948:	c3                   	retq   

0000000000400949 <_Z25explicit_write_field1_aosP16TestClassCompare>:
  400949:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40094e:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400953:	c7 40 04 77 77 00 00 	movl   $0x7777,0x4(%rax)
  40095a:	90                   	nop
  40095b:	c3                   	retq   

000000000040095c <_Z24explicit_read_field1_aosP16TestClassCompare>:
  40095c:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400961:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400966:	8b 40 04             	mov    0x4(%rax),%eax
  400969:	c3                   	retq   

000000000040096a <_ZnwmPv>:
  40096a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40096f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400974:	48 8b 44 24 f0       	mov    -0x10(%rsp),%rax
  400979:	c3                   	retq   

000000000040097a <_ZdlPvS_>:
  40097a:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  40097f:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400984:	90                   	nop
  400985:	c3                   	retq   

0000000000400986 <_ZN9TestClass15increase_field0Ev>:
  400986:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  40098b:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  40098f:	41 ff 72 f8          	pushq  -0x8(%r10)
  400993:	55                   	push   %rbp
  400994:	48 89 e5             	mov    %rsp,%rbp
  400997:	41 52                	push   %r10
  400999:	48 83 ec 48          	sub    $0x48,%rsp
  40099d:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  4009a1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4009a8:	00 00 
  4009aa:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4009ae:	31 c0                	xor    %eax,%eax
  4009b0:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4009b4:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4009b8:	ba 55 55 00 00       	mov    $0x5555,%edx
  4009bd:	48 89 ce             	mov    %rcx,%rsi
  4009c0:	48 89 c7             	mov    %rax,%rdi
  4009c3:	e8 e6 00 00 00       	callq  400aae <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEmLEi>
  4009c8:	90                   	nop
  4009c9:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4009cd:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4009d4:	00 00 
  4009d6:	74 05                	je     4009dd <_ZN9TestClass15increase_field0Ev+0x57>
  4009d8:	e8 83 fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  4009dd:	48 83 c4 48          	add    $0x48,%rsp
  4009e1:	41 5a                	pop    %r10
  4009e3:	5d                   	pop    %rbp
  4009e4:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  4009e8:	c3                   	retq   
  4009e9:	90                   	nop

00000000004009ea <_ZN9TestClass15increase_field1Ev>:
  4009ea:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
  4009ef:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
  4009f3:	41 ff 72 f8          	pushq  -0x8(%r10)
  4009f7:	55                   	push   %rbp
  4009f8:	48 89 e5             	mov    %rsp,%rbp
  4009fb:	41 52                	push   %r10
  4009fd:	48 83 ec 48          	sub    $0x48,%rsp
  400a01:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
  400a05:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400a0c:	00 00 
  400a0e:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  400a12:	31 c0                	xor    %eax,%eax
  400a14:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  400a18:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  400a1c:	ba 44 44 00 00       	mov    $0x4444,%edx
  400a21:	48 89 ce             	mov    %rcx,%rsi
  400a24:	48 89 c7             	mov    %rax,%rdi
  400a27:	e8 ca 00 00 00       	callq  400af6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEmLEi>
  400a2c:	90                   	nop
  400a2d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  400a31:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400a38:	00 00 
  400a3a:	74 05                	je     400a41 <_ZN9TestClass15increase_field1Ev+0x57>
  400a3c:	e8 1f fc ff ff       	callq  400660 <__stack_chk_fail@plt>
  400a41:	48 83 c4 48          	add    $0x48,%rsp
  400a45:	41 5a                	pop    %r10
  400a47:	5d                   	pop    %rbp
  400a48:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
  400a4c:	c3                   	retq   
  400a4d:	90                   	nop

0000000000400a4e <_ZN9TestClass14storage_bufferEv>:
  400a4e:	b8 80 20 60 00       	mov    $0x602080,%eax
  400a53:	c3                   	retq   

0000000000400a54 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>:
  400a54:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a59:	90                   	nop
  400a5a:	c3                   	retq   
  400a5b:	90                   	nop

0000000000400a5c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEC1Ev>:
  400a5c:	48 83 ec 18          	sub    $0x18,%rsp
  400a60:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a65:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a6a:	48 89 c7             	mov    %rax,%rdi
  400a6d:	e8 e2 ff ff ff       	callq  400a54 <_ZN4ikra3soa10SizeNDummyILm0EEC1Ev>
  400a72:	90                   	nop
  400a73:	48 83 c4 18          	add    $0x18,%rsp
  400a77:	c3                   	retq   

0000000000400a78 <_ZN9TestClassC1Ev>:
  400a78:	48 83 ec 18          	sub    $0x18,%rsp
  400a7c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a81:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a86:	48 89 c7             	mov    %rax,%rdi
  400a89:	e8 ce ff ff ff       	callq  400a5c <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEC1Ev>
  400a8e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400a93:	48 89 c7             	mov    %rax,%rdi
  400a96:	e8 27 01 00 00       	callq  400bc2 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC1Ev>
  400a9b:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400aa0:	48 89 c7             	mov    %rax,%rdi
  400aa3:	e8 22 01 00 00       	callq  400bca <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC1Ev>
  400aa8:	90                   	nop
  400aa9:	48 83 c4 18          	add    $0x18,%rsp
  400aad:	c3                   	retq   

0000000000400aae <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEmLEi>:
  400aae:	48 83 ec 28          	sub    $0x28,%rsp
  400ab2:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400ab7:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400abc:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400ac0:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400ac5:	48 89 c7             	mov    %rax,%rdi
  400ac8:	e8 9f 01 00 00       	callq  400c6c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400acd:	8b 10                	mov    (%rax),%edx
  400acf:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400ad4:	89 10                	mov    %edx,(%rax)
  400ad6:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400adb:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ae0:	48 89 d6             	mov    %rdx,%rsi
  400ae3:	48 89 c7             	mov    %rax,%rdi
  400ae6:	e8 9d 01 00 00       	callq  400c88 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC1ERKS3_>
  400aeb:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400af0:	48 83 c4 28          	add    $0x28,%rsp
  400af4:	c3                   	retq   
  400af5:	90                   	nop

0000000000400af6 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEmLEi>:
  400af6:	48 83 ec 28          	sub    $0x28,%rsp
  400afa:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  400aff:	48 89 74 24 10       	mov    %rsi,0x10(%rsp)
  400b04:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  400b08:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400b0d:	48 89 c7             	mov    %rax,%rdi
  400b10:	e8 7f 01 00 00       	callq  400c94 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400b15:	8b 10                	mov    (%rax),%edx
  400b17:	0f af 54 24 0c       	imul   0xc(%rsp),%edx
  400b1c:	89 10                	mov    %edx,(%rax)
  400b1e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  400b23:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b28:	48 89 d6             	mov    %rdx,%rsi
  400b2b:	48 89 c7             	mov    %rax,%rdi
  400b2e:	e8 7d 01 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC1ERKS3_>
  400b33:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b38:	48 83 c4 28          	add    $0x28,%rsp
  400b3c:	c3                   	retq   

0000000000400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>:
  400b3d:	48 83 ec 08          	sub    $0x8,%rsp
  400b41:	e8 08 ff ff ff       	callq  400a4e <_ZN9TestClass14storage_bufferEv>
  400b46:	48 83 c4 08          	add    $0x8,%rsp
  400b4a:	c3                   	retq   

0000000000400b4b <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_>:
  400b4b:	41 54                	push   %r12
  400b4d:	55                   	push   %rbp
  400b4e:	53                   	push   %rbx
  400b4f:	e8 e9 ff ff ff       	callq  400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400b54:	48 89 c5             	mov    %rax,%rbp
  400b57:	48 89 ee             	mov    %rbp,%rsi
  400b5a:	bf 20 00 00 00       	mov    $0x20,%edi
  400b5f:	e8 06 fe ff ff       	callq  40096a <_ZnwmPv>
  400b64:	48 89 c3             	mov    %rax,%rbx
  400b67:	48 85 db             	test   %rbx,%rbx
  400b6a:	74 23                	je     400b8f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400b6c:	48 89 df             	mov    %rbx,%rdi
  400b6f:	e8 48 01 00 00       	callq  400cbc <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>
  400b74:	eb 19                	jmp    400b8f <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18initialize_storageIIEEEvDpT_+0x44>
  400b76:	49 89 c4             	mov    %rax,%r12
  400b79:	48 89 ee             	mov    %rbp,%rsi
  400b7c:	48 89 df             	mov    %rbx,%rdi
  400b7f:	e8 f6 fd ff ff       	callq  40097a <_ZdlPvS_>
  400b84:	4c 89 e0             	mov    %r12,%rax
  400b87:	48 89 c7             	mov    %rax,%rdi
  400b8a:	e8 f1 fa ff ff       	callq  400680 <_Unwind_Resume@plt>
  400b8f:	90                   	nop
  400b90:	5b                   	pop    %rbx
  400b91:	5d                   	pop    %rbp
  400b92:	41 5c                	pop    %r12
  400b94:	c3                   	retq   

0000000000400b95 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEnwEm>:
  400b95:	48 83 ec 18          	sub    $0x18,%rsp
  400b99:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400b9e:	e8 64 01 00 00       	callq  400d07 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  400ba3:	e8 95 ff ff ff       	callq  400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400ba8:	be 01 00 00 00       	mov    $0x1,%esi
  400bad:	48 89 c7             	mov    %rax,%rdi
  400bb0:	e8 73 01 00 00       	callq  400d28 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>
  400bb5:	48 89 c7             	mov    %rax,%rdi
  400bb8:	e8 4c 01 00 00       	callq  400d09 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE3getEm>
  400bbd:	48 83 c4 18          	add    $0x18,%rsp
  400bc1:	c3                   	retq   

0000000000400bc2 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC1Ev>:
  400bc2:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bc7:	90                   	nop
  400bc8:	c3                   	retq   
  400bc9:	90                   	nop

0000000000400bca <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC1Ev>:
  400bca:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bcf:	90                   	nop
  400bd0:	c3                   	retq   

0000000000400bd1 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EEdlEPv>:
  400bd1:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bd6:	90                   	nop
  400bd7:	c3                   	retq   

0000000000400bd8 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEaSEi>:
  400bd8:	48 83 ec 18          	sub    $0x18,%rsp
  400bdc:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400be1:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400be5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bea:	48 89 c7             	mov    %rax,%rdi
  400bed:	e8 7a 00 00 00       	callq  400c6c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400bf2:	48 89 c2             	mov    %rax,%rdx
  400bf5:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400bf9:	89 02                	mov    %eax,(%rdx)
  400bfb:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c00:	48 83 c4 18          	add    $0x18,%rsp
  400c04:	c3                   	retq   
  400c05:	90                   	nop

0000000000400c06 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEaSEi>:
  400c06:	48 83 ec 18          	sub    $0x18,%rsp
  400c0a:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c0f:	89 74 24 04          	mov    %esi,0x4(%rsp)
  400c13:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c18:	48 89 c7             	mov    %rax,%rdi
  400c1b:	e8 74 00 00 00       	callq  400c94 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400c20:	48 89 c2             	mov    %rax,%rdx
  400c23:	8b 44 24 04          	mov    0x4(%rsp),%eax
  400c27:	89 02                	mov    %eax,(%rdx)
  400c29:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c2e:	48 83 c4 18          	add    $0x18,%rsp
  400c32:	c3                   	retq   
  400c33:	90                   	nop

0000000000400c34 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEcvRiEv>:
  400c34:	48 83 ec 18          	sub    $0x18,%rsp
  400c38:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c3d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c42:	48 89 c7             	mov    %rax,%rdi
  400c45:	e8 22 00 00 00       	callq  400c6c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400c4a:	48 83 c4 18          	add    $0x18,%rsp
  400c4e:	c3                   	retq   
  400c4f:	90                   	nop

0000000000400c50 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEcvRiEv>:
  400c50:	48 83 ec 18          	sub    $0x18,%rsp
  400c54:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c59:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c5e:	48 89 c7             	mov    %rax,%rdi
  400c61:	e8 2e 00 00 00       	callq  400c94 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>
  400c66:	48 83 c4 18          	add    $0x18,%rsp
  400c6a:	c3                   	retq   
  400c6b:	90                   	nop

0000000000400c6c <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE8data_ptrEv>:
  400c6c:	48 83 ec 18          	sub    $0x18,%rsp
  400c70:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c75:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400c7a:	48 89 c7             	mov    %rax,%rdi
  400c7d:	e8 e0 00 00 00       	callq  400d62 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400c82:	48 83 c4 18          	add    $0x18,%rsp
  400c86:	c3                   	retq   
  400c87:	90                   	nop

0000000000400c88 <_ZN4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassEC1ERKS3_>:
  400c88:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c8d:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400c92:	90                   	nop
  400c93:	c3                   	retq   

0000000000400c94 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE8data_ptrEv>:
  400c94:	48 83 ec 18          	sub    $0x18,%rsp
  400c98:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400c9d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ca2:	48 89 c7             	mov    %rax,%rdi
  400ca5:	e8 ee 00 00 00       	callq  400d98 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>
  400caa:	48 83 c4 18          	add    $0x18,%rsp
  400cae:	c3                   	retq   
  400caf:	90                   	nop

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassEC1ERKS3_>:
  400cb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cb5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cba:	90                   	nop
  400cbb:	c3                   	retq   

0000000000400cbc <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC1Ev>:
  400cbc:	48 83 ec 18          	sub    $0x18,%rsp
  400cc0:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400cc5:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cca:	48 89 c7             	mov    %rax,%rdi
  400ccd:	e8 fe 00 00 00       	callq  400dd0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>
  400cd2:	bf c0 91 00 00       	mov    $0x91c0,%edi
  400cd7:	e8 64 f9 ff ff       	callq  400640 <malloc@plt>
  400cdc:	48 89 c2             	mov    %rax,%rdx
  400cdf:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ce4:	48 89 50 18          	mov    %rdx,0x18(%rax)
  400ce8:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ced:	c7 40 10 00 00 00 00 	movl   $0x0,0x10(%rax)
  400cf4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cf9:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
  400d00:	00 
  400d01:	90                   	nop
  400d02:	48 83 c4 18          	add    $0x18,%rsp
  400d06:	c3                   	retq   

0000000000400d07 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d07:	90                   	nop
  400d08:	c3                   	retq   

0000000000400d09 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE3getEm>:
  400d09:	48 83 ec 18          	sub    $0x18,%rsp
  400d0d:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d12:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d17:	48 83 c0 01          	add    $0x1,%rax
  400d1b:	48 89 c7             	mov    %rax,%rdi
  400d1e:	e8 c0 00 00 00       	callq  400de3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE4get_ILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPS2_E4typeEm>
  400d23:	48 83 c4 18          	add    $0x18,%rsp
  400d27:	c3                   	retq   

0000000000400d28 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE13increase_sizeEm>:
  400d28:	48 83 ec 28          	sub    $0x28,%rsp
  400d2c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d31:	48 89 34 24          	mov    %rsi,(%rsp)
  400d35:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d3a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  400d3f:	48 8b 04 24          	mov    (%rsp),%rax
  400d43:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d48:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  400d4d:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400d52:	48 89 d6             	mov    %rdx,%rsi
  400d55:	48 89 c7             	mov    %rax,%rdi
  400d58:	e8 91 00 00 00       	callq  400dee <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>
  400d5d:	48 83 c4 28          	add    $0x28,%rsp
  400d61:	c3                   	retq   

0000000000400d62 <_ZNK4ikra3soa6Field_IiLm4663ELj0ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d62:	48 83 ec 28          	sub    $0x28,%rsp
  400d66:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d6b:	e8 cd fd ff ff       	callq  400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400d70:	48 89 c7             	mov    %rax,%rdi
  400d73:	e8 ac 00 00 00       	callq  400e24 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400d78:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400d7d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400d82:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  400d89:	00 
  400d8a:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400d8f:	48 01 d0             	add    %rdx,%rax
  400d92:	48 83 c4 28          	add    $0x28,%rsp
  400d96:	c3                   	retq   
  400d97:	90                   	nop

0000000000400d98 <_ZNK4ikra3soa6Field_IiLm4663ELj4ELi0ELi1ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1ELi1EEENSt9enable_ifIXaaaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEeqT1_L_ZNS0_L14kLayoutModeSoaEEEPiE4typeEv>:
  400d98:	48 83 ec 28          	sub    $0x28,%rsp
  400d9c:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400da1:	e8 97 fd ff ff       	callq  400b3d <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE7storageEv>
  400da6:	48 89 c7             	mov    %rax,%rdi
  400da9:	e8 76 00 00 00       	callq  400e24 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400dae:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400db3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400db8:	48 c1 e0 02          	shl    $0x2,%rax
  400dbc:	48 8d 90 e0 48 00 00 	lea    0x48e0(%rax),%rdx
  400dc3:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400dc8:	48 01 d0             	add    %rdx,%rax
  400dcb:	48 83 c4 28          	add    $0x28,%rsp
  400dcf:	c3                   	retq   

0000000000400dd0 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC1Ev>:
  400dd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dd5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dda:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  400de1:	90                   	nop
  400de2:	c3                   	retq   

0000000000400de3 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageELi1EE4get_ILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L14kLayoutModeSoaEEEPS2_E4typeEm>:
  400de3:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400de8:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400ded:	c3                   	retq   

0000000000400dee <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEE10atomic_addIyEET_PS7_S7_>:
  400dee:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400df3:	48 89 74 24 e0       	mov    %rsi,-0x20(%rsp)
  400df8:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400dfd:	48 8b 00             	mov    (%rax),%rax
  400e00:	48 89 44 24 f8       	mov    %rax,-0x8(%rsp)
  400e05:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e0a:	48 8b 10             	mov    (%rax),%rdx
  400e0d:	48 8b 44 24 e0       	mov    -0x20(%rsp),%rax
  400e12:	48 01 c2             	add    %rax,%rdx
  400e15:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400e1a:	48 89 10             	mov    %rdx,(%rax)
  400e1d:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e22:	c3                   	retq   
  400e23:	90                   	nop

0000000000400e24 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400e24:	48 83 ec 18          	sub    $0x18,%rsp
  400e28:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400e2d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400e32:	48 89 c7             	mov    %rax,%rdi
  400e35:	e8 06 00 00 00       	callq  400e40 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400e3a:	48 83 c4 18          	add    $0x18,%rsp
  400e3e:	c3                   	retq   
  400e3f:	90                   	nop

0000000000400e40 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400e40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400e45:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400e4a:	48 8b 40 18          	mov    0x18(%rax),%rax
  400e4e:	c3                   	retq   
  400e4f:	90                   	nop

0000000000400e50 <__libc_csu_init>:
  400e50:	41 57                	push   %r15
  400e52:	41 56                	push   %r14
  400e54:	41 89 ff             	mov    %edi,%r15d
  400e57:	41 55                	push   %r13
  400e59:	41 54                	push   %r12
  400e5b:	4c 8d 25 8e 0f 20 00 	lea    0x200f8e(%rip),%r12        # 601df0 <__frame_dummy_init_array_entry>
  400e62:	55                   	push   %rbp
  400e63:	48 8d 2d 8e 0f 20 00 	lea    0x200f8e(%rip),%rbp        # 601df8 <__init_array_end>
  400e6a:	53                   	push   %rbx
  400e6b:	49 89 f6             	mov    %rsi,%r14
  400e6e:	49 89 d5             	mov    %rdx,%r13
  400e71:	4c 29 e5             	sub    %r12,%rbp
  400e74:	48 83 ec 08          	sub    $0x8,%rsp
  400e78:	48 c1 fd 03          	sar    $0x3,%rbp
  400e7c:	e8 77 f7 ff ff       	callq  4005f8 <_init>
  400e81:	48 85 ed             	test   %rbp,%rbp
  400e84:	74 20                	je     400ea6 <__libc_csu_init+0x56>
  400e86:	31 db                	xor    %ebx,%ebx
  400e88:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e8f:	00 
  400e90:	4c 89 ea             	mov    %r13,%rdx
  400e93:	4c 89 f6             	mov    %r14,%rsi
  400e96:	44 89 ff             	mov    %r15d,%edi
  400e99:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e9d:	48 83 c3 01          	add    $0x1,%rbx
  400ea1:	48 39 eb             	cmp    %rbp,%rbx
  400ea4:	75 ea                	jne    400e90 <__libc_csu_init+0x40>
  400ea6:	48 83 c4 08          	add    $0x8,%rsp
  400eaa:	5b                   	pop    %rbx
  400eab:	5d                   	pop    %rbp
  400eac:	41 5c                	pop    %r12
  400eae:	41 5d                	pop    %r13
  400eb0:	41 5e                	pop    %r14
  400eb2:	41 5f                	pop    %r15
  400eb4:	c3                   	retq   
  400eb5:	90                   	nop
  400eb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400ebd:	00 00 00 

0000000000400ec0 <__libc_csu_fini>:
  400ec0:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400ec4 <_fini>:
  400ec4:	48 83 ec 08          	sub    $0x8,%rsp
  400ec8:	48 83 c4 08          	add    $0x8,%rsp
  400ecc:	c3                   	retq   
