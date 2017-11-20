
bin/clang++-3.8_-O0_DynamicStorage_0:     file format elf64-x86-64


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
  40064f:	49 c7 c0 50 0e 40 00 	mov    $0x400e50,%r8
  400656:	48 c7 c1 e0 0d 40 00 	mov    $0x400de0,%rcx
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
  40075c:	e8 4f 02 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400761:	31 c0                	xor    %eax,%eax
  400763:	89 c7                	mov    %eax,%edi
  400765:	e8 56 02 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40076a:	48 89 c7             	mov    %rax,%rdi
  40076d:	48 89 c1             	mov    %rax,%rcx
  400770:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400775:	48 89 c7             	mov    %rax,%rdi
  400778:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40077d:	e8 6e 02 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400782:	e9 00 00 00 00       	jmpq   400787 <main+0x37>
  400787:	be 77 77 00 00       	mov    $0x7777,%esi
  40078c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400791:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400796:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40079b:	e8 90 02 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4007a0:	be 88 88 00 00       	mov    $0x8888,%esi
  4007a5:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007aa:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4007af:	e8 bc 02 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  4007b4:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007b9:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007be:	e8 ed 02 00 00       	callq  400ab0 <_ZN9TestClass15increase_field0Ev>
  4007c3:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007c8:	e8 13 03 00 00       	callq  400ae0 <_ZN9TestClass15increase_field1Ev>
  4007cd:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007d2:	e8 39 03 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  4007d7:	8b 30                	mov    (%rax),%esi
  4007d9:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007de:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007e2:	e8 39 03 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  4007e7:	48 bf 64 0e 40 00 00 	movabs $0x400e64,%rdi
  4007ee:	00 00 00 
  4007f1:	8b 10                	mov    (%rax),%edx
  4007f3:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007f7:	b0 00                	mov    $0x0,%al
  4007f9:	e8 e2 fd ff ff       	callq  4005e0 <printf@plt>
  4007fe:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400803:	89 44 24 10          	mov    %eax,0x10(%rsp)
  400807:	e8 04 03 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40080c:	31 d2                	xor    %edx,%edx
  40080e:	88 d1                	mov    %dl,%cl
  400810:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400816:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40081a:	0f 85 17 00 00 00    	jne    400837 <main+0xe7>
  400820:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400825:	e8 f6 02 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40082a:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400830:	0f 94 c1             	sete   %cl
  400833:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  400837:	8a 44 24 0f          	mov    0xf(%rsp),%al
  40083b:	34 ff                	xor    $0xff,%al
  40083d:	24 01                	and    $0x1,%al
  40083f:	0f b6 c0             	movzbl %al,%eax
  400842:	48 83 c4 58          	add    $0x58,%rsp
  400846:	c3                   	retq   
  400847:	89 d1                	mov    %edx,%ecx
  400849:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40084e:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400852:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  400857:	e8 c4 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  40085c:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400861:	e8 ba fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400866:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40086d:	00 00 00 

0000000000400870 <_Z12new_instancev>:
  400870:	48 83 ec 28          	sub    $0x28,%rsp
  400874:	31 c0                	xor    %eax,%eax
  400876:	89 c7                	mov    %eax,%edi
  400878:	e8 43 01 00 00       	callq  4009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40087d:	48 89 c7             	mov    %rax,%rdi
  400880:	48 89 c1             	mov    %rax,%rcx
  400883:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400888:	48 89 c7             	mov    %rax,%rdi
  40088b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400890:	e8 5b 01 00 00       	callq  4009f0 <_ZN9TestClassC2Ev>
  400895:	e9 00 00 00 00       	jmpq   40089a <_Z12new_instancev+0x2a>
  40089a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40089f:	48 83 c4 28          	add    $0x28,%rsp
  4008a3:	c3                   	retq   
  4008a4:	89 d1                	mov    %edx,%ecx
  4008a6:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  4008ab:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  4008af:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008b4:	e8 67 01 00 00       	callq  400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  4008b9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008be:	e8 5d fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008c3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008ca:	84 00 00 00 00 00 

00000000004008d0 <_Z12write_field0P9TestClass>:
  4008d0:	48 83 ec 18          	sub    $0x18,%rsp
  4008d4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008d9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008de:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008e3:	e8 48 01 00 00       	callq  400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>
  4008e8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008ed:	48 83 c4 18          	add    $0x18,%rsp
  4008f1:	c3                   	retq   
  4008f2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008f9:	1f 84 00 00 00 00 00 

0000000000400900 <_Z11read_field0P9TestClass>:
  400900:	50                   	push   %rax
  400901:	48 89 3c 24          	mov    %rdi,(%rsp)
  400905:	48 8b 3c 24          	mov    (%rsp),%rdi
  400909:	e8 02 02 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>
  40090e:	8b 00                	mov    (%rax),%eax
  400910:	59                   	pop    %rcx
  400911:	c3                   	retq   
  400912:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400919:	1f 84 00 00 00 00 00 

0000000000400920 <_Z12write_field1P9TestClass>:
  400920:	48 83 ec 18          	sub    $0x18,%rsp
  400924:	be 77 77 00 00       	mov    $0x7777,%esi
  400929:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40092e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400933:	e8 38 01 00 00       	callq  400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>
  400938:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40093d:	48 83 c4 18          	add    $0x18,%rsp
  400941:	c3                   	retq   
  400942:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400949:	1f 84 00 00 00 00 00 

0000000000400950 <_Z11read_field1P9TestClass>:
  400950:	50                   	push   %rax
  400951:	48 89 3c 24          	mov    %rdi,(%rsp)
  400955:	48 8b 3c 24          	mov    (%rsp),%rdi
  400959:	e8 c2 01 00 00       	callq  400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>
  40095e:	8b 00                	mov    (%rax),%eax
  400960:	59                   	pop    %rcx
  400961:	c3                   	retq   
  400962:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400969:	1f 84 00 00 00 00 00 

0000000000400970 <_Z21explicit_write_field0m>:
  400970:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400975:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40097a:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  400981:	77 77 00 00 
  400985:	c3                   	retq   
  400986:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40098d:	00 00 00 

0000000000400990 <_Z20explicit_read_field0m>:
  400990:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400995:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40099a:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  4009a1:	c3                   	retq   
  4009a2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4009a9:	00 00 00 
  4009ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009b0:	50                   	push   %rax
  4009b1:	e8 8a fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009b6:	48 89 c7             	mov    %rax,%rdi
  4009b9:	e8 32 03 00 00       	callq  400cf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009be:	58                   	pop    %rax
  4009bf:	c3                   	retq   

00000000004009c0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009c0:	50                   	push   %rax
  4009c1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009c5:	e8 a6 03 00 00       	callq  400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009ca:	e8 71 fd ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  4009cf:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009d4:	89 ce                	mov    %ecx,%esi
  4009d6:	48 89 c7             	mov    %rax,%rdi
  4009d9:	e8 c2 03 00 00       	callq  400da0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009de:	48 89 c7             	mov    %rax,%rdi
  4009e1:	e8 9a 03 00 00       	callq  400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009e6:	59                   	pop    %rcx
  4009e7:	c3                   	retq   
  4009e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009ef:	00 

00000000004009f0 <_ZN9TestClassC2Ev>:
  4009f0:	48 83 ec 18          	sub    $0x18,%rsp
  4009f4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009f9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009fe:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400a03:	e8 28 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>
  400a08:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400a0d:	e8 2e 01 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>
  400a12:	48 83 c4 18          	add    $0x18,%rsp
  400a16:	c3                   	retq   
  400a17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a1e:	00 00 

0000000000400a20 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a25:	c3                   	retq   
  400a26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a2d:	00 00 00 

0000000000400a30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEaSEi>:
  400a30:	48 83 ec 28          	sub    $0x28,%rsp
  400a34:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a39:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a3d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a42:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a46:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a4b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a4f:	e8 5c 01 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a54:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a58:	89 30                	mov    %esi,(%rax)
  400a5a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a5f:	48 83 c4 28          	add    $0x28,%rsp
  400a63:	c3                   	retq   
  400a64:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a6b:	00 00 00 
  400a6e:	66 90                	xchg   %ax,%ax

0000000000400a70 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEaSEi>:
  400a70:	48 83 ec 28          	sub    $0x28,%rsp
  400a74:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a79:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a7d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a82:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a86:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a8b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a8f:	e8 fc 01 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a94:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a98:	89 30                	mov    %esi,(%rax)
  400a9a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a9f:	48 83 c4 28          	add    $0x28,%rsp
  400aa3:	c3                   	retq   
  400aa4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400aab:	00 00 00 
  400aae:	66 90                	xchg   %ax,%ax

0000000000400ab0 <_ZN9TestClass15increase_field0Ev>:
  400ab0:	48 83 ec 18          	sub    $0x18,%rsp
  400ab4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ab9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400abe:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ac3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400ac8:	48 89 c7             	mov    %rax,%rdi
  400acb:	e8 80 00 00 00       	callq  400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>
  400ad0:	48 83 c4 18          	add    $0x18,%rsp
  400ad4:	c3                   	retq   
  400ad5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400adc:	00 00 00 
  400adf:	90                   	nop

0000000000400ae0 <_ZN9TestClass15increase_field1Ev>:
  400ae0:	48 83 ec 18          	sub    $0x18,%rsp
  400ae4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ae9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400aee:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400af3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400af8:	48 89 c7             	mov    %rax,%rdi
  400afb:	e8 30 01 00 00       	callq  400c30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>
  400b00:	48 83 c4 18          	add    $0x18,%rsp
  400b04:	c3                   	retq   
  400b05:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b0c:	00 00 00 
  400b0f:	90                   	nop

0000000000400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEcvRiEv>:
  400b10:	50                   	push   %rax
  400b11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b19:	e8 92 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b1e:	59                   	pop    %rcx
  400b1f:	c3                   	retq   

0000000000400b20 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEcvRiEv>:
  400b20:	50                   	push   %rax
  400b21:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b25:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b29:	e8 62 01 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b2e:	59                   	pop    %rcx
  400b2f:	c3                   	retq   

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2Ev>:
  400b30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b35:	c3                   	retq   
  400b36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b3d:	00 00 00 

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2Ev>:
  400b40:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b45:	c3                   	retq   
  400b46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b4d:	00 00 00 

0000000000400b50 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEmLEi>:
  400b50:	48 83 ec 38          	sub    $0x38,%rsp
  400b54:	48 89 f8             	mov    %rdi,%rax
  400b57:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b5c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b60:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b65:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b69:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b6e:	48 89 f7             	mov    %rsi,%rdi
  400b71:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b76:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b7a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b7f:	e8 2c 00 00 00       	callq  400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b84:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b88:	0f af 10             	imul   (%rax),%edx
  400b8b:	89 10                	mov    %edx,(%rax)
  400b8d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b92:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b97:	e8 24 00 00 00       	callq  400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>
  400b9c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400ba1:	48 83 c4 38          	add    $0x38,%rsp
  400ba5:	c3                   	retq   
  400ba6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400bad:	00 00 00 

0000000000400bb0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bb0:	50                   	push   %rax
  400bb1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bb5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bb9:	e8 12 00 00 00       	callq  400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400bbe:	59                   	pop    %rcx
  400bbf:	c3                   	retq   

0000000000400bc0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassEC2ERKS3_>:
  400bc0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bc5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bca:	c3                   	retq   
  400bcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bd0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400bd0:	48 83 ec 18          	sub    $0x18,%rsp
  400bd4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400bd9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400bde:	48 89 3c 24          	mov    %rdi,(%rsp)
  400be2:	e8 59 fb ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400be7:	48 89 c7             	mov    %rax,%rdi
  400bea:	e8 21 00 00 00       	callq  400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400bef:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400bf4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400bf9:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bfd:	48 c1 e7 02          	shl    $0x2,%rdi
  400c01:	48 01 f8             	add    %rdi,%rax
  400c04:	48 83 c4 18          	add    $0x18,%rsp
  400c08:	c3                   	retq   
  400c09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400c10:	50                   	push   %rax
  400c11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c19:	e8 02 00 00 00       	callq  400c20 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c1e:	59                   	pop    %rcx
  400c1f:	c3                   	retq   

0000000000400c20 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c25:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c2a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c2e:	c3                   	retq   
  400c2f:	90                   	nop

0000000000400c30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEmLEi>:
  400c30:	48 83 ec 38          	sub    $0x38,%rsp
  400c34:	48 89 f8             	mov    %rdi,%rax
  400c37:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c3c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c40:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c45:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c49:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c4e:	48 89 f7             	mov    %rsi,%rdi
  400c51:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c56:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c5a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c5f:	e8 2c 00 00 00       	callq  400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c64:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c68:	0f af 10             	imul   (%rax),%edx
  400c6b:	89 10                	mov    %edx,(%rax)
  400c6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c72:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c77:	e8 24 00 00 00       	callq  400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>
  400c7c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c81:	48 83 c4 38          	add    $0x38,%rsp
  400c85:	c3                   	retq   
  400c86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c8d:	00 00 00 

0000000000400c90 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400c90:	50                   	push   %rax
  400c91:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c95:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c99:	e8 12 00 00 00       	callq  400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c9e:	59                   	pop    %rcx
  400c9f:	c3                   	retq   

0000000000400ca0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassEC2ERKS3_>:
  400ca0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400ca5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400caa:	c3                   	retq   
  400cab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cb0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0E9TestClassE22data_ptr_uninitializedILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400cb0:	48 83 ec 18          	sub    $0x18,%rsp
  400cb4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400cb9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cbe:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cc2:	e8 79 fa ff ff       	callq  400740 <_ZN9TestClass7storageEv>
  400cc7:	48 89 c7             	mov    %rax,%rdi
  400cca:	e8 41 ff ff ff       	callq  400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400ccf:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400cd4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400cd9:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400cdf:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ce3:	48 c1 e7 02          	shl    $0x2,%rdi
  400ce7:	48 01 f8             	add    %rdi,%rax
  400cea:	48 83 c4 18          	add    $0x18,%rsp
  400cee:	c3                   	retq   
  400cef:	90                   	nop

0000000000400cf0 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400cf0:	48 83 ec 18          	sub    $0x18,%rsp
  400cf4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400cf9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cfe:	48 89 f8             	mov    %rdi,%rax
  400d01:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d06:	48 89 c7             	mov    %rax,%rdi
  400d09:	e8 32 00 00 00       	callq  400d40 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d0e:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400d13:	89 cf                	mov    %ecx,%edi
  400d15:	e8 d6 f8 ff ff       	callq  4005f0 <malloc@plt>
  400d1a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400d1f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400d23:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400d2a:	00 
  400d2b:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400d32:	00 
  400d33:	48 83 c4 18          	add    $0x18,%rsp
  400d37:	c3                   	retq   
  400d38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d3f:	00 

0000000000400d40 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400d40:	50                   	push   %rax
  400d41:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d45:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d49:	e8 02 00 00 00       	callq  400d50 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d4e:	58                   	pop    %rax
  400d4f:	c3                   	retq   

0000000000400d50 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d50:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d55:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d5a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d61:	c3                   	retq   
  400d62:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d69:	00 00 00 
  400d6c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d70 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d70:	c3                   	retq   
  400d71:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d78:	00 00 00 
  400d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400d80:	50                   	push   %rax
  400d81:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d85:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d89:	48 83 c7 01          	add    $0x1,%rdi
  400d8d:	e8 3e 00 00 00       	callq  400dd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400d92:	59                   	pop    %rcx
  400d93:	c3                   	retq   
  400d94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d9b:	00 00 00 
  400d9e:	66 90                	xchg   %ax,%ax

0000000000400da0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400da0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400da5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400daa:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400daf:	48 8b 3e             	mov    (%rsi),%rdi
  400db2:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400db7:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400dbc:	48 03 3e             	add    (%rsi),%rdi
  400dbf:	48 89 3e             	mov    %rdi,(%rsi)
  400dc2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400dc7:	c3                   	retq   
  400dc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400dcf:	00 

0000000000400dd0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400dd0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400dd5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dda:	c3                   	retq   
  400ddb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400de0 <__libc_csu_init>:
  400de0:	41 57                	push   %r15
  400de2:	41 56                	push   %r14
  400de4:	41 89 ff             	mov    %edi,%r15d
  400de7:	41 55                	push   %r13
  400de9:	41 54                	push   %r12
  400deb:	4c 8d 25 ee 0f 20 00 	lea    0x200fee(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400df2:	55                   	push   %rbp
  400df3:	48 8d 2d ee 0f 20 00 	lea    0x200fee(%rip),%rbp        # 601de8 <__init_array_end>
  400dfa:	53                   	push   %rbx
  400dfb:	49 89 f6             	mov    %rsi,%r14
  400dfe:	49 89 d5             	mov    %rdx,%r13
  400e01:	4c 29 e5             	sub    %r12,%rbp
  400e04:	48 83 ec 08          	sub    $0x8,%rsp
  400e08:	48 c1 fd 03          	sar    $0x3,%rbp
  400e0c:	e8 97 f7 ff ff       	callq  4005a8 <_init>
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
