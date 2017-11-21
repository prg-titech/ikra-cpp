
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
  40064f:	49 c7 c0 60 0e 40 00 	mov    $0x400e60,%r8
  400656:	48 c7 c1 f0 0d 40 00 	mov    $0x400df0,%rcx
  40065d:	48 c7 c7 40 07 40 00 	mov    $0x400740,%rdi
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

0000000000400740 <main>:
  400740:	48 83 ec 58          	sub    $0x58,%rsp
  400744:	c7 44 24 54 00 00 00 	movl   $0x0,0x54(%rsp)
  40074b:	00 
  40074c:	e8 4f 02 00 00       	callq  4009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>
  400751:	31 c0                	xor    %eax,%eax
  400753:	89 c7                	mov    %eax,%edi
  400755:	e8 56 02 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40075a:	48 89 c7             	mov    %rax,%rdi
  40075d:	48 89 c1             	mov    %rax,%rcx
  400760:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
  400765:	48 89 c7             	mov    %rax,%rdi
  400768:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40076d:	e8 6e 02 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  400772:	e9 00 00 00 00       	jmpq   400777 <main+0x37>
  400777:	be 77 77 00 00       	mov    $0x7777,%esi
  40077c:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  400781:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  400786:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40078b:	e8 90 02 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>
  400790:	be 88 88 00 00       	mov    $0x8888,%esi
  400795:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40079a:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40079f:	e8 bc 02 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>
  4007a4:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007a9:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4007ae:	e8 ed 02 00 00       	callq  400aa0 <_ZN9TestClass15increase_field0Ev>
  4007b3:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007b8:	e8 13 03 00 00       	callq  400ad0 <_ZN9TestClass15increase_field1Ev>
  4007bd:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007c2:	e8 39 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  4007c7:	8b 30                	mov    (%rax),%esi
  4007c9:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007ce:	89 74 24 14          	mov    %esi,0x14(%rsp)
  4007d2:	e8 39 03 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  4007d7:	48 bf 74 0e 40 00 00 	movabs $0x400e74,%rdi
  4007de:	00 00 00 
  4007e1:	8b 10                	mov    (%rax),%edx
  4007e3:	8b 74 24 14          	mov    0x14(%rsp),%esi
  4007e7:	b0 00                	mov    $0x0,%al
  4007e9:	e8 f2 fd ff ff       	callq  4005e0 <printf@plt>
  4007ee:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  4007f3:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4007f7:	e8 04 03 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  4007fc:	31 d2                	xor    %edx,%edx
  4007fe:	88 d1                	mov    %dl,%cl
  400800:	81 38 83 2d d2 27    	cmpl   $0x27d22d83,(%rax)
  400806:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  40080a:	0f 85 17 00 00 00    	jne    400827 <main+0xe7>
  400810:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  400815:	e8 f6 02 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  40081a:	81 38 20 64 68 24    	cmpl   $0x24686420,(%rax)
  400820:	0f 94 c1             	sete   %cl
  400823:	88 4c 24 0f          	mov    %cl,0xf(%rsp)
  400827:	8a 44 24 0f          	mov    0xf(%rsp),%al
  40082b:	34 ff                	xor    $0xff,%al
  40082d:	24 01                	and    $0x1,%al
  40082f:	0f b6 c0             	movzbl %al,%eax
  400832:	48 83 c4 58          	add    $0x58,%rsp
  400836:	c3                   	retq   
  400837:	89 d1                	mov    %edx,%ecx
  400839:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40083e:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
  400842:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  400847:	e8 c4 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  40084c:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
  400851:	e8 ca fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  400856:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40085d:	00 00 00 

0000000000400860 <_Z12new_instancev>:
  400860:	48 83 ec 28          	sub    $0x28,%rsp
  400864:	31 c0                	xor    %eax,%eax
  400866:	89 c7                	mov    %eax,%edi
  400868:	e8 43 01 00 00       	callq  4009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>
  40086d:	48 89 c7             	mov    %rax,%rdi
  400870:	48 89 c1             	mov    %rax,%rcx
  400873:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400878:	48 89 c7             	mov    %rax,%rdi
  40087b:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  400880:	e8 5b 01 00 00       	callq  4009e0 <_ZN9TestClassC2Ev>
  400885:	e9 00 00 00 00       	jmpq   40088a <_Z12new_instancev+0x2a>
  40088a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40088f:	48 83 c4 28          	add    $0x28,%rsp
  400893:	c3                   	retq   
  400894:	89 d1                	mov    %edx,%ecx
  400896:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40089b:	89 4c 24 1c          	mov    %ecx,0x1c(%rsp)
  40089f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008a4:	e8 67 01 00 00       	callq  400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>
  4008a9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4008ae:	e8 6d fd ff ff       	callq  400620 <_Unwind_Resume@plt>
  4008b3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008ba:	84 00 00 00 00 00 

00000000004008c0 <_Z12write_field0P9TestClass>:
  4008c0:	48 83 ec 18          	sub    $0x18,%rsp
  4008c4:	be 77 77 00 00       	mov    $0x7777,%esi
  4008c9:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4008ce:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4008d3:	e8 48 01 00 00       	callq  400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>
  4008d8:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4008dd:	48 83 c4 18          	add    $0x18,%rsp
  4008e1:	c3                   	retq   
  4008e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4008e9:	1f 84 00 00 00 00 00 

00000000004008f0 <_Z11read_field0P9TestClass>:
  4008f0:	50                   	push   %rax
  4008f1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4008f5:	48 8b 3c 24          	mov    (%rsp),%rdi
  4008f9:	e8 02 02 00 00       	callq  400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>
  4008fe:	8b 00                	mov    (%rax),%eax
  400900:	59                   	pop    %rcx
  400901:	c3                   	retq   
  400902:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400909:	1f 84 00 00 00 00 00 

0000000000400910 <_Z12write_field1P9TestClass>:
  400910:	48 83 ec 18          	sub    $0x18,%rsp
  400914:	be 77 77 00 00       	mov    $0x7777,%esi
  400919:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  40091e:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400923:	e8 38 01 00 00       	callq  400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>
  400928:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40092d:	48 83 c4 18          	add    $0x18,%rsp
  400931:	c3                   	retq   
  400932:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400939:	1f 84 00 00 00 00 00 

0000000000400940 <_Z11read_field1P9TestClass>:
  400940:	50                   	push   %rax
  400941:	48 89 3c 24          	mov    %rdi,(%rsp)
  400945:	48 8b 3c 24          	mov    (%rsp),%rdi
  400949:	e8 c2 01 00 00       	callq  400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>
  40094e:	8b 00                	mov    (%rax),%eax
  400950:	59                   	pop    %rcx
  400951:	c3                   	retq   
  400952:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  400959:	1f 84 00 00 00 00 00 

0000000000400960 <_Z21explicit_write_field0m>:
  400960:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400965:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40096a:	c7 04 bd 80 20 60 00 	movl   $0x7777,0x602080(,%rdi,4)
  400971:	77 77 00 00 
  400975:	c3                   	retq   
  400976:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40097d:	00 00 00 

0000000000400980 <_Z20explicit_read_field0m>:
  400980:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400985:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  40098a:	8b 04 bd 80 20 60 00 	mov    0x602080(,%rdi,4),%eax
  400991:	c3                   	retq   
  400992:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400999:	00 00 00 
  40099c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004009a0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18initialize_storageIJEEEvDpT_>:
  4009a0:	50                   	push   %rax
  4009a1:	e8 5a 02 00 00       	callq  400c00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  4009a6:	48 89 c7             	mov    %rax,%rdi
  4009a9:	e8 52 03 00 00       	callq  400d00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>
  4009ae:	58                   	pop    %rax
  4009af:	c3                   	retq   

00000000004009b0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEnwEm>:
  4009b0:	50                   	push   %rax
  4009b1:	48 89 3c 24          	mov    %rdi,(%rsp)
  4009b5:	e8 c6 03 00 00       	callq  400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>
  4009ba:	e8 41 02 00 00       	callq  400c00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  4009bf:	b9 01 00 00 00       	mov    $0x1,%ecx
  4009c4:	89 ce                	mov    %ecx,%esi
  4009c6:	48 89 c7             	mov    %rax,%rdi
  4009c9:	e8 e2 03 00 00       	callq  400db0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>
  4009ce:	48 89 c7             	mov    %rax,%rdi
  4009d1:	e8 ba 03 00 00       	callq  400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>
  4009d6:	59                   	pop    %rcx
  4009d7:	c3                   	retq   
  4009d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4009df:	00 

00000000004009e0 <_ZN9TestClassC2Ev>:
  4009e0:	48 83 ec 18          	sub    $0x18,%rsp
  4009e4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  4009e9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  4009ee:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  4009f3:	e8 28 01 00 00       	callq  400b20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC2Ev>
  4009f8:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  4009fd:	e8 2e 01 00 00       	callq  400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC2Ev>
  400a02:	48 83 c4 18          	add    $0x18,%rsp
  400a06:	c3                   	retq   
  400a07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400a0e:	00 00 

0000000000400a10 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEEdlEPv>:
  400a10:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400a15:	c3                   	retq   
  400a16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a1d:	00 00 00 

0000000000400a20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEaSEi>:
  400a20:	48 83 ec 28          	sub    $0x28,%rsp
  400a24:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a29:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a2d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a32:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a36:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a3b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a3f:	e8 5c 01 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a44:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a48:	89 30                	mov    %esi,(%rax)
  400a4a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a4f:	48 83 c4 28          	add    $0x28,%rsp
  400a53:	c3                   	retq   
  400a54:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a5b:	00 00 00 
  400a5e:	66 90                	xchg   %ax,%ax

0000000000400a60 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEaSEi>:
  400a60:	48 83 ec 28          	sub    $0x28,%rsp
  400a64:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400a69:	89 74 24 1c          	mov    %esi,0x1c(%rsp)
  400a6d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400a72:	8b 74 24 1c          	mov    0x1c(%rsp),%esi
  400a76:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400a7b:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  400a7f:	e8 1c 02 00 00       	callq  400ca0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400a84:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  400a88:	89 30                	mov    %esi,(%rax)
  400a8a:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  400a8f:	48 83 c4 28          	add    $0x28,%rsp
  400a93:	c3                   	retq   
  400a94:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400a9b:	00 00 00 
  400a9e:	66 90                	xchg   %ax,%ax

0000000000400aa0 <_ZN9TestClass15increase_field0Ev>:
  400aa0:	48 83 ec 18          	sub    $0x18,%rsp
  400aa4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400aa9:	ba 55 55 00 00       	mov    $0x5555,%edx
  400aae:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ab3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400ab8:	48 89 c7             	mov    %rax,%rdi
  400abb:	e8 80 00 00 00       	callq  400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEmLEi>
  400ac0:	48 83 c4 18          	add    $0x18,%rsp
  400ac4:	c3                   	retq   
  400ac5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400acc:	00 00 00 
  400acf:	90                   	nop

0000000000400ad0 <_ZN9TestClass15increase_field1Ev>:
  400ad0:	48 83 ec 18          	sub    $0x18,%rsp
  400ad4:	48 8d 44 24 08       	lea    0x8(%rsp),%rax
  400ad9:	ba 44 44 00 00       	mov    $0x4444,%edx
  400ade:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400ae3:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  400ae8:	48 89 c7             	mov    %rax,%rdi
  400aeb:	e8 50 01 00 00       	callq  400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEmLEi>
  400af0:	48 83 c4 18          	add    $0x18,%rsp
  400af4:	c3                   	retq   
  400af5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400afc:	00 00 00 
  400aff:	90                   	nop

0000000000400b00 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEcvRiEv>:
  400b00:	50                   	push   %rax
  400b01:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b05:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b09:	e8 92 00 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b0e:	59                   	pop    %rcx
  400b0f:	c3                   	retq   

0000000000400b10 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEcvRiEv>:
  400b10:	50                   	push   %rax
  400b11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400b15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400b19:	e8 82 01 00 00       	callq  400ca0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b1e:	59                   	pop    %rcx
  400b1f:	c3                   	retq   

0000000000400b20 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC2Ev>:
  400b20:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b25:	c3                   	retq   
  400b26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b2d:	00 00 00 

0000000000400b30 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC2Ev>:
  400b30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400b35:	c3                   	retq   
  400b36:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b3d:	00 00 00 

0000000000400b40 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEmLEi>:
  400b40:	48 83 ec 38          	sub    $0x38,%rsp
  400b44:	48 89 f8             	mov    %rdi,%rax
  400b47:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400b4c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400b50:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400b55:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400b59:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400b5e:	48 89 f7             	mov    %rsi,%rdi
  400b61:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400b66:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400b6a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400b6f:	e8 2c 00 00 00       	callq  400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400b74:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400b78:	0f af 10             	imul   (%rax),%edx
  400b7b:	89 10                	mov    %edx,(%rax)
  400b7d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400b82:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400b87:	e8 24 00 00 00       	callq  400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC2ERKS3_>
  400b8c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400b91:	48 83 c4 38          	add    $0x38,%rsp
  400b95:	c3                   	retq   
  400b96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400b9d:	00 00 00 

0000000000400ba0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ba0:	50                   	push   %rax
  400ba1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ba5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ba9:	e8 12 00 00 00       	callq  400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400bae:	59                   	pop    %rcx
  400baf:	c3                   	retq   

0000000000400bb0 <_ZN4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassEC2ERKS3_>:
  400bb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400bb5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400bba:	c3                   	retq   
  400bbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400bc0 <_ZNK4ikra3soa6Field_IiLm4660ELj0ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400bc0:	48 83 ec 18          	sub    $0x18,%rsp
  400bc4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400bc9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400bce:	48 89 3c 24          	mov    %rdi,(%rsp)
  400bd2:	e8 29 00 00 00       	callq  400c00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400bd7:	48 89 c7             	mov    %rax,%rdi
  400bda:	e8 31 00 00 00       	callq  400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400bdf:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400be4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400be9:	48 8b 3c 24          	mov    (%rsp),%rdi
  400bed:	48 c1 e7 02          	shl    $0x2,%rdi
  400bf1:	48 01 f8             	add    %rdi,%rax
  400bf4:	48 83 c4 18          	add    $0x18,%rsp
  400bf8:	c3                   	retq   
  400bf9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000400c00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>:
  400c00:	50                   	push   %rax
  400c01:	e8 1a 00 00 00       	callq  400c20 <_ZN9TestClass14storage_bufferEv>
  400c06:	59                   	pop    %rcx
  400c07:	c3                   	retq   
  400c08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400c0f:	00 

0000000000400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>:
  400c10:	50                   	push   %rax
  400c11:	48 89 3c 24          	mov    %rdi,(%rsp)
  400c15:	48 8b 3c 24          	mov    (%rsp),%rdi
  400c19:	e8 12 00 00 00       	callq  400c30 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>
  400c1e:	59                   	pop    %rcx
  400c1f:	c3                   	retq   

0000000000400c20 <_ZN9TestClass14storage_bufferEv>:
  400c20:	48 b8 60 20 60 00 00 	movabs $0x602060,%rax
  400c27:	00 00 00 
  400c2a:	c3                   	retq   
  400c2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400c30 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE8data_ptrEv>:
  400c30:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400c35:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400c3a:	48 8b 47 18          	mov    0x18(%rdi),%rax
  400c3e:	c3                   	retq   
  400c3f:	90                   	nop

0000000000400c40 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEmLEi>:
  400c40:	48 83 ec 38          	sub    $0x38,%rsp
  400c44:	48 89 f8             	mov    %rdi,%rax
  400c47:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  400c4c:	89 54 24 2c          	mov    %edx,0x2c(%rsp)
  400c50:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  400c55:	8b 54 24 2c          	mov    0x2c(%rsp),%edx
  400c59:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
  400c5e:	48 89 f7             	mov    %rsi,%rdi
  400c61:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400c66:	89 54 24 14          	mov    %edx,0x14(%rsp)
  400c6a:	48 89 74 24 08       	mov    %rsi,0x8(%rsp)
  400c6f:	e8 2c 00 00 00       	callq  400ca0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>
  400c74:	8b 54 24 14          	mov    0x14(%rsp),%edx
  400c78:	0f af 10             	imul   (%rax),%edx
  400c7b:	89 10                	mov    %edx,(%rax)
  400c7d:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  400c82:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  400c87:	e8 24 00 00 00       	callq  400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC2ERKS3_>
  400c8c:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400c91:	48 83 c4 38          	add    $0x38,%rsp
  400c95:	c3                   	retq   
  400c96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400c9d:	00 00 00 

0000000000400ca0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE8data_ptrILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPiE4typeEv>:
  400ca0:	50                   	push   %rax
  400ca1:	48 89 3c 24          	mov    %rdi,(%rsp)
  400ca5:	48 8b 3c 24          	mov    (%rsp),%rdi
  400ca9:	e8 12 00 00 00       	callq  400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>
  400cae:	59                   	pop    %rcx
  400caf:	c3                   	retq   

0000000000400cb0 <_ZN4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassEC2ERKS3_>:
  400cb0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400cb5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400cba:	c3                   	retq   
  400cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400cc0 <_ZNK4ikra3soa6Field_IiLm4660ELj4ELi0ELi1E9TestClassE22data_ptr_uninitializedILi0ELi1EEENSt9enable_ifIXaaeqT_L_ZNS0_L16kAddressModeZeroEEeqT0_L_ZNS0_L19kStorageModeDynamicEEEPiE4typeEv>:
  400cc0:	48 83 ec 18          	sub    $0x18,%rsp
  400cc4:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400cc9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400cce:	48 89 3c 24          	mov    %rdi,(%rsp)
  400cd2:	e8 29 ff ff ff       	callq  400c00 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE7storageEv>
  400cd7:	48 89 c7             	mov    %rax,%rdi
  400cda:	e8 31 ff ff ff       	callq  400c10 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EE14data_referenceEv>
  400cdf:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400ce4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ce9:	48 05 d0 48 00 00    	add    $0x48d0,%rax
  400cef:	48 8b 3c 24          	mov    (%rsp),%rdi
  400cf3:	48 c1 e7 02          	shl    $0x2,%rdi
  400cf7:	48 01 f8             	add    %rdi,%rax
  400cfa:	48 83 c4 18          	add    $0x18,%rsp
  400cfe:	c3                   	retq   
  400cff:	90                   	nop

0000000000400d00 <_ZN4ikra3soa15DynamicStorage_I9TestClassLm0EEC2Ev>:
  400d00:	48 83 ec 18          	sub    $0x18,%rsp
  400d04:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
  400d09:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  400d0e:	48 89 f8             	mov    %rdi,%rax
  400d11:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  400d16:	48 89 c7             	mov    %rax,%rdi
  400d19:	e8 32 00 00 00       	callq  400d50 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>
  400d1e:	b9 a8 91 00 00       	mov    $0x91a8,%ecx
  400d23:	89 cf                	mov    %ecx,%edi
  400d25:	e8 c6 f8 ff ff       	callq  4005f0 <malloc@plt>
  400d2a:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  400d2f:	48 89 47 18          	mov    %rax,0x18(%rdi)
  400d33:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  400d3a:	00 
  400d3b:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
  400d42:	00 
  400d43:	48 83 c4 18          	add    $0x18,%rsp
  400d47:	c3                   	retq   
  400d48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400d4f:	00 

0000000000400d50 <_ZN4ikra3soa19StorageStrategySelfINS0_15DynamicStorage_I9TestClassLm0EEEEC2Ev>:
  400d50:	50                   	push   %rax
  400d51:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d55:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d59:	e8 02 00 00 00       	callq  400d60 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>
  400d5e:	58                   	pop    %rax
  400d5f:	c3                   	retq   

0000000000400d60 <_ZN4ikra3soa19StorageStrategyBaseC2Ev>:
  400d60:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400d65:	48 8b 7c 24 f8       	mov    -0x8(%rsp),%rdi
  400d6a:	48 c7 07 00 00 00 00 	movq   $0x0,(%rdi)
  400d71:	c3                   	retq   
  400d72:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d79:	00 00 00 
  400d7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d80 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE18check_sizeof_classILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEvE4typeEv>:
  400d80:	c3                   	retq   
  400d81:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d88:	00 00 00 
  400d8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400d90 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE3getEm>:
  400d90:	50                   	push   %rax
  400d91:	48 89 3c 24          	mov    %rdi,(%rsp)
  400d95:	48 8b 3c 24          	mov    (%rsp),%rdi
  400d99:	48 83 c7 01          	add    $0x1,%rdi
  400d9d:	e8 3e 00 00 00       	callq  400de0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>
  400da2:	59                   	pop    %rcx
  400da3:	c3                   	retq   
  400da4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400dab:	00 00 00 
  400dae:	66 90                	xchg   %ax,%ax

0000000000400db0 <_ZN4ikra3soa19StorageStrategyBase13increase_sizeEm>:
  400db0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400db5:	48 89 74 24 f0       	mov    %rsi,-0x10(%rsp)
  400dba:	48 8b 74 24 f8       	mov    -0x8(%rsp),%rsi
  400dbf:	48 8b 3e             	mov    (%rsi),%rdi
  400dc2:	48 89 7c 24 e8       	mov    %rdi,-0x18(%rsp)
  400dc7:	48 8b 7c 24 f0       	mov    -0x10(%rsp),%rdi
  400dcc:	48 03 3e             	add    (%rsi),%rdi
  400dcf:	48 89 3e             	mov    %rdi,(%rsi)
  400dd2:	48 8b 44 24 e8       	mov    -0x18(%rsp),%rax
  400dd7:	c3                   	retq   
  400dd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400ddf:	00 

0000000000400de0 <_ZN4ikra3soa9SoaLayoutI9TestClassLm4660ELi0ENS0_14DynamicStorageEE4get_ILi0EEENSt9enable_ifIXeqT_L_ZNS0_L16kAddressModeZeroEEEPS2_E4typeEm>:
  400de0:	48 89 7c 24 f8       	mov    %rdi,-0x8(%rsp)
  400de5:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  400dea:	c3                   	retq   
  400deb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000400df0 <__libc_csu_init>:
  400df0:	41 57                	push   %r15
  400df2:	41 56                	push   %r14
  400df4:	41 89 ff             	mov    %edi,%r15d
  400df7:	41 55                	push   %r13
  400df9:	41 54                	push   %r12
  400dfb:	4c 8d 25 de 0f 20 00 	lea    0x200fde(%rip),%r12        # 601de0 <__frame_dummy_init_array_entry>
  400e02:	55                   	push   %rbp
  400e03:	48 8d 2d de 0f 20 00 	lea    0x200fde(%rip),%rbp        # 601de8 <__init_array_end>
  400e0a:	53                   	push   %rbx
  400e0b:	49 89 f6             	mov    %rsi,%r14
  400e0e:	49 89 d5             	mov    %rdx,%r13
  400e11:	4c 29 e5             	sub    %r12,%rbp
  400e14:	48 83 ec 08          	sub    $0x8,%rsp
  400e18:	48 c1 fd 03          	sar    $0x3,%rbp
  400e1c:	e8 87 f7 ff ff       	callq  4005a8 <_init>
  400e21:	48 85 ed             	test   %rbp,%rbp
  400e24:	74 20                	je     400e46 <__libc_csu_init+0x56>
  400e26:	31 db                	xor    %ebx,%ebx
  400e28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  400e2f:	00 
  400e30:	4c 89 ea             	mov    %r13,%rdx
  400e33:	4c 89 f6             	mov    %r14,%rsi
  400e36:	44 89 ff             	mov    %r15d,%edi
  400e39:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  400e3d:	48 83 c3 01          	add    $0x1,%rbx
  400e41:	48 39 eb             	cmp    %rbp,%rbx
  400e44:	75 ea                	jne    400e30 <__libc_csu_init+0x40>
  400e46:	48 83 c4 08          	add    $0x8,%rsp
  400e4a:	5b                   	pop    %rbx
  400e4b:	5d                   	pop    %rbp
  400e4c:	41 5c                	pop    %r12
  400e4e:	41 5d                	pop    %r13
  400e50:	41 5e                	pop    %r14
  400e52:	41 5f                	pop    %r15
  400e54:	c3                   	retq   
  400e55:	90                   	nop
  400e56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400e5d:	00 00 00 

0000000000400e60 <__libc_csu_fini>:
  400e60:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000400e64 <_fini>:
  400e64:	48 83 ec 08          	sub    $0x8,%rsp
  400e68:	48 83 c4 08          	add    $0x8,%rsp
  400e6c:	c3                   	retq   
