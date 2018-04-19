
Fatbin elf code:
================
arch = sm_30
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0

	code for sm_30

Fatbin elf code:
================
arch = sm_30
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:8655095
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm0EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m:
  REG:14 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0_:
  REG:26 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:348 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0_:
  REG:26 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm:
  REG:18 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0

	code for sm_30
		Function : _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm0EEEJEEEvPT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                  /* 0x22e28042e2c2c307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                          /* 0x2800400110005de4 */
        /*0010*/                   ISUB RZ.CC, c[0x0][0x140], RZ;                 /* 0x4801400503ffde03 */
        /*0018*/                   ISETP.EQ.X.AND P0, PT, RZ, c[0x0][0x144], PT;  /* 0x190e400513f1dc63 */
        /*0020*/               @P0 EXIT;                                          /* 0x80000000000001e7 */
        /*0028*/                   MOV R2, c[0x0][0x140];                         /* 0x2800400500009de4 */
        /*0030*/                   MOV R3, c[0x0][0x144];                         /* 0x280040051000dde4 */
        /*0038*/                   ST.E.64 [R2], RZ;                              /* 0x94000000002fdca5 */
                                                                                  /* 0x200000000002f047 */
        /*0048*/                   ST.E [R2+0x8], RZ;                             /* 0x94000000202fdc85 */
        /*0050*/                   EXIT;                                          /* 0x8000000000001de7 */
        /*0058*/                   BRA 0x58;                                      /* 0x4003ffffe0001de7 */
        /*0060*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0068*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0070*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0078*/                   NOP;                                           /* 0x4000000000001de4 */
		............................................................................................................


		Function : _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                 /* 0x2200428042004307 */
        /*0008*/                   MOV R1, c[0x0][0x44];         /* 0x2800400110005de4 */
        /*0010*/                   MOV R2, c[0x0][0x140];        /* 0x2800400500009de4 */
        /*0018*/                   MOV R3, c[0x0][0x144];        /* 0x280040051000dde4 */
        /*0020*/                   MOV R4, c[0x0][0x148];        /* 0x2800400520011de4 */
        /*0028*/                   MOV R5, c[0x0][0x14c];        /* 0x2800400530015de4 */
        /*0030*/                   ATOM.E.ADD.U64 R4, [R2], R4;  /* 0x547e200000211e05 */
        /*0038*/                   MOV32I R8, 0x0;               /* 0x1800000000021de2 */
                                                                 /* 0x20000002f043f2d7 */
        /*0048*/                   MOV32I R9, 0x0;               /* 0x1800000000025de2 */
        /*0050*/                   CCTL.E.IV [R2];               /* 0x9c00000000201ca5 */
        /*0058*/                   ST.E.64 [R8], R4;             /* 0x9400000000811ca5 */
        /*0060*/                   EXIT;                         /* 0x8000000000001de7 */
        /*0068*/                   BRA 0x68;                     /* 0x4003ffffe0001de7 */
        /*0070*/                   NOP;                          /* 0x4000000000001de4 */
        /*0078*/                   NOP;                          /* 0x4000000000001de4 */
		....................................................................................................


		Function : _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                         /* 0x2200422042004307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                 /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R4, 0x0;                       /* 0x1800000000011de2 */
        /*0018*/                   MOV32I R8, 0x13f;                     /* 0x18000004fc021de2 */
        /*0020*/                   MOV32I R5, 0x0;                       /* 0x1800000000015de2 */
        /*0028*/                   MOV32I R12, 0x1;                      /* 0x1800000004031de2 */
        /*0030*/                   MOV32I R6, 0x0;                       /* 0x1800000000019de2 */
        /*0038*/                   MOV R13, RZ;                          /* 0x28000000fc035de4 */
                                                                         /* 0x22820042f2f20207 */
        /*0048*/                   MOV32I R7, 0x0;                       /* 0x180000000001dde2 */
        /*0050*/                   MOV32I R10, 0x0;                      /* 0x1800000000029de2 */
        /*0058*/                   MOV32I R11, 0x0;                      /* 0x180000000002dde2 */
        /*0060*/                   JCAL 0x0;                             /* 0x1000000000011c07 */
        /*0068*/                   MOV R6, c[0x0][0x140];                /* 0x2800400500019de4 */
        /*0070*/                   S2R R3, SR_LANEID;                    /* 0x2c0000000000dc04 */
        /*0078*/                   VOTE.ALL R0, PT, PT;                  /* 0x49c0000000701c04 */
                                                                         /* 0x2240423042b20047 */
        /*0088*/                   IADD32I R6.CC, R6, 0x8;               /* 0x0c00000020619c02 */
        /*0090*/                   POPC R2, R0, R0;                      /* 0x5400000000009c04 */
        /*0098*/                   FLO.U32 R5, R0;                       /* 0x7800000000015c03 */
        /*00a0*/                   IADD.X R7, RZ, c[0x0][0x144];         /* 0x4800400513f1dc43 */
        /*00a8*/                   ISETP.EQ.U32.AND P0, PT, R5, R3, PT;  /* 0x190e00000c51dc03 */
        /*00b0*/                   IMUL R4, R2, c[0x0][0x148];           /* 0x5000400520211ca3 */
        /*00b8*/                   MOV R2, R6;                           /* 0x2800000018009de4 */
                                                                         /* 0x2373f042e0428047 */
        /*00c8*/                   MOV R3, R7;                           /* 0x280000001c00dde4 */
        /*00d0*/                   S2R R6, SR_LTMASK;                    /* 0x2c000000e4019c04 */
        /*00d8*/               @P0 ATOM.E.ADD R4, [R2], R4;              /* 0x547e200000210005 */
        /*00e0*/                   LOP.AND R0, R6, R0;                   /* 0x6800000000601c03 */
        /*00e8*/                   POPC R0, R0, R0;                      /* 0x5400000000001c04 */
        /*00f0*/                   CCTL.E.IV [R2];                       /* 0x9c00000000201ca5 */
        /*00f8*/                   SHFL.IDX PT, R5, R4, R5, 0x1f;        /* 0x8c007c0014415f45 */
                                                                         /* 0x2222820042c28047 */
        /*0108*/                   IMAD R0, R0, c[0x0][0x148], R5;       /* 0x200a400520001ca3 */
        /*0110*/                   MOV32I R5, 0x0;                       /* 0x1800000000015de2 */
        /*0118*/                   IADD R0.CC, R0, c[0x0][0x140];        /* 0x4801400500001c03 */
        /*0120*/                   IADD.X R9, RZ, c[0x0][0x144];         /* 0x4800400513f25c43 */
        /*0128*/                   IADD32I R8.CC, R0, 0x8;               /* 0x0c00000020021c02 */
        /*0130*/                   MOV32I R0, 0x0;                       /* 0x1800000000001de2 */
        /*0138*/                   MOV R4, R0;                           /* 0x2800000000011de4 */
                                                                         /* 0x2000000002f04287 */
        /*0148*/                   IADD.X R9, RZ, R9;                    /* 0x4800000027f25c43 */
        /*0150*/                   ST.E.64 [R4], R8;                     /* 0x9400000000421ca5 */
        /*0158*/                   EXIT;                                 /* 0x8000000000001de7 */
        /*0160*/                   BRA 0x160;                            /* 0x4003ffffe0001de7 */
        /*0168*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0170*/                   NOP;                                  /* 0x4000000000001de4 */
        /*0178*/                   NOP;                                  /* 0x4000000000001de4 */
		................................................................................................


		Function : _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                       /* 0x22c2b20282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                               /* 0x2800400110005de4 */
        /*0010*/                   S2R R17, SR_TID.X;                                  /* 0x2c00000084045c04 */
        /*0018*/                   S2R R0, SR_CTAID.X;                                 /* 0x2c00000094001c04 */
        /*0020*/                   IMAD R17, R0, c[0x0][0x28], R17;                    /* 0x20224000a0045ca3 */
        /*0028*/                   ISUB RZ.CC, R17, c[0x0][0x150];                     /* 0x48014005411fdd03 */
        /*0030*/                   SHR R16, R17, 0x1f;                                 /* 0x5800c0007d141c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R16, c[0x0][0x154], PT;  /* 0x1b0e40055101dc43 */
                                                                                       /* 0x220042b0420042e7 */
        /*0048*/               @P0 EXIT;                                               /* 0x80000000000001e7 */
        /*0050*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*0058*/                   IADD R17.CC, R17, c[0x0][0x148];                    /* 0x4801400521145c03 */
        /*0060*/                   MOV32I R3, 0x0;                                     /* 0x180000000000dde2 */
        /*0068*/                   SSY 0x118;                                          /* 0x60000002a0001c07 */
        /*0070*/                   LD.E.64 R2, [R2];                                   /* 0x8400000000209ca5 */
        /*0078*/                   IADD.X R16, R16, c[0x0][0x14c];                     /* 0x4800400531041c43 */
                                                                                       /* 0x220042f2c2c3f2c7 */
        /*0088*/                   IADD32I R4.CC, R17, -0x1;                           /* 0x0ffffffffd111c02 */
        /*0090*/                   IADD32I.X R5, R16, -0x1;                            /* 0x0bfffffffd015c42 */
        /*0098*/                   ISUB RZ.CC, R2, R4;                                 /* 0x48010000102fdd03 */
        /*00a0*/                   ISETP.GT.U32.X.AND P0, PT, R3, R5, PT;              /* 0x1a0e00001431dc43 */
        /*00a8*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*00b0*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*00b8*/                   MOV32I R8, 0xab;                                    /* 0x18000002ac021de2 */
                                                                                       /* 0x22f2020200422047 */
        /*00c8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*00d0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*00d8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*00e0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*00e8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*00f0*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*00f8*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
                                                                                       /* 0x2202f2c2c042f2f7 */
        /*0108*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0110*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*0118*/                   IADD RZ.CC, -RZ, R17;                               /* 0x4801000047ffde03 */
        /*0120*/                   SSY 0x1a0;                                          /* 0x60000001e0001c07 */
        /*0128*/                   ISETP.NE.X.AND P0, PT, R16, RZ, PT;                 /* 0x1a8e0000fd01dc63 */
        /*0130*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0138*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
                                                                                       /* 0x2202004220420047 */
        /*0148*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0150*/                   MOV32I R8, 0xd0;                                    /* 0x1800000340021de2 */
        /*0158*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0160*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0168*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0170*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0178*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x2240423042f2f2f7 */
        /*0188*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0190*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0198*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*01a0*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
        /*01a8*/                   SHL R3, R16, 0x2;                                   /* 0x6000c0000900dc03 */
        /*01b0*/                   MOV32I R2, 0x8;                                     /* 0x1800000020009de2 */
        /*01b8*/                   SHR.U32 R4, R17, 0x1e;                              /* 0x5800c00079111c03 */
                                                                                       /* 0x2200423042b04207 */
        /*01c8*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*01d0*/                   SHL R0, R17, -0x1e;                                 /* 0x6000ffff89101c03 */
        /*01d8*/                   PRMT R22, RZ, 0x6540, R17;                          /* 0x2422c19503f59c04 */
        /*01e0*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*01e8*/                   PRMT R16, RZ, 0x6540, R16;                          /* 0x2420c19503f41c04 */
        /*01f0*/                   IADD32I RZ.CC, R2, -0x20;                           /* 0x0fffffff802fdc02 */
        /*01f8*/                   SHL R23, R17, -0x18;                                /* 0x6000ffffa115dc03 */
                                                                                       /* 0x2202c04200423277 */
        /*0208*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*0210*/               @P0 IADD R0, R3, R4;                                    /* 0x4800000010300003 */
        /*0218*/                   CSETP.SFT.AND P1, PT, CC, PT;                       /* 0x040e00000153dc04 */
        /*0220*/                   MOV32I R3, 0x0;                                     /* 0x180000000000dde2 */
        /*0228*/                   IMAD.U32.U32 R24.CC, R17, 0x4, R2;                  /* 0x2005c00011161c03 */
        /*0230*/                   PRMT R17, RZ, 0x7, R17;                             /* 0x2422c0001ff45c04 */
        /*0238*/                   IADD.X R25, R0, R3;                                 /* 0x480000000c065c43 */
                                                                                       /* 0x2282f2c202b04207 */
        /*0248*/                   IADD R22.CC, R22, R2;                               /* 0x4801000009659c03 */
        /*0250*/               @P1 IADD R23, R16, R17;                                 /* 0x480000004505c403 */
        /*0258*/                   MOV32I R17, 0xffffffc0;                             /* 0x1bffffff00045de2 */
        /*0260*/                   IADD.X R23, R23, R3;                                /* 0x480000000d75dc43 */
        /*0268*/                   IADD32I R22.CC, R22, 0x20030;                       /* 0x0c000800c1659c02 */
        /*0270*/                   IADD.X R23, RZ, R23;                                /* 0x480000005ff5dc43 */
        /*0278*/                   IADD32I R0, R2, 0x10;                               /* 0x0800000040201c02 */
                                                                                       /* 0x220042f202c2c287 */
        /*0288*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*0290*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*0298*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*02a0*/                   P2R R20, PR, RZ, 0x1;                               /* 0x3000c00007f51c04 */
        /*02a8*/               @P0 BRA 0x310;                                          /* 0x40000001800001e7 */
        /*02b0*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*02b8*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
                                                                                       /* 0x22f2020200422047 */
        /*02c8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*02d0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*02d8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*02e0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*02e8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*02f0*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*02f8*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
                                                                                       /* 0x2202c2c2828042f7 */
        /*0308*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0310*/                   LD.E R16, [R24+0x10];                               /* 0x8400000041841c85 */
        /*0318*/                   IADD32I R0, R2, 0x820830;                           /* 0x08020820c0201c02 */
        /*0320*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*0328*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*0330*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*0338*/                   P2R R19, PR, RZ, 0x1;                               /* 0x3000c00007f4dc04 */
                                                                                       /* 0x22004220420042f7 */
        /*0348*/               @P0 BRA 0x3a8;                                          /* 0x40000001600001e7 */
        /*0350*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0358*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0360*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0368*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0370*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0378*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
                                                                                       /* 0x22828042f2f20207 */
        /*0388*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0390*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0398*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*03a0*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*03a8*/                   LD.E R0, [R24+0x820830];                            /* 0x84020820c1801c85 */
        /*03b0*/                   IADD32I R3, R2, 0x20030;                            /* 0x08000800c020dc02 */
        /*03b8*/                   LOP32I.AND R3, R3, 0x7;                             /* 0x380000001c30dc02 */
                                                                                       /* 0x2202f043f2c2c047 */
        /*03c8*/                   IADD RZ.CC, -RZ, R3;                                /* 0x480100000fffde03 */
        /*03d0*/                   IADD R3, R16, c[0x0][0x158];                        /* 0x480040056100dc03 */
        /*03d8*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*03e0*/                   P2R R18, PR, RZ, 0x1;                               /* 0x3000c00007f49c04 */
        /*03e8*/                   IADD R16, R0, R3;                                   /* 0x480000000c041c03 */
        /*03f0*/               @P0 BRA 0x458;                                          /* 0x40000001800001e7 */
        /*03f8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
                                                                                       /* 0x2202004220420047 */
        /*0408*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0410*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0418*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0420*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0428*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0430*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0438*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x22f04283f042f2f7 */
        /*0448*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0450*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0458*/                   LD.E R3, [R22];                                     /* 0x840000000160dc85 */
        /*0460*/                   ISETP.NE.AND P6, PT, R20, RZ, PT;                   /* 0x1a8e0000fd4ddc23 */
        /*0468*/                   IADD R3, R16, R3;                                   /* 0x480000000d00dc03 */
        /*0470*/                   ST.E [R22], R3;                                     /* 0x940000000160dc85 */
        /*0478*/               @P6 BRA 0x4e0;                                          /* 0x40000001800019e7 */
                                                                                       /* 0x2202004220420047 */
        /*0488*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0490*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0498*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*04a0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*04a8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*04b0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*04b8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
                                                                                       /* 0x2202f2c042f2f207 */
        /*04c8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*04d0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*04d8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*04e0*/                   LD.E R16, [R24+0x10];                               /* 0x8400000041841c85 */
        /*04e8*/                   ISETP.NE.AND P6, PT, R19, RZ, PT;                   /* 0x1a8e0000fd3ddc23 */
        /*04f0*/               @P6 BRA 0x558;                                          /* 0x40000001800019e7 */
        /*04f8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
                                                                                       /* 0x2202004220420047 */
        /*0508*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0510*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0518*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0520*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0528*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0530*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0538*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x22f043f20042f2f7 */
        /*0548*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0550*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0558*/                   LD.E R0, [R24+0x820830];                            /* 0x84020820c1801c85 */
        /*0560*/                   ISETP.NE.AND P6, PT, R18, RZ, PT;                   /* 0x1a8e0000fd2ddc23 */
        /*0568*/                   IADD R3, R16, c[0x0][0x158];                        /* 0x480040056100dc03 */
        /*0570*/                   IADD R16, R0, R3;                                   /* 0x480000000c041c03 */
        /*0578*/               @P6 BRA 0x5e0;                                          /* 0x40000001800019e7 */
                                                                                       /* 0x2202004220420047 */
        /*0588*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0590*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0598*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*05a0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*05a8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*05b0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*05b8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
                                                                                       /* 0x220283f042f2f207 */
        /*05c8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*05d0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*05d8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*05e0*/                   LD.E R3, [R22+0x4];                                 /* 0x840000001160dc85 */
        /*05e8*/                   ISETP.NE.AND P6, PT, R20, RZ, PT;                   /* 0x1a8e0000fd4ddc23 */
        /*05f0*/                   IADD R3, R16, R3;                                   /* 0x480000000d00dc03 */
        /*05f8*/                   ST.E [R22+0x4], R3;                                 /* 0x940000001160dc85 */
                                                                                       /* 0x22004220420042f7 */
        /*0608*/               @P6 BRA 0x668;                                          /* 0x40000001600019e7 */
        /*0610*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0618*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0620*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0628*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0630*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0638*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
                                                                                       /* 0x22f2c042f2f20207 */
        /*0648*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0650*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0658*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0660*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0668*/                   LD.E R16, [R24+0x10];                               /* 0x8400000041841c85 */
        /*0670*/                   ISETP.NE.AND P6, PT, R19, RZ, PT;                   /* 0x1a8e0000fd3ddc23 */
        /*0678*/               @P6 BRA 0x6e0;                                          /* 0x40000001800019e7 */
                                                                                       /* 0x2202004220420047 */
        /*0688*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0690*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0698*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*06a0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*06a8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*06b0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*06b8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
                                                                                       /* 0x2203f20042f2f207 */
        /*06c8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*06d0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*06d8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*06e0*/                   LD.E R0, [R24+0x820830];                            /* 0x84020820c1801c85 */
        /*06e8*/                   ISETP.NE.AND P6, PT, R18, RZ, PT;                   /* 0x1a8e0000fd2ddc23 */
        /*06f0*/                   IADD R3, R16, c[0x0][0x158];                        /* 0x480040056100dc03 */
        /*06f8*/                   IADD R16, R0, R3;                                   /* 0x480000000c041c03 */
                                                                                       /* 0x22004220420042f7 */
        /*0708*/               @P6 BRA 0x768;                                          /* 0x40000001600019e7 */
        /*0710*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0718*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0720*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0728*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0730*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0738*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
                                                                                       /* 0x2283f042f2f20207 */
        /*0748*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0750*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0758*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0760*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0768*/                   LD.E R3, [R22+0x8];                                 /* 0x840000002160dc85 */
        /*0770*/                   ISETP.NE.AND P6, PT, R20, RZ, PT;                   /* 0x1a8e0000fd4ddc23 */
        /*0778*/                   IADD R3, R16, R3;                                   /* 0x480000000d00dc03 */
                                                                                       /* 0x221200420042f047 */
        /*0788*/                   ST.E [R22+0x8], R3;                                 /* 0x940000002160dc85 */
        /*0790*/               @P6 BRA 0x7f0;                                          /* 0x40000001600019e7 */
        /*0798*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*07a0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*07a8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*07b0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*07b8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
                                                                                       /* 0x22c042f2f2020047 */
        /*07c8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*07d0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*07d8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*07e0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*07e8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*07f0*/                   LD.E R16, [R24+0x10];                               /* 0x8400000041841c85 */
        /*07f8*/                   ISETP.NE.AND P6, PT, R19, RZ, PT;                   /* 0x1a8e0000fd3ddc23 */
                                                                                       /* 0x22004220420042f7 */
        /*0808*/               @P6 BRA 0x868;                                          /* 0x40000001600019e7 */
        /*0810*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0818*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0820*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0828*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0830*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0838*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
                                                                                       /* 0x23f20042f2f20207 */
        /*0848*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0850*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0858*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0860*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0868*/                   LD.E R0, [R24+0x820830];                            /* 0x84020820c1801c85 */
        /*0870*/                   ISETP.NE.AND P6, PT, R18, RZ, PT;                   /* 0x1a8e0000fd2ddc23 */
        /*0878*/                   IADD R3, R16, c[0x0][0x158];                        /* 0x480040056100dc03 */
                                                                                       /* 0x221200420042f047 */
        /*0888*/                   IADD R16, R0, R3;                                   /* 0x480000000c041c03 */
        /*0890*/               @P6 BRA 0x8f0;                                          /* 0x40000001600019e7 */
        /*0898*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*08a0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*08a8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*08b0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*08b8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
                                                                                       /* 0x220042f2f2020047 */
        /*08c8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*08d0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*08d8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*08e0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*08e8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*08f0*/                   LD.E R3, [R22+0xc];                                 /* 0x840000003160dc85 */
        /*08f8*/                   IADD32I R0.CC, R22, 0x10;                           /* 0x0c00000041601c02 */
                                                                                       /* 0x220042e283f042b7 */
        /*0908*/                   IADD32I R17, R17, 0x4;                              /* 0x0800000011145c02 */
        /*0910*/                   ISETP.NE.AND P0, PT, R17, RZ, PT;                   /* 0x1a8e0000fd11dc23 */
        /*0918*/                   IADD.X R4, RZ, R23;                                 /* 0x480000005ff11c43 */
        /*0920*/                   IADD R3, R16, R3;                                   /* 0x480000000d00dc03 */
        /*0928*/                   ST.E [R22+0xc], R3;                                 /* 0x940000003160dc85 */
        /*0930*/                   MOV R22, R0;                                        /* 0x2800000000059de4 */
        /*0938*/                   MOV R23, R4;                                        /* 0x280000001005dde4 */
                                                                                       /* 0x200000000002f2e7 */
        /*0948*/               @P0 BRA 0x278;                                          /* 0x4003ffe4a00001e7 */
        /*0950*/                   EXIT;                                               /* 0x8000000000001de7 */
        /*0958*/                   BRA 0x958;                                          /* 0x4003ffffe0001de7 */
        /*0960*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0968*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0970*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0978*/                   NOP;                                                /* 0x4000000000001de4 */
		......................................................................................................................................................................


		Function : _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                       /* 0x22c2b20282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                               /* 0x2800400110005de4 */
        /*0010*/                   S2R R19, SR_TID.X;                                  /* 0x2c0000008404dc04 */
        /*0018*/                   S2R R0, SR_CTAID.X;                                 /* 0x2c00000094001c04 */
        /*0020*/                   IMAD R19, R0, c[0x0][0x28], R19;                    /* 0x20264000a004dca3 */
        /*0028*/                   ISUB RZ.CC, R19, c[0x0][0x148];                     /* 0x48014005213fdd03 */
        /*0030*/                   SHR R16, R19, 0x1f;                                 /* 0x5800c0007d341c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R16, c[0x0][0x14c], PT;  /* 0x1b0e40053101dc43 */
                                                                                       /* 0x22f2c2c042c042e7 */
        /*0048*/               @P0 EXIT;                                               /* 0x80000000000001e7 */
        /*0050*/                   IADD R17.CC, R19, c[0x0][0x140];                    /* 0x4801400501345c03 */
        /*0058*/                   SSY 0xe8;                                           /* 0x6000000220001c07 */
        /*0060*/                   IADD.X R16, R16, c[0x0][0x144];                     /* 0x4800400511041c43 */
        /*0068*/                   IADD32I RZ.CC, R17, -0x8008;                        /* 0x0ffffdffe11fdc02 */
        /*0070*/                   ISETP.LT.U32.X.AND P0, PT, R16, RZ, PT;             /* 0x188e0000fd01dc43 */
        /*0078*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
                                                                                       /* 0x2202004220420047 */
        /*0088*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0090*/                   MOV32I R8, 0xa0;                                    /* 0x1800000280021de2 */
        /*0098*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*00a0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*00a8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*00b0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*00b8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
                                                                                       /* 0x2202c042f2f2f207 */
        /*00c8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*00d0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*00d8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*00e0*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*00e8*/                   IADD32I R17.CC, R17, 0x1;                           /* 0x0c00000005145c02 */
        /*00f0*/                   SSY 0x180;                                          /* 0x6000000220001c07 */
        /*00f8*/                   IADD.X R16, RZ, R16;                                /* 0x4800000043f41c43 */
                                                                                       /* 0x2220420042f2c2c7 */
        /*0108*/                   IADD RZ.CC, -RZ, R17;                               /* 0x4801000047ffde03 */
        /*0110*/                   ISETP.NE.X.AND P0, PT, R16, RZ, PT;                 /* 0x1a8e0000fd01dc63 */
        /*0118*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0120*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0128*/                   MOV32I R8, 0xd0;                                    /* 0x1800000340021de2 */
        /*0130*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0138*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x22f2f2f202020047 */
        /*0148*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0150*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0158*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0160*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0168*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0170*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0178*/                   NOP.S;                                              /* 0x4000000000001df4 */
                                                                                       /* 0x2202c04282f2c2c7 */
        /*0188*/                   IADD RZ.CC, -RZ, R17;                               /* 0x4801000047ffde03 */
        /*0190*/                   ISETP.EQ.X.AND P0, PT, R16, RZ, PT;                 /* 0x190e0000fd01dc63 */
        /*0198*/               @P0 EXIT;                                               /* 0x80000000000001e7 */
        /*01a0*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*01a8*/                   MOV32I R18, 0x0;                                    /* 0x1800000000049de2 */
        /*01b0*/                   IADD32I R0.CC, R2, 0x10;                            /* 0x0c00000040201c02 */
        /*01b8*/                   IADD.X R3, RZ, R18;                                 /* 0x480000004bf0dc43 */
                                                                                       /* 0x22020042f2c2c2c7 */
        /*01c8*/                   LOP32I.AND RZ.CC, R0, 0x7;                          /* 0x3c0000001c0fdc02 */
        /*01d0*/                   LOP.AND.X RZ.CC, R3, RZ;                            /* 0x68010000fc3fdc23 */
        /*01d8*/                   CSETP.NE.AND P0, PT, CC, PT;                        /* 0x040e00000051dc04 */
        /*01e0*/              @!P0 BRA 0x240;                                          /* 0x40000001600021e7 */
        /*01e8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*01f0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*01f8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*0208*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0210*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0218*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0220*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0228*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0230*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0238*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x22b2027232028047 */
        /*0248*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
        /*0250*/                   SHL R3, R17, -0x1e;                                 /* 0x6000ffff8910dc03 */
        /*0258*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*0260*/                   SHL R0, R16, 0x2;                                   /* 0x6000c00009001c03 */
        /*0268*/                   SHR.U32 R4, R17, 0x1e;                              /* 0x5800c00079111c03 */
        /*0270*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*0278*/                   IMAD.U32.U32 R24.CC, R17, 0x4, R2;                  /* 0x2005c00011161c03 */
                                                                                       /* 0x22c202c042804287 */
        /*0288*/               @P0 IADD R3, R0, R4;                                    /* 0x480000001000c003 */
        /*0290*/                   IADD.X R25, R3, R18;                                /* 0x4800000048365c43 */
        /*0298*/                   MOV R3, c[0x0][0x150];                              /* 0x280040054000dde4 */
        /*02a0*/                   IADD32I R0.CC, R2, 0x820830;                        /* 0x0c020820c0201c02 */
        /*02a8*/                   ST.E [R24+0x10], R3;                                /* 0x940000004180dc85 */
        /*02b0*/                   IADD.X R4, RZ, R18;                                 /* 0x480000004bf11c43 */
        /*02b8*/                   LOP32I.AND RZ.CC, R0, 0x7;                          /* 0x3c0000001c0fdc02 */
                                                                                       /* 0x2220420042f2c2c7 */
        /*02c8*/                   LOP.AND.X RZ.CC, R4, RZ;                            /* 0x68010000fc4fdc23 */
        /*02d0*/                   CSETP.NE.AND P0, PT, CC, PT;                        /* 0x040e00000051dc04 */
        /*02d8*/              @!P0 BRA 0x338;                                          /* 0x40000001600021e7 */
        /*02e0*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*02e8*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*02f0*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*02f8*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x2282f2f202020047 */
        /*0308*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0310*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0318*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0320*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0328*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0330*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0338*/                   MOV32I R0, 0x8;                                     /* 0x1800000020001de2 */
                                                                                       /* 0x2202804220420047 */
        /*0348*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*0350*/                   PRMT R22, RZ, 0x6540, R17;                          /* 0x2422c19503f59c04 */
        /*0358*/                   PRMT R16, RZ, 0x6540, R16;                          /* 0x2420c19503f41c04 */
        /*0360*/                   MOV R4, c[0x0][0x154];                              /* 0x2800400550011de4 */
        /*0368*/                   SHL R3, R17, -0x18;                                 /* 0x6000ffffa110dc03 */
        /*0370*/                   PRMT R17, RZ, 0x7, R17;                             /* 0x2422c0001ff45c04 */
        /*0378*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
                                                                                       /* 0x2200428042a20047 */
        /*0388*/                   IADD R22.CC, R22, R2;                               /* 0x4801000009659c03 */
        /*0390*/                   ST.E [R24+0x820830], R4;                            /* 0x94020820c1811c85 */
        /*0398*/                   IADD R0, R19, c[0x0][0x140];                        /* 0x4800400501301c03 */
        /*03a0*/               @P0 IADD R3, R16, R17;                                  /* 0x480000004500c003 */
        /*03a8*/                   MOV32I R16, 0xffffffc0;                             /* 0x1bffffff00041de2 */
        /*03b0*/                   IADD.X R18, R3, R18;                                /* 0x4800000048349c43 */
        /*03b8*/                   IADD32I R22.CC, R22, 0x20030;                       /* 0x0c000800c1659c02 */
                                                                                       /* 0x2282f20202020287 */
        /*03c8*/                   IMUL32I R0, R0, 0x11;                               /* 0x1000000044001ca2 */
        /*03d0*/                   IADD32I R20, R0, 0x3;                               /* 0x080000000c051c02 */
        /*03d8*/                   IADD32I R19, R0, 0x2;                               /* 0x080000000804dc02 */
        /*03e0*/                   IADD32I R17, R0, 0x1;                               /* 0x0800000004045c02 */
        /*03e8*/                   IADD.X R23, RZ, R18;                                /* 0x480000004bf5dc43 */
        /*03f0*/                   MOV R18, R0;                                        /* 0x2800000000049de4 */
        /*03f8*/                   IADD32I R0, R2, 0x20030;                            /* 0x08000800c0201c02 */
                                                                                       /* 0x220042f202c2c287 */
        /*0408*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*0410*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*0418*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*0420*/                   P2R R21, PR, RZ, 0x1;                               /* 0x3000c00007f55c04 */
        /*0428*/               @P0 BRA 0x490;                                          /* 0x40000001800001e7 */
        /*0430*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0438*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
                                                                                       /* 0x22f2020200422047 */
        /*0448*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0450*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0458*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0460*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0468*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0470*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0478*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
                                                                                       /* 0x22020042f2c042f7 */
        /*0488*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0490*/                   ST.E [R22], R18;                                    /* 0x9400000001649c85 */
        /*0498*/                   ISETP.NE.AND P6, PT, R21, RZ, PT;                   /* 0x1a8e0000fd5ddc23 */
        /*04a0*/               @P6 BRA 0x500;                                          /* 0x40000001600019e7 */
        /*04a8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*04b0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*04b8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*04c8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*04d0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*04d8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*04e0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*04e8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*04f0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*04f8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x2220420042f2c047 */
        /*0508*/                   ST.E [R22+0x4], R17;                                /* 0x9400000011645c85 */
        /*0510*/                   ISETP.NE.AND P6, PT, R21, RZ, PT;                   /* 0x1a8e0000fd5ddc23 */
        /*0518*/               @P6 BRA 0x578;                                          /* 0x40000001600019e7 */
        /*0520*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0528*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0530*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0538*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x2202f2f202020047 */
        /*0548*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0550*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0558*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0560*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0568*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0570*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0578*/                   ST.E [R22+0x8], R19;                                /* 0x940000002164dc85 */
                                                                                       /* 0x221200420042f2c7 */
        /*0588*/                   ISETP.NE.AND P6, PT, R21, RZ, PT;                   /* 0x1a8e0000fd5ddc23 */
        /*0590*/               @P6 BRA 0x5f0;                                          /* 0x40000001600019e7 */
        /*0598*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*05a0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*05a8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*05b0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*05b8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
                                                                                       /* 0x228042f2f2020047 */
        /*05c8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*05d0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*05d8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*05e0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*05e8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*05f0*/                   IADD32I R16, R16, 0x4;                              /* 0x0800000011041c02 */
        /*05f8*/                   IADD32I R0.CC, R22, 0x10;                           /* 0x0c00000041601c02 */
                                                                                       /* 0x220042c200420047 */
        /*0608*/                   ISETP.NE.AND P0, PT, R16, RZ, PT;                   /* 0x1a8e0000fd01dc23 */
        /*0610*/                   ST.E [R22+0xc], R20;                                /* 0x9400000031651c85 */
        /*0618*/                   IADD32I R19, R19, 0x4;                              /* 0x080000001134dc02 */
        /*0620*/                   IADD32I R18, R18, 0x4;                              /* 0x0800000011249c02 */
        /*0628*/                   IADD32I R17, R17, 0x4;                              /* 0x0800000011145c02 */
        /*0630*/                   IADD.X R23, RZ, R23;                                /* 0x480000005ff5dc43 */
        /*0638*/                   MOV R22, R0;                                        /* 0x2800000000059de4 */
                                                                                       /* 0x2000000002f2e047 */
        /*0648*/                   IADD32I R20, R20, 0x4;                              /* 0x0800000011451c02 */
        /*0650*/               @P0 BRA 0x3f8;                                          /* 0x4003fff6800001e7 */
        /*0658*/                   EXIT;                                               /* 0x8000000000001de7 */
        /*0660*/                   BRA 0x660;                                          /* 0x4003ffffe0001de7 */
        /*0668*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0670*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0678*/                   NOP;                                                /* 0x4000000000001de4 */
		..............................................................................


		Function : _ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                            /* 0x22f2c2c3f2820307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                    /* 0x2800400110005de4 */
        /*0010*/                   MOV32I R16, 0x0;                         /* 0x1800000000041de2 */
        /*0018*/                   MOV32I R17, 0x0;                         /* 0x1800000000045de2 */
        /*0020*/                   LD.E.64 R16, [R16];                      /* 0x8400000001041ca5 */
        /*0028*/                   IADD32I RZ.CC, R16, -0x8008;             /* 0x0ffffdffe10fdc02 */
        /*0030*/                   ISETP.LT.U32.X.AND P0, PT, R17, RZ, PT;  /* 0x188e0000fd11dc43 */
        /*0038*/               @P0 BRA 0xa0;                                /* 0x40000001800001e7 */
                                                                            /* 0x2202004220420047 */
        /*0048*/                   MOV32I R4, 0x0;                          /* 0x1800000000011de2 */
        /*0050*/                   MOV32I R8, 0xa0;                         /* 0x1800000280021de2 */
        /*0058*/                   MOV32I R5, 0x0;                          /* 0x1800000000015de2 */
        /*0060*/                   MOV32I R12, 0x1;                         /* 0x1800000004031de2 */
        /*0068*/                   MOV32I R6, 0x0;                          /* 0x1800000000019de2 */
        /*0070*/                   MOV R13, RZ;                             /* 0x28000000fc035de4 */
        /*0078*/                   MOV32I R7, 0x0;                          /* 0x180000000001dde2 */
                                                                            /* 0x22c2c042c2f2f207 */
        /*0088*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0090*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0098*/                   JCAL 0x0;                                /* 0x1000000000011c07 */
        /*00a0*/                   IADD32I R16.CC, R16, 0x1;                /* 0x0c00000005041c02 */
        /*00a8*/                   IADD.X R17, RZ, R17;                     /* 0x4800000047f45c43 */
        /*00b0*/                   IADD RZ.CC, -RZ, R16;                    /* 0x4801000043ffde03 */
        /*00b8*/                   ISETP.NE.X.AND P0, PT, R17, RZ, PT;      /* 0x1a8e0000fd11dc63 */
                                                                            /* 0x22004220420042f7 */
        /*00c8*/               @P0 BRA 0x128;                               /* 0x40000001600001e7 */
        /*00d0*/                   MOV32I R4, 0x0;                          /* 0x1800000000011de2 */
        /*00d8*/                   MOV32I R8, 0xd0;                         /* 0x1800000340021de2 */
        /*00e0*/                   MOV32I R5, 0x0;                          /* 0x1800000000015de2 */
        /*00e8*/                   MOV32I R12, 0x1;                         /* 0x1800000004031de2 */
        /*00f0*/                   MOV32I R6, 0x0;                          /* 0x1800000000019de2 */
        /*00f8*/                   MOV R13, RZ;                             /* 0x28000000fc035de4 */
                                                                            /* 0x22020042f2f20207 */
        /*0108*/                   MOV32I R7, 0x0;                          /* 0x180000000001dde2 */
        /*0110*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0118*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0120*/                   JCAL 0x0;                                /* 0x1000000000011c07 */
        /*0128*/                   MOV32I R2, 0x0;                          /* 0x1800000000009de2 */
        /*0130*/                   MOV R6, c[0x0][0x140];                   /* 0x2800400500019de4 */
        /*0138*/                   MOV32I R3, 0x0;                          /* 0x180000000000dde2 */
                                                                            /* 0x23f20042e0427047 */
        /*0148*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0150*/                   MOV R7, c[0x0][0x144];                   /* 0x280040051001dde4 */
        /*0158*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0160*/                   ST.E.64 [R2], R16;                       /* 0x9400000000241ca5 */
        /*0168*/                   LD.E.64 R4, [R10];                       /* 0x8400000000a11ca5 */
        /*0170*/                   MOV32I R8, 0x0;                          /* 0x1800000000021de2 */
        /*0178*/                   MOV32I R9, 0x0;                          /* 0x1800000000025de2 */
                                                                            /* 0x2000000002f042e7 */
        /*0188*/                   ST.E.64 [R8], R4;                        /* 0x9400000000811ca5 */
        /*0190*/                   RED.E.ADD.U64 [R10], R6;                 /* 0x1400000000a19e05 */
        /*0198*/                   EXIT;                                    /* 0x8000000000001de7 */
        /*01a0*/                   BRA 0x1a0;                               /* 0x4003ffffe0001de7 */
        /*01a8*/                   NOP;                                     /* 0x4000000000001de4 */
        /*01b0*/                   NOP;                                     /* 0x4000000000001de4 */
        /*01b8*/                   NOP;                                     /* 0x4000000000001de4 */
		...............................................................................



Fatbin ptx code:
================
arch = sm_30
code version = [6,1]
producer = cuda
host = linux
compile_size = 64bit
compressed








.version 6.1
.target sm_30
.address_size 64


.extern .func __assertfail
(
.param .b64 __assertfail_param_0,
.param .b64 __assertfail_param_1,
.param .b32 __assertfail_param_2,
.param .b64 __assertfail_param_3,
.param .b64 __assertfail_param_4
)
;
.global .align 16 .b8 __DummyClassdata_buffer[8652880];
.global .align 16 .b8 __unnamed_1[242] = {115, 116, 97, 116, 105, 99, 32, 83, 101, 108, 102, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
.global .align 16 .b8 __unnamed_2[294] = {115, 116, 97, 116, 105, 99, 32, 115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 83, 101, 108, 102, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 95, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
.global .align 16 .b8 __unnamed_3[295] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 84, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 70, 105, 101, 108, 100, 95, 60, 84, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 62, 58, 58, 100, 97, 116, 97, 95, 112, 116, 114, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 84, 32, 61, 32, 105, 110, 116, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 48, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 93, 0};
.global .align 16 .b8 __unnamed_4[365] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 84, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 65, 114, 114, 97, 121, 79, 98, 106, 101, 99, 116, 70, 105, 101, 108, 100, 95, 60, 84, 44, 32, 66, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 62, 58, 58, 100, 97, 116, 97, 95, 112, 116, 114, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 84, 32, 61, 32, 115, 116, 100, 58, 58, 97, 114, 114, 97, 121, 60, 105, 110, 116, 44, 32, 54, 52, 85, 76, 62, 44, 32, 66, 32, 61, 32, 105, 110, 116, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 32, 61, 32, 54, 52, 85, 76, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 52, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 93, 0};
.global .align 16 .b8 __unnamed_5[297] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 84, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 70, 105, 101, 108, 100, 95, 60, 84, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 62, 58, 58, 100, 97, 116, 97, 95, 112, 116, 114, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 84, 32, 61, 32, 105, 110, 116, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 50, 54, 48, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 93, 0};
.global .align 16 .b8 __unnamed_6[138] = {99, 104, 97, 114, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 95, 60, 79, 119, 110, 101, 114, 84, 44, 32, 65, 114, 101, 110, 97, 83, 105, 122, 101, 62, 58, 58, 97, 108, 108, 111, 99, 97, 116, 101, 95, 105, 110, 95, 97, 114, 101, 110, 97, 95, 105, 110, 116, 101, 114, 110, 97, 108, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 79, 119, 110, 101, 114, 84, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 65, 114, 101, 110, 97, 83, 105, 122, 101, 32, 61, 32, 48, 85, 76, 93, 0};
.global .align 16 .b8 __unnamed_7[256] = {115, 116, 97, 116, 105, 99, 32, 83, 101, 108, 102, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
.global .align 8 .u64 _ZN4ikra8executor4cuda14d_storage_sizeE;
.global .align 8 .u64 _ZN4ikra8executor4cuda19d_storage_data_headE;
.global .align 8 .u64 _ZN4ikra3soa23d_previous_storage_sizeE;
.global .align 8 .u64 _ZN4ikra3soa18d_arena_allocationE;
.global .align 16 .b8 $str[15] = {105, 100, 32, 60, 61, 32, 67, 97, 112, 97, 99, 105, 116, 121, 0};
.global .align 16 .b8 $str1[27] = {46, 46, 47, 46, 46, 47, 46, 46, 47, 105, 107, 114, 97, 47, 115, 111, 97, 47, 108, 97, 121, 111, 117, 116, 46, 104, 0};
.global .align 8 .b8 $str2[7] = {105, 100, 32, 62, 32, 48, 0};
.global .align 16 .b8 $str3[32] = {97, 114, 101, 110, 97, 95, 104, 101, 97, 100, 95, 32, 43, 32, 98, 121, 116, 101, 115, 32, 60, 32, 65, 114, 101, 110, 97, 83, 105, 122, 101, 0};
.global .align 16 .b8 $str4[28] = {46, 46, 47, 46, 46, 47, 46, 46, 47, 105, 107, 114, 97, 47, 115, 111, 97, 47, 115, 116, 111, 114, 97, 103, 101, 46, 104, 0};
.global .align 16 .b8 $str5[28] = {105, 100, 32, 60, 32, 83, 101, 108, 102, 58, 58, 115, 116, 111, 114, 97, 103, 101, 40, 41, 46, 115, 105, 122, 101, 40, 41, 0};
.global .align 16 .b8 $str6[48] = {114, 101, 105, 110, 116, 101, 114, 112, 114, 101, 116, 95, 99, 97, 115, 116, 60, 117, 105, 110, 116, 112, 116, 114, 95, 116, 62, 40, 115, 111, 97, 95, 97, 114, 114, 97, 121, 41, 32, 37, 32, 56, 32, 61, 61, 32, 48, 0};
.global .align 16 .b8 $str7[47] = {46, 46, 47, 46, 46, 47, 46, 46, 47, 105, 107, 114, 97, 47, 115, 111, 97, 47, 97, 100, 100, 114, 101, 115, 115, 97, 98, 108, 101, 95, 102, 105, 101, 108, 100, 95, 115, 104, 97, 114, 101, 100, 46, 105, 110, 99, 0};

.visible .entry _ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm(
.param .u64 _ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm_param_0
)
{
.reg .pred %p<3>;
.reg .b32 %r<3>;
.reg .b64 %rd<21>;


ld.param.u64 %rd3, [_ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm_param_0];
ld.global.u64 %rd1, [__DummyClassdata_buffer];
setp.lt.u64	%p1, %rd1, 32776;
@%p1 bra BB0_2;

mov.u64 %rd4, $str;
cvta.global.u64 %rd5, %rd4;
mov.u64 %rd6, $str1;
cvta.global.u64 %rd7, %rd6;
mov.u64 %rd8, __unnamed_7;
cvta.global.u64 %rd9, %rd8;
mov.u32 %r1, 160;
mov.u64 %rd10, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd5;
.param .b64 param1;
st.param.b64	[param1+0], %rd7;
.param .b32 param2;
st.param.b32	[param2+0], %r1;
.param .b64 param3;
st.param.b64	[param3+0], %rd9;
.param .b64 param4;
st.param.b64	[param4+0], %rd10;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB0_2:
add.s64 %rd2, %rd1, 1;
setp.ne.s64	%p2, %rd2, 0;
@%p2 bra BB0_4;

mov.u64 %rd11, $str2;
cvta.global.u64 %rd12, %rd11;
mov.u64 %rd13, $str1;
cvta.global.u64 %rd14, %rd13;
mov.u64 %rd15, __unnamed_2;
cvta.global.u64 %rd16, %rd15;
mov.u32 %r2, 208;
mov.u64 %rd17, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd12;
.param .b64 param1;
st.param.b64	[param1+0], %rd14;
.param .b32 param2;
st.param.b32	[param2+0], %r2;
.param .b64 param3;
st.param.b64	[param3+0], %rd16;
.param .b64 param4;
st.param.b64	[param4+0], %rd17;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB0_4:
st.global.u64 [_ZN4ikra8executor4cuda19d_storage_data_headE], %rd2;
mov.u64 %rd18, __DummyClassdata_buffer;
ld.global.u64 %rd19, [__DummyClassdata_buffer];
st.global.u64 [_ZN4ikra8executor4cuda14d_storage_sizeE], %rd19;
atom.global.add.u64 %rd20, [%rd18], %rd3;
ret;
}


.visible .entry _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0_(
.param .u64 _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_0,
.param .u64 _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_1,
.param .u32 _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_2,
.param .u32 _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_3
)
{
.reg .pred %p<12>;
.reg .b32 %r<36>;
.reg .b64 %rd<104>;


ld.param.u64 %rd8, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_0];
ld.param.u64 %rd9, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_1];
ld.param.u32 %r18, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_2];
ld.param.u32 %r19, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_3];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r20, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r20;
setp.ge.u64	%p1, %rd1, %rd9;
@%p1 bra BB1_20;

add.s64 %rd2, %rd1, %rd8;
setp.lt.u64	%p2, %rd2, 32776;
@%p2 bra BB1_3;

mov.u64 %rd10, $str;
cvta.global.u64 %rd11, %rd10;
mov.u64 %rd12, $str1;
cvta.global.u64 %rd13, %rd12;
mov.u64 %rd14, __unnamed_7;
cvta.global.u64 %rd15, %rd14;
mov.u32 %r21, 160;
mov.u64 %rd16, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd11;
.param .b64 param1;
st.param.b64	[param1+0], %rd13;
.param .b32 param2;
st.param.b32	[param2+0], %r21;
.param .b64 param3;
st.param.b64	[param3+0], %rd15;
.param .b64 param4;
st.param.b64	[param4+0], %rd16;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_3:
add.s64 %rd3, %rd2, 1;
setp.ne.s64	%p3, %rd3, 0;
@%p3 bra BB1_5;

mov.u64 %rd17, $str2;
cvta.global.u64 %rd18, %rd17;
mov.u64 %rd19, $str1;
cvta.global.u64 %rd20, %rd19;
mov.u64 %rd21, __unnamed_2;
cvta.global.u64 %rd22, %rd21;
mov.u32 %r22, 208;
mov.u64 %rd23, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd18;
.param .b64 param1;
st.param.b64	[param1+0], %rd20;
.param .b32 param2;
st.param.b32	[param2+0], %r22;
.param .b64 param3;
st.param.b64	[param3+0], %rd22;
.param .b64 param4;
st.param.b64	[param4+0], %rd23;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_5:
setp.eq.s64	%p4, %rd3, 0;
@%p4 bra BB1_20;

mov.u64 %rd24, __DummyClassdata_buffer;
add.s64 %rd25, %rd24, 16;
cvta.global.u64 %rd26, %rd25;
and.b64 %rd27, %rd26, 7;
setp.eq.s64	%p5, %rd27, 0;
@%p5 bra BB1_8;

mov.u64 %rd28, $str6;
cvta.global.u64 %rd29, %rd28;
mov.u64 %rd30, $str7;
cvta.global.u64 %rd31, %rd30;
mov.u64 %rd32, __unnamed_3;
cvta.global.u64 %rd33, %rd32;
mov.u32 %r23, 123;
mov.u64 %rd34, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd29;
.param .b64 param1;
st.param.b64	[param1+0], %rd31;
.param .b32 param2;
st.param.b32	[param2+0], %r23;
.param .b64 param3;
st.param.b64	[param3+0], %rd33;
.param .b64 param4;
st.param.b64	[param4+0], %rd34;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_8:
shl.b64 %rd35, %rd3, 2;
add.s64 %rd37, %rd24, %rd35;
st.global.u32 [%rd37+16], %r18;
add.s64 %rd38, %rd24, 8521776;
cvta.global.u64 %rd39, %rd38;
and.b64 %rd40, %rd39, 7;
setp.eq.s64	%p6, %rd40, 0;
@%p6 bra BB1_10;

mov.u64 %rd41, $str6;
cvta.global.u64 %rd42, %rd41;
mov.u64 %rd43, $str7;
cvta.global.u64 %rd44, %rd43;
mov.u64 %rd45, __unnamed_5;
cvta.global.u64 %rd46, %rd45;
mov.u32 %r24, 123;
mov.u64 %rd47, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd42;
.param .b64 param1;
st.param.b64	[param1+0], %rd44;
.param .b32 param2;
st.param.b32	[param2+0], %r24;
.param .b64 param3;
st.param.b64	[param3+0], %rd46;
.param .b64 param4;
st.param.b64	[param4+0], %rd47;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_10:
st.global.u32 [%rd37+8521776], %r19;
add.s64 %rd52, %rd8, %rd1;
mul.lo.s64 %rd53, %rd52, 17;
add.s64 %rd54, %rd53, 3;
cvt.u32.u64	%r34, %rd54;
add.s64 %rd55, %rd53, 2;
cvt.u32.u64	%r33, %rd55;
cvt.u32.u64	%r32, %rd53;
add.s64 %rd56, %rd53, 1;
cvt.u32.u64	%r31, %rd56;
shl.b64 %rd57, %rd3, 8;
add.s64 %rd58, %rd57, %rd24;
add.s64 %rd103, %rd58, 131120;
mov.u32 %r35, -64;

BB1_11:
add.s64 %rd60, %rd24, 131120;
cvta.global.u64 %rd61, %rd60;
and.b64 %rd62, %rd61, 7;
setp.eq.s64	%p7, %rd62, 0;
@%p7 bra BB1_13;

mov.u64 %rd63, $str6;
cvta.global.u64 %rd64, %rd63;
mov.u64 %rd65, $str7;
cvta.global.u64 %rd66, %rd65;
mov.u64 %rd67, __unnamed_4;
cvta.global.u64 %rd68, %rd67;
mov.u32 %r27, 123;
mov.u64 %rd69, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd64;
.param .b64 param1;
st.param.b64	[param1+0], %rd66;
.param .b32 param2;
st.param.b32	[param2+0], %r27;
.param .b64 param3;
st.param.b64	[param3+0], %rd68;
.param .b64 param4;
st.param.b64	[param4+0], %rd69;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_13:
st.global.u32 [%rd103], %r32;
@%p7 bra BB1_15;

mov.u64 %rd74, $str6;
cvta.global.u64 %rd75, %rd74;
mov.u64 %rd76, $str7;
cvta.global.u64 %rd77, %rd76;
mov.u64 %rd78, __unnamed_4;
cvta.global.u64 %rd79, %rd78;
mov.u32 %r28, 123;
mov.u64 %rd80, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd75;
.param .b64 param1;
st.param.b64	[param1+0], %rd77;
.param .b32 param2;
st.param.b32	[param2+0], %r28;
.param .b64 param3;
st.param.b64	[param3+0], %rd79;
.param .b64 param4;
st.param.b64	[param4+0], %rd80;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_15:
st.global.u32 [%rd103+4], %r31;
@%p7 bra BB1_17;

mov.u64 %rd85, $str6;
cvta.global.u64 %rd86, %rd85;
mov.u64 %rd87, $str7;
cvta.global.u64 %rd88, %rd87;
mov.u64 %rd89, __unnamed_4;
cvta.global.u64 %rd90, %rd89;
mov.u32 %r29, 123;
mov.u64 %rd91, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd86;
.param .b64 param1;
st.param.b64	[param1+0], %rd88;
.param .b32 param2;
st.param.b32	[param2+0], %r29;
.param .b64 param3;
st.param.b64	[param3+0], %rd90;
.param .b64 param4;
st.param.b64	[param4+0], %rd91;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_17:
st.global.u32 [%rd103+8], %r33;
@%p7 bra BB1_19;

mov.u64 %rd96, $str6;
cvta.global.u64 %rd97, %rd96;
mov.u64 %rd98, $str7;
cvta.global.u64 %rd99, %rd98;
mov.u64 %rd100, __unnamed_4;
cvta.global.u64 %rd101, %rd100;
mov.u32 %r30, 123;
mov.u64 %rd102, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd97;
.param .b64 param1;
st.param.b64	[param1+0], %rd99;
.param .b32 param2;
st.param.b32	[param2+0], %r30;
.param .b64 param3;
st.param.b64	[param3+0], %rd101;
.param .b64 param4;
st.param.b64	[param4+0], %rd102;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB1_19:
add.s64 %rd7, %rd103, 16;
st.global.u32 [%rd103+12], %r34;
add.s32 %r34, %r34, 4;
add.s32 %r33, %r33, 4;
add.s32 %r32, %r32, 4;
add.s32 %r31, %r31, 4;
add.s32 %r35, %r35, 4;
setp.ne.s32	%p11, %r35, 0;
mov.u64 %rd103, %rd7;
@%p11 bra BB1_11;

BB1_20:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_2,
.param .u32 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_3
)
{
.reg .pred %p<17>;
.reg .b32 %r<48>;
.reg .b64 %rd<166>;


ld.param.u64 %rd9, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_1];
ld.param.u64 %rd10, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_2];
ld.param.u32 %r11, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_3];
mov.u32 %r12, %tid.x;
mov.u32 %r13, %ntid.x;
mov.u32 %r14, %ctaid.x;
mad.lo.s32 %r15, %r13, %r14, %r12;
cvt.s64.s32	%rd1, %r15;
setp.ge.u64	%p1, %rd1, %rd10;
@%p1 bra BB2_31;

add.s64 %rd11, %rd9, %rd1;
add.s64 %rd12, %rd11, -1;
ld.global.u64 %rd13, [__DummyClassdata_buffer];
setp.gt.u64	%p2, %rd13, %rd12;
@%p2 bra BB2_3;

mov.u64 %rd14, $str5;
cvta.global.u64 %rd15, %rd14;
mov.u64 %rd16, $str1;
cvta.global.u64 %rd17, %rd16;
mov.u64 %rd18, __unnamed_1;
cvta.global.u64 %rd19, %rd18;
mov.u32 %r16, 171;
mov.u64 %rd20, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd15;
.param .b64 param1;
st.param.b64	[param1+0], %rd17;
.param .b32 param2;
st.param.b32	[param2+0], %r16;
.param .b64 param3;
st.param.b64	[param3+0], %rd19;
.param .b64 param4;
st.param.b64	[param4+0], %rd20;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_3:
setp.ne.s64	%p3, %rd11, 0;
@%p3 bra BB2_5;

mov.u64 %rd21, $str2;
cvta.global.u64 %rd22, %rd21;
mov.u64 %rd23, $str1;
cvta.global.u64 %rd24, %rd23;
mov.u64 %rd25, __unnamed_2;
cvta.global.u64 %rd26, %rd25;
mov.u32 %r17, 208;
mov.u64 %rd27, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd22;
.param .b64 param1;
st.param.b64	[param1+0], %rd24;
.param .b32 param2;
st.param.b32	[param2+0], %r17;
.param .b64 param3;
st.param.b64	[param3+0], %rd26;
.param .b64 param4;
st.param.b64	[param4+0], %rd27;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_5:
shl.b64 %rd28, %rd11, 2;
mov.u64 %rd29, __DummyClassdata_buffer;
add.s64 %rd30, %rd29, %rd28;
add.s64 %rd3, %rd30, 16;
add.s64 %rd4, %rd30, 8521776;
shl.b64 %rd31, %rd11, 8;
add.s64 %rd32, %rd31, %rd29;
add.s64 %rd165, %rd32, 131120;
mov.u32 %r47, -64;

BB2_6:
add.s64 %rd34, %rd29, 16;
cvta.global.u64 %rd35, %rd34;
and.b64 %rd36, %rd35, 7;
setp.eq.s64	%p4, %rd36, 0;
@%p4 bra BB2_8;

mov.u64 %rd37, $str6;
cvta.global.u64 %rd38, %rd37;
mov.u64 %rd39, $str7;
cvta.global.u64 %rd40, %rd39;
mov.u64 %rd41, __unnamed_3;
cvta.global.u64 %rd42, %rd41;
mov.u32 %r19, 123;
mov.u64 %rd43, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd38;
.param .b64 param1;
st.param.b64	[param1+0], %rd40;
.param .b32 param2;
st.param.b32	[param2+0], %r19;
.param .b64 param3;
st.param.b64	[param3+0], %rd42;
.param .b64 param4;
st.param.b64	[param4+0], %rd43;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_8:
ld.global.u32 %r2, [%rd3];
add.s64 %rd45, %rd29, 8521776;
cvta.global.u64 %rd46, %rd45;
and.b64 %rd47, %rd46, 7;
setp.eq.s64	%p5, %rd47, 0;
@%p5 bra BB2_10;

mov.u64 %rd48, $str6;
cvta.global.u64 %rd49, %rd48;
mov.u64 %rd50, $str7;
cvta.global.u64 %rd51, %rd50;
mov.u64 %rd52, __unnamed_5;
cvta.global.u64 %rd53, %rd52;
mov.u32 %r20, 123;
mov.u64 %rd54, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd49;
.param .b64 param1;
st.param.b64	[param1+0], %rd51;
.param .b32 param2;
st.param.b32	[param2+0], %r20;
.param .b64 param3;
st.param.b64	[param3+0], %rd53;
.param .b64 param4;
st.param.b64	[param4+0], %rd54;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_10:
ld.global.u32 %r21, [%rd4];
add.s32 %r22, %r2, %r11;
add.s32 %r3, %r22, %r21;
add.s64 %rd56, %rd29, 131120;
cvta.global.u64 %rd57, %rd56;
and.b64 %rd58, %rd57, 7;
setp.eq.s64	%p6, %rd58, 0;
@%p6 bra BB2_12;

mov.u64 %rd59, $str6;
cvta.global.u64 %rd60, %rd59;
mov.u64 %rd61, $str7;
cvta.global.u64 %rd62, %rd61;
mov.u64 %rd63, __unnamed_4;
cvta.global.u64 %rd64, %rd63;
mov.u32 %r23, 123;
mov.u64 %rd65, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd60;
.param .b64 param1;
st.param.b64	[param1+0], %rd62;
.param .b32 param2;
st.param.b32	[param2+0], %r23;
.param .b64 param3;
st.param.b64	[param3+0], %rd64;
.param .b64 param4;
st.param.b64	[param4+0], %rd65;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_12:
ld.global.u32 %r24, [%rd165];
add.s32 %r25, %r3, %r24;
st.global.u32 [%rd165], %r25;
@%p4 bra BB2_14;

mov.u64 %rd70, $str6;
cvta.global.u64 %rd71, %rd70;
mov.u64 %rd72, $str7;
cvta.global.u64 %rd73, %rd72;
mov.u64 %rd74, __unnamed_3;
cvta.global.u64 %rd75, %rd74;
mov.u32 %r26, 123;
mov.u64 %rd76, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd71;
.param .b64 param1;
st.param.b64	[param1+0], %rd73;
.param .b32 param2;
st.param.b32	[param2+0], %r26;
.param .b64 param3;
st.param.b64	[param3+0], %rd75;
.param .b64 param4;
st.param.b64	[param4+0], %rd76;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_14:
ld.global.u32 %r4, [%rd3];
@%p5 bra BB2_16;

mov.u64 %rd81, $str6;
cvta.global.u64 %rd82, %rd81;
mov.u64 %rd83, $str7;
cvta.global.u64 %rd84, %rd83;
mov.u64 %rd85, __unnamed_5;
cvta.global.u64 %rd86, %rd85;
mov.u32 %r27, 123;
mov.u64 %rd87, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd82;
.param .b64 param1;
st.param.b64	[param1+0], %rd84;
.param .b32 param2;
st.param.b32	[param2+0], %r27;
.param .b64 param3;
st.param.b64	[param3+0], %rd86;
.param .b64 param4;
st.param.b64	[param4+0], %rd87;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_16:
ld.global.u32 %r28, [%rd4];
add.s32 %r29, %r4, %r11;
add.s32 %r5, %r29, %r28;
@%p6 bra BB2_18;

mov.u64 %rd92, $str6;
cvta.global.u64 %rd93, %rd92;
mov.u64 %rd94, $str7;
cvta.global.u64 %rd95, %rd94;
mov.u64 %rd96, __unnamed_4;
cvta.global.u64 %rd97, %rd96;
mov.u32 %r30, 123;
mov.u64 %rd98, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd93;
.param .b64 param1;
st.param.b64	[param1+0], %rd95;
.param .b32 param2;
st.param.b32	[param2+0], %r30;
.param .b64 param3;
st.param.b64	[param3+0], %rd97;
.param .b64 param4;
st.param.b64	[param4+0], %rd98;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_18:
ld.global.u32 %r31, [%rd165+4];
add.s32 %r32, %r5, %r31;
st.global.u32 [%rd165+4], %r32;
@%p4 bra BB2_20;

mov.u64 %rd103, $str6;
cvta.global.u64 %rd104, %rd103;
mov.u64 %rd105, $str7;
cvta.global.u64 %rd106, %rd105;
mov.u64 %rd107, __unnamed_3;
cvta.global.u64 %rd108, %rd107;
mov.u32 %r33, 123;
mov.u64 %rd109, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd104;
.param .b64 param1;
st.param.b64	[param1+0], %rd106;
.param .b32 param2;
st.param.b32	[param2+0], %r33;
.param .b64 param3;
st.param.b64	[param3+0], %rd108;
.param .b64 param4;
st.param.b64	[param4+0], %rd109;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_20:
ld.global.u32 %r6, [%rd3];
@%p5 bra BB2_22;

mov.u64 %rd114, $str6;
cvta.global.u64 %rd115, %rd114;
mov.u64 %rd116, $str7;
cvta.global.u64 %rd117, %rd116;
mov.u64 %rd118, __unnamed_5;
cvta.global.u64 %rd119, %rd118;
mov.u32 %r34, 123;
mov.u64 %rd120, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd115;
.param .b64 param1;
st.param.b64	[param1+0], %rd117;
.param .b32 param2;
st.param.b32	[param2+0], %r34;
.param .b64 param3;
st.param.b64	[param3+0], %rd119;
.param .b64 param4;
st.param.b64	[param4+0], %rd120;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_22:
ld.global.u32 %r35, [%rd4];
add.s32 %r36, %r6, %r11;
add.s32 %r7, %r36, %r35;
@%p6 bra BB2_24;

mov.u64 %rd125, $str6;
cvta.global.u64 %rd126, %rd125;
mov.u64 %rd127, $str7;
cvta.global.u64 %rd128, %rd127;
mov.u64 %rd129, __unnamed_4;
cvta.global.u64 %rd130, %rd129;
mov.u32 %r37, 123;
mov.u64 %rd131, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd126;
.param .b64 param1;
st.param.b64	[param1+0], %rd128;
.param .b32 param2;
st.param.b32	[param2+0], %r37;
.param .b64 param3;
st.param.b64	[param3+0], %rd130;
.param .b64 param4;
st.param.b64	[param4+0], %rd131;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_24:
ld.global.u32 %r38, [%rd165+8];
add.s32 %r39, %r7, %r38;
st.global.u32 [%rd165+8], %r39;
@%p4 bra BB2_26;

mov.u64 %rd136, $str6;
cvta.global.u64 %rd137, %rd136;
mov.u64 %rd138, $str7;
cvta.global.u64 %rd139, %rd138;
mov.u64 %rd140, __unnamed_3;
cvta.global.u64 %rd141, %rd140;
mov.u32 %r40, 123;
mov.u64 %rd142, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd137;
.param .b64 param1;
st.param.b64	[param1+0], %rd139;
.param .b32 param2;
st.param.b32	[param2+0], %r40;
.param .b64 param3;
st.param.b64	[param3+0], %rd141;
.param .b64 param4;
st.param.b64	[param4+0], %rd142;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_26:
ld.global.u32 %r8, [%rd3];
@%p5 bra BB2_28;

mov.u64 %rd147, $str6;
cvta.global.u64 %rd148, %rd147;
mov.u64 %rd149, $str7;
cvta.global.u64 %rd150, %rd149;
mov.u64 %rd151, __unnamed_5;
cvta.global.u64 %rd152, %rd151;
mov.u32 %r41, 123;
mov.u64 %rd153, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd148;
.param .b64 param1;
st.param.b64	[param1+0], %rd150;
.param .b32 param2;
st.param.b32	[param2+0], %r41;
.param .b64 param3;
st.param.b64	[param3+0], %rd152;
.param .b64 param4;
st.param.b64	[param4+0], %rd153;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_28:
ld.global.u32 %r42, [%rd4];
add.s32 %r43, %r8, %r11;
add.s32 %r9, %r43, %r42;
@%p6 bra BB2_30;

mov.u64 %rd158, $str6;
cvta.global.u64 %rd159, %rd158;
mov.u64 %rd160, $str7;
cvta.global.u64 %rd161, %rd160;
mov.u64 %rd162, __unnamed_4;
cvta.global.u64 %rd163, %rd162;
mov.u32 %r44, 123;
mov.u64 %rd164, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd159;
.param .b64 param1;
st.param.b64	[param1+0], %rd161;
.param .b32 param2;
st.param.b32	[param2+0], %r44;
.param .b64 param3;
st.param.b64	[param3+0], %rd163;
.param .b64 param4;
st.param.b64	[param4+0], %rd164;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}

BB2_30:
ld.global.u32 %r45, [%rd165+12];
add.s32 %r46, %r9, %r45;
add.s64 %rd8, %rd165, 16;
st.global.u32 [%rd165+12], %r46;
add.s32 %r47, %r47, 4;
setp.ne.s32	%p16, %r47, 0;
mov.u64 %rd165, %rd8;
@%p16 bra BB2_6;

BB2_31:
ret;
}


.visible .entry _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m(
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_1
)
{
.reg .b32 %r<4>;
.reg .b64 %rd<14>;


ld.param.u64 %rd1, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_0];
cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, $str3;
cvta.global.u64 %rd4, %rd3;
mov.u64 %rd5, $str4;
cvta.global.u64 %rd6, %rd5;
mov.u64 %rd7, __unnamed_6;
cvta.global.u64 %rd8, %rd7;
mov.u32 %r1, 319;
mov.u64 %rd9, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd4;
.param .b64 param1;
st.param.b64	[param1+0], %rd6;
.param .b32 param2;
st.param.b32	[param2+0], %r1;
.param .b64 param3;
st.param.b64	[param3+0], %rd8;
.param .b64 param4;
st.param.b64	[param4+0], %rd9;
call.uni 
__assertfail, 
(
param0, 
param1, 
param2, 
param3, 
param4
);


	}
	ld.param.u32 %r2, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_1];
add.s64 %rd10, %rd2, 8;
atom.global.add.u32 %r3, [%rd10], %r2;
cvt.u64.u32	%rd11, %r3;
add.s64 %rd12, %rd1, %rd11;
add.s64 %rd13, %rd12, 8;
st.global.u64 [_ZN4ikra3soa18d_arena_allocationE], %rd13;
ret;
}


.visible .entry _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m(
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_1
)
{
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_0];
ld.param.u64 %rd2, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm0EEEEEvPT_m_param_1];
cvta.to.global.u64 %rd3, %rd1;
atom.global.add.u64 %rd4, [%rd3], %rd2;
st.global.u64 [_ZN4ikra3soa23d_previous_storage_sizeE], %rd4;
ret;
}


.visible .entry _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm0EEEJEEEvPT_DpT0_(
.param .u64 _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm0EEEJEEEvPT_DpT0__param_0
)
{
.reg .pred %p<2>;
.reg .b32 %r<2>;
.reg .b64 %rd<4>;


ld.param.u64 %rd1, [_ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm0EEEJEEEvPT_DpT0__param_0];
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB5_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
mov.u32 %r1, 0;
st.global.u32 [%rd2+8], %r1;

BB5_2:
ret;
}


