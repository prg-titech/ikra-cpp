
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
  GLOBAL:8917655
 Function _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEJEEEvPT_DpT0_:
  REG:4 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m:
  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m:
  REG:18 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:336 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0_:
  REG:28 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:348 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0_:
  REG:30 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:344 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN4ikra8executor4cuda27increment_class_size_kernelI10DummyClassEEvm:
  REG:18 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:328 TEXTURE:0 SURFACE:0 SAMPLER:0

	code for sm_30
		Function : _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEJEEEvPT_DpT0_
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
        /*0048*/                   ST.E [R2+0x100008], RZ;                        /* 0x94004000202fdc85 */
        /*0050*/                   EXIT;                                          /* 0x8000000000001de7 */
        /*0058*/                   BRA 0x58;                                      /* 0x4003ffffe0001de7 */
        /*0060*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0068*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0070*/                   NOP;                                           /* 0x4000000000001de4 */
        /*0078*/                   NOP;                                           /* 0x4000000000001de4 */
		..................................................................................................................


		Function : _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m
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
		..........................................................................................................


		Function : _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                           /* 0x2202c3f282c28307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                   /* 0x2800400110005de4 */
        /*0010*/                   MOV R16, c[0x0][0x140];                 /* 0x2800400500041de4 */
        /*0018*/                   IADD32I R16.CC, R16, 0x100008;          /* 0x0c00400021041c02 */
        /*0020*/                   IADD.X R17, RZ, c[0x0][0x144];          /* 0x4800400513f45c43 */
        /*0028*/                   LD.E R0, [R16];                         /* 0x8400000001001c85 */
        /*0030*/                   IADD R0.CC, R0, c[0x0][0x148];          /* 0x4801400520001c03 */
        /*0038*/                   IADD.X R2, RZ, c[0x0][0x14c];           /* 0x4800400533f09c43 */
                                                                           /* 0x2220420042f2c2c7 */
        /*0048*/                   IADD32I RZ.CC, R0, -0x100000;           /* 0x0fffc000000fdc02 */
        /*0050*/                   ISETP.LT.U32.X.AND P0, PT, R2, RZ, PT;  /* 0x188e0000fc21dc43 */
        /*0058*/               @P0 BRA 0xb8;                               /* 0x40000001600001e7 */
        /*0060*/                   MOV32I R4, 0x0;                         /* 0x1800000000011de2 */
        /*0068*/                   MOV32I R8, 0x13f;                       /* 0x18000004fc021de2 */
        /*0070*/                   MOV32I R5, 0x0;                         /* 0x1800000000015de2 */
        /*0078*/                   MOV32I R12, 0x1;                        /* 0x1800000004031de2 */
                                                                           /* 0x2202f2f202020047 */
        /*0088*/                   MOV32I R6, 0x0;                         /* 0x1800000000019de2 */
        /*0090*/                   MOV R13, RZ;                            /* 0x28000000fc035de4 */
        /*0098*/                   MOV32I R7, 0x0;                         /* 0x180000000001dde2 */
        /*00a0*/                   MOV32I R10, 0x0;                        /* 0x1800000000029de2 */
        /*00a8*/                   MOV32I R11, 0x0;                        /* 0x180000000002dde2 */
        /*00b0*/                   JCAL 0x0;                               /* 0x1000000000011c07 */
        /*00b8*/                   VOTE.ALL R0, PT, PT;                    /* 0x49c0000000701c04 */
                                                                           /* 0x2202c04280427047 */
        /*00c8*/                   S2R R3, SR_LANEID;                      /* 0x2c0000000000dc04 */
        /*00d0*/                   MOV32I R6, 0x0;                         /* 0x1800000000019de2 */
        /*00d8*/                   FLO.U32 R4, R0;                         /* 0x7800000000011c03 */
        /*00e0*/                   POPC R2, R0, R0;                        /* 0x5400000000009c04 */
        /*00e8*/                   ISETP.EQ.U32.AND P0, PT, R4, R3, PT;    /* 0x190e00000c41dc03 */
        /*00f0*/                   IMUL R3, R2, c[0x0][0x148];             /* 0x500040052020dca3 */
        /*00f8*/               @P0 ATOM.E.ADD R3, [R16], R3;               /* 0x547e18000100c005 */
                                                                           /* 0x228373f28042d047 */
        /*0108*/                   S2R R5, SR_LTMASK;                      /* 0x2c000000e4015c04 */
        /*0110*/                   MOV32I R2, 0x0;                         /* 0x1800000000009de2 */
        /*0118*/                   LOP.AND R0, R5, R0;                     /* 0x6800000000501c03 */
        /*0120*/                   CCTL.E.IV [R16];                        /* 0x9c00000001001ca5 */
        /*0128*/                   POPC R0, R0, R0;                        /* 0x5400000000001c04 */
        /*0130*/                   SHFL.IDX PT, R4, R3, R4, 0x1f;          /* 0x8c007c0010311f45 */
        /*0138*/                   IMAD R0, R0, c[0x0][0x148], R4;         /* 0x2008400520001ca3 */
                                                                           /* 0x22f04282c042c047 */
        /*0148*/                   IADD R0.CC, R0, c[0x0][0x140];          /* 0x4801400500001c03 */
        /*0150*/                   MOV R3, R6;                             /* 0x280000001800dde4 */
        /*0158*/                   IADD.X R5, RZ, c[0x0][0x144];           /* 0x4800400513f15c43 */
        /*0160*/                   IADD32I R4.CC, R0, 0x8;                 /* 0x0c00000020011c02 */
        /*0168*/                   IADD.X R5, RZ, R5;                      /* 0x4800000017f15c43 */
        /*0170*/                   ST.E.64 [R2], R4;                       /* 0x9400000000211ca5 */
        /*0178*/                   EXIT;                                   /* 0x8000000000001de7 */
        /*0180*/                   BRA 0x180;                              /* 0x4003ffffe0001de7 */
        /*0188*/                   NOP;                                    /* 0x4000000000001de4 */
        /*0190*/                   NOP;                                    /* 0x4000000000001de4 */
        /*0198*/                   NOP;                                    /* 0x4000000000001de4 */
        /*01a0*/                   NOP;                                    /* 0x4000000000001de4 */
        /*01a8*/                   NOP;                                    /* 0x4000000000001de4 */
        /*01b0*/                   NOP;                                    /* 0x4000000000001de4 */
        /*01b8*/                   NOP;                                    /* 0x4000000000001de4 */
		......................................................................................................


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
        /*00b8*/                   MOV32I R8, 0xac;                                    /* 0x18000002b0021de2 */
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
        /*0150*/                   MOV32I R8, 0xd1;                                    /* 0x1800000344021de2 */
        /*0158*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0160*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0168*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0170*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0178*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x2200428042f2f2f7 */
        /*0188*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0190*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0198*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*01a0*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
        /*01a8*/                   SHL R4, R16, 0x3;                                   /* 0x6000c0000d011c03 */
        /*01b0*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*01b8*/                   MOV32I R0, 0x3;                                     /* 0x180000000c001de2 */
                                                                                       /* 0x229220420042b047 */
        /*01c8*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*01d0*/                   SHL R3, R17, -0x1d;                                 /* 0x6000ffff8d10dc03 */
        /*01d8*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*01e0*/                   MOV32I R8, 0x0;                                     /* 0x1800000000021de2 */
        /*01e8*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*01f0*/                   MOV R19, RZ;                                        /* 0x28000000fc04dde4 */
        /*01f8*/                   SHR.U32 R7, R17, 0x1d;                              /* 0x5800c0007511dc03 */
                                                                                       /* 0x2202023042804237 */
        /*0208*/                   CSETP.SFT.AND P1, PT, CC, PT;                       /* 0x040e00000153dc04 */
        /*0210*/                   IADD32I R0.CC, R2, 0x120030;                        /* 0x0c004800c0201c02 */
        /*0218*/                   SHL R5, R17, -0x1e;                                 /* 0x6000ffff89115c03 */
        /*0220*/                   SHL R6, R16, 0x2;                                   /* 0x6000c00009019c03 */
        /*0228*/               @P1 IADD R3, R4, R7;                                    /* 0x480000001c40c403 */
        /*0230*/                   IADD.X R16, RZ, R8;                                 /* 0x4800000023f41c43 */
        /*0238*/                   IMAD.U32.U32 R22.CC, R17, 0x8, R2;                  /* 0x2005c00021159c03 */
                                                                                       /* 0x22c202b202027237 */
        /*0248*/                   SHL R4, R17, 0x2;                                   /* 0x6000c00009111c03 */
        /*0250*/                   SHR.U32 R17, R17, 0x1e;                             /* 0x5800c00079145c03 */
        /*0258*/                   IADD.X R23, R3, R8;                                 /* 0x480000002035dc43 */
        /*0260*/                   IADD R24.CC, R4, R2;                                /* 0x4801000008461c03 */
        /*0268*/               @P0 IADD R5, R6, R17;                                   /* 0x4800000044614003 */
        /*0270*/                   IADD.X R25, R5, R8;                                 /* 0x4800000020565c43 */
        /*0278*/                   IADD R17.CC, R0, R4;                                /* 0x4801000010045c03 */
                                                                                       /* 0x22f202c2c28282f7 */
        /*0288*/                   IADD.X R16, R16, R5;                                /* 0x4800000015041c43 */
        /*0290*/                   IADD32I R0, R2, 0x100010;                           /* 0x0800400040201c02 */
        /*0298*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*02a0*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*02a8*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*02b0*/                   P2R R21, PR, RZ, 0x1;                               /* 0x3000c00007f55c04 */
        /*02b8*/               @P0 BRA 0x320;                                          /* 0x40000001800001e7 */
                                                                                       /* 0x2202004220420047 */
        /*02c8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*02d0*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*02d8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*02e0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*02e8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*02f0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*02f8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
                                                                                       /* 0x22c2828042f2f207 */
        /*0308*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0310*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0318*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0320*/                   LD.E R18, [R24+0x100010];                           /* 0x8400400041849c85 */
        /*0328*/                   IADD32I R0, R2, 0x860770;                           /* 0x0802181dc0201c02 */
        /*0330*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*0338*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
                                                                                       /* 0x2220420042f202c7 */
        /*0348*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*0350*/                   P2R R20, PR, RZ, 0x1;                               /* 0x3000c00007f51c04 */
        /*0358*/               @P0 BRA 0x3b8;                                          /* 0x40000001600001e7 */
        /*0360*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0368*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0370*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0378*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x2202f2f202020047 */
        /*0388*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0390*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0398*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*03a0*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*03a8*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*03b0*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*03b8*/                   LD.E R3, [R24+0x860770];                            /* 0x8402181dc180dc85 */
                                                                                       /* 0x2202f043f0420047 */
        /*03c8*/                   ISETP.LT.U32.AND P0, PT, R19, 0x38, PT;             /* 0x188ec000e131dc03 */
        /*03d0*/                   IADD R18, R18, c[0x0][0x158];                       /* 0x4800400561249c03 */
        /*03d8*/                   MOV R4, R17;                                        /* 0x2800000044011de4 */
        /*03e0*/                   MOV R5, R16;                                        /* 0x2800000040015de4 */
        /*03e8*/                   IADD R18, R18, R3;                                  /* 0x480000000d249c03 */
        /*03f0*/               @P0 BRA 0x4b0;                                          /* 0x40000002e00001e7 */
        /*03f8*/                   LD.E.64 R26, [R22+0x820730];                        /* 0x8402081cc1669ca5 */
                                                                                       /* 0x22020042f2c2c3f7 */
        /*0408*/                   SSY 0x490;                                          /* 0x6000000200001c07 */
        /*0410*/                   IADD RZ.CC, -RZ, R26;                               /* 0x480100006bffde03 */
        /*0418*/                   ISETP.NE.X.AND P0, PT, R27, RZ, PT;                 /* 0x1a8e0000fdb1dc63 */
        /*0420*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0428*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0430*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*0438*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*0448*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0450*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0458*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0460*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0468*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0470*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0478*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x23f042f2c042c2f7 */
        /*0488*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*0490*/                   IMAD R0.CC, R19, 0x4, R26;                          /* 0x2035c00011301ca3 */
        /*0498*/                   IMAD.HI.X R26, R19, 0x4, R27;                       /* 0x20b6c00011369ce3 */
        /*04a0*/                   IADD32I R4.CC, R0, -0xe0;                           /* 0x0ffffffc80011c02 */
        /*04a8*/                   IADD32I.X R5, R26, -0x1;                            /* 0x0bfffffffda15c42 */
        /*04b0*/                   LD.E R3, [R4];                                      /* 0x840000000040dc85 */
        /*04b8*/                   ISETP.NE.AND P6, PT, R21, RZ, PT;                   /* 0x1a8e0000fd5ddc23 */
                                                                                       /* 0x2220420042f04287 */
        /*04c8*/                   IADD R3, R18, R3;                                   /* 0x480000000d20dc03 */
        /*04d0*/                   ST.E [R4], R3;                                      /* 0x940000000040dc85 */
        /*04d8*/               @P6 BRA 0x538;                                          /* 0x40000001600019e7 */
        /*04e0*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*04e8*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*04f0*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*04f8*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x2202f2f202020047 */
        /*0508*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0510*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0518*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0520*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0528*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0530*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0538*/                   LD.E R18, [R24+0x100010];                           /* 0x8400400041849c85 */
                                                                                       /* 0x221200420042f2c7 */
        /*0548*/                   ISETP.NE.AND P6, PT, R20, RZ, PT;                   /* 0x1a8e0000fd4ddc23 */
        /*0550*/               @P6 BRA 0x5b0;                                          /* 0x40000001600019e7 */
        /*0558*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0560*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0568*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0570*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0578*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
                                                                                       /* 0x228042f2f2020047 */
        /*0588*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0590*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0598*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*05a0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*05a8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*05b0*/                   LD.E R3, [R24+0x860770];                            /* 0x8402181dc180dc85 */
        /*05b8*/                   IADD32I R19, R19, 0x1;                              /* 0x080000000534dc02 */
                                                                                       /* 0x22c3f042f043f047 */
        /*05c8*/                   IADD R18, R18, c[0x0][0x158];                       /* 0x4800400561249c03 */
        /*05d0*/                   ISETP.LT.U32.AND P0, PT, R19, 0x38, PT;             /* 0x188ec000e131dc03 */
        /*05d8*/                   IADD R18, R18, R3;                                  /* 0x480000000d249c03 */
        /*05e0*/               @P0 BRA 0x6a8;                                          /* 0x40000003000001e7 */
        /*05e8*/                   LD.E.64 R20, [R22+0x820730];                        /* 0x8402081cc1651ca5 */
        /*05f0*/                   SSY 0x678;                                          /* 0x6000000200001c07 */
        /*05f8*/                   IADD RZ.CC, -RZ, R20;                               /* 0x4801000053ffde03 */
                                                                                       /* 0x221200420042f2c7 */
        /*0608*/                   ISETP.NE.X.AND P0, PT, R21, RZ, PT;                 /* 0x1a8e0000fd51dc63 */
        /*0610*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0618*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0620*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*0628*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0630*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0638*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
                                                                                       /* 0x22c2f2f2f2020047 */
        /*0648*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0650*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0658*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0660*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0668*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0670*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*0678*/                   IMAD R0.CC, R19, 0x4, R20;                          /* 0x2029c00011301ca3 */
                                                                                       /* 0x2202f2c2e042c047 */
        /*0688*/                   IMAD.HI.X R20, R19, 0x4, R21;                       /* 0x20aac00011351ce3 */
        /*0690*/                   IADD32I R4.CC, R0, -0xe0;                           /* 0x0ffffffc80011c02 */
        /*0698*/                   IADD32I.X R5, R20, -0x1;                            /* 0x0bfffffffd415c42 */
        /*06a0*/                   BRA 0x6b8;                                          /* 0x4000000040001de7 */
        /*06a8*/                   IADD32I R4.CC, R17, 0x20020;                        /* 0x0c00080081111c02 */
        /*06b0*/                   IADD.X R5, RZ, R16;                                 /* 0x4800000043f15c43 */
        /*06b8*/                   LD.E R3, [R4];                                      /* 0x840000000040dc85 */
                                                                                       /* 0x22e04283f042c207 */
        /*06c8*/                   IADD32I R19, R19, 0x1;                              /* 0x080000000534dc02 */
        /*06d0*/                   IADD32I R17.CC, R17, 0x40040;                       /* 0x0c00100101145c02 */
        /*06d8*/                   ISETP.NE.AND P0, PT, R19, 0x40, PT;                 /* 0x1a8ec0010131dc23 */
        /*06e0*/                   IADD.X R16, RZ, R16;                                /* 0x4800000043f41c43 */
        /*06e8*/                   IADD R3, R18, R3;                                   /* 0x480000000d20dc03 */
        /*06f0*/                   ST.E [R4], R3;                                      /* 0x940000000040dc85 */
        /*06f8*/               @P0 BRA 0x290;                                          /* 0x4003ffee400001e7 */
                                                                                       /* 0x20000000000002f7 */
        /*0708*/                   EXIT;                                               /* 0x8000000000001de7 */
        /*0710*/                   BRA 0x710;                                          /* 0x4003ffffe0001de7 */
        /*0718*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0720*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0728*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0730*/                   NOP;                                                /* 0x4000000000001de4 */
        /*0738*/                   NOP;                                                /* 0x4000000000001de4 */
		......................................................................................................................................................................


		Function : _ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0_
	.headerflags    @"EF_CUDA_SM30 EF_CUDA_PTX_SM(EF_CUDA_SM30)"
                                                                                       /* 0x22c2b20282823307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                               /* 0x2800400110005de4 */
        /*0010*/                   S2R R16, SR_TID.X;                                  /* 0x2c00000084041c04 */
        /*0018*/                   S2R R3, SR_CTAID.X;                                 /* 0x2c0000009400dc04 */
        /*0020*/                   IMAD R16, R3, c[0x0][0x28], R16;                    /* 0x20204000a0341ca3 */
        /*0028*/                   ISUB RZ.CC, R16, c[0x0][0x148];                     /* 0x48014005210fdd03 */
        /*0030*/                   SHR R20, R16, 0x1f;                                 /* 0x5800c0007d051c23 */
        /*0038*/                   ISETP.GE.U32.X.AND P0, PT, R20, c[0x0][0x14c], PT;  /* 0x1b0e40053141dc43 */
                                                                                       /* 0x22f2c2c042c042e7 */
        /*0048*/               @P0 EXIT;                                               /* 0x80000000000001e7 */
        /*0050*/                   IADD R21.CC, R16, c[0x0][0x140];                    /* 0x4801400501055c03 */
        /*0058*/                   SSY 0xe8;                                           /* 0x6000000220001c07 */
        /*0060*/                   IADD.X R20, R20, c[0x0][0x144];                     /* 0x4800400511451c43 */
        /*0068*/                   IADD32I RZ.CC, R21, -0x8008;                        /* 0x0ffffdffe15fdc02 */
        /*0070*/                   ISETP.LT.U32.X.AND P0, PT, R20, RZ, PT;             /* 0x188e0000fd41dc43 */
        /*0078*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
                                                                                       /* 0x2202004220420047 */
        /*0088*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0090*/                   MOV32I R8, 0xa1;                                    /* 0x1800000284021de2 */
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
        /*00e8*/                   IADD32I R21.CC, R21, 0x1;                           /* 0x0c00000005555c02 */
        /*00f0*/                   SSY 0x180;                                          /* 0x6000000220001c07 */
        /*00f8*/                   IADD.X R20, RZ, R20;                                /* 0x4800000053f51c43 */
                                                                                       /* 0x2220420042f2c2c7 */
        /*0108*/                   IADD RZ.CC, -RZ, R21;                               /* 0x4801000057ffde03 */
        /*0110*/                   ISETP.NE.X.AND P0, PT, R20, RZ, PT;                 /* 0x1a8e0000fd41dc63 */
        /*0118*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0120*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0128*/                   MOV32I R8, 0xd1;                                    /* 0x1800000344021de2 */
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
                                                                                       /* 0x22c2828282f2c2c7 */
        /*0188*/                   IADD RZ.CC, -RZ, R21;                               /* 0x4801000057ffde03 */
        /*0190*/                   ISETP.EQ.X.AND P0, PT, R20, RZ, PT;                 /* 0x190e0000fd41dc63 */
        /*0198*/               @P0 EXIT;                                               /* 0x80000000000001e7 */
        /*01a0*/                   MOV32I R19, 0x0;                                    /* 0x180000000004dde2 */
        /*01a8*/                   IADD32I R0, R19, 0x100010;                          /* 0x0800400041301c02 */
        /*01b0*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*01b8*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
                                                                                       /* 0x2220420042f042c7 */
        /*01c8*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*01d0*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*01d8*/               @P0 BRA 0x238;                                          /* 0x40000001600001e7 */
        /*01e0*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*01e8*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*01f0*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*01f8*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
                                                                                       /* 0x2282f2f202020047 */
        /*0208*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0210*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0218*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0220*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0228*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0230*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0238*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
                                                                                       /* 0x2202324230423047 */
        /*0248*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*0250*/                   SHL R18, R21, 0x2;                                  /* 0x6000c00009549c03 */
        /*0258*/                   SHL R17, R21, -0x1e;                                /* 0x6000ffff89545c03 */
        /*0260*/                   MOV R5, c[0x0][0x150];                              /* 0x2800400540015de4 */
        /*0268*/                   SHL R0, R20, 0x2;                                   /* 0x6000c00009401c03 */
        /*0270*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*0278*/                   SHR.U32 R3, R21, 0x1e;                              /* 0x5800c0007950dc03 */
                                                                                       /* 0x22e0428262126207 */
        /*0288*/                   IADD R26.CC, R18, R19;                              /* 0x480100004d269c03 */
        /*0290*/                   MOV32I R19, 0x0;                                    /* 0x180000000004dde2 */
        /*0298*/               @P0 IADD R17, R0, R3;                                   /* 0x480000000c044003 */
        /*02a0*/                   MOV R3, R19;                                        /* 0x280000004c00dde4 */
        /*02a8*/                   IADD.X R27, R17, R2;                                /* 0x480000000916dc43 */
        /*02b0*/                   MOV32I R2, 0x0;                                     /* 0x1800000000009de2 */
        /*02b8*/                   ST.E [R26+0x100010], R5;                            /* 0x9400400041a15c85 */
                                                                                       /* 0x2202f2c2c202c3f7 */
        /*02c8*/                   LD.E R0, [R2+0x100008];                             /* 0x8400400020201c85 */
        /*02d0*/                   IADD32I R0.CC, R0, 0x20;                            /* 0x0c00000080001c02 */
        /*02d8*/                   IADD.X R4, RZ, RZ;                                  /* 0x48000000fff11c43 */
        /*02e0*/                   IADD32I RZ.CC, R0, -0x100000;                       /* 0x0fffc000000fdc02 */
        /*02e8*/                   ISETP.LT.U32.X.AND P0, PT, R4, RZ, PT;              /* 0x188e0000fc41dc43 */
        /*02f0*/               @P0 BRA 0x358;                                          /* 0x40000001800001e7 */
        /*02f8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
                                                                                       /* 0x2202004220420047 */
        /*0308*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0310*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*0318*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0320*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0328*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0330*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0338*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x222280420042f2f7 */
        /*0348*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0350*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0358*/                   IADD32I R4.CC, R2, 0x100000;                        /* 0x0c00400000211c02 */
        /*0360*/                   S2R R5, SR_LANEID;                                  /* 0x2c00000000015c04 */
        /*0368*/                   VOTE.ALL R0, PT, PT;                                /* 0x49c0000000701c04 */
        /*0370*/                   SSY 0x4a0;                                          /* 0x60000004a0001c07 */
        /*0378*/                   FLO.U32 R6, R0;                                     /* 0x7800000000019c03 */
                                                                                       /* 0x2202804230428047 */
        /*0388*/                   POPC R3, R0, R0;                                    /* 0x540000000000dc04 */
        /*0390*/                   IADD.X R8, RZ, R19;                                 /* 0x480000004ff21c43 */
        /*0398*/                   ISETP.EQ.U32.AND P0, PT, R6, R5, PT;                /* 0x190e00001461dc03 */
        /*03a0*/                   SHL R7, R3, 0x5;                                    /* 0x6000c0001431dc03 */
        /*03a8*/                   MOV R5, R8;                                         /* 0x2800000020015de4 */
        /*03b0*/                   S2R R9, SR_LTMASK;                                  /* 0x2c000000e4025c04 */
        /*03b8*/               @P0 ATOM.E.ADD R7, [R4+0x8], R7;                        /* 0x547e38002041c005 */
                                                                                       /* 0x2202c28373f042d7 */
        /*03c8*/                   LOP.AND R0, R9, R0;                                 /* 0x6800000000901c03 */
        /*03d0*/                   POPC R0, R0, R0;                                    /* 0x5400000000001c04 */
        /*03d8*/                   CCTL.E.IV [R4+0x20];                                /* 0x9c00000080401ca5 */
        /*03e0*/                   SHFL.IDX PT, R3, R7, R6, 0x1f;                      /* 0x8c007c001870df45 */
        /*03e8*/                   IMAD R3, R0, 0x20, R3;                              /* 0x2006c0008000dca3 */
        /*03f0*/                   IADD R3.CC, R3, R2;                                 /* 0x480100000830dc03 */
        /*03f8*/                   IADD.X R23, RZ, R19;                                /* 0x480000004ff5dc43 */
                                                                                       /* 0x2202f2c2c28042c7 */
        /*0408*/                   IADD32I R22.CC, R3, 0x8;                            /* 0x0c00000020359c02 */
        /*0410*/                   LOP32I.AND R0, R22, 0x3;                            /* 0x380000000d601c02 */
        /*0418*/                   IADD.X R23, RZ, R23;                                /* 0x480000005ff5dc43 */
        /*0420*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*0428*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*0430*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0438*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
                                                                                       /* 0x2202004220420047 */
        /*0448*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0450*/                   MOV32I R8, 0x5d;                                    /* 0x1800000174021de2 */
        /*0458*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0460*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0468*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0470*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0478*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
                                                                                       /* 0x2232028042f2f2f7 */
        /*0488*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0490*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*0498*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*04a0*/                   MOV32I R0, 0x3;                                     /* 0x180000000c001de2 */
        /*04a8*/                   SHL R20, R20, 0x3;                                  /* 0x6000c0000d451c03 */
        /*04b0*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*04b8*/                   SHL R4, R21, -0x1d;                                 /* 0x6000ffff8d511c03 */
                                                                                       /* 0x2282028202b20277 */
        /*04c8*/                   SHR.U32 R0, R21, 0x1d;                              /* 0x5800c00075501c03 */
        /*04d0*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*04d8*/                   IMAD.U32.U32 R24.CC, R21, 0x8, R2;                  /* 0x2005c00021561c03 */
        /*04e0*/               @P0 IADD R4, R20, R0;                                   /* 0x4800000001410003 */
        /*04e8*/                   IADD32I R0, R2, 0x860770;                           /* 0x0802181dc0201c02 */
        /*04f0*/                   LOP32I.AND R0, R0, 0x7;                             /* 0x380000001c001c02 */
        /*04f8*/                   IADD.X R25, R4, R19;                                /* 0x480000004c465c43 */
                                                                                       /* 0x22020042f2c2c047 */
        /*0508*/                   IADD RZ.CC, -RZ, R0;                                /* 0x4801000003ffde03 */
        /*0510*/                   ST.E.64 [R24+0x820730], R22;                        /* 0x9402081cc1859ca5 */
        /*0518*/                   ISETP.EQ.X.AND P0, PT, RZ, RZ, PT;                  /* 0x190e0000fff1dc63 */
        /*0520*/               @P0 BRA 0x580;                                          /* 0x40000001600001e7 */
        /*0528*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0530*/                   MOV32I R8, 0x7b;                                    /* 0x18000001ec021de2 */
        /*0538*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*0548*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0550*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0558*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0560*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*0568*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0570*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*0578*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x2232004230428047 */
        /*0588*/                   IADD32I R3.CC, R2, 0x120030;                        /* 0x0c004800c020dc02 */
        /*0590*/                   MOV R4, c[0x0][0x154];                              /* 0x2800400550011de4 */
        /*0598*/                   IADD R16, R16, c[0x0][0x140];                       /* 0x4800400501041c03 */
        /*05a0*/                   ST.E [R26+0x860770], R4;                            /* 0x9402181dc1a11c85 */
        /*05a8*/                   IADD.X R0, RZ, R19;                                 /* 0x480000004ff01c43 */
        /*05b0*/                   MOV R19, RZ;                                        /* 0x28000000fc04dde4 */
        /*05b8*/                   IADD R18.CC, R3, R18;                               /* 0x4801000048349c03 */
                                                                                       /* 0x22c042f042304287 */
        /*05c8*/                   IMUL32I R21, R16, 0x11;                             /* 0x1000000045055ca2 */
        /*05d0*/                   IADD.X R17, R0, R17;                                /* 0x4800000044045c43 */
        /*05d8*/                   IADD32I R23, R21, 0x3;                              /* 0x080000000d55dc02 */
        /*05e0*/                   IADD32I R22, R21, 0x2;                              /* 0x0800000009559c02 */
        /*05e8*/                   IADD32I R20, R21, 0x1;                              /* 0x0800000005551c02 */
        /*05f0*/                   ISETP.LT.U32.AND P0, PT, R19, 0x38, PT;             /* 0x188ec000e131dc03 */
        /*05f8*/                   MOV R2, R18;                                        /* 0x2800000048009de4 */
                                                                                       /* 0x22c2c3f20042f047 */
        /*0608*/                   MOV R3, R17;                                        /* 0x280000004400dde4 */
        /*0610*/               @P0 BRA 0x718;                                          /* 0x40000004000001e7 */
        /*0618*/                   LD.E.64 R26, [R24+0x820730];                        /* 0x8402081cc1869ca5 */
        /*0620*/                   SHR R2, R19, 0x1f;                                  /* 0x5800c0007d309c23 */
        /*0628*/                   SSY 0x6b0;                                          /* 0x6000000200001c07 */
        /*0630*/                   IADD RZ.CC, -RZ, R26;                               /* 0x480100006bffde03 */
        /*0638*/                   ISETP.NE.X.AND P0, PT, R27, RZ, PT;                 /* 0x1a8e0000fdb1dc63 */
                                                                                       /* 0x22004220420042f7 */
        /*0648*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0650*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0658*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*0660*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0668*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0670*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*0678*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
                                                                                       /* 0x228042f2f2f20207 */
        /*0688*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*0690*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*0698*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*06a0*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*06a8*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*06b0*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
        /*06b8*/                   SHL R2, R2, 0x2;                                    /* 0x6000c00008209c03 */
                                                                                       /* 0x220282b202723207 */
        /*06c8*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*06d0*/                   SHL R3, R19, -0x1e;                                 /* 0x6000ffff8930dc03 */
        /*06d8*/                   SHR.U32 R4, R19, 0x1e;                              /* 0x5800c00079311c03 */
        /*06e0*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*06e8*/                   IMAD.U32.U32 R0.CC, R19, 0x4, R26;                  /* 0x2035c00011301c03 */
        /*06f0*/               @P0 IADD R3, R2, R4;                                    /* 0x480000001020c003 */
        /*06f8*/                   IADD.X R26, R27, R3;                                /* 0x480000000db69c43 */
                                                                                       /* 0x2202f2c28042f2c7 */
        /*0708*/                   IADD32I R2.CC, R0, -0xe0;                           /* 0x0ffffffc80009c02 */
        /*0710*/                   IADD32I.X R3, R26, -0x1;                            /* 0x0bfffffffda0dc42 */
        /*0718*/                   IADD32I R16, R19, 0x1;                              /* 0x0800000005341c02 */
        /*0720*/                   ST.E [R2], R21;                                     /* 0x9400000000255c85 */
        /*0728*/                   ISETP.LT.U32.AND P0, PT, R16, 0x38, PT;             /* 0x188ec000e101dc03 */
        /*0730*/               @P0 BRA 0x840;                                          /* 0x40000004200001e7 */
        /*0738*/                   LD.E.64 R28, [R24+0x820730];                        /* 0x8402081cc1871ca5 */
                                                                                       /* 0x220042f2c2c3f047 */
        /*0748*/                   SHR R2, R16, 0x1f;                                  /* 0x5800c0007d009c23 */
        /*0750*/                   SSY 0x7d8;                                          /* 0x6000000200001c07 */
        /*0758*/                   IADD RZ.CC, -RZ, R28;                               /* 0x4801000073ffde03 */
        /*0760*/                   ISETP.NE.X.AND P0, PT, R29, RZ, PT;                 /* 0x1a8e0000fdd1dc63 */
        /*0768*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*0770*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*0778*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
                                                                                       /* 0x22f2020200422047 */
        /*0788*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
        /*0790*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*0798*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*07a0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*07a8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*07b0*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*07b8*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
                                                                                       /* 0x227232028042f2f7 */
        /*07c8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
        /*07d0*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*07d8*/                   MOV32I R0, 0x2;                                     /* 0x1800000008001de2 */
        /*07e0*/                   SHL R3, R2, 0x2;                                    /* 0x6000c0000820dc03 */
        /*07e8*/                   IADD32I RZ.CC, R0, -0x20;                           /* 0x0fffffff800fdc02 */
        /*07f0*/                   SHL R2, R16, -0x1e;                                 /* 0x6000ffff89009c03 */
        /*07f8*/                   SHR.U32 R4, R16, 0x1e;                              /* 0x5800c00079011c03 */
                                                                                       /* 0x22e042c20282b207 */
        /*0808*/                   CSETP.SFT.AND P0, PT, CC, PT;                       /* 0x040e00000151dc04 */
        /*0810*/                   IMAD.U32.U32 R0.CC, R16, 0x4, R28;                  /* 0x2039c00011001c03 */
        /*0818*/               @P0 IADD R2, R3, R4;                                    /* 0x4800000010308003 */
        /*0820*/                   IADD.X R28, R29, R2;                                /* 0x4800000009d71c43 */
        /*0828*/                   IADD32I R2.CC, R0, -0xe0;                           /* 0x0ffffffc80009c02 */
        /*0830*/                   IADD32I.X R3, R28, -0x1;                            /* 0x0bfffffffdc0dc42 */
        /*0838*/                   BRA 0x858;                                          /* 0x4000000060001de7 */
                                                                                       /* 0x2202f2c28042f2c7 */
        /*0848*/                   IADD32I R2.CC, R18, 0x20020;                        /* 0x0c00080081209c02 */
        /*0850*/                   IADD.X R3, RZ, R17;                                 /* 0x4800000047f0dc43 */
        /*0858*/                   IADD32I R16, R16, 0x1;                              /* 0x0800000005041c02 */
        /*0860*/                   ST.E [R2], R20;                                     /* 0x9400000000251c85 */
        /*0868*/                   ISETP.LT.U32.AND P0, PT, R16, 0x38, PT;             /* 0x188ec000e101dc03 */
        /*0870*/               @P0 BRA 0x940;                                          /* 0x40000003200001e7 */
        /*0878*/                   LD.E.64 R26, [R24+0x820730];                        /* 0x8402081cc1869ca5 */
                                                                                       /* 0x22020042f2c2c3f7 */
        /*0888*/                   SSY 0x910;                                          /* 0x6000000200001c07 */
        /*0890*/                   IADD RZ.CC, -RZ, R26;                               /* 0x480100006bffde03 */
        /*0898*/                   ISETP.NE.X.AND P0, PT, R27, RZ, PT;                 /* 0x1a8e0000fdb1dc63 */
        /*08a0*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*08a8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*08b0*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*08b8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*08c8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*08d0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*08d8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*08e0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*08e8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*08f0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*08f8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x22e042c042c282f7 */
        /*0908*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*0910*/                   IADD32I R3, R19, 0x2;                               /* 0x080000000930dc02 */
        /*0918*/                   IMAD R0.CC, R3, 0x4, R26;                           /* 0x2035c00010301ca3 */
        /*0920*/                   IMAD.HI.X R3, R3, 0x4, R27;                         /* 0x20b6c0001030dce3 */
        /*0928*/                   IADD32I R2.CC, R0, -0xe0;                           /* 0x0ffffffc80009c02 */
        /*0930*/                   IADD32I.X R3, R3, -0x1;                             /* 0x0bfffffffc30dc42 */
        /*0938*/                   BRA 0x958;                                          /* 0x4000000060001de7 */
                                                                                       /* 0x2202f2c28042f2c7 */
        /*0948*/                   IADD32I R2.CC, R18, 0x40040;                        /* 0x0c00100101209c02 */
        /*0950*/                   IADD.X R3, RZ, R17;                                 /* 0x4800000047f0dc43 */
        /*0958*/                   IADD32I R16, R16, 0x1;                              /* 0x0800000005041c02 */
        /*0960*/                   ST.E [R2], R22;                                     /* 0x9400000000259c85 */
        /*0968*/                   ISETP.LT.U32.AND P0, PT, R16, 0x38, PT;             /* 0x188ec000e101dc03 */
        /*0970*/               @P0 BRA 0xa40;                                          /* 0x40000003200001e7 */
        /*0978*/                   LD.E.64 R26, [R24+0x820730];                        /* 0x8402081cc1869ca5 */
                                                                                       /* 0x22020042f2c2c3f7 */
        /*0988*/                   SSY 0xa10;                                          /* 0x6000000200001c07 */
        /*0990*/                   IADD RZ.CC, -RZ, R26;                               /* 0x480100006bffde03 */
        /*0998*/                   ISETP.NE.X.AND P0, PT, R27, RZ, PT;                 /* 0x1a8e0000fdb1dc63 */
        /*09a0*/               @P0 NOP.S;                                              /* 0x40000000000001f4 */
        /*09a8*/                   MOV32I R4, 0x0;                                     /* 0x1800000000011de2 */
        /*09b0*/                   MOV32I R8, 0x13f;                                   /* 0x18000004fc021de2 */
        /*09b8*/                   MOV32I R5, 0x0;                                     /* 0x1800000000015de2 */
                                                                                       /* 0x22f2f20200421047 */
        /*09c8*/                   MOV32I R6, 0x0;                                     /* 0x1800000000019de2 */
        /*09d0*/                   MOV32I R12, 0x1;                                    /* 0x1800000004031de2 */
        /*09d8*/                   MOV32I R7, 0x0;                                     /* 0x180000000001dde2 */
        /*09e0*/                   MOV R13, RZ;                                        /* 0x28000000fc035de4 */
        /*09e8*/                   MOV32I R10, 0x0;                                    /* 0x1800000000029de2 */
        /*09f0*/                   MOV32I R11, 0x0;                                    /* 0x180000000002dde2 */
        /*09f8*/                   JCAL 0x0;                                           /* 0x1000000000011c07 */
                                                                                       /* 0x22e042c042c282f7 */
        /*0a08*/                   NOP.S;                                              /* 0x4000000000001df4 */
        /*0a10*/                   IADD32I R19, R19, 0x3;                              /* 0x080000000d34dc02 */
        /*0a18*/                   IMAD R0.CC, R19, 0x4, R26;                          /* 0x2035c00011301ca3 */
        /*0a20*/                   IMAD.HI.X R19, R19, 0x4, R27;                       /* 0x20b6c0001134dce3 */
        /*0a28*/                   IADD32I R2.CC, R0, -0xe0;                           /* 0x0ffffffc80009c02 */
        /*0a30*/                   IADD32I.X R3, R19, -0x1;                            /* 0x0bfffffffd30dc42 */
        /*0a38*/                   BRA 0xa58;                                          /* 0x4000000060001de7 */
                                                                                       /* 0x220270420042f2c7 */
        /*0a48*/                   IADD32I R2.CC, R18, 0x60060;                        /* 0x0c00180181209c02 */
        /*0a50*/                   IADD.X R3, RZ, R17;                                 /* 0x4800000047f0dc43 */
        /*0a58*/                   IADD32I R19, R16, 0x1;                              /* 0x080000000504dc02 */
        /*0a60*/                   IADD32I R22, R22, 0x4;                              /* 0x0800000011659c02 */
        /*0a68*/                   IADD32I R18.CC, R18, 0x80080;                       /* 0x0c00200201249c02 */
        /*0a70*/                   ST.E [R2], R23;                                     /* 0x940000000025dc85 */
        /*0a78*/                   ISETP.NE.AND P0, PT, R19, 0x40, PT;                 /* 0x1a8ec0010131dc23 */
                                                                                       /* 0x2002f2e042723047 */
        /*0a88*/                   IADD32I R21, R21, 0x4;                              /* 0x0800000011555c02 */
        /*0a90*/                   IADD32I R20, R20, 0x4;                              /* 0x0800000011451c02 */
        /*0a98*/                   IADD.X R17, RZ, R17;                                /* 0x4800000047f45c43 */
        /*0aa0*/                   IADD32I R23, R23, 0x4;                              /* 0x080000001175dc02 */
        /*0aa8*/               @P0 BRA 0x5f0;                                          /* 0x4003ffed000001e7 */
        /*0ab0*/                   EXIT;                                               /* 0x8000000000001de7 */
        /*0ab8*/                   BRA 0xab8;                                          /* 0x4003ffffe0001de7 */
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
        /*0050*/                   MOV32I R8, 0xa1;                         /* 0x1800000284021de2 */
        /*0058*/                   MOV32I R5, 0x0;                          /* 0x1800000000015de2 */
        /*0060*/                   MOV32I R12, 0x1;                         /* 0x1800000004031de2 */
        /*0068*/                   MOV32I R6, 0x0;                          /* 0x1800000000019de2 */
        /*0070*/                   MOV R13, RZ;                             /* 0x28000000fc035de4 */
        /*0078*/                   MOV32I R7, 0x0;                          /* 0x180000000001dde2 */
                                                                            /* 0x22c2c042c2f2f207 */
        /*0088*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0090*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0098*/                   JCAL 0x0;                                /* 0x1000000000011c07 */
        /*00a0*/                   IADD32I R2.CC, R16, 0x1;                 /* 0x0c00000005009c02 */
        /*00a8*/                   IADD.X R17, RZ, R17;                     /* 0x4800000047f45c43 */
        /*00b0*/                   IADD RZ.CC, -RZ, R2;                     /* 0x480100000bffde03 */
        /*00b8*/                   ISETP.NE.X.AND P0, PT, R17, RZ, PT;      /* 0x1a8e0000fd11dc63 */
                                                                            /* 0x22004220420042f7 */
        /*00c8*/               @P0 BRA 0x128;                               /* 0x40000001600001e7 */
        /*00d0*/                   MOV32I R4, 0x0;                          /* 0x1800000000011de2 */
        /*00d8*/                   MOV32I R8, 0xd1;                         /* 0x1800000344021de2 */
        /*00e0*/                   MOV32I R5, 0x0;                          /* 0x1800000000015de2 */
        /*00e8*/                   MOV32I R12, 0x1;                         /* 0x1800000004031de2 */
        /*00f0*/                   MOV32I R6, 0x0;                          /* 0x1800000000019de2 */
        /*00f8*/                   MOV R13, RZ;                             /* 0x28000000fc035de4 */
                                                                            /* 0x22020042f2f20207 */
        /*0108*/                   MOV32I R7, 0x0;                          /* 0x180000000001dde2 */
        /*0110*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0118*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0120*/                   JCAL 0x0;                                /* 0x1000000000011c07 */
        /*0128*/                   MOV32I R0, 0x0;                          /* 0x1800000000001de2 */
        /*0130*/                   MOV R16, R2;                             /* 0x2800000008041de4 */
        /*0138*/                   MOV32I R3, 0x0;                          /* 0x180000000000dde2 */
                                                                            /* 0x22e0428240421047 */
        /*0148*/                   MOV32I R10, 0x0;                         /* 0x1800000000029de2 */
        /*0150*/                   MOV R6, c[0x0][0x140];                   /* 0x2800400500019de4 */
        /*0158*/                   MOV32I R11, 0x0;                         /* 0x180000000002dde2 */
        /*0160*/                   MOV R7, c[0x0][0x144];                   /* 0x280040051001dde4 */
        /*0168*/                   MOV R2, R0;                              /* 0x2800000000009de4 */
        /*0170*/                   ST.E.64 [R2], R16;                       /* 0x9400000000241ca5 */
        /*0178*/                   MOV32I R8, 0x0;                          /* 0x1800000000021de2 */
                                                                            /* 0x200002f042e3f047 */
        /*0188*/                   LD.E.64 R4, [R10];                       /* 0x8400000000a11ca5 */
        /*0190*/                   MOV32I R9, 0x0;                          /* 0x1800000000025de2 */
        /*0198*/                   ST.E.64 [R8], R4;                        /* 0x9400000000811ca5 */
        /*01a0*/                   RED.E.ADD.U64 [R10], R6;                 /* 0x1400000000a19e05 */
        /*01a8*/                   EXIT;                                    /* 0x8000000000001de7 */
        /*01b0*/                   BRA 0x1b0;                               /* 0x4003ffffe0001de7 */
        /*01b8*/                   NOP;                                     /* 0x4000000000001de4 */
		...............................................................................



Fatbin ptx code:
================
arch = sm_30
code version = [6,0]
producer = cuda
host = linux
compile_size = 64bit
compressed








.version 6.0
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
.global .align 16 .b8 __DummyClassdata_buffer[8914832];
.global .align 16 .b8 __unnamed_1[262] = {115, 116, 97, 116, 105, 99, 32, 83, 101, 108, 102, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 87, 105, 116, 104, 65, 114, 101, 110, 97, 60, 49, 48, 52, 56, 53, 55, 54, 85, 76, 62, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
.global .align 16 .b8 __unnamed_2[314] = {115, 116, 97, 116, 105, 99, 32, 115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 83, 101, 108, 102, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 95, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 87, 105, 116, 104, 65, 114, 101, 110, 97, 60, 49, 48, 52, 56, 53, 55, 54, 85, 76, 62, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
.global .align 16 .b8 __unnamed_3[295] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 84, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 70, 105, 101, 108, 100, 95, 60, 84, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 62, 58, 58, 100, 97, 116, 97, 95, 112, 116, 114, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 84, 32, 61, 32, 105, 110, 116, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 48, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 93, 0};
.global .align 16 .b8 __unnamed_4[144] = {99, 104, 97, 114, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 95, 60, 79, 119, 110, 101, 114, 84, 44, 32, 65, 114, 101, 110, 97, 83, 105, 122, 101, 62, 58, 58, 97, 108, 108, 111, 99, 97, 116, 101, 95, 105, 110, 95, 97, 114, 101, 110, 97, 95, 105, 110, 116, 101, 114, 110, 97, 108, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 79, 119, 110, 101, 114, 84, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 65, 114, 101, 110, 97, 83, 105, 122, 101, 32, 61, 32, 49, 48, 52, 56, 53, 55, 54, 85, 76, 93, 0};
.global .align 16 .b8 __unnamed_5[381] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 118, 111, 105, 100, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 73, 110, 108, 105, 110, 101, 100, 68, 121, 110, 97, 109, 105, 99, 65, 114, 114, 97, 121, 70, 105, 101, 108, 100, 95, 60, 66, 44, 32, 73, 110, 108, 105, 110, 101, 100, 83, 105, 122, 101, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 84, 62, 58, 58, 115, 101, 116, 95, 101, 120, 116, 101, 114, 110, 97, 108, 95, 112, 111, 105, 110, 116, 101, 114, 95, 105, 110, 116, 101, 114, 110, 97, 108, 40, 66, 32, 42, 41, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 66, 32, 61, 32, 105, 110, 116, 44, 32, 73, 110, 108, 105, 110, 101, 100, 83, 105, 122, 101, 32, 61, 32, 53, 54, 85, 76, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 52, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 84, 32, 61, 32, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 93, 0};
.global .align 16 .b8 __unnamed_6[395] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 66, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 73, 110, 108, 105, 110, 101, 100, 68, 121, 110, 97, 109, 105, 99, 65, 114, 114, 97, 121, 70, 105, 101, 108, 100, 95, 60, 66, 44, 32, 73, 110, 108, 105, 110, 101, 100, 83, 105, 122, 101, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 84, 62, 58, 58, 97, 114, 114, 97, 121, 95, 100, 97, 116, 97, 95, 112, 116, 114, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 66, 32, 61, 32, 105, 110, 116, 44, 32, 73, 110, 108, 105, 110, 101, 100, 83, 105, 122, 101, 32, 61, 32, 53, 54, 85, 76, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 52, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 65, 114, 114, 97, 121, 83, 105, 122, 101, 84, 32, 61, 32, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 93, 0};
.global .align 16 .b8 __unnamed_7[297] = {115, 116, 100, 58, 58, 101, 110, 97, 98, 108, 101, 95, 105, 102, 60, 60, 101, 120, 112, 114, 101, 115, 115, 105, 111, 110, 62, 44, 32, 84, 32, 42, 62, 58, 58, 116, 121, 112, 101, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 70, 105, 101, 108, 100, 95, 60, 84, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 79, 102, 102, 115, 101, 116, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 44, 32, 79, 119, 110, 101, 114, 62, 58, 58, 100, 97, 116, 97, 95, 112, 116, 114, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 41, 32, 99, 111, 110, 115, 116, 32, 91, 119, 105, 116, 104, 32, 65, 32, 61, 32, 48, 44, 32, 83, 32, 61, 32, 48, 44, 32, 76, 32, 61, 32, 49, 44, 32, 84, 32, 61, 32, 105, 110, 116, 44, 32, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 55, 53, 85, 76, 44, 32, 79, 102, 102, 115, 101, 116, 32, 61, 32, 50, 51, 54, 85, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 44, 32, 79, 119, 110, 101, 114, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 93, 0};
.global .align 16 .b8 __unnamed_8[276] = {115, 116, 97, 116, 105, 99, 32, 83, 101, 108, 102, 32, 42, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 111, 97, 76, 97, 121, 111, 117, 116, 60, 83, 101, 108, 102, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 62, 58, 58, 103, 101, 116, 95, 117, 110, 105, 110, 105, 116, 105, 97, 108, 105, 122, 101, 100, 40, 117, 110, 115, 105, 103, 110, 101, 100, 32, 108, 111, 110, 103, 41, 32, 91, 119, 105, 116, 104, 32, 83, 101, 108, 102, 32, 61, 32, 68, 117, 109, 109, 121, 67, 108, 97, 115, 115, 44, 32, 85, 115, 101, 114, 67, 97, 112, 97, 99, 105, 116, 121, 32, 61, 32, 51, 50, 55, 54, 56, 85, 76, 44, 32, 65, 100, 100, 114, 101, 115, 115, 77, 111, 100, 101, 32, 61, 32, 48, 44, 32, 83, 116, 111, 114, 97, 103, 101, 83, 116, 114, 97, 116, 101, 103, 121, 32, 61, 32, 105, 107, 114, 97, 58, 58, 115, 111, 97, 58, 58, 83, 116, 97, 116, 105, 99, 83, 116, 111, 114, 97, 103, 101, 87, 105, 116, 104, 65, 114, 101, 110, 97, 60, 49, 48, 52, 56, 53, 55, 54, 85, 76, 62, 44, 32, 76, 97, 121, 111, 117, 116, 77, 111, 100, 101, 32, 61, 32, 49, 93, 0};
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
.global .align 16 .b8 $str8[42] = {114, 101, 105, 110, 116, 101, 114, 112, 114, 101, 116, 95, 99, 97, 115, 116, 60, 117, 105, 110, 116, 112, 116, 114, 95, 116, 62, 40, 112, 116, 114, 41, 32, 37, 32, 52, 32, 61, 61, 32, 48, 0};
.global .align 16 .b8 $str9[48] = {46, 46, 47, 46, 46, 47, 46, 46, 47, 105, 107, 114, 97, 47, 115, 111, 97, 47, 105, 110, 108, 105, 110, 101, 100, 95, 100, 121, 110, 97, 109, 105, 99, 95, 97, 114, 114, 97, 121, 95, 102, 105, 101, 108, 100, 46, 104, 0};
.global .align 16 .b8 $str11[22] = {112, 95, 101, 120, 116, 101, 114, 110, 97, 108, 32, 33, 61, 32, 110, 117, 108, 108, 112, 116, 114, 0};

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
mov.u64 %rd8, __unnamed_8;
cvta.global.u64 %rd9, %rd8;
mov.u32 %r1, 161;
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
mov.u32 %r2, 209;
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
.reg .pred %p<18>;
.reg .b32 %r<46>;
.reg .b64 %rd<146>;


ld.param.u64 %rd28, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_0];
ld.param.u64 %rd29, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_1];
ld.param.u32 %r20, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_2];
ld.param.u32 %r21, [_ZN4ikra8executor4cuda16construct_kernelI10DummyClassJiiEEEvmmDpT0__param_3];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %ctaid.x;
mad.lo.s32 %r22, %r2, %r3, %r1;
cvt.s64.s32	%rd1, %r22;
setp.ge.u64	%p1, %rd1, %rd29;
@%p1 bra BB1_35;

add.s64 %rd2, %rd1, %rd28;
setp.lt.u64	%p2, %rd2, 32776;
@%p2 bra BB1_3;

mov.u64 %rd30, $str;
cvta.global.u64 %rd31, %rd30;
mov.u64 %rd32, $str1;
cvta.global.u64 %rd33, %rd32;
mov.u64 %rd34, __unnamed_8;
cvta.global.u64 %rd35, %rd34;
mov.u32 %r23, 161;
mov.u64 %rd36, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd31;
.param .b64 param1;
st.param.b64	[param1+0], %rd33;
.param .b32 param2;
st.param.b32	[param2+0], %r23;
.param .b64 param3;
st.param.b64	[param3+0], %rd35;
.param .b64 param4;
st.param.b64	[param4+0], %rd36;
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

mov.u64 %rd37, $str2;
cvta.global.u64 %rd38, %rd37;
mov.u64 %rd39, $str1;
cvta.global.u64 %rd40, %rd39;
mov.u64 %rd41, __unnamed_2;
cvta.global.u64 %rd42, %rd41;
mov.u32 %r24, 209;
mov.u64 %rd43, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd38;
.param .b64 param1;
st.param.b64	[param1+0], %rd40;
.param .b32 param2;
st.param.b32	[param2+0], %r24;
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

BB1_5:
setp.eq.s64	%p4, %rd3, 0;
@%p4 bra BB1_35;

mov.u64 %rd44, __DummyClassdata_buffer;
add.s64 %rd45, %rd44, 1048592;
cvta.global.u64 %rd46, %rd45;
and.b64 %rd47, %rd46, 7;
setp.eq.s64	%p5, %rd47, 0;
@%p5 bra BB1_8;

mov.u64 %rd48, $str6;
cvta.global.u64 %rd49, %rd48;
mov.u64 %rd50, $str7;
cvta.global.u64 %rd51, %rd50;
mov.u64 %rd52, __unnamed_3;
cvta.global.u64 %rd53, %rd52;
mov.u32 %r25, 123;
mov.u64 %rd54, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd49;
.param .b64 param1;
st.param.b64	[param1+0], %rd51;
.param .b32 param2;
st.param.b32	[param2+0], %r25;
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

BB1_8:
shl.b64 %rd55, %rd3, 2;
add.s64 %rd57, %rd44, %rd55;
st.global.u32 [%rd57+1048592], %r20;
ld.global.u32 %rd58, [__DummyClassdata_buffer+1048584];
add.s64 %rd59, %rd58, 32;
setp.lt.u64	%p6, %rd59, 1048576;
@%p6 bra BB1_10;

mov.u64 %rd60, $str3;
cvta.global.u64 %rd61, %rd60;
mov.u64 %rd62, $str4;
cvta.global.u64 %rd63, %rd62;
mov.u64 %rd64, __unnamed_4;
cvta.global.u64 %rd65, %rd64;
mov.u32 %r26, 319;
mov.u64 %rd66, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd61;
.param .b64 param1;
st.param.b64	[param1+0], %rd63;
.param .b32 param2;
st.param.b32	[param2+0], %r26;
.param .b64 param3;
st.param.b64	[param3+0], %rd65;
.param .b64 param4;
st.param.b64	[param4+0], %rd66;
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
add.s64 %rd68, %rd44, 1048584;
atom.global.add.u32 %r27, [%rd68], 32;
cvt.u64.u32	%rd69, %r27;
cvta.global.u64 %rd70, %rd44;
add.s64 %rd71, %rd69, %rd70;
add.s64 %rd4, %rd71, 8;
and.b64 %rd72, %rd4, 3;
setp.eq.s64	%p7, %rd72, 0;
@%p7 bra BB1_12;

mov.u64 %rd73, $str8;
cvta.global.u64 %rd74, %rd73;
mov.u64 %rd75, $str9;
cvta.global.u64 %rd76, %rd75;
mov.u64 %rd77, __unnamed_5;
cvta.global.u64 %rd78, %rd77;
mov.u32 %r28, 93;
mov.u64 %rd79, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd74;
.param .b64 param1;
st.param.b64	[param1+0], %rd76;
.param .b32 param2;
st.param.b32	[param2+0], %r28;
.param .b64 param3;
st.param.b64	[param3+0], %rd78;
.param .b64 param4;
st.param.b64	[param4+0], %rd79;
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

BB1_12:
shl.b64 %rd80, %rd3, 3;
add.s64 %rd82, %rd44, %rd80;
add.s64 %rd6, %rd82, 8521520;
st.global.u64 [%rd82+8521520], %rd4;
add.s64 %rd83, %rd44, 8783728;
cvta.global.u64 %rd84, %rd83;
and.b64 %rd85, %rd84, 7;
setp.eq.s64	%p8, %rd85, 0;
@%p8 bra BB1_14;

mov.u64 %rd86, $str6;
cvta.global.u64 %rd87, %rd86;
mov.u64 %rd88, $str7;
cvta.global.u64 %rd89, %rd88;
mov.u64 %rd90, __unnamed_7;
cvta.global.u64 %rd91, %rd90;
mov.u32 %r29, 123;
mov.u64 %rd92, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd87;
.param .b64 param1;
st.param.b64	[param1+0], %rd89;
.param .b32 param2;
st.param.b32	[param2+0], %r29;
.param .b64 param3;
st.param.b64	[param3+0], %rd91;
.param .b64 param4;
st.param.b64	[param4+0], %rd92;
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

BB1_14:
st.global.u32 [%rd57+8783728], %r21;
add.s64 %rd97, %rd44, 1179696;
cvta.global.u64 %rd98, %rd97;
add.s64 %rd141, %rd98, %rd55;
add.s64 %rd100, %rd28, %rd1;
mul.lo.s64 %rd101, %rd100, 17;
add.s64 %rd102, %rd101, 3;
cvt.u32.u64	%r44, %rd102;
add.s64 %rd103, %rd101, 2;
cvt.u32.u64	%r43, %rd103;
cvt.u32.u64	%r42, %rd101;
add.s64 %rd104, %rd101, 1;
cvt.u32.u64	%r41, %rd104;
mov.u32 %r45, 0;

BB1_15:
cvt.s64.s32	%rd10, %r45;
setp.lt.u32	%p9, %r45, 56;
mov.u64 %rd142, %rd141;
@%p9 bra BB1_19;

ld.global.u64 %rd11, [%rd6];
setp.ne.s64	%p10, %rd11, 0;
@%p10 bra BB1_18;

mov.u64 %rd105, $str11;
cvta.global.u64 %rd106, %rd105;
mov.u64 %rd107, $str9;
cvta.global.u64 %rd108, %rd107;
mov.u64 %rd109, __unnamed_6;
cvta.global.u64 %rd110, %rd109;
mov.u32 %r32, 319;
mov.u64 %rd111, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd106;
.param .b64 param1;
st.param.b64	[param1+0], %rd108;
.param .b32 param2;
st.param.b32	[param2+0], %r32;
.param .b64 param3;
st.param.b64	[param3+0], %rd110;
.param .b64 param4;
st.param.b64	[param4+0], %rd111;
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

BB1_18:
shl.b64 %rd112, %rd10, 2;
add.s64 %rd113, %rd112, %rd11;
add.s64 %rd142, %rd113, -224;

BB1_19:
st.u32 [%rd142], %r42;
add.s32 %r33, %r45, 1;
cvt.s64.s32	%rd14, %r33;
setp.lt.u32	%p11, %r33, 56;
@%p11 bra BB1_23;
bra.uni BB1_20;

BB1_23:
add.s64 %rd143, %rd141, 131104;
bra.uni BB1_24;

BB1_20:
ld.global.u64 %rd15, [%rd6];
setp.ne.s64	%p12, %rd15, 0;
@%p12 bra BB1_22;

mov.u64 %rd114, $str11;
cvta.global.u64 %rd115, %rd114;
mov.u64 %rd116, $str9;
cvta.global.u64 %rd117, %rd116;
mov.u64 %rd118, __unnamed_6;
cvta.global.u64 %rd119, %rd118;
mov.u32 %r34, 319;
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

BB1_22:
shl.b64 %rd121, %rd14, 2;
add.s64 %rd122, %rd121, %rd15;
add.s64 %rd143, %rd122, -224;

BB1_24:
st.u32 [%rd143], %r41;
cvt.u32.u64	%r35, %rd14;
add.s32 %r13, %r35, 1;
setp.lt.u32	%p13, %r13, 56;
@%p13 bra BB1_28;
bra.uni BB1_25;

BB1_28:
add.s64 %rd144, %rd141, 262208;
bra.uni BB1_29;

BB1_25:
ld.global.u64 %rd19, [%rd6];
setp.ne.s64	%p14, %rd19, 0;
@%p14 bra BB1_27;

mov.u64 %rd123, $str11;
cvta.global.u64 %rd124, %rd123;
mov.u64 %rd125, $str9;
cvta.global.u64 %rd126, %rd125;
mov.u64 %rd127, __unnamed_6;
cvta.global.u64 %rd128, %rd127;
mov.u32 %r36, 319;
mov.u64 %rd129, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd124;
.param .b64 param1;
st.param.b64	[param1+0], %rd126;
.param .b32 param2;
st.param.b32	[param2+0], %r36;
.param .b64 param3;
st.param.b64	[param3+0], %rd128;
.param .b64 param4;
st.param.b64	[param4+0], %rd129;
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

BB1_27:
add.s32 %r37, %r45, 2;
mul.wide.s32 %rd130, %r37, 4;
add.s64 %rd131, %rd130, %rd19;
add.s64 %rd144, %rd131, -224;

BB1_29:
st.u32 [%rd144], %r43;
add.s32 %r14, %r13, 1;
setp.lt.u32	%p15, %r14, 56;
@%p15 bra BB1_33;
bra.uni BB1_30;

BB1_33:
add.s64 %rd145, %rd141, 393312;
bra.uni BB1_34;

BB1_30:
ld.global.u64 %rd23, [%rd6];
setp.ne.s64	%p16, %rd23, 0;
@%p16 bra BB1_32;

mov.u64 %rd132, $str11;
cvta.global.u64 %rd133, %rd132;
mov.u64 %rd134, $str9;
cvta.global.u64 %rd135, %rd134;
mov.u64 %rd136, __unnamed_6;
cvta.global.u64 %rd137, %rd136;
mov.u32 %r38, 319;
mov.u64 %rd138, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd133;
.param .b64 param1;
st.param.b64	[param1+0], %rd135;
.param .b32 param2;
st.param.b32	[param2+0], %r38;
.param .b64 param3;
st.param.b64	[param3+0], %rd137;
.param .b64 param4;
st.param.b64	[param4+0], %rd138;
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

BB1_32:
cvt.u32.u64	%r39, %rd10;
add.s32 %r40, %r39, 3;
mul.wide.s32 %rd139, %r40, 4;
add.s64 %rd140, %rd139, %rd23;
add.s64 %rd145, %rd140, -224;

BB1_34:
st.u32 [%rd145], %r44;
add.s64 %rd141, %rd141, 524416;
add.s32 %r44, %r44, 4;
add.s32 %r43, %r43, 4;
add.s32 %r42, %r42, 4;
add.s32 %r41, %r41, 4;
add.s32 %r45, %r14, 1;
setp.ne.s32	%p17, %r45, 64;
@%p17 bra BB1_15;

BB1_35:
ret;
}


.visible .entry _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0_(
.param .align 1 .b8 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_0[1],
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_1,
.param .u64 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_2,
.param .u32 _ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_3
)
{
.reg .pred %p<13>;
.reg .b32 %r<32>;
.reg .b64 %rd<108>;


ld.param.u64 %rd17, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_1];
ld.param.u64 %rd18, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_2];
ld.param.u32 %r9, [_ZN4ikra8executor4cuda18kernel_call_lambdaIZNS1_18ExecuteKernelProxyIM10DummyClassFviEXadL_ZNS4_13update_field1EiEEE4callEPS4_miEUlS8_miE_JS8_miEEEvT_DpT0__param_3];
mov.u32 %r10, %tid.x;
mov.u32 %r11, %ntid.x;
mov.u32 %r12, %ctaid.x;
mad.lo.s32 %r13, %r11, %r12, %r10;
cvt.s64.s32	%rd1, %r13;
setp.ge.u64	%p1, %rd1, %rd18;
@%p1 bra BB2_24;

add.s64 %rd19, %rd17, %rd1;
add.s64 %rd20, %rd19, -1;
ld.global.u64 %rd21, [__DummyClassdata_buffer];
setp.gt.u64	%p2, %rd21, %rd20;
@%p2 bra BB2_3;

mov.u64 %rd22, $str5;
cvta.global.u64 %rd23, %rd22;
mov.u64 %rd24, $str1;
cvta.global.u64 %rd25, %rd24;
mov.u64 %rd26, __unnamed_1;
cvta.global.u64 %rd27, %rd26;
mov.u32 %r14, 172;
mov.u64 %rd28, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd23;
.param .b64 param1;
st.param.b64	[param1+0], %rd25;
.param .b32 param2;
st.param.b32	[param2+0], %r14;
.param .b64 param3;
st.param.b64	[param3+0], %rd27;
.param .b64 param4;
st.param.b64	[param4+0], %rd28;
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
setp.ne.s64	%p3, %rd19, 0;
@%p3 bra BB2_5;

mov.u64 %rd29, $str2;
cvta.global.u64 %rd30, %rd29;
mov.u64 %rd31, $str1;
cvta.global.u64 %rd32, %rd31;
mov.u64 %rd33, __unnamed_2;
cvta.global.u64 %rd34, %rd33;
mov.u32 %r15, 209;
mov.u64 %rd35, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd30;
.param .b64 param1;
st.param.b64	[param1+0], %rd32;
.param .b32 param2;
st.param.b32	[param2+0], %r15;
.param .b64 param3;
st.param.b64	[param3+0], %rd34;
.param .b64 param4;
st.param.b64	[param4+0], %rd35;
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
mov.u64 %rd36, __DummyClassdata_buffer;
shl.b64 %rd37, %rd19, 2;
add.s64 %rd38, %rd36, %rd37;
add.s64 %rd3, %rd38, 1048592;
add.s64 %rd4, %rd38, 8783728;
shl.b64 %rd39, %rd19, 3;
add.s64 %rd40, %rd36, %rd39;
add.s64 %rd5, %rd40, 8521520;
add.s64 %rd41, %rd36, 1179696;
cvta.global.u64 %rd42, %rd41;
add.s64 %rd105, %rd42, %rd37;
mov.u32 %r31, 0;

BB2_6:
add.s64 %rd44, %rd36, 1048592;
cvta.global.u64 %rd45, %rd44;
and.b64 %rd46, %rd45, 7;
setp.eq.s64	%p4, %rd46, 0;
@%p4 bra BB2_8;

mov.u64 %rd47, $str6;
cvta.global.u64 %rd48, %rd47;
mov.u64 %rd49, $str7;
cvta.global.u64 %rd50, %rd49;
mov.u64 %rd51, __unnamed_3;
cvta.global.u64 %rd52, %rd51;
mov.u32 %r17, 123;
mov.u64 %rd53, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd48;
.param .b64 param1;
st.param.b64	[param1+0], %rd50;
.param .b32 param2;
st.param.b32	[param2+0], %r17;
.param .b64 param3;
st.param.b64	[param3+0], %rd52;
.param .b64 param4;
st.param.b64	[param4+0], %rd53;
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
add.s64 %rd55, %rd36, 8783728;
cvta.global.u64 %rd56, %rd55;
and.b64 %rd57, %rd56, 7;
setp.eq.s64	%p5, %rd57, 0;
@%p5 bra BB2_10;

mov.u64 %rd58, $str6;
cvta.global.u64 %rd59, %rd58;
mov.u64 %rd60, $str7;
cvta.global.u64 %rd61, %rd60;
mov.u64 %rd62, __unnamed_7;
cvta.global.u64 %rd63, %rd62;
mov.u32 %r18, 123;
mov.u64 %rd64, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd59;
.param .b64 param1;
st.param.b64	[param1+0], %rd61;
.param .b32 param2;
st.param.b32	[param2+0], %r18;
.param .b64 param3;
st.param.b64	[param3+0], %rd63;
.param .b64 param4;
st.param.b64	[param4+0], %rd64;
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
add.s32 %r19, %r2, %r9;
ld.global.u32 %r20, [%rd4];
add.s32 %r3, %r19, %r20;
setp.lt.u32	%p6, %r31, 56;
mov.u64 %rd106, %rd105;
@%p6 bra BB2_14;

ld.global.u64 %rd9, [%rd5];
setp.ne.s64	%p7, %rd9, 0;
@%p7 bra BB2_13;

mov.u64 %rd65, $str11;
cvta.global.u64 %rd66, %rd65;
mov.u64 %rd67, $str9;
cvta.global.u64 %rd68, %rd67;
mov.u64 %rd69, __unnamed_6;
cvta.global.u64 %rd70, %rd69;
mov.u32 %r21, 319;
mov.u64 %rd71, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd66;
.param .b64 param1;
st.param.b64	[param1+0], %rd68;
.param .b32 param2;
st.param.b32	[param2+0], %r21;
.param .b64 param3;
st.param.b64	[param3+0], %rd70;
.param .b64 param4;
st.param.b64	[param4+0], %rd71;
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

BB2_13:
mul.wide.s32 %rd72, %r31, 4;
add.s64 %rd73, %rd72, %rd9;
add.s64 %rd106, %rd73, -224;

BB2_14:
ld.u32 %r22, [%rd106];
add.s32 %r23, %r3, %r22;
st.u32 [%rd106], %r23;
@%p4 bra BB2_16;

mov.u64 %rd78, $str6;
cvta.global.u64 %rd79, %rd78;
mov.u64 %rd80, $str7;
cvta.global.u64 %rd81, %rd80;
mov.u64 %rd82, __unnamed_3;
cvta.global.u64 %rd83, %rd82;
mov.u32 %r24, 123;
mov.u64 %rd84, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd79;
.param .b64 param1;
st.param.b64	[param1+0], %rd81;
.param .b32 param2;
st.param.b32	[param2+0], %r24;
.param .b64 param3;
st.param.b64	[param3+0], %rd83;
.param .b64 param4;
st.param.b64	[param4+0], %rd84;
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
ld.global.u32 %r4, [%rd3];
@%p5 bra BB2_18;

mov.u64 %rd89, $str6;
cvta.global.u64 %rd90, %rd89;
mov.u64 %rd91, $str7;
cvta.global.u64 %rd92, %rd91;
mov.u64 %rd93, __unnamed_7;
cvta.global.u64 %rd94, %rd93;
mov.u32 %r25, 123;
mov.u64 %rd95, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd90;
.param .b64 param1;
st.param.b64	[param1+0], %rd92;
.param .b32 param2;
st.param.b32	[param2+0], %r25;
.param .b64 param3;
st.param.b64	[param3+0], %rd94;
.param .b64 param4;
st.param.b64	[param4+0], %rd95;
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
add.s32 %r7, %r31, 1;
add.s32 %r26, %r4, %r9;
ld.global.u32 %r27, [%rd4];
add.s32 %r6, %r26, %r27;
setp.lt.u32	%p10, %r7, 56;
@%p10 bra BB2_22;
bra.uni BB2_19;

BB2_22:
add.s64 %rd107, %rd105, 131104;
bra.uni BB2_23;

BB2_19:
ld.global.u64 %rd12, [%rd5];
setp.ne.s64	%p11, %rd12, 0;
@%p11 bra BB2_21;

mov.u64 %rd96, $str11;
cvta.global.u64 %rd97, %rd96;
mov.u64 %rd98, $str9;
cvta.global.u64 %rd99, %rd98;
mov.u64 %rd100, __unnamed_6;
cvta.global.u64 %rd101, %rd100;
mov.u32 %r28, 319;
mov.u64 %rd102, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd97;
.param .b64 param1;
st.param.b64	[param1+0], %rd99;
.param .b32 param2;
st.param.b32	[param2+0], %r28;
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

BB2_21:
mul.wide.s32 %rd103, %r7, 4;
add.s64 %rd104, %rd103, %rd12;
add.s64 %rd107, %rd104, -224;

BB2_23:
ld.u32 %r29, [%rd107];
add.s32 %r30, %r6, %r29;
st.u32 [%rd107], %r30;
add.s64 %rd105, %rd105, 262208;
add.s32 %r31, %r7, 1;
setp.ne.s32	%p12, %r31, 64;
@%p12 bra BB2_6;

BB2_24:
ret;
}


.visible .entry _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m(
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_1
)
{
.reg .pred %p<2>;
.reg .b32 %r<4>;
.reg .b64 %rd<17>;


ld.param.u64 %rd2, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_0];
ld.param.u64 %rd3, [_ZN4ikra3soa24allocate_in_arena_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_1];
cvta.to.global.u64 %rd4, %rd2;
add.s64 %rd1, %rd4, 1048584;
ld.global.u32 %rd5, [%rd4+1048584];
add.s64 %rd6, %rd5, %rd3;
setp.lt.u64	%p1, %rd6, 1048576;
@%p1 bra BB3_2;

mov.u64 %rd7, $str3;
cvta.global.u64 %rd8, %rd7;
mov.u64 %rd9, $str4;
cvta.global.u64 %rd10, %rd9;
mov.u64 %rd11, __unnamed_4;
cvta.global.u64 %rd12, %rd11;
mov.u32 %r1, 319;
mov.u64 %rd13, 1;

	{
.reg .b32 temp_param_reg;

	.param .b64 param0;
st.param.b64	[param0+0], %rd8;
.param .b64 param1;
st.param.b64	[param1+0], %rd10;
.param .b32 param2;
st.param.b32	[param2+0], %r1;
.param .b64 param3;
st.param.b64	[param3+0], %rd12;
.param .b64 param4;
st.param.b64	[param4+0], %rd13;
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

BB3_2:
cvt.u32.u64	%r2, %rd3;
atom.global.add.u32 %r3, [%rd1], %r2;
cvt.u64.u32	%rd14, %r3;
add.s64 %rd15, %rd2, %rd14;
add.s64 %rd16, %rd15, 8;
st.global.u64 [_ZN4ikra3soa18d_arena_allocationE], %rd16;
ret;
}


.visible .entry _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m(
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_0,
.param .u64 _ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_1
)
{
.reg .b64 %rd<5>;


ld.param.u64 %rd1, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_0];
ld.param.u64 %rd2, [_ZN4ikra3soa28storage_increase_size_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEEEvPT_m_param_1];
cvta.to.global.u64 %rd3, %rd1;
atom.global.add.u64 %rd4, [%rd3], %rd2;
st.global.u64 [_ZN4ikra3soa23d_previous_storage_sizeE], %rd4;
ret;
}


.visible .entry _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEJEEEvPT_DpT0_(
.param .u64 _ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEJEEEvPT_DpT0__param_0
)
{
.reg .pred %p<2>;
.reg .b32 %r<2>;
.reg .b64 %rd<4>;


ld.param.u64 %rd1, [_ZN4ikra3soa30storage_cuda_initialize_kernelINS0_14StaticStorage_I10DummyClassLm1048576EEEJEEEvPT_DpT0__param_0];
setp.eq.s64	%p1, %rd1, 0;
@%p1 bra BB5_2;

cvta.to.global.u64 %rd2, %rd1;
mov.u64 %rd3, 0;
st.global.u64 [%rd2], %rd3;
mov.u32 %r1, 0;
st.global.u32 [%rd2+1048584], %r1;

BB5_2:
ret;
}


